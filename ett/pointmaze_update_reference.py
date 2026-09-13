"""Fixed-budget three-arm ETT updates: diagonal, averaged NCE, and MC."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import torch

from ett import pointmaze_future_average as average
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha

PREVIOUS=Path('artifacts/pointmaze_region_pilot/future_average_s01_v1')
CONFIG=dict(base='f845b234cc4d7a328d5326e547e07e40880f4b99',
    arms=['diagonal', 'critic', 'mc'], seeds=[0, 1], updates=3,
    directions=4, query_count=32, successors=4, actor_draws=4, paths_per_root=8,
    refresh_steps=400, critic_batch=256, critic_lr=.003, lambda_off=1.,
    sigma_diagonal=.01, sigma_response=.1, rate_diagonal=.01, rate_response=.5,
    cap_diagonal=.03, cap_response=.1, diagonal_tolerance=.02,
    diagonal_rows=128, diagonal_draws=8, guard_rows=256, guard_draws=16,
    validation_rows=512, validation_draws=32, evaluation_repeats=64,
    gamma=.95, bound=1., geometry='rectangle', tolerance=2e-6,
    guard_seed=122000000, update_seed=123000000, evaluation_seed=124000000,
    validation_seed=125000000, bootstrap_seed=126000000, bootstrap_repeats=2000,
    model_output_cap=3000000, native_steps=0, actor_updates=0)


class Ledger(old.Ledger):
    def __init__(self, out):
        super().__init__(out); self.data['cap']=CONFIG['model_output_cap']; self.scope='shared'
    def add(self, number, purpose):
        super().add(number, self.scope+'/'+purpose)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def visitation(records, seed, count=32):
    rng=np.random.default_rng(seed)
    paths=[(r, i) for r in records for i in range(len(r['root']))]
    selected=rng.integers(len(paths), size=count)
    states=[]; actions=[]; lengths=[]; times=[]; roots=[]
    for path in selected:
        r, i=paths[path]; h=r['action'].shape[1]; t=int(rng.integers(h))
        states.append(r['states'][i, t]); actions.append(r['action'][i, t])
        lengths.append(h); times.append(t); roots.append(r['root'][i])
    lengths=np.array(lengths); times=np.array(times)
    return dict(state=np.asarray(states), action=np.asarray(actions), length=lengths,
        t=times, h=lengths-times, weight=lengths*.95**times, path=selected, root=np.array(roots))


def refresh(critic, optimizer, records, context, seed, steps=400):
    # The unused sampled-label output cannot affect this integrated-positive loss.
    x, _=early.positives(records, seed+1, context['mean'], context['std'])
    probability=average.future_probabilities(records)
    x=torch.from_numpy(x); probability=torch.from_numpy(probability)
    rng=np.random.default_rng(seed); stream=hashlib.sha256()
    for param in critic.parameters(): param.requires_grad_(True)
    for _ in range(steps):
        idx=rng.integers(len(x), size=256); stream.update(idx.astype('<i8').tobytes())
        loss=average.nce(critic(x[idx]), None, probability[idx], 'averaged')
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    for param in critic.parameters(): param.requires_grad_(False)
    return dict(loss=float(loss.detach()), steps=steps, rows=len(x),
        uniform_row_stream_sha256=stream.hexdigest(), probabilities_sha256=array_sha(probability))


def mc_continuation(engine, frozen_theta, states, actions, h, key, ledger):
    if h==0: return np.zeros(len(states)), np.empty((len(states), 0), np.float32)
    ledger.add(len(states)*h, 'mc_preupdate_continuations')
    r=engine.rollout(frozen_theta, states, key, h, actions)
    np.testing.assert_array_equal(r['action'][:, 0], actions)
    return .05*r['reward']@.95**np.arange(h), r['reward']


def surrogate(engine, candidate, preupdate, queries, critic, context, seed, ledger):
    n=len(queries['h']); draws=CONFIG['successors']; actors=CONFIG['actor_draws']
    s=np.repeat(queries['state'], draws, 0); a=np.repeat(queries['action'], draws, 0)
    xp=engine.nominal.sample(s, jax.random.PRNGKey(seed), 1, goal=np.broadcast_to(old.GOAL, s.shape))
    ledger.add(len(s), 'signed_first_transitions')
    y, detail=engine.sample(candidate, s, a, xp, jax.random.PRNGKey(seed+1), 1)
    old.validate_selected_set(s, y, detail); ns=np.asarray(y[:, 0])
    repeated=np.repeat(ns, actors, 0)
    next_action=np.asarray(engine.actor_jit(repeated, jax.random.PRNGKey(seed+2)))
    remaining=np.repeat(queries['h']-1, draws*actors)
    reward=np.asarray(old.task_reward(ns, np.broadcast_to(old.GOAL, ns.shape)))
    raw_rewards=None
    if critic is not None:
        q=critic.predict(repeated, next_action, remaining, context['mean'], context['std'])
    else:
        q=np.empty(len(repeated)); raw_rewards=np.full((len(repeated), 48), np.nan, np.float32)
        for h in np.unique(remaining):
            ids=np.flatnonzero(remaining==h)
            values, rewards=mc_continuation(engine, preupdate, repeated[ids], next_action[ids], int(h),
                jax.random.PRNGKey(seed+100+int(h)), ledger)
            q[ids]=values; raw_rewards[ids, :h]=rewards
    integrand=.05*reward.reshape(n, draws, 1)+.95*q.reshape(n, draws, actors)
    value=float(np.mean(queries['weight']*integrand.mean((1, 2))))
    raw=dict(integrand=integrand, q=q.reshape(n, draws, actors),
        successor=ns.reshape(n, draws, 8), next_action=next_action.reshape(n, draws, actors, 2),
        x_prime=np.asarray(xp).reshape(n, draws, 2), reward=reward.reshape(n, draws),
        preupdate_theta=preupdate, candidate_theta=candidate)
    if raw_rewards is not None: raw['continuation_rewards']=raw_rewards.reshape(n, draws, actors, 48)
    return value, raw


def propose(theta, directions, signed, pessimistic):
    sigma=np.r_[np.full(16, .01), np.full(32, .1)]
    components=np.einsum('dt,di->ti', signed[:, 0]-signed[:, 1], directions)/(2*sigma*len(directions))
    components[0, 16:]=0.
    gradient=components[0]+(components[1] if pessimistic else 0.)
    raw_step=gradient*np.r_[np.full(16, .01), np.full(32, .5)]
    step=raw_step.copy()
    for sl, cap in [(slice(0, 16), .03), (slice(16, 48), .1)]:
        step[sl]*=min(1., cap/max(np.linalg.norm(step[sl]), 1e-12))
    return (theta-step).astype(np.float32), dict(diagonal_gradient=components[0],
        surrogate_gradient=components[1], raw_step=raw_step, clipped_step=step,
        diagonal_step_clipped=bool(np.linalg.norm(raw_step[:16])>.03),
        response_step_clipped=bool(np.linalg.norm(raw_step[16:])>.1))


def guard_accept(candidate_score, initial_score):
    return bool(candidate_score<=initial_score+CONFIG['diagonal_tolerance'])


def prepare(out):
    if out.exists(): raise ValueError('Fresh output directory required.')
    average.verify(PREVIOUS)
    inherited=json.loads((PREVIOUS/'preregistration.json').read_text())['sources']
    sources=['ett/pointmaze_update_reference.py', 'ett/eval_pointmaze_update_reference.py',
        'ett/check_pointmaze_update_reference.py', 'scripts/test_pointmaze_update_reference.py',
        'notes/pointmaze_update_reference.md']
    inputs=[PREVIOUS/f'checkpoints/averaged_s{s}.pt' for s in CONFIG['seeds']]
    inputs+=[phase.PREVIOUS/'contexts.npz', average.PREVIOUS/'final_contexts.npz']
    hashes={**inherited, **{str(f):sha(f) for f in sources+inputs}}
    out.mkdir(parents=True); (out/'checkpoints').mkdir(); (out/'updates').mkdir()
    write(out/'config.json', CONFIG)
    (out/'PROTOCOL.md').write_bytes(Path(sources[-1]).read_bytes())
    write(out/'preregistration.json', dict(utc=average.timestamp(), sources=hashes,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=sha(out/'config.json'), protocol_sha256=sha(out/'PROTOCOL.md')))
    print('Three-arm protocol, inputs and source hashes sealed.', flush=True)


def verify(out):
    declaration=json.loads((out/'preregistration.json').read_text())
    assert json.loads((out/'config.json').read_text())==CONFIG
    assert sha(out/'config.json')==declaration['config_sha256']
    assert sha(out/'PROTOCOL.md')==declaration['protocol_sha256']
    for path, digest in declaration['sources'].items(): assert sha(path)==digest, path


def train(engine, out, context, ledger):
    guard_rows=np.arange(256); theta0=np.zeros(48, np.float32)
    initial_es=float(old.diagonal(engine, theta0, context, 'train', guard_rows, 16,
        CONFIG['guard_seed'], ledger, 'initial_guard').mean())
    final={'initial':theta0}; histories={}; first_queries={}
    for seed in CONFIG['seeds']:
        for arm in CONFIG['arms']:
            name=f'{arm}_s{seed}'; ledger.scope=name; theta=theta0.copy(); history=[]
            critic=None; optimizer=None
            if arm=='critic':
                saved=torch.load(PREVIOUS/'checkpoints'/f'averaged_s{seed}.pt', weights_only=True)
                critic=early.Critic(0); critic.load_state_dict(saved['model'])
                optimizer=torch.optim.Adam(critic.parameters(), lr=.003); optimizer.load_state_dict(saved['optimizer'])
            for iteration in range(CONFIG['updates']):
                step_dir=out/'updates'/f'{name}_u{iteration}'; step_dir.mkdir()
                key=CONFIG['update_seed']+seed*100000+iteration*1000
                preupdate=theta.copy(); np.savez_compressed(step_dir/'preupdate.npz', theta=preupdate)
                records=early.collect(engine, preupdate, context['train_roots'], context['train_indices'][:, 1],
                    key, ledger, 'current_visitation_paths', 8)
                phase.save_records(step_dir/'paths.npz', records)
                queries=visitation(records, key+102); np.savez_compressed(step_dir/'queries.npz', **queries)
                if iteration==0:
                    if seed not in first_queries: first_queries[seed]=queries
                    for k, value in queries.items(): np.testing.assert_array_equal(value, first_queries[seed][k])
                fit=None; critic_hash=None; checkpoint_hash=None
                if critic is not None:
                    fit=refresh(critic, optimizer, records, context, key+101)
                    critic_hash=old.parameter_sha(critic)
                    critic_path=out/'checkpoints'/f'{name}_u{iteration}.pt'
                    torch.save(dict(model=critic.state_dict(), optimizer=optimizer.state_dict()), critic_path)
                    checkpoint_hash=sha(critic_path)
                    optimizer_before=copy.deepcopy(optimizer.state_dict())
                rng=np.random.default_rng(key+103)
                directions=rng.normal(size=(4, 48)); indices=rng.integers(512, size=128)
                sigma=np.r_[np.full(16, .01), np.full(32, .1)]; signed=np.zeros((4, 2, 2))
                queries_hash=sha(step_dir/'queries.npz')
                for d, direction in enumerate(directions):
                    for sign_index, sign in enumerate([1, -1]):
                        candidate=(preupdate+sign*sigma*direction).astype(np.float32)
                        signed[d, sign_index, 0]=old.diagonal(engine, candidate, context, 'train', indices, 8,
                            key+300, ledger, 'signed_diagonal').mean()
                        if arm!='diagonal':
                            value, raw=surrogate(engine, candidate, preupdate, queries, critic, context, key+400, ledger)
                            signed[d, sign_index, 1]=value
                            np.savez_compressed(step_dir/f'd{d}_sign{sign_index}.npz', **raw)
                proposed, info=propose(preupdate, directions, signed, arm!='diagonal')
                candidate_es=float(old.diagonal(engine, proposed, context, 'train', guard_rows, 16,
                    CONFIG['guard_seed'], ledger, 'candidate_guard').mean())
                accepted=guard_accept(candidate_es, initial_es)
                if accepted: theta=proposed
                assert np.isfinite(theta).all() and sha(step_dir/'queries.npz')==queries_hash
                if critic is not None:
                    assert old.parameter_sha(critic)==critic_hash
                    # Adam is never called inside a signed/proposed ETT update.
                    optimizer_after=optimizer.state_dict()
                    assert optimizer_before['param_groups']==optimizer_after['param_groups']
                    for k, state in optimizer_before['state'].items():
                        for f, value in state.items(): torch.testing.assert_close(value, optimizer_after['state'][k][f], rtol=0, atol=0)
                np.savez_compressed(step_dir/'update.npz', preupdate=preupdate, proposed=proposed,
                    accepted_theta=theta, directions=directions, diagonal_indices=indices, signed=signed, **info)
                row=dict(iteration=iteration, accepted=accepted, candidate_guard=candidate_es,
                    accepted_guard=candidate_es if accepted else (history[-1]['accepted_guard'] if history else initial_es),
                    initial_guard=initial_es, critic_refresh=fit, frozen_critic_sha256=critic_hash,
                    checkpoint_sha256=checkpoint_hash, preupdate_sha256=array_sha(preupdate),
                    queries_sha256=queries_hash, signed_terms=signed, **info)
                history.append(row); write(step_dir/'update.json', row)
                print(f'{name} update {iteration+1}/3: proposal processed; critic/visitation frozen.', flush=True)
            final[name]=theta.copy(); histories[name]=history
            np.savez_compressed(out/'checkpoints'/f'{name}.npz', theta=theta)
    np.savez_compressed(out/'final_kernels.npz', **final)
    write(out/'training.json', dict(utc=average.timestamp(), initial_guard=initial_es, histories=histories,
        update_attempts=18, critic_refresh_steps=2400, actor_updates=0,
        initial_visitation_paired=True, final_kernels_sha256=sha(out/'final_kernels.npz')))
    return final


def run(out):
    from ett import eval_pointmaze_update_reference as evaluation
    verify(out)
    if (out/'started.json').exists(): raise ValueError('Preserve existing attempt.')
    write(out/'started.json', dict(utc=average.timestamp()))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    engine=old.Kernel(); context=dict(np.load(phase.PREVIOUS/'contexts.npz')); ledger=Ledger(out)
    before=[np.array(x) for tree in [engine.base.params, engine.nominal.params] for x in jax.tree.leaves(tree)]
    final=train(engine, out, context, ledger)
    evaluation.evaluate(engine, out, final, context, ledger)
    after=[np.array(x) for tree in [engine.base.params, engine.nominal.params] for x in jax.tree.leaves(tree)]
    for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
    verify(out)
    write(out/'completion.json', dict(utc=average.timestamp(), updates=18, actor_updates=0,
        nominal_updates=0, immutable_checkpoint_trees_verified=True, native_steps=0,
        charged_model_outputs=ledger.data['charged']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('phase', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, required=True); args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)(args.out)
