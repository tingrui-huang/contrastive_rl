"""One frozen-adversarial-ETT experiment: shared anchors, short/long goals."""
import argparse
import dataclasses
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from crl import checkpoint
from ett.future_window import batch_stream, compact_audit, digest_batch
from ett.policy_improvement import make_generator, motion_diagnostics, replay, tree_sha
from ett.run_convex_adversarial import inputs, setup
from ett.run_policy_improvement import dataset, learner_setup, model_paths, MODEL_ROOT
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha

SPEC = Path('notes/ett_future_window_spec.md')
CONFIG = dict(seeds=[0, 1], arms=['A', 'B', 'C'], updates=2000, refreshes=20,
    block_updates=100, contexts=256, batch_size=256, synthetic_fraction=.1,
    anchor_indices=[0, 1, 2], discount=.95, original_horizon=50,
    context_times=[0, 47], behavior='unchanged initial stochastic actor',
    windows=dict(B='i+1..3', C='i+1..H; H=50-original_time'),
    evaluation_episodes=128, evaluation_reset_base=31000000, model_evaluation_seed=32000000,
    learner_seed_base=33000000, context_seed_base=34000000, model_generation_base=35000000,
    rng_bases=dict(anchor=36000000, future=37000000, permutation=38000000, offline=39000000),
    bootstrap_seed=40000000, primary_model_cap=512000, total_model_cap=650000,
    native_step_cap=50000, checkpoint_selection='final iterate only',
    native_action='deterministic tanh(loc)', failure_negative_alpha=0.)


def frozen_signature(model, nominal, initial):
    diagonal = model.diagonal
    return tree_sha((diagonal.params, diagonal.context_mean, diagonal.context_std,
        diagonal.delta_mean, diagonal.delta_std, nominal.params, nominal.context_mean,
        nominal.context_std, initial.policy_params, np.asarray(model.bound)))


def verify(root):
    assert read(root/'config.json') == CONFIG
    for path, digest in read(root/'provenance.json')['sha256'].items():
        assert file_sha(path) == digest, path
    for name, digest in read(root/'prepared.json')['sha256'].items():
        assert file_sha(root/name) == digest, name


def prepare(root):
    if root.exists():
        raise ValueError('use a fresh output directory')
    paths = inputs()
    previous = Path('artifacts/ett_policy_improvement/h3_mix10_u2000_s01_v1')
    old = read(previous/'provenance.json')
    for path in paths.values():
        assert file_sha(path) == old['sha256'][path.as_posix()]
    for seed in [0, 1]:
        path = model_paths()[f's{seed}_adversarial']
        assert file_sha(path) == read(MODEL_ROOT/'training.json')['arms'][f's{seed}_adversarial']['checkpoint_sha256']
    cfg, _, _ = learner_setup()
    step, initial = checkpoint.load_checkpoint(paths['actor'])
    assert step == 150000 and tree_sha(initial) == old['initial_state_tree_sha256']
    model, nominal, _, _, _ = setup()
    obs, _, ids, heldout = dataset()
    selected = {}
    for seed in [0, 1]:
        rng = np.random.default_rng(CONFIG['context_seed_base']+seed)
        rows, times = [], []
        for _ in range(20):
            rows.append(rng.integers(len(ids), size=256))
            times.append(rng.integers(0, 48, size=256))
        e, t = np.array(rows), np.array(times)
        selected[f's{seed}_episode'] = ids[e]
        selected[f's{seed}_time'] = t
        selected[f's{seed}_state'] = obs[e, t, :8]
    root.mkdir(parents=True)
    (root/'checkpoints').mkdir()
    shutil.copyfile(SPEC, root/'SPEC.md')
    write(root/'config.json', CONFIG)
    write(root/'learner_config.json', dataclasses.asdict(cfg))
    np.savez_compressed(root/'contexts.npz', **selected, train_ids=ids, heldout_ids=heldout,
                        evaluation_reset_seeds=CONFIG['evaluation_reset_base']+np.arange(128))
    sources = ['ett/future_window.py', 'ett/run_future_window.py', 'ett/eval_future_window.py',
        'scripts/test_future_window.py', 'ett/policy_improvement.py', 'ett/run_policy_improvement.py',
        'ett/eval_policy_improvement.py', 'ett/convex_action_transition.py', 'ett/diagonal_transition.py',
        'ett/fixed_actor_continuation.py', 'propensity/nominal_policy.py',
        'crl/losses.py', 'crl/networks.py', 'crl/replay.py', 'crl/envs.py', str(SPEC)]
    required = list(paths.values()) + [paths['nominal'].with_name('config.json'),
        paths['actor'].with_name('arm_provenance.json'), previous/'REPORT.md', previous/'provenance.json',
        previous/'verification.json', MODEL_ROOT/'REPORT.md', MODEL_ROOT/'training.json']
    required += [model_paths()[f's{s}_adversarial'] for s in [0, 1]]
    required += list(map(Path, sources))
    write(root/'provenance.json', dict(sha256={p.as_posix(): file_sha(p) for p in required},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        initial_training_state_sha256=tree_sha(initial), initial_policy_sha256=tree_sha(initial.policy_params),
        initial_critic_sha256=tree_sha(initial.q_params), frozen_signature=frozen_signature(model, nominal, initial),
        training_inputs=['offline obs', 'offline act', 'shared generated states/actions/valid lengths'],
        nominal_conditioning=nominal.spec.conditioning, normalization='original; no refitting',
        upstream_overlap='initial actor and critic previously saw all original episodes; continuation uses 5940 training episodes only',
        device=[str(d) for d in jax.devices()]))
    write(root/'prepared.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        sha256={name: file_sha(root/name) for name in ['config.json', 'learner_config.json', 'SPEC.md', 'contexts.npz', 'provenance.json']}))
    print('Saved protocol, context schedule and evaluation seeds before generation/training.', flush=True)


def generate(root):
    verify(root)
    if (root/'generation.json').exists():
        raise ValueError('shared generation already exists')
    model, nominal, _, _, _ = setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    _, net, _ = learner_setup()
    frozen = frozen_signature(model, nominal, initial)
    schedule = dict(np.load(root/'contexts.npz'))
    records = {}
    total = 0
    # Group by exact remaining length; no model step is queried after timeout.
    generators = {h: make_generator(model, nominal, net, h) for h in range(3, 51)}
    for seed in [0, 1]:
        theta = np.load(model_paths()[f's{seed}_adversarial'])['theta']
        initial_states = schedule[f's{seed}_state'].reshape(-1, 8)
        horizon = (50-schedule[f's{seed}_time']).reshape(-1)
        record = dict(states=np.full((5120, 51, 8), np.nan, np.float32))
        diagnostics = []
        for h in range(3, 51):
            rows = np.flatnonzero(horizon == h)
            if not len(rows):
                continue
            key = jax.random.PRNGKey(CONFIG['model_generation_base']+seed*1000+h)
            value = jax.tree.map(np.asarray, generators[h](initial.policy_params, theta, initial_states[rows], key))
            diagnostics.append(dict(horizon=h, paths=len(rows), **motion_diagnostics(value)))
            record['states'][rows, :h+1] = value['states']
            for field, array in value.items():
                if field == 'states':
                    continue
                if field not in record:
                    record[field] = np.full((5120, 50)+array.shape[2:], False if array.dtype==bool else np.nan, array.dtype)
                record[field][rows, :h] = array
            total += len(rows)*h
            if h % 10 == 0 or h == 50:
                print('shared model', seed, 'remaining horizon', h, 'valid steps', total, flush=True)
        # NaN storage tails are not trajectories or training targets.
        record = {k: v.reshape((20, 256)+v.shape[1:]) for k, v in record.items()}
        auxiliary = {k: record.pop(k) for k in ['aux_x_prime', 'anchor', 'projection', 'boundary', 'valid', 'atom']}
        record['horizon'] = horizon.reshape(20, 256)
        record['episode'] = schedule[f's{seed}_episode']
        record['time'] = schedule[f's{seed}_time']
        np.savez_compressed(root/f's{seed}_shared.npz', **record)
        np.savez_compressed(root/f's{seed}_shared_auxiliary.npz', **auxiliary)
        records[str(seed)] = dict(diagnostics=diagnostics,
            shared_sha256=file_sha(root/f's{seed}_shared.npz'), auxiliary_sha256=file_sha(root/f's{seed}_shared_auxiliary.npz'))
    assert frozen_signature(model, nominal, initial) == frozen
    assert total <= CONFIG['primary_model_cap']
    verify(root)
    write(root/'generation.json', dict(seeds=records, primary_model_transitions=total,
        fixed_behavior_policy_sha256=tree_sha(initial.policy_params), frozen_signature=frozen))


def shared_data(root, seed):
    manifest = read(root/'generation.json')['seeds'][str(seed)]
    assert file_sha(root/f's{seed}_shared.npz') == manifest['shared_sha256']
    with np.load(root/f's{seed}_shared.npz') as z:
        # Rewards, diagnostics and auxiliary actions never enter the sampler.
        return {k: z[k] for k in ['states', 'action', 'horizon']}


def stream(root, seed, arm, obs, act):
    offline = replay(obs, act, CONFIG['rng_bases']['offline']+seed)
    shared = None if arm == 'A' else shared_data(root, seed)
    return batch_stream(offline, shared, arm, seed, CONFIG['rng_bases'])


def audit(root):
    verify(root)
    if (root/'pretraining_audit.json').exists():
        raise ValueError('audit already completed')
    obs, act, _, _ = dataset()
    findings = {}
    for seed in [0, 1]:
        b = stream(root, seed, 'B', obs, act)
        c = stream(root, seed, 'C', obs, act)
        digests = [hashlib.sha256(), hashlib.sha256()]
        different = 0
        for (short, sb), (long, sc) in zip(b, c):
            for field in ['source', 'permutation', 'episode', 'time', 'horizon']:
                np.testing.assert_array_equal(sb[field], sc[field])
            for field in ['episode', 'time', 'future']:
                np.testing.assert_array_equal(sb['offline'][field], sc['offline'][field])
            np.testing.assert_array_equal(short.observation[:, :8], long.observation[:, :8])
            np.testing.assert_array_equal(short.action, long.action)
            np.testing.assert_array_equal(short.next_observation[:, :8], long.next_observation[:, :8])
            assert np.all(sb['future'] <= 3) and np.all(sc['future'] <= sc['horizon'])
            assert np.all(sb['future'] > sb['time']) and np.all(sc['future'] > sc['time'])
            assert np.isfinite(short.observation).all() and np.isfinite(long.observation).all()
            different += int(np.sum(sb['future'] != sc['future']))
            digest_batch(digests[0], short, sb)
            digest_batch(digests[1], long, sc)
        assert digests[0].hexdigest() == digests[1].hexdigest()
        findings[str(seed)] = dict(common_batch_sha256=digests[0].hexdigest(),
            batches=2000, synthetic_anchors=51200, different_future_indices=different)
    write(root/'pretraining_audit.json', dict(status='passed', seeds=findings,
        all_anchor_successor_action_offline_source_permutation_checks_exact=True))
    print('Full 4000-batch B/C matching audit passed before learning.', flush=True)


def train(root, smoke=False):
    verify(root)
    audit_result = read(root/'pretraining_audit.json')
    assert audit_result['status'] == 'passed'
    phase = 'smoke' if smoke else 'training'
    if (root/f'{phase}.json').exists():
        raise ValueError('training phase already exists')
    _, net, update = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    obs, act, _, _ = dataset()
    result = {}
    for seed in ([0] if smoke else [0, 1]):
        for arm in (['B', 'C'] if smoke else ['A', 'B', 'C']):
            name = f's{seed}_{arm}'
            state = initial._replace(key=jax.random.PRNGKey(CONFIG['learner_seed_base']+seed))
            start_sha = tree_sha(state)
            digest = hashlib.sha256()
            goal_digest = hashlib.sha256()
            curves, samples, chunk = [], [], []
            began = time.monotonic()
            budget = 5 if smoke else 2000
            for u, (batch, indices) in enumerate(stream(root, seed, arm, obs, act)):
                digest_batch(digest, batch, indices)
                goal_digest.update(np.ascontiguousarray(batch.observation[:, 8:]).tobytes())
                samples.append(compact_audit(indices))
                state, values = update(state, jax.tree.map(jnp.asarray, batch))
                values = {k: float(v) for k, v in values.items()}
                assert all(np.isfinite(v) for v in values.values()), (name, u)
                chunk.append(values)
                if (u+1) % 100 == 0 or u+1 == budget:
                    curves.append(dict(update=u+1, **{k: np.mean([v[k] for v in chunk]) for k in chunk[0]}))
                    chunk = []
                    print(name, u+1, 'finite losses; seconds', round(time.monotonic()-began, 1), flush=True)
                if u+1 == budget:
                    break
            assert all(np.isfinite(v).all() for v in jax.tree.leaves(state))
            assert tree_sha(state.policy_params) != tree_sha(initial.policy_params)
            assert tree_sha(state.q_params) != tree_sha(initial.q_params)
            if not smoke and arm != 'A':
                assert digest.hexdigest() == audit_result['seeds'][str(seed)]['common_batch_sha256']
            record = dict(updates=budget, initial_state_sha256=start_sha, final_state_sha256=tree_sha(state),
                policy_sha256=tree_sha(state.policy_params), critic_sha256=tree_sha(state.q_params),
                common_batch_sha256=digest.hexdigest(), goal_batch_sha256=goal_digest.hexdigest(),
                synthetic_anchors=sum(int(s['count']) for s in samples), curves=curves, seconds=time.monotonic()-began)
            if not smoke:
                path = root/'checkpoints'/f'{name}_final.pkl'
                assert not path.exists()
                checkpoint.save_named(str(root/'checkpoints'), name+'_final', 152000, state)
                record['checkpoint_sha256'] = file_sha(path)
                np.savez_compressed(root/f'{name}_sample_audit.npz', **{k: np.stack([s[k] for s in samples]) for k in samples[0]})
            result[name] = record
    verify(root)
    write(root/f'{phase}.json', dict(status='complete', runs=result,
        frozen_files_verified=True, no_generator_called_during_training=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--phase', choices=['prepare', 'generate', 'audit', 'smoke', 'train'], required=True)
    args = parser.parse_args()
    root = Path(args.out_dir)
    if args.phase in ['smoke', 'train']:
        train(root, smoke=args.phase == 'smoke')
    else:
        {'prepare': prepare, 'generate': generate, 'audit': audit}[args.phase](root)


if __name__ == '__main__':
    main()
