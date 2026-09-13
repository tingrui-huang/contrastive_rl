"""Bounded fixed-goal NCE pilot using the existing PointMaze convex generator."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as j
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ett.run_convex_adversarial import inputs, setup
from ett.convex_action_transition import emit, energy_score, matrices, validate_selected_set
from ett.diagonal_transition import sample_displacement, _project_samples
from ett.rollout_return import GOAL, task_reward
from ett.finite_crl import write, sha
from ett.finite_neural import parameter_sha

H, GAMMA, B = 10, .95, 32
CONFIG = dict(seeds=[0, 1], arms=['diagonal', 'joint'], updates=6, directions=4,
              roots_per_partition=64, root_time=40, rollout_repeats=8,
              calibration_repeats=16, horizons=[10, 5, 1],
              critic_initial_steps=1500, critic_refresh_steps=400, critic_final_steps=1000,
              critic_lr=.003, critic_batch=256, critic_width=64, embedding=16,
              diagonal_query_rows=128, diagonal_query_draws=8,
              guard_rows=256, guard_draws=16, evaluation_fit_rows=512, evaluation_fit_draws=32,
              visitation_queries=128, next_draws=4, next_actor_draws=4,
              sigma_diagonal=.01, sigma_response=.1, rate_diagonal=.01, rate_response=.5,
              max_diagonal_step=.03, max_response_step=.1, lambda_off=1.,
              diagonal_es_tolerance=.02, constraint_tolerance=2e-6,
              calibration_rmse_gate=.06, calibration_bias_gate=.025,
              outside_rmse_gate=.07, minimum_signal_std=.025,
              bound=1., geometry_mode='rectangle', horizon=H, discount=GAMMA, nce_B=B,
              negative_distribution=[.5, .5], alpha=0., root_seed=90000000,
              preflight_seed=91000000, critic_seed=92000000, update_seed=93000000,
              evaluation_seed=94000000, bootstrap_seed=95000000,
              model_output_cap=1500000, final_iterates_only=True)


class Ledger:
    def __init__(self, out):
        self.out = out
        self.data = dict(charged=0, cap=CONFIG['model_output_cap'], entries=[])
    def add(self, number, purpose):
        if self.data['charged'] + number > self.data['cap']:
            raise RuntimeError('Predeclared output cap exhausted; do not expand.')
        self.data['charged'] += int(number)
        self.data['entries'].append(dict(purpose=purpose, outputs=int(number)))
        write(self.out / 'ledger.json', self.data)


class Kernel:
    """Original sampler with trainable head-bias offsets, not a new projection."""
    def __init__(self):
        self.model, self.nominal, self.actor, _, self.actor_info = setup('rectangle')
        self.base = self.model.diagonal
        assert self.base.params['linear_2']['b'].shape == (16,)
        self.sample = jax.jit(self._sample, static_argnums=(5,))
        self.actor_jit = jax.jit(lambda s, key: self.actor(s, j.broadcast_to(j.asarray(GOAL), s.shape), key))
        self.rollouts = {}
        self.nll = jax.jit(lambda theta, s, a, y: -self.base._log_prob(
            self.parameters(theta), j.concatenate([s, a, a, j.broadcast_to(j.asarray(GOAL), s.shape)], -1), y[:, :2] - s[:, :2]))

    def parameters(self, theta):
        params = dict(self.base.params)
        params['linear_2'] = dict(params['linear_2'])
        params['linear_2']['b'] = params['linear_2']['b'] + theta[:16]
        return params

    def _sample(self, theta, s, a, xp, key, count):
        goal = j.broadcast_to(j.asarray(GOAL), s.shape)
        context = j.concatenate([s, xp, xp, goal], -1)
        distribution = self.base._distribution(self.parameters(theta), context)
        delta, atom = sample_displacement(distribution, key, count, self.base.delta_mean, self.base.delta_std, self.base.spec)
        anchor, details = _project_samples(s, delta, self.base.spec)
        y, diag = emit(theta[16:], s, a, xp, anchor[..., :2], 1.)
        diag['stationary_atom'] = atom
        diag['base_corrected'] = j.any(details['raw_position'] != anchor[..., :2], axis=-1)
        return y, diag

    def rollout(self, theta, states, key, horizon=H, first_actions=None):
        if horizon not in self.rollouts:
            def generate(theta, states, key, first, override):
                def step(s, data):
                    t, key = data
                    bk, ak, tk = jax.random.split(key, 3)
                    goal = j.broadcast_to(j.asarray(GOAL), s.shape)
                    xp = self.nominal.sample(s, bk, 1, goal=goal)
                    a = self.actor(s, goal, ak)
                    a = j.where((t == 0) & override, first, a)
                    y, detail = self._sample(theta, s, a, xp, tk, 1)
                    return y[:, 0], dict(next=y[:, 0], action=a, x_prime=xp,
                        reward=task_reward(y[:, 0], goal), valid=detail['box_valid'][:, 0],
                        atom=detail['stationary_atom'][:, 0], projected=detail['projection_corrected'][:, 0])
                _, data = jax.lax.scan(step, states, (j.arange(horizon), jax.random.split(key, horizon)))
                data = jax.tree.map(lambda x: j.swapaxes(x, 0, 1), data)
                data['states'] = j.concatenate([states[:, None], data.pop('next')], 1)
                return data
            self.rollouts[horizon] = jax.jit(generate)
        first = np.zeros((len(states), 2), np.float32) if first_actions is None else first_actions
        out = jax.tree.map(np.asarray, self.rollouts[horizon](j.asarray(theta), j.asarray(states), key, j.asarray(first), first_actions is not None))
        assert out['valid'].all() and np.isfinite(out['states']).all()
        np.testing.assert_array_equal(out['states'][:, 1:, 2:], out['states'][:, :-1, :6])
        assert np.abs(out['action']).max() <= 1 and np.abs(out['x_prime']).max() <= 1
        return out


def features(state, action, horizon, mean, std):
    return np.concatenate([(np.asarray(state) - mean) / std, np.asarray(action), np.asarray(horizon)[..., None] / H], -1).astype(np.float64)


class RegionCritic(nn.Module):
    """Binary FUTURE-REGION label, not a continuous F4 point-goal density."""
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(11, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 16)).double()
        self.psi = nn.Embedding(2, 16).double()
        nn.init.normal_(self.psi.weight, std=.1)
    def forward(self, x):
        return self.phi(x) @ self.psi.weight.T
    @torch.no_grad()
    def predict(self, state, action, h, mean, std):
        f = self(torch.from_numpy(features(state, action, h, mean, std))).numpy()
        if not np.isfinite(f).all() or f.max() > 600:
            raise FloatingPointError('Unsafe NCE logits; halt without clipping.')
        q = (1 - GAMMA ** np.asarray(h)) * (B - 1) * .5 * np.exp(f[..., 1])
        assert np.isfinite(q).all()
        return np.where(np.asarray(h) == 0, 0., q)


def positives(record, seed, mean, std):
    rng = np.random.default_rng(seed)
    n, hmax = record['action'].shape[:2]
    labels = np.empty((n, hmax), np.int64)
    for t in range(hmax):
        weights = GAMMA ** np.arange(hmax - t)
        offset = rng.choice(hmax - t, n, p=weights / weights.sum())
        labels[:, t] = record['reward'][np.arange(n), t + offset]
    h = np.broadcast_to(np.arange(hmax, 0, -1), (n, hmax))
    x = features(record['states'][:, :-1], record['action'], h, mean, std)
    return x.reshape(-1, 11), labels.ravel()


def fit(critic, optimizer, data, steps, seed):
    x, label = data
    x, label = torch.from_numpy(x), torch.from_numpy(label)
    rng = np.random.default_rng(seed)
    for p in critic.parameters(): p.requires_grad_(True)
    for _ in range(steps):
        idx = rng.integers(len(x), size=CONFIG['critic_batch'])
        logits = critic(x[idx])
        positive = F.softplus(-logits).gather(1, label[idx, None]).mean()
        negative = (B - 1) * F.softplus(logits).mean()
        loss = (positive + negative) / B
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    for p in critic.parameters(): p.requires_grad_(False)
    return float(loss.detach())


def collect(engine, theta, roots, seed, ledger, purpose, repeats=8):
    s = np.repeat(roots, repeats, axis=0)
    ledger.add(len(s) * H, purpose)
    return engine.rollout(theta, s, jax.random.PRNGKey(seed))


def diagonal(engine, theta, data, partition, idx, draws, seed, ledger, purpose):
    s, a, y = [data[partition + '_' + k][idx] for k in ['state', 'action', 'target']]
    ledger.add(len(idx) * draws, purpose)
    samples, detail = engine.sample(j.asarray(theta), s, a, a, jax.random.PRNGKey(seed), draws)
    validate_selected_set(s, samples, detail)
    score = np.asarray(energy_score(samples[..., :2], j.asarray(y[:, :2])))
    return score


def calibration(engine, theta, critic, roots, mean, std, seed, ledger, purpose):
    # Model-generated query histories, then independent repeated continuations
    # holding ONE first execution action fixed per context. No native snapshots.
    start = collect(engine, theta, roots, seed, ledger, purpose + '_query_contexts', repeats=1)
    rows, arrays = [], {}
    for h in CONFIG['horizons']:
        state = start['states'][:, H - h]
        action = np.asarray(engine.actor_jit(state, jax.random.PRNGKey(seed + h)))
        n, rep = len(state), CONFIG['calibration_repeats']
        ledger.add(n * rep * h, purpose + f'_h{h}')
        r = engine.rollout(theta, np.repeat(state, rep, 0), jax.random.PRNGKey(seed + 100 + h), h, np.repeat(action, rep, 0))
        values = ((1 - GAMMA) * r['reward'] @ GAMMA ** np.arange(h)).reshape(n, rep)
        prediction = critic.predict(state, action, np.full(n, h), mean, std)
        target = values.mean(1); outside = np.asarray(task_reward(state, np.broadcast_to(GOAL, state.shape))) == 0
        delta = prediction - target
        rows.append(dict(h=h, rmse=float(np.sqrt(np.mean(delta ** 2))), bias=float(delta.mean()),
                         outside_count=int(outside.sum()), outside_rmse=float(np.sqrt(np.mean(delta[outside] ** 2))) if outside.any() else None,
                         return_mean=float(target.mean()), between_context_std=float(target.std()),
                         max_mc_standard_error=float((values.std(1, ddof=1) / np.sqrt(rep)).max()),
                         value_range_excess=float(max(0., (prediction - (1 - GAMMA ** h)).max()))))
        arrays.update({f'h{h}_' + k: v for k, v in dict(state=state, action=action, prediction=prediction, returns=values, outside=outside).items()})
        if h == H:
            arrays['trajectories'] = r['states'].reshape(n, rep, h + 1, 8)
            arrays['executed_actions'] = r['action'].reshape(n, rep, h, 2)
            arrays['auxiliary_x_prime'] = r['x_prime'].reshape(n, rep, h, 2)
    passed = all(r['rmse'] <= CONFIG['calibration_rmse_gate'] and abs(r['bias']) <= CONFIG['calibration_bias_gate']
                 and (r['outside_rmse'] is None or r['outside_rmse'] <= CONFIG['outside_rmse_gate']) for r in rows)
    passed &= rows[0]['between_context_std'] >= CONFIG['minimum_signal_std']
    return dict(rows=rows, passed=bool(passed)), arrays


def surrogate(engine, theta, critic, record, mean, std, seed, ledger):
    rng = np.random.default_rng(seed)
    n, k = CONFIG['visitation_queries'], CONFIG['next_draws']
    ei = rng.integers(len(record['states']), size=n); ti = rng.integers(H, size=n)
    s = np.repeat(record['states'][ei, ti], k, 0)
    a = np.repeat(record['action'][ei, ti], k, 0)
    goal = j.broadcast_to(j.asarray(GOAL), s.shape)
    xp = engine.nominal.sample(s, jax.random.PRNGKey(seed + 1), 1, goal=goal)
    ledger.add(len(s), 'signed_surrogate_successors')
    y, detail = engine.sample(j.asarray(theta), s, a, xp, jax.random.PRNGKey(seed + 2), 1)
    validate_selected_set(s, y, detail)
    ns = np.asarray(y[:, 0]); count = CONFIG['next_actor_draws']
    repeated = np.repeat(ns, count, 0)
    next_action = np.asarray(engine.actor_jit(repeated, jax.random.PRNGKey(seed + 3)))
    h = np.repeat(np.repeat(H - ti - 1, k), count)
    q = critic.predict(repeated, next_action, h, mean, std).reshape(n, k, count).mean(-1)
    reward = np.asarray(task_reward(ns, np.broadcast_to(GOAL, ns.shape))).reshape(n, k)
    return float(np.mean(H * GAMMA ** ti * ((1 - GAMMA) * reward + GAMMA * q).mean(1)))


def prepare(out):
    if out.exists(): raise ValueError('Use a fresh directory.')
    paths = inputs()
    with np.load(paths['dataset']) as z:
        obs, actions = z['obs'], z['act']  # Audit fields deliberately not loaded.
    assert obs.shape == (6600, 51, 16) and actions.shape == (6600, 51, 2)
    assert np.all(actions[:, -1] == 0)  # Collector's terminal dummy, never executed.
    assert np.all(obs[:, :, 8:] == GOAL) and np.abs(actions).max() <= 1
    np.testing.assert_array_equal(obs[:, 1:, 2:8], obs[:, :-1, :6])
    held = np.random.default_rng(0).permutation(6600)[:660]
    teacher = np.arange(1200, 6000)
    saved = {}
    for i, (part, ids) in enumerate([('train', np.setdiff1d(teacher, held)), ('validation', np.intersect1d(teacher, held))]):
        rng = np.random.default_rng(CONFIG['root_seed'] + i)
        inside = np.asarray(task_reward(obs[ids, 40, :8], obs[ids, 40, 8:])).astype(bool)
        chosen = np.concatenate([rng.choice(ids[inside == flag], 32, replace=False) for flag in [False, True]])
        saved[part + '_roots'] = obs[chosen, 40, :8]
        saved[part + '_root_episode'] = chosen
    fit_path = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1/fit_contexts.npz')
    with np.load(fit_path) as z:
        saved.update({k: z[k][:512] for k in z.files})
    assert not np.intersect1d(saved['train_episode'], saved['validation_episode']).size
    for part in ['train', 'validation']:
        assert np.all(saved[part+'_goal'] == GOAL)
    mean = saved['train_state'].mean(0); std = np.maximum(saved['train_state'].std(0), .1)
    saved.update(mean=mean, std=std)
    out.mkdir(parents=True); (out / 'checkpoints').mkdir()
    np.savez_compressed(out / 'contexts.npz', **saved)
    write(out / 'config.json', CONFIG)
    (out / 'PROTOCOL.md').write_bytes(Path('notes/pointmaze_region_pilot.md').read_bytes())
    source = ['ett/pointmaze_region_pilot.py', 'ett/eval_pointmaze_region_pilot.py', 'scripts/test_pointmaze_region_pilot.py',
              'notes/pointmaze_region_pilot.md', 'ett/convex_action_transition.py', 'ett/diagonal_transition.py',
              'ett/rollout_return.py', 'propensity/nominal_policy.py', 'scripts/collect_swamp_windy_f4.py',
              'ett/run_convex_adversarial.py', 'ett/run_distribution_matching.py', 'crl/envs.py',
              'artifacts/finite_crl/setback_h4_s012_v1/REPORT.md',
              'artifacts/ett_shared_design/one_step_reward_v1/REPORT.md', 'artifacts/ett_geometry_comparison/zero_s01_u16_v1/REPORT.md']
    hashes = {str(p): sha(p) for p in [*paths.values(), fit_path, *source]}
    write(out / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          inputs=hashes, files={k: sha(out / k) for k in ['config.json', 'contexts.npz', 'PROTOCOL.md']},
          paths={k: str(v) for k, v in paths.items()}, jax=jax.__version__, torch=str(torch.__version__)))


def verify(out):
    assert json.loads((out / 'config.json').read_text()) == CONFIG
    p = json.loads((out / 'provenance.json').read_text())
    for name, digest in p['inputs'].items(): assert sha(name) == digest, name
    for name, digest in p['files'].items(): assert sha(out / name) == digest, name


def run(out):
    verify(out)
    if (out / 'started.json').exists(): raise ValueError('Preserve the existing attempt; no outcome-driven retries.')
    write(out / 'started.json', dict(status='started'))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    engine, ledger = Kernel(), Ledger(out)
    data = dict(np.load(out / 'contexts.npz')); mean, std = data['mean'], data['std']
    theta0 = np.zeros(48, np.float32)
    critic = RegionCritic(CONFIG['critic_seed']); opt = torch.optim.Adam(critic.parameters(), lr=CONFIG['critic_lr'])
    collected = collect(engine, theta0, data['train_roots'], CONFIG['preflight_seed'], ledger, 'preflight_training')
    nce = fit(critic, opt, positives(collected, CONFIG['preflight_seed'] + 1, mean, std), CONFIG['critic_initial_steps'], CONFIG['critic_seed'])
    check, arrays = calibration(engine, theta0, critic, data['validation_roots'], mean, std, CONFIG['preflight_seed'] + 1000, ledger, 'preflight_validation')
    np.savez_compressed(out / 'preflight_calibration.npz', **arrays)
    write(out / 'preflight.json', dict(**check, nce_loss=nce, actor=engine.actor_info))
    torch.save(dict(model=critic.state_dict(), optimizer=opt.state_dict()), out / 'checkpoints' / 'initial_critic.pt')
    np.savez(out / 'checkpoints' / 'initial.npz', theta=theta0)
    if not check['passed']:
        write(out / 'training.json', dict(status='stopped_calibration', records=[], updates=0))
        print('Initial independent calibration failed; no ETT optimization run.', flush=True); return
    guard_idx = np.arange(CONFIG['guard_rows'])
    guard_seed = CONFIG['preflight_seed'] + 2000
    initial_es = float(diagonal(engine, theta0, data, 'train', guard_idx, 16, guard_seed, ledger, 'initial_guard').mean())
    initial_model, initial_opt = copy.deepcopy(critic.state_dict()), copy.deepcopy(opt.state_dict())
    records = []
    for seed in CONFIG['seeds']:
        for arm in CONFIG['arms']:
            theta = theta0.copy(); history = []; snapshots = []
            critic = RegionCritic(CONFIG['critic_seed']); critic.load_state_dict(initial_model)
            opt = torch.optim.Adam(critic.parameters(), lr=CONFIG['critic_lr']); opt.load_state_dict(copy.deepcopy(initial_opt))
            for it in range(CONFIG['updates']):
                key = CONFIG['update_seed'] + seed * 100000 + it * 1000
                collected = collect(engine, theta, data['train_roots'], key, ledger, 'refresh_training')
                nce = fit(critic, opt, positives(collected, key + 1, mean, std), CONFIG['critic_refresh_steps'], key + 2)
                snapshots.append(theta.copy()); before = parameter_sha(critic)
                rng = np.random.default_rng(key + 3)
                directions = rng.normal(size=(4, 48))
                sigmas = np.r_[np.full(16, .01), np.full(32, .1)]
                idx = rng.integers(512, size=128)
                signed = []
                for d in directions:
                    terms = []
                    for sign in [1, -1]:
                        trial = (theta + sign * sigmas * d).astype(np.float32)
                        es = float(diagonal(engine, trial, data, 'train', idx, 8, key + 4, ledger, 'signed_diagonal').mean())
                        off = surrogate(engine, trial, critic, collected, mean, std, key + 5, ledger)
                        terms.append([es, off])
                    signed.append(terms)
                signed = np.asarray(signed)
                difference = signed[:, 0] - signed[:, 1]
                components = np.einsum('dt,di->ti', difference, directions) / (2 * sigmas * len(directions))
                # Exact structural zero: x == x_prime cancels the response.
                # Remove cross-coordinate Monte Carlo noise in this derivative.
                components[0, 16:] = 0.
                gradient = components[0] + (components[1] if arm == 'joint' else 0.)
                update = gradient * np.r_[np.full(16, .01), np.full(32, .5)]
                for sl, cap in [(slice(0, 16), .03), (slice(16, 48), .1)]:
                    update[sl] *= min(1., cap / max(np.linalg.norm(update[sl]), 1e-12))
                proposed = (theta - update).astype(np.float32)
                es = float(diagonal(engine, proposed, data, 'train', guard_idx, 16, guard_seed, ledger, 'candidate_guard').mean())
                accepted = es <= initial_es + CONFIG['diagonal_es_tolerance']
                if accepted: theta = proposed
                assert np.isfinite(theta).all() and parameter_sha(critic) == before
                history.append(dict(iteration=it, nce=nce, signed_terms=signed, diagonal_gradient=components[0],
                                    off_gradient=components[1], candidate_train_es=es, accepted=bool(accepted),
                                    matrix_norms=np.asarray(j.linalg.norm(matrices(theta[16:]), axis=(1, 2)))))
            name = f'{arm}_s{seed}'
            np.savez_compressed(out / 'checkpoints' / (name + '.npz'), theta=theta, snapshots=snapshots)
            torch.save(dict(model=critic.state_dict(), optimizer=opt.state_dict()), out / 'checkpoints' / (name + '.pt'))
            records.append(dict(name=name, seed=seed, arm=arm, history=history,
                                theta_sha256=sha(out / 'checkpoints' / (name + '.npz')),
                                critic_sha256=sha(out / 'checkpoints' / (name + '.pt'))))
            print(name + ': fixed-budget updates complete', flush=True)
    verify(out)
    write(out / 'training.json', dict(status='complete', records=records, initial_train_es=initial_es,
          updates=24, estimator='anisotropic Gaussian antithetic parameter smoothing; fixed critic and discounted visitation',
          diagonal_parameters='16 original final-head bias offsets', response_parameters=32))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('phase', choices=['prepare', 'run']); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); (prepare if args.phase == 'prepare' else run)(args.out)
