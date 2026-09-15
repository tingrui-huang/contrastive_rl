"""G0 prescreen: absorbing Manski freeze channel vs the eligible ETT at the fork.

Strictly offline. Reads obs/act of the fixed dataset and the eligible frozen
checkpoints; makes zero environment calls and zero training updates. See
PROTOCOL.md in this directory for the sealed design.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ett import pointmaze_region_pilot as region  # noqa: E402
from ett.diagonal_transition import _project_samples  # noqa: E402
from ett.rollout_return import GOAL, task_reward  # noqa: E402

OUT = Path(__file__).resolve().parent
DATASET = ROOT / 'artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz'
PLAN = ROOT / 'outputs/pointmaze_offline_causal_integration_20260915_v1/sampler_plan.npz'
ETT_FINAL = ROOT / 'outputs/pointmaze_offline_causal_integration_20260915_v1/checkpoints/ett_final.npz'
DIAGONAL = ROOT / 'artifacts/ett_distribution_matching/f4_p30_s01_guarded/s0_L0p25_lambda0/final.pkl'
NOMINAL = ROOT / 'artifacts/nominal_policy/f4_p30_expert_only_mdn_k5_s0/best.pkl'
ACTOR = ROOT / 'artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1/alpha0_seed0/final.pkl'

GRID = (9, 5)
DISCOUNT = .95
SUPPORT_MIN_ONSETS = 10
SUPPORT_MIN_ABSORBING = .5
FORCED = {'down': np.array([0., -1.], np.float32), 'right': np.array([1., 0.], np.float32)}
ARMS = ['eligible_response', 'diagonal_motion', 'diagonal_motion_absorbing', 'manski_absorbing']
CAP = 20_000_000
BOOT = 2000
GOAL2 = np.asarray(GOAL[:2], np.float64)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))


def landing_cell(xy):
    """Visible landing bin: floor of the box-clipped position (box bounds only)."""
    c = np.floor(np.clip(xy, [0., 0.], [GRID[0] - 1e-6, GRID[1] - 1e-6])).astype(np.int64)
    return c


# --------------------------------------------------------------------------- #
# Visible freeze statistics                                                     #
# --------------------------------------------------------------------------- #
def freeze_tables(obs, act, train_ids):
    xy = obs[train_ids, :, :2].astype(np.float64)
    cur, nxt = xy[:, :-1], xy[:, 1:]
    disp = np.linalg.norm(nxt - cur, axis=-1)
    stationary = disp <= 1e-7
    E, T = stationary.shape
    onset = np.zeros_like(stationary)
    onset[:, :-1] = (~stationary[:, :-1]) & stationary[:, 1:]
    # run length of the stationary spell starting at t+1, and whether it reaches the end
    run_to_end = np.zeros_like(stationary)
    tail = np.ones(E, bool)
    for t in range(T - 1, -1, -1):
        tail = tail & stationary[:, t]
        run_to_end[:, t] = tail
    counts = np.zeros(GRID, np.int64)
    absorbing = np.zeros(GRID, np.int64)
    cell = landing_cell(nxt)
    for e, t in zip(*np.nonzero(onset)):
        c = tuple(cell[e, t])
        counts[c] += 1
        absorbing[c] += int(run_to_end[e, t + 1])
    frac = np.where(counts > 0, absorbing / np.maximum(counts, 1), np.nan)
    support = (counts >= SUPPORT_MIN_ONSETS) & (np.nan_to_num(frac) >= SUPPORT_MIN_ABSORBING)
    # persistence given the full frozen F4 signature (reported, not used)
    frames = obs[train_ids, :-1, :8].reshape(E, T, 4, 2)
    signature = (np.abs(frames - frames[:, :, :1]).max(axis=(2, 3)) == 0.) & (np.arange(T)[None] >= 3)
    ccell = landing_cell(cur)
    persist = {}
    for c in zip(*np.nonzero(counts > 0)):
        m = signature & (ccell[..., 0] == c[0]) & (ccell[..., 1] == c[1])
        persist[str(c)] = dict(rows=int(m.sum()), stay=float(stationary[m].mean()) if m.any() else None)
    table = {str((int(i), int(j))): dict(onsets=int(counts[i, j]), absorbing=int(absorbing[i, j]),
                                          absorbing_fraction=float(frac[i, j]), in_support=bool(support[i, j]))
             for i, j in zip(*np.nonzero(counts > 0))}
    return dict(support=support, onset_table=table, persistence_given_signature=persist,
                totals=dict(transitions=int(E * T), stationary=int(stationary.sum()),
                            onsets=int(onset.sum()), absorbing_onsets=int(absorbing.sum())))


def manski_tables(obs, act, ids, support):
    """Descriptive per-step bound at three decision cells; not a model input."""
    xy = obs[ids, :, :2].astype(np.float64)
    cur, nxt = xy[:, :-1], xy[:, 1:]
    disp = np.linalg.norm(nxt - cur, axis=-1)
    stationary = disp <= 1e-7
    freeze_next = np.zeros_like(stationary)
    freeze_next[:, :-1] = stationary[:, 1:]
    alive = ~stationary
    ccell, ncell = landing_cell(cur), landing_cell(nxt)
    out = {}
    for name, c in [('holding_2_3', (2, 3)), ('fork_1_3', (1, 3)), ('swamp_3_3', (3, 3))]:
        sel = alive & (ccell[..., 0] == c[0]) & (ccell[..., 1] == c[1])
        n = int(sel.sum())
        rows = {}
        lc = ncell[sel]
        fz = freeze_next[sel]
        keys, inv, cnt = np.unique(lc, axis=0, return_inverse=True, return_counts=True)
        for k, (key, m) in enumerate(zip(keys, cnt)):
            if m < 20:
                continue
            pi = m / n
            q = float(fz[inv.ravel() == k].mean())
            ins = bool(support[key[0], key[1]])
            rows[str((int(key[0]), int(key[1])))] = dict(pi=float(pi), q_obs=q, in_support=ins,
                                                        manski_upper=float(pi * q + (1 - pi)) if ins else q)
        out[name] = dict(n=n, bins=rows)
    return out


# --------------------------------------------------------------------------- #
# Rollouts                                                                      #
# --------------------------------------------------------------------------- #
class Prescreen:
    def __init__(self, support):
        self.engine = region.Kernel()
        self.base = self.engine.base
        self.support = jnp.asarray(support)
        self.fns = {}

    def rollout(self, arm, states, first, key, horizon, theta=None):
        if arm == 'eligible_response':
            out = self.engine.rollout(theta, states, key, horizon, np.broadcast_to(first, (len(states), 2)).astype(np.float32))
            reward = out['reward']
            xy = out['states'][:, :, :2]
            frozen = np.zeros(reward.shape[0], bool)
            return dict(reward=reward, xy=xy, frozen_end=frozen, atom=out['atom'],
                        agree=np.zeros_like(out['atom']), hazard_alive=np.zeros_like(out['atom']),
                        onset_manski=np.zeros_like(out['atom']), onset_atom=np.zeros_like(out['atom']))
        use_absorb = arm in ('diagonal_motion_absorbing', 'manski_absorbing')
        use_manski = arm == 'manski_absorbing'
        sig = (use_absorb, use_manski, int(horizon))
        if sig not in self.fns:
            self.fns[sig] = jax.jit(self._make(use_absorb, use_manski, int(horizon)))
        out = self.fns[sig](jnp.asarray(states), jnp.asarray(first), key)
        out = jax.tree.map(np.asarray, out)
        assert np.isfinite(out['xy']).all()
        return out

    def _make(self, use_absorb, use_manski, horizon):
        base, nominal, actor, support = self.base, self.engine.nominal, self.engine.actor, self.support
        spec, params = base.spec, base.params
        lo = jnp.array([0., 0.]); hi = jnp.array([GRID[0] - 1e-6, GRID[1] - 1e-6])

        def cell_of(xy):
            return jnp.floor(jnp.clip(xy, lo, hi)).astype(jnp.int32)

        def in_support(cell):
            return support[cell[:, 0], cell[:, 1]]

        def generate(states, first, key):
            first_b = jnp.broadcast_to(first, (states.shape[0], 2))

            def step(carry, data):
                s, frozen = carry
                t, key = data
                kb, ka, kt = jax.random.split(key, 3)
                goal = jnp.broadcast_to(jnp.asarray(GOAL), s.shape)
                xp = nominal.sample(s, kb, 1, goal=goal)
                a = actor(s, goal, ka)
                a = jnp.where(t == 0, first_b, a)
                context = jnp.concatenate([s, a, a, goal], -1)
                delta, atom = base._sample_delta(params, context, kt, 1)
                y, _ = _project_samples(s, delta, spec)
                y, atom = y[:, 0], atom[:, 0]
                # frozen paths: XY unchanged, history shifts
                y = jnp.where(frozen[:, None], jnp.concatenate([s[:, :2], s[:, :6]], -1), y)
                land = cell_of(y[:, :2])
                agree = jnp.all(cell_of(s[:, :2] + a) == cell_of(s[:, :2] + xp), axis=-1)
                ins = in_support(land)
                onset_atom = use_absorb & (~frozen) & atom & ins
                onset_manski = use_manski & (~frozen) & (~atom) & (~agree) & ins
                frozen_next = frozen | onset_atom | onset_manski
                reward = task_reward(y, goal) * (1. - frozen.astype(jnp.float32))
                hazard_alive = (~frozen) & ins
                return (y, frozen_next), dict(xy=y[:, :2], reward=reward, atom=atom, agree=agree,
                                              hazard_alive=hazard_alive, onset_manski=onset_manski,
                                              onset_atom=onset_atom)

            frozen0 = jnp.zeros(states.shape[0], bool)
            (_, frozen_end), data = jax.lax.scan(step, (states, frozen0),
                                                 (jnp.arange(horizon), jax.random.split(key, horizon)))
            data = jax.tree.map(lambda v: jnp.swapaxes(v, 0, 1), data)
            data['xy'] = jnp.concatenate([states[:, None, :2], data['xy']], 1)
            data['frozen_end'] = frozen_end
            return data
        return generate


def path_metrics(out, horizon):
    r = out['reward'][:, :horizon]
    disc = DISCOUNT ** np.arange(horizon)
    xy = out['xy'][:, :horizon + 1]
    dist = np.linalg.norm(xy - GOAL2, axis=-1)
    return dict(
        discounted_return=(r * disc).sum(1),
        goal_reach=(r > 0).any(1).astype(np.float64),
        strict_success=(dist.min(1) < .5).astype(np.float64),
        lower_route=(xy[:, :, 1] < 2.).any(1).astype(np.float64),
        absorbed=out['frozen_end'].astype(np.float64),
        hazard_alive=out['hazard_alive'][:, :horizon].any(1).astype(np.float64),
        onset_manski=out['onset_manski'][:, :horizon].any(1).astype(np.float64),
        onset_atom=out['onset_atom'][:, :horizon].any(1).astype(np.float64),
        agree_rate=out['agree'][:, :horizon].mean(1),
        atom_rate=out['atom'][:, :horizon].mean(1))


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, np.float64)
    n = len(values)
    draws = values[rng.integers(n, size=(BOOT, n))].mean(1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--paths', type=int, default=64)
    args = p.parse_args()
    t0 = time.time()
    with np.load(DATASET, allow_pickle=False) as z:
        obs, act = np.asarray(z['obs']), np.asarray(z['act'])
    assert obs.shape == (6600, 51, 16)
    np.testing.assert_array_equal(obs[:, 1:, 2:8], obs[:, :-1, :6])
    heldout = np.random.default_rng(0).permutation(6600)[:660]
    train = np.setdiff1d(np.arange(6600), heldout)
    plan = np.load(PLAN)
    ce, ct = plan['fork_context_episode'], plan['fork_context_time']
    assert np.isin(ce, heldout).all() and len(ce) == 656
    theta = np.load(ETT_FINAL)
    theta = np.asarray(theta[theta.files[0]] if 'theta' not in theta.files else theta['theta'], np.float32)
    assert theta.shape == (48,)

    tables = freeze_tables(obs, act, train)
    support = tables['support']
    eligible = np.intersect1d(np.arange(1200, 6000), train)
    manski = dict(all_train=manski_tables(obs, act, train, support),
                  eligible_teacher_train=manski_tables(obs, act, eligible, support))
    print('absorbing support cells:', [tuple(map(int, c)) for c in zip(*np.nonzero(support))], flush=True)
    write(OUT / 'freeze_tables.json', dict(onset_table=tables['onset_table'],
                                           persistence_given_signature=tables['persistence_given_signature'],
                                           totals=tables['totals'],
                                           support_cells=[list(map(int, c)) for c in zip(*np.nonzero(support))],
                                           manski=manski))

    if args.smoke:
        ce, ct = ce[:8], ct[:8]
    K = args.paths
    roots = obs[ce, ct, :8].astype(np.float32)
    lengths = (50 - ct).astype(int)
    pre = Prescreen(support)
    theta_j = jnp.asarray(theta)
    charged = 0
    per_root = {arm: {name: {} for name in FORCED} for arm in ARMS}
    for h in sorted(np.unique(lengths)):
        ix = np.flatnonzero(lengths == h)
        states = np.repeat(roots[ix], K, axis=0)
        key = jax.random.PRNGKey(1000 + int(h))
        for name, first in FORCED.items():
            for arm in ARMS:
                charged += len(states) * int(h)
                assert charged <= CAP, 'budget cap'
                out = pre.rollout(arm, states, first, key, int(h), theta=theta_j)
                m = path_metrics(out, int(h))
                for metric, v in m.items():
                    per_root[arm][name].setdefault(metric, np.full(len(ce), np.nan))[ix] = v.reshape(len(ix), K).mean(1)
        print(f'h={h} roots={len(ix)} charged={charged:,} elapsed={time.time() - t0:.0f}s', flush=True)

    results = dict(config=dict(paths=K, roots=int(len(ce)), discount=DISCOUNT, bootstrap=BOOT,
                               support_min_onsets=SUPPORT_MIN_ONSETS, support_min_absorbing=SUPPORT_MIN_ABSORBING,
                               forced={k: v.tolist() for k, v in FORCED.items()}, smoke=args.smoke),
                   charged_model_transitions=int(charged), arms={})
    saved = {}
    for arm in ARMS:
        results['arms'][arm] = {}
        for metric in per_root[arm]['down']:
            d, r = per_root[arm]['down'][metric], per_root[arm]['right'][metric]
            assert np.isfinite(d).all() and np.isfinite(r).all()
            saved[f'{arm}/down/{metric}'], saved[f'{arm}/right/{metric}'] = d, r
            results['arms'][arm][metric] = dict(down=bootstrap(d, 1), right=bootstrap(r, 2),
                                                down_minus_right=bootstrap(d - r, 3))
        gr = results['arms'][arm]['goal_reach']
        results['arms'][arm]['goal_reach_ratio_down_over_right'] = (
            gr['down']['mean'] / gr['right']['mean'] if gr['right']['mean'] > 0 else None)
    a3 = results['arms']['manski_absorbing']
    ret = a3['discounted_return']['down_minus_right']
    ratio = a3['goal_reach_ratio_down_over_right']
    a0 = results['arms']['eligible_response']['discounted_return']['down_minus_right']
    results['decision'] = dict(
        primary_return_gap_positive_ci=bool(ret['lo'] > 0),
        goal_reach_ratio=ratio, ratio_at_least_1p5=bool(ratio is not None and ratio >= 1.5),
        g0_pass=bool(ret['lo'] > 0 and ratio is not None and ratio >= 1.5),
        eligible_ett_action_blind=bool(abs(a0['mean']) < .05))
    results['provenance'] = dict(
        commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [DATASET, PLAN, ETT_FINAL, DIAGONAL, NOMINAL, ACTOR]},
        ett_final_offset_norm=float(np.linalg.norm(theta)),
        ett_final_response_block_frobenius=np.sqrt((theta[16:].reshape(8, 2, 2) ** 2).sum((1, 2))).tolist(),
        environment_calls=0, training_updates=0, hidden_fields_read=[], jax=jax.__version__, numpy=np.__version__,
        elapsed_seconds=time.time() - t0)
    np.savez_compressed(OUT / ('per_root_smoke.npz' if args.smoke else 'per_root.npz'),
                        fork_context_episode=ce, fork_context_time=ct, **saved)
    write(OUT / ('results_smoke.json' if args.smoke else 'results.json'), results)
    for arm in ARMS:
        a = results['arms'][arm]
        print(f"{arm:28s} return down {a['discounted_return']['down']['mean']:.3f} right {a['discounted_return']['right']['mean']:.3f} "
              f"gap {a['discounted_return']['down_minus_right']['mean']:+.3f} [{a['discounted_return']['down_minus_right']['lo']:+.3f},{a['discounted_return']['down_minus_right']['hi']:+.3f}] "
              f"reach d/r {a['goal_reach']['down']['mean']:.3f}/{a['goal_reach']['right']['mean']:.3f} "
              f"absorbed d/r {a['absorbed']['down']['mean']:.3f}/{a['absorbed']['right']['mean']:.3f}")
    print('decision', results['decision'])


if __name__ == '__main__':
    main()
