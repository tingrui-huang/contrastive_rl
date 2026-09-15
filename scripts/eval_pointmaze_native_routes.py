r"""Native PointMaze F4 evaluation with route classification, paired across policies.

Runs each checkpoint's actor in ``point_two_route_swamp_windy_f4_v0`` on the
same reset seeds (identical actuator-noise and hidden-bit streams up to an
absorbing death) and classifies every episode from the physical trajectory:

  lower     first departure from the start/fork region {(0,3),(1,3)} enters
            (1,2) or reaches the lower corridor (y < 2) before any hazard
            landing (x in [3,6), y in [3,4));
  shortcut  a hazard landing happens first;
  stuck     neither within the horizon.

``--policy mode`` acts with the policy mode ``tanh(loc)`` (the repo's greedy
evaluation, ``crl/train.py::evaluate``); ``--policy sample`` draws
``tanh(loc + scale * eps)`` with innovations paired across policies.  Reports
per-policy means, paired bootstrap differences between every pair of
checkpoints, and writes a per-episode CSV.  No training.

Usage::

    python -m scripts.eval_pointmaze_native_routes \
        --ckpt O=runs/pointmaze_absorbing_g4/g4_O_bc0p2_s0/final.pkl \
        --ckpt P=runs/pointmaze_absorbing_g4/g4_P_bc0p2_s0/final.pkl \
        --episodes 200 --policy mode --out results/g4_native_s0
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from crl import checkpoint, networks              # noqa: E402
from crl.envs import TwoRouteSwampWindyF4Env       # noqa: E402

GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
GOAL2 = GOAL[:2].astype(np.float64)
START = np.tile(np.array([.5, 3.5], np.float32), 4)
HORIZON, DISCOUNT = 50, .95
METRICS = ['reach', 'strict_success', 'discounted_return', 'undiscounted_return', 'absorbed',
           'lower_route', 'reached_y_below_2', 'shortcut', 'stuck_other',
           'first_departure_1_2', 'first_departure_2_3', 'hazard_landings']


def make_network():
  return networks.make_networks(8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
                                hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
                                use_image_obs=False, use_layer_norm=False, obs_scale=None)


def cell(xy):
  return np.clip(np.floor(np.asarray(xy)).astype(int), [0, 0], [8, 4])


def hazardous(xy):
  return (xy[..., 0] >= 3) & (xy[..., 0] < 6) & (xy[..., 1] >= 3) & (xy[..., 1] < 4)


def run_policy(params, network, episodes, reset_base, action_base, policy):
  if policy == 'mode':
    act = jax.jit(lambda p, o: jnp.tanh(network.policy_network.apply(p, o).loc))
  else:
    def _sample(p, o, key):
      d = network.policy_network.apply(p, o)
      return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
    act = jax.jit(jax.vmap(lambda p, o, k: _sample(p, o[None], k)[0], in_axes=(None, 0, 0)))
  envs = [TwoRouteSwampWindyF4Env(seed=reset_base + i, active_prob=.3) for i in range(episodes)]
  obs = np.stack([e._get_obs().copy() for e in envs]).astype(np.float32)
  np.testing.assert_array_equal(obs[:, :8], np.broadcast_to(START, (episodes, 8)))
  physical = [np.stack([e.state.copy() for e in envs])]
  rewards, dead = [], []
  for t in range(HORIZON):
    if policy == 'mode':
      a = np.asarray(act(params, jnp.asarray(obs)), np.float32)
    else:
      keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(action_base + t), i))(jnp.arange(episodes))
      a = np.asarray(act(params, jnp.asarray(obs), keys), np.float32)
    step_obs, step_r = [], []
    for e, ai in zip(envs, a):
      o, r, done, _ = e.step(ai)
      assert not done
      step_obs.append(o); step_r.append(r)
    obs = np.stack(step_obs).astype(np.float32)
    physical.append(np.stack([e.state.copy() for e in envs]))
    rewards.append(np.asarray(step_r, np.float32)); dead.append(np.asarray([e.dead for e in envs]))
  return {'physical': np.stack(physical, 1), 'reward': np.stack(rewards, 1), 'dead': np.stack(dead, 1)}


def classify(rec):
  phys, reward, dead = rec['physical'], rec['reward'], rec['dead']
  rows = []
  for i in range(len(phys)):
    c = cell(phys[i]); haz = hazardous(phys[i])
    region = ((c[:, 0] == 0) | (c[:, 0] == 1)) & (c[:, 1] == 3)
    outside = np.flatnonzero(~region)
    departure = tuple(int(v) for v in c[outside[0]]) if len(outside) else None
    in12 = (c[:, 0] == 1) & (c[:, 1] == 2)
    below = phys[i, :, 1] < 2.
    t12 = np.flatnonzero(in12 | below); thz = np.flatnonzero(haz)
    t_low = int(t12[0]) if len(t12) else None
    t_hz = int(thz[0]) if len(thz) else None
    if t_low is not None and (t_hz is None or t_low < t_hz):
      route = 'lower'
    elif t_hz is not None:
      route = 'shortcut'
    else:
      route = 'stuck_other'
    reach = bool((reward[i] > 0).any())
    rows.append({
        'reach': float(reach), 'strict_success': float(np.linalg.norm(phys[i] - GOAL2, axis=1).min() < .5),
        'discounted_return': float(reward[i] @ DISCOUNT ** np.arange(HORIZON)),
        'undiscounted_return': float(reward[i].sum()), 'absorbed': float(dead[i, -1]),
        'lower_route': float(route == 'lower'), 'reached_y_below_2': float(below.any()),
        'shortcut': float(route == 'shortcut'), 'stuck_other': float(route == 'stuck_other'),
        'first_departure_1_2': float(departure == (1, 2)), 'first_departure_2_3': float(departure == (2, 3)),
        'hazard_landings': float((haz[1:] & ~np.concatenate([[False], dead[i, :-1]])).sum()),
        'route_class': route, 'first_departure_cell': str(departure),
        'time_to_goal': int(np.argmax(reward[i] > 0) + 1) if reach else None,
        'death_time': int(np.argmax(dead[i]) + 1) if dead[i].any() else None})
  return rows


def bootstrap(values, seed, replicates=2000):
  rng = np.random.default_rng(seed)
  v = np.asarray(values, np.float64)
  draws = v[rng.integers(0, len(v), (replicates, len(v)))].mean(1)
  return {'mean': float(v.mean()), 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--ckpt', action='append', required=True, help='NAME=path/to/checkpoint.pkl (repeatable)')
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--reset-seed-base', type=int, default=9_300_000)
  ap.add_argument('--action-seed-base', type=int, default=9_400_000)
  ap.add_argument('--policy', choices=('mode', 'sample'), default='mode')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  network = make_network()
  names, per, manifest = [], {}, {}
  for spec in args.ckpt:
    name, path = spec.split('=', 1)
    step, state = checkpoint.load_checkpoint(path)
    rec = run_policy(state.policy_params, network, args.episodes, args.reset_seed_base, args.action_seed_base, args.policy)
    np.savez_compressed(os.path.join(args.out, f'native_{name}.npz'), **rec)
    per[name] = classify(rec); names.append(name)
    manifest[name] = {'path': path, 'step': int(step)}
    s = {k: float(np.mean([r[k] for r in per[name]])) for k in METRICS}
    print('%-14s reach %.3f strict %.3f absorbed %.3f lower %.3f y<2 %.3f shortcut %.3f stuck %.3f dep(1,2) %.3f disc %.2f'
          % (name, s['reach'], s['strict_success'], s['absorbed'], s['lower_route'], s['reached_y_below_2'],
             s['shortcut'], s['stuck_other'], s['first_departure_1_2'], s['discounted_return']), flush=True)
  summary = {'policy': args.policy, 'episodes': args.episodes, 'reset_seed_base': args.reset_seed_base,
             'manifest': manifest, 'policies': {}, 'comparisons': {}}
  for name in names:
    rows = per[name]
    s = {k: bootstrap([r[k] for r in rows], 0) for k in METRICS}
    s['route_classes'] = {c: v / len(rows) for c, v in Counter(r['route_class'] for r in rows).items()}
    s['first_departure_cells'] = {c: v / len(rows) for c, v in Counter(r['first_departure_cell'] for r in rows).most_common()}
    ttg = [r['time_to_goal'] for r in rows if r['time_to_goal'] is not None]
    s['time_to_goal_given_reach'] = float(np.mean(ttg)) if ttg else None
    summary['policies'][name] = s
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      summary['comparisons'][f'{a}_minus_{b}'] = {
          k: bootstrap(np.asarray([r[k] for r in per[a]]) - np.asarray([r[k] for r in per[b]]), 100 + j)
          for j, k in enumerate(METRICS)}
  with open(os.path.join(args.out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  fields = ['episode', 'policy'] + METRICS + ['route_class', 'first_departure_cell', 'time_to_goal', 'death_time']
  with open(os.path.join(args.out, 'per_episode.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for name in names:
      for i, r in enumerate(per[name]):
        w.writerow({'episode': i, 'policy': name, **{k: r[k] for k in fields[2:]}})
  for k, v in summary['comparisons'].items():
    print(k, {m: (round(v[m]['mean'], 3), [round(x, 3) for x in v[m]['ci95']]) for m in ['reach', 'absorbed', 'lower_route', 'shortcut', 'discounted_return']})


if __name__ == '__main__':
  main()
