"""Offline dataset collector for the two-route AntMaze rockfall benchmark.

Sighted-teacher episodes (scripts/tworoute_teacher.py): the teacher reads the
latent and takes the SHORTCUT when clear, the DETOUR when active. The learner
npz follows the repo's strict offline contract (crl/offline_audit.py):

  learner npz: obs [N, L+1, 58] float32, act [N, L+1, 8] float32 (last row
               dummy zeros), lengths [N] (valid obs rows), eval_goals [N, 2],
               meta (json string). NOTHING privileged.
  sidecar npz: per-episode audit fields (rockfall_active, route, outcome,
               return, lengths, ...) + step torso traces. NEVER a training
               input; consumed only by the causal audit.

The latent is drawn by this collector's own recorded rng (Bernoulli p_active)
and passed to reset() together with the route-matched heading (see the
teacher module docstring for why heading is part of the teacher's action).

Usage: python scripts/collect_tworoute_dataset.py [--episodes 400]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import tworoute_rockfall_ant as TR  # noqa: E402
import tworoute_teacher as TT             # noqa: E402

OUT_DIR = 'artifacts/tworoute_rockfall_v0/dataset'
NAME = 'antmaze_tworoute_rockfall_v1'
HORIZON = 400
P_ACTIVE = 0.30


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=400)
  ap.add_argument('--seed', type=int, default=606)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  cfg, teacher = TT.make_teacher()
  cfg.rockfall_max_steps = int(args.horizon)
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=args.seed)
  u_rng = np.random.default_rng(args.seed + 5000)

  N, L = args.episodes, args.horizon + 1
  obs = np.zeros((N, L, 58), np.float32)
  act = np.zeros((N, L, 8), np.float32)
  lengths = np.zeros(N, np.int64)
  eval_goals = np.zeros((N, 2), np.float32)
  tx = np.full((N, L), np.nan, np.float32)
  ty = np.full((N, L), np.nan, np.float32)
  rows = []

  for e in range(N):
    u = bool(u_rng.random() < args.p_active)
    route = 'detour' if u else 'shortcut'
    o = env.reset(rockfall_active=u,
                  heading=('east' if route == 'detour' else 'north'))
    teacher.fresh()
    obs[e, 0] = o
    tx[e, 0], ty[e, 0] = o[0], o[1]
    eval_goals[e] = o[29:31]
    ret, info = 0.0, {}
    for t in range(args.horizon):
      a = teacher.act(o, route)
      o, r, done, info = env.step(a)
      act[e, t] = a
      obs[e, t + 1] = o
      tx[e, t + 1], ty[e, t + 1] = o[0], o[1]
      ret += float(r)
      if done or r > 0:
        break
    lengths[e] = t + 2                     # valid obs rows (0 .. t+1)
    rows.append({'episode_id': e, 'rockfall_active': u,
                 'route_intent': route,
                 'route_realized': info.get('route'),
                 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'rock_dropped': bool(info.get('rock_dropped')),
                 'return': ret, 'ep_length': int(t + 1)})
    if (e + 1) % 50 == 0:
      print(f'  {e + 1}/{N} episodes', flush=True)

  meta = {'name': NAME, 'env': 'offline_ant_umaze_tworoute_rockfall',
          'obs_dim': 29, 'goal_dim': 29, 'action_dim': 8,
          'horizon': args.horizon, 'p_active': args.p_active,
          'collection_seed': args.seed,
          'teacher': 'sighted tworoute_teacher (clear->shortcut, '
                     'active->detour); heading set toward the chosen route '
                     'at reset (see scripts/tworoute_teacher.py)',
          'learner_eval_protocol': "reset(heading='random') -- 50/50 coin "
                                   'independent of the latent',
          'latent_visibility': 'rockfall_active NEVER in obs; sidecar only'}
  learner_path = os.path.join(args.out_dir, f'{NAME}.npz')
  np.savez_compressed(learner_path, obs=obs, act=act, lengths=lengths,
                      eval_goals=eval_goals, meta=json.dumps(meta))
  side_path = os.path.join(args.out_dir, f'{NAME}_sidecar.npz')
  np.savez_compressed(
      side_path,
      episode_id=np.arange(N),
      rockfall_active=np.array([r['rockfall_active'] for r in rows]),
      route_intent=np.array([r['route_intent'] for r in rows]),
      route_realized=np.array([str(r['route_realized']) for r in rows]),
      success=np.array([r['success'] for r in rows]),
      failure=np.array([r['failure'] for r in rows]),
      entered_hazard=np.array([r['entered_hazard'] for r in rows]),
      rock_dropped=np.array([r['rock_dropped'] for r in rows]),
      ep_return=np.array([r['return'] for r in rows]),
      ep_length=np.array([r['ep_length'] for r in rows]),
      step_torso_x=tx, step_torso_y=ty,
      collection_seed=np.int64(args.seed))

  succ = float(np.mean([r['success'] for r in rows]))
  fail = float(np.mean([r['failure'] for r in rows]))
  n_sc = sum(r['route_intent'] == 'shortcut' for r in rows)
  print(f'\n{N} episodes | success {succ:.3f} | failure {fail:.3f} | '
        f'shortcut {n_sc}/{N}')
  print(f'transitions {int(np.sum(lengths - 1))}')
  print('->', learner_path)
  print('->', side_path, flush=True)


if __name__ == '__main__':
  main()
