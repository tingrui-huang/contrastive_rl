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
and passed to reset(). Every episode starts from ONE canonical pose, so the
t=0 observation carries no route information -- checked, not asserted, by the
permutation leak test below and recorded in the npz meta.

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
#: v2 = the coin-free protocol (one canonical start pose). v1 is kept
#: because its initial quaternion decoded the latent perfectly and it is
#: the control the leak check is calibrated against; do not train on it.
NAME = 'antmaze_tworoute_rockfall_v2'
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
  discarded = []

  e = 0
  while e < N:
    u = bool(u_rng.random() < args.p_active)
    route = 'detour' if u else 'shortcut'
    o = env.reset(rockfall_active=u)
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
    #: REDRAW an episode that never committed to EITHER route. Under one
    #: canonical pose the shortcut driver turns the ant itself, and ~5% of
    #: the time that turn fails and the ant shuffles in the start cell for
    #: the whole horizon. That is a demonstrator failure, not a
    #: demonstration, and it would dump ~400 junk steps into precisely the
    #: state where the route decision is made -- the one thing the coin-free
    #: benchmark measures. Episodes that DID commit to a route and then time
    #: out near the goal are kept, which is the pre-existing convention (the
    #: coin-carrying dataset shipped 2.5% such detour timeouts).
    if info.get('route') is None:
      discarded.append({'rockfall_active': u, 'route_intent': route,
                        'final_xy': [round(float(tx[e, t + 1]), 3),
                                     round(float(ty[e, t + 1]), 3)],
                        'ep_length': int(t + 1)})
      obs[e, :] = 0.0
      act[e, :] = 0.0
      continue
    lengths[e] = t + 2                     # valid obs rows (0 .. t+1)
    rows.append({'episode_id': e, 'rockfall_active': u,
                 'route_intent': route,
                 'route_realized': info.get('route'),
                 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'rock_dropped': bool(info.get('rock_dropped')),
                 'return': ret, 'ep_length': int(t + 1)})
    e += 1
    if e % 50 == 0:
      print(f'  {e}/{N} episodes ({len(discarded)} discarded)', flush=True)
  print(f'discarded {len(discarded)} uncommitted episodes '
        f'({len(discarded) / (len(discarded) + N):.3f} of draws)', flush=True)

  #: MEASURED latent-leak check on the t=0 observation. The previous
  #: revision set the initial heading from the route, and because the
  #: teacher's route is a deterministic function of the latent, obs dim 3
  #: (quat w) and dim 6 (quat z) decoded the latent perfectly (d' = 11.93,
  #: |obs[:,0,3]| ranges disjoint across routes on 400/400 episodes). Under
  #: one canonical pose no dim may separate the two route groups.
  #: Tested against a PERMUTATION NULL, not a fixed d' threshold: with
  #: unequal group sizes the sampling noise in a per-dim d' is large (se ~
  #: sqrt(1/n0 + 1/n1)) and we take a max over 58 dims, so any fixed cutoff
  #: is either vacuous or trips on noise. Shuffling the route labels gives
  #: the max-d' distribution under 'no route information at t=0' directly.
  def _dprime(g0, g1):
    pooled = np.sqrt((g0.var(0, ddof=1) + g1.var(0, ddof=1)) / 2.0) + 1e-9
    return np.abs(g0.mean(0) - g1.mean(0)) / pooled

  intent = np.array([r['route_intent'] for r in rows])
  o0, m = obs[:, 0, :], intent == 'shortcut'
  if m.sum() > 1 and (~m).sum() > 1:
    d = _dprime(o0[m], o0[~m])
    rng = np.random.default_rng(0)
    k, B = int(m.sum()), 2000
    null = np.empty(B)
    for b in range(B):
      p = rng.permutation(len(o0))
      null[b] = _dprime(o0[p][:k], o0[p][k:]).max()
    p95 = float(np.percentile(null, 95))
    leak = {'test': 'permutation null on per-dim d-prime at t=0, B=2000',
            'n_shortcut': int(m.sum()), 'n_detour': int((~m).sum()),
            'max_dprime': round(float(d.max()), 4),
            'argmax_dim': int(d.argmax()),
            'null_p95': round(p95, 4),
            'n_dims_above_null_p95': int((d > p95).sum()),
            'p_value': round(float((null >= d.max()).mean()), 4),
            'passes': bool(d.max() <= p95)}
  else:
    leak = {'passes': None, 'reason': 'a route group is empty'}
  print(f"latent-leak check: max d'={leak.get('max_dprime')} (dim "
        f"{leak.get('argmax_dim')}) vs chance p95 {leak.get('null_p95')}, "
        f"p={leak.get('p_value')} -> "
        f"{'PASS' if leak.get('passes') else 'FAIL'}", flush=True)

  meta = {'name': NAME, 'env': 'offline_ant_umaze_tworoute_rockfall',
          'obs_dim': 29, 'goal_dim': 29, 'action_dim': 8,
          'horizon': args.horizon, 'p_active': args.p_active,
          'collection_seed': args.seed,
          'teacher': 'sighted tworoute_teacher (clear->shortcut, '
                     'active->detour); BOTH drivers run from the single '
                     'canonical pose -- the route is the driver choice, not '
                     'an initial condition (see scripts/tworoute_teacher.py)',
          'learner_eval_protocol': 'reset() -- one canonical pose (native '
                                   "d4rl east); the route is the policy's "
                                   'own action',
          'latent_visibility': 'rockfall_active NEVER in obs; sidecar only',
          'latent_leak_check': leak,
          'uncommitted_discards': {
              'n': len(discarded),
              'frac_of_draws': round(len(discarded) / (len(discarded) + N), 4),
              'rule': "redrawn when info['route'] is None, i.e. the ant "
                      'never committed to either route (a failed start-cell '
                      'turn); route-committed timeouts are KEPT',
              'episodes': discarded}}
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
