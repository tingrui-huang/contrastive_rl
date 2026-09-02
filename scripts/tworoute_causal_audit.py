"""Causal dataset audit for the two-route AntMaze rockfall benchmark.

Measures whether the intended confounding actually exists BEFORE any baseline
is trained:

  OBSERVATIONAL (from the collected sidecar): route distribution, conditional
  route behaviour, and P(success | shortcut observed) -- the teacher only
  takes the shortcut when the latent is clear, so the shortcut should look
  near-perfectly safe in the data.

  INTERVENTIONAL (fresh rollouts): an 'always attempt shortcut' policy and an
  'always detour' policy executed under the env's OWN latent draw
  (Bernoulli p_active, hidden), i.e. P(success | do(shortcut)) and
  P(success | do(detour)). The route drivers are the teacher's own frozen
  controllers, so the ONLY difference from the observational number is that
  the route no longer depends on the latent.

  gap_shortcut = P(success | shortcut observed) - P(success | do(shortcut))

Writes artifacts/tworoute_rockfall_v0/causal_audit.json.

Usage: python scripts/tworoute_causal_audit.py [--n-do 250]
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
import tworoute_teacher as TT             # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
SIDE = ('artifacts/tworoute_rockfall_v0/dataset/'
        'antmaze_tworoute_rockfall_v1_sidecar.npz')


def wilson(k, n, z=1.96):
  if n == 0:
    return (None, None)
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (round(float(c - h), 4), round(float(c + h), 4))


def observational(side_path):
  s = np.load(side_path, allow_pickle=True)
  route = np.asarray(s['route_intent'])
  u = np.asarray(s['rockfall_active'], bool)
  succ = np.asarray(s['success'], bool)
  fail = np.asarray(s['failure'], bool)
  n = len(route)
  sc, dt = route == 'shortcut', route == 'detour'

  def m(x):
    return round(float(np.mean(x)), 4) if len(x) else None

  k_sc = int(succ[sc].sum())
  return {
      'n_episodes': n,
      'teacher_success': m(succ),
      'teacher_failure': m(fail),
      'P_shortcut': m(sc), 'P_detour': m(dt),
      'P_shortcut_given_clear': m(sc[~u]),
      'P_shortcut_given_active': m(sc[u]),
      'P_detour_given_clear': m(dt[~u]),
      'P_detour_given_active': m(dt[u]),
      'P_success_given_shortcut_observed': m(succ[sc]),
      'P_success_given_shortcut_observed_ci95': wilson(k_sc, int(sc.sum())),
      'P_success_given_detour_observed': m(succ[dt]),
      'n_shortcut': int(sc.sum()), 'n_detour': int(dt.sum()),
  }


def do_route(route, n, seed, horizon=400):
  """P(success | do(route)): the fixed-route driver under the env's own
  hidden latent draw. Returns per-episode rows."""
  cfg, teacher = TT.make_teacher()
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=seed)
  rows = []
  for k in range(n):
    o = env.reset(heading=('north' if route == 'shortcut' else 'east'))
    u = env.privileged_rockfall_active     # read AFTER reset, audit only
    teacher.fresh()
    info = {}
    for t in range(horizon):
      o, r, done, info = env.step(teacher.act(o, route))
      if done or r > 0:
        break
    rows.append({'u': bool(u), 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'steps': int(t + 1)})
    if (k + 1) % 50 == 0:
      print(f'  do({route}) {k + 1}/{n}', flush=True)
  return rows


def summarize_do(rows):
  n = len(rows)
  succ = [r['success'] for r in rows]
  u = np.asarray([r['u'] for r in rows], bool)
  s = np.asarray(succ, bool)
  return {'n': n, 'P_success': round(float(np.mean(s)), 4),
          'P_success_ci95': wilson(int(s.sum()), n),
          'P_failure': round(float(np.mean(
              [r['failure'] for r in rows])), 4),
          'P_active_drawn': round(float(np.mean(u)), 4),
          'P_success_given_clear': round(float(np.mean(s[~u])), 4),
          'P_success_given_active': (round(float(np.mean(s[u])), 4)
                                     if u.any() else None)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n-do', type=int, default=250)
  ap.add_argument('--seed', type=int, default=808)
  ap.add_argument('--sidecar', default=SIDE)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  obs = observational(args.sidecar)
  print('observational:', json.dumps(obs, indent=2), flush=True)
  do_sc = summarize_do(do_route('shortcut', args.n_do, args.seed))
  do_dt = summarize_do(do_route('detour', args.n_do, args.seed + 1))
  gap = round(obs['P_success_given_shortcut_observed']
              - do_sc['P_success'], 4)

  rep = {'observational': obs,
         'do_shortcut': do_sc, 'do_detour': do_dt,
         'gap_shortcut_observational_minus_interventional': gap,
         'verdict': ('CONFOUNDING PRESENT' if gap >= 0.15
                     else 'GAP TOO WEAK -- benchmark needs tuning')}
  print(json.dumps({k: v for k, v in rep.items()
                    if k != 'observational'}, indent=2), flush=True)
  with open(os.path.join(args.out_dir, 'causal_audit.json'), 'w') as f:
    json.dump(rep, f, indent=2)
  print('-> causal_audit.json', flush=True)


if __name__ == '__main__':
  main()
