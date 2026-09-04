"""Causal dataset audit for the V3 two-route rockfall pair.

Port of scripts/tworoute_causal_audit.py onto the V3 controlled pair
(crl/tworoute_rockfall_v3.py). Measures whether the intended confounding
actually exists BEFORE any baseline is trained:

  OBSERVATIONAL (from the collected sidecar): route distribution, conditional
  route behaviour, and P(success | shortcut observed) -- the teacher only
  takes the shortcut when the latent is clear, so the shortcut should look
  near-perfectly safe in the data. One deliberate deviation from v2: every
  conditional is computed on route_REALIZED (info['route']; band entry is
  authoritative), not route intent -- intent is a deterministic function of
  the latent, so intent-based conditionals restate the teacher's policy
  instead of measuring the data (the same defect the v3 teacher audit
  dropped from its gates).

  INTERVENTIONAL (fresh rollouts): an 'always attempt shortcut' policy and an
  'always detour' policy executed under the env's OWN latent draw
  (Bernoulli p_active, hidden), i.e. P(success | do(shortcut)) and
  P(success | do(detour)). The route drivers are the teacher's frozen
  walker-only relay, so the ONLY difference from the observational number is
  that the route no longer depends on the latent.

  gap_shortcut = P(success | shortcut observed) - P(success | do(shortcut))

  DISCOUNTED (new in V3 -- the pair's manipulated variable): per-episode
  0.99**steps if success else 0.0. The headline mean is over ALL episodes
  (that is the convention behind the pre-registered reference numbers: e.g.
  br do(shortcut) 0.323 = 0.70 * 0.99**77); the *_given_success variant is
  reported alongside. The oracle is composed from the do() arms at nominal
  p_active: (1-p)*E[.|do(shortcut), clear] + p*E[.|do(detour), active] --
  the sighted teacher's routes conditioned on the latent it reads.

Writes artifacts/tworoute_rockfall_v3/<variant>/causal_audit.json.

Usage: python scripts/tworoute_v3_causal_audit.py --variant tr [--n-do 250]
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
import tworoute_v3_teacher as TT          # noqa: E402

OUT_ROOT = 'artifacts/tworoute_rockfall_v3'
NAME_TMPL = 'antmaze_tworoute_rockfall_v3{variant}'
GAMMA = 0.99
#: same confounding-gap bar as the v2 audit.
GAP_THRESHOLD = 0.15

#: Pre-registered driver reference numbers (measured BEFORE any V3 dataset
#: was collected). Keep in sync with collect_tworoute_v3_dataset.py /
#: eval_tworoute_v3_baseline.py. Sparse refs are shared across the pair by
#: design; only the discounted incentive differs.
SPARSE_REFS = {'always_shortcut': 0.70, 'always_detour': 0.96,
               'oracle': 0.988}
DISCOUNTED_REFS = {
    'tr': {'shortcut': 0.146, 'detour': 0.185, 'oracle': 0.201,
           'best_blind': 'detour'},
    'br': {'shortcut': 0.323, 'detour': 0.100, 'oracle': 0.353,
           'best_blind': 'shortcut'},
}


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
  route = np.asarray(s['route_realized'])
  intent = np.asarray(s['route_intent'])
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
      'route_basis': 'route_realized (band entry authoritative); the v2 '
                     'audit conditioned on intent',
      'intent_realized_mismatch': int((route != intent).sum()),
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


def observational_discounted(side_path):
  """Discounted means from the sidecar, by observed (realized) route."""
  s = np.load(side_path, allow_pickle=True)
  route = np.asarray(s['route_realized'])
  succ = np.asarray(s['success'], bool)
  steps = np.asarray(s['ep_length'], float)
  g = np.where(succ, GAMMA ** steps, 0.0)

  def m(mask):
    return round(float(g[mask].mean()), 4) if mask.any() else None

  return {'overall': round(float(g.mean()), 4),
          'shortcut_observed': m(route == 'shortcut'),
          'detour_observed': m(route == 'detour')}


def do_route(variant, route, n, seed, horizon=TT.HORIZON):
  """P(success | do(route)): the teacher's fixed-route relay driver under
  the env's OWN hidden latent draw. Returns per-episode rows (steps kept
  for the discounted section)."""
  cfg, teacher = TT.make_teacher(variant)
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env(TT.env_name(variant), cfg, seed=seed)
  rows = []
  for k in range(n):
    o = env.reset()
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
  u = np.asarray([r['u'] for r in rows], bool)
  s = np.asarray([r['success'] for r in rows], bool)
  steps = np.asarray([r['steps'] for r in rows], float)
  g = np.where(s, GAMMA ** steps, 0.0)

  def _m(x, nd=4):
    return round(float(np.mean(x)), nd) if len(x) else None

  return {'n': n, 'P_success': _m(s),
          'P_success_ci95': wilson(int(s.sum()), n),
          'P_failure': _m([r['failure'] for r in rows]),
          'P_active_drawn': _m(u),
          'P_success_given_clear': _m(s[~u]),
          'P_success_given_active': _m(s[u]),
          'mean_steps_given_success': _m(steps[s], 1),
          'discounted_mean': _m(g),
          'discounted_mean_given_success': _m(GAMMA ** steps[s]),
          'discounted_mean_given_clear': _m(g[~u]),
          'discounted_mean_given_active': _m(g[u])}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=['tr', 'br'], required=True)
  ap.add_argument('--n-do', type=int, default=250)
  ap.add_argument('--seed', type=int, default=808)
  ap.add_argument('--sidecar', default=None)
  ap.add_argument('--out-dir', default=None)
  args = ap.parse_args()
  name = NAME_TMPL.format(variant=args.variant)
  out_dir = args.out_dir or os.path.join(OUT_ROOT, args.variant)
  side = args.sidecar or os.path.join(OUT_ROOT, args.variant, 'dataset',
                                      f'{name}_sidecar.npz')
  os.makedirs(out_dir, exist_ok=True)

  obs = observational(side)
  print('observational:', json.dumps(obs, indent=2), flush=True)
  do_sc = summarize_do(do_route(args.variant, 'shortcut', args.n_do,
                                args.seed))
  do_dt = summarize_do(do_route(args.variant, 'detour', args.n_do,
                                args.seed + 1))
  gap = round(obs['P_success_given_shortcut_observed']
              - do_sc['P_success'], 4)

  p = TT.P_ACTIVE

  def _compose(clear_val, active_val):
    #: oracle = shortcut on clear, detour on active, at nominal p_active.
    if clear_val is None or active_val is None:
      return None
    return round(float((1 - p) * clear_val + p * active_val), 4)

  discounted = {
      'gamma': GAMMA,
      'definition': '0.99**steps if success else 0.0; the headline mean is '
                    'over ALL episodes (the pre-registered-reference '
                    'convention); *_given_success averages successes only',
      'observational': observational_discounted(side),
      'do_shortcut': {k: do_sc[k] for k in
                      ('discounted_mean', 'discounted_mean_given_success')},
      'do_detour': {k: do_dt[k] for k in
                    ('discounted_mean', 'discounted_mean_given_success')},
      'oracle_composition': {
          'p_active': p,
          'formula': '(1-p)*E[.|do(shortcut), clear] '
                     '+ p*E[.|do(detour), active]',
          'sparse_success': _compose(do_sc['P_success_given_clear'],
                                     do_dt['P_success_given_active']),
          'discounted': _compose(do_sc['discounted_mean_given_clear'],
                                 do_dt['discounted_mean_given_active'])},
      'reference': {'sparse_success': SPARSE_REFS,
                    'discounted_gamma_0.99': DISCOUNTED_REFS[args.variant]},
  }

  rep = {'variant': args.variant, 'env': TT.env_name(args.variant),
         'observational': obs,
         'do_shortcut': do_sc, 'do_detour': do_dt,
         'gap_shortcut_observational_minus_interventional': gap,
         'verdict': ('CONFOUNDING PRESENT' if gap >= GAP_THRESHOLD
                     else 'GAP TOO WEAK -- benchmark needs tuning'),
         'discounted': discounted}
  print(json.dumps({k: v for k, v in rep.items()
                    if k != 'observational'}, indent=2), flush=True)
  with open(os.path.join(out_dir, 'causal_audit.json'), 'w') as f:
    json.dump(rep, f, indent=2)
  print(f'-> {out_dir}/causal_audit.json', flush=True)


if __name__ == '__main__':
  main()
