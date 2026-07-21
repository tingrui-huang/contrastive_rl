"""EVAL-ONLY severity sweep on the frozen rockfall env. Does NOT modify the
env, protocol or dataset: it only constructs the env with a different
`severity_probs` at evaluation time (the env already exposes it) and rolls
out fixed policies. Masks/goals are paired across policies and across
severity configs (same env seed; the mask rng is independent of the
severity rng, so only lethality changes, not which sites are active).

Policies: naive best.pkl / final.pkl (learned), sighted teacher / blind side
/ center (scripted frozen protocol). Reports success, collapse rate, and
both_sides-mask success (the confounded worst case).

Usage: python scripts/rockfall_severity_sweep.py [--n 100]
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import diagnose_naive_rockfall as D       # noqa: E402

SEED = 55_301
SEVERITIES = {'current_0.55/0.30/0.15': (0.55, 0.30, 0.15),
              'proposed_0.80/0.15/0.05': (0.80, 0.15, 0.05)}
NAIVE = {'naive_best': 'naive_rockfall_full_s0_300k/best.pkl',
         'naive_final': 'naive_rockfall_full_s0_300k/final.pkl'}
OUT = 'artifacts/naive_rockfall_diagnosis/severity_sweep.json'


def both_sides(mask):
  return (mask[0] or mask[1]) and (mask[2] or mask[3])


def summarize(rows):
  succ = np.array([r['success'] for r in rows])
  dead = np.array([bool(r['dead']) for r in rows])
  masks = [r['mask'] for r in rows]
  bs = np.array([both_sides(m) for m in masks])
  return {'n': len(rows),
          'success': round(float(succ.mean()), 3),
          'collapse_frac': round(float(dead.mean()), 3),
          'both_sides_n': int(bs.sum()),
          'both_sides_success': (round(float(succ[bs].mean()), 3)
                                 if bs.any() else None)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=100)
  args = ap.parse_args()
  os.makedirs(os.path.dirname(OUT), exist_ok=True)

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  naive_acts = {name: D.build_policy(path)[1]
                for name, path in NAIVE.items()}

  def eval_policy(kind, sev):
    env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED)
    env.severity_probs = sev
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for i in range(args.n):
      o = env.reset()
      if kind in naive_acts:
        rows.append(D.rollout(env, naive_acts[kind], o))
      elif kind == 'teacher':
        rows.append(RP.run_route(env, o, walker, base_act,
                                 RP.teacher_route(env.rockfall_mask, rng)))
      elif kind == 'blind':
        rows.append(RP.run_route(env, o, walker, base_act,
                                 'left' if i % 2 == 0 else 'right'))
      else:                                # center
        rows.append(RP.run_route(env, o, walker, base_act, 'center'))
    return summarize(rows)

  policies = ['naive_best', 'naive_final', 'teacher', 'blind', 'center']
  report = {'seed': SEED, 'n': args.n, 'results': {}}
  for sev_name, sev in SEVERITIES.items():
    report['results'][sev_name] = {}
    print(f'\n=== severity {sev_name} = {sev} ===', flush=True)
    for pol in policies:
      s = eval_policy(pol, sev)
      report['results'][sev_name][pol] = s
      print(f'  {pol:12s} success {s["success"]:.3f}  collapse '
            f'{s["collapse_frac"]:.3f}  both_sides '
            f'{s["both_sides_success"]} (n={s["both_sides_n"]})', flush=True)
  json.dump(report, open(OUT, 'w'), indent=2)
  print('\nsaved', OUT, flush=True)

  # delta table
  print('\n--- success delta (proposed - current) ---', flush=True)
  cur = report['results']['current_0.55/0.30/0.15']
  pro = report['results']['proposed_0.80/0.15/0.05']
  for pol in policies:
    d = pro[pol]['success'] - cur[pol]['success']
    dc = pro[pol]['collapse_frac'] - cur[pol]['collapse_frac']
    print(f'  {pol:12s} success {d:+.3f}   collapse {dc:+.3f}', flush=True)


if __name__ == '__main__':
  main()
