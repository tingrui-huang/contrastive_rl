"""Reconcile the H=700 naive numbers between the reconcile harness and the
horizon harness. UN-INTERLEAVED: each rollout path is a complete single-env
sequential pass (as the real harnesses run). Diagnosis only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod            # noqa: E402
import rockfall_v2_teacher as V2            # noqa: E402
from diagnose_naive_rockfall import build_policy      # noqa: E402
from reconcile_rockfall_eval import (draw_natural_masks, draw_balanced_masks,
                                     mask_pattern, natural_weights,
                                     PATTERNS)          # noqa: E402
from horizon_sweep_p30 import SEED            # noqa: E402

CKPT = 'naive_rockfall_v2_p30_s0_300k/final.pkl'
PA = 0.30
W = natural_weights(PA)


def sha(p):
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def rollout_succstep(env, act, o, cap):
  """One neural episode to <=cap; returns (succ_step, dead_step). Break on
  first success or dead+5 -- the shared semantics of BOTH harnesses."""
  succ_step = dead_step = -1
  for t in range(cap):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    if r > 0 and succ_step < 0:
      succ_step = t
    if info['dead'] and dead_step < 0:
      dead_step = t
    if succ_step >= 0 or (dead_step >= 0 and t > dead_step + 5):
      break
  return succ_step, dead_step


def per_pattern(flags, pats):
  d = {}
  for p in PATTERNS:
    v = [f for f, pt in zip(flags, pats) if pt == p]
    if v:
      d[p] = {'n': len(v), 'success': round(float(np.mean(v)), 3)}
  return d


def analytic(pp):
  return round(sum(W[p] * pp[p]['success'] for p in PATTERNS), 4)


def main():
  cfg, act, step = build_policy(CKPT)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  csha = sha(CKPT)
  print('checkpoint', CKPT, '| sha', csha, '| step', step, flush=True)
  masks = draw_natural_masks(SEED + 1, 500, PA)
  pats = [mask_pattern(m) for m in masks]
  report = {'checkpoint': CKPT, 'sha256': csha, 'step': int(step),
            'natural_weights': {p: round(W[p], 4) for p in PATTERNS}}

  for tag, es in (('horizon_env+0', SEED), ('reconcile_env+10', SEED + 10)):
    # PASS R: reconcile path (cap 700), full single-env sequential run
    envR = V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=es), PA)
    R = [rollout_succstep(envR, act, envR.reset(mask=m), 700) for m in masks]
    # PASS H: horizon path (cap 1000), full single-env sequential run
    envH = V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=es), PA)
    envH.max_episode_steps = 1000
    H = [rollout_succstep(envH, act, envH.reset(mask=m), 1000) for m in masks]
    sR = np.array([1 if 0 <= ss < 700 else 0 for ss, _ in R])
    sH = np.array([1 if 0 <= ss < 700 else 0 for ss, _ in H])
    mism = [(i, R[i], H[i]) for i in range(len(masks)) if sR[i] != sH[i]]
    dead_succ = sum(1 for ss, ds in H if ss >= 0 and ds >= 0 and ss > ds)
    ppR = per_pattern(sR, pats)
    block = {'env_seed': es, 'n': len(masks),
             'mismatch_reconcile_vs_horizon_at_700': len(mism),
             'mismatch_detail_first10': mism[:10],
             'pooled_success_700': round(float(sR.mean()), 4),
             'per_pattern': ppR,
             'analytic_from_this_sample': analytic(ppR),
             'dead_counted_as_success': int(dead_succ)}
    report[tag] = block
    print(f'[{tag}] pooled@700={block["pooled_success_700"]} '
          f'mismatch={len(mism)} dead->succ={dead_succ} '
          f'analytic(sample)={block["analytic_from_this_sample"]}', flush=True)

  # authoritative analytic from balanced K=100 (reconcile's construction)
  bal = draw_balanced_masks(SEED, 100, PA)
  envb = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED), PA)
  bs = [1 if 0 <= rollout_succstep(envb, act, envb.reset(mask=m), 700)[0] < 700
        else 0 for p, m in bal]
  bpats = [p for p, m in bal]
  ppB = per_pattern(bs, bpats)
  report['balanced_K100'] = {
      'env_seed': SEED, 'k_per_pattern': 100, 'per_pattern': ppB,
      'balanced_macro': round(float(np.mean([ppB[p]['success']
                                             for p in PATTERNS])), 4),
      'analytic_natural': analytic(ppB),
      'worst_mask': min(ppB[p]['success'] for p in PATTERNS)}
  print(f"[balanced K=100] macro={report['balanced_K100']['balanced_macro']} "
        f"analytic={report['balanced_K100']['analytic_natural']} "
        f"per_pattern={ {p: ppB[p]['success'] for p in PATTERNS} }", flush=True)

  out = 'artifacts/horizon_sweep_p30/h700_reconciliation.json'
  json.dump(report, open(out, 'w'), indent=2)
  print('\nwrote', out, flush=True)


if __name__ == '__main__':
  main()
