"""PART 5/6: frozen N-episode natural bank + authoritative corrected-reset
evaluation. PRIMARY metric = natural SAMPLE-POOLED success = successes / N
(NOT analytic-weighted). Per-pattern / macro / worst are diagnostics only.

Builds (or loads) one ordered bank -- each episode pinned by (env_seed,
reset-index, forced mask, base-side) which, under the corrected reset, fully
determines the episode. Saves the bank + SHA. Evaluates every listed policy on
the exact same bank, at true cap, under the corrected reset.

Usage:
  python scripts/authoritative_eval.py --naive-final <ckpt> [--naive-best <ckpt>]
      [--legacy-naive <ckpt>] --horizon 700 --n 1000 --tag p30_h700_resetfix
"""
import argparse
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
from crl import rockfall_ant as RA          # noqa: E402
import litter_pilot_common as C             # noqa: E402
import rockfall_pilot as RP                 # noqa: E402
import rockfall_v2_teacher as V2            # noqa: E402
from diagnose_naive_rockfall import build_policy      # noqa: E402
from reconcile_rockfall_eval import (draw_natural_masks, draw_sides,
                                     mask_pattern, natural_weights,
                                     PATTERNS)          # noqa: E402

SEED = 20_260_726


def build_bank(bank_seed, n, p_active, env_seed, horizon):
  masks = draw_natural_masks(bank_seed, n, p_active)
  sides = draw_sides(bank_seed + 1, n)
  bank = {'bank_seed': bank_seed, 'env_seed': env_seed, 'n': n,
          'p_active': p_active, 'horizon': horizon,
          'reset_fix_version': V2.RESET_FIX_VERSION,
          'masks': [list(m) for m in masks], 'sides': sides}
  blob = json.dumps(bank, sort_keys=True).encode()
  bank['sha256'] = hashlib.sha256(blob).hexdigest()
  return bank, masks, sides


def rollout(env, o, kind, act, walker, base_act, base_side, cap):
  """One episode at true cap under the (already corrected) reset. Records the
  full behavioural row needed for PART 6."""
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if kind == 'teacher' else [])
  true_goal = o[29:31].copy()
  handoff = False
  handoff_step = succ = dead = -1
  fell = entered = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  ys = []
  for t in range(cap):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True; handoff_step = t
    if kind == 'naive':
      a = np.asarray(act(jnp.asarray(o[None]))[0])
    elif handoff:
      oc = o.copy(); oc[29:] = 0.0; oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    elif kind == 'center':
      x_hist.append(x); yc, vc = RP.route_command('center', t, x_hist, nudge)
      a = walker(o, yc, vc)
    else:
      x_hist.append(x)
      yc, vc = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge, RP.V_SIDE)
      a = walker(o, yc, vc)
    if 2.3 <= x <= 5.7:
      ys.append(y)
    for i, (_, sx, sgn) in enumerate(RA.ROCKFALL_SITES):
      if (abs(x - sx) <= RA.TRIG_HALF_X
          and RA.TRIG_Y_BAND[0] <= sgn * y <= RA.TRIG_Y_BAND[1]):
        entered = True
    o, r, _, info = env.step(a)
    q = np.asarray(env._env.data.qpos)
    if not env.dead and (RP.torso_up_z(q) < 0.0 or float(q[2]) < 0.2):
      fell = True
    if r > 0 and succ < 0:
      succ = t
    if info['dead'] and dead < 0:
      dead = t
    if succ >= 0 or (dead >= 0 and t > dead + 5):
      break
  my = float(np.mean(ys)) if ys else 0.0
  route = 'left' if my > 0.5 else 'right' if my < -0.5 else 'center'
  return {'success': int(succ >= 0), 'succ_step': succ, 'dead': int(dead >= 0),
          'handoff_step': handoff_step, 'fell': int(fell), 'entered': int(entered),
          'route': route, 'pattern': mask_pattern(env.rockfall_mask),
          'triggered': int(any(env._triggered)), 'dropped': int(any(env._dropped)),
          'hit': int(any(env._hit))}


def leakage_and_gaming(cfg, act, env_seed, p_active, horizon, pairs=30):
  """Paired mask-flip leakage + trigger-gaming for a neural policy (corrected)."""
  penv = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=env_seed + 1),
      p_active, reset_fix=True)
  penv.max_episode_steps = horizon
  n_ok = 0
  for _ in range(pairs):
    penv.reset()
    q0 = np.asarray(penv._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(penv._env.data.qvel)[:RA.NV_ANT].copy()
    goal = penv._flatten(penv._last_obs)[29:31].copy()
    tr = {}
    for tag, mask in (('a', (0, 0, 0, 0)), ('b', (1, 1, 1, 1))):
      o = RP.set_state(penv, q0, v0, goal, mask, ('mild',) * 4)
      obs_l, drops = [], []
      for _ in range(min(300, horizon)):
        a = np.asarray(act(jnp.asarray(o[None]))[0]); obs_l.append(o.copy())
        o, _, _, info = penv.step(a); drops.append(any(info['dropped']))
      tr[tag] = (np.asarray(obs_l), drops)
    dif = np.abs(tr['a'][0] - tr['b'][0]).max(axis=1)
    div = int(np.argmax(dif > 1e-9)) if (dif > 1e-9).any() else None
    fdrop = tr['b'][1].index(True) if True in tr['b'][1] else None
    n_ok += int(div is None or (fdrop is not None and div > fdrop))
  return {'paired_maskflip_ok': n_ok, 'pairs': pairs,
          'leakage': 'none' if n_ok == pairs else 'SUSPECTED'}


def summarize(rows):
  n = len(rows)
  succ = [r for r in rows if r['success']]
  steps = [r['succ_step'] + 1 for r in succ]
  per = {}
  for p in PATTERNS:
    v = [r['success'] for r in rows if r['pattern'] == p]
    if v:
      per[p] = round(float(np.mean(v)), 3)
  fail = [r for r in rows if not r['success']]
  return {
      'pooled_success': round(len(succ) / n, 4), 'n_success': len(succ),
      'median_steps': int(np.median(steps)) if steps else None,
      'mean_steps': round(float(np.mean(steps)), 1) if steps else None,
      'timeout_rate': round(sum(1 for r in fail if not r['dead']) / n, 3),
      'fall_rate': round(float(np.mean([r['fell'] for r in rows])), 3),
      'route_center': round(float(np.mean([r['route'] == 'center' for r in rows])), 3),
      'route_left': round(float(np.mean([r['route'] == 'left' for r in rows])), 3),
      'route_right': round(float(np.mean([r['route'] == 'right' for r in rows])), 3),
      'hazard_exposure': round(float(np.mean([r['entered'] for r in rows])), 3),
      'trigger_rate': round(float(np.mean([r['triggered'] for r in rows])), 3),
      'drop_rate': round(float(np.mean([r['dropped'] for r in rows])), 3),
      'impact_rate': round(float(np.mean([r['hit'] for r in rows])), 3),
      'per_pattern': per,
      'balanced_macro': round(float(np.mean(list(per.values()))), 4) if per else None,
      'worst_mask': min(per.values()) if per else None}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--naive-final', required=True)
  ap.add_argument('--naive-best', default=None)
  ap.add_argument('--legacy-naive', default=None)
  ap.add_argument('--horizon', type=int, default=700)
  ap.add_argument('--p-active', type=float, default=0.30)
  ap.add_argument('--n', type=int, default=1000)
  ap.add_argument('--bank-seed', type=int, default=SEED + 5000)
  ap.add_argument('--env-seed', type=int, default=SEED)
  ap.add_argument('--tag', default='p30_h700_resetfix')
  ap.add_argument('--out-dir', default='artifacts/rockfall_reconcile/p30_h700_resetfix')
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  cap = args.horizon

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''; cfg.eval_goal_mode = 'd4rl'
  bank, masks, sides = build_bank(args.bank_seed, args.n, args.p_active,
                                  args.env_seed, args.horizon)
  json.dump(bank, open(os.path.join(args.out_dir, f'bank_{args.tag}.json'), 'w'))
  print(f'bank {args.tag}: N={args.n} sha={bank["sha256"][:16]} '
        f'horizon={cap} reset_fix={V2.RESET_FIX_VERSION}', flush=True)

  def eval_policy(kind, ckpt=None):
    ccfg = cfg
    act = None
    if ckpt is not None:
      ccfg, act, _ = build_policy(ckpt)
      ccfg.offline_dataset = ''; ccfg.eval_goal_mode = 'd4rl'
    env = V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', ccfg, seed=args.env_seed),
        args.p_active, reset_fix=True)
    env.max_episode_steps = cap
    assert env._env.full_reset
    rows = [rollout(env, env.reset(mask=m), kind, act, walker, base_act,
                    sides[i], cap) for i, m in enumerate(masks)]
    s = summarize(rows)
    if kind == 'naive':
      s.update(leakage_and_gaming(ccfg, act, args.env_seed, args.p_active, cap))
    return s

  results = {}
  results['teacher'] = eval_policy('teacher')
  print('teacher pooled', results['teacher']['pooled_success'], flush=True)
  results['center'] = eval_policy('center')
  print('center pooled', results['center']['pooled_success'], flush=True)
  results['blind'] = eval_policy('blind')
  print('blind pooled', results['blind']['pooled_success'], flush=True)
  results['naive_final'] = eval_policy('naive', args.naive_final)
  print('naive_final pooled', results['naive_final']['pooled_success'], flush=True)
  if args.naive_best:
    results['naive_best'] = eval_policy('naive', args.naive_best)
    print('naive_best pooled', results['naive_best']['pooled_success'], flush=True)
  if args.legacy_naive:
    results['legacy_naive_corrected_eval'] = eval_policy('naive', args.legacy_naive)
    print('legacy_naive(corrected eval) pooled',
          results['legacy_naive_corrected_eval']['pooled_success'], flush=True)

  out = {'provenance': {'tag': args.tag, 'bank_sha256': bank['sha256'],
                        'n': args.n, 'horizon': cap, 'p_active': args.p_active,
                        'env_seed': args.env_seed, 'bank_seed': args.bank_seed,
                        'reset_fix_version': V2.RESET_FIX_VERSION,
                        'primary_metric': 'natural_sample_pooled_success',
                        'checkpoints': {'naive_final': args.naive_final,
                                        'naive_best': args.naive_best,
                                        'legacy_naive': args.legacy_naive}},
         'results': results}
  op = os.path.join(args.out_dir, f'authoritative_{args.tag}.json')
  json.dump(out, open(op, 'w'), indent=2)
  print('wrote', op, flush=True)


if __name__ == '__main__':
  main()
