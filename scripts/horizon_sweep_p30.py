"""Diagnosis-only HORIZON SWEEP for the p_active=0.30 local-detour benchmark.

Evaluates H in {700, 800, 900, 1000}. No frozen defaults changed; no dataset/
checkpoint/report overwritten. Horizon is an env-INSTANCE override
(env.max_episode_steps) + the rollout loop bound.

Efficiency: every controller's action depends only on (obs, t), never on the
horizon value, so ONE pass to H=1000 per episode yields all four horizons by
thresholding the first-success step. Behaviour that is fixed during the early
traversal (route/exposure/drop/leakage/trigger-gaming) is horizon-invariant and
is reused from the committed p30 diagnosis (cited in the report), not recomputed.

Part A: scripted teacher / center / blind on PAIRED natural masks+seeds.
Part B: existing p30 naive final.pkl -- EVALUATION-ONLY post-training horizon
        sensitivity (NOT a trained baseline for H>700).

Usage: python scripts/horizon_sweep_p30.py --n-nat 500 --k-bal 60
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

from crl import envs as envs_mod            # noqa: E402
from crl import rockfall_ant as RA          # noqa: E402
import litter_pilot_common as C             # noqa: E402
import rockfall_pilot as RP                 # noqa: E402
import rockfall_v2_teacher as V2            # noqa: E402
from diagnose_naive_rockfall import build_policy      # noqa: E402
from reconcile_rockfall_eval import (draw_balanced_masks, draw_natural_masks,
                                     draw_sides, mask_pattern, natural_weights,
                                     PATTERNS)          # noqa: E402

SEED = 20_260_726
HORIZONS = [700, 800, 900, 1000]
HMAX = 1000
P_ACTIVE = 0.30
STUCK_VX = RA.TRIG_MIN_VX


def scripted_rollout(env, o, walker, base_act, mode, base_side):
  """teacher/center/blind control law, verbatim, single pass to HMAX.
  Does not break on success (records first-success step)."""
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if mode == 'teacher' else [])
  true_goal = o[29:31].copy()
  handoff = False
  handoff_step = succ_step = dead_step = -1
  fell = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  snaps = {}
  for t in range(HMAX):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
      handoff_step = t
    if handoff:
      oc = o.copy(); oc[29:] = 0.0; oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    elif mode == 'center':
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
    else:  # teacher / blind: base lane (+ detour for teacher)
      x_hist.append(x)
      y_cmd, v_cmd = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                       RP.V_SIDE)
      a = walker(o, y_cmd, v_cmd)
    o, r, _, info = env.step(a)
    q = np.asarray(env._env.data.qpos)
    if not env.dead and (RP.torso_up_z(q) < 0.0 or float(q[2]) < 0.2):
      fell = True
    if r > 0 and succ_step < 0:
      succ_step = t
    if info['dead'] and dead_step < 0:
      dead_step = t
    if (t + 1) in HORIZONS:
      d = float(np.linalg.norm(np.array([float(o[0]), float(o[1])]) - true_goal))
      snaps[t + 1] = {'dist': round(d, 3),
                      'ntrig': int(sum(env._triggered)),
                      'ndrop': int(sum(env._dropped)),
                      'nhit': int(sum(env._hit))}
    if succ_step >= 0 or (dead_step >= 0 and t > dead_step + 5):
      break
  final_dist = float(np.linalg.norm(np.array([float(o[0]), float(o[1])])
                                    - true_goal))
  return {'mode': mode, 'mask': list(env.rockfall_mask),
          'pattern': mask_pattern(env.rockfall_mask),
          'succ_step': succ_step, 'dead_step': dead_step,
          'handoff_step': handoff_step, 'fell': fell,
          'final_dist': round(final_dist, 3), 'snaps': snaps,
          'n_trigger': int(sum(env._triggered)),
          'n_drop': int(sum(env._dropped)), 'n_hit': int(sum(env._hit))}


def naive_rollout(env, act, o):
  """Neural policy, single pass to HMAX; records first-success step + route."""
  succ_step = dead_step = -1
  ys_zone, entered = [], False
  dropped_any = False
  for t in range(HMAX):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    if 2.3 <= x <= 5.7:
      ys_zone.append(y)
    for i, (_, sx, sgn) in enumerate(RA.ROCKFALL_SITES):
      if (abs(x - sx) <= RA.TRIG_HALF_X
          and RA.TRIG_Y_BAND[0] <= sgn * y <= RA.TRIG_Y_BAND[1]):
        entered = True
    if any(env._dropped):
      dropped_any = True
    if r > 0 and succ_step < 0:
      succ_step = t
    if info['dead'] and dead_step < 0:
      dead_step = t
    if succ_step >= 0 or (dead_step >= 0 and t > dead_step + 5):
      break
  mean_y = float(np.mean(ys_zone)) if ys_zone else 0.0
  route = ('left' if mean_y > 0.5 else 'right' if mean_y < -0.5 else 'center')
  return {'mask': list(env.rockfall_mask),
          'pattern': mask_pattern(env.rockfall_mask),
          'succ_step': succ_step, 'dead_step': dead_step,
          'route': route, 'entered': entered, 'dropped': dropped_any}


def success_at(r, H):
  return int(0 <= r['succ_step'] < H)


def derive_per_horizon(rows, weights=None):
  """Per-H aggregate metrics for a list of episode records."""
  out = {}
  for H in HORIZONS:
    succ = [r for r in rows if success_at(r, H)]
    steps = [r['succ_step'] + 1 for r in succ]
    # failures at H = not success and not dead-before-H
    dead = [r for r in rows if not success_at(r, H)
            and 0 <= r['dead_step'] < H]
    timeout = [r for r in rows if not success_at(r, H)
               and not (0 <= r['dead_step'] < H)]
    fell_fail = [r for r in rows if not success_at(r, H) and r.get('fell')]
    posthandoff_to = [r for r in timeout
                      if 0 <= r.get('handoff_step', -1) < H]
    by700 = [r for r in rows if 0 <= r['succ_step'] < 700]
    rescued = [r for r in rows if 700 <= r['succ_step'] < H]
    # final dist for timeouts, from the H snapshot (state at t=H-1)
    fdist = [r['snaps'][H]['dist'] for r in timeout
             if 'snaps' in r and H in r.get('snaps', {})]
    row = {'natural_success': round(len(succ) / len(rows), 4),
           'mean_steps_to_success': (round(float(np.mean(steps)), 1)
                                     if steps else None),
           'median_steps_to_success': (int(np.median(steps))
                                        if steps else None),
           'success_by_700': round(len(by700) / len(rows), 4),
           'rescued_after_700': round(len(rescued) / len(rows), 4),
           'timeout_count': len(timeout),
           'physical_fall_count': len(fell_fail),
           'post_handoff_timeout_count': len(posthandoff_to),
           'dead_count': len(dead),
           'final_dist_timeout_median': (round(float(np.median(fdist)), 3)
                                         if fdist else None)}
    # balanced macro + worst-mask over patterns (if weights given -> nat too)
    per = {}
    for p in PATTERNS:
      sub = [r for r in rows if r['pattern'] == p]
      if sub:
        per[p] = round(sum(success_at(r, H) for r in sub) / len(sub), 3)
    if per:
      row['per_pattern_success'] = per
      row['balanced_macro'] = round(float(np.mean(list(per.values()))), 4)
      row['worst_mask'] = min(per.values())
      if weights:
        row['natural_analytic'] = round(
            sum(weights[p] * per.get(p, 0) for p in PATTERNS), 4)
    out[str(H)] = row
  # rockfall interaction (horizon-invariant in practice; report final + max step)
  out['rockfall_totals'] = {
      'episodes_with_trigger': int(sum(r.get('n_trigger', 0) > 0 for r in rows)),
      'episodes_with_drop': int(sum(r.get('n_drop', 0) > 0 for r in rows)),
      'episodes_with_hit': int(sum(r.get('n_hit', 0) > 0 for r in rows))}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n-nat', type=int, default=500)
  ap.add_argument('--k-bal', type=int, default=60)
  ap.add_argument('--naive-ckpt',
                  default='naive_rockfall_v2_p30_s0_300k/final.pkl')
  ap.add_argument('--out', default='artifacts/horizon_sweep_p30/sweep.json')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  weights = natural_weights(P_ACTIVE)

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  nat_masks = draw_natural_masks(SEED + 1, args.n_nat, P_ACTIVE)
  nat_sides = draw_sides(SEED + 2, args.n_nat)

  def fresh(seed):
    e = V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed),
        P_ACTIVE)
    e.max_episode_steps = HMAX
    return e

  # ---- Part A: scripted teacher / center / blind (paired) ----
  partA = {}
  for mode in ('teacher', 'center', 'blind'):
    env = fresh(SEED)
    rows = []
    for i, m in enumerate(nat_masks):
      o = env.reset(mask=m)
      rows.append(scripted_rollout(env, o, walker, base_act, mode,
                                   nat_sides[i]))
    partA[mode] = derive_per_horizon(rows, weights)
    s = partA[mode]
    print(f'[A] {mode}: ' + ' '.join(
        f"H{H}={s[str(H)]['natural_success']}" for H in HORIZONS)
        + f" | rockfall_hit_eps={s['rockfall_totals']['episodes_with_hit']}",
        flush=True)

  # ---- Part B: existing naive final.pkl (eval-only horizon sensitivity) ----
  cfgn, act, step = build_policy(args.naive_ckpt)
  cfgn.offline_dataset = ''
  cfgn.eval_goal_mode = 'd4rl'
  envn = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfgn, seed=SEED),
      P_ACTIVE)
  envn.max_episode_steps = HMAX
  nrows = []
  for m in nat_masks:
    o = envn.reset(mask=m)
    nrows.append(naive_rollout(envn, act, o))
  # balanced block for macro / worst-mask
  bal = draw_balanced_masks(SEED, args.k_bal, P_ACTIVE)
  envb = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfgn, seed=SEED + 7),
      P_ACTIVE)
  envb.max_episode_steps = HMAX
  brows = []
  for p, m in bal:
    o = envb.reset(mask=m)
    brows.append(naive_rollout(envb, act, o))
  partB = {}
  for H in HORIZONS:
    nsucc = [r for r in nrows if success_at(r, H)]
    steps = [r['succ_step'] + 1 for r in nsucc]
    rescued = [r for r in nrows if 700 <= r['succ_step'] < H]
    # balanced per-pattern
    per = {}
    for p in PATTERNS:
      sub = [r for r in brows if r['pattern'] == p]
      if sub:
        per[p] = round(sum(success_at(r, H) for r in sub) / len(sub), 3)
    routes = {k: round(np.mean([r['route'] == k for r in nrows]), 3)
              for k in ('center', 'left', 'right')}
    partB[str(H)] = {
        'natural_success': round(len(nsucc) / len(nrows), 4),
        'balanced_macro': round(float(np.mean(list(per.values()))), 4),
        'worst_mask': min(per.values()),
        'per_pattern_success': per,
        'mean_steps_to_success': (round(float(np.mean(steps)), 1)
                                  if steps else None),
        'median_steps_to_success': int(np.median(steps)) if steps else None,
        'rescued_after_700': round(len(rescued) / len(nrows), 4),
        'route_usage': routes,
        'hazard_exposure': round(float(np.mean([r['entered']
                                 for r in nrows])), 3),
        'drop_rate': round(float(np.mean([r['dropped'] for r in nrows])), 3)}
    print(f"[B] naive H{H}: nat={partB[str(H)]['natural_success']} "
          f"macro={partB[str(H)]['balanced_macro']} "
          f"worst={partB[str(H)]['worst_mask']} "
          f"rescued700={partB[str(H)]['rescued_after_700']}", flush=True)

  out = {
      'provenance': {'seed': SEED, 'p_active': P_ACTIVE, 'horizons': HORIZONS,
                     'severity': list(V2.SEVERITY_V2),
                     'n_natural': args.n_nat, 'k_balanced': args.k_bal,
                     'naive_ckpt': args.naive_ckpt, 'naive_step': int(step),
                     'natural_weights': {p: round(weights[p], 4)
                                         for p in PATTERNS},
                     'note_partB': 'EVALUATION-ONLY post-training horizon '
                     'sensitivity of the H=700-trained naive final.pkl; NOT a '
                     'trained baseline for H>700. Behavioural metrics '
                     '(leakage/trigger-gaming) are horizon-invariant; see the '
                     'committed v2_p30_final diagnosis.'},
      'partA_scripted': partA,
      'partB_naive_eval_only': partB}
  json.dump(out, open(args.out, 'w'), indent=2)
  print('\nwrote', args.out, flush=True)


if __name__ == '__main__':
  main()
