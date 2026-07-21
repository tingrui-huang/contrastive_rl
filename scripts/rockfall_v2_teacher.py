"""Rockfall v2 -- LOCAL-DETOUR privileged teacher (the intended primary spec).

Same FROZEN env (crl/rockfall_ant.py, offline_ant_umaze_rockfall) as
global-route v1; only the TEACHER control law changes. Confounding pathway is
now LOCAL: the base side lane is chosen mask-INDEPENDENTLY (balanced), and the
privileged mask is used only to make a local inward detour around each ACTIVE
site on that lane, returning to the base lane afterwards.

  sighted (85%): balanced base side + local detour at active sites on it;
  blind   (5%) : same base side, NO detours (walks straight into active sites);
  center  (10%): center route throughout (mask-invariant).

The env never changes, so paired hiddenness is inherited from v1 (checked with
a mask-INDEPENDENT policy). The mask enters only through the teacher's action
(U -> local detour -> A), which is the intended confounder.

This module is analysis/qualification + the control law reused by the v2
collector. It does NOT modify any frozen artifact.
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
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402

#: mask bit index per site name (env ROCKFALL_SITES order)
BIT = {'left_1': 0, 'left_2': 1, 'right_1': 2, 'right_2': 3}

#: v2 PROTOCOL VERSION. v2.1 adopts the STRONGER severity 0.80/0.15/0.05
#: (severe/impaired/mild). This is applied to the env INSTANCE at collection/
#: eval time (env.severity_probs = SEVERITY_V2); the frozen env module default
#: (crl/rockfall_ant.py SEVERITY_PROBS = 0.55/0.30/0.15) and the preserved
#: global-route v1 are NOT modified.
V2_PROTOCOL_VERSION = 'local_detour_v2.1_sev0.80'
SEVERITY_V2 = (0.80, 0.15, 0.05)


def apply_v2_config(env):
  """Configure an env instance for the v2.1 protocol (severity only).
  Returns env for chaining. Does not touch the frozen module defaults."""
  env.severity_probs = SEVERITY_V2
  return env

#: local-detour geometry. The trigger band floor is |y| = TRIG_Y_BAND[0] = 1.0;
#: the detour must hold |y| < 1.0 (comfortably) across the site's trigger
#: window |x - sx| <= TRIG_HALF_X (0.6). It dips into the mud taper (|y| < 1.0)
#: -> a real, mask-independent-looking slowdown cost for avoidance.
DETOUR_Y = 0.72        #: commanded |y| during a detour (walker tracks ~0.8-0.9)
DETOUR_PRE = 1.1       #: begin moving inward this far (in x) before the site
DETOUR_POST = 0.8      #: hold inward this far past the site, then return
BASE_LANE = RP.LANE    #: base side lane |y| (1.1)


def active_site_windows(base_sgn, mask):
  """[(x0, x1)] detour windows for ACTIVE sites on the base side lane."""
  wins = []
  for nm, sx, sgn in RA.ROCKFALL_SITES:
    if sgn == base_sgn and mask[BIT[nm]]:
      wins.append((sx - DETOUR_PRE, sx + DETOUR_POST))
  return wins


def detour_command(base_sgn, wins, x, t, x_hist, nudge, v_side):
  """(y_ref, v_ref): base lane, dipping inward inside an active-site window.
  Outside windows a base-lane stall-unstick (outward-only) is kept as a
  wedging fallback, identical to the v1 side protocol."""
  in_win = any(x0 <= x <= x1 for x0, x1 in wins)
  if in_win:
    return base_sgn * DETOUR_Y, v_side        # local inward detour
  y_base = base_sgn * BASE_LANE
  y_cmd = y_base
  if t < nudge['until']:
    y_cmd = base_sgn * RP.SIDE_NUDGE_OUT if nudge['sign'] > 0 else y_base
  elif (len(x_hist) > RP.STALL_WINDOW
        and x_hist[-1] - x_hist[-RP.STALL_WINDOW] < RP.STALL_MIN_DX):
    nudge['until'] = t + RP.NUDGE_STEPS
    nudge['sign'] = -nudge['sign']
    x_hist.clear()
    y_cmd = base_sgn * RP.SIDE_NUDGE_OUT if nudge['sign'] > 0 else y_base
  return y_cmd, v_side


def run_sighted(env, o, walker, base_act, base_side, use_detour=True,
                record=False):
  """One sighted (or blind) side episode. base_side in {'left','right'};
  use_detour=False is the blind policy (same base lane, no detours)."""
  base_sgn = 1.0 if base_side == 'left' else -1.0
  mask = env.rockfall_mask
  wins = active_site_windows(base_sgn, mask) if use_detour else []
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  hit, dead_at, first_hit_t = 0.0, -1, -1
  ys, xs = [], []
  for t in range(env.max_episode_steps):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                    RP.V_SIDE)
      a = walker(o, y_cmd, v_cmd)
      if record and 1.5 < x < 5.6:
        ys.append(y)
        xs.append(x)
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    if info['rock_ant_contact'] and first_hit_t < 0:
      first_hit_t = t
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
      break
  out = {'success': hit, 'dead': dead_at >= 0, 'steps': t + 1,
         'base_side': base_side, 'mask': list(mask),
         'triggered': list(env._triggered), 'dropped': list(env._dropped),
         'hit': list(env._hit), 'first_hit_t': first_hit_t}
  if record:
    out['xs'], out['ys'] = xs, ys
  return out


def min_absy_near(xs, ys, sx, lo=None, hi=None):
  xs, ys = np.asarray(xs), np.asarray(ys)
  lo = sx - DETOUR_PRE if lo is None else lo
  hi = sx + DETOUR_POST if hi is None else hi
  sel = (xs >= lo) & (xs <= hi)
  return float(np.min(np.abs(ys[sel]))) if sel.any() else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--eps', type=int, default=60)
  ap.add_argument('--seeds', type=int, nargs='+', default=[41_001, 42_001])
  ap.add_argument('--out', default='artifacts/rockfall_v2/qual_report.json')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  # ---- 3. base-side balance + mask independence ----
  balance = []
  for seed in args.seeds:
    env = apply_v2_config(envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed))
    side_rng = np.random.default_rng(seed + 999)   # INDEPENDENT of mask rng
    sides, left_active = [], []
    for _ in range(400):
      env.reset()
      sides.append(1 if side_rng.random() < 0.5 else 0)   # base side draw
      left_active.append(int(bool(env.rockfall_mask[0] or env.rockfall_mask[1])))
    sides, left_active = np.array(sides), np.array(left_active)
    corr = float(np.corrcoef(sides, left_active)[0, 1])
    balance.append({'seed': seed, 'frac_base_left': float(sides.mean()),
                    'corr_side_vs_leftactive': round(corr, 3)})
  print('base-side balance:', balance, flush=True)

  # ---- 1+2. detour verification (sighted, base side + detour) ----
  # active/inactive: MIN |y| in the site window (dip depth).
  # recover: MEAN |y| in the clean post-window segment (does it climb back).
  det = {'active': [], 'inactive': [], 'recover': []}
  succ = {'sighted': [], 'blind': [], 'center': []}
  detour_avoid = {'active_triggered': 0, 'active_total': 0}
  for seed in args.seeds:
    env = apply_v2_config(envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed + 3))
    side_rng = np.random.default_rng(seed + 999)
    for i in range(args.eps):
      base = 'left' if side_rng.random() < 0.5 else 'right'
      base_sgn = 1.0 if base == 'left' else -1.0
      o = env.reset()
      r = run_sighted(env, o, walker, base_act, base, use_detour=True,
                      record=True)
      succ['sighted'].append(r['success'])
      xs, ys = r['xs'], r['ys']
      if not xs:
        continue
      base_sites = [(nm, sx) for nm, sx, sgn in RA.ROCKFALL_SITES
                    if sgn == base_sgn]
      active_x = [sx for nm, sx in base_sites if r['mask'][BIT[nm]]]
      for nm, sx in base_sites:
        a = min_absy_near(xs, ys, sx)
        if a is None:
          continue
        if r['mask'][BIT[nm]]:
          det['active'].append(a)
          detour_avoid['active_total'] += 1
          if r['triggered'][BIT[nm]]:
            detour_avoid['active_triggered'] += 1
          # recovery: mean |y| just past the window, skipping if the next
          # active site's detour window overlaps (avoid contamination).
          rlo, rhi = sx + DETOUR_POST + 0.3, sx + DETOUR_POST + 1.1
          if not any(ax != sx and (ax - DETOUR_PRE) < rhi
                     and (ax + DETOUR_POST) > rlo for ax in active_x):
            seg = (np.asarray(xs) >= rlo) & (np.asarray(xs) <= rhi)
            if seg.any():
              det['recover'].append(float(np.mean(np.abs(
                  np.asarray(ys)[seg]))))
        else:
          det['inactive'].append(a)

  # blind + center success (same base-side draw stream)
  for seed in args.seeds:
    envb = apply_v2_config(envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed + 5))
    side_rng = np.random.default_rng(seed + 999)
    for i in range(args.eps):
      base = 'left' if side_rng.random() < 0.5 else 'right'
      o = envb.reset()
      succ['blind'].append(run_sighted(envb, o, walker, base_act, base,
                                       use_detour=False)['success'])
    envc = apply_v2_config(envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed + 7))
    for i in range(args.eps):
      o = envc.reset()
      succ['center'].append(RP.run_route(envc, o, walker, base_act,
                                         'center')['success'])

  def ms(a):
    a = np.array(a)
    return None if not len(a) else round(float(a.mean()), 3)
  active_trig_rate = (detour_avoid['active_triggered']
                      / max(detour_avoid['active_total'], 1))
  am, im, rm = ms(det['active']), ms(det['inactive']), ms(det['recover'])
  report = {
      'detour_params': {'detour_y': DETOUR_Y, 'pre': DETOUR_PRE,
                        'post': DETOUR_POST, 'base_lane': BASE_LANE},
      'base_side_balance': balance,
      'detour_min_absy': {
          'active_mean': am, 'inactive_mean': im,
          'active_n': len(det['active']), 'inactive_n': len(det['inactive']),
          'recover_mean': rm, 'recover_n': len(det['recover']),
          'separation_inactive_minus_active':
              (round(im - am, 3) if (am is not None and im is not None)
               else None)},
      'active_site_trigger_rate_under_detour': round(active_trig_rate, 3),
      'active_total': detour_avoid['active_total'],
      'success': {k: ms(v) for k, v in succ.items()},
      'checks': {}}
  c = report['checks']
  bal = report['base_side_balance']
  c['base_side_balanced'] = all(0.42 <= b['frac_base_left'] <= 0.58
                                for b in bal)
  c['base_side_mask_independent'] = all(abs(b['corr_side_vs_leftactive']) < 0.1
                                        for b in bal)
  # active sites dip well inward (below the 1.0 band floor and the wobble
  # baseline); inactive sites stay near the base-lane wobble; the detour is
  # site-SELECTIVE (clear separation); and |y| recovers past the window.
  c['active_detour_dips_inward'] = am is not None and am < 0.80
  c['inactive_stays_near_base'] = im is not None and im > 0.82
  c['detour_site_selective'] = (am is not None and im is not None
                                and im - am >= 0.10)
  c['returns_to_base_lane'] = (rm is not None and am is not None
                               and rm >= am + 0.12)
  c['detour_avoids_trigger'] = active_trig_rate <= 0.15
  c['sighted_gt_blind_plus_0.10'] = (report['success']['sighted']
                                     >= report['success']['blind'] + 0.10)
  json.dump(report, open(args.out, 'w'), indent=2)
  print(json.dumps(report, indent=2), flush=True)
  print('ALL CHECKS PASS' if all(c.values()) else 'CHECKS FAILED:',
        {k: v for k, v in c.items() if not v}, flush=True)


if __name__ == '__main__':
  main()
