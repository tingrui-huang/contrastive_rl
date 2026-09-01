"""Lateral profile y(x) on the first pass: the learner vs the teacher that
collected D_clean. Analysis only.

The teacher's detour signature is local and conditional: inside an ARMED site's
x-window it holds |y| below the trigger band floor (1.0) and comes back out
afterwards; at an unarmed window it stays on the ordinary side lane inside the
band. If the learner had picked that up, its profile would show the same dip at
the same x, and only when the site is armed.

Teacher profiles come from the pilot sidecar (the real collected trajectories,
no simulation). Learner profiles are rolled fresh under the same protocol.

Usage: python scripts/probe_lane_profile.py [--n 120]
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
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from leak_probe_clean_ant import build    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
SIDECAR = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot_sidecar.npz')
SEED = 90_517
BINS = np.arange(1.6, 6.01, 0.2)


def profile(xs, ys, bins=BINS):
  """Mean y per x-bin; NaN where the trajectory never visited the bin."""
  out = np.full(len(bins) - 1, np.nan)
  xs, ys = np.asarray(xs), np.asarray(ys)
  for i, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
    m = (xs >= a) & (xs < b)
    if m.any():
      out[i] = np.mean(ys[m])
  return out


def teacher_profiles():
  """Per-episode first-pass profiles from the collected pilot, tagged with the
  base lane and which sites on that lane were armed."""
  s = np.load(SIDECAR, allow_pickle=True)
  X, Y, H = s['step_torso_x'], s['step_torso_y'], s['step_handoff']
  route, dead = np.asarray(s['route']), np.asarray(s['dead'], bool)
  mask = np.asarray(s['rockfall_mask'])
  rows = []
  for i in range(len(route)):
    if dead[i] or route[i] == 'center':
      continue
    pre = (H[i] == 0) & np.isfinite(X[i])
    rows.append({'lane': str(route[i]),
                 'mask': [int(v) for v in mask[i]],
                 'prof': profile(X[i][pre], Y[i][pre])})
  return rows


def learner_profiles(env, act, n, horizon):
  rows = []
  for k in range(n):
    o = env.reset()
    mask = [int(v) for v in env.rockfall_mask]
    xs, ys, passed = [], [], False
    hit, dead_at = 0.0, -1
    for t in range(horizon):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, _, info = env.step(a)
      x, y = float(o[0]), float(o[1])
      if not passed:
        xs.append(x)
        ys.append(y)
      if x >= RP.HANDOFF_X or y >= 2.0:
        passed = True
      hit = max(hit, float(r))
      if info['dead'] and dead_at < 0:
        dead_at = t
      if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
        break
    p = profile(xs, ys)
    zone = [v for v, a_ in zip(p, BINS[:-1]) if 2.3 <= a_ <= 5.7
            and np.isfinite(v)]
    m = float(np.mean(zone)) if zone else 0.0
    rows.append({'lane': ('left' if m > 0.5 else 'right' if m < -0.5
                          else 'center'),
                 'mask': mask, 'prof': p, 'dead': dead_at >= 0})
    if (k + 1) % 25 == 0:
      print(f'  {k + 1}/{n} learner episodes', flush=True)
  return rows


def agg(rows, lane, site_idx, armed):
  """Mean profile over episodes on `lane` whose site `site_idx` is armed
  (or not). Returns (mean profile, n)."""
  sel = [r['prof'] for r in rows
         if r['lane'] == lane and bool(r['mask'][site_idx]) == armed]
  if not sel:
    return np.full(len(BINS) - 1, np.nan), 0
  with np.errstate(invalid='ignore'):
    return np.nanmean(np.stack(sel), axis=0), len(sel)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--n', type=int, default=120)
  ap.add_argument('--horizon', type=int, default=800)
  ap.add_argument('--p-active', type=float, default=0.30)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = float(args.p_active)
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.rockfall_reset_fix = True
  act, step = build(cfg, args.ckpt)
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)

  T = teacher_profiles()
  print(f'teacher side-lane episodes from the pilot: {len(T)}', flush=True)
  L = learner_profiles(env, act, args.n, args.horizon)

  out = {'bins': BINS.tolist(), 'sites': [[nm, sx, sgn] for nm, sx, sgn
                                          in RA.ROCKFALL_SITES],
         'band_floor': RA.TRIG_Y_BAND[0], 'ckpt': args.ckpt, 'step': step,
         'teacher': {}, 'learner': {}}
  for lane, idxs in (('left', (0, 1)), ('right', (2, 3))):
    for i in idxs:
      nm = RA.ROCKFALL_SITES[i][0]
      for who, rows in (('teacher', T), ('learner', L)):
        for armed in (True, False):
          p, n = agg(rows, lane, i, armed)
          out[who][f'{lane}_{nm}_{"armed" if armed else "clear"}'] = {
              'n': n, 'profile': [None if not np.isfinite(v) else round(float(v), 4)
                                  for v in p]}
  p = os.path.join(args.out_dir, 'lane_profiles.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)

  print('\nmean |y| inside each site window (first pass):')
  print(f'{"":28s} {"armed":>8s} {"clear":>8s}   n_armed/n_clear')
  for lane, idxs in (('left', (0, 1)), ('right', (2, 3))):
    for i in idxs:
      nm, sx, sgn = RA.ROCKFALL_SITES[i]
      w = (BINS[:-1] >= sx - RA.TRIG_HALF_X) & (BINS[:-1] <= sx +
                                                RA.TRIG_HALF_X)
      for who, rows in (('teacher', T), ('learner', L)):
        pa, na = agg(rows, lane, i, True)
        pc, nc = agg(rows, lane, i, False)
        va = np.nanmean(np.abs(pa[w])) if np.isfinite(pa[w]).any() else np.nan
        vc = np.nanmean(np.abs(pc[w])) if np.isfinite(pc[w]).any() else np.nan
        print(f'  {who:8s} {lane:5s} {nm:8s} {va:8.3f} {vc:8.3f}   {na}/{nc}')
  print('\n->', p, flush=True)


if __name__ == '__main__':
  main()
