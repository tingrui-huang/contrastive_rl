"""Can the learner EXECUTE a lane it never chooses? Analysis only.

"The learner has no notion of lanes" and "the learner never selects the right
lane" are different claims, and the lane-position histogram cannot separate
them. This probe does.

Take teacher states from the middle of each lane (the ant already walking at
y ~ +1.0 / 0.0 / -1.0, well past the entrance), hand control to the learner
from that exact state, and watch what it does for the rest of the first pass:

  HOLDS the lane      -> the lane IS in the learner's repertoire; the smeared
                         start-lane histogram is a SELECTION failure at the
                         entrance, not a missing behaviour.
  DRIFTS to the middle-> the learner has no such behaviour to fall back on;
                         the middle is where it goes from anywhere.

The teacher's own continuation from the same states is the reference: by
construction it holds its lane, so the drift it shows is the measurement floor.

Usage: python scripts/probe_lane_holding.py [--per-lane 40]
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
from leak_probe_clean_ant import build, set_state, mannwhitney  # noqa: E402
from probe_center_route_cause import qpos_qvel_from_obs  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
CLEAN = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
         'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
SIDECAR = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot_sidecar.npz')
SEED = 41_887
#: hand over here: the ant is committed to its lane and the hazard corridor
#: (sites at x = 3.0 .. 4.9) is still entirely ahead of it.
HANDOVER_X = 2.2
#: score the lane over the site stretch that follows the handover.
SCORE_X = (3.0, 5.5)


def run_from(env, act, o, horizon):
  """Learner takes over. Returns the lane trace over the scoring stretch."""
  ys, passed = [], False
  y0 = float(o[1])
  hit, dead_at = 0.0, -1
  for t in range(horizon):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    if not passed and SCORE_X[0] <= x <= SCORE_X[1]:
      ys.append(y)
    if x >= RP.HANDOFF_X or y >= 2.0:
      passed = True
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
      break
  return {'y_start': round(y0, 3),
          'y_lane': (round(float(np.mean(ys)), 3) if ys else None),
          'n_scored': len(ys), 'success': float(hit > 0),
          'dead': dead_at >= 0, 'steps': int(t + 1)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--per-lane', type=int, default=40)
  ap.add_argument('--horizon', type=int, default=800)
  ap.add_argument('--p-active', type=float, default=0.30)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  d = np.load(CLEAN, allow_pickle=True)
  s = np.load(SIDECAR, allow_pickle=True)
  dead = np.asarray(s['dead'], bool)
  ci = np.where(~dead)[0]
  route = np.asarray(s['route'])[ci]
  masks = np.asarray(s['rockfall_mask'])[ci]
  sevs = np.asarray(s['severity'])[ci]
  X = np.asarray(s['step_torso_x'])[ci]
  Y = np.asarray(s['step_torso_y'])[ci]
  H = np.asarray(s['step_handoff'])[ci]
  obs = d['obs']

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
  print(f'ckpt {args.ckpt} @ step {step} | handover at x={HANDOVER_X}, '
        f'lane scored over x in {SCORE_X}', flush=True)

  out = {'ckpt': args.ckpt, 'step': step, 'handover_x': HANDOVER_X,
         'score_x': list(SCORE_X), 'lanes': {}}
  for lane in ('left', 'center', 'right'):
    eps = [e for e in range(len(obs)) if route[e] == lane]
    rows, teacher_ref = [], []
    for e in eps:
      # the teacher's own step where it first passes the handover line
      pre = (H[e] == 0) & np.isfinite(X[e])
      idx = np.where(pre & (X[e] >= HANDOVER_X))[0]
      if len(idx) == 0:
        continue
      t0 = int(idx[0])
      if t0 + 1 >= obs.shape[1]:
        continue
      # teacher reference: its own lane over the scoring stretch
      m = pre & (X[e] >= SCORE_X[0]) & (X[e] <= SCORE_X[1])
      if m.any():
        teacher_ref.append(float(np.nanmean(Y[e][m])))
      o0 = obs[e, t0]
      qpos, qvel = qpos_qvel_from_obs(o0)
      goal = o0[29:31].copy()
      o = set_state(env, qpos, qvel, goal,
                    tuple(int(b) for b in masks[e]),
                    tuple(str(x) for x in sevs[e]))
      r = run_from(env, act, o, args.horizon)
      r['ep'] = int(ci[e])
      rows.append(r)
      if len(rows) >= args.per_lane:
        break
    ys = [r['y_lane'] for r in rows if r['y_lane'] is not None]
    y0 = [r['y_start'] for r in rows if r['y_lane'] is not None]
    out['lanes'][lane] = {
        'n': len(rows), 'n_scored': len(ys),
        'teacher_lane_y': (round(float(np.mean(teacher_ref)), 3)
                           if teacher_ref else None),
        'handover_y': round(float(np.mean(y0)), 3) if y0 else None,
        'learner_lane_y': round(float(np.mean(ys)), 3) if ys else None,
        'learner_lane_y_sd': round(float(np.std(ys)), 3) if ys else None,
        'drift_toward_middle': (round(float(np.mean(np.abs(y0))
                                            - np.mean(np.abs(ys))), 3)
                                if ys else None),
        'held_frac': (round(float(np.mean([abs(v) > 0.5 for v in ys])), 3)
                      if ys else None),
        'success': round(float(np.mean([r['success'] for r in rows])), 3),
        'dead': round(float(np.mean([r['dead'] for r in rows])), 3),
        'episodes': rows}
    L = out['lanes'][lane]
    print(f"  {lane:6s} n={L['n']:3d} | handover y={L['handover_y']} -> "
          f"learner lane y={L['learner_lane_y']} (sd {L['learner_lane_y_sd']}) "
          f"| teacher {L['teacher_lane_y']} | still off-center "
          f"{L['held_frac']} | success {L['success']} dead {L['dead']}",
          flush=True)

  # left vs right holding, as a distribution comparison
  lv = [r['y_lane'] for r in out['lanes']['left']['episodes']
        if r['y_lane'] is not None]
  rv = [r['y_lane'] for r in out['lanes']['right']['episodes']
        if r['y_lane'] is not None]
  out['left_vs_right_after_handover'] = mannwhitney(lv, rv)
  print('\nleft vs right lane after handover:',
        out['left_vs_right_after_handover'])
  p = os.path.join(args.out_dir, 'lane_holding.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('->', p, flush=True)


if __name__ == '__main__':
  main()
