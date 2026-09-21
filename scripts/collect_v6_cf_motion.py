#!/usr/bin/env python
"""AntMaze V6 -- targeted CF-controlled motion sequences for the one-step ETT (user's step (1) after 5650f8d, 2026-09-21).

The frozen CF s0 actor's mode is rolled from PRE-SELECTED start points of the training pool (anchors of the reset / start_early /
pre_zone1_early / pre_mouth / between strata, seeded uniform samples, EXCLUDING the 192 anchors of the repeated-draws diagnostic hold-out),
with natural hidden draws (both hazards Bernoulli 0.5, fresh clocks / jitter per rollout), two rollouts per start.  Every transition
(s_t, a_q_t, s_{t+1}) is recorded with the FULL 29-d state (pose and velocities), plus per row the speed, the hazard-band flag and the
stall flag, and per rollout the start, the hidden draws and the outcome.  Successes, deaths, timeouts and slow walks are all kept; no
selection by outcome.  Coverage targets: the turn after reset, the turn and the far-route walk, deceleration and stalls (swaying and
slow progress are moving rows with small deltas, not frozen rows).  The rollouts are split by START POINT into train / val / test
(70 / 15 / 15, seeded): train = the new supervision, val = model selection during the fit, test = the three-layer acceptance.
The teacher's advice / onset are NOT recorded here (the motion model and the stationary gate take (s, a_q) only in v4); a later pass
with the teacher walked along these paths would be needed for onset / generator supervision.  Oracle (simulator) supervision:
engineering-verification stage.
Output: cf_motion/rollouts_cf.npz, cf_motion/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_repeated_draws as RD  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402
from fit_v6_ett_one_step_v2 import in_band, STAT_TOL  # noqa: E402

OUT = MP.OUT / 'cf_motion'
STATE_DIM, OBS_W, ACTION_DIM, HORIZON = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM, MP.HORIZON
START_SEED = 4343
HAZARD_SEED0 = 307_000_000
WORKER_SEED0 = 308_000_000
N_STARTS = {'reset': 128, 'start_early': 128, 'pre_zone1_early': 128, 'pre_mouth': 128, 'between': 64}
ROLLOUTS_PER_START = 2
SPLIT = (0.70, 0.15, 0.15)
SLOW_SPEED = 0.03           # |xy delta| per step below which a moving row counts as slow (the stall / swaying regime; STAT_TOL = fully static)


def select_starts():
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  xy = anchors.state[:, :2].astype(np.float64); reg = region_of(xy); t = anchors.t.astype(np.int64); x = xy[:, 0]
  masks = {'reset': t == 0, 'start_early': (reg == REGIONS.index('start')) & (t >= 1) & (t <= 40), 'pre_zone1_early': (reg == REGIONS.index('pre_zone1')) & (x < 4.5),
           'pre_mouth': (reg == REGIONS.index('pre_zone1')) & (x >= 4.5), 'between': reg == REGIONS.index('between')}
  held = set(np.load(RD.OUT / 'states.npz', allow_pickle=False)['anchor_id'].astype(int).tolist())
  rng = np.random.default_rng(START_SEED); rows = []
  for g, n in N_STARTS.items():
    idx = np.array([k for k in np.nonzero(masks[g])[0] if int(k) not in held]); sel = np.sort(rng.choice(idx, n, replace=False))
    for k in sel:
      rows.append({'group': g, 'anchor_id': int(k), 'episode': int(anchors.episode[k]), 't': int(anchors.t[k]), 'state': anchors.state[k].astype(np.float32), 'goal_xy': anchors.goal_xy[k].astype(np.float32)})
  n = len(rows); perm = rng.permutation(n); split = np.empty(n, dtype='<U5')
  a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
  split[perm[:a]] = 'train'; split[perm[a:b]] = 'val'; split[perm[b:]] = 'test'
  return rows, split


def _worker(args):
  jobs, worker_seed = args
  import build_v6_branch_replay as B
  import diag_v6_first_step_crossover as DG
  env, _ = B._worker_env(worker_seed)
  cf = MP.mode_policy(RD.ckpt('CF'))
  out = []
  for (si, r, state, goal, t) in jobs:
    hz = HAZARD_SEED0 + si * 10 + r
    DG.reseed(env, hz)
    o = B.restore(env, state.astype(np.float64), goal.astype(np.float64), int(t))          # natural hidden draws
    u1, u2 = bool(env._active[1]), bool(env._active[2]); t0 = (int(env._t0[1]), int(env._t0[2]))
    rows_s, rows_a = [o[:STATE_DIM].astype(np.float32)], []
    step, reached, done = 0, False, False; max_steps = HORIZON - int(t)
    while not done and not reached and step < max_steps:
      a = np.asarray(cf(o), np.float32); rows_a.append(a)
      o, reward, done, info = env.step(a); rows_s.append(o[:STATE_DIM].astype(np.float32)); step += 1; reached = bool(reward > 0)
    outcome = 'success' if reached else ('death' if bool(info.get('failure', False)) or done else 'timeout')
    out.append({'start': int(si), 'rep': int(r), 'hazard_seed': hz, 'u1': u1, 'u2': u2, 't0_1': t0[0], 't0_2': t0[1], 'outcome': outcome, 'steps': step, 'S': np.stack(rows_s), 'A': np.stack(rows_a)})
  return out


def mode_collect(args):
  from multiprocessing import get_context
  OUT.mkdir(parents=True, exist_ok=True); p = OUT / 'rollouts_cf.npz'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  starts, split = select_starts()
  jobs = [(si, r, s['state'], s['goal_xy'], s['t']) for si, s in enumerate(starts) for r in range(ROLLOUTS_PER_START)]
  if args.limit:
    jobs = [j for j in jobs if j[0] < int(args.limit)]
  n_parts = max(1, args.workers * 4); parts = [jobs[i::n_parts] for i in range(n_parts)]; parts = [q for q in parts if q]
  t0 = time.time()
  with get_context('spawn').Pool(args.workers) as pool:
    res = pool.map(_worker, [(q, WORKER_SEED0 + i) for i, q in enumerate(parts)])
  res = sorted([r for part in res for r in part], key=lambda r: (r['start'], r['rep']))
  # rows: transitions (s_t, a_t, s_{t+1}); per row the rollout id, the step, speed, band, stall flags
  S = np.concatenate([r['S'][:-1] for r in res]); S2 = np.concatenate([r['S'][1:] for r in res]); A = np.concatenate([r['A'] for r in res])
  rid = np.concatenate([np.full(r['steps'], i, np.int64) for i, r in enumerate(res)]); step = np.concatenate([np.arange(r['steps']) for r in res])
  speed = np.linalg.norm(S2[:, :2] - S[:, :2], axis=1); static = np.abs(S2 - S).max(axis=1) < STAT_TOL; slow = (speed < SLOW_SPEED) & ~static
  band = in_band(S[:, :2].astype(np.float64)); reg = region_of(S[:, :2].astype(np.float64))
  start_group = np.array([s['group'] for s in starts]); roll_start = np.array([r['start'] for r in res])
  np.savez_compressed(p, s=S.astype(np.float32), a=A.astype(np.float32), s2=S2.astype(np.float32), rollout=rid, step=step, speed=speed.astype(np.float32), static=static, slow=slow, band=band, region=reg.astype(np.int8),
                      roll_start=roll_start, roll_rep=np.array([r['rep'] for r in res]), roll_u1=np.array([r['u1'] for r in res]), roll_u2=np.array([r['u2'] for r in res]), roll_t0_1=np.array([r['t0_1'] for r in res]), roll_t0_2=np.array([r['t0_2'] for r in res]),
                      roll_outcome=np.array([r['outcome'] for r in res]), roll_steps=np.array([r['steps'] for r in res]), roll_split=split[roll_start], roll_group=start_group[roll_start], roll_hazard_seed=np.array([r['hazard_seed'] for r in res]),
                      start_anchor=np.array([s['anchor_id'] for s in starts]), start_group=start_group, start_split=split, start_state=np.stack([s['state'] for s in starts]), start_goal=np.stack([s['goal_xy'] for s in starts]), start_t=np.array([s['t'] for s in starts]),
                      meta=np.asarray(json.dumps({'policy': 'the frozen CF s0 actor mode (RD.ckpt CF)', 'starts': N_STARTS, 'rollouts_per_start': ROLLOUTS_PER_START, 'hidden_draws': 'natural (p 0.5 each), fresh per rollout', 'split': 'by start point 70/15/15 seeded',
                                                  'excluded': 'the 192 repeated-draws hold-out anchors', 'slow_speed': SLOW_SPEED, 'stat_tol': STAT_TOL, 'wall_seconds': time.time() - t0})))
  summ = {'rollouts': len(res), 'rows': int(len(S)), 'outcomes': {k: int(sum(r['outcome'] == k for r in res)) for k in ('success', 'death', 'timeout')}, 'mean_steps': float(np.mean([r['steps'] for r in res])),
          'rows_by_split': {k: int((split[roll_start][rid] == k).sum()) for k in ('train', 'val', 'test')}, 'rows_by_group': {g: int((start_group[roll_start][rid] == g).sum()) for g in N_STARTS},
          'rows_by_region': {REGIONS[i]: int((reg == i).sum()) for i in range(len(REGIONS)) if (reg == i).any()}, 'far_rows': int(np.isin(reg, FAR).sum()),
          'static_rows': int(static.sum()), 'slow_rows': int(slow.sum()), 'speed_quantiles': {q: float(np.percentile(speed, q)) for q in (10, 25, 50, 75, 90)},
          'rollouts_with_stall_segment_ge20': int(sum(1 for i, r in enumerate(res) if _longest_run((speed[rid == i] < SLOW_SPEED)) >= 20)), 'far_entry_rollouts': int(sum(1 for i, r in enumerate(res) if np.isin(reg[rid == i], FAR).any()))}
  MP.write_json(OUT / 'summary.json', summ); print(json.dumps(summ, indent=1), flush=True)


def _longest_run(flag):
  best = cur = 0
  for f in flag:
    cur = cur + 1 if f else 0; best = max(best, cur)
  return best


def main(argv=None):
  ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('collect',)); ap.add_argument('--workers', type=int, default=18); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv); mode_collect(args); return 0


if __name__ == '__main__':
  sys.exit(main())
