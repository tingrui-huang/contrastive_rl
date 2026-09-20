#!/usr/bin/env python
"""AntMaze V6 -- the HYBRID futures control (user's decision after 694bc02, 2026-09-20): accurate motion + the current learned risk.

An oracle diagnostic control, not a learned method.  It answers: if the body's motion were exact while the danger is still
judged by the learned model, does the policy gain come back?  Per anchor (all 53,747), exactly the learned-ETT construction of
exp_v6_ett_futures.generate_fold with ONE substitution:
  motion       the SIMULATOR (both rockfalls forced inactive: physics only; the simulator never kills, never displaces)
  advice       the advice generator v3 (one hidden context per path from the prior, kept; MAP hold, sampled torque)
  onset        the v4 onset head on (s, a_b, a_q, its own predicted delta, exact history counters), sampled per step
  termination  reach (the simulator's reward) / the sampled onset (the death frame = the simulator's next state) / horizon 800 - t
  continuation the logged torque once, then the start agent's mode on the simulator state; anchor k uses its fold's models
Output: ett_futures_hybrid/branches_ett.npz in the branch-table format; training / evaluation / report through
exp_v6_ett_futures.py --ett hybrid (same recipe, five seeds, the same draw 8909).
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
import exp_v6_ett_futures as EF  # noqa: E402
import fit_v6_ett_advice as FA  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from fit_v6_ett_one_step_v2 import in_band  # noqa: E402

OUT = MP.OUT / 'ett_futures_hybrid'
BRANCHES = OUT / 'branches_ett.npz'
GEN_SEED = 208_000_000
WORKER_SEED0 = 209_000_000
STATE_DIM, OBS_W, ACTION_DIM, HORIZON = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM, MP.HORIZON
NQ, NV = 15, 14


def restore_no_hazard(env, state, goal_xy, t):
  """build_v6_branch_replay.restore with both rockfalls forced inactive: the simulator provides the physics only."""
  import mujoco
  env.reset(rockfall_active_1=False, rockfall_active_2=False)         # RockfallClockV6Env.reset: audit-only overrides; the RNG streams are consumed as usual
  u = env._env
  u.data.qpos[:NQ] = state[:NQ]
  u.data.qvel[:NV] = state[NQ:NQ + NV]
  mujoco.mj_forward(u.model, u.data)
  full = np.array(env._goal_state_full, np.float32)
  full[:2] = goal_xy
  env._goal_state_full = full
  env._goal_vec = full[list(env.goal_indices)]
  u.goal = np.asarray(goal_xy, float).copy()
  env._last_obs = u._obs_dict()
  env._t = int(t)
  return env._flatten(env._last_obs)


def _worker(args):
  jobs, worker_seed = args
  import build_v6_branch_replay as B
  env, _ = B._worker_env(worker_seed)
  act_fn = MP.mode_policy(MP.START_CKPT)
  obs, act, _, _ = MP.load_dataset()
  F2.OUT = F2.OUT_V4
  P = {}; G = {}
  out = []
  for (k, e, t, fold) in jobs:
    if fold not in P:
      P[fold] = F2.Predictor2(F2.load_model(fold)); G[fold] = FA.load_generator(fold)
    Pf, gen = P[fold], G[fold]
    rng = np.random.default_rng(GEN_SEED + int(k))
    obs_row = obs[e, t, :OBS_W].astype(np.float32); goal = obs_row[STATE_DIM:OBS_W].astype(np.float64)
    o = restore_no_hazard(env, obs_row[:STATE_DIM].astype(np.float64), goal, int(t))
    assert env._active[1] is False and env._active[2] is False
    z = FA.sample_context(rng, 1)
    kh_prev = int(FA.zero_run_before(act, np.array([e]), np.array([t]))[0]); prev_ab = (act[e, t - 1] if t >= 1 else np.zeros(ACTION_DIM, np.float32)).astype(np.float32)
    m_gen = FA.first_mouth_in_prefix(obs[e], int(t)); last_h = last_b = 0
    rows_o, rows_a = [obs_row], [act[e, t].astype(np.float32)]
    a_q = rows_a[0]; s = obs_row[:STATE_DIM].astype(np.float32)
    max_steps = HORIZON - int(t); outcome = 'timeout'; step = 0; p_seq = []
    while step < max_steps:
      j = step; t_abs = int(t) + j
      if j > 0:
        a_q = np.asarray(act_fn(o), np.float32); rows_a.append(a_q)            # the continuation torque from the simulator state (as _branch_one)
      o31 = np.concatenate([s, goal.astype(np.float32)])
      band_now = bool(in_band(s[None, :2].astype(np.float64))[0])
      if j == 0 or not band_now:
        last_b = j
      kb_now = (j - last_b) if band_now else 0
      for zi, zz in enumerate((1, 2)):
        if m_gen[zi] < 0 and bool(FA.at_mouth(s[None, :2].astype(np.float64), zz)[0]):
          m_gen[zi] = t_abs
      ctx = FA.context_features(np.array([t_abs]), *[np.asarray(v) for v in z])
      mf = FA.mouth_features(np.array([t_abs]), np.array([m_gen[0]]), np.array([m_gen[1]]), np.asarray(z[2]), np.asarray(z[3]))
      hold_g, ab_g, _ = gen.sample(FA.features(o31[None], ctx, mf, np.array([kh_prev]), np.array([kb_now]), prev_ab[None], gen.norm), rng, decision='map')
      hold = bool(hold_g[0]); ab = ab_g[0]
      if j == 0 or not hold:
        last_h = j
      kh_now = (j - last_h) if hold else 0
      _, p_on, _ = Pf.step(s[None], ab[None], a_q[None], np.array([kh_now]), np.array([kb_now]))
      p = float(p_on[0]); p_seq.append(p)
      o, reward, done, info = env.step(np.asarray(a_q, np.float32))             # physics only
      rows_o.append(o[:OBS_W].astype(np.float32)); step += 1
      assert not bool(info.get('failure', False))
      if rng.random() < p:
        outcome = 'death'; break
      if reward > 0:
        outcome = 'success'; break
      if done:
        break
      kh_prev = kh_prev + 1 if hold else 0; prev_ab = ab
      s = o[:STATE_DIM].astype(np.float32)
    rows_a.append(np.zeros(ACTION_DIM, np.float32))                              # dummy action on the last row
    assert len(rows_a) == len(rows_o)
    out.append({'anchor_id': int(k), 'episode': int(e), 't': int(t), 'obs': np.stack(rows_o), 'act': np.stack(rows_a), 'steps': int(step), 'outcome': outcome,
                'u1': bool(z[0][0]), 'u2': bool(z[1][0]), 't0_1': int(z[2][0]), 't0_2': int(z[3][0]), 'p_death_integrated': float(1.0 - np.prod(1.0 - np.array(p_seq)))})
  return out


def mode_generate(args):
  from multiprocessing import get_context
  if BRANCHES.exists() and not args.force:
    print(f'{BRANCHES} exists', flush=True); return
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  S = np.load(FA.EC / 'supervision.npz', allow_pickle=False); fold_anchor = S['fold_anchor'].astype(np.int64)
  ids = np.arange(anchors.n) if not args.limit else np.arange(int(args.limit))
  jobs = [(int(k), int(anchors.episode[k]), int(anchors.t[k]), int(fold_anchor[k])) for k in ids]
  n_parts = max(1, args.workers * 4); parts = [jobs[i::n_parts] for i in range(n_parts)]; parts = [p for p in parts if p]
  t0 = time.time()
  if args.workers <= 1:
    res = [_worker((p, WORKER_SEED0 + i)) for i, p in enumerate(parts)]
  else:
    with get_context('spawn').Pool(args.workers) as pool:
      res = pool.map(_worker, [(p, WORKER_SEED0 + i) for i, p in enumerate(parts)])
  res = sorted([r for part in res for r in part], key=lambda r: r['anchor_id'])
  L = np.array([len(r['obs']) for r in res], np.int64); off = np.concatenate([[0], np.cumsum(L)[:-1]])
  oc = np.array([r['outcome'] for r in res]); ks = np.array([r['anchor_id'] for r in res], np.int64)
  meta = {'source': 'HYBRID: simulator motion (rockfalls forced inactive) + advice generator v3 + v4 onset head sampled per step; prior context per path', 'continuation_ckpt': str(MP.START_CKPT),
          'continuation_ckpt_sha256': MP.sha256(MP.START_CKPT), 'gen_seed': GEN_SEED, 'n_anchors': int(len(ks)), 'wall_seconds': time.time() - t0,
          'outcome_counts': {k: int((oc == k).sum()) for k in ('success', 'death', 'timeout')}, 'p_death_integrated_mean': float(np.mean([r['p_death_integrated'] for r in res])), 'filtering': 'none',
          'hidden_context': 'sampled from the prior per path (u1 / u2 / t0 stored); the simulator hazards are OFF, the anchor\'s actual context is not read'}
  np.savez_compressed(BRANCHES, obs_rows=np.concatenate([r['obs'] for r in res]), act_rows=np.concatenate([r['act'] for r in res]), offset=off, length=L,
                      episode=anchors.episode[ks], t=anchors.t[ks], outcome=oc, steps=np.array([r['steps'] for r in res], np.int64),
                      u1=np.array([r['u1'] for r in res]), u2=np.array([r['u2'] for r in res]), t0_1=np.array([r['t0_1'] for r in res], np.int64), t0_2=np.array([r['t0_2'] for r in res], np.int64),
                      n_query_steps=np.ones(len(ks), np.int64), restore_maxdiff=np.zeros(len(ks), np.float32), hazard_seed=np.full(len(ks), GEN_SEED, np.int64), meta=np.asarray(json.dumps(meta, sort_keys=True)))
  EF.set_ett('hybrid')
  summ = MP.generation_summary(BRANCHES, anchors)
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    oc_cf, L_cf = d['outcome'].astype(str)[ks], d['length'].astype(np.int64)[ks]
  from exp_v6_learned_ett import REGIONS, region_of
  reg = region_of(anchors.state[ks, :2]); W = anchors.weight[ks]
  cmp = {'pooled_weighted': {k: {'ett': float((W * (oc == k)).sum() / W.sum()), 'sim': float((W * (oc_cf == k)).sum() / W.sum())} for k in ('success', 'death', 'timeout')},
         'mean_rows': {'ett': float(L.mean()), 'sim': float(L_cf.mean())},
         'by_region': {REGIONS[i]: {'n': int((reg == i).sum()), **{k: {'ett': float((oc[reg == i] == k).mean()), 'sim': float((oc_cf[reg == i] == k).mean())} for k in ('success', 'death', 'timeout')}} for i in range(len(REGIONS)) if (reg == i).sum() >= 100}}
  MP.write_json(OUT / 'generation_ett.json', {'summary': summ, 'vs_simulator_table': cmp})
  print(json.dumps({'pooled': cmp['pooled_weighted'], 'rows': cmp['mean_rows'], 'wall': meta['wall_seconds']}, indent=1), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('generate',))
  ap.add_argument('--workers', type=int, default=18)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  mode_generate(args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
