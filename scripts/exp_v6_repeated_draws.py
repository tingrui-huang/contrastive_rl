#!/usr/bin/env python
"""AntMaze V6 -- REPEATED PAIRED HAZARD DRAWS at pre-selected decision states (user's plan after the pairing check, 2026-09-21).

One question: at the SAME decision state, do the candidate first-step torques have a stable average advantage, and do the existing critics
recognise it?  An oracle diagnostic (the hidden variables are never given to a learner); no training.

States (fixed before any outcome is seen; seeded uniform samples, not selected by outcome or critic score):
  training states   64 reset anchors (t = 0), 64 start_early anchors (start region, t 1-40), 64 pre_zone1_early anchors (x < 4.5)
  independent       64 fresh env resets (a seed stream of its own) and the states 20 start-agent steps later (64), never in the dataset
Candidates at each state (all evaluated at the task goal of that state): the logged torque (training states only), the start agent's mode,
the PRE-SPECIFIED CF actor's mode (seed 0), the MF actor's mode (seed 0), plus the CF2 actor's mode (seed 0; an addition to the user's
four, the other single-draw actor).  Each candidate is executed for ONE step; the same frozen start agent (mode) continues, as in the
branch tables.  Paired conditions: for each state and draw index the hazard clocks and rock jitter are identical across the candidates
(the same hidden seed), the hazard activity is STRATIFIED over the four classes U00 / U10 / U01 / U11 with 8 clock / jitter draws each
(32 per candidate), combined with the prior weight 1/4 per class.  Every outcome kept.  Pre-stated extension rule: if the 95 % interval
(state-level s.e.) of the mean success advantage of the CF mode over the start mode is wider than +-0.05 in any state group, extend to
16 draws per class (64).
Measured per rollout: outcome, steps, far-route entry, the geometric-law masses of the path (goal area, reach 0.5, far regions, death
frame) -- the NCE-target quantities -- and the hidden draws.  Report: per state and candidate the class-weighted means and the paired
differences vs the start mode (and vs the logged torque); per group the state-level mean / s.e. / share positive; the critics' (CF, CF2,
MF x 5 seeds) twin-min logit for each candidate at the task goal against the measured success and goal-mass advantages.

CROSSOVER (user's plan after afc12c9): the same states, candidates and hidden seeds with the continuation policy switched to the FROZEN
CF s0 actor's mode for every candidate (--continuation CF; the start-continuation rollouts are re-used), plus 64 'cf_early' states = the
same fresh resets rolled 20 steps by the CF s0 actor (the states the new policy visits while turning; also run with the start
continuation).  Read: (1) success and far-route completion, (2) the geometric-law masses within the reach radius / near the goal / in
the goal area, (3) the critics' REGION-level ranking (importance-corrected p(region | s, a), reference goal set = the family's own training
goal marginal) against the measured region masses under each continuation -- never the exact-goal logit against a region mass.
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
import exp_v6_oracle_draw2 as D2  # noqa: E402
import exp_v6_multi_futures as MFD  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

OUT = MP.OUT / 'repeated_draws'
STATE_DIM, GOAL_DIM, OBS_W, ACTION_DIM, HORIZON = MP.STATE_DIM, MP.GOAL_DIM, MP.OBS_W, MP.ACTION_DIM, MP.HORIZON
GAMMA = MP.GAMMA
NQ, NV = 15, 14
STATE_SEED = 4242
INDEP_ENV_SEED0 = 303_000_000
HAZARD_SEED0 = 304_000_000
WORKER_SEED0 = 305_000_000
N_PER_GROUP = 64
CLASSES = ((False, False), (True, False), (False, True), (True, True))       # U00, U10, U01, U11
CLASS_NAMES = ('U00', 'U10', 'U01', 'U11')
DRAWS_PER_CLASS = 8
PRE_SPECIFIED_SEED = 0
CANDIDATES = ('logged', 'start', 'CF', 'MF', 'CF2')
CI_HALF_WIDTH_RULE = 0.05
REACH_R = 0.5


def ckpt(fam):
  return {'start': MP.START_CKPT, 'CF': EF.run_dir('CF', PRE_SPECIFIED_SEED) / 'final.pkl', 'MF': MFD.run_dir_mf(PRE_SPECIFIED_SEED) / 'final.pkl',
          'CF2': D2.run_dir2(PRE_SPECIFIED_SEED) / 'final.pkl'}[fam]


# ------------------------------------------------------------------ states
def mode_states(args):
  """The fixed state set: seeded samples of the three training strata and fresh independent resets / early states."""
  import build_v6_branch_replay as B
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'states.npz'
  if args.add_cf_early:
    return add_cf_early(p)
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  xy = anchors.state[:, :2].astype(np.float64); reg = region_of(xy); t = anchors.t.astype(np.int64); x = xy[:, 0]
  masks = {'reset': t == 0, 'start_early': (reg == REGIONS.index('start')) & (t >= 1) & (t <= 40), 'pre_zone1_early': (reg == REGIONS.index('pre_zone1')) & (x < 4.5)}
  rng = np.random.default_rng(STATE_SEED)
  rows = []
  for g, m in masks.items():
    idx = np.nonzero(m)[0]; sel = np.sort(rng.choice(idx, N_PER_GROUP, replace=False))
    for k in sel:
      rows.append({'group': g, 'anchor_id': int(k), 'episode': int(anchors.episode[k]), 't': int(anchors.t[k]), 'state': anchors.state[k].astype(np.float32),
                   'goal_xy': anchors.goal_xy[k].astype(np.float32), 'logged': anchors.action[k].astype(np.float32)})
  # independent states: fresh resets from a seed stream of their own, and the start agent 20 steps later (the hazards are irrelevant before x = 6.6)
  env, _ = B._worker_env(INDEP_ENV_SEED0)
  act_fn = MP.mode_policy(MP.START_CKPT)
  for i in range(N_PER_GROUP):
    o = env.reset()
    rows.append({'group': 'indep_reset', 'anchor_id': -1, 'episode': -1, 't': 0, 'state': o[:STATE_DIM].astype(np.float32), 'goal_xy': o[STATE_DIM:OBS_W].astype(np.float32), 'logged': np.full(ACTION_DIM, np.nan, np.float32)})
    for j in range(20):
      o, r, done, info = env.step(np.asarray(act_fn(o), np.float32))
      assert not done and not bool(info.get('failure', False))
    rows.append({'group': 'indep_early', 'anchor_id': -1, 'episode': -1, 't': 20, 'state': o[:STATE_DIM].astype(np.float32), 'goal_xy': o[STATE_DIM:OBS_W].astype(np.float32), 'logged': np.full(ACTION_DIM, np.nan, np.float32)})
  np.savez_compressed(p, group=np.array([r['group'] for r in rows]), anchor_id=np.array([r['anchor_id'] for r in rows], np.int64), episode=np.array([r['episode'] for r in rows], np.int64),
                      t=np.array([r['t'] for r in rows], np.int64), state=np.stack([r['state'] for r in rows]), goal_xy=np.stack([r['goal_xy'] for r in rows]), logged=np.stack([r['logged'] for r in rows]),
                      meta=np.asarray(json.dumps({'state_seed': STATE_SEED, 'indep_env_seed': INDEP_ENV_SEED0, 'n_per_group': N_PER_GROUP, 'selection': 'seeded uniform within stratum; independent = fresh env resets and 20 start-agent steps; no selection by outcome or critic'})))
  print({g: int((np.array([r['group'] for r in rows]) == g).sum()) for g in ('reset', 'start_early', 'pre_zone1_early', 'indep_reset', 'indep_early')}, flush=True)


def add_cf_early(p):
  """64 'cf_early' states: the SAME fresh resets as indep_reset (the env re-seeded, the reset streams replayed; asserted equal), then 20
  steps of the frozen CF s0 actor's mode -- the early states the new policy actually visits while turning.  Appended to states.npz."""
  import build_v6_branch_replay as B
  S = {k: v for k, v in np.load(p, allow_pickle=False).items()}
  if (S['group'].astype(str) == 'cf_early').any():
    print('cf_early states exist', flush=True); return
  env, _ = B._worker_env(INDEP_ENV_SEED0)
  cf = MP.mode_policy(ckpt('CF'))
  base = np.nonzero(S['group'].astype(str) == 'indep_reset')[0]
  rows = []
  for i in range(N_PER_GROUP):
    o = env.reset()
    assert np.abs(o[:STATE_DIM] - S['state'][base[i]]).max() < 1e-5, 'the replayed reset differs from the stored indep_reset state'
    for j in range(20):
      o, r, done, info = env.step(np.asarray(cf(o), np.float32))
      assert not done and not bool(info.get('failure', False))
    rows.append((o[:STATE_DIM].astype(np.float32), o[STATE_DIM:OBS_W].astype(np.float32)))
  meta = json.loads(str(S.pop('meta'))); meta['cf_early'] = 'the indep_reset resets replayed (asserted equal) + 20 steps of the frozen CF s0 mode'
  n = len(rows)
  S['group'] = np.concatenate([S['group'], np.array(['cf_early'] * n)]); S['anchor_id'] = np.concatenate([S['anchor_id'], np.full(n, -1, np.int64)])
  S['episode'] = np.concatenate([S['episode'], np.full(n, -1, np.int64)]); S['t'] = np.concatenate([S['t'], np.full(n, 20, np.int64)])
  S['state'] = np.concatenate([S['state'], np.stack([r[0] for r in rows])]); S['goal_xy'] = np.concatenate([S['goal_xy'], np.stack([r[1] for r in rows])])
  S['logged'] = np.concatenate([S['logged'], np.full((n, ACTION_DIM), np.nan, np.float32)])
  np.savez_compressed(p, **S, meta=np.asarray(json.dumps(meta)))
  print({g: int((S['group'].astype(str) == g).sum()) for g in np.unique(S['group'].astype(str))}, flush=True)


# ------------------------------------------------------------------ rollouts
def restore_forced(env, state, goal_xy, t, u1, u2):
  """build_v6_branch_replay.restore with the hazard activity forced (audit-only overrides; the clock / jitter streams are consumed as usual)."""
  import mujoco
  env.reset(rockfall_active_1=bool(u1), rockfall_active_2=bool(u2))
  u = env._env
  u.data.qpos[:NQ] = state[:NQ]; u.data.qvel[:NV] = state[NQ:NQ + NV]
  mujoco.mj_forward(u.model, u.data)
  full = np.array(env._goal_state_full, np.float32); full[:2] = goal_xy
  env._goal_state_full = full; env._goal_vec = full[list(env.goal_indices)]; u.goal = np.asarray(goal_xy, float).copy()
  env._last_obs = u._obs_dict(); env._t = int(t)
  return env._flatten(env._last_obs)


def path_masses(xy, goal_xy, outcome):
  """The geometric law P(m) ~ gamma^m over 1..L-1 on the path rows: masses near the task goal / in the goal area / in the far regions / on the death frame."""
  L = len(xy); m = np.arange(1, L); g = GAMMA ** m; g /= g.sum()
  d = np.linalg.norm(xy[1:] - goal_xy, axis=1); reg = region_of(xy[1:].astype(np.float64))
  return {'reach0.5': float((g * (d <= REACH_R)).sum()), 'near2.0': float((g * (d <= 2.0)).sum()), 'goal_area': float((g * (reg == REGIONS.index('goal_area'))).sum()),
          'far': float((g * np.isin(reg, FAR)).sum()), 'death_frame': float(g[-1]) if outcome == 'death' else 0.0}


def _worker(args):
  jobs, worker_seed, continuation = args
  import build_v6_branch_replay as B
  import diag_v6_first_step_crossover as DG
  env, _ = B._worker_env(worker_seed)
  pol = {fam: MP.mode_policy(ckpt(fam)) for fam in ('start', 'CF', 'MF', 'CF2')}
  act_fn = pol[continuation]                                                    # the SAME continuation policy for every candidate
  S = np.load(OUT / 'states.npz', allow_pickle=False)
  state, goal, tt, logged = S['state'], S['goal_xy'], S['t'].astype(np.int64), S['logged']
  out = []
  for (si, cand, ci, r) in jobs:
    s = state[si].astype(np.float64); gxy = goal[si].astype(np.float64); t = int(tt[si]); u1, u2 = CLASSES[ci]
    o31 = np.concatenate([state[si], goal[si]]).astype(np.float32)
    if cand == 'logged':
      a0 = logged[si].astype(np.float32)
      if not np.all(np.isfinite(a0)):
        continue
    else:
      a0 = np.asarray(pol[cand](o31), np.float32)
    hz = HAZARD_SEED0 + si * 1000 + ci * DRAWS_PER_CLASS * 2 + r                # identical across candidates
    DG.reseed(env, hz)
    o = restore_forced(env, s, gxy, t, u1, u2)
    assert env._active[1] == u1 and env._active[2] == u2
    t0 = (int(env._t0[1]), int(env._t0[2]))
    rows = [o[:2].astype(np.float32)]
    o, reward, done, info = env.step(a0); rows.append(o[:2].astype(np.float32)); step = 1; reached = bool(reward > 0)
    max_steps = HORIZON - t
    while not done and not reached and step < max_steps:
      o, reward, done, info = env.step(np.asarray(act_fn(o), np.float32)); rows.append(o[:2].astype(np.float32)); step += 1; reached = bool(reward > 0)
    outcome = 'success' if reached else ('death' if bool(info.get('failure', False)) or done else 'timeout')
    xy = np.stack(rows); reg = region_of(xy.astype(np.float64))
    far_entry = bool(np.isin(reg, FAR).any()); zone1_entry = bool((reg == REGIONS.index('zone1')).any())
    out.append({'state': int(si), 'cand': cand, 'cls': int(ci), 'rep': int(r), 'hazard_seed': int(hz), 'u1': u1, 'u2': u2, 't0_1': t0[0], 't0_2': t0[1], 'outcome': outcome, 'steps': int(step), 'continuation': continuation,
                'far_entry': far_entry, 'zone1_entry': zone1_entry, 'a0': a0, **path_masses(xy, gxy, outcome), 'xy': xy})
  return out


def rollout_file(args):
  tag = args.tag or ('base' if not args.groups else '_'.join(args.groups))
  return OUT / ('rollouts.npz' if (args.continuation == 'start' and tag == 'base') else f'rollouts_{args.continuation}_{tag}.npz')


def mode_generate(args):
  from multiprocessing import get_context
  p = rollout_file(args)
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  S = np.load(OUT / 'states.npz', allow_pickle=False); n_states = len(S['t']); groups = S['group'].astype(str)
  states = [si for si in range(n_states) if (not args.groups or groups[si] in args.groups)]
  draws = int(args.draws_per_class)
  jobs = [(si, cand, ci, r) for si in states for cand in CANDIDATES for ci in range(len(CLASSES)) for r in range(draws)]
  if args.limit:
    jobs = [j for j in jobs if j[0] in states[:int(args.limit)]]
  n_parts = max(1, args.workers * 4); parts = [jobs[i::n_parts] for i in range(n_parts)]; parts = [q for q in parts if q]
  t0 = time.time()
  if args.workers <= 1:
    res = [_worker((q, WORKER_SEED0 + i, args.continuation)) for i, q in enumerate(parts)]
  else:
    with get_context('spawn').Pool(args.workers) as pool:
      res = pool.map(_worker, [(q, WORKER_SEED0 + i, args.continuation) for i, q in enumerate(parts)])
  res = [r for part in res for r in part]
  L = np.array([len(r['xy']) for r in res], np.int64); off = np.concatenate([[0], np.cumsum(L)[:-1]])
  np.savez_compressed(p, state=np.array([r['state'] for r in res], np.int64), cand=np.array([r['cand'] for r in res]), cls=np.array([r['cls'] for r in res], np.int64), rep=np.array([r['rep'] for r in res], np.int64),
                      hazard_seed=np.array([r['hazard_seed'] for r in res], np.int64), u1=np.array([r['u1'] for r in res]), u2=np.array([r['u2'] for r in res]), t0_1=np.array([r['t0_1'] for r in res], np.int64), t0_2=np.array([r['t0_2'] for r in res], np.int64),
                      outcome=np.array([r['outcome'] for r in res]), steps=np.array([r['steps'] for r in res], np.int64), far_entry=np.array([r['far_entry'] for r in res]), zone1_entry=np.array([r['zone1_entry'] for r in res]),
                      a0=np.stack([r['a0'] for r in res]), **{k: np.array([r[k] for r in res], np.float64) for k in ('reach0.5', 'near2.0', 'goal_area', 'far', 'death_frame')},
                      xy_rows=np.concatenate([r['xy'] for r in res]), offset=off, length=L,
                      meta=np.asarray(json.dumps({'draws_per_class': draws, 'classes': CLASS_NAMES, 'candidates': list(CANDIDATES), 'pre_specified_seed': PRE_SPECIFIED_SEED, 'hazard_seed0': HAZARD_SEED0, 'worker_seed0': WORKER_SEED0,
                                                  'continuation': f'the frozen {args.continuation} mode after one candidate step (the same for every candidate)', 'groups': (args.groups or 'all'), 'wall_seconds': time.time() - t0, 'n_rollouts': len(res)})))
  print(f'{len(res)} rollouts in {time.time() - t0:.0f} s; mean steps {L.mean() - 1:.0f}', flush=True)


# ------------------------------------------------------------------ report
def _load():
  S = np.load(OUT / 'states.npz', allow_pickle=False); R = np.load(OUT / 'rollouts.npz', allow_pickle=False)
  return S, {k: R[k] for k in R.files if k not in ('xy_rows', 'meta')}, json.loads(str(R['meta']))


def per_state_candidate(R, n_states):
  """Class-weighted (1/4 each) means per state x candidate, and the paired differences vs the start mode on the same draws."""
  keys = ('success', 'death', 'timeout', 'far_entry', 'goal_area', 'reach0.5', 'near2.0', 'far', 'death_frame', 'far_success')
  oc = R['outcome'].astype(str)
  val = {'success': (oc == 'success').astype(float), 'death': (oc == 'death').astype(float), 'timeout': (oc == 'timeout').astype(float), 'far_entry': R['far_entry'].astype(float),
         'goal_area': R['goal_area'], 'reach0.5': R['reach0.5'], 'near2.0': R['near2.0'], 'far': R['far'], 'death_frame': R['death_frame'], 'far_success': ((oc == 'success') & R['far_entry']).astype(float)}
  # index rollouts by (state, cand, cls, rep)
  cands = sorted(set(R['cand'].astype(str))); idx = {}
  for i, (s, c, k, r) in enumerate(zip(R['state'], R['cand'].astype(str), R['cls'], R['rep'])):
    idx[(int(s), c, int(k), int(r))] = i
  n_cls = len(CLASSES); reps = int(R['rep'].max()) + 1
  M = {c: {key: np.full(n_states, np.nan) for key in keys} for c in cands}            # class-weighted means
  D = {c: {key: np.full(n_states, np.nan) for key in keys} for c in cands}            # paired mean difference vs start
  Dse = {c: {key: np.full(n_states, np.nan) for key in keys} for c in cands}
  for s in range(n_states):
    for c in cands:
      rows = [[idx.get((s, c, k, r)) for r in range(reps)] for k in range(n_cls)]
      if any(i is None for row in rows for i in row):
        continue
      base = [[idx[(s, 'start', k, r)] for r in range(reps)] for k in range(n_cls)]
      for key in keys:
        v = np.array([[val[key][i] for i in row] for row in rows]); b = np.array([[val[key][i] for i in row] for row in base])
        M[c][key][s] = v.mean(axis=1).mean()                                          # 1/4 per class, equal reps within class
        d = v - b; D[c][key][s] = d.mean(axis=1).mean(); Dse[c][key][s] = float(np.sqrt((d.var(axis=1, ddof=1) / reps).mean() / n_cls)) if reps > 1 else np.nan
  return cands, keys, M, D, Dse


def critic_scores(S, cands, n_states, R):
  """Every critic's twin-min logit at (state, candidate torque, task goal); the candidate torque is the one actually executed (a0)."""
  import diag_v6_pairing as DP
  N = DP.Nets(); out = {}
  a0 = {}
  for c in cands:
    a0[c] = np.full((n_states, ACTION_DIM), np.nan, np.float32)
    sel = (R['cand'].astype(str) == c) & (R['cls'] == 0) & (R['rep'] == 0)
    a0[c][R['state'][sel]] = R['a0'][sel]
  obs = np.concatenate([S['state'], S['goal_xy']], axis=1).astype(np.float32)
  for fam in ('CF', 'CF2', 'MF'):
    for sd in range(5):
      qp, _ = N.load(DP.ckpt_of(fam, sd)); sc = {}
      for c in cands:
        ok = np.all(np.isfinite(a0[c]), axis=1); f = np.full(n_states, np.nan)
        if ok.any():
          f[ok] = N.diag_logits(qp, obs[ok], a0[c][ok]).min(-1)
        sc[c] = f
      out[f'{fam}_s{sd}'] = sc
  return out


def mode_report(args):
  S, R, meta = _load(); n_states = len(S['t']); groups = S['group'].astype(str)
  cands, keys, M, D, Dse = per_state_candidate(R, n_states)
  Q = critic_scores(S, cands, n_states, R)
  res = {'meta': meta, 'n_states': int(n_states), 'groups': {str(g): int((groups == g).sum()) for g in np.unique(groups)}, 'by_group': {}, 'extension_rule': f'extend to 16 draws per class if the 95 % half-width (state-level s.e.) of the CF-vs-start success advantage exceeds {CI_HALF_WIDTH_RULE} in any group'}
  for g in ('reset', 'start_early', 'pre_zone1_early', 'indep_reset', 'indep_early'):
    m = groups == g; row = {'n_states': int(m.sum()), 'candidate_means': {}, 'advantage_vs_start': {}, 'critic_recognition': {}}
    for c in cands:
      ok = m & np.isfinite(M[c]['success'])
      if ok.sum() == 0:
        continue
      row['candidate_means'][c] = {key: float(np.nanmean(M[c][key][ok])) for key in keys}
      if c != 'start':
        adv = {}
        for key in keys:
          d = D[c][key][ok]; se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
          adv[key] = {'mean': float(d.mean()), 'state_se': se, 'share_positive': float((d > 0).mean()), 'share_negative': float((d < 0).mean()), 'n': int(len(d)),
                      'ci95_half_width': (1.96 * se if se is not None else None), 'within_state_se_median': float(np.nanmedian(Dse[c][key][ok]))}
        row['advantage_vs_start'][c] = adv
    # the critics: margin (candidate - start) vs the measured advantages, across the states of the group
    for cname, sc in Q.items():
      rec = {}
      for c in cands:
        if c == 'start':
          continue
        ok = m & np.isfinite(M[c]['success']) & np.isfinite(sc[c]) & np.isfinite(sc['start'])
        if ok.sum() < 10:
          continue
        marg = sc[c][ok] - sc['start'][ok]; ds = D[c]['success'][ok]; dg = D[c]['goal_area'][ok]; dn = D[c]['near2.0'][ok]
        rec[c] = {'critic_margin_mean': float(marg.mean()), 'share_margin_positive': float((marg > 0).mean()),
                  'corr_margin_vs_success_adv': (float(np.corrcoef(marg, ds)[0, 1]) if ds.std() > 1e-9 and marg.std() > 1e-9 else None),
                  'corr_margin_vs_goalarea_adv': (float(np.corrcoef(marg, dg)[0, 1]) if dg.std() > 1e-9 and marg.std() > 1e-9 else None),
                  'corr_margin_vs_near2_adv': (float(np.corrcoef(marg, dn)[0, 1]) if dn.std() > 1e-9 and marg.std() > 1e-9 else None),
                  'sign_agreement_with_success_adv': float(((marg > 0) == (ds > 0))[ds != 0].mean()) if (ds != 0).any() else None, 'n': int(ok.sum())}
      row['critic_recognition'][cname] = rec
    res['by_group'][g] = row
  # extension rule check
  hw = [res['by_group'][g]['advantage_vs_start'].get('CF', {}).get('success', {}).get('ci95_half_width') for g in res['by_group']]
  res['extension_needed'] = bool(any(h is not None and h > CI_HALF_WIDTH_RULE for h in hw))
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res, cands), encoding='utf-8')
  print(json.dumps({'extension_needed': res['extension_needed'], 'half_widths': hw}, indent=1), flush=True)


def write_md(res, cands):
  L = [f"# Repeated paired hazard draws at pre-selected decision states ({res['meta']['draws_per_class']} per class x 4 classes, prior 1/4 each; candidates = one first step, then the frozen start agent)", '',
       f"States: {res['groups']}.  Candidates: {', '.join(cands)} (CF / MF / CF2 = the seed-{res['meta']['pre_specified_seed']} actors' modes at the task goal).  {res['extension_rule']}.  Extension needed: {res['extension_needed']}.", '']
  for g, row in res['by_group'].items():
    L += [f"## {g} (n states {row['n_states']})", '', '| candidate | success | death | timeout | far entry | far & success | goal_area mass | near2.0 mass | reach0.5 mass | far mass | death-frame mass |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, mm in row['candidate_means'].items():
      L.append(f"| {c} | {mm['success']:.3f} | {mm['death']:.3f} | {mm['timeout']:.3f} | {mm['far_entry']:.3f} | {mm['far_success']:.3f} | {mm['goal_area']:.4f} | {mm['near2.0']:.4f} | {mm['reach0.5']:.4f} | {mm['far']:.3f} | {mm['death_frame']:.4f} |")
    L += ['', '| advantage vs start (paired) | success: mean +- s.e. (share >0 / <0) | far entry | goal_area mass | near2.0 mass | death-frame mass |', '|---|---|---|---|---|---|']
    for c, adv in row['advantage_vs_start'].items():
      def cell(k, fmt='.3f'):
        a = adv[k]; return f"{a['mean']:+{fmt}} +- {a['state_se']:{fmt}} ({a['share_positive']:.2f} / {a['share_negative']:.2f})"
      L.append(f"| {c} | {cell('success')} | {cell('far_entry')} | {cell('goal_area', '.4f')} | {cell('near2.0', '.4f')} | {cell('death_frame', '.4f')} |")
    L += ['', '| critic (seed mean) | candidate | critic margin vs start (share >0) | corr with success adv | corr with goal_area adv | corr with near2.0 adv | sign agreement with success adv |', '|---|---|---|---:|---:|---:|---:|']
    for fam in ('CF', 'CF2', 'MF'):
      for c in cands:
        recs = [row['critic_recognition'][f'{fam}_s{sd}'].get(c) for sd in range(5)]
        recs = [r for r in recs if r]
        if not recs:
          continue
        def mean_of(k):
          v = [r[k] for r in recs if r[k] is not None]; return (f'{np.mean(v):.3f}' if v else 'n/a')
        L.append(f"| {fam} | {c} | {mean_of('critic_margin_mean')} ({mean_of('share_margin_positive')}) | {mean_of('corr_margin_vs_success_adv')} | {mean_of('corr_margin_vs_goalarea_adv')} | {mean_of('corr_margin_vs_near2_adv')} | {mean_of('sign_agreement_with_success_adv')} |")
    L.append('')
  return '\n'.join(L)


# ------------------------------------------------------------------ crossover report
def load_rollouts(continuation):
  """All rollout files of one continuation (rollouts.npz = start / base states; rollouts_<cont>_<tag>.npz otherwise), concatenated."""
  files = [OUT / 'rollouts.npz'] if continuation == 'start' else []
  files += sorted(OUT.glob(f'rollouts_{continuation}_*.npz'))
  parts = []
  for f in files:
    R = np.load(f, allow_pickle=False); parts.append({k: R[k] for k in R.files if k not in ('xy_rows', 'meta', 'offset', 'length')})
  assert parts, f'no rollouts for continuation {continuation}'
  keys = [k for k in parts[0] if all(k in q for q in parts)]
  return {k: np.concatenate([q[k] for q in parts]) for k in keys}, [str(f.name) for f in files]


def reference_goal_sets(anchors, rng, n_ref=4096):
  """The critic families' own training goal marginals (as diag_v6_pairing.region_readout): anchors by weight, the family's table(s), m by the law."""
  T1 = MP.BranchFutures(anchors, MP.OUT / 'branches_cf.npz'); T2 = MP.BranchFutures(anchors, D2.BRANCHES); srcs = {'T1': T1, 'T2': T2}
  cdf = np.cumsum(anchors.weight / anchors.weight.sum()); cdf[-1] = 1.0

  def ref_set(tables):
    ks = np.minimum(np.searchsorted(cdf, rng.random(n_ref), side='right'), anchors.n - 1); which = rng.integers(0, len(tables), size=n_ref); G = np.zeros((n_ref, 2), np.float32)
    for i, (k, w) in enumerate(zip(ks, which)):
      f = srcs[tables[w]]; nf = int(f.lengths[k] - 1); u = rng.random()
      mm = int(np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** nf)) / np.log(GAMMA)), 1, nf)); G[i] = f.goal_at(int(k), mm)
    return G
  return {'CF': ref_set(('T1',)), 'CF2': ref_set(('T2',)), 'MF': ref_set(('T1', 'T2'))}


def critic_region_probs(S, cands, a0, refs):
  """Per critic, per state and candidate: p(region | s, a_cand) over the family's reference goal set for regions goal_area, near2.0 (of THIS
  state's task goal), reach0.5 (of this task goal; few reference goals -- reported with its count)."""
  import diag_v6_pairing as DP
  N = DP.Nets(); n_states = len(S['t']); obs = np.concatenate([S['state'], S['goal_xy']], axis=1).astype(np.float32)
  out = {}
  for fam in ('CF', 'CF2', 'MF'):
    G = refs[fam]; GR = region_of(G.astype(np.float64)); obs_g = np.concatenate([np.zeros((len(G), STATE_DIM), np.float32), G], axis=1)
    d_task = np.linalg.norm(G[None, :, :] - S['goal_xy'][:, None, :].astype(np.float32), axis=2)          # [n_states, n_ref]
    masks = {'goal_area': np.broadcast_to(GR == REGIONS.index('goal_area'), d_task.shape), 'near2.0': d_task <= 2.0, 'reach0.5': d_task <= REACH_R}
    for sd in range(5):
      qp, _ = N.load(DP.ckpt_of(fam, sd)); res = {}
      for c in cands:
        ok = np.all(np.isfinite(a0[c]), axis=1); pr = {k: np.full(n_states, np.nan) for k in masks}
        if ok.any():
          f = N.logits(qp, obs[ok], a0[c][ok], obs_g).mean(-1); f = f - f.max(axis=1, keepdims=True); pmat = np.exp(f); pmat /= pmat.sum(axis=1, keepdims=True)
          for k, m in masks.items():
            pr[k][ok] = (pmat * m[ok]).sum(axis=1)
        res[c] = pr
      out[f'{fam}_s{sd}'] = res
    out[f'{fam}_ref_counts'] = {'goal_area': int((GR == REGIONS.index('goal_area')).sum()), 'near2.0_median_per_state': float(np.median((d_task <= 2.0).sum(axis=1))), 'reach0.5_median_per_state': float(np.median((d_task <= REACH_R).sum(axis=1)))}
  return out


def mode_crossover(args):
  S = np.load(OUT / 'states.npz', allow_pickle=False); n_states = len(S['t']); groups = S['group'].astype(str)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  conts = {}
  for cont in ('start', 'CF'):
    R, files = load_rollouts(cont); cands, keys, M, D, Dse = per_state_candidate(R, n_states); conts[cont] = {'R': R, 'files': files, 'cands': cands, 'M': M, 'D': D}
  cands = conts['start']['cands']
  a0 = {}
  R0 = conts['start']['R']
  for c in cands:
    a0[c] = np.full((n_states, ACTION_DIM), np.nan, np.float32); sel = (R0['cand'].astype(str) == c) & (R0['cls'] == 0) & (R0['rep'] == 0); a0[c][R0['state'][sel]] = R0['a0'][sel]
  for c in cands:                                                                # the CF-continuation runs must have executed the same torques
    R1 = conts['CF']['R']; sel = (R1['cand'].astype(str) == c) & (R1['cls'] == 0) & (R1['rep'] == 0); ok = np.all(np.isfinite(a0[c][R1['state'][sel]]), axis=1)
    assert np.allclose(R1['a0'][sel][ok], a0[c][R1['state'][sel]][ok], atol=1e-6), f'candidate torque differs across continuations: {c}'
  refs = reference_goal_sets(anchors, np.random.default_rng(13))
  P = critic_region_probs(S, cands, a0, refs)
  res = {'files': {c: conts[c]['files'] for c in conts}, 'n_states': int(n_states), 'groups': {str(g): int((groups == g).sum()) for g in np.unique(groups)},
         'reference_counts': {fam: P[f'{fam}_ref_counts'] for fam in ('CF', 'CF2', 'MF')}, 'by_group': {}}
  keys = ('success', 'death', 'timeout', 'far_entry', 'far_success', 'goal_area', 'near2.0', 'reach0.5', 'far', 'death_frame')
  for g in ('reset', 'start_early', 'pre_zone1_early', 'indep_reset', 'indep_early', 'cf_early'):
    m = groups == g
    if not m.any():
      continue
    row = {'n_states': int(m.sum())}
    for cont in ('start', 'CF'):
      M, D = conts[cont]['M'], conts[cont]['D']; cr = {'candidate_means': {}, 'advantage_vs_start': {}}
      for c in cands:
        ok = m & np.isfinite(M[c]['success'])
        if not ok.any():
          continue
        cr['candidate_means'][c] = {k: float(np.nanmean(M[c][k][ok])) for k in keys}
        cr['candidate_means'][c]['far_completion'] = (float(np.nansum(M[c]['far_success'][ok]) / np.nansum(M[c]['far_entry'][ok])) if np.nansum(M[c]['far_entry'][ok]) > 0 else None)
        if c != 'start':
          cr['advantage_vs_start'][c] = {k: {'mean': float(D[c][k][ok].mean()), 'state_se': float(D[c][k][ok].std(ddof=1) / np.sqrt(ok.sum())), 'share_positive': float((D[c][k][ok] > 0).mean()), 'share_negative': float((D[c][k][ok] < 0).mean()), 'share_zero': float((D[c][k][ok] == 0).mean())} for k in keys}
      row[cont] = cr
    # the critics' region-level ranking vs the measured region masses under each continuation
    rec = {}
    for fam in ('CF', 'CF2', 'MF'):
      for c in cands:
        if c == 'start':
          continue
        entry = {}
        for region in ('goal_area', 'near2.0', 'reach0.5'):
          margins = []
          for sd in range(5):
            pr = P[f'{fam}_s{sd}']; ok = m & np.isfinite(pr[c][region]) & np.isfinite(pr['start'][region]); margins.append((pr[c][region] - pr['start'][region], ok))
          e = {}
          for cont in ('start', 'CF'):
            D = conts[cont]['D']; cs, sa, mm = [], [], []
            for marg, ok in margins:
              ok2 = ok & np.isfinite(D[c][region]); dm = D[c][region][ok2]; mg = marg[ok2]
              if ok2.sum() >= 10 and dm.std() > 1e-12 and mg.std() > 1e-12:
                cs.append(float(np.corrcoef(mg, dm)[0, 1])); nz = dm != 0; sa.append(float(((mg > 0) == (dm > 0))[nz].mean()) if nz.any() else np.nan); mm.append(float(mg.mean()))
            e[cont] = {'corr_margin_vs_measured': (float(np.mean(cs)) if cs else None), 'sign_agreement': (float(np.nanmean(sa)) if sa else None), 'critic_margin_mean': (float(np.mean(mm)) if mm else None), 'n_seeds': len(cs)}
          entry[region] = e
        rec[f'{fam}:{c}'] = entry
    row['critic_region_ranking'] = rec
    res['by_group'][g] = row
  MP.write_json(OUT / 'report_crossover.json', res)
  (OUT / 'REPORT_crossover.md').write_text(write_crossover_md(res, cands), encoding='utf-8')
  print(json.dumps({g: {cont: {c: round(res['by_group'][g][cont]['advantage_vs_start'][c]['success']['mean'], 3) for c in res['by_group'][g][cont]['advantage_vs_start']} for cont in ('start', 'CF')} for g in res['by_group']}, indent=1), flush=True)


def write_crossover_md(res, cands):
  L = ['# Crossover: the same states, candidates and hidden seeds under the start continuation and the frozen CF s0 continuation (every candidate with the same continuation)', '',
       f"States {res['groups']}; rollout files {res['files']}; reference goal sets (4096 from each family's own training marginal): counts {res['reference_counts']}.", '']
  for g, row in res['by_group'].items():
    L += [f"## {g} (n states {row['n_states']})", '', '| continuation | candidate | success | death | timeout | far entry | far completion | goal_area mass | near2.0 mass | reach0.5 mass | success adv vs start (mean +- s.e.; >0 / <0 / =0) | goal_area adv | near2.0 adv | reach0.5 adv |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for cont in ('start', 'CF'):
      if cont not in row:
        continue
      for c, mm in row[cont]['candidate_means'].items():
        adv = row[cont]['advantage_vs_start'].get(c)
        fc = '' if mm['far_completion'] is None else f"{mm['far_completion']:.2f}"
        cell = '' if adv is None else f"{adv['success']['mean']:+.3f} +- {adv['success']['state_se']:.3f} ({adv['success']['share_positive']:.2f} / {adv['success']['share_negative']:.2f} / {adv['success']['share_zero']:.2f})"
        cell2 = '' if adv is None else f"{adv['goal_area']['mean']:+.4f} +- {adv['goal_area']['state_se']:.4f}"
        cell3 = '' if adv is None else f"{adv['near2.0']['mean']:+.4f} +- {adv['near2.0']['state_se']:.4f}"
        cell4 = '' if adv is None else f"{adv['reach0.5']['mean']:+.4f} +- {adv['reach0.5']['state_se']:.4f}"
        L.append(f"| {cont} | {c} | {mm['success']:.3f} | {mm['death']:.3f} | {mm['timeout']:.3f} | {mm['far_entry']:.3f} | {fc} | {mm['goal_area']:.4f} | {mm['near2.0']:.4f} | {mm['reach0.5']:.4f} | {cell} | {cell2} | {cell3} | {cell4} |")
    L += ['', '| critic : candidate | region | critic margin (cand - start) | corr with the measured mass adv, start cont. | sign agr. | corr, CF cont. | sign agr. |', '|---|---|---:|---:|---:|---:|---:|']
    for key, entry in row['critic_region_ranking'].items():
      for region, e in entry.items():
        f = lambda v: 'n/a' if v is None else f'{v:.3f}'
        L.append(f"| {key} | {region} | {f(e['start']['critic_margin_mean'])} | {f(e['start']['corr_margin_vs_measured'])} | {f(e['start']['sign_agreement'])} | {f(e['CF']['corr_margin_vs_measured'])} | {f(e['CF']['sign_agreement'])} |")
    L.append('')
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('states', 'generate', 'report', 'crossover'))
  ap.add_argument('--add-cf-early', action='store_true', help='states: append the 64 cf_early states (the indep resets rolled 20 steps by the CF s0 actor)')
  ap.add_argument('--continuation', choices=('start', 'CF'), default='start', help='generate: the frozen continuation policy for every candidate')
  ap.add_argument('--groups', nargs='*', default=None, help='generate: restrict to these state groups')
  ap.add_argument('--tag', default=None)
  ap.add_argument('--workers', type=int, default=18)
  ap.add_argument('--draws-per-class', type=int, default=DRAWS_PER_CLASS)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'states': mode_states, 'generate': mode_generate, 'report': mode_report, 'crossover': mode_crossover}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
