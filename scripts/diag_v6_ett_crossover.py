#!/usr/bin/env python
"""AntMaze V6 -- the learned v4 ETT on the CROSSOVER (user's plan after 48d603c, 2026-09-21; no training).

The simulator crossover (repeated_draws/) showed that the first step and the continuation must match: at the reset states CF's first step
+ start continuation 0.234, CF's first step + CF continuation 0.531, the logged first step + CF continuation 0.228.  A learned ETT that only
reproduces "logged step + start continuation" cannot supply that.  This diagnostic replays the SAME fixed states and candidates through
the v4 learned ETT (motion / stationary on (s, a_q), the onset head with exact path-local history counters, the advice generator v3 with
its hidden context from the prior) instead of the simulator, with the continuation policy (start or the frozen CF s0 actor) acting on the
MODEL's predicted state, and asks three things:
  Q1  is the initial turn predicted right?  the first 30 steps' xy / pose / velocity errors against the simulator's deterministic early
      trajectory for the same (state, first torque, continuation), and whether the closed loop enters the far route;
  Q2  after entering the far route, is completion vs stall predicted right?  (success | far entry, timeout | far entry) model vs simulator;
  Q3  is the same-state ACTION difference preserved?  the four corners (first step x continuation) and the paired contrasts, e.g.
      (CF first + CF cont) - (logged first + CF cont), model vs simulator, as means over draws and per-state correlations.
Hidden context: the four activity classes x 8 clock draws (32 per state x candidate x continuation), prior weight 1/4 per class, as the
simulator runs; the onset sampled per step; every outcome kept; the history counters come from each model path itself (the logged prefix
for training anchors, the actual start / CF prefix for the t = 20 independent states).  Models of the anchor's fold for training states;
fold 0 for the independent states (disclosed).  Outputs under ett_crossover/.
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
import exp_v6_repeated_draws as RD  # noqa: E402
import fit_v6_ett_advice as FA  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from fit_v6_ett_one_step_v2 import in_band  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

OUT = MP.OUT / 'ett_crossover'
MODEL_DIR = None                # --model-dir: the one-step model dir (default the v4 models); --tag names the output subdir
STATE_DIM, GOAL_DIM, OBS_W, ACTION_DIM, HORIZON = MP.STATE_DIM, MP.GOAL_DIM, MP.OBS_W, MP.ACTION_DIM, MP.HORIZON
REACH_R = RD.REACH_R
CANDS = ('logged', 'start', 'CF', 'MF', 'CF2')
CONTS = ('start', 'CF')
DRAWS_PER_CLASS = 8
EARLY_K = 30
SEED = 306_000_000


def policy_mode_batched(ckpt):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt); pp = st.policy_params
  f = jax.jit(lambda o: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  return lambda o31: np.asarray(f(jnp.asarray(o31, jnp.float32)), np.float32)


# ------------------------------------------------------------------ the fixed states and their prefixes
def prepare_states():
  """The repeated-draws state set with what the model needs at the root: fold, the history counters and the previous advice from the ACTUAL
  prefix (logged for training anchors; the start / CF prefix re-rolled for the t = 20 independent states), the first-mouth prefix."""
  import build_v6_branch_replay as B
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); n = len(S['t']); groups = S['group'].astype(str)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  sup = np.load(FA.EC / 'supervision.npz', allow_pickle=False); fold_anchor = sup['fold_anchor'].astype(np.int64)
  fold = np.zeros(n, np.int64); kh0 = np.zeros(n, np.int64); prev_ab = np.zeros((n, ACTION_DIM), np.float32); m0 = np.full((n, 2), -1, np.int64)
  for i in range(n):
    k = int(S['anchor_id'][i])
    if k >= 0:
      e, t = int(anchors.episode[k]), int(anchors.t[k]); fold[i] = fold_anchor[k]
      kh0[i] = FA.zero_run_before(act, np.array([e]), np.array([t]))[0]; prev_ab[i] = act[e, t - 1] if t >= 1 else 0.0; m0[i] = FA.first_mouth_in_prefix(obs[e], t)
  # the independent t = 20 states: replay the resets and the 19 prefix steps of the respective policy for the previous executed torque
  # one env instance per prefix policy, both replaying the same reset stream (a fresh reset each time, as the states were made)
  pol = {'indep_early': MP.mode_policy(MP.START_CKPT), 'cf_early': MP.mode_policy(RD.ckpt('CF'))}
  idx = {g: np.nonzero(groups == g)[0] for g in ('indep_reset', 'indep_early', 'cf_early')}
  for g in ('indep_early', 'cf_early'):
    B._ENV = None; env, _ = B._worker_env(RD.INDEP_ENV_SEED0)
    for i in range(RD.N_PER_GROUP):
      o = env.reset()
      assert np.abs(o[:STATE_DIM] - S['state'][idx['indep_reset'][i]]).max() < 1e-5
      a = None
      for j in range(20):
        a = np.asarray(pol[g](o), np.float32); o, _, _, _ = env.step(a)
      assert np.abs(o[:STATE_DIM] - S['state'][idx[g][i]]).max() < 1e-4, g
      prev_ab[idx[g][i]] = a
  return S, fold, kh0, prev_ab, m0


# ------------------------------------------------------------------ the model rollouts
def rollout_model(P, gen, mode_fn, s0, goal, a0, t0, z, kh0, prev_ab0, m0, rng, early_k=EARLY_K, death_u=None, onset_off=False, max_steps_cap=None):
  """n paths stepped together through the fold's models (as exp_v6_ett_futures.generate_fold) with an arbitrary first torque a0 and an
  arbitrary continuation policy on the predicted state; returns outcome, length, far entry, the path masses and the first early_k
  predicted states."""
  n = len(s0); s = s0.astype(np.float32).copy(); t = np.asarray(t0, np.int64)
  st = FA.PathState(kh0, prev_ab0, m0)
  max_steps = HORIZON - t
  if max_steps_cap is not None:
    max_steps = np.minimum(max_steps, int(max_steps_cap))
  alive = np.ones(n, bool); outcome = np.full(n, 'timeout', dtype='<U7'); L = np.ones(n, np.int64)
  xy_rows = [[s0[i, :2].copy()] for i in range(n)]
  early = np.full((n, early_k + 1, STATE_DIM), np.nan, np.float32); early[:, 0] = s0
  idx_all = np.arange(n)
  for j in range(int(max_steps.max())):
    active = alive & (j < max_steps)
    if not active.any():
      break
    o31 = np.concatenate([s, goal], axis=1)
    aq = a0 if j == 0 else mode_fn(o31)
    t_abs = t + j
    band_now = in_band(s[:, :2].astype(np.float64)); kb_now = st.band_counter(idx_all, j, band_now)
    m1, m2 = st.mouth_update(idx_all, t_abs, s[:, :2].astype(np.float64))
    ctx = FA.context_features(t_abs, *z); mf = FA.mouth_features(t_abs, m1, m2, z[2], z[3])
    x = FA.features(o31, ctx, mf, st.kh_prev, kb_now, st.prev_ab, gen.norm)
    hold, ab, _ = gen.sample(x, rng, decision='map'); kh_now = st.hold_counter(idx_all, j, hold)
    s_next, p_on, _ = P.step(s, ab, aq, kh_now, kb_now)
    u_death = rng.random(n) if death_u is None else death_u[:, j]                   # paired across candidates / continuations when death_u is given
    died = active & (u_death < p_on) & (not onset_off)
    reached = active & ~died & (np.linalg.norm(s_next[:, :2] - goal, axis=1) <= REACH_R)
    for i in np.flatnonzero(active):
      xy_rows[i].append(s_next[i, :2].copy())
    if j < early_k:
      early[active, j + 1] = s_next[active]
    L[active] += 1
    outcome[died] = 'death'; outcome[reached] = 'success'; alive &= ~(died | reached)
    st.advance(idx_all, hold, ab)
    s = np.where(active[:, None], s_next, s).astype(np.float32)
  masses = {k: np.zeros(n) for k in ('reach0.5', 'near2.0', 'goal_area', 'far', 'death_frame')}
  far_entry = np.zeros(n, bool); zone1_entry = np.zeros(n, bool)
  for i in range(n):
    xy = np.stack(xy_rows[i]); reg = region_of(xy.astype(np.float64)); far_entry[i] = np.isin(reg, FAR).any(); zone1_entry[i] = (reg == REGIONS.index('zone1')).any()
    for k, v in RD.path_masses(xy, goal[i].astype(np.float64), outcome[i]).items():
      masses[k][i] = v
  return {'outcome': outcome, 'length': L, 'far_entry': far_entry, 'zone1_entry': zone1_entry, 'early': early, **masses}


def paired_draws(states, cls, reps):
  """Hidden draws keyed by (state, class, rep) only -- identical for every candidate torque and continuation: the two clocks and the
  per-step death uniforms."""
  t01 = np.zeros(len(states), np.int64); t02 = np.zeros(len(states), np.int64); U = np.zeros((len(states), HORIZON), np.float32)
  for i, (si, ci, r) in enumerate(zip(states, cls, reps)):
    g = np.random.default_rng([SEED, int(si), int(ci), int(r)])
    t01[i] = g.integers(FA.Z_PRIOR['t0_1'][0], FA.Z_PRIOR['t0_1'][1] + 1); t02[i] = g.integers(FA.Z_PRIOR['t0_2'][0], FA.Z_PRIOR['t0_2'][1] + 1); U[i] = g.random(HORIZON)
  return t01, t02, U


def mode_run(args):
  OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
  S, fold, kh0, prev_ab, m0 = prepare_states(); n_states = len(S['t']); groups = S['group'].astype(str)
  print('states', {g: int((groups == g).sum()) for g in np.unique(groups)}, 'folds', np.bincount(fold).tolist(), flush=True)
  F2.OUT = Path(MODEL_DIR) if MODEL_DIR else F2.OUT_V4
  P = {f: F2.Predictor2(F2.load_model(f)) for f in range(FA.N_FOLDS)}; G = {f: FA.load_generator(f) for f in range(FA.N_FOLDS)}
  modes = {'start': policy_mode_batched(MP.START_CKPT), 'CF': policy_mode_batched(RD.ckpt('CF'))}
  pol1 = {c: policy_mode_batched(RD.ckpt(c)) for c in ('start', 'CF', 'MF', 'CF2')}
  obs31 = np.concatenate([S['state'], S['goal_xy']], axis=1).astype(np.float32)
  a0 = {'logged': S['logged'].astype(np.float32)}
  for c in ('start', 'CF', 'MF', 'CF2'):
    a0[c] = pol1[c](obs31)
  states = np.arange(n_states) if not args.limit else np.arange(int(args.limit))
  rows = []
  for cont in CONTS:
    for c in CANDS:
      ok = states[np.all(np.isfinite(a0[c][states]), axis=1)]
      if len(ok) == 0:
        continue
      for f in range(FA.N_FOLDS):
        sf = ok[fold[ok] == f]
        if len(sf) == 0:
          continue
        # 4 classes x 8 draws per state, all in one batch
        rep = np.repeat(sf, 4 * DRAWS_PER_CLASS); cls = np.tile(np.repeat(np.arange(4), DRAWS_PER_CLASS), len(sf)); r = np.tile(np.tile(np.arange(DRAWS_PER_CLASS), 4), len(sf))
        rng = np.random.default_rng(SEED + 1000 * f + 10 * CONTS.index(cont) + CANDS.index(c))          # the generator's torque samples only
        u1 = np.array([RD.CLASSES[k][0] for k in cls]); u2 = np.array([RD.CLASSES[k][1] for k in cls])
        t01, t02, U = paired_draws(rep, cls, r)
        res = rollout_model(P[f], G[f], modes[cont], S['state'][rep], S['goal_xy'][rep].astype(np.float32), a0[c][rep], S['t'][rep], (u1, u2, t01, t02), kh0[rep], prev_ab[rep], m0[rep], rng, death_u=U)
        for i in range(len(rep)):
          rows.append({'state': int(rep[i]), 'cand': c, 'cont': cont, 'cls': int(cls[i]), 'rep': int(r[i]), 'outcome': str(res['outcome'][i]), 'length': int(res['length'][i]), 'far_entry': bool(res['far_entry'][i]), 'zone1_entry': bool(res['zone1_entry'][i]),
                       **{k: float(res[k][i]) for k in ('reach0.5', 'near2.0', 'goal_area', 'far', 'death_frame')}})
        print(f'{cont} cont, {c} first, fold {f}: {len(sf)} states x 32, {time.time() - t0:.0f} s', flush=True)
  # the motion-only early pass (onset off, exactly 30 steps) for Q1
  early_store = {}
  for cont in CONTS:
    for c in CANDS:
      ok = states[np.all(np.isfinite(a0[c][states]), axis=1)]
      for f in range(FA.N_FOLDS):
        sf = ok[fold[ok] == f]
        if len(sf) == 0:
          continue
        z0 = (np.zeros(len(sf), bool), np.zeros(len(sf), bool), np.full(len(sf), 30), np.full(len(sf), 110))
        res = rollout_model(P[f], G[f], modes[cont], S['state'][sf], S['goal_xy'][sf].astype(np.float32), a0[c][sf], S['t'][sf], z0, kh0[sf], prev_ab[sf], m0[sf], np.random.default_rng(1), onset_off=True, max_steps_cap=EARLY_K)
        for i, si in enumerate(sf):
          early_store[(int(si), c, cont)] = res['early'][i]
  keys = list(rows[0].keys())
  arr = {k: np.array([row[k] for row in rows]) for k in keys}
  ek = sorted(early_store); early = np.stack([early_store[k] for k in ek])
  np.savez_compressed(OUT / 'model_rollouts.npz', **arr, early_state=np.array([k[0] for k in ek]), early_cand=np.array([k[1] for k in ek]), early_cont=np.array([k[2] for k in ek]), early=early,
                      a0=np.stack([a0[c] for c in CANDS]), meta=np.asarray(json.dumps({'draws_per_class': DRAWS_PER_CLASS, 'seed': SEED, 'ett': (str(MODEL_DIR) if MODEL_DIR else 'v4'), 'folds': 'anchor fold for training states; fold 0 for independent', 'early_k': EARLY_K, 'paired_draws': 'clocks and death uniforms keyed by (state, class, rep)', 'early_pass': 'motion only, onset off, exactly 30 steps'})))
  print(f'{len(rows)} model paths in {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------ the simulator's early trajectories (deterministic; hazards off)
def sim_early(S, a0, early_k=EARLY_K):
  import build_v6_branch_replay as B
  env, _ = B._worker_env(RD.WORKER_SEED0 + 999)
  pol = {'start': MP.mode_policy(MP.START_CKPT), 'CF': MP.mode_policy(RD.ckpt('CF'))}
  out = {}
  for si in range(len(S['t'])):
    for ci, c in enumerate(CANDS):
      a = a0[ci, si]
      if not np.all(np.isfinite(a)):
        continue
      for cont in CONTS:
        o = RD.restore_forced(env, S['state'][si].astype(np.float64), S['goal_xy'][si].astype(np.float64), int(S['t'][si]), False, False)
        tr = np.full((early_k + 1, STATE_DIM), np.nan, np.float32); tr[0] = S['state'][si]
        o, _, done, info = env.step(np.asarray(a, np.float32)); tr[1] = o[:STATE_DIM]
        for j in range(1, early_k):
          if done:
            break
          o, _, done, info = env.step(np.asarray(pol[cont](o), np.float32)); tr[j + 1] = o[:STATE_DIM]
        out[(si, c, cont)] = tr
  return out


def mode_report(args):
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); n_states = len(S['t']); groups = S['group'].astype(str)
  Mz = np.load(OUT / 'model_rollouts.npz', allow_pickle=False); meta = json.loads(str(Mz['meta']))
  R = {k: Mz[k] for k in Mz.files if k not in ('meta', 'early', 'early_state', 'early_cand', 'early_cont', 'a0')}
  # ---- model per-state means (class-weighted) per (cand, cont)
  oc = R['outcome'].astype(str); cand = R['cand'].astype(str); cont = R['cont'].astype(str)
  val = {'success': (oc == 'success').astype(float), 'death': (oc == 'death').astype(float), 'timeout': (oc == 'timeout').astype(float), 'far_entry': R['far_entry'].astype(float),
         'far_success': ((oc == 'success') & R['far_entry']).astype(float), 'far_timeout': ((oc == 'timeout') & R['far_entry']).astype(float), 'goal_area': R['goal_area'], 'near2.0': R['near2.0'], 'reach0.5': R['reach0.5'], 'length': R['length'].astype(float)}
  keys = list(val)
  Mm = {(c, k2): {key: np.full(n_states, np.nan) for key in keys} for c in CANDS for k2 in CONTS}
  for c in CANDS:
    for k2 in CONTS:
      sel = (cand == c) & (cont == k2)
      for si in np.unique(R['state'][sel]):
        s2 = sel & (R['state'] == si)
        for key in keys:
          Mm[(c, k2)][key][si] = np.mean([val[key][s2 & (R['cls'] == kk)].mean() for kk in range(4)])
  # ---- simulator per-state means from the crossover rollouts
  Ms = {}
  for k2 in CONTS:
    Rs, files = RD.load_rollouts(k2); cands_s, keys_s, M_s, D_s, _ = RD.per_state_candidate(Rs, n_states)
    ocs = Rs['outcome'].astype(str)
    for c in CANDS:
      Ms[(c, k2)] = {key: M_s[c][key] for key in ('success', 'death', 'timeout', 'far_entry', 'far_success', 'goal_area', 'near2.0', 'reach0.5')}
      # far timeout share and mean length from the raw rollouts (class-weighted)
      ft = np.full(n_states, np.nan); ln = np.full(n_states, np.nan)
      sel = Rs['cand'].astype(str) == c
      for si in np.unique(Rs['state'][sel]):
        s2 = sel & (Rs['state'] == si)
        ft[si] = np.mean([((ocs[s2 & (Rs['cls'] == kk)] == 'timeout') & Rs['far_entry'][s2 & (Rs['cls'] == kk)]).mean() for kk in range(4)])
        ln[si] = np.mean([Rs['steps'][s2 & (Rs['cls'] == kk)].mean() + 1 for kk in range(4)])
      Ms[(c, k2)]['far_timeout'] = ft; Ms[(c, k2)]['length'] = ln
  # ---- Q1: early trajectories
  a0 = Mz['a0']; sim_e = sim_early(S, a0)
  es, ec, ecn, E = Mz['early_state'], Mz['early_cand'].astype(str), Mz['early_cont'].astype(str), Mz['early']
  q1 = {}
  for g in ('reset', 'start_early', 'pre_zone1_early', 'indep_reset', 'indep_early', 'cf_early'):
    m = groups == g; row = {}
    for c in ('logged', 'start', 'CF'):
      for k2 in CONTS:
        errs = {'xy': [], 'pose': [], 'vel': [], 'xy_at_30': [], 'north_sim': [], 'north_model': []}
        for i in np.flatnonzero((ec == c) & (ecn == k2)):
          si = int(es[i])
          if not m[si] or (si, c, k2) not in sim_e:
            continue
          tm, ts = E[i], sim_e[(si, c, k2)]
          ok = np.all(np.isfinite(tm), axis=1) & np.all(np.isfinite(ts), axis=1)
          if ok.sum() < 5 or not ok[-1]:
            continue                                                            # exactly step 30 required (motion-only pass: no early termination expected)
          d = tm[ok] - ts[ok]
          errs['xy'].append(np.linalg.norm(d[:, :2], axis=1)); errs['pose'].append(np.abs(d[:, 2:15]).mean(axis=1)); errs['vel'].append(np.abs(d[:, 15:29]).mean(axis=1))
          if ok[-1]:
            errs['xy_at_30'].append(float(np.linalg.norm(d[-1, :2])))
          errs['north_sim'].append(float(ts[-1, 1] > 1.5)); errs['north_model'].append(float(tm[-1, 1] > 1.5))
        if errs['xy']:
          steps = (1, 5, 10, 20, 30)
          def at(k, j):
            v = [e[j] for e in errs[k] if len(e) > j]; return float(np.mean(v)) if v else None
          row[f'{c}+{k2}'] = {'n': len(errs['xy']), 'xy_err_by_step': {j: at('xy', j) for j in steps}, 'pose_err_by_step': {j: at('pose', j) for j in steps}, 'vel_err_by_step': {j: at('vel', j) for j in steps},
                             'heading_north_at_30_sim': float(np.mean(errs['north_sim'])), 'heading_north_at_30_model': float(np.mean(errs['north_model'])),
                             'heading_agreement_at_30': float(np.mean(np.array(errs['north_sim']) == np.array(errs['north_model'])))}
    q1[g] = row
  # ---- Q2 / Q3 tables
  def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 10 and a[ok].std() > 1e-9 and b[ok].std() > 1e-9 else None
  q23 = {}
  for g in ('reset', 'start_early', 'pre_zone1_early', 'indep_reset', 'indep_early', 'cf_early'):
    m = groups == g; row = {'corners': {}, 'contrasts': {}}
    for c in CANDS:
      for k2 in CONTS:
        ok = m & np.isfinite(Mm[(c, k2)]['success']) & np.isfinite(Ms[(c, k2)]['success'])
        if ok.sum() == 0:
          continue
        e = {}
        for key in ('success', 'death', 'timeout', 'far_entry', 'far_success', 'far_timeout', 'goal_area', 'near2.0', 'reach0.5', 'length'):
          e[key] = {'model': float(np.nanmean(Mm[(c, k2)][key][ok])), 'sim': float(np.nanmean(Ms[(c, k2)][key][ok])), 'corr_states': corr(Mm[(c, k2)][key][ok], Ms[(c, k2)][key][ok])}
        fe_m, fe_s = np.nansum(Mm[(c, k2)]['far_entry'][ok]), np.nansum(Ms[(c, k2)]['far_entry'][ok])
        e['completion_given_far'] = {'model': (float(np.nansum(Mm[(c, k2)]['far_success'][ok]) / fe_m) if fe_m > 0 else None), 'sim': (float(np.nansum(Ms[(c, k2)]['far_success'][ok]) / fe_s) if fe_s > 0 else None)}
        e['timeout_given_far'] = {'model': (float(np.nansum(Mm[(c, k2)]['far_timeout'][ok]) / fe_m) if fe_m > 0 else None), 'sim': (float(np.nansum(Ms[(c, k2)]['far_timeout'][ok]) / fe_s) if fe_s > 0 else None)}
        e['n'] = int(ok.sum())
        row['corners'][f'{c}+{k2}'] = e
    for (ca, ka, cb, kb) in (('CF', 'CF', 'logged', 'CF'), ('CF', 'CF', 'start', 'CF'), ('CF', 'start', 'start', 'start'), ('CF', 'CF', 'CF', 'start'), ('start', 'CF', 'start', 'start'), ('logged', 'CF', 'logged', 'start'), ('CF2', 'CF', 'start', 'CF'), ('MF', 'CF', 'start', 'CF')):
      ok = m & np.isfinite(Mm[(ca, ka)]['success']) & np.isfinite(Mm[(cb, kb)]['success']) & np.isfinite(Ms[(ca, ka)]['success']) & np.isfinite(Ms[(cb, kb)]['success'])
      if ok.sum() < 10:
        continue
      e = {}
      for key in ('success', 'far_entry', 'goal_area', 'near2.0'):
        dm = Mm[(ca, ka)][key][ok] - Mm[(cb, kb)][key][ok]; ds = Ms[(ca, ka)][key][ok] - Ms[(cb, kb)][key][ok]
        e[key] = {'model_mean': float(dm.mean()), 'model_se': float(dm.std(ddof=1) / np.sqrt(len(dm))), 'sim_mean': float(ds.mean()), 'sim_se': float(ds.std(ddof=1) / np.sqrt(len(ds))), 'corr_states': corr(dm, ds),
                  'sign_agreement': (float(((dm > 0) == (ds > 0))[ds != 0].mean()) if (ds != 0).any() else None)}
      row['contrasts'][f'({ca}+{ka}) - ({cb}+{kb})'] = e
    q23[g] = row
  res = {'meta': meta, 'groups': {str(g): int((groups == g).sum()) for g in np.unique(groups)}, 'Q1_early': q1, 'Q2_Q3': q23}
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res), encoding='utf-8')
  print('report written', flush=True)


def write_md(res):
  L = [f"# The learned ETT ({res['meta'].get('ett', 'v4')}) on the crossover: model vs simulator at the same states, first torques and continuations", '', f"States {res['groups']}; {res['meta']['draws_per_class']} draws per class x 4 classes per (state, first torque, continuation); models of the anchor's fold (fold 0 for the independent states).", '']
  L += ['## Q1 -- the early trajectory (first 30 steps; the simulator is deterministic here, hazards off): mean errors model - simulator', '', '| group | first + cont | n | xy err at 1 / 5 / 10 / 20 / 30 | pose err (mean abs, dims 2-14) at 1 / 10 / 30 | vel err (dims 15-28) at 1 / 10 / 30 | heading north (y > 1.5) at step 30: sim / model / agreement |', '|---|---|---:|---|---|---|---|']
  for g, row in res['Q1_early'].items():
    for key, e in row.items():
      f = lambda d, js: ' / '.join(('-' if d[j] is None else f'{d[j]:.3f}') for j in js)
      L.append(f"| {g} | {key} | {e['n']} | {f(e['xy_err_by_step'], (1, 5, 10, 20, 30))} | {f(e['pose_err_by_step'], (1, 10, 30))} | {f(e['vel_err_by_step'], (1, 10, 30))} | {e['heading_north_at_30_sim']:.2f} / {e['heading_north_at_30_model']:.2f} / {e['heading_agreement_at_30']:.2f} |")
  L += ['', '## Q2 / Q3 -- the corners (first torque x continuation): model vs simulator, per-state means and the across-state correlation', '', '| group | first + cont | n | success model / sim (corr) | far entry model / sim (corr) | completion given far model / sim | timeout given far model / sim | goal_area model / sim (corr) | near2.0 model / sim (corr) | length model / sim |', '|---|---|---:|---|---|---|---|---|---|---|']
  for g, row in res['Q2_Q3'].items():
    for key, e in row['corners'].items():
      c = lambda k: ('-' if e[k]['corr_states'] is None else f"{e[k]['corr_states']:.2f}")
      cg = lambda d: ' / '.join(('-' if d[k] is None else f'{d[k]:.2f}') for k in ('model', 'sim'))
      L.append(f"| {g} | {key} | {e['n']} | {e['success']['model']:.3f} / {e['success']['sim']:.3f} ({c('success')}) | {e['far_entry']['model']:.3f} / {e['far_entry']['sim']:.3f} ({c('far_entry')}) | {cg(e['completion_given_far'])} | {cg(e['timeout_given_far'])} | {e['goal_area']['model']:.4f} / {e['goal_area']['sim']:.4f} ({c('goal_area')}) | {e['near2.0']['model']:.4f} / {e['near2.0']['sim']:.4f} ({c('near2.0')}) | {e['length']['model']:.0f} / {e['length']['sim']:.0f} |")
  L += ['', '## Q3 -- the same-state contrasts: model vs simulator (mean +- state s.e.; across-state corr; sign agreement)', '', '| group | contrast | success model / sim | far entry model / sim | goal_area model / sim | near2.0 model / sim |', '|---|---|---|---|---|---|']
  for g, row in res['Q2_Q3'].items():
    for key, e in row['contrasts'].items():
      cell = lambda k: f"{e[k]['model_mean']:+.3f}+-{e[k]['model_se']:.3f} / {e[k]['sim_mean']:+.3f}+-{e[k]['sim_se']:.3f} (r {('-' if e[k]['corr_states'] is None else format(e[k]['corr_states'], '.2f'))}, sign {('-' if e[k]['sign_agreement'] is None else format(e[k]['sign_agreement'], '.2f'))})"
      L.append(f"| {g} | {key} | {cell('success')} | {cell('far_entry')} | {cell('goal_area')} | {cell('near2.0')} |")
  L.append('')
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('run', 'report')); ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--model-dir', default=None, help='one-step model dir (default the v4 models)'); ap.add_argument('--tag', default=None, help='output subdir under ett_crossover/')
  args = ap.parse_args(argv)
  global MODEL_DIR, OUT
  MODEL_DIR = args.model_dir
  if args.tag:
    OUT = OUT / args.tag
  {'run': mode_run, 'report': mode_report}[args.mode](args); return 0


if __name__ == '__main__':
  sys.exit(main())
