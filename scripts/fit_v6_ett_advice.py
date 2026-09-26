#!/usr/bin/env python
"""AntMaze V6 one-step ETT -- the ADVICE GENERATOR (user's decision 2026-09-20): a_b along a path from the state, its own
history and a persistent hidden hazard context, learnt from the branch supervision; the motion model and the onset head frozen.

Oracle-supervised engineering revision (disclosed): the supervision is the simulator teacher's advice walked along every
pilot branch under the branch's redrawn timetable (ett_context/supervision.npz), and the timetable of every branch is
recorded (branch_advice.npz: u1 / u2 / t0_1 / t0_2 redrawn per anchor), so the hidden context is OBSERVED in training.
At generation a path draws its own context from the prior (u ~ Bernoulli(0.5) per zone, t0_1 ~ U{10..45}, t0_2 ~
U{90..125}; the environment's design, checked on the 1,000 logged contexts and the 53,747 redraws) and keeps it for the
whole path; nothing is read from the evaluation episode's actual hidden draw.  No hand-forced holds: a hold arises only
from the learnt map.

Model (per fold, cross-fitted by the supervision folds; fixed final iterate, no selection):
  inputs   s (29, standardised), goal xy (2), t / 800, u1, u2, (t - t0_z) / 72 clipped, in-burst flags (2; a function of the
           context and the clock), the consecutive holds ending at the PREVIOUS row (at the root: the logged zero-torque run
           before t), the exact band counter of the current row (state-based), the previous advice torque (teacher-forced in
           training; the generator's own at generation).  The current row's own hold flag never enters its features.
  outputs  P(hold | .) (Bernoulli head) and, for a non-hold row, the driving torque ~ MDN (k = 5)
  loss     BCE(hold) + MDN NLL on the non-hold rows

Held-out checks (the user's gates; a generator that does not know the actual context is NOT asked to guess it row by row):
  E1  true context, teacher-forced history: calibration (ECE), in-band AUROC, per-region hold share, torque NLL
  E2a true context, AUTOREGRESSIVE along the real held-out branch states: hold structure; the frozen v3 onset head fed the
      generated advice -> integrated P(death) vs the realised death (rate, AUROC, death time)
  E2b context from the PRIOR, autoregressive: hold share by region / in band, P(hold | hold), run lengths vs the real
      advice; pooled P(death) and death timing (censored: the real path ends where IT ended -- reported, not gated)
Gates (sealed in the manifest before fitting) decide whether the full rollout (diag_v6_ett_rollout.py --variant B) runs.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_ett_rollout as RO  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from fit_v6_ett_one_step_v2 import HOLD_TOL, in_band, run_length, hist_features  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402
from crl import rockfall_clock_v6 as V6  # noqa: E402  (MOUTH_X / HAZARD_X / HAZARD_HALF_Y: the maze geometry, as in_band uses the zone x-ranges)

EC = MP.OUT / 'ett_context'
OUT = MP.OUT / 'ett_advice_gen'
STATE_DIM, OBS_W, ACTION_DIM = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM
HORIZON = MP.HORIZON
N_FOLDS = 3
ROCKFALL_STEPS = 72
Z_PRIOR = {'p_active': 0.5, 't0_1': (10, 45), 't0_2': (90, 125)}
GEN = {'hidden': (256, 256), 'k': 5, 'steps': 20_000, 'batch': 4096, 'lr': 3e-4, 'seed': 0, 'min_log_std': -5.0}
# v2 (disclosed revision after the v1 checks, 2026-09-20): v1's single hold head learnt "hold iff the previous row held" (teacher-forced AUROC
# 0.996) but put p = 0.04 on the rows where a hold STARTS (0.25 % of rows) and 0.56 on the rows where it is released, so autoregressively the
# runs never start (MAP) or fragment (sampling).  v2 has two hold logits -- start (rows whose previous row did not hold) and continue (rows
# whose previous row held) -- trained with the rare event up-weighted (W_START on a start, W_REL on a release) and the weighting inverted at
# inference so that p is calibrated again; the MAP decision is then p > 0.5.  Features, torque head, budget unchanged.
# v3 (disclosed revision after the v2 probe, 2026-09-20): with v2 the calibrated p at the rows where a hold STARTS was still 0.08 -- the start is
# not a function of the v1 / v2 features.  The teacher consults its rule ONCE, at the path's first arrival at a zone's mouth (x in [MOUTH_X,
# HAZARD_X), |y| < HAZARD_HALF_Y), decides wait iff the burst overlaps the crossing window from THAT step, and then holds until the burst ends
# wherever the ant is.  v3 adds, per zone, the visible-history features "arrived at the mouth", steps since that arrival and (arrival - t0) --
# computed from the logged prefix and the path's own positions (the maze geometry only; nothing hidden).  Otherwise v2.
GEN_VERSION = 3
W_START, W_REL = 20.0, 10.0
IDX_KH_PREV = STATE_DIM + 2 + 7 + 6              # the hist_features column of kh_prev (min(k, 100) / 100): > 0 iff the previous row held
MOUTH_CLIP = (0.0, 4.0)
FEAT_DIM = STATE_DIM + 2 + 7 + 6 + 2 + ACTION_DIM     # [state 29][goal 2][context 7][mouth history 6][hist counters 2][previous torque 8]
GATES = {'E1_in_band_auroc_true_context': 0.90, 'E1_ece': 0.05, 'E1_region_hold_share_abs_gap': 0.02,
         'E2a_p_hold_given_hold_abs_gap': 0.05, 'E2a_run_median_rel_gap': 0.30,
         'E2a_death_rate_vs_real_advice_abs_gap': 0.03, 'E2a_death_auroc': 0.90, 'E2a_death_time_ks': 0.15,
         'E2b_never_hold_regions_max_share': 0.01}
# Gate revision (disclosed, 2026-09-20, after the first v3 fold results): the held-out paths are teacher-forced and CENSORED at their realised
# outcome, so (i) the onset head integrated along a path that died at step j* cannot reach 1 by j* (a calibrated hazard gives ~0.5 there):
# the death-rate gate now compares the generated advice with the REAL advice integrated along the same rows by the same frozen head, not
# with the realised rate (the realised rate is the rollout's test; C already matched it); (ii) run persistence and run length are compared
# like for like under the TRUE context (E2a), not prior context vs real (E2b), whose difference is the context distribution and the
# censoring; E2b keeps the support gate (no holds where the real advice never holds) and reports the rest.
GATE_REGIONS = ('pre_zone1', 'zone1', 'between', 'zone2')
NEVER_HOLD_REGIONS = ('start', 'west_column', 'top_corridor', 'east_column', 'goal_area')     # the real advice never holds there (ett_advice: 0.000)
# E2b's in-band / hazard-region hold shares are REPORTED, not gated: the real rows are selected by the real context (a path under an
# active rock dies early and leaves few in-band rows), so a prior-context generator driven along those rows cannot match them row for
# row; the distributional test of the prior-context advice is the full rollout (variant B).
PRIOR_SEED = 205_000_000
K_DRAWS = 4
DECISION = 'sample'                  # --decision map: the hold flag is the generator's MAP decision (p > 0.5) instead of a Bernoulli draw.  Disclosed revision after
                                     # the fold-0 check of the sampled rule (runs fragment: median 2 vs 14, P(hold | hold) 0.93 vs 0.98, deaths 0.14 vs 0.29): the
                                     # teacher's rule is deterministic given the context and the history, so the advice is its MAP path; the torque stays sampled.
MAX_E1_ROWS = 600_000                # E1 rows per fold, subsampled without replacement (seeded)


# --------------------------------------------------------------- features
def context_features(t_abs, u1, u2, t01, t02):
  d1 = np.clip((t_abs - t01) / ROCKFALL_STEPS, -3.0, 4.0); d2 = np.clip((t_abs - t02) / ROCKFALL_STEPS, -3.0, 4.0)
  b1 = u1 & (t01 <= t_abs) & (t_abs < t01 + ROCKFALL_STEPS); b2 = u2 & (t02 <= t_abs) & (t_abs < t02 + ROCKFALL_STEPS)
  return np.stack([t_abs / HORIZON, u1.astype(np.float64), u2.astype(np.float64), d1, d2, b1.astype(np.float64), b2.astype(np.float64)], axis=1)


def at_mouth(xy, z):
  x, y = xy[:, 0], xy[:, 1]
  return (V6.MOUTH_X[z] <= x) & (x < V6.HAZARD_X[z][0]) & (np.abs(y) < V6.HAZARD_HALF_Y)


def mouth_features(t_abs, m1, m2, t01, t02):
  """Per zone: arrived flag, steps since the first mouth arrival / 72, (arrival - t0) / 72; zeros when not yet arrived (m < 0)."""
  cols = []
  for m, t0 in ((m1, t01), (m2, t02)):
    arr = m >= 0
    cols += [arr.astype(np.float64), np.where(arr, np.clip((t_abs - m) / ROCKFALL_STEPS, *MOUTH_CLIP), 0.0), np.where(arr, np.clip((m - t0) / ROCKFALL_STEPS, -3.0, 3.0), 0.0)]
  return np.stack(cols, axis=1)


def first_mouth_in_prefix(obs_e, t):
  """The first logged step <= t at which episode obs_e stands at the mouth of each zone (-1 if none): the consultation step already recorded."""
  xy = obs_e[:t + 1, :2].astype(np.float64)
  out = np.full(2, -1, np.int64)
  for i, z in enumerate((1, 2)):
    hit = np.flatnonzero(at_mouth(xy, z))
    if len(hit):
      out[i] = int(hit[0])
  return out


def features(obs31, ctx, mf, kh_prev, kb_now, prev_ab, norm):
  sn = (obs31[:, :STATE_DIM] - norm['s_mean']) / norm['s_std']
  g = (obs31[:, STATE_DIM:OBS_W] - norm['g_mean']) / norm['g_std']
  return np.concatenate([sn, g, ctx, mf, hist_features(kh_prev, kb_now), np.asarray(prev_ab, np.float64)], axis=1).astype(np.float32)


def zero_run_before(act, e, t):
  """Consecutive exact-zero logged torques ending at t - 1 of episode e (0 at t = 0)."""
  out = np.zeros(len(e), np.int64)
  for i, (ee, tt) in enumerate(zip(e, t)):
    k = 0
    while tt - 1 - k >= 0 and np.abs(act[ee, tt - 1 - k]).max() == 0.0:
      k += 1
    out[i] = k
  return out


def build_net():
  import haiku as hk
  import jax.numpy as jnp
  k = GEN['k']

  def net(x):
    h = hk.nets.MLP(list(GEN['hidden']), activate_final=True)(x)
    hold2 = hk.Linear(2)(h)                                              # [start logit, continue logit]
    cont = x[:, IDX_KH_PREV] > 0
    hold = jnp.where(cont, hold2[:, 1], hold2[:, 0])
    logits = hk.Linear(k)(h); mu = hk.Linear(k * ACTION_DIM)(h).reshape(-1, k, ACTION_DIM)
    ls = jnp.maximum(hk.Linear(k * ACTION_DIM)(h).reshape(-1, k, ACTION_DIM), GEN['min_log_std'])
    return hold, logits, mu, ls
  return hk.without_apply_rng(hk.transform(net))


def calibrate(p_w, cont):
  """Invert the training weights: a start positive weighted W_START, a release (continue-row negative) weighted W_REL."""
  p_start = p_w / (W_START - (W_START - 1.0) * p_w)
  p_cont = W_REL * p_w / (1.0 + (W_REL - 1.0) * p_w)
  return np.where(cont, p_cont, p_start)


def sample_context(rng, n):
  u1 = rng.random(n) < Z_PRIOR['p_active']; u2 = rng.random(n) < Z_PRIOR['p_active']
  t01 = rng.integers(Z_PRIOR['t0_1'][0], Z_PRIOR['t0_1'][1] + 1, n); t02 = rng.integers(Z_PRIOR['t0_2'][0], Z_PRIOR['t0_2'][1] + 1, n)
  return u1, u2, t01, t02


# --------------------------------------------------------------------- data
class Rows:
  def __init__(self):
    with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
      self.bobs, self.bact, self.boff, self.blen = d['obs_rows'], d['act_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64)
    S = np.load(EC / 'supervision.npz', allow_pickle=False)
    self.adv, self.valid, self.onset = S['advice_rows'], S['valid_rows'].astype(bool), S['onset_rows'].astype(bool)
    self.fold_row, self.anchor_of_row, self.j_of_row, self.fold_anchor = S['fold_rows'].astype(np.int64), S['anchor_of_row'].astype(np.int64), S['j_of_row'].astype(np.int64), S['fold_anchor'].astype(np.int64)
    A = np.load(EC / 'branch_advice.npz', allow_pickle=False)
    self.u1, self.u2, self.t01, self.t02 = A['u1_new'].astype(bool), A['u2_new'].astype(bool), A['t0_1_new'].astype(np.int64), A['t0_2_new'].astype(np.int64)
    self.anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
    assert np.array_equal(A['episode'], self.anchors.episode) and np.array_equal(A['t'], self.anchors.t)
    obs, act, lengths, _ = MP.load_dataset()
    R = len(self.adv)
    self.start = np.zeros(R, bool); self.start[self.boff] = True
    self.hold = np.abs(self.adv).max(axis=1) < HOLD_TOL
    self.band = in_band(self.bobs[:, :2].astype(np.float64))
    self.kh, self.kb = run_length(self.hold, self.start), run_length(self.band, self.start)      # the onset head's exact counters (branch-local)
    self.t_abs = self.anchors.t[self.anchor_of_row].astype(np.int64) + self.j_of_row
    # generator history (teacher-forced): consecutive holds ending at the PREVIOUS row (raw count, 1 at a run's first row) and the previous torque;
    # at the root: the logged zero-torque run before t and the logged torque at t - 1 (zeros at t = 0)
    from diag_v6_ett_advice import raw_counter
    raw = raw_counter(self.hold, self.start)
    self.kh_prev = np.zeros(R, np.int64); self.kh_prev[1:] = raw[:-1]
    self.prev_ab = np.zeros((R, ACTION_DIM), np.float32); self.prev_ab[1:] = self.adv[:-1]
    e, t = self.anchors.episode, self.anchors.t
    has_prev = t >= 1
    la = np.zeros((self.anchors.n, ACTION_DIM), np.float32); la[has_prev] = act[e[has_prev], t[has_prev] - 1]
    lrun = zero_run_before(act, e, t)
    self.prev_ab[self.boff] = la; self.kh_prev[self.boff] = lrun
    self.log_prev = (la, lrun)
    self.region = region_of(self.bobs[:, :2].astype(np.float64))
    # mouth-arrival history: the logged prefix's first arrival (per anchor), else the branch's own first arrival up to the row
    self.m_log = np.stack([first_mouth_in_prefix(obs[ee], int(tt)) for ee, tt in zip(e, t)])          # [n_anchors, 2]
    xy = self.bobs[:, :2].astype(np.float64)
    self.m_row = np.full((R, 2), -1, np.int64)
    for i, z in enumerate((1, 2)):
      am = at_mouth(xy, z)
      for k in range(self.anchors.n):
        a, b = self.boff[k], self.boff[k] + self.blen[k]
        if self.m_log[k, i] >= 0:
          self.m_row[a:b, i] = self.m_log[k, i]; continue
        hit = np.flatnonzero(am[a:b])
        if len(hit):
          self.m_row[a + hit[0]:b, i] = int(t[k]) + int(hit[0])
    del obs

  def ctx(self, rows, u1=None, u2=None, t01=None, t02=None):
    k = self.anchor_of_row[rows]
    if u1 is None:
      u1, u2, t01, t02 = self.u1[k], self.u2[k], self.t01[k], self.t02[k]
    return context_features(self.t_abs[rows], u1, u2, t01, t02)

  def mouth(self, rows, m1=None, m2=None, t01=None, t02=None):
    k = self.anchor_of_row[rows]
    if m1 is None:
      m1, m2, t01, t02 = self.m_row[rows, 0], self.m_row[rows, 1], self.t01[k], self.t02[k]
    return mouth_features(self.t_abs[rows], m1, m2, t01, t02)

  def feats(self, rows, norm):
    return features(self.bobs[rows, :OBS_W], self.ctx(rows), self.mouth(rows), self.kh_prev[rows], self.kb[rows], self.prev_ab[rows], norm)


# ---------------------------------------------------------------------- fit
def model_path(fold):
  return OUT / (f'gen_fold{fold}.pkl' if GEN_VERSION == 1 else f'gen{GEN_VERSION}_fold{fold}.pkl')


def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6 one-step ETT: the advice generator (state + own history + persistent hidden hazard context), fitted on the branch supervision; motion model and onset head frozen (v3)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'oracle-supervised engineering revision (the advice supervision and the branch contexts come from the simulator); nothing downstream generated',
         'supervision_sha256': MP.sha256(EC / 'supervision.npz'), 'branch_advice_sha256': MP.sha256(EC / 'branch_advice.npz'), 'v3_models': {f'fold{f}': MP.sha256(MP.OUT / 'ett_one_step_v3' / f'model_fold{f}.json') for f in range(N_FOLDS)},
         'context': {'observed_in_training': 'the branch\'s redrawn timetable (u1, u2, t0_1, t0_2)', 'prior_at_generation': Z_PRIOR, 'burst_length': ROCKFALL_STEPS, 'never_read': 'the evaluation episode\'s actual hidden draw'},
         'inputs': 'standardised state (29), goal xy (2), t / 800, u1, u2, (t - t0_z) / 72 clipped [-3, 4], in-burst flags (2), exact history counters (2), previous advice hold flag (1) and torque (8); teacher-forced in training, own at generation',
         'outputs': 'P(hold); driving torque MDN (k 5) on non-hold rows', 'fit': GEN, 'folds': 'the supervision folds (source episode); fold f held out; fixed final iterate, no selection',
         'hold_definition': f'max |a_b| < {HOLD_TOL}; a generated hold is the exact zero torque', 'gates': GATES, 'gate_regions': list(GATE_REGIONS), 'never_hold_regions': list(NEVER_HOLD_REGIONS),
         'E2b_not_gated': 'in-band and hazard-region hold shares under the prior context (reported only): the real rows are selected by the real context, so they are not a row-for-row target; the distributional test is the rollout',
         'checks': {'E1': 'true context, teacher-forced history, held-out rows', 'E2a': 'true context, autoregressive along the real held-out branch states; frozen onset head -> P(death), compared with the REAL advice through the same head along the same censored rows; hold persistence and run length like for like', 'E2b': f'prior context ({K_DRAWS} draws per path, seed {PRIOR_SEED}), autoregressive; support gate (no holds where the real advice never holds); the rest reported (censored, context distribution)'},
         'gate_revision': 'the first v3 fold results showed the original E2a death-rate gate (censored integral vs realised rate) and the E2b persistence gates (prior context vs real) were not like for like; corrected as documented in the code (GATES) before any rollout',
         'decision': 'all gates in every fold -> the full rollout diag_v6_ett_rollout.py --v3 --hist-exact --variant B (prior context, K = 4 draws per anchor); otherwise stop and report'}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


def mode_fit(args):
  import jax
  import jax.numpy as jnp
  import optax
  fold = int(args.fold)
  out = model_path(fold)
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  R = Rows()
  tr = np.flatnonzero(R.valid & (R.fold_row != fold))
  if args.limit:
    tr = tr[:int(args.limit)]
  norm = {'s_mean': R.bobs[tr, :STATE_DIM].mean(0).astype(np.float32), 's_std': np.maximum(R.bobs[tr, :STATE_DIM].std(0), 1e-3).astype(np.float32),
          'g_mean': R.bobs[tr, STATE_DIM:OBS_W].mean(0).astype(np.float32), 'g_std': np.maximum(R.bobs[tr, STATE_DIM:OBS_W].std(0), 1e-3).astype(np.float32)}
  X = R.feats(tr, norm); Y_h = R.hold[tr].astype(np.float32); Y_a = R.adv[tr].astype(np.float32)
  del R
  net = build_net(); params = net.init(jax.random.PRNGKey(GEN['seed'] + fold), jnp.zeros((1, FEAT_DIM), jnp.float32))
  opt = optax.adam(GEN['lr']); ost = opt.init(params)

  def loss_fn(p, x, yh, ya):
    hold, logits, mu, ls = net.apply(p, x)
    cont = x[:, IDX_KH_PREV] > 0
    wgt = jnp.where(cont, jnp.where(yh < 0.5, W_REL, 1.0), jnp.where(yh > 0.5, W_START, 1.0))
    bce = (wgt * optax.sigmoid_binary_cross_entropy(hold, yh)).sum() / wgt.sum()
    lp = -0.5 * jnp.sum(((ya[:, None, :] - mu) / jnp.exp(ls)) ** 2 + 2 * ls + jnp.log(2 * jnp.pi), axis=-1) + jax.nn.log_softmax(logits, axis=-1)
    nll = -jax.scipy.special.logsumexp(lp, axis=-1)
    w = 1.0 - yh
    return bce + (w * nll).sum() / jnp.maximum(w.sum(), 1.0), (bce, (w * nll).sum() / jnp.maximum(w.sum(), 1.0))

  @jax.jit
  def step(p, o, x, yh, ya):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, yh, ya); up, o = opt.update(g, o, p); return optax.apply_updates(p, up), o, l, aux
  rng = np.random.default_rng(GEN['seed'] + 100 + fold)
  t0, hist = time.time(), []
  for it in range(1, GEN['steps'] + 1):
    i = rng.integers(0, len(tr), GEN['batch'])
    params, ost, l, (bce, nll) = step(params, ost, jnp.asarray(X[i]), jnp.asarray(Y_h[i]), jnp.asarray(Y_a[i]))
    if it % 1000 == 0:
      hist.append({'step': it, 'loss': float(l), 'bce': float(bce), 'torque_nll': float(nll)})
      print(f'[gen fold {fold} step {it:>6}] loss {float(l):.4f} bce {float(bce):.4f} torque nll {float(nll):.3f} {it / (time.time() - t0):.0f} it/s', flush=True)
  md = {'params': jax.tree_util.tree_map(np.asarray, params), 'norm': norm, 'gen': {**GEN, 'version': GEN_VERSION, 'w_start': W_START, 'w_rel': W_REL}, 'fold': fold, 'n_train_rows': int(len(tr)), 'hold_share_train': float(Y_h.mean()), 'history': hist,
        'note': 'fixed final iterate (no selection); cross-fitted by the supervision folds'}
  OUT.mkdir(parents=True, exist_ok=True)
  with out.open('wb') as f:
    pickle.dump(md, f)
  MP.write_json(out.with_suffix('.json'), {k: v for k, v in md.items() if k not in ('params', 'norm')})
  print(f'saved {out} ({time.time() - t0:.0f} s)', flush=True)


# ---------------------------------------------------------------- generator
class Generator:
  """The advice generator of one fold: p_hold and torque sampling, batched; the caller keeps the per-path history."""

  def __init__(self, md):
    import jax
    import jax.numpy as jnp
    net = build_net(); p = jax.tree_util.tree_map(jnp.asarray, md['params'])
    self.norm = md['norm']
    self._f = jax.jit(lambda x: net.apply(p, x))

  def forward(self, x, chunk=65536):
    outs = [[], [], [], []]
    for i in range(0, len(x), chunk):
      for o, v in zip(outs, self._f(np.asarray(x[i:i + chunk], np.float32))):
        o.append(np.asarray(v, np.float64))
    hold, logits, mu, ls = (np.concatenate(o) for o in outs)
    p = calibrate(1.0 / (1.0 + np.exp(-hold)), np.asarray(x)[:, IDX_KH_PREV] > 0)
    return p, logits, mu, np.exp(ls)

  def sample(self, x, rng, decision=None):
    """Returns (hold flag, torque) per row: a hold is the exact zero torque, otherwise an MDN draw clipped to [-1, 1]."""
    p, logits, mu, sd = self.forward(x)
    hold = (p > 0.5) if (decision or DECISION) == 'map' else (rng.random(len(p)) < p)
    pi = np.exp(logits - logits.max(axis=1, keepdims=True)); pi /= pi.sum(axis=1, keepdims=True)
    c = np.minimum((rng.random(len(p))[:, None] > np.cumsum(pi, axis=1)).sum(axis=1), pi.shape[1] - 1)
    a = np.clip(mu[np.arange(len(p)), c] + sd[np.arange(len(p)), c] * rng.standard_normal((len(p), ACTION_DIM)), -1.0, 1.0).astype(np.float32)
    a[hold] = 0.0
    return hold, a, p


class PathState:
  """Per-path history for n paths driven step by step: the generator's kh_prev / prev_ab, and the exact branch-local
  run_length counters of the onset head (last reset index = the path start or the last unflagged row)."""

  def __init__(self, kh_prev0, prev_ab0, m0):
    n = len(kh_prev0)
    self.kh_prev = np.asarray(kh_prev0, np.int64).copy(); self.prev_ab = np.asarray(prev_ab0, np.float32).copy()
    self.last_h = np.zeros(n, np.int64); self.last_b = np.zeros(n, np.int64)
    self.m = np.asarray(m0, np.int64).copy()                               # [n, 2] first mouth arrival per zone (-1 = not yet)

  def mouth_update(self, idx, t_abs_now, xy_now):
    """Record the first mouth arrival of the active paths at their current row (the consultation step), then return the features' inputs."""
    for i, z in enumerate((1, 2)):
      hit = (self.m[idx, i] < 0) & at_mouth(xy_now, z)
      self.m[idx[hit], i] = t_abs_now[hit]
    return self.m[idx, 0], self.m[idx, 1]

  def band_counter(self, idx, j, band_now):
    self.last_b[idx] = np.where((j == 0) | ~band_now, j, self.last_b[idx])
    return np.where(band_now, j - self.last_b[idx], 0)

  def hold_counter(self, idx, j, hold_now):
    self.last_h[idx] = np.where((j == 0) | ~hold_now, j, self.last_h[idx])
    return np.where(hold_now, j - self.last_h[idx], 0)

  def advance(self, idx, hold, ab):
    self.kh_prev[idx] = np.where(hold, self.kh_prev[idx] + 1, 0); self.prev_ab[idx] = ab


def load_generator(fold):
  with model_path(fold).open('rb') as f:
    return Generator(pickle.load(f))


# ------------------------------------------------------------------ checks
def ece(p, y, bins=10):
  e, edges = 0.0, np.linspace(0, 1, bins + 1)
  for a, b in zip(edges[:-1], edges[1:]):
    m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
    if m.any():
      e += m.mean() * abs(p[m].mean() - y[m].mean())
  return float(e)


def auroc(score, pos):
  from scipy.stats import rankdata
  pos = np.asarray(pos, bool); r = rankdata(score); n1, n0 = int(pos.sum()), int((~pos).sum())
  return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def hold_structure(hold, start, band, region):
  from diag_v6_ett_advice import run_lengths, persistence, describe_runs
  out = {'hold_share': float(hold.mean()), 'in_band': float(hold[band].mean()) if band.any() else float('nan'), 'out_of_band': float(hold[~band].mean()),
         **persistence(hold, start), 'runs': describe_runs(run_lengths(hold, start)),
         'by_region': {REGIONS[i]: float(hold[region == i].mean()) for i in range(len(REGIONS)) if (region == i).sum() >= 100}}
  return out


def autoregressive(R, gen, paths, ctx_fn, rng, onset=None):
  """Drive the generator along the REAL states of the given branches (teacher-forced states, generated advice / history).
  paths: anchor ids.  ctx_fn(k_array) -> (u1, u2, t01, t02).  Returns per-row hold flags, torques, p_hold, and per-path death integration."""
  n = len(paths); off, L = R.boff[paths], R.blen[paths] - 1              # valid rows per path (the last row has no advice)
  maxL = int(L.max())
  u1, u2, t01, t02 = ctx_fn(paths)
  la, lrun = R.log_prev
  st = PathState(lrun[paths], la[paths], R.m_log[paths])
  N = int(L.sum())
  hold_rows = np.zeros(N, bool); p_rows = np.zeros(N); ab_rows = np.zeros((N, ACTION_DIM), np.float32); kh_rows = np.zeros(N, np.int64); kb_rows = np.zeros(N, np.int64)
  row_off = np.concatenate([[0], np.cumsum(L)[:-1]])
  p_on = np.zeros(N)
  for j in range(maxL):
    act = np.flatnonzero(L > j)
    n_act = len(act)
    pad = np.concatenate([act, np.full(n - n_act, act[0])])              # fixed batch shape: no XLA recompilation as paths end
    rows = off[pad] + j
    ctx = context_features(R.t_abs[rows], u1[pad], u2[pad], t01[pad], t02[pad])
    band_now = R.band[rows]
    kb_now = st.band_counter(act, j, band_now[:n_act])                    # state-based, known before the decision
    m1, m2 = st.mouth_update(act, R.t_abs[rows][:n_act], R.bobs[rows[:n_act], :2].astype(np.float64))
    mf = mouth_features(R.t_abs[rows][:n_act], m1, m2, t01[act], t02[act])
    kb_pad = np.concatenate([kb_now, np.zeros(n - n_act, np.int64)]); mf_pad = np.concatenate([mf, np.zeros((n - n_act, mf.shape[1]))])
    x = features(R.bobs[rows, :OBS_W], ctx, mf_pad, st.kh_prev[pad], kb_pad, st.prev_ab[pad], gen.norm)
    hold, ab, p = gen.sample(x, rng)
    hold, ab, p = hold[:n_act], ab[:n_act], p[:n_act]
    kh_now = st.hold_counter(act, j, hold)                                # the onset head's exact counter of the current row
    idx = row_off[act] + j
    hold_rows[idx] = hold; p_rows[idx] = p; ab_rows[idx] = ab; kh_rows[idx] = kh_now; kb_rows[idx] = kb_now
    if onset is not None:
      ab_pad = np.concatenate([ab, np.zeros((n - n_act, ACTION_DIM), np.float32)]); kh_pad = np.concatenate([kh_now, np.zeros(n - n_act, np.int64)])
      _, po, _ = onset.step(R.bobs[rows, :STATE_DIM], ab_pad, R.bact[rows], kh_pad, kb_pad)
      p_on[idx] = po[:n_act]
    st.advance(act, hold, ab)
  res = {'hold': hold_rows, 'p_hold': p_rows, 'ab': ab_rows, 'kh': kh_rows, 'kb': kb_rows, 'row_off': row_off, 'L': L, 'rows_real': np.concatenate([np.arange(off[i], off[i] + L[i]) for i in range(n)])}
  if onset is not None:
    surv = np.ones(n); p_death = np.zeros(n); death_t_w = []
    for i in range(n):
      po = p_on[row_off[i]:row_off[i] + L[i]]
      S = np.concatenate([[1.0], np.cumprod(1 - po)])
      w = S[:-1] * po
      p_death[i] = w.sum(); death_t_w.append(w)
    res['p_death'] = p_death; res['death_t_w'] = death_t_w; res['p_onset_rows'] = p_on
  return res


def mode_check(args):
  fold = int(args.fold)
  R = Rows()
  gen = load_generator(fold)
  F2.OUT = F2.OUT_V3
  onset = F2.Predictor2(F2.load_model(fold))
  ho = np.flatnonzero(R.valid & (R.fold_row == fold))
  if len(ho) > MAX_E1_ROWS:
    ho = np.sort(np.random.default_rng(PRIOR_SEED + 100 + fold).choice(ho, size=MAX_E1_ROWS, replace=False))
  res = {'fold': fold, 'held_out_rows': int(len(ho))}
  t0 = time.time()
  # ---- E1: true context, teacher-forced history
  X = R.feats(ho, gen.norm)
  p, logits, mu, sd = gen.forward(X)
  y = R.hold[ho]; band = R.band[ho]; reg = R.region[ho]
  from diag_v6_ett_advice import MDNBatch
  pi = np.exp(logits - logits.max(axis=1, keepdims=True)); pi /= pi.sum(axis=1, keepdims=True)
  nll = MDNBatch.nll(pi, mu, sd, R.adv[ho].astype(np.float64))
  res['E1'] = {'hold_share_real': float(y.mean()), 'hold_share_pred_mean': float(p.mean()), 'ece': ece(p, y.astype(float)), 'auroc_all': auroc(p, y), 'auroc_in_band': auroc(p[band], y[band]),
               'auroc_out_of_band': auroc(p[~band], y[~band]), 'torque_nll_non_hold_rows': float(nll[~y].mean()),
               'by_region': {REGIONS[i]: {'real': float(y[reg == i].mean()), 'pred': float(p[reg == i].mean()), 'rows': int((reg == i).sum())} for i in range(len(REGIONS)) if (reg == i).sum() >= 100}}
  print(f'E1 done {time.time() - t0:.0f} s: auroc in band {res["E1"]["auroc_in_band"]:.3f} ece {res["E1"]["ece"]:.4f}', flush=True)
  # ---- the held-out paths: the rollout diagnostic's anchors of this fold
  paths = RO.sample_anchors(fold)
  if args.limit:
    paths = paths[:int(args.limit)]
  rows_real = np.concatenate([np.arange(R.boff[k], R.boff[k] + R.blen[k] - 1) for k in paths])
  start_sub = np.zeros(len(rows_real), bool); start_sub[np.concatenate([[0], np.cumsum(R.blen[paths] - 1)[:-1]])] = True
  real = hold_structure(R.hold[rows_real], start_sub, R.band[rows_real], R.region[rows_real])
  realised_death = np.array([R.onset[R.boff[k]:R.boff[k] + R.blen[k] - 1].any() for k in paths])
  death_t_real = np.array([int(np.flatnonzero(R.onset[R.boff[k]:R.boff[k] + R.blen[k] - 1])[0]) + 1 if R.onset[R.boff[k]:R.boff[k] + R.blen[k] - 1].any() else -1 for k in paths])
  res['paths'] = {'n': int(len(paths)), 'realised_death_rate': float(realised_death.mean()), 'real_hold_structure': real}
  # ---- the reference: the REAL advice with the real counters through the same frozen onset head along the same (censored) rows
  p_ref = np.concatenate([onset.step(R.bobs[rows_real[i:i + 65536], :STATE_DIM], R.adv[rows_real[i:i + 65536]], R.bact[rows_real[i:i + 65536]], R.kh[rows_real[i:i + 65536]], R.kb[rows_real[i:i + 65536]])[1]
                          for i in range(0, len(rows_real), 65536)])
  Lp = R.blen[paths] - 1; ro = np.concatenate([[0], np.cumsum(Lp)[:-1]])
  p_death_ref = np.array([1.0 - np.prod(1.0 - p_ref[ro[i]:ro[i] + Lp[i]]) for i in range(len(paths))])
  # ---- E2a: true context, autoregressive
  rng = np.random.default_rng(PRIOR_SEED + 10 * fold)
  g = autoregressive(R, gen, paths, lambda ks: (R.u1[ks], R.u2[ks], R.t01[ks], R.t02[ks]), rng, onset=onset)
  gen_struct = hold_structure(g['hold'], start_sub, R.band[rows_real], R.region[rows_real])
  mt = np.concatenate([np.arange(1, len(w) + 1) for w in g['death_t_w']]); mw = np.concatenate(g['death_t_w'])
  res['E2a'] = {'hold_structure': gen_struct, 'row_agreement_with_real_hold': float((g['hold'] == R.hold[rows_real]).mean()), 'in_band_row_auroc_p_hold': auroc(g['p_hold'][R.band[rows_real]], R.hold[rows_real][R.band[rows_real]]),
                'p_death_mean': float(g['p_death'].mean()), 'death_auroc': auroc(g['p_death'], realised_death),
                'p_death_real_advice_mean': float(p_death_ref.mean()), 'death_auroc_real_advice': auroc(p_death_ref, realised_death), 'p_death_corr_with_real_advice': float(np.corrcoef(g['p_death'], p_death_ref)[0, 1]),
                'death_time_ks': RO.wks(mt, mw, death_t_real[realised_death], np.ones(int(realised_death.sum()))) if realised_death.any() else float('nan'),
                'death_time_median_model': float(np.interp(0.5, np.cumsum(mw[np.argsort(mt)]) / mw.sum(), np.sort(mt))) if mw.sum() > 0 else float('nan'), 'death_time_median_real': float(np.median(death_t_real[realised_death])) if realised_death.any() else float('nan')}
  print(f'E2a done {time.time() - t0:.0f} s: p_death {res["E2a"]["p_death_mean"]:.3f} (real advice {p_death_ref.mean():.3f}; realised {realised_death.mean():.3f}) auroc {res["E2a"]["death_auroc"]:.3f}', flush=True)
  # ---- E2b: prior context, K draws
  E2b = {'draws': []}
  for d in range(K_DRAWS):
    rng = np.random.default_rng(PRIOR_SEED + 10 * fold + 1 + d)
    ctx_draw = sample_context(rng, len(paths))
    g = autoregressive(R, gen, paths, lambda ks, c=ctx_draw: c, rng, onset=onset)
    s_ = hold_structure(g['hold'], start_sub, R.band[rows_real], R.region[rows_real])
    mt = np.concatenate([np.arange(1, len(w) + 1) for w in g['death_t_w']]); mw = np.concatenate(g['death_t_w'])
    E2b['draws'].append({'hold_structure': s_, 'p_death_mean': float(g['p_death'].mean()), 'death_time_ks_vs_realised': RO.wks(mt, mw, death_t_real[realised_death], np.ones(int(realised_death.sum()))) if realised_death.any() else float('nan')})
  agg = lambda key: float(np.mean([dd['hold_structure'][key] for dd in E2b['draws']]))  # noqa: E731
  E2b['mean'] = {'hold_share': agg('hold_share'), 'in_band': agg('in_band'), 'p_hold_given_hold': agg('p_hold_given_hold'), 'run_median': float(np.mean([dd['hold_structure']['runs'].get('median', np.nan) for dd in E2b['draws']])),
                 'by_region': {r: float(np.mean([dd['hold_structure']['by_region'].get(r, np.nan) for dd in E2b['draws']])) for r in set(real['by_region']) | set(E2b['draws'][0]['hold_structure']['by_region'])}, 'p_death_mean': float(np.mean([dd['p_death_mean'] for dd in E2b['draws']]))}
  res['E2b'] = E2b
  print(f'E2b done {time.time() - t0:.0f} s: in-band hold {E2b["mean"]["in_band"]:.3f} vs real {real["in_band"]:.3f}; P(h|h) {E2b["mean"]["p_hold_given_hold"]:.3f}; p_death {E2b["mean"]["p_death_mean"]:.3f}', flush=True)
  # ---- gates
  g1 = {'auroc_in_band': res['E1']['auroc_in_band'] >= GATES['E1_in_band_auroc_true_context'], 'ece': res['E1']['ece'] <= GATES['E1_ece'],
        'region_shares': all(abs(res['E1']['by_region'][r]['pred'] - res['E1']['by_region'][r]['real']) <= GATES['E1_region_hold_share_abs_gap'] for r in GATE_REGIONS if r in res['E1']['by_region'])}
  g2 = {'never_hold_regions': all(E2b['mean']['by_region'].get(r, 0.0) <= GATES['E2b_never_hold_regions_max_share'] for r in NEVER_HOLD_REGIONS)}
  gs = res['E2a']['hold_structure']
  g3 = {'death_rate_vs_real_advice': abs(res['E2a']['p_death_mean'] - res['E2a']['p_death_real_advice_mean']) <= GATES['E2a_death_rate_vs_real_advice_abs_gap'],
        'death_auroc': res['E2a']['death_auroc'] >= GATES['E2a_death_auroc'], 'death_time_ks': res['E2a']['death_time_ks'] <= GATES['E2a_death_time_ks'],
        'p_hold_given_hold': abs(gs['p_hold_given_hold'] - real['p_hold_given_hold']) <= GATES['E2a_p_hold_given_hold_abs_gap'],
        'run_median': abs(gs['runs'].get('median', np.nan) - real['runs']['median']) <= GATES['E2a_run_median_rel_gap'] * real['runs']['median']}
  g1, g2, g3 = ({k: bool(v) for k, v in g.items()} for g in (g1, g2, g3))
  res['gates'] = {'E1': g1, 'E2b': g2, 'E2a': g3, 'all': bool(all(g1.values()) and all(g2.values()) and all(g3.values()))}
  res['wall_seconds'] = time.time() - t0
  res['decision'] = DECISION; res['generator_version'] = GEN_VERSION
  MP.write_json(OUT / (f'check{GEN_VERSION if GEN_VERSION > 1 else ""}_fold{fold}.json' if DECISION == 'sample' else f'check{GEN_VERSION if GEN_VERSION > 1 else ""}_fold{fold}_{DECISION}.json'), res)
  print(json.dumps(res['gates'], indent=1), flush=True)


def mode_report(args):
  man = MP.read_json(OUT / 'manifest.json')
  suffix = '' if DECISION == 'sample' else f'_{DECISION}'
  stem = f'check{GEN_VERSION if GEN_VERSION > 1 else ""}_fold'
  C = {f: MP.read_json(OUT / f'{stem}{f}{suffix}.json') for f in range(N_FOLDS) if (OUT / f'{stem}{f}{suffix}.json').exists()}
  all_ok = bool(C) and len(C) == N_FOLDS and all(c['gates']['all'] for c in C.values())
  L = ['# The advice generator: held-out checks (E1 true context / E2a true context autoregressive / E2b prior context autoregressive)', '',
       f"Sealed {man['sealed_at']}.  Gates: {json.dumps(GATES)}.", '',
       '| fold | rows | E1 AUROC in band | E1 ECE | E1 torque NLL (non-hold) | E2a P(death) generated / real advice / realised | E2a death AUROC gen / real advice | E2a death-time KS | E2a hold-row agreement | E2a P(h|h) gen / real | E2a run median gen / real | E2b in-band hold gen / real | E2b P(death) | gates |',
       '|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---|']
  for f, c in C.items():
    r = c['paths']['real_hold_structure']; b = c['E2b']['mean']; a = c['E2a']; gs = a['hold_structure']
    L.append(f"| {f} | {c['held_out_rows']} | {c['E1']['auroc_in_band']:.3f} | {c['E1']['ece']:.4f} | {c['E1']['torque_nll_non_hold_rows']:.2f} | {a['p_death_mean']:.3f} / {a.get('p_death_real_advice_mean', float('nan')):.3f} / {c['paths']['realised_death_rate']:.3f} | {a['death_auroc']:.3f} / {a.get('death_auroc_real_advice', float('nan')):.3f} | {a['death_time_ks']:.3f} | {a['row_agreement_with_real_hold']:.3f} | "
             f"{gs['p_hold_given_hold']:.3f} / {r['p_hold_given_hold']:.3f} | {gs['runs'].get('median', float('nan')):.0f} / {r['runs']['median']:.0f} | {b['in_band']:.3f} / {r['in_band']:.3f} | {b['p_death_mean']:.3f} | {'PASS' if c['gates']['all'] else 'FAIL ' + str([k + ':' + kk for k, v in c['gates'].items() if isinstance(v, dict) for kk, vv in v.items() if not vv])} |")
  L += ['', '## Hold share by region (E1 pred / real; E2b gen / real)', '', '| fold | ' + ' | '.join(GATE_REGIONS) + ' |', '|---|' + '---|' * len(GATE_REGIONS)]
  for f, c in C.items():
    L.append(f'| {f} | ' + ' | '.join(f"{c['E1']['by_region'][r]['pred']:.3f} / {c['E1']['by_region'][r]['real']:.3f}; {c['E2b']['mean']['by_region'].get(r, float('nan')):.3f} / {c['paths']['real_hold_structure']['by_region'].get(r, float('nan')):.3f}" if r in c['E1']['by_region'] else '-' for r in GATE_REGIONS) + ' |')
  L += ['', f"All gates in every fold: **{all_ok}** -> {'the full rollout (variant B) runs' if all_ok else 'stop; report'}.", '']
  L.insert(1, f'Generator version {GEN_VERSION}; hold decision rule: {DECISION}.')
  v = '' if GEN_VERSION == 1 else str(GEN_VERSION)
  (OUT / f'REPORT{v}{suffix}.md').write_text('\n'.join(L), encoding='utf-8')
  MP.write_json(OUT / f'gates{v}{suffix}.json', {'all_folds_pass': all_ok, 'generator_version': GEN_VERSION, 'decision': DECISION, 'per_fold': {f: c['gates'] for f, c in C.items()}})
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('seal', 'fit', 'check', 'report'))
  ap.add_argument('--fold', type=int, default=0)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--decision', choices=('sample', 'map'), default='sample')
  ap.add_argument('--gen-version', type=int, choices=(1, 2, 3), default=3)
  args = ap.parse_args(argv)
  global DECISION, GEN_VERSION
  DECISION = args.decision; GEN_VERSION = args.gen_version
  {'seal': mode_seal, 'fit': mode_fit, 'check': mode_check, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
