"""One-step ETT revision v2 (disclosed), after the Step 3a diagnosis:

  (a) a STATIONARY GATE with a zero-displacement atom -- the simulator's
      stalled ants (39 % of the supervision rows are near-stationary; the
      full state settles by < 1e-3 per step) drifted in the v1 regression
      (0.54 in 100 open-loop steps) until the policy re-engaged; v2 trains a
      classifier P(stationary | s, a_b, a_q) (label: max |delta s| < 1e-3)
      on all rows, the motion regression on MOVING rows only, and at
      prediction emits delta = 0 when the gate fires (the project's
      diagonal-model design, `ett/diagonal_transition.py`);
  (b) two VISIBLE-HISTORY features for the onset head -- the number of
      consecutive steps the advice has been "hold" up to this row, and the
      number of steps since the current hazard band was entered -- both
      computed from the path itself (never the hidden clock; a hold is
      never lengthened); the v1 memoryless hazard put 65 % of the in-band
      death mass in steps 1-5 where the simulator has none.

Everything else is v1 (`fit_v6_ett_one_step.py`): inputs, motion MLP
1024-1024 (MSE, diagonal / off-diagonal averaged separately), onset MLP
256-256 on the features + predicted delta (natural-prevalence BCE), Adam
3e-4, 40k updates, cross-fitting by source episode, selection = earliest
minimum of mse_diag + mse_off + 0.25 * onset_bce + 0.25 * stationary_bce
on the early-stop slice.  Gates: v1's G1-G6 on MOVING held-out rows plus
G7 stationary-gate held-out AUROC >= 0.95 and accuracy >= 0.95, and the
one-step xy error on stationary rows with the atom <= 0.001 median.
Output: `ett_one_step_v2/`.

  python scripts/fit_v6_ett_one_step_v2.py seal | fit --fold f | check
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
import fit_v6_ett_one_step as V1  # noqa: E402
from fit_v6_ett_one_step import FIT, GATES, N_FOLDS, STATE_DIM, ZONE_X, ROLL_H, auroc, features  # noqa: E402

EC = MP.OUT / 'ett_context'
OUT = MP.OUT / 'ett_one_step_v2'
STAT_TOL = 1e-3
HOLD_TOL = 0.05          # advice with max |a_b| < HOLD_TOL counts as a hold (the teacher's hold is exactly zero; a nominal sample is near zero)
HIST_CAP = 100.0
GATES2 = {**GATES, 'G7_stat_auroc': 0.95, 'G7_stat_accuracy': 0.95, 'G7_atom_xy_median': 0.001}
STAT_HIDDEN = (256, 256)
V3 = False                              # --v3: regression on ALL rows + relative error term; selection without the stationary BCE; outputs under ett_one_step_v3/
OUT_V3 = MP.OUT / 'ett_one_step_v3'
V4 = False                              # --v4 (user's revision after ett_motion_ab, 2026-09-20): v3 loss, but the MOTION regression and the STATIONARY gate take
                                        # (s, a_q) only -- the executed torque decides how the body moves; the advice a_b (the hidden context's carrier) stays an
                                        # input of the ONSET head only.  Outputs under ett_one_step_v4/.
OUT_V4 = MP.OUT / 'ett_one_step_v4'
CF_ROWS = None                          # --cf-rows: the CF-controlled motion rows (collect_v6_cf_motion.py); --cf-share of each motion / stationary batch
CF_SHARE = 0.25
INIT_FROM = None                        # --init-from: continue from this model dir's fold model (params + norm); the controlled continuation after 5650f8d
FREEZE_ONSET = False                    # --freeze-onset: the onset head keeps its initial parameters (motion regression + stationary gate only)
MULTISTEP = 0                           # --multistep K: short-sequence unroll term on the CF rollouts (0 = off); --multistep-weight / --multistep-batch / --multistep-stall-share
MS_WEIGHT, MS_BATCH, MS_STALL_SHARE = 1.0, 256, 0.5
MS_SOURCE = 'cf'                        # --multistep-source branch (user's plan after d8feadd): the windows come from the sealed table's START-agent branch paths
                                        # (branches_cf.npz obs_rows / act_rows; the target generation protocol) under the fold's own episode-level split
                                        # (train windows from the training rows, selection windows from the early-stop episodes; never the held-out fold);
                                        # 'cf' = the CF-controlled rollouts (arms B / C).  Slow / static start rows = xy speed < SLOW_SPEED, as in the CF rows.
SLOW_SPEED = 0.03
REL_EPS, REL_W = 0.05, 1.0              # relative term: ||pred - d||^2 / (||d||^2 + REL_EPS^2) in standardised units


# --------------------------------------------------------------- history
def run_length(flag, start):
  """For every row: the number of consecutive rows with `flag` ending at the row (0 if the flag is off), resetting at path starts."""
  idx = np.arange(len(flag))
  reset = (~flag) | start
  last = np.maximum.accumulate(np.where(reset, idx, 0))
  return np.where(flag, idx - last, 0).astype(np.int32)


def in_band(xy):
  x, y = xy[:, 0], xy[:, 1]
  return (np.abs(y) < 2.0) & (((x >= ZONE_X[1][0]) & (x <= ZONE_X[1][1])) | ((x >= ZONE_X[2][0]) & (x <= ZONE_X[2][1])))


def features_q(s, aq, norm):
  """v4 motion / stationary inputs: the standardised state and the executed torque only."""
  return np.concatenate([(s - norm['s_mean']) / norm['s_std'], aq], axis=1).astype(np.float32)


def hist_features(kh, kb):
  return np.stack([np.minimum(kh, HIST_CAP) / HIST_CAP, np.minimum(kb, HIST_CAP) / HIST_CAP], axis=1).astype(np.float32)


class Rows2(V1.Rows):
  def __init__(self):
    super().__init__()
    # branch rows: hold = advice near zero; band from positions; path starts at branch offsets
    hold_b = np.abs(self.adv).max(axis=1) < HOLD_TOL
    start_b = np.zeros(len(hold_b), bool); start_b[self.boff] = True
    with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
      bxy = d['obs_rows'][:, :2]
    self.kh_b = run_length(hold_b, start_b); self.kb_b = run_length(in_band(bxy), start_b)
    # logged rows: the teacher's hold is an exactly zero torque
    hold_l = np.abs(self.lact[self.le, self.lt]).max(axis=1) == 0.0
    start_l = np.concatenate([[True], self.le[1:] != self.le[:-1]])
    lxy = self.lobs[self.le, self.lt, :2]
    self.kh_l = run_length(hold_l, start_l); self.kb_l = run_length(in_band(lxy), start_l)
    # stationarity labels
    self.stat_b = np.zeros(len(self.valid), bool)
    self.stat_b[self.br] = np.abs(self.bobs[self.br + 1] - self.bobs[self.br]).max(axis=1) < STAT_TOL
    self.stat_l = np.abs(self.lobs[self.le, self.lt + 1] - self.lobs[self.le, self.lt]).max(axis=1) < STAT_TOL

  def gather_branch(self, r):
    s, ab, aq, s2, o = super().gather_branch(r)
    return s, ab, aq, s2, o, self.kh_b[r], self.kb_b[r], self.stat_b[r]

  def gather_log(self, i):
    s, ab, aq, s2, o = super().gather_log(i)
    return s, ab, aq, s2, o, self.kh_l[i], self.kb_l[i], self.stat_l[i]


class BranchWindows:
  """K-step windows inside the sealed table's branch paths (the start agent's continuation): rows r..r+K-1 valid and in one path,
  gathered from the branch obs / act rows; 'train' = the fold's training rows, 'es' = the early-stop episodes (selection)."""

  def __init__(self, R, sp):
    self.R = R
    path = np.repeat(np.arange(len(R.boff)), R.blen)
    self.path = path
    xy = R.bobs[:, :2]; sp_ = np.zeros(len(xy), np.float32); sp_[:-1] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    self.speed = sp_
    self.rows = {'train': np.zeros(len(R.valid), bool), 'es': np.zeros(len(R.valid), bool)}
    self.rows['train'][sp['train_b']] = True; self.rows['es'][sp['es_b']] = True

  def windows(self, K, split='train'):
    n = len(self.R.valid); i = np.arange(n - K)
    ok = self.rows[split][i] & self.R.valid[i + K - 1] & (self.path[i] == self.path[i + K - 1])     # one anchor per path -> one fold / one episode per window
    starts = i[ok]; stall = starts[self.speed[starts] < SLOW_SPEED]
    return starts, stall

  def gather_window(self, starts, K):
    idx = starts[:, None] + np.arange(K + 1)[None, :]
    return self.R.bobs[idx].astype(np.float32), self.R.bact[idx[:, :-1]].astype(np.float32)


class CFRows:
  """The CF-controlled motion rows: (s, a_q, s2) with the static flag, by split (train for the fit, val for selection)."""

  def __init__(self, path):
    with np.load(path, allow_pickle=False) as d:
      self.s, self.a, self.s2, self.static = d['s'], d['a'], d['s2'], d['static']
      self.rollout, self.speed, self.slow = d['rollout'], d['speed'], d['slow']
      sp = d['roll_split'].astype(str)[d['rollout']]
    self.idx = {k: np.flatnonzero(sp == k) for k in ('train', 'val', 'test')}
    self.split_row = sp

  def windows(self, K, split='train'):
    """Valid window starts i (rows i..i+K-1 in one rollout, all in `split`) and the subset starting on a slow / static row."""
    n = len(self.s); i = np.arange(n - K)
    ok = (self.rollout[i] == self.rollout[i + K - 1]) & (self.split_row[i] == split)
    starts = i[ok]; stall = starts[(self.speed[starts] < 0.03)]
    return starts, stall

  def gather_window(self, starts, K):
    """S [B, K+1, 29] (the real states s_0..s_K) and A [B, K, 8]."""
    idx = starts[:, None] + np.arange(K)[None, :]
    S = np.concatenate([self.s[idx], self.s2[idx[:, -1:]]], axis=1); A = self.a[idx]
    return S.astype(np.float32), A.astype(np.float32)

  def gather(self, i):
    return self.s[i], self.a[i], self.s2[i], self.static[i]


# ------------------------------------------------------------------- model
def build_nets():
  import haiku as hk
  mot, ons = V1.build_nets()

  def stat(x):
    return hk.nets.MLP(list(STAT_HIDDEN) + [1])(x)[:, 0]
  return mot, ons, hk.without_apply_rng(hk.transform(stat))


def model_path(fold):
  return OUT / f'model_fold{fold}.pkl'


def load_model(fold):
  with model_path(fold).open('rb') as f:
    return pickle.load(f)


class Predictor2:
  """v2 step: the stationary gate (atom), the motion regression, the onset head with the two history features."""

  def __init__(self, md):
    import jax
    import jax.numpy as jnp
    mot, ons, stt = build_nets()
    pm = jax.tree_util.tree_map(jnp.asarray, md['motion_params']); po = jax.tree_util.tree_map(jnp.asarray, md['onset_params']); ps = jax.tree_util.tree_map(jnp.asarray, md['stat_params'])
    self.norm = md['norm']
    self.sq = md.get('motion_inputs') == 'sq'                          # v4: motion / stationary inputs are (s, a_q)
    self._delta = jax.jit(lambda x: mot.apply(pm, x))
    self._logit = jax.jit(lambda x, d, h: ons.apply(po, jnp.concatenate([x, d, h], axis=1)))
    self._stat = jax.jit(lambda x: jax.nn.sigmoid(stt.apply(ps, x)))

  def step(self, s, ab, aq, kh, kb):
    import jax
    x = features(s, ab, aq, self.norm); h = hist_features(np.asarray(kh), np.asarray(kb))
    xm = features_q(s, aq, self.norm) if self.sq else x
    d = np.asarray(self._delta(xm)); p_stat = np.asarray(self._stat(xm))
    stat = p_stat > 0.5
    d_real = d * self.norm['d_std'] + self.norm['d_mean']
    d_real[stat] = 0.0; d_in = d.copy(); d_in[stat] = (0.0 - self.norm['d_mean']) / self.norm['d_std']
    p = np.asarray(jax.nn.sigmoid(self._logit(x, d_in.astype(np.float32), h)))
    return (s + d_real).astype(np.float32), p, p_stat


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'one-step ETT revision v2: stationary gate + zero-displacement atom; onset head with visible-history features (steps in hold, steps in band)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'oracle-supervised engineering stage; disclosed revision after the Step 3a diagnosis (ett_rollout/SUMMARY.md)',
         'v1_manifest_sha256': MP.sha256(V1.OUT / 'manifest.json'), 'v1_check_sha256': MP.sha256(V1.OUT / 'check.json'), 'step3a_report_sha256': MP.sha256(MP.OUT / 'ett_rollout' / 'report.json'),
         'supervision_sha256': MP.sha256(EC / 'supervision.npz'),
         'changes': {'stationary_gate': f'P(stationary | s, a_b, a_q), label max |delta s| < {STAT_TOL}; MLP {STAT_HIDDEN}; BCE on all rows; at prediction delta = 0 when P > 0.5; motion regression trained on moving rows only',
                     'onset_history': f'two features on the onset head: consecutive hold steps (advice max |a_b| < {HOLD_TOL}) and steps since band entry, each min(k, {HIST_CAP:.0f}) / {HIST_CAP:.0f}; computed from the path itself'},
         'v4': ({'motion_stationary_inputs': '(s, a_q) only: the executed torque decides the motion; the advice a_b stays an input of the onset head only', 'why': 'ett_motion_ab: the stationary gate flipped 9-50 % under advice swaps at stall / pre-mouth / start rows and the advice source moved closed-loop reach by up to 0.12; the physical audit found one-step motion independent of the hidden context'} if V4 else None),
         'v3': ({'regression_rows': 'all rows (stationary included)', 'loss': f'standardised MSE + {REL_W} * relative error ||pred - d||^2 / (||d||^2 + {REL_EPS}^2)', 'selection': 'mse_diag + mse_off + 0.25 * onset_bce (no stationary BCE)', 'why': 'v2 did not decelerate after the torque became small (settling probe); fold 1 early-stopped at 4k on the stationary BCE'} if V3 else None),
         'unchanged': 'inputs, motion MLP, onset MLP, optimiser, budget, batches, cross-fitting, selection (+ 0.25 * stationary BCE)', 'gates': GATES2,
         'gate_rule': 'G1 / G2 on moving held-out rows; G3-G6 as v1; G7 stationary gate AUROC and accuracy on held-out rows, and the atom\'s one-step xy error on stationary rows; any failure stops'}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# --------------------------------------------------------------------- fit
def mode_fit(args):
  import jax
  import jax.numpy as jnp
  import optax
  OUT.mkdir(parents=True, exist_ok=True)
  f = int(args.fold); out = model_path(f)
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  steps = int(args.steps or FIT['updates'])
  R = Rows2(); sp = R.split(f)
  rng = np.random.default_rng(FIT['sample_seed'] + 100 + f)
  sub_b = rng.choice(sp['train_b'], size=min(400_000, len(sp['train_b'])), replace=False); sub_l = rng.choice(sp['train_l'], size=min(100_000, len(sp['train_l'])), replace=False)
  s_b, _, _, s2_b, _, _, _, st_b = R.gather_branch(sub_b); s_l, _, _, s2_l, _, _, _, st_l = R.gather_log(sub_l)
  S_ = np.concatenate([s_b, s_l]); D_ = np.concatenate([(s2_b - s_b)[~st_b], (s2_l - s_l)[~st_l]])      # delta statistics on MOVING rows
  norm = {'s_mean': S_.mean(0).astype(np.float32), 's_std': np.maximum(S_.std(0), 1e-4).astype(np.float32), 'd_mean': D_.mean(0).astype(np.float32), 'd_std': np.maximum(D_.std(0), 1e-5).astype(np.float32)}
  init_md = None
  if INIT_FROM is not None:
    with (Path(INIT_FROM) / f'model_fold{f}.pkl').open('rb') as fh:
      init_md = pickle.load(fh)
    assert (init_md.get('motion_inputs') == 'sq') == V4, 'init model / flags mismatch'
    norm = init_md['norm']                                                   # the standardisation stays the init model's (continuity of the inputs)
  CF = CFRows(CF_ROWS) if CF_ROWS else None
  if CF is not None:
    print(f'fold {f}: CF rows train {len(CF.idx["train"])} val {len(CF.idx["val"])} (share {CF_SHARE} of each motion / stationary batch)', flush=True)
  pos_b = R.br_diag[np.searchsorted(R.br, sp['train_b'])]
  mov_b = ~R.stat_b[sp['train_b']]; mov_l = ~R.stat_l[sp['train_l']]
  if V3:                                     # v3: the regression sees every row (the atom still applies at prediction)
    mov_b = np.ones_like(mov_b); mov_l = np.ones_like(mov_l)
  diag_b = sp['train_b'][pos_b & mov_b]; off_b = sp['train_b'][~pos_b & mov_b]; log_mov = sp['train_l'][mov_l]
  ons_pos = sp['train_b'][R.onset[sp['train_b']]]; ons_neg = sp['train_b'][~R.onset[sp['train_b']]]
  all_b, all_l = sp['train_b'], sp['train_l']
  print(f'fold {f}: train branch rows {len(all_b)} (moving diag {len(diag_b)}, moving off {len(off_b)}, stationary {int((~mov_b).sum())}, onsets {len(ons_pos)}), log rows {len(all_l)} (moving {len(log_mov)})', flush=True)
  prevalence = len(ons_pos) / (len(all_b) + len(all_l))
  # validation sample
  vrng = np.random.default_rng(FIT['val_seed'] + 100 + f)
  es_pos = sp['es_b'][R.onset[sp['es_b']]]; es_rest = sp['es_b'][~R.onset[sp['es_b']]]
  vb = vrng.choice(es_rest, size=min(100_000, len(es_rest)), replace=False); vl = vrng.choice(sp['es_l'], size=min(100_000, len(sp['es_l'])), replace=False)
  V = [R.gather_branch(np.concatenate([es_pos, vb])), R.gather_log(vl)]
  Vs, Vab, Vaq, Vs2, Vo, Vkh, Vkb, Vst = (np.concatenate([v[i] for v in V]) for i in range(8))
  VX = features(Vs, Vab, Vaq, norm); VH = hist_features(Vkh, Vkb); VD = ((Vs2 - Vs) - norm['d_mean']) / norm['d_std']; Vdiag = np.abs(Vab - Vaq).max(axis=1) <= FIT['tol_diag']
  VXm = features_q(Vs, Vaq, norm) if V4 else VX
  DM = (STATE_DIM + 8) if V4 else 61
  es_prev = len(es_pos) / (len(sp['es_b']) + len(sp['es_l']))
  cf_val = None
  if CF is not None:
    cs, ca, cs2, cst = CF.gather(CF.idx['val'][:200_000])
    cf_val = {'xm': jnp.asarray(features_q(cs, ca, norm) if V4 else features(cs, ca, ca, norm)), 'd': jnp.asarray(((cs2 - cs) - norm['d_mean']) / norm['d_std'], jnp.float32), 'mov': jnp.asarray((~cst).astype(np.float32)), 'st': jnp.asarray(cst.astype(np.float32))}
  Vw = np.where(Vo, es_prev / max(Vo.mean(), 1e-9), (1 - es_prev) / max(1 - Vo.mean(), 1e-9)).astype(np.float32)
  mot, ons, stt = build_nets()
  key = jax.random.PRNGKey(200 + f)
  pm = mot.init(key, jnp.zeros((1, DM), jnp.float32)); po = ons.init(jax.random.fold_in(key, 1), jnp.zeros((1, 61 + STATE_DIM + 2), jnp.float32)); ps = stt.init(jax.random.fold_in(key, 2), jnp.zeros((1, DM), jnp.float32))
  if init_md is not None:
    pm = jax.tree_util.tree_map(jnp.asarray, init_md['motion_params']); po = jax.tree_util.tree_map(jnp.asarray, init_md['onset_params']); ps = jax.tree_util.tree_map(jnp.asarray, init_md['stat_params'])
  opt_m, opt_o, opt_s = optax.adam(FIT['lr']), optax.adam(FIT['lr']), optax.adam(FIT['lr']); om, oo, os_ = opt_m.init(pm), opt_o.init(po), opt_s.init(ps)
  pm0, po0, ps0 = (jax.tree_util.tree_map(np.asarray, p_) for p_ in (pm, po, ps))
  BD = int(FIT['batch_diag'])

  def mloss(p, x, d):
    pred = mot.apply(p, x); se = jnp.mean((pred - d) ** 2, axis=1)
    loss = jnp.mean(se[:BD]) + jnp.mean(se[BD:])
    if V3:
      rel = jnp.sum((pred - d) ** 2, axis=1) / (jnp.sum(d ** 2, axis=1) + REL_EPS ** 2)
      loss = loss + REL_W * (jnp.mean(rel[:BD]) + jnp.mean(rel[BD:]))
    return loss, (jnp.mean(se[:BD]), jnp.mean(se[BD:]))

  def oloss(p, x, d, h, y, w):
    return jnp.mean(w * optax.sigmoid_binary_cross_entropy(ons.apply(p, jnp.concatenate([x, d, h], axis=1)), y))

  def sloss(p, x, y):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(stt.apply(p, x), y))

  @jax.jit
  def step_m(p, o, x, d):
    (l, parts), g = jax.value_and_grad(mloss, has_aux=True)(p, x, d); up, o = opt_m.update(g, o, p); return optax.apply_updates(p, up), o, l, parts

  @jax.jit
  def step_o(p, o, pm_, x, xm, h, y, w):
    d_pred = jax.lax.stop_gradient(mot.apply(pm_, xm)); l, g = jax.value_and_grad(oloss)(p, x, d_pred, h, y, w); up, o = opt_o.update(g, o, p); return optax.apply_updates(p, up), o, l

  @jax.jit
  def step_s(p, o, x, y):
    l, g = jax.value_and_grad(sloss)(p, x, y); up, o = opt_s.update(g, o, p); return optax.apply_updates(p, up), o, l

  @jax.jit
  def val_fn(pm_, po_, ps_, x, xm, d, h, diag, mov, y, w, st):
    pred = mot.apply(pm_, xm); se = jnp.mean((pred - d) ** 2, axis=1)
    md = jnp.sum(se * diag * mov) / jnp.maximum(jnp.sum(diag * mov), 1); mo = jnp.sum(se * (1 - diag) * mov) / jnp.maximum(jnp.sum((1 - diag) * mov), 1)
    logit = ons.apply(po_, jnp.concatenate([x, jax.lax.stop_gradient(pred), h], axis=1)); bce = jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, y))
    sl = stt.apply(ps_, xm); sbce = jnp.mean(optax.sigmoid_binary_cross_entropy(sl, st)); sacc = jnp.mean((sl > 0) == (st > 0.5))
    return md, mo, bce, jax.nn.sigmoid(logit), sbce, sacc
  WSRC = (BranchWindows(R, sp) if MS_SOURCE == 'branch' else CF) if MULTISTEP > 0 else None
  ms_starts, ms_stall = (WSRC.windows(MULTISTEP, 'train') if WSRC is not None else (None, None))
  if ms_starts is not None:
    print(f'fold {f}: multistep K {MULTISTEP} from {MS_SOURCE}: {len(ms_starts)} train windows ({len(ms_stall)} starting on slow / static rows), weight {MS_WEIGHT}, batch {MS_BATCH}, stall share {MS_STALL_SHARE}', flush=True)
    ms_val_starts, _ = WSRC.windows(MULTISTEP, 'es' if MS_SOURCE == 'branch' else 'val'); ms_val_starts = np.random.default_rng(5).choice(ms_val_starts, size=min(4096, len(ms_val_starts)), replace=False)
    VS_ms, VA_ms = WSRC.gather_window(ms_val_starts, MULTISTEP)
    print(f'fold {f}: multistep selection windows {len(ms_val_starts)} from the {"early-stop episodes" if MS_SOURCE == "branch" else "CF val split"}', flush=True)
  nS_mean, nS_std, nD_mean, nD_std = (jnp.asarray(norm[k]) for k in ('s_mean', 's_std', 'd_mean', 'd_std'))

  def unroll(p, S0, A):
    """The raw regression unrolled on its own predictions from the real start state (no gate): predicted states s_1..s_K [B, K, 29]."""
    s = S0; outs = []
    for j in range(A.shape[1]):
      x = jnp.concatenate([(s - nS_mean) / nS_std, A[:, j]], axis=1) if V4 else None
      d = mot.apply(p, x) * nD_std + nD_mean; s = s + d; outs.append(s)
    return jnp.stack(outs, axis=1)

  def msloss(p, S, A):
    pred = unroll(p, S[:, 0], A); err = (pred - S[:, 1:]) / nS_std
    return jnp.mean(err ** 2)

  @jax.jit
  def step_m2(p, o, x, d, S, A):
    def total(p_):
      l1, parts = mloss(p_, x, d); l2 = msloss(p_, S, A); return l1 + MS_WEIGHT * l2, (parts, l2)
    (l, (parts, l2)), g = jax.value_and_grad(total, has_aux=True)(p); up, o = opt_m.update(g, o, p); return optax.apply_updates(p, up), o, l, parts, l2

  @jax.jit
  def ms_val_fn(p, S, A):
    pred = unroll(p, S[:, 0], A); err = (pred - S[:, 1:]) / nS_std
    xy = jnp.linalg.norm(pred[:, :, :2] - S[:, 1:, :2], axis=2)
    return jnp.mean(err ** 2), jnp.median(xy[:, -1])

  @jax.jit
  def cf_val_fn(pm_, ps_, xm, d, mov, st):
    pred = mot.apply(pm_, xm); se = jnp.mean((pred - d) ** 2, axis=1)
    mse_mov = jnp.sum(se * mov) / jnp.maximum(jnp.sum(mov), 1); mse_all = jnp.mean(se)
    sl = stt.apply(ps_, xm); sacc = jnp.mean((sl > 0) == (st > 0.5))
    return mse_mov, mse_all, sacc
  wpos, wneg = np.float32(prevalence / 0.5), np.float32((1 - prevalence) / 0.5)
  hist, best, best_state, best_step, t0 = [], float('inf'), None, 0, time.time()
  bd, bo, bl = FIT['batch_diag'], FIT['batch_off'], FIT['onset_batch_per_class']
  Vmov = (~Vst).astype(np.float32)
  k_d = int(CF_SHARE * bd) if CF is not None else 0; k_o = int(CF_SHARE * bo) if CF is not None else 0; k_s = int(CF_SHARE * 2048) if CF is not None else 0
  for it in range(1, steps + 1):
    i_d = rng.choice(diag_b, size=(bd - k_d) // 2); i_l = rng.choice(log_mov, size=(bd - k_d) - (bd - k_d) // 2); i_o = rng.choice(off_b, size=bo - k_o)
    g1 = R.gather_branch(i_d); g2 = R.gather_log(i_l); g3 = R.gather_branch(i_o)
    s = np.concatenate([g1[0], g2[0], g3[0]]); ab = np.concatenate([g1[1], g2[1], g3[1]]); aq = np.concatenate([g1[2], g2[2], g3[2]]); s2 = np.concatenate([g1[3], g2[3], g3[3]])
    if CF is not None:                                                        # CF rows replace a share of the diag block and of the off block (the batch size unchanged)
      c1 = CF.gather(rng.choice(CF.idx['train'], size=k_d)); c2 = CF.gather(rng.choice(CF.idx['train'], size=k_o))
      s = np.concatenate([s[:bd - k_d], c1[0], s[bd - k_d:], c2[0]]); aq = np.concatenate([aq[:bd - k_d], c1[1], aq[bd - k_d:], c2[1]]); s2 = np.concatenate([s2[:bd - k_d], c1[2], s2[bd - k_d:], c2[2]])
      ab = np.concatenate([ab[:bd - k_d], c1[1], ab[bd - k_d:], c2[1]])
    x = features(s, ab, aq, norm); d = ((s2 - s) - norm['d_mean']) / norm['d_std']
    xm = features_q(s, aq, norm) if V4 else x
    if ms_starts is not None:
      n_st = int(MS_STALL_SHARE * MS_BATCH); w0 = np.concatenate([rng.choice(ms_stall, size=n_st), rng.choice(ms_starts, size=MS_BATCH - n_st)])
      Sw, Aw = WSRC.gather_window(w0, MULTISTEP)
      pm, om, lm, parts, lms = step_m2(pm, om, jnp.asarray(xm), jnp.asarray(d, jnp.float32), jnp.asarray(Sw), jnp.asarray(Aw))
    else:
      pm, om, lm, parts = step_m(pm, om, jnp.asarray(xm), jnp.asarray(d, jnp.float32)); lms = jnp.float32(0.0)
    ip = rng.choice(ons_pos, size=bl); ineg = rng.choice(ons_neg, size=bl)
    gp = R.gather_branch(ip); gn = R.gather_branch(ineg)
    xo = features(np.concatenate([gp[0], gn[0]]), np.concatenate([gp[1], gn[1]]), np.concatenate([gp[2], gn[2]]), norm); ho = hist_features(np.concatenate([gp[5], gn[5]]), np.concatenate([gp[6], gn[6]]))
    xom = features_q(np.concatenate([gp[0], gn[0]]), np.concatenate([gp[2], gn[2]]), norm) if V4 else xo
    yo = np.concatenate([np.ones(bl), np.zeros(bl)]).astype(np.float32); wo = np.concatenate([np.full(bl, wpos), np.full(bl, wneg)]).astype(np.float32)
    if not FREEZE_ONSET:
      po, oo, lo = step_o(po, oo, pm, jnp.asarray(xo), jnp.asarray(xom), jnp.asarray(ho), jnp.asarray(yo), jnp.asarray(wo))
    else:
      lo = jnp.float32(0.0)
    # stationary gate: a natural mix of branch and log rows (+ the CF share)
    is_ = rng.choice(all_b, size=1536 - (k_s * 3) // 4); il_ = rng.choice(all_l, size=512 - (k_s - (k_s * 3) // 4))
    gs = R.gather_branch(is_); gl = R.gather_log(il_)
    ss_, sab_, saq_, sy_ = np.concatenate([gs[0], gl[0]]), np.concatenate([gs[1], gl[1]]), np.concatenate([gs[2], gl[2]]), np.concatenate([gs[7], gl[7]])
    if CF is not None:
      c3 = CF.gather(rng.choice(CF.idx['train'], size=k_s)); ss_ = np.concatenate([ss_, c3[0]]); sab_ = np.concatenate([sab_, c3[1]]); saq_ = np.concatenate([saq_, c3[1]]); sy_ = np.concatenate([sy_, c3[3]])
    xs = (features_q(ss_, saq_, norm) if V4 else features(ss_, sab_, saq_, norm)); ys = sy_.astype(np.float32)
    ps, os_, ls = step_s(ps, os_, jnp.asarray(xs), jnp.asarray(ys))
    if it % FIT['val_every'] == 0 or it == steps:
      md_, mo_, bce, prob, sbce, sacc = (np.asarray(v) for v in val_fn(pm, po, ps, jnp.asarray(VX), jnp.asarray(VXm), jnp.asarray(VD, jnp.float32), jnp.asarray(VH), jnp.asarray(Vdiag.astype(np.float32)), jnp.asarray(Vmov), jnp.asarray(Vo.astype(np.float32)), jnp.asarray(Vw), jnp.asarray(Vst.astype(np.float32))))
      auc = auroc(prob, Vo); score = float(md_ + mo_ + FIT['onset_weight'] * bce + (0.0 if V3 else 0.25) * sbce)
      cfm = {}
      if cf_val is not None:
        c_mov, c_all, c_acc = (float(np.asarray(v)) for v in cf_val_fn(pm, ps, cf_val['xm'], cf_val['d'], cf_val['mov'], cf_val['st']))
        cfm = {'cf_val_mse_moving': c_mov, 'cf_val_mse_all': c_all, 'cf_val_stat_acc': c_acc}; score = score + c_mov      # the same selection rule for the control and the revision
      if ms_starts is not None:
        m_mse, m_xyK = (float(np.asarray(v)) for v in ms_val_fn(pm, jnp.asarray(VS_ms), jnp.asarray(VA_ms)))
        cfm.update({'ms_val_mse': m_mse, f'ms_val_xy_err_at_{MULTISTEP}': m_xyK, 'train_multistep': float(lms)})
        if MS_SOURCE == 'branch':
          score = score + m_mse                                              # selection on the target protocol: + the K-step MSE on the early-stop episodes' branch windows
      hist.append({'step': it, 'train_motion': float(lm), 'train_onset': float(lo), 'train_stat': float(ls), 'val_mse_diag': float(md_), 'val_mse_off': float(mo_), 'val_onset_bce': float(bce), 'val_auroc': auc, 'val_stat_bce': float(sbce), 'val_stat_acc': float(sacc), 'score': score, **cfm})
      if score < best:
        best, best_step = score, it
        best_state = {k: jax.tree_util.tree_map(np.asarray, p_) for k, p_ in (('motion_params', pm), ('onset_params', po), ('stat_params', ps))}
      print(f'[v2 fold {f} step {it:>6}] motion {float(lm):.4f} onset {float(lo):.4f} stat {float(ls):.4f} | val mse diag {float(md_):.4f} off {float(mo_):.4f} onset bce {float(bce):.4f} auroc {auc:.3f} stat bce {float(sbce):.4f} acc {float(sacc):.4f}' + (f" | cf val mse mov {cfm['cf_val_mse_moving']:.4f} all {cfm['cf_val_mse_all']:.4f} stat acc {cfm['cf_val_stat_acc']:.4f}" if 'cf_val_mse_moving' in cfm else '') + (f" | ms val mse {cfm['ms_val_mse']:.4f} xy@K {cfm[f'ms_val_xy_err_at_{MULTISTEP}']:.4f}" if 'ms_val_mse' in cfm else '') + f' | score {score:.4f} best {best:.4f}@{best_step} {it / (time.time() - t0):.1f} it/s', flush=True)
  md = {**best_state, 'norm': norm, 'fold': f, 'best_step': best_step, 'best_score': best, 'steps': steps, 'fit': FIT, 'history': hist, 'prevalence_train': prevalence, 'es_episodes': sp['es_episodes'],
        'stat_tol': STAT_TOL, 'hold_tol': HOLD_TOL, 'hist_cap': HIST_CAP, 'wall_seconds': time.time() - t0, 'motion_inputs': ('sq' if V4 else 'sabq'), 'version': ('v4' if V4 else ('v3' if V3 else 'v2')),
        'param_delta': {k: float(optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, best_state[k], p0))) for k, p0 in (('motion_params', pm0), ('onset_params', po0), ('stat_params', ps0))},
        'supervision_sha256': MP.sha256(EC / 'supervision.npz'),
        'init_from': (str(INIT_FROM) if INIT_FROM else None), 'cf_rows': (str(CF_ROWS) if CF_ROWS else None), 'cf_share': (CF_SHARE if CF_ROWS else 0.0), 'freeze_onset': FREEZE_ONSET,
        'best_cf_val': next((h for h in hist if h['step'] == best_step), {}), 'selection': 'original validation score' + (' + CF-val one-step MSE (moving rows)' if CF_ROWS else '') + (' + K-step MSE on the early-stop episodes\' branch windows' if (MULTISTEP > 0 and MS_SOURCE == 'branch') else ''),
        'multistep': {'K': MULTISTEP, 'weight': MS_WEIGHT, 'batch': MS_BATCH, 'stall_share': MS_STALL_SHARE, 'source': MS_SOURCE, 'n_train_windows': int(len(ms_starts)), 'n_train_windows_slow': int(len(ms_stall)),
                      'unroll': 'raw regression on its own predictions from the real start (no gate); MSE in standardised state units'} if MULTISTEP > 0 else None}
  with out.open('wb') as fh:
    pickle.dump(md, fh)
  MP.write_json(out.with_suffix('.json'), {k: v for k, v in md.items() if k not in ('motion_params', 'onset_params', 'stat_params', 'norm')})
  print(f'saved {out}: best score {best:.4f} at {best_step} ({time.time() - t0:.0f} s)', flush=True)


# ------------------------------------------------------------------- check
def mode_check(args):
  R = Rows2()
  res = {'per_fold': {}, 'gates': GATES2}
  pooled = {k: [] for k in ('xy_err', 'delta_se', 'xy10', 'xy50', 'prob', 'onset', 'cls', 'in_band', 'stat_p', 'stat_y', 'xy_err_stat')}
  for f in range(N_FOLDS):
    md = load_model(f); P = Predictor2(md); sp = R.split(f)
    ho = sp['ho_b']; rng = np.random.default_rng(FIT['val_seed'] + 110 + f)
    idx = rng.choice(ho, size=min(300_000, len(ho)), replace=False)
    s, ab, aq, s2, o, kh, kb, st = R.gather_branch(idx)
    pred, prob, p_stat = P.step(s, ab, aq, kh, kb)
    xy_err = np.linalg.norm(pred[:, :2] - s2[:, :2], axis=1)
    dse = np.mean((((pred - s) - md['norm']['d_mean']) / md['norm']['d_std'] - ((s2 - s) - md['norm']['d_mean']) / md['norm']['d_std']) ** 2, axis=1)
    mov = ~st
    ho_pos = ho[R.onset[ho]]; s_p, ab_p, aq_p, _, _, kh_p, kb_p, _ = R.gather_branch(ho_pos); _, prob_p, _ = P.step(s_p, ab_p, aq_p, kh_p, kb_p)
    prob_all = np.concatenate([prob[~o], prob_p]); onset_all = np.concatenate([np.zeros((~o).sum(), bool), np.ones(len(ho_pos), bool)])
    cls_all = np.concatenate([R.cls[idx][~o], R.cls[ho_pos]]); x_all = np.concatenate([s[~o, 0], s_p[:, 0]]); y_all = np.concatenate([s[~o, 1], s_p[:, 1]])
    band = in_band(np.stack([x_all, y_all], 1))
    # open-loop rollouts on held-out roots with recorded advice / torques and the history features from the recorded path
    roots = rng.choice(np.flatnonzero(R.fold_anchor == f), size=min(2000, int((R.fold_anchor == f).sum())), replace=False); roots = roots[R.blen[roots] >= ROLL_H + 1]
    off = R.boff[roots]; cur = R.bobs[off].copy(); e10 = None
    for j in range(ROLL_H):
      cur, _, _ = P.step(cur, R.adv[off + j], R.bact[off + j], R.kh_b[off + j], R.kb_b[off + j])
      if j + 1 == 10:
        e10 = np.linalg.norm(cur[:, :2] - R.bobs[off + 10, :2], axis=1)
    e50 = np.linalg.norm(cur[:, :2] - R.bobs[off + ROLL_H, :2], axis=1)
    prev_ho = len(ho_pos) / len(ho)
    fr = {'n_held_out_rows': int(len(ho)), 'n_eval_rows': int(len(idx)), 'stationary_share': float(st.mean()),
          'xy_err_median_moving': float(np.median(xy_err[mov])), 'xy_err_p90_moving': float(np.percentile(xy_err[mov], 90)), 'xy_err_p99_moving': float(np.percentile(xy_err[mov], 99)), 'delta_rmse_moving': float(np.sqrt(dse[mov].mean())),
          'xy_err_median_stationary_with_atom': float(np.median(xy_err[st])), 'xy_err_p99_stationary_with_atom': float(np.percentile(xy_err[st], 99)),
          'stat_gate_auroc': auroc(p_stat, st), 'stat_gate_accuracy': float(((p_stat > 0.5) == st).mean()), 'stat_gate_fires_share': float((p_stat > 0.5).mean()),
          'xy10_median': float(np.median(e10)), 'xy50_median': float(np.median(e50)), 'xy50_p90': float(np.percentile(e50, 90)), 'n_rollouts': int(len(roots)),
          'onset_auroc': auroc(prob_all, onset_all), 'mean_prob_on_onsets': float(prob_p.mean()), 'mean_prob_on_non_onsets': float(prob[~o].mean()), 'held_out_prevalence': prev_ho, 'n_held_out_onsets': int(len(ho_pos)),
          'mean_prob_all_rows': float(prob.mean()), 'best_step': md['best_step'], 'param_delta': md['param_delta']}
    hb, db = band & (cls_all == 'off_hold'), band & (cls_all != 'off_hold')
    fr['band_mean_prob_off_hold'] = float(prob_all[hb].mean()) if hb.any() else None; fr['band_mean_prob_drive'] = float(prob_all[db].mean()) if db.any() else None
    res['per_fold'][str(f)] = fr
    for k, v in (('xy_err', xy_err[mov]), ('delta_se', dse[mov]), ('xy10', e10), ('xy50', e50), ('prob', prob_all), ('onset', onset_all), ('cls', cls_all), ('in_band', band), ('stat_p', p_stat), ('stat_y', st), ('xy_err_stat', xy_err[st])):
      pooled[k].append(v)
    print(f'fold {f}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fr.items() if k != "param_delta"})}', flush=True)
  C = {k: np.concatenate(v) for k, v in pooled.items()}
  prev = float(np.mean([res['per_fold'][str(f)]['held_out_prevalence'] for f in range(N_FOLDS)])); mean_all = float(np.mean([res['per_fold'][str(f)]['mean_prob_all_rows'] for f in range(N_FOLDS)]))
  hb, db = C['in_band'] & (C['cls'] == 'off_hold'), C['in_band'] & (C['cls'] != 'off_hold')
  pr = {'xy_err_median_moving': float(np.median(C['xy_err'])), 'xy_err_p99_moving': float(np.percentile(C['xy_err'], 99)), 'delta_rmse_moving': float(np.sqrt(C['delta_se'].mean())),
        'xy10_median': float(np.median(C['xy10'])), 'xy50_median': float(np.median(C['xy50'])), 'onset_auroc': auroc(C['prob'], C['onset']),
        'ratio_onsets_vs_non': float(C['prob'][C['onset']].mean() / max(C['prob'][~C['onset']].mean(), 1e-9)), 'calibration_mean_over_prevalence': mean_all / max(prev, 1e-9),
        'advice_ratio_in_band': float(C['prob'][hb].mean() / max(C['prob'][db].mean(), 1e-9)) if hb.any() and db.any() else None,
        'stat_gate_auroc': auroc(C['stat_p'], C['stat_y']), 'stat_gate_accuracy': float(((C['stat_p'] > 0.5) == C['stat_y']).mean()), 'atom_xy_err_median_stationary': float(np.median(C['xy_err_stat']))}
  g = {'G1_xy_median': pr['xy_err_median_moving'] < GATES2['G1_xy_median'], 'G1_xy_p99': pr['xy_err_p99_moving'] < GATES2['G1_xy_p99'], 'G1_delta_rmse': pr['delta_rmse_moving'] < GATES2['G1_delta_rmse'],
       'G2_xy50': pr['xy50_median'] < GATES2['G2_xy50_median'], 'G2_xy10': pr['xy10_median'] < GATES2['G2_xy10_median'], 'G3_auroc': pr['onset_auroc'] >= GATES2['G3_auroc'],
       'G4_ratio': pr['ratio_onsets_vs_non'] >= GATES2['G4_ratio'], 'G5_calibration': GATES2['G5_calibration'][0] <= pr['calibration_mean_over_prevalence'] <= GATES2['G5_calibration'][1],
       'G6_advice': (pr['advice_ratio_in_band'] is not None and pr['advice_ratio_in_band'] >= GATES2['G6_advice_ratio']),
       'G7_stat_auroc': pr['stat_gate_auroc'] >= GATES2['G7_stat_auroc'], 'G7_stat_accuracy': pr['stat_gate_accuracy'] >= GATES2['G7_stat_accuracy'], 'G7_atom_xy': pr['atom_xy_err_median_stationary'] <= GATES2['G7_atom_xy_median'],
       'param_delta': all(all(x > GATES2['param_delta_min'] for x in v['param_delta'].values()) for v in res['per_fold'].values())}
  res['pooled'] = pr; res['gate_results'] = g; res['all_passed'] = bool(all(g.values()))
  MP.write_json(OUT / 'check.json', res)
  L = ['# One-step ETT v2 preflight (held-out fold rows, cross-fitted)', '', f'`check.json`; gates in `manifest.json`.  **{"ALL GATES PASSED" if res["all_passed"] else "GATE(S) FAILED -- the line stops here"}**', '',
       '| gate | pass |', '|---|---|'] + [f'| {k} | {"yes" if v else "NO"} |' for k, v in g.items()] + ['', '## Pooled', '', '```', json.dumps(pr, indent=1), '```', '', '## Per fold', '',
       '| fold | held-out rows | stationary share | xy err moving median / p90 / p99 | delta RMSE moving | atom xy err stationary median / p99 | stat gate AUROC / acc / fires | 10 / 50-step xy | onset AUROC | P on onsets / non | best step |', '|---|---:|---:|---|---:|---|---|---|---:|---|---:|']
  for f, v in res['per_fold'].items():
    L.append(f'| {f} | {v["n_held_out_rows"]} | {v["stationary_share"]:.3f} | {v["xy_err_median_moving"]:.4f} / {v["xy_err_p90_moving"]:.4f} / {v["xy_err_p99_moving"]:.4f} | {v["delta_rmse_moving"]:.3f} | {v["xy_err_median_stationary_with_atom"]:.5f} / {v["xy_err_p99_stationary_with_atom"]:.4f} | '
             f'{v["stat_gate_auroc"]:.4f} / {v["stat_gate_accuracy"]:.4f} / {v["stat_gate_fires_share"]:.3f} | {v["xy10_median"]:.3f} / {v["xy50_median"]:.3f} | {v["onset_auroc"]:.3f} | {v["mean_prob_on_onsets"]:.4f} / {v["mean_prob_on_non_onsets"]:.5f} | {v["best_step"]} |')
  (OUT / 'CHECK.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'fit', 'check'))
  ap.add_argument('--fold', type=int, default=0)
  ap.add_argument('--steps', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--v3', action='store_true')
  ap.add_argument('--v4', action='store_true', help='v3 + motion / stationary inputs (s, a_q) only; outputs under ett_one_step_v4/')
  ap.add_argument('--init-from', default=None, help='continue from this model dir (fold model params + norm)')
  ap.add_argument('--cf-rows', default=None, help='cf_motion/rollouts_cf.npz: CF-controlled motion rows (train split) in the motion / stationary batches')
  ap.add_argument('--cf-share', type=float, default=0.25)
  ap.add_argument('--freeze-onset', action='store_true')
  ap.add_argument('--out', default=None, help='output model dir (overrides the version default)')
  ap.add_argument('--multistep', type=int, default=0, help='K: short-sequence unroll term on the CF rollouts (0 = off)')
  ap.add_argument('--multistep-weight', type=float, default=1.0)
  ap.add_argument('--multistep-batch', type=int, default=256)
  ap.add_argument('--multistep-stall-share', type=float, default=0.5)
  ap.add_argument('--multistep-source', choices=('cf', 'branch'), default='cf', help='branch: windows from the sealed table\'s start-agent branch paths (the target protocol) under the fold\'s episode split')
  args = ap.parse_args(argv)
  global V3, V4, OUT, INIT_FROM, CF_ROWS, CF_SHARE, FREEZE_ONSET, MULTISTEP, MS_WEIGHT, MS_BATCH, MS_STALL_SHARE, MS_SOURCE
  INIT_FROM, CF_ROWS, CF_SHARE, FREEZE_ONSET, MS_SOURCE = args.init_from, args.cf_rows, float(args.cf_share), bool(args.freeze_onset), args.multistep_source
  MULTISTEP, MS_WEIGHT, MS_BATCH, MS_STALL_SHARE = int(args.multistep), float(args.multistep_weight), int(args.multistep_batch), float(args.multistep_stall_share)
  if args.v4:
    V4 = True; args.v3 = True
  if args.v3:
    V3 = True; OUT = OUT_V4 if V4 else OUT_V3
  if args.out:
    OUT = Path(args.out)
  OUT.mkdir(parents=True, exist_ok=True)
  {'seal': mode_seal, 'fit': mode_fit, 'check': mode_check}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
