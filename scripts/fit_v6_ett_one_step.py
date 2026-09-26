"""Step 2 of the simulator-paired one-step ETT (AntMaze V6): the one-step
model F(s, a_b, a_q) -> s' with a separate onset head, cross-fitted, and
the pre-registered preflight checks.  Oracle-supervised engineering stage.

Supervision (Step 1, `ett_context/supervision.npz` + `branches_cf.npz` +
the d05 log): every valid branch transition (s_j, a_b_j, a_q_j, s_{j+1},
onset_{j+1}) with the teacher's advice walked along the branch under the
redrawn timetable, and every logged transition (a_b = a_q, onset 0).
Diagonal rows (a_b == a_q within 1e-5) and off-diagonal rows are averaged
separately in the motion loss and enter with equal weight (the PointMaze
protocol); the onset head is trained with natural-prevalence BCE on
class-stratified batches; its gradient does not update the motion model.

Model.  Inputs: standardised s (29), a_b (8), a_q (8), a_q - a_b, a_q * a_b
(61).  Motion: MLP 1024-1024 -> standardised delta s (29), MSE (Step 0
showed one-step motion deterministic to ~1e-6, so a deterministic
regression, not a stochastic generator).  Onset: MLP 256-256 on the same
61 features + the predicted delta (stop-gradient) -> logit.  Adam 3e-4,
40,000 updates, motion batch 2,048 diagonal + 2,048 off-diagonal rows,
onset batch 256 + 256; validation every 1,000 updates on the early-stop
slice (10 % of the training episodes); selection = earliest minimum of
mse_diag + mse_off + 0.25 * onset_bce.  Cross-fitting: model f trains on
folds != f (3 folds by source episode, seed 161000001) and is checked --
and, in Step 3, used -- on fold f only.

Preflight gates (pre-registered here; evaluated on the held-out fold rows
of each model, pooled over the three models):
  G1 motion, one step: held-out xy error median < 0.01 and p99 < 0.10 maze
     units; standardised delta RMSE < 0.30 (all 29 dims);
  G2 motion, open loop: 50-step rollouts from the held-out branch roots
     with the recorded (a_b, a_q) sequences -- xy error at step 50 median
     < 1.0 (the corridors are 4 wide), at step 10 median < 0.2;
  G3 onset: held-out AUROC >= 0.85;
  G4 onset: mean predicted onset on actual onsets >= 5 x on non-onsets;
  G5 onset calibration: mean prediction over all held-out valid rows within
     [0.5, 2] x the held-out prevalence;
  G6 advice sensitivity: inside the hazard bands, mean P(onset) at
     off_hold rows >= 2 x at rows where the advice drives (the effect of
     a_b, the analogue of the protocol's query-effect check);
  finite losses / parameters, parameter deltas > 1e-4.
A failed gate stops before any future is generated.  Persistence after
onset is a rollout rule (absorbing), not a model property.

  python scripts/fit_v6_ett_one_step.py seal
  python scripts/fit_v6_ett_one_step.py fit --fold f          # GPU
  python scripts/fit_v6_ett_one_step.py check                 # all three models present
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

EC = MP.OUT / 'ett_context'
OUT = MP.OUT / 'ett_one_step'
STATE_DIM = MP.STATE_DIM
N_FOLDS, ES_FRACTION, ES_SEED = 3, 0.10, 162_000_001
FIT = {'motion_hidden': (1024, 1024), 'onset_hidden': (256, 256), 'lr': 3e-4, 'updates': 40_000, 'batch_diag': 2048, 'batch_off': 2048, 'onset_batch_per_class': 256,
       'onset_weight': 0.25, 'val_every': 1000, 'val_rows_max': 200_000, 'val_seed': 163_000_000, 'sample_seed': 164_000_000, 'tol_diag': 1e-5}
GATES = {'G1_xy_median': 0.01, 'G1_xy_p99': 0.10, 'G1_delta_rmse': 0.30, 'G2_xy50_median': 1.0, 'G2_xy10_median': 0.2, 'G3_auroc': 0.85, 'G4_ratio': 5.0, 'G5_calibration': (0.5, 2.0),
         'G6_advice_ratio': 2.0, 'param_delta_min': 1e-4}
ZONE_X = {1: (6.6, 9.4), 2: (14.6, 17.4)}
ROLL_H = 50


# -------------------------------------------------------------------- data
class Rows:
  """All supervision rows in memory: branch transitions (valid) and logged transitions, with fold / episode ids."""

  def __init__(self):
    with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
      self.bobs = d['obs_rows'][:, :STATE_DIM]; self.bact = d['act_rows']; self.boff, self.blen = d['offset'].astype(np.int64), d['length'].astype(np.int64)
      self.boc = d['outcome'].astype(str)
    S = np.load(EC / 'supervision.npz', allow_pickle=False)
    self.adv, self.valid, self.onset, self.cls = S['advice_rows'], S['valid_rows'], S['onset_rows'], S['class_rows'].astype(str)
    self.fold_row, self.anchor_of_row, self.j_of_row = S['fold_rows'].astype(np.int64), S['anchor_of_row'].astype(np.int64), S['j_of_row'].astype(np.int64)
    self.fold_anchor, self.log_fold_ep = S['fold_anchor'].astype(np.int64), S['log_fold_episode'].astype(np.int64)
    self.anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
    obs, act, lengths, _ = MP.load_dataset()
    self.lobs, self.lact, self.lL = obs[:, :, :STATE_DIM], act, lengths.astype(np.int64)
    # branch row table: index r valid -> transition (r, r + 1)
    self.br = np.flatnonzero(self.valid)
    # log row table: (e, t) for t < L - 1
    ee = np.repeat(np.arange(len(lengths)), lengths - 1); tt = np.concatenate([np.arange(L - 1) for L in lengths])
    self.le, self.lt = ee.astype(np.int64), tt.astype(np.int64)
    self.log_fold = self.log_fold_ep[self.le]
    self.br_ep = self.anchors.episode[self.anchor_of_row[self.br]]
    self.br_diag = np.abs(self.adv[self.br] - self.bact[self.br]).max(axis=1) <= FIT['tol_diag']

  def gather_branch(self, r):
    s = self.bobs[r]; s2 = self.bobs[r + 1]
    return s, self.adv[r], self.bact[r], s2, self.onset[r]

  def gather_log(self, i):
    e, t = self.le[i], self.lt[i]
    a = self.lact[e, t]
    return self.lobs[e, t], a, a, self.lobs[e, t + 1], np.zeros(len(i), bool)

  def split(self, fold):
    """Training / early-stop / held-out index sets for model `fold` (held-out = fold f)."""
    train_eps = np.unique(np.concatenate([self.anchors.episode[self.fold_anchor != fold], np.flatnonzero((self.log_fold_ep != fold) & (self.log_fold_ep >= 0))]))
    rng = np.random.default_rng(ES_SEED + fold)
    es_eps = set(rng.choice(train_eps, size=int(np.ceil(ES_FRACTION * len(train_eps))), replace=False).tolist())
    es_mask_b = np.isin(self.br_ep, list(es_eps)); es_mask_l = np.isin(self.le, list(es_eps))
    tr_b = self.br[(self.fold_row[self.br] != fold) & ~es_mask_b]; es_b = self.br[(self.fold_row[self.br] != fold) & es_mask_b]; ho_b = self.br[self.fold_row[self.br] == fold]
    li = np.arange(len(self.le))
    tr_l = li[(self.log_fold != fold) & ~es_mask_l]; es_l = li[(self.log_fold != fold) & es_mask_l]; ho_l = li[self.log_fold == fold]
    return {'train_b': tr_b, 'es_b': es_b, 'ho_b': ho_b, 'train_l': tr_l, 'es_l': es_l, 'ho_l': ho_l, 'es_episodes': sorted(es_eps)}


def features(s, ab, aq, norm):
  sn = (s - norm['s_mean']) / norm['s_std']
  return np.concatenate([sn, ab, aq, aq - ab, aq * ab], axis=1).astype(np.float32)


# ------------------------------------------------------------------- model
def build_nets():
  import haiku as hk
  import jax.numpy as jnp

  def motion(x):
    return hk.nets.MLP(list(FIT['motion_hidden']) + [STATE_DIM])(x)

  def onset(x):
    return hk.nets.MLP(list(FIT['onset_hidden']) + [1])(x)[:, 0]
  return hk.without_apply_rng(hk.transform(motion)), hk.without_apply_rng(hk.transform(onset))


def model_path(fold):
  return OUT / f'model_fold{fold}.pkl'


def load_model(fold):
  with model_path(fold).open('rb') as f:
    return pickle.load(f)


class Predictor:
  def __init__(self, md):
    import jax
    import jax.numpy as jnp
    mot, ons = build_nets()
    pm = jax.tree_util.tree_map(jnp.asarray, md['motion_params']); po = jax.tree_util.tree_map(jnp.asarray, md['onset_params'])
    self.norm = md['norm']
    self._delta = jax.jit(lambda x: mot.apply(pm, x))
    self._logit = jax.jit(lambda x, d: ons.apply(po, jnp.concatenate([x, d], axis=1)))

  def step(self, s, ab, aq):
    """Deterministic next state and onset probability for arrays s [N, 29], ab / aq [N, 8]."""
    import jax
    x = features(s, ab, aq, self.norm)
    d = np.asarray(self._delta(x))
    p = np.asarray(jax.nn.sigmoid(self._logit(x, d)))
    return (s + d * self.norm['d_std'] + self.norm['d_mean']).astype(np.float32), p


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  sup = MP.read_json(EC / 'supervision.json')
  man = {'experiment': 'AntMaze V6 simulator-paired one-step ETT: F(s, a_b, a_q) -> s\' + onset head, cross-fitted; preflight before any future is generated',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(),
         'status': 'ORACLE-SUPERVISED ENGINEERING STAGE: the advice a_b comes from the privileged teacher walked along the simulator branches (Step 0 / 1); not offline identification',
         'supervision': {'file': str(EC / 'supervision.npz'), 'sha256': MP.sha256(EC / 'supervision.npz'), 'composition': sup, 'branches_sha256': MP.sha256(MP.OUT / 'branches_cf.npz'), 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz')},
         'model': {'inputs': 'standardised s (29), a_b (8), a_q (8), a_q - a_b, a_q * a_b', 'motion': 'MLP 1024-1024 -> standardised delta s, MSE; diagonal and off-diagonal rows averaged separately, equal weight',
                   'onset': 'MLP 256-256 on the features + predicted delta (stop-gradient) -> logit; natural-prevalence BCE from class-stratified batches; gradient does not reach the motion model',
                   'fit': FIT, 'selection': 'earliest minimum of mse_diag + mse_off + onset_weight * onset_bce on the early-stop slice', 'cross_fitting': {'n_folds': N_FOLDS, 'unit': 'source episode', 'es_fraction': ES_FRACTION}},
         'gates': GATES, 'gate_rule': 'every gate on the pooled held-out fold rows; any failure stops the line before Step 3',
         'held_fixed_downstream': 'fork query coverage closed; the mainline replay interface (identical logged anchors and weights; generated critic futures for the method arm; unchanged actor / BC sampling; no recorded / generated mixture); bc 0.05, NCE, actor loss, critic clip 0.1'}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# --------------------------------------------------------------------- fit
def mode_fit(args):
  import jax
  import jax.numpy as jnp
  import optax
  OUT.mkdir(parents=True, exist_ok=True)
  f = int(args.fold)
  out = model_path(f)
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  steps = int(args.steps or FIT['updates'])
  R = Rows(); sp = R.split(f)
  # normalisation on the training rows (a subsample for speed)
  rng = np.random.default_rng(FIT['sample_seed'] + f)
  sub_b = rng.choice(sp['train_b'], size=min(400_000, len(sp['train_b'])), replace=False); sub_l = rng.choice(sp['train_l'], size=min(100_000, len(sp['train_l'])), replace=False)
  s_b, _, _, s2_b, _ = R.gather_branch(sub_b); s_l, _, _, s2_l, _ = R.gather_log(sub_l)
  S_ = np.concatenate([s_b, s_l]); D_ = np.concatenate([s2_b - s_b, s2_l - s_l])
  norm = {'s_mean': S_.mean(0).astype(np.float32), 's_std': np.maximum(S_.std(0), 1e-4).astype(np.float32), 'd_mean': D_.mean(0).astype(np.float32), 'd_std': np.maximum(D_.std(0), 1e-5).astype(np.float32)}
  # row pools
  diag_b = sp['train_b'][R.br_diag[np.searchsorted(R.br, sp['train_b'])]]; off_b = sp['train_b'][~R.br_diag[np.searchsorted(R.br, sp['train_b'])]]
  pos_b = sp['train_b'][R.onset[sp['train_b']]]; neg_b = sp['train_b'][~R.onset[sp['train_b']]]
  n_log = len(sp['train_l'])
  print(f'fold {f}: train branch rows {len(sp["train_b"])} (diag {len(diag_b)}, off {len(off_b)}, onsets {len(pos_b)}), log rows {n_log}; es branch {len(sp["es_b"])} log {len(sp["es_l"])}; held-out branch {len(sp["ho_b"])}', flush=True)
  prevalence = len(pos_b) / (len(sp['train_b']) + n_log)
  # validation sample (fixed): all es onsets + up to val_rows_max other rows (branch + log), weighted to the natural prevalence for the onset BCE
  vrng = np.random.default_rng(FIT['val_seed'] + f)
  es_pos = sp['es_b'][R.onset[sp['es_b']]]; es_rest_b = sp['es_b'][~R.onset[sp['es_b']]]
  n_v = FIT['val_rows_max']
  vb = vrng.choice(es_rest_b, size=min(n_v // 2, len(es_rest_b)), replace=False); vl = vrng.choice(sp['es_l'], size=min(n_v // 2, len(sp['es_l'])), replace=False)
  Vs, Vab, Vaq, Vs2, Vo = [], [], [], [], []
  for gather, idx in ((R.gather_branch, np.concatenate([es_pos, vb])), (R.gather_log, vl)):
    s, ab, aq, s2, o = gather(idx); Vs.append(s); Vab.append(ab); Vaq.append(aq); Vs2.append(s2); Vo.append(o)
  Vs, Vab, Vaq, Vs2, Vo = (np.concatenate(v) for v in (Vs, Vab, Vaq, Vs2, Vo))
  VX = features(Vs, Vab, Vaq, norm); VD = ((Vs2 - Vs) - norm['d_mean']) / norm['d_std']; Vdiag = np.abs(Vab - Vaq).max(axis=1) <= FIT['tol_diag']
  es_prev = len(es_pos) / (len(sp['es_b']) + len(sp['es_l']))
  Vw = np.where(Vo, es_prev / max(Vo.mean(), 1e-9), (1 - es_prev) / max(1 - Vo.mean(), 1e-9)).astype(np.float32)     # natural-prevalence reweighting of the validation sample
  mot, ons = build_nets()
  key = jax.random.PRNGKey(100 + f)
  pm = mot.init(key, jnp.zeros((1, 61), jnp.float32)); po = ons.init(jax.random.fold_in(key, 1), jnp.zeros((1, 61 + STATE_DIM), jnp.float32))
  opt_m, opt_o = optax.adam(FIT['lr']), optax.adam(FIT['lr']); om, oo = opt_m.init(pm), opt_o.init(po)
  pm0 = jax.tree_util.tree_map(np.asarray, pm); po0 = jax.tree_util.tree_map(np.asarray, po)

  BD = int(FIT['batch_diag'])                                   # the first BD rows of every motion batch are diagonal (static slice)

  def mloss(p, x, d):
    pred = mot.apply(p, x); se = jnp.mean((pred - d) ** 2, axis=1)
    return jnp.mean(se[:BD]) + jnp.mean(se[BD:]), (jnp.mean(se[:BD]), jnp.mean(se[BD:]))

  def oloss(p, x, d, y, w):
    logit = ons.apply(p, jnp.concatenate([x, d], axis=1))
    return jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, y))

  @jax.jit
  def step_m(p, o, x, d):
    (l, parts), g = jax.value_and_grad(mloss, has_aux=True)(p, x, d)
    up, o = opt_m.update(g, o, p); return optax.apply_updates(p, up), o, l, parts

  @jax.jit
  def step_o(p, o, pm_, x, d_true, y, w):
    d_pred = jax.lax.stop_gradient(mot.apply(pm_, x))
    l, g = jax.value_and_grad(oloss)(p, x, d_pred, y, w)
    up, o = opt_o.update(g, o, p); return optax.apply_updates(p, up), o, l

  @jax.jit
  def val_fn(pm_, po_, x, d, diag, y, w):
    pred = mot.apply(pm_, x); se = jnp.mean((pred - d) ** 2, axis=1)
    md = jnp.sum(se * diag) / jnp.maximum(jnp.sum(diag), 1); mo = jnp.sum(se * (1 - diag)) / jnp.maximum(jnp.sum(1 - diag), 1)
    logit = ons.apply(po_, jnp.concatenate([x, jax.lax.stop_gradient(pred)], axis=1))
    bce = jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, y))
    return md, mo, bce, jax.nn.sigmoid(logit)
  wpos, wneg = np.float32(prevalence / 0.5), np.float32((1 - prevalence) / 0.5)
  hist, best, best_state, best_step, t0 = [], float('inf'), None, 0, time.time()
  bd, bo, bl = FIT['batch_diag'], FIT['batch_off'], FIT['onset_batch_per_class']
  for it in range(1, steps + 1):
    # motion batch: diagonal rows = half branch-diagonal, half log; off-diagonal = branch off rows
    i_d = rng.choice(diag_b, size=bd // 2); i_l = rng.choice(n_log, size=bd - bd // 2); i_o = rng.choice(off_b, size=bo)
    s1, ab1, aq1, s21, _ = R.gather_branch(i_d); s2_, ab2, aq2, s22, _ = R.gather_log(sp['train_l'][i_l]); s3, ab3, aq3, s23, _ = R.gather_branch(i_o)
    s = np.concatenate([s1, s2_, s3]); ab = np.concatenate([ab1, ab2, ab3]); aq = np.concatenate([aq1, aq2, aq3]); s2 = np.concatenate([s21, s22, s23])
    x = features(s, ab, aq, norm); d = ((s2 - s) - norm['d_mean']) / norm['d_std']
    pm, om, lm, parts = step_m(pm, om, jnp.asarray(x), jnp.asarray(d, jnp.float32))
    # onset batch
    ip = rng.choice(pos_b, size=bl); ineg = rng.choice(neg_b, size=bl)
    sp_, abp, aqp, s2p, _ = R.gather_branch(ip); sn_, abn, aqn, s2n, _ = R.gather_branch(ineg)
    xo = features(np.concatenate([sp_, sn_]), np.concatenate([abp, abn]), np.concatenate([aqp, aqn]), norm)
    yo = np.concatenate([np.ones(bl), np.zeros(bl)]).astype(np.float32); wo = np.concatenate([np.full(bl, wpos), np.full(bl, wneg)]).astype(np.float32)
    po, oo, lo = step_o(po, oo, pm, jnp.asarray(xo), None, jnp.asarray(yo), jnp.asarray(wo))
    if it % FIT['val_every'] == 0 or it == steps:
      md_, mo_, bce, prob = (np.asarray(v) for v in val_fn(pm, po, jnp.asarray(VX), jnp.asarray(VD, jnp.float32), jnp.asarray(Vdiag.astype(np.float32)), jnp.asarray(Vo.astype(np.float32)), jnp.asarray(Vw)))
      auc = auroc(prob, Vo)
      score = float(md_ + mo_ + FIT['onset_weight'] * bce)
      hist.append({'step': it, 'train_motion': float(lm), 'train_mse_diag': float(parts[0]), 'train_mse_off': float(parts[1]), 'train_onset': float(lo), 'val_mse_diag': float(md_), 'val_mse_off': float(mo_), 'val_onset_bce': float(bce), 'val_auroc': auc, 'score': score})
      if score < best:
        best, best_step = score, it
        best_state = {'motion_params': jax.tree_util.tree_map(np.asarray, pm), 'onset_params': jax.tree_util.tree_map(np.asarray, po)}
      print(f'[fold {f} step {it:>6}] train motion {float(lm):.4f} onset {float(lo):.4f} | val mse diag {float(md_):.4f} off {float(mo_):.4f} onset bce {float(bce):.4f} auroc {auc:.3f} score {score:.4f} best {best:.4f}@{best_step} {it / (time.time() - t0):.1f} it/s', flush=True)
  import optax as _optax
  md = {**best_state, 'norm': norm, 'fold': f, 'best_step': best_step, 'best_score': best, 'steps': steps, 'fit': FIT, 'history': hist, 'prevalence_train': prevalence, 'es_prevalence': es_prev,
        'es_episodes': sp['es_episodes'], 'n_train_branch': int(len(sp['train_b'])), 'n_train_log': int(n_log), 'wall_seconds': time.time() - t0,
        'param_delta': {'motion': float(_optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, best_state['motion_params'], pm0))), 'onset': float(_optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, best_state['onset_params'], po0)))},
        'supervision_sha256': MP.sha256(EC / 'supervision.npz')}
  with out.open('wb') as fh:
    pickle.dump(md, fh)
  MP.write_json(out.with_suffix('.json'), {k: v for k, v in md.items() if k not in ('motion_params', 'onset_params', 'norm')})
  print(f'saved {out}: best score {best:.4f} at {best_step} ({time.time() - t0:.0f} s)', flush=True)


def auroc(score, pos):
  from scipy.stats import rankdata
  pos = np.asarray(pos, bool); r = rankdata(score); n1, n0 = int(pos.sum()), int((~pos).sum())
  return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


# ------------------------------------------------------------------- check
def mode_check(args):
  R = Rows()
  res = {'per_fold': {}, 'gates': GATES}
  pooled = {'xy_err': [], 'delta_se': [], 'xy10': [], 'xy50': [], 'prob': [], 'onset': [], 'cls': [], 'in_band': [], 'diag': []}
  for f in range(N_FOLDS):
    md = load_model(f); P = Predictor(md); sp = R.split(f)
    ho = sp['ho_b']; rng = np.random.default_rng(FIT['val_seed'] + 10 + f)
    idx = rng.choice(ho, size=min(300_000, len(ho)), replace=False)
    s, ab, aq, s2, o = R.gather_branch(idx)
    pred, prob = P.step(s, ab, aq)
    xy_err = np.linalg.norm(pred[:, :2] - s2[:, :2], axis=1); dse = np.mean((((pred - s) - md['norm']['d_mean']) / md['norm']['d_std'] - ((s2 - s) - md['norm']['d_mean']) / md['norm']['d_std']) ** 2, axis=1)
    # all held-out onsets (rare) for the onset metrics
    ho_pos = ho[R.onset[ho]]
    s_p, ab_p, aq_p, _, _ = R.gather_branch(ho_pos); _, prob_p = P.step(s_p, ab_p, aq_p)
    prob_all = np.concatenate([prob[~o], prob_p]); onset_all = np.concatenate([np.zeros((~o).sum(), bool), np.ones(len(ho_pos), bool)])
    cls_all = np.concatenate([R.cls[idx][~o], R.cls[ho_pos]]); x_all = np.concatenate([s[~o, 0], s_p[:, 0]]); y_all = np.concatenate([s[~o, 1], s_p[:, 1]])
    band = ((x_all >= ZONE_X[1][0]) & (x_all <= ZONE_X[1][1]) | (x_all >= ZONE_X[2][0]) & (x_all <= ZONE_X[2][1])) & (np.abs(y_all) < 2)
    # open-loop rollouts from held-out branch roots with the recorded (a_b, a_q)
    roots = rng.choice(np.flatnonzero(R.fold_anchor == f), size=min(2000, int((R.fold_anchor == f).sum())), replace=False)
    roots = roots[R.blen[roots] >= ROLL_H + 1]
    off = R.boff[roots]; cur = R.bobs[off].copy(); e10 = None; e50 = None
    for j in range(ROLL_H):
      cur, _ = P.step(cur, R.adv[off + j], R.bact[off + j])
      if j + 1 == 10:
        e10 = np.linalg.norm(cur[:, :2] - R.bobs[off + 10, :2], axis=1)
    e50 = np.linalg.norm(cur[:, :2] - R.bobs[off + ROLL_H, :2], axis=1)
    prev_ho = len(ho_pos) / len(ho)
    fr = {'n_held_out_rows': int(len(ho)), 'n_eval_rows': int(len(idx)), 'xy_err_median': float(np.median(xy_err)), 'xy_err_p90': float(np.percentile(xy_err, 90)), 'xy_err_p99': float(np.percentile(xy_err, 99)),
          'delta_rmse': float(np.sqrt(dse.mean())), 'xy10_median': float(np.median(e10)), 'xy50_median': float(np.median(e50)), 'xy50_p90': float(np.percentile(e50, 90)), 'n_rollouts': int(len(roots)),
          'onset_auroc': auroc(prob_all, onset_all), 'mean_prob_on_onsets': float(prob_p.mean()), 'mean_prob_on_non_onsets': float(prob[~o].mean()), 'held_out_prevalence': prev_ho, 'n_held_out_onsets': int(len(ho_pos)),
          'mean_prob_all_rows': float(prob.mean()), 'best_step': md['best_step'], 'param_delta': md['param_delta'],
          'xy_err_median_diag_rows': float(np.median(xy_err[R.br_diag[np.searchsorted(R.br, idx)]])), 'xy_err_median_off_rows': float(np.median(xy_err[~R.br_diag[np.searchsorted(R.br, idx)]]))}
    hold_band = band & (cls_all == 'off_hold'); drive_band = band & (cls_all != 'off_hold')
    fr['band_mean_prob_off_hold'] = float(prob_all[hold_band].mean()) if hold_band.any() else None; fr['band_mean_prob_drive'] = float(prob_all[drive_band].mean()) if drive_band.any() else None
    res['per_fold'][str(f)] = fr
    pooled['xy_err'].append(xy_err); pooled['delta_se'].append(dse); pooled['xy10'].append(e10); pooled['xy50'].append(e50); pooled['prob'].append(prob_all); pooled['onset'].append(onset_all); pooled['cls'].append(cls_all); pooled['in_band'].append(band)
    print(f'fold {f}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fr.items() if k != "param_delta"})}', flush=True)
  xy = np.concatenate(pooled['xy_err']); dse = np.concatenate(pooled['delta_se']); e10 = np.concatenate(pooled['xy10']); e50 = np.concatenate(pooled['xy50'])
  prob = np.concatenate(pooled['prob']); on = np.concatenate(pooled['onset']); cls = np.concatenate(pooled['cls']); band = np.concatenate(pooled['in_band'])
  prev = float(np.mean([res['per_fold'][str(f)]['held_out_prevalence'] for f in range(N_FOLDS)]))
  mean_all = float(np.mean([res['per_fold'][str(f)]['mean_prob_all_rows'] for f in range(N_FOLDS)]))
  hb, db = band & (cls == 'off_hold'), band & (cls != 'off_hold')
  pooled_res = {'xy_err_median': float(np.median(xy)), 'xy_err_p99': float(np.percentile(xy, 99)), 'delta_rmse': float(np.sqrt(dse.mean())), 'xy10_median': float(np.median(e10)), 'xy50_median': float(np.median(e50)),
                'onset_auroc': auroc(prob, on), 'ratio_onsets_vs_non': float(prob[on].mean() / max(prob[~on].mean(), 1e-9)), 'calibration_mean_over_prevalence': mean_all / max(prev, 1e-9),
                'advice_ratio_in_band': float(prob[hb].mean() / max(prob[db].mean(), 1e-9)) if hb.any() and db.any() else None, 'n_band_off_hold': int(hb.sum()), 'n_band_drive': int(db.sum())}
  g = {'G1_xy_median': pooled_res['xy_err_median'] < GATES['G1_xy_median'], 'G1_xy_p99': pooled_res['xy_err_p99'] < GATES['G1_xy_p99'], 'G1_delta_rmse': pooled_res['delta_rmse'] < GATES['G1_delta_rmse'],
       'G2_xy50': pooled_res['xy50_median'] < GATES['G2_xy50_median'], 'G2_xy10': pooled_res['xy10_median'] < GATES['G2_xy10_median'], 'G3_auroc': pooled_res['onset_auroc'] >= GATES['G3_auroc'],
       'G4_ratio': pooled_res['ratio_onsets_vs_non'] >= GATES['G4_ratio'], 'G5_calibration': GATES['G5_calibration'][0] <= pooled_res['calibration_mean_over_prevalence'] <= GATES['G5_calibration'][1],
       'G6_advice': (pooled_res['advice_ratio_in_band'] is not None and pooled_res['advice_ratio_in_band'] >= GATES['G6_advice_ratio']),
       'param_delta': all(v['param_delta']['motion'] > GATES['param_delta_min'] and v['param_delta']['onset'] > GATES['param_delta_min'] for v in res['per_fold'].values())}
  res['pooled'] = pooled_res; res['gate_results'] = g; res['all_passed'] = bool(all(g.values()))
  MP.write_json(OUT / 'check.json', res)
  L = ['# One-step ETT preflight (held-out fold rows, cross-fitted)', '', f'`check.json`; gates from `manifest.json`.  **{"ALL GATES PASSED" if res["all_passed"] else "GATE(S) FAILED -- the line stops here"}**', '',
       '| gate | threshold | pooled value | pass |', '|---|---|---|---|']
  for k, v in g.items():
    thr = GATES.get(k, GATES.get(k.split('_')[0] + '_' + k.split('_')[1] + '_median', ''))
    L.append(f'| {k} | {thr if not isinstance(thr, float) else thr} | see pooled | {"yes" if v else "NO"} |')
  L += ['', '## Pooled', '', '```', json.dumps(pooled_res, indent=1), '```', '', '## Per fold', '', '| fold | held-out rows | xy err median / p90 / p99 | delta RMSE | xy err diag / off rows | 10-step / 50-step xy (median) | onset AUROC | P(onset) on onsets / non | prevalence | best step |', '|---|---:|---|---:|---|---|---:|---|---:|---:|']
  for f, v in res['per_fold'].items():
    L.append(f'| {f} | {v["n_held_out_rows"]} | {v["xy_err_median"]:.4f} / {v["xy_err_p90"]:.4f} / {v["xy_err_p99"]:.4f} | {v["delta_rmse"]:.3f} | {v["xy_err_median_diag_rows"]:.4f} / {v["xy_err_median_off_rows"]:.4f} | {v["xy10_median"]:.3f} / {v["xy50_median"]:.3f} | {v["onset_auroc"]:.3f} | {v["mean_prob_on_onsets"]:.4f} / {v["mean_prob_on_non_onsets"]:.5f} | {v["held_out_prevalence"]:.5f} | {v["best_step"]} |')
  (OUT / 'CHECK.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'fit', 'check'))
  ap.add_argument('--fold', type=int, default=0)
  ap.add_argument('--steps', type=int, default=None, help='fit: override (smoke only)')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  {'seal': mode_seal, 'fit': mode_fit, 'check': mode_check}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
