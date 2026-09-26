#!/usr/bin/env python
"""AntMaze V6 one-step ETT -- the ADVICE PROCESS: is the nominal's advice coherent in time and grounded on intervened states?

The ETT line after a408cef (user): the learned nominal loses the danger
(model death 3.6 % vs 29.4 % real) and that is more important than the
counter offset.  Before any new nominal is designed, measure on the existing
data what the advice process has to reproduce and where the memoryless
nominal a_b ~ MDN(s) (diag_v6_ett_rollout.mode_nominal, fitted on the d05
log) fails.  No rollouts, no training; the v3 models are used as fixed
predictors.

A. The REAL advice along the branch rows (the teacher's advice walked along
   every pilot branch under its redrawn timetable; supervision.npz):
   hold share, hold-run lengths, persistence P(hold_{j+1} | hold_j), the
   share of holds inside the hazard bands, by region.
B. The memoryless nominal along the SAME real states (teacher-forced: one
   i.i.d. sample per row): the same statistics -- temporal coherence; and
   its exact per-row hold probability p_h(s) vs the real hold flag (AUROC,
   by class, by steps since the query, in / out of the logged support
   measured by the MDN's NLL of the real advice) -- support on intervened
   states.
C. The v3 onset head's dependence on the hold history (fixed models, held-out
   rows): p_onset on real in-band hold rows with the real counter vs the
   counter forced to k = 0, 1, 2, ...; the realised onset rate by counter
   bin; and the onset rate implied by the nominal's counter distribution.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_ett_rollout as RO  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from fit_v6_ett_one_step_v2 import HOLD_TOL, in_band, run_length  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

EC = MP.OUT / 'ett_context'
OUT = MP.OUT / 'ett_advice'
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W
N_FOLDS = 3
SAMPLE_SEED = 204_000_000            # the nominal's i.i.d. draws along the real states
K_FORCE = (0, 1, 2, 3, 5, 10, 20, 50, 100)
KH_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 10 ** 9]
MAX_ROWS_C = 150_000                 # per fold, subsampled without replacement (seeded)


# ------------------------------------------------------------------ helpers
def raw_counter(flag, start):
  """Consecutive flagged rows up to and INCLUDING the current one (1 at the first flagged row), resetting at path starts."""
  idx = np.arange(len(flag))
  b = np.where(~flag, idx, np.where(start, idx - 1, -1))
  last = np.maximum.accumulate(b)
  return np.where(flag, idx - last, 0).astype(np.int64)


def run_lengths(flag, start):
  raw = raw_counter(flag, start)
  nxt_break = np.ones(len(flag), bool)
  nxt_break[:-1] = (~flag[1:]) | start[1:]
  return raw[flag & nxt_break]


def persistence(flag, start):
  same_path = np.zeros(len(flag), bool); same_path[:-1] = ~start[1:]
  a = flag & same_path
  nxt = np.zeros(len(flag), bool); nxt[:-1] = flag[1:]
  p_hh = float((a & nxt).sum() / max(a.sum(), 1))
  b = (~flag) & same_path
  p_nh = float((b & nxt).sum() / max(b.sum(), 1))
  return {'p_hold_given_hold': p_hh, 'p_hold_given_nohold': p_nh}


def subset_starts(rows, start):
  """Path starts within a row subset: a branch start, or a row whose predecessor is not in the subset."""
  st = start[rows].copy(); st[0] = True
  st |= np.concatenate([[True], rows[1:] != rows[:-1] + 1])
  return st


def describe_runs(r):
  r = np.asarray(r, float)
  if len(r) == 0:
    return {'n_runs': 0}
  return {'n_runs': int(len(r)), 'mean': float(r.mean()), 'median': float(np.median(r)), 'p90': float(np.percentile(r, 90)), 'p99': float(np.percentile(r, 99)),
          'share_len1': float((r == 1).mean()), 'share_len_ge5': float((r >= 5).mean()), 'share_len_ge20': float((r >= 20).mean())}


def auroc(score, pos):
  from scipy.stats import rankdata
  pos = np.asarray(pos, bool); r = rankdata(score); n1, n0 = int(pos.sum()), int((~pos).sum())
  return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def advice_stats(hold, start, xy, label):
  band = in_band(xy)
  reg = region_of(xy)
  out = {'label': label, 'rows': int(len(hold)), 'hold_share': float(hold.mean()), 'hold_share_in_band': float(hold[band].mean()) if band.any() else float('nan'),
         'hold_share_out_of_band': float(hold[~band].mean()), 'share_of_holds_in_band': float(band[hold].mean()) if hold.any() else float('nan'),
         'runs': describe_runs(run_lengths(hold, start)), 'runs_of_hold_in_band': describe_runs(run_lengths(hold & band, start)), **persistence(hold, start),
         'hold_share_by_region': {REGIONS[i]: float(hold[reg == i].mean()) for i in range(len(REGIONS)) if (reg == i).sum() >= 100}}
  return out


class MDNBatch:
  """Batched evaluation of the nominal MDN: exact hold probability, NLL of a given advice, i.i.d. samples."""

  def __init__(self, md):
    import jax
    import jax.numpy as jnp
    net = RO.build_mdn(); p = jax.tree_util.tree_map(jnp.asarray, md['params'])
    self.mean, self.std = md['mean'], md['std']
    self._f = jax.jit(lambda x: net.apply(p, x))

  def params(self, obs31):
    logits, mu, ls = [], [], []
    for i in range(0, len(obs31), 16384):
      x = ((obs31[i:i + 16384] - self.mean) / self.std).astype(np.float32)
      a, b, c = (np.asarray(v, np.float64) for v in self._f(x)); logits.append(a); mu.append(b); ls.append(c)
    logits, mu, ls = np.concatenate(logits), np.concatenate(mu), np.concatenate(ls)
    pi = np.exp(logits - logits.max(axis=1, keepdims=True)); pi /= pi.sum(axis=1, keepdims=True)
    return pi, mu, np.exp(ls)

  @staticmethod
  def hold_prob(pi, mu, sd):
    from scipy.special import erf
    cdf = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))  # noqa: E731
    box = cdf((HOLD_TOL - mu) / sd) - cdf((-HOLD_TOL - mu) / sd)            # per component, per dimension
    return (pi * np.prod(box, axis=2)).sum(axis=1)

  @staticmethod
  def nll(pi, mu, sd, a):
    lp = -0.5 * (((a[:, None, :] - mu) / sd) ** 2 + 2 * np.log(sd) + np.log(2 * np.pi)).sum(axis=2) + np.log(np.maximum(pi, 1e-300))
    m = lp.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))

  @staticmethod
  def sample(pi, mu, sd, rng):
    c = (rng.random(len(pi))[:, None] > np.cumsum(pi, axis=1)).sum(axis=1)
    c = np.minimum(c, pi.shape[1] - 1)
    a = mu[np.arange(len(pi)), c] + sd[np.arange(len(pi)), c] * rng.standard_normal((len(pi), mu.shape[2]))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ driver
def run(args):
  OUT.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    bobs, bact, boff, blen = d['obs_rows'], d['act_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64)
  S = np.load(EC / 'supervision.npz', allow_pickle=False)
  adv, valid, onset, cls = S['advice_rows'], S['valid_rows'].astype(bool), S['onset_rows'].astype(bool), S['class_rows'].astype(str)
  fold_row, anchor_of_row, j_of_row, fold_anchor = S['fold_rows'].astype(np.int64), S['anchor_of_row'].astype(np.int64), S['j_of_row'].astype(np.int64), S['fold_anchor'].astype(np.int64)
  R = len(adv); assert R == len(bobs)
  start = np.zeros(R, bool); start[boff] = True
  hold_real = np.abs(adv).max(axis=1) < HOLD_TOL
  xy = bobs[:, :2].astype(np.float64)
  band = in_band(xy)
  kh_real = run_length(hold_real, start); kb_real = run_length(band, start)
  man = {'diagnostic': 'AntMaze V6 one-step ETT: the advice process -- temporal coherence and support of the memoryless nominal vs the real advice; the onset head\'s dependence on the hold history (fixed v3 models)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'supervision_sha256': MP.sha256(EC / 'supervision.npz'), 'nominal_sha256': MP.sha256((MP.OUT / 'ett_rollout') / 'nominal_mdn.pkl'),
         'v3_models': {f'fold{f}': MP.sha256(MP.OUT / 'ett_one_step_v3' / f'model_fold{f}.json') for f in range(N_FOLDS)},
         'hold_tol': HOLD_TOL, 'sample_anchors': f'the rollout diagnostic\'s held-out anchors ({RO.N_PER_FOLD} per fold, seed {RO.SAMPLE_SEED}); the nominal drawn once per row, seed {SAMPLE_SEED}',
         'C': {'rows': 'held-out valid branch rows with a real hold inside a hazard band; up to {MAX_ROWS_C} per fold', 'forced_counters': list(K_FORCE), 'kh_bins': KH_BINS[:-1]}, 'status': 'oracle-supervised engineering stage; nothing downstream generated'}
  MP.write_json(OUT / 'manifest.json', man)
  res = {'sealed_at': man['sealed_at']}
  # ---- A: the real advice
  rv = np.flatnonzero(valid)                                   # the last row of every branch has no advice (dummy zero): valid rows only
  res['A_real_all_branches'] = advice_stats(hold_real[rv], subset_starts(rv, start), xy[rv], 'real advice, all 53,747 branches, valid rows (teacher-forced along each branch)')
  res['A_real_all_branches']['onset_rows'] = int(onset.sum()); res['A_real_all_branches']['onset_on_hold_rows'] = int((onset & hold_real).sum())
  res['A_real_all_branches']['onset_rate_per_hold_row'] = float(onset[hold_real].mean())
  res['A_real_all_branches']['onset_rate_per_hold_row_in_band'] = float(onset[hold_real & band].mean())
  res['A_real_all_branches']['class_share'] = {c: float((cls == c).mean()) for c in np.unique(cls)}
  print(f'A done {time.time() - t0:.0f} s', flush=True)
  # ---- B: the nominal along the sampled branches' real states
  md = pickle.load(((MP.OUT / 'ett_rollout') / 'nominal_mdn.pkl').open('rb'))
  mdn = MDNBatch(md)
  ks = np.concatenate([RO.sample_anchors(f) for f in range(N_FOLDS)])
  rows = np.concatenate([np.arange(boff[k], boff[k] + blen[k]) for k in ks])
  rows = rows[valid[rows]]
  obs31 = bobs[rows, :OBS_W].astype(np.float32)
  pi, mu, sd = mdn.params(obs31)
  p_h = mdn.hold_prob(pi, mu, sd)
  nll_real = mdn.nll(pi, mu, sd, adv[rows].astype(np.float64))
  rng = np.random.default_rng(SAMPLE_SEED)
  a_nom = mdn.sample(pi, mu, sd, rng)
  hold_nom = np.abs(a_nom).max(axis=1) < HOLD_TOL
  st_s = subset_starts(rows, start)
  hr = hold_real[rows]; xs = xy[rows]; bd = band[rows]; js = j_of_row[rows]; cs = cls[rows]
  res['B_sample'] = {'anchors': int(len(ks)), 'rows': int(len(rows)),
                     'real': advice_stats(hr, st_s, xs, 'real advice, sampled branches'),
                     'nominal_iid': advice_stats(hold_nom, st_s, xs, 'memoryless nominal, one i.i.d. draw per row along the real states'),
                     'nominal_hold_prob_mean': float(p_h.mean()), 'nominal_hold_prob_mean_in_band': float(p_h[bd].mean()),
                     'nominal_hold_prob_mean_on_real_hold_rows': float(p_h[hr].mean()), 'nominal_hold_prob_mean_on_real_nohold_rows': float(p_h[~hr].mean()),
                     'auroc_hold_prob_vs_real_hold': {'all': auroc(p_h, hr), 'in_band': auroc(p_h[bd], hr[bd]), 'out_of_band': auroc(p_h[~bd], hr[~bd]),
                                                      'j_0': auroc(p_h[js == 0], hr[js == 0]), 'j_1_10': auroc(p_h[(js >= 1) & (js <= 10)], hr[(js >= 1) & (js <= 10)]),
                                                      'j_11_50': auroc(p_h[(js > 10) & (js <= 50)], hr[(js > 10) & (js <= 50)]), 'j_gt_50': auroc(p_h[js > 50], hr[js > 50])},
                     'nll_of_real_advice': {'all': float(nll_real.mean()), 'real_hold_rows': float(nll_real[hr].mean()), 'real_nohold_rows': float(nll_real[~hr].mean()),
                                            'by_class': {c: float(nll_real[cs == c].mean()) for c in np.unique(cs)},
                                            'j_0': float(nll_real[js == 0].mean()), 'j_1_10': float(nll_real[(js >= 1) & (js <= 10)].mean()), 'j_11_50': float(nll_real[(js > 10) & (js <= 50)].mean()), 'j_gt_50': float(nll_real[js > 50].mean())},
                     'kh_at_hold_rows': {'real': np.bincount(np.minimum(raw_counter(hr, st_s)[hr], 200)).tolist(), 'nominal': np.bincount(np.minimum(raw_counter(hold_nom, st_s)[hold_nom], 200)).tolist()}}
  # the logged rows for the NLL reference (held-out val episodes of the MDN fit)
  obs, act, lengths, _ = MP.load_dataset()
  val_eps = np.array(md['val_episodes'], np.int64)
  ee = np.repeat(val_eps, lengths[val_eps] - 1); tt = np.concatenate([np.arange(lengths[e] - 1) for e in val_eps])
  pi_l, mu_l, sd_l = mdn.params(obs[ee, tt, :OBS_W].astype(np.float32))
  hl = np.abs(act[ee, tt]).max(axis=1) == 0.0
  res['B_sample']['logged_val_reference'] = {'rows': int(len(ee)), 'nll_of_logged_action': float(mdn.nll(pi_l, mu_l, sd_l, act[ee, tt].astype(np.float64)).mean()),
                                             'hold_share': float(hl.mean()), 'auroc_hold_prob_vs_logged_hold': auroc(mdn.hold_prob(pi_l, mu_l, sd_l), hl)}
  del obs, act
  print(f'B done {time.time() - t0:.0f} s', flush=True)
  # ---- C: the onset head vs the hold history, held-out rows, fixed v3 models
  F2.OUT = F2.OUT_V3
  C = {'per_fold': {}}
  kh_nom_pool = run_length(hold_nom, st_s)[hold_nom]            # the counter the model would see at the nominal's hold rows (training convention)
  for f in range(N_FOLDS):
    P = F2.Predictor2(F2.load_model(f))
    r = np.flatnonzero(valid & (fold_row == f) & hold_real & band)
    rng_c = np.random.default_rng(SAMPLE_SEED + 1 + f)
    if len(r) > MAX_ROWS_C:
      r = np.sort(rng_c.choice(r, size=MAX_ROWS_C, replace=False))
    s, ab, aq, kh, kb, on = bobs[r, :STATE_DIM], adv[r], bact[r], kh_real[r], kb_real[r], onset[r]
    p_real = np.concatenate([P.step(s[i:i + 65536], ab[i:i + 65536], aq[i:i + 65536], kh[i:i + 65536], kb[i:i + 65536])[1] for i in range(0, len(r), 65536)])
    forced = {}
    for k in K_FORCE:
      kk = np.full(len(r), k)
      forced[str(k)] = float(np.concatenate([P.step(s[i:i + 65536], ab[i:i + 65536], aq[i:i + 65536], kk[i:i + 65536], kb[i:i + 65536])[1] for i in range(0, len(r), 65536)]).mean())
    kh_nom = rng_c.choice(kh_nom_pool, size=len(r))                # a counter drawn from the nominal's own hold-row counter distribution
    p_nom = np.concatenate([P.step(s[i:i + 65536], ab[i:i + 65536], aq[i:i + 65536], kh_nom[i:i + 65536], kb[i:i + 65536])[1] for i in range(0, len(r), 65536)])
    bins = np.digitize(kh, KH_BINS[1:-1], right=False)
    by_bin = {}
    for b in range(len(KH_BINS) - 1):
      m = bins == b
      if m.sum() >= 50:
        by_bin[f'kh_{KH_BINS[b]}_{KH_BINS[b + 1] - 1 if b < len(KH_BINS) - 2 else "inf"}'] = {'rows': int(m.sum()), 'onset_rate': float(on[m].mean()), 'p_onset_mean': float(p_real[m].mean())}
    C['per_fold'][f'fold{f}'] = {'rows': int(len(r)), 'onset_rate': float(on.mean()), 'p_onset_real_counter': float(p_real.mean()), 'p_onset_forced_counter': forced,
                                 'p_onset_nominal_counter_draw': float(p_nom.mean()), 'auroc_p_real_vs_onset': auroc(p_real, on), 'by_kh_bin': by_bin,
                                 'kh_real_mean': float(kh.mean()), 'kh_real_median': float(np.median(kh))}
    print(f'C fold {f} done {time.time() - t0:.0f} s', flush=True)
  res['C_onset_head'] = C
  res['wall_seconds'] = time.time() - t0
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res), encoding='utf-8')
  print('done', flush=True)


def f3(x):
  return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def stat_row(S):
  r = S['runs']
  return f"| {S['label']} | {S['rows']} | {f3(S['hold_share'])} | {f3(S['hold_share_in_band'])} | {f3(S['hold_share_out_of_band'])} | {f3(S['share_of_holds_in_band'])} | {f3(S['p_hold_given_hold'])} | {f3(S['p_hold_given_nohold'])} | {r.get('n_runs', 0)} | {f3(r.get('mean'))} | {f3(r.get('median'))} | {f3(r.get('p90'))} | {f3(r.get('share_len1'))} | {f3(r.get('share_len_ge5'))} | {f3(r.get('share_len_ge20'))} |"


def write_md(res):
  A = res['A_real_all_branches']; B = res['B_sample']
  L = ['# The advice process: temporal coherence and support of the memoryless nominal; the onset head vs the hold history', '',
       f"Sealed {res['sealed_at']}; `manifest.json`, `report.json`.  No rollouts, no training.", '',
       '## A / B: hold structure of the real advice vs the memoryless nominal drawn i.i.d. along the same real states', '',
       '| source | rows | hold share | in band | out of band | share of holds in band | P(hold|hold) | P(hold|no hold) | runs | mean | median | p90 | len 1 | len >= 5 | len >= 20 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|', stat_row(A), stat_row(B['real']), stat_row(B['nominal_iid']), '',
       f"Real advice, all branches: onset rows {A['onset_rows']}, of which on hold rows {A['onset_on_hold_rows']}; onset rate per hold row {A['onset_rate_per_hold_row']:.4f} (in band {A['onset_rate_per_hold_row_in_band']:.4f}).  Class share {json.dumps({k: round(v, 4) for k, v in A['class_share'].items()})}.", '',
       f"Hold share by region (real, all branches): {json.dumps({k: round(v, 3) for k, v in A['hold_share_by_region'].items()})}", '',
       f"Hold share by region (nominal i.i.d., sample): {json.dumps({k: round(v, 3) for k, v in B['nominal_iid']['hold_share_by_region'].items()})}", '',
       '## B: support of the nominal on the intervened states', '',
       f"Nominal exact hold probability: mean {B['nominal_hold_prob_mean']:.3f} (in band {B['nominal_hold_prob_mean_in_band']:.3f}); on real hold rows {B['nominal_hold_prob_mean_on_real_hold_rows']:.3f} vs real no-hold rows {B['nominal_hold_prob_mean_on_real_nohold_rows']:.3f}.", '',
       '| AUROC of p_h(s) vs the real hold | ' + ' | '.join(B['auroc_hold_prob_vs_real_hold']) + ' |', '|---|' + '---:|' * len(B['auroc_hold_prob_vs_real_hold']),
       '| | ' + ' | '.join(f3(v) for v in B['auroc_hold_prob_vs_real_hold'].values()) + ' |', '',
       f"Logged reference (the MDN's {B['logged_val_reference']['rows']} validation rows): NLL of the logged action {B['logged_val_reference']['nll_of_logged_action']:.2f}; hold share {B['logged_val_reference']['hold_share']:.3f}; AUROC p_h vs logged hold {f3(B['logged_val_reference']['auroc_hold_prob_vs_logged_hold'])}.", '',
       '| NLL of the REAL advice under the nominal | ' + ' | '.join(k for k in B['nll_of_real_advice'] if k != 'by_class') + ' | ' + ' | '.join(f'class {c}' for c in B['nll_of_real_advice']['by_class']) + ' |',
       '|---|' + '---:|' * (len(B['nll_of_real_advice']) - 1 + len(B['nll_of_real_advice']['by_class'])),
       '| | ' + ' | '.join(f"{v:.2f}" for k, v in B['nll_of_real_advice'].items() if k != 'by_class') + ' | ' + ' | '.join(f'{v:.2f}' for v in B['nll_of_real_advice']['by_class'].values()) + ' |', '',
       'Counter at hold rows (run position, 1 = first hold of a run), share of hold rows: real vs nominal', '']
  kr, kn = np.array(B['kh_at_hold_rows']['real'], float), np.array(B['kh_at_hold_rows']['nominal'], float)
  kr, kn = kr / max(kr.sum(), 1), kn / max(kn.sum(), 1)
  cum = lambda v, a, b: float(v[a:b + 1].sum()) if len(v) > a else 0.0  # noqa: E731
  L += ['| run position | 1 | 2 | 3-4 | 5-9 | 10-19 | 20-49 | >= 50 |', '|---|---:|---:|---:|---:|---:|---:|---:|',
        '| real | ' + ' | '.join(f3(x) for x in (cum(kr, 1, 1), cum(kr, 2, 2), cum(kr, 3, 4), cum(kr, 5, 9), cum(kr, 10, 19), cum(kr, 20, 49), cum(kr, 50, 10 ** 6))) + ' |',
        '| nominal | ' + ' | '.join(f3(x) for x in (cum(kn, 1, 1), cum(kn, 2, 2), cum(kn, 3, 4), cum(kn, 5, 9), cum(kn, 10, 19), cum(kn, 20, 49), cum(kn, 50, 10 ** 6))) + ' |', '',
        '## C: the v3 onset head vs the hold counter (held-out real in-band hold rows; fixed models)', '',
        '| fold | rows | realised onset rate | p_onset (real counter) | AUROC | ' + ' | '.join(f'k = {k}' for k in K_FORCE) + ' | p_onset (nominal counter draw) | kh real mean / median |',
        '|---|---:|---:|---:|---:|' + '---:|' * len(K_FORCE) + '---:|---|']
  for f, c in res['C_onset_head']['per_fold'].items():
    L.append(f"| {f} | {c['rows']} | {c['onset_rate']:.4f} | {c['p_onset_real_counter']:.4f} | {f3(c['auroc_p_real_vs_onset'])} | " + ' | '.join(f"{c['p_onset_forced_counter'][str(k)]:.4f}" for k in K_FORCE) + f" | {c['p_onset_nominal_counter_draw']:.4f} | {c['kh_real_mean']:.1f} / {c['kh_real_median']:.0f} |")
  keys = []
  for c in res['C_onset_head']['per_fold'].values():
    keys += [k for k in c['by_kh_bin'] if k not in keys]
  L += ['', 'Realised onset rate / mean p_onset (rows) by real counter bin:', '', '| fold | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
  for f, c in res['C_onset_head']['per_fold'].items():
    L.append(f'| {f} | ' + ' | '.join((f"{c['by_kh_bin'][k]['onset_rate']:.4f} / {c['by_kh_bin'][k]['p_onset_mean']:.4f} ({c['by_kh_bin'][k]['rows']})" if k in c['by_kh_bin'] else '-') for k in keys) + ' |')
  L.append('')
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='mode', required=True); sub.add_parser('run')
  run(ap.parse_args(argv))


if __name__ == '__main__':
  main()
