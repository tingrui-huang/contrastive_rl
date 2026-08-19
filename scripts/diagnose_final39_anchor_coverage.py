"""Post-hoc diagnostic: are final39 generator failures explained by distance
to the observed failure-training anchors?

Purely diagnostic. Nothing is retrained, no Flow candidate is resampled, the
3.17 radius / lambda / K are untouched, no classifier is fitted, and no new
data is collected. It reads the ALREADY-FROZEN final39 outputs and the 196
factual failure-training anchors.

Hypothesis under test: final39 anchors that the generator failed to cover
(C_i = 0) lie systematically farther from the 196 (s_fail, a_fail) training
anchors than covered ones -- i.e. a failure-support diversity gap.

The decisive control is section 9: if uncovered anchors are equally far from
the ORDINARY CLEAN training pool, the effect is generic distribution shift
rather than a failure-support-specific gap.

Normalizations are all pre-existing and failure-independent:
  * state: the frozen V0/V0.5 state normalization used by the Flow;
  * action: the per-dim mean/std recorded in
    artifacts/flow_v05_clean_action/action_stats.json, computed from the
    CLEAN TRAIN split only, before final39 was opened. No state-vs-action
    weight was chosen after seeing outcomes; d_s, d_a and d_sa are reported
    separately.

Usage:
  python scripts/diagnose_final39_anchor_coverage.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from train_flow_v2 import build_fail_pool  # noqa: E402
from train_flow_v1 import stack_pairs      # noqa: E402

FINAL39 = 'artifacts/flow_v2_final39'
V0_DIR = 'artifacts/flow_v0_clean'
ACT_STATS = 'artifacts/flow_v05_clean_action/action_stats.json'
BAD_DIR = 'artifacts/bad_demo_fixed'
BAD_NAME = 'bad_demo_blind_p30_h800_settle80'
OUT = 'artifacts/flow_v2_final39_anchor_coverage_diagnostic'
R_FATAL = 3.17           # frozen; used only to read the existing labels
KS = (3, 5, 10)
N_PERM, N_BOOT = 10_000, 10_000


def desc(a):
  a = np.asarray(a, float)
  return {'n': int(len(a)), 'median': float(np.median(a)),
          'mean': float(a.mean()),
          'p10': float(np.percentile(a, 10)),
          'p25': float(np.percentile(a, 25)),
          'p75': float(np.percentile(a, 75)),
          'p90': float(np.percentile(a, 90))}


def compare(x_cov, x_unc):
  u, p = mannwhitneyu(x_unc, x_cov, alternative='greater')
  n1, n2 = len(x_unc), len(x_cov)
  cliff = 2.0 * u / (n1 * n2) - 1.0      # +1 => uncovered strictly larger
  return {'median_covered': float(np.median(x_cov)),
          'median_uncovered': float(np.median(x_unc)),
          'median_diff_unc_minus_cov': float(np.median(x_unc)
                                             - np.median(x_cov)),
          'mannwhitney_u': float(u),
          'p_one_sided_uncovered_greater': float(p),
          'cliffs_delta': float(cliff)}


def auc_and_ci(score, label, n_boot, rng):
  """AUC for predicting label==1 from `score` (higher => more likely 1)."""
  def _auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
      return float('nan')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort')
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks over ties
      j = i
      while j + 1 < len(sv) and sv[j + 1] == sv[i]:
        j += 1
      if j > i:
        r[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
      i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))
  a = _auc(score, label)
  bs = []
  n = len(score)
  for _ in range(n_boot):
    idx = rng.integers(0, n, n)
    if len(np.unique(label[idx])) < 2:
      continue
    bs.append(_auc(score[idx], label[idx]))
  return {'auc': a,
          'ci95': [float(np.percentile(bs, 2.5)),
                   float(np.percentile(bs, 97.5))] if bs else None}


def corr_block(x, y, rng):
  sp = spearmanr(x, y)
  pe = pearsonr(x, y)
  # permutation p-value for Spearman
  obs = sp.statistic
  cnt = 0
  yy = np.array(y)
  for _ in range(N_PERM):
    if abs(spearmanr(x, rng.permutation(yy)).statistic) >= abs(obs):
      cnt += 1
  return {'spearman_rho': float(obs),
          'spearman_p_scipy': float(sp.pvalue),
          'spearman_p_permutation_two_sided': (cnt + 1) / (N_PERM + 1),
          'pearson_r': float(pe[0]), 'pearson_p': float(pe[1])}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  # ---- frozen normalizations ----------------------------------------------
  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  s_mean = np.asarray(nz_['state_mean'], np.float32)
  s_std = np.asarray(nz_['state_std'], np.float32)
  d_mean = np.asarray(nz_['delta_mean'], np.float32)
  d_std = np.asarray(nz_['delta_std'], np.float32)
  astat = json.load(open(ACT_STATS))
  a_mean = np.asarray(astat['per_dim_mean'], np.float32)
  a_std = np.asarray(astat['per_dim_std'], np.float32)

  def ns(x):
    return (x - s_mean) / s_std

  def na(x):
    return (x - a_mean) / a_std

  # ---- data ----------------------------------------------------------------
  sf, af, df_fail, _ = build_fail_pool(BAD_DIR, BAD_NAME)       # 196 anchors
  z = np.load(os.path.join(FINAL39, 'candidates_and_scores.npz'),
              allow_pickle=True)
  S = np.asarray(z['anchor'], np.float32)
  A = np.asarray(z['action'], np.float32)
  Sf_t = np.asarray(z['true_fatal'], np.float32)
  eps = np.asarray(z['episode_id'], np.int64)
  rows = list(csv.DictReader(open(os.path.join(FINAL39, 'per_case.csv'))))
  d_fatal = np.array([float(r['d_fatal']) for r in rows])
  d_safe = np.array([float(r['d_safe']) for r in rows])
  Ci = np.array([int(r['covered']) for r in rows])
  n = len(S)
  assert n == 39 and len(sf) == 196
  assert np.allclose(d_fatal <= R_FATAL, Ci.astype(bool)), 'label mismatch'
  print('final39: %d cases (%d covered / %d uncovered) | failure anchors: %d'
        % (n, Ci.sum(), n - Ci.sum(), len(sf)), flush=True)

  # ---- distances to the failure-training anchors ---------------------------
  NS_t, NS_f = ns(S), ns(sf)
  NA_t, NA_f = na(A), na(af)
  Ds = np.linalg.norm(NS_t[:, None] - NS_f[None], axis=2)        # [39, 196]
  Dsa = np.linalg.norm(
      np.concatenate([NS_t, NA_t], 1)[:, None]
      - np.concatenate([NS_f, NA_f], 1)[None], axis=2)
  Da = np.linalg.norm(A[:, None] - af[None], axis=2)             # raw actions
  d_s = Ds.min(1)
  d_sa = Dsa.min(1)
  d_a = Da.min(1)
  nn_id = Ds.argmin(1)
  kmeans = {k: np.sort(Ds, axis=1)[:, :k].mean(1) for k in KS}
  kmeans_sa = {k: np.sort(Dsa, axis=1)[:, :k].mean(1) for k in KS}

  # ---- control: distance to the ORDINARY CLEAN training pool --------------
  split = json.load(open(os.path.join(V0_DIR, 'split_manifest.json')))
  sg, ag, _ = stack_pairs(split['npz'],
                          np.asarray(split['train_episode_ids'], np.int64))
  NS_g = ns(sg)
  NA_g = na(ag)
  d_clean = np.full(n, np.inf)
  d_clean_sa = np.full(n, np.inf)
  CH = 20000
  for i0 in range(0, len(NS_g), CH):
    blk = NS_g[i0:i0 + CH]
    d_clean = np.minimum(
        d_clean, np.linalg.norm(NS_t[:, None] - blk[None], axis=2).min(1))
    blk2 = np.concatenate([blk, NA_g[i0:i0 + CH]], 1)
    d_clean_sa = np.minimum(
        d_clean_sa,
        np.linalg.norm(np.concatenate([NS_t, NA_t], 1)[:, None] - blk2[None],
                       axis=2).min(1))
  print('clean pool: %d anchors' % len(NS_g), flush=True)

  # ---- optional: is the required fatal DELTA different? -------------------
  nDf_t = (Sf_t - S - d_mean) / d_std
  nDf_f = (df_fail - d_mean) / d_std
  delta_gap = np.linalg.norm(nDf_t - nDf_f[nn_id], axis=1)

  # ---- covered vs uncovered ------------------------------------------------
  cov, unc = Ci == 1, Ci == 0
  metrics = {
      'nearest_failure_state_distance': d_s,
      'nearest_failure_state_action_distance': d_sa,
      'nearest_failure_action_only_distance': d_a,
      **{('failure_k%d_mean_state_distance' % k): kmeans[k] for k in KS},
      **{('failure_k%d_mean_state_action_distance' % k): kmeans_sa[k]
         for k in KS},
      'nearest_CLEAN_state_distance': d_clean,
      'nearest_CLEAN_state_action_distance': d_clean_sa,
      'fatal_delta_gap_to_nearest_failure': delta_gap,
  }
  table = {}
  for name, v in metrics.items():
    table[name] = {'covered': desc(v[cov]), 'uncovered': desc(v[unc]),
                   'effect': compare(v[cov], v[unc])}

  # ---- correlations + AUC --------------------------------------------------
  diagnostics = {
      'spearman_d_s_vs_fatal_error': corr_block(d_s, d_fatal, rng),
      'spearman_d_sa_vs_fatal_error': corr_block(d_sa, d_fatal, rng),
      'spearman_d_a_vs_fatal_error': corr_block(d_a, d_fatal, rng),
      'spearman_d_clean_vs_fatal_error': corr_block(d_clean, d_fatal, rng),
      'auc_from_state_failure_distance': auc_and_ci(-d_s, Ci, N_BOOT, rng),
      'auc_from_joint_failure_distance': auc_and_ci(-d_sa, Ci, N_BOOT, rng),
      'auc_from_action_only_failure_distance': auc_and_ci(-d_a, Ci, N_BOOT,
                                                          rng),
      'auc_from_CLEAN_distance_CONTROL': auc_and_ci(-d_clean, Ci, N_BOOT,
                                                    rng),
  }

  # ---- outputs -------------------------------------------------------------
  summary = {
      'hypothesis': ('final39 generator failures concentrate on anchors far '
                     'from the 196 factual failure-training anchors'),
      'n_cases': n, 'n_covered': int(cov.sum()),
      'n_uncovered': int(unc.sum()),
      'n_failure_anchors': int(len(sf)),
      'n_clean_anchors': int(len(NS_g)),
      'normalization': {
          'state': os.path.join(V0_DIR, 'norm_stats.npz') + ' (frozen)',
          'action': ACT_STATS + ' (clean TRAIN split only, pre-existing)',
          'note': ('d_s, d_sa and d_a reported separately; no state-vs-action '
                   'weight was chosen after seeing outcomes')},
      'table': table, 'diagnostics': diagnostics,
      'frozen_inputs': {
          'final39': os.path.join(FINAL39, 'candidates_and_scores.npz'),
          'R_fatal': R_FATAL, 'no_resampling': True},
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'diagnostic_summary.json'),
                          'w'), indent=2)

  with open(os.path.join(args.out, 'per_case.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'covered', 'd_fatal@256', 'd_safe@256',
                'd_fail_state', 'd_fail_state_action', 'd_fail_action_only',
                'd_fail_k5', 'd_clean_state', 'nearest_failure_anchor_idx',
                'fatal_delta_gap'])
    for i in range(n):
      w.writerow([i, int(eps[i]), int(Ci[i]), round(float(d_fatal[i]), 4),
                  round(float(d_safe[i]), 4), round(float(d_s[i]), 4),
                  round(float(d_sa[i]), 4), round(float(d_a[i]), 4),
                  round(float(kmeans[5][i]), 4),
                  round(float(d_clean[i]), 4), int(nn_id[i]),
                  round(float(delta_gap[i]), 4)])

  fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
  for a_, (xv, xl) in zip(ax[:2], [(d_s, 'nearest failure STATE distance'),
                                   (d_sa, 'nearest failure (s,a) distance')]):
    a_.scatter(xv[cov], d_fatal[cov], s=45, color='tab:green', label='covered')
    a_.scatter(xv[unc], d_fatal[unc], s=45, color='tab:red', marker='x',
               label='uncovered')
    a_.axhline(R_FATAL, color='green', ls=':', lw=1.5, label='R_fatal')
    a_.set_xlabel(xl)
    a_.set_ylabel('d_fatal@256')
    a_.legend(fontsize=7)
  bp = ax[2]
  data = [d_s[cov], d_s[unc], d_clean[cov], d_clean[unc]]
  bp.boxplot(data, labels=['fail\ncov', 'fail\nunc', 'clean\ncov',
                           'clean\nunc'])
  for i, dd in enumerate(data):
    bp.scatter(np.full(len(dd), i + 1) + rng.normal(0, 0.04, len(dd)), dd,
               s=14, alpha=0.6,
               color=('tab:green' if i % 2 == 0 else 'tab:red'))
  bp.set_ylabel('nearest-neighbour distance')
  bp.set_title('failure-support vs clean-support distance')
  ax[3].scatter(d_clean, d_s, s=45,
                c=np.where(cov, 'tab:green', 'tab:red'))
  ax[3].set_xlabel('nearest CLEAN distance')
  ax[3].set_ylabel('nearest FAILURE distance')
  ax[3].set_title('control: are uncovered far from everything?')
  fig.suptitle('final39 anchor-coverage diagnostic (%d covered / %d uncovered)'
               % (cov.sum(), unc.sum()))
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'anchor_coverage.png'), dpi=140)
  plt.close(fig)

  # ---- console -------------------------------------------------------------
  print('\n%-42s %12s %12s %10s %9s'
        % ('metric', 'covered(18)', 'uncovered(21)', 'diff', 'p(1-sided)'))
  for name in ('nearest_failure_state_distance',
               'nearest_failure_state_action_distance',
               'failure_k5_mean_state_distance',
               'nearest_CLEAN_state_distance',
               'nearest_failure_action_only_distance',
               'fatal_delta_gap_to_nearest_failure'):
    t = table[name]
    print('%-42s %12.3f %12.3f %10.3f %9.4f'
          % (name, t['covered']['median'], t['uncovered']['median'],
             t['effect']['median_diff_unc_minus_cov'],
             t['effect']['p_one_sided_uncovered_greater']))
  print('\n%-46s %8s %s' % ('diagnostic', 'value', 'CI95 / p'))
  for k_ in ('spearman_d_s_vs_fatal_error', 'spearman_d_sa_vs_fatal_error',
             'spearman_d_a_vs_fatal_error',
             'spearman_d_clean_vs_fatal_error'):
    d_ = diagnostics[k_]
    print('%-46s %8.3f perm p=%.4f'
          % (k_, d_['spearman_rho'],
             d_['spearman_p_permutation_two_sided']))
  for k_ in ('auc_from_state_failure_distance',
             'auc_from_joint_failure_distance',
             'auc_from_action_only_failure_distance',
             'auc_from_CLEAN_distance_CONTROL'):
    d_ = diagnostics[k_]
    print('%-46s %8.3f CI95 [%.3f, %.3f]'
          % (k_, d_['auc'], d_['ci95'][0], d_['ci95'][1]))
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
