"""Phases 7-8: V3 training-support diagnostics + dev16 and reused-final39
development evaluation, compared head-to-head against V2-SA-l001.

Phase 7  training-pool support at K=2048 on three subgroups: the old 196,
         the new 407, and the combined 603. Purpose is only to confirm that
         broadening diversity did not destroy the fit to observed factual
         failures -- NOT model selection.

Phase 8A dev16 at the frozen K=256 / R_fatal=3.17, with the same metric code
         used for V2: d_fatal, FatalCoverage, d_safe, clean-validation
         nearest-candidate error, candidate diversity, tails, physical stats.

Phase 8B the REUSED final39 -- DEVELOPMENT DIAGNOSTIC ONLY. final39 stopped
         being a sealed final test when it was used to motivate the V3
         diversity expansion. Nothing is tuned on it.

Mechanism test: per-anchor support-distance change
    delta_d_i = d_old(i) - d_diverse(i)
(nearest distance to the 196 vs to the 603 failure anchors) against the
per-anchor improvement in fatal error d_fatal^V2 - d_fatal^V3, plus the
V2->V3 coverage transition table and the previously-uncovered subgroup.

Critic C is NOT used to select anything; it appears only as a clearly
labelled secondary readout on newly recovered candidates.

Usage:
  python scripts/eval_flow_v3.py
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
import jax
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from eval_flow_v1_dev16 import make_sampler, R_FATAL  # noqa: E402
from probe_flow_v0_failure_coverage import phys, qstats  # noqa: E402
from train_flow_v2 import build_fail_pool  # noqa: E402

ROOT = 'artifacts/flow_v3_diverse_failure'
V3_CKPT = os.path.join(ROOT, 'flow_v3', 'flow_v3.pkl')
V2_CKPT = 'artifacts/flow_v2_failure_local/V2-SA-l001/flow_v2.pkl'
POOL = os.path.join(ROOT, 'failure_pool_diversity_audit',
                    'failure_pool_diverse.npz')
V0_DIR = 'artifacts/flow_v0_clean'
ACT_STATS = 'artifacts/flow_v05_clean_action/action_stats.json'
OLD_BAD_DIR, OLD_BAD_NAME = ('artifacts/bad_demo_fixed',
                             'bad_demo_blind_p30_h800_settle80')
DEV16 = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
FINAL39 = 'artifacts/flow_v2_final39/candidates_and_scores.npz'
FINAL39_CSV = 'artifacts/flow_v2_final39/per_case.csv'
K_PRIMARY, K_TRAIN = 256, 2048
DEV_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
CLEAN_K, CLEAN_ANCHORS, CLEAN_SEED = 32, 256, 1234
FINAL39_SEED = 11        # the frozen single-draw convention used on final39


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out-dev16', default=os.path.join(ROOT, 'dev16_eval'))
  ap.add_argument('--out-f39',
                  default=os.path.join(ROOT, 'reused_final39_dev_diag'))
  args = ap.parse_args()
  os.makedirs(args.out_dev16, exist_ok=True)
  os.makedirs(args.out_f39, exist_ok=True)

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  ast = json.load(open(ACT_STATS))
  a_mean = np.asarray(ast['per_dim_mean'], np.float32)
  a_std = np.asarray(ast['per_dim_std'], np.float32)
  ns = lambda x: (x - nrm['state_mean']) / nrm['state_std']   # noqa: E731
  na = lambda x: (x - a_mean) / a_std                          # noqa: E731
  nzd = lambda x: (x - nrm['delta_mean']) / nrm['delta_std']   # noqa: E731

  models = {}
  for tag, p in (('V2-SA-l001', V2_CKPT), ('V3-SA-diverse', V3_CKPT)):
    with open(p, 'rb') as f:
      ck = pickle.load(f)
    models[tag] = {'ck': ck,
                   'smp': make_sampler(ck['params'], ck['hidden'], True, nrm)}
  print('models: %s' % list(models), flush=True)

  # ================= Phase 7: training-support subgroups ===================
  z = np.load(POOL, allow_pickle=True)
  pf_s = np.asarray(z['state'], np.float32)
  pf_a = np.asarray(z['action'], np.float32)
  pf_d = np.asarray(z['delta'], np.float32)
  src = np.asarray(z['source'])
  old_m = src == 'old196'
  groups = {'old196': old_m, 'new407': ~old_m,
            'combined603': np.ones(len(src), bool)}
  So, Ao, Do, _ = build_fail_pool(OLD_BAD_DIR, OLD_BAD_NAME)
  phase7 = {}
  for tag, m in models.items():
    phase7[tag] = {}
    for gname, gm in groups.items():
      S_, A_, nD_ = pf_s[gm], pf_a[gm], nzd(pf_d[gm])
      dl = m['smp'](S_, K_TRAIN, jax.random.PRNGKey(0), A_)
      dmin = np.linalg.norm(nzd(dl) - nD_[:, None], axis=2).min(axis=1)
      phase7[tag][gname] = {'n': int(gm.sum()),
                            'd_fatal_at_2048': qstats(dmin),
                            'coverage_at_2048': float((dmin <= R_FATAL).mean())}
      print('  [P7] %-14s %-12s n=%3d cov@2048 %.3f (median d %.3f)'
            % (tag, gname, gm.sum(),
               phase7[tag][gname]['coverage_at_2048'],
               phase7[tag][gname]['d_fatal_at_2048']['median']), flush=True)

  # ================= shared evaluation helpers =============================
  split = json.load(open(os.path.join(V0_DIR, 'split_manifest.json')))
  rng = np.random.default_rng(CLEAN_SEED)
  with np.load(split['npz'], allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    ln = np.asarray(d['lengths'], np.int64)
  A58, N58, ACT = [], [], []
  for e in np.asarray(split['val_episode_ids'], np.int64):
    t = rng.integers(0, ln[e] - 1, size=8)
    A58.append(obs[e, t])
    N58.append(obs[e, t + 1])
    ACT.append(act[e, t])
  A58, N58, ACT = (np.concatenate(A58), np.concatenate(N58),
                   np.concatenate(ACT))
  sel = rng.permutation(len(A58))[:CLEAN_ANCHORS]
  Sc, Snc, Ac = A58[sel][:, :OBS_DIM], N58[sel][:, :OBS_DIM], ACT[sel]
  nDc = nzd(Snc - Sc)
  real_tail = float(np.abs(Snc - Sc).max())

  def clean_block(m):
    dc = m['smp'](Sc, CLEAN_K, jax.random.PRNGKey(CLEAN_SEED), Ac)
    ndc = nzd(dc)
    dmin = np.linalg.norm(ndc - nDc[:, None], axis=2).min(axis=1)
    iu = np.triu_indices(CLEAN_K, k=1)
    div = np.linalg.norm(ndc[:, :, None, :] - ndc[:, None, :, :],
                         axis=3)[:, iu[0], iu[1]].mean(axis=1)
    return {'nearest_candidate_error': qstats(dmin),
            'candidate_diversity': qstats(div),
            'delta_norm': qstats(np.linalg.norm(dc, axis=2)),
            'max_abs_delta': float(np.abs(dc).max()),
            'real_max_abs_delta': real_tail,
            'n_nonfinite': int((~np.isfinite(dc)).sum())}

  def anchor_block(m, S_, A_, nDf_, nDs_, seeds):
    """multi-seed (dev16) or single-seed (final39) evaluation at K=256."""
    df_all, ds_all, best = [], [], None
    bd = np.full(len(S_), np.inf)
    bc = np.zeros_like(S_)
    for sd in seeds:
      dl = m['smp'](S_, K_PRIMARY, jax.random.PRNGKey(sd), A_)
      nd = nzd(dl)
      df = np.linalg.norm(nd - nDf_[:, None], axis=2)
      ds = np.linalg.norm(nd - nDs_[:, None], axis=2)
      df_all.append(df.min(1))
      ds_all.append(ds.min(1))
      j = df.argmin(1)
      cur = df[np.arange(len(S_)), j]
      imp = cur < bd
      bc[imp] = (S_ + dl[np.arange(len(S_)), j])[imp]
      bd[imp] = cur[imp]
    df_all, ds_all = np.array(df_all), np.array(ds_all)
    per_anchor = df_all.mean(0) if len(seeds) > 1 else df_all[0]
    return {'d_fatal': qstats(df_all), 'd_safe': qstats(ds_all),
            'coverage': float(np.mean([(x <= R_FATAL).mean()
                                       for x in df_all])),
            'per_anchor_d_fatal': per_anchor,
            'per_anchor_d_safe': (ds_all.mean(0) if len(seeds) > 1
                                  else ds_all[0]),
            'nearest_candidates': bc}

  # ================= Phase 8A: dev16 =======================================
  p16 = np.load(DEV16, allow_pickle=True)
  S16 = np.asarray(p16['anchor_obs'], np.float32)
  A16 = np.asarray(p16['anchor_action'], np.float32)
  Sf16 = np.asarray(p16['fatal_candidate'], np.float32)
  Ss16 = np.asarray(p16['safe_candidate'], np.float32)
  nDf16, nDs16 = nzd(Sf16 - S16), nzd(Ss16 - S16)
  dev16, clean = {}, {}
  for tag, m in models.items():
    clean[tag] = clean_block(m)
    b = anchor_block(m, S16, A16, nDf16, nDs16, DEV_SEEDS)
    dev16[tag] = {k: v for k, v in b.items()
                  if k not in ('per_anchor_d_fatal', 'per_anchor_d_safe',
                               'nearest_candidates')}
    dev16[tag]['physical'] = {
        f: qstats([phys(x)[f] for x in b['nearest_candidates']])
        for f in ('torso_z', 'v_xy', 'up_z')}
    print('  [P8A] %-14s dev16 d_fatal %.3f cov %.3f | d_safe %.3f | clean '
          '%.3f' % (tag, dev16[tag]['d_fatal']['median'],
                    dev16[tag]['coverage'], dev16[tag]['d_safe']['median'],
                    clean[tag]['nearest_candidate_error']['median']),
          flush=True)
  dev16['reference_physical'] = {
      f: {'true_fatal': qstats([phys(x)[f] for x in Sf16]),
          'safe_successor': qstats([phys(x)[f] for x in Ss16])}
      for f in ('torso_z', 'v_xy', 'up_z')}

  # ================= Phase 8B: reused final39 ==============================
  z39 = np.load(FINAL39, allow_pickle=True)
  S39 = np.asarray(z39['anchor'], np.float32)
  A39 = np.asarray(z39['action'], np.float32)
  Sf39 = np.asarray(z39['true_fatal'], np.float32)
  Ss39 = np.asarray(z39['safe_successor'], np.float32)
  eps39 = np.asarray(z39['episode_id'], np.int64)
  nDf39, nDs39 = nzd(Sf39 - S39), nzd(Ss39 - S39)
  f39, f39_pa = {}, {}
  for tag, m in models.items():
    b = anchor_block(m, S39, A39, nDf39, nDs39, (FINAL39_SEED,))
    f39[tag] = {k: v for k, v in b.items()
                if k not in ('per_anchor_d_fatal', 'per_anchor_d_safe',
                             'nearest_candidates')}
    f39[tag]['physical'] = {
        f: qstats([phys(x)[f] for x in b['nearest_candidates']])
        for f in ('torso_z', 'v_xy', 'up_z')}
    f39_pa[tag] = b
    print('  [P8B] %-14s final39 d_fatal %.3f cov %.3f | d_safe %.3f'
          % (tag, f39[tag]['d_fatal']['median'], f39[tag]['coverage'],
             f39[tag]['d_safe']['median']), flush=True)
  f39['LABEL'] = ('REUSED final39 -- DEVELOPMENT DIAGNOSTIC ONLY; it is no '
                  'longer a sealed final test')

  # ---- mechanism: support-distance change vs improvement -------------------
  NSo, NAo = ns(So), na(Ao)
  NSp, NAp = ns(pf_s), na(pf_a)
  NS39, NA39 = ns(S39), na(A39)
  d_old = np.linalg.norm(NS39[:, None] - NSo[None], axis=2).min(1)
  d_div = np.linalg.norm(NS39[:, None] - NSp[None], axis=2).min(1)
  dd = d_old - d_div
  dfv2 = f39_pa['V2-SA-l001']['per_anchor_d_fatal']
  dfv3 = f39_pa['V3-SA-diverse']['per_anchor_d_fatal']
  improve = dfv2 - dfv3
  cov2 = dfv2 <= R_FATAL
  cov3 = dfv3 <= R_FATAL
  sp = spearmanr(dd, improve)
  trans = {'uncovered_V2_to_covered_V3': int((~cov2 & cov3).sum()),
           'covered_both': int((cov2 & cov3).sum()),
           'uncovered_both': int((~cov2 & ~cov3).sum()),
           'covered_V2_to_uncovered_V3': int((cov2 & ~cov3).sum())}
  prev_unc = ~cov2
  newly = (~cov2) & cov3
  mech = {
      'spearman_support_reduction_vs_fatal_improvement': {
          'rho': float(sp.statistic), 'p': float(sp.pvalue)},
      'coverage_transitions': trans,
      'previously_uncovered_subgroup': {
          'n': int(prev_unc.sum()),
          'newly_covered': int(newly.sum()),
          'V2_median_d_fatal': float(np.median(dfv2[prev_unc])),
          'V3_median_d_fatal': float(np.median(dfv3[prev_unc])),
          'median_support_reduction': float(np.median(dd[prev_unc])),
          'physical_of_newly_recovered': (
              {f: qstats([phys(x)[f] for x in
                          f39_pa['V3-SA-diverse']['nearest_candidates'][newly]])
               for f in ('torso_z', 'v_xy', 'up_z')} if newly.any() else None)},
      'note': 'development diagnostic; no learned classifier was fitted'}
  print('\n  [mech] rho(delta_support, delta_fatal_error) = %.3f (p=%.4f)'
        % (sp.statistic, sp.pvalue))
  print('  [mech] transitions: %s' % trans)
  print('  [mech] previously uncovered: %d/%d newly covered'
        % (newly.sum(), prev_unc.sum()))

  # ---- fidelity flags ------------------------------------------------------
  b2, b3 = models['V2-SA-l001'], models['V3-SA-diverse']
  del b2, b3
  ratios = {
      'clean_error_ratio': (clean['V3-SA-diverse']
                            ['nearest_candidate_error']['median']
                            / clean['V2-SA-l001']
                            ['nearest_candidate_error']['median']),
      'dev16_safe_ratio': (dev16['V3-SA-diverse']['d_safe']['median']
                           / dev16['V2-SA-l001']['d_safe']['median']),
      'final39_safe_ratio': (f39['V3-SA-diverse']['d_safe']['median']
                             / f39['V2-SA-l001']['d_safe']['median']),
      'tail_ratio': (clean['V3-SA-diverse']['max_abs_delta']
                     / clean['V2-SA-l001']['max_abs_delta']),
  }
  flags = {'clean_error_degraded_gt_20pct': ratios['clean_error_ratio'] > 1.2,
           'dev16_safe_degraded_gt_20pct': ratios['dev16_safe_ratio'] > 1.2,
           'final39_safe_degraded_gt_20pct':
               ratios['final39_safe_ratio'] > 1.2,
           'pathological_tail':
               clean['V3-SA-diverse']['max_abs_delta'] > 3.0 * real_tail,
           'nonfinite': clean['V3-SA-diverse']['n_nonfinite'] > 0}

  summary = {'phase7_training_support': phase7,
             'phase8A_dev16': dev16, 'clean_validation': clean,
             'phase8B_reused_final39': f39,
             'mechanism': mech,
             'fidelity': {'ratios': ratios, 'flags': flags},
             'frozen': {'lambda': 0.01, 'K': K_PRIMARY,
                        'R_fatal': R_FATAL, 'dev_seeds': list(DEV_SEEDS),
                        'final39_seed': FINAL39_SEED},
             'critic_C': 'not used for any V3 decision',
             'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out_dev16, 'v3_summary.json'),
                          'w'), indent=2)
  with open(os.path.join(args.out_f39, 'per_anchor_v2_v3.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'd_old_support', 'd_diverse_support',
                'delta_support', 'V2_d_fatal', 'V3_d_fatal', 'improvement',
                'V2_covered', 'V3_covered'])
    for i in range(len(S39)):
      w.writerow([i, int(eps39[i]), round(float(d_old[i]), 4),
                  round(float(d_div[i]), 4), round(float(dd[i]), 4),
                  round(float(dfv2[i]), 4), round(float(dfv3[i]), 4),
                  round(float(improve[i]), 4), int(cov2[i]), int(cov3[i])])

  fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
  ax[0].bar([0, 1], [dev16['V2-SA-l001']['coverage'],
                     dev16['V3-SA-diverse']['coverage']], 0.5,
            color=['tab:purple', 'crimson'])
  ax[0].bar([2.2, 3.2], [f39['V2-SA-l001']['coverage'],
                         f39['V3-SA-diverse']['coverage']], 0.5,
            color=['tab:purple', 'crimson'])
  ax[0].set_xticks([0, 1, 2.2, 3.2])
  ax[0].set_xticklabels(['V2\ndev16', 'V3\ndev16', 'V2\nfinal39*',
                         'V3\nfinal39*'], fontsize=8)
  ax[0].set_ylabel('FatalCoverage@256')
  ax[0].set_title('coverage (*reused dev diagnostic)')
  ax[1].scatter(dd[cov2], improve[cov2], s=40, color='tab:green',
                label='covered by V2')
  ax[1].scatter(dd[~cov2], improve[~cov2], s=40, color='tab:red', marker='x',
                label='uncovered by V2')
  ax[1].axhline(0, color='k', lw=0.8)
  ax[1].set_xlabel('support-distance reduction  d_old - d_diverse')
  ax[1].set_ylabel('fatal-error improvement  V2 - V3')
  ax[1].set_title('mechanism: rho = %.3f' % sp.statistic)
  ax[1].legend(fontsize=8)
  ax[2].scatter(dfv2, dfv3, s=40,
                c=np.where(newly, 'tab:orange',
                           np.where(cov3, 'tab:green', 'tab:red')))
  lim = [0, max(dfv2.max(), dfv3.max()) * 1.05]
  ax[2].plot(lim, lim, 'k--', lw=1)
  ax[2].axhline(R_FATAL, color='green', ls=':', lw=1.4)
  ax[2].axvline(R_FATAL, color='green', ls=':', lw=1.4)
  ax[2].set_xlabel('V2 d_fatal@256')
  ax[2].set_ylabel('V3 d_fatal@256')
  ax[2].set_title('orange = newly recovered')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out_f39, 'v2_vs_v3.png'), dpi=140)
  plt.close(fig)

  print('\n%-26s %14s %14s' % ('metric', 'V2-SA-l001', 'V3-SA-diverse'))
  rowspec = [
      ('clean error', lambda t: clean[t]['nearest_candidate_error']['median']),
      ('dev16 fatal d@256', lambda t: dev16[t]['d_fatal']['median']),
      ('dev16 coverage@256', lambda t: dev16[t]['coverage']),
      ('dev16 safe d@256', lambda t: dev16[t]['d_safe']['median']),
      ('final39* fatal d@256', lambda t: f39[t]['d_fatal']['median']),
      ('final39* coverage@256', lambda t: f39[t]['coverage']),
      ('final39* safe d@256', lambda t: f39[t]['d_safe']['median']),
      ('old196 train cov@2048', lambda t: phase7[t]['old196']
       ['coverage_at_2048']),
      ('new407 train cov@2048', lambda t: phase7[t]['new407']
       ['coverage_at_2048']),
      ('combined603 cov@2048', lambda t: phase7[t]['combined603']
       ['coverage_at_2048']),
  ]
  for name, fn in rowspec:
    print('%-26s %14.3f %14.3f'
          % (name, fn('V2-SA-l001'), fn('V3-SA-diverse')))
  print('\nfidelity ratios: %s' % {k: round(v, 3) for k, v in ratios.items()})
  print('flags: %s' % {k: v for k, v in flags.items() if v} or 'none')
  print('saved -> %s , %s' % (args.out_dev16, args.out_f39))


if __name__ == '__main__':
  main()
