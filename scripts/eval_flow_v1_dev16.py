"""Phase 4: dev16 validation + pre-registered gates for the eight V1 runs.

Evaluates every V1 run (and both clean baselines) with the SAME code, the
SAME 16 pilot development death anchors, the SAME 8 sampling seeds and the
SAME frozen 50-step Euler sampler used for V0/V0.5.

Primary quantity (normalized delta space, frozen V0 normalization):
    d_fatal(K) = min_{k<=K} || norm(dhat_k) - norm(ds_fatal) ||_2
    d_safe(K)  = same toward the same-anchor safe successor (control)

    FatalCoverage@K = #{i : d_i,fatal(K) <= R_FATAL} / 16,  R_FATAL = 3.17
(the empirical settled-failure neighbourhood established BEFORE V1; frozen).

PRE-REGISTERED SUCCESS GATE (primary K = 256):
    median d_fatal@256 <= 3.17           AND
    FatalCoverage@256  >= 8/16           AND  no ordinary-modeling collapse:
    clean nearest-candidate error   <= 1.20 x family baseline
    dev16 safe-successor dist @256   <= 1.20 x family baseline
    generated tails numerically plausible
Baselines: V1-S vs V0, V1-SA vs V0.5.

A run that fails at K=256 but meets the coverage criterion at K=2048 is
classified WEAK (rare fatal support), not a pass.

The 39 sealed same-anchor cases and the 40 fresh death stream are NOT opened.

Usage:
  python scripts/eval_flow_v1_dev16.py
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import make_net, OBS_DIM   # noqa: E402
from train_flow_v05_clean_action import make_net_a  # noqa: E402
from probe_flow_v0_failure_coverage import phys, qstats  # noqa: E402

SWEEP = 'artifacts/flow_v1_sweep'
OUT = 'artifacts/flow_v1_dev16'
PAIRS16 = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
V0_DIR = 'artifacts/flow_v0_clean'
V05_DIR = 'artifacts/flow_v05_clean_action'
R_FATAL = 3.17                       # frozen BEFORE V1; do not change
K_GRID = (32, 64, 128, 256, 2048)
K_PRIMARY = 256
DEV_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
ODE_STEPS = 50
CLEAN_K, CLEAN_ANCHORS, CLEAN_SEED = 32, 256, 1234
COV_MIN = 8 / 16
DEGRADE_MAX = 1.20


def make_sampler(params, hidden, use_a, nrm):
  net = make_net_a(tuple(hidden), OBS_DIM) if use_a \
      else make_net(tuple(hidden), OBS_DIM)

  @jax.jit
  def v(x, t, s, a):
    return net.apply(params, x, t, s, a) if use_a \
        else net.apply(params, x, t, s)

  def sample(s_raw, K, key, act=None):
    n = s_raw.shape[0]
    s_n = (s_raw - nrm['state_mean']) / nrm['state_std']
    s_rep = jnp.asarray(np.repeat(s_n, K, axis=0))
    a_rep = jnp.asarray(np.repeat(act, K, axis=0)) if use_a else \
        jnp.zeros((n * K, 8))
    x = jax.random.normal(key, (n * K, OBS_DIM))
    dt = 1.0 / ODE_STEPS
    for i in range(ODE_STEPS):
      tt = jnp.full((n * K, 1), i * dt)
      x = x + dt * v(x, tt, s_rep, a_rep)
    dlt = np.asarray(x) * nrm['delta_std'] + nrm['delta_mean']
    return dlt.reshape(n, K, OBS_DIM)
  return sample


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--sweep', default=SWEEP)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']

  # ---- dev16 anchors -------------------------------------------------------
  p = np.load(PAIRS16, allow_pickle=True)
  S16 = np.asarray(p['anchor_obs'], np.float32)
  A16 = np.asarray(p['anchor_action'], np.float32)
  nDf = nzd(np.asarray(p['fatal_candidate'], np.float32) - S16)
  nDs = nzd(np.asarray(p['safe_candidate'], np.float32) - S16)
  Sf = np.asarray(p['fatal_candidate'], np.float32)

  # ---- clean validation anchors (identical protocol to the baselines) -----
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

  # ---- model list: baselines + sweep --------------------------------------
  models = []
  for tag, dd, fn, ua in (('V0', V0_DIR, 'flow_v0.pkl', False),
                          ('V0.5', V05_DIR, 'flow_v05.pkl', True)):
    fp = os.path.join(dd, fn)
    if os.path.exists(fp):
      with open(fp, 'rb') as f:
        c = pickle.load(f)
      models.append({'tag': tag, 'family': 'SA' if ua else 'S',
                     'beta': 0.0, 'ck': c, 'use_a': ua, 'dir': dd})
  for rd in sorted(glob.glob(os.path.join(args.sweep, 'V1-*'))):
    fp = os.path.join(rd, 'flow_v1.pkl')
    if not os.path.exists(fp):
      print('  (pending: %s)' % os.path.basename(rd))
      continue
    with open(fp, 'rb') as f:
      c = pickle.load(f)
    models.append({'tag': os.path.basename(rd), 'family': c['family'],
                   'beta': c['beta'], 'ck': c,
                   'use_a': c['family'] == 'SA', 'dir': rd})
  print('evaluating %d models' % len(models), flush=True)

  rows = {}
  for m in models:
    smp = make_sampler(m['ck']['params'], m['ck']['hidden'], m['use_a'], nrm)
    # clean validation (same protocol as the baselines)
    dc = smp(Sc, CLEAN_K, jax.random.PRNGKey(CLEAN_SEED),
             Ac if m['use_a'] else None)
    ndc = nzd(dc)
    clean_dmin = np.linalg.norm(ndc - nDc[:, None], axis=2).min(axis=1)
    tail = float(np.abs(dc).max())
    # dev16
    df_k = {K: np.zeros((len(DEV_SEEDS), 16)) for K in K_GRID}
    ds_k = {K: np.zeros((len(DEV_SEEDS), 16)) for K in K_GRID}
    for si, sd in enumerate(DEV_SEEDS):
      dlt = smp(S16, max(K_GRID), jax.random.PRNGKey(sd),
                A16 if m['use_a'] else None)
      nd = nzd(dlt)
      df = np.linalg.norm(nd - nDf[:, None], axis=2)
      ds = np.linalg.norm(nd - nDs[:, None], axis=2)
      for K in K_GRID:
        df_k[K][si] = df[:, :K].min(axis=1)
        ds_k[K][si] = ds[:, :K].min(axis=1)
    cov = {K: float(np.mean([(df_k[K][si] <= R_FATAL).mean()
                             for si in range(len(DEV_SEEDS))]))
           for K in K_GRID}
    rows[m['tag']] = {
        'family': m['family'], 'beta': m['beta'], 'dir': m['dir'],
        'clean_nearest_candidate_error': qstats(clean_dmin),
        'clean_max_abs_delta': tail,
        'd_fatal': {str(K): qstats(df_k[K]) for K in K_GRID},
        'd_safe': {str(K): qstats(ds_k[K]) for K in K_GRID},
        'fatal_coverage': {str(K): cov[K] for K in K_GRID},
        'median_d_fatal_at_primary': float(np.median(df_k[K_PRIMARY])),
    }
    print('  %-12s clean dmin %.3f | d_fatal@256 %.3f | cov@256 %.3f | '
          'd_safe@256 %.3f'
          % (m['tag'], np.median(clean_dmin), np.median(df_k[K_PRIMARY]),
             cov[K_PRIMARY], np.median(ds_k[K_PRIMARY])), flush=True)

  # ---- gates ---------------------------------------------------------------
  base = {'S': rows.get('V0'), 'SA': rows.get('V0.5')}
  verdicts = {}
  for tag, r in rows.items():
    if tag in ('V0', 'V0.5'):
      continue
    b = base[r['family']]
    cov256 = r['fatal_coverage'][str(K_PRIMARY)]
    med256 = r['median_d_fatal_at_primary']
    cov2048 = r['fatal_coverage'][str(max(K_GRID))]
    med2048 = float(r['d_fatal'][str(max(K_GRID))]['median'])
    clean_ratio = (r['clean_nearest_candidate_error']['median']
                   / b['clean_nearest_candidate_error']['median'])
    safe_ratio = (r['d_safe'][str(K_PRIMARY)]['median']
                  / b['d_safe'][str(K_PRIMARY)]['median'])
    tail_ok = r['clean_max_abs_delta'] < 3.0 * b['clean_max_abs_delta']
    no_collapse = (clean_ratio <= DEGRADE_MAX and safe_ratio <= DEGRADE_MAX
                   and tail_ok)
    passed = (med256 <= R_FATAL and cov256 >= COV_MIN and no_collapse)
    weak = (not passed and med2048 <= R_FATAL and cov2048 >= COV_MIN)
    verdicts[tag] = {
        'family': r['family'], 'beta': r['beta'],
        'median_d_fatal_at_256': med256, 'fatal_coverage_at_256': cov256,
        'median_d_fatal_at_2048': med2048,
        'fatal_coverage_at_2048': cov2048,
        'clean_error_ratio_vs_baseline': clean_ratio,
        'safe_dist_ratio_vs_baseline': safe_ratio,
        'tail_ok': bool(tail_ok), 'no_ordinary_collapse': bool(no_collapse),
        'verdict': ('PASS' if passed else
                    'WEAK (rare fatal support)' if weak else 'FAIL')}

  passing = [t for t, v in verdicts.items() if v['verdict'] == 'PASS']
  selected = None
  if passing:
    selected = sorted(passing, key=lambda t: (verdicts[t]['beta'],
                                              t))[0]
  summary = {
      'gate': {'R_fatal_frozen': R_FATAL, 'primary_K': K_PRIMARY,
               'median_d_fatal_max': R_FATAL,
               'fatal_coverage_min': COV_MIN,
               'max_degradation_vs_family_baseline': DEGRADE_MAX,
               'note': 'thresholds frozen before V1 results were seen'},
      'baselines': {'S': 'V0', 'SA': 'V0.5'},
      'models': rows, 'verdicts': verdicts,
      'passing_runs': passing,
      'selected_run_smallest_passing_beta': selected,
      'protected': '39 sealed cases and the 40 fresh death stream not opened',
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'dev16_summary.json'), 'w'),
            indent=2)

  with open(os.path.join(args.out, 'dev16_table.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['run', 'family', 'beta', 'clean_dmin', 'd_fatal@256',
                'cov@256', 'd_fatal@2048', 'cov@2048', 'd_safe@256',
                'clean_ratio', 'safe_ratio', 'verdict'])
    for tag, r in rows.items():
      v = verdicts.get(tag, {})
      w.writerow([tag, r['family'], r['beta'],
                  round(r['clean_nearest_candidate_error']['median'], 4),
                  round(r['median_d_fatal_at_primary'], 4),
                  round(r['fatal_coverage'][str(K_PRIMARY)], 4),
                  round(float(r['d_fatal'][str(max(K_GRID))]['median']), 4),
                  round(r['fatal_coverage'][str(max(K_GRID))], 4),
                  round(r['d_safe'][str(K_PRIMARY)]['median'], 4),
                  round(v.get('clean_error_ratio_vs_baseline', float('nan')), 3),
                  round(v.get('safe_dist_ratio_vs_baseline', float('nan')), 3),
                  v.get('verdict', 'baseline')])

  ks = np.array(K_GRID)
  fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
  for tag, r in rows.items():
    st = '--' if tag in ('V0', 'V0.5') else '-'
    lw = 1.2 if tag in ('V0', 'V0.5') else 1.8
    col = 'gray' if tag in ('V0', 'V0.5') else None
    ax[0].plot(ks, [r['d_fatal'][str(K)]['median'] for K in K_GRID],
               st, lw=lw, marker='o', ms=4, color=col, label=tag)
    ax[1].plot(ks, [r['fatal_coverage'][str(K)] for K in K_GRID],
               st, lw=lw, marker='o', ms=4, color=col, label=tag)
  ax[0].axhline(R_FATAL, color='green', ls=':', lw=1.6,
                label='R_fatal = 3.17 (frozen)')
  ax[0].set_xscale('log', base=2)
  ax[0].set_xlabel('K')
  ax[0].set_ylabel('median d_fatal')
  ax[0].set_title('V1 sweep: distance to the settled fatal mode')
  ax[0].legend(fontsize=6.5, ncol=2)
  ax[1].axhline(COV_MIN, color='green', ls=':', lw=1.6, label='gate 8/16')
  ax[1].set_xscale('log', base=2)
  ax[1].set_xlabel('K')
  ax[1].set_ylabel('FatalCoverage@K')
  ax[1].set_title('fraction of the 16 anchors covered within R_fatal')
  ax[1].legend(fontsize=6.5, ncol=2)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'v1_dev16.png'), dpi=140)
  plt.close(fig)

  print('\n%-12s %-3s %-5s %8s %8s %8s %8s  %s'
        % ('run', 'fam', 'beta', 'dfat@256', 'cov@256', 'dfat@2k',
           'cov@2k', 'verdict'))
  for tag, r in rows.items():
    v = verdicts.get(tag, {})
    print('%-12s %-3s %-5.2f %8.3f %8.3f %8.3f %8.3f  %s'
          % (tag, r['family'], r['beta'],
             r['median_d_fatal_at_primary'],
             r['fatal_coverage'][str(K_PRIMARY)],
             float(r['d_fatal'][str(max(K_GRID))]['median']),
             r['fatal_coverage'][str(max(K_GRID))],
             v.get('verdict', 'baseline')))
  print('\npassing: %s | selected (smallest beta): %s'
        % (passing or 'NONE', selected))
  print('saved -> %s' % args.out)


if __name__ == '__main__':
  main()
