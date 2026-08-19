"""V2 evaluation: dev16 gate, clean validation, training-anchor diagnostic,
physical comparison, and (secondary) Critic-C scores.

Same 16 pilot development death anchors, same frozen 50-step Euler sampler,
same frozen V0 normalization and the same pre-registered gate as V0/V0.5/V1:

    PASS if median d_fatal@256 <= R_FATAL (3.17) AND FatalCoverage@256 >= 8/16
    AND clean nearest-candidate error  <= 1.20 x V0.5
    AND dev16 safe-successor dist @256 <= 1.20 x V0.5
    AND no pathological tails / non-finite values.

Training-anchor diagnostic (decisive in V1): the same measurement at the 196
factual pre-death anchors whose fatal transitions ARE in D_fail, K=2048.
  * near-zero there  -> not merely an empirical-frequency problem;
  * high there, low on dev16 -> a generalization problem;
  * high on both -> failure-local rebalancing recovered support.

Critic C is SECONDARY: computed only after the geometric/physical analysis is
fixed, never used to train V2 and never used to choose lambda.

The 39 sealed cases and the 40 fresh death stream are NOT opened.

Usage:
  python scripts/eval_flow_v2_dev16.py
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
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from eval_flow_v1_dev16 import make_sampler, R_FATAL, COV_MIN, DEGRADE_MAX  # noqa: E402
from probe_flow_v0_failure_coverage import phys, qstats  # noqa: E402
from train_flow_v2 import build_fail_pool  # noqa: E402

SWEEP = 'artifacts/flow_v2_failure_local'
PAIRS16 = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
V0_DIR = 'artifacts/flow_v0_clean'
V05_DIR = 'artifacts/flow_v05_clean_action'
BAD_DIR = 'artifacts/bad_demo_fixed'
BAD_NAME = 'bad_demo_blind_p30_h800_settle80'
K_GRID = (32, 64, 128, 256, 2048)
K_PRIMARY = 256
DEV_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
CLEAN_K, CLEAN_ANCHORS, CLEAN_SEED = 32, 256, 1234


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--sweep', default=SWEEP)
  ap.add_argument('--out', default=SWEEP)
  ap.add_argument('--train-anchor-k', type=int, default=2048)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']

  # ---- dev16 ---------------------------------------------------------------
  p = np.load(PAIRS16, allow_pickle=True)
  S16 = np.asarray(p['anchor_obs'], np.float32)
  A16 = np.asarray(p['anchor_action'], np.float32)
  Sf16 = np.asarray(p['fatal_candidate'], np.float32)
  Ss16 = np.asarray(p['safe_candidate'], np.float32)
  nDf, nDs = nzd(Sf16 - S16), nzd(Ss16 - S16)

  # ---- training-anchor pool (196 fatal transitions in D_fail) -------------
  sfa, afa, dfa, _ = build_fail_pool(BAD_DIR, BAD_NAME)
  nDfa = nzd(dfa)

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
  real_tail = float(np.abs(Snc - Sc).max())

  # ---- models: V0.5 baseline + the four V2 runs ---------------------------
  models = []
  with open(os.path.join(V05_DIR, 'flow_v05.pkl'), 'rb') as f:
    models.append({'tag': 'V0.5', 'lam': 0.0, 'ck': pickle.load(f)})
  order = {'V2-SA-l001': 0.01, 'V2-SA-l0025': 0.025,
           'V2-SA-l005': 0.05, 'V2-SA-l010': 0.10}
  for rd in sorted(glob.glob(os.path.join(args.sweep, 'V2-SA-*')),
                   key=lambda x: order.get(os.path.basename(x), 99)):
    fp = os.path.join(rd, 'flow_v2.pkl')
    if not os.path.exists(fp):
      print('  (pending: %s)' % os.path.basename(rd))
      continue
    with open(fp, 'rb') as f:
      ck = pickle.load(f)
    models.append({'tag': os.path.basename(rd), 'lam': ck['lam'], 'ck': ck})
  print('evaluating %d models' % len(models), flush=True)

  rows, best_cand = {}, {}
  for m in models:
    smp = make_sampler(m['ck']['params'], m['ck']['hidden'], True, nrm)
    # clean validation
    dc = smp(Sc, CLEAN_K, jax.random.PRNGKey(CLEAN_SEED), Ac)
    ndc = nzd(dc)
    clean_dmin = np.linalg.norm(ndc - nDc[:, None], axis=2).min(axis=1)
    iu = np.triu_indices(CLEAN_K, k=1)
    div = np.linalg.norm(ndc[:, :, None, :] - ndc[:, None, :, :],
                         axis=3)[:, iu[0], iu[1]].mean(axis=1)
    dnorm = np.linalg.norm(dc, axis=2)
    # dev16
    df_k = {K: np.zeros((len(DEV_SEEDS), 16)) for K in K_GRID}
    ds_k = {K: np.zeros((len(DEV_SEEDS), 16)) for K in K_GRID}
    bd = np.full(16, np.inf)
    bc = np.zeros_like(S16)
    for si, sd in enumerate(DEV_SEEDS):
      dlt = smp(S16, max(K_GRID), jax.random.PRNGKey(sd), A16)
      nd = nzd(dlt)
      df = np.linalg.norm(nd - nDf[:, None], axis=2)
      ds = np.linalg.norm(nd - nDs[:, None], axis=2)
      for K in K_GRID:
        df_k[K][si] = df[:, :K].min(axis=1)
        ds_k[K][si] = ds[:, :K].min(axis=1)
      j = df.argmin(axis=1)
      cur = df[np.arange(16), j]
      imp = cur < bd
      bc[imp] = (S16 + dlt[np.arange(16), j])[imp]
      bd[imp] = cur[imp]
    best_cand[m['tag']] = bc
    cov = {K: float(np.mean([(df_k[K][si] <= R_FATAL).mean()
                             for si in range(len(DEV_SEEDS))]))
           for K in K_GRID}
    # training-anchor diagnostic
    dta = smp(sfa, args.train_anchor_k, jax.random.PRNGKey(0), afa)
    dmin_ta = np.linalg.norm(nzd(dta) - nDfa[:, None], axis=2).min(axis=1)
    rows[m['tag']] = {
        'lambda': m['lam'],
        'clean_nearest_candidate_error': qstats(clean_dmin),
        'clean_diversity': qstats(div),
        'clean_delta_norm': qstats(dnorm),
        'clean_max_abs_delta': float(np.abs(dc).max()),
        'clean_n_nonfinite': int((~np.isfinite(dc)).sum()),
        'd_fatal': {str(K): qstats(df_k[K]) for K in K_GRID},
        'd_safe': {str(K): qstats(ds_k[K]) for K in K_GRID},
        'fatal_coverage': {str(K): cov[K] for K in K_GRID},
        'train_anchor': {'k': args.train_anchor_k,
                         'n_anchors': int(len(sfa)),
                         'd_fatal': qstats(dmin_ta),
                         'coverage': float((dmin_ta <= R_FATAL).mean())},
    }
    print('  %-12s clean %.3f | fatal@256 %.3f cov %.3f | safe@256 %.3f | '
          'train-anchor cov@%d %.3f'
          % (m['tag'], np.median(clean_dmin),
             np.median(df_k[K_PRIMARY]), cov[K_PRIMARY],
             np.median(ds_k[K_PRIMARY]), args.train_anchor_k,
             rows[m['tag']]['train_anchor']['coverage']), flush=True)

  # ---- gates ---------------------------------------------------------------
  b = rows['V0.5']
  verdicts = {}
  for tag, r in rows.items():
    if tag == 'V0.5':
      continue
    med = r['d_fatal'][str(K_PRIMARY)]['median']
    cov = r['fatal_coverage'][str(K_PRIMARY)]
    ce = (r['clean_nearest_candidate_error']['median']
          / b['clean_nearest_candidate_error']['median'])
    se = (r['d_safe'][str(K_PRIMARY)]['median']
          / b['d_safe'][str(K_PRIMARY)]['median'])
    tail_ok = (r['clean_max_abs_delta'] < 3.0 * real_tail
               and r['clean_n_nonfinite'] == 0)
    no_collapse = ce <= DEGRADE_MAX and se <= DEGRADE_MAX and tail_ok
    verdicts[tag] = {
        'lambda': r['lambda'],
        'median_d_fatal_at_256': med, 'fatal_coverage_at_256': cov,
        'clean_error_ratio': ce, 'safe_dist_ratio': se,
        'tail_ok': bool(tail_ok), 'no_ordinary_collapse': bool(no_collapse),
        'verdict': ('PASS' if (med <= R_FATAL and cov >= COV_MIN
                               and no_collapse) else 'FAIL')}
  passing = [t for t, v in verdicts.items() if v['verdict'] == 'PASS']
  selected = min(passing, key=lambda t: verdicts[t]['lambda']) if passing \
      else None

  # ---- physical + SECONDARY critic ----------------------------------------
  physical = {}
  for feat in ('torso_z', 'v_xy', 'up_z'):
    physical[feat] = {
        'true_fatal': qstats([phys(x)[feat] for x in Sf16]),
        'safe_successor': qstats([phys(x)[feat] for x in Ss16]),
        **{('%s_nearest' % t): qstats([phys(x)[feat] for x in best_cand[t]])
           for t in best_cand}}

  from crl import networks as networks_mod
  from crl import checkpoint as ckpt_mod
  from crl.replay import obs_to_goal
  from verify_offline_d4rl import build_offline_cfg
  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=OBS_DIM, goal_dim=OBS_DIM, action_dim=8,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  _, c_state = ckpt_mod.load_checkpoint(
      'failneg_settledbank_a01_s0_300k/'
      'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')

  @jax.jit
  def c_score(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  sec = {}
  for name, cand in ([('true_fatal', Sf16), ('safe_successor', Ss16)]
                     + [('%s_nearest' % t, best_cand[t]) for t in best_cand]):
    g = np.asarray(obs_to_goal(cand.astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    og = np.concatenate([S16, g], axis=1)
    f = np.asarray(c_score(jnp.asarray(og), jnp.asarray(A16)))
    sec[name] = qstats(f.min(axis=1))
  secondary = {'critic_C_f_min': sec,
               'note': ('SECONDARY -- computed after the geometric/physical '
                        'analysis was fixed; never trains V2, never chooses '
                        'lambda')}

  summary = {
      'experiment': 'V2 failure-local balanced Flow (SA family only)',
      'gate': {'R_fatal': R_FATAL, 'primary_K': K_PRIMARY,
               'fatal_coverage_min': COV_MIN,
               'max_degradation_vs_V0.5': DEGRADE_MAX},
      'models': rows, 'verdicts': verdicts, 'passing': passing,
      'selected_smallest_passing_lambda': selected,
      'physical': physical, 'secondary_critic': secondary,
      'protected': '39 sealed cases and the 40 fresh death stream not opened',
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'v2_summary.json'), 'w'),
            indent=2)

  with open(os.path.join(args.out, 'v2_table.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['run', 'lambda', 'clean_err', 'safe_d@256', 'fatal_d@256',
                'cov@256', 'fatal_d@2048', 'cov@2048',
                'train_anchor_cov@2048', 'verdict'])
    for tag, r in rows.items():
      v = verdicts.get(tag, {})
      w.writerow([tag, r['lambda'],
                  round(r['clean_nearest_candidate_error']['median'], 4),
                  round(r['d_safe'][str(K_PRIMARY)]['median'], 4),
                  round(r['d_fatal'][str(K_PRIMARY)]['median'], 4),
                  round(r['fatal_coverage'][str(K_PRIMARY)], 4),
                  round(r['d_fatal']['2048']['median'], 4),
                  round(r['fatal_coverage']['2048'], 4),
                  round(r['train_anchor']['coverage'], 4),
                  v.get('verdict', 'baseline')])

  # ---- plots ---------------------------------------------------------------
  tags = [t for t in rows if t != 'V0.5']
  lams = [rows[t]['lambda'] for t in tags]
  fig, ax = plt.subplots(1, 5, figsize=(23, 4.2))
  series = [
      ('dev16 median d_fatal@256',
       [rows[t]['d_fatal'][str(K_PRIMARY)]['median'] for t in tags],
       rows['V0.5']['d_fatal'][str(K_PRIMARY)]['median'], R_FATAL),
      ('FatalCoverage@256',
       [rows[t]['fatal_coverage'][str(K_PRIMARY)] for t in tags],
       rows['V0.5']['fatal_coverage'][str(K_PRIMARY)], COV_MIN),
      ('clean nearest-candidate err',
       [rows[t]['clean_nearest_candidate_error']['median'] for t in tags],
       rows['V0.5']['clean_nearest_candidate_error']['median'],
       DEGRADE_MAX * rows['V0.5']['clean_nearest_candidate_error']['median']),
      ('dev16 d_safe@256',
       [rows[t]['d_safe'][str(K_PRIMARY)]['median'] for t in tags],
       rows['V0.5']['d_safe'][str(K_PRIMARY)]['median'],
       DEGRADE_MAX * rows['V0.5']['d_safe'][str(K_PRIMARY)]['median']),
      ('train-anchor FatalCoverage',
       [rows[t]['train_anchor']['coverage'] for t in tags],
       rows['V0.5']['train_anchor']['coverage'], None)]
  for a_, (title, ys, base, gate) in zip(ax, series):
    a_.plot(lams, ys, 'o-', color='crimson', label='V2')
    a_.axhline(base, color='tab:purple', ls='--', lw=1.2, label='V0.5')
    if gate is not None:
      a_.axhline(gate, color='green', ls=':', lw=1.5, label='gate')
    a_.set_xscale('log')
    a_.set_xlabel('lambda')
    a_.set_title(title, fontsize=10)
    a_.legend(fontsize=7)
  fig.suptitle('V2 failure-local rebalancing sweep')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'v2_sweep.png'), dpi=140)
  plt.close(fig)

  print('\n%-13s %-6s %9s %10s %10s %9s %11s  %s'
        % ('run', 'lam', 'clean', 'safe@256', 'fatal@256', 'cov@256',
           'trainanchor', 'verdict'))
  for tag, r in rows.items():
    v = verdicts.get(tag, {})
    print('%-13s %-6.3g %9.3f %10.3f %10.3f %9.3f %11.3f  %s'
          % (tag, r['lambda'],
             r['clean_nearest_candidate_error']['median'],
             r['d_safe'][str(K_PRIMARY)]['median'],
             r['d_fatal'][str(K_PRIMARY)]['median'],
             r['fatal_coverage'][str(K_PRIMARY)],
             r['train_anchor']['coverage'], v.get('verdict', 'baseline')))
  print('\npassing: %s | selected: %s' % (passing or 'NONE', selected))
  print('saved -> %s' % args.out)


if __name__ == '__main__':
  main()
