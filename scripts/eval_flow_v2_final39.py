"""SEALED ONE-SHOT evaluation: frozen V2 Flow + frozen Critic C on final39.

Everything is frozen before the sealed set is opened, and the script ABORTS
on any provenance mismatch:

  generator  V2-SA-l001  (action-conditioned, lambda = 0.01)
  candidates K = 256, fixed-step 50-step Euler, frozen seed convention
  normalization  the frozen V0/V0.5 stats (never recomputed)
  scorer     Critic C, settled-bank alpha=0.1 best.pkl, twin f_min
  data       the 39 reserved same-anchor cases (sha checked against
             probe_freeze.json), with the action taken exactly as frozen in
             the pair artifact (anchor_action)

Part A (generator): d_fatal, FatalCoverage@256 at the frozen R_fatal = 3.17
with a Clopper-Pearson 95% CI, d_safe, and the physical diagnostics already
used in development.

Part B (selector): f_C over all 256 candidates, s'_wc = argmin_k f_C, then
  SelectionRate = P(selected is fatal-like | Flow covered the fatal mode)
  EndToEndRate  = P(selected is fatal-like) over all 39
plus critic score diagnostics and an explicit off-manifold / argmin
pathology check.

One shot. No retraining, no lambda/K/seed/radius changes, no alternative
model tested after opening the set.

Usage:
  python scripts/eval_flow_v2_final39.py
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import beta as _beta

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

FLOW_CKPT = 'artifacts/flow_v2_failure_local/V2-SA-l001/flow_v2.pkl'
V0_DIR = 'artifacts/flow_v0_clean'
PAIRS = 'artifacts/same_anchor_candidate_probe/pairs_heldout40.npz'
FREEZE = 'artifacts/same_anchor_candidate_probe/probe_freeze.json'
DEV16_SUMMARY = 'artifacts/flow_v2_failure_local/v2_summary.json'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
OUT = 'artifacts/flow_v2_final39'
K = 256                     # frozen
ODE_STEPS = 50              # frozen
LAMBDA = 0.01               # frozen
SEED = 11                   # frozen: the first of the dev sampling seeds


def cp_ci(k, n, alpha=0.05):
  lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
  hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
  return [lo, hi]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  # ================= 1. provenance freeze check ============================
  with open(FLOW_CKPT, 'rb') as f:
    ck = pickle.load(f)
  assert ck['family'] == 'SA', 'generator must be action-conditioned'
  assert abs(ck['lam'] - LAMBDA) < 1e-12, 'lambda is not the frozen 0.01'
  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  for k_ in nrm:
    assert np.array_equal(nrm[k_], np.asarray(ck['norm'][k_], np.float32)), \
        'normalization drifted from the frozen V0 stats'
  freeze = json.load(open(FREEZE))
  pairs_sha = C.sha256_file(PAIRS)
  assert pairs_sha == freeze['pairs_heldout_sha256'], \
      'sealed pair artifact does not match the frozen sha'
  prov = {
      'generator_ckpt': FLOW_CKPT,
      'generator_sha256': C.sha256_file(FLOW_CKPT),
      'family': ck['family'], 'lambda': ck['lam'],
      'flow_config': ck['config'], 'n_params': ck['n_params'],
      'K': K, 'ode_steps': ODE_STEPS, 'solver': 'fixed-step explicit Euler',
      'sampling_seed': SEED,
      'normalization_source': os.path.join(V0_DIR, 'norm_stats.npz'),
      'critic_ckpt': C_CKPT, 'critic_sha256': C.sha256_file(C_CKPT),
      'critic_aggregation': 'twin f_min (frozen, as validated)',
      'sealed_pairs': PAIRS, 'sealed_pairs_sha256': pairs_sha,
      'R_fatal': R_FATAL,
      'action_source': 'anchor_action, exactly as frozen in the pair artifact',
      'git_commit': C.git_commit(),
  }
  print('PROVENANCE OK')
  for k_ in ('generator_ckpt', 'lambda', 'K', 'sampling_seed',
             'critic_ckpt', 'sealed_pairs_sha256', 'git_commit'):
    print('  %-22s %s' % (k_, prov[k_]))

  # ================= 2-3. open the sealed set ==============================
  d = np.load(PAIRS, allow_pickle=True)
  S = np.asarray(d['anchor_obs'], np.float32)
  A = np.asarray(d['anchor_action'], np.float32)
  Sf = np.asarray(d['fatal_candidate'], np.float32)
  Ss = np.asarray(d['safe_candidate'], np.float32)
  eps = np.asarray(d['episode_id'], np.int64)
  n = len(S)
  print('\nsealed cases opened: %d' % n, flush=True)

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  nDf, nDs = nzd(Sf - S), nzd(Ss - S)

  # ================= Part A: generator =====================================
  smp = make_sampler(ck['params'], ck['hidden'], True, nrm)
  dlt = smp(S, K, jax.random.PRNGKey(SEED), A)          # [39, 256, 29]
  cand = S[:, None] + dlt
  nd = nzd(dlt)
  Df = np.linalg.norm(nd - nDf[:, None], axis=2)        # [39, 256]
  Ds = np.linalg.norm(nd - nDs[:, None], axis=2)
  d_fatal = Df.min(axis=1)
  d_safe = Ds.min(axis=1)
  Ci = (d_fatal <= R_FATAL).astype(int)
  n_cov = int(Ci.sum())
  cov = n_cov / n
  best_i = Df.argmin(axis=1)
  best_cand = cand[np.arange(n), best_i]

  partA = {
      'n_cases': n,
      'd_fatal': {**qstats(d_fatal), 'p10': float(np.percentile(d_fatal, 10)),
                  'p90': float(np.percentile(d_fatal, 90))},
      'd_safe': {**qstats(d_safe), 'p10': float(np.percentile(d_safe, 10)),
                 'p90': float(np.percentile(d_safe, 90))},
      'fatal_coverage_at_256': {'count': n_cov, 'n': n, 'rate': cov,
                                'ci95_clopper_pearson': cp_ci(n_cov, n)},
      'R_fatal': R_FATAL,
      'n_nonfinite_candidates': int((~np.isfinite(cand)).sum()),
      'max_abs_candidate_coord': float(np.abs(cand).max()),
  }
  print('\n== Part A: generator ==')
  print('  d_fatal  median %.3f mean %.3f p10 %.3f p90 %.3f (min %.3f max '
        '%.3f)' % (partA['d_fatal']['median'], partA['d_fatal']['mean'],
                   partA['d_fatal']['p10'], partA['d_fatal']['p90'],
                   partA['d_fatal']['min'], partA['d_fatal']['max']))
  print('  d_safe   median %.3f mean %.3f p10 %.3f p90 %.3f'
        % (partA['d_safe']['median'], partA['d_safe']['mean'],
           partA['d_safe']['p10'], partA['d_safe']['p90']))
  print('  FatalCoverage@256 = %d/%d = %.3f  CI95 [%.3f, %.3f]'
        % (n_cov, n, cov, *partA['fatal_coverage_at_256']['ci95_clopper_pearson']))

  physical = {feat: {'true_fatal': qstats([phys(x)[feat] for x in Sf]),
                     'nearest_flow_to_fatal': qstats([phys(x)[feat]
                                                      for x in best_cand]),
                     'safe_successor': qstats([phys(x)[feat] for x in Ss])}
              for feat in ('torso_z', 'v_xy', 'up_z')}
  print('  physical (median): '
        + ' | '.join('%s fatal %.3f / flow %.3f / safe %.3f'
                     % (f, physical[f]['true_fatal']['median'],
                        physical[f]['nearest_flow_to_fatal']['median'],
                        physical[f]['safe_successor']['median'])
                     for f in ('torso_z', 'v_xy')))

  # ================= Part B: Critic C selector =============================
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
  c_step, c_state = ckpt_mod.load_checkpoint(C_CKPT)
  prov['critic_step'] = int(c_step)

  @jax.jit
  def c_score(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  F = np.zeros((n, K, 2), np.float32)
  for k in range(K):
    g = np.asarray(obs_to_goal(cand[:, k].astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    og = np.concatenate([S, g], axis=1)
    F[:, k] = np.asarray(c_score(jnp.asarray(og), jnp.asarray(A)))
  Fmin = F.min(axis=2)
  kstar = Fmin.argmin(axis=1)
  wc = cand[np.arange(n), kstar]
  d_wc_fatal = Df[np.arange(n), kstar]
  d_wc_safe = Ds[np.arange(n), kstar]
  Si = (d_wc_fatal <= R_FATAL).astype(int)

  sel_num = int((Ci * Si).sum())
  sel_rate = sel_num / n_cov if n_cov else float('nan')
  e2e_num = int(Si.sum())
  e2e_rate = e2e_num / n

  def score_of(states):
    g = np.asarray(obs_to_goal(states.astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    og = np.concatenate([S, g], axis=1)
    return np.asarray(c_score(jnp.asarray(og), jnp.asarray(A))).min(axis=1)
  f_safe = score_of(Ss)
  f_fatal = score_of(Sf)
  f_wc = Fmin[np.arange(n), kstar]

  closer_to_fatal = int((np.abs(f_wc - f_fatal)
                         < np.abs(f_wc - f_safe)).sum())
  more_extreme = int((f_wc < f_fatal).sum())
  partB = {
      'selection_rate_given_coverage': {
          'numerator': sel_num, 'denominator': n_cov, 'rate': sel_rate,
          'ci95_clopper_pearson': cp_ci(sel_num, n_cov) if n_cov else None},
      'end_to_end_rate': {'numerator': e2e_num, 'denominator': n,
                          'rate': e2e_rate,
                          'ci95_clopper_pearson': cp_ci(e2e_num, n)},
      'critic_scores_f_min': {
          'safe_successor': qstats(f_safe), 'true_fatal': qstats(f_fatal),
          'selected_wc': qstats(f_wc)},
      'selected_score_position': {
          'closer_to_true_fatal': closer_to_fatal,
          'closer_to_safe': n - closer_to_fatal,
          'more_extreme_than_true_fatal': more_extreme},
  }
  print('\n== Part B: selector ==')
  print('  SelectionRate | coverage = %d/%d = %.3f  CI95 [%.3f, %.3f]'
        % (sel_num, n_cov, sel_rate,
           *(partB['selection_rate_given_coverage']['ci95_clopper_pearson']
             or (float('nan'),) * 2)))
  print('  EndToEndRate = %d/%d = %.3f  CI95 [%.3f, %.3f]'
        % (e2e_num, n, e2e_rate,
           *partB['end_to_end_rate']['ci95_clopper_pearson']))
  print('  f_min medians: safe %.2f | true fatal %.2f | selected %.2f'
        % (np.median(f_safe), np.median(f_fatal), np.median(f_wc)))
  print('  selected score closer to fatal in %d/%d; more extreme than true '
        'fatal in %d' % (closer_to_fatal, n, more_extreme))

  # ---- off-manifold / argmin pathology ------------------------------------
  patho = (Si == 0)                       # selected but not fatal-like
  patho_idx = np.where(patho)[0]
  path_rows = []
  for i in patho_idx:
    path_rows.append({
        'case': int(i), 'episode_id': int(eps[i]),
        'f_wc': float(f_wc[i]), 'f_true_fatal': float(f_fatal[i]),
        'f_safe': float(f_safe[i]),
        'score_below_true_fatal': bool(f_wc[i] < f_fatal[i]),
        'd_wc_to_fatal': float(d_wc_fatal[i]),
        'd_wc_to_safe': float(d_wc_safe[i]),
        'delta_norm': float(np.linalg.norm(wc[i] - S[i])),
        **{('wc_' + k_): v for k_, v in phys(wc[i]).items()}})
  pathology = {
      'definition': ('selected candidate is NOT within R_fatal of the true '
                     'settled fatal successor'),
      'n_pathological': int(patho.sum()),
      'n_pathological_with_score_below_true_fatal':
          int((patho & (f_wc < f_fatal)).sum()),
      'cases': path_rows,
      'selected_delta_norm': qstats(np.linalg.norm(wc - S, axis=1)),
      'true_fatal_delta_norm': qstats(np.linalg.norm(Sf - S, axis=1)),
      'selected_max_abs_coord': float(np.abs(wc).max()),
      'real_fatal_max_abs_coord': float(np.abs(Sf).max()),
  }
  print('  argmin pathology: %d/%d selected candidates are not fatal-like '
        '(%d of those score below the TRUE fatal state)'
        % (pathology['n_pathological'], n,
           pathology['n_pathological_with_score_below_true_fatal']))

  # ================= Part C: dev16 comparison ==============================
  dv = json.load(open(DEV16_SUMMARY))['models']['V2-SA-l001']
  comparison = {
      'fatal_d_at_256': {'dev16': dv['d_fatal']['256']['median'],
                         'final39': partA['d_fatal']['median']},
      'fatal_coverage_at_256': {'dev16': dv['fatal_coverage']['256'],
                                'final39': cov},
      'safe_d_at_256': {'dev16': dv['d_safe']['256']['median'],
                        'final39': partA['d_safe']['median']},
      'selection_rate_given_coverage': {'dev16': None, 'final39': sel_rate},
      'end_to_end_rate': {'dev16': None, 'final39': e2e_rate},
      'note': ('dev16 numbers are medians over 8 sampling seeds x 16 '
               'anchors; final39 is a single frozen-seed draw over 39 '
               'sealed anchors')}
  print('\n== Part C: dev16 vs final39 ==')
  print('  fatal d@256      dev16 %.3f -> final39 %.3f'
        % (comparison['fatal_d_at_256']['dev16'],
           comparison['fatal_d_at_256']['final39']))
  print('  FatalCoverage    dev16 %.3f -> final39 %.3f'
        % (comparison['fatal_coverage_at_256']['dev16'],
           comparison['fatal_coverage_at_256']['final39']))
  print('  safe d@256       dev16 %.3f -> final39 %.3f'
        % (comparison['safe_d_at_256']['dev16'],
           comparison['safe_d_at_256']['final39']))

  # ================= artifacts =============================================
  summary = {'stage': 'SEALED final39 one-shot evaluation',
             'provenance': prov, 'part_A_generator': partA,
             'physical': physical, 'part_B_selector': partB,
             'argmin_pathology': pathology, 'part_C_comparison': comparison}
  json.dump(summary, open(os.path.join(args.out, 'final39_summary.json'),
                          'w'), indent=2)
  np.savez_compressed(
      os.path.join(args.out, 'candidates_and_scores.npz'),
      anchor=S, action=A, true_fatal=Sf, safe_successor=Ss,
      candidates=cand, critic_twin=F, critic_fmin=Fmin,
      d_fatal_all=Df, d_safe_all=Ds, argmin_index=kstar,
      selected_wc=wc, nearest_to_fatal=best_cand, episode_id=eps,
      f_safe=f_safe, f_fatal=f_fatal, f_wc=f_wc,
      meta=json.dumps({'K': K, 'seed': SEED, 'R_fatal': R_FATAL}))
  with open(os.path.join(args.out, 'per_case.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'd_fatal', 'covered', 'd_safe',
                'd_wc_to_fatal', 'selected_fatal_like', 'f_safe', 'f_fatal',
                'f_wc', 'wc_z', 'wc_vxy', 'fatal_z', 'fatal_vxy'])
    for i in range(n):
      w.writerow([i, int(eps[i]), round(float(d_fatal[i]), 4), int(Ci[i]),
                  round(float(d_safe[i]), 4), round(float(d_wc_fatal[i]), 4),
                  int(Si[i]), round(float(f_safe[i]), 3),
                  round(float(f_fatal[i]), 3), round(float(f_wc[i]), 3),
                  round(phys(wc[i])['torso_z'], 4),
                  round(phys(wc[i])['v_xy'], 4),
                  round(phys(Sf[i])['torso_z'], 4),
                  round(phys(Sf[i])['v_xy'], 4)])

  fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
  ax[0].hist(d_fatal, bins=20, color='crimson', alpha=0.8)
  ax[0].axvline(R_FATAL, color='green', ls=':', lw=2, label='R_fatal 3.17')
  ax[0].set_xlabel('d_fatal (min over 256)')
  ax[0].set_ylabel('cases')
  ax[0].set_title('generator: %d/%d covered' % (n_cov, n))
  ax[0].legend(fontsize=8)
  ax[1].scatter([phys(x)['v_xy'] for x in Ss], [phys(x)['torso_z'] for x in Ss],
                s=45, marker='^', color='tab:blue', label='safe successor')
  ax[1].scatter([phys(x)['v_xy'] for x in wc], [phys(x)['torso_z'] for x in wc],
                s=45, marker='o', facecolors='none', edgecolors='darkorange',
                linewidths=1.5, label='selected worst-case')
  ax[1].scatter([phys(x)['v_xy'] for x in Sf], [phys(x)['torso_z'] for x in Sf],
                s=60, marker='X', color='crimson', label='true settled fatal')
  ax[1].set_xlabel('|v_xy|')
  ax[1].set_ylabel('torso z')
  ax[1].set_title('physical placement of the selected candidate')
  ax[1].legend(fontsize=8)
  ax[2].scatter(f_fatal, f_wc, s=45, c=np.where(Si == 1, 'tab:green', 'tab:red'))
  lim = [min(f_fatal.min(), f_wc.min()) - 1, max(f_fatal.max(), f_wc.max()) + 1]
  ax[2].plot(lim, lim, 'k--', lw=1)
  ax[2].set_xlabel('f_C(true fatal)')
  ax[2].set_ylabel('f_C(selected wc)')
  ax[2].set_title('selector: green = fatal-like selected')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'final39_diagnostics.png'), dpi=140)
  plt.close(fig)
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
