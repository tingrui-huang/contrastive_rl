"""ONE-SHOT sealed evaluation of frozen V3 + frozen Critic C on fresh50.

Two modes:
  --seal   Phase 3: verify provenance and write fresh50_freeze.json +
           provenance_manifest.json BEFORE any model touches the set. Records
           explicitly that no model result has been computed yet.
  (default) Phases 4-11: the single frozen evaluation. Refuses to run unless
           the freeze artifact exists and its recorded pair sha still matches.

Everything is frozen: V3-SA-diverse-l001, lambda=0.01, K=256, R_fatal=3.17,
50-step Euler, the frozen V0 state/delta normalization, raw-action
conditioning, Critic C with twin f_min, and the single frozen sampling seed.
No alternative K, seed, model or radius is examined after opening the set.

Usage:
  python scripts/eval_fresh50_sealed.py --seal
  python scripts/eval_fresh50_sealed.py
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

OUT = 'artifacts/flow_v3_fresh50'
PAIRS = os.path.join(OUT, 'fresh50_pairs.npz')
FREEZE = os.path.join(OUT, 'fresh50_freeze.json')
V3_CKPT = 'artifacts/flow_v3_diverse_failure/flow_v3/flow_v3.pkl'
V3_PROV = 'artifacts/flow_v3_diverse_failure/flow_v3/provenance.json'
POOL = ('artifacts/flow_v3_diverse_failure/failure_pool_diversity_audit/'
        'failure_pool_diverse.npz')
V0_DIR = 'artifacts/flow_v0_clean'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
V3_DEV = 'artifacts/flow_v3_diverse_failure/dev16_eval/v3_summary.json'
K, ODE_STEPS, LAMBDA, SEED = 256, 50, 0.01, 11
NEAR_MISS = 4.0     # only used to LABEL near vs real selector misses


def cp(k, n, a=0.05):
  lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
  hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
  return [lo, hi]


def build_prov():
  with open(V3_CKPT, 'rb') as f:
    ck = pickle.load(f)
  assert ck['family'] == 'SA' and abs(ck['lam'] - LAMBDA) < 1e-12
  assert ck.get('run_id') == 'V3-SA-diverse-l001'
  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  for k_ in nrm:
    assert np.array_equal(nrm[k_], np.asarray(ck['norm'][k_], np.float32)), \
        'normalization drifted'
  v3p = json.load(open(V3_PROV))
  assert v3p['n_combined'] == 603, 'diverse pool is not 603'
  assert C.sha256_file(POOL) == v3p['diverse_pool_sha256'], 'pool drifted'
  split = json.load(open(os.path.join(V0_DIR, 'split_manifest.json')))
  assert C.sha256_file(split['npz']) == split['npz_sha256'], 'clean drifted'
  return ck, nrm, {
      'generator_run_id': ck['run_id'],
      'generator_ckpt': V3_CKPT,
      'generator_sha256': C.sha256_file(V3_CKPT),
      'generator_code_commit': v3p['git_commit'],
      'lambda': LAMBDA, 'K': K, 'R_fatal': R_FATAL,
      'ode_steps': ODE_STEPS, 'solver': 'fixed-step explicit Euler',
      'sampling_seed': SEED,
      'normalization_source': os.path.join(V0_DIR, 'norm_stats.npz'),
      'action_conditioning': 'raw bounded actions [-1, 1]',
      'diverse_pool': POOL, 'diverse_pool_sha256': v3p['diverse_pool_sha256'],
      'n_failure_pool': v3p['n_combined'],
      'clean_npz_sha256': split['npz_sha256'],
      'critic_ckpt': C_CKPT, 'critic_sha256': C.sha256_file(C_CKPT),
      'critic_aggregation': 'twin f_min (frozen)',
      'pair_protocol': 'scripts/build_fresh50_pairs.py (validated protocol, '
                       'fresh seeds only)',
      'current_commit': C.git_commit()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seal', action='store_true')
  args = ap.parse_args()
  ck, nrm, prov = build_prov()
  pm = json.load(open(os.path.join(OUT, 'pair_generation_manifest.json')))
  pairs_sha = C.sha256_file(PAIRS)
  assert pairs_sha == pm['pairs_sha256'], 'fresh50 pair artifact drifted'

  # ---------------- Phase 3: seal --------------------------------------------
  if args.seal:
    assert not os.path.exists(FREEZE), 'fresh50 is already sealed'
    fr = {'SEALED': True,
          'statement': ('V3 checkpoint already frozen; lambda = 0.01; '
                        'K = 256; R_fatal = 3.17; Critic C checkpoint '
                        'frozen; NO MODEL RESULT HAS YET BEEN COMPUTED ON '
                        'FRESH50'),
          'pairs_npz': PAIRS, 'pairs_sha256': pairs_sha,
          'n_pairs': int(pm['n_pairs']),
          'pair_episode_ids': [a['episode'] for a in pm['attempts']
                               if a.get('status') == 'ACCEPTED'],
          'env_seed': pm['env_seed'], 'dataset_seed': pm['dataset_seed'],
          'generator_provenance': prov,
          'disjointness': ('fresh env/dataset seeds, checked against every '
                           'consumed seed: pilot, 40-death stream, bad-demo, '
                           'all four V3 arms; no fresh50 episode enters any '
                           'training pool'),
          'git_commit': C.git_commit()}
    json.dump(fr, open(FREEZE, 'w'), indent=2)
    json.dump(prov, open(os.path.join(OUT, 'provenance_manifest.json'), 'w'),
              indent=2)
    print('SEALED fresh50: %d pairs, sha %s' % (pm['n_pairs'], pairs_sha))
    print('freeze -> %s' % FREEZE)
    return

  # ---------------- Phases 4-11: one shot ------------------------------------
  assert os.path.exists(FREEZE), 'fresh50 must be sealed before evaluation'
  fr = json.load(open(FREEZE))
  assert fr['pairs_sha256'] == pairs_sha, 'pairs changed after sealing'
  print('PROVENANCE OK | sealed sha %s' % pairs_sha[:32])
  for k_ in ('generator_run_id', 'lambda', 'K', 'R_fatal', 'sampling_seed',
             'critic_ckpt'):
    print('  %-20s %s' % (k_, prov[k_]))

  d = np.load(PAIRS, allow_pickle=True)
  S = np.asarray(d['anchor_obs'], np.float32)
  A = np.asarray(d['anchor_action'], np.float32)
  Sf = np.asarray(d['fatal_candidate'], np.float32)
  Ss = np.asarray(d['safe_candidate'], np.float32)
  eps = np.asarray(d['episode_id'], np.int64)
  n = len(S)
  print('\nopened sealed set: %d pairs' % n, flush=True)

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  nDf, nDs = nzd(Sf - S), nzd(Ss - S)

  # ---- Phase 4: generator ---------------------------------------------------
  smp = make_sampler(ck['params'], ck['hidden'], True, nrm)
  dlt = smp(S, K, jax.random.PRNGKey(SEED), A)
  cand = S[:, None] + dlt
  nd = nzd(dlt)
  Df = np.linalg.norm(nd - nDf[:, None], axis=2)
  Ds = np.linalg.norm(nd - nDs[:, None], axis=2)
  d_fatal, d_safe = Df.min(1), Ds.min(1)
  Ci = (d_fatal <= R_FATAL).astype(int)
  n_cov = int(Ci.sum())
  best = cand[np.arange(n), Df.argmin(1)]
  iu = np.triu_indices(K, k=1)
  div = np.linalg.norm(nd[:, :, None, :] - nd[:, None, :, :],
                       axis=3)[:, iu[0], iu[1]].mean(1)

  gen = {'n': n,
         'd_fatal': {**qstats(d_fatal),
                     'p10': float(np.percentile(d_fatal, 10)),
                     'p90': float(np.percentile(d_fatal, 90))},
         'fatal_coverage_at_256': {'count': n_cov, 'n': n, 'rate': n_cov / n,
                                   'ci95': cp(n_cov, n)},
         'confirmation_criterion': {
             'median_d_fatal_le_3.17': bool(np.median(d_fatal) <= R_FATAL),
             'coverage_ge_25_of_50': bool(n_cov >= 25)},
         'd_safe': {**qstats(d_safe), 'p10': float(np.percentile(d_safe, 10)),
                    'p90': float(np.percentile(d_safe, 90))},
         'candidate_diversity': qstats(div),
         'max_abs_generated_delta': float(np.abs(dlt).max()),
         'candidate_coord_range': [float(cand.min()), float(cand.max())],
         'real_fatal_coord_range': [float(Sf.min()), float(Sf.max())],
         'n_nonfinite': int((~np.isfinite(cand)).sum())}
  print('\n== Phase 4-5: generator ==')
  print('  d_fatal median %.3f mean %.3f p10 %.3f p90 %.3f (min %.3f max %.3f)'
        % (gen['d_fatal']['median'], gen['d_fatal']['mean'],
           gen['d_fatal']['p10'], gen['d_fatal']['p90'],
           gen['d_fatal']['min'], gen['d_fatal']['max']))
  print('  FatalCoverage@256 = %d/%d = %.3f CI95 [%.3f, %.3f]'
        % (n_cov, n, n_cov / n, *gen['fatal_coverage_at_256']['ci95']))
  print('  confirmation: median<=3.17 %s | cov>=25/50 %s'
        % (gen['confirmation_criterion']['median_d_fatal_le_3.17'],
           gen['confirmation_criterion']['coverage_ge_25_of_50']))
  print('  d_safe median %.3f | diversity %.3f | max|delta| %.2f | '
        'nonfinite %d' % (gen['d_safe']['median'],
                          gen['candidate_diversity']['median'],
                          gen['max_abs_generated_delta'], gen['n_nonfinite']))

  # ---- Phase 6: physical ----------------------------------------------------
  physical = {f: {'true_fatal': qstats([phys(x)[f] for x in Sf]),
                  'nearest_flow_to_fatal': qstats([phys(x)[f] for x in best]),
                  'safe_successor': qstats([phys(x)[f] for x in Ss])}
              for f in ('torso_z', 'v_xy', 'up_z')}
  print('  physical medians: '
        + ' | '.join('%s F %.3f / flow %.3f / S %.3f'
                     % (f, physical[f]['true_fatal']['median'],
                        physical[f]['nearest_flow_to_fatal']['median'],
                        physical[f]['safe_successor']['median'])
                     for f in ('torso_z', 'v_xy')))

  # ---- Phase 7-8: frozen Critic C -------------------------------------------
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

  @jax.jit
  def c_score(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  F = np.zeros((n, K, 2), np.float32)
  for k in range(K):
    g = np.asarray(obs_to_goal(cand[:, k].astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    F[:, k] = np.asarray(c_score(jnp.asarray(np.concatenate([S, g], 1)),
                                 jnp.asarray(A)))
  Fmin = F.min(2)
  kstar = Fmin.argmin(1)
  wc = cand[np.arange(n), kstar]
  d_wc_f, d_wc_s = Df[np.arange(n), kstar], Ds[np.arange(n), kstar]
  Si = (d_wc_f <= R_FATAL).astype(int)

  def score_of(x):
    g = np.asarray(obs_to_goal(x.astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    return np.asarray(c_score(jnp.asarray(np.concatenate([S, g], 1)),
                              jnp.asarray(A))).min(1)
  f_safe, f_fatal = score_of(Ss), score_of(Sf)
  f_wc = Fmin[np.arange(n), kstar]
  sel_num = int((Ci * Si).sum())
  e2e = int(Si.sum())
  sel = {'selection_rate_given_coverage': {
             'numerator': sel_num, 'denominator': n_cov,
             'rate': sel_num / n_cov if n_cov else float('nan'),
             'ci95': cp(sel_num, n_cov) if n_cov else None},
         'end_to_end_rate': {'numerator': e2e, 'denominator': n,
                             'rate': e2e / n, 'ci95': cp(e2e, n)},
         'critic_f_min': {'safe': qstats(f_safe), 'true_fatal':
                          qstats(f_fatal), 'selected': qstats(f_wc)}}
  print('\n== Phase 7-8: selector ==')
  print('  SelectionRate | coverage = %d/%d = %.3f CI95 [%.3f, %.3f]'
        % (sel_num, n_cov, sel['selection_rate_given_coverage']['rate'],
           *(sel['selection_rate_given_coverage']['ci95'] or (0, 0))))
  print('  EndToEndRate = %d/%d = %.3f CI95 [%.3f, %.3f]'
        % (e2e, n, e2e / n, *sel['end_to_end_rate']['ci95']))
  print('  f_min medians: safe %.2f | true fatal %.2f | selected %.2f'
        % (np.median(f_safe), np.median(f_fatal), np.median(f_wc)))

  # ---- Phase 9: miss taxonomy ----------------------------------------------
  cats = []
  for i in range(n):
    if not Ci[i]:
      cats.append('G0_generator_miss')
    elif Si[i]:
      cats.append('G1S1_selector_success')
    elif d_wc_f[i] <= NEAR_MISS:
      cats.append('G1S0_near_miss')
    else:
      cats.append('G1S0_real_miss')
  cats = np.array(cats)
  taxonomy = {c: int((cats == c).sum()) for c in
              ('G0_generator_miss', 'G1S1_selector_success',
               'G1S0_near_miss', 'G1S0_real_miss')}
  taxonomy['near_miss_label_threshold'] = NEAR_MISS
  taxonomy['note'] = ('near/real split is a LABEL only; R_fatal stays 3.17 '
                      'and no threshold was redefined')
  print('  taxonomy: %s' % {k: v for k, v in taxonomy.items()
                            if k.startswith('G')})

  # ---- Phase 10: argmin pathology ------------------------------------------
  patho = (Si == 0) & (f_wc < f_fatal)
  pathology = {
      'definition': ('selected candidate is not fatal-like AND scores below '
                     'the TRUE fatal successor'),
      'n_pathological': int(patho.sum()), 'n': n,
      'n_selected_not_fatal_like': int((Si == 0).sum()),
      'n_score_below_true_fatal': int((f_wc < f_fatal).sum()),
      'selected_delta_norm': qstats(np.linalg.norm(wc - S, axis=1)),
      'true_fatal_delta_norm': qstats(np.linalg.norm(Sf - S, axis=1)),
      'selected_coord_range': [float(wc.min()), float(wc.max())],
      'real_fatal_coord_range': [float(Sf.min()), float(Sf.max())],
      'selected_physical': {f: qstats([phys(x)[f] for x in wc])
                            for f in ('torso_z', 'v_xy', 'up_z')}}
  print('  argmin pathology: %d/%d' % (pathology['n_pathological'], n))

  # ---- Phase 11: dev vs sealed ---------------------------------------------
  dv = json.load(open(V3_DEV))
  comparison = {
      'dev16': {'fatal_d': dv['phase8A_dev16']['V3-SA-diverse']['d_fatal']
                ['median'],
                'coverage': dv['phase8A_dev16']['V3-SA-diverse']['coverage'],
                'safe_d': dv['phase8A_dev16']['V3-SA-diverse']['d_safe']
                ['median']},
      'reused_final39_DEVELOPMENT_DIAGNOSTIC': {
          'fatal_d': dv['phase8B_reused_final39']['V3-SA-diverse']['d_fatal']
          ['median'],
          'coverage': dv['phase8B_reused_final39']['V3-SA-diverse']
          ['coverage'],
          'safe_d': dv['phase8B_reused_final39']['V3-SA-diverse']['d_safe']
          ['median']},
      'fresh50_SEALED': {'fatal_d': gen['d_fatal']['median'],
                         'coverage': n_cov / n,
                         'safe_d': gen['d_safe']['median'],
                         'selection_rate':
                             sel['selection_rate_given_coverage']['rate'],
                         'end_to_end': e2e / n}}

  summary = {'stage': 'FRESH50 one-shot sealed evaluation',
             'provenance': prov, 'freeze': fr['statement'],
             'generator': gen, 'physical': physical, 'selector': sel,
             'miss_taxonomy': taxonomy, 'pathology': pathology,
             'comparison': comparison}
  json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
  np.savez_compressed(os.path.join(OUT, 'candidates.npz'),
                      anchor=S, action=A, true_fatal=Sf, safe=Ss,
                      candidates=cand, critic_twin=F, critic_fmin=Fmin,
                      d_fatal_all=Df, d_safe_all=Ds, argmin_index=kstar,
                      selected=wc, nearest_to_fatal=best, episode_id=eps,
                      f_safe=f_safe, f_fatal=f_fatal, f_wc=f_wc)
  with open(os.path.join(OUT, 'per_case.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'd_fatal', 'covered', 'd_safe',
                'd_wc_fatal', 'selected_fatal_like', 'category',
                'f_safe', 'f_fatal', 'f_wc', 'wc_z', 'wc_vxy',
                'fatal_z', 'fatal_vxy'])
    for i in range(n):
      w.writerow([i, int(eps[i]), round(float(d_fatal[i]), 4), int(Ci[i]),
                  round(float(d_safe[i]), 4), round(float(d_wc_f[i]), 4),
                  int(Si[i]), cats[i], round(float(f_safe[i]), 3),
                  round(float(f_fatal[i]), 3), round(float(f_wc[i]), 3),
                  round(phys(wc[i])['torso_z'], 4),
                  round(phys(wc[i])['v_xy'], 4),
                  round(phys(Sf[i])['torso_z'], 4),
                  round(phys(Sf[i])['v_xy'], 4)])
  with open(os.path.join(OUT, 'selector_misses.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'category', 'best_available_d_fatal',
                'd_wc_fatal', 'f_wc', 'f_fatal', 'score_below_true_fatal'])
    for i in np.where(cats != 'G1S1_selector_success')[0]:
      w.writerow([int(i), int(eps[i]), cats[i], round(float(d_fatal[i]), 4),
                  round(float(d_wc_f[i]), 4), round(float(f_wc[i]), 3),
                  round(float(f_fatal[i]), 3), bool(f_wc[i] < f_fatal[i])])
  json.dump(pathology, open(os.path.join(OUT, 'pathology_report.json'), 'w'),
            indent=2)
  json.dump(physical, open(os.path.join(OUT,
                                        'physical_diagnostics.json'), 'w'),
            indent=2)

  fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
  ax[0].hist(d_fatal, bins=20, color='crimson', alpha=0.85)
  ax[0].axvline(R_FATAL, color='green', ls=':', lw=2, label='R_fatal 3.17')
  ax[0].set_xlabel('d_fatal@256')
  ax[0].set_ylabel('cases')
  ax[0].set_title('sealed generator: %d/%d covered' % (n_cov, n))
  ax[0].legend(fontsize=8)
  ax[1].scatter([phys(x)['v_xy'] for x in Ss], [phys(x)['torso_z'] for x in Ss],
                s=40, marker='^', color='tab:blue', label='safe')
  ax[1].scatter([phys(x)['v_xy'] for x in wc], [phys(x)['torso_z'] for x in wc],
                s=40, marker='o', facecolors='none', edgecolors='darkorange',
                label='selected wc')
  ax[1].scatter([phys(x)['v_xy'] for x in Sf], [phys(x)['torso_z'] for x in Sf],
                s=55, marker='X', color='crimson', label='true fatal')
  ax[1].set_xlabel('|v_xy|')
  ax[1].set_ylabel('torso z')
  ax[1].set_title('physical placement')
  ax[1].legend(fontsize=8)
  ax[2].scatter(f_fatal, f_wc, s=40,
                c=np.where(Si == 1, 'tab:green', 'tab:red'))
  lim = [min(f_fatal.min(), f_wc.min()) - 1, max(f_fatal.max(), f_wc.max()) + 1]
  ax[2].plot(lim, lim, 'k--', lw=1)
  ax[2].set_xlabel('f_C(true fatal)')
  ax[2].set_ylabel('f_C(selected)')
  ax[2].set_title('green = fatal-like selected')
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'physical_diagnostics.png'), dpi=140)
  plt.close(fig)

  print('\n== Phase 11 ==')
  for k_, v in comparison.items():
    print('  %-42s fatal_d %.3f cov %.3f safe_d %.3f'
          % (k_, v['fatal_d'], v['coverage'], v['safe_d']))
  print('\nsaved -> %s' % OUT)


if __name__ == '__main__':
  main()
