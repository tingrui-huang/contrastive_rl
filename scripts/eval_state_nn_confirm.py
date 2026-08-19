"""Stages 3-10: one-shot sealed confirmation of the FROZEN state-NN selector.

  --seal   Stage 3: verify the selector freeze + pair artifact and write
           selector_confirm50_freeze.json asserting that the selector rule was
           frozen first, no Flow candidate has been generated for these cases,
           and no selector result exists yet.
  (default) Stages 4-10: generate K=256 candidates per anchor from the frozen
           V3, apply the FROZEN state-NN rule, and score. The pre-declared
           argmin f_C comparator (defined long before this set) is evaluated
           on the SAME candidates as a secondary readout only.

Nothing is retrained, resampled, reseeded, re-thresholded or combined.

Usage:
  python scripts/eval_state_nn_confirm.py --seal
  python scripts/eval_state_nn_confirm.py
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

OUT = 'artifacts/state_nn_selector_confirm'
PAIRS = os.path.join(OUT, 'selector_confirm50_pairs.npz')
FREEZE_SEL = os.path.join(OUT, 'selector_freeze.json')
FREEZE_SET = os.path.join(OUT, 'selector_confirm50_freeze.json')
V0_DIR = 'artifacts/flow_v0_clean'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
K, SEED, NEAR = 256, 11, 4.0


def cp(k, n, a=0.05):
  lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
  hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
  return [lo, hi]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seal', action='store_true')
  args = ap.parse_args()

  fs = json.load(open(FREEZE_SEL))
  assert fs['SELECTOR_FROZEN'] and fs['R_fatal'] == R_FATAL
  assert C.sha256_file(fs['generator']['ckpt']) == fs['generator']['sha256']
  assert C.sha256_file(fs['negative_bank']['path']) == \
      fs['negative_bank']['sha256'], 'negative bank drifted'
  assert C.sha256_file(fs['normalization']['path']) == \
      fs['normalization']['sha256'], 'normalization drifted'
  pm = json.load(open(os.path.join(OUT,
                                   'pair_generation_manifest.json')))
  pairs_sha = C.sha256_file(PAIRS)
  assert pairs_sha == pm['pairs_sha256'], 'pair artifact drifted'

  if args.seal:
    assert not os.path.exists(FREEZE_SET), 'already sealed'
    fr = {'SEALED': True,
          'assertions': [
              'the state-NN selector rule was frozen BEFORE this test set '
              'was built or evaluated (see selector_freeze.json)',
              'no Flow candidate has yet been generated for these cases',
              'no selector result exists yet'],
          'pairs_npz': PAIRS, 'pairs_sha256': pairs_sha,
          'n_pairs': int(pm['n_pairs']),
          'env_seed': pm['env_seed'], 'dataset_seed': pm['dataset_seed'],
          'disjointness': ('fresh seeds, programmatically excluded against '
                           'every consumed seed incl. fresh50, the 40-death '
                           'stream, bad-demo and all four V3 arms'),
          'selector_freeze_sha_ref': fs['negative_bank']['sha256'],
          'git_commit': C.git_commit()}
    json.dump(fr, open(FREEZE_SET, 'w'), indent=2)
    print('SEALED selector-confirm50: %d pairs, sha %s'
          % (pm['n_pairs'], pairs_sha))
    return

  assert os.path.exists(FREEZE_SET), 'set must be sealed before evaluation'
  fr = json.load(open(FREEZE_SET))
  assert fr['pairs_sha256'] == pairs_sha, 'pairs changed after sealing'
  print('PROVENANCE OK | selector frozen at %s | set sha %s'
        % (fs['git_commit'][:8], pairs_sha[:24]))

  d = np.load(PAIRS, allow_pickle=True)
  S = np.asarray(d['anchor_obs'], np.float32)
  A = np.asarray(d['anchor_action'], np.float32)
  Sf = np.asarray(d['fatal_candidate'], np.float32)
  Ss = np.asarray(d['safe_candidate'], np.float32)
  eps = np.asarray(d['episode_id'], np.int64)
  n = len(S)

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  nDf, nDs = nzd(Sf - S), nzd(Ss - S)

  # ---- Stage 4: frozen generation -----------------------------------------
  with open(fs['generator']['ckpt'], 'rb') as f:
    ck = pickle.load(f)
  smp = make_sampler(ck['params'], ck['hidden'], True, nrm)
  dlt = smp(S, K, jax.random.PRNGKey(SEED), A)
  cand = S[:, None] + dlt
  nd = nzd(dlt)
  Df = np.linalg.norm(nd - nDf[:, None], axis=2)
  Ds = np.linalg.norm(nd - nDs[:, None], axis=2)
  np.savez_compressed(os.path.join(OUT, 'candidates.npz'),
                      anchor=S, action=A, true_fatal=Sf, safe=Ss,
                      candidates=cand, d_fatal_all=Df, d_safe_all=Ds,
                      episode_id=eps)

  # ---- Stage 5: generator coverage ----------------------------------------
  d_best, d_safe = Df.min(1), Ds.min(1)
  Ci = (d_best <= R_FATAL).astype(int)
  n_cov = int(Ci.sum())
  gen = {'coverage_at_256': {'count': n_cov, 'n': n, 'rate': n_cov / n,
                             'ci95': cp(n_cov, n)},
         'd_fatal': qstats(d_best), 'd_safe': qstats(d_safe),
         'n_nonfinite': int((~np.isfinite(cand)).sum())}
  print('\n[S5] GeneratorCoverage@256 = %d/%d = %.3f | median d_fatal %.3f '
        '| d_safe %.3f' % (n_cov, n, n_cov / n, gen['d_fatal']['median'],
                           gen['d_safe']['median']))

  # ---- Stage 6: FROZEN state-NN selector ----------------------------------
  bank = np.asarray(np.load(fs['negative_bank']['path'],
                            allow_pickle=True)['goals'], np.float32)
  assert bank.shape == (16, OBS_DIM)
  bn = (bank - nrm['state_mean']) / nrm['state_std']
  flat = (cand.reshape(-1, OBS_DIM) - nrm['state_mean']) / nrm['state_std']
  dneg = np.empty(len(flat), np.float32)
  for i0 in range(0, len(flat), 4096):
    dneg[i0:i0 + 4096] = np.linalg.norm(
        flat[i0:i0 + 4096][:, None] - bn[None], axis=2).min(1)
  dneg = dneg.reshape(n, K)
  k_nn = dneg.argmin(1)                       # lowest-index tie-break
  d_nn = Df[np.arange(n), k_nn]
  S_nn = (d_nn <= R_FATAL).astype(int)

  # ---- Stage 7: pre-declared comparator ------------------------------------
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
  _, c_state = ckpt_mod.load_checkpoint(C_CKPT)

  @jax.jit
  def c_score(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  Fmin = np.zeros((n, K), np.float32)
  for k in range(K):
    g = np.asarray(obs_to_goal(cand[:, k].astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    Fmin[:, k] = np.asarray(
        c_score(jnp.asarray(np.concatenate([S, g], 1)),
                jnp.asarray(A))).min(1)
  k_c = Fmin.argmin(1)
  d_c = Df[np.arange(n), k_c]
  S_c = (d_c <= R_FATAL).astype(int)

  def score_of(x):
    g = np.asarray(obs_to_goal(x.astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    return np.asarray(c_score(jnp.asarray(np.concatenate([S, g], 1)),
                              jnp.asarray(A))).min(1)
  f_fatal = score_of(Sf)

  def block(k_sel, d_sel, S_sel):
    num = int((Ci * S_sel).sum())
    e2e = int(S_sel.sum())
    f_sel = Fmin[np.arange(n), k_sel]
    patho = (d_sel > R_FATAL) & (f_sel < f_fatal)
    sel = cand[np.arange(n), k_sel]
    return {'selection_rate_given_coverage': {
                'count': '%d/%d' % (num, n_cov),
                'rate': num / n_cov if n_cov else float('nan'),
                'ci95': cp(num, n_cov) if n_cov else None},
            'end_to_end': {'count': '%d/%d' % (e2e, n), 'rate': e2e / n,
                           'ci95': cp(e2e, n)},
            'n_pathological': int(patho.sum()),
            'pathological_cases': np.where(patho)[0].tolist(),
            'physical': {f: qstats([phys(x)[f] for x in sel])
                         for f in ('torso_z', 'v_xy', 'up_z')},
            'delta_norm': qstats(np.linalg.norm(sel - S, axis=1)),
            'coord_range': [float(sel.min()), float(sel.max())],
            '_S': S_sel, '_d': d_sel}
  b_nn, b_c = block(k_nn, d_nn, S_nn), block(k_c, d_c, S_c)

  paired = {'state_nn_wins': int(((S_nn == 1) & (S_c == 0)).sum()),
            'critic_wins': int(((S_nn == 0) & (S_c == 1)).sum()),
            'both_succeed': int(((S_nn == 1) & (S_c == 1)).sum()),
            'both_fail': int(((S_nn == 0) & (S_c == 0)).sum())}

  # ---- Stage 9: taxonomy ---------------------------------------------------
  cats = np.array(['G0_generator_miss' if not Ci[i] else
                   'G1S1_success' if S_nn[i] else
                   'G1S0_near' if d_nn[i] <= NEAR else 'G1S0_real'
                   for i in range(n)])
  tax = {c: int((cats == c).sum()) for c in
         ('G0_generator_miss', 'G1S1_success', 'G1S0_near', 'G1S0_real')}
  tax['pathological'] = b_nn['n_pathological']
  tax['near_label_threshold'] = NEAR
  tax['note'] = 'R_fatal stays 3.17; NEAR is a label only'

  summary = {
      'status': 'ONE-SHOT SEALED CONFIRMATION of the frozen state-NN selector',
      'selector_freeze': FREEZE_SEL, 'set_freeze': FREEZE_SET,
      'pairs_sha256': pairs_sha,
      'generator': gen,
      'state_nn': {k: v for k, v in b_nn.items() if not k.startswith('_')},
      'argmin_critic_comparator': {k: v for k, v in b_c.items()
                                   if not k.startswith('_')},
      'paired_comparison': paired, 'miss_taxonomy': tax,
      'reference_physical': {
          f: {'true_fatal': qstats([phys(x)[f] for x in Sf]),
              'safe_successor': qstats([phys(x)[f] for x in Ss])}
          for f in ('torso_z', 'v_xy', 'up_z')},
      'true_fatal_coord_range': [float(Sf.min()), float(Sf.max())],
      'development_reference': {'fresh50_state_nn': '26/39 = 0.667',
                                'fresh50_argmin_fC': '16/39 = 0.410',
                                'note': 'fresh50 is development data'},
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
  json.dump({'rule': fs['pathology_rule'],
             'state_nn': {'n': b_nn['n_pathological'],
                          'cases': b_nn['pathological_cases']},
             'argmin_critic': {'n': b_c['n_pathological'],
                               'cases': b_c['pathological_cases']},
             'state_nn_coord_range': b_nn['coord_range'],
             'argmin_coord_range': b_c['coord_range'],
             'true_fatal_coord_range': summary['true_fatal_coord_range']},
            open(os.path.join(OUT, 'pathology_report.json'), 'w'), indent=2)
  json.dump(summary['reference_physical'] |
            {'state_nn': b_nn['physical'], 'argmin_critic': b_c['physical']},
            open(os.path.join(OUT, 'physical_diagnostics.json'), 'w'),
            indent=2)
  with open(os.path.join(OUT, 'per_case.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'covered', 'best_d_fatal',
                'nn_k', 'nn_d_fatal', 'nn_success', 'critic_k',
                'critic_d_fatal', 'critic_success', 'category'])
    for i in range(n):
      w.writerow([i, int(eps[i]), int(Ci[i]), round(float(d_best[i]), 4),
                  int(k_nn[i]), round(float(d_nn[i]), 4), int(S_nn[i]),
                  int(k_c[i]), round(float(d_c[i]), 4), int(S_c[i]),
                  cats[i]])
  with open(os.path.join(OUT, 'selector_comparison.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['selector', 'sel_given_coverage', 'rate', 'ci_lo', 'ci_hi',
                'end_to_end', 'e2e_rate', 'n_pathological'])
    for nm, b in (('frozen state-NN similarity', b_nn),
                  ('frozen argmin Critic C', b_c)):
      s_ = b['selection_rate_given_coverage']
      w.writerow([nm, s_['count'], round(s_['rate'], 4),
                  round(s_['ci95'][0], 4), round(s_['ci95'][1], 4),
                  b['end_to_end']['count'], round(b['end_to_end']['rate'], 4),
                  b['n_pathological']])

  fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
  ax[0].bar([0, 1], [b_nn['selection_rate_given_coverage']['rate'],
                     b_c['selection_rate_given_coverage']['rate']], 0.5,
            color=['tab:blue', 'tab:gray'])
  ax[0].set_xticks([0, 1])
  ax[0].set_xticklabels(['state-NN', 'argmin f_C'])
  ax[0].set_ylabel('SelectionRate | coverage')
  ax[0].set_title('sealed confirmation (n_cov=%d)' % n_cov)
  ax[1].scatter([phys(x)['v_xy'] for x in Ss], [phys(x)['torso_z'] for x in Ss],
                s=35, marker='^', color='tab:blue', label='safe')
  ax[1].scatter([phys(x)['v_xy'] for x in cand[np.arange(n), k_c]],
                [phys(x)['torso_z'] for x in cand[np.arange(n), k_c]], s=35,
                marker='s', facecolors='none', edgecolors='tab:gray',
                label='argmin f_C')
  ax[1].scatter([phys(x)['v_xy'] for x in cand[np.arange(n), k_nn]],
                [phys(x)['torso_z'] for x in cand[np.arange(n), k_nn]], s=35,
                marker='o', facecolors='none', edgecolors='darkorange',
                label='state-NN')
  ax[1].scatter([phys(x)['v_xy'] for x in Sf], [phys(x)['torso_z'] for x in Sf],
                s=50, marker='X', color='crimson', label='true fatal')
  ax[1].set_xlabel('|v_xy|')
  ax[1].set_ylabel('torso z')
  ax[1].set_title('physical placement')
  ax[1].legend(fontsize=7)
  ax[2].scatter(d_c, d_nn, s=40,
                c=np.where(Ci == 1, 'tab:green', 'tab:gray'))
  lim = [0, max(d_c.max(), d_nn.max()) * 1.05]
  ax[2].plot(lim, lim, 'k--', lw=1)
  ax[2].axhline(R_FATAL, color='green', ls=':', lw=1.4)
  ax[2].axvline(R_FATAL, color='green', ls=':', lw=1.4)
  ax[2].set_xlabel('argmin f_C selected d_fatal')
  ax[2].set_ylabel('state-NN selected d_fatal')
  ax[2].set_title('below diagonal = state-NN better')
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, 'physical_diagnostics.png'), dpi=140)
  plt.close(fig)

  print('\n[S6-7] %-28s %10s %8s %10s %6s'
        % ('selector', 'sel|cov', 'rate', 'e2e', 'patho'))
  for nm, b in (('frozen state-NN similarity', b_nn),
                ('frozen argmin Critic C', b_c)):
    s_ = b['selection_rate_given_coverage']
    print('       %-28s %10s %8.3f %10s %6d'
          % (nm, s_['count'], s_['rate'], b['end_to_end']['count'],
             b['n_pathological']))
  print('       CI95 state-NN [%.3f, %.3f] | critic [%.3f, %.3f]'
        % (*b_nn['selection_rate_given_coverage']['ci95'],
           *b_c['selection_rate_given_coverage']['ci95']))
  print('[S7] paired: %s' % paired)
  print('[S9] taxonomy: %s' % {k: v for k, v in tax.items()
                               if k.startswith(('G', 'path'))})
  print('[S8] coord range: state-NN %s | argmin %s | true fatal %s'
        % ([round(x, 2) for x in b_nn['coord_range']],
           [round(x, 2) for x in b_c['coord_range']],
           [round(x, 2) for x in summary['true_fatal_coord_range']]))
  rp = summary['reference_physical']
  print('[S8] physical medians z/v: fatal %.3f/%.3f | state-NN %.3f/%.3f | '
        'argmin %.3f/%.3f | safe %.3f/%.3f'
        % (rp['torso_z']['true_fatal']['median'],
           rp['v_xy']['true_fatal']['median'],
           b_nn['physical']['torso_z']['median'],
           b_nn['physical']['v_xy']['median'],
           b_c['physical']['torso_z']['median'],
           b_c['physical']['v_xy']['median'],
           rp['torso_z']['safe_successor']['median'],
           rp['v_xy']['safe_successor']['median']))
  print('\nsaved -> %s' % OUT)


if __name__ == '__main__':
  main()
