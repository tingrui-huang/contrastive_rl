"""DEVELOPMENT diagnostic: should worst-case selection use similarity to
negative samples instead of argmin f_C?

Runs entirely on the FROZEN fresh50 candidates already saved in
artifacts/flow_v3_fresh50/candidates.npz. Nothing is retrained, no candidate
is resampled, K and R_fatal are untouched, no metric is learned, and no
selector is combined with another after seeing results.

Negative reference banks
  Bank A  D_C^-  : the EXACT settled failure states used as Critic-C training
                   negatives (artifacts/settled_failure_bank_alpha01/
                   failure_bank_settled.npz, sha-verified, unmodified).
  Bank B  D_603^-: settled fatal SUCCESSOR states only, from the 603 frozen
                   factual failure transitions. These were NOT all used to
                   train Critic C -- labelled "expanded factual negative
                   reference bank", never "Critic-C training negatives".
                   No pre-death anchors, ordinary bad-demo states, wall
                   collisions, safe transitions or hidden variables.

Selectors compared (all select WITHOUT ever seeing the true fatal state)
  baseline   argmin_k f_C(s, a, s'_k), frozen twin f_min
  state NN   argmin_k min_j || norm(s'_k) - norm(g_j^-) ||_2
  psi cos    argmax_k max_j  cos( psi_h(s'_k), psi_h(g_j^-) ), heads
             L2-normalized, reported per head and as the fixed
             (sim_1 + sim_2)/2 mean -- no head aggregation is searched.

Success is scored only AFTER selection, with the frozen radius:
    S_i = 1[ d(selected_i, fatal_i) <= 3.17 ]  in normalized delta space.

fresh50 is already open, so EVERYTHING here is development-only. Any selector
chosen from this analysis must afterwards be frozen and tested once on a new
untouched sealed set before any RL use.

Usage:
  python scripts/diagnose_negative_similarity_selector.py
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
from scipy.stats import spearmanr, beta as _beta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from eval_flow_v1_dev16 import R_FATAL     # noqa: E402
from probe_flow_v0_failure_coverage import phys, qstats  # noqa: E402

FRESH = 'artifacts/flow_v3_fresh50'
OUT = 'artifacts/flow_v3_fresh50_negative_similarity_dev'
BANK_A = 'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz'
BANK_A_MAN = 'artifacts/settled_failure_bank_alpha01/bank_manifest.json'
POOL = ('artifacts/flow_v3_diverse_failure/failure_pool_diversity_audit/'
        'failure_pool_diverse.npz')
V0_DIR = 'artifacts/flow_v0_clean'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
BASELINE_EXPECTED = (16, 39)
EPS = 1e-8


def cp(k, n, a=0.05):
  lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
  hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
  return [lo, hi]


def psi_apply(qp, g, head):
  """Frozen Critic-C goal encoder psi_h(g). Manual forward pass over the
  stored haiku params -- the network is not modified in any way."""
  pre = 'g_encoder' if head == 0 else 'g_encoder_1'
  h = g
  n_lin = len([k for k in qp if k.startswith(pre + '/~/linear_')])
  for i in range(n_lin):
    p = qp['%s/~/linear_%d' % (pre, i)]
    h = h @ p['w'] + p['b']
    if i < n_lin - 1:
      h = jax.nn.relu(h)
  return h


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  # ---- frozen inputs -------------------------------------------------------
  z = np.load(os.path.join(FRESH, 'candidates.npz'), allow_pickle=True)
  S = np.asarray(z['anchor'], np.float32)
  A = np.asarray(z['action'], np.float32)
  Sf = np.asarray(z['true_fatal'], np.float32)
  Ss = np.asarray(z['safe'], np.float32)
  cand = np.asarray(z['candidates'], np.float32)          # [50, 256, 29]
  Df = np.asarray(z['d_fatal_all'], np.float32)           # [50, 256]
  Fmin = np.asarray(z['critic_fmin'], np.float32)
  f_fatal = np.asarray(z['f_fatal'], np.float32)
  eps_id = np.asarray(z['episode_id'], np.int64)
  n, K = cand.shape[0], cand.shape[1]
  rows = list(csv.DictReader(open(os.path.join(FRESH, 'per_case.csv'))))
  Ci = np.array([int(r['covered']) for r in rows])
  n_cov = int(Ci.sum())

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  s_mean = np.asarray(nz_['state_mean'], np.float32)
  s_std = np.asarray(nz_['state_std'], np.float32)

  # ---- banks ---------------------------------------------------------------
  bam = json.load(open(BANK_A_MAN))
  assert C.sha256_file(BANK_A) == bam['bank']['sha256'], 'Bank A drifted'
  bankA = np.asarray(np.load(BANK_A, allow_pickle=True)['goals'], np.float32)
  assert bankA.shape == (16, OBS_DIM)
  pz = np.load(POOL, allow_pickle=True)
  bankB = (np.asarray(pz['state'], np.float32)
           + np.asarray(pz['delta'], np.float32))       # settled successors
  assert bankB.shape == (603, OBS_DIM)
  banks = {'D_C^- (Critic-C training negatives, n=16)': bankA,
           'D_603^- (expanded factual reference, n=603)': bankB}
  print('Bank A %s sha %s | Bank B %s sha %s'
        % (bankA.shape, bam['bank']['sha256'][:12], bankB.shape,
           C.sha256_file(POOL)[:12]))

  # ---- baseline: reproduce the stored argmin f_C exactly -------------------
  k_base = Fmin.argmin(1)
  d_base = Df[np.arange(n), k_base]
  S_base = (d_base <= R_FATAL).astype(int)
  num, den = int((Ci * S_base).sum()), n_cov
  assert (num, den) == BASELINE_EXPECTED, \
      'baseline drift: got %d/%d, stored %d/%d' % (num, den, *BASELINE_EXPECTED)
  print('baseline argmin f_C reproduced exactly: %d/%d' % (num, den))

  # ---- psi embeddings (frozen critic) -------------------------------------
  from crl import checkpoint as ckpt_mod
  _, c_state = ckpt_mod.load_checkpoint(C_CKPT)
  qp = c_state.q_params

  def psi_norm(x, head):
    e = np.asarray(psi_apply(qp, jnp.asarray(x), head))
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + EPS)

  # ---- selectors -----------------------------------------------------------
  flat = cand.reshape(-1, OBS_DIM)
  flat_n = (flat - s_mean) / s_std
  results, per_cand = {}, {}
  for bname, bank in banks.items():
    bn = (bank - s_mean) / s_std
    # state-space nearest-negative distance
    d_state = np.empty(len(flat), np.float32)
    CH = 4096
    for i0 in range(0, len(flat), CH):
      blk = flat_n[i0:i0 + CH]
      d_state[i0:i0 + CH] = np.linalg.norm(
          blk[:, None] - bn[None], axis=2).min(1)
    d_state = d_state.reshape(n, K)
    k_state = d_state.argmin(1)

    # psi cosine similarity per head
    sims = []
    for head in (0, 1):
      pb = psi_norm(bank, head)
      sc = np.empty(len(flat), np.float32)
      for i0 in range(0, len(flat), CH):
        pc = psi_norm(flat[i0:i0 + CH], head)
        sc[i0:i0 + CH] = (pc @ pb.T).max(1)
      sims.append(sc.reshape(n, K))
    sim_mean = (sims[0] + sims[1]) / 2.0
    k_psi = sim_mean.argmax(1)
    per_cand[bname] = {'d_state': d_state, 'sim1': sims[0],
                       'sim2': sims[1], 'sim_mean': sim_mean}

    for sname, kk in (('state NN similarity', k_state),
                      ('psi cosine similarity', k_psi),
                      ('psi cos head1', sims[0].argmax(1)),
                      ('psi cos head2', sims[1].argmax(1))):
      dsel = Df[np.arange(n), kk]
      Ssel = (dsel <= R_FATAL).astype(int)
      results[(sname, bname)] = {
          'k': kk, 'd_sel': dsel, 'S': Ssel,
          'num': int((Ci * Ssel).sum()), 'den': n_cov,
          'e2e': int(Ssel.sum())}

  results[('argmin Critic C', '--')] = {
      'k': k_base, 'd_sel': d_base, 'S': S_base, 'num': num, 'den': den,
      'e2e': int(S_base.sum())}

  # ---- pathology repair ----------------------------------------------------
  f_wc_base = Fmin[np.arange(n), k_base]
  patho_base = np.where((S_base == 0) & (f_wc_base < f_fatal))[0]
  print('existing argmin pathologies: %d' % len(patho_base))

  def patho_of(kk, Ssel):
    fs = Fmin[np.arange(n), kk]
    return np.where((Ssel == 0) & (fs < f_fatal))[0]

  # ---- table ---------------------------------------------------------------
  main_rows = [('argmin Critic C', '--'),
               ('state NN similarity', list(banks)[0]),
               ('state NN similarity', list(banks)[1]),
               ('psi cosine similarity', list(banks)[0]),
               ('psi cosine similarity', list(banks)[1])]
  table = []
  for key in main_rows:
    r = results[key]
    rep = int(np.isin(patho_base, np.where(r['S'] == 1)[0]).sum())
    newp = patho_of(r['k'], r['S'])
    table.append({
        'selector': key[0], 'negative_bank': key[1],
        'selection_rate_given_coverage': {
            'count': '%d/%d' % (r['num'], r['den']),
            'rate': r['num'] / r['den'], 'ci95': cp(r['num'], r['den'])},
        'end_to_end': {'count': '%d/%d' % (r['e2e'], n), 'rate': r['e2e'] / n,
                       'ci95': cp(r['e2e'], n)},
        'pathologies_repaired_of_12': rep,
        'n_pathologies_own': int(len(newp)),
        'n_new_pathologies_not_in_baseline':
            int(len(np.setdiff1d(newp, patho_base)))})
  head_diag = {}
  for bname in banks:
    for hname in ('psi cos head1', 'psi cos head2'):
      r = results[(hname, bname)]
      head_diag['%s | %s' % (hname, bname)] = {
          'count': '%d/%d' % (r['num'], r['den']),
          'rate': r['num'] / r['den'], 'end_to_end': r['e2e'] / n}

  # ---- near-miss vs clearly wrong -----------------------------------------
  nearmiss = {}
  for key in main_rows:
    r = results[key]
    m = (Ci == 1) & (r['S'] == 0)
    d = r['d_sel'][m]
    nearmiss['%s | %s' % key] = {
        'n_missed_when_covered': int(m.sum()),
        'd_selected': qstats(d) if len(d) else None,
        'n_within_4.0': int((d <= 4.0).sum()) if len(d) else 0,
        'note': 'R_fatal stays 3.17; this is a continuous readout only'}

  # ---- physical ------------------------------------------------------------
  def physblock(states):
    return {f: qstats([phys(x)[f] for x in states])
            for f in ('torso_z', 'v_xy', 'up_z')}
  physical = {'true_fatal': physblock(Sf), 'safe_successor': physblock(Ss)}
  for key in main_rows:
    sel = cand[np.arange(n), results[key]['k']]
    physical['%s | %s' % key] = dict(
        physblock(sel),
        delta_norm=qstats(np.linalg.norm(sel - S, axis=1)),
        coord_range=[float(sel.min()), float(sel.max())])
  physical['true_fatal']['delta_norm'] = qstats(
      np.linalg.norm(Sf - S, axis=1))
  physical['true_fatal']['coord_range'] = [float(Sf.min()), float(Sf.max())]

  # ---- does similarity track fatalness? ------------------------------------
  dtf = Df.reshape(-1)
  corr = {}
  for bname in banks:
    pc = per_cand[bname]
    corr[bname] = {
        'spearman_d_neg_state_vs_d_true_fatal':
            float(spearmanr(pc['d_state'].reshape(-1), dtf).statistic),
        'spearman_neg_sim_psi_vs_d_true_fatal':
            float(spearmanr(-pc['sim_mean'].reshape(-1), dtf).statistic),
        'n_points': int(len(dtf))}

  summary = {
      'status': ('DEVELOPMENT ONLY -- fresh50 is already open. Any selector '
                 'chosen here must be frozen and tested once on a NEW '
                 'untouched sealed set before any RL use.'),
      'frozen_inputs': {'candidates': os.path.join(FRESH,
                                                   'candidates.npz'),
                        'K': K, 'R_fatal': R_FATAL,
                        'n_cases': n, 'n_covered': n_cov,
                        'baseline_reproduced': '%d/%d' % (num, den)},
      'banks': {'A_critic_training_negatives':
                {'path': BANK_A, 'sha256': bam['bank']['sha256'], 'n': 16},
                'B_expanded_factual_reference':
                {'path': POOL, 'sha256': C.sha256_file(POOL), 'n': 603,
                 'label': 'expanded factual negative reference bank -- NOT '
                          'Critic-C training negatives'}},
      'table': table, 'per_head_diagnostics': head_diag,
      'pathology': {'baseline_pathological_cases': patho_base.tolist(),
                    'n_baseline': int(len(patho_base))},
      'near_miss': nearmiss, 'physical': physical,
      'similarity_tracks_fatalness': corr,
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'),
            indent=2)
  np.savez_compressed(
      os.path.join(args.out, 'per_candidate_scores.npz'),
      d_state_bankA=per_cand[list(banks)[0]]['d_state'],
      d_state_bankB=per_cand[list(banks)[1]]['d_state'],
      sim_mean_bankA=per_cand[list(banks)[0]]['sim_mean'],
      sim_mean_bankB=per_cand[list(banks)[1]]['sim_mean'],
      sim1_bankA=per_cand[list(banks)[0]]['sim1'],
      sim2_bankA=per_cand[list(banks)[0]]['sim2'],
      sim1_bankB=per_cand[list(banks)[1]]['sim1'],
      sim2_bankB=per_cand[list(banks)[1]]['sim2'],
      d_true_fatal=Df, critic_fmin=Fmin,
      selected_idx=np.stack([results[k]['k'] for k in main_rows]),
      selector_names=np.array(['%s | %s' % k for k in main_rows]),
      episode_id=eps_id)
  with open(os.path.join(args.out, 'selector_comparison.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['selector', 'negative_bank', 'sel_rate_given_coverage',
                'rate', 'ci_lo', 'ci_hi', 'end_to_end', 'e2e_rate',
                'pathologies_repaired_of_12', 'own_pathologies',
                'new_pathologies'])
    for t in table:
      w.writerow([t['selector'], t['negative_bank'],
                  t['selection_rate_given_coverage']['count'],
                  round(t['selection_rate_given_coverage']['rate'], 4),
                  round(t['selection_rate_given_coverage']['ci95'][0], 4),
                  round(t['selection_rate_given_coverage']['ci95'][1], 4),
                  t['end_to_end']['count'], round(t['end_to_end']['rate'], 4),
                  t['pathologies_repaired_of_12'], t['n_pathologies_own'],
                  t['n_new_pathologies_not_in_baseline']])
  with open(os.path.join(args.out, 'pathology_repair.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'baseline_d_sel']
               + ['%s|%s' % k for k in main_rows[1:]])
    for i in patho_base:
      w.writerow([int(i), int(eps_id[i]), round(float(d_base[i]), 3)]
                 + [round(float(results[k]['d_sel'][i]), 3)
                    for k in main_rows[1:]])

  fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
  names = ['%s\n%s' % (t['selector'], t['negative_bank'].split(' ')[0])
           for t in table]
  vals = [t['selection_rate_given_coverage']['rate'] for t in table]
  ax[0].bar(range(len(vals)), vals,
            color=['tab:gray'] + ['tab:blue'] * 2 + ['crimson'] * 2)
  ax[0].set_xticks(range(len(vals)))
  ax[0].set_xticklabels(names, fontsize=6.5)
  ax[0].set_ylabel('SelectionRate | coverage')
  ax[0].set_title('selector comparison (development)')
  bA = per_cand[list(banks)[0]]
  ax[1].scatter(bA['d_state'].reshape(-1)[::17], dtf[::17], s=3, alpha=0.15)
  ax[1].set_xlabel('distance to nearest negative (state, Bank A)')
  ax[1].set_ylabel('d to TRUE fatal')
  ax[1].set_title('rho = %.3f'
                  % corr[list(banks)[0]]
                  ['spearman_d_neg_state_vs_d_true_fatal'])
  bB = per_cand[list(banks)[1]]
  ax[2].scatter(-bB['sim_mean'].reshape(-1)[::17], dtf[::17], s=3, alpha=0.15,
                color='crimson')
  ax[2].set_xlabel('-cos similarity to nearest negative (psi, Bank B)')
  ax[2].set_ylabel('d to TRUE fatal')
  ax[2].set_title('rho = %.3f'
                  % corr[list(banks)[1]]
                  ['spearman_neg_sim_psi_vs_d_true_fatal'])
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'selector_diagnostics.png'), dpi=140)
  plt.close(fig)

  print('\n%-24s %-34s %10s %8s %8s %6s'
        % ('selector', 'negative bank', 'sel|cov', 'rate', 'e2e', 'rep/12'))
  for t in table:
    print('%-24s %-34s %10s %8.3f %8s %6d'
          % (t['selector'], t['negative_bank'][:34],
             t['selection_rate_given_coverage']['count'],
             t['selection_rate_given_coverage']['rate'],
             t['end_to_end']['count'], t['pathologies_repaired_of_12']))
  print('\nper-head diagnostics:')
  for k_, v in head_diag.items():
    print('  %-52s %8s (%.3f)' % (k_[:52], v['count'], v['rate']))
  print('\nsimilarity tracks fatalness (Spearman vs d_true_fatal):')
  for b, v in corr.items():
    print('  %-46s state %.3f | -psi_sim %.3f'
          % (b[:46], v['spearman_d_neg_state_vs_d_true_fatal'],
             v['spearman_neg_sim_psi_vs_d_true_fatal']))
  print('\nphysical medians (z / v_xy):')
  for k_ in ['true_fatal'] + ['%s | %s' % k for k in main_rows]:
    p = physical[k_]
    print('  %-52s %.3f / %.3f' % (k_[:52], p['torso_z']['median'],
                                   p['v_xy']['median']))
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
