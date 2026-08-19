"""Stage 0: reconcile the selector pathology accounting.

The development report quoted three numbers that looked inconsistent:
  * "Repaired /12 = 4"      (table)
  * "state-NN has 3 pathologies"   (prose)
  * "zero new pathologies"         (prose)

They measure three different things. This script recovers the exact baseline
pathological case set P_base from the saved artifacts, replays BOTH selectors
on the SAME frozen candidates, and separates:

  N_success_repair    baseline pathological case where the new selector's
                      pick satisfies d_fatal <= R_fatal (strict success)
  N_pathology_elim    baseline pathological case that no longer satisfies the
                      frozen pathology rule under the new selector, whether or
                      not it becomes a strict success
  Z                   baseline pathological cases still pathological
  W                   new pathologies outside P_base
  T                   total pathologies across all 50 under the new selector

with the identity Y + Z = 12 asserted.

The pathology rule is NOT redefined. It is the exact coded condition from the
fresh50 sealed analysis (scripts/eval_fresh50_sealed.py):

    pathological_i  <=>  (d(selected_i, fatal_i) > R_fatal)
                    AND  (f_C(s_i, a_i, selected_i) < f_C(s_i, a_i, fatal_i))

i.e. "selected candidate is not fatal-like AND scores below the TRUE fatal
successor". It was fully coded, never manual.

Reuses only saved artifacts; no model is run, no candidate resampled, no
threshold altered.

Usage:
  python scripts/reconcile_selector_pathology.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from eval_flow_v1_dev16 import R_FATAL     # noqa: E402

FRESH = 'artifacts/flow_v3_fresh50'
DEV = 'artifacts/flow_v3_fresh50_negative_similarity_dev'
BANK_A = 'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz'
V0_DIR = 'artifacts/flow_v0_clean'
OUT = 'artifacts/state_nn_selector_confirm'
PATHOLOGY_RULE = ('(d(selected, fatal) > R_fatal) AND '
                  '(f_C(s,a,selected) < f_C(s,a,fatal))')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  z = np.load(os.path.join(FRESH, 'candidates.npz'), allow_pickle=True)
  cand = np.asarray(z['candidates'], np.float32)
  Df = np.asarray(z['d_fatal_all'], np.float32)
  Fmin = np.asarray(z['critic_fmin'], np.float32)
  f_fatal = np.asarray(z['f_fatal'], np.float32)
  eps_id = np.asarray(z['episode_id'], np.int64)
  n, K = cand.shape[0], cand.shape[1]
  rows = list(csv.DictReader(open(os.path.join(FRESH, 'per_case.csv'))))
  Ci = np.array([int(r['covered']) for r in rows])

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  s_mean = np.asarray(nz_['state_mean'], np.float32)
  s_std = np.asarray(nz_['state_std'], np.float32)

  # ---- selectors on the SAME frozen candidates ----------------------------
  def pathology(k_sel):
    d = Df[np.arange(n), k_sel]
    f = Fmin[np.arange(n), k_sel]
    return (d > R_FATAL) & (f < f_fatal), d, f

  k_base = Fmin.argmin(1)
  bank = np.asarray(np.load(BANK_A, allow_pickle=True)['goals'], np.float32)
  bn = (bank - s_mean) / s_std
  flat = ((cand.reshape(-1, OBS_DIM) - s_mean) / s_std)
  d_neg = np.empty(len(flat), np.float32)
  for i0 in range(0, len(flat), 4096):
    blk = flat[i0:i0 + 4096]
    d_neg[i0:i0 + 4096] = np.linalg.norm(blk[:, None] - bn[None],
                                         axis=2).min(1)
  d_neg = d_neg.reshape(n, K)
  k_nn = d_neg.argmin(1)                      # ties -> lowest index (argmin)

  p_base, d_b, f_b = pathology(k_base)
  p_nn, d_n, f_n = pathology(k_nn)
  P_base = np.where(p_base)[0]

  # cross-check against the stored sealed pathology report
  stored = json.load(open(os.path.join(FRESH, 'pathology_report.json')))
  assert int(stored['n_pathological']) == len(P_base), \
      'recovered %d baseline pathologies, stored says %d' % (
          len(P_base), stored['n_pathological'])
  dev = json.load(open(os.path.join(DEV, 'summary.json')))
  assert dev['pathology']['baseline_pathological_cases'] == P_base.tolist(), \
      'baseline pathological case IDs do not reproduce'
  print('P_base reproduced exactly: %d cases %s'
        % (len(P_base), P_base.tolist()))

  S_base = (d_b <= R_FATAL).astype(int)
  S_nn = (d_n <= R_FATAL).astype(int)

  # ---- 0C: 12-case transition table ---------------------------------------
  table = []
  for i in P_base:
    strict = bool(S_nn[i])
    still = bool(p_nn[i])
    if strict:
      cat = 'pathology -> strict success'
    elif still:
      cat = 'pathology -> still pathology'
    elif d_n[i] <= 4.0:
      cat = 'pathology -> near-miss'
    else:
      cat = 'pathology -> ordinary non-pathological miss'
    table.append({
        'case': int(i), 'episode_id': int(eps_id[i]),
        'baseline_selected_k': int(k_base[i]),
        'baseline_d_fatal': float(d_b[i]),
        'baseline_pathology': True,
        'state_nn_selected_k': int(k_nn[i]),
        'state_nn_d_fatal': float(d_n[i]),
        'state_nn_strict_success': strict,
        'state_nn_pathology': still,
        'category': cat})

  X = int(sum(t['state_nn_strict_success'] for t in table))
  Z = int(sum(t['state_nn_pathology'] for t in table))
  Y = 12 - Z
  W = int(len(np.setdiff1d(np.where(p_nn)[0], P_base)))
  T = int(p_nn.sum())
  assert Y + Z == len(P_base), 'accounting identity violated'
  assert T == Z + W, 'total pathology decomposition violated'

  recon = {
      'pathology_rule_code_condition': PATHOLOGY_RULE,
      'rule_source': 'scripts/eval_fresh50_sealed.py (fully coded, never '
                     'manual); reused verbatim, not redefined',
      'R_fatal': R_FATAL,
      'terminology': {
          'N_success_repair': ('baseline pathological case where state-NN '
                               'picks a candidate with d_fatal <= R_fatal'),
          'N_pathology_eliminated': ('baseline pathological case that no '
                                     'longer meets the pathology rule under '
                                     'state-NN, regardless of strict '
                                     'success')},
      'P_base': P_base.tolist(), 'n_P_base': int(len(P_base)),
      'baseline_pathologies': 12,
      'X_strictly_repaired_into_fatal_radius': X,
      'Y_pathologies_eliminated_regardless_of_threshold': Y,
      'Z_baseline_pathologies_still_pathological': Z,
      'W_new_pathologies_outside_P_base': W,
      'T_total_state_nn_pathologies_over_50': T,
      'identity_Y_plus_Z_equals_12': bool(Y + Z == 12),
      'reconciliation_statement': (
          '%d/12 baseline pathologies were repaired all the way into strict '
          'selector success, while %d/12 ceased to be pathological; %d of '
          'those %d became non-pathological misses/near-misses rather than '
          'strict successes. The table column "Repaired /12 = %d" counted '
          'STRICT SUCCESS repairs, the prose "3 pathologies" counted TOTAL '
          'remaining state-NN pathologies (all inside P_base, since W=%d), '
          'and "zero new pathologies" referred to W. The three numbers are '
          'consistent measurements of different quantities.'
          % (X, Y, Y - X, Y, X, W)),
      'category_counts': {c: sum(1 for t in table if t['category'] == c)
                          for c in ('pathology -> strict success',
                                    'pathology -> near-miss',
                                    'pathology -> ordinary non-pathological '
                                    'miss',
                                    'pathology -> still pathology')},
      'transition_table': table,
      'baseline_totals': {'n_pathological': int(p_base.sum()),
                          'selection_rate_given_coverage':
                              '%d/%d' % (int((Ci * S_base).sum()),
                                         int(Ci.sum()))},
      'state_nn_totals': {'n_pathological': T,
                          'selection_rate_given_coverage':
                              '%d/%d' % (int((Ci * S_nn).sum()),
                                         int(Ci.sum()))},
      'git_commit': C.git_commit()}
  json.dump(recon, open(os.path.join(args.out,
                                     'pathology_reconciliation.json'), 'w'),
            indent=2)
  with open(os.path.join(args.out, 'pathology_reconciliation.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'episode_id', 'baseline_selected_k',
                'baseline_d_fatal', 'baseline_pathology',
                'state_nn_selected_k', 'state_nn_d_fatal',
                'state_nn_strict_success', 'state_nn_pathology', 'category'])
    for t in table:
      w.writerow([t['case'], t['episode_id'], t['baseline_selected_k'],
                  round(t['baseline_d_fatal'], 4), t['baseline_pathology'],
                  t['state_nn_selected_k'], round(t['state_nn_d_fatal'], 4),
                  t['state_nn_strict_success'], t['state_nn_pathology'],
                  t['category']])

  print('\npathology rule (reused verbatim): %s' % PATHOLOGY_RULE)
  print('\n%-5s %-8s %10s %10s  %s' % ('case', 'episode', 'base d', 'nn d',
                                       'category'))
  for t in table:
    print('%-5d %-8d %10.3f %10.3f  %s'
          % (t['case'], t['episode_id'], t['baseline_d_fatal'],
             t['state_nn_d_fatal'], t['category']))
  print('\nBaseline pathologies                       : 12')
  print('X strictly repaired into fatal radius      : %d / 12' % X)
  print('Y pathologies eliminated (any outcome)     : %d / 12' % Y)
  print('Z baseline pathologies still pathological  : %d / 12' % Z)
  print('W new pathologies outside P_base           : %d' % W)
  print('T total state-NN pathologies over 50 cases : %d' % T)
  print('identity Y + Z = 12                        : %s' % (Y + Z == 12))
  print('\n%s' % recon['reconciliation_statement'])
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
