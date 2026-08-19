"""Task section 9: positive-goal distribution sanity diagnostic.

Runs BEFORE any learner update. Because the G2 audit rejected D_psi as a
Bernoulli coin, the rho-dependent quantities (empirical branch rate, fraction
of positives from s'_wc) cannot be reported for a REAL coin. This script
therefore separates:

  PART 1 (rho-independent, reported):
    nominal future-goal statistics, worst-case future-goal statistics, and
    their overlap with the 16 failure negatives -- properties of the frozen
    table and the frozen replay law, valid whatever rho turns out to be.

  PART 2 (rho-dependent, reported for a SUPPLIED rho only):
    branch rates under an explicitly supplied rho. ``--rho-source dpsi``
    reports what D_psi WOULD give, clearly labelled NOT-ENDORSED; the default
    ``--rho-source none`` skips it entirely. Nothing here selects or tunes a
    coin.

Usage:
  python scripts/diagnose_positive_goal_distribution.py [--rho-source none|dpsi]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl import static_worstcase as sw                       # noqa: E402
from crl.replay import TrajectoryBuffer                      # noqa: E402
from crl.pessimistic_positive import PessimisticPositiveBuffer  # noqa: E402

OUT = os.path.join(_ROOT, 'artifacts/static_worstcase_rl')
TABLE = os.path.join(OUT, 'worstcase_table.npz')
CLEAN = os.path.join(
    _ROOT, 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
    'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
OBS_DIM, ACT_DIM = 29, 8


def qstats(x):
  x = np.asarray(x, np.float64).ravel()
  return {'mean': float(x.mean()), 'median': float(np.median(x)),
          'p10': float(np.percentile(x, 10)),
          'p90': float(np.percentile(x, 90)),
          'min': float(x.min()), 'max': float(x.max())}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=32768)
  ap.add_argument('--rho-source', default='none', choices=('none', 'dpsi'))
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)
  assert os.path.exists(TABLE), 'worst-case table not built yet'

  with np.load(CLEAN, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
  E, L, W = obs.shape
  buf = TrajectoryBuffer(capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W,
                         action_dim=ACT_DIM, obs_dim=OBS_DIM, start_index=0,
                         end_index=-1, discount=0.99, seed=0,
                         goal_indices=tuple(range(OBS_DIM)))
  for k in range(E):
    buf.add_episode(obs[k], act[k])

  nz = np.load(os.path.join(_ROOT, sw.NORM_NPZ))
  sm = np.asarray(nz['state_mean'], np.float32)
  ss = np.asarray(nz['state_std'], np.float32)
  bank = np.asarray(np.load(os.path.join(_ROOT, sw.BANK_NPZ),
                            allow_pickle=True)['goals'], np.float32)
  bn = (bank - sm) / ss

  def dneg(x):
    xn = (x - sm) / ss
    out = np.empty(len(xn), np.float32)
    for i in range(0, len(xn), 4096):
      out[i:i + 4096] = np.linalg.norm(
          xn[i:i + 4096][:, None] - bn[None], axis=2).min(1)
    return out

  rho_half = lambda s, g, a: np.full(len(s), 0.5)      # noqa: E731 (unused coin)
  pb = PessimisticPositiveBuffer(buf, TABLE, rho_fn=rho_half, seed=0)

  # ---- PART 1: rho-independent goal populations --------------------------
  _, aux_n = pb.sample(args.n, force_branch=1, return_aux=True)
  g_nom = aux_n['goal_state']
  _, aux_w = pb.sample(args.n, force_branch=0, return_aux=True)
  g_wc = aux_w['goal_state']

  d_nom, d_wc = dneg(g_nom), dneg(g_wc)
  R = 3.17                                    # frozen fatal radius, reused
  rep = {
      'n_sampled': int(args.n),
      'table': os.path.basename(TABLE),
      'table_sha256': sw.sha256_file(TABLE),
      'PART1_rho_independent': {
          'nominal_future_goal': {
              'distance_to_nearest_failure_negative': qstats(d_nom),
              'frac_within_R_fatal_3.17': float((d_nom <= R).mean()),
              'xy': {'x': qstats(g_nom[:, 0]), 'y': qstats(g_nom[:, 1])},
              'torso_z': qstats(g_nom[:, 2])},
          'worstcase_future_goal': {
              'distance_to_nearest_failure_negative': qstats(d_wc),
              'frac_within_R_fatal_3.17': float((d_wc <= R).mean()),
              'xy': {'x': qstats(g_wc[:, 0]), 'y': qstats(g_wc[:, 1])},
              'torso_z': qstats(g_wc[:, 2])},
          'overlap_with_16_failure_negatives': {
              'nominal_frac_within_R': float((d_nom <= R).mean()),
              'worstcase_frac_within_R': float((d_wc <= R).mean()),
              'separation_median': float(np.median(d_nom) - np.median(d_wc)),
              'exact_duplicate_of_a_bank_state': int(
                  (dneg(g_wc) < 1e-6).sum()),
              'note': ('a worst-case goal identical to a bank negative would '
                       'put the SAME vector on both sides for the same '
                       'anchor; count should be 0')}},
      'PART2_rho_dependent': {
          'status': 'NOT REPORTED -- no calibrated rho exists (G2 audit 5B)',
          'blocked_metrics': ['mean/median/p10/p90 rho',
                              'empirical nominal-branch rate',
                              'empirical worst-case-branch rate',
                              'fraction of positive goals from s_wc']}}

  if args.rho_source == 'dpsi':
    from propensity.agreement import (load_agreement_model,
                                      agreement_score_batch)
    model = load_agreement_model(
        os.path.join(_ROOT, 'artifacts/support_discriminator/'
                            'D_state_cmdgoal_action'))

    def rho_dpsi(s, g, a):
      return np.asarray(agreement_score_batch(model.params, model.spec,
                                              s, g, a), np.float64)
    pb2 = PessimisticPositiveBuffer(buf, TABLE, rho_fn=rho_dpsi, seed=0)
    _, aux_m = pb2.sample(args.n, return_aux=True)
    rho = aux_m['rho']
    rep['PART2_rho_dependent'] = {
        'status': ('REPORTED FOR REFERENCE ONLY -- D_psi is NOT an endorsed '
                   'coin; the G2 audit found it uncalibrated and near-constant'),
        'rho': qstats(rho),
        'empirical_nominal_branch_rate': float(aux_m['nominal'].mean()),
        'empirical_worstcase_branch_rate':
            float(1 - aux_m['nominal'].mean()),
        'frac_positive_goals_from_swc': float(1 - aux_m['nominal'].mean()),
        'frac_rho_within_0.05_of_0.5': float((np.abs(rho - 0.5) < 0.05).mean()),
        'frac_rho_below_0.1': float((rho < 0.1).mean()),
        'frac_rho_above_0.9': float((rho > 0.9).mean()),
        'WARNING': 'do not use these numbers to select or tune a coin'}

  rep['git_commit'] = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()
  json.dump(rep, open(os.path.join(OUT, 'positive_goal_distribution.json'),
                      'w'), indent=2)

  p1 = rep['PART1_rho_independent']
  print('nominal  goal: d_neg median %.3f p10 %.3f p90 %.3f | frac<=R %.4f'
        % (p1['nominal_future_goal']['distance_to_nearest_failure_negative']
           ['median'],
           p1['nominal_future_goal']['distance_to_nearest_failure_negative']
           ['p10'],
           p1['nominal_future_goal']['distance_to_nearest_failure_negative']
           ['p90'], p1['nominal_future_goal']['frac_within_R_fatal_3.17']))
  print('worstcase goal: d_neg median %.3f p10 %.3f p90 %.3f | frac<=R %.4f'
        % (p1['worstcase_future_goal']['distance_to_nearest_failure_negative']
           ['median'],
           p1['worstcase_future_goal']['distance_to_nearest_failure_negative']
           ['p10'],
           p1['worstcase_future_goal']['distance_to_nearest_failure_negative']
           ['p90'], p1['worstcase_future_goal']['frac_within_R_fatal_3.17']))
  print('exact duplicates of a bank state: %d'
        % p1['overlap_with_16_failure_negatives'][
            'exact_duplicate_of_a_bank_state'])
  print('PART2: %s' % rep['PART2_rho_dependent']['status'])
  print('saved -> %s' % os.path.join(OUT, 'positive_goal_distribution.json'))


if __name__ == '__main__':
  main()
