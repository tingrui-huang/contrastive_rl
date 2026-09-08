"""Collect the clean demonstrator dataset for long two-rockfall V6.

The environment owns and naturally samples the two independent hazard coins
and their absolute schedules at reset.  This collector owns only the
independent teacher-route coin.  Privileged variables are written to a
separate sidecar and composition report; the learner files contain only
``obs``, ``act``, ``lengths``, ``eval_goals`` and aggregate JSON metadata.

Two learner files are emitted from the same rollouts:

* ``antmaze_rockfall_clock_v6.npz`` stores V5's historical 58-column
  ``state(29) | zero-padded goal(29)`` observation.
* ``antmaze_rockfall_clock_v6_gxy.npz`` is a direct column selection with the
  upstream Ant contract, ``state(29) | goal_xy(2)`` (31 columns).  This is the
  default vanilla-CRL training dataset.

No separate failure-bank episodes are collected, and no failure-bank fields
or sampling logic are present here.  Any failure or timeout produced by a
natural teacher draw remains in the demonstrator dataset and its audit.

Example::

  python scripts/collect_rockfall_clock_v6_dataset.py --episodes 1000 \
      --p-active-1 0.35 --p-active-2 0.35 \
      --teacher-detour-prob 0.05 --t0-min-1 10 --t0-max-1 45 \
      --t0-min-2 90 --t0-max-2 125 --horizon 800
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                         # noqa: E402
from crl import rockfall_clock_v6 as V6                  # noqa: E402
import rockfall_clock_v6_teacher as CT                   # noqa: E402


OUT_DIR = os.path.join(CT.OUT, 'dataset')
DATASET_NAME = 'antmaze_rockfall_clock_v6'
ROUTE_SEED_OFFSET = 370_003
PERMUTATION_SEED_OFFSET = 970_003
STATE_DIM = 29
RAW_GOAL_DIM = 29
RAW_OBS_DIM = STATE_DIM + RAW_GOAL_DIM
XY_GOAL_DIM = 2
XY_OBS_DIM = STATE_DIM + XY_GOAL_DIM
ACTION_DIM = 8
LATENT_COMBOS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _validate_args(args):
  if args.episodes <= 0:
    raise ValueError(f'--episodes must be positive, got {args.episodes}')
  if args.horizon <= 0:
    raise ValueError(f'--horizon must be positive, got {args.horizon}')
  for name in ('p_active_1', 'p_active_2', 'teacher_detour_prob'):
    value = float(getattr(args, name))
    if not 0.0 <= value <= 1.0:
      raise ValueError(f'--{name.replace("_", "-")} must be in [0, 1], '
                       f'got {value}')
  for zone in (1, 2):
    lo = int(getattr(args, f't0_min_{zone}'))
    hi = int(getattr(args, f't0_max_{zone}'))
    if lo > hi:
      raise ValueError(f'zone {zone} t0 minimum exceeds maximum: {lo}>{hi}')
  if args.progress_every <= 0:
    raise ValueError('--progress-every must be positive')
  if not args.name or os.path.basename(args.name) != args.name:
    raise ValueError('--name must be a non-empty file stem, not a path')


def _none_int(value):
  return -1 if value is None else int(value)


def _none_text(value):
  return 'none' if value is None else str(value)


def _sampled_t0(env, zone):
  """Read the privileged raw t0, including on inactive episodes.

  V6 deliberately draws t0 on every reset.  The public effective schedule
  reports ``None`` for an inactive zone, so use V6's explicit audit-only
  ``privileged_sampled_start(zone)`` accessor.
  """
  return int(env.privileged_sampled_start(zone))


def _capture_jitter(env, zone):
  return np.asarray(env._drop_jitter[zone], np.float64).copy()  # pylint: disable=protected-access


def _rate(values):
  values = np.asarray(values)
  return round(float(np.mean(values)), 6) if values.size else None


def _corr(left, right):
  left = np.asarray(left, np.float64)
  right = np.asarray(right, np.float64)
  if len(left) < 2 or left.std() == 0.0 or right.std() == 0.0:
    return None
  return round(float(np.corrcoef(left, right)[0, 1]), 6)


def _sha256(path, chunk_size=1 << 20):
  digest = hashlib.sha256()
  with open(path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _schedule_summary(values, low, high):
  values = np.asarray(values, np.int64)
  support = np.arange(low, high + 1, dtype=np.int64)
  counts = np.asarray([np.sum(values == value) for value in support])
  expected = len(values) / len(support) if len(support) else 0.0
  scale = math.sqrt(expected * (1.0 - 1.0 / len(support))) if expected else 0.0
  residuals = ((counts - expected) / scale if scale > 0.0
               else np.zeros_like(counts, np.float64))
  return {
      'configured_range_inclusive': [int(low), int(high)],
      'min': int(values.min()) if len(values) else None,
      'max': int(values.max()) if len(values) else None,
      'mean': round(float(values.mean()), 6) if len(values) else None,
      'expected_uniform_mean': round((low + high) / 2.0, 6),
      'expected_count_per_integer': round(expected, 6),
      'counts': {str(value): int(count)
                 for value, count in zip(support, counts)},
      'max_abs_standardized_count_residual': (
          round(float(np.max(np.abs(residuals))), 6)
          if len(residuals) else None),
  }


def _wilson(successes, n, z=1.959963984540054):
  """Wilson binomial interval without a scipy dependency."""
  if n <= 0:
    return None
  p = successes / n
  den = 1.0 + z * z / n
  centre = (p + z * z / (2.0 * n)) / den
  radius = (z / den) * math.sqrt(
      p * (1.0 - p) / n + z * z / (4.0 * n * n))
  return [round(max(0.0, centre - radius), 6),
          round(min(1.0, centre + radius), 6)]


def _bool_independence(left, right):
  """2x2 Pearson/phi diagnostic; chi-square(1) p-value is exact formula."""
  left = np.asarray(left, bool)
  right = np.asarray(right, bool)
  if left.shape != right.shape:
    raise ValueError('independence inputs must have the same shape')
  n = int(left.size)
  table = np.array([
      [np.sum(~left & ~right), np.sum(~left & right)],
      [np.sum(left & ~right), np.sum(left & right)],
  ], np.float64)
  row = table.sum(axis=1)
  col = table.sum(axis=0)
  expected = np.outer(row, col) / n if n else np.zeros((2, 2))
  valid = expected > 0
  chi2 = (float(np.sum((table[valid] - expected[valid]) ** 2
                       / expected[valid])) if valid.all() else None)
  p_value = (math.erfc(math.sqrt(chi2 / 2.0))
             if chi2 is not None else None)
  phi = _corr(left.astype(float), right.astype(float))
  return {
      'n': n,
      'contingency_rows_left_0_1_cols_right_0_1': table.astype(int).tolist(),
      'expected_under_empirical_marginals': np.round(expected, 4).tolist(),
      'phi': phi,
      'pearson_chi_square_df1': (round(chi2, 6)
                                 if chi2 is not None else None),
      'asymptotic_p_value': (round(p_value, 6)
                             if p_value is not None else None),
  }


def _joint_route_independence(u1, u2, detour, seed, permutations=2000):
  """Permutation diagnostic for route independence from the joint U pair."""
  u1 = np.asarray(u1, np.int8)
  u2 = np.asarray(u2, np.int8)
  detour = np.asarray(detour, bool)
  # Required/reporting order: U00, U10, U01, U11.
  group = u1 + 2 * u2

  def statistic(route):
    table = np.zeros((2, 4), np.float64)
    for latent_id in range(4):
      mask = group == latent_id
      table[0, latent_id] = np.sum(~route & mask)
      table[1, latent_id] = np.sum(route & mask)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / len(route)
    valid = expected > 0
    return (float(np.sum((table[valid] - expected[valid]) ** 2
                         / expected[valid])), table, expected)

  if len(detour) == 0:
    return {'n': 0, 'permutations': permutations, 'p_value': None}
  observed, table, expected = statistic(detour)
  rng = np.random.default_rng(seed)
  exceed = 0
  for _ in range(permutations):
    permuted = rng.permutation(detour)
    perm_stat, _, _ = statistic(permuted)
    exceed += int(perm_stat >= observed - 1e-12)
  return {
      'n': int(len(detour)),
      'contingency_rows_shortcut_detour_cols_U00_U10_U01_U11':
          table.astype(int).tolist(),
      'expected_under_empirical_marginals': np.round(expected, 4).tolist(),
      'pearson_chi_square_df3': round(observed, 6),
      'permutations': int(permutations),
      'permutation_seed': int(seed),
      'permutation_p_value': round((exceed + 1) / (permutations + 1), 6),
  }


def _latent_key(row):
  return f'U{int(bool(row["u1"]))}{int(bool(row["u2"]))}'


def _expected_latent_probabilities(p1, p2):
  return {
      'U00': (1.0 - p1) * (1.0 - p2),
      'U10': p1 * (1.0 - p2),
      'U01': (1.0 - p1) * p2,
      'U11': p1 * p2,
  }


def _population_report(rows, p1, p2, detour_prob, seed,
                       t0_range_1=None, t0_range_2=None):
  """Composition/outcome diagnostics for kept episodes or all draws."""
  if t0_range_1 is None:
    t0_range_1 = (V6.T0_MIN_1, V6.T0_MAX_1)
  if t0_range_2 is None:
    t0_range_2 = (V6.T0_MIN_2, V6.T0_MAX_2)
  n = len(rows)
  u1 = np.asarray([r['u1'] for r in rows], bool)
  u2 = np.asarray([r['u2'] for r in rows], bool)
  detour = np.asarray([r['teacher_take_detour'] for r in rows], bool)
  expected_prob = _expected_latent_probabilities(p1, p2)
  by_latent = {}
  max_abs_z = 0.0
  for a, b in LATENT_COMBOS:
    key = f'U{a}{b}'
    group = [r for r in rows if r['u1'] == bool(a) and r['u2'] == bool(b)]
    count = len(group)
    p = expected_prob[key]
    variance = n * p * (1.0 - p)
    z = ((count - n * p) / math.sqrt(variance)
         if variance > 0.0 else None)
    if z is not None:
      max_abs_z = max(max_abs_z, abs(z))
    n_detour = int(sum(r['teacher_take_detour'] for r in group))
    by_latent[key] = {
        'n': count,
        'observed_fraction': round(count / n, 6) if n else None,
        'expected_probability': round(p, 6),
        'expected_count': round(n * p, 3),
        'standardized_count_residual': round(z, 6) if z is not None else None,
        'detour_n': n_detour,
        'detour_fraction': round(n_detour / count, 6) if count else None,
        'detour_expected_count': round(count * detour_prob, 3),
        'detour_wilson95': _wilson(n_detour, count),
        'detour_wilson99': _wilson(n_detour, count, z=2.5758293035489004),
        'success': _rate([r['success'] for r in group]),
        'failure': _rate([r['failure'] for r in group]),
        'timeout': _rate([r['timeout'] for r in group]),
        'failure_zone_1': int(sum(r['failure_zone'] == 1 for r in group)),
        'failure_zone_2': int(sum(r['failure_zone'] == 2 for r in group)),
    }

  shortcut = [r for r in rows if not r['teacher_take_detour']]
  wait1 = np.asarray([r['wait_zone_1'] for r in shortcut], bool)
  wait2 = np.asarray([r['wait_zone_2'] for r in shortcut], bool)
  decision1 = [_none_text(r['decision_zone_1']) for r in shortcut]
  decision2 = [_none_text(r['decision_zone_2']) for r in shortcut]
  n_detour = int(detour.sum())
  p_u = _bool_independence(u1, u2)
  route_joint = _joint_route_independence(
      u1, u2, detour, seed + PERMUTATION_SEED_OFFSET)
  return {
      'n': n,
      'latent': {
          'p_active_1_configured': p1,
          'p_active_2_configured': p2,
          'u1_observed': _rate(u1),
          'u2_observed': _rate(u2),
          'u1_wilson99': _wilson(int(u1.sum()), n,
                                 z=2.5758293035489004),
          'u2_wilson99': _wilson(int(u2.sum()), n,
                                 z=2.5758293035489004),
          'by_combination': by_latent,
          'max_abs_standardized_combination_residual': round(max_abs_z, 6),
          'u1_vs_u2_independence': p_u,
      },
      'teacher_route': {
          'configured_detour_probability': detour_prob,
          'detour_n': n_detour,
          'detour_fraction': round(n_detour / n, 6) if n else None,
          'expected_detour_count': round(n * detour_prob, 3),
          'detour_wilson95': _wilson(n_detour, n),
          'detour_wilson99': _wilson(n_detour, n,
                                     z=2.5758293035489004),
          'detour_vs_u1': _bool_independence(detour, u1),
          'detour_vs_u2': _bool_independence(detour, u2),
          'detour_vs_joint_u1_u2': route_joint,
      },
      'shortcut_waits': {
          'n_shortcut': len(shortcut),
          'wait_zone_1_n': int(wait1.sum()),
          'wait_zone_1_fraction': _rate(wait1),
          'go_direct_zone_1_n': decision1.count('go'),
          'go_direct_zone_1_fraction': _rate(
              [d == 'go' for d in decision1]),
          'undecided_zone_1_n': decision1.count('none'),
          'wait_zone_2_n': int(wait2.sum()),
          'wait_zone_2_fraction': _rate(wait2),
          'go_direct_zone_2_n': decision2.count('go'),
          'go_direct_zone_2_fraction': _rate(
              [d == 'go' for d in decision2]),
          'undecided_zone_2_n': decision2.count('none'),
          'wait_both_n': int(np.sum(wait1 & wait2)),
          'wait_both_fraction': _rate(wait1 & wait2),
          'wait_zone_1_only_n': int(np.sum(wait1 & ~wait2)),
          'wait_zone_1_only_fraction': _rate(wait1 & ~wait2),
          'wait_zone_2_only_n': int(np.sum(~wait1 & wait2)),
          'wait_zone_2_only_fraction': _rate(~wait1 & wait2),
          'wait_neither_n': int(np.sum(~wait1 & ~wait2)),
          'wait_neither_fraction': _rate(~wait1 & ~wait2),
      },
      'outcomes': {
          'success_n': int(sum(r['success'] for r in rows)),
          'success': _rate([r['success'] for r in rows]),
          'failure_n': int(sum(r['failure'] for r in rows)),
          'failure': _rate([r['failure'] for r in rows]),
          'timeout_n': int(sum(r['timeout'] for r in rows)),
          'timeout': _rate([r['timeout'] for r in rows]),
          'failure_zone_1_n': int(sum(r['failure_zone'] == 1 for r in rows)),
          'failure_zone_2_n': int(sum(r['failure_zone'] == 2 for r in rows)),
      },
      'schedules': {
          't0_1': _schedule_summary(
              [r['t0_1'] for r in rows], *t0_range_1),
          't0_2': _schedule_summary(
              [r['t0_2'] for r in rows], *t0_range_2),
          'corr_t0_1_t0_2': _corr([r['t0_1'] for r in rows],
                                  [r['t0_2'] for r in rows]),
          'corr_t0_1_u1': _corr([r['t0_1'] for r in rows], u1),
          'corr_t0_2_u2': _corr([r['t0_2'] for r in rows], u2),
          'corr_t0_1_teacher_detour': _corr(
              [r['t0_1'] for r in rows], detour),
          'corr_t0_2_teacher_detour': _corr(
              [r['t0_2'] for r in rows], detour),
      },
  }


def _contains(interval, value):
  return bool(interval is not None and interval[0] <= value <= interval[1])


def _composition_report(kept, draws, discarded, args, paths):
  kept_report = _population_report(
      kept, args.p_active_1, args.p_active_2,
      args.teacher_detour_prob, args.seed,
      (args.t0_min_1, args.t0_max_1),
      (args.t0_min_2, args.t0_max_2))
  draw_report = _population_report(
      draws, args.p_active_1, args.p_active_2,
      args.teacher_detour_prob, args.seed + 1,
      (args.t0_min_1, args.t0_max_1),
      (args.t0_min_2, args.t0_max_2))

  selection = {}
  for a, b in LATENT_COMBOS:
    key = f'U{a}{b}'
    attempted = sum(_latent_key(row) == key for row in draws)
    accepted = sum(_latent_key(row) == key for row in kept)
    selection[key] = {
        'draws': attempted, 'kept': accepted,
        'discarded': attempted - accepted,
        'discard_fraction': (round((attempted - accepted) / attempted, 6)
                             if attempted else None),
    }

  kept_latent = kept_report['latent']
  kept_route = kept_report['teacher_route']
  combo_intervals = [
      block['detour_wilson99']
      for block in kept_latent['by_combination'].values()
      if block['n'] > 0
  ]
  t1 = kept_report['schedules']['t0_1']
  t2 = kept_report['schedules']['t0_2']
  p_u = kept_latent['u1_vs_u2_independence']['asymptotic_p_value']
  p_route = kept_route['detour_vs_joint_u1_u2']['permutation_p_value']
  gates = {
      'all_four_latent_combinations_present': all(
          block['n'] > 0
          for block in kept_latent['by_combination'].values()),
      'u1_configured_probability_in_wilson99': _contains(
          kept_latent['u1_wilson99'], args.p_active_1),
      'u2_configured_probability_in_wilson99': _contains(
          kept_latent['u2_wilson99'], args.p_active_2),
      'combination_counts_within_4p5_sigma': (
          kept_latent['max_abs_standardized_combination_residual'] <= 4.5),
      'u1_u2_independence_not_rejected_at_0p001': (
          p_u is None or p_u >= 0.001),
      'overall_detour_probability_in_wilson99': _contains(
          kept_route['detour_wilson99'], args.teacher_detour_prob),
      'per_latent_detour_probability_in_wilson99': all(
          _contains(interval, args.teacher_detour_prob)
          for interval in combo_intervals),
      'teacher_route_joint_independence_not_rejected_at_0p001': (
          p_route is None or p_route >= 0.001),
      't0_1_inside_configured_range': (
          args.t0_min_1 <= t1['min'] <= t1['max'] <= args.t0_max_1),
      't0_2_inside_configured_range': (
          args.t0_min_2 <= t2['min'] <= t2['max'] <= args.t0_max_2),
      't0_1_uniform_counts_within_4p5_sigma': (
          t1['max_abs_standardized_count_residual'] <= 4.5),
      't0_2_uniform_counts_within_4p5_sigma': (
          t2['max_abs_standardized_count_residual'] <= 4.5),
      'teacher_success_above_0p90': (
          kept_report['outcomes']['success'] is not None
          and kept_report['outcomes']['success'] > 0.90),
      'kept_teacher_failures_zero': (
          kept_report['outcomes']['failure_n'] == 0),
      'learner_raw_width_58': paths['raw_obs_shape'][-1] == RAW_OBS_DIM,
      'learner_xy_width_31': paths['xy_obs_shape'][-1] == XY_OBS_DIM,
      'no_failure_bank': True,
  }
  return {
      'environment': CT.ENV_NAME,
      'environment_version': V6.ENV_VERSION,
      'dataset_paths': paths,
      'config': {
          'episodes_kept': args.episodes,
          'collection_seed': args.seed,
          'route_rng_seed': args.seed + ROUTE_SEED_OFFSET,
          'p_active_1': args.p_active_1,
          'p_active_2': args.p_active_2,
          'teacher_detour_prob': args.teacher_detour_prob,
          't0_range_1': [args.t0_min_1, args.t0_max_1],
          't0_range_2': [args.t0_min_2, args.t0_max_2],
          'horizon': args.horizon,
      },
      'n_draws': len(draws),
      'n_kept': len(kept),
      'n_discarded': len(discarded),
      'discard_fraction': round(len(discarded) / len(draws), 6),
      'selection_rule': (
          'none: every natural reset and independent teacher-route draw is '
          'kept, including failures, timeouts, and unrealized routes'),
      'selection_by_latent': selection,
      'kept': kept_report,
      'all_draws_before_selection': draw_report,
      'gates': gates,
      'all_pass': bool(all(gates.values())),
  }


def _row_from_episode(env, teacher, route, draw_index, goal_xy, info, steps,
                      ret, final_obs, t0, jitter, any_contact,
                      first_contact, exit_step):
  u1 = bool(env.privileged_rockfall_active_1)
  u2 = bool(env.privileged_rockfall_active_2)
  decisions = teacher.decisions
  holds = teacher.hold_steps
  releases = teacher.release_steps
  success = bool(info.get('success'))
  failure = bool(info.get('failure'))
  row = {
      'draw_index': int(draw_index),
      'env_seed': int(info.get('env_seed', -1)),
      'reset_index': int(info.get('reset_index', draw_index)),
      'u1': u1, 'u2': u2,
      'latent': f'U{int(u1)}{int(u2)}',
      't0_1': int(t0[1]), 't0_2': int(t0[2]),
      'rockfall_start_1': info.get('rockfall_start_1'),
      'rockfall_start_2': info.get('rockfall_start_2'),
      'rockfall_end_1': info.get('rockfall_end_1'),
      'rockfall_end_2': info.get('rockfall_end_2'),
      'teacher_take_detour': route == 'detour',
      'route': route, 'route_realized': info.get('route'),
      'decision_zone_1': decisions[1],
      'decision_zone_2': decisions[2],
      'wait_zone_1': decisions[1] == 'wait',
      'wait_zone_2': decisions[2] == 'wait',
      'go_direct_zone_1': decisions[1] == 'go',
      'go_direct_zone_2': decisions[2] == 'go',
      'hold_steps_zone_1': int(holds[1]),
      'hold_steps_zone_2': int(holds[2]),
      'release_step_zone_1': releases[1],
      'release_step_zone_2': releases[2],
      'mouth_step_zone_1': info.get('mouth_step_1'),
      'mouth_step_zone_2': info.get('mouth_step_2'),
      'band_entry_step_zone_1': info.get('band_entry_step_1'),
      'band_entry_step_zone_2': info.get('band_entry_step_2'),
      'band_exit_step_zone_1': exit_step[1],
      'band_exit_step_zone_2': exit_step[2],
      'entered_hazard_zone_1': bool(info.get('entered_hazard_1')),
      'entered_hazard_zone_2': bool(info.get('entered_hazard_2')),
      'rock_waves_zone_1': int(info.get('rock_waves_1', 0)),
      'rock_waves_zone_2': int(info.get('rock_waves_2', 0)),
      'rockfall_passed_zone_1': bool(info.get('rockfall_passed_1')),
      'rockfall_passed_zone_2': bool(info.get('rockfall_passed_2')),
      'rock_dropped_final_zone_1': bool(info.get('rock_dropped_1')),
      'rock_dropped_final_zone_2': bool(info.get('rock_dropped_2')),
      'any_rock_contact_zone_1': bool(any_contact[1]),
      'any_rock_contact_zone_2': bool(any_contact[2]),
      'first_rock_contact_step_zone_1': first_contact[1],
      'first_rock_contact_step_zone_2': first_contact[2],
      'success': success, 'failure': failure,
      'timeout': bool(not success and not failure),
      'failure_zone': info.get('failure_zone'),
      'return': float(ret), 'ep_length': int(steps),
      'goal_xy': [float(goal_xy[0]), float(goal_xy[1])],
      'final_xy': [float(final_obs[0]), float(final_obs[1])],
      'nudges': int(teacher.nudges),
      'rock_jitter_zone_1': jitter[1].tolist(),
      'rock_jitter_zone_2': jitter[2].tolist(),
  }
  return row


def _sidecar_columns(rows, discarded, args, meta, traces):
  def col(key, dtype=None, none=None):
    values = [row[key] if row[key] is not None else none for row in rows]
    return np.asarray(values, dtype=dtype)

  def dcol(key, dtype=None, none=None):
    values = [row[key] if row[key] is not None else none
              for row in discarded]
    return np.asarray(values, dtype=dtype)

  return {
      'episode_id': np.arange(len(rows), dtype=np.int64),
      'draw_index': col('draw_index', np.int64),
      'env_seed': col('env_seed', np.int64),
      'reset_index': col('reset_index', np.int64),
      'u1': col('u1', bool), 'u2': col('u2', bool),
      'latent': col('latent', str),
      't0_1': col('t0_1', np.int64), 't0_2': col('t0_2', np.int64),
      'rockfall_start_1': col('rockfall_start_1', np.int64, -1),
      'rockfall_start_2': col('rockfall_start_2', np.int64, -1),
      'rockfall_end_1': col('rockfall_end_1', np.int64, -1),
      'rockfall_end_2': col('rockfall_end_2', np.int64, -1),
      'teacher_take_detour': col('teacher_take_detour', bool),
      'route': col('route', str),
      'route_realized': col('route_realized', str, 'none'),
      'decision_zone_1': col('decision_zone_1', str, 'none'),
      'decision_zone_2': col('decision_zone_2', str, 'none'),
      'wait_zone_1': col('wait_zone_1', bool),
      'wait_zone_2': col('wait_zone_2', bool),
      'go_direct_zone_1': col('go_direct_zone_1', bool),
      'go_direct_zone_2': col('go_direct_zone_2', bool),
      'hold_steps_zone_1': col('hold_steps_zone_1', np.int64),
      'hold_steps_zone_2': col('hold_steps_zone_2', np.int64),
      'release_step_zone_1': col('release_step_zone_1', np.int64, -1),
      'release_step_zone_2': col('release_step_zone_2', np.int64, -1),
      'mouth_step_zone_1': col('mouth_step_zone_1', np.int64, -1),
      'mouth_step_zone_2': col('mouth_step_zone_2', np.int64, -1),
      'band_entry_step_zone_1': col('band_entry_step_zone_1', np.int64, -1),
      'band_entry_step_zone_2': col('band_entry_step_zone_2', np.int64, -1),
      'band_exit_step_zone_1': col('band_exit_step_zone_1', np.int64, -1),
      'band_exit_step_zone_2': col('band_exit_step_zone_2', np.int64, -1),
      'entered_hazard_zone_1': col('entered_hazard_zone_1', bool),
      'entered_hazard_zone_2': col('entered_hazard_zone_2', bool),
      'rock_waves_zone_1': col('rock_waves_zone_1', np.int64),
      'rock_waves_zone_2': col('rock_waves_zone_2', np.int64),
      'rockfall_passed_zone_1': col('rockfall_passed_zone_1', bool),
      'rockfall_passed_zone_2': col('rockfall_passed_zone_2', bool),
      'rock_dropped_final_zone_1': col('rock_dropped_final_zone_1', bool),
      'rock_dropped_final_zone_2': col('rock_dropped_final_zone_2', bool),
      'any_rock_contact_zone_1': col('any_rock_contact_zone_1', bool),
      'any_rock_contact_zone_2': col('any_rock_contact_zone_2', bool),
      'first_rock_contact_step_zone_1': col(
          'first_rock_contact_step_zone_1', np.int64, -1),
      'first_rock_contact_step_zone_2': col(
          'first_rock_contact_step_zone_2', np.int64, -1),
      'success': col('success', bool), 'failure': col('failure', bool),
      'timeout': col('timeout', bool),
      'failure_zone': col('failure_zone', np.int8, -1),
      'ep_return': col('return', np.float64),
      'ep_length': col('ep_length', np.int64),
      'goal_xy': col('goal_xy', np.float32),
      'final_xy': col('final_xy', np.float32),
      'nudges': col('nudges', np.int64),
      'rock_jitter_zone_1': col('rock_jitter_zone_1', np.float32),
      'rock_jitter_zone_2': col('rock_jitter_zone_2', np.float32),
      **traces,
      'discard_draw_index': dcol('draw_index', np.int64),
      'discard_env_seed': dcol('env_seed', np.int64),
      'discard_reset_index': dcol('reset_index', np.int64),
      'discard_u1': dcol('u1', bool), 'discard_u2': dcol('u2', bool),
      'discard_t0_1': dcol('t0_1', np.int64),
      'discard_t0_2': dcol('t0_2', np.int64),
      'discard_teacher_take_detour': dcol('teacher_take_detour', bool),
      'discard_route': dcol('route', str),
      'discard_route_realized': dcol('route_realized', str, 'none'),
      'discard_success': dcol('success', bool),
      'discard_failure': dcol('failure', bool),
      'discard_timeout': dcol('timeout', bool),
      'discard_failure_zone': dcol('failure_zone', np.int8, -1),
      'discard_ep_length': dcol('ep_length', np.int64),
      'discard_final_xy': dcol('final_xy', np.float32).reshape((-1, 2)),
      'discard_reason': dcol('discard_reason', str),
      'discarded_json': np.asarray(json.dumps(discarded)),
      'seed': np.int64(args.seed),
      'collection_seed': np.int64(args.seed),
      'route_rng_seed': np.int64(args.seed + ROUTE_SEED_OFFSET),
      'environment_version': np.asarray(V6.ENV_VERSION),
      'p_active_1': np.float64(args.p_active_1),
      'p_active_2': np.float64(args.p_active_2),
      'teacher_detour_prob': np.float64(args.teacher_detour_prob),
      't0_range_1': np.asarray([args.t0_min_1, args.t0_max_1], np.int64),
      't0_range_2': np.asarray([args.t0_min_2, args.t0_max_2], np.int64),
      'horizon': np.int64(args.horizon),
      'meta': np.asarray(json.dumps(meta)),
  }


def collect(args):
  _validate_args(args)
  os.makedirs(args.out_dir, exist_ok=True)
  cfg, teacher = CT.make_teacher(
      horizon=args.horizon,
      p_active_1=args.p_active_1,
      p_active_2=args.p_active_2,
      t0_range_1=(args.t0_min_1, args.t0_max_1),
      t0_range_2=(args.t0_min_2, args.t0_max_2))
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=args.seed)
  route_rng = np.random.default_rng(args.seed + ROUTE_SEED_OFFSET)

  n, length = args.episodes, args.horizon + 1
  obs = np.zeros((n, length, RAW_OBS_DIM), np.float32)
  act = np.zeros((n, length, ACTION_DIM), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  torso_x = np.full((n, length), np.nan, np.float32)
  torso_y = np.full((n, length), np.nan, np.float32)
  holding_zone = np.full((n, length), -1, np.int8)
  rockfall_open_1 = np.zeros((n, length), bool)
  rockfall_open_2 = np.zeros((n, length), bool)

  kept = []
  discarded = []
  all_draws = []
  # There is deliberately no acceptance/rejection step.  A natural reset is
  # one dataset episode even if the controller stalls, fails, or never reaches
  # the intended route.  This preserves the independence of the three reset-
  # time coins in the actual learner population rather than only in a pre-
  # selection population.
  for episode in range(n):
    draw_index = episode
    take_detour = bool(route_rng.random() < args.teacher_detour_prob)
    route = 'detour' if take_detour else 'shortcut'

    # Natural reset is essential: the env, not this collector, samples U1/U2
    # and both absolute clocks from its six independent fixed-order streams.
    o = env.reset()
    if np.asarray(o).shape != (RAW_OBS_DIM,):
      raise RuntimeError(f'expected raw V6 observation (58,), got {o.shape}')
    t0 = {zone: _sampled_t0(env, zone) for zone in (1, 2)}
    jitter = {zone: _capture_jitter(env, zone) for zone in (1, 2)}
    teacher.fresh(route=route)
    obs[episode, 0] = o
    torso_x[episode, 0], torso_y[episode, 0] = o[0], o[1]
    eval_goals[episode] = o[STATE_DIM:STATE_DIM + 2]
    ret = 0.0
    info = env._info(False)  # privileged audit snapshot, not learner data
    any_contact = {1: False, 2: False}
    first_contact = {1: None, 2: None}
    exit_step = {1: None, 2: None}
    entered_once = {1: False, 2: False}

    for step in range(args.horizon):
      action = teacher.act(o, env.schedule)
      was_holding = teacher.holding_zone
      o, reward, done, info = env.step(action)
      act[episode, step] = np.asarray(action, np.float32)
      obs[episode, step + 1] = o
      torso_x[episode, step + 1], torso_y[episode, step + 1] = o[0], o[1]
      holding_zone[episode, step] = (-1 if was_holding is None
                                     else int(was_holding))
      rockfall_open_1[episode, step + 1] = bool(
          info.get('rockfall_open_1'))
      rockfall_open_2[episode, step + 1] = bool(
          info.get('rockfall_open_2'))
      for zone in (1, 2):
        contact = bool(info.get(f'rock_contact_{zone}'))
        any_contact[zone] = any_contact[zone] or contact
        if contact and first_contact[zone] is None:
          first_contact[zone] = step + 1
        entered_once[zone] = (entered_once[zone]
                              or bool(info.get(f'entered_hazard_{zone}')))
        if (entered_once[zone] and exit_step[zone] is None
            and float(o[0]) > V6.HAZARD_X[zone][1]
            and abs(float(o[1])) < V6.HAZARD_HALF_Y):
          exit_step[zone] = step + 1
      ret += float(reward)
      if done or reward > 0:
        break

    steps = step + 1
    row = _row_from_episode(
        env, teacher, route, draw_index, eval_goals[episode], info, steps,
        ret, o, t0, jitter, any_contact, first_contact, exit_step)
    row['episode_id'] = int(episode)
    row['discard_reason'] = None
    lengths[episode] = steps + 1
    kept.append(row)
    all_draws.append(row)
    if (episode + 1) % args.progress_every == 0 or episode + 1 == n:
      print(f'  collected {episode + 1}/{n} | no rejection sampling',
            flush=True)

  raw_path = os.path.join(args.out_dir, f'{args.name}.npz')
  xy_path = os.path.join(args.out_dir, f'{args.name}_gxy.npz')
  sidecar_path = os.path.join(args.out_dir, f'{args.name}_sidecar.npz')
  audit_path = os.path.join(
      args.out_dir, f'{args.name}_composition_audit.json')

  n_transitions = int(np.sum(lengths - 1))
  common_meta = {
      'benchmark': 'long rectangular AntMaze with two independent rockfalls',
      'environment_version': V6.ENV_VERSION,
      'source_env': CT.ENV_NAME,
      'action_dim': ACTION_DIM,
      'horizon': args.horizon,
      'n_episodes': n,
      'n_transitions': n_transitions,
      'seed': args.seed,
      'collection_seed': args.seed,
      'route_rng_seed': args.seed + ROUTE_SEED_OFFSET,
      'randomness': {
          'natural_environment_reset': True,
          'hazards': 'U1 and U2 sampled by distinct env RNG streams',
          'schedules': 't0_1 and t0_2 sampled by distinct env RNG streams',
          'rock_jitter': 'one distinct env RNG stream per zone',
          'teacher_route': 'independent collector RNG stream',
          'fixed_order_consumption_on_every_reset': True,
          'stream_seeds': {
              'ant_reset': args.seed,
              'goal': args.seed + 777,
              'u1': args.seed + V6._ACTIVE_SEED_OFFSET_1,
              'u2': args.seed + V6._ACTIVE_SEED_OFFSET_2,
              't0_1': args.seed + V6._SCHED_SEED_OFFSET_1,
              't0_2': args.seed + V6._SCHED_SEED_OFFSET_2,
              'rock_jitter_1': args.seed + V6._JITTER_SEED_OFFSET_1,
              'rock_jitter_2': args.seed + V6._JITTER_SEED_OFFSET_2,
              'teacher_take_detour': args.seed + ROUTE_SEED_OFFSET,
          },
      },
      'p_active_1': args.p_active_1,
      'p_active_2': args.p_active_2,
      'teacher_detour_prob': args.teacher_detour_prob,
      'timing': {
          't0_range_1_inclusive': [args.t0_min_1, args.t0_max_1],
          't0_range_2_inclusive': [args.t0_min_2, args.t0_max_2],
          'rockfall_steps': V6.ROCKFALL_STEPS,
          'wave_period': V6.WAVE_PERIOD,
          'teacher_crossing_steps': dict(CT.CROSSING_STEPS),
          'teacher_safety_margin': CT.SAFETY_MARGIN,
          'teacher_release_margin': CT.RELEASE_MARGIN,
      },
      'geometry': V6.geometry_metadata(),
      'teacher': {
          'class': type(teacher).__name__,
          'walker_controller': dict(teacher.walker_provenance),
          'route_rule': 'independent detour coin at episode start; otherwise '
                        'shortcut with a separate prospective timetable '
                        'decision at each zone',
          'hold_action': 'zero torque',
          'detour_turn_bias': CT.DETOUR_TURN_BIAS,
          'detour_turn_bias_steps': CT.DETOUR_TURN_BIAS_STEPS,
          'privileged_schedule_never_in_learner_observation': True,
      },
      'outcomes': {
          'success': _rate([row['success'] for row in kept]),
          'failure': _rate([row['failure'] for row in kept]),
          'timeout': _rate([row['timeout'] for row in kept]),
      },
      'discard': {
          'n_draws': len(all_draws), 'n_discarded': len(discarded),
          'discard_fraction': round(len(discarded) / len(all_draws), 6),
          'selection_rule': 'none; every natural draw is retained',
          'details_location': os.path.basename(sidecar_path),
      },
      'privileged_episode_metadata': os.path.basename(sidecar_path),
      'composition_audit': os.path.basename(audit_path),
      'failure_bank': False,
  }
  raw_meta = {
      **common_meta,
      'name': args.name,
      'env': CT.ENV_NAME,
      'env_name': CT.ENV_NAME,
      'obs_dim': STATE_DIM,
      'goal_dim': RAW_GOAL_DIM,
      'goal_rep': 'full_zero_padded_xy',
      'observation_width': RAW_OBS_DIM,
  }
  np.savez_compressed(
      raw_path, obs=obs, act=act, lengths=lengths,
      eval_goals=eval_goals, meta=json.dumps(raw_meta))

  if not np.all(obs[:, :, STATE_DIM + XY_GOAL_DIM:] == 0.0):
    raise RuntimeError('raw goal padding is not zero; cannot derive XY copy')
  xy_obs = np.ascontiguousarray(obs[:, :, :XY_OBS_DIM])
  if not np.array_equal(xy_obs[:, :, :STATE_DIM], obs[:, :, :STATE_DIM]):
    raise RuntimeError('XY derivation changed state columns')
  xy_meta = {
      **common_meta,
      'name': f'{args.name}_gxy',
      'env': f'{CT.ENV_NAME}_gxy',
      'env_name': f'{CT.ENV_NAME}_gxy',
      'obs_dim': STATE_DIM,
      'goal_dim': XY_GOAL_DIM,
      'goal_rep': 'xy',
      'observation_width': XY_OBS_DIM,
      'derived_from': os.path.basename(raw_path),
      'derivation': 'direct column selection state[0:29] + goal[0:2]; '
                    'no re-simulation',
  }
  np.savez_compressed(
      xy_path, obs=xy_obs, act=act, lengths=lengths,
      eval_goals=eval_goals, meta=json.dumps(xy_meta))

  traces = {
      'step_torso_x': torso_x, 'step_torso_y': torso_y,
      'step_holding_zone': holding_zone,
      'step_rockfall_open_1': rockfall_open_1,
      'step_rockfall_open_2': rockfall_open_2,
  }
  sidecar_meta = {
      **common_meta,
      'name': f'{args.name}_sidecar',
      'learner_input': False,
      'encoding': {
          'none_integer': -1,
          'none_string': 'none',
          'step_trace_padding': 'NaN for XY, -1 for holding zone, false for '
                                'rockfall-open flags',
      },
  }
  np.savez_compressed(
      sidecar_path,
      **_sidecar_columns(kept, discarded, args, sidecar_meta, traces))

  path_info = {
      'raw': raw_path, 'xy': xy_path, 'sidecar': sidecar_path,
      'composition_audit': audit_path,
      'raw_sha256': _sha256(raw_path),
      'xy_sha256': _sha256(xy_path),
      'sidecar_sha256': _sha256(sidecar_path),
      'raw_obs_shape': list(obs.shape), 'xy_obs_shape': list(xy_obs.shape),
      'act_shape': list(act.shape), 'n_transitions': n_transitions,
  }
  composition = _composition_report(
      kept, all_draws, discarded, args, path_info)
  with open(audit_path, 'w', encoding='utf-8') as handle:
    json.dump(composition, handle, indent=2)

  print(json.dumps({
      'n_draws': len(all_draws), 'n_kept': len(kept),
      'n_discarded': len(discarded), 'n_transitions': n_transitions,
      'success': composition['kept']['outcomes']['success'],
      'failure': composition['kept']['outcomes']['failure'],
      'timeout': composition['kept']['outcomes']['timeout'],
      'latent_counts': {
          key: block['n'] for key, block in
          composition['kept']['latent']['by_combination'].items()},
      'detour_fraction':
          composition['kept']['teacher_route']['detour_fraction'],
      'composition_all_pass': composition['all_pass'],
  }, indent=2), flush=True)
  print('->', raw_path)
  print('->', xy_path)
  print('->', sidecar_path)
  print('->', audit_path, flush=True)
  return composition


def build_parser():
  parser = argparse.ArgumentParser(
      description=__doc__.splitlines()[0],
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--episodes', type=int, default=1000,
                      help='number of kept demonstrator episodes')
  parser.add_argument('--seed', type=int, default=606,
                      help='environment/collection root seed')
  parser.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1,
                      help='zone-1 Bernoulli hazard probability')
  parser.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2,
                      help='zone-2 Bernoulli hazard probability')
  parser.add_argument('--teacher-detour-prob', type=float,
                      default=CT.TEACHER_DETOUR_PROB,
                      help='independent probability of the long safe route')
  parser.add_argument('--t0-min-1', type=int, default=V6.T0_MIN_1,
                      help='inclusive zone-1 clock lower endpoint')
  parser.add_argument('--t0-max-1', type=int, default=V6.T0_MAX_1,
                      help='inclusive zone-1 clock upper endpoint')
  parser.add_argument('--t0-min-2', type=int, default=V6.T0_MIN_2,
                      help='inclusive zone-2 clock lower endpoint')
  parser.add_argument('--t0-max-2', type=int, default=V6.T0_MAX_2,
                      help='inclusive zone-2 clock upper endpoint')
  parser.add_argument('--horizon', type=int, default=CT.HORIZON,
                      help='external episode horizon used for collection')
  parser.add_argument('--out-dir', default=OUT_DIR,
                      help='directory for both learner files and audits')
  parser.add_argument('--name', default=DATASET_NAME,
                      help='output file stem (default: %(default)s)')
  parser.add_argument('--progress-every', type=int, default=50)
  return parser


def main():
  args = build_parser().parse_args()
  _validate_args(args)
  composition = collect(args)
  raise SystemExit(0 if composition['all_pass'] else 1)


if __name__ == '__main__':
  main()
