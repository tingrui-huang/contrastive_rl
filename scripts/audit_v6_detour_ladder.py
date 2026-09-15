"""Cross-rung audit of the V6 teacher-detour dataset ladder. Read-only.

The ladder is one dataset per teacher detour probability (0.05, 0.10, ...,
0.30), all collected with the same seed on the same p_active 0.40 benchmark.
The collector draws the route coin as ``route_rng.random() < p`` from a
stream that is separate from the environment's reset streams, and the
teacher is deterministic, so three things must hold between any two rungs:

  * NESTED   every detour episode of a lower rung is a detour episode of
             every higher rung (same uniform draw, larger threshold);
  * SHARED   every episode that is a shortcut on both rungs is byte-identical
             on both (observations, actions, length, goal);
  * SAME ENV every episode has the same U1/U2/t0/reset pose on every rung --
             the route coin never touches the environment streams.

Each rung must also pass its own composition audit and report the configured
probability inside the audit's realized Wilson99 interval.  The report is
written next to the datasets so a training gate can cite it.

Usage::

  python scripts/audit_v6_detour_ladder.py
  python scripts/audit_v6_detour_ladder.py --rungs 0.05 0.15 0.30
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.offline_audit import sha256_file                   # noqa: E402
import train_rockfall_clock_v6_baseline as B                # noqa: E402

REPORT = os.path.join(B.DATASET_DIR, 'detour_ladder_audit.json')
STATE_DIM = 29
#: sidecar columns that must be identical episode-for-episode across rungs
ENV_COLUMNS = ('u1', 'u2', 't0_1', 't0_2', 'env_seed', 'reset_index')


def _rung_paths(detour_prob):
  xy = B._default_dataset(B.ENV_XY, detour_prob)   # pylint: disable=protected-access
  raw = B._default_dataset(B.ENV_BASE, detour_prob)  # pylint: disable=protected-access
  stem = raw[:-len('.npz')]
  return {'raw': raw, 'xy': xy, 'sidecar': stem + '_sidecar.npz',
          'audit': stem + '_composition_audit.json'}


def load_rung(detour_prob):
  paths = _rung_paths(detour_prob)
  for path in paths.values():
    if not os.path.isfile(path):
      raise FileNotFoundError(f'rung {detour_prob:g}: missing {path}')
  with np.load(paths['raw'], allow_pickle=False) as d:
    obs = np.asarray(d['obs'])
    act = np.asarray(d['act'])
    lengths = np.asarray(d['lengths'])
    goals = np.asarray(d['eval_goals'])
    meta = json.loads(str(d['meta']))
  with np.load(paths['xy'], allow_pickle=False) as d:
    obs_xy = np.asarray(d['obs'])
    xy_meta = json.loads(str(d['meta']))
  with np.load(paths['sidecar'], allow_pickle=False) as d:
    side = {k: np.asarray(d[k]) for k in ENV_COLUMNS + ('route',)}
  with open(paths['audit'], encoding='utf-8') as f:
    audit = json.load(f)
  return {'p': float(detour_prob), 'paths': paths, 'obs': obs, 'act': act,
          'lengths': lengths, 'goals': goals, 'meta': meta,
          'xy_meta': xy_meta, 'obs_xy': obs_xy, 'side': side,
          'audit': audit, 'detour': side['route'] == 'detour'}


def _episode_equal(a, b, i):
  return (a['lengths'][i] == b['lengths'][i]
          and np.array_equal(a['obs'][i], b['obs'][i])
          and np.array_equal(a['act'][i], b['act'][i])
          and np.array_equal(a['goals'][i], b['goals'][i]))


def audit_rung(rung):
  """Per-rung facts and gates, all from files the trainer itself checks."""
  audit = rung['audit']
  route = audit['kept']['teacher_route']
  meta = rung['meta']
  n = int(meta['n_episodes'])
  lo, hi = route['detour_wilson99']
  gates = {
      'composition_audit_all_pass': audit.get('all_pass') is True,
      'meta_detour_prob_matches_rung': bool(np.isclose(
          float(meta['teacher_detour_prob']), rung['p'], atol=1e-12)),
      'audit_configured_prob_matches_rung': bool(np.isclose(
          float(route['configured_detour_probability']), rung['p'],
          atol=1e-12)),
      'configured_prob_in_realized_wilson99': lo <= rung['p'] <= hi,
      'sidecar_detour_count_matches_audit':
          int(rung['detour'].sum()) == int(route['detour_n']),
      'raw_sha_matches_audit': sha256_file(rung['paths']['raw'])
          == audit['dataset_paths']['raw_sha256'],
      'xy_sha_matches_audit': sha256_file(rung['paths']['xy'])
          == audit['dataset_paths']['xy_sha256'],
      'xy_is_column_selection_of_raw': bool(np.array_equal(
          rung['obs_xy'],
          np.concatenate([rung['obs'][..., :STATE_DIM],
                          rung['obs'][..., STATE_DIM:STATE_DIM + 2]],
                         axis=-1))),
      'same_benchmark_as_default_rung': (
          float(meta['p_active_1']) == 0.4 and float(meta['p_active_2']) == 0.4
          and int(meta['collection_seed']) == 606 and n == 1000
          and int(meta['horizon']) == 800),
  }
  return {
      'teacher_detour_prob': rung['p'],
      'files': {k: os.path.basename(v) for k, v in rung['paths'].items()},
      'n_episodes': n,
      'detour_n': int(route['detour_n']),
      'detour_fraction': float(route['detour_fraction']),
      'detour_wilson99': [float(lo), float(hi)],
      'shortcut_n': n - int(route['detour_n']),
      'outcomes': audit['kept']['outcomes'],
      'n_transitions': int(meta['n_transitions']),
      'gates': gates,
  }


def audit_pair(low, high):
  """Cross-rung gates between a lower and a higher detour probability."""
  n = len(low['lengths'])
  nested = bool(np.all(~low['detour'] | high['detour']))
  both_shortcut = np.flatnonzero(~low['detour'] & ~high['detour'])
  both_detour = np.flatnonzero(low['detour'] & high['detour'])
  promoted = np.flatnonzero(~low['detour'] & high['detour'])
  shared_equal = [int(i) for i in both_shortcut
                  if not _episode_equal(low, high, i)]
  detour_equal = [int(i) for i in both_detour
                  if not _episode_equal(low, high, i)]
  # a promoted episode must actually differ (the teacher walked a different
  # route) yet start from the identical reset observation
  promoted_same_traj = [int(i) for i in promoted
                        if _episode_equal(low, high, i)]
  promoted_reset_diff = [int(i) for i in promoted
                         if not np.array_equal(low['obs'][i, 0],
                                               high['obs'][i, 0])]
  env_same = {c: bool(np.array_equal(low['side'][c], high['side'][c]))
              for c in ENV_COLUMNS}
  gates = {
      'detour_sets_nested': nested,
      'shared_shortcut_episodes_byte_identical': not shared_equal,
      'shared_detour_episodes_byte_identical': not detour_equal,
      'promoted_episodes_change_trajectory': not promoted_same_traj,
      'promoted_episodes_keep_reset_observation': not promoted_reset_diff,
      'environment_draws_identical': all(env_same.values()),
  }
  return {
      'low': low['p'], 'high': high['p'], 'n_episodes': n,
      'both_shortcut': int(len(both_shortcut)),
      'both_detour': int(len(both_detour)),
      'promoted_to_detour': int(len(promoted)),
      'demoted_to_shortcut': int(np.sum(low['detour'] & ~high['detour'])),
      'shared_shortcut_mismatches': shared_equal[:20],
      'shared_detour_mismatches': detour_equal[:20],
      'promoted_unchanged': promoted_same_traj[:20],
      'promoted_reset_mismatch': promoted_reset_diff[:20],
      'environment_columns_identical': env_same,
      'gates': gates,
  }


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--rungs', type=float, nargs='+', default=B.DETOUR_LADDER)
  ap.add_argument('--out', default=REPORT)
  args = ap.parse_args(argv)
  rungs = sorted(float(r) for r in args.rungs)
  if len(rungs) < 2:
    raise SystemExit('need at least two rungs')

  print('loading', ', '.join(f'{r:g}' for r in rungs))
  loaded = [load_rung(r) for r in rungs]
  per_rung = [audit_rung(r) for r in loaded]
  pairs = [audit_pair(loaded[i], loaded[j])
           for i in range(len(loaded)) for j in range(i + 1, len(loaded))]

  all_pass = (all(all(r['gates'].values()) for r in per_rung)
              and all(all(p['gates'].values()) for p in pairs))
  report = {
      'ladder': rungs,
      'benchmark': {'p_active_1': 0.4, 'p_active_2': 0.4,
                    'collection_seed': 606, 'episodes': 1000, 'horizon': 800},
      'rule': ('per-episode route coin route_rng.random() < p on a stream '
               'separate from the six environment reset streams; the '
               'deterministic teacher walks the shortcut unless the coin '
               'selects the long perimeter route'),
      'rungs': per_rung,
      'pairs': pairs,
      'all_pass': all_pass,
  }
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

  print(f'{"rung":>6} {"detour_n":>9} {"frac":>7} {"wilson99":>18} '
        f'{"shortcut":>9} {"success":>8} {"gates":>6}')
  for r in per_rung:
    print(f'{r["teacher_detour_prob"]:>6g} {r["detour_n"]:>9d} '
          f'{r["detour_fraction"]:>7.3f} '
          f'[{r["detour_wilson99"][0]:.3f}, {r["detour_wilson99"][1]:.3f}] '
          f'{r["shortcut_n"]:>9d} {r["outcomes"]["success"]:>8.3f} '
          f'{"PASS" if all(r["gates"].values()) else "FAIL":>6}')
  print()
  print(f'{"pair":>12} {"shared_sc":>10} {"promoted":>9} {"gates":>6}')
  for p in pairs:
    print(f'{p["low"]:>5g}->{p["high"]:<5g} {p["both_shortcut"]:>10d} '
          f'{p["promoted_to_detour"]:>9d} '
          f'{"PASS" if all(p["gates"].values()) else "FAIL":>6}')
  failed = ([f'rung {r["teacher_detour_prob"]:g}: {g}' for r in per_rung
             for g, ok in r['gates'].items() if not ok]
            + [f'pair {p["low"]:g}->{p["high"]:g}: {g}' for p in pairs
               for g, ok in p['gates'].items() if not ok])
  for line in failed:
    print('  FAILED', line)
  print(f'-> {args.out}')
  print('LADDER AUDIT', 'PASS' if all_pass else 'FAIL')
  return 0 if all_pass else 1


if __name__ == '__main__':
  sys.exit(main())
