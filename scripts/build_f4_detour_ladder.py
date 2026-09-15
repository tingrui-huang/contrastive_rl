"""Build, pin and audit the F4 teacher-detour dataset ladder (PointMaze).

One merged F4 dataset per teacher ``force_safe_prob`` -- the probability that
a teacher episode takes the always-safe lower route instead of heading for
the shortcut -- everything else held at the p=0.3 recipe:

    teacher   6000 episodes, random_frac 0.2, teacher_noise 0.15, seed 0
    bad demo   600 episodes, seed 0                     (shared by every rung)
    merged    scripts/merge_swamp_windy_baddemo.py      (6600 episodes)

The 0.05 rung IS the existing ``datasets/swamp_windy_f4_merged_s0.npz`` (its
content sha is pinned here, and rebuilding it reproduces that sha); the other
rungs are ``datasets/swamp_windy_f4_far{pp}_merged_s0.npz``.  ``datasets/``
is gitignored on this branch, so what the repository carries is the ladder
manifest with every rung's content sha, and any node rebuilds a rung with
``build`` and is refused by the launcher if the sha does not match.

WHAT IS AND IS NOT SHARED BETWEEN RUNGS.  The collector draws the route coin,
the random-episode actions and the per-step teacher noise from ONE stream,
and the environment draws its wind bits and action noise from one stream
whose consumption depends on when an episode dies.  So the 1200 random
episodes (which precede every coin) and the 600 bad-demo episodes are
byte-identical on every rung, while the 4800 teacher episodes are an
independent re-roll of the same process on every rung: the rungs are NOT
nested and shared shortcut episodes are NOT paired, unlike the AntMaze V6
ladder.  ``audit`` measures exactly this.

Usage::

  python scripts/build_f4_detour_ladder.py build --rungs 0.15
  python scripts/build_f4_detour_ladder.py build            # every missing rung
  python scripts/build_f4_detour_ladder.py manifest         # pin content shas
  python scripts/build_f4_detour_ladder.py audit            # cross-rung audit
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from scripts.make_swamp_f4_failure_bank import content_sha   # noqa: E402

RUNGS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_RUNG = 0.05
DATASET_DIR = 'datasets'
LADDER_DIR = 'artifacts/swamp_windy_f4_detour_ladder'
MANIFEST = LADDER_DIR + '/ladder_manifest.json'
AUDIT = LADDER_DIR + '/ladder_audit.json'
BANK = 'artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz'
ENV = 'point_two_route_swamp_windy_f4_v0'
# the p=0.3 F4 recipe, verbatim from scripts/run_f4_failneg_sweep.sh
TEACHER_EPISODES = 6000
BAD_EPISODES = 600
RANDOM_FRAC = 0.2
TEACHER_NOISE = 0.15
SEED = 0
N_RANDOM = int(round(TEACHER_EPISODES * RANDOM_FRAC))
N_TEACHER = TEACHER_EPISODES - N_RANDOM
MODE_RANDOM, MODE_FORCED_SAFE, MODE_BAD = 0, 1, 4
ROUTE_SAFE = 2


def rung_tag(p):
  """0.05 -> 'far05', 0.3 -> 'far30'."""
  return 'far%02d' % int(round(float(p) * 100))


def is_default(p):
  return abs(float(p) - DEFAULT_RUNG) < 1e-12


def rung_paths(p):
  """Repo-relative POSIX paths, so the manifest reads the same on every OS."""
  stem = ('swamp_windy_f4' if is_default(p)
          else 'swamp_windy_f4_%s' % rung_tag(p))
  return {
      'teacher': '%s/%s_teacher_s0.npz' % (DATASET_DIR, stem),
      'bad': '%s/swamp_windy_f4_baddemo_s0.npz' % DATASET_DIR,
      'merged': '%s/%s_merged_s0.npz' % (DATASET_DIR, stem),
  }


def merged_path(p):
  return rung_paths(p)['merged']


def parse_rungs(values):
  rungs = []
  for v in values:
    p = float(v)
    if not any(abs(p - r) < 1e-12 for r in RUNGS):
      raise SystemExit('rung %g is not on the ladder %s' % (p, list(RUNGS)))
    rungs.append(p)
  return rungs


def _run(cmd):
  print('  $', ' '.join(cmd), flush=True)
  subprocess.check_call(cmd, cwd=_ROOT)


def build(p, force=False):
  paths = rung_paths(p)
  py = sys.executable
  if force:
    for key in ('teacher', 'merged'):
      if os.path.exists(paths[key]):
        os.chmod(paths[key], 0o644)
        os.remove(paths[key])
  if not os.path.exists(paths['bad']):
    _run([py, '-m', 'scripts.collect_swamp_windy_f4', '--mode', 'baddemo',
          '--episodes', str(BAD_EPISODES), '--seed', str(SEED),
          '--out', paths['bad']])
  if not os.path.exists(paths['teacher']):
    _run([py, '-m', 'scripts.collect_swamp_windy_f4', '--mode', 'teacher',
          '--episodes', str(TEACHER_EPISODES),
          '--random_frac', str(RANDOM_FRAC),
          '--force_safe_prob', '%g' % p,
          '--teacher_noise', str(TEACHER_NOISE), '--seed', str(SEED),
          '--out', paths['teacher']])
  if not os.path.exists(paths['merged']):
    _run([py, '-m', 'scripts.merge_swamp_windy_baddemo',
          '--main', paths['teacher'], '--bad', paths['bad'],
          '--out', paths['merged']])
  print('rung %g -> %s  content %s' % (p, paths['merged'],
                                       content_sha(paths['merged'])))


def _wilson(k, n, z=2.5758293035489004):
  if n == 0:
    return [None, None]
  phat = k / n
  denom = 1.0 + z * z / n
  centre = (phat + z * z / (2 * n)) / denom
  half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
  return [centre - half, centre + half]


def describe(p):
  """Composition facts of one rung, read from the merged file only."""
  path = merged_path(p)
  with np.load(path, allow_pickle=False) as d:
    meta = json.loads(str(d['meta']))
    mode = np.asarray(d['teacher_mode'])
    route = np.asarray(d['route_label'])
    fsafe = np.asarray(d['force_safe']).astype(bool)
    died = np.asarray(d['entered_active_swamp']).astype(bool)
    n, rows, width = d['obs'].shape
  teacher = (mode != MODE_RANDOM) & (mode != MODE_BAD)
  n_forced = int(fsafe.sum())
  return {
      'force_safe_prob': float(p),
      'merged': path,
      'content_sha256': content_sha(path),
      'teacher_content_sha256': meta['merge']['main_content_sha256'],
      'bad_content_sha256': meta['merge']['bad_content_sha256'],
      'meta_force_safe_prob': float(meta['force_safe_prob']),
      'per_cell_swamp_prob': float(meta['per_cell_swamp_prob']),
      'env_name': meta['env_name'],
      'episodes': int(n), 'rows': int(rows), 'obs_width': int(width),
      'n_random': int((mode == MODE_RANDOM).sum()),
      'n_bad_demo': int((mode == MODE_BAD).sum()),
      'n_teacher': int(teacher.sum()),
      'n_forced_safe': n_forced,
      'forced_safe_fraction_of_teacher': n_forced / int(teacher.sum()),
      'forced_safe_fraction_of_all': n_forced / int(n),
      'forced_safe_wilson99_of_teacher': _wilson(n_forced, int(teacher.sum())),
      'n_route_safe_detour': int((route == ROUTE_SAFE).sum()),
      'teacher_mode_counts': {str(int(m)): int(c) for m, c in
                              zip(*np.unique(mode, return_counts=True))},
      'died_rate_overall': float(died.mean()),
      'died_rate_teacher': float(died[teacher].mean()),
      'died_rate_forced_safe': (float(died[fsafe].mean())
                                if n_forced else None),
      'reached_0p5_rate': float(meta.get('reached_0p5_rate')),
  }


def write_manifest(rungs):
  os.makedirs(LADDER_DIR, exist_ok=True)
  entries = {}
  for p in rungs:
    if not os.path.exists(merged_path(p)):
      raise SystemExit('rung %g not built: %s' % (p, merged_path(p)))
    entries[rung_tag(p)] = describe(p)
  bank = None
  if os.path.exists(BANK):
    with np.load(BANK, allow_pickle=False) as b:
      bmeta = json.loads(str(b['meta']))
    bank = {'path': BANK, 'content_sha256': content_sha(BANK),
            'source_content_sha256': bmeta.get('source_content_sha256'),
            'built_from_rung': next(
                (t for t, e in entries.items()
                 if e['content_sha256'] == bmeta.get('source_content_sha256')),
                None)}
  manifest = {
      'ladder': [float(p) for p in rungs],
      'default_rung': DEFAULT_RUNG,
      'env_name': ENV,
      'recipe': {'teacher_episodes': TEACHER_EPISODES,
                 'bad_demo_episodes': BAD_EPISODES,
                 'random_frac': RANDOM_FRAC, 'teacher_noise': TEACHER_NOISE,
                 'seed': SEED,
                 'force_safe_prob': 'the only value that differs per rung; '
                                    'applies to the %d teacher episodes '
                                    '(not the random or bad-demo ones)'
                                    % N_TEACHER},
      'canonical_bank': bank,
      'rungs': entries,
  }
  with open(MANIFEST, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
  print('-> %s' % MANIFEST)
  print('%6s %10s %8s %8s %18s %8s %8s' % (
      'rung', 'forced', 'of_tchr', 'of_all', 'wilson99(teacher)', 'died',
      'reached'))
  for t, e in entries.items():
    lo, hi = e['forced_safe_wilson99_of_teacher']
    print('%6g %10d %8.3f %8.3f   [%.3f, %.3f] %8.3f %8.3f' % (
        e['force_safe_prob'], e['n_forced_safe'],
        e['forced_safe_fraction_of_teacher'], e['forced_safe_fraction_of_all'],
        lo, hi, e['died_rate_overall'], e['reached_0p5_rate']))
  return manifest


def load_manifest():
  if not os.path.exists(MANIFEST):
    raise SystemExit('ladder manifest missing: %s (run `manifest`)' % MANIFEST)
  with open(MANIFEST, encoding='utf-8') as f:
    return json.load(f)


def audit(rungs):
  manifest = load_manifest()
  loaded = {}
  for p in rungs:
    path = merged_path(p)
    with np.load(path, allow_pickle=False) as d:
      loaded[p] = {k: np.asarray(d[k]) for k in
                   ('obs', 'act', 'swamp_bits', 'teacher_mode', 'route_label',
                    'force_safe', 'entered_active_swamp')}
    loaded[p]['sha'] = content_sha(path)
  per_rung = {}
  for p in rungs:
    e = manifest['rungs'].get(rung_tag(p))
    d = describe(p)
    lo, hi = d['forced_safe_wilson99_of_teacher']
    mode = loaded[p]['teacher_mode']
    gates = {
        'in_manifest': e is not None,
        'content_sha_matches_manifest':
            e is not None and e['content_sha256'] == loaded[p]['sha'],
        'meta_force_safe_prob_matches_rung':
            abs(d['meta_force_safe_prob'] - p) < 1e-12,
        'env_is_f4_p030': d['env_name'] == ENV
                          and abs(d['per_cell_swamp_prob'] - 0.3) < 1e-12,
        'block_sizes_1200_4800_600':
            d['n_random'] == N_RANDOM and d['n_teacher'] == N_TEACHER
            and d['n_bad_demo'] == BAD_EPISODES,
        'random_block_first_then_teacher_then_bad':
            bool(np.all(mode[:N_RANDOM] == MODE_RANDOM))
            and bool(np.all(mode[TEACHER_EPISODES:] == MODE_BAD)),
        'configured_prob_in_realized_wilson99': lo <= p <= hi,
        'forced_safe_equals_safe_detour_route':
            d['n_forced_safe'] == d['n_route_safe_detour'],
    }
    per_rung[rung_tag(p)] = {**d, 'gates': gates}

  def block_equal(a, b, sl):
    return all(np.array_equal(a[k][sl], b[k][sl])
               for k in ('obs', 'act', 'swamp_bits', 'entered_active_swamp'))

  pairs = []
  ordered = sorted(rungs)
  for i in range(len(ordered)):
    for j in range(i + 1, len(ordered)):
      lo_p, hi_p = ordered[i], ordered[j]
      a, b = loaded[lo_p], loaded[hi_p]
      teacher = slice(N_RANDOM, TEACHER_EPISODES)
      same_teacher = [int(k) for k in range(N_RANDOM, TEACHER_EPISODES)
                      if np.array_equal(a['obs'][k], b['obs'][k])
                      and np.array_equal(a['act'][k], b['act'][k])]
      fa = a['force_safe'].astype(bool)[teacher]
      fb = b['force_safe'].astype(bool)[teacher]
      gates = {
          'random_block_byte_identical':
              block_equal(a, b, slice(0, N_RANDOM)),
          'bad_demo_block_byte_identical':
              block_equal(a, b, slice(TEACHER_EPISODES, None)),
          'forced_safe_count_increases': int(fb.sum()) > int(fa.sum()),
      }
      pairs.append({
          'low': lo_p, 'high': hi_p,
          'identical_teacher_episodes': len(same_teacher),
          'identical_teacher_episodes_prefix_note':
              'teacher episodes stay identical only until the first episode '
              'whose coin lands between the two probabilities; after that '
              'the shared noise stream diverges',
          'first_diverging_teacher_episode':
              (None if len(same_teacher) == N_TEACHER
               else int(min(set(range(N_RANDOM, TEACHER_EPISODES))
                            - set(same_teacher)))),
          'forced_safe_low': int(fa.sum()), 'forced_safe_high': int(fb.sum()),
          'nested_forced_safe_sets': bool(np.all(~fa | fb)),
          'gates': gates,
      })
  all_pass = (all(all(r['gates'].values()) for r in per_rung.values())
              and all(all(q['gates'].values()) for q in pairs))
  report = {'ladder': ordered, 'rungs': per_rung, 'pairs': pairs,
            'sharing': ('random block (episodes 0-%d) and bad-demo block '
                        '(episodes %d-%d) are byte-identical on every rung; '
                        'teacher episodes are an independent re-roll per '
                        'rung' % (N_RANDOM - 1, TEACHER_EPISODES,
                                  TEACHER_EPISODES + BAD_EPISODES - 1)),
            'all_pass': all_pass}
  os.makedirs(LADDER_DIR, exist_ok=True)
  with open(AUDIT, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
  print('%6s %8s %8s %18s %6s' % ('rung', 'forced', 'of_tchr',
                                  'wilson99', 'gates'))
  for t, r in per_rung.items():
    lo, hi = r['forced_safe_wilson99_of_teacher']
    print('%6g %8d %8.3f   [%.3f, %.3f] %6s' % (
        r['force_safe_prob'], r['n_forced_safe'],
        r['forced_safe_fraction_of_teacher'], lo, hi,
        'PASS' if all(r['gates'].values()) else 'FAIL'))
  print()
  print('%12s %10s %10s %8s %6s' % ('pair', 'same_tchr', 'diverge@',
                                    'nested', 'gates'))
  for q in pairs:
    print('%5g->%-5g %10d %10s %8s %6s' % (
        q['low'], q['high'], q['identical_teacher_episodes'],
        q['first_diverging_teacher_episode'], q['nested_forced_safe_sets'],
        'PASS' if all(q['gates'].values()) else 'FAIL'))
  failed = ([f'rung {t}: {g}' for t, r in per_rung.items()
             for g, ok in r['gates'].items() if not ok]
            + [f'pair {q["low"]:g}->{q["high"]:g}: {g}' for q in pairs
               for g, ok in q['gates'].items() if not ok])
  for line in failed:
    print('  FAILED', line)
  print('-> %s' % AUDIT)
  print('LADDER AUDIT', 'PASS' if all_pass else 'FAIL')
  return 0 if all_pass else 1


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('build', 'manifest', 'audit'))
  ap.add_argument('--rungs', nargs='+', default=None,
                  help='subset of %s (default: all)' % (RUNGS,))
  ap.add_argument('--force', action='store_true',
                  help='build: delete and rebuild the rung')
  args = ap.parse_args(argv)
  rungs = parse_rungs(args.rungs) if args.rungs else list(RUNGS)
  if args.command == 'build':
    for p in rungs:
      build(p, force=args.force)
    return 0
  if args.command == 'manifest':
    write_manifest(rungs)
    return 0
  return audit(rungs)


if __name__ == '__main__':
  sys.exit(main())
