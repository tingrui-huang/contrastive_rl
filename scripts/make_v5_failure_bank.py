r"""Build the composed failure bank for the V5 rockfall-clock benchmark.

One entry per FAILED EPISODE: the observation the env itself returned on the
fatal step. That row is unique by construction -- V5's failure is absorbing and
the collector breaks on ``done``, so there is exactly one post-death frame per
episode and no episode can win extra weight by repeating it. The bank stores
the full 29-dim learner STATE; crl/train.py slices it to goal coordinates with
the same rule the relabeler uses (config.goal_indices), so ONE bank file serves
both the XY and the XYV goal contracts and the comparison between them is
exact.

  python scripts/make_v5_failure_bank.py --compose noisy=0.6,deliberate=0.4

COMPOSITION is 60% random/noisy + 40% competent unsafe entry by default, and
the two numbers in that sentence are independent of the mixture weight alpha:
this file decides WHAT IS IN the bank, crl/losses.py's ``fail_neg_alpha``
decides HOW MUCH WEIGHT the bank carries inside the negative term.

Classes are the collector's source arms, not guesses (see
scripts/collect_v5_failure_episodes.py):

  random      uniform torque. Measured to produce NO task-relevant failures on
              this benchmark (the ant never leaves the start cell), so its pool
              is normally empty and the "random" half of the mixture is served
              by ``noisy``. Asking for it is an error, not a silent top-up.
  noisy       the walking controller + Gaussian action noise, blind. The
              random/noisy half. Called noisy-controller, never "random".
  blind       the walking controller, no noise, never read the timetable.
  deliberate  the sighted expert read the timetable, its rule said WAIT, and it
              was overridden to GO. The competent unsafe entry.
  mistimed    the sighted expert waited and released too early.

SELECTION READS PRIVILEGED FIELDS. Choosing a subset by behaviour class cannot
be done from the observation, so the manifest records exactly which sidecar
fields were read (source_arm, teacher_decision, schedule_read, intervention).
The STORED VECTORS carry none of it: a bank entry is a 29-dim learner state and
a state has no provenance. This is the same bargain the f4 bank documents --
the bank is a curated training artifact, not something the learner could have
derived -- and the only reason it is sound is that the composition never
touches the learner's inputs.

The manifest also records, per entry, the source file, episode id, death step,
collection seed and the content hash of every arm file it drew from, so a bank
can be traced back to the exact episodes that produced it.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y   # noqa: E402

FAIL_DIR = 'artifacts/v5_failneg/failures'
OUT_DIR = 'artifacts/v5_failneg/bank'
OUT_NAME = 'v5_failure_bank_n60d40.npz'
DEFAULT_COMPOSE = 'noisy=0.6,deliberate=0.4'
CLASSES = ('random', 'noisy', 'blind', 'deliberate', 'mistimed')
STATE_DIM = 29
#: crl/losses.py requires n_bank <= batch_size; the offline AntMaze recipe
#: (scripts/verify_offline_d4rl.build_offline_cfg) uses batch_size 1024.
MAX_BANK_HARD_CAP = 1024
#: privileged sidecar fields the CLASS SELECTION reads. Recorded, never stored.
PRIVILEGED_SELECTION_FIELDS = ('source_arm', 'teacher_decision',
                               'schedule_read', 'schedule_fields_read',
                               'intervention', 'outcome')


def content_sha(path):
  """sha256 over the npz ARRAY CONTENTS (zip metadata ignored)."""
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for key in sorted(d.files):
      value = d[key]
      h.update(key.encode())
      h.update(str(value.dtype).encode())
      h.update(str(value.shape).encode())
      h.update(np.ascontiguousarray(value).tobytes())
  return h.hexdigest()


def file_sha(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def parse_compose(spec):
  """'noisy=0.6,deliberate=0.4' -> {'noisy': 0.6, 'deliberate': 0.4}."""
  if not spec:
    return None
  mix = {}
  for part in spec.split(','):
    if '=' not in part:
      raise SystemExit('bad --compose term %r (expected name=fraction)' % part)
    name, frac = part.split('=', 1)
    name = name.strip()
    if name not in CLASSES:
      raise SystemExit('unknown class %r; expected one of %s'
                       % (name, ', '.join(CLASSES)))
    if name in mix:
      raise SystemExit('class %r given twice in --compose' % name)
    mix[name] = float(frac)
  total = sum(mix.values())
  if abs(total - 1.0) > 1e-9:
    raise SystemExit('--compose fractions sum to %.6f, expected 1.0' % total)
  if any(v <= 0 for v in mix.values()):
    raise SystemExit('--compose fractions must be positive')
  return mix


def load_arm(npz_path):
  """One arm file -> (death states [n, 29], per-entry provenance rows)."""
  side_path = npz_path.replace('.npz', '_sidecar.npz')
  if not os.path.exists(side_path):
    raise SystemExit('missing sidecar for %s (expected %s)'
                     % (npz_path, side_path))
  with np.load(npz_path, allow_pickle=False) as d:
    obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
    meta = json.loads(str(d['meta']))
  with np.load(side_path, allow_pickle=False) as s:
    side = {k: s[k] for k in s.files}
  n = int((lengths > 0).sum())
  death_row = np.asarray(side['death_row'], np.int64)
  ep_id = np.asarray(side['episode_id'], np.int64)
  assert len(death_row) == n, (len(death_row), n)
  #: the fatal observation must BE the last valid row -- if it is not, the
  #: collector recorded post-death frames and an episode would enter the bank
  #: more than once in effect.
  assert np.array_equal(death_row, lengths[:n] - 1), (
      'death_row != lengths-1 in %s: the fatal frame is not the last row'
      % npz_path)
  states = obs[ep_id, death_row, :STATE_DIM].astype(np.float32)
  assert np.isfinite(states).all(), 'non-finite death state in %s' % npz_path
  rows = []
  for i in range(n):
    rows.append({
        'source_file': os.path.basename(npz_path),
        'source_arm': str(side['source_arm'][i]),
        'episode_id': int(ep_id[i]),
        'death_row': int(death_row[i]),
        'outcome': str(side['outcome'][i]),
        'teacher_decision': str(side['teacher_decision'][i]),
        'schedule_read': bool(side['schedule_read'][i]),
        'schedule_fields_read': str(side['schedule_fields_read'][i]),
        'intervention': str(side['intervention'][i]),
        'rockfall_start': int(side['rockfall_start'][i]),
        'mouth_step': int(side['mouth_step'][i]),
        'band_entry_step': int(side['band_entry_step'][i]),
        'hold_steps': int(side['hold_steps'][i]),
        'intervention_step': int(side['intervention_step'][i]),
        'noise_sigma': float(side['noise_sigma'][i]),
        'cut_steps': int(side['cut_steps'][i]),
        'rockfall_active': bool(side['rockfall_active'][i]),
        'death_in_band': bool(side['death_in_band'][i]),
        'death_in_corridor_row': bool(side['death_in_corridor_row'][i]),
        'collection_seed': int(side['collection_seed']),
        'collection_base_seed': int(side['collection_base_seed']),
    })
  #: only real rock deaths may reach a bank. Not a filter -- an assertion:
  #: the collector already refused anything else, and if one got through, the
  #: right response is a loud stop, not a quiet drop.
  bad = [r for r in rows if r['outcome'] != 'rock_death']
  assert not bad, '%d non-death entries in %s' % (len(bad), npz_path)
  assert meta['source_arm'] == rows[0]['source_arm']
  semantic_ok = {
      'random': lambda r: (not r['schedule_read']
                           and r['teacher_decision'] == 'none'
                           and r['intervention'] == 'uniform_action'),
      'noisy': lambda r: (not r['schedule_read']
                          and r['teacher_decision'] == 'none'
                          and r['intervention'] == 'action_noise'
                          and r['noise_sigma'] > 0.0),
      'blind': lambda r: (not r['schedule_read']
                          and r['teacher_decision'] == 'none'
                          and r['intervention'] == 'none'),
      'deliberate': lambda r: (r['schedule_read']
                               and {'active', 'start', 'end'}.issubset(
                                   set(r['schedule_fields_read'].split(',')))
                               and r['teacher_decision'] == 'wait'
                               and r['intervention'] == 'override_wait_to_go'
                               and r['intervention_step'] >= 0),
      'mistimed': lambda r: (r['schedule_read']
                             and {'active', 'start', 'end'}.issubset(
                                 set(r['schedule_fields_read'].split(',')))
                             and r['teacher_decision'] == 'wait'
                             and r['intervention'] == 'early_release'
                             and r['intervention_step'] >= 0
                             and r['cut_steps'] > 0
                             and r['hold_steps'] > 0),
  }
  bad_semantics = [r for r in rows
                   if (r['source_arm'] not in semantic_ok
                       or not r['rockfall_active']
                       or not r['death_in_corridor_row']
                       or not semantic_ok[r['source_arm']](r))]
  assert not bad_semantics, (
      '%d rows do not satisfy the recorded source-class mechanism in %s'
      % (len(bad_semantics), npz_path))
  return states, rows, meta


def compose_indices(labels, mix, n_target, seed):
  """Stratified sample WITHOUT replacement honouring the class fractions.

  Largest-remainder rounding so the counts sum EXACTLY to n_target, and a hard
  failure -- never a silent top-up from another class and never a duplicated
  entry -- if a pool is too small. A bank whose mixture quietly drifted is the
  kind of artifact that invalidates the whole alpha comparison.
  """
  pools = {c: np.flatnonzero(labels == c) for c in mix}
  exact = {c: mix[c] * n_target for c in mix}
  take = {c: int(np.floor(v)) for c, v in exact.items()}
  order = sorted(mix, key=lambda c: (-(exact[c] - take[c]), c))
  k = 0
  while sum(take.values()) < n_target:
    take[order[k % len(order)]] += 1
    k += 1
  short = {c: (int(len(pools[c])), int(take[c]))
           for c in mix if len(pools[c]) < take[c]}
  if short:
    raise SystemExit(
        'pool too small for the requested mixture: %s (have, need). Collect '
        'more episodes for those arms or lower --max-bank; the bank is NOT '
        'topped up from another class and entries are NOT duplicated.' % short)
  rng = np.random.default_rng(seed)
  chosen = [rng.choice(pools[c], take[c], replace=False) for c in sorted(mix)]
  return np.sort(np.concatenate(chosen)), take


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--fail-dir', default=FAIL_DIR)
  ap.add_argument('--pattern', default='v5_failures_*_s*.npz',
                  help='arm files to read from --fail-dir (sidecars excluded)')
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--out-name', default=OUT_NAME)
  ap.add_argument('--max-bank', type=int, default=256,
                  help='entries in the bank; crl/losses.py needs this <= the '
                       'training batch_size (1024 in the offline AntMaze '
                       'recipe)')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--compose', default=DEFAULT_COMPOSE,
                  help='class mixture, e.g. "noisy=0.6,deliberate=0.4". '
                       'Empty = a provenance-blind uniform subsample of every '
                       'failure found.')
  ap.add_argument('--require-in-band', action='store_true',
                  help='keep only deaths inside the band proper')
  args = ap.parse_args()
  mix = parse_compose(args.compose)
  if args.max_bank > MAX_BANK_HARD_CAP:
    raise SystemExit('--max-bank %d exceeds the batch_size cap %d that '
                     'crl/losses.py enforces on the padded second critic '
                     'apply' % (args.max_bank, MAX_BANK_HARD_CAP))

  import glob
  files = sorted(f for f in glob.glob(os.path.join(args.fail_dir,
                                                   args.pattern))
                 if not f.endswith('_sidecar.npz'))
  if not files:
    raise SystemExit('no arm files matched %s in %s -- run '
                     'scripts/collect_v5_failure_episodes.py first'
                     % (args.pattern, args.fail_dir))

  states, rows, sources = [], [], []
  for f in files:
    st, rw, meta = load_arm(f)
    states.append(st)
    rows.extend(rw)
    sources.append({'path': f, 'source_arm': meta['source_arm'],
                    'n_entries': int(len(st)),
                    'collection_seed': meta['collection_seed'],
                    'collection_base_seed': meta['collection_base_seed'],
                    'noise_sigma': meta.get('noise_sigma'),
                    'cut_steps': meta.get('cut_steps'),
                    'keep_rate': meta.get('keep_rate'),
                    'content_sha256': content_sha(f),
                    'sidecar_content_sha256': content_sha(
                        f.replace('.npz', '_sidecar.npz'))})
  states = np.concatenate(states, 0)
  labels = np.array([r['source_arm'] for r in rows])
  if args.require_in_band:
    keep = np.array([r['death_in_band'] for r in rows])
    states, labels = states[keep], labels[keep]
    rows = [r for r, k in zip(rows, keep) if k]
  pool_sizes = {c: int((labels == c).sum()) for c in CLASSES}

  #: one entry per semantic episode identity, independent of file renaming.
  keys = [(r['collection_seed'], r['source_arm'], r['episode_id'])
          for r in rows]
  assert len(set(keys)) == len(keys), (
      'duplicate (collection_seed, source_arm, episode) in the pool')

  if mix is None:
    kept = np.arange(len(states))
    if args.max_bank and len(states) > args.max_bank:
      rng = np.random.default_rng(args.seed)
      kept = np.sort(rng.choice(len(states), args.max_bank, replace=False))
    take = None
  else:
    if not args.max_bank:
      raise SystemExit('--compose needs a positive --max-bank (a mixture is a '
                       'fraction OF something)')
    kept, take = compose_indices(labels, mix, args.max_bank, args.seed)
  bank = states[kept]
  bank_rows = [rows[i] for i in kept]
  bank_labels = labels[kept]
  assert len(set((r['collection_seed'], r['source_arm'], r['episode_id'])
                 for r in bank_rows)) == len(bank_rows), (
      'an episode entered the bank twice')

  xy = bank[:, :2]
  in_band = ((np.abs(xy[:, 1]) < HAZARD_HALF_Y)
             & (xy[:, 0] >= HAZARD_X[0]) & (xy[:, 0] <= HAZARD_X[1]))
  speed = np.linalg.norm(bank[:, 15:18], axis=1)

  meta = {
      'definition': 'the observation the env returned on the fatal step; one '
                    'entry per failed episode',
      'env_name': 'offline_ant_umaze_rockfall_clock_v5',
      'state_dim': STATE_DIM,
      'stored_space': 'the 29-dim learner STATE. crl/train.py slices it to '
                      'goal coordinates with config.goal_indices, so the same '
                      'bank file serves the XY (2) and XYV (8) goal contracts.',
      'extraction_moment': 'settle 0 -- the step V5 itself reports the death '
                           'on. Post-mortem settling was measured (see '
                           'scripts/probe_v5_failure_representation.py) and '
                           'REJECTED: it drives the death state toward zero '
                           'velocity, i.e. toward the expert HOLD, which is '
                           'the behaviour the benchmark wants preserved.',
      'source_files': sources,
      'n_pool': int(len(states)),
      'class_pool_sizes': pool_sizes,
      'class_definitions': {
          'random': 'uniform torque; measured to produce no task-relevant '
                    'failures on this benchmark (never reaches the band)',
          'noisy': 'walking controller + Gaussian action noise, blind '
                   '(NOISY-CONTROLLER, not uniform random)',
          'blind': 'walking controller, no noise, timetable never read',
          'deliberate': 'sighted expert read the timetable, decided WAIT, '
                        'overridden to GO',
          'mistimed': 'sighted expert waited, released early, entered while '
                      'the burst was still open',
      },
      'compose': args.compose or None,
      'composed': mix is not None,
      'requested_mixture': mix,
      'class_counts_kept': {c: int((bank_labels == c).sum()) for c in CLASSES},
      'n_bank': int(len(bank)),
      'sample_seed': int(args.seed),
      'sampling': 'stratified without replacement, largest-remainder rounding; '
                  'a short pool is a hard error, never a top-up or a duplicate',
      'require_in_band': bool(args.require_in_band),
      'frac_in_band': float(in_band.mean()),
      'linear_speed': {'mean': float(speed.mean()), 'min': float(speed.min()),
                       'max': float(speed.max())},
      'xy_range': [[float(xy[:, 0].min()), float(xy[:, 0].max())],
                   [float(xy[:, 1].min()), float(xy[:, 1].max())]],
      'selection_uses_privileged_fields': (list(PRIVILEGED_SELECTION_FIELDS)
                                           if mix is not None else []),
      'privileged_selection_note': (
          'CLASS SELECTION ONLY. The stored vectors are 29-dim learner states '
          'and carry no provenance; the sidecar fields decided WHICH of them '
          'are kept, nothing else. A bank built this way is a curated training '
          'artifact, not something the learner could have derived.'
          if mix is not None else
          'none -- provenance-blind uniform subsample'),
      'normalisation': 'none; raw env units',
      'entries': [{'source_file': r['source_file'],
                   'source_arm': r['source_arm'],
                   'episode_id': r['episode_id'],
                   'death_row': r['death_row'],
                   'teacher_decision': r['teacher_decision'],
                   'schedule_read': r['schedule_read'],
                   'schedule_fields_read': r['schedule_fields_read'],
                   'intervention': r['intervention'],
                   'rockfall_start': r['rockfall_start'],
                   'mouth_step': r['mouth_step'],
                   'band_entry_step': r['band_entry_step'],
                   'hold_steps': r['hold_steps'],
                   'collection_seed': r['collection_seed'],
                   'collection_base_seed': r['collection_base_seed']}
                  for r in bank_rows],
  }

  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, args.out_name)
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(
        f, goals=bank,
        source_arm=bank_labels.astype('U12'),
        episode_id=np.array([r['episode_id'] for r in bank_rows], np.int64),
        death_row=np.array([r['death_row'] for r in bank_rows], np.int64),
        source_file=np.array([r['source_file'] for r in bank_rows], 'U64'),
        collection_seed=np.array([r['collection_seed'] for r in bank_rows],
                                 np.int64),
        meta=np.array(json.dumps(meta)))
  os.replace(tmp, out_path)
  manifest = dict(meta, bank_path=out_path, bank_sha256=file_sha(out_path),
                  bank_content_sha256=content_sha(out_path),
                  bank_shape=list(bank.shape))
  man_name = os.path.splitext(args.out_name)[0] + '_manifest.json'
  with open(os.path.join(args.out_dir, man_name), 'w') as f:
    json.dump(manifest, f, indent=2)

  print('arm files       : %d' % len(files))
  for s in sources:
    print('  %-11s %4d entries  %s' % (s['source_arm'], s['n_entries'],
                                       os.path.basename(s['path'])))
  print('pool            : %d' % len(states))
  print('class pools     : %s' % pool_sizes)
  print('requested mix   : %s' % (args.compose or '(uniform, blind)'))
  print('kept per class  : %s'
        % {c: int((bank_labels == c).sum()) for c in CLASSES
           if (bank_labels == c).any()})
  print('bank            : %s  shape %s' % (out_path, bank.shape))
  print('in band         : %.3f   mean |v| %.3f' % (in_band.mean(),
                                                    speed.mean()))
  print('content sha256  : %s' % manifest['bank_content_sha256'])
  if mix is not None:
    print('SELECTION READ PRIVILEGED FIELDS: %s (recorded in the manifest)'
          % ', '.join(PRIVILEGED_SELECTION_FIELDS))
  print('-> %s' % os.path.join(args.out_dir, man_name))


if __name__ == '__main__':
  main()
