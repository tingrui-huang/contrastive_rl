r"""Build the composed V6 failure bank: 60% random/noisy, 20% Z1, 20% Z2.

Three mixture levels are involved in this experiment and only the second and
third are decided here:

  LEVEL 1  q_alpha = (1-alpha) q_normal + alpha q_fail     -- crl/losses.py,
           set per run by ``fail_neg_alpha``. NOT this file.
  LEVEL 2  q_fail = 0.60 q_random + 0.40 q_deliberate      -- this file.
  LEVEL 3  q_deliberate = 0.50 q_zone1 + 0.50 q_zone2      -- this file, EXACT.

One entry per failed episode: the observation the fatal transition returned
after the collector's ctrl-free settle substeps (the AntMaze settled
convention). The bank stores the full 29-dim learner STATE and crl/train.py
slices it to goal coordinates with the same rule the relabeler uses
(``config.goal_indices``), which is the existing AntMaze bank contract -- so
the file is goal-representation agnostic and the projection is guaranteed to
match whatever V6 CRL is trained with.

HARD COMPOSITION. Counts are computed from N_bank and the requested fractions
and must come out integral; the deliberate halves must be exactly equal. A
short pool is a hard error. Nothing is duplicated, nothing is topped up from
another class, N_bank is never silently reduced and 60/40 is never silently
moved. Sampling is without replacement.

EVERY DELIBERATE ENTRY IS RE-VERIFIED HERE, from the sidecar, not trusted from
the collector:

  zone 1:  normal_decision_zone1 == 'wait'
           executed_decision_zone1 == 'go'
           deliberate_override_zone1 is True
           actual_failure_zone == 1
  zone 2:  survived_zone1 is True
           normal_decision_zone2 == 'wait'
           executed_decision_zone2 == 'go'
           deliberate_override_zone2 is True
           actual_failure_zone == 2

SELECTION READS PRIVILEGED FIELDS -- source arm, both zones' normal/executed
decisions, the override flags, the failure zone. That cannot be done from the
observation and is recorded in the manifest. The STORED VECTORS carry none of
it: a bank entry is a 29-dim ant state and a state has no provenance.

Usage:
  python scripts/make_v6_failure_bank.py                       # N_bank 250
  python scripts/make_v6_failure_bank.py --n-bank 100
"""
import argparse
import glob as globmod
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import rockfall_clock_v6 as V6                # noqa: E402

CAND_DIR = os.path.join('artifacts', 'v6_failneg', 'candidates')
OUT_DIR = os.path.join('artifacts', 'v6_failneg', 'bank')
OUT_NAME = 'v6_failure_bank_r60_z20_z20_p040.npz'
DATASET = os.path.join('artifacts', 'rockfall_clock_v6', 'dataset',
                       'antmaze_rockfall_clock_v6_p040_gxy.npz')
STATE_DIM = 29
#: crl/losses.py pads the second critic apply and needs n_bank <= batch_size;
#: the V6 recipe (train_rockfall_clock_v6_baseline) uses batch_size 1024.
MAX_BANK_HARD_CAP = 1024
#: LEVEL 2 and LEVEL 3, fixed by the experiment definition.
RANDOM_FRACTION = 0.60
DELIBERATE_FRACTION = 0.40
#: which collector arm supplies the random/noisy component. 'random' (uniform
#: torque) is measured to produce nothing on this map, so the component comes
#: from the noisy-controller arm and is named for what it is.
RANDOM_ARM = 'noisy'
Z1_ARM = 'deliberate_z1'
Z2_ARM = 'deliberate_z2'
PRIVILEGED_SELECTION_FIELDS = (
    'source_arm', 'source_type', 'targeted_zone', 'actual_failure_zone',
    'survived_zone1', 'normal_decision_zone1', 'normal_decision_zone2',
    'executed_decision_zone1', 'executed_decision_zone2',
    'deliberate_override_zone1', 'deliberate_override_zone2', 'outcome')
#: fields that must never appear inside the stored learner tensor
LEARNER_FORBIDDEN = (
    'u1', 'u2', 't0_1', 't0_2', 'source_type', 'targeted_zone',
    'actual_failure_zone', 'deliberate_override_zone1',
    'deliberate_override_zone2', 'normal_decision_zone1',
    'normal_decision_zone2', 'executed_decision_zone1',
    'executed_decision_zone2', 'teacher_mode', 'schedule',
    'rockfall_start_1', 'rockfall_start_2', 'rockfall_end_1',
    'rockfall_end_2')


def content_sha(path):
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for key in sorted(d.files):
      v = d[key]
      h.update(key.encode())
      h.update(str(v.dtype).encode())
      h.update(str(v.shape).encode())
      h.update(np.ascontiguousarray(v).tobytes())
  return h.hexdigest()


def file_sha(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def git(*a):
  try:
    return subprocess.check_output(
        ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                          # pylint: disable=broad-except
    return ''


def load_arm(npz_path):
  """One candidate file -> (settled failure states [n, 29], provenance rows)."""
  side_path = npz_path.replace('.npz', '_sidecar.npz')
  if not os.path.exists(side_path):
    raise SystemExit('missing sidecar for %s' % npz_path)
  with np.load(npz_path, allow_pickle=False) as d:
    obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
    meta = json.loads(str(d['meta']))
  with np.load(side_path, allow_pickle=False) as s:
    side = {k: s[k] for k in s.files}
  ep = np.asarray(side['episode_id'], np.int64)
  dr = np.asarray(side['death_row'], np.int64)
  n = len(ep)
  #: the fatal observation must BE the last valid row. If it is not, the
  #: collector kept post-death frames and one episode would effectively enter
  #: the bank more than once.
  assert np.array_equal(dr, lengths[:n] - 1), (
      'death_row != lengths-1 in %s' % npz_path)
  states = obs[ep, dr, :STATE_DIM].astype(np.float32)
  assert np.isfinite(states).all(), 'non-finite failure state in %s' % npz_path
  rows = []
  for i in range(n):
    rows.append({
        'source_file': os.path.basename(npz_path),
        'source_arm': str(side['source_arm'][i]),
        'source_type': str(side['source_type'][i]),
        'targeted_zone': int(side['targeted_zone'][i]),
        'episode_id': int(ep[i]),
        'death_row': int(dr[i]),
        'failure_step': int(side['failure_step'][i]),
        'outcome': str(side['outcome'][i]),
        'actual_failure_zone': int(side['actual_failure_zone'][i]),
        'survived_zone1': bool(side['survived_zone1'][i]),
        'normal_decision_zone1': str(side['normal_decision_zone1'][i]),
        'normal_decision_zone2': str(side['normal_decision_zone2'][i]),
        'executed_decision_zone1': str(side['executed_decision_zone1'][i]),
        'executed_decision_zone2': str(side['executed_decision_zone2'][i]),
        'deliberate_override_zone1': bool(side['deliberate_override_zone1'][i]),
        'deliberate_override_zone2': bool(side['deliberate_override_zone2'][i]),
        'teacher_route': str(side['teacher_route'][i]),
        'u1': bool(side['u1'][i]), 'u2': bool(side['u2'][i]),
        't0_1': int(side['t0_1'][i]), 't0_2': int(side['t0_2'][i]),
        'hold_steps_zone1': int(side['hold_steps_zone1'][i]),
        'hold_steps_zone2': int(side['hold_steps_zone2'][i]),
        'env_seed': int(side['env_seed'][i]),
        'reset_index': int(side['reset_index'][i]),
        'collection_seed': int(side['collection_seed']),
        'death_settle_substeps': int(side['death_settle_substeps']),
    })
  bad = [r for r in rows if r['outcome'] != 'rock_death']
  assert not bad, '%d non-death entries in %s' % (len(bad), npz_path)
  return states, rows, meta


def verify_deliberate(rows, zone):
  """Re-derive the deliberate contract from provenance. Raises on any breach."""
  bad = []
  for r in rows:
    checks = {
        'source_type==deliberate': r['source_type'] == 'deliberate',
        'targeted_zone==%d' % zone: r['targeted_zone'] == zone,
        'normal_decision==wait': r['normal_decision_zone%d' % zone] == 'wait',
        'executed_decision==go': r['executed_decision_zone%d' % zone] == 'go',
        'deliberate_override': r['deliberate_override_zone%d' % zone],
        'actual_failure_zone==%d' % zone: r['actual_failure_zone'] == zone,
    }
    if zone == 2:
      checks['survived_zone1'] = r['survived_zone1']
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
      bad.append((r['source_file'], r['episode_id'], failed))
  if bad:
    raise SystemExit(
        'deliberate zone-%d contract violated by %d candidate(s); first: %s'
        % (zone, len(bad), bad[:3]))
  return len(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--cand-dir', default=CAND_DIR)
  ap.add_argument('--pattern', default='v6_failures_*_s*.npz')
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--out-name', default=OUT_NAME)
  ap.add_argument('--dataset', default=DATASET,
                  help='the V6 training set, recorded in the manifest only')
  ap.add_argument('--n-bank', type=int, default=250,
                  help='total entries; 60%% random, 20%% Z1, 20%% Z2, and the '
                       'three counts must come out integral')
  ap.add_argument('--seed', type=int, default=0, help='composition seed')
  ap.add_argument('--random-arm', default=RANDOM_ARM)
  args = ap.parse_args()

  n_bank = int(args.n_bank)
  if n_bank > MAX_BANK_HARD_CAP:
    raise SystemExit('--n-bank %d exceeds the batch_size cap %d that '
                     'crl/losses.py enforces' % (n_bank, MAX_BANK_HARD_CAP))
  n_random = RANDOM_FRACTION * n_bank
  n_delib = DELIBERATE_FRACTION * n_bank
  if abs(n_random - round(n_random)) > 1e-9 or abs(n_delib - round(n_delib)) > 1e-9:
    raise SystemExit('N_bank %d does not split 60/40 into whole entries '
                     '(%.2f / %.2f); pick a multiple of 10'
                     % (n_bank, n_random, n_delib))
  n_random, n_delib = int(round(n_random)), int(round(n_delib))
  if n_delib % 2:
    raise SystemExit('the deliberate half (%d) is odd, so it cannot be split '
                     '50/50 across the two zones; pick a different N_bank'
                     % n_delib)
  n_z1 = n_z2 = n_delib // 2
  assert n_random + n_z1 + n_z2 == n_bank

  files = sorted(f for f in globmod.glob(os.path.join(args.cand_dir,
                                                      args.pattern))
                 if not f.endswith('_sidecar.npz'))
  if not files:
    raise SystemExit('no candidate files matched %s in %s -- run '
                     'scripts/collect_v6_failure_candidates.py first'
                     % (args.pattern, args.cand_dir))
  pools, sources = {}, []
  for f in files:
    states, rows, meta = load_arm(f)
    arm = meta['source_arm']
    pools.setdefault(arm, {'states': [], 'rows': []})
    pools[arm]['states'].append(states)
    pools[arm]['rows'].extend(rows)
    sources.append({'path': f, 'source_arm': arm, 'n_entries': int(len(states)),
                    'collection_seed': meta['collection_seed'],
                    'noise_sigma': meta.get('noise_sigma'),
                    'death_settle_substeps': meta.get('death_settle_substeps'),
                    'keep_rate': meta.get('keep_rate'),
                    'failure_zone_counts': meta.get('failure_zone_counts'),
                    'content_sha256': content_sha(f)})
  for arm in pools:
    pools[arm]['states'] = np.concatenate(pools[arm]['states'], 0)
  pool_sizes = {a: len(p['rows']) for a, p in pools.items()}

  need = {args.random_arm: n_random, Z1_ARM: n_z1, Z2_ARM: n_z2}
  missing = {a: (pool_sizes.get(a, 0), k) for a, k in need.items()
             if pool_sizes.get(a, 0) < k}
  if missing:
    raise SystemExit(
        'pool too small for the required composition: %s (have, need). '
        'Collect more candidates for those arms. The bank is NOT topped up '
        'from another class, entries are NOT duplicated, N_bank is NOT '
        'reduced and 60/40 is NOT moved.' % missing)

  #: re-verify the deliberate contract on the WHOLE pool before selecting.
  verify_deliberate(pools[Z1_ARM]['rows'], 1)
  verify_deliberate(pools[Z2_ARM]['rows'], 2)

  rng = np.random.default_rng(args.seed)
  chosen_states, chosen_rows = [], []
  for arm, k in ((args.random_arm, n_random), (Z1_ARM, n_z1), (Z2_ARM, n_z2)):
    idx = np.sort(rng.choice(pool_sizes[arm], k, replace=False))
    chosen_states.append(pools[arm]['states'][idx])
    chosen_rows.extend(pools[arm]['rows'][i] for i in idx)
  bank = np.concatenate(chosen_states, 0).astype(np.float32)
  assert bank.shape == (n_bank, STATE_DIM), bank.shape

  ident = [(r['source_file'], r['episode_id']) for r in chosen_rows]
  assert len(set(ident)) == len(ident), 'an episode entered the bank twice'
  assert len(np.unique(bank, axis=0)) == len(bank), 'duplicate bank vectors'

  #: LEVEL 2 / LEVEL 3, asserted on the built object, not on intent.
  kinds = np.array([r['source_type'] for r in chosen_rows])
  zones = np.array([r['actual_failure_zone'] for r in chosen_rows])
  arms = np.array([r['source_arm'] for r in chosen_rows])
  got_random = int((kinds == 'random').sum())
  got_delib = int((kinds == 'deliberate').sum())
  got_z1 = int(((kinds == 'deliberate') & (zones == 1)).sum())
  got_z2 = int(((kinds == 'deliberate') & (zones == 2)).sum())
  assert got_random / n_bank == RANDOM_FRACTION, (got_random, n_bank)
  assert got_delib / n_bank == DELIBERATE_FRACTION, (got_delib, n_bank)
  assert got_z1 == got_z2, (got_z1, got_z2)
  assert got_random + got_z1 + got_z2 == n_bank
  verify_deliberate([r for r in chosen_rows
                     if r['source_type'] == 'deliberate'
                     and r['targeted_zone'] == 1], 1)
  verify_deliberate([r for r in chosen_rows
                     if r['source_type'] == 'deliberate'
                     and r['targeted_zone'] == 2], 2)
  random_by_zone = {str(z): int(((kinds == 'random') & (zones == z)).sum())
                    for z in (1, 2)}
  random_by_zone['ambiguous'] = int(
      ((kinds == 'random') & ~np.isin(zones, [1, 2])).sum())

  settles = sorted({r['death_settle_substeps'] for r in chosen_rows})
  if len(settles) != 1:
    raise SystemExit('mixed settle settings in one bank: %s' % settles)

  meta = {
      'definition': 'one entry per failed episode: the observation the fatal '
                    'transition returned after %d ctrl-free MuJoCo substeps'
                    % settles[0],
      'env_name': 'offline_antmaze_rockfall_clock_v6',
      'env_version': V6.ENV_VERSION,
      'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
      'head': git('rev-parse', 'HEAD'),
      'dirty': bool(git('status', '--porcelain', '--', 'crl', 'scripts')),
      'state_dim': STATE_DIM,
      'stored_space': 'the 29-dim learner STATE. crl/train.py slices it to '
                      'goal coordinates with config.goal_indices, so the bank '
                      'always matches whatever goal projection V6 CRL trains '
                      'with.',
      'settled_state_extraction': {
          'death_settle_substeps': int(settles[0]),
          'convention': 'the AntMaze settled-failure convention (see '
                        'scripts/rebuild_failure_bank_settled.py, SETTLE_N '
                        '80): at the fatal contact the actor loses control, '
                        'ctrl is zeroed and physics advances N substeps inside '
                        'the same env step. No observation field is written '
                        'by hand.',
          'implemented_by': 'crl/rockfall_clock_v6.py '
                            'RockfallClockV6Env.death_settle_substeps '
                            '(opt-in, default 0 = byte-identical V6)'},
      'main_dataset': args.dataset,
      'main_dataset_sha256': (file_sha(args.dataset)
                              if os.path.exists(args.dataset) else None),
      'main_dataset_content_sha256': (content_sha(args.dataset)
                                      if os.path.exists(args.dataset) else None),
      'main_dataset_unchanged': True,
      'candidate_pools': sources,
      'pool_sizes': pool_sizes,
      'composition': {
          'level_1_note': 'q_alpha = (1-alpha) q_normal + alpha q_fail is set '
                          'per run by fail_neg_alpha, not here',
          'level_2': {'random': RANDOM_FRACTION,
                      'deliberate': DELIBERATE_FRACTION},
          'level_3': {'zone1': 0.5, 'zone2': 0.5},
          'random_arm': args.random_arm,
          'random_arm_note': 'uniform torque produces no task-relevant '
                             'failures on this map, so the random/noisy '
                             'component is the noisy-controller arm and is '
                             'named for what it is',
      },
      'n_bank': int(n_bank),
      'n_random': got_random, 'n_deliberate': got_delib,
      'n_deliberate_zone1': got_z1, 'n_deliberate_zone2': got_z2,
      'random_by_zone': random_by_zone,
      'fraction_random': got_random / n_bank,
      'fraction_deliberate': got_delib / n_bank,
      'composition_seed': int(args.seed),
      'sampling': 'stratified without replacement; a short pool is a hard '
                  'error, never a top-up, a duplicate or a reduced N_bank',
      'p_active_1': float(V6.P_ACTIVE_1), 'p_active_2': float(V6.P_ACTIVE_2),
      't0_ranges': {'1': [V6.T0_MIN_1, V6.T0_MAX_1],
                    '2': [V6.T0_MIN_2, V6.T0_MAX_2]},
      'deliberate_override_api':
          "scripts/rockfall_clock_v6_teacher.py: "
          "LongTwoRockfallTeacher.fresh(route=..., "
          "deliberate_override_zone=None|1|2); normal_decisions / "
          "executed_decisions / deliberate_overrides expose the three "
          "quantities separately",
      'collector_detour_coin': 'DISABLED in the candidate collector only '
                               '(route forced to shortcut). The benchmark '
                               'teacher and the frozen dataset keep 0.05.',
      'selection_uses_privileged_fields': list(PRIVILEGED_SELECTION_FIELDS),
      'privileged_selection_note':
          'CLASS SELECTION AND VERIFICATION ONLY. The stored vectors are '
          '29-dim learner ant states and carry no provenance; the sidecar '
          'fields decided WHICH of them are kept and proved each deliberate '
          'entry satisfies its contract. A bank built this way is a curated '
          'training artifact, not something the learner could have derived.',
      'learner_forbidden_fields': list(LEARNER_FORBIDDEN),
      'entries': [{k: r[k] for k in (
          'source_file', 'source_arm', 'source_type', 'targeted_zone',
          'episode_id', 'death_row', 'failure_step', 'actual_failure_zone',
          'survived_zone1', 'normal_decision_zone1', 'normal_decision_zone2',
          'executed_decision_zone1', 'executed_decision_zone2',
          'deliberate_override_zone1', 'deliberate_override_zone2',
          'u1', 'u2', 't0_1', 't0_2', 'hold_steps_zone1', 'hold_steps_zone2',
          'env_seed', 'reset_index', 'collection_seed')}
          for r in chosen_rows],
  }

  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, args.out_name)
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(
        f, goals=bank,
        source_arm=arms.astype('U16'),
        source_type=kinds.astype('U12'),
        targeted_zone=np.array([r['targeted_zone'] for r in chosen_rows],
                               np.int64),
        actual_failure_zone=zones.astype(np.int64),
        episode_id=np.array([r['episode_id'] for r in chosen_rows], np.int64),
        death_row=np.array([r['death_row'] for r in chosen_rows], np.int64),
        source_file=np.array([r['source_file'] for r in chosen_rows], 'U64'),
        collection_seed=np.array([r['collection_seed'] for r in chosen_rows],
                                 np.int64),
        meta=np.array(json.dumps(meta)))
  os.replace(tmp, out_path)
  manifest = dict(meta, bank_path=out_path, bank_sha256=file_sha(out_path),
                  bank_content_sha256=content_sha(out_path),
                  bank_shape=list(bank.shape))
  man = os.path.splitext(args.out_name)[0] + '_manifest.json'
  with open(os.path.join(args.out_dir, man), 'w') as f:
    json.dump(manifest, f, indent=2)

  print('candidate files : %d' % len(files))
  for s in sources:
    print('  %-14s %4d entries  seed %-5d settle %-3s  %s'
          % (s['source_arm'], s['n_entries'], s['collection_seed'],
             s['death_settle_substeps'], os.path.basename(s['path'])))
  print('pool sizes      : %s' % pool_sizes)
  print('N_bank          : %d' % n_bank)
  print('  random   (%s)  %d   = %.2f' % (args.random_arm, got_random,
                                          got_random / n_bank))
  print('  delib Z1        %d   = %.2f' % (got_z1, got_z1 / n_bank))
  print('  delib Z2        %d   = %.2f' % (got_z2, got_z2 / n_bank))
  print('  random by zone  %s' % random_by_zone)
  print('settle substeps : %d' % settles[0])
  print('bank            : %s  shape %s' % (out_path, bank.shape))
  print('bank sha256     : %s' % manifest['bank_sha256'])
  print('content sha256  : %s' % manifest['bank_content_sha256'])
  print('ASSERTIONS PASSED: random 0.60, deliberate 0.40, Z1 == Z2, and every '
        'deliberate entry re-verified from provenance.')
  print('-> %s' % os.path.join(args.out_dir, man))


if __name__ == '__main__':
  main()
