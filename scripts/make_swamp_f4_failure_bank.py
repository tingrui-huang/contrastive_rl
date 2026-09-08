"""Build the frame-stacked failure bank for the windy swamp f4 dataset.

One entry is kept per failed episode: the first learner-visible state whose
four position frames are identical while the current position is in a swamp
cell.  This is the observable post-failure freeze signature; hidden swamp bits
and the dataset's failure flag are used only for cross-checks, never selection.

--compose CHANGES THAT, DELIBERATELY AND ONLY FOR SELECTION. The default build
is unchanged: uniform subsample, provenance-blind. With --compose the KEPT
SUBSET is chosen by behaviour class, which cannot be done without god-view
fields, so the manifest records exactly which privileged fields were read:

  random       teacher_mode == 0 -- the uniform-action episodes. Noise.
  deliberate   teacher_mode != 0 AND at the failure step the agent moved into a
               DIFFERENT cell whose bit was ACTIVE in the very bits the teacher
               reads before acting, while at least one of {forward, wait, back}
               would have landed clear. "Knew, had an out, went in anyway."
  unaware      the agent did not change cell -- it was standing still (or its
               step did not carry it out) and the wind turned that cell on.
  doomed       moved in, but all three candidate landings were active.

The stored GOAL VECTORS are untouched by this: composition only reweights which
observable failure states appear, because a bank goal is a point in goal space
and carries no provenance. See the measurement in the session notes -- the
deliberate and unaware point clouds differ only by ~0.6 maze units along x.
"""
import argparse
import hashlib
import json
import os

import numpy as np


SRC = 'datasets/swamp_windy_f4_merged_s0.npz'
OUT_DIR = 'artifacts/swamp_windy_f4_failure_bank'
OUT_NAME = 'failure_bank_f4_entry.npz'
NF = 4
GOAL_DIM = 2 * NF
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))


def content_sha(path):
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


CLASSES = ('random', 'deliberate', 'unaware', 'doomed')


def parse_compose(spec):
  """'random=0.6,deliberate=0.4' -> {'random': 0.6, 'deliberate': 0.4}."""
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


def death_step(pos_ep, frozen_pos):
  """First row from which the position never changes again.

  The freeze is exact (the env stops integrating), so equality is the right
  test and no tolerance is needed. Returns -1 if the episode never settles.
  """
  same = np.all(pos_ep == frozen_pos, axis=1)
  hit = np.flatnonzero(same)
  return int(hit[0]) if hit.size else -1


def classify(ep, pos, bits, teacher_mode, swamp_cells, frozen_row):
  """Behaviour class of one failed episode. READS PRIVILEGED FIELDS.

  ``teacher_mode`` says which behaviour policy generated the episode and
  ``bits`` is the hidden confounder; neither is in the learner observation.
  Used for SELECTION only, never stored in a goal vector.
  """
  if int(teacher_mode[ep]) == 0:
    return 'random'
  t_death = death_step(pos[ep], pos[ep, frozen_row])
  if t_death <= 0:
    return 'unaware'                      # settled at t=0: no entry decision
  t = t_death - 1

  def cell(p):
    c = np.clip(np.floor(p).astype(int), [0, 0], [8, 4])
    return (int(c[0]), int(c[1]))

  def active(c):
    return c in swamp_cells and bool(bits[ep, t, swamp_cells.index(c)])

  c_from, c_into = cell(pos[ep, t]), cell(pos[ep, t_death])
  if c_from == c_into:
    return 'unaware'                      # never left the cell the wind took
  if not active(c_into):
    return 'unaware'                      # read clear, killed by a later draw
  cx, cy = c_from
  outs = [(cx + 1, cy), (cx, cy), (cx - 1, cy)]   # forward / wait / back
  return 'deliberate' if any(not active(o) for o in outs) else 'doomed'


def compose_indices(labels, mix, n_target, seed):
  """Pick ``n_target`` indices honouring the requested class fractions.

  Largest-remainder rounding so the counts sum EXACTLY to n_target, and a hard
  failure -- never a silent top-up from another class -- if a pool is too
  small, because a silently reweighted bank is the kind of artifact that
  invalidates the whole comparison.
  """
  pools = {c: np.flatnonzero(labels == c) for c in mix}
  exact = {c: mix[c] * n_target for c in mix}
  take = {c: int(np.floor(v)) for c, v in exact.items()}
  order = sorted(mix, key=lambda c: (-(exact[c] - take[c]), c))
  k = 0
  while sum(take.values()) < n_target:
    take[order[k % len(order)]] += 1
    k += 1
  short = {c: (len(pools[c]), take[c]) for c in mix if len(pools[c]) < take[c]}
  if short:
    raise SystemExit(
        'pool too small for the requested mixture: %s (have, need)' % short)
  rng = np.random.default_rng(seed)
  chosen = []
  for c in sorted(mix):
    chosen.append(rng.choice(pools[c], take[c], replace=False))
  return np.sort(np.concatenate(chosen)), take


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--npz', default=SRC)
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--max-bank', type=int, default=256)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--compose', default='',
                  help='class mixture, e.g. "random=0.6,deliberate=0.4". '
                       'Empty = the original provenance-blind uniform '
                       'subsample. Classes: random | deliberate | unaware | '
                       'doomed. Fractions must sum to 1.')
  ap.add_argument('--out-name', default=OUT_NAME,
                  help='bank filename; give a composed bank its own name so '
                       'the provenance-blind one is never overwritten')
  args = ap.parse_args()
  mix = parse_compose(args.compose)

  with np.load(args.npz, allow_pickle=False) as d:
    obs = np.asarray(d['obs'], np.float32)
    audited_failure = np.asarray(d['entered_active_swamp']).astype(bool)
    teacher_mode = np.asarray(d['teacher_mode'])
    swamp_bits = np.asarray(d['swamp_bits'])

  n_eps, length, width = obs.shape
  assert width == 2 * GOAL_DIM, (
      'expected width 16 ([four XY frames | tiled goal]), got %d' % width)
  states = obs[:, :, :GOAL_DIM]
  frames = states.reshape(n_eps, length, NF, 2)
  frozen = np.max(np.abs(frames - frames[:, :, :1]), axis=(2, 3)) == 0.0
  xy = frames[:, :, 0]
  cells = np.floor(xy).astype(int)
  in_swamp = np.zeros((n_eps, length), dtype=bool)
  for cx, cy in SWAMP_CELLS:
    in_swamp |= (cells[:, :, 0] == cx) & (cells[:, :, 1] == cy)
  visible_failure = frozen & in_swamp

  entries, ep_ids, rows = [], [], []
  for ep in range(n_eps):
    found = np.flatnonzero(visible_failure[ep])
    if found.size:
      row = int(found[0])
      entries.append(states[ep, row])
      ep_ids.append(ep)
      rows.append(row)

  bank = np.asarray(entries, np.float32)
  ep_ids = np.asarray(ep_ids, np.int64)
  rows = np.asarray(rows, np.int64)
  expected = np.flatnonzero(audited_failure)
  false_positive = np.setdiff1d(ep_ids, expected)
  missed = np.setdiff1d(expected, ep_ids)
  assert not len(false_positive), (
      'observable frozen-in-swamp signature selected %d non-failures'
      % len(false_positive))
  assert len(bank), 'no observable failure states found'

  labels = np.array([classify(e, xy, swamp_bits, teacher_mode,
                              list(SWAMP_CELLS), r)
                     for e, r in zip(ep_ids, rows)])
  pool_sizes = {c: int((labels == c).sum()) for c in CLASSES}

  if mix is None:
    kept = np.arange(len(bank))
    if args.max_bank and len(bank) > args.max_bank:
      rng = np.random.default_rng(args.seed)
      kept = np.sort(rng.choice(len(bank), args.max_bank, replace=False))
    take = None
  else:
    if not args.max_bank:
      raise SystemExit('--compose needs a positive --max-bank (the mixture is '
                       'a fraction OF something)')
    kept, take = compose_indices(labels, mix, args.max_bank, args.seed)
  bank_out, ep_out, row_out = bank[kept], ep_ids[kept], rows[kept]
  label_out = labels[kept]

  meta = {
      'definition': 'first learner-visible fully frozen four-frame state in a '
                    'swamp cell; one entry per failed episode',
      'env_name': 'point_two_route_swamp_windy_f4_v0',
      'source_npz': args.npz,
      'source_content_sha256': content_sha(args.npz),
      'goal_dim': GOAL_DIM,
      'n_frames': NF,
      'detected_from': 'frame stack and public swamp geometry only; '
                       'entered_active_swamp used only to cross-check',
      'n_episodes': int(n_eps),
      'n_failed_episodes': int(audited_failure.sum()),
      'n_visible_failed_episodes': int(len(ep_ids)),
      'n_right_censored_failures': int(len(missed)),
      'right_censor_note': 'failure occurred too near the episode boundary '
                           'for all four frames to become frozen; excluded '
                           'rather than selected using a hidden label',
      'n_bank_full': int(len(bank)),
      'n_bank': int(len(bank_out)),
      'subsampled': bool(args.max_bank and len(bank) > args.max_bank),
      'subsample_seed': int(args.seed),
      'failure_row_range': [int(rows.min()), int(rows.max())],
      'failed_by_teacher_mode': {
          str(int(mode)): int((teacher_mode[ep_out] == mode).sum())
          for mode in np.unique(teacher_mode[ep_out])},
      'normalisation': 'none; raw maze-unit frame coordinates',
      'class_definitions': {
          'random': 'teacher_mode == 0 (uniform-action episode)',
          'deliberate': 'teacher_mode != 0 and the failure step moved into a '
                        'DIFFERENT cell whose bit was active in the bits the '
                        'teacher reads before acting, while at least one of '
                        '{forward, wait, back} would have landed clear',
          'unaware': 'did not change cell -- the wind turned on the cell it '
                     'was standing in',
          'doomed': 'moved in, but all three candidate landings were active',
      },
      'class_pool_sizes': pool_sizes,
      'compose': args.compose or None,
      'composed': mix is not None,
      'class_counts_kept': ({c: int((label_out == c).sum())
                             for c in CLASSES} if mix is not None else None),
      'selection_uses_privileged_fields': (
          ['teacher_mode', 'swamp_bits'] if mix is not None else []),
      'privileged_selection_note': (
          'CLASS SELECTION ONLY. The stored goal vectors are learner-visible '
          'frozen frame stacks and carry no provenance; teacher_mode and '
          'swamp_bits decided WHICH of them are kept, nothing else. A bank '
          'built this way is a curated training artifact, not something the '
          'learner could have derived.' if mix is not None else
          'none -- provenance-blind uniform subsample'),
  }
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, args.out_name)
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, goals=bank_out, episode_id=ep_out,
                        failure_row=row_out,
                        teacher_mode=teacher_mode[ep_out].astype(np.int64),
                        behaviour_class=label_out.astype('U12'),
                        meta=np.array(json.dumps(meta)))
  os.replace(tmp, out_path)
  manifest = dict(meta, bank_path=out_path,
                  bank_sha256=file_sha(out_path),
                  bank_content_sha256=content_sha(out_path),
                  bank_shape=list(bank_out.shape))
  man_name = os.path.splitext(args.out_name)[0] + '_manifest.json'
  with open(os.path.join(args.out_dir, man_name), 'w') as f:
    json.dump(manifest, f, indent=2)

  print('source           : %s' % args.npz)
  print('episodes         : %d  failed %d' % (n_eps, audited_failure.sum()))
  print('visible/censored : %d / %d' % (len(ep_ids), len(missed)))
  print('bank full/kept   : %d / %d, shape %s' % (
      len(bank), len(bank_out), bank_out.shape))
  print('failure rows     : [%d, %d]' % (rows.min(), rows.max()))
  print('class pools      : %s' % pool_sizes)
  if mix is not None:
    print('requested mix    : %s' % args.compose)
    print('kept per class   : %s'
          % {c: int((label_out == c).sum()) for c in CLASSES
             if (label_out == c).any()})
    print('SELECTION READ PRIVILEGED FIELDS: teacher_mode, swamp_bits '
          '(recorded in the manifest)')
  print('content sha256   : %s' % manifest['bank_content_sha256'])
  print('-> %s' % out_path)
  print('CROSS-CHECK PASSED: every selected observable frozen-in-swamp episode '
        'is an audited failure; right-censored failures were excluded.')


if __name__ == '__main__':
  main()
