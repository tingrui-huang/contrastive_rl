"""Build the 3-D failure bank for point_two_route_swamp_windy_z_v0.

ONE ENTRY PER FAILED EPISODE. For each failed episode find the FIRST transition

    z_t = 0  ->  z_{t+1} < 0

and store g_f = (x_{t+1}, y_{t+1}, z_{t+1}). The post-failure frozen rows carry
the same z < 0 forever, so taking them all would put one episode's single
outcome into the bank dozens of times and silently reweight q_fail toward
whichever episodes died earliest. Exactly one row per failed episode.

The failure event is detected from Z ALONE -- a physical, learner-visible
quantity -- not from swamp_bits. That is the point of the Z variant: the bank
no longer needs the hidden confounder to be constructed. swamp_bits is loaded
only to CROSS-CHECK the count and is never used to select entries.

STORED RAW. z stays at its physical value (z_min = -0.5). The z_physical
scaling lives in the model preprocessing path
(crl.obs_norm.obs_scale_vector -> crl/networks.py), and crl/losses.py splices
the raw bank into the goal half before calling q_network.apply, so the bank is
scaled by exactly the same code as every other goal, exactly once. Pre-scaling
here would double-apply it.

Bank size is capped by crl/losses.py, which requires n_bank <= batch_size.

Usage:
  python scripts/make_swamp_z_failure_bank.py
  python scripts/make_swamp_z_failure_bank.py --max-bank 256
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

SRC = 'datasets/swamp_windy_z_merged_s0.npz'
OUT_DIR = 'artifacts/swamp_windy_z_failure_bank'
GOAL_DIM = 3
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))


def sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def content_sha(path):
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode())
      h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode())
      h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--npz', default=SRC)
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--max-bank', type=int, default=256,
                  help='0 = keep every failed episode; else subsample '
                       '(seeded, order preserved). crl/losses.py needs '
                       'n_bank <= batch_size.')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  with np.load(args.npz, allow_pickle=False) as d:
    obs = d['obs']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
    mode = np.asarray(d['teacher_mode'])
    bits = d['swamp_bits']
  n_eps, L, W = obs.shape
  assert W == 2 * GOAL_DIM, 'expected obs width 6 ([x,y,z,gx,gy,gz]), got %d' % W
  xyz = obs[:, :, :GOAL_DIM]
  z = xyz[:, :, 2]

  # first 0 -> negative transition, per episode
  entries, ep_ids, rows = [], [], []
  for e in range(n_eps):
    idx = np.where((z[e, :-1] == 0.0) & (z[e, 1:] < 0.0))[0]
    if idx.size == 0:
      continue
    t = int(idx[0])
    entries.append(xyz[e, t + 1])
    ep_ids.append(e)
    rows.append(t + 1)
  bank = np.asarray(entries, np.float32)
  ep_ids = np.asarray(ep_ids, np.int64)
  rows = np.asarray(rows, np.int64)

  # cross-checks
  n_failed = int(died.sum())
  assert len(bank) == n_failed, (
      'z-derived failures (%d) disagree with entered_active_swamp (%d)'
      % (len(bank), n_failed))
  assert np.array_equal(np.sort(ep_ids), np.where(died)[0]), (
      'the z-derived failed-episode set differs from the audit field')
  assert np.all(bank[:, 2] < 0), 'a bank entry is not below ground'
  # every entry must sit in a swamp cell whose governing bit was active
  cell = np.clip(np.floor(bank[:, :2]).astype(int), 0, [8, 4])
  in_sw = np.zeros(len(bank), bool)
  for cx, cy in SWAMP_CELLS:
    in_sw |= (cell[:, 0] == cx) & (cell[:, 1] == cy)
  assert in_sw.all(), 'a bank entry is outside the three swamp cells'
  bit_ok = 0
  for k, (e, r) in enumerate(zip(ep_ids, rows)):
    c = (int(cell[k, 0]), int(cell[k, 1]))
    bit_ok += int(bool(bits[e, r - 1, SWAMP_CELLS.index(c)]))
  assert bit_ok == len(bank), (
      'a bank entry was not governed by an active bit (%d/%d)'
      % (bit_ok, len(bank)))

  uniq = np.unique(bank, axis=0)
  n_dup = len(bank) - len(uniq)
  kept = np.arange(len(bank))
  if args.max_bank and len(bank) > args.max_bank:
    rng = np.random.default_rng(args.seed)
    kept = np.sort(rng.choice(len(bank), args.max_bank, replace=False))
  bank_out, ep_out, row_out = bank[kept], ep_ids[kept], rows[kept]
  uniq_out = np.unique(bank_out, axis=0)

  meta = {
      'definition': 'first z=0 -> z<0 transition per failed episode; the '
                    'landed state (x, y, z) of that transition. Exactly one '
                    'entry per failed episode; post-failure frozen rows are '
                    'NOT added.',
      'env_name': 'point_two_route_swamp_windy_z_v0',
      'source_npz': args.npz, 'source_content_sha256': content_sha(args.npz),
      'goal_dim': GOAL_DIM, 'goal_indices': 'range(3) (start=0, end=-1)',
      'detected_from': 'z only (learner-visible); swamp_bits used ONLY to '
                       'cross-check, never to select',
      'n_episodes': int(n_eps), 'n_failed_episodes': n_failed,
      'n_bank_full': int(len(bank)), 'n_bank': int(len(bank_out)),
      'subsampled': bool(args.max_bank and len(bank) > args.max_bank),
      'subsample_seed': int(args.seed),
      'duplicates_full': int(n_dup), 'unique_full': int(len(uniq)),
      'duplicates_kept': int(len(bank_out) - len(uniq_out)),
      'z_values': sorted(float(v) for v in np.unique(bank_out[:, 2])),
      'z_stored_raw': True,
      'z_scaling_note': 'raw; z_physical is applied inside crl/networks.py so '
                        'the bank is scaled by the same code as every other '
                        'goal, exactly once',
      'failed_by_teacher_mode': {str(int(m)): int((mode[ep_out] == m).sum())
                                 for m in np.unique(mode[ep_out])},
      'batch_size_requirement': 'crl/losses.py requires n_bank <= batch_size',
  }
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, 'failure_bank_z.npz')
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, goals=bank_out, episode_id=ep_out,
                        failure_row=row_out,
                        teacher_mode=mode[ep_out].astype(np.int64),
                        meta=np.array(json.dumps(meta)))
  os.replace(tmp, out_path)
  fsha, csha = sha256(out_path), content_sha(out_path)
  man = dict(meta)
  man.update({'bank_path': out_path, 'bank_sha256': fsha,
              'bank_content_sha256': csha,
              'bank_shape': list(bank_out.shape),
              'x_range': [float(bank_out[:, 0].min()),
                          float(bank_out[:, 0].max())],
              'y_range': [float(bank_out[:, 1].min()),
                          float(bank_out[:, 1].max())],
              'z_range': [float(bank_out[:, 2].min()),
                          float(bank_out[:, 2].max())]})
  with open(os.path.join(args.out_dir, 'failure_bank_z_manifest.json'),
            'w') as f:
    json.dump(man, f, indent=2)

  print('source            : %s' % args.npz)
  print('episodes          : %d   failed %d (%.4f)'
        % (n_eps, n_failed, n_failed / n_eps))
  print('bank (full)       : %d entries, %d unique, %d duplicate'
        % (len(bank), len(uniq), n_dup))
  print('bank (kept)       : %s   %d unique, %d duplicate'
        % (bank_out.shape, len(uniq_out), len(bank_out) - len(uniq_out)))
  print('x range           : [%.4f, %.4f]' % (man['x_range'][0],
                                              man['x_range'][1]))
  print('y range           : [%.4f, %.4f]' % (man['y_range'][0],
                                              man['y_range'][1]))
  print('z values          : %s' % meta['z_values'])
  print('failed by mode    : %s' % meta['failed_by_teacher_mode'])
  print('bank file sha256  : %s' % fsha)
  print('bank content sha  : %s' % csha)
  print('-> %s' % out_path)
  print('\nCROSS-CHECKS PASSED: z-derived failures == entered_active_swamp, '
        'all entries below ground,\n  all inside a swamp cell, all governed by '
        'an ACTIVE bit.')


if __name__ == '__main__':
  main()
