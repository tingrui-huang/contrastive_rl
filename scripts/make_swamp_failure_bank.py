"""Build the failure-state bank for the windy-LETHAL swamp (PointMaze).

Analogue of scripts/make_rockfall_failure_split.py, but the dataset stays
WHOLE: per the chosen pipeline the original 6000 episodes are retained as the
training set (fixed length, as in the original CRL), anchors are restricted by
scheme C (crl.offline_audit.compute_anchor_cuts), ordinary in-batch negatives
are untouched, and this bank is added as EXTRA negative goals only.

  g^fail = the terminal death state of each episode whose sidecar
           ``entered_active_swamp`` flag is set: obs[e, death_row, :2], where
           death_row is the first row whose cell is a swamp cell that the bits
           governing the previous step had ACTIVE. The env freezes the point
           there for the rest of the episode, so this is the single distinct
           post-mortem position.

Stored in the learner STATE space (2-dim [x, y]); crl.train slices it to goal
coords with the same rule the relabeler uses (here start=0/end=-1, identity).

Bank size is capped by the critic loss: crl/losses.py requires
``n_bank <= batch_size`` (the bank is written over the first n_bank rows of the
goal half). 514 deaths therefore need batch_size >= 514, or --max-bank.

Usage:
  python scripts/make_swamp_failure_bank.py
  python scripts/make_swamp_failure_bank.py --max-bank 256      # subsample
  python scripts/make_swamp_failure_bank.py --npz <bad_demo.npz> --out <dir>
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

SRC_NPZ = 'datasets/swamp_windy_teacher_s0.npz'
OUT_DIR = 'artifacts/swamp_windy_failure_bank'
OBS_DIM = 2                      # learner state = [x, y]
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
GRID_HI = [8, 4]                 # walls array is 9 x 5


def sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def find_death_rows(obs, bits, died):
  """Row index of the death STATE for every dead episode, -1 otherwise.

  Timing contract (crl.envs.TwoRouteSwampWindyEnv.step): bits[t] govern step t;
  the death check runs on the state REACHED by that step, then the bits redraw.
  So the death state is at row t+1 where cell(row t+1) is swamp k and
  bits[t, k] is set.
  """
  n_eps, L, _ = obs.shape
  cell = np.clip(np.floor(obs[:, :, :2]).astype(int), 0, GRID_HI)
  sw = -np.ones((n_eps, L), int)
  for k, (cx, cy) in enumerate(SWAMP_CELLS):
    sw[(cell[:, :, 0] == cx) & (cell[:, :, 1] == cy)] = k
  death_row = np.full(n_eps, -1, np.int64)
  for e in np.where(died)[0]:
    for t in range(L - 1):
      k = sw[e, t + 1]
      if k >= 0 and bits[e, t, k]:
        death_row[e] = t + 1
        break
  return death_row, sw


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--npz', default=SRC_NPZ)
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--max-bank', type=int, default=0,
                  help='0 = keep every death state; else subsample to N '
                       '(seeded, deterministic, order preserved)')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  src_sha = sha256(args.npz)
  d = np.load(args.npz, allow_pickle=True)
  obs, bits = d['obs'], d['swamp_bits']
  died = np.asarray(d['entered_active_swamp']).astype(bool)
  mode = np.asarray(d['teacher_mode'])
  n_eps, L, W = obs.shape
  assert W == 2 * OBS_DIM, f'expected obs width {2*OBS_DIM}, got {W}'

  death_row, sw = find_death_rows(obs, bits, died)
  fail_idx = np.where(died)[0]
  assert np.all(death_row[fail_idx] > 0), (
      'a dead episode has no reconstructable death row; the timing contract '
      'in find_death_rows does not match this dataset')

  # Cross-checks: the recorded death state must actually be terminal, i.e. the
  # point never moves again for the rest of the episode.
  xy = obs[:, :, :OBS_DIM]
  for e in fail_idx:
    r = death_row[e]
    assert np.allclose(xy[e, r:], xy[e, r], atol=1e-6), (
        f'episode {e}: state changes after the death row -- not absorbing')
  assert not died[mode == 1].any(), 'forced_safe episode died (route invariant)'

  bank = xy[fail_idx, death_row[fail_idx]].astype(np.float32)
  cell = np.clip(np.floor(bank).astype(int), 0, GRID_HI)
  per_cell = {f'S{k}_{SWAMP_CELLS[k]}':
              int(((cell[:, 0] == SWAMP_CELLS[k][0])
                   & (cell[:, 1] == SWAMP_CELLS[k][1])).sum())
              for k in range(3)}
  assert sum(per_cell.values()) == len(bank), (
      'a death state is outside the three swamp cells')

  kept = np.arange(len(bank))
  if args.max_bank and len(bank) > args.max_bank:
    rng = np.random.default_rng(args.seed)
    kept = np.sort(rng.choice(len(bank), args.max_bank, replace=False))
  ep_ids = fail_idx[kept]
  bank_out = bank[kept]

  meta = {
      'definition': ('terminal death state obs[e, death_row, :2] of every '
                     'episode with entered_active_swamp=True; the env freezes '
                     'the point there for the rest of the episode'),
      'env_name': 'point_two_route_swamp_windy_v0',
      'source_npz': args.npz, 'source_sha256': src_sha,
      'obs_dim': OBS_DIM, 'goal_indices': 'range(2)  (start=0, end=-1)',
      'n_dead_episodes_total': int(len(fail_idx)),
      'n_bank': int(len(bank_out)),
      'subsampled': bool(args.max_bank and len(bank) > args.max_bank),
      'subsample_seed': int(args.seed),
      'deaths_per_swamp_cell_full': per_cell,
      'deaths_by_source_full': {
          str(int(m)): int((mode[fail_idx] == m).sum())
          for m in np.unique(mode[fail_idx])},
      'pipeline_note': ('dataset NOT split: all 6000 episodes remain the '
                        'training set; this bank is EXTRA negative goals '
                        'only (crl/losses.py fail branch, alpha>0)'),
      'batch_size_requirement': 'crl/losses.py requires n_bank <= batch_size',
  }
  out_path = os.path.join(args.out_dir, 'failure_bank.npz')
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(
        f, goals=bank_out, episode_id=ep_ids.astype(np.int64),
        death_row=death_row[ep_ids].astype(np.int64),
        teacher_mode=mode[ep_ids].astype(np.int64),
        meta=json.dumps(meta))
  os.replace(tmp, out_path)

  manifest = dict(meta)
  manifest['bank_path'] = out_path
  manifest['bank_sha256'] = sha256(out_path)
  manifest['bank_shape'] = list(bank_out.shape)
  manifest['bank_x_range'] = [float(bank_out[:, 0].min()),
                              float(bank_out[:, 0].max())]
  manifest['bank_y_range'] = [float(bank_out[:, 1].min()),
                              float(bank_out[:, 1].max())]
  man_path = os.path.join(args.out_dir, 'failure_bank_manifest.json')
  with open(man_path, 'w') as f:
    json.dump(manifest, f, indent=2)

  print(f'source     : {args.npz} ({src_sha[:16]}...)')
  print(f'episodes   : {n_eps}, dead {len(fail_idx)} '
        f'({len(fail_idx)/n_eps:.2%})')
  print(f'per cell   : ' + '  '.join(f'{k}={v}' for k, v in per_cell.items()))
  print(f'bank       : {bank_out.shape} -> {out_path}')
  print(f'             x [{manifest["bank_x_range"][0]:.3f}, '
        f'{manifest["bank_x_range"][1]:.3f}]  '
        f'y [{manifest["bank_y_range"][0]:.3f}, '
        f'{manifest["bank_y_range"][1]:.3f}]')
  print(f'sha256     : {manifest["bank_sha256"]}')
  print(f'manifest   : {man_path}')
  print(f'\nREQUIREMENT: run with batch_size >= {len(bank_out)} '
        f'(crl/losses.py guard), or rebuild with --max-bank <= batch_size.')


if __name__ == '__main__':
  main()
