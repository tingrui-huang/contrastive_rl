"""PART 1 (failure-aware negatives), Step 1+2: clean/rock-fail split + bank.

Splits the AUTHORITATIVE rockfall dataset (p30 / H=800 / resetfix_v1) by the
sidecar's per-episode ``dead`` flag into:

  clean npz     D_clean     = D_original \\ D_rock-fail  (learner contract:
                obs/act/eval_goals/lengths/meta -- runs unchanged through
                scripts/naive_rockfall_v2_crl.py --npz <clean>).
  rockfail npz  D_rock-fail = the episodes that END because of rockfall
                (dead=True: severe rock contact -> absorbing burial). Kept for
                provenance only; NEVER a training set in the clean experiments.
  failure bank  D_fail = {g^fail}: ONE state per dead episode -- the terminal
                post-collapse observation obs[e, lengths[e]-1, :29] (the env
                truncates a dead episode at collapse_step+2, so this is the
                single stored buried pose; see collect_rockfall_v2_pilot.py).
                Stored as 29-dim goal-space states (goal_indices=range(29)).

CONSERVATIVE failure-state definition (documented per Step 2):
  * dead episodes only. The 1 ``impaired`` episode (hit but survived) stays in
    D_clean -- its trajectory did not end because of rockfall.
  * only the post-collapse buried pose per dead episode, NOT the preceding
    trajectory. => bank size == number of dead episodes (16).

The original npz/sidecar are read-only; outputs go to a new subdirectory.

Usage:
  python scripts/make_rockfall_failure_split.py
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

SRC_DIR = 'artifacts/rockfall_v2_p30_h800_resetfix/pilot'
SRC_NAME = 'antmaze_rockfall_v2_p30_h800_resetfix_pilot'
OUT_DIR = 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split'
OBS_DIM = 29   # learner state width; goal_indices = range(29).


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
      h.update(blk)
  return h.hexdigest()


def write_learner_npz(out_path, d, idx, note):
  """Write a learner-contract npz (obs/act/eval_goals/lengths/meta) from the
  source arrays and an episode index list. Same contract as
  make_rockfall_subsets.write_subset."""
  idx = np.asarray(idx, int)
  meta = dict(json.loads(str(d['meta'])))
  meta['note'] = note
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, obs=d['obs'][idx], act=d['act'][idx],
                        eval_goals=d['eval_goals'][idx],
                        lengths=d['lengths'][idx], meta=json.dumps(meta))
  os.replace(tmp, out_path)
  lengths = d['lengths'][idx]
  return int(len(idx)), int((lengths - 1).sum())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--src-dir', default=SRC_DIR)
  ap.add_argument('--src-name', default=SRC_NAME)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  src_npz = os.path.join(args.src_dir, f'{args.src_name}.npz')
  src_sidecar = os.path.join(args.src_dir, f'{args.src_name}_sidecar.npz')
  os.makedirs(args.out_dir, exist_ok=True)

  src_sha = sha256(src_npz)
  sidecar_sha = sha256(src_sidecar)

  d = np.load(src_npz, allow_pickle=True)
  s = np.load(src_sidecar, allow_pickle=True)

  dead = np.asarray(s['dead'], bool)
  n = len(dead)
  fail_idx = np.where(dead)[0]
  clean_idx = np.where(~dead)[0]
  lengths = np.asarray(d['lengths'], np.int64)

  # --- Identification cross-checks (fail loudly if the sidecar disagrees). --
  assert n == d['obs'].shape[0]
  hit_any = np.asarray(s['hit'], bool).any(axis=1)
  assert np.all(hit_any[fail_idx]), 'dead episode without a recorded rock hit'
  assert np.all(np.asarray(s['success'])[fail_idx] == 0.0)
  collapse = np.asarray(s['collapse_step'], np.int64)
  assert np.all(collapse[fail_idx] >= 0)
  # collection truncates dead episodes at collapse_step + 2 valid observations
  assert np.all(lengths[fail_idx] == collapse[fail_idx] + 2), (
      'dead-episode length != collapse_step+2; failure-state rule invalid')
  impaired_in_clean = np.asarray(s['impaired'], bool) & ~dead

  # ------------------------------------------------ Step 1: the two datasets
  clean_path = os.path.join(args.out_dir, f'{args.src_name}_clean.npz')
  fail_path = os.path.join(args.out_dir, f'{args.src_name}_rockfail.npz')
  n_clean, t_clean = write_learner_npz(
      clean_path, d, clean_idx,
      note=(f'D_clean: episodes with sidecar dead=False of {args.src_name} '
            f'(source sha256 {src_sha[:16]}...). Rockfall-death episodes '
            'removed; for failure-aware negative-sampling experiments.'))
  n_fail, t_fail = write_learner_npz(
      fail_path, d, fail_idx,
      note=(f'D_rock-fail: episodes with sidecar dead=True of {args.src_name} '
            f'(source sha256 {src_sha[:16]}...). PROVENANCE ONLY -- never an '
            'ordinary CRL training set in the clean experiments.'))

  # ------------------------------------------------ Step 2: failure bank
  # g^fail = terminal post-collapse obs of each dead episode, state part only.
  bank_states = np.stack(
      [d['obs'][e, lengths[e] - 1, :OBS_DIM] for e in fail_idx]).astype(
          np.float32)
  bank_path = os.path.join(args.out_dir, 'failure_bank.npz')
  tmp = bank_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(
        f, goals=bank_states,
        episode_id=fail_idx.astype(np.int64),
        collapse_step=collapse[fail_idx],
        first_hit_step=np.asarray(s['first_hit_step'], np.int64)[fail_idx],
        ep_length=lengths[fail_idx],
        meta=json.dumps({
            'definition': ('terminal post-collapse observation '
                           'obs[e, lengths[e]-1, :29] of each sidecar '
                           'dead=True episode (= the single stored buried '
                           'pose; dead episodes are truncated at '
                           'collapse_step+2 observations at collection)'),
            'source_npz': src_npz, 'source_sha256': src_sha,
            'source_sidecar_sha256': sidecar_sha,
            'obs_dim': OBS_DIM, 'goal_indices': 'range(29)'}))
  os.replace(tmp, bank_path)

  manifest = {
      'source_npz': src_npz,
      'source_sha256': src_sha,
      'source_sidecar_sha256': sidecar_sha,
      'split_rule': 'sidecar dead flag (severe rock contact -> burial)',
      'n_episodes_original': int(n),
      'n_transitions_original': int((lengths - 1).sum()),
      'clean': {'path': clean_path, 'sha256': sha256(clean_path),
                'n_episodes': n_clean, 'n_transitions': t_clean},
      'rockfail': {'path': fail_path, 'sha256': sha256(fail_path),
                   'n_episodes': n_fail, 'n_transitions': t_fail,
                   'episode_ids': fail_idx.tolist()},
      'failure_bank': {'path': bank_path, 'sha256': sha256(bank_path),
                       'n_states': int(len(bank_states)),
                       'state_dim': OBS_DIM},
      'impaired_episodes_kept_in_clean': np.where(impaired_in_clean)[0].tolist(),
  }
  man_path = os.path.join(args.out_dir, 'failure_split_manifest.json')
  with open(man_path, 'w') as f:
    json.dump(manifest, f, indent=2)

  print(f'source    : {n} eps / {(lengths - 1).sum()} transitions '
        f'({src_sha[:16]}...)')
  print(f'D_clean   : {n_clean} eps / {t_clean} transitions -> {clean_path}')
  print(f'D_rockfail: {n_fail} eps / {t_fail} transitions -> {fail_path}')
  print(f'bank      : {len(bank_states)} states ({OBS_DIM}-dim) -> {bank_path}')
  print(f'impaired episode(s) kept in clean: '
        f'{manifest["impaired_episodes_kept_in_clean"]}')
  print(f'manifest  : {man_path}')


if __name__ == '__main__':
  main()
