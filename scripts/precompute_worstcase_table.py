"""Task section 2: precompute the static worst-case table over the frozen
offline dataset.

    (s_t^data, a_t^data)  ->  s'_wc,t

The audit resolved Flow conditioning to the LOGGED action a^data; the Flow,
selector, bank and normalization are frozen and the dataset is fixed, so this
map is a static table. RL then does a lookup and never invokes the Flow.

Noise convention. The sealed selector convention is a single
``jax.random.normal(PRNGKey(11), (n*K, 29))`` draw per call. Reusing that one
key for every chunk of a 227k-row dataset would give transitions ``chunk``
apart IDENTICAL x0 noise, so each chunk uses
``jax.random.fold_in(PRNGKey(11), chunk_index)``. The frozen root seed is
preserved and no selector constant changes (K=256, 50-step Euler, 16-state
bank, frozen V0 normalization, Euclidean L2, lowest-index tie-break). The chunk
size is recorded in the manifest and the table is only reproducible with it.

Usage:
  python scripts/precompute_worstcase_table.py [--chunk 256] [--limit N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import jax                                       # noqa: E402
from crl import static_worstcase as sw           # noqa: E402

OUT = os.path.join(_ROOT, 'artifacts/static_worstcase_rl')
CLEAN = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
         'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
CLEAN_SHA_PIN = ('6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc'
                 '76be1852')
OBS_DIM = 29


def _rel(p):
  try:
    return os.path.relpath(p, _ROOT)
  except ValueError:                      # different drive (scratch dirs)
    return os.path.abspath(p)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--chunk', type=int, default=256)
  ap.add_argument('--limit', type=int, default=0, help='debug: first N rows')
  ap.add_argument('--out', default=os.path.join(OUT, 'worstcase_table.npz'))
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)

  m = sw.StaticWorstCase(root=_ROOT)
  clean = os.path.join(_ROOT, CLEAN)
  clean_sha = sw.sha256_file(clean)
  assert clean_sha == CLEAN_SHA_PIN, (
      'ABORT: dataset sha %s is not the authoritative clean-dataset sha'
      % clean_sha)

  with np.load(clean, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)
  E, L = obs.shape[0], obs.shape[1]

  # every transition (e, t) with t <= len_e - 2, i.e. an action exists AND a
  # successor obs exists -- exactly the rows crl/replay.py can anchor on.
  ei, ti = [], []
  for e in range(E):
    n = int(lengths[e]) - 1
    ei.append(np.full(n, e, np.int64))
    ti.append(np.arange(n, dtype=np.int64))
  ei = np.concatenate(ei)
  ti = np.concatenate(ti)
  flat = ei * L + ti                       # flat index into the [E, L] grid
  if args.limit:
    ei, ti, flat = ei[:args.limit], ti[:args.limit], flat[:args.limit]
  N = len(flat)

  S = obs[ei, ti, :OBS_DIM]
  A = act[ei, ti]
  print('transitions: %d (episodes %d, ep_len_obs %d) chunk %d'
        % (N, E, L, args.chunk))

  s_wc = np.zeros((N, OBS_DIM), np.float32)
  k_sel = np.zeros(N, np.int32)
  d_neg = np.zeros(N, np.float32)
  root = jax.random.PRNGKey(sw.SEED)
  t0 = time.time()
  n_chunks = int(np.ceil(N / args.chunk))
  for c in range(n_chunks):
    lo, hi = c * args.chunk, min((c + 1) * args.chunk, N)
    out, aux = m.worst_case_next_state(
        S[lo:hi], A[lo:hi], return_aux=True,
        key=jax.random.fold_in(root, c))
    s_wc[lo:hi] = out
    k_sel[lo:hi] = aux['k']
    d_neg[lo:hi] = aux['d_neg']
    if c % 50 == 0 or c == n_chunks - 1:
      el = time.time() - t0
      done = hi / N
      print('  chunk %5d/%d  rows %7d/%d  %5.1f%%  elapsed %6.1f min  '
            'eta %6.1f min' % (c + 1, n_chunks, hi, N, 100 * done, el / 60,
                               (el / max(done, 1e-9) - el) / 60), flush=True)

  assert np.isfinite(s_wc).all(), 'non-finite worst-case state in table'
  np.savez_compressed(
      args.out, flat_index=flat, episode_index=ei, time_index=ti,
      state=S, action=A, s_wc=s_wc, candidate_index=k_sel,
      d_nearest_negative=d_neg)
  table_sha = sw.sha256_file(args.out)

  man = {
      'table': _rel(args.out),
      'table_sha256': table_sha,
      'n_transitions': int(N),
      'n_episodes': int(E), 'ep_len_obs': int(L),
      'row_rule': ('every (e, t) with t <= lengths[e] - 2 -- exactly the rows '
                   'crl/replay.py:_draw_indices can anchor on'),
      'flat_index_rule': 'flat = episode_index * ep_len_obs + time_index',
      'dataset': CLEAN, 'dataset_sha256': clean_sha,
      'conditioning_action': 'a_t^data (logged offline action; audit 0C)',
      'noise_convention': {
          'root_key': 'jax.random.PRNGKey(%d)' % sw.SEED,
          'per_chunk': 'jax.random.fold_in(root, chunk_index)',
          'chunk': int(args.chunk),
          'why': ('one key reused across chunks would give transitions '
                  '`chunk` apart identical x0 noise'),
          'frozen_constants_unchanged': True},
      'selector_provenance': m.provenance(),
      'd_nearest_negative_stats': {
          'mean': float(d_neg.mean()), 'median': float(np.median(d_neg)),
          'p10': float(np.percentile(d_neg, 10)),
          'p90': float(np.percentile(d_neg, 90)),
          'min': float(d_neg.min()), 'max': float(d_neg.max())},
      'build_minutes': round((time.time() - t0) / 60, 2),
      'git_commit': subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()}
  json.dump(man, open(os.path.join(OUT, 'worstcase_table_manifest.json'), 'w'),
            indent=2)
  print('\ntable sha %s' % table_sha)
  print('d_nearest_negative: mean %.3f median %.3f p10 %.3f p90 %.3f'
        % (d_neg.mean(), np.median(d_neg), np.percentile(d_neg, 10),
           np.percentile(d_neg, 90)))
  print('built in %.1f min -> %s' % ((time.time() - t0) / 60, args.out))


if __name__ == '__main__':
  main()
