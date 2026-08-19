"""Phase 4: throughput / memory profile of the frozen worst-case module.

K=256 and the 50 Euler steps are FROZEN scientific parameters and are never
reduced here -- the only free variable is how many anchors are batched per
call. Reports candidates/sec, ms per anchor, peak RSS, and the implied
training-time overhead against the authoritative baseline throughput.

Usage:  python scripts/profile_static_worstcase.py [--max-anchors 256]
"""
import argparse
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
# authoritative baseline: batch_size 1024, num_sgd_steps_per_step 4
BASE_BATCH = 1024
BASE_SGD_PER_STEP = 4


def rss_mb():
  try:
    import psutil
    return float(psutil.Process().memory_info().rss) / 1e6
  except Exception:
    return float('nan')


CLEAN_NPZ = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
             'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')


def _precompute(ms_per_anchor):
  """Cost of caching s'_wc for EVERY transition of the frozen dataset.

  0C resolved the conditioning action to a^data. The Flow, the bank, the
  normalization and the dataset are all frozen, so the map
  (s_t^data, a_t^data) -> s'_wc is a STATIC TABLE: it can be computed once and
  looked up during RL at zero per-update cost. This removes throughput as a
  constraint regardless of how the branch-firing rule is eventually defined."""
  with np.load(os.path.join(_ROOT, CLEAN_NPZ), allow_pickle=True) as d:
    lengths = np.asarray(d['lengths'], np.int64)
    n_eps, L = d['obs'].shape[0], d['obs'].shape[1]
  n_trans = int((lengths - 1).sum())
  hours = n_trans * ms_per_anchor / 1000.0 / 3600.0
  return {'dataset': CLEAN_NPZ, 'n_episodes': int(n_eps), 'ep_len_obs': int(L),
          'n_transitions': n_trans,
          'one_time_precompute_hours_this_cpu': round(hours, 2),
          'one_time_precompute_hours_at_50x_gpu': round(hours / 50.0, 3),
          'cache_size_gb_float32': round(n_trans * 29 * 4 / 1e9, 3),
          'implication': ('s_wc is a STATIC per-transition table over the '
                          'frozen dataset; precompute once, then RL training '
                          'carries ZERO Flow cost per update. Throughput is '
                          'therefore NOT a blocker for any branch-firing '
                          'rule.')}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--max-anchors', type=int, default=256)
  ap.add_argument('--repeats', type=int, default=3)
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)

  m = sw.StaticWorstCase(root=_ROOT)
  z = np.load(os.path.join(_ROOT, 'artifacts/state_nn_selector_confirm',
                           'candidates.npz'), allow_pickle=True)
  S0 = np.asarray(z['anchor'], np.float32)
  A0 = np.asarray(z['action'], np.float32)

  sizes = [n for n in (1, 4, 16, 64, 128, 256, 512, 1024)
           if n <= args.max_anchors]
  rows = []
  base_rss = rss_mb()
  for n in sizes:
    reps = int(np.ceil(n / len(S0)))
    S = np.tile(S0, (reps, 1))[:n].copy()
    A = np.tile(A0, (reps, 1))[:n].copy()
    t0 = time.time()
    out = m.worst_case_next_state(S, A)
    jax.block_until_ready(out)
    t_compile = time.time() - t0            # first call: includes XLA compile
    ts = []
    for _ in range(args.repeats):
      t1 = time.time()
      out = m.worst_case_next_state(S, A)
      jax.block_until_ready(out)
      ts.append(time.time() - t1)
    t = float(np.median(ts))
    rows.append({
        'anchors': n, 'candidates': n * sw.K_CANDIDATES,
        'first_call_s_incl_compile': round(t_compile, 3),
        'steady_state_s': round(t, 4),
        'candidates_per_s': round(n * sw.K_CANDIDATES / t, 1),
        'ms_per_anchor': round(1000.0 * t / n, 3),
        'rss_mb': round(rss_mb(), 1)})
    print('n=%-5d %8d cand | %7.3f s | %9.1f cand/s | %8.3f ms/anchor '
          '| RSS %7.1f MB | compile %6.2f s'
          % (n, n * sw.K_CANDIDATES, t, n * sw.K_CANDIDATES / t,
             1000.0 * t / n, rss_mb(), t_compile))

  best = min(rows, key=lambda r: r['ms_per_anchor'])
  ms_anchor = best['ms_per_anchor']
  # implied overhead if EVERY anchor of every sampled batch needed one s'_wc
  per_update_s = BASE_BATCH * BASE_SGD_PER_STEP * ms_anchor / 1000.0
  rep = {
      'frozen_parameters': {'K': sw.K_CANDIDATES, 'ode_steps': sw.ODE_STEPS,
                            'note': 'never reduced for speed'},
      'backend': jax.default_backend(),
      'devices': [str(d) for d in jax.devices()],
      'baseline_batch': BASE_BATCH,
      'baseline_sgd_steps_per_step': BASE_SGD_PER_STEP,
      'baseline_rss_mb_at_start': round(base_rss, 1),
      'measurements': rows,
      'best_batched': best,
      'implied_overhead': {
          'seconds_per_learner_step_if_every_anchor_needs_swc':
              round(per_update_s, 2),
          'hours_for_300k_steps_full_rate':
              round(per_update_s * 300000 / 3600.0, 1),
          'caveat': ('full-rate is the WORST case: the archived worst-case '
                     'semantics fire the branch only on the pessimistic coin '
                     'and only at walk steps, not once per anchor per update. '
                     'The true rate cannot be computed until the branch-firing '
                     'rule (audit gate G2) is decided.')},
      'precompute_analysis': _precompute(ms_anchor),
      'git_commit': subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()}
  json.dump(rep, open(os.path.join(OUT, 'profile.json'), 'w'), indent=2)
  print('\nbest batched: %.3f ms/anchor at n=%d (%s backend)'
        % (ms_anchor, best['anchors'], jax.default_backend()))
  print('full-rate overhead: %.2f s per learner step -> %.1f h for 300k'
        % (per_update_s, per_update_s * 300000 / 3600.0))
  pc = rep['precompute_analysis']
  print('precompute whole frozen dataset (%d transitions): %.2f h here, '
        '%.3f h at 50x GPU, cache %.3f GB -> RL overhead becomes ZERO'
        % (pc['n_transitions'], pc['one_time_precompute_hours_this_cpu'],
           pc['one_time_precompute_hours_at_50x_gpu'],
           pc['cache_size_gb_float32']))
  print('saved -> %s' % os.path.join(OUT, 'profile.json'))


if __name__ == '__main__':
  main()
