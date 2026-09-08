r"""When does the frame-stacked observation separate, and at what jump size?

Diagnostic only. No training, no CRL change, no dynamics change. Nothing here
writes to any existing 2-D, z_v0, z_v1, encoding, Manski or Lipschitz artifact.

CONTEXT. scripts/audit_encoding_lipschitz.py compared five encodings on one
paired simulation and found the action-Lipschitz blow-up to be a property of
the EVENT, not of the depth encoding: on event-switching pairs the ratio
R = ||Delta obs|| / ||Delta a|| follows delta/eps with delta fixed by the
encoding (z_v1 0.24, z_v0 and a termination flag 1.00, xy and a timestamp 0).
The frame-stacked env is a sixth encoding and it does something none of those
did -- it carries no new physical quantity at all, only position history -- so
its delta and, more importantly, WHEN that delta appears, have to be measured
rather than read off the earlier table.

TWO PREDICTIONS, BOTH FALSIFIABLE HERE

  (1) AT THE CONTACT STEP the f4 observation is bit-identical to xy. Both legs
      of a CRN pair start from the same anchor with the same history, so frames
      1..3 cancel exactly and Delta obs = Delta s_t. R_f4(lag 0) should equal
      R_xy to 0.000e+00, not merely approximately. Gate 2 asserts it.

  (2) THE SEPARATION IS DELAYED, NOT ABSENT. One step later the dead leg is
      frozen and repeats s_t while the live leg moves on, so a gap of about one
      step of travel enters the stack. If so, f4 buys a clean contact step at
      the price of a lag -- the same trade the death-lag measurement in
      scripts/audit_swamp_windy_f4.py found from the other direction.

WHAT THIS DOES NOT CLAIM. A delayed jump is still a jump: if the ratio at lag 1
also follows c/eps then frame stacking does not escape Case B, it only moves
it. The point of measuring is to find out which, and the per-lag delta is
reported so f4 can be put in the same table as the other five encodings.

COMMON RANDOM NUMBERS. Both legs share the anchor, its stored frame history,
the hidden swamp bits and the action-noise stream, and continue with the SAME
subsequent logged actions, so the only difference between them is the single
perturbed action. Section 1 revalidates the CRN mechanism before measuring.

Usage:
  python scripts/audit_encoding_f4.py --anchors 300     # smoke
  python scripts/audit_encoding_f4.py                   # full
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

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402

ENV = 'point_two_route_swamp_windy_f4_v0'
DATASET = 'datasets/swamp_windy_f4_merged_s0.npz'
OUT_DIR = 'artifacts/encoding_f4'
TRACKED = ['crl/envs.py', 'scripts/audit_encoding_f4.py']
EPS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3)
NF = 4
LAGS = 4                       # how far past the contact step to follow
PCTS = (50, 90, 95, 99, 99.5)


def stats(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  o = {'n': int(v.size), 'mean': float(v.mean())}
  for p in PCTS:
    o['p%g' % p] = float(np.percentile(v, p))
  o['median'] = o.pop('p50')
  o['max'] = float(v.max())
  return o


def set_anchor(env, xy, frames, bits, noise_seed):
  """Pin position, frame history, hidden bits and the noise stream."""
  env.set_auto_resample(False)
  env._dead = False
  env.state = np.asarray(xy, float).copy()
  env._frames = [np.asarray(f, float).copy() for f in frames]
  env.set_swamp(bits)
  env._rng = np.random.default_rng(int(noise_seed))


def leg(env, xy, frames, bits, noise_seed, actions):
  """Run one leg and return [len(actions), 2*NF] stacked observations."""
  set_anchor(env, xy, frames, bits, noise_seed)
  out = []
  for a in actions:
    env.step(np.asarray(a, np.float32))
    out.append(env.state_f4.copy())
  return np.asarray(out, np.float64)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--anchors', type=int, default=3000,
                  help='per anchor GROUP (uniform and at-risk)')
  ap.add_argument('--dirs', type=int, default=4)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--dataset', default=DATASET)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(['git'] + list(a),
                                     cwd=_ROOT).decode().strip()
    except Exception:                             # pylint: disable=broad-except
      return ''

  def csha(p):
    h = hashlib.sha256()
    with np.load(p, allow_pickle=False) as d:
      for k in sorted(d.files):
        x = d[k]
        h.update(k.encode())
        h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

  os.makedirs(args.out_dir, exist_ok=True)
  out = {'analysis_script': 'scripts/audit_encoding_f4.py', 'env': ENV,
         'dataset': args.dataset, 'dataset_content_sha256': csha(args.dataset),
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'epsilons': list(EPS), 'lags': LAGS,
         'config': {'anchors_per_group': args.anchors, 'dirs': args.dirs,
                    'seed': args.seed}}

  print('=' * 104)
  print('FRAME STACKING: WHEN DOES IT SEPARATE, AND BY HOW MUCH?')
  print('=' * 104)
  print('  dataset sha %s' % out['dataset_content_sha256'])
  print('  code commit %s  dirty %s' % (out['code_commit'][:12], out['dirty']))

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)

  # ------------------------------------------------------------------ 1 CRN
  print('\n1. CRN VALIDATION  (gate)')
  xy0 = np.array([2.5, 3.5])
  fr0 = [xy0.copy() for _ in range(NF)]
  a0 = [np.array([0.8, -0.1], np.float32)]
  r1 = leg(env, xy0, fr0, [1, 0, 0], 12345, a0)
  r2 = leg(env, xy0, fr0, [1, 0, 0], 12345, a0)
  r3 = leg(env, xy0, fr0, [1, 0, 0], 999, a0)
  same = float(np.abs(r1 - r2).max())
  diff = float(np.abs(r1 - r3).max())
  print('  same action + same seed  -> max|diff| %.3e   (must be 0)' % same)
  print('  same action + diff seed  -> max|diff| %.3e   (must be > 0)' % diff)
  assert same == 0.0 and diff > 0.0, 'CRN pairing is not reproducible'
  print('  CRN VALIDATED')
  out['1_crn'] = {'identical_seed_max_diff': same,
                  'different_seed_max_diff': diff, 'validated': True}

  # ---------------------------------------------------------------- anchors
  with np.load(args.dataset, allow_pickle=False) as d:
    obs, act, bits = d['obs'], d['act'], d['swamp_bits']
  n_ep, L, _ = obs.shape
  st = obs[:, :-1, :2 * NF].reshape(-1, 2 * NF)        # the whole stack
  a = act[:, :-1, :].reshape(-1, 2)
  bt = bits[:, :-1, :].reshape(-1, 3)
  ep = np.repeat(np.arange(n_ep), L - 1)
  tt = np.tile(np.arange(L - 1), n_ep)
  pos = st[:, :2]
  # "alive" = the stack is not yet fully frozen. It is the honest available
  # proxy: the f4 observation carries no dead flag, which is the whole point.
  moving = np.abs(st.reshape(-1, NF, 2)
                  - st.reshape(-1, NF, 2)[:, :1, :]).max(axis=(1, 2)) > 0
  ok = moving & (tt + LAGS < L - 1)
  near = ok & (pos[:, 0] >= 2.0) & (pos[:, 0] < 6.0) & (
      pos[:, 1] >= 3.0) & (pos[:, 1] < 4.0)
  rng = np.random.default_rng(args.seed)
  groups = {'uniform': rng.choice(np.where(ok)[0],
                                  min(args.anchors, int(ok.sum())), False),
            'at_risk': rng.choice(np.where(near)[0],
                                  min(args.anchors, int(near.sum())), False)}
  print('\n  anchors: uniform %s (of %s usable) | at_risk %s (of %s)'
        % (format(len(groups['uniform']), ','), format(int(ok.sum()), ','),
           format(len(groups['at_risk']), ','), format(int(near.sum()), ',')))

  # ------------------------------------------------------------------- 2, 3
  dirs = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]),
          np.array([0., -1.])]
  rows = []
  t0 = time.time()
  for gname, idx in groups.items():
    print('\n  simulating group %s ...' % gname, flush=True)
    for c, i in enumerate(idx):
      xy = pos[i]
      frames = st[i].reshape(NF, 2)
      ap_ = a[i]
      u = bt[i]
      ns = int(rng.integers(1 << 30))
      fut = act[int(ep[i]), int(tt[i]) + 1:int(tt[i]) + 1 + LAGS, :]
      vs = list(dirs) + [v / np.linalg.norm(v)
                         for v in rng.normal(size=(args.dirs, 2))]
      base = leg(env, xy, frames, u, ns,
                 np.concatenate([ap_[None, :], fut], 0))
      base_dead = None
      for eps in EPS:
        for v in vs:
          an = np.clip(ap_ + eps * v, -1.0, 1.0).astype(np.float32)
          da = float(np.linalg.norm(an - ap_))
          if da < 1e-9:
            continue
          alt = leg(env, xy, frames, u, ns,
                    np.concatenate([an[None, :], fut], 0))
          # the event is read from the SIMULATOR, audit-only, never from obs
          set_anchor(env, xy, frames, u, ns)
          env.step(ap_)
          d0 = env.dead
          set_anchor(env, xy, frames, u, ns)
          env.step(an)
          d1 = env.dead
          base_dead = d0
          r_lag = [float(np.linalg.norm(alt[k] - base[k]) / da)
                   for k in range(len(base))]
          dxy = float(np.linalg.norm(alt[0, :2] - base[0, :2]) / da)
          rows.append((gname, eps, da, d0 != d1, dxy) + tuple(r_lag))
      if (c + 1) % 500 == 0:
        print('    %d/%d anchors  (%.0fs)'
              % (c + 1, len(idx), time.time() - t0), flush=True)

  E = np.array([r[1] for r in rows])
  SW = np.array([r[3] for r in rows], bool)
  RXY = np.array([r[4] for r in rows])
  RL = np.array([[r[5 + k] for k in range(LAGS + 1)] for r in rows])
  print('\n  pairs: %s total, %s event-switching (%.5f)   [%.0fs]'
        % (format(len(E), ','), format(int(SW.sum()), ','), SW.mean(),
           time.time() - t0))
  out['pairs'] = {'total': int(len(E)), 'event_switching': int(SW.sum()),
                  'event_switch_fraction': float(SW.mean())}

  # ---------------------------------------------------------------------- 2
  print('\n2. PREDICTION 1 -- AT THE CONTACT STEP f4 IS BIT-IDENTICAL TO xy')
  gap = float(np.abs(RL[:, 0] - RXY).max())
  print('  max |R_f4(lag 0) - R_xy| = %.3e   over %s pairs'
        % (gap, format(len(E), ',')))
  print('  Both legs share the anchor and its history, so frames 1..%d cancel'
        % (NF - 1))
  print('  exactly and the difference is the newest frame alone.')
  assert gap == 0.0, 'f4 lag-0 is not bit-identical to xy'
  print('  CONFIRMED (exactly 0, not approximately)')
  out['2_lag0_equals_xy_max_abs_diff'] = gap

  # ---------------------------------------------------------------------- 3
  print('\n3. PREDICTION 2 -- IS THE SEPARATION DELAYED, AND DOES IT BLOW UP?')
  print('  Switching pairs only. delta is recovered as median(R) * eps: if the')
  print('  ratio follows delta/eps then this column is flat in eps and equals')
  print('  the jump size in observation units.')
  by = {}
  for lag in range(LAGS + 1):
    print('\n  lag %d %s' % (lag, '(the contact step)' if lag == 0 else ''))
    print('    %-8s%10s%12s%12s%12s' % ('eps', 'n_switch', 'median R',
                                        'p99 R', 'implied delta'))
    for e in EPS:
      m = (E == e) & SW
      if not m.any():
        continue
      med = float(np.median(RL[m, lag]))
      by['lag%d_eps%g' % (lag, e)] = {
          'lag': lag, 'eps': e, 'n': int(m.sum()), 'median_R': med,
          'p99_R': float(np.percentile(RL[m, lag], 99)),
          'implied_delta': med * e}
      print('    %-8.3f%10s%12.4f%12.4f%12.4f'
            % (e, format(int(m.sum()), ','), med,
               np.percentile(RL[m, lag], 99), med * e))
  out['3_by_lag_and_eps'] = by

  print('\n  event-preserving pairs, for contrast:')
  print('  %-8s%10s%12s%12s' % ('lag', 'n', 'median R', 'p99 R'))
  pres = {}
  for lag in range(LAGS + 1):
    v = RL[~SW, lag]
    pres['lag%d' % lag] = stats(v)
    print('  %-8d%10s%12.4f%12.4f'
          % (lag, format(v.size, ','), np.median(v), np.percentile(v, 99)))
  out['3_preserving_by_lag'] = pres

  p = os.path.join(args.out_dir, 'encoding_f4.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
