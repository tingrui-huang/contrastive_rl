"""Which state ENCODING is responsible for the action-Lipschitz blow-up?

Diagnostic only. No training, no CRL change, no dynamics change, no PGD, no
spectral norm. Nothing here writes to any existing z_v0 / z_v1 artifact.

THE QUESTION. scripts/audit_lipschitz_z_v1.py established Case B: with the
learner state s~ = (x, y, 2z) the ratio

    R = || s~'(a) - s~'(a') ||  /  || a - a' ||

is bounded on event-PRESERVING pairs (p99 = 1.00004) and unbounded on
event-SWITCHING pairs (median 24.02 at eps = 0.01). The open question raised
afterwards was whether that blow-up is a property of the SINKING-DEPTH
encoding specifically, and would go away under one of the two alternatives
suggested instead of a third spatial dimension:

    (a) a TERMINATION flag  d in {0, 1}
    (b) a FULL TIMESTAMP    t

This script answers it by measuring, not by arguing.

DESIGN -- ONE SIMULATION, FIVE READOUTS. Every encoding below is a pure
post-hoc function of the SAME paired rollout, so the anchors, the hidden swamp
realisation U, the action-noise realisation xi and the perturbation directions
are byte-identical across encodings. Nothing can differ between two rows of the
result table except the encoding itself.

    xy      (x, y)                     the original 2-D learner state
    z_v1    (x, y, z/|z_min|)          current design, one sink increment/step
    z_v0    (x, y, z_v0/|z_min|)       the superseded design, full sink in one
                                       step; derived as -0.5*dead and CHECKED
                                       against the real v0 env in section 2
    term    (x, y, d)                  termination flag, d = 1 iff dead
    time    (x, y, t/(L-1))            full timestamp

SCALE IS PART OF THE CLAIM, NOT A NUISANCE. R is not scale-free: multiplying
the extra coordinate by c multiplies its contribution to R by c. A comparison
across encodings is therefore only meaningful at a COMMON normalisation, and
the one used here is unit-range: each extra coordinate is divided by the width
of its own reachable set (|z_min| = 0.5 for the depths, 1 for the flag, L-1 for
the timestamp), so every encoding's third coordinate spans exactly [0, 1]. This
is the same z_physical scaling the accepted audit used (2.0 = 1/|z_min|), so
the z_v1 column is directly comparable to the published Case B numbers. The raw
unnormalised jump sizes are reported alongside so the choice can be re-examined.

COMMON RANDOM NUMBERS -- reused, not reimplemented. one_step() below pins the
position, z, dead flag, swamp bits and the RNG object exactly as
scripts/audit_lipschitz_z_v1.py:one_step does, and section 1 re-runs that
script's CRN validation before any measurement is taken. The only difference is
that this version also returns the dead flag, which the encodings need.

WHAT EACH SECTION DECIDES

  1  CRN validation                 (gate; identical to the accepted audit)
  2  z_v0 derivation check          (gate; -0.5*dead vs the real v0 env)
  3  R vs eps per encoding          PANEL A -- does the blow-up survive?
  4  event-preserving/switching     the decisive split, per encoding
  5  multi-step XY divergence       PANEL B -- is the divergence continuous
                                    when read over a rollout instead of one
                                    step? This is the constructive half.
  6  separability at matched XY     PANEL C -- the other column of the claim:
                                    an encoding that does not jump is only
                                    useful if it still resolves the 2-D
                                    ambiguity that motivated the third
                                    dimension in the first place.

PANEL C IS TAUTOLOGICAL FOR THREE OF THE FIVE ROWS and is reported anyway
rather than quietly dropped: term and z_v0 ARE the death label up to an affine
map, and z_v1 is a 5-level function of it, so a dead/alive classifier built on
them is trivially perfect. The informative cells are `xy` (the original
failure) and `time` (the open question).

Usage:
  python scripts/audit_encoding_lipschitz.py --anchors 200      # smoke
  python scripts/audit_encoding_lipschitz.py                    # full
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

ENV_V1 = 'point_two_route_swamp_windy_z_v1'
ENV_V0 = 'point_two_route_swamp_windy_z_v0'
DATASET = 'datasets/swamp_windy_z_v1_merged_s0.npz'
OUT_DIR = 'artifacts/encoding_lipschitz'
TRACKED = ['crl/envs.py', 'scripts/audit_encoding_lipschitz.py']
EPS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3)
Z_MIN = 0.5                                       # |z_min|; unit-range divisor
ROLLOUT_K = 8
PCTS = (50, 90, 95, 99, 99.5)

ENCODINGS = ('xy', 'z_v1', 'z_v0', 'term', 'time')


def extra_coord(name, z, dead, t_norm):
  """Unit-range third coordinate for each encoding (None = 2-D, no extra)."""
  if name == 'xy':
    return None
  if name == 'z_v1':
    return z / Z_MIN                              # 0, -0.24, ..., -1.0
  if name == 'z_v0':
    return -1.0 * dead                            # 0 or -1.0
  if name == 'term':
    return 1.0 * dead                             # 0 or +1.0
  if name == 'time':
    return t_norm                                 # identical on both legs
  raise ValueError(name)


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


def row(name, d):
  if d.get('n', 0) == 0:
    return '  %-24s%9s' % (name, '-')
  return ('  %-24s%9s%10.4f%10.4f%10.4f%10.4f%12.4f'
          % (name, format(d['n'], ','), d['mean'], d['median'], d['p95'],
             d['p99'], d['max']))


HDR = ('  %-24s%9s%10s%10s%10s%10s%12s'
       % ('', 'n', 'mean', 'median', 'p95', 'p99', 'max'))


def one_step(env, xy, bits, action, noise_seed):
  """S'(action; U=bits, xi=noise_seed) -> (x, y, z, dead).

  Identical pinning to scripts/audit_lipschitz_z_v1.py:one_step -- position, z,
  dead flag, swamp bits and the RNG the action noise is drawn from -- with the
  dead flag also returned. _auto_resample stays off so U cannot move mid-pair.
  """
  env.set_auto_resample(False)
  env._dead = False
  env._z = 0.0
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = np.random.default_rng(int(noise_seed))
  obs, _, _, _ = env.step(np.asarray(action, np.float32))
  return float(obs[0]), float(obs[1]), float(obs[2]), bool(env._dead)


def rollout(env, xy, bits, actions, noise_seed):
  """k-step CRN rollout from a clean alive state. Returns [k, 4]."""
  env.set_auto_resample(False)
  env._dead = False
  env._z = 0.0
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = np.random.default_rng(int(noise_seed))
  out = []
  for a in actions:
    obs, _, _, _ = env.step(np.asarray(a, np.float32))
    out.append((float(obs[0]), float(obs[1]), float(obs[2]), float(env._dead)))
  return np.array(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--anchors', type=int, default=4000,
                  help='per anchor GROUP (uniform and at-risk)')
  ap.add_argument('--dirs', type=int, default=6,
                  help='random unit directions per anchor (4 axis dirs added)')
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
  out = {'analysis_script': 'scripts/audit_encoding_lipschitz.py',
         'env': ENV_V1, 'dataset': args.dataset,
         'dataset_content_sha256': csha(args.dataset),
         # the commit that last touched the AUDITED files, not HEAD at run time
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'encodings': list(ENCODINGS), 'epsilons': list(EPS),
         'normalisation': 'unit-range: z/%g, dead/1, t/(L-1)' % Z_MIN,
         'config': {'anchors_per_group': args.anchors, 'dirs': args.dirs,
                    'seed': args.seed, 'rollout_k': ROLLOUT_K}}

  print('=' * 104)
  print('ENCODING COMPARISON FOR THE ACTION-LIPSCHITZ DISCONTINUITY')
  print('=' * 104)
  print('  dataset sha  %s' % out['dataset_content_sha256'])
  print('  code commit  %s  dirty %s' % (out['code_commit'][:12], out['dirty']))
  print('  normalisation: %s' % out['normalisation'])

  cfg = Config(env_name=ENV_V1)
  env = envs_mod.make_env(ENV_V1, cfg, seed=args.seed)

  # ------------------------------------------------------------------ 1 CRN
  print('\n1. COMMON-RANDOM-NUMBER VALIDATION  (gate)')
  xy0 = np.array([2.5, 3.5])
  a0 = np.array([0.8, -0.1], np.float32)
  r1 = one_step(env, xy0, [1, 0, 0], a0, 12345)
  r2 = one_step(env, xy0, [1, 0, 0], a0, 12345)
  r3 = one_step(env, xy0, [1, 0, 0], a0, 999)
  rc = one_step(env, xy0, [0, 0, 0], a0, 12345)
  same = float(np.abs(np.array(r1[:3]) - np.array(r2[:3])).max())
  diff = float(np.abs(np.array(r1[:3]) - np.array(r3[:3])).max())
  print('  same action + same seed  -> max|diff| %.3e   (must be 0)' % same)
  print('  same action + diff seed  -> max|diff| %.3e   (must be > 0)' % diff)
  print('  same action+seed, U flipped -> z clear %.3f dead %s | active %.3f '
        'dead %s' % (rc[2], rc[3], r1[2], r1[3]))
  assert same == 0.0 and diff > 0.0, 'CRN pairing is not reproducible'
  out['1_crn'] = {'identical_seed_max_diff': same,
                  'different_seed_max_diff': diff,
                  'z_clear': rc[2], 'dead_clear': rc[3],
                  'z_active': r1[2], 'dead_active': r1[3], 'validated': True}
  print('  CRN VALIDATED')

  # ------------------------------------------------------------- 2 z_v0 gate
  print('\n2. z_v0 DERIVATION CHECK  (gate)')
  print('  claim: on the contact step the real v0 env gives z = -0.5 exactly,')
  print('  so z_v0 = -0.5 * dead is exact rather than an approximation.')
  env0 = envs_mod.make_env(ENV_V0, Config(env_name=ENV_V0), seed=args.seed)
  rng0 = np.random.default_rng(args.seed)
  worst, n_chk = 0.0, 0
  for _ in range(400):
    xy = np.array([rng0.uniform(2.0, 6.0), rng0.uniform(3.0, 4.0)])
    bits = rng0.integers(0, 2, 3).tolist()
    action = rng0.uniform(-1, 1, 2).astype(np.float32)
    ns = int(rng0.integers(1 << 30))
    x1, y1, _, d1 = one_step(env, xy, bits, action, ns)
    x0, y0, z0, d0 = one_step(env0, xy, bits, action, ns)
    assert (x0, y0, d0) == (x1, y1, d1), 'v0 and v1 XY/death disagree'
    worst = max(worst, abs(z0 - (-0.5 * d0)))
    n_chk += 1
  print('  checked %d paired v0/v1 steps: XY and death identical; '
        'max|z_v0_env - (-0.5*dead)| = %.3e' % (n_chk, worst))
  assert worst == 0.0, 'z_v0 = -0.5*dead is not exact'
  out['2_zv0_derivation'] = {'n_checked': n_chk, 'max_abs_error': worst,
                             'xy_and_death_identical': True}
  print('  z_v0 DERIVATION EXACT')

  # ---------------------------------------------------------------- anchors
  with np.load(args.dataset, allow_pickle=False) as d:
    obs, act, bits = d['obs'], d['act'], d['swamp_bits']
  n_ep, L, _ = obs.shape
  s = obs[:, :-1, :3].reshape(-1, 3)
  a = act[:, :-1, :].reshape(-1, 2)
  bt = bits[:, :-1, :].reshape(-1, 3)
  tt = np.tile(np.arange(L - 1), n_ep)             # step index of each anchor
  ep = np.repeat(np.arange(n_ep), L - 1)
  alive = s[:, 2] == 0.0
  near = alive & (s[:, 0] >= 2.0) & (s[:, 0] < 6.0) & (s[:, 1] >= 3.0) & (
      s[:, 1] < 4.0)
  rng = np.random.default_rng(args.seed)
  groups = {'uniform': rng.choice(np.where(alive)[0],
                                  min(args.anchors, int(alive.sum())), False),
            'at_risk': rng.choice(np.where(near)[0],
                                  min(args.anchors, int(near.sum())), False)}
  print('\n  anchors: uniform %s (of %s alive) | at_risk %s (of %s in the '
        'holding cell + corridor)'
        % (format(len(groups['uniform']), ','), format(int(alive.sum()), ','),
           format(len(groups['at_risk']), ','), format(int(near.sum()), ',')))
  print('  NB at_risk is deliberately OVERSAMPLED near the swamp; its '
        'marginals are\n  not dataset-representative.')

  # ----------------------------------------------------------------- 3/4 sim
  dirs = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]),
          np.array([0., -1.])]
  E, DA, SW = [], [], []
  DXY, DZ, DD = [], [], []                         # paired deltas
  keep = []                                        # for the rollout panel
  t0 = time.time()
  for gname, idx in groups.items():
    print('\n  simulating group %s ...' % gname, flush=True)
    for c, i in enumerate(idx):
      xy, ap_, u, ns = s[i, :2], a[i], bt[i], int(rng.integers(1 << 30))
      bx, by, bz, bd = one_step(env, xy, u, ap_, ns)
      vs = list(dirs) + [v / np.linalg.norm(v)
                         for v in rng.normal(size=(args.dirs, 2))]
      for eps in EPS:
        for v in vs:
          an = np.clip(ap_ + eps * v, -1.0, 1.0).astype(np.float32)
          da = float(np.linalg.norm(an - ap_))
          if da < 1e-9:
            continue
          ax, ay, az, ad = one_step(env, xy, u, an, ns)
          E.append(eps)
          DA.append(da)
          DXY.append(np.hypot(ax - bx, ay - by))
          DZ.append(az - bz)
          DD.append(float(ad) - float(bd))
          SW.append(ad != bd)
          if ad != bd:
            keep.append((int(ep[i]), int(tt[i]), int(ns), float(ap_[0]),
                         float(ap_[1]), float(an[0]), float(an[1]),
                         float(eps), float(u[0]), float(u[1]), float(u[2]),
                         float(xy[0]), float(xy[1])))
      if (c + 1) % 1000 == 0:
        print('    %d/%d anchors  (%.0fs)'
              % (c + 1, len(idx), time.time() - t0), flush=True)

  E = np.array(E)
  DA = np.array(DA)
  SW = np.array(SW)
  DXY = np.array(DXY)
  DZ = np.array(DZ)
  DD = np.array(DD)
  print('\n  pairs: %s total, %s event-switching (%.5f)   [%.0fs]'
        % (format(len(DA), ','), format(int(SW.sum()), ','), SW.mean(),
           time.time() - t0))
  out['pairs'] = {'total': int(len(DA)), 'event_switching': int(SW.sum()),
                  'event_switch_fraction': float(SW.mean())}

  # per-encoding R.  time contributes an EXACTLY zero delta: both legs are the
  # same step of the same anchor, so t' = t + 1 on both.
  delta = {'xy': np.zeros(len(DA)),
           'z_v1': DZ / Z_MIN,
           'z_v0': -1.0 * DD,
           'term': 1.0 * DD,
           'time': np.zeros(len(DA))}
  R = {k: np.sqrt(DXY ** 2 + v ** 2) / DA for k, v in delta.items()}

  print('\n3. R BY ENCODING  (all pairs, unit-range normalisation)')
  print(HDR)
  for k in ENCODINGS:
    print(row(k, stats(R[k])))
  out['3_overall'] = {k: stats(R[k]) for k in ENCODINGS}
  ident = float(np.abs(R['time'] - R['xy']).max())
  print('  max|R_time - R_xy| = %.3e   (timestamp is action-insensitive: '
        'delta t = 0 on both legs)' % ident)
  out['3_time_equals_xy_max_abs_diff'] = ident

  print('\n4. EVENT-PRESERVING vs EVENT-SWITCHING, BY ENCODING')
  print(HDR)
  for k in ENCODINGS:
    print(row(k + '  preserving', stats(R[k][~SW])))
    print(row(k + '  SWITCHING', stats(R[k][SW])))
  out['4_events'] = {k: {'preserving': stats(R[k][~SW]),
                         'switching': stats(R[k][SW])} for k in ENCODINGS}

  print('\n4b. SWITCHING-PAIR MEDIAN R vs eps   (the blow-up law)')
  hdr = '  %-8s%10s%12s' % ('eps', 'n_switch', 'switch_frac')
  for k in ENCODINGS:
    hdr += '%11s' % k
  print(hdr)
  by_eps = {}
  for e in EPS:
    m = (E == e)
    sm = m & SW
    rec = {'n': int(m.sum()), 'n_switch': int(sm.sum()),
           'switch_fraction': float(SW[m].mean())}
    line = '  %-8.3f%10s%12.5f' % (e, format(int(sm.sum()), ','), SW[m].mean())
    for k in ENCODINGS:
      val = float(np.median(R[k][sm])) if sm.any() else None
      rec['median_R_switching_' + k] = val
      rec['median_R_preserving_' + k] = (float(np.median(R[k][m & ~SW]))
                                         if (m & ~SW).any() else None)
      line += '%11s' % ('%.3f' % val if val is not None else '-')
    by_eps['%g' % e] = rec
    print(line)
  print('\n  predicted lower bound for a jump of size delta:  R >= delta / eps')
  print('  %-24s' % 'eps' + ''.join('%11.3f' % e for e in EPS))
  print('  %-24s' % 'delta/eps, delta=0.24'
        + ''.join('%11.2f' % (0.24 / e) for e in EPS))
  print('  %-24s' % 'delta/eps, delta=1.00'
        + ''.join('%11.2f' % (1.0 / e) for e in EPS))
  out['4b_by_epsilon'] = by_eps

  print('\n4c. THE XY-ONLY TAIL ON EVENT-PRESERVING PAIRS')
  print('  The swamp is not the only discontinuity in this env: the substep')
  print('  loop REJECTS a move that would enter a wall (_is_blocked), so two')
  print('  nearby actions can differ by a whole rejected substep. That shows up')
  print('  as an XY tail on pairs where NEITHER leg died, i.e. it is present in')
  print('  the ORIGINAL 2-D state and has nothing to do with any z encoding.')
  print('  If this tail also follows c/eps then no encoding, z or not, admits a')
  print('  global realized-state L in this maze.')
  print('  %-8s%10s%12s%12s%12s%12s'
        % ('eps', 'n', 'median', 'p99', 'p99.9', 'max'))
  wall = {}
  for e in EPS:
    m = (E == e) & ~SW
    v = R['xy'][m]
    if v.size == 0:
      continue
    wall['%g' % e] = {'n': int(v.size), 'median': float(np.median(v)),
                      'p99': float(np.percentile(v, 99)),
                      'p99.9': float(np.percentile(v, 99.9)),
                      'max': float(v.max())}
    print('  %-8.3f%10s%12.4f%12.4f%12.4f%12.4f'
          % (e, format(v.size, ','), np.median(v), np.percentile(v, 99),
             np.percentile(v, 99.9), v.max()))
  out['4c_xy_tail_preserving'] = wall

  # --------------------------------------------------------- 5 rollout panel
  print('\n5. PANEL B -- MULTI-STEP XY DIVERGENCE ON SWITCHING PAIRS')
  print('  Both legs continue for %d further steps with the SAME logged action'
        % ROLLOUT_K)
  print('  sequence and the SAME noise stream; U is held fixed across the '
        'rollout,')
  print('  which is the paired-counterfactual object, not the per-step '
        'resampling.')
  print('  REPORTED AS A RATIO |dXY_k| / ||a - a\'||, not as a raw distance: a')
  print('  raw distance that grows smoothly proves nothing, because the')
  print('  question is whether the RATIO stays bounded as eps -> 0. The freeze')
  print('  makes one leg stop while the other keeps moving, so |dXY_k| tends to')
  print('  an O(1) gap that does NOT shrink with eps -- if that is what happens,')
  print('  the multi-step reading blows up too and does not rescue the bound.')
  acc = {k: [] for k in range(1, ROLLOUT_K + 1)}
  ratio = {k: [] for k in range(1, ROLLOUT_K + 1)}
  reps = []
  n_roll = 0
  for (e_i, t_i, ns, b0, b1, n0, n1, _eps, u0, u1, u2, x_, y_) in keep:
    if t_i + ROLLOUT_K >= L - 1:
      continue
    da_pair = float(np.hypot(n0 - b0, n1 - b1))
    fut = act[e_i, t_i + 1:t_i + ROLLOUT_K, :]
    seq_b = np.concatenate([np.array([[b0, b1]]), fut], 0)
    seq_a = np.concatenate([np.array([[n0, n1]]), fut], 0)
    u = [u0, u1, u2]
    tb = rollout(env, (x_, y_), u, seq_b, ns)
    ta = rollout(env, (x_, y_), u, seq_a, ns)
    for k in range(1, len(tb) + 1):
      dk = float(np.hypot(ta[k - 1, 0] - tb[k - 1, 0],
                          ta[k - 1, 1] - tb[k - 1, 1]))
      acc[k].append(dk)
      ratio[k].append(dk / da_pair)
    reps.append(_eps)
    n_roll += 1
  reps = np.array(reps)
  print('  rolled out %s switching pairs (of %s; the rest are too close to '
        'the episode end)' % (format(n_roll, ','), format(len(keep), ',')))
  print('  %-6s%10s%13s%13s%13s%13s'
        % ('k', 'n', 'mean|dXY|', 'median|dXY|', 'median RATIO', 'max RATIO'))
  roll = {}
  for k in range(1, ROLLOUT_K + 1):
    v, q = np.array(acc[k]), np.array(ratio[k])
    if v.size == 0:
      continue
    roll[str(k)] = {'n': int(v.size), 'mean': float(v.mean()),
                    'median': float(np.median(v)),
                    'p95': float(np.percentile(v, 95)),
                    'median_ratio': float(np.median(q)),
                    'max_ratio': float(q.max())}
    print('  %-6d%10s%13.5f%13.5f%13.4f%13.4f'
          % (k, format(v.size, ','), v.mean(), np.median(v), np.median(q),
             q.max()))
  out['5_rollout_xy_divergence'] = roll

  print('\n  the decisive cut -- ratio at k = %d, split by eps:' % ROLLOUT_K)
  print('  %-8s%10s%14s%14s%14s'
        % ('eps', 'n', 'median|dXY|', 'median ratio', 'max ratio'))
  bye = {}
  qk = np.array(ratio[ROLLOUT_K])
  vk = np.array(acc[ROLLOUT_K])
  for e in EPS:
    m = reps == e
    if not m.any():
      continue
    bye['%g' % e] = {'n': int(m.sum()), 'median_dxy': float(np.median(vk[m])),
                     'median_ratio': float(np.median(qk[m])),
                     'max_ratio': float(qk[m].max())}
    print('  %-8.3f%10s%14.5f%14.4f%14.4f'
          % (e, format(int(m.sum()), ','), np.median(vk[m]),
             np.median(qk[m]), qk[m].max()))
  print('  if median|dXY| stays ~constant across eps while the ratio scales'
        ' like 1/eps,')
  print('  then the multi-step reading does NOT restore a finite L either.')
  out['5b_rollout_ratio_by_eps'] = bye

  # ---------------------------------------------------- 6 separability panel
  print('\n6. PANEL C -- DEAD/ALIVE SEPARABILITY AT MATCHED XY')
  print('  Region: the swamp cells x in [3,6), y in [3,4) -- where the 2-D')
  print('  collision that motivated the third dimension actually happens.')
  reg = (s[:, 0] >= 3.0) & (s[:, 0] < 6.0) & (s[:, 1] >= 3.0) & (s[:, 1] < 4.0)
  dead_m = reg & (s[:, 2] < 0.0)
  live_m = reg & (s[:, 2] == 0.0)
  nd, nl = int(dead_m.sum()), int(live_m.sum())
  m = min(nd, nl, 6000)
  di = rng.choice(np.where(dead_m)[0], m, False)
  li = rng.choice(np.where(live_m)[0], m, False)
  print('  %s dead / %s alive states in the region; balanced subsample %s each'
        % (format(nd, ','), format(nl, ','), format(m, ',')))
  sep = {}
  tn = tt / float(L - 1)

  def feat(enc, ix):
    base = [s[ix, 0], s[ix, 1]]
    ex = extra_coord(enc, s[ix, 2], (s[ix, 2] < 0.0).astype(float), tn[ix])
    if ex is not None:
      base.append(ex)
    return np.stack(base, 1)

  for k in ENCODINGS:
    X = np.concatenate([feat(k, di), feat(k, li)], 0)
    y = np.concatenate([np.ones(m), np.zeros(m)])
    tr = rng.random(2 * m) < 0.5
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]
    pred = np.empty(len(Xte))
    for b in range(0, len(Xte), 512):
      d2 = ((Xte[b:b + 512, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
      pred[b:b + 512] = ytr[d2.argmin(1)]
    accy = float((pred == yte).mean())
    Xd, Xl = feat(k, di), feat(k, li)
    nn = np.empty(len(Xd))
    for b in range(0, len(Xd), 512):
      d2 = ((Xd[b:b + 512, None, :] - Xl[None, :, :]) ** 2).sum(-1)
      nn[b:b + 512] = np.sqrt(d2.min(1))
    taut = k in ('z_v1', 'z_v0', 'term')
    sep[k] = {'nn_accuracy': accy,
              'median_nn_dist_dead_to_alive': float(np.median(nn)),
              'frac_dead_within_0.05_of_an_alive': float((nn < 0.05).mean()),
              'label_derived': taut}
  print('  CAVEAT ON 1NN ACC: with thousands of points in one maze cell a 1NN')
  print('  can memorise fine-grained position and score well even when the two')
  print('  classes overlap. The honest column is the NEAREST-ALIVE DISTANCE:')
  print('  if a dead state has an alive state essentially on top of it, the')
  print('  encoding has not separated them whatever 1NN reports.')
  print('  %-8s%12s%20s%24s' % ('enc', '1NN acc', 'med d(dead,alive)',
                                'frac dead within .05'))
  for k in ENCODINGS:
    v = sep[k]
    print('  %-8s%12.4f%20.4f%24.4f  %s'
          % (k, v['nn_accuracy'], v['median_nn_dist_dead_to_alive'],
             v['frac_dead_within_0.05_of_an_alive'],
             'TAUTOLOGICAL (label-derived)' if v['label_derived'] else ''))
  out['6_separability'] = sep

  # the sharp version of the timestamp question
  cx = np.floor(s[:, 0] * 4).astype(np.int64)
  cy = np.floor(s[:, 1] * 4).astype(np.int64)
  key = (cx << 40) + (cy << 20) + tt.astype(np.int64)
  kd = np.unique(key[dead_m])
  kl = np.unique(key[live_m])
  both = np.intersect1d(kd, kl)
  frac_amb = float(np.isin(key[dead_m], both).mean())
  print('\n  sharp form: bin XY to 0.25 and hold the timestamp EXACTLY equal.')
  print('  %s of %s (XY-bin, t) cells that contain a dead state ALSO contain '
        'an alive one;' % (format(len(both), ','), format(len(kd), ',')))
  print('  %.4f of dead states sit in such a cell -- (x, y, t) does not '
        'separate them.' % frac_amb)
  out['6_xyt_ambiguity'] = {'xy_bin': 0.25, 'n_cells_with_dead': int(len(kd)),
                            'n_cells_with_both': int(len(both)),
                            'frac_dead_states_ambiguous': frac_amb}

  p = os.path.join(args.out_dir, 'encoding_lipschitz.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
