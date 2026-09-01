"""Action-Lipschitz audit of the ONE-STEP SIMULATOR transition in z_v1.

Diagnostic only. No off-diagonal residual training, no worst-case optimisation,
no PGD, no spectral norm, no CRL or dynamics change.

THE ASSUMPTION UNDER TEST

    || s~'(a) - s~'(a') ||  <=  L || a - a' ||        with  s~ = (x, y, 2z)

using the accepted physical Z scaling. This is a statement about changing the
ACTION FOR THE SAME UNDERLYING CONTEXT, so each pair holds fixed the current
state, the hidden swamp realisation U, the action-noise realisation and every
other source of environment randomness. Hidden variables are used here as an
ORACLE for the benchmark only; nothing here is or becomes a model input.

COMMON RANDOM NUMBERS. crl.envs adds the action noise inside step() as
`action += self._rng.normal(0, action_noise, 2)` BEFORE clipping, and it is
additive and independent of the action. So re-seeding self._rng identically
before each leg gives both legs the same noise vector xi, and the pair really
is S'(a'; U, xi) vs S'(a; U, xi) rather than two draws differing in action AND
noise. _auto_resample is switched off so U cannot move between legs. Section 3
validates this mechanism before any measurement is taken.

NOTE ON THE DIAGONAL LEG. The a' leg is a fresh re-simulation under a fresh
xi, NOT a reproduction of the logged next state -- the logged transition's own
noise draw is not recoverable per-transition without replaying each episode
through the collector's RNG. That is fine and is what the assumption asks for:
the quantity of interest is the PAIRED difference at common xi.

THE HAZARD THIS IS BUILT TO EXPOSE. The sink is a discontinuity: an
arbitrarily small action change can flip whether the step ends inside an active
swamp cell, moving z' from 0 to -0.12 and hence z~' by 0.24. For such a pair
R = ||delta s~|| / ||delta a|| grows without bound as epsilon -> 0. Averaging
those into one percentile would manufacture a finite L that does not exist, so
event-switching pairs are separated out and reported directly rather than
treated as outliers.

Usage:
  python scripts/audit_lipschitz_z_v1.py
  python scripts/audit_lipschitz_z_v1.py --anchors 3000 --dirs 6
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.15')

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from diag_action_lipschitz import wall_margin     # noqa: E402

ENV = 'point_two_route_swamp_windy_z_v1'
DATASET = 'datasets/swamp_windy_z_v1_merged_s0.npz'
MODELS = 'artifacts/swamp_windy_z_v1_transition_diag/models.pkl'
OUT_DIR = 'artifacts/swamp_windy_z_v1_lipschitz'
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
EPS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.30)
Z_SCALE = 2.0                                     # 1/|z_min|, the accepted one
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


def row(name, d):
  if d.get('n', 0) == 0:
    return '  %-30s%9s' % (name, '-')
  return ('  %-30s%9s%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%12.3f'
          % (name, format(d['n'], ','), d['mean'], d['median'], d['p90'],
             d['p95'], d['p99'], d['p99.5'], d['max']))


HDR = ('  %-30s%9s%10s%10s%10s%10s%10s%10s%12s'
       % ('', 'n', 'mean', 'median', 'p90', 'p95', 'p99', 'p99.5', 'max'))


def one_step(env, xy, bits, action, noise_seed):
  """S'(action; U=bits, xi=noise_seed) from a clean alive state at xy.

  Everything that could differ between the two legs of a pair is pinned here:
  position, z, dead flag, the swamp bits, and the RNG the action noise is drawn
  from. _auto_resample stays off so the bits cannot move mid-pair.
  """
  env.set_auto_resample(False)
  env._dead = False
  env._z = 0.0
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = np.random.default_rng(int(noise_seed))
  obs, _, _, _ = env.step(np.asarray(action, np.float32))
  return np.array([obs[0], obs[1], obs[2]], float)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--anchors', type=int, default=2500,
                  help='per anchor GROUP (uniform and at-risk)')
  ap.add_argument('--dirs', type=int, default=6,
                  help='random unit directions per anchor (axis dirs added)')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(
          ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''

  def csha(p):
    h = hashlib.sha256()
    with np.load(p, allow_pickle=False) as d:
      for k in sorted(d.files):
        x = d[k]
        h.update(k.encode()); h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

  out = {'analysis_script': 'scripts/audit_lipschitz_z_v1.py', 'env': ENV,
         'dataset': DATASET, 'dataset_content_sha256': csha(DATASET),
         'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
         'head_at_runtime': git('rev-parse', 'HEAD'),
         'dirty': bool(git('status', '--porcelain', '--', 'crl', 'scripts')),
         'z_scale': Z_SCALE, 'epsilons': list(EPS),
         'config': {'anchors_per_group': args.anchors, 'dirs': args.dirs,
                    'seed': args.seed}}
  print('=' * 110)
  print('Z-v1 ACTION-LIPSCHITZ AUDIT  (simulator, common random numbers)')
  print('=' * 110)
  print('  dataset sha %s' % out['dataset_content_sha256'])
  print('  code commit %s  dirty %s' % (out['code_commit'], out['dirty']))
  print('  scaled state s~ = (x, y, %gz)' % Z_SCALE)

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)

  # ------------------------------------------------------------------ 3
  print('\n3. COMMON-RANDOM-NUMBER VALIDATION')
  xy0 = np.array([2.5, 3.5])
  a0 = np.array([0.8, -0.1], np.float32)
  r1 = one_step(env, xy0, [1, 0, 0], a0, 12345)
  r2 = one_step(env, xy0, [1, 0, 0], a0, 12345)
  r3 = one_step(env, xy0, [1, 0, 0], a0, 999)
  same = float(np.abs(r1 - r2).max())
  diff = float(np.abs(r1 - r3).max())
  print('  same action + same seed  -> max|diff| %.3e   (must be 0)' % same)
  print('  same action + diff seed  -> max|diff| %.3e   (must be > 0: proves'
        ' the noise is live)' % diff)
  # U must matter, action noise held fixed
  rc = one_step(env, xy0, [0, 0, 0], a0, 12345)
  print('  same action+seed, U flipped -> z clear %.3f vs active %.3f'
        % (rc[2], r1[2]))
  crn_ok = (same == 0.0) and (diff > 0.0)
  assert crn_ok, 'CRN pairing is not reproducible'
  out['3_crn'] = {'identical_seed_max_diff': same,
                  'different_seed_max_diff': diff,
                  'z_clear': float(rc[2]), 'z_active': float(r1[2]),
                  'validated': bool(crn_ok)}
  print('  CRN VALIDATED')

  # ---------------------------------------------------------------- anchors
  with np.load(DATASET, allow_pickle=False) as d:
    obs, act, bits = d['obs'], d['act'], d['swamp_bits']
  n, L, _ = obs.shape
  s = obs[:, :-1, :3].reshape(-1, 3)
  a = act[:, :-1, :].reshape(-1, 2)
  bt = bits[:, :-1, :].reshape(-1, 3)
  alive = s[:, 2] == 0.0
  # "at-risk" anchors are chosen from the anchor's OWN position only -- the
  # holding cell and the corridor -- so no next-state information is used.
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
  print('  NB the at_risk group is deliberately OVERSAMPLED near the swamp; '
        'its\n  marginals are not dataset-representative and are reported '
        'separately.')

  # ------------------------------------------------------------------ 2/4
  dirs = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]),
          np.array([0., -1.])]
  rows = []
  for gname, idx in groups.items():
    print('\n  simulating group %s ...' % gname)
    for c, i in enumerate(idx):
      xy = s[i, :2]
      ap_ = a[i]
      u = bt[i]
      ns = int(rng.integers(1 << 30))
      base = one_step(env, xy, u, ap_, ns)
      vs = list(dirs)
      rv = rng.normal(size=(args.dirs, 2))
      rv /= np.linalg.norm(rv, axis=1, keepdims=True)
      vs += [v for v in rv]
      for eps in EPS:
        for v in vs:
          an = np.clip(ap_ + eps * v, -1.0, 1.0).astype(np.float32)
          da = float(np.linalg.norm(an - ap_))
          if da < 1e-9:
            continue
          alt = one_step(env, xy, u, an, ns)
          dxy = float(np.linalg.norm(alt[:2] - base[:2]))
          dz = float(abs(alt[2] - base[2]) * Z_SCALE)
          ds = float(np.sqrt(dxy ** 2 + dz ** 2))
          rows.append((gname, eps, da, ds / da, dxy / da, dz,
                       base[2], alt[2], xy[0], xy[1], int(u.any())))
      if (c + 1) % 1000 == 0:
        print('    %d/%d anchors' % (c + 1, len(idx)), flush=True)

  G = np.array([r[0] for r in rows])
  E = np.array([r[1] for r in rows])
  DA = np.array([r[2] for r in rows])
  R = np.array([r[3] for r in rows])
  RXY = np.array([r[4] for r in rows])
  DZ = np.array([r[5] for r in rows])
  Z0 = np.array([r[6] for r in rows])
  Z1 = np.array([r[7] for r in rows])
  AX = np.array([r[8] for r in rows])
  AY = np.array([r[9] for r in rows])
  UA = np.array([r[10] for r in rows]).astype(bool)
  switch = (Z0 < 0) != (Z1 < 0)
  print('\n  pairs: %s total, %s event-switching (%.4f)'
        % (format(len(R), ','), format(int(switch.sum()), ','), switch.mean()))
  out['pairs'] = {'total': int(len(R)), 'event_switching': int(switch.sum()),
                  'event_switch_fraction': float(switch.mean())}

  # ------------------------------------------------------------------ 4
  print('\n4. OVERALL RATIOS')
  print(HDR)
  for nm, v in (('R  (scaled 3-D)', R), ('R_xy (XY only)', RXY)):
    print(row(nm, stats(v)))
  out['4_overall'] = {'R': stats(R), 'R_xy': stats(RXY),
                      'delta_z_tilde_nonzero_frac': float((DZ > 0).mean())}
  print('  |delta z~| > 0 in %.4f of pairs; when nonzero it is always %.3f'
        % ((DZ > 0).mean(), np.unique(np.round(DZ[DZ > 0], 6))[0]
           if (DZ > 0).any() else 0.0))

  # ------------------------------------------------------------------ 5
  print('\n5. EVENT-PRESERVING vs EVENT-SWITCHING  (the decisive split)')
  print(HDR)
  print(row('event-preserving', stats(R[~switch])))
  print(row('event-switching', stats(R[switch])))
  print(row('  ..of which xy-only (R_xy)', stats(RXY[switch])))
  out['5_events'] = {'preserving': stats(R[~switch]),
                     'switching': stats(R[switch]),
                     'switching_R_xy': stats(RXY[switch])}

  # ------------------------------------------------------------------ 7
  print('\n7. DEPENDENCE ON PERTURBATION SIZE')
  print('  %-8s%10s%10s%10s%10s%12s%14s%14s'
        % ('eps', 'n', 'median', 'p95', 'p99', 'max', 'switch_frac',
           'median|switch'))
  by_eps = {}
  for e in EPS:
    m = E == e
    sm = m & switch
    d = stats(R[m])
    by_eps['%g' % e] = {'n': int(m.sum()), 'median': d['median'],
                        'p95': d['p95'], 'p99': d['p99'], 'max': d['max'],
                        'switch_fraction': float(switch[m].mean()),
                        'median_R_switching':
                            float(np.median(R[sm])) if sm.any() else None,
                        'median_R_preserving': float(np.median(R[m & ~switch]))}
    print('  %-8.3f%10s%10.3f%10.3f%10.3f%12.3f%14.5f%14s'
          % (e, format(int(m.sum()), ','), d['median'], d['p95'], d['p99'],
             d['max'], switch[m].mean(),
             '%.2f' % np.median(R[sm]) if sm.any() else '-'))
  out['7_by_epsilon'] = by_eps
  print('  event-PRESERVING only, by eps:')
  print('  %-8s%10s%10s%10s%10s' % ('eps', 'n', 'median', 'p99', 'max'))
  for e in EPS:
    m = (E == e) & ~switch
    d = stats(R[m])
    print('  %-8.3f%10s%10.3f%10.3f%10.3f'
          % (e, format(d['n'], ','), d['median'], d['p99'], d['max']))
    by_eps['%g' % e]['preserving'] = d

  # ------------------------------------------------------------------ 6
  print('\n6. GEOMETRY AND HIDDEN-U STRATA  (U is audit-only)')
  marg = wall_margin(np.stack([AX, AY], 1))
  cellx = np.clip(np.floor(AX).astype(int), 0, 8)
  celly = np.clip(np.floor(AY).astype(int), 0, 4)
  in_sw = np.zeros(len(R), bool)
  for cx, cy in SWAMP_CELLS:
    in_sw |= (cellx == cx) & (celly == cy)
  d_sw = np.maximum.reduce([3.0 - AX, np.zeros(len(AX)), AX - 6.0])
  d_sw = np.hypot(d_sw, np.maximum.reduce([3.0 - AY, np.zeros(len(AY)),
                                           AY - 4.0]))
  strata = {
      'ordinary free space': (~in_sw) & (marg >= 0.25) & (d_sw >= 0.5),
      'near wall (margin<0.25)': marg < 0.25,
      'near swamp (<0.5, outside)': (~in_sw) & (d_sw < 0.5),
      'inside swamp corridor': in_sw,
      'hidden U clear': ~UA,
      'hidden U active': UA,
      'group=uniform': G == 'uniform',
      'group=at_risk': G == 'at_risk',
  }
  print(HDR)
  out['6_strata'] = {}
  for nm, m in strata.items():
    d = stats(R[m])
    out['6_strata'][nm] = d
    if d['n']:
      print(row(nm, d))
  print('  same strata, EVENT-PRESERVING pairs only:')
  print(HDR)
  out['6_strata_preserving'] = {}
  for nm, m in strata.items():
    d = stats(R[m & ~switch])
    out['6_strata_preserving'][nm] = d
    if d['n']:
      print(row(nm, d))

  # ------------------------------------------------------------------ 9
  print('\n9. MODEL-BASED CROSS-CHECK (secondary): sigma_max(d Delta_xy / d a)')
  if os.path.exists(MODELS):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    with open(MODELS, 'rb') as f:
      b = pickle.load(f)['xy_state_action']
    prm = jax.tree_util.tree_map(jnp.asarray, b['params'])
    mu, sd = jnp.asarray(b['mu'], jnp.float32), jnp.asarray(b['sd'],
                                                            jnp.float32)
    net = hk.without_apply_rng(hk.transform(
        lambda x: hk.nets.MLP([256, 256, 2], activation=jax.nn.relu)(x)))

    def f1(xy_, a_):
      return net.apply(prm, (jnp.concatenate([xy_, a_]) - mu) / sd)
    jb = jax.jit(jax.vmap(jax.jacrev(f1, argnums=1)))
    idx = np.concatenate([groups['uniform'], groups['at_risk']])
    J = np.asarray(jb(jnp.asarray(s[idx, :2], jnp.float32),
                      jnp.asarray(a[idx], jnp.float32)))
    sm = np.linalg.svd(J, compute_uv=False)[:, 0]
    print(HDR)
    print(row('model sigma_max(dxy/da)', stats(sm)))
    print(row('simulator R_xy (preserving)', stats(RXY[~switch])))
    out['9_model_crosscheck'] = {
        'model_sigma_max': stats(sm),
        'simulator_R_xy_preserving': stats(RXY[~switch]),
        'note': 'the fitted XY model reproduces the simulator XY gain; it says '
                'nothing about the sink discontinuity, which is not in it'}
  else:
    print('  models.pkl not found -- skipped')
    out['9_model_crosscheck'] = None

  # ------------------------------------------------------------------ 8/10
  Lpres = stats(R[~switch])
  print('\n' + '=' * 110)
  print('8/10. COMPARISON WITH L=1.25 AND DECISION')
  print('=' * 110)
  print('  historical 2-D XY Jacobian: mean 1.03, p99 1.21, nominal L = 1.25')
  print('  event-PRESERVING scaled 3-D R: mean %.3f  p99 %.3f  p99.5 %.3f  '
        'max %.3f' % (Lpres['mean'], Lpres['p99'], Lpres['p99.5'],
                      Lpres['max']))
  sw = stats(R[switch])
  smalleps = R[switch & (E == 0.01)]
  print('  event-SWITCHING R: median %.2f  p99 %.2f  max %.2f'
        % (sw['median'] if sw['n'] else float('nan'),
           sw['p99'] if sw['n'] else float('nan'),
           sw['max'] if sw['n'] else float('nan')))
  if smalleps.size:
    print('  event-SWITCHING at eps=0.01 alone: median %.2f  max %.2f'
          % (np.median(smalleps), smalleps.max()))
  print('  |delta z~| at a switch is fixed at 0.24, so R >= 0.24/||da||, which')
  print('  is %.1f at eps=0.01 and %.1f at eps=0.30 -- unbounded as eps -> 0.'
        % (0.24 / 0.01, 0.24 / 0.30))
  case_b = bool(sw['n'] and sw['max'] > 5 * Lpres['p99'])
  out['8_10_decision'] = {
      'historical_L': 1.25,
      'event_preserving': Lpres, 'event_switching': sw,
      'event_switching_eps_0.01': stats(smalleps) if smalleps.size else None,
      'case': 'B' if case_b else 'A',
      'recommended_nominal_L_event_preserving': float(
          np.ceil(Lpres['p99'] * 20) / 20) if Lpres['n'] else None,
      'recommended_conservative_L_event_preserving': float(
          np.ceil(Lpres['max'] * 20) / 20) if Lpres['n'] else None,
      'global_realized_state_L_defensible': not case_b,
  }
  os.makedirs(args.out_dir, exist_ok=True)
  p = os.path.join(args.out_dir, 'lipschitz_audit.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
