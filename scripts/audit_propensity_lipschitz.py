r"""Can a smoothness assumption on p(sink | s, do(a)) rescue the SAFE action?

Diagnostic only. No training, no CRL change, no dynamics change. Nothing here
writes to any existing z_v0 / z_v1 / encoding / Manski artifact.

THE QUESTION THIS DECIDES
-------------------------
scripts/audit_propensity_manski.py found the propensity-weighted Manski bound
to be valid everywhere (84/84 intervals cover the truth) and genuinely
informative on the FREQUENT action, but useless on the rare one, because the
interval width is 1 - pi_B and the safe action is the rare one:

    action   pi_B (h=0.30)   Manski upper on P(sink)   true P(sink)
    enter    0.7169          0.2925                    0.1050
    wait     0.0839          0.9161                    0.0000
    back     0.0026          0.9974                    0.0000

A rule that minimises the upper bound therefore picks ENTER, the only
dangerous option. Data alone cannot certify that backing off is safe.

The only way to say anything about an action the demonstrator almost never
took, without new data, is to BORROW from actions it did take. That borrowing
is exactly a smoothness assumption on the failure propensity:

    | p(s,a) - p(s,a') |  <=  L_p || a - a' ||        (*)

This script measures L_p and decides whether (*) is strong enough to be worth
assuming here.

THE PASS MARK IS ARITHMETIC, NOT TASTE. p is a probability, so |p - p'| <= 1
always holds. A Lipschitz bound is therefore worse than the trivial [0,1]
whenever L_p ||Delta a|| >= 1. Two thresholds are reported:

  naive     L_p * d < 1                       d = ||a_target - a_enter||
  chained   (hi_M - lo_M) + 2 L_p d < w_M     the honest one: the learner does
                                              not know p(enter) either, only
                                              its Manski interval, so the
                                              extrapolated interval inherits
                                              that width and adds 2 L_p d. It
                                              must beat the Manski width w_M
                                              at the target action.

With d = 2.0 (enter to back), the enter-side Manski width 0.283 and the
back-side width 0.997, the chained threshold is L_p < 0.179.

WHY THE ANCHOR SET MATTERS. Averaged over anchors the encoding audit already
implies a tiny constant: P(CRN switch)/eps ~ 0.021 and |Delta p| <= P(switch),
so the AVERAGE L_p is ~0.02 and would pass easily. That average is not what (*)
needs. (*) must hold along the whole segment from a_enter to a_target at a
FIXED state, so what matters is the largest local slope encountered on that
path. Section 2 therefore builds a third anchor group on purpose: states whose
noiseless one-step endpoint lands within 0.02 of a swamp-cell boundary, where
dp/da is maximal. A prediction to falsify: with action noise sigma = 0.01 the
endpoint density at the boundary is ~0.4/sigma = 40, so L_p ~ 40 there, which
would fail the threshold by two orders of magnitude.

VARIANCE. p is estimated by Monte Carlo, and at small eps the signal L_p*eps
can be far below the sampling noise of two independent estimates. Both legs of
every pair therefore share (U, xi) -- common random numbers -- so the estimator
is PAIRED: Delta p is the mean of a per-draw difference that is 0 unless that
draw switches, with variance P(switch)/M instead of 2p(1-p)/M.

WHAT THE SIMULATOR BUYS US. Section 1 sweeps the true p(s, .) along the whole
enter -> wait -> back segment. The learner can never see this; it is reported
so the Lipschitz bound can be compared against the actual surface it claims to
approximate, not just against a constant.

Usage:
  python scripts/audit_propensity_lipschitz.py --draws 400 --anchors 30   # smoke
  python scripts/audit_propensity_lipschitz.py                            # full
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

ENV = 'point_two_route_swamp_windy_z_v1'
DATASET = 'datasets/swamp_windy_z_v1_merged_s0.npz'
MANSKI = 'artifacts/propensity_manski/propensity_manski.json'
OUT_DIR = 'artifacts/propensity_lipschitz'
TRACKED = ['crl/envs.py', 'scripts/audit_propensity_lipschitz.py']

QUERIES = (
    ('holding_mid', (2.50, 3.50)),
    ('holding_edge', (2.90, 3.50)),
    ('mouth', (3.10, 3.50)),
    ('corridor_1', (4.10, 3.50)),
)
A_ENTER = np.array([1.0, 0.0])
A_WAIT = np.array([0.0, 0.0])
A_BACK = np.array([-1.0, 0.0])
PATH_N = 41                              # points on the enter -> back segment
EPS = (0.005, 0.01, 0.025, 0.05, 0.10)
# cell boundaries that can flip the outcome. x = 4, 5 count too: the three
# swamp cells carry INDEPENDENT bits, so crossing an internal edge changes
# which bit applies and can flip the event just like the outer rim.
BOUND_X = (3.0, 4.0, 5.0, 6.0)
BOUND_Y = (3.0, 4.0)


def step_state(env, xy, bits, action, rng):
  """One step from a clean alive state. Returns (endpoint_xy, dead)."""
  env.set_auto_resample(False)
  env._dead = False
  env._z = 0.0
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = rng
  env.step(np.asarray(action, np.float32))
  return env.state.copy(), bool(env._dead)


class Zero:
  """An RNG stand-in whose normal() is exactly 0: the noiseless endpoint."""

  def normal(self, *a, **k):
    n = k.get('size', a[2] if len(a) > 2 else 2)
    return np.zeros(n)

  def random(self, *a, **k):
    return 0.0


def boundary_margin(xy_end):
  """Distance from the endpoint to the nearest outcome-flipping cell edge."""
  dx = min(abs(xy_end[0] - b) for b in BOUND_X)
  dy = min(abs(xy_end[1] - b) for b in BOUND_Y)
  return min(dx, dy)


def p_hat(env, xy, action, m, seeds, bits_all):
  """MC estimate of p(sink | s, do(action)) on a FIXED common draw set."""
  n = 0
  for i in range(m):
    _, dead = step_state(env, xy, bits_all[i], action,
                         np.random.default_rng(int(seeds[i])))
    n += dead
  return n / float(m)


def base_deaths(env, xy, a0, m, seeds, bits_all):
  """The base leg's per-draw outcome, computed ONCE per anchor and reused for
  every (direction, eps) pair -- the CRN pairing is unaffected and it removes
  half the simulation cost."""
  return np.array([step_state(env, xy, bits_all[i], a0,
                              np.random.default_rng(int(seeds[i])))[1]
                   for i in range(m)], bool)


def paired_dp(env, xy, d0, a1, m, seeds, bits_all):
  """Paired estimate of p(a1) - p(a0) under COMMON (U, xi), given the base
  leg's outcomes d0. The per-draw difference is 0 except on switching draws,
  so the variance is P(switch)/m rather than 2p(1-p)/m."""
  s = 0
  nsw = 0
  for i in range(m):
    _, d1 = step_state(env, xy, bits_all[i], a1,
                       np.random.default_rng(int(seeds[i])))
    s += int(d1) - int(d0[i])
    nsw += int(bool(d0[i]) != d1)
  return s / float(m), nsw


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--draws', type=int, default=3000,
                  help='MC draws per probability / per paired difference')
  ap.add_argument('--draws2', type=int, default=1000,
                  help='paired MC draws in section 2. The paired estimator has '
                       'variance P(switch)/m rather than 2p(1-p)/m, so it '
                       'needs far fewer draws than section 1.')
  ap.add_argument('--anchors', type=int, default=60,
                  help='anchors per group in section 2')
  ap.add_argument('--dirs', type=int, default=2)
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
  out = {'analysis_script': 'scripts/audit_propensity_lipschitz.py',
         'env': ENV, 'dataset': args.dataset,
         'dataset_content_sha256': csha(args.dataset),
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'epsilons': list(EPS),
         'config': {'draws': args.draws, 'anchors': args.anchors,
                    'draws2': args.draws2,
                    'dirs': args.dirs, 'seed': args.seed,
                    'path_points': PATH_N}}

  print('=' * 104)
  print('IS A SMOOTHNESS ASSUMPTION ON p(sink | s, do(a)) STRONG ENOUGH HERE?')
  print('=' * 104)
  print('  dataset sha %s' % out['dataset_content_sha256'])
  print('  code commit %s  dirty %s' % (out['code_commit'][:12], out['dirty']))

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)
  active_prob = float(env.active_prob)
  sigma = float(env._action_noise)
  print('  active_prob %.3f   action_noise sigma %.4f' % (active_prob, sigma))
  print('  predicted boundary slope ~ 0.4/sigma = %.1f' % (0.4 / sigma))
  out['active_prob'] = active_prob
  out['action_noise'] = sigma

  rng = np.random.default_rng(args.seed)
  M = args.draws
  # one common draw set, reused for every probability in section 1 so that
  # differences along the path are paired as well
  seeds = rng.integers(1 << 30, size=M)
  bits_all = rng.random((M, 3)) < active_prob

  # load the Manski widths so the verdict is computed against real numbers
  man = {}
  if os.path.exists(MANSKI):
    with open(MANSKI) as f:
      mj = json.load(f)
    for k, v in mj.get('panel_b_decision_point', {}).items():
      man[k] = v
    print('  loaded Manski widths from %s' % MANSKI)
  else:
    print('  WARNING: %s missing; the verdict table will be omitted' % MANSKI)

  # ------------------------------------------------------------------- 1
  print('\n' + '=' * 104)
  print('1. THE TRUE PROFILE p(s, a) ALONG  enter -> wait -> back')
  print('=' * 104)
  print('  a(t) = (1-t) a_enter + t a_back,  t in [0,1];  t=0.5 is a_wait.')
  print('  ||a_enter - a_back|| = 2.0, so the step between grid points is')
  print('  %.4f in action units. Each p is %d draws on the SAME (U, xi) set.'
        % (2.0 / (PATH_N - 1), M))
  ts = np.linspace(0.0, 1.0, PATH_N)
  seg = {}
  t0 = time.time()
  for qname, xy in QUERIES:
    ps = []
    for t in ts:
      a = (1 - t) * A_ENTER + t * A_BACK
      ps.append(p_hat(env, xy, a, M, seeds, bits_all))
    ps = np.array(ps)
    dt_a = 2.0 / (PATH_N - 1)
    slope = np.abs(np.diff(ps)) / dt_a
    seg[qname] = {'xy': list(xy), 't': ts.tolist(), 'p': ps.tolist(),
                  'max_local_slope': float(slope.max()),
                  'mean_local_slope': float(slope.mean()),
                  'p_enter': float(ps[0]), 'p_wait': float(ps[PATH_N // 2]),
                  'p_back': float(ps[-1])}
    print('\n  %s  (%.2f, %.2f)   [%.0fs]' % (qname, xy[0], xy[1],
                                              time.time() - t0))
    print('    p(enter) %.4f   p(wait) %.4f   p(back) %.4f'
          % (ps[0], ps[PATH_N // 2], ps[-1]))
    print('    grid-resolution slope: max %.3f  mean %.3f  (per unit action)'
          % (slope.max(), slope.mean()))
    row = '    profile:'
    for i in range(0, PATH_N, 4):
      row += ' %.3f' % ps[i]
    print(row)
  out['1_true_profile'] = seg

  # ------------------------------------------------------------------- 2
  print('\n' + '=' * 104)
  print('2. LOCAL L_p BY EPSILON, WITH A DELIBERATE BOUNDARY GROUP')
  print('=' * 104)
  print('  Paired (common U and xi) so Delta p is a per-draw difference.')
  with np.load(args.dataset, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
  s = obs[:, :-1, :3].reshape(-1, 3)
  a = act[:, :-1, :].reshape(-1, 2)
  alive = s[:, 2] == 0.0
  sa, aa = s[alive][:, :2], a[alive]
  near = ((sa[:, 0] >= 2.0) & (sa[:, 0] < 6.0)
          & (sa[:, 1] >= 3.0) & (sa[:, 1] < 4.0))
  pool_u = np.where(np.ones(len(sa), bool))[0]
  pool_r = np.where(near)[0]

  # the boundary group: noiseless endpoint within 0.02 of a flipping edge
  print('  scanning for boundary anchors (noiseless endpoint within 0.02 of a')
  print('  cell edge; x = 4 and 5 count -- the three cells carry independent')
  print('  bits, so an internal edge flips the outcome too) ...')
  cand = rng.choice(pool_r, min(60000, len(pool_r)), False)
  bnd = []
  zero = Zero()
  for i in cand:
    end, _ = step_state(env, sa[i], [0, 0, 0], aa[i], zero)
    if boundary_margin(end) < 0.02:
      bnd.append(i)
      if len(bnd) >= args.anchors:
        break
  bnd = np.array(bnd, int)
  print('  found %d boundary anchors (scanned %d)' % (len(bnd), len(cand)))

  groups = {'uniform': rng.choice(pool_u, args.anchors, False),
            'at_risk': rng.choice(pool_r, args.anchors, False),
            'boundary': bnd}
  out['2_boundary_anchors_found'] = int(len(bnd))

  M2 = args.draws2
  seeds2, bits2 = seeds[:M2], bits_all[:M2]
  sec2 = {}
  print('  paired draws per difference: %d  (L_p quantum %.4f at eps=%g)'
        % (M2, 1.0 / M2 / EPS[0], EPS[0]))
  print('\n  %-10s%9s%9s%10s%10s%10s%10s%10s'
        % ('group', 'eps', 'n', 'mean Lp', 'p50 Lp', 'p90 Lp', 'p99 Lp',
           'max Lp'))
  for gname, idx in groups.items():
    if len(idx) == 0:
      continue
    acc = {eps: [] for eps in EPS}
    for i in idx:
      # the base leg is the SAME for every (direction, eps) at this anchor, so
      # compute it once: the CRN pairing is unaffected and half the simulation
      # cost disappears.
      d0 = base_deaths(env, sa[i], aa[i], M2, seeds2, bits2)
      for _ in range(args.dirs):
        v = rng.normal(size=2)
        v /= np.linalg.norm(v)
        for eps in EPS:
          a1 = np.clip(aa[i] + eps * v, -1.0, 1.0)
          da = float(np.linalg.norm(a1 - aa[i]))
          if da < 1e-9:
            continue
          dp, _ = paired_dp(env, sa[i], d0, a1, M2, seeds2, bits2)
          acc[eps].append(abs(dp) / da)
    for eps in EPS:
      lps = np.array(acc[eps])
      if lps.size == 0:
        continue
      sec2['%s|eps%g' % (gname, eps)] = {
          'group': gname, 'eps': eps, 'n': int(len(lps)),
          'mean': float(lps.mean()), 'median': float(np.median(lps)),
          'p90': float(np.percentile(lps, 90)),
          'p99': float(np.percentile(lps, 99)),
          'max': float(lps.max())}
      print('  %-10s%9.3f%9d%10.4f%10.4f%10.4f%10.4f%10.4f'
            % (gname, eps, len(lps), lps.mean(), np.median(lps),
               np.percentile(lps, 90), np.percentile(lps, 99), lps.max()))
    print()
  out['2_local_Lp'] = sec2

  # ------------------------------------------------------------------- 3
  print('=' * 104)
  print('3. VERDICT: DOES THE SMOOTHNESS BOUND BEAT MANSKI ON THE SAFE SIDE?')
  print('=' * 104)
  print('  L_p candidates: the max grid slope on the actual enter->back path')
  print('  (section 1) and the boundary-group p99 at the smallest eps')
  print('  (section 2). d = ||a_target - a_enter||.')
  print('  naive   : L_p * d < 1          (beat the trivial [0,1])')
  print('  chained : w_enter + 2 L_p d < w_target   (the honest test: the')
  print('            learner only knows p(enter) up to its own Manski width)')
  cands = {}
  cands['path_max_slope'] = max(v['max_local_slope']
                                for v in seg.values())
  # report the boundary group at BOTH the smallest eps (sharpest local slope,
  # but coarsely quantised at 1/(M2*eps)) and at eps = 0.05 (well resolved),
  # so the verdict cannot hinge on quantisation alone.
  for e in (EPS[0], 0.05):
    bkey = 'boundary|eps%g' % e
    if bkey in sec2:
      cands['boundary_p99_eps%g' % e] = sec2[bkey]['p99']
      cands['boundary_max_eps%g' % e] = sec2[bkey]['max']
  akey = 'at_risk|eps%g' % EPS[0]
  if akey in sec2:
    cands['at_risk_p99_eps%g' % EPS[0]] = sec2[akey]['p99']
  verdict = {}
  H = 0.30                                     # the Manski operating point
  for qname, _ in QUERIES:
    ke = '%s|enter|h%g' % (qname, H)
    if ke not in man:
      continue
    w_enter = man[ke]['width']
    for tgt, a_t in (('wait', A_WAIT), ('back', A_BACK)):
      kt = '%s|%s|h%g' % (qname, tgt, H)
      if kt not in man:
        continue
      w_tgt = man[kt]['width']
      d = float(np.linalg.norm(a_t - A_ENTER))
      print('\n  %s -> %s   d = %.2f   Manski width: enter %.4f, target %.4f'
            % (qname, tgt, d, w_enter, w_tgt))
      print('    %-28s%12s%14s%10s%10s'
            % ('L_p source', 'L_p', 'chained wid', 'naive', 'chained'))
      for cn, lp in cands.items():
        chained = w_enter + 2.0 * lp * d
        print('    %-28s%12.4f%14.4f%10s%10s'
              % (cn, lp, chained,
                 'PASS' if lp * d < 1.0 else 'FAIL',
                 'PASS' if chained < w_tgt else 'FAIL'))
        verdict['%s|%s|%s' % (qname, tgt, cn)] = {
            'query': qname, 'target': tgt, 'd': d, 'L_p_source': cn,
            'L_p': lp, 'w_enter': w_enter, 'w_target': w_tgt,
            'chained_width': chained, 'naive_pass': bool(lp * d < 1.0),
            'chained_pass': bool(chained < w_tgt)}
      # the threshold, stated as a number
      print('    thresholds: naive needs L_p < %.4f ; chained needs L_p < %.4f'
            % (1.0 / d, max(0.0, (w_tgt - w_enter) / (2.0 * d))))
      verdict['%s|%s|thresholds' % (qname, tgt)] = {
          'naive_threshold': 1.0 / d,
          'chained_threshold': max(0.0, (w_tgt - w_enter) / (2.0 * d))}
  out['3_verdict'] = verdict
  out['3_Lp_candidates'] = cands

  p = os.path.join(args.out_dir, 'propensity_lipschitz.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
