"""Semantic audit of the existing off-diagonal V0 candidates. READ-ONLY.

Nothing is retrained here and the objective is not touched. The V0 run
(scripts/diag_offdiag_v0.py) pickles z_phi at the nominal L together with the
exact evaluation design -- the same off-diagonal actions and the same fixed
negative pool -- so this script rebuilds g_diag and g_cf bit-for-bit and then
only measures them. It re-derives mean Delta_B and refuses to continue unless
it matches the committed V0 number, which is what makes "same candidates"
a checked claim rather than an assumption.

THE QUESTION. V0 showed Delta_B = B(g_cf) - B(g_diag) > 0 for ~91% of samples.
B is a soft nearest-negative score over q_alpha = (1-alpha) q_batch + alpha
q_fail, and q_batch negatives are ordinary relabeled future states -- they are
not physically bad. So a rise in B could mean the candidate moved toward
genuine failure states, or merely toward some arbitrary other reachable goal.
These audits separate those readings.

GEOMETRY IS REUSED, NOT REIMPLEMENTED. A live TwoRouteSwampWindyEnv instance is
built and its own helpers are called per point:

    env._is_blocked(p)          bounds + wall test (crl/envs.py)
    env._discretize_state(p)    floor + clip to the walls array
    env._in_swamp_corridor(p)   membership of env.SWAMP_CELLS
    env._low / env._high / env._walls

The wall-crossing test replays the env's OWN integration: 10 substeps of
dt=0.1, one axis at a time, each rejected if _is_blocked -- exactly
TwoRouteSwampWindyEnv.step. If replaying the displacement does not land on
s'_cf, the straight move is not geometrically realizable. That is ~600k
_is_blocked calls and takes a little while; the alternative would be an
approximate reimplementation, which is what the audit is supposed to avoid.

swamp_bits is NOT used here at all -- not as input, not as target, not even for
stratification; the swamp is identified purely from the static cell geometry.
Failure-bank distance is measured for AUDIT ONLY and is never a training loss.

Usage:
  python scripts/audit_offdiag_v0_semantics.py
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402

from diag_transition_mlp import build_transitions, make_mlp  # noqa: E402
from diag_action_lipschitz import wall_margin     # noqa: E402
from diag_offdiag_v0 import make_z_net, soft_neg_score, dist, EPS  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl import envs as envs_mod                  # noqa: E402

Z_PKL = 'artifacts/transition_offdiag_v0/offdiag_v0_z.pkl'
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
ENV = 'point_two_route_swamp_windy_v0'
OUT = 'artifacts/transition_offdiag_v0/semantic_audit.json'
V0_JSON = 'artifacts/transition_offdiag_v0/offdiag_v0.json'
CATS = ['ordinary', 'near_wall', 'near_swamp', 'swamp', 'inside_wall',
        'outside_bounds']


def classify(env, pts, swamp_rect=(3.0, 6.0, 3.0, 4.0), near_sw=0.5,
             near_wl=0.25):
  """Mutually exclusive spatial category per point, priority order as listed.

  Priority matters: a point inside a wall is not also 'near_wall', and an
  out-of-bounds point is not classified by a wall lookup that would have to
  clip its cell index first.
  """
  x0, x1, y0, y1 = swamp_rect
  out = np.empty(len(pts), dtype='<U14')
  marg = wall_margin(pts)
  for k, p in enumerate(pts):
    if np.any(p < env._low) or np.any(p > env._high):
      out[k] = 'outside_bounds'
      continue
    i, j = env._discretize_state(p)
    if env._walls[i, j] == 1:
      out[k] = 'inside_wall'
      continue
    if env._in_swamp_corridor(p):
      out[k] = 'swamp'
      continue
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    if np.hypot(dx, dy) < near_sw:
      out[k] = 'near_swamp'
      continue
    out[k] = 'near_wall' if marg[k] < near_wl else 'ordinary'
  return out


def replay(env, s, delta, n_sub=10):
  """Replay TwoRouteSwampWindyEnv.step's per-axis substep integration."""
  st = np.array(s, dtype=float)
  dt = 1.0 / n_sub
  for _ in range(n_sub):
    for axis in range(2):
      new = st.copy()
      new[axis] += dt * delta[axis]
      if not env._is_blocked(new):
        st = new
  return st


def frac(mask):
  return float(np.mean(mask))


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--z', default=Z_PKL)
  ap.add_argument('--bank', default=BANK)
  ap.add_argument('--out', default=OUT)
  ap.add_argument('--far-support', type=float, default=2.0,
                  help='"substantially farther" = d_data(cf) > this * '
                       'd_data(diag) and an absolute gain over --far-abs')
  ap.add_argument('--far-abs', type=float, default=0.05)
  args = ap.parse_args()

  try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    commit = '(unavailable)'

  with open(args.z, 'rb') as f:
    Z = pickle.load(f)
  L, tau, alpha = Z['L'], Z['tau'], Z['alpha']
  with open(Z['diag_params_path'], 'rb') as f:
    b = pickle.load(f)['state_action']
  d_params = jax.tree_util.tree_map(jnp.asarray, b['params'])
  d_mu, d_sd = jnp.asarray(b['mu'], jnp.float32), jnp.asarray(b['sd'],
                                                              jnp.float32)
  z_params = jax.tree_util.tree_map(jnp.asarray, Z['z_params'])
  z_mu = jnp.asarray(Z['z_mu'], jnp.float32)
  z_sd = jnp.asarray(Z['z_sd'], jnp.float32)
  diag_net, z_net = make_mlp(), make_z_net()

  s = Z['s_te']; a_obs = Z['a_te']; a = Z['a_off_te']; dan = Z['da_norm_te']
  negs = Z['neg_eval']; from_bank = Z['neg_from_bank'].astype(bool)

  print('=' * 96)
  print('SEMANTIC AUDIT OF OFF-DIAGONAL V0 CANDIDATES  (read-only, no '
        'retraining)')
  print('=' * 96)
  print('  commit         : %s' % commit)
  print('  z_phi + design : %s' % args.z)
  print('  nominal        : L = %.2f   alpha = %.2f   tau = %.2f'
        % (L, alpha, tau))
  print('  samples        : %s held-out PRIMARY off-diagonal test transitions'
        % format(len(s), ','))
  print('  negative pool  : %d goals, %d from the failure bank'
        % (len(negs), int(from_bank.sum())))

  # ---- rebuild the exact V0 candidates -----------------------------------
  x = (jnp.concatenate([jnp.asarray(s), jnp.asarray(a_obs)], -1) - d_mu) / d_sd
  dd = np.asarray(diag_net.apply(d_params, x))
  zx = (jnp.concatenate([jnp.asarray(s), jnp.asarray(a),
                         jnp.asarray(a_obs)], -1) - z_mu) / z_sd
  z = np.asarray(z_net.apply(z_params, zx))
  R = (L * np.linalg.norm(a - a_obs, axis=1, keepdims=True))
  zn = np.sqrt((z * z).sum(1, keepdims=True) + 1e-24)
  dphi = z * np.minimum(1.0, R / (zn + EPS))
  dcf = dd + dphi
  g_diag, g_cf = s + dd, s + dcf
  ndphi = np.linalg.norm(dphi, axis=1)
  rho = ndphi / np.maximum(L * dan, 1e-12)

  jn = jnp.asarray(negs)
  B_diag = np.asarray(soft_neg_score(jnp.asarray(g_diag), jn, tau))
  B_cf = np.asarray(soft_neg_score(jnp.asarray(g_cf), jn, tau))
  dB = B_cf - B_diag
  with open(V0_JSON) as f:
    v0 = json.load(f)['results']['L=%.2f' % L]['B_score']
  print('\n  CONSISTENCY: mean dB recomputed %+.5f vs V0 %+.5f  |diff| %.2e'
        % (dB.mean(), v0['mean_change'], abs(dB.mean() - v0['mean_change'])))
  assert abs(dB.mean() - v0['mean_change']) < 1e-6, (
      'rebuilt candidates do not match the V0 run')
  print('  frac improved recomputed %.4f vs V0 %.4f'
        % ((dB > 0).mean(), v0['frac_improved']))

  out = {'commit': commit,
         'analysis_script': 'scripts/audit_offdiag_v0_semantics.py',
         'inputs': {'z_pickle': args.z, 'v0_json': V0_JSON, 'bank': args.bank},
         'nominal': {'L': L, 'alpha': alpha, 'tau': tau},
         'n_samples': int(len(s)),
         'reused_geometry': [
             'crl.envs.TwoRouteSwampWindyEnv._is_blocked',
             'crl.envs.TwoRouteSwampEnv._discretize_state',
             'crl.envs.TwoRouteSwampEnv._in_swamp_corridor',
             'crl.envs.TwoRouteSwampEnv._low/_high/_walls/SWAMP_CELLS',
             'substep replay of TwoRouteSwampWindyEnv.step (10 x dt=0.1, '
             'per-axis, _is_blocked rejection)'],
         'consistency': {'mean_dB_recomputed': float(dB.mean()),
                         'mean_dB_v0': v0['mean_change']}}

  # ---- 1. failure-bank proximity -----------------------------------------
  with np.load(args.bank, allow_pickle=True) as bk:
    F = np.asarray(bk['goals'], np.float64)
  tree_F = cKDTree(F)
  d_fail_diag, _ = tree_F.query(g_diag)
  d_fail_cf, _ = tree_F.query(g_cf)
  dd_fail = d_fail_cf - d_fail_diag
  pear = float(np.corrcoef(dB, -dd_fail)[0, 1])
  spear = float(spearmanr(dB, -dd_fail).statistic)
  print('\n' + '=' * 96)
  print('1. FAILURE-BANK PROXIMITY (audit only; never a training loss)')
  print('=' * 96)
  print('  d_fail(diag) mean %.4f  median %.4f' % (d_fail_diag.mean(),
                                                   np.median(d_fail_diag)))
  print('  d_fail(cf)   mean %.4f  median %.4f' % (d_fail_cf.mean(),
                                                   np.median(d_fail_cf)))
  print('  change       mean %+.5f  median %+.5f' % (dd_fail.mean(),
                                                     np.median(dd_fail)))
  print('  fraction moving CLOSER to the failure bank : %.4f' % frac(dd_fail < 0))
  print('  corr(dB, -d_fail change)  Pearson %+.4f   Spearman %+.4f'
        % (pear, spear))
  out['1_failure_bank'] = {
      'd_fail_diag': dist(d_fail_diag), 'd_fail_cf': dist(d_fail_cf),
      'change_mean': float(dd_fail.mean()),
      'change_median': float(np.median(dd_fail)),
      'frac_closer': frac(dd_fail < 0),
      'corr_pearson_dB_vs_neg_dfail': pear,
      'corr_spearman_dB_vs_neg_dfail': spear}

  # ---- geometry ----------------------------------------------------------
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=0)
  print('\n  classifying %s points with the env helpers...' % format(
      2 * len(s), ','))
  cat_diag = classify(env, g_diag)
  cat_cf = classify(env, g_cf)

  # ---- 2. spatial risk ---------------------------------------------------
  print('\n' + '=' * 96)
  print('2. SPATIAL RISK  (categories from static cell geometry; swamp_bits '
        'NOT used)')
  print('=' * 96)
  print('  %-16s%10s%10s' % ('category', 'diag', 'cf'))
  for c in CATS:
    print('  %-16s%10d%10d' % (c, int((cat_diag == c).sum()),
                               int((cat_cf == c).sum())))
  trans = {}
  for cd in CATS:
    for cc in CATS:
      n = int(((cat_diag == cd) & (cat_cf == cc)).sum())
      if n:
        trans['%s -> %s' % (cd, cc)] = n
  print('\n  transitions (only non-zero, sorted):')
  for k, v in sorted(trans.items(), key=lambda kv: -kv[1])[:14]:
    tag = '' if k.split(' -> ')[0] == k.split(' -> ')[1] else '   <-- changed'
    print('    %-38s %7d%s' % (k, v, tag))
  risky = np.isin(cat_cf, ['swamp', 'near_swamp'])
  risky0 = np.isin(cat_diag, ['swamp', 'near_swamp'])
  to_risky = (~risky0) & risky
  valid0 = ~np.isin(cat_diag, ['inside_wall', 'outside_bounds'])
  validc = ~np.isin(cat_cf, ['inside_wall', 'outside_bounds'])
  print('\n  ordinary/near_wall -> swamp        : %d' % int(
      ((~risky0) & (cat_cf == 'swamp')).sum()))
  print('  ordinary/near_wall -> near_swamp   : %d' % int(
      ((~risky0) & (cat_cf == 'near_swamp')).sum()))
  print('  valid -> invalid                   : %d' % int(
      (valid0 & ~validc).sum()))
  print('\n  does larger dB mean more movement toward risk? (dB quintiles)')
  qs = np.quantile(dB, np.linspace(0, 1, 6))
  print('  %-22s%10s%14s%14s' % ('dB quintile', 'n', 'P(->risky)',
                                 'mean d_fail cf'))
  q_rows = []
  for i in range(5):
    m = (dB >= qs[i]) & (dB <= qs[i + 1] if i == 4 else dB < qs[i + 1])
    q_rows.append({'q': i + 1, 'n': int(m.sum()),
                   'p_to_risky': frac(to_risky[m]),
                   'mean_d_fail_cf': float(d_fail_cf[m].mean())})
    print('  Q%-21d%10d%14.4f%14.4f' % (i + 1, m.sum(), frac(to_risky[m]),
                                        d_fail_cf[m].mean()))
  out['2_spatial'] = {
      'counts_diag': {c: int((cat_diag == c).sum()) for c in CATS},
      'counts_cf': {c: int((cat_cf == c).sum()) for c in CATS},
      'transitions': trans,
      'ordinary_to_swamp': int(((~risky0) & (cat_cf == 'swamp')).sum()),
      'ordinary_to_near_swamp': int(((~risky0) & (cat_cf == 'near_swamp')).sum()),
      'valid_to_invalid': int((valid0 & ~validc).sum()),
      'frac_moving_to_risky': frac(to_risky),
      'dB_quintiles': q_rows}

  # ---- 3. physical plausibility ------------------------------------------
  print('\n' + '=' * 96)
  print('3. PHYSICAL PLAUSIBILITY (env collision geometry, reused directly)')
  print('=' * 96)
  oob = np.array([bool(np.any(p < env._low) or np.any(p > env._high))
                  for p in g_cf])
  inw = (cat_cf == 'inside_wall')
  print('  replaying %s displacements through the env substep loop...'
        % format(len(s), ','))
  crosses = np.zeros(len(s), bool)
  for k in range(len(s)):
    land = replay(env, s[k], dcf[k])
    crosses[k] = np.linalg.norm(land - g_cf[k]) > 1e-6
  over_reach = (np.abs(dcf) > 1.0).any(axis=1)
  ok = (~oob) & (~inw) & (~crosses)
  print('  fraction inside a wall                  : %.5f  (%d)'
        % (frac(inw), inw.sum()))
  print('  fraction outside valid bounds           : %.5f  (%d)'
        % (frac(oob), oob.sum()))
  print('  fraction crossing a wall from s to s_cf : %.5f  (%d)'
        % (frac(crosses), crosses.sum()))
  print('  fraction VALID under env geometry       : %.5f  (%d)'
        % (frac(ok), ok.sum()))
  print('  (supplementary, a DYNAMICS not geometry check)')
  print('  fraction whose |Delta_cf| exceeds a one-step reach (>1 per axis): '
        '%.5f  (%d)' % (frac(over_reach), over_reach.sum()))
  # same for the diagonal prediction, as the reference point
  crosses_d = np.zeros(len(s), bool)
  for k in range(len(s)):
    crosses_d[k] = np.linalg.norm(replay(env, s[k], dd[k]) - g_diag[k]) > 1e-6
  print('  reference: the DIAGONAL prediction crosses a wall %.5f  (%d)'
        % (frac(crosses_d), crosses_d.sum()))
  out['3_physical'] = {
      'frac_inside_wall': frac(inw), 'frac_outside_bounds': frac(oob),
      'frac_crosses_wall': frac(crosses), 'frac_valid': frac(ok),
      'frac_exceeds_one_step_reach': frac(over_reach),
      'frac_diag_crosses_wall': frac(crosses_d)}

  # ---- 4. observed-support plausibility ----------------------------------
  D = build_transitions(Z['dataset'])
  ref = (D['s'] + D['delta'])[D['primary']].astype(np.float64)
  tree_D = cKDTree(ref)
  dd_data, _ = tree_D.query(g_diag)
  dc_data, _ = tree_D.query(g_cf)
  far = (dc_data > args.far_support * dd_data) & \
        (dc_data - dd_data > args.far_abs)
  print('\n' + '=' * 96)
  print('4. OBSERVED-SUPPORT PLAUSIBILITY  (NN distance to %s observed PRIMARY '
        'next states)' % format(len(ref), ','))
  print('=' * 96)
  print('  NOT evidence of causal compatibility -- a plausibility diagnostic '
        'only.')
  print('  %-22s%10s%10s%10s%10s' % ('', 'mean', 'median', 'p90', 'p99'))
  for nm, v in (('d_data(s_diag)', dd_data), ('d_data(s_cf)', dc_data)):
    print('  %-22s%10.5f%10.5f%10.5f%10.5f'
          % (nm, v.mean(), np.median(v), np.percentile(v, 90),
             np.percentile(v, 99)))
  print('  fraction substantially farther (>%.1fx AND >+%.2f) : %.5f  (%d)'
        % (args.far_support, args.far_abs, frac(far), far.sum()))
  out['4_support'] = {
      'n_reference_points': int(len(ref)),
      'd_data_diag': dist(dd_data), 'd_data_cf': dist(dc_data),
      'criterion': 'd_cf > %.1f * d_diag and d_cf - d_diag > %.2f'
                   % (args.far_support, args.far_abs),
      'frac_substantially_farther': frac(far)}

  # ---- 5. ordinary vs failure negative influence -------------------------
  print('\n' + '=' * 96)
  print('5. WHICH NEGATIVE COMPONENT DRIVES THE GAIN?')
  print('=' * 96)
  print('  Same tau, same fixed pool, split by origin: %d q_batch draws and %d'
        % (int((~from_bank).sum()), int(from_bank.sum())))
  print('  bank draws. q_batch negatives are ordinary relabeled future states '
        'and\n  are NOT physically bad, which is exactly why this split '
        'matters.')
  comp = {}
  for nm, sel in (('batch', ~from_bank), ('fail', from_bank)):
    sub = jnp.asarray(negs[sel])
    bd = np.asarray(soft_neg_score(jnp.asarray(g_diag), sub, tau))
    bc = np.asarray(soft_neg_score(jnp.asarray(g_cf), sub, tau))
    dl = bc - bd
    comp[nm] = {'mean_diag': float(bd.mean()), 'mean_cf': float(bc.mean()),
                'mean_change': float(dl.mean()),
                'median_change': float(np.median(dl)),
                'frac_improved': frac(dl > 0), 'n_negatives': int(sel.sum())}
    print('  dB_%-6s mean %+.5f   median %+.5f   frac improved %.4f'
          % (nm, dl.mean(), np.median(dl), frac(dl > 0)))
  # supplementary: the FULL 256-entry bank, statistically steadier than 28
  bd = np.asarray(soft_neg_score(jnp.asarray(g_diag), jnp.asarray(F), tau))
  bc = np.asarray(soft_neg_score(jnp.asarray(g_cf), jnp.asarray(F), tau))
  dl_full = bc - bd
  comp['fail_full_bank'] = {'mean_change': float(dl_full.mean()),
                            'median_change': float(np.median(dl_full)),
                            'frac_improved': frac(dl_full > 0),
                            'n_negatives': int(len(F))}
  print('  dB_fail against the FULL %d-state bank (supplementary, steadier '
        'than %d):\n    mean %+.5f   median %+.5f   frac improved %.4f'
        % (len(F), int(from_bank.sum()), dl_full.mean(), np.median(dl_full),
           frac(dl_full > 0)))
  out['5_components'] = comp

  # ---- 6. budget usage ---------------------------------------------------
  print('\n' + '=' * 96)
  print('6. RELATIONSHIP WITH LIPSCHITZ-BUDGET USAGE rho')
  print('=' * 96)
  edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.999, 1.001])
  labels = ['[0.0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1.0)',
            'rho=1 (saturated)']
  print('  %-20s%9s%12s%12s%12s%12s' % ('rho bin', 'n', 'd_fail cf',
                                        'P(->risky)', 'P(invalid)', 'd_data cf'))
  bins = []
  for i in range(len(edges) - 1):
    m = (rho >= edges[i]) & (rho < edges[i + 1])
    if m.sum() == 0:
      continue
    r = {'bin': labels[i], 'n': int(m.sum()),
         'mean_d_fail_cf': float(d_fail_cf[m].mean()),
         'p_to_risky': frac(to_risky[m]),
         'p_invalid': frac((~ok)[m]),
         'mean_d_data_cf': float(dc_data[m].mean())}
    bins.append(r)
    print('  %-20s%9d%12.4f%12.4f%12.5f%12.5f'
          % (r['bin'], r['n'], r['mean_d_fail_cf'], r['p_to_risky'],
             r['p_invalid'], r['mean_d_data_cf']))
  cors = {}
  for nm, v in (('d_fail_cf', d_fail_cf), ('to_risky', to_risky.astype(float)),
                ('invalid', (~ok).astype(float)), ('d_data_cf', dc_data),
                ('dB', dB)):
    cors[nm] = {'pearson': float(np.corrcoef(rho, v)[0, 1]),
                'spearman': float(spearmanr(rho, v).statistic)}
  print('\n  corr(rho, .)      %-14s%-14s' % ('Pearson', 'Spearman'))
  for nm, c in cors.items():
    print('    %-14s%+14.4f%+14.4f' % (nm, c['pearson'], c['spearman']))
  out['6_budget'] = {'bins': bins, 'correlations': cors,
                     'rho': dist(rho)}

  # ---- 7. representative examples ----------------------------------------
  dB_hi = dB > np.quantile(dB, 0.9)
  picks = {
      'large_dB_and_closer_to_fail': dB_hi & (dd_fail < -0.05),
      'large_dB_but_not_closer_to_fail': dB_hi & (dd_fail > 0.0),
      'moves_into_swamp': (cat_diag != 'swamp') & (cat_cf == 'swamp'),
      'moves_away_from_swamp': (cat_diag == 'swamp') & (cat_cf != 'swamp'),
      'invalid_or_crosses_wall': (~ok),
      'plausible_and_near_full_budget': ok & (rho > 0.98) & (~crosses),
  }
  comp_b = comp['batch']; comp_f = comp['fail']
  sub_b = jnp.asarray(negs[~from_bank]); sub_f = jnp.asarray(negs[from_bank])
  dB_b = np.asarray(soft_neg_score(jnp.asarray(g_cf), sub_b, tau)) - \
      np.asarray(soft_neg_score(jnp.asarray(g_diag), sub_b, tau))
  dB_f = np.asarray(soft_neg_score(jnp.asarray(g_cf), sub_f, tau)) - \
      np.asarray(soft_neg_score(jnp.asarray(g_diag), sub_f, tau))
  erng = np.random.default_rng(0)
  examples = {}
  print('\n' + '=' * 96)
  print('7. REPRESENTATIVE EXAMPLES')
  print('=' * 96)
  for nm, m in picks.items():
    idx = np.where(m)[0]
    print('  %-34s available %7d' % (nm, idx.size))
    if idx.size == 0:
      examples[nm] = []
      continue
    sel = erng.choice(idx, min(3, idx.size), replace=False)
    examples[nm] = [{
        's': s[k].tolist(), 'a_obs': a_obs[k].tolist(),
        'a_intervention': a[k].tolist(), 'da_norm': float(dan[k]),
        's_next_diag': g_diag[k].tolist(), 's_next_cf': g_cf[k].tolist(),
        'dB': float(dB[k]), 'dB_batch': float(dB_b[k]),
        'dB_fail': float(dB_f[k]),
        'd_fail_diag': float(d_fail_diag[k]), 'd_fail_cf': float(d_fail_cf[k]),
        'budget_fraction': float(rho[k]),
        'cat_diag': str(cat_diag[k]), 'cat_cf': str(cat_cf[k]),
        'valid': bool(ok[k]), 'inside_wall': bool(inw[k]),
        'outside_bounds': bool(oob[k]), 'crosses_wall': bool(crosses[k]),
        'd_data_diag': float(dd_data[k]), 'd_data_cf': float(dc_data[k]),
    } for k in sel]
  out['7_examples'] = examples
  print('  saved %d examples across %d categories'
        % (sum(len(v) for v in examples.values()), len(examples)))

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % args.out)


if __name__ == '__main__':
  main()
