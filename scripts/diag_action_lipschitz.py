"""Local action-Lipschitz probe of the already-trained state+action model.

Loads the fitted state+action transition model from
artifacts/transition_diag/transition_mlp_params.pkl (written by
scripts/diag_transition_mlp.py) and measures how hard the predicted delta
reacts to the action. Nothing is retrained, no constraint is added, no weight
is touched -- this is read-only analysis of an existing model.

    J_a(s, a) = d delta_hat(s, a) / d a           in R^{2x2}
    L_local(s, a) = sigma_max(J_a(s, a))

WHY sigma_max AND NOT A NORM OF THE OUTPUT. sigma_max is the worst-case gain
over all unit action perturbations at that point, which is exactly the local
Lipschitz constant in the action, and it is what any later Lipschitz-ball
argument has to be calibrated against. A random-direction finite difference
(section 6) only ever samples a directional slope and is therefore a LOWER
bound on it; both are reported so the gap is visible rather than assumed.

DIFFERENTIATE THROUGH THE STANDARDISATION. The fitted model consumes
(x - mu) / sd, so the Jacobian with respect to the RAW action is the network
Jacobian divided elementwise by sd[2:]. Differentiating the network alone would
overstate every number here by 1/sd ~ 2x. The bundle carries mu/sd for exactly
this reason and f_single below applies them inside the differentiated function,
so the chain rule is handled by autograd rather than by hand.

ACTION CLIPPING. The behaviour actions live in [-1, 1]^2 and a large share sit
exactly on the boundary, where a symmetric perturbation would leave the valid
set. Section 5 therefore falls back to one-sided differences at the boundary,
and section 6 re-derives the perturbation actually taken after clipping
(da = clip(a + eps*u) - a) and divides by its true norm, so R is never inflated
by a step the action space does not allow.

Scope: held-out PRIMARY test transitions only -- same episode-level split and
same post-death-frozen exclusion as the training script, re-derived from the
same seed so the test set is identical.

Usage:
  python scripts/diag_action_lipschitz.py
  python scripts/diag_action_lipschitz.py --fd-subset 4000 --seed 0
"""
import argparse
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
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402

# Same module the model was fitted in: make_mlp() guarantees the identical
# architecture, and build_transitions/split_episodes guarantee the identical
# test set. Importing beats re-implementing either.
from diag_transition_mlp import (                 # noqa: E402
    SWAMP_CELLS, build_transitions, make_mlp, split_episodes)
from crl.envs import _TWO_ROUTE_SWAMP_WALLS as WALLS   # noqa: E402

PARAMS = 'artifacts/transition_diag/transition_mlp_params.pkl'
PCTS = (50, 90, 95, 99)


# --------------------------------------------------------------------------- #
# Geometry: distance from a state to the nearest blocking cell, per axis
# --------------------------------------------------------------------------- #
def barrier_tables():
  """Per free cell, the continuous coordinate of the nearest blocker each way.

  crl.envs._is_blocked treats both wall cells and the outside of [0, 9] x [0, 5]
  as blocking, so the grid edge counts as a barrier. Returned as four [9, 5]
  tables of coordinates, which turn into distances by subtracting the state.
  """
  h, w = WALLS.shape                              # 9 x 5
  xp = np.zeros((h, w)); xm = np.zeros((h, w))
  yp = np.zeros((h, w)); ym = np.zeros((h, w))
  for i in range(h):
    for j in range(w):
      k = next((k for k in range(i + 1, h) if WALLS[k, j] == 1), None)
      xp[i, j] = float(k) if k is not None else float(h)
      k = next((k for k in range(i - 1, -1, -1) if WALLS[k, j] == 1), None)
      xm[i, j] = float(k + 1) if k is not None else 0.0
      k = next((k for k in range(j + 1, w) if WALLS[i, k] == 1), None)
      yp[i, j] = float(k) if k is not None else float(w)
      k = next((k for k in range(j - 1, -1, -1) if WALLS[i, k] == 1), None)
      ym[i, j] = float(k + 1) if k is not None else 0.0
  return xp, xm, yp, ym


def wall_margin(s):
  """Min distance from each state to a blocking boundary along +-x, +-y."""
  xp, xm, yp, ym = barrier_tables()
  i = np.clip(np.floor(s[:, 0]).astype(int), 0, WALLS.shape[0] - 1)
  j = np.clip(np.floor(s[:, 1]).astype(int), 0, WALLS.shape[1] - 1)
  d = np.stack([xp[i, j] - s[:, 0], s[:, 0] - xm[i, j],
                yp[i, j] - s[:, 1], s[:, 1] - ym[i, j]], axis=1)
  return d.min(axis=1)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  out = {'n': int(v.size), 'mean': float(v.mean())}
  for p in PCTS:
    out['p%d' % p] = float(np.percentile(v, p))
  out['median'] = out.pop('p50')
  out['max'] = float(v.max())
  return out


HDR = ('  %-30s%9s%9s%9s%9s%9s%9s%9s'
       % ('', 'n', 'mean', 'median', 'p90', 'p95', 'p99', 'max'))


def row(name, d):
  if d['n'] == 0:
    return '  %-30s%9s' % (name, '-')
  return ('  %-30s%9s%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f'
          % (name, format(d['n'], ','), d['mean'], d['median'],
             d['p90'], d['p95'], d['p99'], d['max']))


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--params', default=PARAMS)
  ap.add_argument('--fd-subset', type=int, default=4000,
                  help='samples used for the finite-difference validation')
  ap.add_argument('--fd-h', type=float, default=1e-3)
  ap.add_argument('--near-wall', type=float, default=0.25,
                  help='wall-margin threshold separating near-wall from free')
  ap.add_argument('--seed', type=int, default=0,
                  help='must match the training seed: it re-derives the split')
  ap.add_argument('--json', default='artifacts/transition_diag/'
                                    'action_lipschitz.json')
  args = ap.parse_args()

  try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    commit = '(unavailable)'

  with open(args.params, 'rb') as f:
    bundle = pickle.load(f)
  b = bundle['state_action']
  params = jax.tree_util.tree_map(jnp.asarray, b['params'])
  mu, sd = jnp.asarray(b['mu'], jnp.float32), jnp.asarray(b['sd'], jnp.float32)

  print('=' * 96)
  print('LOCAL ACTION-LIPSCHITZ PROBE  (read-only; the model is not retrained)')
  print('=' * 96)
  print('  params        : %s' % args.params)
  print('  fitted on     : %s  seed %d  epochs %d'
        % (bundle['dataset'], bundle['seed'], bundle['epochs']))
  print('  best val MSE  : %.6f @ epoch %d' % (b['best_val_mse'],
                                               b['best_epoch']))
  print('  git commit    : %s' % commit)

  # ---- identical test set as the training script -------------------------
  d = build_transitions(bundle['dataset'])
  which = split_episodes(d['ep_mode'], d['ep_died'], args.seed)
  idx = np.where((which[d['ep']] == 'test') & d['primary'])[0]
  s, a, delta = d['s'][idx], d['a'][idx], d['delta'][idx]
  print('  held-out PRIMARY test transitions: %s' % format(len(idx), ','))

  net = make_mlp()

  def f_single(s1, a1):
    x = (jnp.concatenate([s1, a1]) - mu) / sd
    return net.apply(params, x)

  f_batch = jax.jit(jax.vmap(f_single))
  jac_batch = jax.jit(jax.vmap(jax.jacrev(f_single, argnums=1)))

  # ---- 1-2. Jacobian and sigma_max ---------------------------------------
  J = np.asarray(jac_batch(jnp.asarray(s), jnp.asarray(a)))   # [N, 2, 2]
  sv = np.linalg.svd(J, compute_uv=False)                     # [N, 2]
  smax, smin = sv[:, 0], sv[:, 1]
  out = {'commit': commit, 'params_path': args.params,
         'code_paths': ['scripts/diag_action_lipschitz.py',
                        'scripts/diag_transition_mlp.py'],
         'dataset': bundle['dataset'], 'seed': args.seed,
         'n_test_primary': int(len(idx))}

  print('\n' + '=' * 96)
  print('1-3. L_local = sigma_max( d delta_hat / d a ), held-out PRIMARY test')
  print('=' * 96)
  print(HDR)
  print(row('L_local = sigma_max', dist(smax)))
  print(row('sigma_min (for contrast)', dist(smin)))
  out['L_local'] = dist(smax)
  out['sigma_min'] = dist(smin)
  # Printed on its own line: the condition number reaches 1e5 near walls, which
  # overflows the %9.4f column above and silently runs two numbers together.
  cond = dist(smax / np.maximum(smin, 1e-12))
  out['condition_number'] = cond
  print('  condition sigma_max/sigma_min : median %.3f  p90 %.3f  p99 %.1f  '
        'max %.4g' % (cond['median'], cond['p90'], cond['p99'], cond['max']))
  print('    a near-rank-1 Jacobian means the model has learned that one axis '
        'is\n    blocked there, so the action cannot move the state that way.')

  # ---- 4. stratification -------------------------------------------------
  cell = np.clip(np.floor(s).astype(int), 0, [WALLS.shape[0] - 1,
                                              WALLS.shape[1] - 1])
  in_swamp = np.zeros(len(s), bool)
  for cx, cy in SWAMP_CELLS:
    in_swamp |= (cell[:, 0] == cx) & (cell[:, 1] == cy)
  margin = wall_margin(s)
  near = margin < args.near_wall

  print('\n' + '=' * 96)
  print('4. L_local stratified')
  print('=' * 96)
  print('  by the cell the CURRENT state s sits in (L_local is a property of')
  print('  the point (s,a), so the current cell is the right key; the landing')
  print('  cell used by the error table is reported after it for continuity)')
  print(HDR)
  strata = [('non-swamp (state)', ~in_swamp), ('swamp (state)', in_swamp)]
  for nm, m in strata:
    print(row(nm, dist(smax[m])))
    out.setdefault('by_state_cell', {})[nm] = dist(smax[m])

  land_sw = d['landed_swamp'][idx]
  print('\n  by the cell the step LANDS in')
  print(HDR)
  for nm, m in (('non-swamp (landing)', ~land_sw), ('swamp (landing)', land_sw)):
    print(row(nm, dist(smax[m])))
    out.setdefault('by_landing_cell', {})[nm] = dist(smax[m])

  print('\n  by wall margin = min distance from s to a blocking boundary along')
  print('  +-x / +-y, computed from crl.envs._TWO_ROUTE_SWAMP_WALLS (the grid')
  print('  edge counts as blocking, matching _is_blocked)')
  print('  margin distribution: min %.3f  p25 %.3f  median %.3f  p75 %.3f  max %.3f'
        % (margin.min(), np.percentile(margin, 25), np.median(margin),
           np.percentile(margin, 75), margin.max()))
  print(HDR)
  for nm, m in (('near-wall  (margin < %.2f)' % args.near_wall, near),
                ('free-space (margin >= %.2f)' % args.near_wall, ~near)):
    print(row(nm, dist(smax[m])))
    out.setdefault('by_wall_margin', {})[nm] = dist(smax[m])
  out['wall_margin_distribution'] = dist(margin)

  # ---- 5. finite-difference validation of the autograd Jacobian -----------
  rng = np.random.default_rng(args.seed)
  sub = rng.choice(len(idx), min(args.fd_subset, len(idx)), replace=False)
  ss, aa, Js = s[sub], a[sub], J[sub]
  h = args.fd_h
  n_bnd = int(((np.abs(aa) > 1.0 - h) ).any(axis=1).sum())
  Jfd = np.zeros_like(Js)
  for k in range(2):
    e = np.zeros((1, 2), np.float32); e[0, k] = h
    hi_ok = aa[:, k] + h <= 1.0
    lo_ok = aa[:, k] - h >= -1.0
    f0 = np.asarray(f_batch(jnp.asarray(ss), jnp.asarray(aa)))
    fp = np.asarray(f_batch(jnp.asarray(ss), jnp.asarray(aa + e)))
    fm = np.asarray(f_batch(jnp.asarray(ss), jnp.asarray(aa - e)))
    central = (fp - fm) / (2 * h)
    fwd = (fp - f0) / h
    bwd = (f0 - fm) / h
    both = (hi_ok & lo_ok)[:, None]
    col = np.where(both, central, np.where(hi_ok[:, None], fwd, bwd))
    Jfd[:, :, k] = col
  ae = np.abs(Js - Jfd)
  rel = (np.linalg.norm((Js - Jfd).reshape(len(sub), -1), axis=1)
         / np.maximum(np.linalg.norm(Js.reshape(len(sub), -1), axis=1), 1e-12))
  sfd = np.linalg.svd(Jfd, compute_uv=False)[:, 0]
  print('\n' + '=' * 96)
  print('5. finite-difference validation of the autograd Jacobian '
        '(n=%d, h=%g)' % (len(sub), h))
  print('=' * 96)
  print('  one-sided differences used where a component sits within h of the')
  print('  action bound; %d of %d sampled actions touch the boundary'
        % (n_bnd, len(sub)))
  print('  elementwise |J_autograd - J_fd| : max %.3e  mean %.3e'
        % (ae.max(), ae.mean()))
  print('  relative Frobenius error        : max %.3e  mean %.3e  median %.3e'
        % (rel.max(), rel.mean(), float(np.median(rel))))
  print('  |sigma_max diff|                : max %.3e  mean %.3e'
        % (np.abs(smax[sub] - sfd).max(), np.abs(smax[sub] - sfd).mean()))
  out['fd_validation'] = {
      'n': int(len(sub)), 'h': h, 'n_actions_at_bound': n_bnd,
      'abs_elementwise_max': float(ae.max()),
      'abs_elementwise_mean': float(ae.mean()),
      'rel_frobenius_max': float(rel.max()),
      'rel_frobenius_mean': float(rel.mean()),
      'rel_frobenius_median': float(np.median(rel)),
      'sigma_max_absdiff_max': float(np.abs(smax[sub] - sfd).max()),
      'sigma_max_absdiff_mean': float(np.abs(smax[sub] - sfd).mean())}

  # ---- 6. empirical finite-difference ratio at real perturbation scales ---
  print('\n' + '=' * 96)
  print('6. empirical ratio R = ||delta_hat(s,a+da) - delta_hat(s,a)|| / ||da||')
  print('=' * 96)
  print('  da is drawn, then a+da is CLIPPED into [-1,1]^2 and da is recomputed')
  print('  as the step actually taken, so R is never inflated by an inadmissible')
  print('  perturbation. Samples clipped to a zero-length step are dropped.')
  print('  "random dir" samples a direction uniformly -> a LOWER bound on')
  print('  L_local. "worst-case dir" steps along the top right-singular vector')
  print('  of J_a, which is the direction sigma_max is defined by, so it should')
  print('  converge to L_local as the step shrinks.')
  f0_all = np.asarray(f_batch(jnp.asarray(s), jnp.asarray(a)))
  _, _, vt = np.linalg.svd(J)
  v1 = vt[:, 0, :]                                  # top right-singular vector
  out['R'] = {}
  for tag, mkdir_ in (('random dir', 'rand'), ('worst-case dir', 'v1')):
    print('\n  -- %s' % tag)
    print(HDR)
    for eps in (0.01, 0.05, 0.1):
      if mkdir_ == 'rand':
        u = rng.normal(size=(len(idx), 2))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
      else:
        # take whichever sign of v1 survives clipping better
        cand = np.stack([v1, -v1])
        lens = [np.linalg.norm(np.clip(a + eps * c, -1, 1) - a, axis=1)
                for c in cand]
        u = np.where((lens[0] >= lens[1])[:, None], v1, -v1)
      a2 = np.clip(a + eps * u, -1, 1).astype(np.float32)
      da = a2 - a
      nda = np.linalg.norm(da, axis=1)
      keep = nda > 1e-9
      f1 = np.asarray(f_batch(jnp.asarray(s[keep]), jnp.asarray(a2[keep])))
      r = np.linalg.norm(f1 - f0_all[keep], axis=1) / nda[keep]
      st = dist(r)
      st['n_dropped_fully_clipped'] = int((~keep).sum())
      st['mean_effective_step'] = float(nda[keep].mean())
      print(row('eps=%.2f  (eff step %.4f)' % (eps, nda[keep].mean()), st))
      out['R']['%s eps=%.2f' % (tag, eps)] = st

  print('\n  reference: L_local mean %.4f  median %.4f  p99 %.4f  max %.4f'
        % (out['L_local']['mean'], out['L_local']['median'],
           out['L_local']['p99'], out['L_local']['max']))

  if args.json:
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
      json.dump(out, f, indent=2)
    print('\nwrote %s' % args.json)


if __name__ == '__main__':
  main()
