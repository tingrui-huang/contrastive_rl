"""Minimal off-diagonal V0 diagnostic for the windy-swamp PointMaze.

Given an observational transition (s, a', s'_obs), keep the already-fitted
DIAGONAL transition model frozen and construct a pessimistic candidate next
state for a different intervention action a != a'.

    Delta_diag = F_diag(s, a')                  frozen, evaluated at the
                                                OBSERVED action
    R          = L * ||a - a'||                 allowed radius
    delta_phi  = z_phi(s, a, a') * min(1, R / (||z_phi|| + eps))
    Delta_cf   = Delta_diag + delta_phi
    s'_cf      = s + Delta_cf

so ||Delta_cf - Delta_diag|| <= L ||a - a'|| holds BY CONSTRUCTION, and at
a = a' the radius is exactly 0, hence delta_phi is exactly 0 and Delta_cf
reduces to the diagonal prediction. That identity is asserted numerically, and
the frozen parameters are hashed before and after training to prove the
off-diagonal fit never touches them.

THIS IS NOT A COUNTERFACTUAL TRANSITION MODEL. It does not estimate
p(s' | s, do(a)). It only searches inside a Lipschitz-compatible ball for the
candidate that scores best under a negative-similarity objective. L is a
SAMPLING/BUDGET choice, not an identified causal constant, and the same goes
for the ||a - a'|| range.

NEGATIVE OBJECTIVE -- what is reused, verbatim, from the live CRL code. See
the module docstring section "REUSED NEGATIVE SAMPLING" below and the printed
audit block at run time; nothing about the failure bank is invented here.

    B(g) = tau * log sum_j exp( -||g - g^-_j||^2 / tau ),    g^-_j ~ q_alpha

which is a soft nearest-negative score: as tau -> 0 it tends to
-min_j ||g - g^-_j||^2. Maximising B moves the candidate TOWARD the negatives.
The training loss is -B.

Deliberately absent, per the step's scope: no Flow, no discriminator, no policy
training, no spectral normalisation, no GroupSort, no change to the contrastive
loss, and no modification of the CRL/policy pipeline.

REUSED NEGATIVE SAMPLING (verified against the source, not assumed)
------------------------------------------------------------------
  q_batch   crl/losses.py critic_loss: the ordinary negatives are the
            OFF-DIAGONAL of the B x B logits matrix, i.e. for anchor i the
            goals g_j of the other anchors in the same batch. Those goals come
            from crl.replay.TrajectoryBuffer.sample() -> geometric future
            relabeling inside each anchor's own episode, then
            crl.replay.obs_to_goal. This script draws them by CALLING that same
            buffer, so q_batch is the training marginal rather than a lookalike.

  q_fail    uniform over artifacts/swamp_windy_failure_bank/failure_bank.npz,
            key 'goals', shape (256, 2) float32 -- subsampled from 514 deaths
            by make_swamp_failure_bank.py --max-bank 256, because crl/losses.py
            requires n_bank <= batch_size.

  mixture   crl/losses.py implements q_alpha at the LOSS level, not by
            sampling:
              L(alpha) = S_pos/B^2
                       + (1-alpha) * S_neg/B^2
                       + alpha * (B-1)/B * E_fail[BCE]
            with E_fail computed EXACTLY (every bank state, uniform average).
            This script needs a POOL, so it draws the sampling analogue of the
            same mixture: each negative is a q_batch draw with probability
            (1 - alpha) and a uniform bank draw with probability alpha. Same
            alpha, same two components; the difference from the original is
            sampled-vs-closed-form, and it is stated rather than hidden.

  goal map  config.start_index=0, end_index=-1, goal_indices=None, so
            crl.replay.obs_to_goal is the IDENTITY on the 2-D XY state. The
            bank's own meta agrees: "goal_indices: range(2) (start=0, end=-1)".
            Converting a candidate XY into the goal representation is therefore
            a no-op, and no new goal encoding is introduced.

  alpha     crl/config.py default fail_neg_alpha = 0.0, and the qualified
            baseline/control runs used the `baseline` arm, i.e. alpha = 0.
            run_swamp_windy_failneg.py has --alpha default 0.1 and swept
            {0.05, 0.1, 0.2}. There is therefore no single "currently
            configured" non-zero alpha; this script defaults to 0.1 (the
            launcher default and the sweep midpoint) and exposes --alpha. At
            alpha = 0 the bank never appears and the pool degenerates to
            q_batch, which would defeat the "mixed, not only the failure bank"
            requirement.

Usage:
  python scripts/diag_offdiag_v0.py
  python scripts/diag_offdiag_v0.py --alpha 0.2 --tau 0.1
  python scripts/diag_offdiag_v0.py --lipschitz 1.25 --steps 4000
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
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import haiku as hk                                # noqa: E402
import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import optax                                      # noqa: E402

from diag_transition_mlp import (                 # noqa: E402
    SWAMP_CELLS, build_transitions, make_mlp, split_episodes)
from diag_action_lipschitz import wall_margin     # noqa: E402
from crl.config import Config                     # noqa: E402
from crl import envs as envs_mod                  # noqa: E402
from crl.replay import obs_to_goal                # noqa: E402

DIAG_PARAMS = 'artifacts/transition_diag/transition_mlp_params.pkl'
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
ENV = 'point_two_route_swamp_windy_v0'
OUT_DIR = 'artifacts/transition_offdiag_v0'
EPS = 1e-8                                        # the eps in R/(||z||+eps)
PCTS = (50, 90, 95, 99)


def tree_hash(tree):
  """Stable hash of a param pytree, to prove the frozen model never moved."""
  h = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.asarray(leaf)
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
  return h.hexdigest()


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  out = {'n': int(v.size), 'mean': float(v.mean())}
  for p in PCTS:
    out['p%d' % p] = float(np.percentile(v, p))
  out['median'] = out.pop('p50')
  out['max'] = float(v.max())
  out['min'] = float(v.min())
  return out


HDR = ('  %-32s%9s%9s%9s%9s%9s%9s'
       % ('', 'n', 'mean', 'median', 'p95', 'p99', 'max'))


def row(name, d):
  if d.get('n', 0) == 0:
    return '  %-32s%9s' % (name, '-')
  return ('  %-32s%9s%9.4f%9.4f%9.4f%9.4f%9.4f'
          % (name, format(d['n'], ','), d['mean'], d['median'],
             d['p95'], d['p99'], d['max']))


# --------------------------------------------------------------------------- #
# Off-diagonal action sampling
# --------------------------------------------------------------------------- #
def sample_offdiag_actions(a_obs, rng, lo=0.05, hi=0.30, floor=0.01, tries=8):
  """a = clip(a' + m*u, -1, 1) with u uniform on the circle, m ~ U[lo, hi].

  Clipping can shorten or annihilate the step near the action bound, so the
  REALIZED ||a - a'|| is what gets reported, never the requested m. A direction
  whose realized norm falls below `floor` is redrawn (up to `tries`) rather
  than being pushed outward, which would bias the direction distribution; the
  few samples that still fail are dropped and counted.
  """
  n = len(a_obs)
  a = a_obs.copy()
  realized = np.zeros(n)
  need = np.ones(n, bool)
  n_redrawn = 0
  for k in range(tries):
    idx = np.where(need)[0]
    if idx.size == 0:
      break
    if k > 0:
      n_redrawn += idx.size
    u = rng.normal(size=(idx.size, 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    m = rng.uniform(lo, hi, size=(idx.size, 1))
    cand = np.clip(a_obs[idx] + m * u, -1.0, 1.0).astype(np.float32)
    r = np.linalg.norm(cand - a_obs[idx], axis=1)
    ok = r >= floor
    a[idx[ok]] = cand[ok]
    realized[idx[ok]] = r[ok]
    need[idx[ok]] = False
  return a, realized, (~need), int(n_redrawn), int(need.sum())


# --------------------------------------------------------------------------- #
# Negative pool ~ q_alpha  (see REUSED NEGATIVE SAMPLING above)
# --------------------------------------------------------------------------- #
class NegativePool:
  """Draws g^- ~ q_alpha = (1-alpha) q_batch + alpha q_fail."""

  def __init__(self, cfg, dataset, bank_path, alpha, seed):
    from crl.offline_audit import build_offline_buffer
    self.buffer, _ = build_offline_buffer(dataset, cfg)
    with np.load(bank_path, allow_pickle=True) as b:
      self.bank = np.asarray(b['goals'], np.float32)
    self.alpha = float(alpha)
    self.obs_dim = cfg.obs_dim
    self.rng = np.random.default_rng(seed)

  def draw(self, m):
    """Returns (goals [m, 2], from_bank [m] bool)."""
    tr = self.buffer.sample(m)
    g_batch = np.asarray(tr.observation[:, self.obs_dim:], np.float32)
    from_bank = self.rng.random(m) < self.alpha
    g = g_batch.copy()
    k = int(from_bank.sum())
    if k:
      g[from_bank] = self.bank[self.rng.integers(0, len(self.bank), k)]
    return g, from_bank


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_z_net():
  """z_phi(s, a, a') -> R^2. Same 2x256 ReLU shape as the diagonal model."""
  def f(x):
    return hk.nets.MLP([256, 256, 2], activation=jax.nn.relu)(x)
  return hk.without_apply_rng(hk.transform(f))


def soft_neg_score(g, negs, tau):
  """B(g) = tau * log sum_j exp(-||g - g_j||^2 / tau). Soft nearest-negative.

  jax.nn.logsumexp is numerically stable (it subtracts the max), so a small tau
  with far-away negatives underflows gracefully to ~ -min_j d^2 instead of
  producing -inf.
  """
  d2 = jnp.sum((g[:, None, :] - negs[None, :, :]) ** 2, axis=-1)   # [B, M]
  return tau * jax.nn.logsumexp(-d2 / tau, axis=1)                  # [B]


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--diag-params', default=DIAG_PARAMS)
  ap.add_argument('--bank', default=BANK)
  ap.add_argument('--alpha', type=float, default=0.1,
                  help='fail_neg_alpha for q_alpha; see the docstring on why '
                       'there is no single "currently configured" non-zero value')
  ap.add_argument('--tau', type=float, default=0.1,
                  help='temperature of the soft nearest-negative score B')
  ap.add_argument('--lipschitz', type=float, default=1.25,
                  help='nominal L; p99(L_local) measured 1.2094 in '
                       'artifacts/transition_diag/action_lipschitz.json')
  ap.add_argument('--l-sweep', default='1.0,1.25,1.5,2.0')
  ap.add_argument('--da-lo', type=float, default=0.05)
  ap.add_argument('--da-hi', type=float, default=0.30)
  ap.add_argument('--n-neg', type=int, default=256,
                  help='M, the negative pool size; 256 = config.batch_size')
  ap.add_argument('--steps', type=int, default=4000)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--seed', type=int, default=0,
                  help='must match the diagonal training seed: it re-derives '
                       'the episode split')
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--save-z', default='artifacts/transition_offdiag_v0/'
                                      'offdiag_v0_z.pkl',
                  help='pickle z_phi at the NOMINAL L together with the exact '
                       'evaluation design, so a downstream audit can rebuild '
                       'g_cf without retraining; pass "" to skip')
  args = ap.parse_args()

  try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    commit = '(unavailable)'

  # ---------------------------------------------------- 1. frozen diagonal
  with open(args.diag_params, 'rb') as f:
    bundle = pickle.load(f)
  b = bundle['state_action']
  d_params = jax.tree_util.tree_map(jnp.asarray, b['params'])
  d_mu = jnp.asarray(b['mu'], jnp.float32)
  d_sd = jnp.asarray(b['sd'], jnp.float32)
  diag_hash_before = tree_hash(d_params)
  diag_net = make_mlp()

  def f_diag(s, a):
    x = (jnp.concatenate([s, a], axis=-1) - d_mu) / d_sd
    return jax.lax.stop_gradient(diag_net.apply(d_params, x))

  print('=' * 96)
  print('OFF-DIAGONAL V0 DIAGNOSTIC  (pessimistic candidate inside a '
        'Lipschitz ball)')
  print('=' * 96)
  print('  git commit        : %s' % commit)
  print('  frozen diagonal   : %s' % args.diag_params)
  print('    fitted on %s  seed %d  epochs %d'
        % (bundle['dataset'], bundle['seed'], bundle['epochs']))
  print('    best val MSE %.6f @ epoch %d' % (b['best_val_mse'],
                                              b['best_epoch']))
  print('    param sha256 (before) %s' % diag_hash_before[:32])

  # ---------------------------------------------------- data + split
  d = build_transitions(bundle['dataset'])
  which = split_episodes(d['ep_mode'], d['ep_died'], args.seed)
  tr_split = which[d['ep']]
  i_tr = np.where((tr_split == 'train') & d['primary'])[0]
  i_te = np.where((tr_split == 'test') & d['primary'])[0]
  print('  transitions       : train %s   test %s   (PRIMARY, episode split)'
        % (format(len(i_tr), ','), format(len(i_te), ',')))

  # ---------------------------------------------------- negative sampling
  cfg = Config(env_name=ENV)
  envs_mod.make_env(ENV, cfg, seed=args.seed)
  cfg.discount = 0.95            # the established windy recipe
  cfg.batch_size = args.batch
  cfg.offline_dataset = bundle['dataset']
  cfg.seed = args.seed
  print('\n  REUSED NEGATIVE SAMPLING (verified against source)')
  print('    q_batch  crl.replay.TrajectoryBuffer.sample() geometric future')
  print('             relabeling, discount %.2f, then crl.replay.obs_to_goal'
        % cfg.discount)
  print('    q_fail   uniform over %s' % args.bank)
  print('    mixture  crl/losses.py q_alpha, alpha = %.3f  (loss-level there,'
        % args.alpha)
  print('             sampled here -- see the module docstring)')
  pool = NegativePool(cfg, bundle['dataset'], args.bank, args.alpha, args.seed)
  print('    bank     %s  goals %s' % (args.bank, pool.bank.shape))
  gm = obs_to_goal(np.zeros((1, cfg.obs_dim), np.float32),
                   cfg.start_index, cfg.end_index, cfg.goal_indices)
  print('    goal map obs_to_goal(start=%d, end=%d, indices=%s) -> dim %d '
        '(IDENTITY on XY)'
        % (cfg.start_index, cfg.end_index, cfg.goal_indices, gm.shape[1]))
  assert gm.shape[1] == 2, 'goal representation is not the 2-D XY identity'
  print('    tau      %.3f   M (pool size) %d' % (args.tau, args.n_neg))

  # ---------------------------------------------------- z_phi setup
  z_net = make_z_net()
  s_tr, a_tr = d['s'][i_tr], d['a'][i_tr]
  rng = np.random.default_rng(args.seed)
  a_off_tr, _, ok_tr, _, _ = sample_offdiag_actions(
      a_tr, rng, args.da_lo, args.da_hi)
  z_in_tr = np.concatenate([s_tr, a_off_tr, a_tr], axis=1)
  z_mu, z_sd = z_in_tr.mean(0), z_in_tr.std(0)
  z_sd = np.where(z_sd < 1e-6, 1.0, z_sd)
  z_mu_j, z_sd_j = jnp.asarray(z_mu, jnp.float32), jnp.asarray(z_sd, jnp.float32)

  def z_apply(zp, s, a, a_obs):
    x = (jnp.concatenate([s, a, a_obs], axis=-1) - z_mu_j) / z_sd_j
    return z_net.apply(zp, x)

  def deltas(zp, s, a, a_obs, L):
    """Returns (Delta_diag, Delta_cf, delta_phi). Delta_diag uses a' (OBSERVED)."""
    dd = f_diag(s, a_obs)
    z = z_apply(zp, s, a, a_obs)
    # ||a - a'|| EXACT (not softened): R must be exactly 0 at a == a', which
    # is what makes the diagonal identity hold bit-exactly. ||z|| is softened
    # instead, because norm() has a NaN gradient at exactly zero and an MLP
    # output can in principle land there.
    R = L * jnp.linalg.norm(a - a_obs, axis=-1, keepdims=True)
    zn = jnp.sqrt(jnp.sum(z * z, axis=-1, keepdims=True) + 1e-24)
    scale = jnp.minimum(1.0, R / (zn + EPS))
    dphi = z * scale
    return dd, dd + dphi, dphi

  # ---------------------------------------------------- train one z per L
  Ls = [float(v) for v in args.l_sweep.split(',')] if args.l_sweep \
      else [args.lipschitz]
  if args.lipschitz not in Ls:
    Ls = [args.lipschitz] + Ls

  # fixed evaluation design: one action draw and one negative pool, shared by
  # every L so the comparison across L is like-for-like
  eval_rng = np.random.default_rng(args.seed + 1)
  s_te, a_te = d['s'][i_te], d['a'][i_te]
  a_off_te, r_te, ok_te, n_redrawn, n_dropped = sample_offdiag_actions(
      a_te, eval_rng, args.da_lo, args.da_hi)
  s_te, a_te, a_off_te, r_te = (s_te[ok_te], a_te[ok_te], a_off_te[ok_te],
                                r_te[ok_te])
  te_keep = i_te[ok_te]
  neg_eval, neg_from_bank = pool.draw(args.n_neg)
  print('\n  eval design: %s test transitions, %d redrawn for clipping, '
        '%d dropped' % (format(len(s_te), ','), n_redrawn, n_dropped))
  print('  eval negative pool: %d goals, %d from the failure bank (%.1f%%, '
        'alpha=%.2f)' % (args.n_neg, int(neg_from_bank.sum()),
                         100 * neg_from_bank.mean(), args.alpha))

  jneg = jnp.asarray(neg_eval)
  results = {}

  def loss_fn(zp, s, a, a_obs, negs, L):
    _, dcf, _ = deltas(zp, s, a, a_obs, L)
    g_cf = s + dcf                     # obs_to_goal is the identity here
    return -jnp.mean(soft_neg_score(g_cf, negs, args.tau))

  for L in Ls:
    print('\n' + '=' * 96)
    print('L = %.2f' % L)
    print('=' * 96)
    zp = z_net.init(jax.random.PRNGKey(args.seed),
                    jnp.asarray((z_in_tr[:2] - z_mu) / z_sd))
    opt = optax.adam(args.lr)
    ostate = opt.init(zp)

    @jax.jit
    def step(zp, ostate, s, a, a_obs, negs):
      l, g = jax.value_and_grad(loss_fn)(zp, s, a, a_obs, negs, L)
      upd, ostate = opt.update(g, ostate)
      return optax.apply_updates(zp, upd), ostate, l

    trng = np.random.default_rng(args.seed + 2)
    for t in range(args.steps):
      # Oversample 2x and keep exactly `batch` survivors: letting the batch
      # size float would retrace the jitted step on every new shape.
      sl = trng.integers(0, len(i_tr), args.batch * 2)
      ss, aa_obs = d['s'][i_tr[sl]], d['a'][i_tr[sl]]
      aa, _, ok, _, _ = sample_offdiag_actions(aa_obs, trng,
                                               args.da_lo, args.da_hi)
      keep = np.where(ok)[0]
      assert keep.size >= args.batch, 'too many clipped draws to fill a batch'
      keep = keep[:args.batch]
      negs, _ = pool.draw(args.n_neg)   # fresh pool per step, like in-batch negs
      zp, ostate, l = step(zp, ostate, jnp.asarray(ss[keep]),
                           jnp.asarray(aa[keep]), jnp.asarray(aa_obs[keep]),
                           jnp.asarray(negs))
      if t % 1000 == 0 or t == args.steps - 1:
        print('    step %5d  -B = %.5f' % (t, float(l)))

    # ------------------------------------------------ 5. diagonal identity
    dd0, dcf0, dphi0 = deltas(zp, jnp.asarray(s_te[:2048]),
                              jnp.asarray(a_te[:2048]),      # a == a'
                              jnp.asarray(a_te[:2048]), L)
    max_dphi0 = float(jnp.abs(dphi0).max())
    max_gap0 = float(jnp.abs(dcf0 - dd0).max())
    print('  [5] diagonal consistency at a == a\':  max|delta_phi| = %.3e  '
          'max|Delta_cf - Delta_diag| = %.3e' % (max_dphi0, max_gap0))
    assert max_dphi0 == 0.0 and max_gap0 == 0.0, (
        'a == a\' must give exactly zero displacement')

    # ------------------------------------------------ diagnostics
    dd, dcf, dphi = deltas(zp, jnp.asarray(s_te), jnp.asarray(a_off_te),
                           jnp.asarray(a_te), L)
    dd, dcf, dphi = np.asarray(dd), np.asarray(dcf), np.asarray(dphi)
    g_diag, g_cf = s_te + dd, s_te + dcf
    ndphi = np.linalg.norm(dphi, axis=1)
    ratio = ndphi / np.maximum(r_te, 1e-12)
    B_diag = np.asarray(soft_neg_score(jnp.asarray(g_diag), jneg, args.tau))
    B_cf = np.asarray(soft_neg_score(jnp.asarray(g_cf), jneg, args.tau))
    dB = B_cf - B_diag

    print('\n  [A] Lipschitz constraint  R_i = ||Delta_cf - Delta_diag|| / '
          '||a - a\'||')
    print(HDR)
    print(row('R_i', dist(ratio)))
    viol = int((ratio > L + 1e-6).sum())
    print('    violations of R_i <= %.2f : %d of %s' % (L, viol,
                                                        format(len(ratio), ',')))

    print('\n  [B] negative score, same fixed pool before vs after displacement')
    print('    mean   B(g_diag) %+.5f -> B(g_cf) %+.5f   change %+.5f'
          % (B_diag.mean(), B_cf.mean(), dB.mean()))
    print('    median B(g_diag) %+.5f -> B(g_cf) %+.5f   change %+.5f'
          % (np.median(B_diag), np.median(B_cf), np.median(dB)))
    print('    fraction with B(g_cf) > B(g_diag) : %.4f' % (dB > 0).mean())

    print('\n  [C] displacement')
    print(HDR)
    print(row('||a - a\'||  (realized)', dist(r_te)))
    print(row('||delta_phi||', dist(ndphi)))
    print(row('||delta_phi|| / (L||a-a\'||)', dist(ratio / L)))

    results['L=%.2f' % L] = {
        'L': L,
        'diagonal_identity': {'max_abs_delta_phi': max_dphi0,
                              'max_abs_gap': max_gap0},
        'A_lipschitz_ratio': dist(ratio),
        'A_violations': viol,
        'B_score': {
            'mean_B_diag': float(B_diag.mean()), 'mean_B_cf': float(B_cf.mean()),
            'mean_change': float(dB.mean()),
            'median_B_diag': float(np.median(B_diag)),
            'median_B_cf': float(np.median(B_cf)),
            'median_change': float(np.median(dB)),
            'frac_improved': float((dB > 0).mean())},
        'C_displacement': {
            'da_norm': dist(r_te), 'delta_phi_norm': dist(ndphi),
            'budget_fraction': dist(ratio / L)},
    }

    if abs(L - args.lipschitz) < 1e-9 and args.save_z:
      # The evaluation design travels with the params: the audit must score the
      # SAME off-diagonal actions against the SAME negative pool, and both are
      # drawn from RNG streams that depend on the whole sweep's call order.
      os.makedirs(os.path.dirname(args.save_z) or '.', exist_ok=True)
      with open(args.save_z, 'wb') as f:
        pickle.dump({'z_params': jax.tree_util.tree_map(np.asarray, zp),
                     'z_mu': z_mu, 'z_sd': z_sd, 'L': L, 'tau': args.tau,
                     'alpha': args.alpha, 'eps_projection': EPS,
                     'diag_params_path': args.diag_params,
                     'dataset': bundle['dataset'], 'seed': args.seed,
                     'test_index': te_keep, 's_te': s_te, 'a_te': a_te,
                     'a_off_te': a_off_te, 'da_norm_te': r_te,
                     'neg_eval': neg_eval, 'neg_from_bank': neg_from_bank},
                    f)
      print('  saved z_phi + eval design -> %s' % args.save_z)

    if abs(L - args.lipschitz) < 1e-9:
      # ---------------------------------------------- D. spatial examples
      cell = np.clip(np.floor(s_te).astype(int), 0, [8, 4])
      in_sw = np.zeros(len(s_te), bool)
      for cx, cy in SWAMP_CELLS:
        in_sw |= (cell[:, 0] == cx) & (cell[:, 1] == cy)
      marg = wall_margin(s_te)
      regions = {'swamp': in_sw,
                 'near_wall': (~in_sw) & (marg < 0.25),
                 'normal_corridor': (~in_sw) & (marg >= 0.25)}
      ex = {}
      erng = np.random.default_rng(args.seed + 3)
      for rname, rmask in regions.items():
        pick = np.where(rmask)[0]
        if pick.size == 0:
          continue
        pick = erng.choice(pick, min(3, pick.size), replace=False)
        ex[rname] = [{
            's': s_te[k].tolist(), 'a_obs': a_te[k].tolist(),
            'a_intervention': a_off_te[k].tolist(),
            'da_norm': float(r_te[k]),
            'delta_diag': dd[k].tolist(), 'delta_cf': dcf[k].tolist(),
            'delta_phi': dphi[k].tolist(),
            's_next_diag': g_diag[k].tolist(), 's_next_cf': g_cf[k].tolist(),
            'allowed_radius': float(L * r_te[k]),
            'budget_used': float(ratio[k] / L),
            'B_diag': float(B_diag[k]), 'B_cf': float(B_cf[k]),
            'wall_margin': float(marg[k]),
        } for k in pick]
      results['D_examples'] = ex
      results['D_negative_pool'] = {
          'n': int(args.n_neg),
          'n_from_failure_bank': int(neg_from_bank.sum()),
          'goals': neg_eval.tolist(),
          'from_failure_bank': neg_from_bank.tolist()}
      print('\n  [D] saved %d spatial examples (%s)'
            % (sum(len(v) for v in ex.values()),
               ', '.join('%s:%d' % (k, len(v)) for k, v in ex.items())))

  # ---------------------------------------------------- L monotonicity
  print('\n' + '=' * 96)
  print('7. L sensitivity (z_phi RETRAINED per L; the projection radius enters')
  print('   training, so reusing one fit across L would conflate the two)')
  print('=' * 96)
  print('  %-8s%14s%14s%16s%14s' % ('L', 'mean||dphi||', 'budget frac',
                                    'mean dB', 'frac improved'))
  for L in sorted(Ls):
    r = results['L=%.2f' % L]
    print('  %-8.2f%14.4f%14.4f%+16.5f%14.4f'
          % (L, r['C_displacement']['delta_phi_norm']['mean'],
             r['C_displacement']['budget_fraction']['mean'],
             r['B_score']['mean_change'], r['B_score']['frac_improved']))
  print('\n  Expected: L up -> larger allowed displacement -> higher achievable')
  print('  negative similarity. None of these L values is an identified causal')
  print('  Lipschitz constant; they are budgets.')

  diag_hash_after = tree_hash(d_params)
  print('\n  frozen diagonal param sha256 (after)  %s' % diag_hash_after[:32])
  print('  UNCHANGED: %s' % (diag_hash_before == diag_hash_after))
  assert diag_hash_before == diag_hash_after, 'the diagonal model was mutated'

  out = {
      'commit': commit,
      'code_paths': ['scripts/diag_offdiag_v0.py',
                     'scripts/diag_transition_mlp.py',
                     'scripts/diag_action_lipschitz.py'],
      'frozen_diagonal': {
          'path': args.diag_params, 'dataset': bundle['dataset'],
          'seed': bundle['seed'], 'epochs': bundle['epochs'],
          'best_val_mse': b['best_val_mse'],
          'param_sha256_before': diag_hash_before,
          'param_sha256_after': diag_hash_after,
          'unchanged': diag_hash_before == diag_hash_after},
      'reused_negative_sampling': {
          'q_batch': 'crl.replay.TrajectoryBuffer.sample geometric future '
                     'relabeling + crl.replay.obs_to_goal (via '
                     'crl.offline_audit.build_offline_buffer)',
          'q_fail': 'uniform over %s key "goals"' % args.bank,
          'mixture': 'crl/losses.py critic_loss q_alpha; loss-level there, '
                     'sampled analogue here',
          'alpha': args.alpha,
          'alpha_note': 'crl/config.py default 0.0; run_swamp_windy_failneg.py '
                        '--alpha default 0.1, swept {0.05,0.1,0.2}; the '
                        'qualified baseline used the baseline arm at alpha=0',
          'goal_map': 'obs_to_goal(start=%d, end=%d, indices=%s) = identity on '
                      'XY' % (cfg.start_index, cfg.end_index, cfg.goal_indices),
          'discount': cfg.discount,
          'bank_shape': list(pool.bank.shape)},
      'hyperparameters': {
          'L_nominal': args.lipschitz, 'L_sweep': Ls, 'tau': args.tau,
          'alpha': args.alpha, 'M_negatives': args.n_neg,
          'da_range': [args.da_lo, args.da_hi], 'eps_projection': EPS,
          'z_net': '2x256 ReLU MLP, input (s,a,a\') dim 6 -> 2',
          'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
          'seed': args.seed,
          'n_test_transitions': int(len(s_te)),
          'n_redrawn_for_clipping': n_redrawn, 'n_dropped': n_dropped},
      'results': results,
  }
  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'offdiag_v0.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % path)


if __name__ == '__main__':
  main()
