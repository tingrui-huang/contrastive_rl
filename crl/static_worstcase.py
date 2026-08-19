"""Frozen static worst-case next-state module:  (s, a) -> s'_wc.

The complete pipeline sealed by the selector-confirm50 evaluation:

    (s, a)  ->  V3 Flow  ->  K=256 candidates  ->  negative-state similarity  ->  s'_wc

Every constant here is FROZEN and recorded in
``artifacts/state_nn_selector_confirm/selector_freeze.json``, which was written
BEFORE the confirmation set existed. This module re-verifies all of them (plus
file SHA-256s) at construction and refuses to build on any drift.

Selector rule (frozen, verbatim from the freeze file):

    d_neg(s') = min_{g in D_C^-} || norm(s') - norm(g) ||_2
    s'_wc     = argmin_{k <= 256} d_neg(s'_k)          # numpy argmin semantics:
                                                       # ties -> LOWEST index k

with ``norm`` the frozen V0 observable-state standardization and ``D_C^-`` the
16 settled fatal states that Critic C was trained against. Critic C itself does
NOT enter selection; the 603-state expanded bank is NOT used.

Information contract (2B). ``worst_case_next_state`` receives ONLY ``(s, a)``.
It never reads ``_dead``, the rockfall mask, severity, the hidden U, the future
rock schedule, or any same-anchor oracle pairing -- there is no env handle and
no dataset handle in this module at all.

Gradients (0D). Everything returned is a stop-gradient numpy array: the sampler
is jitted for speed but the module exposes no differentiable path, and neither
the generator params, the bank, nor the normalization are ever passed to an
optimizer. This module is import-safe from a numpy replay sampler.

NOTE ON SCOPE. This is the generator/selector ONLY. It does NOT decide where
s'_wc enters the RL objective -- see artifacts/static_worstcase_rl/
integration_audit.md, whose hard gate is currently FAILED. Nothing in crl/
calls this module yet.
"""
import hashlib
import json
import os

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------- constants
# All frozen; see selector_freeze.json. Do not edit.
OBS_DIM = 29
ACT_DIM = 8
K_CANDIDATES = 256
ODE_STEPS = 50
SEED = 11
BANK_SIZE = 16

FREEZE_JSON = 'artifacts/state_nn_selector_confirm/selector_freeze.json'
V3_CKPT = 'artifacts/flow_v3_diverse_failure/flow_v3/flow_v3.pkl'
NORM_NPZ = 'artifacts/flow_v0_clean/norm_stats.npz'
BANK_NPZ = 'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz'

V3_SHA = '7b0ac9c80afa8713155d30ee4b784b52bd6d5b58fc57ff73b6b900a11224803c'
NORM_SHA = '262daa472316773b441e0dfed897275ffac13e10966728d39d4f9e23ffe8d4ca'
BANK_SHA = '8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce'
# The 603-state expanded bank must never be substituted for D_C^-.
FORBIDDEN_BANK_NAMES = ('failure_bank_diverse', 'failure_bank_603')


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def _vfield(hidden, dim):
  """V0.5/V1/V2/V3 action-conditioned velocity field (bit-identical net)."""
  def _v(x, t, s, a):
    h = jnp.concatenate([x, t, s, a], axis=-1)
    return hk.nets.MLP(list(hidden) + [dim], activation=jax.nn.relu,
                       name='vfield')(h)
  return hk.without_apply_rng(hk.transform(_v))


class StaticWorstCase:
  """Frozen (s, a) -> s'_wc module. Construct once; call many times.

  Args:
    root: repo root the frozen artifact paths are resolved against.
    strict: when True (default) every provenance gate is fatal.
  """

  def __init__(self, root='.', strict=True):
    import pickle

    self.root = root
    p_ck = os.path.join(root, V3_CKPT)
    p_nm = os.path.join(root, NORM_NPZ)
    p_bk = os.path.join(root, BANK_NPZ)
    p_fz = os.path.join(root, FREEZE_JSON)

    # ---- 2A provenance gates ------------------------------------------
    gates = {}
    self.sha = {'v3': sha256_file(p_ck), 'norm': sha256_file(p_nm),
                'bank': sha256_file(p_bk)}
    gates['v3_ckpt_sha'] = self.sha['v3'] == V3_SHA
    gates['normalization_sha'] = self.sha['norm'] == NORM_SHA
    gates['negative_bank_sha'] = self.sha['bank'] == BANK_SHA
    gates['bank_is_not_expanded_603'] = not any(
        n in p_bk for n in FORBIDDEN_BANK_NAMES)

    fz = json.load(open(p_fz))
    gates['selector_frozen_flag'] = bool(fz.get('SELECTOR_FROZEN'))
    gates['freeze_K'] = int(fz['generator']['K']) == K_CANDIDATES
    gates['freeze_ode_steps'] = int(fz['generator']['ode_steps']) == ODE_STEPS
    gates['freeze_lambda'] = float(fz['generator']['lambda']) == 0.01
    gates['freeze_bank_sha'] = fz['negative_bank']['sha256'] == BANK_SHA
    gates['freeze_bank_size'] = int(fz['negative_bank']['n_states']) == BANK_SIZE
    gates['freeze_norm_sha'] = fz['normalization']['sha256'] == NORM_SHA
    gates['freeze_metric_is_euclidean'] = 'Euclidean' in fz['metric']
    gates['critic_not_in_selector'] = (
        fz['critic_in_primary_selector'] is False)
    gates['tie_break_lowest_index'] = 'lowest candidate index' in fz['tie_break']

    ck = pickle.load(open(p_ck, 'rb'))
    gates['ckpt_lambda'] = float(ck['lam']) == 0.01
    gates['ckpt_obs_dim'] = int(ck['obs_dim']) == OBS_DIM
    gates['ckpt_run_id'] = ck['run_id'] == fz['generator']['run_id']

    bank = np.asarray(np.load(p_bk, allow_pickle=True)['goals'], np.float32)
    gates['bank_shape'] = bank.shape == (BANK_SIZE, OBS_DIM)
    gates['bank_finite'] = bool(np.isfinite(bank).all())

    self.gates = gates
    failed = [k for k, v in gates.items() if not v]
    if failed and strict:
      raise RuntimeError(
          'STATIC WORST-CASE PROVENANCE GATE FAILED: %s' % failed)

    # ---- frozen tensors ------------------------------------------------
    nz = np.load(p_nm)
    self.nrm = {k: np.asarray(nz[k], np.float32) for k in
                ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
    self.bank = bank
    self.bank_n = (bank - self.nrm['state_mean']) / self.nrm['state_std']
    self.hidden = tuple(ck['hidden'])
    self.run_id = ck['run_id']

    net = _vfield(self.hidden, OBS_DIM)
    params = ck['params']

    @jax.jit
    def _v(x, t, s, a):
      return net.apply(params, x, t, s, a)
    self._v = _v

    bank_n = jnp.asarray(self.bank_n)

    @jax.jit
    def _select(s_raw, s_n, a, key, s_mean, s_std, d_mean, d_std):
      """One fused frozen pass: sample K, then argmin nearest-negative.

      s_raw/s_n [n, 29] raw/normalized anchors, a [n, 8] raw actions. Returns
      (k_sel [n], d_neg_sel [n], delta [n, K, 29]).

      The candidate state is formed as ``s_raw + delta`` -- NOT by
      de-normalizing s_n -- so it is bit-identical to the sealed evaluation.
      The nearest-negative reduction loops over the 16 bank states instead of
      materializing an [n, K, 16, 29] tensor, so memory stays O(n*K*29)."""
      n = s_raw.shape[0]
      s_rep = jnp.repeat(s_n, K_CANDIDATES, axis=0)
      a_rep = jnp.repeat(a, K_CANDIDATES, axis=0)
      x = jax.random.normal(key, (n * K_CANDIDATES, OBS_DIM))
      dt = 1.0 / ODE_STEPS
      for i in range(ODE_STEPS):
        tt = jnp.full((n * K_CANDIDATES, 1), i * dt)
        x = x + dt * _v(x, tt, s_rep, a_rep)
      dlt = (x * d_std + d_mean).reshape(n, K_CANDIDATES, OBS_DIM)
      cand = s_raw[:, None] + dlt                          # [n, K, 29]
      cn = (cand - s_mean) / s_std
      d_neg = jnp.stack(
          [jnp.linalg.norm(cn - bank_n[b], axis=-1) for b in range(BANK_SIZE)],
          axis=-1).min(-1)                                 # [n, K]
      k = jnp.argmin(d_neg, axis=1)                        # ties -> lowest k
      return k, jnp.take_along_axis(d_neg, k[:, None], 1)[:, 0], dlt
    self._select_jit = _select

  # -------------------------------------------------------------- public API
  def worst_case_next_state(self, s, a, return_aux=False):
    """(s, a) -> s'_wc. s [n, 29], a [n, 8]; both raw, numpy or jax.

    Returns s_wc [n, 29] (numpy float32). With ``return_aux`` also returns a
    dict with the selected index, its nearest-negative distance, and the full
    candidate block -- diagnostics only; the selection never uses them.
    """
    s = np.atleast_2d(np.asarray(s, np.float32))
    a = np.atleast_2d(np.asarray(a, np.float32))
    if s.shape[1] != OBS_DIM:
      raise ValueError('expected state dim %d, got %d' % (OBS_DIM, s.shape[1]))
    if a.shape[1] != ACT_DIM:
      raise ValueError('expected action dim %d, got %d' % (ACT_DIM, a.shape[1]))
    if s.shape[0] != a.shape[0]:
      raise ValueError('batch mismatch: %d states vs %d actions'
                       % (s.shape[0], a.shape[0]))

    s_n = (s - self.nrm['state_mean']) / self.nrm['state_std']
    k, d_sel, dlt = self._select_jit(
        jnp.asarray(s), jnp.asarray(s_n), jnp.asarray(a),
        jax.random.PRNGKey(SEED),
        jnp.asarray(self.nrm['state_mean']), jnp.asarray(self.nrm['state_std']),
        jnp.asarray(self.nrm['delta_mean']), jnp.asarray(self.nrm['delta_std']))
    k = np.asarray(k)
    dlt = np.asarray(dlt)
    cand = s[:, None] + dlt
    s_wc = cand[np.arange(len(s)), k]
    if not np.isfinite(s_wc).all():
      raise FloatingPointError('non-finite worst-case state generated')
    if not return_aux:
      return s_wc.astype(np.float32)
    return s_wc.astype(np.float32), {
        'k': k, 'd_neg': np.asarray(d_sel), 'candidates': cand}

  __call__ = worst_case_next_state

  def provenance(self):
    return {'run_id': self.run_id, 'K': K_CANDIDATES, 'ode_steps': ODE_STEPS,
            'seed_convention': 'jax.random.PRNGKey(%d)' % SEED,
            'metric': 'Euclidean L2 in frozen-normalized state space',
            'tie_break': 'lowest candidate index k (numpy argmin semantics)',
            'critic_in_primary_selector': False,
            'bank_size': BANK_SIZE, 'sha256': dict(self.sha),
            'gates': dict(self.gates)}
