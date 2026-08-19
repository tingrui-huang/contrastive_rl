"""Positive-goal sampler with an absorbing pessimistic branch (task section 6).

Approximate support-restricted pessimistic occupancy sampling. This is NOT a
certified Manski lower bound: V3 candidate coverage does not establish
N_Flow(s, a) contains supp P(s'|s,a) (measured 31/50 on selector-confirm50), so
no theorem claim attaches to it.

The ONLY thing this changes is the distribution the contrastive POSITIVE goal
is drawn from:

    B_t ~ Bernoulli(rho_t)
    g+  =  nominal future goal from crl/replay.py           if B_t = 1
           obs_to_goal(s'_wc,t)                             if B_t = 0

with s'_wc,t read from the precomputed static table. The pessimistic state is
ABSORBING by definition (task section 3): s'_wc -> s'_wc -> ..., so every later
discounted-future draw is still s'_wc and the walk simply returns it. The
sampler therefore never continues the dataset walk from a synthetic state,
never queries the policy at s'_wc, never projects s'_wc onto a dataset state,
and never runs the Flow again.

"Positive" means "drawn from the future occupancy of this (s, a)", NOT
"desirable" (task section 4). A fatal-looking s'_wc is a legitimate positive
for a risky anchor under the pessimistic world, and this does not conflict with
the same state acting as a failure NEGATIVE for unrelated anchors -- contrastive
labels are anchor-dependent.

rho IS AN INJECTED INPUT. This module deliberately does NOT choose, calibrate
or default it: ``rho_fn`` is a required constructor argument. The G2 audit
(artifacts/static_worstcase_rl/g2_calibration_audit.json) found the available
discriminator D_psi is NOT a calibrated Bernoulli probability, so no coin is
wired here.

Everything else -- anchor law, the nominal future-goal law, state/action/
next_state fields, ordinary negatives (cross-batch off-diagonals), the
failure-negative bank and alpha -- delegates to the wrapped TrajectoryBuffer
and to crl/losses.py, both untouched.
"""
import numpy as np

from crl.losses import Transition
from crl.replay import obs_to_goal


class PessimisticPositiveBuffer:
  """Wraps a frozen TrajectoryBuffer; only ``sample()``'s goal changes.

  Args:
    base: the frozen ``TrajectoryBuffer`` (offline mode).
    table_npz: path to the static worst-case table produced by
      ``scripts/precompute_worstcase_table.py``.
    rho_fn: ``rho_fn(state, g_cmd, action) -> rho [B] in [0, 1]``. REQUIRED.
      Called with raw numpy arrays. Must not be a thresholded or hand-set
      constant unless the caller has explicitly justified it.
    seed: seed of the coin's OWN rng stream. Kept separate from the base
      buffer's rng so that, with the coin forced to 1, this sampler is
      bitwise identical to ``base.sample()`` under the same base rng state.
  """

  def __init__(self, base, table_npz, rho_fn, seed=0):
    if rho_fn is None:
      raise ValueError(
          'rho_fn is required: the pessimistic branch probability must be '
          'supplied explicitly. See the G2 calibration audit -- D_psi is not '
          'a calibrated Bernoulli probability and must not be used as rho.')
    self._base = base
    self._rho_fn = rho_fn
    self._coin_rng = np.random.default_rng(seed)
    # cumulative branch accounting (diagnostics only; never feeds the loss)
    self._n_nominal = 0
    self._n_worstcase = 0
    self._rho_sum = 0.0
    self._rho_n = 0

    with np.load(table_npz, allow_pickle=True) as t:
      flat = np.asarray(t['flat_index'], np.int64)
      self._s_wc = np.asarray(t['s_wc'], np.float32)
    self._L = base._L
    # dense flat -> table row map; -1 marks a transition with no table entry
    self._row = np.full(base._obs.shape[0] * self._L, -1, np.int64)
    self._row[flat] = np.arange(len(flat), dtype=np.int64)
    self.n_table_rows = len(flat)

  def __getattr__(self, name):
    return getattr(self._base, name)

  # -------------------------------------------------------------- sampling
  def sample(self, batch_size, force_branch=None, return_aux=False):
    """Sample a relabeled Transition batch with the pessimistic branch.

    ``force_branch`` (test hook): 1 = always nominal, 0 = always pessimistic,
    None = draw the coin. Forcing does NOT consume the coin rng, so a forced
    call leaves the coin stream untouched.
    """
    b = self._base
    traj, i, j = b._draw_indices(batch_size)     # EXACT existing anchor+future law

    state = b._obs[traj, i, :b._obs_dim].astype(np.float32, copy=False)
    next_state = b._obs[traj, i + 1, :b._obs_dim].astype(np.float32, copy=False)
    action = b._act[traj, i]
    next_action = b._act[traj, i + 1]
    g_cmd = b._obs[traj, i, b._obs_dim:].astype(np.float32, copy=False)

    # ---- nominal positive goal: byte-identical to TrajectoryBuffer.sample --
    goal_state = b._obs[traj, j, :b._obs_dim].astype(np.float32, copy=False)

    # ---- coin -------------------------------------------------------------
    if force_branch is None:
      rho = np.asarray(self._rho_fn(state, g_cmd, action), np.float64)
      if rho.shape != (batch_size,):
        raise ValueError('rho_fn must return shape (%d,), got %s'
                         % (batch_size, rho.shape))
      if not np.all((rho >= 0.0) & (rho <= 1.0)):
        raise ValueError('rho_fn returned values outside [0, 1]')
      nominal = self._coin_rng.random(batch_size) < rho
    else:
      if force_branch not in (0, 1):
        raise ValueError('force_branch must be 0, 1 or None')
      rho = np.full(batch_size, float(force_branch))
      nominal = np.full(batch_size, bool(force_branch))

    # ---- pessimistic branch: ABSORBING table lookup, nothing else ----------
    if not nominal.all():
      pess = np.where(~nominal)[0]
      rows = self._row[traj[pess] * self._L + i[pess]]
      if np.any(rows < 0):
        raise KeyError('worst-case table has no entry for %d sampled '
                       'transitions' % int((rows < 0).sum()))
      goal_state = goal_state.copy()
      goal_state[pess] = self._s_wc[rows]

    goal = obs_to_goal(goal_state, b._start_index, b._end_index,
                       b._goal_indices)
    out = Transition(
        observation=np.concatenate([state, goal], axis=1),
        action=action,
        reward=np.zeros((batch_size,), np.float32),
        discount=np.full((batch_size,), b._discount, np.float32),
        next_observation=np.concatenate([next_state, goal], axis=1),
        next_action=next_action)
    self._n_nominal += int(nominal.sum())
    self._n_worstcase += int((~nominal).sum())
    self._rho_sum += float(rho.sum())
    self._rho_n += batch_size
    if not return_aux:
      return out
    return out, {'nominal': nominal, 'rho': rho, 'traj': traj, 'i': i, 'j': j,
                 'goal_state': goal_state}

  def branch_stats(self):
    """Cumulative branch accounting. Diagnostics only."""
    tot = self._n_nominal + self._n_worstcase
    return {'n_nominal': self._n_nominal, 'n_worstcase': self._n_worstcase,
            'realized_wc_rate': (self._n_worstcase / tot) if tot else 0.0,
            'mean_rho': (self._rho_sum / self._rho_n) if self._rho_n else 0.0}
