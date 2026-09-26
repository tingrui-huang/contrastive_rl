"""Trajectory replay buffer + goal-relabeling sampler (numpy).

Replaces Reverb's ``EpisodeAdder`` + the TF ``flatten_fn`` data pipeline in
``contrastive/builder.py``. Faithfully reproduces the two statistical properties
that make the original pipeline work for contrastive learning:

  1. Geometric future-goal relabeling. For a transition at time ``i`` in a
     trajectory, the goal is the goal-coords of a FUTURE state ``j > i`` sampled
     with probability proportional to ``discount**(j - i)`` -- exactly the
     ``probs = is_future_mask * discount**(j-i)`` categorical in flatten_fn.
     Implemented here with Gumbel-max sampling over the same log-probs.

  2. Cross-trajectory negatives. Each element of a batch is drawn from an
     independently sampled trajectory, so the off-diagonal (state_a, goal_b)
     pairs used as contrastive negatives come from different trajectories -- the
     effect the original achieved via its "transpose_shuffle".

Only the STATE part ``obs[:, :obs_dim]`` is used for relabeling (as in the
original); the env's own goal half of the stored observation is discarded.
"""
from typing import Optional

import numpy as np

from crl.losses import Transition


def obs_to_goal(states, start_index, end_index, goal_indices=None):
  """Slice a batch of states [., obs_dim] down to goal coords (utils port).

  ``goal_indices`` (a sequence of column indices) overrides the contiguous
  start/end slice when given -- used by the goal-representation ablation."""
  if goal_indices is not None:
    return states[:, np.asarray(goal_indices)]
  if end_index == -1:
    return states[:, start_index:]
  return states[:, start_index:end_index]


class TrajectoryBuffer:
  """Fixed-episode-length ring buffer of trajectories."""

  def __init__(self, capacity_steps, ep_len_obs, full_obs_dim, action_dim,
               obs_dim, start_index, end_index, discount, seed=0,
               goal_indices=None, obs_dtype=np.float32):
    """Args:

      capacity_steps: approx max ENV steps to retain (=> capacity in episodes).
      ep_len_obs: number of observations stored per episode (max_steps + 1).
      full_obs_dim: width of the stored (env) observation = obs_dim + goal_dim.
      action_dim: action width.
      obs_dim: width of the STATE part of the observation.
      start_index/end_index: goal-coordinate slice applied to the state.
      discount: geometric discount used for future-goal sampling.
      obs_dtype: storage dtype for observations. Image envs use uint8 (4x less
        RAM; frames are raw pixels anyway). Sampling always emits float32.
    """
    self._L = int(ep_len_obs)
    self._capacity_eps = max(1, int(capacity_steps) // self._L)
    self._obs_dim = obs_dim
    self._start_index = start_index
    self._end_index = end_index
    self._goal_indices = (None if goal_indices is None
                          else np.asarray(goal_indices, dtype=np.int64))
    self._discount = discount
    self._rng = np.random.default_rng(seed)
    self._obs_dtype = np.dtype(obs_dtype)

    self._obs = np.zeros((self._capacity_eps, self._L, full_obs_dim),
                         dtype=self._obs_dtype)
    self._act = np.zeros((self._capacity_eps, self._L, action_dim),
                         dtype=np.float32)
    # Per-episode TASK goal (goal coords) and outcome label, for the critic's
    # task-goal term. Unlike the relabeled goal these are trajectory-level and
    # anchor-independent. Left at zeros / unlabeled unless add_episode is given
    # them; `has_task_labels` is what a consumer must gate on.
    self._goal_dim = int(obs_to_goal(
        np.zeros((1, obs_dim), np.float32), start_index, end_index,
        self._goal_indices).shape[1])
    self._task_goal = np.zeros((self._capacity_eps, self._goal_dim),
                               dtype=np.float32)
    self._task_success = np.zeros(self._capacity_eps, dtype=np.float32)
    self._n_labeled = 0
    # Death-trajectory hindsight negatives (see config.death_neg_coef). Off
    # until enable_death_negatives() is called; the draw uses its OWN rng so
    # that turning the term on does not shift the ordinary relabeling stream
    # by a single number -- the coef-0 and coef>0 arms then see the identical
    # sequence of ordinary batches and the comparison is exactly paired.
    self._death_batch = 0
    self._death_pool = None
    self._death_src = self   # buffer the death pool indexes into
    self._death_rng = np.random.default_rng(int(seed) + 90_210)
    # Per-episode VALID observation count (incl. the terminal obs). All fixed
    # at L unless add_episode is given an explicit shorter length -- used so
    # future-goal relabeling never samples across a padded tail (see sample()).
    self._lengths_arr = np.full(self._capacity_eps, self._L, dtype=np.int64)
    self._use_lengths = False
    self._write = 0      # next episode slot to write (ring).
    self._num_eps = 0    # number of valid episodes stored.
    self._frozen = False  # offline mode locks the buffer (see freeze()).
    self._log_discount = float(np.log(discount)) if discount > 0 else -np.inf

  def freeze(self):
    """Make the buffer immutable: any later add_episode() raises. Used by the
    strict offline audit so environment collection is structurally impossible
    once the fixed dataset is loaded."""
    self._frozen = True

  @property
  def frozen(self):
    return self._frozen

  @property
  def has_task_labels(self):
    """True when every stored episode carries a task goal + outcome label."""
    return self._num_eps > 0 and self._n_labeled == self._num_eps

  @property
  def task_success(self):
    """Per-episode outcome labels for the stored episodes."""
    return self._task_success[:self._num_eps].copy()

  @property
  def death_episodes(self):
    """Indices of the stored episodes labeled as NOT having reached their
    task goal -- the failure ('death') trajectories. Empty until the buffer
    has per-episode labels (see ``has_task_labels``)."""
    if not self.has_task_labels:
      return np.zeros((0,), np.int64)
    return np.flatnonzero(self._task_success[:self._num_eps] <= 0.5).astype(
        np.int64)

  def enable_death_negatives(self, batch_size):
    """Make ``sample()`` also emit a death-trajectory hindsight batch.

    ``batch_size`` rows are drawn per call: an episode uniformly from the
    death pool, then an anchor time and a geometric future goal time by the
    SAME rule ``sample()`` uses for every other anchor. The only difference
    is downstream -- crl/losses.py scores the pair with label 0 instead of
    the 1 hindsight would assign it."""
    pool = self.death_episodes
    if batch_size <= 0:
      raise ValueError(f'death batch size must be positive, got {batch_size}')
    if not self.has_task_labels:
      raise RuntimeError(
          'enable_death_negatives() needs per-episode outcome labels; this '
          'buffer has none, so no episode can be called a death trajectory.')
    if pool.size == 0:
      raise RuntimeError(
          'enable_death_negatives() found 0 not-reached episodes among '
          f'{self._num_eps}: the term has nothing to draw from. The merged '
          'positives+negatives dataset, not the positives-only file, is what '
          'it needs.')
    self._death_pool = pool
    self._death_batch = int(batch_size)
    return pool.size

  def enable_external_death_negatives(self, source, batch_size):
    """Like ``enable_death_negatives``, but the death pool lives in a
    SEPARATE frozen buffer ``source`` (see config.death_dataset).

    The death episodes are then never anchors of the ordinary batch, so the
    ordinary NCE and the actor's BC term see only this buffer's episodes;
    the death trajectories reach the loss through the label-0 term alone."""
    pool = source.death_episodes
    if batch_size <= 0:
      raise ValueError(f'death batch size must be positive, got {batch_size}')
    if pool.size == 0:
      raise RuntimeError(
          'enable_external_death_negatives() found 0 not-reached episodes '
          'in the death dataset: the term has nothing to draw from.')
    self._death_src = source
    self._death_pool = pool
    self._death_batch = int(batch_size)
    return pool.size

  def add_episode(self, obs, act, length=None, task_goal=None, success=None):
    """Store one episode. obs: [L, full_obs_dim], act: [L, action_dim].

    ``length`` (optional) = number of VALID observations (<= L); the rest of
    the row is padding the relabeler must not sample. Raises if frozen.

    ``task_goal`` [goal_dim] and ``success`` (0.0/1.0) are the episode's task
    goal and whether it reached it; they must be given together and are only
    read by the critic's task-goal term."""
    if self._frozen:
      raise RuntimeError(
          'TrajectoryBuffer is frozen (offline mode): add_episode() is '
          'disabled -- the fixed dataset must never grow or change.')
    obs = np.asarray(obs, dtype=self._obs_dtype)
    act = np.asarray(act, dtype=np.float32)
    assert obs.shape[0] == self._L, (
        f'expected {self._L} obs, got {obs.shape[0]}')
    slot = self._write
    self._obs[slot] = obs
    self._act[slot] = act
    if length is None:
      self._lengths_arr[slot] = self._L
    else:
      length = int(length)
      assert 2 <= length <= self._L, f'episode length {length} out of [2, {self._L}]'
      self._lengths_arr[slot] = length
      if length != self._L:
        self._use_lengths = True
    if (task_goal is None) != (success is None):
      raise ValueError('task_goal and success must be given together')
    if task_goal is None:
      self._task_goal[slot] = 0.0
      self._task_success[slot] = 0.0
    else:
      tg = np.asarray(task_goal, np.float32).reshape(-1)
      assert tg.shape == (self._goal_dim,), (
          f'task_goal must be {self._goal_dim} goal coords, got {tg.shape}')
      self._task_goal[slot] = tg
      self._task_success[slot] = float(success)
      self._n_labeled += 1
    self._write = (self._write + 1) % self._capacity_eps
    self._num_eps = min(self._num_eps + 1, self._capacity_eps)

  def __len__(self):
    """Number of transitions currently available."""
    if self._use_lengths:
      return int(np.sum(self._lengths_arr[:self._num_eps] - 1))
    return self._num_eps * (self._L - 1)

  @property
  def ready_steps(self):
    return len(self)

  @property
  def lengths(self):
    """Per-episode valid observation counts for the stored episodes."""
    return self._lengths_arr[:self._num_eps].copy()

  def flat_transitions(self):
    """Every stored one-step transition, flattened, in storage order.

    Returns ``(states, actions, next_states, episode_ids)``, each with leading
    dimension ``len(self)``. ``states``/``next_states`` are the STATE half only
    (``obs[..., :obs_dim]``); the env's stored goal half is dropped exactly as
    in ``sample()``. Padded tails are excluded via the per-episode valid
    lengths, so no pair straddles an episode boundary.

    READ-ONLY by construction: it copies out of the storage arrays and never
    advances ``self._rng``. That matters because the goal-relabeling stream is
    shared -- a consumer that drew its data through ``sample()`` instead would
    shift every subsequent CRL batch and silently break comparability with
    previous runs. Used to fit the ETT one-step transition model.
    """
    if self._num_eps == 0:
      z = np.zeros((0, self._obs_dim), np.float32)
      return (z, np.zeros((0, self._act.shape[-1]), np.float32), z.copy(),
              np.zeros((0,), np.int64))
    s, a, sn, eid = [], [], [], []
    for e in range(self._num_eps):
      n = int(self._lengths_arr[e]) - 1          # transitions in this episode.
      s.append(self._obs[e, :n, :self._obs_dim].astype(np.float32))
      sn.append(self._obs[e, 1:n + 1, :self._obs_dim].astype(np.float32))
      a.append(self._act[e, :n].astype(np.float32))
      eid.append(np.full(n, e, dtype=np.int64))
    return (np.concatenate(s), np.concatenate(a), np.concatenate(sn),
            np.concatenate(eid))

  def flat_goals(self):
    """The stored GOAL half of every transition's observation. [len(self), G].

    Row-for-row aligned with ``flat_transitions()`` -- same episodes, same
    order, same exclusion of padded tails -- so ``concatenate([states, goals])``
    reconstructs the full observation the agent's policy network consumes.
    That is the only reason this exists: the ETT stage-0 generators
    (``CausalTransitionModel.sample`` / ``.rollout``) have to hand a
    goal-conditioned consumer a full observation, and no head predicts a goal
    -- so the drawn rows' goal is inherited, and ``flat_transitions``
    deliberately drops the goal half.

    READ-ONLY in the same sense as ``flat_transitions``: it copies out of
    storage and never advances ``self._rng``.
    """
    if self._num_eps == 0:
      return np.zeros((0, self._obs.shape[-1] - self._obs_dim), np.float32)
    g = []
    for e in range(self._num_eps):
      n = int(self._lengths_arr[e]) - 1        # transitions, not observations.
      g.append(self._obs[e, :n, self._obs_dim:].astype(np.float32))
    return np.concatenate(g)

  def content_sha256(self):
    """SHA-256 over the stored obs+act tensors (immutability checksum)."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(self._obs[:self._num_eps]).tobytes())
    h.update(np.ascontiguousarray(self._act[:self._num_eps]).tobytes())
    h.update(np.ascontiguousarray(self._lengths_arr[:self._num_eps]).tobytes())
    return h.hexdigest()

  def save(self, path):
    """Atomic snapshot (tmp + replace) of contents, pointers, and RNG state."""
    import json as _json
    import os as _os
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
      np.savez_compressed(
          f, obs=self._obs[:self._num_eps], act=self._act[:self._num_eps],
          write=self._write, num_eps=self._num_eps,
          rng_state=np.frombuffer(
              _json.dumps(self._rng.bit_generator.state).encode(),
              dtype=np.uint8))
    _os.replace(tmp, path)

  def load(self, path):
    """Restore a snapshot produced by save() into this (same-shape) buffer.

    Context-managed np.load: the file handle MUST be closed, or Windows will
    refuse the atomic os.replace of the next snapshot onto this path."""
    import json as _json
    with np.load(path) as d:
      n = int(d['num_eps'])
      assert d['obs'].shape[1:] == self._obs.shape[1:], 'buffer shape mismatch'
      assert n <= self._capacity_eps
      self._obs[:n] = d['obs']
      self._act[:n] = d['act']
      self._num_eps = n
      self._write = int(d['write']) % self._capacity_eps
      self._rng.bit_generator.state = _json.loads(
          d['rng_state'].tobytes().decode())
    return n

  def _draw_indices(self, batch_size, pool=None, rng=None):
    """Draw (traj, i, j): anchor time i and future goal time j>i, BOTH in the
    SAME trajectory (so relabeling never crosses an episode boundary) and both
    within the episode's valid length (so it never samples a padded tail).

    ``pool`` (optional) restricts the trajectory draw to those episode
    indices, and ``rng`` (optional) draws from a different stream; both are
    used only by the death-negative batch, and with both at None this is the
    original sampler, number for number."""
    L = self._L
    ne = self._num_eps
    rng = self._rng if rng is None else rng

    if pool is None:
      traj = rng.integers(0, ne, size=batch_size)        # which trajectory.
    else:
      traj = pool[rng.integers(0, pool.size, size=batch_size)]
    if not self._use_lengths:
      # Fixed-length path -- byte-identical RNG stream to the original.
      i = rng.integers(0, L - 1, size=batch_size)        # anchor in [0, L-2].
      arange = np.arange(L)                              # [L]
      future = arange[None, :] > i[:, None]              # [B, L]
    else:
      # Variable-length: mask the padded tail per row (valid = arange < len).
      Lt = self._lengths_arr[traj]                       # [B] valid obs counts.
      i = np.floor(rng.random(batch_size) * (Lt - 1)).astype(np.int64)
      arange = np.arange(L)                              # [L]
      valid = arange[None, :] < Lt[:, None]              # [B, L] within episode.
      future = (arange[None, :] > i[:, None]) & valid    # [B, L]
    logp = (arange[None, :] - i[:, None]) * self._log_discount  # [B, L]
    logits = np.where(future, logp, -np.inf)
    # Gumbel-max categorical sample over the same distribution as flatten_fn.
    g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
    j = np.argmax(logits + g, axis=1)                    # [B]
    return traj, i, j

  def sampled_indices(self, batch_size):
    """Public alias of _draw_indices for the offline relabel-boundary audit."""
    return self._draw_indices(batch_size)

  def sample(self, batch_size):
    """Sample a relabeled Transition batch (numpy arrays of size batch_size)."""
    traj, i, j = self._draw_indices(batch_size)

    # .astype(float32): no-op for state envs; uint8 -> float32 for image envs
    # (networks normalize by /255 themselves).
    state = self._obs[traj, i, :self._obs_dim].astype(np.float32, copy=False)
    next_state = self._obs[traj, i + 1, :self._obs_dim].astype(np.float32,
                                                               copy=False)
    goal_state = self._obs[traj, j, :self._obs_dim].astype(np.float32,
                                                           copy=False)
    goal = obs_to_goal(goal_state, self._start_index, self._end_index,
                       self._goal_indices)

    new_obs = np.concatenate([state, goal], axis=1)
    new_next_obs = np.concatenate([next_state, goal], axis=1)
    action = self._act[traj, i]
    next_action = self._act[traj, i + 1]

    death_obs, death_act = (None, None)
    if self._death_batch:
      death_obs, death_act = self._draw_death(self._death_batch)

    return Transition(
        observation=new_obs,
        action=action,
        reward=np.zeros((batch_size,), np.float32),       # unused by losses.
        discount=np.full((batch_size,), self._discount, np.float32),
        next_observation=new_next_obs,
        next_action=next_action,
        task_goal=self._task_goal[traj],
        task_success=self._task_success[traj],
        death_observation=death_obs,
        death_action=death_act,
    )

  def _draw_death(self, batch_size):
    """One hindsight-relabeled batch drawn only from the death trajectories.

    Same relabeling rule as ``sample()`` -- the pair is ordinary; what makes
    it a negative is the label crl/losses.py gives it."""
    src = self._death_src
    traj, i, j = src._draw_indices(   # pylint: disable=protected-access
        batch_size, pool=self._death_pool, rng=self._death_rng)
    state = src._obs[traj, i, :self._obs_dim].astype(np.float32, copy=False)
    goal_state = src._obs[traj, j, :self._obs_dim].astype(np.float32,
                                                          copy=False)
    goal = obs_to_goal(goal_state, self._start_index, self._end_index,
                       self._goal_indices)
    return np.concatenate([state, goal], axis=1), src._act[traj, i]
