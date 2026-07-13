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

  def add_episode(self, obs, act, length=None):
    """Store one episode. obs: [L, full_obs_dim], act: [L, action_dim].

    ``length`` (optional) = number of VALID observations (<= L); the rest of
    the row is padding the relabeler must not sample. Raises if frozen."""
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

  def _draw_indices(self, batch_size):
    """Draw (traj, i, j): anchor time i and future goal time j>i, BOTH in the
    SAME trajectory (so relabeling never crosses an episode boundary) and both
    within the episode's valid length (so it never samples a padded tail)."""
    L = self._L
    ne = self._num_eps
    rng = self._rng

    traj = rng.integers(0, ne, size=batch_size)          # which trajectory.
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

    return Transition(
        observation=new_obs,
        action=action,
        reward=np.zeros((batch_size,), np.float32),       # unused by losses.
        discount=np.full((batch_size,), self._discount, np.float32),
        next_observation=new_next_obs,
        next_action=next_action,
    )
