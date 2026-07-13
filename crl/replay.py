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
               goal_indices=None):
    """Args:

      capacity_steps: approx max ENV steps to retain (=> capacity in episodes).
      ep_len_obs: number of observations stored per episode (max_steps + 1).
      full_obs_dim: width of the stored (env) observation = obs_dim + goal_dim.
      action_dim: action width.
      obs_dim: width of the STATE part of the observation.
      start_index/end_index: goal-coordinate slice applied to the state.
      discount: geometric discount used for future-goal sampling.
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

    self._obs = np.zeros((self._capacity_eps, self._L, full_obs_dim),
                         dtype=np.float32)
    self._act = np.zeros((self._capacity_eps, self._L, action_dim),
                         dtype=np.float32)
    self._write = 0      # next episode slot to write (ring).
    self._num_eps = 0    # number of valid episodes stored.
    self._log_discount = float(np.log(discount)) if discount > 0 else -np.inf

  def add_episode(self, obs, act):
    """Store one episode. obs: [L, full_obs_dim], act: [L, action_dim]."""
    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    assert obs.shape[0] == self._L, (
        f'expected {self._L} obs, got {obs.shape[0]}')
    slot = self._write
    self._obs[slot] = obs
    self._act[slot] = act
    self._write = (self._write + 1) % self._capacity_eps
    self._num_eps = min(self._num_eps + 1, self._capacity_eps)

  def __len__(self):
    """Number of transitions currently available."""
    return self._num_eps * (self._L - 1)

  @property
  def ready_steps(self):
    return len(self)

  def sample(self, batch_size):
    """Sample a relabeled Transition batch (numpy arrays of size batch_size)."""
    L = self._L
    ne = self._num_eps
    rng = self._rng

    traj = rng.integers(0, ne, size=batch_size)          # which trajectory.
    i = rng.integers(0, L - 1, size=batch_size)          # anchor time in [0,L-2].

    # Future-goal index j > i with prob proportional to discount**(j-i).
    arange = np.arange(L)                                 # [L]
    future = arange[None, :] > i[:, None]                # [B, L]
    logp = (arange[None, :] - i[:, None]) * self._log_discount  # [B, L]
    logits = np.where(future, logp, -np.inf)
    # Gumbel-max categorical sample over the same distribution as flatten_fn.
    g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
    j = np.argmax(logits + g, axis=1)                    # [B]

    state = self._obs[traj, i, :self._obs_dim]           # [B, obs_dim]
    next_state = self._obs[traj, i + 1, :self._obs_dim]
    goal_state = self._obs[traj, j, :self._obs_dim]
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
