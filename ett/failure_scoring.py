"""Scope-B supervised feasibility scorers, not offline-compatible objectives.

Only visible F4 and commanded goal enter the networks. Simulator death labels
are privileged supervised targets in the separate experiment driver.
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from scripts.diagnose_f4_death_observability import _make_probe

SCOPE = 'B_supervised_privileged_label_feasibility_only'
FEATURES = ('xy', 'motion', 'f4')


def visible_features(state, goal, kind):
  """Newest-first F4; motion = [p0-p1, p1-p2, p2-p3].

  Strict visible-array API rejects dictionaries or augmented hidden inputs.
  The same commanded-goal block is retained in every comparison.
  """
  state, goal = jnp.asarray(state), jnp.asarray(goal)
  if state.shape[-1] != 8 or goal.shape[-1] != 8:
    raise ValueError('expected visible F4 state and commanded goal of width 8')
  goal = jnp.broadcast_to(goal, state.shape)
  if kind == 'xy':
    value = state[..., :2]
  elif kind == 'motion':
    value = state[..., :6] - state[..., 2:8]
  elif kind == 'f4':
    value = state
  else:
    raise ValueError(f'unknown visible feature family: {kind}')
  return jnp.concatenate([value, goal], axis=-1)


def scorer_episode_split(episode_count, selection_seed=9201):
  """Preserve source seed-0 10% holdout; split its complement 90/10.

  The holdout is unused for scorer selection, but NOT untouched by upstream
  experiments. No row-level splitting or outcome stratification is used.
  """
  perm = np.random.default_rng(0).permutation(episode_count)
  test = np.sort(perm[:round(.1*episode_count)])
  pool = np.sort(perm[round(.1*episode_count):])
  shuffled = np.random.default_rng(selection_seed).permutation(pool)
  count = round(.1*len(pool))
  return {'train': np.sort(shuffled[count:]),
          'selection': np.sort(shuffled[:count]), 'test': test}


def labels_from_death_rows(episode_id, timestep, death_row_by_episode):
  """The fatal transition at d-1 is alive; observation d has death age 0."""
  row = np.asarray(death_row_by_episode)[episode_id]
  dead = (row >= 1) & (timestep >= row)
  return {'dead_when_observed': dead,
          'death_age': np.where(dead, timestep-row, -1),
          'dies_during_transition': (row >= 1) & (timestep == row-1)}


class FailureScorer:
  """Frozen differentiable scope-B score accepting [...,8] visible states.

  Sigmoid scores are assessed at observed prevalence; they are not calibrated
  deployment probabilities for optimized synthetic states.
  """
  def __init__(self, checkpoint):
    if checkpoint.get('scope') != SCOPE:
      raise ValueError('checkpoint must declare privileged feasibility scope')
    self.checkpoint = checkpoint
    self.network = _make_probe(tuple(checkpoint['hidden_sizes']))
    self.params = jax.tree.map(jnp.asarray, checkpoint['params'])

  def logits(self, state, goal=None):
    state = jnp.asarray(state)
    if goal is None:
      goal = jnp.asarray(self.checkpoint['fixed_commanded_goal'])
    value = visible_features(state, goal, self.checkpoint['feature_name'])
    normalized = (value-jnp.asarray(self.checkpoint['mean']))/jnp.asarray(self.checkpoint['std'])
    return self.network.apply(self.params, normalized.reshape(-1, value.shape[-1])).reshape(state.shape[:-1])

  def score(self, state, goal=None):
    return jax.nn.sigmoid(self.logits(state, goal))


def load_failure_scorer(path):
  with open(path, 'rb') as stream:
    return FailureScorer(pickle.load(stream))
