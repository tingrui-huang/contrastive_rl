"""Sampleable diagonal transition model for frame-stacked PointMaze.

The density has an exact zero-displacement atom plus a conditional mixture of
diagonal Gaussian densities for moving transitions. Only the newest XY frame
is stochastic. Older frames shift deterministically.

The public API accepts both an intervened action and an observational action,
but checkpoints trained by this module reject off-diagonal calls. Their data
contain only ``action == observational_action`` and cannot identify how the two
arguments should behave separately.
"""
import dataclasses
import json
import os
import pickle
from typing import Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


POINTMAZE_WALLS = np.array([
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
], dtype=np.int32)


@dataclasses.dataclass(frozen=True)
class DiagonalTransitionSpec:
  state_dim: int = 8
  goal_dim: int = 8
  action_dim: int = 2
  delta_dim: int = 2
  num_components: int = 3
  hidden_sizes: Tuple[int, ...] = (128, 128)
  min_scale: float = 0.01
  max_scale: float = 3.0
  loc_limit: float = 8.0
  stationary_tolerance: float = 1e-7
  diagonal_tolerance: float = 1e-6
  max_displacement_per_coordinate: float = 1.0
  state_low: Tuple[float, float] = (0.0, 0.0)
  state_high: Tuple[float, float] = (9.0, 5.0)

  def __post_init__(self):
    if (self.state_dim != 8 or self.goal_dim != 8
        or self.action_dim != 2 or self.delta_dim != 2):
      raise ValueError('the first baseline is specific to PointMaze F4')
    if self.num_components <= 0 or not self.hidden_sizes:
      raise ValueError('at least one component and hidden layer are required')
    if not 0.0 < self.min_scale < self.max_scale:
      raise ValueError('require 0 < min_scale < max_scale')

  @property
  def context_dim(self):
    return self.state_dim + 2 * self.action_dim + self.goal_dim

  def asdict(self):
    value = dataclasses.asdict(self)
    value['hidden_sizes'] = list(self.hidden_sizes)
    value['state_low'] = list(self.state_low)
    value['state_high'] = list(self.state_high)
    value['context_dim'] = self.context_dim
    return value


class TransitionDistribution(NamedTuple):
  stationary_logit: jnp.ndarray
  mixture_logits: jnp.ndarray
  loc: jnp.ndarray
  scale: jnp.ndarray


class TransitionNetwork(NamedTuple):
  init: Callable
  apply: Callable


def make_transition_network(spec):
  """Build the zero-inflated conditional displacement mixture."""
  k, d = spec.num_components, spec.delta_dim

  def forward(context):
    hidden = context
    for width in spec.hidden_sizes:
      hidden = jax.nn.silu(hk.Linear(width)(hidden))
    output = hk.Linear(1 + k + 2 * k * d)(hidden)
    stop_logits = 1
    stop_loc = stop_logits + k
    stop_scale = stop_loc + k * d
    stationary_logit = output[:, 0]
    mixture_logits = output[:, stop_logits:stop_loc]
    raw_loc = jnp.reshape(output[:, stop_loc:stop_scale], (-1, k, d))
    raw_scale = jnp.reshape(output[:, stop_scale:], (-1, k, d))
    loc = spec.loc_limit * jnp.tanh(raw_loc)
    scale = spec.min_scale + (spec.max_scale - spec.min_scale) * jax.nn.sigmoid(
        raw_scale)
    return TransitionDistribution(stationary_logit, mixture_logits, loc, scale)

  transformed = hk.without_apply_rng(hk.transform(forward))
  return TransitionNetwork(transformed.init, transformed.apply)


def make_deterministic_network(spec):
  """Build the shared-metric deterministic displacement baseline."""

  def forward(context):
    hidden = context
    for width in spec.hidden_sizes:
      hidden = jax.nn.silu(hk.Linear(width)(hidden))
    return hk.Linear(spec.delta_dim)(hidden)

  transformed = hk.without_apply_rng(hk.transform(forward))
  return TransitionNetwork(transformed.init, transformed.apply)


def diagonal_transition_log_prob(distribution, delta_xy, delta_mean,
                                 delta_std, spec):
  """Mixed-measure log likelihood in nats per two-dimensional transition.

  Exact zero displacement uses the Bernoulli atom. Every nonzero displacement
  uses the moving Gaussian-mixture density with respect to maze-unit area.
  """
  delta_xy = jnp.asarray(delta_xy, dtype=jnp.float32)
  delta_mean = jnp.asarray(delta_mean, dtype=jnp.float32)
  delta_std = jnp.asarray(delta_std, dtype=jnp.float32)
  normalized = (delta_xy - delta_mean) / delta_std
  z = (normalized[:, None, :] - distribution.loc) / distribution.scale
  component_log_prob = jnp.sum(
      -0.5 * jnp.square(z) - jnp.log(distribution.scale)
      - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)
  moving_log_prob = jax.scipy.special.logsumexp(
      jax.nn.log_softmax(distribution.mixture_logits, axis=-1)
      + component_log_prob, axis=-1)
  moving_log_prob -= jnp.sum(jnp.log(delta_std))
  stationary = jnp.max(jnp.abs(delta_xy), axis=-1) <= spec.stationary_tolerance
  atom_log_prob = jax.nn.log_sigmoid(distribution.stationary_logit)
  moving_log_prob += jax.nn.log_sigmoid(-distribution.stationary_logit)
  return jnp.where(stationary, atom_log_prob, moving_log_prob)


def sample_displacement(distribution, key, num_samples, delta_mean, delta_std,
                        spec):
  """Draw raw XY displacements before maze-geometry postprocessing."""
  key_stationary, key_component, key_noise = jax.random.split(key, 3)
  batch_size, num_components = distribution.mixture_logits.shape
  shape = (int(num_samples), batch_size)
  stationary = jax.random.uniform(key_stationary, shape).T < jax.nn.sigmoid(
      distribution.stationary_logit)[:, None]
  component = jax.random.categorical(
      key_component, distribution.mixture_logits, axis=-1, shape=shape).T
  one_hot = jax.nn.one_hot(component, num_components)
  loc = jnp.einsum('bnk,bkd->bnd', one_hot, distribution.loc)
  scale = jnp.einsum('bnk,bkd->bnd', one_hot, distribution.scale)
  noise = jax.random.normal(
      key_noise, (batch_size, int(num_samples), spec.delta_dim))
  normalized = loc + scale * noise
  delta = normalized * jnp.asarray(delta_std) + jnp.asarray(delta_mean)
  return jnp.where(stationary[..., None], 0.0, delta), stationary


def _assemble_inputs(spec, state_or_observation, action, observational_action,
                     goal):
  state_or_observation = jnp.asarray(state_or_observation, dtype=jnp.float32)
  if goal is None:
    if state_or_observation.shape[-1] != spec.state_dim + spec.goal_dim:
      raise ValueError('expected a complete state|commanded-goal observation')
    state = state_or_observation[..., :spec.state_dim]
    goal = state_or_observation[..., spec.state_dim:]
  else:
    state = state_or_observation
    goal = jnp.asarray(goal, dtype=jnp.float32)
    if state.shape[-1] != spec.state_dim or goal.shape[-1] != spec.goal_dim:
      raise ValueError('state and goal widths do not match the checkpoint')
    if state.shape[:-1] != goal.shape[:-1]:
      raise ValueError('state and goal leading shapes must match')
  action = jnp.asarray(action, dtype=jnp.float32)
  observational_action = jnp.asarray(observational_action, dtype=jnp.float32)
  expected = state.shape[:-1] + (spec.action_dim,)
  if action.shape != expected or observational_action.shape != expected:
    raise ValueError(f'expected both action shapes {expected}')
  context = jnp.concatenate(
      [state, action, observational_action, goal], axis=-1)
  return state, context


def _check_diagonal(action, observational_action, tolerance):
  action = np.asarray(action)
  observational_action = np.asarray(observational_action)
  if not np.allclose(action, observational_action, rtol=0.0, atol=tolerance):
    raise ValueError(
        'this checkpoint was trained only on action == observational_action; '
        'off-diagonal transition calls are unsupported')


def _project_samples(state, raw_delta, spec):
  """Apply known one-step bounds and endpoint geometry to sampled positions."""
  max_delta = spec.max_displacement_per_coordinate
  capped_delta = jnp.clip(raw_delta, -max_delta, max_delta)
  current_xy = state[:, None, :2]
  raw_position = current_xy + raw_delta
  bounded_position = jnp.clip(
      current_xy + capped_delta, jnp.asarray(spec.state_low),
      jnp.asarray(spec.state_high))
  cell = jnp.floor(bounded_position).astype(jnp.int32)
  cell_x = jnp.clip(cell[..., 0], 0, POINTMAZE_WALLS.shape[0] - 1)
  cell_y = jnp.clip(cell[..., 1], 0, POINTMAZE_WALLS.shape[1] - 1)
  blocked = jnp.asarray(POINTMAZE_WALLS)[cell_x, cell_y] == 1
  final_position = jnp.where(blocked[..., None], current_xy, bounded_position)
  old_frames = jnp.broadcast_to(
      state[:, None, :6], final_position.shape[:-1] + (6,))
  next_state = jnp.concatenate([final_position, old_frames], axis=-1)
  return next_state, {
      'raw_delta': raw_delta,
      'capped_delta': capped_delta,
      'raw_position': raw_position,
      'bounded_position': bounded_position,
      'blocked_endpoint_before_projection': blocked,
      'final_position': final_position,
  }


class DiagonalTransitionModel:
  """Loaded stochastic model with diagonal-only likelihood and sampling APIs."""

  def __init__(self, params, spec, context_mean, context_std, delta_mean,
               delta_std, metadata=None):
    self.params = jax.tree_util.tree_map(jnp.asarray, params)
    self.spec = spec
    self.context_mean = jnp.asarray(context_mean, dtype=jnp.float32)
    self.context_std = jnp.asarray(context_std, dtype=jnp.float32)
    self.delta_mean = jnp.asarray(delta_mean, dtype=jnp.float32)
    self.delta_std = jnp.asarray(delta_std, dtype=jnp.float32)
    self.metadata = {} if metadata is None else metadata
    network = make_transition_network(spec)

    def distribution(params, context):
      normalized = (context - self.context_mean) / self.context_std
      return network.apply(params, normalized)

    self._distribution = jax.jit(distribution)
    self._log_prob = jax.jit(
        lambda params, context, delta: diagonal_transition_log_prob(
            distribution(params, context), delta, self.delta_mean,
            self.delta_std, spec))
    self._sample_delta = jax.jit(
        lambda params, context, key, count: sample_displacement(
            distribution(params, context), key, count, self.delta_mean,
            self.delta_std, spec), static_argnums=(3,))

  def _flat_inputs(self, state_or_observation, action, observational_action,
                   goal):
    _check_diagonal(action, observational_action,
                    self.spec.diagonal_tolerance)
    state, context = _assemble_inputs(
        self.spec, state_or_observation, action, observational_action, goal)
    leading = state.shape[:-1]
    return (jnp.reshape(state, (-1, self.spec.state_dim)),
            jnp.reshape(context, (-1, self.spec.context_dim)), leading)

  def distribution(self, state_or_observation, action, observational_action,
                   goal=None):
    _, context, leading = self._flat_inputs(
        state_or_observation, action, observational_action, goal)
    value = self._distribution(self.params, context)
    k, d = self.spec.num_components, self.spec.delta_dim
    return TransitionDistribution(
        jnp.reshape(value.stationary_logit, leading),
        jnp.reshape(value.mixture_logits, leading + (k,)),
        jnp.reshape(value.loc, leading + (k, d)),
        jnp.reshape(value.scale, leading + (k, d)))

  def stationary_probability(self, state_or_observation, action,
                             observational_action, goal=None):
    value = self.distribution(
        state_or_observation, action, observational_action, goal)
    return jax.nn.sigmoid(value.stationary_logit)

  def log_prob(self, state_or_observation, action, observational_action,
               next_state, goal=None):
    state, context, leading = self._flat_inputs(
        state_or_observation, action, observational_action, goal)
    next_state = jnp.asarray(next_state, dtype=jnp.float32)
    expected = leading + (self.spec.state_dim,)
    if next_state.shape != expected:
      raise ValueError(f'expected next_state shape {expected}')
    flat_next = jnp.reshape(next_state, (-1, self.spec.state_dim))
    delta = flat_next[:, :2] - state[:, :2]
    return jnp.reshape(self._log_prob(self.params, context, delta), leading)

  def sample_with_diagnostics(self, state_or_observation, action,
                              observational_action, key, num_samples=1,
                              goal=None):
    if int(num_samples) <= 0:
      raise ValueError('num_samples must be positive')
    state, context, leading = self._flat_inputs(
        state_or_observation, action, observational_action, goal)
    raw_delta, stationary = self._sample_delta(
        self.params, context, key, int(num_samples))
    next_state, diagnostics = _project_samples(state, raw_delta, self.spec)
    count = int(num_samples)
    if count == 1:
      output = jnp.reshape(next_state[:, 0], leading + (self.spec.state_dim,))
      diagnostics = {
          name: jnp.reshape(value[:, 0], leading + value.shape[2:])
          for name, value in diagnostics.items()}
      diagnostics['stationary_atom'] = jnp.reshape(stationary[:, 0], leading)
    else:
      output = jnp.reshape(
          next_state, leading + (count, self.spec.state_dim))
      diagnostics = {
          name: jnp.reshape(value, leading + (count,) + value.shape[2:])
          for name, value in diagnostics.items()}
      diagnostics['stationary_atom'] = jnp.reshape(
          stationary, leading + (count,))
    return output, diagnostics

  def sample(self, state_or_observation, action, observational_action, key,
             num_samples=1, goal=None):
    """Sample next F4 states; off-diagonal calls raise ``ValueError``."""
    return self.sample_with_diagnostics(
        state_or_observation, action, observational_action, key,
        num_samples=num_samples, goal=goal)[0]


class DeterministicTransitionModel:
  """Deterministic comparison model with the same diagonal input contract."""

  def __init__(self, params, spec, context_mean, context_std, delta_mean,
               delta_std, metadata=None):
    self.params = jax.tree_util.tree_map(jnp.asarray, params)
    self.spec = spec
    self.context_mean = jnp.asarray(context_mean, dtype=jnp.float32)
    self.context_std = jnp.asarray(context_std, dtype=jnp.float32)
    self.delta_mean = jnp.asarray(delta_mean, dtype=jnp.float32)
    self.delta_std = jnp.asarray(delta_std, dtype=jnp.float32)
    self.metadata = {} if metadata is None else metadata
    network = make_deterministic_network(spec)
    self._predict = jax.jit(lambda params, context: (
        network.apply(params, (context - self.context_mean) / self.context_std)
        * self.delta_std + self.delta_mean))

  def predict_delta(self, state_or_observation, action, observational_action,
                    goal=None):
    _check_diagonal(action, observational_action,
                    self.spec.diagonal_tolerance)
    state, context = _assemble_inputs(
        self.spec, state_or_observation, action, observational_action, goal)
    leading = state.shape[:-1]
    flat = jnp.reshape(context, (-1, self.spec.context_dim))
    return jnp.reshape(self._predict(self.params, flat), leading + (2,))

  def sample(self, state_or_observation, action, observational_action, key,
             num_samples=1, goal=None):
    if int(num_samples) <= 0:
      raise ValueError('num_samples must be positive')
    del key
    _check_diagonal(action, observational_action,
                    self.spec.diagonal_tolerance)
    state, context = _assemble_inputs(
        self.spec, state_or_observation, action, observational_action, goal)
    leading = state.shape[:-1]
    flat_state = jnp.reshape(state, (-1, self.spec.state_dim))
    flat_context = jnp.reshape(context, (-1, self.spec.context_dim))
    delta = self._predict(self.params, flat_context)[:, None, :]
    delta = jnp.broadcast_to(delta, (len(flat_state), int(num_samples), 2))
    next_state, _ = _project_samples(flat_state, delta, self.spec)
    if int(num_samples) == 1:
      return jnp.reshape(next_state[:, 0], leading + (self.spec.state_dim,))
    return jnp.reshape(
        next_state, leading + (int(num_samples), self.spec.state_dim))


def _load_spec(metadata):
  model_config = dict(metadata['model'])
  model_config.pop('context_dim', None)
  for name in ('hidden_sizes', 'state_low', 'state_high'):
    model_config[name] = tuple(model_config[name])
  return DiagonalTransitionSpec(**model_config)


def load_diagonal_transition(run_dir, checkpoint='best.pkl'):
  with open(os.path.join(run_dir, 'config.json')) as source:
    metadata = json.load(source)
  with open(os.path.join(run_dir, checkpoint), 'rb') as source:
    payload = pickle.load(source)
  spec = _load_spec(metadata)
  context = metadata['context_normalization']
  delta = metadata['delta_normalization']
  model_type = metadata['model_type']
  if model_type == 'zero_inflated_mixture':
    cls = DiagonalTransitionModel
  elif model_type == 'deterministic_mse':
    cls = DeterministicTransitionModel
  else:
    raise ValueError(f'unsupported checkpoint model type: {model_type}')
  return cls(payload['params'], spec, context['mean'], context['std'],
             delta['mean'], delta['std'], metadata)
