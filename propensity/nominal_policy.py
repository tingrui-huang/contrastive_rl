"""Sampleable nominal observational policy for bounded continuous actions.

The policy is a conditional mixture of diagonal Gaussians followed by the same
component-wise clip to ``[-1, 1]`` used by the PointMaze environment.  Training
uses the corresponding censored likelihood: interior actions contribute a
Gaussian density and exact boundary actions contribute the appropriate tail
mass.  This matters for the swamp datasets because the behavior collectors
produce genuine atoms at -1 and +1 by clipping noisy teacher actions.

The default decision context is the complete learner-visible pre-action
observation ``concat(state, commanded_goal)``.  Hidden swamp bits, audit labels,
future states, rewards, and hindsight goals never enter this module.

Downstream example::

    model = load_nominal_policy('artifacts/pointmaze_nominal/mdn_k5')
    x_prime = model.sample(observation, jax.random.PRNGKey(7))       # [2]
    draws = model.sample(observation, jax.random.PRNGKey(8), 32)    # [32, 2]
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


ACTION_LOW = -1.0
ACTION_HIGH = 1.0


@dataclasses.dataclass(frozen=True)
class NominalPolicySpec:
  state_dim: int
  goal_dim: int
  action_dim: int
  num_components: int = 5
  hidden_sizes: Tuple[int, ...] = (128, 128)
  conditioning: str = 'state_goal'
  min_scale: float = 0.02
  max_scale: float = 2.0
  loc_limit: float = 3.0
  boundary_tol: float = 1e-6
  action_low: float = ACTION_LOW
  action_high: float = ACTION_HIGH

  def __post_init__(self):
    if self.conditioning not in ('state_goal', 'state_only'):
      raise ValueError('conditioning must be state_goal or state_only')
    if self.state_dim <= 0 or self.goal_dim < 0 or self.action_dim <= 0:
      raise ValueError('invalid state/goal/action dimensions')
    if self.num_components <= 0 or not self.hidden_sizes:
      raise ValueError('the model needs at least one component and hidden layer')
    if not 0 < self.min_scale < self.max_scale:
      raise ValueError('require 0 < min_scale < max_scale')
    if not self.action_low < self.action_high:
      raise ValueError('invalid action bounds')

  @property
  def context_dim(self):
    return self.state_dim + (self.goal_dim if self.conditioning == 'state_goal'
                             else 0)

  def asdict(self):
    value = dataclasses.asdict(self)
    value['hidden_sizes'] = list(self.hidden_sizes)
    value['context_dim'] = self.context_dim
    return value


class MixtureParams(NamedTuple):
  logits: jnp.ndarray  # [B, K]
  loc: jnp.ndarray     # [B, K, A], before environment clipping
  scale: jnp.ndarray   # [B, K, A]


class PolicyNetwork(NamedTuple):
  init: Callable
  apply: Callable


def make_policy_network(spec: NominalPolicySpec) -> PolicyNetwork:
  """Build the conditional mixture parameter network."""
  k, a = spec.num_components, spec.action_dim

  def forward(context):
    h = context
    for width in spec.hidden_sizes:
      h = jax.nn.silu(hk.Linear(width)(h))
    out = hk.Linear(k * (1 + 2 * a))(h)
    logits, raw_loc, raw_scale = jnp.split(out, [k, k + k * a], axis=-1)
    loc = spec.loc_limit * jnp.tanh(jnp.reshape(raw_loc, (-1, k, a)))
    raw_scale = jnp.reshape(raw_scale, (-1, k, a))
    scale = spec.min_scale + (spec.max_scale - spec.min_scale) * jax.nn.sigmoid(
        raw_scale)
    return MixtureParams(logits=logits, loc=loc, scale=scale)

  transformed = hk.without_apply_rng(hk.transform(forward))
  return PolicyNetwork(init=transformed.init, apply=transformed.apply)


def censored_mixture_log_prob(mixture: MixtureParams, action,
                              spec: NominalPolicySpec):
  """Log likelihood in the clipped/censored ``[low, high]^A`` action space.

  The dominating measure is Lebesgue measure in each open interval plus point
  masses at the two bounds.  Therefore exact -1/+1 actions use Gaussian tail
  probabilities, while interior actions use their ordinary Gaussian density.
  Returns one joint log likelihood per action vector, shape ``[B]``.
  """
  action = jnp.asarray(action)[:, None, :]  # [B, 1, A]
  loc, scale = mixture.loc, mixture.scale
  z = (action - loc) / scale
  log_density = (-0.5 * jnp.square(z) - jnp.log(scale)
                 - 0.5 * jnp.log(2.0 * jnp.pi))
  z_low = (spec.action_low - loc) / scale
  z_high_survival = (loc - spec.action_high) / scale
  log_low_mass = jax.scipy.special.log_ndtr(z_low)
  log_high_mass = jax.scipy.special.log_ndtr(z_high_survival)
  at_low = action <= spec.action_low + spec.boundary_tol
  at_high = action >= spec.action_high - spec.boundary_tol
  per_dim = jnp.where(at_low, log_low_mass,
                      jnp.where(at_high, log_high_mass, log_density))
  per_component = jnp.sum(per_dim, axis=-1)
  log_mix = jax.nn.log_softmax(mixture.logits, axis=-1)
  return jax.scipy.special.logsumexp(log_mix + per_component, axis=-1)


def sample_censored_mixture(mixture: MixtureParams, key, num_samples,
                            spec: NominalPolicySpec):
  """Draw ``[B, num_samples, A]`` actions and apply environment clipping."""
  key_component, key_noise = jax.random.split(key)
  b, k = mixture.logits.shape
  indices = jax.random.categorical(
      key_component, mixture.logits, axis=-1, shape=(int(num_samples), b)).T
  one_hot = jax.nn.one_hot(indices, k)
  loc = jnp.einsum('bnk,bka->bna', one_hot, mixture.loc)
  scale = jnp.einsum('bnk,bka->bna', one_hot, mixture.scale)
  noise = jax.random.normal(key_noise, (b, int(num_samples), spec.action_dim))
  return jnp.clip(loc + scale * noise, spec.action_low, spec.action_high)


def _assemble_context(spec, state_or_observation, goal=None):
  x = jnp.asarray(state_or_observation, dtype=jnp.float32)
  if goal is None:
    if x.shape[-1] != spec.context_dim:
      need = ('the full state|commanded-goal observation'
              if spec.conditioning == 'state_goal' else 'the state')
      raise ValueError(f'expected {need} with width {spec.context_dim}, '
                       f'got {x.shape[-1]}')
    return x
  if spec.conditioning != 'state_goal':
    raise ValueError('this checkpoint is state_only; do not pass a goal')
  g = jnp.asarray(goal, dtype=jnp.float32)
  if x.shape[-1] != spec.state_dim or g.shape[-1] != spec.goal_dim:
    raise ValueError(f'expected state/goal widths {spec.state_dim}/'
                     f'{spec.goal_dim}, got {x.shape[-1]}/{g.shape[-1]}')
  if x.shape[:-1] != g.shape[:-1]:
    raise ValueError('state and goal leading shapes must match')
  return jnp.concatenate([x, g], axis=-1)


class NominalPolicy:
  """Loaded policy with JAX-friendly distribution, likelihood, and sample APIs."""

  def __init__(self, params, spec, context_mean, context_std, metadata=None):
    self.params = jax.tree_util.tree_map(jnp.asarray, params)
    self.spec = spec
    self.context_mean = jnp.asarray(context_mean, dtype=jnp.float32)
    self.context_std = jnp.asarray(context_std, dtype=jnp.float32)
    self.metadata = {} if metadata is None else metadata
    network = make_policy_network(spec)

    def distribution(params, context):
      normalized = (context - self.context_mean) / self.context_std
      return network.apply(params, normalized)

    self._distribution = jax.jit(distribution)
    self._log_prob = jax.jit(
        lambda params, context, action: censored_mixture_log_prob(
            distribution(params, context), action, spec))
    self._sample = jax.jit(
        lambda params, context, key, n: sample_censored_mixture(
            distribution(params, context), key, n, spec),
        static_argnums=(3,))

  def _flat_context(self, state_or_observation, goal=None):
    context = _assemble_context(self.spec, state_or_observation, goal)
    leading = context.shape[:-1]
    return jnp.reshape(context, (-1, self.spec.context_dim)), leading

  def distribution(self, state_or_observation, goal=None):
    context, leading = self._flat_context(state_or_observation, goal)
    value = self._distribution(self.params, context)
    k, a = self.spec.num_components, self.spec.action_dim
    return MixtureParams(
        logits=jnp.reshape(value.logits, leading + (k,)),
        loc=jnp.reshape(value.loc, leading + (k, a)),
        scale=jnp.reshape(value.scale, leading + (k, a)))

  def log_prob(self, state_or_observation, action, goal=None):
    context, leading = self._flat_context(state_or_observation, goal)
    action = jnp.asarray(action, dtype=jnp.float32)
    expected = leading + (self.spec.action_dim,)
    if action.shape != expected:
      raise ValueError(f'expected action shape {expected}, got {action.shape}')
    flat_action = jnp.reshape(action, (-1, self.spec.action_dim))
    return jnp.reshape(self._log_prob(self.params, context, flat_action), leading)

  def sample(self, state_or_observation, key, num_samples=1, goal=None):
    """Sample x_prime; returns ``[..., A]`` for one or ``[..., N, A]`` for N."""
    if int(num_samples) <= 0:
      raise ValueError('num_samples must be positive')
    context, leading = self._flat_context(state_or_observation, goal)
    samples = self._sample(self.params, context, key, int(num_samples))
    if int(num_samples) == 1:
      return jnp.reshape(samples[:, 0], leading + (self.spec.action_dim,))
    return jnp.reshape(samples,
                       leading + (int(num_samples), self.spec.action_dim))


def sample_observational_actions(model, state_or_observation, key,
                                 num_samples=1, goal=None):
  """Future-ETT-facing alias for ``NominalPolicy.sample``."""
  return model.sample(state_or_observation, key, num_samples, goal=goal)


def load_nominal_policy(run_dir, checkpoint='best.pkl'):
  with open(os.path.join(run_dir, 'config.json')) as f:
    metadata = json.load(f)
  with open(os.path.join(run_dir, checkpoint), 'rb') as f:
    payload = pickle.load(f)
  model_cfg = dict(metadata['model'])
  model_cfg.pop('context_dim', None)
  model_cfg['hidden_sizes'] = tuple(model_cfg['hidden_sizes'])
  spec = NominalPolicySpec(**model_cfg)
  normalizer = metadata['context_normalization']
  return NominalPolicy(payload['params'], spec, normalizer['mean'],
                       normalizer['std'], metadata)
