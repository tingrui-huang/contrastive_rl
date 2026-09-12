"""Single-step failure-matching prototype with an anchored action-change bound.

Only newest XY is generated. The base law is the existing diagonal mixed
measure, sampled at (s, x_prime, x_prime). The residual is zero on the diagonal.
This is neither an identified counterfactual kernel nor all-pairs Lipschitzness.
"""
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import (
    DiagonalTransitionModel, _assemble_inputs, _check_diagonal, _load_spec, _project_samples,
    sample_displacement)


def make_residual_network():
  def forward(context):
    hidden = hk.nets.MLP([64, 64], activation=jax.nn.silu,
                         activate_final=True)(context)
    # Start every arm with exactly the same diagonal sampling law.
    return hk.Linear(2, w_init=hk.initializers.Constant(0.0),
                     b_init=hk.initializers.Constant(0.0))(hidden)
  return hk.without_apply_rng(hk.transform(forward))


def gaussian_kernel(left, right, bandwidths):
  """Gaussian mixture kernel; last two input axes are samples and features."""
  distance2 = jnp.sum(jnp.square(
      left[..., :, None, :] - right[..., None, :, :]), axis=-1)
  return jnp.mean(jnp.exp(-distance2[..., None] /
                         (2 * jnp.square(jnp.asarray(bandwidths)))), axis=-1)


def conditional_mmd2(generated, reference, mean, std, bandwidths):
  """Biased V-statistic, separately per fixed (s, x), including self pairs.

  generated: [B, K, 8]; reference: [R, 8]. Returns [B]. There are no
  generated cross-context pairs. All three empirical kernel terms are used.
  """
  if generated.ndim != 3 or reference.ndim != 2:
    raise ValueError('expected generated [B,K,D] and reference [R,D]')
  if generated.shape[-1] != reference.shape[-1]:
    raise ValueError('generated and reference feature widths differ')
  generated = (generated - mean) / std
  reference = (reference - mean) / std
  return (gaussian_kernel(generated, generated, bandwidths).mean(axis=(-2, -1))
          + gaussian_kernel(reference, reference, bandwidths).mean()
          - 2 * gaussian_kernel(generated, reference, bandwidths).mean(
              axis=(-2, -1)))


class AnchoredTransition:
  """JAX sampler retaining the diagonal model's leading-shape conventions."""

  def __init__(self, diagonal, residual_params, action_bound):
    if not np.isfinite(action_bound) or action_bound < 0:
      raise ValueError('action_bound must be finite and nonnegative')
    self.diagonal = diagonal
    self.action_bound = float(action_bound)
    self.residual = make_residual_network()
    self.params = {'diagonal': diagonal.params, 'residual': residual_params}
    self._sample_jit = jax.jit(self.sample_flat, static_argnums=(6,))

  @classmethod
  def initialize(cls, diagonal, key, action_bound):
    params = make_residual_network().init(key, jnp.zeros((1, 24)))
    return cls(diagonal, params, action_bound)

  def sample_flat(self, params, state, action, observational_action, goal,
                  key, num_samples):
    """Differentiable internal API with flat input batches.

    Categorical and Bernoulli choices have no pathwise derivative. MMD uses
    conditional pathwise gradients through selected moving locations/scales
    and the residual, without a score-function or straight-through estimator.
    Thus this is a biased partial gradient of expected MMD: no direct MMD
    gradients reach gate/logit outputs. Diagonal NLL trains all mixture heads.
    Shared trunk updates can still change gate probabilities indirectly.
    Geometry comparisons/fallbacks are hard branches, differentiated only
    through the selected branch. Atoms can move off-diagonal via the residual.
    """
    base = self.diagonal
    context = jnp.concatenate([state, observational_action,
                               observational_action, goal], axis=-1)
    distribution = base._distribution(params['diagonal'], context)
    delta, atom = sample_displacement(distribution, key, num_samples,
                                      base.delta_mean, base.delta_std, base.spec)
    anchor, base_detail = _project_samples(state, delta, base.spec)
    expanded = jnp.broadcast_to(
        ((jnp.concatenate([state, action, observational_action, goal], axis=-1)
          - base.context_mean) / base.context_std)[:, None, :],
        (len(state), num_samples, 20))
    difference = action - observational_action
    residual_context = jnp.concatenate([
        expanded, jnp.broadcast_to(difference[:, None, :],
                                   (len(state), num_samples, 2)),
        anchor[..., :2] - state[:, None, :2]], axis=-1)
    raw = self.residual.apply(params['residual'], residual_context.reshape(-1, 24))
    direction = jnp.tanh(raw.reshape(len(state), num_samples, 2)) / jnp.sqrt(2.0)
    # Euclidean norms, raw maze XY and environment action units (no scaling).
    radius = self.action_bound * jnp.linalg.norm(difference, axis=-1)
    # Leave a conservative float32 margin before adding the anchor. Otherwise
    # saturated tanh outputs can spuriously trip the strict final bound check.
    margin = 8 * jnp.finfo(anchor.dtype).eps * (
        1 + jnp.max(jnp.abs(anchor[..., :2]), axis=-1) + radius[:, None])
    safe_radius = jnp.maximum(radius[:, None] - margin, 0.)
    proposed_xy = anchor[..., :2] + safe_radius[..., None] * direction
    candidate, detail = _project_samples(
        state, proposed_xy - state[:, None, :2], base.spec)
    distance = jnp.linalg.norm(candidate[..., :2] - anchor[..., :2], axis=-1)
    # Nonconvex collision projection is not nonexpansive. Reject any candidate
    # that breaks the final bound, returning the feasible diagonal anchor.
    # The explicit identical-action branch also avoids roundoff on the atom.
    fallback = (distance > radius[:, None]) | (radius[:, None] == 0)
    output = jnp.where(fallback[..., None], anchor, candidate)
    return output, {
        'anchor_xy': anchor[..., :2], 'radius': jnp.broadcast_to(radius[:, None], atom.shape),
        'stationary_atom': atom,
        'bound_fallback': (distance > radius[:, None]) & (radius[:, None] > 0),
        'base_corrected': jnp.any(base_detail['raw_position'] != anchor[..., :2], axis=-1),
        'candidate_corrected': jnp.any(jnp.abs(proposed_xy - candidate[..., :2]) > 1e-6, axis=-1),
        'candidate_blocked': detail['blocked_endpoint_before_projection'],
        'emitted_change': jnp.linalg.norm(output[..., :2] - anchor[..., :2], axis=-1),
    }

  def sample_with_diagnostics(self, state_or_observation, action,
                              observational_action, key, num_samples=1, goal=None):
    if int(num_samples) <= 0:
      raise ValueError('num_samples must be positive')
    state, context = _assemble_inputs(self.diagonal.spec, state_or_observation,
                                     action, observational_action, goal)
    leading = state.shape[:-1]
    context = context.reshape(-1, 20)
    output, detail = self._sample_jit(
        self.params, state.reshape(-1, 8), context[:, 8:10], context[:, 10:12],
        context[:, 12:], key, int(num_samples))
    def restore(value):
      if num_samples == 1:
        return value[:, 0].reshape(leading + value.shape[2:])
      return value.reshape(leading + (num_samples,) + value.shape[2:])
    return restore(output), {name: restore(value) for name, value in detail.items()}

  def sample(self, state_or_observation, action, observational_action, key,
             num_samples=1, goal=None):
    return self.sample_with_diagnostics(state_or_observation, action,
                                       observational_action, key, num_samples, goal)[0]

  def log_prob(self, state_or_observation, action, observational_action,
               next_state, goal=None):
    """Existing diagonal mixed likelihood; off-diagonal densities unsupported.

    Geometry and the residual define a sampleable off-diagonal law, not a
    tractable density. Reject off-diagonal queries rather than mislabel NLL.
    """
    _check_diagonal(action, observational_action, 0.0)
    state, context, leading = self.diagonal._flat_inputs(
        state_or_observation, action, observational_action, goal)
    next_state = jnp.asarray(next_state, dtype=jnp.float32)
    if next_state.shape != leading + (8,):
      raise ValueError('next_state must match the input leading shape and F4 width')
    delta = next_state.reshape(-1, 8)[:, :2] - state[:, :2]
    return self.diagonal._log_prob(self.params['diagonal'], context, delta).reshape(leading)

  def marginal_flat(self, params, nominal, state, action, goal, key, count):
    """[B,K,8] samples: K independent x_prime values for each fixed (s,x)."""
    expert_key, transition_key = jax.random.split(key)
    xp = nominal.sample(state, expert_key, count, goal=goal).reshape(len(state), count, 2)
    repeat = lambda value: jnp.repeat(value, count, axis=0)
    samples, detail = self.sample_flat(
        params, repeat(state), repeat(action), xp.reshape(-1, 2), repeat(goal),
        transition_key, 1)
    return samples.reshape(len(state), count, 8), {
        name: value.reshape((len(state), count) + value.shape[2:])
        for name, value in detail.items()}

  def sample_marginal(self, nominal, state_or_observation, action, key,
                      num_samples=8, goal=None):
    if int(num_samples) <= 0:
      raise ValueError('num_samples must be positive')
    state, context = _assemble_inputs(self.diagonal.spec, state_or_observation,
                                     action, action, goal)
    leading = state.shape[:-1]
    samples, _ = self.marginal_flat(
        self.params, nominal, state.reshape(-1, 8), jnp.asarray(action).reshape(-1, 2),
        context.reshape(-1, 20)[:, 12:], key, int(num_samples))
    return samples.reshape(leading + (num_samples, 8))


def save_anchored(path, model, step):
  """Self-contained local checkpoint; never embeds nominal or actor weights."""
  base = model.diagonal
  payload = {
      'params': jax.tree_util.tree_map(np.asarray, model.params), 'step': step,
      'action_bound': model.action_bound, 'diagonal_metadata': base.metadata,
  }
  with open(path, 'wb') as target:
    pickle.dump(payload, target)


def load_anchored(path):
  with open(path, 'rb') as source:
    payload = pickle.load(source)
  metadata = payload['diagonal_metadata']
  context, delta = metadata['context_normalization'], metadata['delta_normalization']
  base = DiagonalTransitionModel(
      payload['params']['diagonal'], _load_spec(metadata), context['mean'],
      context['std'], delta['mean'], delta['std'], metadata)
  return AnchoredTransition(base,
                           jax.tree_util.tree_map(jnp.asarray, payload['params']['residual']),
                           payload['action_bound'])
