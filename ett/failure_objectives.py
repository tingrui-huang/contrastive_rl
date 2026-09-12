"""Read-only objective diagnostics; neither objective is a validated failure metric."""
import jax.numpy as jnp

from ett.anchored_transition import gaussian_kernel


def mmd_components(generated, references, mean, std, bandwidths):
  """Decompose the original conditional V-statistic without pooling contexts.

  Inputs [B,K,D] and [R,D]; all outputs [B]. Reference-reference pairs include
  self pairs because the empirical reference distribution is the fixed target.
  `mmd_generated_u` only removes generated self pairs: it is unbiased over
  independent generated draws against that empirical target, not an estimate
  against an unknown population of failures. It may be negative in a sample.
  """
  if generated.ndim != 3 or references.ndim != 2:
    raise ValueError('expected generated [B,K,D] and references [R,D]')
  if generated.shape[-1] != references.shape[-1] or min(generated.shape[:2]) < 1 or not len(references):
    raise ValueError('nonempty samples and matching feature dimensions required')
  y, f = (generated - mean) / std, (references - mean) / std
  gg = gaussian_kernel(y, y, bandwidths).mean(axis=(-2, -1))
  rr = jnp.broadcast_to(gaussian_kernel(f, f, bandwidths).mean(), gg.shape)
  gr = gaussian_kernel(y, f, bandwidths).mean(axis=(-2, -1))
  result = {'gg': gg, 'rr': rr, 'gr': gr, 'mmd': gg + rr - 2 * gr}
  count = generated.shape[1]
  if count > 1:
    gg_u = (count * gg - 1.) / (count - 1)
    result['mmd_generated_u'] = gg_u + rr - 2 * gr
    result['generated_self_pair_correction'] = gg - gg_u
  return result


def nearest_reference(generated, references, mean, std):
  """Hard full-F4 distance and its decomposition at the SAME chosen reference.

  Arbitrary leading sample axes are supported. Ties select the first reference
  in saved bank order. The cost is continuous and piecewise quadratic; at a tie
  this implementation differentiates the selected branch, not a soft minimum
  or an average of tie gradients. The selected index/gradient can jump across
  Voronoi boundaries. `history_floor` minimizes old-frame distance separately
  and is the unconstrained lower bound with only newest XY editable.
  """
  if references.ndim != 2 or references.shape[-1] != 8 or generated.shape[-1] != 8:
    raise ValueError('nearest-reference diagnostic requires full F4 width 8')
  if not len(references):
    raise ValueError('reference bank must be nonempty')
  difference = ((generated[..., None, :] - references) / std)
  # The common normalization mean cancels exactly in Euclidean differences.
  del mean
  newest = jnp.sum(jnp.square(difference[..., :2]), axis=-1)
  history = jnp.sum(jnp.square(difference[..., 2:]), axis=-1)
  distances = newest + history
  index = jnp.argmin(distances, axis=-1)
  choose = lambda value: jnp.take_along_axis(value, index[..., None], axis=-1)[..., 0]
  distance = choose(distances)
  return {'distance2': distance, 'newest_distance2': choose(newest),
          'history_distance2': choose(history), 'index': index,
          'history_floor': jnp.min(history, axis=-1),
          'history_nearest_index': jnp.argmin(history, axis=-1),
          'exact_tie_count': jnp.sum(distances == distance[..., None], axis=-1)}


def conditional_set_cost(generated, references, mean, std):
  """Return one mean nearest-set cost per fixed (s,x) group [B,K,8]."""
  if generated.ndim != 3 or min(generated.shape[:2]) < 1:
    raise ValueError('expected nonempty generated [B,K,8]')
  return nearest_reference(generated, references, mean, std)['distance2'].mean(axis=1)


def shift_with_newest(state, newest):
  """[B,8] state and [B,K,2] newest XY -> exact next [B,K,8] history."""
  if state.ndim != 2 or state.shape[-1] != 8 or newest.ndim != 3 or newest.shape[-1] != 2:
    raise ValueError('expected state [B,8] and newest [B,K,2]')
  if state.shape[0] != newest.shape[0]:
    raise ValueError('batch sizes differ')
  return jnp.concatenate([newest, jnp.broadcast_to(state[:, None, :6], newest.shape[:2]+(6,))], -1)
