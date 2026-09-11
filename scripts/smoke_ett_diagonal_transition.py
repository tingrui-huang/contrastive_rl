"""Fast alignment, gradient, and sampling smoke for diagonal PointMaze ETT."""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ett.dataset import ExpertTransitionDataset
from ett.diagonal_transition import (
    DiagonalTransitionModel, DiagonalTransitionSpec,
    diagonal_transition_log_prob, make_transition_network)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--out-dir', default='artifacts/ett_diagonal_smoke')
  args = parser.parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)

  dataset = ExpertTransitionDataset(args.dataset)
  train = dataset.arrays('train')
  context, delta = train.context, train.delta_xy
  context_mean, context_std = context.mean(0), context.std(0)
  context_std = np.where(context_std < 1e-6, 1.0, context_std)
  moving = np.max(np.abs(delta), axis=-1) > 1e-7
  delta_mean, delta_std = delta[moving].mean(0), delta[moving].std(0)
  delta_std = np.where(delta_std < 1e-6, 1.0, delta_std)

  spec = DiagonalTransitionSpec(num_components=2, hidden_sizes=(32, 32))
  network = make_transition_network(spec)
  normalized_context = ((context[:4096] - context_mean) / context_std).astype(
      np.float32)
  target = delta[:4096]
  params = network.init(jax.random.PRNGKey(31), jnp.asarray(
      normalized_context[:2]))
  optimizer = optax.adam(1e-3)
  optimizer_state = optimizer.init(params)

  def objective(parameters):
    distribution = network.apply(parameters, jnp.asarray(normalized_context))
    return -jnp.mean(diagonal_transition_log_prob(
        distribution, jnp.asarray(target), delta_mean, delta_std, spec))

  @jax.jit
  def update(parameters, state):
    loss, gradient = jax.value_and_grad(objective)(parameters)
    updates, state = optimizer.update(gradient, state, parameters)
    return optax.apply_updates(parameters, updates), state, loss

  losses = []
  for _ in range(30):
    params, optimizer_state, loss = update(params, optimizer_state)
    losses.append(float(loss))
  if not np.isfinite(losses).all():
    raise RuntimeError('non-finite smoke loss')

  model = DiagonalTransitionModel(
      params, spec, context_mean, context_std, delta_mean, delta_std)
  state, action, goal = train.state[100], train.action[100], train.goal[100]
  observation = np.concatenate([state, goal])
  single = np.asarray(model.sample(
      observation, action, action, jax.random.PRNGKey(1)))
  multiple = np.asarray(model.sample(
      state, action, action, jax.random.PRNGKey(2), 13, goal=goal))
  repeated = np.asarray(model.sample(
      state, action, action, jax.random.PRNGKey(2), 13, goal=goal))
  log_prob = np.asarray(model.log_prob(
      train.state[:16], train.action[:16], train.action[:16],
      train.next_state[:16], goal=train.goal[:16]))
  off_diagonal_rejected = False
  try:
    model.sample(state, action + np.array([0.1, 0.0], np.float32), action,
                 jax.random.PRNGKey(3), goal=goal)
  except ValueError:
    off_diagonal_rejected = True

  checks = {
      'dataset_transitions': len(dataset.arrays_all.state) == 240000,
      'observed_frame_shift_exact':
          dataset.report['transition_alignment'][
              'observed_frame_shift_exact_fraction'] == 1.0,
      'losses_finite': bool(np.isfinite(losses).all()),
      'single_shape': list(single.shape),
      'multiple_shape': list(multiple.shape),
      'same_key_reproducible': bool(np.array_equal(multiple, repeated)),
      'samples_finite': bool(np.isfinite(multiple).all()),
      'sampled_frame_shift_exact': bool(np.array_equal(
          multiple[:, 2:], np.broadcast_to(state[:6], multiple[:, 2:].shape))),
      'sampled_xy_inside_global_bounds': bool(np.all(
          (multiple[:, :2] >= [0.0, 0.0])
          & (multiple[:, :2] <= [9.0, 5.0]))),
      'log_prob_finite': bool(np.isfinite(log_prob).all()),
      'off_diagonal_call_rejected': off_diagonal_rejected,
  }
  result = {
      'pass': bool(all(value for key, value in checks.items()
                       if not key.endswith('_shape'))
                   and checks['single_shape'] == [8]
                   and checks['multiple_shape'] == [13, 8]),
      'loss_first': losses[0], 'loss_last': losses[-1],
      'jax_backend': jax.default_backend(), 'checks': checks,
  }
  with open(os.path.join(args.out_dir, 'smoke.json'), 'w') as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write('\n')
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result['pass']:
    raise RuntimeError('diagonal ETT smoke failed')
  return result


if __name__ == '__main__':
  main()
