"""Train the expert-positive diagonal PointMaze transition baseline."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ett.dataset import ExpertTransitionDataset
from ett.diagonal_transition import (
    DiagonalTransitionSpec, diagonal_transition_log_prob,
    make_deterministic_network, make_transition_network)
from propensity.expert_population import save_expert_population_manifest


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--model-type', choices=('mixture', 'deterministic'),
                      default='mixture')
  parser.add_argument('--num-components', type=int, default=3)
  parser.add_argument('--hidden-sizes', default='128,128')
  parser.add_argument('--steps', type=int, default=20000)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--eval-every', type=int, default=500)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--split-seed', type=int, default=0)
  parser.add_argument('--val-frac', type=float, default=0.1)
  parser.add_argument('--overwrite', action='store_true')
  return parser


def _write_json(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'w') as output:
    json.dump(value, output, indent=2, sort_keys=True)
    output.write('\n')
  os.replace(temporary, path)


def _write_pickle(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'wb') as output:
    pickle.dump(jax.device_get(value), output, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(temporary, path)


def _normalizer(value):
  mean = value.mean(axis=0)
  std = value.std(axis=0)
  constant = std < 1e-6
  return (mean.astype(np.float32), np.where(constant, 1.0, std).astype(np.float32),
          np.flatnonzero(constant).tolist())


def _git_provenance():
  """Record the source revision without making Git a runtime dependency."""
  def run(*arguments):
    try:
      return subprocess.run(
          ['git', *arguments], cwd=_ROOT, check=True, capture_output=True,
          text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
      return None

  status = run('status', '--short', '--untracked-files=no')
  return {
      'commit': run('rev-parse', 'HEAD'),
      'branch': run('branch', '--show-current'),
      'tracked_worktree_dirty': None if status is None else bool(status),
      'tracked_dirty_paths': None if status is None else status.splitlines(),
      'untracked_files_excluded_from_dirty_check': True,
  }


def _evaluate_mixture(network, params, context, delta, context_mean,
                      context_std, delta_mean, delta_std, spec, batch_size):
  values = []
  for start in range(0, len(context), batch_size):
    stop = start + batch_size
    normalized = (context[start:stop] - context_mean) / context_std
    distribution = network.apply(params, jnp.asarray(normalized))
    log_prob = diagonal_transition_log_prob(
        distribution, jnp.asarray(delta[start:stop]), delta_mean, delta_std,
        spec)
    values.append(np.asarray(log_prob))
  values = np.concatenate(values)
  return float(-values.mean())


def _evaluate_deterministic(network, params, context, delta, context_mean,
                            context_std, delta_mean, delta_std, batch_size):
  squared = []
  for start in range(0, len(context), batch_size):
    stop = start + batch_size
    normalized = (context[start:stop] - context_mean) / context_std
    prediction = np.asarray(network.apply(params, jnp.asarray(normalized)))
    target = (delta[start:stop] - delta_mean) / delta_std
    squared.append(np.sum(np.square(prediction - target), axis=-1))
  return float(np.concatenate(squared).mean())


def main(argv=None):
  args = build_parser().parse_args(argv)
  if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
    raise SystemExit('steps, batch size, and evaluation interval must be positive')
  hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(','))
  if not hidden_sizes:
    raise SystemExit('hidden sizes cannot be empty')
  if os.path.exists(os.path.join(args.out_dir, 'config.json')) and not args.overwrite:
    raise SystemExit('output directory already contains a run; use --overwrite')
  os.makedirs(args.out_dir, exist_ok=True)

  dataset = ExpertTransitionDataset(
      args.dataset, val_frac=args.val_frac, split_seed=args.split_seed)
  train = dataset.arrays('train')
  validation = dataset.arrays('validation')
  context_train, context_validation = train.context, validation.context
  delta_train, delta_validation = train.delta_xy, validation.delta_xy
  spec = DiagonalTransitionSpec(
      num_components=(args.num_components if args.model_type == 'mixture' else 1),
      hidden_sizes=hidden_sizes)
  context_mean, context_std, context_constant = _normalizer(context_train)
  stationary_train = np.max(np.abs(delta_train), axis=-1) <= (
      spec.stationary_tolerance)
  normalization_rows = (~stationary_train if args.model_type == 'mixture'
                        else np.ones(len(delta_train), dtype=bool))
  if not normalization_rows.any():
    raise RuntimeError('no rows are available for displacement normalization')
  delta_mean, delta_std, delta_constant = _normalizer(
      delta_train[normalization_rows])

  model_type = ('zero_inflated_mixture' if args.model_type == 'mixture'
                else 'deterministic_mse')
  config = {
      'format_version': 1,
      'task': 'pointmaze_ett_diagonal_expert_transition',
      'model_type': model_type,
      'diagonal_training_contract': {
          'intervened_action': 'recorded expert action',
          'observational_action': 'the same recorded expert action',
          'target': 'recorded next learner-visible state',
          'off_diagonal_identified': False,
          'sampled_nominal_actions_used_as_labels': False,
      },
      'dataset': dataset.report,
      'model': spec.asdict(),
      'inputs': {
          'ordered_context': ['state_f4', 'action', 'observational_action',
                              'commanded_goal_f4'],
          'learner_visible_only': True,
          'excluded': ['swamp_bits', 'entered_active_swamp', 'teacher_mode',
                       'route_label', 'force_safe', 'wait_count', 'reward',
                       'future_information', 'hindsight_goal'],
      },
      'target_distribution': {
          'modeled_quantity': 'newest XY displacement in raw maze units',
          'stationary_component': 'exact point mass at zero displacement',
          'moving_component': ('conditional mixture of diagonal Gaussian '
                               'densities in two-dimensional displacement'),
          'remaining_frames': 'shifted deterministically from the input state',
          'sampling_geometry': ('componentwise displacement cap at 1 maze '
                                'unit, global state bounds, then blocked '
                                'endpoint fallback to current XY'),
          'likelihood_geometry_note': ('reported likelihood describes the '
                                       'pre-projection displacement mixture'),
      },
      'context_normalization': {
          'fit_split': 'selected expert training trajectories only',
          'mean': context_mean.tolist(), 'std': context_std.tolist(),
          'constant_dimensions': context_constant,
      },
      'delta_normalization': {
          'fit_split': ('non-stationary selected expert training rows'
                        if args.model_type == 'mixture'
                        else 'all selected expert training rows'),
          'mean': delta_mean.tolist(), 'std': delta_std.tolist(),
          'constant_dimensions': delta_constant,
      },
      'optimization': {
          'seed': args.seed, 'steps': args.steps,
          'batch_size': args.batch_size,
          'learning_rate': args.learning_rate,
          'eval_every': args.eval_every,
          'checkpoint_selection': 'lowest validation objective',
      },
      'runtime': {
          'jax_version': jax.__version__, 'jax_backend': jax.default_backend(),
          'jax_devices': [str(device) for device in jax.devices()],
      },
      'source': _git_provenance(),
      'counts': {
          'train_transitions': len(train.state),
          'validation_transitions': len(validation.state),
          'training_exact_stationary': int(stationary_train.sum()),
          'training_exact_stationary_fraction': float(stationary_train.mean()),
      },
      'status': 'running',
  }
  _write_json(os.path.join(args.out_dir, 'config.json'), config)
  save_expert_population_manifest(
      os.path.join(args.out_dir, 'expert_population_manifest.json'),
      dataset.population_manifest)

  network = (make_transition_network(spec) if args.model_type == 'mixture'
             else make_deterministic_network(spec))
  key = jax.random.PRNGKey(args.seed)
  key, init_key = jax.random.split(key)
  params = network.init(init_key, jnp.asarray(
      ((context_train[:2] - context_mean) / context_std)))
  optimizer = optax.adam(args.learning_rate)
  optimizer_state = optimizer.init(params)

  if args.model_type == 'mixture':
    def objective(parameters, context, delta):
      distribution = network.apply(parameters, context)
      return -jnp.mean(diagonal_transition_log_prob(
          distribution, delta, delta_mean, delta_std, spec))
  else:
    def objective(parameters, context, delta):
      prediction = network.apply(parameters, context)
      return jnp.mean(jnp.sum(jnp.square(prediction - delta), axis=-1))

  @jax.jit
  def update(parameters, state, context, delta):
    loss, gradient = jax.value_and_grad(objective)(parameters, context, delta)
    updates, state = optimizer.update(gradient, state, parameters)
    return optax.apply_updates(parameters, updates), state, loss

  normalized_context_train = (context_train - context_mean) / context_std
  normalized_delta_train = (delta_train - delta_mean) / delta_std
  rng = np.random.default_rng(args.seed)
  metric_steps, train_objectives, validation_objectives = [], [], []
  started = time.time()
  print(f'model={model_type} train={len(train.state)} '
        f'validation={len(validation.state)} K={spec.num_components} '
        f'backend={jax.default_backend()}', flush=True)

  def validation_objective(parameters):
    if args.model_type == 'mixture':
      return _evaluate_mixture(
          network, parameters, context_validation, delta_validation,
          context_mean, context_std, delta_mean, delta_std, spec, 8192)
    return _evaluate_deterministic(
        network, parameters, context_validation, delta_validation,
        context_mean, context_std, delta_mean, delta_std, 8192)

  initial = validation_objective(params)
  if not np.isfinite(initial):
    raise RuntimeError('non-finite initial validation objective')
  best_value, best_step, best_params = initial, 0, params
  _write_pickle(os.path.join(args.out_dir, 'best.pkl'), {
      'params': best_params, 'step': best_step,
      'validation_objective': best_value})
  print(f'step=0 validation_objective={initial:.6f}', flush=True)
  for step in range(1, args.steps + 1):
    rows = rng.integers(0, len(train.state), size=args.batch_size)
    batch_context = jnp.asarray(normalized_context_train[rows])
    batch_delta = (jnp.asarray(delta_train[rows])
                   if args.model_type == 'mixture' else
                   jnp.asarray(normalized_delta_train[rows]))
    params, optimizer_state, loss = update(
        params, optimizer_state, batch_context, batch_delta)
    if step % args.eval_every == 0 or step == args.steps:
      train_value = float(loss)
      validation_value = validation_objective(params)
      if not np.isfinite([train_value, validation_value]).all():
        raise RuntimeError('non-finite training or validation objective')
      metric_steps.append(step)
      train_objectives.append(train_value)
      validation_objectives.append(validation_value)
      improved = validation_value < best_value
      if improved:
        best_value, best_step, best_params = validation_value, step, params
        _write_pickle(os.path.join(args.out_dir, 'best.pkl'), {
            'params': best_params, 'step': best_step,
            'validation_objective': best_value})
      marker = ' *' if improved else ''
      print(f'step={step} train={train_value:.6f} '
            f'validation={validation_value:.6f} '
            f'best={best_value:.6f}@{best_step}{marker}', flush=True)

  _write_pickle(os.path.join(args.out_dir, 'latest.pkl'), {
      'params': params, 'optimizer_state': optimizer_state, 'step': args.steps})
  elapsed = time.time() - started
  metrics = {
      'steps': metric_steps,
      'train_objective': train_objectives,
      'validation_objective': validation_objectives,
      'objective_name': ('negative_log_likelihood_nats_per_transition'
                         if args.model_type == 'mixture'
                         else 'standardized_displacement_squared_error'),
      'all_recorded_losses_finite': bool(np.isfinite(
          train_objectives + validation_objectives).all()),
      'best_step': int(best_step),
      'best_validation_objective': float(best_value),
      'elapsed_seconds': float(elapsed),
  }
  _write_json(os.path.join(args.out_dir, 'metrics.json'), metrics)
  config['status'] = 'complete'
  config['result'] = {
      'best_step': int(best_step),
      'best_validation_objective': float(best_value),
      'all_recorded_losses_finite': metrics['all_recorded_losses_finite'],
  }
  _write_json(os.path.join(args.out_dir, 'config.json'), config)
  print(json.dumps(config['result'], sort_keys=True))
  return config['result']


if __name__ == '__main__':
  main()
