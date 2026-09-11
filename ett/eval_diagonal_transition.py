"""Evaluate diagonal PointMaze transition models on expert validation episodes."""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import jax
import numpy as np

from ett.dataset import ExpertTransitionDataset, load_audit_labels
from ett.diagonal_transition import (
    DeterministicTransitionModel, DiagonalTransitionModel, POINTMAZE_WALLS,
    load_diagonal_transition)


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--model', action='append', required=True,
                      help='label=run_dir; repeat for stochastic seeds and '
                           'the deterministic comparison')
  parser.add_argument('--primary-label', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--samples-per-context', type=int, default=64)
  parser.add_argument('--representatives-per-region', type=int, default=2)
  parser.add_argument('--bootstrap-replicates', type=int, default=2000)
  parser.add_argument('--batch-size', type=int, default=4096)
  parser.add_argument('--near-zero-tolerance', type=float, default=0.05)
  parser.add_argument('--seed', type=int, default=1701)
  return parser


def _write_json(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'w') as output:
    json.dump(value, output, indent=2, sort_keys=True)
    output.write('\n')
  os.replace(temporary, path)


def _parse_models(values):
  models, directories = {}, {}
  for value in values:
    label, separator, directory = value.partition('=')
    if not separator or not label or not directory or label in models:
      raise ValueError(f'invalid or duplicate --model value: {value}')
    models[label] = load_diagonal_transition(directory)
    directories[label] = directory
  return models, directories


def _regions(state):
  x, y = state[:, 0], state[:, 1]
  frames = state.reshape(-1, 4, 2)
  motion = np.linalg.norm(frames[:, 0] - frames[:, -1], axis=-1)
  return {
      'start': (x < 1.2) & (y > 2.5),
      'before_swamp': (x >= 1.2) & (x < 3.0) & (y > 2.5),
      'swamp_moving': ((x >= 3.0) & (x < 6.0) & (y > 2.5)
                       & (motion > 0.05)),
      'swamp_stationary': ((x >= 3.0) & (x < 6.0) & (y > 2.5)
                           & (motion <= 1e-5)),
      'safe_route': y < 2.0,
      'after_swamp': (x >= 6.0) & (y > 2.5),
  }


def _error_summary(values):
  values = np.asarray(values)
  if not len(values):
    return {'count': 0, 'mean': None, 'median': None, 'p90': None,
            'p95': None}
  return {
      'count': int(len(values)), 'mean': float(values.mean()),
      'median': float(np.median(values)),
      'p90': float(np.quantile(values, 0.90)),
      'p95': float(np.quantile(values, 0.95)),
  }


def _energy_score(samples_xy, observed_xy):
  """Unbiased O(K) energy-score estimate using independent sample pairs."""
  first = np.linalg.norm(
      samples_xy - observed_xy[:, None, :], axis=-1).mean(axis=1)
  half = samples_xy.shape[1] // 2
  if half == 0:
    return first
  second = np.linalg.norm(
      samples_xy[:, :half] - samples_xy[:, half:2 * half], axis=-1).mean(axis=1)
  return first - 0.5 * second


def _calibration(probability, outcome, bins=10):
  probability = np.asarray(probability, dtype=np.float64)
  outcome = np.asarray(outcome, dtype=np.float64)
  edges = np.linspace(0.0, 1.0, bins + 1)
  rows, ece = [], 0.0
  for index in range(bins):
    include = ((probability >= edges[index])
               & (probability < edges[index + 1]
                  if index + 1 < bins else probability <= edges[index + 1]))
    if not include.any():
      continue
    predicted = float(probability[include].mean())
    observed = float(outcome[include].mean())
    ece += include.mean() * abs(predicted - observed)
    rows.append({'low': float(edges[index]), 'high': float(edges[index + 1]),
                 'count': int(include.sum()), 'predicted': predicted,
                 'observed': observed})
  return {
      'mean_predicted': float(probability.mean()),
      'observed_frequency': float(outcome.mean()),
      'brier': float(np.mean(np.square(probability - outcome))),
      'ece_10_equal_width': float(ece), 'bins': rows,
  }


def _paired_episode_bootstrap(left, right, episodes, replicates, seed):
  delta = np.asarray(left) - np.asarray(right)
  unique = np.unique(episodes)
  episode_delta = np.asarray([
      delta[episodes == episode].mean() for episode in unique])
  rng = np.random.default_rng(seed)
  draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
  estimates = episode_delta[draws].mean(axis=1)
  return {
      'definition': 'left minus right; complete episodes are resampled',
      'left_minus_right': float(episode_delta.mean()),
      'ci95': np.quantile(estimates, [0.025, 0.975]).tolist(),
      'episode_count': int(len(unique)), 'replicates': int(replicates),
  }


def _open_endpoint(position):
  position = np.asarray(position)
  in_bounds = np.all((position >= np.array([0.0, 0.0]))
                     & (position <= np.array([9.0, 5.0])), axis=-1)
  cell = np.floor(position).astype(np.int64)
  cell[..., 0] = np.clip(cell[..., 0], 0, POINTMAZE_WALLS.shape[0] - 1)
  cell[..., 1] = np.clip(cell[..., 1], 0, POINTMAZE_WALLS.shape[1] - 1)
  open_cell = POINTMAZE_WALLS[cell[..., 0], cell[..., 1]] == 0
  return in_bounds & open_cell


def _log_prob(model, arrays, batch_size):
  values = []
  for start in range(0, len(arrays.state), batch_size):
    stop = start + batch_size
    values.append(np.asarray(model.log_prob(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], arrays.next_state[start:stop],
        goal=arrays.goal[start:stop])))
  values = np.concatenate(values)
  if not np.isfinite(values).all():
    raise RuntimeError('non-finite transition log likelihood')
  return values


def _stochastic_samples(model, arrays, count, batch_size, seed):
  samples, diagnostics, atom_probability = [], {}, []
  for batch_index, start in enumerate(range(0, len(arrays.state), batch_size)):
    stop = start + batch_size
    value, detail = model.sample_with_diagnostics(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], jax.random.PRNGKey(seed + batch_index),
        num_samples=count, goal=arrays.goal[start:stop])
    samples.append(np.asarray(value))
    for name, item in detail.items():
      diagnostics.setdefault(name, []).append(np.asarray(item))
    atom_probability.append(np.asarray(model.stationary_probability(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], goal=arrays.goal[start:stop])))
  return (np.concatenate(samples),
          {name: np.concatenate(value) for name, value in diagnostics.items()},
          np.concatenate(atom_probability))


def _deterministic_samples(model, arrays, batch_size):
  samples, raw_delta = [], []
  for start in range(0, len(arrays.state), batch_size):
    stop = start + batch_size
    samples.append(np.asarray(model.sample(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], jax.random.PRNGKey(0),
        goal=arrays.goal[start:stop])))
    raw_delta.append(np.asarray(model.predict_delta(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], goal=arrays.goal[start:stop])))
  return np.concatenate(samples)[:, None, :], np.concatenate(raw_delta)


def _interface_check(model, arrays, seed):
  state, action, goal = arrays.state[0], arrays.action[0], arrays.goal[0]
  observation = np.concatenate([state, goal])
  single = np.asarray(model.sample(
      observation, action, action, jax.random.PRNGKey(seed)))
  multiple = np.asarray(model.sample(
      state, action, action, jax.random.PRNGKey(seed + 1),
      num_samples=17, goal=goal))
  repeated = np.asarray(model.sample(
      state, action, action, jax.random.PRNGKey(seed + 1),
      num_samples=17, goal=goal))
  off_diagonal_rejected = False
  try:
    model.sample(state, action + np.array([0.1, 0.0], np.float32), action,
                 jax.random.PRNGKey(seed + 2), goal=goal)
  except ValueError:
    off_diagonal_rejected = True
  return {
      'single_shape': list(single.shape),
      'multiple_shape': list(multiple.shape),
      'same_key_reproducible': bool(np.array_equal(multiple, repeated)),
      'all_finite': bool(np.isfinite(single).all()
                         and np.isfinite(multiple).all()),
      'inside_global_bounds': bool(np.all(_open_endpoint(multiple[..., :2]))),
      'exact_frame_shift': bool(np.array_equal(
          multiple[..., 2:], np.broadcast_to(state[:6], multiple[..., 2:].shape))),
      'off_diagonal_call_rejected': off_diagonal_rejected,
  }


def _split_check(model, validation_episode_ids, dataset_sha):
  metadata = model.metadata
  split = metadata['dataset']['split']
  recorded_validation = set(split['validation_episode_ids'])
  evaluation = set(validation_episode_ids.tolist())
  recorded_sha = metadata['dataset']['dataset']['sha256']
  return {
      'dataset_sha256_matches': recorded_sha == dataset_sha,
      'all_evaluation_episodes_in_recorded_validation':
          evaluation <= recorded_validation,
      'evaluation_episodes_in_recorded_training': len(
          evaluation.intersection(split['train_episode_ids'])),
  }


def _representative_rows(region_masks, per_region):
  result = {}
  for name, mask in region_masks.items():
    candidates = np.flatnonzero(mask)
    if not candidates.size:
      continue
    quantiles = np.linspace(0.25, 0.75, min(per_region, len(candidates)))
    positions = np.unique(np.floor(quantiles * (len(candidates) - 1)).astype(int))
    result[name] = candidates[positions].tolist()
  return result


def _make_plots(out_dir, arrays, region_masks, representatives,
                samples_by_model, primary_label, metrics_by_model,
                near_zero_tolerance):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  paths = []
  columns = max((len(value) for value in representatives.values()), default=1)
  figure, axes = plt.subplots(
      len(representatives), columns,
      figsize=(4.0 * columns, 3.6 * len(representatives)), squeeze=False)
  row_index = 0
  for region, values in representatives.items():
    for column in range(columns):
      axis = axes[row_index, column]
      if column >= len(values):
        axis.axis('off')
        continue
      row = values[column]
      predicted = samples_by_model[primary_label][row, :, :2]
      current = arrays.state[row, :2]
      target = arrays.next_state[row, :2]
      axis.scatter(predicted[:, 0], predicted[:, 1], s=9, alpha=0.25,
                   label='sampled next XY')
      axis.scatter(current[0], current[1], marker='*', s=90, color='black',
                   label='current XY')
      axis.scatter(target[0], target[1], marker='x', s=70, color='red',
                   label='observed next XY')
      axis.set_xlim(max(0.0, current[0] - 1.1), min(9.0, current[0] + 1.1))
      axis.set_ylim(max(0.0, current[1] - 1.1), min(5.0, current[1] + 1.1))
      axis.set_aspect('equal', adjustable='box')
      axis.set_title(f'{region}: episode {arrays.episode_id[row]} '
                     f't={arrays.timestep[row]}')
      if row_index == 0 and column == 0:
        axis.legend(fontsize=7)
    row_index += 1
  figure.tight_layout()
  path = os.path.join(out_dir, 'representative_next_positions.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  paths.append(os.path.basename(path))

  region_names = list(region_masks)
  labels = list(metrics_by_model)
  x = np.arange(len(region_names))
  width = 0.8 / len(labels)
  figure, axis = plt.subplots(figsize=(10, 4.5))
  for index, label in enumerate(labels):
    values = [metrics_by_model[label]['regions'][name]
              ['mean_position_error_maze_units'] for name in region_names]
    axis.bar(x + (index - (len(labels) - 1) / 2) * width, values,
             width=width, label=label)
  axis.set_xticks(x, region_names, rotation=25, ha='right')
  axis.set_ylabel('mean next-position error (maze units)')
  axis.legend(fontsize=8)
  figure.tight_layout()
  path = os.path.join(out_dir, 'region_position_error.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  paths.append(os.path.basename(path))

  observed = np.linalg.norm(
      arrays.next_state[:, :2] - arrays.state[:, :2], axis=-1)
  predicted = samples_by_model[primary_label]
  observed_rate, predicted_rate = [], []
  for name in region_names:
    mask = region_masks[name]
    observed_rate.append(float((observed[mask] <= near_zero_tolerance).mean()))
    probability = np.mean(
        np.linalg.norm(predicted[mask, :, :2]
                       - arrays.state[mask, None, :2], axis=-1)
        <= near_zero_tolerance, axis=1)
    predicted_rate.append(float(probability.mean()))
  figure, axis = plt.subplots(figsize=(9, 4.5))
  axis.plot(x, observed_rate, marker='o', label='observed')
  axis.plot(x, predicted_rate, marker='o', label=primary_label)
  axis.set_xticks(x, region_names, rotation=25, ha='right')
  axis.set_ylim(0.0, 1.0)
  axis.set_ylabel(f'P(displacement norm <= {near_zero_tolerance:g})')
  axis.legend()
  figure.tight_layout()
  path = os.path.join(out_dir, 'near_zero_motion_calibration.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  paths.append(os.path.basename(path))
  return paths


def main(argv=None):
  args = build_parser().parse_args(argv)
  if min(args.samples_per_context, args.representatives_per_region,
         args.bootstrap_replicates, args.batch_size) <= 0:
    raise SystemExit('sample, representative, bootstrap, and batch counts must be positive')
  if args.samples_per_context < 2:
    raise SystemExit('energy score requires at least two samples')
  os.makedirs(args.out_dir, exist_ok=True)
  models, model_directories = _parse_models(args.model)
  if args.primary_label not in models or not isinstance(
      models[args.primary_label], DiagonalTransitionModel):
    raise ValueError('--primary-label must name a stochastic mixture model')

  dataset = ExpertTransitionDataset(args.dataset, val_frac=0.1, split_seed=0)
  train = dataset.arrays('train')
  validation = dataset.arrays('validation')
  audit_train = load_audit_labels(args.dataset, train)
  audit_validation = load_audit_labels(args.dataset, validation)
  validation_episode_ids = np.unique(validation.episode_id)
  dataset_sha = dataset.behavior_report['sha256']
  split_checks = {
      label: _split_check(model, validation_episode_ids, dataset_sha)
      for label, model in models.items()}
  if not all(value['dataset_sha256_matches']
             and value['all_evaluation_episodes_in_recorded_validation']
             and value['evaluation_episodes_in_recorded_training'] == 0
             for value in split_checks.values()):
    raise RuntimeError('model data or validation split does not match evaluation')

  region_masks = _regions(validation.state)
  strata_masks = {
      'ordinary': ~(audit_validation['entering_death']
                    | audit_validation['post_death']),
      'entering_death': audit_validation['entering_death'],
      'post_death_action_ignored': audit_validation['post_death'],
  }
  observed_delta = validation.delta_xy
  observed_exact = np.max(np.abs(observed_delta), axis=-1) <= 1e-7
  observed_near = np.linalg.norm(observed_delta, axis=-1) <= (
      args.near_zero_tolerance)

  metrics_by_model = {}
  internals = {}
  samples_by_model = {}
  for label, model in models.items():
    if isinstance(model, DiagonalTransitionModel):
      samples, diagnostics, atom_probability = _stochastic_samples(
          model, validation, args.samples_per_context, args.batch_size,
          args.seed)
      log_prob = _log_prob(model, validation, args.batch_size)
      raw_delta = diagnostics['raw_delta']
      raw_position = diagnostics['raw_position']
      blocked_raw = diagnostics['blocked_endpoint_before_projection']
    elif isinstance(model, DeterministicTransitionModel):
      samples, raw_delta_single = _deterministic_samples(
          model, validation, args.batch_size)
      raw_delta = raw_delta_single[:, None, :]
      raw_position = validation.state[:, None, :2] + raw_delta
      bounded = np.clip(
          validation.state[:, None, :2] + np.clip(raw_delta, -1.0, 1.0),
          [0.0, 0.0], [9.0, 5.0])
      blocked_raw = ~_open_endpoint(bounded)
      atom_probability = None
      log_prob = None
    else:
      raise TypeError(f'unsupported model class for {label}')

    samples_by_model[label] = samples
    samples_xy = samples[..., :2]
    predicted_mean = samples_xy.mean(axis=1)
    position_error = np.linalg.norm(
        predicted_mean - validation.next_state[:, :2], axis=-1)
    energy = _energy_score(samples_xy, validation.next_state[:, :2])
    sampled_delta = samples_xy - validation.state[:, None, :2]
    exact_probability = np.mean(
        np.max(np.abs(sampled_delta), axis=-1) <= 1e-7, axis=1)
    near_probability = np.mean(
        np.linalg.norm(sampled_delta, axis=-1) <= args.near_zero_tolerance,
        axis=1)
    frame_consistent = np.array_equal(
        samples[..., 2:], np.broadcast_to(
            validation.state[:, None, :6], samples[..., 2:].shape))
    final_open = _open_endpoint(samples_xy)
    raw_out_of_bounds = ~np.all(
        (raw_position >= np.array([0.0, 0.0]))
        & (raw_position <= np.array([9.0, 5.0])), axis=-1)
    raw_step_violation = np.any(np.abs(raw_delta) > 1.0, axis=-1)

    regions = {}
    for name, mask in region_masks.items():
      regions[name] = {
          'transitions': int(mask.sum()),
          'episodes': int(np.unique(validation.episode_id[mask]).size),
          'mean_position_error_maze_units': float(position_error[mask].mean()),
          'mean_energy_score_maze_units': float(energy[mask].mean()),
          'observed_exact_stationary_frequency': float(observed_exact[mask].mean()),
          'predicted_exact_stationary_frequency': float(
              exact_probability[mask].mean()),
          'observed_near_zero_frequency': float(observed_near[mask].mean()),
          'predicted_near_zero_frequency': float(near_probability[mask].mean()),
          'mean_sample_std_per_coordinate': samples_xy[mask].std(axis=1).mean(
              axis=0).tolist(),
      }
    strata = {}
    for name, mask in strata_masks.items():
      strata[name] = {
          'transitions': int(mask.sum()),
          'episodes': int(np.unique(validation.episode_id[mask]).size),
          'position_error': _error_summary(position_error[mask]),
          'mean_energy_score_maze_units': (
              float(energy[mask].mean()) if mask.any() else None),
          'observed_exact_stationary_frequency': (
              float(observed_exact[mask].mean()) if mask.any() else None),
          'predicted_exact_stationary_frequency': (
              float(exact_probability[mask].mean()) if mask.any() else None),
      }
    model_report = {
        'model_type': model.metadata['model_type'],
        'checkpoint_best_step': model.metadata['result']['best_step'],
        'next_position_error_maze_units': _error_summary(position_error),
        'conditional_energy_score_maze_units': {
            'mean': float(energy.mean()),
            'estimator': ('mean ||X-y|| - 0.5 mean ||X-X_prime|| using '
                          'independent paired samples')},
        'likelihood': ({
            'validation_nll_nats_per_transition': float(-log_prob.mean()),
            'measure': ('point mass at exact zero XY displacement plus '
                        'Lebesgue density for nonzero XY displacement in '
                        'raw maze units; before geometry projection')}
                       if log_prob is not None else None),
        'motion_events': {
            'exact_stationary_tolerance_linf': 1e-7,
            'near_zero_tolerance_l2_maze_units': args.near_zero_tolerance,
            'exact_stationary_calibration': _calibration(
                exact_probability, observed_exact),
            'near_zero_calibration': _calibration(
                near_probability, observed_near),
            'mean_analytic_stationary_atom_probability': (
                float(atom_probability.mean())
                if atom_probability is not None else None),
        },
        'sample_diversity': {
            'mean_conditional_std_per_xy_coordinate': samples_xy.std(
                axis=1).mean(axis=0).tolist(),
            'fraction_contexts_with_std_norm_below_1e-3': float(np.mean(
                np.linalg.norm(samples_xy.std(axis=1), axis=-1) < 1e-3)),
        },
        'geometry': {
            'raw_position_out_of_global_bounds_fraction': float(
                raw_out_of_bounds.mean()),
            'raw_displacement_exceeds_one_per_coordinate_fraction': float(
                raw_step_violation.mean()),
            'bounded_endpoint_in_wall_before_projection_fraction': float(
                blocked_raw.mean()),
            'final_endpoint_out_of_bounds_or_in_wall_fraction': float(
                (~final_open).mean()),
            'sampled_frame_shift_exact': bool(frame_consistent),
            'projection_scope': ('one-step displacement cap, global bounds, '
                                 'and endpoint cell; the environment axiswise '
                                 'substep path is not reconstructed'),
        },
        'regions': regions, 'transition_types': strata,
        'sampling_interface': _interface_check(model, validation, args.seed),
    }
    metrics_by_model[label] = model_report
    internals[label] = {'position_error': position_error, 'energy': energy}

  persistence_state = validation.state.copy()
  persistence_samples = persistence_state[:, None, :]
  persistence_error = np.linalg.norm(
      persistence_state[:, :2] - validation.next_state[:, :2], axis=-1)
  persistence_energy = persistence_error.copy()
  samples_by_model['persistence'] = persistence_samples
  persistence_exact = np.ones(len(validation.state), dtype=float)
  persistence_report = {
      'model_type': 'persistence_s_next_equals_s',
      'next_position_error_maze_units': _error_summary(persistence_error),
      'conditional_energy_score_maze_units': {
          'mean': float(persistence_energy.mean()),
          'estimator': 'deterministic forecast distance'},
      'likelihood': None,
      'motion_events': {
          'exact_stationary_tolerance_linf': 1e-7,
          'near_zero_tolerance_l2_maze_units': args.near_zero_tolerance,
          'exact_stationary_calibration': _calibration(
              persistence_exact, observed_exact),
          'near_zero_calibration': _calibration(
              persistence_exact, observed_near),
      },
      'sample_diversity': {
          'mean_conditional_std_per_xy_coordinate': [0.0, 0.0],
          'fraction_contexts_with_std_norm_below_1e-3': 1.0},
      'geometry': {
          'final_endpoint_out_of_bounds_or_in_wall_fraction': float(
              (~_open_endpoint(persistence_state[:, :2])).mean()),
          'sampled_frame_shift_exact': bool(np.array_equal(
              persistence_state[:, 2:], validation.state[:, :6])),
          'note': 'literal s_next=s baseline does not perform the F4 shift'},
      'regions': {}, 'transition_types': {},
  }
  for name, mask in region_masks.items():
    persistence_report['regions'][name] = {
        'transitions': int(mask.sum()),
        'episodes': int(np.unique(validation.episode_id[mask]).size),
        'mean_position_error_maze_units': float(persistence_error[mask].mean()),
        'mean_energy_score_maze_units': float(persistence_energy[mask].mean()),
        'observed_exact_stationary_frequency': float(observed_exact[mask].mean()),
        'predicted_exact_stationary_frequency': 1.0,
        'observed_near_zero_frequency': float(observed_near[mask].mean()),
        'predicted_near_zero_frequency': 1.0,
        'mean_sample_std_per_coordinate': [0.0, 0.0],
    }
  for name, mask in strata_masks.items():
    persistence_report['transition_types'][name] = {
        'transitions': int(mask.sum()),
        'episodes': int(np.unique(validation.episode_id[mask]).size),
        'position_error': _error_summary(persistence_error[mask]),
        'mean_energy_score_maze_units': (
            float(persistence_energy[mask].mean()) if mask.any() else None),
        'observed_exact_stationary_frequency': (
            float(observed_exact[mask].mean()) if mask.any() else None),
        'predicted_exact_stationary_frequency': 1.0,
    }
  metrics_by_model['persistence'] = persistence_report
  internals['persistence'] = {
      'position_error': persistence_error, 'energy': persistence_energy}

  comparisons = {}
  labels = list(models) + ['persistence']
  for left in models:
    for right in labels:
      if left == right:
        continue
      pair = f'{left}:{right}'
      comparisons[pair] = {
          'mean_position_error': _paired_episode_bootstrap(
              internals[left]['position_error'],
              internals[right]['position_error'], validation.episode_id,
              args.bootstrap_replicates, args.seed),
          'energy_score': _paired_episode_bootstrap(
              internals[left]['energy'], internals[right]['energy'],
              validation.episode_id, args.bootstrap_replicates,
              args.seed + 1),
      }

  representatives = _representative_rows(
      region_masks, args.representatives_per_region)
  representative_report = {}
  for region, rows in representatives.items():
    representative_report[region] = []
    for row in rows:
      entry = {
          'validation_row': int(row),
          'original_episode_id': int(validation.episode_id[row]),
          'timestep': int(validation.timestep[row]),
          'state': validation.state[row].tolist(),
          'action_equals_observational_action': validation.action[row].tolist(),
          'observed_next_state': validation.next_state[row].tolist(),
          'uses_pooled_neighborhood': False,
          'models': {},
      }
      for label in models:
        sample = samples_by_model[label][row]
        entry['models'][label] = {
            'sample_mean_next_xy': sample[:, :2].mean(axis=0).tolist(),
            'sample_std_next_xy': sample[:, :2].std(axis=0).tolist(),
            'sample_min_next_xy': sample[:, :2].min(axis=0).tolist(),
            'sample_max_next_xy': sample[:, :2].max(axis=0).tolist(),
        }
      representative_report[region].append(entry)

  audit_counts = {}
  for split, arrays, audit in (
      ('train', train, audit_train),
      ('validation', validation, audit_validation)):
    audit_counts[split] = {
        'transitions': len(arrays.state),
        'episodes': int(np.unique(arrays.episode_id).size),
        'died_episodes': int(np.unique(
            arrays.episode_id[audit['episode_died']]).size),
        'entering_death_transitions': int(audit['entering_death'].sum()),
        'post_death_action_ignored_transitions': int(audit['post_death'].sum()),
        'post_death_xy_exactly_stationary_fraction': (
            float(audit['exact_stationary'][audit['post_death']].mean())
            if audit['post_death'].any() else None),
    }

  plot_paths = _make_plots(
      args.out_dir, validation, region_masks, representatives,
      samples_by_model, args.primary_label, metrics_by_model,
      args.near_zero_tolerance)
  report = {
      'evaluation_scope': ('checkpoint-selection validation on expert-positive '
                           'diagonal transitions; no independent test split'),
      'dataset': dataset.report,
      'model_directories': model_directories,
      'fixed_protocol': {
          'seed': args.seed,
          'samples_per_context': args.samples_per_context,
          'bootstrap_replicates': args.bootstrap_replicates,
          'near_zero_tolerance_l2_maze_units': args.near_zero_tolerance,
          'same_validation_contexts_targets_and_random_keys_across_models': True,
      },
      'diagonal_contract': {
          'training_and_evaluation': 'action == observational_action',
          'target': 'recorded next state',
          'off_diagonal_identified_or_validated': False,
      },
      'absorbing_behavior_audit_only': audit_counts,
      'region_counts': {
          name: {'transitions': int(mask.sum()),
                 'episodes': int(np.unique(
                     validation.episode_id[mask]).size)}
          for name, mask in region_masks.items()},
      'observed': {
          'frame_shift_exact': bool(np.array_equal(
              validation.next_state[:, 2:], validation.state[:, :6])),
          'next_xy_inside_open_cells_fraction': float(
              _open_endpoint(validation.next_state[:, :2]).mean()),
          'exact_stationary_frequency': float(observed_exact.mean()),
          'near_zero_frequency': float(observed_near.mean()),
      },
      'split_checks': split_checks,
      'models': metrics_by_model,
      'comparisons': comparisons,
      'representative_contexts': representative_report,
      'plots': plot_paths,
  }
  report['all_checks_pass'] = bool(
      report['observed']['frame_shift_exact']
      and report['observed']['next_xy_inside_open_cells_fraction'] == 1.0
      and all(value['sampling_interface']['same_key_reproducible']
              and value['sampling_interface']['all_finite']
              and value['sampling_interface']['inside_global_bounds']
              and value['sampling_interface']['exact_frame_shift']
              and value['sampling_interface']['off_diagonal_call_rejected']
              for value in metrics_by_model.values()
              if 'sampling_interface' in value)
      and all(value['geometry']['final_endpoint_out_of_bounds_or_in_wall_fraction']
              == 0.0 for value in metrics_by_model.values()
              if value['model_type'] != 'persistence_s_next_equals_s'))
  _write_json(os.path.join(args.out_dir, 'evaluation.json'), report)
  print(f'expert validation: {len(validation_episode_ids)} episodes, '
        f'{len(validation.state)} transitions')
  for label, value in metrics_by_model.items():
    nll = value['likelihood']
    nll_text = (f'{nll["validation_nll_nats_per_transition"]:.6f}'
                if nll else 'n/a')
    print(f'{label}: position_error='
          f'{value["next_position_error_maze_units"]["mean"]:.6f} '
          f'energy={value["conditional_energy_score_maze_units"]["mean"]:.6f} '
          f'NLL={nll_text}')
  print(f'wrote {os.path.join(args.out_dir, "evaluation.json")}; '
        f'checks={report["all_checks_pass"]}')
  return report


if __name__ == '__main__':
  main()
