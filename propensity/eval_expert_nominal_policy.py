"""Evaluate nominal-policy models on one fixed expert-positive validation view.

All model comparisons use the same original episode ids, contexts, observed
actions, Gaussian noise, and mixture uniforms. Results are validation results:
the repository has no untouched nominal-policy test split.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import numpy as np

from propensity.dataset import BehaviorDataset
from propensity.expert_population import resolve_expert_positive_episodes
from propensity.nominal_policy import load_nominal_policy
from propensity.audit_boundary_shortcut import (
    boundary_features, fit_logistic, grouped_split)


BOUNDARY_TOL = 1e-6


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--model', action='append', required=True,
                      help='label=run_dir; repeat for expert and historical models')
  parser.add_argument('--compare', action='append', default=[],
                      help='left_label:right_label for paired episode bootstrap')
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--seed', type=int, default=917)
  parser.add_argument('--batch-size', type=int, default=8192)
  parser.add_argument('--boundary-samples', type=int, default=64)
  parser.add_argument('--energy-contexts', type=int, default=1024)
  parser.add_argument('--energy-samples', type=int, default=32)
  parser.add_argument('--representative-samples', type=int, default=1024)
  parser.add_argument('--local-neighbors', type=int, default=256)
  parser.add_argument('--bootstrap-replicates', type=int, default=2000)
  parser.add_argument('--plot-labels', default='',
                      help='comma-separated model labels; default expert models')
  return parser


def _write_json(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'w') as output:
    json.dump(value, output, indent=2, sort_keys=True)
    output.write('\n')
  os.replace(temporary, path)


def _parse_models(specifications):
  result = {}
  for item in specifications:
    label, separator, directory = item.partition('=')
    if not separator or not label or not directory or label in result:
      raise ValueError(f'invalid or duplicate --model value: {item}')
    result[label] = load_nominal_policy(directory)
  return result


def _softmax(logits):
  logits = logits - np.max(logits, axis=-1, keepdims=True)
  values = np.exp(logits)
  return values / values.sum(axis=-1, keepdims=True)


def _fixed_samples(model, context, gaussian, uniform):
  """Sample with caller-supplied banks; return final and latent actions."""
  distribution = model.distribution(context)
  weights = _softmax(np.asarray(distribution.logits))
  location = np.asarray(distribution.loc)
  scale = np.asarray(distribution.scale)
  cumulative = np.cumsum(weights, axis=-1)
  component = np.sum(uniform[..., None] > cumulative[:, None, :], axis=-1)
  component = np.minimum(component, model.spec.num_components - 1)
  row = np.arange(len(context))[:, None]
  raw = location[row, component] + scale[row, component] * gaussian
  return np.clip(raw, model.spec.action_low, model.spec.action_high), raw


def _log_prob(model, context, action, batch_size):
  blocks = []
  for start in range(0, len(context), batch_size):
    stop = start + batch_size
    blocks.append(np.asarray(model.log_prob(context[start:stop],
                                            action[start:stop])))
  result = np.concatenate(blocks)
  if not np.isfinite(result).all():
    raise RuntimeError('non-finite log likelihood')
  return result


def _analytic_boundaries(model, context, batch_size):
  lower, upper = [], []
  for start in range(0, len(context), batch_size):
    stop = start + batch_size
    lo, hi = model.boundary_probabilities(context[start:stop])
    lower.append(np.asarray(lo))
    upper.append(np.asarray(hi))
  return np.concatenate(lower), np.concatenate(upper)


def _action_stats(action):
  flat = np.asarray(action).reshape(-1, action.shape[-1])
  return {
      'mean': flat.mean(axis=0).tolist(),
      'std': flat.std(axis=0).tolist(),
      'quantile_05': np.quantile(flat, 0.05, axis=0).tolist(),
      'quantile_50': np.quantile(flat, 0.50, axis=0).tolist(),
      'quantile_95': np.quantile(flat, 0.95, axis=0).tolist(),
      'min': flat.min(axis=0).tolist(), 'max': flat.max(axis=0).tolist(),
  }


def _boundary_stats(action):
  flat = np.asarray(action).reshape(-1, action.shape[-1])
  lower = flat <= -1.0 + BOUNDARY_TOL
  upper = flat >= 1.0 - BOUNDARY_TOL
  saturated = lower | upper
  status = np.where(lower, -1, np.where(upper, 1, 0))
  patterns, counts = np.unique(status, axis=0, return_counts=True)
  order = np.argsort(counts)[::-1][:10]
  return {
      'lower_per_coordinate': lower.mean(axis=0).tolist(),
      'upper_per_coordinate': upper.mean(axis=0).tolist(),
      'either_per_coordinate': saturated.mean(axis=0).tolist(),
      'any_boundary_coordinate': float(saturated.any(axis=1).mean()),
      'both_coordinates_at_boundary': float(saturated.all(axis=1).mean()),
      'top_joint_patterns': [
          {'pattern': patterns[i].tolist(),
           'fraction': float(counts[i] / len(status)), 'count': int(counts[i])}
          for i in order],
  }


def _regions(context):
  x, y = context[:, 0], context[:, 1]
  frames = context[:, :8].reshape(-1, 4, 2)
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


def _energy_score(samples, observed):
  first = np.linalg.norm(samples - observed[:, None, :], axis=-1).mean(axis=1)
  pair = np.linalg.norm(samples[:, :, None, :] - samples[:, None, :, :],
                        axis=-1)
  k = samples.shape[1]
  second = pair.sum(axis=(1, 2)) / max(k * (k - 1), 1)
  return first - 0.5 * second


def _paired_bootstrap(left, right, episode, replicates, seed):
  delta = np.asarray(left) - np.asarray(right)
  unique = np.unique(episode)
  episode_delta = np.asarray([delta[episode == value].mean() for value in unique])
  rng = np.random.default_rng(seed)
  draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
  estimates = episode_delta[draws].mean(axis=1)
  return {
      'definition': 'left minus right; complete episodes are resampled',
      'left_minus_right': float(episode_delta.mean()),
      'ci95': np.quantile(estimates, [0.025, 0.975]).tolist(),
      'episode_count': int(len(unique)), 'replicates': int(replicates),
  }


def _boundary_classifier(real, generated, episode, seed):
  """Existing boundary-only probe with an episode-separated train/test split."""
  real_features, names = boundary_features(real)
  generated_features, _ = boundary_features(generated)
  features = np.concatenate([real_features, generated_features], axis=0)
  labels = np.concatenate([np.ones(len(real), dtype=int),
                           np.zeros(len(generated), dtype=int)])
  groups = np.concatenate([episode, episode])
  train, test, train_episodes, test_episodes = grouped_split(
      groups, test_frac=0.3, seed=seed)
  result = fit_logistic(features[train], labels[train], features[test],
                        labels[test], names, seed=seed)
  result.update({
      'positive_class': 'real expert action',
      'negative_class': 'generated action',
      'features': 'existing boundary-only feature family; no context or labels',
      'train_episodes': int(train_episodes),
      'test_episodes': int(test_episodes),
      'episode_overlap': 0,
  })
  return result


def _action_pattern(action):
  action = np.asarray(action)
  magnitude = np.linalg.norm(action, axis=-1)
  angle = np.arctan2(action[..., 1], action[..., 0])
  sector = np.floor((angle + np.pi) / (np.pi / 4)).astype(int) % 8
  names = np.asarray(['west', 'southwest', 'south', 'southeast',
                      'east', 'northeast', 'north', 'northwest'])
  result = names[sector]
  return np.where(magnitude < 0.25, 'wait', result)


def _pattern_frequencies(action):
  values, counts = np.unique(_action_pattern(action), return_counts=True)
  return {str(value): float(count / counts.sum())
          for value, count in zip(values, counts)}


def _sample_summary(samples):
  return {
      **_action_stats(samples),
      'boundary': _boundary_stats(samples),
      'mean_conditional_std': np.std(samples, axis=0).tolist(),
      'all_finite': bool(np.isfinite(samples).all()),
  }


def _model_split_check(model, validation_episodes, dataset_sha):
  metadata = model.metadata
  if metadata['dataset']['sha256'] != dataset_sha:
    raise RuntimeError('model dataset SHA-256 does not match evaluation dataset')
  recorded_validation = set(metadata['split']['validation_episode_ids'])
  evaluation = set(validation_episodes.tolist())
  return {
      'recorded_population': metadata.get('population', 'mixed_historical'),
      'evaluation_episodes_all_in_model_validation': evaluation <= recorded_validation,
      'evaluation_episodes_in_model_training': len(evaluation - recorded_validation),
      'recorded_validation_episode_count': len(recorded_validation),
  }


def _make_plots(out_dir, labels, models, representatives, local_actions,
                representative_samples, observed_boundary, model_boundaries):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  region_names = list(representatives)
  figure, axes = plt.subplots(
      len(region_names), len(labels), figsize=(4.2 * len(labels),
                                               3.5 * len(region_names)),
      squeeze=False)
  for row, region in enumerate(region_names):
    for column, label in enumerate(labels):
      axis = axes[row, column]
      samples = representative_samples[label][region]
      nearby = local_actions[region]
      axis.scatter(nearby[:, 0], nearby[:, 1], s=8, alpha=0.20,
                   color='black', label='local expert actions')
      axis.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.25,
                   label=label)
      axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal',
               xlabel='action 0', ylabel='action 1')
      axis.set_title(f'{region}: {label}')
      if row == 0 and column == 0:
        axis.legend(loc='lower left', fontsize=7)
  figure.tight_layout()
  sample_path = os.path.join(out_dir, 'conditional_action_samples.png')
  figure.savefig(sample_path, dpi=160)
  plt.close(figure)

  x = np.arange(2)
  width = 0.8 / (len(labels) + 1)
  figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
  for side_index, side in enumerate(('lower', 'upper')):
    axis = axes[side_index]
    observed = np.asarray(observed_boundary[f'{side}_per_coordinate'])
    axis.bar(x - 0.4 + width / 2, observed, width, label='observed expert')
    for index, label in enumerate(labels):
      value = np.asarray(model_boundaries[label][f'sampled_{side}'])
      axis.bar(x - 0.4 + width * (index + 1.5), value, width, label=label)
    axis.set_xticks(x, ['action 0', 'action 1'])
    axis.set_title(f'{side} boundary mass')
    axis.set_ylabel('frequency')
  axes[0].legend(fontsize=8)
  figure.tight_layout()
  boundary_path = os.path.join(out_dir, 'boundary_frequencies.png')
  figure.savefig(boundary_path, dpi=160)
  plt.close(figure)
  return [os.path.basename(sample_path), os.path.basename(boundary_path)]


def main(argv=None):
  args = build_parser().parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  if min(args.boundary_samples, args.energy_samples,
         args.representative_samples, args.bootstrap_replicates) <= 1:
    raise SystemExit('sample counts and bootstrap replicates must exceed one')
  models = _parse_models(args.model)
  model_directories = {
      item.partition('=')[0]: item.partition('=')[2] for item in args.model}
  selected_ids, selection = resolve_expert_positive_episodes(args.dataset)
  dataset = BehaviorDataset(
      args.dataset, val_frac=0.1, seed=0, state_mode='obs',
      split_level='episode', strict_bounds=True,
      include_episode_ids=selected_ids, split_reference='source')
  validation = dataset.arrays('val')
  row_indices = dataset._val_idx
  episodes = dataset._episode_of_row[row_indices]
  validation_episodes = np.unique(episodes)
  dataset_sha = dataset.fingerprint['sha256']

  split_checks = {label: _model_split_check(model, validation_episodes,
                                             dataset_sha)
                  for label, model in models.items()}
  if not all(item['evaluation_episodes_all_in_model_validation']
             for item in split_checks.values()):
    raise RuntimeError('an evaluation episode was used to train a compared model')

  rng = np.random.default_rng(args.seed)
  shuffled = rng.integers(0, len(validation.state), size=len(validation.state))
  bad = episodes[shuffled] == episodes
  while np.any(bad):
    shuffled[bad] = rng.integers(0, len(validation.state), size=int(bad.sum()))
    bad = episodes[shuffled] == episodes
  gaussian_boundary = rng.standard_normal(
      (len(validation.state), args.boundary_samples, dataset.action_dim),
      dtype=np.float32)
  uniform_boundary = rng.random(
      (len(validation.state), args.boundary_samples), dtype=np.float32)
  n_energy = min(args.energy_contexts, len(validation.state))
  energy_rows = rng.choice(len(validation.state), n_energy, replace=False)
  energy_gaussian = np.random.default_rng(args.seed + 101).standard_normal(
      (n_energy, args.energy_samples, dataset.action_dim), dtype=np.float32)
  energy_uniform = np.random.default_rng(args.seed + 102).random(
      (n_energy, args.energy_samples), dtype=np.float32)

  region_masks = _regions(validation.state)
  region_counts = {
      name: {'transitions': int(mask.sum()),
             'episodes': int(np.unique(episodes[mask]).size)}
      for name, mask in region_masks.items()}
  observed_boundary = _boundary_stats(validation.action)
  report = {
      'evaluation_scope': 'checkpoint-selection validation; no untouched test split',
      'dataset': dataset.report(), 'expert_population': selection,
      'model_directories': model_directories,
      'fixed_protocol': {
          'seed': args.seed, 'validation_episode_count': int(len(validation_episodes)),
          'validation_transition_count': int(len(validation.state)),
          'boundary_samples_per_context': args.boundary_samples,
          'energy_contexts': args.energy_contexts,
          'energy_samples_per_context': args.energy_samples,
          'representative_samples': args.representative_samples,
          'local_neighbors': args.local_neighbors,
          'bootstrap_replicates': args.bootstrap_replicates,
          'comparisons': args.compare,
          'plot_labels': args.plot_labels,
          'same_rows_contexts_actions_gaussian_noise_and_uniforms_across_models': True,
      },
      'split_checks': split_checks, 'region_counts': region_counts,
      'observed_expert': {
          'action_distribution': _action_stats(validation.action),
          'boundary': observed_boundary},
      'models': {}, 'comparisons': {},
  }
  internals = {}
  for model_index, (label, model) in enumerate(models.items()):
    correct_lp = _log_prob(model, validation.state, validation.action,
                           args.batch_size)
    shuffled_lp = _log_prob(model, validation.state[shuffled], validation.action,
                            args.batch_size)
    analytic_lower, analytic_upper = _analytic_boundaries(
        model, validation.state, args.batch_size)
    sampled, raw = _fixed_samples(
        model, validation.state, gaussian_boundary, uniform_boundary)
    sampled_lower_context = (sampled <= -1.0 + BOUNDARY_TOL).mean(axis=1)
    sampled_upper_context = (sampled >= 1.0 - BOUNDARY_TOL).mean(axis=1)
    conditional_std = sampled.std(axis=1)

    energy_correct_samples, _ = _fixed_samples(
        model, validation.state[energy_rows], energy_gaussian, energy_uniform)
    energy_shuffled_samples, _ = _fixed_samples(
        model, validation.state[shuffled[energy_rows]], energy_gaussian,
        energy_uniform)
    energy_correct = _energy_score(
        energy_correct_samples, validation.action[energy_rows])
    energy_shuffled = _energy_score(
        energy_shuffled_samples, validation.action[energy_rows])

    analytic = {
        'lower_per_coordinate': analytic_lower.mean(axis=0).tolist(),
        'upper_per_coordinate': analytic_upper.mean(axis=0).tolist(),
    }
    sampled_boundary = _boundary_stats(sampled)
    boundary_summary = {
        'observed': observed_boundary,
        'analytic': analytic,
        'sampled': sampled_boundary,
        'analytic_sampled_max_abs_error': float(max(
            np.max(np.abs(analytic_lower.mean(axis=0)
                          - sampled_lower_context.mean(axis=0))),
            np.max(np.abs(analytic_upper.mean(axis=0)
                          - sampled_upper_context.mean(axis=0))))),
        'raw_latent_out_of_box_coordinate_fraction': float(
            np.mean((raw < -1.0) | (raw > 1.0))),
        'final_out_of_box_coordinate_fraction': float(
            np.mean((sampled < -1.0) | (sampled > 1.0))),
        'real_vs_generated_boundary_classifier': _boundary_classifier(
            validation.action, sampled[:, 0, :], episodes, args.seed),
    }
    regions = {}
    for region, mask in region_masks.items():
      regions[region] = {
          **region_counts[region],
          'nll_nats_per_action_vector': (
              float(-correct_lp[mask].mean()) if mask.any() else None),
          'observed_boundary': (_boundary_stats(validation.action[mask])
                                if mask.any() else None),
          'predicted_analytic_lower_per_coordinate': (
              analytic_lower[mask].mean(axis=0).tolist() if mask.any() else None),
          'predicted_analytic_upper_per_coordinate': (
              analytic_upper[mask].mean(axis=0).tolist() if mask.any() else None),
          'sampled_lower_per_coordinate': (
              sampled_lower_context[mask].mean(axis=0).tolist()
              if mask.any() else None),
          'sampled_upper_per_coordinate': (
              sampled_upper_context[mask].mean(axis=0).tolist()
              if mask.any() else None),
          'mean_conditional_std_per_coordinate': (
              conditional_std[mask].mean(axis=0).tolist() if mask.any() else None),
      }
    single_a = np.asarray(model.sample(
        validation.state[0], jax.random.PRNGKey(args.seed + model_index)))
    multi_a = np.asarray(model.sample(
        validation.state[0], jax.random.PRNGKey(args.seed + 100 + model_index),
        num_samples=17))
    repeat_a = np.asarray(model.sample(
        validation.state[0], jax.random.PRNGKey(args.seed + 100 + model_index),
        num_samples=17))
    model_report = {
        'population': model.metadata.get('population', 'mixed_historical'),
        'checkpoint_best_step': model.metadata.get('result', {}).get('best_step'),
        'num_components': model.spec.num_components,
        'nll': {
            'correct_context_nats_per_action_vector': float(-correct_lp.mean()),
            'shuffled_context_nats_per_action_vector': float(-shuffled_lp.mean()),
            'shuffled_minus_correct': float(
                -shuffled_lp.mean() + correct_lp.mean()),
            'measure': ('Lebesgue density inside (-1,1)^2 plus point masses '
                        'at componentwise clipped boundaries')},
        'action_distribution': _action_stats(sampled),
        'boundary': boundary_summary,
        'conditional_energy_score': {
            'estimator': ('mean ||X-y|| minus half the unbiased mean '
                          'off-diagonal ||X-X_prime||'),
            'correct_context': float(energy_correct.mean()),
            'shuffled_context': float(energy_shuffled.mean()),
            'shuffled_minus_correct': float(
                energy_shuffled.mean() - energy_correct.mean()),
            'contexts': int(n_energy), 'samples_per_context': args.energy_samples},
        'regions': regions,
        'sampling_interface': {
            'single_shape': list(single_a.shape),
            'multiple_shape': list(multi_a.shape),
            'same_key_reproducible': bool(np.array_equal(multi_a, repeat_a)),
            'all_finite': bool(np.isfinite(single_a).all()
                               and np.isfinite(multi_a).all()),
            'inside_action_bounds': bool((multi_a >= -1).all()
                                         and (multi_a <= 1).all())},
    }
    report['models'][label] = model_report
    internals[label] = {
        'nll': -correct_lp, 'energy': energy_correct,
        'energy_episode': episodes[energy_rows], 'samples': sampled,
        'sampled_lower': sampled_lower_context.mean(axis=0).tolist(),
        'sampled_upper': sampled_upper_context.mean(axis=0).tolist(),
    }

  for specification in args.compare:
    left, separator, right = specification.partition(':')
    if not separator or left not in models or right not in models:
      raise ValueError(f'invalid --compare value: {specification}')
    report['comparisons'][specification] = {
        'nll': _paired_bootstrap(
            internals[left]['nll'], internals[right]['nll'], episodes,
            args.bootstrap_replicates, args.seed),
        'energy_score': _paired_bootstrap(
            internals[left]['energy'], internals[right]['energy'],
            internals[left]['energy_episode'], args.bootstrap_replicates,
            args.seed + 1000),
    }

  representatives = {}
  for region, mask in region_masks.items():
    candidates = np.flatnonzero(mask)
    if candidates.size:
      representatives[region] = int(candidates[len(candidates) // 2])
  first_model = next(iter(models.values()))
  mean = np.asarray(first_model.context_mean)
  std = np.asarray(first_model.context_std)
  normalized = (validation.state - mean) / std
  representative_samples = {label: {} for label in models}
  local_actions = {}
  report['representative_contexts'] = {}
  for region_index, (region, row) in enumerate(representatives.items()):
    distance = np.linalg.norm(normalized - normalized[row], axis=1)
    neighbors = np.argsort(distance)[:min(args.local_neighbors, len(distance))]
    local = validation.action[neighbors]
    local_actions[region] = local
    entry = {
        'validation_row': row, 'original_episode_id': int(episodes[row]),
        'context': validation.state[row].tolist(),
        'observed_action': validation.action[row].tolist(),
        'local_neighbor_count': int(len(neighbors)),
        'local_expert_pattern_frequencies': _pattern_frequencies(local),
        'models': {},
    }
    representative_rng = np.random.default_rng(args.seed + 10000 + region_index)
    representative_gaussian = representative_rng.standard_normal(
        (1, args.representative_samples, dataset.action_dim), dtype=np.float32)
    representative_uniform = representative_rng.random(
        (1, args.representative_samples), dtype=np.float32)
    for label, model in models.items():
      samples, _ = _fixed_samples(
          model, validation.state[row:row + 1], representative_gaussian,
          representative_uniform)
      samples = samples[0]
      representative_samples[label][region] = samples
      generated_patterns = _pattern_frequencies(samples)
      observed_patterns = entry['local_expert_pattern_frequencies']
      supported = sorted(name for name, frequency in observed_patterns.items()
                         if frequency >= 0.05)
      missing = sorted(name for name in supported
                       if generated_patterns.get(name, 0.0) < 0.01)
      nearest = np.min(np.linalg.norm(
          local[:, None, :] - samples[None, :, :], axis=-1), axis=1)
      entry['models'][label] = {
          'sample_summary': _sample_summary(samples),
          'generated_pattern_frequencies': generated_patterns,
          'supported_expert_patterns': supported,
          'missing_supported_expert_patterns': missing,
          'local_expert_to_generated_nearest_l2_mean': float(nearest.mean()),
          'local_expert_to_generated_nearest_l2_p90': float(
              np.quantile(nearest, 0.9)),
      }
    report['representative_contexts'][region] = entry

  if args.plot_labels:
    plot_labels = [x for x in args.plot_labels.split(',') if x]
  else:
    plot_labels = [label for label, model in models.items()
                   if model.metadata.get('population') == 'expert_positive']
  if not plot_labels:
    plot_labels = list(models)[:2]
  if not all(label in models for label in plot_labels):
    raise ValueError('--plot-labels names an unknown model')
  model_boundaries = {
      label: {'sampled_lower': internals[label]['sampled_lower'],
              'sampled_upper': internals[label]['sampled_upper']}
      for label in plot_labels}
  report['plots'] = _make_plots(
      args.out_dir, plot_labels, models, representatives, local_actions,
      representative_samples, observed_boundary, model_boundaries)
  report['all_checks_pass'] = bool(
      all(item['sampling_interface']['same_key_reproducible']
          and item['sampling_interface']['all_finite']
          and item['sampling_interface']['inside_action_bounds']
          for item in report['models'].values())
      and all(item['evaluation_episodes_all_in_model_validation']
              for item in split_checks.values()))
  output_path = os.path.join(args.out_dir, 'evaluation.json')
  _write_json(output_path, report)
  print(f'expert validation: {len(validation_episodes)} episodes, '
        f'{len(validation.state)} transitions')
  for label, value in report['models'].items():
    print(f'{label}: NLL={value["nll"]["correct_context_nats_per_action_vector"]:.6f} '
          f'shuffled={value["nll"]["shuffled_context_nats_per_action_vector"]:.6f} '
          f'energy={value["conditional_energy_score"]["correct_context"]:.6f} '
          f'any-boundary={value["boundary"]["sampled"]["any_boundary_coordinate"]:.4f}')
  for label, value in report['comparisons'].items():
    print(f'{label}: delta NLL={value["nll"]["left_minus_right"]:+.6f} '
          f'CI={value["nll"]["ci95"]}')
  print(f'wrote {output_path}; checks={report["all_checks_pass"]}')
  return report


if __name__ == '__main__':
  main()
