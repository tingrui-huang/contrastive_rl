"""Frozen-checkpoint diagnostic of full-F4 MMD versus nearest-reference cost.

No optimizer, new training targets, parameter updates, or A/B/C training runs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ett.anchored_transition import load_anchored, conditional_mmd2
from ett.dataset import ExpertTransitionDataset
from ett.diagonal_transition import _project_samples
from ett.eval_diagonal_transition import _open_endpoint
from ett.failure_objectives import (
    mmd_components, nearest_reference, conditional_set_cost, shift_with_newest)
from ett.run_distribution_matching import (
    load_references, frozen_actor, kernel_config, kernel_args, write_json)
from ett.train_diagonal_transition import _git_provenance
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha


DEFAULT_RUN = 'artifacts/ett_distribution_matching/f4_p30_s01_guarded'


def read(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def concatenate(parts):
  return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def observable_groups(arrays):
  frozen = (arrays.timestep > 0) & np.all(
      arrays.state.reshape(-1, 4, 2) == arrays.state[:, None, :2], axis=(1, 2))
  moving = np.linalg.norm(arrays.state[:, :2] - arrays.state[:, 2:4], axis=-1) > .05
  return {'all': np.ones(len(frozen), bool), 'frozen_nonreset': frozen,
          'moving': moving, 'moving_upper_corridor': moving & (arrays.state[:, 1] > 2.5),
          'moving_lower_corridor': moving & (arrays.state[:, 1] < 2.),
          'reset': arrays.timestep == 0,
          'near_commanded_goal': np.linalg.norm(arrays.state[:, :2]-arrays.goal[:, :2], axis=-1) < .5}


def group_summary(values, groups):
  return {group: {'contexts': int(mask.sum()), **{
      key: float(np.asarray(value)[mask].mean()) for key, value in values.items()}}
          for group, mask in groups.items() if mask.any()}


def paired_bootstrap(delta, episodes, mask, seed=818, repetitions=500):
  """Context-weighted effect, clustering complete represented episode groups.

  First average Monte Carlo replicates per saved context. Bootstrap episodes,
  preserving the number of selected contexts per episode via sums/counts.
  This is conditional on the fixed bank and models, not a new test set.
  """
  delta = np.asarray(delta)[:, mask]
  point = delta.mean(0)
  unique, inverse = np.unique(episodes[mask], return_inverse=True)
  sums = np.bincount(inverse, weights=point)
  counts = np.bincount(inverse)
  draws = np.random.default_rng(seed).integers(len(unique), size=(repetitions, len(unique)))
  estimates = sums[draws].sum(1) / counts[draws].sum(1)
  return {'mean': float(point.mean()), 'ci95_episode': np.quantile(estimates, [.025, .975]),
          'sampling_seed_means': delta.mean(1), 'episodes': len(unique),
          'contexts': len(point), 'bootstrap_replicates': repetitions}


def geometry(state, samples, anchor=None, radius=None):
  xy = np.asarray(samples)[..., :2]
  current = state[:, None, :2]
  delta = xy-current
  path = current[:, :, None, :] + np.linspace(0., 1., 41)[None, None, :, None] * delta[:, :, None, :]
  result = {
      'endpoint_invalid_fraction': (~_open_endpoint(xy)).mean(1),
      'segment_blocked_fraction': (~_open_endpoint(path)).any(-1).mean(1),
      'one_step_cap_violation_fraction': (np.abs(delta) > 1.+1e-6).any(-1).mean(1),
      'history_shift_error': np.abs(np.asarray(samples)[..., 2:]-state[:, None, :6]).max((1, 2)),
      'motion': np.linalg.norm(delta, axis=-1).mean(1),
      'stationary_fraction': np.all(delta == 0, axis=-1).mean(1),
      'near_stationary_fraction': (np.linalg.norm(delta, axis=-1) <= .05).mean(1),
      'xy_std_norm': np.linalg.norm(xy.std(1), axis=-1),
  }
  if anchor is not None:
    result['anchor_bound_violation_fraction'] = (
        np.linalg.norm(xy-anchor, axis=-1) > radius+1e-7).mean(1)
  return result


def make_evaluator(references, kernel):
  mean, std, bandwidths = kernel_args(kernel)
  @jax.jit
  def numeric(samples):
    parts = mmd_components(samples, references, mean, std, bandwidths)
    detail = nearest_reference(samples, references, mean, std)
    parts.update({'set_cost': detail['distance2'].mean(1),
                  'set_newest': detail['newest_distance2'].mean(1),
                  'set_history': detail['history_distance2'].mean(1),
                  'history_floor': detail['history_floor'].mean(1),
                  'nearest_tie_fraction': (detail['exact_tie_count'] > 1).mean(1)})
    return parts, detail

  def evaluate(samples, state, anchor=None, radius=None):
    parts_list, detail_list = [], []
    for start in range(0, len(state), 128):
      part, detail = numeric(jnp.asarray(samples[start:start+128]))
      parts_list.append({k: np.asarray(v) for k, v in part.items()})
      detail_list.append({k: np.asarray(v) for k, v in detail.items()})
    parts, detail = concatenate(parts_list), concatenate(detail_list)
    parts.update(geometry(state, samples, anchor, radius))
    target_xy = np.asarray(references)[detail['index'], :2]
    target = np.asarray(shift_with_newest(jnp.asarray(state), jnp.asarray(target_xy)))
    target_geometry = geometry(state, target, anchor, radius)
    parts.update({'reference_' + k: v for k, v in target_geometry.items()
                  if k in ('segment_blocked_fraction', 'one_step_cap_violation_fraction', 'anchor_bound_violation_fraction', 'motion')})
    if not all(np.isfinite(v).all() for v in parts.values()):
      raise RuntimeError('non-finite objective or geometry diagnostic')
    return parts, detail
  return evaluate


def sample_models(model, nominal, arrays, actions, count, replicate):
  @jax.jit
  def draw(state, action, goal, key):
    return model.marginal_flat(model.params, nominal, state, action, goal, key, count)
  samples, details = [], []
  for start in range(0, len(actions), 128):
    _, original_key = jax.random.split(jax.random.PRNGKey(7100+start))
    key = original_key if replicate == 0 else jax.random.fold_in(original_key, replicate)
    y, detail = draw(arrays.state[start:start+128], actions[start:start+128], arrays.goal[start:start+128], key)
    samples.append(np.asarray(y))
    details.append({k: np.asarray(v) for k, v in detail.items()})
  return np.concatenate(samples), concatenate(details)


def formula_checks(references, kernel):
  mean, std, bandwidths = kernel_args(kernel)
  f = jnp.asarray(references)
  exact = nearest_reference(f, f, mean, std)['distance2']
  y = f[:1, None, :].repeat(16, axis=1)
  duplicated = jnp.concatenate([f, f[:1].repeat(len(f), axis=0)])
  set_original = float(conditional_set_cost(y, f, mean, std)[0])
  set_duplicated = float(conditional_set_cost(y, duplicated, mean, std)[0])
  original_mmd = float(conditional_mmd2(y, f, mean, std, bandwidths)[0])
  reweighted_mmd = float(conditional_mmd2(y, duplicated, mean, std, bandwidths)[0])
  if not np.all(np.asarray(exact) == 0) or set_original != 0 or set_duplicated != 0:
    raise RuntimeError('exact membership/duplicate formula check failed')
  return {'scope': 'formula checks on training bank vectors, not held-out behavioral evidence',
          'exact_membership_max_cost': float(exact.max()),
          'repeated_single_reference_set_cost': set_original,
          'duplicating_first_reference_set_cost': set_duplicated,
          'repeated_single_reference_mmd': original_mmd,
          'reweighted_reference_mmd': reweighted_mmd,
          'reweighting': f'append {len(f)} exact copies of the first reference in memory; bank file unchanged',
          'unique_reference_vectors': len(np.unique(np.asarray(f), axis=0))}


def gradient_diagnostic(samples, arrays, references, kernel, groups, displacement=None):
  mean, std, bandwidths = kernel_args(kernel)
  result, gradients = {}, {}
  for name in ('mmd', 'set'):
    @jax.jit
    def derivative(y):
      function = (lambda v: conditional_mmd2(v, references, mean, std, bandwidths).sum()) if name == 'mmd' else (
          lambda v: conditional_set_cost(v, references, mean, std).sum())
      return jax.grad(function)(y)
    grad = np.concatenate([np.asarray(derivative(jnp.asarray(samples[i:i+128])))[..., :2]
                           for i in range(0, len(samples), 128)])
    if not np.isfinite(grad).all():
      raise RuntimeError('non-finite newest-XY objective gradients')
    gradients[name] = grad
    values = {'translation_gradient_norm': np.linalg.norm(grad.sum(1), axis=-1),
              'mean_per_sample_gradient_norm_scaled_by_K': np.linalg.norm(grad*grad.shape[1], axis=-1).mean(1),
              'zero_translation_gradient_fraction': (np.linalg.norm(grad.sum(1), axis=-1) < 1e-8).astype(float)}
    if displacement is not None:
      slope = (grad*displacement).sum((1, 2))
      values['actual_displacement_directional_derivative'] = slope
      values['actual_displacement_is_descent_fraction'] = (slope < -1e-8).astype(float)
    result[name] = group_summary(values, groups)
  return result, gradients


def reference_switch_probe(arrays, references, kernel, groups):
  mean, std, _ = kernel_args(kernel)
  s = jnp.asarray(arrays.state)
  center = shift_with_newest(s, s[:, None, :2])
  base = nearest_reference(center, references, mean, std)
  center_index = np.asarray(base['index'])[:, 0]
  directions = np.array([[1,0], [-1,0], [0,1], [0,-1]], np.float32)
  results, cases = {}, []
  for epsilon in (0.0001, 0.001, 0.01, 0.05):
    y = shift_with_newest(s, s[:, None, :2] + epsilon*jnp.asarray(directions)[None])
    detail = nearest_reference(y, references, mean, std)
    index = np.asarray(detail['index'])
    changes = index != center_index[:, None]
    selected_xy = np.asarray(references)[index, :2]
    base_xy = np.asarray(references)[center_index, :2]
    # Jump in raw-XY derivative of the selected quadratic branch at a boundary.
    derivative_jump = np.linalg.norm(2*(selected_xy-base_xy[:, None]) / np.asarray(std[:2])**2, axis=-1)
    stats = {'reference_switch_fraction': changes.mean(1),
             'any_direction_switch_fraction': changes.any(1).astype(float),
             'selected_reference_gradient_jump': derivative_jump.mean(1)}
    results[str(epsilon)] = group_summary(stats, groups)
    if epsilon == .01:
      switched = np.flatnonzero(changes.any(1))
      for i in switched[:8]:
        cases.append({'context_index': int(i), 'episode_id': int(arrays.episode_id[i]),
                      'timestep': int(arrays.timestep[i]), 'reference_before': int(center_index[i]),
                      'references_after': index[i], 'directions': directions})
  return {'epsilon_units': 'raw maze XY, cardinal perturbations with old frames fixed; unprojected diagnostic',
          'scales': results, 'first_changed_examples_in_saved_context_order': cases}


def controlled_candidates(arrays, references, kernel, primary_samples, primary_detail,
                          control_samples, evaluate, groups, spec):
  count = primary_samples.shape[1]
  s = jnp.asarray(arrays.state)
  persistence = np.asarray(shift_with_newest(s, jnp.repeat(s[:, None, :2], count, axis=1)))
  anchor, radius = primary_detail['anchor_xy'], primary_detail['radius']
  candidates = {'persistence': persistence, 'diagonal_only': control_samples,
                'mmd_trained': primary_samples}
  corrections = {}
  for epsilon in (.01, .05):
    for direction, vector in (('right', [1., 0.]), ('left', [-1., 0.]), ('up', [0., 1.]), ('down', [0., -1.])):
      delta = jnp.broadcast_to(epsilon*jnp.asarray(vector), (len(s), count, 2))
      value, detail = _project_samples(s, delta, spec)
      name = f'shift_{direction}_{epsilon:g}'
      candidates[name] = np.asarray(value)
      corrections[name] = np.any(np.abs(np.asarray(detail['raw_position'])-np.asarray(value)[..., :2]) > 1e-6, axis=-1).mean(1)
  angles = np.arange(count)*2*np.pi/count
  delta = jnp.broadcast_to(jnp.asarray(.05*np.stack([np.cos(angles), np.sin(angles)], -1)), (len(s), count, 2))
  jitter, _ = _project_samples(s, delta, spec)
  candidates['symmetric_jitter_radius_0.05'] = np.asarray(jitter)
  nearest = nearest_reference(jnp.asarray(persistence), references, *kernel_args(kernel)[:2])
  # The history-only nearest reference gives the unconstrained minimum possible
  # with fixed old frames. It need not be reachable or respect an action bound.
  target = np.asarray(references)[np.asarray(nearest['history_nearest_index']), :2]
  candidates['history_nearest_collapse_raw'] = np.asarray(shift_with_newest(s, jnp.asarray(target)))
  projected, _ = _project_samples(s, jnp.asarray(target)-s[:, None, :2], spec)
  candidates['history_nearest_collapse_projected'] = np.asarray(projected)
  values, all_parts = {}, {}
  for name, y in candidates.items():
    part, _ = evaluate(y, arrays.state, anchor, radius)
    if name in corrections:
      part['candidate_projection_fraction'] = corrections[name]
    values[name], all_parts[name] = group_summary(part, groups), part
  comparisons = {}
  for name, part in all_parts.items():
    comparisons[name] = group_summary({
        'set_lower_than_persistence_fraction': (part['set_cost'] < all_parts['persistence']['set_cost']-1e-8).astype(float),
        'mmd_lower_than_persistence_fraction': (part['mmd'] < all_parts['persistence']['mmd']-1e-8).astype(float),
        'set_change_from_persistence': part['set_cost']-all_parts['persistence']['set_cost'],
        'mmd_change_from_persistence': part['mmd']-all_parts['persistence']['mmd']}, groups)
  return {'K': count, 'candidates': values, 'versus_persistence': comparisons,
          'candidate_bound_anchor': 'each primary trained draw own diagonal anchor, with its sampled x_prime radius',
          'construction': 'history shift exact; small perturbations/projected collapse use existing endpoint projection; raw collapse is explicitly unprojected'}, candidates, all_parts


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-run', default=DEFAULT_RUN)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--sampling-seeds', default='0,1,2,3')
  parser.add_argument('--sample-counts', default='8,16')
  parser.add_argument('--verified-remote-head', default=None)
  parser.add_argument('--max-contexts', type=int, default=None, help='smoke only; default uses all saved contexts')
  args = parser.parse_args()
  out, source = Path(args.out_dir), Path(args.source_run)
  out.mkdir(parents=True, exist_ok=True)
  if (out/'config.json').exists():
    raise ValueError('use a fresh output directory; previous artifacts are preserved')
  started = time.time()
  old_config, old_metrics = read(source/'config.json'), read(source/'metrics.json')
  run_args = old_config['arguments']
  checkpoint_paths = {name: source/name/'final.pkl' for name in old_metrics['arms']}
  inputs = [Path(run_args['dataset']), Path(run_args['bank']), Path(run_args['nominal'])/'best.pkl', Path(run_args['actor'])] + list(checkpoint_paths.values())
  missing = [str(p) for p in inputs if not p.is_file()]
  if missing:
    write_json(out/'blocked.json', {'missing_dependencies': missing, 'training_attempted': False,
                                   'unrun': 'all checkpoint-based empirical diagnostics'})
    raise FileNotFoundError(f'required existing artifacts are unavailable: {missing}')
  # Include source-run configuration, metrics, figures and all checkpoints in
  # the immutability audit, not just the four models used in a given comparison.
  immutable = list(source.rglob('*')) + inputs
  hashes = {str(p).replace('\\', '/'): file_sha(p) for p in immutable if p.is_file()}
  for p, expected in old_config['input_checkpoint_sha256'].items():
    if file_sha(p) != expected:
      raise ValueError(f'frozen input checkpoint differs: {p}')
  if file_sha(run_args['dataset']) != old_config['dataset']['dataset']['sha256']:
    raise ValueError('dataset file differs from authoritative run')
  for name, path in checkpoint_paths.items():
    if file_sha(path) != old_metrics['arms'][name]['checkpoint_sha256']:
      raise ValueError(f'ETT checkpoint differs: {path}')
  dataset = ExpertTransitionDataset(run_args['dataset'])
  if dataset.report['split'] != old_config['dataset']['split']:
    raise ValueError('established episode split changed')
  references, reference_report = load_references(run_args['bank'], dataset)
  for name in ('bank_content_sha256', 'kept_episode_ids', 'kept_rows'):
    if not np.array_equal(reference_report[name], old_config['references'][name]):
      raise ValueError(f'failure references changed: {name}')
  kernel = old_config['kernel']
  recomputed = kernel_config(dataset.arrays('train'))
  for name in ('mean', 'std', 'bandwidths'):
    if not np.array_equal(np.asarray(kernel[name], np.float32), recomputed[name]):
      raise ValueError(f'training-derived normalization/kernel does not reproduce: {name}')
  validation = dataset.arrays('validation')
  selected = read(source/'evaluation_contexts.json')
  lookup = {(int(e), int(t)): i for i, (e,t) in enumerate(zip(validation.episode_id, validation.timestep))}
  indices = np.asarray([lookup[(e,t)] for e,t in zip(selected['episode_id'], selected['timestep'])])
  if args.max_contexts is not None:
    indices = indices[:args.max_contexts]
  arrays = validation.take(indices)
  groups = observable_groups(arrays)
  if args.max_contexts is None and groups['frozen_nonreset'].sum() != 91:
    raise ValueError('the authoritative 91 frozen non-reset contexts did not reproduce')
  nominal = load_nominal_policy(run_args['nominal'])
  actor, actor_info = frozen_actor(run_args['actor'], reference_report['dataset_content_sha256'])
  models = {name: load_anchored(path) for name, path in checkpoint_paths.items()}
  counts = [int(x) for x in args.sample_counts.split(',')]
  repetitions = [int(x) for x in args.sampling_seeds.split(',')]
  if not counts or min(counts) < 2 or 16 not in counts or 0 not in repetitions or min(repetitions) < 0:
    raise ValueError('include K=16 and sampling seed 0; K must be at least 2 and seeds nonnegative')
  authority = {'local_head': _git_provenance(), 'verified_remote_head': args.verified_remote_head,
               'selected_run': str(source),
               'sibling_run_statuses': {str(p.parent): read(p).get('status') for p in source.parent.glob('*/config.json')},
               'newer_authoritative_run_found': False}
  config = {'arguments': vars(args), 'command': 'python -m ett.diagnose_failure_objectives ' + ' '.join(sys.argv[1:]),
            'authority': authority, 'source_config_sha256': file_sha(source/'config.json'),
            'input_file_sha256': hashes, 'dataset_split': dataset.report['split'],
            'references': reference_report, 'kernel': kernel, 'actor': actor_info,
            'evaluation_episode_ids': arrays.episode_id, 'evaluation_timesteps': arrays.timestep,
            'groups': {name: {'contexts': int(mask.sum()), 'episodes': len(np.unique(arrays.episode_id[mask]))} for name, mask in groups.items()},
            'training_or_parameter_updates': False, 'hidden_labels_loaded': False,
            'sampling': 'actor fixed across models/K/replicates; nominal actions and transition draws use paired keys; seed 0 reproduces original keys',
            'ties': 'first argmin in saved bank order; selected quadratic branch derivative',
            'runtime': {'jax': jax.__version__, 'devices': list(map(str, jax.devices()))},
            'source_file_sha256': {p: file_sha(p) for p in ('ett/failure_objectives.py', 'ett/diagnose_failure_objectives.py', 'scripts/test_failure_objectives.py')},
            'status': 'running'}
  write_json(out/'config.json', config)
  write_json(out/'formula_checks.json', formula_checks(references, kernel))
  references = jnp.asarray(references)
  actions = []
  for start in range(0, len(arrays.state), 128):
    key, _ = jax.random.split(jax.random.PRNGKey(7100+start))
    actions.append(np.asarray(actor(arrays.state[start:start+128], arrays.goal[start:start+128], key)))
  actions = np.concatenate(actions)
  evaluate = make_evaluator(references, kernel)
  primary_name, control_name = 's0_L0p25_lambda1', 's0_L0p25_lambda0'
  all_parts, aggregate, retained = {}, {}, {}
  for name, model in models.items():
    all_parts[name], aggregate[name] = {}, {}
    for count in counts:
      repeated = []
      for replicate in repetitions:
        y, detail = sample_models(model, nominal, arrays, actions, count, replicate)
        part, nn = evaluate(y, arrays.state, detail['anchor_xy'], detail['radius'])
        if max(part['history_shift_error']) != 0 or max(part['endpoint_invalid_fraction']) != 0 or max(part['anchor_bound_violation_fraction']) != 0:
          raise RuntimeError('loaded model sampling contract failed')
        if replicate == 0 and args.max_contexts is None:
          expected = old_metrics['arms'][name]['marginal'][str(count)]['mmd2']
          if abs(part['mmd'].mean()-expected) > 2e-6:
            raise RuntimeError(f'authoritative MMD does not reproduce for {name}, K={count}')
        repeated.append(part)
        if replicate == 0 and count == 16:
          retained[name] = (y, detail, nn)
      all_parts[name][count] = {k: np.stack([p[k] for p in repeated]) for k in repeated[0]}
      aggregate[name][str(count)] = {'mean_over_sampling_seeds': group_summary({k: v.mean(0) for k,v in all_parts[name][count].items()}, groups),
                                     'per_sampling_seed': [group_summary(part, groups) for part in repeated]}
      print(f'{name} K={count} MMD={all_parts[name][count]["mmd"].mean():.6f} set={all_parts[name][count]["set_cost"].mean():.6f}', flush=True)
  write_json(out/'sampled_objectives.json', aggregate)
  paired = {}
  for name in models:
    if old_metrics['arms'][name]['lambda_pess'] == 0:
      continue
    seed = old_metrics['arms'][name]['seed']
    control = f's{seed}_L0p25_lambda0'
    paired[name] = {}
    for count in counts:
      left, right = all_parts[name][count], all_parts[control][count]
      differences = {k: left[k]-right[k] for k in ('gg', 'rr', 'gr', 'mmd', 'set_cost', 'mmd_generated_u')}
      differences['cross_contribution'] = -2*differences['gr']
      residual = differences['mmd'] - (differences['gg']+differences['rr']+differences['cross_contribution'])
      if np.max(np.abs(residual)) > 1e-6:
        raise RuntimeError('paired MMD decomposition identity failed')
      paired[name][str(count)] = {group: {k: paired_bootstrap(v, arrays.episode_id, mask) for k,v in differences.items()}
                                 for group, mask in groups.items() if mask.any()}
  write_json(out/'paired_changes.json', paired)
  # Reproduce the separate original 91-context, K=64 frozen-history check.
  frozen_arrays = arrays.take(np.flatnonzero(groups['frozen_nonreset']))
  frozen_replay = {}
  frozen_actions = actor(frozen_arrays.state, frozen_arrays.goal, jax.random.PRNGKey(413))
  original_frozen = read(source/'frozen_history_evaluation.json')
  for name, model in models.items():
    y = np.asarray(model.sample_marginal(nominal, frozen_arrays.state, frozen_actions, jax.random.PRNGKey(414), 64, frozen_arrays.goal))
    stationary = float(np.all(y[..., :2] == frozen_arrays.state[:, None, :2], axis=-1).mean())
    frozen_replay[name] = stationary
    if args.max_contexts is None and abs(stationary-original_frozen['arms'][name]['exact_zero_fraction']) > 1e-10:
      raise RuntimeError('original frozen-history sampler metrics changed')
  write_json(out/'baseline_reproduction.json', {'seed_0_MMD_reproduced': args.max_contexts is None,
      'frozen_stationary_fraction_K64': frozen_replay, 'frozen_contexts': len(frozen_arrays.state),
      'observed_stationarity_is_not_a_death_label': True})
  primary_y, primary_detail, primary_nn = retained[primary_name]
  controlled, candidates, candidate_parts = controlled_candidates(arrays, references, kernel, primary_y,
      primary_detail, retained[control_name][0], evaluate, groups, models[primary_name].diagonal.spec)
  write_json(out/'controlled_candidates.json', controlled)
  interpolation = {}
  displacement = primary_y[..., :2]-primary_detail['anchor_xy']
  anchor_state = np.asarray(shift_with_newest(jnp.asarray(arrays.state), jnp.asarray(primary_detail['anchor_xy'])))
  for t in (0., .25, .5, .75, 1.):
    value = np.asarray(shift_with_newest(jnp.asarray(arrays.state), jnp.asarray(primary_detail['anchor_xy']+t*displacement)))
    part, _ = evaluate(value, arrays.state, primary_detail['anchor_xy'], primary_detail['radius'])
    interpolation[str(t)] = group_summary(part, groups)
  write_json(out/'anchor_to_generated.json', {'primary': primary_name, 'K': 16, 'sampling_seed': 0,
      'definition': 'linear interpolation along each actual paired own-diagonal-anchor to emitted displacement; no projection applied along the path',
      'fractions': interpolation})
  gradients, raw_gradients = {}, {}
  for name, y in [('persistence', candidates['persistence']), ('own_diagonal_anchor', anchor_state), ('mmd_trained', primary_y)]:
    gradients[name], raw_gradients[name] = gradient_diagnostic(y, arrays, references, kernel, groups,
        displacement if name != 'persistence' else None)
  write_json(out/'gradients.json', gradients)
  write_json(out/'reference_switches.json', reference_switch_probe(arrays, references, kernel, groups))
  heldout_membership = nearest_reference(jnp.asarray(candidates['persistence']), references, *kernel_args(kernel)[:2])
  write_json(out/'heldout_membership.json', group_summary({
      'persistence_exact_bank_membership': (np.asarray(heldout_membership['distance2'])[:, 0] == 0).astype(float),
      'persistence_nearest_squared_distance': np.asarray(heldout_membership['distance2'])[:, 0]}, groups))
  # Selected source episode and row accompany every nearest-reference index.
  write_json(out/'primary_nearest_references.json', {
      'indices_in_filtered_bank': primary_nn['index'],
      'source_episode_id': np.asarray(reference_report['kept_episode_ids'])[primary_nn['index']],
      'source_row': np.asarray(reference_report['kept_rows'])[primary_nn['index']],
      'evaluation_episode_id': arrays.episode_id, 'evaluation_timestep': arrays.timestep,
      'K': 16, 'sampling_seed': 0})
  # Compact per-context numerical records permit recomputing paired statistics.
  write_json(out/'per_context_objectives.json', {
      name: {str(k): {key: value for key,value in values.items() if key in ('gg','rr','gr','mmd','set_cost','mmd_generated_u')}
             for k,values in by_k.items()} for name,by_k in all_parts.items()})
  # Numerical arrays for independent figure rendering; these are evaluation
  # samples, not training data or checkpoints, and contain no hidden labels.
  np.savez_compressed(out/'visualization_samples.npz', state=arrays.state,
      goal=arrays.goal, action=actions, episode_id=arrays.episode_id, timestep=arrays.timestep,
      frozen=groups['frozen_nonreset'], moving=groups['moving'], references=np.asarray(references),
      primary=primary_y, control=retained[control_name][0], anchor=primary_detail['anchor_xy'],
      radius=primary_detail['radius'], mean=np.asarray(kernel['mean']), std=np.asarray(kernel['std']),
      bandwidths=np.asarray(kernel['bandwidths']),
      persistence_mmd_gradient=raw_gradients['persistence']['mmd'].sum(1),
      persistence_set_gradient=raw_gradients['persistence']['set'].sum(1))
  config['status'] = 'complete'
  config['elapsed_seconds'] = time.time()-started
  config['all_original_artifacts_unchanged'] = all(file_sha(p) == expected for p,expected in hashes.items())
  if not config['all_original_artifacts_unchanged']:
    raise RuntimeError('an original artifact changed during the diagnostic')
  write_json(out/'config.json', config)
  print(f'completed objective diagnostic in {time.time()-started:.1f}s; no model updates', flush=True)


if __name__ == '__main__':
  main()
