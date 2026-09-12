"""Bounded single-step ETT experiment; freeze both action policies, train ETT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
import jax
import jax.numpy as jnp
import numpy as np
import optax

from crl import checkpoint, networks
from ett.anchored_transition import (
    AnchoredTransition, conditional_mmd2, gaussian_kernel, save_anchored, load_anchored)
from ett.dataset import ExpertTransitionDataset
from ett.diagonal_transition import load_diagonal_transition
from ett.eval_diagonal_transition import (
    _log_prob, _stochastic_samples, _energy_score, _calibration, _open_endpoint,
    _paired_episode_bootstrap)
from ett.train_diagonal_transition import _git_provenance
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import content_sha, file_sha


DATA = 'artifacts/f4_p30_server_30076/results'


def write_json(path, value):
  def convert(v):
    if isinstance(v, np.ndarray):
      return v.tolist()
    if isinstance(v, np.generic):
      return v.item()
    if isinstance(v, Path):
      return str(v).replace('\\', '/')
    raise TypeError(type(v).__name__)
  Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                  allow_nan=False) + '\n', encoding='utf-8')


def load_references(path, dataset):
  """Filter the existing bank only; no hidden-state target construction."""
  with np.load(path, allow_pickle=False) as source:
    meta = json.loads(str(source['meta']))
    states = np.asarray(source['goals'], np.float32)
    episodes = np.asarray(source['episode_id'], np.int64)
    rows = np.asarray(source['failure_row'], np.int64)
    classes = np.asarray(source['behaviour_class'])
    schema = {name: {'shape': list(source[name].shape),
                      'dtype': str(source[name].dtype)} for name in source.files}
  digest = content_sha(dataset.path)
  if meta['source_content_sha256'] != digest:
    raise ValueError('failure bank source content hash differs from the dataset')
  if meta['env_name'] != dataset.metadata['env_name'] or states.shape[1:] != (8,):
    raise ValueError('failure bank environment or F4 schema mismatch')
  with np.load(dataset.path, allow_pickle=False) as source:
    observation = source['obs']
    n_episodes = len(observation)
    if not np.array_equal(states, observation[episodes, rows, :8]):
      raise ValueError('bank references differ from their original observed rows')
  # Reproduce BehaviorDataset._make_split on the full source. Keep references
  # from all training behavior sources, not just expert-source references.
  split = dataset.report['split']
  permutation = np.random.default_rng(split['seed']).permutation(n_episodes)
  validation = permutation[:round(split['validation_fraction'] * n_episodes)]
  training = np.setdiff1d(np.arange(n_episodes), validation)
  if not np.array_equal(np.intersect1d(training, np.arange(1200, 6000)),
                         dataset.train_episode_ids):
    raise ValueError('full-source split does not reproduce established expert split')
  keep = np.isin(episodes, training)
  if not keep.any():
    raise ValueError('no compatible training-episode failure references remain')
  return states[keep], {
      'schema': schema, 'original_metadata': meta, 'bank_file_sha256': file_sha(path),
      'bank_content_sha256': content_sha(path), 'dataset_content_sha256': digest,
      'original_count': len(states), 'kept_count': int(keep.sum()),
      'kept_episode_ids': episodes[keep], 'kept_rows': rows[keep],
      'removed_validation_episode_ids': episodes[~keep],
      'full_source_validation_episode_ids': validation,
      'reference_expert_validation_overlap': int(np.isin(episodes[keep], dataset.validation_episode_ids).sum()),
      'class_counts_after_filter': {str(c): int((classes[keep] == c).sum()) for c in np.unique(classes)},
      'selection': 'existing bank entries filtered by full-source episode split only',
      'privileged_construction': meta['selection_uses_privileged_fields'],
  }


def frozen_actor(path, dataset_content_sha):
  provenance = json.loads(Path(path).with_name('arm_provenance.json').read_text())
  if (provenance['dataset_content_sha256'] != dataset_content_sha
      or provenance['env_name'] != 'point_two_route_swamp_windy_f4_v0'):
    raise ValueError('actor checkpoint provenance is incompatible')
  step, state = checkpoint.load_checkpoint(path)
  network = networks.make_networks(
      obs_dim=8, goal_dim=8, action_dim=2, repr_dim=64, repr_norm=False,
      repr_norm_temp=True, hidden_layer_sizes=(256, 256), actor_min_std=1e-6,
      twin_q=False, use_image_obs=False, use_layer_norm=False, obs_scale=None)
  @jax.jit
  def sample(s, g, key):
    distribution = network.policy_network.apply(state.policy_params, jnp.concatenate([s, g], -1))
    return jnp.tanh(distribution.loc + distribution.scale *
                    jax.random.normal(key, distribution.loc.shape))
  return sample, {'step': int(step), 'provenance': provenance,
                   'sampling': 'frozen tanh Gaussian actor; one x fixed across K expert draws',
                   'evaluation_overlap': 'actor originally trained on all source episodes; ETT validation is not held out for actor'}


def kernel_config(train):
  value = train.next_state.astype(np.float64)
  mean, std = value.mean(0), value.std(0)
  std = np.where(std < 1e-6, 1., std)
  rng = np.random.default_rng(410)
  a, b = rng.integers(len(value), size=(2, 4096))
  distances = np.linalg.norm((value[a] - value[b]) / std, axis=-1)
  median = float(np.median(distances[distances > 0]))
  return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32),
          'bandwidths': np.asarray([.25, .5, 1., 2.], np.float32) * median,
          'median_training_pair_distance': median, 'bandwidth_seed': 410,
          'fit_split': 'expert training next-state F4 only; 4096 random pairs for bandwidth',
          'estimator': 'biased V-statistic with self pairs; mean of four Gaussian kernels'}


def kernel_args(config):
  return tuple(jnp.asarray(config[name]) for name in ('mean', 'std', 'bandwidths'))


def norm(tree):
  return float(optax.global_norm(tree))


def diagonal_metrics(base, arrays, samples):
  logp = _log_prob(base, arrays, 4096)
  generated, detail, atom = _stochastic_samples(base, arrays, samples, 2048, 1701)
  xy = generated[..., :2]
  energy = _energy_score(xy, arrays.next_state[:, :2])
  delta = xy - arrays.state[:, None, :2]
  exact = np.all(arrays.delta_xy == 0, axis=-1)
  return {
      'nll': float(-logp.mean()), 'mean_position_error': float(np.linalg.norm(
          xy.mean(1) - arrays.next_state[:, :2], axis=-1).mean()),
      'energy_score': float(energy.mean()), 'samples_per_context': samples,
      'atom_calibration': _calibration(atom, exact),
      'sample_stationary_fraction': float(np.all(delta == 0, axis=-1).mean()),
      'mean_xy_std': xy.std(1).mean(0),
      'geometry_violation_count': int((~_open_endpoint(xy)).sum()),
      'shift_max_error': float(np.abs(generated[..., 2:] - arrays.state[:, None, :6]).max()),
      'raw_displacement_capped_fraction': float(np.any(np.abs(detail['raw_delta']) > 1., axis=-1).mean()),
      'blocked_endpoint_fraction': float(detail['blocked_endpoint_before_projection'].mean()),
      'measure': 'unprojected mixed atom/maze-unit XY area; nats per 2D transition',
  }


def marginal_metrics(model, nominal, actor, arrays, references, kernel, count):
  @jax.jit
  def draw(state, goal, key):
    ka, kt = jax.random.split(key)
    action = actor(state, goal, ka)
    value, detail = model.marginal_flat(model.params, nominal, state, action, goal, kt, count)
    mmd = conditional_mmd2(value, references, *kernel_args(kernel))
    return value, detail, mmd
  generated, details, mmd = [], {}, []
  for start in range(0, len(arrays.state), 128):
    value, detail, distance = draw(arrays.state[start:start+128], arrays.goal[start:start+128],
                                  jax.random.PRNGKey(7100 + start))
    generated.append(np.asarray(value))
    mmd.append(np.asarray(distance))
    for name, item in detail.items():
      details.setdefault(name, []).append(np.asarray(item))
  generated = np.concatenate(generated)
  detail = {name: np.concatenate(value) for name, value in details.items()}
  mmd = np.concatenate(mmd)
  xy = generated[..., :2]
  motion = np.linalg.norm(xy - arrays.state[:, None, :2], axis=-1)
  historical_motion = np.linalg.norm(arrays.state[:, :2] - arrays.state[:, 2:4], axis=-1)
  moved = historical_motion > .05
  frozen_input = np.all(arrays.state.reshape(-1, 4, 2) == arrays.state[:, None, :2], axis=(-2, -1))
  mean, std, bandwidth = kernel_args(kernel)
  reference = (np.asarray(references) - np.asarray(mean)) / np.asarray(std)
  normalized = (generated - np.asarray(mean)) / np.asarray(std)
  cross_hist2 = np.square(normalized[:, :, None, 2:] - reference[None, None, :, 2:]).sum(-1)
  cross_new2 = np.square(normalized[:, :, None, :2] - reference[None, None, :, :2]).sum(-1)
  total = cross_hist2 + cross_new2
  saturations = {str(float(h)): float((np.exp(-total/(2*float(h)**2)) < 1e-6).mean()) for h in bandwidth}
  path = (arrays.state[:, None, None, :2] +
          np.linspace(0., 1., 21)[None, None, :, None] *
          (xy - arrays.state[:, None, :2])[:, :, None, :])
  # Inspect gradients in the two coordinates that the model can actually edit.
  emitted_gradient = jax.jit(jax.grad(lambda samples: conditional_mmd2(
      samples, references, *kernel_args(kernel)).sum()))(jnp.asarray(generated))
  newest_gradient_norm = np.linalg.norm(np.asarray(emitted_gradient)[..., :2].reshape(len(arrays.state), -1), axis=-1)
  stats = {
      'K': count, 'contexts': len(mmd), 'mmd2': float(mmd.mean()),
      'mean_motion': float(motion.mean()), 'near_zero_motion_fraction': float((motion <= .05).mean()),
      'exact_zero_motion_fraction': float((motion == 0).mean()),
      'fully_equal_f4_fraction': float(np.all(generated.reshape(-1, count, 4, 2) == xy[:, :, None, :], axis=(-2, -1)).mean()),
      'moving_history_mean_motion': float(motion[moved].mean()),
      'moving_history_near_zero_fraction': float((motion[moved] <= .05).mean()),
      'fully_frozen_input_contexts': int(frozen_input.sum()),
      'fully_frozen_input_mean_motion': float(motion[frozen_input].mean()) if frozen_input.any() else None,
      'fully_frozen_input_exact_zero_fraction': float((motion[frozen_input] == 0).mean()) if frozen_input.any() else None,
      'mean_xy_std': xy.std(1).mean(0),
      'low_diversity_fraction': float((np.linalg.norm(xy.std(1), axis=-1) < .001).mean()),
      'shift_max_error': float(np.abs(generated[..., 2:] - arrays.state[:, None, :6]).max()),
      'geometry_violation_count': int((~_open_endpoint(xy)).sum()),
      'straight_segment_blocked_fraction': float((~_open_endpoint(path)).any(-1).mean()),
      'final_bound_violation_count': int((detail['emitted_change'] > detail['radius'] + 1e-7).sum()),
      'max_bound_excess': float(np.maximum(detail['emitted_change'] - detail['radius'], 0).max()),
      'mean_anchor_change': float(detail['emitted_change'].mean()),
      'bound_utilization_mean': float((detail['emitted_change'] / np.maximum(detail['radius'], 1e-8)).mean()),
      'bound_fallback_fraction': float(detail['bound_fallback'].mean()),
      'base_geometry_correction_fraction': float(detail['base_corrected'].mean()),
      'candidate_geometry_correction_fraction': float(detail['candidate_corrected'].mean()),
      'candidate_blocked_fraction': float(detail['candidate_blocked'].mean()),
      'history_share_mean_cross_squared_distance': float(cross_hist2.sum()/total.sum()),
      'history_share_median_pair': float(np.median(cross_hist2/np.maximum(total, 1e-12))),
      'mean_nearest_reference_history_distance2': float(cross_hist2[:, 0].min(-1).mean()),
      'cross_kernel_saturation_fraction_by_bandwidth': saturations,
      'mean_editable_xy_mmd_gradient_norm': float(newest_gradient_norm.mean()),
      'editable_xy_gradient_below_1e8_fraction': float((newest_gradient_norm < 1e-8).mean()),
      'all_finite': bool(np.isfinite(generated).all() and np.isfinite(mmd).all()),
  }
  return stats, mmd, generated


def pair_examples(model, arrays):
  chosen = [int(np.argmin(np.abs(arrays.state[:, 0] - x))) for x in (1., 2.8, 3.7, 5., 7.)]
  output = []
  for index in chosen:
    state, goal, xp = arrays.state[index], arrays.goal[index], arrays.action[index]
    for name, action in [('identical', xp), ('nearby', np.clip(xp + [.01, -.01], -1, 1)), ('far', np.where(xp >= 0, -1., 1.))]:
      value, detail = model.sample_with_diagnostics(
          state, action, xp, jax.random.PRNGKey(331), 64, goal)
      output.append({'episode_id': int(arrays.episode_id[index]),
                     'timestep': int(arrays.timestep[index]), 'state': state,
                     'pair': name, 'x': action, 'x_prime': xp,
                     'action_distance': float(np.linalg.norm(action - xp)),
                     'mean_xy': np.asarray(value[..., :2]).mean(0),
                     'mean_anchor_xy': np.asarray(detail['anchor_xy']).mean(0),
                     'maximum_change': float(detail['emitted_change'].max()),
                     'radius': float(detail['radius'][0]),
                     'bound_fallback_fraction': float(detail['bound_fallback'].mean())})
  return output


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', default=DATA + '/datasets/swamp_windy_f4_merged_s0.npz')
  parser.add_argument('--bank', default=DATA + '/artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz')
  parser.add_argument('--diagonal', default='artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s0')
  parser.add_argument('--nominal', default='artifacts/nominal_policy/f4_p30_expert_only_mdn_k5_s0')
  parser.add_argument('--actor', default=DATA + '/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1/alpha0_seed0/final.pkl')
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--steps', type=int, default=500)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--eval-every', type=int, default=100)
  parser.add_argument('--eval-contexts', type=int, default=1024)
  parser.add_argument('--diagonal-samples', type=int, default=64)
  parser.add_argument('--bounds', default='0.25,0.75')
  parser.add_argument('--seeds', default='0,1')
  parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='residual Adam learning rate')
  parser.add_argument('--diagonal-learning-rate', type=float, default=1e-5)
  parser.add_argument('--weight', type=float, default=None,
                      help='otherwise choose 0.1 or 1 from training gradient magnitudes')
  parser.add_argument('--second-weight', type=float, default=None)
  parser.add_argument('--sanity-only', action='store_true')
  return parser


def main():
  args = build_parser().parse_args()
  if min(args.steps, args.batch_size, args.eval_every, args.eval_contexts, args.diagonal_samples) <= 0:
    raise ValueError('budgets must be positive')
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'config.json').exists():
    raise ValueError('output already contains a run; choose a new directory')
  started = time.time()
  dataset = ExpertTransitionDataset(args.dataset)
  train, validation = dataset.arrays('train'), dataset.arrays('validation')
  references, bank_report = load_references(args.bank, dataset)
  references = jnp.asarray(references)
  base = load_diagonal_transition(args.diagonal)
  nominal = load_nominal_policy(args.nominal)
  dataset_sha = file_sha(args.dataset)
  if base.metadata['dataset']['dataset']['sha256'] != dataset_sha or nominal.metadata['dataset']['sha256'] != dataset_sha:
    raise ValueError('a model checkpoint comes from a different dataset')
  for recorded in (base.metadata['dataset']['split'], nominal.metadata['split']):
    if set(recorded['validation_episode_ids']) != set(dataset.validation_episode_ids):
      raise ValueError('model checkpoint validation split differs')
  actor, actor_report = frozen_actor(args.actor, bank_report['dataset_content_sha256'])
  kernel = kernel_config(train)
  bounds = [float(x) for x in args.bounds.split(',')]
  seeds = [int(x) for x in args.seeds.split(',')]
  if not bounds or min(bounds) <= 0:
    raise ValueError('positive action bounds are required')
  config = {'arguments': vars(args), 'command': 'python -m ett.run_distribution_matching ' + ' '.join(sys.argv[1:]),
            'dataset': dataset.report, 'references': bank_report, 'kernel': kernel,
            'actor': actor_report, 'source': _git_provenance(),
            'source_file_sha256': {p: file_sha(p) for p in (
                'ett/anchored_transition.py', 'ett/run_distribution_matching.py',
                'scripts/test_ett_distribution_matching.py')},
            'input_checkpoint_sha256': {p: file_sha(p) for p in (
                str(Path(args.diagonal)/'best.pkl'), str(Path(args.nominal)/'best.pkl'), args.actor)},
            'runtime': {'python': sys.version, 'jax': jax.__version__, 'devices': list(map(str, jax.devices()))},
            'objective': 'L_diag + lambda_pess * L_pess; no other loss or weight decay',
            'gradient': 'conditional pathwise; no direct MMD gradient for discrete gates; biased partial expected-MMD gradient',
            'checkpoint_selection': 'final fixed budget primary; best diagonal-validation checkpoint also saved',
            'status': 'running'}
  write_json(out/'config.json', config)
  print(f'bank kept {len(references)}/{bank_report["original_count"]}; backend={jax.default_backend()}', flush=True)
  baseline = diagonal_metrics(base, validation, args.diagonal_samples)
  write_json(out/'baseline_diagonal.json', baseline)
  print(f'reproduced baseline NLL={baseline["nll"]:.6f} energy={baseline["energy_score"]:.6f}', flush=True)
  rng = np.random.default_rng(991)
  val_rows = rng.choice(len(validation.state), min(args.eval_contexts, len(validation.state)), replace=False)
  evaluation = validation.take(val_rows)
  write_json(out/'evaluation_contexts.json', {'episode_id': evaluation.episode_id, 'timestep': evaluation.timestep})
  primary = AnchoredTransition.initialize(base, jax.random.PRNGKey(seeds[0]), bounds[0])

  def losses(model, params, state, observed_action, next_state, goal, key):
    ka, kt = jax.random.split(key)
    action = actor(state, goal, ka)
    context = jnp.concatenate([state, observed_action, observed_action, goal], -1)
    diag = -base._log_prob(params['diagonal'], context, next_state[:, :2] - state[:, :2]).mean()
    samples, _ = model.marginal_flat(params, nominal, state, action, goal, kt, 8)
    pess = conditional_mmd2(samples, references, *kernel_args(kernel)).mean()
    return jnp.stack([diag, pess])

  def gradient_diagnostic(model, params, seed):
    rows = np.random.default_rng(seed).integers(len(train.state), size=args.batch_size)
    batch = (train.state[rows], train.action[rows], train.next_state[rows], train.goal[rows], jax.random.PRNGKey(seed))
    function = lambda p: losses(model, p, *batch)
    values = function(params)
    gradients = jax.jit(jax.jacrev(function))(params)
    diag_grad, pess_grad = [jax.tree_util.tree_map(lambda p: p[i], gradients) for i in (0, 1)]
    report = {'diagonal_loss': float(values[0]), 'mmd2': float(values[1]),
              'diagonal_gradient_norm': norm(diag_grad), 'mmd_gradient_norm': norm(pess_grad),
              'mmd_base_gradient_norm': norm(pess_grad['diagonal']),
              'mmd_residual_gradient_norm': norm(pess_grad['residual']),
              'finite': all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(gradients)),
              'training_row_seed': seed}
    if not report['finite'] or report['mmd_residual_gradient_norm'] <= 1e-10:
      raise RuntimeError(f'invalid training gradient path: {report}')
    return report

  sanity = gradient_diagnostic(primary, primary.params, 300)
  # A small grid chosen solely by initial training gradient magnitudes.
  weight = args.weight
  if weight is None:
    weight = .1 if .1*sanity['mmd_gradient_norm']/sanity['diagonal_gradient_norm'] >= .001 else 1.
  if weight <= 0 or (args.second_weight is not None and args.second_weight <= 0):
    raise ValueError('positive matching weights required')
  config['weight_selection'] = {'selected': weight, 'initial_training_diagnostic': sanity,
      'rule': 'choose 0.1 if weighted gradient ratio >=0.001, otherwise 1.0; explicit overrides recorded',
      'weighted_mmd_to_diag_gradient_ratio': weight*sanity['mmd_gradient_norm']/sanity['diagonal_gradient_norm']}
  write_json(out/'gradient_sanity.json', sanity)
  write_json(out/'config.json', config)
  print(f'training gradient sanity={sanity}; lambda={weight}', flush=True)
  initial_metrics = {}
  initial_mmd = {}
  for count in (8, 16):
    stat, mmd, _ = marginal_metrics(primary, nominal, actor, evaluation, references, kernel, count)
    initial_metrics[str(count)] = stat
    initial_mmd[count] = mmd
  write_json(out/'baseline_marginal.json', initial_metrics)

  arms = [(seeds[0], bounds[0], 0.)]
  if not args.sanity_only:
    arms += [(seeds[0], bound, weight) for bound in bounds]
    for seed in seeds[1:]:
      arms += [(seed, bounds[0], 0.), (seed, bounds[0], weight)]
    if args.second_weight is not None:
      arms += [(seeds[0], bounds[0], args.second_weight)]
  else:
    arms = [(seeds[0], bounds[0], weight)]
  results = {}
  for seed, bound, coefficient in arms:
    name = f's{seed}_L{bound:g}_lambda{coefficient:g}'.replace('.', 'p')
    directory = out/name
    directory.mkdir()
    model = AnchoredTransition.initialize(base, jax.random.PRNGKey(seed), bound)
    original_params = model.params
    optimizer = optax.multi_transform(
        {'diagonal': optax.adam(args.diagonal_learning_rate),
         'residual': optax.adam(args.learning_rate)},
        {'diagonal': 'diagonal', 'residual': 'residual'})
    opt_state = optimizer.init(model.params)
    @jax.jit
    def update(params, opt_state, state, action, next_state, goal, key):
      def total(p):
        terms = losses(model, p, state, action, next_state, goal, key)
        return terms[0] + coefficient * terms[1], terms
      (value, terms), gradients = jax.value_and_grad(total, has_aux=True)(params)
      changes, opt_state = optimizer.update(gradients, opt_state, params)
      return optax.apply_updates(params, changes), opt_state, value, terms, optax.global_norm(gradients)
    best_nll, best_step = baseline['nll'], 0
    save_anchored(directory/'best_diagonal.pkl', model, 0)
    history = []
    rng = np.random.default_rng(seed + 200)
    for step in range(1, args.steps + 1):
      rows = rng.integers(len(train.state), size=args.batch_size)
      model.params, opt_state, loss, terms, gradnorm = update(
          model.params, opt_state, train.state[rows], train.action[rows],
          train.next_state[rows], train.goal[rows], jax.random.PRNGKey(seed*100000+step))
      if not np.isfinite([float(loss), float(gradnorm)]).all():
        raise RuntimeError('non-finite loss or gradient')
      if step == 1 or step % args.eval_every == 0 or step == args.steps:
        base.params = model.params['diagonal']
        nll = float(-_log_prob(base, validation, 4096).mean())
        row = {'step': step, 'diagonal_training_nll': float(terms[0]), 'training_mmd2': float(terms[1]),
               'total_loss': float(loss), 'gradient_norm': float(gradnorm), 'validation_nll': nll}
        history.append(row)
        if nll < best_nll:
          best_nll, best_step = nll, step
          save_anchored(directory/'best_diagonal.pkl', model, step)
        write_json(directory/'training_history.json', history)
        print(f'{name} step={step} diag={float(terms[0]):.5f} mmd={float(terms[1]):.5f} val={nll:.5f}', flush=True)
    save_anchored(directory/'final.pkl', model, args.steps)
    changed = {part: norm(jax.tree_util.tree_map(lambda a,b: a-b, model.params[part], original_params[part]))
               for part in ('diagonal', 'residual')}
    if changed['diagonal'] <= 0 or (coefficient > 0 and changed['residual'] <= 0):
      raise RuntimeError('expected parameters did not change')
    result = {'seed': seed, 'L': bound, 'lambda_pess': coefficient, 'steps': args.steps,
              'parameter_change_norm': changed, 'best_diagonal_step': best_step,
              'best_diagonal_nll': best_nll, 'gradient_final': gradient_diagnostic(model, model.params, 300),
              'diagonal': diagonal_metrics(base, validation, args.diagonal_samples), 'marginal': {}}
    for count in (8, 16):
      stat, mmd, generated = marginal_metrics(model, nominal, actor, evaluation, references, kernel, count)
      stat['paired_episode_mmd_change_from_initial'] = _paired_episode_bootstrap(
          mmd, initial_mmd[count], evaluation.episode_id, 500, 86)
      result['marginal'][str(count)] = stat
      if not stat['all_finite'] or any(stat[k] != 0 for k in (
          'shift_max_error', 'geometry_violation_count', 'final_bound_violation_count')):
        raise RuntimeError('emitted sampling constraints failed')
    result['examples'] = pair_examples(model, evaluation)
    # Verify that the self-contained local checkpoint replays the same samples.
    restored = load_anchored(directory/'final.pkl')
    probe = evaluation.take(np.arange(min(8, len(evaluation.state))))
    key = jax.random.PRNGKey(123)
    left = model.sample(probe.state, -probe.action, probe.action, key, 8, probe.goal)
    right = restored.sample(probe.state, -probe.action, probe.action, key, 8, probe.goal)
    result['checkpoint_replay_exact'] = bool(np.array_equal(left, right))
    if not result['checkpoint_replay_exact']:
      raise RuntimeError('checkpoint replay differs')
    result['checkpoint_sha256'] = file_sha(directory/'final.pkl')
    write_json(directory/'metrics.json', result)
    results[name] = result
    # Every arm must start from the ORIGINAL base checkpoint, never continuation
    # of the previous arm. model.params retains the arm's own immutable tree.
    base.params = original_params['diagonal']
  config['status'] = 'sanity_complete' if args.sanity_only else 'complete'
  config['elapsed_seconds'] = time.time() - started
  config['frozen_input_files_unchanged'] = all(file_sha(path) == digest for path, digest in config['input_checkpoint_sha256'].items())
  write_json(out/'metrics.json', {'baseline_diagonal': baseline, 'baseline_marginal': initial_metrics, 'arms': results})
  write_json(out/'config.json', config)
  print(f'completed {len(arms)} arms in {time.time()-started:.1f}s; {out}', flush=True)


if __name__ == '__main__':
  main()
