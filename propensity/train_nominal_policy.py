"""Train a bounded conditional density for the offline behavior actions.

The default PointMaze decision context is the complete learner-visible
pre-action observation ``concat(state, commanded_goal)``.  The dataset loader
materializes only this observation and the same-row action; hidden swamp bits,
audit labels, rewards, and future observations are structurally unavailable to
the trainer.

Example::

  python -m propensity.train_nominal_policy \
    --dataset datasets/swamp_windy_f4_merged_s0.npz \
    --out-dir artifacts/nominal_policy/f4_p30_mdn_k5 \
    --num-components 5 --steps 20000 --seed 0
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from propensity.dataset import BehaviorDataset
from propensity.expert_population import (
    resolve_expert_positive_episodes, save_expert_population_manifest)
from propensity.nominal_policy import (
    NominalPolicySpec, censored_mixture_log_prob, make_policy_network)


def build_parser():
  parser = argparse.ArgumentParser(
      description='Fit b_phi(a|s,g_cmd) by conditional censored NLL.')
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--population', choices=('expert_positive', 'mixed'),
                      default='expert_positive',
                      help='expert_positive is the nominal-policy default; '
                           'mixed exists only to reproduce historical runs')
  parser.add_argument('--conditioning', choices=('state_goal', 'state_only'),
                      default='state_goal')
  parser.add_argument('--val-frac', type=float, default=0.1)
  parser.add_argument('--split-seed', type=int, default=0)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--val-batch-size', type=int, default=8192)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--steps', type=int, default=20000)
  parser.add_argument('--eval-every', type=int, default=500)
  parser.add_argument('--log-every', type=int, default=100)
  parser.add_argument('--num-components', type=int, default=5)
  parser.add_argument('--hidden-sizes', default='128,128')
  parser.add_argument('--min-scale', type=float, default=0.02)
  parser.add_argument('--max-scale', type=float, default=2.0)
  parser.add_argument('--loc-limit', type=float, default=3.0)
  parser.add_argument('--boundary-tol', type=float, default=1e-6)
  parser.add_argument('--grad-clip', type=float, default=10.0)
  parser.add_argument('--overwrite', action='store_true')
  return parser


def _jsonable(value):
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _write_json(path, value):
  temp = path + '.tmp'
  with open(temp, 'w') as f:
    json.dump(_jsonable(value), f, indent=2, sort_keys=True)
    f.write('\n')
  os.replace(temp, path)


def _write_pickle(path, value):
  temp = path + '.tmp'
  with open(temp, 'wb') as f:
    pickle.dump(jax.device_get(value), f, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(temp, path)


def _episode_hash(episode_ids):
  ids = np.ascontiguousarray(episode_ids, dtype=np.int64)
  return hashlib.sha256(ids.tobytes()).hexdigest()


def _trajectory_duplicate_audit(dataset):
  """Detect byte-identical selected trajectories spanning train and val."""
  train_episodes = set(np.unique(
      dataset._episode_of_row[dataset._train_idx]).tolist())
  validation_episodes = set(np.unique(
      dataset._episode_of_row[dataset._val_idx]).tolist())
  episodes = dataset._episode_of_row
  starts = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1]
  stops = np.r_[starts[1:], episodes.size]
  signatures = {}
  for start, stop in zip(starts, stops):
    episode = int(episodes[start])
    digest = hashlib.sha256()
    digest.update(dataset._state[start:stop].tobytes())
    digest.update(dataset._action[start:stop].tobytes())
    signatures.setdefault(digest.hexdigest(), []).append(episode)
  duplicate_groups = [ids for ids in signatures.values() if len(ids) > 1]
  cross_split = [ids for ids in duplicate_groups
                 if any(x in train_episodes for x in ids)
                 and any(x in validation_episodes for x in ids)]
  return {
      'definition': 'byte-identical complete selected state-action trajectories',
      'duplicate_groups': len(duplicate_groups),
      'duplicate_episodes': int(sum(len(x) for x in duplicate_groups)),
      'cross_split_duplicate_groups': len(cross_split),
      'cross_split_episode_ids': sorted({x for ids in cross_split for x in ids}),
      'passed': not cross_split,
  }


def _evaluate(network, params, context, action, mean, std, batch_size, spec):
  total, count = 0.0, 0
  for start in range(0, len(context), batch_size):
    c = jnp.asarray((context[start:start + batch_size] - mean) / std)
    a = jnp.asarray(action[start:start + batch_size])
    nll = -censored_mixture_log_prob(network.apply(params, c), a, spec)
    values = np.asarray(nll)
    if not np.isfinite(values).all():
      return float('nan')
    total += float(values.sum(dtype=np.float64))
    count += int(values.size)
  return total / count


def main(argv=None):
  args = build_parser().parse_args(argv)
  if args.steps < 0 or args.batch_size <= 0 or args.val_batch_size <= 0:
    raise SystemExit('steps must be nonnegative and batch sizes positive')
  if args.eval_every <= 0 or args.log_every <= 0:
    raise SystemExit('eval-every and log-every must be positive')
  hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(',') if x)
  if not hidden_sizes or min(hidden_sizes) <= 0:
    raise SystemExit('--hidden-sizes must be positive comma-separated integers')

  if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
    raise SystemExit(f'output directory is nonempty: {args.out_dir}; '
                     'pass --overwrite to replace its run files')
  os.makedirs(args.out_dir, exist_ok=True)

  selected_episode_ids = None
  source_selection = {
      'population': 'mixed_historical_baseline',
      'definition': 'all source populations in the merged dataset',
  }
  if args.population == 'expert_positive':
    selected_episode_ids, source_selection = resolve_expert_positive_episodes(
        args.dataset)
  state_mode = 'obs' if args.conditioning == 'state_goal' else 'state'
  dataset = BehaviorDataset(
      args.dataset, val_frac=args.val_frac, seed=args.split_seed,
      state_mode=state_mode, split_level='episode', strict_bounds=True,
      include_episode_ids=selected_episode_ids, split_reference='source')
  passed, gates, details = dataset.check()
  if not passed:
    raise RuntimeError(f'dataset contract checks failed: {gates}')
  if dataset.n_val == 0:
    raise RuntimeError('validation split is empty; use a positive --val-frac')
  duplicate_audit = _trajectory_duplicate_audit(dataset)
  if not duplicate_audit['passed']:
    raise RuntimeError('byte-identical trajectories cross train/validation: '
                       f'{duplicate_audit["cross_split_episode_ids"]}')

  train = dataset.arrays('train')
  val = dataset.arrays('val')
  # This is the only fitted preprocessing transform, and it is deliberately
  # computed from the training trajectories alone.
  context_mean = train.state.mean(axis=0, dtype=np.float64).astype(np.float32)
  raw_std = train.state.std(axis=0, dtype=np.float64).astype(np.float32)
  constant = raw_std < 1e-6
  context_std = np.where(constant, 1.0, raw_std).astype(np.float32)

  raw_meta = dataset.fingerprint.get('meta', {})
  physical_state_dim = int(raw_meta.get(
      'obs_dim', dataset.state_dim if state_mode == 'state'
      else dataset.state_dim // 2))
  goal_dim = int(raw_meta.get('goal_dim', dataset.state_dim - physical_state_dim))
  if args.conditioning == 'state_only':
    goal_dim = int(raw_meta.get('goal_dim', goal_dim))
  spec = NominalPolicySpec(
      state_dim=physical_state_dim, goal_dim=goal_dim,
      action_dim=dataset.action_dim, num_components=args.num_components,
      hidden_sizes=hidden_sizes, conditioning=args.conditioning,
      min_scale=args.min_scale, max_scale=args.max_scale,
      loc_limit=args.loc_limit, boundary_tol=args.boundary_tol)
  if spec.context_dim != dataset.state_dim:
    raise RuntimeError(f'model context width {spec.context_dim} != loaded '
                       f'dataset width {dataset.state_dim}')

  train_eps = np.unique(dataset._episode_of_row[dataset._train_idx])
  val_eps = np.unique(dataset._episode_of_row[dataset._val_idx])
  action = train.action
  boundary_fraction = np.mean(
      (action <= spec.action_low + spec.boundary_tol)
      | (action >= spec.action_high - spec.boundary_tol), axis=0)
  config = {
      'format_version': 1,
      'task': 'nominal_expert_action_density',
      'population': args.population,
      'dataset': dataset.report(),
      'dataset_metadata': raw_meta,
      'source_selection': source_selection,
      'model': spec.asdict(),
      'conditioning': {
          'input': ('learner-visible pre-action state concatenated with the '
                    'commanded goal' if args.conditioning == 'state_goal'
                    else 'learner-visible pre-action state only'),
          'excluded': ['swamp_bits', 'teacher_mode', 'route_label', 'force_safe',
                       'wait_count', 'entered_active_swamp', 'future_state',
                       'reward', 'hindsight_goal'],
      },
      'action_likelihood': {
          'space': 'environment action box [-1, 1]^2',
          'measure': ('Lebesgue density on (-1,1) per coordinate plus point '
                      'masses at exact clipped boundaries -1 and +1'),
          'sampling': 'sample latent Gaussian mixture, then componentwise clip',
          'training_boundary_fraction_per_dimension': boundary_fraction.tolist(),
      },
      'context_normalization': {
          'fit_split': 'training trajectories only',
          'mean': context_mean.tolist(),
          'std': context_std.tolist(),
          'constant_dimensions': np.flatnonzero(constant).tolist(),
      },
      'split': {
          'level': 'complete episodes', 'seed': args.split_seed,
          'val_fraction': args.val_frac,
          'train_episode_count': int(train_eps.size),
          'validation_episode_count': int(val_eps.size),
          'episode_overlap': int(np.intersect1d(train_eps, val_eps).size),
          'train_episode_ids_sha256': _episode_hash(train_eps),
          'validation_episode_ids': val_eps.tolist(),
          'validation_episode_ids_sha256': _episode_hash(val_eps),
          'duplicate_trajectory_audit': duplicate_audit,
      },
      'optimization': {
          'seed': args.seed, 'batch_size': args.batch_size,
          'learning_rate': args.learning_rate, 'steps': args.steps,
          'eval_every': args.eval_every, 'grad_clip': args.grad_clip,
      },
      'runtime': {
          'jax_version': jax.__version__,
          'jax_backend': jax.default_backend(),
          'jax_devices': [str(x) for x in jax.devices()],
      },
      'dataset_checks': {'passed': passed, 'gates': gates, 'details': details},
      'status': 'running',
  }
  if args.population == 'expert_positive':
    save_expert_population_manifest(
        os.path.join(args.out_dir, 'expert_population_manifest.json'),
        source_selection)
  _write_json(os.path.join(args.out_dir, 'config.json'), config)

  network = make_policy_network(spec)
  key = jax.random.PRNGKey(args.seed)
  key, init_key = jax.random.split(key)
  params = network.init(init_key, jnp.zeros((1, spec.context_dim), jnp.float32))
  optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip),
                          optax.adam(args.learning_rate))
  opt_state = optimizer.init(params)

  @jax.jit
  def update(parameters, optimizer_state, context, target_action):
    def objective(p):
      mixture = network.apply(p, context)
      return -jnp.mean(censored_mixture_log_prob(mixture, target_action, spec))
    loss, gradients = jax.value_and_grad(objective)(parameters)
    updates, optimizer_state = optimizer.update(
        gradients, optimizer_state, parameters)
    parameters = optax.apply_updates(parameters, updates)
    return parameters, optimizer_state, loss

  rng = np.random.default_rng(args.seed)
  metrics = []
  best_val = _evaluate(network, params, val.state, val.action, context_mean,
                       context_std, args.val_batch_size, spec)
  if not np.isfinite(best_val):
    raise RuntimeError(f'initial validation NLL is non-finite: {best_val}')
  best_step = 0
  _write_pickle(os.path.join(args.out_dir, 'best.pkl'),
                {'step': 0, 'params': params, 'validation_nll': best_val})
  print(f'dataset={args.dataset} train={dataset.n_train} val={dataset.n_val} '
        f'episodes={train_eps.size}/{val_eps.size} context={spec.context_dim} '
        f'action={spec.action_dim} K={spec.num_components}', flush=True)
  print(f'step=0 validation_nll={best_val:.6f}', flush=True)
  started = time.time()
  last_train_loss = float('nan')

  for step in range(1, args.steps + 1):
    batch = dataset.sample_batch(args.batch_size, 'train', rng)
    normalized = (batch.state - context_mean) / context_std
    params, opt_state, loss = update(
        params, opt_state, jnp.asarray(normalized), jnp.asarray(batch.action))
    last_train_loss = float(loss)
    if not np.isfinite(last_train_loss):
      raise RuntimeError(f'non-finite training NLL at step {step}: {last_train_loss}')
    if step % args.log_every == 0:
      rate = step / max(time.time() - started, 1e-9)
      print(f'step={step} train_nll={last_train_loss:.6f} '
            f'updates_per_sec={rate:.1f}', flush=True)
    should_eval = step % args.eval_every == 0 or step == args.steps
    if should_eval:
      val_nll = _evaluate(network, params, val.state, val.action, context_mean,
                          context_std, args.val_batch_size, spec)
      if not np.isfinite(val_nll):
        raise RuntimeError(f'non-finite validation NLL at step {step}: {val_nll}')
      row = {'step': step, 'train_batch_nll': last_train_loss,
             'validation_nll': val_nll,
             'elapsed_seconds': time.time() - started}
      metrics.append(row)
      _write_json(os.path.join(args.out_dir, 'metrics.json'), metrics)
      improved = val_nll < best_val
      if improved:
        best_val, best_step = val_nll, step
        _write_pickle(os.path.join(args.out_dir, 'best.pkl'),
                      {'step': step, 'params': params,
                       'validation_nll': val_nll})
      _write_pickle(os.path.join(args.out_dir, 'latest.pkl'),
                    {'step': step, 'params': params, 'opt_state': opt_state,
                     'validation_nll': val_nll})
      print(f'step={step} validation_nll={val_nll:.6f} '
            f'best={best_val:.6f}@{best_step}'
            + (' *' if improved else ''), flush=True)

  config['status'] = 'complete'
  config['result'] = {
      'best_step': best_step, 'best_validation_nll': best_val,
      'last_train_batch_nll': last_train_loss,
      'elapsed_seconds': time.time() - started,
      'all_recorded_losses_finite': bool(all(
          np.isfinite(x[k]) for x in metrics
          for k in ('train_batch_nll', 'validation_nll'))),
  }
  _write_json(os.path.join(args.out_dir, 'config.json'), config)
  print(json.dumps(config['result'], sort_keys=True), flush=True)
  return config['result']


if __name__ == '__main__':
  main()
