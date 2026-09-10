"""Evaluate nominal action-density checkpoints on held-out trajectories.

This reports bounded-action NLL, sample diversity at states selected using only
learner-visible coordinates, and an optional natural-environment rollout.  The
rollout is diagnostic rather than an expert-return claim: the fitted policy is
not given the demonstrator's hidden swamp bits.
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
from propensity.nominal_policy import load_nominal_policy


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--model-dir', required=True,
                      help='primary model, normally the K=5 MDN')
  parser.add_argument('--unimodal-dir', default='',
                      help='optional K=1 checkpoint trained with the same split')
  parser.add_argument('--samples-per-state', type=int, default=512)
  parser.add_argument('--representative-seed', type=int, default=917)
  parser.add_argument('--rollout-episodes', type=int, default=50)
  parser.add_argument('--rollout-seed', type=int, default=1200)
  parser.add_argument('--batch-size', type=int, default=8192)
  parser.add_argument('--out', default='')
  return parser


def _write_json(path, value):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  temp = path + '.tmp'
  with open(temp, 'w') as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write('\n')
  os.replace(temp, path)


def _heldout_nll(model, context, action, batch_size):
  values = []
  for start in range(0, len(context), batch_size):
    stop = start + batch_size
    values.append(np.asarray(model.log_prob(context[start:stop],
                                            action[start:stop])))
  log_prob = np.concatenate(values)
  if not np.isfinite(log_prob).all():
    raise RuntimeError('held-out log likelihood contains non-finite values')
  return {
      'n_transitions': int(log_prob.size),
      'nll_nats_per_action_vector': float(-log_prob.mean(dtype=np.float64)),
      'nll_standard_error': float(log_prob.std(dtype=np.float64)
                                  / np.sqrt(log_prob.size)),
      'min_log_prob': float(log_prob.min()),
      'max_log_prob': float(log_prob.max()),
      'action_space': 'environment action box [-1,1]^2',
      'measure': ('continuous density in the open box plus discrete tail mass '
                  'at each exact clipped boundary'),
  }


def _representative_indices(context, state_dim, rng):
  """Pick visible-state examples; never inspect audit fields or hidden U."""
  if state_dim < 2:
    return {'median_visible_state': int(len(context) // 2)}
  x, y = context[:, 0], context[:, 1]
  if state_dim >= 8:
    frames = context[:, :8].reshape(-1, 4, 2)
    motion = np.linalg.norm(frames[:, 0] - frames[:, -1], axis=-1)
  else:
    motion = np.zeros(len(context))
  masks = {
      'start': (x < 1.2) & (y > 2.5),
      'pre_swamp_holding': (x >= 2.0) & (x < 3.0) & (y > 2.5),
      'swamp_moving': (x >= 3.0) & (x < 6.0) & (y > 2.5) & (motion > 0.15),
      'swamp_stationary': (x >= 3.0) & (x < 6.0) & (y > 2.5) & (motion < 1e-5),
      'safe_route': y < 2.0,
      'post_swamp': (x >= 6.0) & (y > 2.5),
  }
  selected = {}
  for name, mask in masks.items():
    candidates = np.flatnonzero(mask)
    if candidates.size:
      selected[name] = int(rng.choice(candidates))
  if not selected:
    selected['median_visible_state'] = int(len(context) // 2)
  return selected


def _sample_summary(model, observation, key, n):
  samples = np.asarray(model.sample(observation, key, num_samples=n))
  if samples.shape != (n, model.spec.action_dim):
    raise RuntimeError(f'wrong sample shape {samples.shape}')
  if not (np.isfinite(samples).all()
          and (samples >= model.spec.action_low).all()
          and (samples <= model.spec.action_high).all()):
    raise RuntimeError('samples are non-finite or outside the action box')
  distribution = model.distribution(observation)
  logits = np.asarray(distribution.logits)
  weights = np.exp(logits - np.max(logits))
  weights /= weights.sum()
  entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-30))))
  component_loc = np.asarray(distribution.loc)
  component_separation = np.linalg.norm(
      component_loc[:, None, :] - component_loc[None, :, :], axis=-1)
  off_diagonal = component_separation[
      ~np.eye(model.spec.num_components, dtype=bool)]
  centered = samples - samples.mean(axis=0, keepdims=True)
  covariance = centered.T @ centered / max(n - 1, 1)
  eig = np.linalg.eigvalsh(covariance)
  std = samples.std(axis=0)
  return {
      'sample_count': int(n),
      'mean': samples.mean(axis=0).tolist(),
      'std': std.tolist(),
      'quantile_05': np.quantile(samples, 0.05, axis=0).tolist(),
      'quantile_50': np.quantile(samples, 0.50, axis=0).tolist(),
      'quantile_95': np.quantile(samples, 0.95, axis=0).tolist(),
      'min': samples.min(axis=0).tolist(),
      'max': samples.max(axis=0).tolist(),
      'boundary_fraction_per_dimension': np.mean(
          (samples <= model.spec.action_low + model.spec.boundary_tol)
          | (samples >= model.spec.action_high - model.spec.boundary_tol),
          axis=0).tolist(),
      'covariance_eigenvalues': eig.tolist(),
      'mixture_weights': weights.tolist(),
      'mixture_entropy_nats': entropy,
      'effective_components': float(np.exp(entropy)),
      'component_locs': component_loc.tolist(),
      'component_scales': np.asarray(distribution.scale).tolist(),
      'mean_pairwise_component_loc_distance': (
          float(off_diagonal.mean()) if off_diagonal.size else 0.0),
      'visible_mode_collapse_flag': bool(np.max(std) < 0.01),
  }


def _rollout(model, episodes, seed):
  if episodes <= 0:
    return {'status': 'skipped', 'episodes': 0}
  from crl.config import Config
  from crl import envs as envs_mod

  env_name = model.metadata['dataset'].get('env_name')
  if not env_name:
    return {'status': 'skipped', 'reason': 'dataset metadata has no env_name'}
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  if cfg.obs_dim + cfg.goal_dim != model.spec.state_dim + model.spec.goal_dim:
    return {'status': 'skipped', 'reason': 'environment/checkpoint dimensions differ'}

  key = jax.random.PRNGKey(seed + 1)
  successes, deaths, min_distances, final_distances = [], [], [], []
  action_mins = np.full(model.spec.action_dim, np.inf)
  action_maxs = np.full(model.spec.action_dim, -np.inf)
  for _ in range(episodes):
    observation = env.reset()
    minimum = float(np.linalg.norm(env.goal - env.state))
    died = False
    for _ in range(cfg.max_episode_steps):
      key, sample_key = jax.random.split(key)
      action = np.asarray(model.sample(observation, sample_key))
      action_mins = np.minimum(action_mins, action)
      action_maxs = np.maximum(action_maxs, action)
      observation, _, _, _ = env.step(action)
      minimum = min(minimum, float(np.linalg.norm(env.goal - env.state)))
      died = died or bool(getattr(env, 'dead', False))
    final_distance = float(np.linalg.norm(env.goal - env.state))
    successes.append(minimum < 0.5)
    deaths.append(died)
    min_distances.append(minimum)
    final_distances.append(final_distance)
  return {
      'status': 'complete', 'environment': env_name,
      'episodes': int(episodes), 'seed': int(seed),
      'success_definition': 'minimum XY distance to goal < 0.5',
      'success_rate': float(np.mean(successes)),
      'death_rate': float(np.mean(deaths)),
      'mean_minimum_goal_distance': float(np.mean(min_distances)),
      'mean_final_goal_distance': float(np.mean(final_distances)),
      'sampled_action_min': action_mins.tolist(),
      'sampled_action_max': action_maxs.tolist(),
      'interpretation': ('Supplementary only: this observational policy does '
                         'not receive the teacher-only swamp bits.'),
  }


def _evaluate_one(name, model, val, representative, args, seed_offset):
  result = {
      'name': name,
      'checkpoint_step': int(model.metadata['result']['best_step']),
      'num_components': int(model.spec.num_components),
      'heldout': _heldout_nll(model, val.state, val.action, args.batch_size),
      'representative_states': {},
  }
  base_key = jax.random.PRNGKey(args.representative_seed + seed_offset)
  for ordinal, (label, index) in enumerate(representative.items()):
    key = jax.random.fold_in(base_key, ordinal)
    result['representative_states'][label] = {
        'validation_row': int(index),
        'learner_visible_context': val.state[index].tolist(),
        'observed_action': val.action[index].tolist(),
        'samples': _sample_summary(
            model, val.state[index], key, args.samples_per_state),
    }
  result['rollout'] = _rollout(model, args.rollout_episodes,
                               args.rollout_seed + seed_offset)
  return result


def main(argv=None):
  args = build_parser().parse_args(argv)
  if args.samples_per_state <= 1 or args.batch_size <= 0:
    raise SystemExit('samples-per-state must exceed 1 and batch-size be positive')
  primary = load_nominal_policy(args.model_dir)
  cfg = primary.metadata
  dataset_cfg = cfg['dataset']
  if os.path.abspath(args.dataset) != os.path.abspath(dataset_cfg['dataset']):
    # Moving the same frozen dataset between machines is expected, so enforce
    # its fingerprint below rather than its absolute path.
    pass
  state_mode = 'obs' if primary.spec.conditioning == 'state_goal' else 'state'
  dataset = BehaviorDataset(
      args.dataset, val_frac=cfg['split']['val_fraction'],
      seed=cfg['split']['seed'], state_mode=state_mode,
      split_level='episode', strict_bounds=True)
  if dataset.fingerprint['sha256'] != dataset_cfg['sha256']:
    raise RuntimeError('dataset SHA-256 differs from the training dataset')
  val = dataset.arrays('val')
  rng = np.random.default_rng(args.representative_seed)
  representative = _representative_indices(
      val.state, primary.spec.state_dim, rng)
  report = {
      'dataset': dataset.report(),
      'selection': ('Representative rows use only learner-visible current XY '
                    'and frame-stack motion; no hidden or audit fields.'),
      'models': {},
  }
  report['models']['mdn'] = _evaluate_one(
      'mdn', primary, val, representative, args, 0)

  if args.unimodal_dir:
    unimodal = load_nominal_policy(args.unimodal_dir)
    other_cfg = unimodal.metadata
    if (other_cfg['dataset']['sha256'] != dataset_cfg['sha256']
        or other_cfg['split']['validation_episode_ids_sha256']
        != cfg['split']['validation_episode_ids_sha256']
        or unimodal.spec.conditioning != primary.spec.conditioning):
      raise RuntimeError('unimodal comparison did not use the same dataset, '
                         'episode split, and conditioning')
    report['models']['unimodal'] = _evaluate_one(
        'unimodal', unimodal, val, representative, args, 10000)
    mdn_nll = report['models']['mdn']['heldout']['nll_nats_per_action_vector']
    uni_nll = report['models']['unimodal']['heldout']['nll_nats_per_action_vector']
    report['comparison'] = {
        'mdn_minus_unimodal_heldout_nll': mdn_nll - uni_nll,
        'lower_is_better': True,
    }

  if args.out:
    _write_json(args.out, report)
  for label, result in report['models'].items():
    held = result['heldout']
    rollout = result['rollout']
    print(f'{label}: heldout NLL={held["nll_nats_per_action_vector"]:.6f} '
          f'nats/action-vector (n={held["n_transitions"]})')
    if rollout['status'] == 'complete':
      print(f'  rollout success={rollout["success_rate"]:.3f} '
            f'death={rollout["death_rate"]:.3f} '
            f'episodes={rollout["episodes"]}')
    for state_name, state_report in result['representative_states'].items():
      sample = state_report['samples']
      print(f'  {state_name}: std={np.round(sample["std"], 3).tolist()} '
            f'eff_K={sample["effective_components"]:.2f} '
            f'collapse={sample["visible_mode_collapse_flag"]}')
  if 'comparison' in report:
    print('MDN - unimodal heldout NLL = '
          f'{report["comparison"]["mdn_minus_unimodal_heldout_nll"]:+.6f}')
  return report


if __name__ == '__main__':
  main()
