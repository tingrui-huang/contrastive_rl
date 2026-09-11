"""Diagnose death observability in the PointMaze F4 learner state.

This script trains small diagnostic probes and evaluates a frozen diagonal ETT
checkpoint. Audit-only death and swamp fields are targets or strata only; they
never enter a probe or transition-model input.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optax
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors

from ett.dataset import ExpertTransitionDataset, load_audit_labels
from ett.diagonal_transition import DiagonalTransitionModel, load_diagonal_transition


SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
FIXED_THRESHOLD = 0.5
ZERO_TOLERANCE = 1e-7
NEAR_MOTION_TOLERANCE = 0.05


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--ett-run-dir', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--seeds', default='0,1')
  parser.add_argument('--hidden-sizes', default='64,64')
  parser.add_argument('--steps', type=int, default=4000)
  parser.add_argument('--batch-size', type=int, default=1024)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--eval-every', type=int, default=250)
  parser.add_argument('--ett-samples', type=int, default=128)
  parser.add_argument('--bootstrap-replicates', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=1701,
                      help='evaluation, sampling, and bootstrap seed')
  parser.add_argument('--overwrite', action='store_true')
  return parser


def _write_json(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'w', encoding='utf-8') as output:
    json.dump(value, output, indent=2, sort_keys=True)
    output.write('\n')
  os.replace(temporary, path)


def _write_pickle(path, value):
  temporary = path + '.tmp'
  with open(temporary, 'wb') as output:
    pickle.dump(jax.device_get(value), output, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(temporary, path)


def _sha256(path):
  digest = hashlib.sha256()
  with open(path, 'rb') as source:
    for block in iter(lambda: source.read(1 << 20), b''):
      digest.update(block)
  return digest.hexdigest()


def _git_provenance():
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


def _normalizer(value):
  mean = value.mean(axis=0).astype(np.float32)
  raw_std = value.std(axis=0).astype(np.float32)
  constant = raw_std < 1e-6
  std = np.where(constant, 1.0, raw_std).astype(np.float32)
  return mean, std, np.flatnonzero(constant).tolist()


def _sigmoid(logits):
  logits = np.asarray(logits, dtype=np.float64)
  probability = np.empty_like(logits)
  nonnegative = logits >= 0.0
  probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
  exponent = np.exp(logits[~nonnegative])
  probability[~nonnegative] = exponent / (1.0 + exponent)
  return probability


def _region_masks(state):
  x, y = state[:, 0], state[:, 1]
  return {
      'start': (x < 1.2) & (y > 2.5),
      'before_swamp': (x >= 1.2) & (x < 3.0) & (y > 2.5),
      'swamp_corridor': (x >= 3.0) & (x < 6.0) & (y > 2.5),
      'safe_route': y < 2.0,
      'after_swamp': (x >= 6.0) & (y > 2.5),
  }


def _region_name(state):
  masks = _region_masks(np.asarray(state)[None])
  for name in ('safe_route', 'start', 'before_swamp', 'swamp_corridor',
               'after_swamp'):
    if masks[name][0]:
      return name
  return 'other'


def _reconstruct_temporal_labels(path, arrays):
  """Reconstruct whether the agent is already dead when each state is seen."""
  with np.load(path, allow_pickle=False) as source:
    bits = np.asarray(source['swamp_bits'], dtype=bool)
    episode_died = np.asarray(source['entered_active_swamp'], dtype=bool)
    observation = np.asarray(source['obs'], dtype=np.float32)
    lengths = (np.asarray(source['lengths'], dtype=np.int64)
               if 'lengths' in source.files else
               np.full(len(observation), observation.shape[1], np.int64))

  death_observation_row = np.full(len(observation), -1, dtype=np.int64)
  selected_episodes = np.unique(arrays.episode_id)
  for episode in selected_episodes:
    episode = int(episode)
    if not episode_died[episode]:
      continue
    for action_row in range(int(lengths[episode]) - 1):
      next_cell = tuple(np.floor(
          observation[episode, action_row + 1, :2]).astype(np.int64))
      if (next_cell in SWAMP_CELLS
          and bits[episode, action_row, SWAMP_CELLS.index(next_cell)]):
        death_observation_row[episode] = action_row + 1
        break
    if death_observation_row[episode] < 1:
      raise RuntimeError(
          f'cannot reconstruct the fatal transition for episode {episode}')

  reconstructed_died = death_observation_row[selected_episodes] >= 0
  if not np.array_equal(reconstructed_died, episode_died[selected_episodes]):
    raise RuntimeError('reconstructed episode death flags do not match audit')

  death_row = death_observation_row[arrays.episode_id]
  dead_when_observed = (death_row >= 0) & (arrays.timestep >= death_row)
  dies_during_transition = ((death_row >= 1)
                            & (arrays.timestep == death_row - 1))
  alive_before_action = ~dead_when_observed
  death_age = np.where(
      dead_when_observed, arrays.timestep - death_row, -1).astype(np.int64)

  existing = load_audit_labels(path, arrays)
  if not np.array_equal(dead_when_observed, existing['post_death']):
    raise RuntimeError('temporal reconstruction disagrees with ETT audit labels')
  if not np.array_equal(dies_during_transition, existing['entering_death']):
    raise RuntimeError('fatal-transition timing disagrees with ETT audit labels')

  frames = arrays.state.reshape(-1, 4, 2)
  frame_spread = np.max(np.abs(frames - frames[:, :1]), axis=(1, 2))
  displacement = np.linalg.norm(arrays.delta_xy, axis=1)
  action_norm = np.linalg.norm(arrays.action, axis=1)
  zero_action = np.max(np.abs(arrays.action), axis=1) <= ZERO_TOLERANCE
  exact_stationary_transition = (
      np.max(np.abs(arrays.delta_xy), axis=1) <= ZERO_TOLERANCE)

  if dead_when_observed.any() and not np.all(
      exact_stationary_transition[dead_when_observed]):
    raise RuntimeError('post-death motion is not strictly zero in the dataset')
  if dies_during_transition.any() and not np.all(
      ~exact_stationary_transition[dies_during_transition]):
    raise RuntimeError('a reconstructed fatal transition did not move')

  return {
      'dead_when_observed': dead_when_observed,
      'alive_before_action': alive_before_action,
      'dies_during_transition': dies_during_transition,
      'death_age': death_age,
      'death_observation_row_by_episode': death_observation_row,
      'frame_spread_linf': frame_spread,
      'all_frames_equal': frame_spread <= ZERO_TOLERANCE,
      'displacement_l2': displacement,
      'action_l2': action_norm,
      'zero_action': zero_action,
      'exact_stationary_transition': exact_stationary_transition,
  }


def _temporal_audit(arrays, labels):
  dead = labels['dead_when_observed']
  alive = ~dead
  age = labels['death_age']
  equal = labels['all_frames_equal']
  zero_action = labels['zero_action']
  exact_stationary = labels['exact_stationary_transition']
  near_stationary = labels['displacement_l2'] <= NEAR_MOTION_TOLERANCE
  nonzero_action = labels['action_l2'] > NEAR_MOTION_TOLERANCE

  first_equal_ages = []
  for episode in np.unique(arrays.episode_id[dead]):
    here = (arrays.episode_id == episode) & dead & equal
    if here.any():
      first_equal_ages.append(int(age[here].min()))

  return {
      'transitions': int(len(dead)),
      'episodes': int(np.unique(arrays.episode_id).size),
      'dead_when_observed': int(dead.sum()),
      'dead_episode_count': int(np.unique(arrays.episode_id[dead]).size),
      'dead_prevalence': float(dead.mean()),
      'alive_before_action': int(alive.sum()),
      'dies_during_transition': int(labels['dies_during_transition'].sum()),
      'post_death_xy_strictly_zero_fraction': (
          float(exact_stationary[dead].mean()) if dead.any() else None),
      'fatal_transition_moves_fraction': (
          float((~exact_stationary[labels['dies_during_transition']]).mean())
          if labels['dies_during_transition'].any() else None),
      'dead_rows_by_age': {
          str(value): int((age == value).sum()) for value in (0, 1, 2)},
      'long_established_dead_age_ge_3': int((age >= 3).sum()),
      'first_all_equal_stack_age_after_death': {
          'episode_count_observed': len(first_equal_ages),
          'minimum': min(first_equal_ages) if first_equal_ages else None,
          'median': (float(np.median(first_equal_ages))
                     if first_equal_ages else None),
          'maximum': max(first_equal_ages) if first_equal_ages else None,
      },
      'alive_all_equal_stack': int((alive & equal).sum()),
      'alive_all_equal_at_reset': int(
          (alive & equal & (arrays.timestep == 0)).sum()),
      'alive_all_equal_after_reset': int(
          (alive & equal & (arrays.timestep > 0)).sum()),
      'alive_zero_recorded_action': int((alive & zero_action).sum()),
      'alive_exact_stationary_transition': int(
          (alive & exact_stationary).sum()),
      'alive_near_stationary_transition_l2_le_0p05': int(
          (alive & near_stationary).sum()),
      'alive_nonzero_action_near_stationary_collision_compatible': int(
          (alive & nonzero_action & near_stationary).sum()),
      'note': ('nonzero-action near-stationary rows are collision-compatible, '
               'not proven collision labels'),
  }


def _features(arrays):
  return {
      'xy': np.concatenate([arrays.state[:, :2], arrays.goal], axis=1),
      'f4': np.concatenate([arrays.state, arrays.goal], axis=1),
      'f4_action': np.concatenate(
          [arrays.state, arrays.action, arrays.goal], axis=1),
  }


def _make_probe(hidden_sizes):
  def forward(inputs):
    value = inputs
    for width in hidden_sizes:
      value = jax.nn.silu(hk.Linear(width)(value))
    return hk.Linear(1)(value)[:, 0]

  return hk.without_apply_rng(hk.transform(forward))


def _binary_cross_entropy(logits, labels):
  labels = jnp.asarray(labels, dtype=jnp.float32)
  return jnp.mean(jnp.maximum(logits, 0.0) - logits * labels
                  + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _predict_logits(network, params, value, batch_size=16384):
  chunks = []
  for start in range(0, len(value), batch_size):
    chunks.append(np.asarray(network.apply(
        params, jnp.asarray(value[start:start + batch_size]))))
  return np.concatenate(chunks)


def _fit_probe(name, seed, train_x, train_y, validation_x, validation_y,
               hidden_sizes, args):
  mean, std, constant = _normalizer(train_x)
  normalized_train = ((train_x - mean) / std).astype(np.float32)
  normalized_validation = ((validation_x - mean) / std).astype(np.float32)
  network = _make_probe(hidden_sizes)
  key = jax.random.PRNGKey(seed)
  params = network.init(key, jnp.asarray(normalized_train[:2]))
  optimizer = optax.adam(args.learning_rate)
  optimizer_state = optimizer.init(params)

  @jax.jit
  def update(parameters, state, inputs, labels):
    def objective(current):
      return _binary_cross_entropy(network.apply(current, inputs), labels)
    loss, gradient = jax.value_and_grad(objective)(parameters)
    updates, state = optimizer.update(gradient, state, parameters)
    return optax.apply_updates(parameters, updates), state, loss

  def validation_loss(parameters):
    logits = _predict_logits(network, parameters, normalized_validation)
    return float(np.mean(
        np.maximum(logits, 0.0) - logits * validation_y
        + np.log1p(np.exp(-np.abs(logits)))))

  initial = validation_loss(params)
  if not np.isfinite(initial):
    raise RuntimeError(f'{name} seed {seed} has non-finite initial loss')
  best_params, best_step, best_loss = params, 0, initial
  history = [{'step': 0, 'train_bce': None, 'validation_bce': initial}]
  rng = np.random.default_rng(seed)
  started = time.time()
  for step in range(1, args.steps + 1):
    rows = rng.integers(0, len(train_x), size=args.batch_size)
    params, optimizer_state, loss = update(
        params, optimizer_state, jnp.asarray(normalized_train[rows]),
        jnp.asarray(train_y[rows], dtype=jnp.float32))
    if step % args.eval_every == 0 or step == args.steps:
      train_loss = float(loss)
      value = validation_loss(params)
      if not np.isfinite([train_loss, value]).all():
        raise RuntimeError(f'{name} seed {seed} produced non-finite loss')
      if value < best_loss:
        best_params, best_step, best_loss = params, step, value
      history.append({'step': step, 'train_bce': train_loss,
                      'validation_bce': value})

  checkpoint = {
      'params': best_params, 'feature_name': name, 'seed': seed,
      'hidden_sizes': hidden_sizes, 'mean': mean, 'std': std,
      'constant_dimensions': constant, 'best_step': best_step,
      'best_validation_bce': best_loss,
  }
  checkpoint_path = os.path.join(args.out_dir, f'probe_{name}_s{seed}.pkl')
  _write_pickle(checkpoint_path, checkpoint)
  validation_logits = _predict_logits(
      network, best_params, normalized_validation)
  return {
      'probability': _sigmoid(validation_logits).astype(np.float64),
      'normalization': {
          'fit_split': 'expert-positive training trajectories only',
          'mean': mean.tolist(), 'std': std.tolist(),
          'constant_dimensions': constant,
      },
      'training': {
          'best_step': int(best_step),
          'best_validation_bce': float(best_loss),
          'all_losses_finite': bool(all(
              np.isfinite(row['validation_bce'])
              and (row['train_bce'] is None
                   or np.isfinite(row['train_bce'])) for row in history)),
          'elapsed_seconds': float(time.time() - started),
          'history': history,
          'checkpoint': os.path.basename(checkpoint_path),
      },
  }


def _calibration(y_true, probability, bins=10):
  edges = np.linspace(0.0, 1.0, bins + 1)
  rows, ece = [], 0.0
  for index in range(bins):
    include = ((probability >= edges[index])
               & (probability < edges[index + 1]
                  if index + 1 < bins else probability <= edges[index + 1]))
    if not include.any():
      continue
    predicted = float(probability[include].mean())
    observed = float(y_true[include].mean())
    ece += include.mean() * abs(predicted - observed)
    rows.append({'low': float(edges[index]), 'high': float(edges[index + 1]),
                 'count': int(include.sum()), 'predicted': predicted,
                 'observed': observed})
  return {'ece_10_equal_width': float(ece), 'bins': rows}


def _classification_metrics(y_true, probability, threshold=FIXED_THRESHOLD):
  y_true = np.asarray(y_true, dtype=bool)
  probability = np.asarray(probability, dtype=np.float64)
  prediction = probability >= threshold
  tp = int(np.sum(prediction & y_true))
  fp = int(np.sum(prediction & ~y_true))
  tn = int(np.sum(~prediction & ~y_true))
  fn = int(np.sum(~prediction & y_true))
  clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
  metrics = {
      'count': int(len(y_true)), 'positive_count': int(y_true.sum()),
      'prevalence': float(y_true.mean()),
      'mean_probability': float(probability.mean()),
      'threshold': float(threshold),
      'threshold_selection': 'fixed a priori; validation labels were not used',
      'confusion_matrix': {'true_negative': tn, 'false_positive': fp,
                           'false_negative': fn, 'true_positive': tp},
      'dead_precision': float(tp / (tp + fp)) if tp + fp else None,
      'dead_recall': float(tp / (tp + fn)) if tp + fn else None,
      'specificity': float(tn / (tn + fp)) if tn + fp else None,
      'brier': float(np.mean(np.square(probability - y_true))),
      'binary_cross_entropy': float(np.mean(
          -(y_true * np.log(clipped) + (~y_true) * np.log(1.0 - clipped)))),
      'calibration': _calibration(y_true, probability),
  }
  if np.unique(y_true).size == 2:
    metrics['roc_auc'] = float(roc_auc_score(y_true, probability))
    metrics['precision_recall_auc_average_precision'] = float(
        average_precision_score(y_true, probability))
  else:
    metrics['roc_auc'] = None
    metrics['precision_recall_auc_average_precision'] = None
  return metrics


def _evaluation_masks(arrays, labels):
  dead = labels['dead_when_observed']
  alive = ~dead
  age = labels['death_age']
  equal = labels['all_frames_equal']
  zero_action = labels['zero_action']
  near_stationary = labels['displacement_l2'] <= NEAR_MOTION_TOLERANCE
  nonzero_action = labels['action_l2'] > NEAR_MOTION_TOLERANCE
  moving_history = labels['frame_spread_linf'] > ZERO_TOLERANCE
  return {
      'fatal_entry_alive_before_action': labels['dies_during_transition'],
      'dead_age_0_first_observation': age == 0,
      'dead_age_1': age == 1,
      'dead_age_2': age == 2,
      'dead_history_transient_age_0_to_2': (age >= 0) & (age <= 2),
      'dead_established_age_ge_3': age >= 3,
      'alive_reset_seeded_equal_stack': (
          alive & equal & (arrays.timestep == 0)),
      'alive_equal_stack_after_reset': (
          alive & equal & (arrays.timestep > 0)),
      'alive_zero_recorded_action_waiting': alive & zero_action,
      'alive_near_stationary_transition_l2_le_0p05': (
          alive & near_stationary),
      'alive_nonzero_action_near_stationary_collision_compatible': (
          alive & nonzero_action & near_stationary),
      'alive_moving_history': alive & moving_history,
  }


def _stratified_probe_report(arrays, labels, probability):
  y_true = labels['dead_when_observed']
  output = {}
  for name, mask in _evaluation_masks(arrays, labels).items():
    if mask.any():
      output[name] = {
          'episodes': int(np.unique(arrays.episode_id[mask]).size),
          **_classification_metrics(y_true[mask], probability[mask]),
      }
    else:
      output[name] = {'episodes': 0, 'count': 0}
  return output


def _region_probe_report(arrays, labels, probability):
  y_true = labels['dead_when_observed']
  output = {}
  for name, mask in _region_masks(arrays.state).items():
    output[name] = {
        'episodes': int(np.unique(arrays.episode_id[mask]).size),
        **_classification_metrics(y_true[mask], probability[mask]),
    }
  return output


def _episode_bootstrap_difference(y_true, left, right, episode_id,
                                  subset, replicates, seed):
  """Return left-minus-right metrics after resampling complete episodes."""
  subset = np.asarray(subset, dtype=bool)
  unique = np.unique(episode_id)
  group = np.searchsorted(unique, episode_id)
  rng = np.random.default_rng(seed)
  values = {'roc_auc': [], 'precision_recall_auc_average_precision': [],
            'brier': []}
  skipped = 0
  for _ in range(replicates):
    draw = rng.integers(0, len(unique), size=len(unique))
    counts = np.bincount(draw, minlength=len(unique))
    weights = counts[group][subset]
    keep = weights > 0
    y = y_true[subset][keep]
    w = weights[keep]
    if np.unique(y).size < 2:
      skipped += 1
      continue
    p_left, p_right = left[subset][keep], right[subset][keep]
    values['roc_auc'].append(
        roc_auc_score(y, p_left, sample_weight=w)
        - roc_auc_score(y, p_right, sample_weight=w))
    values['precision_recall_auc_average_precision'].append(
        average_precision_score(y, p_left, sample_weight=w)
        - average_precision_score(y, p_right, sample_weight=w))
    left_brier = np.average(np.square(p_left - y), weights=w)
    right_brier = np.average(np.square(p_right - y), weights=w)
    values['brier'].append(left_brier - right_brier)
  result = {
      'definition': 'left minus right; complete source episodes resampled',
      'replicates_requested': int(replicates),
      'replicates_skipped_one_class': int(skipped),
      'episode_count': int(len(unique)),
  }
  for name, estimates in values.items():
    result[name] = {
        'mean': float(np.mean(estimates)),
        'ci95': np.quantile(estimates, [0.025, 0.975]).tolist(),
    }
  return result


def _distance_summary(value):
  value = np.asarray(value, dtype=np.float64)
  return {
      'count': int(len(value)), 'mean': float(value.mean()),
      'median': float(np.median(value)),
      'p10': float(np.quantile(value, 0.1)),
      'p90': float(np.quantile(value, 0.9)),
      'maximum': float(value.max()),
  }


def _episode_distinct_neighbors(index, query, bank_rows, train_episode_id,
                                maximum_k, batch_size=256):
  """Find nearest rows from distinct training episodes for every query."""
  candidate_count = min(len(bank_rows), max(2560, maximum_k * 128))
  output_distance = np.full((len(query), maximum_k), np.inf, np.float64)
  output_row = np.full((len(query), maximum_k), -1, np.int64)
  for start in range(0, len(query), batch_size):
    stop = min(start + batch_size, len(query))
    distance, local_row = index.kneighbors(
        query[start:stop], n_neighbors=candidate_count)
    for local_query in range(stop - start):
      seen_episodes = set()
      selected = 0
      for candidate_distance, candidate_local in zip(
          distance[local_query], local_row[local_query]):
        actual_row = int(bank_rows[candidate_local])
        episode = int(train_episode_id[actual_row])
        if episode in seen_episodes:
          continue
        seen_episodes.add(episode)
        output_distance[start + local_query, selected] = candidate_distance
        output_row[start + local_query, selected] = actual_row
        selected += 1
        if selected == maximum_k:
          break
      if selected < maximum_k:
        raise RuntimeError(
            'candidate neighbor search did not cover enough distinct episodes')
  return output_distance, output_row


def _neighbor_query(train_state, train_label, validation_state,
                    validation_label, train_arrays, validation_arrays,
                    validation_death_age, width, k_values=(1, 5, 20)):
  mean, std, constant = _normalizer(train_state[:, :width])
  train_x = ((train_state[:, :width] - mean) / std).astype(np.float32)
  validation_x = ((validation_state[:, :width] - mean) / std).astype(
      np.float32)
  maximum_k = max(k_values)
  train_alive = np.flatnonzero(~train_label)
  train_dead = np.flatnonzero(train_label)
  validation_alive = np.flatnonzero(~validation_label)
  validation_dead = np.flatnonzero(validation_label)

  alive_index = NearestNeighbors(
      n_neighbors=maximum_k, algorithm='auto', n_jobs=1).fit(
          train_x[train_alive])
  dead_index = NearestNeighbors(
      n_neighbors=maximum_k, algorithm='auto', n_jobs=1).fit(
          train_x[train_dead])
  dead_to_alive_distance, dead_to_alive_neighbor = _episode_distinct_neighbors(
      alive_index, validation_x[validation_dead], train_alive,
      train_arrays.episode_id, maximum_k)
  alive_to_dead_distance, alive_to_dead_neighbor = _episode_distinct_neighbors(
      dead_index, validation_x[validation_alive], train_dead,
      train_arrays.episode_id, maximum_k)
  dead_to_dead_distance, _ = dead_index.kneighbors(
      validation_x[validation_dead], n_neighbors=1)
  alive_to_alive_distance, _ = alive_index.kneighbors(
      validation_x[validation_alive], n_neighbors=1)

  radii = (0.05, 0.10, 0.25, 0.50, 1.00)
  result = {
      'distance_space': ('Euclidean distance after per-coordinate '
                         'training-state normalization'),
      'state_width': int(width),
      'normalization': {'mean': mean.tolist(), 'std': std.tolist(),
                        'constant_dimensions': constant},
      'different_episode_guarantee': (
          'queries are validation rows; every kth neighbor comes from a '
          'different training episode'),
      'dead_validation_to_alive_training': {
          'nearest_same_label_distance': _distance_summary(
              dead_to_dead_distance[:, 0]),
          'nearest_opposite_label_distance': _distance_summary(
              dead_to_alive_distance[:, 0]),
          'kth_opposite_distance': {
              str(k): _distance_summary(dead_to_alive_distance[:, k - 1])
              for k in k_values},
          'nearest_opposite_within_radius_fraction': {
              str(radius): float((dead_to_alive_distance[:, 0] <= radius).mean())
              for radius in radii},
      },
      'alive_validation_to_dead_training': {
          'nearest_same_label_distance': _distance_summary(
              alive_to_alive_distance[:, 0]),
          'nearest_opposite_label_distance': _distance_summary(
              alive_to_dead_distance[:, 0]),
          'kth_opposite_distance': {
              str(k): _distance_summary(alive_to_dead_distance[:, k - 1])
              for k in k_values},
          'nearest_opposite_within_radius_fraction': {
              str(radius): float((alive_to_dead_distance[:, 0] <= radius).mean())
              for radius in radii},
      },
  }
  for label, query_rows, distance in (
      ('dead_validation_to_alive_training', validation_dead,
       dead_to_alive_distance[:, 0]),
      ('alive_validation_to_dead_training', validation_alive,
       alive_to_dead_distance[:, 0])):
    result[label]['ambiguous_query_region_composition_at_radius_0p25'] = {
        name: {
            'transitions': int(np.sum(
                (distance <= 0.25)
                & _region_masks(validation_arrays.state[query_rows])[name])),
            'episodes': int(np.unique(validation_arrays.episode_id[
                query_rows[(distance <= 0.25)
                           & _region_masks(
                               validation_arrays.state[query_rows])[name]]
            ]).size),
        }
        for name in _region_masks(validation_arrays.state[query_rows])
    }

  dead_query_age = validation_death_age[validation_dead]
  result['dead_validation_to_alive_training']['by_time_since_death'] = {}
  for name, mask in (
      ('age_0', dead_query_age == 0), ('age_1', dead_query_age == 1),
      ('age_2', dead_query_age == 2), ('age_ge_3', dead_query_age >= 3)):
    result['dead_validation_to_alive_training']['by_time_since_death'][name] = {
        'nearest_opposite_label_distance': _distance_summary(
            dead_to_alive_distance[mask, 0]),
        'nearest_opposite_within_radius_fraction': {
            str(radius): float(
                (dead_to_alive_distance[mask, 0] <= radius).mean())
            for radius in radii},
    }

  return (result, {
      'validation_dead_rows': validation_dead,
      'validation_alive_rows': validation_alive,
      'train_alive_rows': train_alive,
      'train_dead_rows': train_dead,
      'dead_to_alive_distance': dead_to_alive_distance,
      'dead_to_alive_neighbor': dead_to_alive_neighbor,
      'alive_to_dead_distance': alive_to_dead_distance,
      'alive_to_dead_neighbor': alive_to_dead_neighbor,
  })


def _exact_alias_audit(train_arrays, train_labels, validation_arrays,
                       validation_labels):
  state = np.concatenate([train_arrays.state, validation_arrays.state])
  label = np.concatenate([
      train_labels['dead_when_observed'],
      validation_labels['dead_when_observed']])
  episode = np.concatenate([train_arrays.episode_id,
                            validation_arrays.episode_id])
  _, inverse = np.unique(state, axis=0, return_inverse=True)
  minimum = np.ones(inverse.max() + 1, dtype=np.int8)
  maximum = np.zeros(inverse.max() + 1, dtype=np.int8)
  np.minimum.at(minimum, inverse, label.astype(np.int8))
  np.maximum.at(maximum, inverse, label.astype(np.int8))
  mixed = np.flatnonzero(minimum != maximum)
  cross_episode = 0
  examples = []
  for group in mixed:
    rows = np.flatnonzero(inverse == group)
    alive = rows[~label[rows]]
    dead = rows[label[rows]]
    found = None
    for left in alive:
      candidates = dead[episode[dead] != episode[left]]
      if candidates.size:
        found = (left, int(candidates[0]))
        break
    if found is not None:
      cross_episode += 1
      if len(examples) < 5:
        examples.append({
            'state_f4': state[found[0]].tolist(),
            'alive_episode': int(episode[found[0]]),
            'dead_episode': int(episode[found[1]]),
        })
  return {
      'definition': 'bit-identical float32 full four-frame learner states',
      'mixed_label_state_groups': int(len(mixed)),
      'mixed_label_groups_spanning_distinct_episodes': int(cross_episode),
      'examples': examples,
  }


def _concrete_pairs(neighbor, train_arrays, train_labels, validation_arrays,
                    validation_labels, maximum=6):
  candidates = []
  for direction, query_rows, distance, matched in (
      ('dead_validation_to_alive_training',
       neighbor['validation_dead_rows'], neighbor['dead_to_alive_distance'][:, 0],
       neighbor['dead_to_alive_neighbor'][:, 0]),
      ('alive_validation_to_dead_training',
       neighbor['validation_alive_rows'], neighbor['alive_to_dead_distance'][:, 0],
       neighbor['alive_to_dead_neighbor'][:, 0])):
    used_query_episodes = set()
    used_neighbor_episodes = set()
    for local_index in np.argsort(distance):
      query = int(query_rows[local_index])
      match = int(matched[local_index])
      query_episode = int(validation_arrays.episode_id[query])
      match_episode = int(train_arrays.episode_id[match])
      if (query_episode in used_query_episodes
          or match_episode in used_neighbor_episodes):
        continue
      used_query_episodes.add(query_episode)
      used_neighbor_episodes.add(match_episode)
      xy_mean, xy_std, _ = _normalizer(train_arrays.state[:, :2])
      xy_distance = float(np.linalg.norm(
          (validation_arrays.state[query, :2]
           - train_arrays.state[match, :2]) / xy_std))
      entry = {
          'direction': direction,
          'f4_normalized_l2_distance': float(distance[local_index]),
          'xy_normalized_l2_distance': xy_distance,
          'query': {
              'label_dead_when_observed': bool(
                  validation_labels['dead_when_observed'][query]),
              'episode': query_episode,
              'timestep': int(validation_arrays.timestep[query]),
              'death_age': int(validation_labels['death_age'][query]),
              'region': _region_name(validation_arrays.state[query]),
              'frames_newest_first': validation_arrays.state[query].reshape(
                  4, 2).tolist(),
              'recorded_action': validation_arrays.action[query].tolist(),
          },
          'neighbor': {
              'label_dead_when_observed': bool(
                  train_labels['dead_when_observed'][match]),
              'episode': match_episode,
              'timestep': int(train_arrays.timestep[match]),
              'death_age': int(train_labels['death_age'][match]),
              'region': _region_name(train_arrays.state[match]),
              'frames_newest_first': train_arrays.state[match].reshape(
                  4, 2).tolist(),
              'recorded_action': train_arrays.action[match].tolist(),
          },
      }
      candidates.append((direction, entry))
      if sum(item[0] == direction for item in candidates) >= maximum // 2:
        break
  return [item[1] for item in candidates]


def _ett_samples(model, arrays, count, batch_size, seed):
  samples, raw_delta, atom = [], [], []
  for batch_index, start in enumerate(range(0, len(arrays.state), batch_size)):
    stop = start + batch_size
    value, diagnostics = model.sample_with_diagnostics(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], jax.random.PRNGKey(seed + batch_index),
        num_samples=count, goal=arrays.goal[start:stop])
    samples.append(np.asarray(value))
    raw_delta.append(np.asarray(diagnostics['raw_delta']))
    atom.append(np.asarray(model.stationary_probability(
        arrays.state[start:stop], arrays.action[start:stop],
        arrays.action[start:stop], goal=arrays.goal[start:stop])))
  return np.concatenate(samples), np.concatenate(raw_delta), np.concatenate(atom)


def _ett_stratum(arrays, labels, mask, atom, sampled_stationary,
                 motion_magnitude):
  moving = ~sampled_stationary
  return {
      'transitions': int(mask.sum()),
      'episodes': int(np.unique(arrays.episode_id[mask]).size),
      'true_dead_prevalence': float(
          labels['dead_when_observed'][mask].mean()),
      'observed_next_exact_stationary_frequency': float(
          labels['exact_stationary_transition'][mask].mean()),
      'mean_analytic_zero_atom_probability': float(atom[mask].mean()),
      'mean_stationary_probability_after_sampling_and_projection': float(
          sampled_stationary[mask].mean()),
      'mean_predicted_motion_probability_after_projection': float(
          moving[mask].mean()),
      'mean_predicted_motion_magnitude_maze_units': float(
          motion_magnitude[mask].mean()),
      'mean_magnitude_given_predicted_motion_maze_units': (
          float(motion_magnitude[mask][moving[mask]].mean())
          if moving[mask].any() else 0.0),
  }


def _evaluate_ett(model, arrays, labels, probe_probability, args):
  samples, raw_delta, atom = _ett_samples(
      model, arrays, args.ett_samples, 4096, args.seed)
  final_delta = samples[..., :2] - arrays.state[:, None, :2]
  final_motion = np.linalg.norm(final_delta, axis=-1)
  sampled_stationary = np.max(np.abs(final_delta), axis=-1) <= ZERO_TOLERANCE
  sampled_stationary_probability = sampled_stationary.mean(axis=1)
  masks = _evaluation_masks(arrays, labels)
  strata = {
      name: _ett_stratum(
          arrays, labels, mask, atom, sampled_stationary, final_motion)
      for name, mask in masks.items() if mask.any()
  }
  dead = labels['dead_when_observed']
  confidence_masks = {
      'true_dead_probe_probability_lt_0p1': dead & (probe_probability < 0.1),
      'true_dead_probe_probability_0p1_to_0p5': (
          dead & (probe_probability >= 0.1) & (probe_probability < 0.5)),
      'true_dead_probe_probability_0p5_to_0p9': (
          dead & (probe_probability >= 0.5) & (probe_probability < 0.9)),
      'true_dead_probe_probability_ge_0p9': dead & (probe_probability >= 0.9),
  }
  by_confidence = {
      name: _ett_stratum(
          arrays, labels, mask, atom, sampled_stationary, final_motion)
      for name, mask in confidence_masks.items() if mask.any()
  }
  return {
      'contract': 'a = observational_action = recorded expert action',
      'checkpoint_model_type': model.metadata['model_type'],
      'checkpoint_best_step': model.metadata['result']['best_step'],
      'samples_per_transition': int(args.ett_samples),
      'zero_motion_tolerance_linf': ZERO_TOLERANCE,
      'analytic_atom_as_dead_probability': _classification_metrics(
          labels['dead_when_observed'], atom),
      'sampled_projected_stationary_as_dead_probability': (
          _classification_metrics(
              labels['dead_when_observed'], sampled_stationary_probability)),
      'overall': _ett_stratum(
          arrays, labels, np.ones(len(arrays.state), dtype=bool), atom,
          sampled_stationary, final_motion),
      'strata': strata,
      'by_f4_probe_confidence': by_confidence,
      'raw_sample_motion': {
          'mean_l2_maze_units': float(np.linalg.norm(raw_delta, axis=-1).mean()),
          'fraction_exceeding_one_per_coordinate': float(
              np.any(np.abs(raw_delta) > 1.0, axis=-1).mean()),
      },
  }, {'atom': atom, 'sampled_stationary': sampled_stationary,
      'motion_magnitude': final_motion}


def _plot_curves(out_dir, y_true, probability_by_name):
  figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
  for name, probability in probability_by_name.items():
    false_positive, true_positive, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    axes[0].plot(false_positive, true_positive,
                 label=f'{name} AUC={roc_auc_score(y_true, probability):.4f}')
    axes[1].plot(recall, precision, label=(
        f'{name} AP={average_precision_score(y_true, probability):.4f}'))
  axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
  axes[0].set(xlabel='false-positive rate', ylabel='true-positive rate',
              title='Death-state ROC on validation transitions')
  axes[1].axhline(y_true.mean(), color='gray', linestyle='--', linewidth=1)
  axes[1].set(xlabel='recall', ylabel='precision',
              title='Death-state precision-recall')
  for axis in axes:
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
  figure.tight_layout()
  path = os.path.join(out_dir, 'probe_roc_pr.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  return os.path.basename(path)


def _plot_calibration(out_dir, y_true, probability_by_name):
  figure, axis = plt.subplots(figsize=(5.8, 5.0))
  axis.plot([0, 1], [0, 1], color='gray', linestyle='--', label='ideal')
  for name, probability in probability_by_name.items():
    rows = _calibration(y_true, probability)['bins']
    axis.plot([row['predicted'] for row in rows],
              [row['observed'] for row in rows], marker='o', label=name)
  axis.set(xlabel='mean predicted dead probability',
           ylabel='observed dead frequency', title='Validation calibration',
           xlim=(0, 1), ylim=(0, 1))
  axis.grid(alpha=0.2)
  axis.legend()
  figure.tight_layout()
  path = os.path.join(out_dir, 'probe_calibration.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  return os.path.basename(path)


def _plot_death_age(out_dir, labels, probability_by_name, ett_internal):
  ages = [0, 1, 2, 3]
  age_labels = ['age 0', 'age 1', 'age 2', 'age >= 3']
  masks = [labels['death_age'] == age for age in ages[:3]]
  masks.append(labels['death_age'] >= 3)
  x = np.arange(len(masks))
  figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
  width = 0.8 / len(probability_by_name)
  for index, (name, probability) in enumerate(probability_by_name.items()):
    axes[0].bar(x + (index - 1) * width,
                [probability[mask].mean() for mask in masks], width=width,
                label=name)
  axes[0].set_xticks(x, age_labels)
  axes[0].set_ylim(0, 1)
  axes[0].set_ylabel('mean predicted dead probability')
  axes[0].set_title('Probe confidence by time since death')
  axes[0].legend(fontsize=8)
  axes[1].plot(x, [ett_internal['atom'][mask].mean() for mask in masks],
               marker='o', label='analytic zero atom')
  axes[1].plot(x, [ett_internal['sampled_stationary'][mask].mean()
                   for mask in masks], marker='o',
               label='sampled stationary after projection')
  axes[1].set_xticks(x, age_labels)
  axes[1].set_ylim(0, 1)
  axes[1].set_ylabel('probability')
  axes[1].set_title('Frozen diagonal ETT by time since death')
  axes[1].legend(fontsize=8)
  for axis in axes:
    axis.grid(alpha=0.2)
  figure.tight_layout()
  path = os.path.join(out_dir, 'death_age_probe_and_ett.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  return os.path.basename(path)


def _plot_neighbor_cdf(out_dir, xy_neighbor, f4_neighbor):
  figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
  for axis, direction, title in (
      (axes[0], 'dead_to_alive_distance', 'Dead validation to alive training'),
      (axes[1], 'alive_to_dead_distance', 'Alive validation to dead training')):
    for name, source in (('XY', xy_neighbor), ('F4', f4_neighbor)):
      value = np.sort(source[direction][:, 0])
      axis.plot(value, np.arange(1, len(value) + 1) / len(value), label=name)
    axis.set(xlabel='nearest opposite-label distance (train-normalized L2)',
             ylabel='empirical CDF', title=title)
    axis.set_xlim(left=0)
    axis.grid(alpha=0.2)
    axis.legend()
  figure.tight_layout()
  path = os.path.join(out_dir, 'opposite_label_neighbor_cdf.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  return os.path.basename(path)


def _plot_pairs(out_dir, pairs):
  selected = [pair for pair in pairs
              if pair['direction'] == 'dead_validation_to_alive_training'][:3]
  figure, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4),
                             squeeze=False)
  for index, pair in enumerate(selected):
    axis = axes[0, index]
    for role, marker, color in (('query', 'o', 'tab:red'),
                                ('neighbor', 'x', 'tab:blue')):
      frames = np.asarray(pair[role]['frames_newest_first'])
      axis.plot(frames[::-1, 0], frames[::-1, 1], marker=marker, color=color,
                label=f'{role}: {"dead" if pair[role]["label_dead_when_observed"] else "alive"}')
      action = np.asarray(pair[role]['recorded_action'])
      axis.arrow(frames[0, 0], frames[0, 1], 0.2 * action[0],
                 0.2 * action[1], color=color, width=0.003,
                 length_includes_head=True)
    axis.set_title(f'F4 distance {pair["f4_normalized_l2_distance"]:.3f}')
    axis.set_aspect('equal', adjustable='datalim')
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
  figure.tight_layout()
  path = os.path.join(out_dir, 'closest_alive_dead_histories.png')
  figure.savefig(path, dpi=160)
  plt.close(figure)
  return os.path.basename(path)


def main(argv=None):
  args = build_parser().parse_args(argv)
  if min(args.steps, args.batch_size, args.eval_every, args.ett_samples,
         args.bootstrap_replicates) <= 0:
    raise SystemExit('all count arguments must be positive')
  seeds = tuple(int(value) for value in args.seeds.split(','))
  hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(','))
  if not seeds or not hidden_sizes:
    raise SystemExit('seeds and hidden sizes cannot be empty')
  config_path = os.path.join(args.out_dir, 'config.json')
  if os.path.exists(config_path) and not args.overwrite:
    raise SystemExit('output directory already exists; use --overwrite')
  os.makedirs(args.out_dir, exist_ok=True)

  dataset = ExpertTransitionDataset(args.dataset, val_frac=0.1, split_seed=0)
  train = dataset.arrays('train')
  validation = dataset.arrays('validation')
  train_labels = _reconstruct_temporal_labels(args.dataset, train)
  validation_labels = _reconstruct_temporal_labels(args.dataset, validation)
  y_train = train_labels['dead_when_observed'].astype(np.float32)
  y_validation = validation_labels['dead_when_observed'].astype(bool)
  train_features = _features(train)
  validation_features = _features(validation)

  model = load_diagonal_transition(args.ett_run_dir)
  if not isinstance(model, DiagonalTransitionModel):
    raise RuntimeError('the supplied ETT checkpoint is not a stochastic model')
  recorded_dataset_sha = model.metadata['dataset']['dataset']['sha256']
  if recorded_dataset_sha != dataset.behavior_report['sha256']:
    raise RuntimeError('ETT checkpoint and diagnostic dataset hashes differ')

  config = {
      'format_version': 1,
      'task': 'pointmaze_f4_death_observability_diagnostic',
      'scope': {
          'production_state_modified': False,
          'nominal_policy_modified': False,
          'ett_model_modified': False,
          'probe_is_production_discriminator': False,
      },
      'dataset': dataset.report,
      'temporal_label': {
          'target': 'agent already dead when s_t is observed',
          'fatal_transition_label': ('alive before action; death is first '
                                     'positive at s_{t+1}'),
          'source': ('audit-only per-episode death flag plus the first '
                     'pre-action swamp mask whose transition lands in its '
                     'active cell'),
          'stationarity_used_to_define_label': False,
      },
      'representations': {
          'xy': ['current_xy', 'commanded_goal_f4'],
          'f4': ['four_frame_state', 'commanded_goal_f4'],
          'f4_action': ['four_frame_state', 'recorded_current_action',
                        'commanded_goal_f4'],
          'goal_note': ('the commanded goal is included consistently but is '
                        'constant and normalizes to zero'),
          'excluded_inputs': ['swamp_bits', 'death_flags', 'teacher_mode',
                              'source_labels', 'rewards', 'future_frames',
                              'future_outcomes'],
      },
      'probe': {
          'architecture': list(hidden_sizes), 'activation': 'SiLU',
          'output': 'single Bernoulli logit', 'objective': 'unweighted BCE',
          'seeds': list(seeds), 'steps': args.steps,
          'batch_size': args.batch_size,
          'learning_rate': args.learning_rate,
          'eval_every': args.eval_every,
          'checkpoint_selection': 'lowest validation BCE',
          'classification_threshold': FIXED_THRESHOLD,
          'threshold_selection': 'fixed before validation evaluation',
      },
      'ett': {
          'run_dir': os.path.abspath(args.ett_run_dir),
          'best_checkpoint_sha256': _sha256(
              os.path.join(args.ett_run_dir, 'best.pkl')),
          'samples_per_transition': args.ett_samples,
          'sampling_seed': args.seed,
      },
      'runtime': {
          'jax_version': jax.__version__,
          'jax_backend': jax.default_backend(),
          'jax_devices': [str(device) for device in jax.devices()],
          'sklearn_metrics': True,
      },
      'source': _git_provenance(),
      'status': 'running',
  }
  _write_json(config_path, config)

  probe_results = {}
  probability = {}
  for name in ('xy', 'f4', 'f4_action'):
    probe_results[name] = {}
    for seed in seeds:
      print(f'training probe={name} seed={seed}', flush=True)
      fit = _fit_probe(
          name, seed, train_features[name], y_train,
          validation_features[name], y_validation.astype(np.float32),
          hidden_sizes, args)
      key = f'seed_{seed}'
      probability[(name, seed)] = fit.pop('probability')
      fit['validation'] = _classification_metrics(
          y_validation, probability[(name, seed)])
      fit['validation_strata'] = _stratified_probe_report(
          validation, validation_labels, probability[(name, seed)])
      fit['validation_regions'] = _region_probe_report(
          validation, validation_labels, probability[(name, seed)])
      probe_results[name][key] = fit
      metric = fit['validation']
      print(f'  ROC={metric["roc_auc"]:.6f} '
            f'AP={metric["precision_recall_auc_average_precision"]:.6f} '
            f'precision={metric["dead_precision"]:.6f} '
            f'recall={metric["dead_recall"]:.6f}', flush=True)

  principal_seed = seeds[0]
  xy_probability = probability[('xy', principal_seed)]
  f4_probability = probability[('f4', principal_seed)]
  all_rows = np.ones(len(validation.state), dtype=bool)
  swamp_rows = _region_masks(validation.state)['swamp_corridor']
  uncertainty = {
      'principal_seed': principal_seed,
      'left': 'f4', 'right': 'xy',
      'overall': _episode_bootstrap_difference(
          y_validation, f4_probability, xy_probability,
          validation.episode_id, all_rows, args.bootstrap_replicates,
          args.seed),
      'swamp_corridor_only': _episode_bootstrap_difference(
          y_validation, f4_probability, xy_probability,
          validation.episode_id, swamp_rows, args.bootstrap_replicates,
          args.seed + 1),
  }

  xy_neighbors, xy_neighbor_internal = _neighbor_query(
      train.state, y_train.astype(bool), validation.state, y_validation,
      train, validation, validation_labels['death_age'], 2)
  f4_neighbors, f4_neighbor_internal = _neighbor_query(
      train.state, y_train.astype(bool), validation.state, y_validation,
      train, validation, validation_labels['death_age'], 8)
  ambiguity = {
      'exact_full_f4_aliasing': _exact_alias_audit(
          train, train_labels, validation, validation_labels),
      'xy': xy_neighbors,
      'f4': f4_neighbors,
      'concrete_cross_split_opposite_label_pairs': _concrete_pairs(
          f4_neighbor_internal, train, train_labels, validation,
          validation_labels),
      'interpretation_guard': (
          'nearby finite-sample pairs do not prove fundamental '
          'non-identifiability; large distances may reflect sparse coverage'),
  }

  ett_report, ett_internal = _evaluate_ett(
      model, validation, validation_labels, f4_probability, args)
  seed_zero_probability = {
      'XY': xy_probability,
      'F4': f4_probability,
      'F4 + action': probability[('f4_action', principal_seed)],
  }
  plots = [
      _plot_curves(args.out_dir, y_validation, seed_zero_probability),
      _plot_calibration(args.out_dir, y_validation, seed_zero_probability),
      _plot_death_age(args.out_dir, validation_labels,
                      seed_zero_probability, ett_internal),
      _plot_neighbor_cdf(args.out_dir, xy_neighbor_internal,
                         f4_neighbor_internal),
      _plot_pairs(args.out_dir,
                  ambiguity['concrete_cross_split_opposite_label_pairs']),
  ]

  metrics = {
      'evaluation_scope': ('checkpoint-selection validation trajectories; '
                           'there is no untouched test split'),
      'dataset_sha256': dataset.behavior_report['sha256'],
      'temporal_audit': {
          'train': _temporal_audit(train, train_labels),
          'validation': _temporal_audit(validation, validation_labels),
      },
      'probes': probe_results,
      'episode_bootstrap_f4_minus_xy': uncertainty,
      'ambiguity': ambiguity,
      'frozen_diagonal_ett': ett_report,
      'plots': plots,
  }
  metrics['all_checks_pass'] = bool(
      all(result[f'seed_{seed}']['training']['all_losses_finite']
          for result in probe_results.values() for seed in seeds)
      and metrics['temporal_audit']['train'][
          'post_death_xy_strictly_zero_fraction'] == 1.0
      and metrics['temporal_audit']['validation'][
          'post_death_xy_strictly_zero_fraction'] == 1.0)
  _write_json(os.path.join(args.out_dir, 'metrics.json'), metrics)
  config['status'] = 'complete'
  config['result'] = {
      'all_checks_pass': metrics['all_checks_pass'],
      'principal_seed': principal_seed,
      'xy_roc_auc': probe_results['xy'][f'seed_{principal_seed}'][
          'validation']['roc_auc'],
      'f4_roc_auc': probe_results['f4'][f'seed_{principal_seed}'][
          'validation']['roc_auc'],
      'f4_action_roc_auc': probe_results['f4_action'][
          f'seed_{principal_seed}']['validation']['roc_auc'],
  }
  _write_json(config_path, config)
  print(json.dumps(config['result'], sort_keys=True))
  return metrics


if __name__ == '__main__':
  main()
