"""Diagnose a frozen PointMaze critic as a continuation-score surrogate.

This script is read-only with respect to every learned model. It evaluates

  C(s, g) = E[a ~ pi_frozen(. | s, g)] f_frozen(s, a, g)

using the raw aligned contrastive critic logit, which is the score maximized by
the repository actor's critic term. Audit-only death fields are used only to
form evaluation groups. They never enter the actor or critic inputs.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors

from crl import checkpoint as checkpoint_mod
from crl import networks as networks_mod
from ett.dataset import ExpertTransitionDataset
from scripts.diagnose_f4_death_observability import (
    NEAR_MOTION_TOLERANCE, ZERO_TOLERANCE, _reconstruct_temporal_labels,
    _region_name)


DEFAULT_DATASET = (
    'artifacts/f4_p30_server_30076/results/datasets/'
    'swamp_windy_f4_merged_s0.npz')
DEFAULT_RUNS_ROOT = (
    'artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/'
    'p30_a0_a01_a03_s0_s1')
DEFAULT_BANK = (
    'artifacts/f4_p30_server_30076/results/artifacts/'
    'swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz')
DEFAULT_RUNS = (
    'alpha0_seed0,alpha0_seed1,alpha0p1_seed0,alpha0p1_seed1,'
    'alpha0p3_seed0,alpha0p3_seed1')
DEFAULT_PRIMARY = 'alpha0p3_seed0'
DISTANCE_BAND_EDGES = (0.0, 2.0, 3.0, 4.0, 5.0, math.inf)


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--dataset', default=DEFAULT_DATASET)
  parser.add_argument('--runs-root', default=DEFAULT_RUNS_ROOT)
  parser.add_argument('--run-names', default=DEFAULT_RUNS)
  parser.add_argument('--primary-run', default=DEFAULT_PRIMARY)
  parser.add_argument('--failure-bank', default=DEFAULT_BANK)
  parser.add_argument('--out-dir', default=(
      'artifacts/critic_continuation/f4_p30_alpha_s01'))
  parser.add_argument('--small-samples', type=int, default=8)
  parser.add_argument('--large-samples', type=int, default=128)
  parser.add_argument('--gradient-samples', type=int, default=8)
  parser.add_argument('--goal-probe-anchors', type=int, default=256)
  parser.add_argument('--bootstrap-replicates', type=int, default=1000)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--seed', type=int, default=2301)
  parser.add_argument('--overwrite', action='store_true')
  return parser


def _json_default(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.integer,)):
    return int(value)
  if isinstance(value, (np.floating,)):
    return float(value)
  if isinstance(value, (np.bool_,)):
    return bool(value)
  raise TypeError(f'cannot serialize {type(value)}')


def _write_json(path, value):
  with open(path, 'w', encoding='utf-8') as handle:
    json.dump(value, handle, indent=2, sort_keys=True,
              default=_json_default, allow_nan=False)
    handle.write('\n')


def _sha256(path):
  digest = hashlib.sha256()
  with open(path, 'rb') as handle:
    while True:
      block = handle.read(1 << 20)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def _content_sha256(path):
  digest = hashlib.sha256()
  with np.load(path, allow_pickle=False) as data:
    for key in sorted(data.files):
      value = data[key]
      digest.update(key.encode())
      digest.update(str(value.dtype).encode())
      digest.update(str(value.shape).encode())
      digest.update(np.ascontiguousarray(value).tobytes())
  return digest.hexdigest()


def _git(*args):
  try:
    return subprocess.check_output(
        ['git', *args], stderr=subprocess.DEVNULL).decode().strip()
  except (OSError, subprocess.CalledProcessError):
    return ''


def _git_provenance():
  dirty = _git('status', '--porcelain', '--untracked-files=no')
  return {
      'branch': _git('branch', '--show-current'),
      'commit': _git('rev-parse', 'HEAD'),
      'tracked_worktree_dirty': bool(dirty),
      'tracked_dirty_paths': dirty.splitlines(),
      'untracked_files_excluded_from_dirty_check': True,
  }


def _dist(values):
  value = np.asarray(values, dtype=np.float64)
  value = value[np.isfinite(value)]
  if not value.size:
    return {'n': 0}
  return {
      'n': int(value.size),
      'mean': float(np.mean(value)),
      'std': float(np.std(value)),
      'median': float(np.median(value)),
      'p10': float(np.percentile(value, 10)),
      'p25': float(np.percentile(value, 25)),
      'p75': float(np.percentile(value, 75)),
      'p90': float(np.percentile(value, 90)),
      'minimum': float(np.min(value)),
      'maximum': float(np.max(value)),
  }


def _ci(values):
  value = np.asarray(values, dtype=np.float64)
  value = value[np.isfinite(value)]
  if not value.size:
    return None
  return {
      'lower_2p5': float(np.percentile(value, 2.5)),
      'median': float(np.percentile(value, 50)),
      'upper_97p5': float(np.percentile(value, 97.5)),
      'finite_replicates': int(value.size),
  }


def _network():
  return networks_mod.make_networks(
      obs_dim=8, goal_dim=8, action_dim=2, repr_dim=64,
      repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6,
      twin_q=False, use_image_obs=False, use_layer_norm=False,
      obs_scale=None)


def _make_scorers(network):
  def paired_one(q_params, observation, action):
    output = network.q_network.apply(
        q_params, observation[None, :], action[None, :])
    return output[0, 0]

  paired_many = jax.jit(jax.vmap(paired_one, in_axes=(None, 0, 0)))

  @jax.jit
  def sampled(policy_params, q_params, observation, epsilon):
    distribution = network.policy_network.apply(policy_params, observation)
    action = jnp.tanh(
        distribution.loc[:, None, :]
        + distribution.scale[:, None, :] * epsilon)
    n_states, n_samples, action_dim = action.shape
    flat_action = action.reshape(n_states * n_samples, action_dim)
    flat_observation = jnp.repeat(observation, n_samples, axis=0)
    score = jax.vmap(paired_one, in_axes=(None, 0, 0))(
        q_params, flat_observation, flat_action)
    return score.reshape((n_states, n_samples)), action

  @jax.jit
  def mode(policy_params, q_params, observation):
    distribution = network.policy_network.apply(policy_params, observation)
    action = jnp.tanh(distribution.loc)
    score = jax.vmap(paired_one, in_axes=(None, 0, 0))(
        q_params, observation, action)
    return score, action, distribution.scale

  return paired_many, sampled, mode


def _score_pairs(paired_many, q_params, observation, action,
                 batch_size=32768):
  output = []
  for begin in range(0, len(observation), batch_size):
    end = begin + batch_size
    output.append(np.asarray(paired_many(
        q_params, jnp.asarray(observation[begin:end], jnp.float32),
        jnp.asarray(action[begin:end], jnp.float32))))
  return np.concatenate(output, axis=0)


def _continuation_scores(sampled, mode, state, goal, epsilon,
                         policy_params, q_params, batch_size):
  sample_mean, sample_std, action_mean = [], [], []
  mode_score, mode_action, policy_scale = [], [], []
  observation = np.concatenate([state, goal], axis=1).astype(np.float32)
  for begin in range(0, len(state), batch_size):
    end = begin + batch_size
    score, action = sampled(
        policy_params, q_params,
        jnp.asarray(observation[begin:end]),
        jnp.asarray(epsilon[begin:end]))
    score = np.asarray(score)
    action = np.asarray(action)
    sample_mean.append(score.mean(axis=1))
    sample_std.append(score.std(axis=1))
    action_mean.append(action.mean(axis=1))
    ms, ma, ps = mode(
        policy_params, q_params, jnp.asarray(observation[begin:end]))
    mode_score.append(np.asarray(ms))
    mode_action.append(np.asarray(ma))
    policy_scale.append(np.asarray(ps))
  return {
      'mean': np.concatenate(sample_mean),
      'sample_std': np.concatenate(sample_std),
      'action_mean': np.concatenate(action_mean),
      'mode_score': np.concatenate(mode_score),
      'mode_action': np.concatenate(mode_action),
      'policy_scale': np.concatenate(policy_scale),
  }


def _regions(state):
  return np.asarray([_region_name(value) for value in state], dtype=object)


def _distance_bands(state, goal):
  distance = np.linalg.norm(state[:, :2] - goal[:, :2], axis=1)
  edges = np.asarray(DISTANCE_BAND_EDGES)
  band = np.searchsorted(edges, distance, side='right') - 1
  band = np.clip(band, 0, len(edges) - 2)
  labels = []
  for index in band:
    left, right = edges[index], edges[index + 1]
    labels.append(f'[{left:g},{right:g})' if np.isfinite(right)
                  else f'[{left:g},inf)')
  return distance, np.asarray(labels, dtype=object)


def _evaluation_masks(arrays, labels, region):
  dead = labels['dead_when_observed']
  alive = ~dead
  age = labels['death_age']
  near = labels['displacement_l2'] <= NEAR_MOTION_TOLERANCE
  waiting = labels['zero_action']
  moving = labels['displacement_l2'] > NEAR_MOTION_TOLERANCE
  return {
      'all_alive': alive,
      'alive_moving': alive & moving & ~waiting,
      'alive_waiting_or_near_stationary': alive & (waiting | near),
      'alive_zero_action_waiting': alive & waiting,
      'alive_near_stationary': alive & near,
      'fatal_entry_alive_before_action': labels['dies_during_transition'],
      'dead_age_0': age == 0,
      'dead_age_1': age == 1,
      'dead_age_2': age == 2,
      'dead_established_age_ge_3': age >= 3,
      'all_dead': dead,
      'safe_route_alive': alive & (region == 'safe_route'),
      'swamp_corridor_alive': alive & (region == 'swamp_corridor'),
      'swamp_corridor_dead': dead & (region == 'swamp_corridor'),
  }


def _group_report(score, arrays, masks):
  output = {}
  for name, mask in masks.items():
    output[name] = {
        'episodes': int(np.unique(arrays.episode_id[mask]).size),
        'score': _dist(score[mask]),
    }
  return output


def _region_report(score, arrays, labels, region, distance_band):
  output = {}
  dead = labels['dead_when_observed']
  for region_name in sorted(set(region.tolist())):
    output[region_name] = {}
    for status, status_mask in [('alive', ~dead), ('dead', dead)]:
      mask = (region == region_name) & status_mask
      output[region_name][status] = {
          'episodes': int(np.unique(arrays.episode_id[mask]).size),
          'score': _dist(score[mask]),
      }
  bands = {}
  for band_name in sorted(set(distance_band.tolist())):
    bands[band_name] = {}
    for status, status_mask in [('alive', ~dead), ('dead', dead)]:
      mask = (distance_band == band_name) & status_mask
      bands[band_name][status] = {
          'episodes': int(np.unique(arrays.episode_id[mask]).size),
          'score': _dist(score[mask]),
      }
  return {'region': output, 'distance_to_commanded_goal_band': bands}


def _binary_metrics(label, death_score, sample_weight=None):
  label = np.asarray(label, dtype=bool)
  if np.unique(label).size < 2:
    return {'roc_auc': None, 'average_precision': None,
            'death_prevalence': float(np.average(
                label, weights=sample_weight))}
  return {
      'roc_auc': float(roc_auc_score(
          label, death_score, sample_weight=sample_weight)),
      'average_precision': float(average_precision_score(
          label, death_score, sample_weight=sample_weight)),
      'death_prevalence': float(np.average(label, weights=sample_weight)),
  }


def _bootstrap_binary(label, death_score, episode_id, replicates, seed):
  episodes = np.unique(episode_id)
  ep_pos = {int(ep): i for i, ep in enumerate(episodes)}
  row_ep = np.asarray([ep_pos[int(ep)] for ep in episode_id], np.int64)
  rng = np.random.default_rng(seed)
  roc, ap = [], []
  for _ in range(replicates):
    sampled = rng.integers(0, len(episodes), size=len(episodes))
    counts = np.bincount(sampled, minlength=len(episodes))
    weight = counts[row_ep]
    if weight[label].sum() == 0 or weight[~label].sum() == 0:
      continue
    metric = _binary_metrics(label, death_score, sample_weight=weight)
    roc.append(metric['roc_auc'])
    ap.append(metric['average_precision'])
  return {
      'unit': 'complete source episode',
      'method': 'percentile bootstrap with episode multiplicity weights',
      'replicates_requested': int(replicates),
      'roc_auc_95pct': _ci(roc),
      'average_precision_95pct': _ci(ap),
  }


def _bootstrap_group_difference(score, episode_id, left, right,
                                replicates, seed):
  episodes = np.unique(episode_id)
  ep_pos = {int(ep): i for i, ep in enumerate(episodes)}
  row_ep = np.asarray([ep_pos[int(ep)] for ep in episode_id], np.int64)
  rng = np.random.default_rng(seed)
  values = []
  for _ in range(replicates):
    sampled = rng.integers(0, len(episodes), size=len(episodes))
    counts = np.bincount(sampled, minlength=len(episodes))
    weight = counts[row_ep]
    left_weight = weight * left
    right_weight = weight * right
    if left_weight.sum() and right_weight.sum():
      left_mean = np.average(score, weights=left_weight)
      right_mean = np.average(score, weights=right_weight)
      values.append(left_mean - right_mean)
  return {
      'estimand': 'mean score(left) - mean score(right)',
      'unit': 'complete source episode',
      'point': float(score[left].mean() - score[right].mean()),
      '95pct': _ci(values),
  }


def _matched_alive(dead, arrays, region, distance_band):
  alive = ~dead
  match = np.full(len(arrays.state), -1, np.int64)
  query_rows = np.flatnonzero(dead)
  for region_name in sorted(set(region[query_rows].tolist())):
    for band_name in sorted(set(distance_band[query_rows].tolist())):
      queries = query_rows[
          (region[query_rows] == region_name)
          & (distance_band[query_rows] == band_name)]
      candidates = np.flatnonzero(
          alive & (region == region_name) & (distance_band == band_name))
      if not len(queries) or not len(candidates):
        continue
      neighbors = NearestNeighbors(
          n_neighbors=min(len(candidates), 64), metric='euclidean',
          n_jobs=1).fit(arrays.state[candidates])
      _, local = neighbors.kneighbors(arrays.state[queries])
      for row, choices in zip(queries, local):
        possible = candidates[choices]
        possible = possible[
            arrays.episode_id[possible] != arrays.episode_id[row]]
        if possible.size:
          match[row] = possible[0]
  kept = query_rows[match[query_rows] >= 0]
  partner = match[kept]
  if np.any(arrays.episode_id[kept] == arrays.episode_id[partner]):
    raise RuntimeError('matched comparison contains a same-episode pair')
  return kept, partner


def _matched_report(score, query, partner, arrays, labels, goal_distance):
  difference = score[query] - score[partner]
  f4_distance = np.linalg.norm(
      arrays.state[query] - arrays.state[partner], axis=1)
  radius = {}
  for threshold in (0.25, 0.5, 1.0, 2.0):
    selected = f4_distance <= threshold
    selected_by_age = {}
    for age_name, age_mask in (
        ('age_0', labels['death_age'][query] == 0),
        ('age_1', labels['death_age'][query] == 1),
        ('age_2', labels['death_age'][query] == 2),
        ('age_ge_3', labels['death_age'][query] >= 3)):
      both = selected & age_mask
      selected_by_age[age_name] = {
          'pairs': int(both.sum()),
          'dead_score_below_alive_fraction': (
              float(np.mean(difference[both] < 0.0))
              if both.any() else None),
          'score_dead_minus_alive': _dist(difference[both]),
      }
    radius[f'l2_le_{threshold:g}'.replace('.', 'p')] = {
        'pairs': int(selected.sum()),
        'dead_score_below_alive_fraction': (
            float(np.mean(difference[selected] < 0.0))
            if selected.any() else None),
        'score_dead_minus_alive': _dist(difference[selected]),
        'by_death_age': selected_by_age,
    }
  by_age = {}
  query_age = labels['death_age'][query]
  for name, selected in (
      ('age_0', query_age == 0), ('age_1', query_age == 1),
      ('age_2', query_age == 2), ('age_ge_3', query_age >= 3)):
    by_age[name] = {
        'pairs': int(selected.sum()),
        'dead_score_below_alive_fraction': (
            float(np.mean(difference[selected] < 0.0))
            if selected.any() else None),
        'score_dead_minus_alive': _dist(difference[selected]),
        'full_f4_l2_distance': _dist(f4_distance[selected]),
    }
  return {
      'pairs': int(len(query)),
      'query_episodes': int(np.unique(arrays.episode_id[query]).size),
      'partner_episodes': int(np.unique(arrays.episode_id[partner]).size),
      'same_episode_pairs': int(np.sum(
          arrays.episode_id[query] == arrays.episode_id[partner])),
      'dead_score_below_alive_fraction': float(np.mean(difference < 0.0)),
      'score_dead_minus_alive': _dist(difference),
      'full_f4_l2_distance': _dist(f4_distance),
      'current_xy_l2_distance': _dist(np.linalg.norm(
          arrays.state[query, :2] - arrays.state[partner, :2], axis=1)),
      'absolute_goal_distance_difference': _dist(np.abs(
          goal_distance[query] - goal_distance[partner])),
      'matching_constraints': [
          'same exclusive maze region',
          'same fixed Euclidean distance-to-commanded-goal band',
          'different source episode',
          'nearest raw full-F4 history within the constrained candidate set',
      ],
      'confounding_note': (
          'matching narrows observed geometry but does not eliminate '
          'behavior, time, or hidden-swamp confounding'),
      'radius_sensitivity': radius,
      'death_age': by_age,
  }


def _bootstrap_matched(score, query, partner, episode_id, replicates, seed):
  query_ep = episode_id[query]
  episodes = np.unique(query_ep)
  difference = score[query] - score[partner]
  rng = np.random.default_rng(seed)
  mean_diff, lower_fraction = [], []
  for _ in range(replicates):
    sampled = episodes[rng.integers(0, len(episodes), len(episodes))]
    blocks = [np.flatnonzero(query_ep == ep) for ep in sampled]
    rows = np.concatenate(blocks)
    mean_diff.append(float(difference[rows].mean()))
    lower_fraction.append(float(np.mean(difference[rows] < 0.0)))
  return {
      'unit': 'dead query episode, with matched rows kept together',
      'mean_dead_minus_alive_95pct': _ci(mean_diff),
      'dead_below_alive_fraction_95pct': _ci(lower_fraction),
  }


def _semantics():
  return {
      'critic_output': (
          'inner product between a state-action encoder and a goal encoder; '
          'a B-by-B matrix of raw logits'),
      'positive_goals': (
          'diagonal future achieved F4 states sampled from the same episode '
          'with probability proportional to discount^(j-i)'),
      'ordinary_negatives': (
          'off-diagonal goals from independently sampled episodes in the '
          'batch, trained with label zero'),
      'failure_negatives': (
          'for alpha>0, established frozen F4 bank states replace only the '
          'GOAL half in a second critic application; every bank goal is '
          'averaged as a label-zero negative'),
      'actor_goal_training': (
          'the replay sampler discards the environment commanded-goal half '
          'and supplies hindsight future achieved F4 goals; random_goals=0.5 '
          'adds a batch-rolled hindsight goal copy'),
      'diagnostic_actor_goal': (
          'the stored commanded task goal is kept fixed, matching deployment'),
      'actor_critic_term': (
          'with entropy_coefficient=0, the critic component minimizes '
          '-diag(raw_logit); bc_coef=0.5 mixes this with behavior cloning'),
      'transformations': {
          'sigmoid_in_nce_loss': True,
          'sigmoid_in_actor_score': False,
          'sigmoid_in_reported_continuation_score': False,
          'exponentiation': False,
          'representation_normalization': False,
          'temperature': None,
          'observation_normalization': None,
          'twin_head_reduction': (
              'not applicable: all evaluated checkpoints have twin_q=False'),
      },
      'interpretation': (
          'raw contrastive ranking surrogate; not a calibrated Q value, '
          'success probability, death probability, or worst-case Q'),
      'code_path': {
          'critic': 'crl/networks.py make_networks::_critic_fn',
          'actor_objective': 'crl/losses.py build_learner::actor_loss',
          'failure_negative_loss': 'crl/losses.py build_learner::critic_loss',
          'relabeling': 'crl/replay.py TrajectoryBuffer.sample',
          'diagnostic': (
              'scripts/diagnose_f4_critic_continuation.py '
              '_continuation_scores'),
      },
  }


def _load_runs(root, names, dataset_content_sha, bank_content_sha):
  output = []
  for name in names:
    run_dir = pathlib.Path(root) / name
    provenance_path = run_dir / 'arm_provenance.json'
    checkpoint_path = run_dir / 'final.pkl'
    metrics_path = run_dir / 'metrics.json'
    if not provenance_path.exists() or not checkpoint_path.exists():
      raise FileNotFoundError(f'missing run input under {run_dir}')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance['dataset_content_sha256'] != dataset_content_sha:
      raise RuntimeError(f'{name} was trained on a different dataset content')
    alpha = float(provenance['alpha'])
    if alpha > 0 and provenance.get('bank_content_sha256') != bank_content_sha:
      raise RuntimeError(f'{name} was trained with a different failure bank')
    train_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    output.append({
        'name': name,
        'run_dir': str(run_dir.resolve()),
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': _sha256(checkpoint_path),
        'step': int(checkpoint_mod.load_checkpoint(checkpoint_path)[0]),
        'alpha': alpha,
        'seed': int(provenance['seed']),
        'provenance': provenance,
        'training_metrics_last': train_metrics[-1],
    })
  return output


def _goal_slot_probe(paired_many, sampled, mode, state_obj, arrays, masks,
                     bank, epsilon_bank, anchor_rows, batch_size):
  state = arrays.state[anchor_rows]
  action = arrays.action[anchor_rows]
  task_goal = arrays.goal[anchor_rows]
  task_observation = np.concatenate([state, task_goal], axis=1)
  task_score = _score_pairs(
      paired_many, state_obj.q_params, task_observation, action)

  n_anchor, n_bank = len(anchor_rows), len(bank)
  failure_goal_observation = np.concatenate([
      np.repeat(state, n_bank, axis=0),
      np.tile(bank, (n_anchor, 1)),
  ], axis=1).astype(np.float32)
  repeated_action = np.repeat(action, n_bank, axis=0)
  failure_score = _score_pairs(
      paired_many, state_obj.q_params, failure_goal_observation,
      repeated_action).reshape(n_anchor, n_bank)
  failure_mean = failure_score.mean(axis=1)

  bank_goal = np.repeat(task_goal[:1], n_bank, axis=0)
  state_slot = _continuation_scores(
      sampled, mode, bank, bank_goal, epsilon_bank,
      state_obj.policy_params, state_obj.q_params, batch_size)
  return {
      'anchor_definition': (
          'fixed authentic alive-moving validation anchors with recorded '
          'actions'),
      'anchor_count': int(n_anchor),
      'failure_bank_count': int(n_bank),
      'task_goal_in_goal_slot_recorded_action': _dist(task_score),
      'failure_goal_in_goal_slot_recorded_action_all_pairs': _dist(
          failure_score.ravel()),
      'failure_goal_in_goal_slot_per_anchor_mean': _dist(failure_mean),
      'per_anchor_failure_goal_below_task_goal_fraction': float(np.mean(
          failure_mean < task_score)),
      'same_failure_bank_state_in_state_slot_continuation': _dist(
          state_slot['mean']),
      'state_slot_note': (
          'bank F4 state is supplied as s while the commanded task goal stays '
          'in g; the actor is sampled at that state-goal context'),
  }


def _gradient_report(network, state_obj, arrays, labels, masks,
                     sample_count, seed):
  rng = np.random.default_rng(seed)
  chosen = []
  group_of = []
  for name in ('alive_moving', 'alive_waiting_or_near_stationary',
               'dead_age_0', 'dead_established_age_ge_3'):
    rows = np.flatnonzero(masks[name])
    take = min(4, len(rows))
    picked = rng.choice(rows, size=take, replace=False)
    chosen.extend(picked.tolist())
    group_of.extend([name] * take)
  chosen = np.asarray(chosen, np.int64)
  epsilon = rng.normal(
      size=(len(chosen), sample_count, 2)).astype(np.float32)

  def one_score(state, goal, noise, through_actor):
    observation = jnp.concatenate([state, goal], axis=0)[None, :]
    distribution = network.policy_network.apply(
        state_obj.policy_params, observation)
    action = jnp.tanh(
        distribution.loc[0, None, :]
        + distribution.scale[0, None, :] * noise)
    if not through_actor:
      action = jax.lax.stop_gradient(action)

    def score_action(value):
      output = network.q_network.apply(
          state_obj.q_params, observation, value[None, :])
      return output[0, 0]
    return jnp.mean(jax.vmap(score_action)(action))

  direct_grad = jax.jit(jax.vmap(jax.grad(
      lambda s, g, e: one_score(s, g, e, False), argnums=0)))
  total_grad = jax.jit(jax.vmap(jax.grad(
      lambda s, g, e: one_score(s, g, e, True), argnums=0)))
  direct = np.asarray(direct_grad(
      jnp.asarray(arrays.state[chosen]), jnp.asarray(arrays.goal[chosen]),
      jnp.asarray(epsilon)))
  total = np.asarray(total_grad(
      jnp.asarray(arrays.state[chosen]), jnp.asarray(arrays.goal[chosen]),
      jnp.asarray(epsilon)))
  direct_norm = np.linalg.norm(direct, axis=1)
  total_norm = np.linalg.norm(total, axis=1)
  path_delta = np.linalg.norm(total - direct, axis=1)
  records = []
  for index, row in enumerate(chosen):
    records.append({
        'group': group_of[index],
        'episode_id': int(arrays.episode_id[row]),
        'timestep': int(arrays.timestep[row]),
        'death_age': int(labels['death_age'][row]),
        'direct_critic_state_gradient_l2': float(direct_norm[index]),
        'total_actor_and_critic_state_gradient_l2': float(total_norm[index]),
        'actor_path_contribution_l2': float(path_delta[index]),
    })
  return {
      'sample_count': int(sample_count),
      'states': records,
      'all_direct_finite': bool(np.isfinite(direct).all()),
      'all_total_finite': bool(np.isfinite(total).all()),
      'direct_nonzero_fraction': float(np.mean(direct_norm > 0.0)),
      'total_nonzero_fraction': float(np.mean(total_norm > 0.0)),
      'actor_path_nonzero_fraction': float(np.mean(path_delta > 1e-10)),
      'direct_gradient_l2': _dist(direct_norm),
      'total_gradient_l2': _dist(total_norm),
      'actor_path_contribution_l2': _dist(path_delta),
      'interpretation': (
          'parameters are frozen values, but no stop-gradient is placed on '
          'state inputs. The total path differentiates through actor loc and '
          'scale and then through the critic action input. The direct variant '
          'stops only the sampled action.'),
      'ood_limitation': (
          'this checks authentic dataset states only; it does not validate '
          'gradient magnitude or ranking on arbitrary generated OOD states'),
  }


def _examples(score, arrays, labels, region, count=5):
  dead = labels['dead_when_observed']
  high_dead = np.flatnonzero(dead)[np.argsort(score[dead])[-count:][::-1]]
  low_alive = np.flatnonzero(~dead)[np.argsort(score[~dead])[:count]]

  def record(row):
    return {
        'episode_id': int(arrays.episode_id[row]),
        'timestep': int(arrays.timestep[row]),
        'death_age': int(labels['death_age'][row]),
        'region': str(region[row]),
        'frames_newest_first': arrays.state[row].reshape(4, 2).tolist(),
        'commanded_goal_frames': arrays.goal[row].reshape(4, 2).tolist(),
        'recorded_action': arrays.action[row].tolist(),
        'continuation_score': float(score[row]),
    }
  return {
      'highest_scoring_failure_states': [record(int(row)) for row in high_dead],
      'lowest_scoring_alive_states': [record(int(row)) for row in low_alive],
  }


def _plot_age(path, score, masks):
  names = ['alive_moving', 'alive_waiting_or_near_stationary', 'dead_age_0',
           'dead_age_1', 'dead_age_2', 'dead_established_age_ge_3']
  labels = ['alive moving', 'alive wait/near', 'dead age 0', 'dead age 1',
            'dead age 2', 'dead age >=3']
  values = [score[masks[name]] for name in names]
  figure, axis = plt.subplots(figsize=(10, 5.5))
  violin = axis.violinplot(values, showmedians=True, showextrema=False)
  for body in violin['bodies']:
    body.set_alpha(0.55)
  axis.axhline(0.0, color='black', lw=0.7, alpha=0.5)
  axis.set_xticks(range(1, len(labels) + 1), labels, rotation=18, ha='right')
  axis.set_ylabel('raw continuation score C(s, g_task)')
  axis.set_title('Frozen critic continuation score by verified death age')
  figure.tight_layout()
  figure.savefig(path, dpi=160)
  plt.close(figure)


def _plot_roc_pr(path, label, score, primary_name):
  fpr, tpr, _ = roc_curve(label, -score)
  precision, recall, _ = precision_recall_curve(label, -score)
  roc_value = roc_auc_score(label, -score)
  ap_value = average_precision_score(label, -score)
  figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
  axes[0].plot(fpr, tpr, label=f'ROC-AUC {roc_value:.3f}')
  axes[0].plot([0, 1], [0, 1], '--', color='grey')
  axes[0].set(xlabel='false positive rate', ylabel='true positive rate',
              title='Negative continuation score as death ranking')
  axes[0].legend()
  axes[1].plot(recall, precision, label=f'AP {ap_value:.3f}')
  axes[1].axhline(label.mean(), ls='--', color='grey',
                  label=f'prevalence {label.mean():.3f}')
  axes[1].set(xlabel='recall', ylabel='precision', title=primary_name)
  axes[1].legend()
  figure.tight_layout()
  figure.savefig(path, dpi=160)
  plt.close(figure)


def _plot_budget(path, small, large, label):
  rng = np.random.default_rng(991)
  rows = rng.choice(len(small), size=min(4000, len(small)), replace=False)
  alive_rows = rows[~label[rows]]
  dead_rows = rows[label[rows]]
  figure, axis = plt.subplots(figsize=(6, 5.5))
  axis.scatter(small[alive_rows], large[alive_rows],
               s=5, alpha=0.18, label='alive')
  axis.scatter(small[dead_rows], large[dead_rows],
               s=7, alpha=0.35, label='dead')
  bounds = [float(min(small[rows].min(), large[rows].min())),
            float(max(small[rows].max(), large[rows].max()))]
  axis.plot(bounds, bounds, '--', color='black', lw=0.8)
  axis.set(xlabel='small-budget score', ylabel='large-budget score',
           title='Nested actor-sampling budget stability')
  axis.legend()
  figure.tight_layout()
  figure.savefig(path, dpi=160)
  plt.close(figure)


def _plot_checkpoint(path, run_metrics):
  names = list(run_metrics)
  roc = [run_metrics[name]['death_ranking']['roc_auc'] for name in names]
  ap = [run_metrics[name]['death_ranking']['average_precision']
        for name in names]
  matched = [run_metrics[name]['matched']['dead_score_below_alive_fraction']
             for name in names]
  tight = [run_metrics[name]['matched']['radius_sensitivity']['l2_le_0p5'][
      'dead_score_below_alive_fraction'] for name in names]
  x = np.arange(len(names))
  figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
  axes[0].bar(x - 0.18, roc, 0.36, label='ROC-AUC')
  axes[0].bar(x + 0.18, ap, 0.36, label='average precision')
  axes[0].set_ylim(0, 1.02)
  axes[0].set_xticks(x, names, rotation=25, ha='right')
  axes[0].set_title('Death ranking with -C (small budget)')
  axes[0].legend()
  axes[1].bar(x - 0.18, matched, 0.36, label='all constrained pairs')
  axes[1].bar(x + 0.18, tight, 0.36, label='full-F4 L2 <= 0.5')
  axes[1].axhline(0.5, ls='--', color='grey')
  axes[1].set_ylim(0, 1.02)
  axes[1].set_xticks(x, names, rotation=25, ha='right')
  axes[1].set_title('Matched dead score below alive')
  axes[1].legend()
  figure.tight_layout()
  figure.savefig(path, dpi=160)
  plt.close(figure)


def _plot_region(path, score, labels, region):
  names, values, colors = [], [], []
  dead = labels['dead_when_observed']
  for name in ('start', 'before_swamp', 'swamp_corridor', 'safe_route',
               'after_swamp', 'other'):
    for status, mask in [('alive', ~dead), ('dead', dead)]:
      chosen = (region == name) & mask
      if chosen.any():
        names.append(f'{name}\n{status}')
        values.append(score[chosen])
        colors.append('#d95f02' if status == 'dead' else '#1b9e77')
  figure, axis = plt.subplots(figsize=(12, 5.5))
  boxes = axis.boxplot(values, showfliers=False, patch_artist=True)
  for patch, color in zip(boxes['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
  axis.set_xticks(range(1, len(names) + 1), names, rotation=25, ha='right')
  axis.set_ylabel('raw continuation score C(s, g_task)')
  axis.set_title('Continuation scores by region and verified status')
  figure.tight_layout()
  figure.savefig(path, dpi=160)
  plt.close(figure)


def _fmt(value, digits=4):
  return 'n/a' if value is None else f'{value:.{digits}f}'


def _write_report(path, config, metrics):
  primary = metrics['primary']
  groups = primary['groups_large_budget']
  death = primary['death_ranking_large_budget']
  match = primary['matched_large_budget']
  tight_05 = match['radius_sensitivity']['l2_le_0p5']
  tight_10 = match['radius_sensitivity']['l2_le_1']
  budget = primary['budget_stability']
  gradient = primary['input_gradients']
  decision = metrics['decision']
  run = config['checkpoints']['primary']
  age = [groups[f'dead_age_{i}']['score']['mean'] for i in range(3)]
  age.append(groups['dead_established_age_ge_3']['score']['mean'])
  alive_move = groups['alive_moving']['score']['mean']
  waiting = groups['alive_waiting_or_near_stationary']['score']['mean']
  safe = groups['safe_route_alive']['score']['mean']
  goal_probe = primary['goal_slot_distinction']
  boot = primary['episode_bootstrap']
  comparison_summary = metrics['checkpoint_comparison_summary']
  examples = primary['examples']
  tight_interpretation = (
      'A rate below 50% on the adequately sized 0.5 subset is systematic '
      'inversion and is the decisive failed criterion.'
      if (tight_05['pairs'] >= 100
          and tight_05['dead_score_below_alive_fraction'] < 0.5) else
      'The adequately sized 0.5 subset is not systematically inverted.')

  def example_line(item):
    frames = ', '.join(
        f'({frame[0]:.4f},{frame[1]:.4f})'
        for frame in item['frames_newest_first'])
    goal_xy = item['commanded_goal_frames'][0]
    return (
        '- Episode %d, t=%d, death age %d, region `%s`, score `%.4f`; '
        'frames newest-first `[%s]`; commanded goal `(%.4f, %.4f)` tiled '
        'over four frames.' % (
            item['episode_id'], item['timestep'], item['death_age'],
            item['region'], item['continuation_score'], frames,
            goal_xy[0], goal_xy[1]))

  example_lines = [
      '### Highest-scoring failure states', '',
      *[example_line(item) for item in
        examples['highest_scoring_failure_states'][:3]], '',
      '### Lowest-scoring alive states', '',
      *[example_line(item) for item in
        examples['lowest_scoring_alive_states'][:3]], '',
  ]
  lines = [
      '# PointMaze frozen-critic continuation-score diagnostic', '',
      '## Decision', '',
      decision['text'], '',
      'The tested quantity is a surrogate ranking score. It is not a '
      'calibrated value, success probability, death probability, or '
      'worst-case Q. No ETT, nominal policy, actor, or critic parameter was '
      'trained or modified.', '',
      '## Evaluated score and checkpoint', '',
      ('The primary checkpoint is `%s` at step %d (failure-negative alpha '
       '`%g`, learner seed `%d`). Its SHA-256 is `%s`.') % (
           run['name'], run['step'], run['alpha'], run['seed'],
           run['checkpoint_sha256']), '',
      'The diagnostic evaluates', '',
      '$$C(s,g_{task}) = \\mathbb{E}_{a \\sim '
      '\\pi_{frozen}(\\cdot|s,g_{task})}'
      '[f_{frozen}(s,a,g_{task})].$$', '',
      ('`f` is the raw aligned contrastive logit. No sigmoid, exponential, '
       'temperature, or value conversion is applied. The training actor '
       'critic term minimizes `-diag(f)`; these runs have '
       '`entropy_coefficient=0`, `bc_coef=0.5`, and one critic head. The '
       'reported expectation uses %d nested samples; %d samples and the '
       'actor mode are stability checks.') % (
           config['evaluation']['large_samples'],
           config['evaluation']['small_samples']), '',
      'Contrastive positives are future achieved F4 goals from the same '
      'episode. Ordinary negatives are off-diagonal goals from other sampled '
      'episodes. Failure-negative training places established frozen bank '
      'states only in the GOAL slot. Actor training uses hindsight goals '
      '(including the `random_goals=0.5` batch roll); this diagnostic keeps '
      'the commanded task goal fixed, as deployment does.', '',
      '## Evaluation population and timing', '',
      ('Evaluation uses %s p=0.30 expert-positive validation transitions from '
       '%s source episodes. The split is useful for comparison with the '
       'nominal/ETT diagnostics, but it is in-sample for these critics: their '
       'offline CRL training used all 6,600 source episodes.') % (
           format(config['dataset']['split']['validation_transitions'], ','),
           format(config['dataset']['split']['validation_episodes'], ',')), '',
      'Death is reconstructed with the verified timing: the fatal-entry state '
      'is alive before its action; death age 0 is the first observation after '
      'fatal contact; the F4 history becomes fully established at age 3. '
      'Hidden swamp bits and death fields are used only for grouping.', '',
      '## Primary results', '',
      ('Using negative continuation score as an uncalibrated death ranking '
       'gives ROC-AUC `%s` and average precision `%s` at death prevalence '
       '`%s`. The episode-bootstrap intervals are ROC-AUC `%s` to `%s` and '
       'AP `%s` to `%s`.') % (
           _fmt(death['roc_auc'], 5), _fmt(death['average_precision'], 5),
           _fmt(death['death_prevalence'], 5),
           _fmt(boot['death_ranking']['roc_auc_95pct']['lower_2p5'], 5),
           _fmt(boot['death_ranking']['roc_auc_95pct']['upper_97p5'], 5),
           _fmt(boot['death_ranking']['average_precision_95pct']['lower_2p5'], 5),
           _fmt(boot['death_ranking']['average_precision_95pct']['upper_97p5'], 5)), '',
      ('Mean continuation score is `%.4f` for alive moving states and '
       '`%.4f` for alive waiting/near-stationary states. Death-age means are '
       '`%.4f`, `%.4f`, `%.4f`, and `%.4f` for ages 0, 1, 2, and at least 3. '
       'The early-age pattern must be read with the prior F4 observability '
       'result: recently dead histories still resemble moving histories.') % (
           alive_move, waiting, age[0], age[1], age[2], age[3]), '',
      ('There are %s cross-episode matched dead/alive pairs constrained to '
       'the same maze region and distance-to-goal band. Dead scores are lower '
       'in `%.2f%%` of pairs; mean dead-minus-alive score is `%.4f`. Median '
       'raw full-history F4 distance is `%.4f`. Matching narrows observed '
       'geometry and does not remove confounding.') % (
           format(match['pairs'], ','),
           100 * match['dead_score_below_alive_fraction'],
           match['score_dead_minus_alive']['mean'],
           match['full_f4_l2_distance']['median']), '',
      ('The tighter raw-F4 subsets contain %s pairs at distance at most 0.5 '
       'and %s pairs at distance at most 1.0; their dead-below-alive rates '
       'are `%s` and `%s`. These radius checks expose how the conclusion '
       'changes as observed-history similarity is tightened. %s') % (
           format(tight_05['pairs'], ','), format(tight_10['pairs'], ','),
           ('n/a' if tight_05['dead_score_below_alive_fraction'] is None else
            f"{100 * tight_05['dead_score_below_alive_fraction']:.2f}%"),
           ('n/a' if tight_10['dead_score_below_alive_fraction'] is None else
            f"{100 * tight_10['dead_score_below_alive_fraction']:.2f}%"),
           tight_interpretation), '',
      ('Across all six frozen checkpoints, small-budget death ROC-AUC ranges '
       'from `%.5f` to `%.5f`, AP from `%.5f` to `%.5f`, and matched '
       'dead-below-alive frequency from `%.2f%%` to `%.2f%%`. On the tight '
       'F4-distance-at-most-0.5 subset, every checkpoint is inverted: the '
       'range is only `%.2f%%` to `%.2f%%`. The alpha-specific seed means '
       'are stored in `metrics.json`. Failure '
       'negative training does not consistently improve this STATE-slot '
       'continuation ranking over alpha 0, so the observed signal cannot be '
       'attributed uniquely to the failure bank.') % (
           comparison_summary['ranges']['death_roc_auc']['minimum'],
           comparison_summary['ranges']['death_roc_auc']['maximum'],
           comparison_summary['ranges']['average_precision']['minimum'],
           comparison_summary['ranges']['average_precision']['maximum'],
           100 * comparison_summary['ranges'][
               'matched_dead_below_alive_fraction']['minimum'],
           100 * comparison_summary['ranges'][
               'matched_dead_below_alive_fraction']['maximum'],
           100 * comparison_summary['ranges'][
               'tight_l2_le_0p5_dead_below_alive_fraction']['minimum'],
           100 * comparison_summary['ranges'][
               'tight_l2_le_0p5_dead_below_alive_fraction']['maximum']), '',
      ('Alive safe-route states have mean score `%.4f`. The report records '
       'their frequency below the established-death median and below the '
       'alive-moving lower decile, so a sparse but legitimate detour is not '
       'silently treated as failure.') % safe, '',
      '## State slot versus goal slot', '',
      ('For authentic alive-moving anchors with recorded actions, the mean '
       'failure-bank GOAL score is below the commanded task-goal score for '
       '`%.2f%%` of anchors. Supplying the same bank observations in the '
       'STATE slot and pursuing the commanded goal gives mean continuation '
       'score `%.4f`. These are different queries; the first is directly '
       'trained as a negative, while the second is the proposed ETT '
       'objective and must be validated empirically.') % (
           100 * goal_probe[
               'per_anchor_failure_goal_below_task_goal_fraction'],
           goal_probe['same_failure_bank_state_in_state_slot_continuation'][
               'mean']), '',
      '## Stability and gradients', '',
      ('The %d-versus-%d sample scores have Spearman correlation `%.6f`, '
       'Pearson correlation `%.6f`, and mean absolute difference `%.5f`. '
       'Matched-pair score direction agrees in `%.2f%%` of cases. Results '
       'for alpha 0, 0.1, and 0.3 with seeds 0 and 1 are in `metrics.json`. '
       'Every checkpoint has `twin_q=False`, so no head reduction comparison '
       'exists.') % (
           config['evaluation']['small_samples'],
           config['evaluation']['large_samples'],
           budget['spearman'], budget['pearson'],
           budget['mean_absolute_difference'],
           100 * budget['matched_direction_agreement']), '',
      ('Direct critic-state gradients and total gradients through the frozen '
       'actor are finite (`%s` / `%s`). The actor path changes the state '
       'gradient in `%.2f%%` of checked authentic states. Frozen parameters '
       'do not detach state inputs. This does not validate rankings or '
       'gradients on arbitrary generated OOD states.') % (
           gradient['all_direct_finite'], gradient['all_total_finite'],
           100 * gradient['actor_path_nonzero_fraction']), '',
      '## Concrete authentic states', '',
      *example_lines,
      '## Limitations', '',
      '- The critic evaluation is in-sample at the episode level.',
      '- Failure labels and matching diagnose ranking; they do not calibrate '
      'the raw logit.',
      '- Safe-route support is sparse, and matching cannot remove hidden '
      'swamp or behavior-policy confounding.',
      '- Authentic offline states do not test arbitrary ETT-generated states.',
      '- This diagnostic neither recovers nor estimates worst-case Q.', '',
      '## Reproduction', '', '```bash',
      ('python scripts/diagnose_f4_critic_continuation.py \\\n'
       '  --dataset %s \\\n'
       '  --runs-root %s \\\n'
       '  --failure-bank %s \\\n'
       '  --out-dir %s \\\n'
       '  --small-samples %d --large-samples %d \\\n'
       '  --gradient-samples %d --bootstrap-replicates %d --seed %d') % (
           config['inputs']['dataset_argument'],
           config['inputs']['runs_root_argument'],
           config['inputs']['failure_bank_argument'],
           config['result']['out_dir_argument'],
           config['evaluation']['small_samples'],
           config['evaluation']['large_samples'],
           config['evaluation']['gradient_samples'],
           config['evaluation']['bootstrap_replicates'],
           config['evaluation']['seed']),
      '```', '',
      'If this score is used next, the complete two-loss ETT objective and '
      'transition constraint should be written down before any off-diagonal '
      'training. This task introduced no Lipschitz penalty, bank-distance '
      'objective, death classifier, or new bank.', '',
  ]
  with open(path, 'w', encoding='utf-8', newline='\n') as handle:
    handle.write('\n'.join(lines))


def main(argv=None):
  args = build_parser().parse_args(argv)
  counts = (args.small_samples, args.large_samples, args.gradient_samples,
            args.goal_probe_anchors, args.bootstrap_replicates,
            args.batch_size)
  if min(counts) <= 0 or args.small_samples > args.large_samples:
    raise SystemExit('count arguments must be positive and small <= large')
  run_names = tuple(x for x in args.run_names.split(',') if x)
  if args.primary_run not in run_names:
    raise SystemExit('--primary-run must be present in --run-names')
  out_dir = pathlib.Path(args.out_dir)
  config_path = out_dir / 'config.json'
  if config_path.exists() and not args.overwrite:
    raise SystemExit('output directory exists; pass --overwrite')
  out_dir.mkdir(parents=True, exist_ok=True)

  dataset = ExpertTransitionDataset(args.dataset, val_frac=0.1, split_seed=0)
  arrays = dataset.arrays('validation')
  labels = _reconstruct_temporal_labels(args.dataset, arrays)
  dead = labels['dead_when_observed']
  region = _regions(arrays.state)
  goal_distance, distance_band = _distance_bands(arrays.state, arrays.goal)
  masks = _evaluation_masks(arrays, labels, region)
  dataset_content_sha = _content_sha256(args.dataset)
  with np.load(args.failure_bank, allow_pickle=False) as bank_data:
    bank = np.asarray(bank_data['goals'], dtype=np.float32)
  bank_content_sha = _content_sha256(args.failure_bank)
  runs = _load_runs(
      args.runs_root, run_names, dataset_content_sha, bank_content_sha)

  # Every offline CRL run used all source episodes. The validation split below
  # belongs to later nominal/ETT diagnostics and is therefore not critic-heldout.
  critic_training_episode_ids = np.arange(
      dataset.behavior_report['source_n_episodes'], dtype=np.int64)
  unseen_ids = np.setdiff1d(
      dataset.validation_episode_ids, critic_training_episode_ids)

  rng = np.random.default_rng(args.seed)
  epsilon_large = rng.normal(
      size=(len(arrays.state), args.large_samples, 2)).astype(np.float32)
  epsilon_small = epsilon_large[:, :args.small_samples]
  alive_moving_rows = np.flatnonzero(masks['alive_moving'])
  anchor_count = min(args.goal_probe_anchors, len(alive_moving_rows))
  anchor_rows = rng.choice(
      alive_moving_rows, size=anchor_count, replace=False)
  epsilon_bank = rng.normal(
      size=(len(bank), args.small_samples, 2)).astype(np.float32)

  query, partner = _matched_alive(dead, arrays, region, distance_band)
  if not len(query):
    raise RuntimeError('no constrained cross-episode dead/alive pairs found')

  network = _network()
  paired_many, sampled, mode = _make_scorers(network)
  scores = {}
  comparison = {}
  primary_state = None
  primary_run = None
  primary_large = None
  primary_goal_probe = None
  for run in runs:
    _, state_obj = checkpoint_mod.load_checkpoint(run['checkpoint'])
    small = _continuation_scores(
        sampled, mode, arrays.state, arrays.goal, epsilon_small,
        state_obj.policy_params, state_obj.q_params, args.batch_size)
    small_score = small['mean']
    goal_probe = _goal_slot_probe(
        paired_many, sampled, mode, state_obj, arrays, masks, bank,
        epsilon_bank, anchor_rows, args.batch_size)
    comparison[run['name']] = {
        'alpha': run['alpha'],
        'learner_seed': run['seed'],
        'death_ranking': _binary_metrics(dead, -small_score),
        'matched': _matched_report(
            small_score, query, partner, arrays, labels, goal_distance),
        'groups': _group_report(small_score, arrays, masks),
        'goal_slot_distinction': goal_probe,
        'mode_vs_sample': {
            'spearman': float(spearmanr(
                small['mode_score'], small_score).statistic),
            'pearson': float(pearsonr(
                small['mode_score'], small_score).statistic),
            'mean_absolute_difference': float(np.mean(np.abs(
                small['mode_score'] - small_score))),
        },
    }
    scores[run['name']] = small_score
    if run['name'] == args.primary_run:
      primary_state, primary_run = state_obj, run
      primary_goal_probe = goal_probe
      primary_large = _continuation_scores(
          sampled, mode, arrays.state, arrays.goal, epsilon_large,
          state_obj.policy_params, state_obj.q_params, args.batch_size)
    del state_obj

  by_alpha = {}
  for alpha in sorted(set(run['alpha'] for run in runs)):
    selected = [value for name, value in comparison.items()
                if value['alpha'] == alpha]
    by_alpha[f'{alpha:g}'] = {
        'runs': len(selected),
        'mean_death_roc_auc': float(np.mean([
            value['death_ranking']['roc_auc'] for value in selected])),
        'mean_average_precision': float(np.mean([
            value['death_ranking']['average_precision']
            for value in selected])),
        'mean_matched_dead_below_alive_fraction': float(np.mean([
            value['matched']['dead_score_below_alive_fraction']
            for value in selected])),
        'mean_tight_l2_le_0p5_dead_below_alive_fraction': float(np.mean([
            value['matched']['radius_sensitivity']['l2_le_0p5'][
                'dead_score_below_alive_fraction'] for value in selected])),
    }
  comparison_values = list(comparison.values())
  comparison_summary = {
      'by_failure_negative_alpha_across_seeds': by_alpha,
      'ranges': {
          'death_roc_auc': {
              'minimum': float(min(value['death_ranking']['roc_auc']
                                   for value in comparison_values)),
              'maximum': float(max(value['death_ranking']['roc_auc']
                                   for value in comparison_values)),
          },
          'average_precision': {
              'minimum': float(min(
                  value['death_ranking']['average_precision']
                  for value in comparison_values)),
              'maximum': float(max(
                  value['death_ranking']['average_precision']
                  for value in comparison_values)),
          },
          'matched_dead_below_alive_fraction': {
              'minimum': float(min(
                  value['matched']['dead_score_below_alive_fraction']
                  for value in comparison_values)),
              'maximum': float(max(
                  value['matched']['dead_score_below_alive_fraction']
                  for value in comparison_values)),
          },
          'tight_l2_le_0p5_dead_below_alive_fraction': {
              'minimum': float(min(
                  value['matched']['radius_sensitivity']['l2_le_0p5'][
                      'dead_score_below_alive_fraction']
                  for value in comparison_values)),
              'maximum': float(max(
                  value['matched']['radius_sensitivity']['l2_le_0p5'][
                      'dead_score_below_alive_fraction']
                  for value in comparison_values)),
          },
      },
      'interpretation': (
          'failure-negative alpha does not consistently improve state-slot '
          'continuation ranking over alpha 0 across the two learner seeds'),
  }

  large_score = primary_large['mean']
  small_score = scores[args.primary_run]
  small_difference = small_score[query] - small_score[partner]
  large_difference = large_score[query] - large_score[partner]
  budget = {
      'nested_samples': True,
      'small_count': int(args.small_samples),
      'large_count': int(args.large_samples),
      'pearson': float(pearsonr(small_score, large_score).statistic),
      'spearman': float(spearmanr(small_score, large_score).statistic),
      'mean_absolute_difference': float(np.mean(np.abs(
          small_score - large_score))),
      'p95_absolute_difference': float(np.percentile(np.abs(
          small_score - large_score), 95)),
      'matched_direction_agreement': float(np.mean(
          (small_difference < 0) == (large_difference < 0))),
      'small_death_ranking': _binary_metrics(dead, -small_score),
      'large_death_ranking': _binary_metrics(dead, -large_score),
      'large_sample_rowwise_monte_carlo_sd': _dist(
          primary_large['sample_std']),
      'large_sample_rowwise_standard_error': _dist(
          primary_large['sample_std'] / math.sqrt(args.large_samples)),
      'mode_vs_large': {
          'pearson': float(pearsonr(
              primary_large['mode_score'], large_score).statistic),
          'spearman': float(spearmanr(
              primary_large['mode_score'], large_score).statistic),
          'mean_absolute_difference': float(np.mean(np.abs(
              primary_large['mode_score'] - large_score))),
      },
  }

  matched_large = _matched_report(
      large_score, query, partner, arrays, labels, goal_distance)
  established = masks['dead_established_age_ge_3']
  alive_moving = masks['alive_moving']
  episode_bootstrap = {
      'death_ranking': _bootstrap_binary(
          dead, -large_score, arrays.episode_id,
          args.bootstrap_replicates, args.seed + 101),
      'established_dead_minus_alive_moving': _bootstrap_group_difference(
          large_score, arrays.episode_id, established, alive_moving,
          args.bootstrap_replicates, args.seed + 102),
      'matched': _bootstrap_matched(
          large_score, query, partner, arrays.episode_id,
          args.bootstrap_replicates, args.seed + 103),
  }
  established_median = float(np.median(large_score[established]))
  alive_lower_decile = float(np.percentile(large_score[alive_moving], 10))
  legitimate_alive = {}
  for name in ('alive_waiting_or_near_stationary', 'alive_zero_action_waiting',
               'alive_near_stationary', 'safe_route_alive'):
    chosen = masks[name]
    legitimate_alive[name] = {
        'count': int(chosen.sum()),
        'episodes': int(np.unique(arrays.episode_id[chosen]).size),
        'score': _dist(large_score[chosen]),
        'fraction_below_established_dead_median': float(np.mean(
            large_score[chosen] < established_median)) if chosen.any() else None,
        'fraction_below_alive_moving_p10': float(np.mean(
            large_score[chosen] < alive_lower_decile)) if chosen.any() else None,
    }

  gradients = _gradient_report(
      network, primary_state, arrays, labels, masks,
      args.gradient_samples, args.seed + 301)
  examples = _examples(large_score, arrays, labels, region)
  primary = {
      'run': args.primary_run,
      'groups_large_budget': _group_report(large_score, arrays, masks),
      'regions_and_goal_distance_large_budget': _region_report(
          large_score, arrays, labels, region, distance_band),
      'death_ranking_large_budget': _binary_metrics(dead, -large_score),
      'matched_large_budget': matched_large,
      'episode_bootstrap': episode_bootstrap,
      'legitimate_alive_checks': legitimate_alive,
      'budget_stability': budget,
      'goal_slot_distinction': primary_goal_probe,
      'input_gradients': gradients,
      'examples': examples,
      'actor_distribution': {
          'mode_action': {
              'minimum': primary_large['mode_action'].min(axis=0).tolist(),
              'maximum': primary_large['mode_action'].max(axis=0).tolist(),
          },
          'policy_scale': _dist(primary_large['policy_scale'].ravel()),
      },
  }

  safe_bad = legitimate_alive['safe_route_alive'][
      'fraction_below_established_dead_median']
  tight_matched = matched_large['radius_sensitivity']['l2_le_0p5']
  criteria = {
      'death_roc_auc_at_least_0p70': bool(
          primary['death_ranking_large_budget']['roc_auc'] >= 0.70),
      'matched_dead_below_alive_at_least_0p60': bool(
          matched_large['dead_score_below_alive_fraction'] >= 0.60),
      'closest_matched_l2_le_0p5_has_at_least_100_pairs': bool(
          tight_matched['pairs'] >= 100),
      'closest_matched_l2_le_0p5_not_systematically_inverted': bool(
          tight_matched['dead_score_below_alive_fraction'] is not None
          and tight_matched['dead_score_below_alive_fraction'] >= 0.50),
      'established_dead_mean_below_alive_moving': bool(
          large_score[established].mean() < large_score[alive_moving].mean()),
      'small_large_spearman_at_least_0p95': bool(
          budget['spearman'] >= 0.95),
      'safe_route_not_majority_below_established_dead_median': bool(
          safe_bad is not None and safe_bad < 0.50),
      'authentic_state_gradients_finite': bool(
          gradients['all_direct_finite'] and gradients['all_total_finite']),
  }
  provisional = all(criteria.values())
  decision = {
      'provisional_objective_supported': provisional,
      'bounded_decision_criteria': criteria,
      'text': (
          'The frozen raw contrastive continuation score passes the bounded '
          'diagnostic and is suitable only as a provisional surrogate '
          'objective for a tightly monitored off-diagonal experiment.'
          if provisional else
          'The frozen raw contrastive continuation score separates obvious '
          'failure histories globally, but it is systematically inverted on '
          'the closest authentic dead/alive matches and is not suitable as '
          'the pessimistic term in off-diagonal ETT training.'),
      'next_step_if_passed': (
          'write down the complete two-loss ETT objective and transition '
          'constraint before any off-diagonal training'),
      'most_important_issue_if_unsuitable': (
          'the goal-slot failure-negative construction does not transfer to '
          'a reliable state-slot continuation ordering for the closest '
          'authentic dead/alive histories; resolve this semantics and '
          'held-out generalization issue before defining the ETT objective'),
      'not_claimed': ['worst-case Q recovery', 'calibrated value',
                      'OOD generated-state validity'],
  }

  metrics = {
      'all_checks_pass': bool(
          np.isfinite(large_score).all()
          and np.isfinite(small_score).all()
          and len(query) > 0
          and gradients['all_direct_finite']
          and gradients['all_total_finite']),
      'score_semantics': _semantics(),
      'population': {
          'critic_episode_status': 'in_sample',
          'critic_training_source_episodes': int(
              len(critic_training_episode_ids)),
          'evaluation_validation_episodes': int(
              len(dataset.validation_episode_ids)),
          'evaluation_episode_ids_unseen_by_critic': unseen_ids.tolist(),
          'reason': (
              'offline CRL consumed the complete 6600-episode dataset; this '
              'split was created later for nominal-policy and ETT work'),
      },
      'checkpoint_comparison_small_budget': comparison,
      'checkpoint_comparison_summary': comparison_summary,
      'primary': primary,
      'decision': decision,
  }

  runtime = {
      'python': sys.version,
      'jax': jax.__version__,
      'jax_backend': jax.default_backend(),
      'jax_devices': [str(device) for device in jax.devices()],
      'numpy': np.__version__,
  }
  primary_meta = next(x for x in runs if x['name'] == args.primary_run)
  config = {
      'format_version': 1,
      'task': 'pointmaze_f4_frozen_critic_continuation_score_diagnostic',
      'status': 'complete',
      'scope': {
          'pointmaze_only': True,
          'ett_trained_or_modified': False,
          'nominal_policy_trained_or_modified': False,
          'actor_trained_or_modified': False,
          'critic_trained_or_modified': False,
          'off_diagonal_training_implemented': False,
          'lipschitz_constraint_implemented': False,
          'new_failure_bank_created': False,
      },
      'inputs': {
          'dataset_argument': args.dataset,
          'runs_root_argument': args.runs_root,
          'failure_bank_argument': args.failure_bank,
          'dataset_path': str(pathlib.Path(args.dataset).resolve()),
          'dataset_file_sha256': _sha256(args.dataset),
          'dataset_content_sha256': dataset_content_sha,
          'failure_bank_path': str(pathlib.Path(args.failure_bank).resolve()),
          'failure_bank_file_sha256': _sha256(args.failure_bank),
          'failure_bank_content_sha256': bank_content_sha,
      },
      'dataset': dataset.report,
      'checkpoints': {
          'primary': primary_meta,
          'all': runs,
          'architecture': {
              'obs_dim': 8, 'goal_dim': 8, 'action_dim': 2,
              'hidden_layer_sizes': [256, 256], 'repr_dim': 64,
              'repr_norm': False, 'repr_norm_temp': True,
              'twin_q': False, 'use_layer_norm': False,
              'obs_normalization': None,
          },
          'shared_training_config': {
              'use_td': False, 'use_cpc': False, 'use_gcbc': False,
              'bc_coef': 0.5, 'random_goals': 0.5,
              'entropy_coefficient': 0.0, 'discount': 0.95,
              'batch_size': 256, 'learning_rate': 3e-4,
              'actor_learning_rate': 3e-4,
              'num_sgd_steps_per_step': 10, 'training_steps': 150000,
          },
      },
      'score': {
          'expression': (
              'C(s,g_task)=E_{a~pi_frozen(.|s,g_task)}'
              '[raw_f_frozen(s,a,g_task)]'),
          'primary_action_estimator': 'nested Monte Carlo mean',
          'task_goal': 'stored commanded task goal held fixed',
          'failure_observation_slot': 'state',
          'interpretation': 'uncalibrated contrastive ranking surrogate',
      },
      'evaluation': {
          'small_samples': args.small_samples,
          'large_samples': args.large_samples,
          'gradient_samples': args.gradient_samples,
          'goal_probe_anchors': anchor_count,
          'bootstrap_replicates': args.bootstrap_replicates,
          'batch_size': args.batch_size,
          'seed': args.seed,
          'nested_noise': True,
          'death_label_input_to_score': False,
          'hidden_fields_input_to_score': False,
          'matching_distance': 'raw full-F4 Euclidean distance',
          'distance_to_goal_band_edges': [
              0.0, 2.0, 3.0, 4.0, 5.0, 'infinity'],
      },
      'source': _git_provenance(),
      'runtime': runtime,
      'result': {
          'out_dir_argument': args.out_dir,
          'all_checks_pass': metrics['all_checks_pass'],
          'provisional_objective_supported': provisional,
      },
  }

  plot_paths = {
      'score_by_death_age': 'score_by_death_age.png',
      'death_roc_pr': 'death_roc_pr.png',
      'sampling_budget_stability': 'sampling_budget_stability.png',
      'checkpoint_comparison': 'checkpoint_comparison.png',
      'score_by_region': 'score_by_region.png',
  }
  _plot_age(out_dir / plot_paths['score_by_death_age'], large_score, masks)
  _plot_roc_pr(
      out_dir / plot_paths['death_roc_pr'], dead, large_score,
      args.primary_run)
  _plot_budget(
      out_dir / plot_paths['sampling_budget_stability'],
      small_score, large_score, dead)
  _plot_checkpoint(
      out_dir / plot_paths['checkpoint_comparison'], comparison)
  _plot_region(
      out_dir / plot_paths['score_by_region'], large_score, labels, region)
  metrics['plots'] = plot_paths
  config['result']['plots'] = plot_paths

  _write_json(out_dir / 'metrics.json', metrics)
  _write_json(config_path, config)
  _write_report(out_dir / 'REPORT.md', config, metrics)
  print(json.dumps({
      'out_dir': str(out_dir),
      'all_checks_pass': metrics['all_checks_pass'],
      'decision': decision,
      'primary_death_ranking': primary['death_ranking_large_budget'],
      'primary_matched': {
          'pairs': matched_large['pairs'],
          'dead_below_alive_fraction': matched_large[
              'dead_score_below_alive_fraction'],
      },
  }, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
