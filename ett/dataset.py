"""Aligned expert-positive transitions for diagonal PointMaze ETT fitting.

The training view contains learner-visible pre-action state, recorded action,
the next learner-visible state, and commanded goal. Audit-only labels are
loaded by :func:`load_audit_labels` and remain separate from training arrays.
"""
import dataclasses
import hashlib
import json

import numpy as np

from propensity.dataset import BehaviorDataset, DatasetContractError
from propensity.expert_population import resolve_expert_positive_episodes


SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))


@dataclasses.dataclass(frozen=True)
class TransitionArrays:
  state: np.ndarray
  action: np.ndarray
  next_state: np.ndarray
  goal: np.ndarray
  episode_id: np.ndarray
  timestep: np.ndarray

  @property
  def delta_xy(self):
    return self.next_state[:, :2] - self.state[:, :2]

  @property
  def context(self):
    return np.concatenate(
        [self.state, self.action, self.action, self.goal], axis=-1)

  def take(self, indices):
    return TransitionArrays(
        state=self.state[indices], action=self.action[indices],
        next_state=self.next_state[indices], goal=self.goal[indices],
        episode_id=self.episode_id[indices], timestep=self.timestep[indices])


def _episode_hash(ids):
  values = np.ascontiguousarray(np.sort(np.asarray(ids, dtype=np.int64)))
  return hashlib.sha256(values.tobytes()).hexdigest()


def _duplicate_audit(arrays, train_ids, validation_ids):
  signatures = {}
  episodes = arrays.episode_id
  starts = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1]
  stops = np.r_[starts[1:], len(episodes)]
  for start, stop in zip(starts, stops):
    digest = hashlib.sha256()
    for value in (arrays.state[start:stop], arrays.action[start:stop],
                  arrays.next_state[start:stop], arrays.goal[start:stop]):
      digest.update(np.ascontiguousarray(value).tobytes())
    signatures.setdefault(digest.hexdigest(), []).append(
        int(episodes[start]))
  duplicate_groups = [value for value in signatures.values()
                      if len(value) > 1]
  train_ids, validation_ids = set(train_ids), set(validation_ids)
  cross_split = [value for value in duplicate_groups
                 if train_ids.intersection(value)
                 and validation_ids.intersection(value)]
  return {
      'definition': ('byte-identical complete selected '
                     '(state, action, next_state, goal) trajectories'),
      'duplicate_groups': len(duplicate_groups),
      'duplicate_episodes': int(sum(map(len, duplicate_groups))),
      'cross_split_duplicate_groups': len(cross_split),
      'cross_split_episode_ids': sorted(
          {episode for group in cross_split for episode in group}),
      'passed': not cross_split,
  }


class ExpertTransitionDataset:
  """Immutable aligned transition view with the established expert split."""

  def __init__(self, path, val_frac=0.1, split_seed=0):
    selected_ids, population = resolve_expert_positive_episodes(path)
    behavior = BehaviorDataset(
        path, val_frac=val_frac, seed=split_seed, state_mode='obs',
        split_level='episode', strict_bounds=True,
        include_episode_ids=selected_ids, split_reference='source')
    passed, gates, details = behavior.check()
    if not passed:
      raise DatasetContractError(f'behavior dataset checks failed: {gates}')

    with np.load(path, allow_pickle=False) as source:
      observation = np.asarray(source['obs'], dtype=np.float32)
      action = np.asarray(source['act'], dtype=np.float32)
      metadata = json.loads(str(source['meta']))
      lengths = (np.asarray(source['lengths'], dtype=np.int64)
                 if 'lengths' in source.files else
                 np.full(observation.shape[0], observation.shape[1],
                         dtype=np.int64))

    state_dim = int(metadata['obs_dim'])
    goal_dim = int(metadata['goal_dim'])
    episode_id = np.repeat(
        np.arange(observation.shape[0], dtype=np.int64), lengths - 1)
    timestep = np.concatenate([
        np.arange(int(length) - 1, dtype=np.int64) for length in lengths])
    selected_rows = np.isin(episode_id, selected_ids)
    episode_id = episode_id[selected_rows]
    timestep = timestep[selected_rows]

    state = observation[episode_id, timestep, :state_dim]
    next_state = observation[episode_id, timestep + 1, :state_dim]
    goal = observation[episode_id, timestep, state_dim:state_dim + goal_dim]
    next_goal = observation[
        episode_id, timestep + 1, state_dim:state_dim + goal_dim]
    recorded_action = action[episode_id, timestep]
    arrays = TransitionArrays(
        state=np.ascontiguousarray(state),
        action=np.ascontiguousarray(recorded_action),
        next_state=np.ascontiguousarray(next_state),
        goal=np.ascontiguousarray(goal),
        episode_id=np.ascontiguousarray(episode_id),
        timestep=np.ascontiguousarray(timestep))

    if arrays.state.shape[1] != 8 or arrays.goal.shape[1] != 8:
      raise DatasetContractError('the F4 transition view requires state/goal 8/8')
    if not np.array_equal(np.concatenate([state, goal], axis=-1),
                          behavior._state):
      raise DatasetContractError('aligned observations disagree with loader')
    if not np.array_equal(recorded_action, behavior._action):
      raise DatasetContractError('aligned actions disagree with loader')
    if not np.array_equal(goal, next_goal):
      raise DatasetContractError('commanded goal changes inside an episode')

    frame_error = np.abs(next_state[:, 2:] - state[:, :-2])
    frame_consistent = bool(np.all(frame_error == 0.0))
    if not frame_consistent:
      raise DatasetContractError('stored F4 transitions do not shift exactly')

    train_mask = np.zeros(len(arrays.state), dtype=bool)
    validation_mask = np.zeros(len(arrays.state), dtype=bool)
    train_mask[behavior._train_idx] = True
    validation_mask[behavior._val_idx] = True
    if np.any(train_mask & validation_mask) or not np.all(
        train_mask | validation_mask):
      raise DatasetContractError('transition split is not a disjoint partition')
    train_ids = np.unique(episode_id[train_mask])
    validation_ids = np.unique(episode_id[validation_mask])
    duplicate_audit = _duplicate_audit(
        arrays, train_ids.tolist(), validation_ids.tolist())
    if not duplicate_audit['passed']:
      raise DatasetContractError('duplicate trajectories cross the split')

    self.path = path
    self.arrays_all = arrays
    self._train_indices = np.flatnonzero(train_mask)
    self._validation_indices = np.flatnonzero(validation_mask)
    self.population_manifest = population
    self.metadata = metadata
    self.behavior_report = behavior.report()
    self.train_episode_ids = train_ids
    self.validation_episode_ids = validation_ids
    self.report = {
        'dataset': self.behavior_report,
        'population': population,
        'transition_alignment': {
            'definition': '(s_t, recorded a_t, s_{t+1}, commanded g_t)',
            'terminal_dummy_actions_excluded': True,
            'padding_excluded': True,
            'all_selected_transitions_retained': True,
            'frame_order': 'newest XY first',
            'modeled_target': 'newest XY displacement in maze units',
            'deterministic_reconstruction':
                'next_state[2:8] = state[0:6] exactly',
            'observed_frame_shift_max_abs_error': float(
                frame_error.max(initial=0.0)),
            'observed_frame_shift_exact_fraction': float(
                np.mean(np.all(frame_error == 0.0, axis=1))),
            'commanded_goal_constant_across_transitions': True,
        },
        'split': {
            'reference': 'historical full-source split restricted to expert',
            'seed': int(split_seed), 'validation_fraction': float(val_frac),
            'train_episodes': int(len(train_ids)),
            'validation_episodes': int(len(validation_ids)),
            'train_transitions': int(train_mask.sum()),
            'validation_transitions': int(validation_mask.sum()),
            'episode_overlap': int(np.intersect1d(
                train_ids, validation_ids).size),
            'train_episode_ids': train_ids.tolist(),
            'validation_episode_ids': validation_ids.tolist(),
            'train_episode_ids_sha256': _episode_hash(train_ids),
            'validation_episode_ids_sha256': _episode_hash(validation_ids),
            'duplicate_trajectory_audit': duplicate_audit,
        },
        'behavior_checks': {'passed': passed, 'gates': gates,
                            'details': details},
    }

  def arrays(self, split='all'):
    if split == 'all':
      return self.arrays_all
    if split == 'train':
      return self.arrays_all.take(self._train_indices)
    if split in ('validation', 'val'):
      return self.arrays_all.take(self._validation_indices)
    raise ValueError("split must be 'all', 'train', or 'validation'")


def load_audit_labels(path, arrays):
  """Return evaluation-only death and swamp labels aligned to ``arrays``.

  This function is never called by the trainer. Its output must not be added
  to a model context.
  """
  with np.load(path, allow_pickle=False) as source:
    required = {'swamp_bits', 'entered_active_swamp'}
    missing = required - set(source.files)
    if missing:
      raise DatasetContractError(f'audit labels missing: {sorted(missing)}')
    bits = np.asarray(source['swamp_bits'], dtype=bool)
    died = np.asarray(source['entered_active_swamp'], dtype=bool)
    observation = np.asarray(source['obs'], dtype=np.float32)

  death_row = np.full(len(died), -1, dtype=np.int64)
  for episode in np.unique(arrays.episode_id[died[arrays.episode_id]]):
    episode = int(episode)
    length = observation.shape[1]
    for timestep in range(length - 1):
      cell = tuple(np.floor(observation[episode, timestep + 1, :2]).astype(int))
      if cell in SWAMP_CELLS and bits[episode, timestep, SWAMP_CELLS.index(cell)]:
        death_row[episode] = timestep + 1
        break
    if death_row[episode] < 1:
      raise DatasetContractError(
          f'death row cannot be reconstructed for episode {episode}')

  row_death = death_row[arrays.episode_id]
  entering_death = died[arrays.episode_id] & (
      arrays.timestep == row_death - 1)
  post_death = died[arrays.episode_id] & (arrays.timestep >= row_death)
  next_cell = np.floor(arrays.next_state[:, :2]).astype(np.int64)
  landed_swamp = np.zeros(len(next_cell), dtype=bool)
  landed_active = np.zeros(len(next_cell), dtype=bool)
  for index, cell in enumerate(SWAMP_CELLS):
    here = np.all(next_cell == np.asarray(cell), axis=1)
    landed_swamp |= here
    landed_active |= here & bits[
        arrays.episode_id, arrays.timestep, index]
  exact_stationary = np.all(arrays.delta_xy == 0.0, axis=1)
  return {
      'episode_died': died[arrays.episode_id],
      'entering_death': entering_death,
      'post_death': post_death,
      'landed_swamp': landed_swamp,
      'landed_active': landed_active,
      'landed_clear_swamp': landed_swamp & ~landed_active,
      'exact_stationary': exact_stationary,
      'death_row_by_source_episode': death_row,
  }
