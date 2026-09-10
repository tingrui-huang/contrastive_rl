"""Authoritative PointMaze nominal-expert population selection.

The merged F4 file contains three source populations with different purposes:

* uniform coverage episodes created by ``collect_swamp_windy.collect``;
* episodes whose actions come from ``make_windy_teacher``;
* appended blind-demonstrator episodes created by the separate bad-demo
  collector for causal support.

Only the middle population defines the nominal expert action distribution.
Selection uses original episode ids and collector provenance. ``teacher_mode``
is read only to verify the source boundaries encoded by the collectors; it is
never returned as a policy feature.
"""
import hashlib
import json
import os

import numpy as np

from propensity.dataset import DatasetContractError, sha256_file


EXPERT_GENERATOR = 'scripts.collect_swamp_windy.make_windy_teacher'
EXPECTED_ENV = 'point_two_route_swamp_windy_f4_v0'
EXPERT_MODE_NAMES = ('forced_safe', 'immediate_shortcut', 'wait_shortcut')


def _array_digest(named_arrays):
  digest = hashlib.sha256()
  for name, value in named_arrays:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def _id_digest(ids):
  return hashlib.sha256(
      np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _ranges(ids):
  ids = np.asarray(ids, dtype=np.int64)
  if ids.size == 0:
    return []
  starts = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1]
  stops = np.r_[starts[1:], ids.size]
  return [[int(ids[s]), int(ids[t - 1] + 1)] for s, t in zip(starts, stops)]


def resolve_expert_positive_episodes(path):
  """Return original expert episode ids and a reproducible source manifest."""
  if not os.path.exists(path):
    raise DatasetContractError(f'dataset not found: {path}')
  with np.load(path, allow_pickle=False) as data:
    required = {'obs', 'act', 'teacher_mode', 'meta'}
    missing = required - set(data.files)
    if missing:
      raise DatasetContractError(
          f'expert population cannot be established; missing {sorted(missing)}')
    try:
      metadata = json.loads(str(data['meta']))
    except (TypeError, ValueError) as error:
      raise DatasetContractError('dataset metadata is not valid JSON') from error
    observation = np.asarray(data['obs'])
    action = np.asarray(data['act'])
    modes = np.asarray(data['teacher_mode'], dtype=np.int64)
    lengths = (np.asarray(data['lengths'], dtype=np.int64)
               if 'lengths' in data.files else
               np.full(observation.shape[0], observation.shape[1], np.int64))

  if metadata.get('env_name') != EXPECTED_ENV:
    raise DatasetContractError(
        f'expected {EXPECTED_ENV}, got {metadata.get("env_name")}')
  merge = metadata.get('merge')
  if not isinstance(merge, dict):
    raise DatasetContractError(
        'expert source selection requires the merged-file provenance block')
  main_range = merge.get('main_index_range')
  bad_range = merge.get('bad_index_range')
  if (main_range != [0, merge.get('main_episodes')]
      or bad_range != [merge.get('main_episodes'), observation.shape[0]]):
    raise DatasetContractError('merged source episode ranges are inconsistent')
  n_main = int(merge['main_episodes'])
  n_bad = int(merge['bad_episodes'])
  if n_main + n_bad != observation.shape[0] or modes.shape != (n_main + n_bad,):
    raise DatasetContractError('source counts do not match episode arrays')

  mode_codes = metadata.get('teacher_mode_code', {})
  if not all(name in mode_codes for name in ('random',) + EXPERT_MODE_NAMES):
    raise DatasetContractError('main collector mode-code provenance is incomplete')
  expert_codes = np.asarray([mode_codes[name] for name in EXPERT_MODE_NAMES],
                            dtype=np.int64)
  random_code = int(mode_codes['random'])
  bad_code = int(merge.get('bad_demo_teacher_mode_code', -1))
  random_count = int(round(n_main * float(metadata.get('random_frac', -1.0))))
  if random_count < 0:
    raise DatasetContractError('main collector random_frac is missing')

  # These assertions recover the source assignment in collect_swamp_windy:
  # ``is_random = ep < n_random`` and every later main-source action comes from
  # ``make_windy_teacher``. The appended range is independently asserted by
  # merge_swamp_windy_baddemo. No outcome or hidden-state field participates.
  if not np.all(modes[:random_count] == random_code):
    raise DatasetContractError('uniform-coverage source range does not match modes')
  if not np.all(np.isin(modes[random_count:n_main], expert_codes)):
    raise DatasetContractError('nominal-teacher source range has unexpected modes')
  if not np.all(modes[n_main:] == bad_code):
    raise DatasetContractError('appended bad-demonstrator source range is invalid')

  selected = np.arange(random_count, n_main, dtype=np.int64)
  excluded_coverage = np.arange(0, random_count, dtype=np.int64)
  excluded_bad = np.arange(n_main, n_main + n_bad, dtype=np.int64)
  transition_count = int(np.sum(lengths[selected] - 1))
  episode_payload_sha = _array_digest([
      ('obs', observation[selected]), ('act', action[selected]),
      ('lengths', lengths[selected])])
  manifest = {
      'format_version': 1,
      'population': 'nominal_expert_positive',
      'definition': (
          'Original main-source episodes whose actions were generated by '
          'make_windy_teacher; uniform coverage and the separately collected '
          'blind demonstrator are excluded from the nominal expert policy.'),
      'selection_predicate': (
          f'original_episode_id in [{random_count}, {n_main}); source bounds '
          'are verified against teacher_mode codes but labels are not inputs'),
      'selection_uses_outcome_or_hidden_state': False,
      'audit_labels_are_policy_inputs': False,
      'expert_generator': EXPERT_GENERATOR,
      'dataset_path': os.path.abspath(path),
      'dataset_sha256': sha256_file(path),
      'environment': metadata['env_name'],
      'per_cell_swamp_probability': metadata.get('per_cell_swamp_prob'),
      'source_provenance': {
          'main_name': merge.get('main_name'),
          'main_content_sha256': merge.get('main_content_sha256'),
          'main_episode_range': main_range,
          'appended_name': merge.get('bad_name'),
          'appended_content_sha256': merge.get('bad_content_sha256'),
          'appended_episode_range': bad_range,
      },
      'selected': {
          'episode_count': int(selected.size),
          'transition_count': transition_count,
          'original_episode_id_ranges': _ranges(selected),
          'original_episode_ids_sha256': _id_digest(selected),
          'episode_payload_sha256': episode_payload_sha,
          'mode_codes_used_for_source_verification': expert_codes.tolist(),
          'mode_counts': {
              name: int(np.sum(modes[selected] == int(mode_codes[name])))
              for name in EXPERT_MODE_NAMES},
      },
      'excluded': {
          'uniform_coverage': {
              'episode_count': int(excluded_coverage.size),
              'transition_count': int(np.sum(lengths[excluded_coverage] - 1)),
              'original_episode_id_ranges': _ranges(excluded_coverage),
              'purpose': 'coverage, not output of the nominal teacher'},
          'appended_blind_demonstrator': {
              'episode_count': int(excluded_bad.size),
              'transition_count': int(np.sum(lengths[excluded_bad] - 1)),
              'original_episode_id_ranges': _ranges(excluded_bad),
              'purpose': ('separate causal-support source; retained in the '
                          'full dataset but excluded from nominal imitation')},
      },
      'full_dataset_is_unchanged': True,
  }
  return selected, manifest


def save_expert_population_manifest(path, manifest):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  temporary = path + '.tmp'
  with open(temporary, 'w') as output:
    json.dump(manifest, output, indent=2, sort_keys=True)
    output.write('\n')
  os.replace(temporary, path)
