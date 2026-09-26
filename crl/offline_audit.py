"""Strict offline-only correctness audit for the contrastive RL pipeline.

Run BEFORE any offline (maze) experiment. It proves the run trains on a fixed,
immutable, learner-clean dataset with environment collection structurally
impossible. Every gate is a hard pass/fail; a single failure aborts training.

Dataset .npz contract
---------------------
  obs   [N, L, obs_dim+goal_dim]   learner observation (state|goal), float32/uint8
  act   [N, L, action_dim]         learner action, float32; act[:, -1] dummy
  meta  json string                env metadata (dims, indices, provenance)
  lengths [N] (optional)           per-episode VALID obs count (<= L)
  audit_* / <known audit key>      AUDIT-ONLY tensors (confounder U, swamp bits,
                                   route labels, ...) -- NEVER fed to the learner

Gates
-----
  G1 FINGERPRINT      sha256 + episode/transition counts + shapes recorded once
  G2 KEY_SEPARATION   learner keys are exactly {obs, act}; audit fields isolated
  G3 SHAPES_DIMS      obs/act shapes match the env dims exactly (no leaked cols)
  G4 DTYPES_FINITE    act float32; obs uint8/float32; no NaN/Inf
  G5 EP_LENGTHS       every episode has >=1 transition and length <= L
  G6 NO_AUDIT_LEAK    obs width == obs_dim+goal_dim; audit arrays are separate
  G7 RELABEL_BOUNDS   sampled (i, j) stay in-episode and within valid length
  G8 FROZEN_BUFFER    add_episode() raises after freeze(); checksum stable
  G9 RESUME_HASH      resume requires the identical dataset sha256
"""
import hashlib
import json
import os

import numpy as np

# Keys that carry AUDIT-ONLY information and must never enter the learner.
AUDIT_KEY_PREFIXES = ('audit_',)
KNOWN_AUDIT_KEYS = frozenset({
    'swamp_bits', 'route_label', 'route_labels', 'u', 'hidden_u',
    'gate', 'gate_open', 'confounder', 'wind',
    # MiniGrid-matched teacher audit fields (behavior-mode labels; never fed to
    # the learner). Additive vocabulary only -- no gate logic changes.
    'teacher_mode', 'force_safe', 'wait_count', 'entered_active_swamp',
})
LEARNER_KEYS = frozenset({'obs', 'act'})
META_KEYS = frozenset({'meta'})
# Allowed non-learner bookkeeping: per-episode valid lengths, and the eval
# env's empirical goal table (read ONLY by envs.make_env for the offline
# antmaze eval env -- never fed to the learner).
STRUCTURAL_KEYS = frozenset({'lengths', 'eval_goals'})


def sha256_file(path, chunk=1 << 20):
  """SHA-256 of the raw dataset file bytes (deterministic identity)."""
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def classify_keys(keys):
  learner, audit, meta, structural, other = [], [], [], [], []
  for k in keys:
    if k in LEARNER_KEYS:
      learner.append(k)
    elif k in META_KEYS:
      meta.append(k)
    elif k in STRUCTURAL_KEYS:
      structural.append(k)
    elif k in KNOWN_AUDIT_KEYS or any(k.startswith(p) for p in AUDIT_KEY_PREFIXES):
      audit.append(k)
    else:
      other.append(k)
  return dict(learner=sorted(learner), audit=sorted(audit), meta=sorted(meta),
              structural=sorted(structural), other=sorted(other))


def fingerprint(path):
  """Load the dataset ONCE and record its immutable identity + structure."""
  sha = sha256_file(path)
  with np.load(path, allow_pickle=False) as d:
    keys = list(d.keys())
    cls = classify_keys(keys)
    obs = d['obs']
    act = d['act']
    n_eps, L = int(obs.shape[0]), int(obs.shape[1])
    if 'lengths' in d:
      lengths = np.asarray(d['lengths']).astype(np.int64)
    else:
      lengths = np.full(n_eps, L, dtype=np.int64)
    n_trans = int(np.sum(lengths - 1))
    meta = {}
    if 'meta' in d:
      try:
        meta = json.loads(str(d['meta']))
      except Exception:  # pylint: disable=broad-except
        meta = {}
  return {
      'path': os.path.abspath(path),
      'sha256': sha,
      'n_episodes': n_eps,
      'n_transitions': n_trans,
      'obs_shape': list(obs.shape),
      'act_shape': list(act.shape),
      'obs_dtype': str(obs.dtype),
      'act_dtype': str(act.dtype),
      'ep_len_obs': L,
      'ep_lengths_min': int(lengths.min()),
      'ep_lengths_max': int(lengths.max()),
      'keys': cls,
      'meta': meta,
  }


# --------------------------------------------------------------------------- #
# Static gates (dataset-only; no learner needed)
# --------------------------------------------------------------------------- #
def static_gates(path, obs_dim, goal_dim, action_dim, ep_len_obs):
  """Run G1-G6 on the dataset file. Returns (gates: dict, fp: dict)."""
  fp = fingerprint(path)
  gates = {}

  # G1 FINGERPRINT: identity recorded, counts self-consistent.
  gates['G1_FINGERPRINT'] = (
      len(fp['sha256']) == 64 and fp['n_episodes'] > 0
      and fp['n_transitions'] > 0)

  # G2 KEY_SEPARATION: learner tensors are EXACTLY {obs, act}; nothing unknown
  # sits in the learner namespace.
  gates['G2_KEY_SEPARATION'] = (
      fp['keys']['learner'] == ['act', 'obs'] and fp['keys']['other'] == [])

  # G3 SHAPES_DIMS: obs/act shapes match the env dims exactly.
  full = obs_dim + goal_dim
  gates['G3_SHAPES_DIMS'] = (
      fp['obs_shape'][1:] == [ep_len_obs, full]
      and fp['act_shape'][1:] == [ep_len_obs, action_dim]
      and fp['obs_shape'][0] == fp['act_shape'][0])

  # G4 DTYPES_FINITE: act float32; obs uint8 or float32; finite (float only).
  with np.load(path, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    dtype_ok = (act.dtype == np.float32
                and obs.dtype in (np.float32, np.uint8))
    finite_ok = bool(np.isfinite(act).all()) and (
        obs.dtype == np.uint8 or bool(np.isfinite(obs).all()))
  gates['G4_DTYPES_FINITE'] = dtype_ok and finite_ok

  # G5 EP_LENGTHS: every episode has >=1 transition and length <= L.
  with np.load(path, allow_pickle=False) as d:
    if 'lengths' in d:
      lengths = np.asarray(d['lengths']).astype(np.int64)
    else:
      lengths = np.full(fp['n_episodes'], ep_len_obs, dtype=np.int64)
  gates['G5_EP_LENGTHS'] = bool(
      (lengths >= 2).all() and (lengths <= ep_len_obs).all())

  # G6 NO_AUDIT_LEAK: obs width is exactly state|goal (no confounder columns
  # concatenated in), and any audit fields are SEPARATE arrays with per-episode
  # leading dim (so they were never merged into obs/act).
  audit_ok = True
  with np.load(path, allow_pickle=False) as d:
    for k in fp['keys']['audit']:
      arr = np.asarray(d[k])
      if arr.ndim >= 1 and arr.shape[0] != fp['n_episodes']:
        audit_ok = False
  gates['G6_NO_AUDIT_LEAK'] = (fp['obs_shape'][2] == full) and audit_ok

  return gates, fp


# --------------------------------------------------------------------------- #
# G7: relabel-boundary test (needs a loaded buffer)
# --------------------------------------------------------------------------- #
def check_relabel_boundaries(buffer, n_batches=64, batch_size=256):
  """Draw many relabel index sets and assert every (i, j) pair stays inside a
  single episode and within that episode's valid length. Returns (ok, stats)."""
  lengths = buffer.lengths                       # [num_eps]
  bad_future = bad_len_i = bad_len_j = 0
  total = 0
  for _ in range(n_batches):
    traj, i, j = buffer.sampled_indices(batch_size)
    total += len(traj)
    Lt = lengths[traj]                           # valid length of each row.
    bad_future += int(np.sum(j <= i))            # goal must be strictly future.
    bad_len_i += int(np.sum(i >= Lt - 1))        # anchor within [0, len-2].
    bad_len_j += int(np.sum(j >= Lt))            # goal within [0, len-1].
  ok = (bad_future == 0 and bad_len_i == 0 and bad_len_j == 0)
  return ok, {'samples': total, 'future_violations': bad_future,
              'anchor_len_violations': bad_len_i,
              'goal_len_violations': bad_len_j}


# --------------------------------------------------------------------------- #
# G8: frozen-buffer test
# --------------------------------------------------------------------------- #
def check_frozen_buffer(buffer):
  """After freeze(), add_episode must raise and the checksum must be stable."""
  before = buffer.content_sha256()
  raised = False
  try:
    L = buffer._L                                # noqa: SLF001 (test-only)
    D = buffer._obs.shape[2]                      # noqa: SLF001
    A = buffer._act.shape[2]                      # noqa: SLF001
    buffer.add_episode(np.zeros((L, D), np.float32), np.zeros((L, A), np.float32))
  except RuntimeError:
    raised = True
  after = buffer.content_sha256()
  ok = raised and (before == after)
  return ok, {'add_episode_raised': raised, 'checksum_stable': before == after,
              'checksum': before[:16]}


# --------------------------------------------------------------------------- #
# G9: resume dataset-hash guard
# --------------------------------------------------------------------------- #
def _hash_sidecar(ckpt_dir):
  return os.path.join(ckpt_dir, 'offline_dataset.sha256')


def record_dataset_hash(ckpt_dir, sha256, meta=None):
  """Write the dataset hash sidecar at the start of a fresh offline run."""
  if not ckpt_dir:
    return
  os.makedirs(ckpt_dir, exist_ok=True)
  with open(_hash_sidecar(ckpt_dir), 'w') as f:
    json.dump({'sha256': sha256, 'meta': meta or {}}, f, indent=2)


def require_same_dataset_hash(ckpt_dir, sha256):
  """On resume, require the identical dataset hash. Returns (ok, recorded)."""
  side = _hash_sidecar(ckpt_dir)
  if not os.path.exists(side):
    return True, None                            # nothing to compare to yet.
  with open(side) as f:
    recorded = json.load(f).get('sha256')
  return (recorded == sha256), recorded


def build_offline_buffer(path, config):
  """Load the fixed dataset into a TrajectoryBuffer sized EXACTLY to it, freeze
  it, and return (buffer, fingerprint). No env, no growth room."""
  from crl.replay import TrajectoryBuffer
  fp = fingerprint(path)
  n_eps, L = fp['n_episodes'], fp['ep_len_obs']
  buffer = TrajectoryBuffer(
      capacity_steps=n_eps * L, ep_len_obs=L,
      full_obs_dim=config.obs_dim + config.goal_dim,
      action_dim=config.action_dim, obs_dim=config.obs_dim,
      start_index=config.start_index, end_index=config.end_index,
      discount=config.discount, seed=config.seed,
      goal_indices=config.goal_indices,
      obs_dtype=np.uint8 if config.use_image_obs else np.float32)
  with np.load(path, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths = (np.asarray(d['lengths']).astype(np.int64)
               if 'lengths' in d else None)
    task_goals, task_success = _task_goal_labels(d, obs, lengths, config)
    for e in range(n_eps):
      buffer.add_episode(
          obs[e], act[e],
          length=None if lengths is None else int(lengths[e]),
          task_goal=None if task_goals is None else task_goals[e],
          success=None if task_success is None else task_success[e])
  buffer.freeze()
  return buffer, fp


def _task_goal_labels(d, obs, lengths, config):
  """Per-episode (task goal, reached-it) labels derived from the dataset.

  The label is read off the data, not off a field the collector had to think
  to write: an episode's task goal is its ``eval_goals`` row, and it counts as
  successful when its LAST valid state lands within
  ``config.task_goal_success_dist`` of that goal in goal coordinates -- the
  same test the environment scores with. Returns ``(None, None)`` for a
  dataset with no ``eval_goals``, which leaves the buffer unlabeled and the
  task-goal term unusable (crl/train.py refuses that combination rather than
  training on silently-zero labels).
  """
  if 'eval_goals' not in d:
    return None, None
  from crl.replay import obs_to_goal
  goals = np.asarray(d['eval_goals'], np.float32)
  n_eps = obs.shape[0]
  last = (obs[np.arange(n_eps), lengths - 1, :config.obs_dim]
          if lengths is not None else obs[:, -1, :config.obs_dim])
  last_goal = obs_to_goal(np.asarray(last, np.float32), config.start_index,
                          config.end_index, config.goal_indices)
  if goals.shape != last_goal.shape:
    raise ValueError(
        f'eval_goals {goals.shape} does not match the goal contract '
        f'{last_goal.shape}; refusing to guess the task-goal labels.')
  dist = np.linalg.norm(last_goal - goals, axis=1)
  reached = (dist <= float(config.task_goal_success_dist)).astype(np.float32)
  return goals, reached


def run_static_audit(path, config, buffer=None):
  """Run every dataset-only + buffer gate that does not require a smoke run.
  Returns (all_pass, gates: dict[str,bool], report: dict)."""
  gates, fp = static_gates(
      path, config.obs_dim, config.goal_dim, config.action_dim,
      config.max_episode_steps + 1)
  report = {'fingerprint': fp, 'stats': {}}

  own_buffer = buffer is None
  if own_buffer:
    buffer, _ = build_offline_buffer(path, config)

  ok7, s7 = check_relabel_boundaries(buffer)
  gates['G7_RELABEL_BOUNDS'] = ok7
  report['stats']['relabel'] = s7

  ok8, s8 = check_frozen_buffer(buffer)
  gates['G8_FROZEN_BUFFER'] = ok8
  report['stats']['frozen'] = s8

  all_pass = all(gates.values())
  report['gates'] = gates
  report['verdict'] = 'PASS' if all_pass else 'FAIL'
  return all_pass, gates, report


# --------------------------------------------------------------------------- #
# Benchmark manifest: the run's environment/timing/algorithm contract
# --------------------------------------------------------------------------- #
# `offline_dataset.sha256` pins WHICH data a checkpoint was trained on. It says
# nothing about the environment the eval-during-training env was built with, the
# learning math, or -- for crl/ETT_train.py -- whether the causal transition
# model's generated rollouts were mixed into the minibatches. An authoritative
# evaluation needs all three, because the eval process rebuilds the environment
# from its own CLI flags and has no other way to know they match training.
#
# scripts/train_rockfall_clock_v6_baseline.py has written this file for its own
# runs from the start. The recorder below is the launcher-agnostic version, so
# that runs driven by configs/*.yml through crl/ETT_train.py carry the same
# record and become evaluable by the same authoritative script. It reads the
# live Config and the dataset fingerprint only: nothing here is inferred or
# defaulted from a benchmark module.
MANIFEST_NAME = 'benchmark_config.json'

# Values that may never change across a resume: the science, not the budget.
_MANIFEST_INVARIANTS = (
    'environment_version', 'env_name', 'dataset_sha256', 'learner_contract',
    'horizon', 'p_active_1', 'p_active_2', 't0_range_1', 't0_range_2',
    'seed', 'dataset_collection_seed', 'teacher_detour_prob', 'algorithm',
    'algorithm_contract', 'failure_bank', 'ett')


def _manifest_path(ckpt_dir):
  return os.path.join(ckpt_dir, MANIFEST_NAME)


def algorithm_contract(config):
  """The learning-math fields an evaluator must see unchanged.

  Mirrors ALGORITHM_CONTRACT in scripts/eval_rockfall_clock_v6_baseline.py.
  `recipe` names the contrastive recipe only; an ETT run's additions are
  recorded separately in the manifest's `ett` block, because the stage adds
  data rather than changing any term of the contrastive loss.
  """
  return {
      'recipe': 'vanilla_crl_v5',
      'binary_nce': not config.use_td and not config.use_cpc,
      'use_td': bool(config.use_td),
      'use_cpc': bool(config.use_cpc),
      'use_gcbc': bool(config.use_gcbc),
      'add_mc_to_td': bool(config.add_mc_to_td),
      'twin_q': bool(config.twin_q),
      'bc_coef': float(config.bc_coef),
      'random_goals': float(config.random_goals),
      'batch_size': int(config.batch_size),
      'repr_dim': config.repr_dim,
      'repr_norm': bool(config.repr_norm),
      'repr_norm_temp': bool(config.repr_norm_temp),
      'hidden_layer_sizes': [int(n) for n in config.hidden_layer_sizes],
      'use_image_obs': bool(config.use_image_obs),
      'use_layer_norm': bool(config.use_layer_norm),
      'discount': float(config.discount),
      'tau': float(config.tau),
      'actor_learning_rate': float(config.actor_learning_rate),
      'learning_rate': float(config.learning_rate),
      'num_sgd_steps_per_step': int(config.num_sgd_steps_per_step),
      'updates_per_step': int(config.updates_per_step),
  }


def ett_contract(config):
  """The stage-0 block: which arm this run is, and the knobs that define it.

  `augmentation_enabled` is the arm discriminator -- the one thing that
  separates the standard offline baseline from the augmented run, and the
  thing no model-fit metric records. Both halves of the stage are gated on
  `dyn_enable`, so a run with the master switch off is the standard arm
  whatever the finer values say.
  """
  enabled = bool(getattr(config, 'dyn_enable', True))
  fit = enabled and int(getattr(config, 'dyn_train_steps', 0)) > 0
  frac = float(getattr(config, 'dyn_augment_frac', 0.0))
  # crl/ETT_train_v2.py's arm: generated trajectories as critic-only negatives.
  gen = float(getattr(config, 'gen_neg_coef', 0.0) or 0.0)
  if enabled and gen > 0.0:
    arm = 'ett_gen_negative'
  elif enabled and frac > 0.0:
    arm = 'ett_augmented'
  else:
    arm = 'standard'
  return {
      'dyn_enable': enabled,
      'model_fitted': fit,
      'augmentation_enabled': bool(enabled and (frac > 0.0 or gen > 0.0)),
      'arm': arm,
      'gen_neg_coef': gen,
      'dyn_train_steps': int(getattr(config, 'dyn_train_steps', 0)),
      'dyn_augment_frac': frac,
      'dyn_rollout_steps': int(getattr(config, 'dyn_rollout_steps', 0)),
      'dyn_augment_refresh_every': int(
          getattr(config, 'dyn_augment_refresh_every', 0)),
      'dyn_next_action_input': bool(
          getattr(config, 'dyn_next_action_input', False)),
      'dyn_policy_head': getattr(config, 'dyn_policy_head', ''),
      'dyn_policy_components': int(
          getattr(config, 'dyn_policy_components', 0)),
      'dyn_off_diagonal': getattr(config, 'dyn_off_diagonal', 'none'),
      'dyn_off_diagonal_action': getattr(
          config, 'dyn_off_diagonal_action', ''),
      'dyn_off_diagonal_train_action': getattr(
          config, 'dyn_off_diagonal_train_action', ''),
      'dyn_off_diagonal_coef': float(
          getattr(config, 'dyn_off_diagonal_coef', 0.0)),
      'dyn_negative_dataset': getattr(config, 'dyn_negative_dataset', ''),
      'dyn_spectral_norm': bool(getattr(config, 'dyn_spectral_norm', False)),
      'dyn_spectral_norm_coef': float(
          getattr(config, 'dyn_spectral_norm_coef', 0.0)),
  }


def build_benchmark_manifest(config, fingerprint, provenance):
  """Assemble the manifest payload from the live Config and the fingerprint."""
  meta = (fingerprint.get('meta') or {}) if fingerprint else {}
  goal_dim = int(config.goal_dim)
  horizon = config.rockfall_max_steps
  if horizon is None:
    horizon = config.max_episode_steps
  return {
      'provenance': provenance,
      'environment_version': meta.get('environment_version'),
      'env_name': config.env_name,
      'dataset': config.offline_dataset,
      'dataset_sha256': fingerprint.get('sha256') if fingerprint else None,
      'learner_contract': {
          'state_dim': int(config.obs_dim),
          'goal_dim': goal_dim,
          'flat_dim': int(config.obs_dim) + goal_dim,
      },
      'horizon': int(horizon),
      'p_active_1': config.rockfall_p_active_1,
      'p_active_2': config.rockfall_p_active_2,
      't0_range_1': [config.rockfall_t0_min_1, config.rockfall_t0_max_1],
      't0_range_2': [config.rockfall_t0_min_2, config.rockfall_t0_max_2],
      'steps': int(config.max_number_of_steps),
      'seed': int(config.seed),
      # Provenance of the data, read off the dataset's own meta -- never a
      # value this process chose.
      'dataset_collection_seed': meta.get('collection_seed'),
      'teacher_detour_prob': meta.get('teacher_detour_prob'),
      'algorithm': 'vanilla_crl_v5_recipe',
      'algorithm_contract': algorithm_contract(config),
      'failure_bank': {
          'path': getattr(config, 'fail_bank_path', ''),
          'alpha': float(getattr(config, 'fail_neg_alpha', 0.0)),
          'enabled': bool(getattr(config, 'fail_bank_path', '')
                          and float(getattr(config, 'fail_neg_alpha', 0.0))),
      },
      'ett': ett_contract(config),
  }


def record_benchmark_manifest(ckpt_dir, config, fingerprint):
  """Write (or, on resume, re-check) the run's benchmark manifest.

  On a fresh run the file is written from the launch configuration. On resume
  the recorded contract must be unchanged -- only the step budget may grow --
  which is the same rule the resume dataset-hash gate applies to the data.
  Resuming a run that predates this recorder writes the manifest, but labels
  it so nobody can mistake the resuming launch's config for the original's.
  """
  if not ckpt_dir:
    return None
  os.makedirs(ckpt_dir, exist_ok=True)
  path = _manifest_path(ckpt_dir)
  resuming = bool(getattr(config, 'resume', False))
  previous = None
  if resuming and os.path.exists(path):
    with open(path) as f:
      previous = json.load(f)
  provenance = 'written_at_run_start'
  if resuming:
    provenance = (previous.get('provenance') if previous
                  else 'written_at_resume_of_unrecorded_run')
  payload = build_benchmark_manifest(config, fingerprint, provenance)
  if previous is not None:
    changed = {key: {'recorded': previous.get(key),
                     'requested': payload.get(key)}
               for key in _MANIFEST_INVARIANTS
               if previous.get(key) != payload.get(key)}
    if changed:
      raise RuntimeError(
          f'OFFLINE RESUME CONTRACT MISMATCH: {path} records a different '
          f'benchmark contract than this launch: {changed}')
    recorded_steps = previous.get('steps')
    if isinstance(recorded_steps, int) and payload['steps'] < recorded_steps:
      raise RuntimeError(
          f'OFFLINE RESUME step budget must not shrink: '
          f'recorded={recorded_steps}, requested={payload["steps"]}')
  elif resuming:
    print(f'  [offline] WARNING: resuming a run with no {MANIFEST_NAME}; '
          'writing one from THIS launch\'s config and marking it '
          f'provenance={provenance!r}. It describes the resuming launch, '
          'not the original.')
  with open(path, 'w') as f:
    json.dump(payload, f, indent=2)
  return path
