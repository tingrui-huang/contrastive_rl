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
})
LEARNER_KEYS = frozenset({'obs', 'act'})
META_KEYS = frozenset({'meta'})
STRUCTURAL_KEYS = frozenset({'lengths'})   # allowed non-learner bookkeeping.


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
    for e in range(n_eps):
      buffer.add_episode(obs[e], act[e],
                         length=None if lengths is None else int(lengths[e]))
  buffer.freeze()
  return buffer, fp


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
