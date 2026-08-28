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
# G10: anchor-cut contract (only meaningful when scheme C is active)
# --------------------------------------------------------------------------- #
def check_anchor_cut(buffer, batch_size=4096, n_batches=8):
  """Both halves of scheme C must hold at once:

    (a) every drawn anchor i is inside its episode's [0, cut)  -- the cut bites;
    (b) goals ARE still drawn from rows >= cut                 -- the future
        window was NOT truncated (this is what separates scheme C from the
        'lengths' path, and it is the half a silent regression would break).
  """
  if not getattr(buffer, 'use_anchor_cut', False):
    return True, {'active': False}
  cuts = buffer.anchor_cuts
  bad_anchor = total = past_cut_goal = 0
  for _ in range(n_batches):
    traj, i, j = buffer.sampled_indices(batch_size)
    Ct = cuts[traj]
    bad_anchor += int(np.sum(i >= Ct))
    past_cut_goal += int(np.sum(j >= Ct))
    total += len(traj)
  ok = (bad_anchor == 0 and past_cut_goal > 0)
  return ok, {'active': True, 'samples': total,
              'anchor_past_cut_violations': bad_anchor,
              'goals_drawn_at_or_past_cut': past_cut_goal,
              'goals_past_cut_fraction': past_cut_goal / max(total, 1)}


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


def compute_anchor_cuts(obs, obs_dim, radius, obs_to_goal_fn=None):
  """Per-episode anchor cut row for ``anchor_cut_mode='arrival'`` (scheme C).

  cut[e] = min(first arrival row, start of the frozen terminal run), clipped
  into [1, L-1]. Anchors are then drawn from [0, cut) only; the future-goal
  window is untouched (see TrajectoryBuffer.set_anchor_cuts).

    arrival -- first t with ||obs_to_goal(state_t) - goal_half_t|| < radius.
               Requires the goal half of the stored observation to live in the
               same space as obs_to_goal(state), which is the contract for
               every env in crl.envs.
    frozen  -- first index of the terminal run over which the STATE never
               changes again (an absorbing/dead state, or a perfectly still
               agent). L when the state keeps moving to the end.

  Derived from the learner's own observation only -- no audit field is read,
  so this cannot leak the confounder.
  """
  obs = np.asarray(obs)
  n_eps, L = obs.shape[0], obs.shape[1]
  state = obs[:, :, :obs_dim].astype(np.float64)
  goal_half = obs[:, :, obs_dim:].astype(np.float64)
  if obs_to_goal_fn is None:
    gs = state
  else:  # obs_to_goal slices the LAST axis; apply it on a flattened view.
    gs = obs_to_goal_fn(state.reshape(-1, state.shape[-1]))
    gs = gs.reshape(n_eps, L, -1)
  if gs.shape[-1] != goal_half.shape[-1]:
    raise ValueError(
        f'anchor_cut_mode=arrival needs the goal half ({goal_half.shape[-1]}d) '
        f'and obs_to_goal(state) ({gs.shape[-1]}d) in the same space')
  hit = np.linalg.norm(gs - goal_half, axis=2) < float(radius)   # [N, L]
  arrival = np.where(hit.any(axis=1), hit.argmax(axis=1), L)

  # Start of the terminal constant run of the STATE.
  #   moved[e, t] is True iff state[e, t+1] != state[e, t]   (t = 0 .. L-2)
  # so the run begins one row after the LAST True. argmax on the reversed row
  # gives the first True from the end; (L-1) - that index is already last+1.
  moved = np.any(state[:, 1:] != state[:, :-1], axis=2)      # [N, L-1]
  any_move = moved.any(axis=1)
  run_start = (L - 1) - moved[:, ::-1].argmax(axis=1)        # = last True + 1
  frozen = np.where(any_move, run_start, 0)                  # never moved -> 0
  frozen = np.where(moved[:, -1], L, frozen)                 # no tail at all

  cut = np.clip(np.minimum(arrival, frozen), 1, L - 1).astype(np.int64)
  stats = {
      'n_episodes': int(n_eps), 'ep_len_obs': int(L),
      'radius': float(radius),
      'n_arrived': int((arrival < L).sum()),
      'n_frozen_tail': int((frozen < L).sum()),
      'cut_min': int(cut.min()), 'cut_max': int(cut.max()),
      'cut_mean': float(cut.mean()),
      'anchor_rows_kept': int(cut.sum()),
      'anchor_rows_total': int(n_eps * (L - 1)),
      'anchor_row_fraction': float(cut.sum() / (n_eps * (L - 1))),
  }
  return cut, stats


def compute_balanced_buckets(obs, act, obs_dim, cell_size, n_sectors, cuts=None,
                             zero_action_eps=0.1):
  """Bucket every eligible anchor by (discretised state cell, action sector).

  Returns (traj_idx, row_idx, bucket_id, stats) for TrajectoryBuffer
  .set_balanced_buckets. Continuous states/actions have no natural buckets --
  raw (s, a) pairs are almost all unique, so balancing over them would be a
  no-op -- therefore the discretisation is an explicit modelling choice and is
  recorded in stats.

  Two details that are not cosmetic:

  * Sector bins are ROTATED half a width so the cardinal directions land at bin
    CENTRES. With edges on the cardinals, the dominant behaviour modes (which
    here are essentially axis-aligned moves) get split across two bins. The
    same correction was needed by the earlier propensity work.
  * Near-zero actions ("wait") get their own bucket rather than being assigned
    an arbitrary angle -- arctan2(0, 0) is 0, which would silently pile every
    wait into the +x sector.

  cuts (optional): per-episode anchor cut, so buckets are built over exactly
  the rows the anchor-cut sampler would consider.
  """
  obs = np.asarray(obs)
  act = np.asarray(act)
  n_eps, L = obs.shape[0], obs.shape[1]
  n_rows = L - 1                                  # anchors live in [0, L-2]

  if cuts is None:
    elig = np.ones((n_eps, n_rows), bool)
  else:
    cuts = np.asarray(cuts, np.int64)
    elig = np.arange(n_rows)[None, :] < cuts[:, None]

  tj, rw = np.nonzero(elig)
  s = obs[tj, rw, :obs_dim].astype(np.float64)
  a = act[tj, rw].astype(np.float64)

  cell = np.floor(s / float(cell_size)).astype(np.int64)
  mag = np.linalg.norm(a, axis=1)
  ang = np.arctan2(a[:, 1], a[:, 0])
  width = 2.0 * np.pi / int(n_sectors)
  sector = np.floor((ang + width / 2.0) / width).astype(np.int64) % int(n_sectors)
  sector = np.where(mag < zero_action_eps, int(n_sectors), sector)  # wait bucket

  key = np.concatenate([cell, sector[:, None]], axis=1)
  _, bucket = np.unique(key, axis=0, return_inverse=True)
  bucket = bucket.astype(np.int64).ravel()
  counts = np.bincount(bucket)
  stats = {
      'cell_size': float(cell_size), 'n_sectors': int(n_sectors),
      'zero_action_eps': float(zero_action_eps),
      'n_anchor_rows': int(len(bucket)), 'n_buckets': int(len(counts)),
      'bucket_min': int(counts.min()), 'bucket_max': int(counts.max()),
      'bucket_median': int(np.median(counts)),
      'n_wait_bucket_rows': int((sector == int(n_sectors)).sum()),
      'sector_bins_rotated_half_width': True,
  }
  return tj.astype(np.int64), rw.astype(np.int64), bucket, stats


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
    # Scheme C: restrict ANCHORS to the pre-parking rows. Applied before
    # freeze(); the future-goal window is untouched, so every stored row stays
    # samplable as a relabeled goal.
    mode = getattr(config, 'anchor_cut_mode', '') or ''
    if mode:
      if mode != 'arrival':
        raise ValueError(f'unknown anchor_cut_mode {mode!r} (expected '
                         "'' or 'arrival')")
      from crl.replay import obs_to_goal as _o2g
      cuts, cut_stats = compute_anchor_cuts(
          obs, config.obs_dim, config.anchor_cut_radius,
          obs_to_goal_fn=lambda s: _o2g(s, config.start_index,
                                        config.end_index, config.goal_indices))
      buffer.set_anchor_cuts(cuts)
      fp['anchor_cut'] = cut_stats
    else:
      cuts = None
    # Balanced (s, a) anchor sampling -- applied on top of the anchor cut, so
    # the buckets cover exactly the rows the cut leaves eligible.
    if getattr(config, 'balanced_sampling', False):
      tj, rw, bk, bal_stats = compute_balanced_buckets(
          obs, act, config.obs_dim, config.balanced_cell_size,
          config.balanced_action_sectors, cuts=cuts)
      _cap = getattr(config, 'balanced_cap', 0) or None
      buffer.set_balanced_buckets(tj, rw, bk, cap=_cap)
      bal_stats['cap'] = _cap
      fp['balanced'] = bal_stats
      print(f'BALANCED (s,a) SAMPLING: cell {bal_stats["cell_size"]}, '
            f'{bal_stats["n_sectors"]} rotated sectors (+wait) -> '
            f'{bal_stats["n_buckets"]} buckets over '
            f'{bal_stats["n_anchor_rows"]:,} rows; sizes min '
            f'{bal_stats["bucket_min"]} / median '
            f'{bal_stats["bucket_median"]} / max {bal_stats["bucket_max"]}; '
            f'cap {_cap}')
      print(f'ANCHOR CUT (scheme C, mode={mode!r}, radius='
            f'{config.anchor_cut_radius}): keeping '
            f'{cut_stats["anchor_rows_kept"]:,}/'
            f'{cut_stats["anchor_rows_total"]:,} anchor rows '
            f'({cut_stats["anchor_row_fraction"]:.1%}); cut mean '
            f'{cut_stats["cut_mean"]:.1f}, arrived {cut_stats["n_arrived"]}, '
            f'frozen-tail {cut_stats["n_frozen_tail"]}; future window UNCHANGED')
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

  ok10, s10 = check_anchor_cut(buffer)
  report['stats']['anchor_cut'] = s10
  if s10.get('active'):                 # only a gate when scheme C is on.
    gates['G10_ANCHOR_CUT'] = ok10

  all_pass = all(gates.values())
  report['gates'] = gates
  report['verdict'] = 'PASS' if all_pass else 'FAIL'
  return all_pass, gates, report
