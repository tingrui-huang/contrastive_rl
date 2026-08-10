"""Stage 1: offline (state, action) dataset interface for the behavior model.

Reads the repository's EXISTING frozen offline ``.npz`` episode datasets (the
same contract ``crl/offline_audit.py`` documents and ``crl/train.py`` trains
from) and exposes the ONLY thing a behavior-action model ``mu(a | s)`` needs:

    state   [B, state_dim]   float32
    action  [B, action_dim]  float32

Nothing else is exposed. ``next_state``, ``reward``, ``goal``, the confounder,
route labels, teacher modes and every other audit field are never materialized;
they are structurally excluded at load time (see ``_load_learner_arrays``).

Dataset contract (unchanged, see crl/offline_audit.py)
------------------------------------------------------
  obs      [E, L, obs_dim + goal_dim]  learner observation, ``state | goal``
  act      [E, L, action_dim]          float32; row ``length-1`` is a zero dummy
  lengths  [E]                (optional) valid observation count per episode
  meta                        JSON provenance (obs_dim / goal_dim / action_dim)
  <audit>                     AUDIT-ONLY tensors -- never read here
  eval_goals                  EVAL-ONLY tensor  -- never read here

A transition row ``t`` of episode ``e`` is valid for ``0 <= t <= length[e]-2``;
``act[e, length[e]-1]`` is the terminal dummy, so the total number of usable
(state, action) pairs is ``sum(lengths - 1)`` -- identical to the transition
count the training buffer reports.

What "state" means
------------------
``state`` defaults to the STATE half of the stored observation,
``obs[..., :obs_dim]`` -- the same slice ``crl.replay.TrajectoryBuffer.sample``
calls ``state``, and the conditioning variable ``S`` in ``P(A = a | S = s)``.
Pass ``state_mode='obs'`` to condition on the full stored ``state | goal``
vector instead (that is what the earlier ``propensity_net/make_pairs.py`` did);
this is an explicit, logged choice, never a silent one.

Actions are NOT normalized here
-------------------------------
Every conforming dataset in this repo already stores actions inside the env's
``[-1, 1]`` action box, so no normalization convention is applied or invented at
this stage. The loader only measures and reports per-dimension ranges, and warns
if it ever sees values outside ``[-1, 1]``.

Offline safety
--------------
  * only ``obs`` / ``act`` / ``lengths`` / ``meta`` are ever read from the file;
  * any unrecognized ("other") key aborts the load, so a future dataset cannot
    smuggle a field past this interface;
  * ``eval_goals`` -- the eval env's goal table -- is refused explicitly;
  * the loaded arrays are marked read-only, and a sha256 fingerprint is taken;
  * when a ``.manifest.json`` sidecar exists its sha256 must match;
  * train/validation are split BY EPISODE with an explicit seed, so no two rows
    of the same trajectory straddle the split;
  * no env is constructed and no environment module is imported.

Run the smoke test / diagnostics:

  python -m propensity.dataset --dataset datasets/swamp_windy_teacher_s0.npz
  python -m propensity.dataset --list
"""
import argparse
import collections
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

# Read-only reuse of the repo's single source of truth for the frozen-dataset
# key vocabulary (pure-python module: importing it pulls in no JAX and no env).
from crl import offline_audit

# Keys this interface is allowed to read. Everything else is refused.
_LEARNER_KEYS = frozenset({'obs', 'act'})
_READABLE_KEYS = frozenset({'obs', 'act', 'lengths', 'meta'})
# Eval-only bookkeeping: the offline antmaze eval env's empirical goal table.
# Training data must never contain it -- refused by name, not merely ignored.
_EVAL_ONLY_KEYS = frozenset({'eval_goals'})

# The public training batch. Exactly two fields, by design.
BehaviorBatch = collections.namedtuple('BehaviorBatch', ['state', 'action'])


def sha256_file(path, chunk=1 << 20):
  """SHA-256 of the raw dataset bytes (delegates to the audit implementation)."""
  return offline_audit.sha256_file(path, chunk=chunk)


def _sha256_arrays(*arrays):
  """Content checksum over the materialized (state, action) tensors."""
  h = hashlib.sha256()
  for a in arrays:
    h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


class DatasetContractError(RuntimeError):
  """The .npz does not satisfy the offline dataset contract."""


def _load_learner_arrays(path, verify_sha256=True):
  """Open the frozen npz and materialize ONLY obs/act (+ lengths, meta).

  Returns (obs, act, lengths, meta, fingerprint). Raises DatasetContractError
  when the file carries keys this interface refuses to touch."""
  if not os.path.exists(path):
    raise DatasetContractError(f'dataset not found: {path}')

  sha = sha256_file(path)
  manifest_path = path + '.manifest.json'
  manifest_sha = None
  if os.path.exists(manifest_path):
    with open(manifest_path) as f:
      manifest_sha = json.load(f).get('sha256')
    if verify_sha256 and manifest_sha != sha:
      raise DatasetContractError(
          f'sha256 mismatch against {manifest_path}: file={sha[:16]}... '
          f'manifest={str(manifest_sha)[:16]}... -- the frozen dataset changed.')

  with np.load(path, allow_pickle=False) as d:
    keys = list(d.keys())
    cls = offline_audit.classify_keys(keys)
    if cls['other']:
      raise DatasetContractError(
          f'unrecognized keys {cls["other"]} in {path}: refusing to load a '
          'dataset whose extra fields are not classified as learner/audit/'
          'structural (see crl/offline_audit.py).')
    missing = _LEARNER_KEYS - set(keys)
    if missing:
      raise DatasetContractError(f'missing learner keys {sorted(missing)}')

    # Materialize the learner tensors only. Audit keys and eval_goals are never
    # indexed, so they are never decompressed into memory.
    obs = np.asarray(d['obs'])
    act = np.asarray(d['act'])
    lengths = (np.asarray(d['lengths']).astype(np.int64)
               if 'lengths' in keys else None)
    meta = {}
    if 'meta' in keys:
      try:
        meta = json.loads(str(d['meta']))
      except (ValueError, TypeError):
        meta = {}

  fingerprint = {
      'path': os.path.abspath(path),
      'sha256': sha,
      'manifest_sha256': manifest_sha,
      'manifest_verified': bool(manifest_sha is not None and verify_sha256),
      'keys': cls,
      'refused_eval_only_keys': sorted(_EVAL_ONLY_KEYS.intersection(keys)),
      'refused_audit_keys': cls['audit'],
      'meta': meta,
  }
  return obs, act, lengths, meta, fingerprint


class BehaviorDataset:
  """Fixed offline (state, action) pairs with a deterministic train/val split.

  The dataset is immutable: the underlying arrays are marked non-writeable and
  there is no append/collect path, so evaluation rollouts cannot be inserted.
  """

  def __init__(self, path, val_frac=0.1, seed=0, state_mode='state',
               split_level='episode', verify_sha256=True, strict_bounds=False):
    """Args:

      path: .npz episode dataset following the crl/offline_audit.py contract.
      val_frac: fraction held out for validation, in [0, 1).
      seed: split seed. The split is a pure function of (seed, val_frac,
        split_level, episode count) -- rerunning reproduces it exactly.
      state_mode: 'state' => obs[..., :obs_dim] (default, matches crl.replay);
        'obs' => the full stored state|goal observation.
      split_level: 'episode' (default) keeps every row of a trajectory on one
        side of the split; 'transition' splits i.i.d. over rows and is offered
        only for ablations -- it lets near-duplicate consecutive rows of the
        same trajectory appear on both sides.
      verify_sha256: check the .manifest.json sidecar hash when one exists.
      strict_bounds: raise (instead of warn) if actions leave [-1, 1].
    """
    if not 0.0 <= val_frac < 1.0:
      raise ValueError(f'val_frac must be in [0, 1), got {val_frac}')
    if state_mode not in ('state', 'obs'):
      raise ValueError(f"state_mode must be 'state' or 'obs', got {state_mode}")
    if split_level not in ('episode', 'transition'):
      raise ValueError(
          f"split_level must be 'episode' or 'transition', got {split_level}")

    obs, act, lengths, meta, fp = _load_learner_arrays(
        path, verify_sha256=verify_sha256)
    self._fingerprint = fp
    self._meta = meta
    self._path = path
    self._split_seed = int(seed)
    self._val_frac = float(val_frac)
    self._state_mode = state_mode
    self._split_level = split_level

    # ---- structural checks on the raw tensors ------------------------------ #
    if obs.ndim != 3 or act.ndim != 3:
      raise DatasetContractError(
          f'expected obs/act of rank 3 [E, L, D], got {obs.shape} / {act.shape}')
    if obs.shape[0] != act.shape[0] or obs.shape[1] != act.shape[1]:
      raise DatasetContractError(
          f'obs/act episode-time misalignment: {obs.shape} vs {act.shape}')
    if act.dtype != np.float32:
      raise DatasetContractError(f'act must be float32, got {act.dtype}')
    if obs.dtype == np.uint8:
      raise DatasetContractError(
          f'{path} stores uint8 image observations. Stage 1 targets state-based '
          'continuous control; an image behavior model needs an explicit '
          'encoder/scaling decision that is out of scope here.')
    if obs.dtype != np.float32:
      raise DatasetContractError(f'obs must be float32, got {obs.dtype}')

    n_eps, ep_len_obs = int(obs.shape[0]), int(obs.shape[1])
    full_obs_dim, action_dim = int(obs.shape[2]), int(act.shape[2])

    # obs_dim: from meta when present, else assume state|goal halves.
    obs_dim = int(meta.get('obs_dim', full_obs_dim // 2))
    goal_dim = int(meta.get('goal_dim', full_obs_dim - obs_dim))
    if state_mode == 'state':
      if not 0 < obs_dim <= full_obs_dim:
        raise DatasetContractError(
            f'meta obs_dim={obs_dim} inconsistent with obs width {full_obs_dim}')
      state_width = obs_dim
    else:
      state_width = full_obs_dim

    if lengths is None:
      lengths = np.full(n_eps, ep_len_obs, dtype=np.int64)
    if lengths.shape != (n_eps,):
      raise DatasetContractError(f'lengths shape {lengths.shape} != ({n_eps},)')
    if not ((lengths >= 2).all() and (lengths <= ep_len_obs).all()):
      raise DatasetContractError(
          f'episode lengths out of [2, {ep_len_obs}]: '
          f'min={int(lengths.min())} max={int(lengths.max())}')

    # ---- flatten to valid transition rows ---------------------------------- #
    # Row t of episode e is a real (s, a) pair iff 0 <= t <= lengths[e]-2; the
    # row at lengths[e]-1 holds the terminal observation and a zero dummy
    # action, and anything beyond it is padding.
    ep_idx = np.repeat(np.arange(n_eps, dtype=np.int64), lengths - 1)
    t_idx = np.concatenate(
        [np.arange(int(n) - 1, dtype=np.int64) for n in lengths])
    assert ep_idx.shape == t_idx.shape

    state = obs[ep_idx, t_idx, :state_width]
    action = act[ep_idx, t_idx, :]
    # Alignment is by construction (one fancy-index pass with shared indices);
    # assert the invariant anyway so a future refactor cannot break it silently.
    if state.shape[0] != action.shape[0]:
      raise DatasetContractError(
          f'state/action count mismatch: {state.shape[0]} vs {action.shape[0]}')

    state = np.ascontiguousarray(state, dtype=np.float32)
    action = np.ascontiguousarray(action, dtype=np.float32)
    state.flags.writeable = False
    action.flags.writeable = False
    self._state = state
    self._action = action
    self._episode_of_row = ep_idx
    self._episode_of_row.flags.writeable = False

    self._n_episodes = n_eps
    self._ep_len_obs = ep_len_obs
    self._obs_dim = obs_dim
    self._goal_dim = goal_dim
    self._full_obs_dim = full_obs_dim
    self._state_dim = int(state.shape[1])
    self._action_dim = action_dim
    self._lengths = lengths
    self._content_sha256 = _sha256_arrays(state, action)

    # ---- finiteness -------------------------------------------------------- #
    self._n_nonfinite_state = int((~np.isfinite(state)).sum())
    self._n_nonfinite_action = int((~np.isfinite(action)).sum())
    if self._n_nonfinite_state or self._n_nonfinite_action:
      raise DatasetContractError(
          f'non-finite values: {self._n_nonfinite_state} in state, '
          f'{self._n_nonfinite_action} in action')

    # ---- action bounds (reported, never applied) --------------------------- #
    self._out_of_box = int(((action < -1.0) | (action > 1.0)).sum())
    if self._out_of_box:
      msg = (f'{self._out_of_box} action entries lie outside the [-1, 1] box '
             f'(min={float(action.min()):.4f}, max={float(action.max()):.4f}). '
             'No normalization is applied at Stage 1.')
      if strict_bounds:
        raise DatasetContractError(msg)
      print(f'[propensity.dataset] WARNING: {msg}')

    # ---- deterministic split ----------------------------------------------- #
    self._train_idx, self._val_idx = self._make_split()

  # ------------------------------------------------------------------------- #
  # split
  # ------------------------------------------------------------------------- #
  def _make_split(self):
    rng = np.random.default_rng(self._split_seed)
    if self._split_level == 'episode':
      perm = rng.permutation(self._n_episodes)
      n_val_eps = int(round(self._val_frac * self._n_episodes))
      val_eps = np.zeros(self._n_episodes, dtype=bool)
      val_eps[perm[:n_val_eps]] = True
      is_val = val_eps[self._episode_of_row]
    else:
      n_rows = self._state.shape[0]
      perm = rng.permutation(n_rows)
      n_val = int(round(self._val_frac * n_rows))
      is_val = np.zeros(n_rows, dtype=bool)
      is_val[perm[:n_val]] = True
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)
    return train_idx, val_idx

  def _indices(self, split):
    if split == 'train':
      return self._train_idx
    if split in ('val', 'validation'):
      return self._val_idx
    if split == 'all':
      return np.arange(self._state.shape[0], dtype=np.int64)
    raise ValueError(f"split must be 'train' / 'val' / 'all', got {split}")

  # ------------------------------------------------------------------------- #
  # public accessors
  # ------------------------------------------------------------------------- #
  @property
  def state_dim(self):
    return self._state_dim

  @property
  def context_dim(self):
    """Stage-2 name for the same width.

    The behavior flow model conditions on the DECISION CONTEXT
    ``c = concat(s, g_cmd)``, which is exactly what ``state_mode='obs'``
    returns (the stored ``state | goal`` observation, with ``g_cmd`` the
    pre-action commanded goal fixed at ``env.reset()``). Under
    ``state_mode='state'`` this is just ``state_dim``. Provided so Stage 2 can
    say ``context_dim`` instead of calling a 58-wide vector a 'state'."""
    return self._state_dim

  @property
  def context_mode(self):
    """'context' when the loaded vector is (s, g_cmd); 'state_only' otherwise."""
    return 'context' if self._state_mode == 'obs' else 'state_only'

  @property
  def action_dim(self):
    return self._action_dim

  @property
  def n_transitions(self):
    return int(self._state.shape[0])

  @property
  def n_train(self):
    return int(self._train_idx.size)

  @property
  def n_val(self):
    return int(self._val_idx.size)

  @property
  def n_episodes(self):
    return self._n_episodes

  @property
  def split_seed(self):
    return self._split_seed

  @property
  def fingerprint(self):
    return dict(self._fingerprint)

  @property
  def content_sha256(self):
    """Checksum of the materialized (state, action) tensors."""
    return self._content_sha256

  def __len__(self):
    return self.n_transitions

  def arrays(self, split='train'):
    """Read-only (state, action) views for a split. Same row order, aligned."""
    idx = self._indices(split)
    return BehaviorBatch(state=self._state[idx], action=self._action[idx])

  # ------------------------------------------------------------------------- #
  # sampling
  # ------------------------------------------------------------------------- #
  def sample_batch(self, batch_size, split='train', rng=None):
    """Uniformly sample ``batch_size`` (state, action) pairs with replacement."""
    idx_pool = self._indices(split)
    if idx_pool.size == 0:
      raise ValueError(f"split '{split}' is empty")
    rng = np.random.default_rng() if rng is None else rng
    rows = idx_pool[rng.integers(0, idx_pool.size, size=int(batch_size))]
    return BehaviorBatch(state=self._state[rows], action=self._action[rows])

  def iterate(self, batch_size, split='train', shuffle=True, seed=None,
              drop_last=True):
    """Yield BehaviorBatch minibatches covering the split exactly once."""
    idx_pool = self._indices(split)
    order = np.array(idx_pool, copy=True)
    if shuffle:
      rng = np.random.default_rng(self._split_seed if seed is None else seed)
      rng.shuffle(order)
    n = order.size
    stop = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, stop, batch_size):
      rows = order[start:start + batch_size]
      yield BehaviorBatch(state=self._state[rows], action=self._action[rows])

  # ------------------------------------------------------------------------- #
  # diagnostics
  # ------------------------------------------------------------------------- #
  def action_stats(self, split='all'):
    """Per-dimension min / max / mean / std of the raw (unnormalized) actions."""
    a = self._action[self._indices(split)]
    return {
        'min': np.min(a, axis=0),
        'max': np.max(a, axis=0),
        'mean': np.mean(a, axis=0),
        'std': np.std(a, axis=0),
    }

  def state_stats(self, split='all'):
    a = self._state[self._indices(split)]
    return {
        'min': np.min(a, axis=0),
        'max': np.max(a, axis=0),
        'mean': np.mean(a, axis=0),
        'std': np.std(a, axis=0),
    }

  def check(self):
    """Run every Stage-1 invariant. Returns (all_pass, gates, details)."""
    gates, details = {}, {}

    # C1 alignment: identical sample counts on both tensors.
    gates['C1_STATE_ACTION_ALIGNED'] = bool(
        self._state.shape[0] == self._action.shape[0]
        and self._state.shape[0] == self._episode_of_row.shape[0])
    details['n_state'] = int(self._state.shape[0])
    details['n_action'] = int(self._action.shape[0])

    # C1b transition count agrees with the dataset's own episode lengths.
    expected = int(np.sum(self._lengths - 1))
    gates['C1b_TRANSITION_COUNT'] = bool(expected == self.n_transitions)
    details['expected_transitions'] = expected

    # C2 finiteness.
    gates['C2_FINITE'] = bool(self._n_nonfinite_state == 0
                              and self._n_nonfinite_action == 0)

    # C3 dtypes.
    gates['C3_DTYPES'] = bool(self._state.dtype == np.float32
                              and self._action.dtype == np.float32)

    # C4 train/val disjoint AND exhaustive at the row level.
    inter = np.intersect1d(self._train_idx, self._val_idx)
    union = self.n_train + self.n_val
    gates['C4_SPLIT_DISJOINT'] = bool(inter.size == 0
                                      and union == self.n_transitions)
    details['index_overlap'] = int(inter.size)

    # C4b trajectory-level disjointness: no episode contributes rows to both
    # sides (skipped by construction when split_level == 'transition').
    tr_eps = np.unique(self._episode_of_row[self._train_idx])
    va_eps = np.unique(self._episode_of_row[self._val_idx])
    ep_overlap = np.intersect1d(tr_eps, va_eps)
    details['episode_overlap'] = int(ep_overlap.size)
    details['train_episodes'] = int(tr_eps.size)
    details['val_episodes'] = int(va_eps.size)
    if self._split_level == 'episode':
      gates['C4b_EPISODE_DISJOINT'] = bool(ep_overlap.size == 0)
    else:
      gates['C4b_EPISODE_DISJOINT'] = True   # not asserted for row-level splits.

    # C5 no evaluation / audit data admitted.
    fp = self._fingerprint
    gates['C5_NO_EVAL_OR_AUDIT_DATA'] = bool(
        fp['keys']['learner'] == ['act', 'obs'] and not fp['keys']['other'])
    details['refused_eval_only_keys'] = fp['refused_eval_only_keys']
    details['refused_audit_keys'] = fp['refused_audit_keys']
    details['loaded_keys'] = sorted(_READABLE_KEYS.intersection(
        set(sum(fp['keys'].values(), []))))

    # C6 immutability: the materialized tensors reject writes.
    ro = (not self._state.flags.writeable) and (not self._action.flags.writeable)
    gates['C6_READ_ONLY'] = bool(ro)

    # C7 the public batch carries exactly (state, action).
    gates['C7_BATCH_FIELDS'] = tuple(BehaviorBatch._fields) == ('state', 'action')

    return all(gates.values()), gates, details

  def report(self):
    """Diagnostics dict (JSON-serializable) for the smoke test / CLI."""
    a = self.action_stats('all')
    s = self.state_stats('all')
    ok, gates, details = self.check()
    return {
        'dataset': self._path,
        'sha256': self._fingerprint['sha256'],
        'manifest_verified': self._fingerprint['manifest_verified'],
        'content_sha256': self._content_sha256,
        'env_name': self._meta.get('env_name'),
        'behavior_policy': self._meta.get('behavior_policy'),
        'n_episodes': self.n_episodes,
        'ep_len_obs': self._ep_len_obs,
        'n_transitions': self.n_transitions,
        'train_size': self.n_train,
        'val_size': self.n_val,
        'val_frac': self._val_frac,
        'split_seed': self.split_seed,
        'split_level': self._split_level,
        'state_mode': self._state_mode,
        'obs_dim': self._obs_dim,
        'goal_dim': self._goal_dim,
        'full_obs_dim': self._full_obs_dim,
        'state_shape': [self.n_transitions, self.state_dim],
        'action_shape': [self.n_transitions, self.action_dim],
        'state_dtype': str(self._state.dtype),
        'action_dtype': str(self._action.dtype),
        'action_min': a['min'].tolist(),
        'action_max': a['max'].tolist(),
        'action_mean': a['mean'].tolist(),
        'action_std': a['std'].tolist(),
        'action_out_of_unit_box': self._out_of_box,
        'actions_normalized_by_loader': False,
        'state_min': s['min'].tolist(),
        'state_max': s['max'].tolist(),
        'gates': gates,
        'details': details,
        'verdict': 'PASS' if ok else 'FAIL',
    }


# --------------------------------------------------------------------------- #
# CLI / smoke test
# --------------------------------------------------------------------------- #
def _conforming_datasets(roots=('datasets', 'artifacts')):
  """Enumerate .npz files that satisfy the learner-key contract."""
  found = []
  for root in roots:
    if not os.path.isdir(root):
      continue
    for dirpath, _, filenames in os.walk(root):
      for fn in filenames:
        if not fn.endswith('.npz'):
          continue
        path = os.path.join(dirpath, fn)
        try:
          with np.load(path, allow_pickle=False) as d:
            keys = list(d.keys())
            if not _LEARNER_KEYS <= set(keys):
              continue
            cls = offline_audit.classify_keys(keys)
            if cls['other']:
              continue
            obs, act = d['obs'], d['act']
            if obs.ndim != 3 or act.ndim != 3:
              continue
            found.append((path, tuple(obs.shape), tuple(act.shape),
                          str(obs.dtype)))
        except Exception:  # pylint: disable=broad-except
          continue
  return sorted(found)


def _fmt_vec(v, width=10, prec=4):
  return '[' + ' '.join(f'{float(x):{width}.{prec}f}' for x in v) + ']'


def main(argv=None):
  p = argparse.ArgumentParser(
      description='Stage 1 offline (state, action) dataset interface + smoke '
                  'test for the continuous-action propensity module.')
  p.add_argument('--dataset', default=None,
                 help='path to a frozen offline .npz (crl/offline_audit.py '
                      'contract). Required unless --list is given.')
  p.add_argument('--val-frac', type=float, default=0.1)
  p.add_argument('--seed', type=int, default=0, help='split seed')
  p.add_argument('--state-mode', choices=('state', 'obs'), default='state',
                 help="'state' = obs[..., :obs_dim] (default); 'obs' = the "
                      'full stored state|goal vector.')
  p.add_argument('--split-level', choices=('episode', 'transition'),
                 default='episode')
  p.add_argument('--batch-size', type=int, default=256)
  p.add_argument('--no-verify-sha256', action='store_true',
                 help='skip the .manifest.json hash check (not recommended).')
  p.add_argument('--strict-bounds', action='store_true',
                 help='fail instead of warn when actions leave [-1, 1].')
  p.add_argument('--list', action='store_true',
                 help='list conforming datasets under datasets/ and artifacts/ '
                      'and exit.')
  p.add_argument('--json', default=None, help='write the report to this path.')
  args = p.parse_args(argv)

  if args.list:
    rows = _conforming_datasets()
    print(f'conforming offline datasets ({len(rows)}):')
    for path, oshape, ashape, odtype in rows:
      note = ('   [image obs -- not supported at Stage 1]'
              if odtype == 'uint8' else '')
      print(f'  {path}\n      obs={oshape} {odtype}  act={ashape}{note}')
    return 0

  if not args.dataset:
    p.error('--dataset is required (or pass --list to enumerate candidates)')

  ds = BehaviorDataset(
      args.dataset, val_frac=args.val_frac, seed=args.seed,
      state_mode=args.state_mode, split_level=args.split_level,
      verify_sha256=not args.no_verify_sha256,
      strict_bounds=args.strict_bounds)
  rep = ds.report()

  print('=' * 72)
  print('propensity.dataset -- Stage 1 offline (state, action) interface')
  print('=' * 72)
  print(f'dataset            {rep["dataset"]}')
  print(f'sha256             {rep["sha256"][:32]}...  '
        f'(manifest verified: {rep["manifest_verified"]})')
  print(f'env_name           {rep["env_name"]}')
  print(f'behavior_policy    {rep["behavior_policy"]}')
  print()
  print(f'number of transitions   {rep["n_transitions"]}')
  print(f'train size              {rep["train_size"]}')
  print(f'validation size         {rep["val_size"]}   '
        f'(val_frac={rep["val_frac"]})')
  print(f'split seed              {rep["split_seed"]}   '
        f'(level={rep["split_level"]})')
  print(f'episodes                {rep["n_episodes"]}  '
        f'(train {rep["details"]["train_episodes"]} / '
        f'val {rep["details"]["val_episodes"]})')
  print()
  print(f'state mode              {rep["state_mode"]}  '
        f'(obs_dim={rep["obs_dim"]}, goal_dim={rep["goal_dim"]}, '
        f'stored obs width={rep["full_obs_dim"]})')
  print(f'state shape             {tuple(rep["state_shape"])}')
  print(f'action shape            {tuple(rep["action_shape"])}')
  print(f'state dtype             {rep["state_dtype"]}')
  print(f'action dtype            {rep["action_dtype"]}')
  print()
  print(f'action min  per dim  {_fmt_vec(rep["action_min"])}')
  print(f'action max  per dim  {_fmt_vec(rep["action_max"])}')
  print(f'action mean per dim  {_fmt_vec(rep["action_mean"])}')
  print(f'action std  per dim  {_fmt_vec(rep["action_std"])}')
  print(f'entries outside [-1, 1]: {rep["action_out_of_unit_box"]}   '
        f'(normalized by loader: {rep["actions_normalized_by_loader"]})')
  print()

  # Live batch check: the public batch must expose exactly (state, action).
  rng = np.random.default_rng(args.seed)
  b_tr = ds.sample_batch(args.batch_size, split='train', rng=rng)
  b_va = (ds.sample_batch(args.batch_size, split='val', rng=rng)
          if ds.n_val else None)
  n_iter = sum(1 for _ in ds.iterate(args.batch_size, split='train'))
  print(f'sample_batch(train)  fields={b_tr._fields}  '
        f'state={b_tr.state.shape}{b_tr.state.dtype}  '
        f'action={b_tr.action.shape}{b_tr.action.dtype}')
  if b_va is not None:
    print(f'sample_batch(val)    state={b_va.state.shape}  '
          f'action={b_va.action.shape}')
  print(f'iterate(train)       {n_iter} minibatches of {args.batch_size}')
  print()

  print('GATES')
  for g, ok in rep['gates'].items():
    print(f'   {"PASS" if ok else "FAIL":4}  {g}')
  d = rep['details']
  print(f'   index overlap (train n val)   : {d["index_overlap"]}')
  print(f'   episode overlap (train n val) : {d["episode_overlap"]}')
  print(f'   keys loaded                   : {d["loaded_keys"]}')
  print(f'   audit keys refused            : {d["refused_audit_keys"]}')
  print(f'   eval-only keys refused        : {d["refused_eval_only_keys"]}')
  print()
  print(f'VERDICT: {rep["verdict"]}')

  if args.json:
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, 'w') as f:
      json.dump(rep, f, indent=2)
    print(f'report -> {args.json}')

  return 0 if rep['verdict'] == 'PASS' else 1


if __name__ == '__main__':
  sys.exit(main())
