"""Trajectory replay buffer + goal-relabeling sampler (numpy).

Replaces Reverb's ``EpisodeAdder`` + the TF ``flatten_fn`` data pipeline in
``contrastive/builder.py``. Faithfully reproduces the two statistical properties
that make the original pipeline work for contrastive learning:

  1. Geometric future-goal relabeling. For a transition at time ``i`` in a
     trajectory, the goal is the goal-coords of a FUTURE state ``j > i`` sampled
     with probability proportional to ``discount**(j - i)`` -- exactly the
     ``probs = is_future_mask * discount**(j-i)`` categorical in flatten_fn.
     Implemented here with Gumbel-max sampling over the same log-probs.

  2. Cross-trajectory negatives. Each element of a batch is drawn from an
     independently sampled trajectory, so the off-diagonal (state_a, goal_b)
     pairs used as contrastive negatives come from different trajectories -- the
     effect the original achieved via its "transpose_shuffle".

Only the STATE part ``obs[:, :obs_dim]`` is used for relabeling (as in the
original); the env's own goal half of the stored observation is discarded.

Two OPT-IN deviations exist, both inactive by default so the fixed-length draw
stays byte-identical to the original RNG stream:

  * variable ``lengths`` (``add_episode(..., length=)``) -- shortens BOTH the
    anchor range and the future-goal window, for datasets with padded tails;
  * ``set_anchor_cuts()`` -- shortens ONLY the anchor range and leaves the
    future-goal window at full length, for fixed-length datasets whose tails
    are real (an agent parked on the goal, or frozen after a terminal event)
    and should stay samplable as goals while no longer being fitted as anchors.
"""
from typing import Optional

import numpy as np

from crl.losses import Transition


def obs_to_goal(states, start_index, end_index, goal_indices=None):
  """Slice a batch of states [., obs_dim] down to goal coords (utils port).

  ``goal_indices`` (a sequence of column indices) overrides the contiguous
  start/end slice when given -- used by the goal-representation ablation."""
  if goal_indices is not None:
    return states[:, np.asarray(goal_indices)]
  if end_index == -1:
    return states[:, start_index:]
  return states[:, start_index:end_index]


class TrajectoryBuffer:
  """Fixed-episode-length ring buffer of trajectories."""

  def __init__(self, capacity_steps, ep_len_obs, full_obs_dim, action_dim,
               obs_dim, start_index, end_index, discount, seed=0,
               goal_indices=None, obs_dtype=np.float32):
    """Args:

      capacity_steps: approx max ENV steps to retain (=> capacity in episodes).
      ep_len_obs: number of observations stored per episode (max_steps + 1).
      full_obs_dim: width of the stored (env) observation = obs_dim + goal_dim.
      action_dim: action width.
      obs_dim: width of the STATE part of the observation.
      start_index/end_index: goal-coordinate slice applied to the state.
      discount: geometric discount used for future-goal sampling.
      obs_dtype: storage dtype for observations. Image envs use uint8 (4x less
        RAM; frames are raw pixels anyway). Sampling always emits float32.
    """
    self._L = int(ep_len_obs)
    self._capacity_eps = max(1, int(capacity_steps) // self._L)
    self._obs_dim = obs_dim
    self._start_index = start_index
    self._end_index = end_index
    self._goal_indices = (None if goal_indices is None
                          else np.asarray(goal_indices, dtype=np.int64))
    self._discount = discount
    self._rng = np.random.default_rng(seed)
    self._obs_dtype = np.dtype(obs_dtype)

    self._obs = np.zeros((self._capacity_eps, self._L, full_obs_dim),
                         dtype=self._obs_dtype)
    self._act = np.zeros((self._capacity_eps, self._L, action_dim),
                         dtype=np.float32)
    # Per-episode TASK goal (goal coords) and outcome label, for the critic's
    # task-goal term. Unlike the relabeled goal these are trajectory-level and
    # anchor-independent. Left at zeros / unlabeled unless add_episode is given
    # them; `has_task_labels` is what a consumer must gate on.
    self._goal_dim = int(obs_to_goal(
        np.zeros((1, obs_dim), np.float32), start_index, end_index,
        self._goal_indices).shape[1])
    self._task_goal = np.zeros((self._capacity_eps, self._goal_dim),
                               dtype=np.float32)
    self._task_success = np.zeros(self._capacity_eps, dtype=np.float32)
    self._n_labeled = 0
    # Death-trajectory hindsight negatives (see config.death_neg_coef). Off
    # until enable_death_negatives() is called; the draw uses its OWN rng so
    # that turning the term on does not shift the ordinary relabeling stream
    # by a single number -- the coef-0 and coef>0 arms then see the identical
    # sequence of ordinary batches and the comparison is exactly paired.
    self._death_batch = 0
    self._death_pool = None
    self._death_src = self   # buffer the death pool indexes into
    self._death_rng = np.random.default_rng(int(seed) + 90_210)
    # Per-episode VALID observation count (incl. the terminal obs). All fixed
    # at L unless add_episode is given an explicit shorter length -- used so
    # future-goal relabeling never samples across a padded tail (see sample()).
    self._lengths_arr = np.full(self._capacity_eps, self._L, dtype=np.int64)
    self._use_lengths = False
    # Per-episode ANCHOR cut: anchor times i are drawn from [0, cut) only.
    # DISTINCT from _lengths_arr -- it does NOT shrink the future-goal window,
    # which stays the full L rows (see set_anchor_cuts / _draw_indices).
    # Inactive unless set_anchor_cuts() is called.
    self._anchor_cut_arr = np.full(self._capacity_eps, self._L - 1,
                                   dtype=np.int64)
    self._use_anchor_cut = False
    self._write = 0      # next episode slot to write (ring).
    self._num_eps = 0    # number of valid episodes stored.
    self._frozen = False  # offline mode locks the buffer (see freeze()).
    self._log_discount = float(np.log(discount)) if discount > 0 else -np.inf

  def freeze(self):
    """Make the buffer immutable: any later add_episode() raises. Used by the
    strict offline audit so environment collection is structurally impossible
    once the fixed dataset is loaded."""
    self._frozen = True

  @property
  def frozen(self):
    return self._frozen

  @property
  def has_task_labels(self):
    """True when every stored episode carries a task goal + outcome label."""
    return self._num_eps > 0 and self._n_labeled == self._num_eps

  @property
  def task_success(self):
    """Per-episode outcome labels for the stored episodes."""
    return self._task_success[:self._num_eps].copy()

  @property
  def death_episodes(self):
    """Indices of the stored episodes labeled as NOT having reached their
    task goal -- the failure ('death') trajectories. Empty until the buffer
    has per-episode labels (see ``has_task_labels``)."""
    if not self.has_task_labels:
      return np.zeros((0,), np.int64)
    return np.flatnonzero(self._task_success[:self._num_eps] <= 0.5).astype(
        np.int64)

  def enable_death_negatives(self, batch_size):
    """Make ``sample()`` also emit a death-trajectory hindsight batch.

    ``batch_size`` rows are drawn per call: an episode uniformly from the
    death pool, then an anchor time and a geometric future goal time by the
    SAME rule ``sample()`` uses for every other anchor. The only difference
    is downstream -- crl/losses.py scores the pair with label 0 instead of
    the 1 hindsight would assign it."""
    pool = self.death_episodes
    if batch_size <= 0:
      raise ValueError(f'death batch size must be positive, got {batch_size}')
    if not self.has_task_labels:
      raise RuntimeError(
          'enable_death_negatives() needs per-episode outcome labels; this '
          'buffer has none, so no episode can be called a death trajectory.')
    if pool.size == 0:
      raise RuntimeError(
          'enable_death_negatives() found 0 not-reached episodes among '
          f'{self._num_eps}: the term has nothing to draw from. The merged '
          'positives+negatives dataset, not the positives-only file, is what '
          'it needs.')
    self._death_pool = pool
    self._death_batch = int(batch_size)
    return pool.size

  def enable_external_death_negatives(self, source, batch_size):
    """Like ``enable_death_negatives``, but the death pool lives in a
    SEPARATE frozen buffer ``source`` (see config.death_dataset).

    The death episodes are then never anchors of the ordinary batch, so the
    ordinary NCE and the actor's BC term see only this buffer's episodes;
    the death trajectories reach the loss through the label-0 term alone."""
    pool = source.death_episodes
    if batch_size <= 0:
      raise ValueError(f'death batch size must be positive, got {batch_size}')
    if pool.size == 0:
      raise RuntimeError(
          'enable_external_death_negatives() found 0 not-reached episodes '
          'in the death dataset: the term has nothing to draw from.')
    self._death_src = source
    self._death_pool = pool
    self._death_batch = int(batch_size)
    return pool.size

  def add_episode(self, obs, act, length=None, task_goal=None, success=None):
    """Store one episode. obs: [L, full_obs_dim], act: [L, action_dim].

    ``length`` (optional) = number of VALID observations (<= L); the rest of
    the row is padding the relabeler must not sample. Raises if frozen.

    ``task_goal`` [goal_dim] and ``success`` (0.0/1.0) are the episode's task
    goal and whether it reached it; they must be given together and are only
    read by the critic's task-goal term."""
    if self._frozen:
      raise RuntimeError(
          'TrajectoryBuffer is frozen (offline mode): add_episode() is '
          'disabled -- the fixed dataset must never grow or change.')
    obs = np.asarray(obs, dtype=self._obs_dtype)
    act = np.asarray(act, dtype=np.float32)
    assert obs.shape[0] == self._L, (
        f'expected {self._L} obs, got {obs.shape[0]}')
    slot = self._write
    self._obs[slot] = obs
    self._act[slot] = act
    if length is None:
      self._lengths_arr[slot] = self._L
    else:
      length = int(length)
      assert 2 <= length <= self._L, f'episode length {length} out of [2, {self._L}]'
      self._lengths_arr[slot] = length
      if length != self._L:
        self._use_lengths = True
    if (task_goal is None) != (success is None):
      raise ValueError('task_goal and success must be given together')
    if task_goal is None:
      self._task_goal[slot] = 0.0
      self._task_success[slot] = 0.0
    else:
      tg = np.asarray(task_goal, np.float32).reshape(-1)
      assert tg.shape == (self._goal_dim,), (
          f'task_goal must be {self._goal_dim} goal coords, got {tg.shape}')
      self._task_goal[slot] = tg
      self._task_success[slot] = float(success)
      self._n_labeled += 1
    self._write = (self._write + 1) % self._capacity_eps
    self._num_eps = min(self._num_eps + 1, self._capacity_eps)

  def __len__(self):
    """Number of transitions currently available."""
    if self._use_lengths:
      return int(np.sum(self._lengths_arr[:self._num_eps] - 1))
    return self._num_eps * (self._L - 1)

  @property
  def ready_steps(self):
    return len(self)

  @property
  def lengths(self):
    """Per-episode valid observation counts for the stored episodes."""
    return self._lengths_arr[:self._num_eps].copy()

  def flat_transitions(self):
    """Every stored one-step transition, flattened, in storage order.

    Returns ``(states, actions, next_states, episode_ids)``, each with leading
    dimension ``len(self)``. ``states``/``next_states`` are the STATE half only
    (``obs[..., :obs_dim]``); the env's stored goal half is dropped exactly as
    in ``sample()``. Padded tails are excluded via the per-episode valid
    lengths, so no pair straddles an episode boundary.

    READ-ONLY by construction: it copies out of the storage arrays and never
    advances ``self._rng``. That matters because the goal-relabeling stream is
    shared -- a consumer that drew its data through ``sample()`` instead would
    shift every subsequent CRL batch and silently break comparability with
    previous runs. Used to fit the ETT one-step transition model.
    """
    if self._num_eps == 0:
      z = np.zeros((0, self._obs_dim), np.float32)
      return (z, np.zeros((0, self._act.shape[-1]), np.float32), z.copy(),
              np.zeros((0,), np.int64))
    s, a, sn, eid = [], [], [], []
    for e in range(self._num_eps):
      n = int(self._lengths_arr[e]) - 1          # transitions in this episode.
      s.append(self._obs[e, :n, :self._obs_dim].astype(np.float32))
      sn.append(self._obs[e, 1:n + 1, :self._obs_dim].astype(np.float32))
      a.append(self._act[e, :n].astype(np.float32))
      eid.append(np.full(n, e, dtype=np.int64))
    return (np.concatenate(s), np.concatenate(a), np.concatenate(sn),
            np.concatenate(eid))

  def flat_goals(self):
    """The stored GOAL half of every transition's observation. [len(self), G].

    Row-for-row aligned with ``flat_transitions()`` -- same episodes, same
    order, same exclusion of padded tails -- so ``concatenate([states, goals])``
    reconstructs the full observation the agent's policy network consumes.
    That is the only reason this exists: the ETT stage-0 generators
    (``CausalTransitionModel.sample`` / ``.rollout``) have to hand a
    goal-conditioned consumer a full observation, and no head predicts a goal
    -- so the drawn rows' goal is inherited, and ``flat_transitions``
    deliberately drops the goal half.

    READ-ONLY in the same sense as ``flat_transitions``: it copies out of
    storage and never advances ``self._rng``.
    """
    if self._num_eps == 0:
      return np.zeros((0, self._obs.shape[-1] - self._obs_dim), np.float32)
    g = []
    for e in range(self._num_eps):
      n = int(self._lengths_arr[e]) - 1        # transitions, not observations.
      g.append(self._obs[e, :n, self._obs_dim:].astype(np.float32))
    return np.concatenate(g)

  def set_anchor_cuts(self, cuts):
    """Restrict ANCHOR times to rows [0, cut_e) per episode ("scheme C").

    The future-goal window is deliberately LEFT AT FULL LENGTH: j is still
    drawn from all rows > i up to L-1, so the geometric relabeling law
    ``P(j) prop discount**(j-i)`` is exactly the fixed-length one. Only the
    marginal over anchors changes -- episodes are still drawn uniformly, then
    the row is drawn uniformly inside that episode's [0, cut).

    Why this is the safe half of the knob: the anchor marginal decides only
    WHERE the critic is fitted, not the conditional target p(g|s,a) at each
    (s, a). The induced change in the goal marginal p(g) shifts the NCE
    optimum ``log[p(g|s,a)/p(g)]`` by a term depending on g alone, which
    cancels in the actor's argmax over a.

    Mutually exclusive with variable ``lengths`` (that path truncates the
    future window, which is exactly what this mode must not do).

    Args:
      cuts: [num_eps] ints, clipped into [1, L-1].
    """
    if self._frozen:
      raise RuntimeError('TrajectoryBuffer is frozen: set_anchor_cuts() would '
                         'change the sampling distribution after the audit.')
    if self._use_lengths:
      raise ValueError(
          'set_anchor_cuts() is incompatible with variable episode lengths: '
          'the lengths path also truncates the future-goal window, while the '
          'anchor cut must leave it at full length.')
    cuts = np.asarray(cuts, dtype=np.int64)
    if cuts.shape != (self._num_eps,):
      raise ValueError(f'expected cuts of shape ({self._num_eps},), '
                       f'got {cuts.shape}')
    self._anchor_cut_arr[:self._num_eps] = np.clip(cuts, 1, self._L - 1)
    self._use_anchor_cut = True

  @property
  def anchor_cuts(self):
    """Per-episode anchor cut rows (L-1 for every episode when inactive)."""
    return self._anchor_cut_arr[:self._num_eps].copy()

  @property
  def use_anchor_cut(self):
    return self._use_anchor_cut

  def set_balanced_buckets(self, traj_idx, row_idx, bucket_id, cap=None):
    """Enable BALANCED (s, a) anchor sampling: pick a bucket uniformly, then a
    row uniformly inside it.

    Motivation: in an offline dataset each transition appears in proportion to
    how often the behaviour policy chose it, so drawing anchors uniformly over
    ROWS is really drawing them weighted by the behaviour policy's preference.
    That fights the whole point of a pessimistic method, whose job is to move
    the agent onto a route the behaviour policy rarely took: the safe route is
    rare *because* it needs protecting, and rare means quiet in a
    frequency-weighted gradient. Measured at the fork of this benchmark, the
    shortcut outnumbers the safe branch 5:1 under uniform row sampling and
    13.8:1 once anchor cuts are on.

    Caveat, recorded rather than hidden: balancing changes the distribution the
    loss takes its expectation over, which shifts the contrastive optimum by a
    term depending on the goal alone. Harmless while a single fixed goal is
    ever commanded (this env), but it must be re-derived before any
    cross-goal/HER-style comparison.

    ``cap`` guards the other end. Strictly uniform-over-buckets upweights a
    bucket by N / (n_buckets * count), so on continuous data a bucket holding
    one row gets amplified ~1000x: measured here, 22.5% of every batch would
    have come from buckets backed by 348 of 93,779 rows. With a cap, a bucket
    is drawn with probability proportional to ``min(count, cap)``, so buckets
    at or below the cap keep their relative frequencies (no amplification of
    near-empty ones) while over-represented buckets are flattened -- which is
    the part that actually matters. ``None`` restores strict uniform.

    Args:
      traj_idx, row_idx: [M] eligible anchor coordinates.
      bucket_id: [M] contiguous bucket ids in [0, n_buckets).
      cap: weight ceiling per bucket, or None for strict uniform.
    """
    if self._frozen:
      raise RuntimeError('TrajectoryBuffer is frozen: set_balanced_buckets() '
                         'would change the sampling distribution after audit.')
    traj_idx = np.asarray(traj_idx, np.int64)
    row_idx = np.asarray(row_idx, np.int64)
    bucket_id = np.asarray(bucket_id, np.int64)
    if not (len(traj_idx) == len(row_idx) == len(bucket_id)):
      raise ValueError('traj_idx/row_idx/bucket_id must be the same length')
    if len(traj_idx) == 0:
      raise ValueError('no eligible anchors for balanced sampling')
    # Sort by bucket so each bucket is a contiguous slice (CSR-style), which
    # makes "uniform bucket, then uniform member" a vectorised gather.
    order = np.argsort(bucket_id, kind='stable')
    self._bal_traj = traj_idx[order]
    self._bal_row = row_idx[order]
    b = bucket_id[order]
    uniq, first, counts = np.unique(b, return_index=True, return_counts=True)
    self._bal_offset = first.astype(np.int64)
    self._bal_count = counts.astype(np.int64)
    self._n_buckets = len(uniq)
    if cap is None:
      self._bal_cdf = None                        # strict uniform over buckets
    else:
      w = np.minimum(self._bal_count, int(cap)).astype(np.float64)
      self._bal_cdf = np.cumsum(w / w.sum())
      self._bal_cdf[-1] = 1.0                     # guard fp drift
    self._bal_cap = cap
    self._use_balanced = True

  @property
  def use_balanced(self):
    return getattr(self, '_use_balanced', False)

  @property
  def balanced_bucket_sizes(self):
    return self._bal_count.copy() if self.use_balanced else None

  def set_anchor_strata(self, strata, counts):
    """Enable STRATIFIED anchor sampling: every batch holds a fixed number of
    anchors from each stratum, each stratum drawn from its own weighted list
    of (traj, row) anchors; the future-goal window and the discounted
    relabeling law are untouched (as in the anchor-cut and balanced paths).

    Motivation (PointMaze fork diagnosis, Step 10b): the rows that decide the
    route -- fork anchors whose relabeled goal is the task goal cell -- are
    0.7% of what the critic trains on, and the 30k critics recover none to a
    third of the fork margin the relabeling law asks for.  A stratum that
    fixes a share of every batch to fork anchors raises that weight without
    changing any loss.  The absorbing line's G1 experiment used the same
    composition (128 ordinary + 64 down + 64 right anchors per batch of 256).

    The first stratum is normally the buffer's own law -- every eligible
    anchor with weight 1 / (L_e - 1), i.e. episode uniform then anchor
    uniform -- so counts (B, 0, ...) reproduce the plain law up to the RNG
    stream.  A batch size other than sum(counts) is split proportionally, the
    remainder going to the first stratum.

    Args:
      strata: list of (traj_idx, row_idx, weight) triples; weight None means
        uniform over that stratum's rows.
      counts: anchors per batch from each stratum (same length as strata).
    """
    if self._frozen:
      raise RuntimeError('TrajectoryBuffer is frozen: set_anchor_strata() '
                         'would change the sampling distribution after audit.')
    if getattr(self, '_use_balanced', False):
      raise ValueError('set_anchor_strata() and set_balanced_buckets() are '
                       'mutually exclusive')
    if len(strata) == 0 or len(strata) != len(counts):
      raise ValueError('strata and counts must be non-empty and equal in length')
    lengths = self._lengths_arr
    parts = []
    for k, (tj, rw, w) in enumerate(strata):
      tj = np.asarray(tj, np.int64)
      rw = np.asarray(rw, np.int64)
      if len(tj) == 0 or len(tj) != len(rw):
        raise ValueError(f'stratum {k}: empty or mismatched traj/row arrays')
      if np.any(tj < 0) or np.any(tj >= self._num_eps):
        raise ValueError(f'stratum {k}: trajectory index out of range')
      if np.any(rw < 0) or np.any(rw >= lengths[tj] - 1):
        raise ValueError(f'stratum {k}: anchor row outside [0, len - 2]')
      w = (np.ones(len(tj), np.float64) if w is None
           else np.asarray(w, np.float64))
      if len(w) != len(tj) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError(f'stratum {k}: invalid weights')
      cdf = np.cumsum(w / w.sum())
      cdf[-1] = 1.0                               # guard fp drift
      parts.append((tj, rw, cdf))
    counts = np.asarray(counts, np.int64)
    if np.any(counts < 0) or not counts.sum() > 0:
      raise ValueError('counts must be non-negative with a positive sum')
    self._strata = parts
    self._strata_counts = counts
    self._use_strata = True

  @property
  def use_anchor_strata(self):
    return getattr(self, '_use_strata', False)

  @property
  def anchor_strata_sizes(self):
    return ([len(p[0]) for p in self._strata] if self.use_anchor_strata
            else None)

  def content_sha256(self):
    """SHA-256 over the stored obs+act tensors (immutability checksum)."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(self._obs[:self._num_eps]).tobytes())
    h.update(np.ascontiguousarray(self._act[:self._num_eps]).tobytes())
    h.update(np.ascontiguousarray(self._lengths_arr[:self._num_eps]).tobytes())
    return h.hexdigest()

  def save(self, path):
    """Atomic snapshot (tmp + replace) of contents, pointers, and RNG state."""
    import json as _json
    import os as _os
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
      np.savez_compressed(
          f, obs=self._obs[:self._num_eps], act=self._act[:self._num_eps],
          write=self._write, num_eps=self._num_eps,
          rng_state=np.frombuffer(
              _json.dumps(self._rng.bit_generator.state).encode(),
              dtype=np.uint8))
    _os.replace(tmp, path)

  def load(self, path):
    """Restore a snapshot produced by save() into this (same-shape) buffer.

    Context-managed np.load: the file handle MUST be closed, or Windows will
    refuse the atomic os.replace of the next snapshot onto this path."""
    import json as _json
    with np.load(path) as d:
      n = int(d['num_eps'])
      assert d['obs'].shape[1:] == self._obs.shape[1:], 'buffer shape mismatch'
      assert n <= self._capacity_eps
      self._obs[:n] = d['obs']
      self._act[:n] = d['act']
      self._num_eps = n
      self._write = int(d['write']) % self._capacity_eps
      self._rng.bit_generator.state = _json.loads(
          d['rng_state'].tobytes().decode())
    return n

  def _draw_indices(self, batch_size, pool=None, rng=None):
    """Draw (traj, i, j): anchor time i and future goal time j>i, BOTH in the
    SAME trajectory (so relabeling never crosses an episode boundary) and both
    within the episode's valid length (so it never samples a padded tail).

    ``pool`` (optional) restricts the trajectory draw to those episode
    indices, and ``rng`` (optional) draws from a different stream; both are
    used only by the death-negative batch, and with both at None this is the
    original sampler, number for number."""
    L = self._L
    ne = self._num_eps
    rng = self._rng if rng is None else rng

    if getattr(self, '_use_strata', False):
      # Fixed per-stratum counts, weighted rows inside each stratum, then the
      # SAME discounted future-goal law over the episode's valid rows.
      counts = self._strata_counts
      if batch_size != int(counts.sum()):
        counts = np.floor(counts * (batch_size / counts.sum())).astype(np.int64)
        counts[0] += batch_size - int(counts.sum())
      traj_parts, i_parts = [], []
      for (tj, rw, cdf), n in zip(self._strata, counts):
        if n <= 0:
          continue
        pos = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'),
                         len(cdf) - 1)
        traj_parts.append(tj[pos])
        i_parts.append(rw[pos])
      perm = rng.permutation(batch_size)
      traj = np.concatenate(traj_parts)[perm]
      i = np.concatenate(i_parts)[perm]
      arange = np.arange(L)
      valid = arange[None, :] < self._lengths_arr[traj][:, None]
      future = (arange[None, :] > i[:, None]) & valid
      logp = (arange[None, :] - i[:, None]) * self._log_discount
      logits = np.where(future, logp, -np.inf)
      g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
      return traj, i, np.argmax(logits + g, axis=1)

    if getattr(self, '_use_balanced', False):
      # Bucket uniform, then a row uniform inside that bucket. The future-goal
      # window is untouched, exactly as in the anchor-cut path.
      if self._bal_cdf is None:
        kb = rng.integers(0, self._n_buckets, size=batch_size)
      else:
        kb = np.searchsorted(self._bal_cdf, rng.random(batch_size), side='right')
        kb = np.minimum(kb, self._n_buckets - 1)
      pos = self._bal_offset[kb] + np.floor(
          rng.random(batch_size) * self._bal_count[kb]).astype(np.int64)
      traj = self._bal_traj[pos]
      i = self._bal_row[pos]
      arange = np.arange(L)
      future = arange[None, :] > i[:, None]              # [B, L] FULL length.
      logp = (arange[None, :] - i[:, None]) * self._log_discount
      logits = np.where(future, logp, -np.inf)
      g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
      return traj, i, np.argmax(logits + g, axis=1)

    if pool is None:
      traj = rng.integers(0, ne, size=batch_size)        # which trajectory.
    else:
      traj = pool[rng.integers(0, pool.size, size=batch_size)]
    if not self._use_lengths and not self._use_anchor_cut:
      # Fixed-length path -- byte-identical RNG stream to the original.
      i = rng.integers(0, L - 1, size=batch_size)        # anchor in [0, L-2].
      arange = np.arange(L)                              # [L]
      future = arange[None, :] > i[:, None]              # [B, L]
    elif self._use_anchor_cut:
      # Scheme C: episode uniform (above), then anchor uniform inside that
      # episode's [0, cut). The FUTURE WINDOW IS NOT TOUCHED -- j still ranges
      # over every row > i up to L-1, so the relabeling law is identical to the
      # fixed-length one and post-cut (parked / dead) states remain reachable
      # as positive goals, exactly as in the original.
      Ct = self._anchor_cut_arr[traj]                    # [B] cut per row.
      i = np.floor(rng.random(batch_size) * Ct).astype(np.int64)
      arange = np.arange(L)                              # [L]
      future = arange[None, :] > i[:, None]              # [B, L] FULL length.
    else:
      # Variable-length: mask the padded tail per row (valid = arange < len).
      Lt = self._lengths_arr[traj]                       # [B] valid obs counts.
      i = np.floor(rng.random(batch_size) * (Lt - 1)).astype(np.int64)
      arange = np.arange(L)                              # [L]
      valid = arange[None, :] < Lt[:, None]              # [B, L] within episode.
      future = (arange[None, :] > i[:, None]) & valid    # [B, L]
    logp = (arange[None, :] - i[:, None]) * self._log_discount  # [B, L]
    logits = np.where(future, logp, -np.inf)
    # Gumbel-max categorical sample over the same distribution as flatten_fn.
    g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
    j = np.argmax(logits + g, axis=1)                    # [B]
    return traj, i, j

  def sampled_indices(self, batch_size):
    """Public alias of _draw_indices for the offline relabel-boundary audit."""
    return self._draw_indices(batch_size)

  def sample(self, batch_size):
    """Sample a relabeled Transition batch (numpy arrays of size batch_size)."""
    traj, i, j = self._draw_indices(batch_size)

    # .astype(float32): no-op for state envs; uint8 -> float32 for image envs
    # (networks normalize by /255 themselves).
    state = self._obs[traj, i, :self._obs_dim].astype(np.float32, copy=False)
    next_state = self._obs[traj, i + 1, :self._obs_dim].astype(np.float32,
                                                               copy=False)
    goal_state = self._obs[traj, j, :self._obs_dim].astype(np.float32,
                                                           copy=False)
    goal = obs_to_goal(goal_state, self._start_index, self._end_index,
                       self._goal_indices)

    new_obs = np.concatenate([state, goal], axis=1)
    new_next_obs = np.concatenate([next_state, goal], axis=1)
    action = self._act[traj, i]
    next_action = self._act[traj, i + 1]

    death_obs, death_act = (None, None)
    if self._death_batch:
      death_obs, death_act = self._draw_death(self._death_batch)

    return Transition(
        observation=new_obs,
        action=action,
        reward=np.zeros((batch_size,), np.float32),       # unused by losses.
        discount=np.full((batch_size,), self._discount, np.float32),
        next_observation=new_next_obs,
        next_action=next_action,
        task_goal=self._task_goal[traj],
        task_success=self._task_success[traj],
        death_observation=death_obs,
        death_action=death_act,
    )

  def _draw_death(self, batch_size):
    """One hindsight-relabeled batch drawn only from the death trajectories.

    Same relabeling rule as ``sample()`` -- the pair is ordinary; what makes
    it a negative is the label crl/losses.py gives it."""
    src = self._death_src
    traj, i, j = src._draw_indices(   # pylint: disable=protected-access
        batch_size, pool=self._death_pool, rng=self._death_rng)
    state = src._obs[traj, i, :self._obs_dim].astype(np.float32, copy=False)
    goal_state = src._obs[traj, j, :self._obs_dim].astype(np.float32,
                                                          copy=False)
    goal = obs_to_goal(goal_state, self._start_index, self._end_index,
                       self._goal_indices)
    return np.concatenate([state, goal], axis=1), src._act[traj, i]
