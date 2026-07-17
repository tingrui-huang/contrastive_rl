"""Continuous-Manski primitives (Steps 1-2 of the swamp PointMaze port).

The causal lower-bound occupancy d_lb (per-step Manski bound; continuous
theory archived in notes/continuous_manski_lemma2prime.md) needs three
ingredients before the Thm-2 sampler can run on the frozen offline dataset:

  * a NON-DEGENERATE propensity: continuous actions have P(X=x|s)=0, so the
    mixing weight must be the probability of an action NEIGHBORHOOD -- here
    K direction sectors + a near-zero 'stay' bin, conditioned on the maze
    cell (the state-side coarsening). P_hat(bin|cell) is counted from the
    frozen dataset with Laplace smoothing.
  * a V_lb ORDERING on cells: the sampler's step (ii) only takes an argmin,
    so any monotone transform of the true value works. First version: BFS
    distance-to-goal (larger = worse).
  * the neighborhood N(s,x): current cell + 4-adjacent passable cells -- a
    superset of the one-step reachable set under EVERY swamp configuration
    (per-step displacement <= 1 cell), which is what validity requires.
    The pessimistic teleport target is the member with the LARGEST BFS
    distance (ties prefer staying put).

Everything here is pure numpy over (walls, obs, act): no env interaction,
no learned components, and NO ACCESS to audit-only fields (swamp_bits,
route_label, ...) -- the crl.offline_audit contract applies.
"""
import collections

import numpy as np


def n_bins(n_sectors):
  return n_sectors + 1                 # bin 0 = stay, 1..K = direction sectors


def action_bins(act, n_sectors=8, zero_thresh=0.15):
  """Map 2D actions to bins: 0 = stay (|a| < zero_thresh), else 1 + sector.

  Sectors are ROTATED by half a width so that the cardinal and diagonal
  directions (0, 45, 90, ... deg) are sector CENTERS, not edges: the
  behavior policy moves along those directions, and an edge there would
  split one behavioral mode across two bins, artificially halving the
  taken-action propensity (and exponentially inflating teleports over a
  Geom(1-gamma) walk)."""
  act = np.asarray(act, np.float64)
  mag = np.linalg.norm(act, axis=-1)
  ang = np.arctan2(act[..., 1], act[..., 0])            # (-pi, pi]
  frac = (ang + np.pi) / (2.0 * np.pi) * n_sectors + 0.5
  sector = np.floor(frac).astype(int) % n_sectors
  return np.where(mag < zero_thresh, 0, 1 + sector)


def cells_of(walls, xy):
  """Vectorized floor+clip cell lookup, matching env._discretize_state."""
  ij = np.floor(np.asarray(xy, np.float64)).astype(int)
  ij[..., 0] = np.clip(ij[..., 0], 0, walls.shape[0] - 1)
  ij[..., 1] = np.clip(ij[..., 1], 0, walls.shape[1] - 1)
  return ij


def fit_propensity(walls, cells, bins, n_sectors, alpha=1.0):
  """Counts and Laplace-smoothed P_hat(bin | cell) over the grid."""
  k = n_bins(n_sectors)
  counts = np.zeros((walls.shape[0], walls.shape[1], k))
  np.add.at(counts, (cells[:, 0], cells[:, 1], bins), 1.0)
  probs = (counts + alpha) / (counts.sum(-1, keepdims=True) + alpha * k)
  return counts, probs


def mean_loglik(probs, cells, bins):
  return float(np.mean(np.log(probs[cells[:, 0], cells[:, 1], bins])))


def entropy_map(counts, alpha=1.0):
  """Per-cell propensity entropy (nats); NaN where the cell was never visited."""
  tot = counts.sum(-1)
  k = counts.shape[-1]
  p = (counts + alpha) / (tot + alpha * k)[..., None]
  h = -(p * np.log(p)).sum(-1)
  h[tot == 0] = np.nan
  return h


def bfs_dist_map(walls, goal_cell):
  """BFS distance (cells, 4-conn) to goal over passable cells; -1 elsewhere."""
  dist = np.full(walls.shape, -1, int)
  goal_cell = tuple(int(v) for v in goal_cell)
  dist[goal_cell] = 0
  q = collections.deque([goal_cell])
  while q:
    c = q.popleft()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      n = (c[0] + di, c[1] + dj)
      if (0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1]
          and walls[n] == 0 and dist[n] < 0):
        dist[n] = dist[c] + 1
        q.append(n)
  return dist


def neighborhood(walls, cell):
  """N(s,x): the cell itself + 4-adjacent passable cells (self listed FIRST
  so that argmax tie-breaking prefers staying put -- 'stuck' pessimism)."""
  out = [tuple(int(v) for v in cell)]
  for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    n = (out[0][0] + di, out[0][1] + dj)
    if (0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1]
        and walls[n] == 0):
      out.append(n)
  return out


def worst_neighbor_map(walls, dist, hazard_cells=()):
  """Pessimistic teleport target per passable cell: argmax BFS distance
  over N(s,x). Returns {cell: worst_cell}.

  ``hazard_cells`` (static geometry, e.g. the swamp corridor -- NOT the
  hidden bits) are treated as V_lb = 0 absorbing states: any N containing
  one teleports INTO it, mirroring the discrete WindyCorridor V_lower's
  LAVA_PENALTY. V_lb = 0 is a valid lower bound (success prob >= 0), so
  Thm-1 validity is unaffected -- d_lb only gets more pessimistic."""
  hazard = set(map(tuple, hazard_cells))
  worst = {}
  for i, j in np.argwhere(walls == 0):
    cand = neighborhood(walls, (i, j))
    ds = [(np.inf if c in hazard else dist[c]) for c in cand]
    worst[(int(i), int(j))] = cand[int(np.argmax(ds))]
  return worst


class ManskiSampler:
  """Thm-2 sampler for d_lb positives on a frozen episodic dataset.

  Walk semantics, starting from a dataset anchor (episode e, step t):
    * stop with prob (1-gamma) each step (T ~ Geom(1-gamma), T=0 allowed);
    * else, with prob P_hat(bin(x_t) | cell(s_t)): advance one step along
      the stored trajectory (the empirical P_obs transition; x_t is the
      logged behavior action, which IS the policy being evaluated);
    * else TELEPORT to the pessimistic worst neighbor cell (argmax BFS
      distance over N) and re-anchor to a random dataset point in that
      cell (its logged action is a fresh behavior draw, matching
      "X_{t+1} ~ pi(.|S_{t+1})" in Thm 2).
  Episode truncation (reaching the last stored action) re-anchors to a
  random dataset point in the SAME cell and keeps walking: the env is
  non-terminating, episode ends are collection artifacts. NOTE this
  continuation differs from the existing replay.py law (truncated
  geometric WITHIN one trajectory) -- for single-variable comparisons the
  confounded baseline should be retrained with this walk at p_override=1
  so that turning the propensity on is the only change.

  `p_override=1.0` disables all pessimism (degeneration check / baseline);
  audit-only dataset fields are never read.
  """

  def __init__(self, obs, act, walls, probs, n_sectors, zero_thresh,
               dist, gamma, seed=0, hazard_cells=(), reachable_n=False):
    ne, length = obs.shape[0], obs.shape[1]
    self._length = length
    self._gamma = float(gamma)
    self._rng = np.random.default_rng(seed)
    self._walls = walls
    # state-flat arrays over (e, t), t in 0..L-1
    self._xy = obs[:, :, :2].reshape(-1, 2)
    self._t_of = np.tile(np.arange(length), ne)
    cells = cells_of(walls, self._xy)
    self._cell_i, self._cell_j = cells[:, 0], cells[:, 1]
    # action-dependent tables; rows with t == L-1 hold a dummy action and are
    # never consulted (walkers re-anchor before branching there)
    bins = action_bins(act.reshape(-1, 2), n_sectors, zero_thresh)
    self._phat = probs[self._cell_i, self._cell_j, bins]
    self._act = act.reshape(-1, 2)
    # anchor pools per cell (t <= L-2 so an action exists)
    self._anchorable = self._t_of <= length - 2
    self._pool = {}
    for (i, j) in map(tuple, np.argwhere(walls == 0)):
      members = np.where(self._anchorable & (self._cell_i == i)
                         & (self._cell_j == j))[0]
      if len(members):
        self._pool[(i, j)] = members
    worst = worst_neighbor_map(walls, dist, hazard_cells=hazard_cells)
    # teleport target pool per source cell (fallback: stay in place)
    self._worst_pool = {c: self._pool.get(w, self._pool[c])
                        for c, w in worst.items() if c in self._pool}
    # hazard cells are V_lb = 0 ABSORBING: a walker whose pessimistic
    # teleport lands in one dies there (endpoint != goal, forever) --
    # the continuous analogue of the discrete lava jump. Walkers passing
    # through the same cells on the EMPIRICAL branch (i) are unaffected.
    self._hazard_mask = np.zeros(walls.shape, bool)
    for c in map(tuple, hazard_cells):
      self._hazard_mask[c] = True
    # reachable-set N(s,x) (the discrete worst_case_kernel semantics: N =
    # the outcomes of THIS action under every u). In the swamp env u only
    # modulates motion inside swamp cells, so an action is u-affected iff
    # its motion starts in or enters a swamp cell; its worst outcome is
    # "stuck in the active swamp" (absorbing). Every other action has a
    # singleton reachable set => the pessimistic branch is a no-op, and in
    # particular there is NO backward teleport anywhere.
    self._reachable = bool(reachable_n)
    if self._reachable:
      if not hazard_cells:
        raise ValueError('reachable_n needs hazard_cells (the u support)')
      nxt = self._xy + np.clip(self._act, -1.0, 1.0)
      ncells = cells_of(walls, nxt)
      self._u_flag = (self._hazard_mask[self._cell_i, self._cell_j]
                      | self._hazard_mask[ncells[:, 0], ncells[:, 1]])

  def _group_reanchor(self, cur, idx, pools):
    """Re-anchor walkers `idx` via `pools[source_cell]`, grouped by cell so the
    python-level loop is over <= n_cells, not over walkers (training path)."""
    key = (self._cell_i[cur[idx]] * self._walls.shape[1]
           + self._cell_j[cur[idx]])
    for k in np.unique(key):
      grp = idx[key == k]
      pool = pools.get((int(k) // self._walls.shape[1],
                        int(k) % self._walls.shape[1]))
      if pool is None:        # cell with no anchorable data: stay in place
        continue
      cur[grp] = pool[self._rng.integers(len(pool), size=len(grp))]

  def walk_from(self, start_flat, p_override=None, max_steps=2000,
                collect_maps=False):
    """Run the Thm-2 walk from given dataset flat indices; returns endpoint
    flat indices (and teleport/visit heatmaps when collect_maps)."""
    rng = self._rng
    cur = np.asarray(start_flat, np.int64).copy()
    batch = len(cur)
    alive = np.ones(batch, bool)
    teleports = np.zeros(self._walls.shape, np.int64) if collect_maps else None
    visits = np.zeros(self._walls.shape, np.int64) if collect_maps else None
    n_steps = 0
    while alive.any() and n_steps < max_steps:
      n_steps += 1
      alive &= rng.random(batch) < self._gamma        # stop w.p. 1-gamma
      idx = np.where(alive)[0]
      if not len(idx):
        break
      # episode truncation: no action at t == L-1 -> same-cell re-anchor
      trunc = idx[self._t_of[cur[idx]] == self._length - 1]
      if len(trunc):
        self._group_reanchor(cur, trunc, self._pool)
      ci, cj = self._cell_i[cur[idx]], self._cell_j[cur[idx]]
      if collect_maps:
        np.add.at(visits, (ci, cj), 1)
      p = self._phat[cur[idx]] if p_override is None else p_override
      if self._reachable and p_override is None:
        # reachable-set N: the coin only matters for u-affected actions,
        # whose worst outcome is stuck-in-swamp (absorb). All other
        # actions walk regardless: their reachable set is a singleton.
        stuck = self._u_flag[cur[idx]] & (rng.random(len(idx)) >= p)
        cur[idx[~stuck]] += 1
        if stuck.any():
          if collect_maps:
            np.add.at(teleports, (ci[stuck], cj[stuck]), 1)
          alive[idx[stuck]] = False
        continue
      walk = rng.random(len(idx)) < p
      cur[idx[walk]] += 1                             # empirical transition
      tele = idx[~walk]
      if len(tele):
        if collect_maps:
          np.add.at(teleports, (ci[~walk], cj[~walk]), 1)
        self._group_reanchor(cur, tele, self._worst_pool)
        # absorbing hazard: pessimistic landings in a hazard cell terminate
        dead = self._hazard_mask[self._cell_i[cur[tele]],
                                 self._cell_j[cur[tele]]]
        if dead.any():
          alive[tele[dead]] = False
    return (cur, teleports, visits) if collect_maps else cur

  def sample(self, batch, p_override=None, max_steps=2000):
    """Probe API: random anchors + walk; returns endpoints and diagnostics."""
    all_anchors = np.concatenate(list(self._pool.values()))
    anchor = all_anchors[self._rng.integers(len(all_anchors), size=batch)]
    cur, teleports, visits = self.walk_from(
        anchor, p_override=p_override, max_steps=max_steps, collect_maps=True)
    return dict(anchor_xy=self._xy[anchor], anchor_act=self._act[anchor],
                endpoint_xy=self._xy[cur], endpoint_flat=cur,
                teleport_map=teleports, visit_map=visits)


class ManskiPositiveBuffer:
  """Frozen-buffer wrapper whose sample() draws the POSITIVE goal from the
  Thm-2 d_lb walk instead of the same-trajectory truncated-geometric future.

  Everything else -- anchor law (uniform trajectory x uniform t), the
  state/action/next_state fields, negatives (cross-batch pairs downstream),
  and the whole frozen/audit surface -- delegates to the wrapped
  TrajectoryBuffer. The base buffer's G7 relabel audit gates the BASE law;
  the walk itself is gated by scripts/manski_sampler_probe.py (G5-G8).
  Walk goals CROSS episode boundaries BY DESIGN (teleport + re-anchor).

  p_override=1.0 gives the matched no-pessimism BASELINE arm: identical walk
  machinery (including truncation re-anchor continuation), teleports off --
  so turning the propensity on is the single experimental variable.
  """

  def __init__(self, base, sampler, p_override=None):
    assert not base._use_lengths, 'Manski wrapper assumes fixed-length episodes'
    self._base = base
    self._sampler = sampler
    self._p_override = p_override

  def __getattr__(self, name):
    return getattr(self._base, name)

  def sample(self, batch_size):
    from crl.losses import Transition
    from crl.replay import obs_to_goal as _obs_to_goal
    b = self._base
    rng = self._sampler._rng
    L = b._L
    traj = rng.integers(0, b._num_eps, size=batch_size)
    i = rng.integers(0, L - 1, size=batch_size)
    endpoint = self._sampler.walk_from(traj * L + i,
                                       p_override=self._p_override)
    tj, ti = endpoint // L, endpoint % L
    state = b._obs[traj, i, :b._obs_dim].astype(np.float32, copy=False)
    next_state = b._obs[traj, i + 1, :b._obs_dim].astype(np.float32,
                                                         copy=False)
    goal_state = b._obs[tj, ti, :b._obs_dim].astype(np.float32, copy=False)
    goal = _obs_to_goal(goal_state, b._start_index, b._end_index,
                        b._goal_indices)
    return Transition(
        observation=np.concatenate([state, goal], axis=1),
        action=b._act[traj, i],
        reward=np.zeros((batch_size,), np.float32),
        discount=np.full((batch_size,), b._discount, np.float32),
        next_observation=np.concatenate([next_state, goal], axis=1),
        next_action=b._act[traj, i + 1],
    )


def build_positive_buffer(base_buffer, table_path, walls, goal_xy, gamma,
                          seed, p_override=None, hazard_cells=(),
                          reachable_n=False):
  """Construct the ManskiPositiveBuffer from a frozen TrajectoryBuffer and a
  saved propensity table (artifacts of scripts/fit_propensity.py)."""
  import json
  table = np.load(table_path, allow_pickle=True)
  tcfg = json.loads(str(table['config']))
  goal_cell = tuple(cells_of(walls, np.asarray(goal_xy)))
  dist = bfs_dist_map(walls, goal_cell)
  n = base_buffer._num_eps
  sampler = ManskiSampler(
      base_buffer._obs[:n], base_buffer._act[:n], walls, table['probs'],
      tcfg['sectors'], tcfg['zero_thresh'], dist, gamma, seed=seed,
      hazard_cells=hazard_cells, reachable_n=reachable_n)
  return ManskiPositiveBuffer(base_buffer, sampler, p_override)


def replay_law_endpoints(obs, discount, batch, seed=0):
  """Reference endpoints under the EXISTING replay.py law: anchor (e,i)
  uniform, future j>i in the same episode, prob ~ discount**(j-i)
  (truncated-geometric categorical). Returns endpoint xy [batch, 2]."""
  rng = np.random.default_rng(seed)
  ne, length = obs.shape[0], obs.shape[1]
  # per remaining-length r, cdf over offsets d = 1..r
  cdfs = {}
  for r in range(1, length):
    w = discount ** np.arange(1, r + 1, dtype=float)
    cdfs[r] = np.cumsum(w / w.sum())
  e = rng.integers(ne, size=batch)
  i = rng.integers(length - 1, size=batch)            # anchor t in 0..L-2
  out = np.zeros((batch, 2), np.float32)
  for k in range(batch):
    r = length - 1 - i[k]
    d = 1 + int(np.searchsorted(cdfs[r], rng.random()))
    out[k] = obs[e[k], i[k] + d, :2]
  return out
