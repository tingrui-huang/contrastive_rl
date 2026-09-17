"""Balanced rows for the actor's BC term (Step 8 of the PointMaze fork diagnosis).

The critic term and the NCE critic keep the replay buffer's own batches; only
the BC term's rows come from ``GroupBalancedBCSampler``, which draws from the
buffer's relabeling law reweighted inside (state cell, goal cell) groups so
that no recorded-action region holds more than ``cap`` of its group.  The
(state, goal) marginal is unchanged; only the conditional over action regions
is flattened.  History: balancing every term at once was tried twice on
PointMaze (global (s, a) buckets, commit 434cb0b, null; key-balanced actor
batches, outputs/pointmaze_absorbing_balanced_g2c_20260915_v1, retracted for
importing random-walker rows); this BC-only rule is the first positive
(outputs/pointmaze_fixed_dcritic_actor_bcbal_v1).
"""
from __future__ import annotations

import numpy as np

from crl.losses import Transition


SECTOR_NAMES_8 = ('R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR')


class GroupBalancedBCSampler:
  """Rows for the BC term only: the buffer's relabeling law, reweighted so
  that inside every (state cell, goal cell) group the recorded-action regions
  are more evenly represented.  The critic term keeps the buffer's own draws.

  Every (episode e, anchor i, future j) triple is enumerated with exactly the
  weight the TrajectoryBuffer's variable-length sampler gives it,
      w = 1/N * 1/(L_e - 1) * gamma^(j-i) / sum_{k=1}^{L_e-1-i} gamma^k,
  (checked against the buffer empirically at start-up).  Group key = the
  floor(xy) cell of the anchor's newest frame and of the relabeled goal's
  newest frame; region = the recorded action's angle in ``n_sectors`` bins
  rotated half a width so the cardinals AND the diagonals sit at bin centres
  (the D fix's diagonal queries are their own region), plus a wait bucket
  for |a| < wait_eps.  Within a group a region's share is ceiled at ``cap``:
      w' = w * min(share, cap) / share,  renormalised so the group's total
  mass is unchanged.  So the (state, goal) marginal of the BC rows is the
  original one; only the conditional over action regions is flattened, and
  regions below the ceiling keep their natural relative frequency (no
  amplification of near-empty regions -- the failure of the strict-uniform
  key balancing in outputs/pointmaze_absorbing_balanced_g2c_20260915_v1).
  ``cap=None`` reproduces the original law with independent rows.
  """

  def __init__(self, obs, act, lengths, discount, obs_dim, cell=1.0,
               n_sectors=8, wait_eps=0.1, cap=0.25, seed=0):
    obs = np.asarray(obs)
    act = np.asarray(act)
    lengths = np.asarray(lengths, np.int64)
    n_eps = len(obs)
    tr, ii, jj = [], [], []
    for e in range(n_eps):
      lt = int(lengths[e])
      i, j = np.meshgrid(np.arange(lt - 1), np.arange(lt), indexing='ij')
      m = j > i
      tr.append(np.full(int(m.sum()), e, np.int32))
      ii.append(i[m].astype(np.int16))
      jj.append(j[m].astype(np.int16))
    self.tr, self.ii, self.jj = (np.concatenate(x) for x in (tr, ii, jj))
    lt = lengths[self.tr]
    k = lt - 1 - self.ii.astype(np.int64)
    g = float(discount)
    w = (g ** (self.jj - self.ii).astype(np.float64)
         / (g * (1.0 - g ** k) / (1.0 - g)) / (lt - 1) / n_eps)
    s_cell = np.floor(obs[self.tr, self.ii, :2] / cell).astype(np.int64)
    g_cell = np.floor(obs[self.tr, self.jj, :2] / cell).astype(np.int64)
    a = act[self.tr, self.ii].astype(np.float64)
    mag = np.linalg.norm(a, axis=1)
    ang = np.arctan2(a[:, 1], a[:, 0])
    width = 2.0 * np.pi / n_sectors
    region = np.floor((ang + width / 2.0) / width).astype(np.int64) % n_sectors
    region = np.where(mag < wait_eps, n_sectors, region)
    key = np.stack([s_cell[:, 0], s_cell[:, 1], g_cell[:, 0], g_cell[:, 1]], 1)
    self.group_keys, g_id = np.unique(key, axis=0, return_inverse=True)
    g_id = g_id.ravel().astype(np.int64)
    n_reg = n_sectors + 1
    b_id = g_id * n_reg + region
    n_groups = len(self.group_keys)
    w_group = np.bincount(g_id, weights=w, minlength=n_groups)
    w_bucket = np.bincount(b_id, weights=w, minlength=n_groups * n_reg)
    share = w_bucket / np.repeat(w_group, n_reg)
    if cap is None:
      mult = np.ones_like(w_bucket)
    else:
      m = np.where(share > 0, np.minimum(share, cap) / np.maximum(share, 1e-300), 0.0)
      new_mass = np.bincount(np.repeat(np.arange(n_groups), n_reg),
                             weights=m * w_bucket, minlength=n_groups)
      mult = m * np.repeat(w_group / np.maximum(new_mass, 1e-300), n_reg)
    self.w = w * mult[b_id]
    self.w_original = w
    self.b_id, self.g_id, self.region = b_id, g_id, region
    self.n_reg, self.n_sectors, self.cap, self.cell = n_reg, n_sectors, cap, cell
    self.wait_eps = wait_eps
    self.w_bucket, self.w_bucket_new = w_bucket, w_bucket * mult
    self.cdf = np.cumsum(self.w / self.w.sum())
    self.cdf[-1] = 1.0
    self.rng = np.random.default_rng(seed)
    self.obs, self.act, self.obs_dim = obs, act, obs_dim
    self.stats = {
        'n_pairs': int(len(w)), 'n_groups': int(n_groups),
        'n_nonempty_buckets': int((w_bucket > 0).sum()), 'cap': cap,
        'cell': cell, 'n_sectors': n_sectors, 'wait_eps': wait_eps,
        'max_multiplier': float(mult.max()), 'min_nonzero_multiplier':
            float(mult[mult > 0].min()),
        'ess_original': float(w.sum() ** 2 / (w ** 2).sum()),
        'ess_reweighted': float(self.w.sum() ** 2 / (self.w ** 2).sum()),
        'group_marginal_max_abs_dev': float(np.abs(
            np.bincount(g_id, weights=self.w, minlength=n_groups) - w_group).max()),
    }

  def group_composition(self, s_cell, g_cell):
    """(original shares, reweighted shares) over regions for one group."""
    hit = np.nonzero((self.group_keys == np.array([*s_cell, *g_cell])).all(1))[0]
    if len(hit) == 0:
      return None
    sl = slice(hit[0] * self.n_reg, (hit[0] + 1) * self.n_reg)
    b, bn = self.w_bucket[sl], self.w_bucket_new[sl]
    names = list(SECTOR_NAMES_8 if self.n_sectors == 8 else
                 [f's{r}' for r in range(self.n_sectors)]) + ['wait']
    return ({n: float(v) for n, v in zip(names, b / b.sum())},
            {n: float(v) for n, v in zip(names, bn / bn.sum())},
            float(b.sum()))

  def bucket_of(self, traj, i, j):
    s_cell = np.floor(self.obs[traj, i, :2] / self.cell).astype(np.int64)
    g_cell = np.floor(self.obs[traj, j, :2] / self.cell).astype(np.int64)
    a = self.act[traj, i].astype(np.float64)
    mag = np.linalg.norm(a, axis=1)
    ang = np.arctan2(a[:, 1], a[:, 0])
    width = 2.0 * np.pi / self.n_sectors
    region = np.floor((ang + width / 2.0) / width).astype(np.int64) % self.n_sectors
    region = np.where(mag < self.wait_eps, self.n_sectors, region)
    key = np.stack([s_cell[:, 0], s_cell[:, 1], g_cell[:, 0], g_cell[:, 1]], 1)
    # groups are lexicographically sorted by np.unique -> searchsorted on a view
    flat = lambda arr: np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * 4))).ravel()
    gk = flat(self.group_keys)
    pos = np.searchsorted(gk, flat(key))
    pos = np.minimum(pos, len(gk) - 1)
    ok = gk[pos] == flat(key)
    return np.where(ok, pos * self.n_reg + region, -1)

  def law_check(self, buffer, n=200_000, chunk=1000):
    """Max abs deviation between the buffer's empirical bucket shares and the
    enumerated original weights."""
    cnt = np.zeros(len(self.w_bucket))
    rng_state = buffer._rng.bit_generator.state     # restored below: the
    try:                                             # buffer's stream must
      for _ in range(n // chunk):                    # not move
        t, i, j = buffer.sampled_indices(chunk)
        b = self.bucket_of(t, i, j)
        if (b < 0).any():
          raise RuntimeError('buffer drew a (state, goal) group missing from '
                             'the enumeration')
        cnt += np.bincount(b, minlength=len(cnt))
    finally:
      buffer._rng.bit_generator.state = rng_state
    return float(np.abs(cnt / n - self.w_bucket).max())

  def sample(self, batch_size):
    pos = np.searchsorted(self.cdf, self.rng.random(batch_size), side='right')
    pos = np.minimum(pos, len(self.cdf) - 1)
    traj, i, j = self.tr[pos], self.ii[pos].astype(np.int64), self.jj[pos].astype(np.int64)
    state = self.obs[traj, i, :self.obs_dim].astype(np.float32)
    next_state = self.obs[traj, i + 1, :self.obs_dim].astype(np.float32)
    goal = self.obs[traj, j, :self.obs_dim].astype(np.float32)   # obs_to_goal(0, -1)
    return Transition(
        observation=np.concatenate([state, goal], 1),
        action=self.act[traj, i].astype(np.float32),
        reward=np.zeros((batch_size,), np.float32),
        discount=np.full((batch_size,), 0.95, np.float32),
        next_observation=np.concatenate([next_state, goal], 1),
        next_action=self.act[traj, i + 1].astype(np.float32))


def tree_sha(tree):
  import jax
  d = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.ascontiguousarray(np.asarray(leaf))
    d.update(str(a.dtype).encode())
    d.update(str(a.shape).encode())
    d.update(a.tobytes())
  return d.hexdigest()
