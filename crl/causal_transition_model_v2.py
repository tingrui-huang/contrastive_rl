"""Causal transition model v2: rollouts that start mid-episode at a route split.

``CausalTransitionModelV2`` is ``crl.causal_transition_model.CausalTransitionModel``
with one change: ``rollout``. Fitting, sampling, saving and loading are v1's
unchanged, and so is the pickle format -- a v1 checkpoint is turned into a v2
model with ``CausalTransitionModelV2.from_v1(model)``, with no refit.

v1 starts every generated trajectory at the source episode's INITIAL state and
copies no real prefix. v2 starts it at a route-dependent step ``t0`` inside the
source episode and KEEPS the real prefix::

    route = label of the source episode (sidecar, else geometry)
    t0    = first step with x > detour_min_x
            and y < detour_max_y                   if route == 'detour'
            first step inside the trap box          if route == 'shortcut'
            (hazard zone ``shortcut_zone``, default 2)

    row = s_0, a_0, ..., s_{t0-1}, a_{t0-1}        real, copied from the dataset
          s_{t0}                                   real, the rollout start
          a_hat_t ~ pi_phi(s_t), s_hat_{t+1} = f(s_t, a_hat_t, a'_t)
                                                   t = t0 .. t0+T-1, generated

so a row has ``t0 + T + 1`` observations, of which the first ``t0 + 1`` are
real. Stored as one trajectory, the relabeler can draw a future goal in the
generated segment for an anchor in the real prefix -- the real history is what
reached the split, and the model supplies what happens after it.

Episodes with no usable start -- a label other than detour/shortcut (the
dataset's 'none' timeouts), or a detour that never passes ``detour_min_x`` or
a shortcut that never enters the trap -- are SKIPPED: the source-episode draw
is uniform over the eligible episodes, which is exactly skip-and-redraw.

Everything after ``t0`` is v1's stepping loop, called on the start states, so
the expert head, the counterfactual coin, the agent's a', the feasibility clip
and the done head behave identically in both versions.
"""
import dataclasses
import os
from typing import Optional

import numpy as np

from crl import causal_transition_model as causal_mod
from crl import rockfall_clock_v6

ROUTES = ('detour', 'shortcut')
#: The V6 hazard zone a shortcut rollout starts in, and its (x0, x1, y0, y1).
DEFAULT_SHORTCUT_ZONE = 2
DEFAULT_TRAP_BOX = rockfall_clock_v6.hazard_zone(DEFAULT_SHORTCUT_ZONE)


def trap_box_for(zone):
  """(x0, x1, y0, y1) of V6 hazard ``zone`` (1 or 2)."""
  return rockfall_clock_v6.hazard_zone(int(zone))
DEFAULT_DETOUR_MIN_X = 20.0
DEFAULT_DETOUR_MAX_Y = 7.5


@dataclasses.dataclass
class RouteSplitRollout(causal_mod.RolloutTrajectories):
  """v1's ``RolloutTrajectories`` plus where each row's generated part begins.

  ``generated`` is False over the real prefix, so ``off_diagonal_frac`` and
  ``clipped_frac`` stay shares of GENERATED observations, as in v1.
  """
  start_step: Optional[np.ndarray] = None   # [n] t0: index of the real start.
  route: Optional[np.ndarray] = None        # [n] 'detour' / 'shortcut'.

  @property
  def mean_start_step(self):
    return (float(np.mean(self.start_step)) if self.start_step is not None
            else 0.0)

  def route_frac(self, route):
    return (float(np.mean(self.route == route)) if self.route is not None
            else 0.0)

  def generated_only(self):
    """v1-layout view: each row shifted to start at its t0, prefix dropped.

    v1's ``drift`` measures horizon h from index 0, so it is run on this view
    to measure h from the rollout start rather than from the reset.
    """
    L = int((self.lengths - self.start_step).max())

    def shift(a, fill=0):
      if a is None:
        return None
      out = np.full((a.shape[0], L) + a.shape[2:], fill, a.dtype)
      for b, k in enumerate(self.start_step):
        seg = a[b, k:k + L]
        out[b, :seg.shape[0]] = seg
      return out

    return causal_mod.RolloutTrajectories(
        episode=self.episode, obs=shift(self.obs), act=shift(self.act),
        lengths=self.lengths - self.start_step,
        generated=shift(self.generated, False), steps=self.steps,
        state_dim=self.state_dim, data_states=shift(self.data_states),
        data_act=shift(self.data_act), data_valid=shift(self.data_valid, False),
        components=shift(self.components, -1),
        off_diagonal_trained=self.off_diagonal_trained,
        off_diagonal=shift(self.off_diagonal, False),
        clipped=shift(self.clipped, False), terminated=self.terminated)

  def drift(self):
    return self.generated_only().drift()


def sidecar_path(offline_dataset):
  """``..._gxy.npz`` / ``....npz`` -> the ``..._sidecar.npz`` next to it."""
  stem = offline_dataset[:-4] if offline_dataset.endswith('.npz') \
      else offline_dataset
  if stem.endswith('_gxy'):
    stem = stem[:-4]
  return stem + '_sidecar.npz'


def load_route_labels(path, n_episodes):
  """Per-episode route labels from a dataset sidecar, or None if unavailable.

  Reads ``route_realized`` (the route the episode actually took), falling back
  to ``route`` (the one the teacher chose). Row e of the sidecar must be
  episode e of the dataset -- that is checked, because a misaligned label
  would send a rollout to start at the wrong route's split.
  """
  if not path or not os.path.exists(path):
    return None
  with np.load(path, allow_pickle=False) as d:
    key = ('route_realized' if 'route_realized' in d.files
           else 'route' if 'route' in d.files else None)
    if key is None:
      return None
    labels = np.asarray(d[key]).astype(str)
    if 'episode_id' in d.files and not np.array_equal(
        np.asarray(d['episode_id']), np.arange(labels.size)):
      raise ValueError(f'{path}: episode_id is not 0..{labels.size - 1}; '
                       'route labels cannot be matched to dataset episodes')
  if labels.size != int(n_episodes):
    raise ValueError(f'{path} labels {labels.size} episodes, the dataset has '
                     f'{n_episodes}')
  return labels


def _in_box(xy, box):
  x0, x1, y0, y1 = box
  return ((xy[:, 0] >= x0) & (xy[:, 0] <= x1)
          & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))


def geometric_route(xy, trap_box=DEFAULT_TRAP_BOX):
  """Label from the XY path alone: 'shortcut' if it ever enters the trap."""
  return 'shortcut' if bool(_in_box(xy, trap_box).any()) else 'detour'


def start_step(xy, route, detour_min_x=DEFAULT_DETOUR_MIN_X,
               trap_box=DEFAULT_TRAP_BOX, detour_max_y=DEFAULT_DETOUR_MAX_Y):
  """First index of ``xy`` meeting ``route``'s start condition, or -1."""
  if route == 'detour':
    hit = (xy[:, 0] > detour_min_x) & (xy[:, 1] < detour_max_y)
  elif route == 'shortcut':
    hit = _in_box(xy, trap_box)
  else:
    return -1
  idx = np.flatnonzero(hit)
  return int(idx[0]) if idx.size else -1


class CausalTransitionModelV2(causal_mod.CausalTransitionModel):
  """v1's model; ``rollout`` starts at the route split and keeps the prefix."""

  @classmethod
  def from_v1(cls, model):
    if not isinstance(model, causal_mod.CausalTransitionModel):
      raise TypeError(f'from_v1 needs a CausalTransitionModel, got '
                      f'{type(model).__name__} (a dynamics-only checkpoint '
                      'has no expert head to roll out with)')
    return cls(**{f.name: getattr(model, f.name)
                  for f in dataclasses.fields(model)})

  def rollout(self, states, actions, next_states, episode_ids, steps,
              goals=None, n=1, seed=0, episodes=None, pad_to=None,
              ground_truth=True, off_diagonal='none',
              off_diagonal_action='uniform', agent_action_fn=None,
              stop_on_done=True, routes=None,
              detour_min_x=DEFAULT_DETOUR_MIN_X, trap_box=DEFAULT_TRAP_BOX,
              detour_max_y=DEFAULT_DETOUR_MAX_Y):
    """T model steps from each source episode's route split, real prefix kept.

    Arguments are v1's (``CausalTransitionModel.rollout``) plus:
      routes: [E] per-episode labels indexed by episode id, e.g. from
        ``load_route_labels``. None labels each episode by geometry
        (``geometric_route``).
      detour_min_x, detour_max_y: a detour row starts at its first state
        with x > detour_min_x and y < detour_max_y.
      trap_box: (x0, x1, y0, y1); a shortcut row starts at its first state
        inside it.

    ``episodes``, when given, must all be eligible. ``ground_truth`` returns
    the source episode's own states on the row's time axis from s_0, so the
    generated segment is compared against what really followed s_{t0}.

    Returns:
      A ``RouteSplitRollout``.
    """
    T = int(steps)
    s_all = np.asarray(states, np.float32)
    a_all = np.asarray(actions, np.float32)
    sn_all = np.asarray(next_states, np.float32)
    eid = np.asarray(episode_ids)
    if eid.shape != (s_all.shape[0],):
      raise ValueError(f'episode_ids must be [{s_all.shape[0]}], got '
                       f'{eid.shape}')
    g_all = None if goals is None else np.asarray(goals, np.float32)

    ep_ids, first, count = np.unique(eid, return_index=True,
                                     return_counts=True)
    for e, f, c in zip(ep_ids, first, count):
      if not np.array_equal(eid[f:f + c], np.full(c, e)):
        raise ValueError(f'episode {e} is not a contiguous run of rows in '
                         'episode_ids; the real prefix needs it in time order')
    slot = {int(e): (int(f), int(c)) for e, f, c in zip(ep_ids, first, count)}

    def ep_states(e):
      f, c = slot[e]            # s_0 .. s_{c-1}, then the terminal s_c.
      return np.concatenate([s_all[f:f + c], sn_all[f + c - 1][None]])

    # -- route label and split step of every episode; ineligible ones drop out.
    route_of, t0_of = {}, {}
    for e in slot:
      xy = ep_states(e)[:, :2]
      if routes is None:
        r = geometric_route(xy, trap_box)
      else:
        if e >= len(routes):
          raise ValueError(f'episode {e} has no route label ({len(routes)} '
                           'given)')
        r = str(routes[e])
      t = start_step(xy, r, detour_min_x, trap_box, detour_max_y)
      if r in ROUTES and t >= 0:
        route_of[e], t0_of[e] = r, t
    eligible = np.array(sorted(t0_of), dtype=np.int64)
    if eligible.size == 0:
      raise ValueError('no episode has a route-split start: check the route '
                       f'labels, detour_min_x={detour_min_x}, detour_max_y='
                       f'{detour_max_y} and trap_box='
                       f'{trap_box}')

    rng = np.random.default_rng(seed)
    if episodes is None:
      ep = eligible[rng.integers(0, eligible.size, size=int(n))]
    else:
      ep = np.asarray(episodes, np.int64)
      bad = [int(e) for e in np.unique(ep) if int(e) not in t0_of]
      if bad:
        raise ValueError(f'episodes {bad[:10]} have no route-split start')
    rows = ep.size
    t0 = np.array([t0_of[int(e)] for e in ep], np.int64)
    route = np.array([route_of[int(e)] for e in ep])

    # -- v1's stepping loop from the start states. Each row is handed over as
    # its own one-transition "episode" whose initial state is s_{t0}, so v1's
    # draw, coin, clip and done head run unchanged; only the start differs.
    start_s = np.stack([ep_states(int(e))[k] for e, k in zip(ep, t0)])
    start_g = (None if g_all is None
               else np.stack([g_all[slot[int(e)][0]] for e in ep]))
    gen = super().rollout(
        start_s, np.zeros((rows, self.action_dim), np.float32), start_s,
        np.arange(rows), T, goals=start_g, episodes=np.arange(rows),
        seed=seed, ground_truth=False, off_diagonal=off_diagonal,
        off_diagonal_action=off_diagonal_action,
        agent_action_fn=agent_action_fn, stop_on_done=stop_on_done)

    # -- prepend the real prefix s_0..s_{t0-1}, a_0..a_{t0-1}.
    lengths = t0 + gen.lengths
    L = int(lengths.max())
    if pad_to is not None:
      if int(pad_to) < L:
        raise ValueError(f'pad_to={pad_to} is shorter than the rollouts in '
                         f'this batch ({L})')
      L = int(pad_to)
    obs = np.zeros((rows, L, gen.obs.shape[2]), np.float32)
    act = np.zeros((rows, L, self.action_dim), np.float32)
    gen_mask = np.zeros((rows, L), bool)
    comps = None if gen.components is None else np.full((rows, L), -1,
                                                         np.int64)
    off = None if gen.off_diagonal is None else np.zeros((rows, L), bool)
    clip = None if gen.clipped is None else np.zeros((rows, L), bool)
    d_states = (np.zeros((rows, L, self.state_dim), np.float32)
                if ground_truth else None)
    d_act = (np.zeros((rows, L, self.action_dim), np.float32)
             if ground_truth else None)
    d_valid = np.zeros((rows, L), bool) if ground_truth else None
    for b in range(rows):
      e, k, m = int(ep[b]), int(t0[b]), int(gen.lengths[b])
      f, c = slot[e]
      src = ep_states(e)
      obs[b, :k, :self.state_dim] = src[:k]
      if start_g is not None:
        obs[b, :k, self.state_dim:] = start_g[b]
      act[b, :k] = a_all[f:f + k]
      obs[b, k:k + m] = gen.obs[b, :m]
      act[b, k:k + m] = gen.act[b, :m]
      gen_mask[b, k:k + m] = gen.generated[b, :m]
      if comps is not None:
        comps[b, k:k + m] = gen.components[b, :m]
      if off is not None:
        off[b, k:k + m] = gen.off_diagonal[b, :m]
      if clip is not None:
        clip[b, k:k + m] = gen.clipped[b, :m]
      if ground_truth:
        nk, na = min(L, c + 1), min(L, c)
        d_states[b, :nk] = src[:nk]
        d_act[b, :na] = a_all[f:f + na]
        d_valid[b, :nk] = True

    return RouteSplitRollout(
        episode=ep, obs=obs, act=act, lengths=lengths, generated=gen_mask,
        steps=T, state_dim=self.state_dim,
        data_states=d_states, data_act=d_act, data_valid=d_valid,
        components=comps, off_diagonal_trained=gen.off_diagonal_trained,
        off_diagonal=off, clipped=clip, terminated=gen.terminated,
        start_step=t0, route=route)
