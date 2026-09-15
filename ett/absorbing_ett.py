"""Absorbing-freeze ETT rollout backend for the PointMaze F4 pipeline.

This is the G0 prescreen arm ``manski_absorbing`` packaged as a drop-in
replacement for ``ett.pointmaze_region_pilot.Kernel.rollout``.  The transition
``F(s, x, x')`` has three visible components:

* alive XY motion: the eligible diagonal law queried at the executed action,
  ``y ~ P_diag(. | s, x, x)``, including its stationary atom.  This assumes the
  hidden confounder acts only through freeze, not through alive motion.
* absorbing persistence: once a path is frozen its XY never changes again
  while the F4 history keeps shifting and reward is zero.  A raw atom draw
  whose cell lies in the absorbing support ``S_abs`` freezes the path.
* Manski onset: if the landing bin of the executed action ``x`` differs from
  the landing bin of the nominal draw ``x'`` and the sampled landing cell lies
  in ``S_abs``, the path freezes from the next step (the fatal incoming
  movement is retained).  Averaged over ``x' ~ nominal`` this is the per-step
  Manski upper bound ``pi_B q_obs + (1 - pi_B) 1[landing in S_abs]``.

``S_abs`` is discovered from ``obs``/``act`` alone: landing cells of freeze
onsets (a moving transition followed by an exactly stationary one) whose
stationary run continues to the episode end.  No hidden bits, death labels,
hazard labels or environment internals are read anywhere in this module.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import _project_samples
from ett.rollout_return import GOAL, task_reward

FIELDS_READ = ('obs', 'act')
GRID = (9, 5)
MODES = ('eligible_response', 'diagonal_motion', 'diagonal_motion_absorbing',
         'manski_absorbing')
SUPPORT_MIN_ONSETS = 10
SUPPORT_MIN_ABSORBING = .5
STATIONARY_TOLERANCE = 1e-7


def landing_cell(xy):
    """Visible landing bin: floor of the box-clipped position, box bounds only."""
    clipped = np.clip(np.asarray(xy, np.float64), [0., 0.],
                      [GRID[0] - 1e-6, GRID[1] - 1e-6])
    return np.floor(clipped).astype(np.int64)


def freeze_tables(obs, act, episode_ids, min_onsets=SUPPORT_MIN_ONSETS,
                  min_absorbing=SUPPORT_MIN_ABSORBING):
    """Absorbing support, onset table and persistence from visible data only.

    ``act`` is accepted for interface symmetry and read only for its shape; the
    tables are functions of the recorded positions.
    """
    episode_ids = np.asarray(episode_ids)
    assert act.shape[:2] == obs.shape[:2]
    xy = obs[episode_ids, :, :2].astype(np.float64)
    cur, nxt = xy[:, :-1], xy[:, 1:]
    stationary = np.linalg.norm(nxt - cur, axis=-1) <= STATIONARY_TOLERANCE
    E, T = stationary.shape
    onset = np.zeros_like(stationary)
    onset[:, :-1] = (~stationary[:, :-1]) & stationary[:, 1:]
    run_to_end = np.zeros_like(stationary)
    tail = np.ones(E, bool)
    for t in range(T - 1, -1, -1):
        tail = tail & stationary[:, t]
        run_to_end[:, t] = tail
    counts = np.zeros(GRID, np.int64)
    absorbing = np.zeros(GRID, np.int64)
    cell = landing_cell(nxt)
    for e, t in zip(*np.nonzero(onset)):
        c = tuple(cell[e, t])
        counts[c] += 1
        absorbing[c] += int(run_to_end[e, t + 1])
    fraction = np.where(counts > 0, absorbing / np.maximum(counts, 1), np.nan)
    support = (counts >= min_onsets) & (np.nan_to_num(fraction) >= min_absorbing)
    frames = obs[episode_ids, :-1, :8].reshape(E, T, 4, 2)
    signature = ((np.abs(frames - frames[:, :, :1]).max(axis=(2, 3)) == 0.)
                 & (np.arange(T)[None] >= 3))
    ccell = landing_cell(cur)
    persistence = {}
    for c in zip(*np.nonzero(counts > 0)):
        m = signature & (ccell[..., 0] == c[0]) & (ccell[..., 1] == c[1])
        persistence[str((int(c[0]), int(c[1])))] = {
            'rows': int(m.sum()),
            'stay_stationary': float(stationary[m].mean()) if m.any() else None}
    onset_table = {str((int(i), int(j))): {
        'onsets': int(counts[i, j]), 'absorbing': int(absorbing[i, j]),
        'absorbing_fraction': float(fraction[i, j]), 'in_support': bool(support[i, j])}
        for i, j in zip(*np.nonzero(counts > 0))}
    return {
        'support': support,
        'support_cells': [[int(i), int(j)] for i, j in zip(*np.nonzero(support))],
        'onset_table': onset_table,
        'persistence_given_signature': persistence,
        'totals': {'episodes': int(E), 'transitions': int(E * T),
                   'stationary': int(stationary.sum()), 'onsets': int(onset.sum()),
                   'absorbing_onsets': int(absorbing.sum())},
        'rule': {'min_onsets': int(min_onsets), 'min_absorbing_fraction': float(min_absorbing),
                 'onset': 'moving transition followed by exactly stationary transition',
                 'absorbing': 'stationary run continues to the episode end',
                 'bin': 'floor(clip(xy)) on the 9x5 grid'},
        'fields_read': list(FIELDS_READ),
    }


def manski_tables(obs, act, episode_ids, support, cells=((2, 3), (1, 3), (3, 3)),
                  min_rows=20):
    """Descriptive per-step Manski upper bounds by landing bin; not a model input."""
    episode_ids = np.asarray(episode_ids)
    xy = obs[episode_ids, :, :2].astype(np.float64)
    cur, nxt = xy[:, :-1], xy[:, 1:]
    stationary = np.linalg.norm(nxt - cur, axis=-1) <= STATIONARY_TOLERANCE
    freeze_next = np.zeros_like(stationary)
    freeze_next[:, :-1] = stationary[:, 1:]
    alive = ~stationary
    ccell, ncell = landing_cell(cur), landing_cell(nxt)
    out = {}
    for c in cells:
        sel = alive & (ccell[..., 0] == c[0]) & (ccell[..., 1] == c[1])
        n = int(sel.sum())
        keys, inverse, count = np.unique(ncell[sel], axis=0, return_inverse=True,
                                         return_counts=True)
        bins = {}
        for k, (key, m) in enumerate(zip(keys, count)):
            if m < min_rows:
                continue
            pi = m / n
            q = float(freeze_next[sel][inverse.ravel() == k].mean())
            ins = bool(support[key[0], key[1]])
            bins[str((int(key[0]), int(key[1])))] = {
                'pi': float(pi), 'q_obs': q, 'in_support': ins,
                'manski_upper': float(pi * q + (1. - pi)) if ins else q}
        out[str((int(c[0]), int(c[1])))] = {'n': n, 'bins': bins}
    return out


class AbsorbingRollout:
    """Rollout backend with the ``Kernel.rollout`` interface and freeze extras.

    ``mode='eligible_response'`` delegates to the wrapped ``Kernel`` unchanged.
    The other modes ignore ``theta`` (they must receive zeros) and never touch
    the wrapped diagonal or nominal parameters.
    """

    def __init__(self, engine, support, mode='manski_absorbing'):
        if mode not in MODES:
            raise ValueError(mode)
        self.engine = engine
        self.base = engine.base
        self.nominal = engine.nominal
        self.actor = engine.actor
        self.actor_jit = engine.actor_jit
        self.mode = mode
        self.support = np.asarray(support, bool)
        assert self.support.shape == GRID
        self._support = jnp.asarray(self.support)
        self._fns = {}
        self._sample_fn = None

    # -- one-step diagonal-compatible sampler ---------------------------------
    def sample(self, theta, s, a, xp, key, count):
        """One-step law at the executed action ``a``; ``xp`` only enters agreement."""
        if self.mode == 'eligible_response':
            return self.engine.sample(theta, s, a, xp, key, count)
        self._check_theta(theta)
        if self._sample_fn is None:
            base, spec, params = self.base, self.base.spec, self.base.params

            def fn(s, a, key, count):
                goal = jnp.broadcast_to(jnp.asarray(GOAL), s.shape)
                context = jnp.concatenate([s, a, a, goal], -1)
                delta, atom = base._sample_delta(params, context, key, count)
                y, details = _project_samples(s, delta, spec)
                return y, {'stationary_atom': atom,
                           'box_valid': jnp.ones(atom.shape, bool),
                           'projection_corrected': details['blocked_endpoint_before_projection'],
                           'box_low': jnp.broadcast_to(s[:, None, :2] - 1., y[..., :2].shape),
                           'box_high': jnp.broadcast_to(s[:, None, :2] + 1., y[..., :2].shape)}
            self._sample_fn = jax.jit(fn, static_argnums=(3,))
        return self._sample_fn(jnp.asarray(s), jnp.asarray(a), key, int(count))

    def _check_theta(self, theta):
        theta = np.asarray(theta)
        if theta.size and np.any(theta != 0):
            raise ValueError(f'{self.mode} ignores ETT offsets; pass zeros')

    # -- multi-step rollout ------------------------------------------------------
    def rollout(self, theta, states, key, horizon=10, first_actions=None):
        states = np.asarray(states, np.float32)
        n = len(states)
        if self.mode == 'eligible_response':
            out = self.engine.rollout(theta, states, key, horizon, first_actions)
            h = out['reward'].shape[1]
            zeros = np.zeros((n, h), bool)
            out.update(frozen=np.zeros((n, h + 1), bool), onset_manski=zeros.copy(),
                       onset_atom=zeros.copy(), agree=zeros.copy(), hazard_alive=zeros.copy())
            return out
        self._check_theta(theta)
        sig = int(horizon)
        if sig not in self._fns:
            self._fns[sig] = jax.jit(self._make(sig))
        first = (np.zeros((n, 2), np.float32) if first_actions is None
                 else np.asarray(first_actions, np.float32))
        assert first.shape == (n, 2)
        out = self._fns[sig](jnp.asarray(states), jnp.asarray(first),
                             first_actions is not None, key)
        out = jax.tree.map(np.asarray, out)
        assert out['valid'].all() and np.isfinite(out['states']).all()
        np.testing.assert_array_equal(out['states'][:, 1:, 2:], out['states'][:, :-1, :6])
        if first_actions is not None:
            np.testing.assert_array_equal(out['action'][:, 0], first)
        assert np.abs(out['action']).max() <= 1 and np.abs(out['x_prime']).max() <= 1
        # frozen states never move again
        moved = np.linalg.norm(np.diff(out['states'][:, :, :2], axis=1), axis=-1) > 0
        assert not np.any(moved & out['frozen'][:, :-1])
        assert not np.any(out['reward'] * out['frozen'][:, :-1])
        return out

    def _make(self, horizon):
        base, nominal, actor = self.base, self.nominal, self.actor
        spec, params, support = base.spec, base.params, self._support
        use_absorb = self.mode in ('diagonal_motion_absorbing', 'manski_absorbing')
        use_manski = self.mode == 'manski_absorbing'
        lo = jnp.array([0., 0.])
        hi = jnp.array([GRID[0] - 1e-6, GRID[1] - 1e-6])

        def cell_of(xy):
            return jnp.floor(jnp.clip(xy, lo, hi)).astype(jnp.int32)

        def generate(states, first, override, key):
            def step(carry, data):
                s, frozen = carry
                t, key = data
                bk, ak, tk = jax.random.split(key, 3)
                goal = jnp.broadcast_to(jnp.asarray(GOAL), s.shape)
                xp = nominal.sample(s, bk, 1, goal=goal)
                a = actor(s, goal, ak)
                a = jnp.where((t == 0) & override, first, a)
                context = jnp.concatenate([s, a, a, goal], -1)
                delta, atom = base._sample_delta(params, context, tk, 1)
                y, details = _project_samples(s, delta, spec)
                y, atom = y[:, 0], atom[:, 0]
                projected = details['blocked_endpoint_before_projection'][:, 0]
                y = jnp.where(frozen[:, None], jnp.concatenate([s[:, :2], s[:, :6]], -1), y)
                land = cell_of(y[:, :2])
                ins = support[land[:, 0], land[:, 1]]
                agree = jnp.all(cell_of(s[:, :2] + a) == cell_of(s[:, :2] + xp), axis=-1)
                onset_atom = use_absorb & (~frozen) & atom & ins
                onset_manski = use_manski & (~frozen) & (~atom) & (~agree) & ins
                frozen_next = frozen | onset_atom | onset_manski
                reward = task_reward(y, goal) * (1. - frozen.astype(jnp.float32))
                record = dict(next=y, action=a, x_prime=xp, reward=reward,
                              valid=jnp.ones(s.shape[0], bool), atom=atom,
                              projected=projected, frozen_next=frozen_next,
                              onset_manski=onset_manski, onset_atom=onset_atom,
                              agree=agree, hazard_alive=(~frozen) & ins)
                return (y, frozen_next), record

            frozen0 = jnp.zeros(states.shape[0], bool)
            _, data = jax.lax.scan(step, (states, frozen0),
                                   (jnp.arange(horizon), jax.random.split(key, horizon)))
            data = jax.tree.map(lambda v: jnp.swapaxes(v, 0, 1), data)
            data['states'] = jnp.concatenate([states[:, None], data.pop('next')], 1)
            data['frozen'] = jnp.concatenate([frozen0[:, None], data.pop('frozen_next')], 1)
            return data
        return generate
