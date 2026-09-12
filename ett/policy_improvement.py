"""Frozen-model rollout integration for the existing Monte Carlo CRL learner."""
import hashlib
import jax
import jax.numpy as jnp
import numpy as np

from crl.losses import Transition
from crl.replay import TrajectoryBuffer
from ett.eval_diagonal_transition import _open_endpoint
from ett.rollout_return import GOAL, task_reward


def tree_sha(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.ascontiguousarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class SegmentReplay(TrajectoryBuffer):
    """Reuse the original geometric future sampler on actual segment lengths.

    Unused MC fields carry NaN sentinels, not fabricated rewards, terminal
    labels or an action beyond the segment boundary. TD is prohibited.
    """
    def sample_audited(self, size):
        episode, time, future = self.sampled_indices(size)
        state = self._obs[episode, time, :8]
        goal = self._obs[episode, future, :8]
        following = self._obs[episode, time + 1, :8]
        action = self._act[episode, time]
        assert np.all(future > time) and np.all(future < self._L)
        batch = Transition(np.concatenate([state, goal], -1), action,
                           np.full(size, np.nan, np.float32),
                           np.full(size, np.nan, np.float32),
                           np.concatenate([following, goal], -1),
                           np.full((size, 2), np.nan, np.float32))
        assert batch.observation.shape == (size, 16)
        assert np.isfinite(batch.observation).all() and np.isfinite(action).all()
        np.testing.assert_array_equal(batch.next_observation[:, 2:8], state[:, :6])
        return batch, dict(episode=episode, time=time, future=future)


def replay(observation, action, seed):
    n, length, width = observation.shape
    assert width == 16 and action.shape == (n, length - 1, 2)
    np.testing.assert_array_equal(observation[:, 1:, 2:8], observation[:, :-1, :6])
    buffer = SegmentReplay(n * length, length, 16, 2, 8, 0, -1, .95, seed)
    for obs, act in zip(observation, action):
        buffer.add_episode(obs, np.concatenate([act, np.full((1, 2), np.nan, np.float32)]))
    buffer.freeze()
    return buffer


def synthetic_count(update, batch_size=256):
    """Exactly 10 percent cumulatively every five batches, without rounding bias."""
    return ((update + 1) * batch_size) // 10 - (update * batch_size) // 10


def mixed_batch(offline, synthetic, update, rng):
    # Draw the same full offline batch in every arm, then replace its last n rows.
    original, offline_index = offline.sample_audited(256)
    n = synthetic_count(update) if synthetic is not None else 0
    synthetic_index = None
    if n:
        generated, synthetic_index = synthetic.sample_audited(n)
        original = jax.tree.map(lambda a, b: np.concatenate([a[:-n], b]), original, generated)
    order = rng.permutation(256)
    source = np.zeros(256, bool)
    if n:
        source[-n:] = True
    return jax.tree.map(lambda x: x[order], original), dict(
        synthetic_count=n, source=source[order], offline=offline_index,
        synthetic=synthetic_index, permutation=order)


def make_generator(model, nominal, networks, horizon, deterministic=False):
    """Actor parameters are explicit; model and nominal never receive gradients."""
    @jax.jit
    def generate(params, theta, initial, key):
        goal = jnp.broadcast_to(jnp.asarray(GOAL), initial.shape)
        def step(state, key):
            bk, ak, tk = jax.random.split(key, 3)
            xp = nominal.sample(state, bk, 1, goal=goal)
            distribution = networks.policy_network.apply(params, jnp.concatenate([state, goal], -1))
            action = networks.sample_eval(distribution, None) if deterministic else networks.sample(distribution, ak)
            successor, detail = model.sample_flat(theta, state, action, xp, goal, tk, 1)
            following = successor[:, 0]
            return following, dict(state=following, action=action, aux_x_prime=xp,
                reward=task_reward(following, goal), anchor=detail['anchor_xy'][:, 0],
                projection=detail['projection_corrected'][:, 0],
                boundary=detail['projected_to_boundary'][:, 0], valid=detail['box_valid'][:, 0],
                atom=detail['stationary_atom'][:, 0])
        _, record = jax.lax.scan(step, initial, jax.random.split(key, horizon))
        record = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), record)
        record['states'] = jnp.concatenate([initial[:, None], record.pop('state')], 1)
        return record
    return generate


def motion_diagnostics(record):
    states = np.asarray(record['states'])
    before, after = states[:, :-1], states[:, 1:]
    np.testing.assert_array_equal(after[..., 2:], before[..., :6])
    assert np.isfinite(states).all() and np.asarray(record['valid']).all()
    assert _open_endpoint(after[..., :2]).all()
    assert np.max(np.abs(after[..., :2] - before[..., :2])) <= 1 + 2e-6
    for name in ['action', 'aux_x_prime']:
        assert np.isfinite(record[name]).all() and np.max(np.abs(record[name])) <= 1
    delta = after[..., :2] - before[..., :2]
    speed = np.linalg.norm(delta, axis=-1)
    stationary = np.all(before.reshape(*before.shape[:2], 4, 2) == before[..., None, :2], axis=(-1, -2))
    segment = before[..., None, :2] + np.linspace(0, 1, 21)[None, None, :, None] * delta[..., None, :]
    xy = after[..., :2]
    return dict(mean_motion=float(speed.mean()), stationary_fraction=float(np.mean(speed == 0)),
        stationary_history_count=int(stationary.sum()),
        renewed_stationary_fraction=float(np.mean(speed[stationary] > 1e-5)) if stationary.any() else None,
        projection_fraction=float(np.mean(record['projection'])),
        boundary_fraction=float(np.mean(record['boundary'])),
        atom_moved_fraction=float(np.mean(np.linalg.norm(xy - record['anchor'], axis=-1)[record['atom']] > 1e-5)) if np.any(record['atom']) else None,
        sampled_segment_wall_fraction=float(np.mean(~_open_endpoint(segment).all(-1))),
        mean_xy=xy.mean((0, 1)), detour_fraction=float(np.mean(xy[..., 1] < 2)),
        corridor_fraction=float(np.mean((xy[..., 0] >= 3) & (xy[..., 0] < 6) & (xy[..., 1] >= 3))),
        action_saturation=float(np.mean(np.abs(record['action']) > .99)))
