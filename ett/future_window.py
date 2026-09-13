"""Shared first-three-transition anchors with independent future-window RNG."""
import jax
import numpy as np
from crl.losses import Transition
from ett.policy_improvement import synthetic_count
from ett.rollout_return import GOAL


def future_index(anchor, horizon, uniform, gamma=.95):
    """Inverse CDF of weights gamma**(j-i), i < j <= horizon."""
    anchor, horizon, uniform = np.broadcast_arrays(anchor, horizon, uniform)
    if np.any(horizon <= anchor) or np.any((uniform < 0) | (uniform >= 1)):
        raise ValueError('empty future window or invalid uniform draw')
    count = horizon-anchor
    lag = np.ceil(np.log1p(-uniform*(1-gamma**count))/np.log(gamma)).astype(np.int64)
    return anchor + np.clip(lag, 1, count)


def synthetic_batch(states, actions, horizons, size, long_window, anchor_rng, future_rng):
    """Length does not change anchor weights; only achieved-goal indices differ."""
    episode = anchor_rng.integers(len(states), size=size)
    time = anchor_rng.integers(3, size=size)
    stop = horizons[episode] if long_window else np.full(size, 3)
    future = future_index(time, stop, future_rng.random(size))
    assert np.all(horizons >= 3) and np.all(future <= horizons[episode])
    state, successor = states[episode, time], states[episode, time+1]
    goal, action = states[episode, future], actions[episode, time]
    assert np.isfinite(state).all() and np.isfinite(goal).all() and np.isfinite(action).all()
    np.testing.assert_array_equal(successor[:, 2:], state[:, :6])
    batch = Transition(np.concatenate([state, goal], -1), action,
        np.full(size, np.nan, np.float32), np.full(size, np.nan, np.float32),
        np.concatenate([successor, goal], -1), np.full((size, 2), np.nan, np.float32))
    return batch, dict(episode=episode, time=time, future=future, horizon=horizons[episode],
        goal_xy_distance=np.linalg.norm(goal[:, :2]-GOAL[:2], axis=-1),
        goal_f4_distance=np.linalg.norm(goal-GOAL, axis=-1))


def batch_stream(offline, shared, arm, seed, rng_bases):
    anchor_rng = np.random.default_rng(rng_bases['anchor']+seed)
    goal_rng = np.random.default_rng(rng_bases['future']+seed)
    permutation_rng = np.random.default_rng(rng_bases['permutation']+seed)
    for update in range(2000):
        batch, off_index = offline.sample_audited(256)
        count = 0 if arm == 'A' else synthetic_count(update)
        index = {}
        if count:
            block = update//100
            generated, index = synthetic_batch(shared['states'][block], shared['action'][block],
                shared['horizon'][block], count, arm == 'C', anchor_rng, goal_rng)
            batch = jax.tree.map(lambda a, b: np.concatenate([a[:-count], b]), batch, generated)
        source = np.zeros(256, bool)
        if count:
            source[-count:] = True
        permutation = permutation_rng.permutation(256)
        batch = jax.tree.map(lambda x: x[permutation], batch)
        audit = dict(permutation=permutation, source=source[permutation], offline=off_index, count=count)
        audit.update(index)
        yield batch, audit


def digest_batch(digest, batch, audit):
    """Hash everything required to be identical except future goals themselves."""
    values = [batch.observation[:, :8], batch.action, batch.next_observation[:, :8],
              audit['source'], audit['permutation']]
    values += [audit['offline'][k] for k in ['episode', 'time', 'future']]
    values += [audit[k] for k in ['episode', 'time'] if k in audit]
    for value in values:
        digest.update(np.ascontiguousarray(value).tobytes())


def compact_audit(audit):
    out = dict(count=np.array(audit['count']), permutation=audit['permutation'].astype(np.int16),
               source=audit['source'])
    for field in ['episode', 'time', 'future', 'horizon', 'goal_xy_distance', 'goal_f4_distance']:
        dtype = np.float32 if 'distance' in field else np.int16
        value = np.full(26, -1, dtype=dtype)
        if field in audit:
            value[:audit['count']] = audit[field]
        out[field] = value
    return out
