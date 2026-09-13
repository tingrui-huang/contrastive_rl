"""Fixed visible-XY controllers and frozen ETT rollouts; no learning interface."""
import jax
import jax.numpy as jnp
import numpy as np

from ett.rollout_return import GOAL, task_reward

ROUTES = {
    'shortcut': np.array([[1.5, 3.5], [7.5, 3.5], [8.5, 3.5]], np.float32),
    'lower': np.array([[1.5, 3.5], [1.5, 1.5], [7.5, 1.5],
                       [7.5, 3.5], [8.5, 3.5]], np.float32),
}


@jax.jit
def controller(xy, index, waypoints):
    """Only current visible XY, progress index, and fixed public waypoints.

    At most one advance per observation; distinct waypoints are > .5 apart,
    so a second advance could not be triggered on the same observation.
    The final waypoint is retained, including after reaching it.
    """
    reached = jnp.linalg.norm(waypoints[index] - xy) <= .25
    index = jnp.minimum(index + reached.astype(jnp.int32), len(waypoints) - 1)
    return jnp.clip(waypoints[index] - xy, -1., 1.), index


def make_rollout(model, nominal):
    """One episode per compiled call keeps RNG and numerical batch shape fixed."""
    @jax.jit
    def rollout(theta, initial, waypoints, key):
        goal = jnp.asarray(GOAL)[None]

        def step(carry, key):
            state, index = carry
            bk, tk = jax.random.split(key)
            xp = nominal.sample(state[None], bk, 1, goal=goal)
            action, index = controller(state[:2], index, waypoints)
            y, detail = model.sample_flat(theta, state[None], action[None], xp,
                                          goal, tk, 1)
            following = y[0, 0]
            record = dict(state=following, action=action, waypoint=index,
                          reward=task_reward(following, goal[0]), x_prime=xp[0])
            # Preserve every available projection diagnostic, including base.
            record.update({name: value[0] for name, value in detail.items()})
            return (following, index), record

        _, record = jax.lax.scan(step, (initial, jnp.int32(0)),
                                 jax.random.split(key, 50))
        record['states'] = jnp.concatenate([initial[None], record.pop('state')])
        record['goals'] = jnp.broadcast_to(goal, (51, 8))
        return record
    return rollout


def passage_crossed(xy, lower):
    """Ordered left-to-right traversal through each central passage cell.

    Require visits to x cells 2,3,4,5,6 in order, staying in the same open
    horizontal passage from first cell 2 through first cell 6. Merely dipping
    below y=2, or an isolated teleport to the exit, does not pass this test.
    This is macro-observation adherence, not substep reachability certification.
    """
    lo, hi = (1., 2.) if lower else (3., 4.)
    for start in np.flatnonzero((xy[:, 0] >= 2) & (xy[:, 0] < 3) &
                               (xy[:, 1] >= lo) & (xy[:, 1] < hi)):
        next_cell = 3
        for pos in xy[start + 1:]:
            if not lo <= pos[1] < hi:
                break
            if next_cell <= pos[0] < next_cell + 1:
                next_cell += 1
                if next_cell == 7:
                    return True
    return False


def validate(record, route):
    """Independent NumPy reconstruction; no simulator or model calls."""
    states, actions = record['states'], record['action']
    assert states.shape[1:] == (51, 8) and actions.shape[1:] == (50, 2)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    assert np.abs(actions).max() <= 1
    np.testing.assert_array_equal(states[:, 1:, 2:], states[:, :-1, :6])
    np.testing.assert_array_equal(record['goals'], np.broadcast_to(GOAL, record['goals'].shape))
    reward = (np.linalg.norm(states[:, 1:, :2].astype(float) - GOAL[:2], axis=-1) < 2)
    np.testing.assert_array_equal(record['reward'], reward)
    points = ROUTES[route]
    completions = np.zeros((len(states), len(points)), bool)
    for e, episode in enumerate(states):
        index = 0
        for t, state in enumerate(episode):
            if np.linalg.norm(points[index] - state[:2]) <= .25:
                completions[e, index] = True
                index = min(index + 1, len(points) - 1)
            if t < 50:
                assert record['waypoint'][e, t] == index
                np.testing.assert_array_equal(actions[e, t], np.clip(points[index] - state[:2], -1, 1))
    returns = record['reward'].astype(float) @ (.95 ** np.arange(50))
    np.testing.assert_allclose(record['return'], returns, rtol=0, atol=1e-12)
    return dict(returns=returns,
                success=np.min(np.linalg.norm(states[..., :2].astype(float) - GOAL[:2], axis=-1), axis=1) < .5,
                lower_passage=np.array([passage_crossed(s[:, :2], True) for s in states]),
                shortcut_passage=np.array([passage_crossed(s[:, :2], False) for s in states]),
                waypoint_completed=completions,
                route_complete=completions.all(1))
