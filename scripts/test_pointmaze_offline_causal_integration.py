"""Synthetic contract tests for the strictly offline PointMaze integration."""
from pathlib import Path

import numpy as np

from ett import pointmaze_offline_causal_integration as integration


def synthetic_offline():
    obs = np.zeros((3, 51, 16), np.float32)
    act = np.zeros((3, 51, 2), np.float32)
    for e in range(3):
        obs[e, 0, :8] = np.arange(8, dtype=np.float32) + e
        for t in range(50):
            obs[e, t + 1, :2] = obs[e, t, :2] + np.array([.01, -.02], np.float32)
            obs[e, t + 1, 2:8] = obs[e, t, :6]
            act[e, t] = np.array([.2 + .1 * e, -.4], np.float32)
        obs[e, :, 8:] = integration.GOAL
    return obs, act


class FakeKernel:
    def rollout(self, theta, roots, key, horizon, first_actions):
        del theta, key
        n = len(roots)
        states = np.empty((n, horizon + 1, 8), np.float32)
        states[:, 0] = roots
        actions = np.zeros((n, horizon, 2), np.float32)
        actions[:, 0] = first_actions
        for t in range(horizon):
            states[:, t + 1, :2] = states[:, t, :2] + .01 * actions[:, t]
            states[:, t + 1, 2:] = states[:, t, :6]
        return {"states": states, "action": actions,
                "x_prime": np.zeros((n, horizon, 2), np.float32),
                "reward": np.zeros((n, horizon), np.float32),
                "atom": np.zeros((n, horizon), bool),
                "projected": np.zeros((n, horizon), bool)}


class FakeLedger:
    def __init__(self):
        self.charged = 0

    def add(self, number, purpose):
        del purpose
        self.charged += int(number)


def test_discounted_offsets_are_strictly_later_and_within_remaining_horizon():
    length = np.repeat(np.arange(1, 51, dtype=np.int32), 50)
    offset = integration.draw_offsets(np.random.default_rng(2), length)
    assert np.all(offset >= 1)
    assert np.all(offset <= length)
    assert np.all(offset[length == 1] == 1)


def test_matched_rows_change_only_positive_future_source():
    obs, act = synthetic_offline()
    plan = {"train_pool_episode": np.array([0, 1], np.int32),
            "train_pool_time": np.array([2, 3], np.int32)}
    cache_states = np.full((2, 51, 8), np.nan, np.float32)
    cache_states[0, :49] = 7.; cache_states[1, :48] = 9.
    cache = {"states": cache_states, "length": np.array([48, 47], np.int32)}
    ids, offsets = np.array([0, 1]), np.array([4, 5])
    o_batch, o_goal = integration.rows_from_pool(
        obs, act, plan, cache, "train_pool", ids, offsets, "O")
    p_batch, p_goal = integration.rows_from_pool(
        obs, act, plan, cache, "train_pool", ids, offsets, "P")
    np.testing.assert_array_equal(o_batch.observation[:, :8], p_batch.observation[:, :8])
    np.testing.assert_array_equal(o_batch.action, p_batch.action)
    np.testing.assert_array_equal(o_batch.next_observation[:, :8],
                                  p_batch.next_observation[:, :8])
    np.testing.assert_array_equal(o_batch.observation[:, 8:], o_goal)
    np.testing.assert_array_equal(p_batch.observation[:, 8:], p_goal)
    assert np.any(o_goal != p_goal)


def test_cache_preserves_recorded_first_action_f4_and_padding():
    obs, act = synthetic_offline()
    episode = np.array([0, 1, 2], np.int32)
    anchor_time = np.array([49, 47, 45], np.int32)
    ledger = FakeLedger()
    cache = integration.generate_cache(
        FakeKernel(), np.zeros(48, np.float32), obs, act, episode, anchor_time,
        123, ledger, "synthetic")
    np.testing.assert_array_equal(cache["action"][:, 0], act[episode, anchor_time])
    for i, horizon in enumerate(cache["length"]):
        np.testing.assert_array_equal(cache["states"][i, 1:horizon + 1, 2:],
                                      cache["states"][i, :horizon, :6])
        assert np.isfinite(cache["states"][i, :horizon + 1]).all()
        assert np.isnan(cache["states"][i, horizon + 1:]).all()
        assert np.array_equal(cache["valid"][i], np.arange(50) < horizon)


def test_action_strata_are_mutually_exclusive():
    action = np.array([[0., -1.], [1., 0.], [.8, -.8], [0., 0.]], np.float32)
    down, right = integration.direction_masks(action)
    assert not np.any(down & right)
    np.testing.assert_array_equal(down, [True, False, True, False])
    np.testing.assert_array_equal(right, [False, True, False, False])


def test_driver_has_no_environment_or_oracle_import():
    modules = integration.imported_modules(Path(integration.__file__))
    forbidden = ("gym", "gymnasium", "dm_control", "native_alive_motion",
                 "true_transition", "oracle")
    assert not any(any(token in module.lower() for token in forbidden)
                   for module in modules)
    assert integration.CONFIG["environment_calls"] == 0
    assert integration.CONFIG["native_steps"] == 0
