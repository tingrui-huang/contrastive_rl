"""Exact alive-motion map and native equivalence checks for PointMaze."""
from __future__ import annotations

import numpy as np


WALLS = np.array(
    [
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
    ],
    dtype=np.int64,
)
LOW = np.array([0.0, 0.0], dtype=np.float64)
HIGH = np.array(WALLS.shape, dtype=np.float64)
GOAL_XY = np.array([8.5, 3.5], dtype=np.float64)


def discretize(physical_xy):
    ij = np.floor(np.asarray(physical_xy, np.float64)).astype(np.int64)
    return np.clip(ij, np.array([0, 0]), np.array(WALLS.shape) - 1)


def blocked(physical_xy):
    physical_xy = np.asarray(physical_xy, np.float64)
    outside = np.any((physical_xy < LOW) | (physical_xy > HIGH), axis=-1)
    ij = discretize(physical_xy)
    wall = WALLS[ij[..., 0], ij[..., 1]] == 1
    return outside | wall


def hazardous(physical_or_emitted_xy):
    xy = np.asarray(physical_or_emitted_xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


def native_alive_motion(physical_xy, queried_action, motion_noise):
    """Apply only TwoRouteSwampWindyEnv's alive movement, with explicit noise.

    Inputs and persistent physical state use float64. ``motion_noise`` is the
    already-scaled N(0, 0.01) actuator perturbation; no RNG is consulted here.
    """
    state = np.asarray(physical_xy, np.float64).copy()
    action = np.asarray(queried_action, np.float64).copy()
    action += np.asarray(motion_noise, np.float64)
    action = np.clip(action, -1.0, 1.0)
    blocked_updates = np.zeros(state.shape[:-1] + (2,), dtype=np.int64)
    for _ in range(10):
        for axis in (0, 1):
            proposed = state.copy()
            proposed[..., axis] += 0.1 * action[..., axis]
            reject = blocked(proposed)
            blocked_updates[..., axis] += reject
            state = np.where(reject[..., None], state, proposed)
    return state, action, blocked_updates


def emit_f4(next_physical_xy, current_f4):
    next_xy = np.asarray(next_physical_xy, np.float64).astype(np.float32)
    current_f4 = np.asarray(current_f4, np.float32)
    return np.concatenate([next_xy, current_f4[..., :6]], axis=-1).astype(np.float32)


class FixedNoiseRNG:
    """Minimal RNG shim that injects one declared native action-noise draw."""

    def __init__(self, noise):
        self.noise = np.asarray(noise, np.float64).copy()
        self.normal_calls = 0

    def normal(self, loc, scale, size):
        if loc != 0 or scale != 0.01 or tuple(size) != (2,):
            raise AssertionError("unexpected native normal request")
        self.normal_calls += 1
        return self.noise.copy()

    def random(self, *args, **kwargs):
        raise AssertionError("hidden-bit resampling must be disabled in movement verification")


def _set_native_state(env, physical_xy, f4):
    physical_xy = np.asarray(physical_xy, np.float64)
    f4 = np.asarray(f4, np.float32).reshape(4, 2)
    np.testing.assert_array_equal(f4[0], physical_xy.astype(np.float32))
    env.state = physical_xy.copy()
    env._frames = [np.asarray(frame, np.float64).copy() for frame in f4]
    env._dead = False
    env.set_auto_resample(False)


def _native_step(env_class, physical_xy, f4, action, noise, bits):
    env = env_class(action_noise=0.01, max_episode_steps=50, seed=9917, active_prob=0.3)
    _set_native_state(env, physical_xy, f4)
    env.set_swamp(bits)
    fixed_rng = FixedNoiseRNG(noise)
    env._rng = fixed_rng
    obs, reward, done, info = env.step(action)
    if fixed_rng.normal_calls != 1 or done or info:
        raise AssertionError("unexpected native step contract")
    return {
        "physical_xy": np.asarray(env.state, np.float64).copy(),
        "f4": np.asarray(obs[:8], np.float32).copy(),
        "dead": bool(env.dead),
        "reward": float(reward),
    }


def _history(physical_xy):
    current = np.asarray(physical_xy, np.float64).astype(np.float32)
    return np.tile(current, 4).astype(np.float32)


def verify_against_native(env_class, noise_seed):
    """Run the sealed 34-call one-step and multi-step equivalence suite."""
    cases = [
        ("free_movement", [1.5, 1.5], [0.4, 0.3]),
        ("static_wall", [3.5, 1.5], [0.0, 1.0]),
        ("corner_axis_order", [1.95, 1.95], [0.8, 0.8]),
        ("action_clipping", [1.5, 1.5], [3.0, -2.0]),
        ("hazard_entry", [2.4, 3.5], [0.8, 0.0]),
        ("hazard_exit", [5.6, 3.5], [0.8, 0.0]),
        ("fatal_incoming_cell_1", [3.6, 3.5], [0.8, 0.0]),
        ("outer_boundary", [0.05, 3.5], [-1.0, 0.0]),
    ]
    bit_sets = [np.zeros(3, bool), np.ones(3, bool)]
    sequences = [
        ("direct_hazard_traversal", [0.5, 3.5], [[1.0, 0.0]] * 8),
        (
            "lower_route_and_history",
            [1.5, 3.5],
            [[0.0, -1.0]] * 2 + [[1.0, 0.0]] * 6 + [[0.0, 1.0]] * 2,
        ),
    ]
    total_calls = len(cases) * len(bit_sets) + sum(len(actions) for _, _, actions in sequences)
    noise_count = len(cases) + sum(len(actions) for _, _, actions in sequences)
    if total_calls != 34:
        raise AssertionError("sealed verification call count changed")
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, 0.01, (noise_count, 2)).astype(np.float64)
    noise_cursor = 0
    native_calls = 0
    one_step = []
    for name, start, action in cases:
        start = np.asarray(start, np.float64)
        f4 = _history(start)
        pair = []
        shared_noise = noise[noise_cursor]
        for bits in bit_sets:
            expected_xy, clipped, blocked_updates = native_alive_motion(start, action, shared_noise)
            expected_f4 = emit_f4(expected_xy, f4)
            actual = _native_step(env_class, start, f4, action, shared_noise, bits)
            np.testing.assert_array_equal(actual["physical_xy"], expected_xy)
            np.testing.assert_array_equal(actual["f4"], expected_f4)
            pair.append(actual)
            one_step.append(
                {
                    "case": name,
                    "bits": bits.tolist(),
                    "start_physical_xy": start.tolist(),
                    "action": np.asarray(action, np.float64).tolist(),
                    "motion_noise": shared_noise.tolist(),
                    "clipped_noisy_action": np.asarray(clipped).tolist(),
                    "blocked_updates_by_axis": np.asarray(blocked_updates).tolist(),
                    "next_physical_xy": expected_xy.tolist(),
                    "next_emitted_xy": expected_f4[:2].tolist(),
                    "hazard_physical": bool(hazardous(expected_xy)),
                    "hazard_emitted": bool(hazardous(expected_f4[:2])),
                    "native_dead_after": actual["dead"],
                    "native_reward_discarded": actual["reward"],
                }
            )
            native_calls += 1
        np.testing.assert_array_equal(pair[0]["physical_xy"], pair[1]["physical_xy"])
        np.testing.assert_array_equal(pair[0]["f4"], pair[1]["f4"])
        noise_cursor += 1

    multistep = []
    for name, start, actions in sequences:
        physical = np.asarray(start, np.float64)
        f4 = _history(physical)
        env = env_class(action_noise=0.01, max_episode_steps=50, seed=9917, active_prob=0.3)
        _set_native_state(env, physical, f4)
        env.set_swamp(np.zeros(3, bool))
        records = []
        for step, action in enumerate(actions):
            step_noise = noise[noise_cursor]
            expected_xy, clipped, blocked_updates = native_alive_motion(physical, action, step_noise)
            expected_f4 = emit_f4(expected_xy, f4)
            fixed_rng = FixedNoiseRNG(step_noise)
            env._rng = fixed_rng
            obs, reward, done, info = env.step(action)
            if fixed_rng.normal_calls != 1 or env.dead or done or info:
                raise AssertionError("unexpected all-clear multistep native contract")
            np.testing.assert_array_equal(np.asarray(env.state, np.float64), expected_xy)
            np.testing.assert_array_equal(np.asarray(obs[:8], np.float32), expected_f4)
            records.append(
                {
                    "step": step,
                    "action": np.asarray(action, np.float64).tolist(),
                    "motion_noise": step_noise.tolist(),
                    "clipped_noisy_action": np.asarray(clipped).tolist(),
                    "blocked_updates_by_axis": np.asarray(blocked_updates).tolist(),
                    "physical_xy": expected_xy.tolist(),
                    "f4": expected_f4.tolist(),
                    "native_reward_discarded": float(reward),
                }
            )
            physical, f4 = expected_xy, expected_f4
            noise_cursor += 1
            native_calls += 1
        multistep.append({"sequence": name, "records": records})
    if native_calls != total_calls or noise_cursor != noise_count:
        raise AssertionError("verification noise/call accounting mismatch")
    flat = one_step + [record for sequence in multistep for record in sequence["records"]]
    hazard_mismatch = sum(
        int(record.get("hazard_physical", False) != record.get("hazard_emitted", False))
        for record in one_step
    )
    clipping_cases = [record for record in one_step if record["case"] == "action_clipping"]
    if not clipping_cases or not all(np.max(np.abs(record["clipped_noisy_action"])) <= 1 for record in clipping_cases):
        raise AssertionError("action clipping verification missing")
    fatal = [record for record in one_step if record["hazard_physical"] and record["bits"] == [True, True, True]]
    if not fatal or not all(record["native_dead_after"] for record in fatal):
        raise AssertionError("fatal incoming movement was not exercised")
    return {
        "status": "passed",
        "native_step_calls": total_calls,
        "one_step_native_calls": len(cases) * len(bit_sets),
        "multistep_native_calls": sum(len(actions) for _, _, actions in sequences),
        "bit_exact_physical_xy": True,
        "bit_exact_emitted_f4": True,
        "hidden_bits_do_not_change_incoming_motion": True,
        "fatal_incoming_cases": len(fatal),
        "physical_emitted_hazard_disagreements": hazard_mismatch,
        "native_death_and_reward_discarded": True,
        "one_step_cases": one_step,
        "multistep_sequences": multistep,
        "all_explicit_noise_draws": noise.tolist(),
        "record_count": len(flat),
    }
