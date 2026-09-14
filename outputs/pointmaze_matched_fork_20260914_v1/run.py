"""Bounded matched-fork diagnostic for the frozen PointMaze C1 configuration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PILOT_DIR = ROOT / "outputs" / "pointmaze_oracle_training_pilot_20260914_v1"
ALIVE_DIR = ROOT / "outputs" / "pointmaze_oracle_alive_motion_20260914_v1"
REPAIR_DIR = ROOT / "outputs" / "pointmaze_supervised_repair_20260914_v1"
PERSISTENT_DIR = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
INDEPENDENT_REVIEW = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\oracle-training-84d378f\REVIEW.md")
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ALIVE_DIR))

from crl import checkpoint  # noqa: E402
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402
from ett.fixed_actor_continuation import TimedSimulator  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402
from oracle_motion import emit_f4, hazardous, native_alive_motion  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PILOT = load_module("matched_fork_pilot", PILOT_DIR / "run.py")


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dependency_paths():
    nominal = Path(PILOT.original.inputs()["nominal"])
    dataset = Path(PILOT.original.inputs()["dataset"])
    return [
        PILOT_DIR / "C_s1_native.npz",
        PILOT_DIR / "B_s1_batch_audit.npz",
        PILOT_DIR / "checkpoints" / "C_s1_final.pkl",
        PILOT_DIR / "config.json",
        PILOT_DIR / "REPORT.md",
        PILOT_DIR / "verification.json",
        PILOT_DIR / "run.py",
        ALIVE_DIR / "oracle_motion.py",
        ALIVE_DIR / "movement_equivalence.json",
        ALIVE_DIR / "verification.json",
        REPAIR_DIR / "head_repaired_s1.npz",
        PERSISTENT_DIR / "head_feature_scaling.npz",
        nominal,
        dataset,
        ROOT / "crl" / "envs.py",
        ROOT / "crl" / "networks.py",
        ROOT / "crl" / "checkpoint.py",
        ROOT / "ett" / "fixed_actor_continuation.py",
        ROOT / "ett" / "pointmaze_native_policy.py",
        INDEPENDENT_REVIEW,
    ]


def source_paths():
    return [OUT / "config.json", OUT / "PROTOCOL.md", OUT / "run.py", OUT / "analyze.py", OUT / "verify_saved.py"]


class Ledger:
    CAPS = {
        "initialization_native_steps": CONFIG["initialization_native_step_cap"],
        "primary_native_paths": CONFIG["primary_native_path_cap"],
        "primary_model_paths": CONFIG["primary_model_path_cap"],
        "primary_native_transitions": CONFIG["primary_native_transition_cap"],
        "primary_model_transitions": CONFIG["primary_model_transition_cap"],
        "followup_native_paths": CONFIG["conditional_followup_native_path_cap"],
        "followup_model_paths": CONFIG["conditional_followup_model_path_cap"],
        "followup_native_transitions": CONFIG["conditional_followup_native_transition_cap"],
        "followup_model_transitions": CONFIG["conditional_followup_model_transition_cap"],
        "actor_probe_updates": 3 * CONFIG["conditional_actor_probe_update_cap_per_objective"],
        "training_updates": 0,
        "new_training_episodes": 0,
        "checkpoint_searches": 0,
    }

    def __init__(self):
        self.path = OUT / "ledger.json"
        if self.path.exists():
            self.data = load_json(self.path)
        else:
            self.data = {key: {"counts": {}, "total": 0, "cap": int(cap)} for key, cap in self.CAPS.items()}
            self.save()

    def charge(self, category, count, purpose):
        count = int(count)
        row = self.data[category]
        if purpose in row["counts"]:
            if row["counts"][purpose] != count:
                raise RuntimeError(f"ledger charge changed: {category}/{purpose}")
            return False
        if row["total"] + count > row["cap"]:
            raise RuntimeError(f"budget exceeded: {category}")
        row["counts"][purpose] = count
        row["total"] += count
        self.save()
        return True

    def save(self):
        write_json(self.path, self.data)


def native_source():
    return load_npz(PILOT_DIR / "C_s1_native.npz")


def candidate_arrays():
    actions = CONFIG["candidate_actions"]
    return {name: np.asarray(actions[name], np.float32) for name in ("down", "right")}


def previous_hazard_mask(data):
    hazard = np.asarray(data["hazard_landing"], bool)
    return np.concatenate(
        [np.zeros((len(hazard), 1), bool), np.maximum.accumulate(hazard, axis=1)[:, :-1]], axis=1
    )


def reconstruct_root(data, episode, time_index):
    simulator = TimedSimulator(int(data["reset_seed"][episode]))
    np.testing.assert_array_equal(simulator.observation()[:8], data["states"][episode, 0])
    np.testing.assert_array_equal(simulator.env.state, data["physical_states"][episode, 0])
    for step in range(int(time_index)):
        failed_before = simulator.env.dead
        observation, reward, _ = simulator.step(data["action"][episode, step])
        np.testing.assert_array_equal(observation[:8], data["states"][episode, step + 1])
        np.testing.assert_array_equal(simulator.env.state, data["physical_states"][episode, step + 1])
        if bool(failed_before) != bool(data["failed_before"][episode, step]):
            raise AssertionError("saved failed-before mismatch during root replay")
        if bool(simulator.env.dead) != bool(data["failed_after"][episode, step]):
            raise AssertionError("saved failed-after mismatch during root replay")
        if float(reward) != float(data["reward"][episode, step]):
            raise AssertionError("saved reward mismatch during root replay")
    if simulator.elapsed != int(time_index) or simulator.env.dead:
        raise AssertionError("selected root is not a coherent alive native state")
    return simulator


def select_roots(data):
    states = data["states"][:, :-1]
    xy = states[:, :, :2]
    geometry = CONFIG["fork_geometry"]
    mask = (
        (xy[:, :, 0] >= geometry["x_min_inclusive"])
        & (xy[:, :, 0] < geometry["x_max_exclusive"])
        & (xy[:, :, 1] >= geometry["y_min_inclusive"])
        & (xy[:, :, 1] < geometry["y_max_exclusive"])
        & (~data["failed_before"])
        & (~previous_hazard_mask(data))
    )
    times = np.arange(data["action"].shape[1])[None, :]
    mask &= (CONFIG["native_horizon"] - times) >= geometry["minimum_remaining_steps"]
    actions = candidate_arrays()
    rows = []
    for episode in range(len(states)):
        for time_index in np.flatnonzero(mask[episode]):
            physical = data["physical_states"][episode, time_index][None]
            noiseless = {}
            cells = {}
            good = True
            for name in ("down", "right"):
                moved = native_alive_motion(physical, actions[name][None], np.zeros((1, 2), np.float64))[0][0]
                noiseless[name] = moved
                cells[name] = np.floor(moved).astype(np.int64)
                good &= bool(np.array_equal(cells[name], np.asarray(CONFIG["candidate_exit_cells"][name])))
            if good:
                rows.append((episode, int(time_index), noiseless, cells))
                break
        if len(rows) == CONFIG["root_target"]:
            break
    return rows


def scheduled_step(simulator, action, motion_noise, post_step_bits):
    """One native step with explicit exogenous draws and correct hidden timing.

    The naturally reached current bits stay installed for this step. Gaussian
    action noise is supplied explicitly. Automatic resampling is disabled, then
    fresh post-step bits are installed for the next step, including after death.
    """
    bits_before = simulator.env.swamp_bits
    simulator.env._action_noise = 0.0
    simulator.env._auto_resample = False
    noisy_action = np.clip(
        np.asarray(action, np.float64) + np.asarray(motion_noise, np.float64), -1.0, 1.0
    )
    simulator.require_horizon(1)
    before = simulator.observation()
    observation, reward, done, _ = simulator.env.step(noisy_action)
    simulator.elapsed += 1
    if done:
        raise AssertionError("fixed-length native environment terminated")
    np.testing.assert_array_equal(observation[2:8], before[:6])
    np.testing.assert_array_equal(observation[8:], before[8:])
    visible_reward = float(np.linalg.norm(observation[:2].astype(float) - observation[8:10]) < 2)
    if visible_reward != reward:
        raise RuntimeError("native reward and visible reconstruction disagree")
    simulator.env.set_swamp(np.asarray(post_step_bits, bool))
    return observation, float(reward), bits_before, simulator.env.swamp_bits


def initialization_checks(data, rows, ledger):
    checks = []
    root_bits = []
    replay_steps = 0
    schedule_steps = 0
    actions = candidate_arrays()
    for root_id, (episode, time_index, _, _) in enumerate(rows):
        simulator = reconstruct_root(data, episode, time_index)
        replay_steps += time_index
        snapshot = simulator.snapshot()
        root_bits.append(simulator.env.swamp_bits)

        future_seed = CONFIG["initialization_check_seed"] + root_id
        automatic = TimedSimulator.restore(snapshot, future_seed=future_seed)
        predictor = np.random.default_rng(future_seed)
        noise = predictor.normal(0.0, CONFIG["motion_noise_standard_deviation"], 2)
        after_bits = predictor.random(3) < CONFIG["active_probability"]
        auto_observation, auto_reward, _ = automatic.step(actions["right"])

        explicit = TimedSimulator.restore(snapshot)
        explicit_observation, explicit_reward, bits_before, bits_after = scheduled_step(
            explicit, actions["right"], noise, after_bits
        )
        schedule_steps += 2
        np.testing.assert_array_equal(auto_observation, explicit_observation)
        np.testing.assert_array_equal(automatic.env.state, explicit.env.state)
        np.testing.assert_array_equal(automatic.env.swamp_bits, explicit.env.swamp_bits)
        if automatic.env.dead != explicit.env.dead or auto_reward != explicit_reward:
            raise AssertionError("explicit schedule does not reproduce native timing")
        checks.append(
            {
                "root": root_id,
                "episode": episode,
                "time": time_index,
                "bits_before": bits_before,
                "predicted_post_step_bits": after_bits,
                "actual_post_step_bits": bits_after,
                "noise": noise,
                "automatic_equals_explicit": True,
            }
        )

    # Focused current-bit death check at an identical hazard landing.
    lethal = []
    for active in (False, True):
        env = TwoRouteSwampWindyF4Env(
            seed=CONFIG["initialization_check_seed"] + 100,
            active_prob=CONFIG["active_probability"],
            action_noise=0.0,
            max_episode_steps=CONFIG["native_horizon"],
        )
        env.state = np.asarray([2.5, 3.5], np.float64)
        env._frames = [env.state.copy() for _ in range(4)]
        env._dead = False
        env._auto_resample = False
        env.set_swamp(np.asarray([active, False, False]))
        observation, reward, done, _ = env.step(np.asarray([1.0, 0.0], np.float32))
        schedule_steps += 1
        lethal.append(
            {
                "active_current_cell": active,
                "next_xy": observation[:2],
                "dead_after": env.dead,
                "reward": reward,
                "done": done,
            }
        )
    np.testing.assert_array_equal(lethal[0]["next_xy"], lethal[1]["next_xy"])
    if lethal[0]["dead_after"] or not lethal[1]["dead_after"]:
        raise AssertionError("current pre-action bits did not govern native death check")

    ledger.charge("initialization_native_steps", replay_steps + schedule_steps, "root_replay_and_timing")
    return {
        "root_count": len(rows),
        "root_replay_native_steps": replay_steps,
        "schedule_equivalence_native_steps": schedule_steps,
        "all_roots_reached_from_reset": True,
        "saved_state_physical_time_and_absorption_exact": True,
        "current_bits_used_for_current_death_check": True,
        "bits_resampled_after_every_step": True,
        "explicit_schedule_matches_native_rng_order": True,
        "native_hidden_fields_actor_input": False,
        "native_hidden_fields_learned_head_input": False,
        "checks": checks,
        "lethal_timing_pair": lethal,
        "root_bits": np.asarray(root_bits, bool),
    }


def make_streams(seed, roots, repeats, max_horizon):
    rng = np.random.default_rng(seed)
    shape = (roots, repeats, max_horizon)
    return {
        "actor_epsilon": rng.standard_normal(shape + (2,)).astype(np.float32),
        "motion_noise": rng.normal(0.0, CONFIG["motion_noise_standard_deviation"], shape + (2,)).astype(np.float64),
        "native_bits_after": (rng.random(shape + (3,)) < CONFIG["active_probability"]),
        "model_onset_uniform": rng.random(shape).astype(np.float32),
        "nominal_key": np.stack(
            [np.asarray(jax.random.PRNGKey(CONFIG["nominal_key_seed"] + seed + step), np.uint32) for step in range(max_horizon)]
        ),
    }


def prepare():
    generated = [
        OUT / "preregistration.json",
        OUT / "root_selection.npz",
        OUT / "streams_primary.npz",
        OUT / "bootstrap_weights.npz",
        OUT / "actor_probe_streams.npz",
        OUT / "actor_probe_common_batch.npz",
        OUT / "ledger.json",
    ]
    if any(path.exists() for path in generated):
        raise RuntimeError("fresh diagnostic preparation required; existing sealed/generated files found")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != CONFIG["reviewed_commit"]:
        raise RuntimeError(f"expected reviewed commit {CONFIG['reviewed_commit']}, found {head}")
    if not INDEPENDENT_REVIEW.exists():
        raise RuntimeError("required independent review is unavailable")
    if load_json(PILOT_DIR / "verification.json")["status"] != "passed":
        raise RuntimeError("historical pilot verification did not pass")
    if load_json(ALIVE_DIR / "movement_equivalence.json")["status"] != "passed":
        raise RuntimeError("oracle movement equivalence did not pass")

    ledger = Ledger()
    data = native_source()
    rows = select_roots(data)
    if not rows:
        raise RuntimeError("no qualifying coherent roots")
    checks = initialization_checks(data, rows, ledger)
    actions = candidate_arrays()
    episodes = np.asarray([row[0] for row in rows], np.int64)
    times = np.asarray([row[1] for row in rows], np.int64)
    states = data["states"][episodes, times]
    physical = data["physical_states"][episodes, times]
    noiseless = np.stack([[row[2][name] for name in ("down", "right")] for row in rows])
    cells = np.floor(noiseless).astype(np.int64)
    np.savez_compressed(
        OUT / "root_selection.npz",
        episode=episodes,
        time=times,
        reset_seed=data["reset_seed"][episodes],
        state=states,
        physical_state=physical,
        root_bits=checks.pop("root_bits"),
        candidate_name=np.asarray(["down", "right"]),
        candidate_action=np.stack([actions["down"], actions["right"]]),
        noiseless_next_physical=noiseless,
        noiseless_displacement=noiseless - physical[:, None, :],
        noiseless_exit_cell=cells,
        saved_prefix_reward=np.stack(
            [data["reward"][episode, :time_index] for episode, time_index in zip(episodes, times)]
        ),
    )
    write_json(OUT / "initialization_checks.json", checks)
    write_json(
        OUT / "root_selection.json",
        {
            "selection_rule": CONFIG["root_order"],
            "target": CONFIG["root_target"],
            "selected": len(rows),
            "limitation": None if len(rows) == CONFIG["root_target"] else "fewer than 16 roots qualified",
            "rows": [
                {
                    "root": index,
                    "saved_episode": int(episodes[index]),
                    "reset_seed": int(data["reset_seed"][episodes[index]]),
                    "time": int(times[index]),
                    "remaining_steps": int(CONFIG["native_horizon"] - times[index]),
                    "state_f4": states[index],
                    "physical_xy": physical[index],
                    "root_bits_audit_only": load_npz(OUT / "root_selection.npz")["root_bits"][index],
                    "candidate_down": actions["down"],
                    "candidate_right": actions["right"],
                    "noiseless_next_down": noiseless[index, 0],
                    "noiseless_next_right": noiseless[index, 1],
                }
                for index in range(len(rows))
            ],
        },
    )
    max_horizon = int(np.max(CONFIG["native_horizon"] - times))
    streams = make_streams(CONFIG["primary_stream_seed"], len(rows), CONFIG["paths_per_root_action_backend"], max_horizon)
    np.savez_compressed(OUT / "streams_primary.npz", **streams)
    rng = np.random.default_rng(CONFIG["root_bootstrap_seed"])
    weights = rng.multinomial(
        len(rows), np.full(len(rows), 1.0 / len(rows)), size=CONFIG["bootstrap_replicates"]
    ).astype(np.float64) / len(rows)
    np.savez_compressed(OUT / "bootstrap_weights.npz", weights=weights)
    probe_rng = np.random.default_rng(CONFIG["primary_stream_seed"] + 1000)
    np.savez_compressed(
        OUT / "actor_probe_streams.npz",
        update_epsilon=probe_rng.standard_normal(
            (CONFIG["conditional_actor_probe_update_cap_per_objective"], 512, 2)
        ).astype(np.float32),
        evaluation_epsilon=probe_rng.standard_normal(
            (len(rows), CONFIG["actor_probe_eval_draws_per_root"], 2)
        ).astype(np.float32),
    )
    observation, offline_action, _, _ = PILOT.original.dataset()
    audit = load_npz(PILOT_DIR / "B_s1_batch_audit.npz")
    episode = audit["offline_episode"][0]
    anchor = audit["offline_time"][0]
    future = audit["offline_future"][0]
    order = audit["permutation"][0]
    common_state = observation[episode, anchor, :8]
    common_goal = observation[episode, future, :8]
    common_action = offline_action[episode, anchor]
    np.savez_compressed(
        OUT / "actor_probe_common_batch.npz",
        observation=np.concatenate([common_state, common_goal], axis=-1)[order].astype(np.float32),
        action=common_action[order].astype(np.float32),
        episode=episode[order],
        anchor=anchor[order],
        future=future[order],
        source_audit_update=np.asarray(0, np.int64),
    )

    preregistration = {
        "status": "sealed_before_candidate_continuation_outcomes",
        "utc": utc(),
        "working_head": head,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dependency_sha256": {str(path): sha256(path) for path in dependency_paths()},
        "diagnostic_source_sha256": {str(path): sha256(path) for path in source_paths()},
        "sealed_generated_sha256": {
            path.name: sha256(path)
            for path in [
                OUT / "root_selection.npz",
                OUT / "root_selection.json",
                OUT / "initialization_checks.json",
                OUT / "streams_primary.npz",
                OUT / "bootstrap_weights.npz",
                OUT / "actor_probe_streams.npz",
                OUT / "actor_probe_common_batch.npz",
            ]
        },
        "checkpoint_step": checkpoint.load_checkpoint(PILOT_DIR / "checkpoints" / "C_s1_final.pkl")[0],
        "checkpoint_sha256": sha256(PILOT_DIR / "checkpoints" / "C_s1_final.pkl"),
        "head_sha256": sha256(REPAIR_DIR / "head_repaired_s1.npz"),
        "nominal_sha256": sha256(Path(PILOT.original.inputs()["nominal"])),
        "selected_roots": len(rows),
        "planned_primary_paths": len(rows) * 2 * 2 * CONFIG["paths_per_root_action_backend"],
        "maximum_primary_transitions": len(rows) * 2 * 2 * CONFIG["paths_per_root_action_backend"] * max_horizon,
        "training_updates": 0,
        "new_training_episodes": 0,
    }
    write_json(OUT / "preregistration.json", preregistration)
    print(f"sealed {len(rows)} coherent roots; primary path budget {preregistration['planned_primary_paths']}", flush=True)


def verify_seal():
    sealed = load_json(OUT / "preregistration.json")
    if sealed["status"] != "sealed_before_candidate_continuation_outcomes":
        raise RuntimeError("invalid preregistration status")
    for path, expected in sealed["dependency_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"dependency changed: {path}")
    for path, expected in sealed["diagnostic_source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"diagnostic source changed: {path}")
    for name, expected in sealed["sealed_generated_sha256"].items():
        if sha256(OUT / name) != expected:
            raise RuntimeError(f"sealed generated input changed: {name}")
    return sealed


def frozen_components():
    cfg, network, _ = PILOT.learner_setup()
    step, state = checkpoint.load_checkpoint(PILOT_DIR / "checkpoints" / "C_s1_final.pkl")
    nominal, heads, state_mean, state_std = PILOT.oracle_components()
    return cfg, network, step, state, nominal, heads[1], state_mean, state_std


def policy_action(network, params, states, epsilon):
    goal = np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)
    observation = jnp.asarray(np.concatenate([states, goal], axis=-1))
    distribution = network.policy_network.apply(params, observation)
    return np.asarray(jnp.tanh(distribution.loc + distribution.scale * jnp.asarray(epsilon)), np.float32)


def critic_scores(network, q_params, roots):
    names = ("down", "right")
    actions = candidate_arrays()
    goal = np.broadcast_to(np.asarray(GOAL, np.float32), roots.shape)
    observation = jnp.asarray(np.concatenate([roots, goal], axis=-1))

    def score(one_action):
        batch_action = jnp.asarray(np.broadcast_to(one_action, (len(roots), 2)))
        phi, psi = network.representation_network.apply(q_params, observation, batch_action)
        value = jnp.sum(phi * psi, axis=1)
        if value.ndim == 2:
            value = value[:, 0]
        return np.asarray(value, np.float32)

    return np.stack([score(actions[name]) for name in names], axis=1)


def blank_trace(root_count, action_count, repeats, horizon, model=False):
    path = (root_count, action_count, repeats, horizon)
    result = {
        "valid": np.zeros(path, bool),
        "states": np.full(path[:-1] + (horizon + 1, 8), np.nan, np.float32),
        "physical_states": np.full(path[:-1] + (horizon + 1, 2), np.nan, np.float64),
        "action": np.full(path + (2,), np.nan, np.float32),
        "actor_epsilon": np.full(path + (2,), np.nan, np.float32),
        "motion_noise": np.full(path + (2,), np.nan, np.float64),
        "clipped_noisy_action": np.full(path + (2,), np.nan, np.float64),
        "reward": np.zeros(path, np.float32),
        "failed_before": np.zeros(path, bool),
        "failed_after": np.zeros(path, bool),
        "hazard_landing": np.zeros(path, bool),
        "lower_route": np.zeros(path, bool),
        "strict_success": np.zeros(path, bool),
    }
    if model:
        result.update(
            xb=np.full(path + (2,), np.nan, np.float32),
            blocked_updates=np.full(path + (2,), -1, np.int16),
            onset=np.zeros(path, bool),
            onset_uniform=np.full(path, np.nan, np.float32),
            failure_probability=np.full(path, np.nan, np.float32),
            raw_failure_probability=np.full(path, np.nan, np.float32),
        )
    else:
        result.update(
            swamp_bits_before=np.zeros(path + (3,), bool),
            swamp_bits_after=np.zeros(path + (3,), bool),
        )
    return result


def intervention_actions(kind):
    actions = candidate_arrays()
    if kind == "primary":
        return ("down", "right"), [[actions["down"]], [actions["right"]]]
    if kind == "prefix":
        return ("prefix",), [[np.asarray(action, np.float32) for action in CONFIG["conditional_prefix_actions"]]]
    raise ValueError(kind)


def native_rollouts(kind, data, roots, streams, network, policy_params):
    action_names, forced_sequences = intervention_actions(kind)
    root_count = len(roots["episode"])
    repeats = CONFIG["paths_per_root_action_backend"]
    remaining = CONFIG["native_horizon"] - roots["time"]
    max_horizon = int(remaining.max())
    trace = blank_trace(root_count, len(action_names), repeats, max_horizon, model=False)
    base_snapshots = []
    for episode, time_index in zip(roots["episode"], roots["time"]):
        simulator = reconstruct_root(data, int(episode), int(time_index))
        base_snapshots.append(simulator.snapshot())

    for action_id, (name, forced) in enumerate(zip(action_names, forced_sequences)):
        simulators = [
            TimedSimulator.restore(base_snapshots[root_id])
            for root_id in range(root_count)
            for _ in range(repeats)
        ]
        initial_states = np.repeat(roots["state"], repeats, axis=0)
        trace["states"][:, action_id, :, 0] = initial_states.reshape(root_count, repeats, 8)
        initial_physical = np.repeat(roots["physical_state"], repeats, axis=0)
        trace["physical_states"][:, action_id, :, 0] = initial_physical.reshape(root_count, repeats, 2)
        for local_step in range(max_horizon):
            current = np.stack([simulator.observation()[:8] for simulator in simulators]).astype(np.float32)
            if local_step < len(forced):
                action = np.broadcast_to(forced[local_step], (len(simulators), 2)).copy()
            else:
                epsilon = streams["actor_epsilon"][:, :, local_step].reshape(-1, 2)
                action = policy_action(network, policy_params, current, epsilon)
            for flat_id, simulator in enumerate(simulators):
                root_id, repeat_id = divmod(flat_id, repeats)
                if local_step >= remaining[root_id]:
                    continue
                noise = streams["motion_noise"][root_id, repeat_id, local_step]
                post_bits = streams["native_bits_after"][root_id, repeat_id, local_step]
                failed_before = bool(simulator.env.dead)
                observation, reward, bits_before, bits_after = scheduled_step(simulator, action[flat_id], noise, post_bits)
                failed_after = bool(simulator.env.dead)
                physical = simulator.env.state.copy()
                hazard = bool(hazardous(physical[None])[0])
                strict = bool(np.linalg.norm(physical - np.asarray(CONFIG["task_goal_xy"])) < CONFIG["strict_success_radius"] and not failed_after)
                lower = bool(physical[1] < 2.0 and not failed_after)
                idx = (root_id, action_id, repeat_id, local_step)
                trace["valid"][idx] = True
                trace["action"][idx] = action[flat_id]
                trace["actor_epsilon"][idx] = streams["actor_epsilon"][root_id, repeat_id, local_step]
                trace["motion_noise"][idx] = noise
                trace["clipped_noisy_action"][idx] = np.clip(action[flat_id].astype(np.float64) + noise, -1, 1)
                trace["reward"][idx] = reward
                trace["failed_before"][idx] = failed_before
                trace["failed_after"][idx] = failed_after
                trace["hazard_landing"][idx] = hazard
                trace["lower_route"][idx] = lower
                trace["strict_success"][idx] = strict
                trace["swamp_bits_before"][idx] = bits_before
                trace["swamp_bits_after"][idx] = bits_after
                trace["states"][root_id, action_id, repeat_id, local_step + 1] = observation[:8]
                trace["physical_states"][root_id, action_id, repeat_id, local_step + 1] = physical
            if (local_step + 1) % 10 == 0:
                print(f"native {name}: {local_step + 1}/{max_horizon}", flush=True)
    trace["action_name"] = np.asarray(action_names)
    trace["root_time"] = roots["time"]
    return trace


def model_rollouts(kind, roots, streams, network, policy_params, nominal, head, state_mean, state_std):
    action_names, forced_sequences = intervention_actions(kind)
    root_count = len(roots["episode"])
    repeats = CONFIG["paths_per_root_action_backend"]
    remaining = CONFIG["native_horizon"] - roots["time"]
    max_horizon = int(remaining.max())
    trace = blank_trace(root_count, len(action_names), repeats, max_horizon, model=True)
    head = jnp.asarray(head, jnp.float32)
    mean = jnp.asarray(state_mean, jnp.float32)
    std = jnp.asarray(state_std, jnp.float32)
    predict = jax.jit(lambda features: jax.nn.sigmoid(PILOT.REPAIR_COMMON.PARENT.head_logits(head, features)))

    for action_id, (name, forced) in enumerate(zip(action_names, forced_sequences)):
        state = np.repeat(roots["state"], repeats, axis=0).astype(np.float32)
        physical = np.repeat(roots["physical_state"], repeats, axis=0).astype(np.float64)
        failed = np.zeros(len(state), bool)
        trace["states"][:, action_id, :, 0] = state.reshape(root_count, repeats, 8)
        trace["physical_states"][:, action_id, :, 0] = physical.reshape(root_count, repeats, 2)
        goal = np.broadcast_to(np.asarray(GOAL, np.float32), state.shape)
        for local_step in range(max_horizon):
            if local_step < len(forced):
                action = np.broadcast_to(forced[local_step], (len(state), 2)).copy()
            else:
                epsilon = streams["actor_epsilon"][:, :, local_step].reshape(-1, 2)
                action = policy_action(network, policy_params, state, epsilon)
            key = jnp.asarray(streams["nominal_key"][local_step])
            xb = np.asarray(nominal.sample(jnp.asarray(state), key, 1, goal=jnp.asarray(goal)), np.float32)
            noise = streams["motion_noise"][:, :, local_step].reshape(-1, 2)
            moved, clipped, blocked = native_alive_motion(physical, action, noise)
            next_physical = np.where(failed[:, None], physical, moved)
            next_state = emit_f4(next_physical, state)
            support = np.asarray(hazardous(next_state[:, :2]), bool)
            displacement = jnp.asarray(next_state[:, :2]) - jnp.asarray(state[:, :2])
            features = jnp.concatenate([(jnp.asarray(state) - mean) / std, jnp.asarray(xb), jnp.asarray(action), displacement], axis=-1)
            raw_probability = np.asarray(predict(features), np.float32)
            probability = np.where(failed, 0.0, np.where(support, raw_probability, 0.0)).astype(np.float32)
            onset_uniform = streams["model_onset_uniform"][:, :, local_step].reshape(-1)
            onset = (~failed) & (onset_uniform < probability)
            next_failed = failed | onset
            reward = ((np.linalg.norm(next_physical - np.asarray(CONFIG["task_goal_xy"]), axis=-1) < CONFIG["reward_radius"]) & (~next_failed)).astype(np.float32)
            strict = (np.linalg.norm(next_physical - np.asarray(CONFIG["task_goal_xy"]), axis=-1) < CONFIG["strict_success_radius"]) & (~next_failed)
            lower = (next_physical[:, 1] < 2.0) & (~next_failed)
            valid = local_step < np.repeat(remaining, repeats)
            shape2 = (root_count, repeats)
            idx = (slice(None), action_id, slice(None), local_step)
            trace["valid"][idx] = valid.reshape(shape2)
            for field, value in {
                "action": action,
                "actor_epsilon": streams["actor_epsilon"][:, :, local_step].reshape(-1, 2),
                "motion_noise": noise,
                "clipped_noisy_action": clipped,
                "reward": reward,
                "failed_before": failed,
                "failed_after": next_failed,
                "hazard_landing": support,
                "lower_route": lower,
                "strict_success": strict,
                "xb": xb,
                "blocked_updates": blocked,
                "onset": onset,
                "onset_uniform": onset_uniform,
                "failure_probability": probability,
                "raw_failure_probability": raw_probability,
            }.items():
                reshaped = value.reshape(shape2 + value.shape[1:])
                trace[field][idx] = reshaped
            trace["states"][:, action_id, :, local_step + 1] = next_state.reshape(root_count, repeats, 8)
            trace["physical_states"][:, action_id, :, local_step + 1] = next_physical.reshape(root_count, repeats, 2)
            state, physical, failed = next_state, next_physical, next_failed
            if (local_step + 1) % 10 == 0:
                print(f"model {name}: {local_step + 1}/{max_horizon}", flush=True)
    trace["action_name"] = np.asarray(action_names)
    trace["root_time"] = roots["time"]
    trace["head_seed"] = np.asarray(1, np.int64)
    return trace


def save_trace(path, trace):
    np.savez_compressed(path, **trace)


def execute_primary():
    sealed = verify_seal()
    if (OUT / "primary_execution_started.json").exists():
        raise RuntimeError("primary execution already started; historical attempt protected")
    write_json(OUT / "primary_execution_started.json", {"utc": utc(), "sealed_sha256": sha256(OUT / "preregistration.json")})
    started = time.monotonic()
    ledger = Ledger()
    roots = load_npz(OUT / "root_selection.npz")
    streams = load_npz(OUT / "streams_primary.npz")
    data = native_source()
    _, network, step, state, nominal, head, state_mean, state_std = frozen_components()
    if step != sealed["checkpoint_step"]:
        raise RuntimeError("checkpoint step changed")
    root_count = len(roots["episode"])
    repeats = CONFIG["paths_per_root_action_backend"]
    transitions = int(np.sum(CONFIG["native_horizon"] - roots["time"]) * 2 * repeats)
    paths = root_count * 2 * repeats
    ledger.charge("initialization_native_steps", int(np.sum(roots["time"])), "primary_root_regeneration")
    ledger.charge("primary_native_paths", paths, "down_right")
    ledger.charge("primary_model_paths", paths, "down_right")
    ledger.charge("primary_native_transitions", transitions, "down_right")
    ledger.charge("primary_model_transitions", transitions, "down_right")

    scores = critic_scores(network, state.q_params, roots["state"])
    np.savez_compressed(OUT / "critic_scores.npz", score=scores, action_name=np.asarray(["down", "right"]))
    native = native_rollouts("primary", data, roots, streams, network, state.policy_params)
    save_trace(OUT / "primary_native_traces.npz", native)
    model = model_rollouts("primary", roots, streams, network, state.policy_params, nominal, head, state_mean, state_std)
    save_trace(OUT / "primary_model_traces.npz", model)
    elapsed = time.monotonic() - started
    if elapsed > CONFIG["runtime_cap_seconds"]:
        raise RuntimeError("runtime cap exceeded")
    write_json(
        OUT / "primary_execution.json",
        {
            "utc": utc(),
            "runtime_seconds": elapsed,
            "roots": root_count,
            "paths_per_root_action_backend": repeats,
            "native_paths": paths,
            "model_paths": paths,
            "total_paths": 2 * paths,
            "native_transitions": transitions,
            "model_transitions": transitions,
            "training_updates": 0,
            "checkpoint_searches": 0,
            "same_frozen_actor_for_all_continuations": True,
            "native_model_shared_randomness": ["actor Gaussian innovations", "physical motion noise"],
            "native_only_shared_between_actions": ["current root bits", "future post-step swamp bits"],
            "model_only_shared_between_actions": ["nominal sampling keys", "failure onset uniforms"],
        },
    )
    print(f"primary complete: {2 * paths} paths, {2 * transitions} backend transitions", flush=True)


def execute_followup():
    verify_seal()
    decision = load_json(OUT / "primary_decision.json")
    if decision["selected_branch"] != "A":
        raise RuntimeError(f"conditional prefix prohibited for selected branch {decision['selected_branch']}")
    if (OUT / "followup_execution_started.json").exists():
        raise RuntimeError("follow-up execution already started; historical attempt protected")
    write_json(OUT / "followup_execution_started.json", {"utc": utc(), "selected_branch": "A"})
    started = time.monotonic()
    ledger = Ledger()
    roots = load_npz(OUT / "root_selection.npz")
    # Reuse the primary streams so the prefix is paired with the already-saved
    # rightward arm; no second rightward arm is generated.
    streams = load_npz(OUT / "streams_primary.npz")
    data = native_source()
    _, network, _, state, nominal, head, state_mean, state_std = frozen_components()
    root_count = len(roots["episode"])
    repeats = CONFIG["paths_per_root_action_backend"]
    transitions = int(np.sum(CONFIG["native_horizon"] - roots["time"]) * repeats)
    paths = root_count * repeats
    ledger.charge("initialization_native_steps", int(np.sum(roots["time"])), "followup_root_regeneration")
    ledger.charge("followup_native_paths", paths, "three_step_prefix")
    ledger.charge("followup_model_paths", paths, "three_step_prefix")
    ledger.charge("followup_native_transitions", transitions, "three_step_prefix")
    ledger.charge("followup_model_transitions", transitions, "three_step_prefix")
    native = native_rollouts("prefix", data, roots, streams, network, state.policy_params)
    save_trace(OUT / "followup_native_traces.npz", native)
    model = model_rollouts("prefix", roots, streams, network, state.policy_params, nominal, head, state_mean, state_std)
    save_trace(OUT / "followup_model_traces.npz", model)
    elapsed = time.monotonic() - started
    if elapsed > CONFIG["runtime_cap_seconds"]:
        raise RuntimeError("runtime cap exceeded")
    write_json(
        OUT / "followup_execution.json",
        {
            "utc": utc(),
            "selected_branch": "A",
            "runtime_seconds": elapsed,
            "native_paths": paths,
            "model_paths": paths,
            "total_paths": 2 * paths,
            "native_transitions": transitions,
            "model_transitions": transitions,
            "comparison_right_arm_reused_from_primary": True,
        },
    )
    print(f"branch A prefix complete: {2 * paths} additional paths", flush=True)


def actor_probe_loss(network, q_params, policy_params, observation, original_action, epsilon, objective):
    state = observation[:, :8]
    goal = observation[:, 8:]
    new_state = jnp.concatenate([state, state], axis=0)
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
    new_observation = jnp.concatenate([new_state, new_goal], axis=1)
    behavior_action = jnp.concatenate([original_action, original_action], axis=0)
    distribution = network.policy_network.apply(policy_params, new_observation)
    sampled_action = jnp.tanh(distribution.loc + distribution.scale * epsilon)
    q_action = network.q_network.apply(q_params, new_observation, sampled_action)
    value_term = -jnp.diag(q_action)
    bc_term = -network.log_prob(distribution, behavior_action)
    if objective == "full":
        loss = 0.5 * bc_term + 0.5 * value_term
    elif objective == "value":
        loss = value_term
    elif objective == "bc":
        loss = bc_term
    else:
        raise ValueError(objective)
    return jnp.mean(loss), (jnp.mean(value_term), jnp.mean(bc_term))


def probe_root_distribution(network, params, roots, epsilon):
    root_count, draws, _ = epsilon.shape
    states = np.repeat(roots["state"][:, None, :], draws, axis=1).reshape(-1, 8)
    goal = np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)
    distribution = network.policy_network.apply(
        params, jnp.asarray(np.concatenate([states, goal], axis=-1))
    )
    action = np.asarray(
        jnp.tanh(distribution.loc + distribution.scale * jnp.asarray(epsilon.reshape(-1, 2))), np.float32
    )
    physical = np.repeat(roots["physical_state"][:, None, :], draws, axis=1).reshape(-1, 2)
    moved = native_alive_motion(physical, action, np.zeros_like(physical))[0]
    cells = np.floor(moved).astype(np.int64).reshape(root_count, draws, 2)
    action = action.reshape(root_count, draws, 2)
    loc = np.asarray(distribution.loc, np.float32).reshape(root_count, draws, 2)[:, 0]
    scale = np.asarray(distribution.scale, np.float32).reshape(root_count, draws, 2)[:, 0]
    return {
        "action": action,
        "next_cell": cells,
        "down_exit": np.all(cells == np.asarray(CONFIG["candidate_exit_cells"]["down"]), axis=-1),
        "right_exit": np.all(cells == np.asarray(CONFIG["candidate_exit_cells"]["right"]), axis=-1),
        "loc": loc,
        "scale": scale,
    }


def execute_actor_probe():
    verify_seal()
    decision = load_json(OUT / "primary_decision.json")
    if decision["selected_branch"] != "D":
        raise RuntimeError(f"actor probe prohibited for selected branch {decision['selected_branch']}")
    if (OUT / "actor_probe_started.json").exists():
        raise RuntimeError("actor probe already started; historical attempt protected")
    write_json(OUT / "actor_probe_started.json", {"utc": utc(), "selected_branch": "D"})
    ledger = Ledger()
    roots = load_npz(OUT / "root_selection.npz")
    streams = load_npz(OUT / "actor_probe_streams.npz")
    batch = load_npz(OUT / "actor_probe_common_batch.npz")
    cfg, network, _, state, _, _, _, _ = frozen_components()
    optimizer = optax.adam(cfg.actor_learning_rate, eps=1e-7)
    grad = {
        objective: jax.jit(
            jax.value_and_grad(
                lambda params, obs, act, eps, name=objective: actor_probe_loss(
                    network, state.q_params, params, obs, act, eps, name
                ),
                has_aux=True,
            )
        )
        for objective in ("full", "value", "bc")
    }
    before = probe_root_distribution(network, state.policy_params, roots, streams["evaluation_epsilon"])
    arrays = {f"before_{key}": value for key, value in before.items()}
    result = {
        "selected_branch": "D",
        "common_batch": "exact B_s1 saved-audit update-0 offline batch after its saved permutation",
        "batch_size": len(batch["action"]),
        "updates_per_objective": CONFIG["conditional_actor_probe_update_cap_per_objective"],
        "critic_frozen": True,
        "identical_C1_initialization": True,
        "identical_batches_and_actor_innovations": True,
        "objectives": {},
        "native_continuation_validation_paths": 0,
        "continuation_improvement_claimed": False,
    }
    for objective in ("full", "value", "bc"):
        params = copy.deepcopy(state.policy_params)
        optimizer_state = copy.deepcopy(state.policy_optimizer_state)
        history = []
        for update_index in range(CONFIG["conditional_actor_probe_update_cap_per_objective"]):
            (loss, aux), gradient = grad[objective](
                params,
                jnp.asarray(batch["observation"]),
                jnp.asarray(batch["action"]),
                jnp.asarray(streams["update_epsilon"][update_index]),
            )
            updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            history.append(
                {
                    "update": update_index + 1,
                    "loss": float(loss),
                    "value_term": float(aux[0]),
                    "bc_term": float(aux[1]),
                    "gradient_norm": float(optax.global_norm(gradient)),
                }
            )
        after = probe_root_distribution(network, params, roots, streams["evaluation_epsilon"])
        for key, value in after.items():
            arrays[f"{objective}_{key}"] = value
        before_down = before["down_exit"].mean(axis=1)
        after_down = after["down_exit"].mean(axis=1)
        before_right = before["right_exit"].mean(axis=1)
        after_right = after["right_exit"].mean(axis=1)
        result["objectives"][objective] = {
            "history": history,
            "down_exit_probability_before": float(before_down.mean()),
            "down_exit_probability_after": float(after_down.mean()),
            "down_exit_change": float((after_down - before_down).mean()),
            "right_exit_probability_before": float(before_right.mean()),
            "right_exit_probability_after": float(after_right.mean()),
            "right_exit_change": float((after_right - before_right).mean()),
            "mean_action_before": before["action"].mean(axis=(0, 1)),
            "mean_action_after": after["action"].mean(axis=(0, 1)),
            "local_noiseless_movement_only": True,
        }
        ledger.charge(
            "actor_probe_updates",
            CONFIG["conditional_actor_probe_update_cap_per_objective"],
            objective,
        )
    np.savez_compressed(OUT / "actor_probe_arrays.npz", **arrays)
    write_json(OUT / "actor_probe.json", result)
    print(json.dumps(plain(result), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=["prepare", "primary", "followup", "actor_probe"])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "primary":
        execute_primary()
    elif args.phase == "followup":
        execute_followup()
    else:
        execute_actor_probe()


if __name__ == "__main__":
    main()
