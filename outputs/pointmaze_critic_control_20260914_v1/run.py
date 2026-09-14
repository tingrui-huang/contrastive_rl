"""Bounded PointMaze critic-only A/B control with native fork data."""
from __future__ import annotations

import argparse
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
MATCHED_DIR = ROOT / "outputs" / "pointmaze_matched_fork_20260914_v1"
PILOT_DIR = ROOT / "outputs" / "pointmaze_oracle_training_pilot_20260914_v1"
ALIVE_DIR = ROOT / "outputs" / "pointmaze_oracle_alive_motion_20260914_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ALIVE_DIR))

from crl import checkpoint  # noqa: E402
from ett.fixed_actor_continuation import TimedSimulator  # noqa: E402
from ett.rollout_return import GOAL, START  # noqa: E402
from oracle_motion import emit_f4, hazardous, native_alive_motion  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MATCHED = load_module("critic_control_matched", MATCHED_DIR / "run.py")
PILOT = MATCHED.PILOT


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


def tree_sha(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source_paths():
    return [OUT / name for name in ("config.json", "PROTOCOL.md", "run.py", "analyze.py", "verify.py")]


def dependency_paths():
    return [
        PILOT_DIR / "checkpoints" / "C_s1_final.pkl",
        PILOT_DIR / "run.py",
        PILOT_DIR / "learner_config.json",
        MATCHED_DIR / "run.py",
        MATCHED_DIR / "root_selection.npz",
        MATCHED_DIR / "primary_model_traces.npz",
        ALIVE_DIR / "oracle_motion.py",
        ROOT / "crl" / "losses.py",
        ROOT / "crl" / "networks.py",
        ROOT / "crl" / "checkpoint.py",
        ROOT / "ett" / "fixed_actor_continuation.py",
    ]


def frozen_components():
    cfg, network, _ = PILOT.learner_setup()
    step, state = checkpoint.load_checkpoint(PILOT_DIR / "checkpoints" / "C_s1_final.pkl")
    nominal, heads, state_mean, state_std = PILOT.oracle_components()
    if step != 151000 or cfg.batch_size != CONFIG["critic_batch_size"]:
        raise RuntimeError("frozen learner contract changed")
    if cfg.use_td or cfg.use_cpc or cfg.use_gcbc or cfg.twin_q:
        raise RuntimeError("expected single-head Monte Carlo sigmoid NCE")
    if cfg.discount != CONFIG["discount"] or cfg.learning_rate != CONFIG["critic_learning_rate"]:
        raise RuntimeError("frozen learner hyperparameters changed")
    return cfg, network, step, state, nominal, heads[1], state_mean, state_std


def candidate_actions():
    return np.asarray(
        [CONFIG["candidate_actions"]["down"], CONFIG["candidate_actions"]["right"]],
        np.float32,
    )


def policy_action(network, params, states, epsilon):
    goal = np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)
    observation = jnp.asarray(np.concatenate([states, goal], axis=-1))
    distribution = network.policy_network.apply(params, observation)
    return np.asarray(jnp.tanh(distribution.loc + distribution.scale * jnp.asarray(epsilon)), np.float32)


def prepare():
    if (OUT / "preregistration.json").exists():
        raise RuntimeError("already prepared")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if current_head != CONFIG["source_commit"]:
        raise RuntimeError(f"expected {CONFIG['source_commit']}, found {current_head}")
    cfg, _, step, state, nominal, head, state_mean, state_std = frozen_components()
    rng = np.random.default_rng(CONFIG["discovery_actor_seed"])
    discovery_streams = {
        "reset_seed": np.arange(CONFIG["discovery_episode_count"], dtype=np.int64)
        + CONFIG["discovery_reset_seed_base"],
        "actor_epsilon": rng.standard_normal(
            (CONFIG["discovery_episode_count"], CONFIG["horizon"], 2)
        ).astype(np.float32),
    }
    np.savez_compressed(OUT / "discovery_streams.npz", **discovery_streams)
    seal = {
        "status": "sealed_before_new_native_outcomes",
        "utc": utc(),
        "source_commit": current_head,
        "branch": branch,
        "source_sha256": {str(path): sha256(path) for path in source_paths()},
        "dependency_sha256": {str(path): sha256(path) for path in dependency_paths()},
        "discovery_streams_sha256": sha256(OUT / "discovery_streams.npz"),
        "checkpoint_step": step,
        "checkpoint_state_sha256": tree_sha(state),
        "actor_sha256": tree_sha(state.policy_params),
        "critic_sha256": tree_sha(state.q_params),
        "critic_optimizer_sha256": tree_sha(state.q_optimizer_state),
        "nominal_sha256": tree_sha(nominal.params),
        "head_sha256": tree_sha(head),
        "head_scaling_sha256": tree_sha((state_mean, state_std)),
        "learner_config": {
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "discount": cfg.discount,
            "use_td": cfg.use_td,
            "use_cpc": cfg.use_cpc,
            "twin_q": cfg.twin_q,
        },
        "device": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
    }
    write_json(OUT / "preregistration.json", seal)
    print(json.dumps({"status": seal["status"], "branch": branch, "head": current_head}, indent=2))


def verify_initial_seal():
    seal = load_json(OUT / "preregistration.json")
    if seal["status"] != "sealed_before_new_native_outcomes":
        raise RuntimeError("invalid initial seal")
    verify_source_hashes(seal["source_sha256"])
    for path, expected in seal["dependency_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"dependency changed: {path}")
    if sha256(OUT / "discovery_streams.npz") != seal["discovery_streams_sha256"]:
        raise RuntimeError("discovery streams changed")
    return seal


def verify_source_hashes(expected_hashes):
    mismatches = {
        path: {"old": expected, "new": sha256(path)}
        for path, expected in expected_hashes.items()
        if sha256(path) != expected
    }
    if not mismatches:
        return
    amendment_path = OUT / "training_seal_amendment.json"
    if not amendment_path.exists():
        raise RuntimeError(f"unsealed source changes: {mismatches}")
    amendment = load_json(amendment_path)
    if amendment["status"] != "sealed_after_zero_update_shape_fix_before_training_restart":
        raise RuntimeError("invalid source amendment status")
    if amendment["preregistration_sha256"] != sha256(OUT / "preregistration.json"):
        raise RuntimeError("amendment does not bind the initial seal")
    if amendment["training_seal_sha256"] != sha256(OUT / "training_seal.json"):
        raise RuntimeError("amendment does not bind the training seal")
    if amendment["critic_updates_before_failure"] != 0:
        raise RuntimeError("source amendment permitted only after zero updates")
    if mismatches != amendment["changed_sources"]:
        raise RuntimeError(f"source changes exceed sealed amendment: {mismatches}")


def collect_discovery():
    verify_initial_seal()
    if (OUT / "discovery_native.npz").exists():
        raise RuntimeError("discovery already exists")
    started = time.monotonic()
    streams = load_npz(OUT / "discovery_streams.npz")
    _, network, _, state, _, _, _, _ = frozen_components()
    simulators = [TimedSimulator(int(seed)) for seed in streams["reset_seed"]]
    current = np.stack([sim.observation()[:8] for sim in simulators]).astype(np.float32)
    np.testing.assert_array_equal(current, np.broadcast_to(np.asarray(START, np.float32), current.shape))
    states = [current.copy()]
    physical = [np.stack([sim.env.state.copy() for sim in simulators])]
    actions, rewards, failed_before, failed_after, hazard = [], [], [], [], []
    for step_index in range(CONFIG["horizon"]):
        before = np.asarray([sim.env.dead for sim in simulators], bool)
        action = policy_action(
            network, state.policy_params, current, streams["actor_epsilon"][:, step_index]
        )
        outcomes = [sim.step(one) for sim, one in zip(simulators, action)]
        current = np.stack([outcome[0][:8] for outcome in outcomes]).astype(np.float32)
        position = np.stack([sim.env.state.copy() for sim in simulators])
        reward = np.asarray([outcome[1] for outcome in outcomes], np.float32)
        after = np.asarray([sim.env.dead for sim in simulators], bool)
        states.append(current.copy())
        physical.append(position.copy())
        actions.append(action)
        rewards.append(reward)
        failed_before.append(before)
        failed_after.append(after)
        hazard.append(hazardous(position))
    record = {
        "reset_seed": streams["reset_seed"],
        "states": np.stack(states, axis=1),
        "physical_states": np.stack(physical, axis=1),
        "action": np.stack(actions, axis=1),
        "reward": np.stack(rewards, axis=1),
        "failed_before": np.stack(failed_before, axis=1),
        "failed_after": np.stack(failed_after, axis=1),
        "hazard_landing": np.stack(hazard, axis=1),
    }
    np.testing.assert_array_equal(record["states"][:, 1:, 2:], record["states"][:, :-1, :6])
    np.savez_compressed(OUT / "discovery_native.npz", **record)
    write_json(
        OUT / "discovery_execution.json",
        {
            "utc": utc(),
            "episodes": len(simulators),
            "native_steps": len(simulators) * CONFIG["horizon"],
            "runtime_seconds": time.monotonic() - started,
            "actor_updates": 0,
            "model_updates": 0,
        },
    )
    print(f"saved {len(simulators)} complete native discovery episodes")


def reconstruct_discovery_root(data, episode, time_index):
    simulator = TimedSimulator(int(data["reset_seed"][episode]))
    for step_index in range(int(time_index)):
        observation, reward, _ = simulator.step(data["action"][episode, step_index])
        np.testing.assert_array_equal(observation[:8], data["states"][episode, step_index + 1])
        np.testing.assert_array_equal(simulator.env.state, data["physical_states"][episode, step_index + 1])
        if reward != float(data["reward"][episode, step_index]):
            raise AssertionError("discovery replay reward mismatch")
    if simulator.elapsed != int(time_index) or simulator.env.dead:
        raise AssertionError("root replay did not end alive at requested time")
    return simulator


def select_roots(data):
    state = data["states"][:, :-1]
    xy = state[:, :, :2]
    geometry = CONFIG["fork_geometry"]
    previous_hazard = np.concatenate(
        [np.zeros((len(data["hazard_landing"]), 1), bool),
         np.maximum.accumulate(data["hazard_landing"], axis=1)[:, :-1]],
        axis=1,
    )
    times = np.arange(CONFIG["horizon"])[None]
    mask = (
        (xy[:, :, 0] >= geometry["x_min_inclusive"])
        & (xy[:, :, 0] < geometry["x_max_exclusive"])
        & (xy[:, :, 1] >= geometry["y_min_inclusive"])
        & (xy[:, :, 1] < geometry["y_max_exclusive"])
        & (~data["failed_before"])
        & (~previous_hazard)
        & ((CONFIG["horizon"] - times) >= geometry["minimum_remaining_steps"])
    )
    candidate = candidate_actions()
    rows = []
    for episode in range(len(state)):
        for time_index in np.flatnonzero(mask[episode]):
            position = data["physical_states"][episode, time_index][None]
            cells = []
            for action in candidate:
                moved = native_alive_motion(position, action[None], np.zeros((1, 2), np.float64))[0][0]
                cells.append(np.floor(moved).astype(np.int64))
            expected = np.asarray(
                [CONFIG["candidate_exit_cells"]["down"], CONFIG["candidate_exit_cells"]["right"]]
            )
            if np.array_equal(np.asarray(cells), expected):
                rows.append((episode, int(time_index)))
                break
        if len(rows) == CONFIG["root_target"]:
            break
    if len(rows) != CONFIG["root_target"]:
        raise RuntimeError(f"only {len(rows)} eligible roots; required {CONFIG['root_target']}")
    return rows


def prepare_forks():
    verify_initial_seal()
    if (OUT / "fork_preregistration.json").exists():
        raise RuntimeError("forks already prepared")
    data = load_npz(OUT / "discovery_native.npz")
    rows = select_roots(data)
    episodes = np.asarray([row[0] for row in rows], np.int64)
    times = np.asarray([row[1] for row in rows], np.int64)
    root_state = data["states"][episodes, times]
    root_physical = data["physical_states"][episodes, times]
    root_bits = []
    for episode, time_index in rows:
        simulator = reconstruct_discovery_root(data, episode, time_index)
        np.testing.assert_array_equal(simulator.observation()[:8], data["states"][episode, time_index])
        root_bits.append(simulator.env.swamp_bits)
    split_rng = np.random.default_rng(CONFIG["split_seed"])
    order = split_rng.permutation(CONFIG["root_target"])
    train = np.sort(order[: CONFIG["train_root_count"]])
    validation = np.sort(order[CONFIG["train_root_count"] :])
    np.savez_compressed(
        OUT / "roots_and_split.npz",
        discovery_episode=episodes,
        reset_seed=data["reset_seed"][episodes],
        time=times,
        state=root_state,
        physical_state=root_physical,
        root_bits=np.asarray(root_bits, bool),
        train_root=train,
        validation_root=validation,
    )

    nr = CONFIG["root_target"]
    native_repeats = CONFIG["native_paths_per_root_action"]
    model_repeats = CONFIG["model_paths_per_root_action"]
    horizon = CONFIG["horizon"]
    native_rng = np.random.default_rng(CONFIG["native_fork_seed"])
    model_rng = np.random.default_rng(CONFIG["model_fork_seed"])
    native_epsilon = native_rng.standard_normal((nr, native_repeats, horizon, 2)).astype(np.float32)
    native_noise = native_rng.normal(
        0.0, CONFIG["motion_noise_standard_deviation"], (nr, native_repeats, horizon, 2)
    )
    native_bits = native_rng.random((nr, native_repeats, horizon, 3)) < CONFIG["active_probability"]
    model_epsilon = model_rng.standard_normal((nr, model_repeats, horizon, 2)).astype(np.float32)
    model_noise = model_rng.normal(
        0.0, CONFIG["motion_noise_standard_deviation"], (nr, model_repeats, horizon, 2)
    )
    model_epsilon[:, :native_repeats] = native_epsilon
    model_noise[:, :native_repeats] = native_noise
    onset_uniform = model_rng.random((nr, model_repeats, horizon)).astype(np.float32)
    nominal_key = np.stack(
        [np.asarray(jax.random.PRNGKey(CONFIG["model_fork_seed"] + 1000 + step)) for step in range(horizon)]
    )
    np.savez_compressed(
        OUT / "fork_streams.npz",
        native_actor_epsilon=native_epsilon,
        native_motion_noise=native_noise,
        native_bits_after=native_bits,
        model_actor_epsilon=model_epsilon,
        model_motion_noise=model_noise,
        model_onset_uniform=onset_uniform,
        model_nominal_key=nominal_key,
    )
    seal = {
        "status": "sealed_after_root_split_before_any_fork_outcome",
        "utc": utc(),
        "initial_seal_sha256": sha256(OUT / "preregistration.json"),
        "discovery_native_sha256": sha256(OUT / "discovery_native.npz"),
        "roots_and_split_sha256": sha256(OUT / "roots_and_split.npz"),
        "fork_streams_sha256": sha256(OUT / "fork_streams.npz"),
        "eligible_roots": len(rows),
        "train_roots": len(train),
        "validation_roots": len(validation),
        "split_uses_outcomes": False,
        "old_reference_roots_used": False,
    }
    write_json(OUT / "fork_preregistration.json", seal)
    print(json.dumps(seal, indent=2))


def verify_fork_seal():
    verify_initial_seal()
    seal = load_json(OUT / "fork_preregistration.json")
    checks = {
        "discovery_native.npz": seal["discovery_native_sha256"],
        "roots_and_split.npz": seal["roots_and_split_sha256"],
        "fork_streams.npz": seal["fork_streams_sha256"],
    }
    for name, expected in checks.items():
        if sha256(OUT / name) != expected:
            raise RuntimeError(f"fork input changed: {name}")
    return seal


def blank_native_trace(root_count, repeats, max_horizon):
    path = (root_count, 2, repeats, max_horizon)
    return {
        "valid": np.zeros(path, bool),
        "states": np.full(path[:-1] + (max_horizon + 1, 8), np.nan, np.float32),
        "physical_states": np.full(path[:-1] + (max_horizon + 1, 2), np.nan, np.float64),
        "action": np.full(path + (2,), np.nan, np.float32),
        "reward": np.zeros(path, np.float32),
        "failed_before": np.zeros(path, bool),
        "failed_after": np.zeros(path, bool),
        "hazard_landing": np.zeros(path, bool),
        "swamp_bits_before": np.zeros(path + (3,), bool),
        "swamp_bits_after": np.zeros(path + (3,), bool),
    }


def native_fork_rollouts(data, roots, streams, network, policy_params):
    repeats = CONFIG["native_paths_per_root_action"]
    remaining = CONFIG["horizon"] - roots["time"]
    max_horizon = int(remaining.max())
    trace = blank_native_trace(len(roots["time"]), repeats, max_horizon)
    snapshots = [
        reconstruct_discovery_root(data, int(ep), int(t)).snapshot()
        for ep, t in zip(roots["discovery_episode"], roots["time"])
    ]
    candidate = candidate_actions()
    for action_index in range(2):
        simulators = [
            TimedSimulator.restore(snapshots[root])
            for root in range(len(snapshots))
            for _ in range(repeats)
        ]
        trace["states"][:, action_index, :, 0] = np.repeat(roots["state"], repeats, axis=0).reshape(
            len(roots["time"]), repeats, 8
        )
        trace["physical_states"][:, action_index, :, 0] = np.repeat(
            roots["physical_state"], repeats, axis=0
        ).reshape(len(roots["time"]), repeats, 2)
        for local_step in range(max_horizon):
            current = np.stack([sim.observation()[:8] for sim in simulators]).astype(np.float32)
            if local_step == 0:
                action = np.broadcast_to(candidate[action_index], (len(simulators), 2)).copy()
            else:
                epsilon = streams["native_actor_epsilon"][:, :, local_step].reshape(-1, 2)
                action = policy_action(network, policy_params, current, epsilon)
            for flat_index, simulator in enumerate(simulators):
                root_index, repeat_index = divmod(flat_index, repeats)
                if local_step >= remaining[root_index]:
                    continue
                before = bool(simulator.env.dead)
                observation, reward, bits_before, bits_after = MATCHED.scheduled_step(
                    simulator,
                    action[flat_index],
                    streams["native_motion_noise"][root_index, repeat_index, local_step],
                    streams["native_bits_after"][root_index, repeat_index, local_step],
                )
                idx = (root_index, action_index, repeat_index, local_step)
                trace["valid"][idx] = True
                trace["action"][idx] = action[flat_index]
                trace["reward"][idx] = reward
                trace["failed_before"][idx] = before
                trace["failed_after"][idx] = simulator.env.dead
                trace["hazard_landing"][idx] = hazardous(simulator.env.state[None])[0]
                trace["swamp_bits_before"][idx] = bits_before
                trace["swamp_bits_after"][idx] = bits_after
                trace["states"][root_index, action_index, repeat_index, local_step + 1] = observation[:8]
                trace["physical_states"][root_index, action_index, repeat_index, local_step + 1] = simulator.env.state
        print(f"native fork action {action_index + 1}/2 complete", flush=True)
    weights = CONFIG["discount"] ** np.arange(max_horizon)
    trace["discounted_return"] = np.sum(trace["reward"] * trace["valid"] * weights, axis=-1)
    trace["root_time"] = roots["time"]
    trace["action_name"] = np.asarray(["down", "right"])
    return trace


def model_fork_rollouts(roots, streams, network, policy_params, nominal, head, state_mean, state_std):
    repeats = CONFIG["model_paths_per_root_action"]
    remaining = CONFIG["horizon"] - roots["time"]
    max_horizon = int(remaining.max())
    path = (len(roots["time"]), 2, repeats, max_horizon)
    trace = {
        "valid": np.zeros(path, bool),
        "states": np.full(path[:-1] + (max_horizon + 1, 8), np.nan, np.float32),
        "physical_states": np.full(path[:-1] + (max_horizon + 1, 2), np.nan, np.float64),
        "action": np.full(path + (2,), np.nan, np.float32),
        "reward": np.zeros(path, np.float32),
        "failed_before": np.zeros(path, bool),
        "failed_after": np.zeros(path, bool),
        "failure_probability": np.full(path, np.nan, np.float32),
        "onset": np.zeros(path, bool),
    }
    candidate = candidate_actions()
    head = jnp.asarray(head, jnp.float32)
    mean = jnp.asarray(state_mean, jnp.float32)
    std = jnp.asarray(state_std, jnp.float32)
    predict = jax.jit(lambda features: jax.nn.sigmoid(PILOT.REPAIR_COMMON.PARENT.head_logits(head, features)))
    for action_index in range(2):
        state = np.repeat(roots["state"], repeats, axis=0).astype(np.float32)
        physical = np.repeat(roots["physical_state"], repeats, axis=0).astype(np.float64)
        failed = np.zeros(len(state), bool)
        trace["states"][:, action_index, :, 0] = state.reshape(len(roots["time"]), repeats, 8)
        trace["physical_states"][:, action_index, :, 0] = physical.reshape(len(roots["time"]), repeats, 2)
        goal = np.broadcast_to(np.asarray(GOAL, np.float32), state.shape)
        for local_step in range(max_horizon):
            if local_step == 0:
                action = np.broadcast_to(candidate[action_index], (len(state), 2)).copy()
            else:
                epsilon = streams["model_actor_epsilon"][:, :, local_step].reshape(-1, 2)
                action = policy_action(network, policy_params, state, epsilon)
            xb = np.asarray(
                nominal.sample(
                    jnp.asarray(state), jnp.asarray(streams["model_nominal_key"][local_step]), 1,
                    goal=jnp.asarray(goal),
                ),
                np.float32,
            )
            noise = streams["model_motion_noise"][:, :, local_step].reshape(-1, 2)
            moved, _, _ = native_alive_motion(physical, action, noise)
            next_physical = np.where(failed[:, None], physical, moved)
            next_state = emit_f4(next_physical, state)
            support = hazardous(next_physical)
            displacement = jnp.asarray(next_state[:, :2]) - jnp.asarray(state[:, :2])
            features = jnp.concatenate(
                [(jnp.asarray(state) - mean) / std, jnp.asarray(xb), jnp.asarray(action), displacement], axis=-1
            )
            raw_probability = np.asarray(predict(features), np.float32)
            probability = np.where(failed, 0.0, np.where(support, raw_probability, 0.0)).astype(np.float32)
            onset_uniform = streams["model_onset_uniform"][:, :, local_step].reshape(-1)
            onset = (~failed) & (onset_uniform < probability)
            next_failed = failed | onset
            reward = (
                (np.linalg.norm(next_physical - np.asarray(CONFIG["task_goal_xy"]), axis=-1)
                 < CONFIG["reward_radius"])
                & (~next_failed)
            ).astype(np.float32)
            valid = local_step < np.repeat(remaining, repeats)
            idx = (slice(None), action_index, slice(None), local_step)
            trace["valid"][idx] = valid.reshape(len(roots["time"]), repeats)
            for name, value in {
                "action": action,
                "reward": reward,
                "failed_before": failed,
                "failed_after": next_failed,
                "failure_probability": probability,
                "onset": onset,
            }.items():
                trace[name][idx] = value.reshape((len(roots["time"]), repeats) + value.shape[1:])
            trace["states"][:, action_index, :, local_step + 1] = next_state.reshape(
                len(roots["time"]), repeats, 8
            )
            trace["physical_states"][:, action_index, :, local_step + 1] = next_physical.reshape(
                len(roots["time"]), repeats, 2
            )
            state, physical, failed = next_state, next_physical, next_failed
        print(f"model fork action {action_index + 1}/2 complete", flush=True)
    weights = CONFIG["discount"] ** np.arange(max_horizon)
    trace["discounted_return"] = np.sum(trace["reward"] * trace["valid"] * weights, axis=-1)
    trace["root_time"] = roots["time"]
    trace["action_name"] = np.asarray(["down", "right"])
    return trace


def sample_future(rng, anchor, length):
    offsets = np.arange(1, length - anchor + 1)
    weights = CONFIG["discount"] ** offsets
    return int(anchor + rng.choice(offsets, p=weights / weights.sum()))


def make_nce_indices(roots, split_roots, updates, seed, validation=False):
    rng = np.random.default_rng(seed)
    repeats = CONFIG["native_paths_per_root_action"]
    paths = [(int(root), action, repeat) for root in split_roots for action in range(2) for repeat in range(repeats)]
    batch_size = len(paths) if validation else CONFIG["critic_batch_size"]
    if len(paths) != batch_size:
        raise AssertionError((len(paths), batch_size))
    by_action = {action: [i for i, path in enumerate(paths) if path[1] == action] for action in range(2)}
    anchor_each = 32 if validation else CONFIG["root_anchor_rows_per_batch_per_action"]
    arrays = {name: np.zeros((updates, batch_size), np.int16) for name in ("root", "action", "repeat", "anchor", "future")}
    for update in range(updates):
        root_anchored = set()
        for action in range(2):
            chosen = rng.choice(by_action[action], anchor_each, replace=False)
            root_anchored.update(int(value) for value in chosen)
        order = rng.permutation(batch_size)
        for output_row, path_index in enumerate(order):
            root, action, repeat = paths[int(path_index)]
            length = int(CONFIG["horizon"] - roots["time"][root])
            anchor = 0 if int(path_index) in root_anchored else int(rng.integers(1, length))
            arrays["root"][update, output_row] = root
            arrays["action"][update, output_row] = action
            arrays["repeat"][update, output_row] = repeat
            arrays["anchor"][update, output_row] = anchor
            arrays["future"][update, output_row] = sample_future(rng, anchor, length)
    return arrays


def generate_forks_and_training_inputs():
    verify_fork_seal()
    if (OUT / "native_fork_traces.npz").exists():
        raise RuntimeError("fork outcomes already exist")
    started = time.monotonic()
    data = load_npz(OUT / "discovery_native.npz")
    roots = load_npz(OUT / "roots_and_split.npz")
    streams = load_npz(OUT / "fork_streams.npz")
    _, network, _, state, nominal, head, state_mean, state_std = frozen_components()
    native = native_fork_rollouts(data, roots, streams, network, state.policy_params)
    np.savez_compressed(OUT / "native_fork_traces.npz", **native)
    model = model_fork_rollouts(
        roots, streams, network, state.policy_params, nominal, head, state_mean, state_std
    )
    np.savez_compressed(OUT / "model_fork_traces.npz", **model)
    model_gap = model["discounted_return"][:, 0].mean(axis=1) - model["discounted_return"][:, 1].mean(axis=1)
    label = np.sign(model_gap).astype(np.int8)
    if np.any(label == 0):
        raise RuntimeError("model-return tie has no preregistered ranking rule")
    native_gap = native["discounted_return"][:, 0].mean(axis=1) - native["discounted_return"][:, 1].mean(axis=1)
    np.savez_compressed(
        OUT / "ranking_labels.npz",
        model_return_gap=model_gap,
        native_return_gap=native_gap,
        preferred_sign=label,
    )
    train_indices = make_nce_indices(
        roots, roots["train_root"], CONFIG["critic_updates_per_condition"], CONFIG["nce_train_seed"]
    )
    validation_indices = make_nce_indices(
        roots, roots["validation_root"], CONFIG["validation_nce_batches"],
        CONFIG["nce_validation_seed"], validation=True
    )
    np.savez_compressed(OUT / "nce_train_indices.npz", **train_indices)
    np.savez_compressed(OUT / "nce_validation_indices.npz", **validation_indices)
    execution = {
        "utc": utc(),
        "native_fork_paths": int(CONFIG["root_target"] * 2 * CONFIG["native_paths_per_root_action"]),
        "native_fork_transition_slots": int(native["valid"].sum()),
        "model_fork_paths": int(CONFIG["root_target"] * 2 * CONFIG["model_paths_per_root_action"]),
        "model_fork_transition_slots": int(model["valid"].sum()),
        "runtime_seconds": time.monotonic() - started,
        "actor_updates": 0,
        "model_updates": 0,
    }
    write_json(OUT / "fork_execution.json", execution)
    names = [
        "native_fork_traces.npz", "model_fork_traces.npz", "ranking_labels.npz",
        "nce_train_indices.npz", "nce_validation_indices.npz", "fork_execution.json",
    ]
    seal = {
        "status": "sealed_common_data_and_labels_before_critic_training",
        "utc": utc(),
        "fork_preregistration_sha256": sha256(OUT / "fork_preregistration.json"),
        "input_sha256": {name: sha256(OUT / name) for name in names},
        "source_sha256": {str(path): sha256(path) for path in source_paths()},
        "A_and_B_share_identical_nce_indices": True,
        "rank_labels_computed_from_model_returns": True,
        "native_paths_are_common_nce_data": True,
    }
    write_json(OUT / "training_seal.json", seal)
    print(json.dumps(execution, indent=2))


def verify_training_seal():
    verify_fork_seal()
    seal = load_json(OUT / "training_seal.json")
    for name, expected in seal["input_sha256"].items():
        if sha256(OUT / name) != expected:
            raise RuntimeError(f"training input changed: {name}")
    verify_source_hashes(seal["source_sha256"])
    return seal


def nce_batch(trace, indices, update):
    root = indices["root"][update].astype(np.int64)
    action_index = indices["action"][update].astype(np.int64)
    repeat = indices["repeat"][update].astype(np.int64)
    anchor = indices["anchor"][update].astype(np.int64)
    future = indices["future"][update].astype(np.int64)
    state = trace["states"][root, action_index, repeat, anchor]
    goal = trace["states"][root, action_index, repeat, future]
    action = trace["action"][root, action_index, repeat, anchor]
    return np.concatenate([state, goal], axis=1).astype(np.float32), action.astype(np.float32)


def matched_scores(network, q_params, states):
    states = np.asarray(states, np.float32)
    goal = np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)
    observation = jnp.asarray(np.concatenate([states, goal], axis=1))
    values = []
    for action in candidate_actions():
        batch_action = jnp.asarray(np.broadcast_to(action, (len(states), 2)))
        phi, psi = network.representation_network.apply(q_params, observation, batch_action)
        value = jnp.sum(phi * psi, axis=1)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise AssertionError("expected single critic head")
        values.append(np.asarray(value, np.float32))
    return np.stack(values, axis=1)


def train_critics():
    verify_training_seal()
    if (OUT / "training_results.json").exists():
        raise RuntimeError("training already complete")
    started = time.monotonic()
    cfg, network, step, initial, nominal, head, state_mean, state_std = frozen_components()
    frozen_aux_hash = tree_sha((initial.policy_params, nominal.params, head, state_mean, state_std))
    roots = load_npz(OUT / "roots_and_split.npz")
    labels = load_npz(OUT / "ranking_labels.npz")["preferred_sign"].astype(np.float32)
    trace = load_npz(OUT / "native_fork_traces.npz")
    train_indices = load_npz(OUT / "nce_train_indices.npz")
    validation_indices = load_npz(OUT / "nce_validation_indices.npz")
    reference_roots = load_npz(MATCHED_DIR / "root_selection.npz")["state"]
    reference_model = load_npz(MATCHED_DIR / "primary_model_traces.npz")["reward"]
    reference_valid = load_npz(MATCHED_DIR / "primary_model_traces.npz")["valid"]
    weights = CONFIG["discount"] ** np.arange(reference_model.shape[-1])
    reference_return = np.sum(reference_model * reference_valid * weights, axis=-1).mean(axis=2)
    reference_label = np.sign(reference_return[:, 0] - reference_return[:, 1]).astype(np.float32)
    if np.any(reference_label == 0):
        raise RuntimeError("old reference model tie")

    eye = jnp.eye(CONFIG["critic_batch_size"], dtype=jnp.float32)
    train_states = jnp.asarray(roots["state"][roots["train_root"]])
    train_labels = jnp.asarray(labels[roots["train_root"]])
    task_goal = jnp.broadcast_to(jnp.asarray(GOAL, jnp.float32), train_states.shape)
    rank_observation = jnp.concatenate([train_states, task_goal], axis=1)
    down_action = jnp.broadcast_to(jnp.asarray(candidate_actions()[0]), (len(train_states), 2))
    right_action = jnp.broadcast_to(jnp.asarray(candidate_actions()[1]), (len(train_states), 2))

    def objective(q_params, observation, action, rank_weight):
        logits = network.q_network.apply(q_params, observation, action)
        nce = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=eye))
        down_logits = jnp.diag(network.q_network.apply(q_params, rank_observation, down_action))
        right_logits = jnp.diag(network.q_network.apply(q_params, rank_observation, right_action))
        gap = down_logits - right_logits
        rank = jnp.mean(jax.nn.softplus(-train_labels * gap))
        return nce + rank_weight * rank, (nce, rank, jnp.mean(train_labels * gap))

    grad_fn = jax.jit(jax.value_and_grad(objective, has_aux=True))
    q_optimizer = optax.adam(CONFIG["critic_learning_rate"])
    eval_steps = set(CONFIG["evaluation_steps"])
    conditions = {
        "A_nce_only": CONFIG["rank_weight_A"],
        "B_nce_plus_rank": CONFIG["rank_weight_B"],
    }
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    curves = {}
    evaluations = []

    def heldout_nce(q_params):
        values = []
        for batch_index in range(CONFIG["validation_nce_batches"]):
            observation, action = nce_batch(trace, validation_indices, batch_index)
            logits = network.q_network.apply(q_params, jnp.asarray(observation), jnp.asarray(action))
            target = jnp.eye(len(observation), dtype=jnp.float32)
            values.append(float(jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=target))))
        return float(np.mean(values))

    def evaluate(label_name, update_index, q_params):
        train_score = matched_scores(network, q_params, roots["state"][roots["train_root"]])
        validation_score = matched_scores(network, q_params, roots["state"][roots["validation_root"]])
        reference_score = matched_scores(network, q_params, reference_roots)
        evaluations.append({
            "condition": label_name,
            "update": update_index,
            "train_gap": train_score[:, 0] - train_score[:, 1],
            "validation_gap": validation_score[:, 0] - validation_score[:, 1],
            "reference_gap": reference_score[:, 0] - reference_score[:, 1],
            "train_label": labels[roots["train_root"]],
            "validation_label": labels[roots["validation_root"]],
            "reference_label": reference_label,
            "heldout_native_nce_loss": heldout_nce(q_params),
        })

    for condition, rank_weight in conditions.items():
        q_params = initial.q_params
        target_q_params = initial.target_q_params
        optimizer_state = initial.q_optimizer_state
        rows = {name: [] for name in ("total_loss", "nce_loss", "rank_loss", "train_signed_gap", "grad_norm")}
        evaluate(condition, 0, q_params)
        for update_index in range(CONFIG["critic_updates_per_condition"]):
            observation, action = nce_batch(trace, train_indices, update_index)
            (loss_value, aux), gradient = grad_fn(
                q_params, jnp.asarray(observation), jnp.asarray(action), jnp.asarray(rank_weight, jnp.float32)
            )
            updates, optimizer_state = q_optimizer.update(gradient, optimizer_state, q_params)
            q_params = optax.apply_updates(q_params, updates)
            target_q_params = jax.tree.map(
                lambda old, new: old * (1 - cfg.tau) + new * cfg.tau, target_q_params, q_params
            )
            metrics = [loss_value, aux[0], aux[1], aux[2], optax.global_norm(gradient)]
            if not all(np.isfinite(float(value)) for value in metrics):
                raise FloatingPointError((condition, update_index, metrics))
            for name, value in zip(rows, metrics):
                rows[name].append(float(value))
            completed = update_index + 1
            if completed in eval_steps:
                evaluate(condition, completed, q_params)
            if completed % 100 == 0:
                print(condition, completed, "nce", round(float(aux[0]), 6),
                      "rank", round(float(aux[1]), 6), flush=True)
            if time.monotonic() - started > CONFIG["caps"]["runtime_seconds"]:
                raise TimeoutError("runtime cap exceeded")
        final_state = initial._replace(
            q_params=q_params,
            target_q_params=target_q_params,
            q_optimizer_state=optimizer_state,
        )
        checkpoint.save_named(str(checkpoint_dir), condition, step + CONFIG["critic_updates_per_condition"], final_state)
        if tree_sha(final_state.policy_params) != tree_sha(initial.policy_params):
            raise AssertionError("actor changed")
        curves[condition] = {name: np.asarray(value, np.float32) for name, value in rows.items()}

    arrays = {}
    for condition in conditions:
        selected = [row for row in evaluations if row["condition"] == condition]
        arrays[f"{condition}__update"] = np.asarray([row["update"] for row in selected], np.int32)
        for key in ("train_gap", "validation_gap", "reference_gap", "train_label", "validation_label", "reference_label"):
            arrays[f"{condition}__{key}"] = np.stack([row[key] for row in selected])
        arrays[f"{condition}__heldout_native_nce_loss"] = np.asarray(
            [row["heldout_native_nce_loss"] for row in selected], np.float32
        )
        for name, value in curves[condition].items():
            arrays[f"{condition}__curve_{name}"] = value
    np.savez_compressed(OUT / "training_curves_and_evaluations.npz", **arrays)
    result = {
        "status": "complete",
        "utc": utc(),
        "conditions": conditions,
        "updates_per_condition": CONFIG["critic_updates_per_condition"],
        "common_nce_batches": True,
        "common_native_fork_data": True,
        "starting_critic_sha256": tree_sha(initial.q_params),
        "starting_optimizer_sha256": tree_sha(initial.q_optimizer_state),
        "frozen_actor_sha256": tree_sha(initial.policy_params),
        "frozen_model_bundle_sha256": frozen_aux_hash,
        "final_checkpoints": {
            condition: {
                "path": f"checkpoints/{condition}.pkl",
                "sha256": sha256(checkpoint_dir / f"{condition}.pkl"),
                "critic_sha256": tree_sha(checkpoint.load_checkpoint(checkpoint_dir / f"{condition}.pkl")[1].q_params),
            }
            for condition in conditions
        },
        "actor_updates": 0,
        "model_updates": 0,
        "new_actor_training_episodes": 0,
        "checkpoint_searches": 0,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(OUT / "training_results.json", result)
    print(json.dumps(result, indent=2))


def preflight():
    cfg, network, step, state, _, _, _, _ = frozen_components()
    states = np.broadcast_to(np.asarray(START, np.float32), (4, 8)).copy()
    actions = np.zeros((4, 2), np.float32)
    goals = np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)
    logits = network.q_network.apply(state.q_params, jnp.asarray(np.concatenate([states, goals], axis=1)), jnp.asarray(actions))
    if logits.shape != (4, 4):
        raise AssertionError(logits.shape)
    print(json.dumps({"status": "passed", "checkpoint_step": step, "batch_size": cfg.batch_size,
                      "critic_shape": list(logits.shape)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "prepare", "discover", "prepare_forks", "forks", "train"))
    args = parser.parse_args()
    {
        "preflight": preflight,
        "prepare": prepare,
        "discover": collect_discovery,
        "prepare_forks": prepare_forks,
        "forks": generate_forks_and_training_inputs,
        "train": train_critics,
    }[args.stage]()


if __name__ == "__main__":
    main()
