"""Bounded end-to-end PointMaze CRL pilot with frozen oracle alive motion."""
from __future__ import annotations

import argparse
import dataclasses
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
ORACLE = ROOT / "outputs" / "pointmaze_oracle_alive_motion_20260914_v1"
REPAIR = ROOT / "outputs" / "pointmaze_supervised_repair_20260914_v1"
PERSISTENT = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
PRIOR_INTEGRATION = ROOT / "artifacts" / "pointmaze_region_pilot" / "full_f4_integration_s01_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORACLE))

from crl import checkpoint  # noqa: E402
from ett import pointmaze_native_policy as policy_tools  # noqa: E402
from ett import run_policy_improvement as original  # noqa: E402
from ett.fixed_actor_continuation import TimedSimulator  # noqa: E402
from ett.policy_improvement import SegmentReplay, mixed_batch, replay, tree_sha  # noqa: E402
from ett.rollout_return import GOAL, START  # noqa: E402
from oracle_motion import emit_f4, hazardous, native_alive_motion  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REPAIR_COMMON = load_module("pilot_repair_common", REPAIR / "common.py")


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


def learner_setup():
    cfg, network, update = original.learner_setup()
    cfg.max_number_of_steps = CONFIG["joint_updates_per_trained_policy"]
    assert cfg.obs_dim == cfg.goal_dim == 8 and cfg.action_dim == 2
    assert cfg.max_episode_steps == CONFIG["horizon"] and cfg.discount == CONFIG["discount"]
    assert cfg.batch_size == CONFIG["batch_size"]
    assert cfg.actor_learning_rate == CONFIG["actor_learning_rate"]
    assert cfg.learning_rate == CONFIG["critic_learning_rate"]
    assert not cfg.use_td and not cfg.use_cpc and not cfg.use_gcbc and not cfg.twin_q
    assert cfg.random_goals == 0.5 and cfg.bc_coef == 0.5 and cfg.entropy_coefficient == 0.0
    return cfg, network, update


def dependency_paths():
    inputs = original.inputs()
    return [
        Path(inputs["dataset"]),
        Path(inputs["nominal"]),
        Path(inputs["actor"]),
        Path(inputs["actor"]).with_name("arm_provenance.json"),
        REPAIR / "head_repaired_s0.npz",
        REPAIR / "head_repaired_s1.npz",
        REPAIR / "checkpoints_frozen.json",
        REPAIR / "common.py",
        PERSISTENT / "head_feature_scaling.npz",
        PERSISTENT / "run.py",
        ORACLE / "oracle_motion.py",
        ORACLE / "movement_equivalence.json",
        ORACLE / "completion.json",
        ORACLE / "REPORT.md",
        PRIOR_INTEGRATION / "PROTOCOL.md",
        PRIOR_INTEGRATION / "REPORT.md",
        ROOT / "crl" / "losses.py",
        ROOT / "crl" / "networks.py",
        ROOT / "crl" / "replay.py",
        ROOT / "crl" / "checkpoint.py",
        ROOT / "crl" / "envs.py",
        ROOT / "ett" / "policy_improvement.py",
        ROOT / "ett" / "run_policy_improvement.py",
        ROOT / "ett" / "pointmaze_native_policy.py",
        ROOT / "ett" / "fixed_actor_continuation.py",
    ]


def local_source_paths():
    return [
        OUT / "PROTOCOL.md",
        OUT / "config.json",
        OUT / "equivalent_experiment_search.json",
        OUT / "run.py",
        OUT / "analyze.py",
        OUT / "verify_saved.py",
    ]


class Ledger:
    CAPS = {
        "native_steps": CONFIG["native_step_cap"],
        "oracle_transition_slots": CONFIG["total_oracle_transition_cap"],
        "complete_model_paths": CONFIG["total_complete_model_path_cap"],
        "actor_updates": CONFIG["actor_update_cap"],
        "critic_updates": CONFIG["critic_update_cap"],
    }

    def __init__(self):
        path = OUT / "ledger.json"
        if path.exists():
            self.data = load_json(path)
        else:
            self.data = {name: {"counts": {}, "total": 0, "cap": cap} for name, cap in self.CAPS.items()}
            self.data["new_training_native_episodes"] = {"total": 0, "cap": 0}
            self.save()

    def charge(self, category, count, purpose):
        count = int(count)
        row = self.data[category]
        if purpose in row["counts"]:
            if row["counts"][purpose] != count:
                raise RuntimeError(f"ledger charge changed for {category}/{purpose}")
            return False
        if row["total"] + count > row["cap"]:
            raise RuntimeError(f"{category} cap exceeded")
        row["counts"][purpose] = count
        row["total"] += count
        self.save()
        return True

    def save(self):
        write_json(OUT / "ledger.json", self.data)


def oracle_components():
    engine = REPAIR_COMMON.PARENT.load_position_engine()
    scaling = load_npz(PERSISTENT / "head_feature_scaling.npz")
    heads = {seed: load_npz(REPAIR / f"head_repaired_s{seed}.npz")["theta"] for seed in (0, 1)}
    return engine.kernel.nominal, heads, scaling["state_mean"], scaling["state_std"]


class OracleGenerator:
    """Current-policy rollouts with exact native movement and a frozen repaired head."""

    def __init__(self, network, nominal, head, state_mean, state_std):
        self.network = network
        self.nominal = nominal
        self.head = jnp.asarray(head, jnp.float32)
        self.mean = jnp.asarray(state_mean, jnp.float32)
        self.std = jnp.asarray(state_std, jnp.float32)
        self.sample_actor = jax.jit(lambda params, state, key: policy_tools.sample_actor(network, params, state, key))
        self.predict_head = jax.jit(
            lambda features: jax.nn.sigmoid(REPAIR_COMMON.PARENT.head_logits(self.head, features))
        )

    def rollout(self, policy_params, initial_state, seed, horizon):
        initial_state = np.asarray(initial_state, np.float32)
        if initial_state.ndim != 2 or initial_state.shape[1] != 8:
            raise ValueError("oracle roots must be [paths,8] observable F4 states")
        paths = len(initial_state)
        state = initial_state.copy()
        physical = state[:, :2].astype(np.float64)
        failed = np.zeros(paths, bool)
        goal = np.broadcast_to(np.asarray(GOAL, np.float32), state.shape).copy()
        goal_physical = np.asarray(GOAL[:2], np.float64)
        step_keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(horizon))
        records = {name: [] for name in (
            "action", "xb", "motion_noise", "clipped_noisy_action", "blocked_updates",
            "physical_xy", "failed_before", "failed_after", "onset", "onset_uniform",
            "failure_probability", "raw_failure_probability", "hazard_landing", "reward",
        )}
        states = [state.copy()]
        physical_states = [physical.copy()]
        failure_states = [failed.copy()]
        for step in range(int(horizon)):
            nominal_key, actor_key, motion_key, onset_key = jax.random.split(step_keys[step], 4)
            state_j = jnp.asarray(state)
            goal_j = jnp.asarray(goal)
            xb = np.asarray(self.nominal.sample(state_j, nominal_key, 1, goal=goal_j), np.float32)
            action = np.asarray(self.sample_actor(policy_params, state_j, actor_key), np.float32)
            motion_noise = np.asarray(
                jax.random.normal(motion_key, (paths, 2), dtype=jnp.float32), np.float64
            ) * 0.01
            moved_physical, clipped_action, blocked_updates = native_alive_motion(physical, action, motion_noise)
            next_physical = np.where(failed[:, None], physical, moved_physical)
            next_state = emit_f4(next_physical, state)
            emitted_xy = next_state[:, :2]
            support = hazardous(emitted_xy)
            displacement = jnp.asarray(emitted_xy) - state_j[:, :2]
            features = jnp.concatenate(
                [(state_j - self.mean) / self.std, jnp.asarray(xb), jnp.asarray(action), displacement], axis=-1
            )
            raw_probability = np.asarray(self.predict_head(features), np.float32)
            if not np.isfinite(raw_probability).all():
                raise FloatingPointError("non-finite repaired-head probability")
            probability = np.where(failed, 0.0, np.where(support, raw_probability, 0.0)).astype(np.float32)
            onset_uniform = np.asarray(jax.random.uniform(onset_key, (paths,), dtype=jnp.float32), np.float32)
            onset = (~failed) & (onset_uniform < probability)
            next_failed = failed | onset
            reward = ((np.linalg.norm(next_physical - goal_physical, axis=-1) < 2) & (~next_failed)).astype(np.float32)
            if np.any(probability[~support] != 0) or np.any(onset & failed):
                raise AssertionError("failure support or irreversibility violated")
            if np.any(next_physical[failed] != physical[failed]):
                raise AssertionError("post-failure physical XY moved")
            np.testing.assert_array_equal(next_state[:, 2:], state[:, :6])
            if np.any(reward[next_failed] != 0):
                raise AssertionError("failed path received generated task reward")
            values = {
                "action": action,
                "xb": xb,
                "motion_noise": motion_noise,
                "clipped_noisy_action": clipped_action,
                "blocked_updates": blocked_updates,
                "physical_xy": next_physical,
                "failed_before": failed,
                "failed_after": next_failed,
                "onset": onset,
                "onset_uniform": onset_uniform,
                "failure_probability": probability,
                "raw_failure_probability": raw_probability,
                "hazard_landing": support,
                "reward": reward,
            }
            for name, value in values.items():
                records[name].append(np.asarray(value))
            state, physical, failed = next_state, next_physical, next_failed
            states.append(state.copy())
            physical_states.append(physical.copy())
            failure_states.append(failed.copy())
        result = {name: np.stack(value, axis=1) for name, value in records.items()}
        result["states"] = np.stack(states, axis=1)
        result["physical_states"] = np.stack(physical_states, axis=1)
        result["failure_states"] = np.stack(failure_states, axis=1)
        result["return"] = np.sum(
            result["reward"] * np.power(np.float32(CONFIG["discount"]), np.arange(horizon, dtype=np.float32)),
            axis=-1,
            dtype=np.float32,
        )
        return result


def synthetic_replay(records, seed):
    paths = sum(len(record["states"]) for record in records)
    if paths <= 0:
        raise ValueError("empty oracle records")
    buffer = SegmentReplay(paths * 51, 51, 16, 2, 8, 0, -1, CONFIG["discount"], seed)
    failed_states = np.zeros((paths, 51), bool)
    rewards = np.zeros((paths, 50), np.float32)
    lengths = np.zeros(paths, np.int64)
    cursor = 0
    for record in records:
        for k in range(len(record["states"])):
            states = record["states"][k]
            actions = record["action"][k]
            length = len(states)
            observation = np.full((51, 16), np.nan, np.float32)
            action = np.full((51, 2), np.nan, np.float32)
            observation[:length] = np.concatenate(
                [states, np.broadcast_to(np.asarray(GOAL, np.float32), states.shape)], axis=-1
            )
            action[: length - 1] = actions
            buffer.add_episode(observation, action, length=length)
            failed_states[cursor, :length] = record["failure_states"][k]
            rewards[cursor, : length - 1] = record["reward"][k]
            lengths[cursor] = length
            cursor += 1
    buffer.freeze()
    if cursor != paths or len(buffer) != int(np.sum(lengths - 1)):
        raise AssertionError("oracle replay path/transition count mismatch")
    return buffer, {"failed_states": failed_states, "reward": rewards, "lengths": lengths}


def adapter_verification_from_saved():
    saved = load_npz(ORACLE / "oracle_rollout_s0.npz")
    flat_failed = saved["failed_after"].reshape(-1, 50)
    candidates = np.flatnonzero(flat_failed.any(axis=1))[:8]
    if len(candidates) != 8:
        raise AssertionError("prior oracle arrays do not contain eight failed traces")
    states = saved["states"].reshape(-1, 51, 8)[candidates]
    physical = saved["physical_states"].reshape(-1, 51, 2)[candidates]
    action = saved["action"].reshape(-1, 50, 2)[candidates]
    failed_after = flat_failed[candidates]
    failure_states = np.concatenate([np.zeros((8, 1), bool), failed_after], axis=1)
    reward = saved["reward"].reshape(-1, 50)[candidates]
    record = {"states": states, "physical_states": physical, "action": action,
              "failure_states": failure_states, "reward": reward}
    buffer, audit = synthetic_replay([record], CONFIG["training_seed_base"] - 1)
    batch, index = buffer.sample_audited(4096)
    future_failed = audit["failed_states"][index["episode"], index["future"]]
    future_xy = batch.observation[:, 8:10]
    failed_task_goals = np.linalg.norm(future_xy[future_failed] - np.asarray(GOAL[:2]), axis=-1) < 2
    post_failed = failure_states[:, :-1]
    if np.any(np.diff(physical, axis=1)[post_failed] != 0):
        raise AssertionError("saved absorbing trace moved after failure")
    np.testing.assert_array_equal(states[:, 1:, 2:], states[:, :-1, :6])
    if np.any(reward[failed_after] != 0) or np.any(failed_task_goals):
        raise AssertionError("failed trace was treated as a task success")
    return {
        "status": "passed",
        "source": str(ORACLE / "oracle_rollout_s0.npz"),
        "saved_failed_paths_checked": 8,
        "sampled_replay_pairs": 4096,
        "strict_future_indices": bool(np.all(index["future"] > index["time"])),
        "padding_selected": int(np.sum(index["future"] >= audit["lengths"][index["episode"]])),
        "sampled_failed_future_goals": int(future_failed.sum()),
        "failed_future_goals_inside_commanded_reward_radius": int(failed_task_goals.sum()),
        "post_failure_xy_frozen": True,
        "f4_shift_correct": True,
        "reward_after_failure_zero": True,
        "failure_flag_in_actor_observation": False,
    }


def prepare():
    if (OUT / "preregistration.json").exists():
        raise RuntimeError("pilot already prepared")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_head != CONFIG["source_commit"]:
        raise RuntimeError(f"expected HEAD {CONFIG['source_commit']}, found {current_head}")
    subprocess.check_call(["git", "cat-file", "-e", CONFIG["source_commit"] + "^{commit}"], cwd=ROOT)
    equivalent = load_json(OUT / "equivalent_experiment_search.json")
    if equivalent["equivalent_completed_pilot_found"]:
        raise RuntimeError("equivalent completed pilot exists")
    movement = load_json(ORACLE / "movement_equivalence.json")
    completion = load_json(ORACLE / "completion.json")
    if movement["status"] != "passed" or movement["native_step_calls"] != 34:
        raise RuntimeError("verified native movement dependency is invalid")
    if not completion["pathwise_first_hazard_identity"]:
        raise RuntimeError("prior oracle completion contract is invalid")
    cfg, _, _ = learner_setup()
    inputs = original.inputs()
    step, initial = checkpoint.load_checkpoint(inputs["actor"])
    if step != 150000:
        raise RuntimeError("starting learner checkpoint step changed")
    observation, action, ids, heldout = original.dataset()
    if len(ids) != 5940 or len(heldout) != 660:
        raise RuntimeError("offline continuation partition changed")
    (OUT / "checkpoints").mkdir(exist_ok=False)
    (OUT / "rounds").mkdir(exist_ok=False)
    checkpoint.save_named(str(OUT / "checkpoints"), "A", step, initial)
    np.savez_compressed(OUT / "partition.npz", train=ids, heldout=heldout)
    bank_rng = np.random.default_rng(CONFIG["policy_bank_seed"])
    flat = observation.reshape(-1, 16)
    bank = flat[bank_rng.choice(len(flat), 1024, replace=False), :8]
    np.savez_compressed(OUT / "policy_bank.npz", states=bank)
    write_json(OUT / "learner_config.json", dataclasses.asdict(cfg))
    write_json(OUT / "adapter_verification.json", adapter_verification_from_saved())
    Ledger()
    nominal, heads, state_mean, state_std = oracle_components()
    dependency_hashes = {str(path): sha256(path) for path in dependency_paths()}
    source_hashes = {str(path): sha256(path) for path in local_source_paths()}
    sealed_names = [
        "config.json", "PROTOCOL.md", "equivalent_experiment_search.json", "learner_config.json",
        "partition.npz", "policy_bank.npz", "adapter_verification.json", "ledger.json", "checkpoints/A.pkl",
    ]
    preregistration = {
        "status": "sealed_before_training_or_new_native_evaluation",
        "utc": utc(),
        "source_commit": CONFIG["source_commit"],
        "working_head": current_head,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "repository_instruction_files": [],
        "dependency_sha256": dependency_hashes,
        "diagnostic_source_sha256": source_hashes,
        "sealed_output_sha256": {name: sha256(OUT / name) for name in sealed_names},
        "starting_checkpoint_step": step,
        "starting_state_tree_sha256": tree_sha(initial),
        "starting_actor_tree_sha256": tree_sha(initial.policy_params),
        "starting_critic_tree_sha256": tree_sha(initial.q_params),
        "nominal_tree_sha256": tree_sha(nominal.params),
        "head_tree_sha256": {f"s{seed}": tree_sha(heads[seed]) for seed in (0, 1)},
        "head_scaling_tree_sha256": tree_sha((state_mean, state_std)),
        "offline_training_episodes": len(ids),
        "heldout_episode_ids_not_used_for_training": len(heldout),
        "initial_checkpoint_saw_all_6600_episodes": True,
        "movement_equivalence_reused_native_calls": 34,
        "new_native_calls_during_prepare": 0,
        "device": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
    }
    write_json(OUT / "preregistration.json", preregistration)
    print("Protocol, hashes, adapter checks, initialization, and budgets sealed before training.", flush=True)


def verify_sealed():
    prereg = load_json(OUT / "preregistration.json")
    if prereg["status"] != "sealed_before_training_or_new_native_evaluation":
        raise RuntimeError("invalid preregistration")
    if load_json(OUT / "config.json") != CONFIG:
        raise RuntimeError("configuration changed")
    for path, expected in prereg["dependency_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"dependency changed: {path}")
    for path, expected in prereg["diagnostic_source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"pilot source changed: {path}")
    for name, expected in prereg["sealed_output_sha256"].items():
        if name == "ledger.json":
            continue
        if sha256(OUT / name) != expected:
            raise RuntimeError(f"sealed prepared output changed: {name}")
    return prereg


def generated_replay_audit(meta, index):
    future_failed = meta["failed_states"][index["episode"], index["future"]]
    anchor_failed = meta["failed_states"][index["episode"], index["time"]]
    future_xy = meta["buffer"]._obs[index["episode"], index["future"], :2]
    task_goal = np.linalg.norm(future_xy - np.asarray(GOAL[:2]), axis=-1) < 2
    if np.any(index["future"] <= index["time"]):
        raise AssertionError("non-future synthetic goal sampled")
    if np.any(index["future"] >= meta["lengths"][index["episode"]]):
        raise AssertionError("synthetic padding sampled")
    if np.any(task_goal & future_failed):
        raise AssertionError("failed achieved state became a commanded task success")
    return {
        "pairs": len(index["time"]),
        "anchor_failed": int(anchor_failed.sum()),
        "future_failed": int(future_failed.sum()),
        "future_task_goal": int(task_goal.sum()),
        "failed_future_task_goal": int((task_goal & future_failed).sum()),
    }


def train():
    prereg = verify_sealed()
    if (OUT / "training_started.json").exists():
        raise RuntimeError("existing training attempt protected")
    write_json(OUT / "training_started.json", {"utc": utc(), "preregistration_sha256": sha256(OUT / "preregistration.json")})
    started = time.monotonic()
    cfg, network, update = learner_setup()
    _, initial = checkpoint.load_checkpoint(original.inputs()["actor"])
    observation, action, ids, _ = original.dataset()
    bank = load_npz(OUT / "policy_bank.npz")["states"]
    change_fn = jax.jit(lambda params: policy_tools.policy_change(network, initial.policy_params, params, bank))
    nominal, heads, state_mean, state_std = oracle_components()
    frozen_hash = tree_sha((nominal.params, heads, state_mean, state_std))
    if tree_sha(initial) != prereg["starting_state_tree_sha256"]:
        raise RuntimeError("initial learner state changed")
    ledger = Ledger()
    runs = {}
    curve_arrays = {}
    for seed in CONFIG["training_seeds"]:
        initial_hashes = []
        offline_hashes = []
        permutation_hashes = []
        for condition in CONFIG["conditions"]:
            label = ("B" if condition == "B_offline_only" else "C") + f"_s{seed}"
            state = initial._replace(key=jax.random.PRNGKey(CONFIG["training_seed_base"] + seed))
            initial_hashes.append(tree_sha(state))
            offline = replay(observation, action, CONFIG["training_seed_base"] + seed + 10)
            order_rng = np.random.default_rng(CONFIG["training_seed_base"] + seed + 20)
            root_rng = np.random.default_rng(CONFIG["training_root_seed_base"] + seed)
            generator = OracleGenerator(network, nominal, heads[seed], state_mean, state_std)
            synthetic = None
            synthetic_meta = None
            rows = []
            audit_rows = []
            refresh_records = []
            total_synthetic = 0
            replay_audit = {"pairs": 0, "anchor_failed": 0, "future_failed": 0,
                            "future_task_goal": 0, "failed_future_task_goal": 0}
            offline_digest = hashlib.sha256()
            permutation_digest = hashlib.sha256()
            ledger.charge("actor_updates", CONFIG["joint_updates_per_trained_policy"], label)
            ledger.charge("critic_updates", CONFIG["joint_updates_per_trained_policy"], label)
            for update_index in range(CONFIG["joint_updates_per_trained_policy"]):
                if condition == "C_oracle_augmented" and update_index in CONFIG["refresh_updates"]:
                    folder = OUT / "rounds" / f"{label}_u{update_index}"
                    folder.mkdir()
                    records = []
                    refresh_root_ids = []
                    refresh_root_times = []
                    for root_time in CONFIG["root_times"]:
                        picked = root_rng.integers(len(observation), size=CONFIG["roots_per_time"])
                        roots = observation[picked, root_time, :8]
                        horizon = CONFIG["horizon"] - root_time
                        purpose = f"train/{label}/u{update_index}/t{root_time}"
                        ledger.charge("complete_model_paths", len(roots), purpose)
                        ledger.charge("oracle_transition_slots", len(roots) * horizon, purpose)
                        rollout_seed = CONFIG["training_rollout_seed_base"] + seed * 100000 + update_index * 100 + root_time
                        record = generator.rollout(state.policy_params, roots, rollout_seed, horizon)
                        record["offline_episode_id"] = ids[picked]
                        record["root_time"] = np.full(len(roots), root_time, np.int64)
                        np.savez_compressed(folder / f"h{horizon}.npz", **record)
                        records.append(record)
                        refresh_root_ids.append(ids[picked])
                        refresh_root_times.append(np.full(len(roots), root_time, np.int64))
                    synthetic, synthetic_meta = synthetic_replay(
                        records, CONFIG["training_seed_base"] + seed + 100 + update_index
                    )
                    synthetic_meta["buffer"] = synthetic
                    if len(synthetic) != 4000 or synthetic._num_eps != 128:
                        raise AssertionError("refresh replay size changed")
                    refresh_records.append({
                        "update": update_index,
                        "actor_sha256": tree_sha(state.policy_params),
                        "paths": 128,
                        "transitions": 4000,
                        "failure_paths": int(sum(record["failed_after"].any(axis=1).sum() for record in records)),
                        "reward_paths": int(sum(record["reward"].any(axis=1).sum() for record in records)),
                        "root_episode_sha256": hashlib.sha256(np.concatenate(refresh_root_ids).tobytes()).hexdigest(),
                        "root_time_sha256": hashlib.sha256(np.concatenate(refresh_root_times).tobytes()).hexdigest(),
                    })
                batch, audit = mixed_batch(offline, synthetic, update_index, order_rng)
                total_synthetic += audit["synthetic_count"]
                for key in ("episode", "time", "future"):
                    offline_digest.update(np.asarray(audit["offline"][key]).tobytes())
                permutation_digest.update(np.asarray(audit["permutation"]).tobytes())
                if audit["synthetic"] is not None:
                    one = generated_replay_audit(synthetic_meta, audit["synthetic"])
                    for key in replay_audit:
                        replay_audit[key] += one[key]
                compact = {f"offline_{key}": value for key, value in audit["offline"].items()}
                compact["permutation"] = audit["permutation"]
                compact["source"] = audit["source"]
                compact["synthetic_count"] = np.asarray(audit["synthetic_count"], np.int16)
                for key in ("episode", "time", "future"):
                    value = np.full(26, -1, np.int32)
                    if audit["synthetic"] is not None:
                        actual = np.asarray(audit["synthetic"][key], np.int32)
                        value[: len(actual)] = actual
                    compact[f"synthetic_{key}"] = value
                audit_rows.append(compact)
                state, metrics = update(state, jax.tree.map(jnp.asarray, batch))
                metric_row = {key: float(value) for key, value in metrics.items()}
                if not all(np.isfinite(value) for value in metric_row.values()):
                    raise FloatingPointError((label, update_index, metric_row))
                rows.append(metric_row)
                if (update_index + 1) % 100 == 0:
                    print(label, update_index + 1, "critic", round(metric_row["critic_loss"], 6),
                          "actor", round(metric_row["actor_loss"], 6), flush=True)
                if time.monotonic() - started > CONFIG["runtime_cap_seconds"]:
                    raise TimeoutError("sealed runtime cap exceeded")
            expected_synthetic = 0 if condition == "B_offline_only" else 25600
            if total_synthetic != expected_synthetic:
                raise AssertionError("synthetic row count changed")
            if condition == "C_oracle_augmented" and replay_audit["failed_future_task_goal"] != 0:
                raise AssertionError("failed future goal audit failed")
            final_path = OUT / "checkpoints" / f"{label}_final.pkl"
            checkpoint.save_named(str(OUT / "checkpoints"), f"{label}_final", 151000, state)
            audit_arrays = {key: np.stack([row[key] for row in audit_rows]) for key in audit_rows[0]}
            np.savez_compressed(OUT / f"{label}_batch_audit.npz", **audit_arrays)
            for metric in rows[0]:
                curve_arrays[f"{label}__{metric}"] = np.asarray([row[metric] for row in rows], np.float32)
            policy_change = np.asarray(change_fn(state.policy_params))
            actor_l2 = float(optax.global_norm(jax.tree.map(lambda a, b: a - b, state.policy_params, initial.policy_params)))
            critic_l2 = float(optax.global_norm(jax.tree.map(lambda a, b: a - b, state.q_params, initial.q_params)))
            if actor_l2 <= 0 or critic_l2 <= 0:
                raise AssertionError("intended learner parameters did not update")
            entry = {
                "condition": condition,
                "updates": 1000,
                "critic_warmup_updates": 0,
                "total_batch_rows": 256000,
                "synthetic_rows": total_synthetic,
                "synthetic_fraction": total_synthetic / 256000,
                "refreshes": refresh_records,
                "replay_audit": replay_audit,
                "initial_state_sha256": initial_hashes[-1],
                "final_state_sha256": tree_sha(state),
                "final_actor_sha256": tree_sha(state.policy_params),
                "final_critic_sha256": tree_sha(state.q_params),
                "checkpoint_sha256": sha256(final_path),
                "actor_parameter_l2": actor_l2,
                "critic_parameter_l2": critic_l2,
                "policy_change_kl_rms_max_scale": policy_change,
                "offline_draw_sha256": offline_digest.hexdigest(),
                "permutation_sha256": permutation_digest.hexdigest(),
                "loss_summary": {
                    metric: {
                        "first_100_mean": float(np.mean([row[metric] for row in rows[:100]])),
                        "last_100_mean": float(np.mean([row[metric] for row in rows[-100:]])),
                        "minimum": float(np.min([row[metric] for row in rows])),
                        "maximum": float(np.max([row[metric] for row in rows])),
                    }
                    for metric in ("critic_loss", "actor_loss", "logits_gap", "actor_grad_norm", "critic_grad_norm")
                },
            }
            runs[label] = entry
            write_json(OUT / "training_progress.json", {"runs": runs})
            offline_hashes.append(entry["offline_draw_sha256"])
            permutation_hashes.append(entry["permutation_sha256"])
        if len(set(initial_hashes)) != 1 or len(set(offline_hashes)) != 1 or len(set(permutation_hashes)) != 1:
            raise AssertionError(f"B/C pairing failed for seed {seed}")
    if tree_sha((nominal.params, heads, state_mean, state_std)) != frozen_hash:
        raise AssertionError("frozen transition components changed during training")
    np.savez_compressed(OUT / "training_curves.npz", **curve_arrays)
    result = {
        "status": "complete",
        "utc": utc(),
        "runs": runs,
        "joint_updates": 4000,
        "actor_updates": 4000,
        "critic_updates": 4000,
        "training_oracle_transitions": 32000,
        "training_generated_paths": 1024,
        "frozen_components_sha256": frozen_hash,
        "starting_state_unchanged": tree_sha(initial) == prereg["starting_state_tree_sha256"],
        "final_checkpoint_selection": CONFIG["final_checkpoint_selection"],
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(OUT / "training.json", result)
    verify_sealed()
    print("Four bounded continuation runs complete; final checkpoints sealed from native outcomes.", flush=True)


def metric_terms(record, native):
    reward = np.asarray(record["reward"], np.float64)
    failed_before = np.asarray(record["failed_before"], bool)
    failed_after = np.asarray(record["failed_after"], bool)
    physical_states = np.asarray(record["physical_states"], np.float64)
    hazard = np.asarray(record["hazard_landing"], bool) & (~failed_before)
    returns = reward @ np.power(CONFIG["discount"], np.arange(CONFIG["horizon"]))
    failure = failed_after[:, -1].astype(np.float64)
    reward_any = reward.any(axis=1).astype(np.float64)
    success = (np.linalg.norm(physical_states - np.asarray(GOAL[:2]), axis=-1).min(axis=1) < 0.5).astype(np.float64)
    entry = hazard.any(axis=1)
    first = np.argmax(hazard, axis=1)
    ones = np.ones(len(reward), np.float64)
    terms = {
        "discounted_return": (returns, ones),
        "failure_rate": (failure, ones),
        "reward_occurrence": (reward_any, ones),
        "strict_success_rate": (success, ones),
        "first_hazard_entry": (entry.astype(np.float64), ones),
        "first_hazard_time_conditional": (entry * first, entry.astype(np.float64)),
        "at_risk_hazardous_opportunities": (hazard.sum(axis=1).astype(np.float64), ones),
    }
    if not native:
        survival = (~reward.any(axis=1)) & (~failed_after.any(axis=1))
        terms["survival_without_reward"] = (survival.astype(np.float64), ones)
    return terms


def estimate(term, weights):
    numerator, denominator = term
    point = float(numerator.sum() / denominator.sum())
    boot_den = weights @ denominator
    valid = boot_den > 0
    boot = (weights[valid] @ numerator) / boot_den[valid]
    return point, boot


def summarize_terms(terms, weights):
    result = {}
    for name, term in terms.items():
        point, boot = estimate(term, weights)
        result[name] = {"estimate": point, "ci95": np.quantile(boot, [0.025, 0.975])}
    return result


def compare_terms(left, right, weights):
    result = {}
    for name in left:
        lp, lb = estimate(left[name], weights)
        rp, rb = estimate(right[name], weights)
        diff = lb - rb
        result[name] = {
            "difference": lp - rp,
            "ci95": np.quantile(diff, [0.025, 0.975]),
            "positive_fraction": float(np.mean(diff > 0)),
        }
        numerator_left, denominator_left = left[name]
        numerator_right, denominator_right = right[name]
        if np.all(denominator_left == 1) and np.all(denominator_right == 1):
            result[name]["all_episode_differences_zero"] = bool(np.all(numerator_left == numerator_right))
    return result


def load_actors(training):
    actors = {"A": checkpoint.load_checkpoint(OUT / "checkpoints" / "A.pkl")[1].policy_params}
    for label, row in training["runs"].items():
        path = OUT / "checkpoints" / f"{label}_final.pkl"
        if sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed: {label}")
        actors[label] = checkpoint.load_checkpoint(path)[1].policy_params
    return actors


def native_evaluate_policy(network, params, label, ledger):
    episodes = CONFIG["native_evaluation_episodes"]
    horizon = CONFIG["horizon"]
    purpose = f"native_eval/{label}"
    ledger.charge("native_steps", episodes * horizon, purpose)
    seeds = np.arange(episodes, dtype=np.int64) + CONFIG["native_evaluation_reset_seed_base"]
    simulators = [TimedSimulator(int(seed)) for seed in seeds]
    state = np.stack([simulator.observation()[:8] for simulator in simulators]).astype(np.float32)
    np.testing.assert_array_equal(state, np.broadcast_to(np.asarray(START, np.float32), state.shape))
    physical = np.stack([simulator.env.state.copy() for simulator in simulators]).astype(np.float64)
    states = [state.copy()]
    physical_states = [physical.copy()]
    actions, rewards, failed_before_rows, failed_after_rows, hazard_rows = [], [], [], [], []
    sample = jax.jit(lambda p, s, key: policy_tools.sample_actor(network, p, s, key))
    for step in range(horizon):
        failed_before = np.asarray([simulator.env.dead for simulator in simulators], bool)
        action = np.asarray(
            sample(params, jnp.asarray(state), jax.random.PRNGKey(CONFIG["native_evaluation_action_seed_base"] + step)),
            np.float32,
        )
        outcomes = [simulator.step(one_action) for simulator, one_action in zip(simulators, action)]
        state = np.stack([outcome[0][:8] for outcome in outcomes]).astype(np.float32)
        physical = np.stack([simulator.env.state.copy() for simulator in simulators]).astype(np.float64)
        reward = np.asarray([outcome[1] for outcome in outcomes], np.float32)
        failed_after = np.asarray([simulator.env.dead for simulator in simulators], bool)
        hazard = hazardous(physical)
        if np.any(reward[failed_after] != 0):
            raise AssertionError("native failed path received reward")
        states.append(state.copy())
        physical_states.append(physical.copy())
        actions.append(action)
        rewards.append(reward)
        failed_before_rows.append(failed_before)
        failed_after_rows.append(failed_after)
        hazard_rows.append(hazard)
    record = {
        "reset_seed": seeds,
        "states": np.stack(states, axis=1),
        "physical_states": np.stack(physical_states, axis=1),
        "action": np.stack(actions, axis=1),
        "reward": np.stack(rewards, axis=1),
        "failed_before": np.stack(failed_before_rows, axis=1),
        "failed_after": np.stack(failed_after_rows, axis=1),
        "hazard_landing": np.stack(hazard_rows, axis=1),
    }
    np.testing.assert_array_equal(record["states"][:, 1:, 2:], record["states"][:, :-1, :6])
    np.savez_compressed(OUT / f"{label}_native.npz", **record)
    return record


def model_evaluate_policy(generator, params, label, head_seed, ledger):
    paths = CONFIG["model_evaluation_paths_per_policy_head"]
    horizon = CONFIG["horizon"]
    purpose = f"model_eval/s{head_seed}/{label}"
    ledger.charge("complete_model_paths", paths, purpose)
    ledger.charge("oracle_transition_slots", paths * horizon, purpose)
    roots = np.broadcast_to(np.asarray(START, np.float32), (paths, 8)).copy()
    seed = CONFIG["model_evaluation_seed_base"] + head_seed
    record = generator.rollout(params, roots, seed, horizon)
    np.savez_compressed(OUT / f"model_s{head_seed}_{label}.npz", **record)
    return record


def evaluate():
    verify_sealed()
    started = time.monotonic()
    training = load_json(OUT / "training.json")
    if training["status"] != "complete":
        raise RuntimeError("training is incomplete")
    if (OUT / "evaluation_started.json").exists():
        raise RuntimeError("existing native evaluation attempt protected")
    _, network, _ = learner_setup()
    actors = load_actors(training)
    if set(actors) != {"A", "B_s0", "C_s0", "B_s1", "C_s1"}:
        raise RuntimeError("unexpected policy set")
    actor_hashes = {name: tree_sha(params) for name, params in actors.items()}
    write_json(
        OUT / "evaluation_started.json",
        {"utc": utc(), "training_sha256": sha256(OUT / "training.json"), "actor_tree_sha256": actor_hashes,
         "selection": CONFIG["final_checkpoint_selection"]},
    )
    ledger = Ledger()
    native = {}
    for label, params in actors.items():
        native[label] = native_evaluate_policy(network, params, label, ledger)
        print(label, ": 200 fresh paired native episodes complete", flush=True)
        if time.monotonic() - started > CONFIG["runtime_cap_seconds"]:
            raise TimeoutError("sealed evaluation runtime cap exceeded")
    nominal, heads, state_mean, state_std = oracle_components()
    model = {}
    for seed in CONFIG["training_seeds"]:
        generator = OracleGenerator(network, nominal, heads[seed], state_mean, state_std)
        for label in ("A", f"B_s{seed}", f"C_s{seed}"):
            key = f"s{seed}/{label}"
            model[key] = model_evaluate_policy(generator, actors[label], label, seed, ledger)
            print(key, ": 256 independent oracle-model paths complete", flush=True)
            if time.monotonic() - started > CONFIG["runtime_cap_seconds"]:
                raise TimeoutError("sealed evaluation runtime cap exceeded")
    native_weights = np.random.default_rng(CONFIG["bootstrap_seed"]).multinomial(
        CONFIG["native_evaluation_episodes"],
        np.full(CONFIG["native_evaluation_episodes"], 1 / CONFIG["native_evaluation_episodes"]),
        CONFIG["bootstrap_replicates"],
    )
    model_paths = CONFIG["model_evaluation_paths_per_policy_head"]
    model_weights = np.random.default_rng(CONFIG["model_bootstrap_seed"]).multinomial(
        model_paths, np.full(model_paths, 1 / model_paths), CONFIG["bootstrap_replicates"]
    )
    np.savez_compressed(OUT / "bootstrap_weights.npz", native=native_weights, model=model_weights)
    native_terms = {label: metric_terms(record, native=True) for label, record in native.items()}
    model_terms = {label: metric_terms(record, native=False) for label, record in model.items()}
    native_contrasts = {}
    model_contrasts = {}
    for seed in CONFIG["training_seeds"]:
        for left, right in ((f"C_s{seed}", "A"), (f"C_s{seed}", f"B_s{seed}"), (f"B_s{seed}", "A")):
            native_contrasts[f"{left}_minus_{right}"] = compare_terms(
                native_terms[left], native_terms[right], native_weights
            )
        prefix = f"s{seed}/"
        for left, right in ((f"C_s{seed}", "A"), (f"C_s{seed}", f"B_s{seed}"), (f"B_s{seed}", "A")):
            model_contrasts[f"s{seed}/{left}_minus_{right}"] = compare_terms(
                model_terms[prefix + left], model_terms[prefix + right], model_weights
            )
    per_seed_evidence = {}
    for seed in CONFIG["training_seeds"]:
        ca = native_contrasts[f"C_s{seed}_minus_A"]["discounted_return"]
        cb = native_contrasts[f"C_s{seed}_minus_B_s{seed}"]["discounted_return"]
        per_seed_evidence[f"s{seed}"] = {
            "point_return_above_A_and_B": ca["difference"] > 0 and cb["difference"] > 0,
            "resolved_return_above_A_and_B": ca["ci95"][0] > 0 and cb["ci95"][0] > 0,
        }
    results = {
        "native": {label: summarize_terms(terms, native_weights) for label, terms in native_terms.items()},
        "native_contrasts": native_contrasts,
        "model": {label: summarize_terms(terms, model_weights) for label, terms in model_terms.items()},
        "model_contrasts": model_contrasts,
        "per_seed_evidence": per_seed_evidence,
        "both_training_seeds_point_improve_over_both_controls": all(
            row["point_return_above_A_and_B"] for row in per_seed_evidence.values()
        ),
        "both_training_seeds_resolve_improvement_over_both_controls": all(
            row["resolved_return_above_A_and_B"] for row in per_seed_evidence.values()
        ),
        "uncertainty": {
            "native": "paired bootstrap over 200 shared fresh native reset-seed episodes",
            "model": "paired bootstrap over 256 predictive paths; these are not native samples",
            "replicates": CONFIG["bootstrap_replicates"],
        },
    }
    write_json(OUT / "results.json", results)
    expected = {
        "native_steps": 50000,
        "oracle_transition_slots": 108800,
        "complete_model_paths": 2560,
        "actor_updates": 4000,
        "critic_updates": 4000,
    }
    for category, total in expected.items():
        if ledger.data[category]["total"] != total:
            raise AssertionError((category, ledger.data[category]["total"], total))
    if {name: tree_sha(params) for name, params in actors.items()} != actor_hashes:
        raise AssertionError("sealed actor changed during evaluation")
    completion = {
        "status": "complete",
        "utc": utc(),
        "training_sha256": sha256(OUT / "training.json"),
        "results_sha256": sha256(OUT / "results.json"),
        "actor_tree_sha256": actor_hashes,
        "ledger_sha256": sha256(OUT / "ledger.json"),
        "native_training_episodes": 0,
        "new_native_evaluation_episodes": 1000,
        "new_native_evaluation_steps": 50000,
        "final_only_native_evaluation": True,
        "native_outcomes_used_for_training_or_selection": False,
        "oracle_native_hidden_or_death_used_for_generated_outcomes": False,
    }
    write_json(OUT / "completion.json", completion)
    verify_sealed()
    print("Final native and predictive evaluation complete within every sealed cap.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == "__main__":
    main()
