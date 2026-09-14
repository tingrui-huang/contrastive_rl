"""Bounded oracle alive-motion substitution diagnostic for PointMaze."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
REPAIR = ROOT / "outputs" / "pointmaze_supervised_repair_20260914_v1"
PERSISTENT = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
RESPONSE = ROOT / "outputs" / "pointmaze_response_gate_20260914_v1"
REVIEW = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\supervised-repair-813c3db\REVIEW.md")
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402
from oracle_motion import emit_f4, hazardous, native_alive_motion, verify_against_native  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REPAIR_COMMON = load_module("oracle_repair_common", REPAIR / "common.py")


EXPECTED_DEPENDENCIES = {
    ROOT / "crl" / "envs.py": "4ceb7d2cbd5f295fd97363f07badcb11a2e8bdce5efe98ab50d59c68f6c484c2",
    REPAIR / "REPORT.md": "0f04d741e167d5bf84e8bc70a8a000707b1d8e11b402473bebe8ae7a35f329dc",
    REPAIR / "PROTOCOL.md": "93726b322e9dc46732c3c8664737970be2404ce448091094c75a6f4ee34de080",
    REPAIR / "config.json": "3b9dab322ee625f4be49a9882fa640b79c3ccd5b0a88dcdcdc5c3cccd2d1b009",
    REPAIR / "provenance.json": "9b902ffbe36800d3baa1fb0ae13ff9f705d7bb959818df58d27ce8c9a3ff0185",
    REPAIR / "checkpoints_frozen.json": "a11182e966ede13de6dbbb8dfddeab71b65927ecdbe77fd53c4c68ce5762e25b",
    REPAIR / "head_repaired_s0.npz": "0fae8b126c801da90a106885ff13883a19494cec6ad1790e9a644e806ecca655",
    REPAIR / "head_repaired_s1.npz": "912484f9cb5931dc3e8abf77ca4786e938e31ef25f656a3a542fc1a350f9388d",
    PERSISTENT / "head_feature_scaling.npz": "63b8d13acb3bf5f749866900121fc3cb43657848c0722760b2499538752cfc42",
    REPAIR / "final_paired.npz": "ca699495f5c8c3b4016ad2a2cbf315ba89775b6d326401e557c9034f471f5998",
    REPAIR / "rollout_s0_prepaired_hrepaired.npz": "e3ece5a35a853a09a890aaf318cecbe5fff92774b7b0e39ba316a36b5ab58ab2",
    REPAIR / "rollout_s1_prepaired_hrepaired.npz": "726fa2c45dc33320500b614309aed3930e3a9fd5ca08044a7eee5e7725bb3d7b",
    REPAIR / "trajectory_metrics.json": "48fab93764e984c31af50786d7f59a47d91890fb7006bdf5588b18c60475a181",
    REPAIR / "bootstrap_weights.npz": "771c0cdeba48aca8d9df255c54efaa82e74c2ebec971e7d5ca9689e4b796a5e6",
    REPAIR / "evaluate.py": "25c5492a52d83f65eabce3f0b93addf5d74763e495624239ff11f9a84dd20bf6",
    REPAIR / "common.py": "5e0b5dafe21734666cc0020add7a9e03a4cd7ea2654ee9bba64f8ca02564e2c3",
    PERSISTENT / "run.py": "2b8334e00212571f763d8a8f981285dbdc4e1615225c2aa22e06c69304903e74",
    RESPONSE / "run.py": "1b13fa14504906493e792eb8af1b5a6c0cdb00c717c6c182fc1d71cc5afaf1e9",
    ROOT / "artifacts" / "nominal_policy" / "f4_p30_expert_only_mdn_k5_s0" / "best.pkl": "6376e60185aa616c2d09c5e0250d762cde2fb845c5042488e64910fb3b745f23",
    ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "runs" / "f4_p30_sweep" / "p30_a0_a01_a03_s0_s1" / "alpha0_seed0" / "final.pkl": "ea8a71d3cb8d963259a54d47250b454462dae2ae62f03a55c370ca81a2c8ec54",
    REVIEW: "8c8a86d26bb602bc11659780e8b5ad43ce369945cdac2c7405b7135b0fb73664",
}


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


class Ledger:
    def __init__(self):
        path = OUT / "ledger.json"
        if path.exists():
            saved = load_json(path)
            self.native = OrderedDict(saved["native_verification_steps"]["counts"])
            self.slots = OrderedDict(saved["oracle_motion_slots"]["counts"])
            self.paths = OrderedDict(saved["complete_model_paths"]["counts"])
        else:
            self.native = OrderedDict()
            self.slots = OrderedDict()
            self.paths = OrderedDict()
            self.save()

    @staticmethod
    def total(values):
        return int(sum(values.values()))

    def _charge(self, values, count, purpose, cap, label):
        count = int(count)
        if purpose in values:
            return False
        if self.total(values) + count > cap:
            raise RuntimeError(f"{label} cap exhausted")
        values[purpose] = count
        self.save()
        return True

    def charge_native(self, count, purpose):
        return self._charge(self.native, count, purpose, CONFIG["native_verification_step_cap"], "native verification")

    def charge_slots(self, count, purpose):
        return self._charge(self.slots, count, purpose, CONFIG["oracle_motion_slot_cap"], "oracle motion slot")

    def charge_paths(self, count, purpose):
        return self._charge(self.paths, count, purpose, CONFIG["complete_model_path_cap"], "complete model path")

    def save(self):
        write_json(
            OUT / "ledger.json",
            {
                "native_verification_steps": {"counts": self.native, "total": self.total(self.native), "cap": CONFIG["native_verification_step_cap"]},
                "oracle_motion_slots": {"counts": self.slots, "total": self.total(self.slots), "cap": CONFIG["oracle_motion_slot_cap"]},
                "complete_model_paths": {"counts": self.paths, "total": self.total(self.paths), "cap": CONFIG["complete_model_path_cap"]},
                "training_updates": {"total": 0, "cap": 0},
                "new_training_episodes": {"total": 0, "cap": 0},
                "new_complete_native_evaluation_episodes": {"total": 0, "cap": 0},
            },
        )


def preflight():
    for path, expected in EXPECTED_DEPENDENCIES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"dependency hash mismatch: {path}: {actual} != {expected}")
    subprocess.check_call(["git", "cat-file", "-e", CONFIG["source_commit"] + "^{commit}"], cwd=ROOT)
    completion = load_json(REPAIR / "completion.json")
    verification = load_json(REPAIR / "verification.json")
    frozen = load_json(REPAIR / "checkpoints_frozen.json")
    equivalent = load_json(OUT / "equivalent_experiment_search.json")
    if completion["status"] != "complete" or verification["status"] != "passed":
        raise RuntimeError("completed repair dependency is not complete and verified")
    if frozen["status"] != "frozen_before_final_collection":
        raise RuntimeError("repaired checkpoints are not frozen")
    if equivalent["equivalent_completed_diagnostic_found"]:
        raise RuntimeError("equivalent diagnostic exists")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    local_sources = [OUT / "PROTOCOL.md", OUT / "config.json", OUT / "equivalent_experiment_search.json", OUT / "oracle_motion.py", Path(__file__)]
    result = {
        "source_commit": CONFIG["source_commit"],
        "working_head_at_execution": current_head,
        "head_equality_not_required": True,
        "dependency_sha256": {str(path): expected for path, expected in EXPECTED_DEPENDENCIES.items()},
        "diagnostic_source_sha256": {str(path): sha256(path) for path in local_sources},
        "frozen_head_sha256": {f"s{seed}": sha256(REPAIR / f"head_repaired_s{seed}.npz") for seed in (0, 1)},
        "equivalent_completed_diagnostic_found": False,
        "known_hazard_support_is_additional_environment_information": True,
        "simulator_assisted_motion_oracle": True,
        "native_death_and_reward_not_used_in_oracle_rollouts": True,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    write_json(OUT / "provenance.json", result)
    return result


def run_equivalence(ledger):
    path = OUT / "movement_equivalence.json"
    if path.exists():
        result = load_json(path)
        if result["status"] != "passed":
            raise RuntimeError("saved movement equivalence did not pass")
        return result
    ledger.charge_native(CONFIG["planned_native_verification_steps"], "movement_equivalence")
    result = verify_against_native(TwoRouteSwampWindyF4Env, CONFIG["verification_noise_seed"])
    if result["native_step_calls"] != CONFIG["planned_native_verification_steps"]:
        raise RuntimeError("native verification count changed")
    write_json(path, result)
    return result


def head_parameters(seed):
    return load_npz(REPAIR / f"head_repaired_s{seed}.npz")["theta"]


def head_scaling():
    scaling = load_npz(PERSISTENT / "head_feature_scaling.npz")
    return scaling["state_mean"], scaling["state_std"]


def run_oracle_arm(seed, ledger):
    path = OUT / f"oracle_rollout_s{seed}.npz"
    if path.exists():
        return load_npz(path)
    episodes = CONFIG["native_final_episodes_reused"]
    paths_per_root = CONFIG["model_paths_per_native_root"]
    horizon = CONFIG["horizon"]
    flat_paths = episodes * paths_per_root
    ledger.charge_paths(flat_paths, f"oracle_rollout_s{seed}")
    ledger.charge_slots(flat_paths * horizon, f"oracle_rollout_s{seed}")

    final = load_npz(REPAIR / "final_paired.npz")
    roots = np.repeat(final["prefix_states"][:, 0], paths_per_root, axis=0).astype(np.float32)
    if roots.shape != (flat_paths, 8):
        raise RuntimeError("root shape changed")
    if not np.all(roots == np.array([0.5, 3.5] * 4, np.float32)):
        raise RuntimeError("native roots are not the exact reset state")
    state = roots.copy()
    physical = state[:, :2].astype(np.float64)
    failed = np.zeros(flat_paths, bool)
    goal = np.broadcast_to(REPAIR_COMMON.PARENT.GOAL, state.shape).astype(np.float32).copy()
    goal_physical = np.asarray(REPAIR_COMMON.PARENT.GOAL[:2], np.float64)
    mean, std = head_scaling()
    theta = jnp.asarray(head_parameters(seed), jnp.float32)
    mean_j, std_j = jnp.asarray(mean, jnp.float32), jnp.asarray(std, jnp.float32)
    engine = REPAIR_COMMON.PARENT.load_position_engine()
    nominal, actor = engine.kernel.nominal, engine.kernel.actor
    step_keys = jax.random.split(jax.random.PRNGKey(CONFIG["rollout_seed"]), horizon)

    records = {name: [] for name in (
        "action", "xb", "motion_noise", "clipped_noisy_action", "blocked_updates",
        "position_proposal_xy", "position_proposal_physical_xy", "failed_before",
        "failed_after", "onset", "onset_uniform", "failure_probability",
        "raw_failure_probability", "hazard_landing", "hazard_landing_physical",
        "reward", "reward_if_emitted", "hazard_classification_disagreement",
        "reward_classification_disagreement",
    )}
    states = [state.copy()]
    physical_states = [physical.copy()]
    for step in range(horizon):
        nominal_key, actor_key, motion_key, onset_key = jax.random.split(step_keys[step], 4)
        state_j, goal_j = jnp.asarray(state), jnp.asarray(goal)
        xb = np.asarray(nominal.sample(state_j, nominal_key, 1, goal=goal_j), np.float32)
        action = np.asarray(actor(state_j, goal_j, actor_key), np.float32)
        motion_noise = np.asarray(
            jax.random.normal(motion_key, (flat_paths, 2), dtype=jnp.float32), np.float64
        ) * CONFIG["motion_noise_standard_deviation"]
        moved_physical, clipped_action, blocked_updates = native_alive_motion(physical, action, motion_noise)
        moved_emitted = moved_physical.astype(np.float32)
        next_physical = np.where(failed[:, None], physical, moved_physical)
        next_emitted = next_physical.astype(np.float32)
        next_state = emit_f4(next_physical, state)
        support_emitted = hazardous(next_emitted)
        support_physical = hazardous(next_physical)
        displacement = jnp.asarray(next_emitted) - state_j[:, :2]
        features = jnp.concatenate([(state_j - mean_j) / std_j, jnp.asarray(xb), jnp.asarray(action), displacement], axis=-1)
        raw_probability = np.asarray(jax.nn.sigmoid(REPAIR_COMMON.PARENT.head_logits(theta, features)), np.float32)
        probability = np.where(failed, 0.0, np.where(support_emitted, raw_probability, 0.0)).astype(np.float32)
        onset_uniform = np.asarray(jax.random.uniform(onset_key, probability.shape), np.float32)
        onset = (~failed) & (onset_uniform < probability)
        next_failed = failed | onset
        reward_physical = ((np.linalg.norm(next_physical - goal_physical, axis=-1) < 2) & (~next_failed)).astype(np.float32)
        reward_emitted = ((np.linalg.norm(next_emitted - goal[:, :2], axis=-1) < 2) & (~next_failed)).astype(np.float32)

        values = {
            "action": action,
            "xb": xb,
            "motion_noise": motion_noise,
            "clipped_noisy_action": clipped_action,
            "blocked_updates": blocked_updates,
            "position_proposal_xy": next_emitted,
            "position_proposal_physical_xy": next_physical,
            "failed_before": failed,
            "failed_after": next_failed,
            "onset": onset,
            "onset_uniform": onset_uniform,
            "failure_probability": probability,
            "raw_failure_probability": raw_probability,
            "hazard_landing": support_emitted,
            "hazard_landing_physical": support_physical,
            "reward": reward_physical,
            "reward_if_emitted": reward_emitted,
            "hazard_classification_disagreement": support_emitted != support_physical,
            "reward_classification_disagreement": reward_physical != reward_emitted,
        }
        for name, value in values.items():
            records[name].append(np.asarray(value))
        state, physical, failed = next_state, next_physical, next_failed
        states.append(state.copy())
        physical_states.append(physical.copy())

    arrays = {}
    for name, values in records.items():
        value = np.stack(values, axis=1)
        arrays[name] = value.reshape((episodes, paths_per_root) + value.shape[1:])
    state_array = np.stack(states, axis=1)
    physical_array = np.stack(physical_states, axis=1)
    arrays["states"] = state_array.reshape((episodes, paths_per_root) + state_array.shape[1:])
    arrays["physical_states"] = physical_array.reshape((episodes, paths_per_root) + physical_array.shape[1:])
    discount = np.power(np.float32(CONFIG["discount"]), np.arange(horizon, dtype=np.float32))
    arrays["return"] = np.sum(arrays["reward"] * discount, axis=-1, dtype=np.float32)
    arrays["head_seed"] = np.asarray(seed, np.int64)
    np.savez_compressed(path, **arrays)
    write_json(
        OUT / f"oracle_rollout_s{seed}_metadata.json",
        {
            "head_seed": seed,
            "episodes": episodes,
            "paths_per_root": paths_per_root,
            "complete_paths": flat_paths,
            "oracle_motion_slots": flat_paths * horizon,
            "physical_emitted_hazard_disagreements": int(arrays["hazard_classification_disagreement"].sum()),
            "physical_emitted_reward_disagreements": int(arrays["reward_classification_disagreement"].sum()),
            "learned_anchor_response_gate_atom_projection": "not applicable",
            "native_hidden_bits_death_termination_reward_used": False,
            "output_sha256": sha256(path),
        },
    )
    return arrays


def first_event_terms(event, positions):
    event = np.asarray(event, bool)
    positions = np.asarray(positions)
    if event.ndim == 2:
        event = event[:, None, :]
        positions = positions[:, None, :, :]
    entered = event.any(axis=-1)
    first = np.argmax(event, axis=-1)
    index = np.broadcast_to(first[..., None, None], positions.shape[:-2] + (1, 2))
    xy = np.take_along_axis(positions, index, axis=-2)[..., 0, :]
    return {
        "probability": (entered.mean(axis=1), np.ones(event.shape[0])),
        "time": ((entered * first).mean(axis=1), entered.mean(axis=1)),
        "x": ((entered * xy[..., 0]).mean(axis=1), entered.mean(axis=1)),
        "y": ((entered * xy[..., 1]).mean(axis=1), entered.mean(axis=1)),
    }


def trajectory_terms(record, native=False):
    if native:
        reward = record["prefix_rewards"]
        failed_before = record["prefix_dead_before"]
        failed_after = record["prefix_dead_after"]
        onset = record["prefix_onset"]
        positions = record["prefix_states"][:, 1:, :2]
        returns = np.sum(reward * CONFIG["discount"] ** np.arange(CONFIG["horizon"]), axis=-1)
    else:
        reward = record["reward"]
        failed_before = record["failed_before"]
        failed_after = record["failed_after"]
        onset = record["onset"]
        positions = record["position_proposal_xy"]
        returns = record["return"]
    reward_any = reward.any(axis=-1)
    failed_any = failed_after.any(axis=-1)
    survival = (~reward_any) & (~failed_any)
    root_mean = lambda value: value if value.ndim == 1 else value.mean(axis=1)
    ones = np.ones(reward.shape[0], np.float64)
    terms = {
        "reward_occurrence": (root_mean(reward_any.astype(float)), ones),
        "failure_probability": (root_mean(failed_any.astype(float)), ones),
        "survival_without_reward": (root_mean(survival.astype(float)), ones),
        "mean_discounted_return": (root_mean(returns.astype(float)), ones),
        "return_conditional_on_reward": (root_mean(returns * reward_any), root_mean(reward_any.astype(float))),
    }
    onset_terms = first_event_terms(onset, positions)
    hazard_event = hazardous(positions) & (~failed_before)
    hazard_terms = first_event_terms(hazard_event, positions)
    terms["onset_time_zero_based_conditional_on_failure"] = onset_terms["time"]
    terms["first_hazard_probability"] = hazard_terms["probability"]
    terms["first_hazard_time_zero_based_conditional"] = hazard_terms["time"]
    terms["first_hazard_mean_x"] = hazard_terms["x"]
    terms["first_hazard_mean_y"] = hazard_terms["y"]
    exposure = hazard_event.sum(axis=-1).astype(float)
    terms["at_risk_hazardous_opportunities"] = (root_mean(exposure), ones)
    return {name: (np.asarray(num, np.float64), np.asarray(den, np.float64)) for name, (num, den) in terms.items()}


def estimate(term, weights):
    numerator, denominator = term
    if denominator.sum() == 0:
        return None, np.full(len(weights), np.nan)
    boot_den = weights @ denominator
    valid = boot_den > 0
    replicates = np.full(len(weights), np.nan)
    replicates[valid] = (weights[valid] @ numerator) / boot_den[valid]
    return float(numerator.sum() / denominator.sum()), replicates


def summarize(terms, weights):
    result = {}
    for name, term in terms.items():
        point, replicates = estimate(term, weights)
        valid = replicates[np.isfinite(replicates)]
        result[name] = {"estimate": point, "ci95": np.quantile(valid, [0.025, 0.975]) if len(valid) else [None, None]}
    return result


def compare(left, right, weights):
    result = {}
    for name in left:
        left_point, left_boot = estimate(left[name], weights)
        right_point, right_boot = estimate(right[name], weights)
        valid = np.isfinite(left_boot) & np.isfinite(right_boot)
        diff = left_boot[valid] - right_boot[valid]
        result[name] = {
            "estimate": None if left_point is None or right_point is None else left_point - right_point,
            "ci95": np.quantile(diff, [0.025, 0.975]) if len(diff) else [None, None],
        }
    return result


def absolute_error(left, native, weights):
    result = {}
    for name in left:
        left_point, left_boot = estimate(left[name], weights)
        native_point, native_boot = estimate(native[name], weights)
        valid = np.isfinite(left_boot) & np.isfinite(native_boot)
        error = np.abs(left_boot[valid] - native_boot[valid])
        result[name] = {
            "estimate": None if left_point is None or native_point is None else abs(left_point - native_point),
            "ci95": np.quantile(error, [0.025, 0.975]) if len(error) else [None, None],
        }
    return result


def absolute_error_change(oracle, learned, native, weights):
    result = {}
    for name in oracle:
        oracle_point, oracle_boot = estimate(oracle[name], weights)
        learned_point, learned_boot = estimate(learned[name], weights)
        native_point, native_boot = estimate(native[name], weights)
        valid = np.isfinite(oracle_boot) & np.isfinite(learned_boot) & np.isfinite(native_boot)
        change = np.abs(oracle_boot[valid] - native_boot[valid]) - np.abs(learned_boot[valid] - native_boot[valid])
        result[name] = {
            "estimate": None if None in (oracle_point, learned_point, native_point) else abs(oracle_point - native_point) - abs(learned_point - native_point),
            "ci95": np.quantile(change, [0.025, 0.975]) if len(change) else [None, None],
            "negative_means_oracle_is_closer": True,
        }
    return result


def evaluate(oracle_records):
    weights = load_npz(REPAIR / "bootstrap_weights.npz")["weights"]
    final = load_npz(REPAIR / "final_paired.npz")
    native_terms = trajectory_terms(final, native=True)
    result = {"metric_definitions": {
        "hazard": CONFIG["hazard_definition"],
        "first_hazard": "first emitted hazardous landing while failed_before is false",
        "at_risk_hazardous_opportunities": "emitted hazardous landings with failed_before false, including fatal onset",
        "uncertainty": "2000 paired whole-native-episode/root bootstrap replicates; 64 paths per root are predictive integration",
    }, "native": summarize(native_terms, weights), "seeds": {}}
    prior_metrics = load_json(REPAIR / "trajectory_metrics.json")
    for name, summary in result["native"].items():
        if name == "first_hazard_mean_x":
            expected = prior_metrics["native"]["first_hazard_mean_xy"]["x"]["estimate"]
        elif name == "first_hazard_mean_y":
            expected = prior_metrics["native"]["first_hazard_mean_xy"]["y"]["estimate"]
        else:
            expected = prior_metrics["native"][name]["estimate"]
        np.testing.assert_allclose(summary["estimate"], expected, rtol=0, atol=2e-6)
    for seed in (0, 1):
        learned_record = load_npz(REPAIR / f"rollout_s{seed}_prepaired_hrepaired.npz")
        learned_terms = trajectory_terms(learned_record)
        oracle_terms = trajectory_terms(oracle_records[seed])
        learned_summary = summarize(learned_terms, weights)
        for name, summary in learned_summary.items():
            if name == "first_hazard_mean_x":
                expected = prior_metrics["arms"][f"s{seed}"]["prepaired_hrepaired"]["metrics"]["first_hazard_mean_xy"]["x"]["estimate"]
            elif name == "first_hazard_mean_y":
                expected = prior_metrics["arms"][f"s{seed}"]["prepaired_hrepaired"]["metrics"]["first_hazard_mean_xy"]["y"]["estimate"]
            else:
                expected = prior_metrics["arms"][f"s{seed}"]["prepaired_hrepaired"]["metrics"][name]["estimate"]
            np.testing.assert_allclose(summary["estimate"], expected, rtol=0, atol=2e-6)
        result["seeds"][f"s{seed}"] = {
            "saved_repaired_position_repaired_head": learned_summary,
            "oracle_motion_repaired_head": summarize(oracle_terms, weights),
            "oracle_minus_saved_learned": compare(oracle_terms, learned_terms, weights),
            "oracle_minus_native": compare(oracle_terms, native_terms, weights),
            "saved_learned_absolute_native_error": absolute_error(learned_terms, native_terms, weights),
            "oracle_absolute_native_error": absolute_error(oracle_terms, native_terms, weights),
            "oracle_minus_learned_absolute_error": absolute_error_change(oracle_terms, learned_terms, native_terms, weights),
        }
    write_json(OUT / "metrics.json", result)
    return result


def first_hazard_arrays(record):
    event = hazardous(record["position_proposal_xy"]) & (~record["failed_before"])
    entered = event.any(axis=-1)
    first = np.argmax(event, axis=-1)
    index = np.broadcast_to(first[..., None, None], record["position_proposal_xy"].shape[:-2] + (1, 2))
    xy = np.take_along_axis(record["position_proposal_xy"], index, axis=-2)[..., 0, :]
    return entered, first, xy


def contract_checks(oracle_records, equivalence):
    final = load_npz(REPAIR / "final_paired.npz")
    initial = final["prefix_states"][:, 0]
    disagreement = {"hazard": {}, "reward": {}}
    for seed, record in oracle_records.items():
        np.testing.assert_array_equal(record["states"][:, :, 0], np.broadcast_to(initial[:, None], record["states"][:, :, 0].shape))
        np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
        np.testing.assert_array_equal(record["physical_states"][:, :, 1:], record["position_proposal_physical_xy"])
        np.testing.assert_array_equal(record["states"][..., :2], record["physical_states"].astype(np.float32))
        if not REPAIR_COMMON.RESPONSE_MODULE.native_legal(record["states"][..., :2]).all():
            raise AssertionError("oracle rollout emitted an illegal state")
        if np.any(record["failure_probability"][~record["hazard_landing"]] != 0):
            raise AssertionError("onset probability outside emitted hazard")
        np.testing.assert_array_equal(record["failed_after"], record["failed_before"] | record["onset"])
        np.testing.assert_array_equal(record["failed_before"][:, :, 1:], record["failed_after"][:, :, :-1])
        if np.any(np.diff(record["failed_after"].astype(np.int8), axis=-1) < 0):
            raise AssertionError("failure reversal")
        frozen = record["failed_before"]
        np.testing.assert_array_equal(record["states"][:, :, 1:, :2][frozen], record["states"][:, :, :-1, :2][frozen])
        np.testing.assert_array_equal(record["physical_states"][:, :, 1:][frozen], record["physical_states"][:, :, :-1][frozen])
        onset = record["onset"]
        np.testing.assert_array_equal(record["states"][:, :, 1:, :2][onset], record["position_proposal_xy"][onset])
        if np.any(record["reward"][record["failed_after"]] != 0):
            raise AssertionError("reward after failure")
        disagreement["hazard"][f"s{seed}"] = int(record["hazard_classification_disagreement"].sum())
        disagreement["reward"][f"s{seed}"] = int(record["reward_classification_disagreement"].sum())
    left, right = oracle_records[0], oracle_records[1]
    np.testing.assert_array_equal(left["motion_noise"], right["motion_noise"])
    np.testing.assert_array_equal(left["onset_uniform"], right["onset_uniform"])
    left_entry = first_hazard_arrays(left)
    right_entry = first_hazard_arrays(right)
    for left_value, right_value in zip(left_entry, right_entry):
        np.testing.assert_array_equal(left_value, right_value)
    entered, first, _ = left_entry
    time = np.arange(CONFIG["horizon"])[None, None, :]
    pre_or_entry = (~entered[..., None]) | (time <= first[..., None])
    for name in ("states", "physical_states"):
        # State arrays have one extra terminal time; compare pre-step states.
        np.testing.assert_array_equal(left[name][:, :, :-1][pre_or_entry], right[name][:, :, :-1][pre_or_entry])
    for name in ("xb", "action", "position_proposal_xy", "position_proposal_physical_xy"):
        np.testing.assert_array_equal(left[name][pre_or_entry], right[name][pre_or_entry])
    result = {
        "movement_equivalence_passed_before_rollouts": equivalence["status"] == "passed",
        "native_verification_steps": equivalence["native_step_calls"],
        "oracle_arms": 2,
        "complete_paths": sum(record["return"].size for record in oracle_records.values()),
        "oracle_motion_slots": sum(record["reward"].size for record in oracle_records.values()),
        "correct_initial_roots_and_goal": True,
        "same_motion_and_onset_random_streams_across_heads": True,
        "pathwise_first_hazard_identical_across_heads": True,
        "pre_and_first_entry_states_actions_advice_motion_identical_across_heads": True,
        "zero_onset_outside_emitted_hazard": True,
        "failure_irreversible": True,
        "fatal_incoming_motion_retained": True,
        "post_failure_physical_and_emitted_xy_frozen": True,
        "f4_shift_exact": True,
        "failed_reward_zero": True,
        "learned_anchor_response_gate_projection_used": False,
        "native_hidden_death_or_reward_used": False,
        "physical_emitted_classification_disagreements": disagreement,
    }
    write_json(OUT / "contract_checks.json", result)
    return result


def main():
    started = time.time()
    provenance = preflight()
    if not (OUT / "execution_started.json").exists():
        write_json(OUT / "execution_started.json", {"status": "started_after_protocol_and_source_hashes_were_sealed", "unix_time": started})
    ledger = Ledger()
    equivalence = run_equivalence(ledger)
    oracle_records = {seed: run_oracle_arm(seed, ledger) for seed in (0, 1)}
    metrics = evaluate(oracle_records)
    checks = contract_checks(oracle_records, equivalence)
    ledger.save()
    if ledger.total(ledger.native) != CONFIG["planned_native_verification_steps"]:
        raise RuntimeError("native verification accounting mismatch")
    if ledger.total(ledger.paths) != CONFIG["complete_model_path_cap"]:
        raise RuntimeError("path accounting mismatch")
    if ledger.total(ledger.slots) != CONFIG["oracle_motion_slot_cap"]:
        raise RuntimeError("slot accounting mismatch")
    completion = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "source_commit": CONFIG["source_commit"],
        "provenance_sha256": sha256(OUT / "provenance.json"),
        "movement_equivalence_sha256": sha256(OUT / "movement_equivalence.json"),
        "oracle_rollout_sha256": {f"s{seed}": sha256(OUT / f"oracle_rollout_s{seed}.npz") for seed in (0, 1)},
        "metrics_sha256": sha256(OUT / "metrics.json"),
        "contract_checks_sha256": sha256(OUT / "contract_checks.json"),
        "native_verification_steps": ledger.total(ledger.native),
        "new_complete_model_paths": ledger.total(ledger.paths),
        "oracle_motion_slots": ledger.total(ledger.slots),
        "training_updates": 0,
        "new_training_episodes": 0,
        "new_complete_native_evaluation_episodes": 0,
        "historical_artifacts_modified": False,
        "result_seed_keys": list(metrics["seeds"]),
        "pathwise_first_hazard_identity": checks["pathwise_first_hazard_identical_across_heads"],
    }
    write_json(OUT / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
