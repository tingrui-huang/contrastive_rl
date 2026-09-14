"""Bounded matched real-context PointMaze failure/exposure audit."""
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
SOURCE = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
RESPONSE = ROOT / "outputs" / "pointmaze_response_gate_20260914_v1"
REVIEW = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\persistent-failure-e433fd5")
OLD_TEACHER = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\branch-source-f85a5f4"
    r"\scripts\collect_swamp_windy.py"
)
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARENT = load_module("completed_persistent_failure", SOURCE / "run.py")
RESPONSE_MODULE = PARENT.RESPONSE_MODULE


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
    Path(path).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hazardous(xy):
    xy = np.asarray(xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


class Ledger:
    def __init__(self):
        path = OUT / "ledger.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.native_counts = OrderedDict(saved["native_replay_steps"]["counts"])
            self.position_counts = OrderedDict(saved["position_successors"]["counts"])
            self.head_counts = OrderedDict(saved["head_inference_rows"]["counts"])
        else:
            self.native_counts = OrderedDict()
            self.position_counts = OrderedDict()
            self.head_counts = OrderedDict()
            self.save()

    @property
    def native_total(self):
        return int(sum(self.native_counts.values()))

    @property
    def position_total(self):
        return int(sum(self.position_counts.values()))

    @property
    def head_total(self):
        return int(sum(self.head_counts.values()))

    def native(self, number, purpose):
        number = int(number)
        if self.native_total + number > CONFIG["native_step_cap"]:
            raise RuntimeError("native replay step cap exhausted")
        self.native_counts[purpose] = self.native_counts.get(purpose, 0) + number
        self.save()

    def position(self, number, purpose):
        number = int(number)
        if self.position_total + number > CONFIG["position_successor_cap"]:
            raise RuntimeError("position successor cap exhausted")
        self.position_counts[purpose] = self.position_counts.get(purpose, 0) + number
        self.save()

    def head(self, number, purpose):
        self.head_counts[purpose] = self.head_counts.get(purpose, 0) + int(number)
        self.save()

    def save(self):
        write_json(
            OUT / "ledger.json",
            {
                "native_replay_steps": {
                    "counts": self.native_counts,
                    "total": self.native_total,
                    "cap": CONFIG["native_step_cap"],
                },
                "position_successors": {
                    "counts": self.position_counts,
                    "total": self.position_total,
                    "cap": CONFIG["position_successor_cap"],
                },
                "head_inference_rows": {
                    "counts": self.head_counts,
                    "total": self.head_total,
                },
                "training_updates": 0,
                "new_complete_evaluation_episodes": 0,
                "new_complete_model_rollouts": 0,
                "actor_calls": 0,
                "nominal_policy_calls": 0,
            },
        )


EXPECTED = {
    SOURCE / "native_actor_episodes.npz": "ba30901475f6fee8fc3334fc6734ffa84d0d0b5f225f6df05716aa1415d7b0f5",
    SOURCE / "baseline_p0_rollouts.npz": "0a67f1af547dc48be48d1a7760a128a3d2b5e5a5bf7e63bc16375f278373ab36",
    SOURCE / "baseline_p1_rollouts.npz": "18c3551a16d0fee4e657223aca971d690a0253f744d6c9a230ade511791c53c9",
    SOURCE / "head_s0.npz": "06b6a5a7815f5720c9adba7c0eae5d53239c5366b73c731bf5ad98fdfdbf1579",
    SOURCE / "head_s1.npz": "b9a13cc6fc06f52874e105a2eba5319d150312a2fd9afb87c57770e428abe661",
    SOURCE / "head_feature_scaling.npz": "63b8d13acb3bf5f749866900121fc3cb43657848c0722760b2499538752cfc42",
    SOURCE / "run.py": "2b8334e00212571f763d8a8f981285dbdc4e1615225c2aa22e06c69304903e74",
    RESPONSE / "candidate_s0.npz": "c31e8383edae3ca20fd54d3966d480f6c8687dc89c9db1f7456016a476cb0183",
    RESPONSE / "candidate_s1.npz": "7169a710fe0adb0999a790c9757d4f9f4d125f567aa57e06f09bd3dcbbae243d",
    RESPONSE / "gate_feature_scaling.npz": "24391dfc715ad892148e79a130e2186fba3cae311eb5e6cd3509f3e9633c313d",
    RESPONSE / "run.py": "1b13fa14504906493e792eb8af1b5a6c0cdb00c717c6c182fc1d71cc5afaf1e9",
    ROOT / "crl" / "envs.py": "4ceb7d2cbd5f295fd97363f07badcb11a2e8bdce5efe98ab50d59c68f6c484c2",
    OLD_TEACHER: "474d007907a33ba5a1633c5bb64bf2f4c154cff1a4f94a08839f7364a248ae44",
    REVIEW / "REVIEW.md": "ea90222666ea3b84e923106f0ab739677af19e734a1c161574bcd7396abbf88e",
    REVIEW / "review_results.json": "7a06bebd3e6a9e74bac13e510b907f559e2878c6b1a194a8009c6378f4e07b89",
    ROOT / "artifacts" / "nominal_policy" / "f4_p30_expert_only_mdn_k5_s0" / "best.pkl": "6376e60185aa616c2d09c5e0250d762cde2fb845c5042488e64910fb3b745f23",
    ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "runs" / "f4_p30_sweep" / "p30_a0_a01_a03_s0_s1" / "alpha0_seed0" / "final.pkl": "ea8a71d3cb8d963259a54d47250b454462dae2ae62f03a55c370ca81a2c8ec54",
    ROOT / "artifacts" / "ett_distribution_matching" / "f4_p30_s01_guarded" / "s0_L0p25_lambda0" / "final.pkl": "9e9e1c56387d446b5788612e27e5b7f7743e120186bebc90a78fc6774476184e",
}


def preflight():
    provenance_path = OUT / "provenance.json"
    if provenance_path.exists():
        saved = json.loads(provenance_path.read_text(encoding="utf-8"))
        for path, digest in saved["dependency_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError(f"dependency changed after seal: {path}")
        return saved
    for path, digest in EXPECTED.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"pinned dependency mismatch: {path}")
    completion = json.loads((SOURCE / "completion.json").read_text(encoding="utf-8"))
    verification = json.loads((SOURCE / "verification.json").read_text(encoding="utf-8"))
    equivalent = json.loads((OUT / "equivalent_experiment_search.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete" and verification["status"] == "pass"
    assert not equivalent["equivalent_completed_diagnostic_found"]
    local_sources = [
        OUT / "PROTOCOL.md",
        OUT / "config.json",
        OUT / "equivalent_experiment_search.json",
        OUT / "collect_shadow_replay.py",
        OUT / "run.py",
    ]
    for path in local_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dependencies = list(EXPECTED) + local_sources
    result = {
        "source_commit": CONFIG["source_commit"],
        "working_head_at_execution": head,
        "head_equality_not_required": True,
        "dependency_sha256": {str(path): sha256(path) for path in dependencies},
        "checkpoint_sha256": {
            "position_p0": EXPECTED[RESPONSE / "candidate_s0.npz"],
            "position_p1": EXPECTED[RESPONSE / "candidate_s1.npz"],
            "failure_h0": EXPECTED[SOURCE / "head_s0.npz"],
            "failure_h1": EXPECTED[SOURCE / "head_s1.npz"],
            "actor": EXPECTED[ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "runs" / "f4_p30_sweep" / "p30_a0_a01_a03_s0_s1" / "alpha0_seed0" / "final.pkl"],
            "nominal": EXPECTED[ROOT / "artifacts" / "nominal_policy" / "f4_p30_expert_only_mdn_k5_s0" / "best.pkl"],
            "base_transition": EXPECTED[ROOT / "artifacts" / "ett_distribution_matching" / "f4_p30_s01_guarded" / "s0_L0p25_lambda0" / "final.pkl"],
        },
        "historical_support_review_reused_not_recomputed": True,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    write_json(provenance_path, result)
    return result


def bootstrap_weights():
    path = OUT / "bootstrap_weights.npz"
    if path.exists():
        return load_npz(path)["weights"]
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    weights = rng.multinomial(
        CONFIG["native_episodes"],
        np.full(CONFIG["native_episodes"], 1 / CONFIG["native_episodes"]),
        CONFIG["bootstrap_replicates"],
    )
    np.savez_compressed(path, weights=weights)
    return weights


def aggregate_rows(values, episodes, mask):
    values = np.asarray(values, np.float64)
    episodes = np.asarray(episodes, int)
    mask = np.asarray(mask, bool)
    sums = np.zeros(CONFIG["native_episodes"], np.float64)
    counts = np.zeros(CONFIG["native_episodes"], np.float64)
    np.add.at(sums, episodes[mask], values[mask])
    np.add.at(counts, episodes[mask], 1.0)
    return sums, counts


def mean_with_bootstrap(values, episodes, mask, weights):
    sums, counts = aggregate_rows(values, episodes, mask)
    total = counts.sum()
    if total == 0:
        return {"estimate": None, "ci95": [None, None]}
    denominators = weights @ counts
    valid = denominators > 0
    replicates = (weights[valid] @ sums) / denominators[valid]
    return {
        "estimate": float(sums.sum() / total),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
    }


def ratio_with_bootstrap(numerator, denominator, weights):
    numerator = np.asarray(numerator, np.float64)
    denominator = np.asarray(denominator, np.float64)
    if denominator.sum() == 0:
        return {"estimate": None, "ci95": [None, None]}
    den = weights @ denominator
    valid = den > 0
    replicates = (weights[valid] @ numerator) / den[valid]
    return {
        "estimate": float(numerator.sum() / denominator.sum()),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
    }


def paired_episode_mean_difference(left, right, weights):
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    replicates = weights @ delta / weights.sum(axis=1)
    return {"estimate": float(delta.mean()), "ci95": np.quantile(replicates, [0.025, 0.975])}


def paired_ratio_difference(left_num, left_den, right_num, right_den, weights):
    left_num = np.asarray(left_num, np.float64)
    left_den = np.asarray(left_den, np.float64)
    right_num = np.asarray(right_num, np.float64)
    right_den = np.asarray(right_den, np.float64)
    if left_den.sum() == 0 or right_den.sum() == 0:
        return {"estimate": None, "ci95": [None, None]}
    ld, rd = weights @ left_den, weights @ right_den
    valid = (ld > 0) & (rd > 0)
    replicates = (weights[valid] @ left_num) / ld[valid] - (weights[valid] @ right_num) / rd[valid]
    return {
        "estimate": float(left_num.sum() / left_den.sum() - right_num.sum() / right_den.sum()),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
    }


def binary_summary(probability, target, episodes, mask, weights, calibration=False):
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    result = {
        "rows": int(mask.sum()),
        "episodes": int(len(np.unique(np.asarray(episodes)[mask]))),
        "positive_events": int(target[mask].sum()),
        "observed_rate": mean_with_bootstrap(target, episodes, mask, weights),
        "predicted_rate": mean_with_bootstrap(probability, episodes, mask, weights),
        "calibration_in_the_large_predicted_minus_observed": mean_with_bootstrap(
            probability - target, episodes, mask, weights
        ),
        "brier": mean_with_bootstrap((probability - target) ** 2, episodes, mask, weights),
        "log_loss": mean_with_bootstrap(
            -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)),
            episodes,
            mask,
            weights,
        ),
    }
    if calibration:
        bins = []
        edges = np.asarray(CONFIG["calibration_bin_edges"])
        for index in range(len(edges) - 1):
            bin_mask = mask & (probability >= edges[index])
            if index == len(edges) - 2:
                bin_mask &= probability <= edges[index + 1]
            else:
                bin_mask &= probability < edges[index + 1]
            bins.append({
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": int(bin_mask.sum()),
                "episodes": int(len(np.unique(np.asarray(episodes)[bin_mask]))),
                "observed_rate": mean_with_bootstrap(target, episodes, bin_mask, weights),
                "predicted_rate": mean_with_bootstrap(probability, episodes, bin_mask, weights),
            })
        result["calibration_bins"] = bins
    return result


def input_groups(state):
    xy = state[:, :2]
    approach = (xy[:, 0] >= 2) & (xy[:, 0] < 3) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    inside = hazardous(xy)
    right = (xy[:, 0] >= 6) & (xy[:, 0] < 7) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    return OrderedDict([
        ("all", np.ones(len(state), bool)),
        ("approach_left", approach),
        ("inside_hazard", inside),
        ("right_exit", right),
        ("elsewhere", ~(approach | inside | right)),
    ])


def collect_replay(ledger):
    path = OUT / "augmented_shadow_replay.npz"
    if path.exists():
        return load_npz(path)
    if ledger.native_total:
        raise RuntimeError("native calls were charged but replay artifact is absent; investigate interruption")
    ledger.native(CONFIG["planned_native_steps"], "saved_action_shadow_advice_replay")
    subprocess.run([sys.executable, str(OUT / "collect_shadow_replay.py")], cwd=ROOT, check=True)
    return load_npz(path)


def load_heads():
    scaling = load_npz(SOURCE / "head_feature_scaling.npz")
    heads = [load_npz(SOURCE / f"head_s{seed}.npz")["theta"] for seed in (0, 1)]
    return heads, scaling["state_mean"], scaling["state_std"]


def evaluate_actual_successor(data, heads, mean, std, groups, weights, ledger):
    path = OUT / "actual_head_predictions.npz"
    if path.exists():
        prediction = load_npz(path)["probability"]
    else:
        features = PARENT.head_features(
            data["s"], data["xb_shadow"], data["xq_saved_actor"], data["y_real"][:, :2], mean, std
        )
        prediction = np.stack([PARENT.predict_numpy(theta, features) for theta in heads], axis=1)
        ledger.head(prediction.size, "actual_successor_failure_predictions")
        np.savez_compressed(path, probability=prediction, target=data["onset"])
    actual_hazard = hazardous(data["y_real"][:, :2])
    masks = OrderedDict(groups)
    masks["actual_hazardous_landing"] = actual_hazard
    masks["actual_nonhazardous_landing"] = ~actual_hazard
    metrics = {}
    for head_seed in (0, 1):
        metrics[f"h{head_seed}"] = {
            name: binary_summary(prediction[:, head_seed], data["onset"], data["episode"], mask, weights, True)
            for name, mask in masks.items()
        }
    return prediction, metrics


def sample_positions(data, ledger):
    engine = PARENT.load_position_engine()
    samples = {}
    for position_seed in (0, 1):
        path = OUT / f"position_p{position_seed}_samples.npz"
        if path.exists():
            samples[position_seed] = load_npz(path)
            continue
        purpose = f"matched_alive_contexts_p{position_seed}"
        if purpose in ledger.position_counts:
            raise RuntimeError(f"position calls charged but {path.name} is absent; investigate interruption")
        count = CONFIG["position_samples_per_alive_context"]
        ledger.position(len(data["s"]) * count, purpose)
        output_parts = []
        detail_parts = {name: [] for name in ("stationary_atom", "h", "projection_corrected")}
        theta = PARENT.position_theta(position_seed)
        for start in range(0, len(data["s"]), 128):
            stop = min(start + 128, len(data["s"]))
            output, detail = engine.candidate_sample(
                jnp.asarray(theta),
                jnp.asarray(data["s"][start:stop]),
                jnp.asarray(data["xq_saved_actor"][start:stop]),
                jnp.asarray(data["xb_shadow"][start:stop]),
                jax.random.PRNGKey(CONFIG["position_sample_seed_base"] + position_seed * 100000 + start),
                count,
            )
            output_parts.append(np.asarray(output))
            for name in detail_parts:
                detail_parts[name].append(np.asarray(detail[name]))
        output = np.concatenate(output_parts)
        result = {
            "samples_xy": output[..., :2],
            "atom": np.concatenate(detail_parts["stationary_atom"]),
            "h": np.concatenate(detail_parts["h"]),
            "projection": np.concatenate(detail_parts["projection_corrected"]),
        }
        result["legal"] = RESPONSE_MODULE.native_legal(result["samples_xy"])
        if not result["legal"].all():
            raise RuntimeError("frozen position model emitted an illegal sample")
        expected_old = np.broadcast_to(data["s"][:, None, :6], output[..., 2:].shape)
        np.testing.assert_array_equal(output[..., 2:], expected_old)
        np.savez_compressed(path, **result)
        samples[position_seed] = result
    write_json(
        OUT / "argument_order_check.json",
        {
            "sampler_signature": "candidate_sample(theta, state, action, xb, key, count)",
            "state_argument": "saved native F4 state",
            "action_argument": "saved actor action xq_saved_actor",
            "xb_argument": "new same-context shadow advice xb_shadow",
            "natural_and_executed_actions_not_swapped": True,
            "position_samples_reused_across_both_failure_heads": True,
            "all_samples_native_legal": True,
            "f4_shift_exact": True,
        },
    )
    return samples


def position_metrics(data, samples, groups, weights):
    threshold = CONFIG["stationary_threshold"]
    current_xy = data["s"][:, :2]
    real_xy = data["y_real"][:, :2]
    real_delta = real_xy - current_xy
    current_hazard = hazardous(current_xy)
    real_events = {
        "landing_hazard": hazardous(real_xy),
        "hazard_entry_from_outside": hazardous(real_xy),
        "remain_in_hazard": hazardous(real_xy),
        "exit_hazard": ~hazardous(real_xy),
        "rightward_progress": real_delta[:, 0] > threshold,
        "stationary": np.linalg.norm(real_delta, axis=1) <= threshold,
    }
    eligibility = {
        "landing_hazard": np.ones(len(current_xy), bool),
        "hazard_entry_from_outside": ~current_hazard,
        "remain_in_hazard": current_hazard,
        "exit_hazard": current_hazard,
        "rightward_progress": np.ones(len(current_xy), bool),
        "stationary": np.ones(len(current_xy), bool),
    }
    result = {}
    for position_seed, saved in samples.items():
        sample_delta = saved["samples_xy"] - current_xy[:, None]
        model_events = {
            "landing_hazard": hazardous(saved["samples_xy"]).mean(axis=1),
            "hazard_entry_from_outside": hazardous(saved["samples_xy"]).mean(axis=1),
            "remain_in_hazard": hazardous(saved["samples_xy"]).mean(axis=1),
            "exit_hazard": (~hazardous(saved["samples_xy"])).mean(axis=1),
            "rightward_progress": (sample_delta[..., 0] > threshold).mean(axis=1),
            "stationary": (np.linalg.norm(sample_delta, axis=-1) <= threshold).mean(axis=1),
        }
        arm = {"groups": {}, "predictive_distribution": {}}
        for group_name, group_mask in groups.items():
            arm["groups"][group_name] = {}
            for event_name in real_events:
                mask = group_mask & eligibility[event_name]
                arm["groups"][group_name][event_name] = binary_summary(
                    model_events[event_name], real_events[event_name], data["episode"], mask, weights
                )
            mask = group_mask
            residual = sample_delta[..., 0].mean(axis=1) - real_delta[:, 0]
            arm["groups"][group_name]["horizontal_displacement"] = {
                "rows": int(mask.sum()),
                "episodes": int(len(np.unique(data["episode"][mask]))),
                "actual_mean": mean_with_bootstrap(real_delta[:, 0], data["episode"], mask, weights),
                "predicted_mean": mean_with_bootstrap(sample_delta[..., 0].mean(axis=1), data["episode"], mask, weights),
                "predicted_minus_actual": mean_with_bootstrap(residual, data["episode"], mask, weights),
                "absolute_mean_prediction_error": mean_with_bootstrap(np.abs(residual), data["episode"], mask, weights),
            }
        all_mask = groups["all"]
        arm["predictive_distribution"] = {
            "actual_horizontal_displacement_quantiles": np.quantile(real_delta[all_mask, 0], [0, 0.1, 0.5, 0.9, 1]),
            "sampled_horizontal_displacement_quantiles": np.quantile(sample_delta[all_mask, :, 0], [0, 0.1, 0.5, 0.9, 1]),
            "actual_displacement_norm_quantiles": np.quantile(np.linalg.norm(real_delta[all_mask], axis=1), [0, 0.1, 0.5, 0.9, 1]),
            "sampled_displacement_norm_quantiles": np.quantile(np.linalg.norm(sample_delta[all_mask], axis=-1), [0, 0.1, 0.5, 0.9, 1]),
            "model_draws_are_predictive_integration_not_native_evidence": True,
        }
        result[f"p{position_seed}"] = arm
    return result


def joint_metrics(data, heads, mean, std, samples, groups, weights, ledger):
    arrays = {}
    metrics = {}
    actual_hazard = hazardous(data["y_real"][:, :2])
    target = data["onset"].astype(bool)
    for position_seed, saved in samples.items():
        sample_hazard = hazardous(saved["samples_xy"])
        metrics[f"p{position_seed}"] = {}
        features = PARENT.head_features(
            data["s"], data["xb_shadow"], data["xq_saved_actor"], saved["samples_xy"], mean, std
        )
        for head_seed, theta in enumerate(heads):
            probability = PARENT.predict_numpy(theta, features)
            ledger.head(probability.size, f"joint_position_onset_p{position_seed}_h{head_seed}")
            q = probability.mean(axis=1)
            q_hazard = (probability * sample_hazard).mean(axis=1)
            q_nonhazard = (probability * ~sample_hazard).mean(axis=1)
            covariance = q_hazard - sample_hazard.mean(axis=1) * q
            prefix = f"p{position_seed}_h{head_seed}"
            arrays[f"sample_probability_{prefix}"] = probability.astype(np.float32)
            arrays[f"q_{prefix}"] = q.astype(np.float32)
            arrays[f"q_hazard_{prefix}"] = q_hazard.astype(np.float32)
            arrays[f"q_nonhazard_{prefix}"] = q_nonhazard.astype(np.float32)
            arm = {"groups": {}}
            for group_name, mask in groups.items():
                hazard_draw_numerator = (probability * sample_hazard).sum(axis=1)
                hazard_draw_denominator = sample_hazard.sum(axis=1)
                nonhaz_draw_numerator = (probability * ~sample_hazard).sum(axis=1)
                nonhaz_draw_denominator = (~sample_hazard).sum(axis=1)
                ep_h_num, _ = aggregate_rows(hazard_draw_numerator, data["episode"], mask)
                ep_h_den, _ = aggregate_rows(hazard_draw_denominator, data["episode"], mask)
                ep_n_num, _ = aggregate_rows(nonhaz_draw_numerator, data["episode"], mask)
                ep_n_den, _ = aggregate_rows(nonhaz_draw_denominator, data["episode"], mask)
                arm["groups"][group_name] = {
                    "joint_onset": binary_summary(q, target, data["episode"], mask, weights),
                    "hazardous_generated_landing_component": binary_summary(
                        q_hazard, target & actual_hazard, data["episode"], mask, weights
                    ),
                    "nonhazardous_generated_landing_component": binary_summary(
                        q_nonhazard, target & ~actual_hazard, data["episode"], mask, weights
                    ),
                    "generated_hazard_probability": mean_with_bootstrap(
                        sample_hazard.mean(axis=1), data["episode"], mask, weights
                    ),
                    "head_probability_conditional_on_generated_hazard": ratio_with_bootstrap(ep_h_num, ep_h_den, weights),
                    "head_probability_conditional_on_generated_nonhazard": ratio_with_bootstrap(ep_n_num, ep_n_den, weights),
                    "hazard_indicator_head_probability_covariance": mean_with_bootstrap(
                        covariance, data["episode"], mask, weights
                    ),
                }
            arm["exact_decomposition_max_abs_error"] = float(np.max(np.abs(q - q_hazard - q_nonhazard)))
            arm["actual_outside_hazard_onsets"] = int((target & ~actual_hazard).sum())
            arm["no_support_mask"] = True
            metrics[f"p{position_seed}"][f"h{head_seed}"] = arm
    np.savez_compressed(OUT / "joint_predictions.npz", **arrays)
    return metrics


def first_entry_components(positions):
    positions = np.asarray(positions)
    if positions.ndim == 3:
        positions = positions[:, None]
    hazard = hazardous(positions)
    any_entry = hazard.any(axis=-1)
    first = np.argmax(hazard, axis=-1)
    first_index = np.broadcast_to(
        first[..., None, None], positions.shape[:2] + (1, positions.shape[-1])
    )
    first_xy = np.take_along_axis(positions, first_index, axis=2)[..., 0, :]
    first_xy = np.where(any_entry[..., None], first_xy, np.nan)
    cell = np.floor(first_xy[..., 0]).astype(np.float64)
    probability = any_entry.mean(axis=1)
    time_num = np.where(any_entry, first, 0).sum(axis=1) / any_entry.shape[1]
    denominator = any_entry.sum(axis=1) / any_entry.shape[1]
    x_num = np.where(any_entry, first_xy[..., 0], 0).sum(axis=1) / any_entry.shape[1]
    y_num = np.where(any_entry, first_xy[..., 1], 0).sum(axis=1) / any_entry.shape[1]
    cells = {
        str(index): ((cell == index) & any_entry).sum(axis=1) / any_entry.shape[1]
        for index in (3, 4, 5)
    }
    return {
        "probability": probability,
        "time_num": time_num,
        "denominator": denominator,
        "x_num": x_num,
        "y_num": y_num,
        "cells": cells,
    }


def summarize_first_entry(components, weights):
    result = {
        "probability": ratio_with_bootstrap(components["probability"], np.ones(CONFIG["native_episodes"]), weights),
        "time_zero_based_conditional_on_entry": ratio_with_bootstrap(components["time_num"], components["denominator"], weights),
        "mean_x_conditional_on_entry": ratio_with_bootstrap(components["x_num"], components["denominator"], weights),
        "mean_y_conditional_on_entry": ratio_with_bootstrap(components["y_num"], components["denominator"], weights),
        "cell_probability_conditional_on_entry": {
            cell: ratio_with_bootstrap(value, components["denominator"], weights)
            for cell, value in components["cells"].items()
        },
    }
    return result


def exposure_metrics(heads, mean, std, weights, ledger):
    native = load_npz(SOURCE / "native_actor_episodes.npz")
    native_positions = native["states"][:, 1:, :2]
    native_first = first_entry_components(native_positions)
    first_results = {"native": summarize_first_entry(native_first, weights)}
    native_at_risk = (hazardous(native_positions) & ~native["dead_before"]).sum(axis=1)
    exposure_results = {
        "native": {
            "at_risk_hazardous_opportunities": ratio_with_bootstrap(
                native_at_risk, np.ones(CONFIG["native_episodes"]), weights
            ),
            "definition": "hazardous landing with dead_before false, including fatal opportunity",
        }
    }
    saved_arrays = {"native_at_risk_opportunities": native_at_risk}
    for position_seed in (0, 1):
        baseline = load_npz(SOURCE / f"baseline_p{position_seed}_rollouts.npz")
        model_first = first_entry_components(baseline["states"][:, :, 1:, :2])
        first_results[f"p{position_seed}"] = summarize_first_entry(model_first, weights)
        first_results[f"p{position_seed}"]["paired_difference_from_native"] = {
            "probability": paired_episode_mean_difference(
                model_first["probability"], native_first["probability"], weights
            ),
            "time_zero_based_conditional_on_entry": paired_ratio_difference(
                model_first["time_num"], model_first["denominator"],
                native_first["time_num"], native_first["denominator"], weights
            ),
            "mean_x_conditional_on_entry": paired_ratio_difference(
                model_first["x_num"], model_first["denominator"],
                native_first["x_num"], native_first["denominator"], weights
            ),
            "mean_y_conditional_on_entry": paired_ratio_difference(
                model_first["y_num"], model_first["denominator"],
                native_first["y_num"], native_first["denominator"], weights
            ),
            "cell_probability_conditional_on_entry": {
                cell: paired_ratio_difference(
                    model_first["cells"][cell], model_first["denominator"],
                    native_first["cells"][cell], native_first["denominator"], weights
                )
                for cell in ("3", "4", "5")
            },
        }
        state = baseline["states"][:, :, :-1].reshape(-1, 8)
        xb = baseline["xb"].reshape(-1, 2)
        action = baseline["action"].reshape(-1, 2)
        next_xy = baseline["states"][:, :, 1:, :2].reshape(-1, 2)
        features = PARENT.head_features(state, xb, action, next_xy, mean, std)
        hazard = hazardous(next_xy).reshape(CONFIG["native_episodes"], 64, CONFIG["horizon"])
        for head_seed, theta in enumerate(heads):
            probability = PARENT.predict_numpy(theta, features).reshape(
                CONFIG["native_episodes"], 64, CONFIG["horizon"]
            )
            ledger.head(probability.size, f"saved_baseline_survival_integration_p{position_seed}_h{head_seed}")
            survival_before = np.concatenate(
                [np.ones(probability.shape[:-1] + (1,)), np.cumprod(1 - probability[..., :-1], axis=-1)],
                axis=-1,
            )
            at_risk = (survival_before * hazard).sum(axis=-1).mean(axis=1)
            name = f"p{position_seed}_h{head_seed}"
            exposure_results[name] = {
                "survival_weighted_at_risk_hazardous_opportunities": ratio_with_bootstrap(
                    at_risk, np.ones(CONFIG["native_episodes"]), weights
                ),
                "paired_difference_from_native": paired_episode_mean_difference(at_risk, native_at_risk, weights),
                "depends_on_frozen_onset_law": True,
                "support_mask_applied": False,
            }
            saved_arrays[f"failure_probability_{name}"] = probability.astype(np.float32)
            saved_arrays[f"survival_weighted_opportunities_{name}"] = at_risk.astype(np.float32)
    np.savez_compressed(OUT / "exposure_predictions.npz", **saved_arrays)
    write_json(OUT / "first_hazard_entry.json", first_results)
    write_json(OUT / "exposure_metrics.json", exposure_results)
    return first_results, exposure_results


def main():
    started = time.monotonic()
    if (OUT / "completion.json").exists():
        raise RuntimeError("diagnostic already complete; use saved artifacts")
    provenance = preflight()
    ledger = Ledger()
    if not (OUT / "execution_started.json").exists():
        write_json(
            OUT / "execution_started.json",
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
                "config_sha256": sha256(OUT / "config.json"),
                "source_commit": CONFIG["source_commit"],
                "no_training": True,
            },
        )
    full = collect_replay(ledger)
    replay_verification = json.loads((OUT / "replay_verification.json").read_text(encoding="utf-8"))
    assert replay_verification["status"] == "pass"
    alive = ~full["dead_before"]
    data = {name: value[alive] for name, value in full.items()}
    assert len(data["s"]) == 1165
    assert int(data["onset"].sum()) == 28
    groups = input_groups(data["s"])
    actual_hazard = hazardous(data["y_real"][:, :2])
    np.savez_compressed(
        OUT / "matched_groups.npz",
        episode=data["episode"], time=data["time"], actual_hazard=actual_hazard, **groups
    )
    weights = bootstrap_weights()
    heads, mean, std = load_heads()
    actual_prediction, actual_metrics = evaluate_actual_successor(
        data, heads, mean, std, groups, weights, ledger
    )
    samples = sample_positions(data, ledger)
    position_results = position_metrics(data, samples, groups, weights)
    joint_results = joint_metrics(data, heads, mean, std, samples, groups, weights, ledger)
    first_results, exposure_results = exposure_metrics(heads, mean, std, weights, ledger)
    impossible = int((data["onset"] & ~actual_hazard).sum())
    metrics = {
        "scope": {
            "native_episodes_reused": CONFIG["native_episodes"],
            "all_replay_rows": len(full["episode"]),
            "alive_before_rows": len(data["episode"]),
            "alive_before_onsets": int(data["onset"].sum()),
            "actual_hazardous_landings": int(actual_hazard.sum()),
            "actual_nonhazardous_landings": int((~actual_hazard).sum()),
            "impossible_actual_onsets_outside_hazard": impossible,
            "input_group_counts": {
                name: {"rows": int(mask.sum()), "episodes": int(len(np.unique(data["episode"][mask])))}
                for name, mask in groups.items()
            },
        },
        "check_A_actual_successor_failure": actual_metrics,
        "check_B_position": position_results,
        "check_C_joint": joint_results,
        "interpretation_contract": {
            "actual_successor_and_marginalized_successor_have_different_information": True,
            "brier_difference_alone_does_not_identify_position_error": True,
            "actual_outcome_hazard_groups_not_used_to_claim_generated_hazard_conditioning": True,
            "position_draws_not_counted_as_native_observations": True,
            "shadow_teacher_audit_does_not_validate_historical_learned_nominal_distribution": True,
            "frozen_position_model_was_fit_to_mixed_alive_dead_outcomes_without_internal_mode": True,
        },
    }
    write_json(OUT / "matched_metrics.json", metrics)
    ledger.save()
    assert ledger.native_total == CONFIG["planned_native_steps"]
    assert ledger.position_total == len(data["s"]) * CONFIG["position_samples_per_alive_context"] * 2
    completion = {
        "status": "complete",
        "elapsed_seconds": time.monotonic() - started,
        "source_commit": CONFIG["source_commit"],
        "provenance_sha256": sha256(OUT / "provenance.json"),
        "replay_verification_sha256": sha256(OUT / "replay_verification.json"),
        "matched_metrics_sha256": sha256(OUT / "matched_metrics.json"),
        "first_hazard_entry_sha256": sha256(OUT / "first_hazard_entry.json"),
        "exposure_metrics_sha256": sha256(OUT / "exposure_metrics.json"),
        "native_replay_steps": ledger.native_total,
        "position_successors": ledger.position_total,
        "head_inference_rows": ledger.head_total,
        "training_updates": 0,
        "new_complete_evaluation_episodes": 0,
        "new_complete_model_rollouts": 0,
        "historical_support_review_reused_not_recomputed": provenance["historical_support_review_reused_not_recomputed"],
    }
    write_json(OUT / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
