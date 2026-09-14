"""Independent verification of the completed response-gate experiment.

This script reads only saved arrays, JSON records, and hashed inputs.  It does
not import or call the model, actor, nominal policy, or environment, and it
does not generate trajectories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
GOAL_XY = np.array([8.5, 3.5], dtype=np.float32)
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
    dtype=np.int8,
)


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name: str, value: dict) -> None:
    def plain(item):
        if isinstance(item, dict):
            return {str(key): plain(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(val) for val in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item

    (OUT / name).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def energy_score(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Full U-statistic, including all ordered sample-pair terms."""
    samples = np.asarray(samples, np.float64)
    target = np.asarray(target, np.float64)
    rows, count, _ = samples.shape
    result = np.empty(rows, np.float64)
    for start in range(0, rows, 128):
        stop = min(rows, start + 128)
        value = samples[start:stop]
        first = np.linalg.norm(value - target[start:stop, None], axis=-1).mean(1)
        pairs = np.linalg.norm(value[:, :, None] - value[:, None, :], axis=-1)
        result[start:stop] = first - pairs.sum((1, 2)) / (
            2 * count * (count - 1)
        )
    return result


def native_legal(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy)
    floor = np.floor(xy).astype(np.int64)
    bounded = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= 9)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= 5)
    )
    cell = WALLS[
        np.clip(floor[..., 0], 0, 8), np.clip(floor[..., 1], 0, 4)
    ]
    return bounded & (cell == 0)


def assert_close(actual, expected, atol=1e-12) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


config = read_json("config.json")
provenance = read_json("provenance.json")
started = read_json("execution_started.json")
completion = read_json("completion.json")
ledger = read_json("ledger.json")

# The original sealed driver is identified by execution_started.json.  Three
# narrowly documented reporting/verification amendments followed failures;
# resume_2 records the current driver hash.
assert started["protocol_sha256"] == sha256(OUT / "PROTOCOL.md")
assert started["config_sha256"] == sha256(OUT / "config.json")
assert started["collector_sha256"] == sha256(OUT / "collect_evaluation.py")
assert started["driver_sha256"] == provenance["dependency_sha256"][str(OUT / "run.py")]
resume_records = [read_json(name) for name in ["resume.json", "resume_1.json", "resume_2.json"]]
assert resume_records[-1]["resumed_driver_sha256"] == sha256(OUT / "run.py")
assert all(not record["repeated_training"] for record in resume_records)
assert all(not record["repeated_model_evaluation"] for record in resume_records)
assert all(not record["repeated_native_collection"] for record in resume_records)

hash_checks = {}
for path_text, expected in provenance["dependency_sha256"].items():
    path = Path(path_text)
    if path == OUT / "run.py":
        continue
    actual = sha256(path)
    assert actual == expected, path
    hash_checks[str(path)] = True

assert completion["status"] == "complete"
assert ledger["model"]["total"] == sum(ledger["model"]["counts"].values())
assert ledger["native"]["total"] == sum(ledger["native"]["counts"].values())
assert ledger["model"]["total"] == completion["model_successors"] <= config["model_successor_cap"]
assert ledger["native"]["total"] == completion["native_steps"] <= config["native_step_cap"]
assert completion["failed_verification_model_successor_overhead"] == 307200
assert completion["model_successors"] == completion["planned_model_successors"] + 307200
assert completion["native_steps"] == config["planned_native_steps"]["total"]

# Verify matched optimizer continuations from the saved signed losses.  This
# reconstructs all final parameters and optimizer moments without model calls.
train_path = next(
    Path(path)
    for path in provenance["dependency_sha256"]
    if "supervised_ett_native_probe_v1" in path and path.endswith("paired_transitions.npz")
)
train = load_npz(train_path)
train_mask = train["episode"] < 72
diagonal = np.all(train["xb"] == train["xq"], axis=1)
diagonal_ids = np.flatnonzero(train_mask & diagonal)
off_ids = np.flatnonzero(train_mask & ~diagonal)
source_dir = OUT.parent / "supervised_stochastic_ett_20260914_v1"
optimizer_checks = {}
for seed in (0, 1):
    parent = load_npz(source_dir / f"B_s{seed}.npz")
    for arm in ("control", "candidate"):
        name = f"{arm}_s{seed}"
        saved = load_npz(OUT / f"{name}.npz")
        history = read_json(f"{name}_training.json")
        assert len(history) == config["additional_updates"] == 120
        assert saved["trajectory"].shape == (120, 48 if arm == "control" else 69)
        np.testing.assert_array_equal(saved["parent_theta"], parent["theta"])
        assert int(saved["inherited_adam_step"]) == 120
        if arm == "control":
            theta = parent["theta"].copy()
            adam_m = parent["adam_m"].astype(np.float64).copy()
            adam_v = parent["adam_v"].astype(np.float64).copy()
        else:
            theta = np.r_[parent["theta"], np.zeros(config["gate_dimension"], np.float32)]
            adam_m = np.r_[parent["adam_m"].astype(np.float64), np.zeros(config["gate_dimension"])]
            adam_v = np.r_[parent["adam_v"].astype(np.float64), np.zeros(config["gate_dimension"])]
        dimension = len(theta)
        sigma = np.r_[
            np.full(16, config["diagonal_sigma"]),
            np.full(32, config["response_sigma"]),
            np.full(dimension - 48, config["gate_sigma"]),
        ]
        rate = np.r_[
            np.full(16, config["diagonal_learning_rate"]),
            np.full(32, config["response_learning_rate"]),
            np.full(dimension - 48, config["gate_learning_rate"]),
        ]
        for step, row in enumerate(history):
            rng = np.random.default_rng(
                config["training_direction_seed"] + seed * 100000 + step * 10
            )
            rng.choice(diagonal_ids, config["batch_per_term"])
            rng.choice(off_ids, config["batch_per_term"])
            directions = rng.normal(size=(config["directions"], 48))
            if arm == "candidate":
                directions = np.c_[
                    directions,
                    rng.normal(size=(config["directions"], config["gate_dimension"])),
                ]
            terms = [
                np.asarray(row["signed_diagonal_losses"]),
                np.asarray(row["signed_off_losses"]),
            ]
            components = np.stack(
                [
                    np.mean(
                        (values[:, 0] - values[:, 1])[:, None] * directions,
                        axis=0,
                    )
                    / (2 * sigma)
                    for values in terms
                ]
            )
            components[0, 16:] = 0
            gradient = components.sum(axis=0)
            global_step = 121 + step
            adam_m = config["adam_beta1"] * adam_m + (1 - config["adam_beta1"]) * gradient
            adam_v = config["adam_beta2"] * adam_v + (1 - config["adam_beta2"]) * gradient**2
            update = rate * (
                adam_m / (1 - config["adam_beta1"] ** global_step)
            ) / (
                np.sqrt(adam_v / (1 - config["adam_beta2"] ** global_step))
                + config["adam_epsilon"]
            )
            for section, cap in [
                (slice(0, 16), config["diagonal_update_cap"]),
                (slice(16, 48), config["response_update_cap"]),
                *(
                    [(slice(48, dimension), config["gate_update_cap"])]
                    if arm == "candidate"
                    else []
                ),
            ]:
                norm = np.linalg.norm(update[section])
                update[section] *= min(1.0, cap / max(norm, 1e-12))
            theta = (theta - update).astype(np.float32)
            np.testing.assert_allclose(
                theta, saved["trajectory"][step], rtol=2e-7, atol=2e-7
            )
        np.testing.assert_array_equal(theta, saved["theta"])
        assert_close(adam_m, saved["adam_m"], atol=2e-12)
        assert_close(adam_v, saved["adam_v"], atol=2e-12)
        np.testing.assert_array_equal(saved["trajectory"][-1], saved["theta"])
        gate_gradient_steps = sum(row["gate_gradient_norm"] > 0 for row in history)
        gate_update_steps = sum(row["gate_update_norm"] > 0 for row in history)
        if arm == "candidate":
            effectiveness = read_json(f"{name}_gate_effectiveness.json")
            assert gate_gradient_steps == effectiveness["nonzero_gate_gradient_updates"] == 120
            assert gate_update_steps == effectiveness["nonzero_gate_parameter_updates"] == 120
            assert_close(
                np.linalg.norm(saved["theta"][48:]),
                effectiveness["final_gate_weight_norm"],
                atol=1e-7,
            )
        optimizer_checks[name] = {
            "updates_reconstructed": 120,
            "final_parameters_exact": True,
            "optimizer_moments_reconstructed": True,
            "nonzero_gate_gradient_updates": gate_gradient_steps,
            "nonzero_gate_parameter_updates": gate_update_steps,
        }

# Verify the new paired complete-episode set and its exact grouping.
paired = load_npz(OUT / "paired_evaluation.npz")
assert paired["s"].shape == (9600, 8)
assert set(np.unique(paired["episode"])) == set(range(48))
assert np.all(np.bincount(paired["episode"]) == 200)
assert np.array_equal(np.unique(paired["prefix_mode"], return_counts=True)[1], [4800, 4800])
for episode in range(48):
    expected_mode = 0 if episode % 2 == 0 else 1
    assert np.all(paired["prefix_mode"][paired["episode"] == episode] == expected_mode)
for offset in range(0, len(paired["s"]), 4):
    block = np.arange(offset, offset + 4)
    assert len(set(zip(paired["episode"][block], paired["time"][block]))) == 1
    assert set(paired["query"][block]) == {0, 1, 2, 3}
    np.testing.assert_array_equal(
        paired["s"][block], np.broadcast_to(paired["s"][block[0]], (4, 8))
    )
    np.testing.assert_array_equal(
        paired["xb"][block], np.broadcast_to(paired["xb"][block[0]], (4, 2))
    )
    query_zero = block[paired["query"][block] == 0][0]
    np.testing.assert_array_equal(paired["xq"][query_zero], paired["xb"][query_zero])

threshold = config["stationary_threshold"]
actual_stationary = (
    np.linalg.norm(paired["y"][:, :2] - paired["s"][:, :2], axis=1) <= threshold
)
diagonal = np.all(paired["xq"] == paired["xb"], axis=1)
q0_stationary = np.empty(len(diagonal), bool)
for offset in range(0, len(diagonal), 4):
    block = np.arange(offset, offset + 4)
    query_zero = block[paired["query"][block] == 0][0]
    q0_stationary[block] = actual_stationary[query_zero]
alive = ~paired["dead_before"]
groups = {
    "diagonal": diagonal,
    "all_off_diagonal": ~diagonal,
    "already_dead": (~diagonal) & (~alive),
    "alive_outside_goal": (~diagonal)
    & alive
    & (np.linalg.norm(paired["s"][:, :2] - GOAL_XY, axis=1) >= 2),
    "alive_moving": (~diagonal) & alive & (~actual_stationary),
    "fatal_onset": (~diagonal) & alive & paired["dead_after"],
    "alive_stationary": (~diagonal) & alive & actual_stationary,
    "alive_stationary_diagonal_moving_off_diagonal": (~diagonal)
    & alive
    & q0_stationary
    & (~actual_stationary),
}
saved_groups = load_npz(OUT / "one_step_groups.npz")
np.testing.assert_array_equal(saved_groups["episode"], paired["episode"])
np.testing.assert_array_equal(saved_groups["actual_stationary"], actual_stationary)
np.testing.assert_array_equal(saved_groups["q0_stationary"], q0_stationary)
for label, mask in groups.items():
    np.testing.assert_array_equal(saved_groups[label], mask)
assert {label: int(mask.sum()) for label, mask in groups.items()} == {
    "diagonal": 2527,
    "all_off_diagonal": 7073,
    "already_dead": 2080,
    "alive_outside_goal": 931,
    "alive_moving": 4993,
    "fatal_onset": 90,
    "alive_stationary": 0,
    "alive_stationary_diagonal_moving_off_diagonal": 0,
}

one_step_metrics = read_json("one_step_metrics.json")
one_step_arrays = {}
one_step_checks = {}
model_names = [
    "parent_s0",
    "control_s0",
    "candidate_s0",
    "parent_s1",
    "control_s1",
    "candidate_s1",
]
for name in model_names:
    saved = load_npz(OUT / f"{name}_evaluation.npz")
    assert saved["samples_xy"].shape == (9600, 64, 2)
    assert saved["atom"].dtype == np.bool_
    assert saved["atom"].shape == (9600, 64)
    recomputed_es = energy_score(saved["samples_xy"], paired["y"][:, :2])
    assert_close(recomputed_es, saved["es"], atol=2e-12)
    mean = saved["samples_xy"].mean(axis=1)
    np.testing.assert_array_equal(mean, saved["mean"])
    squared = np.sum((mean - paired["y"][:, :2]) ** 2, axis=1)
    euclidean = np.linalg.norm(mean - paired["y"][:, :2], axis=1)
    displacement = np.linalg.norm(
        saved["samples_xy"] - paired["s"][:, None, :2], axis=-1
    )
    np.testing.assert_array_equal(squared, saved["squared_error"])
    np.testing.assert_array_equal(euclidean, saved["euclidean_error"])
    np.testing.assert_array_equal(displacement, saved["displacement"])
    np.testing.assert_array_equal(displacement <= threshold, saved["stationary"])
    np.testing.assert_array_equal(native_legal(saved["samples_xy"]), saved["legal"])
    np.testing.assert_array_equal(native_legal(mean), saved["mean_legal"])
    assert saved["legal"].all()
    assert np.all((saved["h"] >= 0) & (saved["h"] <= 1))
    if not name.startswith("candidate"):
        np.testing.assert_array_equal(saved["h"], np.ones_like(saved["h"]))
    for label, mask in groups.items():
        reported = one_step_metrics[name][label]
        if not mask.any():
            assert reported == {"rows": 0, "episodes": 0, "available": False}
            continue
        assert reported["rows"] == int(mask.sum())
        assert reported["episodes"] == len(np.unique(paired["episode"][mask]))
        assert_close(recomputed_es[mask].mean(), reported["energy_score"])
        assert_close(np.sqrt(squared[mask].mean()), reported["mean_xy_rmse"], atol=1e-7)
        assert_close(euclidean[mask].mean(), reported["mean_xy_euclidean_error"], atol=1e-7)
        assert_close(saved["stationary"][mask].mean(), reported["sample_stationary_fraction"])
        assert_close(displacement[mask].mean(), reported["mean_sample_displacement"], atol=1e-7)
        assert_close((~saved["legal"][mask]).mean(), reported["illegal_sample_fraction"])
        assert_close((~saved["mean_legal"][mask]).mean(), reported["illegal_mean_fraction"])
        assert_close(saved["h"][mask].mean(), reported["gate"]["mean"], atol=1e-7)
    one_step_arrays[name] = saved
    one_step_checks[name] = {
        "full_energy_score_reconstructed": True,
        "mean_and_errors_reconstructed": True,
        "atom_shape_and_type_verified": True,
        "all_samples_native_legal": True,
        "mean_legality_separately_reconstructed": True,
        "gate_bounds_verified": True,
    }

# Reconstruct primary candidate-control energy-score and RMSE intervals using
# the predeclared whole-episode bootstrap stream.
rng = np.random.default_rng(config["bootstrap_seed"])
weights = rng.multinomial(
    48, np.full(48, 1 / 48), size=config["bootstrap_replicates"]
)
reported_contrasts = read_json("one_step_contrasts.json")
bootstrap_checks = {}
for seed in (0, 1):
    left = one_step_arrays[f"candidate_s{seed}"]
    right = one_step_arrays[f"control_s{seed}"]
    bootstrap_checks[f"candidate_minus_control_s{seed}"] = {}
    for label, mask in groups.items():
        if not mask.any():
            continue
        count = np.array([np.sum(mask & (paired["episode"] == episode)) for episode in range(48)])
        denominator = weights @ count
        valid = denominator > 0
        for metric, field, transform in [
            ("energy_score", "es", None),
            ("mean_xy_rmse", "squared_error", np.sqrt),
        ]:
            a = np.array(
                [left[field][mask & (paired["episode"] == episode)].sum() for episode in range(48)]
            )
            b = np.array(
                [right[field][mask & (paired["episode"] == episode)].sum() for episode in range(48)]
            )
            av = (weights[valid] @ a) / denominator[valid]
            bv = (weights[valid] @ b) / denominator[valid]
            point_a = left[field][mask].mean()
            point_b = right[field][mask].mean()
            if transform is not None:
                av, bv = transform(av), transform(bv)
                point_a, point_b = transform(point_a), transform(point_b)
            reported = reported_contrasts[str(seed)]["candidate_minus_control"][label][metric]
            assert_close(point_a - point_b, reported["difference"], atol=1e-12)
            assert_close(np.quantile(av - bv, [0.025, 0.975]), reported["ci95"], atol=1e-12)
        bootstrap_checks[f"candidate_minus_control_s{seed}"][label] = True

# Verify native and model return arrays and reconstruct all return contrasts.
native = load_npz(OUT / "native_actor_episodes.npz")
assert native["states"].shape == (48, 51, 8)
assert native["actions"].shape == (48, 50, 2)
assert native["rewards"].shape == (48, 50)
np.testing.assert_array_equal(native["states"][:, 1:, 2:], native["states"][:, :-1, :6])
native_reward = (
    np.linalg.norm(native["states"][:, 1:, :2] - GOAL_XY, axis=-1) < 2
).astype(np.float64)
np.testing.assert_array_equal(native_reward, native["rewards"])
native_return = np.sum(
    native_reward * config["return_discount"] ** np.arange(config["return_horizon"]),
    axis=1,
)
assert_close(native_return, native["returns"], atol=2e-12)
np.testing.assert_array_equal(
    native["environment_seed"],
    np.arange(48) + config["return_environment_seed"],
)

return_metrics = read_json("return_metrics.json")
return_contrasts = read_json("return_contrasts.json")
return_arrays = {}
return_checks = {}
discount32 = np.power(
    np.float32(config["return_discount"]),
    np.arange(config["return_horizon"]),
    dtype=np.float32,
)
for name in model_names:
    saved = load_npz(OUT / f"{name}_return_rollouts.npz")
    assert saved["states"].shape == (48, 64, 51, 8)
    np.testing.assert_array_equal(
        saved["states"][:, :, 0],
        np.broadcast_to(native["states"][:, None, 0], (48, 64, 8)),
    )
    np.testing.assert_array_equal(
        saved["states"][:, :, 1:, 2:], saved["states"][:, :, :-1, :6]
    )
    assert native_legal(saved["states"][:, :, 1:, :2]).all()
    reward = (
        np.linalg.norm(saved["states"][:, :, 1:, :2] - GOAL_XY, axis=-1) < 2
    ).astype(np.float32)
    np.testing.assert_array_equal(reward, saved["reward"])
    recomputed_return = np.sum(
        reward * discount32, axis=-1, dtype=np.float32
    )
    assert_close(recomputed_return, saved["return"], atol=1e-6)
    assert np.all((saved["h"] >= 0) & (saved["h"] <= 1))
    if not name.startswith("candidate"):
        np.testing.assert_array_equal(saved["h"], np.ones_like(saved["h"]))
    predicted = saved["return"].mean(axis=1).astype(np.float64)
    assert_close(predicted.mean(), return_metrics[name]["predicted_discounted_return"]["estimate"], atol=1e-7)
    assert_close((predicted - native_return).mean(), return_metrics[name]["return_bias_predicted_minus_native"]["estimate"], atol=1e-7)
    return_arrays[name] = {
        "predicted_return": predicted,
        "absolute_error": np.abs(predicted - native_return),
        "squared_error": (predicted - native_return) ** 2,
        "reward_time_mae": np.abs(saved["reward"].mean(axis=1) - native_reward).mean(axis=1),
    }
    return_checks[name] = {
        "common_native_roots_verified": True,
        "f4_shift_verified": True,
        "reward_reconstructed": True,
        "discounted_return_reconstructed": True,
        "all_successors_native_legal": True,
        "gate_bounds_verified": True,
    }

rng = np.random.default_rng(config["bootstrap_seed"] + 1)
weights = rng.multinomial(
    48, np.full(48, 1 / 48), size=config["bootstrap_replicates"]
)
for seed in (0, 1):
    left = return_arrays[f"candidate_s{seed}"]
    right = return_arrays[f"control_s{seed}"]
    for metric, transform in [
        ("predicted_return", None),
        ("absolute_error", None),
        ("squared_error", np.sqrt),
        ("reward_time_mae", None),
    ]:
        point_left, point_right = left[metric].mean(), right[metric].mean()
        replicate_left = weights @ left[metric] / weights.sum(axis=1)
        replicate_right = weights @ right[metric] / weights.sum(axis=1)
        if transform is not None:
            point_left, point_right = transform(point_left), transform(point_right)
            replicate_left, replicate_right = transform(replicate_left), transform(replicate_right)
        report_key = {
            "absolute_error": "absolute_return_error",
            "squared_error": "return_rmse",
        }.get(metric, metric)
        reported = return_contrasts[str(seed)]["candidate_minus_control"][report_key]
        assert_close(point_left - point_right, reported["difference"], atol=1e-12)
        assert_close(
            np.quantile(replicate_left - replicate_right, [0.025, 0.975]),
            reported["ci95"],
            atol=1e-12,
        )

checks = read_json("invariant_checks.json")
assert all(row["diagonal_identity"] for row in checks.values())
assert all(row["f4_shift"] for row in checks.values())
assert all(row["coupled_anchor_and_atom"] for row in checks.values())
assert all(row["query_independent_h_under_coupling"] for row in checks.values())
assert all(row["coupled_action_ratio_max"] <= 1 + 2e-6 for row in checks.values())
assert all(row["all_samples_native_legal"] for row in checks.values())

result = {
    "status": "pass",
    "verification_scope": "saved arrays and hashed inputs only; zero model, actor, nominal, or environment calls",
    "hash_contract": {
        "sealed_protocol_config_collector": True,
        "sealed_original_driver_recorded": True,
        "current_driver_matches_final_resume_record": True,
        "other_dependency_hashes_unchanged": True,
        "checked_dependency_count": len(hash_checks),
    },
    "budget": {
        "model_successors": ledger["model"]["total"],
        "model_cap": config["model_successor_cap"],
        "failed_verification_overhead": completion["failed_verification_model_successor_overhead"],
        "native_steps": ledger["native"]["total"],
        "native_cap": config["native_step_cap"],
        "extra_calls_from_this_verification": 0,
    },
    "optimizer": optimizer_checks,
    "paired_evaluation": {
        "episodes": 48,
        "rows": 9600,
        "complete_steps_per_episode": 50,
        "queries_per_context": 4,
        "teacher_prefix_episodes": 24,
        "blind_prefix_episodes": 24,
        "row_alignment_verified": True,
        "groups_reconstructed": {label: int(mask.sum()) for label, mask in groups.items()},
    },
    "one_step": one_step_checks,
    "one_step_candidate_control_bootstraps_reconstructed": bootstrap_checks,
    "returns": {
        "native_rewards_and_returns_reconstructed": True,
        "candidate_control_bootstraps_reconstructed": True,
        "models": return_checks,
    },
    "saved_invariants_confirmed": True,
}
write_json("verification.json", result)
print(json.dumps({"status": "pass", "model_successors": ledger["model"]["total"], "native_steps": ledger["native"]["total"]}))
