"""Independent saved-array checks for the completed PointMaze repair experiment.

This script performs no native stepping, model sampling, training, or checkpoint
selection.  It only reads the frozen arrays and JSON results produced by run.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
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
    dtype=np.int32,
)


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_npz(name):
    with np.load(OUT / name, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


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


def native_legal(xy):
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


def full_energy_score(samples, target):
    samples = np.asarray(samples, np.float64)
    target = np.asarray(target, np.float64)
    result = np.empty(len(samples), np.float64)
    count = samples.shape[1]
    for start in range(0, len(samples), 128):
        stop = min(start + 128, len(samples))
        value = samples[start:stop]
        first = np.linalg.norm(value - target[start:stop, None], axis=-1).mean(axis=1)
        pair = np.linalg.norm(value[:, :, None] - value[:, None, :], axis=-1)
        result[start:stop] = first - pair.sum(axis=(1, 2)) / (2 * count * (count - 1))
    return result


def close(actual, expected, label, atol=2e-10):
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def first_event(event, positions):
    event = np.asarray(event, bool)
    positions = np.asarray(positions)
    entered = event.any(axis=-1)
    first = np.argmax(event, axis=-1)
    index = np.broadcast_to(first[..., None, None], positions.shape[:-2] + (1, 2))
    xy = np.take_along_axis(positions, index, axis=-2)[..., 0, :]
    xy = np.where(entered[..., None], xy, 0)
    return entered, first, xy


def verify_dependencies(checks):
    provenance = load_json("provenance.json")
    for name, expected in provenance["dependency_sha256"].items():
        if sha256(name) != expected:
            raise AssertionError(f"dependency changed: {name}")
    frozen = load_json("checkpoints_frozen.json")
    names = {
        "position_s0": "position_repaired_s0.npz",
        "position_s1": "position_repaired_s1.npz",
        "head_s0": "head_repaired_s0.npz",
        "head_s1": "head_repaired_s1.npz",
    }
    for key, name in names.items():
        if sha256(OUT / name) != frozen["checkpoint_sha256"][key]:
            raise AssertionError(f"frozen checkpoint changed: {name}")
    checks["dependency_hashes_unchanged"] = True
    checks["frozen_checkpoint_hashes_match"] = True


def verify_splits(checks):
    manifests = {"split_unit": "complete episode", "splits": {}}
    expected_sources = {
        "train": {0: 64, 1: 64},
        "validation": {0: 16, 1: 16},
        "final": {0: 0, 1: 64},
    }
    for split in ("train", "validation", "final"):
        data = load_npz(f"{split}_paired.npz")
        spec = CONFIG["splits"][split]
        episodes = spec["episodes"]
        horizon = CONFIG["horizon"]
        expected_rows = episodes * horizon * 4
        if len(data["episode"]) != expected_rows:
            raise AssertionError(f"{split}: wrong row count")
        if data["prefix_states"].shape != (episodes, horizon + 1, 8):
            raise AssertionError(f"{split}: wrong prefix state shape")
        if data["prefix_actions"].shape != (episodes, horizon, 2):
            raise AssertionError(f"{split}: wrong prefix action shape")
        np.testing.assert_array_equal(data["prefix_environment_seed"], spec["environment_seed_base"] + np.arange(episodes))
        sources = data["prefix_prefix_source"]
        for source, count in expected_sources[split].items():
            if int((sources == source).sum()) != count:
                raise AssertionError(f"{split}: wrong prefix-source balance")
        episode_manifest = []
        for episode in range(episodes):
            mask = data["episode"] == episode
            if int(mask.sum()) != horizon * 4:
                raise AssertionError(f"{split} episode {episode}: incomplete rows")
            np.testing.assert_array_equal(data["time"][mask], np.repeat(np.arange(horizon), 4))
            np.testing.assert_array_equal(data["query_type"][mask], np.tile(np.arange(4), horizon))
            source = int(sources[episode])
            if not np.all(data["prefix_source"][mask] == source):
                raise AssertionError(f"{split} episode {episode}: source mismatch")
            episode_manifest.append(
                {
                    "episode": episode,
                    "environment_seed": int(data["prefix_environment_seed"][episode]),
                    "prefix_source": "expert" if source == 0 else "actor",
                    "paired_rows": int(mask.sum()),
                    "prefix_onsets": int(data["prefix_onset"][episode].sum()),
                    "prefix_reward_occurrence": bool(data["prefix_rewards"][episode].any()),
                }
            )
        context_shape = (episodes, horizon, 4)
        for key in ("s", "xb", "dead_before"):
            value = data[key].reshape(context_shape + data[key].shape[1:])
            np.testing.assert_array_equal(value, np.broadcast_to(value[:, :, :1], value.shape))
        xq = data["xq"].reshape(episodes, horizon, 4, 2)
        xb = data["xb"].reshape(episodes, horizon, 4, 2)
        np.testing.assert_array_equal(xq[:, :, 0], xb[:, :, 0])
        if not np.array_equal(data["onset"], (~data["dead_before"]) & data["dead_after"]):
            raise AssertionError(f"{split}: onset/death inconsistency")
        np.testing.assert_array_equal(data["y"][:, 2:], data["s"][:, :6])
        chosen_query = np.where(sources[:, None] == 0, 0, 1)
        row = (np.arange(episodes)[:, None] * horizon + np.arange(horizon)[None, :]) * 4 + chosen_query
        np.testing.assert_array_equal(data["y"][row], data["prefix_states"][:, 1:])
        np.testing.assert_array_equal(data["reward"][row], data["prefix_rewards"])
        np.testing.assert_array_equal(data["dead_before"][row], data["prefix_dead_before"])
        np.testing.assert_array_equal(data["dead_after"][row], data["prefix_dead_after"])
        np.testing.assert_array_equal(data["onset"][row], data["prefix_onset"])
        collection = load_json(f"{split}_collection.json")
        if sha256(OUT / f"{split}_paired.npz") != collection["output_sha256"]:
            raise AssertionError(f"{split}: collection hash mismatch")
        manifests["splits"][split] = {
            "episodes": episodes,
            "paired_rows": expected_rows,
            "environment_seed_base": spec["environment_seed_base"],
            "teacher_seed_base": spec["teacher_seed_base"],
            "actor_seed_base": spec["actor_seed_base"],
            "random_seed_base": spec["random_seed_base"],
            "array_sha256": collection["output_sha256"],
            "episodes_manifest": episode_manifest,
        }
    seed_ranges = []
    for split in ("train", "validation", "final"):
        spec = CONFIG["splits"][split]
        for kind in ("environment", "teacher", "actor", "random"):
            lo = spec[f"{kind}_seed_base"]
            seed_ranges.append((split, kind, lo, lo + spec["episodes"] - 1))
    for index, left in enumerate(seed_ranges):
        for right in seed_ranges[index + 1 :]:
            if left[1] == right[1] and max(left[2], right[2]) <= min(left[3], right[3]):
                raise AssertionError(f"overlapping {left[1]} seed ranges: {left} {right}")
    (OUT / "split_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    checks["complete_episode_splits_and_four_query_rows_verified"] = True
    checks["designated_branch_equals_saved_prefix_transition"] = True
    checks["split_seed_ranges_disjoint"] = True


def verify_matched_metrics(checks):
    full = load_npz("final_paired.npz")
    alive = ~full["dead_before"]
    data = {key: full[key][alive] for key in ("episode", "s", "xb", "xq", "y", "onset")}
    if len(data["episode"]) != 4864 or int(data["onset"].sum()) != 119:
        raise AssertionError("final alive scope changed")
    actual_hazard = hazardous(data["y"][:, :2])
    diagonal = np.all(data["xq"] == data["xb"], axis=1)

    head_predictions = load_npz("final_actual_head_predictions.npz")
    head_metrics = load_json("final_actual_head_metrics.json")
    np.testing.assert_array_equal(head_predictions["target"], data["onset"])
    np.testing.assert_array_equal(head_predictions["actual_hazard"], actual_hazard)
    for seed in (0, 1):
        for variant in ("old", "repaired"):
            probability = head_predictions[f"probability_s{seed}_{variant}"].astype(np.float64)
            if np.any(probability[~actual_hazard] != 0):
                raise AssertionError("actual-successor probability outside support")
            target = data["onset"].astype(np.float64)
            saved = head_metrics[f"s{seed}"][variant]
            for name, mask in (("all", np.ones(len(target), bool)), ("actual_hazardous_landing", actual_hazard)):
                close(probability[mask].mean(), saved[name]["predicted_rate"]["estimate"], f"head {seed} {variant} {name} rate")
                close((probability[mask] - target[mask]).mean(), saved[name]["calibration_bias"]["estimate"], f"head {seed} {variant} {name} bias")
                close(((probability[mask] - target[mask]) ** 2).mean(), saved[name]["brier"]["estimate"], f"head {seed} {variant} {name} brier")

    position_metrics = load_json("final_position_metrics.json")
    position_arrays = {}
    for seed in (0, 1):
        for variant in ("old", "repaired"):
            name = f"final_position_s{seed}_{variant}.npz"
            saved_array = load_npz(name)
            samples = saved_array["samples_xy"]
            if samples.shape != (4864, 64, 2):
                raise AssertionError(f"{name}: wrong shape")
            if not native_legal(samples).all() or not saved_array["legal"].all():
                raise AssertionError(f"{name}: illegal sample")
            score = full_energy_score(samples, data["y"][:, :2])
            position_arrays[(seed, variant)] = saved_array
            saved_metric = position_metrics[f"s{seed}"][variant]["groups"]
            close(score.mean(), saved_metric["all"]["energy_score"]["estimate"], f"position {seed} {variant} all energy")
            close(score[diagonal].mean(), saved_metric["exact_diagonal"]["energy_score"]["estimate"], f"position {seed} {variant} diagonal energy")
            close(score[~diagonal].mean(), saved_metric["off_diagonal"]["energy_score"]["estimate"], f"position {seed} {variant} off-diagonal energy")

    joint = load_npz("final_joint_predictions.npz")
    joint_metrics = load_json("final_joint_metrics.json")
    np.testing.assert_array_equal(joint["target"], data["onset"].astype(float))
    target = data["onset"].astype(np.float64)
    for seed in (0, 1):
        for position_variant in ("old", "repaired"):
            support = hazardous(position_arrays[(seed, position_variant)]["samples_xy"])
            for head_variant in ("old", "repaired"):
                arm = f"p{position_variant}_h{head_variant}"
                sample_probability = joint[f"sample_probability_s{seed}_{arm}"].astype(np.float64)
                q = joint[f"q_s{seed}_{arm}"].astype(np.float64)
                if np.any(sample_probability[~support] != 0):
                    raise AssertionError(f"joint probability outside support: seed {seed} {arm}")
                # q was reduced by JAX in float32; the independent NumPy
                # reduction above promotes to float64.
                np.testing.assert_allclose(q, sample_probability.mean(axis=1), rtol=0, atol=2e-7)
                saved = joint_metrics[f"s{seed}"]["arms"][arm]["all"]["onset"]
                close(q.mean(), saved["predicted_rate"]["estimate"], f"joint {seed} {arm} rate", atol=2e-9)
                close(((q - target) ** 2).mean(), saved["brier"]["estimate"], f"joint {seed} {arm} brier", atol=2e-9)
    checks["actual_successor_head_metrics_recomputed"] = True
    checks["full_energy_u_statistic_recomputed_with_all_sample_pairs"] = True
    checks["matched_joint_integration_recomputed"] = True
    checks["all_matched_head_probabilities_zero_outside_hazard"] = True


def trajectory_point_metrics(record, native=False):
    if native:
        reward = record["prefix_rewards"]
        failed_before = record["prefix_dead_before"]
        failed_after = record["prefix_dead_after"]
        onset = record["prefix_onset"]
        positions = record["prefix_states"][:, 1:, :2]
        returns = np.sum(reward * CONFIG["discount"] ** np.arange(CONFIG["horizon"]), axis=-1)
        reward_any = reward.any(axis=-1)
        failed_any = failed_after.any(axis=-1)
        hazard = hazardous(positions) & ~failed_before
    else:
        reward = record["reward"]
        failed_before = record["failed_before"]
        failed_after = record["failed_after"]
        onset = record["onset"]
        positions = record["position_proposal_xy"]
        returns = record["return"]
        reward_any = reward.any(axis=-1)
        failed_any = failed_after.any(axis=-1)
        hazard = hazardous(positions) & ~failed_before
    survival = (~reward_any) & (~failed_any)
    onset_any, onset_time, _ = first_event(onset, positions)
    hazard_any, hazard_time, hazard_xy = first_event(hazard, positions)
    result = {
        "reward_occurrence": reward_any.mean(),
        "failure_probability": failed_any.mean(),
        "survival_without_reward": survival.mean(),
        "mean_discounted_return": returns.mean(),
        "return_conditional_on_reward": returns[reward_any].mean(),
        "onset_time_zero_based_conditional_on_failure": onset_time[onset_any].mean(),
        "first_hazard_probability": hazard_any.mean(),
        "first_hazard_time_zero_based_conditional": hazard_time[hazard_any].mean(),
        "first_hazard_x": hazard_xy[..., 0][hazard_any].mean(),
        "first_hazard_y": hazard_xy[..., 1][hazard_any].mean(),
        "at_risk_hazardous_opportunities": hazard.sum(axis=-1).mean(),
    }
    return {key: float(value) for key, value in result.items()}


def verify_rollouts(checks):
    saved_metrics = load_json("trajectory_metrics.json")
    final = load_npz("final_paired.npz")
    native = trajectory_point_metrics(final, native=True)
    mapping = {
        "first_hazard_x": ("first_hazard_mean_xy", "x"),
        "first_hazard_y": ("first_hazard_mean_xy", "y"),
    }
    for key, value in native.items():
        if key in mapping:
            outer, inner = mapping[key]
            expected = saved_metrics["native"][outer][inner]["estimate"]
        else:
            expected = saved_metrics["native"][key]["estimate"]
        close(value, expected, f"native trajectory {key}", atol=1e-6)
    initial = final["prefix_states"][:, 0, :]
    for seed in (0, 1):
        for position_variant in ("old", "repaired"):
            for head_variant in ("old", "repaired"):
                arm = f"p{position_variant}_h{head_variant}"
                record = load_npz(f"rollout_s{seed}_{arm}.npz")
                if record["states"].shape != (64, 64, 51, 8):
                    raise AssertionError(f"rollout seed {seed} {arm}: wrong shape")
                np.testing.assert_array_equal(record["states"][:, :, 0], np.broadcast_to(initial[:, None], record["states"][:, :, 0].shape))
                np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
                if not native_legal(record["states"][..., :2]).all():
                    raise AssertionError(f"rollout seed {seed} {arm}: illegal state")
                if np.any(record["failure_probability"][~record["hazard_landing"]] != 0):
                    raise AssertionError(f"rollout seed {seed} {arm}: outside-support failure")
                np.testing.assert_array_equal(record["failed_after"], record["failed_before"] | record["onset"])
                if np.any(np.diff(record["failed"].astype(np.int8), axis=-1) < 0):
                    raise AssertionError(f"rollout seed {seed} {arm}: failure reversal")
                frozen = record["failed_before"]
                np.testing.assert_array_equal(record["states"][:, :, 1:, :2][frozen], record["states"][:, :, :-1, :2][frozen])
                np.testing.assert_array_equal(record["states"][:, :, 1:, :2][record["onset"]], record["position_proposal_xy"][record["onset"]])
                if np.any(record["reward"][record["failed_after"]] != 0):
                    raise AssertionError(f"rollout seed {seed} {arm}: failed reward nonzero")
                calculated = trajectory_point_metrics(record)
                saved = saved_metrics["arms"][f"s{seed}"][arm]["metrics"]
                for key, value in calculated.items():
                    if key in mapping:
                        outer, inner = mapping[key]
                        expected = saved[outer][inner]["estimate"]
                    else:
                        expected = saved[key]["estimate"]
                    close(value, expected, f"rollout seed {seed} {arm} {key}", atol=2e-6)
    checks["all_eight_trajectory_point_summaries_recomputed"] = True
    checks["irreversible_failure_and_absorbing_xy_verified"] = True
    checks["fatal_incoming_motion_retained"] = True
    checks["f4_shift_and_zero_failed_reward_verified"] = True


def verify_accounting(checks):
    ledger = load_json("ledger.json")
    expected = {
        "native_steps": (56000, CONFIG["native_step_cap"]),
        "position_successors": (21547520, CONFIG["position_successor_cap"]),
        "training_updates": (3480, CONFIG["training_update_cap"]),
        "complete_model_paths": (32768, CONFIG["complete_model_path_cap"]),
    }
    for name, (total, cap) in expected.items():
        if ledger[name]["total"] != total or ledger[name]["total"] > cap:
            raise AssertionError(f"ledger mismatch: {name}")
    if ledger["head_rows"]["total"] != 4264560:
        raise AssertionError("head-row ledger mismatch")
    acceptance = load_json("acceptance.json")
    if acceptance["overall"] != {
        "head_passed_both_seeds": True,
        "position_passed_both_seeds": False,
        "combined_passed_both_seeds": False,
    }:
        raise AssertionError("acceptance status changed")
    checks["ledger_within_sealed_caps"] = True
    checks["predeclared_acceptance_status_verified"] = True


def main():
    checks = {}
    verify_dependencies(checks)
    verify_splits(checks)
    verify_matched_metrics(checks)
    verify_rollouts(checks)
    verify_accounting(checks)
    result = {
        "status": "passed",
        "saved_arrays_only": True,
        "new_native_steps": 0,
        "new_model_samples": 0,
        "new_training_updates": 0,
        "checks": checks,
        "split_manifest_sha256": sha256(OUT / "split_manifests.json"),
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
