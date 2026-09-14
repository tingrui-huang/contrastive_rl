"""Independent saved-array verifier; makes no model or environment calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def hazardous(xy):
    xy = np.asarray(xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


def groups(state):
    xy = state[:, :2]
    approach = (xy[:, 0] >= 2) & (xy[:, 0] < 3) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    inside = hazardous(xy)
    right = (xy[:, 0] >= 6) & (xy[:, 0] < 7) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    return {
        "all": np.ones(len(state), bool),
        "approach_left": approach,
        "inside_hazard": inside,
        "right_exit": right,
        "elsewhere": ~(approach | inside | right),
    }


def assert_close(left, right, atol=2e-9):
    np.testing.assert_allclose(float(left), float(right), rtol=0, atol=atol)


def check_binary_point(saved, probability, target, mask):
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    assert saved["rows"] == int(mask.sum())
    assert saved["positive_events"] == int(target[mask].sum())
    if not mask.any():
        assert saved["observed_rate"]["estimate"] is None
        return
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    assert_close(saved["observed_rate"]["estimate"], target[mask].mean())
    assert_close(saved["predicted_rate"]["estimate"], probability[mask].mean())
    assert_close(
        saved["calibration_in_the_large_predicted_minus_observed"]["estimate"],
        (probability[mask] - target[mask]).mean(),
    )
    assert_close(saved["brier"]["estimate"], ((probability[mask] - target[mask]) ** 2).mean())
    logloss = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    assert_close(saved["log_loss"]["estimate"], logloss[mask].mean())


def first_components(positions):
    positions = np.asarray(positions)
    if positions.ndim == 3:
        positions = positions[:, None]
    hit = hazardous(positions)
    entered = hit.any(axis=-1)
    first = np.argmax(hit, axis=-1)
    index = np.broadcast_to(first[..., None, None], positions.shape[:2] + (1, 2))
    xy = np.take_along_axis(positions, index, axis=2)[..., 0, :]
    xy = np.where(entered[..., None], xy, np.nan)
    paths = entered.shape[1]
    denominator = entered.sum(axis=1) / paths
    return {
        "probability": entered.mean(axis=1),
        "time_num": np.where(entered, first, 0).sum(axis=1) / paths,
        "denominator": denominator,
        "x_num": np.where(entered, xy[..., 0], 0).sum(axis=1) / paths,
        "y_num": np.where(entered, xy[..., 1], 0).sum(axis=1) / paths,
        "cell3": ((np.floor(xy[..., 0]) == 3) & entered).sum(axis=1) / paths,
        "cell4": ((np.floor(xy[..., 0]) == 4) & entered).sum(axis=1) / paths,
        "cell5": ((np.floor(xy[..., 0]) == 5) & entered).sum(axis=1) / paths,
    }


def check_first(saved, component):
    assert_close(saved["probability"]["estimate"], component["probability"].mean())
    den = component["denominator"].sum()
    assert_close(saved["time_zero_based_conditional_on_entry"]["estimate"], component["time_num"].sum() / den)
    assert_close(saved["mean_x_conditional_on_entry"]["estimate"], component["x_num"].sum() / den, 1e-6)
    assert_close(saved["mean_y_conditional_on_entry"]["estimate"], component["y_num"].sum() / den, 1e-6)
    for cell in (3, 4, 5):
        assert_close(saved["cell_probability_conditional_on_entry"][str(cell)]["estimate"], component[f"cell{cell}"].sum() / den)


def main():
    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    for path, digest in provenance["dependency_sha256"].items():
        assert sha256(path) == digest, path
    completion = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    assert sha256(OUT / "provenance.json") == completion["provenance_sha256"]
    assert sha256(OUT / "replay_verification.json") == completion["replay_verification_sha256"]
    assert sha256(OUT / "matched_metrics.json") == completion["matched_metrics_sha256"]
    assert sha256(OUT / "first_hazard_entry.json") == completion["first_hazard_entry_sha256"]
    assert sha256(OUT / "exposure_metrics.json") == completion["exposure_metrics_sha256"]

    replay_check = json.loads((OUT / "replay_verification.json").read_text(encoding="utf-8"))
    replay = load_npz(OUT / "augmented_shadow_replay.npz")
    native = load_npz(SOURCE / "native_actor_episodes.npz")
    assert replay_check["status"] == "pass" and replay_check["rows"] == 2400
    assert sha256(OUT / "augmented_shadow_replay.npz") == replay_check["output_sha256"]
    episode = np.repeat(np.arange(48), 50)
    time_index = np.tile(np.arange(50), 48)
    np.testing.assert_array_equal(replay["episode"], episode)
    np.testing.assert_array_equal(replay["time"], time_index)
    np.testing.assert_array_equal(replay["s"], native["states"][:, :-1].reshape(-1, 8))
    np.testing.assert_array_equal(replay["xq_saved_actor"], native["actions"].reshape(-1, 2))
    np.testing.assert_array_equal(replay["y_real"], native["states"][:, 1:].reshape(-1, 8))
    for name in ("dead_before", "dead_after", "onset"):
        np.testing.assert_array_equal(replay[name], native[name].reshape(-1))
    np.testing.assert_array_equal(replay["reward"], native["rewards"].reshape(-1))

    alive = ~replay["dead_before"]
    data = {name: value[alive] for name, value in replay.items()}
    assert len(data["episode"]) == 1165 and int(data["onset"].sum()) == 28
    group = groups(data["s"])
    saved_groups = load_npz(OUT / "matched_groups.npz")
    for name, mask in group.items():
        np.testing.assert_array_equal(saved_groups[name], mask)
    actual_hazard = hazardous(data["y_real"][:, :2])
    np.testing.assert_array_equal(saved_groups["actual_hazard"], actual_hazard)

    metrics = json.loads((OUT / "matched_metrics.json").read_text(encoding="utf-8"))
    actual_prediction = load_npz(OUT / "actual_head_predictions.npz")["probability"]
    actual_masks = dict(group)
    actual_masks["actual_hazardous_landing"] = actual_hazard
    actual_masks["actual_nonhazardous_landing"] = ~actual_hazard
    for head_seed in (0, 1):
        for name, mask in actual_masks.items():
            check_binary_point(
                metrics["check_A_actual_successor_failure"][f"h{head_seed}"][name],
                actual_prediction[:, head_seed], data["onset"], mask,
            )

    threshold = CONFIG["stationary_threshold"]
    current = data["s"][:, :2]
    real = data["y_real"][:, :2]
    real_delta = real - current
    current_hazard = hazardous(current)
    real_events = {
        "landing_hazard": hazardous(real),
        "hazard_entry_from_outside": hazardous(real),
        "remain_in_hazard": hazardous(real),
        "exit_hazard": ~hazardous(real),
        "rightward_progress": real_delta[:, 0] > threshold,
        "stationary": np.linalg.norm(real_delta, axis=1) <= threshold,
    }
    eligibility = {
        "landing_hazard": np.ones(len(real), bool),
        "hazard_entry_from_outside": ~current_hazard,
        "remain_in_hazard": current_hazard,
        "exit_hazard": current_hazard,
        "rightward_progress": np.ones(len(real), bool),
        "stationary": np.ones(len(real), bool),
    }
    joint = load_npz(OUT / "joint_predictions.npz")
    for position_seed in (0, 1):
        samples = load_npz(OUT / f"position_p{position_seed}_samples.npz")
        assert samples["samples_xy"].shape == (1165, 64, 2)
        assert samples["legal"].all()
        delta = samples["samples_xy"] - current[:, None]
        sample_hazard = hazardous(samples["samples_xy"])
        predicted_events = {
            "landing_hazard": sample_hazard.mean(axis=1),
            "hazard_entry_from_outside": sample_hazard.mean(axis=1),
            "remain_in_hazard": sample_hazard.mean(axis=1),
            "exit_hazard": (~sample_hazard).mean(axis=1),
            "rightward_progress": (delta[..., 0] > threshold).mean(axis=1),
            "stationary": (np.linalg.norm(delta, axis=-1) <= threshold).mean(axis=1),
        }
        for group_name, group_mask in group.items():
            for event_name in real_events:
                check_binary_point(
                    metrics["check_B_position"][f"p{position_seed}"]["groups"][group_name][event_name],
                    predicted_events[event_name], real_events[event_name], group_mask & eligibility[event_name],
                )
        for head_seed in (0, 1):
            key = f"p{position_seed}_h{head_seed}"
            probability = joint[f"sample_probability_{key}"]
            q = probability.mean(axis=1)
            q_hazard = (probability * sample_hazard).mean(axis=1)
            q_nonhazard = (probability * ~sample_hazard).mean(axis=1)
            np.testing.assert_allclose(joint[f"q_{key}"], q, rtol=0, atol=1e-7)
            np.testing.assert_allclose(joint[f"q_hazard_{key}"], q_hazard, rtol=0, atol=1e-7)
            np.testing.assert_allclose(joint[f"q_nonhazard_{key}"], q_nonhazard, rtol=0, atol=1e-7)
            for group_name, mask in group.items():
                saved = metrics["check_C_joint"][f"p{position_seed}"][f"h{head_seed}"]["groups"][group_name]
                check_binary_point(saved["joint_onset"], q, data["onset"], mask)
                check_binary_point(saved["hazardous_generated_landing_component"], q_hazard, data["onset"] & actual_hazard, mask)
                check_binary_point(saved["nonhazardous_generated_landing_component"], q_nonhazard, data["onset"] & ~actual_hazard, mask)

    first = json.loads((OUT / "first_hazard_entry.json").read_text(encoding="utf-8"))
    native_first = first_components(native["states"][:, 1:, :2])
    check_first(first["native"], native_first)
    for position_seed in (0, 1):
        baseline = load_npz(SOURCE / f"baseline_p{position_seed}_rollouts.npz")
        check_first(first[f"p{position_seed}"], first_components(baseline["states"][:, :, 1:, :2]))

    exposure = json.loads((OUT / "exposure_metrics.json").read_text(encoding="utf-8"))
    exposure_arrays = load_npz(OUT / "exposure_predictions.npz")
    native_opportunities = (hazardous(native["states"][:, 1:, :2]) & ~native["dead_before"]).sum(axis=1)
    np.testing.assert_array_equal(exposure_arrays["native_at_risk_opportunities"], native_opportunities)
    assert_close(exposure["native"]["at_risk_hazardous_opportunities"]["estimate"], native_opportunities.mean())
    for position_seed in (0, 1):
        baseline = load_npz(SOURCE / f"baseline_p{position_seed}_rollouts.npz")
        hazard = hazardous(baseline["states"][:, :, 1:, :2])
        for head_seed in (0, 1):
            key = f"p{position_seed}_h{head_seed}"
            probability = exposure_arrays[f"failure_probability_{key}"]
            survival_before = np.concatenate(
                [np.ones(probability.shape[:-1] + (1,)), np.cumprod(1 - probability[..., :-1], axis=-1)], axis=-1
            )
            opportunities = (survival_before * hazard).sum(axis=-1).mean(axis=1)
            np.testing.assert_allclose(
                exposure_arrays[f"survival_weighted_opportunities_{key}"], opportunities, rtol=0, atol=2e-6
            )
            assert_close(exposure[key]["survival_weighted_at_risk_hazardous_opportunities"]["estimate"], opportunities.mean(), 2e-7)

    ledger = json.loads((OUT / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["native_replay_steps"]["total"] == 2400
    assert ledger["position_successors"]["total"] == 149120
    assert ledger["head_inference_rows"]["total"] == 914970
    assert ledger["training_updates"] == 0
    assert ledger["new_complete_evaluation_episodes"] == 0
    assert ledger["new_complete_model_rollouts"] == 0
    result = {
        "status": "pass",
        "saved_only": True,
        "environment_model_actor_nominal_calls": 0,
        "all_2400_replay_rows_match_saved_native_arrays": True,
        "alive_rows": 1165,
        "position_successors_verified": 149120,
        "joint_decompositions_verified": 4,
        "first_entry_and_survival_weighted_exposure_reconstructed": True,
        "verifier_sha256": sha256(OUT / "verify_saved.py"),
        "report_sha256": sha256(OUT / "REPORT.md") if (OUT / "REPORT.md").exists() else None,
        "artifact_sha256": {
            name: sha256(OUT / name)
            for name in (
                "augmented_shadow_replay.npz",
                "actual_head_predictions.npz",
                "position_p0_samples.npz",
                "position_p1_samples.npz",
                "joint_predictions.npz",
                "exposure_predictions.npz",
                "matched_metrics.json",
                "first_hazard_entry.json",
                "exposure_metrics.json",
                "ledger.json",
            )
        },
    }
    write = json.dumps(result, indent=2) + "\n"
    (OUT / "verification.json").write_text(write, encoding="utf-8")
    print(write, end="")


if __name__ == "__main__":
    main()
