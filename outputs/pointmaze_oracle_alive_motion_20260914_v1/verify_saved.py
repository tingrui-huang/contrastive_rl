"""Independent saved-array verification; performs no simulator or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from oracle_motion import WALLS, hazardous, native_alive_motion


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
REPAIR = ROOT / "outputs" / "pointmaze_supervised_repair_20260914_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_legal(xy):
    xy = np.asarray(xy)
    floor = np.clip(np.floor(xy).astype(np.int64), [0, 0], np.array(WALLS.shape) - 1)
    bounded = (xy[..., 0] >= 0) & (xy[..., 0] <= 9) & (xy[..., 1] >= 0) & (xy[..., 1] <= 5)
    return bounded & (WALLS[floor[..., 0], floor[..., 1]] == 0)


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
    terms["at_risk_hazardous_opportunities"] = (root_mean(hazard_event.sum(axis=-1).astype(float)), ones)
    return {name: (np.asarray(num, np.float64), np.asarray(den, np.float64)) for name, (num, den) in terms.items()}


def estimate(term, weights):
    numerator, denominator = term
    point = float(numerator.sum() / denominator.sum())
    boot_den = weights @ denominator
    replicates = (weights @ numerator) / boot_den
    return point, replicates


def summary(terms, weights):
    result = {}
    for name, term in terms.items():
        point, boot = estimate(term, weights)
        result[name] = {"estimate": point, "ci95": np.quantile(boot, [0.025, 0.975])}
    return result


def comparison(left, right, weights):
    result = {}
    for name in left:
        lp, lb = estimate(left[name], weights)
        rp, rb = estimate(right[name], weights)
        result[name] = {"estimate": lp - rp, "ci95": np.quantile(lb - rb, [0.025, 0.975])}
    return result


def absolute_error(left, native, weights):
    result = {}
    for name in left:
        lp, lb = estimate(left[name], weights)
        np_, nb = estimate(native[name], weights)
        result[name] = {"estimate": abs(lp - np_), "ci95": np.quantile(np.abs(lb - nb), [0.025, 0.975])}
    return result


def absolute_error_change(oracle, learned, native, weights):
    result = {}
    for name in oracle:
        op, ob = estimate(oracle[name], weights)
        lp, lb = estimate(learned[name], weights)
        np_, nb = estimate(native[name], weights)
        result[name] = {
            "estimate": abs(op - np_) - abs(lp - np_),
            "ci95": np.quantile(np.abs(ob - nb) - np.abs(lb - nb), [0.025, 0.975]),
            "negative_means_oracle_is_closer": True,
        }
    return result


def assert_tree_close(calculated, saved, label):
    for name, value in calculated.items():
        expected = saved[name]
        np.testing.assert_allclose(value["estimate"], expected["estimate"], rtol=0, atol=3e-7, err_msg=f"{label} {name} estimate")
        np.testing.assert_allclose(value["ci95"], expected["ci95"], rtol=0, atol=3e-7, err_msg=f"{label} {name} ci")


def verify_rollout(seed, record):
    if record["states"].shape != (64, 64, 51, 8) or record["physical_states"].shape != (64, 64, 51, 2):
        raise AssertionError("oracle rollout shape changed")
    before_physical = record["physical_states"][:, :, :-1]
    moved, clipped, blocked_updates = native_alive_motion(before_physical, record["action"], record["motion_noise"])
    expected_physical = np.where(record["failed_before"][..., None], before_physical, moved)
    np.testing.assert_array_equal(record["position_proposal_physical_xy"], expected_physical)
    np.testing.assert_array_equal(record["clipped_noisy_action"], clipped)
    np.testing.assert_array_equal(record["blocked_updates"], blocked_updates)
    np.testing.assert_array_equal(record["physical_states"][:, :, 1:], expected_physical)
    np.testing.assert_array_equal(record["states"][:, :, 1:, :2], expected_physical.astype(np.float32))
    np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
    if not native_legal(record["states"][..., :2]).all():
        raise AssertionError("illegal emitted state")
    np.testing.assert_array_equal(record["hazard_landing"], hazardous(record["position_proposal_xy"]))
    np.testing.assert_array_equal(record["hazard_landing_physical"], hazardous(expected_physical))
    np.testing.assert_array_equal(record["onset"], (~record["failed_before"]) & (record["onset_uniform"] < record["failure_probability"]))
    np.testing.assert_array_equal(record["failed_after"], record["failed_before"] | record["onset"])
    np.testing.assert_array_equal(record["failed_before"][:, :, 1:], record["failed_after"][:, :, :-1])
    if np.any(record["failure_probability"][~record["hazard_landing"]] != 0):
        raise AssertionError("outside-support probability")
    frozen = record["failed_before"]
    np.testing.assert_array_equal(record["physical_states"][:, :, 1:][frozen], before_physical[frozen])
    goal = np.array([8.5, 3.5], np.float64)
    expected_reward = ((np.linalg.norm(expected_physical - goal, axis=-1) < 2) & (~record["failed_after"])).astype(np.float32)
    np.testing.assert_array_equal(record["reward"], expected_reward)
    discount = np.power(np.float32(CONFIG["discount"]), np.arange(CONFIG["horizon"], dtype=np.float32))
    expected_return = np.sum(record["reward"] * discount, axis=-1, dtype=np.float32)
    np.testing.assert_array_equal(record["return"], expected_return)
    metadata = load_json(OUT / f"oracle_rollout_s{seed}_metadata.json")
    if metadata["output_sha256"] != sha256(OUT / f"oracle_rollout_s{seed}.npz"):
        raise AssertionError("oracle array hash mismatch")
    emitted_xy = record["position_proposal_xy"].astype(np.float64)
    emitted_hazard = hazardous(emitted_xy)
    physical_hazard = hazardous(expected_physical)
    emitted_reward = ((np.linalg.norm(emitted_xy - goal, axis=-1) < 2) & (~record["failed_after"]))
    onset_displacement = np.linalg.norm(expected_physical - before_physical, axis=-1)[record["onset"]]
    stationary_threshold = 1e-7
    if onset_displacement.size == 0 or np.any(onset_displacement <= stationary_threshold):
        raise AssertionError("fatal incoming motion is missing or stationary")
    return {
        "paths": int(record["reward"].shape[0] * record["reward"].shape[1]),
        "first_hazard_paths": int((emitted_hazard & (~record["failed_before"])).any(axis=-1).sum()),
        "failure_paths": int(record["failed_after"].any(axis=-1).sum()),
        "reward_paths": int(record["reward"].any(axis=-1).sum()),
        "survival_without_reward_paths": int(((~record["failed_after"].any(axis=-1)) & (~record["reward"].any(axis=-1))).sum()),
        "fatal_onsets": int(record["onset"].sum()),
        "fatal_incoming_displacement_mean": float(onset_displacement.mean()),
        "fatal_incoming_displacement_minimum": float(onset_displacement.min()),
        "fatal_incoming_stationary_threshold": stationary_threshold,
        "fatal_incoming_stationary_count": int((onset_displacement <= stationary_threshold).sum()),
        "physical_emitted_max_abs_xy_difference": float(np.max(np.abs(expected_physical - emitted_xy))),
        "physical_emitted_hazard_disagreements": int(np.count_nonzero(physical_hazard != emitted_hazard)),
        "physical_emitted_reward_disagreements": int(np.count_nonzero(expected_reward.astype(bool) != emitted_reward)),
    }


def first_hazard(record):
    event = hazardous(record["position_proposal_xy"]) & (~record["failed_before"])
    entered = event.any(axis=-1)
    first = np.argmax(event, axis=-1)
    index = np.broadcast_to(first[..., None, None], record["position_proposal_xy"].shape[:-2] + (1, 2))
    xy = np.take_along_axis(record["position_proposal_xy"], index, axis=-2)[..., 0, :]
    return entered, first, xy


def main():
    completion = load_json(OUT / "completion.json")
    provenance = load_json(OUT / "provenance.json")
    for path, expected in provenance["dependency_sha256"].items():
        if sha256(path) != expected:
            raise AssertionError(f"dependency changed: {path}")
    for path, expected in provenance["diagnostic_source_sha256"].items():
        if sha256(path) != expected:
            raise AssertionError(f"sealed diagnostic source changed: {path}")
    for name, expected in {
        "movement_equivalence.json": completion["movement_equivalence_sha256"],
        "metrics.json": completion["metrics_sha256"],
        "contract_checks.json": completion["contract_checks_sha256"],
        "oracle_rollout_s0.npz": completion["oracle_rollout_sha256"]["s0"],
        "oracle_rollout_s1.npz": completion["oracle_rollout_sha256"]["s1"],
    }.items():
        if sha256(OUT / name) != expected:
            raise AssertionError(f"completed output changed: {name}")
    equivalence = load_json(OUT / "movement_equivalence.json")
    if equivalence["status"] != "passed" or equivalence["native_step_calls"] != 34:
        raise AssertionError("movement equivalence result changed")
    oracle = {seed: load_npz(OUT / f"oracle_rollout_s{seed}.npz") for seed in (0, 1)}
    rollout_diagnostics = {}
    for seed, record in oracle.items():
        rollout_diagnostics[f"s{seed}"] = verify_rollout(seed, record)
    for left, right in zip(first_hazard(oracle[0]), first_hazard(oracle[1])):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(oracle[0]["motion_noise"], oracle[1]["motion_noise"])
    np.testing.assert_array_equal(oracle[0]["onset_uniform"], oracle[1]["onset_uniform"])

    weights = load_npz(REPAIR / "bootstrap_weights.npz")["weights"]
    native_terms = trajectory_terms(load_npz(REPAIR / "final_paired.npz"), native=True)
    saved_metrics = load_json(OUT / "metrics.json")
    assert_tree_close(summary(native_terms, weights), saved_metrics["native"], "native")
    for seed in (0, 1):
        learned_terms = trajectory_terms(load_npz(REPAIR / f"rollout_s{seed}_prepaired_hrepaired.npz"))
        oracle_terms = trajectory_terms(oracle[seed])
        branch = saved_metrics["seeds"][f"s{seed}"]
        assert_tree_close(summary(learned_terms, weights), branch["saved_repaired_position_repaired_head"], f"s{seed} learned")
        assert_tree_close(summary(oracle_terms, weights), branch["oracle_motion_repaired_head"], f"s{seed} oracle")
        assert_tree_close(comparison(oracle_terms, learned_terms, weights), branch["oracle_minus_saved_learned"], f"s{seed} oracle-learned")
        assert_tree_close(comparison(oracle_terms, native_terms, weights), branch["oracle_minus_native"], f"s{seed} oracle-native")
        assert_tree_close(absolute_error(learned_terms, native_terms, weights), branch["saved_learned_absolute_native_error"], f"s{seed} learned abs")
        assert_tree_close(absolute_error(oracle_terms, native_terms, weights), branch["oracle_absolute_native_error"], f"s{seed} oracle abs")
        assert_tree_close(absolute_error_change(oracle_terms, learned_terms, native_terms, weights), branch["oracle_minus_learned_absolute_error"], f"s{seed} abs change")

    ledger = load_json(OUT / "ledger.json")
    if ledger["native_verification_steps"]["total"] != 34 or ledger["oracle_motion_slots"]["total"] != 409600 or ledger["complete_model_paths"]["total"] != 8192:
        raise AssertionError("ledger count changed")
    if any(ledger[name]["total"] for name in ("training_updates", "new_training_episodes", "new_complete_native_evaluation_episodes")):
        raise AssertionError("forbidden training or episode collection recorded")
    result = {
        "status": "passed",
        "saved_arrays_only": True,
        "new_native_steps": 0,
        "new_model_paths": 0,
        "new_training_updates": 0,
        "derived_rollout_diagnostics": rollout_diagnostics,
        "checks": {
            "sealed_dependency_and_output_hashes": True,
            "native_movement_equivalence_result": True,
            "all_oracle_slots_recomputed_from_saved_physical_state_action_and_noise": True,
            "float64_physical_and_float32_f4_history": True,
            "support_failure_absorption_reward_and_return": True,
            "pathwise_first_hazard_identity_across_heads": True,
            "all_metric_points_intervals_differences_and_absolute_errors": True,
            "sealed_call_accounting": True,
        },
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
