"""Independent reconstruction checks using only completed pilot arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run as pilot


OUT = Path(__file__).resolve().parent


def close_tree(calculated, saved, label):
    for name, row in calculated.items():
        expected = saved[name]
        for key in ("estimate", "difference"):
            if key in row:
                np.testing.assert_allclose(row[key], expected[key], rtol=0, atol=3e-7, err_msg=f"{label}/{name}/{key}")
        np.testing.assert_allclose(row["ci95"], expected["ci95"], rtol=0, atol=3e-7, err_msg=f"{label}/{name}/ci")


def verify_generated(record):
    before = record["physical_states"][:, :-1]
    moved, clipped, blocked = pilot.native_alive_motion(before, record["action"], record["motion_noise"])
    expected = np.where(record["failed_before"][..., None], before, moved)
    np.testing.assert_array_equal(record["physical_xy"], expected)
    np.testing.assert_array_equal(record["physical_states"][:, 1:], expected)
    np.testing.assert_array_equal(record["clipped_noisy_action"], clipped)
    np.testing.assert_array_equal(record["blocked_updates"], blocked)
    np.testing.assert_array_equal(record["states"][:, 1:, :2], expected.astype(np.float32))
    np.testing.assert_array_equal(record["states"][:, 1:, 2:], record["states"][:, :-1, :6])
    np.testing.assert_array_equal(record["failure_states"][:, :-1], record["failed_before"])
    np.testing.assert_array_equal(record["failure_states"][:, 1:], record["failed_after"])
    np.testing.assert_array_equal(record["failed_after"], record["failed_before"] | record["onset"])
    np.testing.assert_array_equal(record["hazard_landing"], pilot.hazardous(record["states"][:, 1:, :2]))
    np.testing.assert_array_equal(
        record["onset"], (~record["failed_before"]) & (record["onset_uniform"] < record["failure_probability"])
    )
    if np.any(record["failure_probability"][~record["hazard_landing"]] != 0):
        raise AssertionError("onset support violated")
    goal = np.asarray(pilot.GOAL[:2], np.float64)
    expected_reward = ((np.linalg.norm(expected - goal, axis=-1) < 2) & (~record["failed_after"])).astype(np.float32)
    np.testing.assert_array_equal(record["reward"], expected_reward)
    if np.any(record["reward"][record["failed_after"]] != 0):
        raise AssertionError("failed generated reward is nonzero")


def main():
    pilot.verify_sealed()
    completion = pilot.load_json(OUT / "completion.json")
    training = pilot.load_json(OUT / "training.json")
    results = pilot.load_json(OUT / "results.json")
    if completion["training_sha256"] != pilot.sha256(OUT / "training.json"):
        raise AssertionError("training hash changed")
    if completion["results_sha256"] != pilot.sha256(OUT / "results.json"):
        raise AssertionError("results hash changed")
    for label, row in training["runs"].items():
        checkpoint_path = OUT / "checkpoints" / f"{label}_final.pkl"
        if pilot.sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint changed: {label}")
        audit = pilot.load_npz(OUT / f"{label}_batch_audit.npz")
        if audit["permutation"].shape != (1000, 256) or audit["offline_episode"].shape != (1000, 256):
            raise AssertionError("batch audit shape changed")
        synthetic = audit["synthetic_episode"] >= 0
        if int(synthetic.sum()) != row["synthetic_rows"]:
            raise AssertionError("synthetic participation count changed")
        if np.any(audit["synthetic_future"][synthetic] <= audit["synthetic_time"][synthetic]):
            raise AssertionError("invalid future-goal sample")
        if not np.isfinite(np.asarray(row["actor_parameter_l2"])) or row["actor_parameter_l2"] <= 0:
            raise AssertionError("actor did not update")
        if not np.isfinite(np.asarray(row["critic_parameter_l2"])) or row["critic_parameter_l2"] <= 0:
            raise AssertionError("critic did not update")
    for seed in (0, 1):
        left = pilot.load_npz(OUT / f"B_s{seed}_batch_audit.npz")
        right = pilot.load_npz(OUT / f"C_s{seed}_batch_audit.npz")
        for key in ("offline_episode", "offline_time", "offline_future", "permutation"):
            np.testing.assert_array_equal(left[key], right[key])
        for update in pilot.CONFIG["refresh_updates"]:
            folder = OUT / "rounds" / f"C_s{seed}_u{update}"
            for root_time in pilot.CONFIG["root_times"]:
                verify_generated(pilot.load_npz(folder / f"h{50-root_time}.npz"))
    curves = pilot.load_npz(OUT / "training_curves.npz")
    if not curves or not all(np.isfinite(value).all() and value.shape == (1000,) for value in curves.values()):
        raise AssertionError("training curves invalid")
    weights = pilot.load_npz(OUT / "bootstrap_weights.npz")
    native_terms = {}
    for label in ("A", "B_s0", "C_s0", "B_s1", "C_s1"):
        record = pilot.load_npz(OUT / f"{label}_native.npz")
        if record["states"].shape != (200, 51, 8):
            raise AssertionError("native array shape changed")
        native_terms[label] = pilot.metric_terms(record, native=True)
        close_tree(pilot.summarize_terms(native_terms[label], weights["native"]), results["native"][label], f"native/{label}")
    model_terms = {}
    for seed in (0, 1):
        for label in ("A", f"B_s{seed}", f"C_s{seed}"):
            key = f"s{seed}/{label}"
            record = pilot.load_npz(OUT / f"model_s{seed}_{label}.npz")
            if record["states"].shape != (256, 51, 8):
                raise AssertionError("model array shape changed")
            verify_generated(record)
            model_terms[key] = pilot.metric_terms(record, native=False)
            close_tree(pilot.summarize_terms(model_terms[key], weights["model"]), results["model"][key], f"model/{key}")
    for seed in (0, 1):
        for left, right in ((f"C_s{seed}", "A"), (f"C_s{seed}", f"B_s{seed}"), (f"B_s{seed}", "A")):
            name = f"{left}_minus_{right}"
            close_tree(pilot.compare_terms(native_terms[left], native_terms[right], weights["native"]),
                       results["native_contrasts"][name], f"native contrast/{name}")
            model_name = f"s{seed}/{name}"
            close_tree(pilot.compare_terms(model_terms[f"s{seed}/{left}"], model_terms[f"s{seed}/{right}"], weights["model"]),
                       results["model_contrasts"][model_name], f"model contrast/{model_name}")
    ledger = pilot.load_json(OUT / "ledger.json")
    expected = {"native_steps": 50000, "oracle_transition_slots": 108800, "complete_model_paths": 2560,
                "actor_updates": 4000, "critic_updates": 4000}
    for category, total in expected.items():
        if ledger[category]["total"] != total or ledger[category]["total"] > ledger[category]["cap"]:
            raise AssertionError(f"ledger mismatch: {category}")
    verification = {
        "status": "passed",
        "saved_arrays_only": True,
        "new_native_steps": 0,
        "new_model_paths": 0,
        "new_training_updates": 0,
        "checks": {
            "sealed_sources_dependencies_and_initialization": True,
            "paired_B_C_offline_draws_and_permutations": True,
            "finite_curves_and_nonzero_actor_critic_updates": True,
            "generated_native_motion_reconstructed": True,
            "failure_support_fatal_motion_absorption_f4_and_reward": True,
            "valid_future_goal_sampling_without_padding": True,
            "native_metrics_and_paired_bootstrap_recomputed": True,
            "predictive_metrics_and_paired_bootstrap_recomputed": True,
            "budget_ledger_exact": True,
        },
    }
    pilot.write_json(OUT / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
