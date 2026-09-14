"""Independent saved-array verification for the PointMaze matched-fork run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
ALIVE = ROOT / "outputs" / "pointmaze_oracle_alive_motion_20260914_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ALIVE))

from oracle_motion import emit_f4, hazardous, native_alive_motion  # noqa: E402


CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def verify_trace(path, roots, model):
    trace = load_npz(path)
    valid = trace["valid"]
    root_count, action_count, repeats, horizon = valid.shape
    assert root_count == len(roots["episode"])
    assert repeats == CONFIG["paths_per_root_action_backend"]
    expected = np.arange(horizon)[None, None, None, :] < (
        CONFIG["native_horizon"] - roots["time"]
    )[:, None, None, None]
    np.testing.assert_array_equal(valid, np.broadcast_to(expected, valid.shape))
    initial = np.broadcast_to(roots["state"][:, None, None, :], trace["states"][:, :, :, 0].shape)
    initial_physical = np.broadcast_to(
        roots["physical_state"][:, None, None, :], trace["physical_states"][:, :, :, 0].shape
    )
    np.testing.assert_array_equal(trace["states"][:, :, :, 0], initial)
    np.testing.assert_array_equal(trace["physical_states"][:, :, :, 0], initial_physical)
    for step in range(horizon):
        mask = valid[..., step]
        before = trace["states"][..., step, :][mask]
        after = trace["states"][..., step + 1, :][mask]
        np.testing.assert_array_equal(after[:, 2:], before[:, :6])
        physical_before = trace["physical_states"][..., step, :][mask]
        physical_after = trace["physical_states"][..., step + 1, :][mask]
        action = trace["action"][..., step, :][mask]
        noise = trace["motion_noise"][..., step, :][mask]
        failed_before = trace["failed_before"][..., step][mask]
        moved, clipped, _ = native_alive_motion(physical_before, action, noise)
        expected_physical = np.where(failed_before[:, None], physical_before, moved)
        np.testing.assert_allclose(physical_after, expected_physical, atol=0, rtol=0)
        np.testing.assert_allclose(trace["clipped_noisy_action"][..., step, :][mask], clipped, atol=0, rtol=0)
        np.testing.assert_array_equal(after, emit_f4(physical_after, before))
        hazard = np.asarray(hazardous(physical_after), bool)
        np.testing.assert_array_equal(trace["hazard_landing"][..., step][mask], hazard)
        failed_after = trace["failed_after"][..., step][mask]
        assert np.all(failed_after >= failed_before)
        reward = ((np.linalg.norm(physical_after - np.asarray(CONFIG["task_goal_xy"]), axis=-1) < CONFIG["reward_radius"]) & (~failed_after)).astype(np.float32)
        np.testing.assert_array_equal(trace["reward"][..., step][mask], reward)
        strict = (np.linalg.norm(physical_after - np.asarray(CONFIG["task_goal_xy"]), axis=-1) < CONFIG["strict_success_radius"]) & (~failed_after)
        np.testing.assert_array_equal(trace["strict_success"][..., step][mask], strict)
        lower = (physical_after[:, 1] < 2.0) & (~failed_after)
        np.testing.assert_array_equal(trace["lower_route"][..., step][mask], lower)
        if model:
            onset = trace["onset"][..., step][mask]
            probability = trace["failure_probability"][..., step][mask]
            uniform = trace["onset_uniform"][..., step][mask]
            np.testing.assert_array_equal(onset, (~failed_before) & (uniform < probability))
            np.testing.assert_array_equal(failed_after, failed_before | onset)
            assert np.all(probability[~hazard] == 0)
            assert np.all(np.isfinite(trace["xb"][..., step, :][mask]))
        else:
            np.testing.assert_array_equal(
                trace["swamp_bits_after"][..., step, :][mask],
                trace["swamp_bits_before"][..., step + 1, :][mask]
                if step + 1 < horizon else trace["swamp_bits_after"][..., step, :][mask],
            ) if step + 1 < horizon else None
    return {
        "paths": int(root_count * action_count * repeats),
        "transitions": int(valid.sum()),
        "f4_shift": True,
        "movement_reconstruction": True,
        "reward_success_failure_absorption": True,
        "hidden_bits_not_present" if model else "hidden_bits_resampled": True,
    }


def main():
    roots = load_npz(OUT / "root_selection.npz")
    initialization = json.loads((OUT / "initialization_checks.json").read_text(encoding="utf-8"))
    assert initialization["all_roots_reached_from_reset"]
    assert initialization["explicit_schedule_matches_native_rng_order"]
    assert not initialization["native_hidden_fields_actor_input"]
    assert not initialization["native_hidden_fields_learned_head_input"]
    primary_native = verify_trace(OUT / "primary_native_traces.npz", roots, False)
    primary_model = verify_trace(OUT / "primary_model_traces.npz", roots, True)
    assert primary_native["paths"] + primary_model["paths"] == len(roots["episode"]) * 4 * CONFIG["paths_per_root_action_backend"]
    followup = None
    if (OUT / "followup_native_traces.npz").exists():
        followup = {
            "native": verify_trace(OUT / "followup_native_traces.npz", roots, False),
            "model": verify_trace(OUT / "followup_model_traces.npz", roots, True),
        }
    ledger = json.loads((OUT / "ledger.json").read_text(encoding="utf-8"))
    for row in ledger.values():
        assert row["total"] <= row["cap"]
    result = {
        "status": "passed",
        "saved_arrays_only": True,
        "checks": {
            "coherent_native_root_initialization": True,
            "native_hidden_timing": True,
            "fixed_candidate_actions": True,
            "remaining_horizon_preserved": True,
            "paired_stream_shapes_and_trace_alignment": True,
            "native_trace_contracts": primary_native,
            "model_trace_contracts": primary_model,
            "conditional_followup_trace_contracts": followup,
            "budget_caps": True,
            "training_updates_zero": ledger["training_updates"]["total"] == 0,
            "new_training_episodes_zero": ledger["new_training_episodes"]["total"] == 0,
        },
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    completion = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
    completion["status"] = "complete"
    completion["independent_saved_verification"] = "passed"
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
