"""Final seal, pairing, critic-score, and artifact verification addendum."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    sealed = load_json(OUT / "preregistration.json")
    for path, expected in sealed["dependency_sha256"].items():
        assert sha256(path) == expected, path
    for path, expected in sealed["diagnostic_source_sha256"].items():
        assert sha256(path) == expected, path
    for name, expected in sealed["sealed_generated_sha256"].items():
        assert sha256(OUT / name) == expected, name

    roots = load_npz(OUT / "root_selection.npz")
    native = load_npz(OUT / "primary_native_traces.npz")
    model = load_npz(OUT / "primary_model_traces.npz")
    candidate = roots["candidate_action"]
    for traces in (native, model):
        for action_index in range(2):
            first_action = traces["action"][:, action_index, :, 0]
            expected = np.broadcast_to(candidate[action_index], first_action.shape)
            np.testing.assert_array_equal(first_action, expected)
    np.testing.assert_array_equal(native["actor_epsilon"][:, 0], native["actor_epsilon"][:, 1])
    np.testing.assert_array_equal(model["actor_epsilon"][:, 0], model["actor_epsilon"][:, 1])
    np.testing.assert_array_equal(native["actor_epsilon"], model["actor_epsilon"])
    np.testing.assert_array_equal(native["motion_noise"][:, 0], native["motion_noise"][:, 1])
    np.testing.assert_array_equal(model["motion_noise"][:, 0], model["motion_noise"][:, 1])
    np.testing.assert_array_equal(native["motion_noise"], model["motion_noise"])
    np.testing.assert_array_equal(native["swamp_bits_after"][:, 0], native["swamp_bits_after"][:, 1])
    np.testing.assert_array_equal(model["onset_uniform"][:, 0], model["onset_uniform"][:, 1])
    expected_root_bits = np.broadcast_to(
        roots["root_bits"][:, None, :], native["swamp_bits_before"][:, 0, :, 0].shape
    )
    np.testing.assert_array_equal(native["swamp_bits_before"][:, 0, :, 0], expected_root_bits)
    np.testing.assert_array_equal(native["swamp_bits_before"][:, 1, :, 0], expected_root_bits)

    spec = importlib.util.spec_from_file_location("verify_sealed_run", OUT / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.verify_seal()
    _, network, _, state, _, _, _, _ = module.frozen_components()
    recomputed = module.critic_scores(network, state.q_params, roots["state"])
    np.testing.assert_array_equal(recomputed, load_npz(OUT / "critic_scores.npz")["score"])

    decision = load_json(OUT / "decision.json")
    assert decision["selected_branch"] == "C"
    assert not (OUT / "followup_native_traces.npz").exists()
    assert not (OUT / "actor_probe.json").exists()
    branch = load_json(OUT / "branch_c_results.json")
    assert branch["new_paths"] == 0 and branch["training_updates"] == 0
    assert sha256(OUT / "branch_c_audit.py") == branch["source_sha256"]["script"]
    assert load_json(OUT / "verification.json")["status"] == "passed"

    completion = load_json(OUT / "completion.json")
    for name, expected in completion["artifact_sha256"].items():
        assert sha256(OUT / name) == expected, name
    ledger = load_json(OUT / "ledger.json")
    assert ledger["primary_native_paths"]["total"] == 2048
    assert ledger["primary_model_paths"]["total"] == 2048
    assert ledger["primary_native_transitions"]["total"] == 100352
    assert ledger["primary_model_transitions"]["total"] == 100352
    assert ledger["followup_native_paths"]["total"] == 0
    assert ledger["actor_probe_updates"]["total"] == 0
    assert ledger["training_updates"]["total"] == 0

    result = {
        "status": "passed",
        "sealed_dependencies_sources_and_generated_inputs": True,
        "candidate_actions_exact": True,
        "actor_innovations_paired_across_actions_and_backends": True,
        "motion_noise_paired_across_actions_and_backends": True,
        "native_bits_paired_only_across_native_actions": True,
        "model_onset_uniforms_paired_only_across_model_actions": True,
        "naturally_reached_root_bits_preserved_for_forced_step": True,
        "critic_scores_recomputed_exactly": True,
        "branch_C_only": True,
        "artifact_hashes": True,
        "training_updates": 0,
        "new_paths_in_conditional_audit": 0,
    }
    (OUT / "verification_addendum.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    completion["status"] = "complete"
    completion["independent_saved_verification"] = "passed"
    completion["seal_and_pairing_verification"] = "passed"
    completion["verification_addendum_sha256"] = sha256(OUT / "verification_addendum.json")
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
