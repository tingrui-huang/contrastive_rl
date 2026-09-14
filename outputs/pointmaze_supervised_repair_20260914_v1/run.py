"""Orchestrate the sealed PointMaze supervised repair experiment."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from common import (
    CONFIG,
    MATCHED,
    OLD_TEACHER,
    OUT,
    PERSISTENT,
    RESPONSE,
    REVIEW,
    ROOT,
    SUPERVISED,
    Ledger,
    load_npz,
    sha256,
    write_json,
)
from evaluate import evaluate_all
from train_repairs import train_all


EXPECTED = {
    MATCHED / "REPORT.md": "a949e5ccd73042a4dc0973f0b47ead068b0a185a91963d62046b2b81b0cf7174",
    MATCHED / "matched_metrics.json": "87f3c2a7ba802a04bab7a1281791b7cd0148972c7120c7aa0b0173b22a318cc1",
    MATCHED / "provenance.json": "d3923d0169bc8203cde3031a3c2da128cdfc876cd0c9c46c69960200bd6b8b61",
    MATCHED / "verification.json": "4b84cb05e16752515e04ae1742ba6e418b6b21358008f3ae8194e94e7fdf313f",
    PERSISTENT / "head_s0.npz": "06b6a5a7815f5720c9adba7c0eae5d53239c5366b73c731bf5ad98fdfdbf1579",
    PERSISTENT / "head_s1.npz": "b9a13cc6fc06f52874e105a2eba5319d150312a2fd9afb87c57770e428abe661",
    PERSISTENT / "head_feature_scaling.npz": "63b8d13acb3bf5f749866900121fc3cb43657848c0722760b2499538752cfc42",
    PERSISTENT / "run.py": "2b8334e00212571f763d8a8f981285dbdc4e1615225c2aa22e06c69304903e74",
    PERSISTENT / "REPORT.md": "d8d7548936db7a617b3c331d528da796cde292bfb2b5c82a7e1b48a0a051d238",
    RESPONSE / "candidate_s0.npz": "c31e8383edae3ca20fd54d3966d480f6c8687dc89c9db1f7456016a476cb0183",
    RESPONSE / "candidate_s1.npz": "7169a710fe0adb0999a790c9757d4f9f4d125f567aa57e06f09bd3dcbbae243d",
    RESPONSE / "gate_feature_scaling.npz": "24391dfc715ad892148e79a130e2186fba3cae311eb5e6cd3509f3e9633c313d",
    RESPONSE / "run.py": "1b13fa14504906493e792eb8af1b5a6c0cdb00c717c6c182fc1d71cc5afaf1e9",
    RESPONSE / "REPORT.md": "72361afcbd0492b0e01fa76abcb16e5f792edac6f42183e73981eb5d118b7dfe",
    SUPERVISED / "REPORT.md": "1c4abd1d0924d34a2104c039c8250fbef62bf0b45a9b0b503a3c3b20e5381e27",
    SUPERVISED / "run.py": "6ef88e2338cd930d65d232d16183095ef8f6a8d5f33c564c5c94184e029da8ca",
    ROOT / "crl" / "envs.py": "4ceb7d2cbd5f295fd97363f07badcb11a2e8bdce5efe98ab50d59c68f6c484c2",
    ROOT / "ett" / "convex_action_transition.py": "a92a11f90bb186daa9578b2ccd4d622676bb60b0824b48022eb43ac64fa91482",
    ROOT / "ett" / "diagonal_transition.py": "53f11305c53186700ad4981a3be1b1fa58a2c77d1ebf29130b2f24cfa6058d94",
    ROOT / "ett" / "pointmaze_region_pilot.py": "bee37ba6f8d30a8ded29609c54899ba7402e399c83745f3380befc9141473faf",
    ROOT / "ett" / "rollout_return.py": "1bcf669b449373b907e257088e507638e2e691b086ff10b89ccec00583cf05dc",
    OLD_TEACHER: "474d007907a33ba5a1633c5bb64bf2f4c154cff1a4f94a08839f7364a248ae44",
    REVIEW / "REVIEW.md": "22c6a10e5efcf9b67721399b9f937ed8500c75648b69886ac26e12260f729f63",
    ROOT / "artifacts" / "nominal_policy" / "f4_p30_expert_only_mdn_k5_s0" / "best.pkl": "6376e60185aa616c2d09c5e0250d762cde2fb845c5042488e64910fb3b745f23",
    ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "runs" / "f4_p30_sweep" / "p30_a0_a01_a03_s0_s1" / "alpha0_seed0" / "final.pkl": "ea8a71d3cb8d963259a54d47250b454462dae2ae62f03a55c370ca81a2c8ec54",
    ROOT / "artifacts" / "ett_distribution_matching" / "f4_p30_s01_guarded" / "s0_L0p25_lambda0" / "final.pkl": "9e9e1c56387d446b5788612e27e5b7f7743e120186bebc90a78fc6774476184e",
}


def preflight():
    path = OUT / "provenance.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for name, digest in saved["dependency_sha256"].items():
            if sha256(name) != digest:
                raise RuntimeError(f"dependency changed after seal: {name}")
        return saved
    for name, digest in EXPECTED.items():
        if not name.is_file() or sha256(name) != digest:
            raise RuntimeError(f"pinned dependency mismatch: {name}")
    matched_verification = json.loads((MATCHED / "verification.json").read_text(encoding="utf-8"))
    assert matched_verification["status"] == "pass"
    search = json.loads((OUT / "equivalent_experiment_search.json").read_text(encoding="utf-8"))
    assert not search["equivalent_completed_repair_found"]
    local_sources = [
        OUT / "PROTOCOL.md",
        OUT / "config.json",
        OUT / "equivalent_experiment_search.json",
        OUT / "common.py",
        OUT / "collect_split.py",
        OUT / "train_repairs.py",
        OUT / "evaluate.py",
        OUT / "run.py",
    ]
    for name in local_sources:
        if not name.is_file():
            raise FileNotFoundError(name)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dependencies = list(EXPECTED) + local_sources
    result = {
        "source_commit": CONFIG["source_commit"],
        "working_head_at_execution": head,
        "head_equality_not_required": True,
        "dependency_sha256": {str(name): sha256(name) for name in dependencies},
        "initialization_sha256": {
            "position_s0": EXPECTED[RESPONSE / "candidate_s0.npz"],
            "position_s1": EXPECTED[RESPONSE / "candidate_s1.npz"],
            "head_s0": EXPECTED[PERSISTENT / "head_s0.npz"],
            "head_s1": EXPECTED[PERSISTENT / "head_s1.npz"],
            "actor": EXPECTED[ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "runs" / "f4_p30_sweep" / "p30_a0_a01_a03_s0_s1" / "alpha0_seed0" / "final.pkl"],
            "nominal": EXPECTED[ROOT / "artifacts" / "nominal_policy" / "f4_p30_expert_only_mdn_k5_s0" / "best.pkl"],
            "base_transition": EXPECTED[ROOT / "artifacts" / "ett_distribution_matching" / "f4_p30_s01_guarded" / "s0_L0p25_lambda0" / "final.pkl"],
        },
        "known_hazard_support_is_additional_environment_information": True,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    write_json(path, result)
    return result


def collect(split, ledger):
    output = OUT / f"{split}_paired.npz"
    manifest = OUT / f"{split}_collection.json"
    if output.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if sha256(output) != saved["output_sha256"]:
            raise RuntimeError(f"{split} collection hash changed")
        return load_npz(output)
    if output.exists() or manifest.exists():
        raise RuntimeError(f"partial {split} collection requires investigation")
    planned = CONFIG["planned_native_steps"][split]
    ledger.native(planned, f"collect_{split}")
    subprocess.run([sys.executable, str(OUT / "collect_split.py"), "--split", split], cwd=ROOT, check=True)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["native_steps"] == planned and sha256(output) == saved["output_sha256"]
    return load_npz(output)


def freeze_manifest():
    path = OUT / "checkpoints_frozen.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["checkpoint_sha256"] = {
        f"position_s{seed}": sha256(OUT / f"position_repaired_s{seed}.npz")
        for seed in (0, 1)
    }
    saved["checkpoint_sha256"].update({
        f"head_s{seed}": sha256(OUT / f"head_repaired_s{seed}.npz")
        for seed in (0, 1)
    })
    saved["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(path, saved)
    return saved


def contract_checks():
    result = {
        "arms_checked": 0,
        "failure_irreversible": True,
        "outside_hazard_probability_zero": True,
        "fatal_incoming_position_retained": True,
        "post_failure_xy_frozen": True,
        "f4_shift_exact": True,
        "failed_reward_zero": True,
        "actor_and_nominal_not_given_failure_mode": True,
    }
    for seed in (0, 1):
        for position_variant in ("old", "repaired"):
            for head_variant in ("old", "repaired"):
                record = load_npz(OUT / f"rollout_s{seed}_p{position_variant}_h{head_variant}.npz")
                assert np.all(record["failed_after"] >= record["failed_before"])
                assert not record["failure_probability"][~record["hazard_landing"]].any()
                current = record["states"][:, :, :-1]
                following = record["states"][:, :, 1:]
                np.testing.assert_array_equal(following[..., 2:], current[..., :6])
                np.testing.assert_array_equal(following[..., :2][record["failed_before"]], current[..., :2][record["failed_before"]])
                np.testing.assert_array_equal(following[..., :2][record["onset"]], record["position_proposal_xy"][record["onset"]])
                assert not record["reward"][record["failed_after"]].any()
                result["arms_checked"] += 1
    write_json(OUT / "contract_checks.json", result)
    return result


def main():
    started = time.monotonic()
    if (OUT / "completion.json").exists():
        raise RuntimeError("experiment already complete; reuse saved outputs")
    provenance = preflight()
    ledger = Ledger()
    if not (OUT / "execution_started.json").exists():
        write_json(OUT / "execution_started.json", {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
            "config_sha256": sha256(OUT / "config.json"),
            "provenance_sha256": sha256(OUT / "provenance.json"),
        })
    collect("train", ledger)
    collect("validation", ledger)
    train_all(ledger)
    frozen = freeze_manifest()
    assert not frozen["final_outcomes_inspected"]
    collect("final", ledger)
    evaluate_all(ledger)
    checks = contract_checks()
    ledger.save()
    assert ledger.total(ledger.native_counts) == CONFIG["planned_native_steps"]["total"]
    assert ledger.total(ledger.update_counts) == CONFIG["training_update_cap"]
    assert ledger.total(ledger.path_counts) == CONFIG["complete_model_path_cap"]
    if ledger.total(ledger.position_counts) > CONFIG["position_successor_cap"]:
        raise RuntimeError("position budget exceeded")
    completion = {
        "status": "complete",
        "elapsed_seconds": time.monotonic() - started,
        "source_commit": CONFIG["source_commit"],
        "provenance_sha256": sha256(OUT / "provenance.json"),
        "frozen_checkpoint_manifest_sha256": sha256(OUT / "checkpoints_frozen.json"),
        "final_scope_sha256": sha256(OUT / "final_scope.json"),
        "acceptance_sha256": sha256(OUT / "acceptance.json"),
        "trajectory_metrics_sha256": sha256(OUT / "trajectory_metrics.json"),
        "contract_checks_sha256": sha256(OUT / "contract_checks.json"),
        "native_steps": ledger.total(ledger.native_counts),
        "position_successors": ledger.total(ledger.position_counts),
        "head_rows": ledger.total(ledger.head_counts),
        "training_updates": ledger.total(ledger.update_counts),
        "complete_model_paths": ledger.total(ledger.path_counts),
        "all_eight_arms_share_hazard_support_and_irreversible_semantics": checks["arms_checked"] == 8,
        "historical_artifacts_modified": False,
    }
    write_json(OUT / "completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
