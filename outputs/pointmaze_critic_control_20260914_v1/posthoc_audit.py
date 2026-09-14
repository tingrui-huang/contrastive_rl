"""Saved-checkpoint parameter and fixed-evaluation audit; no training or rollout."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN = load_module("posthoc_critic_control", OUT / "run.py")


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def tree_l2(tree):
    return float(np.sqrt(sum(float(np.sum(np.asarray(leaf, np.float64) ** 2)) for leaf in jax.tree.leaves(tree))))


def tree_delta_l2(left, right):
    delta = jax.tree.map(lambda a, b: np.asarray(a) - np.asarray(b), left, right)
    return tree_l2(delta)


def main():
    _, _, _, initial, _, _, _, _ = RUN.frozen_components()
    arrays = load_npz(OUT / "training_curves_and_evaluations.npz")
    result = {"saved_arrays_and_checkpoints_only": True, "new_rollouts": 0, "training_updates": 0}
    final_params = {}
    for condition in ("A_nce_only", "B_nce_plus_rank"):
        _, state = RUN.checkpoint.load_checkpoint(OUT / "checkpoints" / f"{condition}.pkl")
        final_params[condition] = state.q_params
        result[condition] = {
            "critic_parameter_l2_from_initial": tree_delta_l2(state.q_params, initial.q_params),
            "critic_parameter_relative_l2_from_initial": tree_delta_l2(state.q_params, initial.q_params)
            / tree_l2(initial.q_params),
            "initial_validation_signed_gap_mean": float(np.mean(
                arrays[f"{condition}__validation_gap"][0]
                * arrays[f"{condition}__validation_label"][0]
            )),
            "initial_validation_accuracy": float(np.mean(
                arrays[f"{condition}__validation_gap"][0]
                * arrays[f"{condition}__validation_label"][0] > 0
            )),
            "final_validation_signed_gap_mean": float(np.mean(
                arrays[f"{condition}__validation_gap"][-1]
                * arrays[f"{condition}__validation_label"][-1]
            )),
            "final_validation_accuracy": float(np.mean(
                arrays[f"{condition}__validation_gap"][-1]
                * arrays[f"{condition}__validation_label"][-1] > 0
            )),
            "initial_reference_signed_gap_mean": float(np.mean(
                arrays[f"{condition}__reference_gap"][0]
                * arrays[f"{condition}__reference_label"][0]
            )),
            "initial_reference_accuracy": float(np.mean(
                arrays[f"{condition}__reference_gap"][0]
                * arrays[f"{condition}__reference_label"][0] > 0
            )),
            "final_reference_signed_gap_mean": float(np.mean(
                arrays[f"{condition}__reference_gap"][-1]
                * arrays[f"{condition}__reference_label"][-1]
            )),
            "final_reference_accuracy": float(np.mean(
                arrays[f"{condition}__reference_gap"][-1]
                * arrays[f"{condition}__reference_label"][-1] > 0
            )),
        }
    result["A_B_final_critic_parameter_l2"] = tree_delta_l2(
        final_params["A_nce_only"], final_params["B_nce_plus_rank"]
    )
    (OUT / "posthoc_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
