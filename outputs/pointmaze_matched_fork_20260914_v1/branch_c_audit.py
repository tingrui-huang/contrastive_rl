"""Conditional branch-C audit of exact saved C1 replay rows and critic semantics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PILOT = ROOT / "outputs" / "pointmaze_oracle_training_pilot_20260914_v1"
DATASET = ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "datasets" / "swamp_windy_f4_merged_s0.npz"
GOAL_XY = np.asarray([8.5, 3.5], np.float32)
GOAL_F4 = np.tile(GOAL_XY, 4)
LABELS = {"B_s1": "offline continuation", "C_s1": "90% offline + 10% oracle generated"}
TEACHER = {0: "random", 1: "forced_safe", 2: "immediate_shortcut", 3: "wait_shortcut", 4: "bad_demo"}


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
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def root_estimate(values, weights):
    root_values = np.asarray(values, np.float64).mean(axis=-1)
    return {
        "point": float(root_values.mean()),
        "ci95": np.quantile(weights @ root_values, [0.025, 0.975]),
        "root_count": len(root_values),
        "draws_per_root": int(np.asarray(values).shape[-1]),
    }


def synthetic_bank(label, refresh, route_all, teacher_all):
    fields = {
        "state": np.full((128, 51, 8), np.nan, np.float32),
        "action": np.full((128, 50, 2), np.nan, np.float32),
        "failed": np.zeros((128, 51), bool),
        "length": np.zeros(128, np.int64),
        "route": np.zeros(128, np.int64),
        "teacher": np.zeros(128, np.int64),
        "path_lower": np.zeros(128, bool),
        "path_strict": np.zeros(128, bool),
    }
    cursor = 0
    for root_time in (0, 10, 25, 40):
        horizon = 50 - root_time
        record = load_npz(PILOT / "rounds" / f"{label}_u{refresh}" / f"h{horizon}.npz")
        n = len(record["states"])
        sl = slice(cursor, cursor + n)
        fields["state"][sl, : horizon + 1] = record["states"]
        fields["action"][sl, :horizon] = record["action"]
        fields["failed"][sl, : horizon + 1] = record["failure_states"]
        fields["length"][sl] = horizon + 1
        original_episode = record["offline_episode_id"]
        fields["route"][sl] = route_all[original_episode]
        fields["teacher"][sl] = teacher_all[original_episode]
        fields["path_lower"][sl] = np.any(record["states"][:, :, 1] < 2, axis=1)
        fields["path_strict"][sl] = np.min(
            np.linalg.norm(record["physical_states"] - GOAL_XY, axis=-1), axis=1
        ) < 0.5
        cursor += n
    if cursor != 128:
        raise AssertionError("synthetic path order/count changed")
    return fields


def reconstruct(label, observation, action_all, train_ids, route_all, teacher_all):
    audit = load_npz(PILOT / f"{label}_batch_audit.npz")
    parts = {name: [] for name in (
        "update", "source", "anchor", "next", "action", "goal", "failed_anchor",
        "failed_goal", "route", "teacher", "path_lower", "path_strict",
    )}
    cache = {}
    for update in range(1000):
        generated_count = int(audit["synthetic_count"][update])
        offline_count = 256 - generated_count
        episode = audit["offline_episode"][update, :offline_count]
        anchor_time = audit["offline_time"][update, :offline_count]
        future_time = audit["offline_future"][update, :offline_count]
        original_episode = train_ids[episode]
        episode_states = observation[episode, :, :8]
        path_lower = np.any(episode_states[:, :, 1] < 2, axis=1)
        path_strict = np.min(np.linalg.norm(episode_states[:, :, :2] - GOAL_XY, axis=-1), axis=1) < 0.5
        parts["update"].append(np.full(offline_count, update, np.int64))
        parts["source"].append(np.zeros(offline_count, bool))
        parts["anchor"].append(observation[episode, anchor_time, :8])
        parts["next"].append(observation[episode, anchor_time + 1, :8])
        parts["action"].append(action_all[episode, anchor_time])
        parts["goal"].append(observation[episode, future_time, :8])
        parts["failed_anchor"].append(np.zeros(offline_count, bool))
        parts["failed_goal"].append(np.zeros(offline_count, bool))
        parts["route"].append(route_all[original_episode])
        parts["teacher"].append(teacher_all[original_episode])
        parts["path_lower"].append(path_lower)
        parts["path_strict"].append(path_strict)
        if generated_count:
            refresh = 250 * (update // 250)
            if refresh not in cache:
                cache[refresh] = synthetic_bank(label, refresh, route_all, teacher_all)
            bank = cache[refresh]
            episode = audit["synthetic_episode"][update, :generated_count]
            anchor_time = audit["synthetic_time"][update, :generated_count]
            future_time = audit["synthetic_future"][update, :generated_count]
            if np.any(future_time >= bank["length"][episode]):
                raise AssertionError("synthetic padding selected")
            parts["update"].append(np.full(generated_count, update, np.int64))
            parts["source"].append(np.ones(generated_count, bool))
            parts["anchor"].append(bank["state"][episode, anchor_time])
            parts["next"].append(bank["state"][episode, anchor_time + 1])
            parts["action"].append(bank["action"][episode, anchor_time])
            parts["goal"].append(bank["state"][episode, future_time])
            parts["failed_anchor"].append(bank["failed"][episode, anchor_time])
            parts["failed_goal"].append(bank["failed"][episode, future_time])
            parts["route"].append(bank["route"][episode])
            parts["teacher"].append(bank["teacher"][episode])
            parts["path_lower"].append(bank["path_lower"][episode])
            parts["path_strict"].append(bank["path_strict"][episode])
    result = {name: np.concatenate(value) for name, value in parts.items()}
    if len(result["anchor"]) != 256000:
        raise AssertionError("learner row count changed")
    return result


def fraction_by(mask, values, mapping):
    if not np.any(mask):
        return {name: None for name in mapping.values()}
    return {name: float(np.mean(values[mask] == code)) for code, name in mapping.items()}


def summarize(label, rows, selected_roots):
    xy = rows["anchor"][:, :2]
    next_xy = rows["next"][:, :2]
    goal_xy = rows["goal"][:, :2]
    broad_fork = (xy[:, 0] >= 1) & (xy[:, 0] < 2) & (xy[:, 1] >= 2) & (xy[:, 1] < 4)
    matched_fork = broad_fork & (xy[:, 1] >= 3)
    down_action = rows["action"][:, 1] < -0.25
    actual_down_exit = matched_fork & (np.floor(next_xy[:, 0]).astype(int) == 1) & (np.floor(next_xy[:, 1]).astype(int) == 2)
    future_reward = np.linalg.norm(goal_xy - GOAL_XY, axis=-1) < 2
    future_strict = np.linalg.norm(goal_xy - GOAL_XY, axis=-1) < 0.5
    goal_history = rows["goal"].reshape(-1, 4, 2)
    stationary_goal = np.all(goal_history == goal_history[:, :1], axis=(1, 2))
    exact_task = np.all(rows["goal"] == GOAL_F4, axis=1)
    relevant = matched_fork & down_action & future_reward
    actual_relevant = actual_down_exit & future_reward
    nearest = np.min(
        np.linalg.norm(rows["anchor"][:, None, :] - selected_roots[None, :, :], axis=-1), axis=1
    )
    close = nearest <= 0.1
    distance_f4 = np.linalg.norm(rows["goal"] - GOAL_F4, axis=1)
    source_rows = {}
    for source_value, source_name in ((False, "offline"), (True, "oracle_generated")):
        source = rows["source"] == source_value
        source_rows[source_name] = {
            "rows": int(source.sum()),
            "matched_fork_rows": int((source & matched_fork).sum()),
            "matched_fork_down_action_rows": int((source & matched_fork & down_action).sum()),
            "matched_fork_actual_down_exit_rows": int((source & actual_down_exit).sum()),
            "down_action_future_reward_region_positive_rows": int((source & relevant).sum()),
            "actual_down_exit_future_reward_region_positive_rows": int((source & actual_relevant).sum()),
            "down_action_future_strict_region_positive_rows": int((source & matched_fork & down_action & future_strict).sum()),
            "down_action_stationary_future_goal_rows": int((source & matched_fork & down_action & stationary_goal).sum()),
            "down_action_source_path_strict_success_rows": int((source & matched_fork & down_action & rows["path_strict"]).sum()),
            "down_action_source_path_lower_route_rows": int((source & matched_fork & down_action & rows["path_lower"]).sum()),
            "failed_before_rows": int((source & rows["failed_anchor"]).sum()),
        }
    batches_relevant = np.unique(rows["update"][relevant])
    return {
        "label": LABELS[label],
        "rows": len(rows["anchor"]),
        "source": source_rows,
        "matched_fork_definition": "1<=x<2, 3<=y<4",
        "down_action_definition": "recorded action_y < -0.25; separate actual-exit count uses observed next cell (1,2)",
        "matched_fork_rows": int(matched_fork.sum()),
        "matched_fork_down_action_rows": int((matched_fork & down_action).sum()),
        "matched_fork_actual_down_exit_rows": int(actual_down_exit.sum()),
        "down_action_future_reward_region_positive_rows": int(relevant.sum()),
        "down_action_future_reward_region_fraction_of_all_positive_rows": float(relevant.mean()),
        "actual_down_exit_future_reward_region_positive_rows": int(actual_relevant.sum()),
        "training_updates_with_any_down_action_future_reward_positive": int(len(batches_relevant)),
        "positive_cells_in_nce_matrices": 256000,
        "all_cells_in_nce_matrices": 1000 * 256 * 256,
        "each_saved_goal_is_off_diagonal_for_other_255_anchors_in_its_batch": True,
        "fixed_task_goal_exact_positive_rows": int(exact_task.sum()),
        "fixed_task_goal_exact_positive_fraction": float(exact_task.mean()),
        "future_goal_f4_distance_to_fixed_task_quantiles": np.quantile(distance_f4, [0, 0.1, 0.5, 0.9, 1]),
        "relevant_future_goal_f4_distance_to_fixed_task_quantiles": np.quantile(distance_f4[relevant], [0, 0.1, 0.5, 0.9, 1]),
        "relevant_future_goal_stationary_fraction": float(stationary_goal[relevant].mean()),
        "relevant_future_goal_strict_region_fraction": float(future_strict[relevant].mean()),
        "rows_within_f4_distance_0p1_of_one_of_16_roots": int(close.sum()),
        "close_root_down_action_future_reward_positive_rows": int((close & down_action & future_reward).sum()),
        "relevant_teacher_fraction": fraction_by(relevant, rows["teacher"], TEACHER),
    }, relevant


def main():
    decision = json.loads((OUT / "primary_decision.json").read_text(encoding="utf-8"))
    if decision["selected_branch"] != "C":
        raise RuntimeError("branch-C replay audit prohibited by sealed primary decision")
    dataset = load_npz(DATASET)
    heldout = np.random.default_rng(0).permutation(len(dataset["obs"]))[:660]
    train_ids = np.setdiff1d(np.arange(len(dataset["obs"])), heldout)
    observation = dataset["obs"][train_ids]
    action = dataset["act"][train_ids, :50]
    roots = load_npz(OUT / "root_selection.npz")["state"]
    reconstructed = {
        label: reconstruct(label, observation, action, train_ids, dataset["route_label"], dataset["teacher_mode"])
        for label in ("B_s1", "C_s1")
    }
    summaries = {}
    saved = {}
    for label, rows in reconstructed.items():
        summaries[label], relevant = summarize(label, rows, roots)
        saved[f"{label}_relevant_anchor"] = rows["anchor"][relevant]
        saved[f"{label}_relevant_action"] = rows["action"][relevant]
        saved[f"{label}_relevant_goal"] = rows["goal"][relevant]
        saved[f"{label}_relevant_source"] = rows["source"][relevant]
        saved[f"{label}_relevant_update"] = rows["update"][relevant]
    np.savez_compressed(OUT / "branch_c_relevant_rows.npz", **saved)

    # Sealed, read-only actor inference at the same roots. The pre-generated
    # Gaussian bank and noiseless native movement helper turn action samples
    # into actual adjacent-cell exits; no continuation outcome is generated.
    spec = importlib.util.spec_from_file_location("branch_c_sealed_run", OUT / "run.py")
    sealed_run = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(sealed_run)
    root_arrays = load_npz(OUT / "root_selection.npz")
    probe_streams = load_npz(OUT / "actor_probe_streams.npz")
    _, network, _, training_state, _, _, _, _ = sealed_run.frozen_components()
    actor = sealed_run.probe_root_distribution(
        network, training_state.policy_params, root_arrays, probe_streams["evaluation_epsilon"]
    )
    bootstrap = load_npz(OUT / "bootstrap_weights.npz")["weights"]
    np.savez_compressed(OUT / "branch_c_actor_root_actions.npz", **actor)

    b = summaries["B_s1"]
    c = summaries["C_s1"]
    audit = {
        "selected_branch": "C",
        "saved_arrays_only": True,
        "new_paths": 0,
        "training_updates": 0,
        "critic_objective": {
            "type": "Monte Carlo sigmoid NCE over the complete 256x256 logit matrix",
            "positive": "one exact full-F4 achieved future state from the same path per row",
            "off_diagonal": "the other 255 saved achieved goals in the batch",
            "native_reward_consumed": False,
            "model_return_consumed": False,
            "task_region_label_consumed": False,
            "failure_negative_alpha": 0.0,
        },
        "runs": summaries,
        "matched_coverage_comparison": {
            "B_s1_down_action_future_reward_positive_rows": b["down_action_future_reward_region_positive_rows"],
            "C_s1_down_action_future_reward_positive_rows": c["down_action_future_reward_region_positive_rows"],
            "difference_C_minus_B": c["down_action_future_reward_region_positive_rows"] - b["down_action_future_reward_region_positive_rows"],
            "B_s1_actual_down_exit_future_reward_positive_rows": b["actual_down_exit_future_reward_region_positive_rows"],
            "C_s1_actual_down_exit_future_reward_positive_rows": c["actual_down_exit_future_reward_region_positive_rows"],
            "fraction_of_C_positive_rows": c["down_action_future_reward_region_fraction_of_all_positive_rows"],
        },
        "frozen_actor_at_matched_roots": {
            "method": "4096 sealed Gaussian draws per root, converted to actual zero-noise next cells by the verified movement helper",
            "down_exit_probability": root_estimate(actor["down_exit"], bootstrap),
            "right_exit_probability": root_estimate(actor["right_exit"], bootstrap),
            "mean_sampled_action": actor["action"].mean(axis=(0, 1)),
            "continuation_outcomes_generated": 0,
        },
        "localization": {
            "insufficient_relevant_coverage": "supported: model augmentation did not increase the count of matched-fork downward rows paired with a future task-region achieved goal over the offline continuation",
            "different_continuation_distribution": "supported: NCE positives are single logged future states under offline teachers or changing training-time actors; the matched label averages 64 continuations of the final frozen C1 actor after a forced action",
            "full_f4_identity_vs_task_region": "real semantic mismatch: native preference uses radius-2 rewards and radius-0.5 strict success, while the critic receives exact full-F4 achieved identities and never receives the canonical stationary task F4 exactly",
            "representation_or_optimization": "unresolved: parameters changed and losses were finite, but this audit contains no capacity-controlled fit or optimization intervention",
        },
        "recommended_single_repair": "A critic-only matched task-preference intervention: on a separately preregistered coherent-alive fork training split, add one pairwise task-goal ranking term whose labels come from frozen-model multi-continuation down-versus-right returns, while keeping the actor, F4 representation, base NCE batches, BC, nominal, head and motion fixed. First require the critic preference to flip on held-out coherent roots; only then test actor response and new complete native episodes.",
        "why_not_goal_sampling_claim": "Zero exact canonical-goal positives and sparse route rows identify plausible mechanisms, not which one caused the critic ranking. The proposed one-change ranking intervention directly tests the missing task preference without simultaneously changing goal proportions or representation.",
        "unresolved": [
            "whether canonicalizing task-region goals alone would fix the ranking",
            "whether the representation can fit both exact F4 reachability and task-region preference without interference",
            "whether a corrected critic gradient will move the continuous actor toward a feasible lower exit",
            "whether any repair improves new complete native episodes over A and matched B",
        ],
        "source_sha256": {
            "C_s1_batch_audit.npz": sha256(PILOT / "C_s1_batch_audit.npz"),
            "B_s1_batch_audit.npz": sha256(PILOT / "B_s1_batch_audit.npz"),
            "dataset": sha256(DATASET),
            "script": sha256(__file__),
        },
    }
    write_json(OUT / "branch_c_results.json", audit)
    print(json.dumps(plain(audit), indent=2))


if __name__ == "__main__":
    main()
