"""Analyze saved matched-fork arrays; never calls an environment or trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
GAMMA = float(CONFIG["discount"])
METRICS = (
    "whole_episode_return",
    "continuation_return",
    "strict_success",
    "reward_occurrence",
    "failure",
    "hazard_exposure",
    "lower_route_entry",
)


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


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path_metrics(trace, roots):
    valid = trace["valid"]
    horizon = valid.shape[-1]
    powers = GAMMA ** np.arange(horizon)
    continuation = np.sum(trace["reward"] * valid * powers, axis=-1)
    prefix_reward = roots["saved_prefix_reward"]
    prefix_powers = GAMMA ** np.arange(prefix_reward.shape[1])
    prefix_return = np.sum(prefix_reward * prefix_powers, axis=-1)
    whole = prefix_return[:, None, None] + (GAMMA ** roots["time"])[:, None, None] * continuation
    return {
        "whole_episode_return": whole,
        "continuation_return": continuation,
        "strict_success": np.any(trace["strict_success"] & valid, axis=-1).astype(np.float64),
        "reward_occurrence": np.any((trace["reward"] > 0) & valid, axis=-1).astype(np.float64),
        "failure": np.any(trace["failed_after"] & valid, axis=-1).astype(np.float64),
        "hazard_exposure": np.sum(trace["hazard_landing"] & (~trace["failed_before"]) & valid, axis=-1).astype(np.float64),
        "lower_route_entry": np.any(trace["lower_route"] & valid, axis=-1).astype(np.float64),
    }


def estimate(paths, weights):
    paths = np.asarray(paths, np.float64)
    if paths.ndim == 1:
        root_mean = paths
        conditional_mc_se = 0.0
        path_sd_mean = 0.0
    else:
        root_mean = paths.mean(axis=-1)
        if paths.shape[-1] > 1:
            variance = paths.var(axis=-1, ddof=1)
            conditional_mc_se = float(np.sqrt(np.sum(variance / paths.shape[-1])) / len(root_mean))
            path_sd_mean = float(np.sqrt(variance).mean())
        else:
            conditional_mc_se = 0.0
            path_sd_mean = 0.0
    boot = weights @ root_mean
    return {
        "point": float(root_mean.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]),
        "root_count": len(root_mean),
        "paths_per_root": int(paths.shape[-1]) if paths.ndim > 1 else 1,
        "root_sd": float(root_mean.std(ddof=1)) if len(root_mean) > 1 else 0.0,
        "conditional_predictive_mc_se": conditional_mc_se,
        "mean_within_root_path_sd": path_sd_mean,
        "root_values": root_mean,
    }


def compact_estimate(row):
    return {key: value for key, value in row.items() if key != "root_values"}


def summarize_backend(trace, roots, weights):
    values = path_metrics(trace, roots)
    names = [str(value) for value in trace["action_name"]]
    arms = {}
    for action_id, name in enumerate(names):
        arms[name] = {metric: compact_estimate(estimate(values[metric][:, action_id], weights)) for metric in METRICS}
    return values, arms


def contrast(values, left, right, weights):
    result = {}
    for metric in METRICS:
        result[metric] = compact_estimate(estimate(values[metric][:, left] - values[metric][:, right], weights))
    return result


def backend_difference(model_values, native_values, action_id, weights):
    result = {}
    for metric in METRICS:
        result[metric] = compact_estimate(
            estimate(model_values[metric][:, action_id] - native_values[metric][:, action_id], weights)
        )
    return result


def critic_summary(scores, weights):
    return compact_estimate(estimate(scores[:, 0] - scores[:, 1], weights))


def reversal_summary(native_trace, model_trace, weights):
    result = {}
    for backend, trace in (("native", native_trace), ("model", model_trace)):
        # Down is action index zero. State index 1 is after the forced action;
        # state index 2 is after the actor's first resumed action.
        xy = trace["physical_states"][:, 0]
        valid = trace["valid"][:, 0]
        first_actor_up = (xy[:, :, 2, 1] - xy[:, :, 1, 1]) > 0.1
        actor_action_up = trace["action"][:, 0, :, 1, 1] > 0.1
        reached_lower = np.any(trace["lower_route"][:, 0] & valid, axis=-1)
        returned_before_lower = np.zeros_like(reached_lower)
        for root in range(xy.shape[0]):
            for repeat in range(xy.shape[1]):
                lowered = False
                for step in range(1, valid.shape[-1] + 1):
                    if not valid[root, repeat, step - 1]:
                        break
                    if xy[root, repeat, step, 1] < 2.0:
                        lowered = True
                        break
                    if step >= 2 and xy[root, repeat, step, 1] >= 3.0:
                        returned_before_lower[root, repeat] = True
                        break
                if lowered:
                    returned_before_lower[root, repeat] = False
        result[backend] = {
            "first_resumed_actor_action_y_positive": compact_estimate(estimate(actor_action_up, weights)),
            "first_resumed_movement_reverses_upward": compact_estimate(estimate(first_actor_up, weights)),
            "returns_to_upper_route_before_lower_entry": compact_estimate(estimate(returned_before_lower, weights)),
            "eventual_lower_route_entry": compact_estimate(estimate(reached_lower, weights)),
            "minimum_y_during_first_four_transitions": compact_estimate(
                estimate(np.nanmin(xy[:, :, 1:5, 1], axis=-1), weights)
            ),
        }
    return result


def decision(native_contrast, model_contrast, critic):
    native_clear = native_contrast["whole_episode_return"]["ci95"][0] > 0
    model_clear = model_contrast["whole_episode_return"]["ci95"][0] > 0
    critic_clear = critic["ci95"][0] > 0
    if not native_clear:
        branch = "A"
    elif not model_clear:
        branch = "B"
    elif not critic_clear:
        branch = "C"
    else:
        branch = "D"
    return {
        "selected_branch": branch,
        "rule": CONFIG["branch_rule"][branch],
        "native_clear_down_advantage": native_clear,
        "model_clear_down_advantage": model_clear,
        "critic_clear_down_preference": critic_clear,
        "no_advantage_established_is_not_equivalence": True,
        "precision_extension_allowed": False,
    }


def analyze_primary():
    roots = load_npz(OUT / "root_selection.npz")
    weights = load_npz(OUT / "bootstrap_weights.npz")["weights"]
    native = load_npz(OUT / "primary_native_traces.npz")
    model = load_npz(OUT / "primary_model_traces.npz")
    scores = load_npz(OUT / "critic_scores.npz")["score"]
    native_values, native_arms = summarize_backend(native, roots, weights)
    model_values, model_arms = summarize_backend(model, roots, weights)
    native_down_right = contrast(native_values, 0, 1, weights)
    model_down_right = contrast(model_values, 0, 1, weights)
    critic = critic_summary(scores, weights)
    result = {
        "design": {
            "roots": len(roots["episode"]),
            "root_episode_indices": roots["episode"],
            "root_times": roots["time"],
            "paths_per_root_action_backend": CONFIG["paths_per_root_action_backend"],
            "uncertainty_unit": "saved native episode/root after averaging predictive continuations",
            "predictive_paths_are_not_independent_episodes": True,
        },
        "native": native_arms,
        "model": model_arms,
        "down_minus_right": {"native": native_down_right, "model": model_down_right},
        "model_minus_native": {
            "down": backend_difference(model_values, native_values, 0, weights),
            "right": backend_difference(model_values, native_values, 1, weights),
        },
        "critic_down_minus_right_logit": critic,
        "reversal_after_one_step_down": reversal_summary(native, model, weights),
    }
    selected = decision(native_down_right, model_down_right, critic)
    write_json(OUT / "primary_results.json", result)
    write_json(OUT / "primary_decision.json", selected)
    np.savez_compressed(
        OUT / "primary_episode_results.npz",
        **{
            f"native_{metric}": native_values[metric]
            for metric in METRICS
        },
        **{
            f"model_{metric}": model_values[metric]
            for metric in METRICS
        },
        critic_score=scores,
        root_episode=roots["episode"],
        root_time=roots["time"],
    )
    print(json.dumps(plain({"decision": selected, "native_down_right": native_down_right, "model_down_right": model_down_right, "critic": critic}), indent=2))
    return result, selected


def analyze_followup(primary):
    roots = load_npz(OUT / "root_selection.npz")
    weights = load_npz(OUT / "bootstrap_weights.npz")["weights"]
    primary_native = load_npz(OUT / "primary_native_traces.npz")
    primary_model = load_npz(OUT / "primary_model_traces.npz")
    prefix_native = load_npz(OUT / "followup_native_traces.npz")
    prefix_model = load_npz(OUT / "followup_model_traces.npz")
    pn_values, pn_arms = summarize_backend(prefix_native, roots, weights)
    pm_values, pm_arms = summarize_backend(prefix_model, roots, weights)
    n_values = path_metrics(primary_native, roots)
    m_values = path_metrics(primary_model, roots)
    # Prefix arrays have one arm; compare with the existing primary right arm.
    native_diff = {
        metric: compact_estimate(estimate(pn_values[metric][:, 0] - n_values[metric][:, 1], weights))
        for metric in METRICS
    }
    model_diff = {
        metric: compact_estimate(estimate(pm_values[metric][:, 0] - m_values[metric][:, 1], weights))
        for metric in METRICS
    }
    return {
        "executed_branch": "A",
        "prefix": CONFIG["conditional_prefix_actions"],
        "right_arm_reused_from_primary": True,
        "native_prefix": pn_arms["prefix"],
        "model_prefix": pm_arms["prefix"],
        "prefix_minus_right": {"native": native_diff, "model": model_diff},
        "prefix_value_not_compared_to_one_step_critic": True,
    }


def fmt(row, digits=3):
    return f"{row['point']:.{digits}f} [{row['ci95'][0]:.{digits}f}, {row['ci95'][1]:.{digits}f}]"


def plot_trajectories(final):
    primary_native = load_npz(OUT / "primary_native_traces.npz")
    primary_model = load_npz(OUT / "primary_model_traces.npz")
    panels = [("Native", primary_native), ("Frozen model", primary_model)]
    if (OUT / "followup_native_traces.npz").exists():
        panels.extend(
            [
                ("Native three-step prefix", load_npz(OUT / "followup_native_traces.npz")),
                ("Model three-step prefix", load_npz(OUT / "followup_model_traces.npz")),
            ]
        )
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    colors = {"down": "tab:blue", "right": "tab:orange", "prefix": "tab:green"}
    for axis, (title, trace) in zip(axes[0], panels):
        for action_id, name in enumerate(trace["action_name"]):
            name = str(name)
            xy = trace["physical_states"][:, action_id]
            for root in range(xy.shape[0]):
                mean_xy = np.nanmean(xy[root], axis=0)
                axis.plot(mean_xy[:, 0], mean_xy[:, 1], color=colors[name], alpha=0.35, linewidth=1)
            axis.plot([], [], color=colors[name], label=f"{name}: root-wise path mean")
        axis.axhspan(3, 4, xmin=3 / 9, xmax=6 / 9, color="tab:red", alpha=0.12, label="hazard corridor")
        axis.axhline(2, color="0.5", linestyle="--", linewidth=0.8)
        axis.scatter([8.5], [3.5], marker="*", s=90, color="black", label="task goal")
        axis.set(xlim=(0, 9), ylim=(0, 5), xlabel="x", ylabel="y", title=title)
        axis.set_aspect("equal")
        axis.legend(fontsize=7, loc="lower right")
    fig.suptitle("Matched fork continuations (64 paths averaged within each of 16 roots)")
    fig.tight_layout()
    fig.savefig(OUT / "matched_trajectories.png", dpi=160)
    plt.close(fig)


def plot_root_preferences():
    arrays = load_npz(OUT / "primary_episode_results.npz")
    native = (arrays["native_whole_episode_return"][:, 0] - arrays["native_whole_episode_return"][:, 1]).mean(axis=1)
    model = (arrays["model_whole_episode_return"][:, 0] - arrays["model_whole_episode_return"][:, 1]).mean(axis=1)
    critic = arrays["critic_score"][:, 0] - arrays["critic_score"][:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for axis, values, title, ylabel in zip(
        axes,
        (native, model, critic),
        ("Native", "Frozen model", "Saved C1 critic"),
        ("down - right return", "down - right return", "down - right logit"),
    ):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.scatter(np.arange(len(values)), values, s=24)
        axis.set(title=title, xlabel="root", ylabel=ylabel)
        axis.grid(alpha=0.2)
    fig.suptitle("Root-level preferences after averaging 64 predictive paths")
    fig.tight_layout()
    fig.savefig(OUT / "root_preferences.png", dpi=160)
    plt.close(fig)


def report(final, decision_row):
    nd = final["primary"]["down_minus_right"]["native"]["whole_episode_return"]
    md = final["primary"]["down_minus_right"]["model"]["whole_episode_return"]
    cd = final["primary"]["critic_down_minus_right_logit"]
    ns = final["primary"]["down_minus_right"]["native"]["strict_success"]
    ms = final["primary"]["down_minus_right"]["model"]["strict_success"]
    rev = final["primary"]["reversal_after_one_step_down"]
    branch = decision_row["selected_branch"]
    lines = [
        "# PointMaze matched-fork diagnostic",
        "",
        "## Answer",
        "",
        f"The sealed primary diagnostic selected branch **{branch}**. Native one-step down-minus-right whole-episode return was {fmt(nd)} and strict-success probability was {fmt(ns)}. The frozen oracle-motion/head-1 model gave return {fmt(md)} and strict success {fmt(ms)}. The saved C1 critic's raw down-minus-right logit was {fmt(cd)}. Predictive paths were first averaged within each root; intervals resample the 16 distinct native root episodes.",
        "",
    ]
    if branch == "A":
        follow = final["followup"]
        npr = follow["prefix_minus_right"]["native"]["whole_episode_return"]
        mpr = follow["prefix_minus_right"]["model"]["whole_episode_return"]
        nps = follow["prefix_minus_right"]["native"]["strict_success"]
        mps = follow["prefix_minus_right"]["model"]["strict_success"]
        lines.extend(
            [
                "The one-step intervention therefore establishes no clear native advantage; this is not evidence that the two actions are equivalent. The only allowed follow-up forced the predeclared `[down, down, right]` route-initiation prefix. Prefix-minus-existing-right return was "
                f"{fmt(npr)} natively and {fmt(mpr)} in the frozen model; strict-success differences were {fmt(nps)} and {fmt(mps)}.",
                "",
                f"After only the single down step, the resumed actor moved upward on its next transition in {fmt(rev['native']['first_resumed_movement_reverses_upward'])} of native paths and entered the lower route in {fmt(rev['native']['eventual_lower_route_entry'])}. This directly checks whether C1 undoes the deviation rather than assuming a one-step action represents the full safe route.",
                "",
            ]
        )
    lines.extend(
        [
            "## Fixed design and validity",
            "",
            "The 16 roots are the first qualifying saved C1 episodes in ascending array order. Every root was regenerated from its native reset seed by replaying saved actions; physical XY, full F4, time, reward and alive state matched exactly. The current native swamp bits were retained for the forced action and freshly resampled after every continuation step. Explicit scheduling reproduced the native Gaussian-noise/resampling order. Native hidden bits entered only native evaluation and initialization audits, never actor observations or the learned head.",
            "",
            "Candidate actions were fixed geometrically as `[0,-1]` and `[1,0]`. Zero-noise helper checks placed them in the lower exit cell `(1,2)` and shortcut holding cell `(2,3)` for every root. C1 actor/critic, repaired head 1, nominal policy, goal, horizon, reward and all model parameters remained frozen.",
            "",
            "The primary used exactly 4,096 paths. Actor innovations and motion noise were paired across actions and backends; future native bits were paired only across native actions, while model nominal draws and onset uniforms were paired only across model actions. Native bits were not equated with learned-head onset randomness.",
            "",
            "## Interpretation",
            "",
            "1. **Does initiating the detour help C1?** See the sealed native contrast above. An interval containing zero is reported as no advantage established, never equivalence.",
            "2. **Does the model reproduce the native preference?** The model contrast is reported beside the native contrast, with model-minus-native rows retained in `results.json`. Agreement concerns this frozen response component only.",
            "3. **Does the critic provide the task-useful preference?** The raw matched-action logit contrast is reported separately. It is not decoded as task return.",
            "4. **Does an actor update move toward it?** No actor update was run unless sealed branch D selected it. Route reversal/prefix evidence is behavior under the unchanged frozen actor, not an optimization claim.",
            "",
        ]
    )
    if branch == "A":
        npr = final["followup"]["prefix_minus_right"]["native"]["whole_episode_return"]
        if npr["ci95"][0] > 0:
            repair = "The supported next intervention is to supply coherent alive predecision training paths with coordinated multi-step lower-route initiation, while retaining complete failures and the unchanged task. A one-action critic target is insufficient for the behavior demonstrated here."
        else:
            repair = "Neither a single downward deviation nor the bounded coordinated prefix established a native return advantage for the frozen C1 continuation. The next justified intervention is not to force a critic down-ranking; it is to collect a preregistered coherent-alive predecision coverage set that varies the complete route initiation and measures native continuation value before changing training."
    elif branch == "B":
        repair = "The next repair should target the first saved model-native divergence in movement, nominal-conditioned onset, or later behavior; actor training is not justified until that model-side mismatch is repaired."
    elif branch == "C":
        repair = "The next intervention should add one controlled critic-learning signal for the demonstrated route preference, chosen after the saved replay audit separates coverage, continuation distribution and exact-F4 versus task-region identity. Do not change all three."
    else:
        repair = "The selected branch requires the bounded frozen-critic actor-objective probe. Any local action-probability movement remains diagnostic until separately validated with the capped native continuation comparison."
    lines.extend(
        [
            "## Decision",
            "",
            repair,
            "",
            "What remains unresolved includes generalization beyond this C1 checkpoint and 16 roots, whether a training repair improves complete native episodes over A and B, and whether learned ETT motion or observational identification succeeds. Confirmation of any modification requires new complete episodes.",
            "",
            "## Artifacts",
            "",
            "`preregistration.json`, `root_selection.*`, and `initialization_checks.json` seal provenance and initialization. `primary_*_traces.npz` and optional `followup_*_traces.npz` contain complete traces. `results.json`, `decision.json`, `matched_trajectories.png`, and `root_preferences.png` contain analysis and plots. `verify_saved.py` independently checks saved-array contracts.",
            "",
            "No training, checkpoint search, commit, push, AntMaze run, or historical-artifact modification occurred.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_final():
    primary, decision_row = analyze_primary()
    final = {"primary": primary, "decision": decision_row}
    if decision_row["selected_branch"] == "A":
        if not (OUT / "followup_native_traces.npz").exists():
            raise RuntimeError("branch A selected but sealed prefix follow-up has not run")
        final["followup"] = analyze_followup(primary)
    elif decision_row["selected_branch"] == "D" and (OUT / "actor_probe.json").exists():
        final["actor_probe"] = load_json(OUT / "actor_probe.json")
    write_json(OUT / "results.json", final)
    decision_final = {
        **decision_row,
        "conditional_followup_executed": "three_step_prefix" if decision_row["selected_branch"] == "A" else ("actor_objective_probe" if "actor_probe" in final else "saved_trace_localization_only"),
        "no_full_training_experiment": True,
        "confirmation_requires_new_complete_episodes": True,
    }
    write_json(OUT / "decision.json", decision_final)
    plot_trajectories(final)
    plot_root_preferences()
    report(final, decision_row)
    artifacts = [
        OUT / "primary_native_traces.npz",
        OUT / "primary_model_traces.npz",
        OUT / "primary_results.json",
        OUT / "primary_decision.json",
        OUT / "results.json",
        OUT / "decision.json",
        OUT / "REPORT.md",
        OUT / "matched_trajectories.png",
        OUT / "root_preferences.png",
    ]
    for optional in (OUT / "followup_native_traces.npz", OUT / "followup_model_traces.npz", OUT / "actor_probe.json"):
        if optional.exists():
            artifacts.append(optional)
    write_json(
        OUT / "completion.json",
        {
            "status": "analysis_complete_pending_independent_saved_verification",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_branch": decision_row["selected_branch"],
            "artifact_sha256": {path.name: sha256(path) for path in artifacts},
            "training_updates": 0,
            "new_training_episodes": 0,
        },
    )
    print(json.dumps(plain(decision_final), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=["primary", "final"])
    args = parser.parse_args()
    if args.phase == "primary":
        analyze_primary()
    else:
        analyze_final()


if __name__ == "__main__":
    main()
