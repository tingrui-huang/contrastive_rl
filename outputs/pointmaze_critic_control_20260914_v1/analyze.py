"""Analyze the fixed-final PointMaze critic A/B control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
CONDITIONS = ("A_nce_only", "B_nce_plus_rank")


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


def summarize(values, weights):
    values = np.asarray(values, np.float64)
    boot = weights @ values / weights.sum(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]),
        "episode_count": len(values),
    }


def final_metrics(evaluations, condition, split, weights):
    gap = evaluations[f"{condition}__{split}_gap"][-1]
    label = evaluations[f"{condition}__{split}_label"][-1]
    signed = label * gap
    result = {
        "signed_gap": summarize(signed, weights),
        "ranking_accuracy": summarize((signed > 0).astype(np.float64), weights),
        "raw_down_minus_right_gap": summarize(gap, weights),
        "per_episode_gap": gap,
        "per_episode_teacher_sign": label,
    }
    return result


def main():
    roots = load_npz(OUT / "roots_and_split.npz")
    labels = load_npz(OUT / "ranking_labels.npz")
    native = load_npz(OUT / "native_fork_traces.npz")
    model = load_npz(OUT / "model_fork_traces.npz")
    evaluations = load_npz(OUT / "training_curves_and_evaluations.npz")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    train_weights = rng.multinomial(
        CONFIG["train_root_count"], np.full(CONFIG["train_root_count"], 1 / CONFIG["train_root_count"]),
        size=CONFIG["bootstrap_replicates"],
    )
    validation_weights = rng.multinomial(
        CONFIG["validation_root_count"],
        np.full(CONFIG["validation_root_count"], 1 / CONFIG["validation_root_count"]),
        size=CONFIG["bootstrap_replicates"],
    )
    reference_weights = rng.multinomial(
        16, np.full(16, 1 / 16), size=CONFIG["bootstrap_replicates"]
    )
    np.savez_compressed(
        OUT / "bootstrap_weights.npz", train=train_weights, validation=validation_weights,
        reference=reference_weights
    )

    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = {
            "train": final_metrics(evaluations, condition, "train", train_weights),
            "validation": final_metrics(evaluations, condition, "validation", validation_weights),
            "reference": final_metrics(evaluations, condition, "reference", reference_weights),
            "heldout_native_nce_loss_initial": float(
                evaluations[f"{condition}__heldout_native_nce_loss"][0]
            ),
            "heldout_native_nce_loss_final": float(
                evaluations[f"{condition}__heldout_native_nce_loss"][-1]
            ),
        }

    val_a_gap = evaluations["A_nce_only__validation_gap"][-1]
    val_b_gap = evaluations["B_nce_plus_rank__validation_gap"][-1]
    val_label = evaluations["A_nce_only__validation_label"][-1]
    paired_signed_improvement = val_label * (val_b_gap - val_a_gap)
    comparison = {
        "heldout_B_minus_A_teacher_signed_gap": summarize(
            paired_signed_improvement, validation_weights
        ),
        "per_episode_signed_improvement": paired_signed_improvement,
    }

    train = roots["train_root"]
    validation = roots["validation_root"]
    teacher = {}
    for name, selected, weights in (
        ("train", train, train_weights), ("validation", validation, validation_weights)
    ):
        model_gap = labels["model_return_gap"][selected]
        native_gap = labels["native_return_gap"][selected]
        agreement = np.sign(model_gap) == np.sign(native_gap)
        teacher[name] = {
            "model_down_minus_right_return": summarize(model_gap, weights),
            "native_down_minus_right_return": summarize(native_gap, weights),
            "direction_agreement": summarize(agreement.astype(np.float64), weights),
            "model_preferred_down_fraction": float(np.mean(model_gap > 0)),
            "native_preferred_down_fraction": float(np.mean(native_gap > 0)),
        }

    a_clear = metrics["A_nce_only"]["validation"]["signed_gap"]["ci95"][0] > 0
    b_clear = metrics["B_nce_plus_rank"]["validation"]["signed_gap"]["ci95"][0] > 0
    improvement_clear = comparison["heldout_B_minus_A_teacher_signed_gap"]["ci95"][0] > 0
    if a_clear:
        decision = "coverage_sufficient"
    elif b_clear and improvement_clear:
        decision = "ranking_adds_missing_signal"
    elif not a_clear and not b_clear:
        decision = "neither_resolves"
    else:
        decision = "inconclusive_other"

    training = json.loads((OUT / "training_results.json").read_text(encoding="utf-8"))
    for condition in CONDITIONS:
        initial_gap = evaluations[f"{condition}__validation_gap"][0]
        if condition == "B_nce_plus_rank":
            np.testing.assert_array_equal(initial_gap, evaluations["A_nce_only__validation_gap"][0])
    result = {
        "status": "complete",
        "decision": decision,
        "decision_rule": CONFIG["decision_rule"],
        "teacher": teacher,
        "conditions": metrics,
        "paired_comparison": comparison,
        "data": {
            "new_complete_native_discovery_episodes": CONFIG["discovery_episode_count"],
            "new_roots": CONFIG["root_target"],
            "train_episode_roots": CONFIG["train_root_count"],
            "validation_episode_roots": CONFIG["validation_root_count"],
            "native_fork_paths": int(np.prod(native["valid"].shape[:3])),
            "native_fork_transitions": int(native["valid"].sum()),
            "model_fork_paths": int(np.prod(model["valid"].shape[:3])),
            "model_fork_transitions": int(model["valid"].sum()),
            "common_nce_batches": True,
            "old_16_roots_role": "external reference only",
        },
        "training": training,
        "limits": {
            "actor_updates": 0,
            "model_updates": 0,
            "new_actor_training_episodes": 0,
            "checkpoint_searches": 0,
            "final_checkpoint_fixed_at_update": CONFIG["critic_updates_per_condition"],
        },
    }
    write_json(OUT / "results.json", result)

    colors = {"A_nce_only": "tab:blue", "B_nce_plus_rank": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for condition in CONDITIONS:
        steps = evaluations[f"{condition}__update"]
        label = evaluations[f"{condition}__validation_label"]
        signed = label * evaluations[f"{condition}__validation_gap"]
        axes[0].plot(steps, signed.mean(axis=1), marker="o", color=colors[condition], label=condition)
        axes[1].plot(
            steps, evaluations[f"{condition}__heldout_native_nce_loss"], marker="o",
            color=colors[condition], label=condition
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="critic updates", ylabel="held-out teacher-signed logit gap",
                title="Held-out task-action ranking")
    axes[1].set(xlabel="critic updates", ylabel="held-out native NCE loss",
                title="Same held-out fork NCE batches")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=150)
    plt.close(fig)

    order = np.arange(CONFIG["validation_root_count"])
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.5))
    signed_a = evaluations["A_nce_only__validation_label"][-1] * val_a_gap
    signed_b = evaluations["B_nce_plus_rank__validation_label"][-1] * val_b_gap
    axis.bar(order - width / 2, signed_a, width, label="A: NCE only")
    axis.bar(order + width / 2, signed_b, width, label="B: NCE + rank")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="held-out native episode root", ylabel="teacher-signed down-right critic gap",
             title="Fixed-final critic ranking on disjoint episodes")
    axis.set_xticks(order)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "heldout_root_gaps.png", dpi=150)
    plt.close(fig)

    a = metrics["A_nce_only"]["validation"]
    b = metrics["B_nce_plus_rank"]["validation"]
    diff = comparison["heldout_B_minus_A_teacher_signed_gap"]
    teacher_val = teacher["validation"]
    report = f"""# PointMaze critic A/B control

## Answer

The preregistered decision is **{decision}**. Both critics started from the same frozen C1 checkpoint, consumed the same sealed native fork NCE rows in the same order, and received {CONFIG['critic_updates_per_condition']} critic-only updates. A used only the existing sigmoid-NCE loss. B used the identical NCE loss plus the model-return-derived pairwise action-ranking loss.

On the eight held-out new episode roots, A's final teacher-signed critic gap was **{a['signed_gap']['mean']:.3f} [{a['signed_gap']['ci95'][0]:.3f}, {a['signed_gap']['ci95'][1]:.3f}]**, with ranking accuracy **{100*a['ranking_accuracy']['mean']:.1f}%**. B's gap was **{b['signed_gap']['mean']:.3f} [{b['signed_gap']['ci95'][0]:.3f}, {b['signed_gap']['ci95'][1]:.3f}]**, with accuracy **{100*b['ranking_accuracy']['mean']:.1f}%**. The paired B-minus-A signed-gap difference was **{diff['mean']:.3f} [{diff['ci95'][0]:.3f}, {diff['ci95'][1]:.3f}]**.

Held-out native NCE loss changed from {metrics['A_nce_only']['heldout_native_nce_loss_initial']:.6f} to {metrics['A_nce_only']['heldout_native_nce_loss_final']:.6f} for A and to {metrics['B_nce_plus_rank']['heldout_native_nce_loss_final']:.6f} for B. These losses and contrastive logits are not calibrated returns.

## New data and teacher check

Thirty-two new complete native episodes were generated with the frozen C1 actor. The first 24 eligible naturally reached alive fork contexts were split by episode into 16 train and 8 validation roots before fork outcomes. From them, the experiment generated {result['data']['native_fork_paths']} native complete fork continuations ({result['data']['native_fork_transitions']} transitions) shared by A/B, plus {result['data']['model_fork_paths']} frozen-model continuations used only to compute ranking labels.

On held-out roots, model down-minus-right discounted return was **{teacher_val['model_down_minus_right_return']['mean']:.3f} [{teacher_val['model_down_minus_right_return']['ci95'][0]:.3f}, {teacher_val['model_down_minus_right_return']['ci95'][1]:.3f}]**. The independently generated native fork estimate was **{teacher_val['native_down_minus_right_return']['mean']:.3f} [{teacher_val['native_down_minus_right_return']['ci95'][0]:.3f}, {teacher_val['native_down_minus_right_return']['ci95'][1]:.3f}]**. Their per-root preferred directions agreed on **{100*teacher_val['direction_agreement']['mean']:.1f}%** of held-out episodes. Labels were computed from model returns per root; no route direction was hard-coded.

Every training NCE batch contained all 256 train root/action/path identities once. Exactly 128 rows anchored at the actual fork, balanced 64 down and 64 right; the remaining rows used later states. Every positive goal was a strictly later full-F4 state from the same native continuation under the original geometric law.

## Interpretation

`coverage_sufficient` means matched native coverage and frozen-policy continuations were enough for ordinary NCE to recover the held-out task-action ordering. `ranking_adds_missing_signal` means the same data and NCE were insufficient, while the explicit ordinal task signal changed held-out ordering. `neither_resolves` leaves representation, optimization budget, ranking strength, and goal semantics unresolved. The fixed old 16 roots are reported only as external reference and did not affect training, splitting, labels, decisions, or checkpoint choice.

This experiment does not update or test the actor. It establishes only whether each critic can learn the desired ordering under a controlled data intervention. Any selected modification still requires a separately preregistered actor test and new complete native confirmation episodes.

## Integrity

Actor updates, model updates, actor-training episodes, checkpoint searches, and hyperparameter sweeps were zero. The explicitly requested discovery episodes and native fork continuations were used only for this critic control. Both final checkpoints were fixed at update {CONFIG['critic_updates_per_condition']}. Historical files were not modified.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    artifact_names = [
        "REPORT.md", "results.json", "learning_curves.png", "heldout_root_gaps.png",
        "training_curves_and_evaluations.npz", "training_results.json", "ranking_labels.npz",
        "native_fork_traces.npz", "model_fork_traces.npz",
    ]
    write_json(
        OUT / "completion.json",
        {
            "status": "analysis_complete_pending_independent_verification",
            "decision": decision,
            "artifact_sha256": {name: sha256(OUT / name) for name in artifact_names},
            "actor_updates": 0,
            "model_updates": 0,
            "new_actor_training_episodes": 0,
        },
    )
    print(json.dumps({"decision": decision, "A_validation": a, "B_validation": b,
                      "B_minus_A": diff}, indent=2, default=plain))


if __name__ == "__main__":
    main()
