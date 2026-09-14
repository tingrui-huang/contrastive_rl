"""Merge the saved-only audit and finalize the concise English report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    result = load("results.json")
    audit = load("posthoc_audit.json")
    result["posthoc_saved_checkpoint_audit"] = audit
    result["recommended_next_intervention"] = (
        "Use matched, naturally reached native fork continuations as an explicit NCE-only replay component "
        "while retaining the ordinary replay distribution; do not add the ranking loss yet. First verify task "
        "ranking and general held-out NCE retention with the actor frozen, then run a separately sealed actor-response "
        "probe and new complete native confirmation episodes."
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    a = result["conditions"]["A_nce_only"]
    b = result["conditions"]["B_nce_plus_rank"]
    diff = result["paired_comparison"]["heldout_B_minus_A_teacher_signed_gap"]
    tv = result["teacher"]["validation"]
    aa = audit["A_nce_only"]
    ba = audit["B_nce_plus_rank"]
    report = f"""# PointMaze critic A/B control

## Answer

The preregistered result is **coverage sufficient**. The saved C1 critic began wrong on every new held-out root: teacher-signed down-minus-right logit was **{aa['initial_validation_signed_gap_mean']:.3f}** with 0/8 correct. After 400 critic-only updates on explicit, matched native fork continuations, A—ordinary sigmoid NCE only—reached **{a['validation']['signed_gap']['mean']:.3f} [{a['validation']['signed_gap']['ci95'][0]:.3f}, {a['validation']['signed_gap']['ci95'][1]:.3f}]** and 8/8 correct. Thus improved route coverage plus continuation-policy matching was sufficient for the existing contrastive objective to learn the task-useful ordering on episode-disjoint roots.

B used the identical native data and NCE batches plus the model-return-derived ordinal loss. It reached **{b['validation']['signed_gap']['mean']:.3f} [{b['validation']['signed_gap']['ci95'][0]:.3f}, {b['validation']['signed_gap']['ci95'][1]:.3f}]** and 8/8 correct. Its paired signed-gap increase over A was **{diff['mean']:.3f} [{diff['ci95'][0]:.3f}, {diff['ci95'][1]:.3f}]**. This large margin is the direct effect of optimizing an ordinal logit gap; it is not evidence of a calibrated return estimate and is not needed to obtain the correct held-out ranking here.

Held-out native NCE loss started at {a['heldout_native_nce_loss_initial']:.6f}, ended at **{a['heldout_native_nce_loss_final']:.6f} for A**, and **{b['heldout_native_nce_loss_final']:.6f} for B**. Both improved over the start, but A fit the held-out native NCE task better than B. Critic parameter L2 movement was {aa['critic_parameter_l2_from_initial']:.3f} for A and {ba['critic_parameter_l2_from_initial']:.3f} for B.

## Data and teacher validity

Thirty-two new complete native episodes were run with the frozen C1 actor. Before fork outcomes, the first 24 eligible naturally reached alive fork contexts were split by episode into 16 train and 8 validation roots. A/B shared 384 complete native fork continuations and every one of their 400 NCE batches. Each batch contained all 256 train root/action/path identities once, including 64 down and 64 right rows anchored exactly at the fork; future full-F4 positives followed the original strictly-future geometric sampler.

The frozen model generated 1,536 additional continuations solely to compute labels. No direction was hard-coded. On held-out roots, model down-minus-right return was **{tv['model_down_minus_right_return']['mean']:.3f} [{tv['model_down_minus_right_return']['ci95'][0]:.3f}, {tv['model_down_minus_right_return']['ci95'][1]:.3f}]** and native was **{tv['native_down_minus_right_return']['mean']:.3f} [{tv['native_down_minus_right_return']['ci95'][0]:.3f}, {tv['native_down_minus_right_return']['ci95'][1]:.3f}]**; preferred direction agreed on 8/8 roots.

The old repeatedly inspected 16 roots were external reference only. They started at mean gap **{aa['initial_reference_signed_gap_mean']:.3f}**, 0/16 correct. Fixed-final A reached **{aa['final_reference_signed_gap_mean']:.3f}**, 16/16 correct; B reached **{ba['final_reference_signed_gap_mean']:.3f}**, 16/16 correct. These roots affected neither training nor the decision.

## What this establishes

The original wrong ranking is not evidence that the critic architecture cannot represent the detour preference. With direct exposure to balanced, naturally rooted native continuations generated under the final frozen actor, unchanged NCE learned the correct ordering and generalized across held-out episodes. The earlier failure is therefore localized to the training distribution/continuation mismatch at least strongly enough to be repaired by data alone in this fork-local pilot.

It does **not** establish that fork-only fine-tuning preserves arbitrary goals or that an updated actor improves full native success. B shows that an ordinal loss can force a much larger margin, but this experiment supplies no benefit of that extra margin and its held-out NCE loss is worse than A's.

## Recommended next intervention

Use the A-style change: add matched, naturally reached native fork continuations as an explicit **NCE-only** replay component while retaining the ordinary replay distribution. Do not add ranking loss yet. First verify both task ranking and general held-out NCE retention with the actor frozen; only then run a separately sealed actor-response probe and new complete native confirmation episodes.

## Integrity

Both conditions started from byte-identical C1 critic parameters and Adam state, used fixed update 400 as the final checkpoint, and performed no checkpoint search or sweep. Actor updates, model updates, and actor-training episodes were zero. One initial scoring attempt stopped before optimizer update because a single-head representation retained a `(N,1)` axis; the zero-update failure and one-line shape fix are preserved in `training_attempt_1.json` and `training_seal_amendment.json`. All data, labels, batches, budgets, and decision rules remained sealed and unchanged.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    completion = load("completion.json")
    names = [
        "REPORT.md", "results.json", "learning_curves.png", "heldout_root_gaps.png",
        "training_curves_and_evaluations.npz", "training_results.json", "ranking_labels.npz",
        "native_fork_traces.npz", "model_fork_traces.npz", "posthoc_audit.json",
        "posthoc_audit.py", "finalize.py", "training_attempt_1.json", "training_seal_amendment.json",
    ]
    completion["status"] = "analysis_complete_pending_independent_verification"
    completion["artifact_sha256"] = {name: sha256(OUT / name) for name in names}
    completion.pop("verification_sha256", None)
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "recommendation": result["recommended_next_intervention"]}, indent=2))


if __name__ == "__main__":
    main()
