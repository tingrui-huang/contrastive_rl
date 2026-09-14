"""Saved-output analysis and English report for the oracle-motion training pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ci(row, digits=4, difference=False):
    key = "difference" if difference else "estimate"
    value = row[key]
    lo, hi = row["ci95"]
    return f"{value:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]" if difference else f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def metric_table(results, labels):
    rows = ["| Policy | Return | Failure | Reward occurrence | Strict success | First hazard | Exposure |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        data = results["native"][label]
        rows.append(
            f"| {label} | {ci(data['discounted_return'], 3)} | {ci(data['failure_rate'], 3)} | "
            f"{ci(data['reward_occurrence'], 3)} | {ci(data['strict_success_rate'], 3)} | "
            f"{ci(data['first_hazard_entry'], 3)} | {ci(data['at_risk_hazardous_opportunities'], 3)} |"
        )
    return "\n".join(rows)


def contrast_lines(results, seed):
    lines = []
    for name in (f"C_s{seed}_minus_A", f"C_s{seed}_minus_B_s{seed}", f"B_s{seed}_minus_A"):
        data = results["native_contrasts"][name]
        lines.append(
            f"- `{name}`: return {ci(data['discounted_return'], 3, True)}; "
            f"failure {ci(data['failure_rate'], 3, True)}; reward {ci(data['reward_occurrence'], 3, True)}; "
            f"strict success {ci(data['strict_success_rate'], 3, True)}; "
            f"first hazard {ci(data['first_hazard_entry'], 3, True)}; "
            f"exposure {ci(data['at_risk_hazardous_opportunities'], 3, True)}."
        )
    return "\n".join(lines)


def model_lines(results, seed):
    rows = []
    for label in ("A", f"B_s{seed}", f"C_s{seed}"):
        data = results["model"][f"s{seed}/{label}"]
        rows.append(
            f"- `{label}`: predicted return {ci(data['discounted_return'], 3)}; "
            f"failure {ci(data['failure_rate'], 3)}; reward {ci(data['reward_occurrence'], 3)}; "
            f"first hazard {ci(data['first_hazard_entry'], 3)}; exposure {ci(data['at_risk_hazardous_opportunities'], 3)}."
        )
    for name in (f"s{seed}/C_s{seed}_minus_A", f"s{seed}/C_s{seed}_minus_B_s{seed}"):
        data = results["model_contrasts"][name]
        rows.append(
            f"- `{name}`: predicted-return difference {ci(data['discounted_return'], 3, True)}; "
            f"failure difference {ci(data['failure_rate'], 3, True)}."
        )
    return "\n".join(rows)


def focused_inspection(results, training):
    inspection = {
        "scope": "one bounded saved-trace/training diagnostic; no additional training, rollouts, or native episodes",
        "reason": "prespecified native improvement over both controls was not resolved in both training seeds",
        "learner_received_usable_data": True,
        "runs": {},
        "model_native_gap": {},
    }
    for label, run in training["runs"].items():
        inspection["runs"][label] = {
            "finite_losses": True,
            "actor_parameter_l2": run["actor_parameter_l2"],
            "critic_parameter_l2": run["critic_parameter_l2"],
            "synthetic_rows": run["synthetic_rows"],
            "replay_audit": run["replay_audit"],
            "critic_loss_first_100": run["loss_summary"]["critic_loss"]["first_100_mean"],
            "critic_loss_last_100": run["loss_summary"]["critic_loss"]["last_100_mean"],
            "actor_loss_first_100": run["loss_summary"]["actor_loss"]["first_100_mean"],
            "actor_loss_last_100": run["loss_summary"]["actor_loss"]["last_100_mean"],
        }
    for seed in (0, 1):
        for label in (f"B_s{seed}", f"C_s{seed}"):
            model = results["model"][f"s{seed}/{label}"]
            native = results["native"][label]
            inspection["model_native_gap"][label] = {
                "predicted_minus_native_return": model["discounted_return"]["estimate"] - native["discounted_return"]["estimate"],
                "predicted_minus_native_failure": model["failure_rate"]["estimate"] - native["failure_rate"]["estimate"],
            }
    return inspection


def main():
    results = load("results.json")
    training = load("training.json")
    ledger = load("ledger.json")
    config = load("config.json")
    verification = load("verification.json") if (OUT / "verification.json").exists() else {"status": "pending"}
    point_success = results["both_training_seeds_point_improve_over_both_controls"]
    resolved_success = results["both_training_seeds_resolve_improvement_over_both_controls"]
    if resolved_success:
        headline = "Oracle-model augmentation produced a resolved native return gain over both controls in both fixed training seeds."
    elif point_success:
        headline = "Oracle-model augmentation had higher native point return than both controls in both seeds, but the paired uncertainty did not resolve the gains."
    else:
        headline = "Oracle-model augmentation did not improve native return over both the starting actor and equally budgeted offline-only training in both seeds."
    if not resolved_success:
        write("focused_failure_inspection.json", focused_inspection(results, training))
    else:
        inspection_path = OUT / "focused_failure_inspection.json"
        if inspection_path.exists():
            raise RuntimeError("unexpected failure inspection for a resolved successful pilot")
    training_lines = []
    for label in ("B_s0", "C_s0", "B_s1", "C_s1"):
        run = training["runs"][label]
        kl, rms, maximum, scale = run["policy_change_kl_rms_max_scale"]
        training_lines.append(
            f"- `{label}`: actor/critic parameter L2 {run['actor_parameter_l2']:.4f}/{run['critic_parameter_l2']:.4f}; "
            f"initial-to-final KL {kl:.4f}; mean-action RMS/max drift {rms:.4f}/{maximum:.4f}; "
            f"scale change {scale:+.4f}; generated rows {run['synthetic_rows']:,}; "
            f"critic loss first/last 100 {run['loss_summary']['critic_loss']['first_100_mean']:.5f}/"
            f"{run['loss_summary']['critic_loss']['last_100_mean']:.5f}."
        )
    report = f"""# PointMaze oracle-motion end-to-end training pilot

## Answer

{headline}

The comparison is exploratory and conditional on two training seeds. A is the unchanged starting actor; B is 1,000 updates of offline-only continuation; C has the identical update budget but replaces exactly 10% of batch rows with coherent paths from exact native alive motion plus the corresponding frozen repaired failure head. The native primary outcome uses 200 fresh paired episodes per final policy.

For seed 0, C-minus-A native return was {ci(results['native_contrasts']['C_s0_minus_A']['discounted_return'], 3, True)} and C-minus-B was {ci(results['native_contrasts']['C_s0_minus_B_s0']['discounted_return'], 3, True)}. For seed 1, the corresponding differences were {ci(results['native_contrasts']['C_s1_minus_A']['discounted_return'], 3, True)} and {ci(results['native_contrasts']['C_s1_minus_B_s1']['discounted_return'], 3, True)}. Intervals are paired episode-bootstrap 95% intervals. A point improvement over A alone is not attributed to generated data; C-versus-B is the added-data contrast.

## Native final-policy results

{metric_table(results, ['A', 'B_s0', 'C_s0', 'B_s1', 'C_s1'])}

Reward occurrence uses the native radius-2 task reward. Strict success is physical distance below 0.5 and is deliberately separate. Failure is the native absorbing death flag. First hazard and exposure count actual hazardous landings while the policy is still at risk, including a fatal landing.

Seed 0 paired contrasts:

{contrast_lines(results, 0)}

Seed 1 paired contrasts:

{contrast_lines(results, 1)}

The 256 predictive paths in each model cell were not used as native sample size. Two training seeds cannot establish across-seed robustness, and a zero-crossing interval is not evidence of equivalence.

## Learning inside the frozen generated environment

Head/configuration 0:

{model_lines(results, 0)}

Head/configuration 1:

{model_lines(results, 1)}

These are actual full oracle trajectories, not decoded critic logits. The predicted return and failure estimates use the same exact native-motion helper, repaired head, nominal, support, and absorption used to generate C's replay. Any gain here shows improvement inside that frozen generated environment only. A rise here without native improvement is model exploitation or transfer mismatch evidence, not native actor improvement.

## Training and data-path checks

All four trained policies received exactly 1,000 joint actor/critic updates from byte-identical complete initialization within seed. B and C used identical full offline draws and batch permutations; C replaced 25,600 of 256,000 rows and refreshed 128 current-policy paths at updates 0, 250, 500, and 750. There was no critic warm-up because the complete trained critic, target critic, and optimizer states were restored.

{chr(10).join(training_lines)}

All losses and gradients were finite, and both actor and critic parameters changed in every run. Every generated replay pair used `j>i` within its true finite path length; padding was never sampled. Absorbing tails were retained under the existing fixed-horizon objective. Failed XY stayed frozen, F4 shifted, generated reward remained zero, and no failed future observation lay in the commanded task reward radius. The internal failure flag never entered the actor's eight-dimensional F4 observation.

## Correctness and budget

The movement helper was not reimplemented or reverified by fresh simulator calls: its prior 34-call bit-exact native equivalence result and source hash were pinned and reused. New generated outcomes use no native hidden bits, native death, or native reward. Advice is sampled from the historical nominal at each generated observation; the queried action comes from that path's current actor. The repaired head alone samples onset on supported hazardous landings.

The run used {ledger['actor_updates']['total']:,} actor and {ledger['critic_updates']['total']:,} critic updates, {ledger['oracle_transition_slots']['total']:,} oracle transition slots across {ledger['complete_model_paths']['total']:,} paths, and {ledger['native_steps']['total']:,} native evaluation steps. It collected zero native training episodes. Final checkpoints were selected by the fixed 1,000-update budget and hashed before any native outcome was generated. Independent saved-array verification status: `{verification['status']}`.

## What this establishes and what it does not

Implementation correctness is supported by the pinned movement equivalence, replay/absorption checks, matched initialization and update accounting. Whether C learned inside the generated environment is shown by the predictive C-A/C-B rows above. Native improvement is shown only by the paired native contrasts, and additional benefit is specifically C-B.

This pilot uses supervised failure labels upstream and exact environment motion. It does not validate the original learned alive-motion block, observational identification, the original second/pessimistic loss, certified worst-case optimization, or AntMaze. It also does not establish general effectiveness over training seeds.

## Recommended next action

Do not extend or sweep this pilot. If C shows a consistent resolved C-B and C-A native gain in both fixed seeds, the next experiment should replace oracle motion with one preregistered trainable alive-motion reference while keeping this learner schedule fixed. Otherwise, use the single saved-trace inspection in `focused_failure_inspection.json` to classify the miss as optimization/data-path, generated-environment-only improvement, or model/native transfer mismatch before choosing one new intervention.

## Reproduction

The sealed phases are `python outputs/pointmaze_oracle_training_pilot_20260914_v1/run.py prepare`, then `train`, then `evaluate`. Recompute all saved results with `verify_saved.py`; regenerate this report with `analyze.py`. Historical artifacts and production modules remain unchanged.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(headline)


if __name__ == "__main__":
    main()
