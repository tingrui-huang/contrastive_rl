"""Attach the selected branch-C audit to final numerical and English outputs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def estimate(values, weights):
    root = np.asarray(values, np.float64).mean(axis=-1)
    return {
        "point": float(root.mean()),
        "ci95": np.quantile(weights @ root, [0.025, 0.975]).tolist(),
        "root_count": len(root),
        "paths_per_root": int(np.asarray(values).shape[-1]),
    }


def fmt(row, digits=3):
    return f"{row['point']:.{digits}f} [{row['ci95'][0]:.{digits}f}, {row['ci95'][1]:.{digits}f}]"


def pct(row):
    scaled = {"point": 100 * row["point"], "ci95": [100 * value for value in row["ci95"]]}
    return fmt(scaled, 1) + " percentage points"


def main():
    results = load(OUT / "results.json")
    decision = load(OUT / "decision.json")
    branch = load(OUT / "branch_c_results.json")
    if decision["selected_branch"] != "C" or branch["selected_branch"] != "C":
        raise RuntimeError("branch-C finalizer used for a different sealed decision")
    with np.load(OUT / "primary_episode_results.npz", allow_pickle=False) as arrays, np.load(
        OUT / "bootstrap_weights.npz", allow_pickle=False
    ) as boot:
        native_difference = arrays["native_whole_episode_return"][:, 0] - arrays["native_whole_episode_return"][:, 1]
        model_difference = arrays["model_whole_episode_return"][:, 0] - arrays["model_whole_episode_return"][:, 1]
        model_minus_native_preference = estimate(model_difference - native_difference, boot["weights"])
        critic_advantage = arrays["critic_score"][:, 0] - arrays["critic_score"][:, 1]
    results["conditional_branch_c_audit"] = branch
    results["primary"]["model_minus_native_down_right_preference"] = {
        "whole_episode_return": model_minus_native_preference
    }
    results["primary"]["critic_fraction_roots_preferring_down"] = float(np.mean(critic_advantage > 0))
    save(OUT / "results.json", results)

    recommendation = branch["recommended_single_repair"]
    decision.update(
        conditional_followup_executed="branch_C_saved_replay_and_objective_audit_only",
        prefix_followup_prohibited=True,
        actor_update_probe_prohibited=True,
        recommended_single_repair=recommendation,
        current_24_development_episode_status="reused exploratory evidence; any repair requires new complete episodes",
    )
    save(OUT / "decision.json", decision)

    primary = results["primary"]
    nd = primary["down_minus_right"]["native"]
    md = primary["down_minus_right"]["model"]
    native = primary["native"]
    model = primary["model"]
    critic = primary["critic_down_minus_right_logit"]
    actor = branch["frozen_actor_at_matched_roots"]
    coverage = branch["matched_coverage_comparison"]
    mn = primary["model_minus_native"]
    report = f"""# PointMaze matched-fork diagnostic

## Answer

The beneficial route decision fails at the saved C1 critic preference, before an actor-update test is warranted. From the same 16 coherent alive C1 roots, forcing one downward action and then resuming the same frozen C1 actor increased native whole-episode discounted return by **{fmt(nd['whole_episode_return'])}** relative to forcing right. Strict success increased by **{pct(nd['strict_success'])}**, failure decreased by **{pct(nd['failure'])}**, and lower-route entry increased by **{pct(nd['lower_route_entry'])}**.

This is already a one-step route-initiation result: after the forced down step, the unchanged actor entered the lower route in {100*native['down']['lower_route_entry']['point']:.1f}% of paths. The natural frozen actor nevertheless made a feasible down-cell exit on only **{100*actor['down_exit_probability']['point']:.1f}% [{100*actor['down_exit_probability']['ci95'][0]:.1f}%, {100*actor['down_exit_probability']['ci95'][1]:.1f}%]** of sealed action draws, versus a right-cell exit on **{100*actor['right_exit_probability']['point']:.1f}% [{100*actor['right_exit_probability']['ci95'][0]:.1f}%, {100*actor['right_exit_probability']['ci95'][1]:.1f}%]**.

The frozen oracle-motion/repaired-head-1 model reproduced the direction: model down-minus-right return was **{fmt(md['whole_episode_return'])}**, with strict-success difference **{pct(md['strict_success'])}**. It underestimated the native preference by {fmt(model_minus_native_preference)} because it slightly undervalued down and more materially overvalued right. In particular, model-minus-native return was {fmt(mn['down']['whole_episode_return'])} for down and {fmt(mn['right']['whole_episode_return'])} for right. This is a magnitude mismatch, not an incorrect model ranking.

The saved C1 critic ranked the exact same state/action/task-goal tuples in the opposite direction on every root: down-minus-right logit **{fmt(critic)}**, with 0/16 roots preferring down. These logits are ranking diagnostics, not calibrated returns.

## Matched outcomes

- Native down: return {fmt(native['down']['whole_episode_return'])}; strict success {100*native['down']['strict_success']['point']:.1f}%; failure {100*native['down']['failure']['point']:.1f}%; lower-route entry {100*native['down']['lower_route_entry']['point']:.1f}%.
- Native right: return {fmt(native['right']['whole_episode_return'])}; strict success {100*native['right']['strict_success']['point']:.1f}%; failure {100*native['right']['failure']['point']:.1f}%; lower-route entry {100*native['right']['lower_route_entry']['point']:.1f}%.
- Model down: return {fmt(model['down']['whole_episode_return'])}; strict success {100*model['down']['strict_success']['point']:.1f}%; failure {100*model['down']['failure']['point']:.1f}%.
- Model right: return {fmt(model['right']['whole_episode_return'])}; strict success {100*model['right']['strict_success']['point']:.1f}%; failure {100*model['right']['failure']['point']:.1f}%.

Each arm has 64 predictive continuations per root. Those paths integrate stochasticity within a root; intervals first average them and then bootstrap the 16 distinct saved native episodes. Conditional predictive Monte Carlo SE for the native return contrast was {nd['whole_episode_return']['conditional_predictive_mc_se']:.3f}, separate from across-root uncertainty.

## Branch-C replay and objective audit

The conditional audit reconstructed all 256,000 actual C1 learner rows, including the saved 90/10 source replacement and exact generated-path indices.

- Only **{coverage['C_s1_down_action_future_reward_positive_rows']} rows ({100*coverage['fraction_of_C_positive_rows']:.3f}%)** combined a matched fork state, recorded downward action, and a future achieved XY inside the task reward region. B1 had {coverage['B_s1_down_action_future_reward_positive_rows']} such rows, so oracle augmentation changed the count by {coverage['difference_C_minus_B']:+d} rather than increasing it.
- Requiring the observed next state to actually enter lower exit cell `(1,2)` leaves {coverage['C_s1_actual_down_exit_future_reward_positive_rows']} C1 rows versus {coverage['B_s1_actual_down_exit_future_reward_positive_rows']} B1 rows.
- C1's generated 10% contributed only 23 matched downward/task-region positive rows; the remaining 348 were offline.
- No learner row used the canonical stationary task F4 exactly as its positive goal. The critic instead received a particular achieved future F4 identity. Native return uses radius-2 reward and strict radius-0.5 success; neither scalar label enters this MC NCE loss.
- The 256x256 sigmoid-NCE loss includes every sample-pair cell. Each row has one logged future-state positive and its goal is off-diagonal for the other 255 anchors. It never receives the matched 64-continuation final-C1 action-value ordering measured here.

This supports sparse, non-improving route-relevant coverage and a different continuation target. Exact-F4 identity versus task-region success is a real semantic difference but is not isolated as the cause. Representation capacity versus optimization also remains unresolved: the checkpoint changed and losses were finite, but there was no controlled fit test in this diagnostic.

## Initialization and validity

Episodes 0–15 were selected in ascending saved C1 order; each first qualifying context occurred at time 1 with 49 steps remaining. Every root was regenerated from reset with its saved actions, reproducing physical XY, full F4, time, reward and alive state exactly. Candidate actions `[0,-1]` and `[1,0]` were fixed by geometry and entered cells `(1,2)` and `(2,3)` under zero noise for every root.

The naturally reached current swamp bits governed the forced step and fresh bits were installed after every subsequent native step. The explicit scheduler reproduced native action-noise and bit-resampling order. Hidden fields were saved only for native evaluation/initialization audit; they never entered actor observations or the learned head. Actor Gaussian innovations and motion noise were paired across actions and backends; native bits and model onset randomness were kept semantically separate.

The primary used exactly 4,096 paths and 100,352 transitions per backend in {load(OUT/'primary_execution.json')['runtime_seconds']:.1f} seconds. Initialization checks and root regeneration used 66 native steps. Training updates, new training episodes, checkpoint searches, prefix paths, and actor-probe updates were all zero. Branch C prohibited the other conditional branches.

## Required interpretation

1. **Does initiating the detour benefit this actor?** Yes for these roots: one forced downward step produces a large, clearly positive native return and success advantage, and the unchanged actor usually completes the descent.
2. **Does the frozen model reproduce the native preference?** Yes directionally. It still overvalues the right/shortcut continuation, so exact magnitude agreement is not established.
3. **Does the saved critic provide the task-useful preference?** No. It prefers right on all 16 matched roots despite both native and model continuation returns favoring down.
4. **Does the actor update move toward it?** Unresolved and deliberately untested. The sealed branch rules allow an actor probe only when the critic already has the correct preference; it does not.
5. **Which repair next?** Run one critic-only matched task-preference intervention. On a separately preregistered coherent-alive fork training split, add a pairwise task-goal ranking term labeled by frozen-model multi-continuation down-versus-right returns, while holding the actor, F4 representation, base NCE batches, BC, nominal, head and motion fixed. First require the critic ranking to flip on held-out coherent roots; only then inspect the actor gradient and new complete native episodes.

The current 16 roots and the prior 24 development episodes are exploratory evidence. They cannot confirm that the repair improves complete native success over A and matched B. Also unresolved are whether canonical task-region goal handling alone would fix the critic, whether the representation can fit both exact-F4 reachability and task preference, and whether a corrected continuous-action gradient changes behavior. Confirmation requires new complete episodes.

## Artifacts

`preregistration.json`, `PROTOCOL.md`, `root_selection.*`, and `initialization_checks.json` seal design and initialization. `primary_native_traces.npz` and `primary_model_traces.npz` contain complete traces and stochastic inputs. `results.json`, `branch_c_results.json`, `decision.json`, `matched_trajectories.png`, and `root_preferences.png` contain numerical results and plots. Reproduction and verification code is `run.py`, `analyze.py`, `branch_c_audit.py`, `finalize_branch_c.py`, `verify_saved.py`, and `verify_complete.py`.

This does not validate learned ETT motion, observational identification, or the original worst-case loss. No historical artifact was changed. No commit or push was performed.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    artifacts = [
        OUT / "PROTOCOL.md",
        OUT / "preregistration.json",
        OUT / "root_selection.npz",
        OUT / "initialization_checks.json",
        OUT / "primary_native_traces.npz",
        OUT / "primary_model_traces.npz",
        OUT / "primary_results.json",
        OUT / "branch_c_results.json",
        OUT / "branch_c_relevant_rows.npz",
        OUT / "branch_c_actor_root_actions.npz",
        OUT / "results.json",
        OUT / "decision.json",
        OUT / "REPORT.md",
        OUT / "matched_trajectories.png",
        OUT / "root_preferences.png",
    ]
    completion = load(OUT / "completion.json")
    completion.update(
        status="analysis_complete_pending_independent_saved_verification",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        selected_branch="C",
        conditional_branch_a_paths=0,
        conditional_branch_d_updates=0,
        conditional_analysis_source_sha256={
            "branch_c_audit.py": sha256(OUT / "branch_c_audit.py"),
            "finalize_branch_c.py": sha256(OUT / "finalize_branch_c.py"),
        },
        artifact_sha256={path.name: sha256(path) for path in artifacts},
    )
    save(OUT / "completion.json", completion)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
