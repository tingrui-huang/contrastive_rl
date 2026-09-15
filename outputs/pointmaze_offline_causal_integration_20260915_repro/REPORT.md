# Strictly offline PointMaze causal-contrastive integration

## Result

This bounded experiment restored the requested sampling pipeline without changing the production sigmoid-NCE loss or actor objective. It used only the fixed original offline dataset and eligible offline-trained checkpoints. No environment, simulator, native trajectory, oracle transition, hidden state, audit label, failure bank, ranking target, or return-regression target entered the run.

The historical optimized ETT checkpoint was not reused because its optimization ancestry included native-collected contexts. The eligible lambda-zero diagonal law, nominal model, and frozen original actor instead seeded a fresh three-update ETT fit on 96 offline roots. 2 of 3 proposed ETT updates passed the fixed diagonal guard; the final offset norm was 0.160771. The run emitted 323,433 model transitions under the sealed 480,000 cap.

## 1. Did the eligible pessimistic ETT change conditional futures?

Yes descriptively: under common validation anchors and random streams, 100.00% of 33,710 valid future-state rows changed in XY, with mean paired displacement 0.2567. The all-future task-region fraction changed from 78.04% to 74.00%; the paired episode-bootstrap change was -4.03 (95% CI -5.45, -2.76) percentage points.

The final validation paths reached the geometric task region on 84.77% of paths and had raw atom frequency 13.20%. There were 4,321 atom events followed by later motion. These atom statistics are not failure statistics.

Irreversible-failure frequency and post-failure consistency are **unsupported**. The eligible model has no learned failure-entry variable, and the original offline observations do not identify off-diagonal death. The user-provided absorption assumption says what should happen after a genuine failure; it does not supply the probability of entering one. No stationary outcome was relabeled as death.

## 2. Did the actual NCE batches preserve the change?

Yes. O and P used the same 102,400 production rows per arm, the same recorded anchor/action identities, and the same strictly-later discounted offsets. P sourced 100% of positive goals from the complete final-ETT cache; O sourced 100% from the matching recorded continuation. Across all rows, 100.00% of paired goals changed in XY (mean L2 1.8489). The sampled task-region fraction was O 48.46% versus P 64.12%; P-O was 15.66 (95% CI 14.39, 16.91) percentage points.

Fork rows were 51,200/102,400 (50.00%), supported by 1,659 source episodes and 2,048 cached anchor identities. Their task-region fraction was O 28.92% versus P 51.14%; P-O was 22.21 (95% CI 20.30, 24.01) percentage points. Each production batch contained exactly 128 ordinary, 64 down-action fork, and 64 right-action fork rows. Cache duplication and supporting-episode counts are in coverage.json.

These are achieved future-goal marginals, not returns. A complete path reaching the task region, a discounted sampled future in that region, and a canonical-goal critic score remain separate quantities.

## 3. Did the unchanged critic learn a different task preference?

At 656 held-out first observable fork contexts, the endpoint down-minus-right canonical-goal gap was initially -0.6029 (95% CI -0.6693, -0.5420), after O -0.3051 (95% CI -0.3614, -0.2520), and after P -0.0876 (95% CI -0.0937, -0.0814). The paired P-O endpoint-gap change was 0.2175 (95% CI 0.1671, 0.2726). This is evidence of a treatment-specific score-preference change under the episode bootstrap; it is not a calibrated return difference.

On common fixed held-out batches, P-minus-O NCE loss was 0.0023 (95% CI 0.0023, 0.0024) for observational goals and -0.0078 (95% CI -0.0084, -0.0072) for ETT goals. These intervals resample fixed evaluation batches because all-pairs negatives make an individual row loss non-separable. Local action derivatives at actual actor samples are saved in offline_evaluation_arrays.npz and summarized in results.json.

## 4. Did the unchanged actor objective translate the preference?

At the same canonical-goal fork contexts, initial/O/P downward probabilities were 4.85%, 7.20%, and 8.73%; rightward probabilities were 84.65%, 85.10%, and 76.91%. Paired P-O changes were 1.53 (95% CI 1.29, 1.75) percentage points down and -8.19 (95% CI -8.52, -7.86) percentage points right. The continuous P-O mean-action shifts were dx -0.1060 (95% CI -0.1104, -0.1019) and dy 0.1537 (95% CI 0.1515, 0.1560).

Both arms applied all 1,000 actor updates and moved from the common initial actor (parameter L2 deltas O 1.4386, P 2.1386). The mean raw BC/critic gradient cosines at the final actors were O -0.5250 and P -0.5437. Non-fork ordinary-goal retention, local critic gradients, policy scales, continuous samples, and optimizer-applied changes are all retained in the numerical outputs. Any action-distribution change is only a strictly offline policy response, not measured native route entry or success.

## What is established and what is unresolved

Established: an eligible-data-only ETT optimization can be connected to complete cached continuations; its future goals can replace all production positives at exactly matched offline anchors; and the unchanged critic/actor response can be measured end to end with full-F4 lineage. All sealed lineage, frozen-tree, padding, first-action, update-count, hash, and static no-environment checks passed (87 checks).

Unresolved: the ETT is not a certified worst-case transition kernel; native success and true counterfactual outcomes were intentionally not measured; the initial production checkpoint had already seen all 6,600 episodes; and no eligible evidence supplies an irreversible-failure entry probability. This one-seed bounded result is exploratory evidence, not causal identification or a lower bound.

## Recommended next intervention

Run one ETT-only controlled repair before any more production learning: fit an observable full-F4 multi-step persistence/response component from eligible original offline suffixes, then require this same matched-anchor sampler audit to show that fork futures are no longer more favorable. Do not label stationary rows as death.

## Reproduction

Run from the PointMaze worktree with a fresh output directory:

```powershell
python -m pytest -q scripts/test_pointmaze_offline_causal_integration.py
python -m ett.pointmaze_offline_causal_integration all --out outputs/pointmaze_offline_causal_integration_20260915_v1
```

The final checkpoints are fixed iterates, not validation-selected checkpoints. No commit or push is part of this experiment.
