# Controlled supervised PointMaze transition repair

## Decision

The onset repair succeeded, but the alive-position repair and the combined repair did not meet the sealed acceptance criteria. The supervised transition reference is therefore **not yet credible enough to proceed to pessimistic optimization**.

The single blocking issue is the alive-position proposal: although it improved broad diagonal and off-diagonal emitted-XY fit in both seeds, it did not improve hazard residence/exit and inside-hazard horizontal displacement together in both seeds, and its complete-rollout effects were seed-inconsistent. A plausible mean return is not sufficient to clear that failure.

This was a bounded, additional-information supervised experiment. It does not establish observational identification, validate the original unlabeled pessimistic objective, or demonstrate actor improvement.

## Integrity and scope

The protocol was sealed before new collection or evaluation. Source and checkpoint dependencies were pinned by SHA-256 at commit `88ec819cd4dde584976ae71a0aa022f6aeadc2b4`; HEAD equality was not required. No historical result or production module was modified, and no old audit, critic, failure-bank, atom-suppression, MC-precision, spectral-norm, or actor-training experiment was rerun.

All eight arms use the same additional native knowledge: onset probability is exactly zero unless `3 <= next_x < 6` and `3 <= next_y < 4`. They also use the same event order and absorption law: propose position while alive, retain fatal incoming motion, make failure irreversible, then freeze later XY while shifting F4 and returning zero reward. The actor and historical nominal policy stayed frozen and received no hidden failure state or labels. Thus the old-head controls here are supported controls, not the historical unmasked model.

New data used complete episodes as split units:

- Train: 128 episodes, 64 expert-prefix and 64 actor-prefix; 25,600 paired rows, 16,288 alive-before rows, 327 onsets, and 1,486 actual hazardous landings.
- Validation: 32 episodes, 16 expert-prefix and 16 actor-prefix; 6,400 paired rows, 3,528 alive-before rows, and 94 onsets.
- Final: 64 untouched actor-prefix episodes; 12,800 paired rows. Matched evaluation used all 4,864 alive-before rows, including 119 onsets and 486 hazardous actual successors.
- Historical training contribution: zero. The old audit episodes were design evidence only.

At each context, the four saved outcomes are the same-snapshot shadow advice, frozen actor, negative advice, and predeclared uniform-random action. The chosen expert/actor branch exactly equals the saved main prefix transition. Final alive input groups contained 384 approach-left rows from 58 episodes, 396 inside-hazard rows from 38 episodes, 52 right-exit rows from 11 episodes, 4,032 elsewhere rows from 64 episodes, 1,253 exact-diagonal rows, and 3,611 off-diagonal rows. Each declared query type contributed 1,216 rows from all 64 episodes.

The position repair updated all 69 allowed parameters: 16 diagonal offsets, 32 response coordinates, and 21 gate weights. Validation-only Energy Score selection chose step 220 for seed 0 and step 180 for seed 1. The head repair updated all 1,569 MLP parameters with uniformly sampled alive rows and supported BCE; validation-only selection chose steps 1,300 and 1,500. Checkpoints were frozen before collecting final outcomes.

The ledger recorded exactly 56,000 native steps, 21,547,520 position successors, 4,264,560 head rows, 3,480 training updates, and 32,768 complete model paths, all within the sealed caps. Uncertainty below is a 2,000-replicate paired whole-episode bootstrap over the 64 final episodes. The 64 model paths per root are predictive integration, not extra native evidence.

## 1. Onset head at actual native successors

Yes. The repaired head improved final actual-successor Brier score and absolute calibration bias overall and on actual hazardous landings in both seeds.

| Seed and group | Rows / episodes | Observed | Predicted old -> repaired | Bias old -> repaired | Brier old -> repaired | Paired Brier change, 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| s0, all | 4,864 / 64 | 0.02447 | 0.03406 -> 0.02731 | +0.00959 -> +0.00284 | 0.01986 -> 0.01509 | -0.00478 [-0.00779, -0.00202] |
| s0, hazardous successor | 486 / 58 | 0.24486 | 0.34087 -> 0.27332 | +0.09602 -> +0.02846 | 0.19880 -> 0.15098 | -0.04782 [-0.07255, -0.02136] |
| s1, all | 4,864 / 64 | 0.02447 | 0.03111 -> 0.02557 | +0.00664 -> +0.00111 | 0.01836 -> 0.01364 | -0.00472 [-0.00853, -0.00133] |
| s1, hazardous successor | 486 / 58 | 0.24486 | 0.31132 -> 0.25596 | +0.06646 -> +0.01110 | 0.18377 -> 0.13653 | -0.04724 [-0.08067, -0.01372] |

This was not merely an effect of zeroing impossible nonhazard predictions: the hazardous-successor subset is entirely inside the support and improved strongly. Nor did aggregation hide an approach/inside reversal. In seed 0, approach bias fell from +0.03992 to +0.01246 and inside-hazard bias from +0.08051 to +0.02290. In seed 1 they changed from +0.00924 to -0.00401 and from +0.07340 to +0.01649, respectively. All probabilities on the 4,378 nonhazardous actual successors were structurally zero in both old and repaired controls.

## 2. Alive-position proposal

Only partially. Full emitted-XY Energy Score improved on both diagonal and off-diagonal rows in both seeds, with all sample-pair terms included in the U-statistic. The repair did not, however, jointly repair the intended hazard residence/exit and horizontal-motion defects.

| Seed | Diagonal Energy Score old -> repaired; paired change [95% CI] | Off-diagonal Energy Score old -> repaired; paired change [95% CI] | Hazard residence observed; predicted old -> repaired | Inside-hazard dx actual; predicted old -> repaired | Result |
|---|---|---|---|---|---|
| s0 | 0.09441 -> 0.08849; -0.00592 [-0.01129, -0.00160] | 0.30755 -> 0.22745; -0.08009 [-0.09132, -0.06856] | 0.76263; 0.78918 -> 0.79534 | 0.09709; 0.08332 -> 0.10730 | residence error worsened; dx absolute error improved |
| s1 | 0.11134 -> 0.10838; -0.00296 [-0.00576, -0.00024] | 0.27558 -> 0.22835; -0.04723 [-0.05806, -0.03674] | 0.76263; 0.82311 -> 0.80717 | 0.09709; 0.13512 -> 0.14021 | residence error improved; dx absolute error worsened |

The residence-probability change was +0.00616 [-0.00809, +0.02125] in seed 0 and -0.01594 [-0.04854, +0.00984] in seed 1. Because exit is its complement on current-hazard rows, both repaired models still underpredicted exit. The inside-hazard dx change was +0.02398 [+0.00485, +0.04820] in seed 0 and +0.00509 [-0.02985, +0.03972] in seed 1. Hazard-entry bias remained small and within the sealed tolerance: -0.00403 -> -0.00328 for seed 0 and -0.00534 -> -0.00501 for seed 1.

The failure is not evidence that all ETT position families are incapable of fitting the data. It is a failure of this one bounded 69-parameter repair and its sealed Energy-Score-only checkpoint rule. In a post-result inspection of the already-saved validation curve, seed 0's selected step 220 missed the residence/dx acceptance components even though step 200 satisfied all position components on validation; seed 1's selected step 180 satisfied them on validation but its dx improvement did not transfer to final data. No checkpoint was reselected after final inspection.

## 3. Joint matched forecasts

The joint forecast averages each head over 64 generated successors and is grouped only by pre-transition inputs. The actual successor's hazard label was not used to condition this marginalized forecast.

The observed onset rate was 0.02447. For seed 0, old-position/old-head predicted 0.03148 (bias +0.00702, Brier 0.01893), old-position/repaired-head 0.02466 (+0.00020, 0.01445), repaired-position/old-head 0.03320 (+0.00874, 0.01965), and repaired/repaired 0.02626 (+0.00179, 0.01496). For seed 1 the same sequence was 0.03007 (+0.00560, 0.01844), 0.02483 (+0.00036, 0.01382), 0.03007 (+0.00561, 0.01788), and 0.02502 (+0.00055, 0.01338).

At fixed old position, head repair reduced joint Brier by -0.00448 [-0.00751, -0.00164] in seed 0 and -0.00462 [-0.00858, -0.00123] in seed 1. Position repair at fixed old head changed it by +0.00072 [-0.00064, +0.00214] and -0.00056 [-0.00157, +0.00023], respectively. The combined Brier changes were -0.00398 [-0.00686, -0.00123] and -0.00506 [-0.00910, -0.00162]. Thus matched joint improvement is attributable primarily to the repaired head, not to a consistently repaired position component. Every nonhazard component was exactly zero by construction and verification.

## 4. Complete trajectories

`OO`, `OH`, `RO`, and `RR` denote old position/old head, old position/repaired head, repaired position/old head, and repaired/repaired. Point estimates are below; all arm-level 95% episode-bootstrap intervals are saved in `trajectory_metrics.json`.

| Seed / arm | Reward occurrence | Failure | Survival without reward | Mean discounted return | Return given reward | Onset time given failure | First hazard: probability / time / mean XY | At-risk hazard opportunities |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Native | 0.2969 | 0.7031 | 0.0000 | 3.136 | 10.564 | 4.911 | 0.8594 / 3.255 / (3.457, 3.410) | 2.250 |
| s0 OO | 0.2498 | 0.7405 | 0.0198 | 2.918 | 11.682 | 5.081 | 0.9795 / 2.825 / (3.363, 3.492) | 3.040 |
| s0 OH | 0.2896 | 0.7014 | 0.0198 | 3.390 | 11.708 | 5.049 | 0.9795 / 2.825 / (3.363, 3.492) | 3.058 |
| s0 RO | 0.2361 | 0.7708 | 0.0012 | 2.696 | 11.421 | 4.836 | 0.9810 / 2.913 / (3.383, 3.528) | 2.764 |
| s0 RR | 0.2922 | 0.7153 | 0.0012 | 3.358 | 11.492 | 4.891 | 0.9810 / 2.913 / (3.383, 3.528) | 2.878 |
| s1 OO | 0.3032 | 0.7017 | 0.0032 | 3.446 | 11.366 | 5.653 | 0.9844 / 2.908 / (3.365, 3.514) | 3.590 |
| s1 OH | 0.3123 | 0.6968 | 0.0015 | 3.594 | 11.510 | 5.340 | 0.9844 / 2.908 / (3.365, 3.514) | 3.137 |
| s1 RO | 0.2683 | 0.7278 | 0.0110 | 3.112 | 11.598 | 5.629 | 0.9895 / 3.110 / (3.398, 3.523) | 3.374 |
| s1 RR | 0.2891 | 0.7112 | 0.0090 | 3.379 | 11.689 | 5.397 | 0.9895 / 3.110 / (3.398, 3.523) | 2.996 |

Native 95% intervals were [0.1875, 0.4063] for reward occurrence, [0.5938, 0.8125] for failure, [1.927, 4.420] for mean return, [9.282, 11.607] for return given reward, [3.690, 6.596] for onset time, [0.7656, 0.9375] for first-hazard probability, [2.400, 4.661] for first-hazard time, and [1.797, 2.766] for at-risk opportunities. These wide native intervals are why model-path precision must not be mistaken for additional native evidence.

The paired factorial effects expose cancellation:

- Head repair at old position changed reward/failure by +0.03979 [+0.03223, +0.04736] / -0.03906 [-0.04688, -0.03125] in seed 0, but only +0.00903 [-0.00024, +0.01831] / -0.00488 [-0.01318, +0.00342] in seed 1.
- Position repair at old head changed reward/failure by -0.01367 [-0.02344, -0.00366] / +0.03027 [+0.02026, +0.04005] in seed 0 and -0.03491 [-0.04370, -0.02686] / +0.02612 [+0.01782, +0.03492] in seed 1. It harmed these two trajectory components in both seeds.
- The combined repair changed reward/failure relative to supported OO by +0.04248 [+0.03296, +0.05249] / -0.02515 [-0.03516, -0.01537] in seed 0, improving both absolute native gaps. In seed 1 it changed them by -0.01416 [-0.02637, -0.00244] / +0.00952 [-0.00049, +0.01978], worsening both absolute native gaps.
- Reward/failure interactions were +0.01636 [+0.00562, +0.02613] / -0.01636 [-0.02661, -0.00562] for seed 0 and +0.01172 [+0.00513, +0.01832] / -0.01172 [-0.01929, -0.00415] for seed 1. The component effects are therefore not simply additive.

Both RR mean returns, 3.358 and 3.379, look close to native 3.136. That agreement cannot pass: in seed 0 it is produced while the head offsets a position repair that independently lowers reward and raises failure, and in seed 1 reward/failure move in the wrong direction. First-hazard probability also remains 0.981-0.990 versus native 0.859, and at-risk exposure remains 2.878-2.996 versus native 2.250. The combined repair therefore did not improve fresh full-episode behavior without error cancellation.

## What is established and what is not

Established:

- The existing conditional head family is capable of substantially better actor-context onset calibration when trained with actor-inclusive matched coverage, supported BCE, and the native hazardous-landing support. Improvement inside the hazardous subset rules out support masking as the sole explanation.
- Broad position distribution error is reducible within the existing 69-parameter family: diagonal and especially off-diagonal Energy Score improved in both seeds.
- The remaining factorial differences cannot be attributed to unequal support rules, resurrection, discarded fatal movement, post-failure motion, actor retraining, final-set checkpoint selection, or return-target tuning. Saved-array verification directly checked these contracts.
- Position changes, not the repaired head alone, are responsible for the seed-inconsistent autonomous behavior and persistent excessive hazardous exposure.

Unresolved:

- Whether a prospectively constrained checkpoint rule within this same position family can make residence/exit and corridor displacement generalize together. The saved validation curves show that the final failure is not proof of a universal representational impossibility.
- Whether the remaining autonomous mismatch comes primarily from generated-state compounding, the nominal-action mixture, or the position response being shared across anchor draws. Matched real-state results cannot distinguish these mechanisms.
- Observational identification, the original unlabeled pessimistic objective, and actor benefit remain untested.

## Recommended next intervention

Run one new, preregistered position-only experiment with the same architecture and training schedule, but make checkpoint eligibility require validation improvement in diagonal and off-diagonal Energy Score **and** non-worsening hazard residence/exit and inside-hazard dx; select the lowest Energy Score only among eligible checkpoints. Save every validation checkpoint so the rule is auditable. Keep the now-credible repaired head frozen. This is the smallest intervention supported by the saved curves: seed 0 had an eligible validation checkpoint at step 200 instead of the selected step 220, while seed 1's selected step was already eligible. It must be confirmed on new complete actor episodes; the current final episodes cannot be reused for selection or confirmation. If that prospective test still fails, anchor-conditioned/per-draw response modeling becomes the next hypothesis, not a conclusion from this experiment.

## Reproducibility map

- `PROTOCOL.md`, `config.json`, `provenance.json`, and `checkpoints_frozen.json`: sealed design and dependency/checkpoint hashes.
- `collect_split.py`, `train_repairs.py`, `evaluate.py`, and `run.py`: complete collection, training, selection, and evaluation code.
- `train_paired.npz`, `validation_paired.npz`, `final_paired.npz`, and `split_manifests.json`: reusable complete-episode paired data and manifests.
- `position_training_s*.json`, `position_validation_s*.json`, `head_training_s*.json`, and `head_validation_s*.json`: training and validation traces.
- `final_actual_head_metrics.json`, `final_position_metrics.json`, `final_joint_metrics.json`, and `trajectory_metrics.json`: full numerical results and bootstrap intervals.
- `final_*predictions.npz`, `final_position_*.npz`, and `rollout_*.npz`: reusable matched predictions, samples, and all eight trajectory arrays.
- `contract_checks.json`, `ledger.json`, `acceptance.json`, `verify_saved.py`, and `verification.json`: semantics, accounting, frozen decision, and independent saved-array verification.

`verification.json` passed every listed check using saved arrays only and charged zero new native steps, model samples, or updates. No commit or push was performed.
