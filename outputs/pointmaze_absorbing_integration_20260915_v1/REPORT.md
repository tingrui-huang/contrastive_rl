# G1: the sealed O/P pipeline with the absorbing-freeze ETT flips the fork endpoint gap

Date: 2026-09-15. Commit `a52c0a5a42d4b836821e462fb2d5b411732aa316` plus two new uncommitted modules (`ett/absorbing_ett.py`, `ett/pointmaze_absorbing_integration.py`) and one note (`notes/pointmaze_absorbing_integration.md`, sealed here as [PROTOCOL_G1.md](PROTOCOL_G1.md)). Driver: `python -m ett.pointmaze_absorbing_integration all --backend manski_absorbing --out <fresh dir>`. Strictly offline: 231,929 model transitions under the sealed 480,000 cap, zero environment calls, zero native steps, 800 critic and 2,000 actor updates, one seed, fixed final iterates. [verification.json](verification.json): 103 checks passed.

## Decision

**Primary criterion met.** With the P-side transition replaced by the absorbing Manski freeze channel and nothing else changed, the P critic's endpoint down-minus-right canonical-goal gap on the 656 held-out first-fork roots is **+1.071 [1.032, 1.108]** (root bootstrap), versus −0.603 initial, −0.305 after O, and −0.088 for the 09-15 eligible ETT. The paired P−O gap is **+1.376 [1.339, 1.417]** (09-15: +0.217). The four-point neighborhood gap is +1.001 [0.968, 1.034].

**Secondary criterion met.** Right-anchored P continuations are absorbed 0.824 of the time and reach the task region 0.169; down-anchored ones 0.475 and 0.362; among down-anchored paths that actually enter the lower route (43%), absorption is 0.092.

The unchanged actor objective moves in the same direction but stays mostly on the shortcut (down 4.9% → 11.3%, right 84.7% → 75.7%); the BC/critic gradient cosine at the final P actor is −0.86. That is the G2 question, not a G1 failure.

## What was run

1. **Baseline reproduction.** The unmodified 09-15 driver was rerun into `outputs/pointmaze_offline_causal_integration_20260915_repro/` (87 checks passed). Its critic/actor checkpoints, train cache and NCE lineage are **byte-identical** to the committed run, and the endpoint numbers are reproduced to all printed digits (initial −0.602890, O −0.305079, P −0.087598, P−O +0.217481).
2. **G1.** `oci.prepare` (same plan, split, config, provenance, seal), then a `fit` phase that generates the three caches with `ett.absorbing_ett.AbsorbingRollout(mode='manski_absorbing')` instead of the fitted gated-response ETT (no auxiliary critic, no offset updates, `ett_final.npz` is zeros), then `oci.train`, `oci.evaluate` unchanged, then a G1 `verify` that repeats the applicable 09-15 checks and adds freeze-specific ones. The initial validation cache is still the eligible model at zero offsets, so the paired audit compares eligible against absorbing continuations under common random numbers.

Fields read from the dataset: `obs`, `act` only (recorded in [g1_provenance.json](g1_provenance.json) and asserted at verification). Hidden bits, death labels, hazard labels, teacher modes and environment internals were not read; neither new module imports an environment or oracle module.

Freeze tables ([freeze_tables.json](freeze_tables.json)) were estimated on the 5,940 training episodes only; the 660 held-out episodes containing the 656 evaluation roots never enter them. Support `S_abs` = {(3,3), (4,3), (5,3)} with 612/338/361 onsets and absorbing fraction 1.000 each; (0,3) with 340 onsets has absorbing fraction 0.009 and is excluded. Descriptive Manski upper bound at the holding cell: 0.318 on the eligible teacher population, 0.488 on all training behaviours; 0 on both fork branches.

**Diagonal unchanged.** The backend's one-step output for `x = x'` is bit-identical to the eligible Kernel's on the 512 sealed validation diagonal rows with the same key (energy-score change exactly 0.0). The diagonal and nominal parameters were hashed before and after and are unchanged.

## Critic

| critic | endpoint down − right [95% CI] | neighborhood | 09-15 endpoint |
|---|---|---|---:|
| initial (step 150,000) | −0.603 [−0.669, −0.542] | −0.504 | −0.603 |
| O (recorded futures) | −0.305 [−0.361, −0.252] | −0.251 | −0.305 |
| **P (absorbing ETT futures)** | **+1.071 [+1.032, +1.108]** | **+1.001** | −0.088 |
| paired P − O | **+1.376 [+1.339, +1.417]** | | +0.217 |

Held-out NCE loss on common fixed batches: initial 0.02236 (observational goals) / 0.02483 (P goals); O 0.02021 / 0.02286; P 0.02138 / 0.02112. P fits P goals better than O by −0.0017 [−0.0020, −0.0015] and observational goals worse by +0.0012 [+0.0011, +0.0012]. Critic parameter L2 change: O 1.77, P 2.39. Both critic loss curves are finite and decrease (O 0.0204 → 0.0198, P 0.0218 → 0.0206 over the first/last 50 updates).

These are logit-gap probes at the canonical task goal, not calibrated returns.

## Actor (unchanged objective, BC 0.5, 1,000 updates on identical batches)

| actor | down prob. | right prob. | mean action (x, y) | 09-15 down / right |
|---|---:|---:|---|---|
| initial | 0.0485 | 0.8465 | (0.804, −0.156) | 0.0485 / 0.8465 |
| O | 0.0720 | 0.8510 | (0.836, −0.310) | 0.0720 / 0.8510 |
| **P** | **0.1129** | **0.7567** | (0.732, −0.268) | 0.0873 / 0.7691 |

Paired P−O: down +0.041 [+0.039, +0.043], right −0.094 [−0.097, −0.092]. Policy parameter L2 change O 1.44, P 1.84. Non-fork ordinary-goal retention: mode-action L2 change 0.066 (O) / 0.080 (P); BC-NLL change intervals include zero for both. At the final actors the BC and critic gradient components are nearly opposite (cosine O −0.53, P −0.86; critic raw gradient norm 17.2 for P versus 10.8 for O). The critic now pulls down; the 0.5-weighted BC term pulls right.

## P positives and caches

Production NCE rows (400 × 256, identical anchors/offsets in O and P; 100% of P positives from the cache):

| stratum | O task-region | P task-region | P − O [95% CI] | P positive frozen |
|---|---:|---:|---|---:|
| all | 0.485 | 0.386 | −0.099 [−0.110, −0.088] | 0.365 |
| fork (down + right anchors) | 0.289 | 0.130 | −0.159 [−0.179, −0.141] | |
| down anchors | 0.110 | 0.152 | +0.041 [+0.025, +0.057] | |
| right anchors | 0.468 | 0.108 | −0.360 [−0.384, −0.334] | |
| ordinary non-fork | 0.680 | 0.642 | −0.038 [−0.049, −0.028] | |

In the 09-15 run the eligible ETT made fork futures *more* favourable than recorded ones (P − O +0.222 on fork rows); here right-anchored futures become much less favourable and down-anchored ones slightly more.

Training-pool caches (4,096 anchors, 64% of paths under the frozen actor after the recorded first action):

| stratum | P reach | P absorbed | entered `S_abs` alive | Manski onset | atom onset | bin agreement | O reach (recorded) | O ends stationary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.469 | 0.471 | 0.543 | 0.389 | 0.082 | 0.579 | 0.571 | 0.221 |
| ordinary non-fork | 0.673 | 0.292 | 0.315 | 0.145 | 0.147 | 0.737 | 0.705 | 0.206 |
| down anchors | 0.362 | 0.475 | 0.561 | 0.460 | 0.015 | 0.437 | 0.187 | 0.242 |
| right anchors | 0.169 | 0.824 | 0.980 | 0.806 | 0.019 | 0.407 | 0.688 | 0.231 |

Held-out validation pool (1,024 anchors): right reach 0.191 / absorbed 0.805; down reach 0.406 / absorbed 0.449; the paired eligible-versus-absorbing task-region change is −0.591 [−0.640, −0.538] on right anchors and −0.469 [−0.527, −0.411] on down anchors.

Why down-anchored absorption (0.475) is higher than in G0 (0.175): the recorded "down" actions are any `a_y ≤ −0.5, |a_y| ≥ |a_x|`, and only 466/1,024 of them land in the lower cell (1,2); 423 land back in the fork cell. From there the frozen actor, which prefers right 85%, takes the shortcut. Conditional on entering the lower route (444 paths) absorption is 0.092; conditional on not entering it, 0.767. G0 forced exactly `(0, −1)` and had 90% lower-route entry. Right-anchored absorption is 0.824 here versus 0.819 in G0.

All backend caches passed the freeze contracts at verification: frozen paths never move and never receive reward, the frozen flag never reverts, every onset lands in `S_abs`, and every Manski onset occurs on bin disagreement.

## What this establishes and what it does not

Established: replacing only the P transition by a visible-data absorbing freeze channel with a Manski off-diagonal rule makes the **unchanged** sigmoid-NCE critic prefer the detour at the fork on held-out roots, by a margin about 12× the 09-15 P−O change, while leaving the diagonal law bit-identical and reading only `obs`/`act`. The pipeline is deterministic (baseline reproduced byte-for-byte), so the change is attributable to the P caches alone.

Not established: native route entry or success (no environment call); actor extraction (the actor moved 4 points toward down and 9 points away from right, but remains a shortcut policy under BC 0.5); calibration of the freeze channel (it is an upper bound: right survives 0.18–0.19 here versus 0.34 for a blind native shortcut, partly because the diagonal motion lingers in hazard cells); seed robustness (one seed); anything about AntMaze. The alive-motion law assumes the hidden bits act only through freeze.

## Next: G2 (actor extraction)

Keep both G1 critics frozen and run the actor stage with a lower BC weight (and/or AWR-style extraction) applied identically to O and P, on the same 1,000 batches and keys; read the canonical-goal down/right probabilities and non-fork retention. Only after a P actor actually prefers the detour offline does G3 (a preregistered native confirmation, 200 paired episodes) become worth running.

## Files

Sealed: `PROTOCOL.md` (09-15 base), `PROTOCOL_G1.md`, `config.json`, `g1_config.json`, `freeze_tables.json`, `g1_provenance.json`, `seal.json`, `seal_g1.json`, `sampler_plan.npz`, `partition.npz`. Produced: the three trajectory-pool caches with freeze arrays, `checkpoints/`, `nce_row_lineage.npz`, `learning_curves.npz`, `production_training.json`, `sampling_interface_pretraining.json`, `results.json`, `offline_evaluation_arrays.npz`, `model_ledger.json`, `verification.json`. Logs: `outputs/g1_20260915.log`, `outputs/repro_20260915.log`. Nothing committed or pushed.
