# G2b: AWR on recorded actions — the sealed naive-landing criterion is not met; the physical landing shows β=5 turns the P actor down (0.555) with retention intact

> **STATUS (2026-09-15): AWR withdrawn as an actor objective (not in the CRL paper; the discrete WindyCorridor repo dropped it for the same reason). Retained as a diagnostic only.** The faithful candidate is the paper objective with bc = 0.2 on the original actor rows; see `outputs/pointmaze_absorbing_faithful_actor_audit_20260915_v1/REPORT.md`.

Date: 2026-09-15. Base G1 `outputs/pointmaze_absorbing_integration_20260915_v1` (frozen critics, plan, batches, keys, 656 roots, canonical goal; hash-pinned in [config.json](config.json)); G2 reference `outputs/pointmaze_absorbing_actor_sweep_20260915_v1`. Sealed [PROTOCOL.md](PROTOCOL.md); driver [run.py](run.py); [results.json](results.json); per-actor audits `actor_audit_{arm}_{beta}.json`; [weight_diagnostics.json](weight_diagnostics.json); [advantage_summary.json](advantage_summary.json) and `advantages_{arm}.npz`; recorded-action bank `bank.npz`; checkpoints, curves, per-root arrays; post-run addenda [addendum_physical_landing.json](addendum_physical_landing.json). Strictly offline: 10 actor runs × 1,000 updates, zero critic updates, zero environment calls; 19,443,200 model-rollout transitions (cap 30,000,000); 282 s.

## Decision under the sealed rule

**No β passes the sealed primary criterion** (P first-step landing in (1,2) ≥ 0.30 with landing = `floor(clip(s + a))`): 0.035 / 0.040 / 0.036 / 0.043 / **0.226** for β = 0 / 0.5 / 1 / 2 / 5. `selected_beta = null`.

The sealed landing bin, however, mis-measures the quantity it was meant to capture. Most of the P actor's first-step samples at β=5 are down-right diagonals whose naive cell `floor(s + a)` is the wall cell (2,2) (naive "wall" share 0.644), but under the maze's kinematics (ten substeps, X then Y, blocked coordinate updates rejected — the same map already inside the diagonal model's projection; no noise, no environment call) the X update is rejected once the point is below y = 3 and the move ends in (1,2). Resolving every sample this way (addendum, computed after the run):

| β | P physical (1,2) | P physical (2,3) | O physical (1,2) | P naive (1,2) | P naive wall |
|---:|---:|---:|---:|---:|---:|
| 0 (BC control) | 0.163 | 0.690 | 0.163 | 0.035 | 0.592 |
| 0.5 | 0.156 | 0.660 | 0.128 | 0.040 | 0.575 |
| 1 | 0.146 | 0.677 | 0.098 | 0.036 | 0.583 |
| 2 | 0.233 | 0.639 | 0.117 | 0.043 | 0.659 |
| **5** | **0.555** | 0.239 | 0.149 | 0.226 | 0.644 |

Under the physical landing, β=5 is a strong pass (≥ 0.40), β=2 is below the hard line, and the O control never rises above its BC value (0.163). Because this definition was not the sealed one, β=5 is reported as a **provisional candidate**, not a selection: it needs one preregistered confirmation with the physical-landing criterion fixed in advance (or G3 native episodes, which measure the route directly).

## Retention, secondary criteria and weights at β=5 (P arm)

All sealed retention conditions hold: away landing-bin agreement 0.829 (β=0: 0.836; threshold 0.711), away-state A3 reach 0.687 (β=0: 0.669), A1 fork reach 0.835 (β=0: 0.824). Away BC-NLL change vs the initial actor +0.245 [0.156, 0.327] (β=0: +0.024), mode-action L2 change 0.130 (β=0: 0.098), OOD proxy 0, mode saturation 0.103 (β=0: 0.179). Secondary criteria hold: A3 fork absorbed 0.517 < 0.673 (G2 bc 0.5) and A3 reach 0.372 > 0.232. Clipping fraction 0.234 is under the 0.25 limit but close; mean effective sample size 128 of 512 doubled rows; max normalised weight 6.2.

## Sweep tables

Fork roots (656 × 128 sealed innovations, canonical goal); naive landing bins as sealed, mask = sealed direction mask:

| β | arm | naive (1,2) | naive (2,3) | naive wall | mask down | mask-down not-(1,2) | mean action (x, y) | Q at samples | Q down / right probe |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 0 | O=P | 0.035 | 0.292 | 0.592 | 0.090 | 0.061 | (0.739, −0.175) | −6.750 | −5.87 / −6.94 |
| 0.5 | P | 0.040 | 0.285 | 0.575 | 0.098 | 0.064 | (0.678, −0.107) | −6.762 | |
| 1 | P | 0.036 | 0.289 | 0.583 | 0.090 | 0.060 | (0.692, −0.070) | −6.780 | |
| 2 | P | 0.043 | 0.232 | 0.659 | 0.130 | 0.093 | (0.747, −0.324) | −6.665 | |
| 5 | P | 0.226 | 0.053 | 0.644 | 0.698 | 0.480 | (0.328, −0.674) | −6.361 | |
| 5 | O | 0.029 | 0.334 | 0.563 | 0.068 | 0.046 | | | |

(β=0 gives identical O and P actors: uniform weights, same batches, same initial state.) Saturation (any component beyond 0.99) is 0.56–0.66 at every β — AWR never produces the G2 corner collapse; the β=5 P policy has std (0.66, 0.59), i.e. it is a broad unimodal Gaussian shifted toward down-left, not a saturated point.

Model rollouts from the fork roots (A3 = `manski_absorbing`, 32 paths; A1 = `diagonal_motion`, 16 paths; common keys with G2):

| β | arm | A3 reach | A3 absorbed | A3 lower route | A3 return | A1 reach | A1 lower route | A1 step disp. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | O=P | 0.186 | 0.714 | 0.189 | 1.49 | 0.824 | 0.202 | 0.366 |
| 0.5 | P | 0.234 | 0.689 | 0.220 | 1.79 | 0.853 | 0.245 | 0.378 |
| 1 | P | 0.234 | 0.733 | 0.155 | 1.95 | 0.900 | 0.171 | 0.375 |
| 2 | P | 0.353 | 0.646 | 0.250 | 2.63 | 0.957 | 0.262 | 0.370 |
| **5** | **P** | **0.372** | **0.517** | **0.423** | 2.53 | 0.835 | 0.427 | 0.358 |
| 5 | O | 0.185 | 0.825 | 0.093 | 1.49 | 0.899 | 0.093 | 0.357 |
| G2 bc 0.5 | P | 0.232 | 0.673 | 0.224 | | 0.880 | | |

Paired over roots, β=5 P versus β=0: lower-route (naive first-step) +0.191 [0.186, 0.197]; A3 reach +0.187 [0.178, 0.195]; A3 absorbed −0.197 [−0.207, −0.187]; P−O A3 lower-route +0.330 [0.321, 0.338]. The O actor moves the other way with β (A3 lower route 0.189 → 0.093; absorbed 0.714 → 0.825), consistent with its own critic.

Non-fork retention (1,024 away rows, P arm):

| β | away bin agreement | BC-NLL change vs initial | mode L2 change | scale mean | saturation (mode) | A3 reach from away | A3 absorbed |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.836 | +0.02 [−0.02, +0.07] | 0.098 | 0.654 | 0.179 | 0.669 | 0.264 |
| 0.5 | 0.836 | +0.08 [+0.03, +0.13] | 0.105 | 0.678 | 0.110 | 0.677 | 0.260 |
| 1 | 0.834 | +0.09 [+0.03, +0.14] | 0.110 | 0.720 | 0.123 | 0.683 | 0.266 |
| 2 | 0.837 | +0.24 [+0.16, +0.31] | 0.139 | 0.670 | 0.118 | 0.705 | 0.256 |
| 5 | 0.829 | +0.25 [+0.16, +0.33] | 0.130 | 0.623 | 0.103 | 0.687 | 0.256 |

Retention is flat across β — the support constraint does what it was meant to do.

Weights (P arm; O in `weight_diagnostics.json`):

| β | clipping fraction | mean ESS (of 512) | min ESS | mean max weight / batch | weight share of fork rows (row share 0.036) | weight share of fork (1,2) rows (row share 0.0026) | mean w, fork (1,2) rows | mean w, fork (2,3) rows |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.017 | 25 | 19 | 37.5 | 0.012 | 0.0015 | 0.57 | 0.25 |
| 1 | 0.045 | 36 | 25 | 20.0 | 0.014 | 0.0038 | 1.48 | 0.14 |
| 2 | 0.089 | 59 | 41 | 11.3 | 0.027 | 0.0047 | 1.81 | 0.12 |
| 5 | 0.234 | 128 | 103 | 4.7 | 0.033 | 0.0053 | 2.04 | 0.27 |

ESS *rises* with β because clipping at ±5 flattens the largest advantages; at low β a few rows with advantage near +3 (the 95th percentile of all advantages is +3.2) dominate each batch.

## Advantages of recorded actions (fixed bank of 32 recorded actions per current cell)

| arm | all rows mean / std | 5% / 95% | corr. with nominal baseline | fork-cell rows | fork, recorded landing (1,2): n / mean A | fork, landing (2,3): n / mean A | fork, other: n / mean A |
|---|---|---|---:|---:|---|---|---|
| O | 0.48 / 3.24 | −1.88 / +3.23 | 0.73 | 18,658 | 1,322 / **+0.51** | 9,098 / **+0.55** | 8,238 / −0.15 |
| P | 0.47 / 2.93 | −1.41 / +3.19 | 0.73 | 18,658 | 1,322 / **+0.86** | 9,098 / **−0.11** | 8,238 / +0.38 |

The P critic's advantage contrast between recorded down and recorded right actions at the fork is +0.97 (the G1 endpoint gap was +1.07); the O critic gives no contrast. The frozen-nominal baseline gives +1.10 vs +0.02 for P (correlation 0.73 with the bank baseline over all rows). Tables by current cell and by recorded landing cell are in `advantage_summary.json`.

## Diagnosis: why β ≤ 2 did not move the first step, and why β=5 did

1. **Support is thin and polluted at the fork.** Recorded down actions are 7.1% of fork-cell rows (1,322 of 18,658 doubled rows across the 1,000 batches) and 0.26% of all rows; recorded right is 49%; the remaining 44% are wanderings (stay, left, wall-directed) that come mostly from the random-policy episodes. Under the P critic those "other" rows carry a positive advantage too (+0.38: anything but right beats right), so at β ≤ 2 the re-weighted fork target is a broad mixture whose Gaussian fit still points right-down; the weight share of the down rows stays below 0.5% of the batch.
2. **Critic sharpness is not the limit.** The recorded-down advantage is +0.86 against −0.11 for right; with β=5 that is e^{4.9} ≈ 130 (clipped at 148) — enough for the down rows to dominate the fork fit despite their 7% share. The policy's samples reach Q −6.36 at β=5 versus −6.75 at β=0 (probe down −5.87).
3. **The actor is unimodal.** Fitting a bimodal weighted target (down vs right) with one tanh-Gaussian gives a broad in-between policy; that is why the mask counts 0.70 "down" at β=5 while the physical first step is 0.555 down and 0.239 right, and why the multi-step lower-route entry (0.42) is below the first-step entry.

## Recommendation

Treat **β=5 as the provisional G2b candidate**: it is the only actor in G0–G2b whose rollouts detour (A3 lower route 0.42, absorbed 0.52, reach 0.37) with non-fork retention intact and no saturation. Two things must happen before G3:

- Preregister the **physical-landing** definition (substep kinematics, map and box only) as the primary first-step metric, and re-evaluate the saved β=5 checkpoint under it on a fresh sealed innovation set — cheap, no retraining.
- Because 0.234 of rows are clipped at β=5 and the down share of the fit rests on 0.26% of rows, run a small **support-side** variant alongside (same β, same batches): AWR over per-state candidate sets drawn from the frozen eligible nominal (a mixture with a clean down mode) instead of the single recorded action, which should reach the same detour behaviour at lower β with less clipping. This is the R7 lesson ("AWR cloning dataset actions") with a better candidate set, not a new critic or a new transition.

If the physical-landing confirmation holds, G3 = 200 paired native episodes under the β=5 P actor against the β=5 O actor and the β=0 BC control. Nothing here was committed or pushed.
