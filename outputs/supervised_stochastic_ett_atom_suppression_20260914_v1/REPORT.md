# Saved-array raw-atom response-suppression diagnostic

Date: 2026-09-14. Target commit: `54b36620ddd6818ad91679ec5f587ff9f59e2474`.

## Answer

Moving raw stationary-atom draws explains a large, but not complete, share of B's already-dead distributional error. On the 1,194 already-dead off-diagonal rows from nine episodes, replacing only saved raw-atom draws by current XY lowers energy score from **0.14499 to 0.08538** for B seed 0 (change -0.05962, paired episode-bootstrap 95% interval [-0.08405, -0.03898]) and from **0.12866 to 0.08401** for seed 1 (-0.04465 [-0.05802, -0.03283]). These are reductions of **41.12%** and **34.70%** of the original B score. Raw atoms comprise 74.65% and 74.39% of these samples; all of them were moved by the original response, so the emitted stationary fraction changes from 0% to those atom fractions. The remaining score near 0.084 and moving non-atom draws show that atom motion is not the whole already-dead error.

The intervention is not harmless in alive contexts. All 2,337 alive off-diagonal queried outcomes in the saved development set move at the preserved `1e-7` threshold. Their energy score worsens from 0.17152 to 0.17392 for seed 0 (+0.00239 [+0.00090, +0.00413]) and from 0.17060 to 0.17288 for seed 1 (+0.00227 [+0.00085, +0.00392]). Fatal onset is more sensitive: on 62 rows from 23 episodes, score worsens from 0.29758 to 0.32636 (+0.02878 [+0.00768, +0.04513]) and from 0.25540 to 0.28362 (+0.02822 [+0.00885, +0.04352]). Thus blanket suppression repairs much of the absorption error while damaging legitimate action consequences, mildly in aggregate alive motion and materially at fatal onset.

This is a diagnostic intervention, not a death rule. It uses only the saved raw atom flag and never uses a death or outcome label to select a sample.

## Construction and verification

The original sampler sets raw displacement exactly to zero on its stationary atom. Geometry projection therefore leaves its anchor at current XY. The original emitter computes one response vector per row from the gate-weighted matrix and `action - observational_action`, broadcasts that response over every anchor draw, and then projects. Replacing emitted XY by current XY on saved atom-flagged draws exactly suppresses this response on those known raw-atom anchors. It neither infers nor changes any non-atom anchor.

Both B evaluation files align exactly to the saved 4,800 development rows (episodes 72--95) and contain Boolean atom arrays of shape `[4800, 64]`. The original four-row `(episode,time)` pairing and query-0 diagonal row were verified for all 1,200 development contexts. Recomputing the full per-row energy-score U-statistic, with all 64-by-64 sample-pair terms and denominator `2*K*(K-1)`, reproduces every saved B score exactly in both files. Saved mean, squared error, stationarity, sample legality, and mean legality also reconstruct exactly. On diagonal atom draws, emitted XY equals current XY exactly; within each row all atom outputs are identical, confirming the row-level response behavior. Source and input hashes match the original provenance and stayed unchanged.

The original exact two-coordinate action equality defines diagonal rows. Actual stationarity, emitted stationarity, and group membership use Euclidean XY displacement at the original `1e-7` threshold. Native legality uses inclusive outer bounds and the original clipped floor-cell convention. Bootstrap differences are modified minus original and use 2,000 paired resamples of the same 24 whole development episodes.

## Requested strata

Except for the final diagonal stratum, outcome strata below are off-diagonal. Values in paired brackets are `[seed 0; seed 1]`.

- **All off-diagonal:** 3,531 rows, 24 episodes; actual outcomes are 33.81% stationary and 66.19% moving. Energy score is `[0.16255 -> 0.14398; 0.15642 -> 0.14283]`, with changes `[-0.01858 (-0.03302, -0.00644); -0.01359 (-0.02363, -0.00475)]`. Emitted stationary/moving fractions change from `[0/100%; 0/100%]` to `[25.79/74.21%; 25.70/74.30%]`, and mean sample displacement changes from `[0.29457 to 0.27097; 0.29390 to 0.27413]`. Mean-XY RMSE is `[0.25131 -> 0.25077; 0.24459 -> 0.24656]`; the changes' intervals are `[-0.01169, +0.00934]` and `[-0.00688, +0.00975]`. Mean Euclidean errors are `[0.18281 -> 0.17774; 0.18113 -> 0.18189]`, with change intervals `[-0.01714, +0.00468]` and `[-0.00530, +0.00660]`.

- **Already dead:** 1,194 rows, nine episodes; actual outcomes are 100% stationary. Energy score changes are the large improvements reported above. Emitted stationary/moving fractions change from `[0/100%; 0/100%]` to `[74.65/25.35%; 74.39/25.61%]`, and mean sample displacement changes from `[0.22536 to 0.16493; 0.22903 to 0.17987]`. Mean-XY RMSE is `[0.28321 -> 0.27197; 0.27346 -> 0.26930]`, but the paired change intervals `[-0.04819, +0.01325]` and `[-0.03347, +0.01547]` include zero. Mean Euclidean errors are `[0.18657 -> 0.16433; 0.18092 -> 0.17614]`, also with intervals crossing zero.

- **Alive with stationary actual queried outcomes:** zero rows and zero episodes. No score or bootstrap interval is estimable. The group was not broadened by changing the threshold or treating actions as WAIT.

- **Alive with moving actual queried outcomes:** 2,337 rows, 24 episodes; actual outcomes are 100% moving. Energy score worsens as reported above. Emitted stationary/moving fractions change from `[0/100%; 0/100%]` to `[0.831/99.169%; 0.820/99.180%]`, and mean sample displacement changes from `[0.32993 to 0.32514; 0.32704 to 0.32229]`. Mean-XY RMSE worsens from `[0.23333 to 0.23921; 0.22844 to 0.23409]`, with paired change intervals `[+0.00222, +0.01000]` and `[+0.00212, +0.00960]`. Mean Euclidean error likewise worsens from `[0.18089 to 0.18460; 0.18124 to 0.18482]`, with intervals `[+0.00143, +0.00633]` and `[+0.00141, +0.00609]`.

- **Alive contexts with stationary recorded diagonal outcome and moving paired off-diagonal outcome:** zero rows and zero episodes. Every alive recorded query-0 diagonal outcome moves at `1e-7`; this exact paired contrast is therefore unavailable in these development episodes.

- **Fatal onset:** 62 rows, 23 episodes; actual outcomes are 100% moving. Energy-score degradation is reported above. Emitted stationary/moving fractions change from `[0/100%; 0/100%]` to `[6.804/93.196%; 6.754/93.246%]`, and mean sample displacement changes from `[0.65280 to 0.59528; 0.65090 to 0.59308]`. Mean-XY RMSE worsens from `[0.42597 to 0.46314; 0.36437 to 0.40600]`, with paired change intervals `[+0.01021, +0.07262]` and `[+0.01287, +0.07581]`. Mean Euclidean error worsens from `[0.30936 to 0.35528; 0.27865 to 0.32386]`.

- **Diagonal:** 1,269 rows, 24 episodes; actual outcomes are 37.04% stationary and 62.96% moving, illustrating why neither diagonal nor stationarity can be equated with WAIT or death. The diagnostic changes nothing: energy score remains `[0.03713; 0.03854]`, emitted stationary/moving fractions remain `[30.34/69.66%; 29.43/70.57%]`, mean sample displacement remains `[0.28820; 0.29588]`, and mean-XY RMSE remains `[0.17254; 0.17403]`, all with exactly zero paired differences.

Mean emitted-sample displacement and every bootstrap contrast are saved in [results.json](results.json). All emitted samples remain native-legal in both variants. Mean legality is separate: for seed 0, 3/3,531 off-diagonal means remain illegal before and after; for seed 1 it changes from 2/3,531 to 3/3,531. At fatal onset, seed-1 illegal means change from 1/62 to 2/62 even though every constituent sample is legal. Already-dead and diagonal means remain fully legal.

## Response conditioning

The eight gates see only horizontal position and the F4 motion summary. Observed stationary off-diagonal outcomes are exactly the 1,194 already-dead rows here; their horizontal positions span 3.482--5.477 and their motion summary spans 0--0.999, with median zero. Alive moving rows span 0.5--8.949 and 0--1.049, with medians 8.481 and 0.293. The supports therefore overlap but most rows are not local neighbors: using the gate's native scales (`x/2`, `motion/0.35`), 4.94% of stationary rows have an alive-moving neighbor within 0.25, and 7.36% of alive-moving rows have a stationary neighbor within 0.25; the minimum distance is 0.00176. The closest gate-weight vectors are nearly identical (minimum L2 distance 0.000214).

This is evidence that a minority of stationary and moving cases are difficult to distinguish for this response component, while most are separated in its two-feature gate space. It is not proof that the full model family cannot fit the conditional law. There are no alive-stationary rows or mixed stationary/moving alive contexts in this split with which to make a stronger same-context claim.

## What is established, unresolved, and next

The diagnostic establishes that moving raw atoms accounts for roughly one third to two fifths of B's already-dead energy score and that unconditional atom-response suppression has a statistically detectable cost on legitimate alive motion, especially fatal onset. It does not identify a valid death detector, justify treating stationarity as WAIT/death, or show that non-atom response should be retained or removed.

The main unresolved question is how much of the remaining already-dead score and alive tradeoff comes from applying the response to non-atom anchors. Those anchors are absent from the saved arrays and were not inferred by reversing projection. A same-head, zero-response comparison would require exact replay of each B checkpoint with its own diagonal-head parameters and the original evaluation keys; A is not a substitute because its diagonal head differs.

**Recommended next intervention:** perform that exact same-head, zero-response checkpoint replay as a narrowly scoped diagnostic before changing the model. It would isolate the full response contribution, including non-atom anchors, and determine whether the next structural experiment should target atom-only response modulation or broader response conditioning. Do not retrain or recollect for that diagnostic. Any chosen model modification must then be confirmed on new complete episodes: the current 24 development episodes remain reused exploratory evidence.

Reproduction: run [analyze.py](analyze.py). Machine-readable outputs are [results.json](results.json), [feature_overlap.json](feature_overlap.json), [verification.json](verification.json), [provenance.json](provenance.json), [per_row_results.npz](per_row_results.npz), and the two saved modified B evaluation arrays. The analysis made zero model-successor calls, training updates, native steps, or trajectories.
