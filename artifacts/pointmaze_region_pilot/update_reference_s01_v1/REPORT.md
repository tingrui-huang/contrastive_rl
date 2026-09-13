# Controlled ETT updates: averaged-NCE versus MC continuation

The averaged-NCE critic supports pessimistic optimization in this bounded experiment. Both pessimistic arms lower independent full-rollout returns versus diagonal-only and initialization in both seeds. Replacing the critic with MC does not improve the outcome: the critic arm obtains slightly lower returns in both seeds. This is evidence against critic estimation blocking these particular updates, not a general accuracy or optimality guarantee.

Three attempts per arm and seed (18 total); all 48 ETT coordinates are available to the two-loss arms. The diagonal-only loss has exactly zero response gradient. Actor/nominal remain frozen. The critic refreshes for 400 steps per update on current-model paths, then freezes with visitation. MC uses the same one-step surrogate, with continuations under its frozen pre-update kernel. Final evaluation instead applies each final kernel throughout the rollout.

## Independent final returns

Normalized discounted region occupancy; lower is more pessimistic. Each kernel has 64 new rollouts at each of 36 held-out episode roots. Streams are paired across kernels. All listed intervals are 95%.

Initialization mean return: 0.61762.

Seed 0:
- diagonal: mean 0.61919; versus initialization +0.00157; episode CI [+0.00073, +0.00253], MC CI [+0.00040, +0.00275].
- critic: mean 0.59869; versus initialization -0.01893; episode CI [-0.02263, -0.01517], MC CI [-0.02262, -0.01523].
- mc: mean 0.60232; versus initialization -0.01529; episode CI [-0.01713, -0.01324], MC CI [-0.01776, -0.01283].
- critic_minus_diagonal: -0.02050; episode CI [-0.02414, -0.01687], MC CI [-0.02437, -0.01662].
- mc_minus_diagonal: -0.01686; episode CI [-0.01901, -0.01475], MC CI [-0.01960, -0.01413].
- mc_minus_critic: +0.00363; episode CI [+0.00068, +0.00659], MC CI [+0.00075, +0.00652].

Seed 1:
- diagonal: mean 0.61735; versus initialization -0.00027; episode CI [-0.00122, +0.00057], MC CI [-0.00130, +0.00075].
- critic: mean 0.60418; versus initialization -0.01344; episode CI [-0.01650, -0.01013], MC CI [-0.01693, -0.00994].
- mc: mean 0.60577; versus initialization -0.01185; episode CI [-0.01467, -0.00890], MC CI [-0.01534, -0.00835].
- critic_minus_diagonal: -0.01316; episode CI [-0.01620, -0.01000], MC CI [-0.01652, -0.00981].
- mc_minus_diagonal: -0.01158; episode CI [-0.01447, -0.00871], MC CI [-0.01493, -0.00822].
- mc_minus_critic: +0.00159; episode CI [+0.00037, +0.00279], MC CI [+0.00002, +0.00316].

## Accepted updates and diagonal fit

Acceptance uses the unchanged fixed train energy-score guard: initialization + .02. Independent validation uses 512 rows and 32 draws, clustered by source episode. Its intervals include finite-sample fitting noise; it does not select kernels.

- diagonal_s0: AAA (A accepted/R rejected); diagonal degradation +0.000181, CI [+0.00008, +0.00028]; train guard 0.013158; clipped steps diagonal/response [0, 0].
- critic_s0: AAA (A accepted/R rejected); diagonal degradation +0.014969, CI [+0.01395, +0.01598]; train guard 0.026397; clipped steps diagonal/response [3, 3].
- mc_s0: AAA (A accepted/R rejected); diagonal degradation +0.016079, CI [+0.01488, +0.01725]; train guard 0.027582; clipped steps diagonal/response [3, 3].
- diagonal_s1: AAA (A accepted/R rejected); diagonal degradation +0.000249, CI [+0.00012, +0.00038]; train guard 0.012968; clipped steps diagonal/response [0, 0].
- critic_s1: ARR (A accepted/R rejected); diagonal degradation +0.008099, CI [+0.00729, +0.00892]; train guard 0.020327; clipped steps diagonal/response [3, 3].
- mc_s1: ARR (A accepted/R rejected); diagonal degradation +0.004701, CI [+0.00400, +0.00543]; train guard 0.017080; clipped steps diagonal/response [3, 3].

All final train guards and geometry/history/Lipschitz checks passed: True. Maximum action-Lipschitz excess 0; updated diagonal identity drift is exactly zero. Diagonal head offsets can change the diagonal law; identity is checked against each updated head with response zeroed. Validation allowance flags and all gradient/proposal values are retained in JSON.

## Matched first update and simulation cost

- Seed 0: critic signed surrogate differences [-0.02243, +0.12655, -0.09596, +0.09788]; MC [-0.02037, +0.12893, -0.02702, +0.05794]; raw sign agreement 4/4, clipped-step cosine +0.891. These signs have no independent uncertainty screen.
- Seed 1: critic signed surrogate differences [-0.09237, -0.00783, -0.00703, -0.05498]; MC [-0.20053, -0.09161, +0.04287, -0.05332]; raw sign agreement 3/4, clipped-step cosine +0.871. These signs have no independent uncertainty screen.
- diagonal_s0: 77,808 training model transitions; 130,368 final evaluation/audit transitions.
- critic_s0: 80,880 training model transitions; 130,368 final evaluation/audit transitions.
- mc_s0: 369,648 training model transitions; 130,368 final evaluation/audit transitions.
- diagonal_s1: 77,808 training model transitions; 130,368 final evaluation/audit transitions.
- critic_s1: 80,880 training model transitions; 130,368 final evaluation/audit transitions.
- mc_s1: 350,832 training model transitions; 130,368 final evaluation/audit transitions.

Total 1,954,528/3,000,000 new transitions; 2,400/2,400 critic refresh steps; zero native simulation or actor updates. Prior averaged critic checkpoints and Adam states are reused; their historical 1,500 steps per seed are outside this new budget. Detailed costs include shared initial fitting guard and initialization evaluation.

## Interpretation and limits

There is no shared failure to lower returns here. Relative to diagonal-only, the critic reduces normalized return by .02050/.01316 and MC by .01686/.01158 in seeds 0/1; both uncertainty intervals exclude zero for each reduction. MC-minus-critic is +.00363/+.00159, also positive under both declared intervals, though seed 1's conditional-MC lower bound is only +.00002. These are descriptive paired comparisons, not a simultaneous claim of universal critic superiority.

The constraints are active in both pessimistic arms: every proposed diagonal and response block is clipped to its step cap, and both arms reject their last two seed-1 proposals. There are 14 acceptances and four rejections overall. Validation diagonal degradation is positive (.00470 to .01608 for the pessimistic kernels), but all its reported intervals remain below the .02 allowance. Thus the observed return reductions accompany permitted diagonal-fit deterioration; this experiment does not separate the contribution of diagonal-head changes from response changes.

MC removes the learned continuation approximation from the update, but still has rollout noise. Both pessimistic arms share finite-difference noise, one-step visitation approximation, only three update attempts, block step caps, geometry and the diagonal guard. After the first update their visitation and current kernels can differ. A failed or unresolved MC improvement cannot isolate critic error as the cause; small differences cannot establish equivalence. The report tests a model-internal pessimistic objective, not native physical validity or global worst-case recovery.

Evaluation roots were inspected in earlier experiments; all new return samples are independent of updates and excluded from training. Episode intervals are descriptive and not simultaneous or training-seed population guarantees. No historical calibration gate was used. No actor training, additional sweep, retuning, extension or push follows this report.

Reproduce in a fresh directory: `python -m ett.pointmaze_update_reference prepare --out <dir>`, then `python -m ett.pointmaze_update_reference run --out <dir>`. Read-only audit: `python -m ett.check_pointmaze_update_reference --out <dir>`.
