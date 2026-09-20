# Step 3a with the v3 one-step ETT (regression on all rows + relative error term; selection without the stationary BCE)

Same sealed protocol and thresholds as `../ett_rollout/` (`manifest.json`);
the v3 models `../ett_one_step_v3/` (preflight: all gates passed, every
fold trained to 39-40k, no early stop).  `REPORT.md` / `report.json`.

## Result: C still fails strata (fewer than v2); A fails as before

Advice C (simulator teacher along the model path; 6,000 held-out anchors):

| | sim | model |
|---|---|---|
| death / reach / timeout (all) | 0.294 / 0.615 / 0.091 | 0.305 / 0.638 / 0.057 |
| KS death time / death x / reach time (all) | | 0.036 / 0.066 / 0.036 |
| start: death / reach / timeout | 0.724 / 0.248 / 0.028 | 0.728 / 0.245 / 0.028 (pass) |
| pre_zone1: timeout / reach | 0.178 / 0.319 | 0.053 / 0.419 (FAIL) |
| zone1: KS death time | | 0.156 (FAIL) |
| between: KS death x | | 0.168 (FAIL) |
| zone2: KS death time / death x | | 0.321 / 0.210 (FAIL) |
| post_zone2 / goal_area | | pass |

Per fold (C): the three folds are now consistent -- each produces some
pre-mouth stalls (pre_zone1 timeouts 0.057 / 0.041 / 0.062 vs the sim's
0.197 / 0.145 / 0.193; v2 had 0.006 / 0.117 / 0.021), death rates 0.30-0.31
vs 0.29-0.30 and death-time KS 0.025-0.052 per fold.  v3 fixed the fold
inconsistency and roughly tripled the model's stall rate, but the stalls
are still ~3x too rare and the in-zone death timing (zone1 / zone2 KS
0.16 / 0.32 pooled over folds) is worse than v2's pooled 0.10 / 0.22.

Advice A (memoryless nominal): death 0.294 -> 0.036, reach 0.615 -> 0.853,
AUROC 0.69 -- as in v1 / v2, the memoryless nominal cannot produce the
in-band holds.

## Reading (user's rule: C bad -> fix motion / onset first)

The remaining C failures are the same two places as before, smaller: the
slow settling into a stall after the torque becomes small (the model
still under-stalls) and the in-zone death timing.  A is not informative
until C passes.  No arm training with these futures.
