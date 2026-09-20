# The pipeline with v4 learned-ETT futures (motion / stationary on (s, a_q) only), five seeds, 2026-09-20: mean up 0.29 -> 0.35, still unstable; primary rule NOT met

`scripts/exp_v6_ett_futures.py --ett v4` (node3); `manifest.json`, `generation_ett.json`, `REPORT.md` / `report.json`, per run
`train_manifest.json` + `eval_mean_s8909.json` (`ett_futures_v4/CF/seed_*`); the oracle CF / O finals and the start agent
re-used with their seed-8909 evaluations.  Same recipe, same anchors, same draw as ett_futures (v3 futures).

## The v4 futures vs the simulator table (anchor-weighted; v3 futures in brackets)

Pooled success 0.644 [0.633] / sim 0.621, death 0.306 [0.307] / 0.291, timeout 0.050 [0.060] / 0.088, mean rows 123 [128] / 143.
Far-route legs closer to the simulator: west column success 0.779 [0.709] / 0.789, top corridor 0.745 [0.708] / 0.822, east
column 0.905 [0.882] / 0.920.  The pre-mouth under-stall unchanged (pre_zone1 timeouts 0.067 [0.047] / 0.184; between 0.041
[0.056] / 0.091); start region 0.260 / 0.243 success, 0.717 / 0.728 death.

## Result (success / detour / death / timeout; seed 8909, the same 300 episodes)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT, v4 futures | **0.483** / 0.48 / 0.19 / 0.32 | 0.213 / 0.16 / 0.14 / 0.65 | 0.337 / 0.36 / 0.20 / 0.46 | 0.390 / 0.47 / 0.15 / 0.46 | 0.310 / 0.32 / 0.08 / 0.61 | 0.347 |
| ETT, v3 futures | 0.283 | 0.450 | 0.417 | 0.167 | 0.137 | 0.291 |
| CF (simulator futures) | 0.523 | 0.470 | 0.437 | 0.407 | 0.407 | 0.449 |
| O (recorded) | 0.297 | 0.287 | 0.290 | 0.300 | 0.310 | 0.297 |

Paired (success): ETT - O +0.187 / -0.073 / +0.047 / +0.090 / +0.000, mean +0.050, seed s.e. 0.044, 3 / 5 -- rule NOT met
(v3 futures: -0.006, 3 / 5); ETT - CF -0.040 / -0.257 / -0.100 / -0.017 / -0.097 (5 / 5 below); CF - O +0.152 (5 / 5).
Deaths: ETT 0.08-0.20 (O 0.63-0.71); timeouts 0.32-0.65, of which 90-99 % never reach the first mouth (v3: 82-97 %); training
metrics normal (BC NLL -11 to -14, scale 0.075-0.11).

## Reading

* The mean moves toward the oracle (0.291 -> 0.347 of 0.449) and the best seed now matches its oracle counterpart (0.483 vs
  0.523), but the seed pattern shuffled rather than repaired: seeds 0 / 3 / 4 rose (0.28 -> 0.48, 0.17 -> 0.39, 0.14 -> 0.31)
  while seeds 1 / 2 fell (0.45 -> 0.21, 0.42 -> 0.34).  The failure mode is the same stall at the start (timeouts 0.32-0.65,
  not on the detour, before the first mouth).  With the simulator futures the same five seeds never stall (timeouts 0.14-0.29,
  most of them on the detour).  So the model futures still carry something that makes the start decision seed-dependent; the
  v4 revision (no advice in the motion / stationary inputs) removed one identified error and improved the far-route futures,
  which shifted the mean but not the instability.
* Remaining known deviations of the futures from the simulator's: the under-stall before the mouth and in the corridor between
  the zones (the model walks where the simulator's start agent gets stuck), shorter paths (123 vs 143 rows), the far-route
  top corridor still 0.745 vs 0.822, and the zone-2 death timing (rollout KS 0.20).  Which of these drives the stall is not
  established; the far-route reliability, the previous hypothesis, was improved by half without stabilising the policy.
* Judgement: primary NOT met; the learned-ETT pipeline is still not a substitute for the simulator futures.  The oracle line's
  result stands (CF - O +0.152, 5 / 5).
