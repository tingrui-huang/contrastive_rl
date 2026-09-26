# Round-1 critic diagnostic: memorisation (A) vs new torques (B) vs new-episode states (C)

Sealed manifest: 2026-09-18 14:27:26; reference commit .  Frozen: continuation seed_0 (5d03b2c110f4), critics seed_0 (d695d118e01d), seed_1 (b9a874e445b2), seed_2 (c56051e94b30).  16 fresh paired draws per anchor (seeds ABC_SEED + 100 i + r), one query torque then the frozen policy mode closed-loop; P_goal = gamma 0.999, radius 0.5.  Primary readout = region-integrated exp(f) over the training NCE goal marginal (radius 0.5, per head and deployed min); secondary = exact recorded-goal logit.  Pair classes on the 16-draw log ratio: tie (= 0), weak (<= 0.3), decided (> 0.3); bootstrap-decided = decided and >= 90% of draw-resamples keep the sign.  Agreement = among decided pairs (ties are never counted as wrong); s.e. = episode bootstrap.  Oracle-stage diagnostic; nothing was trained.

## Anchors

| layer | stratum | anchors | episodes | t min / median / max | goal xy mean (std) | candidate torque spread | dist to mode | log pi(a) | frac saturated |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| C | start_early | 32 | 32 | 0 / 3 / 4 | [24.915, 0.754] ([0.322, 0.333]) | 0.71 | - | - | - |
| C | turn | 32 | 30 | 7 / 18 / 88 | [24.753, 0.754] ([0.301, 0.388]) | 1.22 | - | - | - |
| C | north_leg | 32 | 30 | 18 / 50 / 133 | [24.762, 0.74] ([0.308, 0.384]) | 0.76 | - | - | - |
| C | pooled | 96 | 62 | 0 / 18 / 133 | [24.81, 0.749] ([0.319, 0.369]) | 0.90 | - | - | - |

Standardised-state distance: A anchor to its nearest other A anchor, median -; C anchor to its nearest A anchor, median -.  C differs from A in episode, and may differ in pose, goal and time distributions (see the table); B differs from A only in the torque.

## Fresh outcomes by layer and candidate

| key | paths | reach | death | P_goal | went around |
|---|---:|---:|---:|---:|---:|
| C|mode | 1536 | 0.66 | 0.24 | 0.279 | 0.64 |
| C|recorded | 1536 | 0.63 | 0.26 | 0.257 | 0.61 |
| C|sample0 | 1536 | 0.59 | 0.25 | 0.247 | 0.60 |
| C|sample1 | 1536 | 0.61 | 0.26 | 0.255 | 0.61 |
| C|sample2 | 1536 | 0.64 | 0.24 | 0.268 | 0.64 |

## Labels: original records against fresh outcomes at identical keys

| layer | stratum | keys | Spearman(orig mean P, fresh mean P) | mean abs diff | pairs decided by orig | by fresh | both | labels same / opposite | orig-decided but fresh tie/weak |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| A | start_early | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| A | start_late | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| A | shortcut_early | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| A | turn | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| A | north_leg | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| A | pooled | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | start_early | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | start_late | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | shortcut_early | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | turn | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | north_leg | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |
| C | pooled | 0 | - | - | 0 | 0 | 0 | 0 / 0 | 0 |

## Layer A: the critic against the original labels and against fresh labels at the same keys (region min)

| critic | stratum | vs original: same / opposite (agreement) | vs fresh: same / opposite (agreement) | where the critic matched the original label, fresh same / opposite |
|---|---|---|---|---|

## Layer A: familiar state, familiar torque, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer A, the two policy samples only

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer B: familiar state, three new torques, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## A x B cross pairs (shared anchors and draws)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer C: held-out-episode states, existing five candidates, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 24 / 177 / 119 (89) | **0.42** (0.059) 50 / 69 | 0.36 (0.068) | -0.011 (0.015) | -0.016 (0.021) | 0.121 / 0.132 / 0.137 | +0.049 (0.014) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 46 / 132 / 142 (142) | **0.50** (0.061) 71 / 71 | 0.50 (0.061) | +0.021 (0.020) | +0.002 (0.024) | 0.307 / 0.286 / 0.304 | +0.107 (0.015) |
| random | north_leg | 32 (30) | 6 / 177 / 137 (133) | **0.54** (0.055) 74 / 63 | 0.54 (0.056) | +0.039 (0.026) | +0.009 (0.036) | 0.404 / 0.365 / 0.395 | +0.122 (0.018) |
| random | pooled | 96 (62) | 76 / 486 / 398 (364) | **0.49** (0.037) 195 / 203 | 0.48 (0.039) | +0.016 (0.012) | -0.002 (0.016) | 0.277 / 0.261 / 0.279 | +0.092 (0.010) |
| critics_round1/seed_0 region min | start_early | 32 (32) | 24 / 177 / 119 (89) | **0.45** (0.054) 53 / 66 | 0.42 (0.070) | -0.004 (0.012) | -0.009 (0.018) | 0.128 / 0.132 / 0.137 | +0.049 (0.014) |
| critics_round1/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | turn | 32 (30) | 46 / 132 / 142 (142) | **0.49** (0.058) 69 / 73 | 0.49 (0.058) | -0.014 (0.025) | -0.032 (0.033) | 0.272 / 0.286 / 0.304 | +0.107 (0.015) |
| critics_round1/seed_0 region min | north_leg | 32 (30) | 6 / 177 / 137 (133) | **0.53** (0.050) 73 / 64 | 0.54 (0.052) | +0.004 (0.022) | -0.026 (0.019) | 0.369 / 0.365 / 0.395 | +0.122 (0.018) |
| critics_round1/seed_0 region min | pooled | 96 (62) | 76 / 486 / 398 (364) | **0.49** (0.033) 195 / 203 | 0.49 (0.036) | -0.005 (0.012) | -0.023 (0.014) | 0.256 / 0.261 / 0.279 | +0.092 (0.010) |
| critics_round1/seed_1 region min | start_early | 32 (32) | 24 / 177 / 119 (89) | **0.55** (0.051) 65 / 54 | 0.61 (0.060) | +0.006 (0.009) | +0.001 (0.015) | 0.138 / 0.132 / 0.137 | +0.049 (0.014) |
| critics_round1/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | turn | 32 (30) | 46 / 132 / 142 (142) | **0.46** (0.067) 66 / 76 | 0.46 (0.067) | -0.014 (0.027) | -0.033 (0.038) | 0.272 / 0.286 / 0.304 | +0.107 (0.015) |
| critics_round1/seed_1 region min | north_leg | 32 (30) | 6 / 177 / 137 (133) | **0.53** (0.056) 73 / 64 | 0.53 (0.059) | -0.012 (0.018) | -0.043 (0.031) | 0.353 / 0.365 / 0.395 | +0.122 (0.018) |
| critics_round1/seed_1 region min | pooled | 96 (62) | 76 / 486 / 398 (364) | **0.51** (0.034) 204 / 194 | 0.52 (0.037) | -0.007 (0.011) | -0.025 (0.017) | 0.254 / 0.261 / 0.279 | +0.092 (0.010) |
| critics_round1/seed_2 region min | start_early | 32 (32) | 24 / 177 / 119 (89) | **0.63** (0.052) 75 / 44 | 0.66 (0.064) | +0.012 (0.008) | +0.007 (0.014) | 0.145 / 0.132 / 0.137 | +0.049 (0.014) |
| critics_round1/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | turn | 32 (30) | 46 / 132 / 142 (142) | **0.51** (0.054) 72 / 70 | 0.51 (0.054) | -0.053 (0.025) | -0.071 (0.036) | 0.233 / 0.286 / 0.304 | +0.107 (0.015) |
| critics_round1/seed_2 region min | north_leg | 32 (30) | 6 / 177 / 137 (133) | **0.58** (0.050) 79 / 58 | 0.58 (0.049) | +0.007 (0.023) | -0.024 (0.033) | 0.372 / 0.365 / 0.395 | +0.122 (0.018) |
| critics_round1/seed_2 region min | pooled | 96 (62) | 76 / 486 / 398 (364) | **0.57** (0.032) 226 / 172 | 0.57 (0.033) | -0.011 (0.012) | -0.029 (0.017) | 0.250 / 0.261 / 0.279 | +0.092 (0.010) |

## Layer C, matched candidate types (recorded, sample0, sample1)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 8 / 55 / 33 (25) | **0.45** (0.115) 15 / 18 | 0.36 (0.118) | -0.012 (0.014) | - (-) | 0.112 / 0.125 / - | +0.030 (0.009) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 14 / 35 / 47 (47) | **0.45** (0.096) 21 / 26 | 0.45 (0.096) | +0.008 (0.026) | - (-) | 0.281 / 0.273 / - | +0.111 (0.017) |
| random | north_leg | 32 (30) | 1 / 51 / 44 (44) | **0.55** (0.086) 24 / 20 | 0.55 (0.086) | +0.041 (0.022) | - (-) | 0.402 / 0.361 / - | +0.101 (0.016) |
| random | pooled | 96 (62) | 23 / 141 / 124 (116) | **0.48** (0.061) 60 / 64 | 0.47 (0.061) | +0.012 (0.012) | - (-) | 0.265 / 0.253 / - | +0.081 (0.009) |
| critics_round1/seed_0 region min | start_early | 32 (32) | 8 / 55 / 33 (25) | **0.55** (0.098) 18 / 15 | 0.48 (0.111) | +0.007 (0.011) | - (-) | 0.132 / 0.125 / - | +0.030 (0.009) |
| critics_round1/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_0 region min | turn | 32 (30) | 14 / 35 / 47 (47) | **0.43** (0.069) 20 / 27 | 0.43 (0.069) | -0.020 (0.022) | - (-) | 0.253 / 0.273 / - | +0.111 (0.017) |
| critics_round1/seed_0 region min | north_leg | 32 (30) | 1 / 51 / 44 (44) | **0.48** (0.072) 21 / 23 | 0.48 (0.072) | -0.018 (0.020) | - (-) | 0.342 / 0.361 / - | +0.101 (0.016) |
| critics_round1/seed_0 region min | pooled | 96 (62) | 23 / 141 / 124 (116) | **0.48** (0.048) 59 / 65 | 0.46 (0.052) | -0.011 (0.011) | - (-) | 0.242 / 0.253 / - | +0.081 (0.009) |
| critics_round1/seed_1 region min | start_early | 32 (32) | 8 / 55 / 33 (25) | **0.61** (0.078) 20 / 13 | 0.64 (0.084) | +0.013 (0.009) | - (-) | 0.137 / 0.125 / - | +0.030 (0.009) |
| critics_round1/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_1 region min | turn | 32 (30) | 14 / 35 / 47 (47) | **0.47** (0.088) 22 / 25 | 0.47 (0.088) | -0.016 (0.023) | - (-) | 0.257 / 0.273 / - | +0.111 (0.017) |
| critics_round1/seed_1 region min | north_leg | 32 (30) | 1 / 51 / 44 (44) | **0.52** (0.075) 23 / 21 | 0.52 (0.075) | -0.001 (0.018) | - (-) | 0.360 / 0.361 / - | +0.101 (0.016) |
| critics_round1/seed_1 region min | pooled | 96 (62) | 23 / 141 / 124 (116) | **0.52** (0.046) 65 / 59 | 0.53 (0.048) | -0.001 (0.010) | - (-) | 0.251 / 0.253 / - | +0.081 (0.009) |
| critics_round1/seed_2 region min | start_early | 32 (32) | 8 / 55 / 33 (25) | **0.70** (0.082) 23 / 10 | 0.76 (0.085) | +0.020 (0.008) | - (-) | 0.145 / 0.125 / - | +0.030 (0.009) |
| critics_round1/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_round1/seed_2 region min | turn | 32 (30) | 14 / 35 / 47 (47) | **0.45** (0.085) 21 / 26 | 0.45 (0.085) | -0.039 (0.024) | - (-) | 0.234 / 0.273 / - | +0.111 (0.017) |
| critics_round1/seed_2 region min | north_leg | 32 (30) | 1 / 51 / 44 (44) | **0.61** (0.066) 27 / 17 | 0.61 (0.066) | +0.029 (0.021) | - (-) | 0.390 / 0.361 / - | +0.101 (0.016) |
| critics_round1/seed_2 region min | pooled | 96 (62) | 23 / 141 / 124 (116) | **0.57** (0.046) 71 / 53 | 0.58 (0.048) | +0.003 (0.011) | - (-) | 0.256 / 0.253 / - | +0.081 (0.009) |

## All readouts, pooled (agreement among decided pairs; pick gain)

| scorer | A | A samples-only | B | A x B | C | C matched |
|---|---|---|---|---|---|---|
| random | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (398) / +0.016 | 0.48 (124) / +0.012 |
| critics_round1/seed_0 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (398) / -0.005 | 0.48 (124) / -0.011 |
| critics_round1/seed_0 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.46 (398) / -0.000 | 0.48 (124) / -0.002 |
| critics_round1/seed_0 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.45 (398) / -0.007 | 0.41 (124) / -0.014 |
| critics_round1/seed_0 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.47 (398) / -0.009 | 0.40 (124) / -0.021 |
| critics_round1/seed_0 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.50 (398) / -0.003 | 0.53 (124) / -0.002 |
| critics_round1/seed_0 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (398) / +0.003 | 0.52 (124) / -0.001 |
| critics_round1/seed_1 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (398) / -0.007 | 0.52 (124) / -0.001 |
| critics_round1/seed_1 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.53 (398) / +0.002 | 0.56 (124) / +0.009 |
| critics_round1/seed_1 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (398) / +0.003 | 0.57 (124) / +0.020 |
| critics_round1/seed_1 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.57 (398) / +0.007 | 0.62 (124) / +0.022 |
| critics_round1/seed_1 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (398) / -0.001 | 0.52 (124) / -0.002 |
| critics_round1/seed_1 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.47 (398) / -0.011 | 0.48 (124) / -0.012 |
| critics_round1/seed_2 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.57 (398) / -0.011 | 0.57 (124) / +0.003 |
| critics_round1/seed_2 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (398) / -0.009 | 0.48 (124) / -0.012 |
| critics_round1/seed_2 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (398) / +0.002 | 0.47 (124) / -0.003 |
| critics_round1/seed_2 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (398) / -0.002 | 0.51 (124) / -0.001 |
| critics_round1/seed_2 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.53 (398) / -0.012 | 0.56 (124) / +0.001 |
| critics_round1/seed_2 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.53 (398) / -0.006 | 0.58 (124) / -0.008 |

## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)

- critics_round1/seed_0: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.49 +- 0.033, n 398, gain -0.005]
- critics_round1/seed_1: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.51 +- 0.034, n 398, gain -0.007]
- critics_round1/seed_2: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: inconclusive (interval spans chance and useful agreement) [0.57 +- 0.032, n 398, gain -0.011]
- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): -; B: -; C: +0.092 (a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).
