# Round-1 critic diagnostic: memorisation (A) vs new torques (B) vs new-episode states (C)

Sealed manifest: 2026-09-18 10:31:48; reference commit see git.  Frozen: continuation seed_0 (820cb7e229ed), critics seed_0 (2fc6841e3cc5), seed_1 (53e41d31efab), seed_2 (70d302c796e4).  16 fresh paired draws per anchor (seeds ABC_SEED + 100 i + r), one query torque then the frozen policy mode closed-loop; P_goal = gamma 0.999, radius 0.5.  Primary readout = region-integrated exp(f) over the training NCE goal marginal (radius 0.5, per head and deployed min); secondary = exact recorded-goal logit.  Pair classes on the 16-draw log ratio: tie (= 0), weak (<= 0.3), decided (> 0.3); bootstrap-decided = decided and >= 90% of draw-resamples keep the sign.  Agreement = among decided pairs (ties are never counted as wrong); s.e. = episode bootstrap.  Oracle-stage diagnostic; nothing was trained.

## Anchors

| layer | stratum | anchors | episodes | t min / median / max | goal xy mean (std) | candidate torque spread | dist to mode | log pi(a) | frac saturated |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| C | turn | 32 | 30 | 7 / 18 / 88 | [24.753, 0.754] ([0.301, 0.388]) | 1.59 | 1.04 | -0.3 | 0.40 |
| C | north_leg | 32 | 30 | 18 / 50 / 133 | [24.762, 0.74] ([0.308, 0.384]) | 1.01 | 0.64 | 5.1 | 0.16 |
| C | pooled | 64 | 30 | 7 / 29 / 133 | [24.758, 0.747] ([0.304, 0.386]) | 1.30 | 0.84 | 2.4 | 0.28 |

Standardised-state distance: A anchor to its nearest other A anchor, median -; C anchor to its nearest A anchor, median -.  C differs from A in episode, and may differ in pose, goal and time distributions (see the table); B differs from A only in the torque.

## Fresh outcomes by layer and candidate

| key | paths | reach | death | P_goal | went around |
|---|---:|---:|---:|---:|---:|
| C|mode | 1024 | 0.58 | 0.03 | 0.163 | 0.81 |
| C|recorded | 1024 | 0.68 | 0.04 | 0.205 | 0.81 |
| C|sample0 | 1024 | 0.68 | 0.04 | 0.203 | 0.81 |
| C|sample1 | 1024 | 0.70 | 0.07 | 0.224 | 0.81 |
| C|sample2 | 1024 | 0.69 | 0.04 | 0.217 | 0.82 |

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
| critics_narrow/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer A, the two policy samples only

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer B: familiar state, three new torques, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## A x B cross pairs (shared anchors and draws)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer C: held-out-episode states, existing five candidates, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 54 / 64 / 202 (180) | **0.55** (0.047) 112 / 90 | 0.57 (0.051) | -0.028 (0.022) | +0.008 (0.021) | 0.142 / 0.170 / 0.134 | +0.155 (0.020) |
| random | north_leg | 32 (30) | 7 / 78 / 235 (222) | **0.52** (0.040) 122 / 113 | 0.54 (0.040) | -0.008 (0.024) | +0.035 (0.031) | 0.226 / 0.234 / 0.191 | +0.173 (0.017) |
| random | pooled | 64 (30) | 61 / 142 / 437 (402) | **0.54** (0.024) 234 / 203 | 0.55 (0.025) | -0.018 (0.016) | +0.021 (0.018) | 0.184 / 0.202 / 0.163 | +0.164 (0.013) |
| critics_narrow/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | turn | 32 (30) | 54 / 64 / 202 (180) | **0.48** (0.053) 97 / 105 | 0.46 (0.057) | +0.011 (0.023) | +0.048 (0.027) | 0.182 / 0.170 / 0.134 | +0.155 (0.020) |
| critics_narrow/seed_0 region min | north_leg | 32 (30) | 7 / 78 / 235 (222) | **0.45** (0.042) 106 / 129 | 0.45 (0.043) | +0.003 (0.023) | +0.046 (0.031) | 0.237 / 0.234 / 0.191 | +0.173 (0.017) |
| critics_narrow/seed_0 region min | pooled | 64 (30) | 61 / 142 / 437 (402) | **0.46** (0.030) 203 / 234 | 0.46 (0.032) | +0.007 (0.016) | +0.047 (0.020) | 0.209 / 0.202 / 0.163 | +0.164 (0.013) |
| critics_narrow/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | turn | 32 (30) | 54 / 64 / 202 (180) | **0.50** (0.045) 102 / 100 | 0.49 (0.050) | +0.011 (0.019) | +0.048 (0.025) | 0.182 / 0.170 / 0.134 | +0.155 (0.020) |
| critics_narrow/seed_1 region min | north_leg | 32 (30) | 7 / 78 / 235 (222) | **0.51** (0.035) 120 / 115 | 0.51 (0.038) | +0.022 (0.023) | +0.065 (0.038) | 0.256 / 0.234 / 0.191 | +0.173 (0.017) |
| critics_narrow/seed_1 region min | pooled | 64 (30) | 61 / 142 / 437 (402) | **0.51** (0.032) 222 / 215 | 0.50 (0.035) | +0.017 (0.015) | +0.056 (0.023) | 0.219 / 0.202 / 0.163 | +0.164 (0.013) |
| critics_narrow/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | turn | 32 (30) | 54 / 64 / 202 (180) | **0.45** (0.043) 90 / 112 | 0.45 (0.043) | -0.009 (0.018) | +0.028 (0.029) | 0.162 / 0.170 / 0.134 | +0.155 (0.020) |
| critics_narrow/seed_2 region min | north_leg | 32 (30) | 7 / 78 / 235 (222) | **0.45** (0.045) 105 / 130 | 0.45 (0.049) | +0.006 (0.024) | +0.049 (0.040) | 0.240 / 0.234 / 0.191 | +0.173 (0.017) |
| critics_narrow/seed_2 region min | pooled | 64 (30) | 61 / 142 / 437 (402) | **0.45** (0.033) 195 / 242 | 0.45 (0.037) | -0.001 (0.015) | +0.038 (0.024) | 0.201 / 0.202 / 0.163 | +0.164 (0.013) |

## Layer C, matched candidate types (recorded, sample0, sample1)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 15 / 27 / 54 (48) | **0.50** (0.075) 27 / 27 | 0.54 (0.081) | -0.010 (0.017) | - (-) | 0.162 / 0.172 / - | +0.082 (0.017) |
| random | north_leg | 32 (30) | 2 / 28 / 66 (63) | **0.48** (0.061) 32 / 34 | 0.51 (0.062) | -0.001 (0.021) | - (-) | 0.249 / 0.249 / - | +0.111 (0.016) |
| random | pooled | 64 (30) | 17 / 55 / 120 (111) | **0.49** (0.047) 59 / 61 | 0.52 (0.052) | -0.005 (0.013) | - (-) | 0.205 / 0.211 / - | +0.096 (0.012) |
| critics_narrow/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | turn | 32 (30) | 15 / 27 / 54 (48) | **0.50** (0.092) 27 / 27 | 0.52 (0.104) | +0.008 (0.017) | - (-) | 0.180 / 0.172 / - | +0.082 (0.017) |
| critics_narrow/seed_0 region min | north_leg | 32 (30) | 2 / 28 / 66 (63) | **0.41** (0.069) 27 / 39 | 0.40 (0.075) | +0.006 (0.017) | - (-) | 0.255 / 0.249 / - | +0.111 (0.016) |
| critics_narrow/seed_0 region min | pooled | 64 (30) | 17 / 55 / 120 (111) | **0.45** (0.057) 54 / 66 | 0.45 (0.064) | +0.007 (0.012) | - (-) | 0.218 / 0.211 / - | +0.096 (0.012) |
| critics_narrow/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | turn | 32 (30) | 15 / 27 / 54 (48) | **0.48** (0.086) 26 / 28 | 0.48 (0.094) | +0.003 (0.022) | - (-) | 0.175 / 0.172 / - | +0.082 (0.017) |
| critics_narrow/seed_1 region min | north_leg | 32 (30) | 2 / 28 / 66 (63) | **0.52** (0.054) 34 / 32 | 0.51 (0.055) | +0.001 (0.019) | - (-) | 0.251 / 0.249 / - | +0.111 (0.016) |
| critics_narrow/seed_1 region min | pooled | 64 (30) | 17 / 55 / 120 (111) | **0.50** (0.045) 60 / 60 | 0.50 (0.047) | +0.002 (0.014) | - (-) | 0.213 / 0.211 / - | +0.096 (0.012) |
| critics_narrow/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | turn | 32 (30) | 15 / 27 / 54 (48) | **0.33** (0.074) 18 / 36 | 0.31 (0.075) | -0.052 (0.018) | - (-) | 0.120 / 0.172 / - | +0.082 (0.017) |
| critics_narrow/seed_2 region min | north_leg | 32 (30) | 2 / 28 / 66 (63) | **0.50** (0.065) 33 / 33 | 0.51 (0.071) | +0.008 (0.019) | - (-) | 0.257 / 0.249 / - | +0.111 (0.016) |
| critics_narrow/seed_2 region min | pooled | 64 (30) | 17 / 55 / 120 (111) | **0.42** (0.052) 51 / 69 | 0.42 (0.053) | -0.022 (0.014) | - (-) | 0.189 / 0.211 / - | +0.096 (0.012) |

## All readouts, pooled (agreement among decided pairs; pick gain)

| scorer | A | A samples-only | B | A x B | C | C matched |
|---|---|---|---|---|---|---|
| random | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.54 (437) / -0.018 | 0.49 (120) / -0.005 |
| critics_narrow/seed_0 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.46 (437) / +0.007 | 0.45 (120) / +0.007 |
| critics_narrow/seed_0 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (437) / +0.016 | 0.53 (120) / +0.017 |
| critics_narrow/seed_0 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (437) / -0.018 | 0.46 (120) / -0.017 |
| critics_narrow/seed_0 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (437) / -0.001 | 0.49 (120) / -0.008 |
| critics_narrow/seed_0 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.47 (437) / +0.007 | 0.42 (120) / -0.011 |
| critics_narrow/seed_0 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (437) / +0.011 | 0.52 (120) / +0.011 |
| critics_narrow/seed_1 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (437) / +0.017 | 0.50 (120) / +0.002 |
| critics_narrow/seed_1 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (437) / +0.016 | 0.54 (120) / +0.005 |
| critics_narrow/seed_1 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.53 (437) / +0.037 | 0.48 (120) / +0.002 |
| critics_narrow/seed_1 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.54 (437) / +0.023 | 0.52 (120) / -0.003 |
| critics_narrow/seed_1 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.45 (437) / -0.026 | 0.47 (120) / -0.002 |
| critics_narrow/seed_1 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (437) / -0.007 | 0.50 (120) / +0.009 |
| critics_narrow/seed_2 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.45 (437) / -0.001 | 0.42 (120) / -0.022 |
| critics_narrow/seed_2 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (437) / -0.003 | 0.49 (120) / +0.009 |
| critics_narrow/seed_2 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.43 (437) / -0.005 | 0.43 (120) / -0.021 |
| critics_narrow/seed_2 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (437) / +0.001 | 0.53 (120) / +0.012 |
| critics_narrow/seed_2 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.47 (437) / -0.001 | 0.42 (120) / -0.018 |
| critics_narrow/seed_2 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.47 (437) / -0.017 | 0.44 (120) / -0.018 |

## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)

- critics_narrow/seed_0: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.46 +- 0.030, n 437, gain +0.007]
- critics_narrow/seed_1: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.51 +- 0.032, n 437, gain +0.017]
- critics_narrow/seed_2: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.45 +- 0.033, n 437, gain -0.001]
- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): -; B: -; C: +0.164 (a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).
