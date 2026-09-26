# Round-1 critic diagnostic: memorisation (A) vs new torques (B) vs new-episode states (C)

Sealed manifest: 2026-09-18 14:27:26; reference commit .  Frozen: continuation seed_0 (d0869ab38be1), critics seed_0 (5512470d41fe), seed_1 (0c73f9aa71e7), seed_2 (b665fed7f015).  16 fresh paired draws per anchor (seeds ABC_SEED + 100 i + r), one query torque then the frozen policy mode closed-loop; P_goal = gamma 0.999, radius 0.5.  Primary readout = region-integrated exp(f) over the training NCE goal marginal (radius 0.5, per head and deployed min); secondary = exact recorded-goal logit.  Pair classes on the 16-draw log ratio: tie (= 0), weak (<= 0.3), decided (> 0.3); bootstrap-decided = decided and >= 90% of draw-resamples keep the sign.  Agreement = among decided pairs (ties are never counted as wrong); s.e. = episode bootstrap.  Oracle-stage diagnostic; nothing was trained.

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
| C|mode | 1536 | 0.67 | 0.24 | 0.278 | 0.62 |
| C|recorded | 1536 | 0.64 | 0.25 | 0.270 | 0.61 |
| C|sample0 | 1536 | 0.64 | 0.24 | 0.273 | 0.62 |
| C|sample1 | 1536 | 0.63 | 0.25 | 0.271 | 0.61 |
| C|sample2 | 1536 | 0.68 | 0.23 | 0.274 | 0.62 |

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
| critics_br_d20/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer A, the two policy samples only

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer B: familiar state, three new torques, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | start_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | turn | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | north_leg | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## A x B cross pairs (shared anchors and draws)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | pooled | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |

## Layer C: held-out-episode states, existing five candidates, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 12 / 177 / 131 (107) | **0.47** (0.068) 62 / 69 | 0.50 (0.078) | -0.009 (0.015) | +0.004 (0.030) | 0.126 / 0.135 / 0.122 | +0.064 (0.016) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 45 / 145 / 130 (129) | **0.55** (0.064) 72 / 58 | 0.56 (0.065) | -0.011 (0.030) | -0.024 (0.039) | 0.291 / 0.302 / 0.315 | +0.095 (0.013) |
| random | north_leg | 32 (30) | 3 / 175 / 142 (138) | **0.49** (0.054) 70 / 72 | 0.49 (0.056) | +0.012 (0.021) | -0.003 (0.033) | 0.395 / 0.383 / 0.398 | +0.097 (0.011) |
| random | pooled | 96 (62) | 60 / 497 / 403 (374) | **0.51** (0.037) 204 / 199 | 0.51 (0.041) | -0.003 (0.013) | -0.008 (0.019) | 0.271 / 0.273 / 0.278 | +0.085 (0.008) |
| critics_br_d20/seed_0 region min | start_early | 32 (32) | 12 / 177 / 131 (107) | **0.47** (0.060) 62 / 69 | 0.50 (0.068) | -0.002 (0.016) | +0.011 (0.026) | 0.133 / 0.135 / 0.122 | +0.064 (0.016) |
| critics_br_d20/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | turn | 32 (30) | 45 / 145 / 130 (129) | **0.56** (0.055) 73 / 57 | 0.56 (0.055) | -0.009 (0.022) | -0.022 (0.018) | 0.293 / 0.302 / 0.315 | +0.095 (0.013) |
| critics_br_d20/seed_0 region min | north_leg | 32 (30) | 3 / 175 / 142 (138) | **0.49** (0.054) 70 / 72 | 0.49 (0.054) | -0.046 (0.029) | -0.061 (0.038) | 0.337 / 0.383 / 0.398 | +0.097 (0.011) |
| critics_br_d20/seed_0 region min | pooled | 96 (62) | 60 / 497 / 403 (374) | **0.51** (0.035) 205 / 198 | 0.51 (0.038) | -0.019 (0.013) | -0.024 (0.017) | 0.254 / 0.273 / 0.278 | +0.085 (0.008) |
| critics_br_d20/seed_1 region min | start_early | 32 (32) | 12 / 177 / 131 (107) | **0.55** (0.055) 72 / 59 | 0.55 (0.051) | +0.000 (0.015) | +0.013 (0.022) | 0.135 / 0.135 / 0.122 | +0.064 (0.016) |
| critics_br_d20/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | turn | 32 (30) | 45 / 145 / 130 (129) | **0.55** (0.054) 71 / 59 | 0.54 (0.056) | +0.007 (0.020) | -0.006 (0.027) | 0.309 / 0.302 / 0.315 | +0.095 (0.013) |
| critics_br_d20/seed_1 region min | north_leg | 32 (30) | 3 / 175 / 142 (138) | **0.46** (0.062) 66 / 76 | 0.46 (0.066) | -0.050 (0.031) | -0.064 (0.039) | 0.334 / 0.383 / 0.398 | +0.097 (0.011) |
| critics_br_d20/seed_1 region min | pooled | 96 (62) | 60 / 497 / 403 (374) | **0.52** (0.037) 209 / 194 | 0.52 (0.037) | -0.014 (0.014) | -0.019 (0.018) | 0.259 / 0.273 / 0.278 | +0.085 (0.008) |
| critics_br_d20/seed_2 region min | start_early | 32 (32) | 12 / 177 / 131 (107) | **0.44** (0.047) 58 / 73 | 0.41 (0.053) | +0.007 (0.013) | +0.021 (0.024) | 0.142 / 0.135 / 0.122 | +0.064 (0.016) |
| critics_br_d20/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | turn | 32 (30) | 45 / 145 / 130 (129) | **0.58** (0.069) 75 / 55 | 0.58 (0.068) | -0.002 (0.025) | -0.015 (0.035) | 0.300 / 0.302 / 0.315 | +0.095 (0.013) |
| critics_br_d20/seed_2 region min | north_leg | 32 (30) | 3 / 175 / 142 (138) | **0.46** (0.058) 65 / 77 | 0.46 (0.058) | -0.040 (0.026) | -0.055 (0.041) | 0.343 / 0.383 / 0.398 | +0.097 (0.011) |
| critics_br_d20/seed_2 region min | pooled | 96 (62) | 60 / 497 / 403 (374) | **0.49** (0.034) 198 / 205 | 0.49 (0.035) | -0.012 (0.013) | -0.016 (0.020) | 0.262 / 0.273 / 0.278 | +0.085 (0.008) |

## Layer C, matched candidate types (recorded, sample0, sample1)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 3 / 58 / 35 (23) | **0.46** (0.099) 16 / 19 | 0.43 (0.128) | -0.004 (0.012) | - (-) | 0.132 / 0.136 / - | +0.038 (0.011) |
| random | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| random | turn | 32 (30) | 14 / 42 / 40 (40) | **0.62** (0.087) 25 / 15 | 0.62 (0.087) | +0.019 (0.025) | - (-) | 0.310 / 0.291 / - | +0.093 (0.017) |
| random | north_leg | 32 (30) | 2 / 55 / 39 (39) | **0.41** (0.087) 16 / 23 | 0.41 (0.087) | -0.011 (0.018) | - (-) | 0.376 / 0.388 / - | +0.074 (0.012) |
| random | pooled | 96 (62) | 19 / 155 / 114 (102) | **0.50** (0.065) 57 / 57 | 0.50 (0.066) | +0.001 (0.011) | - (-) | 0.273 / 0.272 / - | +0.068 (0.008) |
| critics_br_d20/seed_0 region min | start_early | 32 (32) | 3 / 58 / 35 (23) | **0.51** (0.086) 18 / 17 | 0.48 (0.109) | +0.002 (0.010) | - (-) | 0.139 / 0.136 / - | +0.038 (0.011) |
| critics_br_d20/seed_0 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_0 region min | turn | 32 (30) | 14 / 42 / 40 (40) | **0.60** (0.094) 24 / 16 | 0.60 (0.094) | +0.021 (0.023) | - (-) | 0.312 / 0.291 / - | +0.093 (0.017) |
| critics_br_d20/seed_0 region min | north_leg | 32 (30) | 2 / 55 / 39 (39) | **0.56** (0.085) 22 / 17 | 0.56 (0.085) | +0.007 (0.018) | - (-) | 0.394 / 0.388 / - | +0.074 (0.012) |
| critics_br_d20/seed_0 region min | pooled | 96 (62) | 19 / 155 / 114 (102) | **0.56** (0.052) 64 / 50 | 0.56 (0.058) | +0.010 (0.010) | - (-) | 0.282 / 0.272 / - | +0.068 (0.008) |
| critics_br_d20/seed_1 region min | start_early | 32 (32) | 3 / 58 / 35 (23) | **0.51** (0.104) 18 / 17 | 0.52 (0.117) | -0.001 (0.012) | - (-) | 0.135 / 0.136 / - | +0.038 (0.011) |
| critics_br_d20/seed_1 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_1 region min | turn | 32 (30) | 14 / 42 / 40 (40) | **0.50** (0.079) 20 / 20 | 0.50 (0.079) | +0.007 (0.021) | - (-) | 0.298 / 0.291 / - | +0.093 (0.017) |
| critics_br_d20/seed_1 region min | north_leg | 32 (30) | 2 / 55 / 39 (39) | **0.54** (0.094) 21 / 18 | 0.54 (0.094) | -0.006 (0.020) | - (-) | 0.381 / 0.388 / - | +0.074 (0.012) |
| critics_br_d20/seed_1 region min | pooled | 96 (62) | 19 / 155 / 114 (102) | **0.52** (0.055) 59 / 55 | 0.52 (0.058) | -0.000 (0.010) | - (-) | 0.271 / 0.272 / - | +0.068 (0.008) |
| critics_br_d20/seed_2 region min | start_early | 32 (32) | 3 / 58 / 35 (23) | **0.51** (0.089) 18 / 17 | 0.39 (0.109) | -0.005 (0.014) | - (-) | 0.132 / 0.136 / - | +0.038 (0.011) |
| critics_br_d20/seed_2 region min | start_late | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | shortcut_early | 0 (0) | 0 / 0 / 0 (0) | **-** (-) 0 / 0 | - (-) | - (-) | - (-) | - / - / - | - (-) |
| critics_br_d20/seed_2 region min | turn | 32 (30) | 14 / 42 / 40 (40) | **0.55** (0.102) 22 / 18 | 0.55 (0.102) | -0.005 (0.025) | - (-) | 0.285 / 0.291 / - | +0.093 (0.017) |
| critics_br_d20/seed_2 region min | north_leg | 32 (30) | 2 / 55 / 39 (39) | **0.51** (0.110) 20 / 19 | 0.51 (0.110) | -0.002 (0.019) | - (-) | 0.385 / 0.388 / - | +0.074 (0.012) |
| critics_br_d20/seed_2 region min | pooled | 96 (62) | 19 / 155 / 114 (102) | **0.53** (0.056) 60 / 54 | 0.50 (0.059) | -0.004 (0.011) | - (-) | 0.267 / 0.272 / - | +0.068 (0.008) |

## All readouts, pooled (agreement among decided pairs; pick gain)

| scorer | A | A samples-only | B | A x B | C | C matched |
|---|---|---|---|---|---|---|
| random | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (403) / -0.003 | 0.50 (114) / +0.001 |
| critics_br_d20/seed_0 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (403) / -0.019 | 0.56 (114) / +0.010 |
| critics_br_d20/seed_0 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (403) / -0.019 | 0.51 (114) / -0.002 |
| critics_br_d20/seed_0 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (403) / +0.007 | 0.52 (114) / +0.005 |
| critics_br_d20/seed_0 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.50 (403) / -0.005 | 0.54 (114) / +0.001 |
| critics_br_d20/seed_0 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (403) / -0.015 | 0.50 (114) / +0.000 |
| critics_br_d20/seed_0 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (403) / -0.014 | 0.46 (114) / -0.002 |
| critics_br_d20/seed_1 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (403) / -0.014 | 0.52 (114) / -0.000 |
| critics_br_d20/seed_1 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (403) / -0.015 | 0.49 (114) / +0.005 |
| critics_br_d20/seed_1 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.50 (403) / -0.003 | 0.49 (114) / +0.004 |
| critics_br_d20/seed_1 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (403) / -0.002 | 0.54 (114) / +0.010 |
| critics_br_d20/seed_1 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.46 (403) / -0.022 | 0.40 (114) / -0.009 |
| critics_br_d20/seed_1 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.45 (403) / -0.018 | 0.39 (114) / -0.012 |
| critics_br_d20/seed_2 region min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.49 (403) / -0.012 | 0.53 (114) / -0.004 |
| critics_br_d20/seed_2 exact min | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.51 (403) / -0.005 | 0.51 (114) / +0.001 |
| critics_br_d20/seed_2 region h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.46 (403) / -0.020 | 0.44 (114) / -0.016 |
| critics_br_d20/seed_2 exact h0 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.48 (403) / -0.013 | 0.44 (114) / -0.012 |
| critics_br_d20/seed_2 region h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (403) / -0.002 | 0.56 (114) / +0.013 |
| critics_br_d20/seed_2 exact h1 | - (0) / - | - (0) / - | - (0) / - | - (0) / - | 0.52 (403) / -0.006 | 0.58 (114) / +0.008 |

## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)

- critics_br_d20/seed_0: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.51 +- 0.035, n 403, gain -0.019]
- critics_br_d20/seed_1: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.52 +- 0.037, n 403, gain -0.014]
- critics_br_d20/seed_2: A: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; B: inconclusive (too few decided pairs) [- +- -, n 0, gain -]; C: fails (interval below 0.6) [0.49 +- 0.034, n 403, gain -0.012]
- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): -; B: -; C: +0.085 (a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).
