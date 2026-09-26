# Query coverage, stage 2: controlled training comparison (control = q0 futures, extended = q0..q5 futures with the query torque as the critic action, at the start-region anchors)

`manifest.json` (sealed before training), `report.json`.  Fresh paired evaluation draw seed 6909, 300 episodes, policy mode; every policy evaluated once.  Current = the lineage agent (the clipped CF final, the source checkpoint of both arms).  Rule = mean over the 3 lineages > 2 x seed s.e. and 3 / 3.

| policy | success | detour | death | timeout | success (no hazard) | success (hazard) | far entered / completed (ledger) | mean steps |
|---|---|---|---|---|---|---|---|---|
| control/seed_0 | 0.497 | 0.627 | 0.000 | 0.503 | 0.556 | 0.475 | 188 / 145 | 680 |
| extended/seed_0 | 0.267 | 0.123 | 0.477 | 0.257 | 0.642 | 0.128 | 37 / 19 | 352 |
| continue/seed_0 | 0.517 | 0.537 | 0.007 | 0.477 | 0.593 | 0.489 | 161 / 136 | 665 |
| frozen/seed_0 | 0.497 | 0.713 | 0.010 | 0.493 | 0.543 | 0.479 | 214 / 130 | 663 |
| current/seed_0 | 0.520 | 0.590 | 0.193 | 0.287 | 0.728 | 0.443 | 177 / 129 | 493 |
| control/seed_1 | 0.053 | 0.397 | 0.000 | 0.947 | 0.037 | 0.059 | 119 / 13 | 790 |
| extended/seed_1 | 0.233 | 0.297 | 0.137 | 0.630 | 0.358 | 0.187 | 89 / 29 | 619 |
| continue/seed_1 | 0.307 | 0.430 | 0.043 | 0.650 | 0.358 | 0.288 | 129 / 43 | 672 |
| frozen/seed_1 | 0.500 | 0.860 | 0.010 | 0.490 | 0.469 | 0.511 | 258 / 144 | 701 |
| current/seed_1 | 0.437 | 0.440 | 0.350 | 0.213 | 0.790 | 0.306 | 132 / 94 | 389 |
| control/seed_2 | 0.030 | 0.003 | 0.000 | 0.970 | 0.062 | 0.018 | 1 / 0 | 788 |
| extended/seed_2 | 0.180 | 0.060 | 0.320 | 0.500 | 0.432 | 0.087 | 18 / 6 | 492 |
| continue/seed_2 | 0.150 | 0.267 | 0.000 | 0.850 | 0.173 | 0.142 | 80 / 38 | 774 |
| frozen/seed_2 | 0.140 | 0.107 | 0.000 | 0.860 | 0.148 | 0.137 | 32 / 20 | 750 |
| current/seed_2 | 0.433 | 0.307 | 0.427 | 0.140 | 0.827 | 0.288 | 92 / 77 | 324 |
| start (reference) | 0.270 | 0.010 | 0.713 | 0.017 | 0.975 | 0.009 | - | 144 |

| comparison | quantity | per lineage | mean | seed s.e. | same direction | rule met |
|---|---|---|---|---|---|---|
| extended_minus_control | success | -0.230 / +0.180 / +0.150 | +0.033 | 0.132 | 2/3 | no |
| extended_minus_current | success | -0.253 / -0.203 / -0.253 | -0.237 | 0.017 | 3/3 | no |
| control_minus_current | success | -0.023 / -0.383 / -0.403 | -0.270 | 0.123 | 3/3 | no |
| frozen_minus_continue (diagnostic) | success | -0.020 / +0.193 / -0.010 | +0.054 | 0.070 | 1/3 | no |
| frozen_minus_current (diagnostic) | success | -0.023 / +0.063 / -0.293 | -0.084 | 0.107 | 2/3 | no |
| continue_minus_current (reference) | success | -0.003 / -0.130 / -0.283 | -0.139 | 0.081 | 3/3 | no |
| extended_minus_continue (reference) | success | -0.250 / -0.073 / +0.030 | -0.098 | 0.082 | 2/3 | no |
| control_minus_continue (reference) | success | -0.020 / -0.253 / -0.120 | -0.131 | 0.068 | 3/3 | no |
| extended_minus_control | detour | -0.503 / -0.100 / +0.057 | -0.182 | 0.167 | 2/3 | no |
| extended_minus_current | detour | -0.467 / -0.143 / -0.247 | -0.286 | 0.095 | 3/3 | no |
| control_minus_current | detour | +0.037 / -0.043 / -0.303 | -0.103 | 0.103 | 2/3 | no |
| frozen_minus_continue (diagnostic) | detour | +0.177 / +0.430 / -0.160 | +0.149 | 0.171 | 2/3 | no |
| frozen_minus_current (diagnostic) | detour | +0.123 / +0.420 / -0.200 | +0.114 | 0.179 | 2/3 | no |
| continue_minus_current (reference) | detour | -0.053 / -0.010 / -0.040 | -0.034 | 0.013 | 3/3 | no |
| extended_minus_continue (reference) | detour | -0.413 / -0.133 / -0.207 | -0.251 | 0.084 | 3/3 | no |
| control_minus_continue (reference) | detour | +0.090 / -0.033 / -0.263 | -0.069 | 0.104 | 2/3 | no |
| extended_minus_control | failure | +0.477 / +0.137 / +0.320 | +0.311 | 0.098 | 3/3 | YES |
| extended_minus_current | failure | +0.283 / -0.213 / -0.107 | -0.012 | 0.151 | 2/3 | no |
| control_minus_current | failure | -0.193 / -0.350 / -0.427 | -0.323 | 0.069 | 3/3 | no |
| frozen_minus_continue (diagnostic) | failure | +0.003 / -0.033 / +0.000 | -0.010 | 0.012 | 1/3 | no |
| frozen_minus_current (diagnostic) | failure | -0.183 / -0.340 / -0.427 | -0.317 | 0.071 | 3/3 | no |
| continue_minus_current (reference) | failure | -0.187 / -0.307 / -0.427 | -0.307 | 0.069 | 3/3 | no |
| extended_minus_continue (reference) | failure | +0.470 / +0.093 / +0.320 | +0.294 | 0.109 | 3/3 | YES |
| control_minus_continue (reference) | failure | -0.007 / -0.043 / +0.000 | -0.017 | 0.013 | 2/3 | no |
| extended_minus_control | timeout | -0.247 / -0.317 / -0.470 | -0.344 | 0.066 | 3/3 | no |
| extended_minus_current | timeout | -0.030 / +0.417 / +0.360 | +0.249 | 0.140 | 2/3 | no |
| control_minus_current | timeout | +0.217 / +0.733 / +0.830 | +0.593 | 0.190 | 3/3 | YES |
| frozen_minus_continue (diagnostic) | timeout | +0.017 / -0.160 / +0.010 | -0.044 | 0.058 | 1/3 | no |
| frozen_minus_current (diagnostic) | timeout | +0.207 / +0.277 / +0.720 | +0.401 | 0.161 | 3/3 | YES |
| continue_minus_current (reference) | timeout | +0.190 / +0.437 / +0.710 | +0.446 | 0.150 | 3/3 | YES |
| extended_minus_continue (reference) | timeout | -0.220 / -0.020 / -0.350 | -0.197 | 0.096 | 3/3 | no |
| control_minus_continue (reference) | timeout | +0.027 / +0.297 / +0.120 | +0.148 | 0.079 | 3/3 | no |

Identity check (first 4 batches): seed 0: anchors equal True, actor equal True, goals differ True, actions differ True; seed 1: anchors equal True, actor equal True, goals differ True, actions differ True; seed 2: anchors equal True, actor equal True, goals differ True, actions differ True

## Read-outs on the training anchors (in-sample; secondary)

| lineage | critic | stratum | anchors (both outcomes) | within-anchor AUROC (success) | argmax query succeeded | f(best) - f(logged) |
|---|---|---|---|---|---|---|
| seed_0 | current | reset rows | 196 (154) | 0.619 | 0.577 | +1.779 |
| seed_0 | current | all start anchors | 4276 (1109) | 0.511 | 0.312 | +0.170 |
| seed_0 | control | reset rows | 196 (154) | 0.562 | 0.531 | +0.711 |
| seed_0 | control | all start anchors | 4276 (1109) | 0.517 | 0.316 | +0.057 |
| seed_0 | extended | reset rows | 196 (154) | 0.622 | 0.577 | +1.016 |
| seed_0 | extended | all start anchors | 4276 (1109) | 0.543 | 0.320 | +0.082 |
| seed_0 | continue | reset rows | 196 (154) | 0.574 | 0.551 | +1.136 |
| seed_0 | continue | all start anchors | 4276 (1109) | 0.511 | 0.313 | +0.103 |
| seed_1 | current | reset rows | 196 (138) | 0.583 | 0.541 | +1.045 |
| seed_1 | current | all start anchors | 4276 (851) | 0.511 | 0.319 | +0.075 |
| seed_1 | control | reset rows | 196 (138) | 0.572 | 0.546 | +1.132 |
| seed_1 | control | all start anchors | 4276 (851) | 0.526 | 0.320 | +0.073 |
| seed_1 | extended | reset rows | 196 (138) | 0.668 | 0.628 | +0.819 |
| seed_1 | extended | all start anchors | 4276 (851) | 0.563 | 0.328 | +0.062 |
| seed_1 | continue | reset rows | 196 (138) | 0.571 | 0.566 | +0.771 |
| seed_1 | continue | all start anchors | 4276 (851) | 0.52 | 0.321 | +0.055 |
| seed_2 | current | reset rows | 196 (111) | 0.621 | 0.546 | +0.793 |
| seed_2 | current | all start anchors | 4276 (1307) | 0.515 | 0.368 | +0.132 |
| seed_2 | control | reset rows | 196 (111) | 0.583 | 0.556 | +0.379 |
| seed_2 | control | all start anchors | 4276 (1307) | 0.525 | 0.380 | +0.077 |
| seed_2 | extended | reset rows | 196 (111) | 0.654 | 0.577 | +0.618 |
| seed_2 | extended | all start anchors | 4276 (1307) | 0.552 | 0.383 | +0.081 |
| seed_2 | continue | reset rows | 196 (111) | 0.594 | 0.526 | +0.877 |
| seed_2 | continue | all start anchors | 4276 (1307) | 0.517 | 0.369 | +0.126 |

| lineage | actor | own rollout from the reset anchors: completed far | entered far | success | death | timeout | mode distance to best query / to logged |
|---|---|---|---|---|---|---|---|
| seed_0 | current | 0.407 | 0.623 | 0.474 | 0.233 | 0.293 | 0.92 / 2.84 |
| seed_0 | control | 0.410 | 0.645 | 0.427 | 0.000 | 0.573 | 2.04 / 2.73 |
| seed_0 | extended | 0.053 | 0.121 | 0.253 | 0.518 | 0.229 | 2.13 / 1.84 |
| seed_0 | continue | 0.410 | 0.522 | 0.485 | 0.009 | 0.507 | 2.87 / 2.69 |
| seed_1 | current | 0.399 | 0.485 | 0.496 | 0.335 | 0.170 | 0.79 / 1.86 |
| seed_1 | control | 0.048 | 0.379 | 0.062 | 0.000 | 0.938 | 2.40 / 3.09 |
| seed_1 | extended | 0.088 | 0.264 | 0.203 | 0.167 | 0.630 | 1.46 / 1.29 |
| seed_1 | continue | 0.163 | 0.498 | 0.284 | 0.048 | 0.667 | 1.74 / 1.36 |
| seed_2 | current | 0.293 | 0.330 | 0.469 | 0.423 | 0.108 | 0.88 / 1.82 |
| seed_2 | control | 0.000 | 0.002 | 0.046 | 0.000 | 0.954 | 3.67 / 3.66 |
| seed_2 | extended | 0.015 | 0.066 | 0.192 | 0.346 | 0.463 | 1.39 / 1.70 |
| seed_2 | continue | 0.112 | 0.267 | 0.139 | 0.000 | 0.861 | 1.29 / 1.37 |

Judgement inputs: `{"complete_lineages": [0, 1, 2], "1_extended_beats_control_and_current": false, "critic_learned_reset_auroc_current_vs_extended": [[0.6194805194805195, 0.6215187590187591], [0.5833534621578099, 0.6680958132045088], [0.6207957957957959, 0.6542042042042042]], "critic_argmax_succeeded_reset_current_vs_extended": [[0.576530612244898, 0.576530612244898], [0.5408163265306123, 0.6275510204081632], [0.5459183673469388, 0.576530612244898]]}`

