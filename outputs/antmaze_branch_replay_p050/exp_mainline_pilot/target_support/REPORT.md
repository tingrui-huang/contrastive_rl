# Task-goal target support per candidate action: current vs absorbing termination law

Sealed 2026-09-20 03:19:02 (git unavaila); `manifest.json`, `report.json`, `per_branch_s*.npz` (not committed).  No rollouts, no training.

Laws: current = P(m) ~ gamma^m over m = 1..L-1 (CriticStream.draw; each path normalised over its own rows); absorbing = P(m) ~ gamma^m over m = 1..H, H = HORIZON - t; rows m >= L-1 of a success / death branch are the branch's actual terminal row; a timeout (L-1 == H) is unchanged.  No relabelling: no death filled in as a goal, no reach position replaced by the goal centre, success and death both get the tail.  gamma 0.999, horizon 800.

## Marginals (the law of the NCE negatives; anchor-weighted over all anchors)

| source | success / death / timeout | reach_0.5 cur / abs | within_1 cur / abs | within_2 cur / abs | goal_area cur / abs | mean tail share (abs) |
|---|---|---:|---:|---:|---:|---:|
| CF_sealed | 0.621 / 0.291 / 0.088 | 0.0194 / 0.4933 | 0.1086 / 0.5087 | 0.2197 / 0.5403 | 0.2730 / 0.5500 | 0.757 |
| O_recorded | 0.999 / 0.000 / 0.001 | 0.0224 / 0.7574 | 0.0968 / 0.7656 | 0.2019 / 0.7810 | 0.2731 / 0.7939 | 0.756 |

## Lineage 0

### all_start: 4276 anchors, weight 0.0794

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.180 | 0.026 | 0.041 | 0.692 | 0.128 | 209 | 0.00061 | 0.1003 | 0.0034 | 0.557 | 0.0164 | 0.1082 |
| mode | 0.195 | 0.046 | 0.070 | 0.667 | 0.138 | 224 | 0.00062 | 0.1024 | 0.0032 | 0.527 | 0.0165 | 0.1105 |
| sample0 | 0.199 | 0.048 | 0.067 | 0.668 | 0.132 | 221 | 0.00063 | 0.1047 | 0.0032 | 0.525 | 0.0171 | 0.1134 |
| sample1 | 0.193 | 0.041 | 0.065 | 0.671 | 0.136 | 222 | 0.00062 | 0.1022 | 0.0032 | 0.529 | 0.0169 | 0.1106 |
| sample2 | 0.196 | 0.045 | 0.066 | 0.670 | 0.134 | 221 | 0.00063 | 0.1038 | 0.0032 | 0.529 | 0.0166 | 0.1118 |
| sample3 | 0.193 | 0.043 | 0.066 | 0.670 | 0.137 | 222 | 0.00062 | 0.1026 | 0.0032 | 0.531 | 0.0172 | 0.1114 |

**R4 far-going vs short-going queries (PRIMARY): 312 anchors with both classes (weight 0.0058 of 0.0794); far-going query share 0.062, mixed-route share 0.0010.**

delta success (far - short) = 0.515 [0.467, 0.560]; success far 0.672 vs short 0.157.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0010 | 0.0005 | 0.0006 [0.0004, 0.0007] | 0.898 | 0.102 | 0.0049 / 0.0005 / 0.0004 |
| reach_0.5 | abs | 0.186 | 0.084 | 0.102 [0.081, 0.124] | 0.895 | 0.105 | 0.0049 / 0.0005 / 0.0004 |
| within_1 | cur | 0.024 | 0.0031 | 0.021 [0.018, 0.024] | 0.986 | 0.014 | 0.0049 / 0.0005 / 0.0004 |
| within_1 | abs | 0.204 | 0.085 | 0.119 [0.099, 0.140] | 0.925 | 0.075 | 0.0049 / 0.0005 / 0.0004 |
| within_2 | cur | 0.069 | 0.0080 | 0.061 [0.056, 0.067] | 0.997 | 0.0034 | 0.0049 / 0.0005 / 0.0004 |
| within_2 | abs | 0.240 | 0.087 | 0.153 [0.131, 0.174] | 0.942 | 0.058 | 0.0049 / 0.0005 / 0.0004 |
| goal_area | cur | 0.042 | 0.011 | 0.031 [0.026, 0.037] | 0.942 | 0.058 | 0.0049 / 0.0005 / 0.0004 |
| goal_area | abs | 0.220 | 0.089 | 0.131 [0.109, 0.151] | 0.929 | 0.071 | 0.0049 / 0.0005 / 0.0004 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.984 | 0.0000 | 8085 | 0.972 | 1139 |
| reach_0.5_abs | 0.984 | 0.0000 | 8085 | 0.977 | 1139 |
| goal_area_cur | 0.953 | 0.0000 | 8085 | 0.886 | 1139 |
| goal_area_abs | 0.977 | 0.0000 | 8085 | 0.965 | 1139 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 1181 | 1.000 / 1.000 | 0.0015 / 0.275 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 1213 | 1.000 / 1.000 | 0.0016 / 0.276 | inf / inf | 0.999 / 1.000 |
| far_success_vs_far_timeout | 2490 | 1.000 / 1.000 | 0.0016 / 0.271 | inf / inf | 0.931 / 0.960 |
| far_success_vs_short_success | 445 | 0.050 / 0.050 | -0.0015 / -0.252 | 0.511 / 0.524 | 0.343 / 0.056 |
| short_success_vs_short_death | 303 | 1.000 / 1.000 | 0.0025 / 0.468 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 5550 | 1.000 / 1.000 | 0.0030 / 0.530 | inf / inf | 0.974 / 0.993 |
| success_vs_non_success | 10971 | 1.000 / 1.000 | 0.0023 / 0.415 | inf / inf | 0.970 / 0.988 |

### reset: 196 anchors, weight 0.0038

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.198 | 0.037 | 0.064 | 0.665 | 0.137 | 235 | 0.00059 | 0.1025 | 0.0030 | 0.517 | 0.0163 | 0.1107 |
| mode | 0.502 | 0.423 | 0.590 | 0.218 | 0.280 | 483 | 0.00091 | 0.1629 | 0.0018 | 0.324 | 0.0284 | 0.1830 |
| sample0 | 0.513 | 0.412 | 0.548 | 0.258 | 0.229 | 442 | 0.00098 | 0.1775 | 0.0019 | 0.346 | 0.0286 | 0.1973 |
| sample1 | 0.423 | 0.330 | 0.502 | 0.291 | 0.286 | 454 | 0.00080 | 0.1418 | 0.0019 | 0.335 | 0.0275 | 0.1616 |
| sample2 | 0.438 | 0.363 | 0.489 | 0.286 | 0.275 | 453 | 0.00082 | 0.1467 | 0.0019 | 0.335 | 0.0288 | 0.1674 |
| sample3 | 0.478 | 0.412 | 0.573 | 0.244 | 0.278 | 470 | 0.00087 | 0.1563 | 0.0018 | 0.327 | 0.0270 | 0.1755 |

**R4 far-going vs short-going queries (PRIMARY): 165 anchors with both classes (weight 0.0032 of 0.0038); far-going query share 0.457, mixed-route share 0.0088.**

delta success (far - short) = 0.542 [0.482, 0.600]; success far 0.709 vs short 0.167.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0011 | 0.0005 | 0.0006 [0.0004, 0.0007] | 0.863 | 0.137 | 0.0027 / 0.0002 / 0.0003 |
| reach_0.5 | abs | 0.199 | 0.091 | 0.108 [0.081, 0.133] | 0.851 | 0.149 | 0.0027 / 0.0002 / 0.0003 |
| within_1 | cur | 0.026 | 0.0035 | 0.022 [0.019, 0.026] | 0.975 | 0.025 | 0.0027 / 0.0002 / 0.0003 |
| within_1 | abs | 0.218 | 0.092 | 0.126 [0.100, 0.153] | 0.901 | 0.099 | 0.0027 / 0.0002 / 0.0003 |
| within_2 | cur | 0.074 | 0.0089 | 0.066 [0.057, 0.076] | 0.994 | 0.0062 | 0.0027 / 0.0002 / 0.0003 |
| within_2 | abs | 0.257 | 0.094 | 0.162 [0.134, 0.190] | 0.925 | 0.075 | 0.0027 / 0.0002 / 0.0003 |
| goal_area | cur | 0.045 | 0.012 | 0.033 [0.025, 0.042] | 0.913 | 0.087 | 0.0027 / 0.0002 / 0.0003 |
| goal_area | abs | 0.235 | 0.096 | 0.139 [0.112, 0.166] | 0.901 | 0.099 | 0.0027 / 0.0002 / 0.0003 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.938 | 0.0000 | 1370 | 0.881 | 167 |
| reach_0.5_abs | 0.942 | 0.0000 | 1370 | 0.891 | 167 |
| goal_area_cur | 0.931 | 0.0000 | 1370 | 0.860 | 167 |
| goal_area_abs | 0.947 | 0.0000 | 1370 | 0.902 | 167 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 841 | 1.000 / 1.000 | 0.0015 / 0.273 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 543 | 1.000 / 1.000 | 0.0016 / 0.290 | inf / inf | 0.998 / 1.000 |
| far_success_vs_far_timeout | 627 | 1.000 / 1.000 | 0.0016 / 0.291 | inf / inf | 0.912 / 0.951 |
| far_success_vs_short_success | 281 | 0.046 / 0.046 | -0.0015 / -0.250 | 0.511 / 0.533 | 0.323 / 0.046 |
| short_success_vs_short_death | 74 | 1.000 / 1.000 | 0.0026 / 0.486 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 111 | 1.000 / 1.000 | 0.0031 / 0.547 | inf / inf | 1.000 / 1.000 |
| success_vs_non_success | 2300 | 1.000 / 1.000 | 0.0018 / 0.318 | inf / inf | 0.973 / 0.987 |

### t_1_30: 4037 anchors, weight 0.0749

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.175 | 0.020 | 0.032 | 0.700 | 0.125 | 204 | 0.00060 | 0.0994 | 0.0035 | 0.569 | 0.0159 | 0.1070 |
| mode | 0.175 | 0.021 | 0.036 | 0.697 | 0.128 | 207 | 0.00060 | 0.0989 | 0.0034 | 0.566 | 0.0156 | 0.1061 |
| sample0 | 0.179 | 0.025 | 0.036 | 0.696 | 0.125 | 206 | 0.00061 | 0.1005 | 0.0034 | 0.561 | 0.0162 | 0.1082 |
| sample1 | 0.178 | 0.022 | 0.035 | 0.697 | 0.125 | 206 | 0.00061 | 0.1002 | 0.0034 | 0.562 | 0.0160 | 0.1077 |
| sample2 | 0.180 | 0.024 | 0.038 | 0.696 | 0.124 | 205 | 0.00061 | 0.1009 | 0.0034 | 0.561 | 0.0157 | 0.1081 |
| sample3 | 0.175 | 0.020 | 0.034 | 0.698 | 0.127 | 206 | 0.00060 | 0.0991 | 0.0034 | 0.566 | 0.0165 | 0.1071 |

**R4 far-going vs short-going queries (PRIMARY): 122 anchors with both classes (weight 0.0022 of 0.0749); far-going query share 0.035, mixed-route share 0.0004.**

delta success (far - short) = 0.523 [0.446, 0.595]; success far 0.641 vs short 0.118.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0010 | 0.0004 | 0.0006 [0.0005, 0.0008] | 0.940 | 0.060 | 0.0020 / 0.0002 / 0.0001 |
| reach_0.5 | abs | 0.181 | 0.065 | 0.116 [0.083, 0.146] | 0.949 | 0.051 | 0.0020 / 0.0002 / 0.0001 |
| within_1 | cur | 0.022 | 0.0023 | 0.020 [0.016, 0.023] | 1.000 | 0.0000 | 0.0020 / 0.0002 / 0.0001 |
| within_1 | abs | 0.197 | 0.066 | 0.131 [0.096, 0.163] | 0.957 | 0.043 | 0.0020 / 0.0002 / 0.0001 |
| within_2 | cur | 0.064 | 0.0059 | 0.058 [0.051, 0.066] | 1.000 | 0.0000 | 0.0020 / 0.0002 / 0.0001 |
| within_2 | abs | 0.230 | 0.068 | 0.162 [0.129, 0.194] | 0.966 | 0.034 | 0.0020 / 0.0002 / 0.0001 |
| goal_area | cur | 0.038 | 0.0082 | 0.030 [0.022, 0.038] | 0.974 | 0.026 | 0.0020 / 0.0002 / 0.0001 |
| goal_area | abs | 0.211 | 0.069 | 0.142 [0.108, 0.174] | 0.966 | 0.034 | 0.0020 / 0.0002 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.993 | 0.0000 | 6386 | 0.987 | 932 |
| reach_0.5_abs | 0.994 | 0.0000 | 6386 | 0.992 | 932 |
| goal_area_cur | 0.960 | 0.0000 | 6386 | 0.895 | 932 |
| goal_area_abs | 0.983 | 0.0000 | 6386 | 0.975 | 932 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 340 | 1.000 / 1.000 | 0.0016 / 0.278 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 535 | 1.000 / 1.000 | 0.0016 / 0.277 | inf / inf | 1.000 / 1.000 |
| far_success_vs_far_timeout | 1560 | 1.000 / 1.000 | 0.0015 / 0.268 | inf / inf | 0.938 / 0.961 |
| far_success_vs_short_success | 110 | 0.049 / 0.049 | -0.0015 / -0.258 | 0.510 / 0.531 | 0.270 / 0.049 |
| short_success_vs_short_death | 229 | 1.000 / 1.000 | 0.0025 / 0.460 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 5374 | 1.000 / 1.000 | 0.0030 / 0.531 | inf / inf | 0.973 / 0.993 |
| success_vs_non_success | 8115 | 1.000 / 1.000 | 0.0026 / 0.452 | inf / inf | 0.970 / 0.988 |

### t_ge_30: 43 anchors, weight 0.0008

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.596 | 0.511 | 0.798 | 0.000 | 0.404 | 588 | 0.00103 | 0.1683 | 0.0017 | 0.283 | 0.0571 | 0.2121 |
| mode | 0.585 | 0.543 | 0.809 | 0.000 | 0.415 | 605 | 0.00093 | 0.1470 | 0.0016 | 0.251 | 0.0393 | 0.1774 |
| sample0 | 0.606 | 0.532 | 0.787 | 0.000 | 0.394 | 596 | 0.00101 | 0.1598 | 0.0017 | 0.264 | 0.0515 | 0.1999 |
| sample1 | 0.511 | 0.457 | 0.777 | 0.000 | 0.489 | 644 | 0.00074 | 0.1057 | 0.0015 | 0.207 | 0.0537 | 0.1498 |
| sample2 | 0.574 | 0.489 | 0.660 | 0.000 | 0.426 | 591 | 0.00105 | 0.1681 | 0.0018 | 0.293 | 0.0394 | 0.1970 |
| sample3 | 0.564 | 0.436 | 0.713 | 0.000 | 0.436 | 578 | 0.00112 | 0.1837 | 0.0020 | 0.326 | 0.0382 | 0.2097 |

**R4 far-going vs short-going queries (PRIMARY): 25 anchors with both classes (weight 0.0004 of 0.0008); far-going query share 0.748, mixed-route share 0.018.**

delta success (far - short) = 0.285 [0.094, 0.482]; success far 0.562 vs short 0.278.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0008 | 0.0007 | 0.0001 [-0.0004, 0.0005] | 0.938 | 0.062 | 0.0003 / 0.0001 / 0.0001 |
| reach_0.5 | abs | 0.120 | 0.127 | -0.0066 [-0.088, 0.068] | 0.938 | 0.062 | 0.0003 / 0.0001 / 0.0001 |
| within_1 | cur | 0.021 | 0.0046 | 0.016 [0.010, 0.023] | 1.000 | 0.0000 | 0.0003 / 0.0001 / 0.0001 |
| within_1 | abs | 0.136 | 0.129 | 0.0080 [-0.074, 0.084] | 0.938 | 0.062 | 0.0003 / 0.0001 / 0.0001 |
| within_2 | cur | 0.061 | 0.012 | 0.048 [0.031, 0.065] | 1.000 | 0.0000 | 0.0003 / 0.0001 / 0.0001 |
| within_2 | abs | 0.169 | 0.133 | 0.036 [-0.049, 0.116] | 0.938 | 0.062 | 0.0003 / 0.0001 / 0.0001 |
| goal_area | cur | 0.044 | 0.023 | 0.022 [0.0025, 0.039] | 1.000 | 0.0000 | 0.0003 / 0.0001 / 0.0001 |
| goal_area | abs | 0.156 | 0.139 | 0.017 [-0.069, 0.101] | 0.938 | 0.062 | 0.0003 / 0.0001 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.997 | 0.0000 | 329 | 1.000 | 40 |
| reach_0.5_abs | 0.983 | 0.0000 | 329 | 1.000 | 40 |
| goal_area_cur | 0.918 | 0.0000 | 329 | 0.791 | 40 |
| goal_area_abs | 0.977 | 0.0000 | 329 | 1.000 | 40 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 0 | | | | |
| far_success_vs_short_timeout | 135 | 1.000 / 1.000 | 0.0014 / 0.213 | inf / inf | 1.000 / 1.000 |
| far_success_vs_far_timeout | 303 | 1.000 / 1.000 | 0.0015 / 0.239 | inf / inf | 0.939 / 0.979 |
| far_success_vs_short_success | 54 | 0.074 / 0.074 | -0.0013 / -0.253 | 0.517 / 0.448 | 0.630 / 0.130 |
| short_success_vs_short_death | 0 | | | | |
| short_success_vs_short_timeout | 65 | 1.000 / 1.000 | 0.0028 / 0.467 | inf / inf | 0.986 / 1.000 |
| success_vs_non_success | 556 | 1.000 / 1.000 | 0.0018 / 0.280 | inf / inf | 0.958 / 0.987 |


## Lineage 1

### all_start: 4276 anchors, weight 0.0794

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.185 | 0.035 | 0.047 | 0.653 | 0.162 | 231 | 0.00064 | 0.1042 | 0.0034 | 0.563 | 0.0167 | 0.1120 |
| mode | 0.208 | 0.057 | 0.073 | 0.638 | 0.155 | 236 | 0.00067 | 0.1101 | 0.0032 | 0.530 | 0.0174 | 0.1184 |
| sample0 | 0.201 | 0.053 | 0.066 | 0.638 | 0.161 | 238 | 0.00066 | 0.1079 | 0.0033 | 0.537 | 0.0168 | 0.1158 |
| sample1 | 0.202 | 0.054 | 0.070 | 0.640 | 0.158 | 237 | 0.00066 | 0.1082 | 0.0032 | 0.535 | 0.0169 | 0.1162 |
| sample2 | 0.200 | 0.054 | 0.068 | 0.638 | 0.162 | 239 | 0.00065 | 0.1062 | 0.0032 | 0.532 | 0.0170 | 0.1143 |
| sample3 | 0.198 | 0.054 | 0.071 | 0.636 | 0.166 | 242 | 0.00064 | 0.1051 | 0.0032 | 0.532 | 0.0166 | 0.1131 |

**R4 far-going vs short-going queries (PRIMARY): 277 anchors with both classes (weight 0.0053 of 0.0794); far-going query share 0.065, mixed-route share 0.0012.**

delta success (far - short) = 0.609 [0.561, 0.655]; success far 0.759 vs short 0.150.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0012 | 0.0005 | 0.0007 [0.0006, 0.0008] | 0.874 | 0.126 | 0.0046 / 0.0003 / 0.0003 |
| reach_0.5 | abs | 0.218 | 0.084 | 0.133 [0.109, 0.155] | 0.881 | 0.119 | 0.0046 / 0.0003 / 0.0003 |
| within_1 | cur | 0.031 | 0.0029 | 0.028 [0.025, 0.032] | 0.982 | 0.018 | 0.0046 / 0.0003 / 0.0003 |
| within_1 | abs | 0.240 | 0.085 | 0.155 [0.132, 0.177] | 0.906 | 0.094 | 0.0046 / 0.0003 / 0.0003 |
| within_2 | cur | 0.092 | 0.0077 | 0.085 [0.077, 0.092] | 1.000 | 0.0000 | 0.0046 / 0.0003 / 0.0003 |
| within_2 | abs | 0.289 | 0.088 | 0.201 [0.178, 0.223] | 0.950 | 0.050 | 0.0046 / 0.0003 / 0.0003 |
| goal_area | cur | 0.052 | 0.012 | 0.040 [0.033, 0.047] | 0.906 | 0.094 | 0.0046 / 0.0003 / 0.0003 |
| goal_area | abs | 0.258 | 0.090 | 0.168 [0.145, 0.191] | 0.914 | 0.086 | 0.0046 / 0.0003 / 0.0003 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.981 | 0.0000 | 6480 | 0.979 | 894 |
| reach_0.5_abs | 0.982 | 0.0000 | 6480 | 0.982 | 894 |
| goal_area_cur | 0.933 | 0.0000 | 6480 | 0.897 | 894 |
| goal_area_abs | 0.968 | 0.0000 | 6480 | 0.971 | 894 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 1295 | 1.000 / 1.000 | 0.0016 / 0.287 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 1129 | 1.000 / 1.000 | 0.0016 / 0.282 | inf / inf | 0.997 / 0.997 |
| far_success_vs_far_timeout | 2265 | 1.000 / 1.000 | 0.0017 / 0.298 | inf / inf | 0.868 / 0.920 |
| far_success_vs_short_success | 388 | 0.048 / 0.048 | -0.0017 / -0.275 | 0.491 / 0.512 | 0.364 / 0.060 |
| short_success_vs_short_death | 119 | 1.000 / 1.000 | 0.0031 / 0.556 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 3904 | 1.000 / 1.000 | 0.0035 / 0.588 | inf / inf | 0.988 / 0.999 |
| success_vs_non_success | 8885 | 1.000 / 1.000 | 0.0025 / 0.430 | inf / inf | 0.958 / 0.979 |

### reset: 196 anchors, weight 0.0038

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.176 | 0.031 | 0.062 | 0.617 | 0.207 | 271 | 0.00058 | 0.0980 | 0.0033 | 0.556 | 0.0139 | 0.1044 |
| mode | 0.482 | 0.390 | 0.520 | 0.337 | 0.181 | 403 | 0.00094 | 0.1619 | 0.0019 | 0.336 | 0.0332 | 0.1856 |
| sample0 | 0.447 | 0.344 | 0.460 | 0.348 | 0.205 | 389 | 0.00095 | 0.1688 | 0.0021 | 0.377 | 0.0280 | 0.1871 |
| sample1 | 0.454 | 0.377 | 0.489 | 0.357 | 0.189 | 391 | 0.00090 | 0.1589 | 0.0020 | 0.350 | 0.0334 | 0.1834 |
| sample2 | 0.441 | 0.333 | 0.430 | 0.368 | 0.192 | 379 | 0.00092 | 0.1629 | 0.0021 | 0.370 | 0.0278 | 0.1810 |
| sample3 | 0.401 | 0.335 | 0.513 | 0.344 | 0.256 | 417 | 0.00079 | 0.1408 | 0.0020 | 0.351 | 0.0257 | 0.1584 |

**R4 far-going vs short-going queries (PRIMARY): 148 anchors with both classes (weight 0.0029 of 0.0038); far-going query share 0.407, mixed-route share 0.0095.**

delta success (far - short) = 0.597 [0.532, 0.663]; success far 0.739 vs short 0.142.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0012 | 0.0005 | 0.0007 [0.0005, 0.0009] | 0.857 | 0.143 | 0.0026 / 0.0002 / 0.0001 |
| reach_0.5 | abs | 0.216 | 0.084 | 0.132 [0.101, 0.161] | 0.864 | 0.136 | 0.0026 / 0.0002 / 0.0001 |
| within_1 | cur | 0.030 | 0.0028 | 0.027 [0.023, 0.033] | 0.987 | 0.013 | 0.0026 / 0.0002 / 0.0001 |
| within_1 | abs | 0.239 | 0.085 | 0.154 [0.121, 0.185] | 0.890 | 0.110 | 0.0026 / 0.0002 / 0.0001 |
| within_2 | cur | 0.089 | 0.0075 | 0.082 [0.071, 0.092] | 1.000 | 0.0000 | 0.0026 / 0.0002 / 0.0001 |
| within_2 | abs | 0.285 | 0.087 | 0.198 [0.166, 0.228] | 0.942 | 0.058 | 0.0026 / 0.0002 / 0.0001 |
| goal_area | cur | 0.048 | 0.013 | 0.036 [0.026, 0.046] | 0.909 | 0.091 | 0.0026 / 0.0002 / 0.0001 |
| goal_area | abs | 0.254 | 0.090 | 0.164 [0.134, 0.193] | 0.896 | 0.104 | 0.0026 / 0.0002 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.944 | 0.0000 | 1266 | 0.918 | 147 |
| reach_0.5_abs | 0.955 | 0.0000 | 1266 | 0.936 | 147 |
| goal_area_cur | 0.895 | 0.0000 | 1266 | 0.813 | 147 |
| goal_area_abs | 0.947 | 0.0000 | 1266 | 0.942 | 147 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 847 | 1.000 / 1.000 | 0.0016 / 0.289 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 477 | 1.000 / 1.000 | 0.0016 / 0.289 | inf / inf | 0.995 / 0.995 |
| far_success_vs_far_timeout | 569 | 1.000 / 1.000 | 0.0016 / 0.290 | inf / inf | 0.900 / 0.951 |
| far_success_vs_short_success | 198 | 0.0089 / 0.0089 | -0.0018 / -0.278 | 0.487 / 0.529 | 0.308 / 0.0089 |
| short_success_vs_short_death | 10 | 1.000 / 1.000 | 0.0029 / 0.515 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 66 | 1.000 / 1.000 | 0.0034 / 0.585 | inf / inf | 1.000 / 1.000 |
| success_vs_non_success | 2078 | 1.000 / 1.000 | 0.0018 / 0.315 | inf / inf | 0.961 / 0.985 |

### t_1_30: 4037 anchors, weight 0.0749

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.181 | 0.029 | 0.038 | 0.662 | 0.158 | 226 | 0.00063 | 0.1033 | 0.0035 | 0.572 | 0.0163 | 0.1108 |
| mode | 0.189 | 0.035 | 0.043 | 0.659 | 0.151 | 225 | 0.00065 | 0.1065 | 0.0034 | 0.563 | 0.0162 | 0.1137 |
| sample0 | 0.184 | 0.032 | 0.039 | 0.659 | 0.157 | 228 | 0.00064 | 0.1037 | 0.0035 | 0.564 | 0.0160 | 0.1108 |
| sample1 | 0.184 | 0.032 | 0.041 | 0.661 | 0.155 | 226 | 0.00064 | 0.1044 | 0.0035 | 0.567 | 0.0158 | 0.1113 |
| sample2 | 0.183 | 0.035 | 0.042 | 0.659 | 0.159 | 229 | 0.00063 | 0.1025 | 0.0034 | 0.561 | 0.0161 | 0.1098 |
| sample3 | 0.182 | 0.033 | 0.041 | 0.658 | 0.160 | 230 | 0.00063 | 0.1023 | 0.0034 | 0.561 | 0.0156 | 0.1092 |

**R4 far-going vs short-going queries (PRIMARY): 110 anchors with both classes (weight 0.0020 of 0.0749); far-going query share 0.040, mixed-route share 0.0006.**

delta success (far - short) = 0.647 [0.565, 0.719]; success far 0.788 vs short 0.141.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0013 | 0.0004 | 0.0008 [0.0006, 0.0010] | 0.889 | 0.111 | 0.0018 / 0.0001 / 0.0001 |
| reach_0.5 | abs | 0.226 | 0.077 | 0.149 [0.114, 0.181] | 0.907 | 0.093 | 0.0018 / 0.0001 / 0.0001 |
| within_1 | cur | 0.033 | 0.0027 | 0.030 [0.024, 0.037] | 0.972 | 0.028 | 0.0018 / 0.0001 / 0.0001 |
| within_1 | abs | 0.251 | 0.078 | 0.172 [0.137, 0.202] | 0.935 | 0.065 | 0.0018 / 0.0001 / 0.0001 |
| within_2 | cur | 0.097 | 0.0071 | 0.090 [0.079, 0.102] | 1.000 | 0.0000 | 0.0018 / 0.0001 / 0.0001 |
| within_2 | abs | 0.300 | 0.080 | 0.220 [0.186, 0.254] | 0.963 | 0.037 | 0.0018 / 0.0001 / 0.0001 |
| goal_area | cur | 0.056 | 0.010 | 0.046 [0.034, 0.058] | 0.898 | 0.102 | 0.0018 / 0.0001 / 0.0001 |
| goal_area | abs | 0.270 | 0.082 | 0.188 [0.154, 0.218] | 0.935 | 0.065 | 0.0018 / 0.0001 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.990 | 0.0000 | 4904 | 0.991 | 709 |
| reach_0.5_abs | 0.989 | 0.0000 | 4904 | 0.991 | 709 |
| goal_area_cur | 0.947 | 0.0000 | 4904 | 0.922 | 709 |
| goal_area_abs | 0.976 | 0.0000 | 4904 | 0.980 | 709 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 448 | 1.000 / 1.000 | 0.0016 / 0.281 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 510 | 1.000 / 1.000 | 0.0016 / 0.287 | inf / inf | 0.998 / 0.998 |
| far_success_vs_far_timeout | 1374 | 1.000 / 1.000 | 0.0017 / 0.300 | inf / inf | 0.859 / 0.910 |
| far_success_vs_short_success | 141 | 0.020 / 0.020 | -0.0017 / -0.284 | 0.481 / 0.495 | 0.371 / 0.053 |
| short_success_vs_short_death | 109 | 1.000 / 1.000 | 0.0032 / 0.559 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 3789 | 1.000 / 1.000 | 0.0035 / 0.589 | inf / inf | 0.987 / 0.999 |
| success_vs_non_success | 6277 | 1.000 / 1.000 | 0.0028 / 0.480 | inf / inf | 0.961 / 0.980 |

### t_ge_30: 43 anchors, weight 0.0008

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.670 | 0.628 | 0.830 | 0.000 | 0.330 | 543 | 0.00127 | 0.2186 | 0.0019 | 0.326 | 0.0606 | 0.2648 |
| mode | 0.649 | 0.585 | 0.745 | 0.000 | 0.351 | 559 | 0.00121 | 0.2018 | 0.0019 | 0.311 | 0.0545 | 0.2420 |
| sample0 | 0.649 | 0.596 | 0.755 | 0.000 | 0.351 | 546 | 0.00123 | 0.2143 | 0.0019 | 0.330 | 0.0460 | 0.2465 |
| sample1 | 0.734 | 0.606 | 0.766 | 0.000 | 0.266 | 530 | 0.00140 | 0.2334 | 0.0019 | 0.318 | 0.0481 | 0.2672 |
| sample2 | 0.660 | 0.617 | 0.787 | 0.000 | 0.340 | 570 | 0.00116 | 0.1883 | 0.0018 | 0.285 | 0.0535 | 0.2271 |
| sample3 | 0.670 | 0.649 | 0.851 | 0.000 | 0.330 | 554 | 0.00122 | 0.2063 | 0.0018 | 0.308 | 0.0670 | 0.2598 |

**R4 far-going vs short-going queries (PRIMARY): 19 anchors with both classes (weight 0.0004 of 0.0008); far-going query share 0.780, mixed-route share 0.018.**

delta success (far - short) = 0.486 [0.269, 0.693]; success far 0.757 vs short 0.271.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0012 | 0.0008 | 0.0004 [-0.0002, 0.0009] | 0.938 | 0.062 | 0.0003 / 0.0000 / 0.0001 |
| reach_0.5 | abs | 0.179 | 0.127 | 0.052 [-0.063, 0.143] | 0.875 | 0.125 | 0.0003 / 0.0000 / 0.0001 |
| within_1 | cur | 0.024 | 0.0050 | 0.019 [0.014, 0.023] | 1.000 | 0.0000 | 0.0003 / 0.0000 / 0.0001 |
| within_1 | abs | 0.198 | 0.129 | 0.069 [-0.040, 0.162] | 0.875 | 0.125 | 0.0003 / 0.0000 / 0.0001 |
| within_2 | cur | 0.092 | 0.012 | 0.079 [0.062, 0.099] | 1.000 | 0.0000 | 0.0003 / 0.0000 / 0.0001 |
| within_2 | abs | 0.254 | 0.132 | 0.121 [0.0051, 0.218] | 0.938 | 0.062 | 0.0003 / 0.0000 / 0.0001 |
| goal_area | cur | 0.056 | 0.020 | 0.037 [0.020, 0.053] | 0.938 | 0.062 | 0.0003 / 0.0000 / 0.0001 |
| goal_area | abs | 0.225 | 0.136 | 0.089 [-0.020, 0.176] | 0.938 | 0.062 | 0.0003 / 0.0000 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 1.000 | 0.0000 | 310 | 1.000 | 38 |
| reach_0.5_abs | 1.000 | 0.0000 | 310 | 1.000 | 38 |
| goal_area_cur | 0.877 | 0.0000 | 310 | 0.780 | 38 |
| goal_area_abs | 0.943 | 0.0000 | 310 | 0.927 | 38 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 0 | | | | |
| far_success_vs_short_timeout | 142 | 1.000 / 1.000 | 0.0016 / 0.240 | inf / inf | 1.000 / 1.000 |
| far_success_vs_far_timeout | 322 | 1.000 / 1.000 | 0.0018 / 0.302 | inf / inf | 0.845 / 0.902 |
| far_success_vs_short_success | 49 | 0.271 / 0.271 | -0.0013 / -0.241 | 0.540 / 0.480 | 0.559 / 0.271 |
| short_success_vs_short_death | 0 | | | | |
| short_success_vs_short_timeout | 49 | 1.000 / 1.000 | 0.0033 / 0.537 | inf / inf | 1.000 / 1.000 |
| success_vs_non_success | 530 | 1.000 / 1.000 | 0.0019 / 0.312 | inf / inf | 0.905 / 0.942 |


## Lineage 2

### all_start: 4276 anchors, weight 0.0794

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.220 | 0.033 | 0.040 | 0.678 | 0.102 | 202 | 0.00072 | 0.1222 | 0.0033 | 0.555 | 0.0190 | 0.1315 |
| mode | 0.235 | 0.048 | 0.056 | 0.663 | 0.102 | 209 | 0.00075 | 0.1265 | 0.0032 | 0.538 | 0.0192 | 0.1359 |
| sample0 | 0.229 | 0.044 | 0.052 | 0.670 | 0.102 | 206 | 0.00073 | 0.1239 | 0.0032 | 0.541 | 0.0186 | 0.1328 |
| sample1 | 0.233 | 0.047 | 0.057 | 0.665 | 0.101 | 208 | 0.00074 | 0.1252 | 0.0032 | 0.537 | 0.0194 | 0.1348 |
| sample2 | 0.228 | 0.043 | 0.051 | 0.670 | 0.102 | 206 | 0.00073 | 0.1235 | 0.0032 | 0.543 | 0.0185 | 0.1323 |
| sample3 | 0.229 | 0.043 | 0.054 | 0.668 | 0.103 | 207 | 0.00073 | 0.1242 | 0.0032 | 0.542 | 0.0188 | 0.1333 |

**R4 far-going vs short-going queries (PRIMARY): 303 anchors with both classes (weight 0.0056 of 0.0794); far-going query share 0.050, mixed-route share 0.0027.**

delta success (far - short) = 0.679 [0.635, 0.720]; success far 0.836 vs short 0.157.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0014 | 0.0005 | 0.0009 [0.0008, 0.0010] | 0.911 | 0.089 | 0.0051 / 0.0003 / 0.0003 |
| reach_0.5 | abs | 0.255 | 0.083 | 0.173 [0.153, 0.190] | 0.915 | 0.085 | 0.0051 / 0.0003 / 0.0003 |
| within_1 | cur | 0.031 | 0.0028 | 0.028 [0.025, 0.031] | 0.987 | 0.013 | 0.0051 / 0.0003 / 0.0003 |
| within_1 | abs | 0.279 | 0.084 | 0.195 [0.175, 0.213] | 0.928 | 0.072 | 0.0051 / 0.0003 / 0.0003 |
| within_2 | cur | 0.088 | 0.0075 | 0.080 [0.073, 0.087] | 0.987 | 0.013 | 0.0051 / 0.0003 / 0.0003 |
| within_2 | abs | 0.324 | 0.086 | 0.238 [0.218, 0.257] | 0.974 | 0.026 | 0.0051 / 0.0003 / 0.0003 |
| goal_area | cur | 0.056 | 0.012 | 0.044 [0.037, 0.052] | 0.879 | 0.121 | 0.0051 / 0.0003 / 0.0003 |
| goal_area | abs | 0.301 | 0.089 | 0.212 [0.192, 0.231] | 0.941 | 0.059 | 0.0051 / 0.0003 / 0.0003 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.989 | 0.0000 | 9687 | 0.978 | 1386 |
| reach_0.5_abs | 0.989 | 0.0000 | 9687 | 0.982 | 1386 |
| goal_area_cur | 0.940 | 0.0000 | 9687 | 0.884 | 1386 |
| goal_area_abs | 0.971 | 0.0000 | 9687 | 0.956 | 1386 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 1372 | 1.000 / 1.000 | 0.0017 / 0.308 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 1479 | 1.000 / 1.000 | 0.0017 / 0.307 | inf / inf | 0.991 / 0.991 |
| far_success_vs_far_timeout | 1205 | 1.000 / 1.000 | 0.0017 / 0.301 | inf / inf | 0.678 / 0.792 |
| far_success_vs_short_success | 511 | 0.059 / 0.059 | -0.0013 / -0.227 | 0.560 / 0.565 | 0.252 / 0.065 |
| short_success_vs_short_death | 2311 | 1.000 / 1.000 | 0.0032 / 0.562 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 5877 | 1.000 / 1.000 | 0.0033 / 0.570 | inf / inf | 0.982 / 0.995 |
| success_vs_non_success | 12393 | 1.000 / 1.000 | 0.0028 / 0.482 | inf / inf | 0.956 / 0.977 |

### reset: 196 anchors, weight 0.0038

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.205 | 0.026 | 0.040 | 0.656 | 0.139 | 235 | 0.00065 | 0.1133 | 0.0032 | 0.553 | 0.0196 | 0.1246 |
| mode | 0.460 | 0.302 | 0.337 | 0.436 | 0.104 | 311 | 0.00106 | 0.1905 | 0.0023 | 0.414 | 0.0231 | 0.2042 |
| sample0 | 0.370 | 0.225 | 0.269 | 0.474 | 0.156 | 316 | 0.00087 | 0.1574 | 0.0024 | 0.425 | 0.0180 | 0.1670 |
| sample1 | 0.416 | 0.256 | 0.317 | 0.452 | 0.132 | 319 | 0.00097 | 0.1719 | 0.0023 | 0.413 | 0.0280 | 0.1899 |
| sample2 | 0.390 | 0.233 | 0.284 | 0.454 | 0.156 | 322 | 0.00093 | 0.1676 | 0.0024 | 0.430 | 0.0272 | 0.1850 |
| sample3 | 0.403 | 0.225 | 0.271 | 0.458 | 0.139 | 316 | 0.00096 | 0.1708 | 0.0024 | 0.424 | 0.0231 | 0.1839 |

**R4 far-going vs short-going queries (PRIMARY): 107 anchors with both classes (weight 0.0020 of 0.0038); far-going query share 0.244, mixed-route share 0.017.**

delta success (far - short) = 0.668 [0.587, 0.745]; success far 0.841 vs short 0.173.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0014 | 0.0006 | 0.0009 [0.0007, 0.0011] | 0.896 | 0.104 | 0.0018 / 0.0001 / 0.0002 |
| reach_0.5 | abs | 0.269 | 0.098 | 0.171 [0.133, 0.208] | 0.915 | 0.085 | 0.0018 / 0.0001 / 0.0002 |
| within_1 | cur | 0.028 | 0.0033 | 0.025 [0.020, 0.030] | 0.991 | 0.0094 | 0.0018 / 0.0001 / 0.0002 |
| within_1 | abs | 0.289 | 0.099 | 0.190 [0.149, 0.228] | 0.915 | 0.085 | 0.0018 / 0.0001 / 0.0002 |
| within_2 | cur | 0.078 | 0.0092 | 0.069 [0.057, 0.080] | 0.991 | 0.0094 | 0.0018 / 0.0001 / 0.0002 |
| within_2 | abs | 0.328 | 0.102 | 0.226 [0.185, 0.264] | 0.962 | 0.038 | 0.0018 / 0.0001 / 0.0002 |
| goal_area | cur | 0.044 | 0.014 | 0.031 [0.020, 0.042] | 0.821 | 0.179 | 0.0018 / 0.0001 / 0.0002 |
| goal_area | abs | 0.304 | 0.105 | 0.199 [0.162, 0.237] | 0.943 | 0.057 | 0.0018 / 0.0001 / 0.0002 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.953 | 0.0000 | 997 | 0.924 | 125 |
| reach_0.5_abs | 0.959 | 0.0000 | 997 | 0.959 | 125 |
| goal_area_cur | 0.871 | 0.0000 | 997 | 0.766 | 125 |
| goal_area_abs | 0.938 | 0.0000 | 997 | 0.938 | 125 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 669 | 1.000 / 1.000 | 0.0017 / 0.321 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 399 | 1.000 / 1.000 | 0.0017 / 0.319 | inf / inf | 0.991 / 0.991 |
| far_success_vs_far_timeout | 179 | 1.000 / 1.000 | 0.0016 / 0.295 | inf / inf | 0.735 / 0.779 |
| far_success_vs_short_success | 203 | 0.018 / 0.018 | -0.0016 / -0.269 | 0.512 / 0.522 | 0.166 / 0.018 |
| short_success_vs_short_death | 105 | 1.000 / 1.000 | 0.0030 / 0.536 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 169 | 1.000 / 1.000 | 0.0032 / 0.564 | inf / inf | 0.991 / 0.995 |
| success_vs_non_success | 1589 | 1.000 / 1.000 | 0.0020 / 0.371 | inf / inf | 0.958 / 0.973 |

### t_1_30: 4037 anchors, weight 0.0749

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.216 | 0.029 | 0.035 | 0.686 | 0.097 | 197 | 0.00072 | 0.1215 | 0.0033 | 0.561 | 0.0187 | 0.1305 |
| mode | 0.220 | 0.031 | 0.037 | 0.681 | 0.099 | 200 | 0.00072 | 0.1221 | 0.0033 | 0.555 | 0.0188 | 0.1311 |
| sample0 | 0.219 | 0.031 | 0.036 | 0.686 | 0.095 | 196 | 0.00072 | 0.1221 | 0.0033 | 0.558 | 0.0184 | 0.1308 |
| sample1 | 0.220 | 0.032 | 0.038 | 0.683 | 0.097 | 198 | 0.00072 | 0.1222 | 0.0033 | 0.555 | 0.0186 | 0.1310 |
| sample2 | 0.216 | 0.028 | 0.034 | 0.688 | 0.096 | 197 | 0.00072 | 0.1207 | 0.0033 | 0.559 | 0.0179 | 0.1289 |
| sample3 | 0.216 | 0.028 | 0.036 | 0.686 | 0.098 | 198 | 0.00072 | 0.1210 | 0.0033 | 0.560 | 0.0181 | 0.1295 |

**R4 far-going vs short-going queries (PRIMARY): 169 anchors with both classes (weight 0.0031 of 0.0749); far-going query share 0.035, mixed-route share 0.0013.**

delta success (far - short) = 0.704 [0.649, 0.756]; success far 0.834 vs short 0.130.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0014 | 0.0004 | 0.0010 [0.0009, 0.0011] | 0.924 | 0.076 | 0.0029 / 0.0001 / 0.0001 |
| reach_0.5 | abs | 0.252 | 0.069 | 0.183 [0.159, 0.207] | 0.936 | 0.064 | 0.0029 / 0.0001 / 0.0001 |
| within_1 | cur | 0.031 | 0.0023 | 0.029 [0.025, 0.033] | 0.982 | 0.018 | 0.0029 / 0.0001 / 0.0001 |
| within_1 | abs | 0.275 | 0.070 | 0.205 [0.182, 0.227] | 0.942 | 0.058 | 0.0029 / 0.0001 / 0.0001 |
| within_2 | cur | 0.091 | 0.0060 | 0.085 [0.077, 0.094] | 0.982 | 0.018 | 0.0029 / 0.0001 / 0.0001 |
| within_2 | abs | 0.324 | 0.072 | 0.253 [0.228, 0.276] | 0.982 | 0.018 | 0.0029 / 0.0001 / 0.0001 |
| goal_area | cur | 0.060 | 0.010 | 0.050 [0.040, 0.061] | 0.912 | 0.088 | 0.0029 / 0.0001 / 0.0001 |
| goal_area | abs | 0.302 | 0.075 | 0.227 [0.204, 0.250] | 0.953 | 0.047 | 0.0029 / 0.0001 / 0.0001 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 0.993 | 0.0000 | 8340 | 0.983 | 1221 |
| reach_0.5_abs | 0.993 | 0.0000 | 8340 | 0.984 | 1221 |
| goal_area_cur | 0.951 | 0.0000 | 8340 | 0.901 | 1221 |
| goal_area_abs | 0.976 | 0.0000 | 8340 | 0.958 | 1221 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 697 | 1.000 / 1.000 | 0.0016 / 0.296 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 855 | 1.000 / 1.000 | 0.0017 / 0.311 | inf / inf | 0.991 / 0.991 |
| far_success_vs_far_timeout | 831 | 1.000 / 1.000 | 0.0017 / 0.304 | inf / inf | 0.626 / 0.766 |
| far_success_vs_short_success | 232 | 0.060 / 0.060 | -0.0012 / -0.210 | 0.581 / 0.597 | 0.300 / 0.060 |
| short_success_vs_short_death | 2204 | 1.000 / 1.000 | 0.0032 / 0.563 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 5583 | 1.000 / 1.000 | 0.0033 / 0.572 | inf / inf | 0.981 / 0.995 |
| success_vs_non_success | 10230 | 1.000 / 1.000 | 0.0029 / 0.508 | inf / inf | 0.957 / 0.978 |

### t_ge_30: 43 anchors, weight 0.0008

**R1 per query (anchor-weighted).**

| query | success | completed far | entered far | death | timeout | length | reach_0.5 cur | reach_0.5 abs | reach_0.5 cur given success | reach_0.5 abs given success | goal_area cur | goal_area abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.638 | 0.511 | 0.564 | 0.000 | 0.362 | 535 | 0.00137 | 0.2332 | 0.0021 | 0.365 | 0.0494 | 0.2660 |
| mode | 0.628 | 0.447 | 0.553 | 0.000 | 0.372 | 530 | 0.00136 | 0.2377 | 0.0022 | 0.379 | 0.0431 | 0.2685 |
| sample0 | 0.500 | 0.394 | 0.489 | 0.021 | 0.479 | 601 | 0.00088 | 0.1399 | 0.0018 | 0.280 | 0.0355 | 0.1652 |
| sample1 | 0.585 | 0.468 | 0.638 | 0.000 | 0.415 | 576 | 0.00111 | 0.1841 | 0.0019 | 0.315 | 0.0541 | 0.2265 |
| sample2 | 0.564 | 0.500 | 0.628 | 0.021 | 0.415 | 563 | 0.00107 | 0.1828 | 0.0019 | 0.324 | 0.0371 | 0.2103 |
| sample3 | 0.628 | 0.532 | 0.734 | 0.000 | 0.372 | 562 | 0.00117 | 0.1982 | 0.0019 | 0.316 | 0.0610 | 0.2478 |

**R4 far-going vs short-going queries (PRIMARY): 27 anchors with both classes (weight 0.0005 of 0.0008); far-going query share 0.571, mixed-route share 0.060.**

delta success (far - short) = 0.573 [0.447, 0.683]; success far 0.825 vs short 0.252.

| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |
|---|---|---:|---:|---|---:|---:|---|
| reach_0.5 | cur | 0.0014 | 0.0006 | 0.0007 [0.0004, 0.0010] | 0.893 | 0.107 | 0.0005 / 0.0000 / 0.0000 |
| reach_0.5 | abs | 0.223 | 0.106 | 0.117 [0.062, 0.170] | 0.786 | 0.214 | 0.0005 / 0.0000 / 0.0000 |
| within_1 | cur | 0.041 | 0.0037 | 0.038 [0.025, 0.053] | 1.000 | 0.0000 | 0.0005 / 0.0000 / 0.0000 |
| within_1 | abs | 0.257 | 0.108 | 0.150 [0.100, 0.195] | 0.893 | 0.107 | 0.0005 / 0.0000 / 0.0000 |
| within_2 | cur | 0.104 | 0.0095 | 0.094 [0.072, 0.123] | 1.000 | 0.0000 | 0.0005 / 0.0000 / 0.0000 |
| within_2 | abs | 0.309 | 0.111 | 0.198 [0.150, 0.241] | 0.964 | 0.036 | 0.0005 / 0.0000 / 0.0000 |
| goal_area | cur | 0.079 | 0.017 | 0.062 [0.037, 0.092] | 0.893 | 0.107 | 0.0005 / 0.0000 / 0.0000 |
| goal_area | abs | 0.289 | 0.116 | 0.174 [0.119, 0.220] | 0.857 | 0.143 | 0.0005 / 0.0000 / 0.0000 |

**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**

| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |
|---|---:|---:|---:|---:|---:|
| reach_0.5_cur | 1.000 | 0.0000 | 350 | 1.000 | 40 |
| reach_0.5_abs | 0.992 | 0.0000 | 350 | 1.000 | 40 |
| goal_area_cur | 0.881 | 0.0000 | 350 | 0.744 | 40 |
| goal_area_abs | 0.947 | 0.0000 | 350 | 0.953 | 40 |

**R2 within-(anchor, draw) pairs by outcome class.**

| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | goal_area: P(good > bad) cur / abs |
|---|---:|---|---|---|---|
| far_success_vs_short_death | 6 | 1.000 / 1.000 | 0.0016 / 0.266 | inf / inf | 1.000 / 1.000 |
| far_success_vs_short_timeout | 225 | 1.000 / 1.000 | 0.0016 / 0.268 | inf / inf | 0.992 / 0.992 |
| far_success_vs_far_timeout | 195 | 1.000 / 1.000 | 0.0017 / 0.295 | inf / inf | 0.844 / 0.915 |
| far_success_vs_short_success | 76 | 0.171 / 0.171 | -0.0008 / -0.167 | 0.652 / 0.604 | 0.341 / 0.207 |
| short_success_vs_short_death | 2 | 1.000 / 1.000 | 0.0013 / 0.165 | inf / inf | 1.000 / 1.000 |
| short_success_vs_short_timeout | 125 | 1.000 / 1.000 | 0.0029 / 0.494 | inf / inf | 0.985 / 1.000 |
| success_vs_non_success | 574 | 1.000 / 1.000 | 0.0020 / 0.330 | inf / inf | 0.926 / 0.963 |

## Judgement (manifest rule; reset stratum, reach_0.5)

| lineage | far better | delta success | delta mass cur | P agree cur | delta mass abs | P agree abs | J1 | J2 |
|---|---|---|---|---:|---|---:|---|---|
| seed_0 | True | 0.542 [0.482, 0.600] | 0.0006 [0.0004, 0.0007] | 0.863 | 0.108 [0.081, 0.133] | 0.851 | False | False |
| seed_1 | True | 0.597 [0.532, 0.663] | 0.0007 [0.0005, 0.0009] | 0.857 | 0.132 [0.101, 0.161] | 0.864 | False | True |
| seed_2 | True | 0.668 [0.587, 0.745] | 0.0009 [0.0007, 0.0011] | 0.896 | 0.171 [0.133, 0.208] | 0.915 | False | True |

J1 and J2 in every lineage: **False**.
