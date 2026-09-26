# AntMaze V6 Phase 1: oracle branch replay

Critic recipe: V6 vanilla at gamma 0.999, 100,000 updates, anchors = row 0 of every branch path; joint stage: 100,000 warm-started joint updates with balanced BC rows (displacement key, cell 4.0, cap 0.25) from the recorded data; evaluation 300 natural draws, seed 909.

Law target on the replay (start region, t <= 5): r0.5 +0.60, r1.0 +0.62.

| arm | seed | gate margin | policy | success | failure | timeout | detour | shortcut | success U00/U10/U01/U11 | discounted (g 0.99) | mouth 1 / 2 median step | after-burst share z1 / z2 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|---|
| branch joint | 0 | +0.165 | mean | 0.603 | 0.027 | 0.370 | 0.003 | 0.790 | 0.727 / 0.568 / 0.574 / 0.537 | 0.019 | 156.000 / 240.000 | 0.926 / 0.889 |
| branch joint | 0 | +0.165 | sample | 0.337 | 0.000 | 0.663 | 0.250 | 0.477 | 0.403 / 0.341 / 0.294 / 0.298 | 0.003 | 285.000 / 447.000 | 1.000 / 0.978 |
| branch joint | 1 | +0.276 | mean | 0.453 | 0.033 | 0.513 | 0.017 | 0.543 | 0.545 / 0.477 / 0.353 / 0.418 | 0.015 | 143.000 / 222.000 | 0.837 / 0.831 |
| branch joint | 1 | +0.276 | sample | 0.450 | 0.023 | 0.527 | 0.050 | 0.563 | 0.468 / 0.375 / 0.515 / 0.463 | 0.013 | 158.000 / 254.000 | 0.926 / 0.896 |
| branch joint | 2 | -0.261 | mean | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | None / None | None / None |
| branch joint | 2 | -0.261 | sample | 0.000 | 0.000 | 1.000 | 0.077 | 0.040 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | 409.500 / None | 1.000 / None |
| branch joint | 3 | -0.056 | mean | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | None / None | None / None |
| branch joint | 3 | -0.056 | sample | 0.000 | 0.000 | 1.000 | 0.243 | 0.140 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | 354.500 / 646.000 | 1.000 / 1.000 |
| branch joint | 4 | -0.164 | mean | 0.093 | 0.027 | 0.880 | 0.007 | 0.160 | 0.169 / 0.102 / 0.059 / 0.030 | 0.004 | 122.000 / 201.000 | 0.667 / 0.750 |
| branch joint | 4 | -0.164 | sample | 0.107 | 0.003 | 0.890 | 0.120 | 0.187 | 0.143 / 0.091 / 0.118 / 0.075 | 0.003 | 186.500 / 262.000 | 0.917 / 1.000 |
| vanilla g0.999 | 0 | +nan | mean | 0.260 | 0.730 | 0.010 | 0.003 | 0.970 | 1.000 / 0.011 / 0.000 / 0.000 | 0.028 | 51.000 / 123.000 | 0.000 / 0.000 |
| vanilla g0.999 | 0 | +nan | sample | 0.260 | 0.737 | 0.003 | 0.003 | 0.960 | 1.000 / 0.000 / 0.000 / 0.015 | 0.027 | 52.000 / 123.000 | 0.007 / 0.014 |
| vanilla g0.999 | 1 | +nan | mean | 0.260 | 0.737 | 0.003 | 0.007 | 0.960 | 1.000 / 0.000 / 0.015 / 0.000 | 0.028 | 51.000 / 123.000 | 0.000 / 0.000 |
| vanilla g0.999 | 1 | +nan | sample | 0.260 | 0.733 | 0.007 | 0.017 | 0.960 | 1.000 / 0.000 / 0.000 / 0.015 | 0.027 | 52.000 / 123.000 | 0.000 / 0.000 |
| vanilla g0.999 | 2 | +nan | mean | 0.260 | 0.733 | 0.007 | 0.007 | 0.967 | 1.000 / 0.011 / 0.000 / 0.000 | 0.028 | 51.000 / 123.000 | 0.000 / 0.000 |
| vanilla g0.999 | 2 | +nan | sample | 0.260 | 0.723 | 0.017 | 0.003 | 0.970 | 0.987 / 0.000 / 0.029 / 0.000 | 0.027 | 52.000 / 124.000 | 0.000 / 0.015 |

Reference: vanilla at gamma 0.99 on the p040 benchmark (notes/v6_detour_ladder.md): success 0.370, detour 0.000 on every seed.  The rockfall clocks run from the reset (zone 1 closes by step 117, zone 2 by 197); an after-burst share near one means the shortcut survives by lateness, not by a route decision.
