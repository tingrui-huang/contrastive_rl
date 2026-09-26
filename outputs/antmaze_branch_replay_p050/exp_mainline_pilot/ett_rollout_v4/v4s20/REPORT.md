# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.305 | 0.615 / 0.631 | 0.091 / 0.063 | 0.012 / 0.012 | 0.87 / 0.85 (71) | 0.024 | 0.051 | 0.026 | 41 / 40 | 100 / 103 | 0.989 | yes |
| anchor in start | 471 | 0.724 / 0.723 | 0.248 / 0.259 | 0.028 / 0.018 | 0.030 / 0.030 | 0.93 / 0.85 (14) | 0.036 | 0.103 | 0.072 | 61 / 60 | 217 / 215 | 0.997 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.528 | 0.319 / 0.347 | 0.178 / 0.125 | 0.000 / 0.000 | - / - (0) | 0.041 | 0.075 | 0.089 | 38 / 38 | 187 / 189 | 0.958 | NO: timeout |
| anchor in zone1 | 580 | 0.426 / 0.449 | 0.550 / 0.537 | 0.024 / 0.014 | 0.000 / 0.000 | - / - (0) | 0.094 | 0.076 | 0.073 | 53 / 50 | 151 / 151 | 0.996 | yes |
| anchor in between | 1366 | 0.283 / 0.294 | 0.616 / 0.645 | 0.102 / 0.061 | 0.000 / 0.000 | - / - (0) | 0.081 | 0.111 | 0.055 | 27 / 25 | 115 / 117 | 0.987 | yes |
| anchor in zone2 | 544 | 0.105 / 0.108 | 0.857 / 0.865 | 0.039 / 0.027 | 0.000 / 0.000 | - / - (0) | 0.240 | 0.187 | 0.065 | 9 / 8 | 80 / 81 | 0.996 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.975 | 0.042 / 0.023 | 0.000 / 0.000 | - / - (0) | - | - | 0.060 | None / 12 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.953 | 0.043 / 0.047 | 0.000 / 0.000 | - / - (0) | - | - | 0.021 | None / 101 | 10 / 10 | - | yes |

## Advice C_decision: **ALL GATES PASSED** (4396 anchors, 4396 paths) -- the decision-state anchor set (reset / start_early / pre_zone1_early; a stratum below the sealed 200-anchor minimum is reported, not gated)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 4396 | 0.737 / 0.737 | 0.246 / 0.255 | 0.017 / 0.007 | 0.020 / 0.017 | 0.80 / 0.88 (86) | 0.015 | 0.090 | 0.037 | 57 / 56 | 208 / 208 | 1.000 | yes |
| reset | 196 | 0.663 / 0.663 | 0.296 / 0.327 | 0.041 / 0.010 | 0.015 / 0.005 | 0.67 / 0.92 (3) | 0.052 | 0.148 | 0.170 | 68 / 68 | 226 / 226 | 1.000 | NO: ks_reach_time (not gated: below the anchor minimum) |
| start_early | 2100 | 0.732 / 0.734 | 0.247 / 0.256 | 0.021 / 0.010 | 0.040 / 0.036 | 0.81 / 0.88 (83) | 0.024 | 0.093 | 0.075 | 60 / 60 | 218 / 217 | 0.999 | yes |
| pre_zone1_early | 2100 | 0.748 / 0.748 | 0.241 / 0.248 | 0.010 / 0.004 | 0.000 / 0.000 | - / - (0) | 0.029 | 0.088 | 0.069 | 40 / 40 | 194 / 193 | 1.000 | yes |

## Advice B: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.310 | 0.615 / 0.629 | 0.091 / 0.062 | 0.012 / 0.012 | 0.87 / 0.85 (284) | 0.031 | 0.064 | 0.027 | 41 / 41 | 100 / 103 | 0.795 | yes |
| anchor in start | 471 | 0.724 / 0.716 | 0.248 / 0.265 | 0.028 / 0.019 | 0.030 / 0.030 | 0.93 / 0.78 (56) | 0.048 | 0.109 | 0.094 | 61 / 61 | 217 / 215 | 0.508 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.538 | 0.319 / 0.337 | 0.178 / 0.125 | 0.000 / 0.000 | - / - (0) | 0.050 | 0.097 | 0.070 | 38 / 39 | 187 / 189 | 0.702 | NO: timeout |
| anchor in zone1 | 580 | 0.426 / 0.455 | 0.550 / 0.535 | 0.024 / 0.010 | 0.000 / 0.000 | - / - (0) | 0.071 | 0.121 | 0.079 | 53 / 52 | 151 / 152 | 0.671 | yes |
| anchor in between | 1366 | 0.283 / 0.301 | 0.616 / 0.643 | 0.102 / 0.056 | 0.000 / 0.000 | - / - (0) | 0.061 | 0.127 | 0.050 | 27 / 25 | 115 / 118 | 0.612 | yes |
| anchor in zone2 | 544 | 0.105 / 0.112 | 0.857 / 0.860 | 0.039 / 0.028 | 0.000 / 0.000 | - / - (0) | 0.203 | 0.195 | 0.071 | 9 / 8 | 80 / 80 | 0.682 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.001 | 0.958 / 0.969 | 0.042 / 0.030 | 0.000 / 0.000 | - / - (0) | - | - | 0.051 | None / 14 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.953 | 0.043 / 0.047 | 0.000 / 0.000 | - / - (0) | - | - | 0.021 | None / 5 | 10 / 10 | - | yes |

## Advice B_decision: **ALL GATES PASSED** (4396 anchors, 17584 paths) -- the decision-state anchor set (reset / start_early / pre_zone1_early; a stratum below the sealed 200-anchor minimum is reported, not gated)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 4396 | 0.737 / 0.738 | 0.246 / 0.252 | 0.017 / 0.011 | 0.020 / 0.017 | 0.80 / 0.77 (344) | 0.023 | 0.094 | 0.031 | 57 / 55 | 208 / 208 | 0.517 | yes |
| reset | 196 | 0.663 / 0.738 | 0.296 / 0.255 | 0.041 / 0.006 | 0.015 / 0.005 | 0.67 / 0.97 (12) | 0.065 | 0.156 | 0.251 | 68 / 67 | 226 / 222 | 0.472 | NO: death, ks_death_x, ks_reach_time (not gated: below the anchor minimum) |
| start_early | 2100 | 0.732 / 0.721 | 0.247 / 0.264 | 0.021 / 0.016 | 0.040 / 0.036 | 0.81 / 0.77 (332) | 0.023 | 0.102 | 0.069 | 60 / 60 | 218 / 216 | 0.540 | yes |
| pre_zone1_early | 2100 | 0.748 / 0.754 | 0.241 / 0.239 | 0.010 / 0.006 | 0.000 / 0.000 | - / - (0) | 0.035 | 0.087 | 0.062 | 40 / 39 | 194 / 194 | 0.497 | yes |

