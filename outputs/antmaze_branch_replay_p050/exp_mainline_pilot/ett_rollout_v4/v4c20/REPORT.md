# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.306 | 0.615 / 0.665 | 0.091 / 0.029 | 0.012 / 0.012 | 0.87 / 0.82 (71) | 0.023 | 0.062 | 0.059 | 41 / 40 | 100 / 107 | 0.991 | NO: reach, timeout |
| anchor in start | 471 | 0.724 / 0.722 | 0.248 / 0.264 | 0.028 / 0.015 | 0.030 / 0.034 | 0.93 / 0.69 (14) | 0.037 | 0.105 | 0.165 | 61 / 62 | 217 / 219 | 0.997 | NO: ks_reach_time |
| anchor in pre_zone1 | 1456 | 0.503 / 0.534 | 0.319 / 0.422 | 0.178 / 0.044 | 0.000 / 0.000 | - / - (0) | 0.043 | 0.074 | 0.143 | 38 / 38 | 187 / 196 | 0.966 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.440 | 0.550 / 0.557 | 0.024 / 0.003 | 0.000 / 0.000 | - / - (0) | 0.099 | 0.083 | 0.131 | 53 / 49 | 151 / 153 | 0.995 | yes |
| anchor in between | 1366 | 0.283 / 0.295 | 0.616 / 0.686 | 0.102 / 0.019 | 0.000 / 0.000 | - / - (0) | 0.074 | 0.151 | 0.066 | 27 / 25 | 115 / 118 | 0.989 | NO: reach, timeout, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.107 | 0.857 / 0.876 | 0.039 / 0.017 | 0.000 / 0.000 | - / - (0) | 0.222 | 0.181 | 0.034 | 9 / 8 | 80 / 80 | 0.996 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.980 | 0.042 / 0.018 | 0.000 / 0.000 | - / - (0) | - | - | 0.053 | None / 12 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.959 | 0.043 / 0.041 | 0.000 / 0.000 | - / - (0) | - | - | 0.045 | None / 11 | 10 / 11 | - | yes |

## Advice C_decision: **ALL GATES PASSED** (4396 anchors, 4396 paths) -- the decision-state anchor set (reset / start_early / pre_zone1_early; a stratum below the sealed 200-anchor minimum is reported, not gated)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 4396 | 0.737 / 0.738 | 0.246 / 0.253 | 0.017 / 0.010 | 0.020 / 0.018 | 0.80 / 0.74 (86) | 0.016 | 0.084 | 0.050 | 57 / 57 | 208 / 210 | 1.000 | yes |
| reset | 196 | 0.663 / 0.665 | 0.296 / 0.323 | 0.041 / 0.012 | 0.015 / 0.000 | 0.67 / - (3) | 0.120 | 0.117 | 0.285 | 68 / 70 | 226 / 232 | 0.992 | NO: ks_reach_time (not gated: below the anchor minimum) |
| start_early | 2100 | 0.732 / 0.734 | 0.247 / 0.249 | 0.021 / 0.017 | 0.040 / 0.037 | 0.81 / 0.74 (83) | 0.031 | 0.086 | 0.084 | 60 / 61 | 218 / 220 | 1.000 | yes |
| pre_zone1_early | 2100 | 0.748 / 0.748 | 0.241 / 0.249 | 0.010 / 0.002 | 0.000 / 0.000 | - / - (0) | 0.026 | 0.083 | 0.079 | 40 / 41 | 194 / 195 | 1.000 | yes |

## Advice B: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.310 | 0.615 / 0.661 | 0.091 / 0.029 | 0.012 / 0.012 | 0.87 / 0.81 (284) | 0.030 | 0.059 | 0.052 | 41 / 41 | 100 / 107 | 0.795 | NO: timeout |
| anchor in start | 471 | 0.724 / 0.716 | 0.248 / 0.271 | 0.028 / 0.012 | 0.030 / 0.033 | 0.93 / 0.77 (56) | 0.055 | 0.107 | 0.064 | 61 / 62 | 217 / 217 | 0.500 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.541 | 0.319 / 0.414 | 0.178 / 0.045 | 0.000 / 0.000 | - / - (0) | 0.055 | 0.094 | 0.135 | 38 / 39 | 187 / 197 | 0.687 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.446 | 0.550 / 0.549 | 0.024 / 0.005 | 0.000 / 0.000 | - / - (0) | 0.072 | 0.111 | 0.082 | 53 / 53 | 151 / 152 | 0.676 | yes |
| anchor in between | 1366 | 0.283 / 0.301 | 0.616 / 0.678 | 0.102 / 0.020 | 0.000 / 0.000 | - / - (0) | 0.072 | 0.133 | 0.063 | 27 / 25 | 115 / 118 | 0.596 | NO: reach, timeout |
| anchor in zone2 | 544 | 0.105 / 0.114 | 0.857 / 0.871 | 0.039 / 0.015 | 0.000 / 0.000 | - / - (0) | 0.201 | 0.204 | 0.035 | 9 / 8 | 80 / 80 | 0.692 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.001 | 0.958 / 0.982 | 0.042 / 0.017 | 0.000 / 0.000 | - / - (0) | - | - | 0.052 | None / 15 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.959 | 0.043 / 0.041 | 0.000 / 0.000 | - / - (0) | - | - | 0.045 | None / 5 | 10 / 11 | - | yes |

## Advice B_decision: **ALL GATES PASSED** (4396 anchors, 17584 paths) -- the decision-state anchor set (reset / start_early / pre_zone1_early; a stratum below the sealed 200-anchor minimum is reported, not gated)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | far completion sim / model (n far sim) | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 4396 | 0.737 / 0.737 | 0.246 / 0.255 | 0.017 / 0.008 | 0.020 / 0.018 | 0.80 / 0.84 (344) | 0.018 | 0.090 | 0.065 | 57 / 57 | 208 / 211 | 0.518 | yes |
| reset | 196 | 0.663 / 0.733 | 0.296 / 0.254 | 0.041 / 0.013 | 0.015 / 0.000 | 0.67 / - (12) | 0.114 | 0.138 | 0.095 | 68 / 69 | 226 / 227 | 0.470 | NO: death (not gated: below the anchor minimum) |
| start_early | 2100 | 0.732 / 0.721 | 0.247 / 0.268 | 0.021 / 0.012 | 0.040 / 0.037 | 0.81 / 0.84 (332) | 0.038 | 0.099 | 0.072 | 60 / 61 | 218 / 219 | 0.541 | yes |
| pre_zone1_early | 2100 | 0.748 / 0.754 | 0.241 / 0.241 | 0.010 / 0.004 | 0.000 / 0.000 | - / - (0) | 0.029 | 0.082 | 0.130 | 40 / 40 | 194 / 197 | 0.499 | yes |

