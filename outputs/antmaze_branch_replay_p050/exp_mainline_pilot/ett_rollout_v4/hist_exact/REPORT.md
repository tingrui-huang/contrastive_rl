# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.304 | 0.615 / 0.649 | 0.091 / 0.047 | 0.012 / 0.012 | 0.029 | 0.056 | 0.037 | 41 / 40 | 100 / 105 | 0.995 | yes |
| anchor in start | 471 | 0.724 / 0.724 | 0.248 / 0.252 | 0.028 / 0.023 | 0.030 / 0.030 | 0.041 | 0.100 | 0.144 | 61 / 62 | 217 / 220 | 1.000 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.525 | 0.319 / 0.410 | 0.178 / 0.065 | 0.000 / 0.000 | 0.018 | 0.075 | 0.157 | 38 / 37 | 187 / 196 | 0.986 | NO: reach, timeout, ks_reach_time |
| anchor in zone1 | 580 | 0.426 / 0.435 | 0.550 / 0.543 | 0.024 / 0.022 | 0.000 / 0.000 | 0.095 | 0.089 | 0.144 | 53 / 49 | 151 / 155 | 0.995 | yes |
| anchor in between | 1366 | 0.283 / 0.295 | 0.616 / 0.667 | 0.102 / 0.039 | 0.000 / 0.000 | 0.074 | 0.146 | 0.060 | 27 / 25 | 115 / 117 | 0.990 | NO: reach, timeout |
| anchor in zone2 | 544 | 0.105 / 0.106 | 0.857 / 0.850 | 0.039 / 0.045 | 0.000 / 0.000 | 0.240 | 0.202 | 0.067 | 9 / 8 | 80 / 81 | 0.996 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.967 | 0.042 / 0.031 | 0.000 / 0.000 | - | - | 0.038 | None / 13 | 48 / 48 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.959 | 0.043 / 0.041 | 0.000 / 0.000 | - | - | 0.030 | None / 4 | 10 / 10 | - | yes |

## Advice B: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.310 | 0.615 / 0.640 | 0.091 / 0.050 | 0.012 / 0.012 | 0.025 | 0.064 | 0.036 | 41 / 41 | 100 / 105 | 0.803 | yes |
| anchor in start | 471 | 0.724 / 0.718 | 0.248 / 0.263 | 0.028 / 0.019 | 0.030 / 0.030 | 0.061 | 0.114 | 0.100 | 61 / 62 | 217 / 219 | 0.512 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.534 | 0.319 / 0.402 | 0.178 / 0.065 | 0.000 / 0.000 | 0.038 | 0.100 | 0.158 | 38 / 39 | 187 / 198 | 0.710 | NO: reach, timeout, ks_reach_time |
| anchor in zone1 | 580 | 0.426 / 0.451 | 0.550 / 0.537 | 0.024 / 0.013 | 0.000 / 0.000 | 0.064 | 0.117 | 0.089 | 53 / 53 | 151 / 152 | 0.677 | yes |
| anchor in between | 1366 | 0.283 / 0.308 | 0.616 / 0.645 | 0.102 / 0.047 | 0.000 / 0.000 | 0.060 | 0.117 | 0.071 | 27 / 25 | 115 / 119 | 0.608 | NO: timeout |
| anchor in zone2 | 544 | 0.105 / 0.110 | 0.857 / 0.852 | 0.039 / 0.039 | 0.000 / 0.000 | 0.197 | 0.203 | 0.066 | 9 / 9 | 80 / 81 | 0.704 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.001 | 0.958 / 0.965 | 0.042 / 0.034 | 0.000 / 0.000 | - | - | 0.038 | None / 16 | 48 / 48 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.957 | 0.043 / 0.043 | 0.000 / 0.000 | - | - | 0.028 | None / 5 | 10 / 10 | - | yes |

