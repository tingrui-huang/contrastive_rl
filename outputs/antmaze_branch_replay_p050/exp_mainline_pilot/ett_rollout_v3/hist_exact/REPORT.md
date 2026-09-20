# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.303 | 0.615 / 0.637 | 0.091 / 0.059 | 0.012 / 0.012 | 0.027 | 0.064 | 0.035 | 41 / 40 | 100 / 104 | 0.995 | yes |
| anchor in start | 471 | 0.724 / 0.726 | 0.248 / 0.246 | 0.028 / 0.028 | 0.030 / 0.028 | 0.038 | 0.083 | 0.097 | 61 / 62 | 217 / 219 | 1.000 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.527 | 0.319 / 0.417 | 0.178 / 0.056 | 0.000 / 0.000 | 0.018 | 0.056 | 0.137 | 38 / 38 | 187 / 195 | 0.985 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.437 | 0.550 / 0.543 | 0.024 / 0.021 | 0.000 / 0.000 | 0.099 | 0.096 | 0.111 | 53 / 48 | 151 / 153 | 0.994 | yes |
| anchor in between | 1366 | 0.283 / 0.291 | 0.616 / 0.649 | 0.102 / 0.060 | 0.000 / 0.000 | 0.069 | 0.153 | 0.049 | 27 / 25 | 115 / 118 | 0.993 | NO: ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.107 | 0.857 / 0.825 | 0.039 / 0.068 | 0.000 / 0.000 | 0.241 | 0.200 | 0.028 | 9 / 8 | 80 / 79 | 0.996 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.940 | 0.042 / 0.058 | 0.000 / 0.000 | - | - | 0.030 | None / 13 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.955 | 0.043 / 0.045 | 0.000 / 0.000 | - | - | 0.028 | None / None | 10 / 10 | - | yes |

## Advice A: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.036 | 0.615 / 0.852 | 0.091 / 0.112 | 0.012 / 0.012 | 0.338 | 0.513 | 0.157 | 41 / 77 | 100 / 131 | 0.687 | NO: death, reach, ks_death_time, ks_death_x, ks_reach_time |
| anchor in start | 471 | 0.724 / 0.067 | 0.248 / 0.822 | 0.028 / 0.111 | 0.030 / 0.028 | 0.488 | 0.610 | 0.094 | 61 / 128 | 217 / 216 | 0.495 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.078 | 0.319 / 0.813 | 0.178 / 0.109 | 0.000 / 0.000 | 0.428 | 0.432 | 0.088 | 38 / 107 | 187 / 192 | 0.444 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone1 | 580 | 0.426 / 0.043 | 0.550 / 0.858 | 0.024 / 0.099 | 0.000 / 0.000 | 0.568 | 0.795 | 0.084 | 53 / 77 | 151 / 153 | 0.495 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in between | 1366 | 0.283 / 0.032 | 0.616 / 0.859 | 0.102 / 0.109 | 0.000 / 0.000 | 0.347 | 0.902 | 0.059 | 27 / 41 | 115 / 118 | 0.459 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.007 | 0.857 / 0.876 | 0.039 / 0.118 | 0.000 / 0.000 | 0.655 | 0.348 | 0.041 | 9 / 18 | 80 / 80 | 0.524 | NO: death, timeout, ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.000 | 0.958 / 0.889 | 0.042 / 0.111 | 0.000 / 0.000 | - | - | 0.034 | None / 9 | 48 / 48 | - | NO: reach, timeout |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.945 | 0.043 / 0.055 | 0.000 / 0.000 | - | - | 0.026 | None / 136 | 10 / 10 | - | yes |

