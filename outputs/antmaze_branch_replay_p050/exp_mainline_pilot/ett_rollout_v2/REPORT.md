# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.311 | 0.615 / 0.624 | 0.091 / 0.065 | 0.012 / 0.012 | 0.027 | 0.072 | 0.028 | 41 / 40 | 100 / 103 | 0.992 | yes |
| anchor in start | 471 | 0.724 / 0.725 | 0.248 / 0.241 | 0.028 / 0.034 | 0.030 / 0.030 | 0.046 | 0.157 | 0.066 | 61 / 62 | 217 / 216 | 1.000 | NO: ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.546 | 0.319 / 0.405 | 0.178 / 0.049 | 0.000 / 0.000 | 0.032 | 0.109 | 0.079 | 38 / 38 | 187 / 192 | 0.977 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.439 | 0.550 / 0.522 | 0.024 / 0.039 | 0.000 / 0.000 | 0.102 | 0.097 | 0.038 | 53 / 48 | 151 / 150 | 0.995 | yes |
| anchor in between | 1366 | 0.283 / 0.300 | 0.616 / 0.627 | 0.102 / 0.074 | 0.000 / 0.000 | 0.072 | 0.166 | 0.055 | 27 / 25 | 115 / 115 | 0.984 | NO: ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.111 | 0.857 / 0.807 | 0.039 / 0.082 | 0.000 / 0.000 | 0.217 | 0.167 | 0.071 | 9 / 8 | 80 / 79 | 0.995 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.006 | 0.958 / 0.906 | 0.042 / 0.089 | 0.000 / 0.000 | - | - | 0.028 | None / 14 | 48 / 48 | - | NO: reach |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.953 | 0.043 / 0.047 | 0.000 / 0.000 | - | - | 0.036 | None / 11 | 10 / 10 | - | yes |

## Advice A: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.032 | 0.615 / 0.860 | 0.091 / 0.109 | 0.012 / 0.012 | 0.313 | 0.462 | 0.149 | 41 / 71 | 100 / 129 | 0.644 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in start | 471 | 0.724 / 0.055 | 0.248 / 0.839 | 0.028 / 0.106 | 0.030 / 0.029 | 0.409 | 0.593 | 0.129 | 61 / 81 | 217 / 213 | 0.523 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.064 | 0.319 / 0.825 | 0.178 / 0.111 | 0.000 / 0.000 | 0.403 | 0.470 | 0.063 | 38 / 97 | 187 / 189 | 0.450 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone1 | 580 | 0.426 / 0.034 | 0.550 / 0.866 | 0.024 / 0.100 | 0.000 / 0.000 | 0.488 | 0.698 | 0.021 | 53 / 73 | 151 / 151 | 0.501 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in between | 1366 | 0.283 / 0.035 | 0.616 / 0.849 | 0.102 / 0.116 | 0.000 / 0.000 | 0.479 | 0.668 | 0.056 | 27 / 50 | 115 / 115 | 0.475 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.006 | 0.857 / 0.882 | 0.039 / 0.112 | 0.000 / 0.000 | 0.377 | 0.229 | 0.024 | 9 / 14 | 80 / 80 | 0.517 | NO: death, timeout, ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.000 | 0.958 / 0.893 | 0.042 / 0.107 | 0.000 / 0.000 | - | - | 0.019 | None / 24 | 48 / 48 | - | NO: reach, timeout |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.954 | 0.043 / 0.046 | 0.000 / 0.000 | - | - | 0.035 | None / 7 | 10 / 11 | - | yes |

