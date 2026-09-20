# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.301 | 0.615 / 0.639 | 0.091 / 0.060 | 0.012 / 0.012 | 0.018 | 0.079 | 0.036 | 41 / 41 | 100 / 105 | 0.995 | yes |
| anchor in start | 471 | 0.724 / 0.727 | 0.248 / 0.256 | 0.028 / 0.017 | 0.030 / 0.028 | 0.071 | 0.162 | 0.120 | 61 / 63 | 217 / 220 | 1.000 | NO: ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.524 | 0.319 / 0.416 | 0.178 / 0.059 | 0.000 / 0.000 | 0.033 | 0.124 | 0.120 | 38 / 39 | 187 / 195 | 0.986 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.431 | 0.550 / 0.547 | 0.024 / 0.023 | 0.000 / 0.000 | 0.104 | 0.108 | 0.088 | 53 / 46 | 151 / 152 | 0.994 | yes |
| anchor in between | 1366 | 0.283 / 0.288 | 0.616 / 0.657 | 0.102 / 0.055 | 0.000 / 0.000 | 0.054 | 0.126 | 0.045 | 27 / 26 | 115 / 117 | 0.992 | yes |
| anchor in zone2 | 544 | 0.105 / 0.107 | 0.857 / 0.827 | 0.039 / 0.066 | 0.000 / 0.000 | 0.241 | 0.200 | 0.030 | 9 / 8 | 80 / 79 | 0.996 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.936 | 0.042 / 0.063 | 0.000 / 0.000 | - | - | 0.031 | None / 13 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.955 | 0.043 / 0.045 | 0.000 / 0.000 | - | - | 0.028 | None / None | 10 / 10 | - | yes |

## Advice A: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.033 | 0.615 / 0.856 | 0.091 / 0.112 | 0.012 / 0.012 | 0.344 | 0.512 | 0.156 | 41 / 77 | 100 / 131 | 0.682 | NO: death, reach, ks_death_time, ks_death_x, ks_reach_time |
| anchor in start | 471 | 0.724 / 0.056 | 0.248 / 0.829 | 0.028 / 0.115 | 0.030 / 0.028 | 0.499 | 0.619 | 0.096 | 61 / 85 | 217 / 216 | 0.496 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.072 | 0.319 / 0.815 | 0.178 / 0.113 | 0.000 / 0.000 | 0.439 | 0.428 | 0.090 | 38 / 110 | 187 / 192 | 0.439 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone1 | 580 | 0.426 / 0.036 | 0.550 / 0.861 | 0.024 / 0.103 | 0.000 / 0.000 | 0.541 | 0.770 | 0.079 | 53 / 75 | 151 / 152 | 0.505 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in between | 1366 | 0.283 / 0.028 | 0.616 / 0.862 | 0.102 / 0.109 | 0.000 / 0.000 | 0.351 | 0.913 | 0.058 | 27 / 41 | 115 / 118 | 0.458 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.007 | 0.857 / 0.891 | 0.039 / 0.103 | 0.000 / 0.000 | 0.655 | 0.349 | 0.034 | 9 / 18 | 80 / 80 | 0.547 | NO: death, timeout, ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.000 | 0.958 / 0.892 | 0.042 / 0.108 | 0.000 / 0.000 | - | - | 0.033 | None / 9 | 48 / 49 | - | NO: reach, timeout |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.947 | 0.043 / 0.053 | 0.000 / 0.000 | - | - | 0.027 | None / 7 | 10 / 10 | - | yes |

