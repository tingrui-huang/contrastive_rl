# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.305 | 0.615 / 0.638 | 0.091 / 0.057 | 0.012 / 0.012 | 0.036 | 0.066 | 0.036 | 41 / 39 | 100 / 105 | 0.996 | yes |
| anchor in start | 471 | 0.724 / 0.728 | 0.248 / 0.245 | 0.028 / 0.028 | 0.030 / 0.028 | 0.038 | 0.085 | 0.096 | 61 / 62 | 217 / 218 | 1.000 | yes |
| anchor in pre_zone1 | 1456 | 0.503 / 0.527 | 0.319 / 0.419 | 0.178 / 0.053 | 0.000 / 0.000 | 0.017 | 0.056 | 0.113 | 38 / 38 | 187 / 194 | 0.987 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.440 | 0.550 / 0.531 | 0.024 / 0.028 | 0.000 / 0.000 | 0.156 | 0.103 | 0.072 | 53 / 45 | 151 / 152 | 0.994 | NO: ks_death_time |
| anchor in between | 1366 | 0.283 / 0.293 | 0.616 / 0.652 | 0.102 / 0.055 | 0.000 / 0.000 | 0.078 | 0.168 | 0.040 | 27 / 24 | 115 / 117 | 0.992 | NO: ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.113 | 0.857 / 0.831 | 0.039 / 0.056 | 0.000 / 0.000 | 0.321 | 0.210 | 0.034 | 9 / 7 | 80 / 80 | 0.997 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.002 | 0.958 / 0.935 | 0.042 / 0.063 | 0.000 / 0.000 | - | - | 0.033 | None / 11 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.955 | 0.043 / 0.045 | 0.000 / 0.000 | - | - | 0.028 | None / None | 10 / 10 | - | yes |

## Advice A: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.036 | 0.615 / 0.853 | 0.091 / 0.111 | 0.012 / 0.012 | 0.336 | 0.515 | 0.154 | 41 / 76 | 100 / 131 | 0.691 | NO: death, reach, ks_death_time, ks_death_x, ks_reach_time |
| anchor in start | 471 | 0.724 / 0.061 | 0.248 / 0.832 | 0.028 / 0.107 | 0.030 / 0.028 | 0.485 | 0.604 | 0.092 | 61 / 83 | 217 / 216 | 0.531 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.080 | 0.319 / 0.809 | 0.178 / 0.111 | 0.000 / 0.000 | 0.437 | 0.440 | 0.092 | 38 / 109 | 187 / 192 | 0.451 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone1 | 580 | 0.426 / 0.041 | 0.550 / 0.863 | 0.024 / 0.095 | 0.000 / 0.000 | 0.524 | 0.761 | 0.079 | 53 / 76 | 151 / 153 | 0.506 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in between | 1366 | 0.283 / 0.032 | 0.616 / 0.863 | 0.102 / 0.105 | 0.000 / 0.000 | 0.346 | 0.903 | 0.052 | 27 / 41 | 115 / 118 | 0.458 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.008 | 0.857 / 0.881 | 0.039 / 0.112 | 0.000 / 0.000 | 0.605 | 0.330 | 0.029 | 9 / 18 | 80 / 80 | 0.545 | NO: death, timeout, ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.000 | 0.958 / 0.890 | 0.042 / 0.109 | 0.000 / 0.000 | - | - | 0.032 | None / 9 | 48 / 49 | - | NO: reach, timeout |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.946 | 0.043 / 0.054 | 0.000 / 0.000 | - | - | 0.026 | None / 8 | 10 / 10 | - | yes |

