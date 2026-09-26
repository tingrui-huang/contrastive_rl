# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)

`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: 4 sampled-advice paths).  Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.

## Advice C: **GATE(S) FAILED** (6000 anchors, 6000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.308 | 0.615 / 0.671 | 0.091 / 0.021 | 0.012 / 0.012 | 0.084 | 0.080 | 0.036 | 41 / 41 | 100 / 105 | 0.991 | NO: reach, timeout |
| anchor in start | 471 | 0.724 / 0.720 | 0.248 / 0.274 | 0.028 / 0.006 | 0.030 / 0.034 | 0.062 | 0.180 | 0.057 | 61 / 63 | 217 / 217 | 1.000 | NO: ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.546 | 0.319 / 0.430 | 0.178 / 0.023 | 0.000 / 0.000 | 0.054 | 0.130 | 0.065 | 38 / 39 | 187 / 189 | 0.971 | NO: reach, timeout |
| anchor in zone1 | 580 | 0.426 / 0.434 | 0.550 / 0.563 | 0.024 / 0.004 | 0.000 / 0.000 | 0.284 | 0.149 | 0.067 | 53 / 51 | 151 / 152 | 0.995 | NO: ks_death_time |
| anchor in between | 1366 | 0.283 / 0.292 | 0.616 / 0.690 | 0.102 / 0.018 | 0.000 / 0.000 | 0.057 | 0.077 | 0.030 | 27 / 27 | 115 / 116 | 0.986 | NO: reach, timeout |
| anchor in zone2 | 544 | 0.105 / 0.112 | 0.857 / 0.877 | 0.039 / 0.011 | 0.000 / 0.000 | 0.764 | 0.350 | 0.022 | 9 / 3 | 80 / 80 | 0.992 | NO: ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.004 | 0.958 / 0.972 | 0.042 / 0.024 | 0.000 / 0.000 | - | - | 0.029 | None / 4 | 48 / 47 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.000 | 0.957 / 0.982 | 0.043 / 0.018 | 0.000 / 0.000 | - | - | 0.073 | None / 400 | 10 / 10 | - | yes |

## Advice A: **GATE(S) FAILED** (6000 anchors, 24000 paths)

| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |
|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|
| all | 6000 | 0.294 / 0.023 | 0.615 / 0.928 | 0.091 / 0.050 | 0.012 / 0.013 | 0.197 | 0.325 | 0.166 | 41 / 58 | 100 / 133 | 0.705 | NO: death, reach, ks_death_time, ks_death_x, ks_reach_time |
| anchor in start | 471 | 0.724 / 0.039 | 0.248 / 0.904 | 0.028 / 0.056 | 0.030 / 0.034 | 0.354 | 0.462 | 0.090 | 61 / 120 | 217 / 214 | 0.571 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in pre_zone1 | 1456 | 0.503 / 0.040 | 0.319 / 0.907 | 0.178 / 0.053 | 0.000 / 0.000 | 0.334 | 0.429 | 0.065 | 38 / 85 | 187 / 190 | 0.479 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone1 | 580 | 0.426 / 0.027 | 0.550 / 0.937 | 0.024 / 0.036 | 0.000 / 0.000 | 0.252 | 0.403 | 0.059 | 53 / 62 | 151 / 152 | 0.480 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in between | 1366 | 0.283 / 0.027 | 0.616 / 0.927 | 0.102 / 0.046 | 0.000 / 0.001 | 0.201 | 0.402 | 0.034 | 27 / 33 | 115 / 117 | 0.445 | NO: death, reach, timeout, ks_death_time, ks_death_x |
| anchor in zone2 | 544 | 0.105 / 0.009 | 0.857 / 0.946 | 0.039 / 0.045 | 0.000 / 0.000 | 0.364 | 0.195 | 0.043 | 9 / 7 | 80 / 80 | 0.520 | NO: death, reach, ks_death_time, ks_death_x |
| anchor in post_zone2 | 879 | 0.000 / 0.001 | 0.958 / 0.953 | 0.042 / 0.046 | 0.000 / 0.000 | - | - | 0.051 | None / 82 | 48 / 49 | - | yes |
| anchor in goal_area | 444 | 0.000 / 0.001 | 0.957 / 0.970 | 0.043 / 0.029 | 0.000 / 0.000 | - | - | 0.047 | None / 350 | 10 / 10 | - | yes |

