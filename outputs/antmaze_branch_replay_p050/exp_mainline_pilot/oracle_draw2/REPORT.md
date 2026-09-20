# A second hazard draw of the simulator branch table (CF2) vs the sealed draw (CF), the recorded futures (O) and the hybrid control: five paired seeds

Sealed 2026-09-20 20:27:04.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| CF2 | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.310 / 0.34 / 0.33 / 0.36 | 0.580 / 0.57 / 0.26 / 0.16 | 0.351 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| hybrid | 0.357 / 0.25 / 0.39 / 0.25 | 0.370 / 0.38 / 0.35 / 0.28 | 0.187 / 0.03 / 0.57 / 0.24 | 0.313 / 0.25 / 0.42 / 0.26 | 0.477 / 0.40 / 0.39 / 0.13 | 0.341 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CF2 - O | -0.017 | -0.053 | +0.060 | +0.010 | +0.270 | +0.054 | 0.057 | 3/5 | False |
| CF - O | +0.227 | +0.183 | +0.147 | +0.107 | +0.097 | +0.152 | 0.024 | 5/5 | True |
| CF2 - CF | -0.243 | -0.237 | -0.087 | -0.097 | +0.173 | -0.098 | 0.076 | 4/5 | False |
| hybrid - CF2 | +0.077 | +0.137 | -0.163 | +0.003 | -0.103 | -0.010 | 0.055 | 2/5 | False |
| hybrid - CF | -0.167 | -0.100 | -0.250 | -0.093 | +0.070 | -0.108 | 0.053 | 4/5 | False |
| CF2 - start | +0.007 | -0.040 | +0.077 | +0.037 | +0.307 | +0.077 | 0.060 | 4/5 | False |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| CF2 - O | +0.303 | -0.335 | +0.281 |
| CF - O | +0.392 | -0.347 | +0.195 |
| CF2 - CF | -0.089 | +0.013 | +0.085 |
| hybrid - CF2 | -0.071 | +0.083 | -0.073 |
| hybrid - CF | -0.160 | +0.096 | +0.012 |
| CF2 - start | +0.324 | -0.361 | +0.284 |

## Draw 2 vs the sealed draw (anchor-weighted outcome shares)

| | success draw2 / sealed | death draw2 / sealed | timeout draw2 / sealed | KS death time |
|---|---|---|---|---|
| pooled | 0.620 / 0.621 | 0.292 / 0.291 | 0.088 / 0.088 | 0.007777992313604917 |
| start (n 4276) | 0.248 / 0.243 | 0.725 / 0.728 | 0.027 / 0.028 | 0.012585119105888776 |
| pre_zone1 (n 12541) | 0.314 / 0.312 | 0.503 / 0.504 | 0.183 / 0.184 | 0.01008002335846836 |
| zone1 (n 4946) | 0.526 / 0.532 | 0.449 / 0.445 | 0.025 / 0.023 | 0.022146714671467116 |
| between (n 12326) | 0.632 / 0.634 | 0.277 / 0.275 | 0.091 / 0.091 | 0.01518159084725118 |
| zone2 (n 4946) | 0.837 / 0.838 | 0.119 / 0.120 | 0.044 / 0.042 | 0.03673626778396799 |
| post_zone2 (n 8069) | 0.955 / 0.955 | 0.000 / 0.000 | 0.045 / 0.045 | None |
| goal_area (n 4140) | 0.976 / 0.976 | 0.000 / 0.000 | 0.024 / 0.024 | None |
| west_column (n 299) | 0.786 / 0.789 | 0.000 / 0.000 | 0.214 / 0.211 | None |
| top_corridor (n 1601) | 0.821 / 0.822 | 0.000 / 0.000 | 0.179 / 0.178 | None |
| east_column (n 603) | 0.920 / 0.920 | 0.000 / 0.000 | 0.080 / 0.080 | None |

Row identity on the anchors inactive in both draws: {"anchors_inactive_in_both_draws": 3338, "identical_paths": 3338, "share": 1.0, "first_diff_row_if_any (p10/med/p90)": null}; per-anchor outcome agreement 0.786.

## Judgement

```
{
 "nature": "replication check under a fresh future-table draw (two tables; no variance estimate; no equivalence threshold)",
 "oracle_draw2_CF2_minus_O_rule_met": false,
 "CF2_minus_O_mean": 0.054000000000000006,
 "CF2_minus_CF_mean": -0.098,
 "CF2_minus_CF_seed_se": 0.07553218592832535,
 "CF2_minus_CF_same_direction": "4/5",
 "hybrid_minus_CF2_mean": -0.010000000000000002,
 "hybrid_minus_CF2_seed_se": 0.055417606508321074,
 "hybrid_minus_CF2_same_direction": "2/5",
 "hybrid_minus_CF_mean": -0.10800000000000001,
 "reading": "left to the written rule (three readings kept open); not computed here"
}
```
