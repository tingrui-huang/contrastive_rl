# Learned-ETT futures vs simulator futures vs recorded futures: five paired seeds

Sealed 2026-09-20 17:25:55.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the five paired seeds > 2 x seed s.e. and 5 / 5.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT | 0.483 / 0.48 / 0.19 / 0.32 | 0.213 / 0.16 / 0.14 / 0.65 | 0.337 / 0.36 / 0.20 / 0.46 | 0.390 / 0.47 / 0.15 / 0.46 | 0.310 / 0.32 / 0.08 / 0.61 | 0.347 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETT - O | +0.187 | -0.073 | +0.047 | +0.090 | +0.000 | +0.050 | 0.044 | 3/5 | False |
| CF - O | +0.227 | +0.183 | +0.147 | +0.107 | +0.097 | +0.152 | 0.024 | 5/5 | True |
| ETT - CF | -0.040 | -0.257 | -0.100 | -0.017 | -0.097 | -0.102 | 0.042 | 5/5 | False |
| ETT - start | +0.210 | -0.060 | +0.063 | +0.117 | +0.037 | +0.073 | 0.045 | 4/5 | False |
| CF - start | +0.250 | +0.197 | +0.163 | +0.133 | +0.133 | +0.175 | 0.022 | 5/5 | True |
| O - start | +0.023 | +0.013 | +0.017 | +0.027 | +0.037 | +0.023 | 0.004 | 5/5 | True |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.329 | -0.524 | +0.474 |
| CF - O | +0.392 | -0.347 | +0.195 |
| ETT - CF | -0.063 | -0.177 | +0.279 |
| ETT - start | +0.351 | -0.551 | +0.477 |
| CF - start | +0.413 | -0.374 | +0.199 |
| O - start | +0.021 | -0.027 | +0.003 |

## The learned-ETT branch table vs the sealed simulator table (anchor-weighted outcome shares)

| | success ETT / sim | death ETT / sim | timeout ETT / sim | mean rows ETT / sim |
|---|---|---|---|---|
| pooled | 0.644 / 0.621 | 0.306 / 0.291 | 0.050 / 0.088 | 123 / 143 |
| start (n 4276) | 0.260 / 0.243 | 0.717 / 0.728 | 0.022 / 0.028 | |
| pre_zone1 (n 12541) | 0.397 / 0.312 | 0.536 / 0.504 | 0.067 / 0.184 | |
| zone1 (n 4946) | 0.512 / 0.532 | 0.465 / 0.445 | 0.023 / 0.023 | |
| between (n 12326) | 0.655 / 0.634 | 0.304 / 0.275 | 0.041 / 0.091 | |
| zone2 (n 4946) | 0.859 / 0.838 | 0.108 / 0.120 | 0.033 / 0.042 | |
| post_zone2 (n 8069) | 0.957 / 0.955 | 0.001 / 0.000 | 0.042 / 0.045 | |
| goal_area (n 4140) | 0.976 / 0.976 | 0.000 / 0.000 | 0.024 / 0.024 | |
| west_column (n 299) | 0.779 / 0.789 | 0.000 / 0.000 | 0.221 / 0.211 | |
| top_corridor (n 1601) | 0.745 / 0.822 | 0.001 / 0.000 | 0.255 / 0.178 | |
| east_column (n 603) | 0.905 / 0.920 | 0.000 / 0.000 | 0.095 / 0.080 | |

## Judgement

```
{
 "primary_ETT_minus_O": false,
 "oracle_CF_minus_O": true,
 "retention_of_oracle_gain": 0.32894736842105265
}
```
