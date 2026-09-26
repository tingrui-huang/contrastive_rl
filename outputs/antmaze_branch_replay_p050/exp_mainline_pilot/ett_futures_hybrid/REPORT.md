# Learned-ETT futures vs simulator futures vs recorded futures: five paired seeds

Sealed 2026-09-20 18:59:02.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the five paired seeds > 2 x seed s.e. and 5 / 5.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT | 0.357 / 0.25 / 0.39 / 0.25 | 0.370 / 0.38 / 0.35 / 0.28 | 0.187 / 0.03 / 0.57 / 0.24 | 0.313 / 0.25 / 0.42 / 0.26 | 0.477 / 0.40 / 0.39 / 0.13 | 0.341 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETT - O | +0.060 | +0.083 | -0.103 | +0.013 | +0.167 | +0.044 | 0.044 | 4/5 | False |
| CF - O | +0.227 | +0.183 | +0.147 | +0.107 | +0.097 | +0.152 | 0.024 | 5/5 | True |
| ETT - CF | -0.167 | -0.100 | -0.250 | -0.093 | +0.070 | -0.108 | 0.053 | 4/5 | False |
| ETT - start | +0.083 | +0.097 | -0.087 | +0.040 | +0.203 | +0.067 | 0.047 | 4/5 | False |
| CF - start | +0.250 | +0.197 | +0.163 | +0.133 | +0.133 | +0.175 | 0.022 | 5/5 | True |
| O - start | +0.023 | +0.013 | +0.017 | +0.027 | +0.037 | +0.023 | 0.004 | 5/5 | True |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.232 | -0.251 | +0.207 |
| CF - O | +0.392 | -0.347 | +0.195 |
| ETT - CF | -0.160 | +0.096 | +0.012 |
| ETT - start | +0.253 | -0.278 | +0.211 |
| CF - start | +0.413 | -0.374 | +0.199 |
| O - start | +0.021 | -0.027 | +0.003 |

## The learned-ETT branch table vs the sealed simulator table (anchor-weighted outcome shares)

| | success ETT / sim | death ETT / sim | timeout ETT / sim | mean rows ETT / sim |
|---|---|---|---|---|
| pooled | 0.612 / 0.621 | 0.300 / 0.291 | 0.088 / 0.088 | 143 / 143 |
| start (n 4276) | 0.246 / 0.243 | 0.726 / 0.728 | 0.028 / 0.028 | |
| pre_zone1 (n 12541) | 0.303 / 0.312 | 0.513 / 0.504 | 0.184 / 0.184 | |
| zone1 (n 4946) | 0.504 / 0.532 | 0.471 / 0.445 | 0.024 / 0.023 | |
| between (n 12326) | 0.614 / 0.634 | 0.295 / 0.275 | 0.091 / 0.091 | |
| zone2 (n 4946) | 0.848 / 0.838 | 0.109 / 0.120 | 0.043 / 0.042 | |
| post_zone2 (n 8069) | 0.954 / 0.955 | 0.001 / 0.000 | 0.045 / 0.045 | |
| goal_area (n 4140) | 0.976 / 0.976 | 0.000 / 0.000 | 0.024 / 0.024 | |
| west_column (n 299) | 0.769 / 0.789 | 0.017 / 0.000 | 0.214 / 0.211 | |
| top_corridor (n 1601) | 0.818 / 0.822 | 0.001 / 0.000 | 0.181 / 0.178 | |
| east_column (n 603) | 0.920 / 0.920 | 0.000 / 0.000 | 0.080 / 0.080 | |

## Judgement

```
{
 "primary_ETT_minus_O": false,
 "oracle_CF_minus_O": true,
 "retention_of_oracle_gain": 0.2894736842105263
}
```
