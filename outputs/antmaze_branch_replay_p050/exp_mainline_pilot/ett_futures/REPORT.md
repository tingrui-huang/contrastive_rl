# Learned-ETT futures vs simulator futures vs recorded futures: five paired seeds

Sealed 2026-09-20 13:14:24.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the five paired seeds > 2 x seed s.e. and 5 / 5.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT | 0.283 / 0.24 / 0.15 / 0.57 | 0.450 / 0.34 / 0.29 / 0.26 | 0.417 / 0.47 / 0.18 / 0.40 | 0.167 / 0.20 / 0.11 / 0.72 | 0.137 / 0.14 / 0.24 / 0.62 | 0.291 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETT - O | -0.013 | +0.163 | +0.127 | -0.133 | -0.173 | -0.006 | 0.067 | 3/5 | False |
| CF - O | +0.227 | +0.183 | +0.147 | +0.107 | +0.097 | +0.152 | 0.024 | 5/5 | True |
| ETT - CF | -0.240 | -0.020 | -0.020 | -0.240 | -0.270 | -0.158 | 0.057 | 5/5 | False |
| ETT - start | +0.010 | +0.177 | +0.143 | -0.107 | -0.137 | +0.017 | 0.063 | 3/5 | False |
| CF - start | +0.250 | +0.197 | +0.163 | +0.133 | +0.133 | +0.175 | 0.022 | 5/5 | True |
| O - start | +0.023 | +0.013 | +0.017 | +0.027 | +0.037 | +0.023 | 0.004 | 5/5 | True |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.251 | -0.482 | +0.488 |
| CF - O | +0.392 | -0.347 | +0.195 |
| ETT - CF | -0.141 | -0.135 | +0.293 |
| ETT - start | +0.273 | -0.509 | +0.491 |
| CF - start | +0.413 | -0.374 | +0.199 |
| O - start | +0.021 | -0.027 | +0.003 |

## The learned-ETT branch table vs the sealed simulator table (anchor-weighted outcome shares)

| | success ETT / sim | death ETT / sim | timeout ETT / sim | mean rows ETT / sim |
|---|---|---|---|---|
| pooled | 0.633 / 0.621 | 0.307 / 0.291 | 0.060 / 0.088 | 128 / 143 |
| start (n 4276) | 0.252 / 0.243 | 0.720 / 0.728 | 0.028 / 0.028 | |
| pre_zone1 (n 12541) | 0.409 / 0.312 | 0.544 / 0.504 | 0.047 / 0.184 | |
| zone1 (n 4946) | 0.501 / 0.532 | 0.460 / 0.445 | 0.039 / 0.023 | |
| between (n 12326) | 0.642 / 0.634 | 0.302 / 0.275 | 0.056 / 0.091 | |
| zone2 (n 4946) | 0.830 / 0.838 | 0.108 / 0.120 | 0.062 / 0.042 | |
| post_zone2 (n 8069) | 0.925 / 0.955 | 0.001 / 0.000 | 0.074 / 0.045 | |
| goal_area (n 4140) | 0.975 / 0.976 | 0.000 / 0.000 | 0.025 / 0.024 | |
| west_column (n 299) | 0.709 / 0.789 | 0.000 / 0.000 | 0.291 / 0.211 | |
| top_corridor (n 1601) | 0.708 / 0.822 | 0.001 / 0.000 | 0.292 / 0.178 | |
| east_column (n 603) | 0.882 / 0.920 | 0.000 / 0.000 | 0.118 / 0.080 | |

## Judgement

```
{
 "primary_ETT_minus_O": false,
 "oracle_CF_minus_O": true,
 "retention_of_oracle_gain": -0.03947368421052632
}
```
