# Learned-ETT futures vs simulator futures vs recorded futures: 5 paired seeds

Sealed 2026-09-21 17:17:37.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the 5 paired seeds > 2 x seed s.e. and 5 / 5.  CF2 = the second oracle table (oracle_draw2), a reference.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT | 0.567 / 0.58 / 0.14 / 0.30 | 0.463 / 0.60 / 0.01 / 0.53 | 0.387 / 0.45 / 0.01 / 0.61 | 0.433 / 0.56 / 0.09 / 0.47 | 0.047 / 0.06 / 0.01 / 0.94 | 0.379 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| CF2 | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.310 / 0.34 / 0.33 / 0.36 | 0.580 / 0.57 / 0.26 / 0.16 | 0.351 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETT - O | +0.270 | +0.177 | +0.097 | +0.133 | -0.263 | +0.083 | 0.091 | 4/5 | False |
| CF - O | +0.227 | +0.183 | +0.147 | +0.107 | +0.097 | +0.152 | 0.024 | 5/5 | True |
| ETT - CF | +0.043 | -0.007 | -0.050 | +0.027 | -0.360 | -0.069 | 0.074 | 3/5 | False |
| CF2 - O | -0.017 | -0.053 | +0.060 | +0.010 | +0.270 | +0.054 | 0.057 | 3/5 | False |
| ETT - CF2 | +0.287 | +0.230 | +0.037 | +0.123 | -0.533 | +0.029 | 0.147 | 4/5 | False |
| CF - CF2 | +0.243 | +0.237 | +0.087 | +0.097 | -0.173 | +0.098 | 0.076 | 4/5 | False |
| ETT - start | +0.293 | +0.190 | +0.113 | +0.160 | -0.227 | +0.106 | 0.088 | 4/5 | False |
| CF - start | +0.250 | +0.197 | +0.163 | +0.133 | +0.133 | +0.175 | 0.022 | 5/5 | True |
| O - start | +0.023 | +0.013 | +0.017 | +0.027 | +0.037 | +0.023 | 0.004 | 5/5 | True |
| CF2 - start | +0.007 | -0.040 | +0.077 | +0.037 | +0.307 | +0.077 | 0.060 | 4/5 | False |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.421 | -0.626 | +0.543 |
| CF - O | +0.392 | -0.347 | +0.195 |
| ETT - CF | +0.029 | -0.279 | +0.348 |
| CF2 - O | +0.303 | -0.335 | +0.281 |
| ETT - CF2 | +0.119 | -0.291 | +0.263 |
| CF - CF2 | +0.089 | -0.013 | -0.085 |
| ETT - start | +0.443 | -0.653 | +0.547 |
| CF - start | +0.413 | -0.374 | +0.199 |
| O - start | +0.021 | -0.027 | +0.003 |
| CF2 - start | +0.324 | -0.361 | +0.284 |

## The learned-ETT branch table vs the sealed simulator table (anchor-weighted outcome shares)

| | success ETT / sim | death ETT / sim | timeout ETT / sim | mean rows ETT / sim |
|---|---|---|---|---|
| pooled | 0.639 / 0.621 | 0.304 / 0.291 | 0.057 / 0.088 | 133 / 143 |
| start (n 4276) | 0.263 / 0.243 | 0.718 / 0.728 | 0.019 / 0.028 | |
| pre_zone1 (n 12541) | 0.338 / 0.312 | 0.534 / 0.504 | 0.128 / 0.184 | |
| zone1 (n 4946) | 0.526 / 0.532 | 0.459 / 0.445 | 0.015 / 0.023 | |
| between (n 12326) | 0.650 / 0.634 | 0.301 / 0.275 | 0.050 / 0.091 | |
| zone2 (n 4946) | 0.869 / 0.838 | 0.108 / 0.120 | 0.024 / 0.042 | |
| post_zone2 (n 8069) | 0.974 / 0.955 | 0.002 / 0.000 | 0.024 / 0.045 | |
| goal_area (n 4140) | 0.973 / 0.976 | 0.000 / 0.000 | 0.027 / 0.024 | |
| west_column (n 299) | 0.870 / 0.789 | 0.000 / 0.000 | 0.130 / 0.211 | |
| top_corridor (n 1601) | 0.866 / 0.822 | 0.001 / 0.000 | 0.134 / 0.178 | |
| east_column (n 603) | 0.934 / 0.920 | 0.000 / 0.000 | 0.066 / 0.080 | |

## Judgement

```
{
 "seeds": [
  0,
  1,
  2,
  3,
  4
 ],
 "primary_ETT_minus_O": false,
 "oracle_CF_minus_O": true,
 "oracle2_CF2_minus_O": false,
 "retention_of_oracle_gain": 0.543859649122807,
 "retention_of_oracle2_gain": 1.5308641975308641
}
```
