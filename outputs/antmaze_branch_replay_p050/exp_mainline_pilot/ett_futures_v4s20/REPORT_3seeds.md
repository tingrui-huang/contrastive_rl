# Learned-ETT futures vs simulator futures vs recorded futures: 3 paired seeds

Sealed 2026-09-21 17:17:37.  Evaluation seed 8909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3 / 3.  CF2 = the second oracle table (oracle_draw2), a reference.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| ETT | 0.567 / 0.58 / 0.14 / 0.30 | 0.463 / 0.60 / 0.01 / 0.53 | 0.387 / 0.45 / 0.01 / 0.61 | 0.472 |
| CF | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.477 |
| O | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.291 |
| CF2 | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.288 |
| start | 0.273 / 0.01 / 0.70 / 0.02 |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---|
| ETT - O | +0.270 | +0.177 | +0.097 | +0.181 | 0.050 | 3/3 | True |
| CF - O | +0.227 | +0.183 | +0.147 | +0.186 | 0.023 | 3/3 | True |
| ETT - CF | +0.043 | -0.007 | -0.050 | -0.004 | 0.027 | 2/3 | False |
| CF2 - O | -0.017 | -0.053 | +0.060 | -0.003 | 0.033 | 2/3 | False |
| ETT - CF2 | +0.287 | +0.230 | +0.037 | +0.184 | 0.076 | 3/3 | True |
| CF - CF2 | +0.243 | +0.237 | +0.087 | +0.189 | 0.051 | 3/3 | True |
| ETT - start | +0.293 | +0.190 | +0.113 | +0.199 | 0.052 | 3/3 | True |
| CF - start | +0.250 | +0.197 | +0.163 | +0.203 | 0.025 | 3/3 | True |
| O - start | +0.023 | +0.013 | +0.017 | +0.018 | 0.003 | 3/3 | True |
| CF2 - start | +0.007 | -0.040 | +0.077 | +0.014 | 0.034 | 2/3 | False |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.528 | -0.644 | +0.463 |
| CF - O | +0.443 | -0.383 | +0.198 |
| ETT - CF | +0.084 | -0.261 | +0.266 |
| CF2 - O | +0.233 | -0.319 | +0.322 |
| ETT - CF2 | +0.294 | -0.326 | +0.141 |
| CF - CF2 | +0.210 | -0.064 | -0.124 |
| ETT - start | +0.536 | -0.653 | +0.454 |
| CF - start | +0.451 | -0.392 | +0.189 |
| O - start | +0.008 | -0.009 | -0.009 |
| CF2 - start | +0.241 | -0.328 | +0.313 |

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
  2
 ],
 "primary_ETT_minus_O": true,
 "oracle_CF_minus_O": true,
 "oracle2_CF2_minus_O": false,
 "retention_of_oracle_gain": 0.9760479041916168,
 "retention_of_oracle2_gain": -54.333333333333286
}
```
