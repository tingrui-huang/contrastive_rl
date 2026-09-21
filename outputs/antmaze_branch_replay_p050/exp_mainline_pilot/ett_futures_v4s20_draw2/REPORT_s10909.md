# Learned-ETT futures vs simulator futures vs recorded futures: 5 paired seeds

Sealed 2026-09-21 19:39:30.  Evaluation seed 10909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the 5 paired seeds > 2 x seed s.e. and 5 / 5.  CF2 = the second oracle table (oracle_draw2), a reference.

## Headline (success / detour / death / timeout)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT | 0.397 / 0.38 / 0.32 / 0.28 | 0.497 / 0.52 / 0.21 / 0.29 | 0.207 / 0.13 / 0.53 / 0.26 | 0.363 / 0.31 / 0.45 / 0.19 | 0.457 / 0.64 / 0.25 / 0.29 | 0.384 |
| CF | 0.467 / 0.65 / 0.19 / 0.34 | 0.483 / 0.53 / 0.31 / 0.21 | 0.413 / 0.35 / 0.44 / 0.14 | 0.360 / 0.36 / 0.43 / 0.21 | 0.400 / 0.42 / 0.35 / 0.25 | 0.425 |
| O | 0.230 / 0.02 / 0.73 / 0.04 | 0.247 / 0.01 / 0.75 / 0.00 | 0.250 / 0.03 / 0.73 / 0.02 | 0.263 / 0.05 / 0.71 / 0.03 | 0.263 / 0.08 / 0.67 / 0.07 | 0.251 |
| CF2 | 0.253 / 0.38 / 0.41 / 0.34 | 0.183 / 0.15 / 0.27 / 0.55 | 0.253 / 0.19 / 0.51 / 0.23 | 0.297 / 0.38 / 0.35 / 0.36 | 0.577 / 0.64 / 0.25 / 0.17 | 0.313 |
| start | 0.213 / 0.01 / 0.75 / 0.04 |  |  |  |  |  |

## Paired differences (success)

| comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | seed s.e. | same direction | rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETT - O | +0.167 | +0.250 | -0.043 | +0.100 | +0.193 | +0.133 | 0.050 | 4/5 | False |
| CF - O | +0.237 | +0.237 | +0.163 | +0.097 | +0.137 | +0.174 | 0.028 | 5/5 | True |
| ETT - CF | -0.070 | +0.013 | -0.207 | +0.003 | +0.057 | -0.041 | 0.046 | 2/5 | False |
| CF2 - O | +0.023 | -0.063 | +0.003 | +0.033 | +0.313 | +0.062 | 0.065 | 4/5 | False |
| ETT - CF2 | +0.143 | +0.313 | -0.047 | +0.067 | -0.120 | +0.071 | 0.076 | 3/5 | False |
| CF - CF2 | +0.213 | +0.300 | +0.160 | +0.063 | -0.177 | +0.112 | 0.082 | 4/5 | False |
| ETT - start | +0.183 | +0.283 | -0.007 | +0.150 | +0.243 | +0.171 | 0.050 | 4/5 | False |
| CF - start | +0.253 | +0.270 | +0.200 | +0.147 | +0.187 | +0.211 | 0.022 | 5/5 | True |
| O - start | +0.017 | +0.033 | +0.037 | +0.050 | +0.050 | +0.037 | 0.006 | 5/5 | True |
| CF2 - start | +0.040 | -0.030 | +0.040 | +0.083 | +0.363 | +0.099 | 0.068 | 4/5 | False |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| ETT - O | +0.359 | -0.364 | +0.231 |
| CF - O | +0.425 | -0.374 | +0.200 |
| ETT - CF | -0.065 | +0.010 | +0.031 |
| CF2 - O | +0.313 | -0.360 | +0.298 |
| ETT - CF2 | +0.047 | -0.004 | -0.067 |
| CF - CF2 | +0.112 | -0.014 | -0.098 |
| ETT - start | +0.383 | -0.393 | +0.222 |
| CF - start | +0.449 | -0.403 | +0.191 |
| O - start | +0.024 | -0.029 | -0.009 |
| CF2 - start | +0.337 | -0.389 | +0.289 |

## The learned-ETT branch table vs the sealed simulator table (anchor-weighted outcome shares)

| | success ETT / sim | death ETT / sim | timeout ETT / sim | mean rows ETT / sim |
|---|---|---|---|---|
| pooled | 0.639 / 0.621 | 0.305 / 0.291 | 0.056 / 0.088 | 132 / 143 |
| start (n 4276) | 0.266 / 0.243 | 0.718 / 0.728 | 0.016 / 0.028 | |
| pre_zone1 (n 12541) | 0.338 / 0.312 | 0.536 / 0.504 | 0.126 / 0.184 | |
| zone1 (n 4946) | 0.515 / 0.532 | 0.469 / 0.445 | 0.015 / 0.023 | |
| between (n 12326) | 0.655 / 0.634 | 0.297 / 0.275 | 0.048 / 0.091 | |
| zone2 (n 4946) | 0.868 / 0.838 | 0.110 / 0.120 | 0.022 / 0.042 | |
| post_zone2 (n 8069) | 0.976 / 0.955 | 0.001 / 0.000 | 0.023 / 0.045 | |
| goal_area (n 4140) | 0.973 / 0.976 | 0.000 / 0.000 | 0.027 / 0.024 | |
| west_column (n 299) | 0.816 / 0.789 | 0.017 / 0.000 | 0.167 / 0.211 | |
| top_corridor (n 1601) | 0.859 / 0.822 | 0.001 / 0.000 | 0.141 / 0.178 | |
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
 "retention_of_oracle_gain": 0.7662835249042145,
 "retention_of_oracle2_gain": 2.150537634408602
}
```
