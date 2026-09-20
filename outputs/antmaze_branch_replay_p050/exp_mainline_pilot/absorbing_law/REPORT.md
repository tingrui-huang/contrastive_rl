# O vs CF under the absorbing future law (sparse-supervision hypothesis)

Sealed 2026-09-20 07:45:21.  Evaluation seed 7909, 300 episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3 / 3.

## Headline (success / detour / death / timeout)

| run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| abs_O | 0.270 / 0.00 / 0.73 / 0.00 | 0.263 / 0.01 / 0.73 / 0.01 | 0.263 / 0.00 / 0.73 / 0.01 |
| clip_O | 0.263 / 0.01 / 0.71 / 0.03 | 0.273 / 0.01 / 0.73 / 0.00 | 0.283 / 0.02 / 0.71 / 0.00 |
| abs_CF | 0.013 / 0.08 / 0.00 / 0.99 | 0.380 / 0.78 / 0.02 / 0.60 | 0.063 / 0.06 / 0.01 / 0.92 |
| clip_CF | 0.517 / 0.62 / 0.19 / 0.29 | 0.470 / 0.51 / 0.31 / 0.22 | 0.423 / 0.33 / 0.43 / 0.15 |
| start | 0.253 / 0.01 / 0.72 / 0.03 | | |

## Paired differences (success; per seed, seed mean, seed s.e., seeds in direction)

| comparison | seed 0 | seed 1 | seed 2 | mean | seed s.e. | in direction | rule |
|---|---:|---:|---:|---:|---:|---:|---|
| CF_abs - O_abs | -0.257 | +0.117 | -0.200 | -0.113 | 0.116 | 2/3 | False |
| CF_clip - O_clip | +0.253 | +0.197 | +0.140 | +0.197 | 0.033 | 3/3 | True |
| CF_abs - CF_clip | -0.503 | -0.090 | -0.360 | -0.318 | 0.121 | 3/3 | False |
| O_abs - O_clip | +0.007 | -0.010 | -0.020 | -0.008 | 0.008 | 2/3 | False |
| abs_CF - start | -0.240 | +0.127 | -0.190 | -0.101 | 0.115 | 2/3 | False |
| abs_O - start | +0.017 | +0.010 | +0.010 | +0.012 | 0.002 | 3/3 | True |
| gain_abs - gain_clip | -0.510 | -0.080 | -0.340 | -0.310 | 0.125 | 3/3 | False |

## Other keys (seed mean of the paired difference)

| comparison | detour | death | timeout |
|---|---:|---:|---:|
| CF_abs - O_abs | +0.307 | -0.718 | +0.831 |
| CF_clip - O_clip | +0.472 | -0.406 | +0.209 |
| CF_abs - CF_clip | -0.176 | -0.300 | +0.618 |
| O_abs - O_clip | -0.010 | +0.012 | -0.004 |
| abs_CF - start | +0.299 | -0.709 | +0.810 |
| abs_O - start | -0.008 | +0.009 | -0.021 |

## Critic read-out: within-anchor AUROC (success) over the stage-1 query candidates

| lineage | CF_abs reset / all | CF_clip reset / all | O_abs reset / all | O_clip reset / all |
|---|---|---|---|---|
| seed_0 | 0.582 / 0.515 | 0.619 / 0.511 | 0.483 / 0.495 | 0.495 / 0.494 |
| seed_1 | 0.525 / 0.509 | 0.583 / 0.511 | 0.460 / 0.484 | 0.604 / 0.511 |
| seed_2 | 0.628 / 0.524 | 0.621 / 0.515 | 0.490 / 0.512 | 0.505 / 0.502 |

## Judgement

```
{
 "primary_CF_abs_minus_O_abs": false,
 "vs_current_CF_abs_minus_CF_clip": false,
 "critic_reset_auroc_up": "1 / 3",
 "supported": false
}
```
