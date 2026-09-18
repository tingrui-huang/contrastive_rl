# Detour-share follow-up: hazard density (A) and discount (B) against the d20 vanilla chain

Sealed 2026-09-18 15:51:52.  Recorded-futures chain only; reading rule: mean over 3 seeds differs from the reference by > 2 x pooled seed s.e. with 3 / 3 seeds in the same direction.  Ladder baseline reference (frozen V6 recipe, 0.30 rung): detour 0.000.

| arm | BC walker: success / detour | seed | actor success | failure | timeout | detour | shortcut |
|---|---|---|---:|---:|---:|---:|---:|
| d20 vanilla (reference: p050, gamma 0.999) | 0.260 / 0.007 | 0 | 0.543 | 0.260 | 0.197 | 0.507 | 0.340 |
| d20 vanilla (reference: p050, gamma 0.999) | 0.260 / 0.007 | 1 | 0.723 | 0.097 | 0.180 | 0.737 | 0.137 |
| d20 vanilla (reference: p050, gamma 0.999) | 0.260 / 0.007 | 2 | 0.647 | 0.253 | 0.100 | 0.630 | 0.313 |
| A: p040 far20, gamma 0.999 | 0.357 / 0.003 | 0 | 0.593 | 0.133 | 0.273 | 0.537 | 0.217 |
| A: p040 far20, gamma 0.999 | 0.357 / 0.003 | 1 | 0.357 | 0.490 | 0.153 | 0.083 | 0.770 |
| A: p040 far20, gamma 0.999 | 0.357 / 0.003 | 2 | 0.590 | 0.233 | 0.177 | 0.493 | 0.343 |
| B: d20 (p050), gamma 0.99 | 0.260 / 0.007 | 0 | 0.353 | 0.543 | 0.103 | 0.183 | 0.720 |
| B: d20 (p050), gamma 0.99 | 0.260 / 0.007 | 1 | 0.367 | 0.503 | 0.130 | 0.253 | 0.633 |
| B: d20 (p050), gamma 0.99 | 0.260 / 0.007 | 2 | 0.330 | 0.450 | 0.220 | 0.223 | 0.587 |
| C: d10 (p050), gamma 0.999 | 0.250 / 0.000 | 0 | 0.393 | 0.487 | 0.120 | 0.257 | 0.627 |
| C: d10 (p050), gamma 0.999 | 0.250 / 0.000 | 1 | 0.460 | 0.300 | 0.240 | 0.460 | 0.380 |
| C: d10 (p050), gamma 0.999 | 0.250 / 0.000 | 2 | 0.330 | 0.580 | 0.090 | 0.150 | 0.743 |
| d05 vanilla (p050, gamma 0.999) | 0.257 / 0.000 | 0 | 0.263 | 0.673 | 0.063 | 0.067 | 0.867 |
| d05 vanilla (p050, gamma 0.999) | 0.257 / 0.000 | 1 | 0.263 | 0.737 | 0.000 | 0.007 | 0.973 |
| d05 vanilla (p050, gamma 0.999) | 0.257 / 0.000 | 2 | 0.260 | 0.733 | 0.007 | 0.013 | 0.960 |

| arm | mean success (seed s.e.) | mean detour (seed s.e.) |
|---|---|---|
| d20 vanilla (reference: p050, gamma 0.999) | 0.638 (0.052) | 0.624 (0.066) |
| A: p040 far20, gamma 0.999 | 0.513 (0.078) | 0.371 (0.144) |
| B: d20 (p050), gamma 0.99 | 0.350 (0.011) | 0.220 (0.020) |
| C: d10 (p050), gamma 0.999 | 0.394 (0.038) | 0.289 (0.091) |
| d05 vanilla (p050, gamma 0.999) | 0.262 (0.001) | 0.029 (0.019) |

## Pre-registered comparisons against the d20 vanilla reference

- A: p040 far20, gamma 0.999 minus reference, success: -0.124 (pooled seed s.e. 0.094; not > 2 s.e.; same-direction seeds 2/3)
- A: p040 far20, gamma 0.999 minus reference, detour: -0.253 (pooled seed s.e. 0.159; not > 2 s.e.; same-direction seeds 2/3)
- B: d20 (p050), gamma 0.99 minus reference, success: -0.288 (pooled seed s.e. 0.053; > 2 s.e.; same-direction seeds 3/3)
- B: d20 (p050), gamma 0.99 minus reference, detour: -0.405 (pooled seed s.e. 0.069; > 2 s.e.; same-direction seeds 3/3)
- C: d10 (p050), gamma 0.999 minus reference, success: -0.243 (pooled seed s.e. 0.064; > 2 s.e.; same-direction seeds 3/3)
- C: d10 (p050), gamma 0.999 minus reference, detour: -0.336 (pooled seed s.e. 0.113; > 2 s.e.; same-direction seeds 3/3)
- d05 vanilla (p050, gamma 0.999) minus reference, success: -0.376 (pooled seed s.e. 0.052; > 2 s.e.; same-direction seeds 3/3)
- d05 vanilla (p050, gamma 0.999) minus reference, detour: -0.596 (pooled seed s.e. 0.069; > 2 s.e.; same-direction seeds 3/3)
