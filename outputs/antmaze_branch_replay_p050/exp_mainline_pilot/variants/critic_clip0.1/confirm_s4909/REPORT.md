# Confirmation of the sealed recipe (critic_clip0.1) on a fresh evaluation draw (seed 4909)

Sealed 2026-09-19 16:41:14 (`manifest.json`: checkpoint hashes, draw, comparisons, rule) before any evaluation on this draw.  300 natural episodes, env seed 4909, p_active 0.5 / 0.5, horizon 800, mode policy; the SAME final checkpoints as the development result (seed 3909); no checkpoint selection.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3/3.

## Per policy

| policy | success | detour | death | timeout | success no hazard (n) | success hazard (n) | mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| start | 0.260 | 0.007 | 0.720 | 0.020 | 0.987 (78) | 0.005 (222) | 143 |
| O/seed_0 | 0.260 | 0.023 | 0.717 | 0.023 | 0.974 (78) | 0.009 (222) | 142 |
| O/seed_1 | 0.260 | 0.017 | 0.727 | 0.013 | 0.962 (78) | 0.014 (222) | 135 |
| O/seed_2 | 0.260 | 0.017 | 0.727 | 0.013 | 0.962 (78) | 0.014 (222) | 135 |
| CF/seed_0 | 0.480 | 0.603 | 0.200 | 0.320 | 0.667 (78) | 0.414 (222) | 501 |
| CF/seed_1 | 0.447 | 0.507 | 0.353 | 0.200 | 0.692 (78) | 0.360 (222) | 385 |
| CF/seed_2 | 0.430 | 0.330 | 0.430 | 0.140 | 0.808 (78) | 0.297 (222) | 325 |
| base:O/seed_0 | 0.253 | 0.030 | 0.700 | 0.047 | 0.872 (78) | 0.036 (222) | 158 |
| base:O/seed_1 | 0.270 | 0.037 | 0.713 | 0.017 | 0.974 (78) | 0.023 (222) | 140 |
| base:O/seed_2 | 0.260 | 0.007 | 0.737 | 0.003 | 0.987 (78) | 0.005 (222) | 126 |
| base:CF/seed_0 | 0.287 | 0.130 | 0.627 | 0.087 | 0.872 (78) | 0.081 (222) | 219 |
| base:CF/seed_1 | 0.320 | 0.187 | 0.557 | 0.123 | 0.859 (78) | 0.131 (222) | 252 |
| base:CF/seed_2 | 0.417 | 0.493 | 0.283 | 0.300 | 0.679 (78) | 0.324 (222) | 428 |

## Route ledger (far route = the env's detour label, the top-west corner reached)

| policy | far route: n (share) | completed | far-route timeouts / deaths | completion | shortcut: n / success / deaths / timeouts | no route: n / deaths / timeouts | success | success if far-route timeouts rescued |
|---|---:|---:|---|---:|---|---|---:|---:|
| start | 2 (0.01) | 1 | 1 / 0 | 0.500 | 286 / 77 / 208 / 1 | 12 / 8 / 4 | 0.260 | 0.263 |
| O/seed_0 | 7 (0.02) | 6 | 1 / 0 | 0.857 | 280 / 72 / 208 / 0 | 13 / 7 / 6 | 0.260 | 0.263 |
| O/seed_1 | 5 (0.02) | 5 | 0 / 0 | 1.000 | 289 / 73 / 216 / 0 | 6 / 2 / 4 | 0.260 | 0.260 |
| O/seed_2 | 5 (0.02) | 5 | 0 / 0 | 1.000 | 283 / 73 / 210 / 0 | 12 / 8 / 4 | 0.260 | 0.260 |
| CF/seed_0 | 181 (0.60) | 119 | 62 / 0 | 0.657 | 87 / 25 / 57 / 5 | 32 / 3 / 29 | 0.480 | 0.687 |
| CF/seed_1 | 152 (0.51) | 107 | 45 / 0 | 0.704 | 135 / 27 / 105 / 3 | 13 / 1 / 12 | 0.447 | 0.597 |
| CF/seed_2 | 99 (0.33) | 85 | 14 / 0 | 0.859 | 171 / 44 / 119 / 8 | 30 / 10 / 20 | 0.430 | 0.477 |
| base:O/seed_0 | 9 (0.03) | 8 | 1 / 0 | 0.889 | 271 / 68 / 203 / 0 | 20 / 7 / 13 | 0.253 | 0.257 |
| base:O/seed_1 | 11 (0.04) | 9 | 2 / 0 | 0.818 | 276 / 72 / 204 / 0 | 13 / 10 / 3 | 0.270 | 0.277 |
| base:O/seed_2 | 2 (0.01) | 2 | 0 / 0 | 1.000 | 290 / 76 / 214 / 0 | 8 / 7 / 1 | 0.260 | 0.260 |
| base:CF/seed_0 | 39 (0.13) | 29 | 10 / 0 | 0.744 | 243 / 57 / 178 / 8 | 18 / 10 / 8 | 0.287 | 0.320 |
| base:CF/seed_1 | 56 (0.19) | 42 | 14 / 0 | 0.750 | 202 / 54 / 146 / 2 | 42 / 21 / 21 | 0.320 | 0.367 |
| base:CF/seed_2 | 148 (0.49) | 93 | 55 / 0 | 0.628 | 111 / 32 / 78 / 1 | 41 / 7 / 34 | 0.417 | 0.600 |

## Paired differences on the common episodes (per seed; seed mean, seed s.e., episode-bootstrap s.e.)

### success

| comparison | per seed (episode s.e.) | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] | +0.220 (0.034) / +0.187 (0.033) / +0.170 (0.028) | +0.192 | 0.015 | 0.024 | 3/3 | met |
| CF(critic_clip0.1) - start [practical] | +0.220 (0.034) / +0.187 (0.032) / +0.170 (0.028) | +0.192 | 0.015 | 0.025 | 3/3 | met |
| O(critic_clip0.1) - start | +0.000 (0.008) / +0.000 (0.009) / +0.000 (0.009) | +0.000 | 0.000 | 0.007 | 0/3 | - |
| base CF - base O [reference] | +0.033 (0.020) / +0.050 (0.022) / +0.157 (0.032) | +0.080 | 0.039 | 0.016 | 3/3 | - |
| CF(critic_clip0.1) - CF(base) [reference] | +0.193 (0.034) / +0.127 (0.032) / +0.013 (0.034) | +0.111 | 0.053 | 0.020 | 3/3 | - |
| O(critic_clip0.1) - O(base) [reference] | +0.007 (0.012) / -0.010 (0.010) / +0.000 (0.009) | -0.001 | 0.005 | 0.006 | 1/3 | - |

### detour

| comparison | per seed (episode s.e.) | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] | +0.580 (0.029) / +0.490 (0.030) / +0.313 (0.028) | +0.461 | 0.078 | 0.021 | 3/3 | - |
| CF(critic_clip0.1) - start [practical] | +0.597 (0.028) / +0.500 (0.030) / +0.323 (0.028) | +0.473 | 0.080 | 0.022 | 3/3 | - |
| O(critic_clip0.1) - start | +0.017 (0.010) / +0.010 (0.009) / +0.010 (0.009) | +0.012 | 0.002 | 0.007 | 3/3 | - |
| base CF - base O [reference] | +0.100 (0.020) / +0.150 (0.023) / +0.487 (0.029) | +0.246 | 0.121 | 0.017 | 3/3 | - |
| CF(critic_clip0.1) - CF(base) [reference] | +0.473 (0.031) / +0.320 (0.030) / -0.163 (0.033) | +0.210 | 0.192 | 0.018 | 2/3 | - |
| O(critic_clip0.1) - O(base) [reference] | -0.007 (0.011) / -0.020 (0.010) / +0.010 (0.007) | -0.006 | 0.009 | 0.006 | 2/3 | - |

### failure

| comparison | per seed (episode s.e.) | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] | -0.517 (0.029) / -0.373 (0.028) / -0.297 (0.026) | -0.396 | 0.064 | 0.023 | 3/3 | - |
| CF(critic_clip0.1) - start [practical] | -0.520 (0.029) / -0.367 (0.029) / -0.290 (0.028) | -0.392 | 0.068 | 0.024 | 3/3 | - |
| O(critic_clip0.1) - start | -0.003 (0.011) / +0.007 (0.011) / +0.007 (0.011) | +0.003 | 0.003 | 0.010 | 2/3 | - |
| base CF - base O [reference] | -0.073 (0.019) / -0.157 (0.021) / -0.453 (0.029) | -0.228 | 0.115 | 0.017 | 3/3 | - |
| CF(critic_clip0.1) - CF(base) [reference] | -0.427 (0.029) / -0.203 (0.024) / +0.147 (0.030) | -0.161 | 0.167 | 0.017 | 2/3 | - |
| O(critic_clip0.1) - O(base) [reference] | +0.017 (0.010) / +0.013 (0.011) / -0.010 (0.007) | +0.007 | 0.008 | 0.006 | 2/3 | - |

### timeout

| comparison | per seed (episode s.e.) | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] | +0.297 (0.028) / +0.187 (0.024) / +0.127 (0.021) | +0.203 | 0.050 | 0.016 | 3/3 | - |
| CF(critic_clip0.1) - start [practical] | +0.300 (0.027) / +0.180 (0.025) / +0.120 (0.022) | +0.200 | 0.053 | 0.016 | 3/3 | - |
| O(critic_clip0.1) - start | +0.003 (0.011) / -0.007 (0.011) / -0.007 (0.011) | -0.003 | 0.003 | 0.009 | 2/3 | - |
| base CF - base O [reference] | +0.040 (0.019) / +0.107 (0.020) / +0.297 (0.027) | +0.148 | 0.077 | 0.013 | 3/3 | - |
| CF(critic_clip0.1) - CF(base) [reference] | +0.233 (0.031) / +0.077 (0.030) / -0.160 (0.031) | +0.050 | 0.114 | 0.018 | 2/3 | - |
| O(critic_clip0.1) - O(base) [reference] | -0.023 (0.014) / -0.003 (0.009) / +0.010 (0.007) | -0.006 | 0.010 | 0.006 | 2/3 | - |

## Development draw (seed 3909) vs confirmation draw (seed 4909)

| quantity | development | confirmation |
|---|---|---|
| O(critic_clip0.1) success per seed | 0.257 / 0.247 / 0.260 | 0.260 / 0.260 / 0.260 |
| O(critic_clip0.1) detour per seed | 0.027 / 0.003 / 0.023 | 0.023 / 0.017 / 0.017 |
| O(critic_clip0.1) death per seed | 0.723 / 0.743 / 0.720 | 0.717 / 0.727 / 0.727 |
| O(critic_clip0.1) timeout per seed | 0.020 / 0.010 / 0.020 | 0.023 / 0.013 / 0.013 |
| CF(critic_clip0.1) success per seed | 0.457 / 0.513 / 0.440 | 0.480 / 0.447 / 0.430 |
| CF(critic_clip0.1) detour per seed | 0.567 / 0.517 / 0.360 | 0.603 / 0.507 / 0.330 |
| CF(critic_clip0.1) death per seed | 0.193 / 0.307 / 0.403 | 0.200 / 0.353 / 0.430 |
| CF(critic_clip0.1) timeout per seed | 0.350 / 0.180 / 0.157 | 0.320 / 0.200 / 0.140 |
| start success / detour | 0.243 / 0.003 | 0.260 / 0.007 |
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] success | +0.216 (seed s.e. 0.026, 3/3), MET | +0.192 (seed s.e. 0.015, 3/3), MET |
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] detour | +0.463 (seed s.e. 0.064, 3/3) | +0.461 (seed s.e. 0.078, 3/3) |
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] failure | -0.428 (seed s.e. 0.062, 3/3) | -0.396 (seed s.e. 0.064, 3/3) |
| CF(critic_clip0.1) - O(critic_clip0.1) [primary] timeout | +0.212 (seed s.e. 0.060, 3/3) | +0.203 (seed s.e. 0.050, 3/3) |
| CF(critic_clip0.1) - start [practical] success | +0.227 (seed s.e. 0.022, 3/3), MET | +0.192 (seed s.e. 0.015, 3/3), MET |
| CF(critic_clip0.1) - start [practical] detour | +0.478 (seed s.e. 0.062, 3/3) | +0.473 (seed s.e. 0.080, 3/3) |
| CF(critic_clip0.1) - start [practical] failure | -0.439 (seed s.e. 0.061, 3/3) | -0.392 (seed s.e. 0.068, 3/3) |
| CF(critic_clip0.1) - start [practical] timeout | +0.212 (seed s.e. 0.061, 3/3) | +0.200 (seed s.e. 0.053, 3/3) |

Far-route ledger on the development draw: 

| policy | far route: n (share) | completed | far-route timeouts / deaths | completion | shortcut: n / success / deaths / timeouts | no route: n / deaths / timeouts | success | success if far-route timeouts rescued |
|---|---:|---:|---|---:|---|---|---:|---:|
| start | 1 (0.00) | 0 | 1 / 0 | 0.000 | 291 / 73 / 217 / 1 | 8 / 5 / 3 | 0.243 | 0.247 |
| O/seed_0 | 8 (0.03) | 5 | 3 / 0 | 0.625 | 286 / 72 / 214 / 0 | 6 / 3 / 3 | 0.257 | 0.267 |
| O/seed_1 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 290 / 73 / 217 / 0 | 9 / 6 / 3 | 0.247 | 0.247 |
| O/seed_2 | 7 (0.02) | 5 | 2 / 0 | 0.714 | 284 / 73 / 211 / 0 | 9 / 5 / 4 | 0.260 | 0.267 |
| CF/seed_0 | 170 (0.57) | 111 | 59 / 0 | 0.653 | 86 / 26 / 52 / 8 | 44 / 6 / 38 | 0.457 | 0.653 |
| CF/seed_1 | 155 (0.52) | 122 | 33 / 0 | 0.787 | 123 / 32 / 89 / 2 | 22 / 3 / 19 | 0.513 | 0.623 |
| CF/seed_2 | 108 (0.36) | 87 | 21 / 0 | 0.806 | 166 / 45 / 113 / 8 | 26 / 8 / 18 | 0.440 | 0.510 |

## Verdict

Primary CF(critic_clip0.1) - O(critic_clip0.1) success on the confirmation draw: +0.192 (seed s.e. 0.015, boot 0.024, 3/3) -> MET; practical CF(critic_clip0.1) - start: +0.192 (seed s.e. 0.015, 3/3) -> MET.  **REPRODUCED** on a draw never used for development; the same final checkpoints, evaluated once.  Oracle evidence (simulator futures) under the disclosed optimizer-stabilisation change; not a learned-ETT result.
