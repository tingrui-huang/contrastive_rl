# Variant critic_clip0.1: {'critic_clip': 0.1} (everything else as the pilot)

Sealed 2026-09-19 15:16:30.  Same 300 evaluation episodes (seed 3909, mode).  Criteria and rules: `manifest.json`.

## Per policy

| policy | success | detour | death | timeout | success no hazard | success hazard | mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| base start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 | 137 |
| base O/seed_0 | 0.250 | 0.023 | 0.710 | 0.040 | 0.973 | 0.013 | 152 |
| base O/seed_1 | 0.263 | 0.030 | 0.723 | 0.013 | 0.986 | 0.027 | 136 |
| base O/seed_2 | 0.250 | 0.003 | 0.740 | 0.010 | 1.000 | 0.004 | 130 |
| base CF/seed_0 | 0.267 | 0.107 | 0.627 | 0.107 | 0.838 | 0.080 | 222 |
| base CF/seed_1 | 0.320 | 0.167 | 0.567 | 0.113 | 0.892 | 0.133 | 241 |
| base CF/seed_2 | 0.480 | 0.500 | 0.260 | 0.260 | 0.689 | 0.412 | 423 |
| critic_clip0.1 O/seed_0 | 0.257 | 0.027 | 0.723 | 0.020 | 0.986 | 0.018 | 140 |
| critic_clip0.1 O/seed_1 | 0.247 | 0.003 | 0.743 | 0.010 | 0.986 | 0.004 | 129 |
| critic_clip0.1 O/seed_2 | 0.260 | 0.023 | 0.720 | 0.020 | 1.000 | 0.018 | 141 |
| critic_clip0.1 CF/seed_0 | 0.457 | 0.567 | 0.193 | 0.350 | 0.676 | 0.385 | 512 |
| critic_clip0.1 CF/seed_1 | 0.513 | 0.517 | 0.307 | 0.180 | 0.784 | 0.425 | 390 |
| critic_clip0.1 CF/seed_2 | 0.440 | 0.360 | 0.403 | 0.157 | 0.838 | 0.310 | 349 |

## Paired differences (per seed; seed mean, seed s.e., episode-bootstrap s.e.; rule > 2 seed s.e. and 3/3)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - CF(base) | +0.190 / +0.193 / -0.040 | +0.114 | 0.077 | 0.020 | 2/3 | not met |
| O(critic_clip0.1) - O(base) | +0.007 / -0.017 / +0.010 | +0.000 | 0.008 | 0.004 | 2/3 | not met |
| CF(critic_clip0.1) - O(critic_clip0.1) | +0.200 / +0.267 / +0.180 | +0.216 | 0.026 | 0.023 | 3/3 | met |
| CF(critic_clip0.1) - start | +0.213 / +0.270 / +0.197 | +0.227 | 0.022 | 0.023 | 3/3 | met |
| CF(base) - start | +0.023 / +0.077 / +0.237 | +0.112 | 0.064 | 0.017 | 3/3 | not met |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - CF(base) | +0.460 / +0.350 / -0.140 | +0.223 | 0.184 | 0.020 | 2/3 | not met |
| O(critic_clip0.1) - O(base) | +0.003 / -0.027 / +0.020 | -0.001 | 0.014 | 0.005 | 1/3 | not met |
| CF(critic_clip0.1) - O(critic_clip0.1) | +0.540 / +0.513 / +0.337 | +0.463 | 0.064 | 0.021 | 3/3 | met |
| CF(critic_clip0.1) - start | +0.563 / +0.513 / +0.357 | +0.478 | 0.062 | 0.021 | 3/3 | met |
| CF(base) - start | +0.103 / +0.163 / +0.497 | +0.254 | 0.122 | 0.017 | 3/3 | met |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - CF(base) | -0.433 / -0.260 / +0.143 | -0.183 | 0.171 | 0.017 | 2/3 | not met |
| O(critic_clip0.1) - O(base) | +0.013 / +0.020 / -0.020 | +0.004 | 0.012 | 0.006 | 2/3 | not met |
| CF(critic_clip0.1) - O(critic_clip0.1) | -0.530 / -0.437 / -0.317 | -0.428 | 0.062 | 0.023 | 3/3 | not met |
| CF(critic_clip0.1) - start | -0.547 / -0.433 / -0.337 | -0.439 | 0.061 | 0.024 | 3/3 | not met |
| CF(base) - start | -0.113 / -0.173 / -0.480 | -0.256 | 0.114 | 0.019 | 3/3 | not met |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(critic_clip0.1) - CF(base) | +0.243 / +0.067 / -0.103 | +0.069 | 0.100 | 0.017 | 2/3 | not met |
| O(critic_clip0.1) - O(base) | -0.020 / -0.003 / +0.010 | -0.004 | 0.009 | 0.005 | 2/3 | not met |
| CF(critic_clip0.1) - O(critic_clip0.1) | +0.330 / +0.170 / +0.137 | +0.212 | 0.060 | 0.014 | 3/3 | met |
| CF(critic_clip0.1) - start | +0.333 / +0.163 / +0.140 | +0.212 | 0.061 | 0.016 | 3/3 | met |
| CF(base) - start | +0.090 / +0.097 / +0.243 | +0.143 | 0.050 | 0.014 | 3/3 | met |

## Criteria

1. native success CF(critic_clip0.1) - CF(base): +0.114 (seed s.e. 0.077, 2/3) -> NOT MET
2. detour gain kept: CF(critic_clip0.1) - CF(base) detour +0.223 (seed s.e. 0.184); CF(critic_clip0.1) - O(critic_clip0.1) detour +0.463 (3/3) -> KEPT
3. continuation from the same handover states: see diag_traj/cont_variant_<name>.json / the diag REPORT section 8.

## Route ledger (far route = the env's detour label, the top-west corner reached; completion = successes / far-route episodes)

| policy | far route: n (share) | completed | far-route timeouts / deaths | completion | shortcut: n / success / deaths / timeouts | no route: n / deaths / timeouts | success | success if far-route timeouts rescued |
|---|---:|---:|---|---:|---|---|---:|---:|
| base start | 1 (0.00) | 0 | 1 / 0 | 0.000 | 291 / 73 / 217 / 1 | 8 / 5 / 3 | 0.243 | 0.247 |
| base O/seed_0 | 7 (0.02) | 4 | 3 / 0 | 0.571 | 277 / 71 / 205 / 1 | 16 / 8 / 8 | 0.250 | 0.260 |
| base O/seed_1 | 9 (0.03) | 7 | 2 / 0 | 0.778 | 283 / 72 / 211 / 0 | 8 / 6 / 2 | 0.263 | 0.270 |
| base O/seed_2 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 285 / 74 / 211 / 0 | 14 / 11 / 3 | 0.250 | 0.250 |
| base CF/seed_0 | 32 (0.11) | 20 | 12 / 0 | 0.625 | 252 / 60 / 182 / 10 | 16 / 6 / 10 | 0.267 | 0.307 |
| base CF/seed_1 | 50 (0.17) | 38 | 12 / 0 | 0.760 | 217 / 58 / 156 / 3 | 33 / 14 / 19 | 0.320 | 0.360 |
| base CF/seed_2 | 150 (0.50) | 112 | 38 / 0 | 0.747 | 111 / 32 / 75 / 4 | 39 / 3 / 36 | 0.480 | 0.607 |
| critic_clip0.1 O/seed_0 | 8 (0.03) | 5 | 3 / 0 | 0.625 | 286 / 72 / 214 / 0 | 6 / 3 / 3 | 0.257 | 0.267 |
| critic_clip0.1 O/seed_1 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 290 / 73 / 217 / 0 | 9 / 6 / 3 | 0.247 | 0.247 |
| critic_clip0.1 O/seed_2 | 7 (0.02) | 5 | 2 / 0 | 0.714 | 284 / 73 / 211 / 0 | 9 / 5 / 4 | 0.260 | 0.267 |
| critic_clip0.1 CF/seed_0 | 170 (0.57) | 111 | 59 / 0 | 0.653 | 86 / 26 / 52 / 8 | 44 / 6 / 38 | 0.457 | 0.653 |
| critic_clip0.1 CF/seed_1 | 155 (0.52) | 122 | 33 / 0 | 0.787 | 123 / 32 / 89 / 2 | 22 / 3 / 19 | 0.513 | 0.623 |
| critic_clip0.1 CF/seed_2 | 108 (0.36) | 87 | 21 / 0 | 0.806 | 166 / 45 / 113 / 8 | 26 / 8 / 18 | 0.440 | 0.510 |

