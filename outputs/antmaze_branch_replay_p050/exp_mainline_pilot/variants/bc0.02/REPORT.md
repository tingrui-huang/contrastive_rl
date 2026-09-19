# Variant bc0.02: {'bc_coef': 0.02} (everything else as the pilot)

Sealed 2026-09-19 10:19:59.  Same 300 evaluation episodes (seed 3909, mode).  Criteria and rules: `manifest.json`.

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
| bc0.02 O/seed_0 | 0.203 | 0.043 | 0.553 | 0.243 | 0.757 | 0.022 | 299 |
| bc0.02 O/seed_1 | 0.240 | 0.000 | 0.743 | 0.017 | 0.959 | 0.004 | 136 |
| bc0.02 O/seed_2 | 0.223 | 0.000 | 0.657 | 0.120 | 0.878 | 0.009 | 211 |
| bc0.02 CF/seed_0 | 0.000 | 0.117 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0.02 CF/seed_1 | 0.000 | 0.400 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0.02 CF/seed_2 | 0.000 | 0.017 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |

## Paired differences (per seed; seed mean, seed s.e., episode-bootstrap s.e.; rule > 2 seed s.e. and 3/3)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0.02) - CF(base) | -0.267 / -0.320 / -0.480 | -0.356 | 0.064 | 0.020 | 3/3 | not met |
| O(bc0.02) - O(base) | -0.047 / -0.023 / -0.027 | -0.032 | 0.007 | 0.009 | 3/3 | not met |
| CF(bc0.02) - O(bc0.02) | -0.203 / -0.240 / -0.223 | -0.222 | 0.011 | 0.022 | 3/3 | not met |
| CF(bc0.02) - start | -0.243 / -0.243 / -0.243 | -0.243 | 0.000 | 0.025 | 3/3 | not met |
| CF(base) - start | +0.023 / +0.077 / +0.237 | +0.112 | 0.064 | 0.017 | 3/3 | not met |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0.02) - CF(base) | +0.010 / +0.233 / -0.483 | -0.080 | 0.212 | 0.019 | 1/3 | not met |
| O(bc0.02) - O(base) | +0.020 / -0.030 / -0.003 | -0.004 | 0.014 | 0.007 | 2/3 | not met |
| CF(bc0.02) - O(bc0.02) | +0.073 / +0.400 / +0.017 | +0.163 | 0.119 | 0.012 | 3/3 | not met |
| CF(bc0.02) - start | +0.113 / +0.397 / +0.013 | +0.174 | 0.115 | 0.012 | 3/3 | not met |
| CF(base) - start | +0.103 / +0.163 / +0.497 | +0.254 | 0.122 | 0.017 | 3/3 | met |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0.02) - CF(base) | -0.627 / -0.567 / -0.260 | -0.484 | 0.114 | 0.023 | 3/3 | not met |
| O(bc0.02) - O(base) | -0.157 / +0.020 / -0.083 | -0.073 | 0.051 | 0.011 | 2/3 | not met |
| CF(bc0.02) - O(bc0.02) | -0.553 / -0.743 / -0.657 | -0.651 | 0.055 | 0.024 | 3/3 | not met |
| CF(bc0.02) - start | -0.740 / -0.740 / -0.740 | -0.740 | 0.000 | 0.025 | 3/3 | not met |
| CF(base) - start | -0.113 / -0.173 / -0.480 | -0.256 | 0.114 | 0.019 | 3/3 | not met |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0.02) - CF(base) | +0.893 / +0.887 / +0.740 | +0.840 | 0.050 | 0.013 | 3/3 | met |
| O(bc0.02) - O(base) | +0.203 / +0.003 / +0.110 | +0.106 | 0.058 | 0.012 | 3/3 | not met |
| CF(bc0.02) - O(bc0.02) | +0.757 / +0.983 / +0.880 | +0.873 | 0.066 | 0.012 | 3/3 | met |
| CF(bc0.02) - start | +0.983 / +0.983 / +0.983 | +0.983 | 0.000 | 0.008 | 3/3 | met |
| CF(base) - start | +0.090 / +0.097 / +0.243 | +0.143 | 0.050 | 0.014 | 3/3 | met |

## Criteria

1. native success CF(bc0.02) - CF(base): -0.356 (seed s.e. 0.064, 3/3) -> NOT MET
2. detour gain kept: CF(bc0.02) - CF(base) detour -0.080 (seed s.e. 0.212); CF(bc0.02) - O(bc0.02) detour +0.163 (3/3) -> KEPT
3. continuation from the same handover states: see diag_traj/cont_variant_<name>.json / the diag REPORT section 8.

