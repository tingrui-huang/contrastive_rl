# Variant bc0: {'bc_coef': 0.0} (everything else as the pilot)

Sealed 2026-09-19 10:52:33.  Same 300 evaluation episodes (seed 3909, mode).  Criteria and rules: `manifest.json`.

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
| bc0 O/seed_0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0 O/seed_1 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0 O/seed_2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0 CF/seed_0 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0 CF/seed_1 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |
| bc0 CF/seed_2 | 0.000 | 0.007 | 0.000 | 1.000 | 0.000 | 0.000 | 800 |

## Paired differences (per seed; seed mean, seed s.e., episode-bootstrap s.e.; rule > 2 seed s.e. and 3/3)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0) - CF(base) | -0.267 / -0.320 / -0.480 | -0.356 | 0.064 | 0.020 | 3/3 | not met |
| O(bc0) - O(base) | -0.250 / -0.263 / -0.250 | -0.254 | 0.004 | 0.025 | 3/3 | not met |
| CF(bc0) - O(bc0) | +0.000 / +0.000 / +0.000 | +0.000 | 0.000 | 0.000 | 0/3 | not met |
| CF(bc0) - start | -0.243 / -0.243 / -0.243 | -0.243 | 0.000 | 0.025 | 3/3 | not met |
| CF(base) - start | +0.023 / +0.077 / +0.237 | +0.112 | 0.064 | 0.017 | 3/3 | not met |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0) - CF(base) | -0.107 / -0.167 / -0.493 | -0.256 | 0.120 | 0.017 | 3/3 | not met |
| O(bc0) - O(base) | -0.023 / -0.030 / -0.003 | -0.019 | 0.008 | 0.006 | 3/3 | not met |
| CF(bc0) - O(bc0) | +0.000 / +0.000 / +0.007 | +0.002 | 0.002 | 0.002 | 1/3 | not met |
| CF(bc0) - start | -0.003 / -0.003 / +0.003 | -0.001 | 0.002 | 0.004 | 2/3 | not met |
| CF(base) - start | +0.103 / +0.163 / +0.497 | +0.254 | 0.122 | 0.017 | 3/3 | met |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0) - CF(base) | -0.627 / -0.567 / -0.260 | -0.484 | 0.114 | 0.023 | 3/3 | not met |
| O(bc0) - O(base) | -0.710 / -0.723 / -0.740 | -0.724 | 0.009 | 0.025 | 3/3 | not met |
| CF(bc0) - O(bc0) | +0.000 / +0.000 / +0.000 | +0.000 | 0.000 | 0.000 | 0/3 | not met |
| CF(bc0) - start | -0.740 / -0.740 / -0.740 | -0.740 | 0.000 | 0.025 | 3/3 | not met |
| CF(base) - start | -0.113 / -0.173 / -0.480 | -0.256 | 0.114 | 0.019 | 3/3 | not met |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(bc0) - CF(base) | +0.893 / +0.887 / +0.740 | +0.840 | 0.050 | 0.013 | 3/3 | met |
| O(bc0) - O(base) | +0.960 / +0.987 / +0.990 | +0.979 | 0.009 | 0.006 | 3/3 | met |
| CF(bc0) - O(bc0) | +0.000 / +0.000 / +0.000 | +0.000 | 0.000 | 0.000 | 0/3 | not met |
| CF(bc0) - start | +0.983 / +0.983 / +0.983 | +0.983 | 0.000 | 0.008 | 3/3 | met |
| CF(base) - start | +0.090 / +0.097 / +0.243 | +0.143 | 0.050 | 0.014 | 3/3 | met |

## Criteria

1. native success CF(bc0) - CF(base): -0.356 (seed s.e. 0.064, 3/3) -> NOT MET
2. detour gain kept: CF(bc0) - CF(base) detour -0.256 (seed s.e. 0.120); CF(bc0) - O(bc0) detour +0.002 (1/3) -> NOT KEPT
3. continuation from the same handover states: see diag_traj/cont_variant_<name>.json / the diag REPORT section 8.

