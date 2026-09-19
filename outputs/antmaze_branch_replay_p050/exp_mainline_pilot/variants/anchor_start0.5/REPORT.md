# Variant anchor_start0.5: {'anchor_coef': 0.5} (everything else as the pilot)

Sealed 2026-09-19 11:36:36.  Same 300 evaluation episodes (seed 3909, mode).  Criteria and rules: `manifest.json`.

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
| anchor_start0.5 O/seed_0 | 0.260 | 0.040 | 0.697 | 0.043 | 0.932 | 0.040 | 163 |
| anchor_start0.5 O/seed_1 | 0.273 | 0.040 | 0.693 | 0.033 | 1.000 | 0.035 | 160 |
| anchor_start0.5 O/seed_2 | 0.257 | 0.027 | 0.700 | 0.043 | 0.946 | 0.031 | 162 |
| anchor_start0.5 CF/seed_0 | 0.247 | 0.000 | 0.753 | 0.000 | 1.000 | 0.000 | 125 |
| anchor_start0.5 CF/seed_1 | 0.290 | 0.087 | 0.667 | 0.043 | 0.946 | 0.075 | 174 |
| anchor_start0.5 CF/seed_2 | 0.257 | 0.027 | 0.737 | 0.007 | 1.000 | 0.013 | 134 |

## Paired differences (per seed; seed mean, seed s.e., episode-bootstrap s.e.; rule > 2 seed s.e. and 3/3)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(anchor_start0.5) - CF(base) | -0.020 / -0.030 / -0.223 | -0.091 | 0.066 | 0.017 | 3/3 | not met |
| O(anchor_start0.5) - O(base) | +0.010 / +0.010 / +0.007 | +0.009 | 0.001 | 0.009 | 3/3 | met |
| CF(anchor_start0.5) - O(anchor_start0.5) | -0.013 / +0.017 / +0.000 | +0.001 | 0.009 | 0.008 | 1/3 | not met |
| CF(anchor_start0.5) - start | +0.003 / +0.047 / +0.013 | +0.021 | 0.013 | 0.007 | 3/3 | not met |
| CF(base) - start | +0.023 / +0.077 / +0.237 | +0.112 | 0.064 | 0.017 | 3/3 | not met |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(anchor_start0.5) - CF(base) | -0.107 / -0.080 / -0.473 | -0.220 | 0.127 | 0.017 | 3/3 | not met |
| O(anchor_start0.5) - O(base) | +0.017 / +0.010 / +0.023 | +0.017 | 0.004 | 0.009 | 3/3 | met |
| CF(anchor_start0.5) - O(anchor_start0.5) | -0.040 / +0.047 / +0.000 | +0.002 | 0.025 | 0.008 | 1/3 | not met |
| CF(anchor_start0.5) - start | -0.003 / +0.083 / +0.023 | +0.034 | 0.026 | 0.008 | 2/3 | not met |
| CF(base) - start | +0.103 / +0.163 / +0.497 | +0.254 | 0.122 | 0.017 | 3/3 | met |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(anchor_start0.5) - CF(base) | +0.127 / +0.100 / +0.477 | +0.234 | 0.121 | 0.017 | 3/3 | not met |
| O(anchor_start0.5) - O(base) | -0.013 / -0.030 / -0.040 | -0.028 | 0.008 | 0.012 | 3/3 | not met |
| CF(anchor_start0.5) - O(anchor_start0.5) | +0.057 / -0.027 / +0.037 | +0.022 | 0.025 | 0.010 | 2/3 | not met |
| CF(anchor_start0.5) - start | +0.013 / -0.073 / -0.003 | -0.021 | 0.027 | 0.007 | 2/3 | not met |
| CF(base) - start | -0.113 / -0.173 / -0.480 | -0.256 | 0.114 | 0.019 | 3/3 | not met |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(anchor_start0.5) - CF(base) | -0.107 / -0.070 / -0.253 | -0.143 | 0.056 | 0.013 | 3/3 | not met |
| O(anchor_start0.5) - O(base) | +0.003 / +0.020 / +0.033 | +0.019 | 0.009 | 0.009 | 3/3 | met |
| CF(anchor_start0.5) - O(anchor_start0.5) | -0.043 / +0.010 / -0.037 | -0.023 | 0.017 | 0.008 | 2/3 | not met |
| CF(anchor_start0.5) - start | -0.017 / +0.027 / -0.010 | +0.000 | 0.013 | 0.008 | 1/3 | not met |
| CF(base) - start | +0.090 / +0.097 / +0.243 | +0.143 | 0.050 | 0.014 | 3/3 | met |

## Criteria

1. native success CF(anchor_start0.5) - CF(base): -0.091 (seed s.e. 0.066, 3/3) -> NOT MET
2. detour gain kept: CF(anchor_start0.5) - CF(base) detour -0.220 (seed s.e. 0.127); CF(anchor_start0.5) - O(anchor_start0.5) detour +0.002 (1/3) -> NOT KEPT
3. continuation from the same handover states: see diag_traj/cont_variant_<name>.json / the diag REPORT section 8.

