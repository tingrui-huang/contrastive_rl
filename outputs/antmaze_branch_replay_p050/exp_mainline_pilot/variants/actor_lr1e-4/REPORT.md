# Variant actor_lr1e-4: {'actor_learning_rate': 0.0001} (everything else as the pilot)

Sealed 2026-09-19 09:23:58.  Same 300 evaluation episodes (seed 3909, mode).  Criteria and rules: `manifest.json`.

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
| actor_lr1e-4 O/seed_0 | 0.247 | 0.020 | 0.710 | 0.043 | 0.959 | 0.013 | 155 |
| actor_lr1e-4 O/seed_1 | 0.250 | 0.027 | 0.710 | 0.040 | 0.973 | 0.013 | 154 |
| actor_lr1e-4 O/seed_2 | 0.247 | 0.027 | 0.733 | 0.020 | 0.959 | 0.013 | 139 |
| actor_lr1e-4 CF/seed_0 | 0.270 | 0.050 | 0.713 | 0.017 | 0.986 | 0.035 | 145 |
| actor_lr1e-4 CF/seed_1 | 0.297 | 0.093 | 0.670 | 0.033 | 0.959 | 0.080 | 167 |
| actor_lr1e-4 CF/seed_2 | 0.297 | 0.063 | 0.693 | 0.010 | 1.000 | 0.066 | 147 |

## Paired differences (per seed; seed mean, seed s.e., episode-bootstrap s.e.; rule > 2 seed s.e. and 3/3)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(actor_lr1e-4) - CF(base) | +0.003 / -0.023 / -0.183 | -0.068 | 0.058 | 0.016 | 2/3 | not met |
| O(actor_lr1e-4) - O(base) | -0.003 / -0.013 / -0.003 | -0.007 | 0.003 | 0.005 | 3/3 | not met |
| CF(actor_lr1e-4) - O(actor_lr1e-4) | +0.023 / +0.047 / +0.050 | +0.040 | 0.008 | 0.009 | 3/3 | met |
| CF(actor_lr1e-4) - start | +0.027 / +0.053 / +0.053 | +0.044 | 0.009 | 0.009 | 3/3 | met |
| CF(base) - start | +0.023 / +0.077 / +0.237 | +0.112 | 0.064 | 0.017 | 3/3 | not met |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(actor_lr1e-4) - CF(base) | -0.057 / -0.073 / -0.437 | -0.189 | 0.124 | 0.015 | 3/3 | not met |
| O(actor_lr1e-4) - O(base) | -0.003 / -0.003 / +0.023 | +0.006 | 0.009 | 0.006 | 1/3 | not met |
| CF(actor_lr1e-4) - O(actor_lr1e-4) | +0.030 / +0.067 / +0.037 | +0.044 | 0.011 | 0.010 | 3/3 | met |
| CF(actor_lr1e-4) - start | +0.047 / +0.090 / +0.060 | +0.066 | 0.013 | 0.012 | 3/3 | met |
| CF(base) - start | +0.103 / +0.163 / +0.497 | +0.254 | 0.122 | 0.017 | 3/3 | met |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(actor_lr1e-4) - CF(base) | +0.087 / +0.103 / +0.433 | +0.208 | 0.113 | 0.016 | 3/3 | not met |
| O(actor_lr1e-4) - O(base) | +0.000 / -0.013 / -0.007 | -0.007 | 0.004 | 0.006 | 2/3 | not met |
| CF(actor_lr1e-4) - O(actor_lr1e-4) | +0.003 / -0.040 / -0.040 | -0.026 | 0.014 | 0.009 | 2/3 | not met |
| CF(actor_lr1e-4) - start | -0.027 / -0.070 / -0.047 | -0.048 | 0.013 | 0.013 | 3/3 | not met |
| CF(base) - start | -0.113 / -0.173 / -0.480 | -0.256 | 0.114 | 0.019 | 3/3 | not met |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF(actor_lr1e-4) - CF(base) | -0.090 / -0.080 / -0.250 | -0.140 | 0.055 | 0.013 | 3/3 | not met |
| O(actor_lr1e-4) - O(base) | +0.003 / +0.027 / +0.010 | +0.013 | 0.007 | 0.006 | 3/3 | not met |
| CF(actor_lr1e-4) - O(actor_lr1e-4) | -0.027 / -0.007 / -0.010 | -0.014 | 0.006 | 0.007 | 3/3 | not met |
| CF(actor_lr1e-4) - start | +0.000 / +0.017 / -0.007 | +0.003 | 0.007 | 0.008 | 1/3 | not met |
| CF(base) - start | +0.090 / +0.097 / +0.243 | +0.143 | 0.050 | 0.014 | 3/3 | met |

## Criteria

1. native success CF(actor_lr1e-4) - CF(base): -0.068 (seed s.e. 0.058, 2/3) -> NOT MET
2. detour gain kept: CF(actor_lr1e-4) - CF(base) detour -0.189 (seed s.e. 0.124); CF(actor_lr1e-4) - O(actor_lr1e-4) detour +0.044 (3/3) -> KEPT
3. continuation from the same handover states: see diag_traj/cont_variant_<name>.json / the diag REPORT section 8.

