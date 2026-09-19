# Round 2: the round-1 CF agent of each lineage as the continuation policy; futures regenerated; O / CF / CFold retrained from that agent

Sealed 2026-09-19 12:08:50.  Same 300 evaluation episodes (seed 3909, mode).  Rule: mean over the 3 paired lineages > 2 x seed s.e. and 3/3.

## Per policy

| policy | success | detour | death | timeout | success no hazard | success hazard | mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| round1 start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 | 137 |
| round1 O/seed_0 | 0.250 | 0.023 | 0.710 | 0.040 | 0.973 | 0.013 | 152 |
| round1 O/seed_1 | 0.263 | 0.030 | 0.723 | 0.013 | 0.986 | 0.027 | 136 |
| round1 O/seed_2 | 0.250 | 0.003 | 0.740 | 0.010 | 1.000 | 0.004 | 130 |
| round1 CF/seed_0 | 0.267 | 0.107 | 0.627 | 0.107 | 0.838 | 0.080 | 222 |
| round1 CF/seed_1 | 0.320 | 0.167 | 0.567 | 0.113 | 0.892 | 0.133 | 241 |
| round1 CF/seed_2 | 0.480 | 0.500 | 0.260 | 0.260 | 0.689 | 0.412 | 423 |
| round2 O/seed_0 | 0.250 | 0.007 | 0.740 | 0.010 | 0.986 | 0.009 | 130 |
| round2 O/seed_1 | 0.250 | 0.007 | 0.737 | 0.013 | 1.000 | 0.004 | 132 |
| round2 O/seed_2 | 0.253 | 0.010 | 0.740 | 0.007 | 1.000 | 0.009 | 129 |
| round2 CF/seed_0 | 0.240 | 0.000 | 0.727 | 0.033 | 0.973 | 0.000 | 147 |
| round2 CF/seed_1 | 0.240 | 0.000 | 0.753 | 0.007 | 0.973 | 0.000 | 125 |
| round2 CF/seed_2 | 0.250 | 0.100 | 0.633 | 0.117 | 0.878 | 0.044 | 221 |
| round2 CFold/seed_0 | 0.263 | 0.163 | 0.513 | 0.223 | 0.703 | 0.119 | 318 |
| round2 CFold/seed_1 | 0.283 | 0.120 | 0.587 | 0.130 | 0.892 | 0.084 | 237 |
| round2 CFold/seed_2 | 0.260 | 0.210 | 0.447 | 0.293 | 0.581 | 0.155 | 368 |

## Paired differences on the common episodes (per lineage; mean, seed s.e., episode-bootstrap s.e.)

### success

| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF2 - O2 (primary) | -0.010 / -0.010 / -0.003 | -0.008 | 0.002 | 0.006 | 3/3 | not met |
| CF2 - CFold2 (regeneration) | -0.023 / -0.043 / -0.010 | -0.026 | 0.010 | 0.016 | 3/3 | not met |
| CFold2 - O2 | +0.013 / +0.033 / +0.007 | +0.018 | 0.008 | 0.017 | 3/3 | met |
| CF2 - CF1 (practical: the lineage agent) | -0.027 / -0.080 / -0.230 | -0.112 | 0.061 | 0.016 | 3/3 | not met |
| O2 - CF1 | -0.017 / -0.070 / -0.227 | -0.104 | 0.063 | 0.017 | 3/3 | not met |
| CFold2 - CF1 | -0.003 / -0.037 / -0.220 | -0.087 | 0.067 | 0.015 | 3/3 | not met |
| CF2 - start (pilot start agent) | -0.003 / -0.003 / +0.007 | +0.000 | 0.003 | 0.006 | 0/3 | not met |

### detour

| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF2 - O2 (primary) | -0.007 / -0.007 / +0.090 | +0.026 | 0.032 | 0.007 | 1/3 | not met |
| CF2 - CFold2 (regeneration) | -0.163 / -0.120 / -0.110 | -0.131 | 0.016 | 0.015 | 3/3 | not met |
| CFold2 - O2 | +0.157 / +0.113 / +0.200 | +0.157 | 0.025 | 0.016 | 3/3 | met |
| CF2 - CF1 (practical: the lineage agent) | -0.107 / -0.167 / -0.400 | -0.224 | 0.089 | 0.016 | 3/3 | not met |
| O2 - CF1 | -0.100 / -0.160 / -0.490 | -0.250 | 0.121 | 0.017 | 3/3 | not met |
| CFold2 - CF1 | +0.057 / -0.047 / -0.290 | -0.093 | 0.103 | 0.015 | 2/3 | not met |
| CF2 - start (pilot start agent) | -0.003 / -0.003 / +0.097 | +0.030 | 0.033 | 0.007 | 1/3 | not met |

### failure

| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF2 - O2 (primary) | -0.013 / +0.017 / -0.107 | -0.034 | 0.037 | 0.008 | 2/3 | not met |
| CF2 - CFold2 (regeneration) | +0.213 / +0.167 / +0.187 | +0.189 | 0.014 | 0.018 | 3/3 | met |
| CFold2 - O2 | -0.227 / -0.150 / -0.293 | -0.223 | 0.041 | 0.020 | 3/3 | not met |
| CF2 - CF1 (practical: the lineage agent) | +0.100 / +0.187 / +0.373 | +0.220 | 0.081 | 0.016 | 3/3 | met |
| O2 - CF1 | +0.113 / +0.170 / +0.480 | +0.254 | 0.114 | 0.018 | 3/3 | met |
| CFold2 - CF1 | -0.113 / +0.020 / +0.187 | +0.031 | 0.087 | 0.013 | 2/3 | not met |
| CF2 - start (pilot start agent) | -0.013 / +0.013 / -0.107 | -0.036 | 0.036 | 0.010 | 2/3 | not met |

### timeout

| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF2 - O2 (primary) | +0.023 / -0.007 / +0.110 | +0.042 | 0.035 | 0.008 | 2/3 | not met |
| CF2 - CFold2 (regeneration) | -0.190 / -0.123 / -0.177 | -0.163 | 0.020 | 0.016 | 3/3 | not met |
| CFold2 - O2 | +0.213 / +0.117 / +0.287 | +0.206 | 0.049 | 0.017 | 3/3 | met |
| CF2 - CF1 (practical: the lineage agent) | -0.073 / -0.107 / -0.143 | -0.108 | 0.020 | 0.013 | 3/3 | not met |
| O2 - CF1 | -0.097 / -0.100 / -0.253 | -0.150 | 0.052 | 0.013 | 3/3 | not met |
| CFold2 - CF1 | +0.117 / +0.017 / +0.033 | +0.056 | 0.031 | 0.016 | 3/3 | not met |
| CF2 - start (pilot start agent) | +0.017 / -0.010 / +0.100 | +0.036 | 0.033 | 0.011 | 2/3 | not met |

## Reading

Primary CF2 - O2 success -0.008 (seed s.e. 0.002, 3/3): NOT MET.
Regeneration CF2 - CFold2 success -0.026 (seed s.e. 0.010, 3/3): NOT MET.
Practical CF2 - CF1 success -0.112 (seed s.e. 0.061, 3/3): NOT MET.

Oracle evidence (the simulator generated the futures); not a learned-ETT result.
