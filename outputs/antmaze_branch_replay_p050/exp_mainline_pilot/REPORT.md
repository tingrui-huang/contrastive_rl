# Mainline pilot: observational (O) vs counterfactual-oracle (CF) positive futures

Contract `notes/MAINLINE_CONTRACT.md`; manifest `manifest.json` (sealed 2026-09-18 23:54:19); checks `AUDIT.md`.  Evaluation: 300 natural draws, seed 3909, mode policy, the same episodes for every policy.  Uncertainty: per-seed episode s.e. (evaluation), seed s.e. over 3 paired training seeds (training variability), episode-paired bootstrap of the seed mean (evaluation uncertainty with the seeds fixed).  Rule: CF - O mean over the 3 paired seeds > 2 x seed s.e. and 3/3 seeds in the same direction.

## Per policy

| policy | success | detour | death | timeout | success no hazard (n) | success hazard (n) | zone-1 / zone-2 deaths | mean steps |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 (74) | 0.000 (226) | 152 / 70 | 137 |
| O/seed_0 | 0.250 | 0.023 | 0.710 | 0.040 | 0.973 (74) | 0.013 (226) | 146 / 67 | 152 |
| O/seed_1 | 0.263 | 0.030 | 0.723 | 0.013 | 0.986 (74) | 0.027 (226) | 150 / 67 | 136 |
| O/seed_2 | 0.250 | 0.003 | 0.740 | 0.010 | 1.000 (74) | 0.004 (226) | 152 / 70 | 130 |
| CF/seed_0 | 0.267 | 0.107 | 0.627 | 0.107 | 0.838 (74) | 0.080 (226) | 134 / 54 | 222 |
| CF/seed_1 | 0.320 | 0.167 | 0.567 | 0.113 | 0.892 (74) | 0.133 (226) | 118 / 52 | 241 |
| CF/seed_2 | 0.480 | 0.500 | 0.260 | 0.260 | 0.689 (74) | 0.412 (226) | 53 / 25 | 423 |

## Seed means (seed s.e., n = 3)

| arm | success | detour | death | timeout | success hazard |
|---|---|---|---|---|---|
| O | 0.254 (0.004) | 0.019 (0.008) | 0.724 (0.009) | 0.021 (0.009) | 0.015 (0.006) |
| CF | 0.356 (0.064) | 0.258 (0.122) | 0.484 (0.114) | 0.160 (0.050) | 0.208 (0.103) |

## Paired comparisons on the common episodes

### success

| comparison | per seed (episode s.e.) | mean | seed s.e. | episode-bootstrap s.e. | seeds same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF - O (primary) | +0.017 (0.017) / +0.057 (0.020) / +0.230 (0.034) | +0.101 | 0.065 | 0.017 | 3/3 | not met |
| CF - start (practical) | +0.023 (0.018) / +0.077 (0.020) / +0.237 (0.033) | +0.112 | 0.064 | 0.017 | 3/3 | not met |
| O - start | +0.007 (0.007) / +0.020 (0.008) / +0.007 (0.005) | +0.011 | 0.004 | 0.005 | 3/3 | - |

### detour

| comparison | per seed (episode s.e.) | mean | seed s.e. | episode-bootstrap s.e. | seeds same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF - O (primary) | +0.083 (0.017) / +0.137 (0.022) / +0.497 (0.029) | +0.239 | 0.130 | 0.016 | 3/3 | - |
| CF - start (practical) | +0.103 (0.018) / +0.163 (0.022) / +0.497 (0.029) | +0.254 | 0.122 | 0.017 | 3/3 | - |
| O - start | +0.020 (0.009) / +0.027 (0.010) / +0.000 (0.005) | +0.016 | 0.008 | 0.007 | 2/3 | - |

### failure

| comparison | per seed (episode s.e.) | mean | seed s.e. | episode-bootstrap s.e. | seeds same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF - O (primary) | -0.083 (0.019) / -0.157 (0.022) / -0.480 (0.029) | -0.240 | 0.122 | 0.017 | 3/3 | - |
| CF - start (practical) | -0.113 (0.021) / -0.173 (0.023) / -0.480 (0.029) | -0.256 | 0.114 | 0.019 | 3/3 | - |
| O - start | -0.030 (0.014) / -0.017 (0.012) / +0.000 (0.009) | -0.016 | 0.009 | 0.011 | 2/3 | - |

### timeout

| comparison | per seed (episode s.e.) | mean | seed s.e. | episode-bootstrap s.e. | seeds same direction | rule |
|---|---|---:|---:|---:|---|---|
| CF - O (primary) | +0.067 (0.020) / +0.100 (0.017) / +0.250 (0.025) | +0.139 | 0.056 | 0.013 | 3/3 | - |
| CF - start (practical) | +0.090 (0.019) / +0.097 (0.019) / +0.243 (0.026) | +0.143 | 0.050 | 0.014 | 3/3 | - |
| O - start | +0.023 (0.013) / -0.003 (0.009) / -0.007 (0.009) | +0.004 | 0.009 | 0.009 | 1/3 | - |

## Reading

Primary (CF - O, success): +0.101, seed s.e. 0.065, 3/3 seeds; rule NOT MET.
Practical (CF - start, success): +0.112, seed s.e. 0.064, 3/3; rule NOT MET.  Beating a degraded control alone is insufficient; the practical check is required separately.

This is oracle evidence for the sampling design (the simulator generated the counterfactual futures); it is not an offline learned-ETT result.
