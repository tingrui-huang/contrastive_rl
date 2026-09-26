# Detour-demonstration share 5% vs 20%: BC walker, vanilla and branch critics, actors

Sealed 2026-09-18 11:41:09.  Nested 1,000-episode datasets from the p050 pool; same recipe per dataset.  Reading rules: effect = mean over 3 seeds > 2 x pooled seed s.e. with 3/3 seeds in the same direction.

## Effective shares of detour-episode rows

| ratio | dataset episodes | dataset rows | vanilla critic anchors | replay: paths from detour episodes | replay general anchors | balanced BC draws |
|---|---:|---:|---:|---:|---:|---:|
| d05 | 0.050 | 0.070 | 0.050 | 0.235 | 0.069 | 0.051 |
| d20 | 0.200 | 0.269 | 0.200 | 0.353 | 0.268 | 0.201 |

## BC walkers (single seed): route choice from the start (300 natural draws) and placed completion (Cnew rows)

| ratio | success | failure | timeout | detour | shortcut | placed north: BC reach (generation) | placed east: BC reach (generation) |
|---|---:|---:|---:|---:|---:|---|---|
| d05 | 0.257 | 0.743 | 0.000 | 0.000 | 0.967 | 0.800 (0.990) | 0.840 (0.990) |
| d20 | 0.260 | 0.730 | 0.010 | 0.007 | 0.950 | 0.920 (0.990) | 0.910 (0.990) |

## Actors (300 natural draws, mean policy; pure-BC init, frozen critic, bc 0.05, 30k)

| ratio | critic | seed | success | failure | timeout | detour | discounted |
|---|---|---|---:|---:|---:|---:|---:|
| d05 | vanilla | 0 | 0.263 | 0.673 | 0.063 | 0.067 | 0.022 |
| d05 | vanilla | 1 | 0.263 | 0.737 | 0.000 | 0.007 | 0.027 |
| d05 | vanilla | 2 | 0.260 | 0.733 | 0.007 | 0.013 | 0.025 |
| d05 | branch | 0 | 0.240 | 0.523 | 0.237 | 0.143 | 0.020 |
| d05 | branch | 1 | 0.240 | 0.670 | 0.090 | 0.020 | 0.024 |
| d05 | branch | 2 | 0.250 | 0.643 | 0.107 | 0.070 | 0.024 |
| d20 | vanilla | 0 | 0.543 | 0.260 | 0.197 | 0.507 | 0.018 |
| d20 | vanilla | 1 | 0.723 | 0.097 | 0.180 | 0.737 | 0.020 |
| d20 | vanilla | 2 | 0.647 | 0.253 | 0.100 | 0.630 | 0.021 |
| d20 | branch | 0 | 0.243 | 0.517 | 0.240 | 0.160 | 0.021 |
| d20 | branch | 1 | 0.290 | 0.537 | 0.173 | 0.163 | 0.020 |
| d20 | branch | 2 | 0.347 | 0.490 | 0.163 | 0.300 | 0.020 |

| ratio | critic | mean success (seed s.e.) | mean detour (seed s.e.) |
|---|---|---|---|
| d05 | vanilla | 0.262 (0.001) | 0.029 (0.019) |
| d05 | branch | 0.243 (0.003) | 0.078 (0.036) |
| d20 | vanilla | 0.638 (0.052) | 0.624 (0.066) |
| d20 | branch | 0.293 (0.030) | 0.208 (0.046) |

## Pre-registered comparisons

- d05 branch - vanilla, success: -0.019 (pooled seed s.e. 0.004; > 2 s.e.; same-direction seeds 3/3)
- d05 branch - vanilla, detour: +0.049 (pooled seed s.e. 0.041; not > 2 s.e.; same-direction seeds 3/3)
- d20 branch - vanilla, success: -0.344 (pooled seed s.e. 0.060; > 2 s.e.; same-direction seeds 3/3)
- d20 branch - vanilla, detour: -0.417 (pooled seed s.e. 0.081; > 2 s.e.; same-direction seeds 3/3)
- vanilla d20 - d05, success: +0.376 (pooled seed s.e. 0.052; > 2 s.e.; same-direction seeds 3/3)
- vanilla d20 - d05, detour: +0.596 (pooled seed s.e. 0.069; > 2 s.e.; same-direction seeds 3/3)
- branch d20 - d05, success: +0.050 (pooled seed s.e. 0.030; not > 2 s.e.; same-direction seeds 3/3)
- branch d20 - d05, detour: +0.130 (pooled seed s.e. 0.058; > 2 s.e.; same-direction seeds 3/3)
