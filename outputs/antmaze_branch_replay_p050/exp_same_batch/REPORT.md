# Identical batches, frozen critics only differ: actors from the fixed agent on the 2909 draw

Start (fixed agent): success 0.507, detour 0.470, timeout 0.237, no-hazard 0.727, hazard 0.430.  Batch audit: `batch_audit.json`.  Rule: mean over 3 seeds, > 2 x pooled seed s.e. and 3 / 3 seeds.

## Batch source S1_dataset_row0 (critic_stub_d20.npz)

| critic | seed | success | failure | timeout | detour | no-hazard success | hazard success | zone-1 / zone-2 deaths |
|---|---|---:|---:|---:|---:|---:|---:|---|
| vanilla | 0 | 0.590 | 0.227 | 0.183 | 0.537 | 0.844 | 0.502 | 40 / 28 |
| vanilla | 1 | 0.663 | 0.153 | 0.183 | 0.667 | 0.792 | 0.619 | 33 / 13 |
| vanilla | 2 | 0.730 | 0.120 | 0.150 | 0.770 | 0.857 | 0.686 | 24 / 12 |
| control | 0 | 0.383 | 0.507 | 0.110 | 0.233 | 0.857 | 0.220 | 93 / 59 |
| control | 1 | 0.277 | 0.613 | 0.110 | 0.087 | 0.883 | 0.067 | 110 / 74 |
| control | 2 | 0.430 | 0.433 | 0.137 | 0.310 | 0.870 | 0.278 | 81 / 49 |
| round1 | 0 | 0.280 | 0.627 | 0.093 | 0.070 | 0.883 | 0.072 | 120 / 68 |
| round1 | 1 | 0.490 | 0.267 | 0.243 | 0.403 | 0.740 | 0.404 | 53 / 27 |
| round1 | 2 | 0.290 | 0.563 | 0.147 | 0.130 | 0.740 | 0.135 | 107 / 62 |
| ext_ag | 0 | 0.283 | 0.630 | 0.087 | 0.070 | 0.896 | 0.072 | 123 / 66 |
| ext_ag | 1 | 0.397 | 0.467 | 0.137 | 0.260 | 0.857 | 0.238 | 84 / 56 |
| ext_ag | 2 | 0.270 | 0.633 | 0.097 | 0.083 | 0.883 | 0.058 | 116 / 74 |
| ext_bc | 0 | 0.590 | 0.220 | 0.190 | 0.600 | 0.844 | 0.502 | 44 / 22 |
| ext_bc | 1 | 0.607 | 0.190 | 0.203 | 0.630 | 0.779 | 0.547 | 34 / 23 |
| ext_bc | 2 | 0.307 | 0.570 | 0.123 | 0.127 | 0.870 | 0.112 | 105 / 66 |

| critic | mean success (seed s.e.) | vs start | mean detour (seed s.e.) |
|---|---|---|---|
| vanilla | 0.661 (0.040) | +0.154 (> 2 s.e.; 3/3) | 0.658 (0.067) |
| control | 0.363 (0.045) | -0.143 (> 2 s.e.; 3/3) | 0.210 (0.066) |
| round1 | 0.353 (0.068) | -0.153 (> 2 s.e.; 3/3) | 0.201 (0.103) |
| ext_ag | 0.317 (0.040) | -0.190 (> 2 s.e.; 3/3) | 0.138 (0.061) |
| ext_bc | 0.501 (0.097) | -0.006 (not > 2 s.e.; 1/3) | 0.452 (0.163) |

Pairwise (success):
- vanilla - control: +0.298 (pooled seed s.e. 0.061; > 2 s.e.; 3/3)
- vanilla - round1: +0.308 (pooled seed s.e. 0.079; > 2 s.e.; 3/3)
- vanilla - ext_ag: +0.344 (pooled seed s.e. 0.057; > 2 s.e.; 3/3)
- vanilla - ext_bc: +0.160 (pooled seed s.e. 0.105; not > 2 s.e.; 2/3)
- control - round1: +0.010 (pooled seed s.e. 0.082; not > 2 s.e.; 2/3)
- control - ext_ag: +0.047 (pooled seed s.e. 0.061; not > 2 s.e.; 2/3)
- control - ext_bc: -0.138 (pooled seed s.e. 0.107; not > 2 s.e.; 2/3)
- round1 - ext_ag: +0.037 (pooled seed s.e. 0.079; not > 2 s.e.; 2/3)
- round1 - ext_bc: -0.148 (pooled seed s.e. 0.119; not > 2 s.e.; 3/3)
- ext_ag - ext_bc: -0.184 (pooled seed s.e. 0.105; not > 2 s.e.; 3/3)

## Batch source S2_control_replay (replay_policy_d20.npz)

| critic | seed | success | failure | timeout | detour | no-hazard success | hazard success | zone-1 / zone-2 deaths |
|---|---|---:|---:|---:|---:|---:|---:|---|
| vanilla | 0 | 0.177 | 0.053 | 0.770 | 0.007 | 0.169 | 0.179 | 7 / 9 |
| vanilla | 1 | 0.253 | 0.563 | 0.183 | 0.040 | 0.857 | 0.045 | 104 / 65 |
| vanilla | 2 | 0.307 | 0.520 | 0.173 | 0.000 | 0.818 | 0.130 | 100 / 56 |
| control | 0 | 0.293 | 0.630 | 0.077 | 0.087 | 0.935 | 0.072 | 120 / 69 |
| control | 1 | 0.280 | 0.530 | 0.190 | 0.133 | 0.805 | 0.099 | 99 / 60 |
| control | 2 | 0.327 | 0.520 | 0.153 | 0.173 | 0.818 | 0.157 | 92 / 64 |
| round1 | 0 | 0.310 | 0.627 | 0.063 | 0.100 | 0.935 | 0.094 | 122 / 66 |
| round1 | 1 | 0.333 | 0.450 | 0.217 | 0.223 | 0.805 | 0.170 | 86 / 49 |
| round1 | 2 | 0.297 | 0.593 | 0.110 | 0.147 | 0.857 | 0.103 | 112 / 66 |
| ext_ag | 0 | 0.300 | 0.647 | 0.053 | 0.090 | 0.935 | 0.081 | 126 / 68 |
| ext_ag | 1 | 0.310 | 0.587 | 0.103 | 0.147 | 0.896 | 0.108 | 107 / 69 |
| ext_ag | 2 | 0.483 | 0.407 | 0.110 | 0.343 | 0.948 | 0.323 | 74 / 48 |
| ext_bc | 0 | 0.450 | 0.260 | 0.290 | 0.497 | 0.701 | 0.363 | 50 / 28 |
| ext_bc | 1 | 0.390 | 0.327 | 0.283 | 0.337 | 0.675 | 0.291 | 58 / 40 |
| ext_bc | 2 | 0.373 | 0.490 | 0.137 | 0.203 | 0.857 | 0.206 | 93 / 54 |

| critic | mean success (seed s.e.) | vs start | mean detour (seed s.e.) |
|---|---|---|---|
| vanilla | 0.246 (0.038) | -0.261 (> 2 s.e.; 3/3) | 0.016 (0.012) |
| control | 0.300 (0.014) | -0.207 (> 2 s.e.; 3/3) | 0.131 (0.025) |
| round1 | 0.313 (0.011) | -0.193 (> 2 s.e.; 3/3) | 0.157 (0.036) |
| ext_ag | 0.364 (0.060) | -0.142 (> 2 s.e.; 3/3) | 0.193 (0.077) |
| ext_bc | 0.404 (0.023) | -0.102 (> 2 s.e.; 3/3) | 0.346 (0.085) |

Pairwise (success):
- vanilla - control: -0.054 (pooled seed s.e. 0.040; not > 2 s.e.; 3/3)
- vanilla - round1: -0.068 (pooled seed s.e. 0.039; not > 2 s.e.; 2/3)
- vanilla - ext_ag: -0.119 (pooled seed s.e. 0.070; not > 2 s.e.; 3/3)
- vanilla - ext_bc: -0.159 (pooled seed s.e. 0.044; > 2 s.e.; 3/3)
- control - round1: -0.013 (pooled seed s.e. 0.018; not > 2 s.e.; 2/3)
- control - ext_ag: -0.064 (pooled seed s.e. 0.061; not > 2 s.e.; 3/3)
- control - ext_bc: -0.104 (pooled seed s.e. 0.027; > 2 s.e.; 3/3)
- round1 - ext_ag: -0.051 (pooled seed s.e. 0.060; not > 2 s.e.; 1/3)
- round1 - ext_bc: -0.091 (pooled seed s.e. 0.026; > 2 s.e.; 3/3)
- ext_ag - ext_bc: -0.040 (pooled seed s.e. 0.064; not > 2 s.e.; 2/3)

## Same critic, different batch source (S1 - S2, success)

- vanilla: +0.416 (pooled seed s.e. 0.055; > 2 s.e.; 3/3)
- control: +0.063 (pooled seed s.e. 0.047; not > 2 s.e.; 2/3)
- round1: +0.040 (pooled seed s.e. 0.069; not > 2 s.e.; 1/3)
- ext_ag: -0.048 (pooled seed s.e. 0.072; not > 2 s.e.; 2/3)
- ext_bc: +0.097 (pooled seed s.e. 0.100; not > 2 s.e.; 2/3)
