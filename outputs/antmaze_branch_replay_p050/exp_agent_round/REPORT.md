# Agent-update round 1: policy-continuation futures (fixed d20 vanilla actor) vs the BC-continuation control

Sealed 2026-09-18 14:27:26.  Oracle transitions (diagnostic upper bound).  Decision rules in `manifest.json`.

## Round-1 replay vs the control replay

Identical anchors / query torques / draw seeds: **True** (19378 paths).  Continuation: round1 = fixed agent, control = d20 BC walker.

| stratum / candidate | n | round1 reach / death / P / around | control reach / death / P / around |
|---|---:|---|---|
| general|recorded | 5189 | 0.65 / 0.25 / 0.414 / 0.23 | 0.66 / 0.27 / 0.409 / 0.22 |
| general|sample0 | 5189 | 0.64 / 0.27 / 0.406 / 0.21 | 0.63 / 0.29 / 0.405 / 0.20 |
| north_leg|recorded | 600 | 0.90 / 0.00 / 0.378 / 0.97 | 0.89 / 0.00 / 0.361 / 0.98 |
| north_leg|sample0 | 600 | 0.90 / 0.00 / 0.376 / 0.98 | 0.87 / 0.00 / 0.355 / 0.98 |
| shortcut_early|recorded | 600 | 0.24 / 0.75 / 0.124 / 0.00 | 0.24 / 0.75 / 0.131 / 0.00 |
| shortcut_early|sample0 | 600 | 0.23 / 0.75 / 0.133 / 0.00 | 0.25 / 0.75 / 0.137 / 0.00 |
| start_early|recorded | 800 | 0.36 / 0.61 / 0.164 / 0.15 | 0.35 / 0.60 / 0.167 / 0.16 |
| start_early|sample0 | 800 | 0.33 / 0.63 / 0.156 / 0.13 | 0.34 / 0.62 / 0.169 / 0.14 |
| start_late|recorded | 400 | 0.24 / 0.74 / 0.122 / 0.00 | 0.26 / 0.74 / 0.122 / 0.00 |
| start_late|sample0 | 400 | 0.25 / 0.74 / 0.133 / 0.00 | 0.25 / 0.74 / 0.127 / 0.00 |
| turn|recorded | 600 | 0.71 / 0.02 / 0.277 / 0.76 | 0.71 / 0.02 / 0.272 / 0.77 |
| turn|sample0 | 600 | 0.69 / 0.03 / 0.287 / 0.74 | 0.70 / 0.02 / 0.264 / 0.77 |

## Critic check on Cdev -- labels under the FIXED AGENT continuation (who guides the current agent better)

| critic set | seed | decided pairs | agreement (s.e.) | pick gain (s.e.) | start_early agreement / gain | turn | north_leg |
|---|---|---:|---|---|---|---|---|
| round1 | 0 | 403 | 0.506 (0.033) | -0.006 (0.012) | 0.44 / -0.032 | 0.55 / +0.002 | 0.52 / +0.012 |
| round1 | 1 | 403 | 0.541 (0.031) | +0.025 (0.012) | 0.56 / +0.012 | 0.54 / +0.017 | 0.52 / +0.046 |
| round1 | 2 | 403 | 0.576 (0.033) | +0.008 (0.012) | 0.57 / -0.005 | 0.60 / +0.019 | 0.56 / +0.010 |
| control | 0 | 403 | 0.509 (0.035) | -0.019 (0.013) | 0.47 / -0.002 | 0.56 / -0.009 | 0.49 / -0.046 |
| control | 1 | 403 | 0.519 (0.037) | -0.014 (0.014) | 0.55 / +0.000 | 0.55 / +0.007 | 0.46 / -0.050 |
| control | 2 | 403 | 0.491 (0.034) | -0.012 (0.013) | 0.44 / +0.007 | 0.58 / -0.002 | 0.46 / -0.040 |
| vanilla | 0 | 403 | 0.529 (0.038) | +0.019 (0.013) | 0.57 / +0.020 | 0.52 / +0.017 | 0.50 / +0.020 |
| vanilla | 1 | 403 | 0.543 (0.039) | +0.021 (0.013) | 0.57 / +0.008 | 0.48 / +0.015 | 0.58 / +0.039 |
| vanilla | 2 | 403 | 0.514 (0.041) | -0.000 (0.014) | 0.54 / -0.015 | 0.48 / +0.020 | 0.51 / -0.006 |
| ext_ag | 0 | 403 | 0.474 (0.035) | -0.008 (0.014) | 0.50 / +0.001 | 0.43 / -0.017 | 0.49 / -0.009 |
| ext_ag | 1 | 403 | 0.516 (0.031) | +0.007 (0.013) | 0.54 / +0.011 | 0.51 / +0.003 | 0.50 / +0.007 |
| ext_ag | 2 | 403 | 0.494 (0.034) | -0.007 (0.014) | 0.60 / +0.026 | 0.35 / -0.049 | 0.53 / +0.001 |
| ext_bc | 0 | 403 | 0.499 (0.033) | -0.004 (0.013) | 0.53 / -0.021 | 0.58 / +0.041 | 0.40 / -0.032 |
| ext_bc | 1 | 403 | 0.514 (0.042) | +0.007 (0.013) | 0.53 / +0.023 | 0.48 / +0.009 | 0.53 / -0.010 |
| ext_bc | 2 | 403 | 0.459 (0.038) | -0.009 (0.014) | 0.45 / +0.004 | 0.47 / -0.027 | 0.46 / -0.003 |
| random | - | 403 | 0.506 | -0.003 | | | |

Cross-fitted selector gain (signal in the outcomes themselves): +0.085

Paired episode bootstrap (403 decided pairs, 53 episodes): pick gain seed-mean round1 +0.009 (0.007), control -0.015 (0.011), vanilla +0.013 (0.012), ext_ag -0.003 (0.011), ext_bc -0.002 (0.011); differences: round1 - control +0.024 +- 0.014 (z +1.8; per seed +0.013 / +0.039 / +0.020); control - round1 -0.024 +- 0.014 (z -1.8; per seed -0.013 / -0.039 / -0.020); vanilla - round1 +0.004 +- 0.013 (z +0.3; per seed +0.025 / -0.004 / -0.008); vanilla - control +0.028 +- 0.017 (z +1.7; per seed +0.038 / +0.035 / +0.012); ext_ag - round1 -0.012 +- 0.010 (z -1.2; per seed -0.003 / -0.018 / -0.015); ext_ag - control +0.012 +- 0.014 (z +0.9; per seed +0.011 / +0.021 / +0.004); ext_bc - round1 -0.011 +- 0.011 (z -1.0; per seed +0.002 / -0.018 / -0.017); ext_bc - control +0.013 +- 0.012 (z +1.1; per seed +0.015 / +0.021 / +0.003)

## Critic check on Cdev -- labels under the BC continuation (the control critic's own target)

| critic set | seed | decided pairs | agreement (s.e.) | pick gain (s.e.) | start_early agreement / gain | turn | north_leg |
|---|---|---:|---|---|---|---|---|
| round1 | 0 | 398 | 0.490 (0.033) | -0.005 (0.012) | 0.45 / -0.004 | 0.49 / -0.014 | 0.53 / +0.004 |
| round1 | 1 | 398 | 0.513 (0.034) | -0.007 (0.011) | 0.55 / +0.006 | 0.46 / -0.014 | 0.53 / -0.012 |
| round1 | 2 | 398 | 0.568 (0.032) | -0.011 (0.012) | 0.63 / +0.012 | 0.51 / -0.053 | 0.58 / +0.007 |
| control | 0 | 398 | 0.457 (0.033) | -0.010 (0.012) | 0.44 / -0.007 | 0.37 / -0.032 | 0.56 / +0.010 |
| control | 1 | 398 | 0.528 (0.032) | -0.015 (0.011) | 0.48 / -0.002 | 0.56 / -0.014 | 0.53 / -0.029 |
| control | 2 | 398 | 0.485 (0.034) | -0.008 (0.012) | 0.45 / -0.005 | 0.53 / -0.011 | 0.47 / -0.008 |
| vanilla | 0 | 398 | 0.490 (0.032) | -0.020 (0.013) | 0.48 / -0.007 | 0.50 / -0.020 | 0.49 / -0.032 |
| vanilla | 1 | 398 | 0.525 (0.032) | +0.001 (0.012) | 0.52 / +0.022 | 0.51 / -0.009 | 0.55 / -0.011 |
| vanilla | 2 | 398 | 0.450 (0.032) | -0.021 (0.013) | 0.41 / -0.016 | 0.48 / -0.025 | 0.45 / -0.021 |
| ext_ag | 0 | 398 | 0.533 (0.037) | +0.005 (0.011) | 0.60 / +0.012 | 0.51 / +0.020 | 0.50 / -0.017 |
| ext_ag | 1 | 398 | 0.475 (0.034) | -0.008 (0.012) | 0.45 / +0.011 | 0.51 / -0.013 | 0.45 / -0.024 |
| ext_ag | 2 | 398 | 0.510 (0.030) | +0.003 (0.010) | 0.54 / +0.004 | 0.54 / +0.001 | 0.45 / +0.002 |
| ext_bc | 0 | 398 | 0.487 (0.035) | +0.003 (0.012) | 0.51 / -0.010 | 0.53 / +0.024 | 0.42 / -0.003 |
| ext_bc | 1 | 398 | 0.500 (0.034) | -0.016 (0.012) | 0.61 / +0.014 | 0.43 / -0.033 | 0.47 / -0.028 |
| ext_bc | 2 | 398 | 0.558 (0.031) | -0.001 (0.012) | 0.66 / +0.011 | 0.53 / +0.008 | 0.50 / -0.022 |
| random | - | 398 | 0.490 | +0.016 | | | |

Cross-fitted selector gain (signal in the outcomes themselves): +0.092

Paired episode bootstrap (398 decided pairs, 51 episodes): pick gain seed-mean round1 -0.008 (0.007), control -0.011 (0.008), vanilla -0.013 (0.008), ext_ag -0.000 (0.009), ext_bc -0.005 (0.008); differences: round1 - control +0.003 +- 0.009 (z +0.4; per seed +0.005 / +0.008 / -0.003); control - round1 -0.003 +- 0.009 (z -0.4; per seed -0.005 / -0.008 / +0.003); vanilla - round1 -0.006 +- 0.012 (z -0.5; per seed -0.015 / +0.007 / -0.009); vanilla - control -0.002 +- 0.011 (z -0.2; per seed -0.010 / +0.016 / -0.012); ext_ag - round1 +0.007 +- 0.010 (z +0.7; per seed +0.010 / -0.002 / +0.014); ext_ag - control +0.011 +- 0.012 (z +0.9; per seed +0.015 / +0.007 / +0.011); ext_bc - round1 +0.003 +- 0.011 (z +0.3; per seed +0.008 / -0.009 / +0.010); ext_bc - control +0.006 +- 0.007 (z +0.9; per seed +0.013 / -0.001 / +0.007)

## Agents on the same fresh development draw (300 episodes, seed 2909, mean policy)

| policy | seed | success | failure | timeout | detour | no-hazard success (detours) | hazard success (detours / n) | zone-1 / zone-2 deaths | mean steps |
|---|---|---:|---:|---:|---:|---|---|---|---:|
| start (fixed agent) | 0 | 0.507 | 0.257 | 0.237 | 0.470 | 0.727 (36) | 0.430 (105) | 50 / 27 | 391 |
| round1 (old queries, agent continuation) | 0 | 0.440 | 0.323 | 0.237 | 0.397 | 0.714 (35) | 0.345 (84) | 60 / 37 | 388 |
| round1 (old queries, agent continuation) | 1 | 0.393 | 0.480 | 0.127 | 0.280 | 0.883 (20) | 0.224 (64) | 94 / 50 | 288 |
| round1 (old queries, agent continuation) | 2 | 0.320 | 0.530 | 0.150 | 0.173 | 0.831 (10) | 0.143 (42) | 101 / 58 | 268 |
| control (old queries, BC continuation) | 0 | 0.293 | 0.630 | 0.077 | 0.087 | 0.935 (5) | 0.072 (21) | 120 / 69 | 201 |
| control (old queries, BC continuation) | 1 | 0.280 | 0.530 | 0.190 | 0.133 | 0.805 (12) | 0.099 (28) | 99 / 60 | 289 |
| control (old queries, BC continuation) | 2 | 0.327 | 0.520 | 0.153 | 0.173 | 0.818 (11) | 0.157 (41) | 92 / 64 | 275 |
| reference (recorded-data critics) | 0 | 0.590 | 0.227 | 0.183 | 0.537 | 0.844 (37) | 0.502 (124) | 40 / 28 | 373 |
| reference (recorded-data critics) | 1 | 0.663 | 0.153 | 0.183 | 0.667 | 0.792 (48) | 0.619 (152) | 33 / 13 | 401 |
| reference (recorded-data critics) | 2 | 0.730 | 0.120 | 0.150 | 0.770 | 0.857 (57) | 0.686 (174) | 24 / 12 | 401 |
| ext_ag (extended queries, agent continuation) | 0 | 0.287 | 0.670 | 0.043 | 0.073 | 0.935 (6) | 0.063 (16) | 128 / 73 | 183 |
| ext_ag (extended queries, agent continuation) | 1 | 0.353 | 0.530 | 0.117 | 0.220 | 0.844 (18) | 0.184 (48) | 100 / 59 | 259 |
| ext_ag (extended queries, agent continuation) | 2 | 0.427 | 0.450 | 0.123 | 0.313 | 0.883 (24) | 0.269 (70) | 87 / 48 | 290 |
| ext_bc (extended queries, BC continuation) | 0 | 0.493 | 0.267 | 0.240 | 0.500 | 0.766 (39) | 0.399 (111) | 51 / 29 | 397 |
| ext_bc (extended queries, BC continuation) | 1 | 0.410 | 0.443 | 0.147 | 0.303 | 0.883 (21) | 0.247 (70) | 83 / 50 | 297 |
| ext_bc (extended queries, BC continuation) | 2 | 0.337 | 0.563 | 0.100 | 0.147 | 0.909 (11) | 0.139 (33) | 106 / 63 | 240 |

| policy | mean success (seed s.e.) | mean detour (seed s.e.) | mean timeout | mean no-hazard success |
|---|---|---|---|---|
| start (fixed agent) | 0.507 (0.000) | 0.470 (0.000) | 0.237 | 0.727 |
| round1 (old queries, agent continuation) | 0.384 (0.035) | 0.283 (0.065) | 0.171 | 0.810 |
| control (old queries, BC continuation) | 0.300 (0.014) | 0.131 (0.025) | 0.140 | 0.853 |
| reference (recorded-data critics) | 0.661 (0.040) | 0.658 (0.067) | 0.172 | 0.831 |
| ext_ag (extended queries, agent continuation) | 0.356 (0.040) | 0.202 (0.070) | 0.094 | 0.887 |
| ext_bc (extended queries, BC continuation) | 0.413 (0.045) | 0.317 (0.102) | 0.162 | 0.853 |

## Pre-registered comparisons (success primary; detour explanatory)

- round1 (old queries, agent continuation) minus start (fixed agent), success: -0.122 (seed s.e. of round1 (old queries, agent continuation) 0.035; > 2 s.e.; same-direction seeds 3/3)
- round1 (old queries, agent continuation) minus start (fixed agent), detour: -0.187 (seed s.e. of round1 (old queries, agent continuation) 0.065; > 2 s.e.; same-direction seeds 3/3)
- control (old queries, BC continuation) minus start (fixed agent), success: -0.207 (seed s.e. of control (old queries, BC continuation) 0.014; > 2 s.e.; same-direction seeds 3/3)
- control (old queries, BC continuation) minus start (fixed agent), detour: -0.339 (seed s.e. of control (old queries, BC continuation) 0.025; > 2 s.e.; same-direction seeds 3/3)
- reference (recorded-data critics) minus start (fixed agent), success: +0.154 (seed s.e. of reference (recorded-data critics) 0.040; > 2 s.e.; same-direction seeds 3/3)
- reference (recorded-data critics) minus start (fixed agent), detour: +0.188 (seed s.e. of reference (recorded-data critics) 0.067; > 2 s.e.; same-direction seeds 3/3)
- ext_ag (extended queries, agent continuation) minus start (fixed agent), success: -0.151 (seed s.e. of ext_ag (extended queries, agent continuation) 0.040; > 2 s.e.; same-direction seeds 3/3)
- ext_ag (extended queries, agent continuation) minus start (fixed agent), detour: -0.268 (seed s.e. of ext_ag (extended queries, agent continuation) 0.070; > 2 s.e.; same-direction seeds 3/3)
- ext_bc (extended queries, BC continuation) minus start (fixed agent), success: -0.093 (seed s.e. of ext_bc (extended queries, BC continuation) 0.045; > 2 s.e.; same-direction seeds 3/3)
- ext_bc (extended queries, BC continuation) minus start (fixed agent), detour: -0.153 (seed s.e. of ext_bc (extended queries, BC continuation) 0.102; not > 2 s.e.; same-direction seeds 2/3)
- round1 (old queries, agent continuation) minus control (old queries, BC continuation), success: +0.084 (pooled seed s.e. 0.038; > 2 s.e.; same-direction seeds 2/3)
- round1 (old queries, agent continuation) minus control (old queries, BC continuation), detour: +0.152 (pooled seed s.e. 0.069; > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus control (old queries, BC continuation), success: +0.056 (pooled seed s.e. 0.043; not > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus control (old queries, BC continuation), detour: +0.071 (pooled seed s.e. 0.074; not > 2 s.e.; same-direction seeds 2/3)
- ext_bc (extended queries, BC continuation) minus control (old queries, BC continuation), success: +0.113 (pooled seed s.e. 0.047; > 2 s.e.; same-direction seeds 3/3)
- ext_bc (extended queries, BC continuation) minus control (old queries, BC continuation), detour: +0.186 (pooled seed s.e. 0.105; not > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus round1 (old queries, agent continuation), success: -0.029 (pooled seed s.e. 0.053; not > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus round1 (old queries, agent continuation), detour: -0.081 (pooled seed s.e. 0.095; not > 2 s.e.; same-direction seeds 2/3)
- ext_bc (extended queries, BC continuation) minus control (old queries, BC continuation), success: +0.113 (pooled seed s.e. 0.047; > 2 s.e.; same-direction seeds 3/3)
- ext_bc (extended queries, BC continuation) minus control (old queries, BC continuation), detour: +0.186 (pooled seed s.e. 0.105; not > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus ext_bc (extended queries, BC continuation), success: -0.058 (pooled seed s.e. 0.061; not > 2 s.e.; same-direction seeds 2/3)
- ext_ag (extended queries, agent continuation) minus ext_bc (extended queries, BC continuation), detour: -0.114 (pooled seed s.e. 0.124; not > 2 s.e.; same-direction seeds 2/3)

Walking kept (timeout <= 0.25 and no-hazard success >= 0.70), per seed: start (fixed agent): yes; round1 (old queries, agent continuation): yes / yes / yes; control (old queries, BC continuation): yes / yes / yes; reference (recorded-data critics): yes / yes / yes; ext_ag (extended queries, agent continuation): yes / yes / yes; ext_bc (extended queries, BC continuation): yes / yes / yes
