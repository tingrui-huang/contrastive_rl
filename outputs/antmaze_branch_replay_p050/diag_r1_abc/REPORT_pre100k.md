# Round-1 critic diagnostic: memorisation (A) vs new torques (B) vs new-episode states (C)

Sealed manifest: 2026-09-18 03:27:56; reference commit 69e6572.  Frozen: continuation seed_0 (820cb7e229ed), critics seed_0 (339a76c02b4a), seed_1 (780d4afb868c), seed_2 (508d46fd6e35).  16 fresh paired draws per anchor (seeds ABC_SEED + 100 i + r), one query torque then the frozen policy mode closed-loop; P_goal = gamma 0.999, radius 0.5.  Primary readout = region-integrated exp(f) over the training NCE goal marginal (radius 0.5, per head and deployed min); secondary = exact recorded-goal logit.  Pair classes on the 16-draw log ratio: tie (= 0), weak (<= 0.3), decided (> 0.3); bootstrap-decided = decided and >= 90% of draw-resamples keep the sign.  Agreement = among decided pairs (ties are never counted as wrong); s.e. = episode bootstrap.  Oracle-stage diagnostic; nothing was trained.

## Anchors

| layer | stratum | anchors | episodes | t min / median / max | goal xy mean (std) | candidate torque spread | dist to mode | log pi(a) | frac saturated |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| A | start_early | 32 | 32 | 0 / 3 / 5 | [24.734, 0.815] ([0.296, 0.335]) | 0.60 | 0.45 | 15.7 | 0.29 |
| A | start_late | 32 | 32 | 7 / 14 / 21 | [24.745, 0.895] ([0.328, 0.291]) | 0.21 | 0.14 | 16.7 | 0.10 |
| A | shortcut_early | 32 | 32 | 23 / 34 / 46 | [24.743, 0.721] ([0.313, 0.295]) | 0.17 | 0.13 | 18.0 | 0.11 |
| A | turn | 32 | 32 | 6 / 14 / 51 | [24.826, 0.74] ([0.271, 0.351]) | 1.42 | 1.06 | 14.4 | 0.44 |
| A | north_leg | 32 | 32 | 24 / 42 / 126 | [24.843, 0.708] ([0.302, 0.332]) | 1.05 | 0.72 | 4.6 | 0.14 |
| A | pooled | 160 | 131 | 0 / 17 / 126 | [24.778, 0.776] ([0.306, 0.329]) | 0.69 | 0.50 | 13.9 | 0.22 |
| B | start_early | 32 | 32 | 0 / 3 / 5 | [24.734, 0.815] ([0.296, 0.335]) | 0.69 | 0.52 | 15.4 | 0.29 |
| B | start_late | 32 | 32 | 7 / 14 / 21 | [24.745, 0.895] ([0.328, 0.291]) | 0.22 | 0.15 | 16.7 | 0.11 |
| B | shortcut_early | 32 | 32 | 23 / 34 / 46 | [24.743, 0.721] ([0.313, 0.295]) | 0.19 | 0.13 | 18.1 | 0.12 |
| B | turn | 32 | 32 | 6 / 14 / 51 | [24.826, 0.74] ([0.271, 0.351]) | 1.39 | 1.08 | 13.4 | 0.42 |
| B | north_leg | 32 | 32 | 24 / 42 / 126 | [24.843, 0.708] ([0.302, 0.332]) | 1.02 | 0.74 | 4.8 | 0.15 |
| B | pooled | 160 | 131 | 0 / 17 / 126 | [24.778, 0.776] ([0.306, 0.329]) | 0.70 | 0.52 | 13.7 | 0.22 |
| C | start_early | 32 | 32 | 0 / 2 / 5 | [24.774, 0.77] ([0.315, 0.274]) | 0.48 | 0.31 | 15.9 | 0.23 |
| C | start_late | 32 | 32 | 6 / 14 / 19 | [24.801, 0.818] ([0.338, 0.375]) | 0.15 | 0.09 | 18.6 | 0.09 |
| C | shortcut_early | 32 | 32 | 19 / 33 / 47 | [24.795, 0.752] ([0.327, 0.302]) | 0.12 | 0.08 | 20.1 | 0.11 |
| C | turn | 32 | 5 | 6 / 12 / 28 | [24.711, 0.678] ([0.408, 0.226]) | 1.50 | 1.01 | 6.7 | 0.33 |
| C | north_leg | 32 | 5 | 19 / 38 / 59 | [24.711, 0.678] ([0.408, 0.226]) | 0.76 | 0.49 | 6.2 | 0.13 |
| C | pooled | 160 | 67 | 0 / 16 / 59 | [24.759, 0.739] ([0.364, 0.291]) | 0.60 | 0.40 | 13.5 | 0.18 |

Standardised-state distance: A anchor to its nearest other A anchor, median 2.49; C anchor to its nearest A anchor, median 2.28.  C differs from A in episode, and may differ in pose, goal and time distributions (see the table); B differs from A only in the torque.

## Fresh outcomes by layer and candidate

| key | paths | reach | death | P_goal | went around |
|---|---:|---:|---:|---:|---:|
| A|recorded | 2560 | 0.44 | 0.44 | 0.166 | 0.36 |
| A|sample0 | 2560 | 0.45 | 0.44 | 0.169 | 0.36 |
| A|sample1 | 2560 | 0.45 | 0.44 | 0.177 | 0.37 |
| B|new0 | 2560 | 0.41 | 0.44 | 0.157 | 0.35 |
| B|new1 | 2560 | 0.45 | 0.44 | 0.174 | 0.36 |
| B|new2 | 2560 | 0.44 | 0.43 | 0.168 | 0.37 |
| C|mode | 2560 | 0.43 | 0.45 | 0.161 | 0.35 |
| C|recorded | 2560 | 0.47 | 0.45 | 0.163 | 0.38 |
| C|sample0 | 2560 | 0.43 | 0.44 | 0.174 | 0.36 |
| C|sample1 | 2560 | 0.43 | 0.45 | 0.172 | 0.36 |
| C|sample2 | 2560 | 0.43 | 0.46 | 0.166 | 0.35 |

## Labels: original records against fresh outcomes at identical keys

| layer | stratum | keys | Spearman(orig mean P, fresh mean P) | mean abs diff | pairs decided by orig | by fresh | both | labels same / opposite | orig-decided but fresh tie/weak |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| A | start_early | 96 | 0.12 | 0.153 | 26 | 37 | 26 | 25 / 1 | 0 |
| A | start_late | 96 | -0.22 | 0.159 | 11 | 32 | 10 | 10 / 0 | 1 |
| A | shortcut_early | 96 | 0.15 | 0.160 | 15 | 24 | 15 | 15 / 0 | 0 |
| A | turn | 96 | 0.82 | 0.067 | 54 | 65 | 49 | 44 / 5 | 5 |
| A | north_leg | 96 | 0.83 | 0.068 | 67 | 72 | 57 | 51 / 6 | 10 |
| A | pooled | 480 | 0.48 | 0.121 | 173 | 230 | 157 | 145 / 12 | 16 |
| C | start_early | 160 | 0.28 | 0.092 | 73 | 104 | 71 | 70 / 1 | 2 |
| C | start_late | 160 | 0.20 | 0.087 | 71 | 97 | 66 | 66 / 0 | 5 |
| C | shortcut_early | 160 | 0.36 | 0.101 | 65 | 88 | 65 | 65 / 0 | 0 |
| C | turn | 160 | 0.82 | 0.058 | 229 | 237 | 206 | 191 / 15 | 23 |
| C | north_leg | 160 | 0.90 | 0.050 | 241 | 232 | 198 | 186 / 12 | 43 |
| C | pooled | 800 | 0.58 | 0.077 | 679 | 758 | 606 | 578 / 28 | 73 |

## Layer A: the critic against the original labels and against fresh labels at the same keys (region min)

| critic | stratum | vs original: same / opposite (agreement) | vs fresh: same / opposite (agreement) | where the critic matched the original label, fresh same / opposite |
|---|---|---|---|---|
| critics_pre100k_only/seed_0 | start_early | 15 / 11 (0.58) | 20 / 17 (0.54) | 15 / 0 |
| critics_pre100k_only/seed_0 | start_late | 5 / 6 (0.45) | 14 / 18 (0.44) | 4 / 0 |
| critics_pre100k_only/seed_0 | shortcut_early | 5 / 10 (0.33) | 11 / 13 (0.46) | 5 / 0 |
| critics_pre100k_only/seed_0 | turn | 20 / 34 (0.37) | 23 / 42 (0.35) | 13 / 4 |
| critics_pre100k_only/seed_0 | north_leg | 31 / 36 (0.46) | 34 / 38 (0.47) | 21 / 3 |
| critics_pre100k_only/seed_0 | pooled | 76 / 97 (0.44) | 102 / 128 (0.44) | 58 / 7 |
| critics_pre100k_only/seed_1 | start_early | 13 / 13 (0.50) | 19 / 18 (0.51) | 13 / 0 |
| critics_pre100k_only/seed_1 | start_late | 3 / 8 (0.27) | 10 / 22 (0.31) | 2 / 0 |
| critics_pre100k_only/seed_1 | shortcut_early | 3 / 12 (0.20) | 6 / 18 (0.25) | 3 / 0 |
| critics_pre100k_only/seed_1 | turn | 23 / 31 (0.43) | 31 / 34 (0.48) | 17 / 3 |
| critics_pre100k_only/seed_1 | north_leg | 29 / 38 (0.43) | 31 / 41 (0.43) | 22 / 4 |
| critics_pre100k_only/seed_1 | pooled | 71 / 102 (0.41) | 97 / 133 (0.42) | 57 / 7 |
| critics_pre100k_only/seed_2 | start_early | 12 / 14 (0.46) | 16 / 21 (0.43) | 12 / 0 |
| critics_pre100k_only/seed_2 | start_late | 4 / 7 (0.36) | 15 / 17 (0.47) | 3 / 0 |
| critics_pre100k_only/seed_2 | shortcut_early | 7 / 8 (0.47) | 10 / 14 (0.42) | 7 / 0 |
| critics_pre100k_only/seed_2 | turn | 21 / 33 (0.39) | 27 / 38 (0.42) | 16 / 3 |
| critics_pre100k_only/seed_2 | north_leg | 28 / 39 (0.42) | 37 / 35 (0.51) | 24 / 2 |
| critics_pre100k_only/seed_2 | pooled | 72 / 101 (0.42) | 105 / 125 (0.46) | 62 / 5 |

## Layer A: familiar state, familiar torque, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.30** (0.094) 11 / 26 | 0.37 (0.104) | -0.022 (0.011) | - (-) | 0.123 / 0.145 / - | +0.047 (0.013) |
| random | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.47** (0.104) 15 / 17 | 0.50 (0.121) | +0.002 (0.009) | - (-) | 0.147 / 0.145 / - | +0.025 (0.006) |
| random | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.50** (0.129) 12 / 12 | 0.61 (0.153) | +0.006 (0.007) | - (-) | 0.137 / 0.131 / - | +0.018 (0.005) |
| random | turn | 32 (32) | 13 / 18 / 65 (59) | **0.63** (0.085) 41 / 24 | 0.66 (0.087) | +0.027 (0.021) | - (-) | 0.181 / 0.153 / - | +0.090 (0.017) |
| random | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.57** (0.072) 41 / 31 | 0.58 (0.071) | +0.013 (0.025) | - (-) | 0.293 / 0.280 / - | +0.122 (0.012) |
| random | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.52** (0.044) 120 / 110 | 0.56 (0.045) | +0.005 (0.007) | - (-) | 0.176 / 0.171 / - | +0.061 (0.006) |
| critics_pre100k_only/seed_0 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.54** (0.101) 20 / 17 | 0.57 (0.110) | -0.004 (0.013) | - (-) | 0.141 / 0.145 / - | +0.047 (0.013) |
| critics_pre100k_only/seed_0 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.44** (0.094) 14 / 18 | 0.50 (0.111) | +0.007 (0.008) | - (-) | 0.153 / 0.145 / - | +0.025 (0.006) |
| critics_pre100k_only/seed_0 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.46** (0.131) 11 / 13 | 0.50 (0.140) | -0.004 (0.009) | - (-) | 0.127 / 0.131 / - | +0.018 (0.005) |
| critics_pre100k_only/seed_0 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.35** (0.071) 23 / 42 | 0.37 (0.072) | -0.033 (0.018) | - (-) | 0.120 / 0.153 / - | +0.090 (0.017) |
| critics_pre100k_only/seed_0 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.47** (0.070) 34 / 38 | 0.46 (0.070) | -0.024 (0.025) | - (-) | 0.256 / 0.280 / - | +0.122 (0.012) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.44** (0.042) 102 / 128 | 0.46 (0.043) | -0.012 (0.007) | - (-) | 0.159 / 0.171 / - | +0.061 (0.006) |
| critics_pre100k_only/seed_1 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.51** (0.105) 19 / 18 | 0.43 (0.119) | -0.003 (0.017) | - (-) | 0.141 / 0.145 / - | +0.047 (0.013) |
| critics_pre100k_only/seed_1 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.31** (0.087) 10 / 22 | 0.27 (0.090) | +0.003 (0.009) | - (-) | 0.148 / 0.145 / - | +0.025 (0.006) |
| critics_pre100k_only/seed_1 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.25** (0.071) 6 / 18 | 0.28 (0.084) | -0.003 (0.007) | - (-) | 0.128 / 0.131 / - | +0.018 (0.005) |
| critics_pre100k_only/seed_1 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.48** (0.074) 31 / 34 | 0.49 (0.076) | -0.021 (0.017) | - (-) | 0.132 / 0.153 / - | +0.090 (0.017) |
| critics_pre100k_only/seed_1 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.43** (0.062) 31 / 41 | 0.44 (0.063) | -0.022 (0.024) | - (-) | 0.259 / 0.280 / - | +0.122 (0.012) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.42** (0.040) 97 / 133 | 0.42 (0.040) | -0.009 (0.007) | - (-) | 0.162 / 0.171 / - | +0.061 (0.006) |
| critics_pre100k_only/seed_2 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.43** (0.086) 16 / 21 | 0.40 (0.092) | +0.002 (0.018) | - (-) | 0.146 / 0.145 / - | +0.047 (0.013) |
| critics_pre100k_only/seed_2 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.47** (0.111) 15 / 17 | 0.46 (0.130) | -0.001 (0.010) | - (-) | 0.144 / 0.145 / - | +0.025 (0.006) |
| critics_pre100k_only/seed_2 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.42** (0.122) 10 / 14 | 0.56 (0.125) | -0.000 (0.008) | - (-) | 0.131 / 0.131 / - | +0.018 (0.005) |
| critics_pre100k_only/seed_2 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.42** (0.069) 27 / 38 | 0.42 (0.067) | -0.012 (0.018) | - (-) | 0.141 / 0.153 / - | +0.090 (0.017) |
| critics_pre100k_only/seed_2 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.51** (0.065) 37 / 35 | 0.51 (0.065) | +0.004 (0.026) | - (-) | 0.285 / 0.280 / - | +0.122 (0.012) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.46** (0.038) 105 / 125 | 0.47 (0.038) | -0.001 (0.008) | - (-) | 0.169 / 0.171 / - | +0.061 (0.006) |

## Layer A, the two policy samples only

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.44** (0.060) 32 / 41 | 0.50 (0.071) | -0.000 (0.006) | - (-) | 0.173 / 0.173 / - | +0.038 (0.005) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.51** (0.056) 37 / 36 | 0.55 (0.064) | -0.001 (0.006) | - (-) | 0.173 / 0.173 / - | +0.038 (0.005) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.38** (0.053) 28 / 45 | 0.38 (0.061) | -0.008 (0.006) | - (-) | 0.165 / 0.173 / - | +0.038 (0.005) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.47** (0.059) 34 / 39 | 0.48 (0.066) | -0.001 (0.006) | - (-) | 0.172 / 0.173 / - | +0.038 (0.005) |

## Layer B: familiar state, three new torques, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.66** (0.076) 31 / 16 | 0.66 (0.084) | +0.028 (0.012) | - (-) | 0.174 / 0.146 / - | +0.050 (0.013) |
| random | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.39** (0.114) 9 / 14 | 0.43 (0.119) | -0.000 (0.007) | - (-) | 0.154 / 0.155 / - | +0.015 (0.004) |
| random | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.38** (0.095) 11 / 18 | 0.36 (0.095) | -0.007 (0.009) | - (-) | 0.121 / 0.128 / - | +0.017 (0.008) |
| random | turn | 32 (32) | 23 / 16 / 57 (50) | **0.56** (0.061) 32 / 25 | 0.60 (0.067) | +0.001 (0.019) | - (-) | 0.161 / 0.160 / - | +0.092 (0.018) |
| random | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.53** (0.066) 35 / 31 | 0.52 (0.070) | -0.003 (0.018) | - (-) | 0.242 / 0.245 / - | +0.102 (0.013) |
| random | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.53** (0.038) 118 / 104 | 0.54 (0.039) | +0.004 (0.006) | - (-) | 0.170 / 0.167 / - | +0.055 (0.006) |
| critics_pre100k_only/seed_0 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.40** (0.090) 19 / 28 | 0.32 (0.092) | -0.015 (0.014) | - (-) | 0.131 / 0.146 / - | +0.050 (0.013) |
| critics_pre100k_only/seed_0 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.35** (0.107) 8 / 15 | 0.38 (0.113) | -0.009 (0.008) | - (-) | 0.146 / 0.155 / - | +0.015 (0.004) |
| critics_pre100k_only/seed_0 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.59** (0.085) 17 / 12 | 0.60 (0.084) | +0.009 (0.008) | - (-) | 0.137 / 0.128 / - | +0.017 (0.008) |
| critics_pre100k_only/seed_0 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.51** (0.054) 29 / 28 | 0.50 (0.060) | +0.010 (0.019) | - (-) | 0.171 / 0.160 / - | +0.092 (0.018) |
| critics_pre100k_only/seed_0 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.42** (0.060) 28 / 38 | 0.43 (0.065) | -0.017 (0.017) | - (-) | 0.228 / 0.245 / - | +0.102 (0.013) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.45** (0.034) 101 / 121 | 0.44 (0.035) | -0.004 (0.006) | - (-) | 0.162 / 0.167 / - | +0.055 (0.006) |
| critics_pre100k_only/seed_1 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.40** (0.089) 19 / 28 | 0.37 (0.105) | -0.008 (0.014) | - (-) | 0.138 / 0.146 / - | +0.050 (0.013) |
| critics_pre100k_only/seed_1 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.26** (0.070) 6 / 17 | 0.24 (0.071) | -0.012 (0.008) | - (-) | 0.143 / 0.155 / - | +0.015 (0.004) |
| critics_pre100k_only/seed_1 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.31** (0.096) 9 / 20 | 0.36 (0.107) | -0.016 (0.010) | - (-) | 0.112 / 0.128 / - | +0.017 (0.008) |
| critics_pre100k_only/seed_1 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.51** (0.067) 29 / 28 | 0.48 (0.070) | -0.004 (0.020) | - (-) | 0.156 / 0.160 / - | +0.092 (0.018) |
| critics_pre100k_only/seed_1 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.52** (0.065) 34 / 32 | 0.52 (0.067) | +0.001 (0.017) | - (-) | 0.245 / 0.245 / - | +0.102 (0.013) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.44** (0.035) 97 / 125 | 0.43 (0.039) | -0.008 (0.006) | - (-) | 0.159 / 0.167 / - | +0.055 (0.006) |
| critics_pre100k_only/seed_2 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.49** (0.073) 23 / 24 | 0.53 (0.083) | +0.006 (0.010) | - (-) | 0.152 / 0.146 / - | +0.050 (0.013) |
| critics_pre100k_only/seed_2 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.48** (0.122) 11 / 12 | 0.43 (0.128) | +0.000 (0.007) | - (-) | 0.155 / 0.155 / - | +0.015 (0.004) |
| critics_pre100k_only/seed_2 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.38** (0.120) 11 / 18 | 0.36 (0.129) | -0.014 (0.010) | - (-) | 0.113 / 0.128 / - | +0.017 (0.008) |
| critics_pre100k_only/seed_2 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.51** (0.080) 29 / 28 | 0.48 (0.083) | +0.019 (0.022) | - (-) | 0.179 / 0.160 / - | +0.092 (0.018) |
| critics_pre100k_only/seed_2 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.41** (0.057) 27 / 39 | 0.43 (0.061) | -0.025 (0.017) | - (-) | 0.220 / 0.245 / - | +0.102 (0.013) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.45** (0.036) 101 / 121 | 0.45 (0.040) | -0.003 (0.006) | - (-) | 0.164 / 0.167 / - | +0.055 (0.006) |

## A x B cross pairs (shared anchors and draws)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.50** (0.023) 332 / 328 | 0.53 (0.026) | - (-) | - (-) | - / - / - | - (-) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.44** (0.028) 293 / 367 | 0.45 (0.029) | - (-) | - (-) | - / - / - | - (-) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.47** (0.025) 307 / 353 | 0.46 (0.025) | - (-) | - (-) | - / - / - | - (-) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.47** (0.026) 310 / 350 | 0.48 (0.028) | - (-) | - (-) | - / - / - | - (-) |

## Layer C: held-out-episode states, existing five candidates, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.58** (0.066) 60 / 44 | 0.56 (0.072) | +0.012 (0.005) | +0.022 (0.018) | 0.153 / 0.141 / 0.131 | +0.024 (0.009) |
| random | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.56** (0.039) 54 / 43 | 0.56 (0.042) | +0.007 (0.007) | +0.003 (0.011) | 0.134 / 0.128 / 0.132 | +0.023 (0.006) |
| random | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.51** (0.071) 45 / 43 | 0.55 (0.076) | -0.005 (0.009) | -0.007 (0.010) | 0.149 / 0.154 / 0.156 | +0.022 (0.010) |
| random | turn | 32 (5) | 18 / 65 / 237 (217) | **0.51** (0.060) 120 / 117 | 0.52 (0.063) | -0.017 (0.024) | +0.002 (0.036) | 0.158 / 0.176 / 0.157 | +0.161 (0.019) |
| random | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.45** (0.038) 104 / 128 | 0.45 (0.040) | -0.009 (0.026) | -0.000 (0.040) | 0.230 / 0.238 / 0.230 | +0.161 (0.014) |
| random | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.51** (0.029) 383 / 375 | 0.51 (0.031) | -0.002 (0.008) | +0.004 (0.012) | 0.165 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_pre100k_only/seed_0 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.53** (0.078) 55 / 49 | 0.60 (0.081) | +0.007 (0.011) | +0.017 (0.023) | 0.148 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_pre100k_only/seed_0 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.44** (0.074) 43 / 54 | 0.44 (0.088) | +0.006 (0.008) | +0.002 (0.014) | 0.134 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_pre100k_only/seed_0 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.50** (0.072) 44 / 44 | 0.49 (0.079) | -0.009 (0.014) | -0.011 (0.019) | 0.145 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_pre100k_only/seed_0 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.54** (0.051) 127 / 110 | 0.57 (0.048) | +0.015 (0.022) | +0.034 (0.029) | 0.190 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_pre100k_only/seed_0 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.49** (0.027) 113 / 119 | 0.48 (0.029) | -0.009 (0.025) | -0.001 (0.046) | 0.229 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.50** (0.025) 382 / 376 | 0.52 (0.024) | +0.002 (0.008) | +0.008 (0.013) | 0.169 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_pre100k_only/seed_1 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.40** (0.059) 42 / 62 | 0.45 (0.065) | -0.007 (0.012) | +0.003 (0.023) | 0.134 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_pre100k_only/seed_1 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.54** (0.092) 52 / 45 | 0.51 (0.103) | -0.002 (0.009) | -0.006 (0.016) | 0.126 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_pre100k_only/seed_1 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.52** (0.068) 46 / 42 | 0.51 (0.077) | +0.003 (0.011) | +0.000 (0.016) | 0.156 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_pre100k_only/seed_1 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.51** (0.042) 120 / 117 | 0.53 (0.039) | +0.005 (0.028) | +0.024 (0.031) | 0.181 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_pre100k_only/seed_1 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.50** (0.036) 115 / 117 | 0.49 (0.038) | +0.009 (0.024) | +0.017 (0.043) | 0.247 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.49** (0.023) 375 / 383 | 0.50 (0.026) | +0.002 (0.008) | +0.008 (0.012) | 0.169 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_pre100k_only/seed_2 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.59** (0.076) 61 / 43 | 0.58 (0.082) | -0.004 (0.012) | +0.006 (0.021) | 0.137 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_pre100k_only/seed_2 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.45** (0.065) 44 / 53 | 0.39 (0.073) | -0.008 (0.010) | -0.012 (0.017) | 0.120 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_pre100k_only/seed_2 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.45** (0.064) 40 / 48 | 0.44 (0.068) | -0.013 (0.011) | -0.015 (0.013) | 0.141 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_pre100k_only/seed_2 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.52** (0.019) 124 / 113 | 0.54 (0.018) | +0.038 (0.024) | +0.057 (0.032) | 0.214 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_pre100k_only/seed_2 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.46** (0.033) 107 / 125 | 0.46 (0.032) | -0.020 (0.023) | -0.012 (0.044) | 0.218 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.50** (0.021) 376 / 382 | 0.49 (0.020) | -0.001 (0.008) | +0.005 (0.012) | 0.166 / 0.167 / 0.161 | +0.078 (0.008) |

## Layer C, matched candidate types (recorded, sample0, sample1)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.54** (0.089) 13 / 11 | 0.56 (0.065) | +0.008 (0.005) | - (-) | 0.150 / 0.142 / - | +0.014 (0.008) |
| random | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.61** (0.118) 19 / 12 | 0.67 (0.127) | +0.013 (0.008) | - (-) | 0.141 / 0.127 / - | +0.023 (0.007) |
| random | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.47** (0.099) 14 / 16 | 0.54 (0.099) | +0.009 (0.010) | - (-) | 0.164 / 0.155 / - | +0.020 (0.008) |
| random | turn | 32 (5) | 6 / 13 / 77 (73) | **0.53** (0.103) 41 / 36 | 0.53 (0.103) | +0.013 (0.023) | - (-) | 0.202 / 0.189 / - | +0.134 (0.013) |
| random | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.45** (0.045) 29 / 36 | 0.44 (0.045) | -0.028 (0.017) | - (-) | 0.206 / 0.234 / - | +0.107 (0.014) |
| random | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.51** (0.039) 116 / 111 | 0.52 (0.044) | +0.003 (0.006) | - (-) | 0.173 / 0.170 / - | +0.060 (0.006) |
| critics_pre100k_only/seed_0 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.46** (0.130) 11 / 13 | 0.50 (0.159) | -0.003 (0.010) | - (-) | 0.139 / 0.142 / - | +0.014 (0.008) |
| critics_pre100k_only/seed_0 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.35** (0.100) 11 / 20 | 0.38 (0.127) | -0.003 (0.007) | - (-) | 0.124 / 0.127 / - | +0.023 (0.007) |
| critics_pre100k_only/seed_0 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.53** (0.121) 16 / 14 | 0.46 (0.129) | -0.007 (0.012) | - (-) | 0.147 / 0.155 / - | +0.020 (0.008) |
| critics_pre100k_only/seed_0 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.43** (0.079) 33 / 44 | 0.45 (0.075) | -0.011 (0.020) | - (-) | 0.178 / 0.189 / - | +0.134 (0.013) |
| critics_pre100k_only/seed_0 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.51** (0.038) 33 / 32 | 0.49 (0.040) | -0.002 (0.018) | - (-) | 0.232 / 0.234 / - | +0.107 (0.014) |
| critics_pre100k_only/seed_0 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.46** (0.041) 104 / 123 | 0.46 (0.042) | -0.005 (0.006) | - (-) | 0.164 / 0.170 / - | +0.060 (0.006) |
| critics_pre100k_only/seed_1 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.38** (0.111) 9 / 15 | 0.39 (0.123) | -0.004 (0.010) | - (-) | 0.139 / 0.142 / - | +0.014 (0.008) |
| critics_pre100k_only/seed_1 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.58** (0.102) 18 / 13 | 0.57 (0.119) | +0.003 (0.006) | - (-) | 0.130 / 0.127 / - | +0.023 (0.007) |
| critics_pre100k_only/seed_1 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.53** (0.086) 16 / 14 | 0.54 (0.101) | +0.013 (0.010) | - (-) | 0.167 / 0.155 / - | +0.020 (0.008) |
| critics_pre100k_only/seed_1 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.45** (0.046) 35 / 42 | 0.48 (0.039) | -0.001 (0.025) | - (-) | 0.188 / 0.189 / - | +0.134 (0.013) |
| critics_pre100k_only/seed_1 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.55** (0.034) 36 / 29 | 0.56 (0.044) | +0.008 (0.022) | - (-) | 0.242 / 0.234 / - | +0.107 (0.014) |
| critics_pre100k_only/seed_1 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.50** (0.028) 114 / 113 | 0.51 (0.031) | +0.004 (0.007) | - (-) | 0.173 / 0.170 / - | +0.060 (0.006) |
| critics_pre100k_only/seed_2 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.50** (0.114) 12 / 12 | 0.39 (0.129) | -0.008 (0.011) | - (-) | 0.134 / 0.142 / - | +0.014 (0.008) |
| critics_pre100k_only/seed_2 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.42** (0.105) 13 / 18 | 0.38 (0.122) | -0.016 (0.009) | - (-) | 0.112 / 0.127 / - | +0.023 (0.007) |
| critics_pre100k_only/seed_2 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.47** (0.099) 14 / 16 | 0.42 (0.104) | -0.007 (0.013) | - (-) | 0.147 / 0.155 / - | +0.020 (0.008) |
| critics_pre100k_only/seed_2 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.47** (0.045) 36 / 41 | 0.49 (0.043) | -0.005 (0.022) | - (-) | 0.184 / 0.189 / - | +0.134 (0.013) |
| critics_pre100k_only/seed_2 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.48** (0.083) 31 / 34 | 0.46 (0.079) | -0.006 (0.019) | - (-) | 0.229 / 0.234 / - | +0.107 (0.014) |
| critics_pre100k_only/seed_2 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.47** (0.032) 106 / 121 | 0.45 (0.034) | -0.008 (0.007) | - (-) | 0.161 / 0.170 / - | +0.060 (0.006) |

## All readouts, pooled (agreement among decided pairs; pick gain)

| scorer | A | A samples-only | B | A x B | C | C matched |
|---|---|---|---|---|---|---|
| random | 0.52 (230) / +0.005 | 0.44 (73) / -0.000 | 0.53 (222) / +0.004 | 0.50 (660) / - | 0.51 (758) / -0.002 | 0.51 (227) / +0.003 |
| critics_pre100k_only/seed_0 region min | 0.44 (230) / -0.012 | 0.51 (73) / -0.001 | 0.45 (222) / -0.004 | 0.44 (660) / - | 0.50 (758) / +0.002 | 0.46 (227) / -0.005 |
| critics_pre100k_only/seed_0 exact min | 0.45 (230) / -0.009 | 0.55 (73) / +0.000 | 0.51 (222) / +0.002 | 0.48 (660) / - | 0.49 (758) / +0.002 | 0.47 (227) / -0.005 |
| critics_pre100k_only/seed_0 region h0 | 0.43 (230) / -0.001 | 0.41 (73) / -0.003 | 0.49 (222) / +0.000 | 0.46 (660) / - | 0.49 (758) / -0.009 | 0.45 (227) / -0.005 |
| critics_pre100k_only/seed_0 exact h0 | 0.48 (230) / -0.005 | 0.49 (73) / +0.005 | 0.55 (222) / +0.005 | 0.50 (660) / - | 0.49 (758) / -0.004 | 0.46 (227) / -0.003 |
| critics_pre100k_only/seed_0 region h1 | 0.42 (230) / -0.013 | 0.48 (73) / -0.005 | 0.45 (222) / -0.003 | 0.43 (660) / - | 0.50 (758) / +0.004 | 0.46 (227) / -0.002 |
| critics_pre100k_only/seed_0 exact h1 | 0.44 (230) / -0.013 | 0.52 (73) / -0.004 | 0.50 (222) / +0.001 | 0.47 (660) / - | 0.51 (758) / +0.004 | 0.47 (227) / -0.003 |
| critics_pre100k_only/seed_1 region min | 0.42 (230) / -0.009 | 0.38 (73) / -0.008 | 0.44 (222) / -0.008 | 0.47 (660) / - | 0.49 (758) / +0.002 | 0.50 (227) / +0.004 |
| critics_pre100k_only/seed_1 exact min | 0.47 (230) / -0.005 | 0.42 (73) / -0.008 | 0.50 (222) / -0.005 | 0.50 (660) / - | 0.51 (758) / +0.007 | 0.49 (227) / +0.003 |
| critics_pre100k_only/seed_1 region h0 | 0.43 (230) / -0.010 | 0.51 (73) / +0.001 | 0.47 (222) / +0.005 | 0.45 (660) / - | 0.50 (758) / +0.005 | 0.47 (227) / +0.003 |
| critics_pre100k_only/seed_1 exact h0 | 0.45 (230) / -0.014 | 0.49 (73) / -0.005 | 0.52 (222) / +0.007 | 0.48 (660) / - | 0.50 (758) / +0.004 | 0.47 (227) / -0.002 |
| critics_pre100k_only/seed_1 region h1 | 0.42 (230) / -0.011 | 0.36 (73) / -0.012 | 0.45 (222) / -0.007 | 0.46 (660) / - | 0.49 (758) / -0.000 | 0.48 (227) / -0.005 |
| critics_pre100k_only/seed_1 exact h1 | 0.47 (230) / -0.006 | 0.37 (73) / -0.011 | 0.49 (222) / -0.009 | 0.49 (660) / - | 0.50 (758) / +0.004 | 0.48 (227) / -0.002 |
| critics_pre100k_only/seed_2 region min | 0.46 (230) / -0.001 | 0.47 (73) / -0.001 | 0.45 (222) / -0.003 | 0.47 (660) / - | 0.50 (758) / -0.001 | 0.47 (227) / -0.008 |
| critics_pre100k_only/seed_2 exact min | 0.45 (230) / -0.004 | 0.42 (73) / -0.005 | 0.49 (222) / -0.003 | 0.50 (660) / - | 0.50 (758) / -0.001 | 0.48 (227) / -0.012 |
| critics_pre100k_only/seed_2 region h0 | 0.49 (230) / +0.000 | 0.47 (73) / -0.001 | 0.44 (222) / -0.006 | 0.47 (660) / - | 0.50 (758) / +0.011 | 0.49 (227) / +0.004 |
| critics_pre100k_only/seed_2 exact h0 | 0.50 (230) / -0.001 | 0.49 (73) / -0.002 | 0.48 (222) / -0.005 | 0.49 (660) / - | 0.51 (758) / +0.010 | 0.52 (227) / +0.008 |
| critics_pre100k_only/seed_2 region h1 | 0.47 (230) / -0.005 | 0.51 (73) / +0.001 | 0.47 (222) / +0.004 | 0.48 (660) / - | 0.47 (758) / -0.012 | 0.45 (227) / -0.012 |
| critics_pre100k_only/seed_2 exact h1 | 0.53 (230) / -0.004 | 0.56 (73) / +0.003 | 0.51 (222) / +0.002 | 0.50 (660) / - | 0.48 (758) / -0.011 | 0.47 (227) / -0.011 |

## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)

- critics_pre100k_only/seed_0: A: fails (interval below 0.6) [0.44 +- 0.042, n 230, gain -0.012]; B: fails (interval below 0.6) [0.45 +- 0.034, n 222, gain -0.004]; C: fails (interval below 0.6) [0.50 +- 0.025, n 758, gain +0.002]
- critics_pre100k_only/seed_1: A: fails (interval below 0.6) [0.42 +- 0.040, n 230, gain -0.009]; B: fails (interval below 0.6) [0.44 +- 0.035, n 222, gain -0.008]; C: fails (interval below 0.6) [0.49 +- 0.023, n 758, gain +0.002]
- critics_pre100k_only/seed_2: A: fails (interval below 0.6) [0.46 +- 0.038, n 230, gain -0.001]; B: fails (interval below 0.6) [0.45 +- 0.036, n 222, gain -0.003]; C: fails (interval below 0.6) [0.50 +- 0.021, n 758, gain -0.001]
- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): +0.061; B: +0.055; C: +0.078 (a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).
