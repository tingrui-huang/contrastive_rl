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
| critics_narrow/seed_0 | start_early | 17 / 9 (0.65) | 22 / 15 (0.59) | 16 / 1 |
| critics_narrow/seed_0 | start_late | 7 / 4 (0.64) | 19 / 13 (0.59) | 7 / 0 |
| critics_narrow/seed_0 | shortcut_early | 9 / 6 (0.60) | 11 / 13 (0.46) | 9 / 0 |
| critics_narrow/seed_0 | turn | 44 / 10 (0.81) | 58 / 7 (0.89) | 40 / 1 |
| critics_narrow/seed_0 | north_leg | 52 / 15 (0.78) | 57 / 15 (0.79) | 43 / 0 |
| critics_narrow/seed_0 | pooled | 129 / 44 (0.75) | 167 / 63 (0.73) | 115 / 2 |
| critics_narrow/seed_1 | start_early | 16 / 10 (0.62) | 21 / 16 (0.57) | 15 / 1 |
| critics_narrow/seed_1 | start_late | 3 / 8 (0.27) | 17 / 15 (0.53) | 3 / 0 |
| critics_narrow/seed_1 | shortcut_early | 8 / 7 (0.53) | 10 / 14 (0.42) | 8 / 0 |
| critics_narrow/seed_1 | turn | 44 / 10 (0.81) | 55 / 10 (0.85) | 40 / 2 |
| critics_narrow/seed_1 | north_leg | 47 / 20 (0.70) | 54 / 18 (0.75) | 40 / 1 |
| critics_narrow/seed_1 | pooled | 118 / 55 (0.68) | 157 / 73 (0.68) | 106 / 4 |
| critics_narrow/seed_2 | start_early | 15 / 11 (0.58) | 24 / 13 (0.65) | 15 / 0 |
| critics_narrow/seed_2 | start_late | 7 / 4 (0.64) | 16 / 16 (0.50) | 7 / 0 |
| critics_narrow/seed_2 | shortcut_early | 9 / 6 (0.60) | 11 / 13 (0.46) | 9 / 0 |
| critics_narrow/seed_2 | turn | 43 / 11 (0.80) | 58 / 7 (0.89) | 39 / 0 |
| critics_narrow/seed_2 | north_leg | 56 / 11 (0.84) | 61 / 11 (0.85) | 46 / 1 |
| critics_narrow/seed_2 | pooled | 130 / 43 (0.75) | 170 / 60 (0.74) | 116 / 1 |

## Layer A: familiar state, familiar torque, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.30** (0.094) 11 / 26 | 0.37 (0.104) | -0.022 (0.011) | - (-) | 0.123 / 0.145 / - | +0.047 (0.013) |
| random | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.47** (0.104) 15 / 17 | 0.50 (0.121) | +0.002 (0.009) | - (-) | 0.147 / 0.145 / - | +0.025 (0.006) |
| random | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.50** (0.129) 12 / 12 | 0.61 (0.153) | +0.006 (0.007) | - (-) | 0.137 / 0.131 / - | +0.018 (0.005) |
| random | turn | 32 (32) | 13 / 18 / 65 (59) | **0.63** (0.085) 41 / 24 | 0.66 (0.087) | +0.027 (0.021) | - (-) | 0.181 / 0.153 / - | +0.090 (0.017) |
| random | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.57** (0.072) 41 / 31 | 0.58 (0.071) | +0.013 (0.025) | - (-) | 0.293 / 0.280 / - | +0.122 (0.012) |
| random | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.52** (0.044) 120 / 110 | 0.56 (0.045) | +0.005 (0.007) | - (-) | 0.176 / 0.171 / - | +0.061 (0.006) |
| critics_narrow/seed_0 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.59** (0.101) 22 / 15 | 0.60 (0.108) | +0.036 (0.015) | - (-) | 0.180 / 0.145 / - | +0.047 (0.013) |
| critics_narrow/seed_0 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.59** (0.106) 19 / 13 | 0.65 (0.112) | +0.006 (0.009) | - (-) | 0.151 / 0.145 / - | +0.025 (0.006) |
| critics_narrow/seed_0 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.46** (0.135) 11 / 13 | 0.50 (0.148) | -0.004 (0.008) | - (-) | 0.127 / 0.131 / - | +0.018 (0.005) |
| critics_narrow/seed_0 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.89** (0.039) 58 / 7 | 0.92 (0.041) | +0.092 (0.017) | - (-) | 0.245 / 0.153 / - | +0.090 (0.017) |
| critics_narrow/seed_0 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.79** (0.050) 57 / 15 | 0.79 (0.050) | +0.062 (0.017) | - (-) | 0.342 / 0.280 / - | +0.122 (0.012) |
| critics_narrow/seed_0 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.73** (0.036) 167 / 63 | 0.75 (0.036) | +0.038 (0.007) | - (-) | 0.209 / 0.171 / - | +0.061 (0.006) |
| critics_narrow/seed_1 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.57** (0.096) 21 / 16 | 0.60 (0.109) | +0.036 (0.015) | - (-) | 0.180 / 0.145 / - | +0.047 (0.013) |
| critics_narrow/seed_1 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.53** (0.103) 17 / 15 | 0.50 (0.106) | +0.008 (0.009) | - (-) | 0.153 / 0.145 / - | +0.025 (0.006) |
| critics_narrow/seed_1 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.42** (0.124) 10 / 14 | 0.56 (0.139) | +0.000 (0.007) | - (-) | 0.132 / 0.131 / - | +0.018 (0.005) |
| critics_narrow/seed_1 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.85** (0.038) 55 / 10 | 0.85 (0.041) | +0.075 (0.017) | - (-) | 0.228 / 0.153 / - | +0.090 (0.017) |
| critics_narrow/seed_1 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.75** (0.054) 54 / 18 | 0.75 (0.054) | +0.056 (0.022) | - (-) | 0.337 / 0.280 / - | +0.122 (0.012) |
| critics_narrow/seed_1 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.68** (0.039) 157 / 73 | 0.71 (0.038) | +0.035 (0.007) | - (-) | 0.206 / 0.171 / - | +0.061 (0.006) |
| critics_narrow/seed_2 region min | start_early | 32 (32) | 4 / 55 / 37 (30) | **0.65** (0.091) 24 / 13 | 0.67 (0.099) | +0.036 (0.015) | - (-) | 0.181 / 0.145 / - | +0.047 (0.013) |
| critics_narrow/seed_2 region min | start_late | 32 (32) | 9 / 55 / 32 (26) | **0.50** (0.116) 16 / 16 | 0.54 (0.128) | -0.002 (0.010) | - (-) | 0.143 / 0.145 / - | +0.025 (0.006) |
| critics_narrow/seed_2 region min | shortcut_early | 32 (32) | 4 / 68 / 24 (18) | **0.46** (0.119) 11 / 13 | 0.56 (0.135) | -0.000 (0.007) | - (-) | 0.131 / 0.131 / - | +0.018 (0.005) |
| critics_narrow/seed_2 region min | turn | 32 (32) | 13 / 18 / 65 (59) | **0.89** (0.040) 58 / 7 | 0.90 (0.043) | +0.080 (0.017) | - (-) | 0.234 / 0.153 / - | +0.090 (0.017) |
| critics_narrow/seed_2 region min | north_leg | 32 (32) | 0 / 24 / 72 (71) | **0.85** (0.035) 61 / 11 | 0.85 (0.036) | +0.089 (0.016) | - (-) | 0.370 / 0.280 / - | +0.122 (0.012) |
| critics_narrow/seed_2 region min | pooled | 160 (131) | 30 / 220 / 230 (204) | **0.74** (0.034) 170 / 60 | 0.77 (0.035) | +0.041 (0.007) | - (-) | 0.212 / 0.171 / - | +0.061 (0.006) |

## Layer A, the two policy samples only

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.44** (0.060) 32 / 41 | 0.50 (0.071) | -0.000 (0.006) | - (-) | 0.173 / 0.173 / - | +0.038 (0.005) |
| critics_narrow/seed_0 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.67** (0.056) 49 / 24 | 0.70 (0.062) | +0.019 (0.005) | - (-) | 0.193 / 0.173 / - | +0.038 (0.005) |
| critics_narrow/seed_1 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.71** (0.052) 52 / 21 | 0.75 (0.055) | +0.023 (0.005) | - (-) | 0.196 / 0.173 / - | +0.038 (0.005) |
| critics_narrow/seed_2 region min | pooled | 160 (131) | 12 / 75 / 73 (60) | **0.70** (0.053) 51 / 22 | 0.73 (0.055) | +0.022 (0.005) | - (-) | 0.195 / 0.173 / - | +0.038 (0.005) |

## Layer B: familiar state, three new torques, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.66** (0.076) 31 / 16 | 0.66 (0.084) | +0.028 (0.012) | - (-) | 0.174 / 0.146 / - | +0.050 (0.013) |
| random | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.39** (0.114) 9 / 14 | 0.43 (0.119) | -0.000 (0.007) | - (-) | 0.154 / 0.155 / - | +0.015 (0.004) |
| random | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.38** (0.095) 11 / 18 | 0.36 (0.095) | -0.007 (0.009) | - (-) | 0.121 / 0.128 / - | +0.017 (0.008) |
| random | turn | 32 (32) | 23 / 16 / 57 (50) | **0.56** (0.061) 32 / 25 | 0.60 (0.067) | +0.001 (0.019) | - (-) | 0.161 / 0.160 / - | +0.092 (0.018) |
| random | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.53** (0.066) 35 / 31 | 0.52 (0.070) | -0.003 (0.018) | - (-) | 0.242 / 0.245 / - | +0.102 (0.013) |
| random | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.53** (0.038) 118 / 104 | 0.54 (0.039) | +0.004 (0.006) | - (-) | 0.170 / 0.167 / - | +0.055 (0.006) |
| critics_narrow/seed_0 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.62** (0.095) 29 / 18 | 0.61 (0.105) | +0.013 (0.017) | - (-) | 0.159 / 0.146 / - | +0.050 (0.013) |
| critics_narrow/seed_0 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.65** (0.109) 15 / 8 | 0.62 (0.111) | +0.004 (0.006) | - (-) | 0.158 / 0.155 / - | +0.015 (0.004) |
| critics_narrow/seed_0 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.72** (0.102) 21 / 8 | 0.72 (0.115) | +0.005 (0.008) | - (-) | 0.132 / 0.128 / - | +0.017 (0.008) |
| critics_narrow/seed_0 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.44** (0.052) 25 / 32 | 0.44 (0.056) | -0.029 (0.017) | - (-) | 0.131 / 0.160 / - | +0.092 (0.018) |
| critics_narrow/seed_0 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.55** (0.071) 36 / 30 | 0.51 (0.073) | +0.002 (0.018) | - (-) | 0.247 / 0.245 / - | +0.102 (0.013) |
| critics_narrow/seed_0 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.57** (0.034) 126 / 96 | 0.55 (0.037) | -0.001 (0.006) | - (-) | 0.166 / 0.167 / - | +0.055 (0.006) |
| critics_narrow/seed_1 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.64** (0.085) 30 / 17 | 0.66 (0.095) | +0.016 (0.015) | - (-) | 0.162 / 0.146 / - | +0.050 (0.013) |
| critics_narrow/seed_1 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.74** (0.114) 17 / 6 | 0.71 (0.123) | +0.004 (0.006) | - (-) | 0.159 / 0.155 / - | +0.015 (0.004) |
| critics_narrow/seed_1 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.69** (0.096) 20 / 9 | 0.68 (0.107) | +0.011 (0.006) | - (-) | 0.139 / 0.128 / - | +0.017 (0.008) |
| critics_narrow/seed_1 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.54** (0.070) 31 / 26 | 0.56 (0.074) | +0.017 (0.020) | - (-) | 0.177 / 0.160 / - | +0.092 (0.018) |
| critics_narrow/seed_1 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.61** (0.067) 40 / 26 | 0.59 (0.071) | +0.018 (0.019) | - (-) | 0.263 / 0.245 / - | +0.102 (0.013) |
| critics_narrow/seed_1 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.62** (0.035) 138 / 84 | 0.62 (0.040) | +0.013 (0.006) | - (-) | 0.180 / 0.167 / - | +0.055 (0.006) |
| critics_narrow/seed_2 region min | start_early | 32 (32) | 1 / 48 / 47 (38) | **0.64** (0.080) 30 / 17 | 0.61 (0.093) | +0.021 (0.015) | - (-) | 0.167 / 0.146 / - | +0.050 (0.013) |
| critics_narrow/seed_2 region min | start_late | 32 (32) | 8 / 65 / 23 (21) | **0.48** (0.131) 11 / 12 | 0.43 (0.135) | -0.011 (0.008) | - (-) | 0.143 / 0.155 / - | +0.015 (0.004) |
| critics_narrow/seed_2 region min | shortcut_early | 32 (32) | 11 / 56 / 29 (25) | **0.59** (0.115) 17 / 12 | 0.64 (0.126) | +0.009 (0.007) | - (-) | 0.136 / 0.128 / - | +0.017 (0.008) |
| critics_narrow/seed_2 region min | turn | 32 (32) | 23 / 16 / 57 (50) | **0.44** (0.065) 25 / 32 | 0.44 (0.070) | -0.017 (0.017) | - (-) | 0.143 / 0.160 / - | +0.092 (0.018) |
| critics_narrow/seed_2 region min | north_leg | 32 (32) | 1 / 29 / 66 (61) | **0.55** (0.072) 36 / 30 | 0.51 (0.074) | -0.008 (0.017) | - (-) | 0.237 / 0.245 / - | +0.102 (0.013) |
| critics_narrow/seed_2 region min | pooled | 160 (131) | 44 / 214 / 222 (195) | **0.54** (0.038) 119 / 103 | 0.52 (0.042) | -0.001 (0.006) | - (-) | 0.165 / 0.167 / - | +0.055 (0.006) |

## A x B cross pairs (shared anchors and draws)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.50** (0.023) 332 / 328 | 0.53 (0.026) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_0 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.58** (0.025) 382 / 278 | 0.61 (0.025) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_1 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.63** (0.023) 418 / 242 | 0.64 (0.026) | - (-) | - (-) | - / - / - | - (-) |
| critics_narrow/seed_2 region min | pooled | 160 (131) | 111 / 669 / 660 (580) | **0.59** (0.023) 391 / 269 | 0.61 (0.025) | - (-) | - (-) | - / - / - | - (-) |

## Layer C: held-out-episode states, existing five candidates, fresh outcomes

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.58** (0.066) 60 / 44 | 0.56 (0.072) | +0.012 (0.005) | +0.022 (0.018) | 0.153 / 0.141 / 0.131 | +0.024 (0.009) |
| random | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.56** (0.039) 54 / 43 | 0.56 (0.042) | +0.007 (0.007) | +0.003 (0.011) | 0.134 / 0.128 / 0.132 | +0.023 (0.006) |
| random | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.51** (0.071) 45 / 43 | 0.55 (0.076) | -0.005 (0.009) | -0.007 (0.010) | 0.149 / 0.154 / 0.156 | +0.022 (0.010) |
| random | turn | 32 (5) | 18 / 65 / 237 (217) | **0.51** (0.060) 120 / 117 | 0.52 (0.063) | -0.017 (0.024) | +0.002 (0.036) | 0.158 / 0.176 / 0.157 | +0.161 (0.019) |
| random | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.45** (0.038) 104 / 128 | 0.45 (0.040) | -0.009 (0.026) | -0.000 (0.040) | 0.230 / 0.238 / 0.230 | +0.161 (0.014) |
| random | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.51** (0.029) 383 / 375 | 0.51 (0.031) | -0.002 (0.008) | +0.004 (0.012) | 0.165 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_narrow/seed_0 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.45** (0.061) 47 / 57 | 0.40 (0.070) | -0.017 (0.010) | -0.007 (0.019) | 0.124 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_narrow/seed_0 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.49** (0.078) 48 / 49 | 0.38 (0.073) | -0.009 (0.010) | -0.013 (0.015) | 0.118 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_narrow/seed_0 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.49** (0.077) 43 / 45 | 0.46 (0.072) | +0.002 (0.012) | -0.000 (0.017) | 0.156 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_narrow/seed_0 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.45** (0.057) 106 / 131 | 0.44 (0.060) | -0.015 (0.022) | +0.004 (0.032) | 0.161 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_narrow/seed_0 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.48** (0.065) 112 / 120 | 0.49 (0.068) | -0.035 (0.027) | -0.026 (0.037) | 0.204 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_narrow/seed_0 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.47** (0.037) 356 / 402 | 0.45 (0.041) | -0.015 (0.008) | -0.009 (0.011) | 0.152 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_narrow/seed_1 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.52** (0.067) 54 / 50 | 0.49 (0.077) | +0.002 (0.009) | +0.011 (0.018) | 0.142 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_narrow/seed_1 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.52** (0.061) 50 / 47 | 0.44 (0.068) | -0.003 (0.008) | -0.007 (0.014) | 0.125 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_narrow/seed_1 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.49** (0.075) 43 / 45 | 0.47 (0.071) | +0.008 (0.011) | +0.006 (0.016) | 0.162 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_narrow/seed_1 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.44** (0.019) 104 / 133 | 0.44 (0.018) | -0.009 (0.024) | +0.010 (0.031) | 0.167 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_narrow/seed_1 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.50** (0.057) 116 / 116 | 0.50 (0.060) | +0.011 (0.028) | +0.019 (0.047) | 0.249 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_narrow/seed_1 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.48** (0.026) 367 / 391 | 0.47 (0.027) | +0.002 (0.008) | +0.008 (0.012) | 0.169 / 0.167 / 0.161 | +0.078 (0.008) |
| critics_narrow/seed_2 region min | start_early | 32 (32) | 21 / 195 / 104 (85) | **0.56** (0.054) 58 / 46 | 0.49 (0.052) | +0.002 (0.009) | +0.012 (0.017) | 0.143 / 0.141 / 0.131 | +0.024 (0.009) |
| critics_narrow/seed_2 region min | start_late | 32 (32) | 21 / 202 / 97 (71) | **0.46** (0.077) 45 / 52 | 0.42 (0.084) | -0.000 (0.008) | -0.004 (0.014) | 0.127 / 0.128 / 0.132 | +0.023 (0.006) |
| critics_narrow/seed_2 region min | shortcut_early | 32 (32) | 26 / 206 / 88 (78) | **0.57** (0.066) 50 / 38 | 0.58 (0.070) | +0.011 (0.012) | +0.009 (0.018) | 0.164 / 0.154 / 0.156 | +0.022 (0.010) |
| critics_narrow/seed_2 region min | turn | 32 (5) | 18 / 65 / 237 (217) | **0.47** (0.020) 112 / 125 | 0.47 (0.019) | -0.013 (0.026) | +0.007 (0.033) | 0.163 / 0.176 / 0.157 | +0.161 (0.019) |
| critics_narrow/seed_2 region min | north_leg | 32 (5) | 4 / 84 / 232 (222) | **0.52** (0.063) 120 / 112 | 0.52 (0.058) | -0.003 (0.028) | +0.006 (0.046) | 0.236 / 0.238 / 0.230 | +0.161 (0.014) |
| critics_narrow/seed_2 region min | pooled | 160 (67) | 90 / 752 / 758 (673) | **0.51** (0.021) 385 / 373 | 0.50 (0.022) | -0.001 (0.008) | +0.006 (0.013) | 0.167 / 0.167 / 0.161 | +0.078 (0.008) |

## Layer C, matched candidate types (recorded, sample0, sample1)

| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |
|---|---|---:|---|---|---|---:|---:|---|---:|
| random | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.54** (0.089) 13 / 11 | 0.56 (0.065) | +0.008 (0.005) | - (-) | 0.150 / 0.142 / - | +0.014 (0.008) |
| random | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.61** (0.118) 19 / 12 | 0.67 (0.127) | +0.013 (0.008) | - (-) | 0.141 / 0.127 / - | +0.023 (0.007) |
| random | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.47** (0.099) 14 / 16 | 0.54 (0.099) | +0.009 (0.010) | - (-) | 0.164 / 0.155 / - | +0.020 (0.008) |
| random | turn | 32 (5) | 6 / 13 / 77 (73) | **0.53** (0.103) 41 / 36 | 0.53 (0.103) | +0.013 (0.023) | - (-) | 0.202 / 0.189 / - | +0.134 (0.013) |
| random | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.45** (0.045) 29 / 36 | 0.44 (0.045) | -0.028 (0.017) | - (-) | 0.206 / 0.234 / - | +0.107 (0.014) |
| random | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.51** (0.039) 116 / 111 | 0.52 (0.044) | +0.003 (0.006) | - (-) | 0.173 / 0.170 / - | +0.060 (0.006) |
| critics_narrow/seed_0 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.46** (0.121) 11 / 13 | 0.39 (0.148) | -0.012 (0.010) | - (-) | 0.131 / 0.142 / - | +0.014 (0.008) |
| critics_narrow/seed_0 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.42** (0.104) 13 / 18 | 0.29 (0.106) | -0.014 (0.009) | - (-) | 0.113 / 0.127 / - | +0.023 (0.007) |
| critics_narrow/seed_0 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.50** (0.105) 15 / 15 | 0.50 (0.107) | -0.005 (0.013) | - (-) | 0.150 / 0.155 / - | +0.020 (0.008) |
| critics_narrow/seed_0 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.39** (0.091) 30 / 47 | 0.40 (0.092) | -0.048 (0.021) | - (-) | 0.141 / 0.189 / - | +0.134 (0.013) |
| critics_narrow/seed_0 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.60** (0.067) 39 / 26 | 0.59 (0.073) | +0.005 (0.018) | - (-) | 0.239 / 0.234 / - | +0.107 (0.014) |
| critics_narrow/seed_0 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.48** (0.054) 108 / 119 | 0.46 (0.056) | -0.015 (0.007) | - (-) | 0.155 / 0.170 / - | +0.060 (0.006) |
| critics_narrow/seed_1 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.62** (0.109) 15 / 9 | 0.61 (0.136) | +0.006 (0.007) | - (-) | 0.148 / 0.142 / - | +0.014 (0.008) |
| critics_narrow/seed_1 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.39** (0.081) 12 / 19 | 0.33 (0.099) | -0.014 (0.009) | - (-) | 0.113 / 0.127 / - | +0.023 (0.007) |
| critics_narrow/seed_1 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.47** (0.102) 14 / 16 | 0.46 (0.101) | -0.001 (0.012) | - (-) | 0.153 / 0.155 / - | +0.020 (0.008) |
| critics_narrow/seed_1 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.44** (0.067) 34 / 43 | 0.44 (0.061) | -0.019 (0.023) | - (-) | 0.171 / 0.189 / - | +0.134 (0.013) |
| critics_narrow/seed_1 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.57** (0.067) 37 / 28 | 0.56 (0.069) | +0.022 (0.020) | - (-) | 0.256 / 0.234 / - | +0.107 (0.014) |
| critics_narrow/seed_1 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.49** (0.039) 112 / 115 | 0.48 (0.044) | -0.001 (0.007) | - (-) | 0.168 / 0.170 / - | +0.060 (0.006) |
| critics_narrow/seed_2 region min | start_early | 32 (32) | 7 / 65 / 24 (18) | **0.50** (0.105) 12 / 12 | 0.33 (0.103) | -0.003 (0.007) | - (-) | 0.140 / 0.142 / - | +0.014 (0.008) |
| critics_narrow/seed_2 region min | start_late | 32 (32) | 7 / 58 / 31 (21) | **0.35** (0.092) 11 / 20 | 0.38 (0.121) | -0.013 (0.008) | - (-) | 0.115 / 0.127 / - | +0.023 (0.007) |
| critics_narrow/seed_2 region min | shortcut_early | 32 (32) | 9 / 57 / 30 (26) | **0.43** (0.089) 13 / 17 | 0.46 (0.101) | +0.005 (0.011) | - (-) | 0.160 / 0.155 / - | +0.020 (0.008) |
| critics_narrow/seed_2 region min | turn | 32 (5) | 6 / 13 / 77 (73) | **0.49** (0.055) 38 / 39 | 0.49 (0.059) | -0.004 (0.025) | - (-) | 0.185 / 0.189 / - | +0.134 (0.013) |
| critics_narrow/seed_2 region min | north_leg | 32 (5) | 3 / 28 / 65 (61) | **0.58** (0.087) 38 / 27 | 0.61 (0.076) | +0.028 (0.020) | - (-) | 0.262 / 0.234 / - | +0.107 (0.014) |
| critics_narrow/seed_2 region min | pooled | 160 (67) | 32 / 221 / 227 (199) | **0.49** (0.043) 112 / 115 | 0.50 (0.046) | +0.003 (0.007) | - (-) | 0.172 / 0.170 / - | +0.060 (0.006) |

## All readouts, pooled (agreement among decided pairs; pick gain)

| scorer | A | A samples-only | B | A x B | C | C matched |
|---|---|---|---|---|---|---|
| random | 0.52 (230) / +0.005 | 0.44 (73) / -0.000 | 0.53 (222) / +0.004 | 0.50 (660) / - | 0.51 (758) / -0.002 | 0.51 (227) / +0.003 |
| critics_narrow/seed_0 region min | 0.73 (230) / +0.038 | 0.67 (73) / +0.019 | 0.57 (222) / -0.001 | 0.58 (660) / - | 0.47 (758) / -0.015 | 0.48 (227) / -0.015 |
| critics_narrow/seed_0 exact min | 0.74 (230) / +0.038 | 0.73 (73) / +0.022 | 0.53 (222) / -0.009 | 0.61 (660) / - | 0.48 (758) / -0.003 | 0.48 (227) / -0.006 |
| critics_narrow/seed_0 region h0 | 0.72 (230) / +0.037 | 0.68 (73) / +0.019 | 0.60 (222) / +0.013 | 0.59 (660) / - | 0.49 (758) / -0.006 | 0.46 (227) / -0.015 |
| critics_narrow/seed_0 exact h0 | 0.74 (230) / +0.040 | 0.71 (73) / +0.021 | 0.56 (222) / +0.004 | 0.60 (660) / - | 0.48 (758) / -0.008 | 0.46 (227) / -0.012 |
| critics_narrow/seed_0 region h1 | 0.73 (230) / +0.040 | 0.70 (73) / +0.020 | 0.50 (222) / -0.008 | 0.55 (660) / - | 0.45 (758) / -0.010 | 0.40 (227) / -0.019 |
| critics_narrow/seed_0 exact h1 | 0.73 (230) / +0.041 | 0.73 (73) / +0.024 | 0.51 (222) / -0.007 | 0.59 (660) / - | 0.47 (758) / -0.006 | 0.42 (227) / -0.016 |
| critics_narrow/seed_1 region min | 0.68 (230) / +0.035 | 0.71 (73) / +0.023 | 0.62 (222) / +0.013 | 0.63 (660) / - | 0.48 (758) / +0.002 | 0.49 (227) / -0.001 |
| critics_narrow/seed_1 exact min | 0.73 (230) / +0.044 | 0.77 (73) / +0.028 | 0.62 (222) / +0.017 | 0.63 (660) / - | 0.51 (758) / +0.010 | 0.52 (227) / +0.004 |
| critics_narrow/seed_1 region h0 | 0.71 (230) / +0.037 | 0.74 (73) / +0.022 | 0.57 (222) / +0.003 | 0.59 (660) / - | 0.50 (758) / +0.005 | 0.48 (227) / -0.007 |
| critics_narrow/seed_1 exact h0 | 0.75 (230) / +0.046 | 0.77 (73) / +0.026 | 0.58 (222) / +0.013 | 0.62 (660) / - | 0.50 (758) / +0.010 | 0.48 (227) / -0.008 |
| critics_narrow/seed_1 region h1 | 0.69 (230) / +0.037 | 0.70 (73) / +0.022 | 0.62 (222) / +0.014 | 0.60 (660) / - | 0.49 (758) / +0.003 | 0.49 (227) / +0.005 |
| critics_narrow/seed_1 exact h1 | 0.69 (230) / +0.035 | 0.71 (73) / +0.025 | 0.64 (222) / +0.011 | 0.61 (660) / - | 0.48 (758) / +0.004 | 0.47 (227) / -0.005 |
| critics_narrow/seed_2 region min | 0.74 (230) / +0.041 | 0.70 (73) / +0.022 | 0.54 (222) / -0.001 | 0.59 (660) / - | 0.51 (758) / -0.001 | 0.49 (227) / +0.003 |
| critics_narrow/seed_2 exact min | 0.73 (230) / +0.043 | 0.73 (73) / +0.024 | 0.52 (222) / -0.000 | 0.60 (660) / - | 0.51 (758) / +0.006 | 0.49 (227) / +0.002 |
| critics_narrow/seed_2 region h0 | 0.75 (230) / +0.042 | 0.70 (73) / +0.022 | 0.55 (222) / -0.002 | 0.59 (660) / - | 0.49 (758) / -0.003 | 0.48 (227) / -0.004 |
| critics_narrow/seed_2 exact h0 | 0.73 (230) / +0.042 | 0.71 (73) / +0.023 | 0.54 (222) / +0.001 | 0.59 (660) / - | 0.50 (758) / -0.000 | 0.48 (227) / -0.005 |
| critics_narrow/seed_2 region h1 | 0.73 (230) / +0.037 | 0.74 (73) / +0.023 | 0.55 (222) / +0.002 | 0.58 (660) / - | 0.53 (758) / +0.013 | 0.50 (227) / +0.005 |
| critics_narrow/seed_2 exact h1 | 0.73 (230) / +0.037 | 0.77 (73) / +0.026 | 0.55 (222) / +0.007 | 0.62 (660) / - | 0.49 (758) / +0.015 | 0.45 (227) / +0.003 |

## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)

- critics_narrow/seed_0: A: succeeds (above chance, positive pick gain) [0.73 +- 0.036, n 230, gain +0.038]; B: inconclusive (interval spans chance and useful agreement) [0.57 +- 0.034, n 222, gain -0.001]; C: fails (interval below 0.6) [0.47 +- 0.037, n 758, gain -0.015]
- critics_narrow/seed_1: A: succeeds (above chance, positive pick gain) [0.68 +- 0.039, n 230, gain +0.035]; B: succeeds (above chance, positive pick gain) [0.62 +- 0.035, n 222, gain +0.013]; C: fails (interval below 0.6) [0.48 +- 0.026, n 758, gain +0.002]
- critics_narrow/seed_2: A: succeeds (above chance, positive pick gain) [0.74 +- 0.034, n 230, gain +0.041]; B: inconclusive (interval spans chance and useful agreement) [0.54 +- 0.038, n 222, gain -0.001]; C: fails (interval below 0.6) [0.51 +- 0.021, n 758, gain -0.001]
- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): +0.061; B: +0.055; C: +0.078 (a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).
