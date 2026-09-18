# Gate (round 1, FIT keys): does f(s, a, g) order the candidate torques the way their validated single-step outcomes do?

Fit anchors (n/a (fit keys) held-out episodes), candidates recorded + 2 samples (dense), recorded + 1 sample (general), 2 (dense) / 1 (general) paired draws per key; the continuation is the frozen policy mode.  A pair (a_i, a_j) at one state is VALIDATED when the log P_goal ratio has the same sign on the two halves of the draws ((0,) vs (1,)) and |ratio| > 0.3 on both.  Agreement = share of validated pairs whose critic ordering matches (chance 0.50); samples-only = pairs of two policy samples (excludes recorded and mode); weighted = |ratio|-weighted sign agreement over all pairs; pick gain = (P[argmax f] - mean P) / (max P - mean P) over anchors with spread (1 = oracle, 0 = random).  PASS = pooled dense-set validated agreement >= 0.65 and > 0.5 by 2 s.e. (s.e. = episode-level bootstrap: the pairs of one anchor share its candidates and draws, the anchors of one episode its poses and goal).  Readouts per critic: exact = the logit at the recorded goal point (min over the twin heads = what the actor optimises; h0 / h1 = the heads); region rX = log of the region-integrated exp(f) over the NCE goal marginal within radius X of the goal (the quantity P_goal measures; diagnostic only, the actor does not optimise it).

## Ceiling: split-half sign agreement of the differences themselves

| set | episodes | anchors | pairs | mean abs log ratio | frac > thr | first-half > thr | tie | weak | same | opposite | agreement among decided (s.e.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| start_early | 335 | 400 | 1200 | 0.58 | 0.15 | 126 | 85 | 1 | 36 | 4 | 0.90 (0.066) |
| turn | 49 | 300 | 900 | 1.95 | 0.55 | 434 | 31 | 38 | 319 | 46 | 0.87 (0.020) |
| start_late | 172 | 200 | 600 | 0.53 | 0.14 | 38 | 27 | 0 | 11 | 0 | 1.00 (0.000) |
| north_leg | 49 | 300 | 900 | 2.08 | 0.71 | 621 | 16 | 117 | 410 | 78 | 0.84 (0.019) |
| shortcut_early | 252 | 300 | 900 | 0.52 | 0.13 | 84 | 65 | 0 | 19 | 0 | 1.00 (0.000) |
| general | 900 | 4378 | 4378 | 0.96 | 0.24 | 0 | 0 | 0 | 0 | 0 | - (-) |

## Critics

| scorer | set | validated pairs (episodes) | agreement | s.e. | samples-only agreement (n) | weighted agreement | pick gain (anchors) | P pick / mean / best |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | start_early | 36 (16) | **0.67** | 0.080 | 0.91 (11) | 0.55 | 0.12 (193) | 0.151 / 0.146 / 0.177 |
| random | turn | 319 (49) | **0.48** | 0.030 | 0.50 (98) | 0.49 | 0.03 (225) | 0.148 / 0.147 / 0.256 |
| random | start_late | 11 (5) | **0.45** | 0.138 | 0.60 (5) | 0.59 | 0.12 (92) | 0.145 / 0.140 / 0.167 |
| random | north_leg | 410 (48) | **0.52** | 0.023 | 0.52 (129) | 0.50 | -0.01 (299) | 0.253 / 0.254 / 0.397 |
| random | shortcut_early | 19 (9) | **0.21** | 0.109 | 0.33 (6) | 0.50 | 0.11 (129) | 0.138 / 0.139 / 0.166 |
| random | general | 0 (0) | **-** | - | - (0) | 0.51 | -0.01 (2620) | 0.408 / 0.408 / 0.482 |
| random | dense_all | 795 (70) | **0.51** | 0.018 | 0.53 (249) | 0.51 | 0.06 (938) | 0.167 / 0.166 / 0.233 |
| policy_logprob | start_early | 36 (16) | **0.39** | 0.095 | 0.18 (11) | 0.47 | -0.06 (193) | 0.140 / 0.146 / 0.177 |
| policy_logprob | turn | 319 (49) | **0.50** | 0.031 | 0.41 (98) | 0.48 | -0.08 (225) | 0.146 / 0.147 / 0.256 |
| policy_logprob | start_late | 11 (5) | **0.55** | 0.189 | 0.40 (5) | 0.44 | -0.27 (92) | 0.130 / 0.140 / 0.167 |
| policy_logprob | north_leg | 410 (48) | **0.49** | 0.028 | 0.49 (129) | 0.51 | 0.02 (299) | 0.252 / 0.254 / 0.397 |
| policy_logprob | shortcut_early | 19 (9) | **0.74** | 0.105 | 1.00 (6) | 0.53 | 0.07 (129) | 0.143 / 0.139 / 0.166 |
| policy_logprob | general | 0 (0) | **-** | - | - (0) | 0.48 | -0.03 (2620) | 0.405 / 0.408 / 0.482 |
| policy_logprob | dense_all | 795 (70) | **0.49** | 0.020 | 0.45 (249) | 0.49 | -0.04 (938) | 0.163 / 0.166 / 0.233 |
| critics_r1/seed_0/final | exact min | start_early | 36 (16) | **0.94** | 0.034 | 1.00 (11) | 0.77 | 0.20 (193) | 0.163 / 0.146 / 0.177 |
| critics_r1/seed_0/final | exact min | turn | 319 (49) | **0.92** | 0.018 | 0.91 (98) | 0.94 | 0.73 (225) | 0.235 / 0.147 / 0.256 |
| critics_r1/seed_0/final | exact min | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.83 | 0.28 (92) | 0.155 / 0.140 / 0.167 |
| critics_r1/seed_0/final | exact min | north_leg | 410 (48) | **0.84** | 0.024 | 0.85 (129) | 0.85 | 0.55 (299) | 0.347 / 0.254 / 0.397 |
| critics_r1/seed_0/final | exact min | shortcut_early | 19 (9) | **0.68** | 0.109 | 0.67 (6) | 0.67 | 0.30 (129) | 0.153 / 0.139 / 0.166 |
| critics_r1/seed_0/final | exact min | general | 0 (0) | **-** | - | - (0) | 0.67 | 0.18 (2620) | 0.432 / 0.408 / 0.482 |
| critics_r1/seed_0/final | exact min | dense_all | 795 (70) | **0.87** | 0.014 | 0.88 (249) | 0.85 | 0.46 (938) | 0.211 / 0.166 / 0.233 |
| critics_r1/seed_0/final | exact h0 | start_early | 36 (16) | **0.94** | 0.034 | 1.00 (11) | 0.75 | 0.09 (193) | 0.161 / 0.146 / 0.177 |
| critics_r1/seed_0/final | exact h0 | turn | 319 (49) | **0.91** | 0.020 | 0.92 (98) | 0.94 | 0.73 (225) | 0.236 / 0.147 / 0.256 |
| critics_r1/seed_0/final | exact h0 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.81 | 0.29 (92) | 0.154 / 0.140 / 0.167 |
| critics_r1/seed_0/final | exact h0 | north_leg | 410 (48) | **0.82** | 0.025 | 0.82 (129) | 0.83 | 0.53 (299) | 0.343 / 0.254 / 0.397 |
| critics_r1/seed_0/final | exact h0 | shortcut_early | 19 (9) | **0.63** | 0.097 | 0.67 (6) | 0.66 | 0.30 (129) | 0.154 / 0.139 / 0.166 |
| critics_r1/seed_0/final | exact h0 | general | 0 (0) | **-** | - | - (0) | 0.66 | 0.16 (2620) | 0.429 / 0.408 / 0.482 |
| critics_r1/seed_0/final | exact h0 | dense_all | 795 (70) | **0.86** | 0.016 | 0.86 (249) | 0.84 | 0.43 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_0/final | exact h1 | start_early | 36 (16) | **0.89** | 0.060 | 1.00 (11) | 0.77 | 0.09 (193) | 0.160 / 0.146 / 0.177 |
| critics_r1/seed_0/final | exact h1 | turn | 319 (49) | **0.91** | 0.017 | 0.87 (98) | 0.93 | 0.72 (225) | 0.233 / 0.147 / 0.256 |
| critics_r1/seed_0/final | exact h1 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.80 | 0.27 (92) | 0.154 / 0.140 / 0.167 |
| critics_r1/seed_0/final | exact h1 | north_leg | 410 (48) | **0.85** | 0.022 | 0.85 (129) | 0.86 | 0.57 (299) | 0.349 / 0.254 / 0.397 |
| critics_r1/seed_0/final | exact h1 | shortcut_early | 19 (9) | **0.68** | 0.109 | 0.67 (6) | 0.70 | 0.24 (129) | 0.154 / 0.139 / 0.166 |
| critics_r1/seed_0/final | exact h1 | general | 0 (0) | **-** | - | - (0) | 0.69 | 0.20 (2620) | 0.435 / 0.408 / 0.482 |
| critics_r1/seed_0/final | exact h1 | dense_all | 795 (70) | **0.87** | 0.013 | 0.86 (249) | 0.85 | 0.43 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r0.5 min | start_early | 36 (16) | **0.94** | 0.035 | 1.00 (11) | 0.75 | 0.17 (193) | 0.161 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r0.5 min | turn | 319 (49) | **0.94** | 0.016 | 0.94 (98) | 0.94 | 0.73 (225) | 0.235 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r0.5 min | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.78 | 0.21 (92) | 0.155 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r0.5 min | north_leg | 410 (48) | **0.87** | 0.018 | 0.89 (129) | 0.87 | 0.59 (299) | 0.345 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r0.5 min | shortcut_early | 19 (9) | **0.74** | 0.084 | 0.67 (6) | 0.63 | 0.18 (129) | 0.148 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r0.5 min | general | 0 (0) | **-** | - | - (0) | 0.64 | 0.13 (2620) | 0.424 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r0.5 min | dense_all | 795 (70) | **0.90** | 0.011 | 0.91 (249) | 0.85 | 0.44 (938) | 0.209 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r0.5 h0 | start_early | 36 (16) | **0.94** | 0.035 | 1.00 (11) | 0.75 | 0.14 (193) | 0.162 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r0.5 h0 | turn | 319 (49) | **0.95** | 0.016 | 0.95 (98) | 0.94 | 0.72 (225) | 0.234 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r0.5 h0 | start_late | 11 (5) | **0.55** | 0.190 | 0.60 (5) | 0.70 | 0.21 (92) | 0.148 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r0.5 h0 | north_leg | 410 (48) | **0.86** | 0.021 | 0.87 (129) | 0.86 | 0.57 (299) | 0.343 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r0.5 h0 | shortcut_early | 19 (9) | **0.63** | 0.099 | 0.50 (6) | 0.64 | 0.34 (129) | 0.155 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r0.5 h0 | general | 0 (0) | **-** | - | - (0) | 0.64 | 0.13 (2620) | 0.425 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r0.5 h0 | dense_all | 795 (70) | **0.89** | 0.013 | 0.89 (249) | 0.84 | 0.45 (938) | 0.209 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r0.5 h1 | start_early | 36 (16) | **0.92** | 0.057 | 1.00 (11) | 0.74 | 0.09 (193) | 0.159 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r0.5 h1 | turn | 319 (49) | **0.95** | 0.014 | 0.94 (98) | 0.95 | 0.77 (225) | 0.240 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r0.5 h1 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.79 | 0.33 (92) | 0.153 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r0.5 h1 | north_leg | 410 (48) | **0.88** | 0.017 | 0.88 (129) | 0.87 | 0.58 (299) | 0.348 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r0.5 h1 | shortcut_early | 19 (9) | **0.79** | 0.082 | 0.83 (6) | 0.64 | 0.19 (129) | 0.149 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r0.5 h1 | general | 0 (0) | **-** | - | - (0) | 0.65 | 0.14 (2620) | 0.427 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r0.5 h1 | dense_all | 795 (70) | **0.91** | 0.010 | 0.90 (249) | 0.85 | 0.45 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r1 min | start_early | 36 (16) | **0.75** | 0.090 | 0.82 (11) | 0.62 | 0.02 (193) | 0.153 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r1 min | turn | 319 (49) | **0.92** | 0.018 | 0.89 (98) | 0.92 | 0.63 (225) | 0.227 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r1 min | start_late | 11 (5) | **0.36** | 0.172 | 0.40 (5) | 0.67 | 0.26 (92) | 0.150 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r1 min | north_leg | 410 (48) | **0.83** | 0.021 | 0.82 (129) | 0.83 | 0.52 (299) | 0.336 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r1 min | shortcut_early | 19 (9) | **0.63** | 0.077 | 0.50 (6) | 0.52 | 0.07 (129) | 0.144 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r1 min | general | 0 (0) | **-** | - | - (0) | 0.56 | 0.04 (2620) | 0.411 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r1 min | dense_all | 795 (70) | **0.85** | 0.015 | 0.83 (249) | 0.80 | 0.36 (938) | 0.202 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r1 h0 | start_early | 36 (16) | **0.72** | 0.077 | 0.73 (11) | 0.57 | -0.03 (193) | 0.151 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r1 h0 | turn | 319 (49) | **0.93** | 0.017 | 0.90 (98) | 0.92 | 0.62 (225) | 0.226 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r1 h0 | start_late | 11 (5) | **0.36** | 0.214 | 0.40 (5) | 0.56 | 0.20 (92) | 0.141 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r1 h0 | north_leg | 410 (48) | **0.82** | 0.022 | 0.81 (129) | 0.82 | 0.48 (299) | 0.331 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r1 h0 | shortcut_early | 19 (9) | **0.58** | 0.090 | 0.50 (6) | 0.53 | 0.13 (129) | 0.144 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r1 h0 | general | 0 (0) | **-** | - | - (0) | 0.55 | 0.07 (2620) | 0.411 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r1 h0 | dense_all | 795 (70) | **0.85** | 0.016 | 0.82 (249) | 0.78 | 0.33 (938) | 0.199 / 0.166 / 0.233 |
| critics_r1/seed_0/final | region r1 h1 | start_early | 36 (16) | **0.72** | 0.089 | 0.82 (11) | 0.59 | -0.13 (193) | 0.148 / 0.146 / 0.177 |
| critics_r1/seed_0/final | region r1 h1 | turn | 319 (49) | **0.92** | 0.016 | 0.90 (98) | 0.91 | 0.65 (225) | 0.231 / 0.147 / 0.256 |
| critics_r1/seed_0/final | region r1 h1 | start_late | 11 (5) | **0.64** | 0.157 | 0.40 (5) | 0.68 | 0.26 (92) | 0.149 / 0.140 / 0.167 |
| critics_r1/seed_0/final | region r1 h1 | north_leg | 410 (48) | **0.82** | 0.024 | 0.78 (129) | 0.82 | 0.50 (299) | 0.331 / 0.254 / 0.397 |
| critics_r1/seed_0/final | region r1 h1 | shortcut_early | 19 (9) | **0.74** | 0.085 | 0.83 (6) | 0.57 | 0.18 (129) | 0.147 / 0.139 / 0.166 |
| critics_r1/seed_0/final | region r1 h1 | general | 0 (0) | **-** | - | - (0) | 0.57 | 0.07 (2620) | 0.414 / 0.408 / 0.482 |
| critics_r1/seed_0/final | region r1 h1 | dense_all | 795 (70) | **0.85** | 0.014 | 0.82 (249) | 0.79 | 0.34 (938) | 0.201 / 0.166 / 0.233 |
| critics_r1/seed_1/final | exact min | start_early | 36 (16) | **0.83** | 0.064 | 0.91 (11) | 0.77 | 0.16 (193) | 0.161 / 0.146 / 0.177 |
| critics_r1/seed_1/final | exact min | turn | 319 (49) | **0.92** | 0.017 | 0.93 (98) | 0.93 | 0.67 (225) | 0.229 / 0.147 / 0.256 |
| critics_r1/seed_1/final | exact min | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.69 | 0.16 (92) | 0.153 / 0.140 / 0.167 |
| critics_r1/seed_1/final | exact min | north_leg | 410 (48) | **0.85** | 0.020 | 0.86 (129) | 0.85 | 0.55 (299) | 0.344 / 0.254 / 0.397 |
| critics_r1/seed_1/final | exact min | shortcut_early | 19 (9) | **0.74** | 0.111 | 0.50 (6) | 0.68 | 0.16 (129) | 0.152 / 0.139 / 0.166 |
| critics_r1/seed_1/final | exact min | general | 0 (0) | **-** | - | - (0) | 0.63 | 0.11 (2620) | 0.426 / 0.408 / 0.482 |
| critics_r1/seed_1/final | exact min | dense_all | 795 (70) | **0.88** | 0.014 | 0.88 (249) | 0.84 | 0.41 (938) | 0.208 / 0.166 / 0.233 |
| critics_r1/seed_1/final | exact h0 | start_early | 36 (16) | **0.72** | 0.077 | 0.73 (11) | 0.66 | 0.05 (193) | 0.157 / 0.146 / 0.177 |
| critics_r1/seed_1/final | exact h0 | turn | 319 (49) | **0.90** | 0.017 | 0.88 (98) | 0.90 | 0.57 (225) | 0.218 / 0.147 / 0.256 |
| critics_r1/seed_1/final | exact h0 | start_late | 11 (5) | **0.64** | 0.153 | 0.60 (5) | 0.56 | 0.02 (92) | 0.144 / 0.140 / 0.167 |
| critics_r1/seed_1/final | exact h0 | north_leg | 410 (48) | **0.77** | 0.020 | 0.77 (129) | 0.78 | 0.46 (299) | 0.325 / 0.254 / 0.397 |
| critics_r1/seed_1/final | exact h0 | shortcut_early | 19 (9) | **0.58** | 0.124 | 0.33 (6) | 0.55 | 0.05 (129) | 0.143 / 0.139 / 0.166 |
| critics_r1/seed_1/final | exact h0 | general | 0 (0) | **-** | - | - (0) | 0.59 | 0.07 (2620) | 0.420 / 0.408 / 0.482 |
| critics_r1/seed_1/final | exact h0 | dense_all | 795 (70) | **0.81** | 0.015 | 0.80 (249) | 0.77 | 0.30 (938) | 0.198 / 0.166 / 0.233 |
| critics_r1/seed_1/final | exact h1 | start_early | 36 (16) | **0.86** | 0.062 | 1.00 (11) | 0.78 | 0.17 (193) | 0.163 / 0.146 / 0.177 |
| critics_r1/seed_1/final | exact h1 | turn | 319 (49) | **0.91** | 0.016 | 0.90 (98) | 0.93 | 0.67 (225) | 0.227 / 0.147 / 0.256 |
| critics_r1/seed_1/final | exact h1 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.75 | 0.23 (92) | 0.155 / 0.140 / 0.167 |
| critics_r1/seed_1/final | exact h1 | north_leg | 410 (48) | **0.86** | 0.020 | 0.87 (129) | 0.86 | 0.56 (299) | 0.348 / 0.254 / 0.397 |
| critics_r1/seed_1/final | exact h1 | shortcut_early | 19 (9) | **0.79** | 0.110 | 0.67 (6) | 0.70 | 0.23 (129) | 0.154 / 0.139 / 0.166 |
| critics_r1/seed_1/final | exact h1 | general | 0 (0) | **-** | - | - (0) | 0.64 | 0.12 (2620) | 0.427 / 0.408 / 0.482 |
| critics_r1/seed_1/final | exact h1 | dense_all | 795 (70) | **0.88** | 0.013 | 0.88 (249) | 0.85 | 0.43 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r0.5 min | start_early | 36 (16) | **0.81** | 0.087 | 0.91 (11) | 0.76 | 0.15 (193) | 0.161 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r0.5 min | turn | 319 (49) | **0.94** | 0.015 | 0.95 (98) | 0.93 | 0.69 (225) | 0.227 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r0.5 min | start_late | 11 (5) | **0.73** | 0.171 | 0.60 (5) | 0.69 | 0.11 (92) | 0.149 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r0.5 min | north_leg | 410 (48) | **0.84** | 0.021 | 0.82 (129) | 0.85 | 0.45 (299) | 0.331 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r0.5 min | shortcut_early | 19 (9) | **0.74** | 0.111 | 0.67 (6) | 0.58 | 0.01 (129) | 0.144 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r0.5 min | general | 0 (0) | **-** | - | - (0) | 0.61 | 0.06 (2620) | 0.422 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r0.5 min | dense_all | 795 (70) | **0.87** | 0.013 | 0.87 (249) | 0.83 | 0.35 (938) | 0.203 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r0.5 h0 | start_early | 36 (16) | **0.67** | 0.097 | 0.73 (11) | 0.62 | 0.11 (193) | 0.155 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r0.5 h0 | turn | 319 (49) | **0.90** | 0.019 | 0.89 (98) | 0.90 | 0.53 (225) | 0.211 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r0.5 h0 | start_late | 11 (5) | **0.55** | 0.167 | 0.60 (5) | 0.54 | 0.05 (92) | 0.142 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r0.5 h0 | north_leg | 410 (48) | **0.76** | 0.022 | 0.72 (129) | 0.77 | 0.38 (299) | 0.312 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r0.5 h0 | shortcut_early | 19 (9) | **0.63** | 0.105 | 0.50 (6) | 0.56 | 0.05 (129) | 0.147 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r0.5 h0 | general | 0 (0) | **-** | - | - (0) | 0.56 | 0.04 (2620) | 0.414 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r0.5 h0 | dense_all | 795 (70) | **0.81** | 0.018 | 0.78 (249) | 0.76 | 0.28 (938) | 0.194 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r0.5 h1 | start_early | 36 (16) | **0.86** | 0.064 | 0.91 (11) | 0.75 | 0.18 (193) | 0.164 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r0.5 h1 | turn | 319 (49) | **0.93** | 0.015 | 0.90 (98) | 0.94 | 0.68 (225) | 0.233 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r0.5 h1 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.70 | 0.15 (92) | 0.150 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r0.5 h1 | north_leg | 410 (48) | **0.87** | 0.016 | 0.85 (129) | 0.86 | 0.55 (299) | 0.340 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r0.5 h1 | shortcut_early | 19 (9) | **0.68** | 0.135 | 0.67 (6) | 0.59 | 0.09 (129) | 0.146 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r0.5 h1 | general | 0 (0) | **-** | - | - (0) | 0.63 | 0.11 (2620) | 0.423 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r0.5 h1 | dense_all | 795 (70) | **0.89** | 0.011 | 0.87 (249) | 0.84 | 0.40 (938) | 0.207 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r1 min | start_early | 36 (16) | **0.69** | 0.097 | 0.73 (11) | 0.63 | 0.11 (193) | 0.155 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r1 min | turn | 319 (49) | **0.88** | 0.022 | 0.86 (98) | 0.90 | 0.46 (225) | 0.208 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r1 min | start_late | 11 (5) | **0.73** | 0.171 | 0.60 (5) | 0.60 | 0.14 (92) | 0.147 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r1 min | north_leg | 410 (48) | **0.74** | 0.026 | 0.73 (129) | 0.77 | 0.22 (299) | 0.295 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r1 min | shortcut_early | 19 (9) | **0.68** | 0.135 | 0.67 (6) | 0.53 | -0.04 (129) | 0.141 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r1 min | general | 0 (0) | **-** | - | - (0) | 0.56 | 0.02 (2620) | 0.414 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r1 min | dense_all | 795 (70) | **0.79** | 0.018 | 0.78 (249) | 0.76 | 0.21 (938) | 0.190 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r1 h0 | start_early | 36 (16) | **0.64** | 0.096 | 0.73 (11) | 0.57 | 0.10 (193) | 0.152 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r1 h0 | turn | 319 (49) | **0.88** | 0.021 | 0.86 (98) | 0.89 | 0.43 (225) | 0.202 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r1 h0 | start_late | 11 (5) | **0.45** | 0.136 | 0.60 (5) | 0.46 | -0.02 (92) | 0.137 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r1 h0 | north_leg | 410 (48) | **0.68** | 0.027 | 0.67 (129) | 0.71 | 0.21 (299) | 0.283 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r1 h0 | shortcut_early | 19 (9) | **0.42** | 0.115 | 0.33 (6) | 0.50 | -0.04 (129) | 0.141 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r1 h0 | general | 0 (0) | **-** | - | - (0) | 0.54 | 0.01 (2620) | 0.411 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r1 h0 | dense_all | 795 (70) | **0.75** | 0.020 | 0.73 (249) | 0.72 | 0.18 (938) | 0.184 / 0.166 / 0.233 |
| critics_r1/seed_1/final | region r1 h1 | start_early | 36 (16) | **0.67** | 0.104 | 0.73 (11) | 0.61 | -0.02 (193) | 0.150 / 0.146 / 0.177 |
| critics_r1/seed_1/final | region r1 h1 | turn | 319 (49) | **0.89** | 0.021 | 0.86 (98) | 0.90 | 0.57 (225) | 0.222 / 0.147 / 0.256 |
| critics_r1/seed_1/final | region r1 h1 | start_late | 11 (5) | **0.82** | 0.168 | 0.80 (5) | 0.65 | 0.11 (92) | 0.147 / 0.140 / 0.167 |
| critics_r1/seed_1/final | region r1 h1 | north_leg | 410 (48) | **0.79** | 0.020 | 0.76 (129) | 0.79 | 0.34 (299) | 0.315 / 0.254 / 0.397 |
| critics_r1/seed_1/final | region r1 h1 | shortcut_early | 19 (9) | **0.74** | 0.113 | 0.67 (6) | 0.54 | 0.02 (129) | 0.142 / 0.139 / 0.166 |
| critics_r1/seed_1/final | region r1 h1 | general | 0 (0) | **-** | - | - (0) | 0.58 | 0.08 (2620) | 0.416 / 0.408 / 0.482 |
| critics_r1/seed_1/final | region r1 h1 | dense_all | 795 (70) | **0.82** | 0.016 | 0.80 (249) | 0.77 | 0.25 (938) | 0.195 / 0.166 / 0.233 |
| critics_r1/seed_2/final | exact min | start_early | 36 (16) | **0.89** | 0.060 | 1.00 (11) | 0.79 | 0.15 (193) | 0.162 / 0.146 / 0.177 |
| critics_r1/seed_2/final | exact min | turn | 319 (49) | **0.91** | 0.017 | 0.89 (98) | 0.93 | 0.73 (225) | 0.233 / 0.147 / 0.256 |
| critics_r1/seed_2/final | exact min | start_late | 11 (5) | **0.64** | 0.153 | 0.60 (5) | 0.66 | 0.01 (92) | 0.143 / 0.140 / 0.167 |
| critics_r1/seed_2/final | exact min | north_leg | 410 (48) | **0.87** | 0.019 | 0.88 (129) | 0.87 | 0.61 (299) | 0.353 / 0.254 / 0.397 |
| critics_r1/seed_2/final | exact min | shortcut_early | 19 (9) | **0.79** | 0.111 | 0.83 (6) | 0.69 | 0.01 (129) | 0.149 / 0.139 / 0.166 |
| critics_r1/seed_2/final | exact min | general | 0 (0) | **-** | - | - (0) | 0.69 | 0.18 (2620) | 0.434 / 0.408 / 0.482 |
| critics_r1/seed_2/final | exact min | dense_all | 795 (70) | **0.88** | 0.012 | 0.88 (249) | 0.85 | 0.40 (938) | 0.209 / 0.166 / 0.233 |
| critics_r1/seed_2/final | exact h0 | start_early | 36 (16) | **0.89** | 0.060 | 1.00 (11) | 0.78 | 0.10 (193) | 0.163 / 0.146 / 0.177 |
| critics_r1/seed_2/final | exact h0 | turn | 319 (49) | **0.91** | 0.016 | 0.89 (98) | 0.93 | 0.72 (225) | 0.232 / 0.147 / 0.256 |
| critics_r1/seed_2/final | exact h0 | start_late | 11 (5) | **0.82** | 0.105 | 0.80 (5) | 0.75 | 0.14 (92) | 0.155 / 0.140 / 0.167 |
| critics_r1/seed_2/final | exact h0 | north_leg | 410 (48) | **0.86** | 0.024 | 0.87 (129) | 0.87 | 0.59 (299) | 0.350 / 0.254 / 0.397 |
| critics_r1/seed_2/final | exact h0 | shortcut_early | 19 (9) | **0.74** | 0.112 | 0.83 (6) | 0.64 | 0.05 (129) | 0.150 / 0.139 / 0.166 |
| critics_r1/seed_2/final | exact h0 | general | 0 (0) | **-** | - | - (0) | 0.70 | 0.18 (2620) | 0.435 / 0.408 / 0.482 |
| critics_r1/seed_2/final | exact h0 | dense_all | 795 (70) | **0.88** | 0.013 | 0.88 (249) | 0.85 | 0.40 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_2/final | exact h1 | start_early | 36 (16) | **0.94** | 0.034 | 1.00 (11) | 0.77 | 0.17 (193) | 0.162 / 0.146 / 0.177 |
| critics_r1/seed_2/final | exact h1 | turn | 319 (49) | **0.91** | 0.020 | 0.89 (98) | 0.93 | 0.75 (225) | 0.235 / 0.147 / 0.256 |
| critics_r1/seed_2/final | exact h1 | start_late | 11 (5) | **0.64** | 0.153 | 0.60 (5) | 0.69 | 0.08 (92) | 0.147 / 0.140 / 0.167 |
| critics_r1/seed_2/final | exact h1 | north_leg | 410 (48) | **0.86** | 0.019 | 0.85 (129) | 0.86 | 0.60 (299) | 0.350 / 0.254 / 0.397 |
| critics_r1/seed_2/final | exact h1 | shortcut_early | 19 (9) | **0.84** | 0.069 | 0.67 (6) | 0.69 | 0.02 (129) | 0.148 / 0.139 / 0.166 |
| critics_r1/seed_2/final | exact h1 | general | 0 (0) | **-** | - | - (0) | 0.66 | 0.15 (2620) | 0.432 / 0.408 / 0.482 |
| critics_r1/seed_2/final | exact h1 | dense_all | 795 (70) | **0.88** | 0.013 | 0.86 (249) | 0.85 | 0.42 (938) | 0.210 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r0.5 min | start_early | 36 (16) | **0.92** | 0.059 | 0.91 (11) | 0.77 | 0.12 (193) | 0.163 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r0.5 min | turn | 319 (49) | **0.94** | 0.013 | 0.91 (98) | 0.95 | 0.77 (225) | 0.237 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r0.5 min | start_late | 11 (5) | **0.73** | 0.094 | 0.60 (5) | 0.71 | 0.16 (92) | 0.149 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r0.5 min | north_leg | 410 (48) | **0.89** | 0.014 | 0.91 (129) | 0.87 | 0.59 (299) | 0.346 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r0.5 min | shortcut_early | 19 (9) | **0.74** | 0.107 | 0.83 (6) | 0.62 | -0.01 (129) | 0.146 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r0.5 min | general | 0 (0) | **-** | - | - (0) | 0.68 | 0.17 (2620) | 0.431 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r0.5 min | dense_all | 795 (70) | **0.90** | 0.010 | 0.90 (249) | 0.85 | 0.41 (938) | 0.209 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r0.5 h0 | start_early | 36 (16) | **0.92** | 0.057 | 1.00 (11) | 0.74 | 0.06 (193) | 0.160 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r0.5 h0 | turn | 319 (49) | **0.93** | 0.014 | 0.90 (98) | 0.95 | 0.76 (225) | 0.234 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r0.5 h0 | start_late | 11 (5) | **0.73** | 0.094 | 0.60 (5) | 0.74 | 0.19 (92) | 0.153 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r0.5 h0 | north_leg | 410 (48) | **0.88** | 0.015 | 0.90 (129) | 0.87 | 0.60 (299) | 0.346 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r0.5 h0 | shortcut_early | 19 (9) | **0.68** | 0.106 | 0.83 (6) | 0.64 | 0.09 (129) | 0.148 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r0.5 h0 | general | 0 (0) | **-** | - | - (0) | 0.68 | 0.15 (2620) | 0.431 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r0.5 h0 | dense_all | 795 (70) | **0.90** | 0.010 | 0.90 (249) | 0.85 | 0.42 (938) | 0.209 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r0.5 h1 | start_early | 36 (16) | **0.92** | 0.059 | 0.91 (11) | 0.76 | 0.23 (193) | 0.164 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r0.5 h1 | turn | 319 (49) | **0.94** | 0.013 | 0.92 (98) | 0.94 | 0.75 (225) | 0.235 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r0.5 h1 | start_late | 11 (5) | **0.64** | 0.153 | 0.60 (5) | 0.66 | 0.04 (92) | 0.144 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r0.5 h1 | north_leg | 410 (48) | **0.86** | 0.017 | 0.89 (129) | 0.86 | 0.56 (299) | 0.344 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r0.5 h1 | shortcut_early | 19 (9) | **0.68** | 0.105 | 0.67 (6) | 0.60 | -0.02 (129) | 0.143 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r0.5 h1 | general | 0 (0) | **-** | - | - (0) | 0.66 | 0.16 (2620) | 0.430 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r0.5 h1 | dense_all | 795 (70) | **0.89** | 0.011 | 0.89 (249) | 0.84 | 0.41 (938) | 0.207 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r1 min | start_early | 36 (16) | **0.83** | 0.076 | 0.91 (11) | 0.67 | 0.01 (193) | 0.155 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r1 min | turn | 319 (49) | **0.90** | 0.016 | 0.86 (98) | 0.91 | 0.62 (225) | 0.226 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r1 min | start_late | 11 (5) | **0.45** | 0.168 | 0.40 (5) | 0.52 | 0.01 (92) | 0.136 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r1 min | north_leg | 410 (48) | **0.80** | 0.021 | 0.82 (129) | 0.80 | 0.43 (299) | 0.317 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r1 min | shortcut_early | 19 (9) | **0.68** | 0.104 | 0.83 (6) | 0.57 | -0.07 (129) | 0.145 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r1 min | general | 0 (0) | **-** | - | - (0) | 0.61 | 0.11 (2620) | 0.421 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r1 min | dense_all | 795 (70) | **0.83** | 0.014 | 0.83 (249) | 0.78 | 0.28 (938) | 0.197 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r1 h0 | start_early | 36 (16) | **0.83** | 0.073 | 0.91 (11) | 0.66 | -0.06 (193) | 0.153 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r1 h0 | turn | 319 (49) | **0.90** | 0.018 | 0.86 (98) | 0.92 | 0.62 (225) | 0.224 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r1 h0 | start_late | 11 (5) | **0.55** | 0.036 | 0.40 (5) | 0.57 | 0.09 (92) | 0.145 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r1 h0 | north_leg | 410 (48) | **0.81** | 0.022 | 0.83 (129) | 0.81 | 0.45 (299) | 0.323 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r1 h0 | shortcut_early | 19 (9) | **0.63** | 0.101 | 0.83 (6) | 0.60 | 0.04 (129) | 0.145 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r1 h0 | general | 0 (0) | **-** | - | - (0) | 0.61 | 0.09 (2620) | 0.419 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r1 h0 | dense_all | 795 (70) | **0.84** | 0.015 | 0.84 (249) | 0.79 | 0.30 (938) | 0.199 / 0.166 / 0.233 |
| critics_r1/seed_2/final | region r1 h1 | start_early | 36 (16) | **0.81** | 0.086 | 0.82 (11) | 0.66 | 0.11 (193) | 0.155 / 0.146 / 0.177 |
| critics_r1/seed_2/final | region r1 h1 | turn | 319 (49) | **0.91** | 0.016 | 0.87 (98) | 0.91 | 0.63 (225) | 0.225 / 0.147 / 0.256 |
| critics_r1/seed_2/final | region r1 h1 | start_late | 11 (5) | **0.45** | 0.168 | 0.40 (5) | 0.48 | -0.09 (92) | 0.136 / 0.140 / 0.167 |
| critics_r1/seed_2/final | region r1 h1 | north_leg | 410 (48) | **0.77** | 0.021 | 0.76 (129) | 0.78 | 0.37 (299) | 0.310 / 0.254 / 0.397 |
| critics_r1/seed_2/final | region r1 h1 | shortcut_early | 19 (9) | **0.68** | 0.111 | 0.83 (6) | 0.57 | -0.02 (129) | 0.145 / 0.139 / 0.166 |
| critics_r1/seed_2/final | region r1 h1 | general | 0 (0) | **-** | - | - (0) | 0.59 | 0.09 (2620) | 0.419 / 0.408 / 0.482 |
| critics_r1/seed_2/final | region r1 h1 | dense_all | 795 (70) | **0.82** | 0.014 | 0.80 (249) | 0.77 | 0.28 (938) | 0.196 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | exact min | start_early | 36 (16) | **0.50** | 0.090 | 0.64 (11) | 0.51 | 0.04 (193) | 0.149 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | exact min | turn | 319 (49) | **0.48** | 0.031 | 0.50 (98) | 0.48 | -0.10 (225) | 0.139 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | exact min | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.55 | 0.07 (92) | 0.141 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | exact min | north_leg | 410 (48) | **0.50** | 0.023 | 0.52 (129) | 0.50 | 0.01 (299) | 0.255 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | exact min | shortcut_early | 19 (9) | **0.58** | 0.130 | 0.50 (6) | 0.51 | 0.04 (129) | 0.140 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | exact min | general | 0 (0) | **-** | - | - (0) | 0.51 | 0.00 (2620) | 0.409 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | exact min | dense_all | 795 (70) | **0.49** | 0.017 | 0.51 (249) | 0.50 | -0.00 (938) | 0.165 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | exact h0 | start_early | 36 (16) | **0.53** | 0.087 | 0.64 (11) | 0.52 | 0.07 (193) | 0.151 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | exact h0 | turn | 319 (49) | **0.47** | 0.031 | 0.45 (98) | 0.46 | -0.07 (225) | 0.145 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | exact h0 | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.55 | 0.03 (92) | 0.141 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | exact h0 | north_leg | 410 (48) | **0.51** | 0.029 | 0.53 (129) | 0.52 | 0.09 (299) | 0.263 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | exact h0 | shortcut_early | 19 (9) | **0.47** | 0.131 | 0.50 (6) | 0.45 | -0.13 (129) | 0.133 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | exact h0 | general | 0 (0) | **-** | - | - (0) | 0.51 | 0.02 (2620) | 0.409 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | exact h0 | dense_all | 795 (70) | **0.49** | 0.020 | 0.49 (249) | 0.49 | 0.01 (938) | 0.167 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | exact h1 | start_early | 36 (16) | **0.47** | 0.090 | 0.36 (11) | 0.53 | 0.06 (193) | 0.145 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | exact h1 | turn | 319 (49) | **0.47** | 0.031 | 0.48 (98) | 0.48 | -0.10 (225) | 0.138 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | exact h1 | start_late | 11 (5) | **0.73** | 0.112 | 0.60 (5) | 0.51 | 0.07 (92) | 0.149 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | exact h1 | north_leg | 410 (48) | **0.50** | 0.027 | 0.51 (129) | 0.51 | 0.03 (299) | 0.257 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | exact h1 | shortcut_early | 19 (9) | **0.58** | 0.150 | 0.50 (6) | 0.57 | 0.12 (129) | 0.143 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | exact h1 | general | 0 (0) | **-** | - | - (0) | 0.48 | -0.04 (2620) | 0.406 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | exact h1 | dense_all | 795 (70) | **0.49** | 0.018 | 0.49 (249) | 0.51 | 0.02 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | start_early | 36 (16) | **0.50** | 0.096 | 0.64 (11) | 0.54 | 0.06 (193) | 0.149 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | turn | 319 (49) | **0.48** | 0.031 | 0.48 (98) | 0.48 | -0.09 (225) | 0.142 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.53 | 0.02 (92) | 0.140 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | north_leg | 410 (48) | **0.51** | 0.025 | 0.52 (129) | 0.51 | -0.00 (299) | 0.256 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | shortcut_early | 19 (9) | **0.47** | 0.110 | 0.33 (6) | 0.48 | -0.01 (129) | 0.139 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | general | 0 (0) | **-** | - | - (0) | 0.51 | -0.01 (2620) | 0.410 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | dense_all | 795 (70) | **0.49** | 0.017 | 0.50 (249) | 0.50 | -0.01 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | start_early | 36 (16) | **0.56** | 0.083 | 0.73 (11) | 0.54 | 0.06 (193) | 0.151 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | turn | 319 (49) | **0.45** | 0.029 | 0.44 (98) | 0.45 | -0.07 (225) | 0.143 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.53 | -0.01 (92) | 0.140 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | north_leg | 410 (48) | **0.51** | 0.031 | 0.52 (129) | 0.51 | 0.09 (299) | 0.264 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | shortcut_early | 19 (9) | **0.42** | 0.119 | 0.33 (6) | 0.43 | -0.15 (129) | 0.131 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | general | 0 (0) | **-** | - | - (0) | 0.50 | 0.01 (2620) | 0.409 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | dense_all | 795 (70) | **0.49** | 0.021 | 0.49 (249) | 0.49 | 0.00 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | start_early | 36 (16) | **0.44** | 0.093 | 0.36 (11) | 0.52 | 0.01 (193) | 0.142 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | turn | 319 (49) | **0.46** | 0.031 | 0.47 (98) | 0.47 | -0.09 (225) | 0.139 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | start_late | 11 (5) | **0.82** | 0.105 | 0.80 (5) | 0.53 | 0.08 (92) | 0.149 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | north_leg | 410 (48) | **0.50** | 0.028 | 0.49 (129) | 0.50 | -0.01 (299) | 0.254 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | shortcut_early | 19 (9) | **0.58** | 0.150 | 0.50 (6) | 0.57 | 0.15 (129) | 0.144 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | general | 0 (0) | **-** | - | - (0) | 0.49 | -0.04 (2620) | 0.407 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | dense_all | 795 (70) | **0.49** | 0.020 | 0.48 (249) | 0.50 | 0.01 (938) | 0.165 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r1 min | start_early | 36 (16) | **0.50** | 0.089 | 0.55 (11) | 0.52 | 0.10 (193) | 0.148 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r1 min | turn | 319 (49) | **0.48** | 0.033 | 0.47 (98) | 0.49 | -0.09 (225) | 0.142 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r1 min | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.50 | 0.02 (92) | 0.140 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r1 min | north_leg | 410 (48) | **0.51** | 0.025 | 0.53 (129) | 0.50 | 0.02 (299) | 0.260 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r1 min | shortcut_early | 19 (9) | **0.42** | 0.092 | 0.33 (6) | 0.48 | -0.06 (129) | 0.138 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r1 min | general | 0 (0) | **-** | - | - (0) | 0.51 | -0.02 (2620) | 0.409 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r1 min | dense_all | 795 (70) | **0.49** | 0.015 | 0.49 (249) | 0.50 | -0.00 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | start_early | 36 (16) | **0.53** | 0.079 | 0.64 (11) | 0.51 | 0.06 (193) | 0.148 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | turn | 319 (49) | **0.47** | 0.032 | 0.44 (98) | 0.48 | -0.01 (225) | 0.148 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | start_late | 11 (5) | **0.36** | 0.154 | 0.20 (5) | 0.52 | -0.00 (92) | 0.140 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | north_leg | 410 (48) | **0.50** | 0.025 | 0.54 (129) | 0.50 | 0.08 (299) | 0.264 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | shortcut_early | 19 (9) | **0.37** | 0.099 | 0.33 (6) | 0.44 | -0.13 (129) | 0.133 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | general | 0 (0) | **-** | - | - (0) | 0.50 | -0.00 (2620) | 0.408 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | dense_all | 795 (70) | **0.48** | 0.019 | 0.49 (249) | 0.49 | 0.02 (938) | 0.167 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | start_early | 36 (16) | **0.39** | 0.095 | 0.36 (11) | 0.51 | -0.01 (193) | 0.141 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | turn | 319 (49) | **0.45** | 0.029 | 0.45 (98) | 0.47 | -0.10 (225) | 0.139 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | start_late | 11 (5) | **0.82** | 0.105 | 0.80 (5) | 0.54 | 0.11 (92) | 0.151 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | north_leg | 410 (48) | **0.49** | 0.026 | 0.50 (129) | 0.49 | -0.03 (299) | 0.254 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | shortcut_early | 19 (9) | **0.53** | 0.142 | 0.50 (6) | 0.55 | 0.14 (129) | 0.142 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | general | 0 (0) | **-** | - | - (0) | 0.50 | -0.04 (2620) | 0.408 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | dense_all | 795 (70) | **0.47** | 0.017 | 0.48 (249) | 0.49 | -0.00 (938) | 0.165 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | exact min | start_early | 36 (16) | **0.47** | 0.092 | 0.55 (11) | 0.52 | -0.06 (193) | 0.145 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | exact min | turn | 319 (49) | **0.46** | 0.029 | 0.44 (98) | 0.47 | -0.04 (225) | 0.145 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | exact min | start_late | 11 (5) | **0.55** | 0.036 | 0.40 (5) | 0.57 | -0.02 (92) | 0.143 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | exact min | north_leg | 410 (48) | **0.49** | 0.023 | 0.47 (129) | 0.49 | 0.06 (299) | 0.255 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | exact min | shortcut_early | 19 (9) | **0.42** | 0.133 | 0.17 (6) | 0.46 | -0.09 (129) | 0.137 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | exact min | general | 0 (0) | **-** | - | - (0) | 0.52 | 0.03 (2620) | 0.412 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | exact min | dense_all | 795 (70) | **0.48** | 0.018 | 0.45 (249) | 0.49 | -0.02 (938) | 0.165 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | exact h0 | start_early | 36 (16) | **0.47** | 0.089 | 0.55 (11) | 0.48 | -0.07 (193) | 0.144 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | exact h0 | turn | 319 (49) | **0.46** | 0.028 | 0.41 (98) | 0.48 | -0.06 (225) | 0.143 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | exact h0 | start_late | 11 (5) | **0.55** | 0.036 | 0.20 (5) | 0.53 | 0.03 (92) | 0.141 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | exact h0 | north_leg | 410 (48) | **0.49** | 0.028 | 0.55 (129) | 0.49 | 0.07 (299) | 0.261 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | exact h0 | shortcut_early | 19 (9) | **0.47** | 0.131 | 0.50 (6) | 0.53 | 0.01 (129) | 0.139 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | exact h0 | general | 0 (0) | **-** | - | - (0) | 0.53 | 0.02 (2620) | 0.412 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | exact h0 | dense_all | 795 (70) | **0.48** | 0.017 | 0.49 (249) | 0.49 | -0.00 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | exact h1 | start_early | 36 (16) | **0.53** | 0.102 | 0.55 (11) | 0.55 | -0.03 (193) | 0.148 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | exact h1 | turn | 319 (49) | **0.47** | 0.035 | 0.44 (98) | 0.47 | -0.07 (225) | 0.142 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | exact h1 | start_late | 11 (5) | **0.55** | 0.136 | 0.60 (5) | 0.58 | 0.09 (92) | 0.146 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | exact h1 | north_leg | 410 (48) | **0.50** | 0.024 | 0.47 (129) | 0.49 | 0.05 (299) | 0.255 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | exact h1 | shortcut_early | 19 (9) | **0.47** | 0.142 | 0.50 (6) | 0.45 | 0.04 (129) | 0.136 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | exact h1 | general | 0 (0) | **-** | - | - (0) | 0.52 | 0.03 (2620) | 0.412 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | exact h1 | dense_all | 795 (70) | **0.49** | 0.020 | 0.47 (249) | 0.49 | 0.01 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | start_early | 36 (16) | **0.44** | 0.085 | 0.36 (11) | 0.51 | -0.09 (193) | 0.146 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | turn | 319 (49) | **0.46** | 0.028 | 0.44 (98) | 0.46 | -0.06 (225) | 0.143 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | start_late | 11 (5) | **0.45** | 0.165 | 0.40 (5) | 0.56 | 0.07 (92) | 0.138 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | north_leg | 410 (48) | **0.48** | 0.029 | 0.47 (129) | 0.50 | 0.06 (299) | 0.255 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | shortcut_early | 19 (9) | **0.37** | 0.112 | 0.50 (6) | 0.49 | 0.02 (129) | 0.139 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | general | 0 (0) | **-** | - | - (0) | 0.52 | 0.04 (2620) | 0.412 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | dense_all | 795 (70) | **0.47** | 0.018 | 0.45 (249) | 0.49 | -0.00 (938) | 0.165 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | start_early | 36 (16) | **0.44** | 0.083 | 0.55 (11) | 0.48 | -0.05 (193) | 0.145 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | turn | 319 (49) | **0.47** | 0.029 | 0.46 (98) | 0.50 | -0.01 (225) | 0.146 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | start_late | 11 (5) | **0.55** | 0.036 | 0.20 (5) | 0.55 | -0.02 (92) | 0.143 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | north_leg | 410 (48) | **0.48** | 0.027 | 0.56 (129) | 0.49 | 0.09 (299) | 0.264 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | shortcut_early | 19 (9) | **0.42** | 0.117 | 0.67 (6) | 0.51 | -0.01 (129) | 0.139 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | general | 0 (0) | **-** | - | - (0) | 0.53 | 0.03 (2620) | 0.412 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | dense_all | 795 (70) | **0.47** | 0.017 | 0.51 (249) | 0.50 | 0.02 (938) | 0.167 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | start_early | 36 (16) | **0.47** | 0.101 | 0.36 (11) | 0.52 | -0.11 (193) | 0.146 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | turn | 319 (49) | **0.47** | 0.034 | 0.47 (98) | 0.47 | -0.06 (225) | 0.141 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | start_late | 11 (5) | **0.55** | 0.136 | 0.60 (5) | 0.59 | 0.14 (92) | 0.148 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | north_leg | 410 (48) | **0.49** | 0.023 | 0.47 (129) | 0.48 | 0.06 (299) | 0.256 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | shortcut_early | 19 (9) | **0.47** | 0.142 | 0.50 (6) | 0.45 | 0.06 (129) | 0.137 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | general | 0 (0) | **-** | - | - (0) | 0.52 | 0.04 (2620) | 0.413 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | dense_all | 795 (70) | **0.48** | 0.018 | 0.47 (249) | 0.49 | 0.00 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r1 min | start_early | 36 (16) | **0.44** | 0.090 | 0.55 (11) | 0.50 | -0.08 (193) | 0.144 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r1 min | turn | 319 (49) | **0.48** | 0.029 | 0.48 (98) | 0.48 | -0.03 (225) | 0.146 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r1 min | start_late | 11 (5) | **0.45** | 0.165 | 0.40 (5) | 0.55 | -0.01 (92) | 0.138 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r1 min | north_leg | 410 (48) | **0.48** | 0.026 | 0.50 (129) | 0.50 | 0.06 (299) | 0.261 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r1 min | shortcut_early | 19 (9) | **0.47** | 0.092 | 0.50 (6) | 0.50 | 0.05 (129) | 0.141 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r1 min | general | 0 (0) | **-** | - | - (0) | 0.51 | 0.02 (2620) | 0.409 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r1 min | dense_all | 795 (70) | **0.48** | 0.018 | 0.49 (249) | 0.50 | 0.00 (938) | 0.166 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | start_early | 36 (16) | **0.42** | 0.093 | 0.55 (11) | 0.48 | -0.04 (193) | 0.144 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | turn | 319 (49) | **0.48** | 0.027 | 0.50 (98) | 0.50 | 0.02 (225) | 0.149 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | start_late | 11 (5) | **0.55** | 0.036 | 0.20 (5) | 0.56 | 0.03 (92) | 0.144 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | north_leg | 410 (48) | **0.48** | 0.026 | 0.56 (129) | 0.50 | 0.09 (299) | 0.265 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | shortcut_early | 19 (9) | **0.42** | 0.117 | 0.67 (6) | 0.49 | -0.06 (129) | 0.136 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | general | 0 (0) | **-** | - | - (0) | 0.52 | 0.02 (2620) | 0.410 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | dense_all | 795 (70) | **0.48** | 0.015 | 0.53 (249) | 0.50 | 0.02 (938) | 0.168 / 0.166 / 0.233 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | start_early | 36 (16) | **0.47** | 0.105 | 0.45 (11) | 0.49 | -0.09 (193) | 0.145 / 0.146 / 0.177 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | turn | 319 (49) | **0.48** | 0.032 | 0.47 (98) | 0.46 | -0.05 (225) | 0.145 / 0.147 / 0.256 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | start_late | 11 (5) | **0.55** | 0.136 | 0.60 (5) | 0.58 | 0.14 (92) | 0.148 / 0.140 / 0.167 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | north_leg | 410 (48) | **0.49** | 0.021 | 0.46 (129) | 0.48 | 0.04 (299) | 0.252 / 0.254 / 0.397 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | shortcut_early | 19 (9) | **0.42** | 0.131 | 0.33 (6) | 0.45 | 0.02 (129) | 0.137 / 0.139 / 0.166 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | general | 0 (0) | **-** | - | - (0) | 0.51 | 0.03 (2620) | 0.410 / 0.408 / 0.482 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | dense_all | 795 (70) | **0.48** | 0.018 | 0.46 (249) | 0.48 | 0.00 (938) | 0.165 / 0.166 / 0.233 |

## Pass

- random: fail
- policy_logprob: fail
- critics_r1/seed_0/final | exact min: PASS
- critics_r1/seed_0/final | exact h0: PASS
- critics_r1/seed_0/final | exact h1: PASS
- critics_r1/seed_0/final | region r0.5 min: PASS
- critics_r1/seed_0/final | region r0.5 h0: PASS
- critics_r1/seed_0/final | region r0.5 h1: PASS
- critics_r1/seed_0/final | region r1 min: PASS
- critics_r1/seed_0/final | region r1 h0: PASS
- critics_r1/seed_0/final | region r1 h1: PASS
- critics_r1/seed_1/final | exact min: PASS
- critics_r1/seed_1/final | exact h0: PASS
- critics_r1/seed_1/final | exact h1: PASS
- critics_r1/seed_1/final | region r0.5 min: PASS
- critics_r1/seed_1/final | region r0.5 h0: PASS
- critics_r1/seed_1/final | region r0.5 h1: PASS
- critics_r1/seed_1/final | region r1 min: PASS
- critics_r1/seed_1/final | region r1 h0: PASS
- critics_r1/seed_1/final | region r1 h1: PASS
- critics_r1/seed_2/final | exact min: PASS
- critics_r1/seed_2/final | exact h0: PASS
- critics_r1/seed_2/final | exact h1: PASS
- critics_r1/seed_2/final | region r0.5 min: PASS
- critics_r1/seed_2/final | region r0.5 h0: PASS
- critics_r1/seed_2/final | region r0.5 h1: PASS
- critics_r1/seed_2/final | region r1 min: PASS
- critics_r1/seed_2/final | region r1 h0: PASS
- critics_r1/seed_2/final | region r1 h1: PASS
- vanilla_g0999_30k/seed_0/final | exact min: fail
- vanilla_g0999_30k/seed_0/final | exact h0: fail
- vanilla_g0999_30k/seed_0/final | exact h1: fail
- vanilla_g0999_30k/seed_0/final | region r0.5 min: fail
- vanilla_g0999_30k/seed_0/final | region r0.5 h0: fail
- vanilla_g0999_30k/seed_0/final | region r0.5 h1: fail
- vanilla_g0999_30k/seed_0/final | region r1 min: fail
- vanilla_g0999_30k/seed_0/final | region r1 h0: fail
- vanilla_g0999_30k/seed_0/final | region r1 h1: fail
- vanilla_g0999_30k/seed_1/final | exact min: fail
- vanilla_g0999_30k/seed_1/final | exact h0: fail
- vanilla_g0999_30k/seed_1/final | exact h1: fail
- vanilla_g0999_30k/seed_1/final | region r0.5 min: fail
- vanilla_g0999_30k/seed_1/final | region r0.5 h0: fail
- vanilla_g0999_30k/seed_1/final | region r0.5 h1: fail
- vanilla_g0999_30k/seed_1/final | region r1 min: fail
- vanilla_g0999_30k/seed_1/final | region r1 h0: fail
- vanilla_g0999_30k/seed_1/final | region r1 h1: fail
