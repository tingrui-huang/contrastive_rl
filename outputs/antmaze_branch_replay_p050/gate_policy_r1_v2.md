# Gate (round 1): does f(s, a, g) order the candidate torques the way their validated single-step outcomes do?

Held-out anchors (100 held-out episodes), candidates ['recorded', 'mode', 'sample0', 'sample1', 'sample2'], 4 paired draws per key; the continuation is the frozen policy mode.  A pair (a_i, a_j) at one state is VALIDATED when the log P_goal ratio has the same sign on the two halves of the draws ((0, 1) vs (2, 3)) and |ratio| > 0.3 on both.  Agreement = share of validated pairs whose critic ordering matches (chance 0.50); samples-only = pairs of two policy samples (excludes recorded and mode); weighted = |ratio|-weighted sign agreement over all pairs; pick gain = (P[argmax f] - mean P) / (max P - mean P) over anchors with spread (1 = oracle, 0 = random).  PASS = pooled dense-set validated agreement >= 0.65 and > 0.5 by 2 s.e. (s.e. = episode-level bootstrap: the pairs of one anchor share its candidates and draws, the anchors of one episode its poses and goal).  Readouts per critic: exact = the logit at the recorded goal point (min over the twin heads = what the actor optimises; h0 / h1 = the heads); region rX = log of the region-integrated exp(f) over the NCE goal marginal within radius X of the goal (the quantity P_goal measures; diagnostic only, the actor does not optimise it).

## Ceiling: split-half sign agreement of the differences themselves

| set | episodes | anchors | pairs | mean abs log ratio | frac > thr | first-half > thr | tie | weak | same | opposite | agreement among decided (s.e.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| start_early | 44 | 60 | 600 | 0.88 | 0.22 | 77 | 48 | 1 | 28 | 0 | 1.00 (0.000) |
| turn | 5 | 60 | 600 | 2.13 | 0.67 | 404 | 28 | 35 | 287 | 54 | 0.84 (0.025) |
| start_late | 48 | 60 | 600 | 0.80 | 0.24 | 94 | 55 | 0 | 39 | 0 | 1.00 (0.000) |
| north_leg | 5 | 60 | 600 | 2.06 | 0.74 | 423 | 1 | 54 | 314 | 54 | 0.85 (0.029) |
| shortcut_early | 46 | 60 | 600 | 0.81 | 0.21 | 61 | 45 | 0 | 16 | 0 | 1.00 (0.000) |
| general | 45 | 60 | 600 | 1.19 | 0.32 | 166 | 15 | 5 | 143 | 3 | 0.98 (0.019) |

## Critics

| scorer | set | validated pairs (episodes) | agreement | s.e. | samples-only agreement (n) | weighted agreement | pick gain (anchors) | P pick / mean / best |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | start_early | 28 (5) | **0.46** | 0.097 | 0.45 (11) | 0.50 | 0.20 (42) | 0.140 / 0.136 / 0.172 |
| random | turn | 287 (5) | **0.48** | 0.030 | 0.52 (97) | 0.52 | -0.00 (57) | 0.149 / 0.162 / 0.338 |
| random | start_late | 39 (7) | **0.49** | 0.131 | 0.58 (12) | 0.41 | -0.23 (40) | 0.096 / 0.104 / 0.133 |
| random | north_leg | 314 (5) | **0.32** | 0.020 | 0.32 (98) | 0.32 | -0.40 (60) | 0.180 / 0.235 / 0.416 |
| random | shortcut_early | 16 (4) | **0.38** | 0.188 | 0.50 (4) | 0.58 | 0.28 (41) | 0.150 / 0.139 / 0.168 |
| random | general | 143 (22) | **0.41** | 0.051 | 0.49 (41) | 0.44 | -0.06 (55) | 0.464 / 0.469 / 0.594 |
| random | dense_all | 684 (19) | **0.41** | 0.015 | 0.43 (222) | 0.45 | -0.06 (240) | 0.143 / 0.155 / 0.245 |
| policy_logprob | start_early | 28 (5) | **0.89** | 0.088 | 1.00 (11) | 0.48 | -0.28 (42) | 0.133 / 0.136 / 0.172 |
| policy_logprob | turn | 287 (5) | **0.52** | 0.019 | 0.54 (97) | 0.52 | -0.01 (57) | 0.154 / 0.162 / 0.338 |
| policy_logprob | start_late | 39 (7) | **0.56** | 0.104 | 0.67 (12) | 0.50 | 0.09 (40) | 0.102 / 0.104 / 0.133 |
| policy_logprob | north_leg | 314 (5) | **0.47** | 0.025 | 0.47 (98) | 0.48 | -0.11 (60) | 0.225 / 0.235 / 0.416 |
| policy_logprob | shortcut_early | 16 (4) | **0.50** | 0.171 | 0.25 (4) | 0.49 | 0.01 (41) | 0.135 / 0.139 / 0.168 |
| policy_logprob | general | 143 (22) | **0.44** | 0.057 | 0.51 (41) | 0.47 | -0.34 (55) | 0.452 / 0.469 / 0.594 |
| policy_logprob | dense_all | 684 (19) | **0.52** | 0.022 | 0.53 (222) | 0.50 | -0.06 (240) | 0.150 / 0.155 / 0.245 |
| knn_fit_w1 | start_early | 28 (5) | **0.46** | 0.228 | 0.45 (11) | 0.33 | -0.02 (42) | 0.132 / 0.136 / 0.172 |
| knn_fit_w1 | turn | 287 (5) | **0.40** | 0.028 | 0.40 (97) | 0.38 | 0.01 (57) | 0.153 / 0.162 / 0.338 |
| knn_fit_w1 | start_late | 39 (7) | **0.18** | 0.081 | 0.25 (12) | 0.22 | 0.22 (40) | 0.111 / 0.104 / 0.133 |
| knn_fit_w1 | north_leg | 314 (5) | **0.48** | 0.041 | 0.52 (98) | 0.51 | 0.01 (60) | 0.232 / 0.235 / 0.416 |
| knn_fit_w1 | shortcut_early | 16 (4) | **0.44** | 0.227 | 1.00 (4) | 0.25 | -0.27 (41) | 0.127 / 0.139 / 0.168 |
| knn_fit_w1 | general | 143 (22) | **0.22** | 0.064 | 0.22 (41) | 0.23 | 0.38 (55) | 0.520 / 0.469 / 0.594 |
| knn_fit_w1 | dense_all | 684 (19) | **0.43** | 0.034 | 0.46 (222) | 0.38 | -0.01 (240) | 0.151 / 0.155 / 0.245 |
| knn_fit_w4 | start_early | 28 (5) | **0.61** | 0.110 | 0.64 (11) | 0.43 | 0.08 (42) | 0.135 / 0.136 / 0.172 |
| knn_fit_w4 | turn | 287 (5) | **0.44** | 0.027 | 0.47 (97) | 0.44 | -0.04 (57) | 0.161 / 0.162 / 0.338 |
| knn_fit_w4 | start_late | 39 (7) | **0.28** | 0.081 | 0.42 (12) | 0.27 | -0.03 (40) | 0.099 / 0.104 / 0.133 |
| knn_fit_w4 | north_leg | 314 (5) | **0.48** | 0.032 | 0.42 (98) | 0.49 | 0.07 (60) | 0.249 / 0.235 / 0.416 |
| knn_fit_w4 | shortcut_early | 16 (4) | **0.38** | 0.142 | 0.00 (4) | 0.37 | -0.08 (41) | 0.147 / 0.139 / 0.168 |
| knn_fit_w4 | general | 143 (22) | **0.39** | 0.054 | 0.46 (41) | 0.40 | 0.40 (55) | 0.477 / 0.469 / 0.594 |
| knn_fit_w4 | dense_all | 684 (19) | **0.45** | 0.025 | 0.45 (222) | 0.43 | 0.00 (240) | 0.158 / 0.155 / 0.245 |
| critics_r1/seed_0/final | exact min | start_early | 28 (5) | **0.61** | 0.057 | 0.64 (11) | 0.49 | -0.12 (42) | 0.134 / 0.136 / 0.172 |
| critics_r1/seed_0/final | exact min | turn | 287 (5) | **0.47** | 0.049 | 0.43 (97) | 0.49 | -0.02 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_0/final | exact min | start_late | 39 (7) | **0.59** | 0.087 | 0.67 (12) | 0.50 | 0.14 (40) | 0.108 / 0.104 / 0.133 |
| critics_r1/seed_0/final | exact min | north_leg | 314 (5) | **0.52** | 0.021 | 0.53 (98) | 0.51 | 0.06 (60) | 0.247 / 0.235 / 0.416 |
| critics_r1/seed_0/final | exact min | shortcut_early | 16 (4) | **0.75** | 0.088 | 0.75 (4) | 0.55 | 0.11 (41) | 0.155 / 0.139 / 0.168 |
| critics_r1/seed_0/final | exact min | general | 143 (22) | **0.49** | 0.064 | 0.37 (41) | 0.51 | 0.09 (55) | 0.445 / 0.469 / 0.594 |
| critics_r1/seed_0/final | exact min | dense_all | 684 (19) | **0.51** | 0.020 | 0.50 (222) | 0.50 | 0.03 (240) | 0.162 / 0.155 / 0.245 |
| critics_r1/seed_0/final | exact h0 | start_early | 28 (5) | **0.43** | 0.102 | 0.45 (11) | 0.44 | -0.30 (42) | 0.123 / 0.136 / 0.172 |
| critics_r1/seed_0/final | exact h0 | turn | 287 (5) | **0.43** | 0.018 | 0.38 (97) | 0.49 | -0.07 (57) | 0.159 / 0.162 / 0.338 |
| critics_r1/seed_0/final | exact h0 | start_late | 39 (7) | **0.56** | 0.091 | 0.67 (12) | 0.56 | 0.10 (40) | 0.106 / 0.104 / 0.133 |
| critics_r1/seed_0/final | exact h0 | north_leg | 314 (5) | **0.49** | 0.034 | 0.48 (98) | 0.49 | -0.11 (60) | 0.230 / 0.235 / 0.416 |
| critics_r1/seed_0/final | exact h0 | shortcut_early | 16 (4) | **0.69** | 0.054 | 0.50 (4) | 0.58 | 0.15 (41) | 0.153 / 0.139 / 0.168 |
| critics_r1/seed_0/final | exact h0 | general | 143 (22) | **0.49** | 0.052 | 0.46 (41) | 0.48 | 0.32 (55) | 0.472 / 0.469 / 0.594 |
| critics_r1/seed_0/final | exact h0 | dense_all | 684 (19) | **0.47** | 0.017 | 0.45 (222) | 0.50 | -0.05 (240) | 0.154 / 0.155 / 0.245 |
| critics_r1/seed_0/final | exact h1 | start_early | 28 (5) | **0.50** | 0.100 | 0.45 (11) | 0.39 | -0.31 (42) | 0.121 / 0.136 / 0.172 |
| critics_r1/seed_0/final | exact h1 | turn | 287 (5) | **0.45** | 0.053 | 0.44 (97) | 0.46 | -0.19 (57) | 0.137 / 0.162 / 0.338 |
| critics_r1/seed_0/final | exact h1 | start_late | 39 (7) | **0.56** | 0.103 | 0.58 (12) | 0.48 | -0.12 (40) | 0.103 / 0.104 / 0.133 |
| critics_r1/seed_0/final | exact h1 | north_leg | 314 (5) | **0.53** | 0.018 | 0.48 (98) | 0.53 | 0.09 (60) | 0.250 / 0.235 / 0.416 |
| critics_r1/seed_0/final | exact h1 | shortcut_early | 16 (4) | **0.62** | 0.187 | 0.50 (4) | 0.49 | -0.19 (41) | 0.143 / 0.139 / 0.168 |
| critics_r1/seed_0/final | exact h1 | general | 143 (22) | **0.50** | 0.063 | 0.39 (41) | 0.52 | 0.25 (55) | 0.482 / 0.469 / 0.594 |
| critics_r1/seed_0/final | exact h1 | dense_all | 684 (19) | **0.50** | 0.026 | 0.47 (222) | 0.48 | -0.13 (240) | 0.151 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r0.5 min | start_early | 28 (5) | **0.54** | 0.095 | 0.64 (11) | 0.50 | -0.31 (42) | 0.129 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r0.5 min | turn | 287 (5) | **0.46** | 0.049 | 0.43 (97) | 0.46 | 0.02 (57) | 0.174 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r0.5 min | start_late | 39 (7) | **0.64** | 0.037 | 0.67 (12) | 0.46 | 0.09 (40) | 0.106 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r0.5 min | north_leg | 314 (5) | **0.49** | 0.043 | 0.50 (98) | 0.48 | 0.04 (60) | 0.239 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r0.5 min | shortcut_early | 16 (4) | **0.50** | 0.195 | 0.75 (4) | 0.50 | -0.24 (41) | 0.141 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r0.5 min | general | 143 (22) | **0.52** | 0.054 | 0.49 (41) | 0.54 | 0.42 (55) | 0.499 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r0.5 min | dense_all | 684 (19) | **0.49** | 0.021 | 0.49 (222) | 0.48 | -0.07 (240) | 0.158 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r0.5 h0 | start_early | 28 (5) | **0.43** | 0.089 | 0.45 (11) | 0.45 | -0.44 (42) | 0.121 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r0.5 h0 | turn | 287 (5) | **0.46** | 0.030 | 0.40 (97) | 0.50 | 0.01 (57) | 0.173 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r0.5 h0 | start_late | 39 (7) | **0.64** | 0.104 | 0.75 (12) | 0.54 | 0.13 (40) | 0.107 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r0.5 h0 | north_leg | 314 (5) | **0.48** | 0.060 | 0.44 (98) | 0.47 | -0.07 (60) | 0.232 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r0.5 h0 | shortcut_early | 16 (4) | **0.56** | 0.182 | 0.75 (4) | 0.56 | -0.01 (41) | 0.142 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r0.5 h0 | general | 143 (22) | **0.48** | 0.046 | 0.44 (41) | 0.46 | 0.18 (55) | 0.472 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r0.5 h0 | dense_all | 684 (19) | **0.48** | 0.026 | 0.45 (222) | 0.49 | -0.07 (240) | 0.155 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r0.5 h1 | start_early | 28 (5) | **0.46** | 0.111 | 0.55 (11) | 0.41 | -0.14 (42) | 0.121 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r0.5 h1 | turn | 287 (5) | **0.45** | 0.028 | 0.45 (97) | 0.47 | -0.20 (57) | 0.134 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r0.5 h1 | start_late | 39 (7) | **0.51** | 0.082 | 0.50 (12) | 0.45 | -0.32 (40) | 0.098 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r0.5 h1 | north_leg | 314 (5) | **0.52** | 0.043 | 0.49 (98) | 0.52 | 0.04 (60) | 0.242 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r0.5 h1 | shortcut_early | 16 (4) | **0.44** | 0.221 | 0.50 (4) | 0.46 | -0.44 (41) | 0.136 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r0.5 h1 | general | 143 (22) | **0.50** | 0.053 | 0.46 (41) | 0.53 | 0.22 (55) | 0.484 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r0.5 h1 | dense_all | 684 (19) | **0.49** | 0.024 | 0.48 (222) | 0.48 | -0.19 (240) | 0.146 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r1 min | start_early | 28 (5) | **0.50** | 0.117 | 0.64 (11) | 0.52 | -0.32 (42) | 0.129 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r1 min | turn | 287 (5) | **0.44** | 0.034 | 0.44 (97) | 0.46 | -0.00 (57) | 0.174 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r1 min | start_late | 39 (7) | **0.59** | 0.053 | 0.58 (12) | 0.46 | 0.07 (40) | 0.106 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r1 min | north_leg | 314 (5) | **0.50** | 0.034 | 0.53 (98) | 0.47 | 0.06 (60) | 0.246 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r1 min | shortcut_early | 16 (4) | **0.44** | 0.184 | 0.75 (4) | 0.50 | -0.20 (41) | 0.142 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r1 min | general | 143 (22) | **0.55** | 0.051 | 0.56 (41) | 0.56 | 0.42 (55) | 0.511 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r1 min | dense_all | 684 (19) | **0.48** | 0.019 | 0.50 (222) | 0.47 | -0.06 (240) | 0.159 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r1 h0 | start_early | 28 (5) | **0.39** | 0.105 | 0.45 (11) | 0.49 | -0.40 (42) | 0.121 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r1 h0 | turn | 287 (5) | **0.49** | 0.021 | 0.47 (97) | 0.53 | -0.01 (57) | 0.167 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r1 h0 | start_late | 39 (7) | **0.67** | 0.126 | 0.75 (12) | 0.52 | -0.00 (40) | 0.104 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r1 h0 | north_leg | 314 (5) | **0.49** | 0.052 | 0.48 (98) | 0.46 | 0.03 (60) | 0.248 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r1 h0 | shortcut_early | 16 (4) | **0.50** | 0.176 | 0.75 (4) | 0.58 | 0.01 (41) | 0.140 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r1 h0 | general | 143 (22) | **0.54** | 0.051 | 0.56 (41) | 0.51 | 0.28 (55) | 0.497 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r1 h0 | dense_all | 684 (19) | **0.50** | 0.027 | 0.50 (222) | 0.51 | -0.07 (240) | 0.156 / 0.155 / 0.245 |
| critics_r1/seed_0/final | region r1 h1 | start_early | 28 (5) | **0.46** | 0.111 | 0.55 (11) | 0.42 | -0.08 (42) | 0.128 / 0.136 / 0.172 |
| critics_r1/seed_0/final | region r1 h1 | turn | 287 (5) | **0.40** | 0.033 | 0.41 (97) | 0.43 | -0.12 (57) | 0.145 / 0.162 / 0.338 |
| critics_r1/seed_0/final | region r1 h1 | start_late | 39 (7) | **0.49** | 0.088 | 0.42 (12) | 0.45 | -0.21 (40) | 0.100 / 0.104 / 0.133 |
| critics_r1/seed_0/final | region r1 h1 | north_leg | 314 (5) | **0.53** | 0.028 | 0.48 (98) | 0.52 | 0.08 (60) | 0.253 / 0.235 / 0.416 |
| critics_r1/seed_0/final | region r1 h1 | shortcut_early | 16 (4) | **0.38** | 0.207 | 0.50 (4) | 0.48 | -0.38 (41) | 0.139 / 0.139 / 0.168 |
| critics_r1/seed_0/final | region r1 h1 | general | 143 (22) | **0.51** | 0.057 | 0.46 (41) | 0.52 | 0.16 (55) | 0.483 / 0.469 / 0.594 |
| critics_r1/seed_0/final | region r1 h1 | dense_all | 684 (19) | **0.47** | 0.021 | 0.45 (222) | 0.47 | -0.12 (240) | 0.153 / 0.155 / 0.245 |
| critics_r1/seed_1/final | exact min | start_early | 28 (5) | **0.61** | 0.143 | 0.64 (11) | 0.44 | -0.30 (42) | 0.137 / 0.136 / 0.172 |
| critics_r1/seed_1/final | exact min | turn | 287 (5) | **0.47** | 0.012 | 0.52 (97) | 0.52 | -0.01 (57) | 0.165 / 0.162 / 0.338 |
| critics_r1/seed_1/final | exact min | start_late | 39 (7) | **0.51** | 0.061 | 0.58 (12) | 0.59 | 0.42 (40) | 0.114 / 0.104 / 0.133 |
| critics_r1/seed_1/final | exact min | north_leg | 314 (5) | **0.54** | 0.027 | 0.52 (98) | 0.52 | -0.09 (60) | 0.230 / 0.235 / 0.416 |
| critics_r1/seed_1/final | exact min | shortcut_early | 16 (4) | **0.62** | 0.142 | 0.25 (4) | 0.48 | 0.14 (41) | 0.147 / 0.139 / 0.168 |
| critics_r1/seed_1/final | exact min | general | 143 (22) | **0.48** | 0.057 | 0.51 (41) | 0.51 | -0.05 (55) | 0.483 / 0.469 / 0.594 |
| critics_r1/seed_1/final | exact min | dense_all | 684 (19) | **0.52** | 0.015 | 0.52 (222) | 0.51 | 0.02 (240) | 0.159 / 0.155 / 0.245 |
| critics_r1/seed_1/final | exact h0 | start_early | 28 (5) | **0.46** | 0.174 | 0.55 (11) | 0.47 | -0.21 (42) | 0.123 / 0.136 / 0.172 |
| critics_r1/seed_1/final | exact h0 | turn | 287 (5) | **0.43** | 0.024 | 0.45 (97) | 0.54 | -0.10 (57) | 0.150 / 0.162 / 0.338 |
| critics_r1/seed_1/final | exact h0 | start_late | 39 (7) | **0.51** | 0.049 | 0.50 (12) | 0.61 | 0.28 (40) | 0.111 / 0.104 / 0.133 |
| critics_r1/seed_1/final | exact h0 | north_leg | 314 (5) | **0.54** | 0.018 | 0.52 (98) | 0.51 | 0.04 (60) | 0.241 / 0.235 / 0.416 |
| critics_r1/seed_1/final | exact h0 | shortcut_early | 16 (4) | **0.31** | 0.100 | 0.25 (4) | 0.47 | -0.10 (41) | 0.139 / 0.139 / 0.168 |
| critics_r1/seed_1/final | exact h0 | general | 143 (22) | **0.49** | 0.052 | 0.59 (41) | 0.49 | 0.04 (55) | 0.463 / 0.469 / 0.594 |
| critics_r1/seed_1/final | exact h0 | dense_all | 684 (19) | **0.49** | 0.021 | 0.49 (222) | 0.52 | -0.02 (240) | 0.153 / 0.155 / 0.245 |
| critics_r1/seed_1/final | exact h1 | start_early | 28 (5) | **0.43** | 0.181 | 0.45 (11) | 0.38 | -0.36 (42) | 0.131 / 0.136 / 0.172 |
| critics_r1/seed_1/final | exact h1 | turn | 287 (5) | **0.48** | 0.025 | 0.54 (97) | 0.50 | -0.01 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_1/final | exact h1 | start_late | 39 (7) | **0.49** | 0.054 | 0.58 (12) | 0.57 | 0.26 (40) | 0.111 / 0.104 / 0.133 |
| critics_r1/seed_1/final | exact h1 | north_leg | 314 (5) | **0.51** | 0.014 | 0.48 (98) | 0.50 | -0.01 (60) | 0.234 / 0.235 / 0.416 |
| critics_r1/seed_1/final | exact h1 | shortcut_early | 16 (4) | **0.75** | 0.089 | 0.50 (4) | 0.47 | 0.01 (41) | 0.145 / 0.139 / 0.168 |
| critics_r1/seed_1/final | exact h1 | general | 143 (22) | **0.48** | 0.068 | 0.41 (41) | 0.50 | -0.12 (55) | 0.473 / 0.469 / 0.594 |
| critics_r1/seed_1/final | exact h1 | dense_all | 684 (19) | **0.50** | 0.014 | 0.51 (222) | 0.49 | -0.02 (240) | 0.157 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r0.5 min | start_early | 28 (5) | **0.54** | 0.167 | 0.55 (11) | 0.41 | -0.28 (42) | 0.132 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r0.5 min | turn | 287 (5) | **0.44** | 0.015 | 0.45 (97) | 0.50 | -0.17 (57) | 0.141 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r0.5 min | start_late | 39 (7) | **0.62** | 0.073 | 0.58 (12) | 0.62 | 0.41 (40) | 0.116 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r0.5 min | north_leg | 314 (5) | **0.51** | 0.044 | 0.49 (98) | 0.50 | -0.14 (60) | 0.224 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r0.5 min | shortcut_early | 16 (4) | **0.31** | 0.100 | 0.25 (4) | 0.43 | -0.18 (41) | 0.136 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r0.5 min | general | 143 (22) | **0.55** | 0.057 | 0.51 (41) | 0.56 | 0.08 (55) | 0.496 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r0.5 min | dense_all | 684 (19) | **0.48** | 0.019 | 0.48 (222) | 0.49 | -0.09 (240) | 0.150 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r0.5 h0 | start_early | 28 (5) | **0.50** | 0.153 | 0.55 (11) | 0.44 | -0.26 (42) | 0.123 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r0.5 h0 | turn | 287 (5) | **0.44** | 0.023 | 0.43 (97) | 0.53 | -0.14 (57) | 0.148 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r0.5 h0 | start_late | 39 (7) | **0.51** | 0.068 | 0.42 (12) | 0.62 | -0.00 (40) | 0.108 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r0.5 h0 | north_leg | 314 (5) | **0.52** | 0.023 | 0.49 (98) | 0.47 | 0.00 (60) | 0.234 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r0.5 h0 | shortcut_early | 16 (4) | **0.31** | 0.100 | 0.25 (4) | 0.49 | -0.12 (41) | 0.139 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r0.5 h0 | general | 143 (22) | **0.48** | 0.050 | 0.54 (41) | 0.51 | 0.06 (55) | 0.471 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r0.5 h0 | dense_all | 684 (19) | **0.48** | 0.022 | 0.46 (222) | 0.50 | -0.10 (240) | 0.150 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r0.5 h1 | start_early | 28 (5) | **0.32** | 0.170 | 0.36 (11) | 0.32 | -0.65 (42) | 0.111 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r0.5 h1 | turn | 287 (5) | **0.45** | 0.034 | 0.52 (97) | 0.49 | 0.02 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r0.5 h1 | start_late | 39 (7) | **0.62** | 0.084 | 0.67 (12) | 0.61 | 0.16 (40) | 0.110 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r0.5 h1 | north_leg | 314 (5) | **0.49** | 0.026 | 0.47 (98) | 0.51 | 0.09 (60) | 0.258 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r0.5 h1 | shortcut_early | 16 (4) | **0.44** | 0.135 | 0.50 (4) | 0.45 | -0.05 (41) | 0.139 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r0.5 h1 | general | 143 (22) | **0.48** | 0.067 | 0.41 (41) | 0.50 | -0.14 (55) | 0.482 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r0.5 h1 | dense_all | 684 (19) | **0.48** | 0.023 | 0.50 (222) | 0.48 | -0.07 (240) | 0.157 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r1 min | start_early | 28 (5) | **0.50** | 0.153 | 0.55 (11) | 0.46 | -0.23 (42) | 0.126 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r1 min | turn | 287 (5) | **0.43** | 0.012 | 0.44 (97) | 0.50 | -0.16 (57) | 0.135 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r1 min | start_late | 39 (7) | **0.62** | 0.102 | 0.58 (12) | 0.63 | 0.47 (40) | 0.120 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r1 min | north_leg | 314 (5) | **0.50** | 0.037 | 0.48 (98) | 0.49 | -0.05 (60) | 0.226 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r1 min | shortcut_early | 16 (4) | **0.31** | 0.100 | 0.25 (4) | 0.40 | -0.26 (41) | 0.129 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r1 min | general | 143 (22) | **0.52** | 0.055 | 0.51 (41) | 0.55 | 0.12 (55) | 0.457 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r1 min | dense_all | 684 (19) | **0.47** | 0.019 | 0.47 (222) | 0.50 | -0.06 (240) | 0.147 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r1 h0 | start_early | 28 (5) | **0.54** | 0.135 | 0.55 (11) | 0.46 | -0.21 (42) | 0.123 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r1 h0 | turn | 287 (5) | **0.42** | 0.022 | 0.41 (97) | 0.51 | -0.10 (57) | 0.148 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r1 h0 | start_late | 39 (7) | **0.46** | 0.068 | 0.33 (12) | 0.60 | -0.00 (40) | 0.109 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r1 h0 | north_leg | 314 (5) | **0.51** | 0.032 | 0.48 (98) | 0.48 | 0.02 (60) | 0.240 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r1 h0 | shortcut_early | 16 (4) | **0.31** | 0.100 | 0.25 (4) | 0.47 | -0.30 (41) | 0.134 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r1 h0 | general | 143 (22) | **0.48** | 0.044 | 0.51 (41) | 0.49 | 0.04 (55) | 0.467 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r1 h0 | dense_all | 684 (19) | **0.46** | 0.023 | 0.44 (222) | 0.50 | -0.11 (240) | 0.151 / 0.155 / 0.245 |
| critics_r1/seed_1/final | region r1 h1 | start_early | 28 (5) | **0.29** | 0.145 | 0.36 (11) | 0.36 | -0.44 (42) | 0.110 / 0.136 / 0.172 |
| critics_r1/seed_1/final | region r1 h1 | turn | 287 (5) | **0.44** | 0.039 | 0.45 (97) | 0.49 | -0.08 (57) | 0.144 / 0.162 / 0.338 |
| critics_r1/seed_1/final | region r1 h1 | start_late | 39 (7) | **0.64** | 0.098 | 0.75 (12) | 0.58 | 0.10 (40) | 0.109 / 0.104 / 0.133 |
| critics_r1/seed_1/final | region r1 h1 | north_leg | 314 (5) | **0.49** | 0.019 | 0.50 (98) | 0.49 | 0.06 (60) | 0.239 / 0.235 / 0.416 |
| critics_r1/seed_1/final | region r1 h1 | shortcut_early | 16 (4) | **0.44** | 0.135 | 0.50 (4) | 0.46 | -0.00 (41) | 0.135 / 0.139 / 0.168 |
| critics_r1/seed_1/final | region r1 h1 | general | 143 (22) | **0.43** | 0.063 | 0.44 (41) | 0.46 | -0.20 (55) | 0.450 / 0.469 / 0.594 |
| critics_r1/seed_1/final | region r1 h1 | dense_all | 684 (19) | **0.46** | 0.025 | 0.49 (222) | 0.48 | -0.06 (240) | 0.148 / 0.155 / 0.245 |
| critics_r1/seed_2/final | exact min | start_early | 28 (5) | **0.36** | 0.162 | 0.27 (11) | 0.33 | -0.77 (42) | 0.112 / 0.136 / 0.172 |
| critics_r1/seed_2/final | exact min | turn | 287 (5) | **0.46** | 0.022 | 0.42 (97) | 0.49 | 0.07 (57) | 0.179 / 0.162 / 0.338 |
| critics_r1/seed_2/final | exact min | start_late | 39 (7) | **0.56** | 0.081 | 0.58 (12) | 0.48 | 0.09 (40) | 0.103 / 0.104 / 0.133 |
| critics_r1/seed_2/final | exact min | north_leg | 314 (5) | **0.54** | 0.023 | 0.49 (98) | 0.54 | 0.05 (60) | 0.262 / 0.235 / 0.416 |
| critics_r1/seed_2/final | exact min | shortcut_early | 16 (4) | **0.50** | 0.176 | 0.50 (4) | 0.50 | -0.03 (41) | 0.141 / 0.139 / 0.168 |
| critics_r1/seed_2/final | exact min | general | 143 (22) | **0.47** | 0.060 | 0.41 (41) | 0.44 | 0.12 (55) | 0.460 / 0.469 / 0.594 |
| critics_r1/seed_2/final | exact min | dense_all | 684 (19) | **0.50** | 0.017 | 0.45 (222) | 0.48 | -0.10 (240) | 0.159 / 0.155 / 0.245 |
| critics_r1/seed_2/final | exact h0 | start_early | 28 (5) | **0.54** | 0.164 | 0.45 (11) | 0.36 | -0.44 (42) | 0.123 / 0.136 / 0.172 |
| critics_r1/seed_2/final | exact h0 | turn | 287 (5) | **0.44** | 0.025 | 0.44 (97) | 0.47 | 0.01 (57) | 0.164 / 0.162 / 0.338 |
| critics_r1/seed_2/final | exact h0 | start_late | 39 (7) | **0.51** | 0.147 | 0.50 (12) | 0.48 | -0.09 (40) | 0.098 / 0.104 / 0.133 |
| critics_r1/seed_2/final | exact h0 | north_leg | 314 (5) | **0.50** | 0.021 | 0.49 (98) | 0.49 | -0.01 (60) | 0.251 / 0.235 / 0.416 |
| critics_r1/seed_2/final | exact h0 | shortcut_early | 16 (4) | **0.44** | 0.133 | 0.50 (4) | 0.50 | 0.05 (41) | 0.144 / 0.139 / 0.168 |
| critics_r1/seed_2/final | exact h0 | general | 143 (22) | **0.42** | 0.067 | 0.39 (41) | 0.44 | 0.03 (55) | 0.470 / 0.469 / 0.594 |
| critics_r1/seed_2/final | exact h0 | dense_all | 684 (19) | **0.48** | 0.019 | 0.47 (222) | 0.47 | -0.08 (240) | 0.156 / 0.155 / 0.245 |
| critics_r1/seed_2/final | exact h1 | start_early | 28 (5) | **0.36** | 0.133 | 0.36 (11) | 0.42 | -0.24 (42) | 0.126 / 0.136 / 0.172 |
| critics_r1/seed_2/final | exact h1 | turn | 287 (5) | **0.48** | 0.024 | 0.46 (97) | 0.53 | 0.08 (57) | 0.179 / 0.162 / 0.338 |
| critics_r1/seed_2/final | exact h1 | start_late | 39 (7) | **0.49** | 0.056 | 0.50 (12) | 0.51 | 0.09 (40) | 0.106 / 0.104 / 0.133 |
| critics_r1/seed_2/final | exact h1 | north_leg | 314 (5) | **0.56** | 0.032 | 0.53 (98) | 0.55 | -0.01 (60) | 0.252 / 0.235 / 0.416 |
| critics_r1/seed_2/final | exact h1 | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.50 (4) | 0.52 | 0.04 (41) | 0.144 / 0.139 / 0.168 |
| critics_r1/seed_2/final | exact h1 | general | 143 (22) | **0.52** | 0.048 | 0.51 (41) | 0.50 | 0.17 (55) | 0.469 / 0.469 / 0.594 |
| critics_r1/seed_2/final | exact h1 | dense_all | 684 (19) | **0.51** | 0.023 | 0.49 (222) | 0.52 | -0.00 (240) | 0.161 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r0.5 min | start_early | 28 (5) | **0.29** | 0.123 | 0.27 (11) | 0.36 | -0.57 (42) | 0.113 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r0.5 min | turn | 287 (5) | **0.47** | 0.032 | 0.47 (97) | 0.53 | -0.01 (57) | 0.160 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r0.5 min | start_late | 39 (7) | **0.49** | 0.080 | 0.67 (12) | 0.43 | -0.24 (40) | 0.093 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r0.5 min | north_leg | 314 (5) | **0.52** | 0.039 | 0.55 (98) | 0.54 | -0.03 (60) | 0.242 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r0.5 min | shortcut_early | 16 (4) | **0.56** | 0.185 | 0.75 (4) | 0.57 | -0.08 (41) | 0.142 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r0.5 min | general | 143 (22) | **0.52** | 0.064 | 0.46 (41) | 0.54 | -0.07 (55) | 0.439 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r0.5 min | dense_all | 684 (19) | **0.49** | 0.021 | 0.51 (222) | 0.50 | -0.16 (240) | 0.150 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r0.5 h0 | start_early | 28 (5) | **0.43** | 0.167 | 0.45 (11) | 0.38 | -0.31 (42) | 0.119 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r0.5 h0 | turn | 287 (5) | **0.47** | 0.019 | 0.44 (97) | 0.50 | -0.09 (57) | 0.152 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r0.5 h0 | start_late | 39 (7) | **0.44** | 0.138 | 0.50 (12) | 0.40 | -0.25 (40) | 0.087 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r0.5 h0 | north_leg | 314 (5) | **0.55** | 0.028 | 0.59 (98) | 0.56 | 0.11 (60) | 0.261 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r0.5 h0 | shortcut_early | 16 (4) | **0.50** | 0.087 | 0.50 (4) | 0.55 | 0.23 (41) | 0.153 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r0.5 h0 | general | 143 (22) | **0.48** | 0.063 | 0.39 (41) | 0.50 | -0.17 (55) | 0.453 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r0.5 h0 | dense_all | 684 (19) | **0.50** | 0.022 | 0.51 (222) | 0.50 | -0.05 (240) | 0.154 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r0.5 h1 | start_early | 28 (5) | **0.32** | 0.106 | 0.27 (11) | 0.43 | -0.30 (42) | 0.118 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r0.5 h1 | turn | 287 (5) | **0.49** | 0.029 | 0.52 (97) | 0.53 | 0.02 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r0.5 h1 | start_late | 39 (7) | **0.54** | 0.089 | 0.58 (12) | 0.52 | -0.03 (40) | 0.103 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r0.5 h1 | north_leg | 314 (5) | **0.51** | 0.030 | 0.50 (98) | 0.53 | -0.13 (60) | 0.226 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r0.5 h1 | shortcut_early | 16 (4) | **0.50** | 0.246 | 0.50 (4) | 0.54 | -0.14 (41) | 0.137 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r0.5 h1 | general | 143 (22) | **0.52** | 0.058 | 0.51 (41) | 0.53 | 0.19 (55) | 0.476 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r0.5 h1 | dense_all | 684 (19) | **0.49** | 0.017 | 0.50 (222) | 0.51 | -0.11 (240) | 0.150 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r1 min | start_early | 28 (5) | **0.36** | 0.135 | 0.36 (11) | 0.42 | -0.35 (42) | 0.124 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r1 min | turn | 287 (5) | **0.47** | 0.032 | 0.47 (97) | 0.51 | -0.05 (57) | 0.153 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r1 min | start_late | 39 (7) | **0.44** | 0.105 | 0.58 (12) | 0.42 | -0.30 (40) | 0.090 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r1 min | north_leg | 314 (5) | **0.49** | 0.052 | 0.50 (98) | 0.48 | 0.10 (60) | 0.261 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r1 min | shortcut_early | 16 (4) | **0.50** | 0.176 | 0.50 (4) | 0.58 | 0.05 (41) | 0.145 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r1 min | general | 143 (22) | **0.55** | 0.057 | 0.59 (41) | 0.57 | 0.11 (55) | 0.451 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r1 min | dense_all | 684 (19) | **0.48** | 0.027 | 0.49 (222) | 0.49 | -0.09 (240) | 0.155 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r1 h0 | start_early | 28 (5) | **0.43** | 0.167 | 0.45 (11) | 0.41 | -0.14 (42) | 0.121 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r1 h0 | turn | 287 (5) | **0.46** | 0.028 | 0.40 (97) | 0.50 | -0.10 (57) | 0.146 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r1 h0 | start_late | 39 (7) | **0.38** | 0.139 | 0.42 (12) | 0.42 | -0.40 (40) | 0.083 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r1 h0 | north_leg | 314 (5) | **0.55** | 0.041 | 0.57 (98) | 0.54 | 0.15 (60) | 0.266 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r1 h0 | shortcut_early | 16 (4) | **0.50** | 0.087 | 0.50 (4) | 0.54 | 0.28 (41) | 0.153 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r1 h0 | general | 143 (22) | **0.54** | 0.051 | 0.63 (41) | 0.56 | 0.24 (55) | 0.503 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r1 h0 | dense_all | 684 (19) | **0.50** | 0.033 | 0.48 (222) | 0.50 | -0.03 (240) | 0.154 / 0.155 / 0.245 |
| critics_r1/seed_2/final | region r1 h1 | start_early | 28 (5) | **0.32** | 0.106 | 0.27 (11) | 0.44 | -0.14 (42) | 0.122 / 0.136 / 0.172 |
| critics_r1/seed_2/final | region r1 h1 | turn | 287 (5) | **0.47** | 0.037 | 0.51 (97) | 0.49 | -0.06 (57) | 0.155 / 0.162 / 0.338 |
| critics_r1/seed_2/final | region r1 h1 | start_late | 39 (7) | **0.49** | 0.103 | 0.58 (12) | 0.54 | -0.06 (40) | 0.107 / 0.104 / 0.133 |
| critics_r1/seed_2/final | region r1 h1 | north_leg | 314 (5) | **0.46** | 0.053 | 0.44 (98) | 0.45 | -0.07 (60) | 0.235 / 0.235 / 0.416 |
| critics_r1/seed_2/final | region r1 h1 | shortcut_early | 16 (4) | **0.38** | 0.207 | 0.50 (4) | 0.57 | -0.02 (41) | 0.139 / 0.139 / 0.168 |
| critics_r1/seed_2/final | region r1 h1 | general | 143 (22) | **0.48** | 0.059 | 0.51 (41) | 0.53 | 0.13 (55) | 0.469 / 0.469 / 0.594 |
| critics_r1/seed_2/final | region r1 h1 | dense_all | 684 (19) | **0.46** | 0.025 | 0.47 (222) | 0.49 | -0.07 (240) | 0.151 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | exact min | start_early | 28 (5) | **0.43** | 0.086 | 0.45 (11) | 0.40 | -0.20 (42) | 0.122 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | exact min | turn | 287 (5) | **0.44** | 0.031 | 0.47 (97) | 0.44 | -0.15 (57) | 0.137 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | exact min | start_late | 39 (7) | **0.62** | 0.094 | 0.58 (12) | 0.44 | -0.60 (40) | 0.082 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | exact min | north_leg | 314 (5) | **0.46** | 0.044 | 0.49 (98) | 0.49 | -0.03 (60) | 0.225 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | exact min | shortcut_early | 16 (4) | **0.50** | 0.123 | 0.50 (4) | 0.51 | 0.07 (41) | 0.144 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | exact min | general | 143 (22) | **0.65** | 0.066 | 0.68 (41) | 0.65 | 0.37 (55) | 0.545 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | exact min | dense_all | 684 (19) | **0.46** | 0.018 | 0.49 (222) | 0.46 | -0.17 (240) | 0.142 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | exact h0 | start_early | 28 (5) | **0.39** | 0.111 | 0.36 (11) | 0.45 | -0.11 (42) | 0.127 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | exact h0 | turn | 287 (5) | **0.46** | 0.032 | 0.54 (97) | 0.45 | -0.06 (57) | 0.153 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | exact h0 | start_late | 39 (7) | **0.59** | 0.086 | 0.58 (12) | 0.44 | -0.48 (40) | 0.085 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | exact h0 | north_leg | 314 (5) | **0.43** | 0.053 | 0.42 (98) | 0.43 | -0.29 (60) | 0.194 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | exact h0 | shortcut_early | 16 (4) | **0.50** | 0.123 | 0.50 (4) | 0.49 | 0.01 (41) | 0.141 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | exact h0 | general | 143 (22) | **0.64** | 0.067 | 0.61 (41) | 0.61 | 0.06 (55) | 0.505 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | exact h0 | dense_all | 684 (19) | **0.45** | 0.030 | 0.48 (222) | 0.45 | -0.18 (240) | 0.140 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | exact h1 | start_early | 28 (5) | **0.46** | 0.153 | 0.45 (11) | 0.46 | 0.15 (42) | 0.134 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | exact h1 | turn | 287 (5) | **0.46** | 0.037 | 0.51 (97) | 0.45 | -0.05 (57) | 0.158 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | exact h1 | start_late | 39 (7) | **0.49** | 0.101 | 0.58 (12) | 0.57 | -0.04 (40) | 0.104 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | exact h1 | north_leg | 314 (5) | **0.48** | 0.034 | 0.51 (98) | 0.51 | -0.00 (60) | 0.230 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | exact h1 | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.25 (4) | 0.44 | 0.03 (41) | 0.130 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | exact h1 | general | 143 (22) | **0.53** | 0.056 | 0.54 (41) | 0.54 | 0.26 (55) | 0.508 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | exact h1 | dense_all | 684 (19) | **0.47** | 0.016 | 0.50 (222) | 0.48 | 0.01 (240) | 0.151 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | start_early | 28 (5) | **0.46** | 0.082 | 0.55 (11) | 0.43 | -0.12 (42) | 0.128 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | turn | 287 (5) | **0.46** | 0.023 | 0.48 (97) | 0.47 | -0.03 (57) | 0.158 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | start_late | 39 (7) | **0.59** | 0.091 | 0.50 (12) | 0.43 | -0.48 (40) | 0.085 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | north_leg | 314 (5) | **0.46** | 0.053 | 0.48 (98) | 0.48 | -0.07 (60) | 0.218 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | shortcut_early | 16 (4) | **0.56** | 0.103 | 0.50 (4) | 0.51 | 0.03 (41) | 0.141 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | general | 143 (22) | **0.57** | 0.063 | 0.54 (41) | 0.55 | 0.22 (55) | 0.519 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r0.5 min | dense_all | 684 (19) | **0.47** | 0.023 | 0.49 (222) | 0.47 | -0.12 (240) | 0.146 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | start_early | 28 (5) | **0.43** | 0.110 | 0.45 (11) | 0.46 | -0.06 (42) | 0.132 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | turn | 287 (5) | **0.46** | 0.028 | 0.55 (97) | 0.46 | -0.06 (57) | 0.158 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | start_late | 39 (7) | **0.54** | 0.094 | 0.50 (12) | 0.44 | -0.60 (40) | 0.080 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | north_leg | 314 (5) | **0.43** | 0.058 | 0.41 (98) | 0.43 | -0.25 (60) | 0.193 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | shortcut_early | 16 (4) | **0.56** | 0.103 | 0.50 (4) | 0.52 | 0.09 (41) | 0.144 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | general | 143 (22) | **0.60** | 0.061 | 0.54 (41) | 0.58 | 0.09 (55) | 0.505 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h0 | dense_all | 684 (19) | **0.45** | 0.029 | 0.48 (222) | 0.46 | -0.17 (240) | 0.141 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | start_early | 28 (5) | **0.50** | 0.145 | 0.55 (11) | 0.46 | 0.15 (42) | 0.134 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | turn | 287 (5) | **0.45** | 0.032 | 0.49 (97) | 0.45 | -0.03 (57) | 0.159 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | start_late | 39 (7) | **0.51** | 0.090 | 0.67 (12) | 0.55 | -0.04 (40) | 0.102 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | north_leg | 314 (5) | **0.46** | 0.037 | 0.48 (98) | 0.50 | -0.04 (60) | 0.224 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | shortcut_early | 16 (4) | **0.62** | 0.189 | 0.25 (4) | 0.45 | -0.23 (41) | 0.127 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | general | 143 (22) | **0.51** | 0.053 | 0.59 (41) | 0.53 | 0.20 (55) | 0.496 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r0.5 h1 | dense_all | 684 (19) | **0.47** | 0.019 | 0.50 (222) | 0.48 | -0.04 (240) | 0.149 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r1 min | start_early | 28 (5) | **0.46** | 0.079 | 0.55 (11) | 0.45 | -0.07 (42) | 0.125 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r1 min | turn | 287 (5) | **0.46** | 0.030 | 0.53 (97) | 0.45 | 0.00 (57) | 0.160 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r1 min | start_late | 39 (7) | **0.59** | 0.091 | 0.50 (12) | 0.48 | -0.32 (40) | 0.090 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r1 min | north_leg | 314 (5) | **0.44** | 0.047 | 0.42 (98) | 0.47 | -0.18 (60) | 0.197 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r1 min | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.25 (4) | 0.50 | -0.04 (41) | 0.133 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r1 min | general | 143 (22) | **0.55** | 0.069 | 0.54 (41) | 0.54 | 0.14 (55) | 0.505 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r1 min | dense_all | 684 (19) | **0.46** | 0.022 | 0.47 (222) | 0.46 | -0.12 (240) | 0.141 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | start_early | 28 (5) | **0.43** | 0.096 | 0.55 (11) | 0.48 | -0.07 (42) | 0.132 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | turn | 287 (5) | **0.47** | 0.018 | 0.54 (97) | 0.46 | -0.03 (57) | 0.162 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | start_late | 39 (7) | **0.54** | 0.094 | 0.50 (12) | 0.46 | -0.56 (40) | 0.082 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | north_leg | 314 (5) | **0.46** | 0.050 | 0.44 (98) | 0.47 | -0.13 (60) | 0.218 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.25 (4) | 0.51 | -0.01 (41) | 0.133 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | general | 143 (22) | **0.60** | 0.064 | 0.61 (41) | 0.57 | 0.09 (55) | 0.515 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r1 h0 | dense_all | 684 (19) | **0.47** | 0.021 | 0.49 (222) | 0.47 | -0.15 (240) | 0.146 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | start_early | 28 (5) | **0.61** | 0.143 | 0.55 (11) | 0.45 | 0.18 (42) | 0.140 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | turn | 287 (5) | **0.44** | 0.031 | 0.54 (97) | 0.44 | -0.05 (57) | 0.152 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | start_late | 39 (7) | **0.49** | 0.092 | 0.58 (12) | 0.53 | 0.03 (40) | 0.105 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | north_leg | 314 (5) | **0.44** | 0.030 | 0.43 (98) | 0.47 | -0.10 (60) | 0.213 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | shortcut_early | 16 (4) | **0.56** | 0.164 | 0.25 (4) | 0.48 | -0.16 (41) | 0.132 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | general | 143 (22) | **0.50** | 0.056 | 0.54 (41) | 0.51 | 0.04 (55) | 0.482 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | region r1 h1 | dense_all | 684 (19) | **0.45** | 0.018 | 0.49 (222) | 0.47 | -0.03 (240) | 0.148 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | exact min | start_early | 28 (5) | **0.21** | 0.027 | 0.18 (11) | 0.48 | 0.23 (42) | 0.136 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | exact min | turn | 287 (5) | **0.45** | 0.032 | 0.54 (97) | 0.46 | -0.16 (57) | 0.151 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | exact min | start_late | 39 (7) | **0.67** | 0.085 | 0.75 (12) | 0.56 | 0.14 (40) | 0.116 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | exact min | north_leg | 314 (5) | **0.47** | 0.034 | 0.48 (98) | 0.48 | -0.01 (60) | 0.233 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | exact min | shortcut_early | 16 (4) | **0.50** | 0.149 | 0.50 (4) | 0.49 | -0.19 (41) | 0.137 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | exact min | general | 143 (22) | **0.42** | 0.059 | 0.49 (41) | 0.46 | 0.14 (55) | 0.498 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | exact min | dense_all | 684 (19) | **0.46** | 0.025 | 0.50 (222) | 0.48 | -0.01 (240) | 0.155 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | exact h0 | start_early | 28 (5) | **0.36** | 0.121 | 0.36 (11) | 0.43 | -0.12 (42) | 0.120 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | exact h0 | turn | 287 (5) | **0.46** | 0.035 | 0.55 (97) | 0.46 | 0.02 (57) | 0.178 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | exact h0 | start_late | 39 (7) | **0.38** | 0.110 | 0.42 (12) | 0.50 | -0.26 (40) | 0.097 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | exact h0 | north_leg | 314 (5) | **0.45** | 0.035 | 0.42 (98) | 0.46 | -0.05 (60) | 0.226 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | exact h0 | shortcut_early | 16 (4) | **0.50** | 0.149 | 0.50 (4) | 0.48 | 0.03 (41) | 0.142 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | exact h0 | general | 143 (22) | **0.49** | 0.063 | 0.44 (41) | 0.51 | 0.24 (55) | 0.511 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | exact h0 | dense_all | 684 (19) | **0.45** | 0.017 | 0.47 (222) | 0.46 | -0.07 (240) | 0.153 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | exact h1 | start_early | 28 (5) | **0.29** | 0.058 | 0.36 (11) | 0.51 | 0.07 (42) | 0.138 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | exact h1 | turn | 287 (5) | **0.45** | 0.030 | 0.54 (97) | 0.45 | -0.12 (57) | 0.157 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | exact h1 | start_late | 39 (7) | **0.69** | 0.084 | 0.75 (12) | 0.53 | 0.16 (40) | 0.115 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | exact h1 | north_leg | 314 (5) | **0.51** | 0.024 | 0.55 (98) | 0.51 | -0.04 (60) | 0.230 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | exact h1 | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.25 (4) | 0.54 | -0.19 (41) | 0.130 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | exact h1 | general | 143 (22) | **0.50** | 0.047 | 0.51 (41) | 0.51 | 0.06 (55) | 0.475 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | exact h1 | dense_all | 684 (19) | **0.49** | 0.021 | 0.54 (222) | 0.50 | -0.03 (240) | 0.154 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | start_early | 28 (5) | **0.18** | 0.052 | 0.18 (11) | 0.42 | 0.07 (42) | 0.125 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | turn | 287 (5) | **0.44** | 0.027 | 0.53 (97) | 0.45 | -0.13 (57) | 0.149 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | start_late | 39 (7) | **0.59** | 0.151 | 0.58 (12) | 0.54 | 0.08 (40) | 0.108 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | north_leg | 314 (5) | **0.47** | 0.036 | 0.44 (98) | 0.48 | -0.04 (60) | 0.228 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | shortcut_early | 16 (4) | **0.50** | 0.149 | 0.50 (4) | 0.49 | 0.13 (41) | 0.142 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | general | 143 (22) | **0.40** | 0.054 | 0.44 (41) | 0.44 | -0.06 (55) | 0.490 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r0.5 min | dense_all | 684 (19) | **0.45** | 0.022 | 0.47 (222) | 0.47 | 0.01 (240) | 0.151 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | start_early | 28 (5) | **0.36** | 0.121 | 0.36 (11) | 0.41 | -0.12 (42) | 0.120 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | turn | 287 (5) | **0.46** | 0.028 | 0.55 (97) | 0.46 | -0.03 (57) | 0.167 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | start_late | 39 (7) | **0.41** | 0.123 | 0.50 (12) | 0.51 | -0.25 (40) | 0.097 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | north_leg | 314 (5) | **0.44** | 0.039 | 0.41 (98) | 0.45 | -0.16 (60) | 0.206 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | shortcut_early | 16 (4) | **0.50** | 0.149 | 0.50 (4) | 0.46 | 0.17 (41) | 0.145 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | general | 143 (22) | **0.48** | 0.062 | 0.41 (41) | 0.51 | 0.10 (55) | 0.512 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h0 | dense_all | 684 (19) | **0.44** | 0.017 | 0.47 (222) | 0.45 | -0.08 (240) | 0.147 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | start_early | 28 (5) | **0.32** | 0.088 | 0.27 (11) | 0.50 | -0.04 (42) | 0.132 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | turn | 287 (5) | **0.44** | 0.036 | 0.51 (97) | 0.45 | -0.17 (57) | 0.152 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | start_late | 39 (7) | **0.64** | 0.096 | 0.58 (12) | 0.54 | 0.11 (40) | 0.115 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | north_leg | 314 (5) | **0.53** | 0.019 | 0.54 (98) | 0.52 | -0.01 (60) | 0.236 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | shortcut_early | 16 (4) | **0.56** | 0.184 | 0.25 (4) | 0.53 | -0.09 (41) | 0.132 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | general | 143 (22) | **0.42** | 0.053 | 0.44 (41) | 0.43 | -0.09 (55) | 0.462 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r0.5 h1 | dense_all | 684 (19) | **0.49** | 0.020 | 0.51 (222) | 0.50 | -0.05 (240) | 0.153 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r1 min | start_early | 28 (5) | **0.29** | 0.131 | 0.36 (11) | 0.36 | -0.16 (42) | 0.114 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r1 min | turn | 287 (5) | **0.44** | 0.028 | 0.57 (97) | 0.45 | -0.03 (57) | 0.158 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r1 min | start_late | 39 (7) | **0.46** | 0.159 | 0.50 (12) | 0.49 | -0.03 (40) | 0.098 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r1 min | north_leg | 314 (5) | **0.44** | 0.035 | 0.42 (98) | 0.47 | -0.21 (60) | 0.197 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r1 min | shortcut_early | 16 (4) | **0.44** | 0.182 | 0.25 (4) | 0.46 | 0.09 (41) | 0.136 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r1 min | general | 143 (22) | **0.42** | 0.054 | 0.39 (41) | 0.47 | -0.17 (55) | 0.486 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r1 min | dense_all | 684 (19) | **0.43** | 0.021 | 0.48 (222) | 0.45 | -0.08 (240) | 0.141 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | start_early | 28 (5) | **0.36** | 0.121 | 0.36 (11) | 0.43 | -0.09 (42) | 0.122 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | turn | 287 (5) | **0.45** | 0.028 | 0.56 (97) | 0.45 | 0.06 (57) | 0.174 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | start_late | 39 (7) | **0.46** | 0.107 | 0.58 (12) | 0.52 | -0.25 (40) | 0.097 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | north_leg | 314 (5) | **0.45** | 0.035 | 0.42 (98) | 0.45 | -0.18 (60) | 0.205 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | shortcut_early | 16 (4) | **0.44** | 0.182 | 0.25 (4) | 0.44 | 0.12 (41) | 0.141 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | general | 143 (22) | **0.47** | 0.061 | 0.46 (41) | 0.49 | 0.01 (55) | 0.504 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r1 h0 | dense_all | 684 (19) | **0.44** | 0.016 | 0.48 (222) | 0.46 | -0.07 (240) | 0.148 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | start_early | 28 (5) | **0.29** | 0.058 | 0.27 (11) | 0.49 | -0.03 (42) | 0.135 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | turn | 287 (5) | **0.46** | 0.038 | 0.55 (97) | 0.46 | 0.09 (57) | 0.176 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | start_late | 39 (7) | **0.59** | 0.108 | 0.50 (12) | 0.55 | 0.13 (40) | 0.112 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | north_leg | 314 (5) | **0.54** | 0.020 | 0.53 (98) | 0.54 | 0.00 (60) | 0.237 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | shortcut_early | 16 (4) | **0.50** | 0.195 | 0.25 (4) | 0.51 | -0.09 (41) | 0.132 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | general | 143 (22) | **0.45** | 0.053 | 0.46 (41) | 0.45 | -0.11 (55) | 0.455 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | region r1 h1 | dense_all | 684 (19) | **0.50** | 0.021 | 0.52 (222) | 0.51 | 0.02 (240) | 0.158 / 0.155 / 0.245 |

## Pass

- random: fail
- policy_logprob: fail
- knn_fit_w1: fail
- knn_fit_w4: fail
- critics_r1/seed_0/final | exact min: fail
- critics_r1/seed_0/final | exact h0: fail
- critics_r1/seed_0/final | exact h1: fail
- critics_r1/seed_0/final | region r0.5 min: fail
- critics_r1/seed_0/final | region r0.5 h0: fail
- critics_r1/seed_0/final | region r0.5 h1: fail
- critics_r1/seed_0/final | region r1 min: fail
- critics_r1/seed_0/final | region r1 h0: fail
- critics_r1/seed_0/final | region r1 h1: fail
- critics_r1/seed_1/final | exact min: fail
- critics_r1/seed_1/final | exact h0: fail
- critics_r1/seed_1/final | exact h1: fail
- critics_r1/seed_1/final | region r0.5 min: fail
- critics_r1/seed_1/final | region r0.5 h0: fail
- critics_r1/seed_1/final | region r0.5 h1: fail
- critics_r1/seed_1/final | region r1 min: fail
- critics_r1/seed_1/final | region r1 h0: fail
- critics_r1/seed_1/final | region r1 h1: fail
- critics_r1/seed_2/final | exact min: fail
- critics_r1/seed_2/final | exact h0: fail
- critics_r1/seed_2/final | exact h1: fail
- critics_r1/seed_2/final | region r0.5 min: fail
- critics_r1/seed_2/final | region r0.5 h0: fail
- critics_r1/seed_2/final | region r0.5 h1: fail
- critics_r1/seed_2/final | region r1 min: fail
- critics_r1/seed_2/final | region r1 h0: fail
- critics_r1/seed_2/final | region r1 h1: fail
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
