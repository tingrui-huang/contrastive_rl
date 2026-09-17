# Gate (round 1): does f(s, a, g) order the candidate torques the way their validated single-step outcomes do?

Held-out anchors (100 held-out episodes), candidates ['recorded', 'mode', 'sample0', 'sample1', 'sample2'], 4 paired draws per key; the continuation is the frozen policy mode.  A pair (a_i, a_j) at one state is VALIDATED when the log P_goal ratio has the same sign on draws {0,1} and {2,3} and |ratio| > 0.3 on both.  Agreement = share of validated pairs whose critic ordering matches (chance 0.50); samples-only = pairs of two policy samples (excludes recorded and mode); weighted = |ratio|-weighted sign agreement over all pairs; pick gain = (P[argmax f] - mean P) / (max P - mean P) over anchors with spread (1 = oracle, 0 = random).  PASS = pooled dense-set validated agreement >= 0.65 and > 0.5 by 2 s.e.

## Ceiling: split-half sign agreement of the differences themselves

| set | pairs | mean abs log ratio | frac > thr | split-half agreement | validated pairs |
|---|---:|---:|---:|---:|---:|
| start_early | 600 | 0.88 | 0.22 | 0.38 | 28 |
| turn | 600 | 2.13 | 0.67 | 0.76 | 287 |
| start_late | 600 | 0.80 | 0.24 | 0.41 | 39 |
| north_leg | 600 | 2.06 | 0.74 | 0.82 | 314 |
| shortcut_early | 600 | 0.81 | 0.21 | 0.26 | 16 |
| general | 600 | 1.19 | 0.32 | 0.89 | 143 |

## Critics

| scorer | set | validated pairs | agreement | s.e. | samples-only agreement (n) | weighted agreement | pick gain (anchors) | P pick / mean / best |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | start_early | 28 | **0.46** | 0.094 | 0.45 (11) | 0.50 | 0.20 (42) | 0.140 / 0.136 / 0.172 |
| random | turn | 287 | **0.48** | 0.029 | 0.52 (97) | 0.52 | -0.00 (57) | 0.149 / 0.162 / 0.338 |
| random | start_late | 39 | **0.49** | 0.080 | 0.58 (12) | 0.41 | -0.23 (40) | 0.096 / 0.104 / 0.133 |
| random | north_leg | 314 | **0.32** | 0.026 | 0.32 (98) | 0.32 | -0.40 (60) | 0.180 / 0.235 / 0.416 |
| random | shortcut_early | 16 | **0.38** | 0.121 | 0.50 (4) | 0.58 | 0.28 (41) | 0.150 / 0.139 / 0.168 |
| random | general | 143 | **0.41** | 0.041 | 0.49 (41) | 0.44 | -0.06 (55) | 0.464 / 0.469 / 0.594 |
| random | dense_all | 684 | **0.41** | 0.019 | 0.43 (222) | 0.45 | -0.06 (240) | 0.143 / 0.155 / 0.245 |
| policy_logprob | start_early | 28 | **0.89** | 0.058 | 1.00 (11) | 0.48 | -0.28 (42) | 0.133 / 0.136 / 0.172 |
| policy_logprob | turn | 287 | **0.52** | 0.029 | 0.54 (97) | 0.52 | -0.01 (57) | 0.154 / 0.162 / 0.338 |
| policy_logprob | start_late | 39 | **0.56** | 0.079 | 0.67 (12) | 0.50 | 0.09 (40) | 0.102 / 0.104 / 0.133 |
| policy_logprob | north_leg | 314 | **0.47** | 0.028 | 0.47 (98) | 0.48 | -0.11 (60) | 0.225 / 0.235 / 0.416 |
| policy_logprob | shortcut_early | 16 | **0.50** | 0.125 | 0.25 (4) | 0.49 | 0.01 (41) | 0.135 / 0.139 / 0.168 |
| policy_logprob | general | 143 | **0.44** | 0.042 | 0.51 (41) | 0.47 | -0.34 (55) | 0.452 / 0.469 / 0.594 |
| policy_logprob | dense_all | 684 | **0.52** | 0.019 | 0.53 (222) | 0.50 | -0.06 (240) | 0.150 / 0.155 / 0.245 |
| knn_fit_w1 | start_early | 28 | **0.46** | 0.094 | 0.45 (11) | 0.33 | -0.02 (42) | 0.132 / 0.136 / 0.172 |
| knn_fit_w1 | turn | 287 | **0.40** | 0.029 | 0.40 (97) | 0.38 | 0.01 (57) | 0.153 / 0.162 / 0.338 |
| knn_fit_w1 | start_late | 39 | **0.18** | 0.061 | 0.25 (12) | 0.22 | 0.22 (40) | 0.111 / 0.104 / 0.133 |
| knn_fit_w1 | north_leg | 314 | **0.48** | 0.028 | 0.52 (98) | 0.51 | 0.01 (60) | 0.232 / 0.235 / 0.416 |
| knn_fit_w1 | shortcut_early | 16 | **0.44** | 0.124 | 1.00 (4) | 0.25 | -0.27 (41) | 0.127 / 0.139 / 0.168 |
| knn_fit_w1 | general | 143 | **0.22** | 0.034 | 0.22 (41) | 0.23 | 0.38 (55) | 0.520 / 0.469 / 0.594 |
| knn_fit_w1 | dense_all | 684 | **0.43** | 0.019 | 0.46 (222) | 0.38 | -0.01 (240) | 0.151 / 0.155 / 0.245 |
| knn_fit_w4 | start_early | 28 | **0.61** | 0.092 | 0.64 (11) | 0.43 | 0.08 (42) | 0.135 / 0.136 / 0.172 |
| knn_fit_w4 | turn | 287 | **0.44** | 0.029 | 0.47 (97) | 0.44 | -0.04 (57) | 0.161 / 0.162 / 0.338 |
| knn_fit_w4 | start_late | 39 | **0.28** | 0.072 | 0.42 (12) | 0.27 | -0.03 (40) | 0.099 / 0.104 / 0.133 |
| knn_fit_w4 | north_leg | 314 | **0.48** | 0.028 | 0.42 (98) | 0.49 | 0.07 (60) | 0.249 / 0.235 / 0.416 |
| knn_fit_w4 | shortcut_early | 16 | **0.38** | 0.121 | 0.00 (4) | 0.37 | -0.08 (41) | 0.147 / 0.139 / 0.168 |
| knn_fit_w4 | general | 143 | **0.39** | 0.041 | 0.46 (41) | 0.40 | 0.40 (55) | 0.477 / 0.469 / 0.594 |
| knn_fit_w4 | dense_all | 684 | **0.45** | 0.019 | 0.45 (222) | 0.43 | 0.00 (240) | 0.158 / 0.155 / 0.245 |
| critics_r1/seed_0/final | start_early | 28 | **0.61** | 0.092 | 0.64 (11) | 0.49 | -0.12 (42) | 0.134 / 0.136 / 0.172 |
| critics_r1/seed_0/final | turn | 287 | **0.47** | 0.029 | 0.43 (97) | 0.49 | -0.02 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_0/final | start_late | 39 | **0.59** | 0.079 | 0.67 (12) | 0.50 | 0.14 (40) | 0.108 / 0.104 / 0.133 |
| critics_r1/seed_0/final | north_leg | 314 | **0.52** | 0.028 | 0.53 (98) | 0.51 | 0.06 (60) | 0.247 / 0.235 / 0.416 |
| critics_r1/seed_0/final | shortcut_early | 16 | **0.75** | 0.108 | 0.75 (4) | 0.55 | 0.11 (41) | 0.155 / 0.139 / 0.168 |
| critics_r1/seed_0/final | general | 143 | **0.49** | 0.042 | 0.37 (41) | 0.51 | 0.09 (55) | 0.445 / 0.469 / 0.594 |
| critics_r1/seed_0/final | dense_all | 684 | **0.51** | 0.019 | 0.50 (222) | 0.50 | 0.03 (240) | 0.162 / 0.155 / 0.245 |

## Pass

- random: fail
- policy_logprob: fail
- knn_fit_w1: fail
- knn_fit_w4: fail
- critics_r1/seed_0/final: fail
