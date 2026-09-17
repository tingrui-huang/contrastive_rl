# Gate (round 1): does f(s, a, g) order the candidate torques the way their validated single-step outcomes do?

Held-out anchors (100 held-out episodes), candidates ['recorded', 'mode', 'sample0', 'sample1', 'sample2'], 4 paired draws per key; the continuation is the frozen policy mode.  A pair (a_i, a_j) at one state is VALIDATED when the log P_goal ratio has the same sign on draws {0,1} and {2,3} and |ratio| > 0.3 on both.  Agreement = share of validated pairs whose critic ordering matches (chance 0.50); samples-only = pairs of two policy samples (excludes recorded and mode); weighted = |ratio|-weighted sign agreement over all pairs; pick gain = (P[argmax f] - mean P) / (max P - mean P) over anchors with spread (1 = oracle, 0 = random).  PASS = pooled dense-set validated agreement >= 0.65 and > 0.5 by 2 s.e. (s.e. = anchor-level bootstrap: the pairs of one anchor share its candidates and draws).

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

| scorer | set | validated pairs (anchors) | agreement | s.e. | samples-only agreement (n) | weighted agreement | pick gain (anchors) | P pick / mean / best |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | start_early | 28 (5) | **0.46** | 0.099 | 0.45 (11) | 0.50 | 0.20 (42) | 0.140 / 0.136 / 0.172 |
| random | turn | 287 (50) | **0.48** | 0.041 | 0.52 (97) | 0.52 | -0.00 (57) | 0.149 / 0.162 / 0.338 |
| random | start_late | 39 (7) | **0.49** | 0.129 | 0.58 (12) | 0.41 | -0.23 (40) | 0.096 / 0.104 / 0.133 |
| random | north_leg | 314 (57) | **0.32** | 0.032 | 0.32 (98) | 0.32 | -0.40 (60) | 0.180 / 0.235 / 0.416 |
| random | shortcut_early | 16 (4) | **0.38** | 0.184 | 0.50 (4) | 0.58 | 0.28 (41) | 0.150 / 0.139 / 0.168 |
| random | general | 143 (28) | **0.41** | 0.051 | 0.49 (41) | 0.44 | -0.06 (55) | 0.464 / 0.469 / 0.594 |
| random | dense_all | 684 (123) | **0.41** | 0.025 | 0.43 (222) | 0.45 | -0.06 (240) | 0.143 / 0.155 / 0.245 |
| policy_logprob | start_early | 28 (5) | **0.89** | 0.087 | 1.00 (11) | 0.48 | -0.28 (42) | 0.133 / 0.136 / 0.172 |
| policy_logprob | turn | 287 (50) | **0.52** | 0.034 | 0.54 (97) | 0.52 | -0.01 (57) | 0.154 / 0.162 / 0.338 |
| policy_logprob | start_late | 39 (7) | **0.56** | 0.109 | 0.67 (12) | 0.50 | 0.09 (40) | 0.102 / 0.104 / 0.133 |
| policy_logprob | north_leg | 314 (57) | **0.47** | 0.037 | 0.47 (98) | 0.48 | -0.11 (60) | 0.225 / 0.235 / 0.416 |
| policy_logprob | shortcut_early | 16 (4) | **0.50** | 0.176 | 0.25 (4) | 0.49 | 0.01 (41) | 0.135 / 0.139 / 0.168 |
| policy_logprob | general | 143 (28) | **0.44** | 0.056 | 0.51 (41) | 0.47 | -0.34 (55) | 0.452 / 0.469 / 0.594 |
| policy_logprob | dense_all | 684 (123) | **0.52** | 0.025 | 0.53 (222) | 0.50 | -0.06 (240) | 0.150 / 0.155 / 0.245 |
| critics_r1/seed_0/final | start_early | 28 (5) | **0.61** | 0.058 | 0.64 (11) | 0.49 | -0.12 (42) | 0.134 / 0.136 / 0.172 |
| critics_r1/seed_0/final | turn | 287 (50) | **0.47** | 0.040 | 0.43 (97) | 0.49 | -0.02 (57) | 0.166 / 0.162 / 0.338 |
| critics_r1/seed_0/final | start_late | 39 (7) | **0.59** | 0.088 | 0.67 (12) | 0.50 | 0.14 (40) | 0.108 / 0.104 / 0.133 |
| critics_r1/seed_0/final | north_leg | 314 (57) | **0.52** | 0.039 | 0.53 (98) | 0.51 | 0.06 (60) | 0.247 / 0.235 / 0.416 |
| critics_r1/seed_0/final | shortcut_early | 16 (4) | **0.75** | 0.085 | 0.75 (4) | 0.55 | 0.11 (41) | 0.155 / 0.139 / 0.168 |
| critics_r1/seed_0/final | general | 143 (28) | **0.49** | 0.057 | 0.37 (41) | 0.51 | 0.09 (55) | 0.445 / 0.469 / 0.594 |
| critics_r1/seed_0/final | dense_all | 684 (123) | **0.51** | 0.025 | 0.50 (222) | 0.50 | 0.03 (240) | 0.162 / 0.155 / 0.245 |
| critics_r1/seed_1/final | start_early | 28 (5) | **0.61** | 0.144 | 0.64 (11) | 0.44 | -0.30 (42) | 0.137 / 0.136 / 0.172 |
| critics_r1/seed_1/final | turn | 287 (50) | **0.47** | 0.038 | 0.52 (97) | 0.52 | -0.01 (57) | 0.165 / 0.162 / 0.338 |
| critics_r1/seed_1/final | start_late | 39 (7) | **0.51** | 0.062 | 0.58 (12) | 0.59 | 0.42 (40) | 0.114 / 0.104 / 0.133 |
| critics_r1/seed_1/final | north_leg | 314 (57) | **0.54** | 0.040 | 0.52 (98) | 0.52 | -0.09 (60) | 0.230 / 0.235 / 0.416 |
| critics_r1/seed_1/final | shortcut_early | 16 (4) | **0.62** | 0.142 | 0.25 (4) | 0.48 | 0.14 (41) | 0.147 / 0.139 / 0.168 |
| critics_r1/seed_1/final | general | 143 (28) | **0.48** | 0.058 | 0.51 (41) | 0.51 | -0.05 (55) | 0.483 / 0.469 / 0.594 |
| critics_r1/seed_1/final | dense_all | 684 (123) | **0.52** | 0.026 | 0.52 (222) | 0.51 | 0.02 (240) | 0.159 / 0.155 / 0.245 |
| critics_r1/seed_2/final | start_early | 28 (5) | **0.36** | 0.156 | 0.27 (11) | 0.33 | -0.77 (42) | 0.112 / 0.136 / 0.172 |
| critics_r1/seed_2/final | turn | 287 (50) | **0.46** | 0.038 | 0.42 (97) | 0.49 | 0.07 (57) | 0.179 / 0.162 / 0.338 |
| critics_r1/seed_2/final | start_late | 39 (7) | **0.56** | 0.081 | 0.58 (12) | 0.48 | 0.09 (40) | 0.103 / 0.104 / 0.133 |
| critics_r1/seed_2/final | north_leg | 314 (57) | **0.54** | 0.042 | 0.49 (98) | 0.54 | 0.05 (60) | 0.262 / 0.235 / 0.416 |
| critics_r1/seed_2/final | shortcut_early | 16 (4) | **0.50** | 0.171 | 0.50 (4) | 0.50 | -0.03 (41) | 0.141 / 0.139 / 0.168 |
| critics_r1/seed_2/final | general | 143 (28) | **0.47** | 0.060 | 0.41 (41) | 0.44 | 0.12 (55) | 0.460 / 0.469 / 0.594 |
| critics_r1/seed_2/final | dense_all | 684 (123) | **0.50** | 0.025 | 0.45 (222) | 0.48 | -0.10 (240) | 0.159 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_0/final | start_early | 28 (5) | **0.43** | 0.087 | 0.45 (11) | 0.40 | -0.20 (42) | 0.122 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_0/final | turn | 287 (50) | **0.44** | 0.040 | 0.47 (97) | 0.44 | -0.15 (57) | 0.137 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_0/final | start_late | 39 (7) | **0.62** | 0.095 | 0.58 (12) | 0.44 | -0.60 (40) | 0.082 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_0/final | north_leg | 314 (57) | **0.46** | 0.041 | 0.49 (98) | 0.49 | -0.03 (60) | 0.225 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_0/final | shortcut_early | 16 (4) | **0.50** | 0.123 | 0.50 (4) | 0.51 | 0.07 (41) | 0.144 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_0/final | general | 143 (28) | **0.65** | 0.065 | 0.68 (41) | 0.65 | 0.37 (55) | 0.545 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_0/final | dense_all | 684 (123) | **0.46** | 0.027 | 0.49 (222) | 0.46 | -0.17 (240) | 0.142 / 0.155 / 0.245 |
| vanilla_g0999_30k/seed_1/final | start_early | 28 (5) | **0.21** | 0.028 | 0.18 (11) | 0.48 | 0.23 (42) | 0.136 / 0.136 / 0.172 |
| vanilla_g0999_30k/seed_1/final | turn | 287 (50) | **0.45** | 0.035 | 0.54 (97) | 0.46 | -0.16 (57) | 0.151 / 0.162 / 0.338 |
| vanilla_g0999_30k/seed_1/final | start_late | 39 (7) | **0.67** | 0.084 | 0.75 (12) | 0.56 | 0.14 (40) | 0.116 / 0.104 / 0.133 |
| vanilla_g0999_30k/seed_1/final | north_leg | 314 (57) | **0.47** | 0.042 | 0.48 (98) | 0.48 | -0.01 (60) | 0.233 / 0.235 / 0.416 |
| vanilla_g0999_30k/seed_1/final | shortcut_early | 16 (4) | **0.50** | 0.153 | 0.50 (4) | 0.49 | -0.19 (41) | 0.137 / 0.139 / 0.168 |
| vanilla_g0999_30k/seed_1/final | general | 143 (28) | **0.42** | 0.060 | 0.49 (41) | 0.46 | 0.14 (55) | 0.498 / 0.469 / 0.594 |
| vanilla_g0999_30k/seed_1/final | dense_all | 684 (123) | **0.46** | 0.026 | 0.50 (222) | 0.48 | -0.01 (240) | 0.155 / 0.155 / 0.245 |

## Pass

- random: fail
- policy_logprob: fail
- critics_r1/seed_0/final: fail
- critics_r1/seed_1/final: fail
- critics_r1/seed_2/final: fail
- vanilla_g0999_30k/seed_0/final: fail
- vanilla_g0999_30k/seed_1/final: fail
