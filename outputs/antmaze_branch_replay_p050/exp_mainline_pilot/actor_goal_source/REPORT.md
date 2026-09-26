# Actor goal-source comparison (frozen critic; anchor-pool actor rows; critic-term goal recorded vs counterfactual; BC unchanged)

`manifest.json`, `report.json`.  Evaluation draw seed 6909, 300 episodes, mode; every policy once.  Rule = mean over the 3 lineages > 2 x seed s.e. and 3 / 3.

| policy | success | detour | death | timeout | far n / completed / timeouts | no-route | mean steps |
|---|---|---|---|---|---|---|---|
| goal_log/seed_0 | 0.407 | 0.477 | 0.197 | 0.397 | 143 / 96 / 47 | 58 | 522 |
| goal_cf/seed_0 | 0.277 | 0.007 | 0.720 | 0.003 | 2 / 2 / 0 | 7 | 131 |
| current/seed_0 | 0.520 | 0.590 | 0.193 | 0.287 | 177 / 129 / 48 | 40 | 493 |
| frozen/seed_0 | 0.497 | 0.713 | 0.010 | 0.493 | 214 / 130 / 84 | 62 | 663 |
| goal_log/seed_1 | 0.293 | 0.297 | 0.417 | 0.290 | 89 / 54 / 35 | 34 | 397 |
| goal_cf/seed_1 | 0.257 | 0.000 | 0.730 | 0.013 | 0 / 0 / 0 | 4 | 134 |
| current/seed_1 | 0.437 | 0.440 | 0.350 | 0.213 | 132 / 94 / 38 | 21 | 389 |
| frozen/seed_1 | 0.500 | 0.860 | 0.010 | 0.490 | 258 / 144 / 114 | 34 | 701 |
| goal_log/seed_2 | 0.393 | 0.257 | 0.303 | 0.303 | 77 / 51 / 26 | 56 | 424 |
| goal_cf/seed_2 | 0.267 | 0.000 | 0.730 | 0.003 | 0 / 0 / 0 | 11 | 128 |
| current/seed_2 | 0.433 | 0.307 | 0.427 | 0.140 | 92 / 77 / 15 | 29 | 324 |
| frozen/seed_2 | 0.140 | 0.107 | 0.000 | 0.860 | 32 / 20 / 12 | 218 | 750 |

| comparison | quantity | per lineage | mean | seed s.e. | same direction | rule met |
|---|---|---|---|---|---|---|
| goal_cf_minus_goal_log | success | -0.130 / -0.037 / -0.127 | -0.098 | 0.031 | 3/3 | no |
| goal_cf_minus_current | success | -0.243 / -0.180 / -0.167 | -0.197 | 0.024 | 3/3 | no |
| goal_log_minus_current | success | -0.113 / -0.143 / -0.040 | -0.099 | 0.031 | 3/3 | no |
| goal_cf_minus_frozen | success | -0.220 / -0.243 / +0.127 | -0.112 | 0.120 | 2/3 | no |
| goal_log_minus_frozen | success | -0.090 / -0.207 / +0.253 | -0.014 | 0.138 | 2/3 | no |
| goal_cf_minus_goal_log | detour | -0.470 / -0.297 / -0.257 | -0.341 | 0.065 | 3/3 | no |
| goal_cf_minus_current | detour | -0.583 / -0.440 / -0.307 | -0.443 | 0.080 | 3/3 | no |
| goal_log_minus_current | detour | -0.113 / -0.143 / -0.050 | -0.102 | 0.028 | 3/3 | no |
| goal_cf_minus_frozen | detour | -0.707 / -0.860 / -0.107 | -0.558 | 0.230 | 3/3 | no |
| goal_log_minus_frozen | detour | -0.237 / -0.563 / +0.150 | -0.217 | 0.206 | 2/3 | no |
| goal_cf_minus_goal_log | failure | +0.523 / +0.313 / +0.427 | +0.421 | 0.061 | 3/3 | YES |
| goal_cf_minus_current | failure | +0.527 / +0.380 / +0.303 | +0.403 | 0.066 | 3/3 | YES |
| goal_log_minus_current | failure | +0.003 / +0.067 / -0.123 | -0.018 | 0.056 | 1/3 | no |
| goal_cf_minus_frozen | failure | +0.710 / +0.720 / +0.730 | +0.720 | 0.006 | 3/3 | YES |
| goal_log_minus_frozen | failure | +0.187 / +0.407 / +0.303 | +0.299 | 0.064 | 3/3 | YES |
| goal_cf_minus_goal_log | timeout | -0.393 / -0.277 / -0.300 | -0.323 | 0.036 | 3/3 | no |
| goal_cf_minus_current | timeout | -0.283 / -0.200 / -0.137 | -0.207 | 0.042 | 3/3 | no |
| goal_log_minus_current | timeout | +0.110 / +0.077 / +0.163 | +0.117 | 0.025 | 3/3 | YES |
| goal_cf_minus_frozen | timeout | -0.490 / -0.477 / -0.857 | -0.608 | 0.125 | 3/3 | no |
| goal_log_minus_frozen | timeout | -0.097 / -0.200 / -0.557 | -0.284 | 0.139 | 3/3 | no |

Identity (first 4 batches): seed_0: anchors equal True, BC rows equal True, critic-term goals differ True; seed_1: anchors equal True, BC rows equal True, critic-term goals differ True; seed_2: anchors equal True, BC rows equal True, critic-term goals differ True

