# The complete actor objective at the visited start states: stalled policy (final) vs progressing policy (20k); critics ['own_d1', 'own_d1_20k']

Objective per row = 0.95 * (-min-twin Q(s, a_policy, g)) + 0.05 * (-log pi_policy(a_logged | s, g)), the actor loss's weighting; goals = the task goal or 8 relabelled future goals of the anchor's logged episode; states = the 64 reset anchors' rollouts at steps [0, 5, 10, 20, 30].

| visited states of | goal | critic | n | Q stall / prog (mean) | critic prefers stall | NLL stall / prog | objective prefers stall | mean action gap | prefers stall by step (critic ; objective) |
|---|---|---|---:|---|---:|---|---:|---:|---|
| stall | task | own_d1 | 320 | -8.53 / -8.62 | 0.54 | 29.41 / 42.06 | 0.60 | 0.192 | 0: 0.69;0.75 5: 0.67;0.67 10: 0.44;0.47 20: 0.48;0.61 30: 0.44;0.52 |
| stall | task | own_d1_20k | 320 | -7.87 / -7.89 | 0.45 | 29.41 / 42.06 | 0.56 | 0.192 | 0: 0.22;0.39 5: 0.62;0.67 10: 0.48;0.47 20: 0.52;0.70 30: 0.39;0.58 |
| stall | relabelled | own_d1 | 320 | -9.26 / -9.48 | 0.67 | 31.89 / 46.24 | 0.71 | 0.210 | 0: 0.98;0.98 5: 0.61;0.67 10: 0.42;0.50 20: 0.70;0.70 30: 0.64;0.69 |
| stall | relabelled | own_d1_20k | 320 | -8.29 / -8.44 | 0.63 | 31.89 / 46.24 | 0.61 | 0.210 | 0: 0.80;0.44 5: 0.58;0.69 10: 0.45;0.50 20: 0.64;0.70 30: 0.67;0.70 |
| prog | task | own_d1 | 320 | -8.71 / -8.98 | 0.61 | 162.89 / 298.28 | 0.67 | 0.218 | 0: 0.69;0.75 5: 0.66;0.59 10: 0.47;0.69 20: 0.62;0.64 30: 0.61;0.66 |
| prog | task | own_d1_20k | 320 | -8.05 / -8.22 | 0.56 | 162.89 / 298.28 | 0.61 | 0.218 | 0: 0.22;0.39 5: 0.69;0.64 10: 0.62;0.70 20: 0.64;0.64 30: 0.62;0.69 |
| prog | relabelled | own_d1 | 320 | -12.05 / -12.41 | 0.68 | 232.87 / 331.11 | 0.69 | 0.223 | 0: 0.98;0.95 5: 0.55;0.55 10: 0.64;0.72 20: 0.53;0.56 30: 0.67;0.69 |
| prog | relabelled | own_d1_20k | 320 | -10.18 / -10.48 | 0.69 | 232.87 / 331.11 | 0.66 | 0.223 | 0: 0.75;0.50 5: 0.75;0.72 10: 0.64;0.77 20: 0.62;0.59 30: 0.70;0.72 |
