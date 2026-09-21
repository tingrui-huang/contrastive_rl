# The complete actor objective at the visited start states: stalled policy (final) vs progressing policy (d2final); critics ['own_d1', 'other_d2']

Objective per row = 0.95 * (-min-twin Q(s, a_policy, g)) + 0.05 * (-log pi_policy(a_logged | s, g)), the actor loss's weighting; goals = the task goal or 8 relabelled future goals of the anchor's logged episode; states = the 64 reset anchors' rollouts at steps [0, 5, 10, 20, 30].

| visited states of | goal | critic | n | Q stall / prog (mean) | critic prefers stall | NLL stall / prog | objective prefers stall | mean action gap | prefers stall by step (critic ; objective) |
|---|---|---|---:|---|---:|---|---:|---:|---|
| stall | task | own_d1 | 320 | -8.53 / -8.75 | 0.52 | 29.41 / 40.63 | 0.52 | 0.567 | 0: 1.00;0.97 5: 0.64;0.59 10: 0.31;0.38 20: 0.39;0.39 30: 0.25;0.27 |
| stall | task | other_d2 | 320 | -10.50 / -8.73 | 0.11 | 29.41 / 40.63 | 0.13 | 0.567 | 0: 0.09;0.05 5: 0.11;0.14 10: 0.19;0.23 20: 0.08;0.12 30: 0.08;0.12 |
| stall | relabelled | own_d1 | 320 | -9.26 / -9.82 | 0.63 | 31.89 / 53.03 | 0.67 | 0.506 | 0: 1.00;1.00 5: 0.52;0.52 10: 0.34;0.47 20: 0.64;0.66 30: 0.66;0.70 |
| stall | relabelled | other_d2 | 320 | -10.91 / -9.67 | 0.21 | 31.89 / 53.03 | 0.24 | 0.506 | 0: 0.00;0.00 5: 0.23;0.25 10: 0.27;0.27 20: 0.23;0.33 30: 0.31;0.34 |
| prog | task | own_d1 | 320 | -8.62 / -9.26 | 0.67 | 391.26 / 552.84 | 0.71 | 0.364 | 0: 1.00;0.97 5: 0.70;0.72 10: 0.47;0.53 20: 0.64;0.64 30: 0.53;0.69 |
| prog | task | other_d2 | 320 | -10.29 / -9.48 | 0.25 | 391.26 / 552.84 | 0.40 | 0.364 | 0: 0.09;0.05 5: 0.23;0.39 10: 0.30;0.41 20: 0.30;0.52 30: 0.31;0.62 |
| prog | relabelled | own_d1 | 320 | -11.31 / -11.78 | 0.71 | 592.24 / 929.07 | 0.63 | 0.270 | 0: 1.00;0.98 5: 0.77;0.61 10: 0.59;0.47 20: 0.69;0.61 30: 0.50;0.50 |
| prog | relabelled | other_d2 | 320 | -13.09 / -12.54 | 0.27 | 592.24 / 929.07 | 0.41 | 0.270 | 0: 0.03;0.00 5: 0.41;0.53 10: 0.36;0.45 20: 0.30;0.56 30: 0.25;0.50 |
