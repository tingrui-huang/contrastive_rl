# Arm E: corner / edge query coverage

replay_E = replay_D + 1100 ETT paths from the 550 training contexts with six first queries [[0.3, -1.0], [1.0, -1.0]]; those paths: reach 0.493, absorbed 0.469, lower 0.492. Five 30k critics (sealed recipe), three fresh balanced-BC actors each (300k, Acme, rg0, bc 0.05, cap 0.25), 300 new-seed episodes. Family = fork mode under the task goal: DOWN (y < -0.9, x < 0.6), CORNER (y < -0.9, x >= 0.6), RIGHT (otherwise).

| critic | Qbar DOWN-CORNER | actor | fork mode | family | mode reach | mode lower | sample lower |
|---|---:|---|---|---|---:|---:|---:|
| E seed 0 | +0.434 | a0 | (+0.90, -0.70) | RIGHT | 0.363 | 0.000 | 0.353 |
| E seed 0 | +0.434 | a1 | (+0.91, -0.66) | RIGHT | 0.363 | 0.000 | 0.377 |
| E seed 0 | +0.434 | a2 | (+0.95, -0.63) | RIGHT | 0.363 | 0.000 | 0.217 |
| E seed 1 | +0.561 | a0 | (+1.00, +0.24) | RIGHT | 0.363 | 0.000 | 0.037 |
| E seed 1 | +0.561 | a1 | (+1.00, +0.19) | RIGHT | 0.363 | 0.000 | 0.047 |
| E seed 1 | +0.561 | a2 | (+1.00, +0.24) | RIGHT | 0.363 | 0.000 | 0.033 |
| E seed 2 | +0.597 | a0 | (+0.36, -1.00) | DOWN | 1.000 | 1.000 | 0.907 |
| E seed 2 | +0.597 | a1 | (+0.29, -1.00) | DOWN | 1.000 | 1.000 | 0.960 |
| E seed 2 | +0.597 | a2 | (+0.36, -1.00) | DOWN | 1.000 | 1.000 | 0.923 |
| E seed 3 | +0.351 | a0 | (+1.00, +0.09) | RIGHT | 0.363 | 0.000 | 0.083 |
| E seed 3 | +0.351 | a1 | (+1.00, +0.02) | RIGHT | 0.363 | 0.000 | 0.123 |
| E seed 3 | +0.351 | a2 | (+1.00, -0.07) | RIGHT | 0.363 | 0.000 | 0.093 |
| E seed 4 | +0.680 | a0 | (+1.00, -0.33) | RIGHT | 0.363 | 0.000 | 0.103 |
| E seed 4 | +0.680 | a1 | (+0.99, -0.31) | RIGHT | 0.363 | 0.000 | 0.123 |
| E seed 4 | +0.680 | a2 | (+1.00, -0.38) | RIGHT | 0.363 | 0.000 | 0.093 |

DOWN-family actors: 3/15; actors with mode lower >= 0.9: 3/15; critics producing at least one DOWN actor: [2].
Reference (D series, 6 critics x 3 actors): 13/18 at >= 0.81, DOWN family only under D1 and joint seed 0.
