# First torque x continuation crossover at the start region (paired hazards, oracle model)

300 start-region anchors (t <= 5); each arm = the recorded first torque of one donor + steps 2..25 of the other or the same donor, then the blind driver by position, zero-torque hold after reaching; gamma 0.999, radius 0.5.  Direction after 25 steps: north = dy > 1 and dy > |dx|, east likewise.

| arm | first | continuation | north @25 | east @25 | neither | driver picks detour | reach | death | timeout | P_goal | reach step (median) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NN | north | north | 0.67 | 0.20 | 0.12 | 0.18 | 0.53 | 0.38 | 0.09 | 0.263 | 306.0 |
| NE | north | east | 0.02 | 0.91 | 0.07 | 0.00 | 0.22 | 0.77 | 0.01 | 0.139 | 229.0 |
| EN | east | north | 0.61 | 0.23 | 0.16 | 0.10 | 0.49 | 0.44 | 0.07 | 0.247 | 286.0 |
| EE | east | east | 0.02 | 0.97 | 0.01 | 0.01 | 0.23 | 0.77 | 0.00 | 0.145 | 226.0 |

| outcome | first-torque effect (N - E, averaged over continuations) | continuation effect (N - E, averaged over first torques) | interaction |
|---|---:|---:|---:|
| P(north after 25) | +0.033 | +0.623 | +0.060 |
| P(driver takes detour) | +0.042 | +0.135 | +0.090 |
| reach | +0.015 | +0.278 | +0.050 |
| death | -0.032 | -0.355 | -0.063 |
| P_goal | +0.005 | +0.113 | +0.022 |

log P_goal ratios: NN/EE +0.59 (both differ), NE/EE -0.05 (only the first torque is north), EN/EE +0.53 (only the continuation is north), NN/EN +0.06 (first torque, given a north continuation).

Reading: the first-torque effect is what a critic scoring (s, a_1) can carry at best; the continuation effect is what the replay row's realised future carries but the critic's input does not see.

Within-anchor, paired hazards (300 anchors): swapping ONLY the first torque changes the 25-step direction in 0.22 of the pairs; swapping ONLY the continuation changes it in 0.77.
