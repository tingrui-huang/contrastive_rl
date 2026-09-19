# Far-route timeouts of the clipped CF policies (critic_clip0.1): late entrance, stalls, entrance state, and the continuation test

Same 300 evaluation episodes (seed 3909, mode policy) replayed with full capture for the six critic_clip0.1 policies (`rollout_check_critic_clip0.1.json`); far route = the env's detour label (top-west corner, y >= 6 at x < 2); turn north = y >= 2.  `detour_audit_critic_clip0.1.json`.

## 1. Route ledger

| policy | far route n (share) | completed | far timeouts / deaths | completion | shortcut n / success / deaths | no route n / deaths / timeouts | success | if far timeouts rescued |
|---|---:|---:|---|---:|---|---|---:|---:|
| start | 1 (0.00) | 0 | 1 / 0 | 0.000 | 291 / 73 / 217 | 8 / 5 / 3 | 0.243 | 0.247 |
| O/seed_0 | 7 (0.02) | 4 | 3 / 0 | 0.571 | 277 / 71 / 205 | 16 / 8 / 8 | 0.250 | 0.260 |
| O/seed_1 | 9 (0.03) | 7 | 2 / 0 | 0.778 | 283 / 72 / 211 | 8 / 6 / 2 | 0.263 | 0.270 |
| O/seed_2 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 285 / 74 / 211 | 14 / 11 / 3 | 0.250 | 0.250 |
| CF/seed_0 | 32 (0.11) | 20 | 12 / 0 | 0.625 | 252 / 60 / 182 | 16 / 6 / 10 | 0.267 | 0.307 |
| CF/seed_1 | 50 (0.17) | 38 | 12 / 0 | 0.760 | 217 / 58 / 156 | 33 / 14 / 19 | 0.320 | 0.360 |
| CF/seed_2 | 150 (0.50) | 112 | 38 / 0 | 0.747 | 111 / 32 / 75 | 39 / 3 / 36 | 0.480 | 0.607 |
| O/seed_0@critic_clip0.1 | 8 (0.03) | 5 | 3 / 0 | 0.625 | 286 / 72 / 214 | 6 / 3 / 3 | 0.257 | 0.267 |
| O/seed_1@critic_clip0.1 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 290 / 73 / 217 | 9 / 6 / 3 | 0.247 | 0.247 |
| O/seed_2@critic_clip0.1 | 7 (0.02) | 5 | 2 / 0 | 0.714 | 284 / 73 / 211 | 9 / 5 / 4 | 0.260 | 0.267 |
| CF/seed_0@critic_clip0.1 | 170 (0.57) | 111 | 59 / 0 | 0.653 | 86 / 26 / 52 | 44 / 6 / 38 | 0.457 | 0.653 |
| CF/seed_1@critic_clip0.1 | 155 (0.52) | 122 | 33 / 0 | 0.787 | 123 / 32 / 89 | 22 / 3 / 19 | 0.513 | 0.623 |
| CF/seed_2@critic_clip0.1 | 108 (0.36) | 87 | 21 / 0 | 0.806 | 166 / 45 / 113 | 26 / 8 / 18 | 0.440 | 0.510 |

The last column is bookkeeping (every far-route timeout counted as a success, nothing else changed), an upper bound for fixing the far-route walking alone.

## 2. (a) Entered too late?

Completion time from the corner (steps) of the successful far-route episodes:

| policy | n | min | p10 | median | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| CF_s0@critic_clip0.1 | 121 | 277 | 348 | 434 | 544 | 690 |
| CF_s1@critic_clip0.1 | 108 | 254 | 300 | 402 | 482 | 680 |
| CF_s2@critic_clip0.1 | 90 | 268 | 296 | 368 | 492 | 616 |
| pooled_clipped_CF | 319 | 254 | 308 | 407 | 511 | 690 |
| CF_s0 | 21 | 280 | 304 | 370 | 496 | 526 |
| CF_s1 | 36 | 270 | 277 | 328 | 420 | 489 |
| CF_s2 | 118 | 262 | 276 | 317 | 430 | 668 |

"Too late" line = steps left at the corner below the pooled p10 of the successful completions = 308 steps (below the minimum 254: certainly too late).

| policy | far route | success | timeout | too late (p10) | too late (min) | steps left at the corner, timeouts: min / median / max | corner reached, successes: median / p90 | corner reached, timeouts: median / p90 |
|---|---:|---:|---:|---:|---:|---|---|---|
| CF_s0@critic_clip0.1 | 177 | 121 | 56 | 1 | 1 | 128 / 712 / 740 | 74 / 101 | 88 / 223 |
| CF_s1@critic_clip0.1 | 157 | 108 | 49 | 3 | 3 | 85 / 720 / 742 | 78 / 108 | 80 / 295 |
| CF_s2@critic_clip0.1 | 106 | 90 | 16 | 4 | 2 | 22 / 706 / 735 | 76 / 93 | 94 / 594 |

## 3. (b) Time enough: how the timeouts end

Classes of diag_v6_pilot_trajectories (last 100 steps); arc = progress along the far route (0-8 west column, 8-32 top corridor, 32-40 east column).

| policy | class | n | mean max arc | mean end arc | stall onset after the corner (median) | stall length (median) | end regions |
|---|---|---:|---:|---:|---:|---:|---|
| CF_s0@critic_clip0.1 | stopped | 33 | 7.3 | 7.2 | 46 | 629 | top_west_corner 29, west_column 3, top_corridor 1 |
| CF_s0@critic_clip0.1 | fallen | 11 | 31.4 | 31.0 | 336 | 384 | east_column 5, goal_area 3, top_corridor 2, top_west_corner 1 |
| CF_s0@critic_clip0.1 | slow_but_moving | 5 | 27.6 | 27.6 | 709 | 2 | east_column 2, top_corridor 2, goal_area 1 |
| CF_s0@critic_clip0.1 | wall_stuck | 4 | 34.3 | 33.1 | 466 | 267 | east_column 4 |
| CF_s0@critic_clip0.1 | oscillating | 2 | 29.3 | 29.1 | 508 | 217 | east_column 1, goal_area 1 |
| CF_s1@critic_clip0.1 | fallen | 19 | 32.9 | 32.2 | 280 | 442 | east_column 16, top_west_corner 1, top_corridor 1, goal_area 1 |
| CF_s1@critic_clip0.1 | stopped | 12 | 16.5 | 16.5 | 162 | 468 | top_west_corner 6, goal_area 5, east_column 1 |
| CF_s1@critic_clip0.1 | oscillating | 9 | 27.4 | 27.0 | 683 | 35 | east_column 5, top_west_corner 2, top_corridor 2 |
| CF_s1@critic_clip0.1 | slow_but_moving | 5 | 18.3 | 18.3 | 507 | 1 | top_corridor 5 |
| CF_s1@critic_clip0.1 | wall_stuck | 1 | 34.3 | 34.2 | 707 | 9 | east_column 1 |
| CF_s2@critic_clip0.1 | slow_but_moving | 4 | 24.8 | 24.7 | 547 | 10 | top_corridor 3, goal_area 1 |
| CF_s2@critic_clip0.1 | stopped | 4 | 32.0 | 32.0 | 308 | 422 | goal_area 4 |
| CF_s2@critic_clip0.1 | fallen | 2 | 38.4 | 38.1 | 386 | 338 | east_column 1, goal_area 1 |
| CF_s2@critic_clip0.1 | wall_stuck | 1 | 10.3 | 10.1 | 592 | 111 | top_corridor 1 |
| CF_s2@critic_clip0.1 | oscillating | 1 | 29.5 | 29.3 | 714 | 3 | top_corridor 1 |

## 4. (c) Entrance state at the corner: timeouts vs successes

AUC = P(timeout value > success value); 0.5 = indistinguishable.

| policy | feature | timeout mean | success mean | AUC |
|---|---|---:|---:|---:|
| CF_s0@critic_clip0.1 | torso_z | 0.562 | 0.546 | 0.52 |
| CF_s0@critic_clip0.1 | up_z | 0.980 | 0.986 | 0.38 |
| CF_s0@critic_clip0.1 | speed | 1.143 | 1.272 | 0.45 |
| CF_s0@critic_clip0.1 | heading_x | 0.024 | 0.057 | 0.49 |
| CF_s0@critic_clip0.1 | joint_speed | 1.192 | 1.447 | 0.39 |
| CF_s1@critic_clip0.1 | torso_z | 0.544 | 0.537 | 0.53 |
| CF_s1@critic_clip0.1 | up_z | 0.986 | 0.986 | 0.50 |
| CF_s1@critic_clip0.1 | speed | 1.189 | 1.255 | 0.47 |
| CF_s1@critic_clip0.1 | heading_x | 0.193 | 0.082 | 0.56 |
| CF_s1@critic_clip0.1 | joint_speed | 1.362 | 1.454 | 0.44 |
| CF_s2@critic_clip0.1 | torso_z | 0.570 | 0.545 | 0.64 |
| CF_s2@critic_clip0.1 | up_z | 0.983 | 0.986 | 0.48 |
| CF_s2@critic_clip0.1 | speed | 1.199 | 1.303 | 0.43 |
| CF_s2@critic_clip0.1 | heading_x | 0.309 | 0.022 | 0.67 |
| CF_s2@critic_clip0.1 | joint_speed | 1.649 | 1.549 | 0.51 |

## 5. The continuation test: same entrance state, same remaining time

Control fidelity: the clipped CF from its own corner entrance reproduces the recorded timeout in 82 / 242 cases (reached 160, died 0).

| seed | entrance | from | continuation | n | reach | timeout | death | reach: too late / time enough (n) |
|---|---|---|---|---:|---:|---:|---:|---|
| 0 | top | timeout entrances | CF clip (control) | 56 | 0.68 | 0.32 | 0.00 | 0.0 / 0.69 (1 / 55) |
| 0 | top | timeout entrances | start | 56 | 0.66 | 0.34 | 0.00 | 0.0 / 0.67 (1 / 55) |
| 0 | top | timeout entrances | O clip | 56 | 0.66 | 0.34 | 0.00 | 0.0 / 0.67 (1 / 55) |
| 0 | top | timeout entrances | CF base | 56 | 0.70 | 0.30 | 0.00 | 0.0 / 0.71 (1 / 55) |
| 0 | top | timeout entrances | driver | 56 | 0.77 | 0.23 | 0.00 | 0.0 / 0.78 (1 / 55) |
| 0 | top | success entrances | start | 121 | 0.64 | 0.36 | 0.00 |  |
| 0 | top | success entrances | O clip | 121 | 0.70 | 0.30 | 0.00 |  |
| 0 | top | success entrances | CF base | 121 | 0.79 | 0.21 | 0.00 |  |
| 0 | top | success entrances | driver | 121 | 0.98 | 0.02 | 0.00 |  |
| 0 | north | timeout entrances | CF clip (control) | 56 | 0.68 | 0.32 | 0.00 |  |
| 0 | north | timeout entrances | start | 56 | 0.52 | 0.48 | 0.00 |  |
| 0 | north | timeout entrances | O clip | 56 | 0.77 | 0.23 | 0.00 |  |
| 0 | north | timeout entrances | CF base | 56 | 0.64 | 0.36 | 0.00 |  |
| 0 | north | timeout entrances | driver | 56 | 0.96 | 0.00 | 0.04 |  |
| 0 | north | success entrances | start | 121 | 0.76 | 0.24 | 0.00 |  |
| 0 | north | success entrances | O clip | 121 | 0.75 | 0.25 | 0.00 |  |
| 0 | north | success entrances | CF base | 121 | 0.66 | 0.34 | 0.00 |  |
| 0 | north | success entrances | driver | 121 | 0.98 | 0.02 | 0.00 |  |
| 1 | top | timeout entrances | CF clip (control) | 49 | 0.65 | 0.35 | 0.00 | 0.33 / 0.67 (3 / 46) |
| 1 | top | timeout entrances | start | 49 | 0.61 | 0.39 | 0.00 | 0.33 / 0.63 (3 / 46) |
| 1 | top | timeout entrances | O clip | 49 | 0.65 | 0.35 | 0.00 | 0.33 / 0.67 (3 / 46) |
| 1 | top | timeout entrances | CF base | 49 | 0.59 | 0.41 | 0.00 | 0.33 / 0.61 (3 / 46) |
| 1 | top | timeout entrances | driver | 49 | 0.78 | 0.22 | 0.00 | 0.0 / 0.83 (3 / 46) |
| 1 | top | success entrances | start | 108 | 0.78 | 0.22 | 0.00 |  |
| 1 | top | success entrances | O clip | 108 | 0.82 | 0.18 | 0.00 |  |
| 1 | top | success entrances | CF base | 108 | 0.77 | 0.23 | 0.00 |  |
| 1 | top | success entrances | driver | 108 | 0.92 | 0.08 | 0.00 |  |
| 1 | north | timeout entrances | CF clip (control) | 49 | 0.65 | 0.35 | 0.00 |  |
| 1 | north | timeout entrances | start | 49 | 0.49 | 0.51 | 0.00 |  |
| 1 | north | timeout entrances | O clip | 49 | 0.69 | 0.31 | 0.00 |  |
| 1 | north | timeout entrances | CF base | 49 | 0.53 | 0.47 | 0.00 |  |
| 1 | north | timeout entrances | driver | 49 | 0.92 | 0.08 | 0.00 |  |
| 1 | north | success entrances | start | 108 | 0.70 | 0.30 | 0.00 |  |
| 1 | north | success entrances | O clip | 108 | 0.83 | 0.17 | 0.00 |  |
| 1 | north | success entrances | CF base | 108 | 0.76 | 0.24 | 0.00 |  |
| 1 | north | success entrances | driver | 108 | 0.98 | 0.01 | 0.01 |  |
| 2 | top | timeout entrances | CF clip (control) | 16 | 0.62 | 0.38 | 0.00 | 0.25 / 0.75 (4 / 12) |
| 2 | top | timeout entrances | start | 16 | 0.56 | 0.44 | 0.00 | 0.25 / 0.67 (4 / 12) |
| 2 | top | timeout entrances | O clip | 16 | 0.62 | 0.38 | 0.00 | 0.25 / 0.75 (4 / 12) |
| 2 | top | timeout entrances | CF base | 16 | 0.56 | 0.44 | 0.00 | 0.25 / 0.67 (4 / 12) |
| 2 | top | timeout entrances | driver | 16 | 0.69 | 0.31 | 0.00 | 0.0 / 0.92 (4 / 12) |
| 2 | top | success entrances | start | 90 | 0.70 | 0.29 | 0.01 |  |
| 2 | top | success entrances | O clip | 90 | 0.73 | 0.26 | 0.01 |  |
| 2 | top | success entrances | CF base | 90 | 0.66 | 0.33 | 0.01 |  |
| 2 | top | success entrances | driver | 90 | 0.89 | 0.10 | 0.01 |  |
| 2 | north | timeout entrances | CF clip (control) | 16 | 0.62 | 0.38 | 0.00 |  |
| 2 | north | timeout entrances | start | 16 | 0.56 | 0.44 | 0.00 |  |
| 2 | north | timeout entrances | O clip | 16 | 0.62 | 0.38 | 0.00 |  |
| 2 | north | timeout entrances | CF base | 16 | 0.56 | 0.44 | 0.00 |  |
| 2 | north | timeout entrances | driver | 16 | 0.94 | 0.06 | 0.00 |  |
| 2 | north | success entrances | start | 90 | 0.71 | 0.28 | 0.01 |  |
| 2 | north | success entrances | O clip | 90 | 0.70 | 0.29 | 0.01 |  |
| 2 | north | success entrances | CF base | 90 | 0.68 | 0.31 | 0.01 |  |
| 2 | north | success entrances | driver | 90 | 0.93 | 0.03 | 0.03 |  |
| pooled | top | timeout entrances | CF clip (control) | 121 | 0.66 | 0.34 | 0.00 |  |
| pooled | top | timeout entrances | start | 121 | 0.63 | 0.37 | 0.00 |  |
| pooled | top | timeout entrances | O clip | 121 | 0.65 | 0.35 | 0.00 |  |
| pooled | top | timeout entrances | CF base | 121 | 0.64 | 0.36 | 0.00 |  |
| pooled | top | timeout entrances | driver | 121 | 0.76 | 0.24 | 0.00 |  |
| pooled | top | success entrances | start | 319 | 0.71 | 0.29 | 0.00 |  |
| pooled | top | success entrances | O clip | 319 | 0.75 | 0.24 | 0.00 |  |
| pooled | top | success entrances | CF base | 319 | 0.75 | 0.25 | 0.00 |  |
| pooled | top | success entrances | driver | 319 | 0.93 | 0.07 | 0.00 |  |
| pooled | north | timeout entrances | CF clip (control) | 121 | 0.66 | 0.34 | 0.00 |  |
| pooled | north | timeout entrances | start | 121 | 0.51 | 0.49 | 0.00 |  |
| pooled | north | timeout entrances | O clip | 121 | 0.72 | 0.28 | 0.00 |  |
| pooled | north | timeout entrances | CF base | 121 | 0.59 | 0.41 | 0.00 |  |
| pooled | north | timeout entrances | driver | 121 | 0.94 | 0.04 | 0.02 |  |
| pooled | north | success entrances | start | 319 | 0.73 | 0.27 | 0.00 |  |
| pooled | north | success entrances | O clip | 319 | 0.76 | 0.23 | 0.00 |  |
| pooled | north | success entrances | CF base | 319 | 0.70 | 0.30 | 0.00 |  |
| pooled | north | success entrances | driver | 319 | 0.97 | 0.02 | 0.01 |  |

## 6. Reading (rules R1-R3 of the docstring, corner entrance)

* seed 0: from the timeout entrances (n = 56) the start policy reaches 0.66, O clip 0.66, CF base 0.70, the driver 0.77, the clipped CF itself 0.68; from the success entrances (n = 121) start 0.64, O clip 0.70, CF base 0.79, driver 0.98.
* seed 1: from the timeout entrances (n = 49) the start policy reaches 0.61, O clip 0.65, CF base 0.59, the driver 0.78, the clipped CF itself 0.65; from the success entrances (n = 108) start 0.78, O clip 0.82, CF base 0.77, driver 0.92.
* seed 2: from the timeout entrances (n = 16) the start policy reaches 0.56, O clip 0.62, CF base 0.56, the driver 0.69, the clipped CF itself 0.62; from the success entrances (n = 90) start 0.70, O clip 0.73, CF base 0.66, driver 0.89.
* seed pooled: from the timeout entrances (n = 121) the start policy reaches 0.63, O clip 0.65, CF base 0.64, the driver 0.76, the clipped CF itself 0.66; from the success entrances (n = 319) start 0.71, O clip 0.75, CF base 0.75, driver 0.93.

R1: driver fails from a timeout entrance -> not completable in the time left.  R2: start reaches from the timeout entrances at about its success-entrance rate while the clipped CF does not -> the updated policy's execution after the entrance.  R3: start falls well below its success-entrance rate -> the entrance (state / time) carries the difference.  The numbers above are the evidence; the sentence is written after them in the SUMMARY.

## 7. The no-route timeouts (neither the corner nor a hazard zone reached)

| policy | n | turned north (y >= 2) | end regions | classes | stall onset (median) |
|---|---:|---:|---|---|---:|
| CF_s0@critic_clip0.1 | 33 | 18 | shortcut_corridor 16, west_column 10, start 7 | stopped 26, fallen 5, wrong_direction 1, oscillating 1 | 157 |
| CF_s1@critic_clip0.1 | 16 | 11 | start 7, west_column 6, shortcut_corridor 3 | stopped 9, fallen 6, oscillating 1 | 114 |
| CF_s2@critic_clip0.1 | 20 | 14 | west_column 10, shortcut_corridor 6, start 4 | stopped 10, slow_but_moving 3, oscillating 3, fallen 3, wrong_direction 1 | 134 |
