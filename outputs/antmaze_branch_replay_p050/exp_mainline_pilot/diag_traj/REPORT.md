# Frozen-model trajectory diagnostic: where the CF timeouts stop, and where the route signal acts

Same 300 evaluation episodes (seed 3909, mode policy) replayed with full capture; every variant of an episode reuses its recorded latents, clocks, rock jitter and initial pose.  `rollout_check.json`, `timeouts.json`, `fork.json`, `cont.json`.

## Replay fidelity

| policy | episodes matching the eval JSON (of 300) | hidden draws identical | success | timeout |
|---|---|---|---|---|
| start | 92/300 | True | 0.227 | 0.033 |
| O_s0 | 122/300 | True | 0.263 | 0.030 |
| O_s1 | 107/300 | True | 0.260 | 0.017 |
| O_s2 | 115/300 | True | 0.250 | 0.010 |
| CF_s0 | 84/300 | True | 0.280 | 0.093 |
| CF_s1 | 88/300 | True | 0.323 | 0.110 |
| CF_s2 | 61/300 | True | 0.497 | 0.243 |

## 1. Where the timeouts stop

Class: fallen (torso down / flipped), stopped (last 100 steps: net < 0.5, path < 3), wall_stuck (net < 1, clearance < 0.6), oscillating (net < 1 otherwise), wrong_direction (progress falling), slow_but_moving.  arc = progress along the route (shortcut 0-24; detour 0-8 west column, 8-32 top corridor, 32-40 east column).

| policy | route label | class | n | mean max arc | mean stall length (steps) | torso z (last 100) | clearance | end regions |
|---|---|---|---|---|---|---|---|---|
| start | None | stopped | 4 | 1.4 | 597 | 0.38 | 1.34 | shortcut_corridor 2, start 2 |
| start | None | fallen | 1 | 2.5 | 734 | 0.27 | 1.46 | west_column 1 |
| start | shortcut | stopped | 5 | 24.0 | 570 | 0.38 | 1.79 | goal_area 5 |
| O_s0 | None | stopped | 7 | 0.8 | 782 | 0.41 | 1.71 | start 7 |
| O_s0 | detour | wall_stuck | 1 | 34.7 | 46 | 0.52 | 0.42 | east_column 1 |
| O_s0 | detour | oscillating | 1 | 21.6 | 27 | 0.53 | 0.88 | top_corridor 1 |
| O_s1 | None | stopped | 4 | 0.8 | 779 | 0.38 | 1.41 | start 4 |
| O_s1 | detour | stopped | 1 | 24.0 | 476 | 0.38 | 1.90 | goal_area 1 |
| O_s2 | None | stopped | 3 | 1.6 | 623 | 0.38 | 0.90 | start 2, west_column 1 |
| CF_s0 | shortcut | stopped | 7 | 15.1 | 578 | 0.38 | 1.24 | shortcut_corridor 7 |
| CF_s0 | shortcut | fallen | 1 | 20.5 | 586 | 0.27 | 1.69 | shortcut_corridor 1 |
| CF_s0 | None | stopped | 6 | 4.2 | 627 | 0.44 | 1.52 | start 3, west_column 2, shortcut_corridor 1 |
| CF_s0 | None | slow_but_moving | 2 | 2.6 | 651 | 0.55 | 0.92 | start 1, west_column 1 |
| CF_s0 | None | wall_stuck | 4 | 2.8 | 401 | 0.55 | 0.50 | start 4 |
| CF_s0 | None | fallen | 1 | 4.8 | 737 | 0.27 | 1.61 | west_column 1 |
| CF_s0 | None | oscillating | 3 | 3.7 | 262 | 0.55 | 0.86 | start 2, west_column 1 |
| CF_s0 | detour | stopped | 3 | 9.7 | 650 | 0.51 | 0.98 | top_west_corner 2, top_corridor 1 |
| CF_s0 | detour | slow_but_moving | 1 | 35.9 | 0 | 0.53 | 1.18 | east_column 1 |
| CF_s1 | None | stopped | 10 | 2.3 | 708 | 0.44 | 1.16 | start 7, shortcut_corridor 3 |
| CF_s1 | None | oscillating | 5 | 2.7 | 274 | 0.50 | 1.00 | start 4, west_column 1 |
| CF_s1 | None | slow_but_moving | 4 | 3.3 | 320 | 0.52 | 0.92 | shortcut_corridor 2, start 1, west_column 1 |
| CF_s1 | None | fallen | 1 | 1.8 | 478 | 0.27 | 1.56 | start 1 |
| CF_s1 | detour | stopped | 3 | 8.0 | 500 | 0.50 | 1.29 | top_west_corner 3 |
| CF_s1 | detour | fallen | 3 | 36.8 | 386 | 0.27 | 1.84 | east_column 2, goal_area 1 |
| CF_s1 | detour | wall_stuck | 1 | 10.3 | 245 | 0.58 | 0.34 | top_west_corner 1 |
| CF_s1 | detour | slow_but_moving | 3 | 22.4 | 0 | 0.54 | 1.31 | top_corridor 1, top_west_corner 1, east_column 1 |
| CF_s1 | detour | oscillating | 2 | 9.2 | 101 | 0.55 | 0.93 | top_corridor 1, west_column 1 |
| CF_s1 | shortcut | stopped | 1 | 6.9 | 335 | 0.38 | 1.20 | shortcut_corridor 1 |
| CF_s2 | detour | slow_but_moving | 14 | 19.9 | 8 | 0.55 | 0.97 | top_corridor 10, east_column 2, goal_area 1, top_west_corner 1 |
| CF_s2 | detour | oscillating | 9 | 15.8 | 78 | 0.53 | 0.83 | top_west_corner 4, top_corridor 3, east_column 2 |
| CF_s2 | detour | fallen | 10 | 25.1 | 396 | 0.29 | 1.38 | east_column 4, top_corridor 3, goal_area 2, west_column 1 |
| CF_s2 | detour | stopped | 9 | 32.9 | 416 | 0.60 | 1.63 | goal_area 9 |
| CF_s2 | None | stopped | 14 | 3.4 | 618 | 0.42 | 1.28 | start 8, shortcut_corridor 6 |
| CF_s2 | None | fallen | 5 | 3.1 | 674 | 0.29 | 1.40 | west_column 4, start 1 |
| CF_s2 | None | oscillating | 7 | 5.2 | 221 | 0.57 | 1.15 | west_column 7 |
| CF_s2 | None | slow_but_moving | 2 | 5.0 | 362 | 0.54 | 1.22 | start 1, west_column 1 |
| CF_s2 | shortcut | stopped | 3 | 24.9 | 554 | 0.38 | 0.95 | shortcut_corridor 2, goal_area 1 |

Torso height reference (mean over successful episodes): start 0.54, O_s0 0.54, O_s1 0.54, O_s2 0.54, CF_s0 0.54, CF_s1 0.54, CF_s2 0.54

### CF_s0: every timeout

| ep | route | U | class | end region | end xy | max arc | stall onset | stall len | net 100 | path 100 | clearance | torso z | up z | first y>=2 | first y>=6 | first east col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | shortcut | U00 | stopped | shortcut_corridor | 17.9,0.4 | 18.0 | 180 | 620 | 0.00 | 0.0 | 1.57 | 0.38 | 1.00 | -1 | -1 | -1 |
| 20 | None | U10 | stopped | west_column | 0.7,4.9 | 5.5 | 58 | 742 | 0.00 | 0.0 | 1.32 | 0.52 | 0.99 | 23 | -1 | -1 |
| 24 | None | U01 | slow_but_moving | west_column | 1.0,2.8 | 3.0 | 53 | 747 | 1.97 | 9.0 | 1.02 | 0.55 | 1.00 | 28 | -1 | -1 |
| 29 | shortcut | U00 | stopped | shortcut_corridor | 4.3,0.7 | 7.2 | 89 | 711 | 0.00 | 0.0 | 1.30 | 0.38 | 1.00 | -1 | -1 | -1 |
| 32 | shortcut | U00 | fallen | shortcut_corridor | 18.7,0.3 | 20.5 | 214 | 586 | 0.00 | 0.0 | 1.69 | 0.27 | -0.94 | -1 | -1 | -1 |
| 36 | None | U10 | wall_stuck | start | 1.6,1.4 | 2.0 | 758 | 42 | 0.25 | 8.2 | 0.55 | 0.55 | 0.99 | 30 | -1 | -1 |
| 59 | None | U00 | fallen | west_column | 0.4,3.0 | 4.8 | 63 | 737 | 0.00 | 0.0 | 1.61 | 0.27 | -0.94 | 25 | -1 | -1 |
| 61 | None | U01 | stopped | start | -0.4,-0.9 | 4.1 | 100 | 700 | 0.00 | 0.0 | 1.62 | 0.41 | 1.00 | 25 | -1 | -1 |
| 74 | shortcut | U00 | stopped | shortcut_corridor | 19.1,0.7 | 19.1 | 217 | 583 | 0.00 | 0.0 | 1.27 | 0.38 | 1.00 | -1 | -1 | -1 |
| 93 | None | U00 | stopped | west_column | 1.0,3.4 | 4.1 | 148 | 652 | 0.00 | 0.0 | 1.04 | 0.54 | 0.99 | 44 | -1 | -1 |
| 97 | shortcut | U01 | stopped | shortcut_corridor | 15.1,0.8 | 15.5 | 572 | 228 | 0.00 | 0.0 | 1.23 | 0.38 | 1.00 | 24 | -1 | -1 |
| 104 | None | U10 | oscillating | west_column | 1.0,3.6 | 5.6 | 179 | 621 | 0.25 | 6.6 | 1.01 | 0.56 | 0.98 | 30 | -1 | -1 |
| 107 | None | U00 | wall_stuck | start | 1.7,1.9 | 2.7 | 170 | 630 | 0.31 | 9.3 | 0.28 | 0.56 | 0.99 | 153 | -1 | -1 |
| 114 | shortcut | U00 | stopped | shortcut_corridor | 18.2,0.8 | 18.3 | 182 | 618 | 0.00 | 0.0 | 1.25 | 0.38 | 1.00 | -1 | -1 | -1 |
| 145 | shortcut | U00 | stopped | shortcut_corridor | 15.6,1.1 | 16.1 | 174 | 626 | 0.00 | 0.0 | 0.93 | 0.38 | 1.00 | -1 | -1 | -1 |
| 177 | None | U00 | oscillating | start | 1.8,1.3 | 3.3 | 666 | 134 | 0.25 | 8.9 | 0.67 | 0.54 | 0.99 | 21 | -1 | -1 |
| 197 | detour | U10 | stopped | top_west_corner | 1.9,6.3 | 10.2 | 102 | 698 | 0.00 | 0.0 | 0.31 | 0.52 | 0.99 | 26 | 65 | -1 |
| 199 | None | U10 | stopped | start | -0.1,-0.9 | 4.6 | 115 | 685 | 0.00 | 0.0 | 1.93 | 0.40 | 1.00 | 22 | -1 | -1 |
| 216 | None | U00 | wall_stuck | start | 1.4,1.6 | 4.1 | 105 | 695 | 0.20 | 8.2 | 0.59 | 0.54 | 0.98 | 42 | -1 | -1 |
| 227 | detour | U10 | slow_but_moving | east_column | 24.8,4.1 | 35.9 | 800 | 0 | 9.57 | 12.6 | 1.18 | 0.53 | 0.98 | 187 | 549 | 757 |
| 230 | detour | U10 | stopped | top_corridor | 2.7,6.7 | 10.7 | 212 | 588 | 0.00 | 0.0 | 0.67 | 0.52 | 0.99 | 27 | 82 | -1 |
| 251 | shortcut | U01 | stopped | shortcut_corridor | 11.7,0.9 | 12.0 | 138 | 662 | 0.00 | 0.0 | 1.10 | 0.38 | 1.00 | -1 | -1 | -1 |
| 253 | None | U10 | oscillating | start | 1.8,1.1 | 2.2 | 768 | 32 | 0.25 | 9.8 | 0.89 | 0.56 | 0.99 | 23 | -1 | -1 |
| 255 | None | U01 | stopped | shortcut_corridor | 3.6,0.7 | 3.7 | 560 | 240 | 0.00 | 0.0 | 1.34 | 0.38 | 1.00 | 33 | -1 | -1 |
| 262 | None | U01 | stopped | start | 0.1,-0.8 | 3.0 | 58 | 742 | 0.00 | 0.0 | 1.87 | 0.40 | 1.00 | 25 | -1 | -1 |
| 273 | None | U01 | slow_but_moving | start | 1.4,1.2 | 2.1 | 245 | 555 | 1.65 | 7.9 | 0.82 | 0.55 | 0.99 | 26 | -1 | -1 |
| 283 | detour | U00 | stopped | top_west_corner | -0.9,8.0 | 8.0 | 135 | 665 | 0.00 | 0.0 | 1.97 | 0.50 | 0.99 | 26 | 92 | -1 |
| 284 | None | U01 | wall_stuck | start | 1.7,1.4 | 2.4 | 563 | 237 | 0.51 | 9.4 | 0.60 | 0.55 | 1.00 | 24 | -1 | -1 |

### CF_s1: every timeout

| ep | route | U | class | end region | end xy | max arc | stall onset | stall len | net 100 | path 100 | clearance | torso z | up z | first y>=2 | first y>=6 | first east col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | None | U11 | stopped | start | -1.0,-0.9 | 0.0 | 2 | 798 | 0.00 | 0.0 | 1.07 | 0.46 | 0.97 | 51 | -1 | -1 |
| 19 | detour | U01 | stopped | top_west_corner | -0.9,8.5 | 8.0 | 108 | 692 | 0.00 | 0.0 | 1.51 | 0.50 | 0.99 | 36 | 91 | -1 |
| 24 | detour | U01 | fallen | east_column | 23.9,4.3 | 36.3 | 403 | 397 | 0.00 | 0.0 | 1.89 | 0.27 | -0.94 | 42 | 81 | 293 |
| 28 | detour | U00 | wall_stuck | top_west_corner | 1.7,6.2 | 10.3 | 555 | 245 | 0.25 | 5.3 | 0.34 | 0.58 | 0.99 | 51 | 98 | -1 |
| 31 | None | U10 | oscillating | start | -0.9,-0.9 | 2.9 | 285 | 515 | 0.69 | 2.0 | 1.09 | 0.47 | 0.97 | 196 | -1 | -1 |
| 36 | None | U10 | slow_but_moving | start | 1.8,1.5 | 1.8 | 800 | 0 | 1.51 | 9.0 | 0.48 | 0.52 | 0.92 | 21 | -1 | -1 |
| 47 | shortcut | U10 | stopped | shortcut_corridor | 6.4,0.8 | 6.9 | 465 | 335 | 0.00 | 0.0 | 1.20 | 0.38 | 1.00 | 26 | -1 | -1 |
| 56 | detour | U00 | stopped | top_west_corner | -0.8,9.3 | 8.0 | 322 | 478 | 0.00 | 0.0 | 1.20 | 0.50 | 0.99 | 43 | 297 | -1 |
| 61 | None | U01 | stopped | start | -0.8,-0.9 | 1.7 | 50 | 750 | 0.00 | 0.0 | 1.23 | 0.46 | 0.97 | -1 | -1 | -1 |
| 71 | None | U11 | slow_but_moving | shortcut_corridor | 2.3,1.2 | 2.3 | 146 | 654 | 1.28 | 9.3 | 0.78 | 0.53 | 0.99 | 82 | -1 | -1 |
| 81 | None | U10 | slow_but_moving | shortcut_corridor | 2.7,0.9 | 3.6 | 176 | 624 | 1.20 | 10.2 | 1.07 | 0.51 | 1.00 | -1 | -1 | -1 |
| 100 | detour | U01 | fallen | goal_area | 24.0,1.0 | 39.4 | 507 | 293 | 0.09 | 0.5 | 1.96 | 0.27 | -0.79 | 34 | 81 | 281 |
| 104 | None | U10 | stopped | start | -0.6,-0.8 | 1.7 | 361 | 439 | 0.00 | 0.0 | 1.39 | 0.46 | 0.97 | -1 | -1 | -1 |
| 133 | None | U10 | oscillating | start | 1.6,1.4 | 2.8 | 672 | 128 | 0.38 | 8.5 | 0.62 | 0.52 | 0.99 | 178 | -1 | -1 |
| 141 | detour | U00 | slow_but_moving | top_west_corner | 1.2,8.5 | 9.2 | 800 | 0 | 2.79 | 7.1 | 1.46 | 0.52 | 1.00 | 48 | 662 | -1 |
| 147 | None | U00 | stopped | shortcut_corridor | 3.8,0.8 | 4.3 | 67 | 733 | 0.00 | 0.0 | 1.25 | 0.38 | 1.00 | -1 | -1 | -1 |
| 151 | None | U11 | stopped | start | -1.0,-1.1 | 0.9 | 31 | 769 | 0.00 | 0.0 | 1.04 | 0.46 | 0.97 | -1 | -1 | -1 |
| 162 | None | U11 | stopped | start | -1.1,-1.2 | 3.0 | 54 | 746 | 0.00 | 0.0 | 0.94 | 0.46 | 0.97 | 29 | -1 | -1 |
| 169 | None | U01 | oscillating | west_column | 0.9,4.8 | 5.8 | 736 | 64 | 0.59 | 6.3 | 1.14 | 0.57 | 0.99 | 31 | -1 | -1 |
| 174 | None | U01 | stopped | shortcut_corridor | 3.8,0.6 | 4.0 | 213 | 587 | 0.00 | 0.0 | 1.42 | 0.38 | 1.00 | -1 | -1 | -1 |
| 180 | detour | U00 | oscillating | west_column | -0.8,5.9 | 8.0 | 602 | 198 | 0.65 | 5.1 | 1.19 | 0.52 | 0.99 | 49 | 564 | -1 |
| 192 | None | U01 | stopped | start | -1.0,-1.4 | 2.1 | 45 | 755 | 0.00 | 0.0 | 1.01 | 0.46 | 0.97 | 42 | -1 | -1 |
| 199 | detour | U10 | stopped | top_west_corner | -0.9,8.8 | 8.0 | 471 | 329 | 0.00 | 0.0 | 1.17 | 0.50 | 0.99 | 28 | 459 | -1 |
| 216 | detour | U00 | oscillating | top_corridor | 2.4,6.7 | 10.5 | 796 | 4 | 0.64 | 5.6 | 0.68 | 0.58 | 0.98 | 34 | 465 | -1 |
| 217 | None | U10 | slow_but_moving | west_column | -0.7,5.5 | 5.5 | 800 | 0 | 1.11 | 7.2 | 1.34 | 0.53 | 1.00 | 560 | -1 | -1 |
| 221 | detour | U01 | fallen | east_column | 24.3,5.8 | 34.6 | 331 | 469 | 0.00 | 0.0 | 1.68 | 0.27 | -0.90 | 23 | 66 | 264 |
| 228 | None | U11 | stopped | shortcut_corridor | 3.1,0.9 | 3.7 | 83 | 717 | 0.00 | 0.0 | 1.13 | 0.38 | 1.00 | -1 | -1 | -1 |
| 253 | detour | U10 | slow_but_moving | east_column | 24.6,3.1 | 36.9 | 800 | 0 | 3.58 | 9.1 | 1.37 | 0.54 | 0.97 | 21 | 459 | 658 |
| 272 | None | U11 | oscillating | start | -1.1,1.3 | 2.0 | 522 | 278 | 0.52 | 7.4 | 0.90 | 0.48 | 0.99 | 521 | -1 | -1 |
| 283 | None | U00 | fallen | start | 1.7,0.4 | 1.8 | 322 | 478 | 0.00 | 0.0 | 1.56 | 0.27 | -0.90 | 30 | -1 | -1 |
| 284 | None | U01 | oscillating | start | -0.8,-0.7 | 0.0 | 414 | 386 | 0.58 | 1.1 | 1.26 | 0.46 | 0.98 | 30 | -1 | -1 |
| 286 | None | U01 | stopped | start | -0.9,-0.9 | 1.1 | 14 | 786 | 0.00 | 0.0 | 1.12 | 0.46 | 0.97 | -1 | -1 | -1 |
| 293 | detour | U01 | slow_but_moving | top_corridor | 13.1,7.1 | 21.1 | 800 | 0 | 10.01 | 11.4 | 1.09 | 0.55 | 0.98 | 31 | 280 | -1 |

### CF_s2: every timeout

| ep | route | U | class | end region | end xy | max arc | stall onset | stall len | net 100 | path 100 | clearance | torso z | up z | first y>=2 | first y>=6 | first east col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | detour | U00 | slow_but_moving | top_corridor | 14.2,6.5 | 22.2 | 800 | 0 | 1.72 | 3.6 | 0.47 | 0.54 | 0.98 | 20 | 77 | -1 |
| 21 | None | U10 | stopped | shortcut_corridor | 2.5,0.8 | 2.5 | 221 | 579 | 0.00 | 0.0 | 1.23 | 0.38 | 1.00 | 33 | -1 | -1 |
| 23 | None | U01 | stopped | start | -0.7,-1.0 | 4.4 | 154 | 646 | 0.03 | 0.2 | 1.33 | 0.45 | 0.97 | 23 | -1 | -1 |
| 27 | detour | U11 | oscillating | top_corridor | 2.0,9.2 | 10.7 | 670 | 130 | 0.34 | 5.0 | 0.83 | 0.52 | 0.99 | 18 | 78 | -1 |
| 29 | detour | U00 | slow_but_moving | east_column | 23.2,5.1 | 35.0 | 795 | 5 | 1.51 | 6.3 | 1.22 | 0.54 | 0.99 | 21 | 64 | 602 |
| 34 | None | U10 | fallen | west_column | 0.4,3.4 | 4.9 | 260 | 540 | 1.04 | 3.1 | 1.64 | 0.38 | -0.91 | 26 | -1 | -1 |
| 39 | None | U01 | fallen | west_column | -0.8,2.2 | 2.3 | 87 | 713 | 0.00 | 0.0 | 1.17 | 0.27 | -0.93 | 56 | -1 | -1 |
| 44 | detour | U01 | slow_but_moving | top_corridor | 3.4,7.1 | 11.4 | 800 | 0 | 1.73 | 5.9 | 1.14 | 0.57 | 0.97 | 52 | 599 | -1 |
| 48 | detour | U00 | fallen | east_column | 24.5,4.8 | 35.7 | 431 | 369 | 0.00 | 0.0 | 1.50 | 0.27 | -0.92 | 25 | 69 | 294 |
| 50 | detour | U10 | slow_but_moving | east_column | 22.1,7.2 | 30.1 | 800 | 0 | 10.40 | 12.1 | 1.16 | 0.54 | 0.98 | 58 | 594 | 799 |
| 51 | detour | U11 | slow_but_moving | top_corridor | 5.8,6.7 | 13.8 | 797 | 3 | 3.05 | 5.5 | 0.65 | 0.54 | 0.98 | 24 | 162 | -1 |
| 52 | None | U11 | stopped | start | -0.7,-0.7 | 3.3 | 374 | 426 | 0.00 | 0.0 | 1.34 | 0.45 | 0.98 | 193 | -1 | -1 |
| 53 | detour | U11 | slow_but_moving | top_corridor | 6.8,7.1 | 15.0 | 766 | 34 | 3.16 | 7.2 | 1.06 | 0.58 | 0.95 | 22 | 92 | -1 |
| 58 | None | U00 | oscillating | west_column | 1.0,5.0 | 5.1 | 782 | 18 | 0.92 | 7.1 | 0.99 | 0.57 | 0.99 | 53 | -1 | -1 |
| 62 | detour | U10 | fallen | top_corridor | 5.4,8.1 | 13.4 | 492 | 308 | 0.00 | 0.0 | 1.86 | 0.27 | -0.92 | 23 | 82 | -1 |
| 65 | detour | U10 | oscillating | top_corridor | 2.0,6.7 | 10.2 | 796 | 4 | 0.33 | 5.8 | 0.66 | 0.59 | 0.98 | 27 | 638 | -1 |
| 66 | None | U11 | oscillating | west_column | 0.9,4.4 | 5.5 | 407 | 393 | 0.53 | 6.6 | 1.06 | 0.59 | 0.99 | 26 | -1 | -1 |
| 68 | detour | U01 | oscillating | top_west_corner | 1.9,9.3 | 10.3 | 746 | 54 | 0.37 | 5.3 | 0.70 | 0.52 | 0.98 | 27 | 102 | -1 |
| 69 | shortcut | U00 | stopped | shortcut_corridor | 20.9,0.7 | 20.9 | 227 | 573 | 0.00 | 0.0 | 1.29 | 0.38 | 1.00 | -1 | -1 | -1 |
| 74 | detour | U00 | stopped | goal_area | 24.4,-0.7 | 40.0 | 413 | 387 | 0.00 | 0.0 | 1.59 | 0.60 | 0.95 | 30 | 116 | 328 |
| 78 | detour | U01 | stopped | goal_area | 24.4,-1.0 | 40.0 | 374 | 426 | 0.00 | 0.0 | 1.65 | 0.60 | 0.95 | 27 | 70 | 283 |
| 88 | None | U11 | fallen | west_column | 0.1,2.2 | 2.2 | 40 | 760 | 0.00 | 0.0 | 1.87 | 0.27 | -0.94 | 36 | -1 | -1 |
| 95 | detour | U00 | stopped | goal_area | 24.9,-0.5 | 24.0 | 373 | 427 | 0.00 | 0.0 | 1.51 | 0.60 | 0.95 | 36 | 81 | 287 |
| 97 | detour | U01 | slow_but_moving | top_corridor | 9.0,7.1 | 17.0 | 800 | 0 | 8.25 | 11.8 | 1.08 | 0.54 | 0.99 | 23 | 714 | -1 |
| 98 | detour | U11 | fallen | east_column | 24.1,5.9 | 34.5 | 346 | 454 | 0.00 | 0.0 | 1.93 | 0.27 | -0.92 | 24 | 70 | 285 |
| 108 | detour | U01 | slow_but_moving | top_corridor | 11.8,7.0 | 19.8 | 800 | 0 | 2.57 | 5.6 | 0.99 | 0.55 | 0.97 | 31 | 361 | -1 |
| 126 | detour | U00 | fallen | east_column | 22.2,7.2 | 34.6 | 356 | 444 | 0.00 | 0.0 | 1.21 | 0.27 | -0.92 | 28 | 66 | 269 |
| 132 | detour | U10 | oscillating | top_corridor | 3.1,7.1 | 11.1 | 800 | 0 | 0.86 | 6.7 | 1.07 | 0.56 | 0.94 | 21 | 63 | -1 |
| 134 | detour | U10 | fallen | goal_area | 24.1,0.3 | 39.9 | 473 | 327 | 0.00 | 0.0 | 1.92 | 0.27 | -0.94 | 24 | 67 | 283 |
| 136 | None | U01 | stopped | shortcut_corridor | 2.6,0.8 | 2.7 | 392 | 408 | 0.00 | 0.0 | 1.19 | 0.38 | 1.00 | 46 | -1 | -1 |
| 146 | None | U10 | fallen | west_column | 0.7,2.2 | 3.2 | 96 | 704 | 0.00 | 0.0 | 1.32 | 0.27 | -0.94 | 61 | -1 | -1 |
| 147 | shortcut | U00 | stopped | shortcut_corridor | 15.2,1.0 | 15.2 | 183 | 617 | 0.00 | 0.0 | 1.02 | 0.38 | 1.00 | -1 | -1 | -1 |
| 149 | detour | U10 | oscillating | top_west_corner | 1.0,9.2 | 9.3 | 783 | 17 | 0.41 | 6.0 | 0.97 | 0.52 | 0.98 | 119 | 364 | -1 |
| 151 | detour | U11 | oscillating | east_column | 22.7,5.5 | 34.5 | 792 | 8 | 0.38 | 6.1 | 0.69 | 0.51 | 0.99 | 33 | 75 | 345 |
| 157 | detour | U11 | slow_but_moving | goal_area | 24.4,-0.3 | 24.0 | 763 | 37 | 5.21 | 9.8 | 1.65 | 0.57 | 0.95 | 146 | 427 | 645 |
| 159 | detour | U01 | stopped | goal_area | 23.9,-0.4 | 40.0 | 360 | 440 | 0.00 | 0.0 | 1.90 | 0.60 | 0.95 | 39 | 84 | 287 |
| 160 | None | U11 | stopped | shortcut_corridor | 3.5,0.6 | 3.7 | 318 | 482 | 0.00 | 0.0 | 1.39 | 0.38 | 1.00 | 36 | -1 | -1 |
| 161 | detour | U11 | stopped | goal_area | 24.1,-0.5 | 40.0 | 356 | 444 | 0.00 | 0.0 | 1.92 | 0.60 | 0.95 | 29 | 74 | 277 |
| 163 | detour | U10 | stopped | goal_area | 24.5,-1.0 | 40.0 | 523 | 277 | 0.00 | 0.0 | 1.47 | 0.60 | 0.95 | 20 | 78 | 439 |
| 164 | None | U10 | slow_but_moving | start | -0.7,-0.9 | 3.9 | 77 | 723 | 1.89 | 3.6 | 1.30 | 0.49 | 0.97 | 25 | -1 | -1 |
| 170 | None | U01 | slow_but_moving | west_column | 0.9,5.9 | 6.0 | 798 | 2 | 2.37 | 7.3 | 1.13 | 0.59 | 0.99 | 30 | -1 | -1 |
| 173 | detour | U00 | slow_but_moving | top_corridor | 3.0,7.2 | 11.0 | 798 | 2 | 1.73 | 5.9 | 1.18 | 0.56 | 0.95 | 19 | 690 | -1 |
| 178 | detour | U01 | slow_but_moving | top_corridor | 12.7,7.0 | 20.7 | 800 | 0 | 10.10 | 11.8 | 1.00 | 0.55 | 0.99 | 22 | 615 | -1 |
| 185 | detour | U01 | slow_but_moving | top_west_corner | 1.7,6.2 | 10.0 | 770 | 30 | 2.01 | 7.3 | 0.27 | 0.56 | 0.99 | 116 | 731 | -1 |
| 186 | None | U10 | stopped | shortcut_corridor | 3.7,0.5 | 3.8 | 145 | 655 | 0.00 | 0.0 | 1.50 | 0.38 | 1.00 | -1 | -1 | -1 |
| 187 | detour | U00 | oscillating | top_west_corner | 1.5,9.1 | 10.6 | 603 | 197 | 0.03 | 5.9 | 0.89 | 0.52 | 0.98 | 40 | 128 | -1 |
| 192 | detour | U01 | fallen | top_corridor | 3.6,6.7 | 12.6 | 732 | 68 | 0.26 | 3.8 | 0.68 | 0.50 | -0.75 | 54 | 448 | -1 |
| 196 | None | U11 | oscillating | west_column | 0.8,4.3 | 4.4 | 798 | 2 | 0.30 | 7.0 | 1.17 | 0.58 | 0.98 | 25 | -1 | -1 |
| 206 | None | U10 | stopped | start | -0.9,-0.9 | 3.6 | 130 | 670 | 0.00 | 0.0 | 1.12 | 0.44 | 0.98 | 36 | -1 | -1 |
| 208 | None | U11 | fallen | start | 1.8,1.0 | 2.7 | 146 | 654 | 0.00 | 0.0 | 1.01 | 0.27 | -0.94 | 23 | -1 | -1 |
| 210 | None | U00 | stopped | start | -0.7,-0.8 | 5.6 | 134 | 666 | 0.00 | 0.0 | 1.30 | 0.43 | 0.99 | 45 | -1 | -1 |
| 215 | None | U11 | oscillating | west_column | 0.9,4.5 | 5.3 | 664 | 136 | 0.45 | 6.3 | 1.08 | 0.56 | 1.00 | 24 | -1 | -1 |
| 218 | detour | U01 | fallen | west_column | 0.2,5.4 | 6.1 | 75 | 725 | 0.00 | 0.0 | 1.82 | 0.27 | -0.90 | 21 | 75 | -1 |
| 219 | None | U00 | stopped | start | -0.8,-0.8 | 5.5 | 261 | 539 | 0.00 | 0.0 | 1.20 | 0.43 | 0.99 | 54 | -1 | -1 |
| 225 | detour | U01 | stopped | goal_area | 24.5,-0.2 | 24.0 | 377 | 423 | 0.00 | 0.0 | 1.80 | 0.60 | 0.95 | 26 | 81 | 298 |
| 228 | detour | U11 | stopped | goal_area | 25.1,-0.8 | 24.0 | 349 | 451 | 0.00 | 0.0 | 1.16 | 0.60 | 0.95 | 23 | 68 | 274 |
| 235 | detour | U01 | fallen | east_column | 24.4,3.3 | 37.2 | 369 | 431 | 0.00 | 0.0 | 1.62 | 0.27 | -0.90 | 29 | 72 | 267 |
| 242 | detour | U00 | stopped | goal_area | 25.0,-0.3 | 24.0 | 332 | 468 | 0.00 | 0.0 | 1.72 | 0.60 | 0.95 | 25 | 66 | 265 |
| 244 | detour | U11 | oscillating | top_west_corner | 1.9,9.0 | 10.6 | 601 | 199 | 0.35 | 5.1 | 1.01 | 0.52 | 0.98 | 33 | 111 | -1 |
| 246 | None | U01 | stopped | start | -0.9,-0.9 | 3.6 | 114 | 686 | 0.00 | 0.0 | 1.11 | 0.46 | 0.97 | 38 | -1 | -1 |
| 256 | shortcut | U00 | stopped | goal_area | 22.5,1.8 | 38.5 | 328 | 472 | 0.00 | 0.0 | 0.53 | 0.38 | 1.00 | -1 | -1 | -1 |
| 257 | None | U00 | oscillating | west_column | 0.7,4.0 | 5.3 | 186 | 614 | 0.57 | 7.4 | 1.32 | 0.56 | 0.99 | 132 | -1 | -1 |
| 261 | detour | U11 | fallen | top_corridor | 5.0,6.5 | 13.1 | 373 | 427 | 0.00 | 0.0 | 0.46 | 0.27 | -0.94 | 29 | 77 | -1 |
| 262 | detour | U01 | oscillating | east_column | 22.6,5.8 | 34.6 | 704 | 96 | 0.06 | 5.6 | 0.60 | 0.49 | 1.00 | 22 | 75 | 303 |
| 265 | detour | U01 | fallen | goal_area | 25.4,-1.2 | 24.0 | 398 | 402 | 0.00 | 0.0 | 0.83 | 0.27 | -0.79 | 39 | 85 | 314 |
| 267 | None | U00 | stopped | shortcut_corridor | 2.8,0.4 | 3.1 | 65 | 735 | 0.00 | 0.0 | 1.62 | 0.38 | 1.00 | -1 | -1 | -1 |
| 270 | None | U11 | oscillating | west_column | 0.8,4.8 | 5.4 | 487 | 313 | 0.43 | 7.2 | 1.25 | 0.57 | 0.99 | 36 | -1 | -1 |
| 281 | detour | U00 | slow_but_moving | top_corridor | 18.1,7.1 | 26.1 | 799 | 1 | 7.92 | 11.3 | 1.09 | 0.56 | 0.97 | 33 | 122 | -1 |
| 286 | detour | U01 | slow_but_moving | top_corridor | 14.1,6.6 | 22.1 | 800 | 0 | 1.36 | 6.3 | 0.56 | 0.56 | 0.98 | 28 | 145 | -1 |
| 287 | None | U11 | stopped | start | -1.0,-0.8 | 0.2 | 7 | 793 | 0.00 | 0.0 | 1.16 | 0.43 | 0.99 | 35 | -1 | -1 |
| 290 | None | U01 | stopped | shortcut_corridor | 2.8,0.7 | 3.3 | 142 | 658 | 0.00 | 0.0 | 1.27 | 0.38 | 1.00 | 29 | -1 | -1 |
| 294 | None | U10 | oscillating | west_column | 0.9,4.5 | 5.1 | 732 | 68 | 0.40 | 7.3 | 1.14 | 0.57 | 0.98 | 33 | -1 | -1 |
| 297 | None | U00 | stopped | start | -0.8,-1.0 | 2.1 | 86 | 714 | 0.01 | 0.1 | 1.15 | 0.46 | 0.98 | 64 | -1 | -1 |

## 2. Where the route signal acts

### 2a. Critic scores of the seven policies' actual reset-state torques (min over the twin heads, task goal; mean over the 300 reset states)

| critic \ torque of | start | O_s0 | O_s1 | O_s2 | CF_s0 | CF_s1 | CF_s2 |
|---|---|---|---|---|---|---|---|
| CF_s0 | -9.97 | -10.05 | -9.98 | -10.03 | -9.25 | -9.56 | -8.70 |
| CF_s1 | -10.81 | -10.28 | -10.23 | -10.31 | -10.01 | -9.70 | -10.01 |
| CF_s2 | -10.04 | -9.98 | -9.93 | -10.00 | -9.25 | -9.27 | -7.83 |
| O_s0 | -8.52 | -8.61 | -8.60 | -8.67 | -8.44 | -8.44 | -8.83 |
| O_s1 | -8.79 | -8.83 | -8.81 | -8.87 | -8.73 | -8.66 | -9.29 |
| O_s2 | -8.18 | -8.36 | -8.35 | -8.40 | -8.27 | -8.15 | -9.30 |
| van | -7.99 | -8.49 | -8.49 | -8.54 | -8.38 | -8.32 | -8.57 |

Per-state preference: share of reset states where the critic scores the CF torque of its own seed above the O torque / above the start torque:

| critic | seed | P(f(CF) > f(O)) | P(f(CF) > f(start)) | P(f(O) > f(start)) |
|---|---|---|---|---|
| CF_s0 | 0 | 0.92 | 0.81 | 0.47 |
| O_s0 | 0 | 0.84 | 0.58 | 0.49 |
| van | 0 | 0.76 | 0.00 | 0.00 |
| CF_s1 | 1 | 0.86 | 0.94 | 0.91 |
| O_s1 | 1 | 0.58 | 0.71 | 0.58 |
| van | 1 | 0.82 | 0.00 | 0.00 |
| CF_s2 | 2 | 1.00 | 1.00 | 0.52 |
| O_s2 | 2 | 0.02 | 0.00 | 0.25 |
| van | 2 | 0.38 | 0.00 | 0.00 |

The CF actor at its own reset states: its mode torque's score vs 64 samples of its own tanh-normal under its own critic (and the O actor likewise):

| actor | f(mode) | mean f(samples) | max f(samples) | frac samples below the mode | policy scale |
|---|---|---|---|---|---|
| CF_s0 | -9.25 | -9.41 | -8.29 | 0.62 | 0.751 |
| O_s0 | -8.61 | -8.63 | -8.36 | 0.57 | 0.490 |
| CF_s1 | -9.70 | -9.84 | -8.76 | 0.60 | 0.738 |
| O_s1 | -8.81 | -8.82 | -8.54 | 0.56 | 0.576 |
| CF_s2 | -7.83 | -8.29 | -7.14 | 0.71 | 1.297 |
| O_s2 | -8.40 | -8.40 | -8.15 | 0.53 | 0.507 |

Log-likelihood of the other policies' torques under each actor (mean over reset states): 

| log pi_actor(torque of ...) | start | O_s0 | O_s1 | O_s2 | CF_s0 | CF_s1 | CF_s2 |
|---|---|---|---|---|---|---|---|
| CF_s0 | -134.3 | +5.7 | +5.6 | +5.0 | +7.9 | +6.0 | -33.1 |
| CF_s1 | -151.6 | +5.9 | +5.7 | +4.7 | +4.4 | +8.7 | -32.7 |
| CF_s2 | -110.2 | -3.3 | -3.8 | -4.1 | -2.7 | -0.8 | +10.2 |
| O_s0 | -519.1 | +13.4 | +10.1 | +11.1 | -2.8 | -1.8 | -187.2 |
| O_s1 | -239.8 | +10.1 | +11.3 | +9.6 | +2.7 | +2.5 | -91.2 |
| O_s2 | -459.7 | +11.3 | +10.2 | +12.2 | -1.2 | -1.1 | -145.0 |

### 2b. First action swapped at the identical reset state and hidden draw (route realised; n = number of episodes)

| seed | continuation | first action from | n | detour | success | death | timeout |
|---|---|---|---|---|---|---|---|
| 0 | start | start | 300 | 0.000 | 0.243 | 0.740 | 0.017 |
| 0 | start | O_s0 | 300 | 0.010 | 0.237 | 0.737 | 0.027 |
| 0 | start | CF_s0 | 300 | 0.020 | 0.240 | 0.693 | 0.067 |
| 0 | O_s0 | start | 300 | 0.170 | 0.273 | 0.463 | 0.263 |
| 0 | O_s0 | O_s0 | 300 | 0.023 | 0.257 | 0.707 | 0.037 |
| 0 | O_s0 | CF_s0 | 300 | 0.070 | 0.267 | 0.620 | 0.113 |
| 0 | CF_s0 | start | 300 | 0.120 | 0.283 | 0.630 | 0.087 |
| 0 | CF_s0 | O_s0 | 300 | 0.043 | 0.237 | 0.703 | 0.060 |
| 0 | CF_s0 | CF_s0 | 300 | 0.103 | 0.277 | 0.630 | 0.093 |
| 1 | start | start | 300 | 0.000 | 0.243 | 0.740 | 0.017 |
| 1 | start | O_s1 | 300 | 0.003 | 0.243 | 0.743 | 0.013 |
| 1 | start | CF_s1 | 300 | 0.043 | 0.237 | 0.627 | 0.137 |
| 1 | O_s1 | start | 300 | 0.190 | 0.327 | 0.487 | 0.187 |
| 1 | O_s1 | O_s1 | 300 | 0.020 | 0.253 | 0.723 | 0.023 |
| 1 | O_s1 | CF_s1 | 300 | 0.100 | 0.280 | 0.577 | 0.143 |
| 1 | CF_s1 | start | 300 | 0.113 | 0.307 | 0.620 | 0.073 |
| 1 | CF_s1 | O_s1 | 300 | 0.037 | 0.260 | 0.713 | 0.027 |
| 1 | CF_s1 | CF_s1 | 300 | 0.160 | 0.323 | 0.567 | 0.110 |
| 2 | start | start | 300 | 0.000 | 0.243 | 0.740 | 0.017 |
| 2 | start | O_s2 | 300 | 0.000 | 0.240 | 0.750 | 0.010 |
| 2 | start | CF_s2 | 300 | 0.143 | 0.257 | 0.567 | 0.177 |
| 2 | O_s2 | start | 300 | 0.113 | 0.277 | 0.593 | 0.130 |
| 2 | O_s2 | O_s2 | 300 | 0.003 | 0.250 | 0.740 | 0.010 |
| 2 | O_s2 | CF_s2 | 300 | 0.140 | 0.307 | 0.543 | 0.150 |
| 2 | CF_s2 | start | 300 | 0.283 | 0.317 | 0.483 | 0.200 |
| 2 | CF_s2 | O_s2 | 300 | 0.133 | 0.257 | 0.600 | 0.143 |
| 2 | CF_s2 | CF_s2 | 300 | 0.523 | 0.450 | 0.257 | 0.293 |

Replay fidelity of the branch primitive (restored episode, one policy end-to-end, outcome+route+steps equal to the rollout): {'start': '23/40', 'CF_s2': '11/40', 'O_s2': '22/40'}

### 2c. Where the O and CF paths of the same episode separate (first step with xy gap > 0.5), and the single-action swap there

| seed | episodes | paths separate | median tau (all) | mean tau | episodes with different routes | median tau (different routes) | median step CF reaches y >= 2 | mean |xy| at tau (CF) |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | 183 | 70 | 66 | 57 | 9 | 26 | 0.7 |
| 1 | 300 | 175 | 12 | 56 | 86 | 9 | 31 | 0.6 |
| 2 | 300 | 284 | 7 | 14 | 193 | 7 | 28 | 0.4 |

| seed | variant | n | detour | success |
|---|---|---|---|---|
| 0 | CF state at tau, CF action, CF continues (control) | 183 | 0.169 | 0.454 |
| 0 | CF state at tau, O action once, CF continues | 183 | 0.148 | 0.404 |
| 0 | O state at tau, CF action once, O continues | 183 | 0.055 | 0.437 |
| 1 | CF state at tau, CF action, CF continues (control) | 175 | 0.274 | 0.526 |
| 1 | CF state at tau, O action once, CF continues | 175 | 0.286 | 0.509 |
| 1 | O state at tau, CF action once, O continues | 175 | 0.046 | 0.411 |
| 2 | CF state at tau, CF action, CF continues (control) | 284 | 0.553 | 0.468 |
| 2 | CF state at tau, O action once, CF continues | 284 | 0.468 | 0.482 |
| 2 | O state at tau, CF action once, O continues | 284 | 0.011 | 0.254 |

### 2d. What the CF branches (the CF critic's positive futures) said at the t = 0 anchors

196 reset-row anchors: 3 branches went around (y >= 6); success around 0.6666666666666666 vs straight 0.29015544041450775 (straight death 0.6735751295336787); mean P_goal (gamma 0.999, r 0.5) around 0.0011407843554060914 vs straight 0.0011393887257296798.

## 3. Continuation from the CF policy's own pre-stall states (its timeout episodes)

prestall = 20 steps before the last progress maximum; enter_detour = the step the CF path first reached y >= 2; t100 = step 100 for the no-route timeouts.  Continuations: the CF policy itself (control), the start d05 policy, the O policy of the seed, the teacher's blind position driver.

| seed | start point | continuation | n | reach | death | timeout | prefix ended early | median total steps (reached) |
|---|---|---|---|---|---|---|---|---|
| 0 | prestall | CF_s0 | 28 | 0.50 | 0.04 | 0.46 | 0.00 | 386 |
| 0 | prestall | start | 28 | 0.32 | 0.04 | 0.64 | 0.00 | 415 |
| 0 | prestall | O_s0 | 28 | 0.46 | 0.04 | 0.50 | 0.00 | 379 |
| 0 | prestall | driver | 28 | 0.64 | 0.04 | 0.32 | 0.00 | 800 |
| 0 | enter_detour | CF_s0 | 4 | 0.75 | 0.00 | 0.25 | 0.00 | 607 |
| 0 | enter_detour | start | 4 | 0.50 | 0.00 | 0.50 | 0.00 | 392 |
| 0 | enter_detour | O_s0 | 4 | 0.50 | 0.00 | 0.50 | 0.00 | 358 |
| 0 | enter_detour | driver | 4 | 1.00 | 0.00 | 0.00 | 0.00 | 800 |
| 0 | t100 | CF_s0 | 16 | 0.38 | 0.00 | 0.62 | 0.00 | 430 |
| 0 | t100 | start | 16 | 0.44 | 0.00 | 0.56 | 0.00 | 564 |
| 0 | t100 | O_s0 | 16 | 0.31 | 0.00 | 0.69 | 0.00 | 381 |
| 0 | t100 | driver | 16 | 0.81 | 0.00 | 0.19 | 0.00 | 800 |
| 1 | prestall | CF_s1 | 33 | 0.24 | 0.00 | 0.76 | 0.06 | 462 |
| 1 | prestall | start | 33 | 0.21 | 0.06 | 0.73 | 0.06 | 519 |
| 1 | prestall | O_s1 | 33 | 0.27 | 0.00 | 0.73 | 0.06 | 412 |
| 1 | prestall | driver | 33 | 0.58 | 0.09 | 0.33 | 0.06 | 800 |
| 1 | enter_detour | CF_s1 | 12 | 0.42 | 0.00 | 0.58 | 0.00 | 409 |
| 1 | enter_detour | start | 12 | 0.58 | 0.00 | 0.42 | 0.00 | 473 |
| 1 | enter_detour | O_s1 | 12 | 0.58 | 0.00 | 0.42 | 0.00 | 378 |
| 1 | enter_detour | driver | 12 | 1.00 | 0.00 | 0.00 | 0.00 | 800 |
| 1 | t100 | CF_s1 | 20 | 0.15 | 0.00 | 0.85 | 0.00 | 519 |
| 1 | t100 | start | 20 | 0.10 | 0.00 | 0.90 | 0.00 | 503 |
| 1 | t100 | O_s1 | 20 | 0.10 | 0.00 | 0.90 | 0.00 | 450 |
| 1 | t100 | driver | 20 | 0.70 | 0.00 | 0.30 | 0.00 | 800 |
| 2 | prestall | CF_s2 | 73 | 0.45 | 0.00 | 0.55 | 0.27 | 399 |
| 2 | prestall | start | 73 | 0.42 | 0.01 | 0.56 | 0.27 | 379 |
| 2 | prestall | O_s2 | 73 | 0.42 | 0.01 | 0.56 | 0.27 | 398 |
| 2 | prestall | driver | 73 | 0.53 | 0.01 | 0.45 | 0.27 | 753 |
| 2 | enter_detour | CF_s2 | 42 | 0.48 | 0.00 | 0.52 | 0.00 | 366 |
| 2 | enter_detour | start | 42 | 0.52 | 0.00 | 0.48 | 0.00 | 388 |
| 2 | enter_detour | O_s2 | 42 | 0.71 | 0.00 | 0.29 | 0.00 | 402 |
| 2 | enter_detour | driver | 42 | 0.98 | 0.00 | 0.02 | 0.00 | 800 |
| 2 | t100 | CF_s2 | 28 | 0.39 | 0.00 | 0.61 | 0.00 | 464 |
| 2 | t100 | start | 28 | 0.43 | 0.00 | 0.57 | 0.00 | 480 |
| 2 | t100 | O_s2 | 28 | 0.36 | 0.00 | 0.64 | 0.00 | 454 |
| 2 | t100 | driver | 28 | 0.64 | 0.00 | 0.36 | 0.00 | 800 |

## 4. Entry times from the recorded xy (per policy, 300 episodes)

| policy | turn north (y>=2) | median step | reach top (y>=6) | median step | reach east column | median step | enter zone 1 | median step | enter zone 2 | median step | entered zone 1 while active / died there | entered zone 2 while active / died there | deaths inside the burst window / deaths | detour episodes that touched a zone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 0.01 | 51.5 | 0.00 | None | 0.00 | None | 0.95 | 66.0 | 0.46 | 137.0 | 143 / 152 | 66 / 70 | 222 / 222 | 0 |
| O_s0 | 0.03 | 29.0 | 0.03 | 66.0 | 0.03 | 275.5 | 0.91 | 62.0 | 0.46 | 134.0 | 135 / 145 | 65 / 67 | 212 / 212 | 0 |
| O_s1 | 0.03 | 25.5 | 0.02 | 64.0 | 0.02 | 258.0 | 0.94 | 63.0 | 0.46 | 134.0 | 142 / 150 | 65 / 67 | 217 / 217 | 0 |
| O_s2 | 0.01 | 69.5 | 0.00 | 67.0 | 0.00 | 278.0 | 0.94 | 63.0 | 0.47 | 134.0 | 139 / 152 | 66 / 70 | 222 / 222 | 0 |
| CF_s0 | 0.15 | 26.0 | 0.08 | 66.0 | 0.07 | 275.0 | 0.84 | 63.0 | 0.39 | 138.0 | 127 / 133 | 53 / 55 | 188 / 188 | 0 |
| CF_s1 | 0.21 | 31.0 | 0.16 | 78.5 | 0.13 | 287.0 | 0.72 | 63.0 | 0.36 | 135.0 | 103 / 118 | 47 / 52 | 170 / 170 | 0 |
| CF_s2 | 0.62 | 28.0 | 0.53 | 73.5 | 0.46 | 284.0 | 0.37 | 67.0 | 0.18 | 145.0 | 52 / 52 | 22 / 26 | 78 / 78 | 0 |

## 5. The actor objective at real start-region rows (each actor with its own critic)

2048 of 6000 logged rows with x < 2, y < 2, t <= 5 (168 next-frame north-moving, 1269 east-moving) and the 196 reset-row anchors.  q-term = -f(s, a_sampled, g) (alpha 0), BC = -log pi(a_logged | s, g); weights 0.95 / 0.05 as in the loss; gradient norms w.r.t. the policy parameters.

### start_region_t<=5

| actor | f(mode) | f(own sample) | f(logged) | P(f mode > f logged) | BC NLL logged (mean) | median | scale | |mode - logged| | 0.95 q-term | 0.05 BC | |grad| q | |grad| BC | BC / q grad ratio | f logged north | f logged east | north - east |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | -8.54 | -8.54 | -8.67 | 0.59 | 3.3 | -12.1 | 0.44 | 0.84 | +8.12 | +0.16 | 1.21 | 28.40 | 23.56 | -8.84 | -8.60 | -0.24 |
| O_s0 | -8.92 | -8.90 | -8.96 | 0.58 | -12.0 | -14.8 | 0.31 | 0.41 | +8.46 | -0.60 | 2.22 | 10.60 | 4.78 | -9.18 | -8.79 | -0.39 |
| O_s1 | -9.02 | -9.04 | -9.08 | 0.54 | -12.1 | -13.1 | 0.35 | 0.43 | +8.59 | -0.60 | 1.91 | 5.45 | 2.85 | -9.34 | -8.98 | -0.37 |
| O_s2 | -8.68 | -8.68 | -8.72 | 0.55 | -12.4 | -14.4 | 0.34 | 0.39 | +8.24 | -0.62 | 1.88 | 8.49 | 4.52 | -8.88 | -8.65 | -0.23 |
| CF_s0 | -10.33 | -10.41 | -10.84 | 0.85 | -9.5 | -9.9 | 0.46 | 0.79 | +9.89 | -0.47 | 4.20 | 3.82 | 0.91 | -10.69 | -10.90 | +0.21 |
| CF_s1 | -10.93 | -11.02 | -11.26 | 0.76 | -10.2 | -10.5 | 0.46 | 0.77 | +10.47 | -0.51 | 5.42 | 4.01 | 0.74 | -10.85 | -11.18 | +0.33 |
| CF_s2 | -9.94 | -10.04 | -10.82 | 0.87 | -7.2 | -7.7 | 0.61 | 1.20 | +9.54 | -0.36 | 4.52 | 4.53 | 1.00 | -11.06 | -10.87 | -0.19 |

### reset_rows_t0

| actor | f(mode) | f(own sample) | f(logged) | P(f mode > f logged) | BC NLL logged (mean) | median | scale | |mode - logged| | 0.95 q-term | 0.05 BC | |grad| q | |grad| BC | BC / q grad ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | -7.98 | -8.00 | -8.67 | 1.00 | 91.5 | 86.4 | 1.01 | 3.37 | +7.60 | +4.58 | 5.54 | 173.51 | 31.29 |
| O_s0 | -8.59 | -8.60 | -8.84 | 0.82 | -5.3 | -9.1 | 0.46 | 0.76 | +8.17 | -0.26 | 13.83 | 30.70 | 2.22 |
| O_s1 | -8.79 | -8.79 | -9.00 | 0.71 | -7.0 | -8.0 | 0.56 | 0.86 | +8.35 | -0.35 | 11.81 | 19.99 | 1.69 |
| O_s2 | -8.40 | -8.43 | -8.54 | 0.69 | -8.0 | -10.4 | 0.50 | 0.67 | +8.01 | -0.40 | 10.85 | 19.01 | 1.75 |
| CF_s0 | -9.20 | -9.48 | -10.07 | 0.89 | -4.4 | -4.7 | 0.75 | 1.40 | +9.00 | -0.22 | 15.08 | 13.99 | 0.93 |
| CF_s1 | -9.65 | -9.70 | -10.39 | 0.89 | -4.8 | -4.4 | 0.72 | 1.43 | +9.22 | -0.24 | 28.59 | 17.98 | 0.63 |
| CF_s2 | -7.90 | -8.35 | -10.01 | 0.99 | 2.0 | 3.3 | 1.27 | 2.86 | +7.93 | +0.10 | 20.79 | 14.64 | 0.70 |

## 6. Route choice at the starts where CF still went straight: candidate first torques under the same CF continuation

Candidates: the CF mode (control), 12 own samples, the other CF seeds' modes, the O and start modes, 4 detour / 2 shortcut logged teacher reset torques; one rollout each, the episode's own hidden draw, the same CF policy continues.  "above mode" = the CF critic scores it above the mode torque at that state.

| CF seed | states (CF went straight) | mode detours on rerun | no candidate detours | a detour candidate exists AND critic ranks one above the mode | detour candidates exist, all ranked below the mode | critic top-1 candidate detours | critic top-1 succeeds | mode succeeds | best candidate succeeds | Spearman f vs outcome (within state) |
|---|---|---|---|---|---|---|---|---|---|---|
| CF_s0 | 275 | 7 | 72 | 188 | 15 | 96 | 109 | 62 | 210 | +0.07 (n 222) |
| CF_s1 | 252 | 9 | 45 | 179 | 28 | 124 | 122 | 62 | 211 | -0.03 (n 216) |
| CF_s2 | 140 | 14 | 6 | 76 | 58 | 51 | 54 | 40 | 129 | -0.27 (n 135) |

Detour rate by candidate family (share of rollouts that realised the detour):

| CF seed | mode | own samples | mode CF_s0 | mode CF_s1 | mode CF_s2 | O | start | teacher detour | teacher shortcut |
|---|---|---|---|---|---|---|---|---|---|
| CF_s0 | 0.03 | 0.09 | nan | 0.11 | 0.27 | 0.02 | 0.10 | 0.25 | 0.00 |
| CF_s1 | 0.04 | 0.10 | 0.08 | nan | 0.21 | 0.02 | 0.10 | 0.35 | 0.00 |
| CF_s2 | 0.10 | 0.26 | 0.16 | 0.20 | nan | 0.06 | 0.24 | 0.44 | 0.02 |

## 7. Paired continuations from the detour entrance where one policy finishes and the other does not

Both continuations rerun with full capture from the same handover state; the earliest anomaly in the failing one, judged against the finishing one: posture (torso down / tilted), slowdown (30-step speed < 0.03 and < half the reference), stall (no progress for 60 steps while the reference progresses), goal-freeze (within 3.0 of the goal, speed < 0.01).

| pair kind | pairs | reproduced on rerun | posture first | slowdown first | stall first | goal-freeze first | no anomaly found | median first-anomaly step after handover | failing arc reached (median) | failing outcomes | mean speed fail / ref | start policy finishes these |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O_finishes_CF_not | 19 | 19 | 1 | 0 | 16 | 0 | 2 | 63.0 | 12.23268461227417 | {'timeout': 19, 'death': 0, 'success': 0} | [0.06183275104589343, 0.11440811360510686] | 0.6842105263157895 |
| CF_finishes_O_not | 8 | 8 | 1 | 2 | 5 | 0 | 0 | 137.5 | 21.076412200927734 | {'timeout': 8, 'death': 0, 'success': 0} | [0.0645366986432665, 0.10950956301595388] | 0.5 |

### Every reproduced pair

| seed | ep | kind | handover step | first anomaly | step after handover | all events | arc reached | final dist to goal | mean speed fail / ref | min torso z | min up z | failing outcome | finisher steps | start finishes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 197 | CF_finishes_O_not | 26 | slowdown | 344 | {"slowdown": 344, "goal_freeze": 348} | 40.0 | 1.5 | 0.056 / 0.107 | 0.37 | 0.92 | timeout | 443 | True |
| 1 | 293 | O_finishes_CF_not | 31 | stall | 19 | {"slowdown": 270, "stall": 19} | 3.4 | 22.3 | 0.033 / 0.112 | 0.36 | 0.93 | timeout | 452 | False |
| 1 | 56 | CF_finishes_O_not | 43 | stall | 183 | {"slowdown": 334, "stall": 183} | 4.6 | 26.1 | 0.037 / 0.097 | 0.35 | 0.73 | timeout | 703 | True |
| 1 | 253 | O_finishes_CF_not | 21 | stall | 43 | {"stall": 43} | 5.7 | 23.1 | 0.080 / 0.123 | 0.40 | 0.88 | timeout | 357 | True |
| 1 | 100 | O_finishes_CF_not | 34 | stall | 29 | {"stall": 29} | 4.4 | 23.2 | 0.087 / 0.125 | 0.39 | 0.90 | timeout | 383 | True |
| 1 | 180 | O_finishes_CF_not | 49 | stall | 57 | {"slowdown": 414, "stall": 57} | 8.0 | 26.5 | 0.046 / 0.122 | 0.35 | 0.92 | timeout | 378 | False |
| 1 | 24 | CF_finishes_O_not | 42 | slowdown | 321 | {"slowdown": 321, "stall": 352} | 34.7 | 5.2 | 0.067 / 0.106 | 0.36 | 0.96 | timeout | 514 | True |
| 2 | 65 | O_finishes_CF_not | 27 | stall | 30 | {"stall": 30} | 18.4 | 15.3 | 0.074 / 0.119 | 0.34 | 0.89 | timeout | 386 | True |
| 2 | 108 | CF_finishes_O_not | 31 | stall | 91 | {"stall": 91} | 10.7 | 23.5 | 0.064 / 0.120 | 0.38 | 0.87 | timeout | 374 | False |
| 2 | 159 | O_finishes_CF_not | 39 | stall | 69 | {"stall": 69} | 21.3 | 12.5 | 0.056 / 0.117 | 0.41 | 0.86 | timeout | 392 | True |
| 2 | 225 | O_finishes_CF_not | 26 | stall | 261 | {"stall": 261} | 34.4 | 5.2 | 0.077 / 0.115 | 0.37 | 0.95 | timeout | 405 | True |
| 2 | 132 | O_finishes_CF_not | 21 | stall | 267 | {"slowdown": 585, "stall": 267} | 34.2 | 4.9 | 0.065 / 0.115 | 0.37 | 0.92 | timeout | 393 | False |
| 2 | 265 | CF_finishes_O_not | 39 | posture | 664 | {"posture": 664, "slowdown": 675, "goal_freeze": 689} | 38.5 | 1.6 | 0.085 / 0.114 | 0.24 | -1.00 | timeout | 439 | False |
| 2 | 17 | O_finishes_CF_not | 20 | stall | 28 | {"stall": 28} | 4.5 | 23.6 | 0.075 / 0.115 | 0.39 | 0.91 | timeout | 413 | True |
| 2 | 192 | CF_finishes_O_not | 54 | stall | 2 | {"slowdown": 589, "stall": 2} | 2.6 | 22.7 | 0.067 / 0.116 | 0.37 | 0.89 | timeout | 328 | False |
| 2 | 74 | O_finishes_CF_not | 30 | stall | 55 | {"slowdown": 311, "stall": 55} | 4.8 | 25.8 | 0.045 / 0.119 | 0.35 | 0.85 | timeout | 380 | False |
| 2 | 235 | O_finishes_CF_not | 29 | stall | 63 | {"posture": 331, "slowdown": 344, "stall": 63} | 12.2 | 22.1 | 0.032 / 0.114 | 0.25 | -1.00 | timeout | 385 | True |
| 2 | 78 | O_finishes_CF_not | 27 | stall | 101 | {"posture": 358, "slowdown": 353, "stall": 101} | 10.8 | 23.5 | 0.032 / 0.100 | 0.38 | -0.75 | timeout | 529 | True |
| 2 | 242 | O_finishes_CF_not | 25 | posture | 377 | {"posture": 377, "slowdown": 388, "goal_freeze": 401} | 40.0 | 1.8 | 0.058 / 0.125 | 0.25 | -1.00 | timeout | 332 | True |
| 2 | 281 | O_finishes_CF_not | 33 | stall | 80 | {"slowdown": 668, "stall": 80} | 22.1 | 12.4 | 0.057 / 0.111 | 0.39 | 0.85 | timeout | 441 | True |
| 2 | 53 | O_finishes_CF_not | 22 | None | None | {} | 30.5 | 6.3 | 0.067 / 0.111 | 0.41 | 0.74 | timeout | 429 | True |
| 2 | 244 | O_finishes_CF_not | 33 | stall | 98 | {"stall": 98} | 10.3 | 24.4 | 0.060 / 0.104 | 0.35 | 0.91 | timeout | 487 | False |
| 2 | 149 | CF_finishes_O_not | 119 | stall | 31 | {"stall": 31} | 21.5 | 13.0 | 0.085 / 0.093 | 0.35 | 0.71 | timeout | 753 | False |
| 2 | 173 | O_finishes_CF_not | 19 | stall | 176 | {"stall": 176} | 36.3 | 2.8 | 0.083 / 0.112 | 0.39 | 0.90 | timeout | 400 | True |
| 2 | 151 | CF_finishes_O_not | 33 | stall | 92 | {"slowdown": 258, "stall": 92} | 20.7 | 13.6 | 0.055 / 0.123 | 0.38 | 0.84 | timeout | 341 | True |
| 2 | 178 | O_finishes_CF_not | 22 | stall | 18 | {"stall": 18} | 5.5 | 24.0 | 0.071 / 0.108 | 0.41 | 0.89 | timeout | 435 | True |
| 2 | 98 | O_finishes_CF_not | 24 | None | None | {} | 39.5 | 1.4 | 0.077 / 0.108 | 0.37 | 0.87 | timeout | 457 | False |

