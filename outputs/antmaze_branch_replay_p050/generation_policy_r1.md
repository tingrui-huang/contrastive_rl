# Round-1 policy-continuation replay (replay_policy_r1.npz)

Continuation: the frozen deployment policy mode (final.pkl); one query torque per path, then closed-loop to the horizon; hazards redrawn per draw, paired across candidates; P_goal = the relabeling law (gamma 0.999, radius 0.5); policy tanh-normal scale 0.445 (median 0.211).  17756 fit paths, 7200 held-out paths (100 held-out episodes), 1791 s.

## Fit paths by state set and candidate

| set | candidate | n | reach | death | P_goal | went around (max_y >= 6) |
|---|---|---:|---:|---:|---:|---:|
| start_early | recorded | 800 | 0.28 | 0.71 | 0.145 | 0.02 |
| start_early | sample0 | 800 | 0.27 | 0.71 | 0.145 | 0.01 |
| start_early | sample1 | 800 | 0.27 | 0.71 | 0.149 | 0.01 |
| turn | recorded | 600 | 0.49 | 0.05 | 0.146 | 0.63 |
| turn | sample0 | 600 | 0.47 | 0.08 | 0.138 | 0.62 |
| turn | sample1 | 600 | 0.48 | 0.10 | 0.158 | 0.57 |
| start_late | recorded | 400 | 0.26 | 0.74 | 0.140 | 0.00 |
| start_late | sample0 | 400 | 0.26 | 0.74 | 0.137 | 0.00 |
| start_late | sample1 | 400 | 0.26 | 0.74 | 0.143 | 0.00 |
| north_leg | recorded | 600 | 0.82 | 0.00 | 0.247 | 1.00 |
| north_leg | sample0 | 600 | 0.85 | 0.00 | 0.269 | 0.99 |
| north_leg | sample1 | 600 | 0.81 | 0.00 | 0.244 | 1.00 |
| shortcut_early | recorded | 600 | 0.25 | 0.75 | 0.139 | 0.00 |
| shortcut_early | sample0 | 600 | 0.25 | 0.75 | 0.144 | 0.00 |
| shortcut_early | sample1 | 600 | 0.25 | 0.75 | 0.134 | 0.00 |
| general | recorded | 4378 | 0.64 | 0.32 | 0.406 | 0.06 |
| general | sample0 | 4378 | 0.64 | 0.33 | 0.410 | 0.06 |

## Held-out paths by state set and candidate

| set | candidate | n | reach | death | P_goal | went around |
|---|---|---:|---:|---:|---:|---:|
| start_early | recorded | 240 | 0.25 | 0.72 | 0.137 | 0.03 |
| start_early | mode | 240 | 0.26 | 0.70 | 0.146 | 0.03 |
| start_early | sample0 | 240 | 0.27 | 0.70 | 0.143 | 0.03 |
| start_early | sample1 | 240 | 0.24 | 0.71 | 0.124 | 0.05 |
| start_early | sample2 | 240 | 0.25 | 0.71 | 0.128 | 0.04 |
| turn | recorded | 240 | 0.65 | 0.08 | 0.178 | 0.80 |
| turn | mode | 240 | 0.53 | 0.11 | 0.138 | 0.76 |
| turn | sample0 | 240 | 0.58 | 0.07 | 0.185 | 0.78 |
| turn | sample1 | 240 | 0.56 | 0.11 | 0.161 | 0.75 |
| turn | sample2 | 240 | 0.51 | 0.15 | 0.149 | 0.70 |
| start_late | recorded | 240 | 0.21 | 0.78 | 0.108 | 0.00 |
| start_late | mode | 240 | 0.22 | 0.78 | 0.102 | 0.00 |
| start_late | sample0 | 240 | 0.22 | 0.78 | 0.104 | 0.00 |
| start_late | sample1 | 240 | 0.22 | 0.78 | 0.107 | 0.00 |
| start_late | sample2 | 240 | 0.21 | 0.78 | 0.098 | 0.00 |
| north_leg | recorded | 240 | 0.83 | 0.00 | 0.238 | 1.00 |
| north_leg | mode | 240 | 0.80 | 0.00 | 0.241 | 1.00 |
| north_leg | sample0 | 240 | 0.70 | 0.00 | 0.218 | 1.00 |
| north_leg | sample1 | 240 | 0.88 | 0.00 | 0.270 | 1.00 |
| north_leg | sample2 | 240 | 0.72 | 0.00 | 0.209 | 1.00 |
| shortcut_early | recorded | 240 | 0.25 | 0.75 | 0.142 | 0.00 |
| shortcut_early | mode | 240 | 0.25 | 0.75 | 0.135 | 0.00 |
| shortcut_early | sample0 | 240 | 0.25 | 0.75 | 0.141 | 0.00 |
| shortcut_early | sample1 | 240 | 0.23 | 0.75 | 0.133 | 0.00 |
| shortcut_early | sample2 | 240 | 0.25 | 0.75 | 0.145 | 0.00 |
| general | recorded | 240 | 0.73 | 0.27 | 0.487 | 0.05 |
| general | mode | 240 | 0.71 | 0.29 | 0.456 | 0.05 |
| general | sample0 | 240 | 0.71 | 0.29 | 0.491 | 0.05 |
| general | sample1 | 240 | 0.69 | 0.29 | 0.470 | 0.05 |
| general | sample2 | 240 | 0.69 | 0.29 | 0.441 | 0.05 |

## Between-candidate differences within an anchor (log P_goal ratio, floor 1e-3)

split-half = sign agreement of the ratio between the first and the second half of the draws, over pairs whose first-half |ratio| > 0.3: the ceiling any critic can reach on these keys (fit set: draw 0 vs draw 1).

| split | set | pairs | mean abs log ratio | frac abs > 0.3 | split-half sign agreement | half pairs | validated pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| fit | start_early | 1200 | 0.58 | 0.15 | 0.29 | 126 | 36 |
| fit | turn | 900 | 1.95 | 0.55 | 0.78 | 434 | 319 |
| fit | start_late | 600 | 0.53 | 0.14 | 0.29 | 38 | 11 |
| fit | north_leg | 900 | 2.08 | 0.71 | 0.78 | 621 | 410 |
| fit | shortcut_early | 900 | 0.52 | 0.13 | 0.23 | 84 | 19 |
| fit | general | 4378 | 0.96 | 0.24 | - | 0 | 0 |
| holdout | start_early | 600 | 0.88 | 0.22 | 0.38 | 77 | 28 |
| holdout | turn | 600 | 2.13 | 0.67 | 0.76 | 404 | 287 |
| holdout | start_late | 600 | 0.80 | 0.24 | 0.41 | 94 | 39 |
| holdout | north_leg | 600 | 2.06 | 0.74 | 0.82 | 423 | 314 |
| holdout | shortcut_early | 600 | 0.81 | 0.21 | 0.26 | 61 | 16 |
| holdout | general | 600 | 1.19 | 0.32 | 0.89 | 166 | 143 |
