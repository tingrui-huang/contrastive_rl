# F4 detour ladder, rung 0.2: 983/4800 teacher episodes on the safe route

BC 0.05, 30000 updates per arm, 200 native episodes (same reset seeds for every arm, seed and rung). O = plain CRL on the rung; C = fixed-ETT query-coverage replay (50/50). Critic = held-out down-minus-right logit on 16 roots (>0 prefers the detour).

| seed | arm | mode reach | mode lower | sample reach | sample lower | critic d-r | roots down |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | O | 0.305 | 0.000 | 0.325 | 0.065 | -0.099 | 9/16 |
| 0 | C | 0.355 | 0.055 | 0.510 | 0.360 | +0.197 | 15/16 |
| 1 | O | 0.305 | 0.000 | 0.335 | 0.120 | +0.157 | 15/16 |
| 1 | C | 0.305 | 0.000 | 0.450 | 0.250 | +0.243 | 16/16 |
| 2 | O | 0.305 | 0.000 | 0.370 | 0.135 | -0.290 | 0/16 |
| 2 | C | 0.700 | 0.595 | 0.590 | 0.485 | +0.418 | 16/16 |
| mean | O | 0.305 | 0.000 | 0.343 | 0.107 | -0.077 | |
| mean | C | 0.453 | 0.217 | 0.517 | 0.365 | +0.286 | |
