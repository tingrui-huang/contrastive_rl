# F4 detour ladder, rung 0.1: 490/4800 teacher episodes on the safe route

BC 0.05, 30000 updates per arm, 200 native episodes (same reset seeds for every arm, seed and rung). O = plain CRL on the rung; C = fixed-ETT query-coverage replay (50/50). Critic = held-out down-minus-right logit on 16 roots (>0 prefers the detour).

| seed | arm | mode reach | mode lower | sample reach | sample lower | critic d-r | roots down |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | O | 0.305 | 0.000 | 0.300 | 0.045 | -0.185 | 0/16 |
| 0 | C | 1.000 | 1.000 | 0.805 | 0.750 | +0.319 | 16/16 |
| 1 | O | 0.305 | 0.000 | 0.325 | 0.055 | +0.276 | 16/16 |
| 1 | C | 0.810 | 0.735 | 0.675 | 0.590 | +0.156 | 15/16 |
| 2 | O | 0.305 | 0.000 | 0.425 | 0.225 | -0.128 | 1/16 |
| 2 | C | 1.000 | 1.000 | 0.770 | 0.750 | +0.130 | 16/16 |
| mean | O | 0.305 | 0.000 | 0.350 | 0.108 | -0.012 | |
| mean | C | 0.937 | 0.912 | 0.750 | 0.697 | +0.202 | |
