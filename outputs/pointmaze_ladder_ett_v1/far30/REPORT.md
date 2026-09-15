# F4 detour ladder, rung 0.3: 1461/4800 teacher episodes on the safe route

BC 0.05, 30000 updates per arm, 200 native episodes (same reset seeds for every arm, seed and rung). O = plain CRL on the rung; C = fixed-ETT query-coverage replay (50/50). Critic = held-out down-minus-right logit on 16 roots (>0 prefers the detour).

| seed | arm | mode reach | mode lower | sample reach | sample lower | critic d-r | roots down |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | O | 0.305 | 0.000 | 0.530 | 0.355 | +0.119 | 13/16 |
| 0 | C | 0.840 | 0.790 | 0.655 | 0.545 | +0.589 | 16/16 |
| 1 | O | 0.820 | 0.745 | 0.700 | 0.615 | +0.120 | 15/16 |
| 1 | C | 0.845 | 0.780 | 0.685 | 0.580 | +0.053 | 14/16 |
| 2 | O | 0.305 | 0.000 | 0.460 | 0.280 | +0.073 | 12/16 |
| 2 | C | 1.000 | 1.000 | 0.655 | 0.615 | +0.347 | 16/16 |
| mean | O | 0.477 | 0.248 | 0.563 | 0.417 | +0.104 | |
| mean | C | 0.895 | 0.857 | 0.665 | 0.580 | +0.330 | |
