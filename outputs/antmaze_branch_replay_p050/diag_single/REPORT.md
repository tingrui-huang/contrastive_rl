# Single query torque, closed-loop blind continuation from step 2 (paired hazards, oracle model)

One torque at the state (N = first torque of a north donor at the same time step, E = of an east donor, own = the row's recorded torque), then the continuation policy acts from the resulting state to the horizon; no donor continuation, no intent label; zero-torque hold after reaching; P_goal = gamma 0.999, radius 0.5.  driver = the replay's blind driver re-deciding its route from its position at every step; bc = the deployment pure-BC actor's mode.

## continuation: driver

| state set | n | arm | north @25 | reach | death | P_goal | final route detour | route flips (mean) | log P_goal N/E | paired: P_goal(N) > P_goal(E) | final-route differs N vs E |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| start (x<2, y<2, t<=5) | 150 | N | 0.00 | 0.21 | 0.79 | 0.132 | 0.00 | 0.00 | **+0.09** | 0.08 | 0.01 |
| start (x<2, y<2, t<=5) | 150 | E | 0.00 | 0.21 | 0.79 | 0.121 | 0.01 | 0.00 | | | |
| start (x<2, y<2, t<=5) | 150 | own | 0.00 | 0.21 | 0.79 | 0.125 | 0.01 | 0.02 | | | |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | N | 0.21 | 0.60 | 0.39 | 0.313 | 0.02 | 0.46 | **-0.01** | 0.23 | 0.02 |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | E | 0.22 | 0.60 | 0.39 | 0.316 | 0.01 | 0.41 | | | |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | own | 0.26 | 0.60 | 0.38 | 0.310 | 0.03 | 0.52 | | | |
| north leg low (x<2, 2<=y<4) | 150 | N | 0.99 | 0.99 | 0.00 | 0.388 | 0.05 | 0.83 | **+0.01** | 0.52 | 0.10 |
| north leg low (x<2, 2<=y<4) | 150 | E | 1.00 | 0.96 | 0.00 | 0.384 | 0.09 | 0.91 | | | |
| north leg low (x<2, 2<=y<4) | 150 | own | 1.00 | 0.97 | 0.00 | 0.383 | 0.06 | 0.84 | | | |
| north leg high (x<2, 4<=y<6) | 150 | N | 1.00 | 0.98 | 0.00 | 0.429 | 0.05 | 0.78 | **+0.00** | 0.37 | 0.11 |
| north leg high (x<2, 4<=y<6) | 150 | E | 1.00 | 0.99 | 0.00 | 0.428 | 0.05 | 0.79 | | | |
| north leg high (x<2, 4<=y<6) | 150 | own | 1.00 | 0.99 | 0.00 | 0.415 | 0.03 | 0.77 | | | |
| shortcut early (2<=x<5, y<2) | 150 | N | 0.00 | 0.25 | 0.75 | 0.173 | 0.01 | 0.01 | **-0.00** | 0.10 | 0.02 |
| shortcut early (2<=x<5, y<2) | 150 | E | 0.00 | 0.25 | 0.75 | 0.173 | 0.02 | 0.00 | | | |
| shortcut early (2<=x<5, y<2) | 150 | own | 0.00 | 0.25 | 0.75 | 0.173 | 0.01 | 0.00 | | | |

## continuation: bc

| state set | n | arm | north @25 | reach | death | P_goal | final route detour | route flips (mean) | log P_goal N/E | paired: P_goal(N) > P_goal(E) | final-route differs N vs E |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| start (x<2, y<2, t<=5) | 150 | N | 0.11 | 0.21 | 0.67 | 0.111 | 0.05 | 0.00 | **-0.02** | 0.09 | 0.06 |
| start (x<2, y<2, t<=5) | 150 | E | 0.02 | 0.21 | 0.77 | 0.113 | 0.01 | 0.00 | | | |
| start (x<2, y<2, t<=5) | 150 | own | 0.04 | 0.23 | 0.75 | 0.124 | 0.01 | 0.00 | | | |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | N | 0.58 | 0.50 | 0.11 | 0.155 | 0.21 | 0.00 | **+0.12** | 0.38 | 0.30 |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | E | 0.62 | 0.45 | 0.09 | 0.138 | 0.26 | 0.00 | | | |
| detour turn in progress (detour eps, x<2, y<2, t>5) | 150 | own | 0.69 | 0.53 | 0.06 | 0.142 | 0.22 | 0.00 | | | |
| north leg low (x<2, 2<=y<4) | 150 | N | 0.99 | 0.81 | 0.00 | 0.252 | 0.20 | 0.00 | **+0.03** | 0.52 | 0.33 |
| north leg low (x<2, 2<=y<4) | 150 | E | 0.97 | 0.79 | 0.00 | 0.245 | 0.26 | 0.00 | | | |
| north leg low (x<2, 2<=y<4) | 150 | own | 1.00 | 0.83 | 0.00 | 0.247 | 0.19 | 0.00 | | | |
| north leg high (x<2, 4<=y<6) | 150 | N | 0.82 | 0.82 | 0.00 | 0.250 | 0.19 | 0.00 | **-0.01** | 0.45 | 0.31 |
| north leg high (x<2, 4<=y<6) | 150 | E | 0.83 | 0.82 | 0.00 | 0.253 | 0.21 | 0.00 | | | |
| north leg high (x<2, 4<=y<6) | 150 | own | 0.84 | 0.83 | 0.00 | 0.244 | 0.23 | 0.00 | | | |
| shortcut early (2<=x<5, y<2) | 150 | N | 0.00 | 0.25 | 0.72 | 0.167 | 0.00 | 0.00 | **+0.22** | 0.15 | 0.00 |
| shortcut early (2<=x<5, y<2) | 150 | E | 0.00 | 0.25 | 0.72 | 0.134 | 0.00 | 0.00 | | | |
| shortcut early (2<=x<5, y<2) | 150 | own | 0.00 | 0.25 | 0.75 | 0.151 | 0.00 | 0.00 | | | |

Reading: log P_goal N/E is the interventional single-step target the critic would be asked to carry at these states under this continuation; paired win = how often, with the same hazards, the north torque's future beats the east torque's; final-route differs = how often the one torque changed the route the continuation ended on.
