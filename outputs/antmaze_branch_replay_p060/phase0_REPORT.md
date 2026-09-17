# Phase 0: the interventional law at the start region, and the placed-on-detour audit

400 recorded start-region anchors (x < 2, y < 2, t <= 60); each branched north (detour legs) and east (shortcut legs, blind go) from its restored Ant state with hazards and clocks redrawn from the priors.  Goal-frame probability = the discounted relabeling law from the anchor over the branch path (paths end at success / death / horizon, unless parking is on: with V6_BRANCH_PARK=1 (default) a path continues to the horizon after reaching, holding with zero torque, so the goal is an absorbing parked tail instead of the single last frame the recorded episodes end on); radius 0.5 = the success distance.

| branch | n | reach | death | timeout | reach step (median) | P_goal g=0.999 r=0.5 | r=1.0 | P_goal g=0.99 r=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start_query / detour | 400 | 0.630 | 0.000 | 0.370 | 425.5 | 0.23703 | 0.25026 | 0.01113 |
| start_query / go | 400 | 0.142 | 0.855 | 0.003 | 220.0 | 0.09207 | 0.09301 | 0.01657 |
| placed_north / detour | 100 | 0.980 | 0.000 | 0.020 | 370.0 | 0.43197 | 0.46261 | 0.03522 |
| placed_east / detour | 100 | 0.990 | 0.000 | 0.010 | 365.0 | 0.60764 | 0.64495 | 0.18085 |

| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |
|---|---:|---:|---:|---:|---:|
| 0-0 | 42 | 1.00 | 0.17 | **+1.35** | +1.41 |
| 0-5 | 200 | 0.92 | 0.15 | **+1.30** | +1.35 |
| 0-10 | 266 | 0.87 | 0.14 | **+1.26** | +1.31 |
| 11-60 | 134 | 0.16 | 0.14 | **-0.43** | -0.43 |

interventional target at the start region, gamma 0.999, r 0.5: log P(detour) / P(shortcut) = **+0.946**

interventional target at the start region, gamma 0.999, r 1.0: log P(detour) / P(shortcut) = **+0.990**

interventional target at the start region, gamma 0.99, r 0.5: log P(detour) / P(shortcut) = **-0.398**

observational counterpart (recorded futures, sighted teacher; 28 detour / 372 shortcut rows), gamma 0.999, r 0.5: **-0.431**
