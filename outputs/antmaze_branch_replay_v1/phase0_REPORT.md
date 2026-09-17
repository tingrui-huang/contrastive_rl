# Phase 0: the interventional law at the start region, and the placed-on-detour audit

400 recorded start-region anchors (x < 2, y < 2, t <= 60); each branched north (detour legs) and east (shortcut legs, blind go) from its restored Ant state with hazards and clocks redrawn from the priors.  Goal-frame probability = the discounted relabeling law from the anchor over the branch path (paths end at success / death / horizon, unless parking is on: with V6_BRANCH_PARK=1 (default) a path continues to the horizon after reaching, holding with zero torque, so the goal is an absorbing parked tail instead of the single last frame the recorded episodes end on); radius 0.5 = the success distance.

| branch | n | reach | death | timeout | reach step (median) | P_goal g=0.999 r=0.5 | r=1.0 | P_goal g=0.99 r=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start_query / detour | 400 | 0.627 | 0.000 | 0.372 | 425.0 | 0.23913 | 0.24897 | 0.01112 |
| start_query / go | 400 | 0.330 | 0.667 | 0.003 | 220.0 | 0.20843 | 0.21634 | 0.03814 |
| placed_north / detour | 60 | 0.967 | 0.000 | 0.033 | 375.0 | 0.43193 | 0.44938 | 0.03471 |
| placed_east / detour | 60 | 0.983 | 0.000 | 0.017 | 368.0 | 0.58149 | 0.64141 | 0.18397 |

| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |
|---|---:|---:|---:|---:|---:|
| 0-0 | 42 | 1.00 | 0.38 | **+0.60** | +0.58 |
| 0-5 | 200 | 0.92 | 0.33 | **+0.58** | +0.57 |
| 0-10 | 266 | 0.86 | 0.32 | **+0.52** | +0.51 |
| 11-60 | 134 | 0.16 | 0.36 | **-1.38** | -1.37 |

interventional target at the start region, gamma 0.999, r 0.5: log P(detour) / P(shortcut) = **+0.137**

interventional target at the start region, gamma 0.999, r 1.0: log P(detour) / P(shortcut) = **+0.140**

interventional target at the start region, gamma 0.99, r 0.5: log P(detour) / P(shortcut) = **-1.233**

observational counterpart (recorded futures, sighted teacher; 28 detour / 372 shortcut rows), gamma 0.999, r 0.5: **-0.477**
