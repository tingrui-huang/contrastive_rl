# Phase 0: the interventional law at the start region, and the placed-on-detour audit

400 recorded start-region anchors (x < 2, y < 2, t <= 60); each branched north (detour legs) and east (shortcut legs, blind go) from its restored Ant state with hazards and clocks redrawn from the priors.  Goal-frame probability = the discounted relabeling law from the anchor over the branch path (paths end at success / death / horizon, unless parking is on: with V6_BRANCH_PARK=1 (default) a path continues to the horizon after reaching, holding with zero torque, so the goal is an absorbing parked tail instead of the single last frame the recorded episodes end on); radius 0.5 = the success distance.

| branch | n | reach | death | timeout | reach step (median) | P_goal g=0.999 r=0.5 | r=1.0 | P_goal g=0.99 r=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start_query / detour | 400 | 0.627 | 0.000 | 0.372 | 426.0 | 0.23875 | 0.24919 | 0.01116 |
| start_query / go | 400 | 0.210 | 0.787 | 0.003 | 221.0 | 0.13585 | 0.13722 | 0.02444 |
| placed_north / detour | 100 | 0.970 | 0.000 | 0.030 | 369.0 | 0.42375 | 0.45901 | 0.03480 |
| placed_east / detour | 100 | 0.990 | 0.000 | 0.010 | 365.0 | 0.60781 | 0.64497 | 0.18088 |

| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |
|---|---:|---:|---:|---:|---:|
| 0-0 | 42 | 0.98 | 0.24 | **+1.00** | +1.04 |
| 0-5 | 200 | 0.91 | 0.23 | **+0.85** | +0.89 |
| 0-10 | 266 | 0.86 | 0.22 | **+0.85** | +0.88 |
| 11-60 | 134 | 0.16 | 0.19 | **-0.72** | -0.71 |

interventional target at the start region, gamma 0.999, r 0.5: log P(detour) / P(shortcut) = **+0.564**

interventional target at the start region, gamma 0.999, r 1.0: log P(detour) / P(shortcut) = **+0.597**

interventional target at the start region, gamma 0.99, r 0.5: log P(detour) / P(shortcut) = **-0.784**

observational counterpart (recorded futures, sighted teacher; 28 detour / 372 shortcut rows), gamma 0.999, r 0.5: **-0.449**
