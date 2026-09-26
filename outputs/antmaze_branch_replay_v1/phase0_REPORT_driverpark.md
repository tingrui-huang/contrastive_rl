# Phase 0: the interventional law at the start region, and the placed-on-detour audit

400 recorded start-region anchors (x < 2, y < 2, t <= 60); each branched north (detour legs) and east (shortcut legs, blind go) from its restored Ant state with hazards and clocks redrawn from the priors.  Goal-frame probability = the discounted relabeling law from the anchor over the branch path (paths end at success / death / horizon, unless parking is on: with V6_BRANCH_PARK=1 (default) a path continues to the horizon after reaching, so the goal is a parked tail as on PointMaze instead of the single last frame the recorded episodes end on); radius 0.5 = the success distance.

| branch | n | reach | death | timeout | reach step (median) | P_goal g=0.999 r=0.5 | r=1.0 | P_goal g=0.99 r=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start_query / detour | 400 | 0.627 | 0.000 | 0.372 | 426.0 | 0.00677 | 0.02017 | 0.00099 |
| start_query / go | 400 | 0.333 | 0.665 | 0.003 | 222.0 | 0.13733 | 0.20377 | 0.02792 |
| placed_north / detour | 60 | 0.983 | 0.000 | 0.017 | 375.0 | 0.01263 | 0.04186 | 0.00311 |
| placed_east / detour | 60 | 0.983 | 0.000 | 0.017 | 368.0 | 0.01722 | 0.06245 | 0.01717 |

| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |
|---|---:|---:|---:|---:|---:|
| 0-0 | 42 | 1.00 | 0.36 | **-2.66** | -2.02 |
| 0-5 | 200 | 0.92 | 0.34 | **-2.59** | -1.90 |
| 0-10 | 266 | 0.86 | 0.33 | **-2.61** | -1.98 |
| 11-60 | 134 | 0.16 | 0.34 | **-4.55** | -3.62 |

interventional target at the start region, gamma 0.999, r 0.5: log P(detour) / P(shortcut) = **-3.010**

interventional target at the start region, gamma 0.999, r 1.0: log P(detour) / P(shortcut) = **-2.313**

interventional target at the start region, gamma 0.99, r 0.5: log P(detour) / P(shortcut) = **-3.338**

observational counterpart (recorded futures, sighted teacher; 28 detour / 372 shortcut rows), gamma 0.999, r 0.5: **-0.477**
