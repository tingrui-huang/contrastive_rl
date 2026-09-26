# Which future supervision produced the critic's preference for the stalled action (user's plan after ccc42f6; 2026-09-22; `scripts/diag_v6_reset_futures.py`; no training)

## A. The reset / start anchors' futures: learned table draw 1 vs draw 2 (same dataset, same S ETT, another generation seed), the sealed simulator table as reference (`FUTURES.md`)

Per anchor: outcome, path length, no-route timeout (a timeout that never leaves the start region), the gamma-law masses (reach 0.5 /
near 2.0 / goal area / far / death frame) plus the STALL mass (rows with xy speed < 0.03) and the START-region mass.

| group | table | success / death / timeout | no-route timeout | length mean | stall mass | start-region mass | near2.0 / goal-area mass | outcome agreement d1-d2 |
|---|---|---|---:|---:|---:|---:|---|---:|
| the 64 diagnostic reset anchors | draw 1 | **0.422** / 0.578 / 0.000 | 0.000 | 147 | 0.005 | 0.189 | 0.034 / **0.046** | 0.58 |
| | draw 2 | 0.156 / 0.844 / 0.000 | 0.000 | 113 | 0.005 | 0.229 | 0.015 / 0.020 | |
| | simulator | 0.281 / 0.672 / 0.047 | 0.000 | 165 | 0.029 | 0.201 | 0.049 / 0.059 | |
| all 196 reset anchors | draw 1 | 0.332 / 0.648 / 0.020 | 0.005 | 154 | 0.014 | 0.198 | 0.037 / 0.047 | 0.60 |
| | draw 2 | 0.204 / 0.781 / 0.015 | 0.005 | 130 | 0.011 | 0.226 | 0.025 / 0.031 | |
| | simulator | 0.296 / 0.663 / 0.041 | 0.000 | 163 | 0.028 | 0.203 | 0.049 / 0.059 | |
| start_early (4,049 anchors) | draw 1 | 0.260 / 0.726 / 0.014 | 0.002 | 130 | 0.008 | 0.115 | 0.030 / 0.037 | 0.64 |
| | draw 2 | 0.269 / 0.720 / 0.011 | 0.002 | 130 | 0.006 | 0.114 | 0.029 / 0.036 | |

* The hypothesis "draw 1 binds stalled / no-route long futures to the reset anchors" is NOT supported: no-route timeouts are 0.0-0.5 %
  in both tables, the stall mass 0.005-0.014 (the simulator's 0.03), the slow-row share ~1 %.  Neither learned table stalls at the
  start.
* What differs at the reset rows is the opposite: draw 1 gives the reset anchors MORE goal-reaching futures -- success 0.42 vs 0.16
  on the 64 diagnostic anchors (0.33 vs 0.20 on all 196; the simulator 0.28-0.30), 2.3x the goal-area mass and 2.3x the near-2.0 mass,
  longer paths (147 vs 113 rows), less start-region mass.  At the start_early anchors (4,049) the two tables agree to 0.01 -- the
  difference is confined to the 196 reset rows (the per-anchor onset / context draw at t = 0: outcome agreement between the draws is
  only 0.58-0.60 per anchor, and each draw agrees with the simulator's outcome in 0.44-0.61 of the anchors).

## B. The critics' Q surface along forward -> stall at the 64 reset states, by goal set (`QSURFACE.md`)

a(l) = (1 - l) a_forward + l a_stall; a_stall = the draw-1 seed-4 final mode (the saturated action), a_forward = the draw-2 seed-4 final
mode / the start agent's mode / the logged first torque; Q = the twin-min logit; goal sets: the task goal, 32 goals within 2.0 of it,
16 relabelled future goals of the anchor's logged episode (the ACTOR stream's law), 256 goals from each table's critic-training marginal.
Bump = Q(a_stall) - Q(a_forward), mean over the 64 states (nats); "stall higher" = share of states with a positive bump; argmax = where
along the path the critic's maximum sits.

| critic | forward action | task goal | near 2.0 | relabelled (actor stream) | critic-training marginal (draw 1 / draw 2) | argmax at the stall end (task) |
|---|---|---|---|---|---|---:|
| draw 1 seed 4, final | draw-2 final (progressing) | **+2.14** (1.00) | +2.16 (1.00) | +0.48 (0.66) | +1.30 (0.95) / +1.51 (0.98) | 0.95 |
| | start agent | **+4.22** (1.00) | +4.44 (1.00) | +1.34 (0.89) | +2.95 (1.00) / +3.18 (1.00) | 0.97 |
| | logged first torque | +2.43 (1.00) | +2.61 (0.98) | -0.32 (0.36) | +0.82 (0.92) / +0.93 (0.94) | 0.84 |
| draw 1 seed 4, 20k | draw-2 final | +1.21 (1.00) | +1.18 (0.98) | -0.01 (0.45) | +0.56 (0.89) / +0.71 (0.95) | 0.98 |
| | start agent | +2.81 (1.00) | +2.86 (1.00) | +0.54 (0.78) | +1.63 (1.00) / +1.74 (1.00) | 0.86 |
| draw 2 seed 4, final | draw-2 final | -1.84 (0.09) | -1.98 (0.05) | -1.28 (0.02) | -1.90 (0.00) / -1.83 (0.02) | 0.06 |
| | start agent | +0.92 (0.83), max INTERIOR 0.94 | +0.94 (0.80), interior 0.92 | -0.02 (0.47), interior 0.91 | +0.49 / +0.46, interior 0.98 | 0.06 |
| | logged first torque | -0.17 (0.50) | -0.28 (0.47) | -1.43 (0.03) | -1.17 (0.09) / -1.26 (0.05) | 0.36 |

* CORRECTION (user's review): the monotone rise holds under the TASK / near-goal goals (the maximum at the stall end in 84-98 % of the
  states) and, weaker, under the critic-training marginals -- NOT under the actor stream's relabelled goals, where the maximum is interior
  in 52-84 % of the states.  Original sentence: the draw-1 critic's Q rises MONOTONICALLY toward the saturated action from every forward
  action and under every goal set; the draw-2 critic's Q falls toward it from its own action and
  from the logged torque, and from the start agent's action its maximum is in the INTERIOR of the path (91-98 %) -- an action between
  the start agent's and the bound, its own actor's.  A monotone Q up to the action bound is what turns the actor's gradient ascent into
  saturation; an interior maximum does not.
* Where the bump is largest: the TASK goal and the goals NEAR it (+2.1 to +4.4 nats), then the critic-training marginals (+1.3 to
  +3.2), least under the actor stream's relabelled goals (+0.5 from the progressing action, +1.3 from the start agent, -0.3 from the
  logged torque).  The 20k critic already carries the task / near / marginal bump (+1.2 / +2.8) with none yet under the relabelled goals
  -- the preference formed first for the goal-area goals and spread to the actor's goals by 30k.
* Put together with A: the draw-1 table paired the reset states with goal-area futures 2.3x more often than draw 2 (CORRECTION: and
  LESS often than the simulator table, 0.046 vs 0.059 -- the earlier '1.5x more often than the simulator' was wrong), so the draw-1 critic
  learned the reset states as places from which the goal area is reached more often than draw 2's critic did -- and the action dependence
  it fitted to that puts the maximum at the corner of the action box for the task / near-goal goals.  What pulled the critic is the goal-area /
  near-goal positives at the reset rows, not stalled futures; the stall is the actor's exploitation of an unbounded extrapolation in
  the action, not an imitation of stalled paths.

## Caveats

* 64 (196) reset anchors: the two draws' difference at the reset rows is a per-anchor sampling fluctuation of the generation
  (agreement 0.58-0.60), not a property of the ETT; a third draw would sit anywhere between.  The success difference 0.42 vs 0.16 on
  64 anchors is ~3 s.e.
* The Q surfaces are read on a straight line in action space between two policies' modes; other directions were not probed.  Both
  critics prefer their own actor's action; the comparison shows the SHAPE (monotone to the bound vs interior maximum), which is the part
  that matters for the actor's update.
* Not tested: whether the monotone shape appears whenever the reset positives are richer (other seeds / draws), or whether it is
  specific to this run; whether the same surface exists at the independent resets (not logged rows: no relabelled goals).
