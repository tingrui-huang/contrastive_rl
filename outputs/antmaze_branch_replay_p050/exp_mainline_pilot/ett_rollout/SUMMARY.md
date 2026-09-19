# Step 3a: full-length rollouts of the one-step model vs the held-out branches -- C FAILS (two specific mechanisms), A fails on top of them

`manifest.json` (thresholds sealed before the rollouts), `REPORT.md` /
`report.json`; probes in the session log (`rollout_failure_probe.py`,
`stall_probe.py`).  6,000 held-out anchors (2,000 per fold, cross-fitted
models), continuation = the start agent's mode, first query = the logged
torque, exact hazard integration along each path.

## C (simulator teacher advice along the model path): the motion + onset model alone

Pooled: death 0.294 (sim) vs 0.308 (model), far 0.012 / 0.012, KS death
time 0.084, KS death x 0.080, KS reach time 0.036, per-anchor AUROC of
P(death) vs the realised death 0.991 -- all within threshold; reach 0.615
vs 0.671 (gap 0.056) and timeout 0.091 vs 0.021 (gap 0.070) FAIL, and
inside the bands the death-time KS fails (zone 1 0.28, zone 2 0.76;
death x in start / zone 2).

Mechanism 1 -- stalls are not fixed points of the model.  The
simulator's timeouts are the start agent STOPPING (mean |torque| over the
last 100 steps 0.04; net displacement 0.0 in 99 % of them): the ant is
stationary in xy (per-step |dxy| = 0.0; the full 29-dim state still
settles by < 1e-3 per step in 27 % of all supervision rows, < 1e-6 in
7.7 %; no row is exactly constant).  Near-stationary rows (|dxy| < 0.005)
are 39 % of all supervision rows.  The regression's one-step error on them is
tiny (0.0007) but biased, so an open-loop rollout with the recorded
torques drifts 0.54 in 100 steps and 1.45 in 200; closed-loop the drift
changes the state until the policy re-engages (|torque| 0.03 -> 0.72 by
step 100) and the path walks to the goal: of the 549 sim timeouts the
model path reaches the goal in 89 % (sim final x median 13.7, model
24.7).  This is the case the PointMaze diagonal model handles with an
EXACT zero-displacement atom (there the stationary rows are exactly
constant; here the atom is an approximation of a settled state whose
residual motion is < 1e-3 per step).

Mechanism 2 -- the onset hazard is memoryless, the simulator's death is
a delay.  Inside zone 2 the sim deaths come 6-15 steps after the anchor
(none in steps 1-5 but one of 57); the model, with a flat per-step
hazard, puts 65 % of its death mass in steps 1-5.  Zone 1: sim median 53
steps (the rocks fall during the crossing), model 27 % of the mass in
steps 1-5.  Totals and who-dies are right (AUROC 0.99); when is wrong.  A
head that sees only (s, a_b, a_q, delta) cannot represent the delay
since the advice turned to hold / since the band was entered.

## A (memoryless nominal P(a_b | s), the protocol control)

Death 0.294 vs 0.023, reach 0.615 vs 0.928, AUROC 0.71 (0.45-0.57 within
strata): the memoryless nominal almost never produces "hold" inside a
band (the log never shows it), so the model almost never dies -- the
futures would be optimistic.  A shares mechanism 1 (timeouts 0.05 vs
0.091).  Per the user's reading order this is moot until C passes.

## Decision (user's rule: C fails -> fix the motion / onset model first)

Revision v2, disclosed, both parts on the visible history only: (a) a
stationary gate with an EXACT zero-displacement atom (P(stationary | s,
a_b, a_q) trained on all rows; the motion regression trained on moving
rows only; at rollout the atom applies when the gate says so), the
project's own diagonal-model design; (b) two visible-history features for
the onset head -- steps since the advice turned to hold on this path and
steps since the current hazard band was entered -- computed from the
path itself (never the hidden clock, never lengthening a hold).  Then
the preflight and this full-length diagnostic again (C and A).
