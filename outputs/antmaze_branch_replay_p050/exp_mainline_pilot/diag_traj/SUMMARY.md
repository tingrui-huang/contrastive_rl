# Frozen-model trajectory diagnostic (no retraining): where the CF timeouts stop, and where the route signal acts

Script `scripts/diag_v6_pilot_trajectories.py`; tables `REPORT.md`; data
`rollout_check.json`, `timeouts.json`, `fork.json`, `cont.json` (per-step
trajectories `traj_*.npz` stay on node 30043).  Policies: the re-derived
d05 start agent, O seeds 0-2, CF seeds 0-2 (final checkpoints).  All rollouts
on the CPU of node 30043.

## 0. Replay fidelity: outcomes are chaotic at the episode level, stable in aggregate

Replaying the 300 evaluation episodes (same env seed, same reset order,
same tanh(loc) policy) reproduces the evaluation's (outcome, route, steps)
triple in only 61-122 of 300 episodes per policy; most differences are 1-2
steps, but some flip the outcome (CF s2 episode 3: evaluation timeout at
800, replay success at 436).  The cause is numerical: a GPU replay on the
same node also matches 2/6, and the branch primitive (state restored from
the float32 observation) matches its own CPU rollout in 11-23 of 40.
Action differences of 1e-7 grow through the Ant's contact dynamics.  The
aggregates are stable (replay vs evaluation: start 0.227 vs 0.243; O 0.263 /
0.260 / 0.250 vs 0.250 / 0.263 / 0.250; CF 0.280 / 0.323 / 0.497 vs 0.267 /
0.320 / 0.480; CF timeouts 0.093 / 0.110 / 0.243 vs 0.107 / 0.113 / 0.260).
Everything below uses the CPU replay as the reference; within the branch
primitive all variants of an episode share the same restored state, so the
swap and continuation tables are internally paired.  A side finding: whether
a CF detour is completed within 800 steps is itself sensitive to 1e-7
perturbations -- the CF walk is marginal, not just slow.

## 1. Where the timeouts stop (CPU replay; CF s2: 73 timeouts = 42 on the detour, 28 without a route label, 3 on the shortcut)

Class = last 100 steps: fallen (torso down / flipped), stopped (net < 0.5,
path < 3), wall_stuck, oscillating, wrong_direction, slow_but_moving.
Progress arc: detour 0-8 west column, 8-32 top corridor, 32-40 east column.

| CF s2 | class | n | where | what the trajectory shows |
|---|---|---|---:|---|
| detour (42) | stopped | 9 | goal area, arc 40 | reached the bottom of the east column, then stood still 1.5-2.7 from the goal for ~430 steps, torso up (z 0.60), torques saturated 0.86: a freeze in front of the goal, never inside the 0.5 success radius |
| | fallen | 10 | east column 4, top corridor 3, goal area 2, west column 1 | torso z 0.29 (normal 0.54); 6 had already reached the east column |
| | slow_but_moving | 14 | top corridor 10 | still progressing at step 800; first y >= 6 at median step 394 (successful detours: ~80): the west-column climb alone ate half the horizon |
| | oscillating | 9 | top-west corner 4, top corridor 3, east column 2 | progress until ~50 steps before the end, then back-and-forth |
| no route (28) | stopped | 14 | start 8, shortcut corridor 6 | frozen from step ~150 on (stall ~650 steps), saturation 0.57 |
| | oscillating | 7 | west column | up and down the west column at arc ~5, never reached y = 6 |
| | fallen | 5 | west column 4 | |
| | slow | 2 | | |
| shortcut (3) | stopped | 3 | | saturation 0.96 |

Seeds 0 / 1 (28 / 33 timeouts) show the same classes with more early
freezes (no-route stopped 6 / 10, wall-stuck 4 / 0) and fewer detour cases
(4 / 12).  The start agent's own 10 timeouts are 5 freezes 0.6-1.6 from the
goal on the shortcut (crouched, z 0.38, saturation 0.60) and 4 early freezes:
the "freeze" mode is inherited from the start agent and amplified by CF;
"fallen" (15 in CF s2, 1 in the start agent, 0 in O) and the slow
west-column climb are CF-specific.

So the three mechanisms behind the CF timeouts: (i) freezing with saturated
torques (start region and goal area, ~23 of 73), (ii) falling (15), (iii) a
slow or oscillating detour walk, worst on the west-column turn (~30).
"No route label" episodes are mostly frozen near the start or oscillating in
the west column -- not "standing at the start" in the sense of never moving.

## 2. Where the route signal acts

2a. Critic scores of the actual reset-state torques (`fork.json`; f = min
over the twin heads at the task goal, per-state comparison over 300 resets):
the CF critic of each seed prefers its own actor's torque over the O and
start torques (P(f(CF) > f(O)) = 0.92 / 0.86 / 1.00; over start 0.81 / 0.94
/ 1.00).  The O critics of seeds 0 / 1 also mildly prefer the CF torque
(0.84 / 0.58); O s2's does not (0.02).  The start agent's frozen vanilla
critic prefers the start torque over everything (P = 0.00 for both CF and O
torques).  So at the reset states the CF critics DO prefer the
detour-leading torque -- case "critic prefers it, actor did not adopt it"
does not apply to seed 2 and only partly to seeds 0-1: the CF actors' mode
sits above 60-71 % of their own 64 samples, and the best sample is ~1 nat
above the mode (-8.29 vs -9.25; s2 -7.14 vs -7.83), i.e. the actors have
not finished climbing their critics.  The CF policies' pre-tanh scale
widened to 0.75 / 0.74 / 1.30 (O 0.49 / 0.58 / 0.51): the BC term on the
teacher's shortcut torques is being absorbed by a wider distribution rather
than by the mode.

2b. First action swapped at the identical reset state and hidden draw (300
episodes each):

| continuation | first torque from start | from O_s | from CF_s | reading |
|---|---|---|---|---|
| start agent | detour 0.000 | 0.000-0.010 | s0 0.020, s1 0.043, **s2 0.143** | one CF s2 torque sends the never-detouring start agent around 14 % of the time |
| O_s2 | 0.113 | 0.003 | 0.140 | the O_s2 actor detours after a foreign first torque, not after its own |
| CF_s2 | 0.283 | 0.133 | **0.523** | with CF's own continuation, replacing only the first torque removes half to three quarters of the detours |
| O_s0 / O_s1 | 0.170 / 0.190 | 0.023 / 0.020 | 0.070 / 0.100 | the O actors are near a decision boundary: a start-agent first torque flips them to 17-19 % detours |

2c. Where the O and CF paths of one episode separate: seed 2 in 284 / 300
episodes at median step 7 (mean |xy| 0.4 from the reset), seeds 0 / 1 at
median 70 / 12; CF reaches y >= 2 at median step 26-31.  The single-action
swap at that step: on CF's own state, one O torque then CF continuing gives
detour 0.468 vs the control 0.553 (s2), 0.148 vs 0.169 (s0), 0.286 vs
0.274 (s1); on O's state, one CF torque then O continuing gives 0.011-0.055
vs O's 0.003-0.023.  Reading: the route is decided by the first ~7-30
torques as a sequence, not by any single one -- a single CF torque moves
the probability by 5-15 points, the CF continuation supplies the rest.

2d. Support in the CF critic's positive futures at the 196 reset-row
anchors: only 3 branches went around (the logged first torques are the
sighted teacher's shortcut torques and the continuation agent never turns);
success around 0.67 vs straight 0.29 (straight death 0.67); mean P_goal
identical (0.00114 vs 0.00114).  The reset rows themselves carry almost no
"around" support; the CF critic's reset-state preference for the turning
torque must come from generalisation over the 9,050 start-region anchors at
t in [6, 50) (success 0.25 / death 0.73 straight) and the detour-episode
anchors (success 0.91 at t >= 150) -- documented, not tested.

## 3. Case (c): entered the detour, could not finish -- continuation from the CF policy's own states

From the CF path's own state at the detour entrance (first y >= 2; n = the
detour timeouts): reach within the horizon

| seed (n) | CF itself | start agent | O_s | teacher's blind position driver |
|---|---|---|---|---|
| s2 (42) | 0.48 | 0.52 | **0.71** | **0.98** |
| s1 (12) | 0.42 | 0.58 | 0.58 | 1.00 |
| s0 (4) | 0.75 | 0.50 | 0.50 | 1.00 |

From 20 steps before the stall onset (all timeouts; s2 n = 73): CF 0.45,
start 0.42, O 0.42, driver 0.53.  From step 100 of the no-route timeouts
(s2 n = 28): CF 0.39, start 0.43, O 0.36, driver 0.64.  (The driver's step
counts in the table are not meaningful: it keeps stepping to the horizon
after reaching.)

Reading: from the detour entrance the states ARE completable (the blind
driver finishes 98-100 %), every d05-derived learned policy finishes about
half, and CF s2 is somewhat worse than O s2 from the same states (0.48 vs
0.71, n = 42, ~2 binomial s.e.); from the pre-stall states CF, start and O
are indistinguishable.  So the update did not damage local execution in a
way that the start or O policies escape -- the detour-walking competence of
every policy trained on 50 detour demonstrations is the limit (~0.5-0.7
completion from the entrance, ~0.4-0.5 from a stalled state), with at most a
weak CF-specific deficit on seed 2.

## What this settles and what it points at

- The CF success shortfall relative to its detour rate is (i) freezes with
  saturated torques at the goal and near the start (inherited from the start
  agent), (ii) falls, (iii) a slow / oscillating detour walk, worst on the
  west-column turn -- not wrong-direction walking and not wall collisions
  (wall_stuck 0-4 per policy).
- The route decision is a short action sequence at the reset (first ~7-30
  steps); the CF critics prefer the turning torque there, the CF actors
  follow it only partly (mode below the best of their own samples, widened
  scale), and the O actors sit near the same boundary.
- Walking the detour to the end is a competence shared by none of the
  learned policies at the driver's level; it is the second, independent loss.
- Not done (by design): no retraining, no new configuration; the
  generalisation path in 2d and the seed-2 continuation deficit are
  documented, not tested.

## 4. Entry times from the recorded xy (`entries.json`, REPORT section 4)

CF s2 turns north (y >= 2) in 62 % of episodes at median step 28, reaches the
top corridor in 53 % (median step 74) and the east column in 46 % (median
284); it enters hazard zone 1 in 37 % of episodes (start agent 95 %, O
91-94 %) and zone 2 in 18 % (46 %).  Every recorded death lies inside the
zone's burst window (78 / 78 for CF s2, 222 / 222 for the start agent), and
entering zone 1 while it is active is fatal almost always (52 / 52; start
143 / 152).  No detour episode of any policy touched a hazard zone.  CF s0 /
s1 turn north in 15 / 21 % and reach the top in 8 / 16 %: the seeds differ
in how often the turn is made, not in what happens after it.

## 5. The actor objective at real start-region rows (`objective.json`, REPORT section 5)

2,048 logged rows with x < 2, y < 2, t <= 5 (168 next-frame north-moving,
1,269 east-moving) and the 196 reset-row anchors; each actor with its own
critic; q-term = -f(s, a_sampled, g), BC = -log pi(a_logged | s, g),
weights 0.95 / 0.05, gradient norms w.r.t. the policy parameters.

- The CF critics prefer their actor's mode over the logged (teacher) torque
  at 76-87 % of the start-region rows (O: 54-58 %) and at 89-99 % of the
  reset rows (O: 69-82 %); the CF modes sit 0.8-1.2 (L2, torque units) from
  the logged torques at start rows and 1.4-2.9 at reset rows (O 0.4 / 0.7).
- On the logged torques themselves the CF critics of seeds 0 / 1 rate
  north-moving above east-moving rows (+0.21 / +0.33 nats); seed 2's does
  not (-0.19), nor do the O critics (-0.23 to -0.39) or the start agent's
  (-0.24).  Seed 2's preference lives on its own torques, not on the
  teacher's.
- Gradient balance of the actor update at these rows: for the O actors the
  weighted BC gradient is 2-5x the critic-term gradient (start region 2.9 /
  4.8 / 4.5; reset rows 1.7-2.2); for the CF actors the two are comparable
  (0.74 / 0.91 / 1.00; reset rows 0.63-0.93); the start agent is BC-driven
  (24 / 31).  The CF actors pay for the moved mode with a wider tanh-normal
  (scale 0.46-0.61 at start rows, 0.72-1.27 at reset rows; O 0.31-0.56) and
  a higher BC NLL (CF s2 at reset rows +2.0 vs O -8.0).

Reading of case (b) with these numbers: the CF actor's mode is where its
critic wants it at the fork (P 0.76-0.99 over the logged torque, above
60-71 % of its own samples), the BC term neither dominates nor vanishes
(gradient ratio ~1), and the compromise shows up as a widened policy rather
than a mode pulled back to the teacher's torque.  The residual "not fully
climbed" gap (best own sample ~1 nat above the mode) is what a longer or
lower-BC update would chase; whether that would change the route rate is
untested.
