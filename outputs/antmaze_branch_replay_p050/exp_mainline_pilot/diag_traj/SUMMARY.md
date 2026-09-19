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

## 6. Route choice at the starts where CF still went straight (`candidates.json`, REPORT section 6)

> Corrected 2026-09-19 after the user's review of 52f0bf1: (i) the share of
> states with a detour-realising candidate is 74 % / 82 % / 96 % per seed
> (203 / 275, 207 / 252, 134 / 140), not "97 %"; (ii) the first Spearman
> used two argsorts and mis-ranked the many tied outcome labels -- with
> average ranks (`candidates_restat`) it is +0.22 / +0.14 / +0.04 (success-
> only label +0.15 / +0.08 / +0.02), so seed 2 reads as "weak overall
> ordering on this candidate set", not as a reversed ranking.  The label
> (2 x success + detour) is a diagnostic combination, not the critic's
> discounted target.

At every reset state where the CF policy's replay went straight (275 / 252 /
140 states for seeds 0 / 1 / 2), 23 candidate first torques -- the CF mode
(control), 12 samples of its own tanh-normal, the other two CF seeds' modes,
the O and start modes, 4 detour-episode and 2 shortcut-episode logged
teacher reset torques -- each executed once under the episode's own hidden
draw with the SAME CF policy continuing; every candidate scored by the CF
critic at that state.

- A detour-realising candidate exists at 74 % / 82 % / 96 % of these states.
  The most reliable ones are the teacher's detour reset torques (detour
  0.25 / 0.35 / 0.44 vs the mode 0.03 / 0.04 / 0.10), then CF s2's mode
  transplanted (0.27 / 0.21) and CF's own samples (0.09 / 0.10 / 0.26); O's
  mode 0.02-0.06, the teacher's shortcut torques 0.00-0.02.
- Seeds 0 / 1: the decisive fact is not that the teacher torques score
  high but that **choosing by the critic's score improves the realised
  outcome**: the critic's top-1 candidate succeeds in 109 / 122 states
  against the mode's 62 / 62 and realises the detour in 96 / 124 against 7 /
  9 (the top-1 is a teacher detour torque in 241 / 227 states; a teacher
  detour torque is scored above the mode in 267 / 275 and 241 / 252 states,
  +0.60 / +0.24 nats).  So "no usable action" and "the critic cannot
  recognise it" are not the main explanation for these seeds: **the policy
  does not fully use the scoring information it has.**  Why is NOT
  established: the actor optimises the mean critic score of its sampled
  actions plus BC, not the single best action; comparable gradient norms
  (section 5) do not show that the two terms oppose each other; "BC holds
  the mode in the middle" remains a conjecture, and nothing here licenses
  lowering BC.
- Seed 2: the critic's top-1 is one of the actor's own samples in 88 / 140
  states and the mode in 12; the teacher detour torques -- the candidates
  that realise the detour most often (0.44) -- are scored 2.4 nats below
  the mode (above it in only 48 / 140 states), as is every other off-policy
  candidate (O -2.2, start -2.4, other CF modes -1.8 / -1.9).  Its top-1
  still detours in 51 states vs the mode's 14 (success 54 vs 40).  So seed
  2's scores are **biased by candidate source** (off-policy candidates
  pressed down) while the within-state ordering is weak (+0.04).  Whether
  this is critic-actor co-adaptation or simply poor critic generalisation
  outside the region the actor moved into is not decided by this data (the
  NCE critic trains on a fixed stream; the actor adapts to it) -- and it
  means the seed-0 / 1 reading ("use the scores harder") cannot be carried
  over to seed 2 without also amplifying its scoring error.
- Caveat: one chaotic rollout per candidate; the "best candidate succeeds"
  column (210 / 211 / 129) is an optimistic oracle over 23 draws, not a
  policy result.

## 7. Paired continuations where O finishes and CF does not (`pairs.json`, `pairs_traj.npz` on node3, REPORT section 7)

From the 42 + 12 + 4 detour-entrance handover states of section 3, the 19
pairs where the O policy finished and CF did not (all 19 reproduced on the
captured rerun), and the 8 reverse pairs.  Earliest anomaly in the failing
continuation, judged against the finishing one from the same state:

| | pairs | first anomaly: stall / posture / slowdown / none | median step after handover | failing arc reached | failing outcome | mean speed fail / ref | start policy finishes these |
|---|---|---|---|---|---|---|---|
| O finishes, CF not | 19 | **16** / 1 / 0 / 2 | 63 | west column 6, top corridor 8, east column or goal 5 (median 12) | timeout 19 / 19 | 0.062 / 0.114 | **13 / 19** |
| CF finishes, O not | 8 | 5 / 1 / 2 / 0 | 138 | 2 / 3 / 3 (median 21) | timeout 8 / 8 | 0.065 / 0.110 | 4 / 8 |

Reading: when CF fails from a state O completes, the first thing that goes
wrong is an early **stall** -- progress stops within ~20-100 steps of the
handover while the O continuation keeps moving, at half O's speed, mostly
in the west column or the first half of the top corridor; posture
instability appears only later (3 of the 19, at 330-380 steps) and never
first except once; two failures are goal freezes after a complete detour.
The start d05 policy finishes 13 of these 19 states.  So on this subset the
CF update did regress local execution relative to the pre-update policy
(the aggregate completion from the entrance, start 0.52 vs CF 0.48, hid
it): the deficit is a stall / slow-progress mode in the detour legs, not a
fall and not a wrong turn.  The 8 reverse pairs show the same update also
improved other states: the change in walking competence is uneven, not a
broken walker.  Together with section 6: **the update learned a safer
route choice at some states and lost a continuing-forward behaviour at
others** -- two opposite needs (change the reset torque; keep the mid-route
torques) that a single BC coefficient cannot serve, which is why no global
BC change or longer training is proposed from this data.

## 8. Pre-registered single-change variants (reference points; `../variants/SUMMARY.md`)

Four variants were run after this diagnostic, one change each against the
pilot: actor lr 1e-4 (walking recovered, route gain to a third), bc 0.02
and bc 0 (no walking at all: 300 / 300 timeouts, bc 0 in both arms),
anchor-to-start 0.5 (walking recovered, route gain gone).  They move along
one axis -- how far the actor may leave the start policy -- on which the
route gain and the walking loss come together; none separates them and
none is a fix.  The first of them is described below as written before
the others ran.

### 8a. Actor learning rate 3e-4 -> 1e-4, everything else fixed

Motivated by section 7 (an update that lost local competence the pre-update
policy still has); not a proven fix and no cure for critic blind spots.
Sealed in `../variants/actor_lr1e-4/manifest.json` before training: O and
CF both, three paired seeds, same data / anchors / branches / streams /
losses (bc 0.05) / critic lr / initialisation / 30,000 updates / evaluation.
Criteria: (1) native success CF(variant) - CF(base) on the common episodes;
(2) the detour gain kept; (3) reach rate from the SAME handover states of
section 3 (`cont_variant_actor_lr1e-4.json`).  Readings fixed in advance:
walks well but no detours = not a fix; continuation recovered and detour
kept = an actionable handle; still stalls = not the update speed (then
check whether the critic rewards the wrong action before the stall).
Result (`../variants/actor_lr1e-4/SUMMARY.md`): criterion 3 recovered (from
the entrance states CF(lr1e-4) reaches 0.75 / 0.67 / 0.71 vs base CF 0.75 /
0.42 / 0.48, start 0.50 / 0.58 / 0.52; native timeouts 0.01-0.03), but
criterion 1 not met (success -0.068 vs base CF, 2/3 negative) and the detour
gain shrinks to a third (0.26 -> 0.07 mean, 3/3; still +0.044 over O and
+0.066 over the start, 3/3).  Reading: "walks well, few detours" -- the
route change and the walking loss scale together with the size of the
update; the actor learning rate does not separate them.  Not a fix.

## 9. Two checks that decide the fix direction (frozen models; `scripts/diag_v6_stall_critic_check.py`, `scripts/diag_v6_objective_checks.py`)

**9a. Does the CF critic reward the stalling torques?** (`stall_critic_check.json`)
Along the CF continuation's own states before the stall in the 19
O-finishes-CF-not pairs (673 states, every 5th step), the CF critic scores
the CF torque vs the O torque at the same state +0.04 nats (P 0.56) and vs
the start torque +0.08 (P 0.58), although the torques differ by 0.76-0.80
(L2); after the stall +0.10 / +0.39.  The O critic is indifferent (-0.007,
P 0.49).  The critic is nearly flat among these torques: it neither rewards
the stall nor points back to walking.  (In the 8 reverse pairs the stalled O
torques are 1.65 away and both critics score them 1.7-2.5 nats lower -- far
off-distribution stalls are recognised, the CF policy's own are not.)

**Correction to 9b and 9c (2026-09-19, after the user's review; `../diag_replay/SUMMARY.md`).**
9b paired the start rows with the episode's TASK goal; the actor trains on
RELABELED goals.  At real actor-stream start rows (real goals) the critic
term does not prefer the teacher detour torques over the actor's own samples
(P 0.34-0.42) nor over the logged torque (P 0.36-0.44), and the full
objective is worse toward them in 96-98 % of rows (+1.7 to +1.9 nats); under
the task goal on the same rows the critic term is +0.2 and the total still
+1.3 to +1.4 worse.  The "exact tie" below is a task-goal, own-candidate
result and is withdrawn as a statement about the training objective.  9c
used one plain SGD step on the final checkpoint; the training replay (a
checkpoint every 1,000 updates, saved Adam state) shows walking collapses
and recoveries before and without any route change -- the single-step
"interference" reading is withdrawn as the explanation of the history.

**9b. Start: the full actor objective, not the action score.**
(`start_objective.json`; 1,024 real start-region rows with their own logged
actions; the critic-preferred candidate at each row -- teacher detour reset
torques in ~47 % of rows, the actor's own samples in ~40 % -- and the
objective along the straight loc path toward it, lambda 0..1.)
- Scale held: E_pi[f] improves monotonically (+0.5 to +1.0 nats for the CF
  actors, P 0.95-0.98) but the BC NLL of the logged action explodes (-10 ->
  +400) and the total is worse at the target in 77 % of rows, monotonically
  -- no barrier, the objective simply does not want the move.
- Scale widened as the mode moves (sd = max(current, half the shift) -- the
  compromise the base CF actors actually made): the CF actors' total is
  FLAT along the path (CF s0 9.34 -> 9.44 -> 9.32 -> 9.23 -> 9.24; end
  -0.10 / +0.05 / +0.05; P(better) 0.43-0.51): the E_f gain (+0.5 to +0.8)
  is cancelled by 0.05 x (NLL +11 to +14).  For the O actors the total gets
  worse (+0.4 to +0.55; their E_f gain is only +0.3 to +0.4).
Reading: at bc 0.05 the full objective at the reset rows is a near-tie
between staying in the shortcut basin and moving (with a wider policy) to
the critic-preferred detour torque -- an objective trade-off at exact
balance, not an optimisation barrier and not a missing signal.  Where the
mode ends up on that flat ridge is set by the update budget and noise, which
is consistent with the seed spread of the detour rate (0.11 / 0.17 / 0.50).

**9c. Mid-route: where does the real actor update push the outputs?**
(`midroute_update.json`; the pre-stall CF states of the pairs: seed 2 642
states, seed 1 31.)
- The critic term's action gradient at the CF mode: |grad_a f| 0.7-0.9 (1.5-
  1.9 at reset rows); its projection toward the O torque +0.001 / +0.008
  (P > 0: 0.51 / 0.52), toward the start torque +0.015 / -0.031 -- orthogonal
  to the walk-vs-stall direction.  The critic term does not push toward the
  stall (and not back either).
- One REAL actor step on real actor-stream batches (the training seed's
  stream, the same actor loss; SGD probe of size lr): the induced change of
  the mode at these mid-route states is |delta| ~ 0.009-0.010 per step, of
  which 87-88 % comes from the batch's shortcut-corridor rows (their batch
  share 0.87-0.88), 8 % from start rows and 4 % from detour-leg rows
  (|delta| 0.0003-0.0004); its projection on the drift direction and toward
  O is ~0 (+-0.0008) -- isotropic interference, no directed push.
Reading: the mid-route regression is parameter interference (case 2 of the
user's split), not a critic that rewards the wrong action (case 1): the
detour-leg outputs are moved every step by gradients from elsewhere and the
rows that could hold them are 4 % of the actor batch.  A slower update
shrinks the interference and the reset-row movement alike (the lr trial).

What the two checks are for (user, 2026-09-19): not to pick a fix but to
decide whether there is evidence that the problem can be repaired at the
sampling layer, or whether it lies outside the boundary the contract draws
around the method (single-step torques, offline data, the original NCE and
actor objectives with bc 0.05, the recipe's actor stream, counterfactual
futures as the only intervention).  "Rebalance the BC rows" and "protect
walking by region" are withdrawn as default proposals; the running bc0.02 /
bc0 / anchor_start0.5 variants stay as reference points only.

Assessment against that question:
- The intervention itself is doing its part at the critic level: the CF
  critics prefer the turning torques at the reset rows (sections 2a, 6);
  nothing in 9a-9c points at a missing or wrong critic signal at the start.
- What stops the signal from becoming behaviour is the actor objective at
  bc 0.05 on the recipe's rows: an exact tie at the reset rows (9b) between
  staying and moving.  The tie is a joint property of the objective and the
  row composition (95 % shortcut logged torques at the reset); tipping it
  means either changing the objective (BC weight, an added term) or the row
  composition -- both outside the boundary as written.  bc 0.02 shows the
  objective side has no slack: below 0.05 the actor leaves the walkable
  manifold entirely (success 0.000, timeout 1.000 on all seeds).
- The mid-route regression is interference from the majority rows (9c), not
  a critic error; within the boundary the only lever that touches it is the
  counterfactual futures at mid-route anchors (a better continuation agent
  would sharpen f there; untested), and even a sharper f cannot hold the
  outputs against interference unless the actor rows or the objective do.
- So: on this evidence the actor-side limits (the tie, the interference)
  are not addressable by the futures intervention alone; repairing them at
  the sampling or objective layer is a boundary decision, not something
  these diagnostics can settle.  What remains inside the boundary and
  untested is the next policy-iteration round (the improved agent as the
  continuation).

## 12. C1 / C2 (2026-09-19, night): the user's verified candidates at their original state, and transplants at the real training rows (clipped models)

The user's frozen-model diagnostic (`codex_clip_choice_v1`, 72 contexts on
the clipped CF checkpoints) found first torques that the seed's own critic
scores above the policy's mode AND whose realised continuation entered the
far route.  Two separate checks, as the user specified (a candidate is
verified only at its own state; elsewhere it is a transplant):

**C1, the original state + task goal** (`diag_v6_candidate_at_origin.py`,
`candidate_at_origin_critic_clip0.1.json`; 100 verified candidates over the
72 contexts, 23 of them in the 16 stable-shortcut contexts).  Primary
reading = the actor's critic term with the distribution moved to the
candidate at the SAME scale.  Stable-shortcut contexts: the moved
expectation exceeds the current E_f in 20 / 23 (mean +0.51), the critic's
action gradient at the mode points toward the candidate in 13 / 23; but the
BC pull that would resist the move -- the nearest 8 dataset reset rows'
teacher torques (detour share 0.04-0.16), NLL at the moved loc -- outweighs
it: the objective proxy 0.95 (-E f) + 0.05 NLL improves in only 5 / 23 (own
samples 2 / 7, teacher / other-seed torques 3 / 16; those sit 6-9 scale
units from the current loc, own samples 1.3-1.8).  At the same nearest rows
with their REAL relabeled goals the critic term prefers the CURRENT policy
in 84-97 % of rows (mean delta -2.0 to -3.2).  All contexts (100): moved >
current 81 / 100, proxy better 38 / 100 (own samples 17 / 46).

**C2, transplants at the actor's real start-region training rows**
(`diag_v6_actor_objective_stream.py --variant critic_clip0.1 --candidates`,
`actor_objective_stream_critic_clip0.1.json`; 3,670-3,899 shortcut-episode
rows per seed; candidates = 4 teacher detour reset torques + the user's 8
top-margin candidates per seed, all transplants).  Under the REAL relabeled
goals the critic signal (best candidate - E_f) is +0.49 / -0.02 / +0.13
(P > 0: 0.67 / 0.48 / 0.55) for the clipped CF critics and negative for
the O critics; under the task goal on the same rows +1.08 / +0.76 / +0.47
(P 0.88 / 0.77 / 0.67).  The full objective along the loc path toward the
best candidate: held scale P(better) 0.00-0.02 (the BC NLL explodes at
scale 0.1); widened scale P(better) 0.13 / 0.08 / 0.09 (O: 0.00-0.01).

**Reading, corrected after the user's review (2026-09-20).**  At the
decision states the clipped CF critics do prefer the verified candidates'
neighbourhoods under the task goal, and the local gradient mostly points
at them.  What C1 / C2 show about why they are not adopted is limited:
C1's BC term is a PROXY (the logged torques of the 8 nearest reset rows,
not the real training batches acting through the shared parameters);
C2's candidates are transplants whose effect at other rows is
unverified; and "5 / 23 objective better" counted only the endpoint of
the loc path -- at some intermediate position of the path 9 / 23 improve
(sampling error applies to both counts).  Also, a training row whose
relabeled goal lies in the shortcut corridor asks for the shortcut: that
row's preferred action and the task-goal decision are two different
tasks and cannot be equated.  What can be said: BC and the real relabeled
goals MAY weaken the gain of moving toward these detour torques; it is
NOT shown that the actor has correctly optimised its full objective, nor
that policy extraction has no room.  The decisive check is a REAL-UPDATE
one (next): from a checkpoint copy with the real actor batches and the
real Adam state, does the actual update move the actions at the verified
states toward or away from the verified candidates, with the critic-term
and BC-term contributions separated and the scale watched.

## 13. The real-update check (2026-09-20; `scripts/diag_v6_actor_real_update.py`, `real_update/real_update_seed{s}[_bc{0,1}].json`)

**Design (user's specification).**  Contexts = the user's diagnostic
contexts where the clipped CF actor stably takes the shortcut and at least
one far-route candidate torque is simulator-verified (11 / 11 / 9 for
seeds 0 / 1 / 2); the target c_v = the verified candidate with the highest
critic score under the TASK goal at the ORIGINAL state (no transplants).
The learner is rebuilt exactly as `train_arm` (clip 0.1) from the seed's
CF checkpoint (30,000 updates; Adam state included), the actor / critic
streams are fast-forwarded to the checkpoint's batch counter, and 2,000
REAL updates are applied to a copy.  Every 30 updates the verified
states are probed: the mode's L2 distance to c_v (8-dim torque space),
the scale, f(c_v) vs f(mode) under the critic, and the actor gradient
decomposed into its critic term and its BC term (loss = bc * BC_NLL +
(1 - bc) * critic term), each projected on the parameter direction that
moves the modes toward the candidates (cosine); the realised Adam step is
projected on the same direction.  Counterfactual replays: bc 0 (critic
term only) and bc 1 (BC term only), same batches, same Adam state.

| bc | seed | n | mode-candidate distance start -> end (max) | contexts closer at end | scale (verified states) start -> end | f(c) > f(mode) start -> end | critic-term cos (share > 0) | BC-term cos (share > 0) | realised-step cos (share > 0) | training-row scale first -> last |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0 | 11 | 2.20 -> 2.18 (2.75) | 5 / 11 | 0.65 -> 0.68 | 0.36 -> 0.27 | +0.058 (100 %) | -0.009 (35 %) | 0.000 (49 %) | 0.077 -> 0.076 |
| 0.05 | 1 | 11 | 2.25 -> 2.27 (2.32) | 6 / 11 | 0.60 -> 0.52 | 0.45 -> 0.45 | +0.080 (100 %) | -0.019 (19 %) | 0.000 (49 %) | 0.077 -> 0.075 |
| 0.05 | 2 | 9 | 3.10 -> 3.03 (3.34) | 7 / 9 | 0.53 -> 0.54 | 0.78 -> 0.78 | +0.053 (99 %) | -0.010 (18 %) | 0.000 (57 %) | 0.080 -> 0.078 |
| 0 | 0 | 11 | 2.20 -> 3.17 (3.25) | 0 / 11 | 0.65 -> 0.0001 | 0.36 -> 0.36 | -0.020 (43 %) | -- | -0.002 (28 %) | 0.076 -> 0.001 |
| 0 | 1 | 11 | 2.25 -> 2.14 (2.34) | 5 / 11 | 0.60 -> 0.0006 | 0.45 -> 0.18 | -0.013 (46 %) | -- | +0.002 (78 %) | 0.076 -> 0.001 |
| 0 | 2 | 9 | 3.10 -> 2.10 (3.12) | 8 / 9 | 0.53 -> 0.0002 | 0.78 -> 0.22 | +0.015 (60 %) | -- | +0.002 (60 %) | 0.079 -> 0.000 |
| 1 | 0 | 11 | 2.20 -> 2.73 (2.76) | 6 / 11 | 0.65 -> 0.46 (1.29 after 4 updates) | 0.36 -> 0.82 | -- | +0.015 (60 %) | 0.000 (54 %) | 0.249 -> 0.064 |
| 1 | 1 | 11 | 2.25 -> 2.59 (2.75) | 2 / 11 | 0.60 -> 0.30 (1.18 after 4) | 0.45 -> 0.73 | -- | +0.001 (51 %) | 0.000 (51 %) | 0.253 -> 0.076 |
| 1 | 2 | 9 | 3.10 -> 3.15 (3.48) | 4 / 9 | 0.53 -> 0.45 (1.18 after 4) | 0.78 -> 0.89 | -- | +0.015 (66 %) | 0.000 (57 %) | 0.271 -> 0.067 |

(cosines are means over the logged probes; realised-step norms 0.19-0.22
per update for bc 0.05 / 0, 1.8 falling to 0.22 for bc 1.)

**What the default replay shows.**  The critic term's descent direction
points toward the verified candidates in every probe of every seed
(cosine +0.05 to +0.08: small, but consistent); the BC term's direction
points away in most probes (-0.01 to -0.02; positive in 18-35 %); the
realised parameter step is orthogonal to the candidate direction (cosine
0.000; positive in 49-57 % of probes).  After 2,000 real updates the
modes are where they started (distance 2.20 -> 2.18, 2.25 -> 2.27,
3.10 -> 3.03; seed 0 drifted to 2.75 in between and came back), the
scale at these states stays 0.5-0.7 (0.077 on the actor's own training
rows), and the critic's own ranking prefers the candidate over the mode
in only 27-45 % of the seed-0 / seed-1 contexts (78 % for seed 2).  So
the real training neither adopts nor fights these choices: its updates
are driven by batches that hardly touch these state-goal pairs, and the
weak critic pull and the weak BC push leave the mode drifting.

**What the counterfactual replays show.**  Critic term only (bc 0): the
policy scale collapses within 2,000 updates (training rows 0.077 ->
0.001, verified states 0.6 -> 0.0001) -- the actor-saturation failure
that BC 0.05 prevents; the mode moves AWAY from the candidates for seed
0 (2.20 -> 3.17, 0 / 11 closer) and toward them for seeds 1 / 2 (2.25 ->
2.14, 3.10 -> 2.10) while the share of contexts where the critic scores
the candidate above the mode FALLS (0.45 -> 0.18, 0.78 -> 0.22): the
critic-driven mode goes to other maxima of the critic, not to the
verified far-route torques (whether those maxima are far-route actions
is not checked here).  Once the scale is ~0 the cosine diagnostics lose
meaning (the expected score is f(mode)).  BC term only (bc 1): the Adam
moments were calibrated for the 0.05-weighted BC gradient, so the first
~100 updates take 8x larger steps (scale 0.65 -> 1.2 after 4 updates,
decaying to 0.3-0.46; a replay artefact, disclosed); the mode moves away
from the candidates (2.20 -> 2.73, 2.25 -> 2.59, 3.10 -> 3.15) toward
the logged (shortcut) torques, and f(c) > f(mode) rises to 0.73-0.89
because the mode leaves the critic's preferred region.

**Reading under the user's decision tree.**  "Critic supports, BC
cancels" is only weakly supported: the BC term does oppose (cosine -0.01
to -0.02), but removing it does not bring the mode to the candidates --
the critic's own dynamics collapse the scale and settle at other critic
maxima.  The closest branch is "the real critic update itself does not
support these choices": the critic's landscape at these state / task-goal
pairs, as shaped by the current futures and queries, has the verified
far-route torques as a direction of weak improvement, not as an
attractor.  Per the user's rule this puts query coverage in the training
contexts (the current policy's one-step torque queries under the sampling
contract) ahead of any change to BC.  Limits: 2,000 updates of a
30,000-update run; 9-11 contexts per seed, one verified candidate each;
the distance is torque-space L2 to that one candidate (other far-route
torques may exist); bc 1 carries the Adam mismatch above; nothing here
says the actor optimised its objective wrongly -- the objective at these
states simply has little to gain along this direction.
