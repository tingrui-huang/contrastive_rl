# AntMaze V6: branch replay with interventional futures -- plan (pre-registration draft)

What worked on PointMaze (feature/pointmaze-causal-transition, Steps 12-13
of `notes/pointmaze_actor_fork_diagnosis.md`): the critic's positive futures
come from a transition model rolled out of EVERY recorded anchor
(p(g | s, do(a)) instead of the recorded continuation), anchors are the
recorded (s, a) rows, the BC term keeps imitating recorded actions with
its action regions balanced inside (state, goal) groups, the critic is
judged by its margin at the decision state on the training goal frames
before any actor is trained, and the actor is either trained on the frozen
critic or jointly after a critic warm start.  Five critic seeds of five and
15/15 (two-stage) / 5/5 (joint) actors took the safe route.

This note carries that recipe to the long two-rockfall AntMaze V6 benchmark
(`crl/rockfall_clock_v6.py`, dataset
`artifacts/rockfall_clock_v6/dataset/antmaze_rockfall_clock_v6_p040_gxy.npz`,
1000 episodes, horizon 800, XY goal).  The environment, the dataset, the
teacher and the evaluation protocol are NOT changed.  Work happens on
`feature/antmaze-causal-transition`, local commits only.

## 0. The arithmetic first (no training)

The CRL critic learns the discounted goal-occupancy ratio.  Whether the
safe route can win at all is decided by the relabeling law before any
model exists.  From the V6 sidecar: successful shortcut episodes reach the
goal at ~222 steps (no wait), successful detours at a median 365; a blind
shortcut survives with 0.6 x 0.6 = 0.36 (measured vanilla success 0.37),
the detour with 0.98.  Goal mass = sum of gamma^k over the frames parked
at the goal until the horizon:

| gamma | blind shortcut 0.36 x m(222) | detour 0.99 x m(365) | log ratio detour / shortcut |
|---|---:|---:|---:|
| 0.99 (frozen V6 recipe) | 3.85 | 2.49 | **-0.44** |
| 0.995 | 22.4 | 28.2 | +0.23 |
| 0.999 | 126.6 | 242.5 | **+0.65** |

At gamma 0.99 a CORRECT interventional critic prefers the shortcut; the
benchmark is then not solvable by any discounted-occupancy learner and no
fix on the causal side can change that.  Decision: the learner's discount
is 0.999 for every arm of this study, vanilla included (it is a recipe
constant, not an environment property; the evaluation does not depend on
it).  +0.65 is the same order as PointMaze's P target (+0.88 at the
actors' width), which produced 15/15.

Two facts from the V5/V6 diagnostics carry over unchanged: the hazard
latent is not separable from the 29-dim observation at the mouth
(permutation d' not significant), and the observation carries no clock, so
a blind "wait k then go" is not representable as a state-conditioned
policy; the only blind-safe choice is the detour, decided in the start
region (x < 2: north = detour, east = shortcut).  The decision state for
every probe is therefore the start region, not the mouths.

## 1. What is reused, what changes

Reused as is: the branch-replay rule (interventional future for every
anchor, anchors = row 0 of every path, `TrajectoryBuffer.set_anchor_strata`
from the PointMaze learner commit cf407d0, to be cherry-picked), the
pre-freeze `prepare` hook, `Config.bc_dataset`, per-iteration milestones,
balanced BC rows (`crl/bc_balanced.py`), the critic gate before actors,
two-stage then joint-with-warm-start, the 300-episode evaluation with the
U00/U10/U01/U11 breakdown, mode and sampled policies.

Changed for the Ant:

* **Intervention granularity.**  A single 8-dim torque has no route
  effect; the causal action is a macro-action executed by the frozen
  walker with the teacher's leg tables (`scripts/rockfall_clock_v6_teacher.py`:
  `_drive(o58, legs)`, shortcut legs with `force_go`, detour legs).  The
  first segment of every branch replays the recorded torques for K = 25
  steps (the anchor's own action, as on PointMaze), then the blind driver
  continues on the route the position implies (y >= 6: detour legs,
  else shortcut legs, never consulting a schedule).  At start-region
  anchors both routes are branched (the query coverage of arm C).
* **The model.**  Phase 1 uses the simulator as the model (an oracle ETT,
  explicitly an upper bound outside the offline setting): restore the
  Ant's 29-dim state at the anchor, park the rocks, and REDRAW U1, U2 and
  both clocks from their priors -- the interventional continuation of a
  blind agent, which is exactly what removes the teacher's u -> a leak.
  Phase 2 learns the model from native same-context intervention tuples on
  the reduced state (XY track + death onset given a macro-action), with
  PointMaze's preflight gates.
* **Balanced BC key.**  Action sectors do not exist in 8-dim; the region
  key is the visible next-frame XY displacement direction (8 sectors +
  still), inside (state cell, goal cell) groups, cap 0.25 as before.
* **Memory.**  All 262k recorded rows branched to the horizon is ~60 GB.
  Futures must NOT be windowed: with the detour reaching the goal at step
  365, a 400-step window leaves it 35 parked frames and flips the target to
  -0.69 (600 steps: +0.46; the full horizon: +0.65).  Every branch runs to
  the episode horizon; the anchor count is cut instead -- every 20th
  recorded step (~13k paths, mean remaining horizon ~400 steps, ~650 MB of
  31-column rows) plus the start-region query branches.  Node RAM (15 GB)
  is the binding constraint, not GPU memory (batch 1024 through two
  1024-unit layers is well under 1 GB of VRAM); at most three learner
  processes at a time, results pulled per stage.
* **Gate.**  At start-region anchors of the recorded data, with the task
  goal (24, 0): mean critic score over recorded actions whose next
  displacement points north minus those pointing east, at the policy's
  own width; and the law's own version of the same quantity from the
  branch replay.  Pass line to be set from the law's number (PointMaze:
  gate = the value of a critic whose actors all detoured).

## 2. Phases, budgets, stop rules

* **Phase 0** (CPU, one day): section 0 with the anchor subsampling; the
  observational counterpart (recorded futures: the sighted teacher's
  shortcut success 1.0) for reference; the placed-on-detour continuation
  audit (does the blind driver complete the perimeter from the north leg?
  the 54 recorded detours say the teacher does; the driver is the
  teacher's own leg table, so this should be 1.0 up to the known 2-4%
  timeouts).  Go/no-go: the interventional target at the start region
  must be >= +0.5 nats at gamma 0.999 (full horizon: +0.65).
* **Phase 1** (oracle-ETT branch replay, GPU node): generate the replay
  (~13k branch paths + ~3k start-region query branches; MuJoCo on the
  node's CPUs, ~1-2 h); critics x 5 seeds, V6 recipe at gamma 0.999,
  100k updates, anchors = row 0; gate; three balanced-BC actors per
  passing critic on the recorded data; 300-episode evaluation (mode and
  sample, U breakdown); then the joint warm-started schedule on the
  passing critics.  Vanilla CRL at gamma 0.999 on the recorded data is
  the control (3 seeds).  Read: if the oracle-model arm does not detour,
  the CRL side is the limit and Phase 2 is pointless.
* **Phase 2** (learned macro-ETT): native same-context intervention tuples
  at matched contexts (state restore + macro-query + K-step outcome and
  onset), model of (XY track, onset | s, macro-action) with PointMaze's
  preflight gates (energy score, action sensitivity, north exits
  represented, onset AUROC, zero recovery); replace the oracle in Phase 1
  and repeat.
* **Phase 3**: the actor recipe details (BC key, cap, joint warm start)
  only if Phases 1-2 pass.

Nothing here changes `crl/rockfall_clock_v6.py`, the dataset bytes, the
teacher or `eval_rockfall_clock_v6_baseline.py`.

## Phase 0 result (2026-09-17, `scripts/build_v6_branch_replay.py phase0`)

400 recorded start-region anchors (x < 2, y < 2, t <= 60; half of them at
t <= 5) branched north (detour legs) and east (shortcut legs, blind go)
from the restored Ant state with hazards and clocks redrawn; 60 recorded
detour rows on the north leg and 60 on the east leg driven on with the
detour legs.  Three conventions for what happens after the goal, in the
order they were tried:

1. Paths end at the first goal reach (the collector's convention).  The
   law's goal frame is then the single last row, diluted by the path
   length (mass ~ 0.999^T / T): +0.27 at t = 0, +0.15 at t <= 5, negative
   over all anchors.  Not a property of the routes.
2. The driver keeps driving after the reach.  The shortcut legs hover near
   the goal, the detour legs push south into the wall: P_goal 0.137 vs
   0.007, "target" -3.0.  A controller artefact, discarded.
3. **Hold with zero torque after the reach** (goal states absorbing; the
   teacher's own waiting action; identical for both routes).  This is the
   convention of the replay.

| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |
|---|---:|---:|---:|---:|---:|
| t = 0 | 42 | 1.00 | 0.38 | **+0.60** | +0.58 |
| t <= 5 | 200 | 0.92 | 0.33 | **+0.58** | +0.57 |
| t <= 10 | 266 | 0.86 | 0.32 | +0.52 | +0.51 |
| 11 <= t <= 60 | 134 | 0.16 | 0.36 | -1.38 | -1.37 |

The blind north turn executes only near the reset pose (x < 0.5: reach
0.94; x >= 1: 0.00), so the query branches of the replay are placed at
t <= 5 (every second step), where the decision the deployed actor faces
(t = 0, the canonical pose) lives; later anchors carry only their natural
continuation, and for them the law correctly says the detour is no longer
available.  Placed on the detour, the driver completes it 0.97-0.98 at
368-375 steps.  Go/no-go: +0.58 at t <= 5 >= +0.5 -- **go**.  The
observational counterpart on the same rows is -0.48 (the sighted teacher's
shortcut), the sign the confounding gives.  The go branches reproduce the
blind prior (reach 0.33, death 0.67).

## Phase 1, first pass (replay v1, driver-torque queries): the gate and what it found

Five critics (V6 recipe, gamma 0.999, 100k updates, row-0 anchors) on the
v1 replay (10,913 branch paths + 3,000 north / 3,000 east query paths whose
first torque is the DRIVER's).  Law target on the replay's query paths at
the start region: +0.49.

| probe (start-region states, t <= 5) | seed 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| paired same-state: recorded north torques - recorded east torques | -0.61 | -0.57 | -0.35 | -0.23 | -0.82 |
| paired same-state: query north torques - query go torques | +0.58 | +0.77 | +1.55 | +0.95 | -0.14 |
| replay rows themselves: f(north query row) - f(go query row) | +6.1 | +5.6 | +6.7 | +6.2 | -- |
| replay go rows: f(branch reached) - f(branch died) | +10.5 | +10.9 | +12.0 | +11.7 | -- |

Reading.  (i) The critics learned the route preference the replay carries
-- but on the replay's own torques: the query north torque (the driver's
first output at the anchor, with the teacher's turn bias and a freshly
reset controller) sits 1.9 torque-units from the recorded north torques,
farther than recorded north is from recorded east (0.87), while the
query go torque is 0.32 from the recorded east torques.  The critics
score recorded north torques 1.1-2.4 BELOW the query north torques at the
same states, so on the torques the BC term imitates and the actor can
produce, the critic prefers east.  (ii) f on the replay rows tracks each
path's own outcome by ~11 nats (reached vs died): the critic keys on
row identity rather than on a torque -> route rule -- the same
"narrow-peak" structure PointMaze showed, now in 8 torque dimensions where
a single torque carries only weak route information (linear AUC north vs
east on recorded rows at t <= 5: 0.80).

Consequence: the query branch's first action must come from the recorded
torque distribution.  Replay v2 (`--query-donor recorded`, the default
now): at each start-region query anchor the first K = 25 torques are a
recorded segment of a donor episode chosen by the VISIBLE record (its
displacement over those K steps points north / east: 49 north donors and
946 east donors per t), then the driver continues on the queried route.
This is PointMaze's rule (the query action is one the data contains),
not a new label.  The actors of the v1 critics are being trained anyway
(warm-started joint, all five seeds) to see whether the torque-signature
mismatch matters for the policy; v2 critics follow.

### v1 joint actor, seed 0 (30-episode look, before the v2 replay)

Warm-started joint training (100k, balanced BC by displacement, bc 0.05)
on the v1 critic seed 0: at 200 recorded start states the actor's mean
torque is 1.41 from the recorded-north centroid and 1.32 from the
recorded-east centroid (the two centroids are 0.91 apart) -- it produces
torques unlike either recorded behaviour -- and on 30 natural draws it
times out 0.87-0.93, succeeds 0.07-0.13, takes the detour 0.03-0.07 and the
shortcut 0.10-0.13; no deaths because it rarely reaches a zone.  The
critic's start-state peak (the query torque signature, section above)
lies off the recorded manifold and the actor chases it; vanilla at 0.99
walks east and succeeds 0.37.  The remaining v1 joint seeds were stopped
as uninformative; the vanilla control at gamma 0.999 runs first (does the
discount alone keep locomotion?), then the v2 replay.

### Vanilla control at gamma 0.999 (recorded data, 100k updates, 3 seeds, 300 natural draws)

| seed | policy | success | failure | timeout | detour | shortcut |
|---|---|---:|---:|---:|---:|---:|
| 0 | mean | 0.363 | 0.630 | 0.007 | 0.000 | 0.980 |
| 0 | sample | 0.373 | 0.627 | 0.000 | 0.000 | 0.970 |
| 1 | mean | 0.370 | 0.623 | 0.007 | 0.000 | 0.980 |
| 1 | sample | 0.373 | 0.623 | 0.003 | 0.003 | 0.960 |
| 2 | mean | 0.370 | 0.627 | 0.003 | 0.000 | 0.967 |

Identical to vanilla at 0.99 (`notes/v6_detour_ladder.md`: 0.370 / 0.000 on
every seed): the discount alone changes neither locomotion nor the route.
The v1 joint actor's stalling therefore came from the branch critic's
off-manifold peak, not from gamma.

### Replay v2 (donor-torque queries): the first critic, and the target's ceiling

Replay v2 on the node: 10,913 branch paths + 3,000 north / 3,000 east
queries whose first 25 torques are a recorded donor segment; north queries
reach 0.72 (the driver's own turn reached 0.91; the transplanted segment
turns less reliably), east 0.36 / death 0.63.  v2 critic seed 0, paired
same-state margin (recorded north vs recorded east torques): -0.53 (s.e.
0.11); with the query torques themselves: -1.67.  Six critics over v1 and
v2 now, all negative on the recorded torques.

The ceiling the law allows on this benchmark, from the measured reach times
(north 365-430, east 222, both parked to the 800-step horizon), as a
function of how well the queried turn executes and of the shortcut's
blind survival (0.36 at p_active 0.40):

| gamma | north reach 0.72 @430 | 0.91 @400 | 0.98 @365 |   east survival 0.36 / 0.25 / 0.16 |
|---|---|---|---|---|
| 0.999 | +0.13 / +0.50 / +0.95 | +0.46 / +0.83 / +1.27 | +0.64 / +1.00 / +1.45 | (p_active 0.40 / 0.50 / 0.60) |
| 1.0 | +0.25 / +0.61 / +1.06 | +0.56 / +0.92 / +1.37 | +0.72 / +1.08 / +1.53 | |

With the benchmark as it is (survival 0.36) the target is +0.13 with the
v2 turn and at most +0.64 even with a perfect turn at gamma -> 1; PointMaze's
P replay asked for +0.88 at the policy width and its critics returned
+0.45..+0.66.  On the Ant the critics' same-state action margin has a
seed spread of ~0.5 and sits below zero on every critic so far, so a
target of a few tenths of a nat is inside the noise -- the same situation
as PointMaze's Step 10 (E critics at 0.00 +- 0.25 against +0.39).  The
levers that would lift the target are the shortcut's survival (hazard
density) or the detour's length, both benchmark properties that this
study keeps fixed; on the learner side only the query turn's execution
(0.72 -> 0.98 is worth +0.5) remains.  The v2 joint actors are still
trained for the record.

## Option B (user decision, 2026-09-17): raise the hazard density to 0.50 / 0.60

The user chose to lift the target by the density lever rather than stop
or rework the query turn.  Nothing else moves: same maze, clocks, teacher
(5% forced detour), frozen walker, horizon, collection seed 606, 1000
episodes.  Both rungs recollected with
`scripts/collect_rockfall_clock_v6_dataset.py --p-active-1 p --p-active-2 p`
(`antmaze_rockfall_clock_v6_p050` / `_p060`); every composition gate passes
on both (no discards; u1/u2 observed 0.49/0.51 and inside the Wilson-99
band at 0.60; teacher success 1.0).  The generator and the driver read the
rung from `V6_DATASET_STEM` / `V6_P_ACTIVE` (commit 9220012); the frozen
p040 files stay the default.

Phase 0 at the two rungs (400 start anchors, driver turn, zero-torque hold
after reaching, gamma 0.999, r 0.5):

| rung | go reach (t<=5) | detour reach (t<=5) | target t<=5 | t=0 | t<=10 | 11-60 | all t<=60 | observational |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p040 (benchmark) | 0.38 | 0.91 | +0.58 | +0.60 | -- | -- | +0.14 | -0.48 |
| p050 | 0.23 | 0.91 | **+0.85** | +1.00 | +0.85 | -0.72 | +0.56 | -0.45 |
| p060 | 0.15 | 0.92 | **+1.30** | +1.35 | +1.26 | -0.43 | +0.95 | -0.43 |

The measured targets land on the ceiling table's driver-turn column
(+0.83 / +1.27 predicted from reach times alone).  With the v2 donor turn
(reach 0.72) the replay's own law target should come out near +0.50
(p050) and +0.95 (p060); the gate stage prints the realised number.

Run order on the node (one learner slot; the p040 v2 chain owns the
other): per rung, v2 replay (`--every 25 --query-every 2 --t-max 5
--query-donor recorded`) -> critics seeds 0-2 -> paired gate.  The joint
actors and the vanilla control at the same density are trained only for
a rung whose critics pass the gate; the gate is the cheap decisive stage.

### Gate results: p040 v2 (five critics) and the p050 rung (three critics)

p040, replay v2, all five critics (`outputs/antmaze_branch_replay_v2/gate.md`,
law target on the replay's query paths +0.23):

| critic | row margin | paired same-state margin | s.e. | states > 0 |
|---|---:|---:|---:|---:|
| seed 0 | +1.73 | **-0.53** | 0.11 | 0.36 |
| seed 1 | +0.72 | **-0.65** | 0.11 | 0.37 |
| seed 2 | +0.68 | **-0.59** | 0.13 | 0.41 |
| seed 3 | +1.88 | **-0.05** | 0.12 | 0.48 |
| seed 4 | +0.78 | **-0.29** | 0.10 | 0.42 |

0/5 pass (mean -0.42).  The p040 v2 joint chain was stopped after its
first actor (seed 0, kept for the record) to free the learner slot.

p050, replay v2 (17,165 paths, 13 min; north queries reach 0.72, east
0.25 / death 0.75), law target on the query paths **+0.60** (r 0.5):

| critic | row margin | paired same-state margin | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---|
| seed 0 | +2.77 | **+0.17** | 0.10 | 0.54 | PASS |
| seed 1 | +2.88 | **+0.28** | 0.10 | 0.57 | PASS |
| seed 2 | +1.80 | **-0.26** | 0.10 | 0.46 | fail |

2/3 pass.  Read against p040: the target moved by +0.37 (0.23 -> 0.60)
and the critics' paired margin moved by +0.48 (mean -0.42 -> +0.06), so
the critic does track the law's number, with an offset of about -0.5 nat
and a seed spread of about 0.3 -- the same shrinkage PointMaze's critics
showed (+0.45..+0.66 against +0.88).  Decision: the p050 rung goes on to
the second half (critics 3-4, five-seed gate, warm-started joint actors
on every critic, vanilla control at p 0.50 / gamma 0.999); the p060 rung
(target ~+0.95 expected) runs its gate in the other slot.

p050 completed to five critics (`gate.md`, target +0.60): seeds 3 and 4
add **-0.06** and **-0.16**, so 2/5 pass and the five-seed mean is
-0.01 (s.e. of each ~0.10).  The p040-to-p050 shift of the critics'
mean is therefore +0.41 for a +0.37 shift of the target: the slope is
one, the intercept about -0.6 nat.  The warm-started joint actors are
trained on all five critics for the record (the actor sees the full f
landscape, not only the north-vs-east sign at the recorded torques).

p060, replay v2 (17,367 paths; north queries reach 0.72, east 0.16 /
death 0.83), law target on the query paths **+1.02** (r 0.5):

| critic | row margin | paired same-state margin | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---|
| seed 0 | +2.14 | **-0.66** | 0.09 | 0.37 | fail |
| seed 1 | +1.61 | **-0.09** | 0.10 | 0.44 | fail |
| seed 2 | +2.11 | **+0.13** | 0.11 | 0.52 | fail |

0/3 pass, mean -0.20.  This breaks the slope-one reading of p040 -> p050:
across the three rungs the paired margin on transplanted recorded
torques is -0.42 / -0.01 / -0.20 while the target is +0.23 / +0.60 /
+1.02.  The row margin (each row scored with its own goal) is positive
on every critic at every rung (+0.7..+2.9) -- the critics do learn that
north-moving rows lead to the goal -- but at a fixed state the recorded
north and east torques are not ranked by the law's number.  Two readings
remain open: (a) the critic's action sensitivity at the start states is
weak and seed-noisy (the torque is a 1.9-unit-wide cloud within each
direction group against 0.87 between the groups' centroids -- a
transplanted torque may not carry its direction to another joint
configuration, so the gate itself is a blunt instrument on the Ant), or
(b) the critic ranks by state only.  The joint actors settle it: they
read the full f landscape at the visited states.  p050's actors are
training; p060's second half (critics 3-4, actors, vanilla control) was
launched in the freed slot for the same reason -- the bigger target is
the more informative test of (a) versus (b).

### p050 joint actor seed 0: a slow shortcut, not a detour (100-episode look, mean policy, density 0.50)

success 0.53 / failure 0.04 / timeout 0.43; route detour 0.01, shortcut
0.72; by latent U00 0.69, U10 0.47, U01 0.48, U11 0.48.  Mean start-state
torque 1.36 from the recorded-north centroid, 1.14 from the east one.

Where the success comes from (`quick_eval_joint_seed0.log`): the actor
reaches mouth 1 at step 166 (median; q10-q90 108-262) and mouth 2 at 251
(195-383), against burst windows that close by step 117 (zone 1) and 197
(zone 2).  Of the shortcut entries under an active latent, 33/35 (zone 1)
and 25/28 (zone 2) arrive AFTER the burst and mostly succeed (23, 19);
only 5 arrive during a burst (2 die).  The 43 timeouts: 25 never reach a
mouth, 17 stall inside the shortcut corridor.

So at gamma 0.999 with the V6 own-clock bursts, a walker three times
slower than the teacher passes both zones after they have closed:
undiscounted success rewards slowness, and the critic's route preference
(if any) never appears in the behaviour.  Vanilla at gamma 0.999 on p040
was not slow (0.37 / detour 0, identical to 0.99), so the slowness comes
with the branch-replay warm start.  The comparison that matters for the
rung is therefore detour rate and the discounted score (gamma 0.99
`discounted` field), not raw success; the chain's 300-episode evaluations
of all five actors and of the vanilla control at p 0.50 follow.

Seed 1 (its critic had the best gate margin, +0.28), same 100-episode
look: success 0.47 / failure 0.01 / timeout 0.52; detour 0.01, shortcut
0.56; mouth 1 at step 140 (median), 25/28 active-latent entries after
the burst; 43 of the 52 timeouts never reach a mouth.  Start-state torque
0.97 from the east centroid, 1.23 from the north one.  Two of five
actors, one picture: the branch-replay warm start makes a slow, stalling
shortcut walker; the critic's gate margin (+0.17 / +0.28 on these two)
does not appear as a north turn in behaviour.

p060 completed to five critics: seeds 3 and 4 add **-0.02** and **-0.14**
(0/5 pass, mean -0.16 against +1.02).  Its joint stage was SIGKILLed on
the node (rc -9: two replay-loaded learners at once on 15 GB) and is
queued to re-run alone after both chains finish; the p060 vanilla
control runs meanwhile.

### The critic memorises rows: the p050 critics on the replay's own query torques

Two probes on the five p050 critics (`probe_v6_torques.py`,
`probe_v6_replay_rows.py`; 200 start states, 150-200 torques per group).

Same 200 recorded start states, four torque groups transplanted onto
each (mean f, min over twin heads):

| critic | rec north - rec east | replay north-donor - replay east-donor | rec north - replay north-donor |
|---|---:|---:|---:|
| seed 0 | -0.07 | **-0.78** | +0.70 |
| seed 1 | -0.00 | **-0.71** | +0.70 |
| seed 2 | -0.35 | **-1.15** | +0.85 |
| seed 3 | -0.15 | **-1.30** | +1.03 |
| seed 4 | +0.06 | **-0.42** | +0.38 |

On the replay's own query rows (identical anchor state, north-donor
torque vs east-donor torque, same goal): row margin **+2.5 .. +3.5** on
every critic -- but the same north-donor torques transplanted onto other
query states of the same kind: **-0.82 / -0.11 / -0.77 / -0.74 / -0.47**.
And among the east ("go") query rows, those whose branch reached the goal
score -10 against -23 for those that died: 13 nats for an outcome that
is decided by the hazard draw made after the row, i.e. not a function of
(s, a) at all.

Reading: the critic fits each replay row's realised future as an
identity of that row's (state, torque) pair.  Every Ant row is a unique
key (29-dim state, 8-dim torque; torques from different donors differ by
~0.85 per dimension), so nothing in the NCE objective forces the
expectation over hazard draws that the law target is: the +2.5..+3.5 on
the training rows is memorised, and away from the exact training torque
the critic falls back on torque typicality (east-going torques are 12x
more common in the recorded data), which is why every transplanted test
-- the gate's recorded torques, the replay's own donor torques -- comes
out at or below zero regardless of the target.  PointMaze never met this:
its (cell, action-angle) keys repeat across many draws, so the critic
averaged them; it also used 30k-update critics, and its 300k critic was
the unusable one.

This is the reason option B could not work: raising the density changes
the expectation, and the critic is not learning an expectation.  The two
levers that address it directly are (i) early stopping -- probe the
paired margin along the critic's own training trajectory (milestones
every 10k) and (ii) repeated futures per query key -- branch each query
anchor several times with the same donor torques and different hazard
draws, so the identical (s, a, g) row appears with different futures and
the row-identity fit stops paying.  (i) is cheap and runs next in the
freed slot; (ii) is a generator flag.

First chain evaluations (300 natural draws, mean policy): p050 joint seed
0 -- success 0.603 / failure 0.027 / timeout 0.370, detour 0.003,
shortcut 0.790, mouth 1 at step 156 (median; 0.93 of active-latent
entries after the burst), mouth 2 at 240 (0.89), discounted 0.019.  p060
vanilla seed 0 -- success 0.150 / failure 0.847, detour 0, mouth 1 at
step 52 (0.006 after the burst), discounted 0.015: the blind learner at
its natural speed dies at the density's survival rate.  Raw success
therefore favours the slow walker 4:1, the benchmark's discounted score
puts both near zero, and neither takes the detour.

### Early stopping: the critic's own trajectory (p050 seed 0 retrained with 10k milestones)

`probe_v6_critic_milestones.py` on `critics_ms/seed_0` (150 start
states; four torque groups transplanted; the replay's own rows):

| updates | rec north - east | replay north-donor - east-donor (transplanted) | own rows north - go | go rows reached - died | north-donor own - transplanted |
|---:|---:|---:|---:|---:|---:|
| 10k | -6.04 | -23.8 | -21.7 | -1.3 | +3.2 |
| 20k | +0.09 | **+0.99** | +1.30 | +4.8 | +0.33 |
| 30k | +0.13 | **+1.29** | +1.60 | +6.2 | +0.60 |
| 40k | +0.25 | **+1.02** | +1.81 | +8.9 | +1.51 |
| 50k | +0.23 | +0.72 | +2.05 | +9.8 | +2.33 |
| 60k | +0.12 | +0.34 | +2.12 | +10.3 | +3.30 |
| 70k | +0.15 | -0.04 | +2.16 | +10.9 | +4.06 |
| 80k | +0.21 | +0.09 | +2.75 | +12.0 | +4.99 |
| 90k | +0.03 | -0.18 | +2.90 | +13.3 | +5.36 |
| 100k | +0.05 | +0.02 | +3.32 | +14.8 | +6.71 |

The action preference the law asks for is there at 20k-40k updates
(+1.0..+1.3 on the replay's own donor torques transplanted across
states, against a query-path target of +0.60) and is then eaten by the
row-identity fit: reached-vs-died grows monotonically to +14.8 and the
own-vs-transplanted gap to +6.7 while the transplanted margin decays to
zero.  Same shape as PointMaze (30k critics usable, 300k not).  The
recorded-torque margin peaks at +0.25 (40k): the gate's recorded torques
are a different cloud from the donor torques the replay was built from,
so that gate stays a blunt instrument here.

Consequences.  (1) Every critic evaluated so far on the Ant was a 100k
critic, i.e. read after the memorisation; the five gate tables above
measure the memorised state, not the recipe.  (2) The warm-started joint
stage keeps updating the critic on the same replay for another 100k
updates, so it drives even a well-stopped critic back into the
memorised state -- the joint actors' slow shortcut is what a memorised
critic teaches.  Next: five 30k critics (`critics_30k`), the two probes
on each, then actors that do not move the critic (frozen critic, actor
+ BC only) and a short joint variant for comparison.  The queued p060
joint retry (100k critics) was dropped.

### Five 30k critics (p050): the interventional action preference, 5/5

`critics_30k/seed_{0..4}` (30,000 updates, same recipe otherwise), the
two probes (`probe_torques_30k.log`, `probe_replay_rows_30k.log`):

| critic (30k) | rec north - east | replay north-donor - east-donor (transplanted, 150 states) | replay north-query torques across north-query states (paired, 200) | s.e. | states > 0 | own rows north - go | go rows reached - died |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | +0.15 | **+0.68** | **+1.06** | 0.12 | 0.77 | +1.82 | +6.9 |
| seed 1 | +0.03 | **+1.01** | **+1.51** | 0.11 | 0.84 | +2.11 | +7.6 |
| seed 2 | +0.22 | **+1.11** | **+1.38** | 0.12 | 0.81 | +2.02 | +7.5 |
| seed 3 | +0.10 | **+0.84** | **+0.98** | 0.14 | 0.70 | +1.32 | +7.2 |
| seed 4 | +0.03 | **+0.83** | **+1.03** | 0.14 | 0.77 | +1.65 | +6.9 |

Against the query-path target +0.60, every 30k critic ranks the north
donor torques above the east donor torques at the same state, on
torques it did not see at that state (+0.7..+1.5), on all five seeds
(the 100k critics: -0.4..-1.3 and -0.1..-0.8 on the same two probes).
The row-identity fit is already present at 30k (reached-vs-died ~+7)
but has not yet overwritten the action preference.  The recorded-torque
gate stays near zero (+0.03..+0.22): the recorded start-region torques
are a different cloud from the donor torques (centroid distance 1.23
against 0.87 between north and east), so that gate cannot read this
critic; the transplanted-donor probes are the gate for the Ant.

Next, queued on the node in this order: the tagged gate on the 30k
critics, frozen-critic actors (critic lr 0, actor + BC, 100k) on all
five, a 30k joint variant on all five, then the p050 vanilla control.

For the record, the 100k-critic joint actors at p050 (300 natural draws;
`joint/seed_*/eval_*.json`):

| seed | policy | success | failure | timeout | detour | shortcut | discounted | mouth 1 median | after-burst z1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mean | 0.603 | 0.027 | 0.370 | 0.003 | 0.790 | 0.019 | 156 | 0.93 |
| 0 | sample | 0.337 | 0.000 | 0.663 | 0.250 | 0.477 | 0.003 | 285 | 1.00 |
| 1 | mean | 0.453 | 0.033 | 0.513 | 0.017 | 0.543 | 0.015 | 143 | 0.84 |
| 1 | sample | 0.450 | 0.023 | 0.527 | 0.050 | 0.563 | 0.013 | 158 | 0.93 |
| 2 | mean | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | -- | -- |
| 2 | sample | 0.000 | 0.000 | 1.000 | 0.077 | 0.040 | 0.000 | 410 | 1.00 |
| 3 | mean | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | -- | -- |
| 3 | sample | 0.000 | 0.000 | 1.000 | 0.243 | 0.140 | 0.000 | 355 | 1.00 |

Seeds 2 and 3 never leave the start under the mean policy (timeout
1.000, no route); the sampled policies wander north (detour 0.24 on seed
3) but never arrive.  A memorised critic teaches either a slow shortcut
or no locomotion at all.

### The actor climbs the 30k critic off the data manifold

Recorded-torque gate on the 30k critics (`gate_30k.md`): +0.11 / +0.26
/ +0.20 / +0.11 / +0.08 (s.e. 0.06) -- positive on all five, small
because those torques are a different cloud from the donor torques.

Frozen-critic actor on critic seed 0 (critic lr 0, actor loss 0.95 (-f)
+ 0.05 BC on balanced rows, 100k): **no locomotion** -- 100-episode
look: timeout 1.000, 89 of 100 never reach a mouth; the 10k milestone
is the same (60/60 timeouts, no mouth).  `probe_v6_actor_f.py` at 200
recorded start states, scored by the critic the actor was trained
against:

| actor ckpt | f(pi mode) | f(recorded torque) | f(north donor) | f(east donor) | f(pi) - f(rec) | policy scale | frac |a| > 0.95 | pi-to-recorded torque distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen30k seed 0 @10k | -8.58 | -14.10 | -13.12 | -13.50 | **+5.5** | 0.64 | 0.43 | 2.60 |
| frozen30k seed 0 @40k | -8.32 | -14.10 | -13.12 | -13.50 | **+5.8** | 0.66 | 0.47 | 2.60 |
| frozen30k seed 0 final | -8.28 | -14.10 | -13.12 | -13.50 | **+5.8** | 1.25 | 0.51 | 2.69 |
| 100k joint seed 0 final | -33.43 | -36.56 | -38.03 | -37.31 | +3.1 | 3.97 | 0.52 | 2.43 |

Within 10k actor updates the mode sits 2.6 torque units from any
recorded torque (the recorded clouds are 0.87 apart), half its
components saturated, and the critic pays it 5.8 nats more than any
real torque: the NCE critic has no negatives off the data manifold, so
its f is unconstrained there and an 8-dimensional gradient ascent finds
the peak at once; the 0.05 BC term is no match for +5.8 nats.  PointMaze
never met this either (2-d action, dense action coverage).  The critic
ranks real torques correctly (previous section) and is useless to an
unconstrained actor; the remaining seeds of the frozen chain and the
30k joint variant (same actor loss) were stopped.

Two remedies inside the recipe, to run next on critic seed 0: (i) the
BC weight -- the paper's own lambda -- raised to 0.5 and 1.0 (pure BC,
which also serves as the proposal for (ii)); (ii) a proposal-and-rank
policy: K samples from the BC policy, the critic picks the argmax -- the
continuous counterpart of the discrete repo's categorical argmax, on
the data manifold by construction.

### BC weight 0.5 on the frozen 30k critic (seed 0): walks, shortcut, no detour

60-episode look (mean policy): success 0.35 / failure 0.45 / timeout
0.20; detour 0.000, shortcut 0.833; mouth 1 at step 68 (median; the
teacher's pace), 21 of 25 active-latent entries during the burst (18
die) -- an ordinary blind shortcut walker.  Manifold probe: f(pi) -9.6,
still +4.5 above the recorded torque (distance 1.92, 33% saturated) --
half the climb of the 0.05 actor.  Start-state mean torque 1.02 from the
recorded-north centroid against 1.34 from the east one, and the ant
still goes east: torque-centroid distance is not direction.  So with
lambda 0.5 the BC term keeps the actor walking and the critic term
still spends its gradient off the manifold rather than on the
north-vs-east choice the critic does rank correctly among real torques.

Its milestones (10k / 30k / 60k, 60 episodes each): shortcut 0.40 /
0.27 / 0.65, detour 0.000 throughout, timeouts 0.58 / 0.75 / 0.43 -- the
actor never passes through a north-going phase; the critic term costs
locomotion early and the BC term wins it back.

### Proposal-and-rank: the first detours on the Ant

`eval_v6_rank_policy.py`: at every step K tanh-normal samples from a BC
actor plus its mode, scored by a 30k critic f(s, a, g) (min over the twin
heads), argmax.  With the lambda-0.5 actor as the proposal and critic seed
0, K = 32, 100 natural draws at density 0.50:

    success 0.41 / failure 0.34 / timeout 0.25; detour 0.12, shortcut
    0.77; by latent U00 0.81, U10 0.30, U01 0.13, U11 0.38; the chosen
    candidate scores +2.3 nats above the BC mode on average.

Twelve per cent of episodes take the detour -- against 0.000-0.017 for
every gradient-trained actor on this rung and 0.000 for the vanilla
controls -- with nothing trained: the same critic that no gradient actor
could use, applied only to actions the data policy proposes.  Still
mostly shortcut: the proposal is a shortcut walker (its samples rarely
point north) and the decision is re-made at every step from a fresh
sample.  The queued runs sharpen both: the pure-BC proposal (lambda 1.0,
which samples the recorded 5% detour torques) with all five 30k critics
at K = 32, and the K dependence (8, 128).

K = 128 with the same proposal and critic (100 draws): success 0.44 /
failure 0.26 / timeout 0.30; detour 0.08, shortcut 0.76; the chosen
candidate now +2.9 nats above the mode.  More candidates do not raise
the detour rate; they raise the chosen f -- i.e. the argmax increasingly
picks the proposal's far samples (scale 0.62 per component), which are
the off-manifold directions the critic overvalues.  The candidate set
has to stay on the data manifold: next a proposal temperature (samples
at 0.3 of the actor's scale) and, if needed, recorded torques of the
nearest recorded states as the candidates.  The lambda-0.5 actor's
300-draw evaluation: mean 0.30 / 0.47 / 0.23, detour 0.003, shortcut
0.75; sampled 0.42 / 0.47 / 0.12, detour 0.037, shortcut 0.88.

Proposal temperature 0.3 (K = 32, same proposal and critic, 100 draws):
success 0.33 / failure 0.46 / timeout 0.21; detour 0.03, shortcut 0.83;
chosen candidate +0.9 nats above the mode.  Closer samples reproduce the
proposal's own east-going mode; the 0.12 at temperature 1 came partly
from the far samples.  With a shortcut-walking proposal the candidate set
cannot be both on-manifold and north-inclusive; the pure-BC proposal (it
samples the recorded 5% detour torques) and recorded torques of the
nearest recorded states as candidates are the two ways to get north
candidates that are real torques.

### Continuation pessimism and the waiting loophole (for the write-up)

The branch replay does not deny the shortcut a future: its east branches
reach at the blind survival rate (0.36 / 0.25 / 0.16 at p 0.40 / 0.50 /
0.60; exactly (1 - p)^2), the target is a log ratio of +0.6, and the same
recipe at gamma 0.99 says go (-0.78 at p 0.40).  The recorded futures are
the optimistic ones for a confounded reason: the sighted teacher went
only when safe (observational target -0.45).  What the recipe does assume
is the continuation after the queried action -- the blind driver, "go
now" -- so do(a) is valued as "a, then the nominal blind continuation":
pessimistic against a better blind continuation, optimistic against a
worse one, as in every ETT.  On V6 a better blind continuation exists:
the burst clocks run from the reset with bounded t0, so arriving after
step 117 / 197 is safe, and the 100k joint actors found it (a slow
shortcut, success 0.60, discounted 0.02).  The replay contains no
waiting, so the critic cannot value it; this is a real pessimism about
the shortcut, of the continuation kind, not of the hazard-redraw kind.
(The replay is if anything pessimistic about the detour: north branches
reach 0.72 because the transplanted turn fails a quarter of the time.)
Remedies: state the continuation assumption and its sensitivity (the
ceiling table already does this for the turn; a waiting continuation can
be tabulated the same way), or iterate -- regenerate the replay with the
learner's own policy as the continuation.  Benchmark-side, a burst
triggered by entry instead of the reset clock would close the loophole;
the environment stays fixed in this study.

Recorded torques of the K = 32 nearest recorded states (60k-row bank,
standardised 29-dim state distance) as the candidates, same critic, 100
draws: success 0.06 / failure 0.12 / timeout 0.82; detour 0.03, shortcut
0.55.  Torques taken from different episodes at every step do not form a
gait; the ant stalls.  The candidate set therefore has to come from one
coherent policy, and the rank-policy summary on this critic reads: K =
32 at temperature 1 detour 0.12 (chosen +2.3 nats), K = 128 0.08 (+2.9),
temperature 0.3 0.03 (+0.9), nearest-recorded torques 0.03 (stalls).  A
shortcut-walking proposal gives the critic few real north candidates and
the argmax spends its choice on far samples.  The pure-BC proposal is the
remaining fair candidate source and its five-critic evaluation is queued.

### Pure BC (lambda 1.0) on the same data: the proposal is unimodal

300 draws: mean policy success 0.253 / failure 0.743 / timeout 0.003,
detour 0.000, shortcut 0.977; sampled 0.260 / 0.740 / 0.000, detour
0.000.  Manifold probe: f(pi) = f(recorded) (+0.03), torque distance
0.29, policy scale 0.26 -- on the manifold, at the teacher's pace, dying
at the density's rate.  The tanh-normal actor fitted to 95/5 route data
does not keep a 5% north mode at the start: the sampled policy detours
0/300.  So the pure-BC proposal offers the rank policy almost no north
candidates either; its five-critic evaluation runs for the record.

### Segment-rank: one critic-ranked route decision over recorded macro-actions -- detour 0.40

`eval_v6_segment_rank_policy.py`: while the ant is in the start region and
no segment is running, K recorded 25-step torque segments (the first
torques of K random recorded episodes at the current time step, both
routes in their recorded 95/5 proportion) are scored by the critic at the
current state with each segment's first torque; the argmax segment runs
open-loop for 25 steps; everywhere else the pure-BC actor's mode acts.
Candidates are recorded torques, the ranking is the learned critic, the
walker is the learned BC actor; the only design choice is the 25-step
commitment (the replay's own K).

Critic seed 0 (30k), K = 64, 100 natural draws at density 0.50:

    success 0.39 / failure 0.31 / timeout 0.30; **detour 0.40**, shortcut
    0.47; by latent U00 0.77, U10 0.37, U01 0.13, U11 0.24; 251
    decisions, 56 north picks (22% against ~5% north candidates).

Against 0.000-0.017 for every gradient-trained actor on this rung, 0.000
for the vanilla controls and 0.03-0.12 for the per-step rank policy.
The critic's same-state preference for north torques (+0.7..+1.5 on the
donor probes) turns into behaviour once the decision is (i) made among
real recorded torques and (ii) committed for the length the critic was
trained on, instead of being re-sampled every step from a unimodal
proposal.  Seeds 1-4 and the K dependence (32, 128) follow.

By route (same 100 draws): the 40 detour episodes -- success 0.57,
failure 0.00, timeout 0.42 (mean 605 steps: the BC walker, fitted to 5%
detour data, stalls on the long corridor); the 47 shortcut episodes --
success 0.34, failure 0.66 (the density's rate); 13 never leave the
start.  Discounted 0.017 overall.  The decision is now the critic's; what
the detour loses is walker competence on the corridor it has 5% of the
data for, not deaths.

Segment-rank on all five 30k critics (K = 64, 100 draws each) and the K
dependence on critic 0:

| critic (30k) | success | failure | timeout | **detour** | shortcut | decisions | north picks |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 0.39 | 0.31 | 0.30 | **0.40** | 0.47 | 251 | 56 |
| seed 1 | 0.45 | 0.24 | 0.31 | **0.52** | 0.33 | 291 | 69 |
| seed 2 | 0.42 | 0.32 | 0.26 | **0.36** | 0.47 | 365 | 53 |
| seed 3 | 0.43 | 0.24 | 0.33 | **0.43** | 0.38 | 386 | 70 |
| seed 4 | 0.34 | 0.38 | 0.28 | **0.26** | 0.52 | 273 | 42 |
| seed 0, K = 32 | 0.38 | 0.23 | 0.39 | 0.35 | 0.37 | 288 | 58 |
| seed 0, K = 128 | 0.36 | 0.29 | 0.35 | 0.34 | 0.45 | 357 | 57 |

5/5 critics detour 0.26-0.52 (mean 0.39) against ~0.05 north candidates,
0.000-0.017 for every gradient actor and 0.000 for vanilla; K 32 / 64 /
128 flat.  The formal version (300 draws per critic), the no-critic
control (same candidates, uniform pick -- the 5% prior) and the same
policy ranked by the vanilla critics (trained on the recorded futures)
are running.

Per-step rank with the pure-BC proposal, for the record (K = 32, 300
draws per critic): detour 0.023 / 0.037 / 0.027 / 0.030 / 0.013 on
critics 0-4 (success 0.25-0.27, failure 0.69-0.73); K = 8 on critic 0
0.017.  A unimodal proposal with scale 0.26 offers no north candidates;
the per-step rank is closed as a variant.

Formal segment-rank, 300 natural draws per critic (K = 64, 25-step
commitment, pure-BC walker, mode everywhere else; density 0.50):

| critic (30k) | success | failure | timeout | **detour** | shortcut | discounted (g 0.99) | detour episodes: success / timeout | shortcut episodes: failure |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| seed 0 | 0.407 | 0.280 | 0.313 | **0.443** | 0.423 | 0.016 | 0.60 / 0.40 | 0.65 |
| seed 1 | 0.390 | 0.290 | 0.320 | **0.457** | 0.383 | 0.013 | 0.63 / 0.37 | 0.69 |
| seed 2 | 0.373 | 0.320 | 0.307 | **0.360** | 0.457 | 0.016 | 0.64 / 0.36 | 0.68 |
| seed 3 | 0.467 | 0.237 | 0.297 | **0.473** | 0.383 | 0.018 | 0.69 / 0.31 | 0.61 |
| seed 4 | 0.377 | 0.313 | 0.310 | **0.330** | 0.447 | 0.017 | 0.68 / 0.32 | 0.65 |

Detour 0.33-0.47 (mean 0.41) on 5/5; detour episodes never die and
reach 0.60-0.69 (the rest are walker timeouts on the long corridor);
shortcut episodes die at 0.61-0.69, the density's rate.  The controls
(same candidates picked uniformly; the same policy ranked by the vanilla
critics) and the vanilla actors at p 0.50 follow.

Control 1, the same candidates picked uniformly (no critic; 300 draws):
success 0.240 / failure 0.557 / timeout 0.203; **detour 0.050**,
shortcut 0.767; 22 north picks in 684 decisions (3.2%, the candidates'
own proportion).  The 0.33-0.47 above is the critic's ranking, not the
candidate set or the commitment rule.  Control 2 (the same policy ranked
by the vanilla critics trained on the recorded futures) waits for the
vanilla learners.

### Control 2 is not clean: the vanilla critic also ranks north segments up

The same segment-rank policy ranked by the p050 vanilla critic seed 0
(100k updates on the recorded futures, gamma 0.999; 300 draws): success
0.397 / failure 0.270 / timeout 0.333; **detour 0.383**, shortcut 0.437;
172 north picks in 1025 decisions -- indistinguishable from the branch
critics (0.33-0.47).  The vanilla actor itself is the blind shortcut
(seed 0, mean: success 0.260 / failure 0.730, detour 0.003).

Four-group same-state probe on the three p050 vanilla critics (200 start
states; replay north-donor minus east-donor torques / recorded north
minus east): +0.22 / +0.23 / +0.38 and +0.07 / +0.06 / +0.03 (the branch
30k critic 0: +0.68 and +0.15).  So the vanilla critics carry a north
preference of a few tenths of a nat at the start state on the same
torques, against their own data's observational target of -0.45, and
the argmax over 64 candidates turns a preference of that size into ~40%
north picks just as it does for the branch critics' +0.7..+1.1.  The
segment-rank number therefore does not separate the two critic families
at this rung; the random-pick control only shows that some critic
preference is being decoded.  What does separate them is the margin
itself (2-3x), and the recorded-torque gate (+0.11..+0.26 against
+0.03..+0.07).

Open: why a critic trained on recorded futures prefers north-donor
torques at all (100k vanilla critics may memorise as the branch ones do;
the rare north rows come from detour episodes that always reach).
Running: vanilla critics at the matched 30k budget (three seeds), their
probes and their segment-rank; the same probe on the p040 and p060
vanilla critics.

All three p050 vanilla critics as the ranker (300 draws each): detour
**0.383 / 0.473 / 0.463** (success 0.40 / 0.41 / 0.43) -- the same
band as the branch critics.  The same four-group probe on the p040 and
p060 vanilla critics: north-donor minus east-donor +0.11 / +0.19 / +0.42
(p040) and +0.07 / +0.18 / +0.28 (p060); recorded north minus east
+0.01..+0.07 everywhere.  Every vanilla critic at every rung carries a
small north tilt on these torques, independent of the density, while
its own observational law at the start is negative at every radius
(p050, gamma 0.999: -0.44 at r 0.5, -0.25 / -0.24 / -0.27 / -0.07 at r
1 / 2 / 4 / 8), and the tagged gate shows the vanilla critics to be
nearly action-blind (paired s.e. 0.003, row margin +0.01..+0.10, states
> 0 up to 0.93): a hair of consistent tilt, which an argmax over 64
candidates turns into 40% north picks exactly as it does for the branch
critics' +0.7..+1.1.

Conclusion for the record: **segment-rank with argmax does not separate
the branch critics from the vanilla critics at this rung**; it amplifies
any consistent tilt, and the vanilla critics have one.  The behavioural
number cannot be attributed to the branch replay.  What separates the
families is the critic-level preference: branch 30k +0.68..+1.11 on the
donor torques and +0.08..+0.26 on recorded torques (5/5 the right sign at
3-10x the vanilla size), vanilla +0.07..+0.42 and +0.01..+0.07.  Next: a
magnitude-respecting decoding (Boltzmann over the candidates' f, tau 1)
on both families, and the matched 30k vanilla critics.

### Remedy 1: Boltzmann segment choice separates the families

Same candidates (K = 64 recorded 25-step segments at the start, pure-BC
walker elsewhere), the segment sampled with p ~ exp(f / 1) instead of
argmax, 100 draws per ranker:

| ranker | success | failure | timeout | **detour** | shortcut | north picks / decisions |
|---|---:|---:|---:|---:|---:|---|
| branch 30k critic 0 | 0.33 | 0.40 | 0.27 | **0.23** | 0.60 | 33 / 286 (0.115) |
| branch 30k critic 1 | 0.30 | 0.39 | 0.31 | **0.21** | 0.54 | 35 / 303 (0.116) |
| branch 30k critic 2 | 0.24 | 0.50 | 0.26 | **0.18** | 0.63 | 30 / 341 (0.088) |
| branch 30k critic 3 | 0.43 | 0.37 | 0.20 | **0.26** | 0.58 | 33 / 218 (0.151) |
| branch 30k critic 4 | 0.32 | 0.40 | 0.28 | **0.20** | 0.60 | 29 / 242 (0.120) |
| vanilla 100k critic 0 | 0.25 | 0.60 | 0.15 | **0.04** | 0.83 | 10 / 197 (0.051) |
| vanilla 100k critic 1 | 0.29 | 0.48 | 0.23 | **0.08** | 0.72 | 18 / 261 (0.069) |
| vanilla 100k critic 2 | 0.27 | 0.52 | 0.21 | **0.11** | 0.72 | 14 / 222 (0.063) |
| (uniform pick, from above) | 0.24 | 0.56 | 0.20 | 0.05 | 0.77 | 22 / 684 (0.032) |

Branch critics detour 0.18-0.26 (mean 0.22; north-pick rate 0.09-0.15),
vanilla critics 0.04-0.11 (mean 0.08; 0.05-0.07), uniform 0.05 (0.03).
With the decoding made to respect the size of the preference, the
branch critics' +0.7..+1.1 nats show as a north-pick rate 3-4x the
candidates' proportion on 5/5 seeds, and the vanilla critics' few tenths
of a nat as 1.5-2x -- consistent with a Boltzmann rate of p_north *
exp(margin) / Z at those margins.  This is the honest behavioural readout
of the rung: the branch replay's contribution is a ~3x larger north
preference at the start state, which under a magnitude-respecting
decoding is a detour rate of ~0.22 against ~0.08, with the argmax figure
(~0.4 for both) an artefact of amplification.

### Remedy 2: matched-budget vanilla critics (30k) -- the argmax reads the tail, not the mean

Three vanilla critics at 30,000 updates on the recorded p050 data
(`vanilla_g0999_30k`), the four-group probe and the argmax segment-rank
(300 draws each):

| vanilla 30k critic | north-donor minus east-donor | recorded north minus east | argmax segment-rank detour | north picks / decisions |
|---|---:|---:|---:|---|
| seed 0 | -0.01 | +0.00 | **0.307** | 0.112 |
| seed 1 | -0.29 | -0.04 | **0.193** | 0.075 |
| seed 2 | -0.07 | +0.01 | **0.280** | 0.105 |

At the matched budget the vanilla critics carry no north tilt at all on
the same-state mean (seed 1 leans east by 0.29), and the argmax
segment-rank still detours 0.19-0.31 -- four to six times the 0.05 of a
uniform pick.  So the argmax over 64 candidates is not reading the
critic's mean preference; it reads the upper tail of f over the
candidate set, and the rare north segments (5% of the data, a
less-trained region of the critic) have the wider f spread and win the
max more often than their mean warrants.  This closes the argmax
decoding as a readout of anything: its 0.33-0.47 for the branch
critics, 0.38-0.47 for the 100k vanilla critics and 0.19-0.31 for the
30k vanilla critics all sit on the same tail effect.  The Boltzmann
choice (remedy 1) is the one that reads the mean and separates the
families (0.18-0.26 against 0.04-0.11); its cell on the 30k vanilla
critics was not run (the study stops here at the user's request) and
is the obvious next measurement.

### Where the rung stands (stop point, 2026-09-17 evening)

* Critic level (the method's own claim): branch-replay critics at 30k
  updates rank real north torques above real east torques at the same
  start state on 5/5 seeds, +0.68..+1.11 on the replay's donor torques
  and +0.08..+0.26 on recorded torques, against a law target of +0.60;
  every vanilla critic (100k or 30k, any rung) is within +-0.4 on the
  donor torques and within +-0.07 on recorded torques.  At 100k the
  branch critics memorise their rows and the preference is gone.
* Actor level (the paper's recipe): no gradient-trained actor on the Ant
  detours (0.00-0.02 on 300 draws for joint 100k, frozen critic with BC
  0.05 / 0.5 / 1.0); the 8-d actor climbs the NCE critic off the data
  manifold within 10k updates.  Vanilla at p 0.50: success 0.26,
  failure 0.73, detour 0.003-0.007 (3 seeds).
* Decoding level: a Boltzmann choice over recorded 25-step segments at
  the start, ranked by the branch critics, detours 0.18-0.26 (5/5)
  against 0.04-0.11 for the 100k vanilla critics and 0.05 uniform; the
  argmax variant is a tail artefact and is withdrawn as evidence.
* Everything on the Ant is at the oracle model (simulator with hazards
  redrawn); Phase 2 (a learned macro-ETT) has not started.

## Diagnostic (user-requested, 2026-09-17 night): does the first torque explain the outcome?

`scripts/diag_v6_first_step_crossover.py crossover` (`diag_crossover/`),
oracle stage, not a training result.  300 start-region anchors (t <= 5)
at p 0.50; per anchor one north donor and one east donor; four arms
crossing the donors' FIRST torque with their steps 2-25; the four arms
share the anchor's hazards, clocks and rock jitter (the env's six hidden
streams reseeded before each reset); after the 25 torques the same blind
driver takes over and picks its route FROM ITS POSITION (never from a
query label), zero-torque hold after reaching.

| arm | first | continuation | north @25 | east @25 | driver picks detour | reach | death | timeout | P_goal (g 0.999, r 0.5) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NN | north | north | 0.67 | 0.20 | 0.18 | 0.53 | 0.38 | 0.09 | 0.263 |
| NE | north | east | 0.02 | 0.91 | 0.00 | 0.22 | 0.77 | 0.01 | 0.139 |
| EN | east | north | 0.61 | 0.23 | 0.10 | 0.49 | 0.44 | 0.07 | 0.247 |
| EE | east | east | 0.02 | 0.97 | 0.01 | 0.23 | 0.77 | 0.00 | 0.145 |

| outcome | first-torque effect (N - E) | continuation effect (N - E) | interaction |
|---|---:|---:|---:|
| P(north after 25) | +0.03 | **+0.62** | +0.06 |
| P(driver takes detour) | +0.04 | +0.14 | +0.09 |
| reach | +0.02 | **+0.28** | +0.05 |
| death | -0.03 | **-0.36** | -0.06 |
| P_goal | +0.005 | **+0.113** | +0.02 |

log P_goal ratios: NN/EE +0.59, EN/EE +0.53 (only the continuation is
north), **NE/EE -0.05** (only the first torque is north), NN/EN +0.06.
Within anchor, paired hazards: swapping only the first torque changes
the 25-step direction in 0.22 of the pairs, swapping only the
continuation in 0.77.

**The first torque explains nothing of the outcome.**  Everything the
replay's north-query rows carried (+0.60 on their own paths) is the
property of the continuation; the (s, a_1) key the critic scores holds
a target of -0.05 .. +0.06 once the continuation is held fixed.  The
30k critics' +0.7..+1.1 on the north-donor torques was therefore not a
property of those torques: it was the label of the episode they came
from, read off a torque signature that happens to identify north
donors.  This is the same confound as PointMaze's Step 9g, one level
down -- there it was the teacher's recorded continuation; here it is
the query intent that steered the driver for the remaining ~400 steps
of every north-query path.

Two further readings from the split by the route the driver actually
took (`rollouts.json`):

| arm | driver route | n | reach | death | reach step (median) |
|---|---|---:|---:|---:|---:|
| NN | detour | 55 | 0.95 | 0.00 | 372 |
| NN | shortcut | 245 | 0.43 | 0.47 | 263 |
| EN | detour | 29 | 0.93 | 0.00 | 372 |
| EN | shortcut | 271 | 0.44 | 0.49 | 272 |
| NE / EE | shortcut | 299 / 298 | 0.22 / 0.23 | 0.77 / 0.77 | 229 / 226 |

(i) Twenty-five north torques do not commit the route: the driver, left
to its position, turns back east in 82-90% of the north-continuation
paths (the ant is not yet past y > 2).  The v2 replay's north-query
paths reached 0.72 because the query INTENT drove the driver north for
the whole episode -- a label, not the recorded segment.  (ii) The
north-continuation paths that turn back still die only 0.47-0.49
against 0.77: they arrive at the mouths ~40 steps later (263-272 vs
226-229) and pass more bursts.  Half of the continuation's benefit in
this position-driven version is the lateness loophole again, not the
detour.

Consequence, as the user framed it.  Keeping a single-step actor with
an honest generator (fix the first torque, one blind continuation from
the position) gives a start-state target of ~0: the law then says,
correctly, that the first torque does not decide the route on the Ant,
and there is nothing at that key for the critic to learn.  The route
lives at the macro-action level -- a committed segment or a command --
and a critic that is to carry it has to score that object (the
segment-rank policy did this by hand at evaluation time; it is not the
paper's actor and changes the action interface).  That choice is the
user's; the walker comparison (`walker` mode) follows for the
"right route, cannot finish" half.

### Honest single-step evaluation, part 1: the blind driver as the continuation

`diag_v6_first_step_crossover.py single` (`diag_single_driver/`): one
query torque at the state (N = a north donor's torque at the same time
step, E = an east donor's, own = the row's recorded torque), then the
replay's blind driver acting closed-loop from step 2, re-deciding its
route from its position at every step (the generator's rule plus the
descent leg x > 22, y > 1.5 -- without it a detour path flips to
"shortcut" as it comes down at x ~ 24 and stalls; that first pass is kept
as `diag_single_driver_rule_v0`).  150 states per set, paired hazards,
gamma 0.999, r 0.5.

| state set | reach (N / E / own) | death | P_goal N / E | log N/E | paired P_goal(N) > P_goal(E) | route differs N vs E |
|---|---|---:|---|---:|---:|---:|
| start (x<2, y<2, t<=5) | 0.21 / 0.21 / 0.21 | 0.79 | 0.132 / 0.121 | **+0.09** | 0.08 | 0.01 |
| detour turn in progress (detour eps, y<2, t>5) | 0.60 / 0.60 / 0.60 | 0.39 | 0.313 / 0.316 | **-0.01** | 0.23 | 0.02 |
| north leg low (x<2, 2<=y<4) | 0.99 / 0.96 / 0.97 | 0.00 | 0.388 / 0.384 | **+0.01** | 0.52 | 0.10 |
| north leg high (x<2, 4<=y<6) | 0.98 / 0.99 / 0.99 | 0.00 | 0.429 / 0.428 | **+0.00** | 0.37 | 0.11 |
| shortcut early (2<=x<5, y<2) | 0.25 / 0.25 / 0.25 | 0.75 | 0.173 / 0.173 | **-0.00** | 0.10 | 0.02 |

Under this continuation the single torque decides nothing anywhere: the
driver's route is a function of position, one torque does not move the
position across the rule's threshold, and from every turning state below
y = 2 the driver goes east (148/150 of the recorded mid-turn states end
on the shortcut).  A start-state target of +0.09 and turning-state
targets of ~0 are what a single-step replay built on this continuation
would ask the critic to learn.  This is the second branch of the user's
reading for the driver: the continuation policy takes every trajectory
back to the shortcut, and a replay generated once under it carries no
policy-improvement signal for the route.  Part 2 (the pure-BC actor as
the continuation) follows.

### Honest single-step evaluation, part 2: the pure-BC actor as the continuation; the walker comparison

Same protocol (`diag_single/`, both continuations in one run, 4,500
rollouts), the deployment pure-BC actor's mode acting closed-loop from
step 2.  Route taken = the path went around (max y >= 6); the
final-position rule misreads successful detours, which end at the goal.

| continuation | state set | around N / E / own | reach N / E / own | death N / E | P_goal N / E | **log N/E** | paired win / tie / loss | around differs N vs E |
|---|---|---|---|---|---|---:|---|---:|
| driver | start | 0.00 / 0.00 / 0.00 | 0.21 / 0.21 / 0.21 | 0.79 / 0.79 | 0.132 / 0.121 | +0.09 | 0.08 / 0.83 / 0.09 | 0.00 |
| driver | turn in progress | 0.24 / 0.23 / 0.27 | 0.60 / 0.60 / 0.60 | 0.39 / 0.39 | 0.313 / 0.316 | -0.01 | 0.23 / 0.43 / 0.34 | 0.03 |
| driver | north leg low / high | 1.00 / 1.00 / 1.00 | 0.99 / 0.96 / 0.97 ; 0.98 / 0.99 / 0.99 | 0 | 0.388 / 0.384 ; 0.429 / 0.428 | +0.01 ; +0.00 | 0.52 / 0.01 ; 0.37 / 0.07 | 0.00 |
| driver | shortcut early | 0 | 0.25 x3 | 0.75 | 0.173 / 0.173 | -0.00 | 0.10 / 0.77 / 0.13 | 0.00 |
| bc | start | 0.05 / 0.01 / 0.03 | 0.21 / 0.21 / 0.23 | 0.67 / 0.77 | 0.111 / 0.113 | -0.02 | 0.09 / 0.77 / 0.14 | 0.04 |
| bc | turn in progress | 0.56 / 0.59 / 0.63 | 0.50 / 0.45 / 0.53 | 0.11 / 0.09 | 0.155 / 0.138 | +0.12 | 0.38 / 0.33 / 0.29 | 0.20 |
| bc | north leg low / high | 0.99 / 0.99 / 0.99 ; 1.00 x3 | 0.81 / 0.79 / 0.83 ; 0.82 / 0.82 / 0.83 | 0 | 0.252 / 0.245 ; 0.250 / 0.253 | +0.03 ; -0.01 | 0.52 / 0.04 ; 0.45 / 0.05 | 0.01 ; 0.00 |
| bc | shortcut early | 0 | 0.25 / 0.25 / 0.25 | 0.72 / 0.72 | 0.167 / 0.134 | +0.22 | 0.15 / 0.77 / 0.08 | 0.00 |

Mid-turn states under the BC continuation, by how far into the turn
the recorded row is: t in [6, 15) n 54, around N 0.50 / E 0.61, log
+0.16, win 0.46 / loss 0.26; t in [15, 30) n 57, around 0.84 / 0.79,
log -0.06, win 0.39 / loss 0.42; t >= 30 n 39, around 0.23 / 0.26, log
+0.60 (P_goal 0.084 / 0.046, win 0.26 / loss 0.13).  The shortcut-early
+0.22 under BC rests on 11 of 150 pairs (positive differences summing
to 5.7 against 0.7 negative; reach steps 190 / 194): a few paths, not
a systematic effect.

What this says.  (1) Under a fixed blind continuation the single torque
carries at most ~0.1 nat about the route anywhere: 0.09 / -0.01 / 0.01
/ 0.00 (driver), -0.02 / +0.12 / +0.03 / -0.01 (BC) at the start,
mid-turn, and north-leg states, with paired wins 0.08-0.52 against ties
of 0.01-0.83 and losses 0.09-0.34.  (2) The route is decided by the
continuation policy's own state: the driver's latched position rule
takes every mid-turn state back east (around 0.23-0.27 for all three
torques); the BC actor goes around from mid-turn states 0.56-0.63 on
its own, and the north torque does not raise that (0.56 against 0.59
for the east torque; the pairs that differ, 0.20, go both ways).  At
the start the BC continuation goes around 0.05 after a north torque
against 0.01 after an east one -- a real but 4-point effect that does
not survive to P_goal (-0.02) because those late detours mostly time
out.  (3) The 0.1-nat figure is below the critics' measured seed noise
(~0.3 on the paired margins), so a replay built honestly on either
continuation gives a single-step critic nothing to separate at the
route level.

This is the second branch of the user's reading, stated precisely: the
"fix this continuation, generate the replay once" recipe has no
policy-improvement signal for the route on this benchmark, because no
fixed blind continuation lets one 8-d torque decide a 400-step route;
the route lives in what the continuation policy does after the torque.
The problem is at the continuation / action-interface level, not in the
critic.  (PointMaze was different in kind: a 2-d action IS a heading,
and one step at the fork cell decides the corridor.)

Walker comparison (`diag_walker/`, the same 100 placed rows per leg,
paired hazards):

| leg | walker | reach | death | timeout | reach step (median) | P_goal |
|---|---|---:|---:|---:|---:|---:|
| north | generation (frozen walker + blind driver) | 0.95 | 0.00 | 0.05 | 322 | 0.434 |
| north | deployment (pure-BC mode) | 0.83 | 0.00 | 0.17 | 380 | 0.256 |
| east | generation | 0.98 | 0.00 | 0.02 | 184 | 0.604 |
| east | deployment (pure-BC mode) | 0.92 | 0.00 | 0.08 | 236 | 0.402 |

Placed on the detour, the deployment walker completes it 0.83 / 0.92
(generation 0.95 / 0.98), about 50 steps slower.  The deployment gap
is modest; the missing piece is entering and sustaining the turn from
the start (BC around 0.01-0.05 from the start, 0.56-0.63 once mid-turn),
which -- per part 2 -- is not a single-torque quantity.

## Round 1 (user decision, 2026-09-18): the single-step actor kept, the replay's continuation = the policy itself

The user rejected two readings of the diagnostic and set the next step.
Rejected: (a) letting the critic choose a ROUTE and training BC on the
rows of that route (that changes the policy-extraction step, so it could
no longer be described as a change to where the positives come from);
(b) the conclusion that "N/E labels carry no consistent advantage" shows
"no concrete torque carries a learnable advantage".  The second point is
the important one: at one state candidate A may lead somewhere better, at
another candidate B, and a state-conditioned critic can pick A here and
B there without any torque ever "meaning north" -- the original CRL actor
optimises f(s, a, g) at the state, action and goal at hand, not a ranking
of action families.  The 20% of mid-turn pairs whose route flipped,
cancelling in aggregate, are therefore not evidence of noise until the
per-key differences have been checked on fresh consequences.

The round (one iteration, then decide):

1. The full continuous problem is kept: critic f(s, a, g) on 8-d torques,
   actor = the original critic term + BC at 0.05, no route command, no
   route-filtered BC rows.  The actor is INITIALISED from the pure-BC
   walker (`joint_purebc/seed_0/final.pkl`, lambda 1.0, 100k) so the
   critic term moves a walker instead of building one; the final policy
   is still the CRL actor, not BC.
2. The replay scores what the current policy would actually execute:
   at a recorded state, one query torque, then the FROZEN current policy
   (its mode) acts closed-loop from the resulting state to the horizon;
   hazards / clocks / jitter redrawn per draw and paired across the
   candidates of one state; successes and failures kept; the future law
   unchanged (gamma 0.999, radius 0.5, anchor = row 0, zero-torque hold
   after reaching).  Candidates = the recorded torque and samples from
   the policy's own tanh-normal at that state (no donor labels).  The
   same (state, torque) key is branched several times so the replay
   carries the expectation over consequences rather than one realised
   future the critic could memorise.  States: the start region (t <= 5),
   the turning process (detour episodes still in the start region after
   t = 5), late start-region rows of shortcut episodes, the north leg,
   the early shortcut, plus every 60th row of every episode.
3. The updated actor is checked on states and consequences it was not
   fitted on: its mode torque once, then the OLD frozen policy, paired
   fresh draws against the old mode torque and the recorded torque; the
   full policy's walking ability by the usual 300-episode evaluation.
   The critic's own scores are not the evidence.

Stop rules (the user's): critic fails to order even the repeated,
validated single-step differences -> fix value estimation, no actor;
critic orders them but the actor update keeps choosing worse or
non-walking torques -> the policy-extraction step is the bottleneck here;
one real improvement -> the updated actor becomes the next round's
continuation (the front torques' values change once the later turning
improves, so "no advantage at the start under the old BC continuation"
is not "never").  The virtual data being regenerated with the policy is a
change to the sampling procedure and is stated as such; this stage is
the oracle (simulator) model, not an offline result.

Implementation: `scripts/build_v6_policy_replay.py` (build / gate /
validate; `V6_JOINT_ACTOR_INIT=<ckpt>` in `run_v6_branch_replay.py` warms
the joint stage's actor), chain `node_r1_30108.sh`: replay (1,500 dense
anchors x {recorded, 2 samples} x 2 draws + 4,378 general anchors x
{recorded, 1 sample} x 1 draw = 17,756 paths; 100 held-out episodes give
360 anchors x {recorded, mode, 3 samples} x 4 draws = 7,200 paths for the
gate), 30k critics (3 seeds, milestones 10k / 20k), 30k vanilla critics
on the recorded futures (2 seeds), the gate, then for every passing
critic the 30k actor update (frozen critic, warm actor, bc 0.05,
balanced BC rows as before), the 300-episode evaluation and the
validation of the 10k / 20k / 30k actor milestones.

Gate definition.  On a held-out state, a pair of candidates (a_i, a_j)
is VALIDATED when the log P_goal ratio has the same sign on draws {0, 1}
and on draws {2, 3}, with |ratio| > 0.3 on both.  Agreement = share of
validated pairs whose critic ordering f(s, a_i, g) - f(s, a_j, g) matches
(chance 0.50), also restricted to pairs of two policy samples (so that
"recorded beats a perturbed torque" alone cannot pass it); pick gain =
(P[argmax f] - mean P) / (max P - mean P) over anchors with spread.
Baselines: random, the frozen policy's own log-likelihood of the
candidate, the vanilla critics.  The split-half sign agreement of the
differences themselves is the ceiling.  PASS = pooled dense-set
validated agreement >= 0.65 and above 0.5 by two s.e.

### Round 1 result (2026-09-18): stopped at the first rule -- the critic fits its own keys and carries nothing to new states

Chain `node_r1_30108.sh`, outputs under `outputs/antmaze_branch_replay_p050/`
(`generation_policy_r1.md`, `gate_policy_r1.md`, `gate_policy_r1_knn.md`,
`probe_r1_fit_agreement.md`, `holdout_policy_r1.npz`; the replay itself
and the checkpoints stay on the node / local disk).  Replay: 17,756 fit
paths in 1,791 s (10 workers); the policy's tanh-normal scale at the
anchors 0.445 (median 0.211), so the sampled candidates are real
perturbations of the mode (torque distance ~0.3-0.5).  Three 30k critics
(7.6 min each), two matched 30k vanilla critics on the recorded futures.

**1. What the replay contains (held-out anchors, 60 per set, 5 candidates
x 4 paired draws).**  [The "split-half sign agreement" column of this
table and the paragraph after it are WRONG -- the statistic counted a
second-half tie (both candidates' P identical, typically both died) as a
reversal; see the correction section below for the tie / same / opposite
breakdown.  Kept as the record of what was reported.]  Between-candidate
differences at ONE state are reproducible across hazard draws only in
the turning process and on the north leg; at the start they are noise:

| set | candidates' reach (recorded / mode / 3 samples) | pairs with abs log ratio > 0.3 | split-half sign agreement | validated pairs |
|---|---|---:|---:|---:|
| start (t <= 5) | 0.25 / 0.26 / 0.27 / 0.24 / 0.25 | 0.22 | **0.38** | 28 |
| start late (shortcut eps, t > 5) | 0.21 / 0.22 / 0.22 / 0.22 / 0.21 | 0.24 | **0.41** | 39 |
| shortcut early | 0.25 x5 | 0.21 | **0.26** | 16 |
| turning (detour eps, x<2, y<2, t>5) | 0.65 / 0.53 / 0.58 / 0.56 / 0.51 | 0.67 | **0.76** | 287 |
| north leg | 0.83 / 0.80 / 0.70 / 0.88 / 0.72 | 0.74 | **0.82** | 314 |
| whole maze (every k-th row) | 0.73 / 0.71 / 0.71 / 0.69 / 0.69 | 0.32 | **0.89** | 143 |

[WRONG, corrected below: the values below 0.5 at the start-type sets are
ties, not reversals -- 0 of 77 / 94 / 61 pairs reversed.]  Where the
differences are real (turning, north leg) they are
about completing the walk -- the reach spread across a state's five
candidates is 0.63 / 0.61 -- and not about the route: within-anchor
correlation of a candidate's P_goal with its going-around is 0.10
(went-around spread 0.33 in the turning set, 0 on the north leg).  The
same picture in the fit set (2 draws): split-half 0.29 / 0.29 / 0.23 at
the three start-type sets, 0.78 / 0.78 at turning / north leg.

**2. The gate (`gate_policy_r1.md`; validated pairs, chance 0.50; s.e. =
anchor bootstrap; a random scorer over 40 seeds gives 0.50 +- 0.03 pooled
and +- 0.035-0.048 per set).**

| scorer | turning (287 pairs, 50 anchors) | north leg (314, 57) | pooled dense (684, 123) | samples-only pooled (222) | pick gain pooled |
|---|---:|---:|---:|---:|---:|
| r1 critic seed 0 / 1 / 2 (30k) | 0.47 / 0.47 / 0.46 | 0.52 / 0.54 / 0.54 | **0.51 / 0.52 / 0.50** (s.e. 0.025) | 0.50 / 0.52 / 0.45 | +0.03 / +0.02 / -0.10 |
| r1 critic seed 0 at 10k / 20k | 0.39 / 0.41 | 0.55 / 0.60 | 0.48 / 0.51 | 0.46 / 0.50 | -0.06 / +0.02 |
| vanilla 30k seed 0 / 1 (recorded futures) | 0.44 / 0.45 | 0.46 / 0.47 | 0.46 / 0.46 | 0.49 / 0.50 | -0.17 / -0.01 |
| policy log-likelihood of the candidate | 0.52 | 0.47 | 0.52 | 0.53 | -0.06 |
| kNN on the fit keys (k 10; torque weight 1 / 4) | 0.40 / 0.44 | 0.48 / 0.48 | 0.43 / 0.45 | 0.46 / 0.45 | -0.01 / 0.00 |
| a predictor that saw two of the four draws at the same state | (1.00 by construction) | | | | 0.76 |

No scorer is above chance on held-out states; the pass line (0.65 pooled)
is not approached by any critic at any milestone; 0/3 pass, chain
stopped before the actor stage (`MARK_R1_STOP_CRITIC`), per the first
stop rule.

**3. Why: the critics DO fit the differences on their own keys
(`probe_r1_fit_agreement.md`, the fit anchors' 3 candidates x 2 draws,
validated the same way).**

| scorer | turning (319 pairs) | north leg (410) | pooled dense (795) | samples-only (249) | pick gain pooled (938 anchors) |
|---|---:|---:|---:|---:|---:|
| r1 critic seed 0 / 1 / 2 (30k) | 0.92 / 0.92 / 0.91 | 0.84 / 0.85 / 0.87 | **0.87 / 0.88 / 0.88** | 0.88 / 0.88 / 0.88 | 0.46 / 0.41 / 0.40 |
| r1 critic seed 0 at 20k | 0.93 | 0.79 | 0.84 | 0.84 | 0.34 |
| random | 0.48 | 0.52 | 0.51 | 0.53 | 0.06 |

On the training states the critic orders the candidates the way their
(two-draw-validated) consequences go, 0.87-0.88, including pairs of two
policy samples; on states from other episodes of the same kind it is at
0.50.  The nearest-neighbour control says the same thing without a
network: the fit keys' own outcomes, averaged over the 10 nearest
(state, torque) keys, order held-out candidates at 0.43-0.45.  So the
value the critic learns is a per-(state, torque) table entry: the effect
of a torque perturbation on whether the walker completes the turn is a
function of the exact pose at a resolution that 300 turning anchors and
300 north-leg anchors do not cover, and it does not transfer by
proximity in (s, a) space at this sample size.

**Reading against the round's premise.**  The premise was right in one
part and not in the other.  Right: concrete torques DO have
state-conditioned consequences that are reproducible at the same state
(0.76-0.82 on the turn and the north leg), which the aggregate N/E
reading had hidden.  Not right: those consequences are not learnable as a
function of the state from this data -- they neither transfer through
an NCE critic nor through nearest neighbours -- and they are about
finishing the walk, not about the route.  At the start region, the only
place where the route is decided, the single torque's consequences are
[withdrawn -- see the correction: mostly unverifiable with four draws,
not shown to be noise] not reproducible even at the same state under
this continuation, so there is nothing for any value estimator to carry
there in round 1, and
the iteration the plan relies on (improve the turn first, then the
front torques' values change) cannot start: the mid-turn signal the
critic would improve from does not generalise to the states the
updated actor will meet.

What "fix value estimation" would have to mean here.  Not a different
loss: the differences are real per state but do not transfer.  The
levers are data density in the turning process (the dataset has 54
detour episodes, 1,277 turning rows; a rung collected with more teacher
detours multiplies that) and a coarser action variable at which the
consequences are smooth across states -- the second is the macro-action
/ subgoal change the user has kept off the table, the first cannot touch
the start region.  Both are the user's call; the actor stage was not run.

### Correction after the user's review of d2583f8 (2026-09-18): ties are not reversals; the gate compared a point score with a region probability

Two errors in the round-1 report, both found by the user from the saved
`holdout_policy_r1.npz` and the code.

**(a) The split-half statistic.**  `np.sign(a) == np.sign(b)` counts a
second-half ratio of exactly zero -- both candidates with identical P,
in practice both died under both second-half draws -- as a disagreement.
The corrected breakdown (`gate_policy_r1_v2.md`; pairs whose first-half
|log ratio| > 0.3, classified by the second half):

| held-out set | episodes | first-half > 0.3 | tie | weak (0 < abs <= 0.3) | same sign | opposite sign | agreement among decided (episode-bootstrap s.e.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| start (t <= 5) | 44 | 77 | 48 | 1 | 28 | 0 | 1.00 |
| start late (shortcut eps) | 48 | 94 | 55 | 0 | 39 | 0 | 1.00 |
| shortcut early | 46 | 61 | 45 | 0 | 16 | 0 | 1.00 |
| turning | 5 | 404 | 28 | 35 | 287 | 54 | 0.84 (0.025) |
| north leg | 5 | 423 | 1 | 54 | 314 | 54 | 0.85 (0.029) |
| whole maze | 45 | 166 | 15 | 5 | 143 | 3 | 0.98 (0.019) |

(fit set, draw 0 vs draw 1: start 126 pairs -> 85 tie / 36 same / 4
opposite; turning 434 -> 31 / 319 / 46, 0.87; north leg 621 -> 16 / 410 /
78, 0.84.)  So at the start-type sets the reported 0.38 / 0.41 / 0.26 were
not "reversal more often than agreement"; 0 of 232 pairs reversed.  What
is true there is weaker and different: with reach 0.21-0.27 and death
0.70-0.78 under this continuation, most (candidate, draw) pairs end at
P = 0 for both candidates, so with four draws only 28 / 39 / 16 pairs per
600 can be decided at all, and those are all same-sign.  The
"regression to the mean" sentence, "the single torque's consequences are
not reproducible even at the same state" and "nothing for any value
estimator to carry there" are withdrawn: the labels at the start are
mostly UNVERIFIABLE at this number of draws, which is not the same as
noise, and the decided ones agree.  (Nor can the reverse be claimed: the
decided pairs are few and selected.)  The split-half agreement was also
mis-described as "the ceiling any critic can reach"; it is the
reproducibility of the labels, not an upper bound on a critic.

Uncertainty is now clustered by episode (the turning and north-leg
held-out anchors come from 5 detour episodes, not 60 independent states);
the pooled dense-set validated pairs come from 19 episodes.  The gate
decision is unchanged by either correction: the validated pairs (same
sign, both halves beyond 0.3) are the same 684, and no critic was above
chance on them.

**(b) The readout.**  The rollouts' answer is the probability that the
future lands within radius 0.5 of the goal (the success region); the gate
scored f(s, a, g) at the exact recorded goal point.  The NCE optimum
gives exp f(s, a, g) proportional to p(g | s, a) / p(g) with p(g) the
negatives' goal marginal, so the region probability is
E_{g ~ p}[1[g in R] exp f(s, a, g)] -- an expectation of exp(f) over
marginal frames in the region, not an average of logits.  Added to the
gate (`region_scorers`): the marginal sampled from the fit replay the
critics trained on (anchor uniform over paths, future row geometric at
0.999, 4 per path, 71,024 XY frames), per anchor the frames within
radius 0.5 (and 1.0) of THAT anchor's goal (the goals differ per
episode: x 24.1-25.4, y 0.1-1.5), up to 512 of them, log of the mean
exp(f) plus log of the in-region fraction; per twin head and for the
deployment min over heads (applied to the logits before the exponential,
as the actor applies it).  The exact-point logits per head are kept.

| readout (pooled dense sets, 684 validated pairs, 19 episodes) | r1 critic seed 0 / 1 / 2 | vanilla 30k seed 0 / 1 |
|---|---|---|
| held-out, exact point, min (the gate as before) | 0.51 / 0.52 / 0.50 | 0.46 / 0.46 |
| held-out, exact point, head 0 ; head 1 | 0.47, 0.50 / 0.49, 0.50 / 0.48, 0.51 | 0.45, 0.47 / 0.45, 0.49 |
| held-out, region r 0.5, min | 0.49 / 0.48 / 0.49 | 0.47 / 0.45 |
| held-out, region r 0.5, head 0 ; head 1 | 0.48, 0.49 / 0.48, 0.48 / 0.50, 0.49 | 0.45, 0.47 / 0.44, 0.49 |
| held-out, region r 1.0, min | 0.48 / 0.47 / 0.48 | 0.46 / 0.43 |
| FIT keys (795 validated pairs, 70 episodes), exact point, min | 0.87 / 0.88 / 0.88 | 0.49 / 0.49 |
| FIT keys, region r 0.5, min ; head 0 ; head 1 (seed 0) | 0.90 ; 0.89 ; 0.91 | 0.49 ; 0.49 ; 0.49 |

Per set on held-out keys the region readouts sit at 0.44-0.55 on
turning and north leg like the exact ones (`gate_policy_r1_v2.md`);
pick gain -0.19..+0.03.  On the critics' own fit keys the region readout
is slightly higher than the exact point (0.90-0.91 against 0.87) and
the two heads agree with each other and with the min throughout.

**Corrected reading.**  The region / point mismatch was real but is not
what separates the fit keys from the held-out keys: every readout is at
0.87-0.91 on the training keys and at chance on states from other
episodes, with vanilla critics at chance on both.  This is the user's
second branch -- train good, validation poor under both readouts -- so
the next budget, if any, goes to a controlled comparison of data
coverage with the action, the loss and BC 0.05 unchanged, not to the
goal-distribution correspondence.  What the training-key figure means
remains open: two draws per key cannot separate "fitted the expected
difference" from "memorised both realised futures", and the held-out
turning / north-leg states come from 5 episodes, so "does not transfer"
is measured on few independent poses.  At the start-type sets the third
branch applies as well: the labels themselves are mostly undecidable at
four draws, and deciding them needs more consequences per key before
any network is asked to fit them.  What stands from round 1: the
pre-registered gate failed for all three critics, and the actor stage
stays unrun.

### Diagnostic A / B / C (2026-09-18, after 69e6572; frozen checkpoints, fresh oracle outcomes): which generalisation boundary fails

Task: distinguish (A) memorisation of the realised training futures, (B)
no generalisation to new torques at a familiar state, (C) no
generalisation to states of other episodes -- with the round-1 critics
and the replay's continuation policy frozen, nothing trained.  Script
`scripts/diag_v6_r1_abc.py` (seal / run / analyze), outputs
`outputs/antmaze_branch_replay_p050/diag_r1_abc/` (manifest.json sealed
before any rollout: hashes, anchors, torques, seeds, rules; outcomes.npz
per draw; metrics.json; REPORT.md; SUMMARY.md).  Design: 5 strata x 32
anchors per pool, chosen round-robin over episodes without consulting
outcomes or scores; A = fit keys with their original three torques
(131 episodes), B = the same anchors with three new policy samples and
the same draw seeds, C = held-out anchors with their existing five
candidates (67 episodes; turning and north leg from the 5 held-out
detour episodes); 16 new paired hazard / clock / jitter draws per anchor
(28,160 paths, 2,192 s on 10 workers); the same continuation mode, hold,
horizon and law as round 1; region-integrated readout (radius 0.5, NCE
goal marginal of the fit replay, per head and deployed min) primary,
exact-goal logit secondary; pair classes tie / weak / decided (|log
ratio| > 0.3 on 16 draws) with ties never counted as wrong; s.e. by
episode bootstrap; a cross-fitted empirical selector (eight draws pick,
the other eight score) as the usable-signal reference.

Result (pooled dense strata, region min, agreement among decided pairs,
seeds 0 / 1 / 2):

| layer | agreement | pick gain in P_goal | verdict |
|---|---|---|---|
| A familiar state + familiar torque, fresh outcomes (230 pairs, 131 eps) | 0.77 / 0.71 / 0.70 (s.e. 0.03) | +0.042 / +0.032 / +0.032 (s.e. 0.007) | succeeds 3/3 |
| A, policy-sample pairs only (73) | 0.77 / 0.71 / 0.67 | +0.025 / +0.020 / +0.016 | succeeds |
| B familiar state + three new torques (222 pairs) | 0.55 / 0.51 / 0.55 (s.e. 0.035) | +0.013 / -0.000 / +0.005 (s.e. 0.006) | fails to inconclusive |
| A x B cross pairs (660) | 0.60 / 0.57 / 0.57 (s.e. 0.023) | - | the familiar member carries it |
| C held-out-episode states, five candidates (758 pairs, 67 eps) | 0.50 / 0.48 / 0.49 (s.e. 0.025) | -0.000 / +0.001 / -0.005 | fails 3/3 |
| C matched types (227) | 0.47 / 0.48 / 0.46 | -0.006 / -0.008 / -0.003 | fails |

Memorisation check at A: the critic against the ORIGINAL two-draw
labels 0.86 / 0.82 / 0.78, against the FRESH sixteen-draw labels 0.77 /
0.71 / 0.70; where it matched the original label it keeps the sign on
fresh draws 130 / 6 (seed 0).  Original and fresh labels agree on the
turning / north-leg strata (Spearman 0.82-0.90 per key) and barely on
the start-type strata (0.12-0.36), where most 16-draw pairs are weak or
tied.  The fresh outcomes carry usable selection signal in every layer
(cross-fitted empirical selector +0.061 / +0.055 / +0.078; the critic's
own pick recovers +0.042 of the +0.061 at its keys and nothing in B or
C).  Exact-goal and per-head readouts: A 0.66-0.79, B 0.48-0.58, C
0.47-0.53 -- the same pattern.

Reading (the task's rules): A succeeds, B fails, C fails -> the critic's
value is specific to the trained (state, torque) keys; it does not
interpolate across torques at a known pose, and consequently carries
nothing to poses of other episodes.  Not memorisation of realised
futures (A held up on fresh consequences), not label noise (the earlier
report's reading, already withdrawn).  Open: B's residual (0.55 on two
seeds, 0.51 on one, 222 pairs at sixteen draws); C's turning / north-leg
strata on 5 episodes; how the critic learned the start-type keys whose
own two draws were nearly uninformative (from the surrounding rows of
the same replay, mechanism not identified).  Single next intervention
best supported: a controlled comparison of torque coverage per state in
the replay (more policy-sampled candidates per anchor, repeated draws;
anchors, continuation, NCE, 8-d torque, actor objective, BC 0.05
unchanged), read out with this A / B / C design -- a diagnostic finding,
not a guarantee; "more episodes alone" is not what the pattern points
at.  Round-1 gate status unchanged; actor stage still unrun; nothing
committed from this diagnostic yet (user to decide).

### Torque-coverage experiment (2026-09-18, after the A / B / C diagnostic): repeated outcomes (R) vs action coverage (COV), equal budget

User task: test whether insufficient within-state torque coverage is
responsible for the B failure, without touching loss, architecture,
actor, route abstraction or environment.  Script
`scripts/exp_v6_r1_coverage.py` (seal / build / analyze / report), the
weighted anchor prepare in `run_v6_branch_replay.py`
(V6_ANCHOR_WEIGHTS=audit -> TrajectoryBuffer.set_anchor_strata with
per-path weights), the weighted goal marginal in
`build_v6_policy_replay.marginal_goal_frames`, and the unsealed-critic /
extra-layer options in `diag_v6_r1_abc.py analyze`; paired comparison
`scripts/probe_v6_coverage_paired_boot.py`.  Outputs
`outputs/antmaze_branch_replay_p050/exp_r1_coverage/` (manifest.json
sealed before any rollout, manifest_git_sidecar.json, build_arm{R,COV}.json,
REPORT.md, SUMMARY.md, logs/) and `diag_r1_abc/{REPORT,metrics}_arm{R,COV}.*`,
`outcomes_armCOV_A.npz`, `paired_boot_{A,B,C}.json`; critics
`critics_arm{R,COV}/seed_{0,1,2}` (final.pkl pulled locally; the arm
replays, 1.6 GB each, stay on the two rented nodes and are reproducible
from the manifest).  Arms: on the 1,500 round-1 dense anchors, R = the
same two policy torques x 8 fresh outcomes, COV = eight new policy
torques (a training-torque stream disjoint from every validation
stream) x 2 outcomes; 16 policy branches per state in both; the round-1
general and recorded paths verbatim; per-path anchor weights 1 / 0.25 so
that state mass (dense 0.5069) and the recorded / policy mixture (0.4155
/ 0.5845, within dense 1/3) equal round 1 exactly (audited from the built
replays); 30k NCE, seeds 0 / 1 / 2, the round-1 recipe unchanged; the
same A / B / C diagnostic and readout, COV's layer A on its own training
torques (cov0 / cov1, run with the diagnostic's draw seeds).

Result (pooled dense, agreement among decided pairs; seed 0 / 1 / 2):

| layer | r1 | R | COV | seed-mean r1 / R / COV |
|---|---|---|---|---|
| A | 0.77 / 0.71 / 0.70 | 0.67 / 0.67 / 0.70 | 0.63 / 0.62 / 0.66 (cov0 / cov1) | 0.73 / 0.68 / 0.64 |
| B | 0.55 / 0.51 / 0.55 | 0.58 / 0.53 / 0.55 | 0.63 / 0.56 / 0.56 | 0.536 / 0.554 / 0.584 (paired s.e. 0.03) |
| C | 0.50 / 0.48 / 0.49 | 0.48 / 0.50 / 0.52 | 0.51 / 0.48 / 0.47 | 0.491 / 0.500 / 0.485 |

Paired episode bootstrap on B (222 decided pairs, 78 episodes): COV - r1
+0.048 +- 0.039 (z 1.2), COV - R +0.030 +- 0.035, R - r1 +0.018 +- 0.026;
pick gains +0.012 / +0.007 / +0.007 (COV), +0.010 / 0 / +0.009 (R),
+0.013 / 0 / +0.005 (r1).  Pre-registered line (above r1 and R by 2 s.e.)
NOT met -> Case 4: neither arm materially improves B; the fresh untouched
action set was not run.  What is established: repeated outcomes do not
help (B z 0.7; A even slightly lower, 0.68 vs 0.73), so target variance
at the training keys is not the limiting factor; broader torque coverage
lowers the familiar-key fit (0.64 vs 0.73, a quarter of the mass per
torque) and raises B by a few points at most, directionally consistent
across seeds but inconclusive at this validation resolution; C at chance
for every arm.  Per the task: no further enlargement of the replay, no
16 / 32 / 64 torques, no longer training, no loss / architecture / route /
macro / subgoal change from this result.  Nothing committed (user to
decide); no actor trained.

### Diagnostic control (2026-09-18, after 355d273): direct supervised prediction of P_goal on the R-arm replay

User task: can a scalar predictor of the rollout target P_goal(s, a, g),
fitted by squared error to the existing R-arm outcomes, generalise where
the NCE critic does not?  Not a method change; no actor, no new rollouts;
NCE checkpoints untouched.  `scripts/diag_v6_supervised_pgoal.py` (seal /
train / analyze), outputs `outputs/antmaze_branch_replay_p050/diag_supervised_pgoal/`
(sealed manifest, training keys, six runs, metrics.json, REPORT.md,
SUMMARY.md).  Training keys = the R replay's 13,256 exact (anchor,
candidate) keys with summed weights and weight-averaged targets (state
mass / mixture exactly round 1); S h(s, a, g) on the critic's inputs, N
h0(s, g) without the torque; (1024, 1024) relu, sigmoid output, Adam 3e-4,
batch 1024, 30k updates, seeds 0-2, identical key draws for S and N;
evaluation on the sealed A / B / C fresh outcomes with the diagnostic's
decided pairs, tie-aware (N ties every pair by construction), pick gain
with uniform tie-breaking, episode-bootstrap paired differences.  Caveat:
the six runs ran on CPU (one device type).

| | training fit (weighted R^2) | A agreement / gain (s.e. 0.03 / 0.009) | B | C |
|---|---|---|---|---|
| S seeds 0/1/2 | 0.95 / 0.95 / 0.95 | 0.87 / 0.87 / 0.85; +.056 / +.054 / +.056 | 0.54 / 0.61 / 0.54; +.005 / +.020 / +.002 | 0.52 / 0.53 / 0.52; +.002 / +.013 / +.008 |
| R NCE seeds 0/1/2 | -- | 0.67 / 0.67 / 0.70; +.029 / +.030 / +.036 | 0.58 / 0.53 / 0.55; +.010 / .000 / +.009 | 0.48 / 0.50 / 0.52; -.009 / +.001 / +.004 |
| N seeds 0/1/2 | 0.79 / 0.79 / 0.79 | all ties; gain ~0 | all ties; ~0 | all ties; ~0 |

Paired S - NCE: A +0.18 +- 0.03 (z 7; every seed +0.16..+0.20); B +0.008 +-
0.036 (z 0.2; per seed -0.045 / +0.081 / -0.014); C +0.022 +- 0.024 (z 0.9);
C matched +0.076 +- 0.037 (z 2.1, the only cell above two s.e., not backed
by the full set).  Level error against the 16-draw means: S at A
0.012-0.013 (constant 0.0175, noise floor 0.003), at B 0.032-0.039 and at
C 0.048-0.063 -- worse than a constant; N 0.014-0.015 / 0.020-0.021 /
0.023-0.026.  Reading: S fits its keys and predicts fresh consequences at
them (better than the 30k critic), and transfers to neither unseen torques
at the same states nor held-out episodes; the predeclared transfer
condition is not met.  Ruled out: a failure specific to the NCE objective
or readout; a lack of learnable single-torque signal at the training
keys; outcome noise at the keys as the limit.  Open: coverage, input
representation, missing conditioning (absolute time / remaining horizon,
absent for every learner), and whether A's success rests on recurring
near-duplicate poses.  Not committed; user to decide.

### Pretrain on the recorded data -> continue NCE on the branch replay (user proposal, 2026-09-18)

Proposal: a from-scratch critic must learn the Ant's pose / torque
relations and the counterfactual value at once from ~13k branch keys;
pretrain the NCE critic on the full recorded data (268k transitions,
the V6 vanilla recipe, 100k), continue the SAME NCE on the branch replay
(R arm: counterfactual futures only, no recorded futures mixed back, no
ordering constraint), and test the pipeline with actors from the pure-BC
walker (frozen critic, bc 0.05, 30k) -- controls with the same actor init:
C1 = random init -> branch (critics_armR), C2 = pretrain only.
Pre-registered (`exp_pretrain_branch/manifest.json`, sidecar 355d273):
critic rule = improvement over C1 on B or C by 2 paired s.e. with
consistent seed direction and positive pick gain; policy rule = T's mean
success above C1 and C2 by 2 pooled seed s.e. and detour > 0.05 on >= 2
seeds.  Driver additions: V6_CRITIC_INIT (critic warm start, step reset)
and V6_EVAL_POLICIES; orchestration `scripts/exp_v6_pretrain_branch.py`.
Cost: pretraining 810 s per seed (4090) on top of the branch stage's 360.

| | A | B | C | actor success (seeds) | detour |
|---|---|---|---|---|---|
| T pretrain -> branch | 0.72 / 0.70 / 0.74 | 0.53 / 0.50 / 0.52 | 0.50 / 0.49 / 0.51 | 0.213 / 0.233 / 0.103 | 0 / 0.007 / 0 |
| C1 random -> branch | 0.67 / 0.67 / 0.70 | 0.58 / 0.53 / 0.55 | 0.48 / 0.50 / 0.52 | 0.303 / 0.000 / 0.000 | 0.007 / 0 / 0 |
| C2 pretrain only | 0.44 / 0.42 / 0.46 | 0.45 / 0.44 / 0.45 | 0.50 / 0.49 / 0.50 | 0.287 / 0.007 / 0.000 | 0 / 0 / 0 |

Paired (seed-averaged, episode bootstrap): B: T - C1 -0.036 +- 0.027 (z
-1.4, all seeds negative), T - C2 +0.069 +- 0.037; C: all differences
within 0.02.  Policy: T - C1 +0.082 (pooled seed s.e. 0.109), T - C2
+0.086 (0.103); detour > 0.05 on 0 / 3 seeds.  Both rules NOT met.
Readings: pretraining adds nothing at B or C (T at chance like C1; the
branch stage still fits its keys, A 0.72); the pretrain-only critic is
below chance at A and B (0.42-0.46), the expert-confounded ordering the
branch stage then undoes; at bc 0.05 with a frozen critic the actor
degrades from its BC init in every arm (T keeps walking at 0.10-0.23
success, C1 / C2 stop walking on two seeds each, the C2 survivor is a
slow after-burst shortcut) and no arm detours.  Caveats: T and C1 / C2
actors on different GPUs; the T chain relaunched once after a CPU
mis-launch (no result reused); C2's region readout uses the dataset's
goal marginal.  Not committed.  Details `exp_pretrain_branch/SUMMARY.md`,
`REPORT.md`, `diag_r1_abc/{REPORT,metrics}_{armT,pre100k}.*`,
`paired_boot_pretrain_{B,C}.json`.

### Episode / state coverage (NARROW vs BROAD): feasibility audit, stopped before generation (2026-09-18)

User task after 1966335: does spreading the counterfactual supervision
over more independent original episodes improve held-out generalisation
(turning / north-leg states first)?  Audit
(`outputs/antmaze_branch_replay_p050/exp_episode_coverage/FEASIBILITY.md`):
the p050 dataset has 54 detour episodes; the round-1 split leaves 49 in
the training pool and 5 held out; the R replay's turning and north-leg
anchors (300 each) already come from all 49 (median 5-6 anchors per
episode), so no BROAD arm exists for those strata at p050; the start-type
strata are not exhausted (335 / 172 / 252 of 851-900 episodes used) but
their held-out labels are mostly undecidable (88-104 decided pairs of 600)
and their states sit near the reset pose, so a contrast there neither
addresses the primary endpoint nor has power.  far10-far30 are p040 and
were not substituted.  STOPPED: no manifest, no rollouts, no training.
Proposed (not executed): Plan A = 5,000 more p050 episodes with the
identical collector settings (detour prob 0.05, seeds 607-611, ~270 new
detour episodes, ~30 reserved for evaluation), then NARROW (current 49
episodes) vs BROAD (~245 episodes, same 600 turn / north-leg anchors,
same R protocol and weights), 30k NCE x 3 seeds, one GPU, evaluation on
>= 30 held-out detour episodes with 16 paired draws; Plan B (1,000
episodes at detour prob 0.30) is cheaper but changes the behaviour-mode
mixture.  Awaiting approval; nothing committed from this step.

### Episode coverage NARROW vs BROAD, run (2026-09-18, Plan A approved)

Data: five new p050 shards (seeds 607-611, the original collector
settings; 1,000 episodes each, composition audits passed, detour 0.040-
0.051) -> 227 new detour episodes; merged `antmaze_rockfall_clock_v6_p050_plus5k`
(6,000 episodes); 30 new detour episodes reserved for evaluation (Cnew),
197 + 49 = 246 in the BROAD pool.  Sealed manifest
`exp_episode_coverage/manifest.json` (before any rollout).  Arms: NARROW =
the R replay verbatim (turning / north-leg anchors from 49 episodes,
median 5-6 per episode); BROAD = the same replay with only those 600
anchors redrawn over 246 episodes (max 2 per episode), same candidates /
draws / weights (effective masses identical to four decimals), same
continuation and law; 30k NCE, random init, seeds 0-2, both arms on the
same 4090.  `scripts/exp_v6_episode_coverage.py`; Cnew = 64 anchors x 5
candidates x 16 paired draws on the reserved episodes (`diag_cnew/`).

Primary (Cnew, 437 decided pairs, 30 episodes): agreement NARROW 0.46 /
0.51 / 0.45 (0.473 +- 0.026), BROAD 0.46 / 0.48 / 0.46 (0.465 +- 0.027);
pick gain NARROW +0.007 / +0.017 / -0.001, BROAD -0.011 / +0.003 / -0.008;
BROAD - NARROW pick gain -0.013 +- 0.011 (z -1.2, all seeds negative);
the outcomes' own cross-fitted selector gain is +0.164.  Pre-registered
criterion NOT met; actor stage not triggered.  Development: old C pooled
0.47-0.51 vs 0.49-0.52 (its 5-episode turning cell leans BROAD, 0.54-0.61
vs 0.44-0.47, the north-leg cell the other way); A / B reclassified
(turning / north-leg A anchors are NARROW keys only); NARROW's fresh
retrain differs from critics_armR by up to 0.09 on one seed (training
randomness).  Reading: 5x more independent detour episodes at equal
budget did not move held-out generalisation for this recipe; both arms
at chance on new episodes despite +0.16 of usable signal.  Not
committed.  Details `exp_episode_coverage/SUMMARY.md`.

### Restore determinism and one-step vs long-horizon predictability (2026-09-18, user's two-part check)

`scripts/diag_v6_restore_onestep.py`, `outputs/antmaze_branch_replay_p050/diag_restore_onestep/`.
(1) Restore: 60 anchors, recorded torque, fixed hidden seeds -- a fresh env
and an env that had just run 150 steps of another anchor agree to 0.0 at
step 1 and step 10; restore + recorded torque reproduces the original
next frame to 5e-7 (float32); warmstart and time are zero after restore
(mj_resetData).  No restoration defect.  (2) One-step change Delta s is
predictable from (s, a) with the critic's inputs: MLP R^2 0.999 (train, A),
0.99 (B, new torques), 0.91 (old C), 0.84 (Cnew, new episodes); kNN
0.87 / 0.86 / 0.76 / 0.60.  Long-horizon P_goal with the same
architecture and data: R^2 0.95 train, 0.25-0.32 A, below zero at B, old
C and Cnew (S predictors).  Within-anchor Spearman(one-step distance, P
difference) 0.34-0.40 on A / B / old C, 0.03 on Cnew.  Reading: the
second branch -- physics and one-step representation are fine; learning
the long-continuation value is the problem; adding samples of the same
kind is not the lever.  Not committed.

### Detour-demonstration share 5% vs 20% (2026-09-18, user's proposal; sealed at b8816b8)

`scripts/exp_v6_detour_ratio.py`, `outputs/antmaze_branch_replay_p050/exp_detour_ratio/`.
Nested 1,000-episode datasets from the merged p050 pool (old held-out and
Cnew excluded; seed 2028): d05 = 50 detour + 950 shortcut (rows 267k,
detour row share 0.070), d20 = 200 + 800 (286k, 0.269).  Per ratio, the
same recipe: pure BC 100k (seed 0) -> BC controls (300 natural draws;
placed completion on Cnew rows) -> vanilla critics x 3 (30k) -> branch
replay (r1 protocol, continuation = this ratio's BC mode; 18.7k / 19.4k
paths) -> branch critics x 3 (30k, uniform anchors) -> six actors (BC
init, frozen critic, bc 0.05, 30k) -> 300-draw evaluations; effective
share audited at every level (dataset 0.05 / 0.20 episodes, BC sampler
draws 0.051 / 0.201, vanilla anchor law 0.05 / 0.20, replay general
anchors 0.069 / 0.268).  d05 on node 30021 (4090), d20 on 30043 (4090L).

BC controls: route choice at the fork is zero for both (0 / 300 and
2 / 300 detours; success 0.257 / 0.260 = the no-hazard episodes); placed
completion of the detour 0.80-0.84 (d05) vs 0.91-0.92 (d20), driver
0.99.  Replay start-state counterfactuals go around 0.04 (d05) vs 0.16
(d20).  Actors (success; detour rate): d05 vanilla 0.263 / 0.263 / 0.260
(0.07 / 0.01 / 0.01), d05 branch 0.240 / 0.240 / 0.250 (0.14 / 0.02 /
0.07); d20 vanilla 0.543 / 0.723 / 0.647 (0.51 / 0.74 / 0.63; hazard
success 0.48-0.67), d20 branch 0.243 / 0.290 / 0.347 (0.16 / 0.16 /
0.30; hazard success 0.06-0.19).  Pre-registered comparisons (> 2 pooled
seed s.e., 3/3): vanilla d20 - d05 success +0.376 (s.e. 0.052) and
detour +0.596 -- met; branch d20 - d05 success +0.050 (0.030, not met),
detour +0.130 (0.058, met); branch - vanilla at d05 -0.019 (0.004, met,
negative), at d20 -0.344 (0.060, met, negative).  Reading: at 20% share
the NCE critic on recorded futures plus the BC-initialised actor learns
the route choice at the fork -- under THIS recipe (p_active 0.50, gamma
0.999, frozen 30k critic, BC-initialised actor with balanced rows); the
ladder's frozen V6 baseline (p_active 0.40, gamma 0.99, joint 100k from
scratch; notes/v6_detour_ladder.md section 5) never detoured even at
0.30, so which difference unlocks the demonstrations is not isolated
(discount is the a-priori candidate, untested); the counterfactual branch
futures as built give a critic that is worse for the actor at both
shares (no-hazard success drops to 0.78-0.90, 9-24% timeouts), and the
"branch adds a gain" reading is refuted at d20.  Why the branch critic
is worse is not established (candidates, untested: 15x fewer (s, a)
keys and every-60th general anchors; BC-mode continuation as the
futures' source instead of the teacher's recorded futures; replay-only
training).  Datasets pulled locally (hashes match the manifest); replays
and checkpoints on the nodes.  Not committed.  Details
`exp_detour_ratio/SUMMARY.md`.

### Detour-share follow-up: density, discount, 10% (2026-09-18)

User's questions after the 5% / 20% result: the ladder's frozen V6
baseline never detoured even at the 0.30 rung (p040, gamma 0.99, joint
100k), so which difference lets the recorded-futures chain detour?  Arms
(`exp_detour_ratio/manifest_followup.json`, `REPORT_followup.md`,
recorded-futures chain only, 3 seeds): A = p040 far20 at gamma 0.999:
success 0.593 / 0.357 / 0.590, detour 0.54 / 0.08 / 0.49 (vs the ladder
baseline 0.357 / 0.000 on that data; vs d20 -0.12, not > 2 s.e.) -> the
hazard density is not the separator; B = d20 at gamma 0.99: 0.353 /
0.367 / 0.330, detour 0.18-0.25 (-0.29 vs d20, 3/3) -> the discount is
the largest factor but does not bring the chain back to zero detours;
C = d10 at gamma 0.999: 0.393 / 0.460 / 0.330, detour 0.15-0.46 ->
monotone in the share (5 / 10 / 20 % = 0.26 / 0.39 / 0.64).  Driver:
V6_DISCOUNT override.  Not committed.

### Agent-update round 1 and the query extension (2026-09-18, user's plan with four corrections)

`scripts/exp_v6_agent_round.py`, `scripts/diag_v6_first_step_replay.py`,
`outputs/antmaze_branch_replay_p050/exp_agent_round/SUMMARY.md`.  Fixed
agent = joint_van_d20/seed_0 (seed-0 rule; 0.543 on the 909 draw).  Round 1
= the control replay's anchors / queries / draw seeds with the agent's mode
as continuation (`--query-ckpt`); critics x3, actors x3 from the fixed
agent; control = critics_br_d20 (BC continuation); reference =
critics_van_d20.  Everything re-evaluated on one fresh draw (seed 2909,
V6_EVAL_SEED).  The round-1 futures equal the control's (start stratum
around 0.15 vs 0.16): first-step check -- own first torque reproduces the
natural trajectories exactly (60/60, xy 1e-9), but a BC / recorded first
torque at the same reset sends the agent east (around 0.00-0.10 vs 1.00),
the teacher's detour torque sends it around even where it went straight
(0.83) -> the query set lacked the agent's own first torques.  Extension
(`--extra-query-ckpt`): agent mode + samples added at every anchor, masses
unchanged (verify_ext), both continuations.  Results, 2909 draw: start
0.507; control 0.300; round1 0.384; ext_bc 0.413; ext_ag 0.356; reference
(recorded-data critics) 0.661.  ext_bc - control +0.113 (3/3, met);
ext_ag - round1 -0.03 and ext_ag - ext_bc -0.06 (not met); every branch
arm below the start (3/3), the reference +0.154 above it (3/3).  Cdev
(96 anchors, 5 candidates, 16 draws, both label sets): all five critic
sets at chance, no pair 2 s.e. apart.  Verdict: round 1 not met, no
second round.  Infrastructure: node 30049's 3090 power-capped (~270 MHz,
CPU only); node 30021 died mid-run; ext_bc actors OOM on node 30108,
moved to 30043.  Not committed.

### Corrected vanilla readout, true t = 0 diagnostic, identical-batch actors (2026-09-18, user's two points)

`outputs/antmaze_branch_replay_p050/exp_same_batch/SUMMARY.md`.  (1) The
vanilla goal marginal is now `marginal_goal_frames(law='vanilla_draw')`
(episode uniform, row uniform within it, geometric future -- the buffer's
draw); readout `vanilla_u`, old `vanilla` kept: no change (Cdev +0.013 vs
+0.013; t0 +0.052 vs +0.046).  True t = 0 diagnostic (64 reset rows of the
old held-out episodes, both label sets): the critics SEPARATE there --
vanilla_u pick gain +0.052 (agent labels) / +0.053 (BC labels), ext_ag
+0.033 / +0.062, ext_bc +0.023 / +0.060, round1 +0.002 / +0.021, control
-0.017 / -0.019 (z vs control 2.0-4.1); the t <= 5 Cdev stays at chance
for all.  (2) train_joint anchors the actor's critic term at row 0 of
V6_BRANCH_REPLAY, so the reference actors trained at the RESET rows while
the branch actors trained at replay anchors.  Same-batch actors (audit:
identical hashes across critic tags), start 0.507: S1 (dataset row 0)
vanilla 0.661, ext_bc 0.501 (0.59/0.61/0.31), control 0.363, round1 0.353,
ext_ag 0.317; S2 (control replay anchors) ext_bc 0.404, ext_ag 0.364,
round1 0.313, control 0.300, vanilla 0.246 (2% detours).  vanilla S1 - S2
+0.416 (3/3): the reference's advantage was largely the batch arrangement;
on S2 ext_bc beats vanilla / control / round1 (3/3, > 2 s.e.); no cell but
vanilla@S1 beats the start.  Not committed.

### Mainline contract and the controlled O vs CF-oracle pilot (2026-09-19, after 157a212): audit complete, execution blocked

`notes/MAINLINE_CONTRACT.md`, `scripts/exp_v6_mainline_pilot.py`,
`outputs/antmaze_branch_replay_p050/exp_mainline_pilot/{SUMMARY,AUDIT,REPORT}.md`,
`manifest.json`.  The contract separates (a) the intended offline
learned-ETT method (single-step torques, offline data, the original NCE and
actor objectives with bc 0.05, counterfactual generation as the only
intervention), (b) the simulator-oracle pilot, and (c) the historical
diagnostic configurations (reset-only actor batches, frozen critics, the d20
initialisation, S1/S2 swapping, native branches as an ETT substitute,
pure-BC init, balanced BC, candidate queries, goal hold, repeated draws).
Every conclusion of the chain from Round 1 onward that read as a statement
about the method is reclassified as a statement about its configuration
(banners on the three SUMMARY files; the text is untouched).  Three explicit
sampling interfaces replace the replay-layout convention: critic anchors =
the recipe buffer's own law subsampled (60k draws, 53,747 unique logged rows
with multiplicity weights, one file for both arms); positive futures =
geometric over the anchor's recorded continuation (O) or over one branch
generated by the logged torque once + the fixed d05 agent's mode to the
first reach frame / death / horizon, no goal hold, hazards redrawn (CF);
actor batches = the recipe's TrajectoryBuffer over the whole d05 file
(random_goals 0, bc 0.05 on the same rows) in both arms.
`crl.losses.build_learner(separate_actor_batch=True)` evaluates the
unchanged losses on the two streams (documented as an interface change, not
a byte-for-byte reproduction of the shared-batch learner).  Matched joint
training: V6 recipe at gamma 0.999, 30,000 optimizer updates counted, paired
fresh critics, actors from the d05 agent, three seeds; evaluation on one
fresh 300-episode draw (seed 3909, mode), primary CF - O paired on common
episodes, practical CF - start separate, evaluation uncertainty separate
from seed variability.  Sealed manifest and audit: dataset / sidecar
hashes, held-out and Cnew exclusion, anchor determinism and law marginals,
critic futures inside the path, actor streams identical across instances
with 99.6 % non-reset rows and no boundary crossing, static gates G1-G8,
no d20 input -- all PASS; the checkpoint-dependent checks (query once, root
timestep, parameters move) PASS in a labelled toy smoke.  BLOCKER: the
contract's start agent `joint_van_d05/seed_0/final.pkl` existed only on node
30021, which died on 2026-09-18; not local, not on the other nodes.  Nothing
substituted; the user decides between a recovered copy, an authorised
re-derivation of the d05 chain (recorded as such), or another course.

### Mainline pilot result (2026-09-19): start agent re-derived, CF above O in 3/3 seeds, pre-registered rule not met

The user chose option (b): the lost d05 seed-0 chain was re-derived with
the exp_detour_ratio recipe and seed (`joint_van_d05/seed_0/REDERIVED.json`;
909-draw 0.247 success / 1/300 detours vs the lost record 0.263 / 20/300 --
the lost seed 0 was the most detour-prone of its siblings 20 / 2 / 4).  Then
the sealed pilot: 53,747 anchors, 53,747 CF branches (success 0.62 / death
0.29 / timeout 0.09; t = 0 anchors 0.30 / 0.66), six 30,000-update runs on
three nodes (both arms of a seed on one GPU), 300-draw evaluation (seed
3909, mode).  Start 0.243 success / 0.003 detour; O 0.250 / 0.263 / 0.250
(detour 0.02); CF 0.267 / 0.320 / 0.480 (detour 0.107 / 0.167 / 0.500, death
0.63 / 0.57 / 0.26, hazard-active success 0.08 / 0.13 / 0.41 vs O 0.00-0.03,
timeout 0.11 / 0.11 / 0.26).  Paired CF - O success +0.017 / +0.057 / +0.230,
mean +0.101, seed s.e. 0.065, episode-bootstrap s.e. 0.017, 3/3 -> rule NOT
met (needs > 0.130); CF - start +0.112 (0.064), 3/3, not met; O - start
+0.011 (0.004).  Detour +0.239, death -0.240, timeout +0.139, all 3/3.
Reading: under the matched procedure the recorded (sighted-teacher) futures
teach nothing at the fork while the oracle counterfactual futures move the
actor onto the detour in every seed, with a seed-dependent size that the
pre-registered rule does not certify; oracle evidence for the sampling
design, not a learned-ETT result; no follow-ups launched (stopping rule).
`exp_mainline_pilot/{SUMMARY,REPORT,AUDIT}.md`.

### Frozen-model trajectory diagnostic of the pilot (2026-09-19, user's single next step; no retraining)

`scripts/diag_v6_pilot_trajectories.py`, `exp_mainline_pilot/diag_traj/`.
Replay is chaotic at the episode level (1e-7 action differences flip some
outcomes; aggregates stable), so the CPU replay is the reference.  CF s2's
73 timeouts: freezes with saturated torques (9 standing 1.5-2.7 from the
goal, 14 near the start; the start agent has the same mode), 15 falls, ~30
slow / oscillating detour walks (first y >= 6 at median step 394; the
west-column turn is the slowest part); no wrong-direction, wall-stuck 0-4.
Route signal: the CF critics prefer the CF torque at the reset states
(P 0.86-1.00) and the actors follow it only partly (mode below the best own
sample by ~1 nat; pre-tanh scale widened to 0.75-1.3 vs O's 0.5); a single
CF s2 first torque sends the start agent around 14 %; the O and CF paths
separate at median step 7 (s2), the decision is the first ~7-30 torques as
a sequence.  Reset-row branches carry no "around" support (3 / 196).
Continuation from the detour entrance: driver 0.98, O s2 0.71, start 0.52,
CF s2 0.48 (n 42); from pre-stall states CF = start = O ~0.42-0.45 --
detour-walking competence is the shared limit, at most a weak CF-specific
deficit on seed 2.

Follow-up (same day, `candidates` / `pairs` modes): at the straight-going
starts a detour-realising first torque exists in 97 % of states (the
teacher's detour reset torques 0.25-0.44 detour vs the mode 0.03-0.10);
seeds 0 / 1's CF critics rank a teacher detour torque above the mode in 96 %
of states (+0.24 / +0.60 nats) and their top-1 realises the detour in
35-49 % vs the mode's 3 % -> "actor did not adopt"; seed 2's critic scores
every off-policy candidate 1.8-2.4 nats below its mode (within-state
Spearman -0.27) -> a blind spot off the actor's manifold.  19 handover
states where O finishes and CF does not: the first anomaly is an early
stall (16 / 19, median 63 steps after handover, half O's speed, west
column / top corridor), posture only later; the start policy finishes 13 /
19 of them -> a local execution regression on this subset.

### Corrections and the pre-registered actor-lr variant (2026-09-19, user's review of 52f0bf1)

Two statistical corrections (banner in diag_traj/SUMMARY.md section 6): the
share of straight-going starts with a detour-realising candidate is 74 / 82 /
96 % per seed (not "97 %"); the Spearman used two argsorts and mis-ranked
the tied outcome labels -- with average ranks it is +0.22 / +0.14 / +0.04
(success-only +0.15 / +0.08 / +0.02), so seed 2 reads as "weak ordering",
not "reversed".  Reframed conclusions: seeds 0 / 1 -- choosing by the
critic's score improves the realised outcome (top-1 success 109 / 122 vs
the mode's 62 / 62), so the policy does not fully use the scoring
information it has (why is not established; nothing licenses lowering BC);
seed 2 -- the scores are biased by candidate source (off-policy candidates
pressed down), co-adaptation vs poor generalisation undecided; the 19
O-finishes-CF-not pairs (start finishes 13) plus the 8 reverse pairs = an
uneven change in walking competence, not a broken walker.  User's
decision: keep bc 0.05 and the recipe; one pre-registered single-change
trial, actor lr 3e-4 -> 1e-4 (`exp_v6_mainline_pilot.py --variant
actor_lr1e-4`, manifest sealed before training, criteria 1-3 fixed).
RESULT: criterion 3 recovered (entrance-state reach 0.75 / 0.67 / 0.71 vs
base CF 0.75 / 0.42 / 0.48; timeouts 0.01-0.03); criterion 1 NOT met
(success -0.068 vs base CF, 2/3 negative; still +0.04 over O(variant) and
the start, 3/3); detour gain shrinks to a third (0.26 -> 0.07, 3/3).  The
route change and the walking loss scale together with the update size; the
actor learning rate is not the lever.  No further variant launched.

### Reference variants and the two boundary checks (2026-09-19, later the same day)

User's request: try lower BC (0.02) and BC 0 as references while keeping
the recipe; my addition: a label-free trust region to the start policy
(anchor 0.5).  All pre-registered (`--variant`), O and CF, 3 paired seeds,
3 nodes.  bc0.02: CF success 0.000 on every seed (300 / 300 timeouts, no
deaths -- the actor leaves the walkable manifold; it still turns north in
12-40 % of episodes); bc0: both arms 0.000.  anchor_start0.5: walking and
entrance continuation recovered (0.75 / 0.67 / 0.62) but the detour rate
back at the O level.  With actor_lr1e-4 these four lie on one axis (how far
the actor may leave the start policy) on which route gain and walking loss
come together.  Two read-only checks (`scripts/diag_v6_objective_checks.py`,
diag_traj section 9): (b) at real start-region rows the full actor
objective at bc 0.05 is at an exact tie between staying and moving (with a
widened policy) to the critic-preferred torque -- E_f +0.5 to +0.8 nats
cancelled by 0.05 x NLL +11 to +14; no barrier; the O actors' objective
gets worse; (c) at the pre-stall mid-route states the critic's action
gradient is orthogonal to walk-vs-stall (P 0.51-0.52) and one real actor
step moves those outputs by ~0.01 per step, 87-88 % of it from the
shortcut-corridor rows of the batch, 4 % from detour-leg rows -- parameter
interference, not a critic error.  User's framing: these decide whether a
sampling-layer repair has evidence or the problem is outside the contract's
boundary; "rebalance BC / protect walking by region" withdrawn as default
proposals.  Assessment in diag_traj/SUMMARY.md section 9: the futures
intervention does its part at the critic level; the tie and the
interference are properties of the actor objective and the row composition
(outside the boundary as written); the untested in-boundary lever is the
next policy-iteration round (the improved agent as the continuation).

### Round 2 (policy iteration) -- negative, below the round-1 futures (2026-09-19, evening)

User's request: use the round-1 CF agent ("already learned a bit of detour")
as the new continuation policy, regenerate the branch replay, retrain critic
and actor.  Sealed `round2/manifest.json`: per lineage s the round-1 CF agent
CF/seed_s continues the branches at the same anchors with the same hazard
seeds and initialises three arms with fresh paired critics -- O (recorded),
CF (regenerated), CFold (round-1 futures); no seed selection; same streams,
losses, bc 0.05, 30,000 updates, same 300 evaluation episodes; three nodes
(30043 / 30125 / 30016), ~55 min.  Result (`round2/REPORT.md`, `SUMMARY.md`):
CF2 success 0.240 / 0.240 / 0.250, detour 0.00 / 0.00 / 0.10 (from 0.11 /
0.17 / 0.50); O2 0.25, detour 0.01; CFold2 0.263 / 0.283 / 0.260, detour
0.16 / 0.12 / 0.21.  CF2 - O2 success -0.008 (seed s.e. 0.002, 3/3
negative) NOT MET; CF2 - CFold2 success -0.026, detour -0.13, 3/3 negative;
every round-2 arm below the lineage agent it started from (CF2 - CF1 -0.11
success, -0.22 detour, 3/3).  Two read-only checks explain it: (i) branch
profiles by logged route (`scripts/diag_v6_round2_branch_profile.py`): after
the logged shortcut first torque the lineage agents go around in 0.1-0.4 %
of the start-region branches (start agent 0.0 %), even the seed-2 agent that
detours natively in 50 % -- the route content of the futures is unchanged
(only the 5 % logged-detour anchors carry detour futures); the lineage
agents walk the top corridor better but are not better from the detour
start rows (more timeouts, longer) and die more on the shortcut anchors
(fewer freezes); (ii) critic readout at the 192 logged-shortcut reset states
(`scripts/diag_v6_round2_critic_readout.py`): the round-1 CF critics prefer
the teacher detour torques by +2.9 / +2.2 / +1.9 nats (top-1 detour share
0.96 / 0.94 / 0.52), the CF2 critics by +0.8 / +1.9 / +1.5 (0.28 / 0.96 /
0.71) -- lineage 0 lost the preference at the critic level (its regenerated
detour-start futures are the slowest), lineages 1 / 2 kept it and the ACTOR
still went to 0.00 / 0.10 detour.  Recorded futures return the actor to the
start level within 30k updates (O2), the round-1 futures hold it at
0.12-0.21 (the 0.50 of round-1 seed 2 was a transient), and a +1.5-1.9 nat
margin at the reset rows does not hold it against bc 0.05 at all.  Critic
replication noise (CFold2 / O2 retrain the round-1 critics on the same data,
seed and stream; lineage 1 bit-identical, 0 / 2 differ by GPU kernels):
~0.5-0.9 nats at the reset rows.  Assessment: policy iteration in this form
(same anchors, logged query torque, one branch per anchor) neither adds
route coverage nor strengthens the critic margin; it is not the in-boundary
lever it was proposed as.  Nothing selected after the evaluation.

### Corrected attribution: real training rows, training replay, full-state resumption (2026-09-19, night)

User's review rejected three attributions: 9b used the task goal (the actor
trains on relabeled goals); 9c used one SGD step on the final checkpoint;
round 2 restarted the critic and Adam, so "regenerated futures are worse"
was confounded with the restart (CFold2 lost capability with unchanged
futures).  Three checks (`scripts/diag_v6_actor_objective_stream.py`,
`scripts/diag_v6_training_replay.py`; `diag_replay/SUMMARY.md`): (1a) at
real actor-stream start rows with their relabeled goals the critic term does
not prefer the teacher detour torques (max_j f - E_pi[f] -0.27 / -0.27 /
-0.54, P 0.34-0.42; vs logged P 0.36-0.44; hazard-corridor goals, 43 % of
rows, -0.4 to -0.8) and the full objective is worse toward them in 96-98 %
of rows (+1.7 to +1.9); task goal, same rows: +0.2 and +1.3 worse.  The
"exact tie" is withdrawn.  (1b) replay of the round-1 CF runs from init.pkl
with the saved state and the streams' own sequence (fast-forward
self-checked), a checkpoint every 1,000 updates, probed by a 100-step route
probe (calibrated: the original finals give 0.127 / 0.18 / 0.57 vs
evaluation detour 0.107 / 0.167 / 0.50), an entrance-continuation walking
probe (45 / 60 / 60 states) and the reset-row critic margin: the reset
torques leave the start policy within 1,000 updates (|delta| 2.5-3.0) long
before any route change; walking collapses to 0.00-0.07 at 5k-11k / 3k-7k /
7k-10k with NO route change and recovers by 12k-20k; the critic margin is
established at 3k / 11k / 12k and the route turns 0-11k updates later (s2
at 23k-24k, s0 spikes at 12k-13k then falls back, s1's replay never turns);
consecutive checkpoints differ by 0.2-0.3 in route share -- the 30k
evaluations are single draws from an unconverged process whose replicate
spread (0.05 vs 0.18, 0.30 vs 0.57) equals the seed spread.  (2) full-state
resumption from CF1 final (actor, critic, target, both Adam states, key;
same futures; +30k): success 0.163 / 0.500 / 0.140, deaths 0.03 / 0.02 /
0.00, timeouts 0.80 / 0.48 / 0.86, detour 0.14 / 0.78 / 0.28, no route
0.71 / 0.12 / 0.68 -- the restart in round 2 froze a still-moving process;
continuing it drives deaths to zero and either raises the route share
where walking survives (s1) or removes walking (s0 / s2).  Withdrawn: the
tie, the interference reading, "equilibrium 0.12-0.21 / seed-2 0.50 a
transient".  Not established: what drives the walking collapses; whether
the objective pushes toward the actor's own route-changing torques.
Mainline unchanged; no BC sweep, no round 3.

### Training spikes: the walking collapses are critic runaways (2026-09-19, night, user's lead)

User found isolated spikes in every training log (CF s0 @4,500: critic
loss 0.133, mean logit -136; s1 @2,500; s2 @6,500; O s1 @8,000) right
before the walking probe falls to zero.  Verified: the values are
bit-identical across runs sharing the critic stream (bc0 variant, CFold2,
replays) -- a critic-batch-sequence event, actor-independent; 0.133 = 136 /
1024 means every logit of the batch collapsed to one value.
`scripts/diag_v6_spike_trace.py` (trace / rows / report;
`diag_replay/spike/SUMMARY.md`): from the replay checkpoints before the
spike, the pilot's update_step one update at a time with per-update
records (training metrics, critic and actor Adam step norms, a FIXED batch's
logits / |phi| / |psi|, the reset rows, a fixed actor batch, batch content)
and probes every 50 updates.  Seeds 0 / 1 reproduce (4,407 / 2,381); seed
2 does not (knife-edge).  Order: one critic gradient impulse (0.9 / 2.5 vs
median 0.02 / 0.03) on an ORDINARY batch -> Adam keeps x10-15 steps for
10+ updates -> all logits shift down together on the fixed batch and the
reset rows, |phi|, |psi| grow x2-4 -> runaway to logits -900 / -2,400 ->
actor q-term explodes and 12-20 updates later the actor's scale (0.06 ->
4) and saturation (0.04 -> 0.6-0.8) blow up -> walking probe 0.00 for the
rest of the window; the critic recovers within ~500 updates with
representation norms x2, the actor does not (scale 0.36 / 1.54 at the
window end).  Per-row attribution: the impulse is spread over many rows
(top-5 share 35-59 %), largest at goal-area anchors with goal-area goals
(near-duplicate pairs the binary NCE must separate); no outlier input.
FROZEN-CRITIC CONTROL (same batches, same actor Adam state, critic restored
after every update): no collapse at all -- walking 0.36-0.73 throughout,
scale <= 0.10, even seed 1's pre-spike drift is gone.  Decision rule's
first branch: the critic update and its signal to the actor.  Shared
learner property (both arms; O s1 / O s2 too), not the futures; the pilot's
30k actors went through different numbers of such events.  No remedy run
(would be a stability safeguard applied to both arms; user's call).

### Critic gradient clip (2026-09-19, night): first remedy, both arms -- the mainline rule met

User's decision after the spike trace: stabilise the critic update first,
one change only -- `optax.clip_by_global_norm(0.1)` on the critic gradient
before Adam (Optax chain; Adam state kept), NCE / actor objective / bc 0.05
unchanged; threshold from the traced gradient norms (median 0.017 / 0.033,
p99 ~0.06 / 0.10, impulses 0.92 / 2.53).  Implemented as the pre-registered
variant `critic_clip0.1` (`exp_v6_mainline_pilot.critic_optimizer`;
`diag_v6_training_replay.build(critic_clip)`, `diag_v6_spike_trace.py trace
--critic-clip` with the checkpoint's Adam moments carried).  Short windows
first (seed 0 from 4,000, seed 1 from 2,000; same batches / RNG): no
runaway (fixed logits min -7 vs -929 / -2,422), the actual Adam step at the
normal maximum (0.077 / 0.049), clip active in 1-3 % of updates, critic
still learning, actor scale <= 0.11, walking kept (seed 0 0.42-0.76; seed 1
no collapse but the slower pre-spike drift 0.65 -> 0.13-0.33 remains and
recovers by 2,750).  Full paired pilot with the clip in both arms
(`variants/critic_clip0.1/`): no spike in any of the six runs; O clip =
O base = start (0.257 / 0.247 / 0.260); CF clip 0.457 / 0.513 / 0.440,
detour 0.57 / 0.52 / 0.36, death 0.19 / 0.31 / 0.40, timeout 0.35 / 0.18 /
0.16, hazard success 0.39 / 0.43 / 0.31; CF - O success +0.216 (seed s.e.
0.026, boot 0.023, 3/3) -- MET; detour +0.46, death -0.43, timeout +0.21
(3/3); CF clip - CF base +0.19 / +0.19 / -0.04; continuation from the same
entrance handover states 0.75 / 0.75 / 0.67 (>= start).  Contract
amendment written (section 8).  Remaining limit -- CORRECTED 2026-09-19
(user's per-episode recount; "completed 40-70 % of the time" was wrong):
far route taken in 170 / 155 / 108 episodes, completed in 111 / 122 / 87
(0.653 / 0.787 / 0.806, pooled 0.739; no far-route deaths, every loss a
timeout); the other losses are shortcut deaths (52 / 89 / 113) and
no-route timeouts (38 / 19 / 18).  Rescuing every far-route timeout would
give 0.653 / 0.623 / 0.510 (bookkeeping), so the route share (0.57 /
0.52 / 0.36) matters as much as the far-route walking; "no-hazard success
0.68-0.84 vs start 0.99" is not a same-route comparison.  Status: oracle
evidence under a disclosed optimizer change.

### Confirmation draw, far-route audit, learned ETT launched (2026-09-19, evening)

User's plan after the clip result: (1) seal and confirm the current recipe
on unused evaluation seeds; (2) turn to the learned ETT -- data / interface
first, a model trained on the simulator branches allowed as an engineering
intermediate experiment if labelled as oracle-label-trained; (3) one small
audit of the clipped policies' far-route timeouts.  Done: (1) `confirm`
mode of exp_v6_mainline_pilot.py (sealed hashes, seed 4909, once):
REPRODUCED -- CF - O +0.192 (seed s.e. 0.015, 3/3), CF - start +0.192,
O = start (`variants/critic_clip0.1/confirm_s4909/`); the route ledger
(`route_ledger`) now in every report.  (3) `diag_v6_detour_audit.py`: too
late 1 / 3 / 4; no entrance-state signature; stalls at the corner (seed 0)
and falls in the east column (seed 1); same-state same-time continuation:
clipped CF 0.66 = start 0.63 = O clip 0.65 from the timeout entrances,
0.71-0.75 from the success entrances, driver 0.76 / 0.93 -> no
update-caused execution regression; the far-route walking limit is the
inherited one; CPU replays reproduce only 48-112 / 300 eval episodes
(rates agree), so per-episode outcomes are chaotic.  (2)
`exp_v6_learned_ett.py`: an ORACLE-LABEL-TRAINED learned ETT -- the
discounted-future-goal marginal p_gamma(g | s, a) of the blind
continuation as a categorical over 0.5-cells (1,508), heads for the
absorbing position at m = 10 / 50 / 200 and the branch outcome, MLP
1024-1024, cross-fitted over 3 episode folds (seed 161000001; the model
that writes an anchor's future never saw its episode), a position-only
reference model, checks (held-out NLL, region-mass L1, death AUC, zone /
far-route masses, horizon heads) -> `futures_learned.npz`; arm CFL =
`LearnedFutures` in the sealed clip recipe (the stream's future uniform
selects the cell; anchor sequence unchanged), 3 seeds, evaluated on the
development draw against O clip / CF clip; rules sealed in
`learned_ett/manifest.json` (primary CFL - O; kept = CFL - CF).  Launched
on node 30016 (fits) with the arm chain queued from 30043.

### Clip round, step A (2026-09-19, night): the current agents' futures at the same anchors -- gate for B NOT PASSED

User's lead after their frozen-model diagnostic (own sampled first steps
continued by the current agent enter the far route 19 / 128 vs 1 / 128 by
the start agent): do the current agents' futures differ from the start
agent's at the SAME training anchors?  `exp_v6_clip_round.py`: every
anchor regenerated per lineage (clipped CF final), same logged torque,
same hazard seed; paired anchor-weighted profile (`clip_round/PROFILE.md`,
`SUMMARY.md`).  Entered-far after the logged torque: reset anchors 0.013 ->
0.035-0.066, start region 0.032 -> 0.040-0.047, logged-shortcut anchors
0.000 -> 0.001; completed-far at the start region +0.002 / +0.010 /
+0.009 (gate: 3/3 > 0 and mean >= 0.02 -> NOT PASSED; B not run).  What
entered instead: worse shortcut walking (success 0.62 -> 0.53-0.58,
timeouts 0.09 -> 0.14-0.18, goal-area mass of the critic marginal 0.273 ->
0.21; between the zones success 0.64 -> 0.45-0.48).  Reading: the logged
query torque commits the continuation to the shortcut basin for the
current agents too; the agent's own-first-step gain can only enter through
counterfactual queries (a != a_logged), a contract dimension the user
would have to re-open.  C1 (`diag_v6_candidate_at_origin.py`, the user's
verified candidates at their original state) and C2
(`diag_v6_actor_objective_stream.py --variant critic_clip0.1 --candidates`,
transplants at real training rows) running.

### Arm CF-learned (oracle-label-trained learned ETT) -- NEGATIVE (2026-09-19, night)

`exp_v6_learned_ett.py` (`learned_ett/SUMMARY.md`).  Model checks: the
state + torque marginal model generalises worse than a position-only
reference (held-out NLL 3.678 vs 3.630; train / val curves diverge --
overfitting; mechanism open); aggregate masses and the start-region
action contrast (far mass 0.49 vs 0.002 for detour vs shortcut torques;
oracle 0.61 vs 0.00) survive, the reset-row contrast does not (0.09 vs
oracle 0.57, n = 4).  The arm under the sealed clip recipe: CFL = O =
start (0.253 / 0.247 / 0.247; far route 3 / 4 / 0 of 300); CFL - O -0.006
NOT MET; CFL - CF(oracle) -0.221 (3/3): the oracle gain is lost entirely;
in-batch accuracy under the learned positives 0.008-0.016 (blurrier
positives).  Causes not separated (blur / reset rows / poorer
conditional); next diagnostic = the CFL critics' reset-state readout vs
the oracle-CF critics.  Structural lesson for the learned-ETT line: a
per-anchor marginal fitted on one branch per anchor loses the effect of
the first action where it matters; the user's separation (ETT = a fixed
transition model, continuation agent free) points to one-step dynamics +
a hazard / death model rolled out with the agent.

### C1 / C2 on the clipped models (2026-09-19, night)

`diag_traj/SUMMARY.md` section 12.  C1 (the user's verified candidates at
their original state, task goal): the same-scale moved expectation beats
the current E_f in 20 / 23 stable-shortcut cases and the local gradient
points at the candidate in 13 / 23, but the BC pull (nearest dataset reset
rows, teacher torques) outweighs it -- objective proxy better in 5 / 23 --
and at those nearest rows with their real relabeled goals the critic term
prefers the current policy in 84-97 %.  C2 (transplants at 3.7-3.9k real
start-region rows): real-goal signal +0.49 / -0.02 / +0.13, task-goal
signal +1.08 / +0.76 / +0.47; full objective P(better) 0.00-0.02 held,
0.08-0.13 widened.  Reading: not adopted because the training objective
(relabeled goals on logged paths + BC at a narrow scale) does not support
the move -- the objective at the training distribution, not an optimiser
failure; policy extraction alone is not the lever.

### Identity correction: the AntMaze "learned ETT" is a future-goal marginal model (2026-09-19, user's review)

The user (with ChatGPT's reading of CHECK.md and the code) pointed out
that `exp_v6_learned_ett.py` does not implement the project's ETT.  The
PointMaze pipeline's ETT (`feature/pointmaze-causal-transition`) is a
one-step model F(s, a_b, a_q) -> s' plus an onset head, fitted on
same-context tuples (s, a_b, a_q, s') -- diagonal rows from the log,
off-diagonal rows from same-context interventions -- and rolled out step
by step with a nominal model of the expert advice (a_b ~ P(a_b | s), an
MDN on the log) and the current agent choosing a_q, absorbing after
onset; the replay is 50 % complete model trajectories + 50 % original
episodes, then the unchanged CRL.  The AntMaze model is a per-anchor
classifier of the discounted future-goal cell under the fixed start agent
(no a_b, no diagonal / off-diagonal structure, no next state, no rollout;
the five heads are summed and only the marginal is used).  Its negative
arm (CFL) says nothing about the one-step ETT.  I should have read the
PointMaze implementation before designing it.  Alignment audit for
AntMaze (next, no training): the d05 log is all-diagonal (the teacher is
the behaviour policy: a_b = a_q on every row); the oracle branches redraw
the hidden hazards, so the logged torque is a_q under a NEW context whose
expert advice a_b is not recorded -- it is computable in the simulator
(the privileged teacher relay at the restored context: `teacher.fresh(route)`,
`teacher.act(o, env.schedule, intent)`, as the blind driver does), which
makes same-context tuples available as an engineering intermediate (the
PointMaze supervision was also native-simulator paired probes); the
off-diagonal rows arise where the redrawn schedule makes the teacher wait
while the logged torque goes (the confounder's footprint) and from query
coverage at fork contexts (route alternatives), which the sampling
contract currently excludes.  One-step Ant dynamics are deterministic
given the 29-dim state and the torque (restore_maxdiff ~ 0); the
stochastic part is the onset (rock contact), which is where a_b vs a_q
carries the hidden state.

### Simulator-paired one-step ETT, Step 0 (2026-09-20): same-context advice reconstructable; motion deterministic

User's go (Steps 0-2; fork query coverage closed; mainline replay interface
kept; BC 0.05 / losses / clip unchanged; oracle-supervised engineering
stage).  `audit_v6_ett_context.py`, `ett_context/SUMMARY.md`.  Hidden
context + intent for all d05 episodes (sidecars 606/610/611; 607-609
regenerated from the collector's fixed-order RNG streams, validated
exactly on the available seeds).  The teacher replayed along the log
reproduces 267,481 / 267,481 torques bit-exactly, holding zones and
decisions too.  a_b at the branch contexts = teacher state reconstructed
from the logged history + zone latches re-derived by the teacher's rule
at the recorded consultation steps under the redrawn timetable (never
from the query or the realised route); original timetable -> a_b == a_q
at 53,747 / 53,747; redrawn -> diagonal 0.83, off_hold 0.09, off_release
0.08 (weighted), off-diagonal rows at the mouths / bands with branch
death 0.65 (off_hold) vs 0.27 (diagonal).  One-step motion: restore +
logged torque reproduces the logged next state to ~1e-6 (1 outlier of
2,236); branch first row vs logged next (different hidden) within 1e-5 in
53,744 / 53,747 (3 angular-velocity outliers 0.04-0.16).  Onset labels
only in the branches (15,612 deaths).  Next: Step 1 supervision builder,
Step 2 model + preflight.

### One-step ETT, Step 1 (supervision) and Step 2 launched (2026-09-20)

`build_v6_ett_supervision.py` -> `ett_context/supervision.npz`: 7.65 M
valid branch transitions with the advice walked along each branch (root
advice = Step 0's, 53,747 / 53,747 exact), 267k logged diagonal rows;
onsets 15,612, 15,600 of them on off_hold rows, 0 on diagonal rows;
folds by source episode.  `fit_v6_ett_one_step.py` (sealed manifest,
gates G1-G6 pre-registered): motion MLP 1024-1024 -> standardised delta
(MSE, diagonal / off-diagonal averaged separately), onset MLP 256-256 on
the features + predicted delta (natural-prevalence BCE, stratified
batches, no gradient to motion), Adam 3e-4, 40k updates, early stop on
the ES slice; cross-fitted 3 folds; smoke at 300 steps: val MSE 0.096,
AUROC 0.94.  Fits running: folds 0 / 1 on node 30125, fold 2 on 30016;
`check` (held-out fold rows: one-step xy error, 50-step open loop, onset
AUROC / ratio / calibration / advice sensitivity) runs on 30125 when the
three models exist.

### One-step ETT, Step 2 result (2026-09-20): all preflight gates passed

`ett_one_step/SUMMARY.md`, `CHECK.md`.  Held-out (cross-fitted) one-step
xy error median 0.0016 / p99 0.027, delta RMSE 0.157; open loop 10 / 50
steps 0.060 / 0.47; onset AUROC 0.996, on-onset vs non 139 x,
calibration 1.05, in-band P(onset) 0.21-0.24 under off_hold vs 0.000
under a driving advice.  Best steps 39-40k (budget end).  Open before
Step 3: (a) long-horizon open-loop validation (outcome / route agreement
of full-length rollouts vs held-out branches); (b) the ADVICE PROCESS
along a rollout -- memoryless nominal P(a_b|s) cannot produce "hold"
inside a band, a latched nominal (mouth decision + visible hold
durations) can, the simulator teacher can (oracle); the deaths in the
generated futures follow from that choice.  Nothing downstream run.

### One-step ETT, Step 3a (2026-09-20): full-length rollouts vs held-out branches -- C fails on stalls and death timing

`diag_v6_ett_rollout.py`, `ett_rollout/SUMMARY.md`.  User's design:
distributional comparison (rates, KS of death time / death x / reach
time), exact hazard integration, C (simulator teacher along the model
path) vs A (memoryless nominal MDN on the log).  C: death 0.294 / 0.308,
far equal, KS pooled fine, AUROC 0.99, but timeout 0.091 -> 0.021 and
reach +0.056 (the sim's stalled ants -- exactly stationary, 39 % of rows
-- drift in the model (0.54 in 100 steps open loop) until the policy
re-engages; 89 % of sim timeouts reach the goal in the model) and in-band
death timing front-loaded (zone 2: 65 % of death mass in steps 1-5 vs sim
none; zone 1 median 53 steps) -- a memoryless hazard cannot express the
delay.  A: death 0.294 -> 0.023 (memoryless advice never holds in a
band).  Revision v2 (disclosed): exact zero-displacement atom with a
stationary gate; onset head + visible-history features (steps since the
advice turned hold, steps since band entry).

### Review of the A / B / C round (user, 2026-09-20): corrections and the real-update check

Corrections written into clip_round/SUMMARY.md and diag_traj/SUMMARY.md
section 12: (A) the omitted positive -- logged-detour start rows complete
the far route 0.529 -> 0.54 / 0.69 / 0.67 (0.4 % of the weight): the
improvement depends on the handed state and first action; generation at
the logged torques does not cover it; B not run, so no claim that those
futures train worse.  (C) C1's BC term is a proxy, C2's candidates are
transplants, "5 / 23" was the endpoint only (9 / 23 improve somewhere on
the path); a training row's shortcut goal is a different task from the
task-goal decision; NOT shown that the actor optimised its objective
correctly or that extraction has no room.  CFL archived (not the ETT; no
more diagnostics).  Mainline: the real ETT continues (v2 rollout check
running); the oracle futures / checkpoints stay the reference; A's files
do not enter the main comparison.  Added actor check = the REAL-UPDATE
check (real batches, real Adam state, on a checkpoint copy): does the
actual update move the verified states' actions toward the verified
candidates; critic-term vs BC-term decomposition; scale watched.  Query
extension stays a candidate to be justified by that evidence (a next
sampling design: the current policy's one-step torque queries, state
weights / actor data / loss / BC 0.05 unchanged).

### One-step ETT, Step 3a v2 result and the v3 revision (2026-09-20)

v2 (atom + stationary gate + visible-history features; preflight G1-G7
all passed): pooled C passes every sealed gate (death 0.294 sim / 0.311
model; KS death time 0.027, death x 0.072), but strata do not: pre_zone1
timeouts 0.178 -> 0.049, post_zone2 timeouts 0.042 -> 0.089, zone2 death
time KS 0.22, death-x KS 0.157-0.167.  Per fold: folds 0 / 2 produce
almost no pre-mouth stalls (timeouts 0.006 / 0.021 vs 0.197 / 0.193),
fold 1 (early-stopped at 4k on the stationary BCE) over-stalls (0.153;
post_zone2 0.236).  Settling probe on the sim's stalled tails: after the
torque becomes small the sim decelerates slowly (|delta s| 0.08 -> 0.02
at 10 steps -> 0.006 at 40), the v2 regression keeps |delta s| ~0.15, the
gate never fires, and the policy re-accelerates after ~40 steps.  Cause:
the standardised MSE is dominated by large-displacement rows.  v3
(disclosed revision; `fit_v6_ett_one_step_v2.py --v3`, outputs
`ett_one_step_v3/`, `ett_rollout_v3/`): regression on all rows plus a
relative error term ||pred - d||^2 / (||d||^2 + 0.05^2); selection score
without the stationary BCE; everything else unchanged.  v3 preflight: all
gates passed on all three folds (one-step xy median 0.0022, 50-step 0.48,
onset AUROC 0.997, gate AUROC 0.997; folds stop at 39-40k, no early
stop).  Full-length C / A rollouts with the v3 models running (node6 went
down at launch -- likely memory; re-queued on node3 behind the v2 A
rollouts).  Judgement rule unchanged (user): C bad -> fix motion / onset
first; C good and A bad -> the advice process is the bottleneck.

### Real-update check result, corrected after the user's review (2026-09-20; diag_traj/SUMMARY.md section 13)

Default bc 0.05, 2,000 real updates from each clipped CF checkpoint with
the real batches and Adam state, 11 / 11 / 9 verified stable-shortcut
contexts.  Weighted projections on the candidate direction (means over
68 probes): critic term +0.128 / +0.189 / +0.138 (positive in ~100 % of
probes), BC term -0.105 / -0.180 / -0.160 -- BC cancels 82 / 95 / 116 %
of the critic component; total +0.024 / +0.009 / -0.022; realised Adam
step ~0; the modes end where they started.  The first reading ("BC
cosine -0.01, so weak cancellation; the critic landscape is not an
attractor -> query coverage first") was WRONG and is withdrawn: cosines
compare directions, the weighted projections show a local
critic-toward / BC-against cancellation under the original recipe.  The
bc 0 / bc 1 replays (scale collapse; mode to other critic maxima; BC-only
moves away) are ablations of a changed objective and cannot overturn
this: the critic keeps updating, a near-zero scale is not saturation
evidence, the candidates were verified once for entry only.  BC stays
0.05.  Separately, the AntMaze generation gives the critic no
supervision at the policy's alternative actions (one path per anchor,
first torque = the logged one; PointMaze kept ~7,700 extra query
branches) -- a real coverage difference, not yet shown to be the cause.
Next (user's plan): (1) a small generation comparison, logged-query vs
extended-query futures (queries = logged torque + current policy mode +
a fixed number of policy samples; same training start / early-decision
states, same continuation policy, goal and paired hazard draws; all
outcomes kept; judged on far-route COMPLETION and task-goal future
quality, not entry); (2) only with a reliable contrast, a training
comparison where the new (s, a_query) pairs become critic anchors
(per-state total weight, budget, actor / BC data, loss, BC 0.05
unchanged; same continuation policy in both arms).  No more BC
ablations or gradient checks.  The one-step ETT continues separately;
first check that it preserves the oracle gain.

### Query coverage, stage 1 (generation comparison), sealed 2026-09-20 before generation (`scripts/exp_v6_query_coverage.py`, `query_coverage/`)

User's plan after the real-update review.  States: every pilot anchor in
the start region (4,276 anchors, weight 0.079, t median 10, 99 % t <= 30)
-- the starts and early decision states; nothing chosen from evaluation.
Current policy = the lineage agent of seed s (clipped CF final).  Queries
per anchor: q0 the logged torque, q1 the agent mode, q2..q5 four pinned
samples tanh(loc + scale * eps).  Every query executed once from the
restored logged state and continued by the SAME agent (mode) to reach /
death / horizon, logged task goal, two PAIRED hazard draws per anchor
(draw 0 = the mainline seed HAZARD_SEED0 + anchor_id, draw 1 = a fresh
offset); nothing filtered.  Read-out: anchor-weighted far-route entry /
COMPLETION, success / death / timeout, length, critic goal-marginal mass
(far / goal / zones) for logged (q0) vs extended union (q0..q5, weight
split equally) vs added (q1..q5), paired-by-anchor bootstrap; draw-to-draw
agreement; strata (reset rows, t in [1, 30), logged shortcut / detour).
Pre-registered gate for stage 2 (every lineage): G1 completed-far gain
extended - logged >= +0.03 with bootstrap 95 % CI lower bound > 0; G2
goal-area mass extended >= 0.8 x logged; G3 >= 100 distinct start anchors
with an added far-complete future in BOTH draws.  Stage 2 (only if the
gate passes, on the user's go): the new (s, a_query) pairs become critic
anchors -- per-state total weight, budget, actor / BC data, losses, clip
0.1, BC 0.05 unchanged -- logged-query vs extended-query futures under
the SAME continuation policy.  Running on node5 (3 lineages x 4,276 x 6
x 2 = 51,312 branches each).

Lineage 0 result (interim, lineages 1 / 2 generating): pooled over the
start anchors completed-far logged 0.026 -> extended 0.041 (mode 0.046,
samples 0.044), paired delta +0.0154 CI [+0.012, +0.019] -> G1 (>= +0.03)
NOT met; G2 (goal mass 0.017 / 0.016) and G3 (377 / 378 distinct anchors)
pass.  Strata: the whole effect sits at the RESET rows (t = 0; 196
anchors, weight 0.0038 = 0.4 % of the anchor weight): 0.037 -> 0.330
(mode 0.423), delta +0.29 CI [+0.25, +0.33]; t in [1, 30): 0.020 ->
0.022 (CI spans 0); logged detour episodes unchanged (0.53).  Draw-to-
draw agreement of the far-complete indicator 0.995 (Jaccard 0.88).
Reading: the agent's own first torque changes the route only at the
reset step (its 42 % matches the agent's evaluation far-route share); one
step into a shortcut episode neither the logged nor the agent's action
leads to the far route.  The missing critic supervision therefore has a
precise locus -- the reset rows -- with 0.4 % of the anchor weight; the
pre-registered pooled gate fails on that weight.  No stage 2 without the
user's call.

Three-lineage result (query_coverage/SUMMARY.md): G1 NOT met in all
three (pooled completed-far +0.015 / +0.016 / +0.010, CIs above 0); G2,
G3 pass.  Reset rows: 0.037 -> 0.330 / 0.031 -> 0.301 / 0.026 -> 0.211
(mode 0.42 / 0.39 / 0.30), success 0.20 -> 0.43 / 0.18 -> 0.40 / 0.21 ->
0.37; t in [1, 30) ~0.  Where the mode's branch fails at a reset state,
a sample completes the far route in 70 / 67 / 40 of 110 / 122 / 134
cases (draw 0): the query set offers improvement beyond the mode.

### Query coverage, stage 2 (controlled training), user's decision 2026-09-20 (`scripts/exp_v6_query_train.py`, `query_coverage/train/`)

User's reading of stage 1: the effect is real and sits at the reset
rows (own first step: far completion 3.7 % -> 42 %, success 20 % -> 43 %
on the same states, draws and continuation); G1 averaged it away (reset
rows = 4.8 % of the start-region weight, so +29 points there is the whole
pooled +1.5); G1 stays "not met" but is not a judgement of training
value (a reset anchor is still drawn ~117k times in 30k x 1024); the
open question is whether these consequences, once in the critic,
improve the CURRENT policy -- so the comparison must keep the current
policy as a reference.  Design (fixed before training): per lineage,
control = start-region anchors get the q0 (logged-torque) futures of
the queries file (both draws), extended = q0..q5 futures with the
branch's own torque as the critic action; every other anchor = the
sealed mainline futures in both arms (same continuation policy in the
start region, same futures elsewhere); anchor weight split equally over
the allowed branches by a separate selection RNG (main stream RNG
untouched -> identical anchor sequence and actor batches across arms,
first-batch hashes); start = the lineage final with its full
TrainingState, streams fast-forwarded, +30,000 updates; NCE, actor
objective, BC 0.05, clip 0.1, actor / BC stream unchanged; no reset
re-weighting.  Evaluation once on a fresh paired draw (seed 6909, 300,
mode) for control / extended / current (+ start agent as reference);
paired extended - control, extended - current, control - current under
the mainline rule.  Read-outs (in-sample, secondary): the critic's
within-anchor ranking of the (anchor, query) branches by their realised
success (lineage vs final critics; reset rows and all start anchors),
and each final actor's own rollout from the 196 reset anchors with the
two draws.  Judgement fixed in advance: (1) extended > control and >
current -> usable improvement, confirm; (2) critic learns, actor does
not -> stop adding queries, extraction / objective conflict; (3) critic
does not learn the added consequences -> value learning.  Stage-1 report
gains the user's read-out: among reset anchors where the mode's branch
fails, how many have another query that completes.  Chain queued on
node5 behind the stage-1 generation.

### One-step ETT, Step 3a v3 result (2026-09-20; ett_rollout_v3/SUMMARY.md)

C pooled passes (death 0.294 / 0.305, KS death time 0.036, death x
0.066); start / post_zone2 / goal_area strata pass; pre_zone1 timeouts
0.178 -> 0.053 (the three folds now agree, 0.04-0.06, vs v2's 0.006 /
0.117 / 0.021), zone1 / zone2 death-time KS 0.16 / 0.32, between / zone2
death-x KS 0.17 / 0.21 fail.  A fails as before (death 0.29 -> 0.04).
Rule: C bad -> motion / onset first; the model still under-stalls ~3x and
the in-zone death timing is off.  No training with these futures.

### Query coverage, stage 2 result (2026-09-20 night; query_coverage/train/SUMMARY.md)

Judgement 1 NOT met: extended - current -0.25 / -0.20 / -0.25 (3 / 3
worse); extended - control -0.23 / +0.18 / +0.15.  The reference arm
`continue` (same resume, unchanged sealed futures; added after two
lineages were seen) loses 0.00 / 0.13 / 0.28 with the stall signature
(deaths -> 0, timeouts 0.48-0.85) -- as the round-1 resumption did: the
process is not stable past 30k even with the clip.  Control (the
lineage's own q0 continuation at the start anchors) stalls far worse
(0.05 / 0.03).  Extended does the opposite: shortcut and death (far share
0.12 / 0.30 / 0.06, deaths 0.48 / 0.14 / 0.32); in-sample, from the 196
reset anchors its own rollout completes the far route 0.05 / 0.09 / 0.02
vs the current 0.41 / 0.40 / 0.29, and its mode moved TOWARD the logged
torque (distance 2.8 -> 1.8, 1.9 -> 1.3, 1.8 -> 1.7).  The extended
critic ranks the queries by realised success a little better (reset-row
AUROC 0.62 / 0.67 / 0.65 vs current 0.62 / 0.58 / 0.62) but only to
~0.65 and with no larger task-goal margin for the best query.  Closest
pre-fixed branch: judgement 2 -- critic learns (modestly), actor does
not improve, and it was pushed to the shortcut; consistent (untested)
mechanism: the actor's relabeled shortcut-corridor goals turn a more
accurate critic into a push toward the logged action.  Per the user's
rule: stop adding queries; the remaining problem is extraction / the
objective (actor goal distribution, BC), on top of a training process
that does not hold past 30k.

### Re-ordering after stage 2 (user, 2026-09-21)

What stage 2 negated: "extend the queries and train one more round ->
keep improving".  NOT negated: the oracle branch-replay gain (~47 % vs
~25 %).  No longer acceptable: attributing everything to insufficient
query coverage.  The most useful finding: with the data unchanged,
continued training destroys capability (continue s1 0.437 -> 0.307,
timeouts 0.21 -> 0.65; s2 0.433 -> 0.150, timeouts 0.14 -> 0.85), so
"better training data" and "a better policy after more training" have
no reliable link right now; but the extended arm's detour share is
below continue's in all three lineages, so the query result is not only
"more training".  Two phenomena to keep apart: continued optimisation
degrades walking; changed query supervision degrades route choice.  The
query experiment did find real information (a one-step 8-dim torque can
change the route and the system can find such torques) but did NOT show
the critic evaluates them reliably on new states (the critic read-out is
in-sample, limited candidates, "success in >= 1 draw" labels against
task-goal scores) -- no "critic fixed, actor disobeys"; 0.4 % reset
weight shows concentration, not proven insufficiency.  Design difference
found vs the successful PointMaze run (run_f4_joint_warm.py): there the
ACTOR's critic term also used branch-generated future goals (BC on the
logs, balanced); the AntMaze contract keeps the actor's critic term and
BC on the logged relabeled goals -- the critic learns new futures while
the actor practises reaching the old logs' futures (a shortcut-corridor
goal row asks for the shortcut).  A sampling-level change worth testing,
not a found cause; counterfactual futures also contain stall positions.
Next, in order: (1) FROZEN-CRITIC continuation from the three 30k
finals -- critic frozen, actor + its Adam + batches + BC 0.05 unchanged,
+30k, vs `continue` (reading: walking kept -> the moving critic drives
the degradation, handle the training schedule; still degrades ->
optimising along a fixed critic and the current actor distribution hurts
deployment, then check the actor's inputs); (2) only then the sampling
revision: actor critic term on logged vs counterfactual future goals at
the same states / weights, BC unchanged (query torques are NOT expert
actions), no reset re-weighting / BC balancing / network changes.  ETT
(parallel): fix the history-counter offset (training run_length starts
at 0 at the first hold / band row, the rollout passed 1); the key gap is
the ADVICE process -- the memoryless nominal gives death 0.036 vs the
teacher's 0.305 because the model reaches contexts the logs rarely cover
-- so the focus moves to training / generation history consistency and
how the nominal handles persistent hidden contexts, not another motion
regression.  Position: counterfactual sampling has a reproducible oracle
gain; the gain cannot yet be enlarged by more training; the full learned
ETT lacks a reliable advice substitute.  "More queries + one more round"
is no longer the default improvement.  Launched: arm 'frozen' (3
lineages, node5; report3) and the v3 rollouts with --hist-fix (C then A,
node3; ett_rollout_v3/hist_fix/).

### Frozen-critic continuation result (2026-09-21; query_coverage/train/SUMMARY.md) and the cheap checks

Frozen (actor only, +30k) vs continue (joint) vs current: success 0.497
/ 0.517 / 0.520 (s0), 0.500 / 0.307 / 0.437 (s1), 0.140 / 0.150 / 0.433
(s2).  Common signature of every frozen run: deaths -> 0, timeouts +0.21
/ +0.28 / +0.72; far share 0.71 / 0.86 / 0.11.  Reading: lineage 1 -> the
moving critic drove the collapse; lineage 2 -> the fixed-critic actor
objective stalls the policy by itself; both branches of the user's tree
are live; next = the actor's inputs (step 2, goal-source comparison,
running); critic-schedule comparison a later option.  Cheap checks
(cheap_checks/SUMMARY.md): (a) the critic's positive law puts 0.3-0.4 %
of a success future within the reach radius, 7 % of a death future on
the death position, > 50 % of a timeout future on stall rows; logged vs
own first torque give identical task-goal mass (0.0006) -- "reach more"
is not "more task-goal positives"; the termination rule is worth a
later, evidence-gated study.  (b) 'clip' vs 'acme' BC log-prob: 1.6 %
boundary components, BC-gradient cosine >= 0.9986, relative difference
0.7-5 % -> not a lever; at bc 0.05 the weighted BC gradient is 4-17x the
weighted critic-term gradient in norm.

### Actor goal-source comparison result (2026-09-21; actor_goal_source/SUMMARY.md): closed, negative

Frozen critic, anchor-pool actor rows, critic-term goal recorded
(goal_log) vs the sealed CF branch (goal_cf), BC unchanged, +30k:
goal_cf 0.277 / 0.257 / 0.267 with far route 0.007 / 0 / 0 and deaths
0.72-0.73 = the START AGENT's profile; goal_cf - goal_log -0.13 /
-0.04 / -0.13 (3 / 3).  Mechanism: the CF branches are the start
agent's shortcut continuations, so their positions become the goals the
actor is asked to reach.  Hypothesis "changing the actor's goal source
alone repairs the recipe" CLOSED (user's rule); PointMaze's actor goals
came from branches whose continuation itself detoured -- not
transferable.  Disclosed: the anchor pool fits the logged actions more
tightly (scale 0.07 -> 0.035); goal_log vs frozen differ by the row law.

### ETT history-counter fix (2026-09-21; ett_rollout_v3/hist_fix/)

Confirmed in code: the training rows' run_length gives 0 at the first
hold / band row of a path, the rollout passed 1 (+1 on every hold / band
row).  `diag_v6_ett_rollout.py --hist-fix` aligns the rollout and writes
under hist_fix/.  v3 C per fold, before -> after: death 0.301 / 0.309 /
0.305 -> 0.299 / 0.304 / 0.301 (sim 0.29-0.30), reach 0.627 / 0.640 /
0.647 unchanged, pre_zone1 timeouts 0.057 / 0.041 / 0.062 -> 0.050 /
0.058 / 0.070 (sim 0.15-0.20), death-time KS 0.025 / 0.043 / 0.052 ->
0.028 / 0.029 / 0.034.  The offset was real but not the driver (as the
user said); the under-stalling before the mouth and the in-zone death
timing remain, and the advice-process gap (memoryless nominal death
0.036 vs the teacher's 0.30) is untouched by it.  A folds running for
the record.

### User's review of a408cef (2026-09-20) and the two parallel lines

Corrections adopted: the frozen-critic result rules out "critic fixed =>
actor stable" (seed 2 stalls with a fixed critic), the failure location
differs by seed; the goal-source hypothesis is closed but its mechanism
("start-agent shortcut futures teach the shortcut") is consistent, not
independently proven, and does not extend to "all generated goals are
useless"; the acme boundary rule leaves the queue; the larger BC gradient
norm does not by itself show BC blocks the detour, no basis to lower BC;
the goal-mass report's inference "many stall positives => the critic
rewards stalling for the task goal" was too strong (a stall position is
another goal; small absolute goal mass need not mean a wrong ranking).
Two independent questions from here on, kept apart: (CRL line) with
simulator futures, how does CRL learn a better policy reliably; (ETT
line) can a learned ETT produce those futures accurately enough.  No
continued training, no goal-mix sweeps, no boundary-rule experiments.

### CRL line: task-goal target support per candidate action (2026-09-20; target_support/SUMMARY.md) -- the termination rule is NOT a found cause

`scripts/diag_v6_target_support.py` on the stage-1 paired query futures
(no rollouts, no training; judgement sealed in the manifest).  At the
same state and task goal, per query: success and the probability that
the critic's positive (P(m) ~ 0.999^m truncated at the path's end) lands
at the task goal; the same under an ABSORBING alternative (terminal
position held to the remaining horizon after reach / death, timeouts
unchanged, no relabelling).  Reset rows: far-going queries succeed 0.71 /
0.74 / 0.84 vs short-going 0.17 / 0.14 / 0.17 and get 2.1-2.6x their
expected reach-0.5 mass under the CURRENT law (0.0011 vs 0.0005, CI > 0,
per-anchor sign agreement 0.86-0.90); rank concordance across the six
queries 0.94-0.95, top-1 0.88-0.92.  The absorbing law multiplies every
mass by ~180 (0.001 -> 0.2; marginal 0.019 -> 0.49) and leaves the
ordering, the ratio and the sign agreement unchanged (+-0.02).  The
success ratio (4-5x) compresses to 2-2.6x under BOTH laws because a far
success is ~2x longer (the discount's speed penalty: a far success
carries half a shortcut success's mass, P(far > short) 0.01-0.05 under
both).  J1 not met in any lineage -> the user's option 2: the target law
already ranks the good actions correctly at these states; the problem
stays in how the learner / policy update uses the target; hypothesis
closed, no O / CF training with a changed termination law.  Noted
outside the rule: under the current law a reset row yields one task-goal
positive per ~250 critic batches (~120 over a 30k run).

### ETT line (2026-09-20): exact history counters; the advice process located (ett_advice/SUMMARY.md)

(1) The user's code check: run_length counts 0 at a flagged path start
but 1 at a flagged row after an unflagged one; the uniform -1 of
`--hist-fix` fixes the first case and breaks the second (verified on
200 random paths: the -1 rule mismatches ~40 % of rows).  `--hist-exact`
replicates run_length row by row (last reset index = path start or the
last unflagged row); v3 C and A rerun with the fixed models under
ett_rollout_v3/hist_exact/ (node3).  (2) The advice process, measured
before designing anything (`scripts/diag_v6_ett_advice.py`): the real
advice holds 4.2 % of branch rows (8.8 % in band), P(hold | hold) 0.98,
runs median 14 / p90 22; 15,600 / 15,612 onsets on hold rows.  The
memoryless nominal, drawn i.i.d. along the same real states: hold share
0.27 but 0.000 in band (AUROC vs the real in-band hold 0.494; 0.981 on
its own logged rows), 0.72 in pre_zone1 -- it holds at the start agent's
STALLS (the log's holds are stationary poses at the mouths; no logged
row holds inside an active zone), so its persistence (0.96) is the
state persisting, not a hidden context; the NLL of the real advice on
the branch rows is 18.6 (27.7 beyond 50 steps) vs -11.4 for the log.
The v3 onset head needs a sustained in-band hold (realised onset 0 at
kh <= 3, 0.09 at 4-7, 0.22 at 8-15, 0.38 at 16-31; the model tracks it,
AUROC 0.75); drawing the nominal's own counter distribution on real hold
rows gives the same p_onset as the real counter (0.17-0.23 vs
0.18-0.22) -> the missing deaths (0.036 vs 0.30) are the absence of
in-band holds, not the counter.  Acceptance metrics for any advice
process written down (in-band hold AUROC and NLL on held-out branches;
P(hold | hold), run lengths and region shares matching the real advice;
the fixed onset head recovering the death rate and timing).  Candidate
for the user's decision: a latent-context advice model fitted on the
branch supervision rows (their redrawn timetables are recorded), z
sampled per path from its prior at generation; not started.

### ETT line: exact-counter recheck result (2026-09-20; ett_rollout_v3/hist_exact/SUMMARY.md)

C with the exact counters, per fold sim / model: death 0.292 / 0.296 /
0.293 vs 0.300 / 0.307 / 0.303, reach 0.611 / 0.619 / 0.614 vs 0.625 /
0.643 / 0.644, pre_zone1 timeouts 0.197 / 0.145 / 0.193 vs 0.052 / 0.046
/ 0.070, death-time KS 0.018 / 0.037 / 0.039.  Pooled strata: pre_zone1
timeout 0.056 vs 0.178 (raw 0.053, hist_fix 0.059), zone1 death-time KS
0.099 (0.156 / 0.104), zone2 death-time KS 0.241 (0.321 / 0.241),
death-x 0.200, between death-x 0.153.  Pooled gates pass; pre_zone1,
between, zone2 still fail -> long-path motion / onset errors, not the
counter.  A: death 0.036 vs 0.294, AUROC 0.687, unchanged -- the
nominal's missing in-band holds (ett_advice).  The learned ETT is not
running end to end; the CRL line's oracle gain is not a learned-ETT
result.  Nothing running; next = the advice process (user's call).

### User's decision after 0a25130 (2026-09-20): one bounded CRL comparison, the ETT advice generator

CRL: the absorbing termination law as a NEW sparse-supervision
hypothesis (the target_support ordering result stands; a reset row yields a
task-goal positive in 0.1 % of its draws); ETT: an advice generator on the
branch supervision with a persistent per-path hidden context (motion /
onset frozen), gated on held-out calibration and whole-trajectory
distributions before any rollout; no BC / actor / query-count / budget
changes; the two lines kept apart.

### CRL line: absorbing law result (2026-09-20; absorbing_law/SUMMARY.md) -- NOT supported

`scripts/exp_v6_absorbing_law.py` (+ `AbsorbingFutures` / `future_law`
override in the pilot): the current mainline (critic_clip0.1) with the
critic's positive law changed in both arms (terminal row held to 800 - t
after reach / death; timeouts unchanged; 76 % of the positive draws land on
a terminal row).  Seed 7909, same 300 episodes for the six new finals, the
six clip finals and the start: CF_abs 0.013 / 0.380 / 0.063 (deaths
0.00-0.02, timeouts 0.99 / 0.60 / 0.92; BC NLL -16 -> -8 / -11 / -12,
scale 0.06 -> 0.09-0.14), O_abs = start (0.26-0.27); CF_abs - O_abs
-0.257 / +0.117 / -0.200 (rule not met); CF_abs - CF_clip -0.50 / -0.09 /
-0.36 (3 / 3 worse); CF_clip - O_clip on the same draw +0.253 / +0.197 /
+0.140 (met); reset-row critic AUROC over the stage-1 candidates CF_abs
0.58 / 0.53 / 0.63 vs CF_clip 0.62 / 0.58 / 0.62 (up 1 / 3).  Reading:
under the absorbing law nearly every O positive is the goal (no contrast
between actions; the O critic teaches nothing) and the CF actor learns to
avoid the death spot rather than to reach (stalls); more task-goal
positives came bundled with death-position positives.  Stop tuning around
the sampling rule (user's rule).

### ETT line: the advice generator v1 -> v3 (2026-09-20; ett_advice_gen/SUMMARY.md)

v1 (single hold head, Bernoulli): teacher-forced AUROC 0.996 but
autoregressively the runs fragment (median 2 vs 14) and P(death) 0.14 vs
0.29; the calibrated p at the rows where a hold STARTS is 0.04.  v2
(start / continue heads, weighted, calibrated by inversion): starts 0.08.
v3 (+ per-zone mouth-arrival history: the teacher consults once at the
first arrival at x in [MOUTH_X, HAZARD_X), |y| < 2; hold iff wait-latch and
t < t0 + 72): starts 0.55; autoregressive under the true context the
generated advice agrees with the real advice row for row 0.993-0.996,
hold structure identical (P(h|h) 0.982-0.984, run median 13-14), and
through the frozen v3 onset head P(death) 0.165 / 0.189 / 0.187 vs the
REAL advice's 0.167 / 0.194 / 0.186 along the same censored rows (realised
0.29; the gap is the censoring, identical for the real advice); death
AUROC 0.965-0.977, death-time KS 0.06-0.08.  Gate revision disclosed
(death rate vs the real advice through the same head; persistence like
for like under the true context; E2b keeps the support gate).  MAP hold
decision (the teacher's rule is deterministic), sampled torque.  All
gates 3 / 3 -> variant B rollouts (prior context per path, K = 4) on
node3; fold 0 pooled death 0.301 / reach 0.615 vs sim 0.292 / 0.611.

### Incident (2026-09-20 09:09 UTC): node5 wiped

The node5 pod's filesystem was reset (home directory emptied, no reboot)
while 13 processes ran (load 47): the same failure mode as the PointMaze
pod.  Lost: the repo / venv on node5, queries_s*.npz (regenerable from
the lineage finals), the absorbing-law and query_train / goal_source /
frozen checkpoints (results already pulled), the generator v1 / v2 files.
node3 holds every essential input (lineage finals, branches, anchors,
ett_context, v3 models, nominal, start, dataset).  Rule from here: at most
three processes per pod; pull results per stage.

### ETT line: variant B rollouts (2026-09-20 evening; ett_rollout_v3/hist_exact/SUMMARY.md)

The advice generator v3 with a prior-drawn persistent context per path,
through the frozen v3 motion / onset models, on the same 6,000 held-out
anchors as C / A: pooled death 0.312 vs sim 0.294 (C 0.303, A 0.036),
reach 0.627 vs 0.615, timeout 0.061 vs 0.091, death-time KS 0.028 (C
0.027, A 0.338), death-x KS 0.050 (C 0.064, A 0.513), reach-time KS
0.035; strata deaths within 0.04 of the simulator everywhere (start 0.722
vs 0.724); the failing strata are C's (pre_zone1 under-stall: timeout
0.046 vs 0.178; zone2 death timing KS 0.20); per-anchor AUROC 0.796 (C
0.995) = what is predictable without the actual context, not gated.
The advice process is no longer the ETT's bottleneck; the remaining gap
to the simulator is the long-path motion / onset error shared by C and
B.  Oracle-supervised engineering revision; no futures for training yet.

### The whole pipeline with learned-ETT futures, five seeds (user's decision 2026-09-20; ett_futures/SUMMARY.md)

`scripts/exp_v6_ett_futures.py`: learned-ETT futures for all 53,747
anchors (v3 motion / onset + advice generator v3, prior context per path,
MAP hold, onset sampled; 126 s batched; cross-fitted by fold), then the
unchanged mainline recipe with those futures (arm ETT) vs the simulator
futures (CF) vs the recorded (O), seeds 0-4 (CF / O seeds 3-4 trained
here), one fresh draw (seed 8909).  Futures vs the simulator table:
pooled success 0.633 / 0.621, death 0.307 / 0.291, timeout 0.060 /
0.088; the model under-stalls before the mouth (pre_zone1 timeouts 0.047
vs 0.184) and OVER-times-out on the far-route legs (top corridor 0.29 vs
0.18).  Policies: CF 0.523 / 0.470 / 0.437 / 0.407 / 0.407, CF - O
+0.152 (s.e. 0.024, 5 / 5, RULE MET at five seeds); ETT 0.283 / 0.450 /
0.417 / 0.167 / 0.137, ETT - O -0.006 (3 / 5, rule NOT met); seeds 1-2
keep the oracle gain (0.450 / 0.417), seeds 0 / 3 / 4 stall at the start
(timeouts 0.57-0.72, 82-97 % never reach the first mouth); training
metrics normal (no manifold exit).  Hypothesis: with the far route less
reliable under the model and the shortcut's deaths unchanged, the actor
has no reliable alternative and stalls.  Next ETT item = the long-path
motion / onset error (far-route legs, pre-mouth stall); the advice
process is no longer the bottleneck.  Eval seeds used so far: 909, 2909,
3909, 4909, 6909, 7909, 8909 (5909 unused; 616_000_005 / 616_500_000
reserved).

### User's check after 4de6e8d (2026-09-20): does the motion model turn the advice into motion?  -> v4 and the pipeline rerun

`scripts/diag_v6_ett_motion_ab.py` (ett_motion_ab/SUMMARY.md): with (s,
a_q) fixed, swapping a_b moves the predicted xy by 2-16 % of what swapping
the executed torque does (small), but the STATIONARY GATE flips 9-50 % of
the time under advice swaps at real stalls / pre-mouth / start rows (as
often as under a_q swaps); closed-loop the advice source moves reach by up
to 0.12 on pre_zone1 anchors.  Two further facts: the model under-stalls
where the simulator's start agent gets stuck before the mouth (reach
0.81-0.89 vs 0.00) and drifts off the simulator's far-route paths by step
~28 (reach shortfall 0.08-0.18 in two folds, independent of the advice).
v4 (`fit_v6_ett_one_step_v2.py --v4`): motion regression and stationary
gate on (s, a_q) only, onset head unchanged; preflight gates pass, one-step
accuracy unchanged; held-out far-leg reach 0.72 / 0.94 / 0.72 (v3 0.68 /
0.86 / 0.74), pre-mouth stall anchors walked through 0.83 / 0.74 / 0.71 (v3
0.87 / 0.89 / 0.81); C / B rollouts under v4 = v3 pooled (B death 0.310,
KS 0.025); the same strata fail (pre_zone1 under-stall, zone2 timing).
Pipeline with v4 futures (ett_futures_v4/SUMMARY.md): futures closer to
the simulator on the far legs (top corridor 0.745 vs 0.822; v3 0.708);
ETT 0.483 / 0.213 / 0.337 / 0.390 / 0.310 (mean 0.347; v3 futures 0.291;
CF 0.449); ETT - O +0.050 (s.e. 0.044, 3 / 5, NOT met); the seed pattern
shuffled (0 / 3 / 4 up, 1 / 2 down), the stall at the start remains
(timeouts 0.32-0.65 before the first mouth) while the simulator futures
never stall.  Remaining deviations: pre-mouth / corridor under-stall,
shorter paths, far-route shortfall (halved), zone-2 death timing.

### User's decision after 694bc02 (2026-09-20): keep v4, no multi-step v5 yet; ONE control = exact motion + the current learned risk (ett_futures_hybrid/SUMMARY.md)

`scripts/exp_v6_hybrid_futures.py` + `exp_v6_ett_futures.py --ett hybrid`:
per anchor the learned-ETT construction with the motion replaced by the
simulator (both rockfalls forced inactive; physics only), the risk still
the advice generator v3 + the v4 onset head sampled per step, the death
frame the physics-only next state; same anchors / weights / context prior
/ future law / continuation / recipe, five seeds, draw 8909.  Reading
rule fixed in advance: recovers to near the oracle -> invest in the
motion model (then targeted multi-step supervision); still clearly
behind -> the motion error does not explain the gap, check the
conditional risk and the actual NCE future distribution, no blind motion
training.  The table: outcome shares 0.612 / 0.300 / 0.088 vs the
simulator's 0.621 / 0.291 / 0.088, every region within 0.03; paths
row-identical to the sealed branches for all 13,319 inactive-context
anchors and, for the simulator's deaths, identical up to the rock contact
(8,911 / 10,986 differ only at the death frame: impact |dvel| 1.3 median;
2,071 diverge 1-8 rows before the kill; 608 survivors contact-perturbed);
cell-level conditional death rates corr 0.995 (inside the simulator's
split-half noise 0.982); the NCE positive distribution identical to
0.003.  Result: ETT 0.357 / 0.370 / 0.187 / 0.313 / 0.477, mean 0.341
(v4 0.347, CF 0.449, O 0.297); ETT - O +0.044 (s.e. 0.044, 4 / 5, NOT
met); ETT - CF -0.108 (4 / 5 below), retention 0.29.  -> the "still
clearly behind" branch: the exact motion changed the failure mode but
did not by itself restore the gain (NOT "the motion error is
unimportant").  The exact motion removed the v4 failures before entering a route
(no-route timeouts 7-41 per 300 vs 58-185; oracle 16-30) and replaced
them by shortcut deaths (shortcut share 0.56-0.91 vs oracle 0.29-0.62;
92-100 % die there when the hazard is active, in every arm) and corridor
stalls (shortcut timeouts 32-62 in three seeds vs 8-24).  Both user
checks pass, but they compare MARGINALS (user's review of b35b529): the
critic learns p(g | s, a), and per anchor the tables agree on the
outcome 78.4 % only (5,714 sim successes -> hybrid deaths, 5,257 the
other way; the counts balance).  Whether that is the ordinary re-draw
of one random future per anchor or model bias cannot be told without an
oracle re-draw reference -- the oracle's "5 / 5 seeds" shared ONE
hazard draw per anchor (generation randomness never checked).  The
impact-frame candidate is dropped by code (goals are xy only:
BranchFutures.goal_at); xy / contact differences not excluded.  NEXT
(user's go): scripts/exp_v6_oracle_draw2.py -- the simulator table
regenerated with fresh hazard seeds (232M + anchor_id), same five
seeds / recipe / draw 8909; a REPLICATION CHECK, no equivalence
threshold (the 0.024 seed s.e. is not one); report CF2 - CF per seed,
CF2 - O, hybrid vs both tables; readings kept open: CF2 ~ 0.45 above
the hybrid -> biased learned risk / conditional futures (not localised
to the cut); CF2 ~ hybrid -> table-draw sensitivity first, the 0.45
restated as one table's result; in between -> not separable.  Then:
table-sensitive -> equal-weight multi-future control from the two
oracle tables (anchor weight unchanged, one complete future chosen
first, then the geometric goal law, all outcomes kept, no splicing, no
loss change); both oracles above the hybrid -> the learned risk at the
same state / torque / MATCHED hidden context (survival probability,
death time, realised future sampling).
Wording corrections adopted (user): the v4 - v3 mean gain +5.6 pp with
paired s.e. 9.1 pp is not established; "all stall at the start" was
wrong (v4 seed 3: 139 timeouts = 57 entered the far route, 10 shortcut,
72 no classified route) -> "the learned futures increase the failures
before entering a route; stuck / wandering / other are not
distinguishable from the event fields"; `drift` is a residual
classification, not a cause.  Process: early evaluations of finished
seeds on the chain's platform (GPU JAX), never with a different
JAX_PLATFORMS; `pkill -f` on a pattern that appears in the ssh command
line kills the session.

### The second hazard draw of the simulator table (2026-09-20 night; oracle_draw2/SUMMARY.md) -- the oracle gain does NOT replicate

`scripts/exp_v6_oracle_draw2.py`: the sealed table's construction with
fresh hazard / clock / jitter seeds (232M + anchor_id), same five seeds
/ recipe / draw 8909.  The table: every marginal equal to the sealed
draw (regions within 0.006, death-time KS 0.008, all 3,338 both-inactive
anchors row-identical), per-anchor outcome agreement 0.786 = the
hybrid's 0.784 (its flips are re-draw sized; the hybrid's death-TIMING
deviations, KS 0.085 / 0.054 / 0.171, exceed the re-draw noise 3-5x).
Policies: CF2 0.280 / 0.233 / 0.350 / 0.310 / 0.580, mean 0.351 (sealed
CF 0.449, hybrid 0.341, O 0.297); CF2 - O +0.054 (s.e. 0.057, 3 / 5,
rule NOT met); CF2 - CF -0.098 (s.e. 0.076, 4 / 5 below); hybrid - CF2
-0.010 (2 / 5).  Reading (ii) of the three: table-draw sensitivity
first.  RESTATEMENT: the mainline oracle result (CF - O +0.152, 5 / 5;
+0.216 at three seeds; the 4909 confirmation) is the result under ONE
hazard draw of the future table; its 5 / 5 covered training and
evaluation randomness only; all "below the oracle" verdicts of the
learned arms, the hybrid, the absorbing law and the query-coverage
stages were against one draw.  The hybrid's -0.108 is not evidence
against the learned risk (-0.010 vs draw 2); the learned futures'
failure modes (stall before a route: CF2 seed 1 139 / 300; far-route
incompletion: seed 0 completion 0.41; shortcut deaths) all appear under
a pure simulator re-draw.  Next (user's plan, needs the go): the
equal-weight multi-future control from the two oracle tables (anchor
weight unchanged; one complete future chosen uniformly, then the
geometric goal law within it; all outcomes kept; no splicing, no NCE /
actor change), five seeds, draw 8909, vs both single-draw arms.

### The equal-weight multi-future control (2026-09-20 night, user's go; multi_futures/SUMMARY.md) -- reading (c)

`exp_v6_mainline_pilot.MultiFutures` (anchor by weight unchanged; one
of the two oracle tables chosen uniformly; m by the geometric law within
that path; all outcomes kept; no splicing; losses / recipe / stream RNG
unchanged, the anchor sequence identical to the single-table arms --
sampler_check.json) + `scripts/exp_v6_multi_futures.py`.  MF 0.387 /
0.250 / 0.337 / 0.267 / 0.317, mean 0.311, spread 0.25-0.39 (CF 0.449,
0.41-0.52; CF2 0.351, 0.23-0.58; O 0.297); MF - O +0.015 (s.e. 0.024,
3 / 5, NOT met); MF - CF -0.137 (5 / 5); MF - CF2 -0.039 (3 / 5).  Two
futures per anchor moved every seed toward the start agent's behaviour
(shortcut 0.67-0.95, detour 0.04-0.26, no stalls, mouth at step 51-53
as O) -- the spread narrowed around the O level, not around the better
draw.  Established: under this recipe more futures per anchor do not
stabilise the gain, and the gain is not the mean-outcome signal that
two draws estimate better than one.  Not established: the mechanism
((i) the single-draw preference carried by the idiosyncrasy of one
future per anchor vs (ii) the mixture law changing P(g | s, a) even at
equal expected outcomes) and whether the oracle gain is noise.
Candidates (user): frozen-critic action ranking at the mouth anchors
under CF / CF2 / MF; or draws 3 / 4 to place 0.449 and 0.351 within the
draw distribution before any further ETT work.

### The pairing check (2026-09-21, user's plan after multi_futures; pairing_check/SUMMARY.md; no training)

`scripts/diag_v6_pairing.py`: data / critic / actor layers on the two
oracle tables with the 15 trained critics and actors, strata on the
reset and early-fork states.  DATA: per-anchor consequence not
repeatable across draws at the fork (corr of the success indicator
T1 vs T2: reset -0.06, start_early +0.11, pre_zone1_early -0.01;
pre_mouth 0.49); the population detour signal is the same in both
tables (start_early goal-area mass 0.066/0.073 detour-source vs
0.035/0.034 shortcut-source; death-frame 0.001 vs 0.011).  CRITIC:
NCE fit identical across families / sources; the pairing test (logit
at (s, a, task goal) vs the realised success) at reset / early fork:
CF own draw 0.24-0.49, other ~0; CF2 the mirror; MF 0.15-0.17 each
(not separable from half-memorisation without a third draw); where a
repeatable component exists (pre-mouth on) both draws are tracked.
ACTOR (same state, same goal, candidates = policy modes at that state,
the full objective with twin-min + BC 0.05): under the CF critic the
far-going action is worth +0.9 nats over pi_MF at reset (5x the BC
cost, 5/5 seeds); under the MF critic +0.04 with a seed-dependent sign
(above pi_start in 92 % but above pi_MF in 54 %), BC decides.  READING:
the sealed critic's preference holds only on its own draw -> 0.449 is
not a reliable oracle benchmark; the key anchors' consequences diverge
across draws -> the user's remedy: repeated paired hazard draws at the
pre-selected decision states (e.g. 32 per candidate, frozen
continuation) to measure the true same-state average advantage; no
full tables, no five-seed training.  Caveats: 'vs avg' not out-of-
sample; the cross-anchor detour signal compares different states.

### User's review of 6b03cdd (2026-09-21): readout fix + the repeated paired hazard draws (repeated_draws/SUMMARY.md; no training)

Corrections adopted: "no repeatable consequence per (state, action)" is
not established by a ~0 draw-to-draw correlation of single outcomes
(two actions at 0.3 / 0.7 give the same); the region read-out was a
density ratio against the wrong reference -> fixed (reference goal set
from each family's own training marginal; predicted goal-area
probabilities now match the true masses; the own-draw correlation
pattern stands); actor objective sampling 16 -> 64 (unchanged numbers);
the pairing SUMMARY reworded.  The repeated draws
(`scripts/exp_v6_repeated_draws.py`): 5 groups x 64 states (reset /
start_early / pre_zone1_early anchors, seeded; fresh independent resets
and their t = 20 states), candidates logged / start / CF s0 / MF s0 /
CF2 s0 modes at the task goal, one step + the frozen start agent,
identical clocks / jitter across candidates, activity stratified U00 /
U10 / U01 / U11 x 16 draws (the pre-registered extension from 8 fired
and changed nothing: the width is between states).  RESULT: the first
step is absorbed by the continuation in 69-80 % of the reset states and
86-97 % elsewhere (identical outcomes in all 64 paired draws); success
advantage vs start at reset +0.001 (CF) / +0.066 +- 0.030 (CF2) / +0.046
+- 0.033 (MF), ~0 at start_early / indep_early, +0.02-0.03 at
pre_zone1_early (the logged torque too), +0.01-0.03 at indep_reset;
where not absorbed the actors' torques put the continuation on the far
route (entry 0.09-0.14; completion 0.41 CF / 0.83 CF2 / 0.73 MF, the
rest timeouts).  The gamma-law goal-area mass of the successful
far-going candidates is LOWER than the start torque's (reset CF2 -0.007
vs +0.066 success; pre_zone1_early -0.015..-0.021): the NCE target and
the task return disagree in sign at the decision states.  The critics
prefer every actor torque over start by 1-4 nats (76-100 % of states)
with |corr| <= 0.19 to the measured per-state advantage and chance sign
agreement.  Branches hit: "no stable advantage under the current
continuation" (-> the continuation policy, per the user's rule; not
"first steps unlearnable") and "success advantage but the goal target
does not favour it".  The oracle CF actor's test-time far route is not
a first-step effect the branch tables could supervise (far entry 0.125
at reset with the start continuation).

### The crossover (2026-09-21; repeated_draws/SUMMARY.md, CROSSOVER section) -- continuation = the frozen CF s0 actor for every candidate

User's take-up: multi-future kept as an estimator; no new p(g|s,a)
network; the continuation question examined with the current policy's
closed loop (no macro-actions, no route labels); terminal weighting
deferred.  The same states / candidates / hidden seeds re-run with the
CF s0 mode as the continuation for ALL candidates (start continuation
re-used), plus 64 cf_early states (the indep resets rolled 20 steps by
the CF actor; replay asserted equal), both continuations.  RESULT:
under the CF continuation a first-step advantage exists ONLY at the
reset states (CF torque +0.102 +- 0.056, CF2 +0.078 +- 0.064; pooled
128 resets +0.09 / +0.10, 2.0-2.1 s.e.; far entry +0.22 / +0.30; a
logged shortcut first step derails the CF loop: -0.200 +- 0.059); the
near-2.0 gamma-law mass follows weakly (+0.014 / +0.016, 1.9-2.2 s.e.),
the goal-area mass does not (< 1 s.e.); everywhere else the CF
continuation is WORSE than the start continuation (0.14-0.19 vs
0.22-0.27; timeouts 0.10-0.16) with no first-step advantage; at the CF
actor's own turning states both continuations give 0.45-0.52 regardless
of the first step.  Critics (importance-corrected region probabilities,
own-marginal reference sets): a blanket +0.1-0.25 goal-area margin for
every actor torque at reset (MF's included, whose measured advantage is
negative), uncorrelated per state with the measured region-mass
advantage under either continuation (abs corr <= 0.31, sign at chance);
margins ~0 at cf_early / pre_zone1_early.  Reading: the user's condition
for a CF-continuation resampling round ("stable advantage + preserved
target") holds only at the reset states (196 / 53,747 anchors) and the
current critics would not judge it; outside reset the third case
("no reliable advantage, more timeouts").  Not concluded: first steps
unlearnable, or that such a round would fail.

### The v4 learned ETT on the crossover (2026-09-21; ett_crossover/SUMMARY.md; no training)

User's take-up after 48d603c: a learned ETT that only reproduces
"logged step + start continuation" cannot supply the fork decision; it
must predict the consequences of the relevant query torque + the
matching continuation.  `scripts/diag_v6_ett_crossover.py`: the same
384 states / 5 first torques / 2 continuations through the v4 ETT (the
continuation acting on the model's state; 4 classes x 8 draws; history
counters from each model path; fold 0 for independent states) vs the
simulator crossover.  Q1 (initial turn): the root one-step error does
not depend on the torque (0.01-0.02; 0.13-0.14 at step 5); under the
start loop the error saturates at 0.28 by step 30, under the CF loop it
grows to 1.4-1.7 with per-state heading agreement 0.66-0.69 (the turning
rate about right); least accurate on the turning states themselves
(cf_early 0.7-1.0 under any loop).  Q2 (completion vs stall): under the
training-like loop the model matches; wherever the CF actor controls the
model REMOVES ITS STALLS -- corridor timeouts 0.10-0.16 -> 0.00 (deaths
+0.15, rows 86-123 vs 157-225), far-route completion from reset
0.81-0.86 vs 0.61-0.68, at the turning states under-completion
0.57-0.63 vs 0.75-0.81; goal-area mass over-predicted on the far route
(0.097 vs 0.035).  Q3 (same-state action differences): the two large
reset contrasts reproduced in the MEAN ((CF+CF)-(logged+CF) +0.38 vs
+0.30, r 0.44; continuation +0.33 vs +0.30, r 0.01), the finer ones not
(CF2 over-stated 3-4x, MF's sign wrong, the continuation effect at
indep resets missed +0.015 vs +0.124).  Earliest error (user's ladder):
rung 2 -- closed-loop motion on CF-controlled paths, chiefly the missing
stall / stationary behaviour and the turn; remedy for the user's
decision: one-step (state, query torque) supervision from CF-controlled
simulator paths at reset / turning / corridor states + short multi-step
unrolling on the existing one-step model; logged supervision kept,
episode hold-out, no route label, one 8-D torque; engineering stage.

### The controlled CF-motion revision (2026-09-21; cf_motion/SUMMARY.md) -- user's steps 1-3 after 5650f8d

(1) `collect_v6_cf_motion.py`: the frozen CF s0 mode from 576
pre-selected training-pool anchors (reset / start_early /
pre_zone1_early / pre_mouth / between; the 192 hold-out anchors
excluded), 2 rollouts each with natural hidden draws, every outcome
kept: 1,152 rollouts, 325k full-state transitions (101k static, 31k
slow, 101k far-route rows; 239 rollouts with a >= 20-row stall), split
by start 70/15/15.  (2) `fit_v6_ett_one_step_v2.py --init-from
ett_one_step_v4 --freeze-onset --steps 10000 --cf-rows --cf-share {0 |
0.25}`: v4c (control) vs v4r (25 % CF rows per batch), same init /
budget / selection rule.  CF-val one-step MSE (moving) 0.163 -> 0.095;
original validation unchanged.  (3) `diag_v6_cf_motion_accept.py` on
the test split: L1 (teacher-forced) clearly better -- slow rows 0.0028
vs 0.0206 (v4 predicted 6x too much motion on slow rows), far rows
0.0031 vs 0.0187, turn rows 0.0119 vs 0.0147; L2 (open loop, real
actions) no better -- drift 0.43 vs 0.37 at step 30, 2.1 at 100, and
MORE motion through real stall segments (0.67 vs 0.43 over a real
0.05); L3 (closed loop with the CF actor) unchanged -- reach 0.94-0.97
vs real 0.61, stalled 0.00 vs 0.39.  The crossover re-run with the
fixed diagnostic (paired clocks + death uniforms keyed by (state,
class, rep); Q1 from a motion-only pass at exactly step 30): early
trajectory under the CF loop -10..-25 % with the revision only, heading
agreement 0.66-0.75; stalls still removed (timeouts 0.02 vs 0.12 vs
sim 0.23; completion 0.92 vs 0.67); coarse contrasts still over-stated;
the control v4c differs from v4 in several corners (closed-loop
sensitivity to small parameter changes ~0.1-0.2).  Acceptance: turn
improved modestly; stalls NOT met; action differences not clearly
closer; no regression under start met.  User's ladder -> next rung:
short-sequence multi-step supervision on the one-step model, including
stall and turn segments; not started.

### CF-motion round 2 (2026-09-21; cf_motion/SUMMARY.md ROUND 2) -- stall-entry check, episode split, multi-step term; three nodes again

User's review of 37b990f adopted: L3 horizon unified (400 steps both
sides); the anchor split had episode overlap (28 / 84 test episodes)
-> re-split by source episode (rollouts_cf_ep.npz, overlap 0); the
"basin not learned" claim withdrawn.  STALL-ENTRY CHECK (no training,
L2b): from the exact real entry with the recorded actions the ORIGINAL
v4 walks 2.6-3.2 over a real 0.04 (90 % of segments; the gate never
fires: these stalls are swaying rows), the CF-supervised models 0.09-0.17
(55-72 % under 0.2) and 0.16-0.27 from 20 rows before -> the trajectory-
level failure was mostly the accumulated error BEFORE the segment.
ARMS under the episode split: A one-step, B one-step + short-sequence
term (K 10 recorded actions, unrolled on the model's own predictions,
half the windows on slow / static rows, weight 1).  B halves the
open-loop drift (xy 0.22 / 0.43 / 1.04 at 30 / 50 / 100 vs v4 0.40 /
0.81 / 2.04), improves the trajectory-level stall reproduction (0.44 vs
0.77-1.04 over 0.05), partly restores the timeouts under the CF loop
(0.15 vs v4r 0.02, sim 0.23) and fixes the CF2 contrast (+0.11 vs sim
+0.08); costs +15-20 % on the original validation and the start+start
early error (0.45 vs 0.28); the closed-loop stall is STILL missing
(stalled 0.05 vs real 0.47; reach 0.82 vs 0.42); per-state contrast
correlations unchanged.  Verdict: bar for CRL retraining not met;
candidates = longer / stall-covering unroll, weight, soft gate.
NODES: node5 (30125) and node6 (30016) are gone (refused); two new
pods 30027 (5070 Ti, 20c, 31 GB) and 30049 (3090, 28c, 15 GB) on
35.199.51.171 bootstrapped (python3.12 + node3's venv copied + repo +
artefacts + tables + ckpts), smoke-verified; rule recorded: always say
when more machines would help (feedback-report-server-capacity).

### CF-motion round 3 (2026-09-21; cf_motion/SUMMARY.md ROUND 3) -- (a) vs (b) on the stall entries, one longer window, stop point

User's rules: compare (a) recorded actions vs (b) the actor's own
feedback on the same stall entries first; only if the fixed-action
accumulation is still the main problem, try ONE longer window (K 20)
with nothing else changed and compare it with K 10; stop point = the
evening of 2026-09-22 (no clear closed-loop / action-contrast
improvement -> this week's AntMaze model revision stops; a smaller
position error alone does not justify new CRL seeds).
(a) vs (b) on 64 real stall segments (median 496 rows, real disp
0.024): the actor's feedback changes almost nothing for the
CF-supervised models (run-away share A 0.25 / 0.25, B 0.39 / 0.36 at
the entry; the actor's extra run-aways 3-8 % at the entry, 11-20 %
from 20 rows before; action deviation 0.08-0.13) -> the fixed-action
accumulation IS the main problem; condition for K 20 met.  Arm C (K 20,
otherwise = B) trained on node3: the unroll at weight 1 is at the
edge of stability (folds 0 / 2 diverged after step 8000; pre-spike
checkpoints selected); its acceptance (node3) and crossover (node
30027) running.  Reproducibility bound found: the same models and
actor give closed-loop reach differing by up to 0.14 (B 0.82 vs 0.68)
between two GPUs -> a +-0.1 noise floor on the 38-sequence closed-loop
numbers.  Correction: the round-2 episode-split entry rows said real
disp 0.05; it is 0.024.
Arm C vs B (same device for the acceptance; B's crossover re-run on
C's node = identical, the crossover is device-deterministic): C is the
best model on everything driven by FIXED actions (open-loop drift 0.33
/ 0.64 at 50 / 100 vs B 0.43 / 1.04; trajectory-start stall segments
0.23 with none > 0.5; entry check 0.06) and reproduces the simulator's
timeouts / completion-given-far under the CF loop in the crossover
within 0.02-0.06 (B off by 0.10-0.17), with the coarse action contrasts
closer in 9 of 13 (MF's sign at reset now right); but the strict L3
stall stays far off (0.13 vs real 0.47; B 0.05), the per-state
contrast correlations are unchanged (<= 0.3, chance sign agreement),
and the START policy regresses (early error 0.59 vs v4 0.28; timeouts
under start + start 0.15 vs sim 0.03).  Verdict under the stop rule:
one clear closed-loop improvement (crossover stalls) + coarse contrasts,
no per-state improvement, a start-policy regression -> the CRL bar is
not cleared; recommendation = stop this week's model revision here, C
recorded as the reference ETT with its two limits; no CRL seeds.
User's next request: the FULL rollout acceptance (Step 3a, sealed
thresholds) for v4c20 -- overall / strata death, timeout, reach, far
completion -- with the decision groups reset / start_early /
pre_zone1_early added as an anchor set (diag_v6_ett_rollout.py
--model-dir / --tag / --anchor-set decision); rule: if start_early /
pre_zone1_early are still far off, no CRL training.  Running on the
three nodes (2026-09-21 evening).
RESULT of the full rollout acceptance for C (ett_rollout_v4/v4c20/):
the route-deciding groups pass every sealed gate under both advice
processes (start_early / pre_zone1_early death / reach / timeout within
0.02, far 0.037 vs 0.040, far completion 0.74-0.84 vs 0.81) -- but they
are 73-75 % deaths decided by the frozen onset head; on the uniform
6,000 the C arm FAILS the pooled gates that v4 passed (reach 0.665 vs
0.615, timeout 0.029 vs 0.091) and fails pre_zone1 / between wider
than v4 (timeouts 0.044 / 0.019 vs sim 0.178 / 0.102; v4 0.065 /
0.039): under the START agent's torques it walks through the start
agent's stalls MORE than v4, while under the CF actor it reproduces
the CF actor's stalls -> the learnt stall is the CF actor's pattern,
not policy-independent dynamics.  Recommendation unchanged: no CRL
training from C; stop the week's model revision.

### CF-motion round 3, arm S (2026-09-21 evening; user's plan after d8feadd): protocol-matched windows, one acceptance, the full run

User's corrections adopted: "C learnt the CF actor's stalls" too
strong -> training-distribution mismatch / conflicting goals; reset
(196) stays a reported error.  Arm S = v4 + K 20 windows from the
sealed table's own start-agent branches (--multistep-source branch;
episode-level split respected; original one-step supervision; onset
frozen).  Stable; sealed gates: pooled and between PASS under both
advice processes, pre_zone1 timeout fails by 0.003 (0.125 vs 0.178; v4
0.065), zone2 KS = the frozen onset head; reset under the learnt
advice death 0.738 vs 0.663 (reported).  Futures table: pre_zone1
timeouts 0.128 vs 0.184 (v4 0.067); far legs over-complete 0.04-0.08.
FULL RUN (ett_futures_v4s20/, seeds 0 1 2 fixed first): Learned 0.567 /
0.463 / 0.387 = 0.472 vs oracle table 1 0.477 (-0.004 +- 0.027), vs O
+0.181 (3/3, rule met), vs table 2 +0.184; completion to five: seed 3
0.433, seed 4 0.047 (94 % timeouts, the start stall) -> five-seed
Learned - O +0.083 +- 0.091, 4/5, NOT met; mean 0.379 between the
tables (0.449 / 0.351).  Failure mode = the seed-dependent start
stall (no route in 20-50 % of episodes, one seed 94 %), deaths
0.01-0.14; the futures pass the gates, so the driver is outside the
gated quantities.  Nothing further launched.

### Verification after 5f4827c (user's plan, 2026-09-21 late evening; pre-registered before any result)

1. FROZEN FINALS ON A FRESH COMMON DRAW: evaluation seed 10909 (300
   episodes, policy mode; 909 / 2909 / 3909 / 4909 / 6909 / 7909 / 8909
   used in development, 5909 pre-registered for the clip round,
   616_000_005 / 616_500_000 reserved and untouched); every frozen
   final of Learned (S futures) seeds 0-4, oracle table 1 (CF) 0-4,
   oracle table 2 (CF2) 0-4, O 0-4 and the start agent; all five seeds
   kept; the same paired rule (mean > 2 x seed s.e. and 5 / 5) read on
   10909 for Learned - O, CF - O, CF2 - O; Learned - CF reported.
2. INDEPENDENT FUTURES TABLE, SAME DATASET AND SAME ETT: table draw 2
   of the S futures (generation seed 210_000_000 = GEN_SEED + 4e6;
   ett_futures_v4s20_draw2/), the same learner seeds 0-4 under the
   unchanged recipe, evaluated on 10909 only; the reading = draw 2 vs
   draw 1 per learner seed (the future-sampling effect alone; the
   oracle's own table-draw dependence measured on the learned side).
   ETT fitting seeds are NOT varied here (only if the pipeline is to
   be called stable).
3. SUPERVISION ASSUMPTION: to be stated with the budget (below, after
   the results); no control launched without the user's design call.
Jobs: node3 = draw-2 generate + seeds 0-2 + Learned draw-1 finals on
10909; node 30027 = CF + CF2 finals; node 30049 = O finals + start,
then draw-2 seeds 3-4.
RESULT (2026-09-21 23:00): (1) fresh draw 10909 reproduces 8909 within
0.02 and every rule verdict: Learned 0.347 (seed 4 collapse 0.057 =
the checkpoint), table 1 0.425 (5/5 met vs O), table 2 0.313, O 0.251;
Learned - O +0.096 +- 0.081 4/5 NOT met.  (2) table draw 2 (same
dataset, same ETT): marginals identical to draw 1 within 0.005, but the
learner seeds move -0.13 .. +0.40 (seed 4 0.057 -> 0.457; seed 2 0.340
-> 0.207 via deaths); draw-2 mean 0.384, - O +0.133 +- 0.050 4/5 NOT
met; failure mode switches from start stalls to deaths -> the learner
responds to the per-anchor pairing of the futures, not their marginal
accuracy (the oracle's table-draw dependence measured on the learned
side).  (3) budget stated in ett_futures_v4s20/SUMMARY.md: ETT
supervision = the sealed table (7.65 M simulator transitions, the
oracle's own futures) + log; privileged: hidden contexts for the
advice generator, teacher advice / onset labels; NOT logs-only; the
same-budget plain transition-model control is the user's design call.

### Stall-onset localisation (user's plan after 282ce17; 2026-09-21 night; stall_onset/SUMMARY.md)

Three targeted checks on Learned seed 4 (draw 1), no hyper-parameter
change.  (1) WHEN: milestones rolled from the same 128 start states --
leave-the-start 0.96 (10k) -> 0.60 (20k) -> 0.19 (final); the stall is
a torque SATURATION at the start states (0.63 of joints at the bound
vs 0.07-0.15 for progressing checkpoints), invisible on the training
batches (0.04 -> 0.07 in every run alike, no spikes).  (2) REPEATS on
the same GPU / env with hash-verified tables: the collapse does not
reproduce exactly (0.057 -> 0.343; the GPU training is not
bit-deterministic although the first batches are identical), the
draw-2 recovery reproduces in direction (0.457 -> 0.367); (table,
seed) fixes a basin, run-to-run spread ~0.3.  (3) OBJECTIVE: at the
reset rows the run's own critic rates the saturated action above the
progressing one in 100 % (full objective 97-100 %), the draw-2 critic
the opposite (91-100 %); BC at 0.05 cannot decide (NLL 29 vs 41, both
far); short actor-only updates from the 20k actor + optimizer state:
under the draw-2 critics it walks again (leave 1.00 / 0.98, saturation
0.08), under its own critics it stays (0.60).  Decision tree: no
abnormal updates; the bad critic does prefer the stall -> next = which
conditional future supervision produced it (per-anchor futures of the
two tables at the start anchors; the critics' Q landscape from the
logged torque to the saturated one) -- the user's design call; BC /
ETT / number of futures untouched.

### Which future supervision produced the stall preference (user's A / B after ccc42f6; 2026-09-22; reset_futures/SUMMARY.md)

A. Reset / start anchors' futures, draw 1 vs draw 2 vs simulator: NO
stalled / no-route futures in either draw (no-route timeouts 0-0.5 %,
stall mass 0.005-0.014); the difference is the opposite -- draw 1 gives
the reset anchors MORE goal-reaching futures (success 0.42 vs 0.16 on
the 64 diagnostic anchors, 0.33 vs 0.20 on all 196; goal-area mass
2.3x), while at the 4,049 start_early anchors the draws agree to 0.01
(per-anchor outcome agreement only 0.58-0.60 at the reset rows).
B. Q surface forward -> stall at the 64 reset states: the draw-1
critic's Q rises monotonically to the saturated action from every
forward action (argmax at the bound in 84-98 % of states), largest under
the TASK / near-goal goals (+2.1 to +4.4 nats), then the critic-training
marginals (+1.3 to +3.2), least under the actor stream's relabelled
goals (+0.5 / +1.3 / -0.3); the 20k critic already has the task / near
bump and none under the relabelled goals.  The draw-2 critic's Q falls
toward the stall from its own action and has an INTERIOR maximum from
the start agent's (91-98 %).  Reading: the draw-1 table's richer
goal-area positives at the reset rows made the critic rate reset states
as goal-reaching, with an action dependence whose maximum sits at the
corner of the action box; the stall is the actor's exploitation of an
unbounded action extrapolation, not an imitation of stalled paths.
Nothing launched beyond the two analyses.

### Stall diagnostics, round 2 (user's review of b4aa391; 2026-09-22; stall_onset/SUMMARY.md ROUND 2)

Corrections adopted: draw 1's goal-area mass at reset is BELOW the
simulator's (not 1.5x above); the monotone-Q claim holds for task /
near goals only (interior maxima under the actor's relabelled goals);
the first short-update control was faulty (batches not replayed,
critic restored per 4-update scan; mode-Q not expected-Q) and is
superseded; "optimizer excluded" softened to "none observed at the
logged resolution".  CORRECTED CONTROL (same batches / keys, critic
frozen per update, hash-verified): own critics 0.43-0.49 leave-start
(deeper stall), draw-2 critics 0.83-0.86 (recovery, smaller than the
faulty 0.98-1.00).  OBJECTIVE AS TRAINED (expected Q over 16 sampled
actions + BC, 2,048 logged start rows): under the own critic the
stalled policy's loss is lower (5.83 vs 6.49; BC penalty 0.16 < Q gain
0.82); under draw 2's the progressing one's.  CONSEQUENCES OF ONE FIRST
ACTION (64 reset states, start continuation, paired draws, simulator
and S ETT): the saturated step is NOT bad -- success +0.035 +- 0.032 vs
logged (n.s.), deaths -0.21, timeouts +0.18, far entries 0.17; goal-area
mass slightly LOWER (-0.007); the ETT agrees on success / masses,
under-states timeouts -> the user's THIRD case: the one-step target
does not describe the closed-loop failure; extra queries under the same
continuation would not lower the critic's target.  DECOMPOSITION: all
cosines negative at reset; the draw-1 critic's stalled action has the
SHORTEST phi (0.85x), the norm shrinkage ~ half of the +2.1 nats ->
config.repr_norm (off in the recipe) is a targeted candidate, untested.
Nothing launched beyond the diagnostics.

### One ruler + the normalised ranking (user's plan after 555f22c; 2026-09-22; critic_ruler/SUMMARY.md; no training)

RULER: the binary-NCE logit is a density ratio against the
critic-training marginal up to the known constant B - 1, so region
masses are read from the critic on the same scale as the simulator's
and the ETT's gamma-law masses (the critic's total over all goals is
0.6-1.4: calibrated).  At the 64 reset states the real goal-area mass
is 0.033-0.040 for all four first actions and the ETT agrees; the
collapsed critic reads the stalled action at 0.354 (17.8x per state,
9.5x at 20k) and as leaving the start region (0.037 vs 0.29); the
draw-2 critic over-rates its own progressing action 2.2x; both
under-rate the start agent's action ~0.08x; the critics' readouts are
uncorrelated with the real masses (|r| <= 0.17), the ETT's 0.27-0.48.
CORRECTION of round 2: a query at the stalled action would remove the
over-estimate (target 0.03 vs the reading 0.35), leaving a near-tie.
NORMRANK (per head cos -> min -> mean; tau* scale-matched): the
collapsed critic's stall-vs-progressing preference is unchanged at the
matched scale (+2.2 nats task, +0.8 relabelled; objective +0.53, 71 %
of rows), the recovering critic's walking preference is destroyed at
the mode level (77 % stall under the task goal, ties elsewhere) ->
repr_norm NOT supported.  Tree: ETT right on these actions, the critic
does not fit them (never sees any action but the logged one; the actor
finds the largest extrapolation) -> the third branch, a LOCAL
calibration at the output actions, is the one the evidence points to;
user's call; caveat: both critics over-rate their own maximiser.
