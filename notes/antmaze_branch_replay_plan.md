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
