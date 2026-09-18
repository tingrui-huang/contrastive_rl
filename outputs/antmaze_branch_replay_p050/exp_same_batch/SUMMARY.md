# Identical batches, frozen critics only differ (point 2), with the true-t = 0 critic diagnostic and the corrected vanilla readout (point 1)

> **Status under `notes/MAINLINE_CONTRACT.md` (2026-09-19): historical diagnostic configuration.**  S1 (dataset reset rows) and S2 (control-replay anchors) are both diagnostic batch sources for the actor's critic term, with FROZEN critics and balanced BC; neither is the mainline actor stream (the recipe buffer's own law over the whole dataset, random_goals 0, BC 0.05 on the same rows, joint training).  What stands: the earlier actor-level ordering was confounded by the batch source, and the critic-level t = 0 ordering.  'vanilla@S1 beats the start' and 'ext_bc is best on S2' are statements about those configurations, superseded as statements about the method (contract section 1c).  The point-1 readout correction (`vanilla_draw`) is unaffected.  Nothing below was edited.

Follows `exp_agent_round/SUMMARY.md`.  Script `scripts/exp_v6_same_batch.py`
(`batch_audit.json`, `REPORT.md`); point 1 in `scripts/exp_v6_agent_round.py`
(`seal_t0`, `analyze --diag t0`, `vanilla_u`) and
`scripts/build_v6_policy_replay.py` (`marginal_goal_frames(law=...)`).

## Why this was needed

`run_v6_branch_replay.train_joint` applies `row0_prepare` to whatever file
`V6_BRANCH_REPLAY` names, so the actor's critic-term batches are anchored at
row 0 of that file's paths.  For the branch critics that is the replay's
anchor states (start rows t <= 5, turning, north leg, every-60th general
rows); for the recorded-data critics the "replay" was the dataset copy, so
the reference actors' critic term was anchored at the 1,000 RESET rows.  The
agent-round actors therefore differed in the states they were updated at,
not only in the critic; the first-step check had shown that the reset-state
torque decides the route.

## Point 1a: corrected vanilla goal marginal

`marginal_goal_frames(law='vanilla_draw')`: episode uniform, anchor row
uniform within the episode's valid rows, future geometric from that row --
the TrajectoryBuffer's default draw (long episodes not up-weighted; the
first attempt weighted episodes by length and was corrected before any
analysis used it).  The old row-0 readout is kept as `vanilla`, the
corrected one is `vanilla_u`.  Difference between the two readouts: Cdev
pick gain +0.013 vs +0.013; t = 0 +0.052 vs +0.046 (+0.006 +- 0.004).  The
correction changes no conclusion.

## Point 1b: the true t = 0 diagnostic (`diag_t0`, `diag_t0_bc`; `manifest_t0.json`)

64 reset rows (t = 0) of the old held-out episodes, one per episode; the
Cdev candidate types (recorded, agent mode, 2 BC samples, 1 agent sample),
16 paired draws, labels under the fixed agent's continuation and under the
BC continuation; region readout, deployed min; paired episode bootstrap.
The earlier Cdev (start rows t <= 5, momentum-committed, plus Cnew) is kept
for comparison.

| labels | critic set | agreement (seeds) | pick gain (paired s.e.) | minus control | minus round1 |
|---|---|---|---|---|---|
| agent continuation (399 decided pairs, 61 episodes; selector signal +0.154) | vanilla_u | 0.63 / 0.61 / 0.59 | **+0.052 (0.014)** | **+0.069 +- 0.019 (z 3.7; 3/3)** | +0.051 +- 0.017 (z 2.9; 3/3) |
| | vanilla (row-0 readout) | 0.64 / 0.62 / 0.59 | +0.046 (0.014) | | |
| | ext_ag | 0.62 / 0.59 / 0.62 | +0.033 (0.014) | +0.050 +- 0.017 (z 2.9) | +0.031 (z 2.0) |
| | ext_bc | 0.60 / 0.61 / 0.63 | +0.023 (0.016) | +0.039 +- 0.020 (z 2.0) | +0.021 (z 1.1) |
| | round1 | 0.48 / 0.57 / 0.53 | +0.002 (0.011) | +0.018 (z 1.6) | -- |
| | control | 0.42 / 0.49 / 0.42 | -0.017 (0.010) | -- | |
| | random | 0.55 | +0.007 | | |
| BC continuation (455 pairs, 64 episodes; selector +0.165) | vanilla_u | 0.64 / 0.65 / 0.59 | +0.053 (0.014) | +0.071 +- 0.021 (z 3.3; 3/3) | +0.032 (z 1.7) |
| | ext_ag | 0.64 / 0.64 / 0.65 | **+0.062 (0.015)** | **+0.081 +- 0.020 (z 4.1; 3/3)** | +0.041 (z 2.5) |
| | ext_bc (its own target) | 0.66 / 0.64 / 0.66 | **+0.060 (0.016)** | **+0.079 +- 0.022 (z 3.5; 3/3)** | +0.039 (z 1.9) |
| | round1 | 0.52 / 0.63 / 0.51 | +0.021 (0.012) | +0.040 +- 0.012 (z 3.3) | -- |
| | control (its own target) | 0.39 / 0.46 / 0.43 | -0.019 (0.011) | -- | |

vanilla_u minus ext_ag / ext_bc: +0.019 +- 0.014 / +0.030 +- 0.018 under
the agent labels, -0.009 +- 0.016 / -0.008 +- 0.016 under the BC labels;
ext_ag minus ext_bc +0.010 / +0.002.  Cdev (t <= 5) with the corrected
readout: every set at chance under both labels (unchanged).

Reading: at the true reset states the critics separate -- the recorded-data
critic and the two extended-query branch critics rank candidate first
torques above chance (pick gain +0.05-0.06 against a selector ceiling of
+0.15-0.17), the old-query branch critics do not (round1 ~0, control
below chance on its own target).  The earlier "all critics at chance on new
states" held only for t >= 1 rows, where the momentum has already decided
the route.  Adding the agent's candidates gives the branch critic a t = 0
signal of the same size as the vanilla critic's (equal under the BC
labels, ~1.5 s.e. below under the agent labels).

## Point 2: same batches, frozen critics only differ

All actors from the fixed agent (0.507 on the 2909 draw), frozen critic,
bc 0.05, 30k, seeds 0-2.  Batch audit (`batch_audit.json`): with the batch
source and seed fixed, the critic-term batches and the BC batches are
identical across critic tags (3 draws hashed, both sources); the JAX key is
seed-determined.  S1 = dataset row-0 anchors (the reference actors' stream:
critic term at the reset rows); S2 = the control replay's anchors (the
control actors' stream).  vanilla@S1 = `joint_vanref_round1`, control@S2 =
`joint_ctrl_round1`; the other 24 actors are new.

| critic | S1 success (seeds) | S1 mean (s.e.) | S1 detour | S2 success (seeds) | S2 mean (s.e.) | S2 detour | S1 - S2 |
|---|---|---|---|---|---|---|---|
| vanilla | 0.590 / 0.663 / 0.730 | **0.661 (0.040)** | 0.66 | 0.177 / 0.253 / 0.307 | 0.246 (0.038) | 0.02 | **+0.416 (s.e. 0.055; 3/3)** |
| control | 0.383 / 0.277 / 0.430 | 0.363 (0.045) | 0.21 | 0.293 / 0.280 / 0.327 | 0.300 (0.014) | 0.13 | +0.063 (n.s.) |
| round1 | 0.280 / 0.490 / 0.290 | 0.353 (0.068) | 0.20 | 0.310 / 0.333 / 0.297 | 0.313 (0.011) | 0.16 | +0.040 (n.s.) |
| ext_ag | 0.283 / 0.397 / 0.270 | 0.317 (0.040) | 0.14 | 0.300 / 0.310 / 0.483 | 0.364 (0.060) | 0.19 | -0.048 (n.s.) |
| ext_bc | 0.590 / 0.607 / 0.307 | 0.501 (0.097) | 0.45 | 0.450 / 0.390 / 0.373 | **0.404 (0.023)** | 0.35 | +0.097 (n.s.) |

Pairwise, success (> 2 pooled seed s.e. and 3/3):

- S1: vanilla - control +0.298 (3/3), - round1 +0.308 (3/3), - ext_ag
  +0.344 (3/3), - ext_bc +0.160 (s.e. 0.105, 2/3: not met); ext_bc -
  control +0.138 (s.e. 0.107, 2/3: not met).
- S2: ext_bc - vanilla **+0.159 (s.e. 0.044; 3/3)**, ext_bc - control
  **+0.104 (0.027; 3/3)**, ext_bc - round1 **+0.091 (0.026; 3/3)**; vanilla
  - control -0.054 (n.s.), vanilla is the lowest critic on S2 (2% detours).
- vs the start (0.507): only vanilla@S1 above (+0.154, 3/3); ext_bc@S1
  -0.006 (1/3); every other cell below by 0.10-0.26 (3/3).

Readings, in the user's three cases:

1. *Branch critic improves under S1?*  Not for control / round1 / ext_ag
   (+0.06 / +0.04 / -0.05, within noise).  ext_bc reaches 0.59 / 0.61 on two
   seeds (0.31 on the third): +0.10, not significant, seed-limited.
2. *Vanilla drops under S2?*  Yes, by 0.42 (3/3): with the replay-anchored
   batches the recorded-data critic gives 0.25 success and 2% detours, the
   worst of the five.  A large part of the reference arm's +0.15 over the
   fixed agent came from the batch arrangement (critic term at the reset
   rows), not from the critic alone.
3. *Same batches, vanilla still beats branch?*  Under S1: yes against the
   old-query branch critics (3/3, > 2 s.e.), not significantly against
   ext_bc.  Under S2: no -- ext_bc beats vanilla, control and round1 (3/3,
   > 2 s.e.), and ext_bc is the only critic that keeps the actor at >= 0.40
   under both batch sources.

Caveats: S1 and S2 change the state AND the goal distribution of the critic
term together, so "S1 helps" is not attributable to the reset rows alone;
three seeds; one fixed agent; the 2909 draw is a development draw; no cell
beats the fixed agent except vanilla@S1.  What this settles: the earlier
actor-level ordering was confounded by the batch source; the critic-level
ordering at t = 0 (vanilla ~ ext_ag ~ ext_bc > round1 > control) is what
survives, and on the replay-anchored batches the coverage-extended
BC-continuation critic is the best of the five.  What it does not settle:
whether any branch critic can move the agent above where it started.
