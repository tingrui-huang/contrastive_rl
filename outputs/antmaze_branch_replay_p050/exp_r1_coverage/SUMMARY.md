# Torque coverage vs repeated outcomes (equal budget): summary

Pre-registered manifest: `exp_r1_coverage/manifest.json` (sealed 2026-09-18
04:38 node time, before any rollout; git sidecar
`manifest_git_sidecar.json`: local HEAD 69e6572 with uncommitted
diagnostic files, deployed-script hashes).  Reference: the round-1 replay
(sha256 e5041e370c20), its three critics, the frozen
continuation (820cb7e229ed), and the sealed A / B / C diagnostic outcomes
(2e1c81bffbc7).  No
actor was trained; nothing was tuned per arm; nothing extended.

## Arms (identical budget, identical anchors, identical protocol)

| arm | distinct policy torques per dense state | outcomes per torque | policy branches per state | recorded / general paths | anchor weights | critics |
|---|---:|---:|---:|---|---|---|
| r1 (reference) | 2 (sample0, sample1) | 2 | 4 | round-1 | uniform over paths (= 1 each) | existing |
| R | 2 (the SAME sample0, sample1) | 8 fresh | 16 | round-1 paths verbatim | round-1 paths 1, new paths 0.25 | 3 new, seeds 0 / 1 / 2 |
| COV | 8 NEW samples from the frozen policy (TRAIN_TORQUE_SEED stream) | 2 fresh | 16 | round-1 paths verbatim | round-1 paths 1, new paths 0.25 | 3 new, seeds 0 / 1 / 2 |

Effective weights audited from the built replays (both arms, 35,756
paths each): state mass dense / general 0.5069 / 0.4931, mixture recorded
/ policy 0.4155 / 0.5845, within-dense recorded 0.3333 -- exactly the
round-1 values (9000/17756, 7378/17756, 1/3).  The weights are realised
exactly by the buffer's weighted anchor stratum (in-batch negatives follow
the same draw).  Training: 30k NCE updates, batch 1024, lr 3e-4, gamma
0.999, repr 16, 1024x1024, twin heads, bc 0.05 in the shared config
(critic-only stage; the critic does not depend on the actor), milestones
10k / 20k; R critics 627-639 s each, COV 352-388 s (different GPUs).

Seed streams: COV training torques `515000003 + 7919 * anchor`, training
draws `515500000 + 1000 * anchor + draw` (R draws 0-7, COV draws 0-1,
shared across the candidates of a state); B validation torques
`909707017 + 7919 * anchor` (the diagnostic's), fresh untouched torques
`616000005 + 7919 * anchor` (reserved, unused); pairwise base differences
are not multiples of the stride (manifest `stream_separation_mod_stride`);
minimum distance of any COV training torque to any B torque at the same
anchor 0.39, to any round-1 torque 0.45.

## Evaluation

The sealed A / B / C outcomes (`diag_r1_abc/outcomes.npz`, 16 fresh paired
draws) with the same script, the same primary readout (region-integrated
exp(f), radius 0.5, the arm's own weighted goal marginal, deployed min
over heads), the same pair classes and episode bootstrap.  Layer A for
COV = recorded + its own training torques cov0 / cov1 at the 160 A
anchors, run with the diagnostic's draw seeds (`outcomes_armCOV_A.npz`);
for r1 / R layer A = recorded + sample0 / sample1 (existing).  B and C
unchanged.  Paired episode bootstrap of the run differences (same pairs,
seed-averaged critics): `diag_r1_abc/paired_boot_{A,B,C}.json`.

## Result (pooled dense strata; agreement among decided pairs, episode-bootstrap s.e.; pick gain in P_goal)

| layer | r1 seed 0 / 1 / 2 | R seed 0 / 1 / 2 | COV seed 0 / 1 / 2 | seed-mean r1 / R / COV (paired s.e.) |
|---|---|---|---|---|
| A familiar state + familiar torque (230 / 230 / 223 pairs) | 0.77 / 0.71 / 0.70; gain +.042 / +.032 / +.032 | 0.67 / 0.67 / 0.70; +.029 / +.030 / +.036 | 0.63 / 0.62 / 0.66; +.024 / +.029 / +.034 | -- (different candidate sets) |
| B familiar state + unseen torque (222 pairs, 78 episodes) | 0.55 / 0.51 / 0.55; +.013 / .000 / +.005 | 0.58 / 0.53 / 0.55; +.010 / .000 / +.009 | **0.63 / 0.56 / 0.56**; +.012 / +.007 / +.007 | 0.536 / 0.554 / 0.584 (0.028 / 0.028 / 0.029) |
| C held-out episodes (758 pairs, 45 episodes) | 0.50 / 0.48 / 0.49; ~0 | 0.48 / 0.50 / 0.52; ~0 | 0.51 / 0.48 / 0.47; ~0 | 0.491 / 0.500 / 0.485 (0.018 / 0.022 / 0.015) |
| C matched types (227) | 0.47 / 0.48 / 0.46 | 0.41 / 0.46 / 0.46 | 0.52 / 0.48 / 0.44 | -- |

Paired differences on B (seed-mean, episode bootstrap, 2000 resamples):
COV - r1 = +0.048 +- 0.039 (z 1.2; per seed +0.086 +- 0.048, +0.045 +-
0.051, +0.014 +- 0.049); COV - R = +0.030 +- 0.035 (z 0.9); R - r1 =
+0.018 +- 0.026 (z 0.7).  On C every difference is within 1 s.e. of zero.
The round-1 A candidates (sample0 / sample1) scored by the COV critics --
for COV these are UNSEEN torques at familiar states, a second B-type
check -- give 0.57 / 0.53 / 0.57 (seed-mean 0.554 +- 0.029), in line with
COV's B.

B by stratum (seed 0 / 1 / 2): COV turn 0.71 / 0.56 / 0.53 (gain +.033 /
+.034 / +.002), north leg 0.69 / 0.69 / 0.66, start_late 0.54 / 0.53 /
0.67, start_early 0.57 / 0.51 / 0.43, shortcut_early 0.65 / 0.57 / 0.57;
r1 turn 0.53 / 0.55 / 0.59, north leg 0.52 / 0.59 / 0.72, start_early 0.66
/ 0.45 / 0.51; R turn 0.61 / 0.61 / 0.56, north leg 0.69 / 0.52 / 0.76.
The stratum cells have 23-66 decided pairs and s.e. 0.06-0.15; no
stratum separates the arms on its own.

Secondary readouts (exact goal, per head) tell the same story: COV B
0.50-0.64, R 0.51-0.62, r1 0.48-0.58, with pick gains 0.00-0.015.

## Pre-registered decision

The line for "COV improves B": pooled dense B agreement above r1 AND above
R by more than 2 paired s.e., above 0.5 by 2 s.e., with a positive pick
gain by 2 s.e.  Outcome: above 0.5 by 2.9 s.e. (met); above r1 by 1.2
s.e. and above R by 0.9 s.e. (NOT met); pick gain +0.012 +- 0.006 (seed
0, met) but +0.007 +- 0.007 (seeds 1, 2, not met).  R does not improve B
(z 0.7).  The fresh untouched action set (reserved streams) was therefore
NOT generated or scored; the reserved seeds remain unused.

## Interpretation (the four cases)

The evidence falls under **Case 4 -- neither arm materially improves B**
by the pre-registered line, with one caveat that must be stated: COV is
directionally better than both r1 and R on B (all three seeds at or
above their r1 counterparts, seed-mean +0.05), at 1.2 paired s.e.  That
is not evidence of an action-coverage effect at this budget, and it is
not evidence against one either; it is inconclusive at 222 decided pairs
from 78 episodes.  What the experiment does establish:

- Repeated outcomes (R, 8 per torque instead of 2) do not improve B (z
  0.7) and do not improve the familiar-key fit against fresh consequences
  either (A: R 0.68 vs r1 0.73 seed-mean, z -1.5).  Target variance at the
  training keys is not the limiting factor (Case 3 is not supported).
- Broader torque coverage at the same states (COV) lowers the familiar-key
  fit (A 0.63 vs 0.73: each training torque now carries a quarter of the
  mass) and raises B by at most a few points; the critic remains far from
  the A level on unseen torques (0.58 vs 0.63-0.77).
- C stays at chance for every arm (0.485-0.500).  Nothing in either arm
  touches the state / episode boundary; no claim about it follows from
  this run.

Under Case 4: do not enlarge the replay further (no 16 / 32 / 64 torques,
no longer training, no loss / architecture / route / macro / subgoal
change from this result).  The current critic representation and
training procedure do not convert additional within-state torque
coverage into rankings of unseen torques at a rate that this budget can
distinguish from zero.  If the small directional COV gain is to be
resolved at all, the only data-neutral way is more evaluation resolution
on B (more validation torques per anchor from the reserved stream, more
episodes), not more training data -- and that is a separate decision.

## Caveats

- The B validation set is the diagnostic's 3 torques x 160 anchors (222
  decided pairs); its resolution (paired s.e. ~0.04 on a difference) is
  the binding constraint on the COV - r1 comparison.
- The two arms trained on different GPUs (RTX 4090 vs 4090 laptop) with the
  same code, seeds and steps; JAX GPU numerics may differ slightly between
  devices.  r1 trained on an RTX 3060 Ti.
- COV's training draws 0-1 share seeds with R's draws 0-1 at the same anchor
  (common random numbers by construction); the arms are therefore not
  independent replicates of the hazard draws.
- The round-1 A candidates are unseen for COV, so the COV "A" row of the
  paired-bootstrap file is a B-type quantity; COV's true A is the cov0 /
  cov1 row above.
- The manifest was sealed on a node without a git checkout; the local
  commit and dirty state are in the sidecar.
- Nothing here bears on the actor stage, which stays unrun.
