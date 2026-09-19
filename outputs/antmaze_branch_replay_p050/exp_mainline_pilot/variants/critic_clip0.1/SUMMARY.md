# critic_clip0.1: the pilot re-run with the critic gradient clipped (global norm 0.1) before Adam, both arms -- no runaways; CF - O success +0.216, rule MET

Change (disclosed as an optimizer-stabilisation change, not the original
learner): `optax.chain(optax.clip_by_global_norm(0.1), optax.adam(3e-4, eps
1e-7))` for the critic only (`exp_v6_mainline_pilot.critic_optimizer`),
identically in O and CF.  NCE, the actor objective, bc 0.05, learning
rates, Adam, batch 1024, gamma 0.999, the anchors, the branches, the
streams and seeds, the initialisation (fresh paired critics, actor from the
d05 start agent), 30,000 updates and the evaluation (300 episodes, seed
3909, mode) are the pilot's.  Threshold from the spike trace: pre-spike
critic gradient norms median 0.017 / 0.033, p99 ~0.06 / 0.10, the
triggering impulses 0.92 / 2.53.  Pre-registered in `manifest.json`
(criteria 0-4) before training; three nodes; checkpoints stay on the nodes.

## Short-window validation first (`../../diag_replay/spike/CF_s{0,1}/clip0.1/`)

Same checkpoints, batches and RNG as the spike windows, the checkpoint's
Adam moments carried, only the clip inserted: no runaway (fixed-batch
positive logit min -7.3 / -6.8 vs -929 / -2,422; |phi|, |psi| unchanged;
action-gradient norm max 0.41 / 0.19 vs 19 / 20); the actual critic Adam
step bounded at the normal maximum (0.077 / 0.049; runaway 0.72 / 0.69), the
clip active in 1.1 % / 3.0 % of updates; actor scale max 0.105 / 0.082,
saturation <= 0.045 (vs 4.3 / 4.7, 0.78 / 0.72); the critic keeps learning
(step median 0.017 as before, fixed-batch loss -6e-5 per 1,000 updates,
reset-row scores moving); walking: seed 0 0.42-0.76 throughout (normal run
0.00 from 4,450), seed 1 no collapse (normal 0.00 from 2,450) but the
slower pre-spike drift remains (0.65 -> 0.13-0.33 at 2,150-2,700, back to
0.55-0.73 by 2,750-3,000; absent with the critic frozen).

## Criterion 0: training stability (all six 30k runs)

| run | max critic loss (500-update log rows) | min mean positive logit | critic loss @30k | categorical accuracy @30k | actor scale median max / final |
|---|---:|---:|---:|---:|---|
| O s0 / s1 / s2 clip | 0.0074 / 0.0074 / 0.0075 | -6.5 / -6.6 / -6.6 | 0.0070 x3 | 0.013 / 0.014 / 0.014 | 0.08-0.12 / 0.06 |
| O base | 0.008 / **0.210** / 0.012 | -6.9 / **-214** / -9.5 | 0.0070 x3 | 0.012 / 0.011 / 0.011 | 0.17 / 1.65 / 0.75 |
| CF s0 / s1 / s2 clip | 0.0073 / 0.0071 / 0.0073 | -6.4 / -6.3 / -6.4 | 0.0065 x3 | 0.026 / 0.024 / 0.025 | 0.09-0.11 / 0.07-0.08 |
| CF base | **0.133 / 0.120 / 0.362** | **-136 / -122 / -370** | 0.0066 / 0.0067 / 0.0066 | 0.020 / 0.021 / 0.022 | 2.98 / 1.76 / 0.63 |

No spike in any run; the critic reaches the same loss as the base and a
slightly higher categorical accuracy; the actor's scale never leaves 0.06-
0.12.  PASSED.

## Evaluation (same 300 episodes)

| policy | success | detour | death | timeout | success without hazard | success with hazard | mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 | 137 |
| O clip s0 / s1 / s2 | 0.257 / 0.247 / 0.260 | 0.027 / 0.003 / 0.023 | 0.723 / 0.743 / 0.720 | 0.020 / 0.010 / 0.020 | 0.99 / 0.99 / 1.00 | 0.018 / 0.004 / 0.018 | 140 / 129 / 141 |
| **CF clip s0 / s1 / s2** | **0.457 / 0.513 / 0.440** | 0.567 / 0.517 / 0.360 | 0.193 / 0.307 / 0.403 | 0.350 / 0.180 / 0.157 | 0.68 / 0.78 / 0.84 | **0.385 / 0.425 / 0.310** | 512 / 390 / 349 |
| CF base (reference) | 0.267 / 0.320 / 0.480 | 0.107 / 0.167 / 0.500 | 0.627 / 0.567 / 0.260 | 0.107 / 0.113 / 0.260 | 0.84 / 0.89 / 0.69 | 0.080 / 0.133 / 0.412 | 222 / 241 / 423 |

Paired differences on the common episodes (per seed; mean, seed s.e.,
episode-bootstrap s.e.; rule > 2 x seed s.e. and 3/3):

| comparison | success | detour | death | timeout |
|---|---|---|---|---|
| **CF clip - O clip (criterion 1, primary)** | **+0.200 / +0.267 / +0.180 = +0.216 (0.026; boot 0.023), 3/3: MET** | +0.540 / +0.513 / +0.337 = +0.463 (0.064), 3/3 | -0.530 / -0.437 / -0.317 = -0.428 (0.062), 3/3 | +0.330 / +0.170 / +0.137 = +0.212 (0.060), 3/3 |
| CF clip - start | +0.213 / +0.270 / +0.197 = +0.227 (0.022), 3/3: met | +0.478 (0.062), 3/3 | -0.439, 3/3 | +0.212, 3/3 |
| CF clip - CF base (criterion 3) | +0.190 / +0.193 / -0.040 = +0.114 (0.077), 2/3 | +0.460 / +0.350 / -0.140 = +0.223, 2/3 | -0.433 / -0.260 / +0.143, 2/3 | +0.243 / +0.067 / -0.103, 2/3 |
| O clip - O base (criterion 3) | +0.007 / -0.017 / +0.010 = +0.000 (0.008) | -0.001 | +0.004 | -0.004 |
| CF base - O base (the pilot) | +0.101 (0.065), 3/3: not met | +0.24 | -0.24 | +0.14 |

Criterion 4, continuation from the SAME detour-entrance handover states
of diag_traj (the base CF policy's own timeout entrances, n = 4 / 12 / 42):
CF clip 0.75 / 0.75 / 0.67 vs CF base 0.75 / 0.42 / 0.48, start 0.50 / 0.58
/ 0.52, O clip 0.50 / 0.42 / 0.55 -- at or above the start policy in all
three seeds.  From the pre-stall states (n = 28 / 33 / 73): CF clip 0.32 /
0.33 / 0.42 vs CF base 0.50 / 0.24 / 0.45, start 0.32 / 0.21 / 0.42 -- at
the start level, mixed against the base.

## Reading

* With the critic runaways removed in both arms, the counterfactual
  futures give a success gain over the recorded futures of +0.216 on the
  same episodes, 3/3 seeds, 8 x the seed s.e. -- the pre-registered rule
  of the mainline contract is met for the first time.  The gain is the
  route: detour 0.36-0.57 (O: 0.00-0.03), deaths 0.19-0.40 (O: 0.72-0.74),
  success with an active hazard 0.31-0.43 (O: <= 0.02).
* The stabilisation does nothing for O (O clip = O base = start): the
  recorded futures carry no route signal; the effect is specific to the
  counterfactual futures, as the contract intends.  Against the base CF,
  the clip lifts the two seeds whose base runs had the early runaways
  (+0.19 / +0.19) and leaves seed 2 where its base already was (-0.04;
  its base run's runaway at 6.5k was followed by the 0.50-detour actor).
* What remains (corrected 2026-09-19 after the user's per-episode recount;
  the sentence "the detour is completed about 40-70 % of the time" that
  stood here was wrong).  Route ledger of the clipped CF (REPORT.md, the
  env's labels): the far route (top-west corner reached) is taken in
  170 / 155 / 108 of the 300 episodes and completed in 111 / 122 / 87 of
  those -- completion 0.653 / 0.787 / 0.806, pooled 0.739; no far-route
  episode dies, every far-route loss is a timeout (59 / 33 / 21).  The
  other losses are shortcut deaths (52 / 89 / 113) and no-route timeouts
  (38 / 19 / 18).  So two things remain open: fewer than half of the
  episodes take the far route (0.57 / 0.52 / 0.36), and a quarter of those
  that do run out of time.  Rescuing every far-route timeout with nothing
  else changed would give 0.653 / 0.623 / 0.510 (bookkeeping, not a
  prediction), so the far-route walking alone cannot carry the success
  rate to 0.70 -- more episodes have to take the far route reliably.
  "Success without a hazard 0.68-0.84 vs start 0.99" is not a same-route
  comparison (the start walks the shortcut, the clipped CF mostly the
  longer far route) and does not show a walking loss from 0.99 to 0.70.
  The far-route timeouts of THESE policies are audited separately (late
  entrance / stall / entrance state, and the continuation from the same
  entrance with the same remaining time: `../../diag_traj/
  DETOUR_AUDIT_critic_clip0.1.md`); the recipe is confirmed once on a
  fresh evaluation draw (`confirm_s4909/REPORT.md`).
* Status of the claim: oracle evidence (the simulator generated the
  futures), under an optimizer-stabilisation change applied to both arms
  and disclosed as such; not the original learner, not a learned-ETT
  result.  No selection: one variant, one threshold, evaluated once.

## Confirmation on a fresh evaluation draw (seed 4909): REPRODUCED

The development draw (seed 3909; 909 / 2909 before it) had been re-used
throughout the development of the pilot and its variants, so the sealed
recipe was re-evaluated ONCE, with the same final checkpoints (hashes sealed
in `confirm_s4909/manifest.json` before any evaluation on the draw), on 300
previously unused episodes.  `confirm_s4909/REPORT.md`: O clip 0.260 /
0.260 / 0.260 (= start 0.260); CF clip 0.480 / 0.447 / 0.430, detour 0.60 /
0.51 / 0.33, death 0.20 / 0.35 / 0.43, timeout 0.32 / 0.20 / 0.14; CF - O
success +0.192 (seed s.e. 0.015, boot 0.024, 3/3) MET, CF - start +0.192
(0.015, 3/3) MET; detour +0.46, death -0.40, timeout +0.20 (3/3).  Route
ledger on the new draw: far route taken 0.60 / 0.51 / 0.33, completed
0.657 / 0.704 / 0.859 (every far-route loss a timeout, none a death).  The
current recipe is therefore the fixed oracle reference for the learned-ETT
work; the confirmation draw is not re-used for development.

## Far-route timeouts of THESE policies (`../../diag_traj/DETOUR_AUDIT_critic_clip0.1.md`)

Audited on the clipped CF policies' own trajectories (not the base CF's
entrances): (a) entered too late (steps left at the corner below the p10 of
the successful completions, 308) -- 1 / 3 / 4 episodes only; (c) entrance
state at the corner (torso height, uprightness, speed, heading, joint
speed) -- AUC 0.38-0.67, no signature; (b) time enough -- seed 0: 33 of
56 stop AT the top-west corner about 46 steps after reaching it and stand
for ~630 steps, 11 fall in the east column; seed 1: 19 fall in the east
column (arc 32, ~280 steps after the corner), 12 stop; seed 2: 16 timeouts,
mixed.  The continuation test from the SAME corner state with the SAME
remaining time (pooled 121 timeout entrances / 319 success entrances):
the clipped CF itself reaches 0.66 from its own timeout entrances (the
recorded timeout is reproduced in only 82 / 242 re-runs: with the policy and
the state fixed, the outcome flips on numerical differences), the start
policy 0.63, O clip 0.65, base CF 0.64, the blind driver 0.76; from the
success entrances: start 0.71, O clip 0.75, base CF 0.75, driver 0.93.
Reading under the pre-stated rules: R2 (an update-caused execution
regression) is NOT supported -- the clipped CF is at the start / O-clip
level from the same states; R3 partly -- every policy, the driver
included, is ~0.1-0.17 lower from the timeout entrances; R1 rarely (the
driver still completes 0.76).  The far-route walking from the corner is
~0.65-0.75 for every learned policy, including the start agent it was
initialised from: the completion limit is the inherited far-route walking
(5 % detour data), not a loss caused by the update.  Caveat on every
per-episode figure: the CPU replays reproduce the GPU evaluation episode by
episode in only 48-112 of 300 (`rollout_check*.json`; the per-policy rates
agree within 0.05), so episode-level outcomes are numerically chaotic and
only rates are comparable.
