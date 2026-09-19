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
* What remains: the detour is completed about 40-70 % of the time --
  timeouts 0.16-0.35 scale with the detour share (0.36-0.57), success
  without a hazard 0.68-0.84 (start 0.99).  This is the slower walking
  loss that the clipped seed-1 window also showed (driven by ordinary
  critic updates, not by spikes) plus the start policy's own detour
  walking (~0.5-0.7 from the entrance).  Not addressed by the clip; it is
  the next question, now examinable without training collapses.
* Status of the claim: oracle evidence (the simulator generated the
  futures), under an optimizer-stabilisation change applied to both arms
  and disclosed as such; not the original learner, not a learned-ETT
  result.  No selection: one variant, one threshold, evaluated once.
