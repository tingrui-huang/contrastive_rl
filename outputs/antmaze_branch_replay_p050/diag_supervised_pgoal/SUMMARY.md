# Direct supervised P_goal prediction as a diagnostic control: summary

Sealed manifest `diag_supervised_pgoal/manifest.json` (2026-09-18 07:11 node
time, before training; git 355d273 in the local checkout -- the node has no
git, see the manifest's `git` field; the deployed script hashes are in
`provenance.scripts`).  Training data: the R-arm replay only
(`exp_r1_coverage/replay_armR.npz`, sha 193d3dde...), aggregated to its
13,256 exact (anchor, candidate) keys -- 8,756 general keys with 1 path,
1,500 dense recorded keys with 2 paths, 3,000 dense policy keys with 8
paths -- with summed weights and weight-averaged targets (squared-error
equivalence documented in the manifest); state mass dense 0.5069, recorded
mixture 0.4155, within-dense recorded 1/3 (= round 1).  Leakage checks: 0
held-out C episodes among training keys; no B torque used as a training key
(minimum distance of a training torque to a B torque at the same anchor
0.028, i.e. distinct keys).  Reserved validation seeds unused.  No actor
trained; no rollouts generated; NCE checkpoints untouched.

Models: S h(s, a, g) on the critic's exact inputs (29-d state + 2-d goal +
8-d torque), N h0(s, g) without the torque; (1024, 1024) relu MLP with the
critic encoders' initialiser, sigmoid output, squared error on P_goal,
Adam 3e-4 (eps 1e-7), batch 1024, 30,000 updates, seeds 0 / 1 / 2, final
checkpoint only, identical weighted key draws for S and N, inputs
standardised with training-key statistics.  Device caveat: all six runs
ran on the node's CPU (the imported replay module pins JAX to CPU), one
device type for all six, 250-310 s each; the NCE critics they are
compared with trained on GPUs.

Target audit (unchanged from round 1): P_goal = sum_{j=1}^{L-1} 0.999^j
1[|xy_j - goal| <= 0.5] / sum 0.999^j over the path from the anchor to
death or the horizon 800 - t, zero-torque hold after reaching; the goal is
the anchor episode's recorded goal, which is the continuation's task goal
and the critic's goal input.  Absolute time (hazard clocks from the reset,
remaining horizon) shapes the target and is in no learner's input --
identical conditioning for NCE, S and N, so the comparison is fair; it is
a shared limitation, not a fix applied here.

## Answers

**1. Can the scalar predictor fit the training-key targets?**  Yes.  S:
weighted MSE 0.0051 / 0.0052 / 0.0053 against a constant-predictor MSE of
0.1048 (weighted R^2 0.95); N: 0.0221 / 0.0221 / 0.0218 (R^2 0.79).  The
state and goal alone explain about four fifths of the target's weighted
variance; the torque explains most of the rest at the training keys.

**2. Does this survive fresh outcomes at A (familiar state, familiar
torque)?**  Largely.  On the 16 fresh draws at the 480 A keys, S predicts
the sample mean with MSE 0.012-0.013 (constant 0.0175, sampling-noise
floor 0.0029; corr 0.69-0.71), and orders the 230 decided pairs at 0.87 /
0.87 / 0.85 (s.e. 0.03) with pick gain +0.056 / +0.054 / +0.056 -- close to
the +0.061 an eight-draw empirical selector obtains, and above the R NCE
critics (0.67 / 0.67 / 0.70, gain +0.029 / +0.030 / +0.036): S - NCE = +0.18
+- 0.03 (z 7.0), every seed +0.16..+0.20.  Per stratum S is 0.69-0.94
everywhere (turn 0.89-0.94, north leg 0.92), including the start-type
strata where NCE sits at 0.38-0.62.  So the training keys carry
learnable single-torque information about fresh consequences, and the
direct objective extracts more of it than the 30k NCE critic did.

**3. Does it transfer to unseen torques at B (same states)?**  No.  S:
0.54 / 0.61 / 0.54 agreement (s.e. 0.035), pick gain +0.005 / +0.020 /
+0.002 (s.e. 0.009); NCE: 0.58 / 0.53 / 0.55, +0.010 / -0.000 / +0.009;
S - NCE = +0.008 +- 0.036 (z 0.2), per seed -0.045 / +0.081 / -0.014 --
no consistent direction.  In level terms S is worse than a constant at
unseen torques: fresh-outcome MSE 0.032-0.039 against 0.0175 (corr
0.20-0.27), while the torque-free N gives 0.020-0.021 (corr 0.31-0.34).
The regressor's dependence on the torque is fitted key by key and does
not interpolate to new samples from the same policy at the same state.

**4. Does it transfer to held-out episodes at C?**  No.  S: 0.52 / 0.53 /
0.52 (758 decided pairs, 67 episodes), pick gain +0.002 / +0.013 / +0.008
(s.e. 0.009-0.015); NCE 0.48 / 0.50 / 0.52; S - NCE = +0.022 +- 0.024
(z 0.9); the matched-candidate subset +0.076 +- 0.037 (z 2.1, 227 pairs,
per seed +0.088 / +0.084 / +0.057 with s.e. 0.04-0.06) is the only cell
above two s.e. and is not backed by the full set or by the level error:
S's MSE at C is 0.048-0.063 against a constant's 0.017 (corr 0.04-0.15),
and even N's is 0.023-0.026 (corr 0.10-0.15) -- the state-level value
itself does not carry to other episodes' poses.  The predeclared
positive-transfer condition (positive B and C pick gain AND an
improvement over NCE by paired uncertainty with consistent per-seed
direction) is not met at B or C.

**5. What has this ruled out, and what remains uncertain?**  Ruled out:
(a) that the B / C failure is specific to the NCE objective or its
readout -- a direct regression on the same inputs, data and weights fails
B and C in the same way (and worse in level); (b) that the training keys
lack learnable single-torque signal -- S reproduces fresh consequences at
its keys better than an eight-draw sample would suggest is needed
(agreement 0.86, gain 0.055 of 0.061); (c) that outcome noise at the
keys is the limit -- the fit survives independent draws.  Not ruled out /
uncertain: whether more torque samples per state, more episodes, a
different input representation, or conditioning that both learners lack
(absolute time / remaining horizon) would let the torque dependence
interpolate; whether A's success rests on near-duplicate states (the
reset pose recurs across episodes) rather than on a generalising
function; whether a level-calibrated readout would change any ranking
conclusion (the NCE comparison here is by ranking only).  N's ties and
its ~0 pick gain (-0.001..+0.0006) confirm the tie handling and the
uniform tie-breaking.  Nothing here says single-step torque control is
impossible, and nothing here says NCE is incapable -- only that with
these data and inputs the direct estimator is as key-bound beyond the
training keys as the critic.

## Files

`manifest.json`, `training_keys.npz`, `{S,N}_seed{0,1,2}/` (final.pkl,
train_summary.json with the curve, train_predictions.npy), `metrics.json`
(all tables incl. per stratum and paired bootstraps), `eval_predictions.npz`
(per-key predictions and the 16-draw means), `REPORT.md`, logs
`sup_*.log`.  Round-1, A/B/C and coverage results are untouched.
