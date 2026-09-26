# Step 2 (simulator-paired one-step ETT): F(s, a_b, a_q) -> s' + onset head, cross-fitted -- ALL PREFLIGHT GATES PASSED

Oracle-supervised engineering stage (the advice comes from the privileged
teacher walked along the simulator branches, Steps 0-1); not offline
identification.  `manifest.json` (sealed before fitting: model, budget,
selection, gates), `CHECK.md` / `check.json`, `model_fold*.json` (full
validation curves); the parameters stay on the nodes.

## Model and fit

Inputs: standardised state (29), a_b (8), a_q (8), a_q - a_b, a_q * a_b.
Motion MLP 1024-1024 -> standardised delta state (MSE; diagonal and
off-diagonal rows averaged separately, equal weight).  Onset MLP 256-256
on the same features + the predicted delta (stop-gradient) -> logit;
natural-prevalence BCE from class-stratified batches (256 + 256); its
gradient never reaches the motion model.  Adam 3e-4, 40,000 updates,
motion batch 2,048 diagonal (half branch-diagonal, half logged) + 2,048
off-diagonal rows.  Cross-fitted: model f trains on folds != f (3 folds
by source episode) minus a 10 % early-stop slice of its training
episodes, selection = earliest minimum of mse_diag + mse_off + 0.25 *
onset_bce (best at 39k / 40k / 40k: the score was still falling at the
end of the pre-registered budget; not extended).  Wall 145-172 s per
fold on one GPU.

## Preflight on the held-out fold rows (2.53-2.57 M valid transitions per fold; pooled)

| gate | threshold | value | pass |
|---|---|---|---|
| G1 one-step xy error median / p99 | < 0.01 / < 0.10 | 0.0016 / 0.027 | yes |
| G1 standardised delta RMSE (29 dims) | < 0.30 | 0.157 | yes |
| G2 open loop with the recorded (a_b, a_q): xy error at step 10 / 50 (median) | < 0.2 / < 1.0 | 0.060 / 0.47 (p90 at 50: 1.1-1.2) | yes |
| G3 onset AUROC | >= 0.85 | 0.996 | yes |
| G4 mean P(onset) on onsets / on non-onsets | >= 5 x | 0.22-0.25 / 0.0013-0.0019 = 139 x | yes |
| G5 calibration: mean prediction / prevalence | in [0.5, 2] | 1.05 | yes |
| G6 advice sensitivity inside the bands: P(onset) at off_hold rows / at drive rows | >= 2 x | 0.21-0.24 / 0.000 | yes |
| parameter deltas | > 1e-4 | motion 46-48, onset 23-24 | yes |

Per fold the one-step xy error median is 0.0016 (diagonal rows 0.0025,
off-diagonal 0.0016), delta RMSE 0.15-0.16, 50-step xy median 0.41 /
0.50 / 0.52.

## Reading

* Motion: a one-step error of ~0.002 maze units against a per-step
  displacement of ~0.1, and 0.47 after 50 open-loop steps.  Adequate for
  short horizons; the drift over hundreds of steps (the branches run to
  reach / death / the horizon, median 83 rows, timeouts ~700) is NOT
  validated by G2 and must be checked before any future is generated
  (Step 3 gate: outcome and route agreement of full-length open-loop
  rollouts against the held-out branches).
* Onset: the head has learnt what Step 1 showed -- a death is possible
  only while the teacher's advice is "hold" (inside the bands P(onset)
  0.21-0.24 under off_hold vs 0.000 under a driving advice; overall
  prevalence 0.2 %, predicted 0.21 %).  So in a rollout the deaths are
  decided by the ADVICE PROCESS: whatever supplies a_b along the path
  decides where the model can die.  This is the design question of Step
  3, not a model defect: the reconstructed advice inside a band is the
  teacher's mouth latch carried along a path the teacher itself never
  takes; an offline, memoryless nominal P(a_b | s) fitted on the log
  cannot produce "hold" inside a band (the log never shows it), a latched
  nominal (a mouth decision with the log's visible hold durations) can,
  and the simulator teacher can (oracle).  To be decided before Step 3.
* Diagonal non-inferiority: the off-diagonal rows are fitted slightly
  better than the diagonal ones (0.0016 vs 0.0025 xy median): the
  diagonal rows are the 5 %-weight logged rows + 44,580 roots, the bulk
  of the data is off-diagonal.

Status: the model passed its pre-registered gates; nothing downstream has
been run.
