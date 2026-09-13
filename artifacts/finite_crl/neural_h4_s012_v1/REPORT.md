# Neural contrastive critic in the certified finite-state ETT experiment

**Success within the declared budget.**
The benchmark, actor, natural-action distribution, constraints, ETT update
and certificate from 3827930 are unchanged. Only the critic is replaced.
All exact values and certified gaps were computed after final kernels were sealed.

## Protocol and calibration

The shared critic is phi(onehot(h,s,a))^T psi(y), with a 10-32-Tanh-16
MLP and a learned 4-by-16 state embedding (944 trainable parameters).
It receives no exact Q or special goal/death value. CPU float64 Adam uses
lr=.01, 512 initial fitting steps, then 128 per refresh with warm weights
and Adam state. Each arm uses 48 ETT updates, the same original ETT
initialization, 512 sampled positives per query and 2,048 visitation paths.
Random streams are paired; trajectories can differ when kernels diverge.

NCE uses sampled truncated-geometric positives, alpha=0 uniform known
negatives, and one-positive/(B-1)-negative weighting with B=32. Counts
compress the exact empirical full-batch objective; they are not exact-Q
targets. Decode Q_h=(1-gamma^h)(B-1)q(y)exp(f); Q_0=0 by definition.
Stable softplus and float64 are the only numerical stabilizations.
No logits, gradients or decoded values are clipped/normalized. A guard
halts nonfinite values or logits >600 instead of silently changing values.
ETT uses expected Q, not mean logits. Its enumerated categorical gradient
and discounted visitation are the reference implementation. Critic weights
and visitation are frozen for each ETT step; no actor is trained.

## Final ETT results

Certified J_opt = 0.10402944750, exactly
41611779/400000000. Close means an absolute gap <=.001.

- tabular_joint_s0: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.002643, gradient RMSE=0.000322.
- neural_joint_s0: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.002763, gradient RMSE=0.000078.
- neural_diagonal_s0: J=0.138968221, gap=0.0349; goal=0.569775, death=0.430225; diagonal TV=0.000000, action TV=0.021755; last-training goal-Q RMSE=0.003141, gradient RMSE=0.000897.
- tabular_joint_s1: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.003940, gradient RMSE=0.004871.
- neural_joint_s1: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.006414, gradient RMSE=0.006797.
- neural_diagonal_s1: J=0.139279788, gap=0.0353; goal=0.571053, death=0.428947; diagonal TV=0.000000, action TV=0.023057; last-training goal-Q RMSE=0.002203, gradient RMSE=0.000483.
- tabular_joint_s2: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.002265, gradient RMSE=0.000056.
- neural_joint_s2: J=0.104029447, gap=-5.55e-17; goal=0.426525, death=0.573475; diagonal TV=0.020000, action TV=0.150000; last-training goal-Q RMSE=0.002450, gradient RMSE=0.000348.
- neural_diagonal_s2: J=0.138899478, gap=0.0349; goal=0.569494, death=0.430506; diagonal TV=0.000000, action TV=0.026645; last-training goal-Q RMSE=0.002223, gradient RMSE=0.001048.

These are the actual last training critics, queried on their final kernel;
errors include any final-update staleness. The full 81-pair cross-kernel
matrix is in results.json, separate from the refits below. Off-own-kernel
errors include distribution transfer, not just critic fitting quality.

## Same-kernel independent refits

For each of all nine final kernels, one new set of 2,048 MC positives/query
is shared by the tabular MLE and three fresh neural initializations.
Every neural refit runs exactly 1,024 Adam steps. No trained weight warm
start, exact target, selection, or subsequent ETT update is used.

- tabular_joint_s0: tabular goal-Q/gradient RMSE 0.001707/0.000056; neural seeds 0/1/2 goal-Q RMSE 0.001706, 0.001578, 0.001537; gradient RMSE 0.000306, 0.000344, 0.000091.
- neural_joint_s0: tabular goal-Q/gradient RMSE 0.001707/0.000056; neural seeds 0/1/2 goal-Q RMSE 0.001706, 0.001578, 0.001537; gradient RMSE 0.000306, 0.000344, 0.000091.
- neural_diagonal_s0: tabular goal-Q/gradient RMSE 0.001238/0.000221; neural seeds 0/1/2 goal-Q RMSE 0.001243, 0.001488, 0.001314; gradient RMSE 0.000133, 0.000435, 0.000334.
- tabular_joint_s1: tabular goal-Q/gradient RMSE 0.001209/0.000322; neural seeds 0/1/2 goal-Q RMSE 0.001586, 0.001369, 0.001306; gradient RMSE 0.000496, 0.000524, 0.000129.
- neural_joint_s1: tabular goal-Q/gradient RMSE 0.001209/0.000322; neural seeds 0/1/2 goal-Q RMSE 0.001586, 0.001369, 0.001306; gradient RMSE 0.000496, 0.000524, 0.000129.
- neural_diagonal_s1: tabular goal-Q/gradient RMSE 0.001356/0.001665; neural seeds 0/1/2 goal-Q RMSE 0.001393, 0.001566, 0.001550; gradient RMSE 0.001471, 0.001638, 0.001517.
- tabular_joint_s2: tabular goal-Q/gradient RMSE 0.001927/0.002669; neural seeds 0/1/2 goal-Q RMSE 0.002193, 0.001979, 0.001858; gradient RMSE 0.002893, 0.002599, 0.002425.
- neural_joint_s2: tabular goal-Q/gradient RMSE 0.001927/0.002669; neural seeds 0/1/2 goal-Q RMSE 0.002193, 0.001979, 0.001858; gradient RMSE 0.002893, 0.002599, 0.002425.
- neural_diagonal_s2: tabular goal-Q/gradient RMSE 0.001462/0.000528; neural seeds 0/1/2 goal-Q RMSE 0.001654, 0.001539, 0.001662; gradient RMSE 0.000774, 0.000799, 0.000740.

Q errors are computed against exact DP on every (h,s,a,y); goal-only
summaries are above, all-label and maximum errors are saved. Gradient
comparisons above use identical exact visitation to isolate critic error.
Per-round diagnostics additionally use actual saved sampled visitation.
Independent root-path return estimates and 95% MC intervals are saved;
exact finite-state objectives need no sampling interval. Repeated equal
kernels and refit seeds are not independent environments.

Maximum refit excess empirical NCE over the tabular optimum: 2.45803e-05.
Maximum refit goal-Q at death (true zero): 0.000200901.
Maximum refit decoded mass error: 0.00521246.
Positive empirical NCE gaps reflect finite neural optimization and possibly
function-class approximation. Finite dot-product logits cannot exactly
represent zero probabilities; no absorbing-state masking hides this.
The experiment does not identify a neural representation lower bound.

## Interpretation and stopping rule

All three neural joint runs reach the predeclared proximity threshold while feasible.
A small ETT gap can coexist with critic error: in this monotone, box-constrained
family, sufficiently accurate gradient directions can reach the same corner.
This is evidence for the loss implementation on this benchmark, not exact
neural calibration, causal identification, or readiness for PointMaze.
The topology sends every failed progress transition to death; it does not
test choosing death over delays or alternative routes. The allowed .02
diagonal error is part of the feasible family and may be fully used.

Training: 21,233,664 transitions and 39,168 neural steps.
Evaluation: 2,064,384 transitions and 27,648 refit steps.
All budgets and source hashes were saved before collection. Final checkpoints
remain local and ignored; public configuration, metrics and evaluation arrays
are saved here. No actor training, loss redesign, further experiments or push.

```powershell
python -m unittest scripts.test_finite_crl scripts.test_finite_neural
python -m ett.finite_neural prepare --out artifacts/finite_crl/neural_fresh
python -m ett.finite_neural train --out artifacts/finite_crl/neural_fresh
python -m ett.finite_neural_eval --out artifacts/finite_crl/neural_fresh
python -m scripts.check_finite_neural --out artifacts/finite_crl/neural_fresh
```

## Verification

All 11 focused tests passed. Saved-data checks replayed all 432 ETT updates,
checked 441 feasible parameter states, 36 common-kernel refits and checkpoint
predictions, and confirmed identical paired RNG streams. All three tabular
trajectories of ETT parameters exactly reproduce the reference checkpoints.
Maximum feasibility excess is 2.78e-17; signed gaps of -5.55e-17 are float64
roundoff, not improvement over the certificate. No new simulator steps were
used by the checker. All 42 ETT/critic/refit checkpoint files remain ignored
and local; hashes are in local_checks.json.
