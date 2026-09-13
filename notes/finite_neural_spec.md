# Neural critic replacement: preregistered finite-state experiment

Reference commit: 3827930. Preserve its tabular implementation, all historical
artifacts, four-parameter affine ETT, pi(1)=.75, b(1)=.25, H=4, gamma=.9,
diagonal TV tolerance .02, full distributional action-TV Lipschitz bound .15,
and projected two-loss ETT update (diagonal NLL + 4*pessimistic gradient).
Import these directly; do not change benchmark, rewards, constraints or actor.
The original rational certificate and exact evaluator are evaluation-only.

## Model and fixed learning schedule

f(h,s,a,y)=phi(onehot(h),onehot(s),onehot(a))^T psi(y), h=1..4.
phi is Linear(10,32), Tanh, Linear(32,16), with ordinary PyTorch linear
initialization. psi is a learned 4-by-16 embedding initialized N(0,.1^2).
All weights and biases are trained. No output bias, normalized embedding,
temperature, analytical value branch, tabular residual, state-specific target,
goal/death masking, exact-Q supervision, or extra loss is used.

Binary NCE averages one positive and B-1 negatives per row, B=32, alpha=0,
q(y)=1/4. Reuse sampled truncated-geometric positives at every (h,s,a).
Aggregate positive counts without changing the empirical objective; integrate
the fixed negative expectation exactly as in the tabular reference. Train the
network with full-batch Adam (lr=.01, default betas/epsilon, no weight decay),
512 steps on refresh 0 and 128 on each of 47 later refreshes. Carry weights AND
Adam state across refreshes. This changes only the critic estimator/optimizer;
sampling, ETT objective, and ETT schedule remain the reference implementation.
The analytic tabular fit has zero optimizer iterations, with the same data.

Decode exactly Q_hat_h=(1-gamma^h)(B-1)q(y)exp(f_h), and define Q_0=0.
Do not normalize decoded Q across y, clamp values to a known range, inject
absorbing values, or replace E_pi Q by exp(E_pi f). Float64 CPU, one thread,
deterministic PyTorch operations; stable softplus computes binary NCE. No
gradient/logit clipping. Nonfinite losses/gradients/values or a logit above 600
halt the run rather than selecting a retry. The guard does not modify logits.
Finite neural logits cannot exactly represent zero occupancy (unlike the
extended tabular MLE); expose rather than mask that error.

Run seeds 0/1/2 in three arms: tabular_joint, neural_joint, neural_diagonal.
Use exactly the reference ETT initializations and NumPy streams, base 82000000.
Each arm runs 48 rounds, 512 positives per (h,s,a), 2,048 root paths for
visitation, and one ETT SGD step lr=.12 per round. Controls have off weight 0;
critic training and model samples are otherwise identical in budget.
Neural initialization seed is 83000000+seed and is paired across neural arms.
The three arms share random-number streams, not necessarily trajectories:
different kernels change outcomes. Record initializations and post-collection
RNG hashes; compare counts only where kernels coincide.

Freeze ETT during collection and critic fitting. Freeze critic weights and
visitation during each ETT update, using the existing enumerated categorical
gradient over all x_prime, execution actions and successors. Verify the network
hash does not change during that update. d receives both loss gradients; v only
the pessimistic term. Keep the goal fixed and retain original absorption and
reward/horizon handling. Save final iterates only, plus audit histories; no
oracle-based tuning, early stopping, selected checkpoints or budget expansion.

## Same-kernel evaluation and fixed budgets

After all nine final kernels/checkpoints are sealed, evaluate every last
training critic on all nine final kernels (81 crossed comparisons). The critic
is unchanged in these comparisons, so off-own-kernel errors include transfer
and staleness; they are not an equal-data capacity comparison.

For EACH of the nine frozen final kernels collect 2,048 NEW independent MC
positives per (h,s,a), using seed 85000000+the kernel's ETT seed (paired streams
across arms). On these SAME counts fit the analytic tabular critic and three
fresh neural critics, seeds 84000000+0/1/2, 1,024 Adam steps each at the same lr.
Refits start from the same seed-specific initialization on every kernel and do
not reuse warm training weights. Exact values never become fitting targets or
selection criteria. All refits stop at step 1,024. This complete 9-kernel x
(tabular + 3 neural refits) comparison separates empirical critic estimation
from which ETT kernel an arm learned. Identical final kernels are still kept
as declared paired evaluations, not counted as independent environments.

Also collect 16,384 root paths per kernel for empirical reward and goal/death
probabilities. Exact DP values and visits are computed ONLY in the separate
post-training evaluator. Report goal-only/all-label RMSE and maximum error,
off-gradient error with common exact visits (isolates critic error), errors
with the stored sampled visits (actual update error), empirical NCE gap to the
tabular fit, decoded mass/range errors and absorbing-state errors. No exact
diagnostic is used to choose models. Report the last training critics separately
from independent refits. Preserve signed roundoff in certified gaps.

Model sampling per arm matches the old experiment. Hard caps across nine runs:
21,300,000 training transitions (planned 21,233,664), 2,100,000 evaluation
transitions (planned 2,064,384), 39,168 training neural Adam steps and 27,648
post-training neural Adam steps. Refits share samples; no extra trajectories
per critic. Focused smoke tests use separate seeds and small bounded draws.
No outcome-driven amendments or reruns. Checkpoints remain local/ignored.

Success criterion: all three neural_joint final kernels are feasible to 1e-12
and have J-J_opt <= .001 (absolute normalized return), with tabular_joint as
the reference. This threshold is a reporting rule only, not early stopping.
If this fails, use saved calibration, empirical fitting gaps, gradient signs,
and actual trajectories of parameter updates to locate supported failure
points. Do not infer fundamental representation impossibility from a finite
optimization budget. Stop after the report; no alpha sweep, actor training,
PointMaze integration, loss redesign, commit or push.
