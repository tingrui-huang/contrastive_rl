# Recoverable setbacks versus absorbing death

**The neural method meets the preregistered criterion.**

The previous benchmark remains unchanged as a regression. This five-state
extension retains H=4, gamma=.9, pi(1)=.75, b(1)=.25, normalized goal
reward, calibrated binary NCE, and alternating two-loss ETT optimization.
S emits progress P, setback R, or death D. P->G; R->P; G and D absorb.
All three emitted outcomes have immediate goal reward zero.

## Analytic distinction and certified optimum

At the decision-relevant remaining horizon 3, V(P)=.271, V(R)=.171,
V(D)=0. Thus R retains strictly positive future goal occupancy despite
the same immediate reward as D. At remaining horizon 0 or 1, R and D
tie at zero. Those ties are reported, not interpreted as uniquely bad death.

The four shared parameters define (d_P,d_R,1-d_P-d_R) plus
(x-x_prime)*(u,v,-u-v). Observed root diagonal law is (.4,.35,.25).
Both parameter pairs obey the declared hexagon TV constraints: diagonal
.02 and full distributional action Lipschitz .15. Every action/natural-action
cell uses this one kernel, marginalized over b. Projection enforces
constraints, with no additional penalty. See PROTOCOL.md for all pairs.

Certified J_min = **0.128254500**, exactly 256509/2000000.
A zero-sum TV-ball linear-functional bound proves this for the entire
declared family; the simultaneous witness attains both hexagon bounds.
Optimal root probabilities (P,R,D)=(.305,.35,.345). This is not a sampled
minimum and no certificate or exact Q was used in learning/selection.

Maximum death alone is NOT equivalent: (.4,.255,.345) has the same
death probability .345 but J=0.136804500, strictly larger.
The task objective also values the timing of surviving progress. It does
not require eliminating all setbacks or maximizing death as a substitute.

## Final iterates and last training critics

- neural_diagonal_s0: J=0.152198156, gap=0.0239; P/R/D=0.408170/0.342076/0.249754; V3(R)/V3(D)=0.182111/0.000016; Q RMSE=0.010884, gradient RMSE=0.009464.
- tabular_joint_s0: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.174403/0.000000; Q RMSE=0.003141, gradient RMSE=0.001712.
- neural_joint_s0: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.174804/0.000014; Q RMSE=0.002948, gradient RMSE=0.002167.
- neural_diagonal_s1: J=0.150864582, gap=0.0226; P/R/D=0.393062/0.357353/0.249584; V3(R)/V3(D)=0.171564/0.000019; Q RMSE=0.004180, gradient RMSE=0.001994.
- tabular_joint_s1: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.167125/0.000000; Q RMSE=0.003221, gradient RMSE=0.001949.
- neural_joint_s1: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.172249/0.000023; Q RMSE=0.004901, gradient RMSE=0.003109.
- neural_diagonal_s2: J=0.154557231, gap=0.0263; P/R/D=0.410328/0.353985/0.235687; V3(R)/V3(D)=0.166236/0.000033; Q RMSE=0.003279, gradient RMSE=0.002458.
- tabular_joint_s2: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.165802/0.000000; Q RMSE=0.004256, gradient RMSE=0.002615.
- neural_joint_s2: J=0.128254500, gap=-2.78e-17; P/R/D=0.305000/0.350000/0.345000; V3(R)/V3(D)=0.159557/0.000031; Q RMSE=0.009805, gradient RMSE=0.009281.

These are the actual final training critics, not post-training refits.
Exact V3(R)/V3(D)=.171/0 on every kernel. Goal probability by episode end
is P+R (both recover within four steps); D remains dead. Full horizon-wise
values, maxima, probabilities and empirical return intervals are saved.

## Common frozen-kernel refits

Each of nine final kernels supplies one independent set of 2,048 MC
positives/query to both the tabular MLE and three fresh neural refits.
Neural refits run 1,024 fixed Adam steps each. Samples and kernels are
identical within comparisons; no exact-Q targets or selected refits.

- neural_diagonal_s0: tabular Q/gradient RMSE 0.001632/0.000252; neural Q RMSE 0.001539, 0.001976, 0.001829; neural V3(R)-V3(D) 0.169021, 0.170898, 0.169557.
- tabular_joint_s0: tabular Q/gradient RMSE 0.001679/0.000252; neural Q RMSE 0.001968, 0.001945, 0.002374; neural V3(R)-V3(D) 0.168883, 0.171151, 0.170558.
- neural_joint_s0: tabular Q/gradient RMSE 0.001679/0.000252; neural Q RMSE 0.001968, 0.001945, 0.002374; neural V3(R)-V3(D) 0.168883, 0.171151, 0.170558.
- neural_diagonal_s1: tabular Q/gradient RMSE 0.001636/0.000164; neural Q RMSE 0.001362, 0.001472, 0.001330; neural V3(R)-V3(D) 0.172169, 0.173404, 0.171996.
- tabular_joint_s1: tabular Q/gradient RMSE 0.001578/0.000164; neural Q RMSE 0.001647, 0.001568, 0.001820; neural V3(R)-V3(D) 0.171798, 0.173520, 0.172579.
- neural_joint_s1: tabular Q/gradient RMSE 0.001578/0.000164; neural Q RMSE 0.001647, 0.001568, 0.001820; neural V3(R)-V3(D) 0.171798, 0.173520, 0.172579.
- neural_diagonal_s2: tabular Q/gradient RMSE 0.001110/0.001267; neural Q RMSE 0.001140, 0.001243, 0.001124; neural V3(R)-V3(D) 0.169108, 0.169719, 0.168932.
- tabular_joint_s2: tabular Q/gradient RMSE 0.001234/0.001267; neural Q RMSE 0.001567, 0.001392, 0.001627; neural V3(R)-V3(D) 0.169225, 0.170345, 0.169903.
- neural_joint_s2: tabular Q/gradient RMSE 0.001234/0.001267; neural Q RMSE 0.001567, 0.001392, 0.001627; neural V3(R)-V3(D) 0.169225, 0.170345, 0.169903.

All refit gradient errors and horizon-wise values are in results.json.
The 81 crossed last-training comparisons are separate: they include
transfer/staleness rather than an equal-data fitting comparison. Repeated
equal kernels and shared seed streams are not independent environments.

## Did critic error change update direction?

- neural_diagonal: raw gradient sign errors=0; progress-versus-setback contrast flips=0; projected step maximum L2 difference=0.01422; minimum defined projected-step cosine=0.9988182371770785; spurious nonzero steps at an exact stationary point=0.
- tabular_joint: raw gradient sign errors=0; progress-versus-setback contrast flips=0; projected step maximum L2 difference=0.00134844; minimum defined projected-step cosine=0.9999072803822193; spurious nonzero steps at an exact stationary point=0.
- neural_joint: raw gradient sign errors=0; progress-versus-setback contrast flips=0; projected step maximum L2 difference=0.00157242; minimum defined projected-step cosine=0.9997707738786066; spurious nonzero steps at an exact stationary point=0.

Gradient diagnostics compare the pessimistic direction with exact model
values. Projected steps compare full NLL+4J updates; for the diagonal-only
arm these are explicitly hypothetical joint updates, not its executed
control update. Zero projected steps have no direction and are excluded
from cosine minima. There is no visitation-gradient sampling error at S,
because S occurs only at t=0; future reward estimation remains Monte Carlo.

## Scope and reproducibility

The critic is the previous dot-product neural architecture, with only
input/state dimensions expanded: 11-32-Tanh-16 phi and 5x16 learned psi.
Uniform known negatives q=1/5, alpha=0, B=32; one-positive/(B-1)-negative
weighting and Q_h=(1-gamma^h)(B-1)q exp(f), Q_0=0. Stable softplus and
float64; no clipping, hand-coded goal/death values or reward changes.
Critics learn sampled truncated-geometric positives. ETT uses enumerated
categorical gradients with critic and visitation frozen, then refreshes.

Neural critics cannot express exact zero occupancy with finite logits.
Report small nonzero values on h=1 ties as calibration error, not evidence
that tied outcomes differ. The family has deterministic recovery and one
parameterized branching state; this does not validate hidden-state models,
arbitrary recovery dynamics or PointMaze integration. All conclusions are
within this declared family and fixed budget. No actor or loss redesign.

Training: 25,657,344 transitions, 39,168 neural steps.
Evaluation: 2,433,024 transitions, 27,648 refit steps.
Final iterates only. Configuration/protocol/source hashes were saved before
collection. Checkpoints stay local; English metrics/evaluation are public.

```powershell
python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback
python -m ett.finite_setback prepare --out artifacts/finite_crl/setback_fresh
python -m ett.finite_setback train --out artifacts/finite_crl/setback_fresh
python -m ett.finite_setback_eval --out artifacts/finite_crl/setback_fresh
python -m scripts.check_finite_setback --out artifacts/finite_crl/setback_fresh
```

Stop after reporting; no follow-up experiment or push is performed.

## Feasibility and verification

All 16 tests passed, including both previous finite-state regression suites.
Saved-data verification reproduced all 432 ETT updates and checked 441 parameter
states, common-sample refits, action/reward alignment, recovery, and absorption.
Maximum constraint violation was 2.7755575615628914e-17 (float64 roundoff).
All diagonal TVs remain <=.02 and all-action distributional TVs <=.15 within
the declared 1e-12 arithmetic tolerance. No improvement relies on a constraint
violation. Signed objective gaps near -2.78e-17 are rounding, not a lower optimum.

neural_diagonal_s0: final diagonal TV=6.97946978e-11, action TV=0.0163395392.
tabular_joint_s0: final diagonal TV=0.02, action TV=0.15.
neural_joint_s0: final diagonal TV=0.02, action TV=0.15.
neural_diagonal_s1: final diagonal TV=1.2648993e-11, action TV=0.0147063815.
tabular_joint_s1: final diagonal TV=0.02, action TV=0.15.
neural_joint_s1: final diagonal TV=0.02, action TV=0.15.
neural_diagonal_s2: final diagonal TV=2.66577316e-11, action TV=0.0286254761.
tabular_joint_s2: final diagonal TV=0.02, action TV=0.15.
neural_joint_s2: final diagonal TV=0.02, action TV=0.15.

The final neural joint critics estimate V3(R)=.174804, .172249, .159557;
corresponding V3(D) values are about .000014, .000023, .000031. All satisfy
the predeclared ordering and gap-accuracy criterion. Same-kernel refits reduce
some errors, but use a larger MC set and a fresh fixed fitting schedule; they
do not retroactively describe the critic used during ETT optimization.
Critic errors change step sizes and slightly rotate some projected updates;
zero raw-sign and contrast flips do not mean the gradients are identical.
The constrained optimum is reached despite these errors in this simple family.
