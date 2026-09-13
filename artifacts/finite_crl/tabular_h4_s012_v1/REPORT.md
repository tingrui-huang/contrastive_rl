# Finite-state Contrastive RL pessimistic-loss validation

This experiment jointly fits the diagonal and minimizes a calibrated
Monte Carlo contrastive continuation objective inside a declared feasible
four-parameter ETT family. It does not train an actor or use PointMaze data.

## Certified benchmark and implementation

The certified feasible return minimum is **0.104029447**
(41611779/400000000 exactly). The analytic witness is d=(.68,.78),
v=(-.15,-.15), with intervention success probabilities (.605,.705).
Its goal probability is 17061/40000; death is
22939/40000. Both stages and both natural-action
cells use one simultaneously feasible shared kernel. The certificate is a
rational monotonicity proof over the complete box, not sampled search.
It also certifies the same optimizer for diagonal NLL + 4J; its numerical
value is 0.972812118. See PROTOCOL.md for the proof and limits.

The horizon is 4; gamma=.9; rewards are .1 after each transition into G.
Goal and death are absorbing, with the original horizon retained. Actor
pi(1)=.75 and natural b(1)=.25 remain fixed. Every diagonal error is bounded
by TV .02; every action pair at fixed state/natural action obeys TV <=
.15 times action distance. This distributional bound is not samplewise.
It provides no general physical or causal guarantee.

Both arms use identical seed-specific initialization, 48 rounds and equal
sampling budgets. d receives both loss gradients in the joint arm; v has
zero diagonal derivative and receives the pessimistic gradient. No diagonal
backbone is frozen. Projection enforces constraints at every update.
Categorical successors, actor actions and natural actions are enumerated
when differentiating the frozen-critic one-step surrogate. Visitation is
held fixed and weighted by gamma^t; with exact Q this derivative equals
the full finite-horizon return gradient. The surrogate value is not J.

## Results (individual seeds, final iterate)

- diagonal, seed 0: J=0.138968221; feasible gap=0.034938773; joint-objective gap=0.138694128; diagonal TV=0.000000; goal=0.569775; death=0.430225; critic goal RMSE=0.001660, max=0.005497; all-label RMSE=0.001280; gradient RMSE=0.000071.
- joint, seed 0: J=0.104029447; feasible gap=-0.000000000; joint-objective gap=-0.000000000; diagonal TV=0.020000; goal=0.426525; death=0.573475; critic goal RMSE=0.001503, max=0.004829; all-label RMSE=0.001232; gradient RMSE=0.000672.
- diagonal, seed 1: J=0.139279788; feasible gap=0.035250341; joint-objective gap=0.139940398; diagonal TV=0.000000; goal=0.571053; death=0.428947; critic goal RMSE=0.001215, max=0.003628; all-label RMSE=0.001265; gradient RMSE=0.000066.
- joint, seed 1: J=0.104029447; feasible gap=-0.000000000; joint-objective gap=-0.000000000; diagonal TV=0.020000; goal=0.426525; death=0.573475; critic goal RMSE=0.002168, max=0.008739; all-label RMSE=0.001779; gradient RMSE=0.000206.
- diagonal, seed 2: J=0.138899478; feasible gap=0.034870031; joint-objective gap=0.138419158; diagonal TV=0.000000; goal=0.569494; death=0.430506; critic goal RMSE=0.001387, max=0.004669; all-label RMSE=0.001014; gradient RMSE=0.000761.
- joint, seed 2: J=0.104029447; feasible gap=-0.000000000; joint-objective gap=-0.000000000; diagonal TV=0.020000; goal=0.426525; death=0.573475; critic goal RMSE=0.001992, max=0.007812; all-label RMSE=0.001515; gradient RMSE=0.002785.

Independent MC paired joint-minus-control return estimates:
- Seed 0: -0.034745, 95% MC interval [-0.03618323514964525, -0.033306828326917226].
- Seed 1: -0.034447, 95% MC interval [-0.03587411652858439, -0.033020487963603094].
- Seed 2: -0.034403, 95% MC interval [-0.03581682482868147, -0.03298846081584976].

Maximum final feasibility excess: 2.78e-17. All training iterates
are also checked and saved in training.json. Improvements are rejected if
any feasibility excess exceeds 1e-12. MC intervals describe simulator
sampling error within a seed, not uncertainty across environments or
training distributions. Exact finite-state results need no sampling CI.

## Critic calibration and error attribution

The repository binary NCE loss averages one positive and B-1 negative
entries. Here negatives have a fixed known distribution rather than the
other batch rows' achieved-state marginal. Their expectation is integrated
exactly. The fitted saturated table is the empirical NCE MLE, computed
from sampled truncated-geometric positives. It is not fitted to exact
Q or TD targets. Zero empirical cells use the extended -infinity-logit
MLE, stored as finite zero odds, without arbitrary clipping.

The stationarity equation gives Q_hat_h=(1-gamma^h)(B-1)q_alpha exp(f_h).
The (B-1), known negative density and remaining-horizon mass all matter.
Expected Q is averaged in probability/value space, never logit space.
On the exact same final MC counts alpha=.5 correctly decoded agrees
with alpha=0 to 2.78e-17.
This is a controlled calibration identity, not evidence that failure
negatives improve representation or optimization. Decoding alpha=.5
with q_0 incorrectly doubles the goal prediction; errors are saved.

Finite MC counts cause critic and gradient estimation error. Saturated
NCE fitting has no iterative fitting error and no finite-support value
representation error (extended logits are permitted). Some zero-count
cells can reflect sampling zeros as well as truly unreachable states.
The feasible-objective gap measures final transition optimization error
inside the declared family. Its exact minimizer is representable, but
this affine two-stage topology excludes alternative delays, routes,
state-dependent policies, hidden-state mixtures and general kernels.
There is no claim of zero representation error relative to such models.

## Decision and reproducibility

All three joint runs reach the certified feasible optimum within numerical tolerance.
This validates the calibrated loss/gradient implementation on this small
fully observed example. Stop here. It does not establish reliable neural
critic extrapolation, conditional ETT identification, the true environment's
worst case, or readiness for large PointMaze/actor-training experiments.

Training used 14,155,776 transitions (cap 15,000,000);
independent evaluation used 1,376,256 (cap 1,500,000).
Source/config/protocol hashes were saved before training. All six final
checkpoint hashes were verified before invoking the separate oracle.
Checkpoints are local and ignored; configuration, metrics, predictions,
trajectories and this English report are publication-ready. No push.

```powershell
python -m unittest scripts.test_finite_crl
python -m ett.finite_crl prepare --out artifacts/finite_crl/fresh_run
python -m ett.finite_crl train --out artifacts/finite_crl/fresh_run
python -m ett.finite_crl_eval --out artifacts/finite_crl/fresh_run
```

Prior implementation/report references and the full estimator derivation
are in PROTOCOL.md and provenance.json. Background: [Eysenbach et al.
(2022)](https://arxiv.org/abs/2206.07568); the finite-horizon and B-1
normalization here is derived for the explicitly declared negative sampler.

## Artifact verification

Seven focused unit tests passed, including a two-round training smoke test.
The saved-data checker passed for 294 parameter iterates and all six runs,
with no new model queries. Signed objective gaps near -6e-17 are retained
as float64 evaluation roundoff, not clipped or interpreted as beating the certificate.
Both joint diagonal errors reach the allowed .02 boundary; the lower return
therefore includes the permitted diagonal-fit tradeoff, not exact diagonal preservation.
The topology forces every failed progress transition into D, so greater death
selection here does not show that the loss would choose death over delay in a richer family.

The main critic errors above use the predeclared post-training MC refresh
(2,048 fresh positives/query), with ETT fixed and no subsequent updates.
The following errors instead evaluate the actual LAST TRAINING critic
(512 positives/query, before the last ETT update) against exact values
of the final kernel; they include any last-update staleness:

- diagonal, seed 0: goal RMSE=0.003319, maximum absolute error=0.014808.
- joint, seed 0: goal RMSE=0.002643, maximum absolute error=0.010236.
- diagonal, seed 1: goal RMSE=0.002619, maximum absolute error=0.008798.
- joint, seed 1: goal RMSE=0.003940, maximum absolute error=0.014418.
- diagonal, seed 2: goal RMSE=0.001415, maximum absolute error=0.003903.
- joint, seed 2: goal RMSE=0.002265, maximum absolute error=0.008221.

Per-round errors of all training critics are in results.json.

```powershell
python -m scripts.check_finite_crl --out artifacts/finite_crl/fresh_run
```
