# Two geometry modes under a fixed actor: pre-execution specification

Starting/requested commit: be1b3c520ce91ea4a7e9c4d9f4a7fa0941e86345.
Use rectangle and fork_segment with optimizer seeds 0,1, all 32 coefficients
exactly zero initially. Preserve the previously specified signed matrix gates,
Frobenius normalization, L=1, diagonal law and geometry construction. The segment
instance is not a pointwise superset. The optional construction is fixed, with
no route/action/outcome-dependent tuning. No downstream CRL/native rollout.

Reuse the original adversarial experiment's stochastic actor, state-goal nominal,
guarded diagonal checkpoint/normalizations and commanded goal. x_prime is sampled
from the nominal, x from the actor, and then the ETT successor; policies respond
to each model's own generated states. Hidden state, failure labels, readouts and
native outcomes never enter training/evaluation. The actor checkpoint contains
critic state, but only actor parameters are executed and nothing in it is updated.

F=L_diag+lambda E[J], J=sum(t=0..49) .95^t 1[||XY_next-(8.5,3.5)||<2],
lambda=1/sum(.95^t). L_diag is the unchanged emitted diagonal energy score, a
constant in theta; it is not joint shared-network diagonal learning. Measure it
on the first 1024 train and 1024 validation rows in the original saved fit contexts
with K=16 fresh coupled draws. This smaller diagnostic estimator does not change
the population objective or optimization differences. No other loss is added.

Exactly 16 updates, 8 normal directions/update, 32 paths per signed query,
sigma=.1, learning rate=1, update L2 cap=.2. Reuse the original antithetic
finite-perturbation gradient and descent helpers; never differentiate hard reward
or use an unavailable emitted likelihood. Pair directions and all rollout keys
across modes within each optimizer seed (and +/- signs). Each query generates its
own states/actions; common noise does not mean native hidden randomness or fixed
nominal actions at different states. Report final iterate only, never monitor-best.

Fresh seed namespaces, checked before preparation against existing code and
configuration/seed artifacts: directions 61000000+seed; optimization keys
62000000+10000*seed+8*iteration+direction; monitor keys
63000000+100*seed+iteration (0..16); evaluation groups 64000000..64000003;
diagonal keys 65000000,65000001; component nominal/anchor keys 65000100,65000101;
initial matched rollout 66000000; bootstrap 66000100; frozen policy probes
66000200. These phases are independent. Presave all exact schedules and hashes.

## Declared transition budget (before any new sampling)

* Optimization: 4*16*8*2*32*50 = 1,638,400.
* Monitoring initial/final and every iterate: 4*17*32*50 = 108,800.
* Evaluation common zero baseline and four finals: 5*4*128*50 = 128,000.
* Matched zero-initialization smoke: 2*8*50 = 800.
* Diagonal checks: 2 zero modes + 4 reloaded finals, each 2048*16 = 196,608.
* Replay of first evaluation group from each reloaded final: 4*128*50 = 25,600.
* One common component anchor panel: 64*8 = 512.
* Planned total: 2,098,720. Hard cap including any failed/repaired attempt:
  2,500,000; remaining 401,280 is defect-repair headroom, not extra experiments.

Ledger reserves each sampler call BEFORE execution; failures remain charged.
Updates/progress and failures are persisted. Resume only missing work without
rerunning completed optimization. Preserve and document any reproducible defect,
source repair and charged failed samples; never restart a run invisibly. No
result-dependent budget extension, checkpoint selection or hyperparameter change.
Deterministic component projections on saved anchors and stub tests draw no new
model transitions. Every actual new anchor/rollout draw is counted above.

## Independent evaluation and explanation

Use four independent groups of 128 paths, same keys across all five models.
Report each final minus the shared zero baseline and fork_segment minus rectangle
separately for optimizer seeds 0 and 1. Use 2000 paired episode-bootstrap replicates
stratified within the four groups; never treat steps or optimizer checkpoints as
independent episode replicates. Report diagonal and weighted/unweighted adversarial
terms separately. These intervals are conditional on the frozen inputs and trained
coefficients, not broad optimizer uncertainty.

Predetermined trajectory figure: group 64000000, path indices 0,1,2,3, for all
models. Diagnose eligible fork contexts (the selector's availability flag),
current XY in the supported fork region, actual segment selection, anchor motion,
matrix correction, emitted motion, projection fraction (exact 0/1 mass), zero
returns, boundaries and visible stationary histories/anchors. Stationarity is not
confirmed death. Audit membership in the selected segment or fallback box.
The twelve old float32 anchor violations of literal real unit steps are preserved;
use the original rounded q+/-1 envelope and 2e-6 floating residual tolerance.

To separate response from state-distribution shifts, evaluate all four learned
theta arrays in BOTH modes on the same newly sampled 64x8 component panel and
the existing saved 225 fork/descent anchor contexts. No modified component output
is fed forward. Check fixed-set invariance and finite action pairs on those panels,
including near-pairs; they support, not replace, the existing continuous proof.
For any fixed conditioning C contains A and does not depend on x, ||M||op<=1 and
convex projection is nonexpansive. This preserves the full diagonal law and the
all-action-pairs bound, not native kinematics/absorption/causal identification.

New coefficient checkpoints must persist format version, geometry mode, bound and
theta; reload evaluation requires the metadata. Historical theta/bound checkpoints
remain rectangle-compatible. Save anchors, corrections, proposals, successors,
set identity/endpoints/fractions, reference boxes and correction/boundary indicators.

Conclude whether neither/both modes establish return decrease, whether the mode
difference is established, and whether mechanism or numerical/degeneracy concerns
make downstream use premature. Lower model return is an adversarial objective
improvement, not actor improvement or proof of useful synthetic data. No downward
turn or route-completion acceptance criterion. End with a justified decision on a
later matched offline-only/rectangle-samples/fork-segment-samples CRL comparison;
do not run it here.
