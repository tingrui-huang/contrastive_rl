# Recoverable setback versus death: fixed pre-run protocol

Reference: 1fb5622. Preserve previous benchmark files/artifacts as regressions.
The sole task extension is a fifth state and a coupled three-outcome root law.
States S,P,R,G,D denote start, progress, recoverable setback, goal and death.
H=4, gamma=.9, reward=(1-gamma)1{next=G}, fixed pi(1)=.75 and b(1)=.25.
S transitions to P,R,D; P->G, R->P, G->G, D->D deterministically. Continue to
the original horizon after absorption. Neither P,R nor D gets immediate goal
reward when entered from S. All critic values are learned; deterministic rows
are simulator dynamics, not injected critic labels.

## Coupled feasible family and certificate specification

Four parameters theta=(d_P,d_R,u,v). At S, probabilities of (P,R,D) are
(d_P,d_R,1-d_P-d_R)+(x-x_prime)*(u,v,-u-v). All four (x,x_prime) cells share
theta; intervention always averages this SAME kernel with fixed b. Diagonal
observed probabilities at S, both actions, are m=(.4,.35,.25), equivalently
exact frequency counts (40,35,25). Other observed diagonal rows are the stated
deterministic simulator rows. Diagonal NLL is mean over the two learned S
diagonal action laws; fixed rows have zero NLL and no fitted parameters.

Let hex(z)=max(|z0|,|z1|,|z0+z1|). Require
hex((d_P,d_R)-(.4,.35))<=.02 and hex((u,v))<=.15.
These are exactly the root diagonal TV tolerance and full action-TV constraint
TV(T(.|s,x1,x_prime),T(.|s,x2,x_prime))<=.15|x1-x2|, for all states, both natural
actions, and all execution-action pairs. State metric is the discrete metric;
TV equals its Wasserstein-1 distance. This is DISTRIBUTIONAL, not samplewise,
and imposes no bound in x_prime. Every probability is at least .25-.02-.15=.08
at the root; simplex validity follows without extra probability clipping.
Project each 2-vector onto its convex hexagon by the nearest point among all
six edges if outside. This is Euclidean projection, not a third training loss.

Certificate derivation is declared here but evaluated only after training.
V_h(P)=1-gamma^h; V_h(R)=gamma*(1-gamma^(h-1)) for h>=2 and V_1(R)=0;
V_h(D)=0. At the root the remaining successor horizon is 3, so V_3(R)=.171>0,
V_3(P)=.271, V_3(D)=0. Setback and death have identical immediate reward zero,
but distinct continuation values. At h=0 or 1, R and D DO tie: no claim of
unique worst outcome applies there.

Root coefficients c=(gamma*(1-gamma^3),gamma^2*(1-gamma^2),0)
=(.2439,.1539,0). Marginalized outcomes are m+delta+.5*(u,v,-u-v), so J=c dot
that vector. For every zero-sum vector with TV<=r, c dot vector >=
-r*(max(c)-min(c)). Hence the exact global lower bound over this SAME product
of hexagons is c dot m-(.02+.5*.15)*.2439. It is attained simultaneously by
d=(.38,.35), (u,v)=(-.15,0). A rational evaluator certifies the bound/witness,
not a sampled search. It yields P=.305,R=.35,D=.345 and J=.1282545.

Maximizing death is NOT equivalent: moving the same total .095 from R to D
instead yields (P,R,D)=(.4,.255,.345), with the SAME maximum death probability
and strictly larger J=.1368045. Maximum-death solutions form a tied segment;
the reward objective prefers removing faster progress. Likewise, transferring
progress mass to R instead of D is strictly less pessimistic at this horizon.
The critic must learn that R retains value. No death classifier, penalty,
distance, shaping, new actor or oracle targets enter either training loss.

## Learning, pairing, budget, and evaluation

Three seeds, arms neural_diagonal, tabular_joint, neural_joint. Minimize root
diagonal NLL plus 4 times the existing discounted visitation one-step gradient
surrogate; control has weight 0. Exactly enumerate action/natural-action and
successor mass gradients with critic and visitation frozen. Recompute critic
and visitation before each update. Goal G and actor never change.

Retain 48 projected ETT SGD updates, lr=.12. At each refresh, 512 MC
continuations per (h=1..4,s=0..4,a=0..1) query plus 2,048 root visitation paths.
One positive per complete continuation, offset k weighted by gamma^(k-1)
truncated at h. Known fixed negatives q(y)=1/5, alpha=0, B=32; one positive and
B-1 negative weighting. NCE is the exact empirical full-batch count-weighted
softplus objective with analytically integrated negative expectation.
Tabular MLE uses the same counts, not exact model Q.

Neural phi: onehot(h,s,a), width 11 -> Linear(32) -> Tanh -> Linear(16).
psi: learned 5x16 embedding initialized N(0,.1^2), ordinary linear initialization.
Same architecture/schedule as prior apart from required state dimensions:
Adam lr=.01, default betas/epsilon, no weight decay; 512 first-fit updates and
128 each later refresh, carrying weights and optimizer state. Float64 CPU,
one thread; stable softplus; no logit/value/gradient clipping or probability
normalization. Unsafe/nonfinite computation halts without retries. Decode
Q_h=(1-gamma^h)(B-1)q(y)exp(f_h); Q_0=0. No hand-coded absorbing values.

Initialization: base 86000000+seed, d=m[:2]+Uniform[-.01,.01]^2,
(u,v)=Uniform[-.03,.03]^2, then hex projection. Paired identical initial theta
and random streams across arms; after kernels diverge do not claim identical
outcomes. Refresh RNG base+10000+1000*seed+round; neural seed 87000000+seed.
Select only final iterates. No oracle-based tuning or early stopping.

After all nine kernels are frozen, independent MC positives (2,048/query) on
each kernel, seed 88000000+ETT seed, are shared by one tabular and three fresh
neural refits (seed 89000000+refit seed, 1,024 Adam steps). Keep last training
critics separate from these refits. Also 16,384 root paths/kernel. Compare
Q, V_h(R)-V_h(D), raw gradient components, progress-versus-setback gradient
contrast, and projected two-loss update direction using common exact visits.
Report per-round differences with actual saved visitation as well. All exact
values and certificate are evaluation-only; no resulting diagnostic affects
training or refitting. Inspect h=1 ties separately from h=3 ordering.

Fixed total caps: 25,700,000 training transitions (planned 25,657,344),
2,440,000 evaluation transitions (planned 2,433,024), 39,168 training neural
steps, 27,648 refit steps. Smoke tests use separately seeded bounded draws.
No outcome-driven collection/budget expansion. Success requires all neural
joint seeds feasible within 1e-12, final certified gap <=.001, and their LAST
training critic correctly ordering V_3(R)>V_3(D) with gap error <=20% of .171.
These thresholds only classify final results. Report failures/ties honestly.
Keep checkpoints local, publish English metrics/report, stop without push.
