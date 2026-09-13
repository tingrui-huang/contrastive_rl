# PointMaze fixed-goal region-NCE integration pilot

Start from 9f17c4e; preserve all finite-state implementations as regressions.
No production policy, critic, generator source, normalization or old checkpoint
is overwritten. This is a bounded estimator/update transfer test, not certified
global pessimism, conditional ETT ground truth, or causal identification.

## Actual generator, distributions and inherited constraints

Use the checkpoint and sampler resolved by ett.run_convex_adversarial.setup:
ConvexActionTransition with rectangle geometry and samplewise L=1, around the
guarded K3 diagonal law from s0_L0p25_lambda0/final.pkl. That older filename's
.25 is not this wrapper's L. The diagonal network is a 20-128-128-16 MLP taking
(F4 state[8], x_prime[2], x_prime[2], commanded F4 goal[8]). It emits one atom
logit, three moving-mixture logits, six locations and six scales. Gaussian
displacement samples/atom are passed through the existing coordinate-step,
maze-boundary and wall postprocessing; older F4 frames shift deterministically.
The old mixed NLL applies before projection; it is not an emitted F4 density.
Retain emitted XY energy score as the convex generator's diagonal fitting
objective (unbiased pairwise U-statistic), and report pre-projection NLL too.

The expert/nominal K5 tanh Gaussian mixture takes state AND goal. It fits the
established noisy teacher-source population, including failures, not universally
optimal demonstrations. The actor is the historical alpha=0 seed=0 checkpoint
at step 150,000, stochastic tanh Gaussian, designated before outcomes. Both
remain frozen. At every step independently sample x_prime~b and x~pi, then
sample this one kernel. Only x is stored as executed action. Do not optimize
individual x_prime cells independently or substitute a bank distribution.

Generator action response is M(s)(x-x_prime), with eight state-gated 2x2
matrices bounded by Frobenius norm 1. A convex rectangle containing each
diagonal anchor is selected independently of x, then anchor+response is
projected onto it. At fixed s,g,x_prime and shared atom/component/noise draw,
arbitrary action pairs obey ||G(x1)-G(x2)||_2 <= ||x1-x2||_2. F4 old frames
cancel. This is samplewise Euclidean L=1 and implies a coupled Wasserstein
bound; it is NOT the toy discrete TV guarantee and does not bound q logits.
Changing diagonal parameters preserves the proof because those parameters
never introduce x into the diagonal conditioning or geometry selection.

Keep exactly the existing rectangle selection and projection. They already
exclude some valid transitions near forks, as documented in the previous
geometry/structure audits; this pilot neither adds exclusions nor claims full
reachability. No death gate, persistent absorption constraint, selected upper
corridor, new convex set, Lipschitz penalty or toy TV constraint is added.
The model carries only visible F4, so native death is not a defined model event.

Train a declared 48-coordinate subspace: additive offsets to all 16 existing
distribution-head biases and the 32 existing response coordinates. Other
diagonal weights and all normalizations stay fixed. This is a limited search
capacity, not a full-network training claim. The same updated distribution head
serves diagonal and every off-diagonal query. Both losses update those biases;
response coordinates have zero diagonal derivative. Initial offsets/response
are zero and exactly reproduce the historical convex diagonal sampler.
Set that structurally zero derivative exactly to zero in the estimator; this
removes cross-coordinate Monte Carlo noise and fixes control response at zero.

## Population, horizon and reward

Use the existing p=.3 F4 dataset. Teacher-source episodes [1200,6000), with the
original seed-0 660-episode holdout; actor training already used these episodes.
The stored arrays have 51 observations and 51 action slots per episode; the
last action is an unused zero dummy. There are exactly 50 actual transitions.
At t=40 choose 32 currently outside and 32 inside the existing goal region in
each partition, uniformly without replacement using seeds 90000000/1. Selection
reads obs only, never audit labels or future states/returns. This is a deliberately
balanced late-task context distribution, not a reset-policy estimate. All F4
histories are actual recorded histories, not fabricated or teleported states.
Reuse first 512 rows/partition of the existing convex fit-context file for
diagonal fitting/evaluation. Standardize critic state using train rows only;
std floor .1. Policies and generator retain their original normalization.

H=10 is the exact remaining time to the task's original 50-step horizon from
t=40, gamma=.95, commanded goal=(8.5,3.5) tiled four times. The existing reward
is 1[||newest XY-goal XY||<2]; evaluate normalized occupancy .05*sum gamma^t r.
Native deaths lie outside this fixed region, so position rewards agree on
reachable states apart from the documented float32 boundary limitation.
No new reward/shaping is introduced. Model membership uses the existing JAX
float32 task_reward function throughout. Do not generalize to other goals.

## Binary region adapter and pre-optimization gate

Production CRL uses achieved full-F4 future goals and batch-dependent point-goal
negatives. Its raw f is not a calibrated probability of this positive-area XY
region; neither continuous density q nor its integration is known here. Leave
that critic unchanged. Instead label sampled future states by the existing
reward-region membership c in {0,1}. This is an isolated region-occupancy
critic, not the F4 point-goal embedding or a death classifier.

For each sampled model path and each t, draw one future offset k=1..h with
probability (1-gamma)gamma^(k-1)/(1-gamma^h), h=10-t. The label is the region
membership of s_(t+k). Condition f on state, actual executed action, and h.
Known negative labels are uniform, q(0)=q(1)=.5, alpha=0. Minimize binary NCE
[E_positive softplus(-f)+(B-1)E_q softplus(f)]/B, B=32. Analytically integrate
negative expectation, retain class-prior weighting. At population optimum,
exp(f_c)=p_h(c|s,a)/((B-1)q(c)), hence
Q_h(s,a,region)=(1-gamma^h)(B-1)*.5*exp(f_1). Set Q_0=0.
There is no claim q=.5 is a density on continuous states: it is on binary labels.
Do not average logits, normalize decoded values or insert absorbing values.

Use phi MLP 11-64-Tanh-64-Tanh-16 (normalized F4, native action, h/10), learned
2x16 psi with N(0,.1^2) initialization, dot-product f. Float64 PyTorch CPU,
one thread, stable softplus; Adam .003/default betas/epsilon, no weight decay,
no value/logit/gradient clipping. Halt nonfinite predictions or logits >600.
Initial 1,500 minibatch-256 steps on 64x8 complete model paths; 400 steps per
refresh, warm weights and optimizer. One fixed preflight initialization is
cloned across arms; collection and minibatch streams are paired within seed.

BEFORE ETT optimization: independently select query histories from 64 held-out
root model paths at t=0,5,9 (remaining h=10,5,1). At each query sample ONE fixed
first actor action, score that action, then 16 independent complete model
continuations with that first action held fixed, later actions stochastic.
Keep nominal/transition draws independent. Compare predicted Q with mean MC
return; report MC standard errors. Require every h RMSE<=.06 and |bias|<=.025,
outside-region RMSE<=.07 where populated, and h10 between-context return
standard deviation>=.025. If the gate fails, STOP before ETT updates and
report estimator failure; no architecture/schedule retry. The same criteria
are reported on final kernels but never used to choose a final checkpoint.

## Paired constrained updates and budgets

Seeds 0/1, diagonal-only/joint, six updates each. Both start with zero ETT
offsets and the exact same preflight critic/Adam state. Every round collect
64 train roots x8 full continuations and fit the critic for 400 steps. Freeze
critic parameters and this empirical visitation while updating ETT.
Sample 128 trajectory/time entries, time uniform on 0..9. Weight by 10*gamma^t
to estimate the discounted visitation sum. Keep each recorded actor action;
at each candidate draw four independent x_prime/successor samples. Evaluate
.05*r(s_next)+.95*mean_{4 new actor samples} Q_(h-1)(s_next,a_next).
The next actor action depends on that candidate successor; never mean logits.
This surrogate's value is not J; its local exact-Q gradient has the standard
kernel likelihood-ratio interpretation. Its approximation is the object tested.

Use four iid standard Gaussian parameter directions and antithetic paired
queries. Perturbation std .01 for head biases and .1 for response coordinates,
same random keys/minibatches across signs and arms. Divide each coordinate by
its own std in the Gaussian score estimator. The entire emitted ES and frozen
critic surrogate are re-evaluated, including discrete atom/component decisions,
hard reward, projections and state-dependent next-actor sampling. This is a
valid estimator of an anisotropically Gaussian-smoothed parameter objective,
not pathwise differentiation or an unbiased unsmoothed gradient.

Minimize ES + lambda*surrogate, lambda=1 for joint and 0 for control. Query ES
on 128 train rows with 8 draws. Learning rates .01 head/.5 response; cap each
block's update norm at .03/.1 respectively. Response norm normalization keeps
the all-action bound for all parameter settings. A proposed iterate is accepted
only if its emitted ES on a fixed 256-row/16-draw train guard is <= initial
guard ES+.02 maze units; otherwise keep the prior iterate. No line-search or
extra query. This training guard is not a held-out certificate.

Final independent checks require the episode-clustered upper 95% CI for joint
minus its matched control ES AND joint minus initial ES <=.02, with all geometry,
F4 and action-bound violations <=2e-6. Reject a return improvement otherwise.
Final iterate only (including repeated rejected updates). No hidden audit label
or independent evaluation target is used to train/select checkpoints.

Independent evaluation uses the same frozen kernel for MC targets and critic
queries. Report the last training critic (possibly one-update stale) separately
from a fresh 1,000-step MC-only refit on new train-root continuations, with no
further ETT update. Reuse identical held-out queries/targets for the two critics.
Compare initial/control/joint root model returns using 2,000 paired stratified
context/episode bootstrap replicates. The first action is common across models,
fixed per root, and not resampled across repeats. Repeats are not independent
contexts. Model rollout changes require an upper paired CI <0, final calibration
pass, and fit/constraint pass for an estimator-and-update success claim.

Audit-only after training: reconstruct native death timing from the existing
collector's logged bits/episode death flags; report model motion/reward from
already-dead and observable stationary-outside contexts, and available alive
outside contexts. Never call these native hidden-context runs conditional ETT
ground truth. No new native simulator run is required. Logged subsequent
behavior rewards are a separate population, not this fixed actor's native Q.

Random namespaces: roots 90000000; preflight 91000000; critic 92000000;
training 93000000+100000*seed+1000*round; independent evaluation 94000000;
bootstrap 95000000. Signed queries reuse explicit offsets within a round.
Hard cap 1,500,000 emitted model transitions/draws, charged BEFORE every call,
including component tests and independent checks. Planned main training uses
about 529,000 outputs; evaluation <260,000. Smoke tests have a separate small
fixed component budget. Interrupted attempts retain markers/ledger. No large
training, budget expansion, loss search, actor update or automatic push.
