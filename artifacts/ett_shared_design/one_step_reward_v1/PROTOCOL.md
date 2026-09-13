# Executable proposal: one-step shared ETT, local task-reward boundary

Status: **design/reference implementation only; training and assessment unrun**.
The `prepare` and deterministic `check` phases ran successfully. The reference
runner's full optimization runtime has not been exercised; a code/runtime defect
must retain its attempt marker and artifacts, not trigger a silent restart.

## Population and information

Use the existing p=0.3 dataset and inherited teacher-source episodes [1200,6000),
with the original seed-0 episode holdout. Include only pre-action visible XY
with 5.5 <= X <= 5.75 and 3.125 <= Y <= 3.875, based solely on current position.
Do not filter by reward, future state, stationary history, hidden status, or model
prediction. Retain full recorded F4, natural action a, and next F4 target.
Both losses sample an eligible source episode uniformly and then an eligible
row uniformly within it; repeated absorbed rows do not overweight that episode.
The measured row counts and source hashes are in `preparation.json`.

This narrow region has a complete coordinate-step envelope whose free endpoints
are in the upper corridor; neither the left, lower, nor right passage can be
reached within one coordinate step. Here a fixed upper rectangle introduces no
extra exclusion among endpoints allowed by the declared static geometry and
step envelope. It does not prove native axiswise reachability. The region is a
deliberate local task test, not a proposed whole-maze architecture or geometry
sweep. All retained recorded targets passed its geometry check. Source episode
holdout is independent of this new fitting but has been inspected previously.

Let c=(s,x_prime,g), with g the unchanged tiled commanded goal, x_prime=a.
Draw x uniformly on [-1,1]^2 independently of a. Exact equality has probability
zero in the continuous proposal; finite equality should be reported as a
diagonal case, not used to change the sampling distribution. Neither frozen actor
nor nominal weights are loaded by the protocol. This avoids substituting a
possibly different nominal context marginal between the two losses.

## One shared trainable generator

Let z~N(0,I_2), v~Uniform(0,1), fixed when comparing execution actions. A single
two-layer tanh context network with widths 32/32 receives normalized c and z.
Its common hidden features h produce 11 unconstrained linear head coordinates:
one p logit, two displacement coordinates, and eight response offsets k.

    p_theta = sigmoid(head_0(h))
    b_theta = 0                        if v < p_theta
              2 tanh(head_1:3(h))      otherwise
    k_theta = head_3:11(h)
    U = U_raw / max(1, ||U_raw||_F)      [8 x 2]
    V = V_raw / max(1, ||V_raw||_F)      [2 x 8]
    R_theta(d) = V [ReLU(U d + k_theta) - ReLU(k_theta)]
    G_theta(s,x,x_prime,z,v)_XY = P_C(s)(q + b_theta + R_theta(x-x_prime))
    G_theta(...)_F4 = [new_XY, s[:6]].

C(s) is `[q-1,q+1]` intersected with the upper rectangle
`[0,9] x [3,nextafter_float32(4,-infinity)]`. It is fixed across execution
actions, nonempty on this protocol's state domain, and contains q. The prototype
is unsupported as a general transition API outside upper-corridor contexts.

There are **2,123 trainable parameters**, including every context layer and
head and both response matrices. No frozen diagonal anchor network enters this
generator. At x=x_prime, R=0, but the resulting diagonal law is itself trainable.
Energy fitting updates the displacement, mixture and shared context features;
the pessimistic loss updates those same features as well as the response.
Pure response-matrix parameters have no diagonal effect, which is explicit;
it is not claimed that every parameter receives both gradients. The common
backbone and diagonal distribution do receive both objectives. The stationary
branch represents a one-step atom, never a death label or persistent state.
The response can depend on the same z/features as the moving proposal, removing
the old deterministic state-only correction restriction, while remaining a
limited conditional network rather than a universal family.

Normalization is fixed from training c with the same episode/row weighting;
each standard deviation is floored at .001. Raw execution difference d stays in
native action units and is **not** normalized. All stored historical models and
normalizations remain unchanged. This trains a new observation-fit generator;
it intentionally relaxes the previous experiment's freeze requirement, as
necessary to test two active losses. It does not overwrite the old diagonal.

## Exact full action bound and numerical contract

Hold s,g,x_prime,z,v and theta fixed. The atom decision, b,k and C are then fixed.
ReLU and Euclidean projection onto one closed convex rectangle are nonexpansive.
For every pair x_1,x_2 in [-1,1]^2,

    ||G_theta(x_1)-G_theta(x_2)||_2
       <= ||V||op ||U||op ||x_1-x_2||_2
       <= ||x_1-x_2||_2.

The subtraction of ReLU(k) cancels in differences; it does not double the bound.
F4's six shifted coordinates also cancel. The arbitrary context network is not
on the varying-x path. Action-dependent mixture logits or action inputs to that
unconstrained network would invalidate this proof and are forbidden. This is
the **same full samplewise execution-action L=1 guarantee**, not merely a bound
relative to the diagonal or a distributional average. It implies a coupled
distributional bound but identifies neither native L nor causal effects.

The proof is over real arithmetic. Implementation checks allow 2e-6 absolute
action-bound excess; rounded q+/-1 is the coordinate-step contract, not a
literal real-arithmetic step repair. The previous float32 anchor excesses remain
historical facts; they are neither moved nor reused as immutable new anchors.
No strict numerical guarantee over a quantized continuum is inferred from grids.

## Objective, estimator, and witness

For K independent emitted diagonal XY draws Y_i and observed y, use

    ES_hat = (1/K) sum_i ||Y_i-y||_2
             - [1/(K(K-1))] sum_(i<j) ||Y_i-Y_j||_2.

This is the unbiased energy-score U-statistic, in maze units, for the emitted
mixed law (including atoms and projected boundary mass). No ambient F4 density
or pre-projection likelihood is substituted. The two-term objective is exactly

    F(theta) = E_D ES_hat(theta)
                 + lambda E_D E_x,z,v 1[||G_theta(s,x,a,z,v)_XY-g_XY||_2 < 2],
    lambda = .01 (maze units per unit reward).

Lambda is a fixed experimental scale, not a teacher value or an established
diagonal-error guarantee. The geometry and task radius supply the reward
ordering. For this fixed goal, all native fatal positions already have reward
zero under the position rule, so no hidden gate or critic is needed for this
one-step score. Reward zero is not identified as death. H=1 deliberately replaces
H=50; gamma has no effect on the single term. This changes the diagnostic target
explicitly and does not claim to solve long-term pessimistic control.

The reference estimates the **entire** objective by four Gaussian antithetic
parameter directions, common minibatches and noise for each plus/minus pair.
There is no pathwise derivative through the hard reward, Bernoulli threshold,
or clipping and no assumed emitted likelihood. The estimator targets a
Gaussian-smoothed parameter objective. Both fitting and reward are queried at
every perturbed model, so L_diag is not cached or treated as constant. All
separate queried terms are saved. A finite search is not a global optimum.

Existence check, fixed before action checks: q=(5.75,3.5), x_prime=(0,0), a
moving proposal b=(.875,0), k=0. Choose the first two U rows as
`(1/sqrt(2),0)` and `(-1/sqrt(2),0)` and V's first row as
`(-1/sqrt(2),-1/sqrt(2),0,...,0)`; remaining entries are zero.
Both Frobenius norms are one, giving R_X=-|x_X-x_prime_X|/2, R_Y=0 for the
entire action square. Diagonal XY=(6.625,3.5) has reward one; d_X=.5 yields
(6.375,3.5), reward zero. The response changes no diagonal output for any c,z,v.
This is a fixed member of the proposed class, **not a trained model, native
off-diagonal target, or globally fitted witness**. On its stationary branch it
still satisfies the same full action bound. Its 15-action implementation check
and two geometry controls use 45 deterministic outputs and have zero observed
bound excess. No witness target or special lookup enters training.

## Fixed future execution and budget

For each optimizer seed 0 and 1, initialize all matrix parameters with iid
N(0,.1^2), biases zero. Perform 128 diagonal-only warmup updates, then clone the
complete parameter vector into matched diagonal-only and joint arms. Both
continue for exactly 256 updates with all parameters trainable. Warmup does not
freeze diagonal parameters for the joint phase. The random initial, warmup and
two final checkpoints are retained; no monitoring or test selection is allowed.

Every update uses batch 16, K=4, four Gaussian parameter directions,
sigma=.02, learning rate .05 and update-norm cap .1. Uniform-episode minibatches,
execution actions, generator draws and perturbations match across arms within
an optimizer seed. They are one-step fixed conditioning inputs, not replayed
successors of a supposed rollout. No policy or nominal model is updated or
sampled, and no simulator runs.

RNG namespaces are explicit in the reference: base 73,000,000; initialization
base+seed; warmup contexts base+1,000,000+100,000*seed+iteration; joint contexts
base+2,000,000+100,000*seed+iteration; directions add 5,000,000 to those phase
offsets; independent evaluation base+9,000,000; bootstrap base+10,000,000.
These are proposal seeds, not claims of untouched source episodes. Deterministic
preparation uses the inherited seed-0 split and never initializes a generator.

Budget with the fixed 66 held-out rows:

* Warmup: 2 seeds x 128 updates x 8 queries x 128 emitted samples = 262,144.
  The reference computes both terms even when lambda=0, so both are counted.
* Continuation: 4 arms x 256 x 8 x 128 = 1,048,576.
* Evaluation: 8 checkpoints x 66 rows x 64 draws x 2 action settings = 67,584.
* Post-reload component checks: 8 x 66 x 12 settings x 4 draws = 25,344.
* Design existence/control check: 45 deterministic outputs.

Total planned outputs **1,403,693**, hard cap **1,500,000**, including the design
check. No native transitions, long rollouts, monitors, geometry sweep, model
selection, downstream learning, or automatic retry. The remaining cap is only
for a documented implementation check, not extra optimization. The 2,123-D
finite-direction optimizer may be underpowered; failure is reported, not fixed
with an expanded search. Preserve the started-attempt marker on interruption.

## Independent evaluation and separate criteria

Use all 66 held-out rows from 26 episodes, one independent execution action and
64 coupled z,v draws per row, shared across checkpoints. Report episode-uniform
means and paired 2,000-replicate **episode** bootstrap intervals; do not pool
optimizer seeds. These intervals condition on the fixed checkpoints and sampled
actions/noise, with limited power from 26 source episodes. Report both objective
terms and distributions, not just the scalar total.

1. **Diagonal fit:** diagonal-only final minus random-initial held-out ES must
   have upper 95% CI <0. Joint minus diagonal-only ES must have upper CI <=.02
   maze units, in each seed. This is an engineering noninferiority tolerance,
   not exact equality to observations or the frozen checkpoint. Report
   stationary-output frequency and predictive errors descriptively, never as
   a death rate. A failed fit gate prevents a pessimistic success claim.
2. **Constraint:** same stored normalized matrices and fixed convex-set proof;
   no nonfinite states, geometry/F4 failures or action-pair excess above 2e-6.
   Check 9 grid actions, the diagonal, the independent execution action and a
   clipped 1e-4 neighbor, under identical z,v. These finite checks support the
   proof. Pure geometric controls at q=(3.5,3.5) and q=(8,3.5) force reward
   respectively zero and one for every output in C. A loss cannot prefer a
   different one-step reward there; do not require their coordinates to freeze.
3. **Pessimistic selection:** joint minus diagonal-only off-diagonal reward
   must have upper CI < -.02, separately in both seeds. Also require upper CI
   <0 for the response contrast
   `[r_joint(x)-r_joint(a)]-[r_control(x)-r_control(a)]`, paired within each
   checkpoint. This distinguishes changed action response from just reducing
   diagonal reward. Both requirements are in addition to the fit and constraint
   gates. Mixed seeds or intervals crossing a threshold are inconclusive.

Checkpoints, common inputs, all emitted diagonal/off-diagonal F4 arrays and
separate losses remain available for re-analysis. `assess` implements the three
numeric gates and reload checks. It does not label death or certify native
kinematics. No result can pass merely by collapsing all ambiguous histories.

## Minimal code changes and commands

The isolated `scripts/shared_ett_one_step_protocol.py` contains the proposed
generator, visible-only preparation, joint two-term perturbation loop and saved
output assessment. It reuses the existing energy score and perturbation update
helpers. No production transition class, CRL replay interface, actor/critic,
nominal policy or historical checkpoint is changed. Integration into those
pipelines is not part of this experiment. The new generator's parameters, mode,
normalization and input hashes are persisted separately.

The following phases were executed for this design only:

```powershell
python -m scripts.shared_ett_one_step_protocol prepare --out-dir artifacts/ett_shared_design/one_step_reward_v1
python -m scripts.shared_ett_one_step_protocol check --out-dir artifacts/ett_shared_design/one_step_reward_v1
```

The following is an executable **future** protocol, supplied but **not run**.
It requires a new execution instruction and a fresh output directory:

```powershell
python -m scripts.shared_ett_one_step_protocol prepare --out-dir artifacts/ett_shared_design/one_step_reward_run_v1
python -m scripts.shared_ett_one_step_protocol check --out-dir artifacts/ett_shared_design/one_step_reward_run_v1
python -m scripts.shared_ett_one_step_protocol run --out-dir artifacts/ett_shared_design/one_step_reward_run_v1
python -m scripts.shared_ett_one_step_protocol assess --out-dir artifacts/ett_shared_design/one_step_reward_run_v1
```

The score is already justified for the narrow immediate-reward question. A
death-versus-survival preference outside the reward region remains unsupported;
that is a specific missing criterion, not a request for another broad audit or
authorization for a latent-death implementation.
