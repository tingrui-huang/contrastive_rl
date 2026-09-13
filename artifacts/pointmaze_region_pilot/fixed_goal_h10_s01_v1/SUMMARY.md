# PointMaze integration pilot: model effect, unresolved native semantics

**The calibrated estimator and constrained update transfer mechanically, but
this pilot does not support actor integration.** Both joint seeds reduce
independent model occupancy within the declared diagonal and action constraints.
The reduction occurs entirely in contexts that the native audit identifies as
already dead. Joint updates make all of those model continuations move, and no
alive outside-goal contexts were sampled. Recovery-versus-death discrimination
therefore remains untested here.

## Frozen setup and implementation

Starting revision: `9f17c4e`, PointMaze branch. The finite-state experiments and
production policy/critic/generator implementations remain unchanged.

The pilot wraps the existing `ConvexActionTransition` rectangle sampler. The
shared diagonal law is conditioned on `[F4 state, x_prime, x_prime, F4 goal]`;
its atom/K3 Gaussian samples pass through the existing projection. The K5
nominal sampler receives **state and goal**. It models the established noisy
teacher-source behavior, including failures. The fixed stochastic actor is
historical alpha=0, seed=0, step 150,000. Input hashes are in `provenance.json`.
There is one shared kernel, sampled with fresh `x_prime ~ b` at every step;
only `x ~ pi` is executed. Auxiliary x_prime records are separate.

Trainable coordinates are 16 original diagonal-head bias offsets and 32
existing response coefficients. Both losses update the head; only the
pessimistic term updates response coefficients. Emitted XY energy score fits
the diagonal; the original diagonal network's pre-projection mixed NLL is
reported separately. The latter is not an emitted-law density. This limited
48-dimensional search is not full-network training.

The inherited normalized response matrices and action-independent convex
projection give **samplewise Euclidean action Lipschitz L=1**, holding state,
goal, x_prime and randomness fixed. Coupling the same x_prime draws also gives
a Wasserstein bound on the marginalized intervention. This is not a TV bound
or a bound on critic outputs. Existing geometry excludes some valid fork
transitions; it is preserved without adding restrictions and is not a complete
physical-validity guarantee.

The isolated critic predicts future **reward-region membership**, not a
continuous F4 point-goal density. Binary NCE uses uniform negative labels,
alpha=0, and positive/negative weights 1:31. Truncated-geometric positive
sampling yields `Q_h=(1-.95^h)*31*.5*exp(f_1)` and `Q_0=0`. No logits are averaged
as values, clipped into range, or replaced by hardcoded goal/death values.
Stable softplus and float64 are the numerical stabilizations. The production
actor and point-goal critic are untouched.

## Budget and numerical checks

The protocol, seeds and source hashes were saved before collection. Roots are
actual teacher-source F4 observations at t=40, 32 outside/32 inside the existing
goal region in each train/validation partition. Complete H=10 continuations
end at task time 50; gamma=.95. No time-limit extension or new reward is used.
The dataset has 51 stored action slots, with one terminal zero dummy, and 50
actual transitions. Root selection does not inspect hidden labels or returns.

The initial independent calibration gate passed at h=10/5/1 with RMSE
0.03288/0.01923/0.00363. Only then were six alternating updates run for each of
two paired seeds and two arms. Each update freezes the fitted critic and
visitation, then uses four antithetic Gaussian parameter directions with
common random numbers. Standard deviations are .01 for head biases and .1
for response. This estimates a Gaussian-smoothed objective's gradient through
all discrete sampling, hard rewards and projections; it is not an unbiased
pathwise gradient of the unsmoothed objective. No extra loss was added.

All 24 proposed final-step updates passed the training diagonal guard. Total
charged outputs, including independent evaluation, were **742,016/1,500,000**.
All 16 finite-state regression checks and 7 PointMaze component checks passed.
A separate artifact audit replayed the signed estimator, descent sign,
acceptance decisions and final iterates, and verified action alignment,
F4/reward/horizon handling, source hashes and budget accounting without
collecting further model outcomes.

## Independent results

Values below are `(1-gamma)*sum gamma^t reward`; unnormalized discounted task
returns are 20 times larger. Each held-out context has one fixed first actor
action and 16 independent model continuations. All kernels share the first
action and random streams; diverged kernels do not share identical paths.

- Seed 0: control **0.219195**, joint **0.213222**. Paired change **-0.005973**,
  episode-stratified bootstrap 95% CI **[-0.008859, -0.003394]**.
- Seed 1: control **0.219057**, joint **0.213690**. Paired change **-0.005367**,
  95% CI **[-0.008019, -0.002926]**.
- Diagonal ES increases versus controls are **0.002590** and **0.001856**;
  clustered upper 95% bounds **0.003172** and **0.002202**, below the declared
  **0.02** tolerance. Comparisons with initialization pass too. This is
  finite-sample ES noninferiority, not a distributional-distance certificate.
- Pre-projection NLL worsens from -5.7566/-5.9070 to -4.8564/-5.4985. Negative
  NLL is possible for continuous densities. The ES tolerance permits measurable
  fitting degradation; the comparison does not establish exact diagonal laws.
- No checked geometry, F4, coordinate-step or action-bound violations; maximum
  sampled Lipschitz excess is zero. The all-action guarantee comes from the
  unchanged construction, not finite sampling alone.
- Last training critics on the **same final kernels** have h10 RMSE
  **0.02683/0.01921** for joint seeds 0/1. Separate 1,000-step fresh MC-only
  post-training refits have **0.03803/0.03809**, against identical held-out
  query/target arrays. All predeclared aggregate calibration gates pass.
  Refits were never fed back into ETT training.

Calibration is approximate: decoded predictions sometimes exceed the legal
occupancy upper bound. For example, the joint refits exceed the h10 bound by
up to **0.09651/0.08556**. These errors are retained without clipping. Aggregate
gate success does not guarantee local gradient accuracy. Neither exact
PointMaze values nor exact pessimistic gradients are available in this pilot.

## What limits the result

Post-training hidden-label auditing finds **32/32 outside roots already dead**
and **0 alive outside roots**. The other 32 are already inside the goal and
receive maximal model occupancy across all arms. There is no basis here for
claiming successful discrimination of recoverable setbacks from death, and
the context budget was not expanded to find such cases.

From the already-dead native contexts, model any-goal probability falls from
**22.85%** in controls to **15.63%/16.41%** in joint arms. However, model
any-motion probability increases from **44.14% to 100%**. Mean step distance
over all roots rises from about **0.191 to 0.216/0.209**; final XY repeat
diversity remains about **0.371** per coordinate. Thus the decrease is not
explained by universal stationary collapse. It includes artificial motion
from contexts whose native continuations are absorbing. The existing visible
F4 sampler has no persistent death state, and its action response can move a
stationary diagonal anchor off diagonal. None of the enforced constraints
prevents this. The optimizer did not violate its declared constraints; those
constraints do not encode the missing native semantics.

Logged subsequent native behavior return is zero for these dead contexts.
That is an audit of the original dataset, not a new fixed-actor simulator run
or conditional ETT ground truth. Hidden labels never enter fitting or model
selection. Results are confined to this balanced late-task distribution and
cannot establish reset performance, causal identification, robust Q, or a
certified global worst case.

**Recommendation: stop.** The estimator/update transfer has a bounded model-internal
effect, but missing recoverable coverage and unmodeled absorption are the
specific obstacles to a stronger interpretation. Local critic calibration
also remains imperfect. No actor training or further experiment family was
launched.

## Reproduction and artifacts

See `REPORT.md` for commands and full per-horizon metrics, `PROTOCOL.md` for
the pre-run specification, and `results.json` for clustered comparisons.
All 64 held-out roots' trajectories, execution actions, auxiliary x_prime,
last/refit predictions and independent returns are in `*_evaluation.npz`.
Audit labels are separate in `audit_only.npz`. Checkpoints remain locally in
the ignored `checkpoints/` directory. Nothing has been committed or pushed.

Additional read-only verification:

```powershell
python -m ett.check_pointmaze_region_pilot --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1
```
