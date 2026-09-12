# Fixed-actor ETT adversarial prototype: specification before training

## Relationship to prior work and RAMBO-RL

[RAMBO-RL, Sections 5.1-5.3](https://arxiv.org/html/2204.12581v3#S5) combines offline transition fitting with model-value minimization. Its model gradient uses successor log likelihood and current-model value estimates, and its full algorithm alternates model and actor updates. Here only the model-minimization structure is borrowed. The actor and nominal policy stay frozen, the reward is known and fixed, and direct Monte Carlo replaces a learned value estimator. Neither the paper's estimator nor its statistical guarantees transfers automatically to this projected, partially observed ETT model.

The six-parameter residual search already minimized hard model-rollout returns. That objective is NOT new. Its distance-to-anchor bound did not establish all-action-pairs continuity after nonconvex projection/fallback. The action-response audit found nearly common drift, renewed motion on stationary histories, and occasional geometry artifacts. Synthetic supervised spline experiments established a continuum slope bound but not fidelity to a prescribed diagonal intercept. Adversarial synthetic fitting and the lambda=.004 diagonal-budget experiment demonstrated that a soft two-term objective can distort that intercept. The frozen-readout and independent fixed-actor reports found only partial transfer and unresolved local ordering. No scorer from those experiments is used here.

This prototype replaces the old residual with 32 context-dependent signed-action matrix coefficients and replaces action-dependent nonconvex fallback with projection onto an action-independent convex rectangle. It proves an all-action-pairs bound on the emitted transition, freezes the entire diagonal sampling law, and scores that emitted diagonal law without pretending its pre-projection density is its output density. It is still a restricted search, not shared-network diagonal/adversarial learning.

## Transition class and exact guarantees

State s has four newest-first XY frames, dimension 8; g repeats (8.5,3.5) four times. Execution action x and auxiliary nominal action x' have dimension 2 in [-1,1]^2. Only the new XY is generated; output F4 is (q_new, s[:6]). Norms are Euclidean in native action and maze-position units. No hidden variable, death label, mask, time index, reward or future state enters either policy or the response conditioner.

Let A(s,x',g,z) be a coupled anchor from the *existing emitted* diagonal sampler, including its zero atom, mixture draw, displacement clipping and endpoint correction. Its source is the preserved zero-residual guarded checkpoint used in the earlier six-parameter experiment. All of its parameters stay frozen. It is independent of x when (s,x',g,z) are held fixed.

Eight fixed softmax gates w_j(s) sum to one. They are radial weights around four current-X centers (0.5,2.5,4.5,7.5), crossed with F4 RMS-motion centers (0,0.7), using length scales (2,0.35). The signed response matrix is

    M_theta(s) = sum_j w_j(s) B_j,
    B_j = L * theta_j / max(1, ||theta_j||_F),  theta_j in R^(2x2).

There are 8*2*2=32 free coordinates. L=1 is a prescribed engineering restriction, not a fitted/certified constant for true physical dynamics. Observational diagonal data does not identify that constant; lethal state changes may not satisfy this samplewise smoothness assumption. Frobenius normalization is conservative relative to the operator norm. Within a context the response is affine in signed action before projection; context gates and boundary clipping make the full model nonlinear, but this is not an unrestricted neural transition family.

The static free grid is covered by four axis-aligned rectangles: [0,9]x[3,4), [1,8)x[1,2), [1,2)x[1,4), [7,8)x[1,4). Upper interior boundaries use the preceding representable float32 coordinate, matching the grid's floor convention; the outer x=9 boundary follows the existing sampler's inclusive bound. Intersect each rectangle with the coordinate box s[:2]+[-1,1]^2. Among intersections containing A, choose the largest area, breaking ties by fixed rectangle order. This choice depends on s and A, NEVER on x or theta. Each reachable anchor is covered; an empty valid set is an error, not an action-dependent fallback.

Let C(s,A) be that rectangle. Emit

    q_new(x) = projection_C(A + M_theta(s)(x-x')).

For fixed conditioning and coupled noise, ||M||_2 <= sum_j w_j||B_j||_F <= L. Euclidean projection onto a fixed nonempty closed convex rectangle is nonexpansive. Therefore, for ALL x1,x2,

    ||T_theta(s,x1,x',g,z)-T_theta(s,x2,x',g,z)||_2 <= L ||x1-x2||_2.

History components cancel in this norm. Rectangle selection can be discontinuous in conditioning variables, but not in x for this fixed-conditioning comparison. All base mixture/geometry branches occur before x enters. There is no subsequent collision/fallback branch. Since A is in C, x=x' emits A exactly. The same coupling also supplies a one-step Wasserstein-2 upper bound for fixed s,x',g (and after common nominal marginalization at fixed s,g). It does not give a bound across states or goals. Numerical audits test implementation consistency, not the continuum theorem; a 2e-6 absolute tolerance covers float32 arithmetic rather than serving as a physical constant.

Guaranteed feasibility means open-grid endpoints, original domain bounds, coordinate displacement <=1, membership in the selected free convex rectangle and exact F4 shifts. It does NOT mean simulator reachability: the rectangle need not contain the current position, and the preserved diagonal anchor already permits rare corner crossings. Straight-segment wall crossings are explicitly audited, not silently called impossible. Motion/absorption, hidden causality and long-horizon consistency are not certified. Tightening these constraints while preserving an imperfect anchor exactly may require a different diagonal model.

## Two-term objective and diagonal acceptance

Exactly two terms are retained:

    L_ETT(theta) = L_diag(theta) + lambda J(theta).
    J = E[sum_(t=0)^49 .95^t * 1[||q_(t+1)-(8.5,3.5)|| < 2]].

L_diag is the sample-based energy fitting score of the emitted XY law at (s,a,a):

    E ||Y-y_obs||_2 - 0.5 E ||Y-Y_tilde||_2,

where Y and Y_tilde are independent diagonal draws. The K=32 unbiased U-statistic excludes self pairs. This is evaluated in maze units and includes projection boundary mass and the stationary atom; deterministic F4 components are checked separately. No ambient eight-dimensional density is assumed. Old unprojected mixed NLL is reported only as a frozen-backbone diagnostic, explicitly not as emitted-law likelihood or an optimization score.

The entire diagonal law is structurally identical for all theta. Thus the diagonal term is a constant in THIS experiment, and only J changes the 32 parameters. This does not test a tradeoff with shared diagonal learning. Acceptance: zero theoretical diagonal-law drift, checked as maximum coupled output change <=2e-6 and energy-score increase <=1e-6 on fixed training and held-out tuples. Mean, p95, p99 and maximum per-tuple predictive errors and score changes are reported; baseline misfit is not hidden by the drift constraint. No third loss is added.

## Sampling and optimization

Each step independently samples x' from the frozen nominal state-goal policy, x from historical stochastic alpha0_seed0 actor, and z for the ETT. x' is auxiliary and stored separately from executed actions. Initial state is the actual fixed reset with four copies of (0.5,3.5). Horizon 50 matches the original task; goal and reward never change and no success termination or hindsight relabeling is used. The nominal-action-marginalized model is a learned rollout kernel, not an identified interventional distribution.

The frozen nominal is the existing teacher-source k=5 MDN (episodes 1200..5999, noisy privileged teacher behavior including failures). The base diagonal uses its established source episode split. Offline numerical fit checks read only obs, act and metadata, never hidden arrays. New simulator continuation outcomes, readouts and raw critic scores are not loaded for optimization or evaluation targets.

Clipping and geometry create singular output mass, and F4 has deterministic coordinates. The emitted off-diagonal density is not available, so no likelihood-ratio estimator is used. No gradient is taken through a hard reward. Instead, for u~N(0,I_32), estimate the gradient of the Gaussian-smoothed full-rollout objective using paired values at theta +/- sigma*u with common complete-rollout random keys:

    g_hat = mean_i [(F(theta+sigma*u_i)-F(theta-sigma*u_i))/(2*sigma)] u_i.

Each perturbed model generates its own current rollouts; actor/nominal decisions naturally respond to its states. Coupled random numbers reduce variance without fixing x' values across diverging states or altering either marginal. The hard parameter decoding admits every perturbed theta. This estimates a smoothed objective, with finite-direction/path variance and no global-optimum guarantee. An analytic quadratic sign test is required.

Compare lambda=0 to lambda=1/sum_(t=0)^49 .95^t, fixed before returns. This gives the adversarial term range [0,1]; because the diagonal is invariant, lambda only sets gradient scale here. No lambda tuning occurs. Both arms start at zero theta and use identical perturbations/rollout keys per optimizer seed; the zero-weight arm executes the same query budget and remains unchanged.

## Fixed budget, seeds and acceptance

- Optimizer seeds 0/1; 16 updates; 8 Gaussian directions/update; sigma=.10; 32 reset paths/query; learning rate 1; maximum parameter-update L2 norm .20. Save the final iterate, with no outcome-based checkpoint selection.
- One initialization signal check: 256 reset paths. Stop without training if fewer than 5% have nonzero return or return standard deviation <.1. No reward shaping or budget extension.
- Monitor 32 paths at the initial and each subsequent iterate, using fresh declared monitor seeds, never for checkpoint selection.
- Independent evaluation: four fixed seeds, 128 paths/seed/arm; 500 paired trajectory bootstrap replicates. All arms use the same evaluation keys. Report both optimizer seeds separately and all evaluation groups.
- Diagonal audit: 2,048 source-training and 2,048 source-held-out recorded tuples, selected once by fixed RNG, K=32 coupled output draws at x=x'=a. Preserve IDs and hashes. No data-driven model fitting occurs.
- Action audit: first 128 held-out audit contexts; one nominal draw/context, 8 coupled transition draws, 9 grid actions plus x=x' and 16 arbitrary action pairs/context; near-pair spacings 1e-3,1e-5 also checked. Feasible-rectangle inclusion, all-pairs excess, coordinate bounds, F4, correction rates and response directions are reported.
- Primary rollout budget: 1,638,400 optimization transitions, 108,800 monitor transitions, 102,400 final evaluation transitions and 12,800 signal-check transitions: 1,862,400. At most 1,000,000 additional single-step fit/constraint/replay samples; hard total cap 2,862,400 model transitions. No new simulator trajectory is requested.

Success requires unchanged diagonal law within tolerance, the stated (limited) constraints, and a negative independent paired return difference with its 95% interval below zero in BOTH optimizer seeds. Otherwise report a failed/inconclusive prototype. Even success establishes only pessimistic optimization inside this class. Actor training is not automatically justified; assess renewed stationary motion, boundary accumulation, corner crossings, action-response collapse and overall model validity first. Stop after this comparison; no actor update, alpha=.5, value fitting, ETT objective change, or automatic push.
