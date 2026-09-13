# Construction audit: what would the death intervention mean?

This is analytic source inspection, with no new simulator or learned-model calls. Symbols denote probability laws, not fitted new models. The result is failure to certify a repair comparison of the requested baseline, not a proof that every augmented model is impossible.

## A: exact baseline interface

Use `ett.pointmaze_norm_comparison.ModeKernel('frobenius')`, with seed-specific `s0_k0_u24` / `s1_k0_u24` from `artifacts/pointmaze_region_pilot/response_search_s01_v1/candidate_pool.npz`, also carried into the latest experiment's `initial_parameters.npz`. These are starting models of E26, not its final spectral models. Freeze all 48 coordinates, the old network, nominal sampler, actor and normalizers. The code's setup uses the original alpha0/seed0 full actor checkpoint and the state-goal K5 nominal checkpoint listed in `artifacts/ett_rollout_return/residual6_s01/config.json`.

For state s (eight F4 coordinates), goal g, natural action x', and fresh anchor noise z:

    a0 ~ K_diag_theta(. | s,x',g)
    M(s) = sum_j softmax_gate_j(s) * W_j/max(1,||W_j||_F)
    y = projection_C(s,a0)(a0 + M(s)(x-x'))
    s_next = [y, s[:6]]

The 16 head offsets are part of K_diag_theta. The projection set is chosen independently of execution action x; it contains the anchor. Consequently x=x' preserves the anchor sample and distribution, and fixed-(s,g,x',z) XY output is 1-Lipschitz in x. The old six history coordinates are shifted, not regenerated. A samples fresh nominal, actor and transition randomness on every step. Its mixture atom means zero displacement; it is not a native death variable. **A's generated death onset/rate is undefined.**

Native `_dead` is different: while alive, add action noise, clip the action, execute ten collision-checked substeps, test the final cell against the current mask, set `_dead` if lethal, and redraw the mask. While dead, ignore actions, leave XY fixed, redraw the mask, return zero reward and `done=False`. F4 still shifts. The task stops externally at 50, with gamma=.95. The reward region around (8.5,3.5) cannot contain a lethal landing.

## B1: shared architecture, unchanged live kernel, switch persistence on

Augment both arms with auxiliary D, a mask U and a death-position record q. In A these have no effect on output, exactly preserving the existing baseline. In B, when alive use the same generated landing y, set `D_next = D OR lethal(y,U)`, and, from the following step onward, hold XY=q=y and continue shifting F4 until t=50. Fatal and subsequent rewards are zero; no shortened horizon. This adds no trainable weights and could isolate the *output consequence of a stipulated event process*.

It does **not yet specify that event process**. To call it a native onset repair, supply a sequential joint distribution of `(D,U,teacher memo,x',z)` conditional on the chosen visible history, with the required K_diag and nominal marginals. The teacher reads U before generating x'. The source process also has an episode-level force-safe choice and teacher memo, and censored noisy nonzero actions versus exact waits. Thus neither drawing U independently at p=.30 after observing x' nor substituting a generic action density constitutes native conditioning.

Knowing a hazard marginal is insufficient: for one cell, let H mean that a generated landing is in it, p=P(U_cell=1|conditioning), and h=P(H|conditioning). Without the joint coupling,

    max(0,h+p-1) <= P(H AND U_cell=1 | conditioning) <= min(h,p).

These are illustrative probability bounds, not measured project values. They show exactly which onset quantity an independent-mask assumption sets to hp. Even a known p would not give the anchor/mask coupling. The learned sampler's z is an artificial mixture draw, not the simulator's structural motion noise. No inspected artifact supplies that coupling. A latent posterior given the *whole* generated history also needs a declared update, especially once alive and dead contexts can share F4.

## B2: retain the diagonal after adding a latent dead component

Let F(s)=[s[:2],s[:6]] be the absorbing successor and K_A the existing emitted diagonal law at a fixed visible conditioning `(s,g,x')`. If q is the posterior probability of already being dead, then an augmented model's visible law is

    K_B = q delta_F(s) + (1-q) K_live.

Holding the live kernel equal to K_A gives `K_B-K_A = q(delta_F(s)-K_A)`. It preserves the visible diagonal only if q=0 or K_A is already that point mass. A shared augmented software architecture does not remove this distributional change.

One can instead choose `K_live = (K_A-q delta_F(s))/(1-q)`, for q<1. A necessary and sufficient condition for this *one-step probability measure* to be nonnegative is `q <= K_A({F(s)})`. Total emitted stationary mass is required, not just the raw atom probability; projection can add mass. This construction changes the live law, needs q, may not lie in the present parameterized family, and does not establish a sequential posterior or an all-action coupling. It is a possible route to a future construction, not a universal impossibility argument.

E10's positive moving support was established at particular older frozen anchors/conditions. It cannot be used numerically as the support mass of the later 16-offset models. We did not rerun its support panel or claim its exact atom probabilities hold for A here. The mixture identity above states the conditional requirement without such an extrapolation.

## Simulator-assisted alternatives and their exact limits

**Copied death time from another branch:** invalid once states/actions diverge. Even matched seeds cannot transfer a death label to a different generated landing. Rejected, not run.

**Native transition from a transplanted generated position:** resetting a simulator's XY/F4 to the generated state while retaining a reference mask/death flag does not create the missing conditional latent law. A previously dead simulator would ignore the action even if its position is forcibly moved. Keeping it alive instead changes its state. Its native step can also land somewhere different from the model output; assigning that step's death to the model endpoint again crosses branches. An explicit transplant intervention can be simulated, but is not automatically a repair of A.

**Joint native branch:** from a complete pre-action native snapshot, generate x' from its actual teacher/memo/noise, execute x in the cloned native state, and thereafter use branch-local native physics and independent per-time noise slots. This is internally consistent; a bank of independent pre-outcome snapshots gives a population of paired simulator interventions. Repeats from one full snapshot condition on that snapshot, not the visible conditional ETT distribution. To query prescribed x', the joint procedure must disintegrate/reweight over latent/memo histories using the teacher action law (including atoms and clipping), rather than relabel an arbitrary snapshot with that x'.

Using the native simulator in both arms and disabling/enabling death would isolate native death's effect on native trajectories. However, neither live arm reproduces the existing learned anchor/response law. Using A versus native B changes motion, collision geometry, action response and conditioning as well as death. This would be a physics replacement comparison, so it cannot establish that death alone caused the learned-model gap. The same augmented architecture in both native arms does not recover A as the control.

**A fully specified synthetic killed model:** begin at START with D=0. Draw U_t as three IID Bernoulli(.30) bits independently of model/nominal/actor noise, before each transition, using separate time-indexed slots. Draw x'_t from the frozen nominal conditional on that branch's F4 and goal; draw x_t from the frozen actor; draw z_t for A's live transition. While alive, use A's generated landing, and set D on that landing if its cell is active under U_t. Carry D irreversibly; after onset use fixed XY and shifted history. Use the same auxiliary state and random-slot allocation in the control, where D is ignored. Compare only until the original horizon. No logged or parallel-branch death labels enter this construction.

This is an internally consistent, privileged-geometry **synthetic diagnostic**, including an explicit independent U/x'/z assumption. It could validly estimate the effect of adding this particular killing rule. It is not the native conditional ETT: the native teacher makes x' informative about U; the synthetic nominal draw does not. Nothing here verifies native onset frequencies conditional on the generated branch. Therefore we did not call its programmed persistence a verified repair of native onset or spend a Stage 2 budget on it. This is an attribution limitation, not a claim that simulator-assisted or hybrid diagnostics are forbidden.

Moreover, under shared randomness A and this B agree until the first synthetic death. B's subsequent rewards are zero and A's are nonnegative. If tau indexes the fatal transition, then

    J_A - J_B = .05 * sum_{k=tau+1}^{H-1} .95^k r_A,k >= 0.

The fatal landing is outside the goal reward region, hence both fatal rewards are zero. A lower B return follows from censoring reward tails even when the synthetic death times are wrong. Independent native action contrasts, not this algebra or its fitted critic, would be needed to test usefulness. Their improvement could establish predictive usefulness of the stipulated hybrid; without onset validation it would still leave death-specific native attribution unresolved.

## Constraints: distinguish what changes

B1 preserves the old live XY emitter and its instantaneous fixed-latent action bound. The already-dead XY branch is constant in action. But its one-step law conditional on a *visible* state changes whenever its posterior has dead mass and the live law has escape mass. The augmented output includes D; D can jump as an action crosses a hazard-cell boundary. Therefore the old Euclidean guarantee does not certify the augmented state or the composed multi-step map. Metric and coupling must be specified and proved anew; neither weakening to a distributional metric nor dropping D from the norm is silently equivalent to the old requirement.

A persistence-only test starting from a simulator-confirmed D=1 snapshot can correctly enforce fixed XY and F4 shift, using that privileged initial fact. It addresses persistence alone. Setting every latent compatible with that F4 dead is unjustified, and an entirely post-death panel gives no decision-relevant onset/route comparison. E15 already motivates that check; it cannot satisfy the present Stage 2 gate by itself.

## Missing construction needed to proceed

Provide either (1) a joint sequential augmentation with A as its exact control marginal, a native-justified onset/noise/x' law, and explicit diagonal/constraint treatment; or (2) a simulator-assisted surrogate whose stipulated coupling is independently validated for the claimed onset population, accepting and quantifying any live-law/model-family changes. One-step diagonal compensation alone is insufficient. A native paired-snapshot simulator is available, but the bridge from its structural noise to A's generated branch is absent in the inspected implementation/artifacts. This is the precise unclosed gate. It does not say that offline identification or probability-one death for every visible alias is required.
