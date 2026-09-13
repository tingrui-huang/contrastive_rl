# Death repair comparison: Stage 1 protocol

Recorded 2026-09-13, after source/ledger reconnaissance and before any new numerical experiment. This is a construction gate, not a retrospectively preregistered experiment. Local and remotely queried PointMaze HEAD are both f85a5f44b176120333a1de0f83d6d0d02a2b290e. The parent checkout is on AntMaze and is excluded. No newer PointMaze commits exist at the remote check.

## Hypotheses and scope

H1: incorrect death onset/persistence materially explains dangerous-action continuation overestimation under the frozen actor. H2: repairing that mechanism improves native policy outcomes beyond matched baseline-model training and ordinary offline continuation. H1 does not imply H2.

Before sampling, determine whether B can be a verified repair of A rather than an arbitrarily more pessimistic process. A is the rectangle/Frobenius ModeKernel with the two inherited 48-coordinate s0_k0_u24 and s1_k0_u24 response-search checkpoints (16 diagonal offsets, 32 response coefficients), frozen nominal and actor, with no new updates. Inspect the latest norm-comparison implementation and provenance to fix the relevant interfaces. This specifies candidate baselines, not a claim that their empirical death rate exists.

B must carry a branch-specific event D and use the same architecture in both arms wherever possible (an unused auxiliary D in A is allowed). Native death means the simulator's persistent _dead flag, first set on landing in a currently active cell, followed by fixed XY, shifting F4, ignored execution actions and zero rewards until the original horizon. Stillness, the stationary mixture atom, zero return and hazardous-cell membership are not death.

## Fixed construction audit

Review exactly these candidate classes; do not run a detector, new critic, optimizer sweep or benchmark:

1. Shared original live kernel, branch-specific latent onset, persistence switch off/on. Specify the joint law of latent mask/death, natural action x', transition noise and anchor, including its sequential update and initial latent law.
2. Same augmented architecture with a marginally compensated live kernel, to retain the frozen visible diagonal after adding persistent death.
3. Simulator assistance: (a) copying a reference event; (b) transplanting generated XY into a simulator; (c) generating a joint native teacher/action/latent branch; (d) defining a fresh-mask killed learned process. Separate an internally consistent synthetic diagnostic from a verified native repair.

For each, record onset versus persistence, differing variables/parameters/sampling, privileged inputs, diagonal law, full action-Lipschitz guarantee and model-family changes. Consider a persistence-only post-death check, but do not promote it to evidence about pre-outcome decisions.

Gate passes only with: an explicit branch-local event/noise law; correct handling of x'; verified onset semantics for the claimed target; persistent consequences; and an interpretable A/B intervention with all inseparable changes declared. Offline identifiability is not required for a clearly specified simulator-assisted diagnostic, but native repair cannot be certified from a synthetic event by definition. A finite latent snapshot is a paired individual context, not a sample from a visible conditional ETT posterior.

If no candidate passes, stop at Stage 1 and state the missing construction/assumption. Do not replace A with native physics and call that a repair comparison of the learned model. Do not claim a universal impossibility result.

## Metrics and decision rules if a construction were available

The following are fixed interpretation criteria, not an executable Stage 2/3 design. A successful gate would additionally require a sealed manifest of contexts, seeds, code, checkpoints and exact transition accounting before any rollout. No such numerical phase is permitted by this protocol alone.

Use fixed goal (8.5,3.5), original next-position radius-2 reward, gamma=.95, original t=50 limit, and normalized occupancy .05*sum(.95^k*r_k); report raw returns alongside it. Death does not terminate the horizon. Pre-outcome roots and dangerous/safer alternatives must be selected without future failure labels. Post-death roots form a separate repair check. Freeze actor and nominal. Couple initial snapshots and independent time-indexed actor/transition/hazard noise slots, not an RNG whose consumption shifts after death. Native outcomes remain evaluation-only except explicitly declared diagnostic latent inputs.

Report branch-event validity and persistent XY/reward/F4 checks; dangerous-action signed bias and absolute error; dangerous-minus-safe contrast error; independent full-rollout return; non-death transition error; diagonal fit/law and constraint changes. Recompute the A/reference discrepancy on the same population: do not import the E20/E21 aggregate gap as a paired denominator. Main decision evidence cannot be restricted to trajectories selected for realized death.

For H1, a practically major correction requires at least .02 normalized dangerous-action error reduction AND at least 25% of that population's positive baseline discrepancy removed, with at least .01 reduction of absolute dangerous-versus-safe contrast error. The relevant paired intervals must support improvement, and both fixed model seeds must meet the criteria. A return decrease alone does not pass. Report error overshoot and non-death degradation; a worsening safe-action error of .01 or more invalidates a clean mechanism claim. Ratios with a nonpositive or unresolved baseline denominator are undefined. These cutoffs are prospective judgments, not prior findings.

If H1 passes, H2 requires matched actor initialization/recipe/budget under A/B plus ordinary offline continuation, actors sealed before fresh paired native evaluation, and at least .02 normalized return improvement and 5 percentage-point reduction in actual absorbing failure versus both controls, consistently across the declared training seeds with intervals supporting improvement. Report success, route occupancy and action/parameter change separately. No claim of general training-seed reliability from only two fixed seeds. Independent units are originating episodes/snapshots for context effects and paired native reset seeds conditional on sealed actors; descendant steps/draws are not independent episodes. Separate context uncertainty from conditional Monte Carlo uncertainty. No significance-only major-cause claim.

## Budget and stopping

This Stage 1 audit permits source/report reads, analytic derivations, file hashing and artifact verification only. Fixed budget: zero simulator transitions, zero stochastic model draws, zero deterministic emitter calls, zero actor/critic/ETT/nominal updates. Stage 2 and Stage 3 allocated budgets are zero until the construction gate passes and a complete numerical supplement is sealed. No result-driven budget extension. If the gate fails, complete the report and ledger and stop.

## Prior-experiment audit and provenance

Reuse E02 for death observability limits; E10 for frozen-family support/geometry; E15 for sampled native-dead roots followed by model motion; E19 for genuine calibrated critic/MC optimization; E20/E21 for native policy and model/native discrepancy; E22-E26 for expanded search, contrast precision/sign, oracle-family and norm corrections. E12-E14 already establish calibrated finite-state feasibility. Do not rerun these. Novelty is the joint branch-event construction and repair-attribution gate, not another absorption capacity audit.

Keep the source ledger byte-for-byte as a snapshot, record hashes of reviewed inputs and this protocol, and append E27 to the requested external ledger after the report is complete. Preserve all historical recommendations in place, marking their current applicability explicitly in the new entry. Use this fresh directory; do not commit or push.
