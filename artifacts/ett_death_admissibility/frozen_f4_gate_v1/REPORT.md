# Can the current pessimistic transition search find death when admissible?

**The representation/admissibility gate is unmet.** At the saved native death
contexts inspected here, no choice of the current 32 coefficients can produce
certain observable absorption under the frozen anchor law. Moreover, the ETT
interface has no defined persistent death event. A verified model death rate is
therefore **undefined**, not zero. The dependent objective comparison and
optimization were not run. A decrease in model return would not resolve this
failure of the diagnostic prerequisite.

Started at **c5810d2f08a966c202189347a510fac8a6d0cd31**, on
`feature/pointmaze-causal-transition`; no subsequent production changes or
checkpoint updates. [Provenance](provenance.json) records 153 input hashes,
including the frozen components and prior experiment sources. All remained
unchanged. This task adds only the analysis script and this audit directory.

## Established findings

Native death is the persistent `_dead` condition: later actions are ignored,
XY stays fixed, rewards remain zero, and `done` remains false. F4 keeps shifting
until all four frames coincide. This differs from a stationary successor or a
zero 50-step return. The first saved shortcut failure by reset-seed order is
seed **51000001**, fatal transition **3** (zero-based), at approximately
**(4.5, 3.5024967)**. Its next **46** recorded transitions retain exactly the
same XY and zero reward despite nonzero execution actions.

The predeclared [12-condition panel](SPEC.md) contains the fatal pre-state,
first post-death state, and repeated post-death history under two fixed nominal
actions; synthetic hazard probes; and three geometry-based controls. Native
death is excluded on the next step from reset (0.5,3.5), lower passage
(1.5,1.5), and goal (8.5,3.5) under every clipped action. Native death is possible
in the hazard cases, but **none is assigned an ETT-admissible absorbing-death
label**. [Hidden validation](native_validation_only.json) is separate from the
[visible generator conditions](conditions.json). No hidden field enters the
network or emitter, and no native outcome is used as a training target.

The ETT draws a fresh frozen diagonal anchor and emits a convex projection of
`anchor + M(state) @ (x - x_prime)`. It carries F4, not persistent death state.
Exact diagonal consistency fixes every output at `x=x_prime` to that anchor.
At the repeated saved death history, the raw stationary-atom probabilities are
**0.937099** for `x_prime=(0,0)` and **0.925549** for the recorded-action nominal
condition. These are neither death probabilities nor total stay probabilities:
geometry projection can add stationary mass. Both moving branches have positive
scale and distinct unmodified free anchors. For example, at the zero-nominal
condition two such anchors are **(4.0138984,3.1463451)** and
**(4.4995770,3.1463451)**. Full values are in [diagonal support](diagonal_support.json).

The obstruction extends beyond the diagonal. At these hazard-interior positions
both modes use the upper rectangle, with current X strictly inside its bounds.
For any fixed action pair, clipping `A_X+c_X` to those bounds equals current X
only when `A_X=current_X-c_X`. A single anchor-independent correction cannot
collapse a positive-density range of anchor X values to that interior point.
This proves lack of probability-one absorption at these conditions for **all
32-coefficient choices**, not merely the learned coefficients. The
[derivations](DERIVATIONS.md) state its assumptions and scope.

## What remains unresolved

This is not a proof that the prescribed L=1 inherently forbids death, or that
every imaginable latent model is incompatible with diagonal data. An absorbing
point-mass diagonal would admit a constant, zero-Lipschitz response; the frozen
diagonal here is not that law. First-postfatal visible histories can also alias
alive and dead native contexts. One hidden-dead realization is not conditional
ETT ground truth, and the current interface does not identify a persistent latent
death component within a mixture. Untested numerical saturation and degenerate
boundary contexts are not classified by this panel.

A fixed admissible matrix does produce a hazard landing from the upper approach
while preserving the full action bound and diagonal law. Its
[existence certificate](hazard_endpoint_only.json) establishes one-step reachability,
**not death**. The missing absorbing consequences cannot be supplied by calling
that endpoint fatal or by adding an absorbing wrapper.

The objective is unchanged: constant diagonal fitting term plus positive lambda
times expected discounted task return. A hypothetical feasible zero-return death
would tie zero-return survivors and beat positive-return survivors. This is only
conditional algebra. **Objective preference and optimization ability remain
untested**, because no certified death-producing witness passed the first gate.

## Verification and accounting

There were **zero new native transitions, stochastic model draws, optimizer
updates, or actor/critic updates**. Deterministic checks used 324 projected
Gaussian support points, zero coefficients plus the four existing final arrays
under both modes, and one fixed nine-action hazard-entry example. All selected-set,
F4, exact diagonal-identity, and matrix-norm checks passed; the example's finite
action-pair checks agreed with the analytic bound within **2e-6**. These finite
checks support implementation correctness, not continuous-domain proofs.

For all tested first-post-death and repeated-death conditions, all 27 deterministic
moving-branch probes emitted motion with each existing coefficient array in both
modes. This is a support-panel result, **not a death-rate or escape-probability
estimate**; see [coefficient comparisons](current_coefficient_components.json).
The rounded float32 coordinate-step contract and its previously documented
approximately 1.19e-7 real-arithmetic excess remain unchanged; anchors and L
were not adjusted.

Actual deterministic emitter accounting is 3,240 main outputs, 3,240 separate
diagonal-identity outputs, and nine example outputs. The predeclared SPEC counted
the main outputs but omitted the separate identity calls from its deterministic
tally. This accounting omission is retained explicitly; there was no added
sampling, optimization, or result-driven panel expansion.
[Verification](verification.json) and [reproduction](REPRODUCE.md) retain the details.

**Next step:** before further pessimistic search, specify a persistent latent
death event and establish whether its absorbing consequences can coexist with
the frozen visible diagonal law after marginalization. The necessary evidence
is a compatible multi-step construction or a precise incompatibility result;
another return-only optimization or downstream CRL experiment is premature.
