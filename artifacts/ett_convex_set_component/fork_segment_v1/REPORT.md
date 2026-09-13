# What additional transitions can this implementation represent, and which guarantees remain intact?

**The optional emitter can represent a continuous action-dependent turn from an
upper-corridor anchor into the left passage that the original selected rectangle
excludes. Exact diagonal preservation, the prescribed full execution-action
L=1 bound, declared endpoint geometry and F4 shifting retain the same convex
projection argument. The default rectangle emitter is unchanged.**

This is a component implementation and verification, not a trained transition
model or navigation result. On existing frozen coefficients, it adds **zero
below-Y=3 entries** in the recorded fork/descent subset. The audit's two findings
remain separate: geometry excluded admissible responses, and the learned
coefficients generally did not propose effective descent. Removing the former
does not establish that geometry caused the failed CRL integrations.

## Construction and guarantees

The exact construction was recorded in [SPEC.md](SPEC.md) before implementation
and component evaluation; its hash is in [provenance.json](provenance.json).
`ConvexActionTransition(..., geometry_mode='fork_segment')` opts in;
`geometry_mode='rectangle'` remains the default. The original `emit` function
is retained verbatim, and `emit_with_geometry` is the explicit component API.
The mode is fixed at construction; create a new instance to change it.

For current XY in [1,2) x [3,4), an eligible upper anchor A defines one fixed
segment [A,Q]. Anchors already inside the left passage use a vertical segment.
Other eligible anchors extend the ray through the fixed interior junction
K=(1.875,3), using half the available horizontal/downward clearance. Endpoints
depend only on current XY, A, public geometry and the existing rounded step box.
They never receive execution action, nominal action, waypoint, return, simulator
state or a failure label. The current action affects the scalar projection
coordinate through the unchanged signed matrix response; it can continuously
select a point on either side of Y=3 within this fixed segment.

The ray crosses Y=3 inside the overlap, so splitting the segment there proves
inclusion in the two free rectangles. Its endpoints are in the convex step box,
so the whole segment is too. Rounded-endpoint crossing checks leave an additional
wall margin. Degenerate segments, unsupported contexts and unavailable rounded
constructions fall back to the original anchor-containing rectangle, with no
action-dependent decision. Invalid original anchors still raise a contract error.
No projection onto a nonconvex union or averaging of mode outputs is introduced.

For fixed state, goal, x_prime, anchor, mode and coefficients, let
T(x)=Pi_C(A+M(s)(x-x_prime)), with the identical history appended.
The eight Frobenius-normalized blocks have norm <=1 and their nonnegative
softmax mixture has operator norm <=1. Projection onto this one fixed closed
convex C is nonexpansive. Thus for **every pair** x1,x2 in [-1,1]^2,
||T(x1)-T(x2)||2 <= ||M(s)(x1-x2)||2 <= ||x1-x2||2.
Because A belongs to C, x=x_prime returns the exact stored A samplewise.
The upstream frozen diagonal draw and its coupled randomness are unchanged,
so the full diagonal sampling law is preserved. Appending s[:6] shifts F4
exactly; these fixed coordinates cancel in the action-pair bound.

The union of available mode families expands the choices. An individual segment
instance does **not** contain every output of the previous rectangle instance.
Neither response sign nor a native off-diagonal target is imposed. L=1 remains
prescribed, not identified from simulator/offline data. These guarantees do not
prove native kinematics, hidden absorption, causal identification or multi-step
route validity. They rely on the declared visible free geometry and endpoint
step envelope, not additional physical motion evidence.

## Numerical contract and focused verification

All saved anchors retain their exact float32 values. Four seed-0 and eight seed-1
fork anchors still exceed a literal real-arithmetic unit coordinate step by
**1.1920928955078125e-7**. The contract remains inclusion in the original
float32-rounded q+/-1 box. No anchor was moved and L was not enlarged. The
real-arithmetic proof and float32 checks are distinct: geometry and action-pair
residual tolerance is 2e-6; diagonal/history checks require exact array equality.
This is not a formal bit-level global Lipschitz certificate.

The deterministic verifier consumed **12,800 saved component inputs**, 6,400
per frozen checkpoint. No modified output was fed into a later state; no new
trajectory or stochastic anchor draw was generated. Six focused tests plus
four existing emitter tests cover:

* Bitwise equality of default outputs and every diagnostic against a byte-preserved
  pre-change module, with identical parameters, inputs and anchor arrays.
* Exact diagonal and F4 equality in both modes, including multiple draws and
  public sample API tests with deterministic stub draws through the real projector.
  The diagonal context remains [state,x_prime,x_prime,goal].
* Action-invariant segment selection/endpoints, selected-set membership, step/free
  geometry, exact endpoint returns, degenerate projection, invalid anchors,
  guarded boundaries and fallback cases.
* Twenty-seven fixed grid/near-pair actions at each of the 225 fork contexts with
  current coefficients, plus saturated-matrix fixtures and the frozen witness.
  Maximum positive all-pairs bound excess was zero in the saved/witness checks.
  This is an implementation check, not the continuous-domain proof.

Default batched reconstruction differs from historical sequential saved outputs
by at most 9.536743e-7 for either seed, within the existing 2e-6 tolerance.
It is bitwise identical to the preserved implementation at the same batch/JIT
shape. Segment endpoint/crossing inclusion was independently checked in float64;
the largest ray crossing X was 1.875000267, below the 1.9375 guard.

## Proven extra representational capability

The deterministic example is the first previously certified seed-0 context by
reset/time: **51000001, step 1**. Its anchor is (2.405497,3.352175).
The new selector, before consulting any action, constructs Q=(1.532920,2.772907).
With d=x-x_prime, choose all eight blocks equal to
M=(Q-A)d^T/||d||^2. Its Frobenius norm is **0.87388743 < 1**.
Freeze these blocks and the mode before checking other actions.

At the recorded execution action, the segment emitter returns Q; the original
rectangle returns (1.532920,3.000000). This is an excluded response belonging to
a fixed full-domain Lipschitz function, not just a point in an anchor-distance
ball. [existence_certificate.json](existence_certificate.json) retains the exact
inputs, matrix, outputs and action-pair checks. The matrix is an **existence
certificate**, not trained coefficients, a ground-truth transition or a
supervised target. Audit action-specific targets were not copied into runtime.

## Observed behavior of the current frozen coefficients

The fork/descent subset is current X in [1,2), Y in [3,4), active waypoint 1.
The waypoint is used only for retrospective grouping, never by the emitter.
Recomputed original upper-box counts are 119/122 and 100/103.

* **Seed 0:** optional segments are selected at 119/122 fork steps. All 119 outputs
  change above tolerance; mean XY change over the subset is 0.073552, maximum
  0.120451. Downward relative to current Y increases from 44 to 62 steps, but
  below-Y=3 entries remain **3 to 3**.
* **Seed 1:** segments are selected at 99/103 fork steps. All 99 outputs change;
  mean change is 0.065232, maximum 0.137739. Downward steps increase from 30 to
  46, but below-Y=3 entries remain **3 to 3**. At reset 51000030 step 12,
  current Y=3.99999976 leaves so little descent clearance that Q rounds to Y=3;
  the predeclared availability check falls back. The six already-below-top
  anchor cases across both seeds also retain rectangles.

Across all 6,400 saved inputs per seed, 143/134 select segments and change outputs.
Mean XY changes are 0.001582/0.001317; below-Y=3 outputs remain 83/63 respectively.
These are conditional one-step comparisons on fixed saved inputs, not new route
adherence, success or return estimates. The coefficients were optimized with
old geometry and a different fixed actor; lack of new entry is not a failed
component test. Shared-coefficient usefulness and later-state behavior remain
unresolved.

## Provenance, artifacts and later integration

Actual starting commit and requested audit commit are both
`b54fadd32dedf199c05f93dea9f2de82defe10c0`, on
`feature/pointmaze-causal-transition`; there were no intervening commits.
Only the transition source, focused test/verifier, this new artifact directory
and scoped Git byte-preservation attributes change. The original 103 other
provenance inputs, including experiment artifacts and frozen checkpoints, were
hashed before and after verification and remain unchanged. Prior hash manifests
are preserved; the intentional transition-source before/after hashes are
recorded in [verification.json](verification.json).

[comparison_summary.json](comparison_summary.json) and
[fork_comparison.csv](fork_comparison.csv) contain recomputed counts and all 225
fork comparisons. `s0_component_outputs.npz` and `s1_component_outputs.npz`
retain all saved conditioning, actions, anchors, both emitted F4 outputs,
reference rectangles and segment diagnostics. Numerical checks and exact witness
data are reproducible with [REPRODUCE.md](REPRODUCE.md).

Later, an experiment can explicitly construct this model with
`geometry_mode='fork_segment'` and persist that choice with its coefficient
checkpoint/configuration. Existing ETT `sample_flat` and CRL `make_generator`
already consume the same successor shape; their nominal-action ordering and
future sampling need no mathematical change. Future instrumentation should
retain geometry_kind/endpoints and interpret box fields as reference/fallback
diagnostics, rather than require segment outputs inside the old box. Legacy
checkpoints/configs continue to select rectangles unless explicitly opted in.
No training launcher, pessimistic objective, CRL learner or replay pipeline
was changed or run here.

**Next implementation step:** in a separately authorized model experiment,
persist and propagate the geometry mode and selected-set diagnostics through
the existing ETT runner before evaluating whether pessimistic optimization uses
this capacity; downstream CRL benefit remains a later question.
