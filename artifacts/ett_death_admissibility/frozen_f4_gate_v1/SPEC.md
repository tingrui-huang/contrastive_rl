# Death representability gate, declared before component evaluation

Start from c5810d2f08a966c202189347a510fac8a6d0cd31 on
feature/pointmaze-causal-transition. No production source, checkpoint, actor,
critic, nominal, diagonal normalization or L=1 change. No new loss.

Read native windy F4 death/absorption, the actual frozen diagonal draw and emitter,
and saved native validation records before assigning categories. Native death is
the persistent _dead flag after a fatal landing, with subsequent actions ignored,
XY fixed, rewards zero, done still False; F4 shifts toward repeated positions.
Stationarity or a finite zero return is not a death definition.

Gate 1 must establish a model witness with absorbing consequences, not just a
reachable hazard endpoint. Distinguish a certified observable absorbing kernel
from native death conditional on an unobserved flag. The latter is not an ETT
input and one native hidden realization is not the diagonal conditional law.
If no justified witness can be certified in this interface/class, stop the
dependent objective comparison and two-seed optimization. Do not call them failed
tests or report a fabricated zero death rate.

## Fixed diagnostic cases and budget

Use the first saved shortcut episode by reset-seed order that contains a native
fatal transition, retaining fatal pre-state, first post-death F4, and death-age-3
F4. At each use its recorded execution action and two fixed nominal-action
conditions: (0,0) and the same action (diagonal). Labels remain in evaluation-only
output; generator conditions contain only s, x, x_prime and the commanded goal.

Add six geometry-specified cases: repeated (4.5,3.5) with x=(0,0) or (1,0),
x_prime=(0,0); reset (0.5,3.5), lower passage (1.5,1.5), and goal (8.5,3.5)
with actions (1,0), (1,1), (-1,0), respectively and x_prime=(0,0); upper approach
(2.5,3.5), x=(1,0), x_prime=(0,0). Histories repeat each named position. Synthetic
histories are geometry probes, not claims of observed hidden contexts.

Controls outside the hazard are justified using reachable native death locations
and the clipped per-coordinate one-step bound, not a predicted return. Native
death-possible cases must not be called ETT-admissible death unless absorption
and diagonal/Lipschitz constraints are jointly certified.

Budget: ZERO new native transitions, ZERO stochastic model draws/rollouts, ZERO
optimizer updates in this gate. Evaluate the frozen diagonal distribution at
12 fixed conditions and at three component noise points per coordinate
{-1,0,1}, for each of its three components: 324 deterministic projected support
points. Evaluate existing theta=0 and four saved final theta arrays in both
emitter modes on these support points (3,240 deterministic outputs), plus a
single analytically chosen hazard-entry endpoint certificate and a small fixed
action grid (at most 32 further outputs). These are not fresh sampled transitions
or fitted targets. Do not expand the panel on the basis of results.

For each condition retain logits, moving/atom probabilities (not death scores),
Gaussian parameters/normalization, support-point anchors, corrections, emitted
successors, selected sets and exact F4 checks. Analytic arguments, not the finite
panel or failed search, determine capacity. Preserve finite-precision caveats and
unresolved exceptional cases; never infer global impossibility from a finite grid.

Only if gate 1 passes would a separate small objective/optimizer budget need to
be declared before that execution. This gate does not authorize substitute
death labels, a new latent state, a modified diagonal law or an artificial
absorbing wrapper. Reuse existing native and model trajectory records instead.
