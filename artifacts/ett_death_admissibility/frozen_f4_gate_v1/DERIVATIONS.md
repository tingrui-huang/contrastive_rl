# Death, observable absorption, and the current admissible class

## Native semantics and the identification boundary

In `TwoRouteSwampWindyEnv.step` in `crl/envs.py`, `_dead` persists until reset.
Once it is true, execution actions are ignored, XY stays fixed, reward is zero,
and `done` remains false. Swamp-bit resampling does not revive the agent. The
F4 wrapper still shifts history, so three further transitions after the fatal
landing yield four copies of the death position. Death is this persistent native
condition, not a stopping criterion inferred from a finite trajectory.

An alive step adds the existing action noise, clips each action coordinate to
[-1,1], performs ten axiswise collision-checked substeps, and checks the landing
cell for an active swamp. Fatal landing positions have X in [3,6), Y in [3,4).
They are outside the radius-2 task-reward region around (8.5,3.5). Thus a native
death has zero remaining task return, independently of actions. The separate
success definition is not used as a death label.

There is an observable alias at the fatal landing. Fix an alive pre-state,
action, and action noise that land in a hazard cell. Switching that cell's
current swamp bit changes the death check but not the preceding physics. Both
successors have the same XY and F4 and reward zero; one is dead and the other
alive. This is a code-level construction, not a new simulator experiment. It
does not prove both contexts have positive conditional probability for every
specified natural action x_prime: that action may carry information about hidden
context. In particular, a saved hidden-dead record does not make its visible
ETT conditioning a supervised point-mass absorption target.

At a reachable position outside the hazard, the native process cannot already
be dead. The per-coordinate one-step bound excludes reaching any hazard from
(0.5,3.5), (1.5,1.5), or (8.5,3.5): respectively X <= 1.5,
X <= 2.5 and Y <= 2.5, or X >= 7.5. These are native next-step death-excluded
controls for every legal action, not claims about infinite-horizon survival.
Synthetic repeated histories are component probes; their full conditional data
support has not been established.

## Implemented stochastic transition

Fix visible F4 state s, goal g, natural action x_prime, coefficients theta, and
one diagonal random draw u. The frozen network is evaluated at
`[s, x_prime, x_prime, g]`. It draws either zero displacement or a three-component
Gaussian displacement, then applies the existing coordinate-step cap and
endpoint geometry guard. Write its resulting XY anchor as A(s,x_prime,g,u).

The trainable response is

    B_j = raw_j / max(1, ||raw_j||_F),   j=1,...,8
    M_theta(s) = sum_j w_j(s) B_j
    T_theta(s,x,x_prime,g,u) = P_C(A + M_theta(s)(x-x_prime)).

The nonnegative gates w_j sum to one. C is the anchor-containing reference
rectangle intersected with the rounded coordinate-step envelope, or the optional
fork segment when its action-independent availability conditions hold. F4 then
becomes `[T_X, T_Y, s[:6]]`. Each subsequent rollout step samples fresh nominal,
actor, and diagonal randomness. There is no persistent model death variable or
native flag input in this path.

For every coupled u, P_C(A)=A, so x=x_prime preserves the exact stored anchor and
its complete diagonal law. Also ||M||_op <= ||M||_F <= 1. Nonexpansiveness of
projection onto this one fixed closed convex set proves

    ||T(x)-T(y)||_2 <= ||M(x-y)||_2 <= ||x-y||_2

for all x,y in [-1,1]^2, with conditioning and u fixed. This is the prescribed
L=1, not an empirically identified native bound. It does not certify native
kinematics, absorption, or hidden-state identification.

## Necessary absorption condition and exact class obstruction at the panel

Let q be the recorded death position and H(q) its repeated F4. Certain
observable absorption requires that each subsequent kernel keeps q with
probability one for all execution actions, including after H(q) is reached.
A necessary condition on the diagonal is A=q almost surely, since T(x_prime)=A
for every theta. The frozen law violates this condition at the tested first
post-death and repeated-history conditions: its moving component has positive
mass and nonzero scale, with unmodified free anchors away from q.

There is a stronger off-diagonal obstruction here; it does not rely on a
continuous actor sampling exactly x_prime. At the tested q near (4.5,3.5025)
and at (4.5,3.5), the step envelope intersects only the upper corridor rectangle.
It cannot reach the left, lower, or right passage. The fork segment is
unavailable. Thus its X bounds l_X,u_X are fixed across anchor draws and q_X is
strictly between them. For any fixed x,x_prime and *any* theta, c=M(s)(x-x_prime)
is independent of the anchor draw. Consequently

    T_X = clip(A_X + c_X, l_X, u_X),
    T_X = q_X  if and only if  A_X = q_X - c_X.

The moving Gaussian branch has positive density on an open free region where
the anchor guard is the identity and A_X is nonconstant. No single c_X can
map that entire positive-mass region to the interior point q_X. Therefore
P(T=q)<1, even for a fixed off-diagonal action. This holds for every coefficient
choice, including coefficients with looser matrix bounds, under this anchor
and rectangle rule. It is not an inference from optimization or an action grid.
It also holds at the tested first post-death F4: changing its old frames changes
the distribution and matrix, but not this argument.

`diagonal_support.json` gives two distinct unmodified support anchors and the
network parameters at each tested condition. These are deterministic probes
supporting the code-grounded density argument, not empirical probability
estimates. The argument uses the intended real-arithmetic Gaussian law. Actual
float32 sigmoid probabilities in this panel are strictly between zero and one;
finite-precision saturation or degenerate geometry at untested contexts is not
globally ruled out by these checks.

The absence of a flag alone would not prove this obstruction: a visible-state
kernel can in principle be absorbing. Here the frozen noisy anchor, its exact
preservation, and the anchor-independent correction prevent certified certain
absorption at these states. At fixed repeated history and actions, a stay
probability p<1 gives p^r for r further repeated stays under fresh independent
draws. Finite stationarity is therefore insufficient as an absorbing certificate.
The model interface also does not define which of its mixed stationary outputs
would constitute a persistent latent death event. A possible latent
decomposition of a finite observable path law remains unidentified; this audit
does not prove that every conceivable latent extension is impossible.

## What this obstruction does not say

L=1 by itself does not exclude absorption. Hypothetically, if the diagonal at
an absorbing condition were A=q almost surely, M=0 would give T(x)=q for every
action, with Lipschitz constant zero, diagonal consistency, and valid geometry
and F4 shifting. That is not the current frozen diagonal and is not an admissible
replacement in this task.

Conversely, entering a hazard is already possible in the current class. At
s=H(2.5,3.5), x_prime=(0,0), choose all eight matrices as diag(1,0). Their
Frobenius norms equal one. For the positive-probability zero-displacement anchor
A=(2.5,3.5), the selected upper rectangle gives

    T(x) = P_C((2.5,3.5) + (x_X,0)).

This fixed map preserves the diagonal and the all-action-pairs bound and emits
(3.5,3.5) at x=(1,0). `hazard_endpoint_only.json` records its fixed coefficients
and a nine-action implementation check. The complete emitter with these
coefficients also preserves every other diagonal anchor draw. This is a
full-action *one-step hazard-entry* example, expressly not an absorbing-death
witness, native ground truth, or training target.

## Objective implication, conditional only

The implemented objective is constant L_diag plus lambda E[J_model], with
lambda=1/sum_{t=0}^{49}0.95^t and successor-position rewards in {0,1}. If a
verified absorbing hazard witness were feasible, its remaining J would be zero.
Under identical starts and prefix rewards it could not have higher objective
than a survivor; it would tie survivors with zero finite-horizon return and
strictly improve on positive-return alternatives. This algebra neither supplies
such a witness nor distinguishes death from harmless zero-reward behavior.
No paired witness/survivor objective comparison or optimizer test is justified
until a compatible persistent death event or absorbing kernel is established.
