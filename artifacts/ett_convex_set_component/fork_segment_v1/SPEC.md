# Optional fork segment: construction specified before implementation/evaluation

The mode is explicitly `rectangle` (unchanged default) or `fork_segment`.
No selector training, coefficient change, simulator query, or rollout is part of
this component task. L remains the prescribed 1. The selector receives only
current visible XY q and the exact sampled diagonal anchor A. Historical F4,
goal, nominal action, parameters and coupled randomness are fixed conditioning;
the selector does not need them once A is supplied.

Let E=[fl32(q-1),fl32(q+1)] be the existing coordinatewise rounded step box.
Use the existing guarded corridor limits: top [0,9] x [3,4^-] and left
[1,2^-] x [1,4^-], where c^- is the preceding float32 value. First obtain
the original anchor-containing rectangle and its validity flag.

A segment is considered only for 1<=qX<2, 3<=qY<4, and a valid anchor in
the top corridor with AX>=1. Define b=max(1,E_low,Y), ell=max(1,E_low,X).

* If AX<=2^-, take Q=(AX,(3+b)/2). Both endpoints lie in the left corridor.
* Otherwise require h=AY-3>0. Fix K=(1.875,3), an interior junction point,
  not an action-specific audit target. Set d=AX-1.875 and
  t=min((3-b)/(2h),(1.875-ell)/(2d)); Q=K+t(K-A).
  This spends at most half the available downward and leftward clearance.
  A--Q crosses Y=3 at K, so its upper portion is in the top corridor and its
  lower portion in the left corridor. Endpoint inclusion in convex E implies
  inclusion of the entire segment in E. Neither q nor the simulator path
  from q to the emitted endpoint is required to lie on this segment.

Accept only finite endpoints in E, QX in [ell,2^-], QY in [b,3), and squared
length greater than 1e-12. For the ray case additionally check the rounded
endpoints' Y=3 crossing lies at or left of 1.9375 using the cross-multiplied
inequality (AX-1.9375)(3-QY) <= (1.9375-QX)(AY-3). This leaves 1/16 clearance
from the wall (the construction itself leaves 1/8). All these checks are
execution-action independent. Unavailable, degenerate, boundary-outside-left
AY=3, below-top anchors, and unsupported current contexts use the original
rectangle. An invalid original anchor remains a contract error. No anchor moves.

For selected C=[A,Q], project P=A+M(s)(x-x') by
u=clip(<P-A,Q-A>/||Q-A||^2,0,1), output=A+u(Q-A).
Return exact A/Q at u=0/1; the standalone degenerate projection returns A.
The fallback is exactly the previous rectangle projection. Append s[:6] to
the emitted XY, preserving newest-first F4 history.

## Guarantee and numerical contract

Each signed matrix block is divided by max(1, its Frobenius norm), unchanged.
The softmax mixture M has operator norm <= Frobenius bound <=1. Fix s,g,x',A,
parameters and mode. C is then one fixed closed convex set containing A.
Euclidean projection is nonexpansive; consequently for ALL x1,x2 in [-1,1]^2,
||T(x1)-T(x2)||2 <= ||M(x1-x2)||2 <= ||x1-x2||2. Identical appended history
cancels. At x=x', T's XY is exactly A. The unchanged upstream diagonal draw
and this pointwise identity preserve its full sampling law under the same noise.

These statements describe the real-arithmetic mapping, with the stored rounded
endpoints/step bounds treated as constants and the inclusion checks certified
as above. Float32 implementation tests use absolute tolerance 2e-6 for geometric
and action-pair residuals, and exact array equality for diagonal and history.
They are not a formal bit-level all-pairs certificate. The twelve audited
float32 anchors exceeding a literal unit coordinate step by 1.19e-7 remain
unchanged: the rounded envelope contract is retained, not repaired or enlarged.
L is not a simulator estimate. Segment/box admissibility proves neither native
kinematics, hidden absorption, causal identification nor successful rollouts.

The available family gains an optional member. A segment instance need not
contain all outputs of the previous rectangle instance. No nonconvex union
projection, action-dependent rectangle switch, output average, imposed response
sign, route waypoint, privileged label, or off-diagonal supervised target is used.

## Fixed verification scope

Use all 12,800 saved lower-controller component inputs from model seeds 0/1;
summarize the previously declared fork/descent subset separately. Do not feed
modified outputs forward. Compare the frozen learned coefficients in both modes.
For an existence certificate use the first prior certified seed-0 example in
reset/time order, construct its segment using this rule, choose one analytic
rank-one matrix to reach Q if admissible, freeze it, then check other actions.
Finite deterministic action pairs and boundary fixtures supplement the proof.
No new trajectories, optimizer calls, coefficient fitting, or policy training.
