# Transition map, capacity and admissible alternatives

All norms below are Euclidean, except explicitly marked Frobenius matrix norms. Execution actions lie in D=[-1,1]^2. Fix visible F4 state s, commanded goal g, observational action x', and coupled sampler randomness z. Write q=s[:2]. The prescribed L remains 1. No bound across different s, x', g or z is assumed.

## 1. Implemented computation

The authoritative implementation is [convex_action_transition.py](../../../ett/convex_action_transition.py), particularly `gates` (line 23), `matrices` (31), `convex_box` (37), `emit` (49) and `sample_flat` (73).

1. The frozen diagonal network receives normalized context `[s,x',x',g]`. Its zero-atom / categorical-mixture / Gaussian draw produces a displacement. The frozen diagonal decoder clips displacement coordinates to [-1,1], clips position to global bounds, and replaces a blocked endpoint with q. Let A=A(s,x',g,z) be the resulting XY anchor. See [diagonal_transition.py](../../../ett/diagonal_transition.py), `sample_displacement` (151) and `_project_samples` (205). This entire emitted anchor law, including its atoms and corrections, is frozen. In this audit A and x' are read from saved arrays, not resampled.
2. Eight fixed softmax gates w_j(s) use newest X and RMS motion over the four frames. Centers are X=(.5,2.5,4.5,7.5), crossed with motion=(0,.7), with scales 2 and .35. They do not condition on absolute Y separately, x', g or A. Decode eight 2x2 blocks B_j=theta_j/max(1,||theta_j||_F), and form M(s)=sum_j w_j(s) B_j.
3. Intersect each of four fixed free rectangles with q+[-1,1]^2. The rectangles cover the top corridor [0,9]x[3,4), lower corridor [1,8)x[1,2), left passage [1,2)x[1,4), and right passage [7,8)x[1,4). Interior upper edges are the predecessor float32 values. Among intersections containing A, select the largest area, breaking ties in listed order. Call the selected closed rectangle C(s,A). Selection is independent of execution action and all 32 coefficients. An uncovered anchor produces an error, not a new fallback.
4. Compute d=x-x', correction M(s)d, proposal P=A+M(s)d, and emitted XY F(x)=projection_C(P), implemented by componentwise clipping.
5. Return T(x)=(F(x),s[:6]). The commanded goal remains external conditioning; history is shifted exactly and does not depend on the current execution action.

Because ||M||_op <= ||M||_F <= sum_j w_j||B_j||_F <= 1, and projection onto a fixed closed convex set is nonexpansive,

    ||T(x1)-T(x2)|| = ||projection_C(A+M(x1-x'))
                          -projection_C(A+M(x2-x'))||
                     <= ||M(x1-x2)|| <= ||x1-x2||.

This is a full action-domain statement, not merely a bound relative to x'. At x=x', the correction vanishes and A belongs to C, so T(x')=(A,s[:6]) samplewise. Consequently the entire diagonal sampling law is preserved. The same coupled-noise argument gives a distributional W2 bound when that is the desired interpretation. It does not identify the true observational or interventional laws.

For completeness, nonexpansiveness of convex projection follows from its variational inequalities: for p=projection_C(u), q=projection_C(v), adding `<u-p,q-p> <= 0` and `<v-q,p-q> <= 0` yields `||p-q||^2 <= <p-q,u-v> <= ||p-q|| ||u-v||`.

The formulation supplies diagonal consistency, the chosen fixed-conditioning action bound, and a pessimistic objective inside an admissible class. Free endpoints, coordinate step limits and the known maze are declared visible feasibility restrictions. Freezing an imperfect estimated diagonal law, using eight particular state gates, Frobenius normalization, affine signed-action correction, four axis-aligned rectangles, and largest-area selection are implementation choices. None follows uniquely from diagonal consistency or L=1. L=1 itself is prescribed; it is not identified from the native simulator or diagonal data.

These guarantees do not establish native substep reachability, absorption, action-response correctness, causal identification, or multi-step physical validity. In particular C need not contain q. The earlier specification already acknowledges rare diagonal corner crossings. Pessimism does not require reproducing every native transition or ranking policies by natural expected return.

## 2. Exact fixed-query capacity of the existing coefficient class

Hold s,x',A,C and one execution action x fixed, and write r=||x-x'||. Every decoded M lies in the Frobenius unit ball. Conversely every matrix in that ball is attainable at this one context by assigning all eight blocks that same matrix. Thus the gate mixture imposes no additional restriction on the set of single-context matrices.

For r>0, the exact set of possible pre-clipping corrections is the closed XY ball of radius r. Necessity follows from ||Md||<=r; for any vector v with ||v||<=r, the rank-one matrix `M=v d^T/r^2` attains Md=v with `||M||_F=||v||/r<=1`. It follows that the exact emitted set for this fixed query is

    C intersect Ball(A,r).

Necessity uses nonexpansiveness and A in C; sufficiency takes a desired output Q in this intersection and sets v=Q-A so clipping does nothing. Therefore

    min_theta emitted_Y = max(C_low,Y, A_Y-r).

The upper bound is immaterial because A is already in C. The minimum is attained with all blocks equal to `-e_Y d^T/r` (or M=0 when r=0). This is an exact single-query result, not a relaxation. The script verifies attainment to 4.45e-16 in independent float64 arithmetic.

For recorded upper rectangles, C_low,Y=3. They prohibit Y<3 regardless of coefficients, but do not prohibit all motion with negative Y displacement. The minimum is below the recorded current Y in all 225 contexts. One **single shared coefficient choice**, all eight blocks `diag(0,1)`, also moves downward at every saved context and crosses Y=3 in all six vertical-box contexts. It is evaluated arithmetically on saved states only, with no rollout or fitting. This establishes available downward response that the learned coefficients often do not exhibit, without assuming that this response must be selected by the pessimistic objective.

The shared class still has limitations. Gate sharing cannot generally attain different independently chosen rank-one optima at all contexts. Identical X/motion gates give identical M despite differences in Y, nominal action or anchor. The correction is affine before projection and common across anchor draws. We neither solve a simultaneous interpolation problem nor claim a globally successful route policy exists in this class from these local bounds. Changing coefficients changes later contexts, nominal draws and anchors; fixed-trace results are not a theorem that the class can never traverse the route.

Frobenius normalization is also stricter than the stated operator-norm requirement. At q=A=(1.5,3.5), x'=(0,0), use the current selected box C=[.5,2.5]x[3,4-epsilon]. The map `projection_C(A+x)` is globally 1-Lipschitz, diagonal-preserving and feasible. Around x=0 its derivative is I. No current M with Frobenius norm <=1 can equal that map on an open neighborhood, because ||I||_F=sqrt(2). This is a separate proven class restriction; it is unnecessary to change it to realize a single fork-turn response, where rank-one matrices suffice.

## 3. Endpoint balls versus globally action-Lipschitz functions

Let G_q be the known free endpoint set intersected with the step envelope, and assume A belongs to G_q. The anchor-ball condition Q in G_q and ||Q-A||<=r is necessary but not sufficient in general.

For any admissible full-domain F, map the straight action segment from x' to x through F. This creates a free output curve from A to F(x) of length at most r. Thus the **intrinsic free-path distance** from A to Q must be <=r. Conversely, if a unit-speed free path gamma from A to Q has length ell<=r, define, for all u in [-1,1]^2,

    v=(x-x')/r,
    F(u)=gamma(clip(<v,u-x'>,0,ell)).

Here v and gamma are fixed constants after selecting the conditioning and witness pair; they are not recomputed from the evaluated u. Unit-speed gamma is 1-Lipschitz in Euclidean output distance, so this F is globally 1-Lipschitz, equals A at x', and equals Q at x. Constant extension at path ends is included. History cancels as before. This proves sufficiency, not just pointwise feasibility. Measurable choices of such paths can define a wider context-conditioned class, but the per-context certificates do not establish one jointly learned finite-dimensional parameterization.

For this fork subset the local geometry has only a top corridor and left vertical passage. Set b=max(1,q_Y-1), h=A_Y-3, and dx=distance(A_X,[1,2-epsilon] intersect [q_X-1,q_X+1]). If A is in the vertical strip, minimum reachable Y is max(b,A_Y-r). For an upper anchor outside it, the shortest distance to the lower-passage entrance corner is c=sqrt(dx^2+h^2). If r<=c there is no strict entry below 3; otherwise the intrinsic minimum is max(b,3-(r-c)). These ordinate bounds are exact for the closed guarded geometry when the anchor is admissible. The endpoint-ball minimum is separately computed over the union of rectangles; it is a relaxation of the full-map requirement. They happen to agree on strict-entry counts in this sample, which is not an equivalence theorem.

A concrete counterexample uses q=(1.5,3.5), A=(2.4,3.1), Q=(1.9,2.7), x'=(0,0), x=(.68,0). A and Q are legal endpoints, and ||A-Q||=.640312<.68. But the shortest free path bends around (2-epsilon,3) and has length .728538>.68. No full 1-Lipschitz function on the square can realize this pair. [additional_counterexamples.json](additional_counterexamples.json) preserves the numbers.

Merely enforcing every point's anchor-ball distance also permits discontinuous maps. For example `F(t,u)=(1.5,3.5+t)` for t<.05 and `(1.5,3.5-t)` otherwise, on [-.1,.1]^2 with x'=0. Every output obeys its anchor-radius bound and lies in free geometry, but nearby actions on opposite sides of .05 have output/input distance ratio about 5000. Likewise nearest-point projection onto the nonconvex maze has ratio about 14142 for the recorded counterexample near the inner corner. An action-dependent rectangle switch can have the same discontinuity problem. Neither operation inherits the convex-projection proof automatically. See [counterexamples.json](counterexamples.json).

## 4. A smaller witness than a bent-path emitter

The geometry change need not replace convex projection. For an upper anchor outside the left strip with A_Y>3, extend the straight ray from A through the guarded inner corner K=(2-epsilon,3) a short distance into the vertical passage. Choose Q on that extension so the entire segment [A,Q] is free, within the step envelope, and ||Q-A||<r. When A is already in the vertical strip, take a short vertical segment instead. The analysis chooses a positive extension bounded by the available horizontal room, step limit and radius slack, with a factor-two margin. Q is an existence witness, not a supervised transition target.

Replace C for this fixed conditioning with the convex segment C*=[A,Q]. Set all eight existing matrix blocks to

    M*=(Q-A)(x-x')^T/r^2.

Then `F*(u)=projection_C*(A+M*(u-x'))` has precisely the same diagonal-consistency and all-action-pairs proof. At the witness action x it emits Q below Y=3, although the old box cannot do so for any coefficients. At x' it emits the original upper anchor. The execution action therefore affects passage choice continuously **inside one fixed convex set**, without an action-dependent selector. The matrix is still in the existing Frobenius-constrained family.

For the first qualifying saved example, seed 0 / reset 51000001 / t=1, A=(2.405497,3.352175), r=1.198496. Choose Q=(1.750315,2.783148). Its segment length is .867788 and ||M*||_F=.724064. This is legal under the strict unit step bound as well as the rounded implementation. The seed-1 example at the same reset/time has Q=(1.751298,2.782568), length .865473 and matrix norm .723749. Complete constants are in [convex_segment_witnesses.json](convex_segment_witnesses.json). No execution action is allowed to redefine C* during evaluation of its function.

There are 119/99 such excluded-response certificates across seeds under the existing rounded step envelope. Each is a full action-domain function, not a sampled action-grid inference. Grid and interpolation checks are implementation checks only; convexity, the segment construction and the norm calculation give the proof. Some permissible entries are extremely shallow (minimum constructed depth 1.19e-7), so this is not a certificate of effective multi-step turning. The intrinsic ordinate bound permits Y<=2.75 in 115/100 contexts before the literal-anchor precision qualification below.

## 5. Preserved numerical failure and exact-assumption qualification

The first segment check attempted literal real-arithmetic `abs(output-q)<=1` on every stored anchor and failed. Twelve anchors (4/8 by seed) already have X displacement 1.0000001192092896 from float32 addition. These are legal under the production float32-rounded box and below its previously declared 2e-6 audit tolerance. We did not change the anchors, enlarge L or silently relabel the check as passed. [failed_strict_step_check.json](failed_strict_step_check.json) preserves the failure and its consequence.

For those twelve literal anchors, **exact** diagonal preservation and an **exact** real-arithmetic unit step limit are incompatible even at x=x'. Thus the initial `all_maps_min_y` ordinate calculation is conditional on anchor admissibility, not an unconditional strict-physics certificate for those rows. The convex witnesses explicitly label their envelope and `strict_unit_step_feasible` status. Of 119/99 certificates under the implemented envelope, **115/91** satisfy the literal unit step bound exactly. The two displayed witnesses do. This small numerical qualification does not explain the large directional or box-exclusion effects, but it matters to an exact proof claim.

One different seed-1 case (reset 51000030, t=11) is obstructed by the actual prescribed action radius: A=q=(1.797633,3.977438), r=.965948, so even A_Y-r=3.011490>3. Geometry enlargement cannot provide a one-step entry there while retaining A and L=1. That is a real fixed-query restriction, distinct from floating-point representability and from the extra chosen box.
