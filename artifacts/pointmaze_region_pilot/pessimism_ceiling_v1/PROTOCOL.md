# Protocol: ETT pessimism headroom ceiling diagnostic

## Status and estimand

This protocol is sealed before model rollouts. It is a pure model diagnostic:
zero learning, gradients, actor updates, native simulation, or checkpoint
writes. Answer-handed A2--A4 are constructive lower bounds on achievable
pessimism under their stated rules, not certified ceilings or valid bounds.

The repository contains no literal 36-root set collected at t=40. The fixed
t=40/H=10 artifact contains 64 roots, while the exact group-balanced 36-root
held-out set used with the learned responses at commit 817e429 has source times
1, 3, 6, and 10. To avoid inventing a 36-of-64 subset, setting 1 uses those
exact 36 held-out roots and imposes a fresh fixed H=10. Source-time metadata are
retained. Consequently its A0 need not reproduce the historical ~0.60 mean.

## Frozen arms and streams

All arms share the loaded diagonal law, nominal policy, actor, goal, geometry,
normalization, anchor draw, coordinate-step rule, boundaries, and wall endpoint
processing. Each step draws x' from the nominal policy and x from the actor.
Transition randomness starts at 192000000 and policy randomness at 193000000;
the same per-setting streams are replayed across arms. The 100000 arm stride is
reserved in configuration but deliberately not added to replay keys, because
adding it would destroy the required cross-arm pairing.

- A0: the diagonal anchor (zero response).
- A1-s0/A1-s1: only the 32 response coordinates of the frozen final critic ETT
  kernels associated with 817e429; their 16 diagonal-head offsets are ignored.
- A2: anchor + ||x-x'|| times the unit vector from goal to anchor, projected by
  the existing selected convex box. At anchor=goal, +x is the sealed tie break.
- A3: the selected convex-box corner farthest from goal. Its pre-projection
  proposal is an outward ray, so the literal active-box-clip indicator remains
  meaningful.
- A4: the farthest point from goal among all free-rectangle intersections with
  the coordinate-step square, followed only by the existing coordinate,
  boundary, and wall endpoint postprocessing.

Every arm has an exact x=x' branch returning the identical sampled anchor.

## Evaluation, intervals, and mechanisms

Setting 1 has 36 roots x 64 rollouts x H=10. Setting 2 has 128 independent
START rollouts x H=50. Returns are 0.05 sum_t .95^t r_t. Report median, p10,
p25, p75, p90, and mean. Paired differences use 2,000 bootstrap replicates at
seed 191000000. Setting 1 resamples roots within its three 12-root strata and
also reports the conditional within-root Monte Carlo interval. Setting 2
resamples paired rollouts. Paired median differences receive bootstrap
intervals and are primary if fewer than 20 nonzero same-direction rollouts
support a mean difference.

The requested signed relative change is C=(A-A0)/A0 (negative is pessimistic).
For unambiguous thresholding, reduction magnitude is R=-C=(A0-A)/A0. The same
convention is applied to p25; a zero A0 p25 makes that threshold unavailable.
Top-5% signed and absolute contribution shares, realized delta-norm summaries,
literal active box-clip fractions, reward-sequence changes versus A0, and
pre-entry/entered splits are reported. Entry at a decision means a positive
reward occurred at an earlier step in that same trajectory.

## Acceptance and charge

Identity probes: 128 contexts x 8 draws x 6 arms. Lipschitz grid: 128 contexts
x 16 paired action comparisons x 2 outputs x 6 arms. Together with both
settings, the exact charge is 207,360, below 600,000. A2 must have samplewise
excess <=2e-6; A3/A4 are expected to fail because of their diagonal branch and
the failure is recorded. Every rollout must be finite, have valid anchors,
exact F4 history shift, legal endpoints, and exact recomputed rewards.

A5 was sealed as five candidates (four selected-box corners plus anchor),
depth 3, eight continuations per candidate at every decision. It would require
3,532,800 lookahead outputs plus 29,440 realized outputs and 5,120 probes, for
3,774,720 total including A0--A4. This exceeds 1.5M, so A5 is skipped before
outcomes.

## Decision rules (ordered)

The target decision uses setting 1; setting 2 is reported prominently as a
reset-start transport check. A1 means the more pessimistic of its two seeds.

1. If A3 and A4 each achieve neither 10% mean nor 10% p25 reduction, conclude
   reward geometry/postprocessing is binding, with a rare-move caveat.
2. Else if A2 mean R>=10% and both learned A1 R<3%, conclude optimizer
   bottleneck rather than response-family limitation.
3. Else if A3 exceeds A2 mean R by >=5 percentage points and the direct paired
   95% root and conditional-MC intervals for A3-A2 are below zero, conclude
   L=1 binds.
4. A5 would analogously exceed A3 by >=5 points with paired intervals below
   zero, but is predeclared skipped.
5. Otherwise conclude measured ceiling/no separation. A material setting
   disagreement (opposite mean direction or >=5-point R gap) is a mandatory
   qualification, not a reason to rerun.
