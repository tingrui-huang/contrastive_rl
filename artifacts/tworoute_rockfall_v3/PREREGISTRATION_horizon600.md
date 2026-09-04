# Pre-registration: the horizon-600 disambiguation round

Committed BEFORE the runs, because this round exists to test a threat to the
100k conclusion that was found only after that conclusion was drawn.

## The threat

The 100k round supported H1 (BR shortcut_rate 0.772 [0.736, 0.803] vs TR
0.403 [0.365, 0.443], disjoint). The stated mechanism was that CRL's
discounted-reachability objective (gamma = 0.99, effective horizon ~100
steps) cannot reach the 236-step BR detour, so the actor is not pulled toward
it. But BR learners that DID take the detour timed out 57-68% of the time
(s0: 28 of 41; s1: 17 of 30), so a competing explanation is available:

  A (claimed):   the objective does not VALUE the long detour.
  B (confound):  the learner cannot EXECUTE the long detour.

These are not separated by the existing data, and they are not independent --
both follow from the detour's length. The detour is completable in principle
(teacher 0.945 at 236 steps; measured do(detour) = 0.876), so B is about the
learner, not the environment. Evidence that B is not the whole story: in TR,
where the detour is 166 steps, the same algorithm completes it at 0.83/0.80.

## Manipulation

Raise ONLY the episode horizon, 400 -> 600 (--rockfall-max-steps 600), for
both variants. This lowers execution difficulty while leaving the incentive
structure untouched: the gamma = 0.99 discounted references (BR shortcut
0.323 vs detour 0.100; TR 0.146 vs 0.185) are horizon-independent. The sparse
references DO move (BR do(detour) should rise from 0.876 toward ~0.95), which
is the point.

TR is run as the control: without it we could not exclude "a longer horizon
raises shortcut_rate everywhere".

## Runs

4 = {tr, br} x seed {0, 1}, 100k steps, horizon 600 in BOTH training and
evaluation. Primary evaluation point: `final` @ 100k, n = 300, seed 909,
canonical pose. Same as the 100k round in every other respect.

## Decision rule, fixed now (BR, pooled over both seeds, n = 600)

  shortcut_rate >= 0.65  -> MECHANISM A. The preference survives when the
      detour is easy to execute, so it is driven by the objective's valuation.
      H1's stated mechanism stands.
  shortcut_rate <= 0.50  -> MECHANISM B. The preference was substantially an
      execution artefact; H1's mechanism must be rewritten and the 100k
      result reinterpreted.
  0.50 < shortcut_rate < 0.65 -> BOTH CONTRIBUTE. Reported as partial
      confounding, with no claim that either mechanism dominates.

Reported regardless: BR detour timeout rate (the quantity this round is meant
to move -- if it does not fall substantially from 0.57-0.68, the manipulation
failed and the round is uninformative), P(success | active), death rate,
per-seed spread, and the TR control's shortcut_rate.

## Known limitation, stated in advance

Still 2 seeds per cell. The 100k round found the two BR seeds statistically
distinguishable (0.703 [0.649, 0.751] vs 0.840 [0.794, 0.877], disjoint), so
seed variance is real and this round cannot estimate it either.
