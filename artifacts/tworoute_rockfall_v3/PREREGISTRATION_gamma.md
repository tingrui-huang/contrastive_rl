# Pre-registration: the gamma round (replaces the horizon-600 design)

## Why the horizon design was withdrawn

PREREGISTRATION_horizon600.md proposed separating "the objective does not
VALUE the long detour" (A) from "the learner cannot EXECUTE it" (B) by raising
the episode horizon 400 -> 600. That design cannot answer the question, and
the offline audit gate refused to run it:

    FAIL G3_SHAPES_DIMS  (dataset 401 rows/episode vs max_episode_steps 600)

Two reasons it was wrong, both found before any result was read:
1. Offline training draws from a FIXED dataset collected at horizon 400. The
   horizon does not change what the policy learns; it only sizes the
   in-training eval env. The audit caught exactly this inconsistency.
2. Even raising it at EVAL time cannot move shortcut_rate: the route is
   committed in the first few steps, long before the budget binds. A longer
   eval horizon changes whether an already-chosen detour COMPLETES -- i.e.
   the timeout rate -- not which route is chosen.

The withdrawn round produced no numbers. Nothing is being discarded here.

## The correct manipulation: gamma

gamma is the knob that sets what CRL's critic considers reachable: the
future-goal relabeling samples a goal at offset j-i with probability
proportional to gamma^(j-i) (crl/replay.py:197), so the effective horizon is
~1/(1-gamma). It changes the objective's VALUATION of the long route without
changing the environment, the dataset, the routes, or their execution
difficulty.

Discounted references for BR (shortcut 77 steps at 1.00 success, detour 236
steps at 0.876):

    gamma = 0.99   horizon ~100    shortcut 0.323  detour 0.083  -> SHORTCUT
    gamma = 0.999  horizon ~1000   shortcut 0.648  detour 0.692  -> DETOUR

So gamma = 0.999 REVERSES which route the objective prefers, while the ant,
the maze, the rocks, the dataset and the 400-step budget are byte-identical.

## Runs

BR only (TR's objective is near-indifferent at both gammas, so it cannot
discriminate). 2 = seed {0, 1}, 100k steps, horizon 400 (dataset-matched),
cfg.discount = 0.999, run dirs v3br_crl_s{0,1}_100k_g999.
Primary evaluation point: `final` @ 100k, n = 300, seed 909, canonical pose,
horizon 400 -- identical to the gamma = 0.99 comparison runs already measured
(BR pooled shortcut_rate 0.772 [0.736, 0.803]).

## Decision rule, fixed now (BR at gamma = 0.999, pooled, n = 600)

  shortcut_rate <= 0.55  -> MECHANISM A. Reversing the objective's preference
      reverses the behaviour, so the gamma=0.99 shortcut preference was driven
      by the objective's valuation, not by inability to execute the detour.
  shortcut_rate >= 0.70  -> MECHANISM B. The preference is insensitive to the
      objective's valuation, so it was an execution artefact. The 100k H1
      result must be reinterpreted.
  0.55 < shortcut_rate < 0.70 -> BOTH CONTRIBUTE; reported as partial, with no
      claim that either dominates.

## Companion measurement (no retraining, reported separately)

The EXISTING gamma=0.99 BR checkpoints re-evaluated at eval horizon 600.
This cannot move shortcut_rate (see above) and is not a test of A vs B; it
measures only the execution component: whether BR detour episodes that timed
out at 400 (57-68% of them) complete when given 600 steps. Prediction stated
in advance: shortcut_rate unchanged within noise, detour timeout substantially
lower. If shortcut_rate DOES move, my stated reason for withdrawing the
horizon design was wrong and that must be reported.

## Known limitation

2 seeds. The gamma=0.99 BR seeds were statistically distinguishable
(0.703 [0.649, 0.751] vs 0.840 [0.794, 0.877]), so this round establishes a
direction, not an effect size.
