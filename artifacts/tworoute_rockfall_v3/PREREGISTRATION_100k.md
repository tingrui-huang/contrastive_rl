# Pre-registration: the V3 100k short-training round

Written and committed to BEFORE the runs were launched, because the 300k round
exposed a free parameter we had not fixed: the evaluation step. Route
preference drifts throughout training (V3 300k, matched steps, both seeds
pooled, n=600/cell: BR shortcut_rate 0.897 at 80k -> 0.668 at 150k -> 0.230 at
300k; TR 0.467 -> 0.410 -> 0.723), so "which checkpoint" is an analyst degree
of freedom big enough to reverse the headline. This round removes it.

## Protocol (fixed in advance)

- Runs: 4 = {tr, br} x seed {0, 1}. Identical to the 300k round except
  --steps 100000. Same datasets, same recipe (bc 0.05, twin-min, alpha 0,
  batch 1024, repr 16, hidden (1024,1024)), same canonical-pose eval.
- PRIMARY EVALUATION POINT: the `final` checkpoint at exactly 100k steps.
  Not `best` (selected as the max of ~10 noisy n=30 in-training evals, an
  upward-biased draw: the 300k BR s0 best reported 0.867 in training and
  0.617 at n=300), and not any milestone chosen after seeing results.
- n = 300 evaluation episodes per checkpoint, seed 909, canonical pose.
- 80k was the strongest contrast in the 300k round; 100k is deliberately NOT
  80k, so this is an out-of-sample test of the effect rather than a re-report
  of the point that produced it.

## Primary hypothesis (directional, fixed in advance)

H1: shortcut_rate(BR) > shortcut_rate(TR) at 100k, pooled over both seeds.

Rationale: the two variants share identical sparse reference numbers
(always-shortcut 0.70 / always-detour 0.96 / oracle 0.988) and identical
confounding (measured gap 0.272 tr / 0.273 br). They differ only in the
incentive that CRL's own discounted-reachability objective (gamma = 0.99,
effective horizon ~100 steps) assigns to the shortcut:
  tr: shortcut 0.146 vs detour 0.185  -> objective favours the DETOUR
  br: shortcut 0.323 vs detour 0.100  -> objective favours the SHORTCUT

DECISION RULE, fixed now: H1 is supported iff the two pooled Wilson 95%
intervals (n = 600 per variant) do not overlap and BR is the higher one.
Any other outcome -- overlap, or reversal -- is reported as H1 not supported.

## Secondary, reported regardless of H1

- P(success | rockfall_active) per variant: the 300k round measured 0.000 for
  BR at its early/best checkpoints (death is certain once the shortcut is
  taken under an armed latent).
- death rate vs p_active = 0.30.
- timeout rate, as the degradation indicator that motivated this round
  (BR timeout rose 0.060 -> 0.357 for s0 and 0.197 -> 0.477 for s1 across
  80k -> 300k).
- gamma = 0.99 discounted return, the objective the policy actually optimises.

## What this round CANNOT settle

Two seeds per variant. The 300k round showed seed-level route-preference
differences as large as the variant effect at some steps (tr s0 0.580 vs
tr s1 0.317 at their best checkpoints), so a 2-seed contrast can establish a
direction, not an effect size. A seed sweep remains outstanding.
