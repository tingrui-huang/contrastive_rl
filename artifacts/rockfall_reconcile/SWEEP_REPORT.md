# p_active sweep — final comparison (v2.1 local-detour, 300k naive, authoritative eval)

Controlled sweep: only `p_active` varies. Severity 0.80/0.15/0.05, mixture 90/0/10,
4 sites, triggers, walker, center dynamics, learner obs, CRL recipe, dataset size,
300k horizon all fixed. p=0.20 is the untouched reference. Naive = final.pkl.

Eval: reconcile harness (K=100/pattern balanced + N=200 natural, paired forced masks,
Wilson CIs) for success aggregations; `diagnose_naive_rockfall.py --v2` for behaviour.
Provenance: naive p30 sha 754a8063 / p50 sha a23ac6fd, git 4867763, GPU, seed 20260726.

## Primary metric — natural-distribution success (analytic-weighted)

| policy | p=0.20 | p=0.30 | p=0.50 |
|---|---|---|---|
| sighted teacher (mask-reading) | 0.885 | 0.867 | 0.843 |
| fixed-center (mask-independent robust) | 0.785 | 0.796 | 0.806 |
| **naive CRL (final)** | **0.796** | **0.706** | **0.668** |
| blind-side | 0.653 | 0.530 | 0.309 |
| naive natural — pooled cross-check | 0.785 | 0.755 | 0.745 |

## The two gaps move in opposite directions (the decisive finding)

| gap | p=0.20 | p=0.30 | p=0.50 | trend |
|---|---|---|---|---|
| **confounder influence** = teacher − naive | +0.088 | +0.161 | +0.175 | widens |
| **value of the mask** = teacher − center | +0.099 | +0.071 | +0.037 | **collapses** |

- The naive→teacher gap widens with density → the confounder is more influential on the
  learned baseline as `p_active` rises. Competing "universal policy" hypothesis REFUTED:
  naive natural success falls (0.796→0.706→0.668), it does not close the gap.
- BUT the value of reading the mask (teacher over the best mask-independent policy, center)
  collapses to +0.037 at p=0.50. When both sides are almost always blocked (both_sides = 56%
  of the natural distribution at p=0.50), "always go center" is near-optimal and needs NO mask.
  The confounder loses causal interest exactly when the naive gap looks biggest.

## Balanced per-pattern success (diagnostic; n=100/pattern)

| | all_clear | left_only | right_only | both_sides | worst-mask |
|---|---|---|---|---|---|
| naive p20 | 0.98 | 0.80 | 0.60 | 0.56 | both 0.56 |
| naive p30 | 0.80 | 0.64 | 0.76 | 0.63 | both 0.63 |
| naive p50 | 0.94 | 0.93 | 0.55 | 0.59 | right 0.55 |
| teacher p20/p30/p50 | 0.92 | .86/.88/.87 | .86/.84/.83 | .86/.83/.83 | ~0.83–0.86 |
| center (all p, invariant) | 0.75 | 0.84 | 0.78 | 0.81 | all_clear 0.75 |
| blind p20/p30/p50 | 0.92 | .60/.56/.55 | .51/.52/.49 | .16/.15/.10 | both ~0.10–0.16 |

Worst-mask headroom (teacher − naive): p20 0.30, p30 0.20, p50 0.28 — meaningful at all.
Center is provably mask- and density-invariant (byte-identical per-pattern) = the viable
mask-independent robust policy, ~0.79–0.81 at every density.

## Naive behaviour (diagnosis)

| | p=0.20 | p=0.30 | p=0.50 |
|---|---|---|---|
| center usage | 0.325 | 0.375 | 0.595 |
| left / right lane | 0.56 / 0.115 | 0.595 / 0.03 | 0.185 / 0.22 |
| hazard exposure | 0.515 | 0.295 | 0.355 |
| drop rate | 0.13 | 0.105 | 0.21 |
| trigger-avoidance (gaming <0.30) | 0.072 | 0.109 | 0.157 |
| leakage (paired mask-flip) | 30/30 clean | 30/30 clean | 30/30 clean |

Naive adapts unconditionally: habitual side lane at low density → center-routing at p=0.50
(0.595). Never the teacher's selective per-site detour (leakage clean at all p). But even at
p=0.50 it under-uses center relative to the pure-center controller and gets hit on the
residual side-lane episodes, so it stays below both teacher (0.843) and center (0.806).

## Adjudication vs. selection criteria

| criterion | p=0.20 | p=0.30 | p=0.50 |
|---|---|---|---|
| high sighted-teacher performance | 0.885 ✓ | 0.867 ✓ | 0.843 ✓ |
| viable mask-independent robust policy (center) | 0.785 ✓ | 0.796 ✓ | 0.806 ✓ |
| clear natural-distribution gap (teacher−naive) | 0.088 (thin) | 0.161 ✓ | 0.175 ✓ |
| value of the mask (teacher−center) | 0.099 ✓ | 0.071 ✓ | 0.037 ✗ weak |
| worst-mask headroom | 0.30 ✓ | 0.20 ✓ | 0.28 ✓ |
| naive does NOT solve via unconditional policy | n/a | ✓ (falls) | ✓ (falls, but drifts to center) |
| no leakage / no trigger-gaming | ✓ / ✓ | ✓ / ✓ | ✓ / 0.157 (ok) |
| pattern balance | all_clear 41% (skewed easy) | 24/25/25/26 (balanced) | both_sides 56% (skewed hard) |

## Recommendation: p_active = 0.30

p=0.30 is the only condition that satisfies every criterion simultaneously. It roughly
DOUBLES the confounder's influence on the natural metric vs the reference (0.088→0.161)
WHILE preserving a real causal signal (mask still worth +0.071 over always-center) and a
balanced pattern mix (no pattern dominates). p=0.50 gives the largest naive gap but for the
wrong reason: the mask stops mattering (teacher−center collapses to +0.037), the env
degenerates toward "both sides always blocked → just go center," and the naive baseline
looks worse mainly because it routes center messily rather than because the confounder is
hard. Per the explicit caution, we do not pick p=0.50 on its lower naive success alone.
p=0.20 keeps the strongest mask-value signal but the natural-distribution gap is too thin
(all_clear dilutes it) to headline a confounder benchmark.
