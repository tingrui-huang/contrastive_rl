# Corrected-reset (resetfix_v1) formal p30 benchmark: H=700 vs H=800

All numbers are natural SAMPLE-POOLED success (successes / 1000) under the
CORRECTED (episode-independent) reset, each policy at its own NATIVE cap, on a
paired N=1000 natural bank (same masks, only horizon differs). Primary metric =
pooled success; per-mask / macro are diagnostics only. Reset fix + all datasets
audited (V1-V10, G1-G8) and the reset correctness gate (A-E) passes at cap-700
AND cap-800. Fresh 300k naive runs: corrected H700 (commit b9a17d0, dataset
ff5a8136), corrected H800 (commit 14335a6, dataset 08bdc44b), both GPU.

## PART 6 -- authoritative pooled tables

### Corrected H=700 (cap-700), bank 343f5a11
| Policy | pooled | / 1000 | med steps | center/L/R | exposure | drop | macro | worst | leak |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Sighted teacher | 0.868 | 868 | 223 | .07/.87/.06 | 0.91 | 0.07 | 0.870 | 0.79 | - |
| Fixed center | 0.800 | 800 | 655 | .84/.16/.00 | 0.00 | 0.00 | 0.800 | 0.79 | - |
| Blind side | 0.500 | 500 | 186 | .00/.74/.26 | 1.00 | 0.51 | 0.514 | 0.11 | - |
| Fresh naive final | 0.615 | 615 | 310 | .32/.35/.33 | 0.66 | 0.30 | 0.625 | 0.31 | none |
| Fresh naive best | 0.704 | 704 | 545 | .59/.25/.16 | 0.35 | 0.15 | 0.708 | 0.58 | none |
| Legacy H700 naive, corrected eval | 0.744 | 744 | 600 | .37/.59/.03 | 0.31 | 0.13 | 0.747 | 0.66 | none |

### Corrected H=800 (cap-800), bank e94abc57
| Policy | pooled | / 1000 | med steps | center/L/R | exposure | drop | macro | worst | leak |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Sighted teacher | 0.870 | 870 | 223 | .07/.87/.06 | 0.91 | 0.07 | 0.872 | 0.79 | - |
| Fixed center | 0.899 | 899 | 659 | .83/.17/.00 | 0.00 | 0.00 | 0.899 | 0.89 | - |
| Blind side | 0.503 | 503 | 186 | .00/.74/.26 | 1.00 | 0.51 | 0.517 | 0.11 | - |
| Fresh naive final | 0.492 | 492 | 202 | .07/.45/.48 | 0.97 | 0.48 | 0.507 | 0.11 | none |
| Fresh naive best | 0.542 | 542 | 215 | .13/.45/.42 | 0.90 | 0.43 | 0.555 | 0.19 | none |
| Legacy H800 naive, corrected eval | 0.655 | 655 | 323 | .24/.49/.26 | 0.68 | 0.30 | 0.665 | 0.41 | none |

Center has ZERO rockfall interaction (exposure/trigger/drop/impact = 0) at both
horizons. Leakage clean (paired mask-flip) for every naive checkpoint.

## PART 7 -- legacy vs corrected reset (the reset error is unbiased)

PART 2 (N=500, cap-700, SAME model & bank, legacy vs corrected reset only):
| policy | legacy | corrected | disagree/500 | net |
|---|---:|---:|---:|---:|
| teacher | 0.890 | 0.890 | 18 (9/9) | 0 |
| center | 0.798 | 0.800 | 47 (23/24) | +0.002 |
| blind | 0.556 | 0.562 | 17 (7/10) | +0.006 |
| naive | 0.750 | 0.762 | 40 (17/23) | +0.012 |

The warmstart bug is real per-episode contamination (3.6-9.4% of episodes flip;
center most, being timeout-boundary sensitive) but statistically UNBIASED: pooled
success moves <=0.012 and flips cancel. So the legacy pooled numbers were
approximately correct; the fix buys reproducibility/independence, not a different
benchmark verdict.

## PART 8 -- corrected H=700 vs H=800 comparison

| Metric | Corrected H700 | Corrected H800 |
|---|---:|---:|
| Teacher pooled | 0.868 | 0.870 |
| Center pooled | 0.800 | 0.899 |
| Blind pooled | 0.500 | 0.503 |
| Fresh naive final pooled | 0.615 | 0.492 |
| Fresh naive best pooled | 0.704 | 0.542 |
| Naive balanced macro (final) | 0.625 | 0.507 |
| Naive worst mask (final) | 0.306 | 0.106 |
| Naive median success steps (final) | 310 | 202 |
| Naive center usage (final) | 0.32 | 0.07 |
| Naive hazard exposure (final) | 0.66 | 0.97 |
| Naive drop rate (final) | 0.30 | 0.48 |
| Teacher - naive (final) | +0.253 | +0.378 |
| Teacher - center | +0.068 | -0.029 |

### Answers

1. **Does the center/teacher ordering still invert at H=800 after the reset fix?
   YES.** teacher - center goes +0.068 (H700) -> -0.029 (H800); the mask-blind
   center (0.899) overtakes the sighted teacher (0.870). This is on the POOLED
   metric under the corrected reset -- so it is neither a reset artifact nor an
   analytic-weighting artifact. The inversion is a real controller-level property.

2. **Does H=800 change the learned naive route preference? YES.** The fresh H=800
   naive nearly abandons the center (usage 0.32 -> 0.07) for the fast side lanes
   (exposure 0.66 -> 0.97, drop 0.30 -> 0.48, median steps 310 -> 202). Same
   direction as the legacy H=800 finding; it does NOT lean on the slow center more.

3. **Does the teacher-naive gap remain? YES**, and it grows (final +0.253 ->
   +0.378; best +0.164 -> +0.328). Magnitude is noisy (single seed) but the gap
   clearly persists.

4. **Episodes rescued during 701-800:** center +99 (800 -> 899, same controller,
   paired bank). Naive +3 (negligible) -- the naive is fast/side-lane and NOT
   timeout-limited; its failures are rockfall deaths/drops, which more horizon
   cannot fix. Center is the only policy the extra horizon materially rescues.

5. **Are the changes larger than the legacy reset effect? YES, for center.**
   Center +0.099 dwarfs the <=0.012 reset effect (~8x). Teacher +0.002 and blind
   +0.003 are within the reset noise (flat). So the horizon-driven center rescue
   and the inversion are real, not reset noise.

6. **Which differences may be single-seed noise?** The naive pooled magnitudes
   (fresh H700 0.615/0.704 vs H800 0.492/0.542) are single-seed each, and the
   final-vs-best spread (~0.09-0.12) shows high checkpoint/training variance --
   treat the absolute naive levels and the H700->H800 naive drop as seed-noise-
   dominated. The scripted anchors (teacher/center/blind) are training-free, so
   their differences -- above all center +0.099 and the inversion -- are real.

## Verdict

The reset correctness fix does NOT change the benchmark conclusion. Under the
corrected, reproducible, episode-independent reset and the pooled primary metric,
the H=800 center/teacher inversion is CONFIRMED (center 0.899 > teacher 0.870).
**H=700 remains the recommended horizon** -- it preserves the intended ordering
teacher 0.868 > center 0.800 > naive > blind 0.500, with the mask worth +0.068
over always-center. Everything the earlier legacy H=800 experiment found survives
the reset fix; the fix's value is reproducibility, and it rules out both the
warmstart bug and analytic-weighting as explanations for the inversion.
