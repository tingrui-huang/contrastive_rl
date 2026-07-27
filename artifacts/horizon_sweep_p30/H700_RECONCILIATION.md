# H=700 naive reconciliation: 0.706 (reconcile) vs 0.770 (horizon sweep)

Checkpoint: `naive_rockfall_v2_p30_s0_300k/final.pkl`, sha
`948019b46fb77e48cf3c069cb9506519c39f9061c6df99ad911daa4d25db6999`, step 300300.
Both harnesses use this exact file.

## Answers to the eight questions

1. **SHA/path** — identical file (above) in both harnesses.
2. **Seed / episode count** — reconcile natural: masks `draw_natural_masks(base+1,
   n=200)`, env-seed **base+10**; balanced K=100/pattern at env-seed=base.
   horizon: masks `draw_natural_masks(base+1, n=500)`, env-seed **base+0**;
   balanced K=60. Same base seed (20260726) + mask-RNG seed, so the first 200
   natural masks are shared; env-seeds and n differ.
3. **Sample-pooled** — reconcile 0.755 (n=200, cap-700); horizon 0.770 (n=500,
   cap-1000-thresholded-700). Recomputed here on n=500: env+0 0.750, env+10 0.758.
4. **Analytic natural-weighted** — reconcile **0.7058** (balanced K=100 × p30
   weights .2401/.2499/.2499/.2601). Reproduced EXACTLY here (0.7058). The horizon
   sweep never computed a naive analytic number, so the reported "0.706 vs 0.770"
   was **analytic-vs-pooled** — different aggregations.
5. **Per-mask (balanced K=100)** — n=100 each; success all_clear 0.80, left_only
   0.64, right_only 0.76, both_sides 0.63; worst-mask 0.63.
6. **Success / termination** — identical in both: success = max reward > 0 (ever
   within SUCCESS_DIST=0.5 of goal); both break on first success and on dead+5.
7. **Does H=1000 one-pass change H=700 semantics?** — the *definition* is identical
   (success@700 = first-success-step < 700 = reconcile's hit-within-700). BUT the
   env runs at a different CAP (1000 vs 700), which changes cross-episode state
   (see 8). So the horizon harness's H=700 number is a cap-1000 evaluation
   thresholded at 700, not a true cap-700 evaluation.
8. **Do dead episodes become success?** — NO. `dead_counted_as_success = 0` in
   both blocks (after burial reward is 0 forever; both break at dead+5).

## Root cause of the residual gap: cross-episode mujoco warmstart

Un-interleaved passes (each a full single-env sequential run) still mismatch on
16/500 (env+0) and 20/500 (env+10) episodes at H=700. These are NOT borderline
flips — e.g. episode 53: cap-700 run never succeeds, cap-1000 run succeeds at step
302 (same initial state). The env is bit-identical-deterministic given the same
cap (verified: 0/30 divergence for cap-700 vs cap-700), so the divergence comes
from `reset_model` NOT zeroing the solver `qacc_warmstart` — so episode i inherits
solver state from episode i-1, whose length differs between the 700- and 1000-cap
passes. Episode outcomes are therefore not independent; the aggregate effect is
~±0.02 on any pooled number.

## Resolution

0.706 vs 0.770 decomposes into three effects, in order of size:
1. **Aggregation** (dominant): analytic-weighted 0.706 vs sample-pooled ~0.755-0.770.
2. **Sampling**: env-seed +10 vs +0 and n=200 vs 500.
3. **Warmstart carry-over** (~0.02): the horizon harness runs the env at cap-1000,
   so long timeout episodes bleed solver state into later episodes; the reconcile
   harness runs at cap-700.

Authoritative H=700 naive numbers (reconcile, cap-700, analytic primary):
**natural (analytic) 0.706**, pooled 0.755, balanced-macro 0.7075, worst-mask 0.63.

Implications:
- The horizon sweep's WITHIN-harness delta (+0.040 H700->H800) is valid: all
  thresholds share the same cap-1000 warmstart regime. The horizon sweep's
  ABSOLUTE values are not authoritative (cap-1000 + pooled).
- Authoritative absolutes should evaluate each horizon at its OWN cap (reconcile
  `--horizon H`). The formal H=800 experiment does this (train fresh @H=800,
  reconcile @cap-800).
- RECOMMENDED benchmark-quality fix (separate decision): zero `qacc_warmstart` on
  reset in the eval rollout so episodes are independent and harnesses agree
  exactly. Not applied here to avoid changing committed-number semantics mid-run.
