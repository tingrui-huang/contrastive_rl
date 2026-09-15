# G1: sealed O/P pipeline with the absorbing-freeze ETT as the P transition

One bounded PointMaze experiment. It reruns the strictly offline
observational-vs-ETT positive-future pipeline of
`notes/pointmaze_offline_causal_integration.md` (driver
`ett/pointmaze_offline_causal_integration.py`, 09-15) and changes exactly one
thing: the transition that generates the P-side continuations. The eligible
gated-response ETT with three pessimistic offset updates is replaced by the
G0 prescreen arm `manski_absorbing` (`ett/absorbing_ett.py`), which has no
trainable offsets and no auxiliary critic. Everything downstream is reused
unchanged: the sealed sampler plan and split, the production sigmoid-NCE
critic (400 updates per arm), the actor objective (1,000 updates per arm,
BC 0.5), the batch composition, the seeds and random streams, the cache file
format, and the evaluation code that produces the 656-root endpoint gap.

## Motivation

G0 (`outputs/pointmaze_absorbing_prescreen_20260915_v1`) showed on the same
656 held-out first-fork roots that (i) the 09-15 final ETT is action-blind
(forced down versus right moves the next XY by 0.02), so its P positives could
not encode a route counterfactual, and (ii) the absorbing Manski freeze
channel makes model continuations prefer the safe detour by +4.28 discounted
return. G1 asks whether the unchanged critic learns that preference when
these continuations replace all production positives at matched anchors.

## The P transition

`F(s, x, x')` with the eligible diagonal, nominal and frozen actor:

1. alive XY: the eligible diagonal law queried at the executed action,
   including its stationary atom (assumes the hidden bits act only through
   freeze, not through alive motion);
2. absorbing persistence: a raw atom draw whose cell is in the absorbing
   support `S_abs` freezes the path; frozen paths keep XY, shift F4, zero
   reward;
3. Manski onset: if the landing bin of `x` differs from that of `x' ~ nominal`
   and the sampled landing cell is in `S_abs`, the path freezes from the next
   step, with the fatal incoming movement retained.

`S_abs`, the onset table and the persistence table are estimated on the 5,940
continuation-training episodes only, from `obs`/`act` only: landing cells
with at least 10 freeze onsets (moving transition followed by exactly
stationary transition) whose stationary run continues to the episode end in
at least half of the cases. The 660 held-out episodes, and therefore the 656
evaluation roots, never enter these tables. No hidden bits, death labels,
hazard labels, teacher modes or environment internals are read; the module
records the fields it reads.

The diagonal law is unchanged by construction: for `x = x'` the one-step
output of the backend is bit-identical to the eligible Kernel's, and this is
asserted on the 512 sealed validation diagonal rows with the same key.

## What is kept from the 09-15 driver

`prepare` (plan, split, config, provenance, seal), `train`, `evaluate`,
`load_prepared`, `Ledger`, `cache_audit`, `cache_pair_audit` and the cache
format are called as they are. The G1 driver `ett/pointmaze_absorbing_integration.py`
adds a `fit` phase that replaces `fit_ett`: it writes the same three caches
(`train_trajectory_pool.npz`, `validation_trajectory_pool.npz`,
`initial_validation_trajectory_pool.npz`), with extra freeze arrays appended,
and a zero `ett_final.npz`. The initial validation cache is still the eligible
model at zero offsets, so the paired initial-versus-final audit now compares
eligible against absorbing continuations under common random numbers.

The backend is switchable (`--backend`) among `eligible_response`
(delegates to the original `fit_ett`), `diagonal_motion`,
`diagonal_motion_absorbing` and `manski_absorbing`; the G1 target is
`manski_absorbing`.

## Baseline reproduction

Before any modified run, the unmodified 09-15 driver is rerun into a fresh
directory to confirm the committed endpoint numbers (initial -0.603,
O -0.305, P -0.088, P-O +0.217) are reproduced on this machine.

## Metrics

From the unchanged `evaluate`: initial/O/P endpoint down-minus-right
canonical-goal gaps and the paired P-O gap on the 656 roots; neighborhood
gaps; held-out NCE under observational and P goals; actor down/right
probabilities and paired changes; non-fork retention. From the caches:
task-region reach, absorbed fraction, absorbing-support entry, Manski and
atom onset fractions for P by stratum (ordinary/down/right), the same reach
and visible-freeze statistics for the matching recorded O continuations, and
the diagonal identity check.

## Success criteria (fixed before the run)

- Primary: the P critic's endpoint down-minus-right gap on the 656 roots is
  positive with a 95% bootstrap interval excluding zero, and the paired P-O
  gap is positive with an interval excluding zero.
- Secondary: right-anchored P continuations are substantially more absorbed
  and reach the task region substantially less than down-anchored ones.
- If the primary fails, the diagnosis proceeds in this order: cache
  integrity (identity, first action, padding, frozen consistency), positive
  consumption (lineage rows equal cache futures), critic response (held-out
  NCE on P goals, endpoint probes), and only then the transition itself.

Budgets: the sealed 480,000 model-output cap, zero environment calls, zero
native steps, 800 critic and 2,000 actor updates. One seed. Fixed final
iterates. No commit or push.
