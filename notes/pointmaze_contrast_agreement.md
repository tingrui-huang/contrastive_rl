# Read-only critic/MC contrast agreement audit

This protocol is fixed before outcome analysis. It uses only arrays already
saved by `continuation_precision_s01_v1` and `response_search_s01_v1`. It will
not instantiate a transition kernel or environment, load a checkpoint, run
inference, generate a transition or rollout, update a parameter, or modify any
existing artifact. The computed-new-transition count for this audit is fixed at
zero.

## Inputs and reconstruction contract

For continuation precision, analyze `s0_k2_u24 - s0_k2_u0` and
`s1_k0_u24 - s1_k0_u0`. Load each pair's `queries.npz` and all nine
`batch*.npz` files. Inspect and use the saved `candidate_critic`,
`reference_critic`, `candidate_mc`, and `reference_mc` arrays produced by the
decoding, horizon masking, and visitation weighting in
`ett/report_continuation_precision.py`. Average all 8 successor groups x 2
next-action draws inside each query before any statistic. Use `queries.npz`
`mass` as the design weight and its `pre_entry` flag for stratification.

Before computing stratified outcomes, reconstruct each pair's unstratified
weighted critic and MC contrast means and require agreement, to floating-point
tolerance, with the corresponding published values in
`continuation_precision_s01_v1/results.json`. Abort the audit on a mismatch.

A continuation query is sequence-changed when at least one of its 16 paired
draws differs between candidate and reference in either the saved first reward
or any saved continuation reward. The all-query sign agreement treats exact
zero in the usual three-way `sign` comparison; the changed-query result removes
the zero-information queries explicitly.

For response search, use the 18 saved `checks/*/matched_mc.json` and
`matched_mc.npz` records (three audit updates for each of six starts). Recompute
the critic and MC point deltas from the arrays and require equality with every
JSON point estimate. For prefix entry, map each saved `query_path` and `query_t`
back to the companion `visitation.npz` trajectory, assert state/action/root/time
identity, recompute its saved rewards from trajectory states with the existing
fixed-goal `task_reward`, and define `pre_entry` as no reward before the query
state. This is saved-state analysis only. If any identity or reward assertion
fails, mark response-search pre-entry analysis unavailable rather than infer a
flag.

## Statistics

For each continuation pair and for both `pre_entry` and `entered`, report query
count, unweighted strict sign-agreement rate, design-weighted mean critic and MC
contrasts, design-weighted Spearman correlation, and design-weighted Kendall
tau-b. Weighted Spearman is the design-weighted Pearson correlation of weighted
midranks: a tie group's rank is cumulative weight below it plus half the tie
group's weight. Weighted Kendall tau-b gives pair `(i,j)` weight `w_i*w_j` and
uses the standard separate tie-adjusted denominators. These definitions reduce
to the usual midrank Spearman and Kendall tau-b under equal weights.

Use a paired cluster bootstrap over the 36 root labels: 2,000 resamples of 36
roots with replacement from one RNG seeded `190000000`, consumed in pair order
`s0_k2`, `s1_k0`. Apply each root multiplicity to both members of every paired
contrast and to its design weight. Report percentile 95% intervals for sign
agreement, weighted Spearman, and weighted Kendall. Use the same root-bootstrap
draws to report a percentile interval for sign agreement among sequence-changed
queries. Means are descriptive point estimates; no multiplicity correction is
made.

For response search, report the 18 critic/MC deltas as point estimates with
their per-check query count, the strict sign-agreement count, ordinary Spearman
correlation across the 18 paired points, and its paired 2,000-replicate
percentile bootstrap interval using seed `190000001`. Repeat those aggregate
statistics after per-check restriction to reconstructed pre-entry queries,
provided every check has at least one such query and reconstruction passes.

Exact-zero signs are a separate category. If either contrast vector in a
reported stratum has zero variance, or a bootstrap resample is degenerate, its
rank correlation is undefined. A zero-variance point stratum is flagged and is
not described as agreement; degenerate bootstrap replicates are counted and
excluded from percentile endpoints. Every interval is descriptive. Quantities
without an interval are explicitly labeled point estimates.

## Interpretation limits

Two candidate pairs and 18 comparisons cannot establish a general property of
the critic. This audit can only distinguish "systematically reversed in the
leverage region" from "unordered noise", and it is underpowered for the latter.
It has no authority to conclude that the critic is fine. The summary will end
with the observed pattern and its precision only and will not recommend a
follow-up experiment.
