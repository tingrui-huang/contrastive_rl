# Read-only critic/MC contrast agreement audit

The critic is not systematically sign-reversed in either tested pre-entry pair: both have above-half sign agreement and positive rank association with root-bootstrap intervals excluding the reversal boundaries. The s0_k2 opposite weighted means therefore reflect contrast magnitudes, not pervasive per-query order reversal.

All intervals below are descriptive percentile 95% intervals. Means and per-comparison deltas are explicitly point estimates.

## Continuation precision

### s0_k2_u24 minus s0_k2_u0

The unstratified acceptance gate passed at n=2304 queries: critic -0.0224 and MC +0.0171 (point estimates), reproducing the published means within absolute and relative tolerance 1e-12.

- `pre_entry`: n=1521; sign agreement 56.0%, 95% CI [53.5%, 58.3%]; weighted Spearman +0.2891, 95% CI [+0.1893, +0.3929]; weighted Kendall tau-b +0.2190, 95% CI [+0.1448, +0.2980]; weighted critic mean -0.1195 and weighted MC mean +0.0715 (point estimates). Changed-sequence agreement uses n=1243 changed queries after excluding n=278 unchanged queries: 68.5%, 95% CI [65.8%, 71.2%].
- `entered`: n=783; sign agreement 3.3%, 95% CI [2.1%, 4.7%]; weighted Spearman -0.0058, 95% CI [-0.0837, +0.0717]; weighted Kendall tau-b -0.0048, 95% CI [-0.0686, +0.0587]; weighted critic mean +0.0027 and weighted MC mean +0.0030 (point estimates). Changed-sequence agreement uses n=15 changed queries after excluding n=768 unchanged queries: 66.7%, 95% CI [44.4%, 88.9%].

### s1_k0_u24 minus s1_k0_u0

The unstratified acceptance gate passed at n=2304 queries: critic -0.0466 and MC -0.0216 (point estimates), reproducing the published means within absolute and relative tolerance 1e-12.

- `pre_entry`: n=1505; sign agreement 57.5%, 95% CI [52.8%, 62.0%]; weighted Spearman +0.5424, 95% CI [+0.4572, +0.6207]; weighted Kendall tau-b +0.4177, 95% CI [+0.3499, +0.4815]; weighted critic mean -0.1462 and weighted MC mean -0.0566 (point estimates). Changed-sequence agreement uses n=1198 changed queries after excluding n=307 unchanged queries: 72.2%, 95% CI [68.9%, 75.6%].
- `entered`: n=799; sign agreement 2.6%, 95% CI [1.7%, 3.6%]; weighted Spearman +0.0892, 95% CI [-0.0296, +0.1777]; weighted Kendall tau-b +0.0729, 95% CI [-0.0242, +0.1455]; weighted critic mean -0.0200 and weighted MC mean -0.0123 (point estimates). Changed-sequence agreement uses n=12 changed queries after excluding n=787 unchanged queries: 66.7%, 95% CI [40.0%, 91.7%].

## Response search

Across n=18 saved comparisons, strict sign agreement is 9/18 (50.0%; point estimate). Spearman is +0.1228, 95% CI [-0.4211, +0.6089].

Exact prefix-entry status was reconstructed for all n=18 comparisons from saved trajectory states only. The restricted per-check deltas use n=103 pre-entry queries in total (point count). One check has n=0 pre-entry queries, so the requested all-18 aggregate is unavailable; across the n=17 estimable comparisons, sign agreement is 8/17 (47.1%; point estimate), and Spearman is +0.0858, 95% CI [-0.4772, +0.6334].

The 18 all-query and pre-entry-restricted deltas are:

- `s0_k0_u1`: all-query n=32, critic +0.0077, MC +0.0010 (point estimates); pre-entry n=6, critic +0.0334, MC +0.0053 (point estimates).
- `s0_k0_u12`: all-query n=32, critic +0.0016, MC -0.0256 (point estimates); pre-entry n=9, critic +0.0058, MC -0.0910 (point estimates).
- `s0_k0_u24`: all-query n=32, critic -0.0008, MC +0.0162 (point estimates); pre-entry n=7, critic -0.0027, MC +0.0740 (point estimates).
- `s0_k1_u1`: all-query n=32, critic -0.0069, MC -0.0244 (point estimates); pre-entry n=0, deltas unavailable (point count).
- `s0_k1_u12`: all-query n=32, critic -0.0011, MC +0.0037 (point estimates); pre-entry n=5, critic -0.0055, MC +0.0236 (point estimates).
- `s0_k1_u24`: all-query n=32, critic -0.0009, MC +0.0014 (point estimates); pre-entry n=5, critic -0.0061, MC +0.0090 (point estimates).
- `s0_k2_u1`: all-query n=32, critic +0.0079, MC +0.0191 (point estimates); pre-entry n=6, critic +0.0439, MC +0.0771 (point estimates).
- `s0_k2_u12`: all-query n=32, critic +0.0022, MC -0.0030 (point estimates); pre-entry n=8, critic +0.0053, MC -0.0121 (point estimates).
- `s0_k2_u24`: all-query n=32, critic -0.0020, MC +0.0018 (point estimates); pre-entry n=5, critic -0.0126, MC +0.0114 (point estimates).
- `s1_k0_u1`: all-query n=32, critic +0.0062, MC -0.0646 (point estimates); pre-entry n=7, critic +0.0214, MC -0.2699 (point estimates).
- `s1_k0_u12`: all-query n=32, critic -0.0075, MC -0.0012 (point estimates); pre-entry n=6, critic -0.0094, MC -0.0034 (point estimates).
- `s1_k0_u24`: all-query n=32, critic -0.0185, MC -0.0050 (point estimates); pre-entry n=7, critic -0.0842, MC -0.0227 (point estimates).
- `s1_k1_u1`: all-query n=32, critic -0.0197, MC +0.0119 (point estimates); pre-entry n=8, critic -0.0895, MC +0.0477 (point estimates).
- `s1_k1_u12`: all-query n=32, critic +0.0060, MC +0.0061 (point estimates); pre-entry n=5, critic +0.0361, MC +0.0389 (point estimates).
- `s1_k1_u24`: all-query n=32, critic +0.0027, MC +0.0001 (point estimates); pre-entry n=4, critic +0.0205, MC +0.0007 (point estimates).
- `s1_k2_u1`: all-query n=32, critic +0.0052, MC -0.0022 (point estimates); pre-entry n=4, critic +0.0142, MC -0.0177 (point estimates).
- `s1_k2_u12`: all-query n=32, critic +0.0005, MC +0.0043 (point estimates); pre-entry n=3, critic +0.0031, MC +0.0463 (point estimates).
- `s1_k2_u24`: all-query n=32, critic -0.0221, MC -0.0047 (point estimates); pre-entry n=8, critic -0.0967, MC -0.0187 (point estimates).

## Audit constraints and interpretation

Computed new transitions: 0 (point count). New rollouts: 0 (point count). Checkpoint loads for inference: 0 (point count). Parameter updates: 0 (point count). Only saved arrays and JSON records were read; no pre-existing input artifact was overwritten, moved, or deleted.

Two candidate pairs and 18 comparisons cannot establish a general property of the critic. This audit can only distinguish systematically reversed in the leverage region from unordered noise, and it is underpowered for the latter. It has no authority to conclude that the critic is fine.

Protocol: `PROTOCOL.md`. Machine-readable values and all bootstrap-validity counts: `results.json`. Concise interpretation: `SUMMARY.md`.
