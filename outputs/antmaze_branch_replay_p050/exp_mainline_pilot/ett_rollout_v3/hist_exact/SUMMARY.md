# v3 rollouts with the EXACT history counters (`--hist-exact`, 2026-09-20): the C strata gaps stand; A unchanged

`diag_v6_ett_rollout.py roll --v3 --hist-exact` (node3; fixed v3 models,
no refit).  `manifest.json` (history_counters = exact), `REPORT.md`,
`report.json`; rollout pickles on node3.

## Why a second fix

The user's code check after a408cef: `fit_v6_ett_one_step_v2.run_length`
counts 0 at a flagged PATH START but 1 at a flagged row that follows an
unflagged one.  `--hist-fix` subtracted 1 uniformly, which fixes the
first case and breaks the second (on 200 random paths the -1 rule
mismatches ~40 % of the rows).  `--hist-exact` replicates run_length row
by row: the last reset index is the path start or the last unflagged
row; the counter is j - last on a flagged row, else 0 (verified equal to
run_length on 200 random paths).  The earlier "counter problem fully
excluded" statement is withdrawn; it is excluded now.

## C (simulator teacher along the model path), per fold, sim / model

| fold | death | reach | timeout | KS death time | pre_zone1 timeout | zone1 death | zone2 death |
|---|---|---|---|---:|---|---|---|
| 0 | 0.292 / 0.300 | 0.611 / 0.625 | 0.097 / 0.075 | 0.018 | 0.197 / 0.052 | 0.404 / 0.423 | 0.104 / 0.095 |
| 1 | 0.296 / 0.307 | 0.619 / 0.643 | 0.085 / 0.050 | 0.037 | 0.145 / 0.046 | 0.433 / 0.449 | 0.089 / 0.096 |
| 2 | 0.293 / 0.303 | 0.614 / 0.644 | 0.093 / 0.053 | 0.039 | 0.193 / 0.070 | 0.439 / 0.437 | 0.122 / 0.131 |

Pooled strata (raw -> hist_fix -> hist_exact): pre_zone1 timeout model
0.053 -> 0.059 -> 0.056 vs sim 0.178 (reach 0.419 -> 0.416 -> 0.417 vs
0.319); zone1 death-time KS 0.156 -> 0.104 -> 0.099; zone2 death-time KS
0.321 -> 0.241 -> 0.241 (death-x 0.210 -> 0.200 -> 0.200); between
death-x KS 0.168 -> 0.126 -> 0.153.  Pooled gates pass; pre_zone1
(reach, timeout), between (death x) and zone2 (death time, death x)
still FAIL.

Reading.  With the counters now exactly the training rows', the C
picture is the one hist_fix showed: rates and AUROC (0.995) are right,
the model under-stalls before the mouth by ~3x (0.056 vs 0.178) and the
in-zone death timing / position are off (zone2 KS 0.24 / 0.20).  These
are motion / onset-model errors along long paths, not counter errors.

## A (memoryless nominal), pooled

death 0.036 vs sim 0.294, reach 0.852 vs 0.615, AUROC 0.687; every
stratum fails on death.  Identical to raw / hist_fix within noise.  The
cause is located in ett_advice/SUMMARY.md: the nominal never holds
inside a hazard band (hold probability 0.000 there), so the onset head
-- which needs a sustained in-band hold -- sees no danger; the counter
was never the missing piece.

## Status of the ETT line

The learned ETT is NOT running end to end: with real teacher advice the
long-path errors above remain; with the learned nominal the danger is
lost.  The oracle gain of the CRL line is not a learned-ETT result.
Next (user's order): the advice process (ett_advice/SUMMARY.md gives the
acceptance metrics and a candidate); the C strata gaps are a separate,
later motion / onset item.
