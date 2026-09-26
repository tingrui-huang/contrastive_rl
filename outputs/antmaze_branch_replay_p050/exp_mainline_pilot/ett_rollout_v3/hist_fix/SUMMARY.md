# v3 full-length check with the history counters aligned to the training rows (`--hist-fix`)

The rollout passed run-length 1 at the first hold / band row where the
training rows carry 0 (an offset of +1 on every hold / band row; user's
finding).  Same v3 models, same sealed thresholds; `REPORT.md` /
`report.json`.

C per fold, before -> after the fix: death 0.301 / 0.309 / 0.305 ->
0.299 / 0.304 / 0.301 (sim 0.29-0.30); reach 0.627 / 0.640 / 0.647
unchanged; pre_zone1 timeouts 0.057 / 0.041 / 0.062 -> 0.050 / 0.058 /
0.070 (sim 0.15-0.20); death-time KS 0.025 / 0.043 / 0.052 -> 0.028 /
0.029 / 0.034.  The offset was real but is not the driver of the
remaining C failures (under-stalling before the mouth; in-zone death
timing), and it does not touch the advice-process gap (memoryless
nominal A: death 0.03 vs the teacher's 0.30).  The pooled tables are in
REPORT.md; the sealed gate verdicts are unchanged (C strata fail; A
fails).  Next for the ETT (user): the advice process -- history
consistency between training and generation, and whether a learned
advice process can retain the persistent hazard information -- not
another motion regression; the simulator teacher stays a diagnostic
reference only.
