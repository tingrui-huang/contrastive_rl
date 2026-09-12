# PointMaze failure-objective diagnostic

The completed diagnostic is documented in [f4_p30_s01/REPORT.md](f4_p30_s01/REPORT.md).
It compares the existing full-F4 conditional MMD objective with hard nearest-reference-set
distance using the preserved guarded ETT checkpoints, training-only failure bank,
normalization, frozen policies, and evaluation contexts.

The full run evaluates five checkpoints, sample counts 8 and 16, and four sampling
seeds. The `smoke` directory contains the preceding 128-context check. No model was
trained or updated. Checkpoints remain local; published NPZ files contain evaluation
samples and objective gradients, not model weights or the offline training dataset.

The smoke configuration preserves the test-source hash from before an absolute
tolerance of `1e-7` was added to the float32 V/U subtraction assertion. The
diagnostic calculations did not change. The full run records the final test-source
hash; all seven tests passed with that tolerance.

The [future A/B/C specification](f4_p30_s01/NEXT_ABC_SPEC.md) describes a possible
follow-up only. That comparison was not implemented or run.
