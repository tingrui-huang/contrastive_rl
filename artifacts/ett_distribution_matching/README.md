# PointMaze single-step ETT distribution matching

The current completed comparison is [f4_p30_s01_guarded/REPORT.md](f4_p30_s01_guarded/REPORT.md).
It contains five fixed-budget 500-update arms, both learner seeds, K=8/16
evaluation, dataset and failure-bank provenance, and reproducible commands.
The nominal policy and agent action policy were frozen throughout.

The MMD objective improves while diagonal likelihood remains almost unchanged
relative to the diagonal-only controls. However, matching the unconditional
failure bank substantially reduces stationarity on observed frozen histories.
This is a tested optimization prototype, not a validated failure kernel.

`sanity` and `sanity_v2` are three-update numerical checks. `f4_p30_s01` is a
superseded comparison: float32 rounding triggered unnecessary bound fallbacks
at saturated residuals. `f4_p30_s01_guarded` reruns the same budget with a
conservative rounding margin, without retuning lambda or L.

Every checkpoint is local and Git-ignored. The configuration, evaluation
records, metrics, reports, and plots are English. Code and non-checkpoint artifacts are published with user authorization.
Checkpoints remain local.
