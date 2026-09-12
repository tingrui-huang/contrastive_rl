# Fixed scalar failure-bank experiment

The authoritative run is
[bank_m3_lambda01_s012/REPORT.md](bank_m3_lambda01_s012/REPORT.md).
It compares diagonal-only and joint bank-distance training with the unchanged
ConditionalSpline, a predeclared bank {-3}, lambda=0.1, and three seeds.

[REFERENCE.md](bank_m3_lambda01_s012/REFERENCE.md) contains the derivation saved
before training, including the nonzero optimal diagonal and a tightly bounded
reference for the finite architecture. It is used only for evaluation.

The sampler preserves the same nominal-action marginal in both losses. The
previous supervised experiments use different sampling and are historical
context only. `smoke/` is a separate five-step execution check.

Checkpoints remain local and Git-ignored. Configurations, metrics, evaluation
arrays, English reports and plots are prepared for version control.
