# Zero-bias ETT return-search ablation

The completed fixed-budget comparison is
[four_weights_s01/REPORT.md](four_weights_s01/REPORT.md).

The two output biases remain exactly zero. Only four selected output-weight
coordinates are searched from the original zero-residual initialization.
The diagonal loss is constant; this is not joint two-loss training.

The `smoke` directory contains a separate one-update check, not a final
comparison result. Existing six-parameter results are reused after exact
fixed-key reproduction. No old artifact is overwritten.

Checkpoints remain local and Git-ignored. All evaluation outputs, configurations,
comments and reports are in English. Code, configurations and evaluation artifacts are versioned; checkpoints remain local.
