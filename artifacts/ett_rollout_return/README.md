# Hard rollout-return ETT experiment

The completed restricted experiment is [residual6_s01/REPORT.md](residual6_s01/REPORT.md).
The `preflight` directory records the frozen-model return-signal gate and a
supplementary fixed-actor real-environment comparison.

The objective has exactly two terms: the preserved diagonal likelihood and
positive expected discounted hard task return, minimized. This run freezes the
diagonal backbone and searches six residual-head parameters only; diagonal
loss is constant, so this is not joint two-loss training.

Two search seeds have matched lambda=0/1 controls, 12 fixed-budget updates and
independent evaluation randomness. No critic, failure scorer, bank objective,
hidden training label, hindsight goal or actor update is used. Previous critic
diagnostics were read and not repeated.

All checkpoints remain local and Git-ignored. Main trajectory files store the
agent action; auxiliary files separately store the sampled observational action.
Code, configurations and evaluation artifacts are versioned; checkpoints remain local.
