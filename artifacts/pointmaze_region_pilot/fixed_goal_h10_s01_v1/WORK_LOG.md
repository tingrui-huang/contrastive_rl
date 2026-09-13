# Execution record

- Worktree: `.worktrees/pointmaze-p30`; branch `feature/pointmaze-causal-transition`;
  base HEAD `9f17c4e`.
- Inspected the finite-state regression reports, original convex and diagonal
  samplers, fixed actor and nominal loaders, previous geometry/reward/death
  audits, dataset collection format and available local input checkpoints.
- Implemented the isolated region-label NCE adapter, shared-kernel updates,
  pre-optimization gate and independent evaluation. No production source was
  edited; checkpoints and original normalization artifacts were retained.
- Ran `python -m unittest scripts.test_finite_crl scripts.test_finite_neural
  scripts.test_finite_setback scripts.test_pointmaze_region_pilot -v`: 22
  tests passed (16 regressions plus the first six integration checks).
- Added the frozen-surrogate smoke check; reran
  `python -m unittest scripts.test_pointmaze_region_pilot -v`: all 7 passed.
  Thus all 23 distinct checks passed. Component tests are a separate fixed
  small budget and did not select pilot hyperparameters.
- Preparation's input assertion caught 51 stored action slots, not 50: the
  collector has a terminal dummy action. Corrected that assertion and
  documented the format before any pilot outcomes or protocol creation.
- Prepared the fresh output directory, copying and hashing protocol, sources,
  configuration and observable-only contexts. No audit labels were loaded.
- Ran the predeclared preflight, passed its calibration gate, then completed
  both seeds and both arms with 6 updates each. No tuning, retry, early
  stopping, checkpoint selection or budget change occurred.
- Ran the one independent evaluation attempt on all five frozen kernels.
  Separate final critic refits used only new model continuation positives.
  The output ledger closed at 742,016, below the 1,500,000 cap.
- Hidden labels were read only for post-training audits. The zero alive-outside
  count was reported without new collection or changing context selection.
- Added and ran `ett.check_pointmaze_region_pilot` after outcomes as a read-only
  execution-integrity audit. It reuses saved arrays and collects zero model
  outcomes. It does not alter the saved protocol, scores, tolerances or
  checkpoint selection. All checks passed.
- `SUMMARY.md` interprets saved results; `REPORT.md` is the mechanically
  generated detailed report. No additional model experiment was run.
- Local checkpoints are ignored. No commit, push or remote training launched.
