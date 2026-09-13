# Execution record

- Branch feature/pointmaze-causal-transition, base 0059c98. Previous sources,
  reports, production actor/critic and generator checkpoints remain unchanged.
- Read the preceding region-NCE implementation/report, finite-state reference,
  dataset collector, original reward/F4 conventions and prior geometry audit.
- Implemented early observable-only episode selection, a separate stop-only
  native coverage audit, fixed generator probes, variable-horizon NCE and
  paired two-loss updates with final-iterate independent evaluation.
- Wrote selection, gates, budgets, model probes, critic schedule and final
  statistical analysis before inspecting coverage or model outcomes.
- Ran all five test modules listed in REPORT.md: **28 tests passed**. Tests
  include variable-horizon calibration/positive timing/visitation weighting,
  synthetic selection access checks, real sampler action/F4 checks, frozen
  surrogate smoke, and all prior finite-state/PointMaze regressions.
- Prepared the new directory. Protocol/configuration/source/input hashes were
  saved before observable root selection. Selection archives were sealed
  before hidden death timing was audited.
- Ran one bounded attempt: coverage passed (72 alive roots), generator gate
  passed, initial 1500-step critic fit completed, independent calibration failed
  at early roots. The executable stopped before ETT optimization as declared.
- No root replacement, additional model sampling, gate changes or refit retry.
  Final-iterate joint/control comparisons were not run and are not claimed.
- The ledger closed at 142,961 emitted outputs. Two ignored local checkpoints
  remain: the unchanged initial ETT offsets and fitted initial region critic.
- Added a post-run read-only checker. It reproduced the stored critic outputs,
  return timing, action alignment and F4 histories, checked padding masks and
  the unchanged ETT, then computed descriptive intervals from saved arrays.
  This added zero model outcomes; it is not part of the original gate.
- SUMMARY.md interprets the outcome and distinguishes the immediate estimator
  blocker from remaining generator limitations and native evidence. REPORT.md
  is the executable-generated report. No commit/push or actor training.
