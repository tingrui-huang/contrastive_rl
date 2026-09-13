# Execution record

- Started on feature/pointmaze-causal-transition at23da83f. Inspected the old
  uniform sampler, early-context reports, evaluator, nominal/actor interfaces,
  and verified F4 teacher collector. Prior production/regression files remain
  unchanged; only a local-input ignore rule was added to .gitignore.
- Specified two critic arms and seeds, shared replayed training paths/labels,
  fixed phase boundaries,1500 steps/arm, fresh final episode collection,
  independent calibration, ranking noise screens and clustered intervals
  before observing new outcomes.
- All33 tests passed: the previous28 plus phase distribution/label independence,
  original NCE optimizer equivalence, query action/horizon alignment,
  ordering/uncertainty handling and native context-interface smoke checks.
- Saved protocol/configuration/source/input hashes in a new directory before
  the experiment. Replayed288 old training paths and their13,648 positives
  once, then shared identical arrays across all four critics.
- Completed all four final1500-step critics. Uniform seed0 exactly reproduced
  the previous initial critic parameter hash and all old diagnostic predictions.
- Sealed training/checkpoint hashes before any final native episode existed.
  Evaluated seen training queries with independent continuations; no further
  fitting occurred. Old target arrays were explicitly marked inspected.
- Collected exactly600 fresh native teacher episodes,30,000 steps. Observable
  selection found35 bypass candidates and enough other candidates; selected12
  episodes per group without hidden-label selection or additional collection.
- Collected the registered independent final query paths and64-repeat model
  targets, shared across all critics. Evaluated all final iterates, including
  mixed outcomes. No selection, retuning, additional fitting or ETT update.
- Model-output ledger closed at538,221/650,000. Native steps30,000/30,000;
  critic optimizer steps6000. No extension of any budget.
- Added a read-only post-run audit reproducing shared labels, row exposures,
  paired initialization/final hashes, old baseline, new episode selection,
  saved queries/predictions/metrics and budget accounting. All checks passed;
  no new model outcomes were generated.
- SUMMARY.md interprets the mixed findings. REPORT.md and JSON/NPZ artifacts
  retain all group/phase calibration and noise-screened ordering results.
  Checkpoints and full raw native source remain local. No commit/push.
