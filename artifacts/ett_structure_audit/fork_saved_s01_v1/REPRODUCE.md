# Reproduce the audit without trajectory collection

Run from the `feature/pointmaze-causal-transition` worktree with the existing NumPy/Matplotlib environment. The requested input commit is `9b8bacaa3f8a83f063fa776271244a9562a17919`. The two analysis scripts are additions made during this audit. Use a new audit output directory:

```bash
python -m scripts.audit_ett_fork_structure --out-dir artifacts/ett_structure_audit/NEW_AUDIT
python -m scripts.verify_fork_convex_witness --audit-dir artifacts/ett_structure_audit/NEW_AUDIT
```

The first script reads the two saved lower-controller model records, their separate nominal-action arrays and frozen coefficient arrays. It validates the prior manifests, but does not deserialize a nominal/diagonal network, load an actor or critic, consume native failure labels, or call an ETT sampler. It independently reconstructs the controller indices, actions, gates, matrices, rectangles, proposals and emitted positions. It writes trace summaries, exact fixed-query ordinate bounds, selected examples, full-domain path witnesses, counterexamples and provenance. The second script adds simpler convex-segment certificates, explicitly labels the rounded versus strict step envelope, and renders the witness figure. These are deterministic component calculations on saved conditioning, not generated successors in a trajectory.

`fork_traces.json` contains all 225 selected steps and their vector/matrix decompositions. `fork_capacity.csv` contains scalar comparisons for every selected step. `selected_examples.json` records the first lexicographic reset/time in each named category, including nulls when no example exists. `trace_summary.json` records seed-specific descriptive counts and mean decompositions. No return is used to select an example.

`full_domain_witnesses.json` contains one fixed-conditioning arclength-path example per seed, checked over the full action square analytically and on a finite grid numerically. `convex_segment_witnesses.json` contains 218 simpler convex-projection certificates. The latter records `strict_unit_step_feasible` for each: 206 pass the exact unit step bound and twelve use only the original float32-rounded envelope. `all_maps_min_y` in the original trace file is an ordinate capacity calculation conditional on anchor admissibility; do not read it as a strict physical certificate for those twelve anchors.

`counterexamples.json` and `additional_counterexamples.json` retain explicit failures of anchor-ball-only reasoning, nearest projection onto nonconvex geometry, and the equivalence of Frobenius and operator constraints. `failed_strict_step_check.json` preserves the first unsuccessful attempt to apply an exact real-arithmetic step bound to every literal float32 anchor. The final scripts already distinguish the two envelopes, so reproduction does not need to fail first.

The report, derivations and example commentary were written after the numerical audit. They are not automatically authored by the commands. `provenance.json`, `verification.json`, `convex_witness_verification.json` and the final hash manifest describe the source/data verification. Existing simulator/model code and checkpoints remain unchanged. No optimizer, action grid search for return, native rollout, model rollout or resampling budget is required.
