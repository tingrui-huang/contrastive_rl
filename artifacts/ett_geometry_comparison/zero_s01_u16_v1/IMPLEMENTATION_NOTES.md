# Preserved implementation qualifications

No training or stochastic evaluation call failed. All 2,098,720 planned draws
were charged once and completed. No coefficient, hyperparameter, seed, geometry
construction or optimization budget was changed in response to the results.

1. Preparation initially rejected the evaluation seed because its namespace scan
   included the newly written evaluator among historical files. This was a
   scanner bookkeeping defect, caught before the experiment directory and before
   any model draw. The repair excluded both current comparison scripts and the
   preserved preparation source snapshot. The requested namespaces then had no
   prior matches. The original runner and failure record are retained in
   [implementation_attempts](../implementation_attempts/seed_scan_failure.json).
   The successful configuration/source/input hashes were recorded after this
   repair and before training.

2. The independent saved-result checker initially imposed 1e-12 on the update
   norm. The reused original `descent_step` deliberately returns float32: its
   largest saved update norm, measured in float64, is 0.20000000596810386.
   Every saved update exactly equals a reconstruction with the original helper.
   The checker was repaired to use the predeclared 2e-6 numerical tolerance;
   neither optimizer nor coefficients changed. This deterministic failed check
   consumed zero model transitions. Its source and failure record are preserved
   [here](../implementation_attempts/saved_checker_failure.json).

3. A diagnostic-only difference appears at fixed component panel index 621:
   current XY=(1.947675824,3.999999762), anchor=(2.947675705,3.684089184).
   Calling the selector without an enclosing JIT rejects the rounded near-ceiling
   segment and returns its fallback anchor. The emitted compiled map accepts
   Q=(1.874999762,2.999999762). This is float32 evaluation/fusion sensitivity at
   the predeclared availability threshold, not an execution-action selector.
   Both obey the rounded geometry/step checks; the fixed action-pair checks pass.
   The raw fixed-component `eligible` summary records the standalone count of
   217. Reporting of actual selection uses the emitter's count of 218 instead;
   [fixed_component_set_audit.json](fixed_component_set_audit.json) retains exact
   corrected counts and the example. The original raw summary is preserved.
   On every independent full rollout, recorded eligibility and actual selection
   agree exactly (586 and 570 selected steps in the two segment models).
   No production geometry, anchor or compiled rollout was changed or rerun.

The numerical contract continues to distinguish a real-arithmetic fixed-convex-set
proof from float32 implementation checks. Cross-shape/compiler bitwise geometry
selection is not promised at rounding thresholds. Same-shape reload replay is
exact for every saved field. The prior twelve literal unit-step anchor excesses
remain documented; this experiment uses the same rounded q+/-1 envelope and L=1.
