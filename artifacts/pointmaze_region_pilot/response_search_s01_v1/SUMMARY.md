# Response search helps one baseline; the dominant bottleneck remains unresolved

Starting from a439486, this experiment froze the two critic-driven ETTs' diagonal
networks and 16 head offsets from 817e429, along with the original actor and nominal
policy. It optimized only 32 response coefficients, from the existing response and
two predeclared feasible nonzero starts per baseline. Each start received 24 updates.
No diagonal-head rejection remained. All samples and optimizer budgets were fixed
before execution; selection and final evaluation used separate full-rollout samples.

**Broader search finds a model-internal gain for baseline 1, not consistently for
both baselines.** Returns below are normalized discounted region occupancy, as in
817e429; lower is more pessimistic. Final evaluation applies each ETT at every step.

- Baseline 0: selection retained the original model, return **0.59177**. This is an
  identity comparison, not evidence that all candidates are equivalent. Its original-
  response run ended at **0.59683**, a change of **+0.00506**, root 95% CI
  **[−0.00025, +0.01034]**; conditional MC CI **[−0.00083, +0.01096]**.
- Baseline 1: selection chose the second random initialization, before optimization.
  It returned **0.59335** versus baseline **0.60129**, difference **−0.00794**, root
  CI **[−0.01299, −0.00243]**, conditional MC CI **[−0.01336, −0.00251]**.
- Baseline 1's predeclared original-response final iterate independently returned
  **0.59008**, difference **−0.01121**, root CI **[−0.01812, −0.00380]**, conditional
  MC CI **[−0.01879, −0.00363]**. This was evaluated as a prespecified endpoint;
  it does **not** replace the selection winner after inspecting final results.
- The selected baseline-1 model's public-START difference was **−0.02121**,
  paired rollout CI **[−0.05429, +0.01214]**: improvement there is unresolved.

**Critic and matched MC disagree descriptively, but direction errors are not
resolved at this precision.** At updates 1, 12 and 24, all 18 comparisons used new
query samples, identical first-transition randomness, and the same frozen
pre-update continuation model. Critic and MC saw the same candidate successors
and next actions. Their mean signs differed in **9/18** comparisons. **0/18** had
opposite directions resolved by both root-bootstrap and conditional-MC intervals.
Every MC directional interval included or touched zero under at least one check.
This is insufficient evidence of accurate agreement as well as insufficient
evidence of reliably wrong critic directions.

One magnitude discrepancy was resolved under both intervals: baseline 1, second
random start, update 1. Critic delta **+0.00517** versus MC delta **−0.00221**;
MC-minus-critic **−0.00737**, root CI **[−0.01533, −0.00069]**, conditional MC CI
**[−0.01305, −0.00170]**. MC's own improvement remained unresolved. All 18 checks
are reported, with descriptive intervals and no multiplicity adjustment.

**Local improvements are not a reliable progress signal here.** The optimizer's
sampled critic surrogate often decreased, yet baseline 0's second random-start
run worsened full return by **+0.00691**, root CI **[+0.00199, +0.01197]**,
conditional MC CI **[+0.00181, +0.01201]**. This is an aggregate mismatch, not proof
that one MC-validated step caused the worsening. The six prespecified final-step
comparisons do not establish an MC-validated local decrease that fails to transfer:
their local MC decreases were unresolved. One final step (baseline 0, first random
start) did lower full return, but its local MC direction was also unresolved.

**Decision:** response exploration/optimization can improve one fixed baseline,
but this run does not distinguish a dominant optimization bottleneck from a
continuation-estimation bottleneck. The next research direction should be a
separately authorized **precision audit on the saved candidate pairs**, increasing
matched-MC precision and checking critic differences before spending on another
optimizer or architecture sweep. No such extension was run. A local or full-model
gain is neither global worst-case recovery nor evidence of native-policy benefit.

Verification: all **144** proposals preserve diagonal samples exactly; all **18**
evaluated models pass geometry, F4-history and action-Lipschitz checks. Frozen
parameter/source hashes passed. Response parameter L2 changes were **1.39–1.76**;
these were substantial moves, with no required sensitivity floor. Actual cost:
**5,687,360 / 6,000,000** computed transitions (including masked MC padding),
**47,400** critic steps, **zero** actor/nominal updates and **zero** native steps.
BC, reward and geometry were unchanged. Existing artifacts were preserved.

Reused held-out roots had been inspected previously; only rollout randomness is
fresh. The finite-difference gradient and candidate-selection samples also have
sampling error. A reporting-only f-string typo was corrected after all samples
were sealed; the original source and both hashes are retained in
sealed_report_source.py and reporting_fix.json. No experimental code or outcomes
were changed by that correction.

Details: PROTOCOL.md, REPORT.md, results.json, audit.json, candidate_pool.npz,
pre_final.npz, checks/*/matched_mc.npz and evaluation/*.npz.
