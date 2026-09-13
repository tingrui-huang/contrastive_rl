# Early PointMaze contexts: coverage improved; value estimation blocks optimization

**The immediate remaining limitation is value estimation.** Coverage and the
bounded generator-capacity checks passed. The initial region-NCE critic failed
its predeclared calibration gate on independent early-root continuations, so
the experiment stopped before any ETT update. There is no two-seed optimization
result, no diagonal-only/joint return-change estimate, and no optimization
failure claim. The paired training/evaluation implementation is present, but
was not authorized to proceed past the failed gate.

## Contexts now precede outcomes

Starting revision is `0059c98`. The protocol and source/input hashes were saved
before inspecting even observable coverage counts. Selection used only recorded
observation prefixes at candidate times 1,3,6,10, outside the existing goal
region, with no prior goal visit and current displacement greater than .05.
There are three visible groups: approach to the hazard, transit through it,
and the lower bypass. Each train/validation partition has 12 distinct episodes
per group; no episode appears in two groups or both partitions.

The separate native-label audit found **72/72 selected roots alive when
observed**. This resolves the previous pilot's all-dead outside-root population
for this sample. Actual root times are 1 for approach, 3/6/10 for transit, and
3 for bypass. Their exact remaining horizons are 49/47/44/40, always ending at
the task's original time 50. F4 histories are recorded, including legitimate
reset-history repetitions at t=1. No teleportation, hidden-label selection or
replacement sampling was used.

## The generator expresses different task-return continuations

The initial kernel and two fixed response probes, +.35 I and -.35 I across
the existing eight gated matrices, used 16 complete continuations per root.
No probe was optimized or selected as an initialization. **Three approach
roots and two transit roots** have at least two goal-reaching and two
finite-horizon-zero paths under at least one probe. There are **393 realized
recovery-after-setback paths** across all three probes. Initial root mean
occupancy SD is **0.05899**, above the .025 signal gate.

Initial-model goal-reaching fractions are **95.31% approach, 93.75% transit,
100% bypass**. Initial normalized occupancies are **.54948/.64829/.61805**.
The response probes change both occupancy and recovery frequency. These are
bounded empirical capacity checks, not proof of full-family expressiveness.
Only five contexts met the mixed-outcome criterion, and bypass remained
goal-reaching in every sampled probe path; capacity coverage remains narrow.

Recovery means an actual later goal visit after a declared observable pause
or distance setback. A finite-horizon-zero path is **not** a death label;
neither stationary output nor failure to reach the goal proves irreversibility.
The generator still has no persistent hidden death state, and its inherited
rectangle geometry excludes some valid fork transitions. Those structural
limitations remain even though this particular capacity gate passed.

## Calibration failure is concentrated at early roots

The critic uses the unchanged binary region-NCE objective, q=(.5,.5), alpha=0,
positive/negative weighting 1:31, and
`Q_h=(1-.95^h)*31*.5*exp(f_1)`, `Q_0=0`. The sole horizon-input adaptation is
scaling by 50 instead of 10. It trained for the fixed **1,500 Adam steps** on
**288 model paths / 13,648 transition-label queries**. Exact values, hidden
labels and native future outcomes were not training targets.

Independent evaluation holds one first action fixed per query and repeats
the same frozen generator 16 times. All values below are normalized discounted
goal occupancy, `.05 * sum(.95^t * reward)`:

- **Root overall:** RMSE **.07950**, bias **+.04974**, failing .06/.025 gates.
- **Approach roots:** RMSE **.06141**, bias **+.05529**; group bias gate .035 fails.
- **Transit roots:** RMSE **.11278**, bias **+.06214**; group .07/.035 gates fail.
- **Bypass roots:** RMSE **.04970**, bias **+.03177**; group gates pass.
- **Midpoint overall:** RMSE **.01788**, bias **+.00958**; gates pass.
- **Last step overall:** RMSE **.00185**, bias **+.00048**; gates pass.

Post-run descriptive reuse of the saved arrays gives overall root bias 95%
episode-stratified bootstrap CI **[+.02895,+.06883]**. Approach bias CI is
**[+.04136,+.07075]**; transit **[+.00815,+.11266]**. Repeats were averaged
before clustering, with no extra model sampling or gate changes. Individual
target Monte Carlo SE reaches .079, so exact per-context errors remain noisy;
the aggregate overestimation is not just an isolated high-error target.

Root predictions do not exceed their theoretical occupancy bounds. Some
midpoint predictions overshoot by up to **.03881**, retained without clipping.
The error pattern supports a failure of early-context continuation estimation
within this budget. It does not isolate architecture, finite-sample fitting,
or optimization of the critic as the unique cause. Root queries are only
288/13,648 (2.11%) of the uniform transition-label training rows, which is a
sampling-allocation fact, not a tested causal explanation. Hidden-state
uncertainty is not an explanation for failing these particular targets: the
critic and targets query the same visible-state model kernel.

## Constraints, native evidence and stopping

No ETT parameters were updated: diagonal parameter degradation is exactly
zero. Both probes leave the diagonal sampling law exactly unchanged at x=x'.
All probe action/F4/geometry checks passed; maximum numerical Lipschitz excess
is zero, and probe matrix Frobenius norm is .494975. The unchanged construction
provides samplewise Euclidean action L=1 and a common-x' Wasserstein coupling,
not the toy TV bound or a complete native-physics guarantee. No new restriction
or third loss was introduced.

Native logged behavior subsequently reached the goal in **8/12 approach,
12/12 transit, and 12/12 bypass episodes**. Four approach episodes later died;
none was already dead at its root. In independent same-generator calibration
paths, model goal fractions were **95.83%/93.23%/100%**. Model mean occupancies
were **.56990/.62408/.62085**, versus native logged **.43425/.72570/.62697**.
This is a disagreement between two continuation populations, not a measured
conditional ETT error: the logged collector is different from the frozen
actor, and logged hidden contexts are not ETT ground truth. No new native
simulator rollout was run. Native evidence was kept separate from selection,
critic fitting and checkpoint decisions.

**Stop at the value-estimation gate.** Coverage is no longer the immediate
blocker, the finite probe gate passes with limited breadth, and ETT optimization
remains untested here. Do not infer death, a globally optimal worst case, or
causal identification from these model returns. No actor training, loss change,
budget expansion, additional experiment family, commit or push occurred.

Total charged model outputs: **142,961 / 1,500,000**. All **28 tests** passed,
including previous finite-state and PointMaze regressions. A separate read-only
audit reproduced predictions, returns, first-action alignment, F4 history and
remaining horizons. Checkpoints remain local in the ignored `checkpoints/`.

## Reproduction

From the PointMaze worktree, with the frozen local inputs in preregistration.json:

```powershell
python -m ett.pointmaze_early_pilot prepare --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1
python -m ett.pointmaze_early_pilot run --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1
python -m ett.check_pointmaze_early_pilot --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1
```

Use a fresh directory for a new reproduction; the completed attempt cannot be
overwritten. `REPORT.md` lists every group/phase metric and test commands.
`PROTOCOL.md` and `preregistration.json` record the pre-outcome rules/hashes.
`generator_*.npz` and `preflight.npz` retain all paths and separate executed
actions/x_prime draws. Unused variable-horizon slots are NaN and never enter
returns. `descriptive_uncertainty.json` and `postrun_checks.json` are explicitly
post-run, zero-new-outcome audits, not retrospective preregistration.
