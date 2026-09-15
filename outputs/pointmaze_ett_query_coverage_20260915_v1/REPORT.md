# Fixed-ETT query-coverage intervention

Primary result: **`QUERY_COVERAGE_NOT_SUFFICIENT_AT_FIXED_BUDGET`**. The selected ETT was not retrained, sigmoid-NCE was unchanged, and both 30k-step CRL arms used BC 0.05 from hash-matched initialization.

## What changed

Both arms used the same 198 supervised-training fork roots, 550 `(s, xb)` contexts, exact retained original half, continuation actor, and paired random streams. B queried its first action from the old observational actor. C supplied three sealed down-neighborhood and three sealed right-neighborhood queries per context. Every one of the 3,300 paths per arm was kept for the full 49 remaining transitions; synthetic padding was excluded with structural lengths.

B's first queries contained 18 down-sector and 953 right-sector paths. C contained 1650 and 1650. Generated B/C region reach was 0.238/0.529; absorption 0.759/0.457. C retained 326 out-of-hazard onsets of 1507 total; no onset correction or outcome filtering was applied.

## Did coverage transmit to the held-out critic?

On 16 evaluation-only coherent alive roots, B's task-goal down-minus-right logit was -1.718 [-1.749, -1.686] with 0/16 roots preferring down. C's was -0.099 [-0.147, -0.048] with 4/16. The paired C-minus-B change was +1.619 [+1.594, +1.644]. Gates: {'absolute_C': False, 'incremental_C_minus_B': True, 'C_accuracy': False}.

The exact 7,680,000-row sampler lineage contains 78,222 synthetic time-zero positives. Candidate and task-near-future counts are in `lineage.json`; this shows what entered NCE rather than relying on total generated transition count.

## Where the remaining exact-goal error sits

C remains negative even on the 198 construction roots: central down-minus-right -0.153 [-0.170, -0.136], with 25/198 positive. The strict held-out failure is therefore not merely a held-out-root generalization failure.

However, the relabeled target distribution contains zero exact copies of the canonical stationary F4 goal. Among C's time-zero positives, down/right have 38,948/39,274 goals; their nearest full-F4 distances to canonical are 0.081/0.099, and only 3/1 lie within 0.1. Down nevertheless has 9,030 futures within 0.5 versus 4,496 for right.

On 512 actual down-route goal-near F4 goals drawn from those positives, C's held-out down-minus-right logit is +0.728, positive on all 16 roots. On a right-route goal bank it is -0.427. Thus C learned action-to-supported-future discrimination, while the exact stationary canonical-goal probe remains a small extrapolation error. `fit_diagnostic.json` also evaluates 512 exact lineage batches: C improves time-zero positive-versus-negative separation, but the more diverse replay is harder overall. These logits are density-ratio diagnostics, not calibrated success probabilities.

Actor samples on the held-out roots enter the noiseless lower cell with probability 0.001 (B) and 0.035 (C). Because C's covered actions also enter the unchanged BC term, actor movement is not a critic-only attribution.

## Fixed-final native evaluation

| policy | B reach | C reach | C-B reach (paired 95% CI) | B lower | C lower | C-B lower (paired 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| mode | 0.305 | 0.845 | +0.540 [+0.475, +0.605] | 0.000 | 0.780 | +0.780 [+0.720, +0.835] |
| sample | 0.315 | 0.640 | +0.325 [+0.250, +0.395] | 0.070 | 0.550 | +0.480 [+0.405, +0.550] |

Native evaluation used 200 fresh paired reset seeds only after training. It did not select checkpoints or change the fixed budget.

Although the strict exact-canonical critic gate fails, the native policy effect is large in both protocols. Query coverage is therefore a demonstrated major cause of the prior policy failure and is sufficient for useful policy learning in this intervention, but the experiment does not establish it as the unique cause or satisfy the stricter canonical critic criterion.

## Trace

`trace.json` follows the deterministic context-0 central-down slot from its exact collected supervised tuple through the shared nominal condition, B/C first queries and learned successors, complete continuation hashes, one exact shared NCE `(trajectory, anchor, future)` occurrence, and final actor/critic preferences. The trace path is fixed by array order and was not selected by outcome.

## Interpretation limits

This one-seed intervention tests sufficiency of a particular sealed action-covering generator at a fixed learner budget. It does not prove uniqueness of the cause. It intentionally leaves the fixed ETT's out-of-support death errors in the replay, so failure would still be compatible with goal-distribution, critic-fit, model-quality, or finite-sample limitations. Model-internal success is not native success.
