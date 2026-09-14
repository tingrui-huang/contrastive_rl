# Matched real-context PointMaze failure and exposure audit

## Decision

The strongest remaining error is **failure-onset overprediction in matched real actor contexts**, including both hazardous and impossible nonhazardous landings. There is also a separate position error: on the actor's actual alive contexts the frozen position models do not enter hazard too often, but they **remain in hazard too often, exit too rarely, and move right much less than the real transition**. The joint predictor inherits both problems. Its apparently good calibration on the left approach is a cancellation between an overpredicting failure head and an underpredicting hazard-entry proposal, not evidence that the components are correct.

Autonomous saved baseline trajectories enter hazard more often than native episodes even though the matched one-step models underpredict entry. That contradiction leaves generated-state compounding and the historical learned nominal-action mixture unresolved. The present audit therefore does not force a single-bottleneck conclusion: onset calibration is the first priority, hazard residence/exit is a second identified defect, and autonomous visitation mismatch cannot yet be assigned between generated states and the nominal policy.

## Execution and replay integrity

The protocol was sealed before replay or sampling. No equivalent completed diagnostic was found. The supplied retrospective support calculation was inspected and treated as complete: its approximately 0.47--0.57 return increase from zeroing onset probability outside hazard was not rerun.

All 48 saved native actor episodes were replayed from their original seeds using their 2,400 saved actions. Before every action, the original shadow teacher was queried with a separate predeclared RNG and one persistent episode memo. The original 0.05 episode-level force-safe law, per-step hidden-bit reaction, clipping, and nonzero-action-only noise convention were preserved. Five of 48 newly sampled shadow episodes drew force-safe; this is sampling variation, not a changed probability. Every saved observation, successor, death state, onset, and reward matched exactly. All 2,400 teacher queries left native XY, F4 frames, goal, hidden bits, death mode, resampling flag, and environment RNG unchanged.

The resulting advice is newly sampled advice coupled to the saved step's hidden situation; it is not recovered historical advice or an independent nominal action. Only 1,165 alive-before rows were scored, containing 28 onsets, 103 hazardous actual landings, and 1,062 nonhazardous actual landings. All 28 onsets occurred on hazardous landings. Position calls used the implementation's order `(theta, state, action=xq_saved_actor, xb=xb_shadow, key, 64)`; natural and executed actions were not swapped. Each position sample array was reused across both heads.

## A. Failure probability at the actual successor

Both frozen heads clearly overpredict onset under the frozen actor's real visitation distribution. Across all 1,165 rows, observed onset is **2.40%**. Head 0 predicts **6.39%**, an excess of **3.98 points** with episode-bootstrap 95% interval `[2.71, 5.58]`; head 1 predicts **4.96%**, excess **2.55 points** `[1.41, 3.91]`. Their all-row Brier scores are 0.0331 and 0.0231, and log losses are 0.1315 and 0.0855.

Conditioning on the actual landing does not remove the error:

- On 103 hazardous landings from 41 episodes, observed onset is **27.18%**. Heads 0 and 1 predict **47.09%** and **39.22%**, excesses of **19.91 points** `[9.37, 30.31]` and **12.04 points** `[2.46, 21.54]`.
- On 1,062 nonhazardous landings, the native onset count is zero, while the heads assign **2.44%** `[1.49, 3.85]` and **1.63%** `[0.86, 2.71]` mean probability.
- In the input-defined left-approach stratum (64 rows, 43 episodes, 18 onsets), observed onset is 28.12%; predictions are 40.18% and 35.68%. The head-0 excess is clear, while the head-1 interval includes zero.
- When the current position is already inside hazard (75 rows, 29 episodes, 9 onsets), observed onset is **12.00%**; predictions are **40.32%** and **33.12%**, excesses of **28.32 points** `[15.49, 41.51]` and **21.12 points** `[8.49, 33.92]`.
- The right-exit stratum has only 17 rows and one onset; its intervals are wide and inconclusive.

This is group-level calibration evidence, not a claim that any individual Bernoulli probability is known to be wrong. Fixed-bin calibration details and every group-level score and interval are in [matched_metrics.json](matched_metrics.json).

## B. Position prediction on the same contexts

Aggregate hazardous-landing probability looks close: the real rate is **8.84%**, versus **8.45%** for position model 0 and **8.43%** for model 1; both difference intervals include zero. The mechanisms underneath do not agree:

- From currently nonhazardous states, actual hazard entry is **4.86%**. The models predict **4.04%** and **3.88%**, underprediction by 0.82 points `[-1.55, -0.25]` and 0.98 points `[-1.74, -0.40]`.
- On the left approach, actual entry is **79.69%**; the models predict **65.70%** and **63.04%**, deficits of 13.99 points `[-24.68, -4.42]` and 16.65 points `[-26.82, -7.00]`.
- From a current hazardous position, actual residence is **66.67%** and exit is 33.33%. The models predict residence of **72.50%** and **74.46%**, excesses of 5.83 points `[2.23, 10.19]` and 7.79 points `[1.10, 14.87]`; exit is underpredicted by the same amounts.
- Rightward progress occurs on **62.40%** of real rows but only **38.72%** and **39.56%** of predictive distributions. Mean real horizontal displacement is 0.2168, versus 0.0130 and 0.0558; residuals are -0.2038 `[-0.2661, -0.1399]` and -0.1610 `[-0.2075, -0.1108]`.
- No alive real transition is stationary at the inherited `1e-7` threshold. Predicted stationary mass is only 0.17% and 0.38% overall, but rises to 2.67% and 4.73% on current-hazard rows.

Thus the position models do not create excessive one-step hazard entry under matched real contexts. They do create excessive modeled residence and deficient exit, alongside a much broader rightward-motion deficit. The 64 samples per row are predictive integration only; each row still supplies one native outcome and all intervals cluster by episode. The position checkpoints were trained on mixed alive/dead outcomes without the new internal failure mode, so this audit does not validate their alive-conditional proposal law.

## C. Joint generated-position plus onset prediction

The four position/head crosses predict total onset of **4.50%--5.71%** against the observed 2.40%. Their excesses range from 2.09 to 3.31 points, and all four episode-bootstrap intervals exclude zero. All-row Brier scores are 0.0223--0.0305.

Inside-hazard current contexts, joint onset predictions are **33.54%--39.77%** against 12.00% observed, so marginalizing over position samples does not fix the head error. Elsewhere, generated hazard probability is exactly zero, yet joint onset remains **0.75%--1.53%**, entirely from the nonhazardous component. Across all contexts, the nonhazardous generated-landing component alone contributes **1.49%--2.33%** predicted onset probability while the corresponding native joint-event count is zero. No support mask was applied.

On the left approach, in contrast, joint predictions of 27.72%--31.18% bracket the observed 28.12%, with all difference intervals spanning zero. This is compensating error: the actual-successor heads overpredict in A while the position models underpredict hazardous entry in B. It must not be read as component validation.

Mean within-row covariance between generated-hazard indicators and head probability is small overall (0.00016--0.00028), while mean head probability conditional on a generated hazardous landing is 33.05%--42.05% versus 1.62%--2.54% on generated nonhazardous landings. These raw quantities show strong level conditioning but do not establish that joint dependence is correct. Because A and B already fail separately, C does not isolate an additional coupling defect. Actual-successor and marginalized-successor predictors contain different information, and their score difference is not used to assign cause.

## Hazard exposure from saved trajectories

First hazardous landing is not death-censored. It occurs in **85.42%** of native episodes, versus **97.82%** of saved position-0 baseline paths and **98.37%** of position-1 paths. Paired root-episode differences are **+12.40 points** `[3.87, 22.98]` and **+12.96 points** `[4.46, 23.57]`. Conditional first-entry time is 3.07 transitions natively and 2.81/2.86 in the models, but both time-difference intervals include zero. Every first landing is in swamp cell x=3; model landings average about 0.068 units farther left within that cell, while y differences are inconclusive.

This autonomous first-entry excess cannot be attributed to matched one-step entry bias, because B finds the opposite sign. It is evidence that compounding generated-state visitation and/or the historical learned nominal-action mixture remains mismatched. This shadow-teacher audit directly validates neither one.

Native trajectories contain **2.146** hazardous opportunities while alive, including the fatal opportunity. Survival-before-transition integration on saved baseline paths gives **2.736--3.321** opportunities across the four position/head crosses, paired excesses of 0.591--1.175 with intervals excluding zero. These values depend on the frozen onset law—including its overprediction—and therefore do not independently prove a position defect. They are nevertheless consistent with the matched residence/exit error and the autonomous first-entry excess. Total uncensored hazard visits from death-disabled model paths are deliberately not compared to death-censored native totals.

## What is established and what remains unresolved

Newly computed evidence establishes real-context onset overprediction, impossible nonhazardous probability, excess modeled hazard residence, deficient hazard exit, and deficient rightward motion. Saved autonomous trajectories also establish excessive first-hazard-entry probability. It does not establish an extra joint-coupling defect beyond those component errors, validate the learned nominal distribution, or separate nominal-policy mismatch from generated-state compounding. The 17-row right-exit stratum is too small for a firm conclusion. Feature overlap or aggregate agreement is never treated as proof about the whole model family.

The historical result remains separate: zeroing probability outside the known hazard support raises expected return only about 0.47--0.57 and leaves a substantial, uncertain residual. Neither that result nor the present one optimizes toward the native mean return.

**Recommended next intervention:** keep both position models, actor, nominal policy, and irreversible absorbing transition fixed, and fit one predeclared **support-aware, alive-only onset recalibration** with a proper probability loss—not a return target or classification threshold. Probability should be structurally zero outside the native hazardous landing support; within hazard, calibration should condition on the input-defined approach-versus-currently-inside distinction revealed here rather than apply only a global scalar. Preserve movement on the fatal incoming transition and freeze XY only afterward. Use these 48 episodes only as development evidence and evaluate the frozen intervention on new complete confirmation episodes. Position residence/exit should be addressed in a later isolated intervention, not co-modified here, so the first correction remains identifiable.

## Reproducibility and accounting

The sealed protocol, seeds, grouping rules, hashes, and caps are in [PROTOCOL.md](PROTOCOL.md), [config.json](config.json), and [provenance.json](provenance.json). Reusable tuples and predictions are in `augmented_shadow_replay.npz`, `actual_head_predictions.npz`, `position_p{0,1}_samples.npz`, `joint_predictions.npz`, and `exposure_predictions.npz`. First-entry and exposure summaries are in [first_hazard_entry.json](first_hazard_entry.json) and [exposure_metrics.json](exposure_metrics.json).

[verify_saved.py](verify_saved.py) independently reconstructs row alignment, all displayed point metrics, joint decompositions, first-entry summaries, survival-weighted exposure, hashes, and call accounting without an environment, model, actor, or nominal-policy call. It passes and writes [verification.json](verification.json).

Execution used exactly **2,400 native replay steps** and **149,120 position successors**, under caps of 2,400 and 310,000. It performed 914,970 frozen-head inference rows, 2,400 diagnostic teacher queries, zero training updates, zero actor or nominal-policy calls, zero new complete evaluation episodes, and zero new complete model rollouts. Historical artifacts and production modules were not modified. The diagnostic itself made no commit or push; repository delivery was performed only after the user's later explicit request.
