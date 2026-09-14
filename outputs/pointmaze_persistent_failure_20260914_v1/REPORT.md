# Failure-label-supervised persistent-mode diagnostic for PointMaze

Date: 2026-09-14. Starting commit: `8bb3c31d227918cdc9f35389011eb3e0985d27da`.

## Answer

The added persistent failure mechanism is highly consequential, but this implementation is not a credible final correction. Across all four crossed position-model/head combinations it removes 6.02--7.24 discounted-return units from the matching unchanged baseline and reduces reward occurrence by 58.0--75.0 percentage points. It therefore repairs the direction and most of the magnitude of the old excessive-reward problem consistently.

It also overshoots consistently. On the new native evaluation, reward occurs in **41.67%** of episodes and failure occurs in **58.33%**. The four joint models produce reward in only **21.39%--26.27%** of paths and failure in **73.63%--79.20%**. Approximately **16.84%--19.65% of sampled onsets occur outside native hazardous cells**, affecting **12.40%--15.56% of all generated paths**. A lower mean return of 2.44--3.02 versus native 4.67 is therefore partly caused by inappropriate failure, not simply corrected dynamics.

The one-step onset head is much more credible than its rollout behavior. Conditioned on actual next positions, aggregate onset rates, Brier scores, and log losses are good; averaging over frozen-model next-position samples only modestly degrades those metrics. Small nonhazardous false-positive probabilities of roughly 0.25%--0.34% per one-step row, however, rise to roughly 0.83%--1.29% on nonhazardous generated transitions and compound over 50 steps. Hazardous onset is also somewhat underpredicted by head seed 1. The head is not constant, but its generated-state conditional probabilities are not sufficiently support-correct or trajectory-calibrated.

This supports missing failure onset and persistence as a consequential missing process in the previous position-only rollouts. It does not show that this head is correct, that failure dynamics are the only bottleneck, or that the original observation-only ETT problem is solved. This experiment explicitly trains with labels that were previously audit-only.

## Fixed experiment

No equivalent completed experiment was found before execution; the only related artifact was the local return-gap diagnosis. Both final gated position candidates from `pointmaze_response_gate_20260914_v1` were frozen, as were their gates, nominal policy, and actor. The two unchanged baselines and all four joint models therefore share the same completed position checkpoints; no XY energy-score, position-model, actor, critic, or nominal update was performed.

The added head is one fixed 14-32-32-1 tanh MLP with sigmoid probability. Inputs are training-standardized observed F4, raw `xb`, raw `xq`, and actual-or-generated next-XY displacement. Only original episodes 0--71 determine scaling and training data. Of 9,248 rows alive before the transition, 214 (2.31%) have `dead_after=true`. Both head seeds receive 2,000 Adam updates, batch size 256, learning rate 0.001, unweighted binary cross-entropy, and final-iterate evaluation. There is no class weighting, threshold, architecture sweep, or evaluation tuning.

This is prominently an **additional-information diagnostic**: `dead_before` and `dead_after`, previously used only for auditing, are now explicit supervised targets. Training uses actual recorded successors and preserves each saved same-context `(s,xb,xq,y)` tuple. Generated rollouts use only sampled successors after divergence and never receive native labels, hidden swamp bits, or future observations.

Both heads are nonconstant. On their 9,248 training rows, head 0 predictions have mean 0.02563, standard deviation 0.12016, and range `6.4e-6`--0.99961; head 1 has mean 0.02299, standard deviation 0.11538, and range `6.4e-6`--0.99992. Mean probabilities on positive versus negative training rows are 0.6563 versus 0.01069 for head 0 and 0.6289 versus 0.00864 for head 1.

## New paired onset evaluation

The frozen heads were followed by collection of 48 new complete paired prefix episodes: 24 teacher-prefix and 24 blind-prefix, 50 steps each, four same-snapshot action queries per context, and all outcomes retained. Among 6,684 alive-before rows from all 48 episodes, 155 onset; 602 land in hazardous cells and contain all 155 onsets, while 6,082 nonhazardous landings contain none. Exact diagonal rows number 1,714 with three onsets; 4,970 off-diagonal rows contain 152.

Conditioned on actual next positions:

- Head 0 predicts an aggregate onset rate of **2.683%** versus **2.319%** observed, a paired difference of +0.364 percentage points with 95% interval `[+0.012,+0.731]`. Brier score is **0.01049**, log loss **0.03474**, and fixed-bin ECE **0.00478**.
- Head 1 predicts **2.331%**, difference +0.012 points `[-0.305,+0.323]`. Brier is **0.00989**, log loss **0.03343**, and ECE **0.00424**.
- On hazardous actual landings, the observed rate is **25.75%**. Head 0 predicts 26.31%, difference interval `[-3.01,+4.11]` points; head 1 predicts 23.32%, interval `[-5.85,+0.82]`.
- On off-diagonal hazardous landings, the observed rate is **35.19%**. Head 0 predicts 33.66% with an interval spanning zero; head 1 predicts 30.69%, underprediction `[-8.85,-0.23]` points.
- On nonhazardous actual landings, where no onset occurs, head 0 still assigns **0.344%** mean probability and head 1 **0.254%**. Diagonal hazardous cases are especially difficult: only 3/170 onset, while the heads predict 7.64% and 4.61%.

For the joint predictor, each frozen position model supplies 64 common next-position samples per row and the two heads reuse those exact samples. Across the four crosses, aggregate predicted onset is **2.181%--2.436%** versus 2.319% observed; every paired difference interval includes zero. Brier scores are **0.01082--0.01125**, log losses **0.03592--0.03701**, and ECE values **0.00258--0.00392**. Thus conditioning on sampled instead of actual positions causes only modest aggregate degradation.

The conditional weakness remains. Head-1 joint predictions on hazardous actual-outcome rows are 21.70%--21.83%, underpredicting observed onset by about four points with paired intervals excluding zero. All four joint predictors retain nonhazardous mean probability of **0.249%--0.315%** despite zero observed events. Full diagonal/off-diagonal, hazardous/nonhazardous intersections, calibration bins, counts, and episode-bootstrap intervals are in [onset_metrics.json](onset_metrics.json).

## New fixed-policy complete trajectories

The separate native set contains 48 fresh 50-step episodes under the unchanged frozen actor. It has 28 failed episodes, 20 rewarded episodes, and no surviving unrewarded episode. Native reward occurrence is **41.67%** with episode-bootstrap interval `[27.08,56.25]%`; failure is **58.33%** `[43.75,72.92]%`; mean discounted return is **4.667** `[3.045,6.256]`; return conditional on reward is **11.201**; and mean failure time conditional on failure is step **5.893**.

The unchanged baselines reproduce the original failure omission on these new seeds:

- Position seed 0: reward occurrence **96.42%**, no modeled failure, survival without reward 3.58%, and return **9.716**. Its return gap from native is +5.049 `[+3.474,+6.647]`.
- Position seed 1: reward occurrence **84.28%**, no modeled failure, survival without reward 15.72%, and return **8.997**. Its return gap is +4.329 `[+2.734,+5.944]`.

All four persistent-mode combinations reverse most of those gaps, with very similar results across position seeds:

- `p0_h0`: reward occurrence **21.39%**, failure **79.20%**, survival without reward 0.65%, return **2.473**, and conditional rewarded return 11.565. Relative to native, reward occurrence is -20.28 points `[-34.41,-5.66]`, failure +20.87 points `[+6.22,+35.06]`, and return -2.194 `[-3.839,-0.558]`.
- `p0_h1`: reward occurrence **25.91%**, failure **73.63%**, survival without reward 1.86%, return **3.015**, and conditional rewarded return 11.637. Native gaps are -15.76 reward points `[-30.08,-1.33]`, +15.30 failure points `[+0.62,+29.59]`, and -1.652 return `[-3.297,+0.027]`.
- `p1_h0`: reward occurrence **21.52%**, failure **78.16%**, survival without reward 0.81%, return **2.440**, and conditional rewarded return 11.338. Native gaps are -20.15 reward points `[-34.09,-5.86]`, +19.82 failure points `[+5.60,+33.76]`, and -2.228 return `[-3.872,-0.574]`.
- `p1_h1`: reward occurrence **26.27%**, failure **74.32%**, survival without reward 0.26%, return **2.977**, and conditional rewarded return 11.334. Native gaps are -15.40 reward points `[-29.46,-1.07]`, +15.98 failure points `[+1.59,+30.15]`, and -1.690 return `[-3.308,-0.024]`.

The mechanism's direction is consistent. Relative to their matching baselines, the four joint arms reduce reward occurrence by 58.01--75.03 points and return by 6.019--7.243; every paired interval excludes zero. Position-model variation is small after adding failure: within a fixed head, changing position seed changes reward occurrence by only 0.13--0.36 points and return by -0.034 to -0.038, with intervals spanning zero. Head seed 1 is systematically less aggressive than head 0 for both position models, increasing reward occurrence by about 4.5--4.8 points and return by about 0.54.

Conditional successful behavior is not destroyed in return magnitude: joint return conditional on reward is 11.34--11.64 versus native 11.20, and all paired difference intervals span zero. Conditional failure timing is also close: 5.70--6.22 versus native 5.89, again with every difference interval spanning zero. The main defect is excessive onset frequency.

That defect is structurally visible rather than hidden by a support mask. The four joint arms put mean failure probability **0.83%--1.29%** on alive transitions whose sampled landing is outside the native hazardous cells. Of 2,262--2,433 sampled onsets per arm, 381--478 occur outside hazard: **16.84%--19.65%** of onsets, causing **12.40%--15.56%** of all paths to acquire a false outside-hazard failure. Repetition over 50 generated steps makes these small per-transition errors consequential.

## Model-contract checks

The generated joint transition implements native event timing: it samples and retains the landing first, then samples failure. Every saved fatal-onset successor exactly equals its frozen position-model proposal; 94.57%--100% of onset transitions move at the original `1e-7` threshold. No incoming fatal movement is frozen.

All mode sequences are monotone, with zero return-to-alive or persistence violations. On every transition already in failed mode, XY is exactly unchanged and F4 shifts normally. Rewards are recomputed from the generated joint outcome and native fixed-task semantics; none is edited or penalized. All position proposals and emitted joint successors satisfy the original native geometry checks.

Disabling the mechanism reproduces the frozen baseline exactly under either head parameter vector. Before each joint path's first sampled onset, its actor action, nominal `xb`, and position proposal exactly match the corresponding baseline under common randomness. The actor and nominal receive only their original observed F4 and fixed goal; the internal mode and all native labels remain outside their inputs. No hazardous-cell support restriction is applied.

The inherited position-only action bound remains applicable to the position proposal conditional on a fixed current mode. The added discrete onset and persistent state change the joint transition and future trajectory, so this report makes no claim that the original full action-Lipschitz guarantee survives for the joint model.

## Interpretation and next action

The result supports missing failure dynamics as a consequential component: introducing labeled onset plus structural persistence consistently removes most of the baseline reward-occurrence and return excess, and position-seed variation becomes small. It does not support accepting the present model, because all four combinations overpredict failure, underproduce successful paths, and create many failures where native failure is impossible. Aggregate one-step calibration alone is insufficient for a rare event repeatedly queried under generated-state shift.

The remaining uncertainties include onset calibration on generated rather than native state distributions, whether an observation-only learner can recover the persistent process, nominal/actor distribution shift after generated divergence, and other position-transition errors. The labels used here provide additional information and cannot be retroactively attributed to the original unlabeled ETT method.

**Single next action:** run one fresh, predeclared exact-checkpoint diagnostic that applies the explicit native hazardous-landing support restriction `p(failure)=0` outside `3<=x<6, 3<=y<4`, leaving the two trained heads unchanged inside hazard. This is additional environment knowledge and must be disclosed as such, not slipped in as an evaluation-time mask. The observed 12.4%--15.6% of paths with outside-hazard onset is large enough that this narrow intervention could distinguish support error from remaining within-hazard/rollout calibration error. Any model choice based on it requires another new complete confirmation set; the present 48+48 episodes remain evaluation evidence only.

## Reproducibility and accounting

The sealed [PROTOCOL.md](PROTOCOL.md), [config.json](config.json), [provenance.json](provenance.json), [run.py](run.py), and isolated [collect_evaluation.py](collect_evaluation.py) specify all seeds, features, timing, and accounting. Final head checkpoints, optimization records, paired tuples, common one-step position samples, probability arrays, native records, and all six complete rollout arrays are saved in this directory. [mechanism_contrasts.json](mechanism_contrasts.json) is generated from saved rollouts by [analyze_saved.py](analyze_saved.py) with zero model/native calls.

[verify_saved.py](verify_saved.py) independently reconstructs both 2,000-update Adam runs, all actual/sampled-position head probabilities, onset metrics and episode bootstraps, native rewards/failure labels/returns, all six joint trajectories, primary trajectory endpoints, hashes, and budgets. It passes and writes [verification.json](verification.json), using zero position-model, actor, nominal, or environment calls.

Execution used exactly **2,151,424 position-model successors** under the 3,000,000 cap and **14,400 native steps** under the 16,000 cap. The two heads received 4,000 total updates and 1,024,000 labeled-row presentations. There were zero position-model, actor, critic, or nominal updates, no sweep, no evaluation tuning, and no failed or repeated execution. Historical artifacts and production modules remain unchanged. The experiment itself made no commit or push; repository delivery was performed only after the user's later explicit request.
