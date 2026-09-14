# Failure-label-supervised persistent-mode diagnostic for PointMaze

Sealed on 2026-09-14 before head training and before collecting or inspecting either new evaluation set. The starting commit is `8bb3c31d227918cdc9f35389011eb3e0985d27da`. Historical artifacts and production modules are read-only; dependency hashes, rather than later working-tree identity, define the execution inputs. Nothing will be committed or pushed by the experiment.

## Scope and explicit additional information

This is an explicitly **failure-label-supervised diagnostic**, not the original observation-only ETT method. For the first time in this line of experiments, the original paired training labels `dead_before` and `dead_after` are authorized as model-training targets. This additional-information result cannot establish unlabeled worst-case optimization, actor benefit, or a solution to observation-only persistent-process learning.

The completed response-gating experiment and the local `pointmaze_return_gap_diagnosis_8bb3c31_v1` report were read first. No equivalent completed persistent-failure/onset-head experiment was found in the repository outputs or the external completed-output directory; the search record is saved. The existing evidence says excess return is primarily excess reward occurrence, and that an observable three-stationary-hazard-step lock would leave about 8.65 of 8.90 predicted return. This experiment therefore models both failure onset and persistence rather than applying a post-hoc stationary-run rule.

## Frozen position models and one head family

Freeze both final gated candidates from `outputs/pointmaze_response_gate_20260914_v1/`, including their position parameters, response gates, original anchor sampler, projection geometry, nominal policy, and actor. The two unchanged position rollouts are the baselines. There is no XY-energy-score, actor, critic, nominal, or position-model training.

Train exactly two seeds of one binary MLP: 14 inputs, two 32-unit tanh layers, and one sigmoid probability. The fixed input order is eight F4 coordinates standardized using means and standard deviations estimated only from original training episodes 0--71 (standard-deviation floor 0.1), raw two-coordinate `xb`, raw two-coordinate `xq`, and raw next-XY displacement from current XY. Weights use independent Glorot-uniform initialization and biases start at zero. Head seeds use initialization seeds 12,140,000 and 12,140,001 and batch streams 13,140,000 and 13,140,001.

Training uses only rows alive before the transition from original episodes 0--71, with `dead_after` as the target and the actual recorded next position in the displacement features. Preserve each saved `(s,xb,xq,y)` tuple; do not construct new expert actions. Optimize unweighted binary cross-entropy for exactly 2,000 Adam updates, batch size 256, learning rate 0.001, beta1 0.9, beta2 0.999, epsilon `1e-8`. Use successful and failed outcomes, no class weights, thresholds, masks, sweep, or checkpoint selection. Save final iterates and optimization records, and measure whether final training predictions vary.

Because head-training inputs do not depend on a position checkpoint, cross both trained heads with both frozen position candidates. This produces four joint arms: `p0_h0`, `p0_h1`, `p1_h0`, and `p1_h1`, plus unchanged baselines `baseline_p0` and `baseline_p1`.

## Joint transition and native timing

Every generated path starts with internal mode `d=0`. At an alive step, first sample the next observed state from the frozen gated position model using the usual `s`, `xb`, and actor action `xq`; then evaluate the head on `s`, `xb`, `xq`, and that sampled next XY; then sample onset and set the next mode. The landing movement is retained on a fatal-onset step. At every later step with `d=1`, keep XY fixed, shift F4 normally by prepending that XY, and keep `d=1` permanently.

Actor and nominal policy inputs remain exactly their original observed F4 and goal; neither receives `d`, hidden swamp bits, simulator flags, labels, or future observations. After generated-state divergence the head receives only the sampled model successor, never a recorded/native successor. Reward is the native fixed-task reward on the generated joint outcome: radius-two visible position reward only while the next mode is alive. No reward editing, penalties, multipliers, native death injection, or unsupported absorbing wrapper is used outside this explicitly modeled joint transition.

The head is allowed to predict failure outside the native hazardous landing set `3 <= x < 6, 3 <= y < 4`; no evaluation-time support mask is applied. Such probability mass and sampled onsets are reported as possible model errors.

The inherited position-only action bound still applies to the sampled position proposal conditional on fixed current mode. The discrete onset draw and persistent joint trajectory change the modeled state and contract, so no full action-Lipschitz claim is made for the joint transition or its future trajectories.

## New onset evaluation

After final head hashes are frozen, collect 48 new complete 50-step paired prefix episodes with base seed 2,114,000, teacher/query offsets 10,000/20,000, active probability 0.3, teacher noise 0.15, and force-safe probability 0.05. Even episodes use teacher prefixes and odd episodes use blind prefixes. Query the same native snapshot using diagonal teacher advice, blind forward, reverse teacher, and a uniform action; advance the real prefix with its declared population action. Retain all outcomes and charge 12,000 native steps.

Evaluate each head on alive-before rows twice: first conditioned on the actual recorded successor, and then averaged over 64 sampled successors from each frozen position candidate. Reuse each position sample array across both heads. Position keys use base seed 7,014,000 with fixed checkpoint/batch offsets. Report counts, observed and predicted onset rates, Brier score, log loss, fixed 0.1-wide calibration bins, and expected calibration error for all alive rows, exact diagonal/off-diagonal rows, hazardous/nonhazardous actual landings, and their intersections. Use 2,000 paired whole-episode bootstrap resamples; no classification threshold, accuracy gate, or AUC decision is used.

## New fixed-policy trajectory evaluation

Collect another 48 fresh native 50-step episodes from the same frozen actor, using environment seeds beginning 3,114,000 and per-step actor keys beginning 4,114,000. Save F4 observations, actions, rewards, `dead_before`, `dead_after`, and exact onset for evaluation only. Charge 2,400 native steps.

For each of the two baselines and four crossed joint arms, generate 64 paths per initial observation for 50 steps using common base key 5,114,000, the original fixed goal, discount 0.95, and actor/nominal policies on each model's own observed state. Report reward-occurrence probability, failure probability and conditional timing, survival without reward, return conditional on reward, mean discounted return and native gap, expected failure probability and sampled onset outside hazardous cells, and persistence. Use paired whole-episode bootstrap uncertainty. Lower return counts as improvement only when reward occurrence and survival/failure behavior are credible rather than destroyed by false failures.

## Contract checks and accounting

Check fatal-onset movement, monotone/absorbing mode, absorbing F4 shift, actor/nominal input isolation, absence of native labels from generated rollouts, exact disabled-mechanism baseline reproduction, inherited position geometry, common pre-onset trajectories, and support errors without masking. Conditioning on supplied `d=1` and freezing is structural only, not recognition evidence.

Planned position-model successor accounting is 1,024 contract-check outputs + 1,228,800 one-step sampled successors + 921,600 complete-rollout successors = **2,151,424**, below the hard cap of 3,000,000. Planned native accounting is 12,000 paired branch/prefix steps + 2,400 frozen-actor steps = **14,400**, below 16,000. Head-only computations are recorded separately and make no position-model successor calls. Charge before calls and do not silently extend either cap.

Final settings only are evaluated; neither new evaluation set may tune this experiment. Any follow-up chosen from these results requires new complete confirmation episodes. Save code, configurations, hashes, checkpoints, samples, metrics, report, and independent saved-artifact verification in this fresh directory.
