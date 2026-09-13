# Shared synthetic anchors: short versus long future goals

Prepared before generation and learning on feature/pointmaze-causal-transition, starting commit 6c9ccb70d2e9530a9b83d719846f9db77e666371. Preserve all prior experiments. This is one bounded intervention on the available future-goal window, not a claim that truncation explained the earlier null result or that longer model trajectories are valid.

## Fixed models and learner

Restore the exact initial actor, critic, target critic and full Adam state from the historical alpha0_seed0 final checkpoint at step 150000 used in the preceding experiment. All three arms and both seeds start with the same parameters; only learner RNG differs between seeds. The learner remains crl.losses.build_learner MC NCE: batch 256, 256x256 networks, representation 64, Adam 3e-4 eps 1e-7, gamma .95, BC .5, random goals .5, entropy zero, failure-negative alpha=0, no bank. Continue both actor and critic for 2000 updates. No TD target, SAC replacement, new return regressor, new scorer or ETT update.

B and C use the SAME frozen adversarial ETT, with learner seed 0/1 paired to model seed 0/1. This couples learner/model variability, not two independently identified factors. The prior action-invariant control is not used. Freeze the nominal state_goal K5 behavior policy, its original normalization, the entire diagonal law, L=1, convex projection rules and 32 adversarial coefficients. Verify all prior weight/config hashes; never substitute missing models. Initial actor and critic were previously trained on all source episodes; this experiment does not remove that upstream reuse.

## Shared context schedule and generation

Use all original behavior sources from the established 5940 training episodes: exclude the first 660 IDs of seed-0 permutation of 6600 episodes. No old native-continuation evaluation arrays, hidden fields, rewards or failure labels enter training. Context selection reads offline obs/actions only, with uniform training-episode draws and uniform times 0..47, following the previous protocol.

Before training, precompute 20 batches of 256 contexts per model seed. The behavior actor for ALL these rollouts is the unchanged initial stochastic actor, not an evolving learner. Keep commanded goal tiled (8.5,3.5). Every valid step draws nominal x_prime, samples an execution action from that initial actor, then samples the frozen ETT. x_prime is never selected, reweighted or exposed to the learner.

For original time t, generate exactly H=50-t transitions. Group all contexts by H and call the existing generator at that exact length, once per length and model seed. No transition is queried beyond the task horizon. Scatter into storage with NaN tails after H; these tails are never valid observations or targets. Save auxiliary draws and model diagnostics separately. The short view is sliced from these exact same long trajectories; there is no second rollout. The learner receives only shared states, actions and H.

Context RNG base 34000000; generation keys 35000000+1000*model_seed+H. The complete generation schedule and fresh evaluation reset IDs are saved before outcomes. Precomputed batch k is presented to B/C during updates 100*k..100*k+99; only that batch is eligible for anchors. Fixed-data replay is intentionally not refreshed with evolving actor actions.

## Sampling intervention and matching

For B/C, independently choose a trajectory uniformly among 256, then i uniformly among {0,1,2}, irrespective of H. Their state, execution action and immediate successor are identical. B chooses j in i+1..3; C chooses j in i+1..H. The probability is proportional to .95^(j-i). Use an audited inverse-CDF sampler with one uniform per future draw. Anchor and future RNG streams are separate (bases 36000000 and 37000000 plus learner seed). C/B use the same uniforms under their respective CDFs; changing the window cannot advance the anchor RNG. Existing replay lengths/anchor-cut options cannot express both constraints simultaneously, so this small sampler is explicit.

Positive goal is the full eight-coordinate achieved F4 at j from that trajectory. Commanded goal is not relabeled as achieved unless the ordinary sample selects that achieved state. Only i=0,1,2 can be anchors; later states are goals only. Preserve history order (newest XY first) and exact shifts. No terminal, death or continuation-reward labels are invented. MC-unused reward/discount/next_action slots retain NaN sentinels, as in the previous checked integration.

Draw a full matched offline batch of 256 using the original replay sampler (RNG 39000000+seed). Replace the last n rows in B/C using the existing floor((u+1)*256/10)-floor(u*256/10) schedule; exactly 51200 synthetic anchors among 512000 rows per run. Then use matched permutation RNG 38000000+seed. A uses the same full offline draws and permutations without replacement. Save matching hashes of mixed states/actions/successors, offline indices, source masks, permutations and synthetic anchor IDs. Before full learning, audit all 2000 batches for both seeds and exact B/C equality. Future-index histograms and distribution tests are required, not merely matching final hashes.

Changing sampled goals also changes in-batch negative goals, actor relabeled/random-goal inputs and goal conditioning of the BC term. All these downstream effects are part of this intervention. BC on synthetic actions remains enabled; no claim is made to isolate only the positive critic term. Raw B/C training losses integrate different goal distributions and are not directly comparable as identical-target performance.

## Evaluation, diagnostics and uncertainty

Save pre-run source/checkpoint hashes, configs and seeds. Smoke: five updates each for B/C seed 0, from unchanged initial state, using already shared data and no new model interaction. Save only fixed final main checkpoints after 2000 updates. All source/model identities and actual actor/critic parameter changes must verify.

Evaluate initial actor once and each of six final actors on 128 native episodes with common fresh reset seeds 31000000..31000127. Use the previously fixed 128-row inference shape for both main and small replays to avoid the prior batch-size roundoff issue. Native deployment is deterministic tanh(loc), natural per-cell swamp p=.3, fixed goal, original 50-step external horizon, native rewards and gamma .95. Success is minimum true XY distance <.5; task reward uses radius 2; report both. Absorbing death retains done=False through timeout. Hidden env.dead is used only in separate evaluation diagnostics, not selection, replay or targets. Initial RNG seeds are paired; policy-dependent branches need not consume identical future randomness.

Also evaluate initial/A/B/C in each seed's frozen adversarial model for 128 paths, deterministic actors, horizon 50, key 32000000. These eight sets are MODEL outcomes, separate from native results. Primary paired comparisons are C-B and C-A; also report C-initial. Use 2000 paired whole-episode bootstrap resamples with seed 40000000, 95% percentile intervals. Keep per-seed results; no pooling as independent 256 runs, no multiplicity correction. Zero empirical differences yield degenerate bootstrap intervals and do not prove population equivalence.

From saved data only, report actual future j and lag distributions, C j>3 fraction, goal proximity in XY/full F4, complete valid model-continuation returns, stationary-history/anchor-atom motion, outer box projection and sampled segment geometry, exact F4 shifts, action differences and routes. Stationarity is not a death label. Grouped diagnostic denominators and changing horizon lengths must remain explicit. Inspect first four reset IDs as representatives, never select by outcomes. Native selected trajectories and final actor actions must replay; replay generation groups H=3,26,50 for both model seeds and initial/final-C model evaluation paths as additional checks.

## Hard bounds and stopping

Primary generation cap: 2*20*256*50 = 512000 model transitions; actual sum uses H. Evaluation: 8*128*50 = 51200. Total model cap 650000 includes generation, evaluation, up to 80000 generation/evaluation replay steps and 6800 check reserve. The predeclared generation-group replay counts are checked against the reserve before executing; never extend the cap.

Native cap 50000: 7*128*50=44800 main, 7*4*50=1400 selected replay, 3800 focused-check reserve. No outcome-driven extra collection. Tests with synthetic arrays do not count as simulator transitions. Preserve old files and use fresh output paths. Stop on a concrete missing/incompatible checkpoint or invalid integration rather than changing the experiment.

Answer whether longer goals changed learning, improved C over B/A in the simulator, and whether model gains transfer. A null result does not uniquely identify the ETT objective or model family as the bottleneck; a positive result does not identify causal ETT or global worst-case behavior. Longer trajectories may magnify existing errors. Stop after this experiment; no L change, new bank/scorer, larger training budget, alternating update or automatic push.
