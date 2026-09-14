# Controlled supervised PointMaze transition repair

Sealed on 2026-09-14 before new collection, training, validation, or final evaluation. The experiment is pinned to the completed matched audit at commit `88ec819cd4dde584976ae71a0aa022f6aeadc2b4` by dependency hashes, not by requiring the working HEAD to remain equal. Historical artifacts and production modules are read-only. Nothing will be committed or pushed.

## Fixed semantics and factorial arms

Every arm uses the disclosed native hazardous-landing support `3 <= next_x < 6 and 3 <= next_y < 4`; onset probability is structurally zero elsewhere. This additional environment knowledge applies equally to old-head controls and repaired heads. While alive, each arm generates position first and then samples onset. Fatal incoming motion is retained. Failure is irreversible; after onset, later XY is frozen while F4 shifts and reward is zero. Stationarity alone never implies failure. The actor and historical state-goal nominal policy are frozen and receive neither hidden swamp bits, labels, nor internal failure mode.

For each initialization seed 0 and 1, cross old/repaired position with old/repaired onset, yielding eight arms. The old position checkpoints are the response-gate candidates with SHA-256 `c31e8383edae3ca20fd54d3966d480f6c8687dc89c9db1f7456016a476cb0183` and `7169a710fe0adb0999a790c9757d4f9f4d125f567aa57e06f09bd3dcbbae243d`. Old failure heads have SHA-256 `06b6a5a7815f5720c9adba7c0eae5d53239c5366b73c731bf5ad98fdfdbf1579` and `b9a13cc6fc06f52874e105a2eba5319d150312a2fd9afb87c57770e428abe661`. Scaling remains fixed. The historical unmasked rollouts are inspected evidence, not a ninth arm.

## Complete-episode data splits

Collect 128 training prefix episodes and 32 validation prefix episodes, each alternating expert and frozen-actor main prefixes, plus 64 final actor-prefix episodes. All use horizon 50, active probability 0.3, the original teacher's 0.05 episode-level force-safe law, persistent memo/route state, nonzero-action-only Gaussian noise at 0.15, clipping, and independent declared RNGs. Environment, teacher, actor, and random-action seeds are fixed in `config.json`; split seed ranges are disjoint.

At every context, query the shadow teacher on the exact snapshot, then generate one frozen-actor action. Deep-copy that snapshot and evaluate shadow advice, frozen actor, negative advice, and one uniform random action with paired native randomness. Advance the main environment with the declared expert or actor action and require exact identity with its saved branch. Retain all surviving, fatal, rewarding, and already-dead outcomes, recording `(episode,time,prefix_source,s,xb,xq,y,dead_before,dead_after,onset,reward,query_type)`. Hidden bits are audit-only. Shadow advice is generated on the same hidden situation and is never replaced by an independent nominal draw.

The repeatedly inspected 48 matched-audit episodes and historical validation episodes are excluded from training, validation, and final collection. Final collection begins only after both repaired checkpoints are selected and frozen; its outcomes cannot affect settings.

## Alive-position repair

Train only on new training rows with `dead_before=false`, including fatal incoming transitions. Exact `xq==xb` rows form the diagonal term and all other alive rows the off-diagonal term. No survivor selection is applied. Already-dead motion is handled only by the fixed absorbing branch.

Initialize from each 69-parameter candidate. All 16 diagonal distribution offsets may update from diagonal and off-diagonal loss. All 32 normalized-response coordinates and 21 query-independent response-gate weights may update from off-diagonal loss; their structurally zero diagonal gradient is explicitly set to zero. The underlying diagonal network, projection geometry, actor, and nominal policy remain frozen.

Use the full emitted-XY energy-score U-statistic with eight samples, separate 128-row diagonal/off-diagonal batches, equal term weight, eight Gaussian directions and 16 antithetic signed candidates. Use 240 fixed updates—not the prior 120—under the existing stable groupwise perturbation scales, Adam learning rates, and update caps in `config.json`, with Adam moments reset to zero. Validate at steps 0,20,...,240 using 32 samples per alive validation row and fixed common keys. Select the earliest checkpoint minimizing the equally weighted validation diagonal/off-diagonal mean energy score. Hazard entry/residence/exit and horizontal displacement are validation diagnostics, never alternative checkpoint selectors.

## Failure-onset repair

Fine-tune all 1,569 parameters of each existing 14-32-32-1 tanh head from its corresponding old checkpoint; this is not merely a new alive filter. Retain the existing inputs: standardized observed F4, raw `xb`, raw `xq`, and next-XY displacement. Adam moments reset to zero. Uniformly sample 512 alive training rows for each of 1,500 fixed updates and minimize supported binary cross-entropy on the actual successor and onset label. Because nonhazard probability is structurally zero and native nonhazard onset is impossible, such rows have exact zero loss; sampling remains uniform so event prevalence is not silently changed.

Validate at steps 0,50,...,1500. Select the earliest checkpoint minimizing the equal-weight mean hazardous-landing BCE over nonempty current-position groups: left approach `[2,3)x[3,4)`, currently inside hazard, and all other positions. This prevents aggregate dominance from hiding opposing approach/inside errors. Track overall and group Brier, log loss, calibration bias, and impossible outside-support probability. No classification threshold, global empirical death rate, probability multiplier, or return objective is used.

## Frozen final evaluation

On every alive row of the untouched 64 actor-prefix final episodes, evaluate old and repaired heads at actual successors; old and repaired positions with 64 common-key samples; and all four matched joint crosses per seed. Position draws are predictive integration and remain clustered within episode. Joint forecasts are grouped only by pre-transition inputs; actual landing hazard is not used to claim conditional calibration of a marginalized forecast. Outside-support onset must be exactly zero.

From each native initial state, generate 64 complete 50-step paths for all eight arms with the fixed goal, historical nominal, actor, discount 0.95, and common keys. Report reward occurrence, failure, onset timing, survival without reward, return conditional on reward, mean discounted return, first hazardous landing, and at-risk exposure. Reward occurrence remains distinct from tighter success radii. Native later exposure counts only alive-before hazardous opportunities including fatal onset; model exposure likewise uses only `failed_before=false`. No death-censored native total is compared to an uncensored model total.

Use 2,000 whole-episode bootstrap replicates over 64 final episodes. Report head effects at fixed position, position effects at fixed head, combined effects, and factorial interaction, with seed consistency. Model paths and four same-context queries never become independent native evidence.

## Budgets and acceptance

Charge calls before execution and resume saved stages after interruption. Native collection is capped at exactly 56,000 steps: 32,000 train, 8,000 validation, 16,000 final. Position successors are capped at 26,100,000; worst-case planned use is 25,972,736 including training, validation, final matched evaluation, eight complete-rollout arms, and checks. Training is capped at 480 position plus 3,000 head updates. Complete model paths are capped at 32,768. There is no sweep or schedule extension.

Practical acceptance is frozen in `config.json`. Head repair must improve final actual-successor Brier and absolute calibration bias overall and on hazardous landings in both seeds. Position repair must improve diagonal/off-diagonal energy score, current-hazard residence error, and horizontal-displacement residual in both seeds without worsening hazard-entry absolute error by more than three points. Combined arms must improve absolute native reward-occurrence and failure gaps versus supported old/old in both seeds while retaining component improvements; mean return alone cannot pass.

A failed restricted repair does not imply the whole ETT family is incapable. Successful matched checks with bad autonomous rollouts leave generated-state and nominal-mixture mismatch unresolved; the later valid nominal audit is a forecast marginalized over `xb~b` at fixed real state and actor action, never one independently attached nominal action as a labeled conditional tuple. Any proposal preserves irreversible absorption. This supervised reference does not establish observational identification, pessimistic optimization validity, or actor benefit.
