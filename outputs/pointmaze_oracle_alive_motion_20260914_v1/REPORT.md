# Oracle alive-motion substitution diagnostic for PointMaze

## Answer

Exact native alive motion removes most of the remaining pre-failure trajectory discrepancy. First hazardous landing falls from 98.10%/98.95% in the saved learned-position arms to 89.43% in both oracle arms, near the native 85.94%. At-risk hazardous opportunities fall from 2.88/3.00 to 2.30/2.37, near native 2.25. Entry time and location also move toward native.

The downstream trajectory is plausible but not confirmed sufficient. Oracle failure is 66.31%/65.11% versus native 70.31%, and reward occurrence is 34.13%/35.50% versus native 29.69%. Every oracle-native episode-bootstrap interval includes zero, but 64 reused native episodes cannot establish that these 4-6 percentage-point gaps are negligible. Mean returns, 3.75/3.90 versus native 3.14, have the same limitation.

The predeclared interpretation is therefore **D, with a decisive primary localization**: correct motion fixes the known first-entry/exposure problem, while native uncertainty prevents choosing between A (the remaining fixed components are sufficient) and B (nominal-weighted onset remains wrong). This is not C because the oracle implementation passed exact equivalence and the primary outcomes became compatible with native.

This is retrospective, simulator-assisted PointMaze evidence. It is not a learned ETT result, observational-identification result, pessimistic-objective validation, actor-improvement result, or claim about AntMaze.

## What was run

The diagnostic is pinned to the completed repair at `813c3db1242d7f30c6b2ae20b02a1770382f4338`. It reuses its 64 native final actor episodes, repaired/repaired seed-0 and seed-1 rollouts, bootstrap weights, frozen repaired heads, historical nominal, actor, goal, and reset roots. No baseline was regenerated.

Only two new arms were generated:

- exact native alive motion plus repaired failure head 0;
- exact native alive motion plus repaired failure head 1.

Each has 64 paths for each of 64 roots, horizon 50: exactly 8,192 new paths and 409,600 oracle-motion slots total. There were zero training updates, zero new training episodes, and zero new complete native evaluation episodes.

At each generated step, nominal advice and the actor use that path's own current F4 observation. The learned anchor, residual response, gate, atom, and proposal projection are absent. Motion adds explicit Gaussian action noise with standard deviation 0.01, clips to `[-1,1]`, and performs ten float64 `dt=0.1` substeps with X then Y collision checks. The resulting position is emitted to F4 as float32. Advice remains an input to the repaired head but not to native motion.

Native hidden bits, native death, termination, and native reward are never used in oracle rollout outcomes. The repaired head is evaluated on observable F4, `xb`, actor action, and emitted displacement. Its probability is zero outside `3 <= x < 6, 3 <= y < 4`; onset uses the unchanged independent uniform stream. Fatal incoming movement is retained, failure is irreversible, later XY freezes while F4 shifts, and failed reward is zero. Reward on surviving paths uses native physical-position goal distance.

## Oracle equivalence and contracts

Before rollouts, 34 separately budgeted native `step` calls compared the standalone movement map with `TwoRouteSwampWindyF4Env.step`: 16 paired one-step calls covered free motion, walls, a corner, action clipping, hazard entry/exit, fatal incoming movement, and the outer boundary; 18 calls covered two multi-step sequences. Starting float64 position, F4, action, and motion noise were identical.

Physical XY and emitted F4 agreed bit-for-bit. Switching from all-clear to all-active hidden bits never changed incoming motion, including fatal landings. Native death and reward were recorded only as discarded audit outputs. Across the two oracle arms, motion noise and onset uniforms were identical, and every corresponding path had identical first-entry occurrence, time, and location. State, advice, action, and movement also matched through the first landing.

Across all oracle slots there were zero physical-versus-emitted hazard classifications and zero physical-versus-emitted reward classifications that disagreed. Float64 physical and float32 emitted coordinates differed by at most `4.77e-7`, without changing either decision. Every onset transition moved by more than the stationary threshold; mean fatal incoming displacement was 0.968 for head 0 and 0.961 for head 1. All support, legality, F4, reward, fatal-motion, and absorption checks passed.

## Primary result: first hazardous landing

Uncertainty is a paired 2,000-replicate bootstrap over the original 64 native episode/root units. The 64 paths per root are predictive integration, not independent native samples.

| Arm | Entry probability | Conditional entry time | Mean entry X | Mean entry Y |
|---|---:|---:|---:|---:|
| Native | 0.8594 [0.7656, 0.9375] | 3.255 [2.400, 4.661] | 3.4569 [3.4076, 3.5082] | 3.4103 [3.3231, 3.4943] |
| Saved learned s0 | 0.9810 [0.9768, 0.9849] | 2.913 [2.851, 2.973] | 3.3828 [3.3757, 3.3901] | 3.5281 [3.5221, 3.5344] |
| Oracle, head 0 | 0.8943 [0.8840, 0.9041] | 3.135 [3.044, 3.226] | 3.4384 [3.4317, 3.4454] | 3.4686 [3.4580, 3.4800] |
| Saved learned s1 | 0.9895 [0.9861, 0.9927] | 3.110 [3.007, 3.213] | 3.3978 [3.3901, 3.4060] | 3.5226 [3.5171, 3.5282] |
| Oracle, head 1 | 0.8943 [0.8840, 0.9041] | 3.135 [3.044, 3.226] | 3.4384 [3.4317, 3.4454] | 3.4686 [3.4580, 3.4800] |

The two oracle rows are exactly identical for primary outcomes, as required: a head cannot affect motion before the first supported onset opportunity.

Oracle-minus-native differences were +0.0349 [-0.0425, +0.1265] for entry probability, -0.119 [-1.494, +0.741] steps, -0.0185 [-0.0696, +0.0339] in X, and +0.0583 [-0.0304, +0.1481] in Y. All include zero.

For seed 0, absolute native errors in probability/time/X/Y changed from 0.1216/0.3412/0.0741/0.1179 under learned motion to 0.0349/0.1191/0.0185/0.0583 under oracle motion. For seed 1 they changed from 0.1301/0.1450/0.0590/0.1124 to the same oracle values. The probability absolute-error reductions were -0.0867 [-0.0962, -0.0005] and -0.0952 [-0.1052, -0.0093]. Location and time point errors all decreased, although several absolute-error-change intervals cross zero because the native reference is small.

This establishes that the old 98-99% entry rate is not caused by either repaired head. It arises before the first possible failure, inside the learned-motion/closed-loop generation block. The oracle does not separate position-model capacity, optimization, nominal dependence inside the old proposal, actor response to accumulated state error, or their interactions.

## Secondary trajectory behavior

| Arm | Failure | Onset time | Reward occurrence | Survive/no reward | Return given reward | Mean return | At-risk opportunities |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 0.7031 | 4.911 | 0.2969 | 0.0000 | 10.564 | 3.136 | 2.250 |
| Saved learned s0 | 0.7153 | 4.891 | 0.2922 | 0.0012 | 11.492 | 3.358 | 2.878 |
| Oracle, head 0 | 0.6631 | 4.640 | 0.3413 | 0.0002 | 10.982 | 3.748 | 2.298 |
| Saved learned s1 | 0.7112 | 5.397 | 0.2891 | 0.0090 | 11.689 | 3.379 | 2.996 |
| Oracle, head 1 | 0.6511 | 4.785 | 0.3550 | 0.0002 | 10.992 | 3.902 | 2.370 |

Relative to the corresponding learned arms, oracle motion reduced failure by -0.0522 [-0.0674, -0.0371] and -0.0601 [-0.0762, -0.0442], increased reward by +0.0491 [+0.0330, +0.0645] and +0.0659 [+0.0496, +0.0828], and reduced exposure by -0.5808 [-0.6487, -0.5112] and -0.6262 [-0.6926, -0.5569]. These are precise model-arm contrasts under common roots and streams.

Against native, however, uncertainty is governed by only 64 episodes. Head 0 differences were -0.0400 [-0.1514, +0.0725] for failure, +0.0444 [-0.0688, +0.1558] for reward, +0.612 [-0.684, +1.834] for mean return, and +0.0476 [-0.4652, +0.5015] for exposure. Head 1 differences were -0.0520 [-0.1621, +0.0591], +0.0581 [-0.0544, +0.1694], +0.766 [-0.519, +1.977], and +0.1196 [-0.3865, +0.5638]. None establishes a native discrepancy, but none is a narrow equivalence result.

The secondary pattern is mixed rather than a hidden all-metrics win. Exposure, survival-without-reward, and return conditional on reward become substantially closer in point estimate; seed-1 onset time also improves. Failure and reward point errors increase to about 4-6 points, and mean return moves farther above native. The repaired heads differ after entry, but native sampling uncertainty is too wide to determine whether their lower failure/higher reward represents residual nominal-weighted onset misspecification.

## Conclusion and next action

The experiment answers the mechanism question in two parts:

1. **Pre-failure motion:** yes. Correct alive motion makes first-entry behavior credible and removes the dominant excess-exposure error. Further frozen-anchor checkpoint tweaking is not the highest-value response to the 98-99% entry problem.
2. **Whole trajectory:** unresolved. The existing head, nominal, actor, and absorption mechanism produce plausible oracle trajectories, but this retrospective 64-root comparison cannot confirm that their remaining 4-6-point failure/reward gaps are practically negligible.

The one recommended next action is a preregistered fresh complete-episode actor confirmation study, with its episode count chosen before collection to resolve approximately five-percentage-point reward and failure gaps. Keep this verified oracle implementation, both repaired heads, nominal, actor, support, and absorption frozen. Do not start another checkpoint search or learned-motion architecture experiment until that confirmation distinguishes A from B.

## Artifacts

- `PROTOCOL.md`, `config.json`, `provenance.json`, and `equivalent_experiment_search.json`: sealed design and hashes.
- `oracle_motion.py` and `movement_equivalence.json`: exact helper and the 34-call native equivalence record.
- `oracle_rollout_s0.npz` and `oracle_rollout_s1.npz`: complete reusable oracle arrays, including float64 physical states and explicit noise.
- `metrics.json`: all estimates, intervals, comparisons, absolute native errors, and changes in absolute error.
- `contract_checks.json`, `ledger.json`, `completion.json`, and `decision.json`: semantics, accounting, completion, and interpretation.
- `verify_saved.py` and `verification.json`: independent saved-array reconstruction and verification.

No historical artifact or production module was changed. No commit or push was performed.
