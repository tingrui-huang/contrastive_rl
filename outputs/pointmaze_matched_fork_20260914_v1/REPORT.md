# PointMaze matched-fork diagnostic

## Answer

The beneficial route decision fails at the saved C1 critic preference, before an actor-update test is warranted. From the same 16 coherent alive C1 roots, forcing one downward action and then resuming the same frozen C1 actor increased native whole-episode discounted return by **6.859 [6.443, 7.284]** relative to forcing right. Strict success increased by **70.0 [67.0, 73.2] percentage points**, failure decreased by **-70.0 [-73.2, -67.0] percentage points**, and lower-route entry increased by **97.2 [95.6, 98.4] percentage points**.

This is already a one-step route-initiation result: after the forced down step, the unchanged actor entered the lower route in 97.4% of paths. The natural frozen actor nevertheless made a feasible down-cell exit on only **7.2% [3.3%, 12.1%]** of sealed action draws, versus a right-cell exit on **88.7% [83.5%, 92.8%]**.

The frozen oracle-motion/repaired-head-1 model reproduced the direction: model down-minus-right return was **5.919 [5.412, 6.401]**, with strict-success difference **62.4 [58.7, 65.9] percentage points**. It underestimated the native preference by -0.940 [-1.578, -0.351] because it slightly undervalued down and more materially overvalued right. In particular, model-minus-native return was -0.078 [-0.130, -0.022] for down and 0.863 [0.288, 1.492] for right. This is a magnitude mismatch, not an incorrect model ranking.

The saved C1 critic ranked the exact same state/action/task-goal tuples in the opposite direction on every root: down-minus-right logit **-0.647 [-0.856, -0.447]**, with 0/16 roots preferring down. These logits are ranking diagnostics, not calibrated returns.

## Matched outcomes

- Native down: return 10.213 [10.036, 10.351]; strict success 96.6%; failure 3.4%; lower-route entry 97.4%.
- Native right: return 3.354 [2.915, 3.776]; strict success 26.6%; failure 73.4%; lower-route entry 0.2%.
- Model down: return 10.135 [9.947, 10.291]; strict success 95.7%; failure 4.3%.
- Model right: return 4.217 [3.758, 4.723]; strict success 33.3%; failure 66.7%.

Each arm has 64 predictive continuations per root. Those paths integrate stochasticity within a root; intervals first average them and then bootstrap the 16 distinct saved native episodes. Conditional predictive Monte Carlo SE for the native return contrast was 0.185, separate from across-root uncertainty.

## Branch-C replay and objective audit

The conditional audit reconstructed all 256,000 actual C1 learner rows, including the saved 90/10 source replacement and exact generated-path indices.

- Only **371 rows (0.145%)** combined a matched fork state, recorded downward action, and a future achieved XY inside the task reward region. B1 had 388 such rows, so oracle augmentation changed the count by -17 rather than increasing it.
- Requiring the observed next state to actually enter lower exit cell `(1,2)` leaves 110 C1 rows versus 116 B1 rows.
- C1's generated 10% contributed only 23 matched downward/task-region positive rows; the remaining 348 were offline.
- No learner row used the canonical stationary task F4 exactly as its positive goal. The critic instead received a particular achieved future F4 identity. Native return uses radius-2 reward and strict radius-0.5 success; neither scalar label enters this MC NCE loss.
- The 256x256 sigmoid-NCE loss includes every sample-pair cell. Each row has one logged future-state positive and its goal is off-diagonal for the other 255 anchors. It never receives the matched 64-continuation final-C1 action-value ordering measured here.

This supports sparse, non-improving route-relevant coverage and a different continuation target. Exact-F4 identity versus task-region success is a real semantic difference but is not isolated as the cause. Representation capacity versus optimization also remains unresolved: the checkpoint changed and losses were finite, but there was no controlled fit test in this diagnostic.

## Initialization and validity

Episodes 0–15 were selected in ascending saved C1 order; each first qualifying context occurred at time 1 with 49 steps remaining. Every root was regenerated from reset with its saved actions, reproducing physical XY, full F4, time, reward and alive state exactly. Candidate actions `[0,-1]` and `[1,0]` were fixed by geometry and entered cells `(1,2)` and `(2,3)` under zero noise for every root.

The naturally reached current swamp bits governed the forced step and fresh bits were installed after every subsequent native step. The explicit scheduler reproduced native action-noise and bit-resampling order. Hidden fields were saved only for native evaluation/initialization audit; they never entered actor observations or the learned head. Actor Gaussian innovations and motion noise were paired across actions and backends; native bits and model onset randomness were kept semantically separate.

The primary used exactly 4,096 paths and 100,352 transitions per backend in 50.8 seconds. Initialization checks and root regeneration used 66 native steps. Training updates, new training episodes, checkpoint searches, prefix paths, and actor-probe updates were all zero. Branch C prohibited the other conditional branches.

## Required interpretation

1. **Does initiating the detour benefit this actor?** Yes for these roots: one forced downward step produces a large, clearly positive native return and success advantage, and the unchanged actor usually completes the descent.
2. **Does the frozen model reproduce the native preference?** Yes directionally. It still overvalues the right/shortcut continuation, so exact magnitude agreement is not established.
3. **Does the saved critic provide the task-useful preference?** No. It prefers right on all 16 matched roots despite both native and model continuation returns favoring down.
4. **Does the actor update move toward it?** Unresolved and deliberately untested. The sealed branch rules allow an actor probe only when the critic already has the correct preference; it does not.
5. **Which repair next?** Run one critic-only matched task-preference intervention. On a separately preregistered coherent-alive fork training split, add a pairwise task-goal ranking term labeled by frozen-model multi-continuation down-versus-right returns, while holding the actor, F4 representation, base NCE batches, BC, nominal, head and motion fixed. First require the critic ranking to flip on held-out coherent roots; only then inspect the actor gradient and new complete native episodes.

The current 16 roots and the prior 24 development episodes are exploratory evidence. They cannot confirm that the repair improves complete native success over A and matched B. Also unresolved are whether canonical task-region goal handling alone would fix the critic, whether the representation can fit both exact-F4 reachability and task preference, and whether a corrected continuous-action gradient changes behavior. Confirmation requires new complete episodes.

## Artifacts

`preregistration.json`, `PROTOCOL.md`, `root_selection.*`, and `initialization_checks.json` seal design and initialization. `primary_native_traces.npz` and `primary_model_traces.npz` contain complete traces and stochastic inputs. `results.json`, `branch_c_results.json`, `decision.json`, `matched_trajectories.png`, and `root_preferences.png` contain numerical results and plots. Reproduction and verification code is `run.py`, `analyze.py`, `branch_c_audit.py`, `finalize_branch_c.py`, `verify_saved.py`, and `verify_complete.py`.

This does not validate learned ETT motion, observational identification, or the original worst-case loss. No historical artifact was changed. No commit or push was performed.
