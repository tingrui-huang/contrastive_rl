# Step 3a with the v2 one-step ETT (atom + stationary gate + visible-history onset features)

Same sealed protocol and thresholds as `../ett_rollout/` (manifest
`manifest.json`; the v2 models `../ett_one_step_v2/`, preflight G1-G7 all
passed).  `REPORT.md` / `report.json` are the full tables; `roll_*.pkl`
(node3) hold the per-anchor paths.

## Result: C passes pooled, fails strata; A fails on deaths (as v1)

Advice C (simulator teacher walked along the model path, latches
re-derived; 6,000 held-out anchors, one path each):

| | sim | model |
|---|---|---|
| death / reach / timeout (all anchors) | 0.294 / 0.615 / 0.091 | 0.311 / 0.624 / 0.065 |
| KS death time / death x / reach time (all) | | 0.027 / 0.072 / 0.028 |
| pre_zone1 timeout / reach | 0.178 / 0.319 | 0.049 / 0.405 |
| post_zone2 timeout / reach | 0.042 / 0.958 | 0.089 / 0.906 |
| zone2 KS death time / death x | | 0.217 / 0.167 |
| start / between KS death x | | 0.157 / 0.166 |

Pooled: every gate passes (v1 failed on the stall drift and the death
timing; the atom and the history features fixed the pooled picture).
Strata: pre_zone1 (the corridor before the first mouth) still loses most
of the simulator's timeouts (0.178 -> 0.049) and over-reaches; post_zone2
over-times-out; zone2 death timing (KS 0.22) and the death-x
distributions in start / between / zone2 (0.16-0.17) exceed the 0.15
threshold.

Per fold (`perfold` probe): folds 0 / 2 produce almost no pre-mouth
stalls (pre_zone1 timeouts 0.006 / 0.021 vs 0.197 / 0.193 in the sim);
fold 1, which early-stopped at 4k updates on the stationary BCE, over-
stalls (0.153; post_zone2 timeouts 0.236).  The pooled pass is the
average of one under-stalling and one over-stalling behaviour.

Mechanism (settling probe on the simulator's stalled tails, fold-0
timeouts): after the agent's torque becomes small the simulator
decelerates slowly (median |delta s| 0.08 at the switch, 0.02 after 10
steps, 0.006 after 40), whereas the v2 model from the same pre-stall states keeps
|delta s| ~0.15 with the gate probability near 0, so the stationary
gate never fires in closed loop and the policy re-accelerates after ~40
steps.  Cause: the standardised MSE is dominated by the large-
displacement rows; the small-motion rows are fitted coarsely.

Advice A (memoryless nominal a_b ~ P(a_b | s), MDN k = 5, 4 paths per
anchor): death 0.294 -> 0.032, reach 0.615 -> 0.860, AUROC 0.64 -- the
memoryless nominal cannot produce the in-band holds, as with v1.

## Consequence

v3 (disclosed revision, `fit_v6_ett_one_step_v2.py --v3`, `../ett_one_step_v3/`,
`../ett_rollout_v3/`): the motion regression trains on ALL rows
(stationary included) with an added relative error term
||pred - d||^2 / (||d||^2 + 0.05^2), so small-motion rows are fitted at
their own scale; the model-selection score drops the stationary BCE
(which stopped fold 1 at 4k); the atom, gate, onset head, optimiser,
budget, batches and cross-fitting are unchanged.  v3 preflight: all
gates passed on all three folds.  The v3 full-length C / A rollouts are
the next entry (`../ett_rollout_v3/`).
