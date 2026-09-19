# Oracle-label-trained learned ETT: checks against the oracle branches (cross-fitted, every anchor scored by a model that never saw its episode)

> **Identity correction (2026-09-19, after the user's review).**  The model
> trained here is NOT the project's ETT.  The ETT of the PointMaze pipeline
> (`feature/pointmaze-causal-transition`: `ett/diagonal_transition.py`,
> `outputs/pointmaze_learned_ett_crl_20260915_v1/PROTOCOL.md`,
> `scripts/run_f4_ladder_ett.py`) is a ONE-STEP model `F(s, a_b, a_q) -> s'`
> with a separate irreversible-failure (onset) head, fitted on same-context
> tuples `(s, a_b, a_q, s')` (diagonal `a_q = a_b` from the log, off-diagonal
> from same-context interventions), rolled out step by step with a nominal
> model of the expert advice `a_b ~ P(a_b | s)` and the current agent
> choosing `a_q`, with absorbing tails after onset.  What this directory
> trains is a per-anchor classifier of the discounted future-goal cell,
> `p_gamma(g | s, a_logged)`, under the fixed start agent: no `a_b`, no
> diagonal / off-diagonal structure, no next state, no step-wise rollout, and
> the death head never constrains a trajectory.  It tests only whether a
> learned sampling backend can replace the simulator branches in arm CF; its
> failure is not evidence about the one-step ETT.  Wherever this directory or
> the notes say "learned ETT", read "future-goal marginal model".


`check.json`.  Marginal NLL = exact expectation over the oracle path under the truncated geometric law of -log p(cell); region L1 = sum over the 11 maze regions of |model mass - oracle mass|; death AUC = the outcome head's P(death) ranking the branches that died; masses are anchor-weighted means.  "full" = state + torque input (the model the arm uses); "xy" = position-only reference.

## Held-out likelihood and region agreement by stratum

| stratum | n | model | marginal NLL | region L1 | death AUC | P(death) model / oracle | zone mass model / oracle | far-route mass model / oracle | goal-area mass model / oracle |
|---|---:|---|---:|---:|---:|---|---|---|---|
| all | 53747 | full | 3.678 | 0.497 | 0.871 | 0.314 / 0.291 | 0.179 / 0.175 | 0.040 / 0.040 | 0.269 / 0.273 |
| all | 53747 | xy | 3.630 | 0.500 | 0.863 | 0.270 / 0.291 | 0.181 / 0.175 | 0.040 / 0.040 | 0.273 / 0.273 |
| reset (t = 0) | 196 | full | 4.588 | 0.692 | 0.541 | 0.686 / 0.674 | 0.149 / 0.154 | 0.018 / 0.010 | 0.034 / 0.057 |
| reset (t = 0) | 196 | xy | 4.451 | 0.668 | 0.471 | 0.701 / 0.674 | 0.149 / 0.154 | 0.019 / 0.010 | 0.039 / 0.057 |
| detour-episode anchors | 2764 | full | 5.181 | 0.385 | 0.980 | 0.009 / 0.005 | 0.003 / 0.002 | 0.796 / 0.797 | 0.172 / 0.179 |
| detour-episode anchors | 2764 | xy | 4.766 | 0.352 | 0.993 | 0.017 / 0.005 | 0.004 / 0.002 | 0.799 / 0.797 | 0.173 / 0.179 |
| shortcut-episode anchors | 50983 | full | 3.599 | 0.503 | 0.862 | 0.330 / 0.306 | 0.188 / 0.184 | 0.001 / 0.000 | 0.274 / 0.278 |
| shortcut-episode anchors | 50983 | xy | 3.570 | 0.508 | 0.854 | 0.283 / 0.306 | 0.190 / 0.184 | 0.001 / 0.000 | 0.279 / 0.278 |
| anchors in start | 4276 | full | 4.501 | 0.659 | 0.578 | 0.775 / 0.728 | 0.143 / 0.160 | 0.024 / 0.028 | 0.044 / 0.038 |
| anchors in start | 4276 | xy | 4.355 | 0.652 | 0.586 | 0.713 / 0.728 | 0.162 / 0.160 | 0.031 / 0.028 | 0.039 / 0.038 |
| anchors in pre_zone1 | 12541 | full | 3.537 | 0.682 | 0.804 | 0.538 / 0.505 | 0.249 / 0.236 | 0.001 / 0.000 | 0.051 / 0.055 |
| anchors in pre_zone1 | 12541 | xy | 3.515 | 0.741 | 0.773 | 0.471 / 0.505 | 0.239 / 0.236 | 0.000 / 0.000 | 0.055 / 0.055 |
| anchors in zone1 | 4946 | full | 4.155 | 0.730 | 0.556 | 0.494 / 0.446 | 0.360 / 0.386 | 0.000 / 0.000 | 0.097 / 0.111 |
| anchors in zone1 | 4946 | xy | 4.018 | 0.700 | 0.554 | 0.395 / 0.446 | 0.391 / 0.386 | 0.000 / 0.000 | 0.112 / 0.111 |
| anchors in between | 12326 | full | 3.725 | 0.582 | 0.633 | 0.305 / 0.273 | 0.241 / 0.214 | 0.000 / 0.000 | 0.165 / 0.170 |
| anchors in between | 12326 | xy | 3.665 | 0.593 | 0.632 | 0.244 / 0.273 | 0.233 / 0.214 | 0.000 / 0.000 | 0.170 / 0.170 |
| anchors in zone2 | 4946 | full | 3.771 | 0.380 | 0.611 | 0.111 / 0.123 | 0.203 / 0.244 | 0.000 / 0.000 | 0.314 / 0.319 |
| anchors in zone2 | 4946 | xy | 3.626 | 0.402 | 0.636 | 0.113 / 0.123 | 0.246 / 0.244 | 0.000 / 0.000 | 0.321 / 0.319 |
| anchors in post_zone2 | 8069 | full | 3.273 | 0.200 | nan | 0.005 / 0.000 | 0.012 / 0.000 | 0.000 / 0.000 | 0.631 / 0.623 |
| anchors in post_zone2 | 8069 | xy | 3.154 | 0.148 | nan | 0.000 / 0.000 | 0.001 / 0.000 | 0.000 / 0.000 | 0.627 / 0.623 |
| anchors in goal_area | 4140 | full | 2.480 | 0.040 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.002 / 0.000 | 0.980 / 1.000 |
| anchors in goal_area | 4140 | xy | 3.053 | 0.008 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.996 / 1.000 |
| anchors in west_column | 299 | full | 6.114 | 0.352 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.936 / 0.907 | 0.061 / 0.093 |
| anchors in west_column | 299 | xy | 5.862 | 0.324 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.914 / 0.907 | 0.086 / 0.093 |
| anchors in top_corridor | 1601 | full | 5.081 | 0.347 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.895 / 0.873 | 0.105 / 0.127 |
| anchors in top_corridor | 1601 | xy | 4.750 | 0.333 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.878 / 0.873 | 0.122 / 0.127 |
| anchors in east_column | 603 | full | 3.975 | 0.379 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.635 / 0.678 | 0.365 / 0.322 |
| anchors in east_column | 603 | xy | 3.379 | 0.275 | nan | 0.000 / 0.000 | 0.000 / 0.000 | 0.693 / 0.678 | 0.307 / 0.322 |

Per-fold marginal NLL (full): 3.652 / 3.647 / 3.735; (xy): 3.552 / 3.537 / 3.802

## Horizon heads (absorbing position m steps after the query), all anchors

| model | m | NLL | expected-position error | argmax-cell error |
|---|---:|---:|---:|---:|
| full | 10 | 1.770 | 0.27 | 0.34 |
| full | 50 | 2.598 | 0.92 | 0.96 |
| full | 200 | 2.923 | 3.69 | 3.74 |
| xy | 10 | 1.037 | 0.23 | 0.20 |
| xy | 50 | 2.046 | 1.01 | 0.95 |
| xy | 200 | 2.954 | 3.83 | 3.82 |

## Models

| model | best step | best held-out marginal NLL (early-stop slice) |
|---|---:|---:|
| full_fold0 | 1000 | 3.6737 |
| full_fold1 | 1000 | 3.7293 |
| full_fold2 | 3000 | 3.6132 |
| xy_fold0 | 8000 | 3.5853 |
| xy_fold1 | 29000 | 3.5818 |
| xy_fold2 | 24000 | 3.5116 |
