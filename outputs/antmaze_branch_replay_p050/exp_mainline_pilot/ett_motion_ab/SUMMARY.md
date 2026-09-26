# Does the motion / stationary model turn the advice a_b into motion?  (user's check after 4de6e8d, 2026-09-20): the motion regression barely, the STATIONARY GATE substantially

`scripts/diag_v6_ett_motion_ab.py run` (node3, fixed v3 models, held-out rows / anchors per fold; no training).  `REPORT.md`, `report.json`.
Read-out thresholds stated in advance: a_b-swap xy shift >= 0.30 x the a_q-swap shift; stationary-gate flip rate >= 0.05; closed-loop
outcome gap between advice sources >= 0.05.

## A. One-step sensitivity with (s, a_q) fixed (held-out rows, 40k per stratum)

| stratum | real xy displ. (med) | model xy error (med / p90) | a_b -> generator draw: xy shift (med / p90), gate flip | a_b -> zero (hold) | a_b -> a_q | a_b -> another row's | **a_q swapped between rows** (the executed torque) |
|---|---:|---|---|---|---|---|---|
| all | 0.105 | 0.0013 / 0.0063 | 0.0002 / 0.0017, 0.005-0.012 | 0.0011 / 0.0037, 0.04-0.09 | 0.0001 / 0.0020, 0.04-0.06 | 0.0017 / 0.0053, 0.13 | **0.0105 / 0.0255, 0.20-0.22** |
| start | 0.07-0.12 | 0.002 / 0.008 | 0.0004 / 0.0024, 0.001-0.026 | 0.0018 / 0.0045, **0.26-0.29** (folds 0-1) | 0.0001 / 0.0025, 0.03-0.06 | 0.0023 / 0.0059, 0.03-0.16 | 0.0105 / 0.025, 0.04-0.19 |
| pre_zone1 (mostly stalls) | 0.0002 | 0.0002 / 0.0025 | 0.0000 / 0.0005, 0.01-0.03 | 0.0003 / 0.0018, **0.09-0.18** | 0.0001 / 0.0011, **0.09-0.15** | 0.0006 / 0.0028, **0.19-0.20** | 0.0003 / 0.019, 0.13-0.19 |
| stationary (real) | 0.0000 | 0.0000 | 0.0000, 0.01-0.03 | 0.0000 / 0.0008, **0.09-0.30** | 0.0000 / 0.0007, **0.09-0.19** | 0.0002 / 0.0025, **0.43-0.50** | 0.002 / 0.014, 0.44-0.53 |
| far legs, moving | 0.121 | 0.0045 / 0.017 | 0.0014 / 0.0053, 0.000 | 0.0027 / 0.0065, 0.002 | 0.0014 / 0.0053, 0.002 | 0.0034 / 0.0082, 0.002 | 0.0126 / 0.029, 0.003 |

* Motion regression: swapping the advice moves the predicted xy by 2-16 % of what swapping the executed torque does (below the
  0.30 threshold) -- small, but of the same order as the model's own one-step error on moving rows (0.0014-0.0034 vs 0.0013-0.0046).
* Stationary gate: at the rows where it decides (real stalls, the pre-mouth corridor, the start region) the advice flips the gate
  9-50 % of the time under advice swaps -- as often as swapping the executed torque (13-53 %) -- and replacing the advice by a hold
  flips it for 26-29 % of the start rows in two folds.  The generator's own draw (what generation uses) flips it less (1-3 %)
  because it stays close to the teacher's advice.

## B. Closed-loop from held-out anchors vs the simulator branch of the same anchor (deterministic motion; the onset integrated)

| group (fold 0 / 1 / 2) | simulator | model, generator advice | model, a_b = a_q | model, the branch's real advice |
|---|---|---|---|---|
| far-route legs (905 / 689 / 909 anchors): reach | 0.86 / 0.85 / 0.82 | 0.68 / 0.86 / 0.74 | 0.71 / 0.83 / 0.69 | 0.71 / 0.90 / 0.77 |
| ... stationary gate on, share of steps | (real stationary share ~0.20) | 0.19 / 0.07 / 0.17 | 0.24 / 0.08 / 0.33 | 0.11 / 0.05 / 0.19 |
| pre_zone1 anchors whose simulator branch STALLED (780 / 759 / 768): reach | 0.00 (timeout 1.00) | **0.87 / 0.89 / 0.81** | 0.69 / 0.70 / 0.70 | 0.84 / 0.92 / 0.70 |
| pre_zone1 anchors whose simulator branch reached (1329 / 1311 / 1279): reach | 1.00 | 0.90 / 0.94 / 0.93 | 0.78 / 0.87 / 0.89 | 0.88 / 0.94 / 0.93 |

Every model path leaves the simulator path (|xy| > 0.25) by step ~28 (far legs) / ~45-65 (pre_zone1), classified as DRIFT (never
"the gate stalled while the simulator moved"); where the simulator stalls before the mouth, the model walks on (the deviation
kind "sim_stalled" ~50 %).  Teacher-forced along the simulator paths the gate is accurate (false-stationary 0.3-0.5 % on the far
legs, 4-7 % on the stall anchors; missed stationary 2-7 % except after a reach path 26-46 % of rare stationary rows).

## Reading

The user's suspicion is confirmed for the stationary gate and only weakly for the motion regression: the gate -- which zeroes
all 29 state increments when it fires -- depends on the un-executed advice about as much as on the executed torque exactly
where stalls are decided, and in closed loop the advice source moves the reach rate by up to 0.12 (a_b = a_q vs the generator's
advice on the pre_zone1 groups) and 0.03-0.05 on the far legs.  Two further facts for the revision: (i) the model UNDER-stalls
where the simulator stalls before the mouth (reach 0.81-0.89 vs 0.00) and (ii) it drifts off the simulator's far-route paths
early, with a reach shortfall of 0.08-0.18 in two folds that no advice source repairs -- long-path motion error, not advice.
Decision (user's plan): retrain the motion regression and the stationary gate on (s, a_q) only, keep the onset head on
(s, a_b, a_q, history); re-check the gate's accuracy, the far-route completion and the pre-mouth stalls on held-out paths, the
onset calibration, then the rollout and the futures.
