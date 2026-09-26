# The v4 learned ETT on the crossover (user's plan after 48d603c, 2026-09-21; no training): under the CF loop the model's early trajectory diverges per state (heading agreement 0.66-0.69 at reset), it turns the CF actor's STALLS into deaths or completions (corridor timeouts 0.10-0.16 -> 0.00, deaths +0.15; far-route completion 0.81-0.86 vs 0.61-0.68 from reset), and it reproduces the large same-state contrasts only in the mean (per-state r <= 0.44), over-stating the finer ones

`scripts/diag_v6_ett_crossover.py run / report` (node3; 122,880 model paths in 21 min): `model_rollouts.npz` (node3), `REPORT.md` / `report.json`.
Same fixed states (384: reset / start_early / pre_zone1_early anchors, indep_reset, indep_early, cf_early), same first torques (logged / start /
CF s0 / MF s0 / CF2 s0 modes), same continuations (start, frozen CF s0), through the v4 ETT instead of the simulator: motion / stationary on
(s, a_q), the onset head with exact path-local counters, the advice generator v3 with the hidden context from the prior -- the four activity
classes x 8 clock draws (32 per state x torque x continuation), prior 1/4 per class; the onset sampled per step; every outcome kept; the
continuation policy acts on the MODEL's predicted state; the history counters come from each model path (logged prefix for anchors, the
actual start / CF prefix re-rolled and asserted for the t = 20 states); models of the anchor's fold, fold 0 for the independent states.
Compared with the simulator crossover (repeated_draws/, 16 draws per class) per state and torque and continuation.

## Q1 -- is the initial turn predicted right?  (first 30 steps vs the simulator's deterministic early trajectory, hazards off)

| group | first + cont | xy error at step 1 / 5 / 10 / 20 / 30 | pose (dims 2-14) at 10 / 30 | velocity (15-28) at 10 / 30 | heading north at 30: sim / model / per-state agreement |
|---|---|---|---|---|---|
| reset | start + start | 0.02 / 0.14 / 0.24 / 0.28 / 0.28 | 0.13 / 0.15 | 1.14 / 1.39 | 0.00 / 0.00 / 1.00 |
| reset | CF + CF | 0.01 / 0.14 / 0.44 / 1.07 / 1.69 | 0.21 / 0.26 | 1.65 / 1.57 | 0.52 / 0.64 / **0.69** |
| reset | start + CF | 0.02 / 0.13 / 0.34 / 0.87 / 1.52 | 0.23 / 0.26 | 1.78 / 1.73 | 0.36 / 0.39 / 0.66 |
| reset | CF + start | 0.01 / 0.13 / 0.33 / 0.50 / 0.75 | 0.15 / 0.19 | 1.18 / 1.28 | 0.12 / 0.06 / 0.84 |
| indep_reset | CF + CF | 0.01 / 0.19 / 0.41 / 0.94 / 1.37 | 0.25 / 0.27 | 1.80 / 1.57 | 0.42 / 0.56 / 0.67 |
| cf_early | any | 0.02 / 0.14-0.15 / 0.24-0.28 / 0.47-0.63 / 0.73-1.01 | 0.13-0.16 / 0.17-0.20 | 1.0-1.2 / 1.2-1.5 | 0.50-0.58 / 0.48-0.55 / 0.91-0.94 |
| start_early, pre_zone1_early, indep_early | any | 0.003-0.005 / 0.02-0.05 / 0.05-0.10 / 0.09-0.17 / 0.12-0.24 | 0.05-0.06 / 0.10-0.15 | 0.6-0.7 / 1.0-1.3 | 0.00-0.05 / same / 0.98-1.00 |

* The one-step error at the root does not depend on the torque (step 1: 0.01-0.02 for every first torque at reset) and the first five
  steps are equally accurate under both loops (0.13-0.14).  Under the start loop the error then saturates (0.28 at step 30: the start
  agent's corridor walk is self-correcting); under the CF loop it grows to 1.4-1.7 by step 30 and the model turns north in different
  states than the simulator (agreement 0.66-0.69 while the turning RATE is about right, 0.39 vs 0.36, 0.64 vs 0.52).  The CF loop
  amplifies small state errors at the decision states; the model is also less accurate on the turning states themselves (cf_early:
  0.73-1.01 by step 30 under any loop vs 0.12-0.28 in the corridor).

## Q2 -- after entering the far route, completion vs stall?  (and the corridor under the CF loop)

| group | first + cont | success model / sim | death model / sim | timeout model / sim | far entry model / sim | completion given far model / sim | timeout given far model / sim | mean rows model / sim |
|---|---|---|---|---|---|---|---|---|
| reset | start + start | 0.236 / 0.233 | 0.752 / 0.738 | 0.012 / 0.029 | 0.000 / 0.020 | - | - | 135 / 148 |
| reset | CF + CF | 0.576 / 0.531 | 0.301 / 0.238 | **0.124 / 0.231** | 0.650 / 0.656 | **0.81 / 0.67** | 0.17 / 0.33 | 396 / 453 |
| reset | start + CF | 0.417 / 0.428 | 0.553 / 0.402 | 0.030 / 0.170 | 0.386 / 0.457 | 0.86 / 0.68 | 0.08 / 0.32 | 240 / 359 |
| reset | CF2 + CF | 0.704 / 0.506 | | | 0.785 / 0.734 | 0.86 / 0.61 | 0.11 / 0.39 | 434 / 534 |
| reset | MF + CF | 0.450 / 0.355 | | | 0.435 / 0.457 | 0.84 / 0.50 | 0.10 / 0.50 | 280 / 397 |
| indep_reset | CF + CF | 0.391 / 0.438 | 0.395 / 0.285 | 0.215 / 0.277 | 0.587 / 0.562 | 0.55 / 0.57 | 0.37 / 0.43 | 387 / 447 |
| cf_early | start + start | 0.466 / 0.492 | 0.388 / 0.291 | 0.146 / 0.217 | 0.467 / 0.536 | 0.72 / 0.74 | 0.27 / 0.26 | 304 / 352 |
| cf_early | CF2 + start | 0.410 / 0.506 | | | 0.500 / 0.547 | 0.57 / 0.75 | 0.40 / 0.25 | 327 / 362 |
| start_early / pre_zone1_early / indep_early | any + CF | 0.12-0.15 / 0.16-0.19 | **0.85-0.88 / 0.67-0.73** | **0.00 / 0.10-0.16** | ~0.03 / ~0.04 | | | 86-123 / 157-225 |
| start_early / pre_zone1_early / indep_early | any + start | 0.22-0.26 / 0.21-0.27 | 0.73-0.75 / 0.73-0.75 | 0.00-0.03 / 0.01-0.04 | 0.00-0.03 | | | 95-140 / 96-145 |

* Under the training-like loop (start continuation from corridor / reset states) the model matches the simulator in every outcome.
* Wherever the CF actor controls, the model removes its STALLS: in the corridor the simulator's CF paths time out 0.10-0.16 of the time
  and last 157-225 rows, the model's time out 0.00, last 86-123 rows and die instead (deaths +0.15); from reset the far-route paths
  complete 0.81-0.86 in the model vs 0.61-0.68 in the simulator (timeouts given far entry 0.08-0.17 vs 0.32-0.50); at the CF actor's own
  turning states the model under-completes (0.57-0.63 vs 0.75-0.81 for CF2 / CF first + start) and over-dies (0.38-0.44 vs 0.28-0.29).
  This is the known v4 deviation (the model walks where the real dynamics stall) now measured where it matters: on the CF-controlled
  paths.  The goal-area mass is over-predicted on the far route accordingly (cf_early 0.097 vs 0.035; reset CF + CF 0.076 vs 0.036).

## Q3 -- is the same-state action difference preserved?  (mean +- state s.e., model / simulator; per-state corr, sign agreement)

| group | contrast | success | far entry | near-2.0 mass |
|---|---|---|---|---|
| reset | (CF + CF) - (logged + CF) | +0.383 +- 0.056 / +0.302 +- 0.063 (r 0.44, sign 0.78) | +0.557 / +0.578 (r 0.42) | +0.077 / +0.039 (r 0.03) |
| reset | (CF + CF) - (start + CF) | +0.159 +- 0.063 / +0.102 +- 0.056 (r -0.06, sign 0.37) | +0.264 / +0.199 (r 0.02) | +0.046 / +0.019 (r -0.01) |
| reset | (CF + CF) - (CF + start) | +0.325 / +0.296 (r 0.01) | +0.587 / +0.531 (r 0.12) | +0.047 / +0.031 |
| reset | (start + CF) - (start + start) | +0.180 / +0.196 (r -0.03) | +0.386 / +0.438 (r 0.19) | +0.019 / +0.008 |
| reset | (CF2 + CF) - (start + CF) | **+0.288 / +0.078** (r -0.18) | +0.398 / +0.277 | +0.072 / +0.021 |
| reset | (MF + CF) - (start + CF) | **+0.034 / -0.073** (r 0.03) | +0.049 / -0.000 | +0.010 / -0.010 |
| reset | (CF + start) - (start + start) | +0.014 / +0.001 | +0.062 / +0.105 | +0.018 / -0.003 |
| indep_reset | (CF + CF) - (start + CF) | +0.136 / +0.081 (r 0.14) | +0.277 / +0.234 (r 0.25) | +0.034 / +0.009 |
| indep_reset | (start + CF) - (start + start) | **+0.015 / +0.124** (r 0.38) | +0.310 / +0.328 (r 0.47) | +0.015 / +0.009 |
| corridor groups | (any + CF) - (same + start) | -0.09 .. -0.12 / -0.05 .. -0.08 (sign 0.78-0.95) | ~0 | -0.012 .. -0.025 / -0.015 .. -0.027 |
| cf_early | all | within +- 0.10, sim within +- 0.07; r -0.20 .. +0.31 | | |

* The two large contrasts at reset -- the first step (CF vs logged under the CF loop) and the continuation (CF vs start) -- are reproduced
  in the MEAN, the first even over-stated (+0.38 vs +0.30), with weak per-state correlation (0.44 and ~0).  The finer contrasts are not:
  CF2's torque advantage is over-stated 3-4x (+0.29 vs +0.08), MF's has the wrong sign (+0.03 vs -0.07), and the continuation effect at
  the independent resets is missed (+0.015 vs +0.124).  The corridor's "CF loop is worse" is reproduced and exaggerated (deaths instead of
  timeouts).  The near-2.0 mass contrasts are over-stated 2-3x through the over-completion.

## Reading (the user's ladder: fix the earliest error)

* Rung 1 ("the first steps already wrong"): NOT the case for the root step -- the one-step error is the same for every first torque
  (0.01-0.02 at step 1, 0.13-0.14 at step 5).
* Rung 2 ("one step accurate, the rollout drifts"): YES, and specifically under the CF loop and on the far-route / turning states -- the
  error grows from step 5 to step 30 (0.14 -> 1.4-1.7) where the start loop's saturates (0.28), the per-state heading disagreement is
  31-34 %, and the model is least accurate exactly on the states the CF actor visits (cf_early 0.7-1.0).  The dominant consequence is
  the missing STALL: the CF actor's hesitations (corridor timeouts 0.10-0.16, far-route timeouts 0.32-0.50) become deaths or completions.
* Rung 3 (death probability / timing) is not separable here because the deaths the model adds are the stalls it removed (the paths walk
  into the zone instead of waiting); the onset head is not shown wrong by this run.
* Rung 4 ("the model already reproduces the action differences"): only the coarse mean contrasts; not per state, not the finer ones.
* So the earliest error is the closed-loop motion on CF-controlled paths (the stall / stationary behaviour and the turn), and the
  supervision that would address it is one-step (state, query torque) rows from CF-controlled simulator paths at the reset / turning /
  corridor states (the states this run used: the CF actor's own torques where the real dynamics stall or turn), with short multi-step
  unrolling on the existing one-step model as the constraint on accumulation -- the logged supervision kept, validation held out by
  episode, the ETT still outputting the next state with no route label, the deployed action still one 8-D torque.  New simulator
  supervision stays in the engineering-verification stage.  Not started: the user decides.
* Caveats: 64 states per group; fold 0 models for the independent states; the model's completion / stall numbers under the start loop from
  cf_early are ALSO off (0.57-0.63 vs 0.75-0.81), so the far-route walking error is not only a CF-loop effect; the simulator's per-state
  contrasts carry their own draw noise (16 per class), which bounds the attainable per-state correlations.
