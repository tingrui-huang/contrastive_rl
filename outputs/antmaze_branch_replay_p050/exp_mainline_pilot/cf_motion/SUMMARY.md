# The controlled CF-motion revision of the v4 one-step ETT (user's steps 1-3 after 5650f8d, 2026-09-21): one-step accuracy on CF-controlled states clearly improves (CF-val MSE 0.163 -> 0.095; slow rows 7x), the early closed-loop trajectory improves 10-25 %, but the drift and the STALLS in rollouts do not -- the model reproduces a stall only from the exact stuck state

## (1) The targeted sequences -- `scripts/collect_v6_cf_motion.py` (node3, 18 workers, 8 min); `cf_motion/summary.json`; rollouts on node3

The frozen CF s0 actor's mode rolled from 576 pre-selected start anchors of the training pool (reset 128, start_early 128, pre_zone1_early
128, pre_mouth 128, between 64; seeded uniform; the 192 repeated-draws hold-out anchors excluded), two rollouts each with natural hidden
draws (both hazards Bernoulli 0.5, fresh clocks / jitter), every outcome kept: 1,152 rollouts, 325,027 transitions (s_t, a_q_t, s_t+1) with
the full 29-d state; outcomes 353 success / 559 death / 240 timeout; far-route rows 101k; fully static rows 101k (31 %), slow rows (xy
speed < 0.03, not static) 31k; 239 rollouts with a stall segment >= 20 rows; split by start point 70 / 15 / 15 (train 239k / val 47k / test
39k rows).  The teacher's advice / onset are not recorded (the v4 motion model and gate take (s, a_q) only); a teacher-walk pass would be
needed for onset / generator supervision on these paths.  Oracle supervision: engineering-verification stage.

## (2) The controlled comparison -- `fit_v6_ett_one_step_v2.py fit --v4 --init-from ett_one_step_v4 --freeze-onset --steps 10000 --cf-rows ... --cf-share {0 | 0.25}`

Both arms start from the v4 fold models (params and standardisation), 10,000 updates, Adam re-initialised, the onset head frozen, the same
selection rule (the original validation score + the CF-val one-step MSE on moving rows; both arms).  Control v4c: the old data only (CF share
0).  Revision v4r: 25 % of every motion and stationary batch replaced by CF train rows (the batch size unchanged; disclosed as the mixing
choice).  `ett_one_step_v4c/`, `ett_one_step_v4r/` (jsons in the repo, pkls on node3).

| arm | selected step (folds 0 / 1 / 2) | original val MSE diag / off | onset AUROC | stat acc | CF-val MSE moving / all | CF-val stat acc | motion param delta |
|---|---|---|---|---|---|---|---|
| v4 (original) | 39k / 40k / 40k | 0.0129 / 0.0332, 0.0137 / 0.0333, 0.0097 / 0.0281 | 0.997-0.998 | 0.983-0.988 | (0.13 at the first check) | | |
| v4c (control) | 9k / 10k / 6k | 0.0127 / 0.0325, 0.0131 / 0.0322, 0.0094 / 0.0274 | 0.997-0.998 | 0.983-0.988 | **0.163 / 0.135, 0.149 / 0.123, 0.146 / 0.110** | 0.983-0.988 | 4.5-6.9 |
| v4r (revision) | 9k / 9k / 9k | 0.0129 / 0.0335, 0.0135 / 0.0330, 0.0096 / 0.0280 | 0.997-0.998 | 0.986-0.989 | **0.100 / 0.073, 0.095 / 0.070, 0.097 / 0.071** | 0.986-0.990 | 16.3-16.8 |

No regression on the original validation (start-agent / logged rows) in either arm; the revision lowers the one-step error on CF-controlled
states by ~40 % (standardised units), the control does not move it.

## (3) The three-layer acceptance -- `scripts/diag_v6_cf_motion_accept.py` on the TEST split (174 rollouts, 39,416 rows); `cf_motion/accept/`

| layer | quantity | v4 | v4c | v4r |
|---|---|---|---|---|
| L1 teacher-forced | one-step xy error median / p90: moving rows (n 21,780) | 0.0063 / 0.0236 | 0.0063 / 0.0238 | **0.0053 / 0.0181** |
| L1 | slow rows (n 8,083) | 0.0206 / 0.0293 | 0.0218 / 0.0308 | **0.0028 / 0.0081** |
| L1 | turn rows (first 30 from reset, n 840) / far route (n 12,116) / corridor / zone | 0.0147 / 0.0187 / 0.0031 / 0.0015 | 0.0156 / 0.0191 / 0.0031 / 0.0014 | **0.0119 / 0.0031** / 0.0028 / 0.0014 |
| L1 | slow rows: real speed / predicted speed / predicted frozen share | 0.0034 / 0.0187 / 0.09 | 0.0034 / 0.0203 / 0.07 | 0.0034 / **0.0042** / 0.00 |
| L1 | gate fires on static / slow / moving rows | 0.970 / 0.092 / 0.000 | 0.970 / 0.066 / 0.000 | 0.970 / 0.002 / 0.000 |
| L2 open loop (real actions) | xy error median at step 5 / 10 / 20 / 30 / 50 / 100 | 0.04 / 0.10 / 0.20 / 0.37 / 0.64 / 2.10 | 0.04 / 0.10 / 0.22 / 0.40 / 0.86 / 2.25 | 0.04 / 0.08 / 0.24 / 0.43 / 0.81 / 2.14 |
| L2 | real stall segments (14, median 26 rows): real / model displacement; share model > 0.5 | 0.05 / 0.43; 0.43 | 0.05 / 0.47; 0.43 | 0.05 / **0.67; 0.71** |
| L3 closed loop (CF actor on the model state; 31 hazard-free sequences) | heading north at 30: real / model / agreement | 0.06 / 0.10 / 0.97 | 0.06 / 0.03 / 0.97 | 0.06 / 0.06 / 0.94 |
| L3 | reach by 400: real / model / agreement | 0.61 / 0.97 / 0.65 | 0.61 / 0.97 / 0.65 | 0.61 / 0.94 / 0.61 |
| L3 | stalled in the last 100 steps: real / model / agreement | 0.39 / 0.00 / 0.61 | 0.39 / 0.03 / 0.65 | 0.39 / 0.00 / 0.61 |

* L1 improves as the CF-val said: on the CF states the revision predicts slow / swaying rows at the right speed (0.004 vs 0.019 predicted
  for a real 0.003) and far-route rows 6x better; static rows are gated identically (97 %).
* L2 / L3 do not: with the real action sequence the revision drifts as much as v4 (0.43 vs 0.37 at step 30; 2.1 at 100) and moves MORE
  through the real stall segments (0.67 vs 0.43 over a real 0.05); with the CF actor in the loop no model stalls (0.00-0.03 vs real
  0.39) and all over-reach (0.94-0.97 vs 0.61).  The one-step model reproduces a stall only when handed the exact stuck state; once its
  own state is slightly off, the gate does not fire and the actor's torque moves it on.  A stall is a basin of attraction of the real
  contact dynamics, which the point-wise one-step fit does not represent.

## The crossover re-run with the fixed diagnostic (`diag_v6_ett_crossover.py --model-dir ... --tag {v4fix, v4c, v4r}`; paired clocks and death uniforms keyed by (state, class, rep); Q1 from a motion-only pass at exactly step 30)

| group | first + cont | Q1 xy error at 10 / 20 / 30, heading agreement: v4 | v4c | v4r |
|---|---|---|---|---|
| reset | CF + CF | 0.44 / 1.07 / 1.68, 0.69 | 0.39 / 1.05 / 1.73, 0.70 | **0.33 / 0.85 / 1.32**, 0.73 |
| reset | start + CF | 0.34 / 0.88 / 1.52, 0.64 | 0.34 / 0.87 / 1.49, 0.58 | 0.28 / 0.76 / 1.38, 0.66 |
| indep_reset | CF + CF | 0.41 / 0.94 / 1.38, 0.67 | 0.37 / 0.92 / 1.59, 0.61 | 0.33 / 0.84 / 1.29, 0.75 |
| cf_early | CF + CF / start + start | 1.01 / 0.78 at 30 | 0.81 / 0.66 | **0.67 / 0.61** |
| reset | start + start (no regression) | 0.28 at 30, 1.00 | 0.29, 1.00 | 0.33, 1.00 |

| group | corner | success model / sim, timeout model / sim, completion given far model / sim: v4 | v4c | v4r |
|---|---|---|---|---|
| reset | CF + CF | 0.57 / 0.53, 0.12 / 0.23, 0.82 / 0.67 | 0.60 / 0.53, 0.13 / 0.23, 0.77 / 0.67 | 0.61 / 0.53, **0.02** / 0.23, **0.92** / 0.67 |
| indep_reset | CF + CF | 0.39 / 0.44, 0.22 / 0.28, 0.56 / 0.57 | 0.59 / 0.44, 0.14 / 0.28, 0.76 / 0.57 | 0.50 / 0.44, 0.06 / 0.28, 0.78 / 0.57 |
| cf_early | CF + CF | 0.41 / 0.45, 0.14 / 0.27, 0.67 / 0.61 | 0.48 / 0.45, 0.08 / 0.27, 0.77 / 0.61 | 0.52 / 0.45, 0.04 / 0.27, 0.86 / 0.61 |
| start_early | CF + CF | 0.13 / 0.18, 0.02 / 0.14 | 0.13 / 0.18, 0.02 / 0.14 | 0.16 / 0.18, 0.00 / 0.14 |

| group | contrast (success) | sim | v4 | v4c | v4r |
|---|---|---|---|---|---|
| reset | (CF + CF) - (logged + CF) | +0.302 | +0.392 (r 0.45) | +0.413 (r 0.36) | +0.455 (r 0.38) |
| reset | (CF + CF) - (start + CF) | +0.102 | +0.156 (r -0.04) | +0.216 (r 0.01) | +0.205 (r 0.14) |
| reset | (CF2 + CF) - (start + CF) | +0.078 | +0.288 | +0.266 | +0.318 |
| reset | (MF + CF) - (start + CF) | -0.073 | +0.036 | +0.140 | +0.013 |
| indep_reset | (start + CF) - (start + start) | +0.124 | +0.018 (r 0.37) | +0.162 (r 0.29) | +0.125 (r 0.40) |

* The early trajectory under the CF loop improves 10-25 % with the revision and not with the control (the continued training itself does
  nothing there); the heading agreement moves little (0.66-0.75 vs 0.64-0.69).
* The stalls are still removed -- the revision's rollouts time out even less (reset CF + CF 0.02 vs v4 0.12, simulator 0.23) and complete the
  far route even more (0.92 vs 0.82, simulator 0.67), consistent with L2 / L3.
* The same-state contrasts: the coarse ones stay over-stated (CF vs logged first +0.46 vs +0.30; CF2 +0.32 vs +0.08); the continuation effect
  at the independent resets is recovered by the revision (+0.125 vs +0.124) BUT the control shows it too (+0.162 vs v4's +0.018), and v4c
  differs from v4 in several corners (indep_reset CF + CF 0.59 vs 0.39) -- the model-based closed-loop outcomes have a sensitivity to small
  parameter changes of the same order as the effects read here.  Per-state correlations remain <= 0.45 for every model.

## Reading against the user's acceptance conditions

* Turn prediction on held-out starts: improved modestly (L1 turn rows 0.0119 vs 0.0147; Q1 CF-loop error at step 30 -10 to -25 %; heading
  agreement +0.02 to +0.08).
* Real stall paths no longer predicted as progress: NOT met -- L2 stall displacement 0.67 vs 0.43 (real 0.05), L3 stalled 0.00 vs 0.39,
  crossover timeouts 0.00-0.06 vs 0.10-0.28.
* Same-state action differences closer to the simulator: not clearly -- one contrast recovered (also by the control), the coarse ones
  over-stated, per-state correlations unchanged.
* No regression under the start policy: met (original validation and the start + start corners unchanged).
* The user's ladder: layer 1 is now accurate on the CF states (one-step xy error 0.003-0.005 on moving / slow rows, the same order as on
  the original validation) while the rollout still drifts and loses the stalls -> the next rung is the SHORT-SEQUENCE MULTI-STEP
  supervision on the existing one-step model (from a real state, the recorded actions unrolled a few steps, matched to the real states),
  which must include the stall segments and the turn segments so that the model learns to stay in a stall from nearby states and to keep
  the turn; the deployed action stays one 8-D torque, the ETT still predicts the next state, no CRL loss change.  Not started.
* Caveats: 64 states per group and 31 hazard-free test sequences for L3; the closed-loop outcomes vary between nearly identical models
  (v4 vs v4c), so corner differences below ~0.1 should not be read as the revision's effect; the onset head was not retrained and its
  inputs now come from a different motion model -- its validation AUROC / BCE at the selected step are unchanged (0.997-0.998), but the
  closed-loop death timing under the new motion was not re-measured separately from the stall removal.
