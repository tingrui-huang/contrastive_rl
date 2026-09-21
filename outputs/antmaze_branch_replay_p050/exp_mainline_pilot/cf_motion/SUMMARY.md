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
  0.39) and all over-reach (0.94-0.97 vs 0.61).  [Withdrawn on the user's review: this comparison starts from the trajectory start, so
  the model may reach the segment in a different pose; whether the local stall dynamics are learned is settled by the entry check of
  ROUND 2 below -- they are, for the CF-supervised models; the failure here was the accumulated error before the segment.]

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

# ROUND 2 (user's review of 37b990f): the stall-entry check, the episode-level split, and the short-sequence multi-step term

## Fixes adopted

* L3's horizon unified: the real rollout and the model are both judged inside the first 400 steps (before, the real used its full length).
* The split by start anchor let anchors of one source episode fall into different splits: 28 of the 84 anchor-split test episodes also had
  train anchors.  Re-split BY SOURCE EPISODE (`collect_v6_cf_motion.py resplit` -> `rollouts_cf_ep.npz`: 307 / 66 / 66 episodes, 412 / 77 / 87
  starts, 224k / 52k / 50k rows; overlap 0).  v4c / v4r keep the anchor split (their numbers are reported on it); the new arms use the episode split.
* "the stall's basin of attraction is not learned" was too strong for a comparison from the trajectory start -- replaced by the entry check below.

## The stall-entry check (no training; `diag_v6_cf_motion_accept.py` L2b): the model restarted at the REAL entry state of each real stall segment (and 20 rows before it), the recorded actions fed through the segment

| model (split) | restart | n segments | real disp | model disp median / p90 | share model disp > 0.5 / < 0.2 | xy error at the entry | gate fires |
|---|---|---:|---:|---|---|---:|---:|
| v4 (anchor) | at entry | 59 | 0.038 | **2.59 / 95.0** | 0.90 / 0.05 | 0 | 0.00 |
| v4 | 20 before | 53 | 0.038 | 3.12 / 36.9 | 0.96 / 0.02 | 1.17 | 0.00 |
| v4c | at entry | 59 | 0.038 | 2.18 / 151 | 0.85 / 0.14 | 0 | 0.00 |
| v4r (anchor) | at entry | 59 | 0.038 | **0.15 / 10.3** | 0.32 / 0.64 | 0 | 0.00 |
| v4r | 20 before | 53 | 0.038 | 0.16 / 6.5 | 0.28 / 0.58 | 0.14 | 0.00 |
| v4 (episode) | at entry | 64 | 0.024 | 3.23 / 1261 | 0.94 / 0.03 | 0 | 0.00 |
| v4a one-step (episode) | at entry | 64 | 0.024 | **0.09 / 4.4** | 0.25 / 0.72 | 0 | 0.00 |
| v4a | 20 before | 64 | 0.024 | 0.16 / 4.9 | 0.36 / 0.55 | 0.22 | 0.00 |
| v4b one-step + multi-step (episode) | at entry | 64 | 0.024 | 0.17 / 3.2 | 0.39 / 0.55 | 0 | 0.00 |
| v4b | 20 before | 64 | 0.024 | 0.27 / 4.5 | 0.41 / 0.34 | 0.19 | 0.00 |

* (Correction, round 3: the episode-split rows above said real disp 0.05 in the committed version of this table; report.json has 0.024 --
  the 0.05 was the trajectory-start segments' figure from L2.)
* The ORIGINAL v4 has not learned the local stall dynamics at all: from the exact entry state with the recorded actions it walks 2.6-3.2
  over a real 0.04 in 90 % of the segments (the gate never fires -- these stalls are swaying / slow rows, not fully static ones).
* The CF-supervised models HAVE learned them locally: from the entry, 0.09-0.17 median (55-72 % of segments under 0.2), and from 20 rows
  before the entry 0.16-0.27 with an entry error of 0.14-0.22.  So the trajectory-level failure (37b990f) was mostly the accumulated
  state error BEFORE the segment; the local dynamics are now mostly right, with a heavy tail (p90 3-10) of segments that still run away.

## Arms A / B under the episode split (`fit --init-from ett_one_step_v4 --freeze-onset --steps 10000 --cf-rows rollouts_cf_ep.npz --cf-share 0.25 [--multistep 10]`)

B adds a short-sequence term: windows of K = 10 recorded actions from the CF train rollouts (half starting on slow / static rows), the raw
regression unrolled on its OWN predictions from the real start state (no gate), MSE against the recorded states in standardised state
units, weight 1, batch 256; the one-step training kept; same init, budget and selection rule as A.

| arm | one-step CF-val MSE (moving) | original val diag / off | 10-step open-loop xy error on CF-val windows |
|---|---|---|---|
| A one-step | 0.113-0.119 | 0.0096-0.0132 / 0.028-0.034 (unchanged) | (not computed in the fit) |
| B + multi-step | 0.119-0.126 | 0.0117-0.0158 / 0.031-0.036 (+15-20 %) | 0.064 |

## Acceptance on the episode-level test split (66 episodes, 174 rollouts, 49,542 rows; `cf_motion/accept_ep/`)

| layer | quantity | v4 | A one-step | B + multi-step |
|---|---|---|---|---|
| L1 | one-step xy error median: moving / slow / turn30 / far / corridor | 0.0070 / 0.0248 / 0.0140 / 0.0113 / 0.0017 | 0.0059 / 0.0023 / 0.0113 / 0.0046 / 0.0016 | 0.0065 / 0.0029 / 0.0122 / 0.0051 / 0.0018 |
| L1 | slow rows: predicted speed (real 0.0082) | 0.0250 | 0.0095 | 0.0096 |
| L2 open loop (real actions) | xy error median at 5 / 10 / 20 / 30 / 50 / 100 | 0.04 / 0.11 / 0.22 / 0.40 / 0.81 / 2.04 | 0.04 / 0.10 / 0.19 / 0.35 / 0.65 / 1.59 | 0.04 / 0.09 / 0.16 / **0.22 / 0.43 / 1.04** |
| L2 | real stall segments from the trajectory start (8; real disp 0.05): model disp; share > 0.5 | 0.77; 0.75 | 1.04; 0.75 | **0.44; 0.25** |
| L3 closed loop (38 hazard-free seqs, 400-step window) | heading north at 30 real / model / agreement | 0.13 / 0.16 / 0.97 | 0.13 / 0.18 / 0.95 | 0.13 / 0.13 / 0.95 |
| L3 | far entry by 100 real / model / agreement | 0.16 / 0.18 / 0.97 | 0.16 / 0.18 / 0.97 | 0.16 / 0.13 / 0.97 |
| L3 | reach by 400 real / model / agreement | 0.42 / 0.87 / 0.55 | 0.42 / 0.87 / 0.55 | 0.42 / 0.82 / 0.55 |
| L3 | stalled in the last 100 real / model / agreement | 0.47 / 0.00 / 0.53 | 0.47 / 0.03 / 0.55 | 0.47 / 0.05 / 0.47 |

## The crossover with A and B (fixed diagnostic; `ett_crossover/v4a`, `ett_crossover/v4b`)

| quantity | sim | v4 | v4r | A | B |
|---|---|---|---|---|---|
| Q1 reset CF + CF: xy error at 10 / 20 / 30 (heading agreement) | | 0.44 / 1.07 / 1.68 (0.69) | 0.33 / 0.85 / 1.32 (0.73) | 0.32 / 0.82 / 1.40 (0.70) | 0.30 / 0.78 / 1.30 (0.69) |
| Q1 cf_early CF + CF: at 30 (agreement) | | 1.01 (0.94) | 0.67 (0.92) | 0.68 (0.94) | **0.53** (0.94) |
| Q1 reset start + start: at 30 (no regression) | | 0.28 | 0.33 | 0.47 | 0.45 |
| reset CF + CF: success / timeout / completion given far | 0.53 / 0.23 / 0.67 | 0.57 / 0.12 / 0.82 | 0.61 / 0.02 / 0.92 | 0.54 / 0.14 / 0.69 | 0.60 / **0.15** / **0.75** |
| indep_reset CF + CF: success / timeout / completion | 0.44 / 0.28 / 0.57 | 0.39 / 0.22 / 0.56 | 0.50 / 0.06 / 0.78 | 0.61 / 0.14 / 0.79 | 0.51 / 0.11 / 0.76 |
| cf_early CF + CF: success / timeout / completion | 0.45 / 0.27 / 0.61 | 0.41 / 0.14 / 0.67 | 0.52 / 0.04 / 0.86 | 0.53 / 0.07 / 0.83 | 0.51 / 0.11 / 0.72 |
| reset (CF + CF) - (logged + CF), success | +0.302 | +0.392 (r 0.45) | +0.455 (r 0.38) | +0.376 (r 0.36) | +0.450 (r 0.27) |
| reset (CF + CF) - (start + CF) | +0.102 | +0.156 | +0.205 | +0.114 | +0.135 |
| reset (CF2 + CF) - (start + CF) | +0.078 | +0.288 | +0.318 | +0.161 | **+0.110** |
| reset (MF + CF) - (start + CF) | -0.073 | +0.036 | +0.013 | -0.000 | +0.016 |
| indep_reset (start + CF) - (start + start) | +0.124 | +0.018 | +0.125 | +0.117 | +0.235 |

## Reading against the user's decision rule for retraining CRL

* Long-range drift down: YES for B -- the open-loop xy error is halved at 30 / 50 / 100 steps (0.22 / 0.43 / 1.04 vs 0.40 / 0.81 / 2.04) and the
  early error under the CF loop at the turning states falls to 0.53 (v4 1.01).
* Stalls reproduced: PARTLY -- locally yes (entry check 0.09-0.17 vs v4's 2.6-3.2), from the trajectory start improved (0.44 vs 0.77-1.04 over a
  real 0.05; 25 % of segments still run > 0.5), the timeouts under the CF loop partly restored (0.15 vs v4r's 0.02, simulator 0.23), but in the
  closed loop with the CF actor the model still does not stall (0.05 vs real 0.47) and over-reaches (0.82 vs 0.42).
* Normal walking preserved: MOSTLY -- one-step error on moving rows 0.0065 (v4 0.0070), the original validation +15-20 % worse in B, and the
  start + start early error at reset 0.45 vs 0.28 in both A and B (a regression to watch; v4r had 0.33).
* Same-state action differences closer: PARTLY -- the CF2 over-statement fixed (+0.11 vs sim +0.08; v4 +0.29), the continuation effect at
  reset closer (+0.135 vs +0.102); the coarse first-step contrast still over-stated (+0.45 vs +0.30), MF's sign still wrong, per-state
  correlations unchanged (<= 0.3).
* Verdict: the multi-step term is the right rung (it fixes what the one-step term could not, at a small one-step cost), but the closed-loop
  stall is still missing, so the bar for re-running the critic + actor is not met yet.  Candidates (user's decision): a longer / stall-covering
  unroll (K 20-30, windows spanning whole stall segments), a higher weight, or the gate replaced by a soft slow-down learned from the same
  rows; each ~30 min on the three nodes now available.  The oracle's own instability (0.449 vs 0.351 across draws) is a separate open problem
  that a better ETT does not touch.


# ROUND 3 (user's plan after 20b9401): (a) recorded actions vs (b) the actor's feedback on the same stall entries; one longer window (K 20); stop point

Rules set by the user before this round: compare (a) and (b) on the same stall segments first; only if the accumulated error under
FIXED actions is still the main problem, try ONE longer unroll window (K 20) with everything else unchanged (no weight, gate or network
change at the same time) and compare it with the K 10 arm; stop point = the evening of 2026-09-22 -- if the closed-loop reach / stall
deviation and the same-state action contrasts show no clear improvement, this week's AntMaze model revision stops; a further drop in the
position error alone does NOT justify new CRL seeds.

## (a) vs (b) on the same 64 real stall segments (episode-split test rollouts; median segment length 496 rows; real displacement 0.024; `cf_motion/accept_ep2/`, node 30049)

Both start from the SAME state (the real entry, or 20 rows before it) and run the same number of steps: (a) the recorded actions fed to
the model; (b) the same frozen CF s0 actor choosing the action from the MODEL's predicted state (`diag_v6_cf_motion_accept.py` L2b,
`cf_mode`).  Action deviation = mean |a_actor - a_recorded| per dimension (actions in [-1, 1]).

| model | restart | (a) model disp median / p90; share > 0.5 / < 0.2 | (b) closed-loop disp median / p90; share > 0.5 / < 0.2 | share (b) - (a) > 0.3 | action dev | entry xy err (a) / (b) |
|---|---|---|---|---:|---:|---|
| v4 (original) | at entry | 3.23 / 1261; 0.94 / 0.03 | 2.80 / 170; 0.94 / 0.03 | 0.14 | 0.42 | 0 / 0 |
| v4 | 20 before | 3.03 / 39.0; 0.98 / 0.02 | 4.71 / 18.5; 0.97 / 0.03 | 0.45 | 0.67 | 1.05 / 1.06 |
| A one-step | at entry | 0.090 / 4.36; 0.25 / 0.72 | 0.052 / 2.28; 0.25 / 0.72 | 0.03 | 0.10 | 0 / 0 |
| A | 20 before | 0.162 / 4.87; 0.36 / 0.55 | 0.159 / 14.1; 0.38 / 0.58 | 0.20 | 0.21 | 0.22 / 0.35 |
| B one-step + K 10 | at entry | 0.171 / 3.25; 0.39 / 0.55 | 0.260 / 4.00; 0.36 / 0.39 | 0.08 | 0.08 | 0 / 0 |
| B | 20 before | 0.266 / 4.52; 0.41 / 0.34 | 0.246 / 3.33; 0.38 / 0.48 | 0.11 | 0.13 | 0.19 / 0.30 |

* For the CF-supervised models the actor's feedback changes almost nothing: the medians and the share of run-away segments (> 0.5) are
  the same under (a) and (b) (A 0.25 / 0.25, B 0.39 / 0.36 at the entry; 0.36 / 0.38 and 0.41 / 0.38 from 20 rows before); the segments
  that run away under the actor are the ones that already run away under the recorded actions; the actor's extra run-aways are 3-8 % of
  the segments at the entry and 11-20 % from 20 rows before.  The actor's torques stay close to the recorded ones (0.08-0.13 per
  dimension) because the model's state stays close to the real one wherever the stall is held.
* The original v4 is the only model where the feedback matters (0.45 of the segments worse by > 0.3 from 20 rows before, action deviation
  0.67): on a wrong state the actor picks different torques -- but v4 is not a candidate.
* Reading under the user's rule: the accumulated error under FIXED actions is the main problem (25-40 % of the 500-row segments run away
  with the recorded actions; the actor's feedback adds a small fraction on top).  The condition for trying one longer window is met.

## The one longer window: arm C = arm B with K 20 (`--multistep 20`; nothing else changed: episode split, v4 init, onset frozen, CF share 0.25, weight 1, batch 256, stall share 0.5, 10k updates, same selection)

Training (node3, three folds in parallel, 4 min per fold): the 20-step unroll at weight 1 is at the edge of stability -- folds 0 and 2
diverged after step 8000 (fold 2: motion loss 0.15 -> 42, original validation MSE 0.012 -> 5.1 at step 9000, recovering to 1.1 at 10000;
fold 0 a smaller spike, 0.016 -> 0.034); the pre-registered selection rule took the pre-spike checkpoints (step 8000 / 10000 / 8000).
At the selected steps: original validation diag 0.012-0.016 / off 0.031-0.037 (B 0.012-0.016 / 0.030-0.036), CF-val one-step MSE on
moving rows 0.123-0.133 (B 0.119-0.126), 20-step open-loop xy error on the CF-val windows 0.10-0.11 (B's 10-step 0.064; not the same
horizon).  A longer window than 20 would need the weight or gradient clipping changed, i.e. two things at once -- outside this round.

The acceptance (v4 / B / C, episode split, with the (a) / (b) check; node3, `cf_motion/accept_ep_c20/`) and the crossover for C (node
30027, `ett_crossover/v4c20/`): see the two sections after the reproducibility bound.

## A reproducibility bound for the closed-loop numbers (found by re-running the same acceptance on a second GPU)

The SAME models and the SAME frozen actor give different 400-step closed-loop outcomes on the 38 hazard-free sequences on the two GPUs
(node3 4090L vs node 30049 3090; JAX numerics, chaotic rollouts): reach by 400 -- v4 0.87 vs 0.89, A 0.87 vs 0.82, B 0.82 vs 0.68;
stalled in the last 100 -- B 0.05 vs 0.08.  So the closed-loop reach / stall deviation has a device-noise floor of about +-0.1 on this
set; "clear improvement" under the stop rule has to exceed it.  (L1 / L2 / L2b, which restart from real states, agree to the third
decimal across the two runs.)

## Arm C (K 20) vs arm B (K 10): the acceptance (node3, same device for all three models; `cf_motion/accept_ep_c20/`)

| layer | quantity | v4 | B (K 10) | C (K 20) |
|---|---|---|---|---|
| L1 | one-step xy error median: moving / slow / turn30 / far / corridor | 0.0070 / 0.0248 / 0.0140 / 0.0113 / 0.0017 | 0.0065 / 0.0029 / 0.0122 / 0.0051 / 0.0018 | 0.0067 / 0.0035 / 0.0123 / 0.0054 / 0.0019 |
| L2 open loop (real actions) | xy error median at 5 / 10 / 20 / 30 / 50 / 100 | 0.04 / 0.11 / 0.22 / 0.40 / 0.81 / 2.04 | 0.04 / 0.09 / 0.16 / 0.22 / 0.43 / 1.04 | 0.04 / 0.09 / 0.16 / 0.22 / **0.33 / 0.64** |
| L2 | real stall segments from the trajectory start (8; real disp 0.05): model disp; share > 0.5 | 0.77; 0.75 | 0.44; 0.25 | **0.23; 0.00** |
| L2b (a) at the entry | model disp median / p90; share > 0.5 | 3.23 / 1261; 0.94 | 0.17 / 3.2; 0.39 | **0.06 / 2.8; 0.25** |
| L2b (a) 20 before | model disp median; share > 0.5; entry err | 2.96; 0.97; 1.05 | 0.27; 0.41; 0.19 | **0.12; 0.28**; 0.26 |
| L2b (b) at the entry / 20 before | closed-loop disp median; share > 0.5; share (b) - (a) > 0.3 | 2.74; 0.94; 0.14 / 4.09; 0.97; 0.42 | 0.26; 0.36; 0.08 / 0.23; 0.34; 0.11 | 0.07; 0.28; 0.11 / 0.25; 0.38; **0.28** |
| L3 closed loop (38 hazard-free seqs, 400 steps) | heading north at 30 real / model / agreement | 0.13 / 0.16 / 0.97 | 0.13 / 0.13 / 0.95 | 0.13 / 0.08 / 0.89 |
| L3 | far entry by 100 real / model / agreement | 0.16 / 0.18 / 0.97 | 0.16 / 0.13 / 0.97 | 0.16 / 0.08 / 0.92 |
| L3 | reach by 400 real / model / agreement | 0.42 / 0.87 / 0.55 | 0.42 / 0.82 / 0.55 | 0.42 / 0.76 / 0.55 |
| L3 | stalled in the last 100 real / model / agreement | 0.47 / 0.00 / 0.53 | 0.47 / 0.05 / 0.47 | 0.47 / **0.13** / 0.61 |
| original validation | val mse diag / off at the selected step (3 folds) | 0.0097-0.0137 / 0.028-0.033 | 0.0117-0.0158 / 0.031-0.036 | 0.0119-0.0165 / 0.031-0.037 |

## Arm C vs arm B: the crossover (`ett_crossover/v4c20/`, node 30027; B re-run on the same node as a device control, `ett_crossover/v4b_n27/`: IDENTICAL to B's node3 numbers to the third decimal -- the crossover is device-deterministic, the 3090 / 4090L difference above concerns the acceptance's L3 only)

| quantity | sim | v4 | B (K 10) | C (K 20) |
|---|---|---|---|---|
| Q1 reset CF + CF: xy error at 10 / 20 / 30 (heading agreement) | | 0.44 / 1.07 / 1.68 (0.69) | 0.30 / 0.78 / 1.30 (0.69) | 0.34 / 0.87 / 1.42 (0.70) |
| Q1 cf_early CF + CF: at 30 (agreement) | | 1.01 (0.94) | **0.53** (0.94) | 0.63 (0.95) |
| Q1 reset start + start: at 30 (no-regression check) | | **0.28** | 0.45 | 0.59 |
| reset CF + CF: success / timeout / completion given far / timeout given far / length | 0.53 / 0.23 / 0.67 / 0.33 / 453 | 0.57 / 0.12 / 0.82 / 0.17 / 396 | 0.60 / 0.15 / 0.75 / 0.20 / 466 | **0.51 / 0.25 / 0.62 / 0.33 / 484** |
| reset start + CF: same | 0.43 / 0.17 / 0.68 / 0.32 / 359 | 0.42 / 0.03 / 0.87 / 0.08 / 241 | 0.46 / 0.04 / 0.83 / 0.09 / 311 | **0.39 / 0.16 / 0.65 / 0.32 / 365** |
| reset CF2 + CF: same | 0.51 / 0.33 / 0.61 / 0.39 / 534 | 0.71 / 0.11 / 0.86 / 0.12 / 435 | 0.57 / 0.27 / 0.62 / 0.30 / 540 | 0.54 / 0.28 / 0.62 / 0.34 / 513 |
| indep_reset CF + CF: same | 0.44 / 0.28 / 0.57 / 0.43 / 447 | 0.39 / 0.22 / 0.56 / 0.37 / 389 | 0.51 / 0.11 / 0.76 / 0.16 / 374 | **0.45 / 0.27 / 0.61 / 0.37 / 494** |
| cf_early CF + CF: same | 0.45 / 0.27 / 0.61 / 0.39 / 427 | 0.41 / 0.14 / 0.67 / 0.27 / 305 | 0.51 / 0.11 / 0.72 / 0.18 / 361 | **0.39 / 0.23 / 0.58 / 0.39 / 438** |
| reset start + start: success / timeout / length (the start policy) | 0.23 / 0.03 / 148 | 0.23 / 0.01 / 134 | 0.23 / 0.08 / 189 | 0.20 / **0.15** / 242 |
| cf_early start + start: timeout / completion given far / length | 0.22 / 0.74 / 352 | 0.15 / 0.72 / 304 | 0.15 / 0.74 / 325 | 0.19 / 0.66 / 445 |
| start_early CF + CF: completion given far | 0.82 | 0.41 | 0.28 | 0.45 |
| reset (CF + CF) - (logged + CF), success | +0.302 | +0.392 (r 0.45) | +0.450 (r 0.27) | +0.388 (r 0.25) |
| reset (CF + CF) - (start + CF) | +0.102 | +0.156 | +0.135 | +0.123 (r 0.13) |
| reset (CF2 + CF) - (start + CF) | +0.078 | +0.288 | **+0.110** | +0.150 |
| reset (MF + CF) - (start + CF) | -0.073 | +0.036 | +0.016 | **-0.069** |
| reset (start + CF) - (start + start) | +0.196 | +0.184 | +0.230 | +0.192 |
| reset (CF + CF) - (CF + start) | +0.296 | +0.326 | +0.343 | +0.276 |
| indep_reset (start + CF) - (start + start) | +0.124 | +0.018 | +0.235 | +0.140 |
| indep_reset (CF2 + CF) - (start + CF) | +0.117 | +0.193 | +0.194 | +0.091 |
| cf_early (CF + CF) - (start + CF) / (MF + CF) - (start + CF) | -0.013 / +0.033 | -0.022 / -0.062 | **-0.039** / -0.100 | +0.065 / +0.085 |
| start_early (CF2 + CF) - (start + CF) | -0.058 | -0.019 | +0.017 | **-0.026** |
| reset (CF + CF) - (start + CF): near-2.0 / goal-area mass | +0.019 / +0.006 | +0.047 / +0.035 | +0.037 / +0.029 | **+0.011 / +0.008** |
| per-state correlation / sign agreement of the contrasts (all groups) | | <= 0.45 / 0.4-0.8 | <= 0.3 / chance | <= 0.3 / chance (unchanged) |

## Reading against the user's stop rule (closed-loop reach / stall deviation and same-state action contrasts: clear improvement or not)

* Open loop and the stall entries (fixed actions): C is the best model so far -- drift 0.33 / 0.64 at 50 / 100 (B 0.43 / 1.04, v4 0.81 / 2.04), the
  trajectory-start stall segments 0.23 with none over 0.5 (B 0.44, 25 %), the entry check 0.06 (B 0.17); the one-step accuracy and the original
  validation at B's level.  The longer window is what fixed the fixed-action accumulation that (a) / (b) identified.
* Closed-loop stall under the CF loop -- two readouts disagree in size.  (i) The crossover (64 states x 32 paired hazard draws per corner, full
  episodes, device-controlled): C reproduces the simulator's timeouts, completion-given-far and timeout-given-far at reset, indep_reset and
  cf_early within 0.02-0.06 (B off by 0.10-0.17, v4 by 0.10-0.24), and the success rates and path lengths follow (reset CF + CF 0.51 / 484 vs
  0.53 / 453) -- a clear improvement.  (ii) The acceptance's L3 (38 hazard-free sequences, strict "stationary in the last 100 of 400 steps"):
  stalled 0.13 vs B 0.05 and real 0.47, reach 0.76 vs 0.82 and real 0.42 -- the right direction, but a step no larger than the 3090 / 4090L
  device difference measured for B (0.14), and the deviation to the real sequences remains 0.34 on both.  What C fixed is the long-horizon
  meander / timeout under hazards, not the strict stationary stall inside 400 steps.
* Same-state action contrasts: the coarse means are closer to the simulator in 9 of the 13 listed contrasts (MF's sign at reset now right,
  -0.069 vs -0.073; CF2's sign at start_early right; the reset goal-mass contrasts within 0.005 of the simulator where B was 2-5x too large);
  worse in 4 (CF2 at reset +0.150 vs +0.078; cf_early CF / MF).  The per-state correlations (<= 0.3) and sign agreements (chance) are unchanged:
  no model in this chain ranks first steps per state.
* Costs: the start policy regresses further -- early error at reset 0.59 (B 0.45, v4 0.28), timeouts under start + start at reset 0.15 vs the
  simulator's 0.03 (path length 242 vs 148), cf_early start + start length 445 vs 352 -- C adds stalls the start agent does not have (the
  "normal walking preserved" condition, met in round 1, is NOT met by C); the early error under the CF loop is slightly worse than B's (1.42
  vs 1.30 at reset, 0.63 vs 0.53 at cf_early); L3's heading / far-entry agreement drops to 0.89 / 0.92 (B 0.95 / 0.97); from 20 rows before
  the stall entry the actor's feedback now matters more (0.28 of the segments worse than the recorded actions by > 0.3; B 0.11); and the K 20
  unroll at weight 1 diverged in two of three folds (pre-spike checkpoints selected).
* Verdict under the rule: a clear improvement on ONE closed-loop readout (the crossover's stall / timeout reproduction) and on the coarse
  action contrasts; no clear improvement on the strict closed-loop stall (L3) or on the per-state contrasts; a regression on the start policy.
  This does not clear the bar for re-running the CRL critic + actor (the per-state judge is unchanged and the start policy got worse), and under
  the stop point set for the evening of 2026-09-22 the recommendation is to STOP this week's AntMaze model revision here, with C recorded as
  the reference ETT for the CF-loop stall reproduction and its two limits (instability at K 20; the start-policy regression) recorded with it.
  No CRL seeds are launched.  Everything beyond one window change (weight, gradient clipping, a soft gate, stall-covering windows) is outside
  this round by the user's rule and is left as a list, not a plan.

## The full rollout acceptance for C (user's request after the crossover reading; running)

`diag_v6_ett_rollout.py --model-dir ett_one_step_v4c20 --tag v4c20` (outputs `ett_rollout_v4/v4c20/`): the Step 3a acceptance with the
SEALED thresholds (6,000 held-out anchors, strata = anchor region, rate gaps <= 0.05, entered-far <= 0.03, KS <= 0.15, gated in strata
with >= 200 anchors; advice C = the simulator teacher along the model path, advice B = the learnt generator v3 with 4 paths per anchor;
continuation = the start agent's mode, first query = the logged torque), plus the DECISION-STATE anchor set (`--anchor-set decision`:
every reset anchor (196) and up to 700 start_early / pre_zone1_early anchors per fold under the crossover's group definitions, each
rolled with its held-out fold's model; the reset group is below the sealed minimum and is reported, not gated) and a far-completion
column (reach given entered far; reported, not gated).  The v4 reference under the same gates (`ett_rollout_v4/hist_exact/`): pooled
pass; pre_zone1 fails on reach (0.41 vs 0.32) and timeout (0.065 vs 0.178), between on timeout, zone2 on the death-time KS.
User's rule: if start_early / pre_zone1_early are still far outside the thresholds, no CRL training.  Jobs: node3 (C uniform, B
decision folds 0-1), node 30027 (B uniform), node 30049 (C decision, B decision fold 2), ~70 min.

## The full rollout acceptance for C: RESULT (`ett_rollout_v4/v4c20/REPORT.md`; the v4 reference `ett_rollout_v4/hist_exact/`)

Sealed gates (Step 3a): rate gaps <= 0.05 (entered-far <= 0.03), KS <= 0.15, on the pooled sample and in every stratum with >= 200
anchors; continuation = the start agent's mode, first query = the logged torque, one path per anchor under advice C (the simulator
teacher along the model path) and 4 sampled paths under advice B (the learnt generator v3); the hazard integrated exactly along the path.
Wall: C 7-8 min per fold of 2,000 anchors, B 21-25 min (20 workers sharing one GPU; 12 on node 30049); three nodes, 85 min in all.

### Uniform held-out anchors (6,000; strata = anchor region) -- death / reach / timeout, sim vs model; the verdict per stratum

| stratum | anchors | sim | v4 (C) | C arm (C) | v4 (B) | C arm (B) |
|---|---:|---|---|---|---|---|
| all | 6000 | 0.294 / 0.615 / 0.091 | 0.304 / 0.649 / 0.047 pass | 0.306 / **0.665** / **0.029** FAIL reach, timeout | 0.310 / 0.640 / 0.050 pass | 0.310 / 0.661 / **0.029** FAIL timeout |
| start | 471 | 0.724 / 0.248 / 0.028 | 0.724 / 0.252 / 0.023 pass | 0.722 / 0.264 / 0.015 FAIL KS reach time 0.165 | 0.718 / 0.263 / 0.019 pass | 0.716 / 0.271 / 0.012 pass |
| pre_zone1 | 1456 | 0.503 / 0.319 / 0.178 | 0.525 / 0.410 / 0.065 FAIL | 0.534 / **0.422** / **0.044** FAIL | 0.534 / 0.402 / 0.065 FAIL | 0.541 / 0.414 / 0.045 FAIL |
| zone1 | 580 | 0.426 / 0.550 / 0.024 | pass | 0.440 / 0.557 / 0.003 pass | pass | pass |
| between | 1366 | 0.283 / 0.616 / 0.102 | 0.295 / 0.667 / 0.039 FAIL | 0.295 / **0.686** / **0.019** FAIL (+ KS death x 0.151) | 0.308 / 0.645 / 0.047 FAIL | 0.301 / 0.678 / 0.020 FAIL |
| zone2 | 544 | 0.105 / 0.857 / 0.039 | KS death time / x FAIL | 0.107 / 0.876 / 0.017; KS 0.22 / 0.18 FAIL | KS FAIL | KS FAIL |
| post_zone2, goal_area | 879, 444 | 0 / 0.96 / 0.04 | pass | pass (timeout 0.018 vs 0.042 in post_zone2) | pass | pass |

Far completion (reach given entered far; reported, not gated): pooled 0.82 (C) / 0.81 (B) vs sim 0.87 (71 / 284 far entries); start
stratum 0.69 (n 14) / 0.77 (n 56) vs 0.93.

### The decision-state anchor set (every reset anchor + 700 start_early + 700 pre_zone1_early per fold; each rolled with its held-out fold model)

| group | anchors | death sim / C / B | reach sim / C / B | timeout sim / C / B | far sim / C / B | far completion sim / C / B (n sim far) | verdict |
|---|---:|---|---|---|---|---|---|
| all | 4396 | 0.737 / 0.738 / 0.737 | 0.246 / 0.253 / 0.255 | 0.017 / 0.010 / 0.008 | 0.020 / 0.018 / 0.018 | 0.80 / 0.74 / 0.84 (86 / 344) | pass |
| reset (not gated, 196 < 200) | 196 | 0.663 / 0.665 / **0.733** | 0.296 / 0.323 / 0.254 | 0.041 / 0.012 / 0.013 | 0.015 / 0.000 / 0.000 | 0.67 / - / - (3 / 12) | C: KS reach time 0.285; B: death +0.07 |
| start_early | 2100 | 0.732 / 0.734 / 0.721 | 0.247 / 0.249 / 0.268 | 0.021 / 0.017 / 0.012 | 0.040 / 0.037 / 0.037 | 0.81 / 0.74 / 0.84 (83 / 332) | pass (both) |
| pre_zone1_early | 2100 | 0.748 / 0.748 / 0.754 | 0.241 / 0.249 / 0.241 | 0.010 / 0.002 / 0.004 | 0 / 0 / 0 | - | pass (both) |

### Reading

* The route-deciding groups are NOT far off: start_early and pre_zone1_early pass every sealed gate under both advice processes
  (death / reach / timeout within 0.01-0.02, entered-far 0.037 vs 0.040), the far completion at start_early 0.74 (C) / 0.84 (B) vs 0.81
  on 83 / 332 far entries; reset (below the minimum) matches on the rates under C (reach +0.027, timeout -0.029) with a failing reach-time
  KS (0.285), and under B over-predicts the death (0.733 vs 0.663) -- the advice process at the reset row, not the motion (C is fine there).
  But these groups are 73-75 % deaths under the start continuation: the outcome is decided by the frozen v4 onset head and the exact
  hazard integration within ~60 steps, so the motion model has little room to show its long-path error here.
* Where the long path matters the C arm is WORSE than v4 against the sealed gates: pooled it fails (reach 0.665 vs 0.615, timeout 0.029
  vs 0.091; v4 0.649 / 0.047 passed), pre_zone1 (timeout 0.044 vs 0.178; v4 0.065) and between (0.019 vs 0.102; v4 0.039) fail by a
  wider margin, post_zone2 timeouts halve.  Under the START agent's torques the C arm walks through the start agent's stalls before
  the mouth and in the corridor MORE than v4 did, while under the CF actor's torques it reproduces the CF actor's stalls (the crossover).
  The stall behaviour learnt from the CF rollouts is therefore the CF actor's stall pattern, not a policy-independent property of the
  dynamics; the multi-step term moved the model towards the policy whose sequences it was trained on and away from the sealed table's
  continuation.
* Against the user's rule: start_early / pre_zone1_early do not block CRL training by themselves.  Against the Step 3a rule sealed with
  these gates ("C fails -> fix the motion / onset model first"), the C arm's futures would not be generated -- and v4's already failed the
  same strata.  Together with the crossover's start-policy regression and the unchanged per-state contrasts, the recommendation stands:
  no CRL training from the C arm; stop this week's model revision here.  Note for later: a model that must serve BOTH continuations (the
  sealed start-agent table and the CF actor's own futures) needs supervision from both policies' sequences, or one model per
  continuation -- neither is a "one window change" and both are outside this round.
