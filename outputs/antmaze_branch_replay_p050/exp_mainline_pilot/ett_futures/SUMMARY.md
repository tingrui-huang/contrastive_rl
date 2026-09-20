# The whole pipeline with learned-ETT futures (user's decision 2026-09-20): runs end to end; 2 / 5 seeds keep the oracle gain, 3 / 5 stall at the start; primary rule NOT met; the oracle gain re-established at five seeds

`scripts/exp_v6_ett_futures.py` (node3); `manifest.json` (sealed before
generation), `generation_ett.json`, `REPORT.md` / `report.json`, per run
`train_manifest.json` + `eval_mean_s8909.json` (ETT under `ett_futures/CF/`,
the oracle arms under `variants/critic_clip0.1/`; checkpoints and
`branches_ett.npz` on node3).  Oracle-supervised engineering stage: the
ETT's supervision (advice, onset labels, branch contexts, the context prior)
came from the simulator; the futures used for training are the model's.

## The pipeline

1. **Futures from the learned ETT** for all 53,747 anchors (126 s, batched):
   from the logged row, the logged torque once, then the start agent's mode
   closed-loop through the v3 motion / stationary-gate / onset models with
   the advice generator v3 (one hidden context per path from the prior,
   MAP hold, sampled torque, exact counters), the onset sampled step by step
   (one realised path per anchor, like one hazard draw), ending on reach /
   death / horizon 800 - t; anchor k uses the models of its own
   source-episode fold; the anchor's actual context is never read.
2. **Training**: the current mainline recipe (variant critic_clip0.1: start
   checkpoint, anchors, actor / BC rows from the logged buffer, bc 0.05,
   critic clip 0.1, 30k updates) with the critic futures = the ETT table
   (arm ETT), the sealed simulator table (CF) or the recorded continuation
   (O); seeds 0-4 (CF / O seeds 3-4 trained here, 0-2 = the existing finals).
3. **Evaluation**: all 15 finals and the start agent on one fresh paired
   draw (seed 8909, 300 episodes, policy mode).  Rule: mean over the five
   paired seeds > 2 x seed s.e. and 5 / 5.

## The learned futures vs the simulator table (anchor-weighted outcome shares)

Pooled: success 0.633 / 0.621, death 0.307 / 0.291, timeout 0.060 /
0.088, mean rows 128 / 143; start region 0.252 / 0.243, 0.720 / 0.728,
0.028 / 0.028.  Known deviations, all on long paths: pre_zone1 timeouts
0.047 vs 0.184 (the under-stall before the mouth; success 0.409 vs 0.312)
and the far-route legs, where the MODEL times out more than the simulator
(west column 0.29 vs 0.21, top corridor 0.29 vs 0.18, east column 0.12 vs
0.08): under the model the far route is a less reliable success than it
is in the simulator.

## Result (success / detour / death / timeout; seed 8909)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT (learned futures) | 0.283 / 0.24 / 0.15 / 0.57 | **0.450** / 0.34 / 0.29 / 0.26 | **0.417** / 0.47 / 0.18 / 0.40 | 0.167 / 0.20 / 0.11 / 0.72 | 0.137 / 0.14 / 0.24 / 0.62 | 0.291 |
| CF (simulator futures) | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O (recorded futures) | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 |
| start | 0.273 / 0.01 / 0.70 / 0.02 | | | | | |

Paired (success): **CF - O +0.227 / +0.183 / +0.147 / +0.107 / +0.097,
mean +0.152, seed s.e. 0.024, 5 / 5 -- rule MET at five seeds** (the two
new seeds carry the smaller gains); ETT - O -0.013 / +0.163 / +0.127 /
-0.133 / -0.173, mean -0.006, s.e. 0.067, 3 / 5 -- rule NOT met; ETT - CF
-0.240 / -0.020 / -0.020 / -0.240 / -0.270.  Deaths: ETT - O -0.48 (every
seed far below O's 0.63-0.71), CF - O -0.35.  O - start +0.023 (5 / 5).

## Reading

* The pipeline runs end to end: model-generated futures at every anchor,
  the unchanged learner, a policy.  The futures are close to the
  simulator's in outcome shares, and every ETT policy learns what the CF
  policies learn about the hazard (deaths 0.11-0.29 vs O 0.63-0.71,
  detour 0.14-0.47 vs O 0.01-0.06).
* Two seeds (1, 2) keep the oracle gain almost entirely (0.450 vs CF
  0.470; 0.417 vs 0.437).  Three seeds (0, 3, 4) stall: timeouts 0.57 /
  0.72 / 0.62, and 82-97 % of those timeouts never reach the first mouth
  and are not on the detour -- the actor does not leave the start.  The
  training metrics show no manifold exit (BC NLL -12 to -13.4, scale
  0.08-0.095, as CF's), so this is the route-choice stall seen before
  (frozen-critic lineage 2, absorbing CF seeds 0 / 2), here in 3 / 5
  seeds against 0 / 5 with the simulator futures.
* The plausible mechanism is in the futures' known long-path error: under
  the model the far route completes less and times out more than in the
  simulator (0.29 vs 0.18 on the top corridor), while the shortcut's death
  share is the simulator's; with both routes looking poor from the start
  region the actor's critic term has no reliable alternative and the
  policy stalls.  A hypothesis, not a demonstrated cause.
* Judgement: primary NOT met; the learned-ETT pipeline is not yet a
  substitute for the simulator futures.  The next ETT item is the
  long-path motion / onset error (the far-route legs and the under-stall
  before the mouth) -- the advice process is no longer the bottleneck.

Limits: one evaluation draw; the ETT table is one realised path per anchor
(as the simulator table); the context prior is the environment's design.
