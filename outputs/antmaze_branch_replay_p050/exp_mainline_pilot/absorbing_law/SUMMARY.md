# O vs CF under the absorbing future law (the sparse-supervision hypothesis, user's decision 2026-09-20): NOT supported

`scripts/exp_v6_absorbing_law.py` (node5); `manifest.json` sealed before
training (hypothesis, change, unchanged items, judgement), `REPORT.md` /
`report.json` / `readout.json`, per run `train_manifest.json` +
`eval_mean_s7909.json` (checkpoints on node5).  Oracle (simulator)
futures; a changed learning target, disclosed; registered as a NEW
hypothesis (target_support's ordering result stands).

## What was run

The current mainline (variant `critic_clip0.1`) with ONE change: the
critic's positive law, in both arms identically.  Truncated (sealed
recipe): P(m) ~ 0.999^m over m = 1..L-1.  Absorbing: P(m) ~ 0.999^m over m
= 1..H, H = 800 - t; rows m >= L-1 of a path that ended before the
horizon (reach or death) return its actual terminal row; a path that ran
to the horizon is unchanged; no relabelling
(`exp_v6_mainline_pilot.AbsorbingFutures`, `future_law` override recorded
in every train manifest; tail rows anchor-weighted mean O 532 / CF 525;
share of the positive draws on the terminal row O 0.757 / CF 0.759).
Same sealed start checkpoint, anchors, branch file, actor / BC rows, bc
0.05, critic clip 0.1, 30k updates, seeds and stream seeds (first-batch
anchor sequence asserted equal to the clip arms).  Evaluation on a fresh
paired draw (seed 7909, 300 episodes, mode) of the six new finals, the
six clip finals and the start agent.

## Result (success / detour / death / timeout)

| run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| O absorbing | 0.270 / 0.00 / 0.73 / 0.00 | 0.263 / 0.01 / 0.73 / 0.01 | 0.263 / 0.00 / 0.73 / 0.01 |
| O clip (current) | 0.263 / 0.01 / 0.71 / 0.03 | 0.273 / 0.01 / 0.73 / 0.00 | 0.283 / 0.02 / 0.71 / 0.00 |
| CF absorbing | **0.013** / 0.08 / 0.00 / **0.99** | **0.380** / 0.78 / 0.02 / 0.60 | **0.063** / 0.06 / 0.01 / 0.92 |
| CF clip (current) | 0.517 / 0.62 / 0.19 / 0.29 | 0.470 / 0.51 / 0.31 / 0.22 | 0.423 / 0.33 / 0.43 / 0.15 |
| start | 0.253 / 0.01 / 0.72 / 0.03 | | |

Paired (success): CF_abs - O_abs -0.257 / +0.117 / -0.200 (mean -0.113,
seed s.e. 0.116, 2 / 3 negative) -- rule NOT met; CF_abs - CF_clip -0.503
/ -0.090 / -0.360 (3 / 3 worse); the current law's CF_clip - O_clip on
the same episodes +0.253 / +0.197 / +0.140 (rule met again on this
draw); O_abs - O_clip -0.008 (O_abs = the start agent + 0.012).  Critic
read-out (within-anchor AUROC over the stage-1 candidates at the reset
rows): CF_abs 0.582 / 0.525 / 0.628 vs CF_clip 0.619 / 0.583 / 0.621 --
up in 1 / 3.  Judgement: not supported.

## Reading

1. The absorbing law does not make the critic rank unseen start
   candidates better (AUROC unchanged or lower) although ~180x more
   task-goal positives were drawn; the sparse-supervision hypothesis is
   not supported and, by the user's rule, tuning around the sampling
   rule stops here.
2. What the law does instead: under it 76 % of every positive draw is a
   TERMINAL position.  In the O arm nearly every future ends at the goal,
   so the positive is the goal for every (s, a) -- no contrast between
   actions -- and the O critic teaches nothing (O_abs = start).  In the
   CF arm the terminal is the death spot for 73 % of the start-region
   branches, the goal for the successes and the stall for the timeouts;
   the actor learns to avoid the death spot rather than to reach the
   goal: deaths -> 0.00-0.02 in all three seeds, timeouts 0.60-0.99, with
   the policy leaving the logged manifold (BC NLL -16 -> -8 / -11 / -12,
   policy scale 0.06 -> 0.09-0.14).  Seed 1 detours (0.78) but completes
   only half of it; seeds 0 and 2 do not leave the start.  So "more
   task-goal positives" came bundled with "death-position positives",
   and the learner used the latter.
3. This is a result about THIS law under THIS learner and budget; it does
   not say the truncated law is optimal, only that replacing it with the
   absorbing law is worse.  No further variants of the termination rule
   (user's instruction: stop if no improvement).

Limits: three seeds, one evaluation draw; the clip arms re-evaluated on
the same draw give the reference (+0.197, 3 / 3), so the comparison is
paired.
