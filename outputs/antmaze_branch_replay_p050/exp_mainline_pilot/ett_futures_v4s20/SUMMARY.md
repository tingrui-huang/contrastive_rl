# The pipeline with the protocol-matched ETT futures (arm S = v4 + K 20 windows from the sealed table's start-agent branches), 2026-09-21: three pre-fixed seeds at the first oracle table's level, the five-seed completion NOT met (one seed collapses into the start stall)

`scripts/exp_v6_ett_futures.py --ett v4s20 --seeds 0 1 2` (node3; then `--seeds 0 1 2 3 4`); `manifest.json`, `generation_ett.json`,
`REPORT_3seeds.md` / `report_3seeds.json` (the pre-fixed three), `REPORT.md` / `report.json` (five), per run `train_manifest.json` +
`eval_mean_s8909.json` (`ett_futures_v4s20/CF/seed_*`); the oracle CF (table 1), CF2 (table 2, `oracle_draw2/CF`), O and the start
agent re-used with their seed-8909 evaluations.  Same recipe (critic_clip0.1: bc 0.05, critic clip 0.1, 30k updates, start-checkpoint
init, the logged actor / BC rows), same anchors, same 300 evaluation episodes for every checkpoint.  The user's framing: O / Oracle /
Learned side by side, both oracle tables kept, the main reading = Learned - Oracle success, timeout and seed variation; three seeds fixed
first, five if worth confirming, every result kept.

## The ETT revision (cf_motion/SUMMARY.md ROUND 3, "arm S") in one paragraph

The user's diagnosis after the C arm (d8feadd): the multi-step loss used only the CF actor's sequences while the futures we generate are
start-agent continuations -- a training-distribution mismatch between two goals.  Arm S keeps the oracle's generation protocol fixed
(logged anchors, the logged first torque, the start agent's mode) and moves the K 20 windows onto that protocol: `fit --v4 --init-from
ett_one_step_v4 --freeze-onset --multistep 20 --multistep-source branch` (windows cut from `branches_cf.npz`'s own start-agent paths,
~4.0 M per fold, half starting on slow / static rows, every outcome kept, under each fold's episode-level split; the one-step
supervision the original; selection = original validation score + the K-step MSE on the early-stop episodes' windows).  Training stable
(no fold diverged); original validation +10-17 % like B / C.  Acceptance (`ett_rollout_v4/v4s20/`, sealed gates): under both advice
processes the pooled gates PASS (C: death 0.305 / reach 0.631 / timeout 0.063 vs 0.294 / 0.615 / 0.091; v4 0.649 / 0.047), `between`
passes (timeout 0.061 vs 0.102; v4 0.039 failed), `pre_zone1` fails on the timeout by 0.003 (0.125 vs 0.178; v4 0.065) and passes on
the reach (0.347 vs 0.319; v4 0.410 failed), zone2's death-time / x KS fail as always (the onset head is frozen v4); the decision groups
pass; reset (196, not gated) keeps a reach-time KS failure (C 0.170) and, under the learnt advice, the death over-prediction (B 0.738
vs 0.663) -- reported as errors, not passes.

## The revised futures vs the sealed simulator table (anchor-weighted; v4 futures in brackets)

Pooled success 0.639 [0.644] / sim 0.621, death 0.304 [0.306] / 0.291, timeout 0.057 [0.050] / 0.088, mean rows 133 [123] / 143.
pre_zone1 timeouts 0.128 [0.067] / 0.184 and success 0.338 [0.397] / 0.312 -- two thirds of the pre-mouth under-stall repaired;
between 0.050 [0.041] / 0.091 -- little change; the far legs now over-complete: west column 0.870 [0.779] / 0.789, top corridor 0.866
[0.745] / 0.822, east column 0.934 [0.905] / 0.920; post_zone2 timeouts 0.024 / 0.045.

## Result (success / detour / death / timeout; seed 8909, the same 300 episodes)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean (3) | mean (5) |
|---|---|---|---|---|---|---|---|
| Learned (S futures) | **0.567** / 0.58 / 0.14 / 0.30 | 0.463 / 0.60 / 0.01 / 0.53 | 0.387 / 0.45 / 0.01 / 0.61 | 0.433 / 0.56 / 0.09 / 0.47 | **0.047** / 0.06 / 0.01 / 0.94 | 0.472 | 0.379 |
| Learned (v4 futures, before) | 0.483 | 0.213 | 0.337 | 0.390 | 0.310 | 0.344 | 0.347 |
| Oracle table 1 (CF) | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.477 | 0.449 |
| Oracle table 2 (CF2) | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.310 / 0.34 / 0.33 / 0.36 | 0.580 / 0.57 / 0.26 / 0.16 | 0.288 | 0.351 |
| O (recorded) | 0.297 | 0.287 | 0.290 | 0.300 | 0.310 | 0.291 | 0.297 |
| start agent | 0.273 / 0.01 / 0.70 / 0.02 | | | | | | |

Paired (success; mean +- seed s.e., same direction):

| comparison | three pre-fixed seeds | five seeds | rule (five) |
|---|---|---|---|
| Learned - O | +0.181 +- 0.050, 3 / 3 (rule met) | +0.083 +- 0.091, 4 / 5 | NOT met |
| Learned - Oracle 1 | -0.004 +- 0.027 (+0.04 / -0.01 / -0.05) | -0.069 +- 0.074, 3 / 5 | -- |
| Learned - Oracle 2 | +0.184 +- 0.076, 3 / 3 | +0.029 +- 0.147, 4 / 5 | NOT met |
| Oracle 1 - O | +0.186 +- 0.023, 3 / 3 | +0.152 +- 0.024, 5 / 5 | met |
| Oracle 2 - O | -0.003 +- 0.033 | +0.054 +- 0.057, 3 / 5 | NOT met |
| Oracle 1 - Oracle 2 | +0.189 +- 0.051 | +0.098 +- 0.076, 4 / 5 | -- |

Other keys (five-seed mean of the paired difference): Learned - Oracle 1 timeout +0.35 (5 / 5 above), death -0.28 (5 / 5 below), detour
+0.03 (3 / 5).  Where the Learned actors fail: almost never a death (0.01-0.14 vs the oracle's 0.19-0.42); the far route completed at
the oracle's rate (0.67-0.86 vs 0.73-0.86); the START STALL -- episodes with no route (never leaving the start) 60 / 115 / 150 / 86 /
277 of 300 (oracle 1: 17-36; oracle 2: 15-139) -- the same failure mode as the v4-futures arm (63 / 188 / 90 / ...), now less frequent in
four seeds and total in the fifth.  Seed 4's training trace (critic 0.0065, actor 5.35, BC NLL -14.0 at 30k) is indistinguishable
from the other seeds'.

## Reading

* The protocol-matched revision is the first learned ETT whose futures reach the oracle's level: on the three pre-fixed seeds the
  Learned arm equals oracle table 1 (0.472 vs 0.477; -0.004 +- 0.027) and clears the pre-registered rule against O (+0.181, 3 / 3);
  on the same three seeds the v4 futures gave 0.344.  Seed 3 confirms (0.433 vs 0.407); seed 4 collapses (0.047, 94 % timeouts -- the
  actor does not leave the start), so the five-seed rule is NOT met (+0.083 +- 0.091, 4 / 5) and the five-seed mean (0.379) sits
  between the two oracle tables (0.449 / 0.351).
* The failure mode that remains is the seed-dependent start stall, not deaths: the Learned actors take the far route as often as the
  oracle's and complete it as often, but a fraction of episodes (20-50 %, one seed 94 %) never leave the start; the oracle-table
  actors show the same mode at 6-12 % (table 2: 5-46 %).  The futures that produced it pass every sealed rate gate except pre_zone1's
  timeout by 0.003 -- so whatever drives the start stall is not among the gated quantities (outcome shares and timing / position
  distributions by region); candidates, untested: the far-route over-completion of the model futures (0.87 vs 0.79 on the west column)
  making the detour goals look surer than they are, the start-region futures' goal positions rather than their outcome shares, and
  the seed-dependence of the actor's own start decision that the oracle tables also show (table 2 seed 1: 46 % stalls).
* The oracle is not a fixed yardstick: table 1 - table 2 = +0.10 +- 0.08 (five seeds; +0.19 on the first three), and their per-seed
  outcomes range 0.23-0.58.  Learned - O (+0.08) and Oracle 2 - O (+0.05) are of the same size; only oracle table 1 clears the rule.
* Not claimed: that the learned futures beat or equal the oracle in general (one of five seeds fails outright); that the start stall
  is a futures defect rather than a property of this recipe under any imperfect table (table 2 shows it too).  Pickles, rollouts and
  logs stay on the nodes.
