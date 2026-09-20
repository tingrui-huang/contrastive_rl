# A second hazard draw of the simulator branch table (2026-09-20, user's go after the hybrid control): the oracle gain does NOT replicate under a fresh future-table draw -- CF2 0.351 (rule not met), the hybrid sits at the same level

`scripts/exp_v6_oracle_draw2.py generate / seal / train / evaluate / report` (node3; generation 18 workers, 18 min).  `manifest.json`,
`generation_draw2.json`, `REPORT.md` / `report.json`, per run `CF/seed_*/{train_manifest.json, eval_mean_s8909.json}`.  Everything of the
sealed table's construction kept (anchors, the logged torque once from the restored logged state, the start agent's mode closed-loop
in the simulator, termination reach / death / horizon 800 - t, no filtering); only the hazard / clock / jitter seeds differ
(232_000_000 + anchor_id vs 132_000_000 + anchor_id).  Same five training seeds, recipe (critic clip 0.1, BC 0.05, 30k updates), draw 8909.
Nature (user's rule, fixed before stage 2): a REPLICATION CHECK of the oracle reference under the generation randomness that the five
training seeds of one table never covered; two tables do not estimate a variance; no equivalence threshold.

## The second table is the first table in every marginal

Pooled success / death / timeout 0.620 / 0.292 / 0.088 vs 0.621 / 0.291 / 0.088; every region within 0.006; mean rows 143.4 vs 143.3;
death-time KS 0.008 pooled (zone1 0.022, between 0.015, zone2 0.037); length KS 0.002; all 3,338 anchors inactive in both draws give
row-identical paths.  Per-anchor outcome agreement between the two draws 0.786 -- the same as hybrid vs sealed (0.784): the
5,714 / 5,257 outcome flips of the hybrid are the size of an ordinary re-draw of one future per anchor.  (The hybrid's death-time
deviations, KS 0.085 / 0.054 / 0.171 in zone1 / between / zone2, exceed this re-draw noise 3-5x -- the one re-draw-calibrated fact
about the learned risk so far.)

## Result (success / detour / death / timeout; seed 8909, the same 300 episodes)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| CF2 (simulator, draw 2) | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.310 / 0.34 / 0.33 / 0.36 | **0.580** / 0.57 / 0.26 / 0.16 | 0.351 |
| CF (simulator, sealed draw) | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| hybrid (simulator motion + learned risk) | 0.357 | 0.370 | 0.187 | 0.313 | 0.477 | 0.341 |
| O (recorded) | 0.297 | 0.287 | 0.290 | 0.300 | 0.310 | 0.297 |

Paired (success): CF2 - O -0.017 / -0.053 / +0.060 / +0.010 / +0.270, mean +0.054, seed s.e. 0.057, 3 / 5 -- the mainline rule NOT met
under the second draw (sealed draw: +0.152, s.e. 0.024, 5 / 5).  CF2 - CF -0.243 / -0.237 / -0.087 / -0.097 / +0.173, mean -0.098, s.e.
0.076, 4 / 5 below.  hybrid - CF2 +0.077 / +0.137 / -0.163 / +0.003 / -0.103, mean -0.010, s.e. 0.055, 2 / 5.  Spread under the second
table 0.233-0.580 (sealed 0.407-0.523).

Failure modes of the CF2 policies are the same set the learned futures showed, seed by seed: seed 1 stalls before entering a route (139
of 300 episodes without a classified route, timeouts 0.50), seed 0 enters the far route and does not complete it (61 of 105 far-route
episodes time out; completion 0.41 vs 0.83 in seed 4), seeds 0 / 3 take the shortcut and die (shortcut share 0.54 / 0.45, deaths
0.39 / 0.33); seed 4 is the best single run of the whole study (0.580, far route 0.57 with completion 0.83).

## Reading (the user's three readings, none forced)

* This is reading (ii): CF2 comes down to the hybrid's level (means 0.351 vs 0.341; paired -0.010, 2 / 5) and the oracle gain does not
  survive the re-draw under the pre-registered rule (+0.054, 3 / 5).  Table-draw sensitivity comes first.
* Restatement required: the mainline oracle result (CF - O +0.152 at five seeds, 5 / 5; earlier +0.216 at three seeds, the seed-4909
  confirmation) is the result under ONE particular hazard draw of the future table.  Its "5 / 5 seeds" and seed s.e. 0.024 covered the
  training and evaluation randomness only; with the generation randomness included, the two draws differ by 0.098 in the mean and the
  per-seed results by up to 0.24.  Every comparison made against the single sealed draw (the learned-ETT arms, the hybrid, the
  absorbing law, the query-coverage stages) inherits this: "below the oracle" meant "below one draw".
* Consequently the hybrid's shortfall (-0.108 vs the sealed draw) is NOT evidence against the learned risk: against the second draw it
  is -0.010.  The learned futures' failure modes (stalls before a route, far-route incompletion, shortcut deaths) all appear under a pure
  simulator re-draw.  What the learned risk demonstrably does differently is the death timing (KS 3-5x the re-draw noise); whether that
  matters for the policy cannot be read from any single-draw comparison.
* Two tables cannot say which of the two draws is "typical"; the seed spread under draw 2 (0.23-0.58) says the recipe's outcome depends
  on which single future each anchor was given.
* Next (user's plan for this branch, needs the go): the equal-weight multi-future control from the two oracle tables -- anchor weight
  unchanged, one complete future of the anchor chosen uniformly first, then the geometric goal law within it; every outcome kept; no
  splicing, no change to NCE or the actor loss -- five seeds, draw 8909, compared with both single-draw arms.  Not: more learned-risk
  work against a single-draw reference.
