# The equal-weight multi-future control (user's plan after oracle_draw2; go 2026-09-20 night): two complete futures per anchor -- reading (c): the spread narrows around a LOWER level (0.311); the policies take the shortcut like the start agent

`scripts/exp_v6_multi_futures.py check / seal / train / evaluate / report` (node3; `exp_v6_mainline_pilot.MultiFutures`, `train_arm(branch_path=[...])`).
`manifest.json`, `sampler_check.json`, `REPORT.md` / `report.json`, per run `CF/seed_*/{train_manifest.json, eval_mean_s8909.json}`.

## Construction (as specified)

Anchor drawn by its weight, unchanged.  Then ONE of the two oracle tables (the sealed draw, draw 2) is chosen uniformly, and the goal row
follows the original geometric law (P(m) ~ 0.999^m over 1..L_j - 1) within that table's path.  Every outcome kept, no splicing, NCE and actor
losses, recipe (critic clip 0.1, BC 0.05, 30k updates, the start agent), streams and seeds unchanged.  The one future uniform is split
(j = floor(2u), u' = 2u - j), so the anchor sequence is identical to the single-table arms.  `sampler_check.json` on the real tables:
table share 0.498 / 0.502, m median 31 / 32 (the single-table law), goals equal to the chosen table's rows, anchor sequence identical.
Training 565 s per run (52 upd/s).

## Result (success / detour / death / timeout; seed 8909, the same 300 episodes)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | min .. max |
|---|---|---|---|---|---|---|---|
| MF (both tables) | 0.387 / 0.26 / 0.47 / 0.15 | 0.250 / 0.04 / 0.66 / 0.09 | 0.337 / 0.12 / 0.60 / 0.07 | 0.267 / 0.13 / 0.58 / 0.16 | 0.317 / 0.22 / 0.51 / 0.17 | 0.311 | 0.250 .. 0.387 |
| CF (sealed draw) | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 | 0.407 .. 0.523 |
| CF2 (draw 2) | 0.280 / 0.35 / 0.39 / 0.33 | 0.233 / 0.17 / 0.26 / 0.50 | 0.350 / 0.23 / 0.48 / 0.17 | 0.310 / 0.34 / 0.33 / 0.36 | 0.580 / 0.57 / 0.26 / 0.16 | 0.351 | 0.233 .. 0.580 |
| O (recorded) | 0.297 / 0.02 / 0.68 / 0.02 | 0.287 / 0.01 / 0.71 / 0.01 | 0.290 / 0.01 / 0.70 / 0.01 | 0.300 / 0.03 / 0.67 / 0.03 | 0.310 / 0.06 / 0.63 / 0.06 | 0.297 | 0.287 .. 0.310 |
| hybrid | 0.357 | 0.370 | 0.187 | 0.313 | 0.477 | 0.341 | 0.187 .. 0.477 |

Paired (success): MF - O +0.090 / -0.037 / +0.047 / -0.033 / +0.007, mean +0.015, seed s.e. 0.024, 3 / 5 -- mainline rule NOT met;
MF - CF -0.137 (s.e. 0.023, 5 / 5 below); MF - CF2 -0.039 (s.e. 0.061, 3 / 5); MF - hybrid -0.029 (3 / 5).
Other keys (seed mean): MF - CF detour -0.267, death +0.232, timeout -0.095; MF - CF2 detour -0.178, death +0.219, timeout -0.180.

Routes: shortcut 0.67 / 0.95 / 0.84 / 0.83 / 0.72 (CF 0.29-0.62, CF2 0.36-0.72, O 0.87-0.97), detour 0.26 / 0.04 / 0.12 / 0.13 / 0.22, no
classified route 4-20 per 300 (CF 17-36, CF2 15-139), mean episode length 199-313 steps (CF 320-500, CF2 322-518, O 135-181); first mouth
arrival at step 51-53 like O (52), the single-draw arms 53-62.  The far route, when taken, is completed 0.65-0.91 (CF 0.73-0.86).

## Reading (the three readings written before the run; none forced)

* (c): MF sits at or below the lower draw (0.311 vs 0.351 and 0.449) and within 0.015 of O; the mainline rule is not met.  The seed spread
  did narrow (0.25-0.39 vs 0.41-0.52 and 0.23-0.58) -- but around the O / start-agent level, not around the better draw.
* Two complete futures per anchor did not average the two single-draw results (0.351 .. 0.449); they moved every seed toward the start
  agent's behaviour: the shortcut taken 0.67-0.95 of the time, no stalls (timeouts 0.07-0.17, no-route 4-20), deaths 0.47-0.66 where the
  hazard is active, the mouth reached at step 51-53 exactly as O.  Whatever produced the far-route preference in the single-draw arms is
  weakened, not strengthened, by giving each anchor a second independent future.
* What this does and does not establish.  It establishes that, under this recipe, more futures per anchor do not stabilise the gain (the
  user's stated question) and that the gain is not the mean-outcome signal that two draws estimate better than one.  It does NOT identify
  the mechanism; two hypotheses fit and are not separated here: (i) the single-draw far-route preference is carried by the
  idiosyncrasy of one random future per anchor (a critic exploiting particular (s, a) -> future pairings that a second future dilutes);
  (ii) the mixture law itself changes the critic's targets (the positive of a pair is drawn from two paths of different length and end,
  a different effective P(g | s, a) than either table alone even at the same expected outcome).  Two tables, one mixture, five seeds.
* Not concluded: that the oracle "gain" is noise; only that it is not reproduced by the second draw and not recovered by combining the two.
* Candidates for the user's decision (none started): a frozen-critic readout at the mouth anchors under CF / CF2 / MF (the action ranking
  far-going vs straight torques for the same states) to see what the single-draw critics rank that the two-table critic does not;
  or a third / fourth draw with the same recipe to place 0.449 and 0.351 within the draw distribution before any further ETT work.
