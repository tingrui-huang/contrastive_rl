# G0 prescreen: an absorbing Manski freeze channel flips the fork preference

Date: 2026-09-15. Commit `a52c0a5a42d4b836821e462fb2d5b411732aa316`. Sealed [PROTOCOL.md](PROTOCOL.md); code [run.py](run.py); numbers [results.json](results.json), [freeze_tables.json](freeze_tables.json), per-root arrays `per_root.npz`. Strictly offline: 16,141,312 model transitions (cap 20,000,000), zero environment steps, zero training updates, no hidden bits, death labels or teacher-mode labels read. 103 s on CPU.

## Decision

**G0 passes.** With the same frozen actor, nominal and diagonal, replacing the pessimistic XY response by an absorbing Manski freeze channel makes model continuations from the 656 held-out first-fork contexts prefer the safe detour: down-minus-right discounted return **+4.279 [4.175, 4.377]** (root bootstrap), goal-reach ratio down/right **4.13** (threshold 1.5). The eligible 09-15 ETT is action-blind on the same roots (+0.023; identical goal-reach 0.900/0.900), and the observational freeze alone does not flip (−1.645). G1 (the sealed O/P production pipeline with this ETT) is justified.

## Visible freeze statistics (train episodes, obs/act only)

| landing cell | onsets | absorbing (run to episode end) | in `S_abs` |
|---|---:|---:|---|
| (3,3) | 612 | 1.000 | yes |
| (5,3) | 361 | 1.000 | yes |
| (4,3) | 338 | 1.000 | yes |
| (0,3) | 340 | 0.009 | no (corner press, median run 1) |
| (1,1) | 44 | 0.045 | no |
| (7,1), (8,3) | 2, 2 | — | no (below the 10-onset minimum) |

Persistence given the full frozen F4 signature: 1.000 on 45,620 rows in the three support cells; 0.09 on 11 rows at (0,3). The support is discovered from the data's absorbing runs, not from the map.

Descriptive per-step Manski upper bound `pi_B q_obs + (1 - pi_B) 1[landing in S_abs]` by landing bin (not a model input):

| current cell | landing bin | population | pi_B | q_obs | upper bound |
|---|---|---|---:|---:|---:|
| holding (2,3) | enter (3,3) | eligible teacher | 0.682 | 0.000 | **0.318** |
| holding (2,3) | enter (3,3) | all train | 0.546 | 0.062 | 0.488 |
| holding (2,3) | wait (2,3) | either | 0.32–0.36 | 0.000 | 0.000 |
| fork (1,3) | right (2,3) / down (1,2) | either | — | 0.000 | **0.000 / 0.000** |
| swamp (3,3) | continue (4,3) | eligible teacher | 0.665 | 0.006 | 0.339 |
| swamp (3,3) | stay (3,3) | eligible teacher | 0.242 | 0.040 | 0.768 |

The teacher-population bound at the entry decision (0.318) is close to the true per-cell activation 0.30; the confounded observational rate there is 0.000–0.062. Neither fork branch is penalised; pessimism bites one step later, only on the shortcut.

## Model arms, 656 roots x 64 paths x 2 forced first actions

| arm | return down | return right | down − right [95% CI] | reach d/r | absorbed d/r | lower route d/r |
|---|---:|---:|---|---:|---:|---:|
| A0 eligible_response (09-15 final ETT) | 10.300 | 10.277 | +0.023 [+0.012, +0.034] | 0.900 / 0.900 | 0 / 0 | 0.027 / 0.026 |
| A1 diagonal_motion (no freeze) | 7.248 | 10.142 | −2.894 [−2.967, −2.826] | 0.911 / 0.978 | 0 / 0 | 0.898 / 0.018 |
| A2 + observational absorbing atom | 7.098 | 8.743 | −1.645 [−1.722, −1.575] | 0.871 / 0.743 | 0.074 / 0.291 | 0.898 / 0.017 |
| **A3 + Manski onset** | **6.525** | **2.246** | **+4.279 [+4.175, +4.377]** | **0.769 / 0.186** | 0.175 / 0.819 | 0.897 / 0.015 |

Native reference from the matched-fork diagnostic (different actor C1, 16 roots, not comparable in level): down 10.21 vs right 3.35, gap +6.86.

Attribution:

- **A0 is action-blind.** The 09-15 final response blocks have Frobenius norms 0.03–0.08, so a forced down `(0,−1)` versus right `(1,0)` moves the next XY by 0.023 on average (max 0.033); the "down" first step lands at y = 3.53, above the root. The 09-15 P positives therefore could not encode a route counterfactual regardless of the freeze question.
- **A1** shows the diagonal queried at the executed action navigates both routes (down reaches the goal 0.911; the detour is entered on 0.898 of down paths) and, without any freeze, prefers the shortcut by the discount alone.
- **A2 − A1**: the confounded observational atom absorbs 0.291 of right paths and moves the gap by +1.25, not enough to flip.
- **A3 − A2**: the Manski onset adds +5.92 to the gap. Right paths land in a support cell while alive 0.988 of the time and are absorbed 0.819 of the time; down paths are absorbed 0.175 (0.10 never take the detour and go right; the rest drift into (5,3) after reaching the goal under the fixed horizon).

## What this does and does not establish

Established, within the model: an ETT whose off-diagonal pessimism lives in an absorbing freeze channel bounded by the per-step Manski upper bound produces a large, root-bootstrap-supported preference for the safe detour, with the same actor, nominal, diagonal, roots and random streams as the arms that do not. The support set, persistence and bins are all visible functions of obs/act.

Not established: native outcomes (no environment call), critic or actor response (no training), calibration of the bound (it is an upper bound: right survives 0.18 here versus 0.34 for a blind native shortcut, partly because the diagonal motion lingers in hazard cells, a defect the matched-failure audit already recorded), or anything about the AntMaze response family. The alive-motion law assumes the hidden bits act only through freeze; this is stated, not tested here.

## Next: G1

Run the sealed 09-15 O/P pipeline (`ett/pointmaze_offline_causal_integration.py`) with arm A3 as the P transition, unchanged critic/actor objectives and budgets, and read the 656-root endpoint down-minus-right gap (initial −0.60, O −0.31, P-with-eligible-ETT −0.09). Pass = gap positive with CI excluding zero. Then G2 (actor extraction with bc 0.5) and G3 (preregistered native confirmation). Nothing here was committed or pushed.
