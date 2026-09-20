# Query coverage, stage 1: what the current policy's own first torque adds at the training start anchors

`manifest.json` (sealed before generation; gates G1-G3 pre-registered),
`REPORT.md` / `report.json` (three lineages), `generation_s{0,1,2}.json`;
the branch files `queries_s{s}.npz` (1.25 GB each; node5) hold every path.
Script `scripts/exp_v6_query_coverage.py`.

## Question and design (user's plan after the real-update review)

The mainline critic sees, at every anchor, ONE future whose first torque is
the LOGGED one; the policy's alternative actions never enter the NCE.  Is
that missing supervision worth anything?  Measured before any training:
at every pilot anchor in the start region (4,276 anchors, 7.9 % of the
anchor weight; t median 10, 99 % t <= 30), six first torques -- q0 the
logged one, q1 the lineage agent's mode, q2-q5 four pinned samples -- each
executed once from the restored logged state and continued by the SAME
lineage agent (mode) to reach / death / horizon, under the logged task
goal and two paired hazard draws; nothing filtered.  Anchor-weighted
shares; the extended union spreads an anchor's weight equally over its six
queries; deltas paired by anchor.

## Result: the gate fails pooled; the effect is real and sits at the reset rows

| lineage | completed far: logged / mode / samples / extended | extended - logged (95 % CI) | G1 (>= +0.03) | G2 | G3 (distinct anchors, draw 0 / 1) |
|---|---|---|---|---|---|
| 0 | 0.026 / 0.046 / 0.044 / 0.041 | +0.015 [+0.012, +0.019] | not met | pass | 377 / 378 |
| 1 | 0.035 / 0.057 / 0.054 / 0.051 | +0.016 [+0.012, +0.021] | not met | pass | 389 / 389 |
| 2 | 0.033 / 0.048 / 0.044 / 0.043 | +0.010 [+0.006, +0.013] | not met | pass | 367 / 363 |

**Gate NOT PASSED** (G1 in all three lineages).  Success rises 0.180 ->
0.193, 0.185 -> 0.199, 0.220 -> 0.229; deaths fall by 0.01-0.02; the
goal-area mass of the critic goal marginal is unchanged (G2).  The two
hazard draws agree on 99.2-99.5 % of the far-complete indicators (Jaccard
0.86-0.89).

Where the effect is (completed far, logged -> extended, mode in brackets):

| stratum | n | weight | lineage 0 | lineage 1 | lineage 2 |
|---|---|---|---|---|---|
| reset rows (t = 0) | 196 | 0.0038 | 0.037 -> 0.330 (0.423), +0.29 [+0.25, +0.33] | 0.031 -> 0.301 (0.390), +0.27 [+0.23, +0.31] | 0.026 -> 0.211 (0.302), +0.18 [+0.15, +0.22] |
| t in [1, 30) | 4,037 | 0.0749 | 0.020 -> 0.022, +0.002 [-0.001, +0.004] | 0.029 -> 0.033, +0.004 [+0.001, +0.007] | 0.029 -> 0.030, +0.001 [-0.002, +0.004] |
| logged shortcut episodes | 4,071 | 0.0758 | 0.002 -> 0.018 | 0.003 -> 0.020 | 0.003 -> 0.015 |
| logged detour episodes | 205 | 0.0037 | 0.529 -> 0.515 (CI spans 0) | 0.683 -> 0.700 (spans 0) | 0.654 -> 0.624 (spans 0) |

At the reset rows the success rate under the agent's own first step is
0.425 / 0.400 / 0.374 vs 0.198 / 0.176 / 0.205 under the logged one: the
alternative first step produces more COMPLETE successes, not only more
far-route entries.  One step into a shortcut episode, neither the logged
nor the agent's own torque leads to the far route (this holds for these
states, these candidates and this fixed continuation policy; it is not a
statement that the route can never be changed later).

The user's read-out -- where the mode's own branch does NOT complete the
far route, does another query? (reset rows, draw 0 / draw 1):

| lineage | mode not far-complete | of which a sample far-complete | mode not success | of which another query success |
|---|---|---|---|---|
| 0 | 110 / 113 | 70 / 73 | 92 / 99 | 62 / 71 |
| 1 | 122 / 121 | 67 / 66 | 105 / 98 | 58 / 57 |
| 2 | 134 / 137 | 39 / 42 | 105 / 103 | 38 / 41 |

So the query set is not only restating the current policy's ability: at
30-64 % of the reset states where the mode fails, one of the four samples
completes the far route (the logged torque almost never does), i.e. it
offers improvement opportunities beyond the mode.

## Reading (user's, 2026-09-20)

* The place where a different first action changes the outcome is found:
  the current policy's reset states.  Both "can an 8-dim torque change the
  outcome" and "can the generator supply that supervision" are answered
  positively there.
* G1 averaged the effect away: the reset rows are 4.8 % of the start-region
  weight, so +29 points there is +1.4 points pooled -- almost the whole
  pooled gain.  G1 stays "not met" (no post-hoc pass), but it does not
  judge the training value of these queries; a reset anchor is still drawn
  ~117k times in 30k x 1024 batches.  Whether that is enough, or cancelled
  by other gradients, is a training question.
* What this stage proved is "the current policy's action beats the logged
  action at the reset states", NOT "training on these queries beats the
  current policy".  Stage 2 (`train/`) is the added, explicitly
  exploratory training comparison that keeps the current policy as a
  reference.
