# Task-goal target support per candidate action (user's step after a408cef, 2026-09-20): the current law ranks correctly; the termination rule is NOT a found cause

`scripts/diag_v6_target_support.py run` on node5 (30 s; no rollouts, no
training).  `manifest.json` (judgement sealed before the numbers),
`REPORT.md` / `report.json`, `per_branch_s*.npz` (the per-(anchor, draw,
query) table: outcome class, length, masses under both laws; local copy in
this directory, not committed).

## Question

At the same state and task goal, does the truly better candidate action --
the one whose future reaches the task goal -- receive higher actual NCE
target support (the probability that the critic's positive, drawn by the
stream's own law `P(m) ~ 0.999^m` truncated at the path's end, lands at the
task goal)?  And does the current termination rule (a future ends on its
reach / death frame; each path normalised over its own rows) weaken or
reverse that advantage compared with an explicit ABSORBING alternative
(after reach or death the actual terminal position is held to the anchor's
remaining horizon 800 - t; timeouts unchanged; no relabelling)?

Data: the stage-1 paired query futures (q0 logged torque, q1 mode, q2..q5
samples of the lineage agent; two hazard draws paired across the six
queries) at the 4,276 start-region anchors, per lineage.  Task-goal sets:
reach 0.5 (the success set, primary), within 1 / 2, goal_area.  Marginals
(the NCE negatives' law) from the sealed CF table and the recorded futures
over all 53,747 anchors.

## Result (reset rows, anchor-weighted; the other strata in REPORT.md agree)

Per query at the reset rows (lineage 0 / 1 / 2): the logged torque succeeds
0.20 / 0.18 / 0.21 (far completion 0.04 / 0.03 / 0.03) and the agent's mode
0.50 / 0.48 / 0.46 (0.42 / 0.39 / 0.30); their expected reach-0.5 mass under
the CURRENT law is 0.00059 / 0.00058 / 0.00065 (logged) vs 0.00091 /
0.00094 / 0.00106 (mode) -- ordered like the success rates, at a scale of
0.1 % of the positive draws (marginal 0.019).

R4 (PRIMARY): far-going vs short-going queries within the same anchor
(165 / 148 / 107 reset anchors with both classes):

| lineage | success far / short | reach mass CURRENT far / short (ratio) | sign agreement per anchor | reach mass ABSORBING far / short (ratio) | sign agreement |
|---|---|---|---:|---|---:|
| 0 | 0.709 / 0.167 (4.2x) | 0.00110 / 0.00052 (2.1x), delta CI [0.0004, 0.0007] | 0.863 | 0.199 / 0.091 (2.2x), CI [0.081, 0.133] | 0.851 |
| 1 | 0.739 / 0.142 (5.2x) | 0.00119 / 0.00050 (2.4x), CI [0.0005, 0.0009] | 0.857 | 0.216 / 0.084 (2.6x), CI [0.101, 0.161] | 0.864 |
| 2 | 0.841 / 0.173 (4.9x) | 0.00145 / 0.00056 (2.6x), CI [0.0007, 0.0011] | 0.896 | 0.269 / 0.098 (2.7x), CI [0.133, 0.208] | 0.915 |

R3 (rank agreement across the six queries, success rate vs expected reach
mass; reset): concordance 0.938 / 0.944 / 0.953 under the current law,
0.942 / 0.955 / 0.959 absorbing; top-1 agreement 0.881 / 0.918 / 0.924 vs
0.891 / 0.936 / 0.959.  Over all start anchors 0.98-0.99 both laws.

R2 (within-(anchor, draw) pairs): every success-vs-non-success pair has the
success on top under both laws (a non-success has zero reach mass).  A
far-route success vs a shortcut success of the SAME (anchor, draw): the
far success carries about half the reach mass (ratio 0.51 / 0.49 / 0.51
current, 0.53 / 0.53 / 0.52 absorbing; P(far > short) 0.05 / 0.01 / 0.02
under both) -- the discount's speed penalty, identical under both laws.

**Judgement (manifest rule).**  J1 ("the current rule weakens or reverses
the good action's advantage") is NOT met in any lineage: the far-going
queries succeed 4-5x more often and their expected task-goal support under
the current law is 2.1-2.6x higher with a CI above zero and 0.86-0.90
per-anchor sign agreement; the absorbing law scales every mass by ~180x
(0.001 -> 0.2; marginal 0.019 -> 0.49) but leaves the ordering, the ratio
(2.2-2.7x) and the sign agreement (+-0.02) where they are.  So, by the
user's rule, the termination rule is not a found cause: the target law
already gives the correct ordering at the reset rows and along t in [1, 30);
the problem stays in how the learner / policy update uses this target.  The
hypothesis is closed; no O / CF training with a changed termination law.

What the compression is: the success ratio (4-5x) becomes a 2-2.6x mass
ratio under BOTH laws because a far-route success is ~2x longer and the
positive law discounts it (the far success vs shortcut success pairs
above).  That is the gamma 0.999 objective, not the termination rule.

Observation outside the rule (a count, not a proposal): at the reset rows
(anchor weight 0.0038, ~3.9 rows per critic batch of 1,024) the current law
puts 0.1 % of the positives at the task goal, i.e. about one task-goal
positive from a reset row every ~250 batches (~120 over a 30k run); the
absorbing law would give ~0.6 per batch.  The ordering the critic must
learn at these states is carried by very few direct samples and otherwise
by generalisation across goals and states (the learned critics' reset-row
AUROC was 0.58-0.67, see query_coverage/train).

Limits: the queries' continuation is the lineage agent (the sealed
branches' is the start agent); 107-169 reset anchors carry both route
classes; two hazard draws per query.
