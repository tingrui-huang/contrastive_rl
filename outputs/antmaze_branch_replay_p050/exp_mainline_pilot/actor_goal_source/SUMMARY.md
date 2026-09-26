# The actor goal-source comparison (user's step 2, 2026-09-21): closed, negative

`manifest.json` (sealed before training), `REPORT.md` / `report.json`, per
run `train_manifest.json` + `eval_mean_s6909.json` (checkpoints on node5).
Script `scripts/exp_v6_actor_goal_source.py`.  A sampling-level
diagnostic under oracle futures; not a mainline result.

## Question and design

The successful PointMaze run fed the actor's critic term branch-generated
future goals; the AntMaze contract keeps the actor on the logged relabeled
goals.  Does changing ONLY the actor's critic-term goal source help the
current policy?  Per lineage: the lineage's 30k critic frozen in both
arms (restored after every update; final hash = loaded), the same actor
start with its Adam state, +30,000 actor-only updates, BC 0.05.  Actor
rows in both arms = the pilot anchor pool (the 53,747 logged rows that
have a counterfactual branch, drawn by the recipe's anchor weights; every
row uses its OWN branch, no nearest-state matching); one RNG, so both
arms draw the same anchors and the same future-row quantile (first-batch
hashes: anchors equal, BC rows equal, critic-term goals differ).
`goal_log`: the critic-term goal = the row's recorded future goal;
`goal_cf`: = the sealed CF branch's future goal of the same anchor
(start-agent continuation).  BC in both arms: the same rows with the
logged action and the recorded goal (the learner's separate-BC-rows path;
loss body unchanged).  Evaluation once on the stage-2 draw (seed 6909).

## Result

| lineage | current | frozen (buffer stream, logged goals) | goal_log | goal_cf | start agent (reference) |
|---|---|---|---|---|---|
| 0 | 0.520 / far 0.59 / death 0.19 / timeout 0.29 | 0.497 / 0.71 / 0.01 / 0.49 | 0.407 / 0.48 / 0.20 / 0.40 | **0.277 / 0.007 / 0.72 / 0.00** | 0.270 / 0.01 / 0.71 / 0.02 |
| 1 | 0.437 / 0.44 / 0.35 / 0.21 | 0.500 / 0.86 / 0.01 / 0.49 | 0.293 / 0.30 / 0.42 / 0.29 | **0.257 / 0.000 / 0.73 / 0.01** | |
| 2 | 0.433 / 0.31 / 0.43 / 0.14 | 0.140 / 0.11 / 0.00 / 0.86 | 0.393 / 0.26 / 0.30 / 0.30 | **0.267 / 0.000 / 0.73 / 0.00** | |

Paired (success): goal_cf - goal_log -0.130 / -0.037 / -0.127 (3 / 3
worse; mean -0.098, seed s.e. 0.031); goal_cf - current -0.24 / -0.18 /
-0.17; deaths goal_cf - goal_log +0.52 / +0.31 / +0.43 (3 / 3, rule met
in the WRONG direction); far-route share -0.47 / -0.30 / -0.26.

**Reading (user's rule: no improvement -> close the hypothesis).**  With
the counterfactual future goals in its critic term the actor becomes the
START AGENT (success 0.26-0.28, far route 0.0-0.7 %, deaths 0.72-0.73:
the reference row).  The mechanism is direct: the sealed branches are
the start agent's continuations, which take the shortcut in ~99 % of the
anchors and die in most of them, so their positions -- shortcut corridor,
zone, death spot -- become the goals the actor is asked to reach; a
training row whose goal lies on the shortcut asks for the shortcut, and
under the CF goals nearly every row does (under the logged goals the 5 %
detour episodes still ask for the far route).  The hypothesis "changing
the actor's goal source alone repairs the current recipe" is CLOSED; no
ratio sweeps.  What PointMaze did is not transferable as is: its actor
goals came from branches whose continuation policy itself detoured.

Side observations (disclosed): both anchor-pool arms fit the logged
actions more tightly than the buffer stream (BC NLL -14 -> -18.5 /
-22.5, policy scale 0.07 -> 0.03-0.04; the pool is a fixed subsample, so
each row recurs more often), and `goal_log` is itself below `frozen` in
two lineages (-0.09 / -0.21 / +0.25) -- the row law alone moves the
policy; the comparison that matters here is within the pool
(goal_cf vs goal_log).

Limits: one evaluation draw; three lineages; frozen critic (the joint
case was not run and is not needed to close this hypothesis).
