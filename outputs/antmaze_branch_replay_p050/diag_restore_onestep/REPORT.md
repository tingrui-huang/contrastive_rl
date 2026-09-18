# Restore determinism and one-step vs long-horizon predictability

Script `scripts/diag_v6_restore_onestep.py`; run on node 30021; results
`restore_check.json`, `onestep.json`.  No training entered the method.

## 1. Restore: same inputs -> same physical start (60 recorded anchors, 12 per stratum)

`build_v6_branch_replay.restore` = env.reset (mj_resetData: warmstart and
qacc zeroed, rocks parked, hidden draws consumed) + qpos / qvel written +
goal + absolute time + mj_forward.  With the recorded torque and one fixed
hidden-seed set:

| comparison | max abs difference over 31 obs dims | median |
|---|---:|---:|
| restored obs vs the dataset row obs[e, t] | 0.0 | 0.0 |
| fresh env instance vs an env that had just executed 150 continuation steps of another anchor: step 1 | 0.0 | 0.0 |
| same, after 10 further steps under the frozen policy | 0.0 | 0.0 |
| restore + recorded torque vs the original episode's next frame obs[e, t+1] | 5.4e-7 | 2.4e-7 |
| same, XY only | 4.8e-7 | 3.7e-9 |
| (scale: the recorded one-step change itself) | 7.5 | 4.1 |

After restore in both instances: qacc_warmstart = 0, data.time = 0, ncon
median 9 (normal foot contacts).  The restore is bit-for-bit independent
of what the simulator executed before, and it reproduces the original
episode's physics to float32 rounding (the original step had a non-zero
warm start and, when rocks were dropped, other contacts; neither shows
up at these anchors).  No restoration defect; the learner's inputs
(29-d state, 2-d goal, 8-d torque) determine the next physical state.

## 2. One-step consequence vs long-horizon consequence (same keys, same inputs)

Training keys: the R replay's 13,256 exact keys, one-step change Delta s
from row 0 to row 1 (identical across the key's draws: within-key
variance 1e-15, across-key 1.3).  Evaluation keys: the diagnostic's A
(480 keys), B (480), old C (800) and Cnew (320), one simulator step each.
Predictors fitted on the training keys only: an MLP of the S predictor's
architecture for Delta s (standardised, 29-d output, 30k updates, seeds
0 / 1) and kNN (k 10); against the saved S predictors of P_goal (three
seeds) and kNN on P, scored by R^2 against the 16-draw sample means.

| set | Delta s: MLP R^2 (seeds 0 / 1) | Delta s: kNN R^2 | P_goal: S R^2 (seeds 0 / 1 / 2) | P_goal: kNN R^2 |
|---|---|---:|---|---:|
| training keys | 0.999 / 0.999 | -- | 0.95 / 0.95 / 0.95 | -- |
| A familiar state, training torque | 0.999 / 0.999 | 0.87 | 0.25 / 0.32 / 0.25 | -0.11 |
| B familiar state, new torque | 0.986 / 0.987 | 0.86 | -1.20 / -0.83 / -1.22 | -0.28 |
| old C (5 held-out detour episodes + start strata) | 0.913 / 0.917 | 0.76 | -2.64 / -2.32 / -1.78 | -0.69 |
| Cnew (30 reserved detour episodes) | 0.841 / 0.848 | 0.60 | -1.50 / -2.18 / -1.42 | +0.03 |

Within an anchor (pairs of candidates): Spearman correlation between the
candidates' one-step distance and their P_goal difference 0.40 (A), 0.34
(B), 0.36 (old C), 0.03 (Cnew); pairs with a below-median one-step
distance but a P_goal difference above 0.1: 10% / 13% / 12% / 28%; the
MLP's predicted one-step distances track the true ones at Spearman
0.93-0.98 on every set.

## Reading

The first branch is closed: restoration is exact and the single step is
predictable and generalises -- to new torques at familiar states (R^2
0.99), to states of other episodes (0.84-0.92), with the same inputs the
critic receives.  The long-horizon target is not: the same architecture,
data and inputs give P_goal R^2 of 0.25-0.32 at familiar keys with fresh
outcomes and below zero (worse than a constant) at new torques and new
episodes; and within a state, torques that move the Ant almost alike
often end with very different completion probabilities (and vice versa),
especially on the new episodes (correlation 0.03).  The difficulty sits
in learning the long-continuation value from these inputs, not in the
physics, the restore, or the one-step representation.  "More samples"
of the same kind is not what this points at; how the value is built
from step-wise, predictable changes is.
