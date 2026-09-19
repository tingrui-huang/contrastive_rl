# Clip round, step A: the current agents' futures at the same training anchors -- the gain does not enter the branch data at the logged torques; gate for B NOT PASSED

Question (user's lead, 2026-09-19 evening): the same first torques sampled by
the clipped CF policy enter the far route 19 / 128 times when the current
agent continues and 1 / 128 when the start agent continues, while the
critic's positive futures still come from the start agent.  Does the current
agents' capability enter the TRAINING data -- the branches at the logged
query torques?

Design: for each lineage (the clipped CF final of seed 0 / 1 / 2) every one of
the 53,747 anchors was regenerated with the same logged query torque and the
same hazard seed, the lineage agent's mode continuing (`generate_c_s*.json`;
roots, first actions and hazard seeds identical to `branches_cf.npz`, checked
on 24 anchors and by `BranchFutures`' own assertions).  `PROFILE.md` /
`profile.json`: anchor-weighted, paired by anchor.

## Result

| stratum (weight) | entered far old -> new (3 lineages) | completed far old -> new | success old -> new | timeout old -> new | goal-area mass of the critic marginal old -> new |
|---|---|---|---|---|---|
| reset rows t = 0 (0.004) | 0.013 -> 0.066 / 0.066 / 0.035 | 0.009 -> 0.035 / 0.026 / 0.022 | 0.286 -> 0.19 / 0.17 / 0.22 | 0.040 -> 0.18 / 0.24 / 0.17 | 0.057 -> 0.016 / 0.015 / 0.024 |
| start region (0.079) | 0.032 -> 0.041 / 0.047 / 0.040 | 0.025 -> 0.026 / 0.035 / 0.033 | 0.243 -> 0.17 / 0.17 / 0.21 | 0.029 -> 0.12 / 0.16 / 0.11 | 0.038 -> 0.015 / 0.016 / 0.019 |
| start region, t in [1, 30) (0.075) | 0.031 -> 0.032 / 0.038 / 0.035 | 0.024 -> 0.020 / 0.029 / 0.029 | 0.240 -> 0.17 / 0.17 / 0.20 | 0.021 -> 0.12 / 0.16 / 0.10 | 0.037 -> 0.015 / 0.015 / 0.018 |
| logged detour episodes, start region (0.004) | 0.69 -> 0.82 / 0.89 / 0.79 | 0.53 -> 0.54 / 0.69 / 0.67 | 0.59 -> 0.57 / 0.71 / 0.72 | 0.34 -> 0.42 / 0.25 / 0.26 | 0.063 -> 0.039 / 0.065 / 0.051 |
| pre_zone1 (0.234) | 0 -> 0 | 0 -> 0 | 0.31 -> 0.25 / 0.26 | 0.18 -> 0.18 / 0.22 | 0.055 -> 0.028 / 0.027 |
| between the zones (0.229) | 0 -> 0 | 0 -> 0 | 0.64 -> 0.45 / 0.48 | 0.09 -> 0.26 / 0.25 | 0.170 -> 0.081 / 0.086 |
| post_zone2 (0.152) | 0 -> 0 | 0 -> 0 | 0.955 -> 0.94 / 0.93 | 0.045 -> 0.06 / 0.07 | 0.62 -> 0.53 / 0.53 |
| logged shortcut episodes, all (0.950) | 0.000 -> 0.000 / 0.001 | 0 -> 0 | 0.61 -> 0.51 / 0.52 | 0.08 -> 0.17 / 0.18 | 0.278 -> 0.213 / 0.214 |
| all (1.000) | 0.013 -> 0.013 / 0.014 / 0.013 | 0.010 -> 0.010 / 0.011 / 0.012 | 0.62 -> 0.53 / 0.54 / 0.58 | 0.09 -> 0.17 / 0.18 / 0.14 | 0.273 -> 0.210 / 0.212 / 0.215 |

Gate for B (pre-registered: completed-far share at start-region anchors,
new - old > 0 in 3/3 and mean >= 0.02): +0.002 / +0.010 / +0.009, mean
+0.007 -> **NOT PASSED**; B (the paired continuation) was not run.  The gate
is an engineering threshold for spending the training, not a statistical
statement.

## Reading

1. After the LOGGED query torque the current agents go around hardly more
   than the start agent: at the 196 reset anchors 0.013 -> 0.035-0.066
   entered, 0.009 -> 0.022-0.035 completed; at the whole start region
   +0.01 entered; at the logged-shortcut anchors 0.000 -> 0.001.  The
   improvement the user measured with the policy's OWN sampled first
   steps (19 / 128) does not appear after the teacher's logged torques:
   the logged torque commits the ant to the shortcut basin for the
   current agents as it did for the round-1 agents (round 2: <= 0.4 %).
   The lag between the critic's futures and the agent's capability is
   real, but regenerating the branches at the logged torques does not
   close it.
2. What DOES enter the regenerated data is the current agents' worse
   shortcut walking: at the same states, same torque, same hazard draw,
   success 0.62 -> 0.53-0.58, timeouts 0.09 -> 0.14-0.18, and the
   goal-area mass of the critic's goal marginal 0.273 -> 0.21 (-23 %) --
   between the two hazard zones success 0.64 -> 0.45-0.48 and timeouts
   0.09 -> 0.25.  This is same-route, same-state evidence of a
   mid-corridor walking regression of the clipped CF agents relative to
   the start agent (the far-route audit found none from the corner; the
   shortcut corridor is a different place).  Switching the critic to
   these futures would remove positive support for the goal, not add
   route information.
3. Implication (not a result): the route information after the agent's
   OWN first step can enter the critic's data only through the query
   action -- counterfactual queries a != a_logged at the logged states
   -- which the sampling contract deliberately excludes (both arms query
   the logged torque; the earlier query-extension study was pre-clip and
   negative).  Whether to re-open that dimension under the clip recipe is
   the user's decision; nothing here was run.
4. Round 2's negative (pre-clip) is not evidence about this version; this
   round replaces it for the "regenerated futures" question at the
   logged torques, under the clip, without a restart.

Status: oracle (simulator) futures under the disclosed optimizer
-stabilisation recipe; the B continuation and its confirmation (seed 5909,
pre-registered) were not run.
