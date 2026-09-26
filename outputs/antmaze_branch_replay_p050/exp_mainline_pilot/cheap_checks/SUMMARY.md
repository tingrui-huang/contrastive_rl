# Two cheap checks on existing data (user's step 4, 2026-09-21; no rollouts, no training)

`scripts/diag_v6_cheap_checks.py`; `GOAL_MASS.md` / `goal_mass.json`, `BC_MODE.md` / `bc_mode.json`.

## (a) Where the critic's positive-goal law puts its mass along the existing futures

The critic's positives are rows m >= 1 of a future, drawn with P(m) ∝ 0.999^m
truncated at the future's end (the stream's own law).  Masses below are
anchor-weighted means of each future's mass in a category; "reach" = within
the 0.5 success radius of the task goal.

| source | mean length | success / death / timeout | reach 0.5 | goal area | stall rows | death position (last 5) | terminal 10 rows | mid-route |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| O recorded, all anchors | 135 | 0.999 / 0 / 0.001 | 0.022 | 0.273 | 0.022 | 0.000 | 0.155 | 0.955 |
| CF sealed, all anchors | 143 | 0.621 / 0.291 / 0.088 | 0.019 | 0.273 | 0.051 | 0.056 | 0.230 | 0.873 |
| O recorded, start anchors | 255 | 1.000 / 0 / 0 | 0.0035 | 0.076 | 0.030 | 0.000 | 0.036 | 0.966 |
| CF sealed, start anchors | 138 | 0.243 / 0.728 / 0.029 | 0.0009 | 0.038 | 0.015 | 0.053 | 0.115 | 0.931 |
| lineage 0 queries, logged q0, start anchors | 209 | 0.180 / 0.692 / 0.128 | 0.0006 | 0.016 | 0.076 | 0.050 | 0.107 | 0.874 |
| lineage 0 queries, mode q1 | 224 | 0.195 / 0.667 / 0.138 | 0.0006 | 0.017 | 0.077 | 0.048 | 0.104 | 0.874 |
| lineage 0 queries, extended | 220 | 0.193 / 0.673 / 0.134 | 0.0006 | 0.017 | 0.076 | 0.049 | 0.105 | 0.875 |

By outcome at the start anchors (per-future masses):

| outcome | mean length | reach 0.5 | goal area | stall | death position | terminal 10 |
|---|---:|---:|---:|---:|---:|---:|
| a success future (CF sealed / q0 / mode) | 237 / 283 / 308 | 0.004 / 0.003 / 0.003 | 0.11 / 0.08 / 0.08 | 0.00 | 0.00 | 0.03-0.04 |
| a death future | 80-83 | 0 | 0 | 0.00 | 0.072 | 0.145 |
| a timeout future | 781-790 | 0 | 0.01-0.36 | 0.53-0.59 | 0.00 | 0.008 |

**Reading.**  "Reaches more" does NOT mean "more task-goal positives": with
gamma 0.999 the law is nearly uniform over a 250-300-row success future,
so one success adds 0.3-0.4 % of its mass within the reach radius (8-11 %
in the goal area), whereas one death future puts 7 % on its death
position and 15 % on its last ten rows, and one timeout future puts more
than HALF of its mass on stall rows.  Between the logged first torque
and the agent's own (q0 vs mode / extended) the task-goal mass is
identical to three decimals (0.0006; goal area 0.016 vs 0.017) even
though success rises 0.18 -> 0.20 and far completion 0.03 -> 0.05.  The
critic's positive set is 87-97 % mid-route positions; the recorded
futures (arm O) reach 99.9 % yet carry only 2.2 % reach mass.  So the
termination rule (no goal hold; a future ends on its reach frame, a
timeout parks at the stall) is worth studying: it decides that a
far-route success is nearly invisible to the task-goal positives while a
stall is very visible.  This is a measurement, not a proposal to change
the rule now (user: evidence-gated follow-up).

## (b) The BC term under the recipe's 'clip' log-prob vs dm-acme's 'acme' boundary rule

Real actor batches of each lineage (the actor stream fast-forwarded to
the 30k checkpoint, 20 batches of 1,024), the lineage's own actor;
weighted gradients (0.05 x BC, 0.95 x critic term).

| lineage | NLL clip | NLL acme | boundary components / rows with one | weighted BC grad norm clip / acme | cos(clip, acme) | rel. difference | weighted critic-term grad norm |
|---|---:|---:|---|---|---:|---:|---:|
| 0 | -13.34 | -13.19 | 1.6 % / 10.6 % | 12.04 / 11.96 | 0.9994 | 3.5 % | 2.46 |
| 1 | -13.69 | -13.55 | 1.6 % / 10.6 % | 10.78 / 10.61 | 0.9986 | 5.2 % | 2.91 |
| 2 | -13.11 | -12.98 | 1.5 % / 10.7 % | 43.10 / 43.09 | 1.0000 | 0.7 % | 2.57 |

**Reading.**  Only 1.6 % of the logged action components sit at the
boundary; the two rules give BC gradients with cosine >= 0.9986 and a
relative difference of 0.7-5 %, and the same angle to the critic-term
gradient.  The log-prob implementation is NOT a lever on these batches;
switching to 'acme' would not change the AntMaze training materially
(PointMaze's use of it is not transferable evidence).  Not worth a
training.  A side observation to keep: at bc 0.05 the weighted BC
gradient norm is 4-17x the weighted critic-term gradient norm on the
actor's own rows.
