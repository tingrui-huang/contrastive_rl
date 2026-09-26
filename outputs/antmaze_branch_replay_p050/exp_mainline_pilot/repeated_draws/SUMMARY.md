# Repeated paired hazard draws at pre-selected decision states (user's plan after the pairing check, 2026-09-21; no training): under the frozen start-agent continuation a first-step torque changes the outcome in a minority of decision states; the average advantage is small (<= +0.07 at reset, ~0 elsewhere) and the NCE target does not favour it; the critics' 1-4 nat margins for the actors' torques do not track it

`scripts/exp_v6_repeated_draws.py states / generate / report` (node3; 8 per class 17 min, the pre-registered extension to 16 per class 35 min).
`states.npz`, `rollouts.npz` (node3; xy rows of every rollout), `REPORT.md` / `report.json` (16 per class), `REPORT_8perclass.md` / `report_8perclass.json`.

## Design (fixed before any outcome was seen)

* States: 64 reset anchors (t = 0), 64 start_early anchors (start region, t 1-40), 64 pre_zone1_early anchors (x < 4.5) -- seeded uniform
  samples of the training strata (seed 4242) -- plus 64 fresh env resets from a seed stream of their own (indep_reset) and the states 20
  start-agent steps later (indep_early); never selected by outcome or critic score.
* Candidates at the same state, all at that state's task goal: the logged torque (training states), the start agent's mode, the
  PRE-SPECIFIED CF actor's mode (seed 0), the MF actor's mode (seed 0), and the CF2 actor's mode (seed 0; an addition to the user's four).
  One candidate step, then the same frozen start agent (mode) -- the branch tables' construction.
* Pairing: identical hidden seed (clocks, rock jitter) across the candidates for each (state, class, draw); the hazard activity
  stratified over U00 / U10 / U01 / U11 with 8 (then 16) clock / jitter draws each, combined with the prior 1/4 per class; every outcome
  kept.  Verified: t0 identical across the candidates for every (state, class, draw); classes exact.
* Extension rule: 95 % half-width (state-level s.e.) of the CF-vs-start success advantage > 0.05 in any group -> 16 per class.  It fired
  (reset 0.049, indep_reset 0.056) and the extension changed nothing at the third decimal: the residual width is BETWEEN states (a
  minority of states with large effects of either sign), not within-state sampling noise -- more draws per state cannot narrow it.
* Measured per rollout: outcome, far-route entry, and the geometric-law masses of the path (goal area, reach 0.5, near 2.0, far, death
  frame) = the NCE-target quantities; per state the class-weighted means and the paired differences vs the start mode; per group the
  state-level mean, s.e., and the share of states with a positive / negative / exactly zero paired difference.

## Result (16 draws per class; class-weighted; 64 states per group)

| group | candidate | success | death | timeout | far entry | far & success | goal_area mass | success adv vs start (mean +- s.e.; states >0 / <0 / =0) | goal_area adv | far entry adv |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| reset | start | 0.233 | 0.738 | 0.029 | 0.020 | 0.006 | 0.0352 | | | |
| reset | logged | 0.235 | 0.737 | 0.027 | 0.016 | 0.000 | 0.0401 | +0.003 +- 0.008 (0.06 / 0.03 / 0.91) | +0.0049 | -0.004 |
| reset | CF (s0) | 0.234 | 0.551 | 0.215 | 0.125 | 0.051 | 0.0289 | +0.001 +- 0.025 (0.11 / 0.20 / 0.69) | -0.0062 | +0.105 |
| reset | CF2 (s0) | 0.299 | 0.609 | 0.092 | 0.125 | 0.104 | 0.0280 | +0.066 +- 0.030 (0.12 / 0.08 / 0.80) | -0.0071 +- 0.0036 | +0.105 |
| reset | MF (s0) | 0.279 | 0.539 | 0.182 | 0.141 | 0.103 | 0.0316 | +0.046 +- 0.033 (0.12 / 0.16 / 0.72) | -0.0036 | +0.121 |
| start_early | start | 0.265 | 0.727 | 0.008 | 0.031 | 0.031 | 0.0373 | | | |
| start_early | CF / CF2 / MF / logged | 0.255 / 0.248 / 0.252 / 0.237 | 0.727 | 0.02-0.04 | 0.03 | 0.02-0.03 | 0.040-0.048 | -0.010 / -0.017 / -0.013 / -0.028 (+- 0.007-0.010; =0 in 0.86-0.97 of the states) | +0.003 .. +0.011 | 0.000-0.004 |
| pre_zone1_early | start | 0.215 | 0.750 | 0.035 | 0.000 | 0.000 | 0.0544 | | | |
| pre_zone1_early | CF / CF2 / MF / logged | 0.242 / 0.238 / 0.246 / 0.246 | 0.750 | 0.004-0.012 | 0.000 | 0.000 | 0.034-0.040 | +0.027 / +0.023 / +0.031 / +0.031 (+- 0.010-0.012; >0 in 0.12-0.14, =0 in 0.86) | -0.015 .. -0.021 (+- 0.006) | 0.000 |
| indep_reset | start | 0.234 | 0.750 | 0.016 | 0.000 | 0.000 | 0.0344 | | | |
| indep_reset | CF / CF2 / MF | 0.258 / 0.263 / 0.240 | 0.60-0.62 | 0.12-0.17 | 0.094 / 0.125 / 0.078 | 0.062 / 0.056 / 0.046 | 0.024-0.033 | +0.023 / +0.028 / +0.006 (+- 0.022-0.029; =0 in 0.77-0.82) | -0.011 / -0.001 / -0.006 | +0.09 .. +0.13 |
| indep_early | start | 0.239 | 0.750 | 0.012 | 0.000 | 0.000 | 0.0374 | | | |
| indep_early | CF / CF2 / MF | 0.239 / 0.239 / 0.234 | 0.74-0.75 | 0.02 | 0.000 | 0.000 | 0.034-0.039 | +0.000 / +0.000 / -0.004 (+- 0.010; =0 in 0.88) | -0.003 .. +0.001 | 0.000 |

* "=0" is the share of states in which the candidate and the start torque give the SAME outcome in all 64 paired draws: the start-agent
  continuation absorbs the first step in 69-80 % of the reset states, 86-97 % of the early / pre-zone-1 states.
* Where the first step is not absorbed at reset, it mostly puts the continuation onto the far route (far entry 0.09-0.14 vs 0.00-0.02
  for start / logged) and the far route under the start agent then completes 0.41 (CF's entries) / 0.83 (CF2's) / 0.73 (MF's) of the
  time, the rest timing out (timeouts 0.09-0.22 vs 0.03); deaths fall from 0.74 to 0.54-0.61.
* The NCE target: the geometric-law goal-area mass of the successful far-going candidates is LOWER than the start torque's at reset
  (CF2 -0.0071 +- 0.0036 with +0.066 success) and at pre_zone1_early (-0.015 .. -0.021 with +0.02 .. +0.03 success): the far route is
  long and its timeouts are long, so the gamma-law puts less mass near the goal even when the success rate is higher.

## The critics against the measured advantage (twin-min logit at (state, executed candidate torque, task goal); seed mean of five)

| group | critic | candidate | margin vs start (share > 0) | corr with the success adv | corr with the goal-area adv | sign agreement with the success adv |
|---|---|---|---|---:|---:|---:|
| reset | CF | CF / CF2 / MF / logged | +1.65 (0.94) / +1.19 (0.85) / +0.90 (0.76) / -0.02 (0.46) | -0.06 / +0.07 / +0.06 / -0.18 | +0.01 / -0.31 / +0.01 / +0.23 | 0.35 / 0.57 / 0.43 / 0.20 |
| reset | CF2 | CF / CF2 / MF / logged | +3.35 (0.99) / +4.41 (1.00) / +3.45 (1.00) / +2.13 (0.96) | -0.02 / +0.02 / +0.17 / -0.16 | +0.04 / +0.09 / -0.01 / +0.19 | 0.35 / 0.62 / 0.44 / 0.57 |
| reset | MF | CF / CF2 / MF / logged | +1.15 (0.96) / +1.59 (1.00) / +1.28 (0.93) / +0.21 (0.62) | +0.08 / +0.15 / +0.19 / -0.15 | -0.04 / -0.33 / -0.01 / +0.17 | 0.35 / 0.62 / 0.44 / 0.33 |
| indep_reset | CF / CF2 / MF | CF | +1.52 (0.92) / +3.27 (0.99) / +1.06 (0.93) | +0.01 / -0.01 / +0.10 | +0.07 / +0.08 / -0.14 | 0.33 / 0.39 / 0.40 |
| pre_zone1_early | CF / CF2 / MF | CF | +0.16 (0.77) / +0.02 (0.57) / +0.07 (0.72) | +0.14 / +0.06 / +0.11 | -0.15 / -0.07 / -0.15 | 0.78 / 0.56 / 0.58 |

* At the reset states every critic prefers every actor's torque over the start torque by 1-4 nats in 76-100 % of the states -- a blanket
  preference for "actor-like" torques (the CF2 critic most of all) -- and this preference is uncorrelated with the measured per-state
  success advantage (|corr| <= 0.19) and with the goal-area-mass advantage, with sign agreement at chance (0.33-0.62).  The CF critic's
  own-actor margin (+1.65) exceeds its margin for MF's torque (+0.90) although the measured success of CF's torque is the lower one
  (0.234 vs 0.279; within noise).
* The logged torque, which the critics were trained on, gets no margin from CF / MF (-0.02 / +0.21) and +2.1 from CF2.

## Reading against the user's decision tree (none forced)

* "No stable advantage under the current continuation": largely YES -- with one candidate step and the frozen start agent, the first-step
  torque is absorbed in 69-97 % of the decision states; the average success advantage is +0.00 (CF) / +0.07 (CF2) / +0.05 (MF) at reset,
  ~0 at start_early and indep_early, +0.02-0.03 at pre_zone1_early (for the logged torque too), +0.01-0.03 at indep_reset -- none of
  them 2 s.e. clear except CF2 at reset (2.2 s.e.) and the pre_zone1_early set (2.3-3 s.e., but shared by the logged torque).  Repeated
  draws do not create an advantage that the continuation does not carry; per the user's rule this points at the continuation policy,
  and does NOT show that the first-step torque is unlearnable.
* "A success advantage exists but the goal-sampling target does not favour it": YES where the advantage exists -- the far-going
  candidates' goal-area mass under the gamma law is lower than the start torque's (reset CF2 -0.007 vs +0.066 success; pre_zone1_early
  -0.015 .. -0.021 vs +0.02 .. +0.03).  The NCE target and the task return disagree in sign at the decision states.
* "An advantage exists and the critics do not recognise it": the critics' margins are large and uncorrelated with the measured per-state
  advantage of either kind; the critics are not a per-state judge of the first step here.
* "The critics are reliable and the actor objective refuses the good choice": NO.
* What the oracle CF actor's test-time far route (0.28-0.61) is NOT: a first-step effect measurable by the branch tables -- with the same
  continuation its first step yields far entry 0.125 at reset and completion 0.41.  The far route emerges from the CF actor's own
  closed-loop control; the branch tables (one counterfactual step + start-agent continuation) could not have supervised it, and the
  CF critic's 0.9-nat preference for CF's torque (pairing_check/) is not a measured advantage.
* Caveats: 64 states per group; the CF / MF / CF2 torques are the seed-0 actors' modes (pre-specified); classes weighted by the prior
  1/4 (the eval draws are Bernoulli 0.5 x 0.5, the same expectation); the geometric masses are the branch-table quantities, not the
  actor's relabelled goals.

# CROSSOVER (user's plan after afc12c9): the same states, candidates and hidden seeds with the continuation switched to the frozen CF s0 actor for every candidate -- a first-step advantage exists ONLY at the reset states (~+0.09, 2 s.e. pooled), the near-goal NCE mass follows it weakly there, the CF continuation is worse than the start continuation everywhere else, and the critics' region-level ranking tracks none of it

`exp_v6_repeated_draws.py states --add-cf-early / generate --continuation {start, CF} / crossover` (node3; the CF-continuation run 110,592 rollouts,
75 min).  `REPORT_crossover.md` / `report_crossover.json`; rollout files on node3 (`rollouts.npz`, `rollouts_start_cf_early.npz`, `rollouts_CF_base.npz`).
Added group `cf_early` (64): the SAME fresh resets as indep_reset (replayed and asserted equal) rolled 20 steps by the CF s0 actor -- the
early states the new policy visits while turning; run under both continuations.  Every column's continuation is the same for all
candidates; the executed candidate torques are asserted identical across the two continuation runs.  Region-level critic readout:
importance-corrected p(region | s, a) over each family's own training-marginal reference set (4,096 goals; goal area 1,090-1,101 of
them, near-2.0 ~870 per state, reach-0.5 ~140), the candidate-minus-start margin compared per state with the measured region-mass
advantage under each continuation.

## Success and far-route completion (class-weighted, 16 draws per class; 64 states per group)

| group | continuation | start torque | CF torque (adv vs start) | CF2 torque (adv) | MF torque (adv) | logged torque (adv) |
|---|---|---|---|---|---|---|
| reset | start | 0.233 (far 0.02) | 0.234 (+0.001 +- 0.025; far 0.125, compl 0.40) | 0.299 (+0.066 +- 0.030; far 0.125, compl 0.83) | 0.279 (+0.046 +- 0.033) | 0.235 (+0.003) |
| reset | CF | 0.428 (far 0.457, compl 0.68) | 0.531 (**+0.102 +- 0.056**, >0 / <0 / =0 0.41 / 0.31 / 0.28; far 0.656, compl 0.67) | 0.506 (+0.078 +- 0.064; far 0.734, compl 0.61) | 0.355 (-0.073 +- 0.066; far 0.457) | 0.228 (**-0.200 +- 0.059**; far 0.078) |
| indep_reset | start | 0.234 (far 0.00) | 0.258 (+0.023 +- 0.029) | 0.263 (+0.028 +- 0.024) | 0.240 (+0.006) | -- |
| indep_reset | CF | 0.358 (far 0.328) | 0.438 (+0.081 +- 0.065; far 0.562) | 0.475 (+0.117 +- 0.071; far 0.657) | 0.414 (+0.056 +- 0.057) | -- |
| start_early | start | 0.265 | 0.255 (-0.010) | 0.248 (-0.017) | 0.252 (-0.013) | 0.237 (-0.028) |
| start_early | CF | 0.194 (far 0.04) | 0.183 (-0.012 +- 0.022) | 0.137 (-0.058 +- 0.024) | 0.160 (-0.035) | 0.160 (-0.034) |
| pre_zone1_early | start | 0.215 | 0.242 (+0.027) | 0.238 (+0.023) | 0.246 (+0.031) | 0.246 (+0.031) |
| pre_zone1_early | CF | 0.168 (far 0.00) | 0.176 (+0.008 +- 0.012) | 0.166 (-0.002) | 0.157 (-0.011) | 0.176 (+0.008) |
| indep_early | start | 0.239 | 0.239 (+0.000) | 0.239 (+0.000) | 0.234 (-0.004) | -- |
| indep_early | CF | 0.167 (far 0.00) | 0.169 (+0.002 +- 0.016) | 0.180 (+0.013) | 0.179 (+0.011) | -- |
| cf_early | start | 0.492 (far 0.536, compl 0.74) | 0.519 (+0.027 +- 0.035; far 0.516, compl 0.81) | 0.506 (+0.014 +- 0.047) | 0.503 (+0.011 +- 0.041) | -- |
| cf_early | CF | 0.459 (far 0.578, compl 0.65) | 0.447 (-0.013 +- 0.062; compl 0.61) | 0.512 (+0.052 +- 0.057; compl 0.72) | 0.493 (+0.033 +- 0.053) | -- |

Pooled over the two reset groups (128 states), CF continuation: CF torque success +0.091 +- 0.043 (2.1 s.e.), far entry +0.217 +- 0.052;
CF2 torque +0.098 +- 0.048 (2.0 s.e.), far entry +0.303; MF torque -0.009 +- 0.043.  Timeouts under the CF continuation 0.10-0.33 in every
group vs 0.01-0.04 (non-reset) / 0.03-0.22 (reset) under the start continuation.

## The geometric-law masses (the NCE target) under the CF continuation, candidate minus start torque

| group | candidate | near2.0 mass adv | goal_area mass adv | reach0.5 |
|---|---|---|---|---|
| reset | CF / CF2 / MF / logged | **+0.019 +- 0.008** / +0.021 +- 0.009 / -0.010 +- 0.007 / -0.019 +- 0.007 | +0.006 +- 0.007 / +0.008 +- 0.008 / -0.011 +- 0.005 / -0.009 +- 0.006 | +-0.0001 |
| indep_reset | CF / CF2 / MF | +0.009 +- 0.012 / +0.011 +- 0.011 / -0.000 +- 0.009 | +0.004 +- 0.011 / -0.000 +- 0.009 / -0.004 +- 0.008 | ~0 |
| pooled resets | CF / CF2 | +0.014 +- 0.007 (1.9 s.e.) / +0.016 +- 0.007 (2.2 s.e.) | +0.005 +- 0.006 (0.8) / +0.004 +- 0.006 (0.6) | |
| start_early, pre_zone1_early, indep_early, cf_early | all | within +-0.004 (n.s.) | within +-0.006 (n.s.) | ~0 |

(Under the start continuation the same masses were <= 0 for the far-going candidates: reset CF2 near2.0 -0.004 +- 0.004, goal_area -0.007 +- 0.004.)

## The critics' region-level ranking (seed mean of five; margin = p(region | s, a_cand) - p(region | s, a_start))

| group | critic : candidate | goal_area margin | corr / sign agreement with the measured goal_area adv, start cont. | ..., CF cont. | near2.0 margin | corr / sign, start | corr / sign, CF |
|---|---|---:|---|---|---:|---|---|
| reset | CF : CF | +0.230 | +0.14 / 0.47 | +0.00 / 0.56 | +0.191 | +0.19 / 0.42 | -0.16 / 0.53 |
| reset | CF : CF2 | +0.104 | -0.31 / 0.46 | -0.10 / 0.56 | +0.100 | -0.23 / 0.48 | -0.05 / 0.59 |
| reset | CF : MF | +0.090 | +0.07 / 0.47 | -0.02 / 0.42 | +0.088 | +0.13 / 0.51 | -0.13 / 0.38 |
| reset | CF2 : CF / CF2 / MF | +0.197 / +0.249 / +0.165 | +0.08 / -0.18 / -0.06 (sign 0.43-0.55) | +0.01 / -0.15 / -0.09 (0.37-0.59) | +0.162 / +0.205 / +0.142 | +0.16 / -0.17 / +0.04 | -0.09 / -0.13 / -0.04 |
| reset | MF : CF / CF2 / MF | +0.142 / +0.133 / +0.110 | +0.05 / -0.25 / +0.05 | +0.00 / -0.14 / -0.01 | +0.116 / +0.115 / +0.098 | +0.12 / -0.18 / +0.10 | -0.12 / -0.07 / -0.05 |
| indep_reset | all | +0.08 .. +0.24 | -0.30 .. +0.05 (sign 0.35-0.53) | -0.27 .. +0.11 (0.45-0.59) | +0.08 .. +0.20 | -0.13 .. +0.17 | -0.24 .. +0.19 |
| cf_early | all | -0.013 .. +0.000 | abs corr <= 0.14 (sign 0.44-0.53) | abs corr <= 0.22 (0.39-0.53) | -0.011 .. +0.000 | | |
| pre_zone1_early | all | -0.000 .. +0.008 | abs corr <= 0.13 | abs corr <= 0.16 | ~0 | | |

* At the reset states every critic raises the goal-area probability by +0.09 .. +0.25 for EVERY actor torque relative to the start torque
  -- for MF's torque as much as for CF's, although MF's measured advantage under the CF continuation is negative (-0.011 goal area,
  -0.073 success) and CF's is positive (+0.006, +0.102).  The true region-mass differences are 0.005-0.02; the critics' margins are
  10-30x larger and uncorrelated per state with the measured advantage under either continuation (abs corr <= 0.31, sign agreement at
  chance).  At the CF actor's own early states (cf_early) and at pre_zone1_early the margins are ~0 -- the critics separate the
  candidate torques only at the reset states, where the actors' torques are far from the logged ones.

## Reading against the user's three cases

* "Under the CF continuation a good first step has a stable advantage and the original goal sampling keeps it": PARTLY, and only at the
  reset states -- CF / CF2 torques +0.09 / +0.10 success (2.0-2.1 s.e. over 128 reset states; 41-48 % of states positive, 28-39 % negative)
  through far entry +0.22 / +0.30, with the near-2.0 mass of the gamma law following at +0.014 / +0.016 (1.9-2.2 s.e.) and the goal-area
  mass not clearly (+0.004 / +0.005, < 1 s.e.).  At every other state group the CF continuation is WORSE than the start continuation
  (0.14-0.19 vs 0.22-0.27 success; timeouts 0.10-0.16 vs 0.01-0.04) and no first-step advantage exists (abs adv <= 0.06, mostly within 1
  s.e.; CF2 at start_early -0.058 +- 0.024).  At the CF actor's own turning states both continuations give 0.45-0.52 regardless of the first
  step.
* "The advantage holds in the data but the critics do not learn it": the critics' per-state region ranking is uncorrelated with the
  measured region-mass advantage under either continuation, and their margins are a blanket +0.1-0.25 for any actor-like torque at the
  reset states (MF's included).  So even where the CF continuation creates a target advantage (reset, near-2.0), the existing critics
  do not rank it per state.
* "After switching to the CF continuation there is still no reliable advantage, mainly more timeouts": TRUE outside the reset states.
* Structural facts for the next decision: (i) the CF actor's far route is decided in the first steps from reset and is fragile -- one
  logged (shortcut-teacher) first step derails it (success 0.228, far entry 0.078; -0.200 +- 0.059 vs the start torque); (ii) the reset
  states are 196 of 53,747 anchors, so a CF-continuation table would carry the first-step advantage on a small weight while degrading the
  futures everywhere else; (iii) the near-2.0 mass is the gamma-law quantity that moves with the advantage; the reach-0.5 mass (0.001)
  and the goal-area mass do not resolve it.
* Not concluded: that the first-step torque is unlearnable, or that a CF-continuation resampling round would fail; only that the
  crossover does not show the "stable advantage + preserved target" condition the user set for that round beyond the reset states,
  and that the current critics would not be the judge of it.
