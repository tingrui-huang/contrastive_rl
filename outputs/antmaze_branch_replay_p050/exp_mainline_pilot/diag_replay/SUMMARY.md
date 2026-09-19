# Corrected attribution: the actor objective at its real training rows, the training replay, and the full-state resumption control

User's review of the round-2 report (2026-09-19) rejected three attributions
as unproven: (1) "the full actor objective is at a tie" was computed with the
episode's TASK goal, while the actor trains on RELABELED goals; (2) "the
mid-route regression is interference from the majority rows" used one plain
SGD step on the FINAL checkpoint, not the Adam state at the checkpoint where
the change happened; (3) round 2 restarted the critic and the actor's Adam
state, so "regenerated futures are worse" is confounded with "learner
restarted" (CFold2 lost capability with unchanged futures).  Three checks,
all frozen models or the pilot's own learner, no change of method:

* `scripts/diag_v6_actor_objective_stream.py` -> `diag_traj/actor_objective_stream.json` (step 1a)
* `scripts/diag_v6_training_replay.py replay / probe / calib` -> `diag_replay/CF_s*/{probe,calibration}.json`, `REPORT.md` (step 1b)
* `scripts/diag_v6_training_replay.py resume` -> `round2/resume/CF/seed_*/` (step 2), milestone probes in `round2/resume/CF/seed_*/probe.json`

## 1a. The actor objective at the rows it actually trains on

48 real actor batches (49,152 rows) per seed from the training seed's own
`ActorStream` (same buffer, same relabeling law, gamma 0.999).  Start-region
rows are 7.8-8.3 % of a batch (3,849 / 4,098 / 3,939 rows); 95 % of them are
shortcut-episode rows.  Their relabeled goals lie in the hazard corridor
(43 %), the corridor before hazard 1 (25 %), after hazard 2 (15 %), the goal
area (7-8 %), the start (5 %), the detour legs (4 %); median goal offset 118-
121 steps.  For each round-1 CF actor / critic pair (and the O pairs), at the
shortcut-episode start rows, with the REAL goal and, on the same rows, the
task goal:

| quantity (shortcut-episode start rows) | CF s0 / s1 / s2, real goal | CF, task goal (same rows) | O s0 / s1 / s2, real goal |
|---|---|---|---|
| max_j f(teacher detour_j) - E_pi[f] (nats), P > 0 | -0.27 / -0.27 / -0.54; P 0.42 / 0.35 / 0.34 | +0.19 / +0.20 / -0.03; P 0.55 / 0.58 / 0.52 | -0.67 / -0.88 / -0.51; P 0.14-0.16 |
| P(best teacher detour torque > logged torque) | 0.44 / 0.38 / 0.36 | 0.59 / 0.60 / 0.59 | 0.14 / 0.15 / 0.16 |
| critic-term action gradient projected toward the detour torque, P > 0 | +0.08 / +0.04 / -0.01; P 0.58 / 0.55 / 0.50 | +0.07 / +0.25 / +0.09 | +0.02 / +0.02 / +0.04 |
| full objective 0.95 (-E_f) + 0.05 NLL toward the detour torque (scale widened): delta at the target, P(better) | +1.68 / +1.66 / +1.85; P 0.04 / 0.02 / 0.03 | +1.31 / +1.36 / +1.43; P 0.15 / 0.11 / 0.09 | +2.01 / +2.07 / +1.84; P <= 0.01 |

By the relabeled goal's location (CF, real goal): hazard-corridor goals
-0.51 / -0.41 / -0.82 (P 0.21-0.33); corridor before hazard 1 +0.05 / -0.10 /
+0.06 (P ~0.5); goal area -0.06 / +0.08 / -0.28 (P 0.46-0.53); start goals
-0.24 / +0.01 / +0.38; no goal region has the full objective better at the
detour torque (P <= 0.12).  By goal offset: m < 50: about 0; m >= 50: -0.3
to -0.9.

Reading.  In the actor's own training pairing there is no critic preference
for the teacher detour torques at the start-region rows that BC could be
cancelling: the critic term itself is slightly against them (most clearly
when the relabeled goal lies in the hazard corridor -- a shortcut torque is
what reaches a corridor goal), and the full objective is worse toward them
in 96-98 % of rows.  Under the task goal the critic term is mildly for them
(+0.2 nats, P 0.55-0.60) and the full objective still worse (+1.3 to +1.4).
Section 9b's "exact tie" (task goal, candidate set including the actor's own
samples) is withdrawn as a statement about the training objective.  What
this check does NOT measure: whether the objective pushes toward the
actor's OWN route-changing torques (the teacher torques are transplants from
other states; the CF actors' own samples score above them), and the t = 0
rows separately (the earlier reset-row readout of +1.9 to +2.9 nats over the
logged torque is specific to t = 0 under the task goal; over the broader
start region under the task goal it is P 0.59-0.60).

## 1b. Training replay of the round-1 CF runs, probed every 1,000 updates

Replay from `init.pkl` with the full saved `TrainingState` and the critic /
actor streams' own sequence (the RNG fast-forward reproduces
`TrajectoryBuffer.sampled_indices` exactly; `selfcheck`); a checkpoint every
1,000 updates.  Probes per checkpoint: route -- the mode from the 300
evaluation resets for 100 steps, share with max y >= 2.5 (turned north);
walking -- continuation from the recorded CF policy's own detour-entrance
states (all recorded entrances, 45 / 60 / 60 states; the cont.json timeout-
entrance set 4 / 12 / 42 as well), reach rate; the checkpoint's own critic at
the 192 logged-shortcut reset rows (task goal): best teacher detour torque
minus f(mode) and minus f(logged); the reset-row mode's distance to the
start agent's mode.

Calibration (`calibration.json`): the ORIGINAL finals give north 0.127 /
0.180 / 0.570 (evaluation detour 0.107 / 0.167 / 0.500), so the 100-step
probe tracks the evaluation; the replays end at 0.147 / 0.053 / 0.303 --
GPU non-determinism makes a replay a statistical replicate, and the 30k
snapshot varies between replicates by as much as between seeds.  Start
agent: north 0.000, entrance reach 0.47 / 0.68 / 0.67.

Curves (update: north / entrance reach):

* seed 0 (45 entrances): 0k .00/.47 | 1k .00/.60 | 2k .01/.64 | 3k .01/.58 | 4k .27/.51 | 5k .00/.00 | 6k .11/.00 | 7k .30/.02 | 8k .28/.07 | 9k .00/.00 | 10k .01/.00 | 11k .07/.02 | 12k .47/.20 | 13k .49/.20 | 14k .14/.22 | 15k .07/.33 | 16k .05/.42 | 17k .02/.31 | 18k .00/.36 | 19k .02/.36 | 20k .03/.36 | 21k .01/.38 | 22k .06/.44 | 23k .02/.36 | 24k .13/.49 | 25k .04/.47 | 26k .03/.62 | 27k .32/.49 | 28k .14/.47 | 29k .20/.33 | 30k .15/.42
* seed 1 (60): 0k .00/.68 | 1k .01/.13 | 2k .00/.65 | 3k .00/.00 | 4k .01/.02 | 5k .00/.05 | 6k .11/.38 | 7k .00/.00 | 8k .01/.50 | 9k .03/.50 | 10k .00/.40 | 11k .01/.45 | 12k .04/.47 | 13k .06/.48 | 14k .02/.43 | 15k .03/.60 | 16k .04/.48 | 17k .00/.53 | 18k .02/.55 | 19k .00/.63 | 20k .00/.00 | 21k .01/.37 | 22k .01/.22 | 23k .01/.50 | 24k .01/.58 | 25k .04/.53 | 26k .09/.50 | 27k .00/.48 | 28k .21/.50 | 29k .02/.43 | 30k .05/.53
* seed 2 (60): 0k .00/.67 | 1k .00/.72 | 2k .01/.75 | 3k .00/.65 | 4k .00/.73 | 5k .00/.73 | 6k .00/.80 | 7k .00/.00 | 8k .01/.05 | 9k .09/.03 | 10k .00/.02 | 11k .00/.30 | 12k .01/.38 | 13k .02/.53 | 14k .05/.40 | 15k .01/.47 | 16k .05/.58 | 17k .04/.60 | 18k .04/.60 | 19k .05/.68 | 20k .08/.72 | 21k .03/.52 | 22k .02/.62 | 23k .11/.70 | 24k .29/.73 | 25k .28/.48 | 26k .15/.67 | 27k .27/.67 | 28k .27/.65 | 29k .26/.57 | 30k .30/.53

Critic margin at the reset rows (best teacher detour - f(logged), task
goal): positive from 3k (s0: +0.6 -> +2 to +5), 11k (s1: +0.4 -> +1.5 to
+2.3), 12k (s2: +1.4 -> +1.5 to +2.3), with dips (s0 9k -0.9 -> +1.9; s2 7k-
11k and 15k negative or near zero).  Reset-row mode distance to the start
mode: 2.5-3.0 after the first 1,000 updates and 3.0-3.5 for the rest of
training, in every seed.

Findings.

1. The reset-row torques leave the start policy within the first 1,000
   updates (|delta| 2.5-3.0 of a maximum ~5.7) and stay there, long before
   any route change (north <= 0.01 for the first 3k-20k updates).  The
   route is not read off the distance from the start policy.
2. Walking collapses early and recovers, with no route change: entrance
   reach falls to 0.00-0.07 at 5k-11k (s0), 1k / 3k-5k / 7k / 20k (s1),
   7k-10k (s2) while north is <= 0.01-0.09, and recovers to 0.4-0.7 by
   12k-20k.  The collapse coincides with a large positive critic margin in
   seed 0 (+4.6 at 5k) and with a negative one in seed 2 (-0.7 to -1.0):
   the margin's sign does not predict it.
3. The critic's preference for the detour torques at the reset rows is
   established at 3k / 11k / 12k and stays at +1.5 to +2.9 nats; the
   actor's route share does not track it: seed 2 turns at 23k-24k (11k
   updates later), seed 0 spikes at 12k-13k (0.47-0.49), falls back to
   <= 0.07 for 10k updates and fluctuates 0.03-0.32 afterwards, seed 1's
   replay never turns (<= 0.06 except 0.21 at 28k).
4. Walking at the time of the route turn is at its recovered level (s2:
   0.70-0.73 at 23k-24k; s0: 0.20 at 12k-13k, during the recovery) and
   fluctuates afterwards (s2: 0.48-0.67, final 0.53 vs the start agent's
   0.67 and the pre-collapse 0.80; s0: 0.33-0.62, final 0.42 vs 0.47).
5. Consecutive checkpoints 1,000 updates apart differ by up to 0.2-0.3 in
   the route share and up to 0.4 in walking: the process is not converged
   at 30k, and a single 30k evaluation is one draw from a fluctuating
   process (the replicate-to-replicate spread, 0.05 vs 0.18 and 0.30 vs
   0.57, is as large as the seed spread).

Reading.  The detour-walking loss is not caused by the route change (it
occurs before and without it), and the route change is not a direct
readout of the critic margin (it lags it by 0-11k updates and fluctuates).
The mid-route "interference" reading of section 9c described one SGD
displacement at the final checkpoint; the actual history is a sequence of
walking collapses and recoveries that this replay locates but does not
explain.  Which rows or which term drive the collapses is not established.

## 2. Full-state resumption control (step 2)

From each round-1 CF `final.pkl` with the whole learner state (actor,
critic, target, both Adam states, sampling key), streams fast-forwarded by
30,000 batches, the round-1 futures, +30,000 updates, same losses (bc 0.05),
same evaluation (300 episodes, seed 3909, mode):

| policy | success | detour | death | timeout | no route | mean steps |
|---|---:|---:|---:|---:|---:|---:|
| round 1 CF s0 / s1 / s2 (start of the continuation) | 0.267 / 0.320 / 0.480 | 0.107 / 0.167 / 0.500 | 0.627 / 0.567 / 0.260 | 0.107 / 0.113 / 0.260 | -- | 222 / 241 / 423 |
| round 2 CFold (restarted critic + Adam, same futures) | 0.263 / 0.283 / 0.260 | 0.163 / 0.120 / 0.210 | 0.513 / 0.587 / 0.447 | 0.223 / 0.130 / 0.293 | -- | 318 / 237 / 368 |
| **resume (carried state, same futures)** | **0.163 / 0.500 / 0.140** | 0.137 / 0.783 / 0.277 | **0.033 / 0.020 / 0.003** | **0.803 / 0.480 / 0.857** | 0.707 / 0.120 / 0.677 | 709 / 641 / 755 |

Paired (per seed; mean, seed s.e.): resume - CFold2 success -0.100 / +0.217
/ -0.120 (-0.001, 0.109); death -0.480 / -0.567 / -0.443 (3/3); timeout
+0.580 / +0.350 / +0.563 (3/3); detour -0.027 / +0.663 / +0.067.  resume -
CF1 success -0.103 / +0.180 / -0.340; death -0.47 (3/3); timeout +0.55
(3/3).  Milestone probes 30k -> 40k -> 50k -> 60k: north s0 0.13 -> 0.39 ->
0.58 -> 0.38, s1 0.18 -> 0.22 -> 0.52 -> 0.76, s2 0.57 -> 0.65 -> 0.54 ->
0.38; entrance reach s0 0.49 -> 0.44 -> 0.20 -> 0.22, s1 0.63 -> 0.60 ->
0.43 -> 0.53, s2 0.55 -> 0.32 -> 0.28 -> 0.35; share of resets that reach
hazard 1 within 100 steps 0.44 / 0.39 / 0.18 -> 0.02 / 0.02 / 0.00.

Reading.  The round-2 loss was not the loss of a stable capability: the
restart froze a still-moving process near its 30k state (CFold2 is at the
CF1 level on success), whereas continuing it with the carried state moves
it further in the same direction in all three seeds -- the actor stops
going through the hazard (deaths -> 0.00-0.03), and either the route share
rises while the detour is completed about half the time (s1: detour 0.78,
success 0.50, timeout 0.48) or the walker stops reaching any route (s0 /
s2: 68-71 % of episodes with no route, success 0.14-0.16).  Within round 2
the comparison CF2 - CFold2 (same restart in both arms) stands; the
practical loss CF2 - CF1 is the restart plus the continuation, not the
regenerated futures.  The user's hypothesis that the restart explains the
loss is confirmed for the restart's *direction* (it stopped the process)
but the continued process is not the capable one either.

## What is now established, what is not

* Established: at the actor's real training rows the objective does not
  favour the teacher detour torques (1a); the route turns 0-11k updates
  after the critic margin appears and fluctuates by 0.2-0.3 between
  checkpoints 1,000 updates apart (1b); walking collapses occur before and
  without any route change and recover (1b); continued training with the
  carried state drives deaths to zero and timeouts to 0.5-0.86 on the same
  futures (2); the 30k evaluations of the pilot are single draws from an
  unconverged process whose replicate spread equals the seed spread (1b).
* Withdrawn: "the full objective is at an exact tie at the reset rows"
  (task-goal artefact); "the mid-route regression is interference from the
  majority rows" (a single SGD displacement, not the history); "the actor's
  detour rate has an equilibrium at 0.12-0.21" and "the 0.50 of round-1
  seed 2 was a transient" (the process moves in the other direction when
  continued).
* The walking collapses are critic runaways -- one update at a time, with a
  frozen-critic control: `spike/SUMMARY.md` (user's lead from the training
  logs; 2026-09-19, later).
* Not established: what drives the walking collapses (RESOLVED in `spike/SUMMARY.md`: the critic); whether the
  objective pushes toward the actor's own route-changing torques (1a
  measures teacher transplants only); why the continuation goes to "no
  route" in seeds 0 / 2 and to the detour in seed 1.

Mainline unchanged (NCE, actor objective, 8-dim torques, bc 0.05); the
replay / resumption runs are diagnostics, the futures oracle.  Checkpoints
stay on the nodes (diag_replay/CF_s*/upd_*.pkl, round2/resume/CF/seed_*/).
