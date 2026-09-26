# Agent-update round 1 (policy-continuation futures) and the query extension: summary

> **Status under `notes/MAINLINE_CONTRACT.md` (2026-09-19): historical diagnostic configuration.**  The fixed agent is the d20 agent; every actor here was trained with a FROZEN critic from that agent's initialisation, with balanced BC rows, and (as `exp_same_batch` later showed) the reference actors' critic-term batches were anchored at the dataset's reset rows while the branch actors' were anchored at replay row 0.  'Every branch arm below the fixed agent; the recorded-data critic +0.15' is superseded as a statement about the method (contract section 1c).  The first-step sensitivity finding (the fixed agent's own first torque reproduces its natural trajectories; a BC / recorded first torque flips the route) and the query-extension mass audit stand as diagnostic facts.  Nothing below was edited.

Sealed `manifest.json` (2026-09-18 14:27, before any generation) and the
addendum `manifest_ext.json` (16:05, before any extended replay).  Tables:
`REPORT.md` (`scripts/exp_v6_agent_round.py report`), `replay_check.json`,
`replay_check_ext.json`; diagnostics `diag_round1/` (labels under the fixed
agent's continuation), `diag_round1_bc/` (labels under the BC continuation),
`diag_first_step/` (natural-trajectory reproduction and first-step
replacement).  One round only: fix the agent -> generate futures -> train
critics -> update the agent.  Oracle transitions throughout (a diagnostic
upper bound for a learned ETT; the nominal policy is not the continuation).

## Fixed inputs

- Fixed agent: `joint_van_d20/seed_0/final.pkl` (sha d0869ab3...), chosen by
  the standing "seed 0" rule; the weakest of the three d20 vanilla actors on
  the 909 draw (0.543 / detour 0.507).  BC walker `joint_purebc_d20/seed_0`
  (5d03b2c1...).  Dataset d20 (200 detour + 800 shortcut episodes).
- Control replay `replay_policy_d20.npz` (r1 protocol: 1,500 dense anchors x
  {recorded, 2 BC samples} x 2 paired draws + every-60th general anchors x
  {recorded, 1 BC sample}; continuation = the BC walker; 19,378 paths).

## Round 1: same anchors, queries and draw seeds, continuation = the fixed agent

`build_v6_policy_replay build --query-ckpt <BC> --cont-ckpt <agent>`
(new option: candidates from one policy, continuation by another).
`verify_replay`: anchors, query torques and hazard-draw seeds identical to
the control (19,378 paths; 1,857 s with 40 workers).  The futures barely
differ from the control's:

| stratum (recorded candidate) | round1 reach / death / around | control reach / death / around |
|---|---|---|
| start_early | 0.36 / 0.61 / 0.15 | 0.35 / 0.60 / 0.16 |
| turn | 0.71 / 0.02 / 0.76 | 0.71 / 0.02 / 0.77 |
| north_leg | 0.90 / 0.00 / 0.97 | 0.89 / 0.00 / 0.98 |
| general | 0.65 / 0.25 / 0.23 | 0.66 / 0.27 / 0.22 |

By anchor time and episode route at the start stratum: from shortcut
episodes' t = 0..5 anchors the agent continuation goes around 0.00-0.04
(control 0.00-0.01); from detour episodes' t = 1..5 anchors both go around
0.7-0.9.  The agent, which detours on half of its own natural draws, does
not detour after a recorded or BC first torque.

### Why (first-step check, `scripts/diag_v6_first_step_replay.py`)

Natural draws (seed 909) re-run in-process: hidden draws identical 300 /
300, outcome + route as stored 258 / 300, exact step counts 76 / 300 (the
stored evaluation used GPU inference, the rerun CPU; step difference median
5).  Through the generator path (restore + the natural hidden draws + the
agent's OWN first torque + the agent closed-loop) 60 / 60 selected episodes
reproduce their natural outcome and route with xy deviation <= 4e-9: the
protocol reproduces natural trajectories exactly.  First-step replacement at
the same reset states, same hidden draws, agent continuation from step 2:

| reset states | first torque | around | success | death |
|---|---|---|---|---|
| 30 where the agent naturally detoured | own | 1.00 | 0.90 | 0.00 |
| | BC mode / BC samples / teacher shortcut | 0.00-0.10 | 0.20-0.23 | 0.73-0.80 |
| | teacher detour torque | 0.67 | 0.70 | 0.23 |
| 30 where it naturally went straight | own | 0.00 | 0.23 | 0.50 |
| | BC mode / samples / teacher shortcut | 0.00-0.07 | 0.30-0.40 | 0.60-0.67 |
| | teacher detour torque | 0.83 | 0.83 | 0.07 |
| dataset t = 0 anchors, shortcut episodes (20 x 2 draws) | own / recorded / BC mode | 0.55 / 0.00 / 0.00 | 0.60 / 0.28 / 0.28 | 0.12 / 0.72 / 0.72 |
| dataset t = 0 anchors, detour episodes | own / recorded / BC mode | 0.25 / 0.75 / 0.10 | 0.38 / 0.62 / 0.30 | 0.33 / 0.10 / 0.70 |

One 8-d first torque decides this agent's route: after a BC or shortcut
first torque it follows the momentum east and dies; after a north-turning
first torque (its own, or the teacher's detour torque) it goes around and
succeeds -- even from reset states where it would have gone straight (0.83).
The old query set (recorded + BC samples) never contains a north-turning
first torque at shortcut anchors, so the round-1 futures equal the
control's.  Reading 1 of the user's three: the query distribution misses
the agent's own actions; not an implementation defect (own torque
reproduces) and not mainly anchor coverage (own torque at dataset t = 0
anchors also detours).

### Critics on Cdev (development set; 96 anchors: 32 start states of the old held-out episodes, the 64 Cnew turning / north-leg anchors; candidates recorded / agent mode / 2 BC samples / 1 agent sample; 16 paired draws; region readout, deployed min; s.e. = episode bootstrap)

| labels | critic set | agreement (seeds 0 / 1 / 2) | pick gain seed-mean (paired s.e.) |
|---|---|---|---|
| agent continuation | round1 | 0.51 / 0.54 / 0.58 | +0.009 (0.007) |
| | control | 0.51 / 0.52 / 0.49 | -0.015 (0.011) |
| | vanilla (recorded data) | 0.53 / 0.54 / 0.51 | +0.013 (0.012) |
| | random | 0.51 | -0.003 |
| BC continuation | round1 | 0.49 / 0.51 / 0.57 | -0.008 (0.007) |
| | control (its own target) | 0.46 / 0.53 / 0.49 | -0.011 (0.008) |
| | vanilla | 0.49 / 0.53 / 0.45 | -0.013 (0.008) |

403 / 398 decided pairs (60 / 76 ties, 497 / 486 weak); the outcomes' own
cross-fitted selector gains +0.085 / +0.092.  round1 - control +0.024 +-
0.014 (z 1.8) under the agent labels, +0.003 +- 0.009 under the BC labels.
No critic set ranks candidate first torques at new episodes' states better
than chance under either label -- the control does not on its own target
either (as every earlier held-out probe found).

### Agents on the same fresh development draw (300 episodes, seed 2909, mean policy; all actors from the fixed agent, frozen critic, bc 0.05, d20 BC rows, 30k)

| policy | success (seeds 0 / 1 / 2) | detour | timeout | no-hazard success | hazard success | mean success vs start |
|---|---|---|---|---|---|---|
| start (fixed agent, re-evaluated) | 0.507 | 0.470 | 0.237 | 0.727 | 0.430 | -- |
| round1 (old queries, agent continuation) | 0.440 / 0.393 / 0.320 | 0.40 / 0.28 / 0.17 | 0.24 / 0.13 / 0.15 | 0.71 / 0.88 / 0.83 | 0.35 / 0.22 / 0.14 | -0.122 (s.e. 0.035; 3/3) |
| control (old queries, BC continuation) | 0.293 / 0.280 / 0.327 | 0.09 / 0.13 / 0.17 | 0.08 / 0.19 / 0.15 | 0.94 / 0.81 / 0.82 | 0.07 / 0.10 / 0.16 | -0.207 (0.014; 3/3) |
| reference (recorded-data critics) | 0.590 / 0.663 / 0.730 | 0.54 / 0.67 / 0.77 | 0.18 / 0.18 / 0.15 | 0.84 / 0.79 / 0.86 | 0.50 / 0.62 / 0.69 | +0.154 (0.040; 3/3) |

round1 - control +0.084 (pooled s.e. 0.038; 2/3 seeds).  Walking kept by
every actor (timeout <= 0.24, no-hazard success >= 0.71).

**Pre-registered rules for round 1: not met** -- the critic check shows no
advantage on new states, the updated agents are below the fixed agent (and
so below the control-beating threshold), while the same start with the
recorded-data critics gains +0.15 (3/3): one more actor round against the
vanilla critic still improves the agent; both branch critics degrade it.
No second round.

## Query extension (user direction after the first-step check)

Candidates kept (recorded + BC samples) and the fixed agent's mode + samples
ADDED at every anchor (dense 3 -> 6, general 2 -> 4; labels `ag_*`; new seed
stream); the control's draw seeds, so every candidate of an anchor is
paired; uniform anchors keep the per-stratum masses (`verify_ext`: masses
equal to 1e-9, the 19,378 shared paths identical to the control).  Two
continuations: `ext_bc` (BC walker) and `ext_ag` (the fixed agent), giving
{old, extended queries} x {BC, agent continuation} with control and round1.
ext_ag: 38,756 paths, 3,946 s (44 workers, node 30049); the agent's own
candidates at the start stratum go around 0.21 (recorded / BC 0.13-0.15),
death 0.55 vs 0.61; general 0.29 vs 0.22 -- smaller than the first-step
check suggests because the start anchors are t <= 5 rows of logged
episodes, where the state already carries the logged momentum.

ext_bc: 38,756 paths, 2,953 s (10 workers, node 30108); the same
profile under the BC continuation (start stratum: agent candidates around
0.23 vs 0.16, death 0.55 vs 0.60).  `verify_ext` passes for both (shared
paths identical, masses equal).  Critics x3 per arm (ext_ag node 30043,
ext_bc node 30108), actors x3 from the fixed agent and evaluations on the
2909 draw (both on node 30043; the node-30108 actors were OOM-killed).

### Critics on Cdev (extended sets added to the table above)

| labels | critic set | agreement (seeds 0 / 1 / 2) | pick gain seed-mean (paired s.e.) | vs control | vs round1 |
|---|---|---|---|---|---|
| agent continuation | ext_ag | 0.47 / 0.52 / 0.49 | -0.003 (0.011) | +0.012 +- 0.014 | -0.012 +- 0.010 |
| | ext_bc | 0.50 / 0.51 / 0.46 | -0.002 (0.011) | +0.013 +- 0.012 | -0.011 +- 0.011 |
| BC continuation | ext_ag | 0.53 / 0.48 / 0.51 | -0.000 (0.009) | +0.011 +- 0.012 | |
| | ext_bc (its own target) | 0.49 / 0.50 / 0.56 | -0.005 (0.008) | +0.006 +- 0.007 | |

Chance for every set under both labels; no pair of sets differs by 2 s.e.

### Agents (2909 draw; the {old, extended queries} x {BC, agent continuation} table)

| policy | success (seeds 0 / 1 / 2) | detour | timeout | no-hazard success | hazard success | mean success (s.e.) |
|---|---|---|---|---|---|---|
| start (fixed agent) | 0.507 | 0.470 | 0.237 | 0.727 | 0.430 | 0.507 |
| control: old queries, BC continuation | 0.293 / 0.280 / 0.327 | 0.09 / 0.13 / 0.17 | 0.08-0.19 | 0.81-0.94 | 0.07-0.16 | 0.300 (0.014) |
| round1: old queries, agent continuation | 0.440 / 0.393 / 0.320 | 0.40 / 0.28 / 0.17 | 0.13-0.24 | 0.71-0.88 | 0.14-0.35 | 0.384 (0.035) |
| ext_bc: extended queries, BC continuation | 0.493 / 0.410 / 0.337 | 0.50 / 0.30 / 0.15 | 0.10-0.24 | 0.77-0.91 | 0.14-0.40 | 0.413 (0.045) |
| ext_ag: extended queries, agent continuation | 0.287 / 0.353 / 0.427 | 0.07 / 0.22 / 0.31 | 0.04-0.12 | 0.84-0.94 | 0.06-0.27 | 0.356 (0.040) |
| reference: recorded-data critics | 0.590 / 0.663 / 0.730 | 0.54 / 0.67 / 0.77 | 0.15-0.18 | 0.79-0.86 | 0.50-0.69 | 0.661 (0.040) |

Pre-registered attributions (success; > 2 x pooled seed s.e. and 3 / 3):

- coverage alone, ext_bc - control: **+0.113** (s.e. 0.047; 3/3) -- met;
  detour +0.19 (2/3, within noise);
- coverage under the agent continuation, ext_ag - round1: -0.029 (0.053;
  2/3) -- not met;
- continuation given the extended coverage, ext_ag - ext_bc: -0.058
  (0.061; 2/3) -- not met;
- every branch arm vs the fixed agent: control -0.207, round1 -0.122,
  ext_ag -0.151, ext_bc -0.093 (all 3/3, > 2 s.e.); the reference +0.154
  (3/3).  Walking kept everywhere.

**Reading.**  Adding the agent's own first torques to the queries does
what the first-step check predicted for the BC-continuation replay: the
futures now contain north-turning branches at the start anchors, and the
actor trained on them detours and succeeds more than the control's (+0.11,
3/3).  It does not carry over to the agent-continuation replay (ext_ag =
round1 within noise, both below ext_bc), and no branch arm reaches the
fixed agent it started from, let alone the recorded-data critic that lifts
the same start by +0.15.  At the critic level none of the five sets ranks
candidate first torques on new episodes' states above chance under either
label, so the actor-level ordering (reference > start > ext_bc > round1 >
ext_ag ~ control) is not explained by held-out candidate ranking; it comes
from the critics' landscapes along the data the actors are trained on.

**Verdict (rules in `manifest.json` / `manifest_ext.json`):** round 1 not
met on all three checks; the query extension answers "why the round-1
futures did not change" (the query set lacked the agent's own first
torques) and improves the BC-continuation branch, but the two open
questions -- generalisation of the new signal to new states, and an actor
above the starting agent and the control -- remain negative.  No second
round.  What would change the picture is not more of the same futures:
the recorded-data critic is the only one that improves this agent, and
the branch critics trained on 19k-39k generated paths degrade it.

## Infrastructure notes

Node 30049 (RTX 3090) is power-capped at ~270 MHz ("SW Power Cap: Active";
critics at 6 steps / s) -- used for CPU work only.  Node 30021 (RTX 4090)
went down (connection refused) during the ext_bc generation; the arm was
restarted on node 30108 (3060 Ti, one learner at a time).  `pkill -f` must
not match the ssh command line; heredoc patches with backslashes corrupt
source (use Write / Edit).
