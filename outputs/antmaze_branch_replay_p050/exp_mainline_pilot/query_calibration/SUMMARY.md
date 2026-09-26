# Local calibration of the critic at the actor's output actions (user's design after 4b33528; 2026-09-22; `scripts/exp_v6_query_calibration.py`; pre-registered in `manifest.json` before any result; a DIAGNOSTIC arrangement, not the mainline)

Question: the S ETT gives ~3 % goal-area mass for the actions the collapsed critic reads at ~35 %; handed to the critic's training, does
the over-estimate come down, and does the actor walk again?

Correction carried from the user's review: the earlier query-extension round already ran under clip 0.1 (4b33528 called it "pre-clip"
-- wrong); its difference from this design is the agent's own continuation.

## Design (fixed before any result)

* Fixed: the S ETT (`ett_one_step_v4s20`, the draw-1 table's generator settings), the start agent's continuation, the training state =
  draw-1 seed-4 at 20,000 updates (the pre-collapse actor with its critic and both optimizer states).
* Query anchors: the 4,245 start-region anchors (region start, t <= 40; 991 episodes) split by SOURCE EPISODE into Q (2,121 anchors,
  108 resets; 3.9 % of the anchor weight) and H (2,124; 88 resets), seed fixed.  Queries per Q anchor: the frozen 20k actor's mode
  under the task goal + 3 torques sampled from the same actor = 4 branches (8,484); every consequence kept (no success filter, no
  direction, no simulator rows).  Control: 4 extra branches at the LOGGED torque per Q anchor (8,484; new context / onset draws).
* The query branches' own gamma-law goal-area mass 0.037 (mode 0.037, samples 0.036; the control's 0.037): the critic's target at the
  queried actions is the same ~3-4 % as at the logged action.
* Critic-only continuation for 10,000 updates (the run's remaining budget) in three arms -- plain (no extra branches), control, query --
  from the same state, the actor restored after every scan (hash-verified), NCE / clip 0.1 / lr unchanged; the critic stream re-seeded
  as the run's and fast-forwarded 20k batches (the first critic batch's anchor ids hash-identical in the three arms; the control differs
  in goals only, the query in goals and actions); at a Q anchor the future is drawn uniformly from its 1 + 4 branches (weights
  unchanged); extra rows = 3.15 % of the critic rows in both arms.
* Then the corrected short-update control (`diag_v6_stall_onset.py short`, 5,000 actor-only updates from the 20k actor + Adam state,
  identical batches, each critic frozen), the ruler on queried / held-out / independent states and on the updated actors' new modes,
  and the 300-episode evaluation (mode policy; two common draws 8909 / 10909).

## 1. Calibration (`critic_ruler/qc1_reset`, `qc1_indep`): goal-area over-estimation, critic (twin-min) mass / real mass, median per state

| critic | saturated final action (NEVER queried): H / Q / indep | 20k mode a20k (queried at Q): H / Q / indep | logged | progressing (draw 2) | start agent |
|---|---|---|---|---|---|
| 20k (the start point) | 9.5 / 9.5 / 9.6 | 8.4 / 6.5 / 8.9 | 1.9 / 1.7 | 2.6 / 2.5 / 2.0 | 0.38 / 0.42 |
| original final (30k, joint) | 14.2 / 18.4 / 16.1 | 4.8 / 5.4 / 5.2 | 1.8 / 1.8 | 1.1 / 1.3 / 1.0 | 0.09 / 0.10 |
| plain (critic-only, no extra) | 12.0 / 15.2 / 12.6 | 7.1 / 5.6 / 6.8 | 1.7 / 1.9 | 1.3 / 1.3 / 1.0 | 0.10 / 0.10 |
| control (logged repeats) | 8.8 / 10.3 / 8.1 | 4.9 / 5.7 / 6.1 | 1.7 / 1.9 | 2.6 / 2.4 / 2.0 | 0.15 / 0.25 |
| **query** | **1.9 / 2.3 / 2.1** | **2.3 / 1.9 / 2.4** | 1.5 / 1.5 | 1.25 / 1.14 / 0.94 | 0.16 / 0.15 |

(H = the 28 diagnostic reset states of held-out episodes, Q = the 36 of queried episodes, indep = the 64 independent reset states, non-
anchors.)  Critic masses at the saturated action: plain 0.28-0.30, control 0.19-0.22, query 0.047-0.049 (real 0.02-0.05, ETT 0.024-0.036).

* Pre-registered reading "calibration improved": MET.  At the never-queried saturated action on held-out episodes the query critic's
  factor is 1.9 vs the control's 8.8 (a 4.7x reduction; the control's own drop from plain is 27 %), the same on the independent
  states (2.1 vs 8.1); at the queried 20k mode on H states the query critic reads 0.063 vs the ETT's 0.046 (1.4x).  The calibration
  generalises across states (H, non-anchors) and to an action outside the queried set (the saturated corner).
* Not fixed by the queries: the under-estimate of the actions the data executes (the start agent's 0.15x, the logged 1.5x) -- not queried.
* The residual ~2x is the query critic's level for every action (logged 1.5, stalled 1.9, a20k 2.3, progressing 1.2) -- a flat bias,
  no longer a preferential one.

## 2. The actor under each frozen critic (`stall_onset/qc/SHORT.md`; from the 20k actor: leave-the-start 0.60, saturation 0.37)

| frozen critic | left the start at 100 steps | xy disp median | torque saturation | E Q(stalled) - E Q(progressing) on the 2,048 start rows (share) |
|---|---:|---:|---:|---|
| plain | 0.40 | 0.71 | 0.46 | +0.86 (0.82) |
| control | 0.47 | 0.87 | 0.54 | +0.51 (0.82) |
| **query** | **0.80** | **10.5** | **0.18** | +0.41 (0.80) |
| original final (own) | 0.38 | 0.58 | 0.55 | +0.87 (0.81) |
| draw-2 final | 0.80 | 10.4 | 0.18 | -1.03 (0.21) |

* "Less stall": MET -- 0.80 vs the control's 0.47 (+0.33), saturation 0.18 vs 0.54; the query critic's actor equals the draw-2
  critic's on the start test.  Under the query critic the stalled policy still scores above the progressing one on the actor's rows
  (+0.41), but the updated actor finds a lower loss than either (6.002 vs the stalled 6.133) -- the residual preference is too small
  against BC to pull the actor to the corner.

## 3. The updated actors' new modes (`critic_ruler/qc2_*`): another over-estimated action?

Under the query critic the after_query mode reads 2.26 (H) / 2.94 (Q) / 2.38 (indep) x the real goal-area mass (mass 0.055-0.069 vs real
0.024-0.033, ETT 0.027-0.035) -- the same level as every other action under that critic, not a new corner.  The pre-registered threshold
"> 2x at H" is touched numerically (2.26) by the critic's flat ~2x bias, not by a preferential score; under the plain / control critics
the after_query mode reads 3.8 / 3.3 (H), the after_plain mode 15.5 / 10.4, the draw-2 actor's mode 1.0 / 2.2.  Reading: the query
actor did NOT move to another spuriously high action at the reset states.

## 4. The full task (300 episodes, mode policy; draws 8909 / 10909; `eval/summary.json`)

| actor | success | death | timeout | detour share | shortcut share | no route (stall) | detour completion (n) |
|---|---|---|---|---|---|---|---|
| 20k actor (the start of the short updates) | 0.267 / 0.253 | 0.047 / 0.033 | 0.687 / 0.713 | **0.420 / 0.443** | 0.067 / 0.043 | 0.513 / 0.513 | 0.59 (126) / 0.54 (133) |
| after plain critic | 0.147 / 0.097 | 0.063 / 0.083 | 0.79 / 0.82 | 0.113 / 0.113 | 0.107 / 0.093 | 0.78 / 0.79 | 0.91 (34) / 0.77 (34) |
| after control critic | 0.277 / 0.220 | 0.027 / 0.023 | 0.70 / 0.76 | 0.320 / 0.270 | 0.040 / 0.033 | 0.64 / 0.70 | 0.82 (96) / 0.79 (81) |
| **after query critic** | **0.293 / 0.300** | **0.403 / 0.433** | 0.303 / 0.267 | **0.180 / 0.193** | **0.577 / 0.580** | 0.243 / 0.227 | 0.70 (54) / 0.91 (58) |
| after own final critic | 0.123 / 0.153 | 0.09 / 0.06 | 0.79 / 0.79 | 0.09 / 0.12 | 0.13 / 0.10 | 0.78 / 0.77 | 0.82 (27) / 0.89 (37) |
| after draw-2 critic | 0.213 / 0.183 | 0.49 / 0.54 | 0.30 / 0.27 | 0.017 / 0.033 | 0.683 / 0.683 | 0.30 / 0.28 | 1.0 (5) / 1.0 (10) |
| original final (30k, joint) | 0.057 / 0.050 | 0.003 / 0.017 | 0.94 / 0.93 | 0.057 / 0.067 | 0.013 / 0.020 | 0.93 / 0.91 | 0.88 (17) / 0.70 (20) |

* "Task preserved": NOT MET.  The query critic removes the stall (no-route 0.51 -> 0.24, timeouts 0.69 -> 0.30) and the success moves
  +0.03 / +0.05 (300-episode s.e. ~0.026) -- but the DETOUR SHARE falls from 0.42-0.44 to 0.18-0.19 and the shortcut share rises from
  0.04-0.07 to 0.58, deaths from 0.03-0.05 to 0.40-0.43.  This is the pattern the user named in advance: the stall is removed and the
  policy is pulled back to the shortcut.  The draw-2 critic does the same, further (detour 0.02-0.03, shortcut 0.68, deaths 0.5).
* Mechanism visible in the earlier numbers: the 20k actor's detour share was carried by the saturated kick -- executed once and handed
  to the start agent the saturated / progressing first actions enter the far route in 17 / 16 % of the reset states, the logged torque
  in 1.6 % (first_action_consequences); calibrating the critic at the start region removes the kick and with it the heading that the
  start agent turns into the far route.  Whether a detour preference exists in the critic at the fork itself is not tested here.

## Verdict against the pre-registered readings

| reading | result |
|---|---|
| calibration improved at held-out states and never-queried actions | MET (1.9-2.3x vs 8.1-8.8x control; generalises to non-anchors) |
| less stall under the query critic | MET (0.80 vs 0.47; saturation 0.18) |
| the new mode is not another over-estimate | met in substance (the query critic's flat ~2x, not a corner) |
| task preserved (detour share / completion) | NOT MET (detour 0.42 -> 0.18, shortcut 0.07 -> 0.58, deaths x8; success +0.03 n.s.) |

A better-calibrated score at the start region is not a fix: it converts the collapse into the draw-2 / shortcut behaviour.  Per the
user's rule the joint pipeline is NOT to be launched on this basis, and by the stop rule the answer to "keep adding queries" is no --
not because the actor found a new over-estimated action (it did not), but because the static calibration at the start removes the
route along with the stall.  The route preference itself (the fork, not the start) is the next thing to localise; not launched.

Files: `manifest.json` (pre-registration), `critics/*/train_manifest.json`, `gen_*.log` (the branches' statistics), `critic_ruler/qc1_reset|qc1_indep|qc2_reset|qc2_indep/`, `stall_onset/qc/`, `eval/`.  Branch tables, critics, rollouts and the updated actors stay on the nodes (query_branches_*.npz, critics/*/final.pkl on 30027; after_*.pkl on all three).
