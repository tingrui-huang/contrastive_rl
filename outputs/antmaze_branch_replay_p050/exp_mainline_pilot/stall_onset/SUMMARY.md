# Localising the start stall of the Learned (S futures) seed 4 (user's plan after 282ce17; 2026-09-21 night; diagnostics only, no training variant, no checkpoint selected)

`scripts/diag_v6_stall_onset.py roll / objective / short` (node 30027); the seed-4 repeats `exp_v6_ett_futures.py train --rerun-tag node3`
(node3, the machine of the original draw-1 seed 4); outputs `stall_onset/s4d1/{ROLL, OBJECTIVE_vs_d2final, OBJECTIVE_vs20k, SHORT}.md`
+ json, `ett_futures_v4s20{,_draw2}/rerun_node3/CF/seed_4/`, `stall_onset/rerun_s4_hashes.log`.

## 1. WHEN: the milestones rolled from the same 128 start states (64 reset anchors + 64 independent resets; hazards off; mode policy; 100 steps)

| checkpoint (seed 4 unless noted) | left the start (> 1.0) at 30 / 50 / 100 | xy displacement median at 10 / 30 / 100 | torque saturation share (abs > 0.99) | first-step torque deviation vs init |
|---|---|---|---:|---:|
| init (= the start agent) | 0.99 / 0.99 / 1.00 | 0.48 / 2.65 / 10.4 | 0.05 | - |
| draw 1, 10k | 0.96 / 0.97 / 0.96 | 0.75 / 2.72 / 8.8 | 0.11 | 0.87 |
| draw 1, 20k | 0.59 / 0.60 / 0.60 | 0.60 / 1.27 / 5.8 | 0.37 | 1.06 |
| draw 1, final (30k) | **0.13 / 0.17 / 0.19** | 0.33 / 0.40 / 0.40 | **0.63** | 1.35 |
| draw 2, 10k / 20k / final | 0.85 / 0.95 / 0.96 (at 30) | ... 8.2 / 10.6 / 7.9 (at 100) | 0.15 / 0.07 / 0.09 | 0.76 / 0.92 / 0.90 |
| seed 0 draw 1, 10k / 20k / final | 0.95 / 0.77 / 0.80 (at 30); 0.98 / 0.91 / 0.96 (at 100) | | 0.09 / 0.15 / 0.15 | 0.97 / 1.10 / 1.11 |

The stall develops progressively between 10k and 30k (leave 0.96 -> 0.60 -> 0.19), at the same pace in the reset anchors and the
independent resets, and it is a SATURATION of the torques at the start states: the stalled actor pushes 63 % of its joint torques
against the bound (the progressing checkpoints 7-15 %; the start agent 5 %), i.e. a locked posture, not a zero action.  On the TRAINING
batches (the logged buffer's states) the same quantity is 0.04 -> 0.07 in every run alike (seed 4 draw 1 / draw 2, seed 0, the repeats:
policy scale 0.06-0.08, pre-tanh |loc| max 8 -> 30, BC NLL -17 -> -14, no spikes in any actor / critic loss or gradient norm) -- nothing
in the training trace marks the run; the saturation is specific to the start-region states the actor visits at deployment.

## 2. REPEATS on the same GPU and software (node3, RTX 4090 Laptop, jax 0.10.2; hashes of both tables, the ETT models, the advice generator, anchors and the start checkpoint verified identical; the repeats' q_init / policy_init and first critic / actor batches hash-identical to the originals')

| run | success (10909) | deaths | timeouts | no route (of 300) | detour |
|---|---:|---:|---:|---:|---:|
| draw 1, seed 4, original (node3) | 0.057 | 0.017 | 0.927 | 271 | 22 |
| draw 1, seed 4, REPEAT (node3) | 0.343 | 0.000 | 0.657 | 166 | 133 |
| draw 2, seed 4, original (node 30049) | 0.457 | 0.250 | 0.293 | 9 | 193 |
| draw 2, seed 4, REPEAT (node3) | 0.367 | 0.107 | 0.527 | 44 | 215 |

The collapse does NOT reproduce exactly: with the same table, seed, machine and identical first batches the losses already differ at
update 500 (actor loss 5.718 vs 5.801) -- the GPU training is not bit-deterministic -- and the repeat ends half-stalled (166 / 300
without a route, 0.343) instead of fully (271, 0.057).  The recovery under table draw 2 reproduces in direction (9 -> 44 stalls, 0.457
-> 0.367).  So (table, seed) fixes a tendency, not an outcome: under draw 1 seed 4 sits in a basin where run-to-run nondeterminism
decides between 0.06 and 0.34, under draw 2 between 0.37 and 0.46; the run-to-run spread of one configuration (0.1-0.3) is of the
order of the table effect and of the seed spread.

## 3. WHAT the complete actor objective says (the 64 reset anchors' visited states at steps 0 / 5 / 10 / 20 / 30 of the stalled final and of a progressing policy; objective = 0.95 (-min-twin Q) + 0.05 NLL(a_logged), the actor loss's weighting; task goal and 8 relabelled future goals of the anchor's own logged episode)

| critic | goal | states of | critic prefers the stalled action | full objective prefers it | at the exact reset rows (step 0): critic ; objective |
|---|---|---|---:|---:|---|
| own (draw 1, final) | task | stalled traj. / progressing traj. | 0.52 / 0.67 | 0.52 / 0.71 | **1.00 ; 0.97** |
| own (draw 1, final) | relabelled | stalled / progressing | 0.63 / 0.71 | 0.67 / 0.63 | **1.00 ; 1.00** |
| other (draw 2, final) | task | stalled / progressing | 0.11 / 0.25 | 0.13 / 0.40 | 0.09 ; 0.05 |
| other (draw 2, final) | relabelled | stalled / progressing | 0.21 / 0.27 | 0.24 / 0.41 | 0.00 ; 0.00 |
| own at 20k vs the final action | task, step 0 | | 0.22 | 0.39 | (the 20k critic prefers its own 20k action) |

At the reset rows the run's own critic rates the stalled (saturated) action above the progressing policy's action in 100 % of the states
and the full objective (BC included) in 97-100 %; the draw-2 critic prefers the progressing action in 91-100 %.  The BC term does not
rescue it: the NLL of the logged first torque is large under BOTH policies (29 under the stalled, 41 under the progressing -- both far
from the logged action with a policy scale of ~0.07), so at 0.05 it does not decide.  (NLL rows for steps > 0 use the reset row's logged
action as the target and are not a valid BC term; the step-0 rows are exact.)

Short actor-only updates from the pre-collapse actor (seed 4 draw 1 at 20k, WITH its optimizer state; 5,000 updates with the recipe's
actor loss and stream; the critic frozen; then rolled from the same 128 start states):

| frozen critic | left the start at 100 | xy disp median | torque saturation | policy param delta |
|---|---:|---:|---:|---:|
| own, 20k | 0.60 | 2.7 | 0.40 | 13.6 |
| own, final (30k) | 0.60 | 6.8 | 0.35 | 15.2 |
| draw 2, 20k | **1.00** | 10.2 | 0.08 | 14.5 |
| draw 2, final | **0.98** | 10.6 | 0.08 | 16.9 |

The same actor, the same optimizer state, the same actor batches: under the draw-2 critics it walks again within 5k updates (saturation
0.40 -> 0.08); under its own critics it stays where it is.  The parameter movement is the same size under every critic (no abnormal
update), only its direction differs.

## Reading against the user's decision tree

* Abnormal parameter updates: NO -- no spikes, equal-size parameter deltas, training-batch statistics identical across runs.  Optimizer
  stability is not the branch to take.
* The bad critic rates the stalled action higher: YES -- at the start states the draw-1 seed-4 critic prefers the saturated action
  (100 % at the reset rows), the draw-2 critic does not, and swapping the critic alone reverses the actor.  This is the branch: "which
  conditional future supervision produced this preference" -- not yet examined (the per-anchor futures of the two tables at the
  start-region anchors and the two critics' Q-landscape between the logged torque and the saturated one; the user's design call).
* The full objective prefers progress while the update leaves it: NO -- the objective under the own critic prefers the stall.
* What the repeats add: the preference is a basin, not a switch -- with the same table and seed the run lands at 0.06 or 0.34 depending
  on GPU nondeterminism, and the two tables' identical marginals do not tell them apart.  Any future comparison of "fixes" on this seed
  must therefore be read against a run-to-run spread of ~0.3, i.e. on several repeats, never on one run.
* Not touched, as instructed: BC, the ETT, the number of futures (the two-table mixture already failed at 0.311).
