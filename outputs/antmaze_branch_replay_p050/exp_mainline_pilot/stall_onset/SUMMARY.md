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
actor loss and stream; the critic frozen; then rolled from the same 128 start states) -- SUPERSEDED (round 2): this run used different actor
batches per condition and restored the critic only after each 4-update scan; the corrected control is in ROUND 2 below:

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

# ROUND 2 (user's review of b4aa391; 2026-09-22): corrections, the diagnostic fixed and redone, the real consequences of the first actions, the score decomposition

## Corrections to the record

* reset_futures/SUMMARY.md: the draw-1 table's goal-area mass at the reset anchors is BELOW the simulator table's (0.046 vs 0.059 on the 64
  anchors; 0.047 vs 0.059 on all 196), not "1.5x above" as written; it is above draw 2's (0.020 / 0.031).
* "The draw-1 critic's Q rises monotonically to the saturated action under every goal set" was too strong: under the ACTOR stream's
  relabelled goals many curves peak in the interior (argmax interior 52-84 % of the states in those rows).  What holds: the boundary
  preference under the TASK / near-goal goals (argmax at the bound in 84-98 %) and, weaker, under the critic-training marginals.
* The first short-update control (stall_onset/s4d1/SHORT.md) had two implementation faults -- the actor stream was created once outside
  the critic loop (different batches per condition) and the critic was restored only after each 4-update scan (three of four updates saw
  a moved critic) -- and its "complete objective" used the mode action's Q, not the expected Q of sampled actions as trained.  Its
  numbers are superseded by the corrected control below; the direction it showed survives, the size does not.
* "Optimizer factors excluded" is too strong: the training log samples one update in 500; no spike was seen, which does not exclude
  spikes between samples.  Stated as "none observed at the logged resolution".

## The corrected control (`diag_v6_stall_onset.py short`, `stall_onset/s4d1_fixed/`): from the 20k actor + its Adam state, 5,000 actor-only updates -- the SAME actor batches (re-seeded stream, first-batch hash identical in every condition) and the same keys; the critic restored after EVERY update and hash-verified unchanged; then rolled from the 128 start states

| frozen critic | left the start at 100 | xy disp median | torque saturation | policy param delta |
|---|---:|---:|---:|---:|
| own, 20k | 0.49 | 0.95 | 0.48 | 15.1 |
| own, final | 0.43 | 0.74 | 0.49 | 17.6 |
| draw 2, 20k | 0.86 | 7.4 | 0.20 | 15.8 |
| draw 2, final | 0.83 | 10.4 | 0.16 | 18.2 |

The 20k actor itself: 0.60 / 0.37.  Under its own critics the actor sinks further into the stall (0.60 -> 0.43-0.49, saturation 0.37 ->
0.48); under the draw-2 critics it recovers most of the way (0.83-0.86, saturation 0.16-0.20) -- less than the faulty run's 0.98-1.00.  The
critic's update direction is sufficient to deepen or relieve the stall on the 100-step start test; this says nothing about the whole task.

The objective AS TRAINED -- E_{a ~ pi(s, g)}[min-twin Q(s, a, g)] with 16 sampled actions (common random numbers) plus the BC NLL of the
logged action, loss = 0.95 (-E Q) + 0.05 NLL -- on 2,048 valid logged rows of the start region (the 1,000 reset rows t = 0 and 1,048
start_early rows; the actor stream's relabelled goals):

| policy | BC NLL | E Q / loss under own 20k | under own final | under draw-2 20k | under draw-2 final |
|---|---:|---|---|---|---|
| 20k actor | -11.75 | -6.78 / 5.86 | -6.98 / 6.04 | -7.15 / 6.20 | -7.70 / 6.73 |
| stalled final | -8.53 | -6.63 / 5.87 | **-6.58 / 5.83** | -7.39 / 6.59 | -8.12 / 7.29 |
| progressing (draw 2 final) | -11.72 | -7.18 / 6.24 | -7.45 / 6.49 | -6.86 / 5.93 | **-7.09 / 6.15** |

E Q(stalled) - E Q(progressing): under the own critics +0.56 / +0.87 (the stalled policy higher on 81-83 % of the rows; on the reset rows
+0.84 / +1.37, 94 %); under the draw-2 critics -0.53 / -1.03 (21-23 %; reset rows 8-13 %).  The BC term is WORSE for the stalled policy
(NLL -8.5 vs -11.7: 0.05 x 3.2 = 0.16 in the loss) and is outweighed by the critic term (0.95 x 0.87 = 0.82): under its own critic the
actor's loss is lower for the stalled policy (5.83 vs 6.49) -- the actor follows its objective; under the draw-2 critic the objective
prefers the progressing policy (6.15 vs 7.29).

## The real consequences of ONE first action (`diag_v6_first_action_consequences.py`; `first_action_consequences/REPORT.md`): the 64 reset states; the logged torque / the stalled actor's mode / the progressing actor's mode / the start agent's mode, then the frozen START agent; paired hidden draws (simulator: 4 classes x 16; the S ETT with the learnt advice and sampled onset: 4 x 8)

| first action | simulator: success / death / timeout / far entry | sim goal-area / near-2.0 / death-frame mass | S ETT: success / death / timeout / far entry | ETT goal-area / near-2.0 mass |
|---|---|---|---|---|
| logged torque | 0.235 / 0.737 / 0.027 / 0.016 | 0.0401 / 0.0321 / 0.0089 | 0.248 / 0.752 / 0.000 / 0.000 | 0.0328 / 0.0250 |
| stalled actor's | 0.271 / 0.527 / **0.202** / **0.172** | 0.0333 / 0.0311 / 0.0061 | 0.266 / 0.685 / 0.050 / 0.050 | 0.0289 / 0.0230 |
| progressing actor's | 0.239 / 0.574 / 0.187 / 0.156 | 0.0340 / 0.0279 / 0.0069 | 0.250 / 0.684 / 0.066 / 0.031 | 0.0321 / 0.0247 |
| start agent's | 0.233 / 0.738 / 0.029 / 0.020 | 0.0352 / 0.0278 / 0.0085 | 0.232 / 0.740 / 0.027 / 0.000 | 0.0344 / 0.0269 |

Paired vs the logged torque (simulator; mean +- s.e. over states): the stalled action's success +0.035 +- 0.032, timeouts +0.175 +- 0.054,
goal-area mass -0.007 +- 0.011, death-frame mass -0.003 +- 0.001; the progressing action's success +0.003 +- 0.025, timeouts +0.160.  The
model: stalled success +0.018 +- 0.021, timeouts +0.050 +- 0.026, goal-area -0.004 +- 0.003.  Per-state contrast correlations simulator vs
model ~0 (as in every earlier crossover).

* Executed ONCE and handed to the start agent, the saturated action is NOT a bad action: its success equals or slightly exceeds the logged
  torque's (n.s.), it removes deaths (-0.21) at the price of timeouts and far-route entries (0.17 -- the kick puts the ant on a heading the
  start agent turns into the far route), and the goal masses the critic is trained on are equal or slightly LOWER than the logged torque's
  (goal area -0.007).  The model agrees in the success and the goal masses and under-states the timeouts / far entries (0.05 vs 0.20 /
  0.17).
* So, of the user's three cases, this is the third: the one-step consequence with the start continuation -- what the branch tables
  supervise -- does not describe the failure of "the new actor keeps acting like this"; the stall is a closed-loop property, and neither
  the simulator nor the ETT rates the single extreme step badly.  Adding queries at these actions under the same continuation rule would
  give the critic a target that is NOT lower than the logged torque's, i.e. it would not remove the preference by itself.  CORRECTION
  (critic_ruler/SUMMARY.md): on one ruler the critic reads the stalled action's goal-area mass at 0.35 against a target of 0.03, so a
  query there WOULD remove the over-estimate; what it leaves is a near-tie between the actions, not a preference for walking.  And the critic's
  +2 to +4 nats at the task goal are not supported by the goal masses even on the critic's own terms (they are lower).

## The score split into |phi(s, a)|, |psi(g)| and the angle (`diag_v6_reset_futures.py decompose`; `reset_futures/DECOMPOSE.md`)

f = phi(s, a) . psi(g) per twin head, at the 64 reset states; every cosine is NEGATIVE here (task goal -0.77 .. -0.86; the training
marginal -0.45 .. -0.55): the critic says "unlikely" for every action, and the ranking of actions is then driven by |phi|.

| critic | action | abs phi | cos with the task goal's psi | twin-min logit (task) |
|---|---|---:|---:|---:|
| draw 1 seed 4 | stalled | **1.28** | -0.77 | -7.0 |
| | progressing | 1.51 | -0.86 | -9.1 |
| | start agent | 1.93 | -0.83 | -11.2 |
| | logged | 1.76 | -0.78 | -9.4 |
| draw 2 seed 4 | stalled | **2.07** | -0.81 | -10.4 |
| | progressing | 1.60 | -0.85 | -8.6 |
| | start agent | 2.24 | -0.82 | -11.4 |

For the draw-1 critic the stalled action has the SHORTEST phi (ratio 0.85 to the progressing action) and a slightly less negative angle
(+0.08): with negative cosines a shorter phi is a less negative product; holding the angle at the progressing action's value, the norm
change alone accounts for about +1.3 of the +2.1 nats, the angle change alone for about +1.2 (they overlap).  For the draw-2 critic the
stalled action has the LONGEST phi (ratio 1.29), which with the same negative angle makes it worse.  So the extreme action's high score in
the collapsed run comes to a substantial part from the representation length shrinking toward the saturated corner under negative
angles -- the situation in which the recipe's existing `repr_norm` option (config.repr_norm, off in the sealed recipe) becomes a
targeted candidate; not shown effective, not launched.

## Where this leaves the decision tree

* No abnormal updates at the logged resolution; the stall is the actor following its objective under its own critic (loss 5.83 vs
  6.49 for the progressing policy), with BC at 0.05 too weak to hold it (0.16 vs 0.82).
* The critic's preference is not backed by the consequences: the single extreme step is not worse than the logged torque in the
  simulator or in the ETT, and its goal masses are slightly lower -- the preference is an extrapolation artefact (half of it the
  phi-norm shrinkage), formed on a table whose reset-row positives happened to be richer, and which a second draw does not form.
* Not concluded: that repr normalisation fixes it (untested); that reducing the successful futures would help (no basis); that more
  futures per anchor would (the two-table mixture gave 0.311).
