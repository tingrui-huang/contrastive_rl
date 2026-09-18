# Mainline contract: counterfactual future supervision for offline CRL (AntMaze V6)

Written 2026-09-19 after commit 157a212.  This file separates three things
that the AntMaze V6 branch-replay chain had let run together: (a) the method
the project is about, (b) the simulator-oracle pilot that tests one part of it
under a matched procedure (`scripts/exp_v6_mainline_pilot.py`), and (c) the
historical diagnostic configurations whose results stay on file but no longer
stand as statements about the method.

## 1. What is being tested

The question: under an otherwise matched offline CRL training procedure, does
replacing **observational** future supervision (the positive future goal of a
logged anchor (s_t, a_t) is drawn from the recorded continuation of that
episode) with **counterfactual** future supervision (the positive future goal
is drawn from what a fixed blind agent gets after executing a_t at s_t) improve
the policy?

Nothing about the objective, the architecture, the action space, the data or
the evaluation is part of the intervention.  Only the source of the critic's
positive futures is.

### 1a. The intended offline learned-ETT method (not implemented here)

Retains, unchanged:

- single-step 8-dimensional torque actions;
- offline training data only (the logged dataset);
- the original Monte-Carlo binary NCE critic objective (`crl/losses.py`,
  `critic_loss`, `use_td = use_cpc = False`, twin heads, in-batch negatives);
- the original actor objective with BC coefficient 0.05 (`actor_loss`:
  `(1 - 0.05) * (alpha * log pi - min_h f_h(s, a, g)) + 0.05 * (-log pi(a_logged | s, g))`,
  `random_goals = 0`, alpha = 0);
- the intervention: **counterfactual generation** of the critic's positive
  futures by a learned effect-of-treatment-on-the-treated (ETT) model of the
  blind agent's continuation, fitted offline.

Does NOT introduce: macro actions, route commands, subgoals, route-label
supervision, ranking losses, failure-bank losses, or simulator-derived actor
targets.

### 1b. The simulator-oracle pilot implemented in this task

The same as 1a except that the counterfactual futures are produced by the
simulator (restore the logged state, execute the logged torque once, let the
fixed blind agent continue) instead of a learned ETT.  The simulator is used
only as the oracle data-generation backend of arm CF and for evaluation.  A
positive result is oracle evidence for the sampling design; it is **not** an
offline learned-ETT result, and the learned-ETT pipeline is not claimed to be
complete by anything in this pilot.

### 1c. Historical diagnostic configurations (on file; not the method)

The following were used in the V6 chain to locate failure modes.  They are
diagnostic tools.  None of them is part of the method by default, and results
obtained with them are statements about those configurations, not about 1a:

- reset-only actor training (the actor's critic-term batches anchored at the
  dataset's row 0 = environment reset; `row0_prepare` on a dataset copy);
- frozen-critic actor retraining (`V6_JOINT_MODE=frozen`, critic lr 0);
- the strong d20 actor initialisation (`joint_van_d20/seed_0`, or any d20
  artefact) on any other data;
- S1/S2 batch swapping (`exp_same_batch`);
- native simulator branches as a substitute for a learned ETT (every branch
  replay so far, including this pilot's arm CF -- see 1b);
- pure-BC initialisation, displacement-balanced BC rows
  (`bc_sampling='balanced'`), BC weights other than 0.05, gamma sweeps,
  proposal-and-rank / segment-rank decoding, candidate-query extensions
  (agent-sampled or BC-sampled first actions), repeated hazard draws per key.

Historical reports are preserved as written.  Where a historical conclusion
was stated as if it were about the method, the report's header now carries a
"status under MAINLINE_CONTRACT" note saying which configuration it actually
tested (see the SUMMARY files of `exp_detour_ratio`, `exp_agent_round`,
`exp_same_batch`, and `notes/antmaze_branch_replay_plan.md`).

## 2. The sampling contract (fixed before training)

Three interfaces, declared explicitly in `scripts/exp_v6_mainline_pilot.py`,
never inferred from a replay file's row layout:

1. **Critic anchor selection** -- `AnchorSet`.  K = 60,000 i.i.d. draws from
   the recipe buffer's own anchor law (an episode uniformly, then a row
   uniformly over that episode's valid anchor rows [0, L_e - 1)), seed
   131000001.  A row drawn several times is one anchor whose weight is its
   multiplicity / K (53,747 unique anchors).  Every anchor keeps its episode
   id, timestep, 29-dim state, logged torque, task goal, merged-pool source
   episode and weight.  Both arms read the same `anchors.npz`; anchor
   identities and effective weights are equal by construction, and the number
   of continuations generated for an anchor (one) never enters its weight.

2. **Critic positive-future selection** -- `RecordedFutures` (arm O) /
   `BranchFutures` (arm CF), read by `CriticStream`.  For anchor k with a
   path of len_k rows (row 0 = the logged anchor row), the future row m is
   drawn with P(m) proportional to gamma^m over m = 1..len_k - 1: the buffer's
   geometric relabeling law truncated at the path's end.  Arm O's path is the
   recorded episode from row t; arm CF's is the generated branch.  Negatives
   are the other anchors' positive goals in the batch (the original in-batch
   rule), so each arm's negative marginal is its own positive marginal; the
   two marginals differ and are documented, not forced to agree.  Any regional
   critic readout must use the arm's own marginal.

3. **Actor state / goal selection and BC supervision** -- `ActorStream`: the
   recipe's `TrajectoryBuffer` over the whole d05 file under its default law
   (episode uniform, row uniform over the valid rows, geometric future goal
   truncated at the episode end), seed 20000 + training seed.  `random_goals
   = 0`: the critic term is evaluated at (s_i, g_j) with the policy's sampled
   action and the BC term is `-log pi(a_i | s_i, g_j)` on the same rows, BC
   0.05.  No reset restriction, no balancing, no relabeling change.  Identical
   across arms for a training seed.

Row semantics that the historical chain conflated: in the original dataset,
row 0 of an episode is the environment reset; in a branch file, row 0 is an
arbitrary logged timestep.  Applying `row0_prepare` to both and calling the
result matched is not acceptable and is not done here.

**Interface note.**  Feeding the critic loss and the actor loss different
batches is a sampling-interface change: `crl.losses.build_learner(...,
separate_actor_batch=True)` evaluates the unchanged critic loss on the
critic rows and the unchanged actor loss (its critic term and its BC term on
the same rows) on the actor rows.  The objectives are the original's; the
shared-batch pairing of the original implementation is not reproduced
byte-for-byte, in either arm.

**Query actions.**  The queried first action at every anchor is the logged
torque a_t, in both arms.  Agent-sampled or BC-sampled query candidates are a
separate, already-studied dimension (`exp_agent_round`, `--extra-query-ckpt`)
and are not part of this comparison.

## 3. The two arms

Data: the p050 d05 offline dataset
(`artifacts/rockfall_clock_v6/dataset/antmaze_rockfall_clock_v6_p050_d05_gxy.npz`,
sha256 4533c702...; 1,000 episodes = 50 detour + 950 shortcut, 267,481
transitions, all successful teacher episodes ending on their reach frame).
The whole file is the training set; evaluation uses fresh native draws; the
old 100 held-out and the 30 Cnew reserved pool episodes are excluded from
d05 by construction (checked).  No d20 data, policy, selected action or
representation enters either arm.

- **Arm O**: logged anchors; positive futures from the recorded
  continuations.
- **Arm CF-oracle**: the same anchors and weights; the logged torque executed
  once from the restored logged state at absolute time t (hazard latents,
  both clocks and the rock jitter redrawn from the priors, seed 132000000 +
  anchor id; `build_v6_branch_replay.restore`); then one fixed blind
  continuation agent, its mode tanh(loc), closed-loop until the first reach
  frame, death, or the absolute horizon 800.  No goal hold (the recorded
  episodes end on their reach frame; the historical replays parked to the
  horizon -- a deliberate departure from them).  Every outcome kept; no
  success or route filter.  One branch per anchor.  Generated once, before
  training; never regenerated during training.

Continuation agent = the actor initialisation of both arms = the existing
d05 vanilla agent, training seed 0: `joint_van_d05/seed_0/final.pkl` (the
row "d05 vanilla seed 0" of `exp_detour_ratio/SUMMARY.md`; provenance chain
d05-only: pure-BC 100k on d05 -> frozen vanilla critic 30k on d05 -> 30k
frozen-critic actor on d05; observation = 29-dim Ant state + goal xy, no
privileged field).  Chosen before any pilot result.  Its checkpoint hash is
pinned in the manifest when the file is present.  Not substituted by pure BC,
by the d20 agent, or by a retrained d05 agent.

Existing d05 branches (`replay_policy_d05.npz`: pure-BC continuation, dense
strata + every-60th anchors, goal hold, two draws, sampled candidates) do not
match this contract and are not reused.

## 4. Matched training

Both arms: joint critic + actor updates; the V6 recipe (`run_v6_branch_replay.base_config`:
twin-Q binary NCE, batch 1024, 1024x1024 MLPs, repr 16, Adam 3e-4 for both,
alpha 0, tau 0.005, bc 0.05, random_goals 0), gamma 0.999 (the branch-chain
recipe's discount; the frozen V6 ladder used 0.99; pinned, disclosed, not
swept); critic initialised fresh at PRNGKey(seed) -- identical across arms
for a seed; actor initialised from the start agent's policy parameters,
fresh Adam states for both optimizers; **30,000 optimizer updates** counted
in the loop (one critic Adam step and one actor Adam step per update, in
groups of 4 inside one jitted scan; 7,500 scan calls); milestones 10k / 20k
saved but never selected; the evaluated checkpoint is the final one.  Three
paired training seeds (0, 1, 2): same critic init, same critic anchor
sequence, same actor batches across arms.  Same device and software per
seed as far as practical (recorded per run).

## 5. Checks before spending compute

`manifest.json` is sealed before generation and training (dataset and
checkpoint hashes, episode identities, anchor law and weights, continuation
policy and action mode, generation / evaluation seeds, horizon, terminal
handling, goal hold, future law, training configuration and update count,
comparisons and rules).  `audit` then verifies: identical critic anchor
identities and weights across arms; the logged query executed exactly once
before the continuation; branch roots keep the original episode / timestep /
state / action; actor batches identical across arms per seed and drawn from
ordinary non-reset rows; future goals never cross an episode or branch
boundary; the static offline gates (G1-G8) pass; no d20 artefact and no
held-out episode in training; after training, critic and actor parameters
both moved.  A failed check is repaired, not compensated for.  The earlier
physics, pretraining, episode-breadth and coverage studies are not repeated.

## 6. Evaluation and rules

The fixed start agent, arm O (3 seeds) and arm CF (3 seeds) on the same 300
fresh native episodes (env seed 3909, p_active 0.5 / 0.5, horizon 800), mode
policy as the primary endpoint; the reserved validation seeds 616000005 /
616500000 stay untouched.  Reported: success, detour rate, death, timeout,
success with / without an active hazard, per training seed, and paired
per-episode differences.  Evaluation uncertainty (episode s.e. per seed;
episode-paired bootstrap with the seeds fixed) is reported separately from
training-seed variability (seed s.e. over the three paired seeds); three
seeds are three seeds.

Primary: CF - O.  Practical (required separately): CF - start; beating a
degraded control alone is insufficient.  Rule: mean over the three paired
seeds > 2 x the seed s.e. and 3/3 seeds in the same direction; ties never
count against.  True-reset candidate ranking is a secondary diagnostic only;
the old mixed Cdev gate does not reject the experiment.  No checkpoint,
continuation, seed or batch source is chosen after seeing the evaluation.

## 7. Stopping rule

One paired pilot, run only after the checks pass.  If CF does not improve:
report the negative result under this contract, separate what it establishes
from what it leaves untested, and do not launch more diagnostics or tune a
new configuration.  If it improves: report oracle evidence for the sampling
design and name the replacement of the oracle by an offline-learned ETT as the
remaining methodological step.  If the required d05 checkpoint is
unavailable, finish the code and configuration audit, report that blocker,
and substitute nothing.
