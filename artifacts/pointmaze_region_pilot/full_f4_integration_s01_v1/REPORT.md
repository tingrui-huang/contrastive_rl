# Original full-F4 Contrastive RL + BC integration

Pessimistic ETT augmentation did not establish added native benefit over both offline continuation and observational augmentation.

One fixed experiment: seeds 0 and 1; 1,000 updates per trained actor from the same original step-150,000 full checkpoint. I is unchanged; O is offline continuation; B is observational augmentation; C is pessimistic ETT augmentation. Models are the final diagonal/critic-driven ETTs from 817e429. All six actors actually changed.

Seed 0 favors C over B in return (+0.4103, 95% CI [0.0050, 0.8479]), success (+4.30 percentage points) and failure (-4.30 points). Seed 1 does not resolve any of these advantages. Neither seed resolves C over O. The conditional two-seed mean C-O return difference is +0.0343 [-0.2145, 0.2680], and C-B is +0.1703 [-0.1143, 0.4877]. These results are insufficient to claim benefit beyond both controls; they do not establish equivalence or that longer training would behave the same.

## Native outcomes

256 fresh paired native resets per actor, horizon 50, stochastic tanh-Gaussian execution for every actor. Return = sum(.95^t * reward), where reward uses next-XY distance <2. Success = any distance <.5; failure = absorbing death. A failure-to-succeed is not counted as an absorbing failure.

- I: discounted return 3.3236; success 29.2969%; failure 70.7031%.
- O_s0: discounted return 3.5936; success 31.2500%; failure 68.7500%.
- B_s0: discounted return 3.2850; success 28.1250%; failure 71.8750%.
- C_s0: discounted return 3.6953; success 32.4219%; failure 67.5781%.
- O_s1: discounted return 3.1143; success 26.5625%; failure 73.4375%.
- B_s1: discounted return 3.1509; success 26.5625%; failure 73.4375%.
- C_s1: discounted return 3.0811; success 26.9531%; failure 73.0469%.

## Paired native differences

Differences are left minus right. Brackets are 95% paired episode-bootstrap intervals (2,000 replicates). Return/success increases and failure decreases are favorable. The C-O and C-B contrasts are primary.

- C_s0_minus_O_s0: returns +0.1017 [-0.2109, +0.4251]; success +0.0117 [-0.0195, +0.0430]; failure -0.0117 [-0.0430, +0.0195].
- C_s0_minus_B_s0: returns +0.4103 [+0.0050, +0.8479]; success +0.0430 [+0.0077, +0.0820]; failure -0.0430 [-0.0820, -0.0077].
- B_s0_minus_O_s0: returns -0.3086 [-0.7415, +0.0972]; success -0.0312 [-0.0703, +0.0039]; failure +0.0312 [-0.0039, +0.0703].
- O_s0_minus_I: returns +0.2700 [-0.2035, +0.7152]; success +0.0195 [-0.0273, +0.0625]; failure -0.0195 [-0.0625, +0.0273].
- B_s0_minus_I: returns -0.0386 [-0.5417, +0.4380]; success -0.0117 [-0.0586, +0.0312]; failure +0.0117 [-0.0312, +0.0586].
- C_s0_minus_I: returns +0.3717 [-0.0745, +0.8363]; success +0.0312 [-0.0117, +0.0742]; failure -0.0312 [-0.0742, +0.0117].
- C_s1_minus_O_s1: returns -0.0332 [-0.4154, +0.3318]; success +0.0039 [-0.0312, +0.0391]; failure -0.0039 [-0.0391, +0.0312].
- C_s1_minus_B_s1: returns -0.0698 [-0.3978, +0.2385]; success +0.0039 [-0.0312, +0.0391]; failure -0.0039 [-0.0391, +0.0312].
- B_s1_minus_O_s1: returns +0.0366 [-0.3031, +0.3808]; success +0.0000 [-0.0312, +0.0312]; failure +0.0000 [-0.0312, +0.0312].
- O_s1_minus_I: returns -0.2093 [-0.7174, +0.2779]; success -0.0273 [-0.0742, +0.0195]; failure +0.0273 [-0.0195, +0.0742].
- B_s1_minus_I: returns -0.1727 [-0.6588, +0.2584]; success -0.0273 [-0.0742, +0.0156]; failure +0.0273 [-0.0156, +0.0742].
- C_s1_minus_I: returns -0.2425 [-0.6777, +0.1672]; success -0.0234 [-0.0625, +0.0156]; failure +0.0234 [-0.0156, +0.0625].

Binary discordant counts are retained in results.json. These intervals condition on each trained actor; two seeds do not estimate broad training-seed uncertainty. No multiplicity correction was applied.

Descriptive paired mean across the two fixed seeds (joint resampling of shared episode indices):

- C_minus_O: returns +0.0343 [-0.2145, +0.2680]; success +0.0078 [-0.0156, +0.0312]; failure -0.0078 [-0.0312, +0.0156].
- C_minus_B: returns +0.1703 [-0.1143, +0.4877]; success +0.0234 [-0.0039, +0.0527]; failure -0.0234 [-0.0527, +0.0039].
- B_minus_O: returns -0.1360 [-0.4343, +0.1459]; success -0.0156 [-0.0430, +0.0098]; failure +0.0156 [-0.0098, +0.0430].

## Actual policy changes

On a fixed bank of 1,024 offline F4 states with the commanded goal: KL(initial Gaussian || final Gaussian), RMS/max L2 distance between tanh means, average Gaussian-scale change, and parameter L2. These were measured after training and never used as update rejection criteria.

- O_s0: KL 0.093861; action RMS/max 0.154250/0.832885; scale change +0.012299; actor parameter L2 1.29773; critic parameter L2 1.39653; 1,000 joint updates.
- B_s0: KL 0.165814; action RMS/max 0.379822/1.898854; scale change +0.099214; actor parameter L2 1.71353; critic parameter L2 2.02171; 1,000 joint updates.
- C_s0: KL 0.137405; action RMS/max 0.458348/1.963593; scale change +0.029463; actor parameter L2 1.67044; critic parameter L2 1.84734; 1,000 joint updates.
- O_s1: KL 0.069869; action RMS/max 0.125998/0.845530; scale change +0.084834; actor parameter L2 1.34280; critic parameter L2 1.39865; 1,000 joint updates.
- B_s1: KL 0.113885; action RMS/max 0.312008/1.825831; scale change +0.164564; actor parameter L2 1.58785; critic parameter L2 1.97550; 1,000 joint updates.
- C_s1: KL 0.107308; action RMS/max 0.415135/1.938539; scale change +0.153193; actor parameter L2 1.79378; critic parameter L2 1.62426; 1,000 joint updates.

## Independent frozen-model rollouts

Each cell contains 128 independent 50-step rollouts from START, paired across actors, with the specified model acting at every step. These are actual simulated trajectories, not critic predictions. Full paired differences versus I and primary C-O/C-B contrasts are in results.json. The model has no validated absorbing-death label: zero return is reported separately and is not a failure label.

The unchanged actor scores 8.24-10.00 under the four models versus 3.32 natively. Thus model rollout scores are poor estimates of native performance here. Observationally augmented B improves versus I on its own model in both seeds (+1.3222 [0.7373, 1.9353] and +1.1258 [0.5055, 1.7509]); this does not transfer into a resolved native improvement. C does not show resolved own-model improvement: -0.1012 [-0.3557, 0.0946] and -0.0291 [-0.2046, 0.1167]. The full cross-model matrix below avoids evaluating only on each arm's training model.

### obs_s0

- I: return 8.3276; geometric success 95.3125%; zero return 1.5625%.
- O_s0: return 10.0423; geometric success 98.4375%; zero return 0.0000%.
- B_s0: return 9.6498; geometric success 97.6562%; zero return 0.0000%.
- C_s0: return 9.4010; geometric success 96.8750%; zero return 0.7812%.
- O_s1: return 9.6472; geometric success 98.4375%; zero return 1.5625%.
- B_s1: return 9.2538; geometric success 96.8750%; zero return 2.3438%.
- C_s1: return 9.6313; geometric success 96.0938%; zero return 1.5625%.
- C_s0_minus_O_s0 return -0.6413 [-1.1425, -0.1861].
- C_s0_minus_B_s0 return -0.2488 [-0.7262, +0.2323].
- C_s1_minus_O_s1 return -0.0159 [-0.4958, +0.4304].
- C_s1_minus_B_s1 return +0.3775 [-0.1055, +0.8794].

### pess_s0

- I: return 9.9263; geometric success 85.9375%; zero return 11.7188%.
- O_s0: return 9.9546; geometric success 85.9375%; zero return 11.7188%.
- B_s0: return 9.8287; geometric success 84.3750%; zero return 14.0625%.
- C_s0: return 9.8251; geometric success 84.3750%; zero return 13.2812%.
- O_s1: return 9.8389; geometric success 84.3750%; zero return 14.0625%.
- B_s1: return 9.9113; geometric success 85.1562%; zero return 12.5000%.
- C_s1: return 9.8152; geometric success 84.3750%; zero return 14.0625%.
- C_s0_minus_O_s0 return -0.1295 [-0.3805, +0.0763].
- C_s0_minus_B_s0 return -0.0035 [-0.0213, +0.0131].
- C_s1_minus_O_s1 return -0.0236 [-0.2015, +0.1380].
- C_s1_minus_B_s1 return -0.0961 [-0.2626, -0.0013].

### obs_s1

- I: return 8.2360; geometric success 93.7500%; zero return 2.3438%.
- O_s0: return 9.7778; geometric success 97.6562%; zero return 0.7812%.
- B_s0: return 9.6986; geometric success 99.2188%; zero return 0.0000%.
- C_s0: return 9.3927; geometric success 97.6562%; zero return 1.5625%.
- O_s1: return 9.6349; geometric success 99.2188%; zero return 0.0000%.
- B_s1: return 9.3618; geometric success 98.4375%; zero return 0.7812%.
- C_s1: return 9.5675; geometric success 96.8750%; zero return 0.7812%.
- C_s0_minus_O_s0 return -0.3851 [-0.8387, +0.0452].
- C_s0_minus_B_s0 return -0.3060 [-0.7570, +0.1475].
- C_s1_minus_O_s1 return -0.0674 [-0.4381, +0.3076].
- C_s1_minus_B_s1 return +0.2057 [-0.2523, +0.6835].

### pess_s1

- I: return 9.9959; geometric success 88.2812%; zero return 10.9375%.
- O_s0: return 10.0112; geometric success 88.2812%; zero return 10.9375%.
- B_s0: return 9.9629; geometric success 86.7188%; zero return 11.7188%.
- C_s0: return 9.9668; geometric success 87.5000%; zero return 11.7188%.
- O_s1: return 10.0319; geometric success 88.2812%; zero return 10.9375%.
- B_s1: return 9.9204; geometric success 87.5000%; zero return 11.7188%.
- C_s1: return 9.9668; geometric success 87.5000%; zero return 11.7188%.
- C_s0_minus_O_s0 return -0.0444 [-0.2110, +0.0875].
- C_s0_minus_B_s0 return +0.0039 [+0.0000, +0.0116].
- C_s1_minus_O_s1 return -0.0651 [-0.2304, +0.0543].
- C_s1_minus_B_s1 return +0.0464 [+0.0000, +0.1184].

## Learner and synthetic-action BC treatment

The driver directly calls the existing original full-F4 learner. It restores actor, critic, target critic and Adam states. Binary NCE, full-F4 achieved-goal relabeling, actor diagonal-logit maximization, BC=.5, random_goals=.5, batch=256, both learning rates=.0003 and Adam eps=1e-7 remain unchanged. There is no region-Q update, cumulative KL/action-drift rejection, or new gradient clipping.

Every synthetic executed action enters the same BC term as an offline action, duplicated for the hindsight goal and rolled random goal by the existing actor loss. Natural nominal actions are auxiliary model inputs only. B and C use exactly the same treatment; the experiment does not isolate or attribute effects specifically to BC.

B queries diagonal_s at (executed action, executed action). C queries critic_s with the current executed action and an independently sampled action from the frozen nominal policy. The current actor generates all new paths. Each augmented actor uses 25,600 synthetic rows out of 256,000 (10%); trajectories refresh at updates 0/250/500/750. Per refresh: 32 roots at each time 0/10/25/40, rolling to the original horizon 50. Full remaining future goals use the same geometric sampler and matched indices; no padding is eligible.

Offline continuation uses the existing 5,940-episode continuation partition. The original checkpoint saw all 6,600 episodes; the excluded 660 episodes therefore are not independent validation of that checkpoint. Only obs/act feed training.

## Budget, provenance and limits

Exactly 6,000 joint updates; 64,000 training-model transitions + 179,200 independent evaluation-model transitions = 243,200 model transitions; 89,600 native steps. Frozen transition/nominal hashes and the unchanged initial learner state verified. All final checkpoints were sealed before native outcomes. Native outcomes never entered learning or checkpoint selection. No additional experiment followed.

This is a modest repository integration experiment, not an exact paper reproduction. Model error, partial observability, finite training budget and different inherited diagonal heads limit interpretation. Inherited validation diagonal-energy degradation from 817e429 was +0.000181/+0.000249 for B (seeds 0/1), and +0.014969/+0.008099 for C; this comparison does not isolate the ETT response map. A pessimistic ETT optimized against I is not established worst-case for the updated actors; mixed-data training alone does not establish robust max-min optimization. Intervals crossing zero are inconclusive, and observed ties do not prove equivalence.

Artifacts: PROTOCOL.md, preregistration.json, learner_config.json, partition.npz, final full-state checkpoints, rounds/*/h*.npz, *_batch_audit.npz, *_native.npz, *_model.npz, training.json, results.json, ledgers, audit.json.

Reproduce only if separately requested, in a fresh directory: `python -m ett.pointmaze_full_f4_integration prepare --out <dir>`, then phases `train` and `evaluate`; `python -m ett.report_full_f4_integration --out <dir>` audits saved arrays without more rollouts.
