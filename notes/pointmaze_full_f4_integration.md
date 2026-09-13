# Predeclared full-F4 CRL + BC integration

One bounded experiment on branch feature/pointmaze-causal-transition, HEAD
97177d684b00841a2a1314461f0132f33ff113d8. Final transition offsets are those
in 817e429's update_reference_s01_v1/final_kernels.npz. No model fitting,
ETT retraining, sweeps, budget extensions, native training, or checkpoint selection.

I is the unchanged original alpha0_seed0/final.pkl actor at step 150000.
O_s, B_s, C_s restore that SAME full learner state, including both Adam states,
critic and target critic; only learner RNG is replaced with the paired seed.
O continues offline only; B augments with diagonal_s{s}, queried with both
channels equal to its executed action; C augments with critic_s{s}, independently
sampling natural actions from the frozen state-goal nominal policy. All base
transition parameters, offsets and nominal parameters stay frozen.

Use the existing run_policy_improvement.learner_setup and unmodified
crl.losses.build_learner. Original binary NCE, full eight-dimensional F4 achieved
future goals, in-batch negatives, actor diagonal LOGIT objective, random_goals=.5,
BC=.5, entropy coefficient=0, single critic, batch=256, representation=64,
MLP=(256,256), both learning rates=.0003, Adam eps=1e-7, gamma=.95, tau=.005.
No region critic, decoded region-Q actor update, KL/action-drift rejection,
gradient clipping, acceptance test, or added regularizer. Finite-value checks
stop an invalid run, without retuning or replacing an iterate.

BC applies to EVERY sampled row identically in B and C: synthetic executed
actions are treated as recorded actions. random_goals=.5 duplicates each action
for its achieved hindsight goal and rolled random goal. Synthetic natural
actions x_prime never become BC targets. This tests the whole learner/data
combination and does not isolate the causal contribution of BC.

Continuation data follow the existing integration partition: all behavior
sources, 5940 episodes after excluding permutation(6600, seed=0)[:660]. Only
obs/act are loaded. The original checkpoint previously saw all 6600 episodes;
this partition is not a claim of independent checkpoint validation.

Seeds 0 and 1; exactly 1000 joint critic/actor updates per arm, 6000 total.
Exactly 10% synthetic rows cumulatively in each augmented arm, using the
existing 25/26-row schedule, shared full offline draws and batch permutations.
Four refreshes, before updates 0,250,500,750. At each refresh sample 32 offline
episodes per root time 0,10,25,40 (128 roots), uniformly with replacement.
Generate paths with the arm's current stochastic actor for the FULL remaining
horizons 50,40,25,10 respectively. Keep only the current refresh's paths.
Each synthetic path is uniformly sampled, then its anchor uniformly sampled,
then j>i from its valid path with probability proportional to .95^(j-i), using
the repository trajectory sampler. Padding is never eligible. Root draws,
rollout keys, replay RNG and goal-time draws are paired between B/C. Root-time
stratification is a fixed coverage choice, not a tuned parameter.

Train model transitions: 4 augmented learners * 4 refreshes * 32 * 125 = 64000.
All six final full-state checkpoints and I must be saved and hashed before any
native outcomes. Evaluate seven actors on 256 fresh reset seeds 150000000+e,
50 steps each (89600 native steps), stochastic tanh-Gaussian actions with paired
keys 151000000+t, no extra noise or deterministic switch. Train actions use the
same sampling law. Task reward is next-XY distance<2; raw discounted return is
sum(.95^t*r); native success is any XY distance<.5; failure is absorbing death,
distinct from non-success. Continue absorbed paths through external horizon 50.

Independent model evaluation: seven actors on each of obs0,pess0,obs1,pess1,
128 full 50-step rollouts from public START, common fresh key 152000000.
179200 model evaluation transitions, total model budget 243200 (hard cap).
Report actual rollout returns, model geometric success and zero-return rate;
the visible F4 model has no validated absorbing-death label, so do not invent a
model failure rate. No model/native evaluation is used to update or select.

Prespecified native contrasts per seed: C-O and C-B (primary), B-O and each
trained actor minus I (secondary). 2000 paired episode-bootstrap replicates,
seed 153000000, percentile 95% intervals, binary discordant counts. Report
per-seed estimates and descriptive mean across the two paired seed differences
with the same episode index resampled jointly; this is conditional on these
two training seeds, not population training-seed uncertainty. No multiplicity
adjustment. Evidence of added benefit requires positive lower return bounds
for C-O and C-B in both seeds without resolved success/failure worsening.
Intervals crossing zero are inconclusive; ties do not establish equivalence.
Cross-model paired contrasts use the same bootstrap rule.

Measure initial-to-final parameter L2, fixed offline-bank Gaussian KL,
mean-action RMS/max L2 and scale changes. These are diagnostics, never rejection
criteria. Preserve raw paths, batch indices, metrics, final states, source/input
hashes and ledgers. Frozen models can be inaccurate; B/C also differ in diagonal
heads. An ETT optimized against I need not be worst-case for updated actors.
Mixed-data training alone is not robust max-min optimization. This is a repository
integration experiment, not exact paper reproduction. Stop after reporting.
