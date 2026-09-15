# Bounded supervised-ETT to Contrastive-RL pilot

Sealed before optimizer updates or candidate-result inspection on 2026-09-15. The working branch is `feature/pointmaze-causal-transition` at `4ad4aaf3cf85f17652ab8663ad63e7216e599525`. This is a one-seed exploratory pilot, not a seed-robust result. Existing files and historical outputs are read-only; no commit or push is part of the experiment.

## Question and fixed route

Test the complete route: previously collected same-context intervention tuples -> a learned stochastic F4 transition and failure-onset model -> complete generated trajectories -> the repository's sigmoid-NCE critic and paper actor objective -> paired native policy evaluation. A failed learned-model preflight stops the experiment; the G4 rule backend is never a fallback.

## Frozen supervision

Training combines only two established training partitions:

1. `supervised_ett_native_probe_v1/paired_transitions.npz`, episodes 0--71 (14,400 rows). Episodes 72--95 remain previously inspected development evidence and are not used for fitting or checkpoint selection.
2. `pointmaze_supervised_repair_20260914_v1/train_paired.npz`, all 128 established training episodes (25,600 rows). Its 32-episode validation collection is the sole checkpoint-selection and preflight partition. Its 64-episode final collection remains excluded; it has already been inspected and is not fresh confirmation.

Each row is the same-context tuple `(visible F4 state s, recorded shadow/expert advice xb, executed query xq, observed successor y)`. The recorded `xb` is used verbatim. Successful, fatal-onset, ordinary unsuccessful, and already-failed outcomes are retained in the audit. Movement and onset fitting use alive-before rows, including fatal incoming transitions; already-failed rows define and audit the persistent branch rather than a live-motion target. For the older source, `onset = not dead_before and dead_after`; the later source supplies the identical explicit `onset` field. Hidden swamp bits, teacher modes, force-safe flags, reward, `dead_before`, `dead_after`, and `onset` are never model inputs. Death fields are supervision/audit labels only. There is no new training interaction.

Diagonal means `max(abs(xq - xb)) <= 1e-6`, computed directly from recorded actions. Diagonal and off-diagonal Energy Scores are averaged separately and enter with equal weight.

## Learned backend

The live movement model is a small conditional stochastic generator. Its visible inputs are standardized `s`, raw `xb`, raw `xq`, and derived interactions `xq-xb`, `xq*xb`, and `(xq-xb)^2`. A 128-128 context MLP and a 4-dimensional standard-normal innovation feed a 128-unit generator head. The head emits a bounded two-dimensional displacement; eight independent emissions form the full emitted-XY Energy Score U-statistic. No nominal-anchor rectangle or old Lipschitz certificate is used.

Static legality is endpoint-only: emitted XY must be inside `[0,9] x [0,5]` and in a zero cell of the fixed public two-route wall grid. An illegal proposal becomes a stationary live proposal at the current XY. This is not native collision simulation and does not infer death. It allows the demonstrated downward fork exit. The check makes no claim that every free endpoint has a legal continuous path.

The onset head is a separate 64-64 MLP receiving the same visible conditional features plus the candidate displacement. It is trained with natural-prevalence binary cross-entropy on actual alive-row successors. Positive and negative rows are stratified for variance reduction and weighted back to their empirical prevalence. Its gradient does not update movement. There is no hand-written hazard mask, landing-bin disagreement rule, or stationary-is-death rule.

At rollout, live XY is sampled first, then onset is sampled from the learned head. Fatal incoming XY is retained. Once onset is sampled, subsequent XY is frozen while F4 shifts exactly as `[xy_next, s[:6]]`. The latent failed bit exists only inside the rollout engine and is initialized false only at documented environment-reset roots; one declared trace trajectory starts from a collected alive-before tuple. No native state, hidden death label, environment step, recorded-successor lookup, or rule backend enters rollout.

## Optimization and preflight

Seed 2026091501; Adam, learning rate 3e-4, 2,500 fixed updates, batch 256 per diagonal/off-diagonal term, eight movement samples, onset batches of 256 positive and 256 negative rows, onset loss weight 0.25. Validate every 100 updates and select the earliest minimum of `diag_ES + off_ES + 0.25 * onset_BCE`. One architecture and schedule only.

The model advances only if all predeclared checks pass on the established validation partition:

- finite losses/parameters, movement parameter delta L2 > 1e-4, onset parameter delta L2 > 1e-4;
- diagonal ES <= 0.10 and off-diagonal ES <= 0.30;
- at same-context fork groups, predicted query-effect magnitude is at least 25% of observed query-effect magnitude and cosine/correlation of predicted versus observed displacement effects is >= 0.35;
- for demonstrated downward exits (`xq_y <= -0.5` and observed `dy <= -0.2`) the predicted mean `dy <= -0.15` and at least 25% of samples have `dy <= -0.2`;
- onset AUROC >= 0.75, mean predicted onset on actual onsets exceeds non-onsets by >= 0.15, and alive stationary-history contexts do not exceed 0.15 mean onset probability;
- exactly zero recovery after sampled failure in a 1,024-path persistence test;
- exact F4 shifts and zero endpoint-legality violations over validation samples.

These are representation/safety gates, not a requirement that every downward action succeeds and not a return-based substitute. A failed gate saves the checkpoint and localized report and stops before policy training.

## Complete replay and unchanged CRL

If the gate passes, generate 3,300 complete 50-step model trajectories. Trajectory 0 is the declared trace anchor: it begins at one collected alive-before fork tuple and uses its recorded `xq` for the first query; its successor and every later transition are generated. The other 3,299 roots are the known reset F4 state `[0.5,3.5]` repeated four times and therefore start alive by reset semantics. At every generated live step, `xb` is sampled from the frozen expert-only K5 nominal and `xq` from the frozen observational `alpha0_seed0` actor; the actor is sampled, not converted into a detour controller. Later calls see the generated state. All paths continue to horizon, including absorbing tails.

The learned-ETT replay contains 6,600 complete episodes: a seed-2026091504 permutation selects 3,300 original observational episodes and the other 3,300 are complete learned-model trajectories (50% synthetic by episode and transition). The control is all 6,600 original observational episodes. This composition changes critic and actor training contexts in the ETT arm; no claim is made that only positives change. Rare lower-route coverage comes from whatever is present in the randomly selected original half and from the disclosed frozen rollout policy's stochastic support; it is measured, not hard-coded or filtered. No path is selected by success or failure.

Both arms run from independently constructed but hash-verified identical seed-0 initialization with the same bounded recipe: 30,000 learner updates, batch 256, ten updates per scan, representation 64, hidden layers 256-256, discount 0.95, random goals 0.5, no TD/CPC/GCBC/twin Q/failure bank/ranking/AWR, entropy 0, and actor `bc_coef = 0.05`. The repository's original sigmoid-NCE, geometric future sampler, and actor loss are unchanged. No native evaluation occurs during training; only fixed final checkpoints are evaluated.

Before CRL training, the exact deterministic replay sampler stream is materialized for all 30,000 x 256 NCE positives after reproducing the offline audit's RNG consumption. It records arm, source trajectory, anchor time, and future time; dataset and code hashes make every positive auditable.

## Final evaluation and interpretation

Evaluate the two fixed final actors under mode and sampled actions on the same 200 native reset seeds, pairing sampled innovations. Report reach, strict success, lower-route usage, absorption, and discounted return. These evaluation episodes never enter fitting, checkpoint selection, or replay. Historical G4 BC-0.05 results are only a separately labeled mechanism reference.

The experiment learns a conditional outcome distribution. Supervised fitting is not certified worst-case optimization; no 1-Lipschitz or pessimistic guarantee is claimed. G4 kept recorded motion and inserted rule-selected absorbing tails, whereas every synthetic successor here is emitted by the learned backend.

## Budget and wall clock

Hard caps: 2,500 ETT updates; 3,300 generated paths x 50; 30,000 learner updates per CRL arm; 200 paired episodes x 50 x two policies x two action protocols for final native evaluation. On the provided RTX 3060 Ti, estimated end-to-end wall clock is 25--40 minutes (ETT 5--10, CRL 4--8 per arm, generation/audits/evaluation/reporting the remainder). The run is fixed-final and is not extended based on results.
