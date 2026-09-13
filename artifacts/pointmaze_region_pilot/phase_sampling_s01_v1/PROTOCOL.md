# Preregistered critic-only phase-sampling comparison

Base 23da83f. Freeze the exact zero-offset PointMaze generator, stochastic
historical actor, state-goal nominal policy, all normalizations, goal-region
reward, discount .95 and geometry. Preserve the 11-64-tanh-64-tanh-16 region
critic with learned 2x16 goal-label embeddings, h/50 feature, float64, Adam
.003/default settings, 256-row minibatches and 1500 steps. Binary NCE remains
[positive softplus(-f)+31 E_uniform-label softplus(f)]/32. Uniform q=(.5,.5),
alpha=0, Q_h=(1-.95^h)*31*.5*exp(f_1), Q_0=0; stable softplus, no clipping.
The action Lipschitz construction remains samplewise Euclidean L=1; no ETT,
actor, reward, architecture, negative distribution or loss change.

## Shared training data and the only comparison

Replay the previous initial generator's 36 train roots x8 paths exactly with
seed99000000 and positive-label seed99000100. Save the trajectories and all
sampled truncated-geometric labels once. All four critics use those identical
arrays, including both optimizer seeds. Native hidden labels are not loaded.
Normalization and groups are inherited from the old train archive. Previous
evaluation episodes are explicitly diagnostic data already inspected.

For each path of length H (40,44,47,49), phases are relative to that root:
- early: t=0..4;
- middle: t=5..floor(H/2)-1;
- late: t=floor(H/2)..H-1.

Baseline samples uniformly from every transition-label row, exactly as the
previous implementation. Alternative first samples one of the 288 trajectories
uniformly, one of three phases uniformly, then a row uniformly within that
trajectory/phase. Its probability for such a row is 1/(288*3*phase_length),
versus 1/total_rows at baseline. This slightly changes path weighting when
horizons differ; it is part of the declared row distribution, not an extra
model change. No importance correction, label balancing, reward/death
conditioning or label resampling. Phase is used by the sampler, not added as
a critic input. The conditional NCE class-prior correction stays 31.

Use paired initialization seeds98000000/1 and row-sampling seeds98000000/1,
zero Adam state, equal1500 steps for all four models, final iterate only.
Save actual row exposure counts by phase/group and initial/final parameter
hashes. Baseline seed0 must reproduce the old initial critic weights/predictions
as an integrity check; a mismatch stops evaluation for diagnosis, never tuning.
All training finishes and checkpoints are sealed before fresh final collection.

## Untouched final episodes

Existing offline episodes have upstream model exposure; old evaluation
episodes have also been inspected. Therefore reserve a genuinely new native
context source, generated ONLY after training: exactly600 episodes/30,000
native steps, seed103000000, using the verified F4 teacher collector with
random_frac=0, force_safe_prob=.05 and teacher_noise=.15. This matches the old
teacher-source behavior configuration (excluding its random-episode prefix).
This collector is a context-collection policy, NOT the fixed continuation
actor. The established teacher may inspect native bits to act; no such fields
enter selection or critic fitting. Its logged outcomes are not value targets.

Use the same observable selector as 23da83f: candidates t=1,3,6,10; outside
goal, no prior goal membership, newest-step displacement>.05; approach,
transit and bypass geometry; one episode per root, 12/group, selection order
bypass/transit/approach. Root-selection seed103000001. A fresh native episode
namespace prevents numeric ID collisions with old dataset episode IDs.
If any group has fewer than12 eligible episodes, stop and report incomplete
final coverage; no hidden-label gate, replacement collection or rule change.
Keep raw native data local; share selected observable contexts/provenance.

## Train-fit, inspected diagnostic and final calibration

1. Report empirical NCE loss over ALL saved training rows by phase/group,
   and actual sampling exposures. Single Bernoulli labels are not exact Q.
2. Independently calibrate on states/actions PRESENT in training: take the
   first sampled path for each train root, with queries at t=0,5,floor(H/2).
   These are the starts of the three phases, not a claim about uniform phase
   averages. Each first action is the recorded training-row action. Compare
   against64 fresh full continuations, holding that action fixed.
3. Report old evaluation predictions separately as already-inspected diagnostic
   results. Reuse its saved16-repeat targets; do not call it untouched.
4. For untouched final roots, generate ONE query path/root with seed105000000;
   query t=0,5,floor(H/2), using its recorded first action at each query. Draw
   64 independent continuations per query with seeds105001000+100*phase+h.
   Train-context continuations use104001000+100*phase+h. Remaining h is exact,
   and every continuation ends at original task time50. All critics score the
   SAME queries, actions and targets. Auxiliary x_prime remains separate.

For both independent cohorts report RMSE/bias with2000 episode-cluster bootstrap
replicates, range-excess fraction/max and MC SE (mean/max). Group is inherited
from the observable root, even if a later query is at the goal. Overall
intervals stratify by group; repeats are averaged before episode resampling.
Use bootstrap seed106000000. Report paired alternative-minus-baseline RMSE and
bias changes by seed/phase/group, with the same clustered resamples. MC target
noise contributes to raw RMSE; do not equate it to exact critic error.

Within each phase and root group compare ALL unordered episode pairs with
identical remaining h. Fix an informative ordering label only when full64
mean difference exceeds both .01 occupancy and 2.58*sqrt(SE_i^2+SE_j^2), and
both independent32-repeat halves agree on its sign. Report all/equal/noise-
unresolved/informative counts. Accuracy gives prediction ties half credit.
Cluster-bootstrap pair accuracy using endpoint episode multiplicities,
excluding self-pairs; the informative set stays fixed. These are approximate
noise screens, not simultaneous confidence guarantees. Require at least10
informative pairs spanning8 episodes for an ordering-support claim in a cell.
Report underpowered cells without additional sampling or tolerance changes.

The primary contrast is untouched-final early-query RMSE. Evidence for a
sampling bottleneck requires improvement in BOTH seeds, upper paired95% RMSE
change<0, reduced absolute bias, and corresponding train-context improvements;
also report any later-phase harm (paired RMSE upper bound>.01). Improved RMSE
alone does not demonstrate useful ordering. Good seen-query fit with weak
fresh-episode transfer suggests generalization; poor seen-query fit suggests
finite-data/critic-training fit limitations. Do not uniquely attribute the
cause to optimization, architecture or label noise without evidence.

## Budget and stopping

Maximum650,000 model outputs, charged before calls, plus exactly30,000 native
collection steps; fixed288 training paths,36 train roots,36 final roots,
64 evaluation repeats and6000 total optimizer steps. Previous diagnostic
targets are reused with zero model calls. Component tests have their own
small fixed budget. Save protocol/source hashes before outcomes; protect
attempts from overwrite. Missing inputs or insufficient final coverage stop
the run. No early stopping, oracle-based selection, extra fitting, budget
expansion, ETT/actor updates, new experiment family or automatic push.
