# Preregistered early-context region-NCE PointMaze pilot

Continue from 0059c98. Preserve all prior experiments. Use the same shared
48-coordinate Kernel, fixed stochastic alpha=0 seed=0 actor, state-goal K5
nominal sampler, commanded goal, reward region, gamma=.95 and two losses.
No production code or checkpoint is changed. Checkpoints stay local.

## Selection and coverage, before model outcomes

Use the existing teacher-source episodes [1200,6000), with the original seed-0
660-episode holdout. Actor training overlaps both partitions; this is not a
fresh native rollout study. Candidate times are exactly 1,3,6,10. An eligible
root is outside the existing radius-2 goal region, has never been inside it
in its observed prefix, and newest-frame displacement exceeds .05 maze units.
Repeated reset frames at t=1 are legitimate recorded F4, not fabricated history.

Visible position groups (half-open upper limits):
- approach: 1<=x<3, 3<=y<4;
- transit: 3<=x<6, 3<=y<4;
- bypass: 1<=x<6, 1<=y<3.

In each partition process groups in order bypass, transit, approach. Randomly
permute source episodes using root seeds 96000000/1, once per group. Select
the first 12 distinct eligible episodes, choosing each episode's earliest
eligible candidate time for that group. Do not reuse an episode across groups.
No replacement, tolerance change or further collection. The resulting 36
roots per partition are equally weighted, with one root per episode.
Selection reads observations only; the selected index archive is sealed
before any hidden label or future return audit. Fit/normalization rows remain
the same first 512 rows/partition as the previous pilot.

Observable gate: obtain exactly 12 roots/group/partition; assert goal/prefix,
action, F4 and episode-separation contracts. Separately reconstruct native
death timing using the verified collector audit. Before training require at
least 9/12 roots alive when observed in each group and at least 30/36 overall
in each partition. This aggregate pre-outcome coverage check can only STOP;
it cannot remove individual roots or resample. Audit arrays never enter the
trainer. Future native outcomes do not gate coverage and are read for final
reporting only. Hidden labels do not define any training target.

## Generator gate, before critic or ETT fitting

After coverage passes, use 16 stochastic full continuations/validation root,
for three fixed feasible kernels: zero initialization and two response probes
with all eight response matrices respectively +.35 I and -.35 I, diagonal
head unchanged. Probes test capacity only; none is selected as initialization.
All groups/roots/probes are retained. Each path ends at original time 50.
Use paired random streams (97000000 plus time-bucket offsets), not identical
trajectories after kernels diverge. The actor, nominal and geometry stay fixed.

Classify saved paths for diagnosis only:
- goal-reaching: any existing task reward before time 50;
- finite-horizon zero-return: no goal reward before time 50, never called death;
- recovered after observable setback: first goal at step >=5 and, before that
  goal, either three consecutive <=1e-6 XY steps, or goal distance rises by
  >=.25 from a prior running minimum. This is a realized recovery trajectory,
  not a claim that all other zero-return paths are irrecoverable.

Require at least two roots in EACH of approach and transit for which ONE of
the three probes has >=2 goal-reaching and >=2 zero-return paths among 16.
Require >=8 recovered-after-setback paths across the probes, and initial root
mean-occupancy standard deviation >=.025. Every probe must pass the existing
geometry/F4/action checks and diagonal invariance. This bounded sample gate
is not a certificate of full-family expressiveness or its absence. Failure
stops the experiment; explain any independently established structural limits.
The model still has no persistent death variable and the inherited rectangles
exclude some valid fork transitions. No new constraints or reward are added.

## Estimator and pre-optimization calibration

H_i=50-t_i, i.e. 49,47,44,40. Critic inputs remain F4 and executed action plus
remaining horizon, now scaled by 50 instead of 10 to cover the task range.
Reuse the 11-64-tanh-64-tanh-16 dot-product neural architecture, binary future
reward-region labels, uniform negative labels q=(.5,.5), alpha=0, B=32.
Q_h=(1-.95^h)*31*.5*exp(f_1), Q_0=0. No continuous-state q=1/N, clipping,
mean-logit surrogate, death classifier or hand-coded values. Float64, stable
softplus, Adam .003, minibatch256, one CPU thread, deterministic Torch.

For each variable-length path/time sample the correct truncated-geometric
future label using its actual remaining length. Train 1500 initial critic
steps on 36x8 complete training-root paths. Independently evaluate 16 repeats
at root, midpoint and one-step-remaining queries for every validation root.
At each query hold one sampled first action fixed across repeats and score
that action. Vary subsequent actor/nominal/transition randomness. No native
snapshot is used. The same generator supplies queries and all target returns.

Require RMSE<=.06 and |bias|<=.025 for each of the three query phases overall,
and RMSE<=.07 and |bias|<=.035 for every phase/context-group cell. Retain and
report range overshoot. Failure stops before ETT updates; no refit retry.

## Paired joint experiment, only if all gates pass

Two paired seeds 0/1; diagonal-only and joint; six updates each. Identical
zero ETT offsets and identical initial critic/Adam clones. Per round collect
36x8 complete model paths, warm-fit critic400 steps, then freeze critic and
visitation. Use four paired Gaussian parameter directions, head sigma .01,
response sigma .1, and the same old rates .01/.5 and block step caps .03/.1.
Select 128 path indices uniformly (one root has eight equally weighted paths),
then uniformly select a time within that path. Weight the surrogate by
H_i*.95^t. Draw four x_prime/successors per query and four next-actor actions
per successor. Minimize diagonal ES plus lambda times
.05*1[next state in goal]+.95*mean Q_(h-1), with lambda=1 joint/0 control.
Different h_i must not be flattened with a common-horizon visitation weight.

The old anisotropic antithetic Gaussian score estimator handles all hard
rewards, stochastic mixture choices and projections. Set the structurally
zero diagonal derivative of response coordinates to zero. This estimates a
Gaussian-smoothed parameter objective, not an exact unsmoothed gradient.
One shared kernel marginalizes x_prime through frozen b at every step.

Keep emitted-XY energy fitting and tolerance .02: same fixed256-row/16-draw
train guard, signed128-row/8-draw ES queries, no line search. The normalized
response and old convex projection retain samplewise Euclidean action L=1,
and a common-x_prime Wasserstein coupling, not toy TV or physical completeness.
All accepted iterates satisfy the training guard; independently check512
validation rows with32 draws. Pre-projection NLL remains a separate diagnostic.
Final iterates only; no evaluation-driven checkpoint or parameter selection.

## Final evaluation and stopping

Use independent random streams for all frozen initial/control/joint models.
Evaluate last training critics (possibly one-update stale) and separate fresh
1000-step MC-only final refits against the same saved independent targets.
No final refit is used in ETT updates. Report every phase/group calibration,
range overshoot, model recovery/zero-return/path diversity, and available
native logged-behavior outcomes separately. No new native simulator run.
State any disagreement without calling hidden-context evidence ETT truth.

Report per-group and equally group-weighted root return changes, with2000
paired episode bootstrap replicates (stratified by visible group for overall
comparisons). Average repeats before bootstrapping; no transition/repeat
pseudoreplication. Diagonal row comparisons cluster by episode. An effect
requires upper95% return-change CI<0, final calibration gates, all construction
checks, and diagonal ES upper95% increase<=.02 versus both control and initial.
Group-specific intervals are descriptive, not multiplicity-adjusted.
Retain final iterates even if an effect is not established; no further family.

Random namespaces: selection96000000; generator97000000; critic98000000;
initial collection99000000; training100000000+100000*seed+1000*iteration;
independent evaluation101000000; bootstrap102000000. Within-bucket offsets
are recorded in code. The reused action-component check keeps its old fixed
seed94000800; it is independent of this pilot's sampling streams. Fixed
maximum1,500,000 emitted outputs, charged before
calls including probes and final evaluation. Maximum72 roots,16 evaluation
repeats,24 total ETT updates; no cap expansion. Component tests have a separate
small fixed budget. Interrupted attempt markers preserve partial results.
If coverage, generator or estimator gate fails, report that stage and STOP.
Do not train the actor, alter the loss, optimize a probe, or push automatically.
