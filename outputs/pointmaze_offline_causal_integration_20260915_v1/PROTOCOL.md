# Sealed strictly offline causal-contrastive integration

This is one bounded PointMaze experiment restoring the sampling construction
`offline data -> nominal policy and pessimistically optimized ETT -> complete
virtual continuation -> discounted future F4 sample -> unchanged production
sigmoid-NCE -> unchanged actor objective`. It is not a new critic objective.

## Eligible inputs and exclusions

Only `obs` and `act` are loaded from the fixed 6,600-episode F4 dataset. The
production actor, critic and both Adam states come from the original
`alpha0_seed0/final.pkl` step-150000 checkpoint. The frozen K5 state-goal
nominal model and the `s0_L0p25_lambda0` stochastic atom/K3 diagonal checkpoint
are eligible because their fitting ancestry is the same original dataset; the
lambda-zero diagonal arm did not use the privileged failure-reference MMD term.

Historical `update_reference_s01_v1/final_kernels.npz` is excluded: its ETT
optimization ancestry includes native-collected early contexts. All native
diagnostic trajectories/checkpoints, C1, supervised off-diagonal heads, oracle
replay, oracle motion, hidden state, audit labels and analytical dynamics are
excluded. Historical reports may motivate the design but never enter training
or checkpoint selection.

The established episode split remains seed-0 permutation: 5,940 continuation
training episodes and 660 continuation-held-out episodes. The original learner
checkpoint saw all 6,600 episodes, so held-out is not independently unseen by
initialization.

## ETT refit and its limitation

Start at the eligible diagonal law with all 48 offsets zero. The first 16
coordinates are existing diagonal-head offsets; the last 32 are the existing
eight-gate, normalized action-response matrices. Keep the state-goal nominal
draw `x_prime`, queried action `a`, rectangle feasibility projection, samplewise
action-Lipschitz bound 1, full-F4 shift, diagonal emitted-XY energy score, and
the existing pessimistic region-occupancy surrogate.

Use 96 offline training roots selected without outcomes: 48 ordinary rows, 24
observable fork/down rows and 24 observable fork/right rows, with distinct
source episodes within each stratum when available. Every model path executes
the recorded action first, then the fixed original stochastic actor. Refit the
existing auxiliary 11-64-64-16 region critic from scratch on these model paths:
1,000 initial steps and 400 refresh steps before each of three ETT updates.
Each ETT update uses four antithetic directions, 32 visitation queries, four
first successors and four actor draws. Preserve sigma/rates/caps
`(.01,.1)/(.01,.5)/(.03,.1)`, lambda 1, and the initialization-plus-.02
diagonal guard. The fixed final iterate is used; no search or early stopping.

This auxiliary binary region-value estimator optimizes the ETT. It is not the
production full-F4 contrastive critic and is never used by the actor.

The eligible diagonal/action-response model has no explicit learned
irreversible-failure variable or defensible off-diagonal failure-entry
probability. A stationary atom is not declared death. Therefore failure
frequency and post-failure absorption are unsupported, rather than fabricated.
The generated paths still retain all raw atom indicators and valid F4 states so
stationary motion/resumption can be audited without calling either event death.

## Matched O/P positive-future comparison

Fork means `1 <= x < 2, 3 <= y < 4`. Down means `a_y <= -.5` and
`abs(a_y) >= abs(a_x)`; right means `a_x >= .5` and
`abs(a_x) > abs(a_y)`. These mutually exclusive strata use only visible XY and
recorded action.

Seal a training cache of 4,096 recorded anchors: 2,048 ordinary, 1,024 down and
1,024 right. Seal a continuation-held-out cache of 1,024 anchors: 512 ordinary,
256 down and 256 right. Sampling with replacement is allowed and reported.
Every production batch draws 128 ordinary, 64 down and 64 right cache entries.
O and P use the identical cache identity, recorded state/action and strictly
later relative future offset sampled proportional to `.95 ** offset`.

- O goal: the full-F4 state at that offset in the recorded source episode.
- P goal: the full-F4 state at that offset in the cached complete final-ETT
  continuation, whose first action is the same recorded action, subsequent
  actions come from the fixed original actor, and every step independently
  samples nominal `x_prime` and the ETT transition.

All P production positives come from P; there is no 90/10 observational mix.
Padding after the recorded remaining horizon is marked invalid and can never be
sampled. No path or row is filtered by outcome, reward, task-goal proximity,
stationarity, or desired direction. Standard within-batch negatives remain.

## Production learner and evaluation

O and P restore byte-identical eligible production critic/Adam states. Freeze
the actor and run 400 critic-only updates per arm using unchanged 256x256
sigmoid-NCE. Then freeze each final critic, restart both actors from the same
original actor/Adam state, and run 1,000 updates on identical ordinary offline
replay rows, future goals and Gaussian keys. Preserve BC=.5, random_goals=.5,
entropy=0, gamma=.95, both learning rates 3e-4 and Adam eps 1e-7.

Evaluate fixed final checkpoints on common held-out O-source and P-source NCE
batches, 656 first observable fork contexts, canonical down/right endpoints and
their fixed neighborhoods, local gradients at actor samples, actor continuous
distributions, four common gradient-component batches, and 1,024 non-fork
ordinary-goal rows. Use 2,000 fixed-seed episode bootstrap resamples. Report
goal marginals separately because raw logits across changed marginals are not
calibrated returns.

## Budgets and stop rules

One fixed seed. Maximum 480,000 newly emitted ETT successors, 2,200 auxiliary
critic updates, 3 ETT updates, 800 production critic updates, and 2,000 actor
updates. Zero environment/simulator calls and zero native steps. Abort only on
hash, lineage, split, first-action, future-time, padding, F4, feasibility,
nonfinite, frozen-tree or budget failures. Negative scientific results do not
stop either actor arm. Save fixed final iterates; no tuning, sweep, extension,
native evaluation, commit or push.
