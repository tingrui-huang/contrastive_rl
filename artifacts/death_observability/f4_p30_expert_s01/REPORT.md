# PointMaze F4 death-observability diagnostic

## Decision

The four-frame learner state contains substantially more information about
whether the agent is already dead than current XY alone. It does not reveal
death reliably in the first observation after fatal contact, because the
consequence has not yet produced a repeated-position history. Separability
improves over the next two observations and is strong once the four-frame stack
has fully rolled over.

The frozen diagonal ETT model uses most of this information. Its residual
post-death motion is concentrated in observations that the F4 probe also finds
ambiguous, although a smaller residual remains on readily distinguished dead
states. The smallest justified next step is to keep the state and diagonal
checkpoint frozen and proceed to a bounded off-diagonal experiment with
explicit uncertainty and death-age diagnostics. There is no evidence here that
justifies adding a death bit or changing the state representation.

All probe, nearest-neighbor, and frozen-model sampling results in this report
were reproduced locally from the p=0.30 dataset and local ETT checkpoint. The
earlier diagonal report is used only to identify the frozen checkpoint; its
uploaded aggregate metrics are not copied into this diagnostic.

## Exact timing and data audit

The dataset is `swamp_windy_f4_merged_s0.npz`, SHA-256
`83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c`,
restricted to the established expert-positive source episodes `[1200, 6000)`.
The inherited episode split has 4,321 training episodes and 479 validation
episodes.

The collector writes `s_t`, the pre-action swamp mask, and `a_t`, then calls
`env.step(a_t)`. The environment moves first, checks whether the resulting cell
is active, marks the agent dead, and returns `s_{t+1}`. The F4 wrapper pushes
the resulting XY after this physics step. Therefore:

- The fatal-entry transition is alive before `a_t`; it dies during the
  transition and its target moves.
- The first positive label for "dead when the state is observed" is
  `s_{t+1}`, called death age 0 here.
- Ages 0, 1, and 2 retain progressively less pre-contact history. The stack is
  fully equal at age 3 in every observed death episode.
- The action at age 0 is already ignored by the environment, even though its
  input stack still looks like a moving history.

The timing is reconstructed from the per-episode death flag, the first
pre-action swamp mask whose transition lands in the corresponding active cell,
and the observed next position. It exactly matches the existing ETT audit
labels. Stationarity is not used to define death.

Training contains 20,233 dead-state rows from 468 episodes, for prevalence
`9.365%`; validation contains 1,866 rows from 43 episodes, for prevalence
`7.791%`. Each split has one age-0, age-1, and age-2 row per dead episode. The
remaining 18,829 training and 1,737 validation rows are age 3 or later. Every
post-death transition has strictly zero XY motion, while all 468/43 fatal-entry
transitions move.

An alive agent can still look stationary without being dead. Reset seeds all
four frames with the same position. A zero action receives continuous action
noise and usually moves slightly; wall collision rejects each axis separately
and can suppress some or all motion. In the observed data:

- All 4,321/479 alive equal stacks occur at reset; none occur later.
- There are 4,601/499 alive zero-action waiting rows, but no alive transition
  has exactly zero XY displacement.
- There are 8,618/960 alive transitions with displacement norm at most 0.05.
  Of the validation rows, 77 have nonzero action norm above 0.05 and are
  collision-compatible, though the dataset does not provide a collision label.

This empirical exact association between death and zero next displacement does
not make stationarity a valid general definition of death.

## Diagnostic probes

The three probes use current XY, the full F4 state, or F4 plus the recorded
current action. Each also receives the same commanded-goal block. The goal is
constant, is normalized to zero, and cannot affect the comparison. Every probe
uses the same two 64-unit SiLU hidden layers, unweighted binary cross-entropy,
4,000 updates, train-only normalization, and seeds 0 and 1. Hidden swamp bits,
death labels, source labels, rewards, future frames, and outcomes are excluded
from the inputs.

All reported classifications use the fixed probability threshold 0.5. The
threshold was not selected from validation labels. Checkpoints are selected by
validation BCE, so the reported split remains checkpoint-selection validation,
not an independent test set.

For seed 0:

- XY obtains ROC-AUC `0.97360`, average precision `0.65360`, Brier `0.03456`,
  dead precision `0.67682`, and dead recall `0.76206`. Its confusion matrix is
  TN 21,405, FP 679, FN 444, TP 1,422.
- F4 obtains ROC-AUC `0.99826`, average precision `0.98226`, Brier `0.00668`,
  dead precision `0.95016`, and dead recall `0.93998`. Its confusion matrix is
  TN 21,992, FP 92, FN 112, TP 1,754.
- F4 plus action obtains ROC-AUC `0.99848`, average precision `0.98529`, Brier
  `0.00630`, dead precision `0.95541`, and dead recall `0.94159`. Its confusion
  matrix is TN 22,002, FP 82, FN 109, TP 1,757.

Seed 1 reproduces the result: F4 ROC-AUC/AP are `0.99823/0.98175`, and F4 plus
action gives `0.99856/0.98534`. The recorded action provides only a small
increment. It can carry observational information about hidden conditions
because the teacher reacts to the current swamp mask; this improvement says
nothing about arbitrary intervened actions.

Complete-episode bootstrap for seed 0 estimates the F4-minus-XY ROC-AUC change
as `+0.02468`, 95% interval `[+0.01844, +0.03092]`, and the average-precision
change as `+0.32586`, interval `[+0.24136, +0.41245]`. The Brier change is
`-0.02789`, interval `[-0.03334, -0.02312]`. Within the swamp corridor, where
XY cannot win merely by locating the agent outside the hazard, the ROC-AUC
change is `+0.27924`, interval `[+0.22051, +0.34199]`.

The gain follows the verified frame timing:

- At age 0, F4 mean dead probability is `0.1318` and recall is `4.65%`.
- At age 1, the mean is `0.4754` and recall is `44.19%`.
- At age 2, the mean is `0.8369` and recall is `90.70%`.
- At age 3 or later, the mean is `0.9425` and recall is `97.52%`.

F4 distinguishes most observed waiting and near-stationary alive states. It
marks 16 of 499 zero-action waiting rows dead, for specificity `96.79%`, versus
48 false positives for XY. It also marks 16 of 960 alive near-stationary rows
dead. All 77 nonzero-action, near-stationary, collision-compatible rows are
classified alive. F4 plus action slightly improves aggregate metrics but has 23
false positives on the waiting subset, so action is not a uniformly helpful
death cue.

The safe route contains only 187 transitions from 25 validation episodes and
no dead positives. F4 produces no false positives there, but this sparse,
one-class region cannot establish death discrimination performance.

## Exact and approximate observational ambiguity

There are no bit-identical float32 F4 states with different current-death
labels, either within or across the established split. This rules out exact
aliasing in the finite dataset; it does not prove that the representation is
fundamentally identifiable.

Nearest-neighbor distances use full histories, train-only coordinate
normalization, validation queries, and training neighbors. The kth neighbors
come from distinct training episodes, so repeated adjacent dead rows cannot
inflate the evidence.

For a dead validation state, the median nearest alive-training distance is
`0.0080` using XY and `0.3883` using F4. Every dead state has an XY opposite
neighbor within radius 0.05, while no F4 opposite neighbor is within 0.10.
At F4 radii 0.25, 0.50, and 1.00, the fractions are `20.95%`, `62.49%`, and
`97.96%`. The distinct-episode F4 kth-neighbor medians are `0.3883`, `0.5958`,
and `0.7178` for k=1, 5, and 20.

Approximate ambiguity is greatest at age 0: its median nearest opposite-label
F4 distance is `0.2999`, and `32.56%` are within radius 0.25. The corresponding
fractions are `11.63%` at ages 1 and 2. Established dead rows still have
`21.13%` within radius 0.25 because slow or waiting alive histories exist near
some frozen swamp positions. The classifier results show that most of these
nearby cases retain useful differences.

Concrete cross-split pairs include:

- Dead age-0 episode 3821, row 7 has frames
  `[(5.4365,3.6167), (4.4365,3.3748), (3.4526,3.3845),
  (3.4468,3.3910)]` and action `(1.0000,-0.2987)`. Its nearest alive state is
  episode 1905, row 8 with frames `[(5.2833,3.6188), (4.2933,3.3754),
  (3.4144,3.3838), (3.4307,3.3773)]`, action `(0.9986,-0.2913)`, and F4
  distance `0.1101`.
- Established-dead episode 1792, age 23, has four copies of
  `(3.9274,3.6043)` and action `(0.6736,-0.1256)`. Its nearest alive state is
  episode 2592, row 9 with four slowly varying positions around
  `(4.04,3.60)`, action `(1.0000,-0.0408)`, and distance `0.1135`.
- Alive episode 4196, row 6 has frames `[(4.0794,3.3533),
  (4.0999,3.3561), (4.1004,3.3367), (3.1004,3.3809)]` and zero action. Its
  nearest dead state is episode 5629 at age 2, with three repeated newest
  positions `(3.9855,3.3419)`, action `(0.6897,-0.0054)`, and distance
  `0.1208`.

These are approximate pairs in a finite sample, not proof of exact aliasing.
The full six-pair record, episode identities, region composition, k and radius
sensitivity are stored in `metrics.json`.

## Frozen diagonal ETT diagnosis

The seed-0 K3 diagonal model is evaluated afresh on the same 23,950 validation
transitions with `a = a_obs` and 128 samples per transition. Its analytic zero
atom treated as a death score has ROC-AUC `0.99789`, average precision
`0.96508`, Brier `0.00674`, precision `0.94703`, and recall `0.93891`. The F4
plus action probe, which receives comparable observed information, has slightly
better ROC-AUC `0.99848`, average precision `0.98529`, and Brier `0.00630`.
The ETT therefore uses most, but not all, of the available signal.

Sampling plus geometry projection lowers average precision to `0.94516` and
raises Brier to `0.00807`. Analytic and sampled stationary probabilities by
time since death are:

- Age 0: `0.1517` analytic and `0.1799` sampled; predicted motion probability
  `82.01%`, with mean motion magnitude `0.6493` maze units.
- Age 1: `0.4030` and `0.4224`; predicted motion `57.76%`, magnitude `0.4493`.
- Age 2: `0.7953` and `0.8052`; predicted motion `19.48%`, magnitude `0.1490`.
- Age 3 or later: `0.9272` and `0.9289`; predicted motion `7.11%`, magnitude
  `0.0473`.

Residual motion strongly tracks F4-probe ambiguity. For true dead rows with
probe probability below 0.1, the ETT predicts motion `91.73%` of the time. The
rates are `64.19%` for probe probability 0.1-0.5, `14.65%` for 0.5-0.9, and
`4.72%` for the 1,473 dead rows at or above 0.9. The high-confidence residual
and the lower ETT average precision indicate modest fitting underuse, while the
much larger age-0/1 errors reflect observational ambiguity before repeated
frames accumulate.

On the 499 alive zero-action waiting rows, the ETT analytic zero atom averages
`0.0804` and sampled stationary probability is `0.0816`, although every
observed next displacement is nonzero because of action noise. The predicted
motion magnitude remains small at `0.0147`. On the 77 collision-compatible
rows, the probe classifies all states alive; their full ETT metrics remain in
`metrics.json`.

Fatal-entry rows are correctly labeled alive before action, and every observed
target moves. The ETT assigns them only `0.1034` analytic stationary mass and
predicts motion `86.56%` after projection. A low stationary probability on this
stratum is appropriate; it must not be counted as a missed dead-state
prediction.

## Implication for the next experiment

Failure should be defined with the verified temporal event, not by an equal
stack, a zero action, or one near-stationary transition. For evaluation, keep
fatal-entry, dead ages 0-2, and established death as separate strata. The first
state after death is intrinsically uncertain from the observed history in many
cases; a hard classifier output should not be added as a state feature or used
to force the transition stationary.

The next off-diagonal experiment can retain the current F4 state and frozen
diagonal checkpoint. It should carry explicit uncertainty in early post-death
contexts and include a diagonal-preservation gate, especially for established
dead rows where observable evidence is strong. Longer history or an explicit
death coordinate is not justified by this bounded diagnostic. This result does
not establish F4 Markov sufficiency or off-diagonal causal identification.

## Reproduction

```bash
python scripts/diagnose_f4_death_observability.py \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --ett-run-dir artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s0 \
  --out-dir artifacts/death_observability/f4_p30_expert_s01 \
  --seeds 0,1 --hidden-sizes 64,64 --steps 4000 --batch-size 1024 \
  --learning-rate 3e-4 --eval-every 250 --ett-samples 128 \
  --bootstrap-replicates 1000 --seed 1701
```

The six probe checkpoints stay local and are ignored by Git. `config.json`,
`metrics.json`, this report, and all plots are intended to be tracked.
