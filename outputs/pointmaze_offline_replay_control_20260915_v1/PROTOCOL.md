# Sealed strictly offline PointMaze replay-control protocol

Sealed before optimizer updates on 2026-09-15 at repository commit
`5acf76669f1378aaa88809c390fe3f3b625aff71`. The final iterate is the only
primary checkpoint; there is no sweep, early stopping, or checkpoint selection.

## Offline boundary and eligible inputs

Only `obs` and `act` are loaded from the frozen 6,600-episode PointMaze F4
dataset. No environment, simulator, learned transition model, true-dynamics
helper, hidden/audit field, reward, death, return, success label, or newly
collected diagnostic trajectory is read. The eligible initialization is the
original alpha-0/seed-0 full learner checkpoint at step 150,000, whose adjacent
provenance says it was trained only on that fixed dataset at code commit
`4f3da3f1e94cbad647bc56b0f24e4171776090ee`. Both arms restore byte-identical
actor, critic, actor-Adam, and critic-Adam states. C1 and every checkpoint or
optimizer state derived from oracle-generated or new native replay are excluded.

The established split is preserved exactly: seed-0 permutation, first 660
episodes held out, remaining 5,940 used for continuation training. The initial
checkpoint was trained on all 6,600 episodes, so held-out means held out only
from this continuation experiment, not independently unseen by initialization.

## Fixed samplers

Every row uses its recorded state and action and a strictly later F4 achieved
state from the same episode, sampled with probability proportional to
`0.95 ** (future_time - anchor_time)`. Valid anchors are times 0--49 and future
times are 1--50. Terminal/absorbing tails remain in the dataset and are not
filtered.

- F: 256 rows from the original episode-uniform, anchor-uniform replay law.
- R: the first 128 rows of the same ordinary draw, plus 64 observable downward
  anchors and 64 observable rightward anchors, followed by a sealed shuffle.
- Observable fork: `1 <= x < 2` and `3 <= y < 4`.
- Downward stratum: `a_y <= -0.5` and `abs(a_y) >= abs(a_x)`.
- Rightward stratum: `a_x >= 0.5` and `abs(a_x) > abs(a_y)`.

The strata are mutually exclusive and use current observable XY plus recorded
action coordinates only. Sampling is with replacement. No outcome, future-goal
location, task-goal proximity, behavior-policy label, or exact continuous key
is used for selection. The all-pairs 256x256 sigmoid-NCE loss is unchanged.

## Budgets, seeds, and fixed evaluations

- Critic: 400 updates per arm, actor frozen; Adam and learning rate restored.
- Actor: 1,000 updates per arm, final critic frozen; both actors restart from
  the identical original actor and actor-Adam state.
- Actor batches: identical ordinary-replay indices, future goals, and policy
  Gaussian innovations in F and R; original BC coefficient 0.5, random-goal
  fraction 0.5, and entropy coefficient 0.0.
- Seeds: ordinary critic 530915000; focused critic 530915001; focused shuffle
  530915002; actor replay 530915100; actor keys 530915101; evaluations
  530915200--530915205; bootstrap 530915300.
- Critic evaluation: 32 common ordinary held-out NCE batches, 32 common balanced
  fork held-out NCE batches, all 656 held-out episodes with an observable fork
  state (first such state), endpoint actions down `[0,-1]` and right `[1,0]`,
  and four fixed continuous actions around each endpoint.
- Actor evaluation: common Gaussian innovations at those fork states under the
  canonical task goal and a sealed same-episode future goal; 1,024 held-out
  non-fork ordinary-goal rows; four common fixed actor-gradient batches.
- Uncertainty: 2,000 fixed-seed episode bootstrap resamples. Fork evaluation has
  one row per episode, so the cluster unit is the episode.

## Numerical and integrity rules

Abort before or during training on an input-hash mismatch, split overlap,
non-strict future, cross-episode relabel, stratum mismatch, nonfinite loss,
gradient, update, or parameter, changed frozen tree, unequal starting trees,
unequal actor batches/keys, or an unexpected batch allocation. Training is not
aborted for a scientifically negative ranking result. Both final actor stages
must run. Historical artifacts are checked by hash after completion.

Endpoint preference is a probe diagnostic, not ground-truth intervention
value. Local action gradients and actor distributions are reported separately.
No replayed action fraction is called route entry or native success.
