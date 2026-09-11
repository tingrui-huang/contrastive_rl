# PointMaze diagonal ETT transition baseline

## Verdict

The zero-inflated mixture transition model fits the expert-positive diagonal
transitions well enough to serve as the observational reference for the next
ETT experiment. Both stochastic seeds are stable, beat persistence by a large
margin, and beat the deterministic predictor on conditional energy score. The
checkpoint deliberately rejects calls with `action != observational_action`.
It is therefore an observational diagonal baseline, not a validated
counterfactual transition model.

These results use the same validation split used for checkpoint selection.
They are validation diagnostics, not an independent test-set estimate.

## Verified data contract

- Environment: `point_two_route_swamp_windy_f4_v0`.
- Dataset: `swamp_windy_f4_merged_s0.npz`, SHA-256
  `83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c`.
- The established expert-positive population is source episode IDs
  `[1200, 6000)`: 4,800 episodes and 240,000 aligned transitions. The 1,200
  uniform-coverage episodes and 600 separately collected blind-demonstrator
  episodes are excluded. Selection does not use outcomes or hidden state.
- This source is the project's teacher-generated population, whose metadata
  also records a `random_frac=0.2`; it should not be interpreted as uniformly
  optimal behavior.
- The split is inherited from the full-source episode split with seed 0:
  4,321 training episodes (216,050 transitions) and 479 validation episodes
  (23,950 transitions), with no episode overlap or byte-identical trajectory
  duplicates crossing the split.
- Each learner-visible state contains four newest-first XY frames (8 values).
  Actions have 2 values in `[-1, 1]`. The pre-action commanded goal has 8
  values and is constant in this dataset. Goal conditioning is retained because
  it is part of the project's learner-visible observation contract.
- Terminal dummy actions and padding are excluded. Every observed transition
  satisfies `next_state[2:8] == state[0:6]` exactly, so only the newest XY is
  modeled stochastically.
- Expert failures and post-death transitions are retained. The training split
  contains 468 fatal entries and 20,233 post-death action-ignored rows; the
  validation split contains 43 fatal entries and 1,866 post-death rows. All
  audited post-death rows have exactly stationary XY.

The training input is the 20-dimensional concatenation
`[state_f4, action, observational_action, commanded_goal_f4]`. On every fitted
row, both action arguments equal the recorded teacher action. Hidden swamp
bits, death flags, teacher/source labels, rewards, outcomes, future states, and
hindsight goals are not inputs. Death and swamp labels are loaded only by the
separate evaluator for stratified diagnostics.

## Model and environment handling

The stochastic baseline is a two-layer 128-unit MLP with a zero-displacement
point mass and three diagonal-Gaussian components for nonzero newest-XY
displacement. It reconstructs the other three frames by exact shifting. Context
normalization is fit only on training trajectories; moving-displacement
normalization is fit only on nonstationary training rows.

The environment clips actions to `[-1, 1]`, applies ten 0.1 substeps with small
action noise, and rejects collision movement per axis. Sampling caps each
predicted coordinate displacement at one maze unit, applies global
`[0, 9] x [0, 5]` bounds, and sends blocked endpoints back to the current XY.
This guarantees valid final endpoints but approximates the environment's exact
axiswise substep collision path. Reported likelihood is for the unprojected
displacement distribution.

The likelihood is a mixed measure: probability mass at exactly zero XY
displacement and a density with respect to maze-unit area for nonzero XY
displacement. NLL is reported in nats per two-dimensional transition. Negative
NLL values are valid for a continuous density.

## Validation results

The real-data GPU smoke test passed every alignment, finite-value, shape,
reproducibility, frame-shift, bounds, and off-diagonal-rejection check. Its
30-update loss decreased from `1.95649` to `0.89214`.

The best seed-0 checkpoint is step 18,000 with NLL `-5.77057`; the best seed-1
checkpoint is step 19,500 with NLL `-5.74830`. All recorded training and
validation losses are finite. The deterministic baseline's best standardized
displacement MSE is `0.026827` at step 19,000.

- Zero-inflated mixture, seed 0: mean next-position error `0.02405` maze units
  and conditional energy score `0.01391` maze units.
- Zero-inflated mixture, seed 1: mean next-position error `0.02352` and energy
  score `0.01394`.
- Deterministic predictor, seed 0: mean next-position error and deterministic
  energy score `0.02201`.
- Persistence: mean next-position error and deterministic energy score
  `0.36054`.

The deterministic predictor has `0.00204` lower mean position error than the
seed-0 stochastic model, with a 95% complete-episode bootstrap interval of
`[0.00119, 0.00302]`. The stochastic model has `0.00810` lower energy score,
with interval `[0.00702, 0.00929]`. This is the expected distinction between a
sharp conditional mean and a fitted conditional distribution. Against
persistence, the seed-0 stochastic energy-score improvement is `0.34663`, with
interval `[0.33812, 0.35466]`. Both stochastic seeds have indistinguishable
energy scores: seed 0 minus seed 1 is `-0.000030`, with interval
`[-0.000345, 0.000221]`.

The seed-0 energy scores by region are `0.00840` at start, `0.01517` before the
swamp, `0.05556` while moving in the swamp, `0.00985` while stationary in the
swamp, `0.02211` on the safe route, and `0.00971` after the swamp. They are
lower than the deterministic predictor in every listed region. Persistence is
exact on stationary swamp rows, where its energy score is zero; the stochastic
model does not beat that specialized behavior.

The weakest stratum is hidden-outcome-sensitive behavior around death. Seed 0
has energy score `0.07724` on 43 fatal-entry transitions and `0.03100` on 1,866
post-death transitions. It assigns an average sampled exact-stationary
probability of `0.1370` at fatal entry, where the observed outcome is moving,
and `0.8965` after death, where the observed outcome is always stationary.
This is consistent with information about death being absent from the model
inputs and with the small number of fatal episodes.

Overall stationary calibration is strong: observed exact-stationary frequency
is `0.07791`, mean analytic atom probability is `0.07789`, sampled probability
is `0.07971`, and 10-bin ECE is `0.00322`. At an L2 tolerance of 0.05 maze
units, observed near-zero frequency is `0.11800`, predicted frequency is
`0.11856`, and ECE is `0.00269`.

The seed-0 mean conditional XY standard deviations are `[0.03386, 0.01480]`;
only `0.87%` of contexts have sample standard-deviation norm below `1e-3`.
There is no broad collapse to a deterministic output. Representative absorbed
states can correctly collapse to the zero atom, while another stationary-swamp
example shows a rare moving branch and a wide X tail. The two seeds have nearly
identical aggregate diversity and energy score, although individual sparse
safe-route and death contexts remain noisy.

Every final sample is finite, lies inside an open maze cell, and has exact F4
frame shifting. Before projection, seed 0 places `2.386%` of samples beyond the
one-unit per-coordinate displacement cap, `0.0376%` outside global bounds, and
`0.179%` at a blocked endpoint. This invalid pre-projection mass and the
approximate collision projection remain limitations for future rollouts.

## Reproduction

Run the alignment, gradient, and API smoke test:

```bash
python scripts/smoke_ett_diagonal_transition.py \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --out-dir artifacts/ett_diagonal/gpu_smoke
```

Train the selected stochastic configuration; change `--seed` and the output
directory to reproduce seed 1:

```bash
python -m ett.train_diagonal_transition \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --out-dir artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s0 \
  --model-type mixture --num-components 3 --hidden-sizes 128,128 \
  --steps 20000 --batch-size 512 --learning-rate 3e-4 \
  --eval-every 500 --seed 0 --split-seed 0
```

Train the deterministic comparison:

```bash
python -m ett.train_diagonal_transition \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --out-dir artifacts/ett_diagonal/f4_p30_expert_deterministic_s0 \
  --model-type deterministic --hidden-sizes 128,128 \
  --steps 20000 --batch-size 512 --learning-rate 3e-4 \
  --eval-every 500 --seed 0 --split-seed 0
```

Evaluate both stochastic seeds and the deterministic comparison on the fixed
validation trajectories:

```bash
python -m ett.eval_diagonal_transition \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --model zimdn_k3_s0=artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s0 \
  --model zimdn_k3_s1=artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s1 \
  --model deterministic_s0=artifacts/ett_diagonal/f4_p30_expert_deterministic_s0 \
  --primary-label zimdn_k3_s0 \
  --out-dir artifacts/ett_diagonal/f4_p30_expert_evaluation_s01 \
  --samples-per-context 64 --representatives-per-region 2 \
  --bootstrap-replicates 2000 --batch-size 4096 \
  --near-zero-tolerance 0.05 --seed 1701
```

## Sampling API

```python
import jax
import numpy as np

from ett.diagonal_transition import load_diagonal_transition

model = load_diagonal_transition(
    'artifacts/ett_diagonal/f4_p30_expert_zimdn_k3_s0')
with np.load('datasets/swamp_windy_f4_merged_s0.npz') as data:
  observation = data['obs'][1200, 0]
  action = data['act'][1200, 0]

state, commanded_goal = observation[:8], observation[8:16]
one_next_state = model.sample(
    state, action, action, jax.random.PRNGKey(0), goal=commanded_goal)
sixteen_next_states = model.sample(
    state, action, action, jax.random.PRNGKey(1), num_samples=16,
    goal=commanded_goal)
assert one_next_state.shape == (8,)
assert sixteen_next_states.shape == (16, 8)
```

The model is ready to be frozen as the diagonal observational reference and to
support implementation of a separately identified off-diagonal experiment. It
does not by itself justify off-diagonal prediction, causal recovery, Markov
sufficiency of four frames, or multi-step counterfactual rollouts.
