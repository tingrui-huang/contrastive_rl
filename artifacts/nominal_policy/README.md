# PointMaze nominal observational policy baseline

## Dataset interpretation

This run uses `datasets/swamp_windy_f4_merged_s0.npz`, SHA-256
`83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c`,
for `point_two_route_swamp_windy_f4_v0` with per-cell swamp activation
probability 0.30. It contains 6,600 fixed-length episodes and 330,000 usable
transitions. The final action row in each 51-row episode is a dummy and is
excluded.

The data are mixed observational behavior, not pure expert demonstrations:
1,200 episodes use uniform random behavior; 257 use a forced-safe teacher;
1,470 use an immediate-shortcut reactive teacher; 3,073 use a waiting
reactive teacher; and 600 use a blind shortcut bad demonstrator. The audit-only
fields that identify these modes and the hidden swamp state are never loaded by
the policy trainer.

Each learner state has 8 raw maze-unit coordinates: four newest-first XY
frames. The commanded goal has 8 coordinates: the goal XY repeated four times.
The action is two-dimensional and clipped componentwise to `[-1, 1]`. The
policy conditions on the full learner-visible pre-action `(state, commanded
goal)` vector, matching the project's existing propensity-flow convention.
The goal is constant in this dataset, so retaining it preserves the interface
but this run provides no evidence of generalization to unseen goals.

## Model and split

The main model is a 5-component conditional mixture of diagonal Gaussians with
two 128-unit SiLU hidden layers. Sampling first draws from the latent mixture
and then applies the environment's componentwise clip. Training minimizes the
corresponding censored negative log likelihood: interior actions use Gaussian
density, while exact `-1` and `+1` values use Gaussian tail mass. Reported NLL
is therefore in nats per two-dimensional action vector under the mixed measure
on the environment action box: Lebesgue measure in the interior plus point
masses at the boundaries.

The seed-0 split holds out 660 complete trajectories (33,000 transitions) and
trains on 5,940 trajectories (297,000 transitions), with no episode overlap.
Context means and standard deviations are fitted on training trajectories only.
All configuration, split identifiers, and normalization values are in each
run's `config.json`; `best.pkl` is selected by validation NLL.

## Results

Both models ran for 20,000 updates with finite training and validation losses.
The K=5 MDN's best step is 20,000, held-out NLL is -0.199629 nats/action
vector, natural-rollout success is 0.15, and death rate is 0.70. The K=1
censored Gaussian's best step is 16,500, held-out NLL is -0.020800, rollout
success is 0.19, and death rate is 0.76.

The MDN improves held-out NLL by 0.178829 nats/action vector. Across six
learner-visible held-out states, its sampled per-coordinate standard deviations
range from about 0.15 to 0.55, so it has not collapsed to a deterministic
point. Effective mixture counts range from 1.00 to 1.99: some contexts are
locally unimodal and several mixture components receive negligible mass. Full
quantiles, weights, component locations/scales, and sample boundary rates are
in `f4_p30_evaluation_s0.json`.

The 100-episode rollouts use sampled actions and define success as minimum XY
distance to the goal below 0.5. They are supplementary checks that the sampler
can drive the environment. The mixed behavior model does not receive the
teacher-only swamp bits, and matching its observed conditional actions need not
recover teacher returns.

## Reproduce

```bash
python -m scripts.smoke_nominal_policy

python -m propensity.train_nominal_policy \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --out-dir artifacts/nominal_policy/f4_p30_mdn_k5_s0 \
  --num-components 5 --hidden-sizes 128,128 \
  --steps 20000 --batch-size 512 --seed 0 --split-seed 0

python -m propensity.train_nominal_policy \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --out-dir artifacts/nominal_policy/f4_p30_unimodal_k1_s0 \
  --num-components 1 --hidden-sizes 128,128 \
  --steps 20000 --batch-size 512 --seed 0 --split-seed 0

python -m propensity.eval_nominal_policy \
  --dataset datasets/swamp_windy_f4_merged_s0.npz \
  --model-dir artifacts/nominal_policy/f4_p30_mdn_k5_s0 \
  --unimodal-dir artifacts/nominal_policy/f4_p30_unimodal_k1_s0 \
  --samples-per-state 1024 --rollout-episodes 100 \
  --out artifacts/nominal_policy/f4_p30_evaluation_s0.json
```

The future ETT-facing sampling interface accepts a complete observation or
separate state and commanded-goal tensors and preserves arbitrary leading batch
dimensions:

```python
import jax
from propensity.nominal_policy import load_nominal_policy

policy = load_nominal_policy(
    'artifacts/nominal_policy/f4_p30_mdn_k5_s0')
x_prime = policy.sample(observation, jax.random.PRNGKey(7))
x_prime_32 = policy.sample(observation, jax.random.PRNGKey(8), num_samples=32)
# Alternatively: policy.sample(state, key, num_samples=32, goal=commanded_goal)
```
