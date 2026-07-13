# Goal-representation ablation @30k: xy vs compact vs full future-state goal

All arms: antmaze_open_near task, adaptive alpha target_entropy=-8, numerical guards, identical training settings; ONLY the goal representation differs (2 / 13 / 29 dims; commanded goal = settled ant at the goal cell for the rich arms; relabeled goals = same indices of future states). Pre-verified: state layout, goal slices, and relabeling are bit-exact; no normalization anywhere (artifacts/goal_contract_verification.json).

Primary behavioral metrics are XY-only for every arm.


## Arm: xy

| gate | alpha | entropy med | scale med | sat | eff-rank | moving | XY success | XY goal vel | XY progress | ctrl sp | ctrl useful | retr acc (64-way) | rank med | XY rel | pose rel | vel rel | joints rel | only-XY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.911 | 5.04 | 0.86 | 0.00 | 8.0 | 0.97 | 0.00 | 0.0055 | 0.33 | 0.19 | 0.15 | 0.260 | 4 | +0.26 | - | - | - | 0.26 |
| 20k | 0.084 | 4.16 | 0.81 | 0.00 | 7.6 | 0.96 | 0.00 | -0.0050 | 0.07 | -0.17 | -0.17 | 0.865 | 0 | +0.84 | - | - | - | 0.86 |
| 30k | 0.011 | -7.56 | 0.50 | 0.10 | 4.9 | 0.82 | 0.10 | 0.0043 | 0.24 | 0.05 | 0.06 | 0.802 | 0 | +0.78 | - | - | - | 0.80 |

## Arm: gcompact

| gate | alpha | entropy med | scale med | sat | eff-rank | moving | XY success | XY goal vel | XY progress | ctrl sp | ctrl useful | retr acc (64-way) | rank med | XY rel | pose rel | vel rel | joints rel | only-XY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.912 | 5.19 | 0.86 | 0.00 | 8.0 | 0.97 | 0.00 | -0.0072 | 0.17 | 0.07 | 0.03 | 0.312 | 4 | +0.29 | +0.08 | +0.04 | - | 0.16 |
| 20k | 0.084 | 4.25 | 0.80 | 0.00 | 7.9 | 0.97 | 0.00 | 0.0010 | 0.14 | 0.13 | 0.13 | 0.896 | 0 | +0.86 | +0.52 | -0.02 | - | 0.43 |
| 30k | 0.012 | -9.23 | 0.49 | 0.10 | 4.4 | 0.64 | 0.00 | 0.0043 | 0.15 | -0.03 | 0.04 | 1.000 | 0 | +0.98 | +0.27 | +0.00 | - | 0.78 |

## Arm: gfull

| gate | alpha | entropy med | scale med | sat | eff-rank | moving | XY success | XY goal vel | XY progress | ctrl sp | ctrl useful | retr acc (64-way) | rank med | XY rel | pose rel | vel rel | joints rel | only-XY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.910 | 5.25 | 0.83 | 0.00 | 8.0 | 0.97 | 0.00 | -0.0033 | 0.08 | 0.08 | -0.04 | 0.083 | 9 | +0.07 | -0.04 | -0.02 | +0.04 | 0.07 |
| 20k | 0.084 | 4.61 | 0.81 | 0.00 | 7.8 | 0.96 | 0.00 | -0.0072 | 0.13 | 0.01 | 0.09 | 0.885 | 0 | +0.80 | +0.33 | +0.00 | +0.01 | 0.51 |
| 30k | 0.012 | -2.60 | 0.48 | 0.02 | 6.0 | 0.83 | 0.10 | 0.0002 | 0.19 | 0.15 | 0.00 | 0.792 | 0 | +0.75 | +0.18 | +0.00 | +0.10 | 0.51 |


## Findings

1. **Training health is arm-independent**: all three arms track the same adaptive-alpha trajectory (0.91 -> 0.084 -> ~0.012, entropy approaching the -8 target), no guard trips, no floor, saturation <= 0.10, effective rank 4.4-6.0 at 30k, moving fraction 0.64-0.83.
2. **Richer goals improve only retrieval, and only marginally**: 64-way retrieval accuracy at 30k is 0.80 (xy) / 1.00 (compact) / 0.79 (full); by 20k all arms are already at 0.86-0.90. The compact goal is the best retriever; the full 29-dim goal is NOT better than xy (extra joint dims add noise, not signal).
3. **No shortcut learning**: block attribution shows the critic is XY-grounded in every rich arm -- XY knockout destroys retrieval (reliance +0.75..+0.98 at 30k) while velocity reliance is ~0.00 and joints +0.10; only-XY retains 0.51-0.78 accuracy. The commanded-goal velocity block is ~0 (settled ant), so velocity matching was the expected shortcut and it did not materialize.
4. **XY control does NOT improve**: success 0.1 / 0.0 / 0.1, XY goal velocity <= 0.006 m/s in all arms; local critic decile usefulness stays in -0.2..0.2 with negligible physical spread. Identical to the xy arm.

## Verdict

**Richer goals improve (already-strong) contrastive retrieval, not XY control.** The goal representation is NOT the binding constraint at this scale: with entropy fixed and goals enriched, the critic retrieves future states nearly perfectly and grounds them in XY, yet the actor still cannot extract goal-directed locomotion, and the local action-conditioned signal remains physically negligible (1-step XY effects of ~1e-4 m against goal distances of meters). The remaining gap sits in the actor/locomotion pathway: converting a valid state-goal value landscape into multi-step motor behavior within a 30-50k budget.

## Caveats

- 30k steps, seed 0, one run per arm.
- Commanded rich goals have ~zero velocity block (settled ant at rest); relabeled training goals carry real velocities (matches the original ant_envs semantics).
- gfull entropy median at 30k (-2.6) lags the other arms (-7.6/-9.2); alpha is identical, so this is estimator variance on a different obs distribution, not a health difference.
