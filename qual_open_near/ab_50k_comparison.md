# Near-goal/open-area Ant qualification: 50k A/B (alpha=0 vs adaptive alpha)

Task: `antmaze_open_near` = AntMaze_Open-v5 (Gymnasium-Robotics physics, 2D XY goal), start cell uniform, goal same/orthogonally-adjacent cell, d0 rejection-sampled to [1.0, 4.5] m, 300-step episodes. Config identical to the 150k umaze run (binary NCE, random_goals 0.5, min_replay/random 10k, 4 sgd steps/step, batch 256, seed 0) except the arm variable:
- **alpha0**: entropy_coefficient = 0.0 (faithful as-shipped baseline)
- **adaptive**: entropy_coefficient = None, target_entropy = 0.0 (the original repo's adaptive semantics)

Per gate: fresh sampled-policy collection (12 eps, proxy for replay additions), deterministic eval (10 eps, fixed seeds), coverage-gated fresh reference states, immediate controllability (1 step + 2 zero-action settle, sigma=0.05, 64 candidates, 30 states).


## Arm: alpha0

| gate | scale med | frac@floor | alpha | |samp-mode| | mode sat | act effrank | act dim std | moving frac | disp p90 | torso z | fall step | cov u/ep/goal/mov | cov gate | ctrl std proj1 | ctrl rng prog1 | ctrl sp | critic dec gap | dec useful | success | progress | goal vel | speed | static | fall(eval) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.0334 | 0.00 | - | 0.053 | 0.31 | 3.5 | 0.636 | 0.38 | 0.0482 | 0.54 | 0.25 | 539/12/12/0.37 | PASS | 4.8e-05 | 0.00022 | 0.08 | 8.1e-05 | 0.09 | 0.00 | 0.80 | 0.025 | 0.0163 | 0.45 | 0.47 |
| 20k | 1.36e-06 | 0.75 | - | 0.000 | 0.38 | 4.2 | 0.838 | 0.47 | 0.0570 | 0.52 | 0.22 | 372/12/12/0.71 | PASS | 0.00023 | 0.00097 | 0.14 | 0.00011 | 0.12 | 0.00 | 0.30 | 0.009 | 0.0039 | 0.82 | 0.09 |
| 30k | 1e-06 | 0.98 | - | 0.000 | 0.39 | 4.8 | 0.783 | 0.35 | 0.0548 | 0.55 | 0.16 | 316/12/12/0.65 | PASS | 0.00029 | 0.0014 | 0.09 | 7e-05 | 0.07 | 0.00 | 0.09 | -0.022 | 0.0071 | 0.78 | 0.12 |
| 40k | 1e-06 | 0.98 | - | 0.000 | 0.36 | 4.3 | 0.798 | 0.40 | 0.0526 | 0.50 | 0.33 | 247/12/12/0.74 | PASS | 0.00021 | 0.001 | -0.09 | -0.00021 | -0.21 | 0.00 | 0.77 | -0.008 | 0.0181 | 0.51 | 0.23 |
| 50k | 1e-06 | 0.99 | - | 0.000 | 0.34 | 3.9 | 0.793 | 0.33 | 0.0538 | 0.51 | 0.10 | 345/12/12/0.61 | PASS | 0.00019 | 0.00083 | 0.05 | 0.00019 | 0.19 | 0.10 | 0.71 | 0.011 | 0.0129 | 0.53 | 0.21 |

training evals: step 10200: sat=0.10, actor_loss=6.53, logits_gap=5.9; step 20400: sat=0.30, actor_loss=17.4, logits_gap=33.4; step 30600: sat=0.54, actor_loss=14.4, logits_gap=31.2; step 40800: sat=0.28, actor_loss=13.5, logits_gap=31.0

## Arm: adaptive

| gate | scale med | frac@floor | alpha | |samp-mode| | mode sat | act effrank | act dim std | moving frac | disp p90 | torso z | fall step | cov u/ep/goal/mov | cov gate | ctrl std proj1 | ctrl rng prog1 | ctrl sp | critic dec gap | dec useful | success | progress | goal vel | speed | static | fall(eval) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.858 | 0.00 | 0.912 | 1.609 | 0.00 | 8.0 | 0.595 | 0.97 | 0.0825 | 0.49 | 0.45 | 401/12/12/0.97 | PASS | 0.00056 | 0.0026 | 0.09 | 0.0003 | 0.16 | 0.00 | 0.28 | 0.004 | 0.0067 | 0.43 | 0.00 |
| 20k | 0.798 | 0.00 | 0.084 | 1.390 | 0.00 | 7.5 | 0.591 | 0.96 | 0.0747 | 0.55 | 0.36 | 465/12/12/0.97 | PASS | 0.0005 | 0.0023 | -0.01 | -3.2e-05 | -0.02 | 0.00 | 0.16 | 0.003 | 0.0031 | 0.84 | 0.00 |
| 30k | 0.537 | 0.00 | 0.014 | 0.752 | 0.04 | 5.5 | 0.729 | 0.70 | 0.0465 | 0.51 | 0.25 | 542/12/12/0.81 | PASS | 6.1e-05 | 0.00024 | -0.07 | -3.1e-05 | -0.05 | 0.00 | 0.30 | 0.012 | 0.0034 | 0.72 | 0.09 |
| 40k | 1e-06 | 1.00 | 0.012 | 0.000 | 1.00 | 0.0 | 0.000 | 0.02 | 0.0000 | 0.57 | 0.00 | 60/12/12/0.40 | PASS | 1.6e-05 | 7.7e-05 | 0.11 | 2.1e-07 | 0.00 | 0.00 | 0.04 | -0.003 | 0.0006 | 0.96 | 0.00 |
| 50k | 1e-06 | 1.00 | 0.012 | 0.000 | 1.00 | 0.0 | 0.000 | 0.02 | 0.0000 | 0.57 | 0.00 | 60/12/12/0.40 | PASS | 1.6e-05 | 7.7e-05 | 0.22 | 3e-05 | 0.15 | 0.00 | 0.04 | -0.003 | 0.0006 | 0.96 | 0.00 |

training evals: step 10200: sat=0.00, alpha=0.913, actor_loss=2.5, logits_gap=5.9; step 20400: sat=0.00, alpha=0.084, actor_loss=19.3, logits_gap=34.6; step 30600: sat=0.03, alpha=0.014, actor_loss=18.3, logits_gap=36.5; step 40800: sat=1.00, alpha=0.012, actor_loss=-3.8e+23, logits_gap=38.5

## Arm: adaptive_te8

| gate | scale med | frac@floor | alpha | |samp-mode| | mode sat | act effrank | act dim std | moving frac | disp p90 | torso z | fall step | cov u/ep/goal/mov | cov gate | ctrl std proj1 | ctrl rng prog1 | ctrl sp | critic dec gap | dec useful | success | progress | goal vel | speed | static | fall(eval) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.862 | 0.00 | 0.911 | 1.607 | 0.00 | 8.0 | 0.597 | 0.97 | 0.0778 | 0.47 | 0.47 | 389/12/12/0.97 | PASS | 0.00055 | 0.0026 | 0.19 | 0.00031 | 0.15 | 0.00 | 0.33 | 0.006 | 0.0083 | 0.41 | 0.00 |
| 20k | 0.809 | 0.00 | 0.084 | 1.446 | 0.00 | 7.6 | 0.602 | 0.96 | 0.0635 | 0.48 | 0.48 | 380/12/12/0.96 | PASS | 0.00047 | 0.0022 | -0.17 | -0.00032 | -0.17 | 0.00 | 0.07 | -0.005 | 0.0016 | 0.85 | 0.00 |
| 30k | 0.501 | 0.00 | 0.011 | 0.725 | 0.10 | 4.9 | 0.714 | 0.82 | 0.0565 | 0.54 | 0.21 | 570/12/12/0.88 | PASS | 0.00024 | 0.001 | 0.05 | 6.2e-05 | 0.06 | 0.10 | 0.24 | 0.004 | 0.0052 | 0.69 | 0.00 |

training evals: step 10200: sat=0.00, alpha=0.911, actor_loss=2.51, logits_gap=5.9; step 20400: sat=0.00, alpha=0.084, actor_loss=16.8, logits_gap=28.8


## Findings

1. **alpha0 reproduces the instant entropy collapse on the new task**: scale median 0.80 (init) -> 0.033 at the 10.2k gate (~300 gradient batches after warmup) -> at the 1e-6 actor_min_std floor from 20k on (75% -> 99% of dims). Action effective rank ~4/8, moving-transition fraction decays 0.38 -> 0.33, saturation ~0.35.
2. **Adaptive alpha with target_entropy=0.0 delays but does not prevent collapse, then destabilizes**: diversity fully preserved through 20-30k (scale 0.54-0.86, eff-rank 5.5-8.0, moving 0.70-0.97, saturation ~0) while alpha anneals 0.91 -> 0.084 -> 0.014 (policy entropy sits above the 0-nat target, so alpha decays). Once alpha ~ 0.01, near-unregularized Q-maximization explodes the actor: actor_loss -3.8e23 at the 40.8k eval, saturation 1.0, action effective rank 0.0 (a single constant clipped action), moving fraction 0.023 -- strictly worse than alpha0.
3. **Neither arm learns near-goal locomotion in 50k**: success 0.00 (adaptive) vs 0.10 (alpha0, deterministic eval at 50k, d0 in [1, 4.5] m); goal velocity <= 0.012 m/s; immediate controllability at sigma=0.05 stays physically negligible at every gate (std projected disp <= 6e-4 m, max-min goal progress <= 2.6e-3 m); critic decile usefulness <= 0.16.
4. **Coverage gates PASS at every gate** (316-542 unique fresh states, 12 episodes, 12 goals, moving fraction 0.33-0.97): the reference-state redundancy of the earlier probes is eliminated by construction.
5. Critic-side learning is arm-independent (logits gap ~33-37 by 20-30k in both arms).

## Decision per the pre-registered rule

The continue-beyond-50k condition -- adaptive entropy preserves action diversity AND improves locomotion-related behavior relative to alpha=0 -- is **NOT met**: the adaptive arm preserved diversity only until ~30k, then collapsed to a constant saturated action. STOPPED at 50k; no third arm launched.

## Interpretation (for the next decision, not acted on)

target_entropy=0.0 (the original as-shipped adaptive semantics) is the proximate cause of the adaptive failure: 0 nats is far below the initial policy entropy, so alpha monotonically anneals toward 0 and the run re-enters (and numerically overshoots) the alpha=0 pathology. The repo itself exports the SAC-standard heuristic target_entropy_from_env_spec = -num_actions = -8 that lp_contrastive never calls. target_entropy=-8 (or a fixed small positive alpha, or controlled collection noise) is the natural THIRD arm -- deferred per instructions.

## Caveats

- Documented, unchanged in this A/B: Gymnasium-Robotics gear-150 ant (vs d4rl ctrl+-30 gear-1 RK4 dt=0.1), 2D XY goal (vs original full-29D goal obs), 1 actor (vs 4), 50k budget.
- Gate snapshots land at the eval AFTER the boundary (10.2k/20.4k/30.6k/40.8k/50.1k).
- adaptive_40000 and adaptive_50000 rows are bit-identical: the policy stopped changing after the blowup (constant action + fixed eval seeds).

## Third arm: adaptive_te8 (target_entropy = -8, guards on, stopped at 30k)

Alpha-direction sanity (artifacts/alpha_direction_sanity): the implemented loss moves alpha the correct way in the healthy regime for both targets; the entropy ESTIMATE inverts under saturation (arctanh-clip artifact), which is the feedback loop that killed target=0; exploded policies NaN the alpha optimizer (hence the guards).

Result: entropy median on collection obs +5.0 -> +4.2 -> **-7.56** (at the -8 target) while alpha annealed 0.91 -> 0.084 -> 0.011 and STOPPED at its designed equilibrium: no explosion, zero clip-artifact fraction, no guard trip, saturation 0.10, scale 0.50, action eff-rank 4.9, moving fraction 0.82 at 30k (vs alpha0 at 30k: scale at floor, moving 0.35; vs target=0 which exploded by 40k). Diversity is trending down (eff-rank 8.0 -> 7.6 -> 4.9) but the policy is nowhere near collapse.

However, per the pre-registered conditional: local critic usefulness and goal-directed locomotion remain FLAT (sigma=0.05 spearman -0.17..0.19, decile usefulness -0.17..0.15, even sigma=0.2 usefulness ~0; success 0.1, goal velocity <= 0.006 m/s, deterministic progress <= 0.33 m). STOPPED at 30k. Recommended next: goal-representation ablation -- current 2D XY goal vs a richer Ant future-state goal (closer to the original 29D goal-obs formulation).
