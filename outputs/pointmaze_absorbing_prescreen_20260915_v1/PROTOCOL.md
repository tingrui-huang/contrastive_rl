# G0 prescreen: does an absorbing Manski freeze channel flip the fork preference in model rollouts?

Sealed before any rollout. Starting commit `a52c0a5a42d4b836821e462fb2d5b411732aa316`
on `feature/pointmaze-causal-transition`. Strictly offline: zero environment
steps, zero training updates, no hidden bits, no death labels, no teacher-mode
labels. Only `obs` and `act` of the fixed 6,600-episode F4 dataset plus the
eligible frozen checkpoints of the 09-15 pipeline are read.

## Question

The eligible ETT family (diagonal atom/K3 + gated action-response, L=1) cannot
express absorbing failure, so its pessimism is bounded by XY jitter (E25:
-2.4%). This prescreen asks whether replacing the pessimistic *XY response*
by a pessimistic *freeze channel* -- observational freeze law on the diagonal,
per-step Manski upper bound off the diagonal, absorbing persistence -- makes
model continuations prefer the safe detour (down) over the shortcut (right) at
the fork, under the same frozen actor and nominal. It is the analogue of the
archived Manski port's MC prescreen (`scripts/manski_route_diagnosis.py`),
with the hazard discovered from visible data instead of read off the map.

## Eligible inputs (hash-pinned in `provenance.json`)

- dataset `artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz`
  (`obs`, `act` only);
- diagonal `artifacts/ett_distribution_matching/f4_p30_s01_guarded/s0_L0p25_lambda0/final.pkl`;
- nominal `artifacts/nominal_policy/f4_p30_expert_only_mdn_k5_s0/best.pkl`;
- frozen actor `.../alpha0_seed0/final.pkl` (step 150,000), via `ett.pointmaze_region_pilot.Kernel`;
- the 09-15 final ETT offsets `outputs/pointmaze_offline_causal_integration_20260915_v1/checkpoints/ett_final.npz`;
- the 656 held-out first observable fork contexts from the 09-15 `sampler_plan.npz`
  (`fork_context_episode`, `fork_context_time`), so results are on the same roots
  as the 09-15 endpoint-gap probe.

Split: the established seed-0 permutation (660 held-out, 5,940 train). All freeze
tables are estimated on train episodes only.

## Visible freeze statistics (train episodes, obs/act only)

- A transition is *stationary* if newest XY is exactly unchanged (`1e-7`).
- A *freeze onset* is a moving transition followed by a stationary one; its
  landing cell is `floor(clip(next XY))` on the 9x5 grid.
- An onset is *absorbing* if the stationary run continues to the episode end.
- The *absorbing support* `S_abs` is the set of landing cells with at least 10
  observed onsets and absorbing fraction >= 0.5. Both thresholds are fixed here.
- Persistence given the full frozen F4 signature (four equal frames) is reported
  by cell but not used by the model: absorption is implemented as a flag.
- The action bin of `(s, x)` is the landing cell of `floor(clip(s_xy + x))`,
  the same visible map the teacher's own `landing_cell` uses (box bounds only).
- Descriptive Manski tables (propensity of each landing bin, observational
  onset rate, implied upper bound) are reported for the holding, fork and first
  swamp cells, on all train episodes and on the eligible teacher population
  `[1200, 6000)`. They are not model inputs.

## Model arms (common random numbers across arms and forced actions)

State is F4 (8) + a per-path frozen flag. At each step: `x' ~ nominal(s)`,
`x = actor(s)` except `x = forced` at the first step. Frozen paths keep XY,
shift F4, and receive zero reward.

- **A0 `eligible_response`**: the 09-15 final ETT exactly (`Kernel.rollout` with
  `ett_final.npz`; anchor at `x'`, response `M(s)(x - x')`). Predicted: nearly
  action-blind, because `M` is small at these offsets.
- **A1 `diagonal_motion`**: the diagonal queried at the executed action,
  `y ~ P_diag(. | s, x, x)`, including its stationary atom; no absorption.
  This is the observational motion law under the assumption that the hidden
  bits act only through freeze, not through alive motion.
- **A2 `diagonal_motion_absorbing`**: A1, and an atom draw whose cell is in
  `S_abs` makes the path frozen from the next step (observational persistence).
- **A3 `manski_absorbing`**: A2, and if `bin(s, x) != bin(s, x')` and the
  sampled landing cell is in `S_abs`, the path is frozen from the next step
  (fatal incoming movement retained). Averaged over `x'`, this is the per-step
  Manski upper bound `pi_B q_obs + (1 - pi_B) 1[landing in S_abs]`.

Forced first actions: down `(0, -1)` and right `(1, 0)`, as in the matched-fork
diagnostic. 64 paths per root per arm per action; horizon = remaining steps
`50 - t`; discount 0.95; reward `1[dist(newest XY, goal) < 2]`.

## Metrics and decision rule

Per root, averaged over the 64 paths, then paired down-minus-right across the
656 roots with a 2,000-replicate root bootstrap (seed 0): discounted return,
goal-reach (any reward), strict success (min distance < 0.5), lower-route
entry (any y < 2), absorbed-by-end, first hazard landing while alive.

- **Primary (G0 pass)**: in A3, down-minus-right discounted return is positive
  with a 95% interval excluding zero, and the goal-reach ratio down/right is at
  least 1.5 (the archived port's 1.06x flip died in training; 1.79x survived).
- **Attribution**: A0 |down-minus-right| return below 0.05 confirms the
  eligible ETT is action-blind; A1 gives the no-freeze preference; A2 isolates
  the confounded observational freeze; A3 - A2 is the Manski contribution.
- G0 failing does not stop reporting; it stops G1.

Budget cap: 20,000,000 model transitions. Nothing here selects a checkpoint or
tunes a threshold after seeing rollouts. No commit or push.
