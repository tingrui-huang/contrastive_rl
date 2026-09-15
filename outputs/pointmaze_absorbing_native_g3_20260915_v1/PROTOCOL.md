# G3: pre-registered native evaluation of the faithful paper actor, bc = 0.2, P vs O

Sealed before any environment step. No training of any kind. The native
environment `point_two_route_swamp_windy_f4_v0` (`TwoRouteSwampWindyF4Env`,
per-cell activation 0.30, action noise 0.01, 50-step fixed horizon, reward
`1[dist(XY, goal) < 2]`, absorbing death on landing in an active swamp cell)
is the source of truth for movement, route, success and absorption.

## Status of the method path

- AWR (G2b) is withdrawn from the method path (not in the CRL paper); it
  remains a diagnostic showing that support-constrained extraction can express
  the P critic preference.
- G2c / G2c' global key-balanced actor sampling are retracted as invalid
  main-path actor sampling: they change the behaviour prior at the fork from
  the benchmark's rare-detour distribution and amplify random-policy rows.
- G3 tests the faithful paper actor objective
  `(1 - bc) E_pi[Q] + bc log pi(a_data | s, g)` with bc = 0.2 on the original
  uniform actor rows (G2 checkpoints), with bc = 0.5 (G1 actors) as the
  baseline pair. No reweighting, no key balancing, no new actor method.
- The native environment is used because the A1/A3 model rollouts
  under-estimate diagonal down-right movement: the diagonal model's projection
  reverts wall-endpoint moves to the current position instead of sliding
  (faithful-actor audit, 2026-09-15).

## Policies (saved checkpoints only; sha256 in `checkpoint_manifest.json`)

1. `P_bc0.2` = `outputs/pointmaze_absorbing_actor_sweep_20260915_v1/checkpoints/actor_P_bc0p2.pkl`
2. `O_bc0.2` = `.../actor_O_bc0p2.pkl`
3. `P_bc0.5` = `.../actor_P_bc0p5.pkl` (byte-identical to the G1 P actor)
4. `O_bc0.5` = `.../actor_O_bc0p5.pkl` (byte-identical to the G1 O actor)

All four were trained with the G1 stratified critics (`critic_{O,P}_final.pkl`
of `outputs/pointmaze_absorbing_integration_20260915_v1`); the uniform-anchor
critic of the faithful-actor audit is a purity check only and is not used.
bc = 0.1 is excluded from G3 and reserved for a separate G3b.

## Primary hypothesis

Under the same actor objective and the same bc, the absorbing-ETT P critic
makes the actor choose the lower route more often and improves native
performance relative to the O critic actor.

## Design

- 200 paired episodes. Episode `i` uses `TwoRouteSwampWindyF4Env(seed = 9300000 + i)`
  for every policy, so the reset, the actuator-noise stream and the hidden-bit
  stream are identical across policies up to the first divergence in the
  number of RNG draws (which happens only after an absorbing death).
- The actor's Gaussian innovation at step `t` of episode `i` is
  `fold_in(PRNGKey(9400000 + t), i)` for every policy (paired across policies,
  independent across episodes and steps). Actions are `tanh(loc + scale * eps)`
  clipped to the action box by the environment.
- Observations are the environment's 16-dim F4 + goal vector; the actor sees
  the canonical goal exactly as in training/evaluation elsewhere.
- No model rollout metric is a primary outcome.

## Route classification (native physical trajectory only)

Cells are `floor` of the physical XY on the 9x5 grid. The fork visit is the
first step at which the agent is in cell (1,3).

- first-fork landing cell: the cell one step after the fork visit;
- lower-route usage: after the fork visit the agent enters cell (1,2) or the
  lower corridor (y < 2) before any hazard-corridor landing (x in [3,6),
  y in [3,4)); "entered (1,2)" and "reached y < 2" are both recorded;
- shortcut usage: a hazard-corridor landing occurs before any lower-route
  entry;
- stuck/other: neither within the horizon, or the fork cell is never visited.
- success/reach: native reward > 0 at any step; strict success: min physical
  distance to the goal < 0.5; time-to-goal: first rewarded step (1-indexed)
  among reached episodes; absorbed: `env.dead` at the end; returns: discounted
  (0.95) and undiscounted sums of native rewards.

## Statistics

Paired bootstrap over the 200 episode ids (2,000 replicates, seed 0) for the
mean difference of reach, strict success, discounted return, absorbed,
lower-route usage, shortcut usage and stuck/other, for the comparisons
P_bc0.2 − O_bc0.2 (primary), P_bc0.5 − O_bc0.5, P_bc0.2 − P_bc0.5,
O_bc0.2 − O_bc0.5. Per-policy means with the same bootstrap. All four policies
share one paired per-episode table.

## Success criteria (fixed here)

- Strong pass: P_bc0.2 − O_bc0.2 reach > 0 with the 95% CI excluding 0; lower-route
  usage difference >= +0.15 with CI excluding 0; P_bc0.2 stuck/other rate <= 0.10.
- Provisional pass: lower-route usage difference > 0 with CI excluding 0; absorbed
  difference <= 0 or CI including 0; reach difference > 0 (CI may include 0).
- Failure cases to diagnose: (1) more lower-route use without a reach gain;
  (2) no lower-route shift; (3) more lower-route use with stuck behaviour;
  (4) P_bc0.2 ~ P_bc0.5.

Budget: 4 x 200 x 50 = 40,000 native steps; zero training updates. No commit
until the report is reviewed.
