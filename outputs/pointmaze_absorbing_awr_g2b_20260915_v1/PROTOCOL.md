# G2b: advantage-weighted regression on recorded actions, frozen G1 critics

Sealed before any update. Base: G1 `outputs/pointmaze_absorbing_integration_20260915_v1`
(critics, transition caches, sampler plan, 656 fork roots, canonical goal,
actor batches and keys, all hash-pinned in `config.json`); G2
`outputs/pointmaze_absorbing_actor_sweep_20260915_v1` supplies the comparison
numbers. Strictly offline: zero critic updates, zero environment calls, zero
native steps. Model rollouts use the G1 backends `manski_absorbing` (A3) and
`diagonal_motion` (A1).

## Question

G2 showed that the G1 P critic prefers the detour but the tanh-Gaussian
reparameterised actor cannot realise it: lowering BC produced wall-directed
diagonals (mask "down" 0.41, true lower entry 0.10) and, at zero BC, a
saturated corner policy. G2b asks whether support-constrained extraction —
weighting only recorded actions — converts the critic preference into real
lower-route entry, measured by the landing cell of the first step.

## Objective

For each `beta` in `[0.0, 0.5, 1.0, 2.0, 5.0]` and each arm `O`, `P`: start from
the G1 initial actor and actor-optimizer state with the arm's frozen G1
critic; run exactly the sealed 1,000 actor updates on the sealed
`plan["actor_*"]` batches, with the same row doubling as the production actor
stage (each row appears with its own hindsight goal and with the goal rolled
by one position), minimising

    L = - mean_i  w_i * log pi(a_rec,i | s_i, g_i)
    log_w_i = clip(beta * (Q(s_i, a_rec,i, g_i) - V(s_i, g_i)), -5, 5)
    w_i = exp(log_w_i) / mean_batch(exp(log_w))

`Q` is the arm's frozen critic paired score. `beta = 0` is the uniform-weight
BC control. The actor samples nothing during training; the sealed
`plan["actor_keys"]` are carried but unused.

## Baseline V(s, g): fixed recorded-action bank per cell

For each free cell of the 9x5 grid, 32 recorded actions are drawn once
(seed 0) from training-partition rows whose current XY lies in that cell.
`V(s, g) = mean_j Q(s, a_j, g)` over the bank of the cell containing `s`.
The bank is fixed before training, independent of the actor and of the
row's own action, and never contains the current actor's samples. Because
critics, batches and bank are all fixed, `Q`, `V` and the advantage of every
doubled row are precomputed once per arm and saved (`advantages.npz`).

Diagnostic only: the same advantages under a frozen-nominal baseline
`V_nom(s, g) = mean_j Q(s, x'_j, g)`, 32 draws from the eligible K5 nominal at
`s`, to report the correlation with the recorded-bank baseline.

## Metrics per (beta, arm)

1. First-step landing-cell distribution on the 656 canonical fork roots from
   the sealed `fork_eps` samples: (1,2) lower entry, (2,3) right/holding,
   (1,3) stay, (0,3) left, wall cells, other; plus the sealed direction mask
   and its false-positive rate (mask-down but not landing in (1,2)).
2. A3 rollouts from the fork roots (32 paths, horizon 49 masked to remaining
   length, common keys with G2): reach, absorbed, lower-route entry, entered
   support, discounted return, strict success.
3. A1 rollouts (16 paths): reach, lower-route entry, strict success,
   discounted return, mean per-step displacement.
4. Non-fork retention on the sealed 1,024 away rows: BC-NLL of the recorded
   action and change versus the initial actor, mode-action L2 change,
   landing-bin agreement with the recorded action, OOD proxy, policy scale and
   saturation, A3 rollouts from those states (8 paths).
5. Weight diagnostics per beta: clipping fraction, mean effective sample size
   `(sum w)^2 / sum w^2` per batch, max normalised weight, weight share of
   fork-cell rows.
6. Advantage distribution of recorded actions by current cell and by landing
   cell of the recorded action; for the fork cell specifically, split by
   landing in (1,2) versus (2,3).
7. Loss-curve tail and policy scale/saturation.

## Decision rule

- Primary: P first-step entry into (1,2) >= 0.30 (hard pass), >= 0.40 (strong).
- Retention (P arm, same beta): away landing-bin agreement >= 0.85 times its
  value at beta = 0; away A3 reach >= its beta = 0 value - 0.05; A1 fork reach
  >= 0.9 times its beta = 0 value.
- Secondary: A3 absorbed below and A3 reach above the G2 bc = 0.5 P values;
  O entry into (1,2) above 0.25 is flagged as spurious (its critic prefers
  right).
- Clipping above 0.25 of doubled rows is excessive.
- Selection: the smallest beta passing primary and retention. beta = 5 only
  if retention holds and clipping is not excessive; beta = 0 never. If none
  passes, AWR is reported insufficient and the diagnosis addresses support
  (does the data contain lower-route actions at the roots' cells) and critic
  sharpness (advantage magnitude at the fork).

Budgets: 10 actor runs of 1,000 updates; rollout cap 30,000,000 model
transitions. Fixed final iterates. No commit or push.
