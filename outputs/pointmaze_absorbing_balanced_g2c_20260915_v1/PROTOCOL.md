# G2c: balanced (s, a)-key sampling of the actor batches, paper objective, frozen G1 critics

Sealed before any update. Base: G1 `outputs/pointmaze_absorbing_integration_20260915_v1`
(frozen critics, sampler plan, 656 fork roots, canonical goal, actor keys; all
hash-pinned in `config.json`). References: G2 (bc sweep, uniform actor rows)
and G2b (AWR). Strictly offline: zero critic updates, zero environment calls,
zero native steps; model rollouts use the G1 backends `manski_absorbing` (A3)
and `diagonal_motion` (A1).

## Why this and not AWR

The discrete WindyCorridor recipe (`causal-contrastive-rl`, SUMMARY pathology
#5 "rare-route undertraining", `configs/corridor_defaults.py`) kept the paper
actor objective `(1 - lam) E_pi[f] + lam log pi(a_data | s, g)`, lam = 0.5, and
fixed the undertrained rare route by **balanced sampling over distinct (s, a)
keys**: pick a key uniformly, then a row within it. AWR was tested there and
dropped (not in the paper; paper actor >= AWR under the same balancing). The
PointMaze actor stage has used uniform replay rows throughout G1/G2; the
fork -> (1,2) rows are 0.27% of them. G2c applies the discrete fix and nothing
else.

## Intervention

Keys are `(current cell, landing cell of the recorded action)` on the 9x5 grid
(`floor(clip(xy))`, `floor(clip(xy + a))`), built from the 5,940
continuation-training episodes only (`obs`/`act`). Keys with fewer than 10 rows
are excluded (22 of 156 keys; their rows are never sampled). For each of the
1,000 actor updates, 256 rows are drawn by choosing a key uniformly among the
remaining keys and then a row uniformly within the key; the hindsight goal is
`t + offset` with the sealed geometric offset law (`oci.draw_offsets`). O and P
receive the identical balanced batches and the sealed `plan["actor_keys"]`.
Row sampling seed 0 is primary; seeds 1 and 2 are robustness repeats.

Everything else is the sealed actor stage: the production objective
`.5 * bc_nll + .5 * critic_term` (`oci.build_actor_step`), the same row
doubling, the G1 initial actor and optimizer state, 1,000 updates, fixed final
iterate, no selection.

Reference arm: the G1 actors (uniform rows, same objective) already saved.

## Metrics per (seed, arm), all on the sealed 656 fork roots / 1,024 away rows

1. First-step landing-cell distribution with the sealed `fork_eps`: the
   **physical** landing cell (ten substeps, X then Y, blocked coordinate
   updates rejected, box bounds and map only, no noise — the same map already
   inside the diagonal model's projection) is primary; the naive
   `floor(s + a)` bin and the direction mask are reported for continuity with
   G2/G2b.
2. Critic scores at actor samples versus the canonical down/right probes.
3. A3 rollouts (32 paths): reach, absorbed, lower-route entry, entered
   support, discounted return, strict success. A1 rollouts (16 paths): reach,
   lower-route entry, strict success, return, mean step displacement.
4. Non-fork retention on the away rows: BC-NLL of the recorded action and its
   change versus the initial actor, mode L2 change, naive and physical
   landing-bin agreement with the recorded action, OOD proxy, scale,
   saturation, A3 rollouts from those states (8 paths).
5. Gradient components on the 4 sealed batches at the final actor; curve tail.
6. Exposure: share of fork-cell and fork -> (1,2) rows in the balanced batches
   versus the uniform plan.

## Decision rule

- Primary: P physical first-step entry into (1,2) >= 0.30 (hard), >= 0.40
  (strong), at row seed 0, with seeds 1-2 within 0.10 of it.
- Retention (P, seed 0): away naive landing-bin agreement >= 0.85 x the G1
  uniform value (0.834); away-state A3 reach >= the G1 uniform value - 0.05;
  A1 fork reach >= 0.9 x the G1 uniform value.
- Secondary: A3 absorbed below and A3 reach above the G1 uniform P values
  (0.673 / 0.232); O physical (1,2) entry above 0.25 is flagged; the P - O
  contrast is the treatment effect because balancing also changes O's BC target.
- If the primary passes with retention, G2c's P actor is the G3 candidate. If
  it does not, the next step is G1' (balanced critic batches as well), not a
  new objective.

Budgets: 6 actor runs of 1,000 updates; rollout cap 30,000,000 model
transitions. No commit or push.
