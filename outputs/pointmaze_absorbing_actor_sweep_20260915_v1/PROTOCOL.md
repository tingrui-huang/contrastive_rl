# G2: BC-coefficient sweep of the actor stage on the frozen G1 critics

Sealed before any update. Base: G1 `outputs/pointmaze_absorbing_integration_20260915_v1`
(commit `a52c0a5` plus the uncommitted `ett/absorbing_ett.py` and
`ett/pointmaze_absorbing_integration.py`, hash-pinned in `config.json`).
Strictly offline: zero environment calls, zero native steps; model rollouts
use the G1 absorbing-freeze backend (`manski_absorbing`) and, as an optimistic
reference, `diagonal_motion`.

## Question

G1 flipped the P critic's fork preference (+1.071) but the unchanged actor
stage (BC 0.5, 1,000 updates) left the P actor mostly on the shortcut (down
11.3%, right 75.7%) with BC/critic gradient cosine -0.86. G2 isolates whether
the actor bottleneck is the BC coefficient alone. No AWR, no change to the
critics, batches, keys, roots or evaluation code.

## Design

For each `bc_coef` in `[0.5, 0.2, 0.1, 0.05, 0.0]` and each arm `O`, `P`:
start from the G1 initial actor and actor-optimizer state with the arm's
frozen G1 critic (`critic_{arm}_final.pkl`, target and optimizer included),
run exactly the sealed 1,000 actor updates on `plan["actor_*"]` batches with
`plan["actor_keys"]`, objective `bc_coef * bc_nll + (1 - bc_coef) * critic_term`
(the production convex form; `0.5` is G1 and must reproduce its actor
checkpoints byte-for-byte). Fixed final iterate; no selection inside a run.

## Metrics per (coef, arm), all on sealed rows

1. Canonical-goal action distribution on the 656 held-out first-fork roots
   with the sealed `fork_eps` (128 innovations per root): down / right / other
   by the sealed direction masks, mean action, saturation fraction
   (any component beyond 0.99 in magnitude).
2. Paired changes over roots: versus the same arm at 0.5 (the G1 actor), and
   P minus O at the same coefficient; 2,000-replicate root bootstrap, seed 0.
3. Critic-induced preference: the arm's own frozen critic scored at the
   actor's samples minus at the canonical down and right probes.
4. Model rollouts from the 656 fork roots under the extracted actor, canonical
   goal, horizon 49 masked to the remaining length, common keys across actors:
   `manski_absorbing` with 32 paths per root (reach, absorbed, entered support,
   lower-route entry, discounted return) and `diagonal_motion` with 16 paths
   (reach, lower-route entry).
5. Non-fork retention on the sealed 1,024 away rows: BC-NLL of the recorded
   action and its change versus the initial actor, mode-action L2 change
   versus the initial actor, landing-bin agreement of the mode action with the
   recorded action, an OOD proxy (fraction of mode actions whose (current cell,
   landing cell) pair never occurs in the training partition), and
   `manski_absorbing` rollouts from those states with the canonical goal
   (8 paths; reach, absorbed).
6. Gradient components on the 4 sealed gradient batches at the final actor:
   BC and critic raw gradient norms, cosine, and the coefficient-weighted norms.
7. Learning-curve tail (last 100 updates): loss, BC-NLL, critic term, sample
   entropy, scale median, |loc| mean, saturation fraction, gradient and update
   norms.

## Decision rule

- Primary: some coefficient gives the P actor down >= 0.50 and down/right >= 1.5
  on the 656 roots.
- Retention (same coefficient, P arm): away landing-bin agreement at least
  0.85 times its value at 0.5, and away-state `manski_absorbing` reach not more
  than 0.05 below its value at 0.5.
- O control: the O actor at that coefficient must not show a comparable
  down shift (its critic prefers right, endpoint -0.305); an O down
  probability above 0.25 is flagged.
- Selection: the largest coefficient passing primary and retention is the G3
  candidate. `0.0` is diagnostic and is not selected unless retention is
  intact and no positive coefficient passes. If no coefficient passes, stop
  and recommend AWR extraction (G2b).

Budget: 10 actor runs of 1,000 updates; rollout cap 30,000,000 model
transitions; zero critic updates. No commit or push.
