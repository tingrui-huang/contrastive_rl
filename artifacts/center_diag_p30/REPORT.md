# Fixed-center controller — failure diagnosis under p_active=0.30

Diagnosis only. Env/controller/horizon/mud/protocol unchanged (ablations are
instance-level overrides in separate artifacts). Control law = RP.run_route
center, verbatim, with per-step instrumentation added. n=600, seed 20260726,
horizon 700, SUCCESS_DIST 0.5, stuck threshold vx<0.1 for >=100 steps.

## Headline

Center success = **0.7967** (478/600). The shortfall is fully explained; it is
**not** an environment bug and **not** rockfall contamination.

- **ENV-BUG CHECK: PASS.** trigger/drop/impact counts are exactly 0 on all 600
  center episodes. The center route never interacts with rockfall.
- **Every failure is a horizon timeout.** Failed episodes all run the full 700
  steps (mean fail length 700.0 vs success 653.1); the env sets no early
  termination for center (no rockfall -> no `_dead`), so the taxonomy is about
  *why* the ant fails to reach the goal in time.

## Failure taxonomy (122 failures = 20.3%)

| category | count | % of 600 |
|---|---|---|
| controller_handoff_failure | 46 | 7.67 |
| timeout_near_goal | 38 | 6.33 |
| physical_fall | 36 | 6.00 |
| stuck_oscillating | 2 | 0.33 |
| timeout_insufficient_progress | 0 | 0 |
| route_deviation_wall | 0 | 0 |
| numerical_unknown | 0 | 0 |

## Where failures happen

- **max-x histogram (1 m bins):** x=8 → 66, x=9 → 46; 92% of failures reach x≥8.
  The mud is MUD_X 2.4–5.6 and handoff is x=6, so **failures are not in the mud
  itself** — the ant crosses the mud and the handoff, then fails in the final
  approach to the goal because the mud slowdown has eaten its clock budget.
- **final-distance (fail):** p10 0.74, p50 2.69, p90 8.79 — a bimodal spread:
  some end just outside the 0.5 success radius (ran out of time near goal),
  others end far (fell / base-policy circling).
- **min torso-z:** all p50 0.38; fail p10 0.245, min 0.228 — the physical_fall
  subset tips over (low z); the rest stay upright.
- **longest low-vx interval (fail):** mean 41 steps, max 165 — only the 2 stuck
  episodes truly wedge; most failures are slow-but-moving, not jammed.

## Ablations (same episode seeds; diagnosis-only, no artifacts overwritten)

| condition | success | residual failures |
|---|---|---|
| baseline | 0.7967 | handoff 46, near-goal 38, fall 36, stuck 2 |
| **A. mud/drag disabled** | **0.9383** (+0.142) | fall 30, handoff 7 |
| **B. 2× horizon (700→1400)** | **0.9267** (+0.130) | fall 38, handoff 6 |

### What the ablations prove

- **timeout_near_goal (38)** and **most of controller_handoff_failure (46→6–7)**
  are eliminated by **both** ablations → these are **clock-limited timeouts**: the
  ant gets to the far side but the mud slowdown upstream leaves too little horizon
  to converge on the goal. More time *or* less drag fixes them.
- **physical_fall (36)** is **not** fixed by more time (B: 38) and only slightly
  by removing mud (A: 30) → it is **mud- and horizon-independent**: the frozen
  walker/base-controller occasionally tips the ant over regardless of the center
  cost.
- **Extending the horizon recovers as much as removing the mud** (+0.13 vs +0.14).
  If the mud primarily *destabilized* the ant, extra time could not help (a tipped
  ant stays tipped) — but it recovers +0.13. So **the center cost is primarily a
  SLOWDOWN/timeout cost (by design, the viscous drag), not a destabilizing one.**
  The smoke run's n=20 "mud-off = 1.0" was small-sample noise; at n=600 neither
  ablation reaches 1.0.

### Decomposition of the ~20% shortfall

- **~13–14 pts — mud-slowdown timeout (intended cost).** The drag slows the ant
  (V_CENTER 0.8 through the bog) enough that ~1 in 7 episodes cannot finish within
  700 steps. This is the center route's designed cost doing its job.
- **~6 pts — frozen walker/base-controller intrinsic ceiling.** Tip-overs plus a
  few base-policy non-convergences that neither more time nor mud-removal fixes.
  Consistent with the known ~0.85–0.90 controller-stack ceiling on this maze.

## Mask-invariance (the 0.75/0.84/0.78/0.81 question)

Paired initial states across the 4 forced mask patterns (k=40):
`per_pattern_success` = 0.9 / 0.9 / 0.9 / 0.9, **0 success disagreements**,
`max_final_xy_spread = 0.0`. The center trajectory is **exactly mask-invariant**
(mud coverage is position-only; center never enters a trigger band). The
0.75/0.84/0.78/0.81 spread in the reconcile table was **pure initial-state
sampling** across pattern buckets, not any mask effect.

## Bottom line for adopting p_active = 0.30

The center controller at 0.796 is a **healthy, well-understood robust-policy
anchor**: zero rockfall interaction (verified), exactly mask-invariant, and its
shortfall is ~14% intended mud-timeout + ~6% frozen-controller ceiling — nothing
anomalous. It does **not** block adoption of p=0.30.

Caveat worth noting for the sweep interpretation (not an action): part of the
teacher-over-center gap is that the teacher runs the **fast side lane** (1.1) and
only briefly detours, while center runs **slow through the mud** (0.8) — so some
of the "value of the mask" gap is a lane-speed effect, not purely mask-information
value. The blind controller (fast side lane, no detour) at 0.53 isolates the
mask-reading value: teacher − blind on the same fast lane is the clean confounder
signal.

## Representative failure GIFs

Under `gifs/`. Fidelity: mujoco carries solver warmstart across resets, so
reset-advancing does NOT reproduce an episode. The GIFs are rendered by REPLAYING
the exact episode sequence (reset + full rollout) in one env from the diagnosis
seed, capturing frames only on targets — verified byte-identical to the taxonomy
(all 17 rendered episodes are genuine fails in their category).

  physical_fall            ep 3, 13, 32, 37, 52
  controller_handoff_fail  ep 4, 12, 14, 23, 33
  timeout_near_goal        ep 9, 24, 41, 53, 97
  stuck_oscillating        ep 301, 527  (only 2 exist)
