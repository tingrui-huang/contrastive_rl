# Source boundaries and reviewed evidence

This is a repository-grounded design review at 7815167. No original teacher
recording/transcript was found in the local repository or supplied messages.
The user's present description is the authority for the intended shared
generator and two-loss formulation. Repository “teacher” policy code is not a
research-teacher transcript. No statements are attributed to a recording that
was not available.

Relevant repository sources (paths relative to repository root):

* `artifacts/ett_death_admissibility/frozen_f4_gate_v1/REPORT.md`,
  `artifacts/ett_death_admissibility/frozen_f4_gate_v1/DERIVATIONS.md`, and
  `scripts/audit_ett_death_admissibility.py`:
  scoped probability-one absorption obstruction and its stated caveats.
* `crl/envs.py`, especially `TwoRouteSwampWindyEnv.step` and the F4 wrapper:
  native death, action noise, collision, reward and history semantics.
* `artifacts/death_observability/f4_p30_expert_s01/REPORT.md`:
  observation timing, finite-data separability and ambiguity, historical probe
  and diagonal-model results. These do not become generator targets here.
* `notes/ett_adversarial_fixed_actor_spec.md`,
  `ett/convex_action_transition.py`, `ett/diagonal_transition.py`,
  `ett/run_convex_adversarial.py`, `ett/rollout_return.py`:
  explicit freeze, constant diagonal loss, 32-parameter search, convex proof,
  hard task-return objective and actual sampling path.
* `artifacts/ett_geometry_comparison/zero_s01_u16_v1/REPORT.md`:
  both geometries reduce independent return in optimizer seed 0, neither
  establishes reduction in seed 1; geometry differences inconclusive.
* `artifacts/ett_structure_audit/fork_saved_s01_v1/REPORT.md` and
  `artifacts/ett_convex_set_component/fork_segment_v1/SPEC.md`:
  extra geometric exclusions and fixed-convex-set alternative, separately from
  the learned corrections that did not propose effective descent.
* `artifacts/ett_policy_improvement/h3_mix10_u2000_s01_v1/REPORT.md` and
  `artifacts/ett_future_window/shared_h3_full_u2000_s01_v1/REPORT.md`:
  bounded CRL integrations with no observed native benefit; frozen-model scope.
* `artifacts/synthetic_shared_response/l025_1_4_s012/REPORT.md`,
  `artifacts/synthetic_lipschitz_response/l1_s012/REPORT.md`,
  `artifacts/synthetic_diagonal_budget/lambda0004_s012/REPORT.md`:
  shared fitting, bound enforcement and soft-loss/diagonal-error tradeoffs in
  deliberately supplied scalar objectives, not validated PointMaze badness.
* `artifacts/critic_continuation/f4_p30_alpha_s01/REPORT.md`:
  raw critic's inverted local ordering, rejected as a pessimistic score.
* `artifacts/return_readout/f4_h20_g095_v1/REPORT.md` and
  `artifacts/fixed_actor_continuation/f4_h20_g095_v1/SUMMARY.md`:
  broad predictive signal does not establish reliable local continuation
  ordering on the needed off-diagonal conditions.

Original models are preserved and hashed during preparation; their data-derived
scores or hidden labels are not inputs to the proposed objective. Public papers
linked by older specifications were not used as a substitute for the missing
teacher source or treated as new requirements.
