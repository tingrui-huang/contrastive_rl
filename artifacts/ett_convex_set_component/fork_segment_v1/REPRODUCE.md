# Reproduce component verification only

Run from the branch worktree with the existing JAX/NumPy environment. Recorded
versions are in verification.json (Python 3.14.4, JAX 0.10.2, NumPy 2.4.4, CPU).

```bash
python -m unittest scripts.test_convex_set_transition -v
python -m scripts.verify_convex_set_components --out-dir artifacts/ett_convex_set_component/NEW_COMPONENT_CHECK
```

The verifier reads the original saved lower-controller NPZ files, frozen theta
arrays and audit witness identifiers. It never loads a policy, native failure
labels or simulator; it neither samples new stochastic anchors nor chains emitted
states. The single witness matrix is derived analytically after the production
selector has fixed its endpoint, then held fixed across the action grid.
Its numerical results are descriptive component checks, not episode statistics.

The tests also cover the public sampling API with a deterministic stub
displacement draw through the real base projector; this is not trajectory
collection or a replacement frozen checkpoint. To run the four additional
existing emitter checks without invoking even the unrelated toy descent test:

```bash
python -m unittest scripts.test_convex_action_transition.ConvexChecks.test_matrices_and_projection_all_pairs scripts.test_convex_action_transition.ConvexChecks.test_diagonal_identity_and_near_wall_continuity scripts.test_convex_action_transition.ConvexChecks.test_upper_edges_and_uncovered_anchor scripts.test_convex_action_transition.ConvexChecks.test_emitted_energy_score_with_atoms -v
```

Explicit APIs (not a training command):

```python
model = ConvexActionTransition(frozen_diagonal, frozen_theta, bound=1.,
                              geometry_mode='fork_segment')
output, details = emit_with_geometry(theta, state, x, x_prime, anchor,
                                     geometry_mode='fork_segment')
```

Omitting the option retains the historical rectangle implementation and its
diagnostic schema. In the optional mode `geometry_kind==1` identifies segments;
`segment_start`, `segment_end` and `segment_fraction` describe the selected set
and projection. `box_low`, `box_high`, `box_valid` and `rectangle` retain the
reference box, which also certifies the original anchor and supplies fallback.
`projected_to_boundary` refers to relative segment endpoints in segment mode.
The constructor's mode property is read-only; instantiate a new object to change
the mode so JIT configuration remains explicit.

SPEC.md and the preserved historical source snapshot are fixed verification
inputs. Historical experiment manifests are not rewritten to disguise the
authorized production source change. A reproduction output directory gets new
numerical outputs; the report and construction are authored review artifacts.
