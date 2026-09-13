# Reproduce the bounded transition-model comparison

From the `feature/pointmaze-causal-transition` worktree, use the existing local
Python/JAX environment. Exact frozen paths and hashes are in provenance.json.
Missing checkpoints fail explicitly. No native server is needed.

```bash
python -m unittest scripts.test_geometry_comparison scripts.test_convex_set_transition scripts.test_convex_action_transition.ConvexChecks.test_analytic_descent_sign_and_zero_weight -v
python -m scripts.prepare_geometry_reproduction --from-run artifacts/ett_geometry_comparison/zero_s01_u16_v1 --out-dir artifacts/ett_geometry_comparison/NEW_RUN
python -m ett.run_geometry_comparison --out-dir artifacts/ett_geometry_comparison/NEW_RUN --phase train
python -m ett.eval_geometry_comparison --run-dir artifacts/ett_geometry_comparison/NEW_RUN
python -m scripts.check_geometry_comparison --run-dir artifacts/ett_geometry_comparison/NEW_RUN
python -m ett.report_geometry_comparison --run-dir artifacts/ett_geometry_comparison/NEW_RUN
```

The original experiment is already complete. These commands, for a later explicit
reproduction request, prepare a separate directory and deliberately reuse the saved
streams; they are not a new independent evaluation. The reproduction preparer
validates the source experiment's hashes and records intentional seed reuse before
any sampling. It was added for reproducibility after this run and does not start
training. No reproduction run was performed in this task.

The actual first preparation command used for this experiment was:

```bash
python -m ett.run_geometry_comparison --out-dir artifacts/ett_geometry_comparison/zero_s01_u16_v1 --phase prepare
```

Ordinary prepare rejects namespace reuse once the experiment is present. An
authorized new independent replication needs a newly preregistered namespace;
the exact-reproduction helper does not label its reused streams as fresh.

For verification/presentation of this completed experiment without any sampling:

```bash
python -m scripts.check_geometry_comparison --run-dir artifacts/ett_geometry_comparison/zero_s01_u16_v1
python -m ett.report_geometry_comparison --run-dir artifacts/ett_geometry_comparison/zero_s01_u16_v1
```

The checker needs the local cache files named in the ledger (or caches regenerated
by a full exact replication). Cache files preserve each completed signed-query
return, monitor, diagnostic draw and full evaluation record for safe resumption;
they remain local. Consolidated histories, all perturbation directions, all 17
theta iterates, independent evaluation arrays and final small coefficient
checkpoints are retained outside cache. No cache is silently fabricated.

The runner charges a sampler call before executing it and retains any failed
attempt. Re-running train/evaluate on an intact partial directory resumes from
matching completed caches; it verifies signatures rather than reusing a different
theta/key/mode. An uncached retry is charged again and cannot exceed the cap.
Do not change source hashes or prepared manifests to hide a partial-run repair.

New checkpoint API:

```python
save_convex_checkpoint(path, model, final_theta)
model = load_convex_checkpoint(frozen_diagonal, path, require_geometry=True)
```

The file carries theta, L, geometry_mode and format_version. Historical theta/L
files still load as rectangle only when new-format metadata is not required.
The comparison evaluator always requires it and verifies it against the config.

`eval_*.npz` holds 512 paths/model with states, actions, x_prime, rewards, returns,
goals, anchors, corrections, proposals, geometry kinds, eligibility, segment
endpoints/fractions, reference rectangles/boxes and boundary flags. Scalar masks
have [episode,time] shape, XY [episode,time,2], F4 [episode,time,8]; states include
the initial frame. `evaluation_seed` and `path_index` retain pairing identifiers.
Geometry kind 0 means actual reference/fallback box; 1 means actual segment.
For rectangle steps, segment start/end are anchor placeholders and fraction=0;
these are not actual segment-selection or projection statistics.

`fixed_component_outputs.npz` holds 512 common fresh anchor draws followed by
225 previously saved fork/descent contexts. Every trained theta is evaluated
under both geometry modes on those same inputs. These arrays are deterministic
component outputs, not trajectories. The standalone selector count qualification
and corrected emitted set counts are in fixed_component_set_audit.json.

No script here launches a native rollout, CRL update, critic/readout fit, failure
scorer or alternate optimizer. The original historical runner remains available;
the isolated two-mode runner is the command for this controlled comparison.
