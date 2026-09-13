# Fixed route diagnostic: methods and verification

## Frozen setup, provenance and coupling

Actual starting commit is **`a23991f2b81708d1df36a00e71846d3fc640da4e`**, exactly the reviewed commit, on `feature/pointmaze-causal-transition` in `.worktrees/pointmaze-p30`. There are no subsequent commits or existing tracked-source changes. Existing untracked artifacts were preserved. The only additions are this diagnostic's controller, runner, focused tests, saved-data reporter and artifact directory. The reports and relevant implementations of the [future-window experiment](../../ett_future_window/shared_h3_full_u2000_s01_v1/REPORT.md), [policy improvement](../../ett_policy_improvement/h3_mix10_u2000_s01_v1/REPORT.md), and [frozen convex model](../../ett_convex_adversarial/l1_matrix32_s01_v1/REPORT.md) were read first.

Use the unchanged native F4 environment: p=.3, goal (8.5,3.5), H=50, gamma=.95, Gaussian action noise SD=.01, clipping to [-1,1], ten axiswise collision substeps, and native absorbing behavior with done=False until the external horizon. Observation `[:2]` is current visible XY, `[:8]` is newest-first F4, and `[8:16]` is the fixed tiled goal. Reset is four copies of (0.5,3.5). Both prescribed polylines lie in open maze geometry; no waypoint change was needed.

Both environments call the **same controller function**, whose only inputs are current visible XY, progress index and fixed waypoints. It advances when distance is <=.25, clips target minus XY componentwise to [-1,1], and retains the final target. No noise, pause, goal relabeling, hidden state, death flag or future outcome enters it. Each model step samples nominal x_prime, obtains this controller's action, samples the unchanged ETT successor, and shifts F4 exactly. The nominal policy, guarded diagonal sampler, both adversarial coefficient arrays, normalization, L=1 and geometry remain frozen. No actor/critic checkpoint is deserialized or used for inference, and no training function is called. Archived actor/data bytes are hashed only to verify prior provenance.

Main reset seeds **51000000..51000127** are fresh relative to the available prior records. The suggested 41000000 range was rejected because prior native tests used 41000002/3. [seed_audit.json](seed_audit.json) records checks of 316 available NPZ files and their numeric seed arrays; source/config/report searches found no prior use of the selected range. Smoke seeds are 51001000/1. Every native/model pairing has exactly matching initial visible observations.

Native controllers start independent simulator instances with the same reset seed. Their default action-noise and swamp RNG streams align while both are alive; absorbed branches omit action-noise draws, so streams can diverge after death. No simulator snapshot or hidden-state intervention is used. Model episode keys are `fold_in(PRNGKey(52000000), reset_seed)`, paired across controllers and checkpoints, with separate nominal and transition subkeys at each time. Each trajectory uses batch size one, preserving exact numerical and RNG replay. These model draws do **not** represent native hidden randomness. Equal initial states do not establish a shared structural noise model.

Exact SHA256s include:

- Nominal: `6376e60185aa616c2d09c5e0250d762cde2fb845c5042488e64910fb3b745f23`.
- Guarded diagonal source: `9e9e1c56387d446b5788612e27e5b7f7743e120186bebc90a78fc6774476184e`.
- ETT seed 0 coefficients: `ce67746e98d06f211664bb591336b9fc4d265eadbf567d2f12e1422c3281d3f6`.
- ETT seed 1 coefficients: `7433349a4eab6d87221e822508b15d23932ab0d90398cb2dd7e139ac4983ba17`.

Full paths, hashes, source versions, runtime and starting status are in [provenance.json](provenance.json). Missing or changed required artifacts cause failure, never reconstruction or substitution.

## Validation, budget and reproduction

Three zero-transition focused tests pass. Every recorded action and waypoint index was independently reconstructed with NumPy; all F4 shifts, commanded goals, reward sequences and discounted sums agree. Native absorption and zero post-failure reward were checked only in evaluation audits. Every available model projection diagnostic is retained: all boxes/endpoints and displacement limits pass. Outer projection rates, shortcut/lower, are **6.188%/15.344%** for seed 0 and **8.000%/15.750%** for seed 1. Base-sampler correction rates are **4.219%/4.672%** and **4.063%/4.328%**. Sampled 21-point segment wall-crossing rates are **0.0469%/0.0156%** and **0.0313%/0.0313%**; these are not native axiswise reachability tests.

Each model/controller has 128 exactly stationary F4 contexts, all at reset, and all resume motion. Moved stationary anchor atoms are 558/558, 1382/1382, 467/468 and 597/597, ordered seed-0 shortcut/lower then seed-1 shortcut/lower. These diagnostic atoms and stationary histories are not death labels. See [model_diagnostics.json](model_diagnostics.json).

All fields of the first four episodes replay exactly for all six combinations. Pre/post frozen parameter-and-normalization signatures and file hashes match. **38,400 main + 600 smoke + 1,200 replay = 40,200 transitions**, below 45,000, with no retries, budget amendment or outcome-driven expansion. Native/model subtotals are 13,400/26,800. Statistical analysis and plotting consume no transitions. Both figures were visually inspected.

For exact reproduction, use a clean checkout containing these diagnostic sources and the required prior input artifacts, but no outputs from this diagnostic in its artifacts tree. The seed-freshness guard intentionally rejects repeating evaluated seeds when those outputs are present. From that PointMaze worktree, using the recorded local Python/JAX/Haiku/NumPy environment:

```bash
python -m unittest scripts.test_route_diagnostic
python -m ett.run_route_diagnostic --out-dir artifacts/ett_route_diagnostic/NEW_RUN --phase prepare
python -m ett.run_route_diagnostic --out-dir artifacts/ett_route_diagnostic/NEW_RUN --phase run
python -m ett.report_route_diagnostic --run-dir artifacts/ett_route_diagnostic/NEW_RUN
```

The last command rebuilds statistics and figures from saved data; this interpretive report was written after analysis. Reproduction on a fresh seed range requires a new explicitly specified diagnostic rather than silently reusing seeds from a completed run. [episodes.csv](episodes.csv) holds all 768 main episode results; [paired_differences.csv](paired_differences.csv) holds 384 paired differences. `main_*.npz` retains all visible states, goals, execution actions, rewards, returns, progress and model diagnostics. `*_auxiliary.npz` holds nominal draws separately. `*_native_audit.npz` and [native_failure_evaluation_only.csv](native_failure_evaluation_only.csv) isolate privileged evaluation information. Smoke records, configuration, seed audit, transition ledger, verification and output hashes are included.
