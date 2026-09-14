# PointMaze oracle alive-motion substitution diagnostic

Sealed on 2026-09-14 before any new native verification step or oracle rollout. This diagnostic is pinned by hashes to the completed supervised repair experiment at commit `813c3db1242d7f30c6b2ae20b02a1770382f4338`; current HEAD equality is not required. Historical artifacts and production modules are read-only. Nothing is trained, committed, or pushed.

## Question and fixed comparisons

The diagnostic asks whether exact alive-position movement is sufficient for the two frozen repaired failure heads, historical nominal, frozen actor, supported onset law, and irreversible failure mechanism to produce credible trajectories. It reuses the 64 native final actor episodes and the two saved repaired-position/repaired-head rollouts. It generates exactly two arms: native alive motion plus repaired head 0, and native alive motion plus repaired head 1. No other factorial arm, checkpoint, training set, or complete native evaluation episode is generated.

Each arm uses the same 64 reset roots, fixed F4 goal, horizon 50, discount 0.95, and 64 paths per root. The model budget is exactly 8,192 paths and at most 409,600 oracle-motion transition slots. Training updates and new training/evaluation episodes are capped at zero. Interrupted stages resume from saved arrays.

## Oracle movement and numerical conventions

Only the learned alive-position proposal is replaced. At every step the unchanged nominal samples `xb` from the current generated F4 observation and fixed goal, and the unchanged actor samples `xq` from that same observation. The oracle then applies the pinned `TwoRouteSwampWindyEnv.step` alive-motion map: copy the action as NumPy float64, add an explicitly supplied two-dimensional Gaussian noise draw with standard deviation 0.01, clip each coordinate to [-1,1], then run ten `dt=0.1` substeps. Within every substep X is proposed and wall/boundary checked before Y. The physical XY remains float64 across time. The emitted newest frame and F4 history are float32, newest first.

The native environment's hidden bits, death, termination, and reward are never consulted in oracle rollout generation. Natural advice is not supplied to motion. No learned anchor, residual response, gate, mixture atom, or proposal rectangle is evaluated; those diagnostics are not applicable. Native machinery is used only in the separately budgeted equivalence test.

The existing repaired head receives the unchanged observable inputs: standardized emitted F4, `xb`, `xq`, and emitted next-XY displacement. Onset probability is exactly zero unless the emitted next position satisfies `3 <= x < 6` and `3 <= y < 4`; physical-versus-emitted support disagreements are recorded. The existing onset uniform stream is retained. Fatal incoming movement is kept. Failure is permanent; later XY and physical position freeze, F4 shifts normally, and reward is zero. Reward otherwise uses the native physical-position goal radius below 2; disagreement with the same test on emitted float32 XY is recorded rather than hidden.

The root PRNG key is the completed repair rollout seed. Each time key is split, in the existing order, into nominal, actor, motion, and onset keys. Motion noise is generated for every path/time address before failure masking, so paths are never compacted. The same keys are used in both head arms. Once states diverge, nominal and actor react to each arm's own observation. Before first hazard entry, supported onset is zero, so the two arms must have pathwise-identical entry occurrence, time, and location.

## Equivalence verification

Before rollout interpretation, compare the standalone movement helper with native `TwoRouteSwampWindyF4Env.step` at identical float64 physical state, F4 history, action, and explicitly injected noise. Eight one-step cases, each under all-clear and all-active hidden bits, cover free movement, static walls, a corner/sequential-axis case, action clipping, hazard entry, hazard exit, a fatal incoming hazard transition, and an outer boundary. Two all-clear multi-step sequences of lengths 8 and 10 cover direct hazard traversal and lower-route wall/history behavior. The predeclared total is 34 native calls under a cap of 64.

The verification requires bit-exact float64 physical XY and float32 F4 agreement. Changing hidden bits must not change incoming movement, including cases where native death status differs after landing. It separately verifies clipping and reports physical/emitted hazard and reward-boundary classifications. Native death/reward outputs are audit fields only and are discarded. Scientific rollout generation cannot start unless every movement check passes.

## Metrics and uncertainty

The primary outcomes are first hazardous landing occurrence, zero-based time conditional on entry, and mean entry X/Y. They precede the first possible onset. Secondary outcomes are failure occurrence, onset time, reward occurrence, survival without reward, conditional return given reward, mean discounted return, and alive-before hazardous opportunities including the fatal opportunity. Native exposure and both model exposures use the same alive-before definition.

For each head seed, report the oracle arm against its corresponding saved repaired/repaired rollout and the native reference. Save differences, absolute native errors, and the oracle-minus-learned change in absolute error. Reuse the completed experiment's 2,000 whole-episode bootstrap weights; the 64 paths per root are predictive integration, never independent native evidence. Do not require every secondary metric to strictly improve, interpret a confidence interval containing zero as equivalence, or add samples after seeing results.

Interpret results using cases A-D in `config.json`. This is retrospective localization because it reuses the 64 native episodes. It is simulator-assisted and environment-specific, not a learned ETT result, observational-identification result, pessimistic-objective validation, actor-improvement experiment, or claim about AntMaze.
