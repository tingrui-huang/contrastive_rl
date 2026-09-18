# Mainline pilot (O vs CF-oracle futures, matched offline CRL): code and configuration audit complete; execution BLOCKED on the d05 start checkpoint

Contract: `notes/MAINLINE_CONTRACT.md`.  Script: `scripts/exp_v6_mainline_pilot.py`.
Sealed manifest: `manifest.json`.  Checks: `AUDIT.md` / `audit.json` (real anchor
set, no checkpoint) and `_smoke/AUDIT.md` (toy-scale end-to-end code-path test).
Report: `REPORT.md` (no evaluation exists).

## Blocker (specific)

The contract's continuation agent and actor initialisation is the existing
d05 vanilla agent, training seed 0:
`outputs/antmaze_branch_replay_p050/joint_van_d05/seed_0/final.pkl`.  That
checkpoint (with the rest of the d05 chain: `joint_purebc_d05`,
`critics_van_d05` = `vanilla_g0999_d05`) was trained and stored only on node
30021 (35.199.51.171), which died on 2026-09-18 and now refuses connections.
It is not on node 30043, 30049 or 30108, not in this checkout (`.pkl` files
are never committed; only the d05 evaluation JSONs, manifests and logs were
pulled), and not in the session's transfer bundles (`rt_d05.tgz` holds 107
KB of JSON / Markdown).  The d05 dataset and sidecar ARE local and hash-match
the sealed `exp_detour_ratio/datasets.json` (4533c702... / 8b17a572...).

Per the contract, nothing was substituted: not the pure-BC d05 walker, not
the d20 agent, not a re-trained d05 agent (the same recipe and seed would
give a checkpoint that is not byte-identical to the lost one and has no
recorded hash to check against).  The manifest records the start agent's
identity, role and d05-only provenance chain (from the pulled
`prep.json` / `branch_manifest.json` / `bc_sampling_audit.json`) with
`sha256 = UNAVAILABLE`.

What the user can decide: (a) supply a copy of node 30021's
`joint_van_d05/seed_0/final.pkl` if one exists elsewhere -- drop it at the
path above and the pipeline runs unchanged (the seal then pins its hash);
(b) authorise a re-derivation of the d05 chain with the exp_detour_ratio
recipe and seed 0 on node 30043 (~25 min GPU), to be recorded in the
manifest as "re-derived 2026-09-xx, not the 2026-09-18 file"; (c) something
else.  None of these was taken.

## What was delivered and verified

1. **Method boundary** -- `notes/MAINLINE_CONTRACT.md`: (a) the intended
   offline learned-ETT method, (b) this simulator-oracle pilot, (c) the
   historical diagnostic configurations, with the three sampling interfaces,
   the two arms, matched training, the checks, the rules and the stopping
   rule.  Historical reports are untouched except for a "status under
   MAINLINE_CONTRACT" note at the top of `exp_detour_ratio/SUMMARY.md`,
   `exp_agent_round/SUMMARY.md` and `exp_same_batch/SUMMARY.md`, and a new
   section at the end of `notes/antmaze_branch_replay_plan.md`.

2. **Sampling contract in code** (`scripts/exp_v6_mainline_pilot.py`):
   `AnchorSet` (critic anchors with episode / timestep / state / action /
   task goal / source episode / weight; the recipe buffer's own law, K =
   60,000 draws, 53,747 unique anchors, weight = multiplicity / K),
   `RecordedFutures` / `BranchFutures` + `CriticStream` (positive futures by
   the geometric law truncated at the path end; identical anchor sequences
   across arms for a seed), `ActorStream` (the recipe's `TrajectoryBuffer`
   over the whole d05 file, default law; random_goals 0; BC 0.05 on the same
   rows).  `crl/losses.py`: `build_learner(separate_actor_batch=True)` --
   the critic loss on the critic rows, the unchanged actor loss (critic term
   + BC on the same rows) on the actor rows; the shared-batch and
   `bc_transitions` paths are byte-identical to before.  Documented as a
   sampling-interface change, not a byte-for-byte reproduction of the
   shared-batch implementation.

3. **Sealed manifest** (`manifest.json`, before any generation or training):
   dataset / sidecar hashes, training episode identities (the whole d05 file;
   source ids disjoint from the old 100 held-out and 30 Cnew episodes --
   checked), anchor law / seed / weights / file hash, continuation policy and
   action mode (start agent, mode), generation and evaluation seeds
   (hazards 132000000 + anchor id; eval 3909; reserved 616000005 / 616500000
   untouched), horizon 800, terminal handling (first reach frame / death /
   horizon), goal hold (none; a deliberate departure from the historical
   replays' parking), future law, training configuration (V6 recipe at
   gamma 0.999, 30,000 optimizer updates counted in groups of 4, fresh
   paired critics, actor from the start agent, fresh Adam), comparisons and
   decision rules, script hashes.

4. **Checks run** (`AUDIT.md`): dataset and sidecar hashes match the
   sealed d05 files; no held-out / Cnew episode in training; anchors
   deterministic in the seed, on valid rows, roots equal to the logged rows,
   weights sum to 1; the anchor law's marginals match the declared law
   (draws per episode 60.0 +- 7.87 vs 7.75 expected; relative row position
   0.498; t = 0 share 0.0038 vs 0.0038 expected); critic futures always
   inside the path; actor streams identical across instances for every
   seed, 99.6 % non-reset rows, no future goal across an episode boundary,
   the buffer's own law; the static offline gates G1-G8 pass; no d20
   artefact among the pilot's inputs.  Not checkable without the checkpoint
   / branches: query executed exactly once, branch roots keep the timestep,
   parameters move, evaluation-episode identity -- these are implemented and
   PASS in the toy-scale smoke (`_smoke/AUDIT.md`: 48 anchors, stand-in
   zero-torque continuation capped at 12 steps, 8 updates per arm on the CPU
   from a fresh actor: query once, root timestep and logged first action
   kept, restore max |diff| 0.0, anchor sequences identical across arms while
   the goals differ, critic and actor parameters both change, first batches
   identical across arms).  The smoke is a code test, not a result.

5. **Compute plan (staged, not launched)**: generation of 53,747 branches
   with the d05 agent (~9 M env steps; ~20 min on 20 CPU workers), six
   training runs of 30,000 updates (~5 min each on the 4090L), seven
   evaluations of 300 episodes.  Node 30043 has the code and the d05 files
   staged; `node_mainline.sh` in the session scratchpad runs anchors ->
   seal -> generate -> audit -> train -> evaluate -> audit -> report once the
   checkpoint is in place.

## How this comparison differs from the historical runs

- The critic's anchors are the logged rows under the recipe's own law with
  equal weights in both arms -- not dense strata plus every-60th rows, not
  row 0 of a replay whose row 0 is an arbitrary timestep, not the dataset's
  reset rows.
- The intervention is exactly one thing: the positive futures (recorded vs
  generated by the fixed agent after the logged torque).  No candidate
  queries, no repeated draws, no goal hold, no filtering.
- The actor is trained on the whole dataset under the standard pairing
  (random_goals 0, BC 0.05 on the same rows), in both arms, jointly with a
  fresh paired critic -- not a frozen critic, not reset-only anchors, not
  balanced BC, not a pure-BC or d20 initialisation.
- The start agent is the d05 agent; d20 enters nowhere.
- The primary comparison is CF vs O on common episodes with the practical
  check against the start agent kept separate; evaluation uncertainty and
  seed variability are reported separately.

What the pilot would establish if run: whether counterfactual (oracle)
future supervision helps under this matched procedure.  What it would not
establish: anything about a learned ETT (the oracle is the simulator), and
anything about the historical configurations.
