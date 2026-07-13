# Faithful offline antmaze-umaze-v2 path: preparation report (NO training)

Date: 2026-07-13. Phases 1-4 complete; learner training NOT started.

## 1. Dataset audit (Phases 1-2)

**Provenance** (`dataset_provenance.json`): official d4rl v2 URL
(`ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5`),
232,532,949 bytes, sha256 `5ef15257771c50ef4d23c7de001750e96c8bb5d9b6a5e4a821dcfb3065fbd130`,
server Last-Modified 2021-11-05, downloaded 2026-07-13 18:20 local. Stored
OUTSIDE the repo at `D:\Users\trhua\Research\datasets\d4rl\`; not committed.

**Integrity / exact trajectories** (`dataset_audit.json`, phase2):
* 1,000,000 rows; obs 29-dim, actions 8-dim in [-1.000, 1.000] exactly.
* Episode delimiter = `timeouts` ONLY (1,426 true) -> 1,427 episodes:
  1,426 x 701 rows + one 374-row remainder (dropped in conversion, 0.04%).
  `terminals` (8,727) == nonzero `rewards` (8,727): goal-hit markers, the
  episode continues -- exactly the upstream insertion contract
  (distributed_layout.py splits on timeouts, ignores terminals).
* Temporal alignment: within-episode XY step median 0.069 m (p99 0.269,
  max 0.369); across-boundary jump median 9.09 m (min 2.50) -- clean splits,
  no off-by-one. qvel(obs[15:17]) vs dXY/dt correlation 0.967 (dt=0.1).
* Coordinates: 100% of positions within the U_MAZE open-cell footprint;
  all 7 open cells occupied; `infos/goal` constant within episodes,
  clustered in the far corridor (x~0-0.7, y~8.5-10.3).

**Conversion** (`npz_conversion.json`): `antmaze_umaze_v2_offline.npz`
(133.8 MB) -- obs [1426,701,58] (state + ZERO-PADDED goal: zeros except
[:2]=infos/goal), act [1426,701,8] (act[i] taken at obs[i]; final row
zeroed, never sampled), eval_goals [1426,2]. Checks: goal half zero beyond
xy TRUE, goal constant within episode TRUE, actions in +-1 TRUE.

## 2. Online-vs-offline route support (Phase 3)

| metric | online replay (alpha0 250k run) | antmaze-umaze-v2 | ratio |
|---|---|---|---|
| episodes | 360 (self-collected) | 1,427 | -- |
| episode BFS span 0 | 74.2% | 0.0% | -- |
| episode BFS span >= 2 | 5.0% (max span 2) | **97.6%** (span 6: 69%) | 20x |
| detour-containing episodes | 0.0% | **93.2%** | inf |
| corner passage | 3.6% | 95.2% | 26x |
| start cells | 58 spread | all at (1,1) R cell | -- |
| end cells | local | 84% in goal corridor (3,1)/(3,2) | -- |
| action effective rank (of 8) | ~5.5-5.9 (policy) | 7.01 | -- |
| moving fraction (dXY>0.01m) | ~0.69 | 0.793 | -- |
| **positive pairs** (102,400 via exact sampler, gamma=0.99) | | | |
| BFS 0 | 96.14% | 71.2% | -- |
| BFS 1 | 3.72% | 18.8% | 5x |
| **BFS >= 2** | **0.137%** | **10.00%** | **73x** |
| **detour-separated (wall-blocked)** | **0.0059%** | **4.78%** | **815x** |
| future horizon dt | same law | mean 77, median 52, p99 356 | -- |
| euclid dist (median/p90) | -- | 0.46 / 5.63 m | -- |

The dataset preserves route-level and detour supervision after
preprocessing; the online replay had essentially none. Files:
`dataset_audit.json`, `dataset_route_stats.csv`, `visitation_heatmap.png`.

## 3. Exact offline objective (as prepared)

```
L_actor = E_batch[ 0.05 * ( -log pi(a_data | s, g) )
                 + 0.95 * ( alpha * log pi(a | s, g) - min(Q1, Q2)(s, a, g) ) ]
          with alpha = 0 (entropy_coefficient = 0.0, target_entropy unset),
          a ~ pi(.|s,g), a_data = the dataset action stored WITH (s, its episode),
          g = geometric future-state relabel from the SAME episode
              (P(j) ~ 0.99^(j-i), j > i), random_goals = 0.0 (no goal mixing),
L_critic = binary NCE on twin logits [B,B,2] (per-head sigmoid BCE vs identity,
           mean over heads), Polyak tau = 0.005.
Budget    = 1,000,000 gradient updates (offline step clock), batch 1024,
            repr_dim 16, hidden (1024,1024), Adam 3e-4 (both).
Eval goal = zeros(29) with [:2] = goal xy, drawn from the dataset's empirical
            per-episode goals; success = dist(xy, goal_xy) <= 0.5 within 700 steps.
```

## 4. Implementation diff

Base: parallel commit `cb1b90a` had already added `Config.offline_dataset`
(frozen buffer, env eval-only, gradient step clock) and `Config.bc_coef`
with the exact upstream BC blending. Added on top (this change):

| file | change |
|---|---|
| `crl/losses.py` | actor twin-Q reduction `jnp.mean` -> **`jnp.min`** (upstream master; the 2022 vendored snapshot is stale) |
| `crl/d4rl_ant.py` | `OfflineD4rlAntUMazeEnv`: zero-padded XY goal, R-cell-only reset, empirical eval goals; physics/horizon/reward inherited |
| `crl/envs.py` | `offline_ant_umaze` branch (loads `eval_goals` from the offline npz) |
| `crl/train.py` | HARD offline contract: full replay sha256 at freeze + re-checked at every eval and at end; collection `env.step` poisoned to raise; eval-env step counter asserted == evals x episodes x 700; `num_actors` reported 0 offline |
| `scripts/convert_d4rl_to_npz.py` | hdf5 -> npz episodes (timeout-delimited, zero-padded goal, action alignment) |
| `scripts/audit_d4rl_dataset.py` | Phase 2+3 audit |
| `scripts/verify_offline_d4rl.py` | pre-training gates G1-G7 + `build_offline_cfg()` (the canonical run config) |

No online-path behavior changed: all prior runs used `twin_q=False`
(min==mean edit is unreachable there); online 4-actor integration test
re-run after the edit: PASS.

## 5. Pre-training gate results (`pretraining_gates.json`)

| gate | result | evidence |
|---|---|---|
| G1 BC tuples trajectory-aligned | PASS | 512 sampled (s,a,g) bit-exact == (obs[traj,i], act[traj,i], obs[traj,j>i]) of the same episode, via exact RNG replication of TrajectoryBuffer.sample() |
| G2 zero-padded goal contract | PASS | obs[31:]==0 exactly; 10 resets -> >=5 distinct empirical dataset goals; reward=1 standing on the goal |
| G3 twin-MIN in actor objective | PASS | real update_step actor_loss 13.090610 == manual min-version, != mean-version 12.895908 |
| G4 BC gradients raise log pi(a_data) | PASS | mean logprob -652.3 -> -47.4 -> -6.3 -> +5.8 over 3 pure-BC steps |
| G5 offline dry-run, replay immutable, no env.step | PASS | 1426 eps / 998,200 transitions ingested once; 'env collection DISABLED'; frozen sha printed; 0 learner steps; no eval/env interaction |
| G6 runtime hard-asserts armed | PASS | poisoned env.step / per-eval sha / eval-step accounting / end-of-run sha present in crl/train.py |
| G7 run-config contract | PASS | bc 0.05, twin_q, random_goals 0, batch 1024, repr 16, (1024,1024), alpha 0, 1M learner budget, NCE |

**Status: prepared and gated. Learner training NOT started.**
