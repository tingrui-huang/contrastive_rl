# Offline objective audit + route diagnosis (D4RL AntMaze reframing)

Date: 2026-07-13. All numbers from local files; upstream reference =
`google-research/google-research/master/contrastive_rl` (fetched raw sources).

## 0. What the executed 250k run actually was

`d4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k/` trained **online, on
self-collected data**: replay.npz holds 360 episodes = 252,000 transitions =
exactly the run's env-step count, contributed 90/90/90/90 by the 4 in-process
actors. There is **no D4RL dataset anywhere in this repository** (no .hdf5)
and **no dataset-ingestion code path** in `crl/`. The strong original AntMaze
numbers come from the paper's OFFLINE recipe, which this port has never
implemented.

## 1. How the original OFFLINE recipe works (upstream, verbatim locations)

* `lp_contrastive.py`: `num_actors = 0` for `offline_ant*` ("No actors needed
  for the offline RL experiments"); overrides:
  `samples_per_insert=1_000_000`, `samples_per_insert_tolerance_rate=1e8`
  (rate limiter removed), `random_goals=0.0`, `bc_coef=0.05`, `twin_q=True`,
  `batch_size=1024`, `repr_dim=16`, `hidden_layer_sizes=(1024,1024)`.
  `entropy_coefficient=0.0` (non-image default) -- alpha stays 0 offline too.
  `max_number_of_steps` counts **learner** steps offline
  (`distributed_layout.coordinator`: `steps_key='learner_steps'`), budget 1M.
* `contrastive/distributed_layout.py:194-221`: the **learner node inserts the
  whole D4RL dataset once** via `env.get_dataset()` -> Reverb adder, episodes
  split on `dataset['timeouts']`, obs stored as
  `concat([observations[t], infos/goal[t]])`, actions aligned as
  `adder.add(action=dataset['actions'][t-1], next_timestep=ts_t)`.
* `ant_env.py`: `make_offline_ant` = unmodified `gym.make('antmaze-umaze-v2')`
  + `OfflineAntWrapper`; **eval goal = `zeros(29)` with `goal[:2] =
  env.target_goal`** (zero-padded XY, NOT a settled full state -- the settled
  full-state goal belongs to the upstream ONLINE `AntMaze` class, which our
  `crl/d4rl_ant.py` replicates).
* `contrastive/builder.py` flatten_fn: relabeled goals = full future state
  (`start_index=0, end_index=-1`), geometric discount -- identical to our
  `TrajectoryBuffer.sample()`.
* `contrastive/learning.py` (master) actor loss:
  ```python
  if len(q_action.shape) == 3:
      q_action = jnp.min(q_action, axis=-1)        # MIN over twin critics
  actor_loss = alpha * log_prob - jnp.diag(q_action)
  if config.bc_coef > 0:
      orig_action = transitions.action
      if config.random_goals == 0.5:
          orig_action = jnp.concatenate([orig_action, orig_action], axis=0)
      bc_loss = -1.0 * networks.log_prob(dist_params, orig_action)
      actor_loss = config.bc_coef * bc_loss + (1 - config.bc_coef) * actor_loss
  ```

## 2. Divergence table: original offline vs this port

| component | upstream offline (master) | our port / executed run | status |
|---|---|---|---|
| data source | D4RL antmaze-umaze-v2 inserted once, num_actors=0 | online self-collected replay, 4 actors | **MISSING** |
| GC-BC term | `bc_coef*bc_loss + (1-bc_coef)*actor_loss` | absent (no `bc_coef` field in `crl/config.py`) | **MISSING** |
| bc coefficient | 0.05 | n/a | **MISSING** |
| # critics | twin_q=True (2) | twin_q=False (asserted in run notebook) | **DIVERGENT** |
| actor critic-reduction | `jnp.min` over twins | `jnp.mean` (crl/losses.py:194 AND stale vendored `contrastive/learning.py:255`) | **DIVERGENT (stale vendored snapshot)** |
| random_goals | 0.0 offline | 0.5 | **DIVERGENT** |
| batch / repr / hidden | 1024 / 16 / (1024,1024) | 256 / 64 / (256,256) | **DIVERGENT** |
| entropy | alpha = 0 (same) | alpha = 0 | match |
| relabeling | full future state, geometric | same (`goal_indices=range(29)`) | match |
| eval goal | zeros(29), [:2]=goal xy | settled full 29-dim state (matches upstream ONLINE task) | **DIVERGENT for offline** |
| action alignment | dataset actions in [-1,1]; `action[t-1]` paired with obs[t] via adder | replay stores own [-1,1] actions, same next-obs convention | match (convention) |
| action normalization | CanonicalSpecWrapper, spec asserted +-1; ctrl +-30 gear 1 | clip(a,-1,1)*30, same xml | match |
| budget | 1M learner steps on 1M-transition dataset | 250k env steps, 1 update/step | **DIVERGENT** |
| horizon | 700 (umaze) | 700 | match |

Note: the vendored `contrastive/` snapshot in this repo declares `bc_coef` in
config.py but its learning.py never consumes it and uses `jnp.mean` for twins
-- it predates the upstream offline implementation. The port faithfully
copied a snapshot that lacks the offline objective.

## 3. Route content of the data the run actually trained on

`scripts/route_replay_audit.py` on the run's replay (360 eps, 252,000
transitions), `artifacts/ant_route_diagnosis/replay_route_stats.csv`:

| metric | value |
|---|---|
| episodes with BFS span 0 (local) | 74.2% |
| span 1 (short) | 20.8% |
| span >= 2 (route-level) | 5.0% (18/360; max span observed = 2) |
| detour-containing episodes | **0.0%** |
| corner passage | 3.6% |
| **sampled positive pairs** (real `TrajectoryBuffer.sample`, n=102,400) | BFS0 **96.14%**, BFS1 3.72%, BFS>=2 **0.137%**, detour-separated **0.0059%** (6/102,400) |

The critic received essentially zero route-level or detour-separated
positives. (A D4RL dataset was not available locally to audit -- item blocked
until the dataset file exists in the repo environment.)

## 4. Critic route-ranking probe

`scripts/critic_route_ranking.py`: 60 replay states with wall-blocked
commanded goals (BFS>=2, LOS crosses a wall, euclid-vs-geodesic direction
separation > 60 deg); 97 candidates/state (48 uniform + 48 real replay
actions + actor mode action), each rolled 3 env steps from an exact MuJoCo
restore and classified by MEASURED displacement.
`artifacts/ant_route_diagnosis/critic_route_ranking.json`:

| | best_235200 | final_252000 |
|---|---|---|
| mean q geodesic-correct | -19.82 | -20.10 |
| mean q euclidean-direct | -19.40 | -19.74 |
| mean q other | -21.11 | -20.93 |
| P(geodesic > euclid), per-state means | 0.40 | 0.47 |
| spearman(q, cos_euclid) | +0.143 | +0.167 |
| spearman(q, cos_geodesic) | +0.102 | +0.071 |
| actor action mean q | **-15.46** | **-14.93** |

Reading: the critic weakly prefers the **Euclidean shortcut** over the
geodesic route (both barely above unrelated actions), and the **actor's own
saturated action scores ~4-5 logits above ANY physically-classified real
action** -- the actor is climbing a region of action space the critic ranks
highly for reasons unrelated to measured displacement (OOD/saturation
exploitation, consistent with the earlier
CRITIC_GRADIENT_DRIVES_SATURATION gradient audits). Entropy collapse is an
actor-optimization/OOD-instability symptom, not an exploration budget issue
per se: the original offline recipe also runs alpha=0 and is stabilized by
the BC anchor, which this port lacks.

## 5. Verdict for the executed run

**MIXED_FAILURE**, dominated by:
1. **ROUTE_PAIRS_UNDERSAMPLED / ROUTE_DATA_MISSING** (in the training data
   the run actually consumed): 0.14% BFS>=2 positives, 0% detours.
2. **OFFLINE OBJECTIVE NOT IMPLEMENTED**: no dataset ingestion, no BC term,
   bc_coef=0.05 / twin_q / min-over-critics / random_goals=0.0 /
   (1024,1024)x16 offline settings all absent; vendored reference snapshot
   is stale relative to upstream master.
3. Critic route-ranking failure is CONFIRMED but is a downstream consequence
   of (1): with no route positives it prefers the Euclidean shortcut and
   scores the actor's OOD action above every physically-real action.

## 6. Single next experiment

Implement the faithful OFFLINE path and qualify it:
1. add `bc_coef` (+ BC blending exactly as upstream master) and
   min-over-critics to `crl/losses.py`; add `twin_q=True` support already
   present; add a dataset-ingestion entry point that fills TrajectoryBuffer
   from antmaze-umaze-v2 hdf5 episodes (split on `timeouts`);
2. zero-padded-XY eval goal wrapper (offline contract);
3. offline hparams: bc_coef=0.05, twin_q=True, random_goals=0.0,
   batch=1024, repr_dim=16, hidden=(1024,1024), alpha=0, 1M gradient steps;
4. re-run the Part-2/Part-4 audits on the dataset (expect BFS>=2 episode
   fraction >> 5% and detour-containing >> 0%) and re-probe critic ranking
   at 100k/500k/1M learner steps.

Requires downloading `antmaze-umaze-v2` (Ant_maze_u-maze_noisy_multistart_
True_multigoal_False_sparse_fixed.hdf5, ~0.5 GB, rail.eecs.berkeley.edu) --
not yet done (needs approval).
