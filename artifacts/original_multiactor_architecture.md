# Original 4-actor architecture (as shipped, 2022) and the port's mapping

Sources: `contrastive/agents.py` (DistributedContrastive), `contrastive/builder.py`
(make_actor / make_replay_tables / make_dataset_iterator flatten_fn),
`contrastive/utils.py` (InitiallyRandomActor), `lp_contrastive.py`, acme
`distributed_layout` semantics.

## What the original actually runs

| aspect | original | port mapping (this repo) |
|---|---|---|
| actor execution | **separate Launchpad PROCESSES** (`local_mp`; threads under `local_mt`), each with its own env instance built from `environment_factory(actor_seed)` and its own `GenericActor`/`InitiallyRandomActor` | **4 logical actors in one process**: 4 independent env instances (distinct seeds/RNGs) stepped in lockstep with a single BATCHED policy forward pass per step. Deviation forced by Colab CPU count (2 vCPU on T4 runtimes); benchmarked in-notebook. |
| parameter sync | each actor holds a `VariableClient(variable_source, 'policy', device='cpu')` with acme's default update period — actors poll the learner ASYNCHRONOUSLY (non-blocking) and can act on slightly stale params | in-process actors always use the CURRENT params (staleness 0). Deviation: fresher-than-original policies; direction of bias is toward the on-policy end of the original's staleness spectrum. |
| replay | central Reverb server; `EpisodeAdder` inserts ONE ITEM PER EPISODE; all actors insert into the same table | one `TrajectoryBuffer`; each actor's episode is added separately every block (identical semantics: whole fixed-length trajectories). |
| learner schedule | `SampleToInsertRatio(samples_per_insert=256)` on trajectory items: 256 sampled trajectories per inserted trajectory; each sampled trajectory expands to `T-1` transitions in `flatten_fn`, batched at 256 => effectively **1 gradient batch (256) per TOTAL inserted env step**, independent of actor count (the rate limiter throttles the learner to the aggregate insert rate) | learner performs `updates_per_step * (num_actors * max_episode_steps) // G` learner-steps of `G` batches after each block => **1 batch per total env step**, identical ratio. NOT multiplied by 4. |
| env-step accounting | `max_number_of_steps` in `DistributedLayout` terminates on the AGGREGATE actor-step counter (sum across the 4 actors) | `env_steps += num_actors * max_episode_steps` per block; per-actor counters logged. 250k = 250k TOTAL (~62.5k per actor). |
| warmup | `InitiallyRandomActor`: uniform random until the first learner update reaches the actor | `random_steps` in TOTAL steps (uniform per actor until the aggregate warmup budget is consumed). |
| evaluator | separate evaluator process with deterministic policy | in-process eval every `eval_every_steps` total steps (unchanged from port). |
| RNG | per-actor `random_key` split from the root seed by the layout | per-actor `np.random.default_rng(seed + 7919*i)` for warmup + one jax key per block step batch (row-wise independent noise). |

## Ratio guarantee (the item the spec flags)

Original: the Reverb rate limiter enforces sampled-items : inserted-items =
256 : 1 on trajectories; one sampled trajectory yields T-1 relabeled
transitions, consumed in batches of 256, so per inserted episode of length T
the learner consumes ~T-1 batches — i.e. one 256-batch per inserted env step,
summed over ALL actors. Port: one 256-batch per total env step by
construction. **Four actors quadruple the data rate per wall-clock, not the
updates-per-env-step ratio, and the 250k budget is a TOTAL across actors.**

## Deviations (all forced or trivial, none algorithmic)

1. In-process interleaved actors instead of 4 OS processes (Colab CPU bound;
   benchmark cell quantifies 1/2/4-actor throughput before the run).
2. Zero parameter staleness instead of asynchronous polling.
3. numpy collection RNG streams are reseeded deterministically on resume (the
   learner's jax key IS restored from the checkpoint); replay content is
   snapshotted at stage boundaries and restored exactly.
