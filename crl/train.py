"""Single-process train + eval loop for the contrastive RL port.

Replaces the Launchpad ``DistributedContrastive`` program with one process:
collect episodes -> store trajectories -> geometric-relabel sample -> SGD.

Usage:
    python -m crl.train --env_name point_FourRooms --max_number_of_steps 200000
    python -m crl.train --env_name fetch_reach --use_td --twin_q   # on Colab

Algorithm selection mirrors lp_contrastive.py:
    (default)            contrastive_nce
    --use_cpc            contrastive_cpc
    --use_td --twin_q    c_learning
    --use_td --twin_q --add_mc_to_td   nce+c_learning
    --use_gcbc           gcbc
"""
import argparse
import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crl import checkpoint as ckpt_mod
from crl import envs
from crl import losses as losses_mod
from crl import networks as networks_mod
from crl.config import Config
from crl.replay import TrajectoryBuffer


def _build_arg_parser():
  p = argparse.ArgumentParser(description='Contrastive RL (Acme-free port).')
  for f in dataclasses.fields(Config):
    name = '--' + f.name
    if f.type is bool or isinstance(f.default, bool):
      # Support --flag / --no-flag for booleans.
      p.add_argument(name, dest=f.name, action='store_true', default=None)
      p.add_argument('--no-' + f.name, dest=f.name, action='store_false')
    else:
      p.add_argument(name, dest=f.name, default=None)
  return p


def _apply_overrides(config, args):
  for f in dataclasses.fields(Config):
    val = getattr(args, f.name)
    if val is None:
      continue
    if isinstance(getattr(config, f.name), bool) or f.type is bool:
      setattr(config, f.name, bool(val))
    elif f.name in ('hidden_layer_sizes',):
      setattr(config, f.name, tuple(int(x) for x in str(val).split(',')))
    elif f.name == 'entropy_coefficient':
      setattr(config, f.name, None if val in ('None', 'none') else float(val))
    else:
      # Cast to the type of the current default.
      cur = getattr(config, f.name)
      caster = type(cur) if cur is not None else str
      setattr(config, f.name, caster(val))
  return config


def collect_episode(env, act_fn, params, key, random_action, action_dim,
                    np_rng):
  """Runs one fixed-length episode; returns (obs[L,.], act[L,.])."""
  L = env.max_episode_steps + 1
  obs = env.reset()
  # Keep the env's dtype: float32 for state envs, uint8 for image envs (the
  # networks normalize pixels by /255 internally, so uint8 feeds are fine).
  obs_buf = np.zeros((L, obs.shape[0]), obs.dtype)
  act_buf = np.zeros((L, action_dim), np.float32)
  for t in range(env.max_episode_steps):
    obs_buf[t] = obs
    if random_action:
      a = np_rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
    else:
      key, sub = jax.random.split(key)
      a = np.asarray(act_fn(params, jnp.asarray(obs[None]), sub)[0])
    act_buf[t] = a
    obs, _, _, _ = env.step(a)
  obs_buf[-1] = obs  # final observation; act_buf[-1] stays a dummy zero.
  return obs_buf, act_buf, key


def collect_block(envs, act_fn, params, key, random_action, action_dim,
                  np_rngs):
  """One fixed-length episode from EACH of N logical actors, in lockstep.

  Replicates the original multi-actor recipe's data flow in-process: each
  actor has its own env instance (distinct seed) and its own numpy RNG for
  the random warmup; the policy is evaluated as ONE batched forward pass per
  timestep (per-row independent sampling noise). Returns obs [N, L, D] and
  act [N, L, A]."""
  N = len(envs)
  L = envs[0].max_episode_steps + 1
  obs = np.stack([e.reset() for e in envs])
  obs_buf = np.zeros((N, L, obs.shape[1]), obs.dtype)
  act_buf = np.zeros((N, L, action_dim), np.float32)
  for t in range(L - 1):
    obs_buf[:, t] = obs
    if random_action:
      a = np.stack([r.uniform(-1.0, 1.0, action_dim).astype(np.float32)
                    for r in np_rngs])
    else:
      key, sub = jax.random.split(key)
      a = np.asarray(act_fn(params, jnp.asarray(obs), sub))
    act_buf[:, t] = a
    obs = np.stack([envs[i].step(a[i])[0] for i in range(N)])
  obs_buf[:, -1] = obs
  return obs_buf, act_buf, key


def evaluate(env, eval_act_fn, params, episodes, np_rng, action_dim,
             obs_dim, start_index, end_index, goal_indices=None):
  """Greedy rollouts. Returns (success, final_dist, min_dist), each a mean.

  success  = any reward==1 within an episode.
  *_dist   = L2 distance ||achieved_goal - desired_goal||, where achieved_goal
             is obs_to_goal(state) (state[start:end]) and desired_goal is the
             goal half of the observation -- the same quantity the reward uses.
  With ``goal_indices`` (rich-goal ablation) the distance is XY-ONLY (the
  first two goal coords), so metrics stay comparable across goal arms.
  """
  successes, final_dists, min_dists = [], [], []
  for _ in range(episodes):
    obs = env.reset()
    hit = 0.0
    dists = []
    for _ in range(env.max_episode_steps):
      a = np.asarray(eval_act_fn(params, jnp.asarray(obs[None]))[0])
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      # float cast: uint8 (image obs) subtraction would wrap around.
      state = obs[:obs_dim].astype(np.float32, copy=False)
      goal = obs[obs_dim:].astype(np.float32, copy=False)
      if goal_indices is not None:
        ag, goal = state[:2], goal[:2]           # XY primary metric
      else:
        ag = state[start_index:] if end_index == -1 else state[start_index:end_index]
      dists.append(float(np.linalg.norm(ag - goal)))
    successes.append(hit)
    final_dists.append(dists[-1])
    min_dists.append(min(dists))
  return (float(np.mean(successes)), float(np.mean(final_dists)),
          float(np.mean(min_dists)))


def evaluate_push_physical(env, eval_act_fn, params, episodes, np_rng):
  """Greedy rollouts for a FetchPush IMAGE env, scored on PHYSICAL coordinates.

  Returns (success, final_dist, min_dist) where success is the sparse env
  reward (object within 0.05 m of the goal, from sim state) and the distances
  are ``||object_xyz - desired_goal_xyz||`` read from the simulator -- NOT the
  flattened image-L2 that ``evaluate`` would compute for image observations.
  ``env`` must be a ``crl.envs.FetchEnv`` (exposes ``_env.unwrapped._get_obs``).
  """
  u = env._env.unwrapped
  successes, final_dists, min_dists = [], [], []
  for _ in range(episodes):
    env.reset()
    desired = np.asarray(env._desired, dtype=np.float32)
    obs_img = np.concatenate([env._frame(), env._goal_img])
    hit = 0.0
    dists = []
    for _ in range(env.max_episode_steps):
      a = np.asarray(eval_act_fn(params, jnp.asarray(obs_img[None]))[0])
      obs_img, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      d = u._get_obs()
      obj = np.asarray(d['achieved_goal'], dtype=np.float32)
      dists.append(float(np.linalg.norm(obj - desired)))
    successes.append(hit)
    final_dists.append(dists[-1])
    min_dists.append(min(dists))
  return (float(np.mean(successes)), float(np.mean(final_dists)),
          float(np.mean(min_dists)))


def train(config: Config):
  print('Config:', config)
  key = jax.random.PRNGKey(config.seed)
  np_rng = np.random.default_rng(config.seed)

  offline = bool(config.offline_dataset)
  if offline:
    # --- STRICT OFFLINE MODE ---------------------------------------------
    # No TRAINING environment is ever created: `env` stays None and the
    # collection lists are empty, so collect_episode/collect_block cannot run.
    # The eval env is the ONLY env; it fills the config dims and is stepped
    # solely inside evaluate(). Online-only knobs are rejected / disabled.
    if int(config.num_actors) > 1:
      raise ValueError(
          f'offline mode rejects num_actors={config.num_actors} (>1): there '
          'is no collection to parallelize.')
    eval_env = envs.make_env(config.env_name, config, seed=config.seed + 10_000)
    env = None
    N_ACT = 1
    coll_envs, coll_rngs = [], []
    # random_steps / min_replay_size are meaningless with a frozen dataset.
    if config.random_steps:
      print(f'  [offline] disabling random_steps={config.random_steps} -> 0')
    config.random_steps = 0
  else:
    # --- Env (fills obs/goal/action dims into config) ---
    env = envs.make_env(config.env_name, config, seed=config.seed)
    eval_env = envs.make_env(config.env_name, config, seed=config.seed + 10_000)
    # Additional logical actors (original multi-actor recipe): distinct env
    # seeds + distinct warmup RNG streams per actor; actor 0 keeps the legacy
    # seeding so num_actors=1 is byte-identical to previous runs.
    N_ACT = max(1, int(config.num_actors))
    coll_envs = [env] + [envs.make_env(config.env_name, Config(
        env_name=config.env_name), seed=config.seed + 7919 * i)
        for i in range(1, N_ACT)]
    coll_rngs = [np.random.default_rng(config.seed + 104729 * i)
                 for i in range(N_ACT)]
  print(f'obs_dim={config.obs_dim} goal_dim={config.goal_dim} '
        f'action_dim={config.action_dim} '
        f'max_episode_steps={config.max_episode_steps} '
        f'goal_slice=[{config.start_index}:{config.end_index}]')

  # --- Networks + learner ---
  nets = networks_mod.make_networks(
      obs_dim=config.obs_dim, goal_dim=config.goal_dim,
      action_dim=config.action_dim, repr_dim=int(config.repr_dim),
      repr_norm=config.repr_norm, repr_norm_temp=config.repr_norm_temp,
      hidden_layer_sizes=config.hidden_layer_sizes,
      twin_q=config.twin_q, use_image_obs=config.use_image_obs,
      use_layer_norm=config.use_layer_norm)

  policy_optimizer = optax.adam(config.actor_learning_rate, eps=1e-7)
  q_optimizer = optax.adam(config.learning_rate, eps=1e-7)

  start, end = config.start_index, config.end_index
  gidx = (None if config.goal_indices is None
          else jnp.asarray(config.goal_indices))
  def obs_to_goal(states):
    if gidx is not None:
      return states[:, gidx]
    return states[:, start:] if end == -1 else states[:, start:end]

  init_state, update_step = losses_mod.build_learner(
      nets, config, obs_to_goal, policy_optimizer, q_optimizer)

  key, key_init = jax.random.split(key)
  state = init_state(key_init)

  # Optionally resume params/optimizer from a previous run (buffer refills).
  start_step = 0
  if config.resume and config.ckpt_dir:
    latest = os.path.join(config.ckpt_dir, 'latest.pkl')
    if os.path.exists(latest):
      start_step, state = ckpt_mod.load_checkpoint(latest)
      print(f'Resumed from {latest} at step {start_step}.')
    else:
      print(f'--resume set but no checkpoint at {latest}; starting fresh.')

  # --- Jitted helpers ---
  def _act(params, obs, k):
    return nets.sample(nets.policy_network.apply(params, obs), k)

  def _eval_act(params, obs):
    return nets.sample_eval(nets.policy_network.apply(params, obs), None)

  act_fn = jax.jit(_act) if config.jit else _act
  eval_act_fn = jax.jit(_eval_act) if config.jit else _eval_act

  def _multi_update(state, trans_G):
    # trans_G: Transition with leading dim G; scan does G sequential updates.
    state, metrics = jax.lax.scan(update_step, state, trans_G)
    metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
    return state, metrics
  multi_update = jax.jit(_multi_update) if config.jit else _multi_update

  # --- Replay ---
  min_replay = config.min_replay_size
  if offline:
    # Load the fixed dataset ONCE, sized exactly to it, and FREEZE it. Then run
    # the full static + buffer audit BEFORE any gradient step; a single failed
    # gate aborts training. See crl/offline_audit.py for the gate definitions.
    from crl import offline_audit
    buffer, off_fp = offline_audit.build_offline_buffer(
        config.offline_dataset, config)
    passed, gates, audit_report = offline_audit.run_static_audit(
        config.offline_dataset, config, buffer=buffer)
    print('OFFLINE AUDIT (pre-training gates):')
    for gname, ok in gates.items():
      print(f'  {"PASS" if ok else "FAIL"}  {gname}')
    print(f'  dataset={config.offline_dataset}')
    print(f'  sha256={off_fp["sha256"][:16]}...  eps={off_fp["n_episodes"]}  '
          f'trans={off_fp["n_transitions"]}  obs={off_fp["obs_shape"]}  '
          f'act={off_fp["act_shape"]}  ep_len<=[{off_fp["ep_lengths_min"]},'
          f'{off_fp["ep_lengths_max"]}]')
    if off_fp['keys']['audit']:
      print(f'  audit-only fields kept OUT of the learner: '
            f'{off_fp["keys"]["audit"]}')
    if not passed:
      raise RuntimeError(
          f'OFFLINE AUDIT FAILED (gates={gates}); refusing to train.')
    if config.min_replay_size:
      print(f'  [offline] disabling min_replay_size={config.min_replay_size} '
            '-> 0 (the frozen dataset IS the buffer)')
    min_replay = 0   # the frozen dataset IS the buffer; learn from step one.

    # Resume must use the identical dataset; a fresh run records its hash.
    if config.ckpt_dir:
      if config.resume:
        same, recorded = offline_audit.require_same_dataset_hash(
            config.ckpt_dir, off_fp['sha256'])
        if not same:
          raise RuntimeError(
              'OFFLINE RESUME dataset MISMATCH: this checkpoint was trained on '
              f'{recorded} but the current dataset hashes {off_fp["sha256"]}.')
      else:
        offline_audit.record_dataset_hash(
            config.ckpt_dir, off_fp['sha256'], off_fp['meta'])

    # Causal Manski positives: wrap the frozen buffer so sample()'s goal comes
    # from the Thm-2 d_lb walk (audit surface + content delegate to the frozen
    # base; see crl/manski.py). p_override=1.0 is the matched baseline arm.
    if config.manski_positives:
      from crl import manski as manski_mod
      p_ov = (None if config.manski_p_override < 0
              else float(config.manski_p_override))
      hz = (tuple(eval_env.SWAMP_CELLS)
            if (config.manski_hazard or config.manski_reachable) else ())
      buffer = manski_mod.build_positive_buffer(
          buffer, config.manski_table, eval_env._walls, eval_env.GOAL,
          config.discount, seed=config.seed + 31337, p_override=p_ov,
          hazard_cells=hz, reachable_n=config.manski_reachable)
      print(f'  [manski] d_lb positives: table={config.manski_table} '
            f'gamma={config.discount} p_override={p_ov} hazard={hz} '
            f'reachable={config.manski_reachable}')

    # Runtime immutability watchdog (checked at every eval): the collection env
    # does not exist (env is None) and the buffer is frozen, so collection is
    # structurally impossible; here we also pin the content hash and count the
    # eval env's steps so eval cannot mutate or over-step.
    offline_frozen_sha = buffer.content_sha256()
    offline_frozen_ptr = (buffer._num_eps, buffer.ready_steps)
    offline_eval_steps = {'n': 0, 'evals': 0}
    _real_eval_step = eval_env.step
    def _counted_eval_step(a):
      offline_eval_steps['n'] += 1
      return _real_eval_step(a)
    eval_env.step = _counted_eval_step
  else:
    # Online: growable ring sized by max_replay_size.
    buffer = TrajectoryBuffer(
        capacity_steps=config.max_replay_size,
        ep_len_obs=config.max_episode_steps + 1,
        full_obs_dim=config.obs_dim + config.goal_dim,
        action_dim=config.action_dim, obs_dim=config.obs_dim,
        start_index=config.start_index, end_index=config.end_index,
        discount=config.discount, seed=config.seed,
        goal_indices=config.goal_indices,
        obs_dtype=np.uint8 if config.use_image_obs else np.float32)

  G = max(1, config.num_sgd_steps_per_step)
  B = config.batch_size

  def sample_G():
    batches = [buffer.sample(B) for _ in range(G)]
    stacked = losses_mod.Transition(*[
        jnp.asarray(np.stack([getattr(b, field) for b in batches], axis=0))
        for field in losses_mod.Transition._fields])
    return stacked

  # --- Main loop ---
  env_steps = start_step
  last_log = start_step
  last_eval = start_step
  last_ckpt = start_step
  t0 = time.time()
  metrics_history = []
  best_success = -1.0
  ckpt_every = config.ckpt_every_steps or config.eval_every_steps

  # Milestone checkpoints (init/early/mid/final) + optional TensorBoard mirror.
  saved_phases = set()
  if config.ckpt_dir:
    ckpt_mod.save_named(config.ckpt_dir, 'init', env_steps, state)
    saved_phases.add('init')
  writer = None
  if config.tensorboard:
    tb_dir = os.path.join(config.ckpt_dir or '.', 'tb')
    try:
      # Use tensorboardX (pure-python). Do NOT use torch.utils.tensorboard:
      # importing torch alongside JAX on the same GPU can hard-crash the kernel.
      from tensorboardX import SummaryWriter
      writer = SummaryWriter(tb_dir)
    except Exception as ex:  # pylint: disable=broad-except
      print(f'  [tensorboard requested but tensorboardX unavailable ({ex}); '
            'skipping TB. `pip install tensorboardX`]')

  # Maze-scalar hooks (guarded so they never block training).
  is_point_maze = config.env_name.startswith('point_')
  is_antmaze = config.env_name.startswith('antmaze_')
  def _maze_scalars():
    from crl import report_maze
    def _mp(s, g, memo):
      obs = np.concatenate([s, g]).astype(np.float32)
      return np.asarray(eval_act_fn(state.policy_params, jnp.asarray(obs[None]))[0])
    a = report_maze.eval_scalars(eval_env, _mp, episodes=min(config.eval_episodes, 30))
    keep = ('success@2.0', 'success@1.0', 'success@0.5', 'spl', 'collisions',
            'wp_completion')
    return {'maze_' + k: float(v) for k, v in a.items()
            if k in keep and v is not None}
  def _antmaze_scalars():
    from crl import report_antmaze
    def _p(flat):
      return np.asarray(eval_act_fn(state.policy_params, jnp.asarray(flat[None]))[0])
    eps = [report_antmaze.rollout(eval_env, _p)
           for _ in range(min(config.eval_episodes, 3))]
    a = report_antmaze.aggregate(eps)
    return {'ant_action_saturation': a['action_saturation_mean'],
            'ant_torso_height': a['mean_z_mean'],
            'ant_fall_fraction': a['fell_mean'],
            'ant_goal_velocity': a['goal_directed_velocity_mean']}

  # Restore the replay snapshot for staged/multi-stage runs (exact resume).
  learner_updates = 0
  replay_path = (os.path.join(config.ckpt_dir, 'replay.npz')
                 if config.ckpt_dir else None)
  if config.save_replay and config.resume and replay_path and \
     os.path.exists(replay_path):
    n = buffer.load(replay_path)
    print(f'Restored replay snapshot: {n} episodes '
          f'({buffer.ready_steps} transitions).')

  while env_steps < config.max_number_of_steps:
    random_action = env_steps < config.random_steps
    key, key_collect = jax.random.split(key)
    if offline:
      pass                       # frozen dataset: no collection, ever.
    elif N_ACT == 1:
      assert env is not None, 'online collection requires an env'
      obs_buf, act_buf, _ = collect_episode(
          env, act_fn, state.policy_params, key_collect, random_action,
          config.action_dim, np_rng)
      buffer.add_episode(obs_buf, act_buf)
    else:
      obs_blk, act_blk, _ = collect_block(
          coll_envs, act_fn, state.policy_params, key_collect, random_action,
          config.action_dim, coll_rngs)
      for i in range(N_ACT):
        buffer.add_episode(obs_blk[i], act_blk[i])
    env_steps += N_ACT * config.max_episode_steps

    # Learn. Ratio preserved: 1 update-batch per TOTAL env step, so the
    # learner budget scales with N_ACT episodes per block, NOT per actor.
    metrics = {}
    if buffer.ready_steps >= min_replay:
      learner_steps = max(1, config.updates_per_step *
                          (N_ACT * config.max_episode_steps) // G)
      for _ in range(learner_steps):
        state, metrics = multi_update(state, sample_G())
      learner_updates += learner_steps * G

    # Numerical guard (opt-in): abort on non-finite / exploding learner state.
    if config.guard_abort and metrics:
      reason = None
      m = {k: float(v) for k, v in metrics.items()}
      for k in ('actor_loss', 'critic_loss', 'logits_pos', 'logits_neg',
                'alpha', 'alpha_loss'):
        if k in m and not np.isfinite(m[k]):
          reason = f'non-finite {k}={m[k]}'
          break
      if reason is None and abs(m.get('actor_loss', 0.0)) > config.guard_actor_loss_max:
        reason = (f'|actor_loss|={abs(m["actor_loss"]):.3g} > '
                  f'{config.guard_actor_loss_max:.3g}')
      if reason is None:
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in
                     jax.tree_util.tree_leaves((state.policy_params,
                                                state.q_params)))
        if not finite:
          reason = 'non-finite policy/critic parameters'
      if reason is not None:
        print(f'GUARD_ABORT at step {env_steps}: {reason}', flush=True)
        metrics_history.append({'step': int(env_steps), 'guard_abort': reason})
        if config.ckpt_dir:
          ckpt_mod.save_named(config.ckpt_dir, 'abort', env_steps, state)
          best_success = ckpt_mod.save_checkpoint(
              config.ckpt_dir, env_steps, state, metrics_history, None,
              best_success)
        break

    # Logging.
    if env_steps - last_log >= config.log_every_steps:
      sps = (env_steps) / (time.time() - t0)
      msg = f'[step {env_steps:>8}] sps={sps:6.0f}'
      if metrics:
        m = {k: float(v) for k, v in metrics.items()}
        msg += (f' critic={m.get("critic_loss", 0):.3f}'
                f' actor={m.get("actor_loss", 0):.3f}'
                f' cat_acc={m.get("categorical_accuracy", 0):.3f}')
        if 'bc_nll' in m:
          msg += f' bc_nll={m["bc_nll"]:.3f}'
      elif buffer.ready_steps < min_replay:
        msg += f' (filling buffer {buffer.ready_steps}/{min_replay})'
      print(msg, flush=True)
      last_log = env_steps

    # Eval.
    if env_steps - last_eval >= config.eval_every_steps:
      # Snapshot the eval-env step counter BEFORE evaluate() so the offline
      # contract can check evaluate()'s OWN consumption via a delta -- the
      # read-only maze/antmaze scalar rollouts below also legitimately step the
      # eval env (never touching replay), so an absolute count would be wrong.
      eval_n_before = offline_eval_steps['n'] if offline else 0
      if config.physical_eval_push:
        # FetchPush image run: score on PHYSICAL object-goal coordinates, never
        # flattened image-L2 (which is meaningless as a control metric).
        succ, fdist, mdist = evaluate_push_physical(
            eval_env, eval_act_fn, state.policy_params, config.eval_episodes,
            np_rng)
        dist_tag = 'phys_dist'
      else:
        succ, fdist, mdist = evaluate(
            eval_env, eval_act_fn, state.policy_params, config.eval_episodes,
            np_rng, config.action_dim, config.obs_dim, config.start_index,
            config.end_index, config.goal_indices)
        dist_tag = 'final_dist'
      print(f'  >> EVAL step {env_steps}: success_rate={succ:.3f} '
            f'{dist_tag}={fdist:.3f} min_dist={mdist:.3f}', flush=True)
      last_eval = env_steps
      if offline:
        # HARD offline contract (see the audit block above): the frozen dataset
        # is immutable (pointers + full content hash unchanged), and evaluate()
        # consumed EXACTLY eval_episodes*max_episode_steps eval-env steps this
        # eval. There is no collection env (env is None) and the buffer is
        # frozen, so collection is structurally impossible.
        assert (buffer._num_eps, buffer.ready_steps) == offline_frozen_ptr, \
            'offline contract violated: replay pointers changed'
        assert buffer.content_sha256() == offline_frozen_sha, \
            'offline contract violated: replay CONTENT changed'
        offline_eval_steps['evals'] += 1
        consumed = offline_eval_steps['n'] - eval_n_before
        expected = config.eval_episodes * eval_env.max_episode_steps
        assert consumed == expected, (
            'offline contract violated: evaluate() eval-env steps '
            f'{consumed} != {expected} (per-eval)')
      rec = {'step': int(env_steps), 'success': float(succ),
             'final_dist': float(fdist), 'min_dist': float(mdist),
             'learner_updates': int(learner_updates),
             'num_actors': 0 if offline else N_ACT,
             'per_actor_steps': 0 if offline else int(env_steps // N_ACT)}
      if config.physical_eval_push:
        # Unambiguous physical aliases (final_dist/min_dist above are already
        # physical in this mode; image-L2 is never logged).
        rec['physical_success_rate'] = float(succ)
        rec['physical_final_object_goal_distance'] = float(fdist)
        rec['physical_min_object_goal_distance'] = float(mdist)
      rec.update({k: float(v) for k, v in metrics.items()})
      if is_point_maze:
        try:
          rec.update(_maze_scalars())
        except Exception as ex:  # pylint: disable=broad-except
          print('  [maze scalars skipped]', ex)
      if is_antmaze:
        try:
          rec.update(_antmaze_scalars())
        except Exception as ex:  # pylint: disable=broad-except
          print('  [antmaze scalars skipped]', ex)
      # NaN guard: surface non-finite metrics immediately.
      if not np.all(np.isfinite([v for v in rec.values()
                                 if isinstance(v, (int, float))])):
        print(f'  [WARN] non-finite metric at step {env_steps}', flush=True)
      metrics_history.append(rec)
      if writer is not None:
        for k, v in rec.items():
          if isinstance(v, (int, float)):
            writer.add_scalar(k.replace('@', '_'), float(v), env_steps)

      # Periodic checkpoint to (Drive) ckpt_dir -- not just once at the end.
      if config.ckpt_dir and env_steps - last_ckpt >= ckpt_every:
        best_success = ckpt_mod.save_checkpoint(
            config.ckpt_dir, env_steps, state, metrics_history, succ,
            best_success, strict=config.best_strict_improvement)
        last_ckpt = env_steps

      # Milestone checkpoints at 25% / 50% of the budget.
      if config.ckpt_dir:
        for nm, frac in (('early', 0.25), ('mid', 0.5)):
          if nm not in saved_phases and env_steps >= frac * config.max_number_of_steps:
            ckpt_mod.save_named(config.ckpt_dir, nm, env_steps, state)
            saved_phases.add(nm)

      # Explicit step-numbered milestones (e.g. 10k/20k/30k/50k/70k) saved as
      # <step>.pkl the first eval at or past each target.
      if config.ckpt_dir and config.ckpt_milestone_steps:
        for ms in config.ckpt_milestone_steps:
          tag = f'ckpt_{int(ms)}'
          if tag not in saved_phases and env_steps >= ms:
            ckpt_mod.save_named(config.ckpt_dir, str(int(ms)), env_steps, state)
            saved_phases.add(tag)

  # Final offline-contract check (content hash) before the last save.
  if offline:
    assert buffer.content_sha256() == offline_frozen_sha, \
        'offline contract violated: replay CONTENT changed by end of run'
  # Final save (also runs after a guard abort's break).
  if config.ckpt_dir:
    ckpt_mod.save_named(config.ckpt_dir, 'final', env_steps, state)
    ckpt_mod.save_checkpoint(config.ckpt_dir, env_steps, state,
                             metrics_history, None, best_success)
  if config.save_replay and replay_path:
    buffer.save(replay_path)
    print(f'Replay snapshot saved: {buffer.ready_steps} transitions '
          f'-> {replay_path}', flush=True)
  if writer is not None:
    writer.close()
  return state


def main():
  args = _build_arg_parser().parse_args()
  config = _apply_overrides(Config(), args)
  train(config)


if __name__ == '__main__':
  main()
