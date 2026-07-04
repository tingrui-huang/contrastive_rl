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
  obs_buf = np.zeros((L, obs.shape[0]), np.float32)
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


def evaluate(env, eval_act_fn, params, episodes, np_rng, action_dim):
  """Greedy rollouts; success = any reward==1 within an episode."""
  successes = []
  for _ in range(episodes):
    obs = env.reset()
    hit = 0.0
    for _ in range(env.max_episode_steps):
      a = np.asarray(eval_act_fn(params, jnp.asarray(obs[None]))[0])
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
    successes.append(hit)
  return float(np.mean(successes))


def train(config: Config):
  print('Config:', config)
  key = jax.random.PRNGKey(config.seed)
  np_rng = np.random.default_rng(config.seed)

  # --- Env (fills obs/goal/action dims into config) ---
  env = envs.make_env(config.env_name, config, seed=config.seed)
  eval_env = envs.make_env(config.env_name, config, seed=config.seed + 10_000)
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
      twin_q=config.twin_q, use_image_obs=config.use_image_obs)

  policy_optimizer = optax.adam(config.actor_learning_rate, eps=1e-7)
  q_optimizer = optax.adam(config.learning_rate, eps=1e-7)

  start, end = config.start_index, config.end_index
  def obs_to_goal(states):
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
  buffer = TrajectoryBuffer(
      capacity_steps=config.max_replay_size,
      ep_len_obs=config.max_episode_steps + 1,
      full_obs_dim=config.obs_dim + config.goal_dim,
      action_dim=config.action_dim, obs_dim=config.obs_dim,
      start_index=config.start_index, end_index=config.end_index,
      discount=config.discount, seed=config.seed)

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

  while env_steps < config.max_number_of_steps:
    random_action = env_steps < config.random_steps
    key, key_collect = jax.random.split(key)
    obs_buf, act_buf, _ = collect_episode(
        env, act_fn, state.policy_params, key_collect, random_action,
        config.action_dim, np_rng)
    buffer.add_episode(obs_buf, act_buf)
    env_steps += config.max_episode_steps

    # Learn.
    metrics = {}
    if buffer.ready_steps >= config.min_replay_size:
      learner_steps = max(1, config.updates_per_step *
                          (config.max_episode_steps) // G)
      for _ in range(learner_steps):
        state, metrics = multi_update(state, sample_G())

    # Logging.
    if env_steps - last_log >= config.log_every_steps:
      sps = (env_steps) / (time.time() - t0)
      msg = f'[step {env_steps:>8}] sps={sps:6.0f}'
      if metrics:
        m = {k: float(v) for k, v in metrics.items()}
        msg += (f' critic={m.get("critic_loss", 0):.3f}'
                f' actor={m.get("actor_loss", 0):.3f}'
                f' cat_acc={m.get("categorical_accuracy", 0):.3f}')
      elif buffer.ready_steps < config.min_replay_size:
        msg += f' (filling buffer {buffer.ready_steps}/{config.min_replay_size})'
      print(msg, flush=True)
      last_log = env_steps

    # Eval.
    if env_steps - last_eval >= config.eval_every_steps:
      succ = evaluate(eval_env, eval_act_fn, state.policy_params,
                      config.eval_episodes, np_rng, config.action_dim)
      print(f'  >> EVAL step {env_steps}: success_rate={succ:.3f}', flush=True)
      last_eval = env_steps
      rec = {'step': int(env_steps), 'success': float(succ)}
      rec.update({k: float(v) for k, v in metrics.items()})
      metrics_history.append(rec)

      # Periodic checkpoint to (Drive) ckpt_dir -- not just once at the end.
      if config.ckpt_dir and env_steps - last_ckpt >= ckpt_every:
        best_success = ckpt_mod.save_checkpoint(
            config.ckpt_dir, env_steps, state, metrics_history, succ,
            best_success)
        last_ckpt = env_steps

  # Final save.
  if config.ckpt_dir:
    ckpt_mod.save_checkpoint(config.ckpt_dir, env_steps, state,
                             metrics_history, None, best_success)
  return state


def main():
  args = _build_arg_parser().parse_args()
  config = _apply_overrides(Config(), args)
  train(config)


if __name__ == '__main__':
  main()
