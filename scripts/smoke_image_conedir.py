"""Smoke qualification for fetch_push_image_conedir (image obs, same dynamics).

Gates (all must PASS before the Colab run):
  1. WIRING      obs = uint8 [2*64*64*3]; config dims/use_image_obs auto-set.
  2. NO_LEAK     moving u.goal does NOT change the rendered frame (the target
                 site is invisible), and the mocap markers are hidden.
  3. GOAL_IMAGE  the goal frame differs from the reset frame (object displaced
                 to the goal) and the episode's desired_goal matches the goal
                 the image was rendered for.
  4. ORACLE      the scripted state-based push controller (report_push's
                 lift->over->descend->push) succeeds on the image env --
                 dynamics unchanged; also reports the random floor.
  5. TRAIN_STEP  uint8 buffer + conv networks: one NCE update is finite.

Artifacts -> artifacts/push_image_probe/ (PNGs + smoke_image_conedir.json).

Run:  python -m scripts.smoke_image_conedir
"""
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config

OUT = os.path.join('artifacts', 'push_image_probe')
IMG = 64 * 64 * 3


def _png(path, flat):
  import imageio
  imageio.imwrite(path, np.asarray(flat, np.uint8).reshape(64, 64, 3))


def main():
  os.makedirs(OUT, exist_ok=True)
  results = {}

  cfg = Config(env_name='fetch_push_image_conedir')
  env = envs_mod.make_env(cfg.env_name, cfg, seed=0)
  u = env._env.unwrapped

  # ---- gate 1: wiring ----
  obs = env.reset()
  g1 = (obs.dtype == np.uint8 and obs.shape == (2 * IMG,)
        and cfg.obs_dim == IMG and cfg.goal_dim == IMG
        and cfg.start_index == 0 and cfg.end_index == -1
        and cfg.use_image_obs)
  frame_std = float(np.std(obs[:IMG].astype(np.float32)))
  g1 = g1 and frame_std > 5.0          # not a blank/black frame
  results['gate1_wiring'] = {
      'pass': bool(g1), 'dtype': str(obs.dtype), 'shape': list(obs.shape),
      'obs_dim': cfg.obs_dim, 'goal_dim': cfg.goal_dim,
      'use_image_obs': cfg.use_image_obs, 'frame_std': frame_std}
  print(f"gate1 WIRING      {'PASS' if g1 else 'FAIL'} "
        f'dtype={obs.dtype} shape={obs.shape} frame_std={frame_std:.1f}')
  _png(os.path.join(OUT, 'reset_frame.png'), obs[:IMG])
  _png(os.path.join(OUT, 'goal_frame.png'), obs[IMG:])

  # ---- gate 2: goal must not leak into the frame ----
  f_a = env._frame()
  saved_goal = np.array(u.goal, dtype=float).copy()
  u.goal = saved_goal + np.array([0.05, -0.05, 0.0])   # move the target site
  f_b = env._frame()
  u.goal = saved_goal
  leak_pixels = int(np.sum(f_a != f_b))
  mj = u._mujoco
  sid = mj.mj_name2id(u.model, mj.mjtObj.mjOBJ_SITE, 'target0')
  site_alpha = float(u.model.site_rgba[sid, 3]) if sid >= 0 else -1.0
  g2 = leak_pixels == 0 and site_alpha == 0.0
  results['gate2_no_leak'] = {'pass': bool(g2), 'leak_pixels': leak_pixels,
                              'target0_alpha': site_alpha}
  print(f"gate2 NO_LEAK     {'PASS' if g2 else 'FAIL'} "
        f'leak_pixels={leak_pixels} target0_alpha={site_alpha}')

  # ---- gate 3: goal image content + goal consistency ----
  diffs, goal_match = [], []
  for k in range(3):
    obs = env.reset()
    cur, goal_img = obs[:IMG].astype(np.float32), obs[IMG:].astype(np.float32)
    diffs.append(float(np.mean(np.abs(cur - goal_img))))
    # the episode's desired goal is what _render_goal_image placed the object at
    goal_match.append(float(np.linalg.norm(env._desired - u.goal)))
    if k == 0:
      _png(os.path.join(OUT, 'pair_reset.png'), obs[:IMG])
      _png(os.path.join(OUT, 'pair_goal.png'), obs[IMG:])
  g3 = min(diffs) > 0.5 and max(goal_match) < 1e-9
  results['gate3_goal_image'] = {'pass': bool(g3),
                                 'mean_abs_pixel_diff': diffs,
                                 'desired_vs_ugoal': goal_match}
  print(f"gate3 GOAL_IMAGE  {'PASS' if g3 else 'FAIL'} "
        f'pixel_diff={["%.2f" % d for d in diffs]} '
        f'goal_match={["%.1e" % m for m in goal_match]}')

  # ---- gate 4: oracle ceiling / random floor (dynamics unchanged) ----
  from crl.report_push import _oracle_action
  rng = np.random.default_rng(0)

  def _rollout(policy, n):
    succ = []
    for _ in range(n):
      env.reset()
      hit = 0.0
      for _ in range(env.max_episode_steps):
        o = u._get_obs()
        vec = np.concatenate([o['observation'],
                              o['desired_goal']]).astype(np.float32)
        a = policy(vec)
        _, r, _, _ = env.step(a)
        hit = max(hit, float(r))
      succ.append(hit)
    return float(np.mean(succ))

  oracle = _rollout(_oracle_action, 10)
  rand = _rollout(
      lambda _: rng.uniform(-1, 1, size=4).astype(np.float32), 5)
  g4 = oracle >= 0.8
  results['gate4_oracle'] = {'pass': bool(g4), 'oracle_success': oracle,
                             'random_success': rand}
  print(f"gate4 ORACLE      {'PASS' if g4 else 'FAIL'} "
        f'oracle={oracle:.2f} random={rand:.2f}')

  # mid-push frame for the eyeball check
  env.reset()
  for _ in range(25):
    o = u._get_obs()
    vec = np.concatenate([o['observation'], o['desired_goal']]).astype(np.float32)
    obs, _, _, _ = env.step(_oracle_action(vec))
  _png(os.path.join(OUT, 'mid_push_frame.png'), obs[:IMG])

  # ---- gate 5: one finite training update through the uint8 pipeline ----
  import jax
  import optax
  from crl import losses as losses_mod
  from crl import networks as networks_mod
  from crl.replay import TrajectoryBuffer
  from crl.train import collect_episode

  cfg.batch_size = 16
  cfg.max_replay_size = 4_000
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=True)
  start, end = cfg.start_index, cfg.end_index
  def obs_to_goal(states):
    return states[:, start:] if end == -1 else states[:, start:end]
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, obs_to_goal, optax.adam(3e-4, eps=1e-7),
      optax.adam(3e-4, eps=1e-7))
  state = init_state(jax.random.PRNGKey(0))
  buf = TrajectoryBuffer(
      capacity_steps=cfg.max_replay_size,
      ep_len_obs=cfg.max_episode_steps + 1,
      full_obs_dim=cfg.obs_dim + cfg.goal_dim, action_dim=cfg.action_dim,
      obs_dim=cfg.obs_dim, start_index=start, end_index=end,
      discount=cfg.discount, seed=0, obs_dtype=np.uint8)
  np_rng = np.random.default_rng(0)
  for _ in range(2):
    ob, ac, _ = collect_episode(env, None, None, jax.random.PRNGKey(0),
                                True, cfg.action_dim, np_rng)
    buf.add_episode(ob, ac)
  assert buf._obs.dtype == np.uint8
  batch = buf.sample(cfg.batch_size)
  batch = losses_mod.Transition(*[np.asarray(x) for x in batch])
  state, metrics = jax.jit(update_step)(state, batch)
  m = {k: float(v) for k, v in metrics.items()}
  g5 = all(np.isfinite(v) for v in m.values())
  results['gate5_train_step'] = {'pass': bool(g5), 'metrics': m,
                                 'buffer_dtype': str(buf._obs.dtype)}
  print(f"gate5 TRAIN_STEP  {'PASS' if g5 else 'FAIL'} "
        f"critic={m.get('critic_loss', float('nan')):.3f} "
        f"actor={m.get('actor_loss', float('nan')):.3f}")

  ok = all(v['pass'] for v in results.values())
  results['verdict'] = 'PASS' if ok else 'FAIL'
  with open(os.path.join(OUT, 'smoke_image_conedir.json'), 'w') as f:
    json.dump(results, f, indent=2)
  print(f'\nVERDICT: {results["verdict"]}  (artifacts in {OUT})')


if __name__ == '__main__':
  main()
