"""Render a rollout to a GIF so you can *see* the environment / trained agent.

On Colab (Fetch): frames come from MuJoCo (`render_mode='rgb_array'`), so set
`os.environ['MUJOCO_GL']='egl'` before calling. For `point_*` there is no MuJoCo
renderer, so frames are drawn with matplotlib (agent path over the maze).

Usage (CLI):
    python -m crl.visualize --env_name fetch_reach --ckpt runs/best.pkl \
        --out reach.gif --episodes 3
    python -m crl.visualize --env_name point_FourRooms --out point.gif

Usage (notebook):
    from crl.visualize import rollout_gif
    rollout_gif('fetch_reach', ckpt='drive/.../best.pkl', out='reach.gif')
    from IPython.display import Image; Image('reach.gif')
"""
import argparse
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl.config import Config


def _matplotlib_point_frames(env, path, goal):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  walls = env._walls  # pylint: disable=protected-access
  frames = []
  for t in range(len(path)):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.6)
    ax.plot(np.array(path)[:t + 1, 0], np.array(path)[:t + 1, 1], '-', lw=1)
    ax.scatter([path[t][0]], [path[t][1]], c='blue', s=40, label='agent')
    ax.scatter([goal[0]], [goal[1]], c='red', marker='*', s=140, label='goal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
  return frames


def rollout_gif(env_name, ckpt=None, out='rollout.gif', episodes=3,
                greedy=True, fps=20, seed=0, config=None):
  """Runs `episodes` rollouts and writes a GIF. Returns the output path.

  If `ckpt` is given, loads the policy from that checkpoint (see checkpoint.py);
  otherwise uses a random policy (still useful to see what the env looks like).
  """
  config = config or Config(env_name=env_name)
  config.env_name = env_name
  is_point = env_name.startswith('point_')
  render_mode = None if is_point else 'rgb_array'
  env = envs_mod.make_env(env_name, config, seed=seed, render_mode=render_mode)

  # Build networks (structure must match training) and load policy params.
  params = None
  if ckpt:
    from crl import checkpoint as ckpt_mod
    params = ckpt_mod.load_policy_params(ckpt)
    nets = networks_mod.make_networks(
        obs_dim=config.obs_dim, goal_dim=config.goal_dim,
        action_dim=config.action_dim, repr_dim=int(config.repr_dim),
        repr_norm=config.repr_norm, hidden_layer_sizes=config.hidden_layer_sizes,
        twin_q=config.twin_q, use_image_obs=config.use_image_obs)

    @jax.jit
    def act(p, obs, key):
      dist = nets.policy_network.apply(p, obs)
      return nets.sample_eval(dist, key) if greedy else nets.sample(dist, key)
  else:
    act = None

  rng = np.random.default_rng(seed)
  key = jax.random.PRNGKey(seed)
  frames = []
  successes = []
  for _ in range(episodes):
    obs = env.reset()
    hit = 0.0
    if is_point:
      path = [env.state.copy()]
      goal = env.goal.copy()
    for _ in range(env.max_episode_steps):
      if act is None:
        a = rng.uniform(-1, 1, size=config.action_dim).astype(np.float32)
      else:
        key, sub = jax.random.split(key)
        a = np.asarray(act(params, jnp.asarray(obs[None]), sub)[0])
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      if is_point:
        path.append(env.state.copy())
      else:
        frames.append(np.asarray(env.render()))
    successes.append(hit)
    if is_point:
      frames.extend(_matplotlib_point_frames(env, path, goal))

  imageio.mimsave(out, frames, duration=1.0 / fps)
  tag = 'trained' if ckpt else 'random'
  print(f'Wrote {out} ({len(frames)} frames, {episodes} {tag} episodes, '
        f'success={np.mean(successes):.2f})')
  return out


def main():
  p = argparse.ArgumentParser(description='Render a rollout to a GIF.')
  p.add_argument('--env_name', required=True)
  p.add_argument('--ckpt', default=None, help='checkpoint .pkl (else random).')
  p.add_argument('--out', default='rollout.gif')
  p.add_argument('--episodes', type=int, default=3)
  p.add_argument('--fps', type=int, default=20)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()
  rollout_gif(args.env_name, ckpt=args.ckpt, out=args.out,
              episodes=args.episodes, fps=args.fps, seed=args.seed)


if __name__ == '__main__':
  main()
