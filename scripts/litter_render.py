"""Render GIFs of gate-arm episodes in offline_ant_umaze_litter.

One GIF per requested (arm, u_side) case: tracking camera follows the ant
through the litter corridor; frames annotated by filename only. Episodes are
driven by the same carrot controller as scripts/litter_geometry_gate.py.

Usage:
  python scripts/litter_render.py \
      --ckpt offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl \
      [--eps-per-case 2] [--out artifacts/litter_env/gifs]
"""
import argparse
import os
import sys

import imageio
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
import litter_geometry_gate as G          # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

FPS = 25
FRAME_EVERY = 3        # env steps per frame (dt 0.1 -> ~3.3x realtime at 25fps)
SIZE = (240, 320)      # (height, width)


def render_episode(env, act, arm, slow_spec, u_side, ep, renderer, cam):
  o = env.reset(u_side=u_side) if u_side is not None else env.reset()
  true_goal = o[29:31].copy()
  y = G.lane_y(arm, getattr(env, 'u_side', 0) or 0, ep)
  goal_vec = np.zeros(29, np.float32)
  handoff = arm == 'direct'
  frames, hit, dead_at = [], 0.0, None
  for t in range(env.max_episode_steps):
    xy = o[:2]
    if not handoff and (xy[0] >= G.HANDOFF_X or xy[1] >= 2.0):
      handoff = True
    if handoff:
      goal_vec[:2] = true_goal
    else:
      ramp = min(1.0, max(0.0, (G.HANDOFF_X + 0.3 - xy[0]) / 0.8))
      cap = 4.0 if arm == 'pile' else G.HANDOFF_X + 1.0
      goal_vec[:2] = (min(xy[0] + G.LOOKAHEAD, cap), y * ramp)
    in_zone = 1.5 < xy[0] < G.HANDOFF_X
    spec = slow_spec if (arm == 'middle_slow' and in_zone) else ''
    use_policy, scale = G.slow_gate(spec, t)
    o_cmd = o.copy()
    o_cmd[29:] = goal_vec
    a = (np.asarray(act(jnp.asarray(o_cmd[None]))[0]) * scale
         if use_policy else np.zeros(8, np.float32))
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    if info.get('dead') and dead_at is None:
      dead_at = t
    if t % FRAME_EVERY == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    if hit > 0 or (dead_at is not None and t > dead_at + 30):
      break
  return frames, {'success': hit > 0, 'dead': dead_at is not None,
                  'steps': t + 1, 'u_side': env.u_side}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--eps-per-case', type=int, default=2)
  ap.add_argument('--slow', default='duty1_1')
  ap.add_argument('--out', default='artifacts/litter_env/gifs')
  ap.add_argument('--cases', nargs='+', default=[
      'clean:0', 'clean:1', 'middle_fast:', 'middle_slow:', 'pile:0',
      'pile:1', 'direct:0', 'direct:1'])
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=G.ENV_SEED)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(args.ckpt)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=G.ENV_SEED)
  renderer = mujoco.Renderer(env._env.model, SIZE[0], SIZE[1])
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 7.0, -50.0, -90.0

  for case in args.cases:
    arm, u_str = case.split(':')
    u_side = int(u_str) if u_str else None
    for ep in range(args.eps_per_case):
      frames, meta = render_episode(env, act, arm, args.slow, u_side, ep,
                                    renderer, cam)
      tag = ('ok' if meta['success'] else
             ('dead' if meta['dead'] else 'timeout'))
      name = (f'{arm}_u{meta["u_side"]}_ep{ep}_{tag}_'
              f'{meta["steps"]}steps.gif')
      path = os.path.join(args.out, name)
      imageio.mimsave(path, frames, fps=FPS, loop=0)
      print(f'saved {path} ({len(frames)} frames)')


if __name__ == '__main__':
  main()
