"""Deterministic GIFs for the naive-policy diagnosis: paired U-flip clones
(identical until litter touch) + wrong-side-then-switch examples. Analysis only.
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
import diagnose_naive_policy as D         # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/naive_policy_diagnosis/gifs'
SEED = 71717


def render(env, act, obs0, renderer, cam, actions=None):
  o = obs0
  frames = []
  T = env.max_episode_steps if actions is None else len(actions)
  hit, dead_at = 0.0, -1
  for t in range(T):
    a = actions[t] if actions is not None else np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    if t % 4 == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    hit = max(hit, float(r))
    if info.get('dead') and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 6):
      break
  return frames, hit


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg = build_offline_cfg(); cfg.offline_dataset = ''; cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = ckpt_mod.load_checkpoint('naive_litter_crl_s0_60k/best.pkl')
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  R = mujoco.Renderer  # noqa
  # ---- paired U-flip: first 3 clones (seed+5), render u0 and u1 ----
  penv = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=SEED + 5)
  ren = mujoco.Renderer(penv._env.model, 240, 320)
  cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0
  for k in range(3):
    penv.reset(u_side=0)
    q0 = penv._env.data.qpos.copy(); v0 = penv._env.data.qvel.copy()
    goal = penv._flatten(penv._last_obs)[29:31].copy()
    for u in (0, 1):
      o = D.set_state(penv, q0, v0, goal, u)
      frames, hit = render(penv, act, o, ren, cam)
      nm = f'paired{k}_u{u}_{"succ" if hit>0 else "fail"}.gif'
      imageio.mimsave(os.path.join(OUT, nm), frames, fps=25, loop=0)
      print('saved', nm, len(frames), 'frames', flush=True)

  # ---- wrong-side-then-switch examples: reproduce the eval loop ----
  sel = json.load(open('artifacts/naive_policy_diagnosis/gif_selection.json'))['switch_eps']
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=SEED)
  ren2 = mujoco.Renderer(env._env.model, 240, 320)
  maxi = max(sel) if sel else -1
  for i in range(maxi + 1):
    o = env.reset(u_side=i % 2)
    if i not in sel:
      continue
    frames, hit = render(env, act, o, ren2, cam)
    nm = f'switch_ep{i}_u{i%2}_{"succ" if hit>0 else "fail"}.gif'
    imageio.mimsave(os.path.join(OUT, nm), frames, fps=25, loop=0)
    print('saved', nm, len(frames), 'frames', flush=True)
  print('done ->', OUT)


if __name__ == '__main__':
  main()
