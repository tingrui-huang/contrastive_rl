"""Top-down GIFs of the V3 two-route rockfall benchmark (BR): the sighted
expert vs the u-blind naive CRL agent.

Expert clips (latent forced through reset(rockfall_active=...)):
  expert_clear_shortcut      u=clear  -> teacher takes the shortcut
  expert_active_detour       u=active -> teacher takes the detour
  expert_active_forced_shortcut  u=active but the teacher is told
                             'shortcut' anyway: the do(shortcut) experiment
                             behind the 0.724 interventional success rate.

Agent clips (v3br_crl_s1_100k/final.pkl, deterministic tanh(mu), eval seed
909): episodes are replayed by running the same seed sequence from episode
0, so the k-th clip is exactly the k-th episode of the CPU probe
(probe_tworoute_v3_detour.py) -- same route labels, same outcomes.
  agent_k<k>_shortcut_success_clear / shortcut_death_active /
  detour_success / detour_timeout / none_corner

Frames every 4 env steps at 20 fps, fixed top-down camera over the whole
maze, goal ring and a caption drawn with PIL. Rocks are real MuJoCo bodies
and render where they land.

Usage: python scripts/render_tworoute_v3_gifs.py [--variant br]
"""
import argparse
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
from PIL import Image, ImageDraw           # noqa: E402
from crl import envs as envs_mod          # noqa: E402
import tworoute_v3_teacher as TT           # noqa: E402
from probe_tworoute_v3_detour import (     # noqa: E402
    OUT_ROOT, build_policy, make_env)

EVERY, FPS, SIZE = 4, 20, 360
#: top-down camera over the maze; world (x, y) -> pixel via the affine map
#: measured on a test frame (maze spans about -2..10 in both axes).
CAM = dict(lookat=(4.0, 4.0, 0.0), distance=20.0, elevation=-90.0,
           azimuth=90.0)
PX_PER_UNIT, CENTER = 23.3, (180.0, 180.0)
#: agent clips to render: (name, episode index k in the seed-909 sequence)
AGENT_CLIPS = {
    'br': [('shortcut_success_clear', 1), ('shortcut_death_active', 0),
           ('detour_success', 6), ('detour_timeout', 26),
           ('none_corner', 21)],
}
AGENT_CKPT = {'br': 'v3br_crl_s1_100k/final.pkl'}
GOAL_R = 0.5


def to_px(x, y):
  return (CENTER[0] + (x - 4.0) * PX_PER_UNIT,
          CENTER[1] - (y - 4.0) * PX_PER_UNIT)


class Recorder:
  def __init__(self, env, title):
    self.env, self.title = env, title
    self.renderer = mujoco.Renderer(env._env.model, SIZE, SIZE)
    self.cam = mujoco.MjvCamera()
    self.cam.lookat[:] = CAM['lookat']
    self.cam.distance, self.cam.elevation, self.cam.azimuth = (
        CAM['distance'], CAM['elevation'], CAM['azimuth'])
    self.frames = []

  def grab(self, t, goal_xy, u, status=''):
    self.renderer.update_scene(self.env._env.data, camera=self.cam)
    im = Image.fromarray(self.renderer.render().copy())
    dr = ImageDraw.Draw(im)
    gx, gy = to_px(*goal_xy)
    r = GOAL_R * PX_PER_UNIT
    dr.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(220, 30, 30),
               width=2)
    dr.rectangle([0, 0, SIZE, 30], fill=(0, 0, 0))
    dr.text((4, 2), self.title, fill=(255, 255, 255))
    dr.text((4, 16), f'u={"ACTIVE" if u else "clear"}  t={t:3d}  {status}',
            fill=(255, 220, 120) if u else (180, 255, 180))
    self.frames.append(np.asarray(im))

  def save(self, path):
    #: hold the last frame so the outcome is readable.
    frames = self.frames + [self.frames[-1]] * FPS
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    print(f'  {path}  ({len(self.frames)} frames)', flush=True)
    self.renderer.close()


def outcome(info):
  if info.get('success'):
    return 'SUCCESS'
  if info.get('failure'):
    return 'DEAD (rock)'
  return 'timeout'


def expert_clips(variant, out):
  cfg, teacher = TT.make_teacher(variant)
  env = envs_mod.make_env(TT.env_name(variant), cfg, seed=101)
  for name, u, route in (('expert_clear_shortcut', False, 'shortcut'),
                         ('expert_active_detour', True, 'detour'),
                         ('expert_active_forced_shortcut', True, 'shortcut')):
    rec = Recorder(env, f'EXPERT (sees u) -> {route}'
                   + ('  [forced: do(shortcut)]' if 'forced' in name else ''))
    o = env.reset(rockfall_active=u)
    teacher.fresh()
    goal = o[29:31].copy()
    info = {}
    for t in range(TT.HORIZON):
      if t % EVERY == 0:
        rec.grab(t, goal, u)
      a = teacher.act(o, route)
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    rec.grab(t + 1, goal, u, outcome(info))
    rec.save(os.path.join(out, f'{name}.gif'))


def agent_clips(variant, out, seed=909):
  act, _, step = build_policy(AGENT_CKPT[variant], variant)
  clips = dict((k, n) for n, k in AGENT_CLIPS[variant])
  _, env = make_env(variant, seed)
  for k in range(max(clips) + 1):
    o = env.reset()
    u = bool(env.privileged_rockfall_active)
    goal = o[29:31].copy()
    rec = Recorder(env, f'NAIVE CRL (blind to u)  ep {k}') if k in clips else None
    info = {}
    for t in range(TT.HORIZON):
      if rec is not None and t % EVERY == 0:
        rec.grab(t, goal, u)
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    if rec is not None:
      rec.grab(t + 1, goal, u, outcome(info))
      rec.save(os.path.join(out, f'agent_k{k}_{clips[k]}.gif'))
      print(f'    ep {k}: u={u} route={info.get("route")} {outcome(info)} '
            f'steps={t + 1}', flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', default='br', choices=['br', 'tr'])
  args = ap.parse_args()
  out = os.path.join(OUT_ROOT, args.variant, 'gifs')
  os.makedirs(out, exist_ok=True)
  print('expert clips', flush=True)
  expert_clips(args.variant, out)
  print('agent clips', flush=True)
  agent_clips(args.variant, out)


if __name__ == '__main__':
  main()
