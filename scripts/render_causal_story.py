"""Causal-story illustration GIFs for a talk. Analysis only; frozen env.

Same base side, same start, same hidden hazard map -> the SIGHTED teacher reads
the mask and detours locally around the ACTIVE site; the BLIND learner cannot
see the mask, walks straight in, and (under v2.1 severity) collapses. Rendered
top-down so the lateral detour is visible, side by side with labels.
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
try:
  from PIL import Image, ImageDraw         # noqa: E402
  _PIL = True
except Exception:                          # pylint: disable=broad-except
  _PIL = False
from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402

OUT = 'artifacts/rockfall_v2/causal_story'
SEED = 71_003
#: force a clear scenario: base=left, left_2 (x=4.3) ACTIVE + SEVERE.
MASK = (0, 1, 0, 0)
SEV = ('mild', 'severe', 'mild', 'mild')
BASE = 'left'


def label(frames, text, color):
  if not _PIL:
    return frames
  out = []
  for f in frames:
    im = Image.fromarray(f)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=color)
    out.append(np.asarray(im))
  return out


def run(env, o, walker, base_act, use_detour, renderer, cam, every=3):
  base_sgn = 1.0
  wins = V2.active_site_windows(base_sgn, env.rockfall_mask) if use_detour \
      else []
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  frames = []
  hit, dead_at = 0.0, -1
  for t in range(env.max_episode_steps):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                       RP.V_SIDE)
      a = walker(o, y_cmd, v_cmd)
    o, r, _, info = env.step(a)
    if t % every == 0:
      d = env._env.data
      cam.lookat[:] = (min(float(d.qpos[0]), 5.5), 0.0, 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 14):
      break
  return frames, hit, dead_at >= 0


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED))
  ren = mujoco.Renderer(env._env.model, 300, 420)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 7.0, -80.0, -90.0

  # identical start for both branches
  env.reset(mask=MASK, severities=SEV)
  q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
  v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
  goal = env._flatten(env._last_obs)[29:31].copy()

  def restore():
    o = V2.apply_v2_config(env)  # noqa
    env.reset(mask=MASK, severities=SEV)
    d = env._env.data
    d.qpos[:RA.NQ_ANT] = q0
    d.qvel[:RA.NV_ANT] = v0
    mujoco.mj_forward(env._env.model, d)
    env._last_obs = env._env._obs_dict()
    env._goal_vec = np.zeros(29, np.float32)
    env._goal_vec[:2] = goal
    env._env.goal = np.asarray(goal, float).copy()
    return env._flatten(env._last_obs)

  fs, hs, ds = run(env, restore(), walker, base_act, True, ren, cam)
  fb, hb, db = run(env, restore(), walker, base_act, False, ren, cam)
  print(f'sighted: success={hs>0} dead={ds} frames={len(fs)}', flush=True)
  print(f'blind:   success={hb>0} dead={db} frames={len(fb)}', flush=True)

  fs = label(fs, 'SIGHTED  (reads hazard map -> detours)', (90, 230, 120))
  fb = label(fb, 'BLIND learner (no map -> walks in)', (240, 120, 110))
  imageio.mimsave(os.path.join(OUT, 'sighted_detour.gif'), fs, fps=22, loop=0)
  imageio.mimsave(os.path.join(OUT, 'blind_straight.gif'), fb, fps=22, loop=0)

  n = max(len(fs), len(fb))
  pad = lambda f: f + [f[-1]] * (n - len(f))
  gap = np.full((fs[0].shape[0], 6, 3), 255, np.uint8)
  combo = [np.concatenate([a, gap, b], axis=1)
           for a, b in zip(pad(fs), pad(fb))]
  imageio.mimsave(os.path.join(OUT, 'causal_sighted_vs_blind.gif'), combo,
                  fps=22, loop=0)
  print('saved ->', OUT, flush=True)


if __name__ == '__main__':
  main()
