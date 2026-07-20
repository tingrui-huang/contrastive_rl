"""Render the deterministically-selected coverage GIFs (first 3 per failure
category + 3 successes) by re-simulating the frozen pipeline. Analysis only.
"""
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import walker_gate as WG                  # noqa: E402
import collect_litter_pilot as CL         # noqa: E402

SIDE = 'artifacts/litter_dataset/full/antmaze_litter_full_sidecar.npz'
ENV_SEED = 25_770_061
OUT = 'artifacts/coverage_failure_analysis/gifs'
HANDOFF_X, COVERAGE_V = WG.HANDOFF_X, CL.COVERAGE_MIDDLE_SLOW_V


def render_episode(env, walker, base_act, renderer, cam):
  o = env._flatten(env._last_obs)
  true_goal = o[29:31].copy()
  handoff, dead_at = False, -1
  x_hist, nudge_until, nudge_sign = [], -1, 1.0
  frames = []
  for t in range(CL.HORIZON):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      y_cmd = 0.0
      x_hist.append(x)
      if t < nudge_until:
        y_cmd = nudge_sign * WG.NUDGE_Y
      elif (len(x_hist) > WG.STALL_WINDOW
            and x_hist[-1] - x_hist[-WG.STALL_WINDOW] < WG.STALL_MIN_DX):
        nudge_until = t + WG.NUDGE_STEPS
        nudge_sign = -nudge_sign
        x_hist.clear()
        y_cmd = nudge_sign * WG.NUDGE_Y
      a = walker(o, y_cmd, COVERAGE_V)
    o2, r, _, info = env.step(a)
    if t % 4 == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    if info.get('dead') and dead_at < 0:
      dead_at = t
    if dead_at >= 0 and t > dead_at + 8:
      break
    o = o2
  return frames


def main():
  os.makedirs(OUT, exist_ok=True)
  rep = json.load(open('artifacts/coverage_failure_analysis/report.json'))
  sel = rep['gif_selection']
  targets = {}
  for cat, ids in sel.items():
    for e in ids:
      targets[int(e)] = cat
  if not targets:
    print('no targets')
    return
  max_id = max(targets)
  sc = np.load(SIDE, allow_pickle=True)
  u_side = sc['u_side'].astype(int)

  cfg, walker, base_act, _, _ = C.load_controllers(
      'artifacts/walker/phase1/walker_best.pkl',
      'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl')
  cfg.offline_dataset = ''
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=ENV_SEED)
  renderer = mujoco.Renderer(env._env.model, 240, 320)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0

  for e in range(max_id + 1):
    env.reset()
    assert int(env.u_side) == u_side[e]
    if e not in targets:
      continue
    frames = render_episode(env, walker, base_act, renderer, cam)
    cat = targets[e]
    name = f'{cat}_ep{e}_u{u_side[e]}.gif'
    imageio.mimsave(os.path.join(OUT, name), frames, fps=25, loop=0)
    print('saved', name, len(frames), 'frames', flush=True)
  print('done ->', OUT)


if __name__ == '__main__':
  main()
