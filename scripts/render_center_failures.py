"""Render GIFs of representative center-controller failures from
diagnose_center_p30.py. Diagnosis-only; reproduces exact episodes.

FIDELITY: mujoco carries solver warmstart state across resets, so an episode's
outcome depends on the full preceding sequence -- reset-advancing alone does NOT
reproduce it. We therefore REPLAY the exact episode sequence (reset + full center
rollout) in ONE env from the diagnosis seed, capturing frames only on target
episodes. Verified byte-identical to the taxonomy (success + final_dist match).

Usage: python scripts/render_center_failures.py --per-cat 5
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                            # noqa: E402
from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from diagnose_center_p30 import SEED       # noqa: E402

MAJOR = ['physical_fall', 'controller_handoff_failure', 'timeout_near_goal',
         'timeout_insufficient_progress', 'stuck_oscillating',
         'route_deviation_wall', 'numerical_unknown']


def center_rollout(env, o, walker, base_act, renderer=None, cam=None, every=4):
  """One center episode (RP.run_route control law). If renderer given, capture
  frames. Returns (frames, success)."""
  true_goal = o[29:31].copy()
  handoff = False
  frames = []
  hit_r = 0.0
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  dead_at = None
  for t in range(env.max_episode_steps):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy(); oc[29:] = 0.0; oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
    o, r, _, info = env.step(a)
    if renderer is not None and t % every == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    hit_r = max(hit_r, float(r))
    if info['dead'] and dead_at is None:
      dead_at = t
    if hit_r > 0:
      break
    if dead_at is not None and t > dead_at + 5:
      break
  return frames, hit_r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--summary', default='artifacts/center_diag_p30/summary.json')
  ap.add_argument('--out', default='artifacts/center_diag_p30/gifs')
  ap.add_argument('--per-cat', type=int, default=5)
  ap.add_argument('--p-active', type=float, default=0.30)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  reps = json.load(open(args.summary))['representative_failure_ids']

  # target episode id -> category
  targets = {}
  for cat in MAJOR:
    for ep_id in reps.get(cat, [])[:args.per_cat]:
      targets[ep_id] = cat
  if not targets:
    print('no targets'); return
  last = max(targets)

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED),
      args.p_active)
  renderer = mujoco.Renderer(env._env.model, 240, 320)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0

  # REPLAY the exact sequence; render only targets
  for i in range(last + 1):
    o = env.reset()
    if i in targets:
      frames, hit_r = center_rollout(env, o, walker, base_act, renderer, cam)
      nm = f'{targets[i]}_ep{i}_{"succ" if hit_r > 0 else "fail"}.gif'
      imageio.mimsave(os.path.join(args.out, nm), frames, fps=25, loop=0)
      print('saved', nm, len(frames), 'frames | success', int(hit_r > 0),
            flush=True)
    else:
      center_rollout(env, o, walker, base_act)   # advance state faithfully


if __name__ == '__main__':
  main()
