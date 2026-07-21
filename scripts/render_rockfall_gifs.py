"""Deterministic GIFs for the rockfall pilot (analysis only):
clear-side traversal, severe hit, impaired-leg failure, mild-hit recovery,
center-safe traversal, and a paired-mask pair rendered side by side
(identical until the rockfall event).

Scenario masks/severities are FORCED via the probe-only reset overrides;
episode selection within a scenario is a deterministic search over reset
indices (recorded in the printed log and gif filename).
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
from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402

OUT = 'artifacts/rockfall_pilot/gifs'
SEED = 60_777


def rollout_frames(env, o, walker, base_act, route, renderer, cam,
                   every=4):
  true_goal = o[29:31].copy()
  handoff = False
  frames = []
  hit_r = 0.0
  dead_at = None
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  for t in range(env.max_episode_steps):
    x = float(o[0])
    if not handoff and (x >= RP.HANDOFF_X or float(o[1]) >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command(route, t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
    o, r, _, info = env.step(a)
    if t % every == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    hit_r = max(hit_r, float(r))
    if info['dead'] and dead_at is None:
      dead_at = t
    if hit_r > 0 or (dead_at is not None and t > dead_at + 20):
      break
  return frames, hit_r, {'dead': dead_at is not None, 'steps': t + 1,
                         'hit': list(env._hit)}


def find_and_render(env, walker, base_act, renderer, cam, name, route,
                    mask, severities, want, max_tries=12):
  """Deterministic search: first reset whose episode matches `want`."""
  for k in range(max_tries):
    o = env.reset(mask=mask, severities=severities)
    frames, hit_r, meta = rollout_frames(env, o, walker, base_act, route,
                                         renderer, cam)
    got = ('success' if hit_r > 0 else
           'collapse' if meta['dead'] else 'fail')
    hit_any = any(meta['hit'])
    ok = (got == want if want != 'hit_fail'
          else (hit_any and got != 'success'))
    if ok or k == max_tries - 1:
      nm = f'{name}_try{k}_{got}.gif'
      imageio.mimsave(os.path.join(OUT, nm), frames, fps=25, loop=0)
      print(('saved' if ok else 'saved (fallback)'), nm,
            len(frames), 'frames', flush=True)
      return
  raise AssertionError('unreachable')


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED)
  renderer = mujoco.Renderer(env._env.model, 240, 320)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0

  mild4 = ('mild',) * 4
  find_and_render(env, walker, base_act, renderer, cam,
                  'clear_side', 'left', (0, 0, 1, 1), None, 'success')
  find_and_render(env, walker, base_act, renderer, cam,
                  'severe_hit', 'left', (1, 0, 0, 0), ('severe',) * 4,
                  'collapse')
  find_and_render(env, walker, base_act, renderer, cam,
                  'impaired_fail', 'left', (1, 0, 0, 0), ('impaired',) * 4,
                  'hit_fail')
  find_and_render(env, walker, base_act, renderer, cam,
                  'mild_recovery', 'left', (1, 0, 0, 0), mild4, 'success')
  find_and_render(env, walker, base_act, renderer, cam,
                  'center_safe', 'center', (1, 1, 1, 1), None, 'success')

  # paired masks, side by side: identical until the rockfall event
  penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg,
                           seed=SEED + 1)
  ren2 = mujoco.Renderer(penv._env.model, 240, 320)
  penv.reset()
  q0 = np.asarray(penv._env.data.qpos)[:RA.NQ_ANT].copy()
  v0 = np.asarray(penv._env.data.qvel)[:RA.NV_ANT].copy()
  goal = penv._flatten(penv._last_obs)[29:31].copy()
  sides = {}
  for tag, mask in (('a', (0, 0, 0, 0)), ('b', (1, 1, 0, 0))):
    o = RP.set_state(penv, q0, v0, goal, mask, ('mild',) * 4)
    frames, hit_r, meta = rollout_frames(penv, o, walker, base_act,
                                         'left', ren2, cam, every=3)
    sides[tag] = frames
    print(f'paired branch {tag}: mask {mask} success {hit_r} '
          f'steps {meta["steps"]}', flush=True)
  n = max(len(sides['a']), len(sides['b']))
  pad = lambda f: f + [f[-1]] * (n - len(f))
  combo = [np.concatenate([fa, fb], axis=1)
           for fa, fb in zip(pad(sides['a']), pad(sides['b']))]
  imageio.mimsave(os.path.join(OUT, 'paired_mask_clear_vs_left.gif'),
                  combo, fps=25, loop=0)
  print('saved paired_mask_clear_vs_left.gif', len(combo), 'frames ->',
        OUT, flush=True)


if __name__ == '__main__':
  main()
