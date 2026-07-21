"""Visualize the v2 local-detour teacher + confirm paired hiddenness.
Analysis only; frozen env unchanged.

Outputs (artifacts/rockfall_v2/):
  * detour_sighted_ep*.gif    -- a sighted episode with active site(s): the
                                 ant dips inward around each active site and
                                 returns to the base lane;
  * straight_blind_ep*.gif    -- same base side, detours OFF: walks straight
                                 into the active site (contrast);
  * detour_traces.json        -- y(x) traces + site active/inactive labels
                                 for a handful of episodes (for plotting);
  * hiddenness.json           -- paired-reset env hiddenness (mask-independent
                                 policy; env is shared with v1 so this is a
                                 confirmation).
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
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402

OUT = 'artifacts/rockfall_v2'
SEED = 43_517


def render_detour_episode(env, o, walker, base_act, base_side, use_detour,
                          renderer, cam, every=4):
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = V2.active_site_windows(base_sgn, env.rockfall_mask) if use_detour \
      else []
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  frames, xs, ys = [], [], []
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
      if 1.5 < x < 5.7:
        xs.append(x)
        ys.append(y)
    o, r, _, info = env.step(a)
    if t % every == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      renderer.update_scene(d, camera=cam)
      frames.append(renderer.render().copy())
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 12):
      break
  return frames, xs, ys, hit, (dead_at >= 0)


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED))
  ren = mujoco.Renderer(env._env.model, 240, 320)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0

  # deterministic search: sighted episode whose base side has >=1 active site
  side_rng = np.random.default_rng(SEED + 999)
  traces = []
  n_gif = 0
  for k in range(80):
    base = 'left' if side_rng.random() < 0.5 else 'right'
    base_sgn = 1.0 if base == 'left' else -1.0
    o = env.reset()
    mask = list(env.rockfall_mask)
    active_on_base = [(nm, sx) for nm, sx, sgn in RA.ROCKFALL_SITES
                      if sgn == base_sgn and mask[V2.BIT[nm]]]
    q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
    goal = env._flatten(env._last_obs)[29:31].copy()
    if active_on_base and n_gif < 2:
      # sighted (detour ON)
      fr, xs, ys, hit, dead = render_detour_episode(
          env, o, walker, base_act, base, True, ren, cam)
      nm = f'detour_sighted_ep{k}_{base}_{"succ" if hit>0 else "fail"}.gif'
      imageio.mimsave(os.path.join(OUT, nm), fr, fps=25, loop=0)
      print('saved', nm, len(fr), 'frames | active_on_base', active_on_base,
            flush=True)
      traces.append({'ep': k, 'base': base, 'mask': mask, 'policy': 'sighted',
                     'active_on_base': active_on_base, 'xs': xs, 'ys': ys})
      # blind contrast (detour OFF), same reset via set_state
      o2 = V2 and env.reset(mask=tuple(mask))
      # restore identical start
      d = env._env.data
      d.qpos[:RA.NQ_ANT] = q0
      d.qvel[:RA.NV_ANT] = v0
      mujoco.mj_forward(env._env.model, d)
      env._last_obs = env._env._obs_dict()
      env._goal_vec = np.zeros(29, np.float32)
      env._goal_vec[:2] = goal
      env._env.goal = np.asarray(goal, float).copy()
      o2 = env._flatten(env._last_obs)
      fr2, xs2, ys2, hit2, dead2 = render_detour_episode(
          env, o2, walker, base_act, base, False, ren, cam)
      nm2 = f'straight_blind_ep{k}_{base}_{"succ" if hit2>0 else "fail"}.gif'
      imageio.mimsave(os.path.join(OUT, nm2), fr2, fps=25, loop=0)
      print('saved', nm2, len(fr2), 'frames', flush=True)
      traces.append({'ep': k, 'base': base, 'mask': mask, 'policy': 'blind',
                     'active_on_base': active_on_base, 'xs': xs2, 'ys': ys2})
      n_gif += 1
    if n_gif >= 2:
      break
  json.dump(traces, open(os.path.join(OUT, 'detour_traces.json'), 'w'))
  print('saved detour_traces.json', flush=True)

  # ---- paired hiddenness (env unchanged; mask-independent policy) ----
  penv = V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED + 11))
  pairs = RP.paired_hiddenness(penv, walker, base_act, 8)
  n_ok = sum(p['ok'] for p in pairs)
  json.dump({'pairs': pairs, 'n_ok': n_ok, 'n': len(pairs)},
            open(os.path.join(OUT, 'hiddenness.json'), 'w'), indent=2)
  print(f'paired hiddenness: {n_ok}/{len(pairs)} identical until the drop',
        flush=True)

  # ---- compact y(x) trace table for the detour episode ----
  for tr in traces:
    if tr['policy'] != 'sighted':
      continue
    xs, ys = np.asarray(tr['xs']), np.asarray(tr['ys'])
    print(f"\nsighted ep{tr['ep']} base={tr['base']} "
          f"active_on_base={tr['active_on_base']}")
    for xq in np.arange(2.0, 5.6, 0.4):
      sel = (xs >= xq - 0.2) & (xs < xq + 0.2)
      tag = ''
      for nm, sx in tr['active_on_base']:
        if abs(xq - sx) <= 0.4:
          tag = f' <- active {nm}'
      print(f"  x={xq:.1f}  |y|={np.mean(np.abs(ys[sel])):.2f}" if sel.any()
            else f"  x={xq:.1f}  --", tag, flush=True)
    break


if __name__ == '__main__':
  main()
