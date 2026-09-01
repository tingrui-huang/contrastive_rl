"""Paired walk GIFs on the rockfall-v2 CLEAN-dataset protocol. Analysis only:
frozen env, frozen walker/base controllers, no retraining, nothing written
outside --out-dir.

For each episode the SAME reset (start pose, goal, hidden mask + severities) is
walked twice:

  * expert_*  -- the sighted local-detour teacher that COLLECTED the clean
                 dataset (scripts/rockfall_v2_teacher.py driving the frozen
                 walker pre-handoff, frozen base policy after handoff). Only
                 episodes the teacher survives are kept, because D_clean is
                 exactly the dead=False episodes of the pilot;
  * ant_*     -- the CRL policy trained on that clean npz, acting from the
                 58-dim learner obs (mask invisible).

Protocol matches artifacts/rockfall_v2_p30_h800_resetfix (v2.1 severity,
p_active=0.30, H=800, canonical full reset), i.e. the protocol the clean split
and the failneg_clean_* runs were built on.

Usage:
  python scripts/render_clean_walk_gifs.py                     # 8 pairs
  python scripts/render_clean_walk_gifs.py --n 4 --mp4         # fewer, mp4
  python scripts/render_clean_walk_gifs.py --ckpt <run>/best.pkl
"""
import argparse
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
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402

OUT = 'artifacts/clean_walk_gifs'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
#: fresh render seed, disjoint from every collection/eval seed in the repo.
SEED = 52_411
P_ACTIVE = 0.30
HORIZON = 800


def load_ant(cfg, ckpt_path):
  """Learned CRL policy (deterministic tanh-mean action)."""
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  return act, int(step)


def set_state(env, qpos, qvel, goal_xy, mask, severities):
  """Restore an identical episode start under a forced mask/severity."""
  env.reset(mask=mask, severities=severities)
  d = env._env.data
  d.qpos[:RA.NQ_ANT] = qpos
  d.qvel[:RA.NV_ANT] = qvel
  d.qacc_warmstart[:] = 0.0
  env._goal_vec = np.zeros(29, np.float32)
  env._goal_vec[:2] = goal_xy
  env._goal_state_full = env._goal_vec.copy()
  env._env.goal = np.asarray(goal_xy, float).copy()
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()
  return env._flatten(env._last_obs)


def grab(env, ren, cam, frames):
  d = env._env.data
  cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
  ren.update_scene(d, camera=cam)
  frames.append(ren.render().copy())


def summarize(ys, hit, dead_at, t, mask, env):
  #: ys is the FIRST-PASS hazard-zone lane only (2.3<=x<=5.7 before the
  #: handoff condition x>=HANDOFF_X or y>=2.0 is met). The naive
  #: "2.3<=x<=5.7" window also catches the return leg of the U at y~8, which
  #: swamps the mean and mislabels right-lane episodes as 'left'.
  ys = np.asarray(ys) if len(ys) else np.zeros(1)
  mean_y = float(np.mean(ys))
  return {'success': float(hit > 0), 'dead': dead_at >= 0, 'steps': int(t + 1),
          'mean_y_zone': round(mean_y, 3),
          'route': ('left' if mean_y > 0.5 else
                    'right' if mean_y < -0.5 else 'center'),
          'mask': list(mask), 'triggered': list(env._triggered),
          'hit': list(env._hit)}


def run_expert(env, o, walker, base_act, base_side, ren, cam, every):
  """The sighted local-detour teacher: exactly the collector's control law."""
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = V2.active_site_windows(base_sgn, env.rockfall_mask)
  true_goal = o[29:31].copy()
  handoff, x_hist, nudge = False, [], {'until': -1, 'sign': 1.0}
  frames, ys, trace = [], [], []
  hit, dead_at, t = 0.0, -1, 0
  for t in range(env.max_episode_steps):
    x, y = float(o[0]), float(o[1])
    trace.append((round(x, 3), round(y, 3)))
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
    if not handoff and 2.3 <= float(o[0]) <= 5.7:
      ys.append(float(o[1]))
    if t % every == 0:
      grab(env, ren, cam, frames)
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 12):
      break
  st = summarize(ys, hit, dead_at, t, env.rockfall_mask, env)
  st['trace'] = trace[::2]
  st['base_side'] = base_side
  st['active_on_base'] = [nm for nm, _, sgn in RA.ROCKFALL_SITES
                          if sgn == base_sgn and env.rockfall_mask[V2.BIT[nm]]]
  return frames, st


def run_ant(env, o, act, ren, cam, every):
  """The learned policy: 58-dim obs only, hidden mask invisible."""
  frames, ys, trace = [], [], []
  hit, dead_at, t = 0.0, -1, 0
  passed = False
  for t in range(env.max_episode_steps):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    trace.append((round(x, 3), round(y, 3)))
    if not passed and 2.3 <= x <= 5.7:
      ys.append(y)
    if x >= RP.HANDOFF_X or y >= 2.0:
      passed = True
    if t % every == 0:
      grab(env, ren, cam, frames)
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 12):
      break
  st = summarize(ys, hit, dead_at, t, env.rockfall_mask, env)
  st['trace'] = trace[::2]
  return frames, st


def save(frames, path, fps, mp4):
  if mp4:
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=None)
  else:
    imageio.mimsave(path, frames, fps=fps, loop=0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT, help='CRL policy trained on D_clean')
  ap.add_argument('--n', type=int, default=8, help='paired episodes to render')
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--every', type=int, default=4, help='render 1 of N steps')
  ap.add_argument('--fps', type=int, default=25)
  ap.add_argument('--height', type=int, default=240)
  ap.add_argument('--width', type=int, default=320)
  ap.add_argument('--mp4', action='store_true', help='write .mp4 not .gif')
  ap.add_argument('--max-tries', type=int, default=60,
                  help='reset budget while searching for surviving teacher '
                       'episodes (D_clean == teacher dead=False)')
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  ext = 'mp4' if args.mp4 else 'gif'

  cfg, walker, base_act, base_step, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = float(args.p_active)
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.rockfall_reset_fix = True
  ant_act, ant_step = load_ant(cfg, args.ckpt)
  print(f'ant policy {args.ckpt} @ step {ant_step} | base policy @ '
        f'{base_step} | protocol {V2.protocol_version(args.p_active)}'
        f'_h{args.horizon}_resetfix_v1', flush=True)

  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)
  assert env.max_episode_steps == args.horizon, env.max_episode_steps
  ren = mujoco.Renderer(env._env.model, args.height, args.width)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0

  rows, k, tries, skipped = [], 0, 0, 0
  while k < args.n and tries < args.max_tries:
    tries += 1
    o = env.reset()
    mask = tuple(env.rockfall_mask)
    sev = tuple(env.privileged_severity)
    q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
    goal = env._flatten(env._last_obs)[29:31].copy()
    side = 'left' if k % 2 == 0 else 'right'

    fr_e, st_e = run_expert(env, o, walker, base_act, side, ren, cam,
                            args.every)
    if st_e['dead']:
      #: teacher deaths are the rock-fail split, not D_clean -- redraw.
      skipped += 1
      print(f'  [skip try {tries}] teacher died at step {st_e["steps"]} '
            f'(rock-fail episode, not in D_clean)', flush=True)
      continue

    o2 = set_state(env, q0, v0, goal, mask, sev)
    fr_a, st_a = run_ant(env, o2, ant_act, ren, cam, args.every)

    tag_e = 'succ' if st_e['success'] else 'timeout'
    tag_a = ('succ' if st_a['success'] else
             'dead' if st_a['dead'] else 'timeout')
    nm_e = f'ep{k}_expert_{side}_{tag_e}.{ext}'
    nm_a = f'ep{k}_ant_{st_a["route"]}_{tag_a}.{ext}'
    save(fr_e, os.path.join(args.out_dir, nm_e), args.fps, args.mp4)
    save(fr_a, os.path.join(args.out_dir, nm_a), args.fps, args.mp4)
    print(f'[pair {k}] mask={list(mask)} side={side} '
          f'active_on_base={st_e["active_on_base"]}\n'
          f'    expert -> {nm_e} ({len(fr_e)} frames, {st_e["steps"]} steps)\n'
          f'    ant    -> {nm_a} ({len(fr_a)} frames, {st_a["steps"]} steps, '
          f'route={st_a["route"]})', flush=True)
    rows.append({'pair': k, 'mask': list(mask), 'severity': list(sev),
                 'base_side': side, 'expert': st_e, 'ant': st_a,
                 'files': {'expert': nm_e, 'ant': nm_a}})
    k += 1

  man = {'ckpt': args.ckpt, 'ckpt_step': ant_step, 'seed': args.seed,
         'protocol': (V2.protocol_version(args.p_active) +
                      f'_h{args.horizon}_resetfix_v1'),
         'walker': RP.WALKER, 'base_policy': RP.BASE,
         'render': {'every': args.every, 'fps': args.fps,
                    'size': [args.height, args.width], 'format': ext},
         'pairs_rendered': k, 'resets_used': tries,
         'teacher_deaths_skipped': skipped, 'episodes': rows}
  with open(os.path.join(args.out_dir, 'walk_manifest.json'), 'w') as f:
    json.dump(man, f, indent=2)
  n_e = sum(r['expert']['success'] for r in rows)
  n_a = sum(r['ant']['success'] for r in rows)
  print(f'\n{k} pairs -> {args.out_dir} (expert success {int(n_e)}/{k}, '
        f'ant success {int(n_a)}/{k}, {skipped} teacher deaths skipped)',
        flush=True)


if __name__ == '__main__':
  main()
