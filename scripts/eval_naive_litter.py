"""Behavioural eval of a naive offline-CRL checkpoint in offline_ant_umaze_litter.

Rolls out the learned actor (greedy) over N balanced-U episodes and reports
success (overall + per U), collapse/fall/timeout, litter-corridor lane choice
and speed, and a characterization: fixed-side / fast-middle / middle-slow /
unstable mixture.

Usage: python scripts/eval_naive_litter.py --ckpt <best.pkl> --eps 200
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

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl.d4rl_ant import LITTER_ZONE_X    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ZONE = LITTER_ZONE_X


def torso_up(qpos):
  x, y = qpos[4], qpos[5]
  return 1.0 - 2.0 * (x * x + y * y)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--eps', type=int, default=200)
  ap.add_argument('--seed', type=int, default=20250)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)  # fill dims
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

  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed)
  rows = []
  for i in range(args.eps):
    o = env.reset(u_side=i % 2)
    u = int(env.u_side)
    goal = o[29:31].copy()
    hit, dead_at, fell = 0.0, -1, 0
    zy, zvx = [], []
    for t in range(env.max_episode_steps):
      x, y = float(o[0]), float(o[1])
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, _, info = env.step(a)
      hit = max(hit, float(r))
      if info.get('dead') and dead_at < 0:
        dead_at = t
      q = env._env.data.qpos
      if torso_up(np.asarray(q)) < 0.3 or float(q[2]) < 0.2:
        fell += 1
      if ZONE[0] <= x <= ZONE[1] and abs(y) < 2.0:
        zy.append(y)
        zvx.append(float(env._env.data.qvel[0]))
      if hit > 0 or (dead_at >= 0 and t > dead_at + 3):
        break
    rows.append({'u': u, 'success': hit, 'dead': dead_at >= 0,
                 'fell': fell > 0, 'timeout': hit == 0 and dead_at < 0,
                 'zone_mean_y': float(np.mean(zy)) if zy else np.nan,
                 'zone_mean_vx': float(np.mean(zvx)) if zvx else np.nan,
                 'min_dist': float(np.linalg.norm(o[:2] - goal))})

  def rate(key, sub=None):
    r = [x for x in rows if sub is None or x['u'] == sub]
    return float(np.mean([x[key] for x in r])) if r else 0.0

  zys = np.array([r['zone_mean_y'] for r in rows if not np.isnan(r['zone_mean_y'])])
  zvxs = np.array([r['zone_mean_vx'] for r in rows if not np.isnan(r['zone_mean_vx'])])
  # per-U mean lateral (does the policy pick a side, and does the side flip
  # with U? a fixed-side policy ignores U).
  zy_u0 = np.array([r['zone_mean_y'] for r in rows
                    if r['u'] == 0 and not np.isnan(r['zone_mean_y'])])
  zy_u1 = np.array([r['zone_mean_y'] for r in rows
                    if r['u'] == 1 and not np.isnan(r['zone_mean_y'])])
  frac_center = float(np.mean(np.abs(zys) < 0.5)) if len(zys) else 0.0
  frac_side = 1.0 - frac_center
  med_vx = float(np.median(zvxs)) if len(zvxs) else 0.0

  # characterization
  mean_abs_y = float(np.mean(np.abs(zys))) if len(zys) else 0.0
  side_consistency = (float(abs(np.mean(np.sign(zys)))) if len(zys) else 0.0)
  if frac_side >= 0.6 and med_vx >= 1.0:
    if side_consistency >= 0.5:
      label = 'fixed-side (fast)'
    else:
      label = 'side, U-dependent or mixed sign'
  elif frac_center >= 0.6 and med_vx >= 1.0:
    label = 'fast-middle'
  elif frac_center >= 0.6 and med_vx < 0.9:
    label = 'middle-slow'
  elif frac_center >= 0.5:
    label = 'middle (moderate speed)'
  else:
    label = 'unstable mixture'

  out = {
      'ckpt': args.ckpt, 'step': int(step), 'eps': args.eps,
      'success': rate('success'),
      'success_u0': rate('success', 0), 'success_u1': rate('success', 1),
      'collapse': rate('dead'), 'fall': rate('fell'),
      'timeout': rate('timeout'),
      'corridor_lane_mean_abs_y': mean_abs_y,
      'corridor_lane_center_fraction': frac_center,
      'corridor_lane_side_fraction': frac_side,
      'lane_sign_consistency': side_consistency,
      'lane_mean_y_u0': float(np.mean(zy_u0)) if len(zy_u0) else None,
      'lane_mean_y_u1': float(np.mean(zy_u1)) if len(zy_u1) else None,
      'corridor_speed_median_vx': med_vx,
      'characterization': label,
  }
  print(json.dumps(out, indent=2))
  if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
  return out


if __name__ == '__main__':
  main()
