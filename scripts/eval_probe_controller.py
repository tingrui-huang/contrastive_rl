"""Offline in-zone evaluation of a probe residual checkpoint.

Metrics per command (y_ref x v_ref), measured ONLY inside the obstacle zone
x in [2.5, 5.5] (whole-episode stats have a ~1.0 floor for side lanes from
the lateral entry transient): y_err p50/p90, vx mean, falls, litter contacts.

Usage:
  python scripts/eval_probe_controller.py \
      [--residual artifacts/probe_controller/phase1e/residual_latest.pkl] \
      [--env litter|plain] [--eps 10] [--out ...json]
Omit --residual to evaluate the frozen base actor alone.
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
from crl import probe                     # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ZONE = (2.5, 5.5)


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default='offline_umaze_bc005_twinmin_s0_50k/'
                                    'checkpoints/best.pkl')
  ap.add_argument('--residual', default=None)
  ap.add_argument('--env', choices=('plain', 'litter'), default='plain')
  ap.add_argument('--eps', type=int, default=10)
  ap.add_argument('--seed', type=int, default=555)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = ckpt_mod.load_checkpoint(args.ckpt)
  params = st.policy_params

  @jax.jit
  def base_act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  res_params, meta = (None, {})
  if args.residual:
    res_params, meta = probe.load_residual(args.residual)
  actor_t, _ = probe.make_residual_networks()
  ctrl = probe.ProbeController(
      lambda o: base_act(jnp.asarray(o)), res_params, actor_t)

  env_name = ('offline_ant_umaze_litter' if args.env == 'litter'
              else 'offline_ant_umaze')
  env = envs_mod.make_env(env_name, cfg, seed=args.seed)
  if hasattr(env, 'collapse_force'):
    env.collapse_force = None

  results = []
  for y_ref in (-probe.LANE_Y, 0.0, probe.LANE_Y):
    for v_ref in (probe.V_SLOW, probe.V_FAST):
      ep_rows = []
      for ep in range(args.eps):
        o = env.reset()
        ze, vz, falls, contacts = [], [], 0, 0
        for t in range(250):
          a, _ = ctrl(o, y_ref, v_ref)
          o, _, _, info = env.step(a)
          if ZONE[0] <= o[0] <= ZONE[1]:
            ze.append(abs(float(o[1]) - y_ref))
            vz.append(float(env._env.data.qvel[0]))
            contacts += int(info.get('pile_contacts', 0) > 0
                            or info.get('rubble_contacts', 0) > 0)
          q = env._env.data.qpos
          if torso_up_z(np.asarray(q)) < 0.3 or float(q[2]) < 0.2:
            falls = 1
            break
          if o[0] >= 6.0:
            break
        ep_rows.append({'y_err_p90': float(np.percentile(ze, 90)) if ze
                        else 2.0,
                        'vx': float(np.mean(vz)) if vz else 0.0,
                        'fall': falls, 'zone_contacts': contacts,
                        'u_side': getattr(env, 'u_side', None)})
      agg = {'y_ref': y_ref, 'v_ref': v_ref,
             'y_err_p90': float(np.mean([r['y_err_p90'] for r in ep_rows])),
             'vx': float(np.mean([r['vx'] for r in ep_rows])),
             'fall_rate': float(np.mean([r['fall'] for r in ep_rows])),
             'zone_contacts': float(np.mean([r['zone_contacts']
                                             for r in ep_rows]))}
      results.append({'agg': agg, 'episodes': ep_rows})
      print(f'y{y_ref:+.1f} v{v_ref:.1f}: y_err_p90 {agg["y_err_p90"]:.2f}  '
            f'vx {agg["vx"]:.2f}  fall {agg["fall_rate"]:.2f}  '
            f'contacts {agg["zone_contacts"]:.1f}')

  if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump({'residual': args.residual, 'meta': {k: v for k, v in
                                                   meta.items()},
               'env': args.env, 'eps': args.eps, 'results': results},
              open(args.out, 'w'), indent=2, default=str)
    print('saved', args.out)


if __name__ == '__main__':
  main()
