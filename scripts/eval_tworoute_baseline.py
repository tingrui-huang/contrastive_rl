"""Authoritative evaluation of a trained baseline on the two-route benchmark.

Protocol: reset() -- ONE canonical start pose for every episode, so nothing
in the initial observation carries route or latent information and the route
is the policy's own decision. The latent is drawn by the env at its natural
Bernoulli(0.30). The policy is the deterministic tanh-mean actor from the
checkpoint, exactly as the repo's diagnosis scripts evaluate.

Note what this costs in variance: with the affordance coin retired and a
deterministic actor, the only remaining variation across episodes is the
+-0.1 qpos reset jitter. A shortcut_rate near 0.5 therefore does NOT mean
the policy chooses 50/50; it can equally mean every episode runs the same
averaged behaviour that gets LABELLED shortcut half the time. Read it with
scripts/probe_tworoute_route_choice.py, not on its own.

Per run reports: success, failure(death), timeout, route distribution
(shortcut/detour/none via info['route']) with a Wilson CI on shortcut_rate,
P(entered hazard), success and death conditioned on the drawn latent, mean
return and steps, and the realised start-yaw range as a protocol receipt.
Writes <ckpt_dir>/eval_tworoute.json and appends a row to
artifacts/tworoute_rockfall_v0/baseline_results.json.

Usage: python scripts/eval_tworoute_baseline.py --ckpt <run>/best.pkl \
           [--n 300] [--method-label crl_s0]
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
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
HORIZON = 400


def _yaw_deg(quat):
  """Torso yaw in degrees from the (w, x, y, z) obs slice. The obs quat is
  unnormalised (INIT_QPOS carries +-0.1 reset noise), so normalise first."""
  q = np.asarray(quat, float)
  w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
  return float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                     1 - 2 * (y * y + z * z))))


def wilson(k, n, z=1.96):
  """Wilson score interval -- shortcut_rate is now the primary readout."""
  if n == 0:
    return [None, None]
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return [round(float(c - h), 4), round(float(c + h), 4)]


def build_policy(ckpt_path, seed=1):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg, seed=seed)
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

  return cfg, act, int(step)


def evaluate(act, n, seed):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=seed)
  rows = []
  for k in range(n):
    o = env.reset()
    #: PROTOCOL ASSERTION, not a measurement: every episode must start from
    #: the one canonical pose. Over 200 resets the native pose yaws within
    #: +-12.6 deg (the INIT_QPOS +-0.1 reset noise) while the retired north
    #: option sat at 77.6-102.4 deg, so 15 deg separates them with headroom
    #: and fires loudly if a route-dependent start pose ever comes back.
    yaw = _yaw_deg(o[3:7])
    assert abs(yaw) <= 15.0, f'episode {k}: non-canonical start yaw {yaw:.1f}'
    u = env.privileged_rockfall_active     # audit only
    ret, info = 0.0, {}
    for t in range(HORIZON):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      ret += float(r)
      if done or r > 0:
        break
    rows.append({'u': bool(u), 'start_yaw_deg': round(float(yaw), 2),
                 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'route': info.get('route'),
                 'return': ret, 'steps': int(t + 1)})
    if (k + 1) % 50 == 0:
      print(f'  {k + 1}/{n}', flush=True)
  return rows


def summarize(rows):
  n = len(rows)

  def m(xs, key):
    return round(float(np.mean([x[key] for x in xs])), 4) if xs else None

  def block(xs):
    return {'n': len(xs), 'success': m(xs, 'success'),
            'failure': m(xs, 'failure'),
            'shortcut_rate': (round(float(np.mean(
                [x['route'] == 'shortcut' for x in xs])), 4) if xs else None),
            'detour_rate': (round(float(np.mean(
                [x['route'] == 'detour' for x in xs])), 4) if xs else None),
            'mean_return': m(xs, 'return'), 'mean_steps': m(xs, 'steps')}

  #: by_heading is GONE: under one canonical pose it emitted an all-None
  #: phantom 'north' row and an 'east' row byte-identical to 'overall'.
  #: shortcut_rate is the decision variable now -- with the affordance coin
  #: removed the route is the policy's own output, so report its CI.
  n_sc = int(sum(r['route'] == 'shortcut' for r in rows))
  out = {'overall': block(rows),
         'timeout': round(float(np.mean(
             [not r['success'] and not r['failure'] for r in rows])), 4),
         'shortcut_rate_ci95': wilson(n_sc, n),
         'start_yaw_deg_range': [round(float(min(r['start_yaw_deg']
                                                 for r in rows)), 2),
                                 round(float(max(r['start_yaw_deg']
                                                 for r in rows)), 2)],
         'by_latent': {'clear': block([r for r in rows if not r['u']]),
                       'active': block([r for r in rows if r['u']])},
         'death_given_active_and_shortcut': m(
             [r for r in rows if r['u'] and r['route'] == 'shortcut'],
             'failure')}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--method-label', default=None)
  args = ap.parse_args()
  _, act, step = build_policy(args.ckpt)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)
  rows = evaluate(act, args.n, args.seed)
  s = summarize(rows)
  print(json.dumps(s, indent=2), flush=True)
  label = args.method_label or os.path.basename(os.path.dirname(args.ckpt))
  rec = {'label': label, 'ckpt': args.ckpt, 'ckpt_step': step,
         'n_eval': args.n, 'eval_seed': args.seed, 'summary': s}
  out_local = os.path.join(os.path.dirname(args.ckpt) or '.',
                           'eval_tworoute.json')
  with open(out_local, 'w') as f:
    json.dump({**rec, 'episodes': rows}, f, indent=2)
  os.makedirs(OUT, exist_ok=True)
  agg_path = os.path.join(OUT, 'baseline_results.json')
  agg = []
  if os.path.exists(agg_path):
    agg = json.load(open(agg_path))
  agg = [r for r in agg if r['label'] != label] + [rec]
  with open(agg_path, 'w') as f:
    json.dump(agg, f, indent=2)
  print('->', out_local, 'and', agg_path, flush=True)


if __name__ == '__main__':
  main()
