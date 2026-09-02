"""Authoritative evaluation of a trained baseline on the two-route benchmark.

Protocol: reset(heading='random') -- the 50/50 route-affordance coin,
independent of the hidden latent (which the env draws at its natural
Bernoulli(0.30)). The policy is the deterministic tanh-mean actor from the
checkpoint, exactly as the repo's diagnosis scripts evaluate.

Per run reports: success, failure(death), timeout, route distribution
(shortcut/detour/none via info['route']), P(entered hazard), success and
death conditioned on the drawn latent, mean return and steps -- plus the same
metrics split by the initial heading (the affordance the learner actually
sees). Writes <ckpt_dir>/eval_tworoute.json and appends a row to
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
    o = env.reset(heading='random')
    #: which side of the coin this episode saw (from the initial quat: the
    #: north yaw makes |w| ~ cos(45) < 0.85; east keeps |w| ~ 1).
    head = 'north' if abs(float(o[3])) < 0.85 else 'east'
    u = env.privileged_rockfall_active     # audit only
    ret, info = 0.0, {}
    for t in range(HORIZON):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      ret += float(r)
      if done or r > 0:
        break
    rows.append({'u': bool(u), 'heading': head,
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

  out = {'overall': block(rows),
         'timeout': round(float(np.mean(
             [not r['success'] and not r['failure'] for r in rows])), 4),
         'by_heading': {h: block([r for r in rows if r['heading'] == h])
                        for h in ('north', 'east')},
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
