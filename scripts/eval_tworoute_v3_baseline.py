"""Authoritative evaluation of a trained baseline on the V3 two-route pair.

Port of scripts/eval_tworoute_baseline.py. Protocol: reset() -- ONE
canonical start pose for every episode, so nothing in the initial
observation carries route or latent information and the route is the
policy's own decision. The latent is drawn by the env at its natural
Bernoulli(0.30). The policy is the deterministic tanh-mean actor from the
checkpoint, exactly as the repo's diagnosis scripts evaluate.

Note what this costs in variance: with the affordance coin retired and a
deterministic actor, the only remaining variation across episodes is the
+-0.1 qpos reset jitter (plus the one-sided goal noise). A shortcut_rate
near 0.5 therefore does NOT mean the policy chooses 50/50; it can equally
mean every episode runs the same averaged behaviour that gets LABELLED
shortcut half the time. Read it against a route-choice probe, not on its
own.

New in V3: the DISCOUNTED return per episode -- 0.99**steps if success else
0.0 -- with means overall / by-route / by-latent. This is the pair's
manipulated variable, scored against the pre-registered driver references
(sparse, both variants: always-shortcut 0.70 / always-detour 0.96 / oracle
0.988; discounted tr: shortcut 0.146 / detour 0.185 best-blind / oracle
0.201; br: shortcut 0.323 best-blind / detour 0.100 / oracle 0.353).

Per run reports: success, failure(death), timeout, route distribution
(shortcut/detour/none via info['route']) with a Wilson CI on shortcut_rate,
P(entered hazard), success and death conditioned on the drawn latent, mean
sparse and discounted return and steps, and the realised start-yaw range as
a protocol receipt. Writes <ckpt_dir>/eval_tworoute_v3.json and appends a
row to artifacts/tworoute_rockfall_v3/<variant>/baseline_results.json
(--out-dir / --results-root redirect both, e.g. for harness smoke tests
whose numbers must not land in artifacts/).

Usage: python scripts/eval_tworoute_v3_baseline.py --variant tr \
           --ckpt <run>/best.pkl [--n 300] [--method-label v3tr_crl_s0]
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

OUT_ROOT = 'artifacts/tworoute_rockfall_v3'
HORIZON = 400
GAMMA = 0.99

#: Pre-registered driver reference numbers. Keep in sync with
#: collect_tworoute_v3_dataset.py / tworoute_v3_causal_audit.py.
SPARSE_REFS = {'always_shortcut': 0.70, 'always_detour': 0.96,
               'oracle': 0.988}
DISCOUNTED_REFS = {
    'tr': {'shortcut': 0.146, 'detour': 0.185, 'oracle': 0.201,
           'best_blind': 'detour'},
    'br': {'shortcut': 0.323, 'detour': 0.100, 'oracle': 0.353,
           'best_blind': 'shortcut'},
}


def env_id(variant):
  return f'offline_ant_umaze_tworoute_rockfall_v3{variant}'


def _yaw_deg(quat):
  """Torso yaw in degrees from the (w, x, y, z) obs slice. The obs quat is
  unnormalised (INIT_QPOS carries +-0.1 reset noise), so normalise first."""
  q = np.asarray(quat, float)
  w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
  return float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                     1 - 2 * (y * y + z * z))))


def wilson(k, n, z=1.96):
  """Wilson score interval -- shortcut_rate is the primary readout."""
  if n == 0:
    return [None, None]
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return [round(float(c - h), 4), round(float(c + h), 4)]


def build_policy(ckpt_path, variant, seed=1, horizon=HORIZON):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = horizon
  envs_mod.make_env(env_id(variant), cfg, seed=seed)
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


def evaluate(act, variant, n, seed, horizon=HORIZON):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env(env_id(variant), cfg, seed=seed)
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
    for t in range(horizon):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      ret += float(r)
      if done or r > 0:
        break
    steps = int(t + 1)
    success = bool(info.get('success'))
    rows.append({'u': bool(u), 'start_yaw_deg': round(float(yaw), 2),
                 'success': success,
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'route': info.get('route'),
                 'return': ret, 'steps': steps,
                 #: the pair's manipulated variable (see module doc).
                 'discounted': (round(float(GAMMA ** steps), 6)
                                if success else 0.0)})
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
            'mean_return': m(xs, 'return'),
            'mean_discounted': m(xs, 'discounted'),
            'mean_steps': m(xs, 'steps')}

  #: shortcut_rate is the decision variable -- with the affordance coin
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
         #: discounted by realized route (0.0 on failure, mean over the
         #: route group); 'none' = never-committed episodes.
         'mean_discounted_by_route': {
             'shortcut': m([r for r in rows if r['route'] == 'shortcut'],
                           'discounted'),
             'detour': m([r for r in rows if r['route'] == 'detour'],
                         'discounted'),
             'none': m([r for r in rows if r['route'] is None],
                       'discounted')},
         'death_given_active_and_shortcut': m(
             [r for r in rows if r['u'] and r['route'] == 'shortcut'],
             'failure')}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=['tr', 'br'], required=True)
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--method-label', default=None)
  #: must match the horizon the checkpoint was TRAINED with.
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=None,
                  help='dir for eval_tworoute_v3.json (default: ckpt dir)')
  ap.add_argument('--results-root', default=OUT_ROOT,
                  help='root holding <variant>/baseline_results.json')
  args = ap.parse_args()
  _, act, step = build_policy(args.ckpt, args.variant, horizon=args.horizon)
  print(f'ckpt {args.ckpt} @ step {step} | variant {args.variant}',
        flush=True)
  rows = evaluate(act, args.variant, args.n, args.seed, horizon=args.horizon)
  s = summarize(rows)
  print(json.dumps(s, indent=2), flush=True)
  label = args.method_label or os.path.basename(os.path.dirname(args.ckpt))
  rec = {'label': label, 'variant': args.variant,
         'env': env_id(args.variant), 'ckpt': args.ckpt, 'ckpt_step': step,
         'n_eval': args.n, 'eval_seed': args.seed,
         'horizon': args.horizon,
         'reference_numbers': {
             'sparse_success': SPARSE_REFS,
             'discounted_gamma_0.99': DISCOUNTED_REFS[args.variant]},
         'summary': s}
  out_dir = args.out_dir or (os.path.dirname(args.ckpt) or '.')
  os.makedirs(out_dir, exist_ok=True)
  out_local = os.path.join(out_dir, 'eval_tworoute_v3.json')
  with open(out_local, 'w') as f:
    json.dump({**rec, 'episodes': rows}, f, indent=2)
  agg_dir = os.path.join(args.results_root, args.variant)
  os.makedirs(agg_dir, exist_ok=True)
  agg_path = os.path.join(agg_dir, 'baseline_results.json')
  agg = []
  if os.path.exists(agg_path):
    agg = json.load(open(agg_path))
  agg = [r for r in agg if r['label'] != label] + [rec]
  with open(agg_path, 'w') as f:
    json.dump(agg, f, indent=2)
  print('->', out_local, 'and', agg_path, flush=True)


if __name__ == '__main__':
  main()
