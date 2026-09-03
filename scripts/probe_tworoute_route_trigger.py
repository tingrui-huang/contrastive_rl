"""What actually decides the coin-free learner's route?

P1/P2 (scripts/probe_tworoute_route_choice.py) established the paradox: the
actor's ACTION is an average of the two dataset modes on 11 of 12 early steps
(never at either mode), yet its BEHAVIOUR is cleanly bimodal -- each episode
commits to a route, the wall sector is empty and jamming is rare. The
explanation on the table is that the averaged action leaves the ant on an
unstable ridge, and the +-0.1 qpos reset jitter -- the only variation left
once the heading coin is gone -- is amplified into a committed route by a
fork that separates ~25x faster than a lane.

That explanation is testable and this script tests it: if the reset jitter
decides the route, then the t=0 OBSERVATION must predict the realised route.
If it does not, something later in the episode does, and the story is wrong.

Measured, on rollouts of the trained policy:
  - per-dim d' between the shortcut-route and detour-route groups at t=0,
    against a permutation null (the same instrument the dataset leak check
    uses), so a nonzero d' cannot be read off sampling noise;
  - a leave-one-out logistic decoder of route from obs[0], reported as
    balanced accuracy against its own permutation null;
  - REPLAY DETERMINISM: re-running the same reset draw must reproduce the
    same route (it is a deterministic actor), which is what makes the jitter
    the only candidate cause.

Note the sign convention: predicting the route from t=0 here is NOT a latent
leak. The latent is drawn independently of the reset jitter, so obs[0] can
carry route information while carrying no information about rockfall_active
-- and the eval confirms the route is latent-independent (shortcut rate 0.272
clear vs 0.265 active). What it would show is that the route is decided by
noise the policy cannot control, rather than chosen.

Usage:
  python scripts/probe_tworoute_route_trigger.py --ckpt <run>/final.pkl [--n 120]
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

  return act, int(step)


def rollout(env, act, o):
  info = {}
  for _ in range(HORIZON):
    o, r, done, info = env.step(np.asarray(act(jnp.asarray(o[None]))[0]))
    if done or r > 0:
      break
  return info.get('route')


def collect(act, n, seed):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=seed)
  o0s, routes = [], []
  for k in range(n):
    o = env.reset()
    o0s.append(o.copy())
    routes.append(rollout(env, act, o))
    if (k + 1) % 40 == 0:
      print(f'  {k + 1}/{n}', flush=True)
  return np.asarray(o0s), np.asarray([r if r else 'none' for r in routes])


def dprime_test(o0, y, B=2000, seed=0):
  m = y == 'shortcut'
  keep = np.isin(y, ('shortcut', 'detour'))
  o0, m = o0[keep], m[keep]

  def dp(g0, g1):
    pooled = np.sqrt((g0.var(0, ddof=1) + g1.var(0, ddof=1)) / 2.0) + 1e-9
    return np.abs(g0.mean(0) - g1.mean(0)) / pooled

  d = dp(o0[m], o0[~m])
  rng = np.random.default_rng(seed)
  k = int(m.sum())
  null = np.array([dp(o0[p][:k], o0[p][k:]).max()
                   for p in (rng.permutation(len(o0)) for _ in range(B))])
  p95 = float(np.percentile(null, 95))
  return {'n_shortcut': int(m.sum()), 'n_detour': int((~m).sum()),
          'max_dprime': round(float(d.max()), 4),
          'argmax_dim': int(d.argmax()),
          'top5_dims': [[int(i), round(float(d[i]), 3)]
                        for i in np.argsort(-d)[:5]],
          'null_p95': round(p95, 4),
          'n_dims_above_null_p95': int((d > p95).sum()),
          'p_value': round(float((null >= d.max()).mean()), 4),
          'route_predictable_from_t0': bool(d.max() > p95)}


def loo_decoder(o0, y, B=200, seed=0):
  """Leave-one-out balanced accuracy of a ridge-regularised linear decoder of
  route from obs[0], against a label-permutation null. Plain numpy: the repo
  has no sklearn dependency and this must not add one."""
  keep = np.isin(y, ('shortcut', 'detour'))
  X, t = o0[keep], (y[keep] == 'shortcut').astype(float)
  #: drop constant dims and standardise -- the goal block is identical
  #: across episodes up to the goal draw, and zero-variance columns blow up
  sd = X.std(0)
  X = X[:, sd > 1e-8]
  X = (X - X.mean(0)) / (X.std(0) + 1e-9)
  X = np.c_[X, np.ones(len(X))]
  lam = 10.0

  def bal_acc(target):
    pred = np.empty(len(X))
    for i in range(len(X)):
      k = np.ones(len(X), bool)
      k[i] = False
      A = X[k].T @ X[k] + lam * np.eye(X.shape[1])
      w = np.linalg.solve(A, X[k].T @ (target[k] - target[k].mean()))
      pred[i] = X[i] @ w + target[k].mean()
    yh = (pred > 0.5).astype(float)
    p = target == 1
    if p.sum() == 0 or (~p).sum() == 0:
      return float('nan')
    return float(0.5 * ((yh[p] == 1).mean() + (yh[~p] == 0).mean()))

  obs_acc = bal_acc(t)
  rng = np.random.default_rng(seed)
  null = np.array([bal_acc(rng.permutation(t)) for _ in range(B)])
  return {'n': int(len(X)), 'n_features': int(X.shape[1] - 1),
          'balanced_accuracy': round(obs_acc, 4),
          'null_mean': round(float(np.nanmean(null)), 4),
          'null_p95': round(float(np.nanpercentile(null, 95)), 4),
          'p_value': round(float((null >= obs_acc).mean()), 4),
          'beats_chance': bool(obs_acc > np.nanpercentile(null, 95))}


def replay_determinism(act, seed, n=20):
  """Same reset draw -> same route? Two independent env instances on the same
  seed replay the identical reset sequence, so any disagreement would mean the
  route is NOT a function of the initial state."""
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  eA = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg, seed=seed)
  eB = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg, seed=seed)
  agree, rows = 0, []
  for _ in range(n):
    ra = rollout(eA, act, eA.reset())
    rb = rollout(eB, act, eB.reset())
    agree += int(ra == rb)
    rows.append([ra, rb])
  return {'n': n, 'agreement': round(agree / n, 4), 'pairs': rows}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--n', type=int, default=120)
  ap.add_argument('--seed', type=int, default=1234)
  ap.add_argument('--label', default=None)
  args = ap.parse_args()
  act, step = build_policy(args.ckpt)
  label = args.label or os.path.basename(os.path.dirname(args.ckpt))
  print(f'{label}: ckpt {args.ckpt} @ step {step}', flush=True)

  o0, y = collect(act, args.n, args.seed)
  counts = {k: int((y == k).sum()) for k in ('shortcut', 'detour', 'none')}
  print('routes:', counts, flush=True)
  dt = dprime_test(o0, y)
  print('d-prime test:', json.dumps(dt, indent=2), flush=True)
  dec = loo_decoder(o0, y)
  print('LOO decoder:', json.dumps(dec, indent=2), flush=True)
  det = replay_determinism(act, args.seed + 77)
  print('replay determinism:', json.dumps(
      {k: v for k, v in det.items() if k != 'pairs'}, indent=2), flush=True)

  os.makedirs(OUT, exist_ok=True)
  path = os.path.join(OUT, f'route_trigger_probe_{label}.json')
  with open(path, 'w') as f:
    json.dump({'label': label, 'ckpt': args.ckpt, 'ckpt_step': step,
               'route_counts': counts, 'dprime_test': dt,
               'loo_decoder': dec, 'replay_determinism': det}, f, indent=2)
  print('->', path, flush=True)


if __name__ == '__main__':
  main()
