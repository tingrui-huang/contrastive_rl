"""Fixed-seed 100-episode deterministic evaluation of two checkpoints.

Identical start/goal sequences for both checkpoints (same env seed; the env
consumes reset randomness independently of the policy). Reports success with
bootstrap 95% CI, final/min distance stats, goal-directed velocity (dt=0.1),
fall rate, and success split by BFS cell distance (easy vs detour pairs).
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.d4rl_ant import U_MAZE, SCALING

DT = 0.1
FALL_Z = 0.3
N_EPS = 100
ENV_SEED = 12345


def bfs_dist(a, b):
  open_cells = {(r, c) for r in range(len(U_MAZE))
                for c in range(len(U_MAZE[0])) if U_MAZE[r][c] != 1}
  if a == b:
    return 0
  q, seen = deque([(a, 0)]), {a}
  while q:
    (r, c), d = q.popleft()
    for nr, nc in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
      if (nr, nc) == b:
        return d + 1
      if (nr, nc) in open_cells and (nr, nc) not in seen:
        seen.add((nr, nc))
        q.append(((nr, nc), d + 1))
  return -1


def xy_to_cell(xy):
  return (int(round((xy[1] + 4.0) / SCALING)),
          int(round((xy[0] + 4.0) / SCALING)))


def evaluate(ckpt, n_eps):
  cfg = Config(env_name='d4rl_ant_umaze_gfull')
  env = envs_mod.make_env('d4rl_ant_umaze_gfull', cfg, seed=ENV_SEED)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  step, st = ckpt_mod.load_checkpoint(ckpt)

  @jax.jit
  def _mode(obs):
    return jnp.tanh(nets.policy_network.apply(st.policy_params, obs).loc)

  u = env._env.unwrapped
  rows = []
  for ep in range(n_eps):
    obs = env.reset()
    start = obs[:2].copy()
    goal = obs[29:31].copy()
    d_prev = float(np.linalg.norm(start - goal))
    d0, dmin, succ, fall = d_prev, d_prev, 0.0, 0
    gvel = []
    for t in range(env.max_episode_steps):
      a = np.asarray(_mode(jnp.asarray(obs[None]))[0])
      obs, r, _, _ = env.step(a)
      d = float(np.linalg.norm(obs[:2] - goal))
      gvel.append((d_prev - d) / DT)
      dmin = min(dmin, d)
      succ = max(succ, float(r))
      fall += float(u.data.qpos[2]) < FALL_Z
      d_prev = d
    rows.append(dict(
        ep=ep, success=succ, d0=d0, dmin=dmin, final=d_prev,
        gvel=float(np.mean(gvel)), fall_frac=fall / env.max_episode_steps,
        bfs=bfs_dist(xy_to_cell(start), xy_to_cell(goal))))
    if ep % 20 == 0:
      print(f'  {ckpt} ep {ep}: succ so far '
            f'{np.mean([r["success"] for r in rows]):.2f}', flush=True)
  return step, rows


def boot_ci(x, n=2000, seed=0):
  x = np.asarray(x, float)
  r = np.random.default_rng(seed)
  m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
  return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def agg(step, rows):
  s = np.array([r['success'] for r in rows])
  by_bfs = {}
  for r in rows:
    by_bfs.setdefault(r['bfs'], []).append(r['success'])
  return {
      'step': step, 'episodes': len(rows),
      'success_rate': float(s.mean()), 'success_ci95': boot_ci(s),
      'final_dist_mean': float(np.mean([r['final'] for r in rows])),
      'final_dist_median': float(np.median([r['final'] for r in rows])),
      'min_dist_mean': float(np.mean([r['dmin'] for r in rows])),
      'min_dist_median': float(np.median([r['dmin'] for r in rows])),
      'goal_vel_mean_mps': float(np.mean([r['gvel'] for r in rows])),
      'fall_rate_mean': float(np.mean([r['fall_frac'] for r in rows])),
      'success_by_bfs_dist': {str(k): {'n': len(v),
                                       'success': float(np.mean(v))}
                              for k, v in sorted(by_bfs.items())},
      'easy_success (bfs<=2)': float(np.mean(
          [r['success'] for r in rows if r['bfs'] <= 2])),
      'detour_success (bfs>=3)': float(np.mean(
          [r['success'] for r in rows if r['bfs'] >= 3]))
      if any(r['bfs'] >= 3 for r in rows) else None,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--best', default='d4rl_ant_umaze_gfull_gfull29_alpha0_'
                  '4actor_s0_250k/checkpoints/best.pkl')
  ap.add_argument('--final', default='d4rl_ant_umaze_gfull_gfull29_alpha0_'
                  '4actor_s0_250k/gates/gate_250000.pkl')
  ap.add_argument('--out', default='artifacts/audit_4actor_250k')
  ap.add_argument('--n_eps', type=int, default=N_EPS)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  out = {}
  for name, ck in (('best', args.best), ('final', args.final)):
    step, rows = evaluate(ck, args.n_eps)
    out[name] = {'ckpt': ck, **agg(step, rows), 'episodes_detail': rows}
    print(name, json.dumps({k: v for k, v in out[name].items()
                            if k != 'episodes_detail'}, indent=1))
  # same-seed pairing: per-episode delta
  bs = [r['success'] for r in out['best']['episodes_detail']]
  fs = [r['success'] for r in out['final']['episodes_detail']]
  out['paired'] = {'best_minus_final_success': float(np.mean(bs) - np.mean(fs)),
                   'delta_ci95': boot_ci(np.array(bs) - np.array(fs))}
  json.dump(out, open(os.path.join(args.out, 'fixed_eval_100.json'), 'w'),
            indent=2)
  print('PAIRED:', out['paired'])
  print('saved', os.path.join(args.out, 'fixed_eval_100.json'))


if __name__ == '__main__':
  main()
