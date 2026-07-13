"""Critic route-ranking probe (offline-diagnosis item 4).

At replay states whose commanded episode goal is wall-blocked (BFS cell
distance >= 2 AND straight line crosses a wall), compare critic scores
f(s, a, g) for actions whose SHORT-TERM PHYSICAL EFFECT is known:

  euclidean-direct  measured 3-step XY displacement toward the goal
                    (through the wall direction)
  geodesic-correct  displacement toward the next cell on the BFS shortest
                    path to the goal
  random/other      neither
  actor             the policy's own (mode) action at s

Actions are NOT invented from torques: every candidate (uniform random +
actions drawn from real replay transitions) is rolled from an exact MuJoCo
restore of s for 3 env steps and classified by its measured displacement.
Probe states require angle(euclid_dir, geodesic_dir) > 60 deg so the two
hypotheses are distinguishable.
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.d4rl_ant import U_MAZE, SCALING

OPEN = [(r, c) for r in range(len(U_MAZE)) for c in range(len(U_MAZE[0]))
        if U_MAZE[r][c] != 1]
WALL = {(r, c) for r in range(len(U_MAZE)) for c in range(len(U_MAZE[0]))
        if U_MAZE[r][c] == 1}
CENTERS = np.array([[c * SCALING - 4.0, r * SCALING - 4.0] for r, c in OPEN])

REPLAY = ('d4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k/'
          'checkpoints/replay.npz')
CKPTS = {
    'best_235200': ('d4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k/'
                    'checkpoints/best.pkl'),
    'final_252000': ('d4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k/'
                     'gates/gate_250000.pkl'),
}
H = 3                 # rollout horizon (env steps of 0.1 s) per candidate
N_STATES = 60
N_RAND, N_REPLAY_ACT = 48, 48
COS_THR = 0.7
MIN_DISP = 0.03       # metres over H steps; below => 'immobile' (excluded)


def bfs_next(a, b):
  """BFS distance and the next cell on a shortest path a->b."""
  if a == b:
    return 0, b
  q, seen = deque([(a, [a])]), {a}
  while q:
    u, path = q.popleft()
    for v in ((u[0]+1, u[1]), (u[0]-1, u[1]), (u[0], u[1]+1), (u[0], u[1]-1)):
      if v == b:
        return len(path), (path + [b])[1]
      if v in set(OPEN) and v not in seen:
        seen.add(v)
        q.append((v, path + [v]))
  return -1, a


def nearest_open(xy):
  return OPEN[int(np.argmin(np.linalg.norm(CENTERS - xy[None], axis=1)))]


def los_blocked(p, q, step=0.25):
  n = max(2, int(np.ceil(np.linalg.norm(q - p) / step)))
  pts = p[None] + np.linspace(0, 1, n)[:, None] * (q - p)[None]
  cols = np.round((pts[:, 0] + 4.0) / SCALING).astype(int)
  rows = np.round((pts[:, 1] + 4.0) / SCALING).astype(int)
  return any((r, c) in WALL or not (0 <= r < 5 and 0 <= c < 5)
             for r, c in zip(rows, cols))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='artifacts/ant_route_diagnosis')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  with np.load(REPLAY) as d:
    obs = d['obs'][:int(d['num_eps'])].copy()
    act = d['act'][:int(d['num_eps'])].copy()
  n_eps, L = obs.shape[0], obs.shape[1]

  # ---- probe-state selection: wall-blocked commanded goals, upright ant ----
  cand = []
  for k in range(16, n_eps):                       # skip random warmup eps
    goal = obs[k, 0, 29:58]
    gcell = nearest_open(goal[:2].astype(float))
    for t in range(50, L - 50, 97):
      s = obs[k, t, :29].astype(float)
      if s[2] < 0.35:                              # fallen: skip
        continue
      scell = nearest_open(s[:2])
      d_bfs, nxt = bfs_next(scell, gcell)
      if d_bfs < 2 or not los_blocked(s[:2], goal[:2].astype(float)):
        continue
      e_dir = goal[:2] - s[:2]
      e_dir = e_dir / np.linalg.norm(e_dir)
      g_dir = (CENTERS[OPEN.index(nxt)] - s[:2])
      g_dir = g_dir / np.linalg.norm(g_dir)
      if float(np.dot(e_dir, g_dir)) > 0.5:        # need >60 deg separation
        continue
      cand.append((k, t, s, goal.astype(float), e_dir, g_dir, d_bfs))
  rng.shuffle(cand)
  probes = cand[:N_STATES]
  print(f'probe states: {len(probes)} (from {len(cand)} candidates)',
        flush=True)
  assert probes, 'no wall-blocked probe states found'

  # ---- env sim for exact restore ----
  cfg = Config(env_name='d4rl_ant_umaze_gfull')
  env = envs_mod.make_env('d4rl_ant_umaze_gfull', cfg, seed=7)
  u = env._env.unwrapped

  def restore(s):
    u.data.qpos[:2] = s[:2]
    u.data.qpos[2:] = s[2:15]
    u.data.qvel[:] = s[15:29]
    mujoco.mj_forward(u.model, u.data)

  def roll(s, a):
    restore(s)
    for _ in range(H):
      u.step(a)
    return np.asarray(u.data.qpos[:2], float) - s[:2]

  # ---- networks ----
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)

  flat_act = act[16:, :-1].reshape(-1, 8)

  out = {'n_probe_states': len(probes), 'horizon_steps': H,
         'cos_threshold': COS_THR, 'per_ckpt': {}}
  for tag, ck in CKPTS.items():
    step, st = ckpt_mod.load_checkpoint(ck)

    @jax.jit
    def qdiag(og, a, p=st.q_params):
      return jnp.diag(nets.q_network.apply(p, og, a))

    @jax.jit
    def pmode(og, p=st.policy_params):
      return jnp.tanh(nets.policy_network.apply(p, og).loc)

    rows = []
    for (k, t, s, goal, e_dir, g_dir, d_bfs) in probes:
      A = np.concatenate([
          rng.uniform(-1, 1, (N_RAND, 8)),
          flat_act[rng.integers(0, len(flat_act), N_REPLAY_ACT)],
      ]).astype(np.float32)
      a_actor = np.asarray(
          pmode(jnp.asarray(np.concatenate([s, goal])[None]))[0], np.float32)
      A = np.concatenate([A, a_actor[None]])
      disp = np.stack([roll(s, a) for a in A])
      mag = np.linalg.norm(disp, axis=1)
      moving = mag > MIN_DISP
      dirs = disp / np.maximum(mag[:, None], 1e-9)
      cos_e = dirs @ e_dir
      cos_g = dirs @ g_dir
      cat = np.full(len(A), 'other', dtype=object)
      cat[~moving] = 'immobile'
      cat[moving & (cos_e > COS_THR) & (cos_g <= COS_THR)] = 'euclid'
      cat[moving & (cos_g > COS_THR) & (cos_e <= COS_THR)] = 'geodesic'
      og = np.repeat(np.concatenate([s, goal])[None], len(A),
                     axis=0).astype(np.float32)
      q = np.asarray(qdiag(jnp.asarray(og), jnp.asarray(A)))
      row = {'ep': k, 't': t, 'bfs': d_bfs,
             'actor_cat': str(cat[-1]), 'actor_q': float(q[-1]),
             'actor_disp': float(mag[-1])}
      for c in ('euclid', 'geodesic', 'other', 'immobile'):
        m = cat[:-1] == c
        row[f'n_{c}'] = int(m.sum())
        row[f'q_{c}_mean'] = float(q[:-1][m].mean()) if m.any() else None
        row[f'q_{c}_max'] = float(q[:-1][m].max()) if m.any() else None
      # rank correlation of score with each direction (moving candidates)
      from scipy.stats import spearmanr
      mv = moving[:-1]
      if mv.sum() > 8:
        row['sp_q_cos_euclid'] = float(spearmanr(q[:-1][mv],
                                                 cos_e[:-1][mv]).statistic)
        row['sp_q_cos_geodesic'] = float(spearmanr(q[:-1][mv],
                                                   cos_g[:-1][mv]).statistic)
      rows.append(row)

    both = [r for r in rows if r['n_euclid'] and r['n_geodesic']]
    agg = {
        'ckpt': ck, 'step': int(step),
        'states_with_both_cats': len(both),
        'mean_q_geodesic': float(np.mean([r['q_geodesic_mean'] for r in both])),
        'mean_q_euclid': float(np.mean([r['q_euclid_mean'] for r in both])),
        'mean_q_other': float(np.mean([r['q_other_mean'] for r in rows
                                       if r['q_other_mean'] is not None])),
        'frac_geodesic_beats_euclid_mean': float(np.mean(
            [r['q_geodesic_mean'] > r['q_euclid_mean'] for r in both])),
        'frac_geodesic_beats_euclid_max': float(np.mean(
            [r['q_geodesic_max'] > r['q_euclid_max'] for r in both])),
        'mean_sp_q_cos_euclid': float(np.mean(
            [r['sp_q_cos_euclid'] for r in rows if 'sp_q_cos_euclid' in r])),
        'mean_sp_q_cos_geodesic': float(np.mean(
            [r['sp_q_cos_geodesic'] for r in rows
             if 'sp_q_cos_geodesic' in r])),
        'actor_action': {
            'cat_counts': {c: int(sum(r['actor_cat'] == c for r in rows))
                           for c in ('euclid', 'geodesic', 'other',
                                     'immobile')},
            'mean_disp_m': float(np.mean([r['actor_disp'] for r in rows])),
            'mean_q': float(np.mean([r['actor_q'] for r in rows])),
        },
        'rows': rows,
    }
    out['per_ckpt'][tag] = agg
    print(tag, json.dumps({k: v for k, v in agg.items() if k != 'rows'},
                          indent=1), flush=True)

  json.dump(out, open(os.path.join(args.out, 'critic_route_ranking.json'),
                      'w'), indent=2)
  print('saved', os.path.join(args.out, 'critic_route_ranking.json'))


if __name__ == '__main__':
  main()
