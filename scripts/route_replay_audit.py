"""Part 2 of the route diagnosis: replay route-support audit.

For every episode in a run's replay snapshot, maps the ant XY trajectory to
maze BFS cells and reports route-level content; then estimates the BFS
distribution of the positive (state, future-goal) pairs the critic actually
trains on, by drawing batches from the REAL TrajectoryBuffer.sample() code
path (same discount, same Gumbel-max relabeling) on the loaded snapshot.

Definitions (U-maze, 7 open cells, SCALING=4):
  cell            nearest OPEN cell center to an XY position
  BFS span        max BFS distance between any two cells visited in an episode
  boundary cross  count of consecutive-step cell changes
  corner passage  episode visits a turning cell AND both its arms
                  ((1,3): (1,2)+(2,3);  (3,3): (2,3)+(3,2))
  detour segment  exists t1<t2 whose straight XY segment crosses a wall cell
                  (goedesic != straight line between two visited states)
  categories      local: span 0 | short: span 1 | route: span >= 2
"""
import argparse
import json
import os
import sys
from collections import deque
from itertools import combinations

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from crl.replay import TrajectoryBuffer
from crl.d4rl_ant import U_MAZE, SCALING

OPEN = [(r, c) for r in range(len(U_MAZE)) for c in range(len(U_MAZE[0]))
        if U_MAZE[r][c] != 1]
WALL = {(r, c) for r in range(len(U_MAZE)) for c in range(len(U_MAZE[0]))
        if U_MAZE[r][c] == 1}
CENTERS = np.array([[c * SCALING - 4.0, r * SCALING - 4.0] for r, c in OPEN])
CORNERS = {(1, 3): [(1, 2), (2, 3)], (3, 3): [(2, 3), (3, 2)]}


def bfs_table():
  d = {}
  for a in OPEN:
    dist = {a: 0}
    q = deque([a])
    while q:
      u = q.popleft()
      for v in ((u[0]+1, u[1]), (u[0]-1, u[1]), (u[0], u[1]+1), (u[0], u[1]-1)):
        if v in set(OPEN) and v not in dist:
          dist[v] = dist[u] + 1
          q.append(v)
    for b, dd in dist.items():
      d[(a, b)] = dd
  return d


BFS = bfs_table()


def xy_to_open_cell_idx(xy):
  """Nearest open-cell index for [N,2] positions."""
  d = np.linalg.norm(xy[:, None, :] - CENTERS[None, :, :], axis=2)
  return np.argmin(d, axis=1)


def los_blocked(p, q, step=0.25):
  """Straight segment p->q crosses a wall cell (grid-rounded sampling)."""
  n = max(2, int(np.ceil(np.linalg.norm(q - p) / step)))
  pts = p[None] + np.linspace(0, 1, n)[:, None] * (q - p)[None]
  cols = np.round((pts[:, 0] + 4.0) / SCALING).astype(int)
  rows = np.round((pts[:, 1] + 4.0) / SCALING).astype(int)
  for r, c in zip(rows, cols):
    if (r, c) in WALL or not (0 <= r < 5 and 0 <= c < 5):
      return True
  return False


def episode_stats(xy):
  """xy: [L, 2] trajectory. Returns route-content dict."""
  ci = xy_to_open_cell_idx(xy)
  cells = [OPEN[i] for i in ci]
  uniq = sorted(set(cells))
  span = max((BFS[(a, b)] for a, b in combinations(uniq, 2)), default=0)
  crossings = int(np.sum(ci[1:] != ci[:-1]))
  corner = False
  for k, arms in CORNERS.items():
    if k in uniq and all(a in uniq for a in arms):
      corner = True
  # detour: any wall-blocked straight line between two visited states.
  # sufficient check on cell-center pairs of visited cells (wall-blocked
  # LOS between cells implies blocked LOS between states in those cells'
  # interiors for this maze); verify with actual state pair.
  detour = False
  for a, b in combinations(uniq, 2):
    if BFS[(a, b)] >= 2:
      ta = np.where(ci == OPEN.index(a))[0][0]
      tb = np.where(ci == OPEN.index(b))[0][0]
      if los_blocked(xy[ta], xy[tb]):
        detour = True
        break
  disp = BFS[(cells[0], cells[-1])]
  return dict(unique_cells=len(uniq), bfs_span=span,
              boundary_crossings=crossings,
              crosses_2_boundaries=crossings >= 2,
              corner_passage=corner, detour_segment=detour,
              start_end_bfs=disp)


def audit_replay(path, tag, ep_len=701, warmup_eps=16):
  with np.load(path) as d:
    obs = d['obs'].copy()
    n = int(d['num_eps'])
  obs = obs[:n]
  rows = []
  for k in range(n):
    st = episode_stats(obs[k, :, :2].astype(np.float64))
    st.update(ep=k, actor=k % 4, phase_50k=int((k * ep_len) // 50_000),
              random_warmup=k < warmup_eps, run=tag)
    rows.append(st)
  return rows, obs


def fractions(rows, key_steps=700):
  n = len(rows)
  spans = np.array([r['bfs_span'] for r in rows])
  det = np.array([r['detour_segment'] for r in rows])
  f = {
      'episodes': n, 'transitions': n * key_steps,
      'local_span0': float(np.mean(spans == 0)),
      'short_span1': float(np.mean(spans == 1)),
      'route_span_ge2': float(np.mean(spans >= 2)),
      'detour_containing': float(np.mean(det)),
      'corner_passage': float(np.mean([r['corner_passage'] for r in rows])),
      'crosses_2_boundaries': float(np.mean(
          [r['crosses_2_boundaries'] for r in rows])),
      'mean_unique_cells': float(np.mean([r['unique_cells'] for r in rows])),
      'mean_start_end_bfs': float(np.mean([r['start_end_bfs'] for r in rows])),
      'span_hist': {str(s): int(np.sum(spans == s))
                    for s in sorted(set(spans.tolist()))},
  }
  return f


def sampler_estimate(path, n_pairs=102_400, batch=512, seed=0):
  """BFS distribution of positives drawn by the REAL sampler on a snapshot."""
  with np.load(path) as d:
    n = int(d['num_eps'])
    L = d['obs'].shape[1]
    W = d['obs'].shape[2]
    A = d['act'].shape[2]
  buf = TrajectoryBuffer(
      capacity_steps=1_000_000, ep_len_obs=L, full_obs_dim=W, action_dim=A,
      obs_dim=29, start_index=0, end_index=-1, discount=0.99, seed=seed,
      goal_indices=tuple(range(29)))
  buf.load(path)
  assert buf._num_eps == n
  s_cells, g_cells, blocked = [], [], []
  for _ in range(n_pairs // batch):
    tr = buf.sample(batch)
    sxy = tr.observation[:, :2].astype(np.float64)
    gxy = tr.observation[:, 29:31].astype(np.float64)
    si = xy_to_open_cell_idx(sxy)
    gi = xy_to_open_cell_idx(gxy)
    s_cells.append(si)
    g_cells.append(gi)
    for k in range(batch):
      if si[k] != gi[k] and BFS[(OPEN[si[k]], OPEN[gi[k]])] >= 2:
        blocked.append(los_blocked(sxy[k], gxy[k]))
      else:
        blocked.append(False)
  si = np.concatenate(s_cells)
  gi = np.concatenate(g_cells)
  d = np.array([BFS[(OPEN[a], OPEN[b])] for a, b in zip(si, gi)])
  blocked = np.array(blocked)
  return {
      'pairs_sampled': int(len(d)),
      'bfs0': float(np.mean(d == 0)), 'bfs1': float(np.mean(d == 1)),
      'bfs_ge2': float(np.mean(d >= 2)),
      'bfs_hist': {str(s): int(np.sum(d == s))
                   for s in sorted(set(d.tolist()))},
      'detour_separated (wall-blocked LOS)': float(np.mean(blocked)),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs', nargs='+', default=[
      'alpha0=d4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k/'
      'checkpoints/replay.npz'])
  ap.add_argument('--out', default='artifacts/ant_route_diagnosis')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  all_rows, summary = [], {}
  for spec in args.runs:
    tag, path = spec.split('=', 1)
    print(f'=== {tag}: {path} ===', flush=True)
    rows, _ = audit_replay(path, tag)
    all_rows += rows
    pol = [r for r in rows if not r['random_warmup']]
    rnd = [r for r in rows if r['random_warmup']]
    summary[tag] = {
        'replay': path,
        'all_episodes': fractions(rows),
        'policy_episodes': fractions(pol),
        'random_warmup_episodes': fractions(rnd) if rnd else None,
        'by_50k_phase': {str(p): fractions(
            [r for r in rows if r['phase_50k'] == p])
            for p in sorted({r['phase_50k'] for r in rows})},
        'positive_pairs_sampler': sampler_estimate(path),
    }
    print(json.dumps({k: v for k, v in summary[tag].items()
                      if k in ('all_episodes', 'positive_pairs_sampler')},
                     indent=1), flush=True)
  import csv
  cols = ['run', 'ep', 'actor', 'phase_50k', 'random_warmup', 'unique_cells',
          'bfs_span', 'boundary_crossings', 'crosses_2_boundaries',
          'corner_passage', 'detour_segment', 'start_end_bfs']
  with open(os.path.join(args.out, 'replay_route_stats.csv'), 'w',
            newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in all_rows:
      w.writerow({c: r[c] for c in cols})
  json.dump(summary, open(os.path.join(args.out,
                                       'replay_route_summary.json'), 'w'),
            indent=2)
  print('saved', os.path.join(args.out, 'replay_route_stats.csv'))


if __name__ == '__main__':
  main()
