"""Phase 2+3: antmaze-umaze-v2 dataset reconstruction + route-support audit.

Phase 2 (integrity / exact trajectories):
  * keys, dims, dtypes, total transitions;
  * episode boundaries from `timeouts` (upstream contract:
    distributed_layout.py splits on timeouts; obs[t], act[t] -> obs[t+1]
    within an episode, adder pairs action[t-1] with next_timestep obs[t]);
  * temporal alignment checks: within-episode XY step lengths vs
    across-boundary jumps; qvel consistency (obs[15:17] ~ dxy/dt, dt=0.1);
  * terminal handling, trajectory-length histogram, action range,
    observation dimension;
  * empirical maze-cell occupancy vs the U_MAZE OPEN set (verifies the
    coordinate convention before any BFS statistic is trusted).

Phase 3 (route support):
  * per-episode BFS span / detour / corner / coverage (same tooling as
    scripts/route_replay_audit.py);
  * start-cell and end-cell coverage, XY visitation heatmap;
  * action coverage + effective rank; moving/stationary fractions;
  * >=100k positive tuples via the EXACT intended future-goal sampler
    (flatten_fn math: i uniform, j>i ~ discount**(j-i), gamma=0.99):
    future-horizon distribution, BFS-span distribution, detour-separated
    fraction, euclidean vs geodesic distances, route-level fraction;
  * direct comparison against the online replay baseline
    (BFS>=2 positives 0.137%, detour-separated 0.0059%).
"""
import json
import os
import sys
from collections import Counter, deque
from itertools import combinations

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from route_replay_audit import (OPEN, WALL, CENTERS, BFS, xy_to_open_cell_idx,
                                los_blocked, episode_stats, fractions)
from crl.d4rl_ant import SCALING

DATA = r'D:\Users\trhua\Research\datasets\d4rl\antmaze-umaze-v2.hdf5'
OUT = 'artifacts/offline_d4rl'
GAMMA = 0.99
N_PAIRS = 102_400
ONLINE_BASELINE = {'bfs_ge2': 0.00137, 'detour_separated': 5.86e-05}

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Phase 2
print('=== Phase 2: integrity / exact trajectories ===', flush=True)
with h5py.File(DATA, 'r') as f:
  def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
      print(f'  {name:24s} {str(obj.shape):16s} {obj.dtype}')
  f.visititems(walk)
  obs = f['observations'][:]
  act = f['actions'][:]
  rew = f['rewards'][:]
  term = f['terminals'][:]
  tout = f['timeouts'][:]
  goal_info = f['infos/goal'][:] if 'infos/goal' in f else None

N = obs.shape[0]
audit = {
    'file': DATA, 'total_transitions_rows': int(N),
    'obs_dim': int(obs.shape[1]), 'action_dim': int(act.shape[1]),
    'action_min': float(act.min()), 'action_max': float(act.max()),
    'obs_xy_min': [float(v) for v in obs[:, :2].min(0)],
    'obs_xy_max': [float(v) for v in obs[:, :2].max(0)],
    'rewards_nonzero': int((rew != 0).sum()),
    'terminals_true': int(term.sum()),
    'timeouts_true': int(tout.sum()),
    'goal_info_unique': (np.unique(np.round(goal_info, 3), axis=0).tolist()
                         if goal_info is not None else None),
}
assert audit['obs_dim'] == 29, audit['obs_dim']
assert audit['action_dim'] == 8
assert -1.001 <= audit['action_min'] and audit['action_max'] <= 1.001, \
    'dataset actions not in [-1, 1]'

# episode boundaries: split AFTER each timeouts[t]==True row (upstream: the
# row with timeouts[t] is the LAST of its episode; t==0 or timeouts[t-1]
# starts a new one).
ends = np.where(tout)[0]
starts = np.concatenate([[0], ends + 1])
starts = starts[starts < N]
ep_bounds = []
for s in starts:
  nxt = ends[ends >= s]
  e = int(nxt[0]) if len(nxt) else N - 1
  ep_bounds.append((int(s), e))          # inclusive [s, e]
lens = np.array([e - s + 1 for s, e in ep_bounds])
audit['episodes'] = len(ep_bounds)
audit['episode_len_hist'] = {str(k): int(v)
                             for k, v in sorted(Counter(lens.tolist()).items())}
audit['transitions_within_episodes'] = int((lens - 1).sum())

# temporal alignment: within-episode step sizes vs boundary jumps
within, across = [], []
for (s, e) in ep_bounds[:200]:
  d = np.linalg.norm(np.diff(obs[s:e + 1, :2], axis=0), axis=1)
  within.append(d)
for k in range(min(len(ep_bounds) - 1, 200)):
  e = ep_bounds[k][1]
  across.append(float(np.linalg.norm(obs[e + 1, :2] - obs[e, :2])))
within = np.concatenate(within)
audit['xy_step_within_ep'] = {
    'median': float(np.median(within)), 'p99': float(np.percentile(within, 99)),
    'max': float(within.max())}
audit['xy_jump_across_boundary'] = {
    'median': float(np.median(across)), 'min': float(np.min(across))}
# qvel consistency: obs[t,15:17] should predict (xy[t+1]-xy[t])/dt, dt=0.1
s0, e0 = ep_bounds[0]
v_pred = obs[s0:e0, 15:17]
v_emp = np.diff(obs[s0:e0 + 1, :2], axis=0) / 0.1
corr = float(np.corrcoef(v_pred.ravel(), v_emp.ravel())[0, 1])
audit['qvel_vs_dxy_corr_ep0'] = corr

# empirical cell occupancy vs U_MAZE OPEN set
ci = xy_to_open_cell_idx(obs[::37, :2].astype(np.float64))
occ = Counter(OPEN[i] for i in ci)
snap = np.linalg.norm(
    obs[::37, :2].astype(np.float64) - CENTERS[ci], axis=1)
audit['cell_occupancy'] = {str(k): int(v) for k, v in sorted(occ.items())}
audit['xy_within_2.83m_of_nearest_open_center'] = float(
    np.mean(snap <= np.sqrt(2) * SCALING / 2 + 0.75))
print(json.dumps(audit, indent=1), flush=True)

# ---------------------------------------------------------------- Phase 3
print('=== Phase 3: route-support audit ===', flush=True)
rows = []
for k, (s, e) in enumerate(ep_bounds):
  st = episode_stats(obs[s:e + 1, :2].astype(np.float64))
  st.update(ep=k, actor=-1, phase_50k=-1, random_warmup=False, run='d4rl')
  rows.append(st)
frac = fractions(rows, key_steps=int(np.median(lens)) - 1)

start_cells = Counter(str(OPEN[xy_to_open_cell_idx(
    obs[s:s + 1, :2].astype(np.float64))[0]]) for s, _ in ep_bounds)
end_cells = Counter(str(OPEN[xy_to_open_cell_idx(
    obs[e:e + 1, :2].astype(np.float64))[0]]) for _, e in ep_bounds)

# heatmap (0.25 m bins over [-2.5, 10.5]^2)
H, xe, ye = np.histogram2d(obs[:, 0], obs[:, 1],
                           bins=[np.arange(-2.5, 10.75, 0.25)] * 2)
np.save(f'{OUT}/visitation_heatmap.npy', H)
try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.figure(figsize=(5, 5))
  plt.imshow(np.log1p(H.T), origin='lower',
             extent=[-2.5, 10.5, -2.5, 10.5], cmap='viridis')
  for (r, c) in WALL:
    if 0 <= r < 5 and 0 <= c < 5:
      plt.gca().add_patch(plt.Rectangle((c * 4 - 6, r * 4 - 6), 4, 4,
                                        fill=False, ec='r', lw=0.5))
  plt.title('antmaze-umaze-v2 log visitation')
  plt.savefig(f'{OUT}/visitation_heatmap.png', dpi=120, bbox_inches='tight')
  heat_png = True
except Exception as ex:  # noqa: BLE001
  heat_png = f'matplotlib unavailable: {ex}'

# action coverage / effective rank; moving fraction
aidx = rng.integers(0, N, 100_000)
A = act[aidx]
sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
p = (sv ** 2) / (sv ** 2).sum()
eff_rank = float(np.exp(-(p * np.log(p)).sum()))
dxy = np.concatenate([np.linalg.norm(np.diff(obs[s:e + 1, :2], axis=0), axis=1)
                      for s, e in ep_bounds])
route = {
    'episodes': frac,
    'start_cell_coverage': dict(sorted(start_cells.items())),
    'end_cell_coverage': dict(sorted(end_cells.items())),
    'heatmap': {'npy': f'{OUT}/visitation_heatmap.npy', 'png': heat_png},
    'action_coverage': {
        'per_dim_mean': [round(float(m), 3) for m in A.mean(0)],
        'per_dim_std': [round(float(s), 3) for s in A.std(0)],
        'frac_saturated_|a|>0.99': float(np.mean(np.abs(A) > 0.99)),
        'effective_rank_of_8': eff_rank},
    'moving_fractions': {
        'dxy_gt_0.01m': float(np.mean(dxy > 0.01)),
        'dxy_gt_0.05m': float(np.mean(dxy > 0.05)),
        'stationary_le_0.01m': float(np.mean(dxy <= 0.01))},
}
print(json.dumps(route, indent=1, default=str), flush=True)

# ---- positive tuples via the EXACT intended sampler (flatten_fn math) ----
print(f'=== sampling {N_PAIRS} positive tuples (gamma={GAMMA}) ===',
      flush=True)
log_g = np.log(GAMMA)
eps_idx = rng.integers(0, len(ep_bounds), N_PAIRS)
res = {'dt_hist': Counter(), 'bfs': [], 'euclid': [], 'geo': [],
       'blocked': 0}
dts, s_xys, g_xys = [], [], []
for b in range(0, N_PAIRS, 4096):
  batch = eps_idx[b:b + 4096]
  for e_id in batch:
    s, e = ep_bounds[e_id]
    L = e - s + 1
    i = rng.integers(0, L - 1)
    arange = np.arange(L)
    logp = np.where(arange > i, (arange - i) * log_g, -np.inf)
    g = -np.log(-np.log(rng.uniform(size=L).clip(1e-20, 1.0)))
    j = int(np.argmax(logp + g))
    dts.append(j - i)
    s_xys.append(obs[s + i, :2])
    g_xys.append(obs[s + j, :2])
s_xys = np.array(s_xys, np.float64)
g_xys = np.array(g_xys, np.float64)
dts = np.array(dts)
si = xy_to_open_cell_idx(s_xys)
gi = xy_to_open_cell_idx(g_xys)
bfs_d = np.array([BFS[(OPEN[a], OPEN[bb])] for a, bb in zip(si, gi)])
euclid = np.linalg.norm(s_xys - g_xys, axis=1)
geo = np.where(bfs_d >= 1, bfs_d * SCALING, euclid)
blocked = np.zeros(len(bfs_d), bool)
sel = np.where(bfs_d >= 2)[0]
for k in sel:
  blocked[k] = los_blocked(s_xys[k], g_xys[k])

pairs = {
    'n': int(len(bfs_d)),
    'future_horizon_dt': {
        'mean': float(dts.mean()), 'median': float(np.median(dts)),
        'p90': float(np.percentile(dts, 90)),
        'p99': float(np.percentile(dts, 99)), 'max': int(dts.max())},
    'bfs_hist': {str(k): int(v) for k, v in sorted(
        Counter(bfs_d.tolist()).items())},
    'bfs0': float(np.mean(bfs_d == 0)), 'bfs1': float(np.mean(bfs_d == 1)),
    'bfs_ge2': float(np.mean(bfs_d >= 2)),
    'route_level_positive_fraction': float(np.mean(bfs_d >= 2)),
    'detour_separated': float(blocked.mean()),
    'euclid_dist': {'mean': float(euclid.mean()),
                    'median': float(np.median(euclid)),
                    'p90': float(np.percentile(euclid, 90))},
    'geodesic_dist_cellapprox': {'mean': float(geo.mean()),
                                 'median': float(np.median(geo))},
    'geo_gt_euclid_1.5x_frac': float(np.mean(geo > 1.5 * euclid)),
    'vs_online_replay': {
        'online_bfs_ge2': ONLINE_BASELINE['bfs_ge2'],
        'online_detour_separated': ONLINE_BASELINE['detour_separated'],
        'ratio_bfs_ge2': float(np.mean(bfs_d >= 2)
                               / ONLINE_BASELINE['bfs_ge2']),
        'ratio_detour': float(blocked.mean()
                              / ONLINE_BASELINE['detour_separated']),
    },
}
print(json.dumps(pairs, indent=1), flush=True)

json.dump({'phase2_integrity': audit, 'phase3_route': route,
           'phase3_positive_pairs': pairs},
          open(f'{OUT}/dataset_audit.json', 'w'), indent=2, default=str)

import csv
with open(f'{OUT}/dataset_route_stats.csv', 'w', newline='') as fcsv:
  cols = ['run', 'ep', 'unique_cells', 'bfs_span', 'boundary_crossings',
          'crosses_2_boundaries', 'corner_passage', 'detour_segment',
          'start_end_bfs']
  w = csv.DictWriter(fcsv, fieldnames=cols)
  w.writeheader()
  for r in rows:
    w.writerow({c: r[c] for c in cols})
print('saved', f'{OUT}/dataset_audit.json', 'and dataset_route_stats.csv')
