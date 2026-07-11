"""PointMaze evidence pipeline (extends the Fetch-style report structure).

Post-hoc, reads a checkpoint (optional) + the env; produces the standard scalars
(success/final_dist/min_dist come from training metrics.json) PLUS maze-specific
evidence, so qualification rests on more than a single success number:

  * success at distance thresholds 2.0 / 1.0 / 0.5
  * path length + shortest-path (BFS) normalized efficiency
  * wall collisions
  * U-maze corridor / BFS-waypoint completion
  * endpoint + visitation heatmaps
  * trajectory overlays: random / direct-to-goal / BFS-oracle / trained
  * critic action-sensitivity scan vs short clone-rollout goal progress

The heavy plots are post-hoc (never block training). `eval_scalars` returns the
cheap maze scalars for merging into metrics.json during training.

Run:  python -m crl.report_maze --env_name point_U --ckpt run/best.pkl --out artifacts/point_U
"""
import argparse
import collections
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.config import Config

THRESHOLDS = (2.0, 1.0, 0.5)
WAYPOINT_RADIUS = 1.0          # visited a BFS waypoint if within this distance
COLLISION_DEFICIT = 0.5        # step moved < 50% of intended => wall collision


# --------------------------------------------------------------------------- #
# grid / BFS
# --------------------------------------------------------------------------- #
def cell_of(walls, state):
  ij = np.floor(state).astype(int)
  return tuple(np.clip(ij, [0, 0], np.array(walls.shape) - 1))


def bfs_path(walls, start_cell, goal_cell):
  """Shortest cell path (4-connectivity) over free cells; None if unreachable."""
  if walls[start_cell] == 1 or walls[goal_cell] == 1:
    return None
  q = collections.deque([start_cell])
  prev = {start_cell: None}
  while q:
    c = q.popleft()
    if c == goal_cell:
      path = []
      while c is not None:
        path.append(c); c = prev[c]
      return path[::-1]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      n = (c[0] + di, c[1] + dj)
      if (0 <= n[0] < walls.shape[0] and 0 <= n[1] < walls.shape[1]
          and walls[n] == 0 and n not in prev):
        prev[n] = c; q.append(n)
  return None


def bfs_waypoints(walls, start, goal):
  """Continuous waypoint polyline start -> cell-centers -> goal (or None)."""
  path = bfs_path(walls, cell_of(walls, start), cell_of(walls, goal))
  if path is None:
    return None
  centers = [np.array([i + 0.5, j + 0.5]) for (i, j) in path]
  wps = [np.asarray(start, float)] + centers[1:-1] + [np.asarray(goal, float)]
  return wps


def polyline_len(pts):
  return float(sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)))


# --------------------------------------------------------------------------- #
# shared evaluation manifest (fixed start-goal pairs, used by ALL policies)
# --------------------------------------------------------------------------- #
def sample_free(walls, rng):
  free = np.argwhere(walls == 0).astype(float)
  return free[rng.integers(len(free))] + rng.uniform(size=2)


def segment_blocked(env, s, g, n=25):
  """True if the straight line s->g crosses a wall (direct route obstructed)."""
  s, g = np.asarray(s, float), np.asarray(g, float)
  return any(env._is_blocked(s + t * (g - s)) for t in np.linspace(0, 1, n))


def build_manifest(env, n_random=30, n_hard=20, seed=0, min_sep=1.5):
  """One fixed set of (start,goal) pairs shared by every policy.
    'random'      = original reset distribution (reachable, min-separated)
    'hard_detour' = subset whose straight line is wall-blocked (needs to go
                    around the U barrier)."""
  walls = env._walls
  rng = np.random.default_rng(seed)
  randoms, hards, tries = [], [], 0
  while (len(randoms) < n_random or len(hards) < n_hard) and tries < 40000:
    tries += 1
    s, g = sample_free(walls, rng), sample_free(walls, rng)
    if np.linalg.norm(s - g) < min_sep:
      continue
    if bfs_waypoints(walls, s, g) is None:              # unreachable
      continue
    if len(randoms) < n_random:
      randoms.append((s, g))                            # original distribution
    if segment_blocked(env, s, g) and len(hards) < n_hard:
      hards.append((s, g))                              # hard-detour subset
  return {'random': randoms, 'hard_detour': hards}


# --------------------------------------------------------------------------- #
# policies:  fn(state, goal, memo) -> action in [-1,1]^2
# --------------------------------------------------------------------------- #
def make_random(rng):
  return lambda s, g, memo: rng.uniform(-1, 1, 2).astype(np.float32)


def direct_to_goal(s, g, memo):
  d = np.asarray(g) - np.asarray(s)
  return np.clip(d, -1, 1).astype(np.float32)


def make_oracle(walls):
  def policy(s, g, memo):
    if 'wps' not in memo:
      wps = bfs_waypoints(walls, s, g)
      memo['wps'] = wps if wps else [np.asarray(g, float)]
      memo['i'] = 1 if len(memo['wps']) > 1 else 0
    wps, i = memo['wps'], memo['i']
    while i < len(wps) - 1 and np.linalg.norm(wps[i] - s) < 0.5:
      i += 1
    memo['i'] = i
    return np.clip(wps[i] - s, -1, 1).astype(np.float32)
  return policy


def make_trained(greedy):
  def policy(s, g, memo):
    obs = np.concatenate([s, g]).astype(np.float32)
    return np.asarray(greedy(obs), np.float32)
  return policy


# --------------------------------------------------------------------------- #
# rollout + per-episode metrics
# --------------------------------------------------------------------------- #
def rollout(env, policy, fixed=None):
  """One episode. `fixed`=(start,goal) to override the reset sampling."""
  walls = env._walls
  env.reset()
  if fixed is not None:
    env.state = np.asarray(fixed[0], float).copy()
    env.goal = np.asarray(fixed[1], float).copy()
  s0, g = env.state.copy(), env.goal.copy()
  traj = [s0.copy()]
  memo = {}
  collisions = 0
  dists = [float(np.linalg.norm(s0 - g))]
  for _ in range(env.max_episode_steps):
    a = policy(env.state.copy(), g, memo)
    before = env.state.copy()
    env.step(a)
    traj.append(env.state.copy())
    dists.append(float(np.linalg.norm(env.state - g)))
    moved = np.linalg.norm(env.state - before)
    intended = np.linalg.norm(np.clip(a, -1, 1))
    if intended > 1e-3 and moved < COLLISION_DEFICIT * intended:
      collisions += 1
  traj = np.array(traj)
  wps = bfs_waypoints(walls, s0, g)
  sp_len = polyline_len(wps) if wps else np.nan
  path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
  # waypoint completion: fraction of BFS waypoints the trajectory passed near
  if wps:
    hit = sum(any(np.linalg.norm(traj - w, axis=1).min() < WAYPOINT_RADIUS
                  for _ in [0]) for w in wps)
    wp_completion = hit / len(wps)
  else:
    wp_completion = np.nan
  min_dist = float(min(dists))
  success05 = min_dist < 0.5
  # SPL: Success weighted by (normalized inverse) Path Length. 0 for failures.
  if not np.isfinite(sp_len):
    spl = np.nan
  elif sp_len < 1e-6:
    spl = float(success05)
  else:
    spl = float(success05) * sp_len / max(path_len, sp_len)
  eff_cond = (sp_len / max(path_len, sp_len)) if (np.isfinite(sp_len)
             and path_len > 1e-6) else np.nan     # success-conditional efficiency
  return dict(
      traj=traj, start=s0, goal=g, dists=dists,
      min_dist=min_dist, final_dist=float(dists[-1]),
      success_at={t: bool(min_dist < t) for t in THRESHOLDS},
      path_len=path_len, shortest_path_len=sp_len,
      spl=spl, efficiency=eff_cond,
      collisions=collisions, wp_completion=wp_completion)


def aggregate(eps):
  if not eps:
    return {'n_episodes': 0}
  def m(f):
    vals = [f(e) for e in eps]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None
  out = {'n_episodes': len(eps),
         'final_dist': m(lambda e: e['final_dist']),
         'min_dist': m(lambda e: e['min_dist']),
         'path_len': m(lambda e: e['path_len']),
         'collisions': m(lambda e: e['collisions']),
         'wp_completion': m(lambda e: e['wp_completion']),
         'spl': m(lambda e: e['spl'])}                 # headline efficiency metric
  for t in THRESHOLDS:
    out[f'success@{t}'] = float(np.mean([e['success_at'][t] for e in eps]))
  # secondary: success-conditional efficiency (mean over reached-0.5 episodes)
  succ = [e for e in eps if e['success_at'][0.5]]
  out['efficiency_at_success'] = (
      float(np.mean([e['efficiency'] for e in succ
                     if np.isfinite(e['efficiency'])])) if succ else None)
  return out


# --------------------------------------------------------------------------- #
# nets / critic loading
# --------------------------------------------------------------------------- #
def load_nets(env_name, ckpt, cfg):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  step, state = ckpt_mod.load_checkpoint(ckpt)

  @jax.jit
  def greedy(obs):
    return nets.sample_eval(nets.policy_network.apply(state.policy_params, obs), None)

  def greedy_np(obs):
    return np.asarray(greedy(jnp.asarray(obs[None]))[0])
  return nets, state, greedy_np, step


# --------------------------------------------------------------------------- #
# action-sensitivity scan (critic ranking vs clone-rollout progress)
# --------------------------------------------------------------------------- #
def action_scan(env, nets, q_params, state_goal_pairs, n_dirs=16, k_steps=3):
  """At each (state,goal): rank K action directions by critic Q vs by actual
  short-clone-rollout goal progress; report Spearman corr + argmax agreement."""
  thetas = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
  cand = np.stack([np.cos(thetas), np.sin(thetas)], 1).astype(np.float32)  # [K,2]
  corrs, agree = [], []
  for (s, g) in state_goal_pairs:
    obs = np.concatenate([s, g]).astype(np.float32)
    obs_k = jnp.asarray(np.tile(obs, (n_dirs, 1)))
    q = np.diag(np.asarray(nets.q_network.apply(q_params, obs_k, jnp.asarray(cand))))
    prog = np.zeros(n_dirs)
    for i, a in enumerate(cand):
      env.state = np.asarray(s, float).copy(); env.goal = np.asarray(g, float).copy()
      d0 = np.linalg.norm(env.state - g)
      for _ in range(k_steps):
        env.step(a)
      prog[i] = d0 - np.linalg.norm(env.state - g)
    # Spearman = Pearson on ranks
    rq, rp = _rank(q), _rank(prog)
    if rq.std() > 0 and rp.std() > 0:
      corrs.append(float(np.corrcoef(rq, rp)[0, 1]))
    agree.append(int(np.argmax(q) == np.argmax(prog)))
  return {'n_pairs': len(state_goal_pairs), 'n_dirs': n_dirs, 'k_steps': k_steps,
          'spearman_mean': float(np.mean(corrs)) if corrs else None,
          'argmax_agree_frac': float(np.mean(agree)) if agree else None}


def _rank(x):
  order = np.argsort(x); r = np.empty_like(order, float); r[order] = np.arange(len(x))
  return r


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _draw_walls(ax, walls):
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55,
            extent=[0, walls.shape[0], 0, walls.shape[1]])
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')


def plot_overlays(walls, policy_eps, out):
  n = len(policy_eps)
  fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
  if n == 1:
    axes = [axes]
  for ax, (name, eps) in zip(axes, policy_eps.items()):
    _draw_walls(ax, walls)
    for e in eps[:12]:
      t = e['traj']
      ax.plot(t[:, 0], t[:, 1], '-', lw=0.8, alpha=0.6)
      ax.scatter(*e['start'], c='tab:green', s=18, zorder=3)
      ax.scatter(*e['goal'], c='red', marker='*', s=70, zorder=3)
    sr = np.mean([e['success_at'][0.5] for e in eps])
    ax.set_title(f'{name}\nsucc@0.5={sr:.2f}', fontsize=9)
  fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def plot_checkpoint_progression(env_name, ckpts, labels, fixed, out, seed=0):
  """Same fixed (start,goal); trained trajectory at each checkpoint (early/mid/final)."""
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  walls = env._walls
  fig, axes = plt.subplots(1, len(ckpts), figsize=(3.2 * len(ckpts), 3.4))
  if len(ckpts) == 1:
    axes = [axes]
  for ax, ck, lab in zip(axes, ckpts, labels):
    _, _, greedy, step = load_nets(env_name, ck, cfg)
    e = rollout(env, make_trained(greedy), fixed=fixed)
    _draw_walls(ax, walls)
    t = e['traj']
    ax.plot(t[:, 0], t[:, 1], '-', lw=1.2, color='tab:blue')
    ax.scatter(*e['start'], c='tab:green', s=30, zorder=3)
    ax.scatter(*e['goal'], c='red', marker='*', s=110, zorder=3)
    ax.set_title(f'{lab} (step {step})\nmin_dist={e["min_dist"]:.2f}', fontsize=9)
  fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def plot_heatmaps(walls, eps, out):
  H, W = walls.shape
  visit = np.concatenate([e['traj'] for e in eps], 0)
  ends = np.stack([e['traj'][-1] for e in eps])
  fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
  wall_mask = np.ma.masked_where(walls == 0, walls)   # show only wall cells
  for a, data, title in [(ax[0], visit, 'visitation'), (ax[1], ends, 'endpoints')]:
    hh, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=[H, W],
                              range=[[0, H], [0, W]])
    a.imshow(hh.T, origin='lower', cmap='viridis', extent=[0, H, 0, W])
    a.imshow(wall_mask.T, origin='lower', cmap='Greys', vmin=0, vmax=1,
             alpha=0.9, extent=[0, H, 0, W])          # walls in grey on top
    a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])
    a.set_aspect('equal')
  fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def eval_scalars(env, policy, episodes=30):
  """Cheap maze scalars for metrics.json enrichment during training."""
  eps = [rollout(env, policy) for _ in range(episodes)]
  return aggregate(eps)


def full_report(env_name, ckpt=None, n_random=40, n_hard=25, seed=123, out=None,
                ckpt_label='trained'):
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  walls = env._walls
  rng = np.random.default_rng(seed)
  manifest = build_manifest(env, n_random=n_random, n_hard=n_hard, seed=seed)

  policies = {'random': make_random(rng), 'direct': direct_to_goal,
              'oracle': make_oracle(walls)}
  step = None
  if ckpt:
    nets, state, greedy, step = load_nets(env_name, ckpt, cfg)
    policies[ckpt_label] = make_trained(greedy)     # 'trained' or 'random-init'

  # EVERY policy runs on the SAME fixed manifest, per subset.
  eps_by, by_policy = {}, {}
  for name, pol in policies.items():
    eps_by[name], by_policy[name] = {}, {}
    for subset, pairs in manifest.items():
      eps = [rollout(env, pol, fixed=pair) for pair in pairs]
      eps_by[name][subset] = eps
      by_policy[name][subset] = aggregate(eps)

  report = {'env_name': env_name, 'ckpt': ckpt, 'step': step,
            'ckpt_label': ckpt_label if ckpt else None,
            'manifest': {k: len(v) for k, v in manifest.items()},
            'by_policy': by_policy}
  if ckpt:
    pairs = manifest['hard_detour'][:8] + manifest['random'][:8]
    report['action_scan'] = action_scan(env, nets, state.q_params, pairs)

  if out:
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, 'maze_report.json'), 'w'), indent=2)
    ov = 'hard_detour' if manifest['hard_detour'] else 'random'
    plot_overlays(walls, {n: eps_by[n][ov] for n in eps_by},
                  os.path.join(out, f'trajectory_overlays_{ov}.png'))
    hkey = ckpt_label if ckpt else 'oracle'
    plot_heatmaps(walls, eps_by[hkey]['random'], os.path.join(out, 'heatmaps.png'))
  return report, eps_by


def _print(report):
  m = report['manifest']
  print('=' * 74)
  print(f'MAZE REPORT  {report["env_name"]}  (step {report["step"]}; manifest '
        f'random={m["random"]} hard_detour={m["hard_detour"]})')
  print('=' * 74)
  for subset in ('random', 'hard_detour'):
    print(f'\n[{subset}]')
    print(f'{"policy":>10} | {"s@2.0":>6} {"s@1.0":>6} {"s@0.5":>6} {"SPL":>6} '
          f'{"eff*":>6} {"path":>6} {"coll":>5} {"wp":>5} {"fdist":>6}')
    for name, d in report['by_policy'].items():
      a = d[subset]
      if a.get('n_episodes', 0) == 0:
        print(f'{name:>10} | (no pairs)'); continue
      eff, spl = a['efficiency_at_success'], a['spl']
      print(f'{name:>10} | {a["success@2.0"]:>6.2f} {a["success@1.0"]:>6.2f} '
            f'{a["success@0.5"]:>6.2f} {spl if spl is None else round(spl,2):>6} '
            f'{eff if eff is None else round(eff,2):>6} {a["path_len"]:>6.1f} '
            f'{a["collisions"]:>5.1f} {a["wp_completion"]:>5.2f} {a["final_dist"]:>6.2f}')
  if 'action_scan' in report:
    s = report['action_scan']
    print(f'\naction-scan: spearman(Q,progress)={s["spearman_mean"]}  '
          f'argmax-agree={s["argmax_agree_frac"]}  ({s["n_pairs"]}x{s["n_dirs"]})')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--env_name', default='point_U')
  p.add_argument('--ckpt', default=None)
  p.add_argument('--n_random', type=int, default=40)
  p.add_argument('--n_hard', type=int, default=25)
  p.add_argument('--ckpt_label', default='trained')
  p.add_argument('--seed', type=int, default=123)
  p.add_argument('--out', default=None)
  args = p.parse_args()
  report, _ = full_report(args.env_name, ckpt=args.ckpt, n_random=args.n_random,
                          n_hard=args.n_hard, seed=args.seed, out=args.out,
                          ckpt_label=args.ckpt_label)
  _print(report)
  if args.out:
    print(f'\nsaved report + plots under {args.out}')


if __name__ == '__main__':
  main()
