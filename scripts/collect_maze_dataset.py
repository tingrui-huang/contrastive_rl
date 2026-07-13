"""Collect a FROZEN offline dataset from a point-maze env (numpy, CPU).

Maze analog of scripts/collect_push_dataset.py. A behavior policy generates N
episodes ONCE with random start+goal per episode (the online PointEnv reset
distribution); training then runs on the fixed .npz with ZERO env interaction
(`--offline_dataset` in crl.train). This is the data layer the offline-only
requirement needs: dataset generated + frozen + hashed BEFORE any optimization.

Behavior policy (per episode, i.i.d.):
  * 'oracle' : BFS-waypoint follower (report_maze.make_oracle over env._walls)
               plus Gaussian action noise -> goal-reaching demonstrations.
  * 'random' : uniform actions in [-1,1]^A -> action-space / state coverage.
  * 'mix'    : an ``oracle_frac`` fraction oracle, the rest random (default).

Output .npz (matches the online collect_episode layout exactly):
  obs  [N, L, obs_dim+goal_dim]  float32   (L = max_episode_steps + 1)
  act  [N, L, action_dim]        float32   (act[:, -1] is a dummy zero row)
  meta (json string: env, policy, seed, per-arm behavior success, boundaries)

A sidecar ``<out>.manifest.json`` records the sha256 of the .npz, shapes, and
provenance so the frozen dataset can be verified before/after training. The
script REFUSES to overwrite an existing dataset unless --force (freeze safety).

Run:
  python -m scripts.collect_maze_dataset --env_name point_U \
      --episodes 5000 --oracle_frac 0.5 --noise 0.1 --seed 0 \
      --out datasets/point_U_offline_s0.npz
"""
import argparse
import hashlib
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import make_oracle, make_random, bfs_waypoints


def _sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def collect(env_name, episodes, oracle_frac, noise, seed, out, force=False):
  if os.path.exists(out) and not force:
    raise SystemExit(
        f'REFUSING to overwrite frozen dataset {out} (pass --force to replace). '
        'A frozen dataset must not be silently clobbered.')
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  if not hasattr(env, '_walls'):
    raise SystemExit(f'{env_name} is not a point-maze env (no ._walls).')
  walls = env._walls
  rng = np.random.default_rng(seed)
  L = env.max_episode_steps + 1
  D = cfg.obs_dim + cfg.goal_dim
  A = cfg.action_dim

  obs_out = np.zeros((episodes, L, D), dtype=np.float32)
  act_out = np.zeros((episodes, L, A), dtype=np.float32)
  oracle = make_oracle(walls)
  rand_pol = make_random(rng)
  succ, reach, kinds = [], [], []
  n_oracle = int(round(episodes * oracle_frac))

  for ep in range(episodes):
    oracle_ep = ep < n_oracle          # leading block; start/goal are i.i.d.
    kinds.append('oracle' if oracle_ep else 'random')
    env.reset()                        # random start + goal (online distribution)
    s0, g = env.state.copy(), env.goal.copy()
    memo = {}
    hit = 0.0
    min_d = float(np.linalg.norm(s0 - g))
    for t in range(env.max_episode_steps):
      obs_out[ep, t] = np.concatenate([env.state, g]).astype(np.float32)
      if oracle_ep:
        a = oracle(env.state.copy(), g, memo)
        a = np.clip(a + rng.normal(0.0, noise, size=A), -1, 1).astype(np.float32)
      else:
        a = rand_pol(env.state.copy(), g, memo)
      act_out[ep, t] = a
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      min_d = min(min_d, float(np.linalg.norm(env.state - g)))
    obs_out[ep, -1] = np.concatenate([env.state, g]).astype(np.float32)
    succ.append(hit)
    reach.append(float(min_d < 0.5))
    if (ep + 1) % 500 == 0:
      print(f'  {ep + 1}/{episodes} episodes '
            f'(reward-hit {np.mean(succ):.3f}, reached@0.5 {np.mean(reach):.3f})',
            flush=True)

  succ, reach = np.asarray(succ), np.asarray(reach)
  is_orc = np.asarray([k == 'oracle' for k in kinds])
  # BFS-reachability of the start->goal pairs (sanity that the maze is solvable)
  meta = {
      'env_name': env_name, 'episodes': int(episodes), 'seed': int(seed),
      'behavior_policy': f'mix(oracle_frac={oracle_frac})',
      'oracle_frac': float(oracle_frac), 'action_noise': float(noise),
      'obs_dim': int(cfg.obs_dim), 'goal_dim': int(cfg.goal_dim),
      'action_dim': int(A), 'max_episode_steps': int(env.max_episode_steps),
      'episode_len_rows_L': int(L), 'obs_width_D': int(D),
      'n_transitions': int(episodes * (L - 1)),
      'episode_boundaries': 'fixed-length independent blocks; relabel within-episode only',
      'reward_hit_rate': float(succ.mean()),
      'reached_0p5_rate': float(reach.mean()),
      'reached_0p5_oracle': float(reach[is_orc].mean()) if is_orc.any() else None,
      'reached_0p5_random': float(reach[~is_orc].mean()) if (~is_orc).any() else None,
  }
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  np.savez(out, obs=obs_out, act=act_out, meta=np.array(json.dumps(meta)))

  digest = _sha256(out)
  size_mb = os.path.getsize(out) / 1e6
  manifest = dict(
      path=os.path.abspath(out), sha256=digest,
      size_bytes=int(os.path.getsize(out)),
      obs_shape=list(obs_out.shape), act_shape=list(act_out.shape),
      dtype=str(obs_out.dtype), frozen=True, meta=meta)
  man_path = out + '.manifest.json'
  json.dump(manifest, open(man_path, 'w'), indent=2)
  # Read-only flag on the frozen artifact (defense-in-depth; --force can clear).
  try:
    os.chmod(out, 0o444)
  except OSError:
    pass
  print(f'\nFROZEN dataset -> {out} ({size_mb:.1f} MB)')
  print(f'sha256 = {digest}')
  print(f'manifest -> {man_path}')
  print(json.dumps(meta, indent=2))
  return manifest


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--env_name', default='point_U')
  p.add_argument('--episodes', type=int, default=5000)
  p.add_argument('--oracle_frac', type=float, default=0.5)
  p.add_argument('--noise', type=float, default=0.1)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', required=True)
  p.add_argument('--force', action='store_true',
                 help='overwrite an existing frozen dataset (clears read-only)')
  a = p.parse_args()
  if a.force and os.path.exists(a.out):
    try:
      os.chmod(a.out, 0o644)
    except OSError:
      pass
  collect(a.env_name, a.episodes, a.oracle_frac, a.noise, a.seed, a.out,
          force=a.force)


if __name__ == '__main__':
  main()
