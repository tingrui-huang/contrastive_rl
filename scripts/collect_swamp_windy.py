"""Collect the FROZEN offline dataset for the windy-LETHAL swamp env.

Env: point_two_route_swamp_windy_v0 (crl.envs.TwoRouteSwampWindyEnv) --
bits resample EVERY step everywhere (wind), ending a step in an active
swamp cell is terminal (lava). See the env docstring.

Behavior policy (the confounded teacher, u -> a at EVERY corridor step):
  * force_safe episodes (episode-level coin): take the always-safe lower
    route (BFS on swamp-blocked walls).
  * otherwise: head for the shortcut (BFS on the true walls); each step,
    look at where the intended move would land THIS step and dodge:
    candidates [forward, wait, backward], first whose landing cell is not
    a currently-active swamp cell wins (the bits read before acting are
    exactly the bits of this step's death check). This is the WindyCorridor
    expert geometry: the confounder is visible in the ACTION DISTRIBUTION
    along the whole corridor, not just at the entrance.
  * random_frac episodes: uniform actions (coverage).

Output .npz matches the matched collector's layout exactly (obs, act, meta
+ audit-only fields swamp_bits / route_label / teacher_mode / force_safe /
wait_count / entered_active_swamp -- 'entered_active_swamp' records DEATH
here). Manifest with sha256; refuses to overwrite without --force.

Run:
  python -m scripts.collect_swamp_windy --episodes 6000 --random_frac 0.2 \
      --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 \
      --out datasets/swamp_windy_teacher_s0.npz
"""
import argparse
import hashlib
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import make_oracle
from scripts.qualify_two_route_swamp import swamp_blocked_walls

ENV = 'point_two_route_swamp_windy_v0'
MODE = {'random': 0, 'forced_safe': 1, 'immediate_shortcut': 2,
        'wait_shortcut': 3}
ROUTE = {'random': 0, 'shortcut': 1, 'safe_detour': 2, 'other': 3}


def _sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def make_windy_teacher(env, rng, force_safe_prob):
  """Per-step-reactive confounded teacher (reads env.swamp_bits each step)."""
  base_oracle = make_oracle(env._walls)
  safe_oracle = make_oracle(swamp_blocked_walls(env._walls))
  swamp = list(env.SWAMP_CELLS)

  def landing_cell(s, a):
    n = np.clip(s + np.clip(a, -1, 1), [0, 0],
                np.asarray(env._walls.shape, float) - 1e-6)
    return tuple(np.floor(n).astype(int))

  def policy(s, g, memo):
    if 'force_safe' not in memo:
      memo['force_safe'] = bool(rng.random() < force_safe_prob)
      memo['teacher_mode'] = ('forced_safe' if memo['force_safe']
                              else 'immediate_shortcut')
      memo['wait_count'] = 0
      memo['_m'] = {}
    if memo['force_safe']:
      return safe_oracle(s, g, memo['_m'])
    a = base_oracle(s, g, memo['_m'])
    bits = env.swamp_bits
    def deadly(cell):
      return cell in swamp and bool(bits[swamp.index(cell)])
    for cand in (a, np.zeros(2, np.float32), -a):
      if not deadly(landing_cell(s, cand)):
        if not np.array_equal(cand, a):
          memo['wait_count'] += 1
          memo['teacher_mode'] = 'wait_shortcut'
        return np.asarray(cand, np.float32)
    return a                                     # doomed either way: forward
  return policy


def collect(episodes, random_frac, force_safe_prob, teacher_noise, seed, out,
            force=False):
  if os.path.exists(out) and not force:
    raise SystemExit(f'REFUSING to overwrite frozen dataset {out} (use --force).')
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=seed)
  rng = np.random.default_rng(seed)
  teacher = make_windy_teacher(env, rng, force_safe_prob)
  L = env.max_episode_steps + 1
  D, A = cfg.obs_dim + cfg.goal_dim, cfg.action_dim

  obs_out = np.zeros((episodes, L, D), np.float32)
  act_out = np.zeros((episodes, L, A), np.float32)
  bits_out = np.zeros((episodes, L, 3), np.uint8)
  route_out = np.zeros((episodes,), np.int64)
  mode_out = np.zeros((episodes,), np.int64)
  fsafe_out = np.zeros((episodes,), np.uint8)
  wait_out = np.zeros((episodes,), np.int64)
  died_out = np.zeros((episodes,), np.uint8)
  n_random = int(round(episodes * random_frac))
  succ = []

  for ep in range(episodes):
    env.reset()
    g = env.goal.copy()
    memo = {}
    is_random = ep < n_random
    traj = [env.state.copy()]
    for t in range(env.max_episode_steps):
      obs_out[ep, t] = np.concatenate([env.state, g]).astype(np.float32)
      bits_out[ep, t] = env.swamp_bits.astype(np.uint8)
      if is_random:
        a = rng.uniform(-1, 1, A).astype(np.float32)
      else:
        a = np.asarray(teacher(env.state.copy(), g, memo), np.float32)
        if teacher_noise > 0 and np.any(a != 0):
          a = np.clip(a + rng.normal(0, teacher_noise, A), -1, 1).astype(np.float32)
      act_out[ep, t] = a
      env.step(a)
      traj.append(env.state.copy())
    obs_out[ep, -1] = np.concatenate([env.state, g]).astype(np.float32)
    bits_out[ep, -1] = env.swamp_bits.astype(np.uint8)
    traj = np.array(traj)
    died_out[ep] = int(env.dead)
    if is_random:
      mode_out[ep] = MODE['random']
      route_out[ep] = ROUTE['random']
    else:
      mode_out[ep] = MODE[memo.get('teacher_mode', 'immediate_shortcut')]
      fsafe_out[ep] = int(memo.get('force_safe', False))
      wait_out[ep] = int(memo.get('wait_count', 0))
      used_safe = bool(np.any(traj[:, 1] < 2.0))
      crossed = bool(np.any(traj[:, 0] > 6.0)) and not used_safe
      route_out[ep] = (ROUTE['shortcut'] if crossed else
                       ROUTE['safe_detour'] if used_safe else ROUTE['other'])
    succ.append(float(np.min(np.linalg.norm(traj - g, axis=1)) < 0.5))
    if (ep + 1) % 500 == 0:
      print(f'  {ep + 1}/{episodes} (reached@0.5 {np.mean(succ):.3f}, '
            f'died {died_out[:ep + 1].mean():.3f})', flush=True)

  succ = np.asarray(succ)
  inv = {v: k for k, v in MODE.items()}
  modes, counts = np.unique(mode_out, return_counts=True)
  meta = {
      'env_name': ENV, 'setting': 'windy_lethal',
      'episodes': int(episodes), 'seed': int(seed),
      'behavior_policy': f'windy_reactive_teacher(force_safe={force_safe_prob}, '
                         f'per-step dodge) + random_frac={random_frac}, '
                         f'teacher_noise={teacher_noise}',
      'random_frac': float(random_frac),
      'force_safe_prob': float(force_safe_prob),
      'teacher_noise': float(teacher_noise),
      'per_cell_swamp_prob': float(env.active_prob),
      'resample': 'every step, everywhere (wind); active swamp = terminal (lava)',
      'obs_dim': int(cfg.obs_dim), 'goal_dim': int(cfg.goal_dim),
      'action_dim': int(A), 'max_episode_steps': int(env.max_episode_steps),
      'episode_len_rows_L': int(L), 'obs_width_D': int(D),
      'n_transitions': int(episodes * (L - 1)),
      'teacher_mode_code': MODE, 'route_label_code': ROUTE,
      'teacher_mode_frequencies': {inv[int(m)]: int(c)
                                   for m, c in zip(modes, counts)},
      'died_rate_overall': float(died_out.mean()),
      'died_rate_teacher': (float(died_out[n_random:].mean())
                            if episodes > n_random else None),
      'reached_0p5_rate': float(succ.mean()),
      'audit_fields': ['swamp_bits', 'route_label', 'teacher_mode',
                       'force_safe', 'wait_count', 'entered_active_swamp'],
      'note': 'entered_active_swamp == DIED (lethal env); all audit fields '
              'AUDIT-ONLY; learner obs is [x,y,gx,gy] only',
  }
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  np.savez(out, obs=obs_out, act=act_out, swamp_bits=bits_out,
           route_label=route_out, teacher_mode=mode_out, force_safe=fsafe_out,
           wait_count=wait_out, entered_active_swamp=died_out,
           meta=np.array(json.dumps(meta)))
  digest = _sha256(out)
  json.dump(dict(path=os.path.abspath(out), sha256=digest,
                 size_bytes=int(os.path.getsize(out)),
                 obs_shape=list(obs_out.shape), act_shape=list(act_out.shape),
                 frozen=True, meta=meta),
            open(out + '.manifest.json', 'w'), indent=2)
  try:
    os.chmod(out, 0o444)
  except OSError:
    pass
  print(f'\nFROZEN windy dataset -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
  print(f'sha256 = {digest}')
  print(json.dumps(meta, indent=2))


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--episodes', type=int, default=6000)
  p.add_argument('--random_frac', type=float, default=0.2)
  p.add_argument('--force_safe_prob', type=float, default=0.05)
  p.add_argument('--teacher_noise', type=float, default=0.15)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', required=True)
  p.add_argument('--force', action='store_true')
  a = p.parse_args()
  if a.force and os.path.exists(a.out):
    try:
      os.chmod(a.out, 0o644)
    except OSError:
      pass
  collect(a.episodes, a.random_frac, a.force_safe_prob, a.teacher_noise,
          a.seed, a.out, force=a.force)


if __name__ == '__main__':
  main()
