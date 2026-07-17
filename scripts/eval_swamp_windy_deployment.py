"""Deployment eval for the windy-LETHAL swamp env (point_two_route_swamp_windy_v0).

Conditions:
  all_clear   bits frozen [0,0,0]  (shortcut is free)
  all_active  bits frozen [1,1,1]  (corridor entry = instant death)
  natural     per-step resampling  (the env's real process)

Per condition: success (min dist < 0.5), swamp-corridor entry rate, DIED
rate, safe-route rate. Always-safe reference included. Compare FINAL
checkpoints across arms (never best.pkl -- rollout-based selection leaks).

Run:
  python -m scripts.eval_swamp_windy_deployment --ckpt <run>/final.pkl \
      --out artifacts/<name> [--episodes 100]
"""
import argparse
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import load_nets, make_oracle
from scripts.qualify_two_route_swamp import swamp_blocked_walls

ENV = 'point_two_route_swamp_windy_v0'
CONDITIONS = {'all_clear': [0, 0, 0], 'all_active': [1, 1, 1],
              'natural': None}


def rollout(env, policy, bits):
  env.set_auto_resample(True)
  env.reset()
  if bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(bits)
  g = env.goal.copy()
  memo = {}
  traj = [env.state.copy()]
  for _ in range(env.max_episode_steps):
    env.step(np.asarray(policy(env.state.copy(), g, memo), np.float32))
    traj.append(env.state.copy())
  env.set_auto_resample(True)
  traj = np.array(traj)
  cells = [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                         np.array(env._walls.shape) - 1)) for p in traj]
  return dict(
      success=float(np.min(np.linalg.norm(traj - g, axis=1)) < 0.5),
      entry=float(any(c in env.SWAMP_CELLS for c in cells)),
      died=float(env.dead),
      safe=float(np.any(traj[:, 1] < 2.0)))


def run_policy(env, policy, episodes):
  out = {}
  for cond, bits in CONDITIONS.items():
    rows = [rollout(env, policy, bits) for _ in range(episodes)]
    out[cond] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--episodes', type=int, default=100)
  args = ap.parse_args()

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=123)
  nets, state, greedy_np, step = load_nets(ENV, args.ckpt, cfg)

  def learner(s, g, memo):
    return greedy_np(np.concatenate([s, g]).astype(np.float32))

  safe_oracle = make_oracle(swamp_blocked_walls(env._walls))
  learner_r = run_policy(env, learner, args.episodes)
  safe_r = run_policy(env, safe_oracle, args.episodes)

  print(f'WINDY-LETHAL DEPLOYMENT  ckpt={args.ckpt} (step {step})')
  for name, res in (('LEARNER', learner_r), ('ALWAYS-SAFE', safe_r)):
    print(f'[{name}]')
    print(f'{"cond":>11s} |  succ  entry  died  safe')
    for cond, m in res.items():
      print(f'{cond:>11s} |  {m["success"]:.2f}   {m["entry"]:.2f}  '
            f'{m["died"]:.2f}  {m["safe"]:.2f}')
  gap = learner_r['all_clear']['success'] - learner_r['all_active']['success']
  worst = min(v['success'] for v in learner_r.values())
  verdict = ('CONFOUNDED_SHORTCUT_BIAS' if gap > 0.5 else 'NO_CLEAR_BIAS')
  print(f'gap(clear-active) = {gap:+.2f} | worst-case = {worst:.2f} | {verdict}')

  os.makedirs(args.out, exist_ok=True)
  json.dump(dict(ckpt=args.ckpt, step=int(step), episodes=args.episodes,
                 learner=learner_r, always_safe=safe_r,
                 gap=float(gap), worst_case=float(worst), verdict=verdict),
            open(os.path.join(args.out, 'deployment_report.json'), 'w'),
            indent=1)
  print(f'saved {args.out}/deployment_report.json')


if __name__ == '__main__':
  main()
