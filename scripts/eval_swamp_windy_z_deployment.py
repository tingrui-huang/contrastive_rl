"""Deployment evaluation for point_two_route_swamp_windy_z_v0.

Mirrors scripts/eval_swamp_windy_deployment.py condition-for-condition and
metric-for-metric. A separate file is needed only because the 2-D evaluator
builds the policy input as np.concatenate([env.state, env.goal]) -- width 4 --
while the Z learner needs the 3-D state and goal, width 6. The maze, horizon,
conditions, success criterion and metric definitions are otherwise identical,
so numbers are directly comparable to the 2-D line.

Conditions (bits frozen unless stated):
  all_clear   [0,0,0]   the shortcut is free
  all_active  [1,1,1]   corridor entry is instant death -- the worst case
  natural     resampled every step, the env's real process

Additionally every one of the 8 hidden bit patterns is run separately, which is
the "success stratified by hidden swamp pattern" the 2-D evaluator does not
provide.

Metrics per condition: success (min distance to goal < 0.5), swamp-corridor
entry rate, died, safe-route usage.

SECONDARY EVIDENCE. Policy success must not be used to overrule the critic
audit; see scripts/audit_swamp_z_failneg.py.

Usage:
  python -m scripts.eval_swamp_windy_z_deployment --ckpt <run>/final.pkl \
      --out artifacts/swamp_windy_z_failneg_v1/<name> --episodes 100
"""
import argparse
import itertools
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import load_nets, make_oracle
from scripts.qualify_two_route_swamp import swamp_blocked_walls

ENV = 'point_two_route_swamp_windy_z_v0'
CONDITIONS = {'all_clear': [0, 0, 0], 'all_active': [1, 1, 1], 'natural': None}


def rollout(env, policy, bits, uses_3d):
  env.set_auto_resample(True)
  env.reset()
  if bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(bits)
  g2 = env.goal.copy()
  memo = {}
  traj = [env.state.copy()]
  for _ in range(env.max_episode_steps):
    if uses_3d:
      a = policy(np.concatenate([env.state_z, env.goal_z]).astype(np.float32))
    else:
      a = policy(env.state.copy(), g2, memo)
    env.step(np.asarray(a, np.float32))
    traj.append(env.state.copy())
  env.set_auto_resample(True)
  traj = np.array(traj)
  cells = [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                         np.array(env._walls.shape) - 1)) for p in traj]
  return dict(
      success=float(np.min(np.linalg.norm(traj - g2, axis=1)) < 0.5),
      entry=float(any(c in env.SWAMP_CELLS for c in cells)),
      died=float(env.dead),
      safe=float(np.any(traj[:, 1] < 2.0)),
      final_z=float(env.z))


def run_policy(env, policy, episodes, uses_3d, conditions=CONDITIONS):
  out = {}
  for cond, bits in conditions.items():
    rows = [rollout(env, policy, bits, uses_3d) for _ in range(episodes)]
    out[cond] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--episodes', type=int, default=100)
  ap.add_argument('--seed', type=int, default=123)
  args = ap.parse_args()

  cfg = Config(env_name=ENV)
  # The checkpoint was trained with z_physical; load_nets rebuilds the networks
  # from cfg, so the same obs_norm settings must be present or the scaling
  # would silently not be applied at evaluation time.
  cfg.obs_norm_mode = 'z_physical'
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)
  cfg.obs_norm_z_scale = abs(env.z_min)
  nets, state, greedy_np, step = load_nets(ENV, args.ckpt, cfg)

  safe_oracle = make_oracle(swamp_blocked_walls(env._walls))
  learner_r = run_policy(env, greedy_np, args.episodes, True)
  safe_r = run_policy(env, safe_oracle, args.episodes, False)

  # stratified by hidden swamp pattern
  pats = {'bits_%d%d%d' % p: list(p)
          for p in itertools.product([0, 1], repeat=3)}
  strat = run_policy(env, greedy_np, args.episodes, True, pats)

  print('=' * 72)
  print('Z DEPLOYMENT EVAL -- %s  (step %d)' % (args.ckpt, step))
  print('=' * 72)
  for name, r in (('learner', learner_r), ('always-safe', safe_r)):
    print('  %s' % name)
    print('  %11s |  succ  entry  died  safe  final_z' % 'cond')
    for cond, m in r.items():
      print('  %11s |  %.2f   %.2f  %.2f  %.2f   %+.3f'
            % (cond, m['success'], m['entry'], m['died'], m['safe'],
               m['final_z']))
  print('\n  stratified by hidden swamp pattern (learner)')
  print('  %11s |  succ  entry  died  safe' % 'pattern')
  for k, m in strat.items():
    print('  %11s |  %.2f   %.2f  %.2f  %.2f'
          % (k, m['success'], m['entry'], m['died'], m['safe']))

  gap = learner_r['all_clear']['success'] - learner_r['all_active']['success']
  worst = min(v['success'] for v in learner_r.values())
  verdict = ('CONFOUNDED_SHORTCUT_BIAS' if gap > 0.5 and worst < 0.2
             else 'ROBUST' if worst > 0.8 else 'PARTIAL')
  print('\n  clear-active success gap %.2f   worst-case %.2f   -> %s'
        % (gap, worst, verdict))

  os.makedirs(args.out, exist_ok=True)
  rep = {'checkpoint': args.ckpt, 'step': int(step), 'env': ENV,
         'episodes': args.episodes, 'seed': args.seed,
         'obs_norm_mode': cfg.obs_norm_mode,
         'obs_norm_z_scale': cfg.obs_norm_z_scale,
         'learner': learner_r, 'always_safe': safe_r,
         'learner_by_bit_pattern': strat,
         'clear_active_gap': float(gap), 'worst_case': float(worst),
         'verdict': verdict}
  p = os.path.join(args.out, 'deployment_report.json')
  with open(p, 'w') as f:
    json.dump(rep, f, indent=2)
  print('wrote %s' % p)


if __name__ == '__main__':
  main()
