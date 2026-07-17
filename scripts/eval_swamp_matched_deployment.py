"""Deployment-gap report for the MiniGrid-matched swamp setting.

Evaluates a trained OBSERVATIONAL CRL policy (no causal objective) under
all-clear / all-active / natural U and forced matched holding/fork starts, and
reports the full metric suite requested for the MiniGrid-matched comparison:
success, shortcut-attempt, safe-route, wait/stall, trapped, swamp-entry,
slowed-step count, recovery, SPL, worst-case success, deployment gap.

Compares the learner against: an always-safe controller, the matched teacher,
and (loaded from disk) the strong-swamp observational baseline.

Run:  python scripts/eval_swamp_matched_deployment.py \
        --ckpt swamp_matched_obs_s0/best.pkl --out artifacts/swamp_matched_deployment
Stops after the observational baseline report -- NO causal objective.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import load_nets, bfs_waypoints, polyline_len
from scripts.qualify_two_route_swamp import follower, swamp_blocked_walls, SWAMP, POST_CELL
from scripts.qualify_two_route_swamp_matched import make_matched_teacher

ENV = 'point_two_route_swamp_matched_v0'
SUCCESS_RADIUS = 0.5
HOLD_CENTER = np.array([2.5, 3.5])
FORK_CENTER = np.array([1.5, 3.5])
CONDITIONS = {
    'all_clear':   dict(bits=[0, 0, 0], start=None),
    'all_active':  dict(bits=[1, 1, 1], start=None),
    'natural':     dict(bits=None,       start=None),
    'holding_nat': dict(bits=None,       start=HOLD_CENTER),
    'fork_nat':    dict(bits=None,       start=FORK_CENTER),
}


def greedy_policy(greedy):
  def policy(s, g, memo):
    return np.asarray(greedy(np.concatenate([s, g]).astype(np.float32)), np.float32)
  return policy


def rollout(env, policy, bits=None, start=None):
  env.set_auto_resample(True)
  env.reset()
  if bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(bits)
  if start is not None:
    env.state = np.asarray(start, float).copy()
  g = env.goal.copy()
  traj = [env.state.copy()]
  bits_log = [env.swamp_bits]
  memo = {}
  for _ in range(env.max_episode_steps):
    env.step(np.asarray(policy(env.state.copy(), g, memo), np.float32))
    traj.append(env.state.copy())
    bits_log.append(env.swamp_bits)
  env.set_auto_resample(True)
  traj, bits_log = np.array(traj), np.array(bits_log)
  cells = [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                         np.array(env._walls.shape) - 1)) for p in traj]
  entered = any(c in SWAMP for c in cells)
  crossed = entered and any(c == POST_CELL for c in cells)
  used_safe = bool(np.any(traj[:, 1] < 2.0))
  slowed = sum(1 for t in range(len(bits_log) - 1)
               if cells[t + 1] in SWAMP and bits_log[t][SWAMP.index(cells[t + 1])])
  entered_active = slowed > 0
  disp = np.linalg.norm(np.diff(traj, axis=0), axis=1)
  # stall = a run of >=5 consecutive near-zero-motion steps
  run = maxrun = 0
  for d in disp:
    run = run + 1 if d < 0.05 else 0
    maxrun = max(maxrun, run)
  stall = maxrun >= 5
  min_d = float(np.min(np.linalg.norm(traj - g, axis=1)))
  success = bool(min_d < SUCCESS_RADIUS)
  path_len = float(disp.sum())
  wps = bfs_waypoints(env._walls if crossed else swamp_blocked_walls(env._walls),
                      traj[0], g)
  ref = polyline_len(wps) if wps else np.nan
  spl = (success * ref / max(path_len, ref)) if (np.isfinite(ref) and path_len > 1e-6) else float(success)
  return dict(success=success, min_dist=min_d, entered=entered, crossed=crossed,
              used_safe=used_safe, slowed=slowed, entered_active=entered_active,
              trapped=entered_active and not crossed,
              recovered=entered_active and success, stall=stall,
              path_len=path_len, spl=float(spl), traj=traj)


def summarize(eps):
  f = lambda k: float(np.mean([e[k] for e in eps]))
  active = [e for e in eps if e['entered_active']]
  return dict(
      n=len(eps), success=f('success'),
      shortcut_attempt_rate=f('entered'), crossed_rate=f('crossed'),
      safe_route_rate=float(np.mean([e['used_safe'] and not e['entered'] for e in eps])),
      wait_stall_rate=f('stall'), trapped_rate=f('trapped'),
      swamp_entry_rate=f('entered'), slowed_step_mean=f('slowed'),
      recovery_rate=(float(np.mean([e['recovered'] for e in active])) if active else None),
      spl=f('spl'))


def eval_all(env, policy, episodes):
  out = {}
  for name, c in CONDITIONS.items():
    eps = [rollout(env, policy, bits=c['bits'], start=c['start']) for _ in range(episodes)]
    out[name] = dict(summary=summarize(eps), eps=eps)
  return out


def _draw(ax, env, active=None):
  walls = env._walls
  H, W = walls.shape
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55, extent=[0, H, 0, W])
  for k, c in enumerate(SWAMP):
    if active is not None and active[k]:
      ax.add_patch(Rectangle(c, 1, 1, facecolor='tab:red', alpha=0.3))
    ax.add_patch(Rectangle(c, 1, 1, fill=False, hatch='///', edgecolor='darkorange', lw=1.3))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')


def plot(env, learner, out):
  fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
  for a, name in zip(ax, ('all_clear', 'all_active', 'natural')):
    _draw(a, env, CONDITIONS[name]['bits'] or [1, 1, 1])
    for e in learner[name]['eps'][:15]:
      t = e['traj']; col = 'tab:green' if e['success'] else 'tab:red'
      a.plot(t[:, 0], t[:, 1], '-', lw=1.0, alpha=0.7, color=col)
    a.scatter(*env.START, c='black', s=30); a.scatter(*env.GOAL, c='red', marker='*', s=110)
    s = learner[name]['summary']
    a.set_title('%s\nsucc=%.2f entry=%.2f trapped=%.2f' %
                (name, s['success'], s['swamp_entry_rate'], s['trapped_rate']), fontsize=9)
  fig.suptitle('MiniGrid-matched observational learner deployment', fontsize=11)
  fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=105); plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', default='artifacts/swamp_matched_deployment')
  ap.add_argument('--episodes', type=int, default=150)
  ap.add_argument('--strong_baseline',
                  default='artifacts/swamp_obs_deployment/deployment_report.json')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=123)
  _, _, greedy, step = load_nets(ENV, args.ckpt, cfg)
  rng = np.random.default_rng(0)

  learner = eval_all(env, greedy_policy(greedy), args.episodes)
  safe = eval_all(env, follower(swamp_blocked_walls(env._walls)), args.episodes)
  teacher = eval_all(env, make_matched_teacher(env, rng), args.episodes)

  L = {k: v['summary'] for k, v in learner.items()}
  S = {k: v['summary'] for k, v in safe.items()}
  T = {k: v['summary'] for k, v in teacher.items()}
  gap = L['all_clear']['success'] - L['all_active']['success']
  worst = min(L['all_clear']['success'], L['all_active']['success'])
  shortcut_bias = (L['all_active']['swamp_entry_rate'] >= 0.5
                   and L['all_clear']['swamp_entry_rate'] >= 0.5)

  strong = None
  if os.path.exists(args.strong_baseline):
    sb = json.load(open(args.strong_baseline))
    strong = dict(all_clear=sb['learner']['all_clear']['success'],
                  all_active=sb['learner']['all_active']['success'],
                  natural=sb['learner']['natural']['success'],
                  deployment_gap=sb['deployment_gap'])

  report = dict(
      setting='minigrid_matched', env=ENV, ckpt=os.path.abspath(args.ckpt), step=step,
      episodes=args.episodes,
      learner=L, always_safe=S, matched_teacher=T,
      deployment_gap=gap, worst_case_success=worst,
      shortcut_bias_detected=bool(shortcut_bias),
      strong_swamp_observational_baseline=strong,
      verdict=('CONFOUNDED_SHORTCUT_BIAS' if (shortcut_bias and gap >= 0.2)
               else 'NO_CLEAR_BIAS'))
  json.dump(report, open(os.path.join(args.out, 'deployment_report.json'), 'w'), indent=2)
  plot(env, learner, os.path.join(args.out, 'deployment_trajectories.png'))

  print('=' * 88)
  print('MINIGRID-MATCHED OBSERVATIONAL DEPLOYMENT  (step %s)' % step)
  print('=' * 88)
  hdr = (f'{"cond":>12} | {"succ":>5} {"entry":>6} {"safe":>5} {"stall":>6} '
         f'{"trap":>5} {"slow":>5} {"recov":>6} {"spl":>5}')
  for tag, tab in (('LEARNER', L), ('ALWAYS-SAFE', S), ('MATCHED-TEACHER', T)):
    print(f'\n[{tag}]'); print(hdr)
    for name in CONDITIONS:
      s = tab[name]
      rc = '  None' if s['recovery_rate'] is None else f'{s["recovery_rate"]:>6.2f}'
      print(f'{name:>12} | {s["success"]:>5.2f} {s["swamp_entry_rate"]:>6.2f} '
            f'{s["safe_route_rate"]:>5.2f} {s["wait_stall_rate"]:>6.2f} '
            f'{s["trapped_rate"]:>5.2f} {s["slowed_step_mean"]:>5.1f} {rc} {s["spl"]:>5.2f}')
  print('-' * 88)
  print(f'deployment gap (learner) = {gap:+.2f} | worst-case success = {worst:.2f}')
  if strong:
    print(f'strong-swamp baseline: clear={strong["all_clear"]:.2f} '
          f'active={strong["all_active"]:.2f} gap={strong["deployment_gap"]:+.2f}')
  print('VERDICT:', report['verdict'])
  print('saved', os.path.join(args.out, 'deployment_report.json'))


if __name__ == '__main__':
  main()
