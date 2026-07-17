"""Deployment-gap probe for an OBSERVATIONAL policy on the two-route swamp env.

Given a trained checkpoint (vanilla / observational CRL, no causal objective),
measure whether it formed a SHORTCUT BIAS and therefore suffers a CLOSED-SWAMP
DEPLOYMENT GAP:

  * The learner never observes the swamp bits, so at the (fixed) holding cell it
    must commit to ONE route. The teacher's data is dominated by successful
    SHORTCUT trajectories (it only shortcuts when the swamp is clear), so an
    observational learner is expected to prefer the shortcut REGARDLESS of the
    (hidden) swamp state.
  * We deploy the greedy policy under forced swamp configurations and compare:
        success(all-clear)  vs  success(all-active)   -> deployment gap
    plus how often it ATTEMPTS the shortcut (enters the swamp corridor) and how
    often it gets TRAPPED (enters an active swamp cell and fails to cross).

An always-safe reference (detour follower) is included: it ignores U and should
succeed under every configuration (gap ~ 0), unlike the biased learner.

NO causal / pessimistic objective here -- this only establishes that the
observational pathology exists before we try to fix it.

Run:  python scripts/eval_swamp_deployment.py --ckpt swamp_obs_s0/best.pkl \
        --out artifacts/swamp_obs_deployment --episodes 100
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
from crl.report_maze import load_nets
from scripts.qualify_two_route_swamp import (
    follower, swamp_blocked_walls, SWAMP, POST_CELL, HOLDING)

SUCCESS_RADIUS = 0.5
ENV = 'point_two_route_swamp_v0'
CONFIGS = {'all_clear': [0, 0, 0], 'all_active': [1, 1, 1], 'natural': None}


def greedy_policy(greedy):
  def policy(s, g, memo):
    obs = np.concatenate([s, g]).astype(np.float32)
    return np.asarray(greedy(obs), np.float32)
  return policy


def rollout(env, policy, force_bits):
  env.set_auto_resample(True)
  env.reset()
  if force_bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(force_bits)
  g = env.goal.copy()
  traj = [env.state.copy()]
  bits_log = [env.swamp_bits]
  memo = {}
  for _ in range(env.max_episode_steps):
    a = policy(env.state.copy(), g, memo)
    env.step(a)
    traj.append(env.state.copy())
    bits_log.append(env.swamp_bits)
  env.set_auto_resample(True)
  traj = np.array(traj)
  bits_log = np.array(bits_log)
  walls = env._walls
  cells = [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                         np.array(walls.shape) - 1)) for p in traj]
  entered = any(c in SWAMP for c in cells)
  crossed = entered and any(c == POST_CELL for c in cells)
  used_safe = bool(np.any(traj[:, 1] < 2.0))
  # trapped: entered a swamp cell that was ACTIVE for the step, and never crossed
  trapped = entered and (not crossed) and any(
      cells[t + 1] in SWAMP and bits_log[t][SWAMP.index(cells[t + 1])]
      for t in range(len(bits_log) - 1))
  min_d = float(np.min(np.linalg.norm(traj - g, axis=1)))
  return dict(traj=traj, success=bool(min_d < SUCCESS_RADIUS), min_dist=min_d,
              shortcut_attempt=entered, crossed=crossed, used_safe=used_safe,
              trapped=trapped)


def summarize(eps):
  n = len(eps)
  f = lambda k: float(np.mean([e[k] for e in eps]))
  return dict(n=n, success=f('success'), min_dist=f('min_dist'),
              shortcut_attempt_rate=f('shortcut_attempt'),
              crossed_rate=f('crossed'), detour_rate=f('used_safe'),
              trapped_rate=f('trapped'))


def eval_policy(env, policy, episodes, seed=0):
  out = {}
  for name, cfg in CONFIGS.items():
    eps = [rollout(env, policy, cfg) for _ in range(episodes)]
    out[name] = {'summary': summarize(eps), 'eps': eps}
  return out


def _draw(ax, env, active=None):
  walls = env._walls
  H, W = walls.shape
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55, extent=[0, H, 0, W])
  for k, c in enumerate(SWAMP):
    if active is not None and active[k]:
      ax.add_patch(Rectangle(c, 1, 1, facecolor='tab:red', alpha=0.3))
    ax.add_patch(Rectangle(c, 1, 1, fill=False, hatch='///',
                           edgecolor='darkorange', lw=1.3))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')


def plot(env, results, out):
  fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
  for a, name in zip(ax, ('all_clear', 'all_active', 'natural')):
    active = CONFIGS[name]
    _draw(a, env, active if active else [1, 1, 1])
    for e in results[name]['eps'][:15]:
      t = e['traj']
      col = 'tab:green' if e['success'] else 'tab:red'
      a.plot(t[:, 0], t[:, 1], '-', lw=1.0, alpha=0.7, color=col)
    a.scatter(*env.START, c='black', s=30, zorder=5)
    a.scatter(*env.GOAL, c='red', marker='*', s=110, zorder=5)
    s = results[name]['summary']
    a.set_title('%s\nsucc=%.2f  shortcut=%.2f  trapped=%.2f'
                % (name, s['success'], s['shortcut_attempt_rate'],
                   s['trapped_rate']), fontsize=9)
  fig.suptitle('Observational policy deployment (green=goal, red=fail; '
               'red fill=active swamp)', fontsize=11)
  fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=105); plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', default='artifacts/swamp_obs_deployment')
  ap.add_argument('--episodes', type=int, default=100)
  ap.add_argument('--seed', type=int, default=123)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)
  _, _, greedy, step = load_nets(ENV, args.ckpt, cfg)

  learner = eval_policy(env, greedy_policy(greedy), args.episodes)
  safe = eval_policy(env, follower(swamp_blocked_walls(env._walls)), args.episodes)

  L = {k: v['summary'] for k, v in learner.items()}
  S = {k: v['summary'] for k, v in safe.items()}
  gap = L['all_clear']['success'] - L['all_active']['success']
  safe_gap = S['all_clear']['success'] - S['all_active']['success']
  # shortcut bias: does it attempt the shortcut just as often when the swamp is
  # ACTIVE as when clear (i.e., it ignores the hidden U)?
  shortcut_bias = (L['all_active']['shortcut_attempt_rate'] >= 0.5
                   and L['all_clear']['shortcut_attempt_rate'] >= 0.5)
  gap_present = gap >= 0.3 and L['all_active']['trapped_rate'] >= 0.3

  report = dict(
      env=ENV, ckpt=os.path.abspath(args.ckpt), step=step,
      episodes=args.episodes,
      learner=L, always_safe=S,
      deployment_gap=gap, safe_deployment_gap=safe_gap,
      shortcut_bias_detected=bool(shortcut_bias),
      deployment_gap_present=bool(gap_present),
      verdict=('CONFOUNDED_SHORTCUT_BIAS' if (shortcut_bias and gap_present)
               else 'NO_CLEAR_BIAS'),
      interpretation=(
          'Observational policy prefers the shortcut regardless of the hidden '
          'swamp state and is trapped when it is active -> a causal/pessimistic '
          'objective is motivated.' if (shortcut_bias and gap_present) else
          'No clear shortcut-bias/deployment-gap; revisit dataset or training '
          'before adding a causal objective.'))
  json.dump(report, open(os.path.join(args.out, 'deployment_report.json'), 'w'),
            indent=2)
  plot(env, learner, os.path.join(args.out, 'deployment_trajectories.png'))

  print('=' * 74)
  print('SWAMP OBSERVATIONAL DEPLOYMENT PROBE  (step %s)' % step)
  print('=' * 74)
  hdr = f'{"condition":>12} | {"succ":>5} {"shortcut":>9} {"crossed":>8} {"detour":>7} {"trapped":>8}'
  for label, tab in (('LEARNER', L), ('ALWAYS-SAFE', S)):
    print(f'\n[{label}]')
    print(hdr)
    for name in ('all_clear', 'all_active', 'natural'):
      s = tab[name]
      print(f'{name:>12} | {s["success"]:>5.2f} {s["shortcut_attempt_rate"]:>9.2f} '
            f'{s["crossed_rate"]:>8.2f} {s["detour_rate"]:>7.2f} {s["trapped_rate"]:>8.2f}')
  print('-' * 74)
  print(f'deployment gap (learner) = succ(clear) - succ(active) = {gap:+.2f} '
        f'| always-safe gap = {safe_gap:+.2f}')
  print('VERDICT:', report['verdict'])
  print(report['interpretation'])
  print('saved', os.path.join(args.out, 'deployment_report.json'))


if __name__ == '__main__':
  main()
