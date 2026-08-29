"""Critic-level read-out for the windy-swamp sweep, independent of the actor.

WHY THIS EXISTS. Deployment success is a joint test of the critic AND the
policy-extraction layer. On this exact benchmark the extraction layer has
already been the deciding factor once: the earlier Manski runs had a correct
critic on every seed while the actor's route still varied, and AWR was what
fixed it. This branch has NO AWR (crl.config has no use_awr; that knob lives
only on manski-port-archive), so a flat worst_case across arms would be
ambiguous -- "the bank did not move the critic" and "the critic moved but the
actor did not follow" look identical from the deployment table alone.

This probe reads the critic directly, at the ONE input the deployed policy
actually consults: the commanded goal (8.5, 3.5).

  fork margin  =  f(s_fork, a_toward_safe, g_goal) - f(s_fork, a_toward_shortcut, g_goal)

  > 0  the critic ranks the safe route above the shortcut  (deconfounded)
  < 0  the critic prefers the shortcut                     (confounded)

It also reports the mean critic value assigned to the BANK states as goals.
That is the quantity the failure-negative term directly pushes down, so it
tells you whether the bank did anything at all -- separately from whether that
had any effect on the route decision. Those two can dissociate, and if they do
(bank logits fall, fork margin unmoved) that is the concrete signature of the
bank acting on a goal input the policy never queries.

Usage:
  python scripts/probe_swamp_windy_critic.py                # every run dir
  python scripts/probe_swamp_windy_critic.py --ckpt <dir>/final.pkl
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

# This probe is normally run WHILE the sweep is training. Do not let it grab
# the default 75% of VRAM out from under the live run.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.08')

import jax                                     # noqa: E402
import jax.numpy as jnp                        # noqa: E402

from crl import envs as envs_mod               # noqa: E402
from crl.report_maze import load_nets          # noqa: E402
# Take the network shape from the launcher itself (repr_dim 16, hidden
# 256x256, twin_q) -- a local Config() would use the library defaults
# (repr_dim 64) and fail to load these checkpoints.
from run_swamp_windy_failneg import build_cfg  # noqa: E402

ENV = 'point_two_route_swamp_windy_v0'
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
FORK = np.array([1.5, 3.5])          # the route decision point
GOAL = np.array([8.5, 3.5])          # the ONLY goal ever commanded

# Out of the fork, +x continues to the holding cell / shortcut and -y drops to
# (1,2) and the safe lower route.
#
# The margin is taken between the BEST action in each direction, not between
# two hand-picked unit vectors. Measured on real checkpoints, f varies by ~8
# logits across the action square, and both [1,0] and [0,-1] sit near the TOP
# of it -- so differencing those two subtracts two near-optimal values and
# yields a tiny, fragile number that says nothing about the route ranking.
# "Best achievable going each way" is the quantity the actor is effectively
# choosing between.
GRID = 21                            # 21x21 sweep of [-1,1]^2
# The two half-planes MUST be disjoint. A first version used
# (a_y < -0.3) and (a_x > 0.3), which overlap: [+1.0, -0.4] satisfies both, so
# the same action could be the argmax of each side and the margin came out
# exactly 0.000 (seen on seed 2). Classify instead by which move DOMINATES,
# which partitions the square.
SAFE_WARD = lambda a: (a[:, 1] < -0.3) & (a[:, 1] < -np.abs(a[:, 0]))
SHORT_WARD = lambda a: (a[:, 0] > 0.3) & (a[:, 0] > np.abs(a[:, 1]))


def critic_fn(nets, state, cfg):
  """f(s, a, g) with the twin-Q min, exactly as the actor consumes it."""
  def f(states, actions, goals):
    obs = jnp.concatenate([jnp.asarray(states, jnp.float32),
                           jnp.asarray(goals, jnp.float32)], axis=1)
    out = nets.q_network.apply(state.q_params, obs, jnp.asarray(actions,
                                                                jnp.float32))
    out = jnp.diagonal(out, axis1=0, axis2=1) if out.ndim >= 2 else out
    return out
  return f


def probe(ckpt, bank_states, jitter=0.05, n=256, seed=0):
  # exact training recipe; arm/alpha do not affect the network shape
  cfg = build_cfg('baseline', 0.0, 0, '', 1)
  envs_mod.make_env(ENV, cfg, seed=0)          # fills obs/goal/action dims
  nets, state, _, step = load_nets(ENV, ckpt, cfg)
  rng = np.random.default_rng(seed)

  gx, gy = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID))
  acts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

  def sweep(state_pt):
    """f(state_pt, a, GOAL) for every a on the grid."""
    m = len(acts)
    s = np.repeat(np.asarray(state_pt, np.float32)[None, :], m, axis=0)
    g = np.repeat(GOAL[None, :], m, axis=0)
    obs = np.concatenate([s, g], axis=1).astype(np.float32)
    out = np.asarray(nets.q_network.apply(state.q_params, jnp.asarray(obs),
                                          jnp.asarray(acts)))
    if out.ndim == 3:
      out = out.min(axis=-1)                   # twin-Q min, as the actor uses
    return np.einsum('ii->i', out) if out.ndim == 2 else out

  # Average the action-value surface over a small cloud of fork states, so the
  # ranking is not read off one lucky point.
  pts = FORK[None, :] + rng.normal(0, jitter, (max(n // 8, 8), 2))
  F = np.stack([sweep(p) for p in pts])        # [pts, actions]
  f_mean, f_sd = F.mean(axis=0), F.std(axis=0)

  m_safe, m_short = SAFE_WARD(acts), SHORT_WARD(acts)
  i_safe = int(np.argmax(np.where(m_safe, f_mean, -np.inf)))
  i_short = int(np.argmax(np.where(m_short, f_mean, -np.inf)))
  v_safe, sd_safe = float(f_mean[i_safe]), float(f_sd[i_safe])
  v_short, sd_short = float(f_mean[i_short]), float(f_sd[i_short])
  i_best = int(np.argmax(f_mean))
  best_a = acts[i_best].tolist()
  spread = float(f_mean.max() - f_mean.min())

  # What value does the critic give the BANK states AS GOALS, from the fork?
  nb = min(len(bank_states), n)
  sb = FORK[None, :] + rng.normal(0, jitter, (nb, 2))
  ob = np.concatenate([sb, bank_states[:nb]], axis=1).astype(np.float32)
  ab = np.repeat(acts[i_short][None, :], nb, axis=0)   # heading into the swamp
  outb = np.asarray(nets.q_network.apply(state.q_params, jnp.asarray(ob),
                                         jnp.asarray(ab)))
  if outb.ndim == 3:
    outb = outb.min(axis=-1)
  vb = float(np.einsum('ii->i', outb).mean()) if outb.ndim == 2 \
      else float(outb.mean())

  return {'step': int(step), 'v_safe': v_safe, 'v_shortcut': v_short,
          'fork_margin': v_safe - v_short, 'sd_safe': sd_safe,
          'sd_shortcut': sd_short, 'v_bank_as_goal': vb,
          'best_action': best_a, 'action_value_spread': spread,
          'a_safe': acts[i_safe].tolist(), 'a_short': acts[i_short].tolist()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default='')
  ap.add_argument('--glob', default='swamp_windy_*_s[0-9]/final.pkl')
  ap.add_argument('--json', default='artifacts/swamp_windy_failure_bank/'
                                    'critic_fork_margin.json')
  args = ap.parse_args()

  with np.load(BANK, allow_pickle=True) as b:
    bank = np.asarray(b['goals'], np.float32)

  ckpts = [args.ckpt] if args.ckpt else sorted(glob.glob(args.glob))
  if not ckpts:
    raise SystemExit(f'no checkpoints matched {args.glob}')

  rows, out = [], {}
  for c in ckpts:
    name = os.path.basename(os.path.dirname(c))
    try:
      r = probe(c, bank)
    except Exception as e:                     # pylint: disable=broad-except
      print(f'  SKIP {name}: {type(e).__name__}: {e}')
      continue
    out[name] = r
    rows.append((name, r))

  print('=' * 96)
  print('CRITIC FORK MARGIN at the commanded goal (8.5, 3.5) -- actor-independent')
  print('  margin = f(fork, toward-safe, goal) - f(fork, toward-shortcut, goal)')
  print('  margin > 0  => critic ranks the SAFE route higher (deconfounded)')
  print('=' * 96)
  print(f'{"run":<30}{"f(safe)":>9}{"f(short)":>9}{"margin":>9}'
        f'{"spread":>9}{"argmax a":>15}{"f(bank)":>10}')
  print('-' * 96)
  for name, r in rows:
    flag = '  safe' if r['fork_margin'] > 0 else '  short'
    ba = f'[{r["best_action"][0]:+.1f},{r["best_action"][1]:+.1f}]'
    print(f'{name:<30}{r["v_safe"]:>9.3f}{r["v_shortcut"]:>9.3f}'
          f'{r["fork_margin"]:>9.3f}{r["action_value_spread"]:>9.2f}'
          f'{ba:>15}{r["v_bank_as_goal"]:>10.3f}{flag}')
  print('\n  margin = max f over safe-ward actions - max f over shortcut-ward '
        'actions,\n  averaged over a cloud of fork states. spread = range of f '
        'across the whole\n  action square (a small spread would mean the '
        'diagnostic cannot discriminate).')

  # group by arm
  byarm = {}
  for name, r in rows:
    m = re.match(r'swamp_windy_(.+)_s\d+$', name)
    if m:
      byarm.setdefault(m.group(1), []).append(r)
  if byarm:
    print('\nby arm (mean over seeds):')
    print(f'{"arm":<28}{"fork_margin":>14}{"f(bank as goal)":>18}')
    for a, rs in byarm.items():
      fm = np.mean([x['fork_margin'] for x in rs])
      vb = np.mean([x['v_bank_as_goal'] for x in rs])
      print(f'{a:<28}{fm:>14.3f}{vb:>18.3f}')
    print('\n  If f(bank as goal) falls with alpha but fork_margin does NOT '
          'move,\n  the bank changed only the goal input the policy never '
          'queries -- which is\n  the failure mode predicted before the run.')

  if args.json:
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'\nwrote {args.json}')


if __name__ == '__main__':
  main()
