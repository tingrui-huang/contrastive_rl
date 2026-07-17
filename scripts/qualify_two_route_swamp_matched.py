"""MiniGrid-matched continuous confounder: env + teacher qualification.

Apples-to-apples continuous-control counterpart of the MiniGrid WindyCorridor
setting. Same two-route swamp geometry/dynamics as the strong stress test, but:
  * per-cell swamp activation p = 0.10 (vs 0.20 strong);
  * MiniGrid-matched TEACHER:
      - episode-level force_safe ~ Bernoulli(0.05), INDEPENDENT of U;
      - if force_safe: take the long safe route regardless of the swamp;
      - else: go to the holding cell; if any swamp cell is active, WAIT (the
        config resamples next step) and keep waiting until all three are clear,
        then traverse the shortcut;
      - never knowingly enters an active swamp; ~100% success in both branches.

The 5% safe-route is behavior-policy randomness, NOT the Manski propensity (the
propensity is local/single-step at the holding/fork and is estimated later; it
is deliberately NOT computed here). This script only certifies the env + teacher
before any dataset is frozen. The strong point_two_route_swamp_v0 is untouched.

Run:  python scripts/qualify_two_route_swamp_matched.py --out artifacts/swamp_matched_qual
Exit 0 iff all 13 gates pass.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from crl import envs as envs_mod
from crl.config import Config
from scripts.qualify_two_route_swamp import (
    _follow, follower, swamp_blocked_walls, SWAMP, HOLDING, POST_CELL,
    HOLD_CENTER, matched_states)

ENV = 'point_two_route_swamp_matched_v0'
SUCCESS_RADIUS = 0.5
P = 0.10                       # per-cell swamp activation probability
FORCE_SAFE_P = 0.05            # episode-level safe-route probability (indep. of U)
TEACHER_MODE = {'random': 0, 'forced_safe': 1, 'immediate_shortcut': 2,
                'wait_shortcut': 3}
ROUTE_CODE = {'random': 0, 'shortcut': 1, 'safe_detour': 2, 'other': 3}
ALL_CONFIGS = [[int(b) for b in np.binary_repr(m, 3)] for m in range(8)]


# --------------------------------------------------------------------------- #
# MiniGrid-matched teacher (also imported by the collector)                    #
# --------------------------------------------------------------------------- #
def make_matched_teacher(env, rng, force_safe_prob=FORCE_SAFE_P):
  """Behavior policy. Decides force_safe once per episode (fresh memo), then:
     forced_safe -> detour; else approach holding, wait-until-clear, shortcut."""
  base = env._walls
  blocked = swamp_blocked_walls(base)

  def policy(s, g, memo):
    if 'phase' not in memo:                       # first step of the episode
      memo['force_safe'] = bool(rng.random() < force_safe_prob)
      memo['wait_count'] = 0
      if memo['force_safe']:
        memo['phase'] = 'long'
        memo['teacher_mode'] = 'forced_safe'
      else:
        memo['phase'] = 'approach'
    if memo['phase'] == 'approach':
      if np.linalg.norm(s - HOLD_CENTER) < 0.35:
        memo['phase'] = 'decide'
      else:
        return np.clip(HOLD_CENTER - s, -1, 1).astype(np.float32)
    if memo['phase'] == 'decide':
      bits = env.swamp_bits
      if bits.any():                              # any active -> wait in place
        memo['wait_count'] += 1
        return np.zeros(2, np.float32)            # env resamples next step
      memo['phase'] = 'short'                     # all clear -> commit shortcut
      memo['teacher_mode'] = ('immediate_shortcut' if memo['wait_count'] == 0
                              else 'wait_shortcut')
    return _follow(base if memo['phase'] == 'short' else blocked, s, g, memo)
  return policy


def _cells(traj, walls):
  return [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                        np.array(walls.shape) - 1)) for p in traj]


def rollout_matched(env, teacher, rng=None, force_bits=None, noise=0.0):
  """Run the matched teacher for one episode; return the full audit record."""
  env.set_auto_resample(True)
  env.reset()
  if force_bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(force_bits)
  g = env.goal.copy()
  memo = {}
  traj = [env.state.copy()]
  bits_log = [env.swamp_bits]
  for _ in range(env.max_episode_steps):
    a = np.asarray(teacher(env.state.copy(), g, memo), np.float32)
    if noise > 0 and np.any(a != 0):
      a = np.clip(a + (rng or env._rng).normal(0, noise, 2), -1, 1).astype(np.float32)
    env.step(a)
    traj.append(env.state.copy())
    bits_log.append(env.swamp_bits)
  env.set_auto_resample(True)
  traj = np.array(traj)
  bits_log = np.array(bits_log)
  cells = _cells(traj, env._walls)
  entered = any(c in SWAMP for c in cells)
  crossed = entered and any(c == POST_CELL for c in cells)
  used_safe = bool(np.any(traj[:, 1] < 2.0))
  entered_active = int(any(
      cells[t + 1] in SWAMP and bits_log[t][SWAMP.index(cells[t + 1])]
      for t in range(len(bits_log) - 1)))
  min_d = float(np.min(np.linalg.norm(traj - g, axis=1)))
  route = ('shortcut' if crossed else 'safe_detour' if used_safe and not entered
           else 'other')
  return dict(traj=traj, bits_log=bits_log, memo=memo,
              success=bool(min_d < SUCCESS_RADIUS), min_dist=min_d,
              crossed=crossed, used_safe=used_safe, entered_active=entered_active,
              force_safe=bool(memo.get('force_safe', False)),
              wait_count=int(memo.get('wait_count', 0)),
              teacher_mode=memo.get('teacher_mode', 'other'),
              route=route, initial_bits=bits_log[0].copy())


def matched_action(env, teacher, rng, s, bits, phase='decide'):
  """Matched-teacher action at state s under FORCED bits (fresh episode memo).

  phase='decide' probes the decision region (holding); phase='approach' probes
  the U-invariant prefix (the teacher only reads U once it reaches 'decide')."""
  env.set_auto_resample(False)
  env.set_swamp(bits)
  memo = {'phase': phase, 'wait_count': 0, 'force_safe': False}
  a = np.asarray(teacher(np.asarray(s, float).copy(), env.GOAL.copy(), memo), float)
  env.set_auto_resample(True)
  return a


# --------------------------------------------------------------------------- #
# gates                                                                         #
# --------------------------------------------------------------------------- #
def g1_g2_g3_stats(env):
  env._action_noise = 0.0
  env.set_auto_resample(True)
  env.reset()
  logs, changes, prev = [], 0, env.swamp_bits
  n = 6000
  for _ in range(n):
    env.step(np.zeros(2))
    b = env.swamp_bits
    logs.append(b)
    changes += int(np.any(b != prev))
    prev = b
  L = np.array(logs, float)
  rates = L.mean(0)
  corr = np.corrcoef(L.T)
  offdiag = float(np.max(np.abs(corr[~np.eye(3, dtype=bool)])))
  # freeze inside corridor
  env.set_swamp([0, 1, 0]); env.state = np.array([3.5, 3.5])
  fchg, prev = 0, env.swamp_bits
  for _ in range(60):
    env.step(np.zeros(2)); fchg += int(np.any(env.swamp_bits != prev)); prev = env.swamp_bits
  env._action_noise = 0.01
  g1 = bool(np.all(np.abs(rates - P) < 0.03))
  g2 = offdiag < 0.08
  g3 = fchg == 0
  return (dict(name='G1_per_cell_activation_0.10', passed=g1,
               metrics=dict(per_cell_rates=[float(r) for r in rates], target=P),
               thresholds='|rate-0.10|<0.03 per cell'),
          dict(name='G2_bits_independent', passed=g2,
               metrics=dict(max_abs_pairwise_corr=offdiag),
               thresholds='max|corr|<0.08'),
          dict(name='G3_freeze_inside_corridor', passed=g3,
               metrics=dict(frozen_changes_inside=int(fchg)),
               thresholds='0 config changes while inside the swamp corridor'))


def g4_g5_g6_g7_teacher(env, teacher_eps):
  n = len(teacher_eps)
  fs = np.array([e['force_safe'] for e in teacher_eps])
  fs_rate = float(fs.mean())
  nonfs = [e for e in teacher_eps if not e['force_safe']]
  # G5: every non-forced-safe episode ends in a shortcut crossing; and it only
  # crossed once all three cells were clear (bits at the commit step == 0).
  waited_ok = all(e['crossed'] for e in nonfs)
  # bits at entry were all-clear for crossings
  entry_clear = []
  for e in nonfs:
    cells = _cells(e['traj'], env._walls)
    idx = next((t for t in range(len(cells) - 1) if cells[t + 1] in SWAMP), None)
    if idx is not None:
      entry_clear.append(int(not e['bits_log'][idx].any()))
  g5 = waited_ok and (np.mean(entry_clear) > 0.99 if entry_clear else False)
  succ = float(np.mean([e['success'] for e in teacher_eps]))
  active_entries = int(np.sum([e['entered_active'] for e in teacher_eps]))
  g4 = abs(fs_rate - FORCE_SAFE_P) < 0.02
  g6 = succ >= 0.99
  g7 = active_entries == 0
  return (dict(name='G4_forced_safe_rate_0.05', passed=bool(g4),
               metrics=dict(forced_safe_rate=fs_rate, target=FORCE_SAFE_P, n=n),
               thresholds='|rate-0.05|<0.02'),
          dict(name='G5_wait_until_clear_then_shortcut', passed=bool(g5),
               metrics=dict(nonfs_all_crossed=bool(waited_ok),
                            entry_all_clear_frac=float(np.mean(entry_clear)) if entry_clear else None,
                            n_nonfs=len(nonfs)),
               thresholds='every non-forced-safe episode crosses, entry config all-clear'),
          dict(name='G6_teacher_success_~1.0', passed=bool(g6),
               metrics=dict(success=succ), thresholds='success>=0.99'),
          dict(name='G7_teacher_active_entries_exactly_0', passed=bool(g7),
               metrics=dict(active_swamp_entries=active_entries),
               thresholds='==0'))


def g8_u_to_action(env, teacher, rng, per_cell=40):
  hold = matched_states([HOLDING], per_cell, rng, lo=0.3, hi=0.7)
  clear_short, active_wait = [], []
  for s in hold:
    a_clear = matched_action(env, teacher, rng, s, [0, 0, 0])
    clear_short.append(float(a_clear[0] > 0.3 and abs(a_clear[1]) < 0.3))  # +x shortcut
    for cfg in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
      a_act = matched_action(env, teacher, rng, s, cfg)
      active_wait.append(float(np.linalg.norm(a_act) < 0.05))              # wait=0
  # prefix U-invariance: in the approach region (before holding) action ignores U
  prefix = matched_states([(0, 3), (1, 3)], per_cell, rng)
  pgap = []
  for s in prefix:
    a0 = matched_action(env, teacher, rng, s, [0, 0, 0], phase='approach')
    for cfg in ([1, 1, 1], [1, 0, 0]):
      pgap.append(float(np.linalg.norm(
          a0 - matched_action(env, teacher, rng, s, cfg, phase='approach'))))
  clear_ok = float(np.mean(clear_short))
  wait_ok = float(np.mean(active_wait))
  prefix_gap = float(np.mean(pgap))
  passed = clear_ok > 0.95 and wait_ok > 0.95 and prefix_gap < 0.05
  return dict(name='G8_U_to_action_matched_holding', passed=bool(passed),
              metrics=dict(clear_takes_shortcut=clear_ok, active_waits=wait_ok,
                           prefix_action_gap=prefix_gap),
              thresholds='clear->+x shortcut>0.95, active->wait>0.95, prefix gap<0.05')


def g9_u_to_next_state(env, rng, per_cell=40, k=3):
  env._action_noise = 0.0
  env.set_auto_resample(False)
  a = np.array([1.0, 0.0])
  def end_x(s, cfg):
    env.set_swamp(cfg); env.state = np.asarray(s, float).copy(); env.goal = env.GOAL.copy()
    for _ in range(k):
      env.step(a)
    return float(env.state[0])
  clear_prog, active_prog = [], []
  for s in matched_states([(3, 3)], per_cell, rng, lo=0.3, hi=0.5):
    x0 = float(s[0])
    clear_prog.append(end_x(s, [0, 0, 0]) - x0)
    active_prog.append(end_x(s, [1, 0, 0]) - x0)
  env._action_noise = 0.01
  env.set_auto_resample(True)
  cp, ap = float(np.mean(clear_prog)), float(np.mean(active_prog))
  passed = cp > 1.0 and ap < 0.15 and (cp - ap) > 1.0
  return dict(name='G9_U_to_next_state_matched_action', passed=bool(passed),
              metrics=dict(clear_progress=cp, active_progress=ap, delta=cp - ap, k=k),
              thresholds='clear dx>1.0, active dx<0.15 (severe slowdown), gap>1.0')


def g10_hiddenness(env, rng, per_cell=25):
  env.set_auto_resample(False)
  states = matched_states([(0, 3), HOLDING] + list(SWAMP), per_cell, rng)
  maxd = 0.0
  for s in states:
    env.goal = env.GOAL.copy()
    obs = []
    for cfg in ALL_CONFIGS:
      env.set_swamp(cfg); env.state = np.asarray(s, float).copy()
      obs.append(env._get_obs().copy())
    obs = np.array(obs)
    maxd = max(maxd, float(np.max(np.abs(obs - obs[0]))))
  env.set_auto_resample(True)
  passed = maxd == 0.0 and env.obs_dim == 2 and int(env._get_obs().shape[0]) == 4
  return dict(name='G10_hiddenness_of_U', passed=bool(passed),
              metrics=dict(max_obs_abs_diff_over_8_configs=maxd, obs_len=int(env._get_obs().shape[0])),
              thresholds='max|obs diff|==0 across all 8 configs; obs=[x,y,gx,gy]')


def g11_obs_vs_intervention(env, teacher_eps, n_iv=400):
  rng = np.random.default_rng(0)
  crossed = [e for e in teacher_eps if e['crossed']]
  obs_succ = float(np.mean([e['success'] for e in crossed])) if crossed else float('nan')
  do_short = follower(env._walls)
  iv = []
  for _ in range(n_iv):
    env.set_auto_resample(True); env.reset()
    g = env.goal.copy(); memo = {}
    traj = [env.state.copy()]
    for _ in range(env.max_episode_steps):
      env.step(np.asarray(do_short(env.state.copy(), g, memo), np.float32))
      traj.append(env.state.copy())
    traj = np.array(traj)
    iv.append(float(np.min(np.linalg.norm(traj - g, axis=1)) < SUCCESS_RADIUS))
  iv_succ = float(np.mean(iv))
  theory = (1 - P) ** 3
  passed = obs_succ >= 0.98 and abs(iv_succ - theory) < 0.06
  return dict(name='G11_observational_vs_do_shortcut', passed=bool(passed),
              metrics=dict(observational_crossing_success=obs_succ,
                           do_shortcut_success=iv_succ, theory_0p9_cubed=theory,
                           gap=obs_succ - iv_succ),
              thresholds='obs>=0.98; do(shortcut)~(0.9)^3=0.729 (|.|<0.06)')


def g12_always_safe(env, n=150):
  safe = follower(swamp_blocked_walls(env._walls))
  res = {}
  for label, fb in (('natural', None), ('all_clear', [0, 0, 0]), ('all_active', [1, 1, 1])):
    ss = []
    for _ in range(n):
      env.set_auto_resample(True); env.reset()
      if fb is not None:
        env.set_auto_resample(False); env.set_swamp(fb)
      g = env.goal.copy(); memo = {}
      traj = [env.state.copy()]
      for _ in range(env.max_episode_steps):
        env.step(np.asarray(safe(env.state.copy(), g, memo), np.float32))
        traj.append(env.state.copy())
      env.set_auto_resample(True)
      traj = np.array(traj)
      ss.append(float(np.min(np.linalg.norm(traj - g, axis=1)) < SUCCESS_RADIUS))
    res[label] = float(np.mean(ss))
  passed = all(v >= 0.99 for v in res.values())
  return dict(name='G12_always_safe_all_conditions', passed=bool(passed),
              metrics=dict(safe_success=res), thresholds='success>=0.99 under natural/all-clear/all-active')


def g13_force_safe_independent(env, teacher_eps):
  # force_safe must be independent of the initial swamp config
  fs = np.array([int(e['force_safe']) for e in teacher_eps])
  any_active = np.array([int(e['initial_bits'].any()) for e in teacher_eps])
  if fs.std() == 0 or any_active.std() == 0:
    corr = 0.0
  else:
    corr = float(np.corrcoef(fs, any_active)[0, 1])
  # force_safe rate conditional on initial-active vs initial-clear should match
  rate_active = float(fs[any_active == 1].mean()) if (any_active == 1).any() else float('nan')
  rate_clear = float(fs[any_active == 0].mean()) if (any_active == 0).any() else float('nan')
  passed = abs(corr) < 0.08 and abs(rate_active - rate_clear) < 0.05
  return dict(name='G13_force_safe_independent_of_U', passed=bool(passed),
              metrics=dict(corr_force_safe_initial_active=corr,
                           force_safe_rate_given_active=rate_active,
                           force_safe_rate_given_clear=rate_clear),
              thresholds='|corr|<0.08 and |rate_active-rate_clear|<0.05')


# --------------------------------------------------------------------------- #
# plots                                                                         #
# --------------------------------------------------------------------------- #
def _draw(ax, env, active=None):
  walls = env._walls
  H, W = walls.shape
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55, extent=[0, H, 0, W])
  for k, c in enumerate(SWAMP):
    if active is not None and active[k]:
      ax.add_patch(Rectangle(c, 1, 1, facecolor='tab:red', alpha=0.3))
    ax.add_patch(Rectangle(c, 1, 1, fill=False, hatch='///', edgecolor='darkorange', lw=1.3))
  ax.add_patch(Rectangle(HOLDING, 1, 1, fill=False, edgecolor='tab:blue', lw=1.5, ls='--'))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')


def plot_schematic(env, out):
  from scripts.qualify_two_route_swamp import bfs_path  # noqa
  fig, ax = plt.subplots(figsize=(6.6, 3.8))
  _draw(ax, env)
  ax.scatter(*env.START, c='black', s=55, zorder=5); ax.text(*(env.START + [0, .3]), 'S')
  ax.scatter(*env.GOAL, c='red', marker='*', s=150, zorder=5); ax.text(*(env.GOAL + [-.1, .3]), 'G')
  ax.scatter(*HOLD_CENTER, c='tab:blue', s=40, zorder=5)
  ax.set_title('point_two_route_swamp_matched_v0  (p=0.10, MiniGrid-matched)', fontsize=10)
  fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_teacher_panels(env, out):
  rng = np.random.default_rng(1)
  teacher = make_matched_teacher(env, rng)
  panels = []
  # find one example of each requested behavior
  want = {'forced_safe': None, 'immediate_shortcut': None, 'wait1': None, 'waitmulti': None}
  for _ in range(4000):
    e = rollout_matched(env, teacher, rng)
    m = e['teacher_mode']
    if m == 'forced_safe' and want['forced_safe'] is None:
      want['forced_safe'] = e
    elif m == 'immediate_shortcut' and want['immediate_shortcut'] is None:
      want['immediate_shortcut'] = e
    elif m == 'wait_shortcut' and e['wait_count'] == 1 and want['wait1'] is None:
      want['wait1'] = e
    elif m == 'wait_shortcut' and e['wait_count'] >= 3 and want['waitmulti'] is None:
      want['waitmulti'] = e
    if all(v is not None for v in want.values()):
      break
  # do(shortcut) under active swamp (probe -- NOT teacher behavior): trapped
  do_short = follower(env._walls)
  env.set_auto_resample(False); env.reset(); env.set_swamp([1, 1, 1])
  g = env.goal.copy(); memo = {}; tj = [env.state.copy()]
  for _ in range(env.max_episode_steps):
    env.step(np.asarray(do_short(env.state.copy(), g, memo), np.float32)); tj.append(env.state.copy())
  env.set_auto_resample(True)
  forced = dict(traj=np.array(tj), success=False, active=[1, 1, 1],
                title='do(shortcut) | active swamp\n(trapped -- NOT teacher)')
  order = [('forced_safe', 'forced-safe route'),
           ('immediate_shortcut', 'clear shortcut (wait=0)'),
           ('wait1', 'one-step wait -> shortcut'),
           ('waitmulti', 'multi-step wait -> shortcut')]
  fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
  for ax, (key, title) in zip(axes[:4], order):
    e = want[key]
    _draw(ax, env, active=e['bits_log'][0] if e is not None else None)
    if e is not None:
      t = e['traj']; col = 'tab:green' if e['success'] else 'tab:red'
      ax.plot(t[:, 0], t[:, 1], '-', lw=1.3, color=col)
      title += f"\nwait={e['wait_count']} succ={int(e['success'])}"
    ax.scatter(*env.START, c='black', s=25); ax.scatter(*env.GOAL, c='red', marker='*', s=90)
    ax.set_title(title, fontsize=8.5)
  _draw(axes[4], env, active=forced['active'])
  axes[4].plot(forced['traj'][:, 0], forced['traj'][:, 1], '-', lw=1.3, color='tab:red')
  axes[4].scatter(*env.START, c='black', s=25); axes[4].scatter(*env.GOAL, c='red', marker='*', s=90)
  axes[4].set_title(forced['title'], fontsize=8.5)
  fig.suptitle('MiniGrid-matched teacher behaviors (green=goal, red=fail/trapped)', fontsize=11)
  fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=105); plt.close(fig)


def plot_confounding(env, report, out):
  g = {x['name'].split('_')[0]: x['metrics'] for x in report['gates']}
  fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
  ax[0].bar(['c0', 'c1', 'c2'], g['G1']['per_cell_rates'], color='darkorange')
  ax[0].axhline(P, color='k', ls='--', lw=1); ax[0].set_ylim(0, 0.2)
  ax[0].set_title('G1 per-cell activation (target 0.10)', fontsize=9)
  tf = report['teacher_frequencies']
  ax[1].bar(list(tf.keys()), list(tf.values()), color='tab:blue')
  ax[1].tick_params(axis='x', labelsize=7, rotation=20)
  ax[1].set_title('teacher mode frequencies', fontsize=9)
  g11 = g['G11']
  ax[2].bar(['obs cross', 'do(shortcut)'],
            [g11['observational_crossing_success'], g11['do_shortcut_success']],
            color=['tab:blue', 'tab:red'])
  ax[2].axhline(g11['theory_0p9_cubed'], color='k', ls='--', lw=1)
  ax[2].set_ylim(0, 1.05); ax[2].set_title('G11 obs vs do(shortcut)~0.729', fontsize=9)
  for a in ax:
    a.grid(alpha=0.3, axis='y')
  fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='artifacts/swamp_matched_qual')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n_teacher', type=int, default=2000)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)
  teacher = make_matched_teacher(env, rng)

  print('running matched-teacher batch (%d)...' % args.n_teacher)
  teacher_eps = [rollout_matched(env, teacher, rng) for _ in range(args.n_teacher)]

  g1, g2, g3 = g1_g2_g3_stats(env)
  g4, g5, g6, g7 = g4_g5_g6_g7_teacher(env, teacher_eps)
  gates = [g1, g2, g3, g4, g5, g6, g7,
           g8_u_to_action(env, teacher, rng), g9_u_to_next_state(env, rng),
           g10_hiddenness(env, rng), g11_obs_vs_intervention(env, teacher_eps),
           g12_always_safe(env), g13_force_safe_independent(env, teacher_eps)]
  all_pass = all(g['passed'] for g in gates)

  # teacher frequency table (empirical vs theoretical)
  modes = [e['teacher_mode'] for e in teacher_eps]
  n = len(modes)
  tf = {k: modes.count(k) / n for k in ('forced_safe', 'immediate_shortcut', 'wait_shortcut')}
  pc = (1 - P) ** 3
  theory = dict(forced_safe=FORCE_SAFE_P,
                immediate_shortcut=(1 - FORCE_SAFE_P) * pc,
                wait_shortcut=(1 - FORCE_SAFE_P) * (1 - pc))
  freq_table = {k: dict(empirical=round(tf[k], 4), theory=round(theory[k], 4))
                for k in tf}

  report = dict(
      env=ENV, setting='minigrid_matched', seed=args.seed,
      params=dict(per_cell_swamp_prob=P, force_safe_prob=FORCE_SAFE_P,
                  slow_factor=env.slow_factor, success_radius=SUCCESS_RADIUS,
                  horizon=env.max_episode_steps),
      teacher_frequencies=tf, frequency_table=freq_table,
      note=('force_safe (5%) is behavior randomness INDEPENDENT of U; it is NOT '
            'the Manski propensity. Propensity is local/single-step at the '
            'holding/fork and is estimated in a later task, not here.'),
      gates=gates, all_gates_passed=bool(all_pass),
      verdict='QUALIFIED' if all_pass else 'NOT_QUALIFIED')
  json.dump(report, open(os.path.join(args.out, 'qualification_report.json'), 'w'), indent=2)
  plot_schematic(env, os.path.join(args.out, 'map_schematic.png'))
  plot_teacher_panels(env, os.path.join(args.out, 'teacher_panels.png'))
  plot_confounding(env, report, os.path.join(args.out, 'confounding_summary.png'))

  print('=' * 74)
  print('MINIGRID-MATCHED QUALIFICATION  %s  (seed %d)' % (ENV, args.seed))
  print('=' * 74)
  for g in gates:
    print('  [%s]  %s' % ('PASS' if g['passed'] else 'FAIL', g['name']))
  print('-' * 74)
  print('teacher freq (emp/theory): ' + '  '.join(
      f'{k}={v["empirical"]}/{v["theory"]}' for k, v in freq_table.items()))
  print('VERDICT:', report['verdict'], '(%d/%d)' % (sum(g['passed'] for g in gates), len(gates)))
  print('saved report + 3 plots under', args.out)
  sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
