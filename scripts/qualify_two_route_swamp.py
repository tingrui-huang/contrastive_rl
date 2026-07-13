"""Confounder-qualification harness for ``point_two_route_swamp_v0``.

Re-runs causal qualification FROM SCRATCH for the dynamic three-cell swamp
prototype (which replaces the static hidden-gate v0). Nothing here trains a
learner or touches the CRL / causal objectives -- it only certifies the
confounding structure, with a machine-readable verdict.

Hidden state U_t = 3 swamp bits on the short route, each i.i.d. active with
p=0.2, resampled every step while the agent is outside the swamp corridor and
FROZEN while inside. Active cells slow motion by 50x (trap: recoverable, not a
wall, no termination). Teacher decision at the pre-swamp HOLDING cell:
all clear -> shortcut; any active -> wait 1 step, re-check: clear -> shortcut,
else always-safe lower route. Learner sees XY + goal XY only.

Gates:
  G1 map/dynamics correctness    (walls ok; swamp enterable/not-wall; trap costly; recoverable)
  G2 swamp statistics + freeze   (per-cell rate ~0.2, independent, frozen inside corridor)
  G3 teacher behavior            (shortcut/wait/detour frequencies match theory; never slowed)
  G4 U -> A on matched states    (bits flip the holding-cell action; prefix unaffected)
  G5 U -> S' matched actions     (same state+action, different bits -> different S')
  G6 hiddenness of U             (obs identical across all 8 configs; no wait counter)
  G7 obs vs interventional gap   (P(succ|teacher crossed) ~1 vs do(shortcut) ~(1-p)^3)
  G8 always-safe invariance      (lower route succeeds under every config, never enters swamp)

Run:  python scripts/qualify_two_route_swamp.py
      python scripts/qualify_two_route_swamp.py --out artifacts/point_two_route_swamp_v0 --seed 0
Exit code 0 iff every gate passes (STOP before any learner training).
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

from crl.envs import TwoRouteSwampEnv
from crl.report_maze import bfs_path, bfs_waypoints

SUCCESS_RADIUS = 0.5
SWAMP = TwoRouteSwampEnv.SWAMP_CELLS
HOLDING = TwoRouteSwampEnv.HOLDING_CELL
FORK = TwoRouteSwampEnv.FORK_CELL
START_CELL = TwoRouteSwampEnv.START_CELL
GOAL_CELL = TwoRouteSwampEnv.GOAL_CELL
POST_CELL = (6, 3)                     # first top-row cell past the swamp
HOLD_CENTER = np.array([HOLDING[0] + 0.5, HOLDING[1] + 0.5])
ACTIVE_CONFIGS = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1])


def swamp_blocked_walls(walls):
  """Wall grid with the swamp cells marked blocked (for safe-route BFS)."""
  w = walls.copy()
  for c in SWAMP:
    w[c] = 1
  return w


# --------------------------------------------------------------------------- #
# policies                                                                     #
# --------------------------------------------------------------------------- #
def _follow(walls, s, g, memo):
  """BFS-waypoint follower step (waypoints computed once at commit time)."""
  if 'wps' not in memo:
    wps = bfs_waypoints(walls, s, g)
    memo['wps'] = wps if wps else [np.asarray(g, float)]
    memo['i'] = 1 if len(memo['wps']) > 1 else 0
  wps, i = memo['wps'], memo['i']
  while i < len(wps) - 1 and np.linalg.norm(wps[i] - s) < 0.5:
    i += 1
  memo['i'] = i
  return np.clip(wps[i] - s, -1, 1).astype(np.float32)


def follower(walls):
  """Stateless-U route follower (used for do(shortcut) + always-safe)."""
  return lambda s, g, memo: _follow(walls, s, g, memo)


def make_teacher(env):
  """Swamp-aware teacher. Observes env.swamp_bits (full 3-bit access).
  FSM: approach the HOLDING cell; there: all clear -> shortcut; any active ->
  wait one step; after the resample: clear -> shortcut else lower route."""
  base = env._walls
  blocked = swamp_blocked_walls(base)

  def policy(s, g, memo):
    ph = memo.setdefault('phase', 'approach')
    if ph == 'approach':
      if np.linalg.norm(s - HOLD_CENTER) < 0.35:
        memo['phase'] = 'decide'
      else:
        return np.clip(HOLD_CENTER - s, -1, 1).astype(np.float32)
    if memo['phase'] == 'decide':
      bits = env.swamp_bits
      memo.setdefault('decision_bits', []).append(bits.copy())
      if not bits.any():
        memo['phase'] = 'short'
      elif not memo.get('waited', False):
        memo['waited'] = True
        memo['n_waits'] = memo.get('n_waits', 0) + 1
        return np.zeros(2, np.float32)          # WAIT: hold position one step
      else:
        memo['phase'] = 'long'
    return _follow(base if memo['phase'] == 'short' else blocked, s, g, memo)
  return policy


def make_probe_retreat(env):
  """Scripted recoverability probe: step INTO the swamp, crawl back out, then
  take the safe lower route (no bit access -- pure physical recoverability)."""
  blocked = swamp_blocked_walls(env._walls)

  def policy(s, g, memo):
    ph = memo.setdefault('phase', 'approach')
    if ph == 'approach':
      if np.linalg.norm(s - HOLD_CENTER) < 0.35:
        memo['phase'] = 'probe'
      else:
        return np.clip(HOLD_CENTER - s, -1, 1).astype(np.float32)
    if memo['phase'] == 'probe':
      memo['phase'] = 'retreat'
      return np.array([1.0, 0.0], np.float32)   # one full step into the swamp
    if memo['phase'] == 'retreat':
      if s[0] >= 2.6:
        return np.array([-1.0, 0.0], np.float32)
      memo['phase'] = 'long'
      memo['recovered_at'] = memo.get('t', None)
    return _follow(blocked, s, g, memo)
  return policy


# --------------------------------------------------------------------------- #
# rollout                                                                      #
# --------------------------------------------------------------------------- #
def rollout(env, policy, force_bits=None):
  """One episode. force_bits freezes a configuration (resampling disabled)."""
  env.set_auto_resample(True)
  env.reset()
  if force_bits is not None:
    env.set_auto_resample(False)
    env.set_swamp(force_bits)
  g = env.goal.copy()
  traj = [env.state.copy()]
  bits_log = []
  dists = [float(np.linalg.norm(env.state - g))]
  memo = {}
  for t in range(env.max_episode_steps):
    memo['t'] = t
    bits_log.append(env.swamp_bits)           # bits governing THIS step
    a = policy(env.state.copy(), g, memo)
    env.step(a)
    traj.append(env.state.copy())
    dists.append(float(np.linalg.norm(env.state - g)))
  env.set_auto_resample(True)
  traj = np.array(traj)
  bits_log = np.array(bits_log)
  cells = [tuple(np.clip(np.floor(p).astype(int), [0, 0],
                         np.array(env._walls.shape) - 1)) for p in traj]
  entered = any(c in SWAMP for c in cells)
  crossed = entered and any(c == POST_CELL for c in cells)
  used_safe = bool(np.any(traj[:, 1] < 2.0))
  # steps spent inside an ACTIVE swamp cell (bits governing that step)
  slowed = sum(1 for t in range(len(bits_log))
               if cells[t + 1] in SWAMP and bits_log[t][SWAMP.index(cells[t + 1])])
  min_dist = float(min(dists))
  steps_succ = next((t for t, d in enumerate(dists) if d < SUCCESS_RADIUS), None)
  return dict(traj=traj, bits_log=bits_log, memo=memo, dists=dists,
              min_dist=min_dist, success=bool(min_dist < SUCCESS_RADIUS),
              entered_swamp=entered, crossed=crossed, used_safe=used_safe,
              slowed_steps=int(slowed), steps_to_success=steps_succ,
              n_waits=int(memo.get('n_waits', 0)))


def classify_teacher(ep):
  if ep['crossed'] and ep['n_waits'] == 0:
    return 'immediate_shortcut'
  if ep['crossed'] and ep['n_waits'] == 1:
    return 'wait_shortcut'
  if ep['used_safe'] and ep['n_waits'] == 1:
    return 'wait_detour'
  return 'other'


def matched_states(cells, per_cell, rng, lo=0.15, hi=0.85):
  pts = []
  for (i, j) in cells:
    for _ in range(per_cell):
      pts.append(np.array([i, j], float) + rng.uniform(lo, hi, size=2))
  return pts


def fresh_action(env, teacher, s, bits, waited=False):
  """Teacher action at state s under forced bits (fresh memo; no history)."""
  env.set_auto_resample(False)
  env.set_swamp(bits)
  memo = {'waited': True} if waited else {}
  a = teacher(np.asarray(s, float).copy(), env.GOAL.copy(), memo)
  env.set_auto_resample(True)
  return np.asarray(a, float)


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #
def gate1_map_dynamics(env, rng):
  """Walls ok under all configs; swamp cells are traps, not walls; trap is
  costly (cannot cross within horizon) yet recoverable (back out + detour)."""
  walls = env._walls
  env._action_noise = 0.0
  mism = 0
  for cfg in ([0, 0, 0], [1, 1, 1]):        # blocking must be config-INDEPENDENT
    env.set_auto_resample(False)
    env.set_swamp(cfg)
    for i in range(env._height):
      for j in range(env._width):
        for _ in range(4):
          p = np.array([i, j], float) + rng.uniform(0.1, 0.9, size=2)
          if env._is_blocked(p) != bool(walls[i, j] == 1):
            mism += 1
  for p in ([-0.1, 3.5], [9.1, 3.5], [3.5, -0.1], [3.5, 5.1]):
    if not env._is_blocked(np.array(p, float)):
      mism += 1
  # active swamp is enterable (NOT a wall): position advances past the boundary
  env.set_swamp([1, 1, 1])
  env.state = np.array([2.95, 3.5]); env.goal = env.GOAL.copy()
  env.step(np.array([1.0, 0.0]))
  enter_x = float(env.state[0])
  enterable = enter_x > 3.005
  # costly: pushing through ONE active cell does not cross it within 45 steps
  env.set_swamp([1, 0, 0])
  env.state = np.array([2.5, 3.5])
  for _ in range(45):
    env.step(np.array([1.0, 0.0]))
  x_active = float(env.state[0])
  # clear corridor crosses fast
  env.set_swamp([0, 0, 0])
  env.state = np.array([2.5, 3.5])
  for _ in range(6):
    env.step(np.array([1.0, 0.0]))
  x_clear = float(env.state[0])
  env._action_noise = 0.01
  env.set_auto_resample(True)
  # recoverable: probe-in / crawl-out / detour succeeds under ALL-ACTIVE
  probe_eps = [rollout(env, make_probe_retreat(env), force_bits=[1, 1, 1])
               for _ in range(20)]
  probe_succ = float(np.mean([e['success'] for e in probe_eps]))
  probe_steps = float(np.mean([e['steps_to_success'] for e in probe_eps
                               if e['steps_to_success'] is not None]))
  # two routes: upper (through swamp cells) strictly shorter than lower
  po = bfs_path(walls, START_CELL, GOAL_CELL)
  pc = bfs_path(swamp_blocked_walls(walls), START_CELL, GOAL_CELL)
  routes_ok = (po is not None and pc is not None and len(pc) > len(po)
               and all(c in po for c in SWAMP) and any(c[1] < 2 for c in pc)
               and not any(c in SWAMP for c in pc))
  passed = (mism == 0 and enterable and x_active < 4.0 and x_clear > 6.0
            and probe_succ >= 0.99 and routes_ok)
  return dict(
      name='G1_map_dynamics_correctness', passed=bool(passed),
      metrics=dict(cell_mismatches=int(mism), active_cell_enterable=bool(enterable),
                   enter_x=enter_x, x_after_45_push_one_active=x_active,
                   x_after_6_push_clear=x_clear,
                   probe_retreat_detour_success=probe_succ,
                   probe_mean_steps_to_goal=probe_steps,
                   len_upper_cells=len(po), len_lower_cells=len(pc)),
      thresholds=('mismatches==0; active cell enterable (not wall); 45-step push '
                  'stays <4.0 (costly); clear cross >6.0 in 6 steps; probe-retreat-'
                  'detour succ>=0.99 under all-active (recoverable); upper<lower'),
      detail='swamp alters DYNAMICS only: never blocks, never terminates')


def gate2_swamp_statistics(env):
  """Per-cell activation ~0.2, independent, i.i.d. outside; FROZEN inside."""
  env._action_noise = 0.0
  env.set_auto_resample(True)
  env.reset()                                  # agent at start (outside corridor)
  logs, changes, prev = [], 0, env.swamp_bits
  n = 4000
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
  change_rate = changes / n
  # freeze: agent parked INSIDE the corridor -> bits must never change
  env.set_swamp([0, 1, 0])
  env.state = np.array([3.5, 3.5])             # inside swamp cell 0 (clear)
  frozen_changes, prev = 0, env.swamp_bits
  for _ in range(60):
    env.step(np.zeros(2))
    b = env.swamp_bits
    frozen_changes += int(np.any(b != prev))
    prev = b
  env._action_noise = 0.01
  p = env.active_prob
  th_change = 1.0 - (p * p + (1 - p) * (1 - p)) ** 3
  passed = (np.all(np.abs(rates - p) < 0.03) and offdiag < 0.08
            and abs(change_rate - th_change) < 0.05 and frozen_changes == 0)
  return dict(
      name='G2_swamp_statistics_and_freeze', passed=bool(passed),
      metrics=dict(per_cell_rates=[float(r) for r in rates],
                   target_rate=p, max_abs_pairwise_corr=offdiag,
                   outside_config_change_rate=float(change_rate),
                   theory_change_rate=float(th_change),
                   frozen_changes_inside_corridor=int(frozen_changes),
                   n_samples=n),
      thresholds='|rate-0.2|<0.03 per cell; |corr|<0.08; change-rate ~ theory; 0 changes inside',
      detail='U_t i.i.d. Bernoulli(0.2)^3 while outside; frozen while inside the corridor')


def gate3_teacher_behavior(env, teacher_eps):
  """Shortcut/wait/detour frequencies match theory; teacher never slowed."""
  n = len(teacher_eps)
  routes = [classify_teacher(e) for e in teacher_eps]
  freq = {r: routes.count(r) / n for r in
          ('immediate_shortcut', 'wait_shortcut', 'wait_detour', 'other')}
  p_clear = (1 - env.active_prob) ** 3
  theory = dict(immediate_shortcut=p_clear,
                wait_shortcut=(1 - p_clear) * p_clear,
                wait_detour=(1 - p_clear) ** 2, other=0.0)
  success = float(np.mean([e['success'] for e in teacher_eps]))
  slowed = int(sum(e['slowed_steps'] for e in teacher_eps))
  wait_freq = float(np.mean([e['n_waits'] > 0 for e in teacher_eps]))
  # bits at the FIRST decision are a fresh sample -> per-cell rate ~0.2
  first_bits = np.array([e['memo']['decision_bits'][0] for e in teacher_eps], float)
  dec_rates = first_bits.mean(0)
  match = all(abs(freq[k] - theory[k]) < 0.05 for k in theory)
  passed = (match and success >= 0.995 and slowed == 0 and freq['other'] == 0.0
            and np.all(np.abs(dec_rates - env.active_prob) < 0.04))
  return dict(
      name='G3_teacher_behavior_frequencies', passed=bool(passed),
      metrics=dict(n_episodes=n, frequencies=freq, theory=theory,
                   wait_frequency=wait_freq, theory_wait=1 - p_clear,
                   success=success, slowed_steps_total=slowed,
                   decision_bit_rates=[float(r) for r in dec_rates]),
      thresholds='|freq-theory|<0.05 each; success>=0.995; slowed==0; other==0',
      detail='teacher: clear->shortcut; active->wait 1; then clear->shortcut else detour')


def gate4_u_to_action(env, teacher, rng, per_cell=30):
  """Matched holding states: bits flip the teacher action; prefix unaffected."""
  hold_states = matched_states([HOLDING], per_cell, rng, lo=0.3, hi=0.7)
  gaps_wait, gaps_detour = [], []
  for s in hold_states:
    a_clear = fresh_action(env, teacher, s, [0, 0, 0])
    for cfg in ACTIVE_CONFIGS:
      gaps_wait.append(float(np.linalg.norm(
          a_clear - fresh_action(env, teacher, s, cfg))))
      gaps_detour.append(float(np.linalg.norm(
          a_clear - fresh_action(env, teacher, s, cfg, waited=True))))
  prefix_states = matched_states([START_CELL, FORK], per_cell, rng)
  gaps_prefix = []
  for s in prefix_states:
    a0 = fresh_action(env, teacher, s, [0, 0, 0])
    for cfg in ACTIVE_CONFIGS:
      gaps_prefix.append(float(np.linalg.norm(
          a0 - fresh_action(env, teacher, s, cfg))))
  m = dict(holding_gap_clear_vs_wait=float(np.mean(gaps_wait)),
           holding_gap_clear_vs_detour=float(np.mean(gaps_detour)),
           prefix_gap=float(np.mean(gaps_prefix)),
           n_holding_states=len(hold_states), n_prefix_states=len(prefix_states))
  passed = (m['holding_gap_clear_vs_wait'] > 0.5
            and m['holding_gap_clear_vs_detour'] > 0.5
            and m['prefix_gap'] < 0.05)
  return dict(
      name='G4_U_to_action_matched_states', passed=bool(passed),
      metrics=m,
      thresholds='holding gaps > 0.5 (U changes A) AND prefix gap < 0.05',
      detail='identical XY at the holding cell; only the hidden bits differ')


def gate5_u_to_next_state(env, rng, per_cell=30, k=3):
  """Same state + same action, different bits -> different S' (clones)."""
  env._action_noise = 0.0
  env.set_auto_resample(False)
  action = np.array([1.0, 0.0])

  def end_state(s, cfg):
    env.set_swamp(cfg)
    env.state = np.asarray(s, float).copy()
    env.goal = env.GOAL.copy()
    for _ in range(k):
      env.step(action)
    return env.state.copy()

  deltas = {tuple(c): [] for c in ACTIVE_CONFIGS}
  for s in matched_states([HOLDING], per_cell, rng, lo=0.3, hi=0.7):
    e_clear = end_state(s, [0, 0, 0])
    for cfg in ACTIVE_CONFIGS:
      deltas[tuple(cfg)].append(float(np.linalg.norm(end_state(s, cfg) - e_clear)))
  per_cfg = {str(list(c)): float(np.mean(v)) for c, v in deltas.items()}
  control = []
  for s in matched_states([(5, 1)], per_cell, rng):     # lower corridor: swamp-free
    e_clear = end_state(s, [0, 0, 0])
    control.append(float(np.linalg.norm(end_state(s, [1, 1, 1]) - e_clear)))
  env._action_noise = 0.01
  env.set_auto_resample(True)
  mean_delta = float(np.mean(list(per_cfg.values())))
  min_delta = float(min(per_cfg.values()))
  ctrl = float(np.mean(control))
  passed = mean_delta > 0.5 and min_delta > 0.2 and ctrl < 0.05
  return dict(
      name='G5_U_to_next_state_matched_actions', passed=bool(passed),
      metrics=dict(delta_by_config=per_cfg, mean_delta=mean_delta,
                   min_delta=min_delta, control_delta=ctrl, k_steps=k),
      thresholds="mean dS'>0.5, min over configs >0.2; lower-route control <0.05",
      detail='pure U->S\' effect: clone rollouts differ only in the swamp bits')


def gate6_hiddenness(env, rng, per_cell=20):
  """Observation identical across all 8 configs; no wait counter in the obs."""
  env.set_auto_resample(False)
  states = matched_states([START_CELL, HOLDING] + list(SWAMP), per_cell, rng)
  configs = [[int(b) for b in np.binary_repr(m, 3)] for m in range(8)]
  max_diff = 0.0
  for s in states:
    env.goal = env.GOAL.copy()
    obs = []
    for cfg in configs:
      env.set_swamp(cfg)
      env.state = np.asarray(s, float).copy()
      obs.append(env._get_obs().copy())
    obs = np.array(obs)
    max_diff = max(max_diff, float(np.max(np.abs(obs - obs[0]))))
  env.set_auto_resample(True)
  obs_len = int(env._get_obs().shape[0])
  passed = (max_diff == 0.0 and env.obs_dim == 2 and obs_len == 4)
  return dict(
      name='G6_hiddenness_of_U', passed=bool(passed),
      metrics=dict(max_obs_abs_diff_across_8_configs=max_diff,
                   n_states=len(states), obs_dim=int(env.obs_dim),
                   obs_len=obs_len, bayes_optimal_U_accuracy_from_one_obs=0.5),
      thresholds='max |obs(cfg_i) - obs(cfg_j)| == 0; obs = [x,y,gx,gy] only',
      detail='no swamp bits, no wait counter, no time index in the observation')


def gate7_obs_vs_intervention(env, teacher_eps, n_iv=500):
  """P(success | teacher crossed) vs do(shortcut) success on natural U."""
  crossed = [e for e in teacher_eps if e['crossed']]
  obs_success = float(np.mean([e['success'] for e in crossed]))
  # corr between "all-clear at final decision" and "crossed" (teacher couples U->route)
  clear_final = np.array([not np.any(e['memo']['decision_bits'][-1])
                          for e in teacher_eps], float)
  crossed_v = np.array([e['crossed'] for e in teacher_eps], float)
  corr = float(np.corrcoef(clear_final, crossed_v)[0, 1])
  # interventional: force the shortcut policy regardless of U
  do_short = follower(env._walls)
  iv = [rollout(env, do_short) for _ in range(n_iv)]
  iv_success = float(np.mean([e['success'] for e in iv]))
  iv_crossed = float(np.mean([e['crossed'] for e in iv]))
  p_clear = (1 - env.active_prob) ** 3
  gap = obs_success - iv_success
  passed = (obs_success >= 0.99 and iv_success <= p_clear + 0.08
            and gap >= 0.25 and corr >= 0.9)
  return dict(
      name='G7_observational_vs_interventional_gap', passed=bool(passed),
      metrics=dict(observational_crossing_success=obs_success,
                   n_observational_crossings=len(crossed),
                   interventional_shortcut_success=iv_success,
                   interventional_crossing_rate=iv_crossed,
                   theory_interventional=p_clear, confounding_gap=gap,
                   corr_clearU_crossed=corr, n_interventional=n_iv),
      thresholds='obs>=0.99; do(shortcut)<=theory+0.08; gap>=0.25; corr(U,cross)>=0.9',
      detail='crossing looks safe in teacher data only because it happens iff U=clear')


def gate8_always_safe(env, n=150):
  """Lower route succeeds under every config and never touches the swamp."""
  safe = follower(swamp_blocked_walls(env._walls))
  res, entries = {}, 0
  for label, fb in (('natural', None), ('all_active', [1, 1, 1]),
                    ('all_clear', [0, 0, 0])):
    eps = [rollout(env, safe, force_bits=fb) for _ in range(n)]
    res[label] = float(np.mean([e['success'] for e in eps]))
    entries += sum(e['entered_swamp'] for e in eps)
  invariant = abs(res['all_active'] - res['all_clear']) < 0.02
  passed = (all(v >= 0.99 for v in res.values()) and invariant and entries == 0)
  return dict(
      name='G8_always_safe_invariance', passed=bool(passed),
      metrics=dict(safe_success=res, invariant_across_U=bool(invariant),
                   swamp_entries=int(entries), n_per_condition=n),
      thresholds='success >= 0.99 under natural/all-active/all-clear; 0 swamp entries',
      detail='a U-independent policy solves the task; the swamp is fully avoidable')


# --------------------------------------------------------------------------- #
# plots                                                                        #
# --------------------------------------------------------------------------- #
def _draw_base(ax, env, active_cfg=None):
  walls = env._walls
  H, W = walls.shape
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55, extent=[0, H, 0, W])
  for k, c in enumerate(SWAMP):
    if active_cfg is not None and active_cfg[k]:
      ax.add_patch(Rectangle(c, 1, 1, facecolor='tab:red', alpha=0.3))
    ax.add_patch(Rectangle(c, 1, 1, fill=False, hatch='///',
                           edgecolor='darkorange', lw=1.4))
  ax.add_patch(Rectangle(HOLDING, 1, 1, fill=False, edgecolor='tab:blue',
                         lw=1.6, linestyle='--'))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
  ax.set_xlim(0, H); ax.set_ylim(0, W)


def plot_schematic(env, out):
  fig, ax = plt.subplots(figsize=(6.6, 3.8))
  _draw_base(ax, env)
  po = np.array([np.array(c) + 0.5 for c in
                 bfs_path(env._walls, START_CELL, GOAL_CELL)])
  pc = np.array([np.array(c) + 0.5 for c in
                 bfs_path(swamp_blocked_walls(env._walls), START_CELL, GOAL_CELL)])
  ax.plot(po[:, 0], po[:, 1], '-', color='tab:blue', lw=2.4,
          label='UPPER / short (3 swamp cells, each active w.p. 0.2)')
  ax.plot(pc[:, 0], pc[:, 1], '--', color='tab:green', lw=2.2,
          label='LOWER / long (safe, always clear)')
  ax.scatter(*env.START, c='black', s=60, zorder=5); ax.text(*(env.START + [0, .3]), 'S')
  ax.scatter(*env.GOAL, c='red', marker='*', s=160, zorder=5)
  ax.text(*(env.GOAL + [-.1, .3]), 'G')
  ax.scatter(*HOLD_CENTER, c='tab:blue', s=45, zorder=5, label='holding cell (decide/wait)')
  ax.scatter(FORK[0] + .5, FORK[1] + .5, c='purple', s=45, zorder=5, label='fork')
  ax.set_title('point_two_route_swamp_v0 — hidden dynamic swamp U_t (3 bits, '
               'frozen inside corridor)', fontsize=9.5)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7.6,
            frameon=False)
  fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_trajectories(env, teacher_eps, out):
  teacher = make_teacher(env)
  by_route = {}
  for e in teacher_eps:
    by_route.setdefault(classify_teacher(e), []).append(e)
  do_short = follower(env._walls)
  panels = []
  for route, title in (('immediate_shortcut', 'teacher: clear at look 1\n(immediate shortcut)'),
                       ('wait_shortcut', 'teacher: wait 1, then clear\n(delayed shortcut)'),
                       ('wait_detour', 'teacher: active twice\n(safe detour)')):
    panels.append((title, by_route.get(route, [])[:8], None))
  iv_eps = [rollout(env, do_short) for _ in range(30)]
  panels.append(('do(shortcut), natural U\n(succeeds iff clear at entry)',
                 iv_eps[:12], None))
  trap_eps = [rollout(env, do_short, force_bits=[1, 1, 1]) for _ in range(8)]
  panels.append(('do(shortcut), all-active\n(trapped in first swamp cell)',
                 trap_eps, [1, 1, 1]))
  rec_eps = [rollout(env, make_probe_retreat(env), force_bits=[1, 1, 1])
             for _ in range(8)]
  panels.append(('probe-retreat-detour, all-active\n(trap is recoverable)',
                 rec_eps, [1, 1, 1]))
  fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
  for ax, (title, eps, cfg) in zip(axes.ravel(), panels):
    _draw_base(ax, env, active_cfg=cfg)
    for e in eps:
      t = e['traj']
      col = 'tab:green' if e['success'] else 'tab:red'
      ax.plot(t[:, 0], t[:, 1], '-', lw=1.0, alpha=0.65, color=col)
      ax.scatter(t[-1, 0], t[-1, 1], c=col, s=12, zorder=4)
    ax.scatter(*env.START, c='black', s=30, zorder=5)
    ax.scatter(*env.GOAL, c='red', marker='*', s=100, zorder=5)
    sr = float(np.mean([e['success'] for e in eps])) if eps else float('nan')
    ax.set_title('%s  succ=%.2f' % (title, sr), fontsize=8.4)
  fig.suptitle('Trajectories (green=reached goal, red=failed; red fill=active swamp)',
               fontsize=10.5)
  fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=105); plt.close(fig)


def plot_confounding(env, report, out):
  g = {x['name'].split('_')[0]: x['metrics'] for x in report['gates']}
  fig, ax = plt.subplots(2, 2, figsize=(10.5, 7))
  # (a) per-cell activation rates
  a = ax[0, 0]
  a.bar(['cell 0\n(3,3)', 'cell 1\n(4,3)', 'cell 2\n(5,3)'], g['G2']['per_cell_rates'],
        color='darkorange')
  a.axhline(env.active_prob, color='k', ls='--', lw=1, label='target 0.2')
  a.set_ylim(0, 0.35); a.legend(fontsize=8)
  a.set_title('G2  empirical per-cell activation rate', fontsize=9)
  # (b) teacher route frequencies vs theory
  a = ax[0, 1]
  keys = ['immediate_shortcut', 'wait_shortcut', 'wait_detour']
  emp = [g['G3']['frequencies'][k] for k in keys]
  th = [g['G3']['theory'][k] for k in keys]
  x = np.arange(3)
  a.bar(x - 0.18, emp, 0.36, label='empirical', color='tab:blue')
  a.bar(x + 0.18, th, 0.36, label='theory', color='lightsteelblue')
  a.set_xticks(x); a.set_xticklabels(['immediate\nshortcut', 'wait →\nshortcut',
                                      'wait →\ndetour'], fontsize=8)
  a.legend(fontsize=8); a.set_title('G3  teacher route frequencies', fontsize=9)
  # (c) observational vs interventional
  a = ax[1, 0]
  a.bar(['observational\nP(succ | teacher crossed)', 'do(shortcut)\nP(succ)'],
        [g['G7']['observational_crossing_success'],
         g['G7']['interventional_shortcut_success']],
        color=['tab:blue', 'tab:red'])
  a.axhline(g['G7']['theory_interventional'], color='k', ls='--', lw=1,
            label='theory (1-p)^3')
  a.set_ylim(0, 1.05); a.legend(fontsize=8)
  a.set_title('G7  confounding gap = %.2f' % g['G7']['confounding_gap'], fontsize=9)
  # (d) U->A and U->S' effect sizes
  a = ax[1, 1]
  vals = [g['G4']['prefix_gap'], g['G4']['holding_gap_clear_vs_wait'],
          g['G4']['holding_gap_clear_vs_detour'], g['G5']['mean_delta'],
          g['G5']['control_delta']]
  cols = ['grey', 'tab:purple', 'tab:purple', 'darkorange', 'grey']
  a.bar(['U→A\nprefix', 'U→A hold\n(vs wait)', 'U→A hold\n(vs detour)',
         "U→S'\nholding", "U→S'\ncontrol"], vals, color=cols)
  a.tick_params(axis='x', labelsize=7.5)
  a.set_title("G4/G5  U→A action gap and U→S' next-state gap", fontsize=9)
  for aa in ax.ravel():
    aa.grid(alpha=0.3, axis='y')
  fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def _py(o):
  """Recursively convert numpy scalars for json.dump."""
  if isinstance(o, dict):
    return {k: _py(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [_py(v) for v in o]
  if isinstance(o, (np.bool_,)):
    return bool(o)
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return float(o)
  return o


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='artifacts/point_two_route_swamp_v0')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n_teacher', type=int, default=1000)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  env = TwoRouteSwampEnv(seed=args.seed)
  teacher = make_teacher(env)

  print('running teacher batch (%d episodes)...' % args.n_teacher)
  teacher_eps = [rollout(env, teacher) for _ in range(args.n_teacher)]

  gates = [
      gate1_map_dynamics(env, rng),
      gate2_swamp_statistics(env),
      gate3_teacher_behavior(env, teacher_eps),
      gate4_u_to_action(env, teacher, rng),
      gate5_u_to_next_state(env, rng),
      gate6_hiddenness(env, rng),
      gate7_obs_vs_intervention(env, teacher_eps),
      gate8_always_safe(env),
  ]
  all_pass = all(g['passed'] for g in gates)
  strip = lambda e: {k: v for k, v in e.items()
                     if k not in ('traj', 'bits_log', 'memo', 'dists')}
  report = dict(
      env_name='point_two_route_swamp_v0', seed=args.seed,
      params=dict(active_prob=env.active_prob, slow_factor=env.slow_factor,
                  success_radius=SUCCESS_RADIUS, horizon=env.max_episode_steps,
                  swamp_cells=[list(c) for c in SWAMP],
                  holding_cell=list(HOLDING), fork_cell=list(FORK),
                  start=env.START.tolist(), goal=env.GOAL.tolist()),
      gates=[_py(g) for g in gates],
      all_gates_passed=bool(all_pass),
      verdict=('QUALIFIED' if all_pass else 'NOT_QUALIFIED'),
      next_step=('STOP: gates passed; do NOT start learner training until instructed'
                 if all_pass else 'STOP: fix the failing gate(s) before proceeding'))
  json.dump(report, open(os.path.join(args.out, 'qualification_report.json'), 'w'),
            indent=2)
  plot_schematic(env, os.path.join(args.out, 'map_schematic.png'))
  plot_trajectories(env, teacher_eps, os.path.join(args.out, 'trajectories.png'))
  plot_confounding(env, report, os.path.join(args.out, 'confounding_summary.png'))

  print('=' * 74)
  print('CONFOUNDER QUALIFICATION  point_two_route_swamp_v0  (seed %d)' % args.seed)
  print('=' * 74)
  for g in gates:
    print('  [%s]  %s' % ('PASS' if g['passed'] else 'FAIL', g['name']))
    print('          thresholds: %s' % g['thresholds'])
  print('-' * 74)
  print('VERDICT:', report['verdict'], '(%d/%d gates passed)'
        % (sum(g['passed'] for g in gates), len(gates)))
  print(report['next_step'])
  print('saved report + 3 plots under', args.out)
  sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
