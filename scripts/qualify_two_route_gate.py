"""Confounder-qualification harness for the ``point_two_route_gate_v0`` env.

Runs the seven HARD GATES that must all pass before this env is used to build a
learner dataset for causal contrastive RL. Nothing here trains a critic or
touches the CRL / causal objectives -- it only certifies that the env has the
confounding structure we need, with a machine-readable verdict.

The hidden confounder is an episode-level binary gate U (open/closed, p=0.5):
  * UPPER route (short) runs straight through the gate cell.
  * LOWER route (long) detours through the bottom corridor and is always open.
  * The learner observes XY only; U is never in the observation.
  * A gate-aware oracle teacher shortcuts when U=open and takes the safe route
    when U=closed -> U drives both the teacher's action (at the fork) and the
    transition (at the gate front), yet is hidden from the learner.

Gates:
  G1 map/collision correctness           (walls block, gate leaks nothing, two routes)
  G2 oracle success under both U         (shortcut-open + safe-route under open & closed)
  G3 U -> A on shared-prefix/fork states (hidden gate changes the teacher's action)
  G4 U -> S' via matched-state clones     (identical action, different next state at gate)
  G5 hiddenness of U on shared states    (obs identical across U -> U undecodable)
  G6 observational vs interventional gap (do(shortcut) success drops vs oracle-observed)
  G7 existence of an always-safe policy  (safe route reaches goal for every U)

Run:  python scripts/qualify_two_route_gate.py
      python scripts/qualify_two_route_gate.py --out artifacts/point_two_route_gate_v0 --seed 0
Exit code 0 iff every gate passes (so a CI/Make step can STOP before CRL).
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

from crl.envs import TwoRouteGateEnv
from crl.report_maze import bfs_path, bfs_waypoints, cell_of, polyline_len

SUCCESS_RADIUS = 0.5           # min-distance-to-goal counted as success
GATE_CELL = TwoRouteGateEnv.GATE_CELL
FRONT_CELL = TwoRouteGateEnv.GATE_FRONT_CELL
FORK_CELL = TwoRouteGateEnv.FORK_CELL
START_CELL = TwoRouteGateEnv.START_CELL


# --------------------------------------------------------------------------- #
# oracles                                                                      #
# --------------------------------------------------------------------------- #
def stateful_oracle(walls):
  """BFS-waypoint follower over a FIXED wall grid (used for full rollouts)."""
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


def oracle_action(walls, state, goal):
  """Stateless single-step teacher action: direction to the next BFS cell."""
  wps = bfs_waypoints(walls, state, goal)
  if not wps or len(wps) < 2:
    return np.zeros(2, np.float32)
  return np.clip(wps[1] - np.asarray(state, float), -1, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# rollout                                                                      #
# --------------------------------------------------------------------------- #
def rollout(env, policy, gate_open, start=None, goal=None, noise=None):
  """One episode with a FORCED gate. Records the trajectory and route usage."""
  env.reset()
  env.set_gate(gate_open)
  if noise is not None:
    env._action_noise = noise
  env.state = np.asarray(start if start is not None else env.START, float).copy()
  env.goal = np.asarray(goal if goal is not None else env.GOAL, float).copy()
  g = env.goal.copy()
  traj = [env.state.copy()]
  dists = [float(np.linalg.norm(env.state - g))]
  memo = {}
  for _ in range(env.max_episode_steps):
    a = policy(env.state.copy(), g, memo)
    env.step(a)
    traj.append(env.state.copy())
    dists.append(float(np.linalg.norm(env.state - g)))
  traj = np.array(traj)
  min_dist = float(min(dists))
  used_shortcut = bool(np.any([cell_of(env._walls, p) == GATE_CELL for p in traj]))
  used_safe = bool(np.any(traj[:, 1] < 2.0))          # entered the bottom corridor
  return dict(traj=traj, gate_open=gate_open, min_dist=min_dist,
              success=bool(min_dist < SUCCESS_RADIUS),
              used_shortcut=used_shortcut, used_safe=used_safe,
              final_dist=float(dists[-1]))


def matched_states(cells, per_cell, rng):
  """Continuous states sampled inside given cells (identical XY under any U)."""
  pts = []
  for (i, j) in cells:
    for _ in range(per_cell):
      pts.append(np.array([i, j], float) + rng.uniform(0.15, 0.85, size=2))
  return pts


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #
def gate1_map_collision(env, rng):
  """Walls classify correctly (both U), the gate leaks nothing, two routes exist."""
  wo, wc = env._walls_open, env._walls_closed
  mism = 0
  for grid, is_open in ((wo, True), (wc, False)):
    env.set_gate(is_open)
    for i in range(env._height):
      for j in range(env._width):
        for _ in range(4):
          p = np.array([i, j], float) + rng.uniform(0.1, 0.9, size=2)
          if env._is_blocked(p) != bool(grid[i, j] == 1):
            mism += 1
  # boundary: outside the arena is blocked
  for p in ([-0.1, 3.5], [9.1, 3.5], [3.5, -0.1], [3.5, 5.1]):
    if not env._is_blocked(np.array(p, float)):
      mism += 1
  # dynamic no-leak: push into the gate for 25 steps, closed must NOT cross
  leak = {}
  for is_open in (True, False):
    env.set_gate(is_open); env._action_noise = 0.0
    env.state = np.array([FRONT_CELL[0] + 0.5, FRONT_CELL[1] + 0.5]); env.goal = env.GOAL.copy()
    for _ in range(25):
      env.step(np.array([1.0, 0.0]))
    leak['open' if is_open else 'closed'] = float(env.state[0])
  env._action_noise = 0.01
  no_leak = (leak['closed'] < float(GATE_CELL[0])) and (leak['open'] > float(GATE_CELL[0]) + 1)
  # two distinct routes; upper strictly shorter; closed forced through bottom
  po = bfs_path(wo, START_CELL, env.GOAL_CELL)
  pc = bfs_path(wc, START_CELL, env.GOAL_CELL)
  routes_ok = (po is not None and pc is not None and len(pc) > len(po)
               and GATE_CELL in po and any(c[1] < 2 for c in pc)
               and GATE_CELL not in pc)
  passed = (mism == 0) and no_leak and routes_ok
  return dict(
      name='G1_map_collision_correctness', passed=bool(passed),
      metrics=dict(cell_mismatches=int(mism), no_leak=bool(no_leak),
                   closed_push_x=leak['closed'], open_push_x=leak['open'],
                   len_upper_cells=len(po), len_lower_cells=len(pc),
                   upper_shorter=bool(len(pc) > len(po))),
      thresholds='mismatches==0 & no_leak & upper<lower & routes distinct',
      detail='closed push must stop before gate x=%d; open must cross' % GATE_CELL[0])


def gate2_oracle_success(env, n=25):
  """Gate-aware teacher, forced-shortcut, and safe route all reach the goal."""
  aware = {}
  for is_open in (True, False):
    walls = env._walls_open if is_open else env._walls_closed
    eps = [rollout(env, stateful_oracle(walls), is_open) for _ in range(n)]
    aware['open' if is_open else 'closed'] = dict(
        success=float(np.mean([e['success'] for e in eps])),
        shortcut=float(np.mean([e['used_shortcut'] for e in eps])),
        safe=float(np.mean([e['used_safe'] for e in eps])))
  # shortcut oracle when the gate IS open (should sail through)
  sc_open = [rollout(env, stateful_oracle(env._walls_open), True) for _ in range(n)]
  sc_open_succ = float(np.mean([e['success'] for e in sc_open]))
  # safe oracle under BOTH gate states (never needs the gate)
  safe = {}
  for is_open in (True, False):
    eps = [rollout(env, stateful_oracle(env._walls_closed), is_open) for _ in range(n)]
    safe['open' if is_open else 'closed'] = float(np.mean([e['success'] for e in eps]))
  passed = (aware['open']['success'] >= 0.99 and aware['closed']['success'] >= 0.99
            and aware['open']['shortcut'] >= 0.99 and aware['closed']['safe'] >= 0.99
            and sc_open_succ >= 0.99 and safe['open'] >= 0.99 and safe['closed'] >= 0.99)
  return dict(
      name='G2_oracle_success_both_gate_states', passed=bool(passed),
      metrics=dict(gate_aware=aware, shortcut_open_success=sc_open_succ,
                   safe_route=safe),
      thresholds='all success >= 0.99; teacher shortcuts iff open, safe iff closed',
      detail='shortcut-open and safe-route both solvable under each U')


def gate3_u_to_action(env, rng, per_cell=40):
  """On shared-prefix (start) + fork states, U changes the teacher action."""
  wo, wc = env._walls_open, env._walls_closed
  goal = env.GOAL
  res = {}
  for label, cell in (('prefix_start', START_CELL), ('fork', FORK_CELL)):
    gaps = []
    for s in matched_states([cell], per_cell, rng):
      a_o = oracle_action(wo, s, goal)
      a_c = oracle_action(wc, s, goal)
      gaps.append(float(np.linalg.norm(a_o - a_c)))
    res[label] = dict(mean_action_gap=float(np.mean(gaps)),
                      frac_gap_gt_0p3=float(np.mean(np.array(gaps) > 0.3)))
  # confounder present iff U flips the action at the fork but NOT on the prefix
  passed = (res['fork']['mean_action_gap'] > 0.5
            and res['fork']['frac_gap_gt_0p3'] > 0.9
            and res['prefix_start']['mean_action_gap'] < 0.1)
  return dict(
      name='G3_U_to_action_on_shared_states', passed=bool(passed),
      metrics=res,
      thresholds='fork gap > 0.5 (U changes A) AND prefix gap < 0.1 (agrees pre-fork)',
      detail='measured ONLY on states reachable + observationally identical under both U')


def gate4_u_to_next_state(env, rng, per_cell=40, k=3):
  """Identical action from matched gate-front states -> different S' under U."""
  goal = env.GOAL
  def clone_delta(cell, action):
    ds = []
    for s in matched_states([cell], per_cell, rng):
      ends = {}
      for is_open in (True, False):
        env.set_gate(is_open); env._action_noise = 0.0
        env.state = s.copy(); env.goal = goal.copy()
        for _ in range(k):
          env.step(action)
        ends[is_open] = env.state.copy()
      ds.append(float(np.linalg.norm(ends[True] - ends[False])))
    env._action_noise = 0.01
    return float(np.mean(ds))
  front = clone_delta(FRONT_CELL, np.array([1.0, 0.0]))    # push toward the gate
  control = clone_delta((5, 1), np.array([1.0, 0.0]))      # bottom corridor: gate-free
  passed = (front > 0.5) and (control < 0.05)
  return dict(
      name='G4_U_to_next_state_matched_clones', passed=bool(passed),
      metrics=dict(gate_front_deltaSprime=front, control_deltaSprime=control,
                   k_steps=k),
      thresholds='gate-front dS\' > 0.5 (U changes dynamics) AND control dS\' < 0.05',
      detail='same start state, same action; only U differs -> pure U->S\' effect')


def gate5_hiddenness(env, rng, per_cell=60):
  """On shared states the observation is bit-identical across U (U undecodable)."""
  max_diff = 0.0
  states = matched_states([START_CELL, FORK_CELL, FRONT_CELL], per_cell, rng)
  for s in states:
    env.goal = env.GOAL.copy()
    env.state = s.copy(); env.set_gate(True); o_open = env._get_obs().copy()
    env.state = s.copy(); env.set_gate(False); o_closed = env._get_obs().copy()
    max_diff = max(max_diff, float(np.max(np.abs(o_open - o_closed))))
  # single-observation Bayes predictor of U is exactly chance (identical dists)
  passed = (max_diff == 0.0) and (env.obs_dim == 2)
  return dict(
      name='G5_hiddenness_of_U_on_shared_states', passed=bool(passed),
      metrics=dict(max_obs_abs_diff=max_diff, n_states=len(states),
                   obs_dim=int(env.obs_dim), bayes_optimal_U_accuracy=0.5),
      thresholds='max |obs_open - obs_closed| == 0 & obs_dim==2 (no gate bit)',
      detail='observation function does not depend on U -> MI(U;obs)=0 on shared states')


def gate6_obs_vs_intervention(env, n=200):
  """Observational shortcut-success (~1) vs do(shortcut) success (~0.5)."""
  rng = np.random.default_rng(0)
  # Observational: gate-aware teacher on random U (its natural behavior policy).
  obs_short_succ, obs_short_n, u_short = [], 0, []
  for _ in range(n):
    is_open = bool(rng.random() < 0.5)
    walls = env._walls_open if is_open else env._walls_closed
    e = rollout(env, stateful_oracle(walls), is_open)
    if e['used_shortcut']:
      obs_short_n += 1
      obs_short_succ.append(e['success'])
    u_short.append((int(is_open), int(e['used_shortcut'])))
  obs_success = float(np.mean(obs_short_succ)) if obs_short_succ else float('nan')
  u_arr = np.array(u_short)
  corr_u_shortcut = float(np.corrcoef(u_arr[:, 0], u_arr[:, 1])[0, 1])
  # Interventional do(shortcut): force the shortcut policy on random U.
  iv = [rollout(env, stateful_oracle(env._walls_open), bool(rng.random() < 0.5))
        for _ in range(n)]
  iv_success = float(np.mean([e['success'] for e in iv]))
  iv_pass = float(np.mean([e['used_shortcut'] for e in iv]))   # physically got through
  gap = obs_success - iv_success
  passed = (obs_success >= 0.95) and (iv_success <= 0.75) and (gap >= 0.3) \
      and (corr_u_shortcut >= 0.9)
  return dict(
      name='G6_observational_vs_interventional_gap', passed=bool(passed),
      metrics=dict(observational_shortcut_success=obs_success,
                   interventional_shortcut_success=iv_success,
                   confounding_gap=gap, corr_U_shortcut=corr_u_shortcut,
                   interventional_shortcut_pass_rate=iv_pass,
                   n_observational_shortcut=obs_short_n),
      thresholds='obs>=0.95 & do(shortcut)<=0.75 & gap>=0.3 & corr(U,shortcut)>=0.9',
      detail='shortcut looks safe under the teacher only because it is taken iff U=open')


def gate7_always_safe(env, n=25):
  """A gate-independent policy (safe route) reaches the goal for every U."""
  res = {}
  entered_gate = 0
  for is_open in (True, False):
    eps = [rollout(env, stateful_oracle(env._walls_closed), is_open) for _ in range(n)]
    res['open' if is_open else 'closed'] = float(np.mean([e['success'] for e in eps]))
    entered_gate += sum(e['used_shortcut'] for e in eps)
  invariant = abs(res['open'] - res['closed']) < 0.02
  passed = (res['open'] >= 0.99 and res['closed'] >= 0.99 and invariant
            and entered_gate == 0)
  return dict(
      name='G7_always_safe_policy_exists', passed=bool(passed),
      metrics=dict(safe_success=res, success_invariant_across_U=bool(invariant),
                   gate_cell_entries=int(entered_gate)),
      thresholds='safe success >= 0.99 for both U, U-invariant, never enters the gate',
      detail='the lower route solves the task without ever depending on U')


# --------------------------------------------------------------------------- #
# plots                                                                        #
# --------------------------------------------------------------------------- #
def _draw_base(ax, env):
  walls = env._walls_closed
  H, W = walls.shape
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.55, extent=[0, H, 0, W])
  # gate cell drawn as a hatched orange box (the U-controlled cell)
  ax.add_patch(Rectangle((GATE_CELL[0], GATE_CELL[1]), 1, 1, fill=False,
                         hatch='///', edgecolor='darkorange', lw=1.5))
  ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
  ax.set_xlim(0, H); ax.set_ylim(0, W)


def plot_schematic(env, out):
  fig, ax = plt.subplots(figsize=(6.2, 3.6))
  _draw_base(ax, env)
  po = [np.array(c) + 0.5 for c in bfs_path(env._walls_open, START_CELL, env.GOAL_CELL)]
  pc = [np.array(c) + 0.5 for c in bfs_path(env._walls_closed, START_CELL, env.GOAL_CELL)]
  po, pc = np.array(po), np.array(pc)
  ax.plot(po[:, 0], po[:, 1], '-', color='tab:blue', lw=2.4,
          label='UPPER / short (gate, U=open)')
  ax.plot(pc[:, 0], pc[:, 1], '--', color='tab:green', lw=2.2,
          label='LOWER / long (safe, always open)')
  ax.scatter(*env.START, c='black', s=60, zorder=5); ax.text(*(env.START + [0, .3]), 'S')
  ax.scatter(*env.GOAL, c='red', marker='*', s=160, zorder=5)
  ax.text(*(env.GOAL + [-.1, .3]), 'G')
  ax.scatter(FORK_CELL[0] + .5, FORK_CELL[1] + .5, c='purple', s=45, zorder=5,
             label='fork')
  ax.scatter(FRONT_CELL[0] + .5, FRONT_CELL[1] + .5, marker='x', c='darkorange',
             s=55, zorder=5, label='gate-front')
  ax.set_title('point_two_route_gate_v0 — hidden gate U toggles the shortcut',
               fontsize=10)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8,
            frameon=False)
  fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_trajectories(env, out):
  rng = np.random.default_rng(0)
  panels = [
      ('teacher, U=open\n(takes shortcut)', stateful_oracle(env._walls_open), True),
      ('teacher, U=closed\n(safe detour)', stateful_oracle(env._walls_closed), False),
      ('do(shortcut), U=open\n(succeeds)', stateful_oracle(env._walls_open), True),
      ('do(shortcut), U=closed\n(stuck at gate)', stateful_oracle(env._walls_open), False),
      ('always-safe, U=open', stateful_oracle(env._walls_closed), True),
      ('always-safe, U=closed', stateful_oracle(env._walls_closed), False),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(11, 6))
  for ax, (title, pol, is_open) in zip(axes.ravel(), panels):
    _draw_base(ax, env)
    for _ in range(6):
      e = rollout(env, pol, is_open)
      t = e['traj']
      col = 'tab:green' if e['success'] else 'tab:red'
      ax.plot(t[:, 0], t[:, 1], '-', lw=1.1, alpha=0.7, color=col)
    ax.scatter(*env.START, c='black', s=35, zorder=5)
    ax.scatter(*env.GOAL, c='red', marker='*', s=110, zorder=5)
    sr = np.mean([rollout(env, pol, is_open)['success'] for _ in range(20)])
    ax.set_title('%s  succ=%.2f' % (title, sr), fontsize=9)
  fig.suptitle('Trajectories (green=reached goal, red=failed)', fontsize=11)
  fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=105); plt.close(fig)


def plot_confounding(env, report, out):
  g6 = next(g for g in report['gates'] if g['name'].startswith('G6'))['metrics']
  g3 = next(g for g in report['gates'] if g['name'].startswith('G3'))['metrics']
  g4 = next(g for g in report['gates'] if g['name'].startswith('G4'))['metrics']
  fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
  ax[0].bar(['observational\nP(succ|shortcut)', 'do(shortcut)\nP(succ)'],
            [g6['observational_shortcut_success'],
             g6['interventional_shortcut_success']],
            color=['tab:blue', 'tab:red'])
  ax[0].set_ylim(0, 1.05); ax[0].set_title('G6 confounding gap = %.2f'
                                           % g6['confounding_gap'], fontsize=9)
  ax[1].bar(['prefix\n(pre-fork)', 'fork'],
            [g3['prefix_start']['mean_action_gap'], g3['fork']['mean_action_gap']],
            color=['grey', 'tab:purple'])
  ax[1].set_title('G3  U→A  mean |Δaction|', fontsize=9)
  ax[2].bar(['gate-front', 'control\n(corridor)'],
            [g4['gate_front_deltaSprime'], g4['control_deltaSprime']],
            color=['darkorange', 'grey'])
  ax[2].set_title("G4  U→S'  mean ΔS' (same action)", fontsize=9)
  for a in ax:
    a.grid(alpha=0.3, axis='y')
  fig.tight_layout(); fig.savefig(out, dpi=105); plt.close(fig)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='artifacts/point_two_route_gate_v0')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  env = TwoRouteGateEnv(seed=args.seed)

  gates = [
      gate1_map_collision(env, rng),
      gate2_oracle_success(env),
      gate3_u_to_action(env, rng),
      gate4_u_to_next_state(env, rng),
      gate5_hiddenness(env, rng),
      gate6_obs_vs_intervention(env),
      gate7_always_safe(env),
  ]
  all_pass = all(g['passed'] for g in gates)
  report = dict(
      env_name='point_two_route_gate_v0', seed=args.seed,
      gate_prob=0.5, success_radius=SUCCESS_RADIUS,
      map=dict(shape=list(env._walls_closed.shape),
               start=env.START.tolist(), goal=env.GOAL.tolist(),
               gate_cell=list(GATE_CELL), fork_cell=list(FORK_CELL),
               gate_front_cell=list(FRONT_CELL)),
      gates=gates,
      all_gates_passed=bool(all_pass),
      verdict=('QUALIFIED' if all_pass else 'NOT_QUALIFIED'),
      next_step=('STOP: gates passed; do NOT start CRL training until instructed'
                 if all_pass else 'STOP: fix the failing gate(s) before proceeding'))

  json.dump(report, open(os.path.join(args.out, 'qualification_report.json'), 'w'),
            indent=2)
  plot_schematic(env, os.path.join(args.out, 'map_schematic.png'))
  plot_trajectories(env, os.path.join(args.out, 'oracle_trajectories.png'))
  plot_confounding(env, report, os.path.join(args.out, 'confounding_summary.png'))

  # ---- console summary ----
  print('=' * 74)
  print('CONFOUNDER QUALIFICATION  point_two_route_gate_v0  (seed %d)' % args.seed)
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
