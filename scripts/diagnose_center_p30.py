"""Fixed-center controller FAILURE TAXONOMY under the p_active=0.30 protocol.

Diagnosis only. Does NOT modify env/controller/horizon/mud/protocol. The main
run replays RP.run_route's EXACT center control law (route_command + walker +
handoff to base_act) with full per-step instrumentation added -- the action
computation is byte-identical to the benchmark controller, only observation is
added.

Because the center route never enters a trigger band (|y| stays < 0.55) the env
never fires rockfall, so _dead is never set and the env never terminates early
(step() returns done=False always). EVERY center failure is therefore a horizon
timeout; the taxonomy explains WHY it timed out.

Outputs (under --out, default artifacts/center_diag_p30/):
  episodes.jsonl     one row per episode, full field list
  summary.json       category counts/%, rates, distributions, env-bug check
Also runs, on the SAME episode seeds (diagnosis-only, separate out dirs):
  A. mud/drag disabled  (env.mud_drag = 0)
  B. longer horizon     (env.max_episode_steps *= --horizon-mult)
and a paired-mask invariance check (center must be mask-invariant).

Usage: python scripts/diagnose_center_p30.py --n 600
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod            # noqa: E402
from crl import rockfall_ant as RA          # noqa: E402
import litter_pilot_common as C             # noqa: E402
import rockfall_pilot as RP                 # noqa: E402
import rockfall_v2_teacher as V2            # noqa: E402

SEED = 20_260_726
STUCK_VX = RA.TRIG_MIN_VX          # 0.1: env's own "marching" threshold
STUCK_STEPS = 100                  # >=100 consecutive low-vx steps = wedged
NEAR_GOAL_DIST = 1.5               # within this of goal = "near goal"
ROUTE_DEV_Y = 0.9                  # pre-handoff |y| beyond this = deviated
UNHEALTHY_TAIL = 30                # steps at episode end to judge "ended down"


def run_center_instrumented(env, o, walker, base_act, horizon=None):
  """RP.run_route(route='center') control law, verbatim, + instrumentation.
  Returns the full taxonomy row for one episode."""
  T = horizon or env.max_episode_steps
  true_goal = o[29:31].copy()
  handoff = False
  handoff_step = -1
  hit_r = 0.0
  falls = 0
  dead_at = first_hit_t = None
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  xs_all, ys_all, zs_all, vxs_all = [], [], [], []
  pre_handoff_absy = []
  low_vx_run = low_vx_max = 0
  any_nan = False
  for t in range(T):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
      handoff_step = t
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
      pre_handoff_absy.append(abs(y))
    o, r, _, info = env.step(a)
    q = np.asarray(env._env.data.qpos)
    z = float(q[2])
    vx = float(env._env.data.qvel[0])
    if not np.isfinite(o).all() or not np.isfinite(q).all():
      any_nan = True
    xs_all.append(float(o[0]))
    ys_all.append(float(o[1]))
    zs_all.append(z)
    vxs_all.append(vx)
    low_vx_run = low_vx_run + 1 if vx < STUCK_VX else 0
    low_vx_max = max(low_vx_max, low_vx_run)
    hit_r = max(hit_r, float(r))
    if info['rock_ant_contact'] and first_hit_t is None:
      first_hit_t = t
    if info['dead'] and dead_at is None:
      dead_at = t
    if not env.dead and (RP.torso_up_z(q) < 0.0 or z < 0.2):
      falls += 1
    if hit_r > 0:
      break
    if dead_at is not None and t > dead_at + 5:
      break
  n = len(xs_all)
  xs = np.asarray(xs_all)
  ys = np.asarray(ys_all)
  zs = np.asarray(zs_all)
  vxs = np.asarray(vxs_all)
  final_xy = np.array([xs[-1], ys[-1]])
  final_dist = float(np.linalg.norm(final_xy - true_goal))
  success = int(hit_r > 0)
  timed_out = int(success == 0 and dead_at is None)
  tail = zs[-UNHEALTHY_TAIL:] if n >= UNHEALTHY_TAIL else zs
  ended_unhealthy = bool(np.mean(tail) < 0.3 or np.min(tail) < 0.2)
  stuck = bool(low_vx_max >= STUCK_STEPS)
  mode_at_end = 'base_handoff' if handoff else 'center'
  row = {
      'success': success,
      'ep_length': n,
      'termination': ('success' if success else
                      'dead' if dead_at is not None else 'timeout'),
      'final_x': round(float(xs[-1]), 3),
      'final_y': round(float(ys[-1]), 3),
      'final_dist_to_goal': round(final_dist, 3),
      'goal_xy': [round(float(true_goal[0]), 3), round(float(true_goal[1]), 3)],
      'max_x': round(float(xs.max()), 3),
      'min_torso_z': round(float(zs.min()), 3),
      'fell': bool(falls > 0),
      'ended_unhealthy': ended_unhealthy,
      'timed_out': bool(timed_out),
      'stuck': stuck,
      'longest_low_vx_interval': int(low_vx_max),
      'mean_vx': round(float(vxs.mean()), 4),
      'min_vx': round(float(vxs.min()), 4),
      'max_abs_y': round(float(np.abs(ys).max()), 3),
      'pre_handoff_max_abs_y': round(float(max(pre_handoff_absy)
                                          if pre_handoff_absy else 0.0), 3),
      'handoff_step': int(handoff_step),
      'mode_at_end': mode_at_end,
      'n_trigger': int(sum(env._triggered)),
      'n_drop': int(sum(env._dropped)),
      'n_hit': int(sum(env._hit)),
      'any_nan': bool(any_nan),
  }
  return row


def classify(r):
  """Exactly one primary category for an UNSUCCESSFUL episode."""
  if r['any_nan']:
    return 'numerical_unknown'
  if r['pre_handoff_max_abs_y'] > ROUTE_DEV_Y:
    return 'route_deviation_wall'
  if r['ended_unhealthy'] or r['fell']:
    return 'physical_fall'
  reached = r['handoff_step'] >= 0
  if not reached:
    return ('stuck_oscillating' if r['stuck']
            else 'timeout_insufficient_progress')
  # reached the handoff / turn region
  if r['final_dist_to_goal'] < NEAR_GOAL_DIST:
    return 'timeout_near_goal'
  return 'controller_handoff_failure'


CATEGORIES = ['timeout_near_goal', 'timeout_insufficient_progress',
              'physical_fall', 'stuck_oscillating', 'route_deviation_wall',
              'controller_handoff_failure', 'numerical_unknown']


def run_block(env, walker, base_act, n, horizon=None):
  rows = []
  for _ in range(n):
    o = env.reset()
    rows.append(run_center_instrumented(env, o, walker, base_act, horizon))
  return rows


def summarize(rows, label):
  n = len(rows)
  succ = [r for r in rows if r['success']]
  fail = [r for r in rows if not r['success']]
  cats = {c: 0 for c in CATEGORIES}
  for r in fail:
    cats[classify(r)] += 1
  env_bug = [i for i, r in enumerate(rows)
             if r['n_trigger'] or r['n_drop'] or r['n_hit']]
  def dist(vals):
    a = np.asarray(vals, float)
    if not len(a):
      return None
    return {'mean': round(float(a.mean()), 3),
            'p10': round(float(np.percentile(a, 10)), 3),
            'p50': round(float(np.percentile(a, 50)), 3),
            'p90': round(float(np.percentile(a, 90)), 3),
            'min': round(float(a.min()), 3), 'max': round(float(a.max()), 3)}
  # failure location histogram along x (max_x reached at failure)
  fx = [r['max_x'] for r in fail]
  hist_edges = list(range(0, 12))
  fx_hist = [int(((np.asarray(fx) >= lo) & (np.asarray(fx) < lo + 1)).sum())
             for lo in hist_edges] if fx else []
  return {
      'label': label, 'n': n,
      'success_rate': round(len(succ) / n, 4),
      'n_success': len(succ), 'n_fail': len(fail),
      'env_bug_episodes_nonzero_rockfall': env_bug,
      'category_counts': cats,
      'category_pct': {c: round(100 * cats[c] / n, 2) for c in CATEGORIES},
      'mean_ep_length_success': (round(float(np.mean([r['ep_length']
                                 for r in succ])), 1) if succ else None),
      'mean_ep_length_fail': (round(float(np.mean([r['ep_length']
                              for r in fail])), 1) if fail else None),
      'final_dist_all': dist([r['final_dist_to_goal'] for r in rows]),
      'final_dist_fail': dist([r['final_dist_to_goal'] for r in fail]),
      'max_x_fail': dist(fx),
      'min_torso_z_all': dist([r['min_torso_z'] for r in rows]),
      'min_torso_z_fail': dist([r['min_torso_z'] for r in fail]),
      'failure_x_hist_1m_bins_0to11': fx_hist,
      'longest_low_vx_fail': dist([r['longest_low_vx_interval'] for r in fail]),
  }


def mask_invariance(cfg, walker, base_act, pa, k=40):
  """For k paired initial states, force each of the 4 mask patterns and check
  the center obs trajectory is byte-identical (center must be mask-invariant)."""
  PATS = {'all_clear': (0, 0, 0, 0), 'left_only': (1, 0, 0, 0),
          'right_only': (0, 0, 1, 0), 'both_sides': (1, 0, 1, 0)}
  envs = {p: V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED + 555), pa)
      for p in PATS}
  per_pattern_succ = {p: [] for p in PATS}
  max_obs_div = 0.0
  succ_disagree = 0
  for _ in range(k):
    base = {}
    for p, mask in PATS.items():
      o = envs[p].reset(mask=mask)
      row = run_center_instrumented(envs[p], o, walker, base_act)
      per_pattern_succ[p].append(row['success'])
      base[p] = row
    # obs-trajectory identity is guaranteed only if inits match; we compare the
    # per-episode success + final state across the 4 forced masks at this index
    ss = [base[p]['success'] for p in PATS]
    fx = [base[p]['final_x'] for p in PATS]
    fy = [base[p]['final_y'] for p in PATS]
    if len(set(ss)) > 1:
      succ_disagree += 1
    max_obs_div = max(max_obs_div, float(np.ptp(fx)), float(np.ptp(fy)))
  return {'k_paired_inits': k,
          'per_pattern_success': {p: round(float(np.mean(v)), 3)
                                  for p, v in per_pattern_succ.items()},
          'paired_success_disagreements': succ_disagree,
          'max_final_xy_spread_across_masks': round(max_obs_div, 6)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=600)
  ap.add_argument('--p-active', type=float, default=0.30)
  ap.add_argument('--horizon-mult', type=float, default=2.0)
  ap.add_argument('--out', default='artifacts/center_diag_p30')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  pa = args.p_active

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  def fresh_env():
    return V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED), pa)

  # ---- MAIN: baseline center diagnosis ----
  print(f'MAIN center diag: n={args.n} p_active={pa}', flush=True)
  env = fresh_env()
  base_horizon = env.max_episode_steps
  rows = run_block(env, walker, base_act, args.n)
  with open(os.path.join(args.out, 'episodes.jsonl'), 'w') as f:
    for i, r in enumerate(rows):
      r2 = dict(r); r2['episode_id'] = i
      f.write(json.dumps(r2) + '\n')
  summ = summarize(rows, 'baseline_p30')
  print('success_rate', summ['success_rate'], '| cats', summ['category_counts'],
        flush=True)
  assert not summ['env_bug_episodes_nonzero_rockfall'], \
      'ENV BUG: nonzero rockfall on a center episode!'
  print('ENV-BUG CHECK OK: trigger/drop/hit == 0 on all center episodes',
        flush=True)

  # ---- ABLATION A: mud/drag disabled (same seeds) ----
  print('ABLATION A: mud_drag=0 (same seeds)', flush=True)
  envA = fresh_env()
  envA.mud_drag = 0.0
  rowsA = run_block(envA, walker, base_act, args.n)
  summA = summarize(rowsA, 'ablationA_mud_off')
  print('  A success_rate', summA['success_rate'], flush=True)

  # ---- ABLATION B: longer horizon (same seeds), mud ON ----
  bh = int(base_horizon * args.horizon_mult)
  print(f'ABLATION B: horizon {base_horizon}->{bh} (same seeds)', flush=True)
  envB = fresh_env()
  envB.max_episode_steps = bh
  rowsB = run_block(envB, walker, base_act, args.n, horizon=bh)
  summB = summarize(rowsB, 'ablationB_long_horizon')
  print('  B success_rate', summB['success_rate'], flush=True)

  # ---- mask-invariance ----
  print('mask-invariance check (paired inits across forced masks)', flush=True)
  inv = mask_invariance(cfg, walker, base_act, pa)
  print('  ', inv, flush=True)

  # ---- representative failures per major category (episode ids) ----
  reps = {c: [i for i, r in enumerate(rows)
              if not r['success'] and classify(r) == c][:8]
          for c in CATEGORIES}

  out = {'provenance': {'seed': SEED, 'p_active': pa,
                        'base_horizon': base_horizon,
                        'stuck_vx': STUCK_VX, 'stuck_steps': STUCK_STEPS,
                        'near_goal_dist': NEAR_GOAL_DIST,
                        'route_dev_y': ROUTE_DEV_Y,
                        'success_dist': 0.5,
                        'controllers': {'walker': RP.WALKER, 'base': RP.BASE}},
         'baseline': summ,
         'ablation_A_mud_off': summA,
         'ablation_B_long_horizon': summB,
         'mask_invariance': inv,
         'representative_failure_ids': reps}
  json.dump(out, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
  print('\nwrote', os.path.join(args.out, 'summary.json'), flush=True)
  # concise verdict
  print(f"\nBASELINE {summ['success_rate']:.3f} | A(mud off) "
        f"{summA['success_rate']:.3f} | B(2x horizon) {summB['success_rate']:.3f}",
        flush=True)


if __name__ == '__main__':
  main()
