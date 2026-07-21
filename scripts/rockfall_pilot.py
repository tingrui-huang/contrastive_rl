"""Rockfall variant: pre-dataset pilot diagnostics (NO dataset collection).

Runs, on the new offline_ant_umaze_rockfall env with the frozen walker +
frozen 0.89 base handoff (both untouched):

  1. per-site activation frequency over many natural resets (~P_ACTIVE);
  2. paired-reset hiddenness: same initial state + deterministic policy,
     different masks -> learner obs must stay identical until the first
     physical rockfall interaction;
  3. success rates: known-clear side / active hazardous side / center route
     / privileged teacher / mask-blind fixed-side controller;
  4. hazardous-side breakdown: severe / impaired / mild rates, recovery
     success, lateral escape, timeout-fall-collapse split;
  5. route statistics: low-level wobble vs high-level route identifiability,
     plus the center route's trigger rate (must be ~0).

Saves artifacts/rockfall_pilot/report.json (+ _detail.json) and prints a
target check. Targets are TUNING aids, not frozen gates:
  clear-side >= 0.85 | hazardous 0.05-0.25 | center 0.65-0.80 |
  teacher >= 0.85 | recovery possible but uncommon.

Usage: python scripts/rockfall_pilot.py [--eps 50] [--seed 60001]
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

import mujoco                              # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import probe                     # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402

WALKER = 'artifacts/walker/phase1/walker_best.pkl'
BASE = 'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl'
OUT_DIR = 'artifacts/rockfall_pilot'

LANE = 1.1
HANDOFF_X = 6.0
V_SIDE = 1.1                   # side-lane speed: 1.4 (probe.V_FAST) costs
                               # 10-20% gait falls even on the PLAIN maze
                               # and 1.2 still fell 0.10-0.17, capping the
                               # teacher below 0.85; 1.1 trades pace for a
                               # stabler gait (still 1.4x the center 0.8)
V_CENTER = 0.8                 # careful pace through the mud drag (the
                               # center cost is now the DRAG, not terrain)
#: center-route unstick (part of the cautious center policy's definition);
#: NUDGE_Y kept small so the center route can NEVER wander into a trigger
#: band (|y| >= 0.55).
STALL_WINDOW, STALL_MIN_DX = 40, 0.25
NUDGE_Y, NUDGE_STEPS = 0.35, 25
ROUTE_ZONE_X = (2.3, 5.7)      #: x-window for route/wobble statistics
ESCAPE_WINDOW = 120            #: steps after first hit to count an escape

#: STRUCTURAL-GAP qualification (user decision 2026-07-21): absolute
#: anchors are hostage to the frozen controller stack's ~0.85-0.90 ceiling
#: and to seed-level variance; what the benchmark needs is the ORDERING
#: with margins. center_tuning_band is a tuning aid only (non-gating).
TARGETS = {'gap_clear_over_center': 0.05, 'gap_teacher_over_center': 0.05,
           'gap_center_over_blind': 0.10, 'hazard_max': 0.25,
           'center_tuning_band': (0.74, 0.80)}


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


#: side-lane unstick references: OUTWARD only (a wedged lane ant must pull
#: AWAY from the center terrain -- an inward nudge sends it back onto the
#: ridge edge at speed). Instrumented clear-lane runs showed 8/11 failures
#: were permanent wedges against ridge-end faces with no unstick to escape
#: them -- the stall-unstick is part of EVERY route's definition.
SIDE_NUDGE_OUT, SIDE_NUDGE_IN = 1.45, 1.30


def route_command(route, t, x_hist, nudge):
  """(y_ref, v_ref) for the route; every route gets the stall-unstick."""
  if route == 'center':
    y0, v = 0.0, V_CENTER
    refs = (NUDGE_Y, -NUDGE_Y)
  else:
    sgn = 1.0 if route == 'left' else -1.0
    y0, v = sgn * LANE, V_SIDE
    refs = (sgn * SIDE_NUDGE_OUT, sgn * SIDE_NUDGE_IN)
  y_cmd = y0
  if t < nudge['until']:
    y_cmd = refs[0] if nudge['sign'] > 0 else refs[1]
  elif (len(x_hist) > STALL_WINDOW
        and x_hist[-1] - x_hist[-STALL_WINDOW] < STALL_MIN_DX):
    nudge['until'] = t + NUDGE_STEPS
    nudge['sign'] = -nudge['sign']
    x_hist.clear()
    y_cmd = refs[0] if nudge['sign'] > 0 else refs[1]
  return y_cmd, v


def run_route(env, o, walker, base_act, route):
  """One full episode from the CURRENT reset; returns the diagnostics row."""
  true_goal = o[29:31].copy()
  handoff = False
  hit_r = 0.0
  falls = 0
  dead_at = first_hit_t = None
  hit_y = None
  escaped = False
  ys, xs = [], []
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  for t in range(env.max_episode_steps):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      if ROUTE_ZONE_X[0] <= x <= ROUTE_ZONE_X[1]:
        ys.append(y)
        xs.append(x)
      x_hist.append(x)
      y_cmd, v_cmd = route_command(route, t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
    o, r, _, info = env.step(a)
    hit_r = max(hit_r, float(r))
    if info['rock_ant_contact'] and first_hit_t is None:
      first_hit_t = t
      hit_y = float(o[1])
    if (first_hit_t is not None and not escaped and not env.dead
        and t <= first_hit_t + ESCAPE_WINDOW
        and hit_y is not None and abs(hit_y) >= TRIG_Y0
        and abs(float(o[1])) < TRIG_Y0):
      escaped = True                     # left the trigger band after a hit
    if info['dead'] and dead_at is None:
      dead_at = t
    q = np.asarray(env._env.data.qpos)
    if not env.dead and (torso_up_z(q) < 0.0 or float(q[2]) < 0.2):
      falls += 1
    if hit_r > 0:
      break
    if dead_at is not None and t > dead_at + 5:
      break
  ys = np.asarray(ys) if ys else np.zeros(1)
  mean_y = float(np.mean(ys))
  obs_route = ('left' if mean_y > 0.5 else
               'right' if mean_y < -0.5 else 'center')
  n_hit = sum(env._hit)
  sev_hits = [env._severity[i] for i in range(4) if env._hit[i]]
  outcome = ('success' if hit_r > 0 else
             'collapse' if dead_at is not None else
             'fall' if falls > 0 else 'timeout')
  return {'route': route, 'mask': list(env.rockfall_mask),
          'severity': list(env._severity),
          'success': float(hit_r), 'outcome': outcome,
          'dead': dead_at is not None, 'fell': falls > 0, 'steps': t + 1,
          'triggered': list(env._triggered), 'dropped': list(env._dropped),
          'hit': list(env._hit), 'n_hit': n_hit, 'sev_hits': sev_hits,
          'first_hit_t': first_hit_t, 'escaped': bool(escaped),
          'mean_y': mean_y, 'std_y': float(np.std(ys)),
          'obs_route': obs_route, 'handoff': handoff}


TRIG_Y0 = RA.TRIG_Y_BAND[0]


def set_state(env, qpos, qvel, goal_xy, mask, severities):
  """Probe-only paired restore (same pattern as diagnose_naive_policy)."""
  env.reset(mask=mask, severities=severities)   # rocks parked, flags clean
  d = env._env.data
  d.qpos[:RA.NQ_ANT] = qpos
  d.qvel[:RA.NV_ANT] = qvel
  d.qacc_warmstart[:] = 0.0
  env._goal_vec = np.zeros(29, np.float32)
  env._goal_vec[:2] = goal_xy
  env._goal_state_full = env._goal_vec.copy()
  env._env.goal = np.asarray(goal_xy, float).copy()
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()
  return env._flatten(env._last_obs)


def paired_hiddenness(env, walker, base_act, n_pairs, horizon=260):
  """Same state + policy, masks (0,0,0,0) vs (1,1,0,0), route left."""
  mask_a, mask_b = (0, 0, 0, 0), (1, 1, 0, 0)
  sev = ('mild', 'mild', 'mild', 'mild')
  rows = []
  for _ in range(n_pairs):
    env.reset()
    q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
    goal = env._flatten(env._last_obs)[29:31].copy()
    tr = {}
    for tag, mask in (('a', mask_a), ('b', mask_b)):
      o = set_state(env, q0, v0, goal, mask, sev)
      obs, acts, any_c, ant_c, drops = [], [], [], [], []
      for t in range(horizon):
        a = walker(o, LANE, V_SIDE)      # deterministic left-lane policy
        obs.append(o.copy())
        acts.append(a.copy())
        o, _, _, info = env.step(a)
        any_c.append(bool(info['rock_any_contact']))
        ant_c.append(bool(info['rock_ant_contact']))
        drops.append(any(info['dropped']))
      tr[tag] = (np.asarray(obs), np.asarray(acts), any_c, ant_c, drops)
    oa, ob = tr['a'][0], tr['b'][0]
    dif = np.abs(oa - ob).max(axis=1)
    div = int(np.argmax(dif > 1e-9)) if (dif > 1e-9).any() else None
    adif = np.abs(tr['a'][1] - tr['b'][1]).max(axis=1)
    adiv = int(np.argmax(adif > 1e-9)) if (adif > 1e-9).any() else None
    first_any = (tr['b'][2].index(True) if True in tr['b'][2] else None)
    first_ant = (tr['b'][3].index(True) if True in tr['b'][3] else None)
    first_drop = (tr['b'][4].index(True) if True in tr['b'][4] else None)
    #: obs[t] is recorded BEFORE step t, so the earliest step an event
    #: during step T can touch is obs[T+1]. The first triggered physical
    #: rockfall interaction is the DROP itself (the shelf releases):
    #: require strict div > drop_step. Empirically nearly all pairs stay
    #: EXACTLY zero until first debris contact; the rare pre-contact
    #: divergence is ~1e-9 global-solver dust in the 1-2 steps between
    #: release and impact (the 3 dropped rocks leave their parked floor
    #: contacts, changing the constraint set MuJoCo solves globally).
    rows.append({'obs_div_step': div, 'act_div_step': adiv,
                 'drop_step': first_drop, 'first_rock_contact': first_any,
                 'first_rock_ant_contact': first_ant,
                 'ok': (div is None or (first_drop is not None
                                        and div > first_drop))})
  return rows


def teacher_route(mask, rng):
  left_active = bool(mask[0] or mask[1])
  right_active = bool(mask[2] or mask[3])
  if left_active and right_active:
    return 'center'
  if left_active:
    return 'right'
  if right_active:
    return 'left'
  return 'left' if rng.random() < 0.5 else 'right'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--eps', type=int, default=40,
                  help='episodes per condition PER SEED')
  ap.add_argument('--pairs', type=int, default=4,
                  help='hiddenness pairs PER SEED')
  ap.add_argument('--seeds', type=int, nargs='+',
                  default=[62_001, 63_001, 64_001],
                  help='base seeds; conditions are pooled across all of '
                       'them (single-seed pilots at n<=60 flip PASS/FAIL '
                       'on ~0.05-wide margins by draw luck alone)')
  ap.add_argument('--activation-resets', type=int, default=2000,
                  help='natural resets per seed for activation frequency')
  ap.add_argument('--out', default=os.path.join(OUT_DIR, 'report.json'))
  args = ap.parse_args()
  os.makedirs(OUT_DIR, exist_ok=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(WALKER, BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  print(f'frozen walker step {wmeta.get("step")} | base step {base_step} | '
        f'seeds {args.seeds}', flush=True)

  # ---- 1. activation frequency --------------------------------------------
  all_masks = []
  for seed in args.seeds:
    env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed)
    all_masks += [(env.reset(), env.rockfall_mask)[1]
                  for _ in range(args.activation_resets)]
  masks = np.array(all_masks)
  act_freq = masks.mean(0)
  print('activation freq:', dict(zip(env.site_names,
                                     act_freq.round(4))), flush=True)

  # ---- 2. paired hiddenness -------------------------------------------------
  pairs = []
  for seed in args.seeds:
    penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg,
                             seed=seed + 1)
    pairs += paired_hiddenness(penv, walker, base_act, args.pairs)
  n_ok = sum(p['ok'] for p in pairs)
  print(f'paired hiddenness: {n_ok}/{len(pairs)} identical until the '
        'triggered rockfall (drop)', flush=True)
  for p in pairs:
    print('  ', p, flush=True)

  # ---- 3-5. policy conditions (pooled across seeds) -------------------------
  conditions = {}
  detail = {}

  def run_condition(name, seed_off, pick_route, accept=None, eps=None):
    rows = []
    skipped = 0
    per_seed = {}
    for seed in args.seeds:
      cenv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg,
                               seed=seed + seed_off)
      rng = np.random.default_rng(seed + seed_off + 5)
      seed_rows = []
      while len(seed_rows) < (eps or args.eps):
        o = cenv.reset()
        if accept is not None and not accept(cenv.rockfall_mask):
          skipped += 1
          continue
        route = pick_route(cenv.rockfall_mask, len(seed_rows), rng)
        seed_rows.append(run_route(cenv, o, walker, base_act, route))
      per_seed[seed] = round(float(np.mean(
          [r['success'] for r in seed_rows])), 3)
      rows += seed_rows
    s = summarize(rows)
    s['skipped_resets'] = skipped
    s['success_per_seed'] = per_seed
    conditions[name] = s
    detail[name] = rows
    print(f'{name:12s} success {s["success"]:.3f} (per-seed {per_seed})  '
          f'outcomes {s["outcomes"]}  '
          f'route_id {s["route_identifiable"]:.2f}', flush=True)
    return rows

  def summarize(rows):
    out = {'eps': len(rows),
           'success': float(np.mean([r['success'] for r in rows])),
           'outcomes': {k: round(float(np.mean(
               [r['outcome'] == k for r in rows])), 3)
               for k in ('success', 'collapse', 'fall', 'timeout')},
           'mean_steps': float(np.mean([r['steps'] for r in rows])),
           'route_identifiable': float(np.mean(
               [r['obs_route'] == r['route'] for r in rows])),
           'wobble_std_y': float(np.mean([r['std_y'] for r in rows])),
           'mean_abs_y': float(np.mean([abs(r['mean_y']) for r in rows]))}
    return out

  left_clear = lambda m: not (m[0] or m[1])
  right_clear = lambda m: not (m[2] or m[3])

  def pick_clear(mask, ep, rng):
    ok = [r for r, c in (('left', left_clear(mask)),
                         ('right', right_clear(mask))) if c]
    return ok[ep % len(ok)]

  def pick_hazard(mask, ep, rng):
    ok = [r for r, c in (('left', not left_clear(mask)),
                         ('right', not right_clear(mask))) if c]
    return ok[ep % len(ok)]

  run_condition('clear_side', 10, pick_clear,
                accept=lambda m: left_clear(m) or right_clear(m))
  hz = run_condition('hazard_side', 20, pick_hazard,
                     accept=lambda m: not (left_clear(m)
                                           and right_clear(m)))
  run_condition('center', 30, lambda m, e, r: 'center')
  run_condition('teacher', 40, lambda m, e, r: teacher_route(m, r))
  run_condition('blind_left', 50, lambda m, e, r: 'left')

  # ---- 4. hazardous-side breakdown -----------------------------------------
  n = len(hz)
  hit_rows = [r for r in hz if r['n_hit'] > 0]
  sev_hit = [r for r in hit_rows if 'severe' in r['sev_hits']]
  imp_hit = [r for r in hit_rows
             if 'impaired' in r['sev_hits'] and 'severe' not in r['sev_hits']]
  mild_hit = [r for r in hit_rows if set(r['sev_hits']) == {'mild'}]
  nonsev = imp_hit + mild_hit
  hazard = {
      'eps': n,
      'hit_frac': len(hit_rows) / n,
      'severe_collapse_rate': len(sev_hit) / n,
      'impaired_rate': len(imp_hit) / n,
      'mild_hit_rate': len(mild_hit) / n,
      'recovery_success': (float(np.mean([r['success'] for r in nonsev]))
                           if nonsev else None),
      'recovery_success_mild': (float(np.mean(
          [r['success'] for r in mild_hit])) if mild_hit else None),
      'recovery_success_impaired': (float(np.mean(
          [r['success'] for r in imp_hit])) if imp_hit else None),
      'lateral_escape_rate': (float(np.mean(
          [r['escaped'] for r in nonsev])) if nonsev else None),
      'outcomes': conditions['hazard_side']['outcomes']}
  print('hazard breakdown:', json.dumps(hazard, indent=2), flush=True)

  # ---- 5. route statistics ---------------------------------------------------
  center_rows = detail['center']
  route_stats = {
      name: {'route_identifiable': conditions[name]['route_identifiable'],
             'wobble_std_y': conditions[name]['wobble_std_y'],
             'mean_abs_y': conditions[name]['mean_abs_y']}
      for name in conditions}
  center_trigger_rate = float(np.mean(
      [any(r['triggered']) for r in center_rows]))
  center_drop_rate = float(np.mean(
      [any(r['dropped']) for r in center_rows]))
  route_stats['center_trigger_rate'] = center_trigger_rate
  route_stats['center_drop_rate'] = center_drop_rate
  print(f'center trigger (flag) rate: {center_trigger_rate:.3f}  '
        f'physical drop rate: {center_drop_rate:.3f}', flush=True)

  # ---- structural-gap qualification ------------------------------------------
  by = {k: conditions[k]['success'] for k in conditions}
  checks = {
      'gap_clear_ge_center_plus_0.05':
          by['clear_side'] >= by['center'] + TARGETS['gap_clear_over_center'],
      'gap_teacher_ge_center_plus_0.05':
          by['teacher'] >= by['center'] + TARGETS['gap_teacher_over_center'],
      'gap_center_ge_blind_plus_0.10':
          by['center'] >= by['blind_left'] + TARGETS['gap_center_over_blind'],
      'hazard_le_0.25': by['hazard_side'] <= TARGETS['hazard_max'],
      'center_zero_physical_rockfall': center_drop_rate == 0.0,
      'paired_hiddenness_all_ok': n_ok == len(pairs),
      'side_trigger_reliability_ge_0.85': hazard['hit_frac'] >= 0.85,
      'activation_near_0.2': bool(np.all(np.abs(act_freq
                                                - env.p_active) < 0.05))}
  clo, chi = TARGETS['center_tuning_band']
  print(f'center tuning band {clo}-{chi} (non-gating): center = '
        f'{by["center"]:.3f}', flush=True)
  for k, v in checks.items():
    print(f'{"PASS" if v else "FAIL"}  {k}', flush=True)

  report = {
      'config': RA.rockfall_config(),
      'walker': {'path': WALKER, 'step': wmeta.get('step')},
      'base_policy': {'path': BASE, 'step': base_step},
      'routes': {'lane': LANE, 'v_side': V_SIDE, 'v_center': V_CENTER,
                 'handoff_x': HANDOFF_X,
                 'center_unstick': {'window': STALL_WINDOW,
                                    'min_dx': STALL_MIN_DX,
                                    'nudge_y': NUDGE_Y,
                                    'nudge_steps': NUDGE_STEPS}},
      'seeds': args.seeds, 'eps_per_condition_per_seed': args.eps,
      'activation': {'resets': args.activation_resets,
                     'freq': {n_: float(f) for n_, f in
                              zip(env.site_names, act_freq)}},
      'paired_hiddenness': {'pairs': pairs, 'n_ok': n_ok},
      'success': by,
      'conditions': conditions,
      'hazard_breakdown': hazard,
      'route_stats': route_stats,
      'targets': TARGETS, 'checks': checks,
      'all_targets_pass': all(checks.values())}
  json.dump(report, open(args.out, 'w'), indent=2)
  json.dump(detail, open(args.out.replace('.json', '_detail.json'), 'w'))
  print('saved', args.out, '| all targets pass:',
        report['all_targets_pass'], flush=True)


if __name__ == '__main__':
  main()
