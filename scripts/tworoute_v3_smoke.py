"""Sanity gate for the V3 two-route rockfall pair (east-leg hazard). Analysis
only -- the 16-check V2 gate (scripts/tworoute_rockfall_smoke.py) ported to
the transposed geometry, plus T17 for the new goal-corner parameter.

Structural checks T1-T3 (contract, resets, latent) plus the four required
behavioural cases, per variant:

  A  rockfall clear  + shortcut -> traverses the band, NO failure, goal
  B  rockfall active + shortcut -> rocks drop, fall, contact kills, absorbing
  C  rockfall clear  + detour   -> no hazard failure, reaches the goal
  D  rockfall active + detour   -> no hazard failure, reaches the goal

Traversal is KINEMATIC (teleport along waypoints + one zero-action step per
waypoint): the hazard/route logic is position-based, so this exercises it
end-to-end without a trained controller (driver quality is gated separately
by scripts/tworoute_v3_teacher.py). Repo teleport pitfalls respected:
ant-dofs-only writes (in-flight rocks keep their velocity), qacc_warmstart
zeroed before mj_forward, fresh reset per case (absorbing flag).

What the V3 transposition changes here, and what it deliberately keeps:

  * The band moved from the west column (|x| < 2, y in [2.6, 5.4]) to the
    BOTTOM ROW (x in [2.6, 5.4], |y| < 2). Every drag that exercised the old
    band is re-aimed east along y = 0; the shortcut continues up the EAST
    column (tr) or stops at the east corner (br); the detour climbs the WEST
    column (through the opened cell (0, 4)) and crosses the top row.
  * Band WIDTH margins are predicate-level (T8): HAZARD_HALF_Y = 2.0 is the
    full corridor half-width, so every OPEN point at band x is inside the
    band and the width edge is enforced by the wall itself -- an env-level
    beyond-width probe would have to teleport into a wall block. The MOUTH
    margins (x = 2.6 / 5.4) border open corridor and are probed env-level.
  * The kill case (T5) places the ant INSIDE the band moving east
    (qvel[0] = 1.2, the axis the V3 drop aim leads along) and then steps
    with zero torques: the trigger must fire, rocks must drop and visibly
    fall, and the contact death must arrive -- WHEN is not asserted (a
    stationary zero-torque ant can die late, ~t = 46, off a grazing
    contact), so the loop budget is generous. The placement sits at the
    band's east end so the drop clip pins the pattern onto the ant and the
    death is jitter-proof (see the in-line note at T5).
  * T16 keeps the EXACT V2 pin values: the latent/jitter rng streams are
    geometry-independent (same seed offsets, same draw order), so the V3
    env must reproduce V2's draws bit-for-bit. A mismatch is a real
    regression in draw order -- report it, never re-pin.
  * T17 (new) pins the goal-corner parameter: 'tr' goals land in
    [8, 9.6] x [8, 9.6], 'br' goals in [8, 9.6] x [0, 1.6] (cell corner +
    one-sided d4rl noise +[0, 1.5] per coord), and the two variants differ.

Pre-registered reference numbers (context, embedded for the record; the
controlled pair shares its sparse refs and differs ONLY in the discounted
incentive): sparse always-shortcut 0.70 / always-detour 0.96 / oracle 0.988
for BOTH variants; discounted gamma=0.99: tr shortcut 0.146, detour 0.185
(best blind), oracle 0.201; br shortcut 0.323 (best blind), detour 0.100,
oracle 0.353.

Writes artifacts/tworoute_rockfall_v3/<variant>/smoke_report.json per
variant; exits 0 iff ALL selected variants PASS every check.

Usage: python scripts/tworoute_v3_smoke.py [--variant tr|br|both]
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import tworoute_rockfall_ant as TR  # noqa: E402
from crl import tworoute_rockfall_v3 as V3   # noqa: E402
from crl.rockfall_ant import NQ_ANT, NV_ANT  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v3'
SEED = 12_345
HORIZON = 400          #: V3 protocol horizon (longest driver ~225 steps)

#: kinematic waypoints (world frame; R cell at origin, band on the bottom
#: row x in [2.6, 5.4]). EAST_LEG mirrors V2's SHORTCUT_WAY with the axes
#: swapped; indices matter: [2] = (3.2, 0) is the first in-band waypoint.
EAST_LEG = [(1.2, 0.0), (2.2, 0.0), (3.2, 0.0), (4.4, 0.0), (5.8, 0.0),
            (7.0, 0.0)]
#: west column (through the opened cell (0, 4)) then the top row; index [2]
#: = (0, 6.5) is where the detour label must fire (y >= 6, x < 2).
WEST_TOP = [(0.0, 2.0), (0.0, 4.0), (0.0, 6.5), (0.0, 8.0), (2.5, 8.0),
            (5.0, 8.0)]
SHORTCUT_WAY = {'tr': EAST_LEG + [(8.0, 2.5), (8.0, 5.0), (8.0, 7.0)],
                'br': list(EAST_LEG)}
DETOUR_WAY = {'tr': WEST_TOP + [(7.0, 8.0)],
              'br': WEST_TOP + [(8.0, 8.0), (8.0, 5.5), (8.0, 3.0),
                                (8.0, 1.0)]}
#: per-variant goal box: cell corner + one-sided d4rl noise (+[0, 1.5]).
GOAL_BOX = {'tr': ((8.0, 9.6), (8.0, 9.6)), 'br': ((8.0, 9.6), (0.0, 1.6))}
GOAL_CELL_XY = {'tr': (8.0, 8.0), 'br': (8.0, 0.0)}

INFO_KEYS = ('failure', 'entered_hazard', 'route', 'success',
             'rock_triggered', 'rock_dropped', 'rock_contact')


def env_name(variant):
  return f'offline_ant_umaze_tworoute_rockfall_v3{variant}'


def _yaw_deg(quat):
  """Torso yaw in degrees from the (w, x, y, z) obs slice (unnormalised:
  INIT_QPOS carries +-0.1 reset noise, so normalise first)."""
  q = np.asarray(quat, float)
  w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
  return float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                     1 - 2 * (y * y + z * z))))


def fresh_env(variant, seed=SEED, **cfg_over):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  for k, v in cfg_over.items():
    setattr(cfg, k, v)
  return envs_mod.make_env(env_name(variant), cfg, seed=seed), cfg


def teleport(env, xy):
  """Place the torso at xy (pose kept), repo-canonical warmstart hygiene.
  Touches ANT dofs only -- in-flight rocks must keep their velocity."""
  d = env._env.data
  d.qpos[0], d.qpos[1] = float(xy[0]), float(xy[1])
  d.qvel[:NV_ANT] = 0.0
  d.qacc_warmstart[:] = 0.0
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()


def place_moving(env, xy, vx):
  """teleport() + an eastward torso velocity: the V3 trigger aim leads along
  qvel[0], so the kill case must present a realistic band crossing."""
  teleport(env, xy)
  d = env._env.data
  d.qvel[0] = float(vx)
  d.qacc_warmstart[:] = 0.0
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()


def wall_contacts(env, depth=-0.05):
  """Number of PENETRATING contacts (dist < depth) involving any block_* wall
  geom. A waypoint teleported INSIDE a wall shows up here as deep penetration,
  so the kinematic traversals cannot silently pass through re-walled geometry;
  a leg grazing a wall face (dist ~ 0) is legitimate locomotion contact and is
  not counted."""
  m, d = env._env.model, env._env.data
  n = 0
  for i in range(d.ncon):
    con = d.contact[i]
    if con.dist >= depth:
      continue
    for g in (con.geom1, con.geom2):
      nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
      if nm is not None and nm.startswith('block_'):
        n += 1
        break
  return n


def drag(env, waypoints, then_goal=True):
  """Teleport along waypoints, one zero-action step each; returns the log."""
  rows, done_at = [], None
  for i, xy in enumerate(waypoints):
    teleport(env, xy)
    o, r, done, info = env.step(np.zeros(8))
    rows.append({'xy': list(xy), 'r': float(r), 'done': bool(done),
                 'wall_contacts': wall_contacts(env),
                 **{k: info[k] for k in INFO_KEYS}})
    if done and done_at is None:
      done_at = i
  if then_goal and done_at is None:
    teleport(env, np.asarray(env._env.goal))
    o, r, done, info = env.step(np.zeros(8))
    rows.append({'xy': [float(v) for v in env._env.goal], 'r': float(r),
                 'done': bool(done),
                 'wall_contacts': wall_contacts(env),
                 **{k: info[k] for k in INFO_KEYS}})
  return rows, done_at


def run_variant(variant):
  checks, detail = {}, {}
  short_way, det_way = SHORTCUT_WAY[variant], DETOUR_WAY[variant]
  (gx0, gx1), (gy0, gy1) = GOAL_BOX[variant]

  # ---- T1: construction + learner obs contract ----------------------------
  env, cfg = fresh_env(variant)
  o0 = env.reset()
  checks['T1_contract'] = (
      o0.shape == (58,) and o0.dtype == np.float32
      and cfg.obs_dim == 29 and cfg.goal_dim == 29 and cfg.action_dim == 8
      and cfg.max_episode_steps == HORIZON     # V3 protocol horizon (400)
      and bool(np.all(o0[31:] == 0.0))           # zero-padded XY goal block
      and env._env.full_reset is True)           # canonical reset by default
  detail['T1'] = {'obs_shape': list(o0.shape), 'goal_block_zero':
                  bool(np.all(o0[31:] == 0.0)),
                  'max_episode_steps': cfg.max_episode_steps}

  # ---- T2: repeated resets (goal box is variant-specific in V3) -----------
  ok, starts, goals = True, [], []
  for _ in range(25):
    o = env.reset()
    ok &= bool(np.all(np.isfinite(o))) and o.shape == (58,)
    starts.append([float(o[0]), float(o[1])])
    goals.append([float(o[29]), float(o[30])])
  s, g = np.asarray(starts), np.asarray(goals)
  checks['T2_resets'] = (ok and float(np.abs(s).max()) < 1.0
                         and bool(np.all((g[:, 0] >= gx0) & (g[:, 0] <= gx1)))
                         and bool(np.all((g[:, 1] >= gy0) & (g[:, 1] <= gy1))))
  detail['T2'] = {'start_absmax': float(np.abs(s).max()),
                  'goal_min': g.min(0).tolist(), 'goal_max': g.max(0).tolist()}

  # ---- T3: latent -- frequency, override, invisibility --------------------
  n_active = 0
  for _ in range(400):
    env.reset()
    n_active += int(env.privileged_rockfall_active)
  freq = n_active / 400.0
  env.reset(rockfall_active=True)
  ov_true = env.privileged_rockfall_active
  env.reset(rockfall_active=False)
  ov_false = env.privileged_rockfall_active
  # paired invisibility: same seed, forced active vs inactive -> identical
  # obs streams while outside the hazard (latent not in obs; fixed rng order).
  # 10 flailing random steps move the ant well under 1 unit; the band mouth
  # is 2.6 east, so the pair cannot reach the trigger and diverge.
  eA, _ = fresh_env(variant, seed=777)
  eB, _ = fresh_env(variant, seed=777)
  oa = eA.reset(rockfall_active=True)
  ob = eB.reset(rockfall_active=False)
  same = bool(np.array_equal(oa, ob))
  rng = np.random.default_rng(3)
  for _ in range(10):
    a = rng.uniform(-1, 1, 8)
    oa = eA.step(a)[0]
    ob = eB.step(a)[0]
    same &= bool(np.array_equal(oa, ob))
  checks['T3_latent'] = (abs(freq - TR.P_ACTIVE) < 0.06
                         and ov_true is True and ov_false is False and same)
  detail['T3'] = {'freq_400': freq, 'override_true': ov_true,
                  'override_false': ov_false, 'paired_obs_identical': same}

  # ---- T4/T5/T6/T7: the four required cases -------------------------------
  env.reset(rockfall_active=False)
  rows, done_at = drag(env, short_way)
  last = rows[-1]
  checks['T4_caseA_clear_shortcut'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and last['failure'] is False and rows[2]['entered_hazard'] is True
      and last['route'] == 'shortcut'
      and rows[2]['rock_triggered'] is True     # trigger fires either latent
      and all(r['rock_dropped'] is False for r in rows)   # ...but no launch
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T4'] = rows

  # T5: active-latent kill. Drag to the band edge, then place the ant INSIDE
  # the band moving east (qvel[0] = 1.2 -- the aim's lead axis) and step with
  # zero torques: trigger fires, rocks drop and VISIBLY fall (z strictly
  # decreasing over >= 3 steps), contact kills, absorbing after done. Death
  # TIME is not asserted (a stationary zero-torque ant dies late, ~t = 46).
  # Placement x = 5.2 (in band, near the east mouth), NOT mid-band: a zero-
  # torque ant slides only ~0.14 before friction stops it, so a mid-band drop
  # lands ~0.5 ahead and the kill decays to a slow forward-lean graze that
  # the +-0.08 jitter flips (measured death at t = 26 / 46 / never across
  # pose+jitter draws -- a knife-edge no gate can stand on). At 5.2 every
  # unclipped rock target (5.85..6.63) hits the x <= 5.7 drop clip, so the
  # whole pattern lands pinned at x = 5.7, on the ant's front legs,
  # jitter-proof (death measured at t = 4 across rng burn-in states).
  env.reset(rockfall_active=True)
  rows, done_at = drag(env, EAST_LEG[:2], then_goal=False)   # edge: x <= 2.2
  place_moving(env, (5.2, 0.0), vx=1.2)
  qa = env._rock_qadr[0]
  rock_z, settle, done_settle = [], [], None
  for k in range(120):
    o_s, r_s, d_s, i_s = env.step(np.zeros(8))
    rock_z.append(round(float(env._env.data.qpos[qa + 2]), 3))
    settle.append({'done': bool(d_s), **{kk: i_s[kk] for kk in
                   ('failure', 'rock_triggered', 'rock_dropped',
                    'rock_contact', 'route', 'success')}})
    if d_s:
      done_settle = k
      break
  fail_row = settle[done_settle] if done_settle is not None else None
  # absorbing after failure: teleporting to the goal must NOT score
  frozen_ok = False
  if done_settle is not None:
    teleport(env, np.asarray(env._env.goal))
    o_f, r_f, d_f, i_f = env.step(np.zeros(8))
    frozen_ok = (r_f == 0.0 and d_f is True and i_f['failure'] is True
                 and i_f['success'] is False)
  falling = (len(rock_z) >= 4 and rock_z[0] > 1.5
             and rock_z[0] > rock_z[1] > rock_z[2] > rock_z[3])
  checks['T5_caseB_active_shortcut'] = (
      done_at is None                       # band edge alone triggers nothing
      and all(r['rock_triggered'] is False for r in rows)
      and settle[0]['rock_triggered'] is True   # in-band placement fires it
      and settle[0]['rock_dropped'] is True     # active latent -> launch
      and done_settle is not None           # rocks physically got the ant
      and fail_row['failure'] is True and fail_row['rock_contact'] is True
      and fail_row['success'] is False and fail_row['route'] == 'shortcut'
      and falling and frozen_ok)
  detail['T5'] = {'entry_rows': rows, 'settle': settle[:8],
                  'rock_z': rock_z[:12], 'done_settle': done_settle,
                  'frozen_ok': frozen_ok}

  env.reset(rockfall_active=False)
  rows, done_at = drag(env, det_way)
  last = rows[-1]
  checks['T6_caseC_clear_detour'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and all(r['entered_hazard'] is False for r in rows)
      and all(r['failure'] is False for r in rows)
      and rows[2]['route'] == 'detour'          # fires at (0, 6.5): y>=6, x<2
      and last['route'] == 'detour'
      and all(r['rock_dropped'] is False for r in rows)
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T6'] = rows

  env.reset(rockfall_active=True)
  rows, done_at = drag(env, det_way)
  last = rows[-1]
  checks['T7_caseD_active_detour'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and all(r['entered_hazard'] is False for r in rows)
      and all(r['failure'] is False for r in rows)
      and last['route'] == 'detour'
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T7'] = rows

  # ---- T8: hazard-band margins -- mouths env-level, width predicate-level.
  # The band spans the FULL corridor width (|y| < 2), so every open point at
  # band x is in the band and the width edge is a wall: probing it env-level
  # would mean teleporting inside a block. The mouths border open corridor.
  env.reset(rockfall_active=True)
  teleport(env, (2.2, 0.5))          # west mouth, just outside (x < 2.6)
  _, _, d1, i1 = env.step(np.zeros(8))
  env.reset(rockfall_active=True)
  teleport(env, (5.8, -0.5))         # east mouth, just outside (x > 5.4)
  _, _, d2, i2 = env.step(np.zeros(8))
  env.reset(rockfall_active=True)
  teleport(env, (3.0, 0.0))          # inside: must trigger (no contact yet)
  _, _, d3, i3 = env.step(np.zeros(8))
  band = env._in_band
  pred_ok = (not band(4.0, 2.0) and not band(4.0, -2.0)   # width edge = wall
             and band(4.0, 1.9) and band(2.6, 0.0) and band(5.4, 0.0)
             and not band(2.59, 0.0) and not band(5.41, 0.0))
  checks['T8_band_margins'] = (
      not d1 and not i1['entered_hazard'] and not i1['rock_triggered']
      and not d2 and not i2['entered_hazard'] and not i2['rock_triggered']
      and not d3 and i3['entered_hazard'] and i3['rock_triggered']
      and pred_ok)
  detail['T8'] = {'west_mouth': i1, 'east_mouth': i2, 'inside': i3,
                  'predicate_width_ok': pred_ok}

  # ---- T9: dynamics path -- random actions, timeout distinctness ----------
  o = env.reset(rockfall_active=False)
  ok = True
  act_rng = np.random.default_rng(9)
  for _ in range(100):
    o, r, done, info = env.step(act_rng.uniform(-1, 1, 8))
    ok &= o.shape == (58,) and bool(np.all(np.isfinite(o))) and not done
  # 20 zero-action steps at the start: neither success nor failure
  env.reset(rockfall_active=True)
  fail_any = succ_any = False
  for _ in range(20):
    _, r, done, info = env.step(np.zeros(8))
    fail_any |= info['failure']
    succ_any |= info['success']
  checks['T9_dynamics_timeout'] = ok and not fail_any and not succ_any
  detail['T9'] = {'random_ok': ok, 'idle_failure': fail_any,
                  'idle_success': succ_any}

  # ---- T10: config overrides land ----------------------------------------
  env2, _ = fresh_env(variant, rockfall_p_active=0.45, rockfall_max_steps=800)
  checks['T10_config_overrides'] = (env2.p_active == 0.45
                                    and env2.max_episode_steps == 800)
  detail['T10'] = {'p_active': env2.p_active,
                   'max_steps': env2.max_episode_steps}

  # ---- T11: geometry (model-level, teleport-proof) + goal-cell resolution -
  m = env._env.model
  gone = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'block_2_1') == -1
  kept = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'block_2_2') != -1
  n_blocks = sum(1 for i in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '')
                 .startswith('block_'))
  # physically stand INSIDE the opened cell: zero wall contacts, ant settles
  env.reset(rockfall_active=False)
  teleport(env, (0.0, 4.0))
  for _ in range(10):
    o, _, _, _ = env.step(np.zeros(8))
  open_ok = (wall_contacts(env) == 0 and abs(float(o[0])) < 1.5
             and abs(float(o[1]) - 4.0) < 1.5)
  # the goal-corner cells must resolve to the advertised world corners, and
  # this env must carry its own variant's corner.
  cells_ok = all(
      bool(np.allclose(env._cell_xy(V3.GOAL_CELLS[v]), GOAL_CELL_XY[v]))
      for v in ('tr', 'br'))
  own_ok = bool(np.allclose(env._eval_goal_cell_xy, GOAL_CELL_XY[variant]))
  checks['T11_geometry_open'] = (gone and kept and n_blocks == 17 and open_ok
                                 and cells_ok and own_ok)
  detail['T11'] = {'block_2_1_absent': gone, 'block_2_2_present': kept,
                   'n_blocks': n_blocks, 'stand_in_opened_cell_ok': open_ok,
                   'goal_cells_resolve': cells_ok,
                   'own_goal_cell_xy': [float(v)
                                        for v in env._eval_goal_cell_xy]}

  # ---- T12: route invariant -- a backtracker that first goes HIGH in the
  # west column (label 'detour') and then crosses the band must END labelled
  # 'shortcut' (band entry is authoritative), and its death carries it. -----
  env.reset(rockfall_active=True)
  rows12, done12 = drag(env, [(0.0, 2.0), (0.0, 4.0), (0.0, 6.5), (0.0, 3.0),
                              (1.2, 0.0), (3.2, 0.0)], then_goal=False)
  fr12 = None
  for k in range(20):                     # walk on east under the falling rocks
    teleport(env, (min(3.2 + 0.25 * (k + 1), 5.2), 0.0))
    _, _, d12, i12 = env.step(np.zeros(8))
    if d12:
      fr12 = i12
      break
  checks['T12_route_invariant'] = (
      done12 is None and rows12[2]['route'] == 'detour'
      and rows12[5]['route'] == 'shortcut'
      and fr12 is not None and fr12['failure'] is True
      and fr12['route'] == 'shortcut')
  detail['T12'] = {'rows': rows12, 'fail_info': fr12}

  # ---- T13: success is latched and immune to later band entry -------------
  env.reset(rockfall_active=True)
  rows13, done13 = drag(env, det_way)        # detour to the goal -> success
  teleport(env, (4.0, 0.0))                  # then wander INTO the live band
  _, r13, d13, i13 = env.step(np.zeros(8))
  checks['T13_success_latched'] = (
      done13 is None and rows13[-1]['success'] is True
      and d13 is False and i13['failure'] is False
      and i13['success'] is True and i13['entered_hazard'] is True)
  detail['T13'] = {'post_goal_band_step': i13, 'done': bool(d13)}

  # ---- T14: parked rocks are latent-independent; inactive trigger is inert
  eA2, _ = fresh_env(variant, seed=901)
  eB2, _ = fresh_env(variant, seed=901)
  eA2.reset(rockfall_active=True)
  eB2.reset(rockfall_active=False)
  same_park = bool(np.array_equal(
      np.asarray(eA2._env.data.qpos)[NQ_ANT:],
      np.asarray(eB2._env.data.qpos)[NQ_ANT:]))
  # drag the INACTIVE env into the V3 band: trigger flag fires, rocks stay put
  rock_before = np.asarray(eB2._env.data.qpos)[NQ_ANT:].copy()
  teleport(eB2, (3.2, 0.0))
  _, _, dB, iB = eB2.step(np.zeros(8))
  rock_after = np.asarray(eB2._env.data.qpos)[NQ_ANT:].copy()
  checks['T14_physical_hiddenness'] = (
      same_park and iB['rock_triggered'] is True
      and iB['rock_dropped'] is False and not dB
      and bool(np.allclose(rock_before, rock_after, atol=0.05)))  # contact settling only
  detail['T14'] = {'parked_identical': same_park,
                   'inactive_trigger': iB,
                   'rock_moved': float(np.abs(rock_after - rock_before).max())}

  # ---- T15: ONE canonical start pose, and the heading API is really gone
  # (V3 inherits reset() from the V2 class; assert it stayed coin-free). ----
  e15, _ = fresh_env(variant, seed=4242)
  yaws = []
  for _ in range(50):
    o15 = e15.reset()
    yaws.append(_yaw_deg(o15[3:7]))
  #: 15 deg is the INIT_QPOS +-0.1 reset-noise floor (measured native range
  #: +-12.6 deg) with headroom; the retired north pose sat at 77.6-102.4 deg.
  sig15 = set(inspect.signature(e15.reset).parameters)
  checks['T15_canonical_pose'] = (
      max(abs(y) for y in yaws) <= 15.0
      and not hasattr(e15, 'default_heading')
      and sig15 == {'rockfall_active'}
      and not hasattr(TR, '_Q_NORTH') and not hasattr(TR, '_quat_mul'))
  detail['T15'] = {'yaw_deg_range': [round(min(yaws), 2), round(max(yaws), 2)],
                   'reset_params': sorted(sig15),
                   'has_default_heading': hasattr(e15, 'default_heading')}

  # ---- T16: rng draw order pinned ACROSS versions -- V2 values VERBATIM.
  # The latent/jitter streams are geometry-independent (same seed offsets,
  # same fixed draw order), so the V3 env must reproduce the V2 draws
  # bit-for-bit. A mismatch is a REAL draw-order regression: never re-pin. --
  pin = {'0': [False, False, True, False, False],
         '7': [False, False, True, False, True],
         '909': [True, False, False, False, False]}
  pin_jit0 = {'0': (0.027365320232, 0.036680888082),
              '7': (0.023794493523, -0.045324846818),
              '909': (0.061394139017, 0.069811856466)}
  ok16, got16 = True, {}
  for s, want in pin.items():
    e16, _ = fresh_env(variant, seed=int(s))
    lat, j0 = [], None
    for i in range(5):
      e16.reset()
      lat.append(bool(e16.privileged_rockfall_active))
      if i == 0:
        j0 = tuple(float(v) for v in np.asarray(e16._drop_jitter)[0])
    got16[s] = {'latents': lat, 'jitter_row0': [round(v, 12) for v in j0]}
    ok16 &= (lat == want) and bool(np.allclose(j0, pin_jit0[s], atol=1e-12))
  checks['T16_rng_order_pinned'] = bool(ok16)
  detail['T16'] = {'expected_latents': pin, 'got': got16}

  # ---- T17: goal corner -- the ONLY manipulated variable of the pair ------
  other = 'br' if variant == 'tr' else 'tr'
  e17o, _ = fresh_env(other, seed=606)
  (ox0, ox1), (oy0, oy1) = GOAL_BOX[other]
  ok17, g_own, g_oth = True, [], []
  for _ in range(30):
    o_own = env.reset()
    o_oth = e17o.reset()
    ok17 &= bool(np.all(o_own[31:] == 0.0)) and bool(np.all(o_oth[31:] == 0.0))
    g_own.append([float(o_own[29]), float(o_own[30])])
    g_oth.append([float(o_oth[29]), float(o_oth[30])])
  g_own, g_oth = np.asarray(g_own), np.asarray(g_oth)
  in_own = bool(np.all((g_own[:, 0] >= gx0) & (g_own[:, 0] <= gx1)
                       & (g_own[:, 1] >= gy0) & (g_own[:, 1] <= gy1)))
  in_oth = bool(np.all((g_oth[:, 0] >= ox0) & (g_oth[:, 0] <= ox1)
                       & (g_oth[:, 1] >= oy0) & (g_oth[:, 1] <= oy1)))
  #: the boxes are y-disjoint ([8, 9.6] vs [0, 1.6]), so in-box + a per-reset
  #: y-gap > 6 pins that the two ids really carry DIFFERENT goals.
  differ = (bool(np.all(np.abs(g_own[:, 1] - g_oth[:, 1]) > 6.0))
            and not np.allclose(env._eval_goal_cell_xy,
                                e17o._eval_goal_cell_xy))
  checks['T17_goal_corner'] = ok17 and in_own and in_oth and differ
  detail['T17'] = {'own_goal_min': g_own.min(0).tolist(),
                   'own_goal_max': g_own.max(0).tolist(),
                   'other_goal_min': g_oth.min(0).tolist(),
                   'other_goal_max': g_oth.max(0).tolist(),
                   'variants_differ': differ}

  return checks, detail


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=('tr', 'br', 'both'), default='both')
  args = ap.parse_args()
  variants = ('tr', 'br') if args.variant == 'both' else (args.variant,)

  all_ok = True
  for variant in variants:
    print(f'=== variant {variant} ({env_name(variant)}) ===', flush=True)
    checks, detail = run_variant(variant)
    v_pass = all(checks.values())
    all_ok &= v_pass
    for k, v in checks.items():
      print(f'{"PASS" if v else "FAIL"}  [{variant}] {k}', flush=True)
    out_dir = os.path.join(OUT, variant)
    os.makedirs(out_dir, exist_ok=True)
    rep = {'env_id': env_name(variant), 'variant': variant, 'seed': SEED,
           'horizon': HORIZON, 'checks': checks, 'all_pass': bool(v_pass),
           'detail': detail, 'hazard_zone': V3.hazard_zone(),
           'goal_cell_xy': list(GOAL_CELL_XY[variant]),
           'p_active': TR.P_ACTIVE}
    path = os.path.join(out_dir, 'smoke_report.json')
    with open(path, 'w') as f:
      json.dump(rep, f, indent=2, default=str)
    print(('ALL PASS' if v_pass else 'FAILED') + f' -> {path}', flush=True)
  sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
  main()
