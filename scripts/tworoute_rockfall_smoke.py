"""Sanity gate for the two-route AntMaze rockfall env (V0). Analysis only.

Structural checks T1-T3 (contract, resets, latent) plus the four required
behavioural cases:

  A  rockfall clear  + shortcut -> traverses, NO failure, reaches the goal
  B  rockfall active + shortcut -> terminal failure at the hazard band
  C  rockfall clear  + detour   -> no hazard failure, reaches the goal
  D  rockfall active + detour   -> no hazard failure, reaches the goal

Traversal is KINEMATIC (teleport along waypoints + one zero-action step per
waypoint): the hazard/route logic is position-based, so this exercises it
end-to-end without needing a trained controller (teacher quality is out of
scope for V0). Random-action stepping and obs-shape checks cover the plain
dynamics path. Repo teleport pitfalls respected: qacc_warmstart zeroed before
mj_forward (solver warmstart leak), fresh reset per case (absorbing flag).

Writes artifacts/tworoute_rockfall_v0/smoke_report.json; exits 0 iff ALL PASS.

Usage: python scripts/tworoute_rockfall_smoke.py
"""
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
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
SEED = 12_345
ENV_ID = 'offline_ant_umaze_tworoute_rockfall'

#: kinematic waypoints (world frame; R cell at origin, goal cell at (0, 8)).
SHORTCUT_WAY = [(0.0, 1.2), (0.0, 2.2), (0.0, 3.2), (0.0, 4.4), (0.0, 5.8),
                (0.0, 7.0)]
DETOUR_WAY = [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0), (8.0, 2.5),
              (8.0, 5.0), (8.0, 8.0), (5.5, 8.0), (3.0, 8.0)]


def fresh_env(seed=SEED, **cfg_over):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  for k, v in cfg_over.items():
    setattr(cfg, k, v)
  return envs_mod.make_env(ENV_ID, cfg, seed=seed), cfg


def teleport(env, xy):
  """Place the torso at xy (pose kept), repo-canonical warmstart hygiene."""
  d = env._env.data
  d.qpos[0], d.qpos[1] = float(xy[0]), float(xy[1])
  d.qvel[:] = 0.0
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
                 **{k: info[k] for k in ('failure', 'entered_hazard',
                                         'route', 'success')}})
    if done and done_at is None:
      done_at = i
  if then_goal and done_at is None:
    teleport(env, np.asarray(env._env.goal))
    o, r, done, info = env.step(np.zeros(8))
    rows.append({'xy': [float(v) for v in env._env.goal], 'r': float(r),
                 'done': bool(done),
                 'wall_contacts': wall_contacts(env),
                 **{k: info[k] for k in ('failure', 'entered_hazard',
                                         'route', 'success')}})
  return rows, done_at


def main():
  os.makedirs(OUT, exist_ok=True)
  checks, detail = {}, {}

  # ---- T1: construction + learner obs contract ----------------------------
  env, cfg = fresh_env()
  o0 = env.reset()
  base_env, _ = fresh_env()  # noqa: F841  (same builder path works twice)
  checks['T1_contract'] = (
      o0.shape == (58,) and o0.dtype == np.float32
      and cfg.obs_dim == 29 and cfg.goal_dim == 29 and cfg.action_dim == 8
      and cfg.max_episode_steps == 700
      and bool(np.all(o0[31:] == 0.0))           # zero-padded XY goal block
      and env._env.full_reset is True)           # canonical reset by default
  detail['T1'] = {'obs_shape': list(o0.shape), 'goal_block_zero':
                  bool(np.all(o0[31:] == 0.0))}

  # ---- T2: repeated resets ------------------------------------------------
  ok, starts, goals = True, [], []
  for _ in range(25):
    o = env.reset()
    ok &= bool(np.all(np.isfinite(o))) and o.shape == (58,)
    starts.append([float(o[0]), float(o[1])])
    goals.append([float(o[29]), float(o[30])])
  s, g = np.asarray(starts), np.asarray(goals)
  checks['T2_resets'] = (ok and float(np.abs(s).max()) < 1.0
                         and bool(np.all((g[:, 0] >= 0) & (g[:, 0] <= 1.6)))
                         and bool(np.all((g[:, 1] >= 8) & (g[:, 1] <= 9.6))))
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
  # obs streams while outside the hazard (latent not in obs; fixed rng order)
  eA, _ = fresh_env(seed=777)
  eB, _ = fresh_env(seed=777)
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
  rows, done_at = drag(env, SHORTCUT_WAY)
  last = rows[-1]
  checks['T4_caseA_clear_shortcut'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and last['failure'] is False and rows[2]['entered_hazard'] is True
      and last['route'] == 'shortcut'
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T4'] = rows

  env.reset(rockfall_active=True)
  rows, done_at = drag(env, SHORTCUT_WAY)
  fail_row = rows[done_at] if done_at is not None else None
  # absorbing after failure: teleporting to the goal must NOT score
  frozen_ok = False
  if done_at is not None:
    teleport(env, np.asarray(env._env.goal))
    o_f, r_f, d_f, i_f = env.step(np.zeros(8))
    frozen_ok = (r_f == 0.0 and d_f is True and i_f['failure'] is True
                 and i_f['success'] is False)
  checks['T5_caseB_active_shortcut'] = (
      done_at == 2                                     # first in-band waypoint
      and fail_row['failure'] is True and fail_row['done'] is True
      and fail_row['r'] == 0.0 and fail_row['success'] is False
      and fail_row['route'] == 'shortcut' and frozen_ok)
  detail['T5'] = {'rows': rows, 'done_at': done_at, 'frozen_ok': frozen_ok}

  env.reset(rockfall_active=False)
  rows, done_at = drag(env, DETOUR_WAY)
  last = rows[-1]
  checks['T6_caseC_clear_detour'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and all(r['entered_hazard'] is False for r in rows)
      and all(r['failure'] is False for r in rows)
      and last['route'] == 'detour'
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T6'] = rows

  env.reset(rockfall_active=True)
  rows, done_at = drag(env, DETOUR_WAY)
  last = rows[-1]
  checks['T7_caseD_active_detour'] = (
      done_at is None and last['r'] == 1.0 and last['success'] is True
      and all(r['entered_hazard'] is False for r in rows)
      and all(r['failure'] is False for r in rows)
      and last['route'] == 'detour'
      and all(r['wall_contacts'] == 0 for r in rows))
  detail['T7'] = rows

  # ---- T8: hazard-band margins (junction wander must not count) -----------
  env.reset(rockfall_active=True)
  teleport(env, (0.5, 2.2))          # shortcut mouth, below the band
  _, _, d1, i1 = env.step(np.zeros(8))
  env.reset(rockfall_active=True)
  teleport(env, (1.0, 6.5))          # goal-row junction, above the band
  _, _, d2, i2 = env.step(np.zeros(8))
  checks['T8_band_margins'] = (not d1 and not i1['entered_hazard']
                               and not d2 and not i2['entered_hazard'])
  detail['T8'] = {'below_band': i1, 'above_band': i2}

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
  env2, _ = fresh_env(rockfall_p_active=0.45, rockfall_max_steps=800)
  checks['T10_config_overrides'] = (env2.p_active == 0.45
                                    and env2.max_episode_steps == 800)
  detail['T10'] = {'p_active': env2.p_active,
                   'max_steps': env2.max_episode_steps}

  # ---- T11: the geometry actually changed (model-level, teleport-proof) ---
  m = env._env.model
  gone = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'block_2_1') == -1
  kept = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'block_2_2') != -1
  n_blocks = sum(1 for i in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '')
                 .startswith('block_'))
  # physically stand INSIDE the opened cell: zero wall contacts, ant settles
  env.reset(rockfall_active=False)
  teleport(env, (0.0, 4.0))
  open_ok = True
  for _ in range(10):
    o, _, _, _ = env.step(np.zeros(8))
  open_ok = (wall_contacts(env) == 0 and abs(float(o[0])) < 1.5
             and abs(float(o[1]) - 4.0) < 1.5)
  checks['T11_geometry_open'] = (gone and kept and n_blocks == 17 and open_ok)
  detail['T11'] = {'block_2_1_absent': gone, 'block_2_2_present': kept,
                   'n_blocks': n_blocks, 'stand_in_opened_cell_ok': open_ok}

  # ---- T12: route invariant -- backtracker dies as 'shortcut', never
  # 'detour' (band entry is authoritative for the label) --------------------
  env.reset(rockfall_active=True)
  rows12, done12 = drag(env, [(4.0, 0.0), (6.5, 0.0), (4.0, 0.0), (0.0, 1.2),
                              (0.0, 3.2)], then_goal=False)
  fr = rows12[done12] if done12 is not None else None
  checks['T12_route_invariant'] = (
      done12 == 4 and fr['failure'] is True and fr['route'] == 'shortcut'
      and rows12[1]['route'] == 'detour')
  detail['T12'] = rows12

  # ---- T13: success is latched and immune to later band entry -------------
  env.reset(rockfall_active=True)
  rows13, done13 = drag(env, DETOUR_WAY)     # detour to the goal -> success
  teleport(env, (0.0, 4.0))                  # then wander INTO the live band
  _, r13, d13, i13 = env.step(np.zeros(8))
  checks['T13_success_latched'] = (
      done13 is None and rows13[-1]['success'] is True
      and d13 is False and i13['failure'] is False
      and i13['success'] is True and i13['entered_hazard'] is True)
  detail['T13'] = {'post_goal_band_step': i13, 'done': bool(d13)}

  all_pass = all(checks.values())
  for k, v in checks.items():
    print(f'{"PASS" if v else "FAIL"}  {k}', flush=True)
  rep = {'env_id': ENV_ID, 'seed': SEED, 'checks': checks,
         'all_pass': bool(all_pass), 'detail': detail,
         'hazard_zone': TR.hazard_zone(), 'p_active': TR.P_ACTIVE}
  path = os.path.join(OUT, 'smoke_report.json')
  with open(path, 'w') as f:
    json.dump(rep, f, indent=2, default=str)
  print(('ALL PASS' if all_pass else 'FAILED') + f' -> {path}', flush=True)
  sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
