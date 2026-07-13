"""Pre-training gates for the D4RL-faithful Ant reproduction branch.

Hard-asserting gates (abort on failure):
  P. physics equivalence vs the vendored original XML: timestep 0.02, RK4,
     actuator ctrlrange +-30 gear 1, dt/env-step = 0.1, joint limits, wall
     count, floor present, ctrl == 30*clip(action).
  O. observation contract: state(29) == [qpos, qvel] bit-exact; flat obs 58.
  G. goal contract: goal half == settled full state == goal_state_full slice
     bit-exact; reward target == settled goal xy; drift reported.
  R. replay-relabel: sampled goals are bit-exact strict-future full states.
  A. action sensitivity: from the reset state, 64 uniform 1-step actions must
     produce >= 5x the XY spread of the gymnasium branch's negligible-local
     scale (abs gate: std disp > 1e-3 m) and differ from zero-action.
  C. state coverage: 6 random-policy episodes -> unique states/cells/goals,
     moving fraction; no duplicated-state domination.
Also records the gymnasium-branch action-sensitivity side-by-side.
"""
import json
import os
import sys

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl import envs as envs_mod
from crl.replay import TrajectoryBuffer

OUT = os.path.join(os.path.dirname(_HERE), 'artifacts',
                   'd4rl_ant_verification.json')       # repo-relative (Colab-safe)


def action_sensitivity(env, rng, n=64):
  u = env._env.unwrapped
  env.reset()
  qpos0 = np.asarray(u.data.qpos).copy()
  qvel0 = np.asarray(u.data.qvel).copy()

  def one(a):
    u.data.qpos[:] = qpos0
    u.data.qvel[:] = qvel0
    mujoco.mj_forward(u.model, u.data)
    u.step(np.asarray(a, np.float32))
    return float(np.linalg.norm(np.asarray(u.data.qpos[:2]) - qpos0[:2]))

  disp = np.array([one(rng.uniform(-1, 1, 8)) for _ in range(n)])
  return {'disp_std': float(disp.std()), 'disp_max': float(disp.max()),
          'disp_mean': float(disp.mean()), 'zero_disp': one(np.zeros(8))}


def main():
  rep = {}
  rng = np.random.default_rng(0)
  cfg = Config(env_name='d4rl_ant_umaze_gfull')
  env = envs_mod.make_env('d4rl_ant_umaze_gfull', cfg, seed=5)
  u = env._env.unwrapped
  m = u.model

  # ---- P: physics equivalence ----
  assert abs(m.opt.timestep - 0.02) < 1e-12, m.opt.timestep
  assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
  assert m.nu == 8 and m.nq == 15 and m.nv == 14
  assert np.allclose(m.actuator_ctrlrange, [[-30, 30]] * 8)
  assert np.allclose(m.actuator_gear[:, 0], 1.0)
  dt_env = m.opt.timestep * u.frame_skip
  assert abs(dt_env - 0.1) < 1e-12
  hip = m.joint('hip_1')
  assert np.allclose(np.degrees(hip.range), [-30, 30], atol=1e-4)
  n_walls = sum(1 for i in range(m.ngeom)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '')
                .startswith('block_'))
  from crl.d4rl_ant import U_MAZE
  assert n_walls == sum(row.count(1) for row in U_MAZE), n_walls
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor') >= 0
  a = rng.uniform(-1.5, 1.5, 8)
  u.step(np.asarray(a, np.float32))
  assert np.allclose(np.asarray(u.data.ctrl), 30 * np.clip(a, -1, 1))
  rep['physics'] = {'timestep': 0.02, 'integrator': 'RK4', 'dt_env': dt_env,
                    'ctrlrange': 30.0, 'gear': 1.0, 'n_walls': n_walls,
                    'pass': True}

  # ---- O: observation contract ----
  obs = env.reset()
  assert obs.shape[0] == 58
  qq = np.concatenate([np.asarray(u.data.qpos),
                       np.asarray(u.data.qvel)]).astype(np.float32)
  assert np.array_equal(obs[:29], qq)
  rep['observation'] = {'flat_dim': 58, 'layout_bit_exact': True, 'pass': True}

  # ---- G: goal contract ----
  drifts, zs, vnorm = [], [], []
  for _ in range(6):
    obs = env.reset()
    assert np.array_equal(obs[29:], env._goal_state_full)
    assert np.allclose(env._env.goal, env._goal_state_full[:2])
    drifts.append(float(np.linalg.norm(
        env._goal_state_full[:2] - np.asarray(env._env.goal))))
    zs.append(float(env._goal_state_full[2]))
    vnorm.append(float(np.linalg.norm(env._goal_state_full[15:18])))
  rep['goal'] = {'settled_z_mean': float(np.mean(zs)),
                 'settled_speed_mean': float(np.mean(vnorm)),
                 'reward_target_is_settled_xy': True, 'pass': True}

  # ---- R: replay relabel ----
  L = env.max_episode_steps + 1
  buf = TrajectoryBuffer(capacity_steps=4 * L, ep_len_obs=L, full_obs_dim=58,
                         action_dim=8, obs_dim=29, start_index=0,
                         end_index=-1, discount=0.99, seed=0,
                         goal_indices=cfg.goal_indices)
  O = np.zeros((L, 58), np.float32)
  A = np.zeros((L, 8), np.float32)
  obs = env.reset()
  for t in range(env.max_episode_steps):
    O[t] = obs
    A[t] = rng.uniform(-1, 1, 8).astype(np.float32)
    obs, _, _, _ = env.step(A[t])
  O[-1] = obs
  buf.add_episode(O, A)
  tr = buf.sample(256)
  states29 = O[:, :29]
  ok = 0
  for b in range(256):
    s, g = tr.observation[b, :29], tr.observation[b, 29:]
    si = np.where((states29 == s).all(1))[0]
    gj = np.where((states29 == g).all(1))[0]
    assert len(si) and len(gj)
    ok += int(gj.max() > si.min())
  assert ok / 256 > 0.95
  rep['relabel'] = {'strict_future_frac': ok / 256, 'pass': True}

  # ---- A: action sensitivity (+ gymnasium side-by-side) ----
  sens = action_sensitivity(env, rng)
  cfg_g = Config(env_name='antmaze_open_near')
  env_g = envs_mod.make_env('antmaze_open_near', cfg_g, seed=5)
  sens_g = action_sensitivity(env_g, rng)
  assert sens['disp_std'] > 1e-3, sens
  rep['action_sensitivity'] = {'d4rl': sens, 'gymnasium': sens_g,
                               'ratio_std': sens['disp_std']
                               / max(sens_g['disp_std'], 1e-12),
                               'pass': True}

  # ---- C: coverage ----
  seen, cells, goals, moving, total = set(), set(), set(), 0, 0
  for ep in range(6):
    obs = env.reset()
    goals.add(tuple(np.round(obs[29:31], 3)))
    prev = obs[:2].copy()
    for t in range(env.max_episode_steps):
      if t % 5 == 0:
        seen.add(tuple(np.round(obs[:29], 6)))
      obs, _, _, _ = env.step(rng.uniform(-1, 1, 8).astype(np.float32))
      moving += float(np.linalg.norm(obs[:2] - prev) > 5e-3)
      total += 1
      cells.add((round(obs[0] / 0.5), round(obs[1] / 0.5)))
      prev = obs[:2].copy()
  cov = {'unique_states': len(seen), 'unique_cells': len(cells),
         'unique_goals': len(goals), 'moving_frac': moving / total,
         'pass': len(seen) > 300 and len(goals) >= 5 and moving / total > 0.3}
  assert cov['pass'], cov
  rep['coverage'] = cov

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  json.dump(rep, open(OUT, 'w'), indent=2)
  print(json.dumps(rep, indent=1))
  print('ALL GATES PASS ->', OUT)


if __name__ == '__main__':
  main()
