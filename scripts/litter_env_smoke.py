"""Stage-0 smoke checks for offline_ant_umaze_litter (structural only).

NOT the Stage-1 geometry gate (no locomotion here). Verifies:
  S1  obs contract: 58-dim, identical to offline_ant_umaze; action_dim 8.
  S2  U sampling: ~Bernoulli(0.5) over resets; active pile up, inactive
      pile buried; u_side override works.
  S3  rubble layout frozen: identical across env instances and seeds.
  S4  U independence plumbing: same env seed => identical reset qpos/qvel
      stream regardless of forced u_side (U rng is a separate stream).
  S5  collision machinery: ant teleported into the pile band contacts the
      ACTIVE pile only; ant dropped on the middle strip contacts rubble.
  S6  step() info sidecar fields present; obs itself carries no litter dims.

Writes artifacts/litter_env/smoke_report.json and prints PASS/FAIL lines.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
from crl.config import Config  # noqa: E402
from crl.envs import make_env  # noqa: E402
from crl.d4rl_ant import (LITTER_HIDE_Z, LITTER_PILE_Y, LITTER_RUBBLE_Y,  # noqa: E402
                          LITTER_ZONE_X)

OUT_DIR = os.path.join('artifacts', 'litter_env')


def _pile_z(env, u):
  return float(env._env.model.geom_pos[env._pile_gid[u]][2])


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  checks = {}

  cfg = Config()
  cfg.env_name = 'offline_ant_umaze_litter'
  env = make_env('offline_ant_umaze_litter', cfg, seed=0)
  base_cfg = Config()
  base = make_env('offline_ant_umaze', base_cfg, seed=0)

  # S1: obs contract unchanged.
  obs = env.reset()
  checks['S1_obs_contract'] = (
      obs.shape == (58,) and obs.dtype == np.float32
      and env.obs_dim == base.obs_dim == 29
      and env.goal_dim == base.goal_dim == 29
      and env.action_dim == 8
      and env.goal_indices == base.goal_indices)

  # S2: U sampling + pile placement over resets.
  us, placement_ok = [], True
  for _ in range(200):
    env.reset()
    us.append(env.u_side)
    placement_ok &= (_pile_z(env, env.u_side) > 0.0
                     and _pile_z(env, 1 - env.u_side) <= LITTER_HIDE_Z + 1e-9)
  rate = float(np.mean(us))
  env.reset(u_side=1)
  override_ok = env.u_side == 1 and _pile_z(env, 1) > 0.0
  env.reset(u_side=0)
  override_ok &= env.u_side == 0 and _pile_z(env, 1) <= LITTER_HIDE_Z + 1e-9
  checks['S2_u_sampling'] = bool(0.4 <= rate <= 0.6 and placement_ok
                                 and override_ok)

  # S3: rubble layout frozen across instances/seeds.
  cfg2 = Config()
  env2 = make_env('offline_ant_umaze_litter', cfg2, seed=123)
  same_layout = env.rubble_layout == env2.rubble_layout
  same_geoms = all(
      np.allclose(env.rubble_layout[i]['x'],
                  env2._env.model.geom_pos[g][0])
      for i, g in enumerate(sorted(env2._rubble_gids)))
  checks['S3_rubble_frozen'] = bool(same_layout and same_geoms)

  # S4: forced U does not perturb the reset-noise/goal rng streams.
  a = make_env('offline_ant_umaze_litter', Config(), seed=7)
  b = make_env('offline_ant_umaze_litter', Config(), seed=7)
  ok4 = True
  for u_a, u_b in [(0, 1), (1, 0), (0, 0)]:
    oa, ob = a.reset(u_side=u_a), b.reset(u_side=u_b)
    ok4 &= bool(np.array_equal(oa, ob))
  checks['S4_u_independent_streams'] = ok4

  # S5: collision machinery (teleport probes; zero-action settling steps).
  # A standing ant's feet can land in the gaps between rubble boxes, so the
  # rubble probe drapes the ant LOW (z=0.2 < torso-bottom clearance) directly
  # onto a box; walking-over-rubble dynamics are Stage 1's job, not S5's.
  def contacts_at(env, xy, steps=5, z=None):
    u = env._env
    u.data.qpos[:2] = xy
    if z is not None:
      u.data.qpos[2] = z
    u.data.qvel[:] = 0.0
    mujoco.mj_forward(u.model, u.data)
    pile = rubble = 0
    for _ in range(steps):
      env.step(np.zeros(8))
      p, r = env._count_litter_contacts()
      pile += p
      rubble += r
    return pile, rubble

  zone_cx = 0.5 * (LITTER_ZONE_X[0] + LITTER_ZONE_X[1])
  pile_cy = 0.5 * (LITTER_PILE_Y[0] + LITTER_PILE_Y[1])
  env.reset(u_side=1)                    # pile on +y
  pile_hits, _ = contacts_at(env, (zone_cx, pile_cy))
  env.reset(u_side=0)                    # pile on -y; +y band must be CLEAR
  ghost_hits, _ = contacts_at(env, (zone_cx, pile_cy))
  env.reset(u_side=1)
  box = env.rubble_layout[0]
  _, rubble_hits = contacts_at(env, (box['x'], box['y']), steps=10, z=0.2)
  checks['S5_collisions'] = bool(pile_hits > 0 and ghost_hits == 0
                                 and rubble_hits > 0)

  # S6: info sidecar fields; obs blind to litter.
  env.reset()
  _, _, _, info = env.step(np.zeros(8))
  need = {'u_side', 'pile_contacts', 'rubble_contacts', 'in_zone'}
  checks['S6_info_sidecar'] = bool(need <= set(info)
                                   and info['u_side'] == env.u_side)

  report = {'checks': {k: bool(v) for k, v in checks.items()},
            'u_rate_200_resets': rate,
            'probe_counts': {'pile_hits': pile_hits, 'ghost_hits': ghost_hits,
                             'rubble_hits': rubble_hits},
            'rubble_layout': env.rubble_layout,
            'geometry': {'zone_x': LITTER_ZONE_X, 'pile_y': LITTER_PILE_Y,
                         'rubble_half_width': LITTER_RUBBLE_Y},
            'all_pass': all(checks.values())}
  path = os.path.join(OUT_DIR, 'smoke_report.json')
  with open(path, 'w') as f:
    json.dump(report, f, indent=2)

  for k, v in checks.items():
    print(f'{"PASS" if v else "FAIL"}  {k}')
  print(f'{"ALL PASS" if report["all_pass"] else "SMOKE FAILED"} -> {path}')
  return 0 if report['all_pass'] else 1


if __name__ == '__main__':
  sys.exit(main())
