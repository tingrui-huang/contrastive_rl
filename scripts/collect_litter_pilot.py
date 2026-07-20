"""Stage 3A: collect a small, fully audited PILOT dataset (no full collection,
no training). Uses the FROZEN litter env + walker + base policy + teacher
selector exactly. Learner npz keeps the 58-dim contract; every privileged /
diagnostic field goes to a SEPARATE sidecar that is never learner input.

Run:  python scripts/collect_litter_pilot.py [--episodes 200] [--smoke N]
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

from crl import envs as envs_mod          # noqa: E402
from crl import probe as P                # noqa: E402
from crl import offline_audit as OA       # noqa: E402
import walker_gate as WG                  # noqa: E402
import litter_pilot_common as C           # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

L = 701
HORIZON = 700
# Brand-new seeds for the mixture pilot. env_seed 12_450_067 was selected by
# pre-screening the env's independent u_side draw stream
# (default_rng(seed+20260719).integers(2,size=200) -> frac_u1=0.505), which
# only previews the U sequence and does not touch physics/policy.
ENV_SEED = 12_450_067         # balanced; brand-new
DATASET_SEED = 14_760_053     # dedicated mixture-assignment RNG (independent)
# every seed consumed by ANY prior development / gate / teacher / pilot run
CONSUMED = [311, 500, 622, 777, 888, 999, 1234,
            8_150_023, 5_090_023, 9_271_033, 6_330_047,
            12_450_067, 14_760_053]        # + Stage-3A mixture-pilot seeds
# Stage-3 dataset mixture (user-approved 2026-07-20), EXACT per-episode counts:
MIX = {'sighted': 0.85, 'blind': 0.05, 'coverage': 0.10}
EPSILON = 0.05                # = blind fraction (confounding component)
TEACHER_BLIND_V = WG.SLOW_V   # 0.6, frozen qualified epsilon-blind speed
COVERAGE_MIDDLE_SLOW_V = 0.8  # deliberate robust-coverage component speed
OUT_ROOT = 'artifacts/litter_dataset'


def rollout(env, walker, base_act, mode):
  """One FROZEN episode for the given mixture mode, run to the fixed horizon
  (truncated only on absorbing collapse). mode in {'sighted','blind',
  'coverage'}. Returns (obs[L,58], act[L,8], valid_len, sidecar, ep)."""
  o = env.reset()
  u = int(env.u_side)
  u_indep = mode in ('blind', 'coverage')   # middle-lane cautious policies
  if u_indep:
    # identical control law (middle lane + unstick); only the speed differs.
    y_ref, v_ref = 0.0, (TEACHER_BLIND_V if mode == 'blind'
                         else COVERAGE_MIDDLE_SLOW_V)
  else:
    clean = -1.0 if u == 1 else 1.0
    y_ref, v_ref = clean * WG.LANE, P.V_FAST
  true_goal = o[29:31].copy()

  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  sc = {k: np.full(L, np.nan, np.float32) for k in
        ('lane_cmd', 'speed_cmd', 'torso_x', 'torso_y', 'vx', 'lateral_err',
         'hforce', 'pre_speed')}
  sc.update({k: np.zeros(L, np.float32) for k in
             ('handoff', 'pile_contacts', 'rubble_contacts', 'dead')})

  handoff = False
  x_hist, nudge_until, nudge_sign = [], -1, 1.0
  dead_at, hit = -1, 0.0
  valid_len = L
  for t in range(HORIZON):
    xy = o[:2]
    if not handoff and (xy[0] >= WG.HANDOFF_X or xy[1] >= 2.0):
      handoff = True
    if handoff:
      o_cmd = o.copy()
      o_cmd[29:] = 0.0
      o_cmd[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(o_cmd[None]))[0])
      lane_cmd = speed_cmd = np.nan
    else:
      y_cmd, v_cmd = y_ref, v_ref
      if u_indep:                           # unstick probe, verbatim
        x_hist.append(float(xy[0]))
        if t < nudge_until:
          y_cmd = nudge_sign * WG.NUDGE_Y
        elif (len(x_hist) > WG.STALL_WINDOW
              and x_hist[-1] - x_hist[-WG.STALL_WINDOW] < WG.STALL_MIN_DX):
          nudge_until = t + WG.NUDGE_STEPS
          nudge_sign = -nudge_sign
          x_hist.clear()
          y_cmd = nudge_sign * WG.NUDGE_Y
      a = walker(o, y_cmd, v_cmd)
      lane_cmd, speed_cmd = float(y_cmd), float(v_cmd)
    o2, r, _, info = env.step(a)
    act[t] = a
    obs[t + 1] = o2
    qv = env._env.data.qvel
    sc['handoff'][t] = float(handoff)
    sc['lane_cmd'][t] = lane_cmd
    sc['speed_cmd'][t] = speed_cmd
    sc['torso_x'][t] = float(o2[0])
    sc['torso_y'][t] = float(o2[1])
    sc['vx'][t] = float(qv[0])
    sc['lateral_err'][t] = (np.nan if handoff else float(o2[1] - lane_cmd))
    sc['pile_contacts'][t] = float(info.get('pile_contacts', 0))
    sc['rubble_contacts'][t] = float(info.get('rubble_contacts', 0))
    sc['hforce'][t] = float(info.get('max_horizontal_normal_force', 0.0))
    sc['pre_speed'][t] = float(info.get('precontact_planar_speed', 0.0))
    sc['dead'][t] = float(bool(info.get('dead')))
    hit = max(hit, float(r))
    if info.get('dead') and dead_at < 0:
      dead_at = t
    if dead_at >= 0:                        # keep the first buried obs, stop
      valid_len = min(dead_at + 2, L)
      break
    o = o2                                  # advance the observation (closed loop)
  ep = {'u_side': u, 'blind': bool(mode == 'blind'),
        'u_independent': bool(u_indep),
        'active_pile_side': 'pos' if u == 1 else 'neg',
        'teacher_mode': mode, 'speed_cmd_nominal': float(v_ref),
        'epsilon_override': bool(mode == 'blind'),
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'ep_length': int(valid_len),
        'final_goal_dist': float(np.linalg.norm(obs[valid_len - 1, :2]
                                                - true_goal)),
        'goal_xy': true_goal.astype(np.float32)}
  return obs, act, valid_len, sc, ep


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--smoke', type=int, default=0)
  ap.add_argument('--out', default=None)
  ap.add_argument('--env-seed', type=int, default=ENV_SEED)
  ap.add_argument('--dataset-seed', type=int, default=DATASET_SEED)
  ap.add_argument('--name', default='antmaze_litter_pilot',
                  help='npz basename (learner + _sidecar)')
  args = ap.parse_args()
  n = args.smoke or args.episodes
  env_seed, dataset_seed, name = args.env_seed, args.dataset_seed, args.name

  # --- A1 gate BEFORE any collection ---
  hard_ok, disc, info = C.check_frozen_integrity()
  clash = C.seed_reuse(CONSUMED, [env_seed], [dataset_seed])
  print('A1 frozen integrity: hard_ok =', hard_ok, '| seed clash =', clash)
  for d in disc:
    print('   discrepancy:', json.dumps(d))
  if not hard_ok or clash:
    print('ABORT: substantive frozen-integrity failure or seed reuse.')
    return 2

  # versioned output dir; never overwrite an existing pilot
  out = args.out or os.path.join(OUT_ROOT, 'smoke' if args.smoke else 'pilot')
  if not args.smoke and os.path.exists(os.path.join(out,
                                                    f'{name}.npz')):
    v = 2
    while os.path.exists(out + f'_v{v}'):
      v += 1
    out = out + f'_v{v}'
    print('existing pilot found; using versioned dir', out)
  os.makedirs(out, exist_ok=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(
      'artifacts/walker/phase1/walker_best.pkl',
      'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl')
  # litter env; empty offline_dataset so make_env does not load the 133MB npz
  # (eval goals come from the frozen d4rl goal sampler, not the dataset).
  cfg.offline_dataset = ''
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=env_seed)
  assert env.collapse_force == 80.0 and env.collapse_speed == 1.2
  dataset_rng = np.random.default_rng(dataset_seed)   # independent stream
  # EXACT-count mixture, assigned to episode slots and shuffled by the
  # dataset RNG (independent of the env's U stream -> mode is U-independent).
  n_blind = int(round(MIX['blind'] * n))
  n_cover = int(round(MIX['coverage'] * n))
  n_sight = n - n_blind - n_cover
  modes = np.array(['sighted'] * n_sight + ['blind'] * n_blind
                   + ['coverage'] * n_cover)
  dataset_rng.shuffle(modes)
  print(f'mixture: sighted={n_sight} blind={n_blind} coverage={n_cover}')

  obs_all = np.zeros((n, L, 58), np.float32)
  act_all = np.zeros((n, L, 8), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  step_keys = ('handoff', 'lane_cmd', 'speed_cmd', 'torso_x', 'torso_y', 'vx',
               'lateral_err', 'pile_contacts', 'rubble_contacts', 'hforce',
               'pre_speed', 'dead')
  step_side = {k: np.zeros((n, L), np.float32) for k in step_keys}
  ep_rows = []
  for e in range(n):
    obs, act, vlen, sc, ep = rollout(env, walker, base_act, str(modes[e]))
    obs_all[e] = obs
    act_all[e] = act
    lengths[e] = vlen
    eval_goals[e] = ep['goal_xy']
    for k in step_keys:
      step_side[k][e] = sc[k]
    ep['episode_id'] = e
    ep['collection_seed'] = env_seed
    ep_rows.append(ep)
    if (e + 1) % 25 == 0:
      print(f'  collected {e + 1}/{n}', flush=True)

  meta = {'env_name': 'offline_ant_umaze_litter', 'obs_dim': 29,
          'goal_dim': 29, 'action_dim': 8, 'ep_len_obs': L,
          'start_index': 0, 'end_index': -1, 'goal_indices': list(range(29)),
          'note': 'Stage-3A litter pilot; learner keys are obs/act only.'}

  npz_path = os.path.join(out, f'{name}.npz')
  tmp = npz_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, obs=obs_all, act=act_all, eval_goals=eval_goals,
                        lengths=lengths, meta=json.dumps(meta))
  os.replace(tmp, npz_path)

  side_path = os.path.join(out, f'{name}_sidecar.npz')
  ep_arr = {k: np.array([r[k] for r in ep_rows]) for k in
            ('episode_id', 'u_side', 'blind', 'u_independent',
             'active_pile_side', 'teacher_mode', 'speed_cmd_nominal',
             'epsilon_override', 'collapse_step', 'dead',
             'success', 'ep_length', 'final_goal_dist', 'collection_seed')}
  tmp = side_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, **{f'step_{k}': v for k, v in step_side.items()},
                        goal_xy=eval_goals, **ep_arr)
  os.replace(tmp, side_path)

  # self-check: the learner npz must load through the real offline loader
  fp = OA.fingerprint(npz_path)
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = 29, 29, 8
  cfg.start_index, cfg.end_index = 0, -1
  cfg.goal_indices = tuple(range(29))
  cfg.max_episode_steps = HORIZON
  cfg.use_image_obs = False
  buf, _ = OA.build_offline_buffer(npz_path, cfg)
  buf.freeze()

  man = {
      'collection_date_utc': None,
      'git_commit': C.git_commit(),
      'freeze_manifest_path': C.FREEZE_PATH,
      'freeze_manifest_sha256': C.sha256_file(C.FREEZE_PATH),
      'walker_path': 'artifacts/walker/phase1/walker_best.pkl',
      'walker_sha256': info['walker_sha256'], 'walker_step': int(wmeta['step']),
      'base_policy_path': 'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl',
      'base_policy_sha256': info['base_sha256'], 'base_policy_step': base_step,
      'env_seed': env_seed, 'dataset_rng_seed': dataset_seed,
      'collection_seeds': [env_seed, dataset_seed],
      'consumed_seeds_checked': CONSUMED, 'seed_reuse': clash,
      'mixture': MIX,
      'mixture_counts': {'sighted': n_sight, 'blind': n_blind,
                         'coverage': n_cover},
      'n_episodes': int(n), 'ep_len_obs': L, 'horizon': HORIZON,
      'n_states_total': int(lengths.sum()),
      'n_transitions_total': int((lengths - 1).sum()),
      'buffer_ready_transitions': int(len(buf)),
      'env_name': 'offline_ant_umaze_litter', 'obs_dim_learner': 58,
      'state_dim': 29, 'goal_dim': 29, 'action_dim': 8,
      'episode_horizon_convention':
          'L=701 obs rows/episode (700 transitions); dead episodes truncated '
          'via lengths (relabeler ignores the padded tail). obs=state29 + '
          'zero-padded goal29 ([:2]=d4rl-sampled goal xy).',
      'epsilon': EPSILON,
      'epsilon_semantics':
          'EXACT-count 3-way mixture assigned to episode slots and shuffled by '
          'the dataset RNG (independent of the env U stream): 85% sighted '
          '(clean side opposite the active pile at V_FAST=1.4), 5% blind '
          '(epsilon confounding component: middle lane + unstick at the frozen '
          'qualified teacher_blind_v=0.6), 10% coverage (deliberate robust '
          'support: SAME middle-lane+unstick control law at '
          'coverage_middle_slow_v=0.8). blind and coverage are both '
          'U-independent; only blind carries epsilon_override.',
      'teacher_blind_v': TEACHER_BLIND_V,
      'coverage_middle_slow_v': COVERAGE_MIDDLE_SLOW_V,
      'a1_hard_ok': bool(hard_ok), 'a1_discrepancies': disc,
      'npz_sha256': C.sha256_file(npz_path),
      'sidecar_sha256': C.sha256_file(side_path),
      'fingerprint': fp,
      'code_locations': {
          'env': 'crl/d4rl_ant.py LitterOfflineAntUMazeEnv',
          'teacher_selector': 'scripts/litter_teacher.py run_teacher_episode '
                              '(replicated in collect_litter_pilot.rollout)',
          'collector': 'scripts/collect_litter_pilot.py',
          'loader': 'crl/offline_audit.py build_offline_buffer / fingerprint',
          'replay_buffer': 'crl/replay.py TrajectoryBuffer'},
  }
  json.dump(man, open(os.path.join(out, 'pilot_manifest.json'), 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

  print(f'\nwrote {npz_path}')
  print(f'wrote {side_path}')
  print(f'episodes={n} states={man["n_states_total"]} '
        f'transitions={man["n_transitions_total"]} '
        f'buffer_ready={man["buffer_ready_transitions"]}')
  print('npz sha256   :', man['npz_sha256'])
  print('sidecar sha256:', man['sidecar_sha256'])
  u = ep_arr['u_side']
  tm = ep_arr['teacher_mode']
  print(f'U balance: u1={int((u==1).sum())} u0={int((u==0).sum())}  '
        f'sighted={int((tm=="sighted").sum())} '
        f'blind={int((tm=="blind").sum())} '
        f'coverage={int((tm=="coverage").sum())}  '
        f'dead={int(ep_arr["dead"].sum())}  '
        f'success={float(ep_arr["success"].mean()):.3f}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
