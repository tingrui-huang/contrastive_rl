"""Rockfall Stage-A: collect a small, fully audited PILOT dataset (~300
episodes; explicitly NOT the full collection). Frozen walker + base policy +
the qualified rockfall pilot route protocol (rockfall_pilot.py constants are
imported, not copied). Learner npz keeps the litter 58-dim contract
(obs/act/eval_goals/lengths/meta ONLY); every privileged field (mask,
severity, trigger/drop/hit, impairment) goes to the sidecar.

Mixture (mirrors the approved litter Stage-3 recipe):
  85% sighted  -- privileged teacher: mask-aware route rule
                  (clear side at V_SIDE, center when both sides active)
   5% blind    -- mask-ignorant side runner (alternating left/right at
                  V_SIDE): the confounding component that shows side
                  travel WITHOUT the teacher's mask filter
  10% coverage -- center route at V_CENTER (robust-support component)

Run:  python scripts/collect_rockfall_pilot.py [--episodes 300] [--smoke N]
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
from crl import rockfall_ant as RA        # noqa: E402
from crl import offline_audit as OA       # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402

L = 701
HORIZON = 700
MIX = {'sighted': 0.85, 'blind': 0.05, 'coverage': 0.10}
OUT_ROOT = 'artifacts/rockfall_dataset'
ROCKFALL_FREEZE = 'artifacts/rockfall_pilot/rockfall_freeze.json'

#: every seed consumed by rockfall development/tuning/pilot runs (plus the
#: litter collection seeds for cross-project discipline).
CONSUMED = [311, 500, 622, 777, 888, 999, 1234,
            8_150_023, 5_090_023, 9_271_033, 6_330_047,
            12_450_067, 14_760_053, 25_770_061, 28_110_043,
            60_001, 60_002, 60_777, 61_020, 61_030, 61_040, 61_050,
            62_001, 62_100, 62_200, 62_300, 63_001, 64_001, 71_717]


def prescreen_env_seed(n_eps, candidates):
  """First candidate whose mask stream gives every site a realized
  activation in [0.15, 0.25] over the first n_eps natural draws. Previews
  ONLY the env's independent mask rng; touches no physics/policy stream."""
  for seed in candidates:
    rng = np.random.default_rng(seed + 41_007)
    bits = (rng.random((n_eps, 4)) < RA.P_ACTIVE).astype(int)
    f = bits.mean(0)
    if np.all((f >= 0.15) & (f <= 0.25)):
      return int(seed), [float(x) for x in f]
  raise RuntimeError('no candidate seed passed the mask prescreen')


def check_rockfall_freeze():
  """Live rockfall config must match the frozen manifest exactly."""
  man = json.load(open(ROCKFALL_FREEZE))
  live = RA.rockfall_config()
  frozen = man['config']
  ok = live == frozen
  diffs = []
  if not ok:
    for k in sorted(set(live) | set(frozen)):
      if live.get(k) != frozen.get(k):
        diffs.append({'field': k, 'code': live.get(k),
                      'manifest': frozen.get(k)})
  proto = man['route_protocol']
  for field, code_val in (('lane', RP.LANE), ('v_side', RP.V_SIDE),
                          ('v_center', RP.V_CENTER),
                          ('handoff_x', RP.HANDOFF_X),
                          ('nudge_y', RP.NUDGE_Y)):
    if abs(float(proto[field]) - float(code_val)) > 1e-9:
      ok = False
      diffs.append({'field': f'route_protocol.{field}', 'code': code_val,
                    'manifest': proto[field]})
  return ok, diffs, man


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=300)
  ap.add_argument('--smoke', type=int, default=0)
  ap.add_argument('--out', default=None)
  ap.add_argument('--env-seed', type=int, default=None,
                  help='default: first prescreen-passing candidate')
  ap.add_argument('--dataset-seed', type=int, default=76_230_011)
  ap.add_argument('--name', default='antmaze_rockfall_pilot')
  args = ap.parse_args()
  n = args.smoke or args.episodes

  # --- gate BEFORE any collection: litter checkpoint integrity (walker +
  # base sha) AND the rockfall freeze manifest ---
  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  if args.env_seed is None:
    env_seed, prescreen_freq = prescreen_env_seed(
        n, [73_500_019 + 97 * k for k in range(200)])
  else:
    env_seed, prescreen_freq = args.env_seed, None
  clash = C.seed_reuse(CONSUMED, [env_seed], [args.dataset_seed])
  print(f'integrity: litter hard_ok={hard_ok} rockfall_ok={rf_ok} '
        f'seed_clash={clash} env_seed={env_seed} '
        f'prescreen_freq={prescreen_freq}')
  for d in disc + rf_diffs:
    print('   discrepancy:', json.dumps(d))
  if not (hard_ok and rf_ok) or clash:
    print('ABORT: frozen-integrity failure or seed reuse.')
    return 2

  out = args.out or os.path.join(OUT_ROOT, 'smoke' if args.smoke else 'pilot')
  if not args.smoke and os.path.exists(os.path.join(out, f'{args.name}.npz')):
    v = 2
    while os.path.exists(out + f'_v{v}'):
      v += 1
    out = out + f'_v{v}'
    print('existing pilot found; using versioned dir', out)
  os.makedirs(out, exist_ok=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(RP.WALKER,
                                                               RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=env_seed)

  dataset_rng = np.random.default_rng(args.dataset_seed)
  n_blind = int(round(MIX['blind'] * n))
  n_cover = int(round(MIX['coverage'] * n))
  n_sight = n - n_blind - n_cover
  modes = np.array(['sighted'] * n_sight + ['blind'] * n_blind
                   + ['coverage'] * n_cover)
  dataset_rng.shuffle(modes)
  #: teacher both-clear tie-break stream, owned by the dataset rng chain
  tie_rng = np.random.default_rng(args.dataset_seed + 1)
  print(f'mixture: sighted={n_sight} blind={n_blind} coverage={n_cover}')

  obs_all = np.zeros((n, L, 58), np.float32)
  act_all = np.zeros((n, L, 8), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  step_keys = ('handoff', 'lane_cmd', 'speed_cmd', 'torso_x', 'torso_y',
               'vx', 'rock_ant_contact', 'dead')
  step_side = {k: np.zeros((n, L), np.float32) for k in step_keys}
  ep_rows = []
  n_blind_seen = 0
  for e in range(n):
    mode = str(modes[e])
    # route decided BEFORE reset for blind/coverage; sighted needs the
    # mask, which exists only after reset -- peek via a same-stream
    # equivalent is impossible, so sighted routes are chosen after reset
    # inside rollout order: here we reset inside rollout, so pass a
    # callable-resolved route: do the reset here instead.
    o = env.reset()
    if mode == 'sighted':
      route = RP.teacher_route(env.rockfall_mask, tie_rng)
    elif mode == 'blind':
      route = 'left' if n_blind_seen % 2 == 0 else 'right'
      n_blind_seen += 1
    else:
      route = 'center'
    obs, act, vlen, sc, ep = rollout(env, o, walker, base_act, mode, route)
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

  meta = {'env_name': 'offline_ant_umaze_rockfall', 'obs_dim': 29,
          'goal_dim': 29, 'action_dim': 8, 'ep_len_obs': L,
          'start_index': 0, 'end_index': -1, 'goal_indices': list(range(29)),
          'note': 'Rockfall Stage-A pilot; learner keys are obs/act only.'}
  npz_path = os.path.join(out, f'{args.name}.npz')
  tmp = npz_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, obs=obs_all, act=act_all, eval_goals=eval_goals,
                        lengths=lengths, meta=json.dumps(meta))
  os.replace(tmp, npz_path)

  side_path = os.path.join(out, f'{args.name}_sidecar.npz')
  ep_arr = {k: np.array([r[k] for r in ep_rows]) for k in
            ('episode_id', 'rockfall_mask', 'severity', 'triggered',
             'dropped', 'hit', 'first_drop_step', 'first_hit_step',
             'impaired', 'impaired_legs', 'teacher_mode', 'route',
             'collapse_step', 'dead', 'success', 'ep_length',
             'final_goal_dist', 'collection_seed')}
  tmp = side_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, **{f'step_{k}': v for k, v in step_side.items()},
                        goal_xy=eval_goals, **ep_arr)
  os.replace(tmp, side_path)

  fp = OA.fingerprint(npz_path)
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = 29, 29, 8
  cfg.start_index, cfg.end_index = 0, -1
  cfg.goal_indices = tuple(range(29))
  cfg.max_episode_steps = HORIZON
  cfg.use_image_obs = False
  buf, _ = OA.build_offline_buffer(npz_path, cfg)
  buf.freeze()

  man = {
      'git_commit': C.git_commit(),
      'rockfall_freeze_path': ROCKFALL_FREEZE,
      'rockfall_freeze_sha256': C.sha256_file(ROCKFALL_FREEZE),
      'rockfall_frozen_code_commit': rf_man.get('frozen_code_commit'),
      'walker_path': RP.WALKER, 'walker_sha256': info['walker_sha256'],
      'walker_step': int(wmeta['step']),
      'base_policy_path': RP.BASE,
      'base_policy_sha256': info['base_sha256'],
      'base_policy_step': base_step,
      'env_seed': env_seed, 'dataset_rng_seed': args.dataset_seed,
      'mask_prescreen_freq': prescreen_freq,
      'consumed_seeds_checked': CONSUMED, 'seed_reuse': clash,
      'mixture': MIX,
      'mixture_counts': {'sighted': n_sight, 'blind': n_blind,
                         'coverage': n_cover},
      'route_protocol': {'lane': RP.LANE, 'v_side': RP.V_SIDE,
                         'v_center': RP.V_CENTER,
                         'handoff_x': RP.HANDOFF_X,
                         'center_unstick': {'window': RP.STALL_WINDOW,
                                            'min_dx': RP.STALL_MIN_DX,
                                            'nudge_y': RP.NUDGE_Y,
                                            'nudge_steps': RP.NUDGE_STEPS}},
      'n_episodes': int(n), 'ep_len_obs': L, 'horizon': HORIZON,
      'n_states_total': int(lengths.sum()),
      'n_transitions_total': int((lengths - 1).sum()),
      'buffer_ready_transitions': int(len(buf)),
      'env_name': 'offline_ant_umaze_rockfall', 'obs_dim_learner': 58,
      'npz_sha256': C.sha256_file(npz_path),
      'sidecar_sha256': C.sha256_file(side_path),
      'fingerprint': fp,
      'integrity': {'litter_hard_ok': bool(hard_ok),
                    'rockfall_ok': bool(rf_ok),
                    'discrepancies': disc + rf_diffs},
      'code_locations': {
          'env': 'crl/rockfall_ant.py RockfallOfflineAntUMazeEnv',
          'route_protocol': 'scripts/rockfall_pilot.py (imported, not copied)',
          'collector': 'scripts/collect_rockfall_pilot.py'},
  }
  json.dump(man, open(os.path.join(out, 'pilot_manifest.json'), 'w'),
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

  print(f'\nwrote {npz_path}')
  print(f'wrote {side_path}')
  print(f'episodes={n} transitions={man["n_transitions_total"]} '
        f'buffer_ready={man["buffer_ready_transitions"]}')
  m = ep_arr['rockfall_mask']
  tm = ep_arr['teacher_mode']
  print(f'site activation: {m.mean(0).round(3)}  '
        f'success={float(ep_arr["success"].mean()):.3f}  '
        f'dead={int(ep_arr["dead"].sum())}  '
        f'routes: ' + ' '.join(f'{r}={int((ep_arr["route"] == r).sum())}'
                               for r in ('left', 'right', 'center')))
  return 0


def rollout(env, o, walker, base_act, mode, route):
  """One frozen episode from an ALREADY-reset env (the sighted route
  needs the post-reset mask). Single code path for every mode."""
  true_goal = o[29:31].copy()
  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  sc = {k: np.full(L, np.nan, np.float32) for k in
        ('lane_cmd', 'speed_cmd', 'torso_x', 'torso_y', 'vx')}
  sc.update({k: np.zeros(L, np.float32) for k in
             ('handoff', 'rock_ant_contact', 'dead')})
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  dead_at, hit = -1, 0.0
  valid_len = L
  for t in range(HORIZON):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
      lane_cmd = speed_cmd = np.nan
    else:
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command(route, t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
      lane_cmd, speed_cmd = float(y_cmd), float(v_cmd)
    o2, r, _, info = env.step(a)
    act[t] = a
    obs[t + 1] = o2
    sc['handoff'][t] = float(handoff)
    sc['lane_cmd'][t] = lane_cmd
    sc['speed_cmd'][t] = speed_cmd
    sc['torso_x'][t] = float(o2[0])
    sc['torso_y'][t] = float(o2[1])
    sc['vx'][t] = float(env._env.data.qvel[0])
    sc['rock_ant_contact'][t] = float(bool(info['rock_ant_contact']))
    sc['dead'][t] = float(bool(info['dead']))
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if dead_at >= 0:
      valid_len = min(dead_at + 2, L)
      break
    o = o2
  drop_steps = env._drop_step
  hit_steps = env._hit_step
  legs = list(env._impaired_leg_ids) + [-1] * 4
  ep = {'rockfall_mask': np.asarray(env.rockfall_mask, np.int8),
        'severity': np.asarray(env.privileged_severity),
        'triggered': np.asarray(env._triggered, bool),
        'dropped': np.asarray(env._dropped, bool),
        'hit': np.asarray(env._hit, bool),
        'first_drop_step': int(min(drop_steps.values()) if drop_steps
                               else -1),
        'first_hit_step': int(min(hit_steps.values()) if hit_steps else -1),
        'impaired': bool(env._impaired_leg_ids),
        'impaired_legs': np.asarray(legs[:4], np.int8),
        'teacher_mode': mode, 'route': route,
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'ep_length': int(valid_len),
        'final_goal_dist': float(np.linalg.norm(obs[valid_len - 1, :2]
                                                - true_goal)),
        'goal_xy': true_goal.astype(np.float32)}
  return obs, act, valid_len, sc, ep


if __name__ == '__main__':
  sys.exit(main())
