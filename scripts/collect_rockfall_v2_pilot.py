"""Rockfall v2 (LOCAL-DETOUR) -- 300-episode PILOT collector.

Same FROZEN env as v1; the TEACHER control law is the qualified local-detour
policy (scripts/rockfall_v2_teacher.py). PRIMARY mixture 90/0/10:
  sighted (90%): balanced base side (mask-independent) + local detour at
                 active sites on that lane;
  blind   (0%) : NOT in the primary dataset. The blind path is retained for
                 qualification/ablation only (--blind-frac > 0);
  center  (10%): center route throughout.

Learner npz keeps the 58-dim contract (obs/act only). Privileged sidecar adds
base_side + the standard mask/route/rockfall-event fields.

Explicitly a PILOT (300 eps). No full collection here.

Usage: python scripts/collect_rockfall_v2_pilot.py [--episodes 300]
       # ablation only: --blind-frac 0.05 --coverage-frac 0.10
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
import rockfall_v2_teacher as V2          # noqa: E402
from rockfall_v2_teacher import apply_v2_config, SEVERITY_V2, V2_PROTOCOL_VERSION  # noqa: E402
from collect_rockfall_pilot import (check_rockfall_freeze, prescreen_env_seed,
                                    CONSUMED)  # noqa: E402

L = 701
HORIZON = 700
#: PRIMARY mixture: 90% sighted local-detour, 0% blind, 10% center.
MIX = {'sighted': 0.90, 'blind': 0.00, 'coverage': 0.10}
OUT_ROOT = 'artifacts/rockfall_v2_dataset'
# fresh v2 seeds (disjoint from every v1/dev seed via CONSUMED + prescreen)
V2_CONSUMED = CONSUMED + [90_100_019, 88_140_077, 41_001, 42_001, 43_517]


def rollout(env, o, walker, base_act, mode, base_side):
  """One frozen v2 episode. mode in {sighted, blind, coverage}. For
  sighted/blind, base_side in {left,right}; coverage ignores it (center)."""
  true_goal = o[29:31].copy()
  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  sc = {k: np.full(L, np.nan, np.float32) for k in
        ('lane_cmd', 'speed_cmd', 'torso_x', 'torso_y', 'vx')}
  sc.update({k: np.zeros(L, np.float32) for k in
             ('handoff', 'rock_ant_contact', 'dead', 'in_detour')})

  is_center = mode == 'coverage'
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if mode == 'sighted' else [])
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
      in_det = False
    else:
      x_hist.append(x)
      if is_center:
        y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
        in_det = False
      else:
        in_det = any(x0 <= x <= x1 for x0, x1 in wins)
        y_cmd, v_cmd = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                         RP.V_SIDE)
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
    sc['in_detour'][t] = float(bool(in_det))
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if dead_at >= 0:
      valid_len = min(dead_at + 2, L)
      break
    o = o2
  drop_steps = env._drop_step
  hit_steps = env._hit_step
  route = 'center' if is_center else base_side
  ep = {'rockfall_mask': np.asarray(env.rockfall_mask, np.int8),
        'severity': np.asarray(env.privileged_severity),
        'triggered': np.asarray(env._triggered, bool),
        'dropped': np.asarray(env._dropped, bool),
        'hit': np.asarray(env._hit, bool),
        'first_drop_step': int(min(drop_steps.values()) if drop_steps
                               else -1),
        'first_hit_step': int(min(hit_steps.values()) if hit_steps else -1),
        'impaired': bool(env._impaired_leg_ids),
        'teacher_mode': mode, 'route': route, 'base_side': base_side,
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'ep_length': int(valid_len),
        'final_goal_dist': float(np.linalg.norm(obs[valid_len - 1, :2]
                                                - true_goal)),
        'goal_xy': true_goal.astype(np.float32)}
  return obs, act, valid_len, sc, ep


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=300)
  ap.add_argument('--out', default=None)
  ap.add_argument('--env-seed', type=int, default=None)
  ap.add_argument('--dataset-seed', type=int, default=51_990_013)
  ap.add_argument('--seed-base', type=int, default=52_400_019)
  ap.add_argument('--name', default='antmaze_rockfall_v2_pilot')
  ap.add_argument('--blind-frac', type=float, default=MIX['blind'],
                  help='ABLATION ONLY: >0 puts blind episodes in the set '
                       '(primary dataset is 0)')
  ap.add_argument('--coverage-frac', type=float, default=MIX['coverage'])
  args = ap.parse_args()
  n = args.episodes

  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  if args.env_seed is None:
    env_seed, prescreen_freq = prescreen_env_seed(
        n, [args.seed_base + 97 * k for k in range(400)], exclude=V2_CONSUMED)
  else:
    env_seed, prescreen_freq = args.env_seed, None
  clash = C.seed_reuse(V2_CONSUMED, [env_seed], [args.dataset_seed])
  print(f'integrity: litter={hard_ok} rockfall={rf_ok} clash={clash} '
        f'env_seed={env_seed} prescreen={prescreen_freq}')
  for d in disc + rf_diffs:
    print('   discrepancy:', json.dumps(d))
  if not (hard_ok and rf_ok) or clash:
    print('ABORT: integrity/seed failure.')
    return 2

  out = args.out or os.path.join(OUT_ROOT, 'pilot')
  if os.path.exists(os.path.join(out, f'{args.name}.npz')):
    v = 2
    while os.path.exists(out + f'_v{v}'):
      v += 1
    out = out + f'_v{v}'
  os.makedirs(out, exist_ok=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(RP.WALKER,
                                                               RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=env_seed))

  dataset_rng = np.random.default_rng(args.dataset_seed)
  side_rng = np.random.default_rng(args.dataset_seed + 1)   # base side, indep
  n_blind = int(round(args.blind_frac * n))
  n_cover = int(round(args.coverage_frac * n))
  n_sight = n - n_blind - n_cover
  modes = np.array(['sighted'] * n_sight + ['blind'] * n_blind
                   + ['coverage'] * n_cover)
  dataset_rng.shuffle(modes)
  is_primary = n_blind == 0
  print(f'mixture: sighted={n_sight} blind={n_blind} coverage={n_cover} '
        f'({"PRIMARY 90/0/10" if is_primary else "ABLATION"})')

  obs_all = np.zeros((n, L, 58), np.float32)
  act_all = np.zeros((n, L, 8), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  step_keys = ('handoff', 'lane_cmd', 'speed_cmd', 'torso_x', 'torso_y',
               'vx', 'rock_ant_contact', 'dead', 'in_detour')
  step_side = {k: np.zeros((n, L), np.float32) for k in step_keys}
  ep_rows = []
  for e in range(n):
    mode = str(modes[e])
    base = 'left' if side_rng.random() < 0.5 else 'right'  # indep of mask
    o = env.reset()
    obs, act, vlen, sc, ep = rollout(env, o, walker, base_act, mode, base)
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
          'note': f'Rockfall v2 LOCAL-DETOUR pilot ({args.name}); '
                  'learner keys obs/act only.'}
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
             'impaired', 'teacher_mode', 'route', 'base_side',
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
      'variant': V2_PROTOCOL_VERSION,
      'severity_probs_v2': list(SEVERITY_V2),
      'severity_note': ('v2.1 protocol: severity 0.80/0.15/0.05 applied to the '
                        'env INSTANCE at collection time; frozen env module '
                        'default (0.55/0.30/0.15) and global-route v1 unchanged.'),
      'git_commit': C.git_commit(),
      'rockfall_frozen_code_commit': rf_man.get('frozen_code_commit'),
      'teacher': 'scripts/rockfall_v2_teacher.py local-detour',
      'detour_params': {'detour_y': V2.DETOUR_Y, 'pre': V2.DETOUR_PRE,
                        'post': V2.DETOUR_POST, 'base_lane': V2.BASE_LANE},
      'walker_path': RP.WALKER, 'walker_sha256': info['walker_sha256'],
      'walker_step': int(wmeta['step']),
      'base_policy_path': RP.BASE, 'base_policy_sha256': info['base_sha256'],
      'base_policy_step': base_step,
      'env_seed': env_seed, 'dataset_rng_seed': args.dataset_seed,
      'base_side_rng_seed': args.dataset_seed + 1,
      'mask_prescreen_freq': prescreen_freq,
      'seed_reuse': clash,
      'is_primary_90_0_10': bool(is_primary),
      'mixture': {'sighted': n_sight / n, 'blind': n_blind / n,
                  'coverage': n_cover / n},
      'mixture_counts': {'sighted': n_sight, 'blind': n_blind,
                         'coverage': n_cover},
      'n_episodes': int(n), 'ep_len_obs': L, 'horizon': HORIZON,
      'n_transitions_total': int((lengths - 1).sum()),
      'buffer_ready_transitions': int(len(buf)),
      'env_name': 'offline_ant_umaze_rockfall', 'obs_dim_learner': 58,
      'npz_sha256': C.sha256_file(npz_path),
      'sidecar_sha256': C.sha256_file(side_path), 'fingerprint': fp,
      'integrity': {'litter_hard_ok': bool(hard_ok), 'rockfall_ok': bool(rf_ok),
                    'discrepancies': disc + rf_diffs},
  }
  json.dump(man, open(os.path.join(out, 'pilot_manifest.json'), 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

  m = ep_arr['rockfall_mask']
  tm = ep_arr['teacher_mode']
  bs = ep_arr['base_side']
  print(f'\nwrote {npz_path}\nwrote {side_path}')
  print(f'episodes={n} transitions={man["n_transitions_total"]}')
  print(f'site activation {m.mean(0).round(3)} | base_left '
        f'{float((bs=="left").mean()):.3f} | success '
        f'{float(ep_arr["success"].mean()):.3f} | dead '
        f'{int(ep_arr["dead"].sum())}')
  for mode in ('sighted', 'blind', 'coverage'):
    sel = tm == mode
    print(f'  {mode:8s} n={int(sel.sum())} success '
          f'{float(ep_arr["success"][sel].mean()):.3f}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
