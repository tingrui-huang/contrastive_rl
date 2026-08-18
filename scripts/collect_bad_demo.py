"""Phase 1: collect the FIXED non-privileged bad-demonstrator dataset.

Controller: the EXISTING 'blind' teacher mode (scripts/collect_rockfall_v2_pilot
rollout, mode='blind'; documented in scripts/rockfall_v2_teacher.py) -- the same
base-lane walker as the expert but with NO detours. It reads only learner-visible
observations plus its own position history; it never touches the rockfall mask,
severities, drop schedule, _dead, or any hidden U. The base side is drawn from an
independent RNG, exactly as in the authoritative collector, so the side choice is
mask-independent. No new policy was invented.

Environment: the patched rockfall env with death_settle_substeps=80, so a factual
fatal transition ends in the physically settled death observation. Protocol is
otherwise the authoritative one (p_active=0.30, H=800, resetfix_v1, v2.1 severity
0.80/0.15/0.05).

PRE-REGISTERED: N_bad = 500 episodes, fixed fresh seeds, all trajectories kept.
No mask forcing, no death forcing, no U modification, no counterfactual worlds,
no filtering of successes or failures, and no collect-until-enough-deaths. This
is a plain factual observational dataset from a fixed protocol.

Usage:
  python scripts/collect_bad_demo.py [--episodes 500]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from rockfall_v2_teacher import (apply_v2_config, SEVERITY_V2,  # noqa: E402
                                 protocol_version)
from collect_rockfall_pilot import (check_rockfall_freeze,       # noqa: E402
                                    prescreen_env_seed)
from collect_rockfall_v2_pilot import rollout, V2_CONSUMED       # noqa: E402

OUT_DIR = 'artifacts/bad_demo_fixed'
NAME = 'bad_demo_blind_p30_h800_settle80'
P_ACTIVE = 0.30
HORIZON = 800
SETTLE_N = 80
N_EPISODES = 500
SEED_BASE = 82_500_019           # fresh, disjoint from every consumed seed
DATASET_SEED = 82_990_013
#: seeds consumed by earlier collections/probes (kept disjoint)
EXTRA_CONSUMED = [52_400_019, 71_400_019, 51_990_013, 51_990_014,
                  71_990_013, 71_990_014]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=N_EPISODES)
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--name', default=NAME)
  ap.add_argument('--env-seed', type=int, default=None)
  ap.add_argument('--dataset-seed', type=int, default=DATASET_SEED)
  args = ap.parse_args()
  n = args.episodes
  L_ep = HORIZON + 1
  os.makedirs(args.out, exist_ok=True)

  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  if not (hard_ok and rf_ok):
    print('ABORT: frozen-integrity failure.', disc + rf_diffs)
    return 2
  excl = V2_CONSUMED + EXTRA_CONSUMED
  if args.env_seed is None:
    env_seed, prescreen = prescreen_env_seed(
        n, [SEED_BASE + 97 * k for k in range(400)], exclude=excl,
        p_active=P_ACTIVE)
  else:
    env_seed, prescreen = args.env_seed, None
  clash = C.seed_reuse(excl, [env_seed], [args.dataset_seed])
  assert not clash, 'seed clash %s' % clash
  print('env_seed %d | prescreen %s' % (env_seed, prescreen), flush=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(RP.WALKER,
                                                               RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_death_settle_substeps = SETTLE_N       # the physics patch
  env = apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=env_seed),
      P_ACTIVE, reset_fix=True)
  assert env.death_settle_substeps == SETTLE_N, 'settle patch not active'
  print('p_active %.2f | settle %d | reset_fix %s'
        % (env.p_active, env.death_settle_substeps, env._env.full_reset),
        flush=True)

  side_rng = np.random.default_rng(args.dataset_seed + 1)
  obs_all = np.zeros((n, L_ep, 58), np.float32)
  act_all = np.zeros((n, L_ep, 8), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  step_keys = ('handoff', 'lane_cmd', 'speed_cmd', 'torso_x', 'torso_y',
               'vx', 'rock_ant_contact', 'dead', 'in_detour')
  step_side = {k: np.zeros((n, L_ep), np.float32) for k in step_keys}
  ep_rows = []
  for e in range(n):
    base = 'left' if side_rng.random() < 0.5 else 'right'
    o = env.reset()
    ob, ac, vlen, sc, ep = rollout(env, o, walker, base_act, 'blind', base,
                                   HORIZON)
    obs_all[e], act_all[e], lengths[e] = ob, ac, vlen
    eval_goals[e] = ep['goal_xy']
    for k in step_keys:
      step_side[k][e] = sc[k]
    ep['episode_id'] = e
    ep['collection_seed'] = env_seed
    ep_rows.append(ep)
    if (e + 1) % 50 == 0:
      nd = sum(r['dead'] for r in ep_rows)
      print('  %d/%d episodes | deaths %d | success %.3f'
            % (e + 1, n, nd,
               float(np.mean([r['success'] for r in ep_rows]))), flush=True)

  meta = {'env_name': 'offline_ant_umaze_rockfall', 'obs_dim': 29,
          'goal_dim': 29, 'action_dim': 8, 'ep_len_obs': L_ep,
          'start_index': 0, 'end_index': -1, 'goal_indices': list(range(29)),
          'note': ('BAD-DEMO (non-privileged blind demonstrator, %s); '
                   'factual observational dataset, death_settle_substeps=%d; '
                   'learner keys obs/act only.' % (args.name, SETTLE_N))}
  npz_path = os.path.join(args.out, args.name + '.npz')
  with open(npz_path + '.tmp', 'wb') as f:
    np.savez_compressed(f, obs=obs_all, act=act_all, eval_goals=eval_goals,
                        lengths=lengths, meta=json.dumps(meta))
  os.replace(npz_path + '.tmp', npz_path)

  ep_arr = {k: np.array([r[k] for r in ep_rows]) for k in
            ('episode_id', 'rockfall_mask', 'severity', 'triggered',
             'dropped', 'hit', 'first_drop_step', 'first_hit_step',
             'impaired', 'teacher_mode', 'route', 'base_side',
             'collapse_step', 'dead', 'success', 'ep_length',
             'final_goal_dist', 'collection_seed')}
  side_path = os.path.join(args.out, args.name + '_sidecar.npz')
  with open(side_path + '.tmp', 'wb') as f:
    np.savez_compressed(f, **{'step_' + k: v for k, v in step_side.items()},
                        goal_xy=eval_goals, **ep_arr)
  os.replace(side_path + '.tmp', side_path)

  man = {
      'name': args.name,
      'controller': ("EXISTING 'blind' teacher mode (no detours, "
                     'mask-independent base side); non-privileged: uses only '
                     'learner-visible obs + own position history'),
      'controller_code': 'scripts/collect_rockfall_v2_pilot.py rollout('
                         "mode='blind')",
      'protocol': protocol_version(P_ACTIVE) + '_h800_resetfix_v1_settle80',
      'p_active': P_ACTIVE, 'horizon': HORIZON,
      'death_settle_substeps': SETTLE_N, 'reset_fix': True,
      'severity_probs_v2': list(SEVERITY_V2),
      'n_episodes': int(n),
      'n_transitions': int((lengths - 1).sum()),
      'pre_registered_n_episodes': N_EPISODES,
      'env_seed': env_seed, 'dataset_seed': args.dataset_seed,
      'base_side_rng_seed': args.dataset_seed + 1,
      'mask_prescreen_freq': prescreen,
      'seed_reuse': clash,
      'walker_path': RP.WALKER, 'walker_sha256': info['walker_sha256'],
      'walker_step': int(wmeta['step']),
      'base_policy_path': RP.BASE, 'base_policy_sha256': info['base_sha256'],
      'base_policy_step': base_step,
      'git_commit': C.git_commit(),
      'rockfall_frozen_code_commit': rf_man.get('frozen_code_commit'),
      'npz': npz_path, 'npz_sha256': C.sha256_file(npz_path),
      'sidecar': side_path, 'sidecar_sha256': C.sha256_file(side_path),
      'collection_rules': ('no mask forcing, no death forcing, no U '
                           'modification, no counterfactual worlds, no '
                           'filtering of successes/failures, no '
                           'collect-until-enough-deaths'),
  }
  json.dump(man, open(os.path.join(args.out, 'collection_manifest.json'),
                      'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  print('\nwrote %s (%d eps / %d transitions)'
        % (npz_path, n, man['n_transitions']))
  print('deaths %d | success %.3f'
        % (int(ep_arr['dead'].sum()), float(ep_arr['success'].mean())))
  print('manifest -> %s'
        % os.path.join(args.out, 'collection_manifest.json'))
  return 0


if __name__ == '__main__':
  sys.exit(main())
