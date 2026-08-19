"""Phase 3: fixed factual collection for one pre-registered V3 arm.

Runs the arm's trained non-privileged CRL policy (deterministic eval action
tanh(loc) on the 58-dim learner observation) in the patched rockfall env with
death_settle_substeps=80, for exactly the pre-registered number of episodes
and seeds. Every trajectory is retained -- successes and failures alike.

The policy sees ONLY the learner observation. No mask forcing, no severity
forcing, no failure-only filtering, no stop-at-target-deaths, no alternate
hidden worlds, no counterfactual replay. Dead episodes are truncated at
collapse_step+2 exactly as in the authoritative collector, so the last stored
observation of a death IS the N=80 physically settled fatal state.

Usage:
  python scripts/collect_bad_demo_diverse.py --arm naive
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from rockfall_v2_teacher import apply_v2_config, SEVERITY_V2  # noqa: E402
from collect_rockfall_pilot import check_rockfall_freeze  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ROOT = 'artifacts/flow_v3_diverse_failure'
OUT_DIR = os.path.join(ROOT, 'bad_demo_diverse')
P_ACTIVE, HORIZON, SETTLE_N = 0.30, 800, 80


def policy_rollout(env, o, act_fn, horizon=HORIZON):
  """One factual episode under a trained non-privileged policy.

  Mirrors collect_rockfall_v2_pilot.rollout's storage contract: run to the
  horizon, and on death stop with valid_len = collapse_step + 2 so the final
  stored observation is the settled fatal state."""
  L = horizon + 1
  true_goal = o[29:31].copy()
  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  sc = {k: np.zeros(L, np.float32) for k in
        ('rock_ant_contact', 'dead', 'torso_x', 'torso_y', 'vx')}
  dead_at, hit = -1, 0.0
  valid_len = L
  for t in range(horizon):
    a = np.asarray(act_fn(jnp.asarray(o[None]))[0], np.float32)
    o2, r, _, info = env.step(a)
    act[t] = a
    obs[t + 1] = o2
    sc['rock_ant_contact'][t] = float(bool(info['rock_ant_contact']))
    sc['dead'][t] = float(bool(info['dead']))
    sc['torso_x'][t] = float(o2[0])
    sc['torso_y'][t] = float(o2[1])
    sc['vx'][t] = float(env._env.data.qvel[0])
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if dead_at >= 0:
      valid_len = min(dead_at + 2, L)
      break
    o = o2
  drop_steps, hit_steps = env._drop_step, env._hit_step
  ep = {'rockfall_mask': np.asarray(env.rockfall_mask, np.int8),
        'severity': np.asarray(env.privileged_severity),
        'triggered': np.asarray(env._triggered, bool),
        'dropped': np.asarray(env._dropped, bool),
        'hit': np.asarray(env._hit, bool),
        'first_drop_step': int(min(drop_steps.values()) if drop_steps else -1),
        'first_hit_step': int(min(hit_steps.values()) if hit_steps else -1),
        'impaired': bool(env._impaired_leg_ids),
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'ep_length': int(valid_len),
        'final_goal_dist': float(np.linalg.norm(obs[valid_len - 1, :2]
                                                - true_goal)),
        'goal_xy': true_goal.astype(np.float32)}
  return obs, act, valid_len, sc, ep


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--arm', required=True)
  ap.add_argument('--out', default=OUT_DIR)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  proto = json.load(open(os.path.join(ROOT, 'collection_protocol.json')))
  arm = next((a for a in proto['arms'] if a['arm'] == args.arm), None)
  assert arm is not None, 'arm %s is not pre-registered' % args.arm
  assert C.sha256_file(arm['ckpt']) == arm['ckpt_sha256'], \
      'policy checkpoint drifted from the pre-registered sha'
  n = int(arm['episodes'])

  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  assert hard_ok and rf_ok, 'frozen-integrity failure %s' % (disc + rf_diffs)

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.env_name = 'offline_ant_umaze_rockfall'
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_severity = SEVERITY_V2
  cfg.rockfall_p_active = P_ACTIVE
  cfg.rockfall_max_steps = HORIZON
  cfg.max_episode_steps = HORIZON
  cfg.rockfall_reset_fix = True
  cfg.rockfall_death_settle_substeps = SETTLE_N
  env = envs_mod.make_env(cfg.env_name, cfg, seed=int(arm['env_seed']))
  env = apply_v2_config(env, P_ACTIVE, reset_fix=True)
  env.death_settle_substeps = SETTLE_N
  assert env.death_settle_substeps == SETTLE_N

  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(arm['ckpt'])

  @jax.jit
  def act_fn(o, p=st.policy_params):
    return jnp.tanh(nets.policy_network.apply(p, o).loc)

  print('ARM %s | policy %s @ step %d | %d episodes | env_seed %d | settle %d'
        % (args.arm, arm['ckpt'], step, n, arm['env_seed'], SETTLE_N),
        flush=True)

  L_ep = HORIZON + 1
  obs_all = np.zeros((n, L_ep, 58), np.float32)
  act_all = np.zeros((n, L_ep, 8), np.float32)
  lengths = np.zeros(n, np.int64)
  eval_goals = np.zeros((n, 2), np.float32)
  step_keys = ('rock_ant_contact', 'dead', 'torso_x', 'torso_y', 'vx')
  step_side = {k: np.zeros((n, L_ep), np.float32) for k in step_keys}
  ep_rows = []
  for e in range(n):
    o = env.reset()
    ob, ac, vlen, sc, ep = policy_rollout(env, o, act_fn)
    obs_all[e], act_all[e], lengths[e] = ob, ac, vlen
    eval_goals[e] = ep['goal_xy']
    for k in step_keys:
      step_side[k][e] = sc[k]
    ep['episode_id'] = e
    ep['arm'] = args.arm
    ep_rows.append(ep)
    if (e + 1) % 50 == 0:
      nd = sum(r['dead'] for r in ep_rows)
      print('  %d/%d | deaths %d | success %.3f'
            % (e + 1, n, nd,
               float(np.mean([r['success'] for r in ep_rows]))), flush=True)

  name = 'bad_demo_%s' % args.arm
  meta = {'env_name': cfg.env_name, 'obs_dim': 29, 'goal_dim': 29,
          'action_dim': 8, 'ep_len_obs': L_ep, 'start_index': 0,
          'end_index': -1, 'goal_indices': list(range(29)),
          'arm': args.arm,
          'note': ('V3 diverse factual collection, arm=%s (non-privileged '
                   'trained CRL policy), death_settle_substeps=%d'
                   % (args.arm, SETTLE_N))}
  npz_path = os.path.join(args.out, name + '.npz')
  with open(npz_path + '.tmp', 'wb') as f:
    np.savez_compressed(f, obs=obs_all, act=act_all, eval_goals=eval_goals,
                        lengths=lengths, meta=json.dumps(meta))
  os.replace(npz_path + '.tmp', npz_path)
  ep_arr = {k: np.array([r[k] for r in ep_rows]) for k in
            ('episode_id', 'rockfall_mask', 'severity', 'triggered',
             'dropped', 'hit', 'first_drop_step', 'first_hit_step',
             'impaired', 'collapse_step', 'dead', 'success', 'ep_length',
             'final_goal_dist', 'arm')}
  side_path = os.path.join(args.out, name + '_sidecar.npz')
  with open(side_path + '.tmp', 'wb') as f:
    np.savez_compressed(f, **{'step_' + k: v for k, v in step_side.items()},
                        goal_xy=eval_goals, **ep_arr)
  os.replace(side_path + '.tmp', side_path)

  man = {'arm': args.arm, 'kind': arm['kind'], 'policy_ckpt': arm['ckpt'],
         'policy_ckpt_sha256': arm['ckpt_sha256'], 'policy_step': int(step),
         'action_convention': 'deterministic eval tanh(loc) on obs58',
         'privileged_inputs': 'none (learner observation only)',
         'env_seed': int(arm['env_seed']),
         'dataset_seed': int(arm['dataset_seed']),
         'p_active': P_ACTIVE, 'horizon': HORIZON,
         'death_settle_substeps': SETTLE_N, 'reset_fix': True,
         'severity_probs_v2': list(SEVERITY_V2),
         'n_episodes': int(n),
         'n_transitions': int((lengths - 1).sum()),
         'n_dead_episodes': int(ep_arr['dead'].sum()),
         'success_rate': float(ep_arr['success'].mean()),
         'npz': npz_path, 'npz_sha256': C.sha256_file(npz_path),
         'sidecar_sha256': C.sha256_file(side_path),
         'walker_sha256': info['walker_sha256'],
         'git_commit': C.git_commit()}
  json.dump(man, open(os.path.join(args.out, name + '_manifest.json'), 'w'),
            indent=2)
  print('\nARM %s done: %d eps / %d transitions | deaths %d | success %.3f'
        % (args.arm, n, man['n_transitions'], man['n_dead_episodes'],
           man['success_rate']))
  print('-> %s' % npz_path)


if __name__ == '__main__':
  main()
