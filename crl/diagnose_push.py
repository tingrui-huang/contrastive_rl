"""FetchPush failure diagnosis: why does the critic learn but the policy not solve?

Rolls out the trained policy (from a checkpoint) and a random policy in
FetchPush, and quantifies data coverage, contact, goal-relabeling diversity, and
actor/critic behavior. State-only (no rendering), so it runs anywhere MuJoCo
imports.

Run:  python -m crl.diagnose_push --ckpt fetch_push_nce_s0/best.pkl --episodes 50

Observation layout (flat, from crl.envs.FetchEnv):
  flat[0:3]   = gripper position
  flat[3:6]   = object position  (== achieved_goal for push; goal slice 3:6)
  flat[25:28] = desired_goal
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.replay import TrajectoryBuffer, obs_to_goal
from crl.config import Config

GRIP = slice(0, 3)
OBJ = slice(3, 6)          # object == achieved_goal
DES = slice(25, 28)        # desired_goal
CONTACT_THRESH = 0.06      # gripper-object distance counted as "near contact"
SUCCESS_THRESH = 0.05


def rollout(env, act_fn, n_eps, rng, key):
  """Returns per-episode arrays. act_fn(flat_obs)->action, or None for random."""
  eps = []
  for _ in range(n_eps):
    obs = env.reset()
    O = [obs.copy()]
    A = []
    for _ in range(env.max_episode_steps):
      if act_fn is None:
        a = rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
      else:
        key, sub = jax.random.split(key)
        a = np.asarray(act_fn(jnp.asarray(obs[None]), sub)[0])
      obs, _, _, _ = env.step(a)
      O.append(obs.copy()); A.append(a)
    eps.append({'obs': np.array(O), 'act': np.array(A)})  # obs [T+1,28] act[T,4]
  return eps, key


def coverage_and_contact(eps, label):
  obj_all, grip_obj_min, disp_max, disp_net = [], [], [], []
  toward = 0
  contact_eps = 0
  move_on_contact, move_no_contact = [], []
  for e in eps:
    obs = e['obs']
    obj = obs[:, OBJ]; grip = obs[:, GRIP]; des = obs[0, DES]
    obj_all.append(obj)
    d_go = np.linalg.norm(obj - des, axis=1)          # object-to-goal per step
    d_gr = np.linalg.norm(grip - obj, axis=1)         # gripper-object per step
    grip_obj_min.append(d_gr.min())
    disp = np.linalg.norm(obj - obj[0], axis=1)       # displacement from start
    disp_max.append(disp.max()); disp_net.append(disp[-1])
    toward += int(d_go[-1] < d_go[0] - 1e-6)          # did it get closer to goal?
    # contact-driven movement: object move next-step, contact vs not.
    step_move = np.linalg.norm(obj[1:] - obj[:-1], axis=1)
    near = d_gr[:-1] < CONTACT_THRESH
    if near.any():
      contact_eps += 1
      move_on_contact.extend(step_move[near].tolist())
    move_no_contact.extend(step_move[~near].tolist())
  obj_all = np.concatenate(obj_all, 0)
  disp_max = np.array(disp_max)
  n = len(eps)
  return {
      'label': label, 'n_episodes': n,
      'object_xyz_mean': obj_all.mean(0).round(3).tolist(),
      'object_xyz_std': obj_all.std(0).round(3).tolist(),
      'displacement_max_over_run': float(disp_max.max()),
      'displacement_mean_per_ep': float(disp_max.mean()),
      'frac_move_gt_0.02': float((disp_max > 0.02).mean()),
      'frac_move_gt_0.05': float((disp_max > 0.05).mean()),
      'frac_move_gt_0.10': float((disp_max > 0.10).mean()),
      'gripper_object_min_dist_mean': float(np.mean(grip_obj_min)),
      'frac_ep_near_contact': float(contact_eps / n),
      'obj_move_per_step_when_contact': float(np.mean(move_on_contact)) if move_on_contact else 0.0,
      'obj_move_per_step_no_contact': float(np.mean(move_no_contact)) if move_no_contact else 0.0,
      'frac_ep_object_closer_to_goal': float(toward / n),
  }


def relabel_diversity(eps, discount=0.99):
  """Feed rollouts through the SAME relabel sampler; measure positive spread."""
  L = eps[0]['obs'].shape[0]
  buf = TrajectoryBuffer(capacity_steps=len(eps) * L, ep_len_obs=L,
                         full_obs_dim=28, action_dim=4, obs_dim=25,
                         start_index=3, end_index=6, discount=discount, seed=0)
  for e in eps:
    act = np.concatenate([e['act'], e['act'][-1:]], 0)  # pad to L (dummy last)
    buf.add_episode(e['obs'], act)
  b = buf.sample(20000)
  anchor_obj = b.observation[:, OBJ]                 # anchor object pos
  future_goal = b.observation[:, 25:28]              # relabeled goal (obj slice)
  spread = np.linalg.norm(future_goal - anchor_obj, axis=1)
  return {
      'positive_spread_mean': float(spread.mean()),
      'positive_spread_median': float(np.median(spread)),
      'positive_spread_p90': float(np.percentile(spread, 90)),
      'frac_positives_gt_0.02': float((spread > 0.02).mean()),
      'frac_positives_gt_0.05': float((spread > 0.05).mean()),
  }


def actor_vs_random(eps_tr, eps_rand, nets, policy_params, q_params, key):
  """Action norms/saturation, and critic score vs actual object movement."""
  def action_stats(eps):
    A = np.concatenate([e['act'] for e in eps], 0)
    return (float(np.linalg.norm(A, axis=1).mean()),
            float((np.abs(A) > 0.99).mean()))
  tr_norm, tr_sat = action_stats(eps_tr)
  rd_norm, rd_sat = action_stats(eps_rand)

  # Critic score for actor actions vs random actions on the SAME states,
  # and the object movement those actions actually produced (trained rollout).
  obs = np.concatenate([e['obs'][:-1] for e in eps_tr], 0)          # [M,28]
  act = np.concatenate([e['act'] for e in eps_tr], 0)              # [M,4]
  nxt_obj = np.concatenate([e['obs'][1:, OBJ] for e in eps_tr], 0)
  cur_obj = np.concatenate([e['obs'][:-1, OBJ] for e in eps_tr], 0)
  move = np.linalg.norm(nxt_obj - cur_obj, axis=1)
  M = min(4096, obs.shape[0])
  idx = np.random.default_rng(0).choice(obs.shape[0], M, replace=False)
  o = jnp.asarray(obs[idx]); a = jnp.asarray(act[idx])
  rand_a = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, (M, 4)).astype(np.float32))
  q_actor = float(np.mean(np.diag(np.asarray(nets.q_network.apply(q_params, o, a)))))
  q_rand = float(np.mean(np.diag(np.asarray(nets.q_network.apply(q_params, o, rand_a)))))
  return {
      'action_norm_trained': tr_norm, 'action_norm_random': rd_norm,
      'action_saturation_trained': tr_sat, 'action_saturation_random': rd_sat,
      'critic_score_actor_actions': q_actor,
      'critic_score_random_actions': q_rand,
      'object_move_per_step_trained': float(move.mean()),
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt', required=True)
  p.add_argument('--env_name', default='fetch_push')
  p.add_argument('--episodes', type=int, default=50)
  p.add_argument('--out', default=None)
  args = p.parse_args()

  cfg = Config(env_name=args.env_name)
  env = envs_mod.make_env(args.env_name, cfg, seed=123)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q)
  step, state = ckpt_mod.load_checkpoint(args.ckpt)
  print(f'loaded checkpoint @ step {step}: {args.ckpt}\n')

  @jax.jit
  def greedy(obs, key):
    return nets.sample_eval(nets.policy_network.apply(state.policy_params, obs), key)

  key = jax.random.PRNGKey(0)
  rng = np.random.default_rng(0)
  eps_tr, key = rollout(env, greedy, args.episodes, rng, key)
  eps_rand, key = rollout(env, None, args.episodes, rng, key)

  cov_tr = coverage_and_contact(eps_tr, 'trained')
  cov_rd = coverage_and_contact(eps_rand, 'random')
  div_tr = relabel_diversity(eps_tr)
  div_rd = relabel_diversity(eps_rand)
  act = actor_vs_random(eps_tr, eps_rand, nets, state.policy_params,
                        state.q_params, key)

  def show(d, title):
    print(f'--- {title} ---')
    for k, v in d.items():
      print(f'   {k:36s}: {v}')
    print()

  print('=' * 68)
  print('FETCHPUSH FAILURE DIAGNOSIS  (trained vs random,'
        f' {args.episodes} eps each)')
  print('=' * 68 + '\n')
  print('[1] REPLAY / ACHIEVED-GOAL COVERAGE + [2] CONTACT')
  show(cov_tr, 'trained policy'); show(cov_rd, 'random policy')
  print('[3] GOAL-RELABELING POSITIVE DIVERSITY (||future_goal - anchor_obj||)')
  show(div_tr, 'from trained rollouts'); show(div_rd, 'from random rollouts')
  print('[4] ACTOR DIAGNOSTICS')
  show(act, 'actor vs random')

  print('[5] QUALITATIVE SUMMARY')
  print(f'   gripper contacts object   : {cov_tr["frac_ep_near_contact"]*100:.0f}% of trained eps')
  print(f'   object actually moves     : {cov_tr["frac_move_gt_0.05"]*100:.0f}% of eps move >0.05'
        f' (max {cov_tr["displacement_max_over_run"]:.2f})')
  print(f'   movement toward the goal  : {cov_tr["frac_ep_object_closer_to_goal"]*100:.0f}% of eps end closer to goal')
  print(f'   actor exploits critic?    : Q(actor)={act["critic_score_actor_actions"]:.2f} vs '
        f'Q(random)={act["critic_score_random_actions"]:.2f}; '
        f'obj move/step={act["object_move_per_step_trained"]:.4f}')

  if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump({'coverage_trained': cov_tr, 'coverage_random': cov_rd,
               'relabel_trained': div_tr, 'relabel_random': div_rd,
               'actor': act}, open(args.out, 'w'), indent=2)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
  main()
