"""Step 2 of the seed-instability diagnosis: fixed critic, fresh actor.

Take the critic (q_params) of an existing query-coverage checkpoint, freeze
it, and train a NEW actor from a fresh initialization with the unchanged
actor objective of the sealed recipe -- (1 - 0.05) * critic term + 0.05 * BC
NLL, random_goals 0.5, alpha 0, tanh-normal policy -- on the same C replay,
same 30k x 10 update budget, Adam 3e-4.  The batch order, the actor
initialization and the reparameterization keys depend only on --actor-seed,
so the same actor seed sees byte-identical inputs under different critics.
No environment step is taken during training; the saved checkpoint carries
the frozen critic as q_params so scripts/eval_pointmaze_native_routes.py can
evaluate the policy exactly as the sealed arms were evaluated.

Usage::

  python scripts/train_f4_actor_fixed_critic.py \\
      --critic outputs/pointmaze_ett_query_coverage_20260915_v1/crl/C/final.pkl \\
      --actor-seed 0 --out outputs/pointmaze_fixed_critic_actor_v1/critic_s0/actor_s0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
BC_COEF = 0.05
STEPS = 30_000
G, BATCH = 10, 256
LR = 3e-4


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_sha(tree):
  import jax
  d = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.ascontiguousarray(np.asarray(leaf))
    d.update(str(a.dtype).encode())
    d.update(str(a.shape).encode())
    d.update(a.tobytes())
  return d.hexdigest()


def make_network(log_prob_mode='clip'):
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None,
      log_prob_mode=log_prob_mode)


def crl_config(replay, seed):
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(replay),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=STEPS,
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.5, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=BATCH, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=LR,
      actor_learning_rate=LR, num_sgd_steps_per_step=G, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed))


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--critic', required=True,
                  help='checkpoint whose q_params are frozen and reused')
  ap.add_argument('--actor-seed', type=int, required=True)
  ap.add_argument('--replay', default=str(SEALED / 'replay_C.npz'))
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--log-prob', choices=('clip', 'acme'), default='clip',
                  help="BC log-prob at the action boundary: 'clip' = this "
                       "port's historical rule (clip to 1-1e-6, density at "
                       "atanh), 'acme' = dm-acme 0.4.0's boundary-band rule "
                       "used by the original contrastive_rl actor")
  ap.add_argument('--random-goals', type=float, choices=(0.0, 0.5, 1.0),
                  default=0.5,
                  help="actor goal pairing, as crl/losses.py actor_loss: 0.5 = "
                       "the sealed recipe (batch doubled, second half with "
                       "rolled goals); 0.0 = the original offline pairing "
                       "(each state with its own relabeled future goal only); "
                       "1.0 = rolled goals only")
  ap.add_argument('--out', required=True)
  args = ap.parse_args(argv)

  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint, offline_audit
  from crl import losses as losses_mod

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'final.pkl').exists():
    print(f'{out}: final.pkl exists', flush=True)
    return 0
  nets = make_network(args.log_prob)
  cfg = crl_config(args.replay, args.actor_seed)
  # buffer RNG = actor seed -> identical batch order across critics
  buffer, fp = offline_audit.build_offline_buffer(cfg.offline_dataset, cfg)
  _, critic_state = checkpoint.load_checkpoint(args.critic)
  q_params = critic_state.q_params

  key = jax.random.PRNGKey(int(args.actor_seed))
  key, k_init = jax.random.split(key)
  policy_params = nets.policy_network.init(k_init)
  optimizer = optax.adam(LR, eps=1e-7)
  opt_state = optimizer.init(policy_params)
  obs_dim = cfg.obs_dim

  random_goals = float(args.random_goals)

  def actor_loss(params, transitions, k):
    # goal pairing exactly as crl/losses.py actor_loss's three branches
    obs = transitions.observation
    state, goal = obs[:, :obs_dim], obs[:, obs_dim:]
    if random_goals == 0.0:
      new_state, new_goal, orig_action = state, goal, transitions.action
    elif random_goals == 0.5:
      new_state = jnp.concatenate([state, state], axis=0)
      new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      orig_action = jnp.concatenate([transitions.action, transitions.action], 0)
    else:
      new_state, new_goal = state, jnp.roll(goal, 1, axis=0)
      orig_action = transitions.action
    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist = nets.policy_network.apply(params, new_obs)
    action = nets.sample(dist, k)
    q = nets.q_network.apply(q_params, new_obs, action)
    q_term = -jnp.diag(q)                       # alpha = 0
    bc_nll = -nets.log_prob(dist, orig_action)
    loss = BC_COEF * bc_nll + (1 - BC_COEF) * q_term
    mode = jnp.tanh(dist.loc)
    return jnp.mean(loss), {
        'q_term': jnp.mean(q_term), 'bc_nll': jnp.mean(bc_nll),
        'policy_scale_median': jnp.median(dist.scale),
        'pre_tanh_loc_abs_mean': jnp.mean(jnp.abs(dist.loc)),
        'action_saturation_fraction':
            jnp.mean((jnp.abs(mode) > 0.99).astype(jnp.float32))}

  def one_update(carry, transitions):
    params, opt_state, k = carry
    k, sub = jax.random.split(k)
    (loss, aux), grads = jax.value_and_grad(actor_loss, has_aux=True)(
        params, transitions, sub)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    aux['loss'] = loss
    return (params, opt_state, k), aux

  @jax.jit
  def multi_update(params, opt_state, k, trans_G):
    (params, opt_state, k), aux = jax.lax.scan(
        one_update, (params, opt_state, k), trans_G)
    return params, opt_state, k, jax.tree_util.tree_map(jnp.mean, aux)

  def sample_G():
    batches = [buffer.sample(BATCH) for _ in range(G)]
    return losses_mod.Transition(*[
        jnp.asarray(np.stack([getattr(b, f) for b in batches], axis=0))
        for f in losses_mod.Transition._fields])

  print(f'fixed critic {args.critic} (q sha {tree_sha(q_params)[:12]}) | '
        f'actor seed {args.actor_seed} init sha {tree_sha(policy_params)[:12]} '
        f'| replay {args.replay} sha {fp["sha256"][:12]} | {args.steps} x {G} '
        f'updates, bc {BC_COEF} | log_prob {args.log_prob} | random_goals '
        f'{random_goals:g}', flush=True)
  init_sha = tree_sha(policy_params)
  history = []
  t0 = time.time()
  for step in range(1, args.steps + 1):
    policy_params, opt_state, key, aux = multi_update(
        policy_params, opt_state, key, sample_G())
    if step % 1000 == 0 or step == args.steps:
      rec = {k: float(v) for k, v in aux.items()}
      rec['step'] = step
      history.append(rec)
      print(f'[step {step:6d}] loss {rec["loss"]:.4f} q_term {rec["q_term"]:.3f} '
            f'bc_nll {rec["bc_nll"]:.3f} scale {rec["policy_scale_median"]:.3f} '
            f'|loc| {rec["pre_tanh_loc_abs_mean"]:.2f} sat '
            f'{rec["action_saturation_fraction"]:.2f} '
            f'({(time.time() - t0) / step * (args.steps - step):.0f}s left)',
            flush=True)
  state = losses_mod.TrainingState(
      policy_optimizer_state=opt_state, q_optimizer_state=None,
      policy_params=policy_params, q_params=q_params, target_q_params=q_params,
      key=key)
  checkpoint.save_named(str(out), 'final', args.steps, state)
  with open(out / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'critic': args.critic, 'critic_sha256': sha256(args.critic),
               'critic_q_tree_sha256': tree_sha(q_params),
               'actor_seed': args.actor_seed, 'actor_init_sha256': init_sha,
               'final_policy_sha256': tree_sha(policy_params),
               'replay': args.replay, 'replay_sha256': fp['sha256'],
               'steps': args.steps, 'sgd_steps_per_step': G, 'batch': BATCH,
               'bc_coef': BC_COEF, 'lr': LR,
               'log_prob_mode': args.log_prob, 'random_goals': random_goals,
               'critic_frozen': True, 'env_steps_during_training': 0,
               'wall_seconds': time.time() - t0, 'history': history,
               'jax_devices': [str(d) for d in jax.devices()]}, f, indent=2)
  print(f'-> {out / "final.pkl"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
