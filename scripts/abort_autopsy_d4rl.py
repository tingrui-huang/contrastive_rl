"""Checkpoint autopsy of the d4rl-faithful 250k run's guard abort (no training).

Brackets the explosion (mid.pkl @31500 healthy -> abort.pkl @32900,
|actor_loss|=4.1e21) and answers: WHICH component exploded first, and through
WHICH term of the actor loss?

Per checkpoint (init/early/mid/abort):
  * per-layer parameter magnitudes (policy MLP, loc head, scale head, critic
    sa/g encoders) + non-finite counts;
  * alpha and alpha-optimizer state health;
  * functional stats on a fixed probe-obs set (collected once with the mid
    policy): |loc|, scale, saturation, critic score at the mode action,
    action-gradient norm;
  * actor-loss decomposition on the probe batch: alpha*log_prob term vs
    diag(Q) term (verbatim loss body incl. random_goals=0.5 mixing).
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

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod

CKPTS = ['init', 'early', 'mid', 'abort']


def param_stats(tree, prefix=''):
  out = {}
  for k, v in tree.items():
    if isinstance(v, dict):
      out.update(param_stats(v, prefix + k + '/'))
    else:
      a = np.asarray(v)
      out[prefix + k] = {'max_abs': float(np.abs(a).max()),
                         'nonfinite': int((~np.isfinite(a)).sum())}
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir',
                  default='d4rl_ant_umaze_gfull_gfull29_te8_1actor_s0_250k')
  ap.add_argument('--out', default='artifacts/d4rl_abort_autopsy')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cfg = Config(env_name='d4rl_ant_umaze_gfull')
  env = envs_mod.make_env('d4rl_ant_umaze_gfull', cfg, seed=9)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)

  states = {}
  for name in CKPTS:
    step, st = ckpt_mod.load_checkpoint(
        os.path.join(args.run_dir, 'checkpoints', f'{name}.pkl'))
    states[name] = (step, st)
  print('steps:', {n: s for n, (s, _) in states.items()})

  # probe obs collected once with the MID (pre-explosion) policy, sampled.
  _, mid = states['mid']
  @jax.jit
  def _sample(params, obs, k):
    return nets.sample(nets.policy_network.apply(params, obs), k)
  key = jax.random.PRNGKey(3)
  obs_set = []
  for ep in range(3):
    obs = env.reset()
    for t in range(env.max_episode_steps):
      if t % 8 == 0:
        obs_set.append(obs.copy())
      key, sk = jax.random.split(key)
      a = np.asarray(_sample(mid.policy_params, jnp.asarray(obs[None]), sk)[0])
      obs, _, _, _ = env.step(a)
  obs_b = jnp.asarray(np.array(obs_set[:256], np.float32))
  print(f'probe obs: {obs_b.shape}')

  @jax.jit
  def _dist(params, obs):
    return nets.policy_network.apply(params, obs)

  @jax.jit
  def _q_diag(qp, obs, act):
    return jnp.diag(nets.q_network.apply(qp, obs, act))

  def actor_loss_terms(st, key):
    obs = obs_b
    s, g = obs[:, :29], obs[:, 29:]
    new_obs = jnp.concatenate(
        [jnp.concatenate([s, s], 0),
         jnp.concatenate([g, jnp.roll(g, 1, axis=0)], 0)], axis=1)
    dist = nets.policy_network.apply(st.policy_params, new_obs)
    a = nets.sample(dist, key)
    lp = nets.log_prob(dist, a)
    q = _q_diag(st.q_params, new_obs, a)
    alpha = (float(np.exp(np.asarray(st.alpha_params)))
             if st.alpha_params is not None else 0.0)
    return {'alpha': alpha,
            'alpha_logp_mean': float(np.mean(alpha * np.asarray(lp))),
            'logp_mean': float(np.mean(np.asarray(lp))),
            'logp_min': float(np.min(np.asarray(lp))),
            'logp_max': float(np.max(np.asarray(lp))),
            'q_mean': float(np.mean(np.asarray(q))),
            'q_min': float(np.min(np.asarray(q))),
            'q_max': float(np.max(np.asarray(q))),
            'actor_loss': float(np.mean(alpha * np.asarray(lp)
                                        - np.asarray(q)))}

  rep = {'steps': {n: int(s) for n, (s, _) in states.items()}}
  for name in CKPTS:
    step, st = states[name]
    ps = param_stats(st.policy_params)
    qs = param_stats(st.q_params)
    worst_p = max(ps.items(), key=lambda kv: kv[1]['max_abs'])
    worst_q = max(qs.items(), key=lambda kv: kv[1]['max_abs'])
    nonfin_p = sum(v['nonfinite'] for v in ps.values())
    nonfin_q = sum(v['nonfinite'] for v in qs.values())
    d = _dist(st.policy_params, obs_b)
    loc = np.asarray(d.loc); scale = np.asarray(d.scale)
    mode = np.tanh(loc)
    q_at_mode = np.asarray(_q_diag(st.q_params, obs_b, jnp.asarray(mode)))

    @jax.jit
    def _f1(obs1, a1, qp=st.q_params):
      return nets.q_network.apply(qp, obs1[None], a1[None])[0, 0]
    gnorms = [float(np.linalg.norm(np.asarray(
        jax.grad(_f1, argnums=1)(obs_b[i], jnp.asarray(mode[i])))))
        for i in range(0, 64, 4)]
    key, sk = jax.random.split(key)
    terms = actor_loss_terms(st, sk)
    rep[name] = {
        'step': int(step),
        'policy_param_max': {worst_p[0]: worst_p[1]['max_abs']},
        'policy_nonfinite': nonfin_p,
        'critic_param_max': {worst_q[0]: worst_q[1]['max_abs']},
        'critic_nonfinite': nonfin_q,
        'per_layer_policy_max': {k: round(v['max_abs'], 3)
                                 for k, v in ps.items()},
        'loc_abs_mean': float(np.abs(loc).mean()),
        'loc_abs_max': float(np.abs(loc).max()),
        'scale_median': float(np.median(scale)),
        'scale_min': float(scale.min()),
        'mode_sat_0.99': float(np.mean(np.abs(mode) > 0.99)),
        'q_at_mode_mean': float(q_at_mode.mean()),
        'q_at_mode_max_abs': float(np.abs(q_at_mode).max()),
        'grad_a_norm_median': float(np.median(gnorms)),
        'actor_loss_terms': terms,
    }
    t = terms
    print(f"{name:6s} step {step:6d} | loc|max| {rep[name]['loc_abs_max']:.3g} "
          f"| scale_med {rep[name]['scale_median']:.3g} "
          f"| sat {rep[name]['mode_sat_0.99']:.2f} "
          f"| q_mode {rep[name]['q_at_mode_mean']:.3g} "
          f"| alpha {t['alpha']:.4f} | alpha*logp {t['alpha_logp_mean']:.3g} "
          f"| q_term {t['q_mean']:.3g} | actor_loss {t['actor_loss']:.3g}")
    print(f"       worst policy param: {rep[name]['policy_param_max']} "
          f"nonfinite p/q: {nonfin_p}/{nonfin_q} "
          f"| worst critic param: {rep[name]['critic_param_max']}")

  json.dump(rep, open(os.path.join(args.out, 'autopsy.json'), 'w'), indent=2)
  print('\nsaved', os.path.join(args.out, 'autopsy.json'))


if __name__ == '__main__':
  main()
