"""Alpha-direction sanity test for the implemented adaptive-alpha loss.

Uses the EXACT production path (losses.build_learner -> update_step, real adam)
rather than a re-derived formula: craft a TrainingState with a given policy and
log_alpha=0, run ONE update_step on a real observation batch, and read which
way alpha moved. Also reports the analytic gradient sign E[-log_pi - target]
and the policy's entropy estimate (with the arctanh-clip artifact flagged).

Policies tested (from the qualification runs):
  init            -- healthy, entropy ~ -2 nats
  adaptive_30k    -- mid-anneal, scale ~ 0.5
  adaptive_40k    -- fully exploded (constant saturated action)
  alpha0_50k      -- alpha=0 collapse (scale at floor, loc ~ 5)

Targets tested: 0.0 (original as-shipped) and -action_dim = -8 (candidate).
Expected correct behavior: entropy < target => alpha UP; entropy > target =>
alpha DOWN. The collapsed policies expose the known clip artifact where the
entropy ESTIMATE explodes positive, so the controller reads "entropy above
target" and pushes alpha DOWN exactly when a rescue would need it UP.
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl import losses as losses_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod

OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/alpha_direction_sanity'
B = 256

CKPTS = {
    'init': 'qual_open_near/adaptive_s0/init.pkl',
    'adaptive_30k': 'qual_open_near/adaptive_s0/gate_30000.pkl',
    'adaptive_40k_exploded': 'qual_open_near/adaptive_s0/gate_40000.pkl',
    'alpha0_50k_collapsed': 'qual_open_near/alpha0_s0/gate_50000.pkl',
}


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg = Config(env_name='antmaze_open_near', entropy_coefficient=None,
               target_entropy=0.0)
  # dims as produced by make_env for antmaze_open_near (29 state + 2 goal)
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim, cfg.max_episode_steps = 29, 2, 8, 300
  cfg.start_index, cfg.end_index = 0, 2
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)

  d = np.load('qual_open_near/gates/refs_adaptive_10000.npz')
  obs = np.tile(d['obs'], (B // len(d['obs']) + 1, 1))[:B].astype(np.float32)
  rng = np.random.default_rng(0)
  trans = losses_mod.Transition(
      observation=jnp.asarray(obs),
      action=jnp.asarray(rng.uniform(-1, 1, (B, 8)).astype(np.float32)),
      reward=jnp.zeros(B), discount=jnp.ones(B),
      next_observation=jnp.asarray(obs),
      next_action=jnp.asarray(rng.uniform(-1, 1, (B, 8)).astype(np.float32)))

  results = {}
  key = jax.random.PRNGKey(0)
  for target in (0.0, -8.0):
    cfg.target_entropy = float(target)
    policy_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
    q_opt = optax.adam(cfg.learning_rate, eps=1e-7)
    obs_to_goal = lambda s: s[:, cfg.start_index:cfg.end_index]
    init_state, update_step = losses_mod.build_learner(
        nets, cfg, obs_to_goal, policy_opt, q_opt)
    key, k0 = jax.random.split(key)
    base = init_state(k0)

    for name, path in CKPTS.items():
      _, st = ckpt_mod.load_checkpoint(path)
      state = base._replace(policy_params=st.policy_params)
      # entropy estimate under the implementation's own sample/log_prob
      key, ks = jax.random.split(key)
      dist = nets.policy_network.apply(st.policy_params, jnp.asarray(obs))
      a = nets.sample(dist, ks)
      lp = np.asarray(nets.log_prob(dist, a))
      ent_mean, ent_med = float(np.mean(-lp)), float(np.median(-lp))
      artifact_frac = float(np.mean(np.abs(lp) > 1e3))
      # analytic gradient sign: d loss / d log_alpha  ~  mean(-logp - target)
      grad_sign = float(np.sign(np.mean(-lp - target)))
      # production path: one real update_step, read alpha movement
      state, _ = update_step(state, trans)
      new_log_alpha = float(np.asarray(state.alpha_params))
      moved = ('UP' if new_log_alpha > 0 else
               'DOWN' if new_log_alpha < 0 else 'FLAT')
      expected = 'UP' if ent_mean < target else 'DOWN'
      results[f'{name}@target={target}'] = {
          'entropy_mean': ent_mean, 'entropy_median': ent_med,
          'entropy_clip_artifact_frac': artifact_frac,
          'analytic_grad_sign(+=alpha down)': grad_sign,
          'alpha_moved': moved, 'new_log_alpha': new_log_alpha,
          'expected_from_mean_entropy': expected,
          'direction_consistent': moved == expected,
      }
      r = results[f'{name}@target={target}']
      print(f"{name:24s} target={target:5.1f}  ent_mean={ent_mean:12.4g} "
            f"med={ent_med:10.4g} artifact={artifact_frac:.2f}  "
            f"alpha {moved:4s} (expected {expected})  "
            f"consistent={r['direction_consistent']}")

  # summary verdict
  healthy = ['init', 'adaptive_30k']
  ok = all(results[f'{n}@target={t}']['direction_consistent']
           for n in healthy for t in (0.0, -8.0))
  inversion = any(
      results[f'{n}@target={t}']['entropy_clip_artifact_frac'] > 0.5
      and results[f'{n}@target={t}']['alpha_moved'] == 'DOWN'
      for n in ('adaptive_40k_exploded', 'alpha0_50k_collapsed')
      for t in (0.0, -8.0))
  verdict = ('DIRECTION_CORRECT_HEALTHY_REGIME'
             + ('_BUT_INVERTED_WHEN_COLLAPSED' if inversion else ''))
  if not ok:
    verdict = 'DIRECTION_INCORRECT'
  results['verdict'] = verdict
  json.dump(results, open(os.path.join(OUT, 'alpha_direction_sanity.json'),
                          'w'), indent=2)
  print('\nVERDICT:', verdict)


if __name__ == '__main__':
  main()
