"""Goal block-attribution probe for the goal-representation ablation.

Measures which goal blocks (XY / pose / velocity / joints) the critic actually
uses for future-goal retrieval, to detect shortcut learning (e.g. matching
velocities or joint phase instead of XY position).

Protocol per checkpoint:
  * collect fresh sampled-policy rollouts (same distribution as training
    collection);
  * build a 64-way retrieval task: anchor (state, deterministic actor action),
    positive = geometric future state of the SAME episode (discount 0.99),
    63 negatives = future states of OTHER episodes; goals are the arm's
    goal_indices slice of those states;
  * baseline: categorical accuracy + median rank of the positive;
  * knock-out: shuffle ONE block's goal columns across the 64 candidates
    (per anchor) -> accuracy drop = how much the critic RELIES on the block;
  * only-block: shuffle all OTHER goal columns -> accuracy retained = how
    SUFFICIENT the block alone is.

Shortcut signature: low reliance/sufficiency for XY with high values for
velocity/pose/joints -- retrieval without XY control value.
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
from crl.envs import ANT_GOAL_BLOCKS
from ant_entropy_audit import make_policy

N_EPS = 10
N_ANCHOR = 96
K = 64
DISCOUNT = 0.99


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--env_name', required=True)
  ap.add_argument('--tag', required=True)
  ap.add_argument('--out', default='qual_open_near/attribution')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  cfg = Config(env_name=args.env_name)
  env = envs_mod.make_env(args.env_name, cfg, seed=31)
  gi = list(cfg.goal_indices) if cfg.goal_indices is not None else [0, 1]
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  step, state = ckpt_mod.load_checkpoint(args.ckpt)
  pol = make_policy(nets, state.policy_params)

  @jax.jit
  def _score(obs_k, acts_k):
    return jnp.diag(nets.q_network.apply(state.q_params, obs_k, acts_k))

  # --- collect rollouts (sampled policy, like training collection) ---
  key = jax.random.PRNGKey(11)
  states, eps = [], []
  for ep in range(N_EPS):
    obs = env.reset()
    for t in range(env.max_episode_steps):
      states.append(obs[:cfg.obs_dim].copy())
      eps.append(ep)
      key, sk = jax.random.split(key)
      obs, _, _, _ = env.step(pol['sample'](obs, sk))
  states = np.array(states, np.float32)
  eps = np.array(eps)
  L = env.max_episode_steps

  # --- build retrieval sets ---
  anchors, pos_goals, neg_goals = [], [], []
  for _ in range(N_ANCHOR):
    ep = int(rng.integers(N_EPS))
    idx = np.where(eps == ep)[0]
    i = int(rng.integers(0, L - 1))
    # geometric future j > i (truncated)
    d = rng.geometric(1 - DISCOUNT)
    j = min(i + d, L - 1)
    anchors.append(idx[i])
    pos_goals.append(idx[j])
    other = np.where(eps != ep)[0]
    neg_goals.append(rng.choice(other, K - 1, replace=False))
  A_state = states[anchors]                                # [N, 29]
  A_act = np.array([pol['mode'](np.concatenate(
      [s, states[g][gi]]).astype(np.float32))
      for s, g in zip(A_state, pos_goals)], np.float32)    # actor action
  G_states = np.stack([np.concatenate([[p], n])
                       for p, n in zip(pos_goals, neg_goals)])  # [N, K] idx

  def batch_scores(goal_mat):
    """goal_mat: [N, K, G] -> scores [N, K]."""
    out = np.zeros((N_ANCHOR, K), np.float32)
    for n in range(N_ANCHOR):
      obs_k = np.concatenate(
          [np.tile(A_state[n], (K, 1)), goal_mat[n]], axis=1)
      acts = np.tile(A_act[n], (K, 1))
      out[n] = np.asarray(_score(jnp.asarray(obs_k), jnp.asarray(acts)))
    return out

  goals0 = states[G_states][:, :, gi]                      # [N, K, G]
  s0 = batch_scores(goals0)
  base_acc = float(np.mean(np.argmax(s0, 1) == 0))
  ranks = np.array([int((s0[n] > s0[n, 0]).sum()) for n in range(N_ANCHOR)])
  rep = {'tag': args.tag, 'env': args.env_name, 'step': int(step),
         'k_way': K, 'chance_acc': 1.0 / K,
         'baseline_acc': base_acc, 'positive_rank_median': float(np.median(ranks)),
         'blocks': {}}

  gpos = {b: [gi.index(i) for i in idx if i in gi]
          for b, idx in ANT_GOAL_BLOCKS.items()}
  for b, p in gpos.items():
    if not p:
      continue
    # knock-out: shuffle block b across candidates (per anchor)
    gk = goals0.copy()
    for n in range(N_ANCHOR):
      gk[n][:, p] = gk[n][rng.permutation(K)][:, p]
    acc_k = float(np.mean(np.argmax(batch_scores(gk), 1) == 0))
    # only-block: shuffle everything EXCEPT block b
    others = [q for q in range(len(gi)) if q not in p]
    go = goals0.copy()
    if others:
      for n in range(N_ANCHOR):
        go[n][:, others] = go[n][rng.permutation(K)][:, others]
      acc_o = float(np.mean(np.argmax(batch_scores(go), 1) == 0))
    else:
      acc_o = base_acc
    rep['blocks'][b] = {'n_dims': len(p),
                        'knockout_acc': acc_k,
                        'reliance': base_acc - acc_k,
                        'only_block_acc': acc_o}
    print(f'  {b:9s} dims={len(p):2d} knockout={acc_k:.3f} '
          f'(reliance {base_acc - acc_k:+.3f})  only-block={acc_o:.3f}')

  path = os.path.join(args.out, f'attr_{args.tag}.json')
  json.dump(rep, open(path, 'w'), indent=2)
  print(f'{args.tag}: base_acc={base_acc:.3f} (chance {1/K:.3f}) '
        f'rank_med={np.median(ranks):.0f} -> {path}')


if __name__ == '__main__':
  main()
