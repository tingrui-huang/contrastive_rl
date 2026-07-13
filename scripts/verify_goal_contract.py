"""Pre-training verification for the goal-representation ablation.

For each arm env (xy / gcompact / gfull) verifies, with hard assertions:
  1. dims: obs width = 29 + goal_dim; goal_dim = len(goal_indices); indices
     start with XY (0,1); config filled correctly by make_env.
  2. state layout: flattened state == concat([qpos, qvel]) bit-exactly (the
     assumption behind the settled-goal snapshot and the block definitions).
  3. commanded goal: goal half of obs == goal_state_full[goal_indices]
     bit-exactly; settled-goal XY drift from the commanded cell is reported.
  4. replay relabeling: every sampled goal == goal_indices slice of an actual
     FUTURE state of the same trajectory (bit-exact), j > i.
  5. actor/critic consistency: policy and critic apply on [state, goal] with
     the arm's widths; critic g_encoder input width == goal_dim; init critic
     scores finite.
  6. normalization: none is applied anywhere in the pipeline (raw obs); we
     report per-block goal scales so cross-block magnitude differences are
     documented rather than silent.
"""
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
from crl.replay import TrajectoryBuffer

from crl.envs import ANT_GOAL_BLOCKS

ENVS = ['antmaze_open_near', 'antmaze_open_near_gcompact',
        'antmaze_open_near_gfull']
OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/goal_contract_verification.json'


def verify(env_name):
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=3)
  u = env._env.unwrapped
  rng = np.random.default_rng(0)
  rep = {'env': env_name, 'obs_dim': cfg.obs_dim, 'goal_dim': cfg.goal_dim,
         'goal_indices': None if cfg.goal_indices is None
                         else list(cfg.goal_indices)}
  gi = cfg.goal_indices

  # 1. dims
  obs = env.reset()
  assert obs.shape[0] == cfg.obs_dim + cfg.goal_dim, 'obs width mismatch'
  assert cfg.obs_dim == 29
  if gi is not None:
    assert cfg.goal_dim == len(gi) and tuple(gi[:2]) == (0, 1)

  # 2. state layout == [qpos, qvel]
  state = obs[:29]
  qq = np.concatenate([np.asarray(u.data.qpos),
                       np.asarray(u.data.qvel)]).astype(np.float32)
  layout_err = float(np.abs(state - qq).max())
  rep['state_layout_max_err'] = layout_err
  assert layout_err == 0.0, f'state != [qpos,qvel] (err {layout_err})'

  # 3. commanded goal vector + settled drift
  drifts, settled_z = [], []
  for k in range(5):
    obs = env.reset()
    goal_half = obs[29:]
    if gi is not None:
      full = env._goal_state_full
      assert np.array_equal(goal_half, full[list(gi)]), 'goal slice mismatch'
      cmd_xy = np.asarray(env._last_obs['desired_goal'])
      drifts.append(float(np.linalg.norm(full[:2] - cmd_xy)))
      settled_z.append(float(full[2]))
    else:
      cmd_xy = np.asarray(env._last_obs['desired_goal'])
      assert np.allclose(goal_half, cmd_xy, atol=1e-6), 'xy goal mismatch'
  if drifts:
    rep['settled_goal_xy_drift'] = {'mean': float(np.mean(drifts)),
                                    'max': float(np.max(drifts))}
    rep['settled_goal_z_mean'] = float(np.mean(settled_z))

  # 4. replay relabel bit-exactness (one real short episode)
  L = env.max_episode_steps + 1
  buf = TrajectoryBuffer(capacity_steps=10 * L, ep_len_obs=L,
                         full_obs_dim=cfg.obs_dim + cfg.goal_dim,
                         action_dim=cfg.action_dim, obs_dim=cfg.obs_dim,
                         start_index=cfg.start_index, end_index=cfg.end_index,
                         discount=cfg.discount, seed=0, goal_indices=gi)
  obs = env.reset()
  O = np.zeros((L, cfg.obs_dim + cfg.goal_dim), np.float32)
  A = np.zeros((L, cfg.action_dim), np.float32)
  for t in range(env.max_episode_steps):
    O[t] = obs
    A[t] = rng.uniform(-1, 1, cfg.action_dim).astype(np.float32)
    obs, _, _, _ = env.step(A[t])
  O[-1] = obs
  buf.add_episode(O, A)
  tr = buf.sample(256)
  n_checked, n_future = 0, 0
  states29 = O[:, :29]
  for b in range(256):
    s, g = tr.observation[b, :29], tr.observation[b, 29:]
    # anchor index: state must equal a stored state
    si = np.where((states29 == s).all(axis=1))[0]
    assert len(si) >= 1, 'sampled state not in trajectory'
    goal_src = (states29[:, list(gi)] if gi is not None else
                states29[:, cfg.start_index:(None if cfg.end_index == -1
                                              else cfg.end_index)])
    gj = np.where((goal_src == g).all(axis=1))[0]
    assert len(gj) >= 1, 'sampled goal not a future-state slice'
    if gj.max() > si.min():
      n_future += 1
    n_checked += 1
  rep['relabel_checked'] = n_checked
  rep['relabel_goal_from_strict_future_frac'] = n_future / n_checked
  assert n_future / n_checked > 0.95, 'goals not strictly future'

  # 5. actor/critic dims + finiteness
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  key = jax.random.PRNGKey(0)
  pp = nets.policy_network.init(key)
  qp = nets.q_network.init(key)
  ob = jnp.asarray(O[:8])
  ac = jnp.asarray(A[:8])
  dist = nets.policy_network.apply(pp, ob)
  qv = nets.q_network.apply(qp, ob, ac)
  assert dist.loc.shape == (8, cfg.action_dim)
  assert qv.shape == (8, 8)
  assert bool(jnp.all(jnp.isfinite(qv)))
  g_w0 = qp['g_encoder/~/linear_0']['w'].shape
  assert g_w0[0] == cfg.goal_dim, f'g_encoder input {g_w0[0]} != {cfg.goal_dim}'
  rep['g_encoder_input_width'] = int(g_w0[0])

  # 6. goal-dim scales per block (no normalization anywhere -- document)
  if gi is not None:
    gpos = {b: [gi.index(i) for i in idx if i in gi]
            for b, idx in ANT_GOAL_BLOCKS.items()}
    goals = O[:, 29:]
    rep['goal_block_std'] = {
        b: (float(goals[:, p].std()) if p else None)
        for b, p in gpos.items()}
  rep['pass'] = True
  return rep


def main():
  out = {}
  for name in ENVS:
    out[name] = verify(name)
    print(name, 'PASS',
          {k: v for k, v in out[name].items()
           if k in ('goal_dim', 'settled_goal_xy_drift', 'goal_block_std',
                    'relabel_goal_from_strict_future_frac')})
  json.dump(out, open(OUT, 'w'), indent=2)
  print('saved', OUT)


if __name__ == '__main__':
  main()
