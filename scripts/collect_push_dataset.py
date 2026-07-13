"""Collect a FROZEN offline dataset from a fetch push env (state or image).

Mirrors the WindyCorridor data layer: a scripted demonstrator generates N
episodes ONCE; training then runs on the fixed .npz with zero env interaction
(`--offline_dataset` in crl.train). The behavior policy is the report_push
oracle (lift->over->descend->push, reads SIM state, so it works for image obs
too) with Gaussian action noise, plus a fraction of fully-random episodes for
action-space coverage.

Output .npz:
  obs  [N, L, obs_dim+goal_dim]  (uint8 for image envs, float32 otherwise)
  act  [N, L, action_dim]        (float32; act[:, -1] is a dummy zero row,
                                  matching the online collect_episode layout)
  meta (json string: env, seed, noise, random_frac, behavior success)

Run:
  python -m scripts.collect_push_dataset --env_name fetch_push_image_conedir \
      --episodes 1000 --noise 0.3 --random_frac 0.2 --seed 0 \
      --out datasets/push_image_conedir_noisy_oracle_s0.npz
"""
import argparse
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_push import _oracle_action


def collect(env_name, episodes, noise, random_frac, seed, out):
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  u = env._env.unwrapped
  rng = np.random.default_rng(seed)
  L = env.max_episode_steps + 1
  D = cfg.obs_dim + cfg.goal_dim
  A = cfg.action_dim
  obs_dtype = np.uint8 if cfg.use_image_obs else np.float32

  obs_out = np.zeros((episodes, L, D), dtype=obs_dtype)
  act_out = np.zeros((episodes, L, A), dtype=np.float32)
  succ, kinds = [], []
  n_random = int(round(episodes * random_frac))

  for ep in range(episodes):
    random_ep = ep < n_random          # leading block; sampling is iid anyway
    kinds.append('random' if random_ep else 'noisy_oracle')
    obs = env.reset()
    hit = 0.0
    for t in range(env.max_episode_steps):
      obs_out[ep, t] = obs
      if random_ep:
        a = rng.uniform(-1.0, 1.0, size=A).astype(np.float32)
      else:
        o = u._get_obs()               # behavior policy reads SIM state
        vec = np.concatenate([o['observation'],
                              o['desired_goal']]).astype(np.float32)
        a = _oracle_action(vec)
        a = np.clip(a + rng.normal(0.0, noise, size=A),
                    -1.0, 1.0).astype(np.float32)
      act_out[ep, t] = a
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
    obs_out[ep, -1] = obs
    succ.append(hit)
    if (ep + 1) % 50 == 0:
      print(f'  {ep + 1}/{episodes} episodes '
            f'(behavior success so far {np.mean(succ):.3f})', flush=True)

  succ = np.asarray(succ)
  is_orc = np.asarray([k == 'noisy_oracle' for k in kinds])
  meta = {
      'env_name': env_name, 'episodes': episodes, 'seed': seed,
      'noise': noise, 'random_frac': random_frac,
      # Self-describing dims so the offline audit needs no env to validate.
      'obs_dim': cfg.obs_dim, 'goal_dim': cfg.goal_dim, 'action_dim': A,
      'start_index': cfg.start_index, 'end_index': cfg.end_index,
      'goal_indices': (None if cfg.goal_indices is None
                       else list(cfg.goal_indices)),
      'use_image_obs': bool(cfg.use_image_obs),
      'max_episode_steps': env.max_episode_steps,
      'behavior_success': float(succ.mean()),
      'behavior_success_oracle': float(succ[is_orc].mean()) if is_orc.any() else None,
      'behavior_success_random': float(succ[~is_orc].mean()) if (~is_orc).any() else None,
  }
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  # obs/act are the ONLY learner tensors; per-episode kinds go to an AUDIT-ONLY
  # field (audit_*), which the offline audit verifies never enters the learner.
  audit_kind = np.array([0 if k == 'noisy_oracle' else 1 for k in kinds],
                        dtype=np.int64)
  np.savez_compressed(out, obs=obs_out, act=act_out,
                      audit_behavior_kind=audit_kind,
                      meta=np.array(json.dumps(meta)))
  size_mb = os.path.getsize(out) / 1e6
  print(f'\nDataset -> {out} ({size_mb:.0f} MB)')
  print(json.dumps(meta, indent=2))
  return meta


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--env_name', default='fetch_push_image_conedir')
  p.add_argument('--episodes', type=int, default=1000)
  p.add_argument('--noise', type=float, default=0.3)
  p.add_argument('--random_frac', type=float, default=0.2)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', required=True)
  a = p.parse_args()
  collect(a.env_name, a.episodes, a.noise, a.random_frac, a.seed, a.out)


if __name__ == '__main__':
  main()
