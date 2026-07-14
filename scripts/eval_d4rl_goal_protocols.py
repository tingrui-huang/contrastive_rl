"""Evaluate one offline-ant checkpoint under the three eval-goal protocols.

Prints a side-by-side table so the FAITHFUL D4RL benchmark number is never
confused with the (harder, non-standard) dataset-goal number again:

  d4rl     -- benchmark goal_sampler: single U_MAZE goal cell (3,1) + per-coord
              noise U(0,0.25*S)+U(0,0.5)*0.25*S, resampled each episode. This is
              the protocol the paper / D4RL score report (mean goal ~ (0.75, 8.75)).
  dataset  -- replay the empirical per-episode infos/goal from the .npz. Collected
              with ~2x the benchmark noise (mean ~ (1.5, 9.5)), so goals sit deeper
              in the maze and it is materially HARDER. Provenance only.
  fixed    -- single fixed goal (0.75, 8.75), held constant across episodes.

Success = any step within 0.5 m of the commanded XY (the env's own sparse reward),
matching d4rl antmaze scoring.

Usage (Colab, on the 1M best checkpoint):
  OFFLINE_NPZ=/content/.../antmaze_umaze_v2_offline.npz \
  python scripts/eval_d4rl_goal_protocols.py \
      --ckpt /content/drive/.../checkpoints/best.pkl --eval_eps 100
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

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from verify_offline_d4rl import build_offline_cfg

ENV_SEED = 12345
MODES = ('d4rl', 'dataset', 'fixed')


def run_mode(cfg, nets, params, mode, episodes):
  cfg.eval_goal_mode = mode
  env = envs_mod.make_env('offline_ant_umaze', cfg, seed=ENV_SEED)

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  succ, mind, find, goals = [], [], [], []
  for _ in range(episodes):
    o = env.reset()
    g = o[29:31].copy()
    goals.append(g)
    dmin = float(np.linalg.norm(o[:2] - g))
    hit = 0.0
    for _ in range(env.max_episode_steps):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, _, _ = env.step(a)
      d = float(np.linalg.norm(o[:2] - g))
      dmin = min(dmin, d)
      hit = max(hit, float(r))
    succ.append(hit)
    mind.append(dmin)
    find.append(d)
  goals = np.asarray(goals)
  return {
      'mode': mode, 'episodes': episodes,
      'success': float(np.mean(succ)),
      'min_dist_mean': float(np.mean(mind)),
      'min_dist_median': float(np.median(mind)),
      'final_dist_mean': float(np.mean(find)),
      'goal_mean': goals.mean(0).tolist(),
      'goal_min': goals.min(0).tolist(),
      'goal_max': goals.max(0).tolist(),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--eval_eps', type=int, default=100)
  ap.add_argument('--out', default=None)
  ap.add_argument('--modes', nargs='+', default=list(MODES))
  args = ap.parse_args()

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=ENV_SEED)  # fills cfg dims
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(args.ckpt)

  rows = [run_mode(cfg, nets, st.policy_params, m, args.eval_eps)
          for m in args.modes]

  print(f'\ncheckpoint: {args.ckpt}  (step {step})  eps={args.eval_eps}')
  print(f'{"mode":9s} {"success":>8s} {"min_med":>8s} {"min_mean":>9s} '
        f'{"goal_mean":>18s}')
  for r in rows:
    gm = r['goal_mean']
    print(f'{r["mode"]:9s} {r["success"]:8.3f} {r["min_dist_median"]:8.3f} '
          f'{r["min_dist_mean"]:9.3f}   ({gm[0]:.2f}, {gm[1]:.2f})')
  print("\nd4rl = FAITHFUL benchmark protocol (compare THIS to the paper's ~0.80).")

  result = {'ckpt': args.ckpt, 'step': int(step), 'eval_eps': args.eval_eps,
            'results': rows}
  if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(result, open(args.out, 'w'), indent=2)
    print('saved', args.out)
  return result


if __name__ == '__main__':
  main()
