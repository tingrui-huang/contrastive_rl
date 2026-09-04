"""Offline baselines on the V4 rockfall-wait benchmark.

Port of scripts/train_tworoute_v3_baseline.py: faithful reuse of the repo's
offline AntMaze recipe (verify_offline_d4rl.build_offline_cfg: bc 0.05,
twin-min, alpha 0, batch 1024, repr 16, hidden (1024,1024)); the ONLY
changes are the env id, the dataset and the 400-step horizon (matches
collection). Nothing about the algorithm is adapted to the wait decision:
every episode starts from one canonical pose and whether the ant stops at
the mouth is the plain goal-conditioned actor's own output.

Methods:
  --method crl    vanilla contrastive RL (bc_coef 0.05)   [default]
  --method gcbc   goal-conditioned BC (bc_coef 1.0)

The G1-G8 offline audit gates run inside train() and abort on any failure.

Usage:
  python scripts/train_rockfall_wait_v4_baseline.py --method crl \
      --steps 100000 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ENV_NAME = 'offline_ant_umaze_rockfall_wait_v4'
NPZ = 'artifacts/rockfall_wait_v4/dataset/antmaze_rockfall_wait_v4.npz'
HORIZON = 400


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--method', choices=['crl', 'gcbc'], default='crl')
  ap.add_argument('--steps', type=int, default=100_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--ckpt-dir', default=None)
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  #: CRL's reachability horizon (goal relabel offset ~ discount**d).
  ap.add_argument('--discount', type=float, default=None)
  args = ap.parse_args()

  run_id = (args.ckpt_dir or
            f'v4wait_{args.method}_s{args.seed}_{args.steps // 1000}k')
  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_id)
  cfg.resume = args.resume
  cfg.env_name = ENV_NAME
  cfg.offline_dataset = args.npz
  cfg.eval_goal_mode = 'd4rl'
  if args.discount is not None:
    cfg.discount = float(args.discount)
  cfg.rockfall_max_steps = args.horizon
  cfg.max_episode_steps = args.horizon
  cfg.bc_coef = 1.0 if args.method == 'gcbc' else 0.05
  cfg.seed = args.seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  print(f'rockfall-wait v4 baseline {args.method} | steps {args.steps} | '
        f'seed {args.seed} | bc_coef {cfg.bc_coef} | npz {args.npz} | '
        f'-> {run_id}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
