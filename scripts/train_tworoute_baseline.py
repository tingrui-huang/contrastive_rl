"""Offline baselines on the two-route AntMaze rockfall dataset.

Faithful reuse of the repo's offline AntMaze recipe (verify_offline_d4rl.
build_offline_cfg: bc 0.05, twin-min, alpha 0, batch 1024, repr 16, hidden
(1024,1024)); the ONLY changes are the env name, the dataset, the 400-step
horizon (matches collection) and the learner-eval heading protocol
(reset(heading='random'): a 50/50 route-affordance coin independent of the
hidden latent).

Methods:
  --method crl    vanilla contrastive RL (bc_coef 0.05)   [default]
  --method gcbc   goal-conditioned BC (bc_coef 1.0: the actor loss is pure
                  behaviour cloning; the critic still trains but has zero
                  weight in the actor objective)

The G1-G8 offline audit gates run inside train() and abort on any failure.

Usage:
  python scripts/train_tworoute_baseline.py --method crl  --steps 300000 --seed 0
  python scripts/train_tworoute_baseline.py --method gcbc --steps 300000 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

NPZ = ('artifacts/tworoute_rockfall_v0/dataset/'
       'antmaze_tworoute_rockfall_v1.npz')
HORIZON = 400


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--method', choices=['crl', 'gcbc'], default='crl')
  ap.add_argument('--steps', type=int, default=300_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--ckpt-dir', default=None)
  ap.add_argument('--resume', action='store_true')
  args = ap.parse_args()

  run_id = (args.ckpt_dir or
            f'tworoute_{args.method}_s{args.seed}_{args.steps // 1000}k')
  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_id)
  cfg.resume = args.resume
  cfg.env_name = 'offline_ant_umaze_tworoute_rockfall'
  cfg.offline_dataset = args.npz
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON             # match the collection horizon
  cfg.max_episode_steps = HORIZON
  cfg.tworoute_eval_heading = 'random'         # learner-eval protocol
  cfg.bc_coef = 1.0 if args.method == 'gcbc' else 0.05
  cfg.seed = args.seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  print(f'two-route baseline {args.method} | steps {args.steps} | '
        f'seed {args.seed} | bc_coef {cfg.bc_coef} | npz {args.npz} | '
        f'-> {run_id}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
