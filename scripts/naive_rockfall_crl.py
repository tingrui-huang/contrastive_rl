"""Rockfall diagnostic: train the FAITHFUL offline CRL recipe on the ~300-ep
rockfall pilot dataset (short sanity run, NOT the final sweep). Uses ONLY the
learner npz (obs/act); no sidecar, no mask, no privileged labels. Evaluates
in offline_ant_umaze_rockfall so route choice / collapse are visible.

Usage: python scripts/naive_rockfall_crl.py --steps 40000 --ckpt-dir <dir>
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

PILOT_NPZ = 'artifacts/rockfall_dataset/pilot/antmaze_rockfall_pilot.npz'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=40_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='artifacts/naive_rockfall_crl')
  ap.add_argument('--npz', default=PILOT_NPZ)
  ap.add_argument('--resume', action='store_true')
  args = ap.parse_args()
  os.makedirs(args.ckpt_dir, exist_ok=True)

  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=args.ckpt_dir)
  cfg.resume = args.resume
  cfg.env_name = 'offline_ant_umaze_rockfall'
  cfg.offline_dataset = args.npz
  cfg.eval_goal_mode = 'd4rl'
  cfg.seed = args.seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  print('naive offline CRL on rockfall pilot:', args.npz, '| steps',
        args.steps, '| eval env offline_ant_umaze_rockfall', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
