"""Diagnostic 2: train the FAITHFUL offline CRL recipe on the litter dataset.

One seed, sanity check (NOT the final sweep). Uses ONLY the learner npz
(obs/act); no sidecar, no U, no teacher mode, no privileged labels. Trains
on artifacts/litter_dataset/full/antmaze_litter_full.npz and evaluates in
offline_ant_umaze_litter (collapse ON) so behaviour (lane / collapse) is
visible. Standard train.py periodic eval reports success; the behavioural
characterization (lane / speed / collapse / per-U) is done separately by
scripts/eval_naive_litter.py on the saved checkpoints.

Usage: python scripts/naive_litter_crl.py --steps 60000 --ckpt-dir <dir>
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

LITTER_NPZ = 'artifacts/litter_dataset/full/antmaze_litter_full.npz'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=60_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='artifacts/naive_litter_crl')
  ap.add_argument('--npz', default=LITTER_NPZ)
  args = ap.parse_args()
  os.makedirs(args.ckpt_dir, exist_ok=True)

  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=args.ckpt_dir)
  # train + EVAL on the litter env (collapse ON) so lane/collapse are visible;
  # dataset is the litter learner npz (obs/act only). Everything else is the
  # byte-identical faithful offline recipe (bc 0.05, twin-min, alpha 0, etc.).
  cfg.env_name = 'offline_ant_umaze_litter'
  cfg.offline_dataset = args.npz
  cfg.eval_goal_mode = 'd4rl'
  cfg.seed = args.seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  print('naive offline CRL on litter:', args.npz, '| steps', args.steps,
        '| eval env offline_ant_umaze_litter (collapse ON)', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
