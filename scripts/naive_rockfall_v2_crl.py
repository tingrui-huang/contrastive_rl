"""Naive offline CRL STRESS TEST on the v2.1 local-detour primary pilot.

Faithful offline recipe (bc 0.05, twin-min, alpha 0, batch 1024, repr 16,
hidden (1024,1024)) on the 90/0/10 local-detour pilot. Learner sees only
obs/act. Eval runs in offline_ant_umaze_rockfall UNDER THE v2.1 SEVERITY
0.80/0.15/0.05 (cfg.rockfall_severity), so the training-eval success curve
reflects the primary benchmark's lethality. Default 40k steps (stress test).

Usage: python scripts/naive_rockfall_v2_crl.py --steps 40000 --ckpt-dir <dir>
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402
from rockfall_v2_teacher import SEVERITY_V2  # noqa: E402

PILOT_NPZ = 'artifacts/rockfall_v2_dataset/pilot/antmaze_rockfall_v2_pilot.npz'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=40_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='artifacts/naive_rockfall_v2_crl')
  ap.add_argument('--npz', default=PILOT_NPZ)
  ap.add_argument('--p-active', type=float, default=None,
                  help='mask density for eval-during-training (sweep: '
                       '0.30/0.50). None keeps the env default 0.20. Must '
                       'match the density the --npz dataset was collected at.')
  ap.add_argument('--horizon', type=int, default=None,
                  help='episode horizon for eval-during-training (H=800 '
                       'experiment). None keeps the env default 700. Must '
                       'match the horizon the --npz dataset was collected at.')
  ap.add_argument('--reset-fix', action='store_true',
                  help='use the canonical episode-independent full reset for '
                       'eval-during-training (matches the corrected dataset)')
  ap.add_argument('--resume', action='store_true')
  args = ap.parse_args()
  os.makedirs(args.ckpt_dir, exist_ok=True)

  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=args.ckpt_dir)
  cfg.resume = args.resume
  cfg.env_name = 'offline_ant_umaze_rockfall'
  cfg.offline_dataset = args.npz
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_severity = SEVERITY_V2      # v2.1 eval lethality 0.80/0.15/0.05
  if args.p_active is not None:
    cfg.rockfall_p_active = float(args.p_active)  # sweep: match dataset density
  if args.horizon is not None:
    cfg.rockfall_max_steps = int(args.horizon)   # H=800: eval env horizon
    cfg.max_episode_steps = int(args.horizon)
  if args.reset_fix:
    cfg.rockfall_reset_fix = True                # corrected episode reset
  cfg.seed = args.seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  print('naive v2.1 stress test on', args.npz, '| steps', args.steps,
        '| eval severity', SEVERITY_V2, flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
