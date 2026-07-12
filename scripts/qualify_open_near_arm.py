"""Near-goal/open-area Ant qualification: run ONE A/B arm.

Config mirrors the 150k antmaze_umaze run exactly (binary NCE, random_goals
0.5, min_replay/random 10k, num_sgd_steps_per_step 4, batch 256, seed 0)
except: env = antmaze_open_near (AntMaze_Open-v5, near goals, 300-step
episodes), 50k steps, and the ONE arm variable entropy_coefficient:
  --arm alpha0       -> entropy_coefficient = 0.0  (faithful baseline)
  --arm adaptive     -> entropy_coefficient = None (adaptive SAC alpha,
                        target_entropy = 0.0, the original's adaptive semantics)
  --arm adaptive_te8 -> entropy_coefficient = None, target_entropy = -8
                        (= -action_dim; direction verified by
                        alpha_direction_sanity.py). Numerical guards ON.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl.train import train


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--arm', choices=['alpha0', 'adaptive', 'adaptive_te8'],
                  required=True)
  ap.add_argument('--steps', type=int, default=50_000)
  ap.add_argument('--out', required=True)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--smoke', action='store_true',
                  help='tiny integration check (1.8k steps)')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  steps = 1_800 if args.smoke else args.steps
  minrep = 600 if args.smoke else 10_000
  cfg = Config(
      env_name='antmaze_open_near', use_td=False, twin_q=False,   # binary NCE
      random_goals=0.5,
      entropy_coefficient=0.0 if args.arm == 'alpha0' else None,
      target_entropy=-8.0 if args.arm == 'adaptive_te8' else 0.0,
      guard_abort=(args.arm == 'adaptive_te8'),
      max_number_of_steps=steps,
      min_replay_size=minrep, random_steps=minrep,
      num_sgd_steps_per_step=4, batch_size=256,
      eval_every_steps=600 if args.smoke else 10_000,
      eval_episodes=2 if args.smoke else 10,
      log_every_steps=600 if args.smoke else 5_000,
      seed=args.seed, ckpt_dir=args.out, resume=False, tensorboard=False,
  )
  assert cfg.use_td is False and cfg.twin_q is False
  print(f'ARM={args.arm} entropy_coefficient={cfg.entropy_coefficient} '
        f'steps={cfg.max_number_of_steps} out={args.out}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
