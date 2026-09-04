"""Offline baselines on the V3 two-route rockfall pair.

Port of scripts/train_tworoute_baseline.py: faithful reuse of the repo's
offline AntMaze recipe (verify_offline_d4rl.build_offline_cfg: bc 0.05,
twin-min, alpha 0, batch 1024, repr 16, hidden (1024,1024)); the ONLY
changes are the env id, the dataset and the 400-step horizon (matches
collection). Nothing about the algorithm is adapted to the two-route
decision: every episode starts from one canonical pose, so the route is the
plain goal-conditioned actor's own output.

--variant picks the goal corner of the controlled pair (identical sparse
reference numbers -- shortcut 0.70 / detour 0.96 / oracle 0.988; only the
discounted incentive differs):
  tr  goal (8,8): equal-length routes, discounted objective ~indifferent
  br  goal (8,0): shortcut 2.9x shorter, discounted objective prefers the
      hazardous route

Methods:
  --method crl    vanilla contrastive RL (bc_coef 0.05)   [default]
  --method gcbc   goal-conditioned BC (bc_coef 1.0: the actor loss is pure
                  behaviour cloning; the critic still trains but has zero
                  weight in the actor objective)

The G1-G8 offline audit gates run inside train() and abort on any failure.

Usage:
  python scripts/train_tworoute_v3_baseline.py --variant tr --method crl \
      --steps 300000 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

NPZ_TMPL = ('artifacts/tworoute_rockfall_v3/{variant}/dataset/'
            'antmaze_tworoute_rockfall_v3{variant}.npz')
HORIZON = 400


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=['tr', 'br'], required=True)
  ap.add_argument('--method', choices=['crl', 'gcbc'], default='crl')
  ap.add_argument('--steps', type=int, default=300_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--npz', default=None)
  ap.add_argument('--ckpt-dir', default=None)
  ap.add_argument('--resume', action='store_true')
  #: episode horizon. The default matches the collection horizon; raising
  #: it lowers EXECUTION difficulty for the long detour without touching the
  #: gamma=0.99 incentive references, which is the horizon-600 manipulation.
  ap.add_argument('--horizon', type=int, default=HORIZON)
  #: CRL's reachability horizon. The critic relabels goals at offset d with
  #: probability ~ discount**d (crl/replay.py), so this sets what the
  #: objective considers reachable -- the knob that decides whether the
  #: 236-step BR detour is valued at all. Default matches build_offline_cfg.
  ap.add_argument('--discount', type=float, default=None)
  args = ap.parse_args()

  npz = args.npz or NPZ_TMPL.format(variant=args.variant)
  run_id = (args.ckpt_dir or
            f'v3{args.variant}_{args.method}_s{args.seed}_'
            f'{args.steps // 1000}k')
  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_id)
  cfg.resume = args.resume
  cfg.env_name = f'offline_ant_umaze_tworoute_rockfall_v3{args.variant}'
  cfg.offline_dataset = npz
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
  print(f'two-route v3{args.variant} baseline {args.method} | '
        f'steps {args.steps} | seed {args.seed} | bc_coef {cfg.bc_coef} | '
        f'npz {npz} | -> {run_id}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
