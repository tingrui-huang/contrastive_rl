"""Offline baselines on the V5 rockfall-clock benchmark.

Port of scripts/train_rockfall_wait_v4_baseline.py: faithful reuse of the
repo's offline AntMaze recipe (verify_offline_d4rl.build_offline_cfg: bc
0.05, twin-min, alpha 0, batch 1024, repr 16, hidden (1024,1024)); the ONLY
changes are the env id, the dataset and the 400-step horizon (matches
collection). Nothing about the algorithm is adapted to the rockfall
timetable: the schedule never enters the 58-dim obs, every episode starts
from one canonical pose, and whether the ant stops at the mouth (or takes
the detour) is the plain goal-conditioned actor's own output.

Variants pick the expert dataset the learner imitates:
  --variant near    expert always takes the BR shortcut (p_far 0)
  --variant far05   expert takes the V3-br detour with probability 0.05
                    (pure coverage of the safe alternative; the route coin
                    carries no information about the latent)

Methods:
  --method crl    vanilla contrastive RL (bc_coef 0.05)   [default]
  --method gcbc   goal-conditioned BC (bc_coef 1.0)

The G1-G8 offline audit gates run inside train() and abort on any failure.

Usage:
  python scripts/train_rockfall_clock_v5_baseline.py --variant far05 \
      --method crl --steps 100000 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.train import train                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ENV_NAME = 'offline_ant_umaze_rockfall_clock_v5'
DATASET_DIR = 'artifacts/rockfall_clock_v5/dataset'
#: --variant -> collector npz (scripts/collect_rockfall_clock_v5_dataset.py
#: names the p_far 0 file without a suffix and the far05 one with it).
NPZ_BY_VARIANT = {
    'near': f'{DATASET_DIR}/antmaze_rockfall_clock_v5.npz',
    'far05': f'{DATASET_DIR}/antmaze_rockfall_clock_v5_far05.npz',
}
HORIZON = 400


def main():
  ap = argparse.ArgumentParser()
  #: far05 is the headline variant: it carries the 5% latent-independent
  #: detour demonstrations, so the dataset contains a safe alternative.
  ap.add_argument('--variant', choices=sorted(NPZ_BY_VARIANT),
                  default='far05')
  ap.add_argument('--method', choices=['crl', 'gcbc'], default='crl')
  ap.add_argument('--steps', type=int, default=100_000)
  ap.add_argument('--seed', type=int, default=0)
  #: Overrides the variant's npz (probes only; the variant still names the run).
  ap.add_argument('--npz', default=None)
  ap.add_argument('--ckpt-dir', default=None)
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  #: CRL's reachability horizon (goal relabel offset ~ discount**d).
  ap.add_argument('--discount', type=float, default=None)
  #: 'xy' (DEFAULT) = the upstream ant contract (lp_contrastive sets
  #: end_index=2 for every 'ant_' env): the relabeled goal and the commanded
  #: goal are both the XY pair. 'full' = this port's historical 29-dim goal,
  #: which commands XY + 27 zeros at evaluation time -- a vector no training
  #: goal ever has, worth ~0.3 success. Kept only to reproduce old runs.
  ap.add_argument('--goal-rep', choices=['full', 'xy'], default='xy')
  args = ap.parse_args()

  #: the XY-goal arm trains on the 31-column copy of the same dataset
  #: (scripts/make_v5_gxy_dataset.py: a column selection, same states
  #: and actions), because the buffer stores the env observation.
  npz = args.npz or NPZ_BY_VARIANT[args.variant]
  env_name = ENV_NAME + ('_gxy' if args.goal_rep == 'xy' else '')
  if args.goal_rep == 'xy' and args.npz is None:
    npz = npz.replace('.npz', '_gxy.npz')
  suffix = '_gxy' if args.goal_rep == 'xy' else ''
  run_id = (args.ckpt_dir or
            f'v5clock_{args.variant}_{args.method}_s{args.seed}_'
            f'{args.steps // 1000}k{suffix}')
  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_id)
  cfg.resume = args.resume
  cfg.env_name = env_name
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
  print(f'rockfall-clock v5 baseline {args.variant} {args.method} | '
        f'steps {args.steps} | seed {args.seed} | bc_coef {cfg.bc_coef} | '
        f'goal {args.goal_rep} ({env_name}) | '
        f'npz {npz} | -> {run_id}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
