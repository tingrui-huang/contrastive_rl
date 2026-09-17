"""Step 12: critics on the branch replay (interventional futures for every anchor), two arms.

The fair version of Step 11's finding (see scripts/build_f4_branch_replay.py):
the critic's positive futures come from the fixed ETT rolled out of EVERY
recorded anchor -- no provenance, no state selection, no outcome selection
-- and the anchors are the first row of every path, i.e. exactly the
recorded (s, a) rows plus the queried (s, a) rows of arms C, D, E.  Vanilla
CRL differs only in whose continuation follows the anchor.

Arms (replay_P, the sealed 30k critic recipe, five seeds each):
  P         every batch = 256 path-first-row anchors, uniform over paths
  P_strat   128 uniform + 64 fork-DOWN + 64 fork-RIGHT first rows (Step 11's
            composition), to separate the replay from the stratification
  P_frozen  P on replay_P_frozen.npz: recorded stationary anchors (the F4
            death signature, t >= 4) stay absorbed instead of being handed
            to the model, which moves out of 87% of them -- an ablation of
            that one model defect, not the uniform rule

Gate, actors, evaluation and summary are Step 11's, unchanged: the width-0.75
margin on the Step 10b jitter rows of replay_E >= +0.10 before any actor;
three fresh balanced-BC actors per passing critic on replay_E (the actor
recipe never changes); 300 new-seed episodes.

Stages: train | gate | actors | summarize | all.
Outputs: outputs/pointmaze_branch_replay_v1/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import run_f4_fork_strata as s11                                   # noqa: E402
from run_f4_corner_coverage_fix import CRL_STEPS, BC_COEF, sha256, write_json   # noqa: E402

OUT = Path(os.environ.get('F4_BRANCH_OUT', ROOT / 'outputs' / 'pointmaze_branch_replay_v1'))
REPLAY_P = Path(os.environ.get('F4_BRANCH_REPLAY', OUT / 'replay_P.npz'))
REPLAYS = {'P': REPLAY_P, 'P_strat': REPLAY_P,
           'P_frozen': Path(os.environ.get('F4_BRANCH_REPLAY_FROZEN', OUT / 'replay_P_frozen.npz'))}
s11.OUT = OUT                       # the shared stages (gate / actors / summarize) write here
s11.TITLE = '# Step 12: critics on the branch replay (interventional futures for every anchor)'
ARMS = ('P', 'P_strat', 'P_frozen')
COUNTS = s11.COUNTS
CRITIC_SEEDS = s11.CRITIC_SEEDS


def build_strata(path, arm):
  """Anchors = row 0 of every path.  P: one stratum, uniform over paths.
  P_strat: + fork-DOWN and fork-RIGHT first rows as their own strata."""
  with np.load(path, allow_pickle=False) as d:
    obs, act = d['obs'][:, 0], d['act'][:, 0]
  n = len(obs)
  e = np.arange(n)
  i = np.zeros(n, np.int64)
  cell = np.floor(obs[:, :2]).astype(np.int64)
  fork = (cell[:, 0] == s11.FORK[0]) & (cell[:, 1] == s11.FORK[1])
  sec = s11.sectors(act)
  if arm in ('P', 'P_frozen'):
    strata, counts = [(e, i, None)], (256,)
  elif arm == 'P_strat':
    strata = [(e, i, None), (e[fork & sec['DOWN']], i[fork & sec['DOWN']], None),
              (e[fork & sec['RIGHT']], i[fork & sec['RIGHT']], None)]
    counts = COUNTS
  else:
    raise ValueError(arm)
  stats = {'arm': arm, 'paths': int(n), 'counts': list(counts),
           'stratum_rows': [int(len(s[0])) for s in strata],
           'fork_first_rows': int(fork.sum()), 'fork_share_uniform': float(fork.mean()),
           'fork_down_first_rows': int((fork & sec['DOWN']).sum()),
           'fork_right_first_rows': int((fork & sec['RIGHT']).sum())}
  return strata, counts, stats


def prepare_hook(arm):
  def prepare(buffer, path):
    strata, counts, stats = build_strata(path, arm)
    buffer.set_anchor_strata(strata, counts)
    print(f'ANCHOR STRATA ({arm}): first rows of {stats["paths"]:,} paths, counts {counts}, '
          f'fork first rows {stats["fork_first_rows"]:,} ({stats["fork_share_uniform"]:.3f})', flush=True)
    return stats
  return prepare


def crl_config(arm, seed, ckpt_dir, steps):
  cfg = s11.crl_config(seed, ckpt_dir, steps)
  cfg.offline_dataset = str(REPLAYS[arm])
  return cfg


def train_one(arm, seed, steps):
  from crl.train import train
  out = s11.critic_dir(arm, seed)
  if (out / 'final.pkl').exists():
    print(f'{out} exists', flush=True)
    return
  cfg = crl_config(arm, seed, out, steps)
  t0 = time.time()
  train(cfg, buffer_prepare=prepare_hook(arm))
  _, counts, stats = build_strata(REPLAYS[arm], arm)
  write_json(out / 'run_summary.json', {
      'arm': arm, 'seed': seed, 'dataset': str(REPLAYS[arm]), 'dataset_sha256': sha256(REPLAYS[arm]),
      'steps': int(steps), 'batch': 256, 'bc_coef': BC_COEF, 'anchor_strata': stats,
      'wall_seconds': time.time() - t0})


def stage_train(arms, seeds, steps, parallel):
  procs = []
  for arm in arms:
    for seed in seeds:
      if s11.critic_path(arm, seed).exists():
        continue
      log = OUT / 'logs' / f'critic_{arm}_seed{seed}.log'
      log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, str(Path(__file__).resolve()), '_train_one', '--arms', arm,
             '--seeds', str(seed), '--steps', str(steps)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, s11.critic_dir(arm, seed)))
      if len(procs) >= parallel:
        s11._wait(procs)
  s11._wait(procs)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('train', 'gate', 'actors', 'summarize', 'all', '_train_one'))
  ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
  ap.add_argument('--seeds', type=int, nargs='+', default=list(CRITIC_SEEDS))
  ap.add_argument('--steps', type=int, default=CRL_STEPS)
  ap.add_argument('--actor-steps', type=int, default=0)
  ap.add_argument('--episodes', type=int, default=s11.NEW_EVAL['episodes'])
  ap.add_argument('--parallel', type=int, default=3)
  ap.add_argument('--all-critics', action='store_true')
  args = ap.parse_args(argv)
  if args.command == '_train_one':
    train_one(args.arms[0], args.seeds[0], args.steps)
    return 0
  stages = ('train', 'gate', 'actors', 'summarize') if args.command == 'all' else (args.command,)
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'train':
      stage_train(args.arms, args.seeds, args.steps, args.parallel)
    elif st == 'gate':
      s11.stage_gate(args.arms, args.seeds)
    elif st == 'actors':
      s11.stage_actors(args.arms, args.seeds, args.parallel, args.all_critics, args.actor_steps, args.episodes)
    elif st == 'summarize':
      s11.stage_summarize(args.arms, args.seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
