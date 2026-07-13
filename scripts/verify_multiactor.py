"""4-actor integration test for the strict-original multi-actor path.

Runs a short REAL training (d4rl-faithful env, strict alpha=0, 4 actors,
replay snapshot on) and then verifies from the artifacts alone:
  1. all four actors contribute episodes to replay (insertion order is
     block-major/actor-minor, so episode k belongs to actor k % 4);
  2. actor trajectory/reset diversity: distinct start cells and goals across
     actors within blocks, disjoint-ish visitation;
  3. no duplicated RNG streams: warmup actions differ pairwise across actors;
  4. total-step accounting: env_steps = blocks * 4 * 700; per-actor = /4;
  5. learner-update ratio: 1 update per TOTAL post-warmup env step;
  6. checkpoint + replay resume: a second train() continues from the saved
     step with the snapshot restored (episode count preserved).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl.train import train

OUT = os.path.join(os.path.dirname(_HERE), 'artifacts',
                   'multiactor_integration.json')
N_ACT = 4
EP = 700
BLOCK = N_ACT * EP                       # 2800 total steps per block


def build_cfg(run_dir, steps, resume):
  return Config(
      env_name='d4rl_ant_umaze_gfull', use_td=False, twin_q=False,
      random_goals=0.5,
      entropy_coefficient=0.0,           # STRICT original
      guard_abort=True,
      num_actors=N_ACT, save_replay=True,
      max_number_of_steps=steps,
      min_replay_size=2 * BLOCK, random_steps=2 * BLOCK,
      num_sgd_steps_per_step=4, batch_size=256,
      eval_every_steps=2 * BLOCK, eval_episodes=2, log_every_steps=BLOCK,
      seed=0, ckpt_dir=run_dir, resume=resume, tensorboard=False)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_dir', default=os.path.join(
      os.path.dirname(_HERE), 'artifacts', 'multiactor_itest_run'))
  args = ap.parse_args()
  if os.path.exists(args.run_dir):
    shutil.rmtree(args.run_dir)
  os.makedirs(args.run_dir)
  rep = {}

  # ---- phase 1: 4 blocks = 11200 total steps ----
  train(build_cfg(args.run_dir, 4 * BLOCK, resume=False))
  # copy arrays out and CLOSE the npz -- train() runs in this same process and
  # must be able to os.replace this file at the end of phase 2 (Windows lock).
  with np.load(os.path.join(args.run_dir, 'replay.npz')) as d:
    n_eps = int(d['num_eps'])
    obs = d['obs'].copy()                 # [n_eps, 701, 58]
    act = d['act'].copy()
  assert n_eps == 16, f'expected 16 episodes (4 blocks x 4 actors), got {n_eps}'

  # 1+2: per-actor contribution + reset/goal diversity
  starts = obs[:, 0, :2]
  goals = obs[:, 0, 29:31]
  div = []
  for b in range(4):
    blk = slice(4 * b, 4 * b + 4)
    s, g = starts[blk], goals[blk]
    div.append({'distinct_starts': int(len(np.unique(np.round(s, 2), axis=0))),
                'distinct_goals': int(len(np.unique(np.round(g, 2), axis=0)))})
  assert all(x['distinct_starts'] >= 3 for x in div), div
  rep['per_block_diversity'] = div
  cells = [set() for _ in range(N_ACT)]
  for k in range(n_eps):
    a = k % N_ACT
    xy = obs[k, :, :2]
    cells[a].update({(round(float(x) / 2), round(float(y) / 2))
                     for x, y in xy})
  overlap = [[round(len(cells[i] & cells[j]) / max(len(cells[i] | cells[j]), 1), 2)
              for j in range(N_ACT)] for i in range(N_ACT)]
  rep['per_actor_unique_cells'] = [len(c) for c in cells]
  rep['pairwise_cell_jaccard'] = overlap

  # 3: RNG streams (warmup blocks are uniform-random per actor)
  for i in range(N_ACT):
    for j in range(i + 1, N_ACT):
      assert not np.array_equal(act[i, :50], act[j, :50]), \
          f'actors {i},{j} share a warmup RNG stream'
  rep['rng_streams_distinct'] = True

  # 4+5: accounting + update ratio from metrics.json
  mets = json.load(open(os.path.join(args.run_dir, 'metrics.json')))
  last = mets[-1]
  assert last['num_actors'] == N_ACT
  assert last['per_actor_steps'] * N_ACT == last['step']
  assert last['step'] == 4 * BLOCK, last['step']
  # Learning fires from the block where the buffer first reaches min_replay
  # (block 2 here, catching up on warmup data -- the original SampleToInsert
  # limiter also allows cumulative updates up to 1x TOTAL inserts). Steady
  # state = exactly BLOCK updates per BLOCK total steps.
  first_learn_block = -(-1 * (2 * BLOCK) // BLOCK)          # ceil = 2
  n_learn_blocks = last['step'] // BLOCK - first_learn_block + 1
  expected = n_learn_blocks * BLOCK
  assert last['learner_updates'] == expected, \
      (last['learner_updates'], expected)
  assert last['learner_updates'] <= last['step'], 'exceeds original cum. limit'
  rep['accounting'] = {'total_steps': last['step'],
                       'per_actor_steps': last['per_actor_steps'],
                       'learner_updates': last['learner_updates'],
                       'steady_state_updates_per_block': BLOCK,
                       'cumulative_ratio_vs_total': round(
                           last['learner_updates'] / last['step'], 3)}

  # 6: resume with replay restore, +2 blocks
  train(build_cfg(args.run_dir, 6 * BLOCK, resume=True))
  with np.load(os.path.join(args.run_dir, 'replay.npz')) as d2:
    n2 = int(d2['num_eps'])
    obs2_first16 = d2['obs'][:16].copy()
  assert n2 == 24, n2
  mets2 = json.load(open(os.path.join(args.run_dir, 'metrics.json')))
  assert mets2[-1]['step'] == 6 * BLOCK
  assert np.array_equal(obs2_first16, obs), 'restored episodes were altered'
  rep['resume'] = {'episodes_after_resume': 24,
                   'first16_bit_exact': True,
                   'final_step': mets2[-1]['step']}

  rep['pass'] = True
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  json.dump(rep, open(OUT, 'w'), indent=2)
  print(json.dumps(rep, indent=1))
  print('MULTIACTOR INTEGRATION PASS ->', OUT)


if __name__ == '__main__':
  main()
