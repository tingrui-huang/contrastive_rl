"""PROOF that offline training performs ZERO environment collection.

Instruments the real training loop (crl.train.train) and runs it twice on a tiny
budget:

  * ONLINE control  (offline_dataset='') -- the instrumentation MUST detect
    collection: collect_episode fires, add_episode fires during the loop, and the
    training env is stepped. This proves the probes actually catch collection.
  * OFFLINE run     (offline_dataset=<frozen .npz>) -- collection MUST be zero:
    collect_episode/collect_block never fire, add_episode fires ONLY during the
    pre-loop dataset preload (never after the first replay.sample), and the
    training env is stepped ZERO times -- while replay.sample and the EVAL env
    are exercised (optimization + eval still happen) and the dataset file hash is
    byte-identical before and after.

Probes (monkeypatched around the unmodified train()):
  - crl.train.collect_episode / collect_block            -> call counters
  - TrajectoryBuffer.add_episode                         -> split preload vs in-loop
  - TrajectoryBuffer.sample                              -> marks "training started"
  - each make_env instance's .step                       -> per-env step counters

Run:  python scripts/audit_offline_no_collection.py \
        --dataset datasets/point_U_offline_s0.npz --env_name point_U
Exit code 0 iff the online control detects collection AND the offline run proves
no collection AND the dataset hash is unchanged.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from crl import envs as envs_mod
from crl import train as train_mod
from crl.config import Config
from crl.replay import TrajectoryBuffer


def sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


class Probes:
  """Monkeypatch set around train(); restores originals on exit."""

  def __init__(self):
    self.reset()
    self._orig = {}

  def reset(self):
    self.collect_episode = 0
    self.collect_block = 0
    self.add_preload = 0
    self.add_in_loop = 0
    self.samples = 0
    self.training_started = False
    self.n_envs = 0            # number of envs constructed via make_env
    self.env_steps = {}        # per-env-construction-index step count
    self._env_idx = 0

  def __enter__(self):
    p = self
    o = self._orig

    o['collect_episode'] = train_mod.collect_episode
    o['collect_block'] = train_mod.collect_block
    o['add_episode'] = TrajectoryBuffer.add_episode
    o['sample'] = TrajectoryBuffer.sample
    o['make_env'] = envs_mod.make_env
    o['train_make_env'] = train_mod.envs.make_env

    def collect_episode(*a, **k):
      p.collect_episode += 1
      return o['collect_episode'](*a, **k)

    def collect_block(*a, **k):
      p.collect_block += 1
      return o['collect_block'](*a, **k)

    def add_episode(self, *a, **k):
      # Count only SUCCESSFUL adds: the frozen-buffer gate calls add_episode
      # expecting it to raise -- that must not count as an added episode.
      out = o['add_episode'](self, *a, **k)
      if p.training_started:
        p.add_in_loop += 1
      else:
        p.add_preload += 1
      return out

    def sample(self, batch_size):
      p.training_started = True      # first sample == optimization has begun
      p.samples += 1
      return o['sample'](self, batch_size)

    def make_env(*a, **k):
      env = o['make_env'](*a, **k)
      idx = p._env_idx
      p._env_idx += 1
      p.n_envs += 1
      p.env_steps[idx] = 0
      orig_step = env.step

      def step(action, _idx=idx, _orig=orig_step):
        p.env_steps[_idx] += 1
        return _orig(action)

      env.step = step
      return env

    train_mod.collect_episode = collect_episode
    train_mod.collect_block = collect_block
    TrajectoryBuffer.add_episode = add_episode
    TrajectoryBuffer.sample = sample
    envs_mod.make_env = make_env
    train_mod.envs.make_env = make_env
    return self

  def __exit__(self, *exc):
    train_mod.collect_episode = self._orig['collect_episode']
    train_mod.collect_block = self._orig['collect_block']
    TrajectoryBuffer.add_episode = self._orig['add_episode']
    TrajectoryBuffer.sample = self._orig['sample']
    envs_mod.make_env = self._orig['make_env']
    train_mod.envs.make_env = self._orig['train_make_env']


def base_config(env_name, ckpt_dir, offline_dataset=''):
  return Config(
      env_name=env_name, offline_dataset=offline_dataset,
      max_number_of_steps=1500, eval_every_steps=500, eval_episodes=5,
      log_every_steps=100000, min_replay_size=100, random_steps=50,
      num_sgd_steps_per_step=1, batch_size=128, ckpt_dir=ckpt_dir, seed=0)


def run_probed(cfg):
  with Probes() as p:
    train_mod.train(cfg)
  # Strict offline mode builds NO training env (env=None): exactly ONE env
  # (eval) is constructed. Online builds >=2 (collection actor(s) + eval).
  total_env_steps = sum(p.env_steps.values())
  return dict(collect_episode=p.collect_episode, collect_block=p.collect_block,
              add_preload=p.add_preload, add_in_loop=p.add_in_loop,
              samples=p.samples, n_envs_constructed=p.n_envs,
              total_env_steps=total_env_steps)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='datasets/point_U_offline_s0.npz')
  ap.add_argument('--env_name', default='point_U')
  ap.add_argument('--out', default='artifacts/offline_pipeline_audit')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  scratch = os.environ.get('TEMP', '/tmp')

  n_eps = int(np.load(args.dataset)['obs'].shape[0])
  hash_before = sha256(args.dataset)
  man = json.load(open(args.dataset + '.manifest.json'))
  manifest_match = (man['sha256'] == hash_before)

  print('=== ONLINE control (must DETECT collection) ===')
  online = run_probed(base_config(
      args.env_name, os.path.join(scratch, 'audit_online_ckpt')))
  print(online)

  print('\n=== OFFLINE run (must PROVE no collection) ===')
  offline = run_probed(base_config(
      args.env_name, os.path.join(scratch, 'audit_offline_ckpt'),
      offline_dataset=args.dataset))
  print(offline)

  hash_after = sha256(args.dataset)
  dataset_frozen = (hash_before == hash_after)

  # ---- checks ----
  control = dict(
      collect_fired=(online['collect_episode'] + online['collect_block']) > 0,
      add_in_loop_positive=online['add_in_loop'] > 0,
      built_collection_env=online['n_envs_constructed'] >= 2)
  control_ok = all(control.values())

  offline_checks = dict(
      collect_episode_zero=offline['collect_episode'] == 0,
      collect_block_zero=offline['collect_block'] == 0,
      add_in_loop_zero=offline['add_in_loop'] == 0,
      add_preload_equals_dataset=offline['add_preload'] == n_eps,
      no_collection_env_built=offline['n_envs_constructed'] == 1,
      optimization_happened=offline['samples'] > 0,
      eval_env_used=offline['total_env_steps'] > 0,
      dataset_hash_unchanged=dataset_frozen,
      manifest_hash_matches=manifest_match)
  offline_ok = all(offline_checks.values())
  all_pass = control_ok and offline_ok

  report = dict(
      dataset=os.path.abspath(args.dataset), env_name=args.env_name,
      dataset_episodes=n_eps, sha256_before=hash_before,
      sha256_after=hash_after, manifest_sha256=man['sha256'],
      online_control=online, offline_run=offline,
      control_checks=control, control_detects_collection=control_ok,
      offline_checks=offline_checks, offline_no_collection=offline_ok,
      all_pass=all_pass,
      verdict=('OFFLINE_VERIFIED' if all_pass else 'FAILED'),
      next_step=('offline loop proven collection-free; cleared to run the '
                 'point_U offline qualification learner'
                 if all_pass else 'DO NOT launch a learner; fix the pipeline'))
  json.dump(report, open(os.path.join(args.out, 'no_collection_audit.json'), 'w'),
            indent=2)

  print('\n' + '=' * 72)
  print('OFFLINE NO-COLLECTION AUDIT')
  print('=' * 72)
  print('[online control] detects collection:', control_ok, control)
  print('[offline run] no collection        :', offline_ok)
  for k, v in offline_checks.items():
    print(f'   [{"PASS" if v else "FAIL"}] {k}')
  print('-' * 72)
  print('VERDICT:', report['verdict'])
  print(report['next_step'])
  print('saved', os.path.join(args.out, 'no_collection_audit.json'))
  sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
