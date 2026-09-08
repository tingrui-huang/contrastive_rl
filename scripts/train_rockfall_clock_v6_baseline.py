"""Train the unchanged vanilla CRL baseline on rockfall-clock V6.

The headline arm uses the upstream Ant goal contract: state(29) + goal XY(2)
= 31 input columns.  V6 changes only the environment, dataset, and episode
horizon relative to the V5 baseline recipe.  In particular, failure-aware
negative sampling is explicitly disabled here.

Example::

  python scripts/train_rockfall_clock_v6_baseline.py --steps 300000 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import rockfall_clock_v6 as V6          # noqa: E402
from crl.offline_audit import sha256_file         # noqa: E402
from crl.train import train                       # noqa: E402
import rockfall_clock_v6_teacher as CT            # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT_ROOT = 'artifacts/rockfall_clock_v6'
ENV_BASE = 'offline_antmaze_rockfall_clock_v6'
ENV_XY = ENV_BASE + '_gxy'
ENV_NAMES = (ENV_BASE, ENV_XY)
DATASET_BASE = os.path.join(
    OUT_ROOT, 'dataset', 'antmaze_rockfall_clock_v6.npz')
DATASET_XY = os.path.join(
    OUT_ROOT, 'dataset', 'antmaze_rockfall_clock_v6_gxy.npz')
HORIZON = int(CT.HORIZON)
assert HORIZON == 800, 'V6 train/eval/teacher horizons must stay synchronized'
ALGORITHM_CONTRACT = {
    'recipe': 'vanilla_crl_v5',
    'binary_nce': True,
    'use_td': False,
    'use_cpc': False,
    'use_gcbc': False,
    'add_mc_to_td': False,
    'twin_q': True,
    'bc_coef': 0.05,
    'random_goals': 0.0,
    'batch_size': 1024,
    'repr_dim': 16,
    'repr_norm': False,
    'repr_norm_temp': True,
    'hidden_layer_sizes': [1024, 1024],
    'use_image_obs': False,
    'use_layer_norm': False,
    'discount': 0.99,
    'tau': 0.005,
    'actor_learning_rate': 3e-4,
    'learning_rate': 3e-4,
    'num_sgd_steps_per_step': 4,
    'updates_per_step': 1,
}


def _validate_benchmark_args(args):
  if args.horizon <= 0:
    raise ValueError(f'--horizon must be positive, got {args.horizon}')
  for name in ('p_active_1', 'p_active_2'):
    value = float(getattr(args, name))
    if not 0.0 <= value <= 1.0:
      raise ValueError(f'--{name.replace("_", "-")} must be in [0, 1], '
                       f'got {value}')
  for zone in (1, 2):
    lo = int(getattr(args, f't0_min_{zone}'))
    hi = int(getattr(args, f't0_max_{zone}'))
    if lo > hi:
      raise ValueError(f'zone {zone} t0 minimum exceeds maximum: {lo}>{hi}')


def _default_dataset(env_name):
  return DATASET_XY if env_name == ENV_XY else DATASET_BASE


def _dataset_contract(npz, args):
  """Reject a V6 learner file whose benchmark or composition is unaudited."""
  if not os.path.isfile(npz):
    raise FileNotFoundError(
        f'V6 dataset not found: {npz}. Run '
        '`python scripts/collect_rockfall_clock_v6_dataset.py` first.')
  with np.load(npz, allow_pickle=False) as data:
    if 'meta' not in data:
      raise ValueError(f'{npz} has no JSON meta field; refusing an unaudited '
                       'baseline dataset')
    try:
      meta = json.loads(str(data['meta']))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError(f'{npz} has invalid JSON metadata') from exc

  goal_dim = 2 if args.env_name == ENV_XY else 29
  expected = {
      'environment_version': V6.ENV_VERSION,
      'env_name': args.env_name,
      'horizon': int(args.horizon),
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      'teacher_detour_prob': float(CT.TEACHER_DETOUR_PROB),
      'obs_dim': 29,
      'goal_dim': goal_dim,
      'observation_width': 29 + goal_dim,
      'failure_bank': False,
  }
  actual = {name: meta.get(name) for name in expected}
  timing = meta.get('timing') or {}
  expected_timing = {
      't0_range_1_inclusive': [int(args.t0_min_1), int(args.t0_max_1)],
      't0_range_2_inclusive': [int(args.t0_min_2), int(args.t0_max_2)],
  }
  mismatches = {}
  for name, wanted in expected.items():
    got = actual[name]
    if isinstance(wanted, float):
      matches = (isinstance(got, (int, float))
                 and np.isclose(float(got), wanted, rtol=0.0, atol=1e-12))
    else:
      matches = got == wanted
    if not matches:
      mismatches[name] = {'dataset': got, 'cli_expected': wanted}
  for name, wanted in expected_timing.items():
    got = timing.get(name)
    if got != wanted:
      mismatches[f'timing.{name}'] = {
          'dataset': got, 'cli_expected': wanted}
  if mismatches:
    raise ValueError(
        f'V6 dataset/CLI benchmark mismatch for {npz}: {mismatches}')

  audit_name = meta.get('composition_audit')
  if not isinstance(audit_name, str) or not audit_name:
    raise ValueError(f'{npz} metadata has no composition_audit reference')
  if os.path.basename(audit_name) != audit_name:
    raise ValueError(
        f'{npz} composition_audit must be a sibling basename, got '
        f'{audit_name!r}')
  audit_path = os.path.join(os.path.dirname(npz), audit_name)
  if not os.path.isfile(audit_path):
    raise FileNotFoundError(
        f'V6 composition audit not found: {audit_path}. Refusing to train on '
        'an unverified collector output.')
  try:
    with open(audit_path, encoding='utf-8') as handle:
      composition = json.load(handle)
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    raise ValueError(f'invalid V6 composition audit: {audit_path}') from exc
  if composition.get('all_pass') is not True:
    failed = [name for name, passed in
              (composition.get('gates') or {}).items() if passed is not True]
    raise ValueError(
        f'V6 composition audit did not pass ({audit_path}); failed gates: '
        f'{failed}')

  dataset_sha = sha256_file(npz)
  hash_key = 'xy_sha256' if args.env_name == ENV_XY else 'raw_sha256'
  audited_sha = (composition.get('dataset_paths') or {}).get(hash_key)
  if audited_sha != dataset_sha:
    raise ValueError(
        f'V6 dataset hash differs from its composition audit for {npz}: '
        f'audit={audited_sha!r}, actual={dataset_sha!r}')
  return meta, dataset_sha, audit_path


def _run_name(args):
  goal_tag = 'gxy' if args.env_name == ENV_XY else 'gfull'
  return os.path.join(
      OUT_ROOT, 'runs',
      f'v6clock_crl_s{args.seed}_{args.steps // 1000}k_{goal_tag}_'
      f'p{args.p_active_1:g}-{args.p_active_2:g}_'
      f't{args.t0_min_1}-{args.t0_max_1}_'
      f'{args.t0_min_2}-{args.t0_max_2}_h{args.horizon}')


def _apply_v6_config(cfg, args, npz):
  """Materialize the complete V6 benchmark contract on canonical CRL cfg."""
  cfg.resume = bool(args.resume)
  cfg.env_name = args.env_name
  cfg.offline_dataset = npz
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.max_episode_steps = int(args.horizon)
  cfg.rockfall_p_active_1 = float(args.p_active_1)
  cfg.rockfall_p_active_2 = float(args.p_active_2)
  cfg.rockfall_t0_min_1 = int(args.t0_min_1)
  cfg.rockfall_t0_max_1 = int(args.t0_max_1)
  cfg.rockfall_t0_min_2 = int(args.t0_min_2)
  cfg.rockfall_t0_max_2 = int(args.t0_max_2)
  cfg.seed = int(args.seed)

  # This is the vanilla baseline.  Make the opt-in failure-negative branch
  # impossible to enable through an inherited/default configuration.
  cfg.fail_bank_path = ''
  cfg.fail_neg_alpha = 0.0
  # The canonical builder has an opt-in OFFLINE_LAYER_NORM environment hook.
  # Pin it off so an ambient shell variable cannot silently change the V5
  # vanilla actor/critic architecture for this benchmark.
  cfg.use_layer_norm = False

  # Match the V5 baseline's reporting cadence (not learning math).
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = 30
  cfg.log_every_steps = 5_000
  return cfg


def _assert_vanilla_crl(cfg):
  """Guard the algorithm-sensitive values copied from V5's baseline."""
  expected = {
      'num_actors': 0,
      'use_td': False,
      'use_cpc': False,
      'add_mc_to_td': False,
      'use_gcbc': False,
      'twin_q': True,
      'bc_coef': 0.05,
      'random_goals': 0.0,
      'entropy_coefficient': 0.0,
      'target_entropy': 0.0,
      'batch_size': 1024,
      'repr_dim': 16,
      'repr_norm': False,
      'repr_norm_temp': True,
      'hidden_layer_sizes': (1024, 1024),
      'use_image_obs': False,
      'use_layer_norm': False,
      'discount': 0.99,
      'tau': 0.005,
      'actor_learning_rate': 3e-4,
      'learning_rate': 3e-4,
      'num_sgd_steps_per_step': 4,
      'updates_per_step': 1,
      'guard_abort': True,
      'fail_bank_path': '',
      'fail_neg_alpha': 0.0,
  }
  changed = {name: (getattr(cfg, name), value)
             for name, value in expected.items()
             if getattr(cfg, name) != value}
  if changed:
    raise RuntimeError(f'V6 baseline drifted from vanilla V5 CRL: {changed}')


def _write_manifest(run_dir, args, npz, dataset_meta, dataset_sha,
                    composition_audit):
  """Persist the benchmark knobs beside checkpoints for exact evaluation."""
  os.makedirs(run_dir, exist_ok=True)
  path = os.path.join(run_dir, 'benchmark_config.json')
  payload = {
      'environment_version': V6.ENV_VERSION,
      'env_name': args.env_name,
      'dataset': npz,
      'dataset_sha256': dataset_sha,
      'composition_audit': composition_audit,
      'learner_contract': {
          'state_dim': 29,
          'goal_dim': 2 if args.env_name == ENV_XY else 29,
          'flat_dim': 31 if args.env_name == ENV_XY else 58,
      },
      'horizon': int(args.horizon),
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      't0_range_1': [int(args.t0_min_1), int(args.t0_max_1)],
      't0_range_2': [int(args.t0_min_2), int(args.t0_max_2)],
      'steps': int(args.steps),
      'seed': int(args.seed),
      'dataset_collection_seed': dataset_meta.get('collection_seed'),
      'teacher_detour_prob': dataset_meta.get('teacher_detour_prob'),
      'algorithm': 'vanilla_crl_v5_recipe',
      'algorithm_contract': ALGORITHM_CONTRACT,
      'failure_bank': {'path': '', 'alpha': 0.0, 'enabled': False},
  }
  if args.resume:
    required = (path, os.path.join(run_dir, 'latest.pkl'),
                os.path.join(run_dir, 'offline_dataset.sha256'))
    missing = [item for item in required if not os.path.isfile(item)]
    if missing:
      raise FileNotFoundError(
          f'--resume requires a complete existing V6 run; missing: {missing}')
    with open(path, encoding='utf-8') as handle:
      previous = json.load(handle)
    with open(required[2], encoding='utf-8') as handle:
      previous_fingerprint = json.load(handle)
    if previous_fingerprint.get('sha256') != dataset_sha:
      raise ValueError(
          'cannot resume V6 checkpoint with a different dataset hash: '
          f'recorded={previous_fingerprint.get("sha256")!r}, '
          f'requested={dataset_sha!r}')
    # The target step budget may grow on resume and the identical dataset may
    # have moved, but no scientific or optimizer-seeding contract may change.
    invariant_keys = (
        'environment_version', 'env_name', 'dataset_sha256',
        'learner_contract', 'horizon', 'p_active_1', 'p_active_2',
        't0_range_1', 't0_range_2', 'seed', 'dataset_collection_seed',
        'teacher_detour_prob', 'algorithm', 'algorithm_contract',
        'failure_bank')
    changed = {key: {'recorded': previous.get(key), 'requested': payload[key]}
               for key in invariant_keys
               if previous.get(key) != payload[key]}
    if changed:
      raise ValueError(
          f'cannot resume V6 checkpoint under a different contract: {changed}')
    previous_steps = previous.get('steps')
    if not isinstance(previous_steps, int) or args.steps < previous_steps:
      raise ValueError(
          f'--resume step budget must not shrink: recorded={previous_steps!r}, '
          f'requested={args.steps}')
  with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2)
  return path


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Unchanged vanilla CRL baseline for two-rockfall V6.')
  parser.add_argument('--steps', type=int, default=100_000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--env-name', choices=ENV_NAMES, default=ENV_XY,
      help='default _gxy environment has state29 + goalXY2 = 31 columns')
  parser.add_argument(
      '--npz', default=None,
      help=f'new V6 dataset (default for headline arm: {DATASET_XY})')
  parser.add_argument('--ckpt-dir', default=None)
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--horizon', type=int, default=HORIZON)
  parser.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1)
  parser.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2)
  parser.add_argument('--t0-min-1', type=int, default=V6.T0_MIN_1)
  parser.add_argument('--t0-max-1', type=int, default=V6.T0_MAX_1)
  parser.add_argument('--t0-min-2', type=int, default=V6.T0_MIN_2)
  parser.add_argument('--t0-max-2', type=int, default=V6.T0_MAX_2)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  _validate_benchmark_args(args)
  if args.steps <= 0:
    raise ValueError(f'--steps must be positive, got {args.steps}')
  if args.steps % args.horizon:
    raise ValueError(
        f'--steps ({args.steps}) must be divisible by --horizon '
        f'({args.horizon}); the offline trainer advances in horizon-sized '
        'update blocks')
  npz = args.npz or _default_dataset(args.env_name)
  run_dir = args.ckpt_dir or _run_name(args)
  dataset_meta, dataset_sha, composition_audit = _dataset_contract(npz, args)

  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_dir)
  _apply_v6_config(cfg, args, npz)
  _assert_vanilla_crl(cfg)
  manifest = _write_manifest(
      run_dir, args, npz, dataset_meta, dataset_sha, composition_audit)

  goal_width = 2 if args.env_name == ENV_XY else 29
  print(
      f'rockfall-clock V6 vanilla CRL | steps {args.steps} | seed {args.seed} '
      f'| env {args.env_name} | input {29 + goal_width} (29+{goal_width}) '
      f'| dataset {npz} | H {args.horizon} '
      f'| p=({args.p_active_1:g},{args.p_active_2:g}) '
      f'| t0=([{args.t0_min_1},{args.t0_max_1}],'
      f'[{args.t0_min_2},{args.t0_max_2}]) '
      f'| failure-bank OFF | -> {run_dir}', flush=True)
  print('benchmark manifest ->', manifest, flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
