r"""Alpha launcher for the V6 failure-aware negative-sampling sweep.

ONE training script for every alpha. It does not reimplement anything: the
config, the benchmark contract, the vanilla-CRL guard and the manifest all come
from scripts/train_rockfall_clock_v6_baseline.py, and the failure-aware loss is
the production one in crl/losses.py reached through the two existing Config
fields. This file sets those two fields and nothing else.

    q_alpha(g) = (1 - alpha) q_normal(g) + alpha q_fail(g)

  alpha = 0.0   the vanilla control. No bank is loaded at all, so crl/losses.py
                skips the failure branch entirely and the critic loss and its
                gradients are byte-identical to the plain V6 CRL baseline.
  alpha > 0     the composed bank (60% random/noisy, 20% deliberate zone 1,
                20% deliberate zone 2) enters the NEGATIVE distribution only.

WHAT IS PROVEN BEFORE TRAINING STARTS
-------------------------------------
``--check-only`` runs the same gate the run does:

  * the training dataset is the frozen V6 one, byte-identical to its own
    composition audit (the baseline's ``_dataset_contract`` does this);
  * the bank stores 29-dim learner states, projects to exactly the goal width
    V6 CRL uses, and fits inside the training batch;
  * the bank composition is 60 / 20 / 20 and every deliberate entry carries
    normal 'wait' + executed 'go' + override + the matching failure zone;
  * ``_assert_vanilla_crl`` passes on this run's config with the two
    failure-negative fields reset to their vanilla values -- i.e. the arms
    differ in ``fail_neg_alpha`` (and the bank path it needs) and in nothing
    else.

The failure trajectories are NEGATIVE-BANK DATA ONLY. The offline dataset is
the untouched frozen V6 file; no failure episode is an anchor, a positive or a
hindsight positive.

Usage:
  python scripts/run_v6_failneg.py --alpha 0.0 --check-only
  python scripts/run_v6_failneg.py --alpha 0.3 --seed 0 --steps 100000
  python scripts/run_v6_failneg.py --alpha 0.3 --seed 0 --smoke --smoke-steps 800
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from crl import envs as envs_mod                          # noqa: E402
from crl.offline_audit import sha256_file                 # noqa: E402
from crl.replay import obs_to_goal as np_obs_to_goal      # noqa: E402
from verify_offline_d4rl import build_offline_cfg         # noqa: E402
import train_rockfall_clock_v6_baseline as B              # noqa: E402

BANK_DEFAULT = os.path.join('artifacts', 'v6_failneg', 'bank',
                            'v6_failure_bank_r60_z20_z20.npz')
RUN_ROOT = os.path.join('artifacts', 'v6_failneg', 'runs')
STEPS = 100_000
SMOKE_STEPS = 800
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
STATE_DIM = 29


def git(*a):
  try:
    return subprocess.check_output(
        ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                          # pylint: disable=broad-except
    return ''


def alpha_tag(a):
  """0.0 -> a0, 0.05 -> a005, 0.5 -> a05. Directory-safe, collision-free."""
  return 'a' + ('%g' % float(a)).replace('0.', '').replace('.', '')


def run_name(args):
  goal_tag = 'gxy' if args.env_name == B.ENV_XY else 'gfull'
  return os.path.join(
      RUN_ROOT,
      'v6fn_%s_s%d_%dk_%s_p%g-%g_h%d'
      % (alpha_tag(args.alpha), args.seed, args.steps // 1000, goal_tag,
         args.p_active_1, args.p_active_2, args.horizon))


def gate_bank(cfg, bank_path, alpha):
  """Everything that must be true of the bank before a run may start."""
  if alpha == 0.0:
    print('  bank            : (none -- alpha 0, the failure branch is '
          'skipped entirely)')
    return None
  if not os.path.exists(bank_path):
    raise SystemExit(
        'failure bank missing: %s\n  build it with\n'
        '    python scripts/collect_v6_failure_candidates.py --all '
        '--episodes 300\n'
        '    python scripts/make_v6_failure_bank.py --n-bank 250' % bank_path)
  with np.load(bank_path, allow_pickle=False) as b:
    goals = np.asarray(b['goals'], np.float32)
    kinds = np.asarray(b['source_type'])
    zones = np.asarray(b['actual_failure_zone'])
    meta = json.loads(str(b['meta']))
    keys = list(b.files)
  n = len(goals)
  if goals.shape[1] != STATE_DIM:
    raise SystemExit('bank stores %d-dim vectors, expected the %d-dim learner '
                     'state (crl/train.py slices it to goal coords)'
                     % (goals.shape[1], STATE_DIM))
  if n > cfg.batch_size:
    raise SystemExit('bank (%d) > batch_size (%d): crl/losses.py pads the '
                     'second critic apply and needs n_bank <= batch_size'
                     % (n, cfg.batch_size))
  proj = np_obs_to_goal(goals, cfg.start_index, cfg.end_index,
                        cfg.goal_indices)
  if proj.shape[1] != cfg.goal_dim:
    raise SystemExit('bank projects to %d goal columns but V6 CRL uses %d'
                     % (proj.shape[1], cfg.goal_dim))
  n_random = int((kinds == 'random').sum())
  n_z1 = int(((kinds == 'deliberate') & (zones == 1)).sum())
  n_z2 = int(((kinds == 'deliberate') & (zones == 2)).sum())
  if not (n_random / n == 0.60 and (n_z1 + n_z2) / n == 0.40
          and n_z1 == n_z2 and n_random + n_z1 + n_z2 == n):
    raise SystemExit('bank composition is not 60/20/20: random %d, Z1 %d, '
                     'Z2 %d of %d' % (n_random, n_z1, n_z2, n))
  spread = float(np.linalg.norm(proj - proj.mean(0), axis=1).mean())
  if spread <= 0.0:
    raise SystemExit('every bank entry projects to the same goal vector')
  print('  bank            : %s' % bank_path)
  print('  bank sha256     : %s' % sha256_file(bank_path))
  print('  bank shape      : %s -> goal dim %d (spread %.4f)'
        % (goals.shape, proj.shape[1], spread))
  print('  bank mix        : random %d (%.2f) | delib Z1 %d | Z2 %d  of %d'
        % (n_random, n_random / n, n_z1, n_z2, n))
  print('  settle substeps : %s'
        % meta.get('settled_state_extraction', {}).get('death_settle_substeps'))
  print('  bank arrays     : %s   (crl/train.py reads "goals" only)' % keys)
  return {'path': bank_path, 'sha256': sha256_file(bank_path),
          'n_bank': n, 'n_random': n_random, 'n_deliberate_zone1': n_z1,
          'n_deliberate_zone2': n_z2, 'goal_dim': int(proj.shape[1]),
          'settle': meta.get('settled_state_extraction', {}).get(
              'death_settle_substeps')}


def assert_only_alpha_differs(cfg):
  """Reuse the baseline's own vanilla guard on this run's config.

  Every algorithm-sensitive field is compared against the frozen V6 recipe
  with the two failure-negative fields reset to their vanilla values. If it
  passes, the run differs from the vanilla baseline in ``fail_neg_alpha`` and
  the bank path it needs, and in nothing else.
  """
  probe = dataclasses.replace(cfg, fail_bank_path='', fail_neg_alpha=0.0)
  B._assert_vanilla_crl(probe)                 # pylint: disable=protected-access


def build_parser():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--alpha', type=float, required=True,
                  help='fail_neg_alpha; 0.0 is the vanilla control')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--bank', default=BANK_DEFAULT)
  ap.add_argument('--env-name', choices=B.ENV_NAMES, default=B.ENV_XY)
  ap.add_argument('--npz', default=None,
                  help='override the frozen V6 dataset (probes only)')
  ap.add_argument('--horizon', type=int, default=B.HORIZON)
  ap.add_argument('--p-active-1', type=float, default=B.V6.P_ACTIVE_1)
  ap.add_argument('--p-active-2', type=float, default=B.V6.P_ACTIVE_2)
  ap.add_argument('--t0-min-1', type=int, default=B.V6.T0_MIN_1)
  ap.add_argument('--t0-max-1', type=int, default=B.V6.T0_MAX_1)
  ap.add_argument('--t0-min-2', type=int, default=B.V6.T0_MIN_2)
  ap.add_argument('--t0-max-2', type=int, default=B.V6.T0_MAX_2)
  ap.add_argument('--ckpt-dir', default=None)
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true')
  ap.add_argument('--smoke-steps', type=int, default=SMOKE_STEPS)
  return ap


def main(argv=None):
  args = build_parser().parse_args(argv)
  if not 0.0 <= args.alpha < 1.0:
    raise SystemExit('--alpha must be in [0, 1); crl/losses.py rejects the '
                     'endpoints above 0 and alpha 0 is the control')
  if args.smoke:
    args.steps = args.smoke_steps
  B._validate_benchmark_args(args)             # pylint: disable=protected-access
  if args.steps <= 0 or args.steps % args.horizon:
    raise SystemExit('--steps (%d) must be positive and divisible by --horizon '
                     '(%d)' % (args.steps, args.horizon))
  npz = args.npz or B._default_dataset(args.env_name)   # pylint: disable=protected-access
  run_dir = args.ckpt_dir or run_name(args)
  if args.smoke:
    run_dir += '_smoke'

  print('=' * 88)
  print('V6 FAILURE-NEGATIVE SWEEP -- GATE   alpha %g   seed %d'
        % (args.alpha, args.seed))
  print('=' * 88)
  print('  code commit     : %s%s'
        % (git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
           '   (WORKING TREE DIRTY)'
           if git('status', '--porcelain', '--', 'crl', 'scripts') else ''))
  #: the frozen dataset, verified against its own composition audit
  dataset_meta, dataset_sha, composition_audit = (
      B._dataset_contract(npz, args))          # pylint: disable=protected-access
  print('  dataset         : %s' % npz)
  print('  dataset sha256  : %s' % dataset_sha)
  print('  dataset outcomes: %s   n_episodes %s'
        % (dataset_meta.get('outcomes'), dataset_meta.get('n_episodes')))
  print('  p_active        : %g / %g   (benchmark default, untouched)'
        % (args.p_active_1, args.p_active_2))

  cfg = build_offline_cfg(max_steps=args.steps, ckpt_dir=run_dir)
  B._apply_v6_config(cfg, args, npz)           # pylint: disable=protected-access
  #: THE ONLY TWO FIELDS THIS LAUNCHER TOUCHES
  cfg.fail_neg_alpha = float(args.alpha)
  cfg.fail_bank_path = args.bank if args.alpha > 0.0 else ''
  assert_only_alpha_differs(cfg)
  print('  vanilla guard   : PASS (only fail_neg_alpha / fail_bank_path '
        'differ from the frozen V6 recipe)')
  #: fill obs_dim / goal_dim / goal_indices exactly the way crl/train.py does,
  #: so the bank gate below projects the bank with the SAME rule the run will.
  envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)
  print('  goal contract   : %s -> obs_dim %d + goal_dim %d, indices %s'
        % (args.env_name, cfg.obs_dim, cfg.goal_dim, cfg.goal_indices))
  bank_info = gate_bank(cfg, args.bank, args.alpha)
  print('  fail_neg_alpha  : %g' % cfg.fail_neg_alpha)
  print('=' * 88)
  print('GATE PASSED')
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0

  manifest = B._write_manifest(                # pylint: disable=protected-access
      run_dir, args, npz, dataset_meta, dataset_sha, composition_audit)
  with open(os.path.join(run_dir, 'failneg_arm.json'), 'w') as f:
    json.dump({'fail_neg_alpha': float(args.alpha),
               'fail_bank_path': cfg.fail_bank_path or None,
               'bank': bank_info,
               'seed': int(args.seed), 'steps': int(args.steps),
               'env_name': args.env_name, 'dataset': npz,
               'dataset_sha256': dataset_sha,
               'p_active_1': float(args.p_active_1),
               'p_active_2': float(args.p_active_2),
               'smoke': bool(args.smoke),
               'failure_episodes_in_training_set': False,
               'note': 'the failure bank is NEGATIVE-distribution data only; '
                       'the offline dataset is the frozen V6 file and no '
                       'failure episode is an anchor, a positive or a '
                       'hindsight positive.',
               'code_commit': git('log', '-1', '--format=%H', '--',
                                  'crl', 'scripts'),
               'command': ' '.join(sys.argv)}, f, indent=2)
  print('benchmark manifest ->', manifest)
  print('run dir: %s  (%s, %d steps)'
        % (run_dir, 'SMOKE' if args.smoke else 'PRODUCTION', args.steps))

  from crl.train import train
  train(cfg)
  print('DONE ->', run_dir)
  return 0


if __name__ == '__main__':
  sys.exit(main())
