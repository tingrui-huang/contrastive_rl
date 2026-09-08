r"""Launcher for the V5 rockfall-clock failure-negative alpha sweep.

Two arms and nothing else:

  base   alpha = 0             the reference. No bank is loaded at all, so
                               crl/losses.py skips the failure branch entirely
                               and the critic loss and its gradients are
                               byte-identical to the plain offline CRL run.
  fail   alpha in (0, 1)       + the composed failure bank, through the
                               EXISTING crl/losses.py failure-negative term.

Everything else is the frozen offline AntMaze recipe that
scripts/train_rockfall_clock_v5_baseline.py uses
(verify_offline_d4rl.build_offline_cfg: bc_coef 0.05, twin-min, entropy 0,
batch 1024, repr 16, hidden (1024, 1024), discount 0.99, 4 sgd steps per step)
with the V5 env, the far05 dataset and the 400-step horizon. Nothing is
adapted to the rockfall timetable: the schedule never enters the observation
and whether the ant holds at the mouth, crosses, or takes the detour is the
plain goal-conditioned actor's own output.

TWO GOAL REPRESENTATIONS, and the difference is the whole point of --goal-rep:

  xy    the frozen V5 default (dc6690a): the goal is the torso XY pair. A rock
        death and a safe crossing have the SAME XY -- five-fold held-out LDA
        AUC 0.440, i.e. chance
        (scripts/probe_v5_failure_representation.py) -- so
        a bank in this space can only say "the corridor is bad". Kept as an
        arm, not deleted, because that is a result worth having.
  xyv   the same XY task goal plus the six torso velocity columns. Separates a
        death from a same-XY crossing at held-out AUC 0.879 and keeps it ~1.5
        sigma away
        from the expert's zero-torque HOLD. This is NOT the old 29-dim padded
        goal: the commanded goal is [gx, gy, 0, 0, 0, 0, 0, 0], "at the goal,
        at rest", and zero velocity is a state the ant really occupies, while
        z = 0 (underground) and quat = (0,0,0,0) (not a rotation) are not.

BOTH representations get their OWN alpha = 0 control, because a representation
change and a failure-bank change must never be read off the same comparison.
The original XY V5 baseline (scripts/train_rockfall_clock_v5_baseline.py) is
untouched and stays available separately.

The failure EPISODES do not enter the positive training set by default. That
is an independent choice, and it is recorded either way in arm_provenance.json
('failure_episodes_in_training_set'); pass --train-npz to override the
training file, and keep it fixed across the whole sweep if you do.

Usage:
  python scripts/run_v5_failneg.py --diff --goal-rep xyv
  python scripts/run_v5_failneg.py --goal-rep xyv --arm base --check-only
  python scripts/run_v5_failneg.py --goal-rep xyv --arm base --smoke
  python scripts/run_v5_failneg.py --goal-rep xyv --arm fail --alpha 0.3 \
      --bank artifacts/v5_failneg/bank/v5_failure_bank_n60d40.npz --run
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config                          # noqa: E402
from crl import rockfall_clock_v5 as V5                 # noqa: E402
from verify_offline_d4rl import build_offline_cfg       # noqa: E402

DATASET_DIR = 'artifacts/rockfall_clock_v5/dataset'
BANK_DEFAULT = 'artifacts/v5_failneg/bank/v5_failure_bank_n60d40.npz'
#: goal representation -> (env id, training npz, goal indices, run tag).
#: 'xy' reproduces the frozen dc6690a default exactly; 'xyv' is the velocity-
#: augmented contract (crl.rockfall_clock_v5.GOAL_INDICES_XYV).
GOAL_REPS = {
    'xy': {'env': 'offline_ant_umaze_rockfall_clock_v5_gxy',
           'npz': f'{DATASET_DIR}/antmaze_rockfall_clock_v5_far05_gxy.npz',
           'indices': V5.GOAL_INDICES_XY, 'tag': 'v5fn_xy'},
    'xyv': {'env': 'offline_ant_umaze_rockfall_clock_v5_gxyv',
            'npz': f'{DATASET_DIR}/antmaze_rockfall_clock_v5_far05_gxyv.npz',
            'indices': V5.GOAL_INDICES_XYV, 'tag': 'v5fn_xyv'},
}
ARMS = ('base', 'fail')
STEPS = 100_000
SMOKE_STEPS = 2_000
HORIZON = 400
BC_COEF = 0.05
EVAL_EPISODES = 30
#: module-level state, set ONCE from main() before any config is built, so
#: gate(), config_diff() and build_cfg() cannot disagree about what is running.
REP = 'xyv'
ENV = GOAL_REPS[REP]['env']
DATASET = GOAL_REPS[REP]['npz']
BANK = BANK_DEFAULT
ALPHA_OVERRIDE = None


def content_sha(path):
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode())
      h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode())
      h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def select_rep(rep, train_npz=''):
  global REP, ENV, DATASET
  REP = rep
  spec = GOAL_REPS[rep]
  ENV = spec['env']
  DATASET = train_npz or spec['npz']
  return spec


def select_bank(path):
  global BANK
  if path:
    BANK = path
  return BANK


def select_alpha(a):
  global ALPHA_OVERRIDE
  if a is None:
    return
  if not 0.0 < a < 1.0:
    raise SystemExit('--alpha must be in (0, 1); crl/losses.py rejects the '
                     'endpoints (alpha = 0 is the base arm)')
  ALPHA_OVERRIDE = float(a)


def effective_alpha(arm):
  if arm == 'fail':
    if ALPHA_OVERRIDE is None:
      raise SystemExit('the fail arm needs --alpha')
    return ALPHA_OVERRIDE
  return 0.0


def alpha_tag(a):
  """0.05 -> a05, 0.5 -> a5. Directory-safe and collision-free."""
  return 'a' + ('%g' % a).replace('0.', '').replace('.', '')


def check_resume(run_id, current, allow_mismatch=False):
  """Refuse to resume a checkpoint under a different experiment contract."""
  prov_path = os.path.join(run_id, 'arm_provenance.json')
  latest_path = os.path.join(run_id, 'latest.pkl')
  if not os.path.exists(prov_path) or not os.path.exists(latest_path):
    raise SystemExit('--resume requires both %s and %s'
                     % (prov_path, latest_path))
  with open(prov_path) as f:
    previous = json.load(f)
  invariant = (
      'code_commit', 'goal_rep', 'goal_indices', 'env', 'arm', 'alpha', 'seed',
      'dataset_content_sha256', 'dataset_train_npz_overridden',
      'failure_episodes_in_training_set', 'bank_content_sha256')
  mismatch = {k: (previous.get(k), current.get(k)) for k in invariant
              if previous.get(k) != current.get(k)}
  old_steps = int(previous.get('steps', 0))
  new_steps = int(current.get('steps', 0))
  if new_steps < old_steps:
    mismatch['steps'] = (old_steps, new_steps)
  if mismatch and not allow_mismatch:
    raise SystemExit(
        'resume provenance mismatch: %s\nRefusing to continue a checkpoint '
        'under a changed objective/artifact. Pass --allow-resume-mismatch '
        'only after auditing the differences.' % mismatch)
  if mismatch:
    print('WARNING: explicitly allowing resume provenance mismatch: %s'
          % mismatch)
  else:
    print('RESUME PROVENANCE MATCHED: objective, representation, dataset, bank, '
          'seed and code commit are unchanged')


def build_cfg(arm, ckpt_dir, steps=STEPS, seed=0, resume=False, alpha=None):
  """The frozen offline AntMaze recipe + the V5 env/dataset, and the failure-
  negative term as the ONLY thing the two arms differ in."""
  use_bank = arm == 'fail'
  cfg = build_offline_cfg(max_steps=steps, ckpt_dir=ckpt_dir)
  cfg.env_name = ENV
  cfg.offline_dataset = DATASET
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  cfg.max_episode_steps = HORIZON
  cfg.bc_coef = BC_COEF
  cfg.seed = seed
  cfg.resume = resume
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = EVAL_EPISODES
  cfg.log_every_steps = 5_000
  #: the ONLY intended difference between the two arms
  cfg.fail_bank_path = BANK if use_bank else ''
  cfg.fail_neg_alpha = (effective_alpha(arm) if alpha is None
                        else (float(alpha) if use_bank else 0.0))
  return cfg


def baseline_cfg(steps=STEPS, seed=0):
  """What scripts/train_rockfall_clock_v5_baseline.py builds for
  ``--variant far05 --method crl --goal-rep <REP>``, reconstructed field for
  field. The base arm must equal this or the sweep has no reference."""
  cfg = build_offline_cfg(max_steps=steps, ckpt_dir='X')
  cfg.resume = False
  cfg.env_name = ENV
  cfg.offline_dataset = DATASET
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  cfg.max_episode_steps = HORIZON
  cfg.bc_coef = BC_COEF
  cfg.seed = seed
  cfg.eval_every_steps = 10_000
  cfg.eval_episodes = EVAL_EPISODES
  cfg.log_every_steps = 5_000
  return cfg


def config_diff(seed=0):
  """Assert the two arms differ ONLY in the failure-negative term, and that
  the base arm equals the published V5 baseline config.

  The fail side is built at ALPHA_OVERRIDE when one was given and at a
  representative 0.1 otherwise, so --check-only on the base arm still proves
  the two-arm invariant without inventing an alpha for the run itself."""
  diff_alpha = 0.1 if ALPHA_OVERRIDE is None else ALPHA_OVERRIDE
  a = dataclasses.asdict(build_cfg('base', 'X', seed=seed))
  b = dataclasses.asdict(build_cfg('fail', 'X', seed=seed, alpha=diff_alpha))
  diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
  allowed = {'fail_neg_alpha', 'fail_bank_path'}
  print('=' * 82)
  print('CONFIG DIFF  base (alpha 0)  vs  fail (alpha %g)   [%s, goal %s]'
        % (diff_alpha, ENV, REP))
  print('=' * 82)
  for k, (x, y) in sorted(diff.items()):
    print('  %-20s %-26r -> %-26r  %s'
          % (k, x, y, 'OK' if k in allowed else 'UNEXPECTED'))
  extra = set(diff) - allowed
  if extra:
    raise SystemExit('arms differ in fields other than the failure-negative '
                     'term: %s' % sorted(extra))
  base_ref = dataclasses.asdict(baseline_cfg(seed=seed))
  bdiff = {k: (base_ref[k], a[k]) for k in base_ref if base_ref[k] != a[k]}
  bdiff.pop('ckpt_dir', None)
  bdiff.pop('tensorboard', None)
  if bdiff:
    raise SystemExit('the alpha = 0 arm is NOT the published V5 baseline '
                     'config; differs in %s' % bdiff)
  print('\n  identical in: dataset, seed, architecture, batch size, optimizer,')
  print('  steps, replay, future-goal relabeling and ordinary negatives.')
  print('  ASSERTION PASSED: the only difference is the failure-negative term,')
  print('  and the alpha = 0 arm reproduces the V5 offline CRL baseline config')
  print('  for goal representation %r.' % REP)
  return diff


def git(*a):
  try:
    return subprocess.check_output(
        ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                            # pylint: disable=broad-except
    return ''


def gate(arm, seed, train_npz_overridden):
  print('=' * 82)
  print('V5 FAILURE-NEGATIVE -- PROVENANCE GATE')
  commit = git('log', '-1', '--format=%H', '--', 'crl', 'scripts')
  head = git('rev-parse', 'HEAD')
  dirty = bool(git('status', '--porcelain', '--', 'crl', 'scripts'))
  print('  code commit  : %s%s' % (commit, '   (WORKING TREE DIRTY)'
                                   if dirty else ''))
  print('  head         : %s' % head)
  print('  goal rep     : %s   env %s   indices %s'
        % (REP, ENV, list(GOAL_REPS[REP]['indices'])))
  print('  arm          : %s   alpha %g   seed %d'
        % (arm, effective_alpha(arm), seed))
  if not os.path.exists(DATASET):
    raise SystemExit(
        'dataset missing: %s\n  build it with\n'
        '    python scripts/collect_rockfall_clock_v5_dataset.py --p-far 0.05\n'
        '    python scripts/make_v5_gxy_dataset.py    # goal-rep xy\n'
        '    python scripts/make_v5_gxyv_dataset.py   # goal-rep xyv'
        % DATASET)
  ds_sha = content_sha(DATASET)
  with np.load(DATASET, allow_pickle=False) as d:
    n_eps, L, W = d['obs'].shape
    lengths = np.asarray(d['lengths'], np.int64)
    meta = json.loads(str(d['meta'])) if 'meta' in d else {}
  goal_dim = len(GOAL_REPS[REP]['indices'])
  if W != 29 + goal_dim:
    raise SystemExit('dataset obs width %d, expected %d (state 29 + goal %d) '
                     'for goal rep %r' % (W, 29 + goal_dim, goal_dim, REP))
  print('  dataset      : %s' % DATASET)
  print('  content sha  : %s' % ds_sha)
  print('  episodes     : %d x %d rows, obs width %d, %d transitions'
        % (n_eps, L, W, int((lengths - 1).sum())))
  print('  route counts : %s   intents %s'
        % (meta.get('route_counts'), meta.get('intent_counts')))
  #: far05 exists so the learner can SEE a safe alternative. Losing it would
  #: quietly change what the sweep is measuring.
  n_detour = (meta.get('route_counts') or {}).get('detour', 0)
  if n_detour <= 0:
    raise SystemExit('the training set has no detour episodes: the safe '
                     'alternative coverage far05 exists for is gone')
  if meta.get('n_deaths', 0) != 0:
    raise SystemExit('the training set contains %d expert deaths; the frozen '
                     'far05 set has none' % meta['n_deaths'])
  print('  detour eps   : %d (safe-alternative coverage preserved)' % n_detour)
  if train_npz_overridden:
    print('  NOTE: --train-npz overrode the frozen far05 training set. This '
          'is recorded; keep it fixed across the whole sweep.')

  bank_sha, bank_shape = None, None
  if arm == 'fail':
    if not os.path.exists(BANK):
      raise SystemExit(
          'failure bank missing: %s\n  build it with\n'
          '    python scripts/collect_v5_failure_episodes.py --all '
          '--episodes 400\n'
          '    python scripts/make_v5_failure_bank.py --compose '
          'noisy=0.6,deliberate=0.4 --max-bank 256' % BANK)
    bank_sha = content_sha(BANK)
    with np.load(BANK, allow_pickle=False) as b:
      g = np.asarray(b['goals'], np.float32)
      arms_in_bank = (b['source_arm'] if 'source_arm' in b.files else None)
      bmeta = json.loads(str(b['meta'])) if 'meta' in b.files else {}
    bank_shape = list(g.shape)
    if g.shape[1] != 29:
      raise SystemExit('bank stores %d-dim vectors, expected the 29-dim '
                       'learner state (crl/train.py slices it to goal coords)'
                       % g.shape[1])
    cfg_probe = build_cfg('fail', 'X', steps=1, seed=seed)
    if g.shape[0] > cfg_probe.batch_size:
      raise SystemExit('bank (%d) > batch_size (%d): crl/losses.py pads the '
                       'second critic apply and needs n_bank <= batch_size'
                       % (g.shape[0], cfg_probe.batch_size))
    proj = g[:, list(GOAL_REPS[REP]['indices'])]
    print('  bank         : %s' % BANK)
    print('  bank content : %s' % bank_sha)
    print('  bank shape   : %s -> goal dim %d under %r'
          % (g.shape, proj.shape[1], REP))
    if arms_in_bank is not None:
      import collections
      print('  bank mix     : %s'
            % dict(collections.Counter(arms_in_bank.tolist())))
    print('  bank compose : %s   privileged selection %s'
          % (bmeta.get('compose'),
             bmeta.get('selection_uses_privileged_fields')))
    #: the failure states must not be degenerate under THIS projection: if
    #: every entry collapses onto the same point the bank is a single goal
    #: wearing 256 hats.
    spread = float(np.linalg.norm(proj - proj.mean(0), axis=1).mean())
    print('  bank spread  : mean |g - gbar| = %.4f under %r' % (spread, REP))
    if spread <= 0.0:
      raise SystemExit('every bank entry projects to the same goal vector')
  else:
    print('  bank         : (none -- alpha 0, the fail branch is skipped)')
  print('=' * 82)
  print('PROVENANCE GATE PASSED')
  return {'code_commit': commit, 'head': head, 'dirty': dirty,
          'goal_rep': REP, 'goal_indices': list(GOAL_REPS[REP]['indices']),
          'env': ENV, 'arm': arm, 'alpha': effective_alpha(arm), 'seed': seed,
          'dataset': DATASET, 'dataset_content_sha256': ds_sha,
          'dataset_train_npz_overridden': bool(train_npz_overridden),
          'failure_episodes_in_training_set': bool(train_npz_overridden),
          'n_episodes': int(n_eps),
          'n_transitions': int((lengths - 1).sum()),
          'n_detour_episodes': int(n_detour),
          'bank': BANK if arm == 'fail' else None,
          'bank_content_sha256': bank_sha, 'bank_shape': bank_shape,
          'steps': STEPS}


def main():
  global STEPS
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--goal-rep', choices=sorted(GOAL_REPS), default='xyv')
  ap.add_argument('--arm', choices=ARMS, default='base')
  ap.add_argument('--alpha', type=float, default=None,
                  help='mixture weight on the failure negatives, in (0, 1); '
                       'required for --arm fail, ignored for base')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--bank', default='',
                  help='failure bank npz (29-dim learner states in "goals")')
  ap.add_argument('--train-npz', default='',
                  help='override the frozen far05 training set, e.g. one with '
                       'the failure episodes merged in. Recorded in '
                       'arm_provenance.json; keep it fixed across the sweep.')
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--smoke-steps', type=int, default=SMOKE_STEPS,
                  help='gradient steps for --smoke (default 2000)')
  ap.add_argument('--ckpt-dir', default='')
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--allow-resume-mismatch', action='store_true',
                  help='override the resume provenance lock after manually '
                       'auditing every reported difference')
  ap.add_argument('--diff', action='store_true')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true')
  ap.add_argument('--run', action='store_true')
  args = ap.parse_args()

  STEPS = args.steps
  select_rep(args.goal_rep, args.train_npz)
  select_alpha(args.alpha)
  select_bank(args.bank)

  if args.diff:
    config_diff(args.seed)
    return 0

  prov = gate(args.arm, args.seed, bool(args.train_npz))
  config_diff(args.seed)
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0
  if not (args.smoke or args.run):
    raise SystemExit('pass one of --diff / --check-only / --smoke / --run')

  tag = GOAL_REPS[REP]['tag']
  #: the alpha suffix is mandatory on the fail arm: without it every alpha in
  #: a sweep writes into ONE directory and later runs overwrite earlier ones.
  run_id = args.ckpt_dir or (
      '%s_base_s%d' % (tag, args.seed) if args.arm == 'base'
      else '%s_%s_s%d' % (tag, alpha_tag(effective_alpha('fail')), args.seed))
  if args.smoke:
    run_id += '_smoke'
  steps = args.smoke_steps if args.smoke else args.steps
  cfg = build_cfg(args.arm, run_id, steps=steps, seed=args.seed,
                  resume=args.resume)
  prov.update({'run_dir': run_id, 'smoke': bool(args.smoke), 'steps': steps,
               'resume': bool(args.resume),
               'command': ' '.join(sys.argv)})
  if args.resume:
    check_resume(run_id, prov, args.allow_resume_mismatch)
  elif (os.path.exists(os.path.join(run_id, 'latest.pkl'))
        or os.path.exists(os.path.join(run_id, 'arm_provenance.json'))):
    raise SystemExit(
        'run directory %s already contains a checkpoint/provenance; use '
        '--resume or choose --ckpt-dir instead of overwriting it' % run_id)
  os.makedirs(run_id, exist_ok=True)
  with open(os.path.join(run_id, 'arm_provenance.json'), 'w') as f:
    json.dump(prov, f, indent=2)
  print('\nrun dir: %s  (%s, %d steps)'
        % (run_id, 'SMOKE' if args.smoke else 'PRODUCTION', steps))

  from crl.train import train
  train(cfg)
  print('\nDONE -> %s' % run_id)
  return 0


if __name__ == '__main__':
  sys.exit(main())
