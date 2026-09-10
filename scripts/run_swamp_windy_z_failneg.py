"""Launcher for the Z-state failure-aware CRL sanity experiment.

Two arms and nothing else:

  zbase   alpha = 0     the Z-state baseline
  zfail   alpha = 0.1   + the 3-D failure bank, through the EXISTING
                        crl/losses.py failure-negative term

Everything else is the accepted windy recipe, read off the run behind the
CONFOUNDED_SHORTCUT_BIAS reference: bc_coef 0.5, random_goals 0.5, twin_q
False, repr_dim 64, hidden 256x256, discount 0.95, batch 256, lr 3e-4,
num_sgd_steps_per_step 10, 150k steps. anchor_cut_mode is '' (the plain
original draw) because the accepted 2-D control was the `baseline` arm.

WHAT IS NEW, AND ONLY THIS: obs_norm_mode='z_physical'. The z column of the
state AND goal halves is multiplied by 1/|z_min| inside crl/networks.py (see
crl.obs_norm.obs_scale_vector), so it is applied exactly once and the raw
failure bank -- which crl/losses.py splices into the goal half before calling
q_network.apply -- is scaled by the same code. Nothing pre-normalizes.

NOT CHANGED, deliberately: TrajectoryBuffer future-goal relabeling (a genuine
z<0 future state may be sampled as a positive, with no death mask and no
future-aware masking), and the B x B ordinary in-batch negatives.

`--diff` builds both configs and asserts that the ONLY fields differing are the
failure-negative term and the checkpoint directory. Run it before training.

--alpha sweeps the mixture weight on the zfail arm and --bank points at a
different failure bank, so one launcher covers a whole alpha sweep without a
new arm per value. The run tag then carries the alpha (zfail_a3_s0), because
without it every alpha in a sweep writes into ONE directory and the later runs
silently overwrite the earlier ones.

Usage:
  python scripts/run_swamp_windy_z_failneg.py --diff
  python scripts/run_swamp_windy_z_failneg.py --arm zbase --run
  python scripts/run_swamp_windy_z_failneg.py --arm zfail --run

  # frame-stacked alpha sweep against a composed bank (see
  # scripts/run_f4_failneg_sweep.sh, which drives exactly this):
  python scripts/run_swamp_windy_z_failneg.py --version f4 --arm zfail       --alpha 0.3       --bank artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz       --run
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

from crl.config import Config                      # noqa: E402
from scripts.merge_swamp_windy_baddemo import dataset_environment  # noqa: E402

# Version registry. v0 is the accepted, already-published configuration and is
# the DEFAULT so nothing about the existing result can change by accident; v1
# points at its own env, dataset, bank and run tag so no v0 artifact is ever
# overwritten.
#
# ``norm`` is part of the spec rather than a module constant because it is NOT
# common to every version: the z envs carry a depth in different units from x
# and y and must be rescaled by 1/|z_min| exactly once inside crl/networks.py,
# whereas the frame-stacked env holds nothing but raw maze-unit positions and
# must NOT be rescaled at all. Keeping it here means v0 and v1 reproduce
# byte-identically while f4 cannot silently inherit a scaling that would be
# wrong for it.
VERSIONS = {
    'v0': {'env': 'point_two_route_swamp_windy_z_v0',
           'dataset': 'datasets/swamp_windy_z_merged_s0.npz',
           'bank': 'artifacts/swamp_windy_z_failure_bank/failure_bank_z.npz',
           'tag': 'swamp_windy_z',
           'norm': ('z_physical', 0.5)},
    'v1': {'env': 'point_two_route_swamp_windy_z_v1',
           'dataset': 'datasets/swamp_windy_z_v1_merged_s0.npz',
           'bank': 'artifacts/swamp_windy_z_v1_failure_bank/'
                   'failure_bank_z_v1_entry.npz',
           'tag': 'swamp_windy_z_v1',
           'norm': ('z_physical', 0.5)},
    'f4': {'env': 'point_two_route_swamp_windy_f4_v0',
           'dataset': 'datasets/swamp_windy_f4_merged_s0.npz',
           'bank': 'artifacts/swamp_windy_f4_failure_bank/'
                   'failure_bank_f4_entry.npz',
           'tag': 'swamp_windy_f4',
           # every coordinate is already an (x, y) in maze units
           'norm': ('', 0.0)},
}
ENV = VERSIONS['v0']['env']
DATASET = VERSIONS['v0']['dataset']
BANK = VERSIONS['v0']['bank']
NORM = VERSIONS['v0']['norm']
ARMS = ('zbase', 'zfail')
ALPHA = {'zbase': 0.0, 'zfail': 0.1}
# Set by --alpha (zfail only). A module-level override rather than a build_cfg
# argument so gate(), config_diff() and build_cfg() cannot disagree about which
# alpha this invocation is running -- the same reason select_version() exists.
ALPHA_OVERRIDE = None
BATCH_SIZE = 256
STEPS = 150_000
SGD_STEPS_PER_STEP = 10
Z_MIN_ABS = 0.5                       # |z_min| of the env; asserted below


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


def effective_alpha(arm):
  if arm == 'zfail' and ALPHA_OVERRIDE is not None:
    return ALPHA_OVERRIDE
  return ALPHA[arm]


def select_alpha(a):
  """Override the zfail alpha. Called once from main(), before any config."""
  global ALPHA_OVERRIDE
  if a is None:
    return
  if not 0.0 < a < 1.0:
    raise SystemExit('--alpha must be in (0, 1); crl/losses.py rejects the '
                     'endpoints (alpha=0 is the zbase arm)')
  ALPHA_OVERRIDE = float(a)


def alpha_tag(a):
  """0.05 -> a005, 0.5 -> a05. Directory-safe and collision-free."""
  return 'a' + ('%g' % a).replace('0.', '').replace('.', '')


def select_bank(path):
  """Point at a different failure bank (e.g. a composed one)."""
  global BANK
  if path:
    BANK = path
  return BANK


def select_version(v):
  """Point the module-level ENV/DATASET/BANK at one version. Called once, from
  main(), BEFORE any config is built, so build_cfg and gate() agree."""
  global ENV, DATASET, BANK, NORM
  spec = VERSIONS[v]
  ENV, DATASET, BANK = spec['env'], spec['dataset'], spec['bank']
  NORM = spec['norm']
  return spec


def build_cfg(arm, ckpt_dir, steps=STEPS, seed=0):
  use_bank = arm == 'zfail'
  return Config(
      env_name=ENV, offline_dataset=DATASET,
      max_number_of_steps=steps,
      # the ONLY intended difference between the two arms
      fail_bank_path=BANK if use_bank else '',
      fail_neg_alpha=effective_alpha(arm),
      # per-version and identical in both arms: 'z_physical' rescales the
      # depth column for the z envs, '' leaves the frame-stacked positions
      # alone because they are already in maze units.
      obs_norm_mode=NORM[0], obs_norm_z_scale=NORM[1],
      # positives and ordinary negatives are untouched
      anchor_cut_mode='', balanced_sampling=False,
      # THE ESTABLISHED WINDY RECIPE
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=0.5, random_goals=0.5,
      entropy_coefficient=0.0, target_entropy=0.0,
      batch_size=BATCH_SIZE, repr_dim=64, hidden_layer_sizes=(256, 256),
      discount=0.95, learning_rate=3e-4, actor_learning_rate=3e-4,
      num_sgd_steps_per_step=SGD_STEPS_PER_STEP, num_actors=0,
      guard_abort=True, jit=True, seed=seed,
      eval_every_steps=10_000, eval_episodes=50, log_every_steps=1_000,
      ckpt_dir=ckpt_dir)


def config_diff(seed=0):
  """Assert the two arms differ ONLY in the failure-negative term."""
  a = dataclasses.asdict(build_cfg('zbase', 'X', seed=seed))
  b = dataclasses.asdict(build_cfg('zfail', 'X', seed=seed))
  diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
  allowed = {'fail_neg_alpha', 'fail_bank_path'}
  print('=' * 78)
  print('CONFIG DIFF  zbase (alpha=0)  vs  zfail (alpha=%g)   [%s]'
        % (effective_alpha('zfail'), ENV))
  print('=' * 78)
  for k, (x, y) in sorted(diff.items()):
    ok = 'OK' if k in allowed else 'UNEXPECTED'
    print('  %-22s %-28r -> %-28r  %s' % (k, x, y, ok))
  extra = set(diff) - allowed
  if extra:
    raise SystemExit('arms differ in fields other than the failure-negative '
                     'term: %s' % sorted(extra))
  print('\n  identical in: dataset, seed, architecture, batch size, optimizer,')
  print('  steps, replay, future relabeling, ordinary negatives, Z scaling.')
  print('  ASSERTION PASSED: the only difference is the failure-negative term.')
  return diff


def gate(arm, seed):
  print('=' * 78)
  print('Z FAILURE-AWARE CRL -- PROVENANCE GATE')
  try:
    commit = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', 'crl', 'scripts'],
        cwd=os.path.dirname(_HERE)).decode().strip()
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(_HERE)).decode().strip()
    dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'crl', 'scripts'],
        cwd=os.path.dirname(_HERE)).decode().strip())
  except Exception:                                # pylint: disable=broad-except
    commit = head = '(unavailable)'
    dirty = None
  print('  code commit  : %s%s' % (commit,
                                   '   (WORKING TREE DIRTY)' if dirty else ''))
  print('  head         : %s' % head)
  print('  arm          : %s   alpha %g   seed %d'
        % (arm, effective_alpha(arm), seed))
  if not os.path.exists(DATASET):
    raise SystemExit('dataset missing: %s\n  regenerate with '
                     'scripts/collect_swamp_windy_z.py + '
                     'scripts/merge_swamp_windy_baddemo.py' % DATASET)
  ds = content_sha(DATASET)
  with np.load(DATASET, allow_pickle=False) as d:
    meta = json.loads(str(d['meta'])) if 'meta' in d else {}
    n_eps, L, W = d['obs'].shape
    n_dead = int(np.asarray(d['entered_active_swamp']).sum())
    ob = d['obs']
  from crl import envs as envs_mod
  env = envs_mod.make_env(ENV, Config(env_name=ENV), seed=seed)
  dataset_env, active_prob = dataset_environment(meta, DATASET)
  env_prob = float(env.active_prob)
  if dataset_env != ENV or not np.isclose(active_prob, env_prob,
                                         rtol=0, atol=1e-8):
    raise SystemExit('dataset environment/probability mismatch: '
                     f'{dataset_env} p={active_prob:g}; '
                     f'current environment {ENV} p={env_prob:g}. '
                     'Regenerate the dataset with the current environment.')
  if 'merge' in meta:
    bad_env, bad_prob = dataset_environment(
        meta['merge'].get('bad_demo_meta', {}), DATASET + ' bad-demo component')
    if bad_env != dataset_env or not np.isclose(bad_prob, active_prob,
                                               rtol=0, atol=1e-8):
      raise SystemExit('merged dataset contains a bad-demo component from a '
                       'different environment or swamp probability')
  print('  dataset      : %s' % DATASET)
  print('  content sha  : %s' % ds)
  print('  environment  : %s, per-cell swamp probability %g'
        % (dataset_env, active_prob))
  print('  episodes     : %d x %d rows, obs width %d, %d transitions'
        % (n_eps, L, W, n_eps * (L - 1)))
  print('  failed eps   : %d (%.4f)' % (n_dead, n_dead / n_eps))
  # The expected observation layout is a property of the VERSION, not of this
  # launcher: the z envs store [x, y, z, gx, gy, gz] and f4 stores a stack of
  # NF positions plus the tiled goal. Hard-coding width 6 here would have
  # silently rejected f4, so the check is derived from the spec instead.
  if NORM[0] == 'z_physical':
    z = ob[:, :, 2]
    print('  z==0 %.5f   z<0 %.5f' % ((z == 0).mean(), (z < 0).mean()))
    if W != 6:
      raise SystemExit('obs width %d, expected 6 ([x,y,z,gx,gy,gz])' % W)
  else:
    nf = W // 4                       # W = 2*NF state + 2*NF goal
    fr = ob[:, :, :2 * nf].reshape(n_eps, L, nf, 2)
    frozen = np.abs(fr - fr[:, :, :1, :]).max(axis=(2, 3)) == 0.0
    print('  layout       : frame stack, NF = %d (%d state + %d goal)'
          % (nf, 2 * nf, 2 * nf))
    print('  frozen rows  : %.5f overall' % frozen.mean())
    if W % 4 != 0 or W < 8:
      raise SystemExit('obs width %d is not a 2*NF + 2*NF frame stack' % W)
    # the goal half must be the tiled task goal, i.e. every goal frame equal
    gf = ob[:, :, 2 * nf:].reshape(n_eps, L, nf, 2)
    if float(np.abs(gf - gf[:, :, :1, :]).max()) != 0.0:
      raise SystemExit('the goal half is not a tiled constant goal')
  if arm == 'zfail':
    if not os.path.exists(BANK):
      maker = ('scripts/make_swamp_f4_failure_bank.py'
               if NORM[0] == '' else 'scripts/make_swamp_z_failure_bank.py')
      raise SystemExit('failure bank missing: %s\n  build it with %s'
                       % (BANK, maker))
    bs = content_sha(BANK)
    with np.load(BANK, allow_pickle=False) as b:
      g = np.asarray(b['goals'])
      bank_meta = json.loads(str(b['meta'])) if 'meta' in b else {}
    if bank_meta.get('source_content_sha256') != ds:
      raise SystemExit('failure bank was not built from the selected dataset; '
                       'rebuild it from the current dataset before training')
    print('  bank         : %s' % BANK)
    print('  bank content : %s' % bs)
    if NORM[0] == 'z_physical':
      print('  bank shape   : %s   z values %s'
            % (g.shape, sorted(set(float(v) for v in g[:, 2]))))
    else:
      print('  bank shape   : %s' % (g.shape,))
    if g.shape[1] != (W // 2):
      raise SystemExit('bank goal dim %d, expected %d' % (g.shape[1], W // 2))
    if g.shape[0] > BATCH_SIZE:
      raise SystemExit('bank (%d) > batch_size (%d)' % (g.shape[0],
                                                        BATCH_SIZE))
    if NORM[0] == 'z_physical' and not (g[:, 2] < 0).all():
      raise SystemExit('a bank entry is not below ground')
  else:
    print('  bank         : (none -- alpha 0)')
  if NORM[0]:
    print('  obs_norm     : %s, |z_min| = %.2f, applied ONCE inside '
          'crl/networks.py' % (NORM[0], NORM[1]))
  else:
    print('  obs_norm     : (none -- every coordinate is already in maze '
          'units)')
  print('=' * 78)
  print('PROVENANCE GATE PASSED')
  return {'code_commit': commit, 'head': head, 'dirty': dirty, 'arm': arm,
          'alpha': effective_alpha(arm), 'seed': seed, 'dataset': DATASET,
          'env_name': dataset_env, 'per_cell_swamp_prob': active_prob,
          'dataset_content_sha256': ds, 'n_episodes': int(n_eps),
          'n_transitions': int(n_eps * (L - 1)), 'n_failed_episodes': n_dead,
          'bank': BANK if arm == 'zfail' else None,
          'bank_content_sha256': content_sha(BANK) if arm == 'zfail' else None,
          'obs_norm_mode': NORM[0], 'obs_norm_z_scale': NORM[1],
          'batch_size': BATCH_SIZE, 'steps': STEPS}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--arm', choices=ARMS, default='zbase')
  ap.add_argument('--version', choices=sorted(VERSIONS), default='v0',
                  help='v0 = the accepted published config (default); v1 = the '
                       'z_v1 env with its own dataset, bank and run tag')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--alpha', type=float, default=None,
                  help='override the zfail mixture weight (0, 1); ignored for '
                       'zbase, which is alpha 0 by definition')
  ap.add_argument('--bank', default='',
                  help='override the version registry failure bank, e.g. a '
                       'composed one. Recorded in arm_provenance.json.')
  ap.add_argument('--ckpt-dir', default='')
  ap.add_argument('--diff', action='store_true')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true')
  ap.add_argument('--run', action='store_true')
  args = ap.parse_args()
  spec = select_version(args.version)
  select_alpha(args.alpha)
  select_bank(args.bank)

  if args.diff:
    config_diff(args.seed)
    return 0

  prov = gate(args.arm, args.seed)
  config_diff(args.seed)
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0
  if not (args.smoke or args.run):
    raise SystemExit('pass one of --diff / --check-only / --smoke / --run')

  # The alpha suffix is mandatory on zfail: without it every alpha in a sweep
  # writes into ONE directory and the later runs silently overwrite the earlier.
  tag = ('%s_zbase_s%d' % (spec['tag'], args.seed) if args.arm == 'zbase' else
         '%s_zfail_%s_s%d' % (spec['tag'], alpha_tag(effective_alpha('zfail')),
                              args.seed))
  ckpt = args.ckpt_dir or (tag + ('_smoke' if args.smoke else ''))
  steps = 2_000 if args.smoke else STEPS
  cfg = build_cfg(args.arm, ckpt, steps=steps, seed=args.seed)
  os.makedirs(ckpt, exist_ok=True)
  prov.update({'run_dir': ckpt, 'smoke': bool(args.smoke), 'steps': steps})
  with open(os.path.join(ckpt, 'arm_provenance.json'), 'w') as f:
    json.dump(prov, f, indent=2)
  print('\nrun dir: %s  (%s, %d steps)'
        % (ckpt, 'SMOKE' if args.smoke else 'PRODUCTION', steps))

  from crl.train import train
  train(cfg)
  print('\nDONE -> %s' % ckpt)
  return 0


if __name__ == '__main__':
  sys.exit(main())
