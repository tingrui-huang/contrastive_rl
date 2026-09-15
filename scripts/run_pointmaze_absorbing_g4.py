r"""G4: from-scratch offline CRL on PointMaze F4, observational (O) vs absorbing (P) dataset.

Both arms run the established windy F4 recipe of ``run_swamp_windy_z_failneg.py``
(zbase arm: alpha 0, no failure bank, 150k steps, batch 256, ten SGD steps per
step, repr 64, (256, 256), discount 0.95, random_goals 0.5, entropy 0) with
one difference between the arms -- the offline dataset:

  O  datasets/swamp_windy_f4_merged_s0.npz        the recorded trajectories
  P  datasets/swamp_windy_f4_absorbing_s0.npz     the same trajectories frozen
                                                  at the first Manski onset
                                                  (scripts/build_pointmaze_absorbing_dataset.py)

``--bc`` sets the actor's BC coefficient (paper objective
``(1-bc) E_pi[Q] + bc log pi(a_data|s,g)``) identically for both arms; the
faithful G3 candidate is 0.2, the original recipe 0.5.  Nothing else differs;
``--diff`` asserts it.  Training-time evaluation (every 10k steps, 50 episodes,
policy mode) is the repo's greedy-rollout success; route usage is measured
afterwards with ``scripts/eval_pointmaze_native_routes.py``.

Usage::

    python -m scripts.run_pointmaze_absorbing_g4 --arm P --bc 0.2 --seed 0 --check-only
    python -m scripts.run_pointmaze_absorbing_g4 --arm P --bc 0.2 --seed 0 --smoke
    python -m scripts.run_pointmaze_absorbing_g4 --arm P --bc 0.2 --seed 0 --run
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
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config                      # noqa: E402

ENV = 'point_two_route_swamp_windy_f4_v0'
DATASETS = {'O': 'datasets/swamp_windy_f4_merged_s0.npz',
            'P': 'datasets/swamp_windy_f4_absorbing_s0.npz'}
EXPECTED_CONTENT_SHA = {
    'O': 'ad8b4470df0ebcc1c03421dfb59b61f9faee48cfbf51e2b9c0a402e0403f544d',
    'P': '6e046bdce64e1dc29f4240441ec45091eef9bd6b830eaeb9f2964c2b3213d754',
}
BATCH_SIZE = 256
STEPS = 150_000
SGD_STEPS_PER_STEP = 10


def content_sha(path):
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode()); h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def build_cfg(arm, bc, ckpt_dir, steps=STEPS, seed=0):
  return Config(
      env_name=ENV, offline_dataset=DATASETS[arm],
      max_number_of_steps=steps,
      fail_bank_path='', fail_neg_alpha=0.0,
      obs_norm_mode='', obs_norm_z_scale=0.0,
      anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=bc, random_goals=0.5,
      entropy_coefficient=0.0, target_entropy=0.0,
      batch_size=BATCH_SIZE, repr_dim=64, hidden_layer_sizes=(256, 256),
      discount=0.95, learning_rate=3e-4, actor_learning_rate=3e-4,
      num_sgd_steps_per_step=SGD_STEPS_PER_STEP, num_actors=0,
      guard_abort=True, jit=True, seed=seed,
      eval_every_steps=10_000, eval_episodes=50, log_every_steps=1_000,
      ckpt_dir=ckpt_dir)


def config_diff(bc, seed):
  a = dataclasses.asdict(build_cfg('O', bc, 'X', seed=seed))
  b = dataclasses.asdict(build_cfg('P', bc, 'X', seed=seed))
  diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
  print('CONFIG DIFF  O vs P  (bc=%g, seed=%d)' % (bc, seed))
  for k, (x, y) in sorted(diff.items()):
    print('  %-22s %r -> %r' % (k, x, y))
  if set(diff) != {'offline_dataset'}:
    raise SystemExit('arms differ in fields other than the dataset: %s' % sorted(diff))
  print('  ASSERTION PASSED: the only difference is the offline dataset.')


def gate(arm, bc, seed):
  root = os.path.dirname(_HERE)
  try:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'crl', 'scripts'], cwd=root).decode().strip())
  except Exception:                                # pylint: disable=broad-except
    head, dirty = '(unavailable)', None
  path = DATASETS[arm]
  if not os.path.exists(path):
    raise SystemExit('dataset missing: %s' % path)
  sha = content_sha(path)
  if sha != EXPECTED_CONTENT_SHA[arm]:
    raise SystemExit('dataset content sha mismatch for arm %s: %s (expected %s)'
                     % (arm, sha, EXPECTED_CONTENT_SHA[arm]))
  with np.load(path, allow_pickle=False) as d:
    meta = json.loads(str(d['meta']))
    n_eps, L, W = d['obs'].shape
    n_dead = int(np.asarray(d['entered_active_swamp']).sum())
    onset = np.asarray(d['audit_absorbing_onset_time']) if 'audit_absorbing_onset_time' in d else None
  if meta['env_name'] != ENV or abs(float(meta['per_cell_swamp_prob']) - 0.3) > 1e-8:
    raise SystemExit('dataset environment mismatch: %s' % meta.get('env_name'))
  print('=' * 78)
  print('G4 PROVENANCE GATE   arm %s   bc %g   seed %d' % (arm, bc, seed))
  print('  head         : %s%s' % (head, '   (crl/scripts DIRTY)' if dirty else ''))
  print('  dataset      : %s   content sha %s' % (path, sha[:16]))
  print('  episodes     : %d x %d rows, obs width %d' % (n_eps, L, W))
  print('  recorded deaths: %d   added absorbing onsets: %s'
        % (n_dead, 'n/a' if onset is None else int((onset >= 0).sum())))
  print('  bc_coef      : %g   random_goals 0.5   steps %d x %d sgd' % (bc, STEPS, SGD_STEPS_PER_STEP))
  print('=' * 78)
  return {'head': head, 'dirty': dirty, 'arm': arm, 'bc_coef': bc, 'seed': seed,
          'dataset': path, 'dataset_content_sha256': sha, 'env_name': ENV,
          'n_episodes': int(n_eps), 'n_failed_episodes': n_dead,
          'added_absorbing_onsets': None if onset is None else int((onset >= 0).sum()),
          'batch_size': BATCH_SIZE, 'steps': STEPS, 'sgd_steps_per_step': SGD_STEPS_PER_STEP,
          'recipe': 'windy F4 zbase recipe, alpha 0, no bank; only the dataset differs between arms'}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--arm', choices=sorted(DATASETS), required=True)
  ap.add_argument('--bc', type=float, default=0.2)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='')
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--diff', action='store_true')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true')
  ap.add_argument('--run', action='store_true')
  args = ap.parse_args()
  if args.diff:
    config_diff(args.bc, args.seed)
    return 0
  prov = gate(args.arm, args.bc, args.seed)
  config_diff(args.bc, args.seed)
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0
  if not (args.smoke or args.run):
    raise SystemExit('pass one of --diff / --check-only / --smoke / --run')
  tag = 'g4_%s_bc%s_s%d' % (args.arm, ('%g' % args.bc).replace('.', 'p'), args.seed)
  ckpt = args.ckpt_dir or os.path.join('runs', 'pointmaze_absorbing_g4', tag + ('_smoke' if args.smoke else ''))
  steps = 2_000 if args.smoke else args.steps
  cfg = build_cfg(args.arm, args.bc, ckpt, steps=steps, seed=args.seed)
  os.makedirs(ckpt, exist_ok=True)
  prov.update({'run_dir': ckpt, 'smoke': bool(args.smoke), 'steps': steps})
  with open(os.path.join(ckpt, 'arm_provenance.json'), 'w') as f:
    json.dump(prov, f, indent=2)
  print('\nrun dir: %s  (%s, %d steps)' % (ckpt, 'SMOKE' if args.smoke else 'PRODUCTION', steps))
  from crl.train import train
  train(cfg)
  print('\nDONE -> %s' % ckpt)
  return 0


if __name__ == '__main__':
  sys.exit(main())
