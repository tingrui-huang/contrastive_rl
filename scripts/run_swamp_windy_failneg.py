"""Sanctioned launcher for the windy-swamp (PointMaze) failure-negative line.

Pipeline (fixed here, not flag-configurable):

    the original 6000 episodes are retained WHOLE -- fixed length, as in the
    original CRL; no clean/fail split, no truncation, no 'lengths' field
  ->  ANCHORS restricted by scheme C (anchor_cut_mode='arrival'): episode
    uniform, then row uniform inside [0, cut), where cut is the first row the
    episode parks (within 0.5 of its goal) or freezes (absorbing death). The
    FUTURE-GOAL WINDOW IS NOT TOUCHED, so the geometric relabeling law and the
    set of samplable goals are exactly the original ones
  ->  ORDINARY in-batch negatives unchanged
  ->  PLUS failure-state negatives from artifacts/swamp_windy_failure_bank.

Three pre-registered arms isolate the two changes:

  baseline    anchor cut OFF, alpha 0   -- the current pipeline, reproduces the
                                           CONFOUNDED_SHORTCUT_BIAS reference
  anchorcut   anchor cut ON,  alpha 0   -- scheme C alone
  failneg     anchor cut ON,  alpha>0   -- scheme C + failure-state negatives

Why a launcher: crl.config defaults silently pick the EASIER setting for every
knob that matters here (anchor_cut_mode '', fail_neg_alpha 0, batch_size 256 --
which would additionally trip the n_bank <= batch_size guard). Pinning them in
module constants is the same discipline as run_causal_transition_v0.py.

Evaluate a finished run with the established harness:
  python -m scripts.eval_swamp_windy_deployment --ckpt <run>/final.pkl \\
      --out artifacts/<name> --episodes 100

Usage:
  python scripts/run_swamp_windy_failneg.py --check-only
  python scripts/run_swamp_windy_failneg.py --arm failneg --smoke
  python scripts/run_swamp_windy_failneg.py --arm failneg --alpha 0.1 --run
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

ENV = 'point_two_route_swamp_windy_v0'
DATASET = 'datasets/swamp_windy_teacher_s0.npz'
DATASET_SHA = 'dfdbbaf7b6a62754f8c865257cea4f3d271ba524f4510c8459ca6fc901e1bfee'
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
BANK_SHA = '719229969f8b025acebd11d7c15a59ff5cf58f6b5fe94b70dda49cc57e193fbf'
BANK_N = 514

# Recipe. batch_size MUST be >= BANK_N (crl/losses.py writes the bank over the
# first n_bank rows of the goal half), hence 1024 rather than the 256 default.
BATCH_SIZE = 1024
STEPS = 150_000
ANCHOR_CUT_RADIUS = 0.5
ARMS = ('baseline', 'anchorcut', 'failneg')


def sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def gate(arm, alpha):
  """Refuse to start unless every artifact is the pre-registered one."""
  print('=' * 70)
  print('WINDY-SWAMP FAILURE-NEGATIVE PROVENANCE GATE')
  try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                     cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    commit = '(unavailable)'
  print(f'  git commit        : {commit}')
  print(f'  arm               : {arm}')
  print(f'  env               : {ENV}')

  if not os.path.exists(DATASET):
    raise SystemExit(f'dataset missing: {DATASET}')
  ds = sha256(DATASET)
  print(f'  dataset           : {DATASET}')
  print(f'  dataset sha       : {ds}')
  if ds != DATASET_SHA:
    raise SystemExit(f'dataset sha mismatch\n  expected {DATASET_SHA}\n'
                     f'  found    {ds}')
  with np.load(DATASET, allow_pickle=True) as d:
    n_eps, L = d['obs'].shape[0], d['obs'].shape[1]
    if 'lengths' in d:
      raise SystemExit(
          "dataset carries a 'lengths' field: this pipeline requires the "
          'fixed-length dataset (scheme C is anchor-side only and is mutually '
          'exclusive with the length-truncation path).')
  if (n_eps, L) != (6000, 51):
    raise SystemExit(f'expected 6000 x 51 episodes, found {n_eps} x {L}')
  print(f'  episodes          : {n_eps} x {L} rows (retained WHOLE)')

  use_cut = arm in ('anchorcut', 'failneg')
  print(f'  anchor_cut_mode   : '
        f'{"arrival (scheme C)" if use_cut else "'' (original draw)"}')
  if use_cut:
    print(f'  anchor_cut_radius : {ANCHOR_CUT_RADIUS}')

  if arm == 'failneg':
    if not os.path.exists(BANK):
      raise SystemExit(f'failure bank missing: {BANK}\n'
                       'build it with scripts/make_swamp_failure_bank.py')
    bs = sha256(BANK)
    print(f'  failure bank      : {BANK}')
    print(f'  failure bank sha  : {bs}')
    if bs != BANK_SHA:
      raise SystemExit(f'bank sha mismatch\n  expected {BANK_SHA}\n'
                       f'  found    {bs}')
    with np.load(BANK, allow_pickle=True) as b:
      g = np.asarray(b['goals'])
    if g.shape != (BANK_N, 2):
      raise SystemExit(f'bank shape {g.shape}, expected ({BANK_N}, 2)')
    if g.shape[0] > BATCH_SIZE:
      raise SystemExit(f'bank ({g.shape[0]}) > batch_size ({BATCH_SIZE}); '
                       'crl/losses.py requires n_bank <= batch_size')
    print(f'  failure bank size : {g.shape[0]} x {g.shape[1]}')
    print(f'  fail_neg_alpha    : {alpha}')
    if not 0.0 < alpha < 1.0:
      raise SystemExit(f'--alpha must be in (0, 1) for the failneg arm, '
                       f'got {alpha}')
  else:
    print('  failure bank      : (none -- ordinary negatives only)')
  print(f'  batch_size        : {BATCH_SIZE}   steps: {STEPS}')
  print('=' * 70)
  print('PROVENANCE GATE PASSED')
  return {'git_commit': commit, 'arm': arm, 'dataset_sha256': ds,
          'bank_sha256': BANK_SHA if arm == 'failneg' else None,
          'fail_neg_alpha': alpha if arm == 'failneg' else 0.0,
          'anchor_cut_mode': 'arrival' if use_cut else '',
          'batch_size': BATCH_SIZE, 'steps': STEPS}


def build_cfg(arm, alpha, seed, ckpt_dir, steps):
  use_cut = arm in ('anchorcut', 'failneg')
  return Config(
      env_name=ENV, offline_dataset=DATASET,
      max_number_of_steps=steps,
      # scheme C -- anchor side only; future window untouched.
      anchor_cut_mode='arrival' if use_cut else '',
      anchor_cut_radius=ANCHOR_CUT_RADIUS,
      # failure-state negatives (extra negative goals).
      fail_bank_path=BANK if arm == 'failneg' else '',
      fail_neg_alpha=alpha if arm == 'failneg' else 0.0,
      # offline contrastive recipe (binary NCE, twin-min actor, BC-regularized)
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=True,
      bc_coef=0.05, random_goals=0.0,
      entropy_coefficient=0.0, target_entropy=0.0,
      batch_size=BATCH_SIZE, repr_dim=16, hidden_layer_sizes=(256, 256),
      discount=0.99, learning_rate=3e-4, actor_learning_rate=3e-4,
      num_sgd_steps_per_step=1, num_actors=0,
      guard_abort=True, jit=True, seed=seed,
      eval_every_steps=10_000, eval_episodes=50, log_every_steps=1_000,
      ckpt_dir=ckpt_dir)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--arm', choices=ARMS, default='failneg')
  ap.add_argument('--alpha', type=float, default=0.1,
                  help='fail_neg_alpha; used by the failneg arm only')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true',
                  help='gate + a handful of learner updates, then stop')
  ap.add_argument('--run', action='store_true')
  args = ap.parse_args()

  prov = gate(args.arm, args.alpha)
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0
  if not (args.smoke or args.run):
    raise SystemExit('pass one of --check-only / --smoke / --run')

  tag = (f'swamp_windy_{args.arm}'
         + (f'_a{args.alpha:g}'.replace('.', '') if args.arm == 'failneg' else '')
         + f'_s{args.seed}')
  ckpt = args.ckpt_dir or (tag + ('_smoke' if args.smoke else ''))
  steps = 2_000 if args.smoke else STEPS
  cfg = build_cfg(args.arm, args.alpha, args.seed, ckpt, steps)
  os.makedirs(ckpt, exist_ok=True)
  prov['run_dir'] = ckpt
  prov['smoke'] = bool(args.smoke)
  prov['seed'] = args.seed
  with open(os.path.join(ckpt, 'arm_provenance.json'), 'w') as f:
    json.dump(prov, f, indent=2)
  print(f'\nrun dir: {ckpt}  ({"SMOKE" if args.smoke else "PRODUCTION"}, '
        f'{steps} steps)')
  print(json.dumps({k: v for k, v in dataclasses.asdict(cfg).items()
                    if k in ('env_name', 'offline_dataset', 'anchor_cut_mode',
                             'anchor_cut_radius', 'fail_bank_path',
                             'fail_neg_alpha', 'batch_size', 'bc_coef',
                             'twin_q', 'random_goals', 'discount', 'seed')},
                   indent=2))

  from crl.train import train
  train(cfg)
  print(f'\nDONE -> {ckpt}')
  print('evaluate with:\n  python -m scripts.eval_swamp_windy_deployment '
        f'--ckpt {ckpt}/final.pkl --out artifacts/{tag} --episodes 100')
  return 0


if __name__ == '__main__':
  sys.exit(main())
