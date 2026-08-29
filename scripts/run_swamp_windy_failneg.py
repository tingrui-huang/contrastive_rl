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
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
BANK_N = 256

# Provenance is pinned on the ARRAY CONTENTS, not the file bytes.
#
# An .npz is a zip container that embeds per-entry timestamps, so the file
# sha256 of a regenerated dataset differs from the original even when every
# array is bit-identical -- confirmed on a fresh GPU node, where the file sha
# changed but content_sha256 matched exactly. datasets/ is gitignored, so the
# dataset is ALWAYS regenerated on a new node; gating on the file sha would
# reject a perfectly correct regeneration. The file shas below are recorded
# for reference only and are never enforced.
BANK_CONTENT_SHA = \
    'b680aab6b224ec5b1243058a54c678d5ab8897935106b84a4addab5429fa5381'
# informational only (valid for the git-tracked bank and the original dataset)
DATASET_FILE_SHA_REF = \
    'dfdbbaf7b6a62754f8c865257cea4f3d271ba524f4510c8459ca6fc901e1bfee'
BANK_FILE_SHA_REF = \
    'b236458e2e45d8bfee3420a3e0e514a804a64d165a34c2772774b294c089077d'

# ---------------------------------------------------------------- datasets
# Which frozen dataset a run trains on. Swapping the dataset is a change of
# the same weight as swapping an arm, so it goes through a REGISTRY with a
# per-entry enforced content hash -- not a free-form --dataset-path flag,
# which would let any file through the provenance gate.
#
#   main        the original 6000 teacher episodes; the run behind the
#               CONFOUNDED_SHORTCUT_BIAS reference.
#   merged_s{n} main + 600 bad-demonstrator episodes (teacher_mode 4) = 6600.
#               The bad demonstrator is the BFS shortcut oracle with the dodge
#               step removed: it restores support for the (forward | U active)
#               branch that the gate-aware teacher visits 0/501 times. It is
#               POSITIVE-side data and does NOT touch the failure bank -- see
#               scripts/collect_swamp_windy_baddemo.py for why that distinction
#               is the whole point. n indexes the bad demo's collection seed,
#               NOT the learner seed; the main half is identical in all three.
#
# Content hashes below cover every array including `meta`, and the merge script
# keeps its meta reproducible (source basenames + source CONTENT hashes, never
# absolute paths or file shas) -- verified by regenerating into a different
# directory and getting an identical hash.
_REGEN_MAIN = ('python -m scripts.collect_swamp_windy --episodes 6000 '
               '--random_frac 0.2 --force_safe_prob 0.05 '
               '--teacher_noise 0.15 --seed 0 --out {path}')
_REGEN_MERGED = (
    'python -m scripts.collect_swamp_windy_baddemo --episodes 600 --seed {n} '
    '--out datasets/swamp_windy_baddemo_s{n}.npz\n'
    '  python -m scripts.merge_swamp_windy_baddemo '
    '--bad datasets/swamp_windy_baddemo_s{n}.npz --out {path}')
DATASETS = {
    'main': dict(
        path='datasets/swamp_windy_teacher_s0.npz',
        content_sha='fd41c45cdb72749fb3b5a071c6f65a3003ec3117af630222f672'
                    '6bfab7ea7952',
        shape=(6000, 51), regen=_REGEN_MAIN),
    'merged_s0': dict(
        path='datasets/swamp_windy_merged_s0.npz',
        content_sha='61bd4ce1b89a0edb951132b85ce705b64fa6753fb89f564026b3'
                    'e98d0586e767',
        shape=(6600, 51), regen=_REGEN_MERGED.format(n=0, path='{path}')),
    'merged_s1': dict(
        path='datasets/swamp_windy_merged_s1.npz',
        content_sha='22db06deb3e04622c29e0585cb73147a0d5bb26bbba6482283d5'
                    '5420eed17162',
        shape=(6600, 51), regen=_REGEN_MERGED.format(n=1, path='{path}')),
    'merged_s2': dict(
        path='datasets/swamp_windy_merged_s2.npz',
        content_sha='14040fa1ea52e0b46848f9ac173aa98459f130471bda02b5cbd4'
                    '20700f3ea26a',
        shape=(6600, 51), regen=_REGEN_MERGED.format(n=2, path='{path}')),
}
DATASET = DATASETS['main']['path']        # default; --dataset overrides

# 256 is the established windy recipe's batch size. crl/losses.py requires
# n_bank <= batch_size (the bank is written over the first n_bank rows of the
# goal half), so the 514-state bank is subsampled to 256 -- see
# make_swamp_failure_bank.py --max-bank. That changes only the RESOLUTION of
# q_fail, not alpha's meaning: alpha is still the share of negative mass.
BATCH_SIZE = 256
STEPS = 150_000
ANCHOR_CUT_RADIUS = 0.5
ARMS = ('baseline', 'anchorcut', 'failneg', 'balanced',
        'balancedfail')

# Gradient updates batched into ONE jax.lax.scan dispatch.
#
# This model is tiny (256x256 encoders on 2-dim obs), so a step is dominated by
# dispatch + host sync, not GPU compute -- measured on a 3090: 8.77 ms/update
# at G=1 but 3.95 ms at G=10 (114 -> 253 steps/s, a 150k run 21.9 -> 9.9 min).
#
# It does NOT change the math. In offline mode the buffer is frozen and its RNG
# is independent of the learner state, so the G batches are exactly the ones
# G=1 would have drawn, and scan applies the same updates in the same order;
# only the number of Python dispatches changes. Logged metrics become means
# over the G scanned steps.
#
# MUST divide max_episode_steps (50): train.py computes
# learner_steps = updates_per_step * (N_ACT * max_episode_steps) // G, so a
# non-divisor silently drops updates (G=4 -> 48 instead of 50 per iteration).
SGD_STEPS_PER_STEP = 10

# Balanced (s,a) bucketing. cell 1.0 x 4 rotated sectors (+wait) was chosen
# because it is the coarsest option measured, hence the fewest near-empty
# buckets: uniform-over-buckets amplifies noise from buckets holding one or
# two samples, and the finer grids measured had min bucket size 1.
BALANCED_CELL = 1.0
BALANCED_SECTORS = 4
# Bucket weight ceiling: draw a bucket with probability proportional to
# min(count, cap). Strict uniform sent 22.2% of every batch to buckets backed
# by 348 of 93,779 rows (a one-row bucket is amplified ~1000x). Measured
# cap sweep: the fork rebalancing is flat at 1.93-1.99x for every cap, while
# tiny-bucket draws fall 22.2% -> 2.0% at 300 (~ the median bucket size), so
# the cap costs nothing and removes the noise.
BALANCED_CAP = 300


def sha256(path, chunk=1 << 20):
  """Raw file bytes. For .npz this is NOT reproducible across machines."""
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def content_sha256(path):
  """SHA-256 over the ARRAY CONTENTS of an .npz, ignoring zip metadata.

  Hashes each key (sorted) with its dtype, shape and raw bytes, so it is invariant
  to the timestamps and entry ordering the zip container embeds, and therefore
  reproducible across machines and regenerations.
  """
  h = hashlib.sha256()
  with np.load(path, allow_pickle=True) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode())
      h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode())
      h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def gate(arm, alpha, dataset='main'):
  """Refuse to start unless every artifact is the pre-registered one."""
  if dataset not in DATASETS:
    raise SystemExit(f'unknown --dataset {dataset!r}; '
                     f'registered: {sorted(DATASETS)}')
  spec = DATASETS[dataset]
  ds_path, ds_sha, ds_shape = spec['path'], spec['content_sha'], spec['shape']
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

  if not os.path.exists(ds_path):
    raise SystemExit(f'dataset missing: {ds_path}\n'
                     'datasets/ is gitignored -- regenerate it with:\n  '
                     + spec['regen'].format(path=ds_path))
  ds = content_sha256(ds_path)
  print(f'  dataset           : {dataset}  ->  {ds_path}')
  print(f'  dataset content   : {ds}   <- ENFORCED')
  print(f'  dataset file sha  : {sha256(ds_path)[:16]}...  (informational; '
        'npz zip metadata is not reproducible)')
  if ds != ds_sha:
    raise SystemExit(f'dataset CONTENT mismatch -- the arrays differ, not just '
                     f'the container\n  expected {ds_sha}\n'
                     f'  found    {ds}\n'
                     '  regenerate with:\n    '
                     + spec['regen'].format(path=ds_path))
  with np.load(ds_path, allow_pickle=True) as d:
    n_eps, L = d['obs'].shape[0], d['obs'].shape[1]
    if 'lengths' in d:
      raise SystemExit(
          "dataset carries a 'lengths' field: this pipeline requires the "
          'fixed-length dataset (scheme C is anchor-side only and is mutually '
          'exclusive with the length-truncation path).')
    modes = np.unique(np.asarray(d['teacher_mode'])) if 'teacher_mode' in d \
        else np.array([])
  if (n_eps, L) != tuple(ds_shape):
    raise SystemExit(f'expected {ds_shape[0]} x {ds_shape[1]} episodes, '
                     f'found {n_eps} x {L}')
  print(f'  episodes          : {n_eps} x {L} rows (retained WHOLE)')
  # teacher_mode 4 is the bad demonstrator. Print it so the log says which
  # dataset a checkpoint came from without re-reading the npz, and refuse the
  # two ways the registry and the file can disagree.
  n_bad = 0
  if modes.size:
    with np.load(ds_path, allow_pickle=True) as d:
      n_bad = int((np.asarray(d['teacher_mode']) == 4).sum())
  if dataset.startswith('merged') and n_bad == 0:
    raise SystemExit(f'{dataset} is registered as a merged dataset but holds '
                     'no teacher_mode==4 episodes')
  if dataset == 'main' and n_bad:
    raise SystemExit(f'main dataset holds {n_bad} bad-demonstrator episodes; '
                     'it is supposed to be the untouched reference')
  print(f'  bad-demo episodes : {n_bad}'
        + ('  (teacher_mode 4, POSITIVE-side support)' if n_bad else ''))

  use_cut = arm in ('anchorcut', 'failneg', 'balanced',
                    'balancedfail')
  # NB: keep nested quotes out of f-strings -- python 3.11 (the node's
  # interpreter) predates PEP 701 and rejects them.
  cut_desc = 'arrival (scheme C)' if use_cut else "'' (original draw)"
  print(f'  anchor_cut_mode   : {cut_desc}')
  if use_cut:
    print(f'  anchor_cut_radius : {ANCHOR_CUT_RADIUS}')

  if arm in ('balanced', 'balancedfail'):
    print(f'  balanced (s,a)    : cell {BALANCED_CELL}, {BALANCED_SECTORS} '
          'rotated sectors + wait bucket')
  if arm in ('failneg', 'balancedfail'):
    if not os.path.exists(BANK):
      raise SystemExit(f'failure bank missing: {BANK}\n'
                       'build it with scripts/make_swamp_failure_bank.py')
    bs = content_sha256(BANK)
    print(f'  failure bank      : {BANK}')
    print(f'  bank content      : {bs}   <- ENFORCED')
    print(f'  bank file sha     : {sha256(BANK)[:16]}...  (informational)')
    if bs != BANK_CONTENT_SHA:
      raise SystemExit(f'bank CONTENT mismatch\n  expected {BANK_CONTENT_SHA}\n'
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
  if STEPS % SGD_STEPS_PER_STEP or 50 % SGD_STEPS_PER_STEP:
    raise SystemExit(f'SGD_STEPS_PER_STEP={SGD_STEPS_PER_STEP} must divide '
                     'max_episode_steps (50), else train.py drops updates')
  print(f'  batch_size        : {BATCH_SIZE}   steps: {STEPS}   '
        f'sgd/step: {SGD_STEPS_PER_STEP}')
  print('=' * 70)
  print('PROVENANCE GATE PASSED')
  return {'git_commit': commit, 'arm': arm,
          'dataset': dataset, 'dataset_path': ds_path,
          'dataset_n_episodes': int(n_eps),
          'dataset_n_bad_demo': int(n_bad),
          'dataset_content_sha256': ds,
          'dataset_file_sha256': sha256(ds_path),
          'bank_content_sha256': (BANK_CONTENT_SHA
                                  if arm in ('failneg', 'balancedfail')
                                  else None),
          'fail_neg_alpha': (alpha if arm in ('failneg', 'balancedfail')
                             else 0.0),
          'balanced_sampling': arm in ('balanced', 'balancedfail'),
          'anchor_cut_mode': 'arrival' if use_cut else '',
          'batch_size': BATCH_SIZE, 'steps': STEPS}


def build_cfg(arm, alpha, seed, ckpt_dir, steps, dataset='main'):
  # `dataset` is LAST and defaults to 'main' so the existing five-positional
  # call in scripts/probe_swamp_windy_critic.py keeps working -- that probe
  # only needs the network shape, which no dataset choice affects.
  use_cut = arm in ('anchorcut', 'failneg', 'balanced', 'balancedfail')
  use_bal = arm in ('balanced', 'balancedfail')
  use_bank = arm in ('failneg', 'balancedfail')
  return Config(
      env_name=ENV, offline_dataset=DATASETS[dataset]['path'],
      max_number_of_steps=steps,
      # scheme C -- anchor side only; future window untouched.
      anchor_cut_mode='arrival' if use_cut else '',
      anchor_cut_radius=ANCHOR_CUT_RADIUS,
      # failure-state negatives (extra negative goals).
      fail_bank_path=BANK if use_bank else '',
      fail_neg_alpha=alpha if use_bank else 0.0,
      balanced_sampling=use_bal,
      balanced_cell_size=BALANCED_CELL,
      balanced_action_sectors=BALANCED_SECTORS,
      balanced_cap=BALANCED_CAP,
      # THE ESTABLISHED WINDY-SWAMP RECIPE -- do not substitute the AntMaze
      # rockfall one (bc 0.05 / twin_q / repr 16 / gamma 0.99 / batch 1024).
      # Every value below is read off the Config banner of
      # swamp_windy_manski_s0_train.log, the run behind the
      # CONFOUNDED_SHORTCUT_BIAS reference in
      # artifacts/windy_manski_s0_deployment/.
      #   bc_coef 0.5      = the paper's offline actor lambda; AWR is NOT in
      #                      the paper and measured no better, so the paper
      #                      actor at its own lambda is the faithful choice.
      #   discount 0.95    = gamma 0.99 is near-vacuous on a 50-step maze
      #                      (mean geometric horizon 100 steps); the earlier
      #                      gamma sweep picked 0.95 as the working point.
      #   twin_q False, repr_dim 64, random_goals 0.5, batch_size 256.
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=0.5, random_goals=0.5,
      entropy_coefficient=0.0, target_entropy=0.0,
      batch_size=BATCH_SIZE, repr_dim=64, hidden_layer_sizes=(256, 256),
      discount=0.95, learning_rate=3e-4, actor_learning_rate=3e-4,
      num_sgd_steps_per_step=SGD_STEPS_PER_STEP, num_actors=0,
      guard_abort=True, jit=True, seed=seed,
      eval_every_steps=10_000, eval_episodes=50, log_every_steps=1_000,
      ckpt_dir=ckpt_dir)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--arm', choices=ARMS, default='failneg')
  ap.add_argument('--alpha', type=float, default=0.1,
                  help='fail_neg_alpha; used by the failneg arm only')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--dataset', choices=sorted(DATASETS), default='main',
                  help='registered frozen dataset; merged_s* add the 600 '
                       'bad-demonstrator episodes')
  ap.add_argument('--ckpt-dir', default='')
  ap.add_argument('--check-only', action='store_true')
  ap.add_argument('--smoke', action='store_true',
                  help='gate + a handful of learner updates, then stop')
  ap.add_argument('--run', action='store_true')
  args = ap.parse_args()

  prov = gate(args.arm, args.alpha, args.dataset)
  if args.check_only:
    print('CHECK-ONLY COMPLETE (no training performed)')
    return 0
  if not (args.smoke or args.run):
    raise SystemExit('pass one of --check-only / --smoke / --run')

  # The dataset goes in the tag for everything but 'main', so a merged-data run
  # never overwrites the reference run of the same arm and seed. 'main' keeps
  # its bare name so the 27 existing run directories stay addressable.
  # merged_s<n> is abbreviated to bd<n> ("bad demo seed n"): spelling it out
  # produced swamp_windy_baseline_merged_s0_s0, whose two _s0 mean different
  # things (collection seed, learner seed) and read as a typo.
  ds_tag = '' if args.dataset == 'main' \
      else '_bd' + args.dataset.rsplit('_s', 1)[-1]
  tag = (f'swamp_windy_{args.arm}'
         + (f'_a{args.alpha:g}'.replace('.', '')
            if args.arm in ('failneg', 'balancedfail') else '')
         + ds_tag + f'_s{args.seed}')
  ckpt = args.ckpt_dir or (tag + ('_smoke' if args.smoke else ''))
  steps = 2_000 if args.smoke else STEPS
  cfg = build_cfg(args.arm, args.alpha, args.seed, ckpt, steps, args.dataset)
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
