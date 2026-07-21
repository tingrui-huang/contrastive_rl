"""Oracle DIAGNOSTIC subsets of the full rockfall learner npz (step 5).

These are NOT fair baselines: they use the privileged sidecar `route` label
to select/upweight episodes. Each output keeps the learner contract exactly
(obs/act/eval_goals/lengths/meta only) so scripts/naive_rockfall_crl.py runs
on them unchanged (`--npz <subset>`).

  center_only : only episodes whose actual route == 'center'. Checks whether
                center behaviour is LEARNABLE from the offline data.
  reweight    : full episode set, but center-route episodes are REPLICATED so
                center transitions reach --center-frac of the total. Checks
                whether correct weighting pushes the CRL pipeline toward the
                robust route. (Episode replication oversamples uniformly at
                the transition level without touching the training code.)

Usage:
  python scripts/make_rockfall_subsets.py --mode center_only
  python scripts/make_rockfall_subsets.py --mode reweight --center-frac 0.5
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C           # noqa: E402

FULL_DIR = 'artifacts/rockfall_dataset/full'
FULL_NAME = 'antmaze_rockfall_full'
OUT_DIR = 'artifacts/rockfall_dataset/oracle'


def load_full(full_dir, name):
  d = np.load(os.path.join(full_dir, f'{name}.npz'), allow_pickle=True)
  s = np.load(os.path.join(full_dir, f'{name}_sidecar.npz'), allow_pickle=True)
  return d, s


def write_subset(out_path, d, idx, note):
  """Write a learner-contract npz from full arrays and an episode index list
  (indices may repeat -> transition-level oversampling)."""
  idx = np.asarray(idx, int)
  meta = json.loads(str(d['meta']))
  meta = dict(meta)
  meta['note'] = note
  obs = d['obs'][idx]
  act = d['act'][idx]
  eval_goals = d['eval_goals'][idx]
  lengths = d['lengths'][idx]
  tmp = out_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, obs=obs, act=act, eval_goals=eval_goals,
                        lengths=lengths, meta=json.dumps(meta))
  os.replace(tmp, out_path)
  return int(len(idx)), int((lengths - 1).sum())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--mode', required=True,
                  choices=('center_only', 'reweight'))
  ap.add_argument('--center-frac', type=float, default=0.5,
                  help='target center transition fraction (reweight mode)')
  ap.add_argument('--full-dir', default=FULL_DIR)
  ap.add_argument('--full-name', default=FULL_NAME)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  d, s = load_full(args.full_dir, args.full_name)
  route = s['route']
  lengths = d['lengths']
  n = len(lengths)
  trans = (lengths - 1)
  center_e = np.where(route == 'center')[0]
  other_e = np.where(route != 'center')[0]
  print(f'full: {n} eps, {int(trans.sum())} transitions, '
        f'center eps {len(center_e)}', flush=True)

  if args.mode == 'center_only':
    idx = center_e
    name = f'{args.full_name}_center_only'
    note = ('ORACLE DIAGNOSTIC (sidecar route label): center-route episodes '
            'ONLY. Not a fair baseline.')
  else:
    # replicate center episodes so center transitions ~= center_frac total.
    ct = int(trans[center_e].sum())
    ot = int(trans[other_e].sum())
    # solve r: (r*ct) / (r*ct + ot) = f  ->  r = f*ot / ((1-f)*ct)
    f = args.center_frac
    r = max(1, round(f * ot / max((1 - f) * ct, 1)))
    idx = np.concatenate([other_e] + [center_e] * r)
    achieved = (r * ct) / (r * ct + ot)
    name = f'{args.full_name}_reweight_c{int(f * 100)}'
    note = (f'ORACLE DIAGNOSTIC (sidecar route label): center episodes '
            f'replicated x{r} -> center transition frac {achieved:.3f}. '
            'Not a fair baseline.')
    print(f'reweight: center x{r} -> center transition frac {achieved:.3f}',
          flush=True)

  out_path = os.path.join(args.out_dir, f'{name}.npz')
  neps, ntr = write_subset(out_path, d, idx, note)
  man = {'source_full_npz': os.path.join(args.full_dir,
                                         f'{args.full_name}.npz'),
         'source_sha256': C.sha256_file(os.path.join(
             args.full_dir, f'{args.full_name}.npz')),
         'mode': args.mode, 'center_frac': args.center_frac,
         'episodes_written': neps, 'transitions_written': ntr,
         'npz_sha256': C.sha256_file(out_path),
         'note': note, 'git_commit': C.git_commit(),
         'oracle_label_source': 'sidecar route (privileged)'}
  json.dump(man, open(os.path.join(args.out_dir, f'{name}_manifest.json'),
                      'w'), indent=2)
  print(f'wrote {out_path}: {neps} eps, {ntr} transitions', flush=True)
  print('sha256:', man['npz_sha256'])


if __name__ == '__main__':
  main()
