"""Derive the XY+VELOCITY-goal (gxyv) copy of a V5 dataset.

Same idea as scripts/make_v5_gxy_dataset.py, one representation further out.
The V5 collector stores the env observation under this port's historical
contract, [state(29) | goal_xy(2) | 27 zeros] = 58 columns. The gxy file keeps
the 31 columns the upstream ant contract commands ([state | goal xy]); this
file keeps 37: [state(29) | goal[GOAL_INDICES_XYV]] where the goal half is the
SAME zero-padded goal vector sliced at (0, 1, 15..20), i.e. [gx, gy, 0, 0, 0,
0, 0, 0].

All 37 columns already exist in the 58-column file, so this is a COLUMN
SELECTION -- no re-simulation. States, actions, lengths and eval goals are
bitwise identical to the gxy and the 58-column datasets, and a training run on
this file differs from the gxy run in the goal representation and nothing else.

WHY THE EXTRA SIX COLUMNS EXIST is measured, not assumed: see
crl.rockfall_clock_v5.GOAL_INDICES_XYV and
scripts/probe_v5_failure_representation.py. Short version: a V5 rock death and
a safe crossing occupy the SAME XY, so an XY-only goal cannot name a failure
state (episode-grouped LDA AUC 0.440 = chance); the torso velocity columns can
(0.879).

Note the goal half of the stored observation is never a training input --
crl/replay.py rebuilds every goal from a future STATE and discards it (see the
TrajectoryBuffer module doc). It only has to be the right WIDTH, and to be the
vector the evaluation env will command, which it is.

  python scripts/make_v5_gxyv_dataset.py             # both variants
  python scripts/make_v5_gxyv_dataset.py --names antmaze_rockfall_clock_v5_far05
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl.rockfall_clock_v5 import GOAL_INDICES_XYV     # noqa: E402

OUT = os.path.join('artifacts', 'rockfall_clock_v5', 'dataset')
NAMES = ('antmaze_rockfall_clock_v5', 'antmaze_rockfall_clock_v5_far05')
STATE_DIM = 29
#: state columns, then the goal half sliced by the same indices the env uses.
KEEP = list(range(STATE_DIM)) + [STATE_DIM + i for i in GOAL_INDICES_XYV]


def derive(name, out_dir=OUT):
  src = os.path.join(out_dir, f'{name}.npz')
  dst = os.path.join(out_dir, f'{name}_gxyv.npz')
  with np.load(src, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths, eval_goals = d['lengths'], d['eval_goals']
    meta = json.loads(str(d['meta'])) if 'meta' in d else {}
    assert obs.shape[-1] == 58, obs.shape
    assert np.abs(obs[:, :, STATE_DIM + 2:]).max() == 0.0, (
        'padding is not all zero; this is not the 58-column V5 contract')
    new_obs = np.ascontiguousarray(obs[:, :, KEEP])
    assert np.array_equal(new_obs[:, :, :STATE_DIM], obs[:, :, :STATE_DIM])
    assert np.array_equal(new_obs[:, :, STATE_DIM:STATE_DIM + 2],
                          obs[:, :, STATE_DIM:STATE_DIM + 2])
    #: the six appended goal columns are the zero pad, i.e. "at the goal, at
    #: rest" -- assert it so a change of the underlying goal contract is loud.
    assert np.abs(new_obs[:, :, STATE_DIM + 2:]).max() == 0.0
    meta = dict(meta)
    meta['goal_rep'] = 'xyv'
    meta['goal_dim'] = len(GOAL_INDICES_XYV)
    meta['goal_indices'] = list(GOAL_INDICES_XYV)
    meta['derived_from'] = os.path.basename(src)
    meta['note'] = ('column selection of the 58-column dataset: state(29) + '
                    'goal[(0,1,15,16,17,18,19,20)] = [gx, gy, 0, 0, 0, 0, 0, '
                    '0]. The commanded goal is the same XY, at rest; the 21 '
                    'dropped goal columns are the zero padding.')
    np.savez_compressed(dst, obs=new_obs, act=act, lengths=lengths,
                        eval_goals=eval_goals, meta=json.dumps(meta))
  print(f'{src} {obs.shape} -> {dst} {new_obs.shape}')
  sc = os.path.join(out_dir, f'{name}_sidecar.npz')
  if os.path.exists(sc):
    print(f'  sidecar unchanged and shared: {sc}')
  return dst


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--names', nargs='*', default=list(NAMES))
  args = ap.parse_args()
  for name in args.names:
    derive(name, args.out_dir)


if __name__ == '__main__':
  main()
