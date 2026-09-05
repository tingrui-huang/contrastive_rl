"""Derive the XY-goal (upstream ant contract) copy of a V5 dataset.

The V5 datasets store the env observation, which under this port's historical
contract is [state(29), goal_xy(2), 27 zeros] = 58 columns. The upstream ant
setting (lp_contrastive.py sets end_index=2 for every 'ant_' env, offline_ant_*
included) commands the goal as the XY pair alone, so its observation is
[state(29), goal_xy(2)] = 31 columns.

Those 31 columns are already present in the 58-column file, so the XY-goal
dataset is a COLUMN SELECTION of the existing one -- no re-simulation. States,
actions, episode lengths and eval goals are therefore bitwise identical, and a
training run on this file differs from the 58-column run in the goal
representation and nothing else.

  python scripts/make_v5_gxy_dataset.py            # both variants
"""
import json
import os

import numpy as np

OUT = os.path.join('artifacts', 'rockfall_clock_v5', 'dataset')
NAMES = ('antmaze_rockfall_clock_v5', 'antmaze_rockfall_clock_v5_far05')
#: state columns plus the commanded goal XY; the 27 dropped columns are the
#: zero padding (asserted below).
KEEP = list(range(29)) + [29, 30]


def derive(name, out_dir=OUT):
  src = os.path.join(out_dir, f'{name}.npz')
  dst = os.path.join(out_dir, f'{name}_gxy.npz')
  with np.load(src, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths, eval_goals = d['lengths'], d['eval_goals']
    meta = json.loads(str(d['meta'])) if 'meta' in d else {}
    assert obs.shape[-1] == 58, obs.shape
    assert np.abs(obs[:, :, 31:]).max() == 0.0, 'padding is not all zero'
    new_obs = np.ascontiguousarray(obs[:, :, KEEP])
    assert np.array_equal(new_obs[:, :, :29], obs[:, :, :29])
    assert np.array_equal(new_obs[:, :, 29:31], obs[:, :, 29:31])
    meta = dict(meta)
    meta['goal_rep'] = 'xy'
    meta['derived_from'] = os.path.basename(src)
    meta['note'] = ('column selection of the 58-column dataset: state(29) + '
                    'goal xy(2); the 27 zero-padding columns are dropped')
    np.savez_compressed(dst, obs=new_obs, act=act, lengths=lengths,
                        eval_goals=eval_goals, meta=json.dumps(meta))
  print(f'{src} {obs.shape} -> {dst} {new_obs.shape}')
  sc = os.path.join(out_dir, f'{name}_sidecar.npz')
  if os.path.exists(sc):
    print(f'  sidecar unchanged and shared: {sc}')
  return dst


def main():
  for name in NAMES:
    derive(name)


if __name__ == '__main__':
  main()
