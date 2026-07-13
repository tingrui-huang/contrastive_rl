"""Convert antmaze-umaze-v2.hdf5 to the crl offline .npz episode format.

Output (written NEXT TO the hdf5, outside the git repo):
  obs        [E, 701, 58] float32 -- state(29) + zero-padded goal(29): the
             goal half is zeros except [:2] = infos/goal of the episode
             (upstream OfflineAntWrapper contract).
  act        [E, 701, 8]  float32 -- act[i] is the action taken AT obs[i]
             (dataset row alignment obs[t], act[t] -> obs[t+1]); the final
             row's action is zeroed (it belongs to the next episode and is
             never sampled: sample() uses i <= L-2 and next_action only for
             TD, which the offline NCE run does not use).
  eval_goals [E, 2]       float32 -- per-episode behavior goal (empirical
             eval-goal distribution for the offline env).

Episode reconstruction contract (verified by scripts/audit_d4rl_dataset.py):
episodes are delimited by `timeouts` ONLY (terminals mark goal hits and the
episode continues); rows [s..e] inclusive with e = timeout index. Episodes
not exactly 701 rows long are dropped and reported (there is exactly one,
374 rows, 0.04% of transitions).
"""
import json
import os

import h5py
import numpy as np

DATA = r'D:\Users\trhua\Research\datasets\d4rl\antmaze-umaze-v2.hdf5'
OUT = r'D:\Users\trhua\Research\datasets\d4rl\antmaze_umaze_v2_offline.npz'
L = 701

with h5py.File(DATA, 'r') as f:
  obs = f['observations'][:]
  act = f['actions'][:]
  tout = f['timeouts'][:]
  goal = f['infos/goal'][:]

N = obs.shape[0]
ends = np.where(tout)[0]
starts = np.concatenate([[0], ends + 1])
starts = starts[starts < N]
kept, dropped = [], []
for s in starts:
  nxt = ends[ends >= s]
  e = int(nxt[0]) if len(nxt) else N - 1
  if e - s + 1 == L:
    kept.append((int(s), e))
  else:
    dropped.append((int(s), e, e - s + 1))

E = len(kept)
obs_out = np.zeros((E, L, 58), np.float32)
act_out = np.zeros((E, L, 8), np.float32)
goals = np.zeros((E, 2), np.float32)
for k, (s, e) in enumerate(kept):
  obs_out[k, :, :29] = obs[s:e + 1]
  g = goal[s]                                  # constant within an episode
  assert np.allclose(goal[s:e + 1], g, atol=1e-5), f'goal varies in ep {k}'
  obs_out[k, :, 29:31] = g                     # zero-padded goal contract
  act_out[k, :L - 1] = act[s:e]                # act[i] taken at obs[i]
  goals[k] = g

tmp = OUT + '.tmp'
with open(tmp, 'wb') as fo:
  np.savez_compressed(fo, obs=obs_out, act=act_out, eval_goals=goals)
os.replace(tmp, OUT)

meta = {
    'source_hdf5': DATA, 'out': OUT, 'episodes_kept': E,
    'episodes_dropped': [{'start': int(s), 'end': int(e), 'len': int(l)}
                         for s, e, l in dropped],
    'transitions_kept': E * (L - 1),
    'ep_len_obs': L,
    'obs_width': 58, 'goal_contract': 'zeros(29) with [:2]=infos/goal',
    'action_alignment': 'act[i] at obs[i] -> obs[i+1]; final row zeroed',
    'out_size_mb': round(os.path.getsize(OUT) / 1e6, 1),
}
json.dump(meta, open('artifacts/offline_d4rl/npz_conversion.json', 'w'),
          indent=2)
print(json.dumps(meta, indent=1))
