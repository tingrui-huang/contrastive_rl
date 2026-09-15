r"""Build the pessimistic (absorbing-freeze) PointMaze F4 dataset for from-scratch CRL.

The absorbing-freeze ETT of ``ett/absorbing_ett.py`` is applied along the
recorded trajectories: at every alive step the recorded action ``a_t`` is the
executed action, a nominal draw ``x'_t`` stands in for the behaviour policy's
alternative, and the Manski onset rule fires when the landing bins of ``a_t``
and ``x'_t`` differ and the recorded landing cell lies in the absorbing
support discovered from the data.  From the first onset on the trajectory is
frozen exactly as the environment freezes an absorbed agent: XY fixed at the
landing, F4 history shifting, actions kept as recorded (ignored).  Nothing
else changes; episodes without an onset are byte-identical to the source.
The XY channel of the ETT is therefore sampled from the recorded outcomes
themselves, and only the freeze channel is modelled.

Training vanilla offline CRL on the result (arm P) versus on the source
dataset (arm O) isolates the effect of the pessimistic continuation on the
whole 150k-step pipeline; the two datasets share every key, shape and audit
field.  Reads ``obs``/``act`` and the frozen nominal only; hidden bits and
labels are copied through untouched as audit fields.

Usage::

    python -m scripts.build_pointmaze_absorbing_dataset \
        --source artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz \
        --nominal artifacts/nominal_policy/f4_p30_expert_only_mdn_k5_s0 \
        --out datasets/swamp_windy_f4_absorbing_s0.npz --seed 0
"""
import argparse
import hashlib
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ett import absorbing_ett as ab                     # noqa: E402
from propensity.nominal_policy import load_nominal_policy  # noqa: E402

GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)


def content_sha(path):
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode()); h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def build(source, nominal_dir, seed, copies):
  with np.load(source, allow_pickle=True) as d:
    arrays = {k: d[k] for k in d.files}
  meta = json.loads(str(arrays['meta']))
  obs, act = arrays['obs'], arrays['act']
  E, L, W = obs.shape
  T = L - 1
  assert W == 16 and act.shape == (E, L, 2)
  np.testing.assert_array_equal(obs[:, 1:, 2:8], obs[:, :-1, :6])
  np.testing.assert_array_equal(obs[:, :, 8:], np.broadcast_to(GOAL, obs[:, :, 8:].shape))

  tables = ab.freeze_tables(obs, act, np.arange(E))
  support = tables['support']
  print('absorbing support (all episodes, obs/act only):', tables['support_cells'])

  nominal = load_nominal_policy(nominal_dir)
  xy = obs[:, :, :2].astype(np.float64)
  cur, nxt = xy[:, :-1], xy[:, 1:]
  stationary = np.linalg.norm(nxt - cur, axis=-1) <= ab.STATIONARY_TOLERANCE
  # recorded absorption: a stationary run that continues to the episode end
  run_to_end = np.zeros_like(stationary)
  tail = np.ones(E, bool)
  for t in range(T - 1, -1, -1):
    tail = tail & stationary[:, t]
    run_to_end[:, t] = tail
  recorded_absorbed_from = np.where(run_to_end.any(1), np.argmax(run_to_end, 1), T)
  landing = ab.landing_cell(nxt)                          # [E, T, 2]
  in_support = support[landing[..., 0], landing[..., 1]]
  bin_a = ab.landing_cell(cur + act[:, :-1])

  key = jax.random.PRNGKey(seed)
  new_obs, new_act, onset_time, copy_id = [], [], [], []
  for c in range(copies):
    key, sub = jax.random.split(key)
    flat_s = obs[:, :-1, :8].reshape(-1, 8)
    xp = np.asarray(nominal.sample(jnp.asarray(flat_s), sub, 1,
                                   goal=jnp.broadcast_to(jnp.asarray(GOAL), flat_s.shape)))
    bin_xp = ab.landing_cell(cur.reshape(-1, 2) + xp).reshape(E, T, 2)
    disagree = np.any(bin_a != bin_xp, axis=-1)
    alive = np.arange(T)[None, :] < recorded_absorbed_from[:, None]
    onset = disagree & in_support & alive & (~stationary)
    t_on = np.where(onset.any(1), np.argmax(onset, 1), -1)
    o = obs.copy()
    for e in np.flatnonzero(t_on >= 0):
      t = int(t_on[e])
      frozen_xy = o[e, t + 1, :2].copy()
      for tt in range(t + 2, L):
        o[e, tt, :2] = frozen_xy
        o[e, tt, 2:8] = o[e, tt - 1, :6]
    new_obs.append(o); new_act.append(act.copy()); onset_time.append(t_on)
    copy_id.append(np.full(E, c, np.int64))
  new_obs = np.concatenate(new_obs); new_act = np.concatenate(new_act)
  onset_time = np.concatenate(onset_time); copy_id = np.concatenate(copy_id)
  np.testing.assert_array_equal(new_obs[:, 1:, 2:8], new_obs[:, :-1, :6])

  stats = {
      'episodes_source': int(E), 'copies': int(copies), 'episodes_out': int(len(new_obs)),
      'recorded_absorbed_episodes': int((recorded_absorbed_from < T).sum()),
      'added_onset_episodes': int((onset_time >= 0).sum()),
      'added_onset_fraction': float((onset_time >= 0).mean()),
      'mean_onset_time': float(onset_time[onset_time >= 0].mean()) if (onset_time >= 0).any() else None,
      'onset_landing_cells': {},
      'support_cells': tables['support_cells'],
      'support_rule': tables['rule'],
  }
  lc = landing[np.arange(E)[None, :].repeat(copies, 0).reshape(-1)[onset_time >= 0],
               onset_time[onset_time >= 0]]
  for cell_, n in zip(*np.unique(lc, axis=0, return_counts=True)):
    stats['onset_landing_cells'][f'({cell_[0]},{cell_[1]})'] = int(n)
  # audit-only breakdown by behaviour mode (labels never enter the learner)
  if 'teacher_mode' in arrays:
    mode = np.tile(arrays['teacher_mode'], copies)
    names = {0: 'random', 1: 'forced_safe', 2: 'immediate_shortcut', 3: 'wait_shortcut'}
    stats['added_onset_fraction_by_behaviour_audit_only'] = {
        names.get(int(m), str(m)): float((onset_time[mode == m] >= 0).mean()) for m in np.unique(mode)}
  out = dict(arrays)
  out['obs'] = new_obs.astype(np.float32); out['act'] = new_act.astype(np.float32)
  for k in ('swamp_bits', 'route_label', 'teacher_mode', 'force_safe', 'wait_count', 'entered_active_swamp'):
    if k in arrays:
      out[k] = np.concatenate([arrays[k]] * copies)
  out['audit_absorbing_onset_time'] = onset_time.astype(np.int64)
  out['audit_source_copy'] = copy_id
  meta = dict(meta)
  meta['setting'] = meta.get('setting', '') + ' + absorbing_freeze_manski_onset'
  meta['absorbing_transform'] = {
      'source_content_sha256': content_sha(source), 'nominal': os.path.abspath(nominal_dir),
      'seed': int(seed), 'copies': int(copies), 'stats': stats,
      'rule': ('at each alive step, x_prime ~ nominal(s); onset if landing_bin(a_rec) != landing_bin(x_prime) '
               'and the recorded landing cell is in the absorbing support; frozen from the landing on'),
      'episodes': int(len(new_obs)),
  }
  meta['episodes'] = int(len(new_obs))
  out['meta'] = np.array(json.dumps(meta))
  return out, stats


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--source', required=True)
  ap.add_argument('--nominal', required=True, help='nominal policy run directory (best.pkl + config.json)')
  ap.add_argument('--out', required=True)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--copies', type=int, default=1)
  args = ap.parse_args()
  out, stats = build(args.source, args.nominal, args.seed, args.copies)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  np.savez_compressed(args.out, **out)
  with open(args.out + '.manifest.json', 'w') as f:
    json.dump({'content_sha256': content_sha(args.out), 'file_sha256': hashlib.sha256(open(args.out, 'rb').read()).hexdigest(),
               'source': args.source, 'source_content_sha256': content_sha(args.source),
               'nominal': args.nominal, 'seed': args.seed, 'copies': args.copies, 'stats': stats}, f, indent=2)
  print(json.dumps(stats, indent=1))
  print('wrote', args.out, 'content sha', content_sha(args.out))


if __name__ == '__main__':
  main()
