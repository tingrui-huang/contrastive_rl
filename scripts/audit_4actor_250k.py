"""Evidence-backed local artifact audit of the 4-actor 250k run (part 1).

Produces artifacts/audit_4actor_250k/{inventory.json, tb_scalars.csv,
replay_integrity.json} strictly from files in the run directory.
"""
import csv
import glob
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import crl.losses  # noqa: E402  (TrainingState for unpickling)

R = 'd4rl_ant_umaze_gfull_gfull29_alpha0_4actor_s0_250k'
OUT = 'artifacts/audit_4actor_250k'
os.makedirs(OUT, exist_ok=True)


def tree_hash(tree):
  h = hashlib.sha256()
  def walk(t, prefix=''):
    if isinstance(t, dict):
      for k in sorted(t):
        walk(t[k], prefix + '/' + k)
    else:
      h.update(prefix.encode())
      h.update(np.ascontiguousarray(np.asarray(t)).tobytes())
  walk(tree)
  return h.hexdigest()[:16]


# ---- 1. checkpoint inventory ----
inv = {}
for p in sorted(glob.glob(f'{R}/checkpoints/*.pkl') + glob.glob(f'{R}/gates/*.pkl')):
  with open(p, 'rb') as f:
    payload = pickle.load(f)
  st = payload['state']
  inv[p.replace('\\', '/')] = {
      'step': payload['step'],
      'size_mb': round(os.path.getsize(p) / 1e6, 2),
      'mtime': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))),
      'fields': {
          'policy_params': st.policy_params is not None,
          'q_params': st.q_params is not None,
          'target_q_params': st.target_q_params is not None,
          'policy_optimizer_state': st.policy_optimizer_state is not None,
          'q_optimizer_state': st.q_optimizer_state is not None,
          'jax_key': st.key is not None,
          'alpha_params': st.alpha_params is not None,
      },
      'policy_hash': tree_hash(st.policy_params),
      'q_hash': tree_hash(st.q_params),
  }
json.dump(inv, open(f'{OUT}/inventory.json', 'w'), indent=2)
print('--- checkpoint inventory ---')
for p, v in inv.items():
  print(f"{p.split('/')[-1]:22s} step {v['step']:7d} {v['size_mb']:5.2f}MB "
        f"{v['mtime']}  pol {v['policy_hash'][:8]} q {v['q_hash'][:8]} "
        f"alpha={v['fields']['alpha_params']}")

same = (inv[f'{R}/checkpoints/latest.pkl']['policy_hash'],
        inv[f'{R}/checkpoints/final.pkl']['policy_hash'],
        inv[f'{R}/gates/gate_250000.pkl']['policy_hash'])
print('latest == final == gate_250000 (policy):', len(set(same)) == 1)

# ---- 2. best semantics from stage logs ----
print('\n--- best.pkl semantics ---')
print('best.pkl step:', inv[f'{R}/checkpoints/best.pkl']['step'])
best_lines = []
for lg in sorted(glob.glob(f'{R}/logs/train_to_*.log')):
  for line in open(lg, encoding='utf-8', errors='replace'):
    if 'new best' in line:
      best_lines.append((os.path.basename(lg), line.strip()))
for lg, l in best_lines:
  print(f'  {lg}: {l}')
json.dump({'best_step': inv[f'{R}/checkpoints/best.pkl']['step'],
           'new_best_log_lines': [f'{a}: {b}' for a, b in best_lines]},
          open(f'{OUT}/best_semantics.json', 'w'), indent=2)

# ---- 3. replay integrity ----
print('\n--- replay integrity ---')
rep = {}
for name in ['checkpoints/replay.npz', 'gates/replay_250000.npz']:
  with np.load(f'{R}/{name}') as d:
    n = int(d['num_eps'])
    obs = d['obs']
    ok = bool(np.all(np.isfinite(obs[:, :, :29])))
    starts = obs[:, 0, :2]
    per_actor = [int(len([k for k in range(n) if k % 4 == a]))
                 for a in range(4)]
    rep[name] = {'episodes': n, 'transitions': n * 700,
                 'obs_shape': list(obs.shape), 'finite': ok,
                 'per_actor_episodes': per_actor,
                 'unique_start_xy_0.5m': int(len(np.unique(
                     np.round(starts * 2) / 2, axis=0)))}
with np.load(f'{R}/checkpoints/replay.npz') as a, \
     np.load(f'{R}/gates/replay_250000.npz') as b:
  rep['replay.npz == replay_250000.npz'] = bool(
      int(a['num_eps']) == int(b['num_eps'])
      and np.array_equal(a['obs'][:int(a['num_eps'])],
                         b['obs'][:int(b['num_eps'])]))
final_step = inv[f'{R}/checkpoints/latest.pkl']['step']
rep['episodes*700_vs_final_step'] = {
    'episodes*700': rep['checkpoints/replay.npz']['episodes'] * 700,
    'final_checkpoint_step': final_step}
json.dump(rep, open(f'{OUT}/replay_integrity.json', 'w'), indent=2)
print(json.dumps(rep, indent=1))

# ---- 4. TensorBoard extraction ----
print('\n--- tensorboard extraction ---')
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402
rows = {}
tags_seen = set()
for ev in sorted(glob.glob(f'{R}/checkpoints/tb/events.out.tfevents.*')):
  ea = event_accumulator.EventAccumulator(ev,
      size_guidance={event_accumulator.SCALARS: 0})
  ea.Reload()
  stage = os.path.basename(ev).split('.')[3]
  for tag in ea.Tags()['scalars']:
    tags_seen.add(tag)
    for s in ea.Scalars(tag):
      rows.setdefault((s.step, stage), {})[tag] = s.value
wanted = ['success', 'min_dist', 'final_dist', 'critic_loss', 'actor_loss',
          'categorical_accuracy', 'logits_pos', 'logits_neg', 'logits_gap',
          'learner_updates', 'per_actor_steps', 'num_actors']
cols = ['step', 'stage_file'] + [t for t in wanted if t in tags_seen]
with open(f'{OUT}/tb_scalars.csv', 'w', newline='') as f:
  w = csv.writer(f)
  w.writerow(cols)
  for (step, stage) in sorted(rows):
    w.writerow([step, stage] + [rows[(step, stage)].get(t, '')
                                for t in cols[2:]])
print('tags present:', sorted(tags_seen))
print(f'rows: {len(rows)} -> {OUT}/tb_scalars.csv')
for (step, stage) in sorted(rows):
  r = rows[(step, stage)]
  print(step, stage[-4:],
        {t: round(r[t], 3) for t in ('success', 'min_dist', 'learner_updates',
                                     'logits_gap', 'categorical_accuracy')
         if t in r})
