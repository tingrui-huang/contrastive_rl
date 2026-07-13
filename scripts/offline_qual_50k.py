"""Staged offline qualification: antmaze-umaze-v2, 50k learner updates.

Uses the prepared faithful offline config EXACTLY (build_offline_cfg from
scripts/verify_offline_d4rl.py) with gates at ~1k/5k/10k/25k/50k learner
updates. The offline step clock advances in blocks of max_episode_steps=700
gradient updates, so nominal gates map to the nearest block multiples:

    nominal   1k    5k    10k    25k    50k
    actual   1400  4900   9800  25200  50400

At every gate: snapshot latest.pkl -> gates/, then run
scripts/offline_gate_report.py (invariants / objective decomposition /
policy health / twin-critic / fixed-seed eval + STOP flags). Any STOP flag
or in-train assert aborts the whole qualification. After the ~10k gate,
runs the checkpoint/resume verification (replay, optimizer, RNG, counter).
Stops after 50400. Does NOT continue to 1M.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from verify_offline_d4rl import build_offline_cfg, NPZ  # noqa: E402

BLOCK = 700
# Canonical 50k qualification gates (nominal -> actual step-clock value hit).
GATES = {1000: 1400, 5000: 4900, 10000: 9800, 25000: 25200, 50000: 50400}
# Extend past 50k by resume: OFFLINE_EXTRA_GATES="75000,100000" appends gates
# (actual = ceil(nominal/BLOCK)*BLOCK). Done gates are skipped (their gate pkl
# exists), so the run continues from latest.pkl to the new targets.
import math as _math  # noqa: E402
_extra = os.environ.get('OFFLINE_EXTRA_GATES', '').strip()
if _extra:
  for _n in (int(x) for x in _extra.split(',') if x.strip()):
    GATES[_n] = int(_math.ceil(_n / BLOCK) * BLOCK)
  GATES = dict(sorted(GATES.items()))
# RUN_DIR is env-overridable: on Colab set OFFLINE_RUN_DIR to a fast local
# /content scratch (atomic checkpoint writes) and OFFLINE_DRIVE_DIR to the
# persistent Drive run dir (mirrored after each gate). Local runs use the
# default and leave OFFLINE_DRIVE_DIR empty (no mirroring).
RUN_DIR = os.environ.get(
    'OFFLINE_RUN_DIR',
    os.path.join(os.path.dirname(_HERE), 'offline_ant_umaze_v2_qual50k_s0'))
DRIVE_DIR = os.environ.get('OFFLINE_DRIVE_DIR', '')
CKPT_DIR = os.path.join(RUN_DIR, 'checkpoints')
_MIRROR_SUBS = ('checkpoints', 'gates', 'reports', 'logs')


def ckpt_step(p):
  with open(p, 'rb') as f:
    return pickle.load(f)['step']


def _atomic_copy(src, dst):
  """Copy src->dst. Tries tmp+rename; falls back to a direct copy because the
  Google Drive FUSE mount often rejects os.replace/rename with an I/O error."""
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  try:
    shutil.copy2(src, dst + '.tmp')
    os.replace(dst + '.tmp', dst)
  except OSError:
    # Drive FUSE hiccup on rename: copy straight through (best-effort).
    if os.path.exists(dst + '.tmp'):
      try:
        os.remove(dst + '.tmp')
      except OSError:
        pass
    shutil.copy2(src, dst)


def mirror_to_drive():
  """Copy the local run dir to Drive (best-effort). No-op if no Drive. A Drive
  I/O hiccup must NEVER abort a run that is already checkpointed locally, so the
  whole mirror is wrapped and only warns on failure."""
  if not DRIVE_DIR:
    return
  try:
    for sub in _MIRROR_SUBS:
      s = os.path.join(RUN_DIR, sub)
      if not os.path.isdir(s):
        continue
      for name in os.listdir(s):
        sp = os.path.join(s, name)
        if os.path.isfile(sp):
          _atomic_copy(sp, os.path.join(DRIVE_DIR, sub, name))
    for name in ('config.json', os.path.join('reports', 'run_plan.json')):
      sp = os.path.join(RUN_DIR, name)
      if os.path.isfile(sp):
        _atomic_copy(sp, os.path.join(DRIVE_DIR, name))
    print(f'  [drive] mirrored run dir -> {DRIVE_DIR}', flush=True)
  except Exception as ex:  # pylint: disable=broad-except
    print(f'  [drive] WARNING: mirror to {DRIVE_DIR} failed ({ex}); '
          'local checkpoints under OFFLINE_RUN_DIR are intact.', flush=True)


def restore_from_drive():
  """Before a resumed run, pull the Drive run dir back to local scratch."""
  if not DRIVE_DIR:
    return
  drive_latest = os.path.join(DRIVE_DIR, 'checkpoints', 'latest.pkl')
  local_latest = os.path.join(CKPT_DIR, 'latest.pkl')
  if os.path.exists(drive_latest) and not os.path.exists(local_latest):
    for sub in _MIRROR_SUBS:
      s = os.path.join(DRIVE_DIR, sub)
      if os.path.isdir(s):
        shutil.copytree(s, os.path.join(RUN_DIR, sub), dirs_exist_ok=True)
    for name in ('config.json',):
      s = os.path.join(DRIVE_DIR, name)
      if os.path.isfile(s):
        _atomic_copy(s, os.path.join(RUN_DIR, name))
    print(f'  [drive] restored run dir from {DRIVE_DIR}', flush=True)


_CFG_KEYS = ('env_name', 'bc_coef', 'twin_q', 'random_goals',
             'entropy_coefficient', 'target_entropy', 'batch_size', 'repr_dim',
             'hidden_layer_sizes', 'discount', 'actor_learning_rate',
             'learning_rate', 'num_sgd_steps_per_step', 'updates_per_step',
             'seed', 'num_actors', 'use_layer_norm')


def persist_and_check_config():
  """Write config.json once; on resume REJECT any config drift. The dataset
  path is excluded (differs across machines) -- dataset identity is enforced
  by SHA in train()'s require_same_dataset_hash + the sidecar."""
  cfg = build_offline_cfg(ckpt_dir=CKPT_DIR)
  cur = {k: getattr(cfg, k) for k in _CFG_KEYS}
  cur = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cur.items()}
  p = os.path.join(RUN_DIR, 'config.json')
  if os.path.exists(p):
    prev = json.load(open(p))
    # Only flag keys PRESENT in the recorded config; a newly-added key (absent
    # from an older run's config.json, e.g. use_layer_norm) must not break the
    # resume of a run started before the key existed.
    mism = {k: (prev.get(k), cur.get(k))
            for k in cur if k in prev and prev.get(k) != cur.get(k)}
    if mism:
      raise RuntimeError(f'offline config MISMATCH on resume: {mism}')
    print('  config unchanged on resume:', p, flush=True)
  else:
    json.dump(cur, open(p, 'w'), indent=2)
    print('  config persisted:', p, flush=True)


def stage_cfg(stage_end, resume):
  cfg = build_offline_cfg(max_steps=stage_end, ckpt_dir=CKPT_DIR)
  cfg.resume = resume
  cfg.eval_every_steps = 1400
  cfg.eval_episodes = 10
  cfg.log_every_steps = 700
  cfg.tensorboard = True
  # freeze contract
  assert cfg.offline_dataset == NPZ and cfg.bc_coef == 0.05
  assert cfg.twin_q and cfg.random_goals == 0.0 and cfg.batch_size == 1024
  assert cfg.repr_dim == 16 and cfg.hidden_layer_sizes == (1024, 1024)
  assert cfg.entropy_coefficient == 0.0 and cfg.target_entropy == 0.0
  assert cfg.guard_abort
  return cfg


class Tee:
  def __init__(self, path, stream):
    self.f, self.s = open(path, 'a', buffering=1), stream
  def write(self, x):
    self.f.write(x); self.s.write(x)
  def flush(self):
    self.f.flush(); self.s.flush()


def run_gate_report(tag, ckpt):
  r = subprocess.run(
      [sys.executable, os.path.join(_HERE, 'offline_gate_report.py'),
       '--ckpt', ckpt, '--tag', str(tag), '--run_dir', RUN_DIR],
      capture_output=True, text=True)
  sys.stdout.write(r.stdout[-2500:])
  if r.returncode != 0:
    sys.stdout.write(r.stderr[-2000:])
    raise RuntimeError(f'gate report {tag} crashed')
  rep = json.load(open(os.path.join(RUN_DIR, 'reports', f'gate_{tag}.json')))
  return rep


def resume_test():
  """After the ~10k gate: verify checkpoint/resume restores everything."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  from crl import checkpoint as ckpt_mod
  from crl import networks as networks_mod
  from crl import losses as losses_mod
  from crl import offline_audit
  import optax

  res = {}
  latest = os.path.join(CKPT_DIR, 'latest.pkl')
  s0, st_a = ckpt_mod.load_checkpoint(latest)
  _, st_b = ckpt_mod.load_checkpoint(latest)
  res['step'] = int(s0)
  res['fields_present'] = {
      'policy_optimizer_state': st_a.policy_optimizer_state is not None,
      'q_optimizer_state': st_a.q_optimizer_state is not None,
      'jax_key': st_a.key is not None,
  }
  # determinism: two independent loads + one update on the same batch must
  # produce IDENTICAL params (proves optimizer + RNG state fully restored).
  cfg = build_offline_cfg()
  from crl import envs as envs_mod
  envs_mod.make_env('offline_ant_umaze', cfg, seed=777)  # fill obs/goal/act dims
  buffer, fp = offline_audit.build_offline_buffer(NPZ, cfg)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  def o2g(s):
    return s[:, jnp.arange(29)]
  _, upd = losses_mod.build_learner(nets, cfg, o2g,
                                    optax.adam(3e-4, eps=1e-7),
                                    optax.adam(3e-4, eps=1e-7))
  tr = buffer.sample(256)
  trans = losses_mod.Transition(*[jnp.asarray(x) for x in tr])
  na, ma = upd(st_a, trans)
  nb, mb = upd(st_b, trans)
  same = all(bool(jnp.array_equal(x, y)) for x, y in zip(
      jax.tree_util.tree_leaves(na.policy_params),
      jax.tree_util.tree_leaves(nb.policy_params)))
  same &= all(bool(jnp.array_equal(x, y)) for x, y in zip(
      jax.tree_util.tree_leaves(na.q_params),
      jax.tree_util.tree_leaves(nb.q_params)))
  same &= bool(jnp.array_equal(na.key, nb.key))
  res['restore_deterministic'] = bool(same)
  res['post_update_actor_loss'] = float(ma['actor_loss'])
  assert same, 'restored states diverge after one identical update'

  # replay restore: rebuilt buffer must hash to the pinned sidecar dataset.
  side = json.load(open(os.path.join(CKPT_DIR, 'offline_dataset.sha256')))
  res['dataset_sidecar_sha'] = side['sha256'][:16]
  res['dataset_rehash_matches'] = side['sha256'] == fp['sha256']
  assert res['dataset_rehash_matches']
  res['buffer_content_sha'] = buffer.content_sha256()[:16]

  # counter continuity: run ONE more block via train(resume=True) and check
  # the step clock continues from s0 (not from 0).
  from crl.train import train
  cfg2 = stage_cfg(s0 + BLOCK, resume=True)
  out0, err0 = sys.stdout, sys.stderr
  sys.stdout = Tee(os.path.join(RUN_DIR, 'logs', 'resume_test.log'), out0)
  sys.stderr = Tee(os.path.join(RUN_DIR, 'logs', 'resume_test.log'), err0)
  try:
    train(cfg2)
  finally:
    sys.stdout, sys.stderr = out0, err0
  log = open(os.path.join(RUN_DIR, 'logs', 'resume_test.log'),
             encoding='utf-8', errors='replace').read()
  assert f'Resumed from' in log and f'at step {s0}' in log, 'resume print missing'
  s1 = ckpt_step(latest)
  assert s1 == s0 + BLOCK, (s1, s0)
  res['counter_continuity'] = {'before': int(s0), 'after_one_block': int(s1)}
  res['pass'] = True
  json.dump(res, open(os.path.join(RUN_DIR, 'reports', 'resume_test.json'),
                      'w'), indent=2)
  print('RESUME TEST PASS:', json.dumps(res, indent=1))
  return s1


def main():
  for d in ('checkpoints', 'gates', 'logs', 'reports'):
    os.makedirs(os.path.join(RUN_DIR, d), exist_ok=True)
  restore_from_drive()            # pull Drive -> local scratch if resuming
  persist_and_check_config()      # write once; reject drift on resume
  json.dump({'gates_nominal_to_actual': GATES, 'block': BLOCK,
             'dataset': NPZ, 'run_dir': RUN_DIR, 'drive_dir': DRIVE_DIR,
             'started': time.strftime('%F %T')},
            open(os.path.join(RUN_DIR, 'reports', 'run_plan.json'), 'w'),
            indent=2)
  from crl.train import train

  # Resume/checkpoint verification at ~10k. Decoupled from the gate loop so it
  # still runs on a relaunch where gate_10000.pkl already exists; idempotent
  # via its output file. It advances the clock by one block (proves counter
  # continuity), which the 25k stage then builds on.
  latest = os.path.join(CKPT_DIR, 'latest.pkl')
  rt_path = os.path.join(RUN_DIR, 'reports', 'resume_test.json')
  if (os.path.exists(latest) and ckpt_step(latest) >= GATES[10000]
      and not os.path.exists(rt_path)):
    print('=== resume/checkpoint verification at ~10k ===', flush=True)
    resume_test()
    mirror_to_drive()

  for nominal, actual in GATES.items():
    gate_pkl = os.path.join(RUN_DIR, 'gates', f'gate_{nominal}.pkl')
    if os.path.exists(gate_pkl):
      print(f'gate {nominal}: exists -- skip', flush=True)
      continue
    latest = os.path.join(CKPT_DIR, 'latest.pkl')
    ls = ckpt_step(latest) if os.path.exists(latest) else None
    target = actual
    if ls is None or ls < target:
      cfg = stage_cfg(target, resume=ls is not None)
      print(f'=== stage -> {target} (nominal {nominal}, resume={ls is not None},'
            f' from {ls}) ===', flush=True)
      out0, err0 = sys.stdout, sys.stderr
      lg = os.path.join(RUN_DIR, 'logs', f'train_to_{nominal}.log')
      sys.stdout = Tee(lg, out0)
      sys.stderr = Tee(lg, err0)
      try:
        train(cfg)
      finally:
        sys.stdout, sys.stderr = out0, err0
    if os.path.exists(os.path.join(CKPT_DIR, 'abort.pkl')):
      print(f'!! GUARD ABORT during stage {nominal} -- stopping.', flush=True)
      mirror_to_drive()
      break
    ls = ckpt_step(latest)
    assert ls >= actual, (ls, actual)
    shutil.copy2(latest, gate_pkl + '.tmp')
    assert ckpt_step(gate_pkl + '.tmp') == ls
    os.replace(gate_pkl + '.tmp', gate_pkl)
    print(f'gate {nominal}: snapshot at step {ls}', flush=True)

    rep = run_gate_report(nominal, gate_pkl)
    mirror_to_drive()               # persist checkpoints + reports after gate
    if rep['stop_flags']:
      print(f'!! STOP CONDITION at gate {nominal}: {rep["stop_flags"]} -- '
            'qualification ABORTED.', flush=True)
      break

  mirror_to_drive()
  print('QUALIFICATION RUN COMPLETE.', flush=True)


if __name__ == '__main__':
  main()
