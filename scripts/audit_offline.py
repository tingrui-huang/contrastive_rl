"""Strict offline-only pipeline audit -- run BEFORE any offline maze experiment.

Executes every offline correctness gate and STOPS (non-zero exit) unless all
pass. Does NOT generate or train on any new maze dataset.

Sections:
  A. Static + buffer gates on the real dataset      (G1-G8)
  B. Audit-field separation, positive AND negative   (a leaked confounder MUST
     fail G6)                                         (G6)
  C. Valid-length relabel masking, positive AND       (unmasked sampling MUST
     negative                                          hit padding)            (G7)
  D. Structurally-impossible collection smoke: run    (G-collect)
     train() offline for a few hundred grad steps and prove NO collection env
     is created and collect_episode/collect_block are never called; then a
     resume with a DIFFERENT dataset hash must be rejected.                    (G9)

Run:
  python -m scripts.audit_offline --dataset datasets/push_state_conedir_smoke.npz
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


import numpy as np

from crl import envs as envs_mod
from crl import offline_audit
from crl.config import Config

OUT = os.path.join('artifacts', 'offline_audit')


def _cfg_from_env(env_name):
  cfg = Config(env_name=env_name)
  envs_mod.make_env(env_name, cfg, seed=0)     # fills dims into cfg
  return cfg


# --------------------------------------------------------------------------- #
def section_A(dataset, cfg):
  passed, gates, report = offline_audit.run_static_audit(dataset, cfg)
  print('A. STATIC + BUFFER GATES')
  for g, ok in gates.items():
    print(f'   {"PASS" if ok else "FAIL":4}  {g}')
  fp = report['fingerprint']
  print(f'   sha256={fp["sha256"][:16]}...  eps={fp["n_episodes"]}  '
        f'trans={fp["n_transitions"]}  obs={fp["obs_shape"]}  '
        f'act={fp["act_shape"]}')
  print(f'   keys: learner={fp["keys"]["learner"]}  audit={fp["keys"]["audit"]}'
        f'  other={fp["keys"]["other"]}')
  print(f'   relabel={report["stats"]["relabel"]}')
  print(f'   frozen={report["stats"]["frozen"]}')
  return passed, {'gates': gates, 'fingerprint': fp,
                  'stats': report['stats']}


# --------------------------------------------------------------------------- #
def _write_fixture(path, obs, act, meta, **extra):
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  np.savez_compressed(path, obs=obs.astype(np.float32), act=act.astype(np.float32),
                      meta=np.array(json.dumps(meta)), **extra)


def section_B():
  """Audit-field separation: clean fixture (audit fields present but isolated)
  PASSES; a fixture that concatenates a confounder column into obs FAILS G6."""
  print('B. AUDIT-FIELD SEPARATION (swamp bits / route labels)')
  rng = np.random.default_rng(0)
  N, L, od, gd, A = 8, 6, 4, 2, 2
  obs = rng.normal(size=(N, L, od + gd)).astype(np.float32)
  act = np.clip(rng.normal(size=(N, L, A)), -0.99, 0.99).astype(np.float32)
  act[:, -1] = 0.0
  meta = {'obs_dim': od, 'goal_dim': gd, 'action_dim': A,
          'start_index': 0, 'end_index': -1, 'max_episode_steps': L - 1}
  cfg = Config(obs_dim=od, goal_dim=gd, action_dim=A, start_index=0,
               end_index=-1, max_episode_steps=L - 1)

  clean = os.path.join(OUT, 'fixture_clean.npz')
  _write_fixture(clean, obs, act, meta,
                 audit_swamp_bits=rng.integers(0, 2, size=(N, L)),
                 audit_route_label=rng.integers(0, 2, size=(N,)),
                 swamp_bits=rng.integers(0, 2, size=(N, L)))  # known audit key
  g_clean, _ = offline_audit.static_gates(clean, od, gd, A, L)
  clean_ok = all(g_clean.values())
  print(f'   clean fixture (audit fields separate): '
        f'{"PASS" if clean_ok else "FAIL"}  '
        f'G2={g_clean["G2_KEY_SEPARATION"]} G6={g_clean["G6_NO_AUDIT_LEAK"]}')

  # LEAKY: append the confounder as an extra obs column -> width mismatch.
  leaky_obs = np.concatenate(
      [obs, rng.integers(0, 2, size=(N, L, 1)).astype(np.float32)], axis=2)
  leaky = os.path.join(OUT, 'fixture_leaky.npz')
  _write_fixture(leaky, leaky_obs, act, meta)
  g_leak, _ = offline_audit.static_gates(leaky, od, gd, A, L)
  leak_caught = (not g_leak['G6_NO_AUDIT_LEAK']) or (not g_leak['G3_SHAPES_DIMS'])
  print(f'   leaky fixture (confounder in obs) correctly REJECTED: '
        f'{"PASS" if leak_caught else "FAIL"}  '
        f'G3={g_leak["G3_SHAPES_DIMS"]} G6={g_leak["G6_NO_AUDIT_LEAK"]}')
  ok = clean_ok and leak_caught
  return ok, {'clean': g_clean, 'leaky': g_leak}


# --------------------------------------------------------------------------- #
def section_C():
  """Valid-length relabel masking: with per-episode lengths, relabel never
  samples the padded tail; WITHOUT the mask it provably would."""
  print('C. VALID-LENGTH RELABEL MASKING')
  from crl.replay import TrajectoryBuffer
  rng = np.random.default_rng(0)
  N, L, od, gd, A = 12, 10, 3, 2, 2
  full = od + gd
  obs = rng.normal(size=(N, L, full)).astype(np.float32)
  act = np.clip(rng.normal(size=(N, L, A)), -0.99, 0.99).astype(np.float32)
  lengths = rng.integers(3, L + 1, size=N)          # valid obs counts in [3, L]

  def _buf(pass_lengths):
    b = TrajectoryBuffer(capacity_steps=N * L, ep_len_obs=L, full_obs_dim=full,
                         action_dim=A, obs_dim=od, start_index=0, end_index=-1,
                         discount=0.99, seed=0)
    for e in range(N):
      b.add_episode(obs[e], act[e],
                    length=int(lengths[e]) if pass_lengths else None)
    return b

  masked = _buf(True)
  ok_masked, s_masked = offline_audit.check_relabel_boundaries(masked)
  print(f'   masked buffer: {"PASS" if ok_masked else "FAIL"}  {s_masked}')

  # Negative control: an UNMASKED buffer (no lengths) sampled and judged against
  # the TRUE short lengths must hit padding -> proves the mask does real work.
  unmasked = _buf(False)
  bad_j = 0
  for _ in range(64):
    traj, i, j = unmasked.sampled_indices(256)
    bad_j += int(np.sum(j >= lengths[traj]))
  print(f'   unmasked buffer WOULD sample padding: '
        f'{"PASS" if bad_j > 0 else "FAIL"}  goal_len_violations={bad_j}')
  ok = ok_masked and bad_j > 0
  return ok, {'masked': s_masked, 'unmasked_violations': int(bad_j)}


# --------------------------------------------------------------------------- #
def section_D(dataset, env_name):
  """Structurally-impossible collection: run train() offline for a few hundred
  gradient steps; assert only the EVAL env is created (never the collection
  seed) and collect_episode/collect_block are never called."""
  print('D. STRUCTURALLY-IMPOSSIBLE COLLECTION SMOKE')
  import crl.train as train_mod

  made_seeds = []
  real_make = envs_mod.make_env

  def counting_make(name, config, seed=0, render_mode=None):
    made_seeds.append(seed)
    return real_make(name, config, seed=seed, render_mode=render_mode)

  collect_calls = {'n': 0}

  def poisoned_collect(*a, **k):
    collect_calls['n'] += 1
    raise AssertionError('collect called in offline mode')

  run_dir = os.path.join(OUT, 'collect_smoke')
  cfg = Config(
      env_name=env_name, offline_dataset=dataset, bc_coef=0.5,
      random_goals=0.0, entropy_coefficient=None, target_entropy=-4.0,
      max_number_of_steps=300, eval_every_steps=150, eval_episodes=2,
      log_every_steps=150, batch_size=32, ckpt_dir=run_dir,
      random_steps=9999, min_replay_size=9999, num_actors=1)  # must be disabled

  # patch train's module refs
  train_mod.envs.make_env = counting_make
  real_ce, real_cb = train_mod.collect_episode, train_mod.collect_block
  train_mod.collect_episode = poisoned_collect
  train_mod.collect_block = poisoned_collect
  try:
    train_mod.train(cfg)
  finally:
    train_mod.envs.make_env = real_make
    train_mod.collect_episode = real_ce
    train_mod.collect_block = real_cb

  # In offline mode train() must create EXACTLY the eval env (seed = seed+10000)
  # and NEVER the collection-seed env (seed = cfg.seed).
  only_eval = made_seeds == [cfg.seed + 10_000]
  no_collect = collect_calls['n'] == 0
  print(f'   envs created (seeds)={made_seeds}  '
        f'(expected only [{cfg.seed + 10_000}]): '
        f'{"PASS" if only_eval else "FAIL"}')
  print(f'   collect_episode/collect_block calls={collect_calls["n"]}: '
        f'{"PASS" if no_collect else "FAIL"}')

  # G9: resume with a DIFFERENT dataset hash must be rejected.
  rejected = False
  fp = offline_audit.fingerprint(dataset)
  offline_audit.record_dataset_hash(run_dir, 'deadbeef' * 8, fp['meta'])
  same, recorded = offline_audit.require_same_dataset_hash(run_dir, fp['sha256'])
  rejected = (not same)
  print(f'   resume with mismatched dataset hash REJECTED: '
        f'{"PASS" if rejected else "FAIL"}')
  # restore the correct hash sidecar for cleanliness
  offline_audit.record_dataset_hash(run_dir, fp['sha256'], fp['meta'])

  ok = only_eval and no_collect and rejected
  return ok, {'made_seeds': made_seeds, 'collect_calls': collect_calls['n'],
              'resume_mismatch_rejected': rejected}


# --------------------------------------------------------------------------- #
def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--dataset', default='datasets/push_state_conedir_smoke.npz')
  p.add_argument('--env_name', default=None,
                 help='override env for dim inference (default: from meta)')
  args = p.parse_args()
  os.makedirs(OUT, exist_ok=True)

  fp0 = offline_audit.fingerprint(args.dataset)
  env_name = args.env_name or fp0['meta'].get('env_name')
  assert env_name, 'env_name not in dataset meta; pass --env_name'
  cfg = _cfg_from_env(env_name)

  results, reports = {}, {}
  results['A_static'], reports['A'] = section_A(args.dataset, cfg)
  results['B_audit_sep'], reports['B'] = section_B()
  results['C_length_mask'], reports['C'] = section_C()
  results['D_no_collect'], reports['D'] = section_D(args.dataset, env_name)

  verdict = 'PASS' if all(results.values()) else 'FAIL'
  summary = {'verdict': verdict, 'sections': results, 'dataset': args.dataset,
             'reports': reports}
  with open(os.path.join(OUT, 'audit_offline.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

  print('\n' + '=' * 60)
  for k, v in results.items():
    print(f'  {"PASS" if v else "FAIL"}  {k}')
  print(f'OFFLINE PIPELINE AUDIT: {verdict}  (report in {OUT})')
  print('=' * 60)
  sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
  main()
