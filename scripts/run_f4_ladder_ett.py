"""Baseline vs fixed-ETT query-coverage CRL on one rung of the F4 detour ladder.

One rung, N learner seeds, two arms per seed, all at BC 0.05:

  O   plain offline CRL on the rung's merged dataset (6600 episodes)
  C   the sealed query-coverage method: a replay that is 50% learned-ETT
      rollouts (fixed six-action first query at the 550 supervised fork
      contexts, frozen nominal advice, frozen observational continuation
      actor) and 50% the predeclared original subset, then the same CRL.

  B   optional: the behaviour-query control of the sealed experiment (first
      query from the observational actor instead of the covered set).

Everything that the sealed experiment (outputs/pointmaze_ett_query_coverage_
20260915_v1) derived from the 0.05 dataset is re-derived from the rung:

  nominal policy   propensity.train_nominal_policy on the rung's 4800 expert
                   episodes, the sealed recipe (MDN k=5, 20k steps, seed 0)
  rollout actor    zbase on the rung (scripts/run_swamp_windy_z_failneg.py,
                   150k updates, bc 0.5, seed 0), the sealed actor's recipe
  original half    the sealed 3300 predeclared episode ids, read from the rung

and everything that does not depend on the dataset is reused byte-for-byte:
the learned ETT checkpoint, the 198 supervised fork roots / 550 contexts, the
six-action query set, the CRL recipe (30k updates, batch 256, bc 0.05), the
16 held-out roots and the 200 native reset seeds.  Within a seed, O and C
start from the identical learner state; across seeds and rungs, native
evaluation reuses the same reset and action seeds.

Usage::

  python scripts/run_f4_ladder_ett.py all --rung 0.20 --seeds 0 1 2
  python scripts/run_f4_ladder_ett.py all --rung 0.20 --seeds 0 1 2 --arms O B C
  python scripts/run_f4_ladder_ett.py aggregate        # table over every rung
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
PILOT = ROOT / 'outputs' / 'pointmaze_learned_ett_crl_20260915_v1'
sys.path.insert(0, str(PILOT))                      # learned_ett.py
OUT_ROOT = ROOT / 'outputs' / 'pointmaze_ladder_ett_v1'
CONFIG = json.loads((SEALED / 'config.json').read_text(encoding='utf-8'))
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
ENV = 'point_two_route_swamp_windy_f4_v0'
ARMS_ALL = ('O', 'B', 'C')
CRL_STEPS = int(CONFIG['crl']['steps_per_arm'])      # 30000
BC_COEF = 0.05
NATIVE = dict(CONFIG['heldout_evaluation'])
#: --smoke shrinks every budget so the whole chain runs on a CPU in minutes;
#: it writes to a separate root and proves nothing scientifically.
SMOKE = False
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from scripts import build_f4_detour_ladder as ladder   # noqa: E402


# ----------------------------------------------------------------- helpers
def sha256(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def plain(value):
  if isinstance(value, dict):
    return {str(k): plain(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [plain(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def write_json(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False)
                        + '\n', encoding='utf-8')


def load_npz(path):
  with np.load(path, allow_pickle=False) as loaded:
    return {key: loaded[key] for key in loaded.files}


def tree_sha(tree):
  import jax
  digest = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    array = np.ascontiguousarray(np.asarray(leaf))
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def run(cmd, log):
  Path(log).parent.mkdir(parents=True, exist_ok=True)
  print('  $', ' '.join(str(c) for c in cmd), '->', log, flush=True)
  with open(log, 'w', encoding='utf-8') as handle:
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True,
                   stdout=handle, stderr=subprocess.STDOUT)


class Rung:
  """Paths and fixed facts of one rung's experiment directory."""

  def __init__(self, p, out_root=OUT_ROOT):
    self.p = ladder.parse_rungs([p])[0]
    self.tag = ladder.rung_tag(self.p)
    self.out = Path(out_root) / self.tag
    self.dataset = ROOT / ladder.merged_path(self.p)
    self.rollout_actor = self.out / 'rollout_actor' / 'final.pkl'
    self.nominal = self.out / 'nominal' / 'best.pkl'
    self.logs = self.out / 'logs'

  def replay(self, arm):
    return self.out / f'replay_{arm}.npz'

  def crl_dir(self, seed, arm):
    return self.out / 'seeds' / f'seed_{seed}' / 'crl' / arm

  def seed_dir(self, seed):
    return self.out / 'seeds' / f'seed_{seed}'


# ------------------------------------------------------------------ stages
def stage_dataset(r: Rung):
  if not r.dataset.exists():
    print(f'rung {r.p:g} dataset missing; building', flush=True)
    ladder.build(r.p)
  manifest = ladder.load_manifest()
  entry = manifest['rungs'][r.tag]
  got = ladder.content_sha(str(r.dataset))
  if got != entry['content_sha256']:
    raise SystemExit(f'rung {r.p:g} content sha {got[:12]} differs from the '
                     f'ladder manifest {entry["content_sha256"][:12]}')
  write_json(r.out / 'dataset.json', {
      'force_safe_prob': r.p, 'dataset': str(r.dataset.relative_to(ROOT)),
      'content_sha256': got, 'file_sha256': sha256(r.dataset),
      'n_forced_safe': entry['n_forced_safe'], 'n_teacher': entry['n_teacher'],
      'episodes': entry['episodes']})
  print(f'rung {r.p:g}: {entry["n_forced_safe"]} of {entry["n_teacher"]} '
        f'teacher episodes on the safe route; sha OK', flush=True)


def stage_rollout_actor(r: Rung):
  if r.rollout_actor.exists():
    print('rollout actor exists', flush=True)
    return
  run([sys.executable, 'scripts/run_swamp_windy_z_failneg.py',
       '--version', 'f4', '--arm', 'zbase', '--force-safe-prob', f'{r.p:g}',
       '--seed', '0', '--ckpt-dir', r.rollout_actor.parent,
       '--smoke' if SMOKE else '--run'],
      r.logs / 'rollout_actor.log')
  if not r.rollout_actor.exists():
    raise RuntimeError('rollout actor training left no final.pkl')


def stage_nominal(r: Rung):
  if r.nominal.exists():
    print('nominal policy exists', flush=True)
    return
  run([sys.executable, '-m', 'propensity.train_nominal_policy',
       '--dataset', r.dataset, '--out-dir', r.nominal.parent,
       '--num-components', '5', '--steps', '500' if SMOKE else '20000',
       '--seed', '0'],
      r.logs / 'nominal.log')
  if not r.nominal.exists():
    raise RuntimeError('nominal training left no best.pkl')


def generate_arm(r: Rung, arm, roots, model, nominal, actor_state, network):
  """Verbatim generation rule of the sealed experiment, rung inputs."""
  import jax
  import jax.numpy as jnp
  import torch
  from learned_ett import legal_numpy

  @jax.jit
  def actor_sample(params, observation, key):
    distribution = network.policy_network.apply(params, observation)
    return network.sample(distribution, key)

  seed = int(CONFIG['seed'])
  context_root = roots['context_root'].astype(np.int32)
  context_state = roots['state'][context_root]
  candidates = roots['candidate_action'].astype(np.float32)
  context_id = np.repeat(np.arange(len(context_root), dtype=np.int32),
                         len(candidates))
  candidate_id = np.tile(np.arange(len(candidates), dtype=np.int8),
                         len(context_root))
  root_id = context_root[context_id]
  paths = len(context_id)
  if paths != CONFIG['generation']['paths_per_arm']:
    raise AssertionError('path budget mismatch')

  state = np.zeros((paths, 51, 8), np.float32)
  action = np.zeros((paths, 51, 2), np.float32)
  advice = np.zeros((paths, 49, 2), np.float32)
  onset_probability = np.zeros((paths, 49), np.float32)
  onset_event = np.zeros((paths, 49), bool)
  failed = np.zeros((paths, 51), bool)
  state[:, 0] = roots['state'][root_id]

  context_goal = np.broadcast_to(GOAL, context_state.shape).astype(np.float32)
  context_xb = np.asarray(nominal.sample(
      jnp.asarray(context_state), jax.random.PRNGKey(seed + 1000), 1,
      goal=jnp.asarray(context_goal)), np.float32).copy()
  first_xb = context_xb[context_id]
  torch_rng = torch.Generator(device='cpu')
  torch_rng.manual_seed(seed + 2000)
  started = time.time()
  for step in range(49):
    current = state[:, step]
    goal = np.broadcast_to(GOAL, current.shape).astype(np.float32)
    if step == 0:
      xb = first_xb.copy()
    else:
      xb = np.asarray(nominal.sample(
          jnp.asarray(current), jax.random.PRNGKey(seed + 3000 + step), 1,
          goal=jnp.asarray(goal)), np.float32).copy()
    observation = jnp.asarray(np.concatenate([current, goal], axis=1))
    actor_action = np.asarray(actor_sample(
        actor_state.policy_params, observation,
        jax.random.PRNGKey(seed + 4000 + step)), np.float32).copy()
    xq = (candidates[candidate_id].copy() if (arm == 'C' and step == 0)
          else actor_action)
    advice[:, step] = xb
    action[:, step] = xq
    with torch.no_grad():
      successor, event, probability, _ = model.sample_live(
          torch.as_tensor(current), torch.as_tensor(xb), torch.as_tensor(xq),
          torch_rng)
    successor = successor.numpy().astype(np.float32)
    event = event.numpy()
    probability = probability.numpy()
    was_failed = failed[:, step]
    persistent = np.concatenate([current[:, :2], current[:, :6]], axis=1)
    successor[was_failed] = persistent[was_failed]
    event[was_failed] = False
    probability[was_failed] = 0
    state[:, step + 1] = successor
    failed[:, step + 1] = was_failed | event
    onset_event[:, step] = event
    onset_probability[:, step] = probability
  state[:, 50] = state[:, 49]
  failed[:, 50] = failed[:, 49]
  lengths = np.full(paths, 50, np.int16)
  if np.any(state[:, 1:50, 2:] != state[:, :49, :6]):
    raise AssertionError('F4 shift violation')
  if not legal_numpy(state[:, :50, :2]).all():
    raise AssertionError('illegal emitted endpoint')
  recovery = failed[:, :49] & np.any(
      state[:, 1:50, :2] != state[:, :49, :2], axis=2)
  if recovery.any():
    raise AssertionError('post-failure recovery')
  xy = state[:, :50, :2]
  lower = np.any(xy[:, :, 1] < 2.0, axis=1)
  reach = np.any(np.linalg.norm(xy - GOAL[:2], axis=2) < 0.5, axis=1)
  data = {'states': state, 'actions': action, 'advice': advice,
          'onset_probability': onset_probability, 'onset_event': onset_event,
          'failed': failed, 'lengths': lengths, 'root_id': root_id,
          'context_id': context_id, 'candidate_id': candidate_id,
          'first_nominal': first_xb}
  summary = {'arm': arm, 'paths': int(paths),
             'lower_route_fraction': float(lower.mean()),
             'reach_0p5_fraction': float(reach.mean()),
             'absorbed_fraction': float(failed[:, 49].mean()),
             'onsets': int(onset_event.sum()),
             'first_query_down_sector': int(np.sum(
                 (np.abs(action[:, 0, 0]) <= 0.35) & (action[:, 0, 1] <= -0.75))),
             'first_query_right_sector': int(np.sum(
                 (action[:, 0, 0] >= 0.75) & (np.abs(action[:, 0, 1]) <= 0.35))),
             'wall_seconds': time.time() - started}
  return data, summary


def build_replay(r: Rung, arm, generated):
  """50% predeclared original episodes (read from the rung) + 50% generated."""
  prior = load_npz(PILOT / 'learned_ett_replay.npz')
  original = prior['audit_source'] == 0
  episode_ids = prior['audit_source_episode'][original].astype(np.int32)
  if len(episode_ids) != CONFIG['replay']['original_episodes']:
    raise AssertionError('prior original subset count changed')
  dataset = load_npz(r.dataset)
  original_obs = dataset['obs'][episode_ids].astype(np.float32)
  original_act = dataset['act'][episode_ids].astype(np.float32)
  goal = np.broadcast_to(GOAL, generated['states'].shape).astype(np.float32)
  synthetic_obs = np.concatenate([generated['states'], goal], axis=2)
  obs = np.concatenate([original_obs, synthetic_obs]).astype(np.float32)
  act = np.concatenate([original_act, generated['actions']]).astype(np.float32)
  lengths = np.concatenate([np.full(len(original_obs), 51, np.int16),
                            generated['lengths']])
  n_original = len(original_obs)
  n_synth = len(generated['states'])
  onset_time = np.where(generated['onset_event'].any(axis=1),
                        generated['onset_event'].argmax(axis=1), -1
                        ).astype(np.int16)
  meta = {'env_name': ENV, 'arm': arm, 'state_dim': 8, 'goal_dim': 8,
          'action_dim': 2, 'force_safe_prob': r.p,
          'original_dataset_content_sha256': ladder.content_sha(str(r.dataset)),
          'original_episode_ids': 'the sealed d5e2da0 P-replay subset, read '
                                  'from this rung',
          'query_intervention_only_at_synthetic_time_zero': arm == 'C',
          'synthetic_valid_observations': 50}
  path = r.replay(arm)
  np.savez_compressed(
      path, obs=obs, act=act, lengths=lengths,
      meta=np.asarray(json.dumps(meta, sort_keys=True)),
      audit_source=np.concatenate([np.zeros(n_original, np.int8),
                                   np.full(n_synth, 1, np.int8)]),
      audit_original_episode=np.concatenate([episode_ids,
                                             np.full(n_synth, -1, np.int32)]),
      audit_root_id=np.concatenate([np.full(n_original, -1, np.int32),
                                    generated['root_id'].astype(np.int32)]),
      audit_context_id=np.concatenate([np.full(n_original, -1, np.int32),
                                       generated['context_id'].astype(np.int32)]),
      audit_candidate_id=np.concatenate([np.full(n_original, -1, np.int8),
                                         generated['candidate_id'].astype(np.int8)]),
      audit_onset_time=np.concatenate([np.full(n_original, -2, np.int16),
                                       onset_time]),
      audit_failed_at_end=np.concatenate([np.zeros(n_original, bool),
                                          generated['failed'][:, 49]]))
  return path


def stage_generate(r: Rung, arms):
  wanted = [a for a in arms if a in ('B', 'C')]
  todo = [a for a in wanted if not r.replay(a).exists()]
  if not todo:
    print('replays exist:', wanted, flush=True)
    return
  import torch
  import jax
  from crl import checkpoint
  from learned_ett import load_checkpoint
  from propensity.nominal_policy import load_nominal_policy
  torch.set_num_threads(2)
  ett_path = ROOT / CONFIG['inputs']['ett_checkpoint']
  if sha256(ett_path) != CONFIG['inputs']['ett_checkpoint_sha256']:
    raise SystemExit(f'ETT checkpoint sha differs from the sealed one: {ett_path}')
  model, _ = load_checkpoint(ett_path, 'cpu')
  nominal = load_nominal_policy(r.nominal.parent)
  _, actor_state = checkpoint.load_checkpoint(r.rollout_actor)
  network = make_network()
  roots = load_npz(SEALED / 'construction_roots.npz')
  record = {'force_safe_prob': r.p,
            'fixed_inputs': {'ett_checkpoint': str(ett_path.relative_to(ROOT)),
                             'ett_checkpoint_sha256': sha256(ett_path),
                             'construction_roots_sha256':
                                 sha256(SEALED / 'construction_roots.npz')},
            'rung_inputs': {'nominal_sha256': sha256(r.nominal),
                            'rollout_actor_sha256': sha256(r.rollout_actor),
                            'dataset_sha256': sha256(r.dataset)},
            'jax_devices': [str(d) for d in jax.devices()], 'arms': {}}
  if (r.out / 'generation.json').exists():
    record['arms'] = json.loads((r.out / 'generation.json').read_text())['arms']
  for arm in todo:
    generated, summary = generate_arm(r, arm, roots, model, nominal,
                                      actor_state, network)
    np.savez_compressed(r.out / f'generated_{arm}.npz', **generated)
    replay = build_replay(r, arm, generated)
    summary['replay_sha256'] = sha256(replay)
    record['arms'][arm] = summary
    print(f'arm {arm}: lower {summary["lower_route_fraction"]:.3f} '
          f'reach {summary["reach_0p5_fraction"]:.3f} '
          f'absorbed {summary["absorbed_fraction"]:.3f} '
          f'| first query down/right {summary["first_query_down_sector"]}/'
          f'{summary["first_query_right_sector"]}', flush=True)
  write_json(r.out / 'generation.json', record)


def crl_config(r: Rung, seed, arm):
  from crl.config import Config
  dataset = r.dataset if arm == 'O' else r.replay(arm)
  return Config(
      env_name=ENV, offline_dataset=str(dataset),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=CRL_STEPS,
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.5, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=256, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=3e-4,
      actor_learning_rate=3e-4, num_sgd_steps_per_step=10, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed),
      eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=1_000,
      ckpt_every_steps=CRL_STEPS, ckpt_dir=str(r.crl_dir(seed, arm)))


def stage_train(r: Rung, seeds, arms):
  from crl.train import train
  for seed in seeds:
    for arm in arms:
      out = r.crl_dir(seed, arm)
      if (out / 'final.pkl').exists():
        print(f'seed {seed} arm {arm} exists', flush=True)
        continue
      cfg = crl_config(r, seed, arm)
      others = [dataclasses.asdict(crl_config(r, seed, a)) for a in arms]
      diff = {k for d in others for k in d
              if d[k] != dataclasses.asdict(cfg)[k]}
      if not diff <= {'offline_dataset', 'ckpt_dir'}:
        raise SystemExit(f'arms differ in more than the dataset: {sorted(diff)}')
      print(f'TRAIN rung {r.p:g} seed {seed} arm {arm}: bc {BC_COEF}, '
            f'{CRL_STEPS} updates, dataset {Path(cfg.offline_dataset).name}',
            flush=True)
      started = time.time()
      train(cfg)
      if not (out / 'final.pkl').exists():
        raise RuntimeError(f'no final.pkl for seed {seed} arm {arm}')
      write_json(out / 'run_summary.json', {
          'force_safe_prob': r.p, 'learner_seed': seed, 'arm': arm,
          'bc_coef': BC_COEF, 'steps': CRL_STEPS,
          'dataset': str(Path(cfg.offline_dataset).relative_to(ROOT)),
          'dataset_sha256': sha256(cfg.offline_dataset),
          'initial_checkpoint_sha256': sha256(out / 'init.pkl'),
          'final_checkpoint_sha256': sha256(out / 'final.pkl'),
          'wall_seconds': time.time() - started})


def verify_seed(r: Rung, seed, arms):
  from crl import checkpoint
  record = {'arms': {}}
  for arm in arms:
    d = r.crl_dir(seed, arm)
    _, initial = checkpoint.load_checkpoint(d / 'init.pkl')
    step, final = checkpoint.load_checkpoint(d / 'final.pkl')
    record['arms'][arm] = {
        'step': int(step), 'initial_tree_sha256': tree_sha(initial),
        'final_policy_sha256': tree_sha(final.policy_params),
        'final_critic_sha256': tree_sha(final.q_params),
        'actor_updated': tree_sha(final.policy_params)
                         != tree_sha(initial.policy_params),
        'critic_updated': tree_sha(final.q_params)
                          != tree_sha(initial.q_params)}
  inits = {record['arms'][a]['initial_tree_sha256'] for a in arms}
  record['matched'] = {
      'identical_initial_state_across_arms': len(inits) == 1,
      'budget': all(record['arms'][a]['step'] == CRL_STEPS for a in arms),
      'all_updated': all(record['arms'][a][k] for a in arms
                         for k in ('actor_updated', 'critic_updated'))}
  record['passed'] = all(record['matched'].values())
  write_json(r.seed_dir(seed) / 'crl_verification.json', record)
  if not record['passed']:
    raise RuntimeError(f'seed {seed}: matched-training verification failed: '
                       f'{record["matched"]}')


def heldout_critic(r: Rung, seed, arms):
  import jax.numpy as jnp
  from crl import checkpoint
  roots_path = ROOT / CONFIG['inputs']['heldout_roots']
  roots = load_npz(roots_path)['state'].astype(np.float32)
  actions = np.asarray(NATIVE['candidate_actions'], np.float32)
  goal = np.broadcast_to(GOAL, roots.shape)
  observation = jnp.asarray(np.concatenate([roots, goal], axis=1))
  network = make_network()
  rng = np.random.default_rng(int(NATIVE['bootstrap_seed']))
  idx = rng.integers(0, len(roots), size=(5000, len(roots)))
  result = {'root_count': int(len(roots)), 'arms': {}}
  for arm in arms:
    _, final = checkpoint.load_checkpoint(r.crl_dir(seed, arm) / 'final.pkl')
    scores = []
    for action in actions:
      batch = jnp.asarray(np.broadcast_to(action, (len(roots), 2)))
      phi, psi = network.representation_network.apply(
          final.q_params, observation, batch)
      score = jnp.sum(phi * psi, axis=1)
      scores.append(np.asarray(score, np.float32).reshape(len(roots), -1)[:, 0])
    margin = np.stack(scores, axis=1)
    margin = margin[:, 0] - margin[:, 1]
    result['arms'][arm] = {
        'down_minus_right_mean': float(margin.mean()),
        'down_minus_right_ci95': [float(x) for x in np.quantile(
            margin[idx].mean(1), [.025, .975])],
        'roots_preferring_down': int((margin > 0).sum()),
        'per_root': margin.tolist()}
  write_json(r.seed_dir(seed) / 'heldout_critic.json', result)
  return result


def stage_evaluate(r: Rung, seeds, arms):
  for seed in seeds:
    verify_seed(r, seed, arms)
    heldout_critic(r, seed, arms)
    for policy in NATIVE['native_protocols']:
      target = r.seed_dir(seed) / 'native' / policy
      if (target / 'summary.json').exists():
        continue
      cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes']
      for arm in arms:
        cmd += ['--ckpt', f'{arm}={r.crl_dir(seed, arm) / "final.pkl"}']
      cmd += ['--episodes', str(NATIVE['native_episodes']),
              '--policy', policy,
              '--reset-seed-base', str(NATIVE['native_reset_seed_base']),
              '--action-seed-base', str(NATIVE['native_action_seed_base']),
              '--out', target]
      run(cmd, r.logs / f'native_seed{seed}_{policy}.log')


def seed_rows(r: Rung, seed, arms):
  critic = json.loads((r.seed_dir(seed) / 'heldout_critic.json').read_text())
  native = {policy: json.loads(
      (r.seed_dir(seed) / 'native' / policy / 'summary.json').read_text())
      for policy in NATIVE['native_protocols']}
  rows = {}
  for arm in arms:
    rows[arm] = {
        'mode_reach': native['mode']['policies'][arm]['reach']['mean'],
        'mode_lower': native['mode']['policies'][arm]['lower_route']['mean'],
        'sample_reach': native['sample']['policies'][arm]['reach']['mean'],
        'sample_lower': native['sample']['policies'][arm]['lower_route']['mean'],
        'critic_down_minus_right': critic['arms'][arm]['down_minus_right_mean'],
        'critic_roots_down': critic['arms'][arm]['roots_preferring_down']}
  return rows


def stage_summarize(r: Rung, seeds, arms):
  per_seed = {int(s): seed_rows(r, s, arms) for s in seeds}
  metrics = ('mode_reach', 'mode_lower', 'sample_reach', 'sample_lower',
             'critic_down_minus_right')
  mean = {a: {m: float(np.mean([per_seed[s][a][m] for s in per_seed]))
              for m in metrics} for a in arms}
  dataset = json.loads((r.out / 'dataset.json').read_text())
  result = {'force_safe_prob': r.p, 'dataset': dataset, 'seeds': list(seeds),
            'arms': list(arms), 'bc_coef': BC_COEF, 'steps': CRL_STEPS,
            'native_episodes': NATIVE['native_episodes'],
            'per_seed': per_seed, 'mean_over_seeds': mean}
  write_json(r.out / 'results.json', result)
  lines = [f'# F4 detour ladder, rung {r.p:g}: '
           f'{dataset["n_forced_safe"]}/{dataset["n_teacher"]} teacher '
           f'episodes on the safe route',
           '', f'BC {BC_COEF}, {CRL_STEPS} updates per arm, '
           f'{NATIVE["native_episodes"]} native episodes (same reset seeds '
           'for every arm, seed and rung). O = plain CRL on the rung; C = '
           'fixed-ETT query-coverage replay (50/50). Critic = held-out '
           'down-minus-right logit on 16 roots (>0 prefers the detour).', '',
           '| seed | arm | mode reach | mode lower | sample reach | '
           'sample lower | critic d-r | roots down |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
  for s in per_seed:
    for a in arms:
      x = per_seed[s][a]
      lines.append(f'| {s} | {a} | {x["mode_reach"]:.3f} | {x["mode_lower"]:.3f} '
                   f'| {x["sample_reach"]:.3f} | {x["sample_lower"]:.3f} | '
                   f'{x["critic_down_minus_right"]:+.3f} | '
                   f'{x["critic_roots_down"]}/16 |')
  for a in arms:
    x = mean[a]
    lines.append(f'| mean | {a} | {x["mode_reach"]:.3f} | {x["mode_lower"]:.3f} '
                 f'| {x["sample_reach"]:.3f} | {x["sample_lower"]:.3f} | '
                 f'{x["critic_down_minus_right"]:+.3f} | |')
  (r.out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines), flush=True)


def aggregate(out_root):
  out_root = Path(out_root)
  rungs = sorted(p for p in out_root.glob('far*') if (p / 'results.json').exists())
  if not rungs:
    raise SystemExit(f'no completed rungs under {out_root}')
  results = [json.loads((p / 'results.json').read_text()) for p in rungs]
  lines = ['# F4 detour ladder: baseline (O) vs query-coverage method (C)', '',
           'Mean over learner seeds; native evaluation with identical reset '
           'seeds everywhere. mode = deterministic actor, sample = sampled '
           'actions; reach = success at 0.5, lower = took the safe lower route.',
           '', '| rung | safe-route eps | seeds | arm | mode reach | mode lower '
           '| sample reach | sample lower | critic d-r |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
  for res in results:
    for a in res['arms']:
      x = res['mean_over_seeds'][a]
      lines.append(f'| {res["force_safe_prob"]:g} | '
                   f'{res["dataset"]["n_forced_safe"]}/{res["dataset"]["n_teacher"]} '
                   f'| {len(res["seeds"])} | {a} | {x["mode_reach"]:.3f} | '
                   f'{x["mode_lower"]:.3f} | {x["sample_reach"]:.3f} | '
                   f'{x["sample_lower"]:.3f} | {x["critic_down_minus_right"]:+.3f} |')
  lines += ['', 'Per-seed rows:', '',
            '| rung | seed | arm | mode reach | mode lower | sample reach | '
            'sample lower | critic d-r | roots down |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
  for res in results:
    for s, arms in res['per_seed'].items():
      for a, x in arms.items():
        lines.append(f'| {res["force_safe_prob"]:g} | {s} | {a} | '
                     f'{x["mode_reach"]:.3f} | {x["mode_lower"]:.3f} | '
                     f'{x["sample_reach"]:.3f} | {x["sample_lower"]:.3f} | '
                     f'{x["critic_down_minus_right"]:+.3f} | '
                     f'{x["critic_roots_down"]}/16 |')
  (out_root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(out_root / 'results.json', {'rungs': results})
  print('\n'.join(lines))


STAGES = ('dataset', 'rollout_actor', 'nominal', 'generate', 'train',
          'evaluate', 'summarize')


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=STAGES + ('all', 'aggregate'))
  ap.add_argument('--rung', type=float, default=None)
  ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  ap.add_argument('--arms', nargs='+', default=['O', 'C'],
                  choices=ARMS_ALL)
  ap.add_argument('--out-root', default=None)
  ap.add_argument('--smoke', action='store_true',
                  help='tiny budgets, separate output root; chain test only')
  args = ap.parse_args(argv)
  global SMOKE, CRL_STEPS
  if args.smoke:
    SMOKE = True
    CRL_STEPS = 500
    NATIVE['native_episodes'] = 10
  out_root = args.out_root or (
      str(OUT_ROOT) + ('_smoke' if args.smoke else ''))
  args.out_root = out_root
  if args.command == 'aggregate':
    aggregate(args.out_root)
    return 0
  if args.rung is None:
    raise SystemExit('--rung is required')
  arms = [a for a in ARMS_ALL if a in args.arms]
  r = Rung(args.rung, args.out_root)
  r.out.mkdir(parents=True, exist_ok=True)
  write_json(r.out / 'config.json', {
      'force_safe_prob': r.p, 'seeds': args.seeds, 'arms': arms,
      'bc_coef': BC_COEF, 'crl_steps': CRL_STEPS, 'smoke': SMOKE,
      'sealed_experiment': str(SEALED.relative_to(ROOT)),
      'sealed_config_sha256': sha256(SEALED / 'config.json'),
      'rollout_actor_recipe': 'zbase 150k bc 0.5 seed 0 on the rung',
      'nominal_recipe': 'expert_positive MDN k=5 20k steps seed 0 on the rung',
      'git_head': subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
  todo = STAGES if args.command == 'all' else (args.command,)
  for stage in todo:
    print(f'=== rung {r.p:g} :: {stage}', flush=True)
    started = time.time()
    if stage == 'dataset':
      stage_dataset(r)
    elif stage == 'rollout_actor':
      stage_rollout_actor(r)
    elif stage == 'nominal':
      stage_nominal(r)
    elif stage == 'generate':
      stage_generate(r, arms)
    elif stage == 'train':
      stage_train(r, args.seeds, arms)
    elif stage == 'evaluate':
      stage_evaluate(r, args.seeds, arms)
    elif stage == 'summarize':
      stage_summarize(r, args.seeds, arms)
    print(f'=== rung {r.p:g} :: {stage} done in {time.time() - started:.0f}s',
          flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
