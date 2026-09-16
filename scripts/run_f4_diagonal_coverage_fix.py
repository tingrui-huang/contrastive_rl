"""Step 3: the one fix the consequence audit points at -- diagonal query coverage.

The audit (outputs/pointmaze_action_consequences_v1) found: both critics rank a
diagonal first action ~(+0.5, -0.3) above DOWN although its real consequence is
far worse (reach 0.41-0.49 vs 0.95), the fixed ETT already predicts that
(0.28-0.32 vs 0.83), and no such action ever entered the C replay as a
covered first-step query (0 of them; DOWN and RIGHT had 1090 each).  So the
critic's high score sits in an action region between the two covered
clusters that NCE never saw.

Fix, in the spirit of "change the samples, not the loss": at the SAME 550
training contexts the sealed generator used (supervised fork roots; the 16
evaluation roots are not touched), add six diagonal first-step queries
spanning the region between DOWN and RIGHT, generate their full consequences
with the fixed ETT / nominal / continuation actor exactly as arm C did, and
append every generated path -- successes and failures alike -- to the C
replay.  Arm D = replay_C + those 3300 paths.  Train C and D from identical
initializations for three seeds with the unchanged CRL loss (bc 0.05, 30k x
10), evaluate natively (mode + sample, 200 paired episodes) and on the 16
held-out roots (DOWN-vs-RIGHT critic margin, and where each critic's argmax
over the action square physically leads).

Stages: generate (needs torch) | train | evaluate | summarize.
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
sys.path.insert(0, str(ROOT / 'scripts'))
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
PILOT = ROOT / 'outputs' / 'pointmaze_learned_ett_crl_20260915_v1'
sys.path.insert(0, str(PILOT))
DIAG = ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
OUT = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
CONFIG = json.loads((SEALED / 'config.json').read_text(encoding='utf-8'))
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ARMS = ('C', 'D')
BC_COEF = 0.05
CRL_STEPS = int(CONFIG['crl']['steps_per_arm'])
NATIVE = CONFIG['heldout_evaluation']
GEN_SEED = int(CONFIG['seed']) + 10_000       # distinct streams from arm C
#: the added first-step queries: the diagonal region between DOWN and RIGHT
#: where both critics' maxima sat (~(+0.5, -0.3)); a fixed, sealed set
DIAG_QUERIES = {
    'diag_c': [0.50, -0.30], 'diag_steep': [0.35, -0.45],
    'diag_flat': [0.65, -0.20], 'diag_low': [0.50, -0.50],
    'diag_in': [0.30, -0.30], 'diag_out': [0.70, -0.40],
}


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def plain(v):
  if isinstance(v, dict):
    return {str(k): plain(x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [plain(x) for x in v]
  if isinstance(v, np.ndarray):
    return v.tolist()
  if isinstance(v, np.generic):
    return v.item()
  return v


def write_json(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(plain(value), indent=2), encoding='utf-8')


def load_npz(path):
  with np.load(path, allow_pickle=False) as d:
    return {k: d[k] for k in d.files}


def tree_sha(tree):
  import jax
  d = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.ascontiguousarray(np.asarray(leaf))
    d.update(str(a.dtype).encode())
    d.update(str(a.shape).encode())
    d.update(a.tobytes())
  return d.hexdigest()


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def run(cmd, log):
  Path(log).parent.mkdir(parents=True, exist_ok=True)
  print('  $', ' '.join(str(c) for c in cmd), flush=True)
  with open(log, 'w', encoding='utf-8') as h:
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, stdout=h,
                   stderr=subprocess.STDOUT)


def replay_path(arm):
  return SEALED / 'replay_C.npz' if arm == 'C' else OUT / 'replay_D.npz'


# ---------------------------------------------------------------- generate
def stage_generate():
  if replay_path('D').exists():
    print('replay_D exists', flush=True)
    return
  import jax
  import jax.numpy as jnp
  import torch
  from crl import checkpoint
  from learned_ett import legal_numpy, load_checkpoint
  from propensity.nominal_policy import load_nominal_policy
  torch.set_num_threads(4)
  model, _ = load_checkpoint(ROOT / CONFIG['inputs']['ett_checkpoint'], 'cpu')
  nominal = load_nominal_policy((ROOT / CONFIG['inputs']['nominal']).parent)
  _, actor_state = checkpoint.load_checkpoint(ROOT / CONFIG['inputs']['rollout_actor'])
  network = make_network()
  roots = load_npz(SEALED / 'construction_roots.npz')

  @jax.jit
  def actor_sample(params, obs, key):
    return network.sample(network.policy_network.apply(params, obs), key)

  names = list(DIAG_QUERIES)
  cands = np.asarray([DIAG_QUERIES[n] for n in names], np.float32)
  context_root = roots['context_root'].astype(np.int32)
  context_state = roots['state'][context_root]
  context_id = np.repeat(np.arange(len(context_root), dtype=np.int32), len(cands))
  cand_id = np.tile(np.arange(len(cands), dtype=np.int8), len(context_root))
  root_id = context_root[context_id]
  paths = len(context_id)
  state = np.zeros((paths, 51, 8), np.float32)
  action = np.zeros((paths, 51, 2), np.float32)
  failed = np.zeros((paths, 51), bool)
  onset = np.zeros((paths, 49), bool)
  state[:, 0] = roots['state'][root_id]
  goal_c = np.broadcast_to(GOAL, context_state.shape).astype(np.float32)
  first_xb = np.asarray(nominal.sample(
      jnp.asarray(context_state), jax.random.PRNGKey(GEN_SEED + 1000), 1,
      goal=jnp.asarray(goal_c)), np.float32)[context_id]
  rng = torch.Generator(device='cpu')
  rng.manual_seed(GEN_SEED + 2000)
  t0 = time.time()
  for step in range(49):
    cur = state[:, step]
    goal = np.broadcast_to(GOAL, cur.shape).astype(np.float32)
    xb = first_xb if step == 0 else np.asarray(nominal.sample(
        jnp.asarray(cur), jax.random.PRNGKey(GEN_SEED + 3000 + step), 1,
        goal=jnp.asarray(goal)), np.float32)
    obs = jnp.asarray(np.concatenate([cur, goal], axis=1))
    a_actor = np.asarray(actor_sample(actor_state.policy_params, obs,
                                      jax.random.PRNGKey(GEN_SEED + 4000 + step)),
                         np.float32)
    xq = cands[cand_id].copy() if step == 0 else a_actor
    action[:, step] = xq
    with torch.no_grad():
      succ, event, _, _ = model.sample_live(
          torch.as_tensor(cur), torch.as_tensor(xb), torch.as_tensor(xq), rng)
    succ = succ.numpy().astype(np.float32)
    event = event.numpy()
    was = failed[:, step]
    persistent = np.concatenate([cur[:, :2], cur[:, :6]], axis=1)
    succ[was] = persistent[was]
    event[was] = False
    state[:, step + 1] = succ
    failed[:, step + 1] = was | event
    onset[:, step] = event
  state[:, 50] = state[:, 49]
  failed[:, 50] = failed[:, 49]
  if np.any(state[:, 1:50, 2:] != state[:, :49, :6]):
    raise AssertionError('F4 shift violation')
  if not legal_numpy(state[:, :50, :2]).all():
    raise AssertionError('illegal emitted endpoint')
  xy = state[:, :50, :2]
  reach = np.any(np.linalg.norm(xy - GOAL[:2], axis=2) < 0.5, axis=1)
  lower = np.any(xy[:, :, 1] < 2.0, axis=1)
  summary = {'paths': int(paths), 'queries': DIAG_QUERIES,
             'reach_0p5_fraction': float(reach.mean()),
             'lower_route_fraction': float(lower.mean()),
             'absorbed_fraction': float(failed[:, 49].mean()),
             'onsets': int(onset.sum()), 'wall_seconds': time.time() - t0,
             'per_query': {n: {'reach': float(reach[cand_id == i].mean()),
                               'absorbed': float(failed[cand_id == i, 49].mean()),
                               'lower': float(lower[cand_id == i].mean())}
                           for i, n in enumerate(names)}}
  # replay D = the whole sealed C replay (original half + C paths) + D paths
  c = load_npz(SEALED / 'replay_C.npz')
  goal_s = np.broadcast_to(GOAL, state.shape).astype(np.float32)
  d_obs = np.concatenate([state, goal_s], axis=2)
  n_c = len(c['obs'])
  meta = json.loads(str(c['meta']))
  meta.update({'arm': 'D', 'added_queries': DIAG_QUERIES,
               'added_paths': int(paths), 'base_replay': 'replay_C.npz',
               'base_replay_sha256': sha256(SEALED / 'replay_C.npz')})
  onset_time = np.where(onset.any(1), onset.argmax(1), -1).astype(np.int16)
  OUT.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      replay_path('D'),
      obs=np.concatenate([c['obs'], d_obs]).astype(np.float32),
      act=np.concatenate([c['act'], action]).astype(np.float32),
      lengths=np.concatenate([c['lengths'], np.full(paths, 50, np.int16)]),
      meta=np.asarray(json.dumps(meta, sort_keys=True)),
      audit_source=np.concatenate([c['audit_source'], np.full(paths, 2, np.int8)]),
      audit_original_episode=np.concatenate([c['audit_original_episode'],
                                             np.full(paths, -1, np.int32)]),
      audit_root_id=np.concatenate([c['audit_root_id'], root_id.astype(np.int32)]),
      audit_context_id=np.concatenate([c['audit_context_id'],
                                       context_id.astype(np.int32)]),
      audit_candidate_id=np.concatenate([c['audit_candidate_id'],
                                         (cand_id + 6).astype(np.int8)]),
      audit_onset_time=np.concatenate([c['audit_onset_time'], onset_time]),
      audit_failed_at_end=np.concatenate([c['audit_failed_at_end'],
                                          failed[:, 49]]))
  np.savez_compressed(OUT / 'generated_D.npz', states=state, actions=action,
                      failed=failed, onset_event=onset, root_id=root_id,
                      context_id=context_id, candidate_id=cand_id)
  summary['replay_D_sha256'] = sha256(replay_path('D'))
  summary['replay_D_episodes'] = int(n_c + paths)
  write_json(OUT / 'generation.json', summary)
  print(json.dumps(plain(summary), indent=1)[:1500], flush=True)


# ------------------------------------------------------------------- train
def crl_config(seed, arm):
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0',
      offline_dataset=str(replay_path(arm)),
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
      ckpt_every_steps=CRL_STEPS,
      ckpt_dir=str(OUT / 'seeds' / f'seed_{seed}' / 'crl' / arm))


def stage_train(seeds, arms):
  from crl.train import train
  for seed in seeds:
    for arm in arms:
      cfg = crl_config(seed, arm)
      out = Path(cfg.ckpt_dir)
      if (out / 'final.pkl').exists():
        print(f'seed {seed} arm {arm} exists', flush=True)
        continue
      diff = {k for k in dataclasses.asdict(cfg)
              if dataclasses.asdict(cfg)[k] != dataclasses.asdict(crl_config(seed, 'C'))[k]}
      assert diff <= {'offline_dataset', 'ckpt_dir'}, diff
      print(f'TRAIN seed {seed} arm {arm} on {Path(cfg.offline_dataset).name}',
            flush=True)
      t0 = time.time()
      train(cfg)
      write_json(out / 'run_summary.json', {
          'seed': seed, 'arm': arm, 'dataset': str(cfg.offline_dataset),
          'dataset_sha256': sha256(cfg.offline_dataset), 'bc_coef': BC_COEF,
          'steps': CRL_STEPS, 'wall_seconds': time.time() - t0})


# ---------------------------------------------------------------- evaluate
def stage_evaluate(seeds, arms):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  network = make_network()
  roots = load_npz(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                   / 'root_selection.npz')['state'].astype(np.float32)
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))
  maps = load_npz(DIAG / 'seed0_heldout_score_maps.npz')
  outcome_maps = maps['outcome_maps']          # [16, GRID, GRID] physics
  axis = maps['grid_axis'].astype(np.float32)
  gx, gy = np.meshgrid(axis, axis, indexing='xy')
  grid = np.stack([gx.ravel(), gy.ravel()], 1)

  @jax.jit
  def score(q_params, o, a):
    phi, psi = network.representation_network.apply(q_params, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  actions = np.asarray(NATIVE['candidate_actions'], np.float32)
  for seed in seeds:
    sd = OUT / 'seeds' / f'seed_{seed}'
    inits = {}
    crit = {}
    for arm in arms:
      _, init = checkpoint.load_checkpoint(sd / 'crl' / arm / 'init.pkl')
      _, final = checkpoint.load_checkpoint(sd / 'crl' / arm / 'final.pkl')
      inits[arm] = tree_sha(init)
      qs = np.stack([np.asarray(score(final.q_params, obs,
                                      jnp.asarray(np.broadcast_to(a, (16, 2)))))
                     for a in actions], 1)
      margin = qs[:, 0] - qs[:, 1]
      argmax_region = []
      argmax_actions = []
      for i in range(16):
        o_rep = jnp.asarray(np.repeat(np.asarray(obs[i:i + 1]), len(grid), 0))
        qg = np.asarray(score(final.q_params, o_rep, jnp.asarray(grid)))
        k = int(np.argmax(qg))
        argmax_actions.append(grid[k])
        argmax_region.append(int(outcome_maps[i].ravel()[k]))
      argmax_region = np.asarray(argmax_region)
      crit[arm] = {
          'down_minus_right_mean': float(margin.mean()),
          'roots_preferring_down': int((margin > 0).sum()),
          'argmax_mean_action': np.mean(argmax_actions, 0),
          'argmax_region_lower': int((argmax_region == 1).sum()),
          'argmax_region_shortcut': int((argmax_region == 2).sum()),
          'argmax_region_other': int((argmax_region == 0).sum()),
          'q_down_mean': float(qs[:, 0].mean()), 'q_right_mean': float(qs[:, 1].mean())}
    write_json(sd / 'heldout_critic.json', {
        'identical_initial_state_across_arms': len(set(inits.values())) == 1,
        'arms': crit})
    for policy in NATIVE['native_protocols']:
      target = sd / 'native' / policy
      if (target / 'summary.json').exists():
        continue
      cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes']
      for arm in arms:
        cmd += ['--ckpt', f'{arm}={sd / "crl" / arm / "final.pkl"}']
      cmd += ['--episodes', str(NATIVE['native_episodes']), '--policy', policy,
              '--reset-seed-base', str(NATIVE['native_reset_seed_base']),
              '--action-seed-base', str(NATIVE['native_action_seed_base']),
              '--out', target]
      run(cmd, OUT / 'logs' / f'native_seed{seed}_{policy}.log')


def stage_summarize(seeds, arms):
  rows = {}
  for seed in seeds:
    sd = OUT / 'seeds' / f'seed_{seed}'
    crit = json.loads((sd / 'heldout_critic.json').read_text())['arms']
    nat = {p: json.loads((sd / 'native' / p / 'summary.json').read_text())
           for p in NATIVE['native_protocols']}
    for arm in arms:
      rows[(seed, arm)] = {
          'mode_reach': nat['mode']['policies'][arm]['reach']['mean'],
          'mode_lower': nat['mode']['policies'][arm]['lower_route']['mean'],
          'sample_reach': nat['sample']['policies'][arm]['reach']['mean'],
          'sample_lower': nat['sample']['policies'][arm]['lower_route']['mean'],
          'critic_down_minus_right': crit[arm]['down_minus_right_mean'],
          'roots_down': crit[arm]['roots_preferring_down'],
          'argmax_shortcut_roots': crit[arm]['argmax_region_shortcut'],
          'argmax_lower_roots': crit[arm]['argmax_region_lower'],
          'argmax_mean_action': crit[arm]['argmax_mean_action']}
  gen = json.loads((OUT / 'generation.json').read_text())
  lines = ['# Diagonal query coverage fix: C vs D', '',
           f'D = C replay + {gen["paths"]} ETT paths from the 550 training '
           f'contexts with six diagonal first queries {list(DIAG_QUERIES.values())}; '
           f'those paths: reach {gen["reach_0p5_fraction"]:.3f}, absorbed '
           f'{gen["absorbed_fraction"]:.3f}, lower {gen["lower_route_fraction"]:.3f}. '
           'Same CRL loss, bc 0.05, 30k x 10, identical initialization per seed; '
           '200 native episodes at the sealed seeds; critic columns on the 16 '
           'held-out roots (argmax = where the critic\'s best action over the '
           'square physically leads: shortcut / lower roots out of 16).', '',
           '| seed | arm | mode reach | mode lower | sample reach | sample lower '
           '| critic d-r | roots down | argmax -> shortcut / lower | argmax action |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
  for (seed, arm), x in rows.items():
    lines.append(f'| {seed} | {arm} | {x["mode_reach"]:.3f} | {x["mode_lower"]:.3f} '
                 f'| {x["sample_reach"]:.3f} | {x["sample_lower"]:.3f} | '
                 f'{x["critic_down_minus_right"]:+.3f} | {x["roots_down"]}/16 | '
                 f'{x["argmax_shortcut_roots"]} / {x["argmax_lower_roots"]} | '
                 f'({x["argmax_mean_action"][0]:+.2f},{x["argmax_mean_action"][1]:+.2f}) |')
  for arm in arms:
    xs = [rows[(s, arm)] for s in seeds]
    lines.append(f'| mean | {arm} | {np.mean([x["mode_reach"] for x in xs]):.3f} | '
                 f'{np.mean([x["mode_lower"] for x in xs]):.3f} | '
                 f'{np.mean([x["sample_reach"] for x in xs]):.3f} | '
                 f'{np.mean([x["sample_lower"] for x in xs]):.3f} | '
                 f'{np.mean([x["critic_down_minus_right"] for x in xs]):+.3f} | | '
                 f'{np.mean([x["argmax_shortcut_roots"] for x in xs]):.1f} / '
                 f'{np.mean([x["argmax_lower_roots"] for x in xs]):.1f} | |')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(OUT / 'results.json', {'generation': gen, 'rows': {
      f'seed{s}_{a}': v for (s, a), v in rows.items()}})
  print('\n'.join(lines), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('generate', 'train', 'evaluate',
                                      'summarize', 'all'))
  ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  args = ap.parse_args(argv)
  stages = (('generate', 'train', 'evaluate', 'summarize')
            if args.command == 'all' else (args.command,))
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'generate':
      stage_generate()
    elif st == 'train':
      stage_train(args.seeds, ARMS)
    elif st == 'evaluate':
      stage_evaluate(args.seeds, ARMS)
    else:
      stage_summarize(args.seeds, ARMS)
  return 0


if __name__ == '__main__':
  sys.exit(main())
