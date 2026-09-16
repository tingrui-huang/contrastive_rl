"""Arm E: corner / bottom-edge query coverage, the same recipe as the Step 3 fix.

Step 9f classified every two-stage actor by its fork mode under the task
goal: DOWN (y = -1, x 0.2-0.4; detour 1.000, robust), CORNER (y = -1,
x 0.65-1.0; a physics race that actor seeds and resets decide) and RIGHT.
Only two of six D critics produce the DOWN family.  In the D replay the fork
rows are DOWN 1728 / RIGHT 4499 / CORNER (1,-1) 212 / gentle-down (0.3,-1)
139 (within 0.2), so whether a critic separates DOWN from CORNER rests on a
thin band of data and comes out seed-dependent.

Fix, in the spirit of Step 3 ("change the samples, not the loss"): at the
SAME 550 training contexts, add first-step queries on the bottom edge of the
action square, generate their full consequences with the fixed ETT / nominal
/ continuation actor exactly as arms C and D did, and append every path to
the D replay.  Six queries were audited against the native physics; the two
the ETT gets right -- the gentle-down action (0.3, -1) and the corner (1, -1)
-- are generated.  Arm E = replay_D + 1100 paths.

Stages
  audit         gate: native vs ETT consequences of the six queries at the 16
                held-out roots (64 paired replicates), reusing
                scripts/audit_f4_action_consequences.py
  generate      replay_E (needs torch)
  train         five 30k critics on replay_E, sealed config, inits matching
                the D series seed for seed
  critic_check  per critic: 16-root DOWN-RIGHT, and the smoothed DOWN-vs-
                CORNER preference Qbar(loc (0.3,-6)) - Qbar(loc (5,-5)) at
                policy width 0.75 (D series: +0.54 / +0.23 / -0.20)
  actors        three fresh balanced-BC actors per critic (300k, Acme, rg0,
                bc 0.05, cap 0.25) on replay_E; 300 new-seed episodes;
                DOWN / CORNER / RIGHT classification of the fork mode
  summarize     REPORT.md
"""
from __future__ import annotations

import argparse
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
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
DIAG = ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
OUT = ROOT / 'outputs' / 'pointmaze_corner_coverage_fix_v1'
CONFIG = json.loads((SEALED / 'config.json').read_text(encoding='utf-8'))
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
BC_COEF = 0.05
BC_CAP = 0.25
CRL_STEPS = int(CONFIG['crl']['steps_per_arm'])
GEN_SEED = int(CONFIG['seed']) + 20_000          # distinct stream from C (0) and D (+10k)
CRITIC_SEEDS = (0, 1, 2, 3, 4)
ACTOR_SEEDS = (0, 1, 2)
NEW_EVAL = {'episodes': 300, 'reset': 31_500_000, 'action': 31_600_000}
#: the added first-step queries.  Six were audited (bottom edge x 0.3 / 0.6
#: / 0.85, corner, right edge y -0.85 / -0.6; audit/REPORT.md): the fixed ETT
#: agrees with the native physics for the gentle-down action (0.3, -1) (lower
#: 1.00 both) and is direction-right but over-pessimistic for the corner
#: (native reach 0.55 / lower 0.46, ETT 0.24 / 0.15 -- the same kind and
#: size of bias as the Step 3 diagonals), while it flattens the rest of the
#: band to RIGHT-like outcomes ((0.6, -1): native lower 0.85, ETT 0.16).
#: Only the two queries the ETT gets right are generated (user decision).
AUDITED_QUERIES = {
    'edge_x030': [0.30, -1.00], 'edge_x060': [0.60, -1.00],
    'edge_x085': [0.85, -1.00], 'corner': [1.00, -1.00],
    'right_y085': [1.00, -0.85], 'right_y060': [1.00, -0.60],
}
CORNER_QUERIES = {'edge_x030': [0.30, -1.00], 'corner': [1.00, -1.00]}
D_CRITICS = {s: FIX / 'seeds' / f'seed_{s}' / 'crl' / 'D' / 'final.pkl' for s in (0, 1, 2)}


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
                   stderr=subprocess.STDOUT,
                   env={**os.environ, 'PYTHONPATH': str(ROOT)})


REPLAY_E = OUT / 'replay_E.npz'


# ------------------------------------------------------------------- audit
def stage_audit(reps):
  """Gate: do the six queries' consequences in the fixed ETT agree with the
  native environment at the 16 held-out roots?"""
  import jax
  from crl import checkpoint
  import audit_f4_action_consequences as aac
  if (OUT / 'audit' / 'results.json').exists():
    print('audit exists', flush=True)
    return
  with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
               / 'root_selection.npz', allow_pickle=False) as d:
    roots = d['state'].astype(np.float32)
    phys = d['physical_state'].astype(float)
  names = list(AUDITED_QUERIES)
  n = len(roots)
  actions = np.zeros((n, len(names), 2), np.float32)
  for a, nm in enumerate(names):
    actions[:, a] = AUDITED_QUERIES[nm]
  # reference actions from the earlier audit for scale: DOWN and RIGHT
  names += ['down', 'right']
  actions = np.concatenate([actions, np.tile(np.array([[[0., -1.]], [[1., 0.]]], np.float32)
                                             .transpose(1, 0, 2), (n, 1, 1))], 1)
  network = make_network()
  _, actor_state = checkpoint.load_checkpoint(aac.ROLLOUT_ACTOR)
  actor_params = actor_state.policy_params
  print(f'native rollouts: {n} roots x {len(names)} actions x {reps} paired replicates', flush=True)
  native = aac.native_rollouts(phys, actions, actor_params, network, reps, aac.KEY_BASE + 5)
  print('ETT rollouts', flush=True)
  ett = aac.ett_rollouts(roots, actions, actor_params, network, reps, aac.KEY_BASE + 5)
  metrics = ('reach', 'strict_success', 'discounted_return', 'lower', 'shortcut', 'absorbed')
  summary = {}
  lines = ['# Corner-query consequence audit: native vs fixed ETT, 16 held-out roots', '',
           f'{reps} paired replicates per (root, action); continuation = the sealed rollout '
           'actor with paired keys; means over roots.', '',
           '| action | ' + ' | '.join(f'native {m}' for m in metrics) + ' | '
           + ' | '.join(f'ETT {m}' for m in metrics) + ' |',
           '|---|' + '---:|' * (2 * len(metrics))]
  for a, nm in enumerate(names):
    nat = {m: float(np.mean([native[i][a][m] for i in range(n)])) for m in metrics}
    et = {m: float(np.mean([ett[i][a][m] for i in range(n)])) for m in metrics}
    summary[nm] = {'action': actions[0, a].tolist(), 'native': nat, 'ett': et}
    lines.append(f'| {nm} {actions[0, a].round(2).tolist()} | '
                 + ' | '.join(f'{nat[m]:.3f}' for m in metrics) + ' | '
                 + ' | '.join(f'{et[m]:.3f}' for m in metrics) + ' |')
  # the gate: the ETT must rank the queries as the environment does
  order_nat = sorted(names, key=lambda k: -summary[k]['native']['reach'])
  order_ett = sorted(names, key=lambda k: -summary[k]['ett']['reach'])
  rho = float(np.corrcoef([summary[k]['native']['reach'] for k in names],
                          [summary[k]['ett']['reach'] for k in names])[0, 1])
  lines += ['', f'reach ordering, native: {order_nat}', f'reach ordering, ETT:    {order_ett}',
            f'correlation of per-action reach (native vs ETT): {rho:.3f}']
  write_json(OUT / 'audit' / 'results.json', {'reps': reps, 'actions': names, 'summary': summary,
                                              'reach_corr': rho, 'order_native': order_nat,
                                              'order_ett': order_ett})
  (OUT / 'audit' / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines), flush=True)


# ---------------------------------------------------------------- generate
def stage_generate():
  if REPLAY_E.exists():
    print('replay_E exists', flush=True)
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

  names = list(CORNER_QUERIES)
  cands = np.asarray([CORNER_QUERIES[n] for n in names], np.float32)
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
  summary = {'paths': int(paths), 'queries': CORNER_QUERIES,
             'reach_0p5_fraction': float(reach.mean()),
             'lower_route_fraction': float(lower.mean()),
             'absorbed_fraction': float(failed[:, 49].mean()),
             'onsets': int(onset.sum()), 'wall_seconds': time.time() - t0,
             'per_query': {n: {'reach': float(reach[cand_id == i].mean()),
                               'absorbed': float(failed[cand_id == i, 49].mean()),
                               'lower': float(lower[cand_id == i].mean())}
                           for i, n in enumerate(names)}}
  d = load_npz(FIX / 'replay_D.npz')
  goal_s = np.broadcast_to(GOAL, state.shape).astype(np.float32)
  e_obs = np.concatenate([state, goal_s], axis=2)
  n_d = len(d['obs'])
  meta = json.loads(str(d['meta']))
  meta.update({'arm': 'E', 'added_queries': CORNER_QUERIES,
               'added_paths': int(paths), 'base_replay': 'replay_D.npz',
               'base_replay_sha256': sha256(FIX / 'replay_D.npz')})
  onset_time = np.where(onset.any(1), onset.argmax(1), -1).astype(np.int16)
  OUT.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      REPLAY_E,
      obs=np.concatenate([d['obs'], e_obs]).astype(np.float32),
      act=np.concatenate([d['act'], action]).astype(np.float32),
      lengths=np.concatenate([d['lengths'], np.full(paths, 50, np.int16)]),
      meta=np.asarray(json.dumps(meta, sort_keys=True)),
      audit_source=np.concatenate([d['audit_source'], np.full(paths, 3, np.int8)]),
      audit_original_episode=np.concatenate([d['audit_original_episode'],
                                             np.full(paths, -1, np.int32)]),
      audit_root_id=np.concatenate([d['audit_root_id'], root_id.astype(np.int32)]),
      audit_context_id=np.concatenate([d['audit_context_id'],
                                       context_id.astype(np.int32)]),
      audit_candidate_id=np.concatenate([d['audit_candidate_id'],
                                         (cand_id + 12).astype(np.int8)]),
      audit_onset_time=np.concatenate([d['audit_onset_time'], onset_time]),
      audit_failed_at_end=np.concatenate([d['audit_failed_at_end'], failed[:, 49]]))
  np.savez_compressed(OUT / 'generated_E.npz', states=state, actions=action,
                      failed=failed, onset_event=onset, root_id=root_id,
                      context_id=context_id, candidate_id=cand_id)
  summary['replay_E_sha256'] = sha256(REPLAY_E)
  summary['replay_E_episodes'] = int(n_d + paths)
  write_json(OUT / 'generation.json', summary)
  print(json.dumps(plain(summary), indent=1)[:1500], flush=True)


# ------------------------------------------------------------------- train
def crl_config(seed):
  """The sealed D-arm critic recipe verbatim (the critic never sees the actor,
  so random_goals / log_prob only matter for the joint actor, which is not
  used here)."""
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(REPLAY_E),
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
      eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=5_000,
      ckpt_every_steps=CRL_STEPS,
      ckpt_dir=str(OUT / 'seeds' / f'seed_{seed}' / 'crl' / 'E'))


def critic_path(seed):
  return OUT / 'seeds' / f'seed_{seed}' / 'crl' / 'E' / 'final.pkl'


def stage_train(seeds):
  from crl.train import train
  for seed in seeds:
    cfg = crl_config(seed)
    if critic_path(seed).exists():
      print(f'seed {seed} exists', flush=True)
      continue
    t0 = time.time()
    train(cfg)
    write_json(Path(cfg.ckpt_dir) / 'run_summary.json', {
        'seed': seed, 'arm': 'E', 'dataset': str(REPLAY_E),
        'dataset_sha256': sha256(REPLAY_E), 'steps': CRL_STEPS,
        'wall_seconds': time.time() - t0})


# ------------------------------------------------------------ critic_check
def critic_metrics(q_params, network, roots, eps):
  import jax
  import jax.numpy as jnp
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))

  @jax.jit
  def score(qp, o, a):
    phi, psi = network.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  def qbar(o_row, loc):
    a = np.tanh(loc[None] + 0.75 * eps).astype(np.float32)
    return float(np.mean(np.asarray(score(q_params, jnp.asarray(np.repeat(o_row[None], len(a), 0)),
                                          jnp.asarray(a)))))
  n = len(roots)
  down = np.asarray(score(q_params, obs, jnp.asarray(np.tile([[0., -1.]], (n, 1)), jnp.float32)))
  right = np.asarray(score(q_params, obs, jnp.asarray(np.tile([[1., 0.]], (n, 1)), jnp.float32)))
  corner = np.asarray(score(q_params, obs, jnp.asarray(np.tile([[1., -1.]], (n, 1)), jnp.float32)))
  gentle = np.asarray(score(q_params, obs, jnp.asarray(np.tile([[0.3, -1.]], (n, 1)), jnp.float32)))
  o_np = np.asarray(obs)
  qb_down = np.array([qbar(o_np[i], np.array([0.3, -6.0], np.float32)) for i in range(n)])
  qb_corner = np.array([qbar(o_np[i], np.array([5.0, -5.0], np.float32)) for i in range(n)])
  qb_right = np.array([qbar(o_np[i], np.array([5.0, -0.1], np.float32)) for i in range(n)])
  return {'down_minus_right': float((down - right).mean()),
          'roots_preferring_down': int((down > right).sum()),
          'gentle_minus_corner_raw': float((gentle - corner).mean()),
          'qbar_down_minus_corner': float((qb_down - qb_corner).mean()),
          'roots_qbar_down_over_corner': int((qb_down > qb_corner).sum()),
          'qbar_down_minus_right': float((qb_down - qb_right).mean())}


def stage_critic_check(seeds):
  from crl import checkpoint
  network = make_network()
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  eps = np.random.default_rng(0).standard_normal((256, 2)).astype(np.float32)
  rows = {}
  for label, path in [(f'E seed {s}', critic_path(s)) for s in seeds] + \
                     [(f'D seed {s} (reference)', p) for s, p in D_CRITICS.items()]:
    if not path.exists():
      continue
    _, st = checkpoint.load_checkpoint(path)
    rows[label] = critic_metrics(st.q_params, network, roots, eps)
  lines = ['# Critic check: DOWN vs RIGHT (16 roots) and the smoothed DOWN vs CORNER preference', '',
           'Qbar = policy-width (0.75) average of the critic over tanh(loc + 0.75 eps); DOWN loc '
           '(0.3, -6), CORNER loc (5, -5), RIGHT loc (5, -0.1). D series reference: only D1 '
           '(+0.54 here) produced DOWN-family actors; D0 (+0.23) and D2 (-0.20) produced CORNER.', '',
           '| critic | d-r raw | roots down | q(0.3,-1) - q(1,-1) raw | Qbar DOWN - CORNER | roots | Qbar DOWN - RIGHT |',
           '|---|---:|---:|---:|---:|---:|---:|']
  for label, m in rows.items():
    lines.append(f'| {label} | {m["down_minus_right"]:+.3f} | {m["roots_preferring_down"]}/16 '
                 f'| {m["gentle_minus_corner_raw"]:+.3f} | {m["qbar_down_minus_corner"]:+.3f} '
                 f'| {m["roots_qbar_down_over_corner"]}/16 | {m["qbar_down_minus_right"]:+.3f} |')
  write_json(OUT / 'critic_check.json', rows)
  (OUT / 'critic_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines), flush=True)


# ------------------------------------------------------------------ actors
def stage_actors(seeds, parallel):
  procs = []
  for s in seeds:
    for a in ACTOR_SEEDS:
      out = OUT / 'seeds' / f'seed_{s}' / f'actor_s{a}'
      if (out / 'final.pkl').exists():
        continue
      log = OUT / 'logs' / f'actor_seed{s}_s{a}.log'
      log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, str(ROOT / 'scripts' / 'train_f4_actor_fixed_critic.py'),
             '--critic', str(critic_path(s)), '--replay', str(REPLAY_E),
             '--actor-seed', str(a), '--log-prob', 'acme', '--random-goals', '0',
             '--bc-sampling', 'balanced', '--bc-cap', str(BC_CAP), '--out', str(out)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
      if len(procs) >= parallel:
        _wait(procs)
  _wait(procs)
  for s in seeds:
    target = OUT / 'seeds' / f'seed_{s}' / 'native_new_mode'
    for pol in ('mode', 'sample'):
      target = OUT / 'seeds' / f'seed_{s}' / f'native_new_{pol}'
      if (target / 'summary.json').exists():
        continue
      cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes']
      for a in ACTOR_SEEDS:
        cmd += ['--ckpt', f'E{s}_a{a}={OUT / "seeds" / f"seed_{s}" / f"actor_s{a}" / "final.pkl"}']
      cmd += ['--episodes', NEW_EVAL['episodes'], '--policy', pol,
              '--reset-seed-base', NEW_EVAL['reset'], '--action-seed-base', NEW_EVAL['action'],
              '--out', target]
      run(cmd, OUT / 'logs' / f'eval_seed{s}_{pol}.log')


def _wait(procs):
  for p, f, out in procs:
    rc = p.wait()
    f.close()
    if rc != 0:
      raise RuntimeError(f'run for {out} failed (rc {rc})')
  procs.clear()


# --------------------------------------------------------------- summarize
def stage_summarize(seeds):
  import jax.numpy as jnp
  from crl import checkpoint
  network = make_network()
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))
  crit = json.loads((OUT / 'critic_check.json').read_text(encoding='utf-8')) \
      if (OUT / 'critic_check.json').exists() else {}
  gen = json.loads((OUT / 'generation.json').read_text(encoding='utf-8'))
  lines = ['# Arm E: corner / edge query coverage', '',
           f'replay_E = replay_D + {gen["paths"]} ETT paths from the 550 training contexts with '
           f'six first queries {list(CORNER_QUERIES.values())}; those paths: reach '
           f'{gen["reach_0p5_fraction"]:.3f}, absorbed {gen["absorbed_fraction"]:.3f}, lower '
           f'{gen["lower_route_fraction"]:.3f}. Five 30k critics (sealed recipe), three fresh '
           'balanced-BC actors each (300k, Acme, rg0, bc 0.05, cap 0.25), 300 new-seed episodes. '
           'Family = fork mode under the task goal: DOWN (y < -0.9, x < 0.6), CORNER (y < -0.9, '
           'x >= 0.6), RIGHT (otherwise).', '',
           '| critic | Qbar DOWN-CORNER | actor | fork mode | family | mode reach | mode lower | sample lower |',
           '|---|---:|---|---|---|---:|---:|---:|']
  families = []
  for s in seeds:
    sm = OUT / 'seeds' / f'seed_{s}' / 'native_new_mode' / 'summary.json'
    ss = OUT / 'seeds' / f'seed_{s}' / 'native_new_sample' / 'summary.json'
    if not sm.exists():
      continue
    pm = json.loads(sm.read_text(encoding='utf-8'))['policies']
    ps = json.loads(ss.read_text(encoding='utf-8'))['policies']
    qc = crit.get(f'E seed {s}', {}).get('qbar_down_minus_corner', float('nan'))
    for a in ACTOR_SEEDS:
      _, st = checkpoint.load_checkpoint(OUT / 'seeds' / f'seed_{s}' / f'actor_s{a}' / 'final.pkl')
      m = np.tanh(np.asarray(network.policy_network.apply(st.policy_params, obs).loc)).mean(0)
      fam = 'DOWN' if (m[1] < -0.9 and m[0] < 0.6) else ('CORNER' if m[1] < -0.9 else 'RIGHT')
      families.append((s, a, fam, pm[f'E{s}_a{a}']['lower_route']['mean']))
      lines.append(f'| E seed {s} | {qc:+.3f} | a{a} | ({m[0]:+.2f}, {m[1]:+.2f}) | {fam} '
                   f'| {pm[f"E{s}_a{a}"]["reach"]["mean"]:.3f} | {pm[f"E{s}_a{a}"]["lower_route"]["mean"]:.3f} '
                   f'| {ps[f"E{s}_a{a}"]["lower_route"]["mean"]:.3f} |')
  n_down = sum(f[2] == 'DOWN' for f in families)
  n_ok = sum(f[3] >= 0.9 for f in families)
  crit_down = {s for s, a, fam, _ in families if fam == 'DOWN'}
  lines += ['', f'DOWN-family actors: {n_down}/{len(families)}; actors with mode lower >= 0.9: '
            f'{n_ok}/{len(families)}; critics producing at least one DOWN actor: {sorted(crit_down)}.',
            'Reference (D series, 6 critics x 3 actors): 13/18 at >= 0.81, DOWN family only under '
            'D1 and joint seed 0.']
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(OUT / 'results.json', {'families': families, 'critic_check': crit, 'generation': gen})
  print('\n'.join(lines), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('audit', 'generate', 'train', 'critic_check',
                                      'actors', 'summarize', 'all'))
  ap.add_argument('--seeds', type=int, nargs='+', default=list(CRITIC_SEEDS))
  ap.add_argument('--reps', type=int, default=64)
  ap.add_argument('--parallel', type=int, default=3)
  args = ap.parse_args(argv)
  stages = (('audit', 'generate', 'train', 'critic_check', 'actors', 'summarize')
            if args.command == 'all' else (args.command,))
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'audit':
      stage_audit(args.reps)
    elif st == 'generate':
      stage_generate()
    elif st == 'train':
      stage_train(args.seeds)
    elif st == 'critic_check':
      stage_critic_check(args.seeds)
    elif st == 'actors':
      stage_actors(args.seeds, args.parallel)
    elif st == 'summarize':
      stage_summarize(args.seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
