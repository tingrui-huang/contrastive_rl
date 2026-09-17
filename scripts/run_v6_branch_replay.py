"""AntMaze V6 Phase 1: critics on the oracle branch replay, gate, warm-started joint actors, vanilla control.

The PointMaze Steps 12-13 recipe on the long two-rockfall V6 benchmark
(`notes/antmaze_branch_replay_plan.md`; replay from
`scripts/build_v6_branch_replay.py replay`).  Nothing in the environment,
the dataset, the teacher or the evaluation changes; the learner's discount
is 0.999 for every arm here (at the frozen recipe's 0.99 a correct
interventional critic prefers the shortcut, section 0 of the plan).

Stages
  critics    the V6 recipe (twin-Q binary NCE, batch 1024, 1024x1024,
             repr 16, bc 0.05, rg 0) at gamma 0.999 on the branch replay,
             anchors = row 0 of every path (set_anchor_strata), 100k
             updates, five seeds; the joint actor of this stage is not used
  gate       start-region recorded anchors (x < 2, y < 2, t <= 5) with their
             recorded goals: mean critic score (min over the twin heads) on
             recorded actions whose next-frame displacement points north
             minus those pointing east; the law's own target from the
             replay's query paths; pass line --gate (default +0.15)
  joint      Step 13's schedule for every passing critic: latest.pkl = the
             critic + its Adam state + a fresh actor, then crl.train resumed
             for --joint-steps updates on the branch replay with the BC rows
             from the recorded dataset (balanced inside (cell, goal cell)
             groups by next-frame displacement direction, cap 0.25),
             milestones every 10k; then the V6 evaluation (300 natural
             draws, seed 909, mean and sampled policies, U breakdown)
  vanilla    the control: the same recipe at gamma 0.999 on the recorded
             dataset, three seeds, evaluated the same way
  summarize  REPORT.md

Outputs: outputs/antmaze_branch_replay_v1/ (V6_BRANCH_OUT).
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

OUT = Path(os.environ.get('V6_BRANCH_OUT', ROOT / 'outputs' / 'antmaze_branch_replay_v1'))
REPLAY = Path(os.environ.get('V6_BRANCH_REPLAY', OUT / 'replay_branch.npz'))
STEM = os.environ.get('V6_DATASET_STEM', 'antmaze_rockfall_clock_v6_p040')
P_ACTIVE = float(os.environ.get('V6_P_ACTIVE', '0.40'))
CRITIC_TAG = os.environ.get('V6_CRITIC_TAG', '')   # side-by-side critic set, e.g. '_30k'
JOINT_TAG = os.environ.get('V6_JOINT_TAG', '')     # side-by-side actor set, e.g. '_frozen30k'
JOINT_MODE = os.environ.get('V6_JOINT_MODE', 'joint')   # 'joint' (critic keeps training) | 'frozen' (critic lr 0)
DATASET = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset' / f'{STEM}_gxy.npz'
ENV_XY = 'offline_antmaze_rockfall_clock_v6_gxy'
HORIZON = 800
DISCOUNT = 0.999
CRITIC_STEPS = 100_000
JOINT_STEPS = 100_000
MILESTONES = tuple(range(10_000, JOINT_STEPS, 10_000))
SEEDS = (0, 1, 2, 3, 4)
VANILLA_SEEDS = (0, 1, 2)
EVAL = {'n': 300, 'seed': 909, 'action_seed': 9909}
STATE_DIM = 29
START = dict(x_max=2.0, y_max=2.0, t_max=5)
BC_CELL, BC_CAP = 4.0, 0.25          # maze scaling 4 -> one cell per group


def sha256(path):
  import hashlib
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(value, indent=2, default=float), encoding='utf-8')


# ------------------------------------------------------------------ config
def base_config(seed, dataset, steps, ckpt_dir):
  """The V6 vanilla recipe (train_rockfall_clock_v6_baseline._apply_v6_config on
  verify_offline_d4rl.build_offline_cfg) with the discount at 0.999 and the
  in-training evaluation off (milestones no longer need it)."""
  from verify_offline_d4rl import build_offline_cfg
  import rockfall_clock_v6_teacher as CT
  from crl import rockfall_clock_v6 as V6
  cfg = build_offline_cfg(max_steps=int(steps), ckpt_dir=str(ckpt_dir))
  cfg.env_name = ENV_XY
  cfg.offline_dataset = str(dataset)
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  cfg.max_episode_steps = HORIZON
  cfg.rockfall_p_active_1 = P_ACTIVE
  cfg.rockfall_p_active_2 = P_ACTIVE
  cfg.rockfall_t0_min_1, cfg.rockfall_t0_max_1 = int(V6.T0_MIN_1), int(V6.T0_MAX_1)
  cfg.rockfall_t0_min_2, cfg.rockfall_t0_max_2 = int(V6.T0_MIN_2), int(V6.T0_MAX_2)
  cfg.seed = int(seed)
  cfg.fail_bank_path = ''
  cfg.fail_neg_alpha = 0.0
  cfg.use_layer_norm = False
  cfg.discount = DISCOUNT
  cfg.eval_every_steps = 10_000_000
  cfg.eval_episodes = 1
  cfg.log_every_steps = 5_000
  cfg.tensorboard = False
  cfg.ckpt_every_steps = int(steps)
  assert HORIZON == int(CT.HORIZON)
  return cfg


def row0_prepare(buffer, path):
  n = int(buffer.lengths.shape[0])
  buffer.set_anchor_strata([(np.arange(n), np.zeros(n, np.int64), None)], (buffer_batch(),))
  return {'anchor_rule': 'row 0 of every path, uniform over paths', 'paths': n}


def buffer_batch():
  return 1024


def manifest(run_dir, cfg, arm, extra=None):
  write_json(Path(run_dir) / 'branch_manifest.json', {
      'arm': arm, 'env_name': cfg.env_name, 'dataset': cfg.offline_dataset,
      'dataset_sha256': sha256(cfg.offline_dataset), 'discount': cfg.discount,
      'steps': cfg.max_number_of_steps, 'batch_size': cfg.batch_size, 'twin_q': cfg.twin_q,
      'bc_coef': cfg.bc_coef, 'bc_sampling': cfg.bc_sampling, 'bc_dataset': cfg.bc_dataset,
      'bc_balance_region': getattr(cfg, 'bc_balance_region', 'action'),
      'horizon': HORIZON, 'p_active': [cfg.rockfall_p_active_1, cfg.rockfall_p_active_2],
      't0_ranges': [[cfg.rockfall_t0_min_1, cfg.rockfall_t0_max_1], [cfg.rockfall_t0_min_2, cfg.rockfall_t0_max_2]],
      'note': 'discount 0.999 deviates from the frozen V6 recipe (0.99) on purpose: plan section 0',
      'dataset_stem': STEM, 'p_active_env': P_ACTIVE,
      **(extra or {})})


# ----------------------------------------------------------------- critics
def critic_dir(seed):
  # V6_CRITIC_TAG names a side-by-side critic set (e.g. '_ms' for the
  # milestone-probed retrain) without touching the chain's own critics
  return OUT / ('critics' + CRITIC_TAG) / f'seed_{seed}'


def train_critic(seed, steps):
  from crl.train import train
  d = critic_dir(seed)
  if (d / 'final.pkl').exists():
    print(f'{d} exists', flush=True)
    return
  cfg = base_config(seed, REPLAY, steps, d)
  cfg.ckpt_milestone_steps = tuple(m for m in MILESTONES if m < steps)   # the critic's own trajectory, for the probes
  t0 = time.time()
  train(cfg, buffer_prepare=row0_prepare)
  manifest(d, cfg, 'branch_critic', {'wall_seconds': time.time() - t0, 'milestones': list(cfg.ckpt_milestone_steps)})


def vanilla_dir(seed):
  return OUT / 'vanilla_g0999' / f'seed_{seed}'


def train_vanilla(seed, steps):
  from crl.train import train
  d = vanilla_dir(seed)
  if (d / 'final.pkl').exists():
    print(f'{d} exists', flush=True)
    return
  cfg = base_config(seed, DATASET, steps, d)
  t0 = time.time()
  train(cfg)
  manifest(d, cfg, 'vanilla_gamma0999', {'wall_seconds': time.time() - t0})


def _spawn(cmd, log):
  Path(log).parent.mkdir(parents=True, exist_ok=True)
  f = open(log, 'w', encoding='utf-8')
  f.write(' '.join(str(c) for c in cmd) + '\n')
  return subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                          env={**os.environ, 'PYTHONPATH': str(ROOT)}), f


def _wait(procs):
  for p, f, tag in procs:
    rc = p.wait()
    f.close()
    if rc != 0:
      raise RuntimeError(f'{tag} failed (rc {rc})')
  procs.clear()


def stage_train(kind, seeds, steps, parallel):
  procs = []
  for s in seeds:
    done = (critic_dir(s) if kind == 'critic' else vanilla_dir(s)) / 'final.pkl'
    if done.exists():
      continue
    cmd = [sys.executable, str(Path(__file__).resolve()), f'_{kind}', '--seeds', str(s), '--steps', str(steps)]
    procs.append((*_spawn(cmd, OUT / 'logs' / f'{kind}_seed{s}.log'), f'{kind} seed {s}'))
    if len(procs) >= parallel:
      _wait(procs)
  _wait(procs)


# -------------------------------------------------------------------- gate
def start_rows(t_max=START['t_max']):
  """Recorded start-region anchors with their recorded goals, split by the
  next-frame XY displacement direction (north / east)."""
  with np.load(DATASET, allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths']
  n, L = obs.shape[:2]
  t = np.arange(L)[None, :]
  valid = t < (lengths[:, None] - 1)
  m = valid & (obs[:, :, 0] < START['x_max']) & (obs[:, :, 1] < START['y_max']) & (t <= t_max)
  e, i = np.nonzero(m)
  disp = obs[e, i + 1, :2] - obs[e, i, :2]
  north = (disp[:, 1] > 0.05) & (disp[:, 1] > np.abs(disp[:, 0]))
  east = (disp[:, 0] > 0.05) & (disp[:, 0] > np.abs(disp[:, 1]))
  o = obs[e, i]                                  # 31 columns: state | goal_xy
  a = act[e, i]
  return {'north': (o[north], a[north]), 'east': (o[east], a[east]), 'n_rows': int(len(e))}


def critic_scores(q_params, nets, o, a):
  import jax
  import jax.numpy as jnp

  @jax.jit
  def f(qp, oo, aa):
    phi, psi = nets.representation_network.apply(qp, oo, aa)
    q = jnp.sum(phi * psi, axis=1)                 # [B, heads]
    return jnp.min(q, axis=1)                       # twin-Q: the actor's min
  out = []
  for k in range(0, len(o), 2048):
    out.append(np.asarray(f(q_params, jnp.asarray(o[k:k + 2048]), jnp.asarray(a[k:k + 2048]))))
  return np.concatenate(out)


def paired_margin(q_params, nets, rows, n_states=300, n_actions=200, seed=0):
  """Same-state comparison: each start-region anchor state (with its own goal)
  is scored with recorded north-moving torques and with recorded east-moving
  torques (transplanted; at t <= 5 every pose is the reset pose up to noise),
  and the per-state difference is averaged.  Removes the state / episode
  confound of the row-wise margin."""
  rng = np.random.default_rng(seed)
  o_n, a_n = rows['north']
  o_e, a_e = rows['east']
  states = np.concatenate([o_n, o_e])
  pick = rng.choice(len(states), size=min(n_states, len(states)), replace=False)
  an = a_n[rng.choice(len(a_n), size=min(n_actions, len(a_n)), replace=False)]
  ae = a_e[rng.choice(len(a_e), size=min(n_actions, len(a_e)), replace=False)]
  diffs = []
  for k in pick:
    o = np.repeat(states[k][None], len(an) + len(ae), 0)
    f = critic_scores(q_params, nets, o, np.concatenate([an, ae]))
    diffs.append(f[:len(an)].mean() - f[len(an):].mean())
  diffs = np.array(diffs)
  return float(diffs.mean()), float(diffs.std() / np.sqrt(len(diffs))), float((diffs > 0).mean())


def law_target():
  """The replay's own answer at the start region: the relabeling law's
  goal-frame probability of the north vs east query paths."""
  from build_v6_branch_replay import goal_frame_probability
  with np.load(REPLAY, allow_pickle=False) as d:
    obs, lengths, kind, intent, at = d['obs'], d['lengths'], d['audit_kind'], d['audit_intent'], d['audit_anchor_time']
  res = {}
  for radius in (0.5, 1.0):
    p = {}
    for name in ('detour', 'go'):
      m = (kind == 'query') & (intent == name) & (at <= START['t_max'])
      vals = [goal_frame_probability(obs[k, :lengths[k]], obs[k, 0, STATE_DIM:STATE_DIM + 2], DISCOUNT, radius)
              for k in np.flatnonzero(m)]
      p[name] = float(np.mean(vals)) if vals else float('nan')
    res[f'r{radius}'] = {'p_detour': p['detour'], 'p_go': p['go'],
                         'log_ratio': float(np.log(p['detour'] / max(p['go'], 1e-12)))}
  return res


def stage_gate(seeds, gate, vanilla_seeds=()):
  from crl import checkpoint, networks
  cfg = base_config(0, REPLAY, 1, OUT / '_cfg')
  from crl import envs as envs_mod
  envs_mod.make_env(cfg.env_name, cfg, seed=1)        # fills obs/goal/action dims
  nets = networks.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, repr_dim=int(cfg.repr_dim),
      repr_norm=cfg.repr_norm, repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  rows = start_rows()
  target = law_target() if REPLAY.exists() else None
  entries = [(f'branch critic seed {s}', critic_dir(s) / 'final.pkl', 'branch') for s in seeds]
  entries += [(f'vanilla g0.999 seed {s}', vanilla_dir(s) / 'final.pkl', 'vanilla') for s in vanilla_seeds]
  res = {'rows': {k: int(len(v[0])) for k, v in rows.items() if k != 'n_rows'}, 'law_target': target, 'critics': {}}
  L = ['# Gate: start-region margin north - east on recorded anchors (t <= 5)', '',
       f'{res["rows"]["north"]} north-moving and {res["rows"]["east"]} east-moving recorded rows (next-frame XY '
       'displacement), each scored with its own recorded goal; critic score = min over the twin heads.  '
       + (f'Law target on the replay\'s query paths (gamma {DISCOUNT}): r0.5 {target["r0.5"]["log_ratio"]:+.2f}, '
          f'r1.0 {target["r1.0"]["log_ratio"]:+.2f}.  ' if target else '')
       + f'PASS = paired same-state margin >= +{gate:.2f} (300 states x 200 north / 200 east recorded torques).', '',
       '| critic | mean f north | mean f east | row margin | s.e. | paired margin (same state) | s.e. | states > 0 | gate |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---|']
  for label, path, arm in entries:
    if not path.exists():
      continue
    _, st = checkpoint.load_checkpoint(path)
    fn = critic_scores(st.q_params, nets, *rows['north'])
    fe = critic_scores(st.q_params, nets, *rows['east'])
    margin = float(fn.mean() - fe.mean())
    se = float(np.sqrt(fn.var() / len(fn) + fe.var() / len(fe)))
    pm, pse, pfrac = paired_margin(st.q_params, nets, rows)
    m = {'arm': arm, 'f_north': float(fn.mean()), 'f_east': float(fe.mean()), 'margin': margin, 'se': se,
         'paired_margin': pm, 'paired_se': pse, 'paired_frac_positive': pfrac,
         'pass': bool(pm >= gate)}
    res['critics'][label] = m
    L.append(f'| {label} | {m["f_north"]:+.3f} | {m["f_east"]:+.3f} | {margin:+.3f} | {se:.3f} | '
             f'**{pm:+.3f}** | {pse:.3f} | {pfrac:.2f} | {"PASS" if m["pass"] else "fail"} |')
    print(L[-1], flush=True)
  write_json(OUT / f'gate{CRITIC_TAG}.json', res)
  (OUT / f'gate{CRITIC_TAG}.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


# ------------------------------------------------------------------- joint
def joint_dir(seed):
  return OUT / ('joint' + JOINT_TAG) / f'seed_{seed}'


def joint_config(seed, steps):
  cfg = base_config(seed, REPLAY, steps, joint_dir(seed))
  cfg.bc_sampling = 'balanced'
  cfg.bc_balance_region = 'displacement'
  cfg.bc_balance_cell = BC_CELL
  cfg.bc_balance_cap = BC_CAP
  cfg.bc_balance_wait_eps = 0.05
  cfg.bc_dataset = str(DATASET)
  cfg.resume = True
  cfg.ckpt_milestone_steps = tuple(m for m in MILESTONES if m < steps)
  if JOINT_MODE == 'frozen':
    cfg.learning_rate = 0.0    # the critic optimiser: a zero step keeps the warm critic exactly (actor + BC only)
  if os.environ.get('V6_BC_COEF'):
    cfg.bc_coef = float(os.environ['V6_BC_COEF'])   # the paper's lambda; the V6 recipe's 0.05 lets an 8-d actor leave the data
  return cfg


def prep_joint(seed):
  import jax
  import optax
  from crl import checkpoint, networks
  d = joint_dir(seed)
  if (d / 'latest.pkl').exists() or (d / 'final.pkl').exists():
    return
  cfg = base_config(seed, REPLAY, 1, d)
  from crl import envs as envs_mod
  envs_mod.make_env(cfg.env_name, cfg, seed=1)
  nets = networks.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, repr_dim=int(cfg.repr_dim),
      repr_norm=cfg.repr_norm, repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = checkpoint.load_checkpoint(critic_dir(seed) / 'final.pkl')
  key = jax.random.PRNGKey(30_000 + int(seed))
  k_pol, key = jax.random.split(key)
  policy_params = nets.policy_network.init(k_pol)
  pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7).init(policy_params)
  new = st._replace(policy_params=policy_params, policy_optimizer_state=pol_opt, key=key)
  d.mkdir(parents=True, exist_ok=True)
  checkpoint.save_named(str(d), 'latest', 0, new)
  write_json(d / 'prep.json', {'seed': seed, 'warm_critic': str(critic_dir(seed) / 'final.pkl'),
                               'warm_critic_sha256': sha256(critic_dir(seed) / 'final.pkl'), 'actor': 'fresh',
                               'joint_mode': JOINT_MODE, 'critic_tag': CRITIC_TAG, 'bc_coef': os.environ.get('V6_BC_COEF')})


def train_joint(seed, steps):
  from crl.train import train
  d = joint_dir(seed)
  if (d / 'final.pkl').exists():
    print(f'{d} exists', flush=True)
    return
  prep_joint(seed)
  cfg = joint_config(seed, steps)
  t0 = time.time()
  train(cfg, buffer_prepare=row0_prepare)
  manifest(d, cfg, 'branch_joint_warm', {'wall_seconds': time.time() - t0, 'milestones': list(cfg.ckpt_milestone_steps)})


# -------------------------------------------------------------------- eval
def evaluate_ckpt(ckpt, out_dir, policy):
  """The V6 evaluation (eval_rockfall_clock_v6_baseline.evaluate / summarize)
  on natural draws, without its manifest contract (which pins the frozen
  recipe's discount); this driver's manifest records the deviation."""
  import eval_rockfall_clock_v6_baseline as EV
  from crl import rockfall_clock_v6 as V6
  out = Path(out_dir) / f'eval_{policy}.json'
  if out.exists():
    return json.loads(out.read_text(encoding='utf-8'))['summary']
  args = EV.parse_args(['--ckpt', str(ckpt), '--n', str(EVAL['n']), '--seed', str(EVAL['seed']),
                        '--policy', policy, '--action-seed', str(EVAL['action_seed']),
                        '--p-active-1', str(P_ACTIVE), '--p-active-2', str(P_ACTIVE)])
  act, step, cfg = EV.build_mean_policy(str(ckpt), args)
  rows = EV.evaluate(act, args)
  summary = EV.summarize(rows, args.p_active_1, args.p_active_2)
  summary['policy_headline'] = 'deterministic_actor_mean' if policy == 'mean' else 'sampled_actor_tanh_normal'
  write_json(out, {'ckpt': str(ckpt), 'ckpt_step': int(step), 'policy': policy, 'n': EVAL['n'],
                   'eval_seed': EVAL['seed'], 'env': ENV_XY, 'environment_version': V6.ENV_VERSION,
                   'p_active': P_ACTIVE, 'dataset_stem': STEM,
                   'summary': summary, 'episodes': rows})
  return summary


def stage_joint(seeds, steps, parallel, all_critics):
  gate = json.loads((OUT / f'gate{CRITIC_TAG}.json').read_text(encoding='utf-8')) if (OUT / f'gate{CRITIC_TAG}.json').exists() else {'critics': {}}
  chosen = [s for s in seeds if all_critics or gate['critics'].get(f'branch critic seed {s}', {}).get('pass')]
  print(f'joint for seeds {chosen}', flush=True)
  procs = []
  for s in chosen:
    if (joint_dir(s) / 'final.pkl').exists():
      continue
    cmd = [sys.executable, str(Path(__file__).resolve()), '_joint', '--seeds', str(s), '--steps', str(steps)]
    procs.append((*_spawn(cmd, OUT / 'logs' / f'joint_seed{s}.log'), f'joint seed {s}'))
    if len(procs) >= parallel:
      _wait(procs)
  _wait(procs)
  for s in chosen:
    for pol in ('mean', 'sample'):
      evaluate_ckpt(joint_dir(s) / 'final.pkl', joint_dir(s), pol)


def stage_vanilla_eval(seeds):
  for s in seeds:
    if (vanilla_dir(s) / 'final.pkl').exists():
      for pol in ('mean', 'sample'):
        evaluate_ckpt(vanilla_dir(s) / 'final.pkl', vanilla_dir(s), pol)


# --------------------------------------------------------------- summarize
def _mouth_timing(episodes):
  """Slow-shortcut audit: the discounted score (gamma 0.99, the benchmark's
  own field) and, for shortcut episodes under an active latent, how many
  reach the zone's mouth only after its burst has closed (the rockfall
  clocks run from the reset, so a walker slower than the teacher passes a
  closed zone without ever having decided anything)."""
  from crl import rockfall_clock_v6 as V6
  out = {'discounted': (float(np.mean([r.get('discounted', 0.0) for r in episodes])) if episodes else None)}
  for z in (1, 2):
    sc = [r for r in episodes if r.get('route') == 'shortcut' and r.get(f'mouth_step_{z}') is not None]
    act = [r for r in sc if r.get(f'rockfall_start_{z}') is not None]
    late = [r for r in act if r[f'mouth_step_{z}'] > r[f'rockfall_start_{z}'] + V6.ROCKFALL_STEPS]
    out[f'mouth{z}_median'] = float(np.median([r[f'mouth_step_{z}'] for r in sc])) if sc else None
    out[f'mouth{z}_after_burst'] = (len(late) / len(act)) if act else None
  return out


def _headline(summary, episodes=None):
  o = summary.get('overall', {})
  routes = o.get('routes', {})
  by = summary.get('by_latent', {})
  return {'success_rate': o.get('success'), 'failure_rate': o.get('failure'), 'timeout_rate': o.get('timeout'),
          'detour_rate': routes.get('detour', {}).get('rate'), 'shortcut_rate': routes.get('shortcut', {}).get('rate'),
          'by_latent': {k: (by.get(k, {}).get('success') if isinstance(by.get(k), dict) else None)
                        for k in ('U00', 'U10', 'U01', 'U11')},
          **(_mouth_timing(episodes) if episodes is not None else {})}


def stage_summarize(seeds, vanilla_seeds):
  from crl import rockfall_clock_v6 as V6
  gate = json.loads((OUT / f'gate{CRITIC_TAG}.json').read_text(encoding='utf-8')) if (OUT / f'gate{CRITIC_TAG}.json').exists() else {}
  L = ['# AntMaze V6 Phase 1: oracle branch replay', '',
       f'Critic recipe: V6 vanilla at gamma {DISCOUNT}, {CRITIC_STEPS:,} updates, anchors = row 0 of every branch '
       f'path; joint stage: {JOINT_STEPS:,} warm-started joint updates with balanced BC rows (displacement key, '
       f'cell {BC_CELL}, cap {BC_CAP}) from the recorded data; evaluation {EVAL["n"]} natural draws, seed {EVAL["seed"]}.', '']
  if gate.get('law_target'):
    L.append(f'Law target on the replay (start region, t <= 5): r0.5 {gate["law_target"]["r0.5"]["log_ratio"]:+.2f}, '
             f'r1.0 {gate["law_target"]["r1.0"]["log_ratio"]:+.2f}.\n')
  L += ['| arm | seed | gate margin | policy | success | failure | timeout | detour | shortcut | success U00/U10/U01/U11 '
        '| discounted (g 0.99) | mouth 1 / 2 median step | after-burst share z1 / z2 |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|---|']
  res = []
  for arm, dirs, label in (('branch joint', {s: joint_dir(s) for s in seeds}, 'branch critic seed {}'),
                           ('vanilla g0.999', {s: vanilla_dir(s) for s in vanilla_seeds}, 'vanilla g0.999 seed {}')):
    for s, d in dirs.items():
      g = gate.get('critics', {}).get(label.format(s), {})
      for pol in ('mean', 'sample'):
        p = d / f'eval_{pol}.json'
        if not p.exists():
          continue
        ev = json.loads(p.read_text(encoding='utf-8'))
        sm = ev['summary']
        h = _headline(sm, ev.get('episodes'))
        res.append({'arm': arm, 'seed': s, 'policy': pol, **h, 'gate_margin': g.get('margin')})
        fmt = lambda v: f'{v:.3f}' if isinstance(v, (int, float)) else str(v)
        L.append(f'| {arm} | {s} | {g.get("paired_margin", g.get("margin", float("nan"))):+.3f} | {pol} | ' + ' | '.join(
            fmt(h[k]) for k in ('success_rate', 'failure_rate', 'timeout_rate', 'detour_rate', 'shortcut_rate'))
            + ' | ' + ' / '.join(fmt(h['by_latent'][u]) for u in ('U00', 'U10', 'U01', 'U11'))
            + f' | {fmt(h.get("discounted"))} | {fmt(h.get("mouth1_median"))} / {fmt(h.get("mouth2_median"))}'
            + f' | {fmt(h.get("mouth1_after_burst"))} / {fmt(h.get("mouth2_after_burst"))} |')
  L.append('\nReference: vanilla at gamma 0.99 on the p040 benchmark (notes/v6_detour_ladder.md): success 0.370, detour '
           '0.000 on every seed.  The rockfall clocks run from the reset (zone 1 closes by step '
           f'{V6.T0_MAX_1 + V6.ROCKFALL_STEPS}, zone 2 by {V6.T0_MAX_2 + V6.ROCKFALL_STEPS}); an after-burst share near one '
           'means the shortcut survives by lateness, not by a route decision.')
  (OUT / f'REPORT{CRITIC_TAG}{JOINT_TAG}.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  write_json(OUT / f'results{CRITIC_TAG}{JOINT_TAG}.json', {'rows': res, 'gate': gate})
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('critics', 'gate', 'joint', 'vanilla', 'vanilla_eval', 'summarize', 'all',
                                      '_critic', '_vanilla', '_joint'))
  ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
  ap.add_argument('--vanilla-seeds', type=int, nargs='*', default=list(VANILLA_SEEDS))
  ap.add_argument('--steps', type=int, default=CRITIC_STEPS)
  ap.add_argument('--joint-steps', type=int, default=JOINT_STEPS)
  ap.add_argument('--parallel', type=int, default=2)
  ap.add_argument('--gate', type=float, default=0.15)
  ap.add_argument('--all-critics', action='store_true')
  ap.add_argument('--eval-n', type=int, default=EVAL['n'], help='evaluation episodes (smoke only)')
  args = ap.parse_args(argv)
  EVAL['n'] = int(args.eval_n)
  OUT.mkdir(parents=True, exist_ok=True)
  if args.command == '_critic':
    train_critic(args.seeds[0], args.steps); return 0
  if args.command == '_vanilla':
    train_vanilla(args.seeds[0], args.steps); return 0
  if args.command == '_joint':
    train_joint(args.seeds[0], args.steps); return 0
  stages = ('critics', 'gate', 'joint', 'vanilla', 'vanilla_eval', 'summarize') if args.command == 'all' else (args.command,)
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'critics':
      stage_train('critic', args.seeds, args.steps, args.parallel)
    elif st == 'gate':
      stage_gate(args.seeds, args.gate, args.vanilla_seeds)
    elif st == 'joint':
      stage_joint(args.seeds, args.joint_steps, args.parallel, args.all_critics)
    elif st == 'vanilla':
      stage_train('vanilla', args.vanilla_seeds, args.steps, args.parallel)
    elif st == 'vanilla_eval':
      stage_vanilla_eval(args.vanilla_seeds)
    elif st == 'summarize':
      stage_summarize(args.seeds, args.vanilla_seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
