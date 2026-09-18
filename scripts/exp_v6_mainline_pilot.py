"""AntMaze V6 mainline pilot: observational (O) vs counterfactual-oracle (CF)
positive futures under an otherwise matched offline CRL procedure.

Contract: notes/MAINLINE_CONTRACT.md (read it first; this script is its
executable form).  One controlled comparison, one round, no sweeps.

  arm O          critic anchors = K logged (s_t, a_t) rows of the d05 dataset
                 drawn by the offline law; positive futures = that anchor's
                 RECORDED continuation (geometric, truncated at the episode end)
  arm CF-oracle  the SAME anchors and weights; the logged torque is executed
                 once in the simulator, then the fixed d05 agent (mode) acts
                 closed-loop to the first reach frame / death / the horizon;
                 positive futures = that GENERATED continuation, same law
  both           joint critic + actor training, 30,000 optimizer updates,
                 actor batches from the dataset under the buffer's own law
                 (random_goals 0, BC 0.05 on the same rows), actor initialised
                 from the same d05 agent, paired fresh critics, gamma 0.999

The three sampling interfaces are explicit objects here, not a property of a
replay file's layout: ``AnchorSet`` (critic anchors + weights), a future
source per arm (``RecordedFutures`` / ``BranchFutures``) read by
``CriticStream``, and ``ActorStream`` (the dataset buffer).  The learner is
``crl.losses.build_learner(separate_actor_batch=True)``: the loss bodies are
untouched, the critic loss sees the critic rows and the actor loss sees the
actor rows.

Modes
  anchors    declare the anchor set -> anchors.npz (deterministic in the seed)
  generate   arm CF branches -> branches_cf.npz (multiprocessing; needs the
             start checkpoint)
  seal       manifest.json -- before generation and training
  audit      the pre-training checks -> audit.json / AUDIT.md
  train      --arm O|CF --seed s -> <arm>/seed_s/{init,10000,20000,final}.pkl
  evaluate   the start agent, O x3, CF x3 on the same 300 natural draws
  report     REPORT.md
  smoke      end-to-end code-path test at toy scale with a stand-in
             continuation (no checkpoint); writes only under _smoke/

Environment: the pilot pins its own V6_* variables (d05, p_active 0.5,
gamma 0.999); do not override them.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

# the pilot's declared benchmark rung; forced so that a stray shell export cannot move it
STEM = 'antmaze_rockfall_clock_v6_p050_d05'
DATASET = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset' / f'{STEM}_gxy.npz'
SIDECAR = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset' / f'{STEM}_sidecar.npz'
P050_OUT = ROOT / 'outputs' / 'antmaze_branch_replay_p050'
os.environ['V6_DATASET_STEM'] = STEM
os.environ['V6_P_ACTIVE'] = '0.5'
os.environ['V6_DISCOUNT'] = '0.999'
os.environ['V6_BRANCH_OUT'] = str(P050_OUT)
os.environ['V6_BRANCH_REPLAY'] = str(DATASET)      # only names the file the recipe config points at
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import run_v6_branch_replay as D  # noqa: E402  (the V6 recipe config; does not touch JAX_PLATFORMS)

EXP = 'exp_mainline_pilot'
OUT = P050_OUT / EXP
CONTRACT = ROOT / 'notes' / 'MAINLINE_CONTRACT.md'
STATE_DIM, GOAL_DIM, ACTION_DIM, OBS_W = 29, 2, 8, 31
HORIZON = 800
GAMMA = 0.999
SEEDS = (0, 1, 2)
ARMS = ('O', 'CF')
UPDATES = 30_000          # optimizer updates per arm and seed, counted in the loop
G = 4                     # updates per jitted scan (the recipe's num_sgd_steps_per_step)
BATCH = 1024
K_DRAWS = 60_000          # anchor draws from the offline law (unique anchors carry their multiplicity)
ANCHOR_SEED = 131_000_001
HAZARD_SEED0 = 132_000_000        # arm CF hazard / clock / jitter draws: HAZARD_SEED0 + anchor_id
GEN_WORKER_SEED0 = 133_000_000
CRITIC_STREAM_SEED0 = 10_000      # + training seed
ACTOR_STREAM_SEED0 = 20_000       # + training seed
EVAL = {'n': 300, 'seed': 3909, 'policy': 'mean'}
RESERVED_SEEDS = {'FRESH_TORQUE_SEED': 616_000_005, 'FRESH_DRAW_SEED': 616_500_000}   # untouched
D.EVAL['seed'] = EVAL['seed']
D.EVAL['n'] = EVAL['n']
# the current agent: the d05 vanilla actor of exp_detour_ratio (table row "d05 vanilla seed 0");
# continuation policy (mode) for arm CF and the actor initialisation of both arms
START_CKPT = P050_OUT / 'joint_van_d05' / 'seed_0' / 'final.pkl'
START_PROVENANCE_FILES = ('joint_van_d05/seed_0/prep.json', 'joint_van_d05/seed_0/branch_manifest.json',
                          'joint_van_d05/seed_0/bc_sampling_audit.json', 'critics_van_d05/seed_0/branch_manifest.json',
                          'joint_purebc_d05/seed_0/prep.json', 'joint_purebc_d05/seed_0/branch_manifest.json')
D05_EXPECTED = {'gxy_sha256': '4533c70278a96e09b9d79ba8ed6c2144792abbab5085d405ae544f50982144d6',
                'sidecar_sha256': '8b17a572356216ec3484d33525faafed6103ef666b6b01d14e6962dc9cd40afe'}
MILESTONES = (10_000, 20_000)
LOG_EVERY = 500


# ------------------------------------------------------------------ helpers
def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def arr_hash(*arrays):
  h = hashlib.sha256()
  for a in arrays:
    h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
  return h.hexdigest()[:16]


def write_json(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(value, indent=1, default=_json_default), encoding='utf-8')


def _json_default(v):
  if isinstance(v, (np.integer,)):
    return int(v)
  if isinstance(v, (np.floating,)):
    return float(v)
  if isinstance(v, np.ndarray):
    return v.tolist()
  if isinstance(v, Path):
    return str(v)
  return str(v)


def read_json(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def load_dataset():
  with np.load(DATASET, allow_pickle=False) as d:
    obs, act, lengths, meta = d['obs'], d['act'], d['lengths'].astype(np.int64), json.loads(str(d['meta']))
  return obs, act, lengths, meta


def git_head():
  try:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
  except Exception as ex:  # pylint: disable=broad-except
    return f'unavailable ({ex})'


# ---------------------------------------------------------------- anchors
@dataclasses.dataclass
class AnchorSet:
  """The critic anchors: logged rows with their identity and sampling weight.

  Law (declared): K i.i.d. draws from the offline anchor law of the recipe's
  buffer -- an episode uniformly, then a row uniformly over that episode's
  valid anchor rows [0, L_e - 1) -- with a pinned seed; a row drawn several
  times is one anchor whose weight is its multiplicity / K.  Both arms read
  the same file, so anchor identities and effective weights are equal by
  construction, and the number of continuations generated for an anchor
  (one, in this pilot) never enters its weight."""
  episode: np.ndarray        # [K] index into the d05 file
  t: np.ndarray              # [K] anchor timestep (row) in that episode
  state: np.ndarray          # [K, 29]
  goal_xy: np.ndarray        # [K, 2]  the episode's task goal
  action: np.ndarray         # [K, 8]  the logged torque
  multiplicity: np.ndarray   # [K] draws of the law that landed on this row
  weight: np.ndarray         # [K] multiplicity / K_DRAWS
  source_episode: np.ndarray  # [K] merged-pool episode id (sidecar)
  ep_length: np.ndarray      # [K] valid rows of the episode

  @property
  def n(self):
    return int(len(self.t))

  @property
  def obs31(self):
    return np.concatenate([self.state, self.goal_xy], axis=1)

  def save(self, path, meta):
    np.savez_compressed(path, episode=self.episode, t=self.t, state=self.state, goal_xy=self.goal_xy, action=self.action,
                        multiplicity=self.multiplicity, weight=self.weight, source_episode=self.source_episode,
                        ep_length=self.ep_length, meta=np.asarray(json.dumps(meta, sort_keys=True)))

  @classmethod
  def load(cls, path):
    with np.load(path, allow_pickle=False) as d:
      a = cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})
      meta = json.loads(str(d['meta']))
    return a, meta


def declare_anchors(seed=ANCHOR_SEED, k_draws=K_DRAWS):
  obs, act, lengths, _ = load_dataset()
  with np.load(SIDECAR, allow_pickle=True) as sc:
    source = sc['source_episode'].astype(np.int64)
  rng = np.random.default_rng(seed)
  n_eps, L = obs.shape[:2]
  e = rng.integers(0, n_eps, size=k_draws)                                   # episode uniform
  t = np.floor(rng.random(k_draws) * (lengths[e] - 1)).astype(np.int64)        # row uniform in [0, L_e - 1)
  key = e * L + t
  uniq, counts = np.unique(key, return_counts=True)
  ee, tt = uniq // L, uniq % L
  assert np.all(tt <= lengths[ee] - 2)
  anchors = AnchorSet(episode=ee, t=tt, state=obs[ee, tt, :STATE_DIM].astype(np.float32),
                      goal_xy=obs[ee, tt, STATE_DIM:OBS_W].astype(np.float32), action=act[ee, tt].astype(np.float32),
                      multiplicity=counts.astype(np.int64), weight=(counts / k_draws).astype(np.float64),
                      source_episode=source[ee], ep_length=lengths[ee])
  meta = {'law': 'K i.i.d. draws: episode uniform over the d05 file, then row uniform over [0, L_e - 1); '
                 'unique rows with weight = multiplicity / K (the recipe buffer\'s own anchor law, subsampled)',
          'seed': int(seed), 'k_draws': int(k_draws), 'n_unique': anchors.n, 'dataset': str(DATASET),
          'dataset_sha256': sha256(DATASET), 'n_t0': int((tt == 0).sum()), 'max_multiplicity': int(counts.max())}
  return anchors, meta


def mode_anchors(args):
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, meta = declare_anchors()
  anchors.save(OUT / 'anchors.npz', meta)
  print(json.dumps(meta, indent=1), flush=True)


# ---------------------------------------------------------- future sources
class RecordedFutures:
  """Arm O: the anchor's recorded continuation, rows t..L_e - 1 of its episode."""
  name = 'recorded'

  def __init__(self, anchors, obs, act):
    self.a, self.obs, self.act = anchors, obs, act
    self.lengths = (anchors.ep_length - anchors.t).astype(np.int64)      # rows including the root

  def goal_at(self, k, m):
    return self.obs[self.a.episode[k], self.a.t[k] + m, :GOAL_DIM]

  def next_rows(self, k):
    e, t = self.a.episode[k], self.a.t[k]
    return self.obs[e, t + 1, :STATE_DIM], self.act[e, t + 1]

  def path(self, k):
    e, t, L = int(self.a.episode[k]), int(self.a.t[k]), int(self.a.ep_length[k])
    return self.obs[e, t:L, :OBS_W], self.act[e, t:L]


class BranchFutures:
  """Arm CF: the generated continuation (compact rows; root = the logged row)."""
  name = 'branch'

  def __init__(self, anchors, path):
    with np.load(path, allow_pickle=False) as d:
      self.obs_rows, self.act_rows = d['obs_rows'], d['act_rows']
      self.offset, self.lengths = d['offset'].astype(np.int64), d['length'].astype(np.int64)
      self.anchor_episode, self.anchor_t = d['episode'], d['t']
      self.outcome = d['outcome'].astype(str)
      self.meta = json.loads(str(d['meta']))
    self.a = anchors
    assert len(self.lengths) == anchors.n, 'branch file does not cover the anchor set'
    assert np.array_equal(self.anchor_episode, anchors.episode) and np.array_equal(self.anchor_t, anchors.t)
    roots = self.obs_rows[self.offset]
    assert np.array_equal(roots, anchors.obs31), 'branch roots differ from the logged anchor rows'
    assert np.array_equal(self.act_rows[self.offset], anchors.action), 'branch first actions differ from the logged torques'

  def goal_at(self, k, m):
    return self.obs_rows[self.offset[k] + m, :GOAL_DIM]

  def next_rows(self, k):
    r = self.offset[k] + 1
    return self.obs_rows[r, :STATE_DIM], self.act_rows[r]

  def path(self, k):
    o, n = int(self.offset[k]), int(self.lengths[k])
    return self.obs_rows[o:o + n], self.act_rows[o:o + n]


class CriticStream:
  """Critic batches: anchor by weight, future row by the geometric law.

  P(m) proportional to gamma^m over m = 1..len_k - 1 (the rows after the root
  up to the path's last frame), i.e. the buffer's relabeling law truncated at
  the episode's / branch's end.  The RNG consumption per batch is one uniform
  per anchor draw and one per future draw in both arms, so with equal seeds
  the ANCHOR sequence is identical across arms; only the goals differ."""

  def __init__(self, anchors, futures, batch, gamma, seed):
    self.a, self.f, self.B, self.gamma = anchors, futures, int(batch), float(gamma)
    self.cdf = np.cumsum(anchors.weight / anchors.weight.sum()); self.cdf[-1] = 1.0
    self.rng = np.random.default_rng(seed)
    self.obs31 = anchors.obs31
    self.log_gamma = np.log(gamma)
    self.n_fut = futures.lengths - 1
    assert np.all(self.n_fut >= 1)

  def draw(self):
    k = np.minimum(np.searchsorted(self.cdf, self.rng.random(self.B), side='right'), self.a.n - 1)
    u = self.rng.random(self.B)
    n = self.n_fut[k]
    m = np.ceil(np.log1p(-u * (1.0 - self.gamma ** n)) / self.log_gamma).astype(np.int64)
    m = np.clip(m, 1, n)
    return k, m

  def sample(self):
    from crl.losses import Transition
    k, m = self.draw()
    goals = np.stack([self.f.goal_at(int(kk), int(mm)) for kk, mm in zip(k, m)]).astype(np.float32)
    nxt = [self.f.next_rows(int(kk)) for kk in k]
    next_state = np.stack([x[0] for x in nxt]).astype(np.float32)
    next_action = np.stack([x[1] for x in nxt]).astype(np.float32)
    obs = np.concatenate([self.a.state[k], goals], axis=1)
    return Transition(observation=obs, action=self.a.action[k], reward=np.zeros(self.B, np.float32),
                      discount=np.full(self.B, self.gamma, np.float32),
                      next_observation=np.concatenate([next_state, goals], axis=1), next_action=next_action), (k, m)


class ActorStream:
  """Actor batches: the recipe's own TrajectoryBuffer over the d05 dataset
  (episode uniform, row uniform over the valid rows, geometric future goal
  truncated at the episode end); random_goals 0 and the BC term on the same
  rows happen inside the actor loss."""

  def __init__(self, cfg, seed, dataset=DATASET):
    import copy
    from crl import offline_audit
    c = copy.copy(cfg)
    c.seed = int(seed)
    self.buffer, self.fingerprint = offline_audit.build_offline_buffer(str(dataset), c, prepare=None)
    self.seed = int(seed)

  def sample(self, batch):
    return self.buffer.sample(int(batch))


# ----------------------------------------------------------------- config
def recipe_config(seed, run_dir, steps=UPDATES):
  """The V6 recipe (run_v6_branch_replay.base_config: twin-Q binary NCE,
  batch 1024, 1024x1024, repr 16, bc 0.05, random_goals 0, lr 3e-4, alpha 0)
  at gamma 0.999 on the d05 dataset; bc_sampling 'shared' (the standard
  actor pairing)."""
  cfg = D.base_config(seed, DATASET, steps, run_dir)
  assert cfg.discount == GAMMA and cfg.bc_coef == 0.05 and cfg.random_goals == 0.0 and cfg.twin_q
  assert (getattr(cfg, 'bc_sampling', 'shared') or 'shared') == 'shared'
  cfg.batch_size = BATCH
  return cfg


def config_dump(cfg):
  keep = ('env_name', 'offline_dataset', 'discount', 'batch_size', 'hidden_layer_sizes', 'repr_dim', 'repr_norm',
          'twin_q', 'bc_coef', 'random_goals', 'entropy_coefficient', 'actor_learning_rate', 'learning_rate', 'tau',
          'use_td', 'use_cpc', 'use_gcbc', 'use_layer_norm', 'bc_sampling', 'num_sgd_steps_per_step', 'goal_indices',
          'obs_dim', 'goal_dim', 'action_dim', 'max_episode_steps', 'rockfall_p_active_1', 'rockfall_p_active_2')
  return {k: getattr(cfg, k, None) for k in keep}


def make_nets(cfg):
  from crl import networks as networks_mod
  from crl.obs_norm import obs_scale_vector
  scale = obs_scale_vector(cfg.obs_dim, cfg.goal_dim, getattr(cfg, 'obs_norm_mode', '') or '', getattr(cfg, 'obs_norm_z_scale', 0.0) or None)
  return networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, repr_dim=int(cfg.repr_dim),
      repr_norm=cfg.repr_norm, repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm, obs_scale=scale,
      log_prob_mode=getattr(cfg, 'log_prob_mode', 'clip') or 'clip')


def fill_dims(cfg):
  from crl import envs as envs_mod
  envs_mod.make_env(cfg.env_name, cfg, seed=1)     # obs / goal / action dims and goal_indices, as crl.train does
  assert (cfg.obs_dim, cfg.goal_dim, cfg.action_dim) == (STATE_DIM, GOAL_DIM, ACTION_DIM)


# --------------------------------------------------------------- generate
def mode_policy(ckpt, cfg=None):
  """The continuation policy: the checkpoint actor's MODE, tanh(loc), at a
  31-column observation (the evaluation convention)."""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  if cfg is None:
    cfg = recipe_config(0, OUT / '_cfg')
    fill_dims(cfg)
  nets = make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt)
  pp = st.policy_params

  @jax.jit
  def _mode(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)
  return lambda o: np.asarray(_mode(jnp.asarray(o[None, :OBS_W], jnp.float32)))[0]


def _branch_one(env, act_fn, obs_row, a_t, t, hazard_seed, max_steps):
  """One arm-CF branch: restore the logged state at absolute time t with
  fresh hidden draws, execute the logged torque ONCE, then act_fn closed-loop
  until the first reach frame, death, or the absolute horizon.  No goal hold
  (the recorded episodes end on their reach frame; so does the branch)."""
  import build_v6_branch_replay as B
  import diag_v6_first_step_crossover as DG
  state = obs_row[:STATE_DIM].astype(np.float64)
  goal_xy = obs_row[STATE_DIM:OBS_W].astype(np.float64)
  DG.reseed(env, int(hazard_seed))
  o = B.restore(env, state, goal_xy, int(t))
  restore_maxdiff = float(np.abs(o[:OBS_W] - obs_row).max())
  rows_o, rows_a = [obs_row.astype(np.float32)], [np.asarray(a_t, np.float32)]
  o, reward, done, info = env.step(np.asarray(a_t, np.float32))           # the logged torque, exactly once
  rows_o.append(o[:OBS_W].astype(np.float32))
  n_query, step, reached = 1, 1, bool(reward > 0)
  while not done and not reached and step < max_steps:
    a = np.asarray(act_fn(o), np.float32)
    rows_a.append(a)
    o, reward, done, info = env.step(a)
    rows_o.append(o[:OBS_W].astype(np.float32))
    step += 1
    reached = bool(reward > 0)
  rows_a.append(np.zeros(ACTION_DIM, np.float32))                          # dummy action on the last row
  outcome = 'success' if reached else ('death' if bool(info.get('failure', False)) or done else 'timeout')
  return {'obs': np.stack(rows_o), 'act': np.stack(rows_a), 'steps': int(step), 'n_query_steps': n_query,
          'outcome': outcome, 'u1': bool(env.privileged_rockfall_active_1), 'u2': bool(env.privileged_rockfall_active_2),
          'restore_maxdiff': restore_maxdiff}


def _gen_worker(args):
  jobs, worker_seed, ckpt, max_steps_cap = args
  import build_v6_branch_replay as B
  env, _ = B._worker_env(worker_seed)
  act_fn = mode_policy(ckpt) if ckpt else (lambda o: np.zeros(ACTION_DIM, np.float32))   # stand-in only in smoke
  obs, act, _, _ = load_dataset()
  out = []
  for (aid, e, t, hz) in jobs:
    ms = HORIZON - int(t) if max_steps_cap is None else min(HORIZON - int(t), int(max_steps_cap))
    r = _branch_one(env, act_fn, obs[e, t, :OBS_W], act[e, t], t, hz, ms)
    r.update({'anchor_id': int(aid), 'episode': int(e), 't': int(t), 'hazard_seed': int(hz)})
    out.append(r)
  return out


def generate_branches(anchors, ckpt, workers, out_path, max_steps_cap=None, limit=None):
  from multiprocessing import get_context
  ids = np.arange(anchors.n) if limit is None else np.arange(min(limit, anchors.n))
  jobs = [(int(k), int(anchors.episode[k]), int(anchors.t[k]), HAZARD_SEED0 + int(k)) for k in ids]
  parts = [jobs[i::max(1, workers * 4)] for i in range(max(1, workers * 4))]
  parts = [p for p in parts if p]
  args = [(p, GEN_WORKER_SEED0 + i, (str(ckpt) if ckpt else ''), max_steps_cap) for i, p in enumerate(parts)]
  t0 = time.time()
  if workers <= 1:
    res = [_gen_worker(a) for a in args]
  else:
    with get_context('spawn').Pool(workers) as pool:
      res = pool.map(_gen_worker, args)
  res = sorted([r for part in res for r in part], key=lambda r: r['anchor_id'])
  wall = time.time() - t0
  lengths = np.array([len(r['obs']) for r in res], np.int64)
  offset = np.concatenate([[0], np.cumsum(lengths)[:-1]])
  outcomes = np.array([r['outcome'] for r in res])
  meta = {'arm': 'CF-oracle', 'continuation_ckpt': (str(ckpt) if ckpt else 'STAND-IN zero torque (smoke only)'),
          'continuation_ckpt_sha256': (sha256(ckpt) if ckpt else None), 'continuation': 'checkpoint actor mode tanh(loc), closed-loop',
          'query': 'the logged torque a_t executed exactly once from the restored logged state at absolute time t',
          'termination': 'first reach frame (reward > 0) / death (env done) / absolute horizon 800; no goal hold',
          'hidden_draws': 'hazard latents, clocks and rock jitter redrawn from the priors at restore, seed HAZARD_SEED0 + anchor_id',
          'restore': 'build_v6_branch_replay.restore (reset, then qpos/qvel from the logged state, goal from the logged row, env time = t)',
          'draws_per_anchor': 1, 'filtering': 'none: every outcome kept', 'anchors': anchors.n, 'wall_seconds': wall,
          'max_steps_cap': max_steps_cap, 'outcome_counts': {k: int((outcomes == k).sum()) for k in np.unique(outcomes)}}
  np.savez_compressed(out_path, obs_rows=np.concatenate([r['obs'] for r in res]).astype(np.float32),
                      act_rows=np.concatenate([r['act'] for r in res]).astype(np.float32),
                      offset=offset, length=lengths, anchor_id=np.array([r['anchor_id'] for r in res], np.int64),
                      episode=np.array([r['episode'] for r in res], np.int64), t=np.array([r['t'] for r in res], np.int64),
                      outcome=outcomes, steps=np.array([r['steps'] for r in res], np.int64),
                      n_query_steps=np.array([r['n_query_steps'] for r in res], np.int64),
                      u1=np.array([r['u1'] for r in res]), u2=np.array([r['u2'] for r in res]),
                      restore_maxdiff=np.array([r['restore_maxdiff'] for r in res], np.float32),
                      hazard_seed=np.array([r['hazard_seed'] for r in res], np.int64),
                      meta=np.asarray(json.dumps(meta, sort_keys=True)))
  return meta, res


def generation_summary(branch_path, anchors):
  with np.load(branch_path, allow_pickle=False) as d:
    outcome, steps, t, length = d['outcome'].astype(str), d['steps'], d['t'], d['length']
    u1, u2, rmd, nq = d['u1'], d['u2'], d['restore_maxdiff'], d['n_query_steps']
    meta = json.loads(str(d['meta']))
  bins = [(0, 1), (1, 6), (6, 50), (50, 150), (150, 400), (400, 801)]
  by_t = {}
  for lo, hi in bins:
    m = (t >= lo) & (t < hi)
    if m.any():
      by_t[f't in [{lo}, {hi})'] = {'n': int(m.sum()), 'success': float((outcome[m] == 'success').mean()),
                                    'death': float((outcome[m] == 'death').mean()), 'timeout': float((outcome[m] == 'timeout').mean()),
                                    'mean_rows': float(length[m].mean())}
  return {'meta': meta, 'n': int(len(outcome)), 'rows_total': int(length.sum()),
          'outcome_share': {k: float((outcome == k).mean()) for k in ('success', 'death', 'timeout')},
          'mean_steps': float(steps.mean()), 'mean_rows': float(length.mean()),
          'hazard_share_u1': float(u1.mean()), 'hazard_share_u2': float(u2.mean()),
          'restore_maxdiff_max': float(rmd.max()), 'n_query_steps_all_one': bool(np.all(nq == 1)), 'by_anchor_time': by_t}


def mode_generate(args):
  anchors, _ = AnchorSet.load(OUT / 'anchors.npz')
  if not START_CKPT.exists():
    raise SystemExit(f'BLOCKED: the continuation checkpoint {START_CKPT} is not available on this machine')
  out_path = OUT / 'branches_cf.npz'
  if out_path.exists() and not args.force:
    print(f'{out_path} exists', flush=True)
    return
  meta, _ = generate_branches(anchors, START_CKPT, args.workers, out_path, limit=args.limit)
  write_json(OUT / 'generation_cf.json', generation_summary(out_path, anchors))
  print(json.dumps(meta, indent=1), flush=True)


# -------------------------------------------------------------------- seal
def start_agent_record():
  rec = {'path': str(START_CKPT), 'available': START_CKPT.exists(),
         'sha256': (sha256(START_CKPT) if START_CKPT.exists() else 'UNAVAILABLE'),
         'role': 'continuation policy (mode) of arm CF and the actor initialisation of both arms',
         'identity': 'exp_detour_ratio d05 vanilla actor, training seed 0 (SUMMARY table row "d05 vanilla seed 0")',
         'provenance_chain': {}}
  for rel in START_PROVENANCE_FILES:
    p = P050_OUT / rel
    rec['provenance_chain'][rel] = read_json(p) if p.exists() else 'missing'
  chain = rec['provenance_chain']

  def _sha(rel):
    v = chain.get(rel)
    return v.get('dataset_sha256') if isinstance(v, dict) else None
  d05 = D05_EXPECTED['gxy_sha256']
  rec['data_provenance_check'] = {
      'joint_van_d05 critic-term data sha == d05': _sha('joint_van_d05/seed_0/branch_manifest.json') == d05,
      'joint_van_d05 bc_dataset is d05': str((chain.get('joint_van_d05/seed_0/branch_manifest.json') or {}).get('bc_dataset', '')).endswith(f'{STEM}_gxy.npz'),
      'critics_van_d05 data sha == d05': _sha('critics_van_d05/seed_0/branch_manifest.json') == d05,
      'joint_purebc_d05 data sha == d05': _sha('joint_purebc_d05/seed_0/branch_manifest.json') == d05,
      'joint_purebc_d05 bc_dataset is d05': str((chain.get('joint_purebc_d05/seed_0/branch_manifest.json') or {}).get('bc_dataset', '')).endswith(f'{STEM}_gxy.npz'),
      'observation contract': '29-dim Ant state + goal xy (31 columns); the env privileged channel never enters _flatten',
      'no d20 reference in the chain': not any('d20' in json.dumps(v) for v in chain.values())}
  rec['historical_training_note'] = ('the start agent itself was produced by the exp_detour_ratio chain (pure-BC 100k init, frozen '
                                     'vanilla critic 30k, displacement-balanced BC rows, 30k): diagnostic components of ITS history, '
                                     'not of this pilot\'s training procedure (contract section 1c)')
  rec['alternative_not_used'] = ('vanilla_g0999_d05/seed_0/final.pkl (the jointly trained vanilla run\'s own actor, never evaluated on '
                                 'natural draws) -- not used: the report row "d05 vanilla seed 0" is the actor above')
  return rec


def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, ameta = AnchorSet.load(OUT / 'anchors.npz')
  obs, act, lengths, dmeta = load_dataset()
  cfg = recipe_config(0, OUT / '_cfg')
  fill_dims(cfg)
  held = read_json(P050_OUT / 'exp_detour_ratio' / 'datasets.json')
  with np.load(SIDECAR, allow_pickle=True) as sc:
    source = sc['source_episode'].astype(np.int64); route = sc['route_realized'].astype(str)
  old_held = set(json.loads(str(np.load(P050_OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  cnew = set(read_json(P050_OUT / 'exp_episode_coverage' / 'manifest.json')['episodes']['new_held_out_detour (reserved, Cnew)'])
  man = {
      'experiment': 'AntMaze V6 mainline pilot: observational (O) vs counterfactual-oracle (CF) positive futures, matched offline CRL',
      'contract': {'path': str(CONTRACT.relative_to(ROOT)), 'sha256': sha256(CONTRACT) if CONTRACT.exists() else None},
      'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': git_head(), 'host': platform.node(),
      'dataset': {'path': str(DATASET), 'sha256': sha256(DATASET), 'expected_sha256': D05_EXPECTED['gxy_sha256'],
                  'sidecar_sha256': sha256(SIDECAR), 'expected_sidecar_sha256': D05_EXPECTED['sidecar_sha256'],
                  'n_episodes': int(len(lengths)), 'rows': int((lengths - 1).sum()), 'length_min_max_mean': [int(lengths.min()), int(lengths.max()), float(lengths.mean())],
                  'route_counts': {k: int((route == k).sum()) for k in np.unique(route)},
                  'selection': held['ratios']['d05'] | {'selection_seed': held['selection_seed'], 'nested': held['nested']},
                  'training_split': 'the whole d05 file is the training set (critic anchors and actor rows); evaluation = fresh native draws; '
                                    'the old 100 held-out and the 30 Cnew reserved pool episodes are excluded from d05 by construction',
                  'source_episodes_disjoint_from_old_held_out_and_cnew': bool(not (set(source.tolist()) & (old_held | cnew))),
                  'training_episode_source_ids_sha256': hashlib.sha256(source.tobytes()).hexdigest()},
      'start_agent': start_agent_record(),
      'anchors': ameta | {'file': str(OUT / 'anchors.npz'), 'sha256': sha256(OUT / 'anchors.npz'), 'weight_sum': float(anchors.weight.sum()),
                          'weights_identical_across_arms': 'by construction (one file read by both arms)',
                          'per_anchor_fields': ['episode', 't', 'state', 'goal_xy', 'action', 'multiplicity', 'weight', 'source_episode', 'ep_length'],
                          'route_share_of_anchor_mass': {k: float(anchors.weight[route[anchors.episode] == k].sum()) for k in np.unique(route)}},
      'arm_O': {'anchors': 'anchors.npz', 'positive_futures': 'recorded rows t+1..L_e-1 of the anchor episode',
                'future_law': 'P(m) proportional to gamma^m, m = 1..L_e-1-t (truncated at the episode end)',
                'negatives': 'in-batch (the other anchors\' positive goals): the arm\'s own goal marginal'},
      'arm_CF': {'anchors': 'anchors.npz (same file, same weights)', 'query_action': 'the logged torque a_t, executed once',
                 'continuation': 'start agent, mode tanh(loc), closed-loop from the state after the query step',
                 'action_selection_mode': 'mode', 'draws_per_anchor': 1,
                 'hidden_draws': f'hazard latents, both clocks and rock jitter redrawn from the priors; seed {HAZARD_SEED0} + anchor_id',
                 'horizon': f'absolute {HORIZON}: a branch from time t runs at most {HORIZON} - t steps',
                 'terminal_handling': 'ends on the first reach frame, on death, or at the horizon; every outcome kept (no success or route filter)',
                 'goal_hold': 'none (matched to the recorded episodes, which end on their reach frame); the historical branch replays parked to the horizon',
                 'future_law': 'P(m) proportional to gamma^m, m = 1..len-1 over the branch rows (truncated at the branch end)',
                 'negatives': 'in-batch: the arm\'s own goal marginal (documented, not forced to match arm O)',
                 'existing_d05_branches': 'replay_policy_d05.npz NOT reused: continuation = the d05 pure-BC walker, dense strata + every-60th anchors, '
                                          'goal hold to the horizon, 2 draws, extra sampled candidates -- none of it matches this contract',
                 'generation_seeds': {'hazard_seed0': HAZARD_SEED0, 'worker_seed0': GEN_WORKER_SEED0}},
      'actor_stream': {'both_arms': 'TrajectoryBuffer over the d05 file, default law: episode uniform, row uniform over [0, L_e - 1), future goal '
                                    'geometric gamma^(j-i) truncated at the episode end (the recipe\'s own draw); seed 20000 + training seed',
                       'pairing': 'random_goals 0: critic term at (s_i, g_j) with the sampled action; BC = -log pi(a_i | s_i, g_j) on the same rows; bc 0.05',
                       'reset_restriction': 'none (the whole row range is eligible)',
                       'interface_note': 'critic and actor batches are separate streams (crl.losses.build_learner(separate_actor_batch=True)); '
                                         'the loss algebra is the original\'s, the pairing of rows is not the shared-batch implementation'},
      'training': {'config': config_dump(cfg), 'optimizer_updates': UPDATES, 'scan_group': G, 'batch_size': BATCH,
                   'update_rule': 'one critic Adam step and one actor Adam step per optimizer update, both every update (joint)',
                   'critic_init': 'fresh: crl.train\'s init_state at PRNGKey(seed) (identical across arms for a seed)',
                   'actor_init': 'policy_params of the start agent; fresh Adam state for both optimizers',
                   'critic_stream_seed': f'{CRITIC_STREAM_SEED0} + seed', 'actor_stream_seed': f'{ACTOR_STREAM_SEED0} + seed',
                   'seeds': list(SEEDS), 'milestones_saved_not_selected': list(MILESTONES), 'evaluated_checkpoint': 'final (update 30,000)',
                   'discount': GAMMA, 'discount_note': 'gamma 0.999 = the branch-chain recipe (the frozen V6 ladder used 0.99); pinned, not swept'},
      'evaluation': {'env': D.ENV_XY, 'p_active': [0.5, 0.5], 'horizon': HORIZON, 'n': EVAL['n'], 'seed': EVAL['seed'], 'policy': EVAL['policy'],
                     'policies': ['start agent', 'O seeds 0-2 (final)', 'CF seeds 0-2 (final)'],
                     'same_episodes': 'one env seed; the six hidden streams and the reset pose are consumed at reset only, so episode k is identical for every policy (checked post hoc)',
                     'reserved_seeds_untouched': RESERVED_SEEDS},
      'comparisons': {'primary': 'CF - O success (mode), paired on the 300 common episodes within each training seed, then over the 3 paired seeds',
                      'practical': 'CF - start and O - start success on the same episodes',
                      'secondary_metrics': ['detour rate', 'death', 'timeout', 'success with / without an active hazard', 'zone deaths'],
                      'rule': 'improvement = mean over the 3 paired seeds > 2 x the seed s.e. of the paired differences AND 3/3 seeds in the same direction; '
                              'ties are never counted against; evaluation uncertainty (episode-paired bootstrap, seeds fixed) reported separately from seed variability',
                      'secondary_diagnostic': 'true-reset candidate ranking: secondary only; not a substitute for policy improvement; the old mixed Cdev gate does not reject',
                      'no_selection': 'no checkpoint, continuation, seed or batch source is chosen after evaluation'},
      'scripts_sha256': {p: sha256(ROOT / p) for p in ('scripts/exp_v6_mainline_pilot.py', 'crl/losses.py', 'crl/replay.py', 'crl/train.py',
                                                        'scripts/run_v6_branch_replay.py', 'scripts/build_v6_branch_replay.py',
                                                        'scripts/diag_v6_first_step_crossover.py', 'scripts/eval_rockfall_clock_v6_baseline.py')},
  }
  write_json(OUT / 'manifest.json', man)
  print(f'sealed {OUT / "manifest.json"}; start agent available: {man["start_agent"]["available"]}', flush=True)


# ------------------------------------------------------------------- audit
def _stream_hashes(anchors, futures, seed, n_batches, batch=BATCH):
  cs = CriticStream(anchors, futures, batch, GAMMA, CRITIC_STREAM_SEED0 + seed)
  rows = []
  for _ in range(n_batches):
    tr, (k, m) = cs.sample()
    assert np.all(m >= 1) and np.all(m <= futures.lengths[k] - 1), 'future row outside the path'
    rows.append({'anchor_ids': arr_hash(k), 'roots': arr_hash(tr.observation[:, :STATE_DIM], tr.action),
                 'goals': arr_hash(tr.observation[:, STATE_DIM:]), 'futures_within_path': True})
  return rows


def mode_audit(args, smoke_dir=None):
  out = smoke_dir or OUT
  anchors, ameta = AnchorSet.load(out / 'anchors.npz')
  obs, act, lengths, _ = load_dataset()
  cfg = recipe_config(0, out / '_cfg')
  fill_dims(cfg)
  res = {'smoke': smoke_dir is not None}
  # 1. dataset identity and the excluded episodes
  res['dataset_sha_matches'] = sha256(DATASET) == D05_EXPECTED['gxy_sha256']
  res['sidecar_sha_matches'] = sha256(SIDECAR) == D05_EXPECTED['sidecar_sha256']
  with np.load(SIDECAR, allow_pickle=True) as sc:
    source = sc['source_episode'].astype(np.int64)
  old_held = set(json.loads(str(np.load(P050_OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  cnew = set(read_json(P050_OUT / 'exp_episode_coverage' / 'manifest.json')['episodes']['new_held_out_detour (reserved, Cnew)'])
  res['no_held_out_or_cnew_episode_in_training'] = not (set(source.tolist()) & (old_held | cnew))
  # 2. anchors: deterministic, valid, the law's marginals
  if smoke_dir is None:
    a2, _ = declare_anchors()
    res['anchors_deterministic'] = all(np.array_equal(getattr(a2, f.name), getattr(anchors, f.name)) for f in dataclasses.fields(AnchorSet))
  res['anchors_valid_rows'] = bool(np.all(anchors.t <= anchors.ep_length - 2) and np.all(anchors.t >= 0))
  res['anchor_weight_sum'] = float(anchors.weight.sum())
  res['anchor_roots_match_dataset'] = bool(np.array_equal(anchors.state, obs[anchors.episode, anchors.t, :STATE_DIM])
                                           and np.array_equal(anchors.action, act[anchors.episode, anchors.t])
                                           and np.array_equal(anchors.goal_xy, obs[anchors.episode, anchors.t, STATE_DIM:OBS_W]))
  per_ep = np.bincount(anchors.episode, weights=anchors.multiplicity, minlength=len(lengths))
  rel_t = anchors.t / np.maximum(anchors.ep_length - 1, 1)
  res['anchor_law'] = {'draws_per_episode_mean_sd': [float(per_ep.mean()), float(per_ep.std())],
                       'expected_sd_if_uniform': float(np.sqrt(ameta['k_draws'] / len(lengths))),
                       'relative_row_position_mean (0.5 if uniform)': float(np.average(rel_t, weights=anchors.multiplicity)),
                       'share_t0': float(anchors.weight[anchors.t == 0].sum()), 'expected_share_t0': float(np.mean(1.0 / (lengths - 1)))}
  # 3. critic streams: identical anchors / roots across arms, futures inside the path
  rec = RecordedFutures(anchors, obs, act)
  branch_path = out / 'branches_cf.npz'
  res['branches_available'] = branch_path.exists()
  h_rec = _stream_hashes(anchors, rec, 0, args.n_batches, batch=(64 if smoke_dir else BATCH))
  res['critic_stream_recorded_first_batches'] = h_rec
  if branch_path.exists():
    br = BranchFutures(anchors, branch_path)   # its constructor asserts root state / action / episode / t identity
    h_br = _stream_hashes(anchors, br, 0, args.n_batches, batch=(64 if smoke_dir else BATCH))
    res['critic_anchor_sequence_identical_across_arms'] = all(x['anchor_ids'] == y['anchor_ids'] and x['roots'] == y['roots'] for x, y in zip(h_rec, h_br))
    res['critic_goals_differ_across_arms'] = any(x['goals'] != y['goals'] for x, y in zip(h_rec, h_br))
    with np.load(branch_path, allow_pickle=False) as d:
      res['branch_roots_keep_timestep'] = bool(np.array_equal(d['t'], anchors.t) and np.array_equal(d['episode'], anchors.episode))
      res['query_executed_exactly_once'] = bool(np.all(d['n_query_steps'] == 1))
      res['branch_first_action_is_logged'] = bool(np.array_equal(d['act_rows'][d['offset']], anchors.action))
      res['branch_lengths_within_horizon'] = bool(np.all(d['length'] - 1 <= HORIZON - anchors.t))
      res['branch_restore_maxdiff_max'] = float(d['restore_maxdiff'].max())
      res['branch_outcomes'] = {k: int((d['outcome'].astype(str) == k).sum()) for k in np.unique(d['outcome'].astype(str))}
      res['branch_all_outcomes_kept'] = int(len(d['length'])) == anchors.n
  # 4. actor streams: identical across arms for a seed, ordinary rows, futures inside the episode
  act_rows = {}
  for s in (SEEDS if smoke_dir is None else (0,)):
    st1 = ActorStream(cfg, ACTOR_STREAM_SEED0 + s); st2 = ActorStream(cfg, ACTOR_STREAM_SEED0 + s)
    st3 = ActorStream(cfg, ACTOR_STREAM_SEED0 + s)      # a third instance: its index draws are the batches' rows
    hs, t0_frac, cross = [], [], False
    for _ in range(args.n_batches):
      b1 = st1.sample(BATCH); b2 = st2.sample(BATCH)
      hs.append(arr_hash(b1.observation, b1.action) == arr_hash(b2.observation, b2.action))
      traj, i, j = st3.buffer.sampled_indices(BATCH)
      assert np.array_equal(b1.action, st3.buffer._act[traj, i]), 'index draw does not reproduce the batch'
      t0_frac.append(float((i == 0).mean()))
      cross = cross or bool(np.any(j <= i) or np.any(j >= st1.buffer.lengths[traj]))
    act_rows[f'seed_{s}'] = {'identical_across_instances': all(hs), 'reset_row_share': float(np.mean(t0_frac)),
                             'non_reset_row_share': float(1 - np.mean(t0_frac)), 'future_crosses_episode_boundary': cross,
                             'sampler': 'TrajectoryBuffer variable-length law', 'law_check': st1.fingerprint.get('prepare', None) is None}
  res['actor_streams'] = act_rows
  # 5. the static offline audit gates on the dataset (crl.offline_audit)
  from crl import offline_audit
  buf, _ = offline_audit.build_offline_buffer(str(DATASET), cfg, prepare=None)
  passed, gates, _ = offline_audit.run_static_audit(str(DATASET), cfg, buffer=buf)
  res['offline_static_audit'] = {'all_pass': bool(passed), 'gates': {k: bool(v) for k, v in gates.items()}}
  # 6. start checkpoint and d20 exclusion
  res['start_checkpoint'] = {'path': str(START_CKPT), 'available': START_CKPT.exists(), 'sha256': (sha256(START_CKPT) if START_CKPT.exists() else 'UNAVAILABLE')}
  man_path = out / 'manifest.json'
  if man_path.exists():
    res['manifest_mentions_no_d20_artifact'] = _no_d20(read_json(man_path))
  # 7. post-training: parameters moved, both arms
  post = {}
  for arm in ARMS:
    for s in (SEEDS if smoke_dir is None else (0,)):
      p = out / arm / f'seed_{s}' / 'train_manifest.json'
      if p.exists():
        tm = read_json(p)
        post[f'{arm}/seed_{s}'] = {'critic_changed': tm['params']['q_init'] != tm['params']['q_final'],
                                   'actor_changed': tm['params']['policy_init'] != tm['params']['policy_final'],
                                   'optimizer_updates': tm['optimizer_updates'], 'actor_init_is_start_agent': tm['params'].get('policy_init_is_start'),
                                   'first_critic_anchor_hash': tm['first_batches']['critic_anchor_ids'], 'first_actor_hash': tm['first_batches']['actor']}
  if post:
    res['post_training'] = post
    for s in SEEDS:
      a, b = post.get(f'O/seed_{s}'), post.get(f'CF/seed_{s}')
      if a and b:
        res.setdefault('paired_streams_identical_across_arms', {})[f'seed_{s}'] = (a['first_critic_anchor_hash'] == b['first_critic_anchor_hash']
                                                                                  and a['first_actor_hash'] == b['first_actor_hash'])
  # 8. post-evaluation: the same episodes for every policy
  evs = {n: p for n, p in eval_paths(out).items() if p.exists()}
  if len(evs) > 1:
    keys = ('u1', 'u2', 'sampled_t0_1', 'sampled_t0_2', 'start_yaw_deg')
    sig = {n: [tuple(r[k] for k in keys) for r in read_json(p)['episodes']] for n, p in evs.items()}
    ref = next(iter(sig.values()))
    res['evaluation_episodes_identical_across_policies'] = all(v == ref for v in sig.values())
  write_json(out / 'audit.json', res)
  (out / 'AUDIT.md').write_text(audit_md(res), encoding='utf-8')
  print(audit_md(res), flush=True)
  return res


def _no_d20(man):
  """No d20 data, policy, replay or critic is referenced as an INPUT of the pilot
  (the manifest's own prose notes that say 'not d20' are excluded)."""
  inputs = {'dataset': man['dataset']['path'], 'start': man['start_agent']['path'], 'anchors': man['anchors']['file'],
            'chain': [v for v in man['start_agent']['provenance_chain'].values()]}
  return 'd20' not in json.dumps(inputs)


def audit_md(res):
  L = ['# Pre-training and post-hoc checks (mainline pilot)', '']
  if res.get('smoke'):
    L.append('SMOKE RUN: toy scale, stand-in continuation, no checkpoint -- a code-path test, not a result.')
  flat = []

  def walk(prefix, v):
    if isinstance(v, dict):
      for k, x in v.items():
        walk(f'{prefix}.{k}' if prefix else k, x)
    elif isinstance(v, list) and v and isinstance(v[0], dict):
      flat.append((prefix, f'{len(v)} batches: ' + '; '.join(f"{d.get('anchor_ids', '')}/{d.get('goals', '')}" for d in v[:3]) + ' ...'))
    else:
      flat.append((prefix, v))
  walk('', res)
  L += ['| check | value |', '|---|---|']
  for k, v in flat:
    mark = ''
    if isinstance(v, bool):
      mark = 'PASS' if v else 'FAIL'
      if k.endswith(('critic_goals_differ_across_arms', 'branches_available', 'smoke')) or 'available' in k:
        mark = str(v)
      if k.endswith('future_crosses_episode_boundary'):
        mark = 'FAIL' if v else 'PASS'
      if k.endswith('actor_init_is_start_agent') and res.get('smoke'):
        mark = 'n/a (smoke: fresh actor)'
    L.append(f'| {k} | {mark or v} |')
  return '\n'.join(L) + '\n'


# ------------------------------------------------------------------- train
def run_dir(arm, seed, base=None):
  return (base or OUT) / arm / f'seed_{seed}'


def _stack(batches):
  import jax.numpy as jnp
  from crl.losses import Transition
  return Transition(*[jnp.asarray(np.stack([getattr(b, f) for b in batches], axis=0)) for f in Transition._fields])


def _tree_hash(tree):
  import jax
  return arr_hash(*[np.asarray(x) for x in jax.tree_util.tree_leaves(tree)])


def train_arm(arm, seed, updates=UPDATES, base=None, batch=BATCH, start_ckpt=START_CKPT, branch_path=None, log_every=LOG_EVERY):
  import jax
  import optax
  from crl import checkpoint
  from crl import losses as losses_mod
  base = base or OUT
  d = run_dir(arm, seed, base)
  if (d / 'final.pkl').exists():
    print(f'{d} exists', flush=True)
    return
  d.mkdir(parents=True, exist_ok=True)
  anchors, _ = AnchorSet.load(base / 'anchors.npz')
  obs, act, lengths, _ = load_dataset()
  cfg = recipe_config(seed, d, steps=updates)
  cfg.batch_size = int(batch)
  fill_dims(cfg)
  nets = make_nets(cfg)
  policy_optimizer = optax.adam(cfg.actor_learning_rate, eps=1e-7)
  q_optimizer = optax.adam(cfg.learning_rate, eps=1e-7)
  gidx = None if cfg.goal_indices is None else np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    import jax.numpy as jnp
    return states[:, jnp.asarray(gidx)] if gidx is not None else states[:, cfg.start_index:cfg.end_index]
  init_state, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, policy_optimizer, q_optimizer, separate_actor_batch=True)
  key = jax.random.PRNGKey(int(seed))
  key, key_init = jax.random.split(key)
  state = init_state(key_init)                     # fresh critic (and a fresh actor that is replaced next), paired by the seed
  if start_ckpt is not None:
    _, st0 = checkpoint.load_checkpoint(start_ckpt)
    policy_params = st0.policy_params
    policy_init_is_start = True
  else:
    policy_params = state.policy_params            # smoke only: no checkpoint
    policy_init_is_start = False
  state = state._replace(policy_params=policy_params, policy_optimizer_state=policy_optimizer.init(policy_params))
  h_q0, h_p0 = _tree_hash(state.q_params), _tree_hash(state.policy_params)
  checkpoint.save_named(str(d), 'init', 0, state)
  # the two streams
  futures = RecordedFutures(anchors, obs, act) if arm == 'O' else BranchFutures(anchors, branch_path or (base / 'branches_cf.npz'))
  critic_stream = CriticStream(anchors, futures, cfg.batch_size, cfg.discount, CRITIC_STREAM_SEED0 + seed)
  actor_stream = ActorStream(cfg, ACTOR_STREAM_SEED0 + seed)

  def _multi(state, pair):
    state, metrics = jax.lax.scan(update_step, state, pair)
    return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
  multi_update = jax.jit(_multi)
  assert updates % G == 0
  n_updates, t0, first, hist = 0, time.time(), {}, []
  for it in range(updates // G):
    cbs, ks = [], []
    for _ in range(G):
      tr, (k, m) = critic_stream.sample(); cbs.append(tr); ks.append(k)
    abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
    if it == 0:
      first = {'critic_anchor_ids': arr_hash(*ks), 'critic_goals': arr_hash(*[b.observation[:, STATE_DIM:] for b in cbs]),
               'actor': arr_hash(*[b.observation for b in abs_], *[b.action for b in abs_])}
    state, metrics = multi_update(state, (_stack(cbs), _stack(abs_)))
    n_updates += G
    if n_updates % log_every == 0 or n_updates == updates:
      m = {k: float(v) for k, v in metrics.items()}
      hist.append({'update': n_updates, **m})
      print(f'[{arm} s{seed} upd {n_updates:>6}] critic {m.get("critic_loss", 0):.4f} cat_acc {m.get("categorical_accuracy", 0):.3f} '
            f'actor {m.get("actor_loss", 0):.4f} bc_nll {m.get("bc_nll", 0):.3f} q_term {m.get("actor_q_term", 0):.3f} '
            f'{n_updates / (time.time() - t0):.1f} upd/s', flush=True)
    for ms in MILESTONES:
      if n_updates == ms:
        checkpoint.save_named(str(d), str(ms), n_updates, state)
  assert n_updates == updates
  checkpoint.save_named(str(d), 'final', n_updates, state)
  import jax as _jax
  write_json(d / 'train_manifest.json', {
      'arm': arm, 'seed': seed, 'optimizer_updates': n_updates, 'scan_group': G, 'batch_size': cfg.batch_size, 'wall_seconds': time.time() - t0,
      'futures': futures.name, 'branch_file': (None if arm == 'O' else str(branch_path or (base / 'branches_cf.npz'))),
      'start_ckpt': (str(start_ckpt) if start_ckpt else None), 'start_ckpt_sha256': (sha256(start_ckpt) if start_ckpt else None),
      'anchors_sha256': sha256(base / 'anchors.npz'), 'critic_stream_seed': CRITIC_STREAM_SEED0 + seed, 'actor_stream_seed': ACTOR_STREAM_SEED0 + seed,
      'jax_key_seed': seed, 'config': config_dump(cfg), 'first_batches': first,
      'params': {'q_init': h_q0, 'q_final': _tree_hash(state.q_params), 'policy_init': h_p0, 'policy_final': _tree_hash(state.policy_params),
                 'policy_init_is_start': policy_init_is_start},
      'device': str(_jax.devices()[0]), 'jax_version': _jax.__version__, 'host': platform.node(), 'history': hist})
  print(f'{arm} seed {seed}: {n_updates} optimizer updates in {time.time() - t0:.0f} s', flush=True)


def mode_train(args):
  if not START_CKPT.exists():
    raise SystemExit(f'BLOCKED: the start checkpoint {START_CKPT} is not available on this machine')
  if args.arm == 'CF' and not (OUT / 'branches_cf.npz').exists():
    raise SystemExit('arm CF needs branches_cf.npz (mode generate)')
  for s in args.seeds:
    train_arm(args.arm, s)


# ---------------------------------------------------------------- evaluate
def eval_paths(base=None):
  base = base or OUT
  paths = {'start': base / 'start_agent' / f'eval_mean_s{EVAL["seed"]}.json'}
  for arm in ARMS:
    for s in SEEDS:
      paths[f'{arm}/seed_{s}'] = run_dir(arm, s, base) / f'eval_mean_s{EVAL["seed"]}.json'
  return paths


def mode_evaluate(args):
  if not START_CKPT.exists():
    raise SystemExit(f'BLOCKED: the start checkpoint {START_CKPT} is not available on this machine')
  D.EVAL['seed'], D.EVAL['n'] = EVAL['seed'], EVAL['n']
  todo = [('start', START_CKPT, OUT / 'start_agent')]
  for arm in ARMS:
    for s in SEEDS:
      ck = run_dir(arm, s) / 'final.pkl'
      if ck.exists():
        todo.append((f'{arm}/seed_{s}', ck, run_dir(arm, s)))
  for name, ck, od in todo:
    if args.only and name not in args.only:
      continue
    print(f'== evaluate {name}', flush=True)
    D.evaluate_ckpt(ck, od, EVAL['policy'])


# ------------------------------------------------------------------ report
def _episodes(path):
  ev = read_json(path)
  rows = ev['episodes']
  return {'success': np.array([bool(r['success']) for r in rows]), 'failure': np.array([bool(r['failure']) for r in rows]),
          'timeout': np.array([bool(r['timeout']) for r in rows]), 'detour': np.array([r['route'] == 'detour' for r in rows]),
          'hazard': np.array([bool(r['u1']) or bool(r['u2']) for r in rows]), 'z1': np.array([bool(r['zone1_death']) for r in rows]),
          'z2': np.array([bool(r['zone2_death']) for r in rows]), 'steps': np.array([r['steps'] for r in rows]), 'ckpt_step': ev.get('ckpt_step')}


def _headline(e):
  h = e['hazard']
  return {'success': e['success'].mean(), 'detour': e['detour'].mean(), 'death': e['failure'].mean(), 'timeout': e['timeout'].mean(),
          'success_no_hazard': e['success'][~h].mean() if (~h).any() else float('nan'), 'success_hazard': e['success'][h].mean() if h.any() else float('nan'),
          'n_no_hazard': int((~h).sum()), 'n_hazard': int(h.sum()), 'zone_deaths': [int(e['z1'].sum()), int(e['z2'].sum())], 'mean_steps': float(e['steps'].mean())}


def paired_block(a_by_seed, b_by_seed, key='success', n_boot=2000, seed=0):
  """a - b on the common episodes.  a_by_seed / b_by_seed: {seed: episode
  dict}; b may be a single dict (the start agent) reused for every seed.
  Returns per-seed paired differences with their episode s.e., the seed mean
  and seed s.e. (n = 3), the count of seeds in the direction of the mean,
  and an episode-paired bootstrap s.e. of the seed mean (seeds fixed)."""
  seeds = sorted(a_by_seed)
  diffs, per = [], {}
  D_ = []
  for s in seeds:
    b = b_by_seed[s] if isinstance(b_by_seed, dict) and s in b_by_seed else b_by_seed
    dd = a_by_seed[s][key].astype(float) - b[key].astype(float)
    D_.append(dd)
    per[s] = {'mean': float(dd.mean()), 'episode_se': float(dd.std(ddof=1) / np.sqrt(len(dd))), 'n_better': int((dd > 0).sum()),
              'n_worse': int((dd < 0).sum()), 'n_tie': int((dd == 0).sum())}
    diffs.append(dd.mean())
  diffs = np.array(diffs)
  D_ = np.stack(D_)                                                   # [seeds, episodes]
  mean = float(diffs.mean())
  seed_se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float('nan')
  same = int(np.sum(np.sign(diffs) == np.sign(mean))) if mean != 0 else 0
  rng = np.random.default_rng(seed)
  n = D_.shape[1]
  boots = [D_[:, rng.integers(0, n, n)].mean() for _ in range(n_boot)]
  met = bool(mean > 0 and (seed_se == seed_se) and mean > 2 * seed_se and same == len(diffs))
  return {'per_seed': per, 'mean': mean, 'seed_se': seed_se, 'seeds_same_direction': f'{same}/{len(diffs)}',
          'episode_bootstrap_se_of_mean': float(np.std(boots)), 'improvement_rule_met': met}


def mode_report(args):
  paths = eval_paths()
  have = {n: p for n, p in paths.items() if p.exists()}
  man = read_json(OUT / 'manifest.json') if (OUT / 'manifest.json').exists() else {}
  L = ['# Mainline pilot: observational (O) vs counterfactual-oracle (CF) positive futures', '',
       f'Contract `notes/MAINLINE_CONTRACT.md`; manifest `manifest.json` (sealed {man.get("sealed_at", "-")}); checks `AUDIT.md`.  '
       f'Evaluation: {EVAL["n"]} natural draws, seed {EVAL["seed"]}, mode policy, the same episodes for every policy.  '
       'Uncertainty: per-seed episode s.e. (evaluation), seed s.e. over 3 paired training seeds (training variability), '
       'episode-paired bootstrap of the seed mean (evaluation uncertainty with the seeds fixed).  Rule: CF - O mean over the 3 paired '
       'seeds > 2 x seed s.e. and 3/3 seeds in the same direction.', '']
  if not have:
    L += ['## Status', '', f'No evaluation exists.  Start checkpoint available: {START_CKPT.exists()} ({START_CKPT}).  See the blocker section of the SUMMARY.']
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L)); return
  E = {n: _episodes(p) for n, p in have.items()}
  L += ['## Per policy', '', '| policy | success | detour | death | timeout | success no hazard (n) | success hazard (n) | zone-1 / zone-2 deaths | mean steps |', '|---|---:|---:|---:|---:|---:|---:|---|---:|']
  for n, e in E.items():
    h = _headline(e)
    L.append(f'| {n} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_no_hazard"]:.3f} ({h["n_no_hazard"]}) | '
             f'{h["success_hazard"]:.3f} ({h["n_hazard"]}) | {h["zone_deaths"][0]} / {h["zone_deaths"][1]} | {h["mean_steps"]:.0f} |')
  by = {arm: {s: E[f'{arm}/seed_{s}'] for s in SEEDS if f'{arm}/seed_{s}' in E} for arm in ARMS}
  res = {}
  if all(len(by[a]) == len(SEEDS) for a in ARMS):
    L += ['', '## Seed means (seed s.e., n = 3)', '', '| arm | success | detour | death | timeout | success hazard |', '|---|---|---|---|---|---|']
    for arm in ARMS:
      hs = [_headline(by[arm][s]) for s in SEEDS]
      f = lambda k: f'{np.mean([h[k] for h in hs]):.3f} ({np.std([h[k] for h in hs], ddof=1) / np.sqrt(3):.3f})'
      L.append(f'| {arm} | {f("success")} | {f("detour")} | {f("death")} | {f("timeout")} | {f("success_hazard")} |')
    L += ['', '## Paired comparisons on the common episodes', '']
    comps = [('CF - O (primary)', by['CF'], by['O'])]
    if 'start' in E:
      comps += [('CF - start (practical)', by['CF'], E['start']), ('O - start', by['O'], E['start'])]
    for key in ('success', 'detour', 'failure', 'timeout'):
      L += [f'### {key}', '', '| comparison | per seed (episode s.e.) | mean | seed s.e. | episode-bootstrap s.e. | seeds same direction | rule |', '|---|---|---:|---:|---:|---|---|']
      for name, a, b in comps:
        r = paired_block(a, b, key=key)
        res[f'{name}:{key}'] = r
        per = ' / '.join(f'{v["mean"]:+.3f} ({v["episode_se"]:.3f})' for v in r['per_seed'].values())
        rule = ('met' if r['improvement_rule_met'] else 'not met') if (key == 'success' and ('primary' in name or 'practical' in name)) else '-'
        L.append(f'| {name} | {per} | {r["mean"]:+.3f} | {r["seed_se"]:.3f} | {r["episode_bootstrap_se_of_mean"]:.3f} | {r["seeds_same_direction"]} | {rule} |')
      L.append('')
    p = res['CF - O (primary):success']
    L += ['## Reading', '',
          f'Primary (CF - O, success): {p["mean"]:+.3f}, seed s.e. {p["seed_se"]:.3f}, {p["seeds_same_direction"]} seeds; rule {"MET" if p["improvement_rule_met"] else "NOT MET"}.']
    if 'CF - start (practical):success' in res:
      q = res['CF - start (practical):success']
      L.append(f'Practical (CF - start, success): {q["mean"]:+.3f}, seed s.e. {q["seed_se"]:.3f}, {q["seeds_same_direction"]}; rule {"MET" if q["improvement_rule_met"] else "NOT MET"}.  '
               'Beating a degraded control alone is insufficient; the practical check is required separately.')
    L += ['', 'This is oracle evidence for the sampling design (the simulator generated the counterfactual futures); it is not an offline learned-ETT result.']
  write_json(OUT / 'results.json', {'headlines': {n: _headline(e) for n, e in E.items()}, 'paired': res})
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


# ------------------------------------------------------------------- smoke
def mode_smoke(args):
  """Toy-scale end-to-end test of the mechanics: 48 anchors, a stand-in
  zero-torque continuation capped at 12 steps, arms O and CF trained for 8
  optimizer updates (batch 64) from a fresh actor on the CPU, then the audit.
  Nothing here is an experimental result; the directory is _smoke/."""
  base = OUT / '_smoke'
  base.mkdir(parents=True, exist_ok=True)
  anchors, meta = declare_anchors(seed=ANCHOR_SEED, k_draws=48)
  anchors.save(base / 'anchors.npz', meta | {'smoke': True})
  gmeta, res = generate_branches(anchors, None, 1, base / 'branches_cf.npz', max_steps_cap=12)
  gs = generation_summary(base / 'branches_cf.npz', anchors)
  write_json(base / 'generation_cf.json', gs)
  for arm in ARMS:
    train_arm(arm, 0, updates=8, base=base, batch=64, start_ckpt=None, log_every=4)
  ns = argparse.Namespace(n_batches=2)
  r = mode_audit(ns, smoke_dir=base)
  ok = (r['branch_roots_keep_timestep'] and r['query_executed_exactly_once'] and r['branch_first_action_is_logged']
        and r['critic_anchor_sequence_identical_across_arms'] and r['critic_goals_differ_across_arms']
        and all(v['critic_changed'] and v['actor_changed'] and v['optimizer_updates'] == 8 for v in r['post_training'].values())
        and r['actor_streams']['seed_0']['identical_across_instances'] and not r['actor_streams']['seed_0']['future_crosses_episode_boundary'])
  print('SMOKE', 'PASS' if ok else 'FAIL', flush=True)
  return 0 if ok else 1


# -------------------------------------------------------------------- main
def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('anchors', 'generate', 'seal', 'audit', 'train', 'evaluate', 'report', 'smoke'))
  ap.add_argument('--arm', choices=ARMS)
  ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None, help='generate: first N anchors only (smoke-scale check, never for training)')
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--n-batches', type=int, default=5)
  ap.add_argument('--only', nargs='*', default=None)
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  if args.mode == 'train' and not args.arm:
    ap.error('train needs --arm')
  r = {'anchors': mode_anchors, 'generate': mode_generate, 'seal': mode_seal, 'audit': mode_audit, 'train': mode_train,
       'evaluate': mode_evaluate, 'report': mode_report, 'smoke': mode_smoke}[args.mode](args)
  return r if isinstance(r, int) else 0


if __name__ == '__main__':
  sys.exit(main())
