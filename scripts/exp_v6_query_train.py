"""AntMaze V6 mainline -- query coverage, stage 2: ONE controlled training
comparison (user's decision, 2026-09-20, after the stage-1 generation result).

Stage 1 (`exp_v6_query_coverage.py`) found where the agent's own first torque
changes the outcome: at the RESET rows (t = 0) of the start region the lineage
agent's mode completes the far route 42 % vs 3.7 % for the logged torque and
the overall success rises 20 % -> 43 % (lineage 0; same states, same paired
hazard draws, same continuation policy).  The pre-registered pooled gate G1
was NOT met (the reset rows carry 0.4 % of the anchor weight, so +29 points
there is +1.4 points pooled) and stays "not met"; this stage is an explicitly
added, exploratory training experiment that asks the remaining question:

    do these alternative-action futures, once they enter the critic, improve
    the CURRENT policy?

Design (fixed before training; nothing else changes).
  * Two arms per lineage s in {0, 1, 2} (all three, none chosen):
      control   start-region anchors -> the q0 (logged-torque) futures of
                `query_coverage/queries_s{s}.npz`, both hazard draws;
      extended  start-region anchors -> the q0..q5 futures, both draws; the
                branch's own query torque is the critic's training ACTION
                for that row.
    Every other anchor -> the sealed mainline futures `branches_cf.npz`
    (start-agent continuation) in both arms.  So the two arms share the
    continuation policy in the start region (the lineage agent) and share
    everything outside it; they differ only in which first torques (and
    their consequences) the critic sees at the start-region anchors.
  * Per-anchor total weight unchanged (the anchor law); the anchor's weight
    is spread equally over its allowed branches (2 in control, 12 in
    extended) by a separate selection RNG, so the main stream RNG consumes
    exactly as in training and BOTH ARMS DRAW THE SAME ANCHOR SEQUENCE and
    the same actor batches (verified by first-batch hashes).
  * Start = the lineage final (30,000 updates; full TrainingState: policy,
    critic, target, both Adam states incl. the clip chain, key), streams
    fast-forwarded by 30,000 batches; +30,000 updates (the B budget); batch
    1024, G 4; the recipe's NCE, actor objective, BC 0.05, critic clip 0.1,
    gamma 0.999, learning rates; the actor / BC stream untouched.
  * Evaluation: every final and the three lineage agents (the "current
    policy") ONCE on a fresh paired evaluation draw (seed 6909, 300
    episodes, the policy MODE); success / detour / death / timeout and the
    route ledger; paired differences extended - control, extended - current,
    control - current with the mainline rule (mean over the 3 lineages >
    2 x seed s.e. and 3 / 3).  The reserved validation seeds are untouched.
  * Read-outs (secondary, on the training anchors -- in-sample, disclosed):
    (a) critic: at the start-region anchors, f(s, a_q, g_task) of the lineage
        critic vs each arm's final critic against the realised outcome of
        the SAME (anchor, query) branches -- within-anchor AUROC of "the
        branch succeeded" and the share of anchors whose critic-argmax query
        succeeded; reported for the reset rows and for all start anchors;
    (b) actor: each final actor's own closed-loop rollout from the 196
        reset anchors with the same two hazard draws (for the lineage agent
        this is stage 1's "mode" row): completed far / success.
  * No selection: every run evaluated once; no re-weighting of the reset
    rows, no BC change, no actor change (user).

Judgement, fixed before training (user):
  1. extended > control AND extended > current policy on success (rule met)
     -> query supervision is a usable improvement; confirm on a fresh draw.
  2. the extended critic learns the candidate differences (read-out a) but
     the actor does not improve -> stop adding queries; the remaining problem
     is policy extraction / the objective conflict (BC vs critic).
  3. the extended critic does not learn even the added actions' consequences
     -> the problem is value learning; further actor changes have no basis.

Modes:
  python scripts/exp_v6_query_train.py seal
  python scripts/exp_v6_query_train.py train --lineage s --arm control|extended
  python scripts/exp_v6_query_train.py evaluate_current [--lineage s]
  python scripts/exp_v6_query_train.py readout --lineage s [--workers 8]
  python scripts/exp_v6_query_train.py report
Outputs under outputs/antmaze_branch_replay_p050/exp_mainline_pilot/query_coverage/train/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_clip_round as CR  # noqa: E402
import exp_v6_query_coverage as QC  # noqa: E402

OUT = QC.OUT / 'train'
ARMS = ('control', 'extended')
ALLOWED = {'control': [0], 'extended': list(range(QC.N_QUERIES))}
# 'continue' (reference, added 2026-09-20 after lineages 0 / 1 showed both arms far below the current policy): the SAME resume
# (+30,000 updates from the lineage final, same streams) on the UNCHANGED sealed futures at every anchor -- no queries file at
# all.  It separates "30,000 more updates" from "the start-region futures changed"; it is not one of the two pre-fixed arms.
# 'frozen' (user's diagnostic, 2026-09-21): the 'continue' resume with the CRITIC FROZEN -- the same joint update step is run on
# the same batches, and the critic parameters, target and critic Adam state are restored after every update; the actor, its
# Adam state, the actor / BC batches and BC 0.05 are exactly those of 'continue'.  Separates "the critic keeps changing" from
# "optimising the actor along a fixed critic and the current training distribution" as the cause of the +30k degradation.
ARMS_ALL = ('control', 'extended', 'continue', 'frozen')
UPDATES = 30_000
EVAL = {'n': 300, 'seed': 6909, 'policy': 'mean'}     # a fresh paired draw for this round (909 / 2909 / 3909 / 4909 used; 5909 pre-registered for B; 616_000_005 / 616_500_000 reserved)
SELECT_SEED0 = 146_000_000                            # the branch-selection RNG (separate from the stream RNG): + lineage
READOUT_WORKERS = 8
RECIPE = CR.RECIPE
SEALED_BRANCHES = CR.OLD_BRANCHES


def run_dir(arm, s):
  return OUT / arm / f'seed_{s}'


def lineage_ckpt(s):
  return CR.lineage_ckpt(s)


def eval_file(d):
  return Path(d) / f'eval_{EVAL["policy"]}_s{EVAL["seed"]}.json'


# ------------------------------------------------------------ futures
class MixedFutures:
  """Branch table of one arm: the sealed mainline branch of every anchor
  outside the start region; for the start-region anchors the allowed query
  branches of the lineage's queries file (their own first torque is the
  critic's action).  Selection is uniform over an anchor's allowed branches
  from a separate RNG (the anchor's total weight is unchanged)."""
  name = 'mixed-query'

  def __init__(self, anchors, sealed_path, queries_path, allowed_queries):
    self.a = anchors
    with np.load(sealed_path, allow_pickle=False) as d:
      s_obs, s_act = d['obs_rows'], d['act_rows']
      s_off, s_len = d['offset'].astype(np.int64), d['length'].astype(np.int64)
      assert np.array_equal(d['episode'], anchors.episode) and np.array_equal(d['t'], anchors.t)
      self.sealed_meta = json.loads(str(d['meta']))
    assert np.array_equal(s_obs[s_off], anchors.obs31) and np.array_equal(s_act[s_off], anchors.action)
    with np.load(queries_path, allow_pickle=False) as d:
      q_obs, q_act = d['obs_rows'], d['act_rows']
      q_off, q_len = d['offset'].astype(np.int64), d['length'].astype(np.int64)
      q_anchor, q_query, q_draw = d['anchor_id'].astype(np.int64), d['query_id'].astype(np.int64), d['draw'].astype(np.int64)
      self.start_ids = d['start_anchor_ids'].astype(np.int64)
      self.queries_meta = json.loads(str(d['meta']))
      self.q_outcome = d['outcome'].astype(str)
    assert np.array_equal(self.start_ids, QC.start_anchor_ids(anchors)), 'the queries file does not cover the start-region anchors of the anchor set'
    keep = np.isin(q_query, allowed_queries)
    ki = np.flatnonzero(keep)
    # roots and logged first torques of the kept query branches
    assert np.array_equal(q_obs[q_off[ki]], anchors.obs31[q_anchor[ki]]), 'query branch roots differ from the logged anchor rows'
    lo = ki[q_query[ki] == 0]
    assert np.array_equal(q_act[q_off[lo]], anchors.action[q_anchor[lo]]), 'q0 branches do not start with the logged torque'
    # the branch table: sealed branches first (ids 0..n-1), then the kept query branches
    n = anchors.n
    q_rows = np.concatenate([np.arange(q_off[i], q_off[i] + q_len[i]) for i in ki]) if len(ki) else np.zeros(0, np.int64)
    self.obs_rows = np.concatenate([s_obs, q_obs[q_rows]])
    self.act_rows = np.concatenate([s_act, q_act[q_rows]])
    q_len_k = q_len[ki]
    q_off_new = len(s_obs) + np.concatenate([[0], np.cumsum(q_len_k)[:-1]]) if len(ki) else np.zeros(0, np.int64)
    self.b_off = np.concatenate([s_off, q_off_new]).astype(np.int64)
    self.b_len = np.concatenate([s_len, q_len_k]).astype(np.int64)
    self.b_anchor = np.concatenate([np.arange(n), q_anchor[ki]]).astype(np.int64)
    self.b_query = np.concatenate([np.full(n, -1), q_query[ki]]).astype(np.int64)
    self.b_draw = np.concatenate([np.full(n, -1), q_draw[ki]]).astype(np.int64)
    self.b_outcome = np.concatenate([np.array(['sealed'] * n), self.q_outcome[ki]])
    # per-anchor allowed branches: start anchors -> their query branches ONLY; others -> the sealed branch
    max_c = int(np.bincount(q_anchor[ki], minlength=n).max()) if len(ki) else 1
    self.choices = np.full((n, max(max_c, 1)), -1, np.int64)
    self.n_choices = np.ones(n, np.int64)
    self.choices[:, 0] = np.arange(n)
    is_start = np.zeros(n, bool); is_start[self.start_ids] = True
    self.n_choices[is_start] = 0
    for j, b in enumerate(range(n, n + len(ki))):
      k = self.b_anchor[b]
      self.choices[k, self.n_choices[k]] = b; self.n_choices[k] += 1
    assert np.all(self.n_choices >= 1), 'a start anchor has no allowed query branch'
    assert np.all(self.n_choices[is_start] == len(allowed_queries) * QC.DRAWS), 'every start anchor must have all allowed (query, draw) branches'
    assert np.all(self.n_choices[~is_start] == 1)
    self.is_start = is_start
    self.lengths = s_len                                   # CriticStream's constructor check only; the stream uses b_len of the selected branch
    self.allowed_queries = list(allowed_queries)

  def select(self, k, r):
    j = np.minimum((r * self.n_choices[k]).astype(np.int64), self.n_choices[k] - 1)
    return self.choices[k, j]

  def goal_of(self, b, m):
    return self.obs_rows[self.b_off[b] + m, :MP.GOAL_DIM]

  def action_of(self, b):
    return self.act_rows[self.b_off[b]]

  def next_rows_of(self, b):
    r = self.b_off[b] + 1
    return self.obs_rows[r, :MP.STATE_DIM], self.act_rows[r]

  def summary(self):
    st = self.is_start
    sel = np.flatnonzero(self.b_query >= 0)
    return {'anchors': int(self.a.n), 'start_anchors': int(st.sum()), 'start_weight': float(self.a.weight[st].sum()), 'branches': int(len(self.b_len)),
            'query_branches': int(len(sel)), 'allowed_queries': self.allowed_queries, 'branches_per_start_anchor': int(self.n_choices[st][0]),
            'query_branch_outcomes': {k: int((self.b_outcome[sel] == k).sum()) for k in ('success', 'death', 'timeout')},
            'sealed_meta_continuation': self.sealed_meta.get('continuation_ckpt'), 'queries_meta_continuation': self.queries_meta.get('continuation_ckpt')}


class QueryCriticStream(MP.CriticStream):
  """The mainline critic stream with a per-anchor branch selection: the main
  RNG consumes exactly two uniform(B) per batch (anchor, future) as in
  training, so the anchor sequence equals the lineage's own; the branch is
  chosen from a separate RNG; the future row follows the geometric law
  truncated at THAT branch's end; the critic action is the branch's own first
  torque."""

  def __init__(self, anchors, futures, batch, gamma, seed, select_seed):
    super().__init__(anchors, futures, batch, gamma, seed)
    self.sel_rng = np.random.default_rng(int(select_seed))

  def draw(self):
    k = np.minimum(np.searchsorted(self.cdf, self.rng.random(self.B), side='right'), self.a.n - 1)
    b = self.f.select(k, self.sel_rng.random(self.B))
    u = self.rng.random(self.B)
    n = self.f.b_len[b] - 1
    m = np.ceil(np.log1p(-u * (1.0 - self.gamma ** n)) / self.log_gamma).astype(np.int64)
    m = np.clip(m, 1, n)
    self._u, self._b = u, b
    return k, m

  def sample(self):
    from crl.losses import Transition
    k, m = self.draw(); b = self._b
    goals = np.stack([self.f.goal_of(int(bb), int(mm)) for bb, mm in zip(b, m)]).astype(np.float32)
    actions = np.stack([self.f.action_of(int(bb)) for bb in b]).astype(np.float32)
    nxt = [self.f.next_rows_of(int(bb)) for bb in b]
    next_state = np.stack([x[0] for x in nxt]).astype(np.float32)
    next_action = np.stack([x[1] for x in nxt]).astype(np.float32)
    obs = np.concatenate([self.a.state[k], goals], axis=1)
    return Transition(observation=obs, action=actions, reward=np.zeros(self.B, np.float32), discount=np.full(self.B, self.gamma, np.float32),
                      next_observation=np.concatenate([next_state, goals], axis=1), next_action=next_action), (k, m)


def build(s, arm, critic_clip):
  import jax.numpy as jnp
  import optax
  from crl import losses as losses_mod
  cfg = MP.recipe_config(s, run_dir(arm, s) / '_cfg')
  cfg.batch_size = MP.BATCH
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
  q_opt = MP.critic_optimizer(cfg, critic_clip)
  gidx = np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    return states[:, jnp.asarray(gidx)]
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, pol_opt, q_opt, separate_actor_batch=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  if arm in ('continue', 'frozen'):
    futures = MP.BranchFutures(anchors, SEALED_BRANCHES)
    critic_stream = MP.CriticStream(anchors, futures, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + s)
  else:
    futures = MixedFutures(anchors, SEALED_BRANCHES, QC.branch_path(s), ALLOWED[arm])
    critic_stream = QueryCriticStream(anchors, futures, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + s, SELECT_SEED0 + s)
  actor_stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + s)
  return cfg, nets, update_step, critic_stream, actor_stream, q_opt, futures


# ------------------------------------------------------------------ seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  qman = MP.read_json(QC.OUT / 'manifest.json')
  man = {'experiment': 'AntMaze V6 mainline, query coverage stage 2: controlled training comparison control (q0 futures) vs extended (q0..q5 futures, query torque = critic action) at the start-region anchors',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'exploratory training experiment added after stage 1 (its pooled gate G1 stays NOT met); oracle (simulator) futures; not offline identification',
         'stage_1': {'manifest_sha256': MP.sha256(QC.OUT / 'manifest.json'), 'report_sha256': (MP.sha256(QC.OUT / 'report.json') if (QC.OUT / 'report.json').exists() else 'pending'),
                     'queries': {f'seed_{s}': {'path': str(QC.branch_path(s)), 'sha256': (MP.sha256(QC.branch_path(s)) if QC.branch_path(s).exists() else 'missing')} for s in MP.SEEDS}},
         'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'), 'sealed_branches': {'path': str(SEALED_BRANCHES), 'sha256': MP.sha256(SEALED_BRANCHES), 'continuation': 'start agent'},
         'lineage_agents': {f'seed_{s}': {'path': str(lineage_ckpt(s)), 'sha256': (MP.sha256(lineage_ckpt(s)) if lineage_ckpt(s).exists() else 'missing')} for s in MP.SEEDS},
         'recipe': RECIPE, 'recipe_overrides': MP.VARIANTS[RECIPE],
         'arms': {'control': 'start-region anchors: q0 (logged torque) branches of the lineage queries file, both hazard draws (2 per anchor); every other anchor: the sealed branch',
                  'extended': 'start-region anchors: q0..q5 branches, both draws (12 per anchor), the branch first torque = the critic action; every other anchor: the sealed branch'},
         'shared': 'continuation policy in the start region (the lineage agent), futures outside it, anchors and weights (equal split over the allowed branches; separate selection RNG), the actor / BC stream, '
                   'NCE, actor objective, bc 0.05, critic clip 0.1, gamma 0.999, lr, batch 1024, G 4, the start TrainingState and the fast-forwarded streams',
         'start': 'the lineage final (30,000 updates) with policy, critic, target, both Adam states (clip chain) and key; streams fast-forwarded 30,000 batches', 'updates': UPDATES,
         'identity_check': 'first-batch hashes: critic anchor ids and actor rows identical across arms; goals / actions differ',
         'evaluation': {**EVAL, 'policies': ['control final', 'extended final', 'the lineage agent (current policy)', 'the start agent (reference only)'], 'paired': 'same draw for every policy',
                        'comparisons': ['extended - control', 'extended - current', 'control - current'], 'rule': 'mean over the 3 lineages > 2 x seed s.e. and 3 / 3 (the mainline rule)',
                        'quantities': ['success (primary)', 'detour', 'death', 'timeout', 'route ledger']},
         'readouts': {'critic': 'at the start-region anchors, f(s, a_q, g_task) (min over the twin) of the lineage critic and each final critic vs the realised outcome of the same (anchor, query) branches: '
                                'within-anchor AUROC of branch success (anchors with both outcomes), share of anchors whose critic-argmax query succeeded in >= 1 draw, f(best) - f(logged); reset rows and all start anchors',
                      'actor': 'each final actor\'s own closed-loop rollout (mode) from the 196 reset anchors with the same two hazard draws: completed far / success; the lineage agent\'s value = stage 1 "mode"',
                      'caveat': 'in-sample (the training anchors); the fresh evaluation draw is primary'},
         'judgement': {'1': 'extended > control AND extended > current (rule met on success) -> usable improvement; confirm on a fresh draw',
                       '2': 'extended critic learns the candidate differences, actor does not improve -> stop adding queries; extraction / objective conflict',
                       '3': 'extended critic does not learn the added actions\' consequences -> value learning problem'},
         'held_fixed_by_decision': ['no re-weighting of the reset rows', 'BC 0.05', 'the actor and its stream', 'no selection: every run evaluated once'],
         'stage_1_gate_status': (qman.get('gates') and 'G1 NOT met at the pooled level (lineage 0); this stage is an added exploratory experiment, not a gate pass')}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# ----------------------------------------------------------------- train
def mode_train(args):
  import jax
  from crl import checkpoint
  import diag_v6_training_replay as DR
  s, arm = int(args.lineage), args.arm
  man = MP.read_json(OUT / 'manifest.json')
  src = lineage_ckpt(s)
  assert MP.sha256(src) == man['lineage_agents'][f'seed_{s}']['sha256'], 'lineage checkpoint differs from the sealed hash'
  if arm not in ('continue', 'frozen'):
    assert MP.sha256(QC.branch_path(s)) == man['stage_1']['queries'][f'seed_{s}']['sha256'], 'queries file differs from the sealed hash'
  out = run_dir(arm, s)
  if (out / 'final.pkl').exists() and not args.force:
    print(f'{out} exists', flush=True)
  else:
    out.mkdir(parents=True, exist_ok=True)
    clip = float(MP.VARIANTS[RECIPE]['critic_clip'])
    cfg, nets, update_step, critic_stream, actor_stream, q_opt, futures = build(s, arm, clip)
    step0, state = checkpoint.load_checkpoint(src)
    fresh = q_opt.init(state.q_params)
    assert jax.tree_util.tree_structure(state.q_optimizer_state) == jax.tree_util.tree_structure(fresh), 'the checkpoint critic optimizer state is not the clip -> Adam chain'
    ff = DR.fast_forward(critic_stream, actor_stream, int(step0))
    fsum = futures.summary() if hasattr(futures, 'summary') else {'arm': arm, 'futures': 'the sealed mainline branches at every anchor (unchanged)', 'sha256': MP.sha256(SEALED_BRANCHES),
                                                                   'critic': ('FROZEN: q_params, target_q_params and the critic Adam state restored after every update' if arm == 'frozen' else 'trained')}
    is_start = futures.is_start if hasattr(futures, 'is_start') else np.isin(np.arange(futures.a.n), QC.start_anchor_ids(futures.a))
    b_query = futures.b_query if hasattr(futures, 'b_query') else None
    print(f'loaded {src} @ step {step0} ({arm}); futures {fsum}; streams fast-forwarded by {step0} batches in {ff:.0f} s', flush=True)
    h0 = {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)}
    checkpoint.save_named(str(out), 'init', int(step0), state)

    def _step(st, tr):
      new, m = update_step(st, tr)
      if arm == 'frozen':
        new = new._replace(q_params=st.q_params, target_q_params=st.target_q_params, q_optimizer_state=st.q_optimizer_state)
      return new, m

    def _multi(state, pair):
      state, metrics = jax.lax.scan(_step, state, pair)
      return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    multi_update = jax.jit(_multi)
    G = MP.G
    n, t0, hist, first = 0, time.time(), [], {}
    q_share_hist = []
    for it in range(UPDATES // G):
      cbs, ks, bs = [], [], []
      for _ in range(G):
        tr, (k, m) = critic_stream.sample(); cbs.append(tr); ks.append(k); bs.append(getattr(critic_stream, '_b', k))
      abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
      if it == 0:
        first = {'critic_anchor_ids': MP.arr_hash(*ks), 'critic_goals': MP.arr_hash(*[b.observation[:, MP.STATE_DIM:] for b in cbs]),
                 'critic_actions': MP.arr_hash(*[b.action for b in cbs]), 'critic_branches': MP.arr_hash(*bs),
                 'actor': MP.arr_hash(*[b.observation for b in abs_], *[b.action for b in abs_]),
                 'start_rows_in_first_batches': int(sum(int(is_start[k].sum()) for k in ks)),
                 'query_rows_in_first_batches': (int(sum(int((b_query[b] >= 1).sum()) for b in bs)) if b_query is not None else 0)}
      if it % 250 == 0:
        q_share_hist.append({'update': int(step0) + n, 'start_rows': int(sum(int(is_start[k].sum()) for k in ks)), 'added_query_rows': (int(sum(int((b_query[b] >= 1).sum()) for b in bs)) if b_query is not None else 0)})
      state, metrics = multi_update(state, (MP._stack(cbs), MP._stack(abs_)))
      n += G
      if n % MP.LOG_EVERY == 0 or n == UPDATES:
        mm = {k: float(v) for k, v in metrics.items()}
        hist.append({'update': int(step0) + n, **mm})
        print(f'[{arm} s{s} upd {int(step0) + n:>6}] critic {mm.get("critic_loss", 0):.4f} cat_acc {mm.get("categorical_accuracy", 0):.3f} actor {mm.get("actor_loss", 0):.4f} '
              f'bc_nll {mm.get("bc_nll", 0):.3f} q_term {mm.get("actor_q_term", 0):.3f} {n / (time.time() - t0):.1f} upd/s', flush=True)
      for ms in MP.MILESTONES:
        if n == ms:
          checkpoint.save_named(str(out), str(int(step0) + ms), int(step0) + n, state)
    checkpoint.save_named(str(out), 'final', int(step0) + n, state)
    if arm == 'frozen':
      assert MP._tree_hash(state.q_params) == h0['q'], 'the frozen critic changed'
    MP.write_json(out / 'train_manifest.json', {
        'arm': arm, 'lineage': s, 'source_ckpt': str(src), 'source_ckpt_sha256': MP.sha256(src), 'source_step': int(step0), 'updates': n, 'final_step': int(step0) + n,
        'state_carried': ['policy_params', 'q_params', 'target_q_params', 'policy_optimizer_state', 'q_optimizer_state (clip -> Adam chain)', 'key'],
        'streams': {'critic_seed': MP.CRITIC_STREAM_SEED0 + s, 'actor_seed': MP.ACTOR_STREAM_SEED0 + s, 'select_seed': SELECT_SEED0 + s, 'fast_forwarded_batches': int(step0)},
        'futures': fsum, 'queries_sha256': (MP.sha256(QC.branch_path(s)) if arm not in ('continue', 'frozen') else None), 'sealed_branches_sha256': MP.sha256(SEALED_BRANCHES), 'critic_clip': clip, 'config': MP.config_dump(cfg),
        'params_at_load': h0, 'first_batches': first, 'query_rows_per_4_batches': q_share_hist,
        'params_final': {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)},
        'device': str(jax.devices()[0]), 'jax_version': jax.__version__, 'wall_seconds': time.time() - t0, 'history': hist})
    print(f'{arm} s{s}: {n} updates from step {step0} in {time.time() - t0:.0f} s', flush=True)
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  print(MP.D.evaluate_ckpt(out / 'final.pkl', out, EVAL['policy']), flush=True)


def mode_evaluate_current(args):
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  todo = [(f'current/seed_{s}', lineage_ckpt(s), lineage_ckpt(s).parent) for s in MP.SEEDS if args.lineage is None or int(args.lineage) == s]
  if args.lineage is None and MP.START_CKPT.exists():
    todo.append(('start (reference)', MP.START_CKPT, MP.OUT / 'start_agent'))
  for name, ck, od in todo:
    print(f'== evaluate {name}', flush=True)
    print(MP.D.evaluate_ckpt(ck, od, EVAL['policy']), flush=True)


# --------------------------------------------------------------- readouts
def critic_scores(ckpt, obs31, actions):
  """f(s, a, g) = min over the twin of <phi(s, a), psi(g)> at obs31 [n, 31], actions [n, 8]."""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt); qp = st.q_params

  @jax.jit
  def f(O, A):
    phi, psi = nets.representation_network.apply(qp, O, A)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)
  out = []
  for i in range(0, len(obs31), 4096):
    out.append(np.asarray(f(jnp.asarray(obs31[i:i + 4096], jnp.float32), jnp.asarray(actions[i:i + 4096], jnp.float32)), np.float64))
  return np.concatenate(out)


def auroc(score, label):
  """Rank AUROC (ties count one half); label boolean."""
  score, label = np.asarray(score, float), np.asarray(label, bool)
  pos, neg = score[label], score[~label]
  gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
  return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def _readout_worker(args):
  jobs, worker_seed, ckpt = args
  import build_v6_branch_replay as B
  env, _ = B._worker_env(worker_seed)
  act_fn = MP.mode_policy(ckpt)
  obs, _, _, _ = MP.load_dataset()
  out = []
  for (aid, e, t, d, hz) in jobs:
    o = obs[e, t, :MP.OBS_W]
    a0 = act_fn(o)                                   # the actor's own first step
    r = MP._branch_one(env, act_fn, o, a0, t, hz, MP.HORIZON - int(t))
    xy = r['obs'][1:, :2]
    far = bool(((xy[:, 1] >= CR.DETOUR_Y) & (xy[:, 0] < CR.DETOUR_MAX_X)).any())
    out.append({'anchor_id': int(aid), 'draw': int(d), 'outcome': r['outcome'], 'entered_far': far, 'completed_far': bool(far and r['outcome'] == 'success'), 'steps': int(r['steps'])})
  return out


def own_rollout(ckpt, anchors, ids, workers, seed):
  from multiprocessing import get_context
  jobs = [(int(k), int(anchors.episode[k]), int(anchors.t[k]), d, (MP.HAZARD_SEED0 if d == 0 else QC.HAZARD_SEED1) + int(k)) for k in ids for d in range(QC.DRAWS)]
  parts = [jobs[i::max(1, workers)] for i in range(max(1, workers))]; parts = [p for p in parts if p]
  wargs = [(p, seed + i, str(ckpt)) for i, p in enumerate(parts)]
  with get_context('spawn').Pool(workers) as pool:
    res = pool.map(_readout_worker, wargs)
  rows = [r for part in res for r in part]
  W = anchors.weight[[r['anchor_id'] for r in rows]]
  return {'n_branches': len(rows), 'completed_far': CR.wmean(np.array([r['completed_far'] for r in rows], float), W), 'entered_far': CR.wmean(np.array([r['entered_far'] for r in rows], float), W),
          'success': CR.wmean(np.array([r['outcome'] == 'success' for r in rows], float), W), 'death': CR.wmean(np.array([r['outcome'] == 'death' for r in rows], float), W),
          'timeout': CR.wmean(np.array([r['outcome'] == 'timeout' for r in rows], float), W), 'rows': rows}


def mode_readout(args):
  s = int(args.lineage)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(QC.branch_path(s), allow_pickle=False) as d:
    q_anchor, q_query, q_draw, q_out = d['anchor_id'].astype(np.int64), d['query_id'].astype(np.int64), d['draw'].astype(np.int64), d['outcome'].astype(str)
    q_action = d['query_action'].astype(np.float32); ids = d['start_anchor_ids'].astype(np.int64)
    xy = d['obs_rows'][:, :2]; off, L = d['offset'].astype(np.int64), d['length'].astype(np.int64)
  row_of = np.repeat(np.arange(len(L)), L); after = (np.arange(len(xy)) - np.repeat(off, L)) >= 1
  far = np.bincount(row_of, weights=((xy[:, 1] >= CR.DETOUR_Y) & (xy[:, 0] < CR.DETOUR_MAX_X) & after), minlength=len(L)) > 0
  succ = q_out == 'success'; cfar = far & succ
  # (anchor, query) labels = mean over the two draws
  key = q_anchor * 100 + q_query
  uk, inv = np.unique(key, return_inverse=True)
  lab_s = np.bincount(inv, weights=succ, minlength=len(uk)) / np.bincount(inv, minlength=len(uk))
  lab_f = np.bincount(inv, weights=cfar, minlength=len(uk)) / np.bincount(inv, minlength=len(uk))
  a_of = np.zeros((len(uk), MP.ACTION_DIM), np.float32); a_of[inv] = q_action
  k_of, q_of = uk // 100, uk % 100
  obs31 = anchors.obs31[k_of]
  reset = anchors.t[k_of] == 0
  critics = {'current': lineage_ckpt(s)}
  for arm in ARMS_ALL:
    if (run_dir(arm, s) / 'final.pkl').exists():
      critics[arm] = run_dir(arm, s) / 'final.pkl'
  res = {'lineage': s, 'n_pairs': int(len(uk)), 'n_anchors': int(len(ids)), 'n_reset_anchors': int(len(np.unique(k_of[reset]))), 'critic': {}, 'actor': {}}
  for name, ck in critics.items():
    f = critic_scores(ck, obs31, a_of)
    blk = {}
    for sname, msk in (('reset rows', reset), ('all start anchors', np.ones(len(uk), bool))):
      aucs, argmax_ok, gap, n_mixed = [], [], [], 0
      for k in np.unique(k_of[msk]):
        i = np.flatnonzero((k_of == k) & msk)
        lab = lab_s[i] > 0
        if lab.any() and (~lab).any():
          n_mixed += 1; aucs.append(auroc(f[i], lab))
        j = i[np.argmax(f[i])]; argmax_ok.append(bool(lab_s[j] > 0))
        i0 = i[q_of[i] == 0]; ib = i[np.argmax(lab_s[i] + 1e-3 * lab_f[i])]
        if len(i0):
          gap.append(float(f[ib] - f[i0[0]]))
      W = anchors.weight[np.unique(k_of[msk])]
      blk[sname] = {'anchors': int(len(np.unique(k_of[msk]))), 'anchors_with_both_outcomes': n_mixed, 'within_anchor_auroc_success': (float(np.mean(aucs)) if aucs else None),
                    'share_argmax_query_succeeded': float(np.mean(argmax_ok)), 'f_best_minus_f_logged_mean': float(np.mean(gap)),
                    'f_mean_by_query': {QC.QUERY_NAMES[q]: float(np.mean(f[msk & (q_of == q)])) for q in range(QC.N_QUERIES)},
                    'success_label_by_query': {QC.QUERY_NAMES[q]: float(np.mean(lab_s[msk & (q_of == q)])) for q in range(QC.N_QUERIES)}}
    res['critic'][name] = blk
  # the actor: own closed-loop rollout from the reset anchors, both draws (stage 1's "mode" row for the current agent; re-run here for every policy identically)
  reset_ids = ids[anchors.t[ids] == 0]
  for name, ck in critics.items():
    r = own_rollout(ck, anchors, reset_ids, int(args.workers), 147_000_000 + 100 * s)
    res['actor'][name] = {k: v for k, v in r.items() if k != 'rows'}
    # the mode's torque distance to the best-label query and to the logged torque at the reset anchors
    act = MP.mode_policy(ck)
    dist_best, dist_log = [], []
    for k in reset_ids:
      i = np.flatnonzero(k_of == k); m = act(anchors.obs31[k]); ib = i[np.argmax(lab_s[i] + 1e-3 * lab_f[i])]; i0 = i[q_of[i] == 0][0]
      dist_best.append(float(np.linalg.norm(m - a_of[ib]))); dist_log.append(float(np.linalg.norm(m - a_of[i0])))
    res['actor'][name]['mode_dist_to_best_query_mean'] = float(np.mean(dist_best)); res['actor'][name]['mode_dist_to_logged_mean'] = float(np.mean(dist_log))
  MP.write_json(OUT / f'readout_s{s}.json', res)
  print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1), flush=True)


# ------------------------------------------------------------------ report
def mode_report(args):
  man = MP.read_json(OUT / 'manifest.json')
  E = {}
  for s in MP.SEEDS:
    for arm in ARMS_ALL:
      p = eval_file(run_dir(arm, s))
      if p.exists():
        E[(arm, s)] = MP._episodes(p)
    p = eval_file(lineage_ckpt(s).parent)
    if p.exists():
      E[('current', s)] = MP._episodes(p)
  p = eval_file(MP.OUT / 'start_agent')
  start = MP._episodes(p) if p.exists() else None
  res = {'manifest_sha256': MP.sha256(OUT / 'manifest.json'), 'eval': EVAL, 'headline': {}, 'ledger': {}, 'paired': {}, 'readouts': {}, 'train': {}}
  for (name, s), e in E.items():
    res['headline'][f'{name}/seed_{s}'] = MP._headline(e)
    src = eval_file(run_dir(name, s)) if name in ARMS_ALL else eval_file(lineage_ckpt(s).parent)
    res['ledger'][f'{name}/seed_{s}'] = MP.route_ledger(MP.read_json(src)['episodes'])
  if start is not None:
    res['headline']['start (reference)'] = MP._headline(start)
  complete = [s for s in MP.SEEDS if all((n, s) in E for n in ('control', 'extended', 'current'))]
  if len(complete) == len(MP.SEEDS):
    for key in ('success', 'detour', 'failure', 'timeout'):
      res['paired'][key] = {'extended_minus_control': MP.paired_block({s: E[('extended', s)] for s in complete}, {s: E[('control', s)] for s in complete}, key),
                            'extended_minus_current': MP.paired_block({s: E[('extended', s)] for s in complete}, {s: E[('current', s)] for s in complete}, key),
                            'control_minus_current': MP.paired_block({s: E[('control', s)] for s in complete}, {s: E[('current', s)] for s in complete}, key)}
      cont = [s for s in complete if ('continue', s) in E]
      frz = [s for s in cont if ('frozen', s) in E]
      if len(frz) == len(MP.SEEDS):
        res['paired'][key]['frozen_minus_continue (diagnostic)'] = MP.paired_block({s: E[('frozen', s)] for s in frz}, {s: E[('continue', s)] for s in frz}, key)
        res['paired'][key]['frozen_minus_current (diagnostic)'] = MP.paired_block({s: E[('frozen', s)] for s in frz}, {s: E[('current', s)] for s in frz}, key)
      if len(cont) == len(MP.SEEDS):
        res['paired'][key]['continue_minus_current (reference)'] = MP.paired_block({s: E[('continue', s)] for s in cont}, {s: E[('current', s)] for s in cont}, key)
        res['paired'][key]['extended_minus_continue (reference)'] = MP.paired_block({s: E[('extended', s)] for s in cont}, {s: E[('continue', s)] for s in cont}, key)
        res['paired'][key]['control_minus_continue (reference)'] = MP.paired_block({s: E[('control', s)] for s in cont}, {s: E[('continue', s)] for s in cont}, key)
  for s in MP.SEEDS:
    p = OUT / f'readout_s{s}.json'
    if p.exists():
      res['readouts'][f'seed_{s}'] = MP.read_json(p)
    for arm in ARMS_ALL:
      p = run_dir(arm, s) / 'train_manifest.json'
      if p.exists():
        tm = MP.read_json(p); res['train'][f'{arm}/seed_{s}'] = {'first_batches': tm['first_batches'], 'wall_seconds': tm['wall_seconds'], 'futures': tm['futures'], 'final_loss': tm['history'][-1] if tm['history'] else None}
  # identity check across arms
  res['identity'] = {}
  for s in MP.SEEDS:
    a, b = res['train'].get(f'control/seed_{s}'), res['train'].get(f'extended/seed_{s}')
    if a and b:
      res['identity'][f'seed_{s}'] = {'anchor_ids_equal': a['first_batches']['critic_anchor_ids'] == b['first_batches']['critic_anchor_ids'], 'actor_equal': a['first_batches']['actor'] == b['first_batches']['actor'],
                                      'goals_differ': a['first_batches']['critic_goals'] != b['first_batches']['critic_goals'], 'actions_differ': a['first_batches']['critic_actions'] != b['first_batches']['critic_actions']}
  # judgement
  J = {'complete_lineages': complete}
  if res['paired']:
    ec, ecur = res['paired']['success']['extended_minus_control'], res['paired']['success']['extended_minus_current']
    J['1_extended_beats_control_and_current'] = bool(ec['improvement_rule_met'] and ecur['improvement_rule_met'])
    cr = [res['readouts'][f'seed_{s}']['critic'] for s in MP.SEEDS if f'seed_{s}' in res['readouts'] and 'extended' in res['readouts'][f'seed_{s}']['critic']]
    if cr:
      J['critic_learned_reset_auroc_current_vs_extended'] = [(c['current']['reset rows']['within_anchor_auroc_success'], c['extended']['reset rows']['within_anchor_auroc_success']) for c in cr]
      J['critic_argmax_succeeded_reset_current_vs_extended'] = [(c['current']['reset rows']['share_argmax_query_succeeded'], c['extended']['reset rows']['share_argmax_query_succeeded']) for c in cr]
  res['judgement'] = J
  MP.write_json(OUT / 'report.json', res)
  write_report_md(res)


def write_report_md(res):
  L = ['# Query coverage, stage 2: controlled training comparison (control = q0 futures, extended = q0..q5 futures with the query torque as the critic action, at the start-region anchors)', '',
       f'`manifest.json` (sealed before training), `report.json`.  Fresh paired evaluation draw seed {EVAL["seed"]}, {EVAL["n"]} episodes, policy mode; every policy evaluated once.  '
       'Current = the lineage agent (the clipped CF final, the source checkpoint of both arms).  Rule = mean over the 3 lineages > 2 x seed s.e. and 3 / 3.', '',
       '| policy | success | detour | death | timeout | success (no hazard) | success (hazard) | far entered / completed (ledger) | mean steps |', '|---|---|---|---|---|---|---|---|---|']
  for name, h in res['headline'].items():
    lg = res['ledger'].get(name, {})
    far = f'{lg["far_route"]["n"]} / {lg["far_route"]["success"]}' if lg else '-'
    L.append(f'| {name} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_no_hazard"]:.3f} | {h["success_hazard"]:.3f} | {far} | {h["mean_steps"]:.0f} |')
  if res['paired']:
    L += ['', '| comparison | quantity | per lineage | mean | seed s.e. | same direction | rule met |', '|---|---|---|---|---|---|---|']
    for key, blocks in res['paired'].items():
      for cname, b in blocks.items():
        per = ' / '.join(f'{b["per_seed"][s]["mean"]:+.3f}' for s in sorted(b['per_seed']))
        L.append(f'| {cname} | {key} | {per} | {b["mean"]:+.3f} | {b["seed_se"]:.3f} | {b["seeds_same_direction"]} | {"YES" if b["improvement_rule_met"] else "no"} |')
  if res['identity']:
    L += ['', 'Identity check (first 4 batches): ' + '; '.join(f'seed {k[-1]}: anchors equal {v["anchor_ids_equal"]}, actor equal {v["actor_equal"]}, goals differ {v["goals_differ"]}, actions differ {v["actions_differ"]}' for k, v in res['identity'].items()), '']
  if res['readouts']:
    L += ['## Read-outs on the training anchors (in-sample; secondary)', '',
          '| lineage | critic | stratum | anchors (both outcomes) | within-anchor AUROC (success) | argmax query succeeded | f(best) - f(logged) |', '|---|---|---|---|---|---|---|']
    for sk, R in res['readouts'].items():
      for cname, blk in R['critic'].items():
        for sname, b in blk.items():
          auc = b['within_anchor_auroc_success']
          L.append(f'| {sk} | {cname} | {sname} | {b["anchors"]} ({b["anchors_with_both_outcomes"]}) | {auc if auc is None else round(auc, 3)} | {b["share_argmax_query_succeeded"]:.3f} | {b["f_best_minus_f_logged_mean"]:+.3f} |')
    L += ['', '| lineage | actor | own rollout from the reset anchors: completed far | entered far | success | death | timeout | mode distance to best query / to logged |', '|---|---|---|---|---|---|---|---|']
    for sk, R in res['readouts'].items():
      for aname, b in R['actor'].items():
        L.append(f'| {sk} | {aname} | {b["completed_far"]:.3f} | {b["entered_far"]:.3f} | {b["success"]:.3f} | {b["death"]:.3f} | {b["timeout"]:.3f} | {b["mode_dist_to_best_query_mean"]:.2f} / {b["mode_dist_to_logged_mean"]:.2f} |')
  L += ['', f'Judgement inputs: `{json.dumps(res["judgement"])}`', '']
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('mode', choices=['seal', 'train', 'evaluate_current', 'readout', 'report'])
  ap.add_argument('--lineage', type=int, default=None)
  ap.add_argument('--arm', choices=ARMS_ALL, default='control')
  ap.add_argument('--workers', type=int, default=READOUT_WORKERS)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': mode_seal, 'train': mode_train, 'evaluate_current': mode_evaluate_current, 'readout': mode_readout, 'report': mode_report}[args.mode](args)


if __name__ == '__main__':
  main()
