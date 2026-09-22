"""Local calibration of the critic at the actor's own output actions (AntMaze V6; the user's design after 4b33528; a DIAGNOSTIC
arrangement, not the mainline).

Question: the S ETT gives ~3 % goal-area mass for the actions the collapsed critic reads at ~35 %; if those consequences enter the
critic's training, does the over-estimate come down, and does the actor walk again?

Fixed: the S ETT (ett_one_step_v4s20, the same generator settings as table draw 1), the start agent's continuation, and the training
state = the draw-1 seed-4 run at 20,000 updates (the pre-collapse actor with its critic and both optimizer states).

seal      pre-registration: the query anchors (the start-region anchors: reset t = 0 and start_early t 1-40), split by SOURCE EPISODE
          into a queried half Q and a held-out half H (fixed seed); the queries per Q anchor = the frozen 20k actor's mode under the task
          goal + N_SAMPLES sampled torques from the same actor (fixed key); the control = the same number of extra branches at the
          LOGGED torque (repeated consequences); the readings, written before any result.
generate  --arm query | control: the extra branches through the S ETT (exp_v6_ett_futures.generate_fold with a0 = the query), cross-
          fitted by the anchor's fold, every consequence kept (no success filter, no direction, no simulator rows), one prior context
          draw and one onset draw per branch; query_branches_<arm>.npz.
train     --arm plain | control | query: critic-only continuation from the 20k state for UPDATES_CONT updates (the run's remaining
          budget), the actor restored after every scan (hash-verified), NCE / clip / lr unchanged; the critic stream re-seeded as the
          run's (CRITIC_STREAM_SEED0 + 4) and fast-forwarded 20k batches so the ANCHOR sequence is the run's own; at a Q anchor the future
          is drawn from one of its 1 + n_extra branches with a separate allocation RNG (uniform; the state's weight unchanged, split
          among the branches); the extra branches' first action is the row's action (query) or the logged one (control); 'plain' has no
          extra branches (the allocation RNG still consumed).  critics/<arm>/final.pkl (+ 5000.pkl).

The measurements run with the existing tools: diag_v6_critic_ruler.py ruler (over-estimation at the Q / H reset states and at the
indep_reset states; queried and never-queried actions), diag_v6_stall_onset.py short (actor-only updates under each frozen critic,
the corrected control), exp_v6_repeated_draws.py / diag_v6_ett_crossover.py (the consequences of the new modes), and the pipeline
evaluation (300 episodes, mode policy) for the updated actors.

  python scripts/exp_v6_query_calibration.py seal
  python scripts/exp_v6_query_calibration.py generate --arm query
  python scripts/exp_v6_query_calibration.py train --arm query
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_ett_futures as EF  # noqa: E402
import fit_v6_ett_advice as FA  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'query_calibration'
RUN20K = MP.OUT / 'ett_futures_v4s20' / 'CF' / 'seed_4' / '20000.pkl'      # the training start point (actor + critic + both optimizer states)
BASE_TABLE = MP.OUT / 'ett_futures_v4s20' / 'branches_ett.npz'            # the run's own futures (table draw 1)
ETT_NAME = 'v4s20'
SEED_TRAIN = 4
UPDATES_DONE = 20_000
UPDATES_CONT = 10_000
N_SAMPLES = 3                                                                  # sampled torques per Q anchor (+ the mode)
N_EXTRA = 1 + N_SAMPLES                                                        # extra branches per Q anchor in both arms
SPLIT_SEED = 219_000_000
QUERY_KEY_SEED = 219_000_001
GEN_SEED = 218_000_000                                                         # + 100 * arm + fold
ALLOC_SEED = 219_000_002
ACTOR_STREAM_SEED = MP.ACTOR_STREAM_SEED0 + SEED_TRAIN + 9000                 # the policy is restored after every scan; the actor batch has no effect on the critic
ARMS = ('plain', 'control', 'query')
STATE_DIM, OBS_W, ACTION_DIM, GOAL_DIM = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM, MP.GOAL_DIM


def start_region_anchors(anchors):
  reg = region_of(anchors.state[:, :2].astype(np.float64)); t = anchors.t.astype(np.int64)
  return np.flatnonzero((reg == REGIONS.index('start')) & (t <= 40))


def load_split():
  with np.load(OUT / 'query_split.npz', allow_pickle=False) as d:
    return {k: d[k] for k in d.files if k != 'meta'}, json.loads(str(d['meta']))


# --------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  if (OUT / 'manifest.json').exists() and not args.force:
    print('manifest exists', flush=True); return
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  ks = start_region_anchors(anchors); eps = np.unique(anchors.episode[ks])
  rng = np.random.default_rng(SPLIT_SEED); perm = rng.permutation(eps); q_eps = np.sort(perm[:len(eps) // 2]); h_eps = np.sort(perm[len(eps) // 2:])
  q = ks[np.isin(anchors.episode[ks], q_eps)]; h = ks[np.isin(anchors.episode[ks], h_eps)]
  t = anchors.t.astype(np.int64)
  meta = {'rule': 'start-region anchors (region start, t <= 40), split by source episode, half / half, seed SPLIT_SEED', 'n_start_region': int(len(ks)), 'n_episodes': int(len(eps)),
          'n_Q': int(len(q)), 'n_H': int(len(h)), 'n_Q_reset': int((t[q] == 0).sum()), 'n_H_reset': int((t[h] == 0).sum()), 'weight_share_Q': float(anchors.weight[q].sum() / anchors.weight.sum())}
  np.savez_compressed(OUT / 'query_split.npz', Q=q, H=h, Q_episodes=q_eps, H_episodes=h_eps, meta=np.asarray(json.dumps(meta)))
  man = {'experiment': 'local calibration of the critic at the actor\'s output actions -- a diagnostic arrangement (critic-only continuation from one checkpoint), not the mainline',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(),
         'fixed': {'ett': ETT_NAME, 'ett_models': {f'fold{f}': MP.sha256(MP.OUT / f'ett_one_step_{ETT_NAME}' / f'model_fold{f}.json') for f in range(FA.N_FOLDS)},
                   'advice_generator': {f'fold{f}': MP.sha256(FA.OUT / f'gen3_fold{f}.json') for f in range(FA.N_FOLDS)}, 'continuation': str(MP.START_CKPT), 'continuation_sha256': MP.sha256(MP.START_CKPT),
                   'training_start': str(RUN20K), 'training_start_sha256': MP.sha256(RUN20K), 'base_table': str(BASE_TABLE), 'base_table_sha256': MP.sha256(BASE_TABLE),
                   'recipe': {'variant': EF.BASE_VARIANT, 'overrides': EF.OVERRIDES, 'nce': 'unchanged', 'bc': 0.05, 'actor_rows': 'the logged buffer, unchanged; query actions never enter the actor loss'}},
         'split': meta,
         'queries': {'per_Q_anchor': f'the frozen 20k actor\'s mode under the task goal + {N_SAMPLES} torques sampled from the same actor (fixed key)', 'n_extra_per_anchor': N_EXTRA,
                     'control': f'{N_EXTRA} extra branches at the LOGGED torque per Q anchor (repeated consequences; new prior-context and onset draws)',
                     'filtering': 'none: every consequence kept, no success selection, no direction imposed, no simulator rows', 'generation_seed': f'{GEN_SEED} + 100 x arm + fold'},
         'allocation': 'at a Q anchor the future is drawn from one of its 1 + n_extra branches uniformly (a separate RNG); the anchor weights are unchanged; the anchor sequence is the run\'s own (stream re-seeded and fast-forwarded 20k batches); identical in all three arms',
         'training': {'arms': list(ARMS), 'updates': UPDATES_CONT, 'from': UPDATES_DONE, 'critic_only': 'the policy and its optimizer state restored after every scan of 4 updates (the MC NCE critic loss does not read the policy); hash-verified at the end'},
         'measurements': {'calibration': 'diag_v6_critic_ruler.py ruler on the 64 diagnostic reset states (labelled Q / H by their anchor\'s episode) and the 64 independent reset states (non-anchors); actions: the logged torque, the queried 20k mode (a20k), the never-queried saturated final action (stall), the progressing draw-2 action (prog), the start agent (start); critics: 20k, final, plain, control, query',
                          'actor': 'diag_v6_stall_onset.py short from the 20k actor + Adam state under each frozen critic (5,000 updates, identical batches); leave-the-start / saturation on the 128 start states; the as-trained objective on the 2,048 logged start rows',
                          'new_actions': 'the consequences (simulator + ETT, start continuation, paired draws) and the ruler at the updated actors\' modes: does the actor move to another over-estimated action?',
                          'task': '300 evaluation episodes (mode policy) of each updated actor: success / death / timeout / detour share / detour completion, vs the 20k actor and the original final'},
         'readings': {'calibration_improved': 'at the H reset states (never queried) and at the never-queried saturated action, the query critic\'s over-estimation factor (goal-area mass / real) is at most half the control critic\'s (median over states) AND the control critic does not show the same drop (otherwise it is the extra futures / the continuation, not the action); at the queried a20k action on H states the query critic reads within 2x of the ETT mass',
                      'less_stall': 'leave-the-start on the 128 start states under the query critic exceeds the control critic\'s by >= 0.15 with lower saturation (the corrected control: own critics 0.43-0.49, draw-2 critics 0.83-0.86; the 20k actor 0.60)',
                      'moved_to_another_high': 'the updated actor\'s new mode has an over-estimation factor > 2x (critic / real) at the H states -> static queries do not control the action extrapolation: STOP adding queries',
                      'task_preserved': 'the query-critic actor\'s detour share and detour completion are not below the 20k actor\'s by more than the 300-episode s.e.; a higher success with the detour share collapsed is NOT a fix (the earlier query round: stalls removed, policy pulled back to the shortcut)',
                      'not_a_fix_by_itself': 'a better-calibrated score alone is not success; the mainline verdict needs the joint pipeline on all seeds, which extends the "logged-action futures only" scope and must be recorded as such'},
         'scope_note': 'extends the sampling-level change beyond "replace the logged action\'s futures": critic rows at (s, a_actor) with ETT futures -- disclosed if carried to the mainline'}
  MP.write_json(OUT / 'manifest.json', man)
  print(json.dumps(man['split'], indent=1), flush=True)


# ----------------------------------------------------------------- generate
def query_actions(anchors, q):
  """The frozen 20k actor's mode and N_SAMPLES sampled torques at each Q anchor (task goal); [len(q), N_EXTRA, ACTION_DIM]."""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  pp = checkpoint.load_checkpoint(RUN20K)[1].policy_params
  o31 = jnp.asarray(anchors.obs31[q].astype(np.float32))
  d = nets.policy_network.apply(pp, o31); mode = np.asarray(jnp.tanh(d.loc))
  keys = jax.random.split(jax.random.PRNGKey(QUERY_KEY_SEED), N_SAMPLES)
  samples = [np.asarray(nets.sample(d, k)) for k in keys]
  A = np.stack([mode] + samples, axis=1).astype(np.float32)
  assert A.shape == (len(q), N_EXTRA, ACTION_DIM) and np.all(np.abs(A) <= 1.0)
  return A


def mode_generate(args):
  arm = args.arm; assert arm in ('query', 'control')
  p = OUT / f'query_branches_{arm}.npz'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  EF.set_ett(ETT_NAME)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  S = np.load(FA.EC / 'supervision.npz', allow_pickle=False); fold_anchor = S['fold_anchor'].astype(np.int64)
  sp, _ = load_split(); q = sp['Q']
  A = query_actions(anchors, q) if arm == 'query' else np.repeat(anchors.action[q][:, None, :], N_EXTRA, axis=1).astype(np.float32)
  kind = (['mode'] + ['sample'] * N_SAMPLES) if arm == 'query' else ['logged'] * N_EXTRA
  ks_all = np.repeat(q, N_EXTRA); a0_all = A.reshape(-1, ACTION_DIM); kind_all = np.array(kind * len(q))
  mode = EF._policy_mode_batched()
  t0 = time.time(); parts = []; order_idx = []
  for f in range(FA.N_FOLDS):
    sel = np.flatnonzero(fold_anchor[ks_all] == f)
    if len(sel) == 0:
      continue
    rng = np.random.default_rng(GEN_SEED + 100 * (1 if arm == 'query' else 2) + f)
    parts.append(EF.generate_fold(f, anchors, obs, act, ks_all[sel], rng, mode, a0=a0_all[sel])); order_idx.append(sel)
    print(f'{arm} fold {f}: {len(sel)} branches, {time.time() - t0:.0f} s; outcome ' + json.dumps({k: round(float((parts[-1]["outcome"] == k).mean()), 3) for k in ("success", "death", "timeout")}), flush=True)
  idx = np.concatenate(order_idx); order = np.argsort(idx)                     # back to (anchor, extra) order
  cat = {k: np.concatenate([p_[k] for p_ in parts]) for k in ('length', 'outcome', 'u1', 'u2', 't0_1', 't0_2', 'anchor_id')}
  obs_all = np.concatenate([p_['obs'] for p_ in parts]); act_all = np.concatenate([p_['act'] for p_ in parts]); off_cat = np.concatenate([[0], np.cumsum(cat['length'])[:-1]])
  L = cat['length'][order]; off_new = np.concatenate([[0], np.cumsum(L)[:-1]])
  obs_rows = np.zeros_like(obs_all); act_rows = np.zeros_like(act_all)
  for i, o in enumerate(order):
    obs_rows[off_new[i]:off_new[i] + L[i]] = obs_all[off_cat[o]:off_cat[o] + L[i]]; act_rows[off_new[i]:off_new[i] + L[i]] = act_all[off_cat[o]:off_cat[o] + L[i]]
  assert np.array_equal(cat['anchor_id'][order], ks_all) and np.allclose(act_rows[off_new], a0_all)
  oc = cat['outcome'][order]
  # the branches' own gamma-law masses (the critic's target at (s, a0)), for the record and the ruler's target column
  import exp_v6_repeated_draws as RD
  masses = {k: np.zeros(len(L)) for k in ('reach0.5', 'near2.0', 'goal_area', 'far', 'death_frame')}
  for i in range(len(L)):
    xy = obs_rows[off_new[i]:off_new[i] + L[i], :2].astype(np.float64)
    for k, v in RD.path_masses(xy, anchors.goal_xy[ks_all[i]].astype(np.float64), str(oc[i])).items():
      masses[k][i] = v
  meta = {'arm': arm, 'ett': ETT_NAME, 'continuation_ckpt': str(MP.START_CKPT), 'gen_seed': GEN_SEED, 'n_branches': int(len(L)), 'n_anchors': int(len(q)), 'n_extra_per_anchor': N_EXTRA, 'wall_seconds': time.time() - t0,
          'outcome_counts': {k: int((oc == k).sum()) for k in ('success', 'death', 'timeout')}, 'query_source': (str(RUN20K) if arm == 'query' else 'the logged torque'), 'filtering': 'none',
          'mean_masses': {k: float(v.mean()) for k, v in masses.items()}, 'mean_masses_by_kind': {kk: {k: float(v[kind_all == kk].mean()) for k, v in masses.items()} for kk in set(kind)}}
  np.savez_compressed(p, obs_rows=obs_rows, act_rows=act_rows, offset=off_new, length=L, anchor_id=ks_all, a0=a0_all, kind=kind_all, outcome=oc, u1=cat['u1'][order], u2=cat['u2'][order], t0_1=cat['t0_1'][order], t0_2=cat['t0_2'][order],
                      **masses, meta=np.asarray(json.dumps(meta, sort_keys=True)))
  print(json.dumps({k: meta[k] for k in ('n_branches', 'outcome_counts', 'mean_masses', 'mean_masses_by_kind', 'wall_seconds')}, indent=1), flush=True)


# ------------------------------------------------------------------- train
class QueryTable:
  """The extra branches: contiguous per anchor (anchor_id sorted as generated: each Q anchor's N_EXTRA branches in a row)."""

  def __init__(self, path):
    with np.load(path, allow_pickle=False) as d:
      self.obs_rows, self.act_rows = d['obs_rows'], d['act_rows']; self.offset, self.lengths = d['offset'].astype(np.int64), d['length'].astype(np.int64)
      self.anchor_id, self.a0 = d['anchor_id'].astype(np.int64), d['a0'].astype(np.float32); self.meta = json.loads(str(d['meta']))
    self.n_fut = self.lengths - 1; assert np.all(self.n_fut >= 1)
    n_anchor = int(self.anchor_id.max()) + 1
    self.nb = np.bincount(self.anchor_id, minlength=n_anchor); self.first = np.full(n_anchor, -1, np.int64)
    ch = np.flatnonzero(np.r_[True, self.anchor_id[1:] != self.anchor_id[:-1]]); self.first[self.anchor_id[ch]] = ch
    assert np.all(np.diff(self.anchor_id)[np.diff(self.anchor_id) != 0] > 0), 'branches must be grouped by anchor in increasing order'

  def goal_at(self, b, m):
    return self.obs_rows[self.offset[b] + m, :GOAL_DIM]

  def next_rows(self, b):
    r = self.offset[b] + 1
    return self.obs_rows[r, :STATE_DIM], self.act_rows[r]


class MultiBranchStream(MP.CriticStream):
  """The recipe's critic stream with extra branches at some anchors: the anchor and the future uniform are drawn exactly as the base
  stream's; a separate RNG picks one of the anchor's 1 + n_extra branches (uniform); an extra branch supplies its own first action, its
  future goal (the same uniform through the branch's own truncated geometric law) and its next rows."""

  def __init__(self, anchors, futures, extra, batch, gamma, seed, alloc_seed):
    super().__init__(anchors, futures, batch, gamma, seed)
    self.x = extra; self.alloc = np.random.default_rng(alloc_seed)
    self.nb = np.zeros(anchors.n, np.int64)
    if extra is not None:
      self.nb[:len(extra.nb)] = extra.nb
    self.n_extra_rows = 0

  def sample(self):
    from crl.losses import Transition
    k, m = self.draw(); u = self._u
    r = self.alloc.random(self.B)                                              # always consumed: identical RNG use in every arm
    b = np.floor(r * (1 + self.nb[k])).astype(np.int64)
    goals = np.stack([self.f.goal_at(int(kk), int(mm)) for kk, mm in zip(k, m)]).astype(np.float32)
    nxt = [self.f.next_rows(int(kk)) for kk in k]
    next_state = np.stack([x[0] for x in nxt]).astype(np.float32); next_action = np.stack([x[1] for x in nxt]).astype(np.float32)
    actions = self.a.action[k].astype(np.float32).copy()
    for i in np.flatnonzero(b > 0):
      bi = int(self.x.first[k[i]] + b[i] - 1); n = int(self.x.n_fut[bi])
      mm = int(np.clip(np.ceil(np.log1p(-u[i] * (1.0 - self.gamma ** n)) / self.log_gamma), 1, n))
      goals[i] = self.x.goal_at(bi, mm); actions[i] = self.x.a0[bi]; next_state[i], next_action[i] = self.x.next_rows(bi)
    self.n_extra_rows += int((b > 0).sum())
    obs = np.concatenate([self.a.state[k], goals], axis=1)
    return Transition(observation=obs, action=actions, reward=np.zeros(self.B, np.float32), discount=np.full(self.B, self.gamma, np.float32),
                      next_observation=np.concatenate([next_state, goals], axis=1), next_action=next_action), (k, m)


def mode_train(args):
  import jax
  import optax
  from crl import checkpoint
  from crl import losses as losses_mod
  arm = args.arm; d = OUT / 'critics' / arm
  if (d / 'final.pkl').exists() and not args.force:
    print(f'{d} exists', flush=True); return
  d.mkdir(parents=True, exist_ok=True)
  man = MP.read_json(OUT / 'manifest.json')
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  cfg = MP.recipe_config(SEED_TRAIN, d / '_cfg', steps=MP.UPDATES); cfg.batch_size = MP.BATCH
  for kk, v in EF.OVERRIDES.items():
    if kk != 'critic_clip':
      setattr(cfg, kk, v)
  MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  gidx = None if cfg.goal_indices is None else np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    import jax.numpy as jnp
    return states[:, jnp.asarray(gidx)] if gidx is not None else states[:, cfg.start_index:cfg.end_index]
  policy_optimizer = optax.adam(cfg.actor_learning_rate, eps=1e-7); q_optimizer = MP.critic_optimizer(cfg, float(EF.OVERRIDES['critic_clip']))
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, policy_optimizer, q_optimizer, separate_actor_batch=True)
  step, st0 = checkpoint.load_checkpoint(RUN20K); assert step == UPDATES_DONE, step
  assert MP.sha256(RUN20K) == man['fixed']['training_start_sha256']
  h_p0, h_q0 = MP._tree_hash(st0.policy_params), MP._tree_hash(st0.q_params)
  state = st0
  extra = QueryTable(OUT / f'query_branches_{arm}.npz') if arm != 'plain' else None
  base = MP.BranchFutures(anchors, BASE_TABLE)
  critic_stream = MultiBranchStream(anchors, base, extra, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + SEED_TRAIN, ALLOC_SEED)
  t0 = time.time()
  for _ in range(UPDATES_DONE):
    critic_stream.draw()                                                       # fast-forward: the anchor sequence continues where the run's would
  print(f'stream fast-forwarded {UPDATES_DONE} batches in {time.time() - t0:.1f} s', flush=True)
  actor_stream = MP.ActorStream(cfg, ACTOR_STREAM_SEED)

  def _multi(state, pair):
    state, metrics = jax.lax.scan(update_step, state, pair)
    return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
  multi_update = jax.jit(_multi)
  G = MP.G; n_updates, first, hist = 0, {}, []
  for it in range(UPDATES_CONT // G):
    cbs, ks = [], []
    for _ in range(G):
      tr, (k, m) = critic_stream.sample(); cbs.append(tr); ks.append(k)
    abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
    if it == 0:
      first = {'critic_anchor_ids': MP.arr_hash(*ks), 'critic_goals': MP.arr_hash(*[b.observation[:, STATE_DIM:] for b in cbs]), 'critic_actions': MP.arr_hash(*[b.action for b in cbs])}
    state, metrics = multi_update(state, (MP._stack(cbs), MP._stack(abs_)))
    state = state._replace(policy_params=st0.policy_params, policy_optimizer_state=st0.policy_optimizer_state)   # critic-only
    n_updates += G
    if n_updates % MP.LOG_EVERY == 0:
      mtr = {kk: float(v) for kk, v in metrics.items()}; hist.append({'update': UPDATES_DONE + n_updates, **mtr})
      print(f'[{arm} upd {UPDATES_DONE + n_updates:>6}] critic {mtr.get("critic_loss", 0):.4f} cat_acc {mtr.get("categorical_accuracy", 0):.3f} extra rows so far {critic_stream.n_extra_rows} {n_updates / (time.time() - t0):.1f} upd/s', flush=True)
    if n_updates == 5000:
      checkpoint.save_named(str(d), '5000', UPDATES_DONE + n_updates, state)
  assert MP._tree_hash(state.policy_params) == h_p0, 'the policy moved'
  checkpoint.save_named(str(d), 'final', UPDATES_DONE + n_updates, state)
  MP.write_json(d / 'train_manifest.json', {'arm': arm, 'from': str(RUN20K), 'from_sha256': MP.sha256(RUN20K), 'updates': n_updates, 'critic_only': True, 'policy_hash_unchanged': True, 'q_init': h_q0, 'q_final': MP._tree_hash(state.q_params),
                                             'base_table_sha256': MP.sha256(BASE_TABLE), 'extra_table': (None if extra is None else str(OUT / f'query_branches_{arm}.npz')), 'extra_table_sha256': (None if extra is None else MP.sha256(OUT / f'query_branches_{arm}.npz')),
                                             'extra_rows_seen': int(critic_stream.n_extra_rows), 'extra_row_share': float(critic_stream.n_extra_rows / (n_updates * cfg.batch_size)), 'critic_stream_seed': MP.CRITIC_STREAM_SEED0 + SEED_TRAIN, 'fast_forward': UPDATES_DONE, 'alloc_seed': ALLOC_SEED,
                                             'actor_stream_seed': ACTOR_STREAM_SEED, 'first_batches': first, 'critic_clip': float(EF.OVERRIDES['critic_clip']), 'wall_seconds': time.time() - t0, 'device': str(jax.devices()[0]), 'host': platform.node(), 'history': hist})
  print(f'{arm}: {n_updates} critic-only updates in {time.time() - t0:.0f} s; extra rows {critic_stream.n_extra_rows}', flush=True)


# ---------------------------------------------------------------- evaluate
def mode_evaluate(args):
  """The pipeline's 300-episode evaluation (mode policy) of the given checkpoints on the given seeds; eval/<name>/eval_mean_s<seed>.json."""
  ck = dict(kv.split('=', 1) for kv in args.ckpts); MP.D.EVAL['n'] = 300
  rows = {}
  for seed in args.seeds:
    MP.D.EVAL['seed'] = int(seed)
    for name, path in ck.items():
      od = OUT / 'eval' / name; od.mkdir(parents=True, exist_ok=True)
      summ = MP.D.evaluate_ckpt(Path(path), od, 'mean'); o = summ['overall']
      rows[f'{name}|{seed}'] = {'success': o['success'], 'death': o['failure'], 'timeout': o['timeout'], 'detour': o['routes']['detour']['rate'], 'shortcut': o['routes']['shortcut']['rate'], 'no_route': o['routes']['no_route']['rate'], 'mean_steps': o['mean_steps']}
      ep = MP.read_json(od / f'eval_mean_s{seed}.json').get('episodes', [])
      det = [e for e in ep if e.get('route') == 'detour']
      rows[f'{name}|{seed}']['detour_completion'] = (float(np.mean([bool(e.get('success')) for e in det])) if det else None)
      print(f'{name} s{seed}: ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[f"{name}|{seed}"].items()}), flush=True)
  MP.write_json(OUT / 'eval' / 'summary.json', rows)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('seal', 'generate', 'train', 'evaluate'))
  ap.add_argument('--arm', default=None); ap.add_argument('--force', action='store_true')
  ap.add_argument('--ckpts', nargs='*', default=[]); ap.add_argument('--seeds', nargs='*', default=['8909', '10909'])
  args = ap.parse_args(argv)
  {'seal': mode_seal, 'generate': mode_generate, 'train': mode_train, 'evaluate': mode_evaluate}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
