"""AntMaze V6 -- the actor goal-source comparison (user's step 2, 2026-09-21): ONE matched comparison, frozen critic.

Question.  The successful PointMaze run fed the ACTOR's critic term branch-generated future goals (BC on the logs); the AntMaze
contract keeps the actor's critic term and BC on the logged relabeled goals, so the critic learns new futures while the actor
practises reaching the old logs' futures.  Does changing ONLY the actor's critic-term goal source help the current policy?

Design (fixed before training).
  * Per lineage s: the lineage's 30k critic FROZEN in both arms (the joint update step is run and the critic parameters,
    target and critic Adam state are restored after every update); the same actor start (the lineage's 30k actor with its
    Adam state); +30,000 updates, batch 1024, G 4; NCE / actor objective / BC 0.05 / learning rates unchanged.
  * Actor rows in BOTH arms: the pilot anchor pool (`anchors.npz`: 53,747 logged rows with the recipe buffer's own anchor
    law as weights -- the rows that HAVE a counterfactual branch), drawn by weight from one RNG, with the future row m from
    the geometric law (gamma 0.999) truncated at the source's end.  No "nearest-state" matching: every row's counterfactual
    future is its own sealed branch.
      goal_log   the actor's critic term sees the state with the RECORDED future goal of that row (the logged episode's
                 row t + m);
      goal_cf    the actor's critic term sees the state with the COUNTERFACTUAL future goal (the sealed CF branch of that
                 anchor, row m; start-agent continuation -- the critic's own positives).
    The BC term in both arms: the same rows with the LOGGED action and the RECORDED future goal (`bc_transitions`, the
    learner's own separate-BC-rows path; the loss body is unchanged).  No synthetic BC, no BC balancing, no re-weighting.
  * Both arms draw the same anchor sequence and the same m (first-batch hashes); only the critic-term goals differ.
  * Evaluation once on the same fresh draw as stage 2 (seed 6909, 300, mode); paired goal_cf - goal_log, and each against
    the lineage agent (current) and the `frozen` arm (actor-only continuation on the buffer stream with logged goals).

Judgement (user): success up WITHOUT harming walking (timeouts / stalls not up, far-route completion kept) -> worth a mainline
validation; more far-route entries but more timeouts -> not a fix; no improvement -> close the hypothesis "changing the actor's
goal source alone repairs the current recipe" (no ratio sweeps).

  python scripts/exp_v6_actor_goal_source.py seal
  python scripts/exp_v6_actor_goal_source.py train --lineage s --arm goal_log|goal_cf
  python scripts/exp_v6_actor_goal_source.py report
Outputs under outputs/antmaze_branch_replay_p050/exp_mainline_pilot/actor_goal_source/.
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
import exp_v6_query_train as QT  # noqa: E402

OUT = MP.OUT / 'actor_goal_source'
ARMS = ('goal_log', 'goal_cf')
UPDATES = 30_000
EVAL = QT.EVAL
ACTOR_ANCHOR_SEED0 = 148_000_000


def run_dir(arm, s):
  return OUT / arm / f'seed_{s}'


class AnchorActorStream:
  """Actor batches from the anchor pool: anchor by weight, future row by the truncated geometric law of the chosen source;
  returns (critic-term rows with the source goal, BC rows with the recorded goal and the logged action)."""

  def __init__(self, anchors, recorded, branch, source, batch, gamma, seed):
    self.a, self.rec, self.br, self.source, self.B, self.gamma = anchors, recorded, branch, source, int(batch), float(gamma)
    self.cdf = np.cumsum(anchors.weight / anchors.weight.sum()); self.cdf[-1] = 1.0
    self.rng = np.random.default_rng(int(seed))
    self.log_gamma = np.log(gamma)
    self.n_rec = recorded.lengths - 1
    self.n_br = branch.lengths - 1
    assert np.all(self.n_rec >= 1) and np.all(self.n_br >= 1)

  def _m(self, u, n):
    m = np.ceil(np.log1p(-u * (1.0 - self.gamma ** n)) / self.log_gamma).astype(np.int64)
    return np.clip(m, 1, n)

  def sample(self):
    from crl.losses import Transition
    k = np.minimum(np.searchsorted(self.cdf, self.rng.random(self.B), side='right'), self.a.n - 1)
    u = self.rng.random(self.B)
    m_rec = self._m(u, self.n_rec[k]); m_br = self._m(u, self.n_br[k])          # the same uniform -> the same quantile of each source's law
    g_rec = np.stack([self.rec.goal_at(int(kk), int(mm)) for kk, mm in zip(k, m_rec)]).astype(np.float32)
    g_src = g_rec if self.source == 'recorded' else np.stack([self.br.goal_at(int(kk), int(mm)) for kk, mm in zip(k, m_br)]).astype(np.float32)
    S = self.a.state[k]; A = self.a.action[k]
    zero = np.zeros(self.B, np.float32); disc = np.full(self.B, self.gamma, np.float32)
    actor = Transition(observation=np.concatenate([S, g_src], axis=1), action=A, reward=zero, discount=disc, next_observation=np.concatenate([S, g_src], axis=1), next_action=A)
    bc = Transition(observation=np.concatenate([S, g_rec], axis=1), action=A, reward=zero, discount=disc, next_observation=np.concatenate([S, g_rec], axis=1), next_action=A)
    return actor, bc, k, (m_rec if self.source == 'recorded' else m_br)


def build(s, arm):
  import jax.numpy as jnp
  import optax
  from crl import losses as losses_mod
  cfg = MP.recipe_config(s, run_dir(arm, s) / '_cfg')
  cfg.batch_size = MP.BATCH
  cfg.bc_sampling = 'independent'                                  # the learner's separate-BC-rows path (loss body unchanged)
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
  q_opt = MP.critic_optimizer(cfg, float(MP.VARIANTS[CR.RECIPE]['critic_clip']))
  gidx = np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    return states[:, jnp.asarray(gidx)]
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, pol_opt, q_opt, separate_actor_batch=False)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  obs, act, lengths, _ = MP.load_dataset()
  recorded = MP.RecordedFutures(anchors, obs, act)
  branch = MP.BranchFutures(anchors, QT.SEALED_BRANCHES)
  stream = AnchorActorStream(anchors, recorded, branch, 'recorded' if arm == 'goal_log' else 'branch', cfg.batch_size, cfg.discount, ACTOR_ANCHOR_SEED0 + s)
  return cfg, nets, update_step, stream, q_opt


def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6: the actor goal-source comparison -- frozen critic, actor rows from the anchor pool, critic-term goal recorded (goal_log) vs counterfactual (goal_cf); BC on the logged action + recorded goal in both',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'sampling-level diagnostic (user\'s step 2); oracle futures; not a mainline result',
         'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'), 'sealed_branches_sha256': MP.sha256(QT.SEALED_BRANCHES), 'dataset_sha256': MP.sha256(MP.DATASET),
         'lineage_agents': {f'seed_{s}': {'path': str(CR.lineage_ckpt(s)), 'sha256': MP.sha256(CR.lineage_ckpt(s))} for s in MP.SEEDS},
         'arms': {'goal_log': 'actor critic-term goal = the recorded future goal of the anchor row (logged episode row t + m)', 'goal_cf': 'actor critic-term goal = the sealed CF branch future goal of the same anchor (row m)'},
         'shared': 'frozen lineage critic (params, target, critic Adam restored after every update); the lineage actor + its Adam state; anchor pool + weights; one RNG -> same anchors and same future uniform in both arms; '
                   'BC rows = the same anchors with the logged action and the recorded goal (bc_sampling independent path, loss body unchanged); bc 0.05; +30,000 updates; batch 1024; G 4',
         'not_changed': ['no synthetic BC', 'no BC balancing', 'no reset re-weighting', 'no nearest-state matching (every row uses its own branch)', 'networks, losses, learning rates'],
         'updates': UPDATES, 'evaluation': {**EVAL, 'paired_against': ['goal_log', 'current (the lineage agent)', 'frozen (actor-only continuation on the buffer stream, query_coverage/train/frozen)']},
         'judgement': {'improve': 'success up without harming walking (timeouts / stalls not up, far-route completion kept) -> mainline validation',
                       'entries_only': 'more far-route entries but more timeouts -> not a fix', 'none': 'no improvement -> close the hypothesis; no ratio sweeps'}}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


def mode_train(args):
  import jax
  from crl import checkpoint
  s, arm = int(args.lineage), args.arm
  man = MP.read_json(OUT / 'manifest.json')
  src = CR.lineage_ckpt(s)
  assert MP.sha256(src) == man['lineage_agents'][f'seed_{s}']['sha256']
  out = run_dir(arm, s)
  if (out / 'final.pkl').exists() and not args.force:
    print(f'{out} exists', flush=True)
  else:
    out.mkdir(parents=True, exist_ok=True)
    cfg, nets, update_step, stream, q_opt = build(s, arm)
    step0, state = checkpoint.load_checkpoint(src)
    assert jax.tree_util.tree_structure(state.q_optimizer_state) == jax.tree_util.tree_structure(q_opt.init(state.q_params))
    h0 = {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)}
    checkpoint.save_named(str(out), 'init', int(step0), state)

    def _step(st, pair):
      new, m = update_step(st, pair)
      new = new._replace(q_params=st.q_params, target_q_params=st.target_q_params, q_optimizer_state=st.q_optimizer_state)   # critic frozen
      return new, m

    def _multi(state, pair):
      state, metrics = jax.lax.scan(_step, state, pair)
      return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    multi_update = jax.jit(_multi)
    G = MP.G
    n, t0, hist, first = 0, time.time(), [], {}
    for it in range(UPDATES // G):
      acts, bcs, ks, ms = [], [], [], []
      for _ in range(G):
        a, b, k, m = stream.sample(); acts.append(a); bcs.append(b); ks.append(k); ms.append(m)
      if it == 0:
        first = {'anchor_ids': MP.arr_hash(*ks), 'future_rows_m': MP.arr_hash(*ms),
                 'critic_term_goals': MP.arr_hash(*[a.observation[:, MP.STATE_DIM:] for a in acts]), 'bc_rows': MP.arr_hash(*[b.observation for b in bcs], *[b.action for b in bcs]),
                 'goals_differ_from_bc_share': float(np.mean(np.concatenate([np.any(a.observation[:, MP.STATE_DIM:] != b.observation[:, MP.STATE_DIM:], axis=1) for a, b in zip(acts, bcs)])))}
      state, metrics = multi_update(state, (MP._stack(acts), MP._stack(bcs)))
      n += G
      if n % MP.LOG_EVERY == 0 or n == UPDATES:
        mm = {k: float(v) for k, v in metrics.items()}
        hist.append({'update': int(step0) + n, **mm})
        print(f'[{arm} s{s} upd {int(step0) + n:>6}] actor {mm.get("actor_loss", 0):.4f} bc_nll {mm.get("bc_nll", 0):.3f} q_term {mm.get("actor_q_term", 0):.3f} scale {mm.get("policy_scale_median", 0):.3f} {n / (time.time() - t0):.1f} upd/s', flush=True)
      for ms_ in MP.MILESTONES:
        if n == ms_:
          checkpoint.save_named(str(out), str(int(step0) + ms_), int(step0) + n, state)
    checkpoint.save_named(str(out), 'final', int(step0) + n, state)
    assert MP._tree_hash(state.q_params) == h0['q'], 'the frozen critic changed'
    MP.write_json(out / 'train_manifest.json', {
        'arm': arm, 'lineage': s, 'source_ckpt': str(src), 'source_ckpt_sha256': MP.sha256(src), 'source_step': int(step0), 'updates': n, 'final_step': int(step0) + n,
        'critic': 'FROZEN (restored after every update)', 'actor_rows': 'anchor pool by weight; future m by the truncated geometric law of the source', 'stream_seed': ACTOR_ANCHOR_SEED0 + s,
        'bc_rows': 'same anchors, logged action, recorded goal', 'config': MP.config_dump(cfg), 'params_at_load': h0, 'first_batches': first,
        'params_final': {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)},
        'device': str(jax.devices()[0]), 'jax_version': jax.__version__, 'wall_seconds': time.time() - t0, 'history': hist})
    print(f'{arm} s{s}: {n} updates in {time.time() - t0:.0f} s', flush=True)
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  print(MP.D.evaluate_ckpt(out / 'final.pkl', out, EVAL['policy']), flush=True)


def mode_report(args):
  E = {}
  for s in MP.SEEDS:
    for arm in ARMS:
      p = QT.eval_file(run_dir(arm, s))
      if p.exists():
        E[(arm, s)] = MP._episodes(p)
    p = QT.eval_file(CR.lineage_ckpt(s).parent)
    if p.exists():
      E[('current', s)] = MP._episodes(p)
    p = QT.eval_file(QT.run_dir('frozen', s))
    if p.exists():
      E[('frozen', s)] = MP._episodes(p)
  res = {'eval': EVAL, 'headline': {}, 'ledger': {}, 'paired': {}, 'train': {}}
  for (name, s), e in E.items():
    res['headline'][f'{name}/seed_{s}'] = MP._headline(e)
    src = QT.eval_file(run_dir(name, s)) if name in ARMS else (QT.eval_file(QT.run_dir('frozen', s)) if name == 'frozen' else QT.eval_file(CR.lineage_ckpt(s).parent))
    res['ledger'][f'{name}/seed_{s}'] = MP.route_ledger(MP.read_json(src)['episodes'])
  for key in ('success', 'detour', 'failure', 'timeout'):
    for a, b in (('goal_cf', 'goal_log'), ('goal_cf', 'current'), ('goal_log', 'current'), ('goal_cf', 'frozen'), ('goal_log', 'frozen')):
      ok = [s for s in MP.SEEDS if (a, s) in E and (b, s) in E]
      if len(ok) == len(MP.SEEDS):
        res['paired'].setdefault(key, {})[f'{a}_minus_{b}'] = MP.paired_block({s: E[(a, s)] for s in ok}, {s: E[(b, s)] for s in ok}, key)
  for s in MP.SEEDS:
    for arm in ARMS:
      p = run_dir(arm, s) / 'train_manifest.json'
      if p.exists():
        tm = MP.read_json(p); res['train'][f'{arm}/seed_{s}'] = {'first_batches': tm['first_batches'], 'wall_seconds': tm['wall_seconds'], 'final': tm['history'][-1] if tm['history'] else None}
  res['identity'] = {f'seed_{s}': {'anchors_equal': res['train'][f'goal_log/seed_{s}']['first_batches']['anchor_ids'] == res['train'][f'goal_cf/seed_{s}']['first_batches']['anchor_ids'],
                                   'bc_rows_equal': res['train'][f'goal_log/seed_{s}']['first_batches']['bc_rows'] == res['train'][f'goal_cf/seed_{s}']['first_batches']['bc_rows'],
                                   'critic_goals_differ': res['train'][f'goal_log/seed_{s}']['first_batches']['critic_term_goals'] != res['train'][f'goal_cf/seed_{s}']['first_batches']['critic_term_goals']}
                     for s in MP.SEEDS if f'goal_log/seed_{s}' in res['train'] and f'goal_cf/seed_{s}' in res['train']}
  MP.write_json(OUT / 'report.json', res)
  L = ['# Actor goal-source comparison (frozen critic; anchor-pool actor rows; critic-term goal recorded vs counterfactual; BC unchanged)', '',
       f'`manifest.json`, `report.json`.  Evaluation draw seed {EVAL["seed"]}, {EVAL["n"]} episodes, mode; every policy once.  Rule = mean over the 3 lineages > 2 x seed s.e. and 3 / 3.', '',
       '| policy | success | detour | death | timeout | far n / completed / timeouts | no-route | mean steps |', '|---|---|---|---|---|---|---|---|']
  for name, h in res['headline'].items():
    lg = res['ledger'][name]
    L.append(f'| {name} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {lg["far_route"]["n"]} / {lg["far_route"]["success"]} / {lg["far_route"]["timeout"]} | {lg["no_route"]["n"]} | {h["mean_steps"]:.0f} |')
  if res['paired']:
    L += ['', '| comparison | quantity | per lineage | mean | seed s.e. | same direction | rule met |', '|---|---|---|---|---|---|---|']
    for key, blocks in res['paired'].items():
      for cname, b in blocks.items():
        per = ' / '.join(f'{b["per_seed"][s]["mean"]:+.3f}' for s in sorted(b['per_seed']))
        L.append(f'| {cname} | {key} | {per} | {b["mean"]:+.3f} | {b["seed_se"]:.3f} | {b["seeds_same_direction"]} | {"YES" if b["improvement_rule_met"] else "no"} |')
  L += ['', 'Identity (first 4 batches): ' + '; '.join(f'{k}: anchors equal {v["anchors_equal"]}, BC rows equal {v["bc_rows_equal"]}, critic-term goals differ {v["critic_goals_differ"]}' for k, v in res['identity'].items()), '']
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('mode', choices=['seal', 'train', 'report'])
  ap.add_argument('--lineage', type=int, default=0)
  ap.add_argument('--arm', choices=ARMS, default='goal_log')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': mode_seal, 'train': mode_train, 'report': mode_report}[args.mode](args)


if __name__ == '__main__':
  main()
