"""Consequence check of the actions the two critics disagree about.

At each of the 16 held-out fork roots, six first-step actions:

  down / right                  the canonical candidates
  actor_s0 / actor_s2           the successful and the failed actor's mode
  argmax_s0 / argmax_s2         each critic's best action over the square

Both critics score all six (same state, canonical goal).  Then every action
is executed from the root, followed by the SAME frozen continuation actor
(the sealed observational rollout actor, sampled with paired keys), in

  * the native environment: K paired replicates per (root, action) -- the
    k-th replicate of every action uses the same env seed, so the wind and
    actuator-noise streams are identical up to a death;
  * the fixed learned ETT: K replicates with paired torch / JAX keys, the
    sealed generation rule (nominal advice from the frozen MDN, continuation
    queries from the same actor).

Recorded per replicate: reach (env reward, i.e. within 2.0 of the goal),
strict success (within 0.5), discounted return (0.95), lower route / shortcut
(the native evaluator's classification), absorbed.  Native interaction is
evaluation only.  Finally, for each action the C replay is searched for
first-step queries / recorded transitions near that action at that root, so
we can say whether the action's consequences ever entered NCE.

Outputs: outputs/pointmaze_action_consequences_v1/{results.json,REPORT.md,
per_root.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
PILOT = ROOT / 'outputs' / 'pointmaze_learned_ett_crl_20260915_v1'
sys.path.insert(0, str(PILOT))
DIAG = ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
OUT = ROOT / 'outputs' / 'pointmaze_action_consequences_v1'
CONFIG = json.loads((SEALED / 'config.json').read_text(encoding='utf-8'))
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
GOAL2 = GOAL[:2].astype(np.float64)
START = np.array([0.5, 3.5])
DISCOUNT = 0.95
REMAINING = 49                 # the roots sit at t = 1 of a 50-step episode
ACTIONS = ('down', 'right', 'actor_s0', 'actor_s2', 'argmax_s0', 'argmax_s2')
CRITICS = {0: SEALED / 'crl' / 'C' / 'final.pkl',
           2: SEALED / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl'}
ROLLOUT_ACTOR = ROOT / CONFIG['inputs']['rollout_actor']
NOMINAL = ROOT / CONFIG['inputs']['nominal']
ETT = ROOT / CONFIG['inputs']['ett_checkpoint']
ENV_SEED_BASE = 61_000_000
KEY_BASE = 61_500_000


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


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def cell(xy):
  return np.clip(np.floor(np.asarray(xy)).astype(int), [0, 0], [8, 4])


def hazardous(xy):
  return ((xy[..., 0] >= 3) & (xy[..., 0] < 6) & (xy[..., 1] >= 3)
          & (xy[..., 1] < 4))


def classify(phys, reward, dead):
  """Per-replicate metrics, the native evaluator's definitions."""
  n = len(phys)
  rows = []
  for i in range(n):
    c = cell(phys[i])
    haz = hazardous(phys[i])
    in12 = (c[:, 0] == 1) & (c[:, 1] == 2)
    below = phys[i, :, 1] < 2.0
    t12 = np.flatnonzero(in12 | below)
    thz = np.flatnonzero(haz)
    t_low = int(t12[0]) if len(t12) else None
    t_hz = int(thz[0]) if len(thz) else None
    if t_low is not None and (t_hz is None or t_low < t_hz):
      route = 'lower'
    elif t_hz is not None:
      route = 'shortcut'
    else:
      route = 'other'
    rows.append({
        'reach': float((reward[i] > 0).any()),
        'strict_success': float(np.linalg.norm(phys[i] - GOAL2, axis=1).min() < 0.5),
        'discounted_return': float(reward[i] @ DISCOUNT ** np.arange(len(reward[i]))),
        'absorbed': float(dead[i, -1]),
        'lower': float(route == 'lower'), 'shortcut': float(route == 'shortcut'),
    })
  return rows


def mean_rows(rows):
  keys = rows[0].keys()
  return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# --------------------------------------------------------------- native
def native_rollouts(roots_phys, actions, actor_params, network, reps, key0):
  """actions: [n_roots, n_actions, 2]. Returns metrics [n_roots, n_actions]."""
  import jax
  import jax.numpy as jnp
  from crl.envs import TwoRouteSwampWindyF4Env

  def _sample(p, o, key):
    d = network.policy_network.apply(p, o)
    return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
  act = jax.jit(jax.vmap(lambda p, o, k: _sample(p, o[None], k)[0],
                         in_axes=(None, 0, 0)))
  n_roots, n_actions = actions.shape[:2]
  results = [[None] * n_actions for _ in range(n_roots)]
  # flatten (root, action, rep) -> one env each; pairing = same seed per rep
  index = [(r, a, k) for r in range(n_roots) for a in range(n_actions)
           for k in range(reps)]
  envs = []
  for (r, a, k) in index:
    e = TwoRouteSwampWindyF4Env(seed=ENV_SEED_BASE + 1000 * r + k,
                                active_prob=0.3)
    e.reset()
    e.state = np.array(roots_phys[r], float)
    e._frames = [e.state.copy()] + [START.copy() for _ in range(3)]
    envs.append(e)
  obs = np.stack([e._get_obs().copy() for e in envs]).astype(np.float32)
  physical = [np.stack([e.state.copy() for e in envs])]
  rewards, dead = [], []
  first = np.stack([actions[r, a] for (r, a, k) in index]).astype(np.float32)
  reps_idx = jnp.asarray([k for (_, _, k) in index])
  for t in range(REMAINING):
    if t == 0:
      a_t = first
    else:
      keys = jax.vmap(lambda i: jax.random.fold_in(
          jax.random.PRNGKey(key0 + t), i))(reps_idx)
      a_t = np.asarray(act(actor_params, jnp.asarray(obs), keys), np.float32)
    step_obs, step_r = [], []
    for e, ai in zip(envs, a_t):
      o, rwd, _, _ = e.step(ai)
      step_obs.append(o)
      step_r.append(rwd)
    obs = np.stack(step_obs).astype(np.float32)
    physical.append(np.stack([e.state.copy() for e in envs]))
    rewards.append(np.asarray(step_r, np.float32))
    dead.append(np.asarray([e.dead for e in envs]))
    if (t + 1) % 10 == 0:
      print(f'  native step {t + 1}/{REMAINING}', flush=True)
  phys = np.stack(physical, 1)
  reward = np.stack(rewards, 1)
  dead = np.stack(dead, 1)
  rows = classify(phys, reward, dead)
  per = {}
  for (r, a, k), row in zip(index, rows):
    per.setdefault((r, a), []).append(row)
  for (r, a), rs in per.items():
    results[r][a] = mean_rows(rs)
  return results


# ------------------------------------------------------------------ ETT
def ett_rollouts(roots_state, actions, actor_params, network, reps, key0):
  import jax
  import jax.numpy as jnp
  import torch
  from learned_ett import load_checkpoint
  from propensity.nominal_policy import load_nominal_policy
  torch.set_num_threads(4)
  model, _ = load_checkpoint(ETT, 'cpu')
  nominal = load_nominal_policy(NOMINAL.parent)

  def _sample(p, o, key):
    d = network.policy_network.apply(p, o)
    return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
  act = jax.jit(jax.vmap(lambda p, o, k: _sample(p, o[None], k)[0],
                         in_axes=(None, 0, 0)))
  n_roots, n_actions = actions.shape[:2]
  index = [(r, a, k) for r in range(n_roots) for a in range(n_actions)
           for k in range(reps)]
  paths = len(index)
  state = np.zeros((paths, REMAINING + 1, 8), np.float32)
  failed = np.zeros((paths, REMAINING + 1), bool)
  reward = np.zeros((paths, REMAINING), np.float32)
  state[:, 0] = np.stack([roots_state[r] for (r, _, _) in index])
  first = np.stack([actions[r, a] for (r, a, k) in index]).astype(np.float32)
  reps_idx = jnp.asarray([k for (_, _, k) in index])
  rng = torch.Generator(device='cpu')
  rng.manual_seed(key0 + 777)
  for t in range(REMAINING):
    current = state[:, t]
    goal = np.broadcast_to(GOAL, current.shape).astype(np.float32)
    # nominal advice: paired per replicate index, like the sealed generator
    xb = np.asarray(nominal.sample(
        jnp.asarray(current), jax.random.PRNGKey(key0 + 3000 + t), 1,
        goal=jnp.asarray(goal)), np.float32)
    if t == 0:
      xq = first
    else:
      keys = jax.vmap(lambda i: jax.random.fold_in(
          jax.random.PRNGKey(key0 + t), i))(reps_idx)
      obs = np.concatenate([current, goal], axis=1)
      xq = np.asarray(act(actor_params, jnp.asarray(obs), keys), np.float32)
    with torch.no_grad():
      succ, event, _, _ = model.sample_live(
          torch.as_tensor(current), torch.as_tensor(xb), torch.as_tensor(xq),
          rng)
    succ = succ.numpy().astype(np.float32)
    event = event.numpy()
    was = failed[:, t]
    persistent = np.concatenate([current[:, :2], current[:, :6]], axis=1)
    succ[was] = persistent[was]
    event[was] = False
    state[:, t + 1] = succ
    failed[:, t + 1] = was | event
    # env reward proxy on the generated position, zero once absorbed
    dist = np.linalg.norm(succ[:, :2] - GOAL2, axis=1)
    reward[:, t] = np.where(failed[:, t + 1], 0.0, (dist < 2.0).astype(np.float32))
  phys = state[:, :, :2].astype(np.float64)
  rows = classify(phys, reward, failed[:, 1:])
  results = [[None] * n_actions for _ in range(n_roots)]
  per = {}
  for (r, a, k), row in zip(index, rows):
    per.setdefault((r, a), []).append(row)
  for (r, a), rs in per.items():
    results[r][a] = mean_rows(rs)
  return results


# -------------------------------------------------------------- replay
def replay_coverage(roots_state, actions, radius_state=0.3, radius_action=0.25):
  """Did (root, action) enter NCE?  Count C-replay transitions near the root
  whose action is near the action, split into synthetic first-step queries
  and everything else, with the outcomes of the episodes they belong to."""
  with np.load(SEALED / 'replay_C.npz', allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths = d['lengths']
    source = d['audit_source']
  n_ep, L = obs.shape[:2]
  goal2 = GOAL2
  ep_reach = np.zeros(n_ep, bool)
  ep_absorbed = np.zeros(n_ep, bool)
  for e in range(n_ep):
    valid = int(lengths[e])
    xy = obs[e, :valid, :2]
    ep_reach[e] = (np.linalg.norm(xy - goal2, axis=1) < 2.0).any()
    if source[e] == 1:
      ep_absorbed[e] = bool(np.all(obs[e, valid - 1, :2] == obs[e, valid - 2, :2])
                            and np.all(obs[e, valid - 1, 2:4] == obs[e, valid - 1, :2]))
  out = []
  xy_all = obs[:, :50, :2]
  for r in range(len(roots_state)):
    near = np.linalg.norm(xy_all - roots_state[r, :2], axis=2) <= radius_state
    row = []
    for a in range(actions.shape[1]):
      close = near & (np.linalg.norm(act[:, :50] - actions[r, a], axis=2)
                      <= radius_action)
      eps, ts = np.nonzero(close)
      synthetic_first = int(np.sum((source[eps] == 1) & (ts == 0)))
      original = int(np.sum(source[eps] == 0))
      synthetic_later = int(np.sum((source[eps] == 1) & (ts > 0)))
      ep_ids = np.unique(eps)
      row.append({
          'transitions': int(close.sum()),
          'synthetic_first_step_queries': synthetic_first,
          'synthetic_later_steps': synthetic_later,
          'original_transitions': original,
          'episodes': int(len(ep_ids)),
          'episode_reach_fraction': (float(ep_reach[ep_ids].mean())
                                     if len(ep_ids) else None),
          'episode_absorbed_fraction': (float(ep_absorbed[ep_ids].mean())
                                        if len(ep_ids) else None),
      })
    out.append(row)
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--reps', type=int, default=64)
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  OUT.mkdir(parents=True, exist_ok=True)

  with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
               / 'root_selection.npz', allow_pickle=False) as d:
    roots = d['state'].astype(np.float32)
    phys = d['physical_state'].astype(float)
  d0 = json.loads((DIAG / 'seed0_heldout.json').read_text())['per_root']
  d2 = json.loads((DIAG / 'seed2_heldout.json').read_text())['per_root']
  n = len(roots)
  actions = np.zeros((n, len(ACTIONS), 2), np.float32)
  for i in range(n):
    actions[i, 0] = [0.0, -1.0]
    actions[i, 1] = [1.0, 0.0]
    actions[i, 2] = d0[i]['mode_action']
    actions[i, 3] = d2[i]['mode_action']
    actions[i, 4] = d0[i]['q_grid_argmax_action']
    actions[i, 5] = d2[i]['q_grid_argmax_action']

  network = make_network()
  obs = np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1
                       ).astype(np.float32)

  # both critics score all six actions at every root
  @jax.jit
  def score(q_params, o, a):
    phi, psi = network.representation_network.apply(q_params, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]
  scores = {}
  for cs, path in CRITICS.items():
    _, st = checkpoint.load_checkpoint(path)
    s = np.zeros((n, len(ACTIONS)))
    for a in range(len(ACTIONS)):
      s[:, a] = np.asarray(score(st.q_params, jnp.asarray(obs),
                                 jnp.asarray(actions[:, a])))
    scores[cs] = s
  print('critic scores done', flush=True)

  _, actor_state = checkpoint.load_checkpoint(ROLLOUT_ACTOR)
  actor_params = actor_state.policy_params
  print(f'native rollouts: {n} roots x {len(ACTIONS)} actions x {args.reps} '
        f'paired replicates, continuation = sealed rollout actor', flush=True)
  native = native_rollouts(phys, actions, actor_params, network, args.reps,
                           KEY_BASE)
  print('ETT rollouts', flush=True)
  ett = ett_rollouts(roots, actions, actor_params, network, args.reps,
                     KEY_BASE)
  print('replay coverage', flush=True)
  cov = replay_coverage(roots, actions)

  metrics = ('reach', 'strict_success', 'discounted_return', 'lower',
             'shortcut', 'absorbed')
  per_root = []
  for i in range(n):
    row = {'root': i, 'xy': phys[i], 'actions': {}}
    for a, name in enumerate(ACTIONS):
      row['actions'][name] = {
          'action': actions[i, a],
          'q_s0': float(scores[0][i, a]), 'q_s2': float(scores[2][i, a]),
          'native': native[i][a], 'ett': ett[i][a], 'replay': cov[i][a]}
    per_root.append(row)
  summary = {}
  for a, name in enumerate(ACTIONS):
    summary[name] = {
        'mean_action': actions[:, a].mean(0),
        'q_s0_mean': float(scores[0][:, a].mean()),
        'q_s2_mean': float(scores[2][:, a].mean()),
        'q_s0_rank_mean': float(np.mean([
            (np.argsort(-scores[0][i]).tolist().index(a) + 1) for i in range(n)])),
        'q_s2_rank_mean': float(np.mean([
            (np.argsort(-scores[2][i]).tolist().index(a) + 1) for i in range(n)])),
        'native': {m: float(np.mean([native[i][a][m] for i in range(n)]))
                   for m in metrics},
        'ett': {m: float(np.mean([ett[i][a][m] for i in range(n)]))
                for m in metrics},
        'replay': {k: float(np.mean([cov[i][a][k] for i in range(n)]))
                   for k in ('transitions', 'synthetic_first_step_queries',
                             'synthetic_later_steps', 'original_transitions')},
        'replay_episode_reach_fraction': float(np.mean([
            cov[i][a]['episode_reach_fraction'] for i in range(n)
            if cov[i][a]['episode_reach_fraction'] is not None] or [np.nan])),
    }
  # paired per-root comparisons between the critic-preferred actions and down
  def paired(name_a, name_b, m, source):
    ia, ib = ACTIONS.index(name_a), ACTIONS.index(name_b)
    src = native if source == 'native' else ett
    diffs = np.array([src[i][ia][m] - src[i][ib][m] for i in range(n)])
    return {'mean_diff': float(diffs.mean()),
            'roots_a_better': int((diffs > 0).sum()),
            'roots_b_better': int((diffs < 0).sum())}
  comparisons = {}
  for src in ('native', 'ett'):
    for m in ('reach', 'discounted_return', 'lower', 'absorbed'):
      comparisons[f'{src}:{m}:argmax_s2_vs_down'] = paired('argmax_s2', 'down', m, src)
      comparisons[f'{src}:{m}:argmax_s0_vs_down'] = paired('argmax_s0', 'down', m, src)
      comparisons[f'{src}:{m}:actor_s2_vs_actor_s0'] = paired('actor_s2', 'actor_s0', m, src)
  result = {'reps': args.reps, 'continuation_actor': str(ROLLOUT_ACTOR.relative_to(ROOT)),
            'critics': {k: str(v.relative_to(ROOT)) for k, v in CRITICS.items()},
            'actions': list(ACTIONS), 'summary': summary,
            'comparisons': comparisons, 'per_root': per_root}
  (OUT / 'results.json').write_text(json.dumps(plain(result), indent=2),
                                    encoding='utf-8')
  with open(OUT / 'per_root.csv', 'w', encoding='utf-8') as f:
    f.write('root,x,y,action,ax,ay,q_s0,q_s2,'
            + ','.join(f'native_{m}' for m in metrics) + ','
            + ','.join(f'ett_{m}' for m in metrics)
            + ',replay_transitions,replay_first_queries\n')
    for row in per_root:
      for name, v in row['actions'].items():
        f.write(f'{row["root"]},{row["xy"][0]:.3f},{row["xy"][1]:.3f},{name},'
                f'{v["action"][0]:.3f},{v["action"][1]:.3f},{v["q_s0"]:.3f},'
                f'{v["q_s2"]:.3f},'
                + ','.join(f'{v["native"][m]:.3f}' for m in metrics) + ','
                + ','.join(f'{v["ett"][m]:.3f}' for m in metrics)
                + f',{v["replay"]["transitions"]},'
                  f'{v["replay"]["synthetic_first_step_queries"]}\n')
  # report
  lines = ['# Consequence check of the six fork actions', '',
           f'16 held-out roots, {args.reps} paired replicates per (root, '
           'action), continuation = the sealed observational rollout actor '
           '(sampled, paired keys), horizon 49. Native = real environment '
           '(evaluation only); ETT = the fixed learned model with the sealed '
           'generation rule. Scores are phi.psi logits of the two critics '
           '(seed 0 = successful arm, seed 2 = failed arm); rank = mean rank '
           'of the action among the six (1 = best).', '',
           '| action | mean action | q s0 (rank) | q s2 (rank) | NATIVE reach '
           '| strict | disc.ret | lower | shortcut | absorbed | ETT reach | '
           'strict | disc.ret | lower | shortcut | absorbed |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for name in ACTIONS:
    s = summary[name]
    nv, et = s['native'], s['ett']
    lines.append(
        f'| {name} | ({s["mean_action"][0]:+.2f},{s["mean_action"][1]:+.2f}) | '
        f'{s["q_s0_mean"]:.3f} ({s["q_s0_rank_mean"]:.1f}) | '
        f'{s["q_s2_mean"]:.3f} ({s["q_s2_rank_mean"]:.1f}) | '
        f'{nv["reach"]:.3f} | {nv["strict_success"]:.3f} | '
        f'{nv["discounted_return"]:.2f} | {nv["lower"]:.3f} | '
        f'{nv["shortcut"]:.3f} | {nv["absorbed"]:.3f} | '
        f'{et["reach"]:.3f} | {et["strict_success"]:.3f} | '
        f'{et["discounted_return"]:.2f} | {et["lower"]:.3f} | '
        f'{et["shortcut"]:.3f} | {et["absorbed"]:.3f} |')
  lines += ['', '## Paired per-root comparisons (mean difference; roots where '
            'the first action is better / worse)', '',
            '| comparison | metric | native | ett |', '|---|---|---|---|']
  for pair in ('argmax_s2_vs_down', 'argmax_s0_vs_down', 'actor_s2_vs_actor_s0'):
    for m in ('reach', 'discounted_return', 'lower', 'absorbed'):
      a = comparisons[f'native:{m}:{pair}']
      b = comparisons[f'ett:{m}:{pair}']
      lines.append(f'| {pair} | {m} | {a["mean_diff"]:+.3f} '
                   f'({a["roots_a_better"]}/{a["roots_b_better"]}) | '
                   f'{b["mean_diff"]:+.3f} ({b["roots_a_better"]}/'
                   f'{b["roots_b_better"]}) |')
  lines += ['', '## Did these actions enter NCE? C-replay transitions within '
            '0.3 of the root with an action within 0.25 of the action (mean '
            'over roots)', '',
            '| action | transitions | synthetic first-step queries | synthetic '
            'later steps | original transitions | reach fraction of those '
            'episodes |', '|---|---:|---:|---:|---:|---:|']
  for name in ACTIONS:
    s = summary[name]['replay']
    lines.append(f'| {name} | {s["transitions"]:.0f} | '
                 f'{s["synthetic_first_step_queries"]:.0f} | '
                 f'{s["synthetic_later_steps"]:.0f} | '
                 f'{s["original_transitions"]:.0f} | '
                 f'{summary[name]["replay_episode_reach_fraction"]:.3f} |')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines), flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
