"""AntMaze V6: natural-trajectory reproduction and first-step replacement for a
frozen agent (the user's small check before the agent-update round is read).

1. Natural draws: the agent's own evaluation loop (eval_rockfall_clock_v6_baseline
   code path, seed 909, mean policy) is re-run in-process; at every reset the
   Ant state (qpos, qvel), the goal, the hidden draws (U1, U2, t0, jitter) and
   the agent's own first torque are captured, and the outcome is checked against
   the stored evaluation rows.
2. Reproduction through the branch-generation path: a generator worker env,
   ``restore`` to the same reset state and goal with the same hidden draws, the
   agent's OWN first torque, then the agent closed-loop (``_continue`` with the
   generator's act_fn).  Should reproduce the natural trajectory.
3. Same start, hidden draws and continuation; only the first torque replaced by
   the replay's query kinds: the BC walker's mode / two samples, the teacher's
   shortcut ('go') and detour torques.
4. Dataset t = 0 anchors (the replay's start states): the agent's own first
   torque vs the recorded / BC torque, agent continuation, paired hazard draws.

  python scripts/diag_v6_first_step_replay.py --agent-ckpt ... --bc-ckpt ... --eval-json ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402
import diag_v6_first_step_crossover as DG  # noqa: E402

OUT_NAME = 'diag_first_step'
VARIANTS = ('own', 'bc_mode', 'bc_sample0', 'bc_sample1', 'teacher_go', 'teacher_detour')


def hidden_of(env):
  return {'active': {z: bool(env._active[z]) for z in (1, 2)}, 't0': {z: int(env._t0[z]) for z in (1, 2)},
          'jitter': {z: np.array(env._drop_jitter[z], np.float64).copy() for z in (1, 2)}}


def set_hidden(env, h):
  env._active = {z: bool(h['active'][z]) for z in (1, 2)}
  env._t0 = {z: int(h['t0'][z]) for z in (1, 2)}
  env._drop_jitter = {z: np.array(h['jitter'][z], np.float64).copy() for z in (1, 2)}


def natural_episodes(args):
  """The evaluation loop with capture; returns per-episode records."""
  import jax.numpy as jnp
  import eval_rockfall_clock_v6_baseline as EV
  ev_args = EV.parse_args(['--ckpt', str(args.agent_ckpt), '--n', str(args.n), '--seed', str(args.seed), '--policy', 'mean',
                           '--p-active-1', str(B.P_ACTIVE), '--p-active-2', str(B.P_ACTIVE)])
  act_mean, _, _ = EV.build_mean_policy(str(args.agent_ckpt), ev_args)
  _, env = EV._configure_env(ev_args, seed=ev_args.seed)
  u = env._env
  recs = []
  for k in range(ev_args.n):
    o = env.reset()
    state = np.concatenate([np.array(u.data.qpos[:B.NQ], np.float64), np.array(u.data.qvel[:B.NV], np.float64)])
    goal_xy = np.array(env._goal_state_full[:2], np.float64)
    h = hidden_of(env)
    a0 = np.asarray(act_mean(jnp.asarray(o[None], jnp.float32))[0], np.float32)
    xy = [o[:2].copy()]; reward = 0.0; info = {}; a = a0
    for step in range(ev_args.horizon):
      o, reward, done, info = env.step(a)
      xy.append(o[:2].copy())
      if done or reward > 0:
        break
      a = np.asarray(act_mean(jnp.asarray(o[None], jnp.float32))[0], np.float32)
    success = bool(info.get('success', reward > 0)); failure = bool(info.get('failure', False))
    xy = np.array(xy)
    recs.append({'episode': k, 'state': state, 'goal_xy': goal_xy, 'hidden': h, 'a0': a0, 'success': success, 'failure': failure, 'timeout': (not success and not failure), 'steps': int(len(xy) - 1),
                 'around': bool(xy[:, 1].max() >= B.DETOUR_Y), 'route': info.get('route'), 'xy': xy})
  return recs


def compare_with_eval(recs, eval_json):
  rows = json.loads(Path(eval_json).read_text(encoding='utf-8'))['episodes']
  n = min(len(rows), len(recs)); same = 0; same_out = 0; same_hidden = 0; mism = []; step_diffs = []
  for k in range(n):
    r, e = recs[k], rows[k]
    out_ok = (bool(e.get('success')) == r['success']) and (bool(e.get('failure')) == r['failure']) and (str(e.get('route')) == str(r['route']))
    hid_ok = (bool(e.get('u1')) == r['hidden']['active'][1]) and (bool(e.get('u2')) == r['hidden']['active'][2]) and (int(e.get('sampled_t0_1', -1)) in (r['hidden']['t0'][1], -1)) and (int(e.get('sampled_t0_2', -1)) in (r['hidden']['t0'][2], -1))
    ok = out_ok and hid_ok and (int(e.get('steps', -1)) in (r['steps'], -1))
    same += int(ok); same_out += int(out_ok); same_hidden += int(hid_ok); step_diffs.append(abs(int(e.get('steps', r['steps'])) - r['steps']))
    if not ok and len(mism) < 10:
      mism.append({'episode': k, 'eval': {kk: e.get(kk) for kk in ('success', 'failure', 'steps', 'route')}, 'rerun': {kk: r[kk] for kk in ('success', 'failure', 'steps', 'route')}})
  return {'n': n, 'same_outcome_route_and_steps': same, 'same_outcome_and_route': same_out, 'same_hidden_draws': same_hidden,
          'step_diff_median': float(np.median(step_diffs)) if step_diffs else None, 'step_diff_max': int(max(step_diffs)) if step_diffs else None,
          'note': 'the stored evaluation ran the policy on a GPU; this rerun is CPU inference -- float differences move step counts through the contact dynamics', 'mismatches': mism}


def first_step_variants(env, teacher, o58, state, goal_xy, bundle_bc, agent_act, rng):
  """The six first torques at a reset state (o58: the worker env's obs after restore)."""
  o31 = o58[:B.STATE_DIM + 2].astype(np.float32)
  loc, scale = bundle_bc['params_fn'](o31[None]); loc, scale = loc[0], scale[0]
  out = {'own': agent_act(o58), 'bc_mode': np.tanh(loc).astype(np.float32)}
  for k in (0, 1):
    out[f'bc_sample{k}'] = np.tanh(loc + scale * rng.standard_normal(loc.shape[-1])).astype(np.float32)
  for intent in ('go', 'detour'):
    teacher.fresh(route='detour' if intent == 'detour' else 'shortcut')
    out[f'teacher_{intent}'] = np.asarray(teacher.act(o58, env.schedule, intent=intent), np.float32)
  return out


def branch(env, teacher, state, goal_xy, hidden, a1, agent_act, t0=0):
  """restore + hidden draws + one torque + the agent closed-loop; returns outcome + xy path."""
  o = B.restore(env, state, goal_xy, t0)
  set_hidden(env, hidden)
  o, reward, done, info = env.step(np.asarray(a1, np.float32))
  xy = [np.array(state[:2]), o[:2].copy()]
  if done:
    return {'success': bool(reward > 0), 'failure': bool(info.get('failure', False)), 'steps': 1, 'xy': np.array(xy), 'route': info.get('route')}
  r = DG._continue(env, teacher, o, t0 + 1, agent_act)
  xy = np.concatenate([np.array(xy), r['obs'][1:, :2]])
  return {'success': bool(r['success']), 'failure': bool(r['failure']), 'steps': int(r['steps']) + 1, 'xy': xy, 'route': None}


def summarize(rows):
  n = len(rows)
  if not n:
    return {'n': 0}
  return {'n': n, 'around': float(np.mean([r['around'] for r in rows])), 'success': float(np.mean([r['success'] for r in rows])),
          'death': float(np.mean([r['failure'] for r in rows])), 'timeout': float(np.mean([(not r['success']) and (not r['failure']) for r in rows])),
          'mean_steps': float(np.mean([r['steps'] for r in rows]))}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--agent-ckpt', required=True)
  ap.add_argument('--bc-ckpt', required=True)
  ap.add_argument('--eval-json', required=True, help='the stored natural evaluation of the agent (same seed)')
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--n-detour', type=int, default=30)
  ap.add_argument('--n-shortcut', type=int, default=15)
  ap.add_argument('--n-anchors', type=int, default=20, help='dataset t=0 anchors per episode route')
  ap.add_argument('--draws', type=int, default=2)
  ap.add_argument('--out', default=OUT_NAME)
  args = ap.parse_args(argv)
  out = B.OUT / args.out; out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  # 1. natural draws with capture
  recs = natural_episodes(args)
  chk = compare_with_eval(recs, args.eval_json)
  det = [r for r in recs if r['around']][:args.n_detour]
  sc = [r for r in recs if not r['around'] and not r['failure']][:args.n_shortcut] + [r for r in recs if not r['around'] and r['failure']][:args.n_shortcut]
  print(f'natural: {len(recs)} episodes, eval rows matched (outcome+route) {chk["same_outcome_and_route"]}/{chk["n"]}, exact steps {chk["same_outcome_route_and_steps"]}; around {sum(r["around"] for r in recs)}; selected {len(det)} detour / {len(sc)} shortcut  ({time.time() - t0:.0f}s)', flush=True)
  # 2/3. generator path: restore + own / replaced first torque, agent continuation, identical hidden draws
  env, teacher = B._worker_env(1)
  agent_act = DG._bc_act_fn(str(args.agent_ckpt))
  bundle_bc = R.policy_bundle(str(args.bc_ckpt))
  rng = np.random.default_rng(args.seed + 77)
  res = {'natural_check': chk, 'selected': {'detour': [r['episode'] for r in det], 'shortcut': [r['episode'] for r in sc]}, 'per_episode': [], 'variants': {}}
  by = {'detour': {v: [] for v in VARIANTS}, 'shortcut': {v: [] for v in VARIANTS}}
  repro = []
  for group, rows in (('detour', det), ('shortcut', sc)):
    for r in rows:
      o58 = B.restore(env, r['state'], r['goal_xy'], 0)
      set_hidden(env, r['hidden'])
      cands = first_step_variants(env, teacher, o58, r['state'], r['goal_xy'], bundle_bc, agent_act, rng)
      rec = {'episode': r['episode'], 'group': group, 'natural': {'success': r['success'], 'failure': r['failure'], 'steps': r['steps'], 'around': r['around']},
             'own_torque_diff_to_eval_a0': float(np.abs(cands['own'] - r['a0']).max()), 'variants': {}}
      for v in VARIANTS:
        b = branch(env, teacher, r['state'], r['goal_xy'], r['hidden'], cands[v], agent_act)
        b['around'] = bool(b['xy'][:, 1].max() >= B.DETOUR_Y)
        if v == 'own':
          m = min(len(b['xy']), len(r['xy']))
          dev = float(np.abs(b['xy'][:m] - r['xy'][:m]).max()) if m else None
          rec['own_reproduction'] = {'same_outcome': (b['success'] == r['success'] and b['failure'] == r['failure']), 'same_steps': b['steps'] == r['steps'],
                                     'same_around': b['around'] == r['around'], 'max_xy_dev_common_prefix': dev, 'steps': (b['steps'], r['steps'])}
          repro.append(rec['own_reproduction'])
        rec['variants'][v] = {k: b[k] for k in ('success', 'failure', 'steps', 'around')}
        rec['variants'][v]['torque_dist_to_own'] = float(np.linalg.norm(cands[v] - cands['own']))
        by[group][v].append(rec['variants'][v])
      res['per_episode'].append(rec)
  res['variants'] = {g: {v: summarize(by[g][v]) for v in VARIANTS} for g in by}
  res['own_reproduction'] = {'n': len(repro), 'same_outcome': int(sum(x['same_outcome'] for x in repro)), 'same_steps': int(sum(x['same_steps'] for x in repro)),
                             'same_around': int(sum(x['same_around'] for x in repro)),
                             'max_xy_dev_median': float(np.median([x['max_xy_dev_common_prefix'] for x in repro if x['max_xy_dev_common_prefix'] is not None])) if repro else None,
                             'max_xy_dev_max': float(np.max([x['max_xy_dev_common_prefix'] for x in repro if x['max_xy_dev_common_prefix'] is not None])) if repro else None}
  print('own-first-step reproduction:', res['own_reproduction'], f'({time.time() - t0:.0f}s)', flush=True)
  # 4. dataset t = 0 anchors: own vs recorded vs BC mode first torque, agent continuation, paired hazard draws
  obs, act, lengths, meta, route = R.load_dataset()
  anch = {'shortcut': [], 'detour': []}
  for e in range(len(route)):
    if len(anch[route[e]]) < args.n_anchors:
      anch[route[e]].append(e)
  res['dataset_t0'] = {}
  for rt, eps in anch.items():
    acc = {v: [] for v in ('own', 'recorded', 'bc_mode')}
    for e in eps:
      st = obs[e, 0, :B.STATE_DIM].astype(np.float64); g = obs[e, 0, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float64)
      o58 = B.restore(env, st, g, 0)
      cands = first_step_variants(env, teacher, o58, st, g, bundle_bc, agent_act, rng)
      cands['recorded'] = act[e, 0].astype(np.float32)
      for d in range(args.draws):
        seed = 131_500_000 + 100 * e + d
        for v in acc:
          DG.reseed(env, seed)
          o = B.restore(env, st, g, 0)     # reset draws the paired hidden values from the reseeded streams
          h = hidden_of(env)
          b = branch(env, teacher, st, g, h, cands[v], agent_act)
          b['around'] = bool(b['xy'][:, 1].max() >= B.DETOUR_Y)
          acc[v].append({k: b[k] for k in ('success', 'failure', 'steps', 'around')})
    res['dataset_t0'][rt] = {v: summarize(acc[v]) for v in acc}
    res['dataset_t0'][rt]['episodes'] = eps
  res['wall_seconds'] = time.time() - t0
  (out / 'results.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (bool(x) if isinstance(x, np.bool_) else float(x))), encoding='utf-8')
  # report
  L = ['# Natural-trajectory reproduction and first-step replacement (fixed d20 vanilla actor, seed 0)', '',
       f'Natural draw seed {args.seed}, {len(recs)} episodes; stored evaluation rows matched on (success, failure, route): {chk["same_outcome_and_route"]} / {chk["n"]}; '
       f'hidden draws (U1, U2, t0) identical: {chk["same_hidden_draws"]} / {chk["n"]}; exact step count: {chk["same_outcome_route_and_steps"]} / {chk["n"]} '
       f'(step difference median {chk["step_diff_median"]}, max {chk["step_diff_max"]}: the stored evaluation used GPU inference, this rerun CPU).',
       f'Selected {len(det)} episodes where the agent went around (max y >= {B.DETOUR_Y}) and {len(sc)} where it did not.', '',
       '## Own first torque through the generator path (restore + same hidden draws + agent continuation)', '',
       f'- same outcome {res["own_reproduction"]["same_outcome"]} / {res["own_reproduction"]["n"]}, same step count {res["own_reproduction"]["same_steps"]}, same around {res["own_reproduction"]["same_around"]}; '
       f'max |xy| deviation over the common prefix: median {res["own_reproduction"]["max_xy_dev_median"]}, max {res["own_reproduction"]["max_xy_dev_max"]}', '',
       '## First-step replacement at the same reset states (same hidden draws, agent continuation from step 2)', '',
       '| group | first torque | n | around | success | death | timeout | mean steps |', '|---|---|---:|---:|---:|---:|---:|---:|']
  for g in ('detour', 'shortcut'):
    for v in VARIANTS:
      s = res['variants'][g][v]
      if s['n']:
        L.append(f'| natural-{g} | {v} | {s["n"]} | {s["around"]:.2f} | {s["success"]:.2f} | {s["death"]:.2f} | {s["timeout"]:.2f} | {s["mean_steps"]:.0f} |')
  L += ['', f'## Dataset t = 0 anchors ({args.n_anchors} per episode route, {args.draws} paired hazard draws), agent continuation', '',
        '| anchor episodes | first torque | n | around | success | death | timeout |', '|---|---|---:|---:|---:|---:|---:|']
  for rt in ('shortcut', 'detour'):
    for v in ('own', 'recorded', 'bc_mode'):
      s = res['dataset_t0'][rt][v]
      L.append(f'| {rt} | {v} | {s["n"]} | {s["around"]:.2f} | {s["success"]:.2f} | {s["death"]:.2f} | {s["timeout"]:.2f} |')
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
