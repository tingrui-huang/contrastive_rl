"""Localising the start stall of a learned-futures actor (AntMaze V6; user's plan after 282ce17): WHEN in training it appears,
and WHAT the complete actor objective says about it.  A diagnostic, not a selection: no checkpoint is picked by it.

  roll       the saved milestones of a run (init / 10k / 20k / final) and reference checkpoints, rolled with their MODE policy from
             the same fixed start states (repeated_draws/states.npz: 64 reset anchors + 64 independent resets; hazards forced off),
             100 steps each -- the share that leaves the start, the xy displacement over the first steps, the actual torques and
             their deviation from the start agent's and from a reference checkpoint's on the same states.
  objective  on the states visited in the first steps (by the stalled final policy and by a progressing policy, from the roll file):
             for each policy's mode action, the twin-min critic logit under each critic (the run's own and another run's), with
             (a) the task goal and (b) training-time relabelled goals (geometric future rows of the anchor's own logged episode,
             gamma 0.999, truncated); the BC term (-log pi(a_logged | s, g)) at the logged reset rows; the complete objective
             0.95 * (-Q) + 0.05 * NLL exactly as the actor loss weighs it; the share of states where each critic / objective prefers
             the stalled policy's action over the progressing one's.

  python scripts/diag_v6_stall_onset.py roll --ckpts init=... 10k=... 20k=... final=... ref=... --tag s4d1
  python scripts/diag_v6_stall_onset.py objective --tag s4d1 --stall final --prog ref --critics own=... other=...
  short      from ONE pre-collapse actor + its optimizer state (a milestone), short actor-only updates (the recipe's actor loss and
             stream, the critic FROZEN) under each of several critics; the resulting actors rolled from the same start states -- a
             diagnostic of what the critic alone does to that actor, not a training variant.
  python scripts/diag_v6_stall_onset.py short --tag s4d1 --start 20k=... --critics own20k=... ownfinal=... other=... --updates 5000
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
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_repeated_draws as RD  # noqa: E402

OUT = MP.OUT / 'stall_onset'
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W
GAMMA = MP.GAMMA
N_STEPS = 100
LEAVE_DIST = 1.0                 # "left the start": xy distance from the start state's xy
GOAL_DRAWS, GOAL_SEED = 8, 214_000_000
STEPS_OBJ = (0, 5, 10, 20, 30)   # rollout steps whose visited states enter the objective check
SEED_ROWS = 216_000_000


def _nets():
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); return MP.make_nets(cfg), cfg


def _load(ckpt):
  from crl import checkpoint
  _, st = checkpoint.load_checkpoint(ckpt)
  return st


def start_states():
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); g = S['group'].astype(str)
  m = (g == 'reset') | (g == 'indep_reset')
  return {k: S[k][m] for k in ('group', 'anchor_id', 'episode', 't', 'state', 'goal_xy', 'logged')}


# --------------------------------------------------------------------- roll
def mode_roll(args):
  import jax
  import jax.numpy as jnp
  import build_v6_branch_replay as B
  out = OUT / args.tag; out.mkdir(parents=True, exist_ok=True)
  S = start_states(); n = len(S['t']); groups = S['group'].astype(str)
  nets, _ = _nets()
  ck = dict(kv.split('=', 1) for kv in args.ckpts)
  pols = {}
  for name, path in ck.items():
    pp = _load(path).policy_params
    pols[name] = jax.jit(lambda o, pp=pp: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  env, _ = B._worker_env(RD.WORKER_SEED0 + 777)
  names = list(ck)
  xy = np.full((len(names), n, N_STEPS + 1, 2), np.nan, np.float32); st = np.full((len(names), n, N_STEPS + 1, STATE_DIM), np.nan, np.float32)
  act = np.full((len(names), n, N_STEPS, 8), np.nan, np.float32); done_at = np.full((len(names), n), -1, np.int64)
  t0 = time.time()
  for ci, name in enumerate(names):
    for si in range(n):
      RD.restore_forced(env, S['state'][si].astype(np.float64), S['goal_xy'][si].astype(np.float64), int(S['t'][si]), False, False)
      o = np.concatenate([S['state'][si], S['goal_xy'][si]]).astype(np.float32)          # the 31-column recipe observation (restore_forced returns the env's wide obs)
      st[ci, si, 0] = o[:STATE_DIM]; xy[ci, si, 0] = o[:2]
      for j in range(N_STEPS):
        a = np.asarray(pols[name](jnp.asarray(o[None], jnp.float32)), np.float32)[0]
        act[ci, si, j] = a
        o, _, done, _ = env.step(a); o = np.asarray(o[:OBS_W], np.float32)          # the env's wide obs -> the 31-column recipe observation
        st[ci, si, j + 1] = o[:STATE_DIM]; xy[ci, si, j + 1] = o[:2]
        if done:
          done_at[ci, si] = j + 1; break
    print(f'{name}: {n} states x {N_STEPS} steps, {time.time() - t0:.0f} s', flush=True)
  np.savez_compressed(out / 'rollouts.npz', names=np.array(names), xy=xy, state=st, action=act, done_at=done_at, group=groups, anchor_id=S['anchor_id'], episode=S['episode'], t=S['t'], goal_xy=S['goal_xy'],
                      meta=np.asarray(json.dumps({'ckpts': ck, 'hazards': 'forced off', 'steps': N_STEPS, 'policy': 'mode'})))
  # ---- report
  res = {'ckpts': ck, 'n_states': int(n), 'per_ckpt': {}}
  d0 = xy[:, :, 0]
  ref = args.ref if args.ref in names else None
  for ci, name in enumerate(names):
    dist = np.linalg.norm(np.nan_to_num(xy[ci]) - d0[ci][:, None], axis=2)      # [n, steps + 1]
    row = {'left_start_share': {str(k): float(np.nanmean(dist[:, k] > LEAVE_DIST)) for k in (30, 50, 100)},
           'xy_disp_median': {str(k): float(np.nanmedian(dist[:, k])) for k in (10, 30, 50, 100)},
           'xy_disp_p25_p75_at_100': [float(np.nanpercentile(dist[:, 100], 25)), float(np.nanpercentile(dist[:, 100], 75))],
           'mean_abs_torque': float(np.nanmean(np.abs(act[ci]))), 'torque_saturation_share': float(np.nanmean(np.abs(act[ci]) > 0.99)),
           'mean_abs_torque_first10': float(np.nanmean(np.abs(act[ci, :, :10]))), 'done_share': float((done_at[ci] >= 0).mean())}
    for g in ('reset', 'indep_reset'):
      m = groups == g
      row[f'left_start_share_100_{g}'] = float(np.nanmean(dist[m, 100] > LEAVE_DIST)); row[f'xy_disp_median_30_{g}'] = float(np.nanmedian(dist[m, 30]))
    if 'init' in names and name != 'init':
      ii = names.index('init')
      row['first_step_dev_vs_init'] = float(np.nanmean(np.abs(act[ci, :, 0] - act[ii, :, 0])))          # same state (the fixed start), different policy
    if ref and name != ref:
      ri = names.index(ref)
      row['first_step_dev_vs_ref'] = float(np.nanmean(np.abs(act[ci, :, 0] - act[ri, :, 0])))
    res['per_ckpt'][name] = row
  # the action difference on the SAME visited states of each trajectory (first 10 steps of the stalled final vs the reference): re-query the policies
  if ref and 'final' in names:
    fi = names.index('final'); ri = names.index(ref)
    for lab, src in (('final_traj', fi), ('ref_traj', ri)):
      s_vis = st[src, :, :10].reshape(-1, STATE_DIM); ok = np.isfinite(s_vis).all(axis=1)
      o31 = np.concatenate([s_vis[ok], np.repeat(S['goal_xy'], 10, axis=0)[ok]], axis=1).astype(np.float32)
      af = np.asarray(pols['final'](jnp.asarray(o31))); ar = np.asarray(pols[ref](jnp.asarray(o31)))
      res[f'same_state_action_gap_{lab}'] = {'mean_abs': float(np.abs(af - ar).mean()), 'n': int(ok.sum())}
  MP.write_json(out / 'roll_report.json', res)
  L = [f'# Stall onset -- rollouts from the same {n} start states (64 reset anchors + 64 independent resets; hazards off; mode policy; {N_STEPS} steps)', '',
       '| checkpoint | left start (> 1.0) at 30 / 50 / 100 | xy disp median at 10 / 30 / 50 / 100 | left at 100: reset / indep | mean abs torque (first 10) | saturation | first-step dev vs init / vs ref |', '|---|---|---|---|---|---:|---|']
  for name, r in res['per_ckpt'].items():
    L.append(f"| {name} | {r['left_start_share']['30']:.2f} / {r['left_start_share']['50']:.2f} / {r['left_start_share']['100']:.2f} | {r['xy_disp_median']['10']:.2f} / {r['xy_disp_median']['30']:.2f} / {r['xy_disp_median']['50']:.2f} / {r['xy_disp_median']['100']:.2f} | "
             f"{r['left_start_share_100_reset']:.2f} / {r['left_start_share_100_indep_reset']:.2f} | {r['mean_abs_torque']:.3f} ({r['mean_abs_torque_first10']:.3f}) | {r['torque_saturation_share']:.2f} | {r.get('first_step_dev_vs_init', float('nan')):.3f} / {r.get('first_step_dev_vs_ref', float('nan')):.3f} |")
  for k in ('same_state_action_gap_final_traj', 'same_state_action_gap_ref_traj'):
    if k in res:
      L.append(f"\n{k}: mean |a_final - a_ref| on the same visited states = {res[k]['mean_abs']:.3f} (n {res[k]['n']})")
  (out / 'ROLL.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


# ---------------------------------------------------------------- objective
def relabelled_goals(obs, lengths, e, t, rng, k=GOAL_DRAWS):
  """k future goals (xy) of the logged episode e from row t: m ~ Geom(1 - gamma) truncated at the episode end (the actor stream's law)."""
  L = int(lengths[e]); n = L - 1 - int(t)
  if n <= 0:
    return np.repeat(obs[e, L - 1, :2][None], k, axis=0)
  u = rng.random(k); m = np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** n)) / np.log(GAMMA)).astype(np.int64), 1, n)
  return obs[e, int(t) + m, :2]


def mode_objective(args):
  import jax
  import jax.numpy as jnp
  out = OUT / args.tag
  Z = np.load(out / 'rollouts.npz', allow_pickle=False); names = list(Z['names'].astype(str))
  nets, cfg = _nets(); bc = float(cfg.bc_coef)
  crit = dict(kv.split('=', 1) for kv in args.critics)
  qparams = {c: _load(p).q_params for c, p in crit.items()}
  pol = {lab: _load(dict(kv.split('=', 1) for kv in args.policies)[lab]).policy_params for lab in ('stall', 'prog')} if args.policies else None
  if pol is None:
    raise SystemExit('--policies stall=... prog=...')

  @jax.jit
  def q_diag(qp, o31, a):
    q = nets.q_network.apply(qp, o31, a)
    return jnp.diag(jnp.min(q, axis=-1))

  @jax.jit
  def mode_and_nll(pp, o31, a_ref):
    d = nets.policy_network.apply(pp, o31)
    return jnp.tanh(d.loc), -nets.log_prob(d, a_ref)

  obs, act, lengths, _ = MP.load_dataset()
  groups = Z['group'].astype(str); reset = groups == 'reset'
  rng = np.random.default_rng(GOAL_SEED)
  rows = []
  for src in ('stall', 'prog'):                                        # the visited states of each policy's own trajectory
    ci = names.index(args.stall if src == 'stall' else args.prog)
    for j in STEPS_OBJ:
      s = Z['state'][ci, :, j]; ok = np.isfinite(s).all(axis=1) & reset          # relabelled goals need a logged episode: the reset anchors
      for si in np.flatnonzero(ok):
        e, t = int(Z['episode'][si]), int(Z['t'][si])
        g_task = Z['goal_xy'][si].astype(np.float32); g_rel = relabelled_goals(obs, lengths, e, t, rng).astype(np.float32)
        a_log = act[e, t].astype(np.float32)
        for gtype, G in (('task', g_task[None]), ('relabelled', g_rel)):
          o31 = np.concatenate([np.repeat(s[si][None], len(G), axis=0), G], axis=1).astype(np.float32)
          A = {}; NLL = {}
          for lab in ('stall', 'prog'):
            a_mode, nll = mode_and_nll(pol[lab], jnp.asarray(o31), jnp.asarray(np.repeat(a_log[None], len(G), axis=0)))
            A[lab] = np.asarray(a_mode); NLL[lab] = float(np.asarray(nll).mean())
          for c in crit:
            Q = {lab: float(np.asarray(q_diag(qparams[c], jnp.asarray(o31), jnp.asarray(A[lab]))).mean()) for lab in ('stall', 'prog')}
            J = {lab: (1 - bc) * (-Q[lab]) + bc * NLL[lab] for lab in ('stall', 'prog')}         # the actor loss's weighting (alpha 0)
            rows.append({'src': src, 'step': j, 'state': int(si), 'goal': gtype, 'critic': c, 'Q_stall': Q['stall'], 'Q_prog': Q['prog'], 'NLL_stall': NLL['stall'], 'NLL_prog': NLL['prog'],
                         'J_stall': J['stall'], 'J_prog': J['prog'], 'a_gap': float(np.abs(A['stall'] - A['prog']).mean())})
  R = rows
  res = {'critics': crit, 'policies': {'stall': args.stall, 'prog': args.prog}, 'bc_coef': bc, 'goal_draws': GOAL_DRAWS, 'steps': list(STEPS_OBJ), 'summary': {}}
  for src in ('stall', 'prog'):
    for gtype in ('task', 'relabelled'):
      for c in crit:
        sel = [r for r in R if r['src'] == src and r['goal'] == gtype and r['critic'] == c]
        if not sel:
          continue
        dq = np.array([r['Q_stall'] - r['Q_prog'] for r in sel]); dj = np.array([r['J_stall'] - r['J_prog'] for r in sel]); dn = np.array([r['NLL_stall'] - r['NLL_prog'] for r in sel])
        res['summary'][f'{src}|{gtype}|{c}'] = {'n': len(sel), 'Q_stall_mean': float(np.mean([r['Q_stall'] for r in sel])), 'Q_prog_mean': float(np.mean([r['Q_prog'] for r in sel])),
                                                 'critic_prefers_stall_share': float((dq > 0).mean()), 'dQ_mean': float(dq.mean()),
                                                 'NLL_stall_mean': float(np.mean([r['NLL_stall'] for r in sel])), 'NLL_prog_mean': float(np.mean([r['NLL_prog'] for r in sel])), 'dNLL_mean': float(dn.mean()),
                                                 'objective_prefers_stall_share': float((dj < 0).mean()), 'dJ_mean': float(dj.mean()), 'action_gap_mean': float(np.mean([r['a_gap'] for r in sel])),
                                                 'by_step': {str(j): {'critic_prefers_stall_share': float(np.mean([r['Q_stall'] > r['Q_prog'] for r in sel if r['step'] == j])), 'objective_prefers_stall_share': float(np.mean([r['J_stall'] < r['J_prog'] for r in sel if r['step'] == j]))} for j in STEPS_OBJ if any(r['step'] == j for r in sel)}}
  MP.write_json(out / 'objective_report.json', res)
  L = [f'# The complete actor objective at the visited start states: stalled policy ({args.stall}) vs progressing policy ({args.prog}); critics {list(crit)}', '',
       f'Objective per row = {1 - bc:.2f} * (-min-twin Q(s, a_policy, g)) + {bc:.2f} * (-log pi_policy(a_logged | s, g)), the actor loss\'s weighting; goals = the task goal or {GOAL_DRAWS} relabelled future goals of the anchor\'s logged episode; states = the 64 reset anchors\' rollouts at steps {list(STEPS_OBJ)}.', '',
       '| visited states of | goal | critic | n | Q stall / prog (mean) | critic prefers stall | NLL stall / prog | objective prefers stall | mean action gap | prefers stall by step (critic ; objective) |', '|---|---|---|---:|---|---:|---|---:|---:|---|']
  for k, v in res['summary'].items():
    src, gtype, c = k.split('|')
    L.append(f"| {src} | {gtype} | {c} | {v['n']} | {v['Q_stall_mean']:.2f} / {v['Q_prog_mean']:.2f} | {v['critic_prefers_stall_share']:.2f} | {v['NLL_stall_mean']:.2f} / {v['NLL_prog_mean']:.2f} | {v['objective_prefers_stall_share']:.2f} | {v['action_gap_mean']:.3f} | "
             + ' '.join(f"{j}: {b['critic_prefers_stall_share']:.2f};{b['objective_prefers_stall_share']:.2f}" for j, b in v['by_step'].items()) + ' |')
  (out / 'OBJECTIVE.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


# -------------------------------------------------------------------- short
N_SAMPLES_OBJ = 16


def logged_start_rows(obs, act, lengths, n_max=2048, seed=SEED_ROWS):
  """Valid logged rows (t < L - 1) in the start region: every reset row (t = 0) and a seeded sample of start_early rows (t 1-40)."""
  from exp_v6_learned_ett import REGIONS, region_of
  n_ep = len(lengths); rows = []
  for e in range(n_ep):
    L = int(lengths[e])
    for t in range(0, min(41, L - 1)):
      rows.append((e, t))
  rows = np.array(rows); reg = region_of(obs[rows[:, 0], rows[:, 1], :2].astype(np.float64))
  rows = rows[reg == REGIONS.index('start')]
  rng = np.random.default_rng(seed)
  reset = rows[rows[:, 1] == 0]; early = rows[rows[:, 1] > 0]
  early = early[rng.choice(len(early), size=min(n_max - len(reset), len(early)), replace=False)]
  return np.concatenate([reset, early]), len(reset)


def objective_as_trained(nets, cfg, pols, qps, obs, act, lengths, rows, rng, key):
  """Per policy and critic: E_{a ~ pi(s, g)}[min-twin Q(s, a, g)] with N_SAMPLES_OBJ common-random-number samples, the BC NLL of the logged
  action, and the loss (1 - bc) (-E Q) + bc NLL, on the given logged rows with the actor stream's relabelled goals (one draw per row)."""
  import jax
  import jax.numpy as jnp
  e, t = rows[:, 0], rows[:, 1]; s = obs[e, t, :STATE_DIM].astype(np.float32); a_log = act[e, t].astype(np.float32)
  n = np.maximum(lengths[e] - 1 - t, 1); u = rng.random(len(rows)); m = np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** n)) / np.log(GAMMA)).astype(np.int64), 1, n)
  g = obs[e, t + m, :2].astype(np.float32); o31 = np.concatenate([s, g], axis=1)
  bc = float(cfg.bc_coef); keys = jax.random.split(key, N_SAMPLES_OBJ)

  @jax.jit
  def eq(pp, qp, o, k):
    d = nets.policy_network.apply(pp, o)
    def one(kk):
      a = nets.sample(d, kk); q = nets.q_network.apply(qp, o, a); return jnp.diag(jnp.min(q, axis=-1))
    return jnp.mean(jax.vmap(one)(k), axis=0)

  @jax.jit
  def nll(pp, o, a):
    return -nets.log_prob(nets.policy_network.apply(pp, o), a)
  out = {}
  for pname, pp in pols.items():
    row = {'bc_nll_mean': float(np.mean(np.asarray(nll(pp, jnp.asarray(o31), jnp.asarray(a_log)))))}
    for cname, qp in qps.items():
      q = np.asarray(eq(pp, qp, jnp.asarray(o31), keys)); row[f'EQ_{cname}'] = float(q.mean()); row[f'loss_{cname}'] = float((1 - bc) * (-q.mean()) + bc * row['bc_nll_mean'])
      row[f'EQ_{cname}_per_row'] = q
    out[pname] = row
  return out


def mode_short(args):
  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint
  from crl import losses as losses_mod
  import build_v6_branch_replay as B
  out = OUT / args.tag; out.mkdir(parents=True, exist_ok=True)
  name0, path0 = args.start.split('=', 1)
  crit = dict(kv.split('=', 1) for kv in args.critics)
  cfg = MP.recipe_config(int(args.seed), out / '_cfg_short', steps=MP.UPDATES); cfg.batch_size = MP.BATCH; MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  gidx = None if cfg.goal_indices is None else np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    return states[:, jnp.asarray(gidx)] if gidx is not None else states[:, cfg.start_index:cfg.end_index]
  policy_optimizer = optax.adam(cfg.actor_learning_rate, eps=1e-7); q_optimizer = MP.critic_optimizer(cfg, 0.1)
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, policy_optimizer, q_optimizer, separate_actor_batch=True)
  step1 = jax.jit(update_step)                                            # ONE update per call: the critic is restored after every update
  st0 = _load(path0)                                                   # the pre-collapse actor AND its optimizer state
  stream_seed = MP.ACTOR_STREAM_SEED0 + int(args.seed) + 5000
  S = start_states(); n = len(S['t']); env, _ = B._worker_env(RD.WORKER_SEED0 + 778)
  res = {'start': {name0: path0}, 'critics': crit, 'updates': int(args.updates), 'actor_stream_seed': stream_seed, 'per_critic': {}, 'replay_check': {}}
  first_hash = None; results_pp = {}
  for cname, cpath in crit.items():
    stc = _load(cpath)
    actor_stream = MP.ActorStream(cfg, stream_seed)                       # re-created per condition: the SAME batch sequence
    frozen = (stc.q_params, stc.target_q_params, stc.q_optimizer_state); h_frozen = MP._tree_hash(frozen[0])
    state = st0._replace(q_params=frozen[0], target_q_params=frozen[1], q_optimizer_state=frozen[2], key=jax.random.PRNGKey(int(args.seed) + 77))
    t0 = time.time(); hist = []; hashes = []
    for it in range(int(args.updates)):
      b = actor_stream.sample(cfg.batch_size)
      if it == 0:
        hashes.append(MP.arr_hash(b.observation, b.action))
      state, metrics = step1(state, (b, b))                                # the critic batch is the same rows; the critic's update is discarded next
      state = state._replace(q_params=frozen[0], target_q_params=frozen[1], q_optimizer_state=frozen[2])
      if (it + 1) % 1000 == 0:
        hist.append({'update': it + 1, **{k: float(v) for k, v in metrics.items() if k in ('actor_loss', 'bc_nll', 'actor_q_term', 'action_saturation_fraction', 'policy_scale_median', 'pre_tanh_loc_abs_max', 'actor_grad_norm')}})
    assert MP._tree_hash(state.q_params) == h_frozen, 'the critic moved'
    if first_hash is None:
      first_hash = hashes[0]
    res['replay_check'][cname] = {'first_actor_batch_hash': hashes[0], 'same_as_first_condition': bool(hashes[0] == first_hash), 'critic_hash_unchanged': True}
    pp = state.policy_params; results_pp[cname] = pp
    checkpoint.save_named(str(out), f'after_{cname}', int(args.updates), state)   # the updated actor (with the frozen critic it was updated under)
    pol = jax.jit(lambda o, pp=pp: jnp.tanh(nets.policy_network.apply(pp, o).loc))
    dist_end = np.zeros(n); sat = []
    for si in range(n):
      RD.restore_forced(env, S['state'][si].astype(np.float64), S['goal_xy'][si].astype(np.float64), int(S['t'][si]), False, False)
      o = np.concatenate([S['state'][si], S['goal_xy'][si]]).astype(np.float32); x0 = o[:2].copy()
      for j in range(N_STEPS):
        a = np.asarray(pol(jnp.asarray(o[None], jnp.float32)), np.float32)[0]; sat.append(float((np.abs(a) > 0.99).mean()))
        o, _, done, _ = env.step(a); o = np.asarray(o[:OBS_W], np.float32)
        if done:
          break
      dist_end[si] = np.linalg.norm(o[:2] - x0)
    res['per_critic'][cname] = {'left_start_share_100': float((dist_end > LEAVE_DIST).mean()), 'xy_disp_median_100': float(np.median(dist_end)), 'torque_saturation_share': float(np.mean(sat)),
                                'policy_param_delta': float(optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, pp, st0.policy_params))), 'history': hist, 'wall_seconds': time.time() - t0}
    print(f'{cname}: left start {res["per_critic"][cname]["left_start_share_100"]:.2f}, disp {res["per_critic"][cname]["xy_disp_median_100"]:.2f}, saturation {res["per_critic"][cname]["torque_saturation_share"]:.2f}, param delta {res["per_critic"][cname]["policy_param_delta"]:.3f}, first batch {hashes[0][:8]} ({time.time() - t0:.0f} s)', flush=True)
  # ---- the objective as trained, on valid logged rows of the start region (common random numbers), for the input / reference / result policies
  obs, act, lengths, _ = MP.load_dataset(); rows, n_reset = logged_start_rows(obs, act, lengths)
  pols = {name0: st0.policy_params, **{f'after_{c}': pp for c, pp in results_pp.items()}}
  for kv in (args.policies or []):
    lab, pth = kv.split('=', 1); pols[lab] = _load(pth).policy_params
  qps = {c: _load(pth).q_params for c, pth in crit.items()}
  OBJ = objective_as_trained(nets, cfg, pols, qps, obs, act, lengths, rows, np.random.default_rng(SEED_ROWS + 1), jax.random.PRNGKey(SEED_ROWS + 2))
  res['objective_as_trained'] = {'rows': {'n': int(len(rows)), 'n_reset': int(n_reset), 'law': 'relabelled goals of the row\'s own episode (geometric, gamma 0.999, truncated), one draw per row', 'samples_per_row': N_SAMPLES_OBJ},
                                 'per_policy': {pn: {k: v for k, v in r.items() if not k.endswith('_per_row')} for pn, r in OBJ.items()}}
  # paired per-row comparison of the stalled vs the progressing policy's expected Q (if both given)
  if 'stall' in OBJ and 'prog' in OBJ:
    for c in qps:
      d = OBJ['stall'][f'EQ_{c}_per_row'] - OBJ['prog'][f'EQ_{c}_per_row']
      res['objective_as_trained'][f'EQ_stall_minus_prog_{c}'] = {'all_rows_mean': float(d.mean()), 'share_stall_higher': float((d > 0).mean()), 'reset_rows_mean': float(d[:n_reset].mean()), 'reset_share_stall_higher': float((d[:n_reset] > 0).mean())}
  MP.write_json(out / 'short_report.json', res)
  L = [f'# Short actor-only updates from {name0} ({args.updates} updates; the recipe actor loss; the SAME actor batches and keys in every condition; the critic frozen at every update, hash-verified); then rolled from the {n} start states', '',
       '| critic | left start at 100 | xy disp median at 100 | torque saturation | policy param delta | actor loss / bc nll / q term at the end | first actor batch identical |', '|---|---:|---:|---:|---:|---|---|']
  for c, r in res['per_critic'].items():
    h = r['history'][-1] if r['history'] else {}
    L.append(f"| {c} | {r['left_start_share_100']:.2f} | {r['xy_disp_median_100']:.2f} | {r['torque_saturation_share']:.2f} | {r['policy_param_delta']:.2f} | {h.get('actor_loss', float('nan')):.3f} / {h.get('bc_nll', float('nan')):.2f} / {h.get('actor_q_term', float('nan')):.3f} | {res['replay_check'][c]['same_as_first_condition']} |")
  L += ['', f"## The actor objective as trained on {len(rows)} valid logged rows of the start region ({n_reset} reset rows + start_early rows; relabelled goals; {N_SAMPLES_OBJ} sampled actions per row with common random numbers)", '',
        '| policy | BC NLL | ' + ' | '.join(f'E Q under {c} / loss' for c in qps) + ' |', '|---|---:|' + '---|' * len(qps)]
  for pn, r in res['objective_as_trained']['per_policy'].items():
    L.append(f"| {pn} | {r['bc_nll_mean']:.2f} | " + ' | '.join(f"{r[f'EQ_{c}']:.3f} / {r[f'loss_{c}']:.3f}" for c in qps) + ' |')
  for c in qps:
    k = f'EQ_stall_minus_prog_{c}'
    if k in res['objective_as_trained']:
      v = res['objective_as_trained'][k]; L.append(f"\nE Q(stall) - E Q(prog) under {c}: all rows {v['all_rows_mean']:+.3f} (stall higher in {v['share_stall_higher']:.2f}); reset rows {v['reset_rows_mean']:+.3f} ({v['reset_share_stall_higher']:.2f})")
  (out / 'SHORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('roll', 'objective', 'short'))
  ap.add_argument('--tag', required=True)
  ap.add_argument('--ckpts', nargs='*', default=[], help='name=path (roll); include init=... and a ref=...')
  ap.add_argument('--ref', default='ref')
  ap.add_argument('--stall', default='final'); ap.add_argument('--prog', default='ref')
  ap.add_argument('--policies', nargs='*', default=None, help='stall=path prog=path (objective; short: extra policies for the as-trained objective)')
  ap.add_argument('--critics', nargs='*', default=[], help='name=path (objective / short): checkpoints whose critics are read')
  ap.add_argument('--start', default=None, help='name=path (short): the pre-collapse actor checkpoint (with its optimizer state)')
  ap.add_argument('--updates', type=int, default=5000); ap.add_argument('--seed', type=int, default=4)
  args = ap.parse_args(argv)
  {'roll': mode_roll, 'objective': mode_objective, 'short': mode_short}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
