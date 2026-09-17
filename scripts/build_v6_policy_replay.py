"""AntMaze V6 round 1: branch replay whose continuation is the frozen deployment
policy itself (oracle model), the critic gate on repeated single-step outcomes,
and the validation of an updated actor under the same continuation.

The single-step diagnostic (`diag_v6_first_step_crossover.py single`) showed
that a query torque labelled by the donor's direction (north / east) carries no
consistent advantage once the future is what a fixed blind continuation does
from the resulting state.  That does not show that CONCRETE torques carry no
state-conditioned advantage: at one state candidate A may lead somewhere
better, at another candidate B, and a critic that sees the state can pick A
here and B there without any torque ever "meaning north".  The original CRL
actor optimises exactly that -- f(s, a, g) for the state, the action and the
goal at hand, not a ranking of action families.  This script builds the
replay that asks that question honestly and the two checks the user's plan
puts on it:

  build     for recorded anchor states (the start region, the turning process,
            the north leg, the early shortcut, plus a sparse cover of the
            whole maze), several candidate torques per state -- the recorded
            torque and samples from the frozen deployment policy's own
            tanh-normal at that state -- each executed ONCE, after which the
            same frozen policy acts closed-loop (its mode, the evaluation
            convention) from whatever state results, to success, death or the
            horizon (zero-torque hold after reaching, the replay's absorbing
            goal).  Hazards, clocks and rock jitter are redrawn from the priors
            per draw and PAIRED across the candidates of one anchor; the same
            (state, torque) key is branched several times (fresh draws) so the
            replay carries the expectation over consequences rather than one
            realised future per key.  Successes and failures are all kept.
            Anchor = row 0 of every path; the dataset's gxy contract.  A
            disjoint set of held-out episodes gives the gate its states,
            with more candidates (recorded, the policy mode, three samples)
            and four draws each.
  gate      on the held-out anchors: are the between-candidate outcome
            differences reproducible across draws at all (split-half sign
            agreement, the ceiling), and does a critic's f(s, a, g) order
            the candidates the way the validated differences do?  Also the
            critic's pick: the realised goal-frame probability of its argmax
            candidate against the candidate mean and the best candidate.
            Baselines: chance, the frozen policy's own log-likelihood of the
            candidate, vanilla critics trained on the recorded futures.
  validate  for an updated actor checkpoint: at the held-out anchors its mode
            torque is executed once and the OLD frozen policy continues (the
            continuation the replay was built under); paired fresh draws
            against the old policy's mode torque and the recorded torque at
            the same state.  Reads the interventional single-step gain the
            actor update actually bought, by state set, plus the actor's
            torque manifold (distance to the old mode, saturation, its own
            critic's f) -- the deployment evaluation of the full policy is the
            driver's usual 300-episode run.

Oracle stage: the simulator is the model; not an offline result.

  V6_DATASET_STEM=antmaze_rockfall_clock_v6_p050 V6_P_ACTIVE=0.5 V6_BRANCH_OUT=... \\
    python scripts/build_v6_policy_replay.py build --cont-ckpt <pure-BC final.pkl>
    python scripts/build_v6_policy_replay.py gate --critics <ckpt ...> --cont-ckpt <...>
    python scripts/build_v6_policy_replay.py validate --cont-ckpt <...> --actors <ckpt ...> --tag r1
"""
from __future__ import annotations

import argparse
import json
import os

for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import diag_v6_first_step_crossover as DG  # noqa: E402

GAMMA, RADIUS = 0.999, 0.5
R1_SEED = 707_000
HOLDOUT_ID0 = 1_000_000
DENSE_SETS = ('start_early', 'turn', 'start_late', 'north_leg', 'shortcut_early')
SET_LABEL = {'start_early': 'start (x<2, y<2, t<=5)', 'turn': 'turning (detour eps, x<2, y<2, t>5)',
             'start_late': 'start region late (shortcut eps, x<2, y<2, t>5)', 'north_leg': 'north leg (x<2, 2<=y<6)',
             'shortcut_early': 'shortcut early (2<=x<5, y<2)', 'general': 'whole maze (every k-th row)'}
DENSE_N = {'start_early': 400, 'turn': 300, 'start_late': 200, 'north_leg': 300, 'shortcut_early': 300}
EPS_P = 1e-3      # floor inside the log ratio (P_goal is exactly 0 on death)


# ------------------------------------------------------------------ states
def load_dataset():
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, act, lengths, meta = d['obs'], d['act'], d['lengths'], str(d['meta'])
  with np.load(B.SIDECAR, allow_pickle=True) as sc:
    route_rec = sc['route_realized'].astype(str)
  return obs, act, lengths, meta, route_rec


def split_episodes(route_rec, frac, rng):
  """Held-out episodes, stratified by realised route: their rows are anchors of
  the gate / validation sets only."""
  held = np.zeros(len(route_rec), bool)
  for name in ('detour', 'shortcut'):
    idx = np.flatnonzero(route_rec == name)
    k = max(1, int(round(frac * len(idx))))
    held[rng.choice(idx, size=k, replace=False)] = True
  return held


def dense_masks(obs, lengths, route_rec, eps_mask):
  L = obs.shape[1]
  valid = (np.arange(L)[None, :] < (lengths[:, None] - 1)) & eps_mask[:, None]
  x, y, t = obs[:, :, 0], obs[:, :, 1], np.arange(L)[None, :]
  det = (route_rec == 'detour')[:, None]
  return {'start_early': valid & (x < 2) & (y < 2) & (t <= 5),
          'turn': valid & det & (x < 2) & (y < 2) & (t > 5),
          'start_late': valid & ~det & (x < 2) & (y < 2) & (t > 5),
          'north_leg': valid & det & (x < 2) & (y >= 2) & (y < 6),
          'shortcut_early': valid & ~det & (x >= 2) & (x < 5) & (y < 2)}


def pick_rows(mask, n, rng):
  ee, tt = np.nonzero(mask)
  if not len(ee):
    return []
  sel = rng.choice(len(ee), size=min(n, len(ee)), replace=False)
  return [(int(ee[j]), int(tt[j])) for j in sel]


def general_rows(lengths, eps_mask, every, rng, n=None):
  rows = [(int(e), int(t)) for e in np.flatnonzero(eps_mask) for t in range(0, int(lengths[e]) - 1, every)]
  if n is not None and len(rows) > n:
    rows = [rows[j] for j in rng.choice(len(rows), size=n, replace=False)]
  return rows


# ------------------------------------------------------------------ policy
def policy_bundle(ckpt):
  """The frozen deployment policy: (loc, scale) at a batch of 31-column obs,
  the critic it carries, and the network handles."""
  import jax
  import jax.numpy as jnp
  import run_v6_branch_replay as D
  from crl import checkpoint, networks
  from crl import envs as envs_mod
  cfg = D.base_config(0, B.DATASET, 1, B.OUT / '_cfg_policy_replay')
  envs_mod.make_env(cfg.env_name, cfg, seed=1)
  nets = networks.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, repr_dim=int(cfg.repr_dim),
      repr_norm=cfg.repr_norm, repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = checkpoint.load_checkpoint(ckpt)

  @jax.jit
  def _params(pp, o):
    p = nets.policy_network.apply(pp, o)
    return p.loc, p.scale

  def params_fn(o, pp=None):
    loc, scale = [], []
    for k in range(0, len(o), 4096):
      l_, s_ = _params(st.policy_params if pp is None else pp, jnp.asarray(o[k:k + 4096, :B.STATE_DIM + 2], jnp.float32))
      loc.append(np.asarray(l_)); scale.append(np.asarray(s_))
    return np.concatenate(loc), np.concatenate(scale)
  return {'state': st, 'nets': nets, 'params_fn': params_fn, 'cfg': cfg}


def candidates_for(loc, scale, a_rec, anchor_id, kinds, rng_seed):
  """Candidate torques at one anchor: 'recorded', 'mode', 'sample<k>'."""
  out = []
  rng = np.random.default_rng(rng_seed + 7919 * anchor_id)
  for kind in kinds:
    if kind == 'recorded':
      a = np.asarray(a_rec, np.float32)
    elif kind == 'mode':
      a = np.tanh(loc).astype(np.float32)
    else:
      eps = rng.standard_normal(loc.shape[-1])
      a = np.tanh(loc + scale * eps).astype(np.float32)
    out.append((kind, a))
  return out


# ----------------------------------------------------------------- workers
def _run_jobs(args):
  """Worker: one query torque, then the frozen policy closed-loop.  Job =
  (anchor_id, episode, t, set, cand, action, draw, pair_seed, keep_path)."""
  jobs, seed, cont_ckpt = args
  env, teacher = B._worker_env(seed)
  act_fn = DG._bc_act_fn(cont_ckpt)
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs = d['obs']
  out = []
  for (aid, e, t, sset, cand, action, draw, pair_seed, keep_path) in jobs:
    state = obs[e, t, :B.STATE_DIM].astype(np.float64)
    goal_xy = obs[e, t, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float64)
    anchor_row = obs[e, t, :B.STATE_DIM + 2].astype(np.float32)
    a1 = np.asarray(action, np.float32)
    DG.reseed(env, int(pair_seed))
    o = B.restore(env, state, goal_xy, t)
    o, reward, done, info = env.step(a1)                         # the ONE query torque
    if done:
      path_o = np.stack([anchor_row, o[:B.STATE_DIM + 2].astype(np.float32)])
      path_a = np.stack([a1, np.zeros(B.ACTION_DIM, np.float32)])
      r = {'steps': 1, 'reach_step': (1 if reward > 0 else None), 'success': bool(reward > 0),
           'failure': bool(info.get('failure', False)),
           'u1': bool(env.privileged_rockfall_active_1), 'u2': bool(env.privileged_rockfall_active_2)}
    else:
      r = DG._continue(env, teacher, o, t + 1, act_fn)
      path_o = np.concatenate([anchor_row[None], r['obs'].astype(np.float32)])
      path_a = np.concatenate([a1[None], r['act'].astype(np.float32)])
      r['steps'] += 1
      if r['reach_step'] is not None:
        r['reach_step'] += 1
    rec = {'anchor_id': int(aid), 'episode': int(e), 't': int(t), 'set': sset, 'cand': cand, 'draw': int(draw),
           'pair_seed': int(pair_seed), 'obs0': anchor_row, 'act0': a1, 'steps': int(r['steps']),
           'reach_step': r['reach_step'], 'success': bool(r['success']), 'failure': bool(r['failure']),
           'u1': bool(r['u1']), 'u2': bool(r['u2']), 'max_y': float(path_o[:, 1].max()),
           'final_xy': path_o[-1, :2].tolist(),
           'p_goal': B.goal_frame_probability(path_o, goal_xy, GAMMA, RADIUS)}
    if keep_path:
      rec['obs'] = path_o; rec['act'] = path_a
    out.append(rec)
  return out


def run_parallel(jobs, workers, seed0, cont_ckpt):
  parts = DG.chunks(jobs, workers * 4)
  args = [(p, seed0 + i, cont_ckpt) for i, p in enumerate(parts)]
  if workers <= 1:
    res = [_run_jobs(a) for a in args]
  else:
    # the parent has initialised JAX (the candidate sampler); a forked child
    # inherits its thread pools and deadlocks on the first policy call, so
    # the workers are spawned fresh
    with get_context('spawn').Pool(workers) as pool:
      res = pool.map(_run_jobs, args)
  return [r for part in res for r in part]


# ------------------------------------------------------------------- build
def make_jobs(anchors, bundle, obs, act, kinds, draws, seed_base, cand_seed, keep_path):
  """anchors: list of (anchor_id, e, t, set)."""
  o0 = np.stack([obs[e, t] for (_, e, t, _) in anchors])
  loc, scale = bundle['params_fn'](o0)
  jobs = []
  for i, (aid, e, t, sset) in enumerate(anchors):
    for kind, a in candidates_for(loc[i], scale[i], act[e, t], aid, kinds, cand_seed):
      for r in range(draws):
        jobs.append((aid, e, t, sset, kind, a, r, seed_base + 1000 * aid + r, keep_path))
  return jobs, {'scale_mean': float(scale.mean()), 'scale_median': float(np.median(scale))}


def stats_table(res, key_sets):
  rows = []
  for sset in key_sets:
    for cand in sorted({r['cand'] for r in res if r['set'] == sset}, key=_cand_order):
      rs = [r for r in res if r['set'] == sset and r['cand'] == cand]
      if not rs:
        continue
      rows.append({'set': sset, 'cand': cand, 'n': len(rs), 'reach': float(np.mean([r['success'] for r in rs])),
                   'death': float(np.mean([r['failure'] for r in rs])),
                   'p_goal': float(np.mean([r['p_goal'] for r in rs])),
                   'around': float(np.mean([r['max_y'] >= B.DETOUR_Y for r in rs]))})
  return rows


def _cand_order(c):
  return {'recorded': 0, 'mode': 1}.get(c, 2 + int(c[6:]) if c.startswith('sample') else 9)


def pair_stats(res, sets, halves=((0, 1), (2, 3)), thr=0.3):
  """Within-anchor candidate pairs: mean |log ratio|, and (when there are >= 2
  draws per key) the split-half sign agreement of the log ratio -- how much of
  the between-candidate difference is reproducible across consequences."""
  out = {}
  for sset in sets:
    by = {}
    for r in res:
      if r['set'] != sset:
        continue
      by.setdefault(r['anchor_id'], {}).setdefault(r['cand'], {})[r['draw']] = r['p_goal']
    d_full, agree, n_pairs, n_valid = [], [], 0, 0
    for aid, cands in by.items():
      names = sorted(cands, key=_cand_order)
      for i in range(len(names)):
        for j in range(i + 1, len(names)):
          pi, pj = cands[names[i]], cands[names[j]]
          draws = sorted(set(pi) & set(pj))
          if not draws:
            continue
          mi, mj = np.mean([pi[d] for d in draws]), np.mean([pj[d] for d in draws])
          d_full.append(np.log((mi + EPS_P) / (mj + EPS_P)))
          n_pairs += 1
          h1 = [d for d in halves[0] if d in draws]; h2 = [d for d in halves[1] if d in draws]
          if h1 and h2:
            a = np.log((np.mean([pi[d] for d in h1]) + EPS_P) / (np.mean([pj[d] for d in h1]) + EPS_P))
            b = np.log((np.mean([pi[d] for d in h2]) + EPS_P) / (np.mean([pj[d] for d in h2]) + EPS_P))
            if abs(a) > thr:
              agree.append(float(np.sign(a) == np.sign(b)))
              n_valid += int(abs(b) > thr and np.sign(a) == np.sign(b))
    d_full = np.array(d_full)
    out[sset] = {'n_pairs': n_pairs, 'mean_abs_log_ratio': float(np.abs(d_full).mean()) if len(d_full) else None,
                 'frac_abs_gt_thr': float((np.abs(d_full) > thr).mean()) if len(d_full) else None,
                 'split_half_sign_agreement': float(np.mean(agree)) if agree else None,
                 'n_half_pairs': len(agree), 'n_validated_pairs': n_valid}
  return out


def build(args):
  rng = np.random.default_rng(args.seed)
  obs, act, lengths, meta, route_rec = load_dataset()
  n_eps, L = obs.shape[:2]
  held = split_episodes(route_rec, args.holdout_frac, rng)
  fit = ~held
  bundle = policy_bundle(args.cont_ckpt)
  # --- fit anchors
  masks = dense_masks(obs, lengths, route_rec, fit)
  anchors, aid = [], 0
  for sset in DENSE_SETS:
    n = int(args.n_dense) if args.n_dense else DENSE_N[sset]
    for (e, t) in pick_rows(masks[sset], n, rng):
      anchors.append((aid, e, t, sset)); aid += 1
  n_dense = aid
  for (e, t) in general_rows(lengths, fit, args.every, rng, args.n_general or None):
    anchors.append((aid, e, t, 'general')); aid += 1
  dense_kinds = ['recorded'] + [f'sample{k}' for k in range(args.n_samples)]
  gen_kinds = ['recorded', 'sample0']
  jobs_d, sc_d = make_jobs(anchors[:n_dense], bundle, obs, act, dense_kinds, args.draws, R1_SEED, R1_SEED + 11, True)
  jobs_g, _ = make_jobs(anchors[n_dense:], bundle, obs, act, gen_kinds, args.draws_general, R1_SEED, R1_SEED + 11, True)
  # --- held-out anchors
  masks_h = dense_masks(obs, lengths, route_rec, held)
  anchors_h, hid = [], HOLDOUT_ID0
  for sset in DENSE_SETS:
    for (e, t) in pick_rows(masks_h[sset], args.n_holdout, rng):
      anchors_h.append((hid, e, t, sset)); hid += 1
  for (e, t) in general_rows(lengths, held, 7, rng, args.n_holdout):
    anchors_h.append((hid, e, t, 'general')); hid += 1
  hold_kinds = ['recorded', 'mode'] + [f'sample{k}' for k in range(args.n_samples_holdout)]
  jobs_h, sc_h = make_jobs(anchors_h, bundle, obs, act, hold_kinds, args.draws_holdout, R1_SEED + 300_000, R1_SEED + 13, False)
  jobs = jobs_d + jobs_g + jobs_h
  if args.limit:
    jobs = jobs_d[:args.limit] + jobs_g[:args.limit] + jobs_h[:args.limit]
  print(f'build: {n_dense} dense anchors x {len(dense_kinds)} candidates x {args.draws} draws = {len(jobs_d)} paths; '
        f'{len(anchors) - n_dense} general anchors x 2 x {args.draws_general} = {len(jobs_g)}; '
        f'{len(anchors_h)} held-out anchors x {len(hold_kinds)} x {args.draws_holdout} = {len(jobs_h)}; '
        f'policy scale mean {sc_d["scale_mean"]:.3f}; {args.workers} workers', flush=True)
  t0 = time.time()
  res = run_parallel(jobs, args.workers, R1_SEED + 100, args.cont_ckpt)
  wall = time.time() - t0
  print(f'rollouts done in {wall:.0f}s ({len(res)} paths)', flush=True)
  B.OUT.mkdir(parents=True, exist_ok=True)
  tag = args.tag
  # --- the replay (fit paths)
  fit_res = [r for r in res if 'obs' in r]
  n = len(fit_res)
  obs_p = np.zeros((n, L, B.STATE_DIM + 2), np.float32)
  act_p = np.zeros((n, L, B.ACTION_DIM), np.float32)
  lengths_p = np.zeros(n, np.int64)
  for i, r in enumerate(fit_res):
    k = len(r['obs'])
    obs_p[i, :k] = r['obs']; act_p[i, :k] = r['act']; lengths_p[i] = k
    obs_p[i, k:] = r['obs'][-1]
  m = json.loads(meta)
  m.update({'arm': 'policy_continuation_replay_oracle_r1', 'source_dataset': str(B.DATASET), 'p_active': B.P_ACTIVE,
            'continuation_ckpt': str(args.cont_ckpt), 'continuation': 'frozen deployment policy mode, closed-loop from step 2',
            'candidates_dense': dense_kinds, 'candidates_general': gen_kinds, 'draws_dense': args.draws,
            'draws_general': args.draws_general, 'every_general': args.every, 'gen_seed': R1_SEED,
            'anchor_rule': 'row 0 of every path (set_anchor_strata in the driver)',
            'hazards': 'redrawn from the priors per draw, paired across the candidates of one anchor',
            'held_out_episodes': int(held.sum())})
  path = B.OUT / (f'replay_policy_{tag}.npz' if not args.limit else f'replay_policy_{tag}_smoke.npz')
  np.savez_compressed(
      path, obs=obs_p, act=act_p, lengths=lengths_p, eval_goals=obs_p[:, 0, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float32),
      meta=np.asarray(json.dumps(m, sort_keys=True)),
      audit_kind=np.array([r['set'] for r in fit_res]), audit_cand=np.array([r['cand'] for r in fit_res]),
      audit_episode=np.array([r['episode'] for r in fit_res], np.int32),
      audit_anchor_time=np.array([r['t'] for r in fit_res], np.int16),
      audit_anchor_id=np.array([r['anchor_id'] for r in fit_res], np.int32),
      audit_draw=np.array([r['draw'] for r in fit_res], np.int8),
      audit_success=np.array([r['success'] for r in fit_res]), audit_failure=np.array([r['failure'] for r in fit_res]),
      audit_u1=np.array([r['u1'] for r in fit_res]), audit_u2=np.array([r['u2'] for r in fit_res]),
      audit_p_goal=np.array([r['p_goal'] for r in fit_res], np.float32),
      audit_max_y=np.array([r['max_y'] for r in fit_res], np.float32))
  # --- the held-out table (no paths)
  hold_res = [r for r in res if 'obs' not in r]
  hpath = B.OUT / (f'holdout_policy_{tag}.npz' if not args.limit else f'holdout_policy_{tag}_smoke.npz')
  save_table(hpath, hold_res, {'continuation_ckpt': str(args.cont_ckpt), 'candidates': hold_kinds,
                               'draws': args.draws_holdout, 'held_out_episodes': int(held.sum()),
                               'held_out_episode_ids': np.flatnonzero(held).tolist()})
  # --- summary
  summary = {'paths': n, 'holdout_paths': len(hold_res), 'wall_seconds': wall, 'replay': str(path), 'holdout': str(hpath),
             'bytes': int(os.path.getsize(path)), 'policy_scale': sc_d, 'held_out_episodes': int(held.sum()),
             'fit_by_set_cand': stats_table(fit_res, list(DENSE_SETS) + ['general']),
             'holdout_by_set_cand': stats_table(hold_res, list(DENSE_SETS) + ['general']),
             'fit_pairs': pair_stats(fit_res, list(DENSE_SETS) + ['general'], halves=((0,), (1,))),
             'holdout_pairs': pair_stats(hold_res, list(DENSE_SETS) + ['general'])}
  (B.OUT / f'generation_policy_{tag}{"_smoke" if args.limit else ""}.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  print(build_report(summary, args), flush=True)
  (B.OUT / f'generation_policy_{tag}{"_smoke" if args.limit else ""}.md').write_text(build_report(summary, args) + '\n', encoding='utf-8')


def save_table(path, res, extra):
  np.savez_compressed(
      path, obs0=np.stack([r['obs0'] for r in res]).astype(np.float32), act0=np.stack([r['act0'] for r in res]).astype(np.float32),
      anchor_id=np.array([r['anchor_id'] for r in res], np.int32), episode=np.array([r['episode'] for r in res], np.int32),
      t=np.array([r['t'] for r in res], np.int16), set=np.array([r['set'] for r in res]), cand=np.array([r['cand'] for r in res]),
      draw=np.array([r['draw'] for r in res], np.int8), pair_seed=np.array([r['pair_seed'] for r in res], np.int64),
      success=np.array([r['success'] for r in res]), failure=np.array([r['failure'] for r in res]),
      steps=np.array([r['steps'] for r in res], np.int32),
      reach_step=np.array([-1 if r['reach_step'] is None else r['reach_step'] for r in res], np.int32),
      u1=np.array([r['u1'] for r in res]), u2=np.array([r['u2'] for r in res]),
      max_y=np.array([r['max_y'] for r in res], np.float32), final_xy=np.array([r['final_xy'] for r in res], np.float32),
      p_goal=np.array([r['p_goal'] for r in res], np.float32), meta=np.asarray(json.dumps(extra)))


def load_table(path):
  with np.load(path, allow_pickle=False) as d:
    keys = [k for k in d.files if k != 'meta']
    T = {k: d[k] for k in keys}
    T['meta'] = json.loads(str(d['meta']))
  return T


def build_report(s, args):
  L = [f'# Round-1 policy-continuation replay ({Path(s["replay"]).name})', '',
       f'Continuation: the frozen deployment policy mode ({Path(args.cont_ckpt).name}); one query '
       f'torque per path, then closed-loop to the horizon; hazards redrawn per draw, paired across candidates; '
       f'P_goal = the relabeling law (gamma {GAMMA}, radius {RADIUS}); policy tanh-normal scale {s["policy_scale"]["scale_mean"]:.3f} '
       f'(median {s["policy_scale"]["scale_median"]:.3f}).  {s["paths"]} fit paths, {s["holdout_paths"]} held-out paths '
       f'({s["held_out_episodes"]} held-out episodes), {s["wall_seconds"]:.0f} s.', '',
       '## Fit paths by state set and candidate', '',
       '| set | candidate | n | reach | death | P_goal | went around (max_y >= 6) |', '|---|---|---:|---:|---:|---:|---:|']
  for r in s['fit_by_set_cand']:
    L.append(f'| {r["set"]} | {r["cand"]} | {r["n"]} | {r["reach"]:.2f} | {r["death"]:.2f} | {r["p_goal"]:.3f} | {r["around"]:.2f} |')
  L += ['', '## Held-out paths by state set and candidate', '',
        '| set | candidate | n | reach | death | P_goal | went around |', '|---|---|---:|---:|---:|---:|---:|']
  for r in s['holdout_by_set_cand']:
    L.append(f'| {r["set"]} | {r["cand"]} | {r["n"]} | {r["reach"]:.2f} | {r["death"]:.2f} | {r["p_goal"]:.3f} | {r["around"]:.2f} |')
  L += ['', '## Between-candidate differences within an anchor (log P_goal ratio, floor 1e-3)', '',
        'split-half = sign agreement of the ratio between the first and the second half of the draws, over pairs whose '
        f'first-half |ratio| > 0.3: the ceiling any critic can reach on these keys (fit set: draw 0 vs draw 1).', '',
        '| split | set | pairs | mean abs log ratio | frac abs > 0.3 | split-half sign agreement | half pairs | validated pairs |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
  for split in ('fit', 'holdout'):
    for sset, r in s[f'{split}_pairs'].items():
      f_ = lambda v, fmt='.2f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
      L.append(f'| {split} | {sset} | {r["n_pairs"]} | {f_(r["mean_abs_log_ratio"])} | {f_(r["frac_abs_gt_thr"])} | '
               f'{f_(r["split_half_sign_agreement"])} | {r["n_half_pairs"]} | {r["n_validated_pairs"]} |')
  return '\n'.join(L)


# -------------------------------------------------------------------- gate
def critic_fn(nets, q_params):
  import jax
  import jax.numpy as jnp

  @jax.jit
  def f(qp, oo, aa):
    phi, psi = nets.representation_network.apply(qp, oo, aa)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)

  def score(o, a):
    out = []
    for k in range(0, len(o), 2048):
      out.append(np.asarray(f(q_params, jnp.asarray(o[k:k + 2048], jnp.float32), jnp.asarray(a[k:k + 2048], jnp.float32))))
    return np.concatenate(out)
  return score


def keyed(T):
  """Held-out table -> per (anchor, cand): obs0, act0, set, p by draw."""
  keys = {}
  for i in range(len(T['p_goal'])):
    k = (int(T['anchor_id'][i]), str(T['cand'][i]))
    d = keys.setdefault(k, {'obs0': T['obs0'][i], 'act0': T['act0'][i], 'set': str(T['set'][i]), 'p': {}})
    d['p'][int(T['draw'][i])] = float(T['p_goal'][i])
  return keys


def score_table(keys, scorers):
  """scorers: name -> fn(obs0[N,31], act0[N,8]) -> score[N]."""
  ks = list(keys)
  o = np.stack([keys[k]['obs0'] for k in ks]); a = np.stack([keys[k]['act0'] for k in ks])
  return ks, {name: fn(o, a) for name, fn in scorers.items()}


def gate_metrics(keys, ks, scores, sets, thr=0.3, halves=((0, 1), (2, 3))):
  """Per scorer and set: sign agreement with the validated between-candidate
  differences (pairs whose two split halves agree in sign, both beyond thr),
  |ratio|-weighted agreement over all pairs, and the pick gain of the argmax
  candidate: (P[argmax] - mean P) / (max P - mean P), 1 = oracle, 0 = random."""
  idx = {k: i for i, k in enumerate(ks)}
  by_anchor = {}
  for (aid, cand) in ks:
    by_anchor.setdefault(aid, []).append(cand)
  out = {}
  for name, sc in scores.items():
    out[name] = {}
    for sset in sets + ['dense_all']:
      agree_v, w_agree, w_all, gains, pick_p, best_p, mean_p, agree_samples = [], 0.0, 0.0, [], [], [], [], []
      for aid, cands in by_anchor.items():
        s0 = keys[(aid, cands[0])]['set']
        if not (s0 == sset or (sset == 'dense_all' and s0 in DENSE_SETS)):
          continue
        cands = sorted(cands, key=_cand_order)
        P = {c: keys[(aid, c)]['p'] for c in cands}
        Pm = {c: np.mean(list(P[c].values())) for c in cands}
        F = {c: sc[idx[(aid, c)]] for c in cands}
        for i in range(len(cands)):
          for j in range(i + 1, len(cands)):
            ci, cj = cands[i], cands[j]
            dr = sorted(set(P[ci]) & set(P[cj]))
            if not dr:
              continue
            d = np.log((Pm[ci] + EPS_P) / (Pm[cj] + EPS_P))
            fs = np.sign(F[ci] - F[cj])
            if d != 0:
              w_all += abs(d); w_agree += abs(d) * float(fs == np.sign(d))
            h1 = [x for x in halves[0] if x in dr]; h2 = [x for x in halves[1] if x in dr]
            if h1 and h2:
              a = np.log((np.mean([P[ci][x] for x in h1]) + EPS_P) / (np.mean([P[cj][x] for x in h1]) + EPS_P))
              b = np.log((np.mean([P[ci][x] for x in h2]) + EPS_P) / (np.mean([P[cj][x] for x in h2]) + EPS_P))
              if abs(a) > thr and abs(b) > thr and np.sign(a) == np.sign(b):
                agree_v.append(float(fs == np.sign(a)))
                if ci != 'recorded' and cj != 'recorded' and ci != 'mode' and cj != 'mode':
                  agree_samples.append(float(fs == np.sign(a)))
        # the pick
        pv = np.array([Pm[c] for c in cands]); fv = np.array([F[c] for c in cands])
        k_star = int(np.argmax(fv))
        pick_p.append(pv[k_star]); best_p.append(pv.max()); mean_p.append(pv.mean())
        if pv.max() - pv.mean() > 1e-6:
          gains.append((pv[k_star] - pv.mean()) / (pv.max() - pv.mean()))
      n_v = len(agree_v)
      m = {'n_validated_pairs': n_v, 'validated_agreement': float(np.mean(agree_v)) if n_v else None,
           'validated_se': float(np.sqrt(np.mean(agree_v) * (1 - np.mean(agree_v)) / n_v)) if n_v else None,
           'validated_agreement_samples_only': float(np.mean(agree_samples)) if agree_samples else None,
           'n_validated_samples_only': len(agree_samples),
           'weighted_agreement': float(w_agree / w_all) if w_all > 0 else None,
           'pick_gain': float(np.mean(gains)) if gains else None, 'n_anchors_with_spread': len(gains),
           'pick_p_goal': float(np.mean(pick_p)) if pick_p else None, 'best_p_goal': float(np.mean(best_p)) if best_p else None,
           'mean_p_goal': float(np.mean(mean_p)) if mean_p else None}
      out[name][sset] = m
  return out


def gate(args):
  import jax.numpy as jnp
  from crl import checkpoint, networks
  T = load_table(B.OUT / args.holdout)
  keys = keyed(T)
  bundle = policy_bundle(args.cont_ckpt)
  nets = bundle['nets']
  scorers = {}
  rng = np.random.default_rng(0)
  scorers['random'] = lambda o, a: rng.standard_normal(len(o))
  # the frozen policy's own likelihood of the candidate: "stay near the mode"
  def bc_logprob(o, a):
    l_, s_ = bundle['params_fn'](o)
    return np.asarray(networks.tanh_normal_log_prob(networks.TanhNormalParams(jnp.asarray(l_), jnp.asarray(s_)), jnp.asarray(a)))
  scorers['policy_logprob'] = bc_logprob
  labels = {}
  for ck in args.critics:
    p = Path(ck)
    label = f'{p.parent.parent.name}/{p.parent.name}/{p.stem}'
    _, st = checkpoint.load_checkpoint(p)
    scorers[label] = critic_fn(nets, st.q_params)
    labels[label] = str(p)
  ks, scores = score_table(keys, scorers)
  sets = list(DENSE_SETS) + ['general']
  metrics = gate_metrics(keys, ks, scores, sets, thr=args.thr)
  ceiling = pair_stats([{'set': keys[k]['set'], 'anchor_id': k[0], 'cand': k[1], 'draw': d, 'p_goal': p}
                        for k in keys for d, p in keys[k]['p'].items()], sets, thr=args.thr)
  # pass rule: pooled dense sets, validated agreement above chance by > 2 s.e. and >= the line
  res = {'holdout': str(B.OUT / args.holdout), 'thr': args.thr, 'ceiling': ceiling, 'metrics': metrics, 'critics': labels, 'pass': {}}
  for name in scorers:
    m = metrics[name]['dense_all']
    ok = (m['validated_agreement'] is not None and m['validated_agreement'] >= args.gate
          and m['validated_agreement'] - 2 * (m['validated_se'] or 1) > 0.5)
    res['pass'][name] = bool(ok)
  L = [f'# Gate (round 1): does f(s, a, g) order the candidate torques the way their validated single-step outcomes do?', '',
       f'Held-out anchors ({T["meta"]["held_out_episodes"]} held-out episodes), candidates {T["meta"]["candidates"]}, '
       f'{T["meta"]["draws"]} paired draws per key; the continuation is the frozen policy mode.  A pair (a_i, a_j) at one state '
       f'is VALIDATED when the log P_goal ratio has the same sign on draws {{0,1}} and {{2,3}} and |ratio| > {args.thr} on both.  '
       f'Agreement = share of validated pairs whose critic ordering matches (chance 0.50); samples-only = pairs of two policy '
       f'samples (excludes recorded and mode); weighted = |ratio|-weighted sign agreement over all pairs; pick gain = '
       f'(P[argmax f] - mean P) / (max P - mean P) over anchors with spread (1 = oracle, 0 = random).  '
       f'PASS = pooled dense-set validated agreement >= {args.gate:.2f} and > 0.5 by 2 s.e.', '',
       '## Ceiling: split-half sign agreement of the differences themselves', '',
       '| set | pairs | mean abs log ratio | frac > thr | split-half agreement | validated pairs |', '|---|---:|---:|---:|---:|---:|']
  f_ = lambda v, fmt='.2f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
  for sset, c in ceiling.items():
    L.append(f'| {sset} | {c["n_pairs"]} | {f_(c["mean_abs_log_ratio"])} | {f_(c["frac_abs_gt_thr"])} | '
             f'{f_(c["split_half_sign_agreement"])} | {c["n_validated_pairs"]} |')
  L += ['', '## Critics', '',
        '| scorer | set | validated pairs | agreement | s.e. | samples-only agreement (n) | weighted agreement | pick gain (anchors) | P pick / mean / best |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
  for name in scorers:
    for sset in sets + ['dense_all']:
      m = metrics[name][sset]
      L.append(f'| {name} | {sset} | {m["n_validated_pairs"]} | **{f_(m["validated_agreement"])}** | {f_(m["validated_se"], ".3f")} | '
               f'{f_(m["validated_agreement_samples_only"])} ({m["n_validated_samples_only"]}) | {f_(m["weighted_agreement"])} | '
               f'{f_(m["pick_gain"])} ({m["n_anchors_with_spread"]}) | {f_(m["pick_p_goal"], ".3f")} / {f_(m["mean_p_goal"], ".3f")} / {f_(m["best_p_goal"], ".3f")} |')
  L += ['', '## Pass', ''] + [f'- {name}: {"PASS" if ok else "fail"}' for name, ok in res['pass'].items()]
  out_md = B.OUT / f'gate_policy_{args.tag}.md'
  out_md.write_text('\n'.join(L) + '\n', encoding='utf-8')
  (B.OUT / f'gate_policy_{args.tag}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
  print('\n'.join(L), flush=True)


# ---------------------------------------------------------------- validate
def validate(args):
  """Updated actors at the held-out anchors: their mode torque once, then the
  OLD frozen policy; paired fresh draws against the old mode and the recorded
  torque."""
  from crl import checkpoint
  T = load_table(B.OUT / args.holdout)
  obs, act, lengths, meta, route_rec = load_dataset()
  bundle = policy_bundle(args.cont_ckpt)
  # unique anchors of the held-out table
  seen, anchors = set(), []
  for i in range(len(T['anchor_id'])):
    aid = int(T['anchor_id'][i])
    if aid in seen:
      continue
    seen.add(aid); anchors.append((aid, int(T['episode'][i]), int(T['t'][i]), str(T['set'][i])))
  if args.n_anchors:
    anchors = anchors[:args.n_anchors]
  o0 = np.stack([obs[e, t] for (_, e, t, _) in anchors])
  loc_old, scale_old = bundle['params_fn'](o0)
  a_rec = np.stack([act[e, t] for (_, e, t, _) in anchors])
  a_old = np.tanh(loc_old)
  arms = {'recorded': a_rec, 'old_mode': a_old}
  manifold = {}
  for ck in args.actors:
    p = Path(ck)
    label = f'{p.parent.parent.name}/{p.parent.name}/{p.stem}'
    _, st = checkpoint.load_checkpoint(p)
    loc, scale = bundle['params_fn'](o0, pp=st.policy_params)
    a_new = np.tanh(loc)
    arms[label] = a_new
    f_own = critic_fn(bundle['nets'], st.q_params)
    manifold[label] = {'torque_dist_to_old_mode': float(np.linalg.norm(a_new - a_old, axis=1).mean()),
                       'torque_dist_to_recorded': float(np.linalg.norm(a_new - a_rec, axis=1).mean()),
                       'old_mode_dist_to_recorded': float(np.linalg.norm(a_old - a_rec, axis=1).mean()),
                       'frac_saturated': float(np.mean(np.abs(a_new) > 0.95)), 'scale_mean': float(scale.mean()),
                       'own_f_new_minus_old': float(np.mean(f_own(o0, a_new) - f_own(o0, a_old))),
                       'own_f_new_minus_recorded': float(np.mean(f_own(o0, a_new) - f_own(o0, a_rec)))}
  jobs = []
  for i, (aid, e, t, sset) in enumerate(anchors):
    for name, A in arms.items():
      for r in range(args.draws):
        jobs.append((aid, e, t, sset, name, np.asarray(A[i], np.float32), r, R1_SEED + 50_000_000 + 1000 * (aid - HOLDOUT_ID0) + r, False))
  print(f'validate: {len(anchors)} anchors x {len(arms)} arms x {args.draws} draws = {len(jobs)} paths, {args.workers} workers', flush=True)
  t0 = time.time()
  res = run_parallel(jobs, args.workers, R1_SEED + 200, args.cont_ckpt)
  wall = time.time() - t0
  save_table(B.OUT / f'validate_{args.tag}.npz', res, {'arms': list(arms), 'draws': args.draws, 'actors': list(args.actors)})
  sets = list(DENSE_SETS) + ['general']
  by = {}
  for r in res:
    by.setdefault((r['set'], r['anchor_id']), {}).setdefault(r['cand'], {})[r['draw']] = r
  out = {'wall_seconds': wall, 'manifold': manifold, 'arms': {}}
  L = [f'# Validation ({args.tag}): the updated actor\'s single torque under the old continuation', '',
       f'{len(anchors)} held-out anchors, {args.draws} fresh paired draws per arm; each arm = one torque at the state, then the OLD '
       f'frozen policy mode closed-loop.  Gain = mean over anchors of log((P_arm + 1e-3) / (P_old_mode + 1e-3)); paired win = share of '
       f'(anchor, draw) with P_arm > P_old_mode among those that differ.', '',
       '| arm | set | n anchors | reach | death | P_goal | went around | gain vs old mode | s.e. | paired win | frac differ |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for name in arms:
    out['arms'][name] = {}
    for sset in sets + ['dense_all']:
      gains, wins, diff, reach, death, pg, around = [], [], 0, [], [], [], []
      for (s_, aid), cands in by.items():
        if not (s_ == sset or (sset == 'dense_all' and s_ in DENSE_SETS)) or name not in cands or 'old_mode' not in cands:
          continue
        pa = np.mean([r['p_goal'] for r in cands[name].values()]); po = np.mean([r['p_goal'] for r in cands['old_mode'].values()])
        gains.append(np.log((pa + EPS_P) / (po + EPS_P)))
        for d, r in cands[name].items():
          if d in cands['old_mode']:
            q = cands['old_mode'][d]['p_goal']
            if abs(r['p_goal'] - q) > 1e-9:
              diff += 1; wins.append(float(r['p_goal'] > q))
        reach.append(np.mean([r['success'] for r in cands[name].values()])); death.append(np.mean([r['failure'] for r in cands[name].values()]))
        pg.append(pa); around.append(np.mean([r['max_y'] >= B.DETOUR_Y for r in cands[name].values()]))
      if not gains:
        continue
      g = np.array(gains)
      m = {'n_anchors': len(g), 'reach': float(np.mean(reach)), 'death': float(np.mean(death)), 'p_goal': float(np.mean(pg)),
           'around': float(np.mean(around)), 'gain': float(g.mean()), 'gain_se': float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None,
           'paired_win': float(np.mean(wins)) if wins else None, 'frac_differ': diff / max(1, len(g) * args.draws)}
      out['arms'][name][sset] = m
      f_ = lambda v, fmt='.2f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
      L.append(f'| {name} | {sset} | {m["n_anchors"]} | {m["reach"]:.2f} | {m["death"]:.2f} | {m["p_goal"]:.3f} | {m["around"]:.2f} | '
               f'**{m["gain"]:+.3f}** | {f_(m["gain_se"], ".3f")} | {f_(m["paired_win"])} | {m["frac_differ"]:.2f} |')
  L += ['', '## Torque manifold of the updated actors (at the held-out anchors)', '',
        '| actor | dist to old mode | dist to recorded | old mode to recorded | frac |a| > 0.95 | scale | own f(new) - f(old mode) | own f(new) - f(recorded) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
  for name, m in manifold.items():
    L.append(f'| {name} | {m["torque_dist_to_old_mode"]:.3f} | {m["torque_dist_to_recorded"]:.3f} | {m["old_mode_dist_to_recorded"]:.3f} | '
             f'{m["frac_saturated"]:.2f} | {m["scale_mean"]:.3f} | {m["own_f_new_minus_old"]:+.2f} | {m["own_f_new_minus_recorded"]:+.2f} |')
  (B.OUT / f'validate_{args.tag}.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (B.OUT / f'validate_{args.tag}.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('build', 'gate', 'validate'))
  ap.add_argument('--cont-ckpt', required=True, help='the frozen deployment policy (continuation + candidate sampler)')
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--tag', default='r1')
  ap.add_argument('--holdout-frac', type=float, default=0.10)
  ap.add_argument('--n-dense', type=int, default=0, help='build: anchors per dense set (0 = the per-set defaults)')
  ap.add_argument('--n-samples', type=int, default=2, help='build: policy samples per dense anchor (plus the recorded torque)')
  ap.add_argument('--draws', type=int, default=2, help='build: draws per dense key')
  ap.add_argument('--every', type=int, default=60, help='build: general anchors every k recorded rows')
  ap.add_argument('--n-general', type=int, default=0, help='build: cap on general anchors (0 = all)')
  ap.add_argument('--draws-general', type=int, default=1)
  ap.add_argument('--n-holdout', type=int, default=60, help='build: held-out anchors per set')
  ap.add_argument('--n-samples-holdout', type=int, default=3)
  ap.add_argument('--draws-holdout', type=int, default=4)
  ap.add_argument('--limit', type=int, default=0, help='build: smoke with the first N jobs of each block')
  ap.add_argument('--holdout', default='holdout_policy_r1.npz', help='gate/validate: the held-out table (under V6_BRANCH_OUT)')
  ap.add_argument('--critics', nargs='*', default=[], help='gate: critic checkpoints')
  ap.add_argument('--thr', type=float, default=0.3, help='gate: |log ratio| line for a validated pair')
  ap.add_argument('--gate', type=float, default=0.65, help='gate: pooled validated agreement pass line')
  ap.add_argument('--actors', nargs='*', default=[], help='validate: updated actor checkpoints')
  ap.add_argument('--n-anchors', type=int, default=0, help='validate: cap on held-out anchors (smoke)')
  args = ap.parse_args(argv)
  if args.mode == 'build':
    build(args)
  elif args.mode == 'gate':
    gate(args)
  else:
    validate(args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
