"""Who are the V3 detour-takers, and what decided their route?

Framing (the question this answers): under gamma = 0.99 the BR objective
prefers the shortcut (discounted refs 0.323 vs 0.100) and the latent is
invisible at t = 0, so the rational blind policy takes the shortcut EVERY
episode. Yet 10-15 % of the trained learner's episodes are labelled 'detour'
(and 6-16 % commit to nothing). This probe characterises those episodes
instead of asking whether the learner could be made to value the detour.

Every rollout reuses the authoritative eval's env seed, so the episodes are
the SAME ones the reported success numbers came from (receipt: route labels
must match the eval json episode-for-episode).

Measurements, per checkpoint:
  R  rollouts with full xy traces: route, outcome, first corridor entered
     (east / north), heading at 2u / 3u of displacement, jam runs, the leg
     reached on the detour, final xy.
  P  route predictability from the t = 0 observation (state dims 0-28 and
     goal dims 29-30 separately): per-dim d' against a permutation null and a
     leave-one-out ridge decoder against a label-permutation null. Ported
     from scripts/probe_tworoute_route_trigger.py.
  D  determinism: same reset draw -> same route (two env instances on one
     seed); forced latent u = 1 vs u = 0 on identical resets -> same route
     (the route is decided before the band, so any disagreement is a leak).
  X  cause attribution by replay (full mujoco reset so the replay is a pure
     function of (qpos, qvel, goal)):
       - control: replay each episode's own (qpos, qvel, goal) -> same route?
       - perturbation: +-eps on the ant qpos (eps = 0.01 and 0.03, i.e.
         1/10 and 3/10 of the reset jitter) -> route flip fraction, for
         detour-takers vs shortcut-takers;
       - swap: (state_i, goal_j) and (state_j, goal_i) across detour /
         shortcut pairs -> does the route follow the state or the goal?
  C  what the objective says: at each rollout's s0, the twin-min critic's
     mean Q over the dataset's t = 0 shortcut-mode actions minus its mean Q
     over the detour-mode actions, and Q of the actor's own action; grouped
     by realised route, and the finite-difference slope of Q along the
     shortcut-minus-detour mode axis AT the actor's own action (the push
     the actor loss exerts). Plus the actor's t = 0 action on the mode axis
     (lambda = 0 detour mean, 1 shortcut mean) and the dataset-state mode
     averaging table (P1 of probe_tworoute_route_choice.py, ported).

Usage:
  python scripts/probe_tworoute_v3_detour.py --variant br \
      --ckpt v3br_crl_s0_100k/final.pkl --label v3br_s0_100k [--n 300]
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT_ROOT = 'artifacts/tworoute_rockfall_v3'
DATASET = ('artifacts/tworoute_rockfall_v3/{v}/dataset/'
           'antmaze_tworoute_rockfall_v3{v}.npz')
HORIZON = 400
NQ_ANT = 15
STATE_DIMS = slice(0, 29)
GOAL_DIMS = slice(29, 31)
#: corridor half-width is 2 (SCALING 4); the central block is x, y in [2, 6].
CORR = 2.0
JAM_EPS, JAM_RUN = 0.004, 50
MODE_STEPS = 12
#: finite-difference step (action units) for the critic slope along the
#: shortcut-minus-detour mode axis.
AXIS_DELTA = 0.2


def env_id(variant):
  return f'offline_ant_umaze_tworoute_rockfall_v3{variant}'


def make_env(variant, seed, full_reset=False):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  env = envs_mod.make_env(env_id(variant), cfg, seed=seed)
  env._env.full_reset = bool(full_reset)
  return cfg, env


def build_policy(ckpt_path, variant):
  cfg, _ = make_env(variant, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def q_diag(o, a):
    """Q(o_i, a_i) with o_i's own goal; twin-min exactly as the actor loss."""
    q = nets.q_network.apply(qp, o, a)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  return act, q_diag, int(step)


# ---- R: rollouts ------------------------------------------------------------
def _leg(x, y):
  """Coarse maze position: which corridor the torso centre is in."""
  if abs(y) < CORR:
    return 'start' if x < CORR else 'bottom_row' if x < 6.0 else 'goal_cell'
  if x < CORR:
    return 'west_column' if y < 6.0 else 'top_left'
  if y >= 6.0:
    return 'top_row' if x < 6.0 else 'top_right'
  if x >= 6.0:
    return 'east_column'
  return 'block'


def rollout(env, act, o, keep_steps=MODE_STEPS):
  """Run one episode from an already-reset env; return the trace record."""
  xs, ys, acts = [float(o[0])], [float(o[1])], []
  jam_run, jam_max, info = 0, 0, {}
  for t in range(HORIZON):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    if t < keep_steps:
      acts.append(a.copy())
    o, r, done, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    if abs(x - xs[-1]) < JAM_EPS and abs(y - ys[-1]) < JAM_EPS:
      jam_run += 1
      jam_max = max(jam_max, jam_run)
    else:
      jam_run = 0
    xs.append(x)
    ys.append(y)
    if done or r > 0:
      break
  xs, ys = np.array(xs), np.array(ys)
  legs = [_leg(x, y) for x, y in zip(xs, ys)]
  first = None
  for i, lg in enumerate(legs):
    if lg == 'bottom_row':
      first = ('east', i)
      break
    if lg == 'west_column':
      first = ('north', i)
      break
  d = np.hypot(xs - xs[0], ys - ys[0])

  def heading_at(r):
    i = int(np.argmax(d >= r)) if (d >= r).any() else -1
    return (None if i < 0 else
            round(float(np.degrees(np.arctan2(ys[i] - ys[0],
                                              xs[i] - xs[0]))), 1))

  return {'route': info.get('route'), 'success': bool(info.get('success')),
          'failure': bool(info.get('failure')),
          'entered_hazard': bool(info.get('entered_hazard')),
          'steps': int(len(xs) - 1),
          'first_corridor': None if first is None else first[0],
          'first_corridor_step': None if first is None else int(first[1]),
          'heading_2u_deg': heading_at(2.0), 'heading_3u_deg': heading_at(3.0),
          'max_disp': round(float(d.max()), 3),
          'max_jam_run': int(jam_max), 'jammed': bool(jam_max >= JAM_RUN),
          'legs_visited': sorted(set(legs)),
          'final_leg': legs[-1],
          'final_xy': [round(float(xs[-1]), 3), round(float(ys[-1]), 3)],
          'xy': np.stack([xs, ys], 1).astype(np.float32),
          'acts_early': np.asarray(acts, np.float32)}


def collect(act, variant, n, seed):
  """Same env seed as the authoritative eval -> the same episodes."""
  _, env = make_env(variant, seed)
  eps = []
  for k in range(n):
    o = env.reset()
    sim = env._env
    rec = {'k': k, 'u': bool(env.privileged_rockfall_active),
           'o0': o.copy().astype(np.float32),
           'qpos0': np.asarray(sim.data.qpos).copy(),
           'qvel0': np.asarray(sim.data.qvel).copy(),
           'goal_xy': np.asarray(sim.goal, float).copy()}
    rec.update(rollout(env, act, o))
    eps.append(rec)
    if (k + 1) % 50 == 0:
      print(f'  R {k + 1}/{n}', flush=True)
  return eps


# ---- P: predictability from obs[0] -------------------------------------------
def _dp(g0, g1):
  pooled = np.sqrt((g0.var(0, ddof=1) + g1.var(0, ddof=1)) / 2.0) + 1e-9
  return np.abs(g0.mean(0) - g1.mean(0)) / pooled


def dprime_test(X, m, B=2000, seed=0):
  """max-over-dims d' of group m vs ~m against a permutation null."""
  d = _dp(X[m], X[~m])
  rng = np.random.default_rng(seed)
  k = int(m.sum())
  null = np.array([_dp(X[p][:k], X[p][k:]).max()
                   for p in (rng.permutation(len(X)) for _ in range(B))])
  p95 = float(np.percentile(null, 95))
  return {'n_pos': int(m.sum()), 'n_neg': int((~m).sum()),
          'max_dprime': round(float(d.max()), 4),
          'argmax_dim': int(d.argmax()),
          'top5_dims': [[int(i), round(float(d[i]), 3)]
                        for i in np.argsort(-d)[:5]],
          'null_p95': round(p95, 4),
          'n_dims_above_null_p95': int((d > p95).sum()),
          'p_value': round(float((null >= d.max()).mean()), 4),
          'predictable': bool(d.max() > p95)}


def loo_decoder(X, t, B=200, seed=0, lam=10.0):
  """LOO balanced accuracy of a ridge decoder vs a label-permutation null.
  Plain numpy -- the repo has no sklearn dependency."""
  sd = X.std(0)
  X = X[:, sd > 1e-8]
  X = (X - X.mean(0)) / (X.std(0) + 1e-9)
  X = np.c_[X, np.ones(len(X))]
  t = t.astype(float)

  def bal_acc(target):
    pred = np.empty(len(X))
    for i in range(len(X)):
      k = np.ones(len(X), bool)
      k[i] = False
      A = X[k].T @ X[k] + lam * np.eye(X.shape[1])
      w = np.linalg.solve(A, X[k].T @ (target[k] - target[k].mean()))
      pred[i] = X[i] @ w + target[k].mean()
    yh = (pred > 0.5).astype(float)
    p = target == 1
    if p.sum() == 0 or (~p).sum() == 0:
      return float('nan')
    return float(0.5 * ((yh[p] == 1).mean() + (yh[~p] == 0).mean()))

  obs_acc = bal_acc(t)
  rng = np.random.default_rng(seed)
  null = np.array([bal_acc(rng.permutation(t)) for _ in range(B)])
  return {'n': int(len(X)), 'n_features': int(X.shape[1] - 1),
          'balanced_accuracy': round(obs_acc, 4),
          'null_mean': round(float(np.nanmean(null)), 4),
          'null_p95': round(float(np.nanpercentile(null, 95)), 4),
          'p_value': round(float((null >= obs_acc).mean()), 4),
          'beats_chance': bool(obs_acc > np.nanpercentile(null, 95))}


def predictability(eps):
  o0 = np.stack([e['o0'] for e in eps])
  y = np.array([e['route'] or 'none' for e in eps])
  out = {}
  #: the goal is (8,0) + one-sided d4rl noise (y in [0, 1.5]); the route
  #: rate by goal-y tercile is the plain-language form of the goal-dim d'.
  gy = np.array([float(e['goal_xy'][1]) for e in eps])
  qs = np.quantile(gy, [0, 1 / 3, 2 / 3, 1])
  out['route_by_goal_y_tercile'] = [
      {'goal_y_range': [round(float(qs[i]), 3), round(float(qs[i + 1]), 3)],
       'n': int(m.sum()),
       **{g: round(float(np.mean(y[m] == g)), 4)
          for g in ('shortcut', 'detour', 'none')}}
      for i, m in ((i, (gy >= qs[i]) & (gy <= qs[i + 1])) for i in range(3))]
  out['goal_y_mean_by_route'] = {
      g: round(float(gy[y == g].mean()), 4) for g in ('shortcut', 'detour',
                                                      'none') if (y == g).any()}
  for name, pos, neg in (('detour_vs_shortcut', 'detour', 'shortcut'),
                         ('none_vs_shortcut', 'none', 'shortcut')):
    keep = np.isin(y, (pos, neg))
    X, m = o0[keep], y[keep] == pos
    if m.sum() < 5 or (~m).sum() < 5:
      out[name] = {'skipped': f'{int(m.sum())} vs {int((~m).sum())}'}
      continue
    out[name] = {
        'state_dims_0_28': dprime_test(X[:, STATE_DIMS], m),
        'goal_dims_29_30': dprime_test(X[:, GOAL_DIMS], m),
        'decoder_state': loo_decoder(X[:, STATE_DIMS], m),
        'decoder_goal': loo_decoder(X[:, GOAL_DIMS], m),
        'decoder_both': loo_decoder(X[:, :31], m)}
  return out


# ---- D: determinism ----------------------------------------------------------
def determinism(act, variant, seed, n_same=20, n_forced=40):
  eA, eB = make_env(variant, seed)[1], make_env(variant, seed)[1]
  same = [[rollout(eA, act, eA.reset())['route'],
           rollout(eB, act, eB.reset())['route']] for _ in range(n_same)]
  #: FULL mujoco reset for the forced pairs: with the legacy reset the two
  #: instances' histories differ (one always dies in the band), the solver
  #: warmstart bleeds across episodes, and on knife-edge episodes that alone
  #: flips the label (measured 36/40 legacy vs 40/40 full on br s0) -- which
  #: would read as a latent leak when it is chaos.
  eT, eF = (make_env(variant, seed + 5, full_reset=True)[1],
            make_env(variant, seed + 5, full_reset=True)[1])
  forced, o0_equal = [], 0
  for _ in range(n_forced):
    oT = eT.reset(rockfall_active=True)
    oF = eF.reset(rockfall_active=False)
    o0_equal += int(np.array_equal(oT, oF))
    rT, rF = rollout(eT, act, oT), rollout(eF, act, oF)
    forced.append([rT['route'], rF['route'], rT['failure']])
  return {'same_seed': {'n': n_same,
                        'agreement': round(float(np.mean(
                            [a == b for a, b in same])), 4)},
          'forced_latent': {'n': n_forced,
                            'o0_identical': int(o0_equal),
                            'route_agreement': round(float(np.mean(
                                [a == b for a, b, _ in forced])), 4),
                            'pairs': forced}}


# ---- X: replay attribution ---------------------------------------------------
def reset_to(env, qpos, qvel, goal_xy, u=False):
  """Full mujoco reset, then impose (qpos, qvel, goal): the replay is a pure
  function of those three. env.reset() first so the episode flags and the
  rng order are exactly as in a normal episode."""
  env.reset(rockfall_active=u)
  sim = env._env
  mujoco.mj_resetData(sim.model, sim.data)
  sim.data.qpos[:] = qpos
  sim.data.qvel[:] = qvel
  mujoco.mj_forward(sim.model, sim.data)
  env._goal_vec = np.zeros(29, np.float32)
  env._goal_vec[:2] = goal_xy
  env._goal_state_full = env._goal_vec.copy()
  sim.goal = np.asarray(goal_xy, float).copy()
  env._last_obs = sim._obs_dict()
  return env._flatten(env._last_obs)


def attribution(act, variant, eps, seed, n_per_group=12, n_pert=8,
                eps_list=(0.01, 0.03)):
  _, env = make_env(variant, seed + 9, full_reset=True)
  rng = np.random.default_rng(seed)
  groups = {g: [e for e in eps if (e['route'] or 'none') == g][:n_per_group]
            for g in ('detour', 'shortcut', 'none')}
  out = {}
  for g, sel in groups.items():
    if not sel:
      continue
    ctrl, pert = [], {str(x): [] for x in eps_list}
    for e in sel:
      o = reset_to(env, e['qpos0'], e['qvel0'], e['goal_xy'])
      r = rollout(env, act, o, keep_steps=0)['route']
      ctrl.append([e['route'], r])
      for x in eps_list:
        flips = 0
        for _ in range(n_pert):
          q = e['qpos0'].copy()
          q[:NQ_ANT] += rng.uniform(-x, x, NQ_ANT)
          o = reset_to(env, q, e['qvel0'], e['goal_xy'])
          rr = rollout(env, act, o, keep_steps=0)['route']
          flips += int(rr != r)
        pert[str(x)].append(flips / n_pert)
    out[g] = {'n': len(sel),
              'control_replay_agreement': round(float(np.mean(
                  [a == b for a, b in ctrl])), 4),
              'control_pairs': ctrl,
              'flip_fraction_vs_control': {
                  x: round(float(np.mean(v)), 4) for x, v in pert.items()},
              'flip_fraction_per_episode': pert}
    print(f'  X {g}: control {out[g]["control_replay_agreement"]} '
          f'flips {out[g]["flip_fraction_vs_control"]}', flush=True)
  #: swap test: does the route follow the STATE or the GOAL?
  dts, scs = groups['detour'], groups['shortcut']
  swaps = []
  for i in range(min(len(dts), len(scs))):
    a, b = dts[i], scs[i]
    r_sa_gb = rollout(env, act, reset_to(env, a['qpos0'], a['qvel0'],
                                         b['goal_xy']), 0)['route']
    r_sb_ga = rollout(env, act, reset_to(env, b['qpos0'], b['qvel0'],
                                         a['goal_xy']), 0)['route']
    swaps.append({'detour_state_shortcut_goal': r_sa_gb,
                  'shortcut_state_detour_goal': r_sb_ga})
  if swaps:
    out['swap'] = {
        'n_pairs': len(swaps),
        'route_follows_state': round(float(np.mean(
            [s['detour_state_shortcut_goal'] == 'detour' and
             s['shortcut_state_detour_goal'] == 'shortcut'
             for s in swaps])), 4),
        'route_follows_goal': round(float(np.mean(
            [s['detour_state_shortcut_goal'] == 'shortcut' and
             s['shortcut_state_detour_goal'] == 'detour'
             for s in swaps])), 4),
        'pairs': swaps}
  return out


# ---- C: what the objective says ----------------------------------------------
def objective_view(act, q_diag, eps, npz_path):
  a = np.load(npz_path, allow_pickle=True)
  s = np.load(npz_path.replace('.npz', '_sidecar.npz'), allow_pickle=True)
  obs, acts = a['obs'], a['act']
  intent = np.asarray(s['route_intent'])
  sc, dt = intent == 'shortcut', intent == 'detour'
  w = float(sc.mean())
  #: P1 port -- actor on the dataset's own early states vs the mode means.
  p1 = []
  for t in range(MODE_STEPS):
    m_sc, m_dt = acts[sc, t].mean(0), acts[dt, t].mean(0)
    mix, mid = w * m_sc + (1 - w) * m_dt, 0.5 * (m_sc + m_dt)
    pi = np.asarray(act(jnp.asarray(obs[:, t]))).mean(0)
    cand = {'shortcut': m_sc, 'detour': m_dt, 'mixture': mix, 'midpoint': mid}
    dist = {k: round(float(np.linalg.norm(pi - v)), 4) for k, v in cand.items()}
    p1.append({'t': t, 'mode_separation': round(float(
        np.linalg.norm(m_sc - m_dt)), 4), 'dist': dist,
        'argmin': min(dist, key=dist.get)})
  n_mix = sum(r['argmin'] == 'mixture' for r in p1)
  seps = np.array([r['mode_separation'] for r in p1])
  d_mix = np.array([r['dist']['mixture'] for r in p1])

  #: per-rollout critic preference at s0 over the dataset's t=0 actions.
  a_sc0, a_dt0 = acts[sc, 0], acts[dt, 0]
  m_sc0, m_dt0 = a_sc0.mean(0), a_dt0.mean(0)
  axis = m_sc0 - m_dt0
  rows = []
  for e in eps:
    o0 = e['o0']
    q_sc = np.asarray(q_diag(jnp.asarray(np.tile(o0, (len(a_sc0), 1))),
                             jnp.asarray(a_sc0)))
    q_dt = np.asarray(q_diag(jnp.asarray(np.tile(o0, (len(a_dt0), 1))),
                             jnp.asarray(a_dt0)))
    pi0 = e['acts_early'][0]
    q_pi = float(np.asarray(q_diag(jnp.asarray(o0[None]),
                                   jnp.asarray(pi0[None])))[0])
    lam = float((pi0 - m_dt0) @ axis / (axis @ axis))
    #: the push the actor loss exerts at s0: finite-difference slope of Q
    #: along the mode axis at the actor's OWN action (+ = toward shortcut).
    ahat = axis / (np.linalg.norm(axis) + 1e-9)
    a_pm = np.clip(np.stack([pi0 + AXIS_DELTA * ahat,
                             pi0 - AXIS_DELTA * ahat]), -1.0, 1.0)
    q_pm = np.asarray(q_diag(jnp.asarray(np.tile(o0, (2, 1))),
                             jnp.asarray(a_pm)))
    dq_axis = float((q_pm[0] - q_pm[1]) / (2 * AXIS_DELTA))
    #: lambda along the first MODE_STEPS steps, phase-matched mode means
    lam_t = []
    for t in range(min(MODE_STEPS, len(e['acts_early']))):
      ax_t = acts[sc, t].mean(0) - acts[dt, t].mean(0)
      lam_t.append(float((e['acts_early'][t] - acts[dt, t].mean(0)) @ ax_t
                         / (ax_t @ ax_t)))
    rows.append({'route': e['route'] or 'none',
                 'q_shortcut_mode': float(q_sc.mean()),
                 'q_detour_mode': float(q_dt.mean()),
                 'q_pref_shortcut': float(q_sc.mean() - q_dt.mean()),
                 'frac_sc_actions_beat_dt_mean': float(
                     (q_sc > q_dt.mean()).mean()),
                 'q_actor': q_pi, 'dq_along_axis_at_actor': dq_axis,
                 'lambda_t0': lam, 'lambda_t': lam_t,
                 'dist_to_sc_mode': float(np.linalg.norm(pi0 - m_sc0)),
                 'dist_to_dt_mode': float(np.linalg.norm(pi0 - m_dt0))})

  def grp(g, key):
    v = [r[key] for r in rows if r['route'] == g]
    return (None if not v else
            {'n': len(v), 'mean': round(float(np.mean(v)), 4),
             'sd': round(float(np.std(v)), 4)})

  by_route = {g: {k: grp(g, k) for k in
                  ('q_pref_shortcut', 'q_shortcut_mode', 'q_detour_mode',
                   'q_actor', 'dq_along_axis_at_actor', 'lambda_t0',
                   'dist_to_sc_mode',
                   'dist_to_dt_mode')}
              for g in ('shortcut', 'detour', 'none')}
  lam_traj = {g: [round(float(np.mean([r['lambda_t'][t] for r in rows
                                       if r['route'] == g
                                       and len(r['lambda_t']) > t])), 3)
                  for t in range(MODE_STEPS)]
              for g in ('shortcut', 'detour', 'none')
              if any(r['route'] == g and len(r['lambda_t']) == MODE_STEPS
                     for r in rows)}
  #: d' of the t=0 lambda between detour- and shortcut-takers
  l_sc = np.array([r['lambda_t0'] for r in rows if r['route'] == 'shortcut'])
  l_dt = np.array([r['lambda_t0'] for r in rows if r['route'] == 'detour'])
  d_lam = (float(_dp(l_sc[:, None], l_dt[:, None])[0])
           if len(l_sc) > 1 and len(l_dt) > 1 else None)
  return {'dataset_shortcut_mass_weight': round(w, 4),
          'mode_separation_t0': round(float(seps[0]), 4),
          'p1_mode_averaging': {
              'argmin_counts': {k: sum(r['argmin'] == k for r in p1)
                                for k in ('shortcut', 'detour', 'mixture',
                                          'midpoint')},
              'n_steps_argmin_is_mixture': n_mix,
              'mean_dist_to_mixture': round(float(d_mix.mean()), 4),
              'threshold_0p25_x_separation': round(float(
                  0.25 * seps.mean()), 4),
              'mode_averaging_confirmed': bool(
                  n_mix >= 8 and d_mix.mean() < 0.25 * seps.mean()),
              'per_step': p1},
          'critic_at_s0_by_route': by_route,
          'frac_episodes_critic_prefers_shortcut': {
              g: round(float(np.mean([r['q_pref_shortcut'] > 0 for r in rows
                                      if r['route'] == g])), 4)
              for g in ('shortcut', 'detour', 'none')
              if any(r['route'] == g for r in rows)},
          'frac_episodes_critic_pushes_actor_toward_shortcut': {
              g: round(float(np.mean([r['dq_along_axis_at_actor'] > 0
                                      for r in rows if r['route'] == g])), 4)
              for g in ('shortcut', 'detour', 'none')
              if any(r['route'] == g for r in rows)},
          'lambda_t0_dprime_detour_vs_shortcut': (
              None if d_lam is None else round(d_lam, 4)),
          'lambda_trajectory_by_route': lam_traj,
          'per_episode': [{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in r.items() if k != 'lambda_t'}
                          for r in rows]}


# ---- summary -----------------------------------------------------------------
def summarize_rollouts(eps):
  def blk(sel):
    if not sel:
      return None
    fc = [e['first_corridor'] for e in sel]
    h3 = [e['heading_3u_deg'] for e in sel if e['heading_3u_deg'] is not None]
    fcs = [e['first_corridor_step'] for e in sel
           if e['first_corridor_step'] is not None]
    return {'n': len(sel),
            'active': int(sum(e['u'] for e in sel)),
            'success': round(float(np.mean([e['success'] for e in sel])), 4),
            'failure': round(float(np.mean([e['failure'] for e in sel])), 4),
            'timeout': round(float(np.mean(
                [not e['success'] and not e['failure'] for e in sel])), 4),
            'first_corridor': {str(k): fc.count(k)
                               for k in ('east', 'north', None)},
            'first_corridor_step_median': (
                None if not fcs else float(np.median(fcs))),
            'heading_3u_deg': (None if not h3 else
                               {'mean': round(float(np.mean(h3)), 1),
                                'min': round(float(min(h3)), 1),
                                'max': round(float(max(h3)), 1)}),
            'jammed': round(float(np.mean([e['jammed'] for e in sel])), 4),
            'max_disp_median': round(float(np.median(
                [e['max_disp'] for e in sel])), 3),
            'final_leg': {k: sum(e['final_leg'] == k for e in sel)
                          for k in sorted(set(e['final_leg'] for e in sel))},
            'steps_success': sorted(e['steps'] for e in sel if e['success'])}
  by = {g: blk([e for e in eps if (e['route'] or 'none') == g])
        for g in ('shortcut', 'detour', 'none')}
  by['detour_timeouts'] = blk([e for e in eps
                               if e['route'] == 'detour' and not e['success']])
  return by


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=['tr', 'br'], required=True)
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=909,
                  help='env seed of the authoritative eval (same episodes)')
  ap.add_argument('--label', default=None)
  ap.add_argument('--eval-json', default=None,
                  help='eval_tworoute_v3.json to match routes against '
                       '(default: <ckpt dir>/eval_tworoute_v3.json)')
  args = ap.parse_args()
  act, q_diag, step = build_policy(args.ckpt, args.variant)
  label = args.label or os.path.basename(os.path.dirname(args.ckpt))
  print(f'{label}: ckpt {args.ckpt} @ step {step}', flush=True)

  eps = collect(act, args.variant, args.n, args.seed)
  counts = {k: sum((e['route'] or 'none') == k for e in eps)
            for k in ('shortcut', 'detour', 'none')}
  print('routes:', counts, flush=True)

  #: receipt -- these must be the eval's episodes
  ev_path = args.eval_json or os.path.join(
      os.path.dirname(args.ckpt) or '.', 'eval_tworoute_v3.json')
  receipt = None
  if os.path.exists(ev_path):
    ev = json.load(open(ev_path))['episodes']
    m = min(len(ev), len(eps))
    #: the eval ran on the server GPU; this probe runs on CPU. Same seeds,
    #: same latents, but float32 matmul differences (TF32 vs exact) are
    #: enough to move knife-edge episodes across the label boundary, so the
    #: mismatch table is itself a sensitivity readout, not a bug.
    mism = [(ev[i]['route'] or 'none', eps[i]['route'] or 'none')
            for i in range(m) if ev[i]['route'] != eps[i]['route']]
    receipt = {'eval_json': ev_path, 'n_compared': m,
               'route_match': m - len(mism),
               'u_match': int(sum(ev[i]['u'] == eps[i]['u']
                                  for i in range(m))),
               'route_mismatch_eval_to_probe': {
                   f'{a}->{b}': sum(1 for x in mism if x == (a, b))
                   for a, b in sorted(set(mism))}}
    print('receipt vs eval json:', receipt, flush=True)

  roll = summarize_rollouts(eps)
  print('R rollouts:', json.dumps(roll, indent=1), flush=True)
  pred = predictability(eps)
  print('P predictability:', json.dumps(pred, indent=1), flush=True)
  det = determinism(act, args.variant, args.seed + 77)
  print('D determinism:', json.dumps(
      {k: {kk: vv for kk, vv in v.items() if kk != 'pairs'}
       for k, v in det.items()}, indent=1), flush=True)
  attr = attribution(act, args.variant, eps, args.seed)
  print('X attribution:', json.dumps(
      {g: {k: v for k, v in d.items()
           if k not in ('control_pairs', 'flip_fraction_per_episode', 'pairs')}
       for g, d in attr.items()}, indent=1), flush=True)
  obj = objective_view(act, q_diag, eps, DATASET.format(v=args.variant))
  print('C objective view:', json.dumps(
      {k: v for k, v in obj.items() if k not in ('per_episode',)},
      indent=1), flush=True)

  out_dir = os.path.join(OUT_ROOT, args.variant)
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f'detour_probe_{label}.json')
  slim = []
  for e in eps:
    row = {k: v for k, v in e.items()
           if k not in ('o0', 'qpos0', 'qvel0', 'xy', 'acts_early')}
    row['goal_xy'] = [round(float(x), 4) for x in e['goal_xy']]
    row['o0_x'], row['o0_y'] = (round(float(e['o0'][0]), 4),
                                round(float(e['o0'][1]), 4))
    slim.append(row)
  with open(path, 'w') as f:
    json.dump({'label': label, 'variant': args.variant, 'ckpt': args.ckpt,
               'ckpt_step': step, 'n': args.n, 'seed': args.seed,
               'receipt': receipt, 'route_counts': counts,
               'rollouts': roll, 'predictability': pred,
               'determinism': det, 'attribution': attr,
               'objective_view': obj, 'episodes': slim}, f, indent=1)
  np.savez_compressed(
      path.replace('.json', '.npz'),
      o0=np.stack([e['o0'] for e in eps]),
      qpos0=np.stack([e['qpos0'] for e in eps]),
      qvel0=np.stack([e['qvel0'] for e in eps]),
      route=np.array([e['route'] or 'none' for e in eps]),
      u=np.array([e['u'] for e in eps]),
      xy=np.array([e['xy'] for e in eps], dtype=object),
      acts_early=np.array([e['acts_early'] for e in eps], dtype=object))
  print('->', path, flush=True)


if __name__ == '__main__':
  main()
