"""Ant immediate local controllability probe (no training).

Follows LOCAL_RANKING_WEAK: is the weak local critic ranking caused by
(a) actor-neighborhood actions producing near-identical immediate physical
effects, or (b) physically distinct effects the critic fails to rank?

Protocol: every candidate action is executed for EXACTLY ONE env step from a
bit-exact restored MuJoCo state (the actor never resumes). A separate
persistence measurement continues with 2 ZERO-action steps (not the actor, not
the repeated candidate). Candidate sets per state: local Gaussian around the
deterministic actor action (sigma 0.01..0.20), replay-neighbor actions, and
broad uniform as an OOD control. Local and uniform sets are NEVER pooled for
the verdict. Critic-score-scale artifacts are checked by decomposing the score
into phi(s,a), psi(g) norms + cosine (repr_norm=False so score = raw dot).
"""
import argparse
import json
import os
import sys

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # repo root, for `crl`
from ant_action_validity import load_actor_critic, restore, saturation, _spearman, _boot_ci
from ant_critic_local_ranking import build_behavior_buffer

from crl.config import Config
from crl import envs as envs_mod
from crl import checkpoint as ckpt_mod

CKPT = 'D:/Users/trhua/Research/contrastive_rl/antmaze_umaze_s0/latest.pkl'
NPZ = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_action_validity/ant_action_validity_samples.npz'
OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_immediate_controllability'
SIGMAS = [0.01, 0.03, 0.05, 0.10, 0.20]
SMALL = [0.01, 0.03, 0.05]
N_LOCAL = 128
N_UNIFORM = 128
K_NBR = 10
FALL_Z = 0.3
JAC_DELTA = 0.05

# Physical-distinguishability thresholds (within-state, across candidates).
# Scale anchor: one env step = 0.05 s; a healthy Ant covers ~3-5 cm/step, and
# the success threshold is 0.5 m. Spread below 2 mm across 128 candidates is
# behaviorally negligible; >=5 mm std (or >=2 cm max-min goal progress) is a
# clearly usable signal.
NEGL_STD_PROJ = 0.002
NEGL_RNG_PROG = 0.010
MEAN_STD_PROJ = 0.005
MEAN_RNG_PROG = 0.020


# --------------------------------------------------------------------------- #
def make_repr_fn(cfg, ckpt_path):
  """Rebuilds the critic's phi(s,a)/psi(g) encoders with the checkpoint params.

  Valid because repr_norm=False and twin_q=False for this run: the critic score
  is exactly dot(phi, psi). Module names match networks._repr_fn so q_params
  apply directly; a consistency gate asserts dot == critic score.
  """
  _, state = ckpt_mod.load_checkpoint(ckpt_path)
  sizes = list(cfg.hidden_layer_sizes) + [int(cfg.repr_dim)]
  w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')

  def _fn(obs, action):
    s = obs[:, :cfg.obs_dim]
    g = obs[:, cfg.obs_dim:]
    sa = hk.nets.MLP(sizes, w_init=w_init, activation=jax.nn.relu,
                     name='sa_encoder')(jnp.concatenate([s, action], axis=-1))
    gr = hk.nets.MLP(sizes, w_init=w_init, activation=jax.nn.relu,
                     name='g_encoder')(g)
    return sa, gr

  t = hk.without_apply_rng(hk.transform(_fn))
  apply = jax.jit(t.apply)

  def reprs(obs, acts):
    obs_k = jnp.asarray(np.tile(obs, (len(acts), 1)))
    phi, psi = apply(state.q_params, obs_k, jnp.asarray(acts))
    return np.asarray(phi), np.asarray(psi)
  return reprs


def torso_body_id(u):
  for i in range(u.model.nbody):
    name = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and 'torso' in name:
      return i
  return 1


def quat_heading(quat):
  w, x, y, z = quat
  return float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def xmat_heading(u, bid):
  m = np.asarray(u.data.xmat[bid]).reshape(3, 3)
  return float(np.arctan2(m[1, 0], m[0, 0]))


def contact_summary(u):
  try:
    n = int(u.data.ncon)
    f = np.zeros(6)
    tot = 0.0
    for i in range(n):
      mujoco.mj_contactForce(u.model, u.data, i, f)
      tot += abs(float(f[0]))
    return n, tot
  except Exception:
    return -1, np.nan


# --------------------------------------------------------------------------- #
def probe_candidate(u, ref, action, n_settle=2):
  """Restore exact state, execute candidate ONE step, then 2 zero-action steps.

  Records immediate (post-step-1) and persistent (post-step-3) physics. The
  actor never runs inside this function."""
  restore(u, ref['qpos'], ref['qvel'])
  u.goal = np.asarray(ref['goal'], float)
  mujoco.mj_forward(u.model, u.data)
  qpos0, qvel0 = ref['qpos'], ref['qvel']
  xy0, goal, d0 = qpos0[:2], ref['goal'], ref['d0']
  gdir = (goal - xy0) / (np.linalg.norm(goal - xy0) + 1e-9)

  u.step(np.asarray(action, np.float32))
  qpos1 = np.asarray(u.data.qpos).copy()
  qvel1 = np.asarray(u.data.qvel).copy()
  ctrl_err = float(np.abs(np.asarray(u.data.ctrl)
                          - np.clip(action, -1, 1)).max())
  ncon, cfrc = contact_summary(u)
  xy1 = qpos1[:2]
  m = dict(
      proj1=float(np.dot(xy1 - xy0, gdir)),
      disp1=float(np.linalg.norm(xy1 - xy0)),
      prog1=float(d0 - np.linalg.norm(xy1 - goal)),
      dvlin=float(np.linalg.norm(qvel1[:3] - qvel0[:3])),
      dvxy=float(np.linalg.norm(qvel1[:2] - qvel0[:2])),
      dvxy_proj=float(np.dot(qvel1[:2] - qvel0[:2], gdir)),
      dw=float(np.linalg.norm(qvel1[3:6] - qvel0[3:6])),
      djpos=float(np.linalg.norm(qpos1[7:] - qpos0[7:])),
      djvel=float(np.linalg.norm(qvel1[6:] - qvel0[6:])),
      dz=float(qpos1[2] - qpos0[2]), z1=float(qpos1[2]),
      fall1=bool(qpos1[2] < FALL_Z), ncon=ncon, cfrc=cfrc, ctrl_err=ctrl_err,
      qpos1=qpos1, qvel1=qvel1)
  minz = float(qpos1[2])
  zero = np.zeros(len(action), np.float32)
  for _ in range(n_settle):
    u.step(zero)
    minz = min(minz, float(u.data.qpos[2]))
  xy3 = np.asarray(u.data.qpos[:2]).copy()
  m.update(proj3=float(np.dot(xy3 - xy0, gdir)),
           disp3=float(np.linalg.norm(xy3 - xy0)),
           prog3=float(d0 - np.linalg.norm(xy3 - goal)),
           fall3=bool(minz < FALL_Z))
  return m


def probe_path(u, ref, action, n_settle=2):
  """XY path of the 1-candidate-step + zero-action rollout (for plots)."""
  restore(u, ref['qpos'], ref['qvel'])
  mujoco.mj_forward(u.model, u.data)
  path = [ref['qpos'][:2].copy()]
  u.step(np.asarray(action, np.float32))
  path.append(np.asarray(u.data.qpos[:2]).copy())
  zero = np.zeros(len(action), np.float32)
  for _ in range(n_settle):
    u.step(zero)
    path.append(np.asarray(u.data.qpos[:2]).copy())
  return np.array(path)


def jacobian_proxy(u, ref, api, delta=JAC_DELTA):
  """Central finite differences d(proj1)/d(a_i) around the actor action."""
  J = np.zeros(len(api))
  for i in range(len(api)):
    ap = api.copy(); ap[i] = np.clip(ap[i] + delta, -1, 1)
    am = api.copy(); am[i] = np.clip(am[i] - delta, -1, 1)
    pp = probe_candidate(u, ref, ap, n_settle=0)['proj1']
    pm = probe_candidate(u, ref, am, n_settle=0)['proj1']
    span = float(ap[i] - am[i])
    J[i] = (pp - pm) / span if span > 1e-9 else 0.0
  return J


def effective_rank(C):
  """Participation ratio of the candidate-action covariance eigenvalues."""
  if len(C) < 3:
    return float('nan')
  ev = np.linalg.eigvalsh(np.cov(C.T))
  ev = np.clip(ev, 0, None)
  s = ev.sum()
  return float(s * s / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0


# --------------------------------------------------------------------------- #
def gates(env, u, refs, actor, critic, reprs, rng):
  g = {}
  # Gate A: bit-exact restore determinism (restore -> 3-step rollout, twice).
  worst = 0.0
  for ref in refs[:3]:
    a = rng.uniform(-1, 1, 8).astype(np.float32)
    m1 = probe_candidate(u, ref, a)
    m2 = probe_candidate(u, ref, a)
    worst = max(worst, float(np.abs(m1['qpos1'] - m2['qpos1']).max()),
                abs(m1['prog3'] - m2['prog3']))
  g['gateA_restore'] = {'max_diff': worst, 'pass': bool(worst < 1e-9)}
  # Gate B: restored state reproduces the saved observation's xy/goal slice.
  errs = []
  for ref in refs[:20]:
    restore(u, ref['qpos'], ref['qvel'])
    mujoco.mj_forward(u.model, u.data)
    errs.append(float(np.abs(np.asarray(u.data.qpos[:2]) - ref['obs'][:2]).max()))
  g['gateB_obs_consistency'] = {'max_xy_err': float(max(errs)),
                                'pass': bool(max(errs) < 1e-6)}
  # Gate C: repr path consistency: dot(phi,psi) == critic score.
  ref = refs[0]
  C = rng.uniform(-1, 1, (32, 8)).astype(np.float32)
  sc = critic(ref['obs'], C)
  phi, psi = reprs(ref['obs'], C)
  dot = np.einsum('kd,kd->k', phi, psi)
  err = float(np.abs(dot - sc).max())
  g['gateC_repr_consistency'] = {'max_abs_err': err, 'pass': bool(err < 1e-3)}
  # Gate D: orientation computation validated against MuJoCo xmat.
  bid = torso_body_id(u)
  dmax = 0.0
  for ref in refs[:25]:
    restore(u, ref['qpos'], ref['qvel'])
    mujoco.mj_forward(u.model, u.data)
    hq = quat_heading(ref['qpos'][3:7])
    hx = xmat_heading(u, bid)
    d = abs((hq - hx + np.pi) % (2 * np.pi) - np.pi)
    dmax = max(dmax, d)
  g['gateD_orientation'] = {'torso_body_id': int(bid),
                            'max_quat_vs_xmat_rad': float(dmax),
                            'pass': bool(dmax < 1e-6)}
  return g


def state_features(ref, api, use_quat_heading):
  qpos, qvel = ref['qpos'], ref['qvel']
  xy, goal = qpos[:2], ref['goal']
  gdir = (goal - xy) / (np.linalg.norm(goal - xy) + 1e-9)
  speed = float(np.linalg.norm(qvel[:2]))
  vel_align = float(np.dot(qvel[:2] / (speed + 1e-9), gdir)) if speed > 0.05 else np.nan
  head = quat_heading(qpos[3:7]) if use_quat_heading else np.nan
  ga = float(np.arctan2(gdir[1], gdir[0]))
  head_err = abs((head - ga + np.pi) % (2 * np.pi) - np.pi) if np.isfinite(head) else np.nan
  return dict(z=float(qpos[2]), d0=float(ref['d0']), speed=speed,
              ang_speed=float(np.linalg.norm(qvel[3:6])), vz=float(qvel[2]),
              vel_align=vel_align, head_err=head_err,
              fallen=bool(qpos[2] < FALL_Z), sat=saturation(api))


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--out', default=OUT)
  ap.add_argument('--n_states', type=int, default=1000)
  ap.add_argument('--n_local', type=int, default=N_LOCAL)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  cfg = Config(env_name='antmaze_umaze')
  env = envs_mod.make_env('antmaze_umaze', cfg, seed=7)
  u = env._env.unwrapped
  actor, critic, step = load_actor_critic('antmaze_umaze', args.ckpt, cfg)
  reprs = make_repr_fn(cfg, args.ckpt)
  d = np.load(args.npz)
  refs = [dict(qpos=d['qpos'][k], qvel=d['qvel'][k], goal=d['goal'][k],
               obs=d['obs'][k], d0=float(d['d0'][k]))
          for k in range(min(args.n_states, len(d['qpos'])))]
  print(f'ckpt step {step} | {len(refs)} reference states')
  if len(refs) < min(args.n_states, 100):
    raise SystemExit('need >=100 reference states')

  G = gates(env, u, refs, actor, critic, reprs, rng)
  for k, v in G.items():
    print(f'  {k}: pass={v["pass"]} {v}')
  hard_fail = not (G['gateA_restore']['pass'] and G['gateB_obs_consistency']['pass']
                   and G['gateC_repr_consistency']['pass'])
  if hard_fail:
    json.dump({'verdict': 'INCONCLUSIVE', 'reason': 'gate failure', 'gates': G},
              open(os.path.join(args.out, 'ant_immediate_controllability.json'), 'w'),
              indent=2)
    print('VERDICT: INCONCLUSIVE (gate failure)')
    return
  use_quat = G['gateD_orientation']['pass']

  print('building behavior buffer for replay neighbors...')
  Sbuf, Abuf = build_behavior_buffer(env, actor)
  Smu, Ssd = Sbuf.mean(0), Sbuf.std(0) + 1e-6
  Sn = (Sbuf - Smu) / Ssd
  print(f'  {len(Sbuf)} transitions')

  set_keys = [f'local_s{s}' for s in SIGMAS] + ['replay_nbr', 'uniform']
  per_state = {k: [] for k in set_keys}     # list of per-state stat dicts
  feats = []
  actor_out = []
  jac_norms, jac_dims = [], []
  nbr_info = []
  pooled = {k: {'sc': [], 'proj1': [], 'prog1': []} for k in set_keys}
  ctrl_err_max = 0.0
  contact_ok = True
  # raw sample dump
  S_sid, S_set, S_act, S_sc, S_cos, S_phin = [], [], [], [], [], []
  S_m = {q: [] for q in ('proj1', 'prog1', 'proj3', 'prog3', 'dvxy', 'dvxy_proj',
                         'dw', 'djpos', 'djvel', 'dz', 'z1', 'disp1', 'disp3',
                         'fall1', 'fall3', 'ncon', 'cfrc')}
  S_qpos1, S_qvel1 = [], []
  psi_norms = []
  path_states, path_data = [], []

  for si, ref in enumerate(refs):
    api = actor(ref['obs'])
    ft = state_features(ref, api, use_quat)
    feats.append(ft)
    a_m = probe_candidate(u, ref, api)
    actor_out.append({q: a_m[q] for q in ('proj1', 'prog1', 'proj3', 'prog3',
                                          'disp1', 'fall1', 'fall3')})
    J = jacobian_proxy(u, ref, api)
    jac_norms.append(float(np.linalg.norm(J)))
    jac_dims.append(np.abs(J))

    # replay neighbors
    sn = (ref['obs'][:cfg.obs_dim] - Smu) / Ssd
    dists = np.linalg.norm(Sn - sn[None], axis=1)
    nn = np.argsort(dists)[:K_NBR]
    nbrA = Abuf[nn].astype(np.float32)
    nbr_info.append(dict(
        n=int(len(nn)), obs_dist=float(dists[nn].mean()),
        act_std=float(nbrA.std(0).mean()),
        act_dist_actor=float(np.linalg.norm(nbrA - api[None], axis=1).mean())))

    cand_sets = {}
    for s in SIGMAS:
      raw = api[None] + rng.normal(0, s, (args.n_local, 8))
      cand_sets[f'local_s{s}'] = np.clip(raw, -1, 1).astype(np.float32)
      cand_sets[f'local_s{s}_rawclipfrac'] = float(np.mean(np.abs(raw) > 1))
    cand_sets['replay_nbr'] = nbrA
    cand_sets['uniform'] = rng.uniform(-1, 1, (N_UNIFORM, 8)).astype(np.float32)

    for key in set_keys:
      C = cand_sets[key]
      sc = critic(ref['obs'], C)
      phi, psi = reprs(ref['obs'], C)
      dot = np.einsum('kd,kd->k', phi, psi)
      phin = np.linalg.norm(phi, axis=1)
      psin = float(np.linalg.norm(psi[0]))
      cos = dot / (phin * psin + 1e-12)
      psi_norms.append(psin)

      M = [probe_candidate(u, ref, C[k]) for k in range(len(C))]
      arr = {q: np.array([m[q] for m in M], float) for q in S_m}
      ctrl_err_max = max(ctrl_err_max, max(m['ctrl_err'] for m in M))
      if arr['ncon'].min() < 0:
        contact_ok = False
      dist_a = np.linalg.norm(C - api[None], axis=1)

      nd = max(1, len(C) // 10)
      o_sc = np.argsort(sc)
      o_tr = np.argsort(arr['prog1'])
      ib, iw = int(np.argmax(sc)), int(np.argmin(sc))
      ir = int(rng.integers(len(C)))
      st = dict(
          std_proj=float(arr['proj1'].std()),
          rng_proj=float(arr['proj1'].max() - arr['proj1'].min()),
          std_prog=float(arr['prog1'].std()),
          rng_prog=float(arr['prog1'].max() - arr['prog1'].min()),
          std_djvel=float(arr['djvel'].std()),
          std_dvxy=float(arr['dvxy'].std()),
          std_proj3=float(arr['proj3'].std()),
          rng_prog3=float(arr['prog3'].max() - arr['prog3'].min()),
          sp_proj=_spearman(sc, arr['proj1']),
          sp_prog=_spearman(sc, arr['prog1']),
          sp_dvxy=_spearman(sc, arr['dvxy_proj']),
          sp_proj3=_spearman(sc, arr['proj3']),
          sp_cos_prog=_spearman(cos, arr['prog1']),
          sp_dot_cos=_spearman(sc, cos),
          sp_dist_sc=_spearman(dist_a, sc),
          sp_dist_absproj=_spearman(dist_a, np.abs(arr['proj1'] - a_m['proj1'])),
          sp_p1_p3=_spearman(arr['proj1'], arr['proj3']),
          persist_sign=float(np.mean(np.sign(arr['proj3'][np.abs(arr['proj1']).argsort()[-nd:]])
                                     == np.sign(arr['proj1'][np.abs(arr['proj1']).argsort()[-nd:]]))),
          cb_prog=float(arr['prog1'][ib]), cw_prog=float(arr['prog1'][iw]),
          rand_prog=float(arr['prog1'][ir]), actor_prog=float(a_m['prog1']),
          cb_proj=float(arr['proj1'][ib]), cw_proj=float(arr['proj1'][iw]),
          cb_prog3=float(arr['prog3'][ib]), cw_prog3=float(arr['prog3'][iw]),
          top_dec=float(arr['prog1'][o_sc[-nd:]].mean()),
          bot_dec=float(arr['prog1'][o_sc[:nd]].mean()),
          true_top=float(arr['prog1'][o_tr[-nd:]].mean()),
          true_bot=float(arr['prog1'][o_tr[:nd]].mean()),
          top_dec3=float(arr['prog3'][o_sc[-nd:]].mean()),
          bot_dec3=float(arr['prog3'][o_sc[:nd]].mean()),
          fall1=float(arr['fall1'].mean()), fall3=float(arr['fall3'].mean()),
          score_std=float(sc.std()), score_rng=float(sc.max() - sc.min()),
          phin_std=float(phin.std()), phin_mean=float(phin.mean()),
          cos_std=float(cos.std()),
          mean_dist_actor=float(dist_a.mean()),
          dim_std=float(C.std(0).mean()),
          clip_frac=float(np.mean(np.abs(C) >= 1.0 - 1e-6)),
          raw_clip_frac=float(cand_sets.get(f'{key}_rawclipfrac', np.nan)),
          eff_rank=effective_rank(C),
          mean_score=float(sc.mean()),
      )
      per_state[key].append(st)
      pooled[key]['sc'].extend((sc - sc.mean()).tolist())
      pooled[key]['proj1'].extend(arr['proj1'].tolist())
      pooled[key]['prog1'].extend(arr['prog1'].tolist())
      # raw dump
      S_sid.extend([si] * len(C)); S_set.extend([set_keys.index(key)] * len(C))
      S_act.append(C); S_sc.extend(sc.tolist()); S_cos.extend(cos.tolist())
      S_phin.extend(phin.tolist())
      for q in S_m:
        S_m[q].extend(arr[q].tolist())
      S_qpos1.append(np.array([m['qpos1'] for m in M], np.float32))
      S_qvel1.append(np.array([m['qvel1'] for m in M], np.float32))
    if si < 10:
      path_states.append(si)
      C = cand_sets['local_s0.05'][:40]
      sc = critic(ref['obs'], C)
      paths = np.array([probe_path(u, ref, c) for c in C])
      apath = probe_path(u, ref, api)
      path_data.append(dict(paths=paths, scores=np.asarray(sc), actor=apath,
                            xy=ref['qpos'][:2].copy(), goal=ref['goal']))
    if si % 10 == 0:
      print(f'  state {si}/{len(refs)}', flush=True)

  # ------------------------------- aggregate ------------------------------- #
  def med(key, q):
    v = np.array([st[q] for st in per_state[key]], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None

  def frac_pos(key, q):
    v = np.array([st[q] for st in per_state[key]], float)
    v = v[np.isfinite(v)]
    return float(np.mean(v > 0)) if len(v) else None

  def mean_ci(key, q):
    v = [st[q] for st in per_state[key]]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)) if v else None), _boot_ci(v)

  actor_disp1_med = float(np.median([o['disp1'] for o in actor_out]))

  per_set = {}
  for key in set_keys:
    dec_gap = [st['top_dec'] - st['bot_dec'] for st in per_state[key]]
    true_gap = [st['true_top'] - st['true_bot'] for st in per_state[key]]
    dec_gap3 = [st['top_dec3'] - st['bot_dec3'] for st in per_state[key]]
    per_set[key] = {
        'std_proj1_median': med(key, 'std_proj'),
        'rng_proj1_median': med(key, 'rng_proj'),
        'std_prog1_median': med(key, 'std_prog'),
        'rng_prog1_median': med(key, 'rng_prog'),
        'std_djvel_median': med(key, 'std_djvel'),
        'std_dvxy_median': med(key, 'std_dvxy'),
        'std_proj3_median': med(key, 'std_proj3'),
        'rng_prog3_median': med(key, 'rng_prog3'),
        'rel_std_proj1_vs_actor_disp': (med(key, 'std_proj') / actor_disp1_med
                                        if actor_disp1_med > 0 else None),
        'spearman_proj1_median': med(key, 'sp_proj'),
        'spearman_proj1_frac_pos': frac_pos(key, 'sp_proj'),
        'spearman_prog1_median': med(key, 'sp_prog'),
        'spearman_prog1_frac_pos': frac_pos(key, 'sp_prog'),
        'spearman_dvxy_median': med(key, 'sp_dvxy'),
        'spearman_proj3_median': med(key, 'sp_proj3'),
        'spearman_cos_prog1_median': med(key, 'sp_cos_prog'),
        'spearman_dot_vs_cos_median': med(key, 'sp_dot_cos'),
        'spearman_dist_score_median': med(key, 'sp_dist_sc'),
        'spearman_dist_absproj_median': med(key, 'sp_dist_absproj'),
        'persistence_sp_proj1_proj3_median': med(key, 'sp_p1_p3'),
        'persistence_sign_frac': med(key, 'persist_sign'),
        'critic_best_prog1': mean_ci(key, 'cb_prog'),
        'critic_worst_prog1': mean_ci(key, 'cw_prog'),
        'random_prog1': mean_ci(key, 'rand_prog'),
        'actor_prog1': mean_ci(key, 'actor_prog'),
        'critic_decile_gap_prog1': (float(np.mean(dec_gap)), _boot_ci(dec_gap)),
        'true_decile_gap_prog1': (float(np.mean(true_gap)), _boot_ci(true_gap)),
        'critic_decile_gap_prog3': (float(np.mean(dec_gap3)), _boot_ci(dec_gap3)),
        'decile_gap_usefulness': (float(np.mean(dec_gap) / (np.mean(true_gap) + 1e-12))),
        'fall1_frac': med(key, 'fall1'), 'fall3_frac': med(key, 'fall3'),
        'score_std_median': med(key, 'score_std'),
        'score_rng_median': med(key, 'score_rng'),
        'phi_norm_std_median': med(key, 'phin_std'),
        'phi_norm_mean_median': med(key, 'phin_mean'),
        'cos_std_median': med(key, 'cos_std'),
        'mean_dist_actor': med(key, 'mean_dist_actor'),
        'per_dim_std_median': med(key, 'dim_std'),
        'clip_frac_median': med(key, 'clip_frac'),
        'raw_clip_frac_median': med(key, 'raw_clip_frac'),
        'eff_rank_median': med(key, 'eff_rank'),
        'mean_score': med(key, 'mean_score'),
    }

  # physical-effect label per sigma
  def phys_label(key):
    sp, rp = per_set[key]['std_proj1_median'], per_set[key]['rng_prog1_median']
    if sp >= MEAN_STD_PROJ or rp >= MEAN_RNG_PROG:
      return 'meaningful'
    if sp < NEGL_STD_PROJ and rp < NEGL_RNG_PROG:
      return 'negligible'
    return 'weak'
  labels = {f'local_s{s}': phys_label(f'local_s{s}') for s in SIGMAS}

  # ranking label at a sigma set
  def rank_state(key):
    ps = per_set[key]
    gap, gap_ci = ps['critic_decile_gap_prog1']
    useful = ps['decile_gap_usefulness']
    spm = ps['spearman_prog1_median'] or 0.0
    fp = ps['spearman_prog1_frac_pos'] or 0.0
    valid = (spm >= 0.2 and fp >= 0.7 and gap_ci[0] is not None
             and gap_ci[0] > 0 and useful >= 0.3)
    fails = (spm < 0.1 or gap_ci[0] is None or gap_ci[0] <= 0 or useful < 0.1)
    return {'valid': bool(valid), 'fails': bool(fails), 'spearman': spm,
            'frac_pos': fp, 'gap': gap, 'gap_ci': gap_ci, 'usefulness': useful}
  ranking = {f'local_s{s}': rank_state(f'local_s{s}') for s in SIGMAS}
  ranking['replay_nbr'] = rank_state('replay_nbr')

  # ------------------------- regimes / stratification ---------------------- #
  F = {q: np.array([f[q] for f in feats], float)
       for q in ('z', 'd0', 'speed', 'ang_speed', 'vz', 'vel_align',
                 'head_err', 'sat')}
  med_speed = float(np.nanmedian(F['speed']))
  regimes = {
      'standing_moving': (F['z'] >= 0.5) & (F['speed'] >= med_speed),
      'standing_stationary': (F['z'] >= 0.5) & (F['speed'] < med_speed),
      'low_torso': F['z'] < 0.5,
      'falling_vz': F['vz'] < -0.1,
      'near_goal': F['d0'] <= np.median(F['d0']),
      'far_goal': F['d0'] > np.median(F['d0']),
      'high_sat': F['sat'] > np.nanmedian(F['sat']),
      'low_sat': F['sat'] <= np.nanmedian(F['sat']),
      'fast_spin': F['ang_speed'] > np.nanmedian(F['ang_speed']),
      'goal_aligned_heading': F['head_err'] < np.pi / 2,
      'goal_misaligned_heading': F['head_err'] >= np.pi / 2,
      'moving_toward_goal': F['vel_align'] > 0.5,
  }
  PRIMARY = 'local_s0.05'
  regime_table = {}
  for name, mask in regimes.items():
    idx = np.where(mask)[0]
    if len(idx) < 5:
      regime_table[name] = {'n': int(len(idx))}
      continue
    sub = [per_state[PRIMARY][i] for i in idx]
    sp = np.array([s['sp_prog'] for s in sub], float)
    sp = sp[np.isfinite(sp)]
    gaps = [s['top_dec'] - s['bot_dec'] for s in sub]
    tgaps = [s['true_top'] - s['true_bot'] for s in sub]
    regime_table[name] = {
        'n': int(len(idx)),
        'std_proj1_median': float(np.median([s['std_proj'] for s in sub])),
        'rng_prog1_median': float(np.median([s['rng_prog'] for s in sub])),
        'spearman_prog1_median': float(np.median(sp)) if len(sp) else None,
        'critic_decile_gap': float(np.mean(gaps)),
        'critic_decile_gap_ci': _boot_ci(gaps),
        'true_decile_gap': float(np.mean(tgaps)),
    }

  # ------------------------------- verdict --------------------------------- #
  small_keys = [f'local_s{s}' for s in SMALL]
  large_keys = [f'local_s{s}' for s in [0.10, 0.20]]
  small_meaningful = any(labels[k] == 'meaningful' for k in small_keys)
  small_all_negl = all(labels[k] == 'negligible' for k in small_keys)
  large_meaningful = any(labels[k] == 'meaningful' for k in large_keys)
  # primary ranking sigma: largest small sigma with meaningful physics
  prim = [k for k in small_keys if labels[k] == 'meaningful']
  prim_key = prim[-1] if prim else PRIMARY
  rk = ranking[prim_key]

  # state-dependent: some regime clearly controllable+ranked while overall not
  def regime_valid(rt):
    return (rt.get('n', 0) >= 15
            and rt.get('std_proj1_median', 0) >= MEAN_STD_PROJ
            and (rt.get('spearman_prog1_median') or 0) >= 0.25
            and rt.get('critic_decile_gap_ci', (None,))[0] is not None
            and rt['critic_decile_gap_ci'][0] > 0
            and rt['critic_decile_gap'] >= 0.3 * max(rt['true_decile_gap'], 1e-9))
  valid_regimes = [n for n, rt in regime_table.items() if regime_valid(rt)]

  if small_all_negl and large_meaningful:
    verdict = 'ONLY_LARGE_PERTURBATIONS_HAVE_EFFECT'
  elif small_all_negl and not large_meaningful:
    verdict = 'LOCAL_ACTIONS_PHYSICALLY_INDISTINGUISHABLE'
  elif small_meaningful and rk['valid']:
    verdict = 'PHYSICAL_EFFECT_AND_CRITIC_RANKING_VALID'
  elif small_meaningful and not rk['valid'] and valid_regimes:
    verdict = 'STATE_DEPENDENT_CONTROLLABILITY'
  elif small_meaningful and rk['fails']:
    verdict = 'PHYSICAL_EFFECT_EXISTS_CRITIC_FAILS_TO_RANK'
  elif small_meaningful:
    # between valid and fails: decide on behavioral usefulness of the ranking
    verdict = ('PHYSICAL_EFFECT_EXISTS_CRITIC_FAILS_TO_RANK'
               if rk['usefulness'] < 0.3 else
               'PHYSICAL_EFFECT_AND_CRITIC_RANKING_VALID')
  else:
    # small sigma only 'weak': not negligible, not meaningful
    verdict = ('ONLY_LARGE_PERTURBATIONS_HAVE_EFFECT' if large_meaningful
               else 'INCONCLUSIVE')

  report = {
      'ckpt': args.ckpt, 'step': int(step), 'n_states': len(refs),
      'n_local_per_sigma': args.n_local, 'sigmas': SIGMAS,
      'protocol': 'restore exact state; candidate 1 step; +2 zero-action steps '
                  '(persistence); actor never resumes',
      'gates': G,
      'orientation_source': 'validated quaternion (matches xmat)' if use_quat
                            else 'UNVALIDATED - heading strata skipped',
      'ctrl_err_max': float(ctrl_err_max),
      'contact_data_available': bool(contact_ok),
      'actor_disp1_median': actor_disp1_med,
      'actor_prog1_mean': float(np.mean([o['prog1'] for o in actor_out])),
      'actor_fall3_frac': float(np.mean([o['fall3'] for o in actor_out])),
      'state_features_summary': {q: {'median': float(np.nanmedian(F[q])),
                                     'p10': float(np.nanpercentile(F[q], 10)),
                                     'p90': float(np.nanpercentile(F[q], 90))}
                                 for q in F},
      'jacobian_proxy': {'norm_median': float(np.median(jac_norms)),
                         'norm_p10': float(np.percentile(jac_norms, 10)),
                         'norm_p90': float(np.percentile(jac_norms, 90)),
                         'per_dim_mean_abs': np.mean(jac_dims, 0).tolist(),
                         'delta': JAC_DELTA,
                         'units': 'projected displacement (m) per action unit'},
      'replay_neighbors': {
          'k': K_NBR,
          'obs_dist_mean': float(np.mean([n['obs_dist'] for n in nbr_info])),
          'action_std_mean': float(np.mean([n['act_std'] for n in nbr_info])),
          'action_dist_actor_mean': float(np.mean([n['act_dist_actor']
                                                   for n in nbr_info]))},
      'psi_norm_mean': float(np.mean(psi_norms)),
      'thresholds': {'negligible_std_proj1': NEGL_STD_PROJ,
                     'negligible_rng_prog1': NEGL_RNG_PROG,
                     'meaningful_std_proj1': MEAN_STD_PROJ,
                     'meaningful_rng_prog1': MEAN_RNG_PROG},
      'per_set': per_set,
      'physical_labels': labels,
      'ranking_assessment': ranking,
      'primary_ranking_set': prim_key,
      'regimes': regime_table,
      'valid_regimes': valid_regimes,
      'verdict': verdict,
  }
  with open(os.path.join(args.out, 'ant_immediate_controllability.json'), 'w') as f:
    json.dump(report, f, indent=2)

  np.savez_compressed(
      os.path.join(args.out, 'ant_immediate_controllability_samples.npz'),
      state_idx=np.array(S_sid, np.int32), set_idx=np.array(S_set, np.int16),
      set_keys=np.array(set_keys), action=np.concatenate(S_act).astype(np.float32),
      score=np.array(S_sc, np.float32), cos=np.array(S_cos, np.float32),
      phi_norm=np.array(S_phin, np.float32),
      qpos_post=np.concatenate(S_qpos1), qvel_post=np.concatenate(S_qvel1),
      ref_qpos=d['qpos'][:len(refs)], ref_qvel=d['qvel'][:len(refs)],
      ref_goal=d['goal'][:len(refs)], ref_obs=d['obs'][:len(refs)],
      **{q: np.array(S_m[q], np.float32) for q in S_m})

  _plots(args.out, report, per_state, pooled, path_data, env)
  _md(args.out, report)
  print('\nVERDICT:', verdict)
  print('saved artifacts to', args.out)


# --------------------------------------------------------------------------- #
def _plots(out, r, per_state, pooled, path_data, env):
  keys_small = [f'local_s{s}' for s in SMALL]
  # 1+2: critic score (within-state centered) vs immediate proj / progress
  for q, fname in (('proj1', 'scatter_score_vs_proj1.png'),
                   ('prog1', 'scatter_score_vs_prog1.png')):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, k in zip(axes, keys_small):
      ax.scatter(pooled[k]['sc'], pooled[k][q], s=3, alpha=0.15)
      ax.axhline(0, color='k', lw=.5); ax.axvline(0, color='k', lw=.5)
      ax.set_title(f'{k} (sp med '
                   f'{r["per_set"][k]["spearman_" + q + "_median"]:.2f})')
      ax.set_xlabel('critic score (centered)')
    axes[0].set_ylabel('immediate ' + ('projected displacement' if q == 'proj1'
                                       else 'goal progress') + ' (m)')
    fig.suptitle('critic score vs immediate 1-step physical outcome')
    fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=100); plt.close()

  # 3: action distance vs physical effect (per-sigma curves)
  sig_keys = [f'local_s{s}' for s in SIGMAS]
  dist = [r['per_set'][k]['mean_dist_actor'] for k in sig_keys]
  fig, ax1 = plt.subplots(figsize=(6, 4))
  ax1.plot(dist, [r['per_set'][k]['std_proj1_median'] for k in sig_keys],
           'o-', label='std proj disp (m)')
  ax1.plot(dist, [r['per_set'][k]['rng_prog1_median'] for k in sig_keys],
           's-', label='max-min goal progress (m)')
  ax1.axhline(MEAN_STD_PROJ, color='g', ls=':', lw=1, label='meaningful thr')
  ax1.axhline(NEGL_STD_PROJ, color='r', ls=':', lw=1, label='negligible thr')
  ax1.set_xlabel('mean action distance from actor action')
  ax1.set_ylabel('within-state physical spread (m)')
  ax2 = ax1.twinx()
  ax2.plot(dist, [r['per_set'][k]['fall3_frac'] for k in sig_keys],
           'x--', color='gray', label='fall frac @3')
  ax2.set_ylabel('fall fraction @3 steps')
  ax1.legend(fontsize=7); ax1.set_title('action distance vs physical effect')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'action_distance_vs_effect.png'),
                                  dpi=100); plt.close()

  # 4: top vs bottom decile (critic and true) per set
  fig, ax = plt.subplots(figsize=(9, 4))
  x = np.arange(len(sig_keys) + 1)
  ks = sig_keys + ['replay_nbr']
  ax.bar(x - 0.3, [r['per_set'][k]['critic_decile_gap_prog1'][0] for k in ks],
         0.28, label='critic top-bot decile gap')
  ax.bar(x, [r['per_set'][k]['true_decile_gap_prog1'][0] for k in ks],
         0.28, label='true top-bot decile gap (physical range)')
  ax.bar(x + 0.3, [r['per_set'][k]['critic_decile_gap_prog3'][0] for k in ks],
         0.28, label='critic gap @3 steps')
  ax.set_xticks(x); ax.set_xticklabels(ks, rotation=20)
  ax.axhline(0, color='k', lw=.5); ax.set_ylabel('goal progress gap (m)')
  ax.legend(fontsize=8); ax.set_title('score-decile vs true-decile 1-step outcome')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'decile_gaps.png'), dpi=100)
  plt.close()

  # 5: per-state controllability range histograms
  fig, axes = plt.subplots(1, len(SIGMAS), figsize=(15, 3), sharex=False)
  for ax, s in zip(axes, SIGMAS):
    v = [st['rng_prog'] for st in per_state[f'local_s{s}']]
    ax.hist(v, bins=25)
    ax.axvline(MEAN_RNG_PROG, color='g', ls=':')
    ax.axvline(NEGL_RNG_PROG, color='r', ls=':')
    ax.set_title(f'sigma={s}'); ax.set_xlabel('max-min prog1 (m)')
  axes[0].set_ylabel('states')
  fig.suptitle('per-state physical controllability range (1 step)')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'per_state_controllability.png'),
                                  dpi=100); plt.close()

  # 6: raw dot vs cosine ranking agreement
  fig, axes = plt.subplots(1, 2, figsize=(9, 4))
  for k in sig_keys:
    axes[0].plot([r['per_set'][k]['mean_dist_actor']],
                 [r['per_set'][k]['spearman_dot_vs_cos_median']], 'o', label=k)
    axes[1].plot([r['per_set'][k]['spearman_prog1_median']],
                 [r['per_set'][k]['spearman_cos_prog1_median']], 'o', label=k)
  axes[0].set_ylim(-1.05, 1.05); axes[0].axhline(1, color='k', lw=.5, ls=':')
  axes[0].set_xlabel('mean action dist from actor')
  axes[0].set_ylabel('spearman(dot score, cosine)')
  axes[0].set_title('raw-dot vs cosine ranking agreement')
  lim = max(0.5, max(abs(np.array(axes[1].get_xlim())).max(),
                     abs(np.array(axes[1].get_ylim())).max()))
  axes[1].plot([-lim, lim], [-lim, lim], 'k:', lw=.5)
  axes[1].set_xlabel('spearman(dot, prog1) median')
  axes[1].set_ylabel('spearman(cosine, prog1) median')
  axes[1].set_title('does cosine rank physics better than dot?')
  axes[1].legend(fontsize=6)
  fig.tight_layout(); fig.savefig(os.path.join(out, 'dot_vs_cosine.png'), dpi=100)
  plt.close()

  # 7: representative local rollouts (>=10 states)
  u = env._env.unwrapped
  mz = u.maze
  fig, axes = plt.subplots(2, 5, figsize=(17, 7))
  for ax, pd in zip(axes.ravel(), path_data):
    for row in range(len(mz.maze_map)):
      for col in range(len(mz.maze_map[0])):
        if mz.maze_map[row][col] == 1:
          xx, yy = mz.cell_rowcol_to_xy(np.array([row, col]))
          s = mz.maze_size_scaling
          ax.add_patch(plt.Rectangle((xx - s / 2, yy - s / 2), s, s,
                                     color='0.9', zorder=0))
    sc = pd['scores']
    cn = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
    for p, c in zip(pd['paths'], cn):
      ax.plot(p[:, 0], p[:, 1], '-', color=plt.cm.viridis(c), lw=.9, alpha=.8)
    ax.plot(pd['actor'][:, 0], pd['actor'][:, 1], 'r-', lw=1.8, label='actor')
    ax.scatter(*pd['xy'], c='k', s=25, zorder=3)
    dx = np.ptp([p[:, 0] for p in pd['paths']]); dy = np.ptp([p[:, 1] for p in pd['paths']])
    ax.set_xlim(pd['xy'][0] - max(3 * dx, .05), pd['xy'][0] + max(3 * dx, .05))
    ax.set_ylim(pd['xy'][1] - max(3 * dy, .05), pd['xy'][1] + max(3 * dy, .05))
    gv = pd['goal'] - pd['xy']
    gv = gv / (np.linalg.norm(gv) + 1e-9) * max(3 * dx, .05) * .8
    ax.annotate('', xy=pd['xy'] + gv, xytext=pd['xy'],
                arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
  axes.ravel()[0].legend(fontsize=6)
  fig.suptitle('local sigma=0.05 candidates: 1 candidate step + 2 zero steps '
               '(color = critic score, red arrow = goal direction; zoomed)')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'representative_local_rollouts.png'), dpi=90)
  plt.close()


def _md(out, r):
  ps = r['per_set']
  L = ['# Ant immediate local controllability probe\n',
       f'**Verdict: `{r["verdict"]}`** (ckpt step {r["step"]}, '
       f'{r["n_states"]} states, {r["n_local_per_sigma"]} candidates/sigma)\n',
       f'Protocol: {r["protocol"]}.\n',
       '## Gates',
       *[f'- {k}: pass={v["pass"]}' for k, v in r['gates'].items()],
       f'- orientation: {r["orientation_source"]}',
       f'- max |ctrl - clipped action| across all rollouts: {r["ctrl_err_max"]:.2e}',
       f'- contact data available: {r["contact_data_available"]}\n',
       f'Actor 1-step displacement median: {r["actor_disp1_median"]:.4g} m; '
       f'actor prog1 mean {r["actor_prog1_mean"]:.4g} m; '
       f'actor fall@3 {r["actor_fall3_frac"]:.2f}\n',
       f'Jacobian proxy |d proj1 / d a|: median norm '
       f'{r["jacobian_proxy"]["norm_median"]:.4g} m/unit '
       f'(p10 {r["jacobian_proxy"]["norm_p10"]:.4g}, '
       f'p90 {r["jacobian_proxy"]["norm_p90"]:.4g})\n',
       '## Physical spread per candidate set (within-state, 1 step)',
       '| set | std proj1 | rng prog1 | std dvxy | std djvel | rng prog3 | '
       'fall@3 | label |',
       '|---|---|---|---|---|---|---|---|']
  labels = dict(r['physical_labels'])
  for k in ps:
    L.append(f'| {k} | {ps[k]["std_proj1_median"]:.4g} | '
             f'{ps[k]["rng_prog1_median"]:.4g} | {ps[k]["std_dvxy_median"]:.4g} | '
             f'{ps[k]["std_djvel_median"]:.4g} | {ps[k]["rng_prog3_median"]:.4g} | '
             f'{ps[k]["fall3_frac"]:.2f} | {labels.get(k, "-")} |')
  L += [f'\nThresholds: negligible if std proj1 < {r["thresholds"]["negligible_std_proj1"]}'
        f' and rng prog1 < {r["thresholds"]["negligible_rng_prog1"]}; meaningful if '
        f'std proj1 >= {r["thresholds"]["meaningful_std_proj1"]} or rng prog1 >= '
        f'{r["thresholds"]["meaningful_rng_prog1"]} (m).\n',
        '## Critic ranking per set (immediate physics)',
        '| set | sp(prog1) med | frac>0 | sp(proj1) | sp(dvxy) | sp(proj3) | '
        'critic dec gap | true dec gap | usefulness | cb prog1 | cw prog1 | '
        'rand | actor |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for k in ps:
    g = ps[k]['critic_decile_gap_prog1']; t = ps[k]['true_decile_gap_prog1']
    L.append(f'| {k} | {ps[k]["spearman_prog1_median"]:.3f} | '
             f'{ps[k]["spearman_prog1_frac_pos"]:.2f} | '
             f'{ps[k]["spearman_proj1_median"]:.3f} | '
             f'{ps[k]["spearman_dvxy_median"]:.3f} | '
             f'{ps[k]["spearman_proj3_median"]:.3f} | '
             f'{g[0]:.4g} [{g[1][0]:.4g},{g[1][1]:.4g}] | {t[0]:.4g} | '
             f'{ps[k]["decile_gap_usefulness"]:.2f} | '
             f'{ps[k]["critic_best_prog1"][0]:.4g} | '
             f'{ps[k]["critic_worst_prog1"][0]:.4g} | '
             f'{ps[k]["random_prog1"][0]:.4g} | {ps[k]["actor_prog1"][0]:.4g} |')
  L += ['\n## Candidate diversity / clipping',
        '| set | mean dist actor | per-dim std | clip frac | raw clip frac | '
        'eff rank | mean score |',
        '|---|---|---|---|---|---|---|']
  for k in ps:
    rcf = ps[k]['raw_clip_frac_median']
    L.append(f'| {k} | {ps[k]["mean_dist_actor"]:.3f} | '
             f'{ps[k]["per_dim_std_median"]:.4f} | {ps[k]["clip_frac_median"]:.3f} | '
             f'{rcf if rcf is None or np.isnan(rcf) else round(rcf, 3)} | '
             f'{ps[k]["eff_rank_median"]:.2f} | {ps[k]["mean_score"]:.2f} |')
  rn = r['replay_neighbors']
  L += [f'\nReplay neighbors: k={rn["k"]}, mean obs dist {rn["obs_dist_mean"]:.2f}, '
        f'action std {rn["action_std_mean"]:.3f}, '
        f'dist from actor {rn["action_dist_actor_mean"]:.2f}\n',
        '## Score-scale artifacts (Control 4; repr_norm=False, score = raw dot)',
        '| set | phi-norm std | phi-norm mean | cos std | sp(dot,cos) | '
        'sp(cos,prog1) | sp(dot,prog1) |',
        '|---|---|---|---|---|---|---|']
  for k in ps:
    L.append(f'| {k} | {ps[k]["phi_norm_std_median"]:.3f} | '
             f'{ps[k]["phi_norm_mean_median"]:.2f} | {ps[k]["cos_std_median"]:.4f} | '
             f'{ps[k]["spearman_dot_vs_cos_median"]:.3f} | '
             f'{ps[k]["spearman_cos_prog1_median"]:.3f} | '
             f'{ps[k]["spearman_prog1_median"]:.3f} |')
  L += [f'\npsi(g) norm mean: {r["psi_norm_mean"]:.2f}\n',
        '## Persistence (1 candidate step + 2 zero steps)',
        '| set | sp(proj1,proj3) med | sign persistence (top |proj1| decile) |',
        '|---|---|---|']
  for k in ps:
    L.append(f'| {k} | {ps[k]["persistence_sp_proj1_proj3_median"]:.3f} | '
             f'{ps[k]["persistence_sign_frac"]:.2f} |')
  L += ['\n## Control 1: action distance vs effect (local sets)',
        '| set | dist | std proj1 | fall@3 | sp(dist,score) | sp(dist,|dproj|) |',
        '|---|---|---|---|---|---|']
  for k in [f'local_s{s}' for s in r['sigmas']]:
    L.append(f'| {k} | {ps[k]["mean_dist_actor"]:.3f} | '
             f'{ps[k]["std_proj1_median"]:.4g} | {ps[k]["fall3_frac"]:.2f} | '
             f'{ps[k]["spearman_dist_score_median"]:.3f} | '
             f'{ps[k]["spearman_dist_absproj_median"]:.3f} |')
  L += ['\n## Control 3: state-regime split (at local sigma=0.05)',
        '| regime | n | std proj1 | rng prog1 | sp(prog1) med | critic gap '
        '[ci] | true gap |',
        '|---|---|---|---|---|---|---|']
  for name, rt in r['regimes'].items():
    if rt.get('n', 0) < 5:
      L.append(f'| {name} | {rt.get("n", 0)} | - | - | - | - | - |')
      continue
    ci = rt['critic_decile_gap_ci']
    L.append(f'| {name} | {rt["n"]} | {rt["std_proj1_median"]:.4g} | '
             f'{rt["rng_prog1_median"]:.4g} | {rt["spearman_prog1_median"]:.3f} | '
             f'{rt["critic_decile_gap"]:.4g} [{ci[0]:.4g},{ci[1]:.4g}] | '
             f'{rt["true_decile_gap"]:.4g} |')
  L += [f'\nRegimes passing the validity bar: {r["valid_regimes"] or "none"}',
        f'\nPrimary ranking set: `{r["primary_ranking_set"]}` -> '
        f'{json.dumps(r["ranking_assessment"][r["primary_ranking_set"]])}',
        f'\n**Verdict: `{r["verdict"]}`**']
  with open(os.path.join(out, 'ant_immediate_controllability.md'), 'w') as f:
    f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
