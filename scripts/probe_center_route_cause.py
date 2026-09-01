"""Why does the clean-dataset ant walk down the MIDDLE 39% of the time when
the dataset only walks there 10.6% of the time? Analysis only.

Composition of D_clean (284 episodes): left 0.444 / right 0.451 / center 0.106.
The learner: left 0.45 / right 0.16 / center 0.39. The left mode survives at
its data rate; the RIGHT mode is what disappears, and almost exactly that much
mass reappears in the middle.

The structural suspect: the collector draws the lane from an INDEPENDENT rng,
  base = 'left' if side_rng.random() < 0.5 else 'right'   (collect_rockfall_v2
  _pilot.py), independent of the state and of the mask,
so the lane is exogenous randomness that is NOT a function of anything in the
58-dim observation. The conditional action distribution at the corridor
entrance is therefore genuinely bimodal, while the actor is a single tanh-
Gaussian trained with a BC log-likelihood term (crl/losses.py: loss =
bc*bc_nll + (1-bc)*(alpha*logp - Q)). A unimodal density fit to a 50/50
bimodal target puts its mean between the modes -- and the corridor is wide
enough that "between the modes" is itself a drivable route: the middle.

Tests:
  1  START-STATE REPLAY. Restore the exact t=0 state of each clean episode
     (obs[0] carries the full qpos/qvel), force that episode's mask/severity,
     roll the learner out, and record which lane it picks. If the lane is
     exogenous, the learner's lane must be statistically INDEPENDENT of the
     episode's recorded base_side -- it cannot know something the state does
     not contain. Reported as a contingency table + chi-square.
  2  POLICY SPREAD BY PHASE. The Gaussian scale and |tanh(loc)| of the actor
     at dataset states, binned by corridor x. A unimodal density covering two
     modes has to widen where the modes disagree; once the lane is committed
     there is only one mode left to fit.
  3  LATERAL PROFILE. Learner vs teacher mean y(x) on the first pass, teacher
     split by whether the site ahead is armed. This is the detour signature:
     the teacher dips below the |y|=1.0 band floor at ARMED windows only; a
     flat learner profile means no detour of either kind.

Usage:
  python scripts/probe_center_route_cause.py            # 200 replayed starts
  python scripts/probe_center_route_cause.py --n 40     # smoke
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from leak_probe_clean_ant import set_state  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
CLEAN = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
         'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
SIDECAR = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot_sidecar.npz')
SEED = 77_101
P_ACTIVE = 0.30
HORIZON = 800
#: the first pass through the hazard corridor, before the teacher's handoff.
ZONE = (2.3, 5.7)


def build_full(cfg, ckpt_path):
  """Deterministic action AND the raw Gaussian params, for test 2."""
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  @jax.jit
  def dist(o):
    d = nets.policy_network.apply(params, o)
    return jnp.tanh(d.loc), d.scale

  return act, dist, int(step)


def qpos_qvel_from_obs(o):
  """obs = [xy(2) | qpos[2:15](13) | qvel[:14](14) | goal(29)]."""
  qpos = np.concatenate([o[:2], o[2:2 + RA.NQ_ANT - 2]]).astype(float)
  qvel = o[RA.NQ_ANT:RA.NQ_ANT + RA.NV_ANT].astype(float)
  return qpos, qvel


def roll_lane(env, act, o, horizon):
  """Roll the learner and label the lane it takes on the first pass."""
  ys, passed = [], False
  hit, dead_at = 0.0, -1
  for t in range(horizon):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    if not passed and ZONE[0] <= x <= ZONE[1]:
      ys.append(y)
    if x >= RP.HANDOFF_X or y >= 2.0:
      passed = True
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
      break
  m = float(np.mean(ys)) if ys else 0.0
  return {'lane': ('left' if m > 0.5 else 'right' if m < -0.5 else 'center'),
          'mean_y': round(m, 3), 'success': float(hit > 0),
          'dead': dead_at >= 0, 'steps': int(t + 1)}


def chi2_2xk(table):
  """Pearson chi-square on a 2xK contingency table + Cramer's V. No scipy."""
  o = np.asarray(table, float)
  n = o.sum()
  if n == 0:
    return None
  e = np.outer(o.sum(1), o.sum(0)) / n
  keep = e > 0
  chi2 = float(((o[keep] - e[keep]) ** 2 / e[keep]).sum())
  dof = (o.shape[0] - 1) * (o.shape[1] - 1)
  # survival function of chi2 for the small dof this probe uses
  from math import exp, sqrt, pi, erfc
  if dof == 1:
    p = erfc(sqrt(chi2 / 2.0))
  elif dof == 2:
    p = exp(-chi2 / 2.0)
  elif dof == 3:
    p = erfc(sqrt(chi2 / 2.0)) + sqrt(2 * chi2 / pi) * exp(-chi2 / 2.0)
  else:
    p = float('nan')
  v = float(np.sqrt(chi2 / (n * (min(o.shape) - 1)))) if n else None
  return {'chi2': round(chi2, 3), 'dof': dof, 'p': round(float(p), 4),
          'cramers_v': round(v, 3), 'n': int(n)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--n', type=int, default=200, help='clean starts to replay')
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  d = np.load(CLEAN, allow_pickle=True)
  s = np.load(SIDECAR, allow_pickle=True)
  dead = np.asarray(s['dead'], bool)
  clean_idx = np.where(~dead)[0]          # clean npz rows follow this order
  route = np.asarray(s['route'])[clean_idx]
  base_side = np.asarray(s['base_side'])[clean_idx]
  masks = np.asarray(s['rockfall_mask'])[clean_idx]
  sevs = np.asarray(s['severity'])[clean_idx]
  obs = d['obs']
  assert len(obs) == len(clean_idx), (len(obs), len(clean_idx))
  print(f'clean episodes {len(obs)} | route '
        f'{ {k: int((route == k).sum()) for k in ("left", "right", "center")} }',
        flush=True)

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = float(args.p_active)
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.rockfall_reset_fix = True
  act, dist, step = build_full(cfg, args.ckpt)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)

  # ---- test 1: replay the dataset's own start states -----------------------
  n = min(args.n, len(obs))
  sel = np.linspace(0, len(obs) - 1, n).astype(int)
  rows = []
  for j, e in enumerate(sel):
    o0 = obs[e, 0]
    qpos, qvel = qpos_qvel_from_obs(o0)
    goal = o0[29:31].copy()
    o = set_state(env, qpos, qvel, goal, tuple(int(b) for b in masks[e]),
                  tuple(str(x) for x in sevs[e]))
    r = roll_lane(env, act, o, args.horizon)
    r.update({'ep': int(clean_idx[e]), 'data_route': str(route[e]),
              'data_base_side': str(base_side[e])})
    rows.append(r)
    if (j + 1) % 25 == 0:
      print(f'  [1] {j + 1}/{n} starts replayed', flush=True)

  lanes = ('left', 'center', 'right')
  side_rows = [r for r in rows if r['data_route'] != 'center']
  tab = [[sum(1 for r in side_rows
              if r['data_base_side'] == b and r['lane'] == L) for L in lanes]
         for b in ('left', 'right')]
  chi = chi2_2xk(tab)
  learner_mix = {L: round(sum(r['lane'] == L for r in rows) / len(rows), 3)
                 for L in lanes}
  data_mix = {L: round(float((route == L).mean()), 3) for L in lanes}
  print('\n[1] learner lane on replayed dataset starts:', learner_mix)
  print('    dataset lane composition               :', data_mix)
  print('    contingency (rows = dataset base_side left/right, '
        'cols = learner left/center/right):')
  for b, r_ in zip(('left', 'right'), tab):
    print(f'      data {b:5s} -> {r_}')
  print('    chi-square:', chi)

  # ---- test 2: policy spread by corridor phase -----------------------------
  X = np.asarray(s['step_torso_x'])[clean_idx]
  bank = {'entrance x<0.8': [], 'lane committed 2.3<x<5.7': [],
          'post-handoff x>6.5': []}
  for e in sel[:120]:
    ep = obs[e]
    xs = X[e]
    T = min(len(ep) - 1, np.sum(np.isfinite(xs)))
    for t in range(0, int(T), 7):
      x = float(ep[t, 0])
      if x < 0.8:
        bank['entrance x<0.8'].append(ep[t])
      elif ZONE[0] <= x <= ZONE[1]:
        bank['lane committed 2.3<x<5.7'].append(ep[t])
      elif x > 6.5:
        bank['post-handoff x>6.5'].append(ep[t])
  spread = {}
  for k, v in bank.items():
    if not v:
      continue
    a, sc = dist(jnp.asarray(np.asarray(v)))
    a, sc = np.asarray(a), np.asarray(sc)
    spread[k] = {'n_states': len(v),
                 'mean_scale': round(float(sc.mean()), 4),
                 'median_scale': round(float(np.median(sc)), 4),
                 'mean_abs_action': round(float(np.abs(a).mean()), 4),
                 'saturated_frac': round(float((np.abs(a) > 0.99).mean()), 4)}
  print('\n[2] actor spread by phase:', json.dumps(spread, indent=2))

  # ---- test 4: is the actor's action the MEAN of the two data modes? -------
  #: restricted to t < 12, where every episode is still in the shared reset
  #: pose, so averaging actions across episodes is gait-phase matched.
  TMAX = 12
  aL = {t: [] for t in range(TMAX)}
  aR = {t: [] for t in range(TMAX)}
  sL = {t: [] for t in range(TMAX)}
  sR = {t: [] for t in range(TMAX)}
  acts = d['act']
  for e in range(len(obs)):
    if route[e] == 'center':
      continue
    tgt_a, tgt_s = ((aL, sL) if base_side[e] == 'left' else (aR, sR))
    for t in range(TMAX):
      tgt_a[t].append(acts[e, t])
      tgt_s[t].append(obs[e, t])
  d4 = []
  for t in range(TMAX):
    if not aL[t] or not aR[t]:
      continue
    mL = np.mean(aL[t], axis=0)
    mR = np.mean(aR[t], axis=0)
    mid = 0.5 * (mL + mR)
    st = np.asarray(sL[t] + sR[t])
    pi = np.asarray(act(jnp.asarray(st))).mean(axis=0)
    d4.append({'t': t,
               'dist_to_left_mode': round(float(np.linalg.norm(pi - mL)), 4),
               'dist_to_right_mode': round(float(np.linalg.norm(pi - mR)), 4),
               'dist_to_midpoint': round(float(np.linalg.norm(pi - mid)), 4),
               'mode_separation': round(float(np.linalg.norm(mL - mR)), 4)})
  nearest = [min(('left', r['dist_to_left_mode']),
                 ('right', r['dist_to_right_mode']),
                 ('midpoint', r['dist_to_midpoint']), key=lambda z: z[1])[0]
             for r in d4]
  T4 = {'per_step': d4,
        'nearest_counts': {k: nearest.count(k)
                           for k in ('left', 'right', 'midpoint')},
        'mean_dist': {k: round(float(np.mean([r[f'dist_to_{k}'] for r in d4])), 4)
                      for k in ('left_mode', 'right_mode', 'midpoint')}
        if d4 else None}
  print('\n[4] actor action vs the two data modes (t<12, phase matched):')
  for r in d4:
    print(f"    t={r['t']:2d}  |pi-left|={r['dist_to_left_mode']:.3f}  "
          f"|pi-right|={r['dist_to_right_mode']:.3f}  "
          f"|pi-midpoint|={r['dist_to_midpoint']:.3f}  "
          f"(modes are {r['mode_separation']:.3f} apart)")
  print('    nearest:', T4['nearest_counts'])

  rep = {'ckpt': args.ckpt, 'step': step, 'n_replayed': n,
         'T4_action_mode_averaging': T4,
         'dataset_lane_composition': data_mix,
         'learner_lane_on_dataset_starts': learner_mix,
         'contingency_base_side_x_learner_lane': {
             'rows': ['data base_side left', 'data base_side right'],
             'cols': list(lanes), 'table': tab, 'chi_square': chi},
         'actor_spread_by_phase': spread,
         'episodes': rows}
  p = os.path.join(args.out_dir, 'center_route_cause.json')
  with open(p, 'w') as f:
    json.dump(rep, f, indent=2)
  print('\n->', p, flush=True)


if __name__ == '__main__':
  main()
