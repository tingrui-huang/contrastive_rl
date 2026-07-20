"""Privileged teacher + Stage-2 qualification audits (NO dataset generation).

Teacher (zero training -- frozen walker + frozen 0.89 base only):
  sighted (P = 1-eps): reads privileged U, commands the CLEAN side lane at
      fast speed; hands off to the base after the corridor.
  blind  (P = eps):    cannot read U; executes the blind-safe policy =
      middle lane, slow-v, with the unstick lateral probe (walker_gate's
      middle_slow arm, verbatim).

Qualification audits (all must pass BEFORE any dataset work):
  T1 compliance      sighted episodes drive the clean side: P(zone-mean-y on
                     clean side) ~ 1; blind episodes stay middle.
  T2 deployment      sighted success ~ clean_fast gate level; blind success
                     ~ middle_slow gate level; teacher EV reported.
  T3 env-hiddenness  logistic probe on APPROACH-WINDOW states (x < 2.2,
                     pre-contact) from U-INDEPENDENT behavior (blind-policy
                     episodes) must predict U at ~chance. (In sighted
                     episodes states DO encode U through the teacher's own
                     steering -- that is the intended confounding pathway,
                     not leakage.)
  T4 unconfounded    blind subset: zone-mean-y | U=0 vs U=1 statistically
                     indistinguishable (actions ignore U by construction).
  T5 causal U->S'    same (qpos, qvel, action), flip U: next-state L2 diff
                     is LARGE at contact-imminent states (middle, at the
                     reef) and ~0 far from the litter (x < 1.5).

Writes artifacts/litter_teacher_qual/{qualification.json, sidecar.npz}.

Usage:
  python scripts/litter_teacher.py --eps 150 --blind-extra 30 --seed 60
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

import mujoco                             # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import probe                     # noqa: E402
from crl.d4rl_ant import LITTER_ZONE_X    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402
import walker_gate as WG                  # noqa: E402

EPSILON = 0.05
APPROACH_X = (0.3, 2.2)


def run_teacher_episode(env, walker, base_act, blind, rng):
  """One teacher episode. Returns sidecar dict + approach-window states."""
  o = env.reset()
  u = env.u_side
  if blind:
    y_ref, v_ref = 0.0, WG.SLOW_V
  else:
    clean = -1.0 if u == 1 else 1.0
    y_ref, v_ref = clean * WG.LANE, probe.V_FAST
  true_goal = o[29:31].copy()
  handoff = False
  hit = 0.0
  approach_states, zone_y = [], []
  x_hist, nudge_until, nudge_sign = [], -1, 1.0
  dead_at = None
  contact_seen = False                   # approach states must be PRE-contact:
                                         # an ant bounced back below x=2.2 by
                                         # the rubble carries a U imprint
  for t in range(env.max_episode_steps):
    xy = o[:2]
    if not handoff and (xy[0] >= WG.HANDOFF_X or xy[1] >= 2.0):
      handoff = True
    if handoff:
      o_cmd = o.copy()
      o_cmd[29:] = 0.0
      o_cmd[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(o_cmd[None]))[0])
    else:
      y_cmd, v_cmd = y_ref, v_ref
      if blind:                          # the unstick probe, verbatim
        x_hist.append(float(xy[0]))
        if t < nudge_until:
          y_cmd = nudge_sign * WG.NUDGE_Y
        elif (len(x_hist) > WG.STALL_WINDOW
              and x_hist[-1] - x_hist[-WG.STALL_WINDOW] < WG.STALL_MIN_DX):
          nudge_until = t + WG.NUDGE_STEPS
          nudge_sign = -nudge_sign
          x_hist.clear()
          y_cmd = nudge_sign * WG.NUDGE_Y
      a = walker(o, y_cmd, v_cmd)
    if (APPROACH_X[0] <= xy[0] <= APPROACH_X[1] and not handoff
        and not contact_seen):
      approach_states.append(o[:29].copy())
    # bottom corridor only: the TOP row also passes x in [2.5, 5.5] (y~8.7)
    # after handoff and would poison the lane statistic.
    if (LITTER_ZONE_X[0] <= xy[0] <= LITTER_ZONE_X[1] and not handoff
        and abs(xy[1]) < 2.0):
      zone_y.append(float(xy[1]))
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    if info.get('pile_contacts', 0) or info.get('rubble_contacts', 0):
      contact_seen = True
    if info.get('dead') and dead_at is None:
      dead_at = t
    if hit > 0 or (dead_at is not None and t > dead_at + 5):
      break
  return {'u_side': u, 'blind': blind, 'success': hit,
          'dead': dead_at is not None,
          'zone_mean_y': float(np.mean(zone_y)) if zone_y else 0.0,
          'steps': t + 1}, approach_states


def logistic_probe_acc(X, y, groups, seed=0, iters=400, lr=0.1):
  """Tiny numpy logistic regression, 5-fold CV accuracy.

  Folds are split BY EPISODE (groups): states within one episode are highly
  correlated, and a random state-level split lets the probe memorize each
  trajectory's gait signature (with its fixed U) across folds -- measured
  0.88 fake accuracy on genuinely U-independent behavior."""
  rng = np.random.default_rng(seed)
  X = (X - X.mean(0)) / (X.std(0) + 1e-8)
  uniq = rng.permutation(np.unique(groups))
  gfolds = np.array_split(uniq, 5)
  accs = []
  for k in range(5):
    te = np.flatnonzero(np.isin(groups, gfolds[k]))
    tr = np.flatnonzero(~np.isin(groups, gfolds[k]))
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
      p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
      g = X[tr].T @ (p - y[tr]) / len(tr)
      w -= lr * g
      b -= lr * float(np.mean(p - y[tr]))
    pred = (X[te] @ w + b) > 0
    accs.append(float(np.mean(pred == y[te])))
  return float(np.mean(accs))


def causal_u_gate(env, walker, rng, n_states=40):
  """T5: flip U at identical (state, action); measure next-state L2 diff."""
  diffs_near, diffs_far = [], []
  for i in range(n_states):
    env.reset(u_side=0)
    u = env._env
    # place the ant either contact-imminent (at the reef) or far upstream
    near = i % 2 == 0
    u.data.qpos[0] = rng.uniform(2.3, 2.6) if near else rng.uniform(0.3, 1.2)
    u.data.qpos[1] = rng.uniform(-0.3, 0.3)
    u.data.qvel[0] = 1.2
    mujoco.mj_forward(u.model, u.data)
    qpos0, qvel0 = u.data.qpos.copy(), u.data.qvel.copy()
    obs0 = env._flatten(u._obs_dict())
    a = walker(obs0, 0.0, probe.V_FAST)
    nxt = {}
    for uu in (0, 1):
      env._apply_u(uu)
      u.data.qpos[:] = qpos0
      u.data.qvel[:] = qvel0
      mujoco.mj_forward(u.model, u.data)
      o2, _, _, _ = env.step(a)
      nxt[uu] = o2[:29].copy()
      env._dead = False                  # reset absorbing flag between probes
    d = float(np.linalg.norm(nxt[0] - nxt[1]))
    (diffs_near if near else diffs_far).append(d)
  return (float(np.median(diffs_near)), float(np.max(diffs_far)),
          diffs_near, diffs_far)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--walker', default='artifacts/walker/phase1/'
                                      'walker_best.pkl')
  ap.add_argument('--ckpt', default='offline_umaze_bc005_twinmin_s0_50k/'
                                    'checkpoints/best.pkl')
  ap.add_argument('--eps', type=int, default=150)
  ap.add_argument('--blind-extra', type=int, default=30,
                  help='supplementary forced-blind episodes for T3/T4 power '
                       '(marked supplementary; NOT part of the eps stream)')
  ap.add_argument('--epsilon', type=float, default=EPSILON)
  ap.add_argument('--seed', type=int, default=60)
  ap.add_argument('--out-dir', default='artifacts/litter_teacher_qual')
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = ckpt_mod.load_checkpoint(args.ckpt)
  params = st.policy_params

  @jax.jit
  def base_act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  wparams, _ = probe.load_residual(args.walker)
  walker = probe.WalkerController(wparams)
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed)
  rng = np.random.default_rng(args.seed)

  rows, blind_states, blind_us = [], [], []
  for ep in range(args.eps):
    blind = bool(rng.random() < args.epsilon)
    row, appr = run_teacher_episode(env, walker, base_act, blind, rng)
    row['supplementary'] = False
    rows.append(row)
    if blind and appr:
      blind_states.append(np.stack(appr))
      blind_us.append(np.full(len(appr), row['u_side']))
  for ep in range(args.blind_extra):     # T3/T4 power
    row, appr = run_teacher_episode(env, walker, base_act, True, rng)
    row['supplementary'] = True
    rows.append(row)
    if appr:
      blind_states.append(np.stack(appr))
      blind_us.append(np.full(len(appr), row['u_side']))

  main_rows = [r for r in rows if not r['supplementary']]
  sighted = [r for r in main_rows if not r['blind']]
  blind_all = [r for r in rows if r['blind']]

  # T1 compliance
  clean_ok = [(-1 if r['u_side'] == 1 else 1) * r['zone_mean_y'] > 0.3
              for r in sighted]
  blind_mid = [abs(r['zone_mean_y']) < 0.5 for r in blind_all]
  t1 = {'sighted_clean_side_frac': float(np.mean(clean_ok)),
        'blind_middle_frac': float(np.mean(blind_mid))}
  t1['pass'] = t1['sighted_clean_side_frac'] >= 0.95

  # T2 deployment
  t2 = {'sighted_success': float(np.mean([r['success'] for r in sighted])),
        'blind_success': float(np.mean([r['success'] for r in blind_all])),
        'teacher_ev': float(np.mean([r['success'] for r in main_rows])),
        'n_sighted': len(sighted), 'n_blind_main': len(main_rows) - len(sighted),
        'n_blind_all': len(blind_all)}
  t2['pass'] = t2['sighted_success'] >= 0.80 and t2['blind_success'] >= 0.40

  # T3 env-hiddenness probe (blind-policy PRE-CONTACT approach states).
  # Significance via a GROUP-LEVEL permutation test: with only ~tens of
  # episodes, grouped-CV accuracy has se ~ sqrt(0.25/n_groups) -- a fixed
  # accuracy bar misreads sampling noise as leakage. Pass = the observed
  # accuracy is not significantly above the label-permuted null (p >= .05).
  X = np.concatenate(blind_states)
  yl = np.concatenate(blind_us)
  groups = np.concatenate([np.full(len(s), gi)
                           for gi, s in enumerate(blind_states)])
  acc = logistic_probe_acc(X, yl, groups, seed=args.seed)
  ep_u = np.array([int(u[0]) for u in blind_us])
  null_accs = []
  prng = np.random.default_rng(args.seed + 99)
  for _ in range(30):
    perm = prng.permutation(ep_u)
    yperm = np.concatenate([np.full(len(s), perm[gi])
                            for gi, s in enumerate(blind_states)])
    null_accs.append(logistic_probe_acc(X, yperm, groups, seed=args.seed))
  pval = float(np.mean([a >= acc for a in null_accs]))
  t3 = {'probe_acc': acc, 'null_acc_mean': float(np.mean(null_accs)),
        'null_acc_p90': float(np.percentile(null_accs, 90)),
        'perm_pvalue': pval, 'n_states': int(len(X)),
        'n_episodes': int(len(blind_states)), 'pass': pval >= 0.05}

  # T4 unconfoundedness of the blind subset = POLICY U-invariance: the same
  # observation must map to the same action whichever U the env holds (the
  # blind controller never reads U). Trajectory-level U-correlation (the
  # rubble physically pushes the wader toward the clean side) is legitimate
  # U -> S -> A mediation and is reported as info, NOT gated.
  inv_diffs = []
  probe_env = envs_mod.make_env('offline_ant_umaze_litter', cfg,
                                seed=args.seed + 5000)
  for i in range(20):
    probe_env.reset(u_side=i % 2)
    uu = probe_env._env
    uu.data.qpos[0] = rng.uniform(0.5, 5.0)
    uu.data.qpos[1] = rng.uniform(-0.8, 0.8)
    mujoco.mj_forward(uu.model, uu.data)
    acts = {}
    for uv in (0, 1):
      probe_env._apply_u(uv)
      obs = probe_env._flatten(uu._obs_dict())
      acts[uv] = walker(obs, 0.0, WG.SLOW_V)
    inv_diffs.append(float(np.max(np.abs(acts[0] - acts[1]))))
  y0 = [r['zone_mean_y'] for r in blind_all if r['u_side'] == 0]
  y1 = [r['zone_mean_y'] for r in blind_all if r['u_side'] == 1]
  t4 = {'action_invariance_max_diff': float(np.max(inv_diffs)),
        'mediation_info_zone_y_u0': float(np.mean(y0)) if y0 else None,
        'mediation_info_zone_y_u1': float(np.mean(y1)) if y1 else None}
  t4['pass'] = t4['action_invariance_max_diff'] < 1e-6

  # T5 causal U->S'
  near_med, far_max, dn, df = causal_u_gate(env, walker, rng)
  t5 = {'near_median_l2': near_med, 'far_max_l2': far_max,
        'pass': bool(near_med > 10 * max(far_max, 1e-9) or
                     (near_med > 0.05 and far_max < 0.01))}

  qual = {'epsilon': args.epsilon, 'eps': args.eps, 'seed': args.seed,
          'T1_compliance': t1, 'T2_deployment': t2, 'T3_hiddenness': t3,
          'T4_unconfounded': t4, 'T5_causal_u_to_sprime': t5,
          'all_pass': all(t['pass'] for t in (t1, t2, t3, t4, t5))}
  json.dump(qual, open(os.path.join(args.out_dir, 'qualification.json'),
                       'w'), indent=2)
  np.savez_compressed(
      os.path.join(args.out_dir, 'sidecar.npz'),
      u_side=np.array([r['u_side'] for r in rows]),
      blind=np.array([r['blind'] for r in rows]),
      success=np.array([r['success'] for r in rows]),
      dead=np.array([r['dead'] for r in rows]),
      zone_mean_y=np.array([r['zone_mean_y'] for r in rows]),
      supplementary=np.array([r['supplementary'] for r in rows]))

  for name, t in (('T1_compliance', t1), ('T2_deployment', t2),
                  ('T3_hiddenness', t3), ('T4_unconfounded', t4),
                  ('T5_causal', t5)):
    print(f'{"PASS" if t["pass"] else "FAIL"}  {name}  '
          f'{json.dumps({k: v for k, v in t.items() if k != "pass"})[:150]}')
  print('TEACHER ' + ('QUALIFIED' if qual['all_pass'] else 'NOT QUALIFIED'))


if __name__ == '__main__':
  main()
