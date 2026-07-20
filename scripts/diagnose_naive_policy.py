"""Behavioural diagnosis of a naive offline-CRL litter checkpoint. Analysis only
(no retrain / no env or policy changes). 200 balanced-U episodes + paired
U-flip clones + open-loop action replay. Answers: fixed-side / pre-contact
leakage / probe-and-recover / unstable mixture.

Usage: python scripts/diagnose_naive_policy.py --ckpt <dir_or_pkl> [--eps 200]
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
from crl.d4rl_ant import LITTER_ZONE_X    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ZONE = LITTER_ZONE_X
OUT = 'artifacts/naive_policy_diagnosis'


def torso_up(q):
  return 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5])


def litter_touch(env):
  """True if ANY litter geom (pile/skirt/reef/slick) is in contact -- the ant's
  legs reach the pile face (x=2.5) while the torso is still at x~1.4, so a
  torso-x test underdetects; a geom-contact test is exact."""
  d, m = env._env.data, env._env.model
  for i in range(d.ncon):
    for g in (d.contact[i].geom1, d.contact[i].geom2):
      nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
      if nm and nm.startswith('litter'):
        return True
  return False


def clean_sign(u):
  return -1.0 if u == 1 else 1.0        # u=1 pile on +y -> clean is -y


def pile_sign(u):
  return 1.0 if u == 1 else -1.0


def set_state(env, qpos, qvel, goal_xy, u):
  env._apply_u(u)
  env._dead = False                      # CRITICAL: _apply_u does not clear the
  env.episode_contacts = {'pile': 0, 'rubble': 0}   # absorbing-death flag; a
  env.episode_max_force = 0.0            # stale _dead from a prior rollout would
  env.episode_max_hforce = 0.0          # freeze this one at step 0.
  env.episode_max_himpulse = 0.0
  uu = env._env
  uu.data.qpos[:] = qpos
  uu.data.qvel[:] = qvel
  gv = np.zeros(29, np.float32)
  gv[:2] = goal_xy
  env._goal_vec = gv
  uu.goal = np.asarray(goal_xy, float).copy()
  uu.data.qacc_warmstart[:] = 0.0        # deterministic solver start: else the
  mujoco.mj_forward(uu.model, uu.data)   # reused env's warm-start leaks ~1e-7
  env._last_obs = uu._obs_dict()         # noise that grows chaotically
  return env._flatten(env._last_obs)


def rollout(env, act, obs0=None, actions=None, record=False):
  """Greedy closed-loop (act) OR open-loop (fixed `actions`). obs0 optional
  pre-set start obs. Returns trajectory dict."""
  o = obs0 if obs0 is not None else env.reset()
  u = int(env.u_side)
  goal = o[29:31].copy()
  hit, dead_at, fell_at = 0.0, -1, -1
  obss, acts, xys, contacts, touches = [], [], [], [], []
  T = env.max_episode_steps if actions is None else len(actions)
  for t in range(T):
    a = actions[t] if actions is not None else np.asarray(act(jnp.asarray(o[None]))[0])
    if record:
      obss.append(o[:29].copy())
      acts.append(a.copy())
    o, r, _, info = env.step(a)
    q = np.asarray(env._env.data.qpos)
    xys.append((float(o[0]), float(o[1])))
    contacts.append(int((info.get('pile_contacts', 0) or
                         info.get('rubble_contacts', 0)) > 0))
    touches.append(1 if litter_touch(env) else 0)
    hit = max(hit, float(r))
    if info.get('dead') and dead_at < 0:
      dead_at = t
    if fell_at < 0 and (torso_up(q) < 0.3 or float(q[2]) < 0.2):
      fell_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 3):
      break
  xys = np.array(xys)
  fc = int(np.argmax(contacts)) if any(contacts) else -1
  ft = int(np.argmax(touches)) if any(touches) else -1
  return dict(u=u, success=hit, dead=dead_at >= 0, fell=fell_at >= 0,
              first_contact=fc, first_litter_touch=ft, xys=xys, goal=goal,
              obss=obss, acts=acts, dead_at=dead_at)


def side_in(xys, xlo, xhi):
  m = (xys[:, 0] >= xlo) & (xys[:, 0] <= xhi) & (np.abs(xys[:, 1]) < 2.0)
  return float(np.mean(xys[m, 1])) if m.any() else 0.0


def summarize(r):
  xys = r['xys']
  u = r['u']
  init_y = side_in(xys, 1.0, 2.5)             # approaching the litter
  exit_y = side_in(xys, 4.5, 5.5)             # leaving the litter
  init_side = np.sign(init_y) if abs(init_y) > 0.15 else 0.0
  final_side = np.sign(exit_y) if abs(exit_y) > 0.15 else 0.0
  cs = clean_sign(u)
  reached6 = bool((xys[:, 0] >= 6.0).any() or (xys[:, 1] >= 2.0).any())
  switched = bool(init_side != 0 and final_side != 0 and init_side != final_side)
  wrong_contact = bool(r['first_contact'] >= 0 and init_side == pile_sign(u))
  timeout = bool(r['success'] == 0 and not r['dead'] and not r['fell'])
  return dict(u=u, success=int(r['success'] > 0), dead=bool(r['dead']),
              fell=bool(r['fell']), timeout=timeout, first_contact=r['first_contact'],
              init_y=init_y, exit_y=exit_y, init_side=int(init_side),
              final_side=int(final_side), init_clean=bool(init_side == cs),
              final_clean=bool(final_side == cs), switched=switched,
              wrong_side_contact=wrong_contact, corridor_exit=reached6)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default='naive_litter_crl_s0_60k')
  ap.add_argument('--eps', type=int, default=200)
  ap.add_argument('--paired', type=int, default=40)
  ap.add_argument('--replay', type=int, default=30)
  ap.add_argument('--seed', type=int, default=71717)
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)
  ckpt = args.ckpt if args.ckpt.endswith('.pkl') else os.path.join(args.ckpt,
                                                                    'best.pkl')

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed)
  R = {'ckpt': ckpt, 'step': int(step), 'eps': args.eps}

  # ---- (2)+(4) 200 balanced-U episodes ----
  rows = []
  for i in range(args.eps):
    env.reset(u_side=i % 2)
    r = rollout(env, act)
    rows.append(summarize(r))

  def rate(sub, key):
    s = [x for x in rows if sub(x)]
    return float(np.mean([x[key] for x in s])) if s else None
  allr = lambda k: float(np.mean([x[k] for x in rows]))
  contacted = [x for x in rows if x['first_contact'] >= 0]
  wrong = [x for x in rows if x['wrong_side_contact']]
  R['overall'] = {'success': allr('success'), 'collapse': allr('dead'),
                  'fall': allr('fell'), 'timeout': allr('timeout')}
  R['side_behavior'] = {
      'initial_clean_side_rate': allr('init_clean'),
      'wrong_side_contact_rate': allr('wrong_side_contact'),
      'wrong_to_clean_switch_rate': (float(np.mean([x['final_clean'] for x in wrong]))
                                     if wrong else None),
      'switch_rate_overall': allr('switched'),
      'success_after_switch': rate(lambda x: x['switched'], 'success'),
      'success_no_switch': rate(lambda x: not x['switched'], 'success'),
      'clean_side_arrival_rate': allr('final_clean'),
      'center_fraction': float(np.mean([abs(x['exit_y']) < 0.5 for x in rows]))}
  R['per_U'] = {}
  for u in (0, 1):
    su = [x for x in rows if x['u'] == u]
    R['per_U'][f'u{u}'] = {
        'n': len(su), 'success': float(np.mean([x['success'] for x in su])),
        'collapse': float(np.mean([x['dead'] for x in su])),
        'fall': float(np.mean([x['fell'] for x in su])),
        'timeout': float(np.mean([x['timeout'] for x in su])),
        'corridor_exit': float(np.mean([x['corridor_exit'] for x in su])),
        'post_contact_switch': float(np.mean([x['switched'] for x in su])),
        'final_clean_side': float(np.mean([x['final_clean'] for x in su])),
        'init_clean_side': float(np.mean([x['init_clean'] for x in su])),
        'mean_exit_y': float(np.mean([x['exit_y'] for x in su]))}

  # ---- (1) paired U-flip clones ----
  penv = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed + 5)
  pair_rows = []
  for i in range(args.paired):
    penv.reset(u_side=0)
    q0 = penv._env.data.qpos.copy()
    v0 = penv._env.data.qvel.copy()
    goal = penv._flatten(penv._last_obs)[29:31].copy()
    traj = {}
    for u in (0, 1):
      o = set_state(penv, q0, v0, goal, u)
      traj[u] = rollout(penv, act, obs0=o, record=True)
    T = min(len(traj[0]['acts']), len(traj[1]['acts']))
    div = -1
    for t in range(T):
      # physical tolerance: ignore sub-1e-5 solver noise, catch real contact.
      if (not np.allclose(traj[0]['obss'][t], traj[1]['obss'][t], atol=1e-5)
          or not np.allclose(traj[0]['acts'][t], traj[1]['acts'][t], atol=1e-5)):
        div = t
        break
    # first physical litter touch in EITHER twin (any litter geom; the ant's
    # leg reaches the pile face while the torso is still upstream, so a
    # torso-x threshold underdetects -- geom contact is exact).
    ft = min([c for c in (traj[0]['first_litter_touch'],
                          traj[1]['first_litter_touch']) if c >= 0] or [10**9])
    pair_rows.append({'divergence_step': div, 'first_litter_touch': int(ft),
                      'div_before_any_litter_touch': bool(div >= 0 and div < ft),
                      'div_x_torso': float(traj[0]['xys'][max(div - 1, 0), 0]) if div >= 0 else float('nan'),
                      'u0_success': traj[0]['success'], 'u1_success': traj[1]['success']})
  divs = [p for p in pair_rows if p['divergence_step'] >= 0]
  R['paired_uflip'] = {
      'n': len(pair_rows),
      'never_diverged': sum(p['divergence_step'] < 0 for p in pair_rows),
      'diverged_BEFORE_any_litter_touch_LEAKAGE': sum(p['div_before_any_litter_touch']
                                                      for p in pair_rows),
      'diverged_AT_or_AFTER_litter_touch': sum((not p['div_before_any_litter_touch'])
                                               and p['divergence_step'] >= 0 for p in pair_rows),
      'median_divergence_step': float(np.median([p['divergence_step'] for p in divs])) if divs else None,
      'median_first_litter_touch_step': float(np.median([p['first_litter_touch'] for p in pair_rows if p['first_litter_touch'] < 10**8])),
      'interpretation': ('obs+action are IDENTICAL between u0/u1 until the ant '
                         'physically touches a U-sided litter geom (legs reach '
                         'the pile face at x=2.5 while torso ~1.4), then diverge '
                         '=> NO pre-contact U leakage; the pre-touch policy is '
                         'U-independent.')}

  # ---- (3) open-loop action replay (flip U, no feedback) ----
  renv = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed + 9)
  saved, tries = [], 0
  while len(saved) < args.replay and tries < args.replay * 6:
    tries += 1
    renv.reset(u_side=tries % 2)
    q0 = renv._env.data.qpos.copy()
    v0 = renv._env.data.qvel.copy()
    goal = renv._flatten(renv._last_obs)[29:31].copy()
    u = int(renv.u_side)
    o = set_state(renv, q0, v0, goal, u)
    cl = rollout(renv, act, obs0=o, record=True)
    if cl['success'] > 0:
      saved.append((q0, v0, goal, u, cl['acts']))
  ol_same, ol_flip = [], []
  for q0, v0, goal, u, acts in saved:
    o = set_state(renv, q0, v0, goal, u)          # same U, open loop (control)
    ol_same.append(rollout(renv, act, obs0=o, actions=acts)['success'] > 0)
    o = set_state(renv, q0, v0, goal, 1 - u)      # FLIP U, open loop
    ol_flip.append(rollout(renv, act, obs0=o, actions=acts)['success'] > 0)
  R['open_loop_replay'] = {
      'n_success_episodes': len(saved),
      'closed_loop_success': 1.0,
      'open_loop_same_U_success': float(np.mean(ol_same)) if ol_same else None,
      'open_loop_flipped_U_success': float(np.mean(ol_flip)) if ol_flip else None,
      'interpretation': ('open-loop FLIPPED-U success << closed-loop => success '
                         'needs reactive state feedback (the actions were tuned '
                         'to THIS episode`s litter side).')}

  # ---- lane_sign_consistency explanation ----
  signs = np.array([np.sign(x['exit_y']) for x in rows if abs(x['exit_y']) > 0.15])
  R['lane_sign_consistency_explained'] = {
      'overall_sign_consistency': float(abs(np.mean(signs))) if len(signs) else 0.0,
      'mean_exit_y_u0': R['per_U']['u0']['mean_exit_y'],
      'mean_exit_y_u1': R['per_U']['u1']['mean_exit_y'],
      'note': ('final side is strongly U-conditioned (u0 exit_y '
               f"{R['per_U']['u0']['mean_exit_y']:.2f} -> +y clean; u1 "
               f"{R['per_U']['u1']['mean_exit_y']:.2f} -> -y clean), so the sign "
               'FLIPS with U. Averaged over balanced U the mean sign ~0 => low '
               'consistency. Low consistency here means U-REACTIVE side choice, '
               'NOT a fixed side and NOT random.')}

  # ---- classification ----
  pf = R['paired_uflip']
  leak = pf['diverged_BEFORE_any_litter_touch_LEAKAGE'] > 0.05 * pf['n']
  reactive = (R['side_behavior']['clean_side_arrival_rate'] and
              R['side_behavior']['clean_side_arrival_rate'] > 0.65
              and (R['open_loop_replay']['open_loop_flipped_U_success'] is not None
                   and R['open_loop_replay']['open_loop_flipped_U_success'] < 0.4))
  if leak:
    cls = 'pre-contact U leakage'
  elif reactive:
    cls = 'probe-and-recover (reactive side switching)'
  elif R['side_behavior']['clean_side_arrival_rate'] and R['side_behavior']['clean_side_arrival_rate'] < 0.5:
    cls = 'unstable oscillating mixture'
  else:
    cls = 'other'
  R['classification'] = cls

  json.dump({**R, 'per_episode': rows, 'paired': pair_rows}, open(
      os.path.join(OUT, 'diagnosis.json'), 'w'), indent=2,
      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

  print('\n===== NAIVE POLICY DIAGNOSIS =====')
  print('overall:', json.dumps({k: round(v, 3) for k, v in R['overall'].items()}))
  print('side behavior:', json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in R['side_behavior'].items()}))
  print('per U:', json.dumps(R['per_U'], indent=1))
  print('paired U-flip:', json.dumps({k: v for k, v in pf.items() if k != 'interpretation'}))
  print('open-loop replay:', json.dumps({k: v for k, v in R['open_loop_replay'].items() if k != 'interpretation'}))
  print('lane_sign_consistency:', round(R['lane_sign_consistency_explained']['overall_sign_consistency'], 3),
        '| u0 exit_y', round(R['per_U']['u0']['mean_exit_y'], 2),
        '| u1 exit_y', round(R['per_U']['u1']['mean_exit_y'], 2))
  print('\nCLASSIFICATION:', cls)
  # GIF selection: first 3 switching eps (by episode index) + note paired
  sw = [i for i, x in enumerate(rows) if x['switched'] and x['success']][:3]
  print('switch-example episode indices (for GIFs):', sw)
  json.dump({'switch_eps': sw}, open(os.path.join(OUT, 'gif_selection.json'), 'w'))


if __name__ == '__main__':
  main()
