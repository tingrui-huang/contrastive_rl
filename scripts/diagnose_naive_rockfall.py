"""Behavioral diagnosis of the naive offline CRL rockfall checkpoint.
Analysis only -- no retraining, no env/dataset modification.

Answers, per the benchmark question:
  1. Where does the naive learner sit against the scripted anchors
     (blind-side ~0.6 / center ~0.8 / teacher ~0.88)?
  2. Which ROUTE does it take (left/right/center distribution, per-episode
     zone-mean-y classification)? Did it discover the always-center
     shortcut, or does it stay at blind-side level?
  3. Shortcut scan: does it exploit the trigger definition -- creeping
     through site windows below TRIG_MIN_VX, stalling, or hugging the band
     edge (|y| just under 1.0)?
  4. Paired mask-flip: does behaviour diverge BEFORE the first physical
     rockfall event (leakage), or only after (reactive)?

Saves artifacts/naive_rockfall_diagnosis/diagnosis.json (+ GIF selection).

Usage: python scripts/diagnose_naive_rockfall.py --ckpt <dir>/best.pkl
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
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/naive_rockfall_diagnosis'
SEED = 81_313
N_EVAL = 200
N_PAIRS = 30


def build_policy(ckpt):
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

  return cfg, act, step


def set_state(env, qpos, qvel, goal_xy, mask, severities):
  env.reset(mask=mask, severities=severities)
  d = env._env.data
  d.qpos[:RA.NQ_ANT] = qpos
  d.qvel[:RA.NV_ANT] = qvel
  d.qacc_warmstart[:] = 0.0
  env._goal_vec = np.zeros(29, np.float32)
  env._goal_vec[:2] = goal_xy
  env._goal_state_full = env._goal_vec.copy()
  env._env.goal = np.asarray(goal_xy, float).copy()
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()
  return env._flatten(env._last_obs)


def rollout(env, act, o, horizon=None):
  T = horizon or env.max_episode_steps
  hit, dead_at = 0.0, -1
  ys, xs, vxs = [], [], []
  band_edge = 0          # pre-handoff steps with 0.75 <= |y| < 1.0
  first_hit_t = -1
  #: per-site region bookkeeping (region = the env's trigger geometry):
  #: total in-region steps, low-vx in-region steps (stop-and-go signature),
  #: and the longest run of consecutive marching in-region steps (dwell
  #: pressure actually applied to the trigger).
  reg_steps = [0] * 4
  reg_lowvx = [0] * 4
  reg_run = [0] * 4
  reg_maxrun = [0] * 4
  for t in range(T):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    x, y = float(o[0]), float(o[1])
    vx = float(env._env.data.qvel[0])
    for i, (_, sx, sgn) in enumerate(RA.ROCKFALL_SITES):
      in_reg = (abs(x - sx) <= RA.TRIG_HALF_X
                and RA.TRIG_Y_BAND[0] <= sgn * y <= RA.TRIG_Y_BAND[1])
      if in_reg:
        reg_steps[i] += 1
        if vx < RA.TRIG_MIN_VX:
          reg_lowvx[i] += 1
          reg_run[i] = 0
        else:
          reg_run[i] += 1
          reg_maxrun[i] = max(reg_maxrun[i], reg_run[i])
      else:
        reg_run[i] = 0
    if 1.5 < x < 6.0 and 0.75 <= abs(y) < RA.TRIG_Y_BAND[0]:
      band_edge += 1
    if 2.3 <= x <= 5.7:
      ys.append(y)
      xs.append(x)
      vxs.append(vx)
    if info['rock_ant_contact'] and first_hit_t < 0:
      first_hit_t = t
    if info['dead'] and dead_at < 0:
      dead_at = t
    hit = max(hit, float(r))
    if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
      break
  ys = np.asarray(ys) if ys else np.zeros(1)
  mean_y = float(np.mean(ys))
  entered = [reg_steps[i] > 0 for i in range(4)]
  triggered = list(env._triggered)
  #: per-site avoidance: entered the region but the trigger never fired
  #: (possible only via low-vx or short-dwell behaviour -- the exploit).
  avoided = [entered[i] and not triggered[i] for i in range(4)]
  return {'success': hit, 'dead': dead_at >= 0, 'steps': t + 1,
          'mean_y': mean_y, 'std_y': float(np.std(ys)),
          'route': ('left' if mean_y > 0.5 else
                    'right' if mean_y < -0.5 else 'center'),
          'mean_vx_zone': float(np.mean(vxs)) if vxs else 0.0,
          'reg_steps': reg_steps, 'reg_lowvx': reg_lowvx,
          'reg_maxrun': reg_maxrun, 'entered': entered, 'avoided': avoided,
          'band_edge_steps': band_edge, 'first_hit_t': first_hit_t,
          'mask': list(env.rockfall_mask),
          'triggered': triggered,
          'dropped': list(env._dropped), 'hit': list(env._hit)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default='artifacts/naive_rockfall_crl/best.pkl')
  ap.add_argument('--n', type=int, default=N_EVAL)
  ap.add_argument('--pairs', type=int, default=N_PAIRS)
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)
  cfg, act, step = build_policy(args.ckpt)
  print(f'checkpoint {args.ckpt} @ step {step}', flush=True)

  # ---- 1+2+3: balanced evaluation ----
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED)
  rows = [rollout(env, act, env.reset()) for _ in range(args.n)]
  succ = float(np.mean([r['success'] for r in rows]))
  routes = {k: float(np.mean([r['route'] == k for r in rows]))
            for k in ('left', 'right', 'center')}
  m = np.array([r['mask'] for r in rows])
  # success conditioned on whether the CHOSEN side had active sites
  cond = {'chosen_side_clear': [], 'chosen_side_active': [], 'center': []}
  for r in rows:
    if r['route'] == 'center':
      cond['center'].append(r['success'])
      continue
    idx = (0, 1) if r['route'] == 'left' else (2, 3)
    key = ('chosen_side_active' if (r['mask'][idx[0]] or r['mask'][idx[1]])
           else 'chosen_side_clear')
    cond[key].append(r['success'])
  cond_stats = {k: {'n': len(v), 'success': (round(float(np.mean(v)), 3)
                                             if v else None)}
                for k, v in cond.items()}

  # ---- required headline metrics -------------------------------------------
  # hazard exposure: the trajectory entered >=1 site trigger region
  exposure = [any(r['entered']) for r in rows]
  n_exposed = int(sum(exposure))
  # trigger avoidance: among (episode, site) region entries, the fraction
  # where the trigger never fired -- the stop-and-go / short-dwell exploit
  # surface of the vx>0.1 + dwell-3 trigger definition.
  entries = sum(sum(r['entered']) for r in rows)
  avoids = sum(sum(r['avoided']) for r in rows)
  trigger_avoidance_rate = float(avoids / entries) if entries else None
  # decompose the avoided entries
  avoid_lowvx = avoid_shortdwell = 0
  for r in rows:
    for i in range(4):
      if r['avoided'][i]:
        if r['reg_lowvx'][i] > 0:
          avoid_lowvx += 1
        elif r['reg_maxrun'][i] < RA.TRIG_DWELL:
          avoid_shortdwell += 1
  # impact recovery: success among episodes with >=1 rock-ant hit that
  # did NOT end in absorbing collapse (i.e. non-severe impacts)
  nonfatal_hit = [r for r in rows if any(r['hit']) and not r['dead']]
  hit_any = [r for r in rows if any(r['hit'])]
  # success by mask pattern (grouped: 16 raw cells are too sparse at n=200)
  def mask_class(m):
    la, ra = (m[0] or m[1]), (m[2] or m[3])
    return ('all_clear' if not (la or ra) else
            'left_only' if la and not ra else
            'right_only' if ra and not la else 'both_sides')
  by_pattern = {}
  for r in rows:
    by_pattern.setdefault(mask_class(r['mask']), []).append(r['success'])
  naive_success_by_mask_pattern = {
      k: {'n': len(v), 'success': round(float(np.mean(v)), 3)}
      for k, v in sorted(by_pattern.items())}

  headline = {
      'naive_success': round(succ, 3),
      'naive_center_fraction': round(routes['center'], 3),
      'naive_left_fraction': round(routes['left'], 3),
      'naive_right_fraction': round(routes['right'], 3),
      'naive_hazard_exposure_rate': round(n_exposed / len(rows), 3),
      'naive_drop_rate': round(float(np.mean(
          [any(r['dropped']) for r in rows])), 3),
      'naive_impact_recovery_rate': (round(float(np.mean(
          [r['success'] for r in nonfatal_hit])), 3) if nonfatal_hit
          else None),
      'naive_trigger_avoidance_rate': (round(trigger_avoidance_rate, 3)
                                       if trigger_avoidance_rate is not None
                                       else None),
      'naive_success_by_mask_pattern': naive_success_by_mask_pattern}
  shortcut = {
      'region_entries': int(entries), 'avoided_entries': int(avoids),
      'avoided_via_lowvx_stop_and_go': int(avoid_lowvx),
      'avoided_via_short_dwell_crossing': int(avoid_shortdwell),
      'eps_with_lowvx_in_region': int(sum(
          sum(r['reg_lowvx']) > 0 for r in rows)),
      'mean_lowvx_steps_in_regions': float(np.mean(
          [sum(r['reg_lowvx']) for r in rows])),
      'band_edge_eps_ge10': int(sum(r['band_edge_steps'] >= 10
                                    for r in rows)),
      'mean_vx_zone': float(np.mean([r['mean_vx_zone'] for r in rows])),
      'trigger_rate': float(np.mean([any(r['triggered']) for r in rows])),
      'hit_rate': float(np.mean([any(r['hit']) for r in rows])),
      'success_given_any_hit': (round(float(np.mean(
          [r['success'] for r in hit_any])), 3) if hit_any else None)}
  print('HEADLINE:', json.dumps(headline, indent=2), flush=True)
  print(f'cond {cond_stats}', flush=True)
  print('shortcut scan:', json.dumps(shortcut), flush=True)

  # ---- 4: paired mask-flip ----
  penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=SEED + 1)
  sev = ('mild',) * 4
  pair_rows = []
  for _ in range(args.pairs):
    penv.reset()
    q0 = np.asarray(penv._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(penv._env.data.qvel)[:RA.NV_ANT].copy()
    goal = penv._flatten(penv._last_obs)[29:31].copy()
    tr = {}
    for tag, mask in (('a', (0, 0, 0, 0)), ('b', (1, 1, 1, 1))):
      o = set_state(penv, q0, v0, goal, mask, sev)
      obs_l, drops, anyc = [], [], []
      for t in range(300):
        a = np.asarray(act(jnp.asarray(o[None]))[0])
        obs_l.append(o.copy())
        o, _, _, info = penv.step(a)
        drops.append(any(info['dropped']))
        anyc.append(bool(info['rock_any_contact']))
      tr[tag] = (np.asarray(obs_l), drops, anyc)
    dif = np.abs(tr['a'][0] - tr['b'][0]).max(axis=1)
    div = int(np.argmax(dif > 1e-9)) if (dif > 1e-9).any() else None
    fdrop = tr['b'][1].index(True) if True in tr['b'][1] else None
    fany = tr['b'][2].index(True) if True in tr['b'][2] else None
    pair_rows.append({'div': div, 'first_drop': fdrop,
                      'first_contact': fany,
                      'ok': div is None or (fdrop is not None
                                            and div > fdrop)})
  n_ok = sum(p['ok'] for p in pair_rows)
  print(f'paired mask-flip: {n_ok}/{len(pair_rows)} no pre-drop divergence',
        flush=True)

  # ---- classification ----
  anchors = {'blind_side': 0.60, 'center': 0.82, 'teacher': 0.88}
  dominant = max(routes, key=routes.get)
  if (trigger_avoidance_rate is not None and entries >= 10
      and trigger_avoidance_rate > 0.30):
    verdict = 'TRIGGER-GAMING (stop-and-go / short-dwell through regions)'
  elif routes['center'] >= 0.6:
    verdict = 'found-center-shortcut'
  elif max(routes['left'], routes['right']) >= 0.6:
    verdict = 'blind-side-level (habitual side)'
  else:
    verdict = 'mixed-routes'
  leakage = 'none' if n_ok == len(pair_rows) else 'SUSPECTED'
  report = {
      'checkpoint': args.ckpt, 'step': int(step), 'n_eval': args.n,
      **headline,
      'route_distribution': routes,
      'dominant_route': dominant,
      'success_conditioned': cond_stats,
      'shortcut_scan': shortcut,
      'paired_maskflip': {'pairs': pair_rows, 'n_ok': n_ok,
                          'leakage': leakage},
      'anchors': anchors,
      'verdict': verdict,
      'gap_to_center_anchor': round(anchors['center'] - succ, 3),
      'gap_to_teacher_anchor': round(anchors['teacher'] - succ, 3)}
  json.dump(report, open(os.path.join(OUT, 'diagnosis.json'), 'w'),
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  # deterministic GIF selection: first 2 successes + first 2 fails per route
  sel = {}
  for kind in ('succ', 'fail'):
    for rt in ('left', 'right', 'center'):
      ids = [i for i, r in enumerate(rows)
             if r['route'] == rt and (r['success'] > 0) == (kind == 'succ')]
      sel[f'{kind}_{rt}'] = ids[:2]
  json.dump(sel, open(os.path.join(OUT, 'gif_selection.json'), 'w'))
  print('verdict:', verdict, '| success', succ, '->', OUT, flush=True)


if __name__ == '__main__':
  main()
