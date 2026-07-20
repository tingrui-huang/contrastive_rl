"""Stage 1B/1C: contact calibration + the pre-registered 4-arm gate,
driven by the FROZEN walker (corridor) + frozen 0.89 base (rest of route).

Arms (command only; same controller):
  clean_fast    y_ref = clean side * LANE, v = fast
  pile_fast     y_ref = litter side * LANE, v = fast
  middle_fast   y_ref = 0, v = fast
  middle_slow   y_ref = 0, v = slow
  nolitter      plain env, alternating side lanes, fast (baseline)

Full-route episode: walker drives until the handoff latch (x >= 6 or
y >= 2), then the 0.89 base takes the true episode goal. Success =
env sparse reward within 700 steps (the raw task metric).

Modes:
  --calibrate   collapse OFF; dumps per-step contact records (horizontal
                normal force, horizontal impulse, precontact planar speed)
                per arm for threshold selection. U forced 50/50.
  --collapse F  run the gate with collapse_force = F (horizontal normal
                force trigger, crl/d4rl_ant.py). Use a DIFFERENT --seed
                than the calibration run (holdout discipline).

Usage:
  python scripts/walker_gate.py --calibrate --eps 50
  python scripts/walker_gate.py --collapse 120 --eps 50 --seed 999
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

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import probe                     # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

LANE = 1.1
HANDOFF_X = 6.0
ARMS = ('nolitter', 'clean_fast', 'pile_fast', 'middle_fast', 'middle_slow')


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


SLOW_V = probe.V_SLOW      # middle_slow command speed (--slow-v overrides:
                           # 0.6 wades at ~0.17 u/s effective and TIMES OUT
                           # the 700-step budget; 0.8 stays well under the
                           # 1.2 collapse speed gate but finishes)


def arm_command(arm, u_side, ep):
  """(y_ref, v_ref) for the arm given the episode's U."""
  clean = -1.0 if u_side == 1 else 1.0
  if arm == 'clean_fast':
    return clean * LANE, probe.V_FAST
  if arm == 'pile_fast':
    return -clean * LANE, probe.V_FAST
  if arm == 'middle_fast':
    return 0.0, probe.V_FAST
  if arm == 'middle_slow':
    return 0.0, SLOW_V
  return (LANE if ep % 2 == 0 else -LANE), probe.V_FAST   # nolitter


#: Blind-safe unstick heuristic (middle_slow ONLY -- it is part of the
#: cautious policy's DEFINITION, not given to the naive arms): a blind slow
#: wader that stalls probes sideways. Without it the deterministic walker
#: wedges against the same rubble blockage forever (holdout3: 20/23
#: middle_slow "timeouts" never left the corridor).
STALL_WINDOW = 40
STALL_MIN_DX = 0.25
NUDGE_Y = 0.5
NUDGE_STEPS = 25
NUDGE_V = None             # None: keep the arm's own v while nudging.
                           # (gentler nudge variants measured WORSE within
                           # noise -- do not re-tune these on gate seeds)


def run_episode(env, walker, base_act, arm, ep, u_side=None, record=None):
  o = (env.reset(u_side=u_side) if u_side is not None
       and hasattr(env, 'u_side') else env.reset())
  u = getattr(env, 'u_side', None)
  y_ref, v_ref = arm_command(arm, u if u is not None else 0, ep)
  true_goal = o[29:31].copy()
  handoff = False
  hit, dmin = 0.0, float(np.linalg.norm(o[:2] - true_goal))
  falls = corridor_steps = 0
  dead_at = None
  x_hist, nudge_until, nudge_sign = [], -1, 1.0
  for t in range(env.max_episode_steps):
    xy = o[:2]
    if not handoff and (xy[0] >= HANDOFF_X or xy[1] >= 2.0):
      handoff = True
    if handoff:
      o_cmd = o.copy()
      o_cmd[29:] = 0.0
      o_cmd[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(o_cmd[None]))[0])
    else:
      y_cmd, v_cmd = y_ref, v_ref
      if arm == 'middle_slow':
        x_hist.append(float(xy[0]))
        nv = v_ref if NUDGE_V is None else NUDGE_V
        if t < nudge_until:
          y_cmd, v_cmd = nudge_sign * NUDGE_Y, nv
        elif (len(x_hist) > STALL_WINDOW
              and x_hist[-1] - x_hist[-STALL_WINDOW] < STALL_MIN_DX):
          nudge_until = t + NUDGE_STEPS
          nudge_sign = -nudge_sign
          x_hist.clear()
          y_cmd, v_cmd = nudge_sign * NUDGE_Y, nv
      a = walker(o, y_cmd, v_cmd)
      corridor_steps += 1
    speed_before = float(np.hypot(env._env.data.qvel[0],
                                  env._env.data.qvel[1]))
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    dmin = min(dmin, float(np.linalg.norm(o[:2] - true_goal)))
    if record is not None and info.get('max_litter_normal_force', 0) > 5.0:
      record.append({'arm': arm, 'u': u, 'ep': ep, 't': t,
                     'hforce': info['max_horizontal_normal_force'],
                     'himpulse': info['max_horizontal_impulse'],
                     'force': info['max_litter_normal_force'],
                     'nz': info['contact_normal_z'],
                     'pre_speed': speed_before,
                     'handoff': handoff})
    if info.get('dead') and dead_at is None:
      dead_at = t
    q = env._env.data.qpos
    if torso_up_z(np.asarray(q)) < 0.0 or float(q[2]) < 0.2:
      falls += 1
    if hit > 0:
      break
    if dead_at is not None and t > dead_at + 5:
      break                              # absorbing: fast-forward the freeze
  return {'success': hit, 'min_dist': dmin, 'u_side': u,
          'dead': dead_at is not None, 'fell': falls > 0, 'steps': t + 1,
          'corridor_steps': corridor_steps,
          'max_force': float(getattr(env, 'episode_max_force', 0.0)),
          'max_hforce': float(getattr(env, 'episode_max_hforce', 0.0)),
          'max_himpulse': float(getattr(env, 'episode_max_himpulse', 0.0))}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--walker', default='artifacts/walker/phase1/'
                                      'walker_best.pkl')
  ap.add_argument('--ckpt', default='offline_umaze_bc005_twinmin_s0_50k/'
                                    'checkpoints/best.pkl')
  ap.add_argument('--eps', type=int, default=50)
  ap.add_argument('--seed', type=int, default=311)
  ap.add_argument('--calibrate', action='store_true')
  ap.add_argument('--collapse', type=float, default=None,
                  help='collapse_force (horizontal N) for the gate run')
  ap.add_argument('--collapse-speed', type=float, default=None,
                  help='precontact planar speed gate for collapse')
  ap.add_argument('--slow-v', type=float, default=None,
                  help='override middle_slow command speed')
  ap.add_argument('--arms', nargs='+', default=list(ARMS))
  ap.add_argument('--out', default=None)
  args = ap.parse_args()
  mode = 'calibrate' if args.calibrate else 'gate'
  out = args.out or f'artifacts/litter_env/walker_{mode}.json'
  if args.slow_v is not None:
    global SLOW_V
    SLOW_V = args.slow_v

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

  wparams, wmeta = probe.load_residual(args.walker)
  walker = probe.WalkerController(wparams)
  print(f'walker: {args.walker} (step {wmeta.get("step")})  mode={mode}  '
        f'collapse={args.collapse}')

  records = [] if args.calibrate else None
  summary, detail = [], {}
  for arm in args.arms:
    env_name = ('offline_ant_umaze' if arm == 'nolitter'
                else 'offline_ant_umaze_litter')
    env = envs_mod.make_env(env_name, cfg, seed=args.seed)
    if hasattr(env, 'collapse_force'):
      env.collapse_force = args.collapse
      env.collapse_speed = args.collapse_speed
    rows = [run_episode(env, walker, base_act, arm, ep,
                        u_side=(ep % 2 if arm != 'nolitter' else None),
                        record=records)
            for ep in range(args.eps)]
    s = {'arm': arm, 'eps': args.eps,
         'success': float(np.mean([r['success'] for r in rows])),
         'dead_frac': float(np.mean([r['dead'] for r in rows])),
         'fall_frac': float(np.mean([r['fell'] for r in rows])),
         'min_dist_median': float(np.median([r['min_dist'] for r in rows])),
         'mean_corridor_steps': float(np.mean([r['corridor_steps']
                                               for r in rows]))}
    for u in (0, 1):
      sel = [r['success'] for r in rows if r['u_side'] == u]
      s[f'success_u{u}'] = float(np.mean(sel)) if sel else None
    summary.append(s)
    detail[arm] = rows
    print(f'{arm:12s} success {s["success"]:.2f} '
          f'(u0 {s["success_u0"]}, u1 {s["success_u1"]})  '
          f'dead {s["dead_frac"]:.2f}  fall {s["fall_frac"]:.2f}  '
          f'corridor {s["mean_corridor_steps"]:5.1f}')

  result = {'mode': mode, 'walker': args.walker, 'collapse': args.collapse,
            'seed': args.seed, 'eps': args.eps, 'summary': summary}
  if not args.calibrate:
    by = {s['arm']: s['success'] for s in summary}
    gates = {}
    result_extra = {}
    if 'clean_fast' in by and 'nolitter' in by:
      gates['G1_clean_near_baseline'] = by['clean_fast'] >= by['nolitter'] - 0.10
    if 'middle_fast' in by and 'clean_fast' in by:
      gates['G2_middle_fast_drops'] = by['middle_fast'] <= by['clean_fast'] - 0.30
    if 'middle_slow' in by and 'middle_fast' in by:
      # FROZEN encoding (2026-07-20 user decision): clear recovery = +0.15.
      # The original +0.20 draft is reported alongside, non-gating.
      gates['G3_middle_slow_recovers'] = (by['middle_slow']
                                          >= by['middle_fast'] + 0.15)
      result_extra['G3_strict_plus020'] = (by['middle_slow']
                                           >= by['middle_fast'] + 0.20)
    if 'pile_fast' in by:
      gates['G4_pile_fails'] = by['pile_fast'] <= 0.15
    result['gates'] = gates
    result.update(result_extra)
    result['all_pass'] = bool(gates) and all(gates.values())
    for k, v in gates.items():
      print(f'{"PASS" if v else "FAIL"}  {k}')
    print('GATE ' + ('PASS' if result['all_pass'] else 'FAILED'))
  os.makedirs(os.path.dirname(out), exist_ok=True)
  json.dump(result, open(out, 'w'), indent=2)
  json.dump(detail, open(out.replace('.json', '_detail.json'), 'w'),
            indent=2)
  if records is not None:
    json.dump(records, open(out.replace('.json', '_contacts.json'), 'w'))
    print(f'{len(records)} contact records saved')
  print('saved', out)


if __name__ == '__main__':
  main()
