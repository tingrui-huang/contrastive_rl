"""Stage-1 GEOMETRY GATE for offline_ant_umaze_litter.

Drives the validated 1M best checkpoint (0.89 under the d4rl protocol) as a
waypoint-following low-level controller and measures, per lane, the ONLY
metric that counts: Pr(reach the true episode goal within 700 steps).

Arms (litter env unless noted):
  nolitter     waypoints through the corridor middle, NO litter env --
               controller baseline (does waypoint-following itself cost?)
  clean        waypoints through the clean lane (uses privileged_u), full speed
  middle_fast  waypoints through the middle rubble strip, full speed
  middle_slow  same, low speed (duty-cycled actions)
  pile         waypoints into the pile side (wrong side), full speed
  direct       no waypoints: true goal from step 0 in the litter env (what the
               plain policy does when litter appears; diagnostic only)

Gate targets (plan):
  clean       ~ nolitter  (and both ~ the 0.89-checkpoint level)
  middle_fast << clean    (>= 30-point drop)
  middle_slow >> middle_fast (clear recovery)
  pile        ~ 0

Aux per arm: min_dist to true goal, fall fraction (torso upside-down or
collapsed), zone transit steps, pile/rubble episode contacts, per-U split.

Usage:
  python scripts/litter_geometry_gate.py \
      --ckpt offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl \
      --eps 50 [--probe] [--arms clean pile ...]

--probe compares slow-mode candidates (duty patterns / action scaling) on the
middle lane of the NO-litter env first; use it to pick --slow before trusting
middle_slow.
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
from crl.d4rl_ant import LITTER_ZONE_X    # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

ENV_SEED = 424242
CLEAN_LANE_ABS_Y = 1.05    # center of the clean half-corridor [-2, 0];
                           # swept: 1.1 @ LOOKAHEAD 3.0 matches pure policy
PILE_LANE_ABS_Y = 1.35     # center of the pile band [0.7, 2.0]
HANDOFF_X = 6.0            # past the zone -> command the true goal
LOOKAHEAD = 3.0            # swept over {1.8, 2.5, 3.0}: larger = smoother
                           # lateral tracking, fewer falls
ARMS = ('nolitter', 'clean', 'middle_fast', 'middle_slow', 'pile', 'direct')


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


def lane_y(arm, u_side, ep=0):
  clean_sign = -1.0 if u_side == 1 else 1.0
  if arm == 'clean':
    return clean_sign * CLEAN_LANE_ABS_Y
  if arm == 'pile':
    return -clean_sign * PILE_LANE_ABS_Y
  if arm == 'nolitter':                   # side-lane baseline, alternate sides
    return (1.0 if ep % 2 == 0 else -1.0) * CLEAN_LANE_ABS_Y
  return 0.0                              # middle_* -> middle


def slow_gate(spec, t):
  """spec: '' (full speed) | 'dutyK_M' (K policy steps, M zero steps) |
  'scaleF' (scale actions by F). Returns (use_policy, scale)."""
  if not spec:
    return True, 1.0
  if spec.startswith('duty'):
    k, m = (int(v) for v in spec[4:].split('_'))
    return (t % (k + m)) < k, 1.0
  if spec.startswith('scale'):
    return True, float(spec[5:])
  raise ValueError(spec)


def run_episode(env, act, arm, slow_spec, ep=0):
  o = env.reset()
  true_goal = o[29:31].copy()
  u_side = getattr(env, 'u_side', None)
  y = lane_y(arm, u_side if u_side is not None else 0, ep)
  goal_vec = np.zeros(29, np.float32)

  hit, dmin = 0.0, float(np.linalg.norm(o[:2] - true_goal))
  falls = zone_steps = 0
  slow_in_zone_only = arm == 'middle_slow'
  handoff = arm == 'direct'                # once True, true goal forever
  for t in range(env.max_episode_steps):
    xy = o[:2]
    in_zone_x = xy[0] > 1.5 and xy[0] < HANDOFF_X
    if not handoff and (xy[0] >= HANDOFF_X or xy[1] >= 2.0):
      handoff = True                       # left the bottom corridor: LATCH
    if handoff:
      goal_vec[:2] = true_goal
    else:
      # continuous carrot: lane point LOOKAHEAD ahead (no discrete switches;
      # the policy tracks far-ish goals far more smoothly than close ones).
      # Exit ramp: past the zone, taper the lane back to the centerline so
      # the corner turn is not taken with a full lateral offset at speed.
      # pile arm: carrot capped at the pile center so it keeps pushing in.
      ramp = min(1.0, max(0.0, (HANDOFF_X + 0.3 - xy[0]) / 0.8))
      cap = 4.0 if arm == 'pile' else HANDOFF_X + 1.0
      goal_vec[:2] = (min(xy[0] + LOOKAHEAD, cap), y * ramp)
    o_cmd = o.copy()
    o_cmd[29:] = goal_vec
    use_policy, scale = slow_gate(
        '' if (slow_in_zone_only and not in_zone_x) else slow_spec, t)
    a = (np.asarray(act(jnp.asarray(o_cmd[None]))[0]) * scale
         if use_policy else np.zeros(8, np.float32))
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    dmin = min(dmin, float(np.linalg.norm(o[:2] - true_goal)))
    q = env._env.data.qpos
    if torso_up_z(np.asarray(q)) < 0.0 or float(q[2]) < 0.2:
      falls += 1
    if LITTER_ZONE_X[0] <= o[0] <= LITTER_ZONE_X[1]:
      zone_steps += 1
    if hit > 0:
      break
  contacts = dict(getattr(env, 'episode_contacts', {'pile': 0, 'rubble': 0}))
  return {'success': hit, 'min_dist': dmin, 'u_side': u_side,
          'falls': falls, 'zone_steps': zone_steps, 'steps': t + 1,
          'dead': bool(getattr(env, 'dead', False)),
          'max_force': float(getattr(env, 'episode_max_force', 0.0)),
          **{f'{k}_contacts': v for k, v in contacts.items()}}


def run_arm(cfg, act, arm, eps, slow_spec):
  env_name = ('offline_ant_umaze' if arm == 'nolitter'
              else 'offline_ant_umaze_litter')
  env = envs_mod.make_env(env_name, cfg, seed=ENV_SEED)
  rows = [run_episode(env, act, arm, slow_spec if arm == 'middle_slow' else '',
                      ep) for ep in range(eps)]
  out = {'arm': arm, 'episodes': eps,
         'success': float(np.mean([r['success'] for r in rows])),
         'min_dist_median': float(np.median([r['min_dist'] for r in rows])),
         'fall_ep_frac': float(np.mean([r['falls'] > 0 for r in rows])),
         'mean_zone_steps': float(np.mean([r['zone_steps'] for r in rows])),
         'mean_pile_contacts': float(np.mean([r.get('pile_contacts', 0)
                                              for r in rows])),
         'mean_rubble_contacts': float(np.mean([r.get('rubble_contacts', 0)
                                                for r in rows])),
         'dead_frac': float(np.mean([r['dead'] for r in rows])),
         'max_force_p50': float(np.median([r['max_force'] for r in rows])),
         'max_force_p90': float(np.percentile([r['max_force'] for r in rows],
                                              90))}
  for u in (0, 1):
    sel = [r['success'] for r in rows if r['u_side'] == u]
    out[f'success_u{u}'] = float(np.mean(sel)) if sel else None
  return out, rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--eps', type=int, default=50)
  ap.add_argument('--arms', nargs='+', default=list(ARMS))
  ap.add_argument('--slow', default='duty1_2')
  ap.add_argument('--probe', action='store_true',
                  help='compare slow-mode candidates on nolitter middle lane')
  ap.add_argument('--out', default='artifacts/litter_env/geometry_gate.json')
  args = ap.parse_args()

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=ENV_SEED)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(args.ckpt)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  if args.probe:
    print(f'ckpt step {step}: slow-mode probe, nolitter middle lane, '
          f'{args.eps} eps each')
    results = []
    for spec in ('', 'duty1_1', 'duty1_2', 'duty1_3', 'scale0.5', 'scale0.3'):
      env = envs_mod.make_env('offline_ant_umaze', cfg, seed=ENV_SEED)
      rows = [run_episode(env, act, 'middle_slow', spec, None)
              for _ in range(args.eps)]
      r = {'slow': spec or 'full',
           'success': float(np.mean([x['success'] for x in rows])),
           'mean_zone_steps': float(np.mean([x['zone_steps'] for x in rows])),
           'fall_ep_frac': float(np.mean([x['falls'] > 0 for x in rows])),
           'mean_steps': float(np.mean([x['steps'] for x in rows]))}
      results.append(r)
      print(f'  {r["slow"]:8s} success {r["success"]:.2f}  '
            f'zone_steps {r["mean_zone_steps"]:6.1f}  '
            f'falls {r["fall_ep_frac"]:.2f}  steps {r["mean_steps"]:6.1f}')
    out = args.out.replace('.json', '_probe.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'ckpt': args.ckpt, 'step': int(step), 'eps': args.eps,
               'probe': results}, open(out, 'w'), indent=2)
    print('saved', out)
    return

  print(f'ckpt step {step}, {args.eps} eps/arm, slow={args.slow}')
  summary, detail = [], {}
  for arm in args.arms:
    s, rows = run_arm(cfg, act, arm, args.eps, args.slow)
    summary.append(s)
    detail[arm] = rows
    print(f'{arm:12s} success {s["success"]:.2f} '
          f'(u0 {s["success_u0"]}, u1 {s["success_u1"]})  '
          f'min_d {s["min_dist_median"]:5.2f}  falls {s["fall_ep_frac"]:.2f}  '
          f'zone {s["mean_zone_steps"]:6.1f}  '
          f'pile_c {s["mean_pile_contacts"]:6.1f}  '
          f'rub_c {s["mean_rubble_contacts"]:6.1f}')

  by = {s['arm']: s['success'] for s in summary}
  gates = {}
  if 'clean' in by and 'nolitter' in by:
    gates['G1_clean_near_baseline'] = by['clean'] >= by['nolitter'] - 0.10
  if 'middle_fast' in by and 'clean' in by:
    gates['G2_middle_fast_drops'] = by['middle_fast'] <= by['clean'] - 0.30
  if 'middle_slow' in by and 'middle_fast' in by:
    gates['G3_middle_slow_recovers'] = (by['middle_slow']
                                        >= by['middle_fast'] + 0.20)
  if 'pile' in by:
    gates['G4_pile_fails'] = by['pile'] <= 0.15
  result = {'ckpt': args.ckpt, 'step': int(step), 'eps': args.eps,
            'slow': args.slow, 'summary': summary, 'gates': gates,
            'all_pass': bool(gates) and all(gates.values())}
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  json.dump(result, open(args.out, 'w'), indent=2)
  json.dump(detail, open(args.out.replace('.json', '_detail.json'), 'w'),
            indent=2)
  for k, v in gates.items():
    print(f'{"PASS" if v else "FAIL"}  {k}')
  print(('GEOMETRY GATE PASS' if result['all_pass']
         else 'GEOMETRY GATE FAILED') + f' -> {args.out}')


if __name__ == '__main__':
  main()
