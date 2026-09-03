"""Sighted teacher for the two-route AntMaze rockfall benchmark + its audit.

Route policy (privileged: reads the latent the learner never sees):

    rockfall_active == False  ->  SHORTCUT (fast, through the hazard band)
    rockfall_active == True   ->  DETOUR   (long, always safe)

Execution is real Ant locomotion by the repo's frozen controllers -- no
teleportation. BOTH routes start from the SINGLE canonical pose (the native
d4rl east pose); the route is the choice of DRIVER, never an initial
condition. The env has no heading option:

  * shortcut: the walker drives +y through a static 90-degree world-frame
    observation remap (rot_north), tracking x ~ 0, then x -> goal_x for the
    final approach. The walker's training reward penalises body yaw away
    from the +x axis OF THE FRAME IT IS SHOWN, and under rot_north that axis
    is world north -- so from the east pose it sees a -90 deg yaw error and
    turns the ant itself over ~25-40 steps. Measured from the east pose:
    0.948 goal rate (109/115, three disjoint seed blocks), ~88 steps, vs
    1.000 / ~78 steps when the ant was pre-yawed north.
  * detour: walker +x along the bottom corridor, latched handoff to the
    goal-conditioned base policy at x >= 6. Unchanged; 0.95 from the east
    pose. Do NOT route the shortcut through the base policy: its eastward
    habit is a property of its goal representation (it was frozen on the
    original U-maze, where the shortcut cell is a WALL), so north waypoints
    score 0.733 with physical freezes.

Collection/audit protocol: the episode's latent is drawn by the CALLER's rng
(Bernoulli p_active, recorded seed) and passed to reset(). The joint
(latent, data) distribution is identical to letting the env draw; the env's
own latent stream is still consumed in fixed order. The learner is evaluated
from the same canonical pose, so nothing in its initial observation carries
route or latent information.

Run the audit:  python scripts/tworoute_teacher.py [--n 300]
Writes artifacts/tworoute_rockfall_v0/teacher_audit.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
P_ACTIVE = 0.30
HORIZON = 400          #: teacher episodes finish in <= ~250 steps; 400 caps

#: Rz(-90) (world +y -> virtual +x) for the north remap.
_QM = np.array([np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)])


def _qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot_north(s29):
  """State in the virtual frame whose +x is world +y (ang vel is body-frame
  and joints are internal: only xy, quat and linear velocity rotate)."""
  v = s29.copy()
  x, y = s29[0], s29[1]
  v[0], v[1] = y, -x
  v[3:7] = _qmul(_QM, s29[3:7])
  vx, vy = s29[15], s29[16]
  v[15], v[16] = vy, -vx
  return v


class TwoRouteTeacher:
  """Frozen-controller route executor. fresh() before every episode."""

  def __init__(self, walker, base_act):
    self._walker = walker
    self._base = base_act
    self._handoff = False

  def fresh(self):
    self._handoff = False

  def act_shortcut(self, o58):
    y = float(o58[1])
    gx = float(o58[29])
    track_x = 0.0 if y < 6.0 else gx      # final approach onto the goal x
    s = rot_north(o58[:29])
    o = np.concatenate([s, o58[29:]]).astype(np.float32)
    return self._walker(o, -track_x, RP.V_SIDE)

  def act_detour(self, o58):
    x, y = float(o58[0]), float(o58[1])
    if not self._handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      self._handoff = True
    if self._handoff:
      oc = o58.copy()
      oc[29:] = 0.0
      oc[29:31] = o58[29:31]
      return np.asarray(self._base(jnp.asarray(oc[None]))[0])
    return self._walker(o58, 0.0, RP.V_SIDE)

  def act(self, o58, route):
    return self.act_shortcut(o58) if route == 'shortcut' \
        else self.act_detour(o58)


def make_teacher():
  """(cfg, env-builder-ready teacher). Loads the frozen controllers once."""
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  return cfg, TwoRouteTeacher(walker, base_act)


def teacher_episode(env, teacher, u, horizon=HORIZON, on_step=None):
  """One teacher episode under latent u. Returns the episode record."""
  route = 'detour' if u else 'shortcut'
  o = env.reset(rockfall_active=bool(u))
  teacher.fresh()
  ret, t = 0.0, 0
  info = {}
  for t in range(horizon):
    a = teacher.act(o, route)
    o2, r, done, info = env.step(a)
    if on_step is not None:
      on_step(o, a, o2, r, done, info)
    o = o2
    ret += float(r)
    if done or r > 0:
      break
  return {'rockfall_active': bool(u), 'route_intent': route,
          'route_realized': info.get('route'),
          'success': bool(info.get('success')),
          'failure': bool(info.get('failure')),
          'entered_hazard': bool(info.get('entered_hazard')),
          'rock_dropped': bool(info.get('rock_dropped')),
          'steps': int(t + 1), 'return': ret}


def audit(n=300, seed=101, p_active=P_ACTIVE):
  cfg, teacher = make_teacher()
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=seed)
  u_rng = np.random.default_rng(seed + 5000)
  rows = []
  for k in range(n):
    u = bool(u_rng.random() < p_active)
    rows.append(teacher_episode(env, teacher, u))
    if (k + 1) % 50 == 0:
      print(f'  {k + 1}/{n} episodes', flush=True)
  return rows


def summarize(rows):
  n = len(rows)
  clear = [r for r in rows if not r['rockfall_active']]
  active = [r for r in rows if r['rockfall_active']]
  sc = [r for r in rows if r['route_intent'] == 'shortcut']
  dt = [r for r in rows if r['route_intent'] == 'detour']

  def rate(xs, key='success'):
    return round(float(np.mean([x[key] for x in xs])), 4) if xs else None

  return {
      'n': n, 'n_clear': len(clear), 'n_active': len(active),
      'success': rate(rows),
      'failure': rate(rows, 'failure'),
      'timeout': round(float(np.mean(
          [not r['success'] and not r['failure'] for r in rows])), 4),
      'P_shortcut_given_clear': (round(float(np.mean(
          [r['route_intent'] == 'shortcut' for r in clear])), 4)
          if clear else None),
      'P_detour_given_active': (round(float(np.mean(
          [r['route_intent'] == 'detour' for r in active])), 4)
          if active else None),
      'route_choice_accuracy': round(float(np.mean(
          [(r['route_intent'] == 'detour') == r['rockfall_active']
           for r in rows])), 4),
      'success_by_route': {'shortcut': rate(sc), 'detour': rate(dt)},
      'failure_by_route': {'shortcut': rate(sc, 'failure'),
                           'detour': rate(dt, 'failure')},
      'mean_steps_by_route': {
          'shortcut': (round(float(np.mean([r['steps'] for r in sc])), 1)
                       if sc else None),
          'detour': (round(float(np.mean([r['steps'] for r in dt])), 1)
                     if dt else None)},
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=101)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  rows = audit(args.n, args.seed, args.p_active)
  s = summarize(rows)
  print(json.dumps(s, indent=2), flush=True)
  ok = (s['success'] is not None and s['success'] > 0.90
        and s['P_shortcut_given_clear'] > 0.90
        and s['P_detour_given_active'] > 0.90)
  print('TARGETS ' + ('MET' if ok else 'NOT MET')
        + ' (success>0.90, P(sc|clear)>0.90, P(dt|active)>0.90)', flush=True)
  with open(os.path.join(args.out_dir, 'teacher_audit.json'), 'w') as f:
    json.dump({'summary': s, 'targets_met': bool(ok),
               'seed': args.seed, 'p_active': args.p_active,
               'episodes': rows}, f, indent=2)
  print('-> teacher_audit.json', flush=True)


if __name__ == '__main__':
  main()
