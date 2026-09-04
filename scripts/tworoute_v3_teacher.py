"""Sighted teacher for the V3 two-route rockfall pair + its qualification audit.

Route policy (privileged: reads the latent the learner never sees):

    rockfall_active == False  ->  SHORTCUT (through the EAST-leg hazard band)
    rockfall_active == True   ->  DETOUR   (long, always safe)

Execution is a WALKER-ONLY RELAY. The goal-conditioned base policy of the V2
teacher is RETIRED here: it is anchored to the original U-maze goal (0, 8)
by its frozen goal representation and cannot be steered to the V3 corners.
Every leg is the frozen corridor walker driven through a rotated + translated
view of the world:

  * Frames. Each frame is the world rotated about z so that the walker's
    +x-seeking becomes a compass direction; shown a rotated frame, the walker
    turns the ant itself (measured 0.948-1.000 across frames). rot_north is
    imported from scripts/tworoute_teacher.py; rot_south is its mirror.
  * Lane translation. The walker was trained for lane commands within
    ~+-1.2; a raw y_ref of 8.0 stalls it. So the frame is TRANSLATED so the
    target lane sits at 0 and the command is always y_ref = 0:
      native frame, target world y = Y:  s[1] -= Y
      north  frame, target world x = X:  s[1] += X   (virtual y = -world x)
      south  frame, target world x = X:  s[1] -= X   (virtual y = +world x)
  * Frame-switch threshold 7.5, not 6.0, for corner switches: at 6.3 the
    lateral error to the next lane (8) is 1.7 and the ant stalls (measured
    0/25); at 7.5 it is 0.5 and the relay works (25/25). Run-in switches to
    the GOAL lane keep their own measured thresholds (6.0 / 2.0 below) --
    there the next lane is at most ~1.5 away, inside the walker's range.

Drivers (latched leg advance; measured success / mean steps over 25 eps):

  tr shortcut  1.000 / 156.0 : native lane 0 until x>=7.5; north lane 8
                               until y>=6; north lane goal_x
  tr detour    0.960 / 163.8 : north lane 0 until y>=7.5; native lane 8
                               until x>=6; native lane goal_y
  br shortcut  1.000 /  77.3 : native lane 0 until x>=6; native lane goal_y
  br detour    0.960 / 224.8 : north lane 0 until y>=7.5; native lane 8
                               until x>=7.5; south lane 8 until y<=2;
                               south lane goal_x

Reference numbers (pre-registered, identical protocol): sparse, BOTH
variants: always-shortcut 0.70, always-detour 0.96, oracle 0.988.
Discounted 0.99**steps on success: tr shortcut 0.146, detour 0.185 (best
blind), oracle 0.201; br shortcut 0.323 (best blind), detour 0.100, oracle
0.353. The pair is a controlled comparison -- identical sparse refs, only
the incentive the discounted objective assigns to the shortcut differs.

Audit gates are computed from route_REALIZED (info['route']; band entry is
authoritative for 'shortcut', y >= 6 with x < 2 for 'detour'), NOT from
route intent: V2's intent-based gate was vacuous by construction (intent is
a deterministic function of the latent, so P(intent|latent) == 1 always) and
that defect is not ported. Latents are drawn by the CALLER's rng and passed
to reset(), exactly as in V2, so the env's own latent stream stays in fixed
order.

Run the audit:  python scripts/tworoute_v3_teacher.py [--variant tr|br|both]
                       [--n 300] [--seed 101]
Writes artifacts/tworoute_rockfall_v3/<variant>/teacher_audit.json.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from tworoute_teacher import rot_north, _qmul  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v3'
P_ACTIVE = 0.30
HORIZON = 400          #: longest driver (br detour) finishes in ~225 steps
GAMMA = 0.99           #: discount of the pre-registered reference numbers

#: Rz(+90) (world -y -> virtual +x) for the south remap.
_QP = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])


def rot_south(s29):
  """State in the virtual frame whose +x is world -y (mirror of rot_north;
  ang vel is body-frame and joints are internal: only xy, quat and linear
  velocity rotate)."""
  v = s29.copy()
  x, y = s29[0], s29[1]
  v[0], v[1] = -y, x
  v[3:7] = _qmul(_QP, s29[3:7])
  vx, vy = s29[15], s29[16]
  v[15], v[16] = -vy, vx
  return v


def env_name(variant):
  """Registered env id for a variant ('tr' goal (8,8) / 'br' goal (8,0))."""
  if variant not in ('tr', 'br'):
    raise ValueError(f"variant must be 'tr' or 'br', got {variant!r}")
  return f'offline_ant_umaze_tworoute_rockfall_v3{variant}'


#: driver tables: one (frame, lane, advance) triple per leg. lane 'gx'/'gy'
#: means the goal coordinate read from o58 at act time; advance = (coord,
#: sign, threshold) latches the NEXT leg when sign*coord >= sign*threshold
#: (None = terminal leg). Corner switches sit at 7.5 (see module doc);
#: run-in switches to the goal lane keep their measured 6.0 / 2.0.
_DRIVERS = {
    ('tr', 'shortcut'): (('native', 0.0, ('x', 1, 7.5)),
                         ('north', 8.0, ('y', 1, 6.0)),
                         ('north', 'gx', None)),
    ('tr', 'detour'): (('north', 0.0, ('y', 1, 7.5)),
                       ('native', 8.0, ('x', 1, 6.0)),
                       ('native', 'gy', None)),
    ('br', 'shortcut'): (('native', 0.0, ('x', 1, 6.0)),
                         ('native', 'gy', None)),
    ('br', 'detour'): (('north', 0.0, ('y', 1, 7.5)),
                       ('native', 8.0, ('x', 1, 7.5)),
                       ('south', 8.0, ('y', -1, 2.0)),
                       ('south', 'gx', None)),
}


class TwoRouteV3Teacher:
  """Walker-only relay route executor. fresh() before every episode.

  STALL-UNSTICK (ported from rockfall_pilot, which applies it on every
  deployed route): the walker occasionally enters a limit cycle during the
  start-cell 90-deg turn (~5-10% of north-opening legs; the n=300 tr audit
  measured P(detour|active) = 71/79 = 0.899 against the > 0.90 gate, all 8
  misses shuffling within ~2 units of the start for the full horizon). If
  total displacement over the last STALL_WINDOW steps falls below
  STALL_MIN_DISP, the lane reference is offset laterally by NUDGE_LANE for
  NUDGE_STEPS steps -- alternating sign per engagement -- which breaks the
  cycle the same way RP's NUDGE_Y does. Deterministic: no rng anywhere."""

  STALL_WINDOW = 40          #: steps of history examined (RP.STALL_WINDOW)
  STALL_MIN_DISP = 0.25      #: min displacement over the window (RP)
  NUDGE_LANE = 0.35          #: temporary lateral lane offset (RP.NUDGE_Y)
  NUDGE_STEPS = 25           #: nudge duration (RP.NUDGE_STEPS)

  def __init__(self, walker, variant):
    if variant not in ('tr', 'br'):
      raise ValueError(f"variant must be 'tr' or 'br', got {variant!r}")
    self._walker = walker
    self.variant = variant
    self._leg = 1
    self._trail = []
    self._nudge_left = 0
    self._nudge_sign = 1.0
    self._nudges = 0

  @property
  def leg(self):
    """Current 1-based leg (diagnostics only)."""
    return self._leg

  @property
  def nudges(self):
    """Stall-unstick engagements this episode (diagnostics only)."""
    return self._nudges

  def fresh(self):
    self._leg = 1
    self._trail = []
    self._nudge_left = 0
    self._nudge_sign = 1.0
    self._nudges = 0

  def _walk(self, o58, frame, lane):
    """One walker step in `frame`, translated so `lane` sits at 0."""
    if frame == 'north':
      s = rot_north(o58[:29])
      s[1] += lane
    elif frame == 'south':
      s = rot_south(o58[:29])
      s[1] -= lane
    else:
      s = o58[:29].copy()
      s[1] -= lane
    o = np.concatenate([s, o58[29:]]).astype(np.float32)
    return self._walker(o, 0.0, RP.V_SIDE)

  def act(self, o58, route):
    legs = _DRIVERS[(self.variant, route)]
    x, y = float(o58[0]), float(o58[1])
    #: latched advance: the leg index only ever grows within an episode.
    while self._leg < len(legs):
      coord, sgn, thr = legs[self._leg - 1][2]
      v = x if coord == 'x' else y
      if sgn * v >= sgn * thr:
        self._leg += 1
      else:
        break
    frame, lane, _ = legs[self._leg - 1]
    if lane == 'gx':
      lane = float(o58[29])
    elif lane == 'gy':
      lane = float(o58[30])
    #: stall-unstick bookkeeping (see class doc).
    self._trail.append((x, y))
    if self._nudge_left > 0:
      self._nudge_left -= 1
      lane = lane + self._nudge_sign * self.NUDGE_LANE
    elif len(self._trail) > self.STALL_WINDOW:
      x0, y0 = self._trail[-1 - self.STALL_WINDOW]
      if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 < self.STALL_MIN_DISP:
        self._nudge_left = self.NUDGE_STEPS
        self._nudge_sign = -self._nudge_sign
        self._nudges += 1
        lane = lane + self._nudge_sign * self.NUDGE_LANE
    return self.act_raw(o58, frame, lane)

  def act_raw(self, o58, frame, lane):
    """Direct (frame, lane) drive, bypassing the leg machinery (probes)."""
    return self._walk(o58, frame, lane)


def make_teacher(variant):
  """(cfg, teacher). Loads the frozen walker once; the base policy that
  load_controllers also restores is deliberately discarded (see module
  doc: it cannot be steered off the original U-maze goal)."""
  cfg, walker, _base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  return cfg, TwoRouteV3Teacher(walker, variant)


def teacher_episode(env, teacher, u, horizon=HORIZON, on_step=None):
  """One teacher episode under latent u. Returns the V2-shaped record plus
  final_xy / final_leg for driver diagnostics."""
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
          'steps': int(t + 1), 'return': ret,
          'final_xy': [round(float(o[0]), 3), round(float(o[1]), 3)],
          'final_leg': int(teacher.leg)}


def audit(variant, n=300, seed=101, p_active=P_ACTIVE):
  cfg, teacher = make_teacher(variant)
  env = envs_mod.make_env(env_name(variant), cfg, seed=seed)
  u_rng = np.random.default_rng(seed + 5000)
  rows = []
  for k in range(n):
    u = bool(u_rng.random() < p_active)
    rows.append(teacher_episode(env, teacher, u))
    if (k + 1) % 50 == 0:
      print(f'  [{variant}] {k + 1}/{n} episodes', flush=True)
  return rows


def summarize(rows):
  """Realized-route summary. Every gated quantity uses route_realized (the
  env's label), never route_intent -- see module doc."""
  n = len(rows)
  clear = [r for r in rows if not r['rockfall_active']]
  active = [r for r in rows if r['rockfall_active']]
  sc = [r for r in rows if r['route_realized'] == 'shortcut']
  dt = [r for r in rows if r['route_realized'] == 'detour']

  def rate(xs, key='success'):
    return round(float(np.mean([x[key] for x in xs])), 4) if xs else None

  def steps(xs):
    return round(float(np.mean([x['steps'] for x in xs])), 1) if xs else None

  def disc(xs):
    #: 0.99**steps averaged over SUCCESSES only (the reference convention).
    ok = [x for x in xs if x['success']]
    return (round(float(np.mean([GAMMA ** x['steps'] for x in ok])), 4)
            if ok else None)

  return {
      'n': n, 'n_clear': len(clear), 'n_active': len(active),
      'success': rate(rows),
      'failure': rate(rows, 'failure'),
      'timeout': round(float(np.mean(
          [not r['success'] and not r['failure'] for r in rows])), 4),
      'P_shortcut_given_clear': (round(float(np.mean(
          [r['route_realized'] == 'shortcut' for r in clear])), 4)
          if clear else None),
      'P_detour_given_active': (round(float(np.mean(
          [r['route_realized'] == 'detour' for r in active])), 4)
          if active else None),
      'route_choice_accuracy': round(float(np.mean(
          [r['route_realized'] == ('detour' if r['rockfall_active']
                                   else 'shortcut') for r in rows])), 4),
      'route_realized_counts': {
          'shortcut': len(sc), 'detour': len(dt),
          'none': n - len(sc) - len(dt)},
      'success_by_route': {'shortcut': rate(sc), 'detour': rate(dt)},
      'failure_by_route': {'shortcut': rate(sc, 'failure'),
                           'detour': rate(dt, 'failure')},
      'mean_steps_by_route': {'shortcut': steps(sc), 'detour': steps(dt)},
      'discounted_return_by_route': {'shortcut': disc(sc),
                                     'detour': disc(dt)},
      'discounted_return': disc(rows),
  }


def _targets_met(s):
  vals = (s['success'], s['P_shortcut_given_clear'],
          s['P_detour_given_active'])
  return all(v is not None and v > 0.90 for v in vals)


def run_variant(variant, n, seed, p_active, out_root):
  print(f'== variant {variant} ({env_name(variant)}) ==', flush=True)
  rows = audit(variant, n, seed, p_active)
  s = summarize(rows)
  print(json.dumps(s, indent=2), flush=True)
  ok = _targets_met(s)
  print(f'[{variant}] TARGETS ' + ('MET' if ok else 'NOT MET')
        + ' (success>0.90, P(sc|clear)>0.90, P(dt|active)>0.90;'
        ' realized routes)', flush=True)
  out_dir = os.path.join(out_root, variant)
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'teacher_audit.json'), 'w') as f:
    json.dump({'variant': variant, 'env': env_name(variant),
               'summary': s, 'targets_met': bool(ok),
               'seed': seed, 'p_active': p_active, 'horizon': HORIZON,
               'episodes': rows}, f, indent=2)
  print(f'-> {out_dir}/teacher_audit.json', flush=True)
  return ok


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=('tr', 'br', 'both'), default='both')
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=101)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  variants = ('tr', 'br') if args.variant == 'both' else (args.variant,)
  all_ok = True
  for v in variants:
    all_ok = run_variant(v, args.n, args.seed, args.p_active,
                         args.out_dir) and all_ok
  sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
  main()
