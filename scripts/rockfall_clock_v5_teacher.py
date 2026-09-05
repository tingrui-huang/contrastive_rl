"""Expert for the V5 rockfall-clock benchmark + its qualification audit.

The rockfall runs on its own clock (crl/rockfall_clock_v5.py): an active
latent drops six waves over ROCKFALL_STEPS steps starting at t0 ~ U{0..30},
whether or not the ant is near the band. The expert KNOWS the timetable
from t = 0 (``env.schedule``, a privileged side channel that never reaches
the 58-dim obs) but only ACTS on it once, at the mouth line:

    clear, or the burst misses the crossing  ->  GO    walk straight through
    burst overlaps the crossing              ->  WAIT  hold at x >= MOUTH_X
                                                       with zero torque until
                                                       the rocks are parked,
                                                       then walk

Overlap rule. With t the env step at the mouth, the burst [start, end]
overlaps the crossing iff ``start < t + CROSS_MAX`` and ``end > t +
MOUTH_TO_BAND_MIN``. Both bounds are on ROCK CONTACT, not on band
entry/exit: the pattern's lethal reach extends ~0.9 units west of the
band's west edge (first landing x ~3.2 minus the rock radius, ant leg reach
~0.75) and ~0.7 units east of its east edge (rocks clipped to x <= 5.4 hit
an ant at x ~5.5-6 during their fall). Measured with forced 'go' on active
episodes and the burst placed by forced rockfall_start (deaths/n):

  by d_end = end - mouth_step:    2:0/2  3:0/2  4:0/2  5:0/5  6:0/5  7:0/2
                                  8:0/9  9:1/8  10:0/6 11:1/7 12:3/7 13:4/5
                                  14:6/6 15:5/5 16:2/3 17:1/1 18:2/2 19:2/2
  by d_start = start - mouth_step: 33:0/2 34:1/1 35:1/6 36:1/7 37:0/7 38:0/7
                                  39:0/4 40:0/3 41:0/8 42:0/8 43..51:0/37

so the first lethal end is d_end = 9 (one step BEFORE band entry, min 9)
and the last lethal start is d_start = 36 (contact at drop+6 just east of
the band exit). MOUTH_TO_BAND_MIN = 4 and CROSS_MAX = 42 leave 5-6 steps
of margin on either edge. The hold releases at ``end + RELEASE_MARGIN``:
the env parks the rocks at the top of the ``end`` step, before physics,
and the first contact is still >= 9 steps away, so a zero margin is safe.
The decision is made once and is final. In the natural range (t0 in
[0, 30], mouth at 12-21) d_end >= 51 and d_start <= 18, so the rule is
'wait' on every active shortcut episode; the edges are only reachable
through forced rockfall_start probes.

Route. The expert takes the BR shortcut, except that with probability
P_FAR -- a coin drawn by the CALLER's rng at episode start, independent of
the latent, exactly like the discrete WindyCorridor expert's ``NEAR if
rng.random() < p_near else FAR`` -- it takes the V3-br DETOUR (north
column, top row, down the east column). The detour never crosses the band
and never waits: pure coverage for the learner, no information about the
latent.

Forced intents (the do() experiments): 'go' never holds; 'wait' is the
BLIND always-wait reference and holds at the mouth until env step
``max(BLIND_WAIT_UNTIL, t)`` = 102 regardless of the latent; 'detour' takes
the detour. The hold is literally a = 0 (see the V4 teacher for why: a
walker commanded with v_ref = 0 drifts 1-1.9 units, zero torque leaves the
ant crouched in place); the V3 stall-unstick trail is cleared on release.

Audit gates (from the env's own info, never from intent):

  * P(success)                              > 0.90
  * P(failure | active)                     = 0
  * P(failure | active, do(go))             > 0.90 (the hazard is real)
  * rocks parked after the window in every active shortcut episode that
    outlived it
  * band entered in every shortcut episode, never in a detour episode
  * blind do(wait) failure = 0; do(detour) failure = 0

Run:  python scripts/rockfall_clock_v5_teacher.py [--n 300] [--seed 101]
                                                  [--p-far 0.05]
Writes artifacts/rockfall_clock_v5/teacher_audit.json.
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
from crl import rockfall_clock_v5 as V5   # noqa: E402
import tworoute_v3_teacher as TT           # noqa: E402

OUT = 'artifacts/rockfall_clock_v5'
ENV_NAME = 'offline_ant_umaze_rockfall_clock_v5'
P_ACTIVE = TT.P_ACTIVE
P_FAR = 0.05              #: detour coin (coverage only; 0 in the plain set)
HORIZON = TT.HORIZON      #: go ~77, wait <= ~77 + 102, detour ~225: < 400
GAMMA = TT.GAMMA
ROUTES = ('shortcut', 'detour')
INTENTS = ('go', 'wait', 'detour')
#: the burst must END after mouth + this to matter. Bound on the first ROCK
#: CONTACT after the mouth, not on band entry: measured first death at
#: d_end = 9 with forced go (0/31 at d_end <= 8), one step before band
#: entry (min 9) because the pattern's reach extends ~0.9 units west of the
#: band edge. 4 leaves 5 steps of margin; the extra hold is free.
MOUTH_TO_BAND_MIN = 4
#: the burst must START before mouth + this to matter. Bound on the last
#: LETHAL DROP after the mouth, not on band exit (max 38): measured last
#: death at d_start = 36 with forced go (0/53 at d_start >= 42) -- a wave
#: dropped then lands at drop+6 just east of the band exit (rocks clipped
#: to x <= 5.4 reach an ant at x ~5.5-6). 42 leaves ~6 steps of margin.
CROSS_MAX = 42
#: release offset from the parking step (0: the rocks are parked at the top
#: of that step, before physics, and the first contact is still >= 9 steps
#: away).
RELEASE_MARGIN = 0


class ClockV5Teacher(TT.TwoRouteV3Teacher):
  """BR relay with a schedule-driven zero-torque hold at the mouth.

  act(o58, schedule, intent=None): ``schedule`` is ``env.schedule``. With
  intent None the expert decides ONCE at the mouth from the timetable (see
  module doc). 'go' never holds; 'wait' is the blind always-wait reference
  (hold until env step BLIND_WAIT_UNTIL); 'detour' -- or fresh(route=
  'detour') -- runs the V3-br detour driver and never holds. The hold is
  latched: once released the teacher never holds again in the episode.

  Clocks: the teacher keeps its OWN step counter (one act() call per env
  step, reset by fresh()). Only the sighted decision reads the timetable
  (``schedule['start'] / ['end']``, with ``schedule['t']`` as the mouth
  time); the blind 'wait' release and the hold countdown use the teacher's
  counter, so the blind arm never dereferences the privileged channel."""

  def __init__(self, walker):
    super().__init__(walker, 'br')
    self._route = 'shortcut'
    self._decision = None
    self._release_step = None
    self._holding = False
    self._released = False
    self._hold_steps_done = 0
    self._step = 0

  @property
  def decision(self):
    """'go' / 'wait' once made at the mouth (sighted mode), else None."""
    return self._decision

  @property
  def route(self):
    """Intended route this episode ('shortcut' / 'detour')."""
    return self._route

  @property
  def holding(self):
    return bool(self._holding)

  @property
  def hold_steps_done(self):
    """Zero-torque steps emitted this episode (diagnostics)."""
    return self._hold_steps_done

  @property
  def release_step(self):
    """Env step at which the hold ends (None if no hold was scheduled)."""
    return self._release_step

  def fresh(self, route='shortcut'):
    if route not in ROUTES:
      raise ValueError(f'route must be one of {ROUTES}, got {route!r}')
    super().fresh()
    self._route = route
    self._decision = None
    self._release_step = None
    self._holding = False
    self._released = False
    self._hold_steps_done = 0
    self._step = 0

  def act(self, o58, schedule, intent=None):
    if schedule is None:
      raise RuntimeError('pass schedule=env.schedule (the privileged '
                         'timetable side channel)')
    #: the teacher's own clock: env steps taken so far in this episode
    #: (== schedule['t'] under the one-act-per-step contract).
    t = self._step
    self._step += 1
    if self._route == 'detour' or intent == 'detour':
      #: the detour never crosses the band and never holds.
      return super().act(o58, 'detour')
    x, y = float(o58[0]), float(o58[1])
    at_mouth = V5.RockfallClockV5Env._at_mouth(x, y)
    if intent is None:
      #: the sighted expert: nothing to decide before the line; at the line
      #: the timetable says whether the burst overlaps the crossing, and
      #: the decision is final. This is the only place the timetable is
      #: read.
      if self._decision is None and at_mouth:
        if not schedule['active']:
          self._decision = 'go'
        else:
          t_env = int(schedule['t'])
          overlap = (schedule['start'] < t_env + CROSS_MAX
                     and schedule['end'] > t_env + MOUTH_TO_BAND_MIN)
          if overlap:
            self._decision = 'wait'
            self._release_step = int(schedule['end']) + RELEASE_MARGIN
          else:
            self._decision = 'go'
      intent = self._decision or 'go'
    elif intent == 'wait':
      #: the BLIND always-wait reference: hold until the latest possible
      #: burst has been parked, whatever the latent. Timed from the
      #: teacher's own counter; the schedule is not consulted.
      if self._release_step is None and at_mouth:
        self._release_step = max(V5.BLIND_WAIT_UNTIL, t)
    if intent not in INTENTS:
      raise ValueError(f'intent must be one of {INTENTS}, got {intent!r}')
    if (intent == 'wait' and not self._released
        and self._release_step is not None):
      if t < self._release_step:
        self._holding = True
        self._hold_steps_done += 1
        return np.zeros(8, np.float32)
      #: release: clear the V3 stall-unstick trail so the standstill is not
      #: read as a stall, and latch.
      self._holding = False
      self._released = True
      self._trail = []
    return super().act(o58, 'shortcut')


def make_teacher():
  cfg, walker, _base_act, _, _ = TT.C.load_controllers(TT.RP.WALKER, TT.RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  return cfg, ClockV5Teacher(walker)


def teacher_episode(env, teacher, u, route='shortcut', intent=None,
                    rockfall_start=None, horizon=HORIZON, on_step=None):
  """One teacher episode. u = latent (None lets the env draw it: the
  'natural latent' arms); route = intended route (the caller's coin);
  intent None = the expert, 'go' / 'wait' / 'detour' force it (the do()
  experiments); rockfall_start overrides t0 (probes). The returned 'intent'
  is what was done."""
  if intent == 'detour':
    route = 'detour'
  o = env.reset(rockfall_active=None if u is None else bool(u),
                rockfall_start=rockfall_start)
  teacher.fresh(route=route)
  ret, t = 0.0, 0
  info = {}
  for t in range(horizon):
    a = teacher.act(o, env.schedule, intent)
    o2, r, done, info = env.step(a)
    if on_step is not None:
      on_step(o, a, o2, r, done, info)
    o = o2
    ret += float(r)
    if done or r > 0:
      break
  if route == 'detour':
    done_intent = 'detour'
  elif intent is None:
    done_intent = teacher.decision or 'go'
  else:
    done_intent = intent
  return {'rockfall_active': bool(env.privileged_rockfall_active),
          'route': route, 'route_realized': info.get('route'),
          'intent': done_intent,
          'success': bool(info.get('success')),
          'failure': bool(info.get('failure')),
          'entered_hazard': bool(info.get('entered_hazard')),
          'rock_dropped': bool(info.get('rock_dropped')),
          'rock_waves': int(info.get('rock_waves', 0)),
          'rockfall_passed': bool(info.get('rockfall_passed')),
          'rockfall_start': info.get('rockfall_start'),
          'rockfall_end': info.get('rockfall_end'),
          'mouth_step': info.get('mouth_step'),
          'band_entry_step': info.get('band_entry_step'),
          'hold_steps': int(teacher.hold_steps_done),
          'release_step': teacher.release_step,
          'steps': int(t + 1), 'return': ret,
          'final_xy': [round(float(o[0]), 3), round(float(o[1]), 3)],
          'nudges': int(teacher.nudges)}


def audit(n=300, seed=101, p_active=P_ACTIVE, p_far=P_FAR, n_forced=100):
  """Sighted expert with the caller's latent coin (seed+5000) and route
  coin (seed+7000), then the forced arms on fresh envs: do(go) | active,
  the blind do(wait) and do(detour) under the natural latent."""
  cfg, teacher = make_teacher()
  env = envs_mod.make_env(ENV_NAME, cfg, seed=seed)
  u_rng = np.random.default_rng(seed + 5000)
  route_rng = np.random.default_rng(seed + 7000)
  rows = []
  for k in range(n):
    u = bool(u_rng.random() < p_active)
    route = 'detour' if route_rng.random() < p_far else 'shortcut'
    rows.append(teacher_episode(env, teacher, u, route=route))
    if (k + 1) % 50 == 0:
      print(f'  sighted {k + 1}/{n} episodes', flush=True)
  #: do(go) under the ACTIVE latent: the hazard must be real.
  env_f = envs_mod.make_env(ENV_NAME, cfg, seed=seed + 1)
  forced_go = [teacher_episode(env_f, teacher, True, intent='go')
               for _ in range(n_forced)]
  print(f'  do(go) | active: {n_forced} episodes', flush=True)
  #: the blind always-wait reference and the detour, natural latent.
  env_w = envs_mod.make_env(ENV_NAME, cfg, seed=seed + 2)
  forced_wait = [teacher_episode(env_w, teacher, None, intent='wait')
                 for _ in range(50)]
  print('  do(wait blind): 50 episodes', flush=True)
  env_d = envs_mod.make_env(ENV_NAME, cfg, seed=seed + 3)
  forced_detour = [teacher_episode(env_d, teacher, None, intent='detour')
                   for _ in range(50)]
  print('  do(detour): 50 episodes', flush=True)
  forced = {'do_go_given_active': forced_go,
            'do_wait_blind': forced_wait,
            'do_detour': forced_detour}
  return rows, forced


def _rate(xs, key='success'):
  return round(float(np.mean([x[key] for x in xs])), 4) if xs else None


def _steps(xs):
  return round(float(np.mean([x['steps'] for x in xs])), 1) if xs else None


def _disc(xs):
  #: 0.99**steps averaged over SUCCESSES only (the reference convention).
  ok = [x for x in xs if x['success']]
  return (round(float(np.mean([GAMMA ** x['steps'] for x in ok])), 4)
          if ok else None)


def _arm(xs):
  return {'n': len(xs), 'success': _rate(xs), 'failure': _rate(xs, 'failure'),
          'entered_band': _rate(xs, 'entered_hazard'),
          'mean_steps': _steps(xs), 'discounted_return': _disc(xs),
          'n_active': int(sum(x['rockfall_active'] for x in xs)),
          'mean_hold_steps': (round(float(np.mean(
              [x['hold_steps'] for x in xs])), 1) if xs else None)}


def summarize(rows, forced):
  n = len(rows)
  clear = [r for r in rows if not r['rockfall_active']]
  active = [r for r in rows if r['rockfall_active']]
  sc = [r for r in rows if r['route'] == 'shortcut']
  dt = [r for r in rows if r['route'] == 'detour']
  sc_active = [r for r in sc if r['rockfall_active']]
  sc_clear = [r for r in sc if not r['rockfall_active']]
  outlived = [r for r in sc_active if r['rockfall_passed']]
  holds = [r['hold_steps'] for r in sc_active]
  return {
      'n': n, 'n_clear': len(clear), 'n_active': len(active),
      'n_shortcut': len(sc), 'n_detour': len(dt),
      'success': _rate(rows), 'failure': _rate(rows, 'failure'),
      'timeout': round(float(np.mean(
          [not r['success'] and not r['failure'] for r in rows])), 4),
      'success_by_latent': {'clear': _rate(clear), 'active': _rate(active)},
      'failure_by_latent': {'clear': _rate(clear, 'failure'),
                            'active': _rate(active, 'failure')},
      'success_by_route': {'shortcut': _rate(sc), 'detour': _rate(dt)},
      'failure_by_route': {'shortcut': _rate(sc, 'failure'),
                           'detour': _rate(dt, 'failure')},
      'mean_steps_by_route': {'shortcut': _steps(sc), 'detour': _steps(dt)},
      'discounted_return_by_route': {'shortcut': _disc(sc),
                                     'detour': _disc(dt)},
      'entered_band_shortcut': _rate(sc, 'entered_hazard'),
      'entered_band_detour': _rate(dt, 'entered_hazard'),
      'mean_steps_by_latent': {'clear': _steps(clear),
                               'active': _steps(active)},
      'hold_steps_active_shortcut': (
          {'mean': round(float(np.mean(holds)), 1),
           'min': int(np.min(holds)), 'max': int(np.max(holds))}
          if holds else None),
      'P_wait_given_active_shortcut': (round(float(np.mean(
          [r['intent'] == 'wait' for r in sc_active])), 4)
          if sc_active else None),
      'P_wait_given_clear': (round(float(np.mean(
          [r['intent'] == 'wait' for r in sc_clear])), 4)
          if sc_clear else None),
      'mean_waves_active': (round(float(np.mean(
          [r['rock_waves'] for r in active])), 2) if active else None),
      'rocks_parked_after_window': (round(float(np.mean(
          [not r['rock_dropped'] for r in outlived])), 4)
          if outlived else None),
      'discounted_return_by_latent': {'clear': _disc(clear),
                                      'active': _disc(active)},
      'discounted_return': _disc(rows),
      'do_go_given_active': _arm(forced['do_go_given_active']),
      'do_wait_blind': _arm(forced['do_wait_blind']),
      'do_detour': _arm(forced['do_detour']),
  }


def targets_met(s):
  def ok_or_absent(v, target):
    return v is None or v == target
  return (s['success'] is not None and s['success'] > 0.90
          and ok_or_absent(s['failure_by_latent']['active'], 0.0)
          and s['do_go_given_active']['failure'] is not None
          and s['do_go_given_active']['failure'] > 0.90
          and ok_or_absent(s['rocks_parked_after_window'], 1.0)
          and ok_or_absent(s['entered_band_shortcut'], 1.0)
          and ok_or_absent(s['entered_band_detour'], 0.0)
          and ok_or_absent(s['do_wait_blind']['failure'], 0.0)
          and ok_or_absent(s['do_detour']['failure'], 0.0))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--n-forced', type=int, default=100)
  ap.add_argument('--seed', type=int, default=101)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--p-far', type=float, default=P_FAR)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  rows, forced = audit(args.n, args.seed, args.p_active, args.p_far,
                       args.n_forced)
  s = summarize(rows, forced)
  print(json.dumps(s, indent=2), flush=True)
  ok = targets_met(s)
  print('TARGETS ' + ('MET' if ok else 'NOT MET')
        + ' (success>0.90, P(fail|active)=0, P(fail|active,do(go))>0.90,'
        ' rocks parked after the window, band entered in every shortcut'
        ' episode and never on the detour, blind do(wait) and do(detour)'
        ' never fail)', flush=True)
  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'teacher_audit.json')
  with open(path, 'w') as f:
    json.dump({'env': ENV_NAME, 'summary': s, 'targets_met': bool(ok),
               'seed': args.seed, 'p_active': args.p_active,
               'p_far': args.p_far, 'horizon': HORIZON,
               't0_range': [V5.T0_MIN, V5.T0_MAX],
               'blind_wait_until': V5.BLIND_WAIT_UNTIL,
               'mouth_to_band_min': MOUTH_TO_BAND_MIN,
               'cross_max': CROSS_MAX, 'release_margin': RELEASE_MARGIN,
               'rockfall_steps': V5.ROCKFALL_STEPS,
               'wave_period': V5.WAVE_PERIOD, 'mouth_x': V5.MOUTH_X,
               'aim_x': list(V5.AIM_X),
               'episodes': rows, **forced}, f, indent=2)
  print(f'-> {path}', flush=True)
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
