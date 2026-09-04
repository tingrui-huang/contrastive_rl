"""Expert for the V4 rockfall-wait benchmark + its qualification audit.

One route (the BR shortcut, goal (8,0)); the latent decides whether the
expert STOPS at the band mouth:

    revealed clear   ->  GO    walk straight through
    revealed active  ->  WAIT  at x >= MOUTH_X hold with zero torque
                               for HOLD_STEPS, then walk through

The expert does NOT know the latent before the mouth. It drives 'go' until
the ant is past the mouth line, reads ``env.revealed_rockfall_active`` in
that same observation (None before the trigger, the latent afterwards) and
decides once. Before the mouth its trajectory is identical for both
latents; the only thing the latent ever changes is the hold at the mouth.
intent='go' / 'wait' bypass the mouth reading (the do() experiments).

The walk is the V3 walker-only relay ('br','shortcut' driver: native lane 0
until x >= 6, then native lane goal_y) at the unchanged RP.V_SIDE = 1.1.
The hold is literally a = 0: the walker commanded with v_ref = 0 is NOT a
hold (measured drift 1-1.9 units), whereas zero torque leaves the ant
crouched in place (drift 0.1-0.4, z 0.38-0.57) and the walker restarts
from that pose 5/5 (feasibility, 60-step hold, this scratchpad).

HOLD_STEPS = ROCKFALL_STEPS + HOLD_MARGIN, counted from the step the ant is
first observed past the mouth line. The env fires its trigger on the same
condition BEFORE physics, so the env's clock is one step ahead of the
teacher's at most: releasing HOLD_MARGIN steps late keeps the teacher on the
safe side of the window by a wide margin, and the ~12 steps from the mouth
to the band add more.

Intents are 'go' / 'wait' (route intent is fixed). The audit computes every
gated quantity from the env's own info (band entry, failure, success), not
from intent:

  * P(success)                         > 0.90
  * P(failure | active)                = 0   (the waiting expert never dies)
  * P(failure | active, do(go))        > 0.90 (the hazard is real)
  * band entered in every kept episode (one route, no detour by accident)
  * rocks parked (rock_dropped False) at the end of every active episode
    that outlived the window

Run:  python scripts/rockfall_wait_v4_teacher.py [--n 300] [--seed 101]
Writes artifacts/rockfall_wait_v4/teacher_audit.json.
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
from crl import rockfall_wait_v4 as V4    # noqa: E402
import tworoute_v3_teacher as TT           # noqa: E402

OUT = 'artifacts/rockfall_wait_v4'
ENV_NAME = 'offline_ant_umaze_rockfall_wait_v4'
P_ACTIVE = TT.P_ACTIVE
HORIZON = TT.HORIZON      #: go ~77 steps, wait ~77 + 84 = ~161: far inside 400
GAMMA = TT.GAMMA
HOLD_MARGIN = 12
HOLD_STEPS = V4.ROCKFALL_STEPS + HOLD_MARGIN
INTENTS = ('go', 'wait')


class WaitV4Teacher(TT.TwoRouteV3Teacher):
  """BR-shortcut relay with an optional zero-torque hold at the mouth.

  act(o58, intent=None, revealed=None): with intent None the expert
  decides at the mouth from `revealed` (what the env shows there; it must
  be a bool by the time the ant is past the mouth line). intent 'go'
  walks; 'wait' walks until the ant is past the mouth line, holds for
  HOLD_STEPS, then walks. The hold is latched: once released the teacher
  never holds again in the episode. The V3 stall-unstick trail is cleared
  on release so the standstill is not read as a stall."""

  def __init__(self, walker):
    super().__init__(walker, 'br')
    self._hold_left = None
    self._released = False
    self._hold_steps_done = 0
    self._decision = None

  @property
  def decision(self):
    """'go' / 'wait' once made at the mouth (sighted mode), else None."""
    return self._decision

  @property
  def holding(self):
    return self._hold_left is not None and self._hold_left > 0

  @property
  def hold_steps_done(self):
    """Zero-torque steps emitted this episode (diagnostics)."""
    return self._hold_steps_done

  def fresh(self):
    super().fresh()
    self._hold_left = None
    self._released = False
    self._hold_steps_done = 0
    self._decision = None

  def act(self, o58, intent=None, revealed=None):
    x, y = float(o58[0]), float(o58[1])
    if intent is None:
      #: sighted-at-the-mouth: nothing to decide before the line; at the
      #: line the world has revealed the latent and the decision is final.
      if self._decision is None and V4.RockfallWaitV4Env._at_mouth(x, y):
        if revealed is None:
          raise RuntimeError('past the mouth line but the env revealed '
                             'nothing; pass revealed=env.revealed_'
                             'rockfall_active')
        self._decision = 'wait' if revealed else 'go'
      intent = self._decision or 'go'
    if intent not in INTENTS:
      raise ValueError(f"intent must be one of {INTENTS}, got {intent!r}")
    if intent == 'wait' and not self._released:
      if self._hold_left is None and V4.RockfallWaitV4Env._at_mouth(x, y):
        self._hold_left = HOLD_STEPS
      if self._hold_left is not None:
        if self._hold_left > 0:
          self._hold_left -= 1
          self._hold_steps_done += 1
          return np.zeros(8, np.float32)
        self._released = True
        self._trail = []
    return super().act(o58, 'shortcut')


def make_teacher():
  cfg, walker, _base_act, _, _ = TT.C.load_controllers(TT.RP.WALKER, TT.RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  return cfg, WaitV4Teacher(walker)


def teacher_episode(env, teacher, u, intent=None, horizon=HORIZON,
                    on_step=None):
  """One teacher episode under latent u. intent None = the expert, which
  reads the latent at the mouth ('wait' iff active); 'go' / 'wait' force
  it (the do() experiments). The returned 'intent' is what was done."""
  o = env.reset(rockfall_active=bool(u))
  teacher.fresh()
  ret, t = 0.0, 0
  info = {}
  for t in range(horizon):
    a = teacher.act(o, intent, revealed=env.revealed_rockfall_active)
    o2, r, done, info = env.step(a)
    if on_step is not None:
      on_step(o, a, o2, r, done, info)
    o = o2
    ret += float(r)
    if done or r > 0:
      break
  if intent is None:
    intent = teacher.decision or 'go'
  return {'rockfall_active': bool(u), 'intent': intent,
          'success': bool(info.get('success')),
          'failure': bool(info.get('failure')),
          'entered_hazard': bool(info.get('entered_hazard')),
          'rock_dropped': bool(info.get('rock_dropped')),
          'rock_waves': int(info.get('rock_waves', 0)),
          'rockfall_passed': bool(info.get('rockfall_passed')),
          'trigger_step': info.get('trigger_step'),
          'band_entry_step': info.get('band_entry_step'),
          'hold_steps': int(teacher.hold_steps_done),
          'route_realized': info.get('route'),
          'steps': int(t + 1), 'return': ret,
          'final_xy': [round(float(o[0]), 3), round(float(o[1]), 3)],
          'nudges': int(teacher.nudges)}


def audit(n=300, seed=101, p_active=P_ACTIVE, n_forced=100):
  cfg, teacher = make_teacher()
  env = envs_mod.make_env(ENV_NAME, cfg, seed=seed)
  u_rng = np.random.default_rng(seed + 5000)
  rows = []
  for k in range(n):
    u = bool(u_rng.random() < p_active)
    rows.append(teacher_episode(env, teacher, u))
    if (k + 1) % 50 == 0:
      print(f'  sighted {k + 1}/{n} episodes', flush=True)
  #: do(go) under the ACTIVE latent: the hazard must be real.
  env_f = envs_mod.make_env(ENV_NAME, cfg, seed=seed + 1)
  forced = [teacher_episode(env_f, teacher, True, intent='go')
            for _ in range(n_forced)]
  print(f'  do(go) | active: {n_forced} episodes', flush=True)
  return rows, forced


def summarize(rows, forced):
  n = len(rows)
  clear = [r for r in rows if not r['rockfall_active']]
  active = [r for r in rows if r['rockfall_active']]

  def rate(xs, key='success'):
    return round(float(np.mean([x[key] for x in xs])), 4) if xs else None

  def steps(xs):
    return round(float(np.mean([x['steps'] for x in xs])), 1) if xs else None

  def disc(xs):
    ok = [x for x in xs if x['success']]
    return (round(float(np.mean([GAMMA ** x['steps'] for x in ok])), 4)
            if ok else None)

  outlived = [r for r in active if r['rockfall_passed']]
  return {
      'n': n, 'n_clear': len(clear), 'n_active': len(active),
      'success': rate(rows), 'failure': rate(rows, 'failure'),
      'timeout': round(float(np.mean(
          [not r['success'] and not r['failure'] for r in rows])), 4),
      'success_by_latent': {'clear': rate(clear), 'active': rate(active)},
      'failure_by_latent': {'clear': rate(clear, 'failure'),
                            'active': rate(active, 'failure')},
      'entered_band': rate(rows, 'entered_hazard'),
      'mean_steps_by_latent': {'clear': steps(clear), 'active': steps(active)},
      'mean_hold_steps_active': (round(float(np.mean(
          [r['hold_steps'] for r in active])), 1) if active else None),
      'mean_waves_active': (round(float(np.mean(
          [r['rock_waves'] for r in active])), 2) if active else None),
      'rocks_parked_after_window': (round(float(np.mean(
          [not r['rock_dropped'] for r in outlived])), 4)
          if outlived else None),
      'discounted_return_by_latent': {'clear': disc(clear),
                                      'active': disc(active)},
      'discounted_return': disc(rows),
      'do_go_given_active': {
          'n': len(forced), 'success': rate(forced),
          'failure': rate(forced, 'failure'),
          'mean_steps': steps(forced), 'discounted_return': disc(forced)},
  }


def targets_met(s):
  return (s['success'] is not None and s['success'] > 0.90
          and s['failure_by_latent']['active'] == 0.0
          and s['entered_band'] == 1.0
          and s['rocks_parked_after_window'] == 1.0
          and s['do_go_given_active']['failure'] > 0.90)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--n-forced', type=int, default=100)
  ap.add_argument('--seed', type=int, default=101)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  rows, forced = audit(args.n, args.seed, args.p_active, args.n_forced)
  s = summarize(rows, forced)
  print(json.dumps(s, indent=2), flush=True)
  ok = targets_met(s)
  print('TARGETS ' + ('MET' if ok else 'NOT MET')
        + ' (success>0.90, P(fail|active)=0, band entered always, rocks'
        ' parked after the window, P(fail|active,do(go))>0.90)', flush=True)
  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'teacher_audit.json')
  with open(path, 'w') as f:
    json.dump({'env': ENV_NAME, 'summary': s, 'targets_met': bool(ok),
               'seed': args.seed, 'p_active': args.p_active,
               'horizon': HORIZON, 'hold_steps': HOLD_STEPS,
               'rockfall_steps': V4.ROCKFALL_STEPS,
               'wave_period': V4.WAVE_PERIOD, 'mouth_x': V4.MOUTH_X,
               'aim_x': list(V4.AIM_X),
               'episodes': rows, 'do_go_given_active': forced}, f, indent=2)
  print(f'-> {path}', flush=True)
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
