"""Privileged demonstrator and audit helpers for long two-rockfall V6.

The episode route coin is independent of both environment hazard coins.  A
detour episode follows the single safe perimeter route and never consults a
hazard schedule.  Every other episode follows the straight shortcut and
makes one local go/wait decision at each hazard mouth.  Zone ``i`` consults
only schedule ``i``; waiting at zone 1 may of course change the physical
arrival time at zone 2.

Safety is prospective, not merely ``open_now``: entry is allowed only when
the estimated mouth-to-clear interval plus a margin does not overlap that
zone's reset-time unsafe interval.  The timetable exists before the Ant
moves and is never added to the learner observation.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                       # noqa: E402
from crl import probe as probe_mod                     # noqa: E402
from crl import rockfall_clock_v6 as V6                # noqa: E402
import litter_pilot_common as C                        # noqa: E402
import rockfall_pilot as RP                            # noqa: E402
from tworoute_v3_teacher import (TwoRouteV3Teacher,    # noqa: E402
                                 rot_south)
from verify_offline_d4rl import build_offline_cfg       # noqa: E402

OUT = 'artifacts/rockfall_clock_v6'
ENV_NAME = 'offline_antmaze_rockfall_clock_v6'
TEACHER_DETOUR_PROB = 0.05
HORIZON = 800
GAMMA = 0.99

# Initial estimates are intentionally explicit.  The timing audit reports
# measured mouth/band/clear durations and these values together.  A teacher
# asks whether [now, now + crossing + margin] overlaps [t0, t0 + 72).
CROSSING_STEPS = {1: 50, 2: 50}
SAFETY_MARGIN = 8
RELEASE_MARGIN = 0

# One unambiguous route with latched turns.  The shortcut remains in the
# native east-facing frame.  The detour goes north, east, then south.
_DETOUR_LEGS = (
    ('north', 0.0, ('y', 1, 7.5)),
    ('native', 8.0, ('x', 1, 23.5)),
    ('south', 'gx', None),
)
# A deterministic early lane bias breaks the frozen walker's occasional
# symmetric limit cycle while turning north from the canonical east pose.
# Calibration over 30 reset-noise draws: -0.8 for the first 20/40/60 steps
# reached the north leg in 30/30 cases (unbiased: 27/30).  The target remains
# comfortably inside the four-unit-wide west corridor.
DETOUR_TURN_BIAS = -0.8
DETOUR_TURN_BIAS_STEPS = 60
_SHORTCUT_LEGS = (
    ('native', 0.0, ('x', 1, 22.0)),
    ('native', 'gy', None),
)


def intervals_overlap(start_a, end_a, start_b, end_b):
  """Half-open interval overlap used by both zone decisions."""
  return start_a < end_b and end_a > start_b


class LongTwoRockfallTeacher(TwoRouteV3Teacher):
  """Frozen walker relay with two independent privileged wait decisions."""

  def __init__(self, walker, crossing_steps=None, safety_margin=SAFETY_MARGIN):
    # The inherited V3 driver table is not used; its frame transforms and
    # frozen-walker adapter are reused without changing controller weights.
    super().__init__(walker, 'br')
    self.crossing_steps = dict(CROSSING_STEPS if crossing_steps is None
                               else crossing_steps)
    self.safety_margin = int(safety_margin)
    self._route = 'shortcut'
    #: EXECUTED decision per zone -- what the controller actually did. This
    #: keeps the meaning ``decisions`` has always had, so every existing
    #: reader (collector sidecar, audits) is unaffected.
    self._decision = {1: None, 2: None}
    #: NORMAL decision per zone -- what the privileged sighted rule says,
    #: preserved even when a deliberate override makes the controller do
    #: something else. ``None`` means the rule was never consulted (the
    #: blind ``intent='go'`` audit path, which must stay able to run against
    #: a redacted schedule).
    self._normal_decision = {1: None, 2: None}
    #: zone whose WAIT is deliberately overridden to GO for failure-bank
    #: collection; None in every ordinary episode.
    self._override_zone = None
    self._deliberate_override = {1: False, 2: False}
    self._release_step = {1: None, 2: None}
    self._hold_steps = {1: 0, 2: 0}
    self._holding_zone = None
    self._step = 0

  @property
  def route(self):
    return self._route

  @property
  def decisions(self):
    """The EXECUTED decision per zone (unchanged meaning)."""
    return dict(self._decision)

  @property
  def executed_decisions(self):
    """Alias of :attr:`decisions`, for code that wants to be explicit."""
    return dict(self._decision)

  @property
  def normal_decisions(self):
    """What the normal privileged sighted rule decided, per zone.

    Equal to :attr:`decisions` in every ordinary episode. Under a deliberate
    override it stays ``'wait'`` while the executed decision is ``'go'`` --
    that difference is the evidence a deliberate failure needs. ``None`` for
    a zone whose rule was never evaluated.
    """
    return dict(self._normal_decision)

  @property
  def deliberate_overrides(self):
    return dict(self._deliberate_override)

  @property
  def deliberate_override_zone(self):
    return self._override_zone

  @property
  def release_steps(self):
    return dict(self._release_step)

  @property
  def hold_steps(self):
    return dict(self._hold_steps)

  @property
  def holding_zone(self):
    return self._holding_zone

  def fresh(self, route='shortcut', deliberate_override_zone=None):
    """Reset for one episode.

    ``deliberate_override_zone`` (None / 1 / 2) is the FAILURE-COLLECTION
    interface: at that zone only, a normal decision of ``'wait'`` is executed
    as ``'go'`` and the hold is skipped. Every other zone -- and every
    episode that does not pass it -- behaves exactly as before.
    """
    if route not in ('shortcut', 'detour'):
      raise ValueError(f'route must be shortcut or detour, got {route!r}')
    if deliberate_override_zone not in (None, 1, 2):
      raise ValueError('deliberate_override_zone must be None, 1 or 2, got '
                       f'{deliberate_override_zone!r}')
    super().fresh()
    self._route = route
    self._decision = {1: None, 2: None}
    self._normal_decision = {1: None, 2: None}
    self._override_zone = deliberate_override_zone
    self._deliberate_override = {1: False, 2: False}
    self._release_step = {1: None, 2: None}
    self._hold_steps = {1: 0, 2: 0}
    self._holding_zone = None
    self._step = 0

  def _drive(self, o58, legs):
    """V3's deterministic latched relay, generalized to the long map."""
    x, y = float(o58[0]), float(o58[1])
    while self._leg < len(legs):
      coord, sign, threshold = legs[self._leg - 1][2]
      value = x if coord == 'x' else y
      if sign * value >= sign * threshold:
        self._leg += 1
      else:
        break
    frame, lane, _ = legs[self._leg - 1]
    if lane == 'gx':
      lane = float(o58[29])
    elif lane == 'gy':
      lane = float(o58[30])
    if (legs is _DETOUR_LEGS and self._leg == 1
        and self._step <= DETOUR_TURN_BIAS_STEPS):
      lane = DETOUR_TURN_BIAS

    # Same deterministic stall escape as TwoRouteV3Teacher.act().
    self._trail.append((x, y))
    if self._nudge_left > 0:
      self._nudge_left -= 1
      lane += self._nudge_sign * self.NUDGE_LANE
    elif len(self._trail) > self.STALL_WINDOW:
      x0, y0 = self._trail[-1 - self.STALL_WINDOW]
      if np.hypot(x - x0, y - y0) < self.STALL_MIN_DISP:
        self._nudge_left = self.NUDGE_STEPS
        self._nudge_sign = -self._nudge_sign
        self._nudges += 1
        lane += self._nudge_sign * self.NUDGE_LANE
    return self.act_raw(o58, frame, lane)

  def _shortcut_action(self, o58, schedule, force_go=False):
    t = int(schedule['t'])
    x, y = float(o58[0]), float(o58[1])
    for zone in (1, 2):
      if self._decision[zone] is not None:
        continue
      if not V6.RockfallClockV6Env._at_mouth(zone, x, y):
        continue
      if force_go:
        #: the blind audit intent. The sighted rule is NOT consulted at all,
        #: and the timetable is not even INDEXED -- so this path runs against
        #: a redacted schedule and a blind collector's blindness is a
        #: structural fact rather than a promise. The normal decision stays
        #: None rather than being overwritten with a 'go' it never made.
        self._normal_decision[zone] = None
        self._decision[zone] = 'go'
        break
      zschedule = schedule['zones'][zone]
      if not zschedule['active']:
        normal = 'go'
      else:
        cross_end = t + self.crossing_steps[zone] + self.safety_margin
        normal = ('wait' if intervals_overlap(
            int(zschedule['start']), int(zschedule['end']), t, cross_end)
            else 'go')
      self._normal_decision[zone] = normal
      #: DELIBERATE OVERRIDE, this zone only: keep the normal decision as the
      #: record and execute the opposite. Nothing happens unless the caller
      #: asked for this zone AND the normal rule actually said 'wait'.
      override = bool(self._override_zone == zone and normal == 'wait')
      self._deliberate_override[zone] = override
      executed = 'go' if override else normal
      self._decision[zone] = executed
      if executed == 'wait':
        self._release_step[zone] = int(zschedule['end']) + RELEASE_MARGIN
      break

    for zone in (1, 2):
      #: keyed on the EXECUTED decision. A deliberately overridden zone has
      #: executed == 'go' (its release step was never set), so it walks in
      #: while ``normal_decisions[zone]`` still records the 'wait' it knew.
      release = self._release_step[zone]
      if (self._decision[zone] == 'wait' and release is not None
          and t < release):
        self._holding_zone = zone
        self._hold_steps[zone] += 1
        return np.zeros(8, np.float32)

    if self._holding_zone is not None:
      # A long zero-torque hold looks like a stall to the relay.  Clear the
      # trail once, on release, rather than nudging the safe waiting pose.
      self._trail = []
      self._nudge_left = 0
      self._holding_zone = None
    return self._drive(o58, _SHORTCUT_LEGS)

  def act(self, o58, schedule, intent=None):
    """Return an action for sighted mode or a forced audit intent.

    ``intent='go'`` forces a blind shortcut with no waits;
    ``intent='detour'`` forces the safe route.  Normal ``None`` mode uses the
    independently selected route and, on shortcut episodes, both schedules.
    """
    if schedule is None:
      raise RuntimeError('pass env.schedule as the privileged side channel')
    self._step += 1
    if self._route == 'detour' or intent == 'detour':
      return self._drive(o58, _DETOUR_LEGS)
    if intent not in (None, 'go'):
      raise ValueError(f'intent must be None, go, or detour; got {intent!r}')
    return self._shortcut_action(o58, schedule, force_go=(intent == 'go'))


def apply_env_config(cfg, horizon=HORIZON, p_active_1=V6.P_ACTIVE_1,
                     p_active_2=V6.P_ACTIVE_2,
                     t0_range_1=(V6.T0_MIN_1, V6.T0_MAX_1),
                     t0_range_2=(V6.T0_MIN_2, V6.T0_MAX_2)):
  """Materialize every V6 benchmark parameter on a config object."""
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = int(horizon)
  cfg.rockfall_p_active_1 = float(p_active_1)
  cfg.rockfall_p_active_2 = float(p_active_2)
  cfg.rockfall_t0_min_1, cfg.rockfall_t0_max_1 = map(int, t0_range_1)
  cfg.rockfall_t0_min_2, cfg.rockfall_t0_max_2 = map(int, t0_range_2)
  return cfg


def make_teacher(horizon=HORIZON, p_active_1=V6.P_ACTIVE_1,
                 p_active_2=V6.P_ACTIVE_2,
                 t0_range_1=(V6.T0_MIN_1, V6.T0_MAX_1),
                 t0_range_2=(V6.T0_MIN_2, V6.T0_MAX_2)):
  # V6 is walker-only.  Load precisely that frozen controller instead of
  # making dataset generation depend on the unrelated legacy base-policy
  # checkpoint that TwoRoute V3 retired.
  cfg = build_offline_cfg()
  walker_params, walker_meta = probe_mod.load_residual(RP.WALKER)
  walker = probe_mod.WalkerController(walker_params)
  apply_env_config(cfg, horizon, p_active_1, p_active_2,
                   t0_range_1, t0_range_2)
  teacher = LongTwoRockfallTeacher(walker)
  teacher.walker_provenance = {
      'path': RP.WALKER,
      'sha256': C.sha256_file(RP.WALKER),
      'step': walker_meta.get('step'),
      'qualified': walker_meta.get('qualified'),
  }
  return cfg, teacher


def teacher_episode(env, teacher, hazards=None, route='shortcut', intent=None,
                    starts=None, horizon=HORIZON, on_step=None):
  """Run one episode; ``hazards``/``starts`` are optional forced audit maps."""
  if intent == 'detour':
    route = 'detour'
  hazards = {} if hazards is None else dict(hazards)
  starts = {} if starts is None else dict(starts)
  o = env.reset(
      rockfall_active_1=hazards.get(1), rockfall_active_2=hazards.get(2),
      rockfall_start_1=starts.get(1), rockfall_start_2=starts.get(2))
  u1 = env.privileged_rockfall_active_1
  u2 = env.privileged_rockfall_active_2
  t01 = env.privileged_sampled_start(1)
  t02 = env.privileged_sampled_start(2)
  teacher.fresh(route=route)
  ret, info, o_last = 0.0, {}, o
  for step in range(horizon):
    action = teacher.act(o_last, env.schedule, intent=intent)
    o_next, reward, done, info = env.step(action)
    if on_step is not None:
      on_step(o_last, action, o_next, reward, done, info)
    o_last = o_next
    ret += float(reward)
    if done or reward > 0:
      break
  success = bool(info.get('success'))
  failure = bool(info.get('failure'))
  return {
      'u1': bool(u1), 'u2': bool(u2), 'latent': f'U{int(u1)}{int(u2)}',
      't0_1': int(t01), 't0_2': int(t02),
      'route': route, 'route_realized': info.get('route'),
      'success': success, 'failure': failure,
      'timeout': bool(not success and not failure),
      'failure_zone': info.get('failure_zone'),
      'wait_zone_1': teacher.decisions[1] == 'wait',
      'wait_zone_2': teacher.decisions[2] == 'wait',
      'decision_zone_1': teacher.decisions[1],
      'decision_zone_2': teacher.decisions[2],
      'hold_steps_zone_1': int(teacher.hold_steps[1]),
      'hold_steps_zone_2': int(teacher.hold_steps[2]),
      'release_step_zone_1': teacher.release_steps[1],
      'release_step_zone_2': teacher.release_steps[2],
      'mouth_step_zone_1': info.get('mouth_step_1'),
      'mouth_step_zone_2': info.get('mouth_step_2'),
      'band_entry_step_zone_1': info.get('band_entry_step_1'),
      'band_entry_step_zone_2': info.get('band_entry_step_2'),
      'rock_waves_zone_1': int(info.get('rock_waves_1', 0)),
      'rock_waves_zone_2': int(info.get('rock_waves_2', 0)),
      'steps': int(step + 1), 'return': ret,
      'final_xy': [round(float(o_last[0]), 3), round(float(o_last[1]), 3)],
      'nudges': int(teacher.nudges),
  }


def _rate(rows, key):
  return (round(float(np.mean([bool(r[key]) for r in rows])), 4)
          if rows else None)


def summarize(rows):
  """Dataset-style composition and outcome summary."""
  out = {
      'n': len(rows), 'success': _rate(rows, 'success'),
      'failure': _rate(rows, 'failure'), 'timeout': _rate(rows, 'timeout'),
      'route_fraction': {
          'shortcut': _rate(rows, 'route_shortcut'),
          'detour': _rate(rows, 'route_detour'),
      },
      'by_latent': {},
  }
  for u1, u2 in ((0, 0), (1, 0), (0, 1), (1, 1)):
    key = f'U{u1}{u2}'
    group = [r for r in rows if r['u1'] == bool(u1)
             and r['u2'] == bool(u2)]
    out['by_latent'][key] = {
        'n': len(group), 'fraction': round(len(group) / len(rows), 4)
        if rows else None,
        'detour_fraction': _rate(group, 'route_detour'),
        'success': _rate(group, 'success'),
        'failure': _rate(group, 'failure'),
        'timeout': _rate(group, 'timeout'),
    }
  shortcut = [r for r in rows if r['route_shortcut']]
  out['shortcut_waits'] = {
      'n': len(shortcut),
      'zone_1': _rate(shortcut, 'wait_zone_1'),
      'zone_2': _rate(shortcut, 'wait_zone_2'),
      'both': round(float(np.mean([
          r['wait_zone_1'] and r['wait_zone_2'] for r in shortcut])), 4)
      if shortcut else None,
      'neither': round(float(np.mean([
          not r['wait_zone_1'] and not r['wait_zone_2']
          for r in shortcut])), 4) if shortcut else None,
  }
  return out


def audit(n=400, seed=101, teacher_detour_prob=TEACHER_DETOUR_PROB,
          p_active_1=V6.P_ACTIVE_1, p_active_2=V6.P_ACTIVE_2,
          horizon=HORIZON):
  """Natural independent draws plus forced-combination sanity episodes."""
  cfg, teacher = make_teacher(horizon, p_active_1, p_active_2)
  env = envs_mod.make_env(ENV_NAME, cfg, seed=seed)
  route_rng = np.random.default_rng(seed + 370_003)
  rows = []
  for episode in range(n):
    route = ('detour' if route_rng.random() < teacher_detour_prob
             else 'shortcut')
    row = teacher_episode(env, teacher, route=route, horizon=horizon)
    row.update({'episode_id': episode, 'route_shortcut': route == 'shortcut',
                'route_detour': route == 'detour'})
    rows.append(row)
    if (episode + 1) % 50 == 0:
      print(f'  natural teacher {episode + 1}/{n}', flush=True)
  forced = {}
  for u1, u2 in ((0, 0), (1, 0), (0, 1), (1, 1)):
    key = f'U{u1}{u2}'
    forced[key] = {
        'shortcut': teacher_episode(
            env, teacher, {1: u1, 2: u2}, route='shortcut', horizon=horizon),
        'detour': teacher_episode(
            env, teacher, {1: u1, 2: u2}, route='detour', horizon=horizon),
    }
  return rows, forced


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=400)
  parser.add_argument('--seed', type=int, default=101)
  parser.add_argument('--teacher-detour-prob', type=float,
                      default=TEACHER_DETOUR_PROB)
  parser.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1)
  parser.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2)
  parser.add_argument('--horizon', type=int, default=HORIZON)
  parser.add_argument('--out-dir', default=OUT)
  args = parser.parse_args()
  rows, forced = audit(args.n, args.seed, args.teacher_detour_prob,
                       args.p_active_1, args.p_active_2, args.horizon)
  summary = summarize(rows)
  print(json.dumps(summary, indent=2), flush=True)
  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'teacher_audit.json')
  with open(path, 'w') as handle:
    json.dump({
        'env': ENV_NAME, 'environment_version': V6.ENV_VERSION,
        'seed': args.seed, 'route_rng_seed': args.seed + 370_003,
        'p_active_1': args.p_active_1, 'p_active_2': args.p_active_2,
        'teacher_detour_prob': args.teacher_detour_prob,
        't0_range_1': list((V6.T0_MIN_1, V6.T0_MAX_1)),
        't0_range_2': list((V6.T0_MIN_2, V6.T0_MAX_2)),
        'crossing_steps': CROSSING_STEPS,
        'safety_margin': SAFETY_MARGIN, 'horizon': args.horizon,
        'summary': summary, 'episodes': rows, 'forced': forced,
    }, handle, indent=2)
  print('->', path, flush=True)


if __name__ == '__main__':
  main()
