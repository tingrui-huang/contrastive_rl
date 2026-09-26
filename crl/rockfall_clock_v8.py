"""Long two-rockfall AntMaze V8 benchmark: V6 with a goal REGION.

V8 is an isolated successor to :mod:`crl.rockfall_clock_v6`; V6 is not
modified and neither are any of its artifacts.  The map, the 58-dim learner
contract, the two-route topology, the eight rock bodies, both hazard coins,
both absolute clocks, every RNG stream offset and every line of the physics
are V6's -- this class inherits them.

THE ONE DIFFERENCE.  Success is no longer "torso within 0.5 of the goal
point in cell (1, 7)".  It is "torso anywhere in the east column", the
region to the right of BOTH routes::

      x=-2      x=22   x=26
       +----------+------+  y=10
       |  upper passage  |  row 3 (detour)
       +--+-------+ GOAL |  y=6
       |  | wall  | REG- |  row 2
       +--+-------+ ION  |  y=2
       |R shortcut|      |  row 1
       +----------+------+  y=-2

    GOAL_REGION = x in [22, 26], y in [-2, 10]   (cells (1,7), (2,7), (3,7))

The shortcut enters the region at (22, 0) and the upper passage at (22, 8),
so the detour no longer has to walk south down the east column to reach
the old goal cell: either route succeeds the moment it leaves its corridor.

The commanded goal the learner sees is still a single zero-padded XY point
(the 58-dim contract is unchanged).  It is drawn around the region centre
(24, 4), which is equidistant (4.47) from both corridor exits, so the goal
vector itself does not favour either route; see ``_eval_goal_xy``.

Everything else -- the latch-on-success rule (a rock contact after success
is not a death), the absorbing death, the route labels, ``info`` fields and
the privileged teacher channel -- is V6's.  ``info`` additionally carries
``in_goal_region`` and ``goal_region_step`` (first step inside the region).
"""
import numpy as np

from crl import rockfall_clock_v6 as v6
from crl.d4rl_ant import SCALING, OfflineD4rlAntUMazeEnv
from crl.rockfall_clock_v6 import RockfallClockV6Env
# Re-exported unchanged so the V8 collectors can read V8.<name> as the V6
# ones read V6.<name>; every one of these is V6's value.
from crl.rockfall_clock_v6 import (  # noqa: F401
    _ACTIVE_SEED_OFFSET_1, _ACTIVE_SEED_OFFSET_2, _JITTER_SEED_OFFSET_1,
    _JITTER_SEED_OFFSET_2, _SCHED_SEED_OFFSET_1, _SCHED_SEED_OFFSET_2,
    DETOUR_MAX_X, DETOUR_Y, HAZARD_HALF_Y, HAZARD_X, P_ACTIVE_1, P_ACTIVE_2,
    ROCKFALL_STEPS, T0_MAX_1, T0_MAX_2, T0_MIN_1, T0_MIN_2, WAVE_PERIOD,
    hazard_zone)

ENV_VERSION = 'rockfall_clock_v8_goal_region'

#: The east column, rows 1-3.  Walls bound it on the east (x = 26), north
#: (y = 10) and south (y = -2), so in practice the test is ``x >= 22``.
GOAL_REGION_X = (22.0, 26.0)
GOAL_REGION_Y = (-2.0, 10.0)
GOAL_REGION_CENTER_XY = (24.0, 4.0)
#: Half-width of the uniform noise on the commanded goal in 'd4rl' mode.
#: 0.75 matches the mean of V6's one-sided d4rl noise, but symmetric, so the
#: goal stays centred between the two corridor exits.
GOAL_NOISE = 0.75
#: 'detour_end': a fixed commanded goal at the detour's end point in the V6
#: far50 dataset. Detours come down the east column from the north and stop
#: ~0.42 north of their goal, median end y 1.18; x is the column centre.
DETOUR_END_XY = (24.0, 1.18)


def in_goal_region(x, y):
  return (GOAL_REGION_X[0] <= x <= GOAL_REGION_X[1]
          and GOAL_REGION_Y[0] <= y <= GOAL_REGION_Y[1])


class RockfallClockV8Env(RockfallClockV6Env):
  """V6 with success = reaching the east column after either route."""

  def __init__(self, *args, **kwargs):
    # Two V8-only goal modes; success stays the region test in both, and the
    # base class is handed 'd4rl' so it never sees the names.
    #   'shortcut_end': V6's goal -- the old goal cell (1, 7) at (24, 0) plus
    #     V6's one-sided d4rl noise, mean ~(24.75, 0.75).
    #   'detour_end': the fixed point DETOUR_END_XY, no noise.
    self._v8_goal_mode = kwargs.get('eval_goal_mode')
    if self._v8_goal_mode in ('shortcut_end', 'detour_end'):
      kwargs['eval_goal_mode'] = 'd4rl'
    else:
      self._v8_goal_mode = None
    super().__init__(*args, **kwargs)
    self._shortcut_end_xy = self._cell_xy(v6.GOAL_CELL)
    self._eval_goal_cell_xy = np.array(GOAL_REGION_CENTER_XY)
    self._goal_region_step = None

  def _eval_goal_xy(self):
    if self._v8_goal_mode == 'detour_end':
      return np.asarray(DETOUR_END_XY, np.float32)
    if self._v8_goal_mode == 'shortcut_end':
      # Same draws as d4rl_ant._d4rl_goal_sampler on V6's goal cell.
      noise = (self._rng.uniform(0.0, 0.25 * SCALING, 2)
               + self._rng.uniform(0.0, 0.5, 2) * 0.25 * SCALING)
      return np.maximum(self._shortcut_end_xy + noise, 0.0).astype(np.float32)
    center = np.asarray(GOAL_REGION_CENTER_XY, np.float32)
    if self.eval_goal_mode == 'd4rl':
      noise = self._rng.uniform(-GOAL_NOISE, GOAL_NOISE, 2)
      return (center + noise).astype(np.float32)
    if self.eval_goal_mode == 'fixed':
      return center
    return super()._eval_goal_xy()

  def reset(self, *args, **kwargs):
    self._goal_region_step = None
    return super().reset(*args, **kwargs)

  def _info(self, success):
    info = super()._info(success)
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))
    info['env_version'] = ENV_VERSION
    info['in_goal_region'] = bool(in_goal_region(x, y))
    info['goal_region_step'] = self._goal_region_step
    return info

  def step(self, action):
    # V6.step with the point-distance reward replaced by the region test.
    # It is copied rather than wrapped because the success latch must be set
    # BEFORE the contact check, exactly where V6 sets it.
    if self._failed:
      return self._flatten(self._last_obs), 0.0, True, self._info(False)

    for zone in (1, 2):
      if not self._active[zone] or self._passed[zone]:
        continue
      since = self._t - self._t0[zone]
      if since >= self.rockfall_steps:
        self._passed[zone] = True
        if self._dropped[zone]:
          self._park_rocks(zone)
      elif since >= 0 and since % v6.WAVE_PERIOD == 0:
        self._drop_rocks(zone)

    obs, _, _, _ = OfflineD4rlAntUMazeEnv.step(self, action)
    self._t += 1
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))
    reward = float(in_goal_region(x, y))
    if reward > 0:
      self._succeeded = True
      if self._goal_region_step is None:
        self._goal_region_step = int(self._t)

    for zone in (1, 2):
      if self._in_zone(zone, x, y):
        if not self._entered[zone]:
          self._entry_step[zone] = int(self._t)
        self._entered[zone] = True
        self._route = 'shortcut'
      if (self._mouth_step[zone] is None and self._route != 'detour'
          and self._at_mouth(zone, x, y)):
        self._mouth_step[zone] = int(self._t)
    if self._route is None and y >= v6.DETOUR_Y and x < v6.DETOUR_MAX_X:
      self._route = 'detour'

    contact_zone = self._contact_zone()
    for zone in (1, 2):
      self._contact[zone] = contact_zone == zone
    if contact_zone is not None and not self._succeeded:
      self._failure_zone = int(contact_zone)
      self._failed = True
      if self.death_settle_substeps > 0:
        obs = self._settle_after_death()
      return obs, 0.0, True, self._info(False)
    return obs, reward, False, self._info(self._succeeded)


def geometry_metadata():
  """V6's geometry block with the goal region added."""
  meta = v6.geometry_metadata()
  meta.update({
      'goal_region': {'x': list(GOAL_REGION_X), 'y': list(GOAL_REGION_Y)},
      'goal_region_center_xy': list(GOAL_REGION_CENTER_XY),
      'goal_noise': GOAL_NOISE,
      'success': 'torso xy inside goal_region (east column, rows 1-3)',
  })
  return meta
