"""Long two-rockfall AntMaze V6 benchmark.

V6 is an isolated successor to :mod:`crl.rockfall_clock_v5`; V5 is not
modified.  The topology is still deliberately binary: a straight shortcut
from the start to the goal and one longer, safe perimeter detour.  Two
spatially separated physical rockfall zones lie on the shortcut.

The three episode-level coins used by the benchmark are intentionally kept
separate.  This module owns two reset-time hazard coins::

    U1 ~ Bernoulli(p_active_1)
    U2 ~ Bernoulli(p_active_2)

using distinct RNG streams.  The teacher's independent detour coin lives in
``scripts/rockfall_clock_v6_teacher.py``.  Each zone also has its own reset-
time absolute clock and its own rock-jitter stream.  Every stream is consumed
on every reset even when a test forces a value, which makes paired and forced
audits reproducible without changing later draws.

The learner contract remains V5's: state is the 29 Ant coordinates only and
the default goal is the 29-column zero-padded task goal (58 flattened
columns).  The eight rock bodies, both latents, and both timetables are never
included in the observation.  Privileged properties and ``info`` fields are
for teachers and audits only.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from crl.d4rl_ant import (G, R, SCALING, OfflineD4rlAntUMazeEnv,
                          build_maze_xml)
from crl.rockfall_ant import NQ_ANT, NV_ANT
from crl.tworoute_rockfall_ant import (
    ROCK_DENSITY, ROCK_DROP_LEAD, ROCK_DROP_OFFSETS, ROCK_DROP_VZ,
    ROCK_JITTER, ROCK_RADII, ROCK_RGBA, _TwoRouteRockSim,
    _body_in_subtree)

# A 5 x 9 rectangular ring.  ``0`` is traversable but is not a goal-sampling
# cell.  There are exactly two start-to-goal paths: the straight row-1
# shortcut and the row-3 perimeter detour.
LONG_TWO_ROCKFALL_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ENV_VERSION = 'rockfall_clock_v6_long_two_rockfall'
MAP_ROWS = len(LONG_TWO_ROCKFALL_MAZE)
MAP_COLS = len(LONG_TWO_ROCKFALL_MAZE[0])
START_CELL = (1, 1)
GOAL_CELL = (1, 7)
START_XY = (0.0, 0.0)
GOAL_CELL_XY = (24.0, 0.0)

#: Hazard-coin densities. Raised from 0.35 to 0.40 for the failure-negative
#: round; nothing was tuned against a result, because no V6 training run
#: existed when this changed (artifacts/rockfall_clock_v6/runs was empty).
#: The 0.35 artifacts are kept: their dataset is
#: artifacts/rockfall_clock_v6/dataset/antmaze_rockfall_clock_v6{,_gxy}.npz and
#: reproducing them needs the values passed explicitly
#: (--p-active-1 0.35 --p-active-2 0.35 --npz <that file>), because every
#: default here now describes the 0.40 benchmark.
#: At 0.40 the four latent cells sit at P(neither) 0.360, P(exactly one) 0.480,
#: P(both) 0.160, against 0.423 / 0.455 / 0.122 at 0.35.
P_ACTIVE_1 = 0.40
P_ACTIVE_2 = 0.40

# Interior bands of two different shortcut cells, with the same 0.6-unit
# junction margin used by V5.  Their closest edges are 5.2 world units apart.
HAZARD_X = {1: (6.6, 9.4), 2: (14.6, 17.4)}
HAZARD_HALF_Y = 2.0
MOUTH_X = {1: 5.4, 2: 13.4}
AIM_X = {1: (7.0, 8.3), 2: (15.0, 16.3)}
ROCK_X_MAX = {1: HAZARD_X[1][1], 2: HAZARD_X[2][1]}

# Route label: reaching the north part of the west column commits to detour.
DETOUR_Y = 6.0
DETOUR_MAX_X = 2.0

# Six physical waves over 72 steps, exactly as in V5, but independently per
# zone.  The natural ranges are deliberately different because zone 2 is
# farther from the start.  The timing audit script measures arrivals and can
# override all four endpoints; these defaults are calibrated to the frozen
# shortcut driver (see artifacts/rockfall_clock_v6/timing_audit.json).
ROCKFALL_STEPS = 72
WAVE_PERIOD = 12
T0_MIN_1, T0_MAX_1 = 10, 45
T0_MIN_2, T0_MAX_2 = 90, 125

# Separate deterministic streams.  None overlaps the Ant reset/goal streams
# inherited from d4rl_ant.py or V5's stream offsets.
_ACTIVE_SEED_OFFSET_1 = 191_211
_ACTIVE_SEED_OFFSET_2 = 291_211
_SCHED_SEED_OFFSET_1 = 177_419
_SCHED_SEED_OFFSET_2 = 277_419
_JITTER_SEED_OFFSET_1 = 155_803
_JITTER_SEED_OFFSET_2 = 255_803

ROCKS_PER_ZONE = len(ROCK_RADII)
ROCK_STORE_X = {1: -30.0, 2: -40.0}
ROCK_STORE_DY = 2.0
ROCK_REACH_DIST = 0.6

# Same headline goal projection used by the vanilla V5 baseline.
GOAL_INDICES_XY = (0, 1)


def hazard_zone(zone):
  """Return ``(x0, x1, y0, y1)`` for hazard ``zone`` (1 or 2)."""
  if zone not in (1, 2):
    raise ValueError(f'zone must be 1 or 2, got {zone!r}')
  return (*HAZARD_X[zone], -HAZARD_HALF_Y, HAZARD_HALF_Y)


def build_long_two_rockfall_xml():
  """Build the elongated maze with a disjoint four-rock set per zone."""
  xml, offset = build_maze_xml(LONG_TWO_ROCKFALL_MAZE)
  root = ET.fromstring(xml)
  worldbody = root.find('.//worldbody')
  for zone in (1, 2):
    for k, radius in enumerate(ROCK_RADII):
      body = ET.SubElement(
          worldbody, 'body', name=f'rock_z{zone}_{k}',
          pos=f'{ROCK_STORE_X[zone]} {k * ROCK_STORE_DY} {radius}')
      ET.SubElement(body, 'freejoint', name=f'rockjoint_z{zone}_{k}')
      ET.SubElement(
          body, 'geom', name=f'rockgeom_z{zone}_{k}', type='sphere',
          size=f'{radius}', density=f'{ROCK_DENSITY}', contype='1',
          conaffinity='1', material='', rgba=ROCK_RGBA)
  return ET.tostring(root, encoding='unicode'), offset


class RockfallClockV6Env(OfflineD4rlAntUMazeEnv):
  """Elongated two-route AntMaze with two independent absolute rock clocks."""

  def __init__(self, max_episode_steps=800, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl',
               p_active_1=P_ACTIVE_1, p_active_2=P_ACTIVE_2,
               t0_min_1=T0_MIN_1, t0_max_1=T0_MAX_1,
               t0_min_2=T0_MIN_2, t0_max_2=T0_MAX_2,
               death_settle_substeps=0):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    self.seed = int(seed)
    self._reset_index = -1
    #: DEATH-SETTLE (the AntMaze rock-death observability convention, ported
    #: from crl/rockfall_ant.py; 0 = legacy freeze-at-contact and is
    #: byte-identical to every existing V6 run and to the frozen dataset).
    #: >0: once the fatal contact is flagged the actor loses control -- ctrl
    #: is zeroed, no further action is applied -- and MuJoCo physics advances
    #: this many EXTRA substeps inside the same env.step, so the fatal
    #: transition ends in a settled post-impact ant state instead of the
    #: mid-stride snapshot. The learner observation stays the ordinary 29-dim
    #: ant slice; nothing rock-, latent- or death-related is exposed and no
    #: observation entry is written by hand -- the signature must come out of
    #: the physics. Terminal semantics are unchanged: the episode stays
    #: absorbing and later steps return the frozen (now settled) _last_obs.
    #:
    #: Difference from crl/rockfall_ant.py, which is inherent to V6 (and to
    #: V2-V5): that env detects contact per substep and can cut the frame_skip
    #: short, while V6 flags contact only after the whole env step, so the
    #: settle begins at the end of the fatal step rather than inside it. The
    #: definition of the stored state -- the observation returned by the fatal
    #: transition, after N ctrl-free substeps -- is the same one
    #: scripts/rebuild_failure_bank_settled.py uses.
    #:
    #: This is a DATASET/BANK-construction knob. Training and evaluation runs
    #: leave it at 0.
    self.death_settle_substeps = int(death_settle_substeps)
    if self.death_settle_substeps < 0:
      raise ValueError('death_settle_substeps must be >= 0, got '
                       f'{death_settle_substeps!r}')
    for name, value in (('p_active_1', p_active_1),
                        ('p_active_2', p_active_2)):
      if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {value}')
    for zone, lo, hi in ((1, t0_min_1, t0_max_1),
                         (2, t0_min_2, t0_max_2)):
      if int(lo) > int(hi):
        raise ValueError(f'zone {zone} t0 minimum exceeds maximum: {lo}>{hi}')

    xml, offset = build_long_two_rockfall_xml()
    self._env = _TwoRouteRockSim(xml, seed)
    self._env.full_reset = True
    self._torso_offset = offset
    self._eval_goal_cell_xy = self._cell_xy(GOAL_CELL)
    self._open, self._goal_cells = [], []
    for row in range(MAP_ROWS):
      for col in range(MAP_COLS):
        cell = LONG_TWO_ROCKFALL_MAZE[row][col]
        if cell in (R, G, 0):
          self._open.append((row, col))
        if cell == G:
          self._goal_cells.append((row, col))

    model = self._env.model
    expected_nq = NQ_ANT + 2 * ROCKS_PER_ZONE * 7
    assert model.nq == expected_nq, (model.nq, expected_nq)
    self._rock_qadr = {1: [], 2: []}
    self._rock_vadr = {1: [], 2: []}
    rock_gids = {1: [], 2: []}
    for zone in (1, 2):
      for k in range(ROCKS_PER_ZONE):
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f'rockjoint_z{zone}_{k}')
        self._rock_qadr[zone].append(int(model.jnt_qposadr[joint]))
        self._rock_vadr[zone].append(int(model.jnt_dofadr[joint]))
        rock_gids[zone].append(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f'rockgeom_z{zone}_{k}'))
    self._rock_gids = {z: frozenset(gs) for z, gs in rock_gids.items()}
    self._all_rock_gids = self._rock_gids[1] | self._rock_gids[2]
    ant_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    self._ant_gids = frozenset(
        gid for gid in range(model.ngeom)
        if _body_in_subtree(model, model.geom_bodyid[gid], ant_body))

    self.p_active_1 = float(p_active_1)
    self.p_active_2 = float(p_active_2)
    self.t0_ranges = {1: (int(t0_min_1), int(t0_max_1)),
                      2: (int(t0_min_2), int(t0_max_2))}
    self._active_rng = {
        1: np.random.default_rng(seed + _ACTIVE_SEED_OFFSET_1),
        2: np.random.default_rng(seed + _ACTIVE_SEED_OFFSET_2),
    }
    self._sched_rng = {
        1: np.random.default_rng(seed + _SCHED_SEED_OFFSET_1),
        2: np.random.default_rng(seed + _SCHED_SEED_OFFSET_2),
    }
    self._jitter_rng = {
        1: np.random.default_rng(seed + _JITTER_SEED_OFFSET_1),
        2: np.random.default_rng(seed + _JITTER_SEED_OFFSET_2),
    }
    self._drop_jitter = {z: np.zeros((ROCKS_PER_ZONE, 2)) for z in (1, 2)}
    self._active = {1: False, 2: False}
    self._t0 = {1: 0, 2: 0}
    self._waves = {1: 0, 2: 0}
    self._passed = {1: False, 2: False}
    self._dropped = {1: False, 2: False}
    self._contact = {1: False, 2: False}
    self._mouth_step = {1: None, 2: None}
    self._entry_step = {1: None, 2: None}
    self._entered = {1: False, 2: False}
    self._failed = False
    self._succeeded = False
    self._failure_zone = None
    self._route = None
    self._t = 0

  # ---- privileged teacher/audit channel; none of this enters _flatten ----
  @property
  def privileged_rockfall_active_1(self):
    return bool(self._active[1])

  @property
  def privileged_rockfall_active_2(self):
    return bool(self._active[2])

  def privileged_rockfall_start(self, zone):
    return int(self._t0[zone]) if self._active[zone] else None

  def privileged_sampled_start(self, zone):
    """Raw reset-time t0 draw, including inactive episodes (audit only)."""
    if zone not in (1, 2):
      raise ValueError(f'zone must be 1 or 2, got {zone!r}')
    return int(self._t0[zone])

  def privileged_rockfall_end(self, zone):
    return (int(self._t0[zone]) + ROCKFALL_STEPS
            if self._active[zone] else None)

  @property
  def schedule(self):
    """Complete timetable visible only to the sighted teacher."""
    return {
        't': int(self._t),
        'zones': {
            z: {'active': bool(self._active[z]),
                'sampled_start': self.privileged_sampled_start(z),
                'start': self.privileged_rockfall_start(z),
                'end': self.privileged_rockfall_end(z)}
            for z in (1, 2)
        },
    }

  @property
  def dead(self):
    return bool(self._failed)

  def rock_ant_distance(self, zone):
    if not self._dropped[zone]:
      return None
    model, data = self._env.model, self._env.data
    best = float('inf')
    for rock_gid in self._rock_gids[zone]:
      for ant_gid in self._ant_gids:
        dist = float(mujoco.mj_geomDistance(
            model, data, int(rock_gid), int(ant_gid), best, None))
        best = min(best, dist)
    return best

  def rock_within_reach(self, zone):
    dist = self.rock_ant_distance(zone)
    return bool(dist is not None and dist < ROCK_REACH_DIST)

  # ---- geometry -----------------------------------------------------------
  @staticmethod
  def _in_zone(zone, x, y):
    x0, x1 = HAZARD_X[zone]
    return x0 <= x <= x1 and abs(y) < HAZARD_HALF_Y

  @staticmethod
  def _at_mouth(zone, x, y):
    # An upper bound prevents zone 1 from relatching near zone 2.
    return (MOUTH_X[zone] <= x < HAZARD_X[zone][0]
            and abs(y) < HAZARD_HALF_Y)

  # ---- lifecycle ----------------------------------------------------------
  def reset(self, rockfall_active_1=None, rockfall_active_2=None,
            rockfall_start_1=None, rockfall_start_2=None):
    """Reset and sample two independent hazards and absolute schedules.

    Overrides are audit-only.  All six hidden RNG streams are consumed in a
    fixed order even when values are forced.
    """
    self._reset_index += 1
    active_overrides = {1: rockfall_active_1, 2: rockfall_active_2}
    start_overrides = {1: rockfall_start_1, 2: rockfall_start_2}
    probs = {1: self.p_active_1, 2: self.p_active_2}
    for zone in (1, 2):
      drawn_active = bool(self._active_rng[zone].random() < probs[zone])
      override = active_overrides[zone]
      self._active[zone] = (drawn_active if override is None
                            else bool(override))
      lo, hi = self.t0_ranges[zone]
      drawn_start = int(self._sched_rng[zone].integers(lo, hi + 1))
      self._t0[zone] = (drawn_start if start_overrides[zone] is None
                        else int(start_overrides[zone]))
      self._drop_jitter[zone] = self._jitter_rng[zone].uniform(
          -ROCK_JITTER, ROCK_JITTER, size=(ROCKS_PER_ZONE, 2))
      self._waves[zone] = 0
      self._passed[zone] = False
      self._dropped[zone] = False
      self._contact[zone] = False
      self._mouth_step[zone] = None
      self._entry_step[zone] = None
      self._entered[zone] = False
    self._failed = False
    self._succeeded = False
    self._failure_zone = None
    self._route = None
    self._t = 0
    return super().reset()

  def _info(self, success):
    info = {
        'env_version': ENV_VERSION,
        'env_seed': self.seed,
        'reset_index': int(self._reset_index),
        't': int(self._t),
        'success': bool(success),
        'failure': bool(self._failed),
        'dead': bool(self._failed),
        'failure_zone': self._failure_zone,
        'route': self._route,
        'entered_hazard': bool(any(self._entered.values())),
    }
    for zone in (1, 2):
      info.update({
          f'rockfall_active_{zone}': bool(self._active[zone]),
          f'rockfall_sampled_start_{zone}': self.privileged_sampled_start(zone),
          f'rockfall_start_{zone}': self.privileged_rockfall_start(zone),
          f'rockfall_end_{zone}': self.privileged_rockfall_end(zone),
          f'rockfall_open_{zone}': bool(
              self._active[zone] and self._dropped[zone]
              and not self._passed[zone]),
          f'rockfall_passed_{zone}': bool(self._passed[zone]),
          f'rock_dropped_{zone}': bool(self._dropped[zone]),
          f'rock_contact_{zone}': bool(self._contact[zone]),
          f'rock_waves_{zone}': int(self._waves[zone]),
          f'mouth_step_{zone}': self._mouth_step[zone],
          f'band_entry_step_{zone}': self._entry_step[zone],
          f'entered_hazard_{zone}': bool(self._entered[zone]),
      })
    return info

  # ---- physical rocks -----------------------------------------------------
  def _drop_rocks(self, zone):
    data = self._env.data
    x, y = float(data.qpos[0]), float(data.qpos[1])
    x0, x1 = HAZARD_X[zone]
    if (MOUTH_X[zone] <= x <= x1 + 1.0
        and abs(y) < HAZARD_HALF_Y):
      lead = ROCK_DROP_LEAD * float(np.clip(data.qvel[0], 0.0, 2.0))
      base = float(np.clip(x + lead, *AIM_X[zone]))
      y_ref = y
    else:
      base, y_ref = float(AIM_X[zone][0]), 0.0
    for k, (across, along, height) in enumerate(ROCK_DROP_OFFSETS):
      jitter_x, jitter_y = self._drop_jitter[zone][k]
      qadr, vadr = self._rock_qadr[zone][k], self._rock_vadr[zone][k]
      data.qpos[qadr] = np.clip(
          base + along + jitter_x, AIM_X[zone][0], ROCK_X_MAX[zone])
      data.qpos[qadr + 1] = np.clip(
          y_ref + across + jitter_y, -1.75, 1.75)
      data.qpos[qadr + 2] = height
      data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
      data.qvel[vadr:vadr + 6] = 0.0
      data.qvel[vadr + 2] = -ROCK_DROP_VZ
    self._dropped[zone] = True
    self._waves[zone] += 1

  def _park_rocks(self, zone):
    data, home = self._env.data, self._env._home_qpos
    for qadr, vadr in zip(self._rock_qadr[zone], self._rock_vadr[zone]):
      data.qpos[qadr:qadr + 7] = home[qadr:qadr + 7]
      data.qvel[vadr:vadr + 6] = 0.0
    self._dropped[zone] = False

  def _contact_zone(self):
    data = self._env.data
    active_gids = set()
    for zone in (1, 2):
      if self._dropped[zone]:
        active_gids.update(self._rock_gids[zone])
    if not active_gids:
      return None
    for idx in range(data.ncon):
      g1, g2 = data.contact[idx].geom1, data.contact[idx].geom2
      if ((g1 in active_gids and g2 in self._ant_gids)
          or (g2 in active_gids and g1 in self._ant_gids)):
        rock_gid = g1 if g1 in active_gids else g2
        return 1 if rock_gid in self._rock_gids[1] else 2
    return None

  def step(self, action):
    if self._failed:
      return self._flatten(self._last_obs), 0.0, True, self._info(False)

    # Both absolute clocks advance before physics and never consult the Ant.
    for zone in (1, 2):
      if not self._active[zone] or self._passed[zone]:
        continue
      since = self._t - self._t0[zone]
      if since >= ROCKFALL_STEPS:
        self._passed[zone] = True
        if self._dropped[zone]:
          self._park_rocks(zone)
      elif since >= 0 and since % WAVE_PERIOD == 0:
        self._drop_rocks(zone)

    obs, reward, _, _ = OfflineD4rlAntUMazeEnv.step(self, action)
    self._t += 1
    if reward > 0:
      self._succeeded = True
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))

    for zone in (1, 2):
      if self._in_zone(zone, x, y):
        if not self._entered[zone]:
          self._entry_step[zone] = int(self._t)
        self._entered[zone] = True
        self._route = 'shortcut'
      if (self._mouth_step[zone] is None and self._route != 'detour'
          and self._at_mouth(zone, x, y)):
        self._mouth_step[zone] = int(self._t)
    if self._route is None and y >= DETOUR_Y and x < DETOUR_MAX_X:
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
    return obs, float(reward), False, self._info(self._succeeded)

  def _settle_after_death(self):
    """Ctrl-free physics after the fatal contact; returns the settled obs.

    The actor gets no further decision. Only the environment's own physics
    (gravity, the dropped rocks, contacts) runs, exactly as in
    crl/rockfall_ant.py's death settle. Nothing about the observation
    contract changes.
    """
    sim = self._env
    sim.data.ctrl[:] = 0.0
    for _ in range(self.death_settle_substeps):
      mujoco.mj_step(sim.model, sim.data)
    self._last_obs = sim._obs_dict()
    return self._flatten(self._last_obs)


def geometry_metadata():
  """Machine-readable geometry block shared by collectors and reports."""
  return {
      'map_cells': [MAP_ROWS, MAP_COLS],
      'cell_size': SCALING,
      'outer_world_bounds': {'x': [-6.0, 30.0], 'y': [-6.0, 14.0]},
      'start_cell': list(START_CELL), 'start_xy': list(START_XY),
      'goal_cell': list(GOAL_CELL), 'goal_cell_xy': list(GOAL_CELL_XY),
      'shortcut': 'row 1: x=0 -> 24 at |y|<2',
      'detour': 'west column north to y=8, east to x=24, south to goal',
      'hazard_zone_1': list(hazard_zone(1)),
      'hazard_zone_2': list(hazard_zone(2)),
      'mouth_x': {'1': MOUTH_X[1], '2': MOUTH_X[2]},
      'rocks_per_zone': ROCKS_PER_ZONE,
  }
