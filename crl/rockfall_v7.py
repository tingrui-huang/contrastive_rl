"""Three-trap probabilistic-rockfall AntMaze V7 benchmark.

V7 is an isolated successor to :mod:`crl.rockfall_clock_v6`; V6 is not
modified and neither are any of its artifacts.  The map, the 58-dim learner
contract, the two-route topology, the rock bodies and the physics are V6's.
Two things change: there are now THREE hazard zones on the shortcut instead
of two, and the hazard is no longer a clock.

TOPOLOGY.  The same 5 x 9 ring as V6: a straight row-1 shortcut from the
start cell (1, 1) at the world origin to the goal cell (1, 7) at x = 24, and
one longer, safe row-3 perimeter detour.  The three traps occupy the
interiors of three shortcut cells with V6's 0.6-unit junction margin, so each
band is 2.8 x 4.0 world units -- exactly V6's hazard size::

    zone 1  "start"   cell (1, 2)   x in [ 2.6,  5.4]   |y| < 2
    zone 2  "center"  cell (1, 4)   x in [10.6, 13.4]   |y| < 2
    zone 3  "end"     cell (1, 6)   x in [18.6, 21.4]   |y| < 2

Zone 1 is the first cell after the start and zone 3 the last cell before the
goal, so neither the reset pose nor the goal region lies inside a trap, and
the detour (which runs through the west and east columns at x = 0 and
x = 24) cannot brush any band.

HAZARD.  V6's absolute rock clocks are gone.  Each trap is an independent
coin::

    U1 ~ Bernoulli(p_kill_1)
    U2 ~ Bernoulli(p_kill_2)
    U3 ~ Bernoulli(p_kill_3)

If the Ant never enters trap ``z``, ``Uz`` has no physical consequence
whatsoever -- no rock of that zone ever moves.  The first time the Ant enters
trap ``z``, that zone's coin is REVEALED: on heads the four rocks of the zone
are launched onto the Ant and the episode ends in death; on tails nothing
happens, ever, and the Ant walks through.  So the contract is exactly

    P(death | the Ant enters trap z) = p_kill_z,

and taking the whole shortcut survives with probability
``(1 - p1)(1 - p2)(1 - p3)`` = 0.216 at the default 0.40 per zone.

WHY THE COINS ARE DRAWN AT RESET.  They are sampled in ``reset`` from three
dedicated RNG streams, not at the moment of entry.  This is observationally
identical -- nothing about a coin reaches the simulator, the observation or
the ``info`` dict before the trap is entered -- but it keeps V6's audit
convention: every hidden stream is consumed on every reset in a fixed order
even when a test forces a value, so paired and forced probes stay
reproducible without disturbing later draws.  It also makes the episode's
hazard an honest exogenous latent, drawn before the Ant moves, which is what
the privileged teacher conditions its route choice on.

THE KILL IS CERTAIN, AND IT IS PHYSICAL.  On heads the zone drops a fresh
wave of four heavy spheres aimed at the Ant (V6's velocity-lead aim and
per-episode presampled jitter) every ``WAVE_PERIOD`` steps, up to
``MAX_KILL_WAVES``, and the episode is flagged dead on the first rock-Ant
contact -- so the fatal transition carries a real impact signature rather
than a hand-written flag.  The barrage is aimed straight down at a walking
Ant from ~3 units up and the first wave hits in practice; the deadline at
``KILL_DEADLINE`` steps after entry exists only so that
``P(death | entry) = p`` is exact rather than approximate, and an episode
that reaches it is reported with ``kill_by_deadline`` set so the smoke gate
can assert the rate is zero.

Waiting cannot help in V7, and that is the point: V6 asked the sighted
teacher WHEN to cross, V7 asks it WHETHER to.  The only safe response to a
hot trap is the detour.

The learner contract is unchanged from V5/V6: state is the 29 Ant
coordinates only, the default goal is the 29-column zero-padded task goal
(58 flattened columns), and the twelve rock bodies and the three latents are
never in the observation.  Privileged properties and ``info`` fields are for
teachers and audits only.
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

#: The V6 map, unchanged.  ``0`` is traversable but is not a goal-sampling
#: cell.  Exactly two start-to-goal paths: the row-1 shortcut and the row-3
#: perimeter detour.
THREE_TRAP_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ENV_VERSION = 'rockfall_v7_three_trap'
MAP_ROWS = len(THREE_TRAP_MAZE)
MAP_COLS = len(THREE_TRAP_MAZE[0])
START_CELL = (1, 1)
GOAL_CELL = (1, 7)
START_XY = (0.0, 0.0)
GOAL_CELL_XY = (24.0, 0.0)

ZONES = (1, 2, 3)
#: Human-readable position of each trap along the shortcut.
ZONE_LABELS = {1: 'start', 2: 'center', 3: 'end'}

#: Per-trap kill probability.  0.40 is V6's hazard-coin density, carried over
#: so that a single zone is as hot as a V6 zone; the three-zone shortcut is
#: therefore clear only 0.6^3 = 0.216 of the time.  The eight latent cells
#: sit at P(none hot) 0.216, P(exactly one) 0.432, P(exactly two) 0.288,
#: P(all three) 0.064.  All three are constructor arguments and every
#: consumer should pass them explicitly rather than rely on this default.
P_KILL_1 = 0.40
P_KILL_2 = 0.40
P_KILL_3 = 0.40

#: Interiors of three shortcut cells (centres x = 4, 12, 20) with V6's
#: 0.6-unit junction margin.  Each band is 2.8 wide -- V6's hazard size.
HAZARD_X = {1: (2.6, 5.4), 2: (10.6, 13.4), 3: (18.6, 21.4)}
HAZARD_HALF_Y = 2.0
#: Approach strip, 1.2 units west of each band: where a sighted teacher makes
#: its decision.  The strip of zone z+1 never overlaps the band of zone z.
MOUTH_X = {z: HAZARD_X[z][0] - 1.2 for z in ZONES}
#: Rock aim window inside each band, V6's offsets from the band centre.
AIM_X = {z: (0.5 * (lo + hi) - 1.0, 0.5 * (lo + hi) + 0.3)
         for z, (lo, hi) in HAZARD_X.items()}
ROCK_X_MAX = {z: HAZARD_X[z][1] for z in ZONES}

# Route label: reaching the north part of the west column commits to detour.
DETOUR_Y = 6.0
DETOUR_MAX_X = 2.0

#: The lethal barrage.  ``WAVE_PERIOD`` is V6's and is one full fall time at
#: the release heights in ROCK_DROP_OFFSETS, so consecutive waves do not
#: overlap in flight.  KILL_DEADLINE closes the contract (see module doc).
WAVE_PERIOD = 12
MAX_KILL_WAVES = 4
KILL_DEADLINE = WAVE_PERIOD * MAX_KILL_WAVES

#: Separate deterministic streams.  None overlaps the Ant reset/goal streams
#: inherited from d4rl_ant.py, nor V5's or V6's offsets.
_KILL_SEED_OFFSET = {1: 391_211, 2: 491_211, 3: 591_211}
_JITTER_SEED_OFFSET = {1: 355_803, 2: 455_803, 3: 555_803}

ROCKS_PER_ZONE = len(ROCK_RADII)
ROCK_STORE_X = {1: -30.0, 2: -40.0, 3: -50.0}
ROCK_STORE_DY = 2.0
ROCK_REACH_DIST = 0.6

# Same headline goal projection used by the vanilla V5/V6 baselines.
GOAL_INDICES_XY = (0, 1)


def hazard_zone(zone):
  """Return ``(x0, x1, y0, y1)`` for hazard ``zone`` (1, 2 or 3)."""
  if zone not in ZONES:
    raise ValueError(f'zone must be one of {ZONES}, got {zone!r}')
  return (*HAZARD_X[zone], -HAZARD_HALF_Y, HAZARD_HALF_Y)


def build_three_trap_xml():
  """Build the V6 maze with a disjoint four-rock set per trap."""
  xml, offset = build_maze_xml(THREE_TRAP_MAZE)
  root = ET.fromstring(xml)
  worldbody = root.find('.//worldbody')
  for zone in ZONES:
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


class RockfallV7Env(OfflineD4rlAntUMazeEnv):
  """Two-route AntMaze with three independent probabilistic death traps."""

  def __init__(self, max_episode_steps=800, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl',
               p_kill_1=P_KILL_1, p_kill_2=P_KILL_2, p_kill_3=P_KILL_3,
               death_settle_substeps=0):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    self.seed = int(seed)
    self._reset_index = -1
    #: DEATH-SETTLE (the AntMaze rock-death observability convention, ported
    #: unchanged from V6; 0 = legacy freeze-at-contact).  >0: once the fatal
    #: contact is flagged the actor loses control -- ctrl is zeroed, no
    #: further action is applied -- and MuJoCo advances this many EXTRA
    #: substeps inside the same env.step, so the fatal transition ends in a
    #: settled post-impact Ant state instead of a mid-stride snapshot.  The
    #: learner observation stays the ordinary 29-dim Ant slice; nothing
    #: rock- or latent-related is exposed and no observation entry is
    #: written by hand -- the signature must come out of the physics.
    #: Terminal semantics are unchanged: the episode stays absorbing and
    #: later steps return the frozen (now settled) _last_obs.  This is a
    #: DATASET/BANK-construction knob; training and evaluation leave it at 0.
    self.death_settle_substeps = int(death_settle_substeps)
    if self.death_settle_substeps < 0:
      raise ValueError('death_settle_substeps must be >= 0, got '
                       f'{death_settle_substeps!r}')
    probs = {1: p_kill_1, 2: p_kill_2, 3: p_kill_3}
    for zone, value in probs.items():
      if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f'p_kill_{zone} must be in [0, 1], got {value}')

    xml, offset = build_three_trap_xml()
    self._env = _TwoRouteRockSim(xml, seed)
    self._env.full_reset = True
    self._torso_offset = offset
    self._eval_goal_cell_xy = self._cell_xy(GOAL_CELL)
    self._open, self._goal_cells = [], []
    for row in range(MAP_ROWS):
      for col in range(MAP_COLS):
        cell = THREE_TRAP_MAZE[row][col]
        if cell in (R, G, 0):
          self._open.append((row, col))
        if cell == G:
          self._goal_cells.append((row, col))

    model = self._env.model
    expected_nq = NQ_ANT + len(ZONES) * ROCKS_PER_ZONE * 7
    assert model.nq == expected_nq, (model.nq, expected_nq)
    self._rock_qadr = {z: [] for z in ZONES}
    self._rock_vadr = {z: [] for z in ZONES}
    rock_gids = {z: [] for z in ZONES}
    for zone in ZONES:
      for k in range(ROCKS_PER_ZONE):
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f'rockjoint_z{zone}_{k}')
        self._rock_qadr[zone].append(int(model.jnt_qposadr[joint]))
        self._rock_vadr[zone].append(int(model.jnt_dofadr[joint]))
        rock_gids[zone].append(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f'rockgeom_z{zone}_{k}'))
    self._rock_gids = {z: frozenset(gs) for z, gs in rock_gids.items()}
    self._all_rock_gids = frozenset().union(*self._rock_gids.values())
    ant_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    self._ant_gids = frozenset(
        gid for gid in range(model.ngeom)
        if _body_in_subtree(model, model.geom_bodyid[gid], ant_body))

    self.p_kill = {z: float(probs[z]) for z in ZONES}
    self._kill_rng = {
        z: np.random.default_rng(seed + _KILL_SEED_OFFSET[z]) for z in ZONES}
    self._jitter_rng = {
        z: np.random.default_rng(seed + _JITTER_SEED_OFFSET[z])
        for z in ZONES}
    self._drop_jitter = {z: np.zeros((ROCKS_PER_ZONE, 2)) for z in ZONES}
    self._lethal = {z: False for z in ZONES}
    self._waves = {z: 0 for z in ZONES}
    self._dropped = {z: False for z in ZONES}
    self._contact = {z: False for z in ZONES}
    self._mouth_step = {z: None for z in ZONES}
    self._entry_step = {z: None for z in ZONES}
    self._entered = {z: False for z in ZONES}
    self._revealed = {z: False for z in ZONES}
    self._failed = False
    self._succeeded = False
    self._failure_zone = None
    self._kill_by_deadline = False
    self._route = None
    self._t = 0

  # ---- privileged teacher/audit channel; none of this enters _flatten ----
  def privileged_trap_lethal(self, zone):
    """Reset-time coin of ``zone``: True iff entering it kills."""
    if zone not in ZONES:
      raise ValueError(f'zone must be one of {ZONES}, got {zone!r}')
    return bool(self._lethal[zone])

  @property
  def privileged_trap_lethal_1(self):
    return bool(self._lethal[1])

  @property
  def privileged_trap_lethal_2(self):
    return bool(self._lethal[2])

  @property
  def privileged_trap_lethal_3(self):
    return bool(self._lethal[3])

  @property
  def latent(self):
    """``'U010'``-style label of the three reset-time coins."""
    return 'U' + ''.join(str(int(self._lethal[z])) for z in ZONES)

  @property
  def hazards(self):
    """Complete trap table visible only to the sighted teacher.

    Mirrors V6's ``schedule`` side channel: one privileged dict, read by the
    teacher and the audits, never written into the learner observation.
    """
    return {
        't': int(self._t),
        'zones': {
            z: {'lethal': bool(self._lethal[z]),
                'p_kill': float(self.p_kill[z]),
                'label': ZONE_LABELS[z],
                'entered': bool(self._entered[z]),
                'revealed': bool(self._revealed[z])}
            for z in ZONES
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
    # An upper bound stops a zone relatching in front of the next trap.
    return (MOUTH_X[zone] <= x < HAZARD_X[zone][0]
            and abs(y) < HAZARD_HALF_Y)

  # ---- lifecycle ----------------------------------------------------------
  def reset(self, trap_lethal_1=None, trap_lethal_2=None, trap_lethal_3=None):
    """Reset and sample the three independent trap coins.

    Overrides are audit-only.  All six hidden RNG streams are consumed in a
    fixed order even when values are forced.
    """
    self._reset_index += 1
    overrides = {1: trap_lethal_1, 2: trap_lethal_2, 3: trap_lethal_3}
    for zone in ZONES:
      drawn = bool(self._kill_rng[zone].random() < self.p_kill[zone])
      override = overrides[zone]
      self._lethal[zone] = drawn if override is None else bool(override)
      self._drop_jitter[zone] = self._jitter_rng[zone].uniform(
          -ROCK_JITTER, ROCK_JITTER, size=(ROCKS_PER_ZONE, 2))
      self._waves[zone] = 0
      self._dropped[zone] = False
      self._contact[zone] = False
      self._mouth_step[zone] = None
      self._entry_step[zone] = None
      self._entered[zone] = False
      self._revealed[zone] = False
    self._failed = False
    self._succeeded = False
    self._failure_zone = None
    self._kill_by_deadline = False
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
        'kill_by_deadline': bool(self._kill_by_deadline),
        'route': self._route,
        'latent': self.latent,
        'entered_hazard': bool(any(self._entered.values())),
    }
    for zone in ZONES:
      info.update({
          f'trap_lethal_{zone}': bool(self._lethal[zone]),
          f'trap_revealed_{zone}': bool(self._revealed[zone]),
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
    """Launch this zone's four spheres at the Ant (V6's aim, verbatim)."""
    data = self._env.data
    x, y = float(data.qpos[0]), float(data.qpos[1])
    x0, x1 = HAZARD_X[zone]
    if MOUTH_X[zone] <= x <= x1 + 1.0 and abs(y) < HAZARD_HALF_Y:
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
      data.qpos[qadr + 1] = np.clip(y_ref + across + jitter_y, -1.75, 1.75)
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
    for zone in ZONES:
      if self._dropped[zone]:
        active_gids.update(self._rock_gids[zone])
    if not active_gids:
      return None
    for idx in range(data.ncon):
      g1, g2 = data.contact[idx].geom1, data.contact[idx].geom2
      if ((g1 in active_gids and g2 in self._ant_gids)
          or (g2 in active_gids and g1 in self._ant_gids)):
        rock_gid = g1 if g1 in active_gids else g2
        for zone in ZONES:
          if rock_gid in self._rock_gids[zone]:
            return zone
    return None

  def step(self, action):
    if self._failed:
      return self._flatten(self._last_obs), 0.0, True, self._info(False)

    # The barrage of every revealed-lethal trap advances before physics.  A
    # trap that was never entered, and a trap whose coin came up tails, make
    # no call here at all -- their rocks stay parked for the whole episode.
    for zone in ZONES:
      if not self._revealed[zone] or not self._lethal[zone]:
        continue
      since = self._t - int(self._entry_step[zone])
      if (since >= 0 and since % WAVE_PERIOD == 0
          and self._waves[zone] < MAX_KILL_WAVES):
        self._drop_rocks(zone)

    obs, reward, _, _ = OfflineD4rlAntUMazeEnv.step(self, action)
    self._t += 1
    if reward > 0:
      self._succeeded = True
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))

    for zone in ZONES:
      if self._in_zone(zone, x, y):
        if not self._entered[zone]:
          self._entry_step[zone] = int(self._t)
          # The coin is revealed by the entry, and only by the entry.
          self._revealed[zone] = True
        self._entered[zone] = True
        self._route = 'shortcut'
      if (self._mouth_step[zone] is None and self._route != 'detour'
          and self._at_mouth(zone, x, y)):
        self._mouth_step[zone] = int(self._t)
    if self._route is None and y >= DETOUR_Y and x < DETOUR_MAX_X:
      self._route = 'detour'

    contact_zone = self._contact_zone()
    for zone in ZONES:
      self._contact[zone] = contact_zone == zone
    fatal_zone = contact_zone
    if fatal_zone is None:
      # Contract backstop: a revealed lethal trap kills, full stop.  The
      # barrage is aimed straight down at the Ant and lands first wave in
      # practice, so this branch is expected never to fire (the smoke gate
      # asserts a zero rate); it is here so that P(death | entry) is exactly
      # p_kill rather than p_kill times a hit rate.
      for zone in ZONES:
        if (self._revealed[zone] and self._lethal[zone]
            and self._t - int(self._entry_step[zone]) >= KILL_DEADLINE):
          fatal_zone = zone
          self._kill_by_deadline = True
          break
    if fatal_zone is not None and not self._succeeded:
      self._failure_zone = int(fatal_zone)
      self._failed = True
      if self.death_settle_substeps > 0:
        obs = self._settle_after_death()
      return obs, 0.0, True, self._info(False)
    self._kill_by_deadline = False
    return obs, float(reward), False, self._info(self._succeeded)

  def _settle_after_death(self):
    """Ctrl-free physics after the fatal contact; returns the settled obs.

    The actor gets no further decision.  Only the environment's own physics
    (gravity, the dropped rocks, contacts) runs, exactly as in V6's death
    settle.  Nothing about the observation contract changes.
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
      'zones': {str(z): {'label': ZONE_LABELS[z],
                         'band': list(hazard_zone(z)),
                         'mouth_x': MOUTH_X[z]} for z in ZONES},
      'rocks_per_zone': ROCKS_PER_ZONE,
  }
