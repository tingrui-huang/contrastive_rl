"""Two-route AntMaze with a latent rockfall hazard on the shortcut (V0).

Minimal runnable benchmark for the causal-transition line, built ON the
existing AntMaze stack (crl/d4rl_ant.py): same vendored d4rl ant physics,
same 58-dim learner obs contract, same offline goal protocol. The only
geometric change is the maze map: the U-maze's centre-left wall cell (2,1)
is OPENED, which creates a second route:

      1 1 1 1 1            SHORTCUT: R(0,0) -> straight +y through the
      1 R g g 1              opened cell -> goal (0,8).   ~9 world units.
      1 . 1 g 1            DETOUR:   R(0,0) -> +x to (8,0) -> +y to (8,8)
      1 G g g 1              -> -x to the goal.          ~24 world units.
      1 1 1 1 1            ('.' = the opened cell; corridors 4 units wide,
                            identical to every existing corridor.)

Latent hazard: at reset, ``rockfall_active ~ Bernoulli(p_active=0.30)`` is
drawn from a DEDICATED rng (drawn every episode in a fixed order even when a
probe forces the value, so paired probes keep byte-identical downstream rng
state -- same convention as rockfall_ant._begin_episode).

PHYSICAL ROCKFALL (V1). Four free-joint rock spheres are compiled into the
model, parked far outside the maze on the floor (static, latent-independent
contacts: the parked state is byte-identical for every latent value -- the
physical-hiddenness trick of rockfall_ant). When the ant ENTERS the hazard
band the trigger fires exactly once per episode; if the latent is ACTIVE the
rocks are teleported above the ant (per-episode presampled jitter, velocity-
lead aim) with a downward launch speed and FALL under gravity over the
following steps -- visibly, and heavy enough (density 65 vs the ant's 5) to
knock the ant around on impact. An inactive trigger sets a flag only: zero
physical difference. The FAILURE DETECTOR is rock-ant contact: the first
contact between a dropped rock and any ant geom ends the episode with
done=True and info['failure']=True, absorbing thereafter (physics frozen).
Crossing the band alone no longer fails anything -- a fast or lucky ant the
rocks miss survives the active shortcut. Dropped rocks stay in the corridor
as lethal rubble for the rest of the episode.

The hazard band is the interior of the opened cell with a 0.6-unit margin at
both mouths (x in (-2, 2), y in [2.6, 5.4]): a bottom-corridor ant wandering
to y slightly above 2, or a goal-row ant dipping below 6, cannot brush it,
while any actual traversal of the shortcut must cross the full band. The
detour never satisfies |x| < 2 with y in that band (its left leg runs in the
goal row, y >= 6). The safe detour is traversable regardless of the latent.

Observation contract: the learner obs is IDENTICAL to offline_ant_umaze
(58-dim, zero-padded XY goal); the latent is never written into it. Teacher /
oracle code reads the ``privileged_rockfall_active`` property (the
``privileged_*`` convention of rockfall_ant); step() info carries it too,
for debugging and evaluation only.

Success keeps the inherited convention: reward = (dist(xy, goal_xy) <= 0.5),
done stays False on success (fixed-length episode contract). done=True is
returned ONLY for the rockfall failure. Episode outcomes are EXCLUSIVE by
construction: success latches on the first rewarded step and suppresses any
later hazard entry (a goal-reached ant cannot retroactively die), failure is
absorbing, and an episode with neither is an ordinary timeout.

info['route'] labels the episode 'shortcut' the moment the hazard band is
entered (authoritative: a detour never touches the band, so failure always
implies route=='shortcut'), else 'detour' once x >= 6 (the right column),
else None; info['entered_hazard'] is the raw band-entry flag. Rock
diagnostics: info['rock_triggered'] (trigger fired, either latent),
info['rock_dropped'] (rocks physically launched: active episodes only),
info['rock_contact'] (a dropped rock touches the ant this step).
"""
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from crl.d4rl_ant import (_Sim, OfflineD4rlAntUMazeEnv, build_maze_xml,
                          INIT_QPOS, R, G, SCALING)
from crl.rockfall_ant import NQ_ANT, NV_ANT

#: U-maze with the centre-left wall cell (2,1) opened ('0' = open, never a
#: reset/goal-sample cell). Everything else is byte-identical to U_MAZE.
TWO_ROUTE_MAZE = [[1, 1, 1, 1, 1],
                  [1, R, G, G, 1],
                  [1, 0, 1, G, 1],
                  [1, G, G, G, 1],
                  [1, 1, 1, 1, 1]]

#: latent hazard density (the exogenous confounder draw).
P_ACTIVE = 0.30
#: hazard band = interior of the opened cell, world frame (R cell at origin).
#: The cell spans x in [-2, 2], y in [2, 6]; the margin keeps junction wander
#: out (see module doc).
HAZARD_HALF_X = 2.0
HAZARD_Y = (2.6, 5.4)
#: crossing into the right column commits the episode to the detour route.
DETOUR_X = 6.0
#: rng stream offset for the latent draw (disjoint from _Sim's seed and the
#: goal rng's seed+777).
_ACTIVE_SEED_OFFSET = 91_211
#: rng stream offset for the per-episode rock-drop jitter (fixed draw order,
#: consumed every episode regardless of latent/override -- paired-probe safe).
_JITTER_SEED_OFFSET = 55_803

#: -------- physical rocks (mechanism ported from crl/rockfall_ant.py) -------
#: drop pattern relative to the ant at trigger time: (dx, dy, z). dy is along
#: the ant's travel direction through the band (sign of vy), so the pattern
#: works for either crossing direction; dx spans the 4-wide corridor.
ROCK_DROP_OFFSETS = ((0.00, 0.35, 2.6), (-0.55, 0.70, 3.0),
                     (0.55, 0.70, 2.8), (0.00, 1.05, 3.4))
ROCK_RADII = (0.17, 0.15, 0.13, 0.15)
ROCK_DENSITY = 65.0            #: ant geoms use density 5 -> rocks hit hard
ROCK_DROP_VZ = 2.0             #: initial downward speed at release
#: (release ~2.6-3.4 high at vz=2 -> ~0.5-0.7 s of VISIBLE fall,
#: impact speed ~7-8 m/s)
ROCK_DROP_LEAD = 0.25          #: aim ahead by lead * clip(|vy|, 0, 2)
ROCK_JITTER = 0.08             #: presampled per-episode xy jitter per rock
ROCK_RGBA = '0.45 0.44 0.48 1.0'
#: parked storage far outside the maze, resting ON the floor plane --
#: identical static contacts for every episode and latent value.
ROCK_STORE_X0, ROCK_STORE_DY = -30.0, 2.0


def build_tworoute_rockfall_xml():
  """Two-route maze xml + the parked free-joint rock bodies."""
  xml, offset = build_maze_xml(TWO_ROUTE_MAZE)
  root = ET.fromstring(xml)
  wb = root.find('.//worldbody')
  for k, r in enumerate(ROCK_RADII):
    body = ET.SubElement(wb, 'body', name=f'rock_{k}',
                         pos=f'{ROCK_STORE_X0} {k * ROCK_STORE_DY} {r}')
    ET.SubElement(body, 'freejoint', name=f'rockjoint_{k}')
    ET.SubElement(body, 'geom', name=f'rockgeom_{k}', type='sphere',
                  size=f'{r}', density=f'{ROCK_DENSITY}', contype='1',
                  conaffinity='1', material='', rgba=ROCK_RGBA)
  return ET.tostring(root, encoding='unicode'), offset


class _TwoRouteRockSim(_Sim):
  """_Sim whose obs and reset touch ONLY the ant dofs (rocks stay hidden);
  the rockfall_ant._RockSim pattern."""

  def __init__(self, xml, seed):
    super().__init__(xml, seed)
    self._home_qpos = np.asarray(self.model.qpos0).copy()

  def reset_model(self):
    if self.full_reset:
      mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = self._home_qpos   # rocks back to storage
    self.data.qvel[:] = 0.0
    self.data.qpos[:NQ_ANT] = INIT_QPOS + self._rng.uniform(-0.1, 0.1, NQ_ANT)
    self.data.qvel[:NV_ANT] = self._rng.standard_normal(NV_ANT) * 0.1
    mujoco.mj_forward(self.model, self.data)

  def _obs_dict(self):
    qpos = np.asarray(self.data.qpos)
    qvel = np.asarray(self.data.qvel)
    return {'achieved_goal': qpos[:2].copy(),
            'observation': np.concatenate([qpos[2:NQ_ANT], qvel[:NV_ANT]]),
            'desired_goal': np.asarray(self.goal, float).copy()}


def hazard_zone():
  """(x0, x1, y0, y1) of the hazard band, for probes/renderers."""
  return (-HAZARD_HALF_X, HAZARD_HALF_X, HAZARD_Y[0], HAZARD_Y[1])


def _body_in_subtree(m, body, root_body):
  while body > 0:
    if body == root_body:
      return True
    body = int(m.body_parentid[body])
  return False


#: Rz(+90 deg) quat (w, x, y, z): initial yaw toward the shortcut corridor.
_Q_NORTH = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
#: heading-coin rng offset (learner-eval 'random' heading; latent-independent)
_HEADING_SEED_OFFSET = 33_407


def _quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class TwoRouteRockfallAntEnv(OfflineD4rlAntUMazeEnv):
  """Offline-contract two-route AntMaze with the latent shortcut hazard."""

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl', p_active=P_ACTIVE,
               default_heading=None):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    xml, offset = build_tworoute_rockfall_xml()
    self._env = _TwoRouteRockSim(xml, seed)  # two-route model + parked rocks
    self._torso_offset = offset          # R cell unchanged -> same world frame
    m = self._env.model
    assert m.nq == NQ_ANT + 7 * len(ROCK_RADII), m.nq
    #: rock dof addresses + geom ids; ant geom ids for the contact detector.
    self._rock_qadr, self._rock_vadr, self._rock_gids = [], [], []
    for k in range(len(ROCK_RADII)):
      j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'rockjoint_{k}')
      self._rock_qadr.append(int(m.jnt_qposadr[j]))
      self._rock_vadr.append(int(m.jnt_dofadr[j]))
      self._rock_gids.append(
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'rockgeom_{k}'))
    self._rock_gids = frozenset(self._rock_gids)
    ant_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    self._ant_gids = frozenset(
        g for g in range(m.ngeom)
        if _body_in_subtree(m, m.geom_bodyid[g], ant_body))
    #: NEW benchmark, no legacy runs to stay byte-identical to -> use the
    #: canonical episode-independent reset from day one (the legacy reset
    #: leaks solver warmstart/qacc/ctrl across episodes; see _Sim.full_reset
    #: and scripts/test_reset_independence.py).
    self._env.full_reset = True
    #: rebuild the cell lists the base class derived from U_MAZE (they feed
    #: the ONLINE task's non_zero_reset only, but keep them correct anyway).
    self._open, self._goal_cells = [], []
    for r in range(len(TWO_ROUTE_MAZE)):
      for c in range(len(TWO_ROUTE_MAZE[0])):
        if TWO_ROUTE_MAZE[r][c] in (R, G, 0):
          self._open.append((r, c))
        if TWO_ROUTE_MAZE[r][c] == G:
          self._goal_cells.append((r, c))
    self.p_active = float(p_active)
    self._active_rng = np.random.default_rng(seed + _ACTIVE_SEED_OFFSET)
    self._jitter_rng = np.random.default_rng(seed + _JITTER_SEED_OFFSET)
    self._heading_rng = np.random.default_rng(seed + _HEADING_SEED_OFFSET)
    #: heading used by UNQUALIFIED reset() calls (trainer eval loops);
    #: None = native east, 'random' = the learner-eval coin protocol.
    self.default_heading = default_heading
    self._drop_jitter = np.zeros((len(ROCK_RADII), 2))
    self._rockfall_active = False
    self._entered_hazard = False
    self._failed = False
    self._succeeded = False
    self._route = None
    self._rock_triggered = False
    self._rock_dropped = False
    self._rock_contact = False

  # ---- teacher / oracle side channel (never in the learner obs) ------------
  @property
  def privileged_rockfall_active(self):
    """Teacher/analysis-only view of the episode's latent hazard state."""
    return bool(self._rockfall_active)

  @property
  def dead(self):
    """Repo-convention alias (rockfall/litter absorbing-death naming)."""
    return bool(self._failed)

  # ---- episode lifecycle ---------------------------------------------------
  def reset(self, rockfall_active=None, heading=None):
    """``rockfall_active`` override is for probes/gates/sanity checks only;
    normal use samples it. The rng is consumed in a fixed order regardless of
    the override, so forced-latent probes keep downstream draws identical.

    ``heading`` sets the ant's INITIAL yaw (an initial condition, drawn
    independently of the latent): None/'east' keeps the native d4rl pose
    (facing +x), 'north' yaws it +90 deg toward the shortcut corridor,
    'random' draws east/north 50/50 from a dedicated rng (the learner eval
    protocol; consumed every call in fixed order for reproducibility). The
    frozen corridor controllers cannot turn the ant in place, so route
    execution starts from the matching heading; route CHOICE is the
    teacher's policy and the heading is part of that action, correlated
    with the latent only through the teacher -- never through the env."""
    drawn = bool(self._active_rng.random() < self.p_active)
    self._rockfall_active = (drawn if rockfall_active is None
                             else bool(rockfall_active))
    #: presampled every episode in fixed order (latent/override independent)
    self._drop_jitter = self._jitter_rng.uniform(
        -ROCK_JITTER, ROCK_JITTER, size=(len(ROCK_RADII), 2))
    self._entered_hazard = False
    self._failed = False
    self._succeeded = False
    self._route = None
    self._rock_triggered = False
    self._rock_dropped = False
    self._rock_contact = False
    #: heading rng consumed every reset (fixed order), used only on 'random'
    coin = bool(self._heading_rng.random() < 0.5)
    obs = super().reset()
    if heading is None:
      heading = self.default_heading
    if heading == 'random':
      heading = 'north' if coin else 'east'
    if heading == 'north':
      d = self._env.data
      d.qpos[3:7] = _quat_mul(_Q_NORTH, np.asarray(d.qpos[3:7]).copy())
      d.qvel[:2] = 0.0
      d.qacc_warmstart[:] = 0.0
      mujoco.mj_forward(self._env.model, d)
      self._last_obs = self._env._obs_dict()
      obs = self._flatten(self._last_obs)
    return obs

  def _info(self, success):
    return {'rockfall_active': bool(self._rockfall_active),
            'entered_hazard': bool(self._entered_hazard),
            'rock_triggered': bool(self._rock_triggered),
            'rock_dropped': bool(self._rock_dropped),
            'rock_contact': bool(self._rock_contact),
            'failure': bool(self._failed),
            'dead': bool(self._failed),   # repo-convention alias
            'success': bool(success),
            'route': self._route}

  def _drop_rocks(self):
    """Teleport the parked rocks above the ant with a downward launch speed;
    gravity does the rest. Aim leads the ant along its band-crossing
    direction so either crossing direction meets the pattern."""
    d = self._env.data
    x, y = float(d.qpos[0]), float(d.qpos[1])
    vy = float(d.qvel[1])
    sgn = 1.0 if vy >= 0 else -1.0
    lead = ROCK_DROP_LEAD * min(abs(vy), 2.0) * sgn
    for k, (dx, dy, z) in enumerate(ROCK_DROP_OFFSETS):
      jx, jy = self._drop_jitter[k]
      qa, va = self._rock_qadr[k], self._rock_vadr[k]
      d.qpos[qa] = np.clip(x + dx + jx, -1.75, 1.75)
      d.qpos[qa + 1] = np.clip(y + lead + sgn * dy + jy, 2.3, 5.7)
      d.qpos[qa + 2] = z
      d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      d.qvel[va:va + 6] = 0.0
      d.qvel[va + 2] = -ROCK_DROP_VZ
    self._rock_dropped = True

  def _dropped_rock_contact(self):
    """True iff any DROPPED rock currently touches an ant geom. Parked rocks
    rest outside the maze and never count."""
    if not self._rock_dropped:
      return False
    d = self._env.data
    for c in range(d.ncon):
      g1, g2 = d.contact[c].geom1, d.contact[c].geom2
      if ((g1 in self._rock_gids and g2 in self._ant_gids)
          or (g2 in self._rock_gids and g1 in self._ant_gids)):
        return True
    return False

  def step(self, action):
    if self._failed:
      #: absorbing terminal failure: physics frozen, action ignored, reward
      #: can never fire (mirrors the repo's dead-state convention) -- but
      #: unlike the fixed-length envs we ALSO return done=True.
      return self._flatten(self._last_obs), 0.0, True, self._info(False)
    #: trigger check BEFORE physics (rockfall_ant convention): band entry
    #: fires the trigger exactly once; only an ACTIVE latent launches rocks,
    #: an inactive trigger is a flag with zero physical difference.
    d = self._env.data
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    if (not self._rock_triggered and abs(x0) < HAZARD_HALF_X
        and HAZARD_Y[0] <= y0 <= HAZARD_Y[1]):
      self._rock_triggered = True
      if self._rockfall_active:
        self._drop_rocks()
    obs, reward, _, _ = super().step(action)
    if reward > 0:
      #: success latches for the episode; a goal-reached ant that later
      #: wanders back into the band cannot retroactively 'die' -- the three
      #: episode outcomes (success / failure / neither) stay exclusive.
      self._succeeded = True
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))
    in_hazard = (abs(x) < HAZARD_HALF_X
                 and HAZARD_Y[0] <= y <= HAZARD_Y[1])
    if in_hazard:
      self._entered_hazard = True
      #: band entry is AUTHORITATIVE for the label (a detour episode can
      #: never touch the band, so failure => route=='shortcut' always holds,
      #: even for backtrackers that first crossed x >= DETOUR_X).
      self._route = 'shortcut'
    if self._route is None and x >= DETOUR_X:
      self._route = 'detour'
    self._rock_contact = self._dropped_rock_contact()
    if self._rock_contact and not self._succeeded:
      self._failed = True
      return obs, 0.0, True, self._info(False)
    return obs, float(reward), False, self._info(self._succeeded)
