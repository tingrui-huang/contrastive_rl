"""Rockfall corridor: hidden 4-bit hazard map ("blind ant under unstable
cliffs"). Confounded variant of the OFFLINE antmaze-umaze task.

This is a NEW variant, fully separate from the frozen litter benchmark
(crl/d4rl_ant.py LitterOfflineAntUMazeEnv): nothing here touches litter
geometry, collapse rules, datasets or the frozen walker.

Story: the bottom corridor runs under two crumbling rock shelves (one over
each side lane). Each shelf has two localized weak spots ("sites"), two per
side at different x. At reset an episode-specific hazard map

    rockfall_mask[i] ~ Bernoulli(P_ACTIVE)   i in (left_1, left_2,
                                                   right_1, right_2)

is drawn and FIXED for the episode. The mask is hidden: the learner obs is
the unchanged 58-dim proprioception + zero-padded goal; the mask lives only
in privileged info/sidecar. It is a 4-bit hazard map, NOT reducible to one
left/right bit (a side can have 0, 1 or 2 active sites independently).

Hiddenness by construction:
  * debris rocks are free-joint bodies stored in contact-free free fall far
    BELOW the floor plane (a MuJoCo plane only collides from above), so
    before a drop the physics/constraint system is byte-identical for every
    mask -- active and inactive sites differ by nothing but a Python bit;
  * the learner obs slices ant dofs only (qpos[:15], qvel[:14]) so rock
    states never enter the observation;
  * every random draw (mask, per-site severity, drop jitter, impaired legs)
    happens at reset from dedicated rng streams -- a trigger consumes no
    randomness, so paired resets with forced different masks evolve
    identically until debris physically appears.

Trigger: when the ant torso enters a site's local region (|x - site_x| <=
TRIG_HALF_X and y in the site's side band), the site fires ONCE. Inactive
site: nothing happens (the flag flips for bookkeeping only). Active site:
its 3 rocks teleport above the ant (small presampled jitter + velocity
lead) and fall. Outcome severity was presampled at reset:

  * severe   -- first rock-ant contact buries the ant: absorbing death,
                physics frozen, reward can never fire (litter freeze
                pattern);
  * impaired -- first rock-ant contact cripples IMPAIR_LEGS presampled legs
                for the rest of the episode (their actuator commands are
                scaled by impair_gear_scale; optionally joint damping is
                multiplied) -- the ant stays alive and free to move;
  * mild     -- physical impulse only: the rocks shove the ant and remain
                on the floor as passive obstacles; recovery is expected.

The ant is never lane-locked: no extra walls, lateral escape is always
physically possible.

Routes (pilot/teacher convention, not part of the env): left lane
(y=+LANE), right lane (y=-LANE), or the CENTER route -- a static
always-present rough strip (visible for every mask, so it leaks nothing)
makes the center slower/rougher than the flat side lanes, and the center
band avoids every trigger region.

All probabilities and impact parameters are constructor-configurable pilot
values, NOT frozen constants.
"""
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from crl.d4rl_ant import (CTRL_SCALE, INIT_QPOS, SUCCESS_DIST, U_MAZE,
                          OfflineD4rlAntUMazeEnv, _Sim, build_maze_xml)

# --------------------------------------------------------------------------
# Geometry (bottom corridor: x in ~[0, 8], walls at |y| = 2, handoff at x=6)
# --------------------------------------------------------------------------
#: (name, x_center, side_sign): two sites per side lane, staggered in x.
#: All trigger windows end by x=5.5: staggering/failing ants wander
#: laterally in the corridor's last stretch (x ~5.5-6, pre-turn), and site
#: windows there caught them (observed center-route leakage cluster).
ROCKFALL_SITES = (('left_1', 3.0, +1.0), ('left_2', 4.3, +1.0),
                  ('right_1', 3.6, -1.0), ('right_2', 4.9, -1.0))
SITE_LANE_Y = 1.05            #: |y| of the site center (the side lanes)
TRIG_HALF_X = 0.6             #: trigger when |x - site_x| <= this ...
TRIG_Y_BAND = (1.0, 2.0)      #: ... and sign*y inside this band: the side
                              #: lanes track y~1.1+-0.2 and dwell there, so
                              #: they trigger reliably, while ridge-deflected
                              #: center excursions (ridge ends at |y|=0.82)
                              #: stay out of every trigger region
TRIG_DWELL = 3                #: consecutive steps inside the region needed
                              #: to fire: a rubble-deflected stumble does
                              #: not disturb the shelf, lane travel does.
                              #: 5 broke side-lane reliability (hit_frac
                              #: 1.0 -> 0.62: lane y oscillates 1.1+-0.25
                              #: and dips below the band line too often
                              #: for 5 consecutive in-band steps).
TRIG_MIN_VX = 0.1             #: dwell accrues only while marching forward:
                              #: staggering/wedged ants (the only way the
                              #: center route ever reaches a shelf band)
                              #: drift backward or stall and never fire it.
                              #: Creeping a lane below 0.1 to dodge shelves
                              #: costs ~300+ steps for the zone alone --
                              #: strictly worse than the center route.
P_ACTIVE = 0.2                #: default Bernoulli(0.2) per site (pilot)
SEVERITY_PROBS = (0.55, 0.30, 0.15)       #: severe / impaired / mild (pilot)
SEVERITIES = ('severe', 'impaired', 'mild')

#: 3 rocks per site: (dx, dy_toward_wall, z). dy is expressed for the +y
#: side and mirrored for the -y side. First rock aims AT the torso, the
#: other two land immediately around it.
ROCK_DROP_OFFSETS = ((0.10, 0.00, 1.5), (0.45, 0.30, 1.9),
                     (-0.30, -0.25, 2.3))
ROCK_RADII = (0.17, 0.14, 0.12)
ROCK_DENSITY = 65.0           #: ant geoms use density 5 (ant ~1 kg total);
                              #: 65 -> rocks ~0.3-1.3 kg (pilot-tuned)
ROCK_DROP_VZ = 6.0            #: initial downward speed of dropped rocks
ROCK_DROP_LEAD = 0.25         #: aim ahead by lead * clip(vx, 0, 2)
ROCK_JITTER = 0.08            #: presampled per-episode xy jitter per rock
ROCK_RGBA = '0.45 0.44 0.48 1.0'
#: storage: parked resting on the floor far outside the maze (the floor
#: plane is an infinite half-space, so "buried" storage would register
#: permanent deep-penetration contacts). Parked-rock contacts are static,
#: identical for every mask, and excluded from debris bookkeeping (only
#: DROPPED sites' rocks count as rockfall contacts).
ROCK_STORE_X0, ROCK_STORE_DX = -30.0, 2.0

#: Center rough strip: static, ALWAYS present (mask-independent -> leaks
#: nothing), makes the center route slower/rougher than the flat side lanes.
#: Center mud ("meltwater bog fed by the canyon runoff"): a U-INDEPENDENT
#: viscous drag region on the corridor centerline. Blocking terrain was
#: RETIRED as the center performance mechanism after 24 tuning passes:
#: the frozen walker crosses any passable static field at 0.85+ when
#: careful, and every blocking variant either got end-run or interfered
#: with the side lanes. Drag is the one cost the walker cannot finesse:
#: a horizontal viscous force F = -MUD_DRAG * c(x, y) * v_xy applied to
#: the torso root, with c = 1 inside |y| <= MUD_CORE_Y tapering linearly
#: to 0 at |y| = MUD_EDGE_Y (side lanes at |y|~1.1 feel nothing). No
#: geometry, no impacts, no lateral ejection, identical for every mask.
MUD_X = (2.4, 5.6)
MUD_CORE_Y = 0.5
MUD_EDGE_Y = 1.0
MUD_DRAG = 40.0                 #: N per (m/s); pilot-tuned against the
                                #: 700-step horizon: 0.783/0.783 on two
                                #: independent seed groups (n=60 each),
                                #: timeout-dominated failures, zero
                                #: triggers. k=35 -> 0.90 (too light),
                                #: k=50 -> 0.0 (past the timeout cliff).
#: light VISUAL scatter (contype/conaffinity 0: renders as debris, never
#: collides) so the mud region reads as a rubble-strewn bog on camera.
CENTER_SCATTER_N = 14
CENTER_SCATTER_HALF_XY = (0.08, 0.16)
CENTER_SCATTER_H = (0.03, 0.07)
CENTER_SCATTER_EDGE = 0.45
ROCKFALL_LAYOUT_SEED = 11     #: center-strip layout rng (frozen per build)

IMPAIR_LEGS = 2               #: legs crippled by an 'impaired' hit (pilot)
IMPAIR_GEAR_SCALE = 0.15      #: ctrl multiplier on the crippled legs
IMPAIR_DAMPING_MULT = 3.5     #: >1 also multiplies their joint damping

NQ_ANT, NV_ANT = 15, 14       #: ant dofs come first (ant body precedes rocks)


def rockfall_config():
  """Echo of every tunable (for reports/sidecars)."""
  return {'sites': [{'name': n, 'x': x, 'side': s}
                    for n, x, s in ROCKFALL_SITES],
          'site_lane_y': SITE_LANE_Y, 'trig_half_x': TRIG_HALF_X,
          'trig_y_band': list(TRIG_Y_BAND), 'trig_dwell': TRIG_DWELL,
          'trig_min_vx': TRIG_MIN_VX, 'p_active': P_ACTIVE,
          'severity_probs': list(SEVERITY_PROBS),
          'rock_drop_offsets': [list(o) for o in ROCK_DROP_OFFSETS],
          'rock_radii': list(ROCK_RADII), 'rock_density': ROCK_DENSITY,
          'rock_drop_vz': ROCK_DROP_VZ, 'rock_drop_lead': ROCK_DROP_LEAD,
          'rock_jitter': ROCK_JITTER,
          'center_mud': {'x': list(MUD_X), 'core_y': MUD_CORE_Y,
                         'edge_y': MUD_EDGE_Y, 'drag': MUD_DRAG},
          'center_scatter_visual_only': {
              'n': CENTER_SCATTER_N,
              'half_xy': list(CENTER_SCATTER_HALF_XY),
              'h': list(CENTER_SCATTER_H), 'edge': CENTER_SCATTER_EDGE,
              'layout_seed': ROCKFALL_LAYOUT_SEED},
          'impair_legs': IMPAIR_LEGS,
          'impair_gear_scale': IMPAIR_GEAR_SCALE,
          'impair_damping_mult': IMPAIR_DAMPING_MULT}


def build_rockfall_xml(rock_density=ROCK_DENSITY):
  """Maze xml + center rough strip (static) + 12 stored rock bodies.

  Rocks are free-joint spheres parked far below the floor plane; the center
  strip comes from ROCKFALL_LAYOUT_SEED and is identical for every episode
  and mask. Returns (xml_string, torso_offset).
  """
  xml, offset = build_maze_xml(U_MAZE)
  root = ET.fromstring(xml)
  worldbody = root.find('.//worldbody')
  rng = np.random.default_rng(ROCKFALL_LAYOUT_SEED)
  # mud region: VISUAL-ONLY dark plate + scatter (contype/conaffinity 0:
  # nothing here collides; the drag itself is a force in step()).
  mx0, mx1 = MUD_X
  ET.SubElement(worldbody, 'geom', name='mudplate_visual', type='box',
                pos=f'{(mx0 + mx1) / 2} 0 0.006',
                size=f'{(mx1 - mx0) / 2} {MUD_EDGE_Y} 0.006',
                contype='0', conaffinity='0', material='',
                rgba='0.28 0.22 0.16 0.85')
  for i in range(CENTER_SCATTER_N):
    sx, sy = rng.uniform(*CENTER_SCATTER_HALF_XY, size=2)
    sy = min(sy, CENTER_SCATTER_EDGE - 0.02)
    bx = rng.uniform(mx0 + sx, mx1 - sx)
    by = rng.uniform(-(CENTER_SCATTER_EDGE - sy), CENTER_SCATTER_EDGE - sy)
    bh = rng.uniform(*CENTER_SCATTER_H)
    ET.SubElement(worldbody, 'geom', name=f'centerscatter_{i}', type='box',
                  pos=f'{bx} {by} {bh / 2}', size=f'{sx} {sy} {bh / 2}',
                  contype='0', conaffinity='0', material='',
                  rgba='0.45 0.4 0.33 1.0')
  k = 0
  for name, _, _ in ROCKFALL_SITES:
    for r in ROCK_RADII:
      body = ET.SubElement(
          worldbody, 'body', name=f'rock_{name}_{k % len(ROCK_RADII)}',
          pos=f'{ROCK_STORE_X0} {ROCK_STORE_X0 + k * ROCK_STORE_DX} {r}')
      ET.SubElement(body, 'freejoint', name=f'rockjoint_{name}_'
                                            f'{k % len(ROCK_RADII)}')
      ET.SubElement(body, 'geom', name=f'rockgeom_{name}_'
                                       f'{k % len(ROCK_RADII)}',
                    type='sphere', size=f'{r}', density=f'{rock_density}',
                    contype='1', conaffinity='1', material='',
                    rgba=ROCK_RGBA)
      k += 1
  return ET.tostring(root, encoding='unicode'), offset


class _RockSim(_Sim):
  """_Sim whose obs and reset touch ONLY the ant dofs (rocks stay hidden)."""

  def __init__(self, xml, seed):
    super().__init__(xml, seed)
    self._home_qpos = np.asarray(self.model.qpos0).copy()

  def reset_model(self):
    self.data.qpos[:] = self._home_qpos        # rocks at storage, quat id
    self.data.qvel[:] = 0.0
    self.data.qpos[:NQ_ANT] = (INIT_QPOS
                               + self._rng.uniform(-0.1, 0.1, NQ_ANT))
    self.data.qvel[:NV_ANT] = self._rng.standard_normal(NV_ANT) * 0.1
    mujoco.mj_forward(self.model, self.data)

  def _obs_dict(self):
    qpos = np.asarray(self.data.qpos)
    qvel = np.asarray(self.data.qvel)
    return {'achieved_goal': qpos[:2].copy(),
            'observation': np.concatenate([qpos[2:NQ_ANT], qvel[:NV_ANT]]),
            'desired_goal': np.asarray(self.goal, float).copy()}


class RockfallOfflineAntUMazeEnv(OfflineD4rlAntUMazeEnv):
  """Confounded rockfall variant of the offline umaze task (see module doc).

  Learner obs contract is IDENTICAL to offline_ant_umaze (58-dim, mask
  invisible). Teacher code reads ``privileged_mask``; step() reports
  diagnostics in the info dict -- sidecar only, NEVER the observation.
  """

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl',
               p_active=P_ACTIVE, severity_probs=SEVERITY_PROBS,
               impair_legs=IMPAIR_LEGS,
               impair_gear_scale=IMPAIR_GEAR_SCALE,
               impair_damping_mult=IMPAIR_DAMPING_MULT,
               rock_density=ROCK_DENSITY, mud_drag=MUD_DRAG):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    xml, offset = build_rockfall_xml(rock_density=rock_density)
    self._env = _RockSim(xml, seed)      # replace sim with the rockfall model
    self._torso_offset = offset
    m = self._env.model
    assert m.nq == NQ_ANT + 7 * len(ROCKFALL_SITES) * len(ROCK_RADII)
    assert m.nu == 8

    self.p_active = float(p_active)
    self.mud_drag = float(mud_drag)
    self.severity_probs = tuple(float(p) for p in severity_probs)
    assert abs(sum(self.severity_probs) - 1.0) < 1e-9
    self.impair_legs = int(impair_legs)
    self.impair_gear_scale = float(impair_gear_scale)
    self.impair_damping_mult = float(impair_damping_mult)

    self.site_names = tuple(n for n, _, _ in ROCKFALL_SITES)
    self._site_x = np.array([x for _, x, _ in ROCKFALL_SITES])
    self._site_sign = np.array([s for _, _, s in ROCKFALL_SITES])
    #: per-site rock joint addresses
    self._rock_qadr, self._rock_vadr, self._rock_gids = [], [], []
    ant_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    self._ant_gids = frozenset(
        g for g in range(m.ngeom)
        if self._body_in_subtree(m, m.geom_bodyid[g], ant_body))
    for name in self.site_names:
      qadr, vadr, gids = [], [], []
      for k in range(len(ROCK_RADII)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                              f'rockjoint_{name}_{k}')
        qadr.append(int(m.jnt_qposadr[j]))
        vadr.append(int(m.jnt_dofadr[j]))
        gids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                      f'rockgeom_{name}_{k}'))
      self._rock_qadr.append(qadr)
      self._rock_vadr.append(vadr)
      self._rock_gids.append(frozenset(gids))
    self._all_rock_gids = frozenset().union(*self._rock_gids)
    #: leg -> actuator indices (motor order: hip_4 ankle_4 hip_1 ankle_1
    #: hip_2 ankle_2 hip_3 ankle_3) and -> dof addresses, via names.
    self._leg_acts, self._leg_dofs = [], []
    for leg in (1, 2, 3, 4):
      acts, dofs = [], []
      for jn in (f'hip_{leg}', f'ankle_{leg}'):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dofs.append(int(m.jnt_dofadr[j]))
        a = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j)
        acts.append(a)
      self._leg_acts.append(acts)
      self._leg_dofs.append(dofs)
    self._orig_damping = np.asarray(m.dof_damping).copy()

    #: independent rng streams: mask / (severity, jitter, impaired legs)
    self._mask_rng = np.random.default_rng(seed + 41_007)
    self._sev_rng = np.random.default_rng(seed + 90_210)
    self.rockfall_mask = (0, 0, 0, 0)
    self._severity = ['mild'] * 4
    self._begin_episode()

  @staticmethod
  def _body_in_subtree(m, body, root_body):
    while body > 0:
      if body == root_body:
        return True
      body = m.body_parentid[body]
    return False

  # ---- privileged / probe surface -----------------------------------------
  @property
  def dead(self):
    """Buried by a severe rockfall: absorbing until episode end."""
    return self._dead

  @property
  def privileged_mask(self):
    """Teacher-only 4-bit hazard map. Learners must never read this."""
    return self.rockfall_mask

  @property
  def privileged_severity(self):
    """Teacher/analysis-only presampled per-site severities."""
    return tuple(self._severity)

  def site_regions(self):
    """Trigger geometry for probes: list of (name, x0, x1, y0, y1)."""
    out = []
    for i, name in enumerate(self.site_names):
      lo = self._site_sign[i] * TRIG_Y_BAND[0]
      hi = self._site_sign[i] * TRIG_Y_BAND[1]
      out.append((name, float(self._site_x[i] - TRIG_HALF_X),
                  float(self._site_x[i] + TRIG_HALF_X),
                  float(min(lo, hi)), float(max(lo, hi))))
    return out

  # ---- episode bookkeeping -------------------------------------------------
  def _begin_episode(self, mask=None, severities=None):
    """Presample ALL episode randomness; restore impairment; clear flags.

    Draw order is FIXED and mask-independent so paired probes that force
    different masks keep byte-identical downstream rng state.
    """
    m = self._env.model
    if self.impair_damping_mult != 1.0:
      m.dof_damping[:] = self._orig_damping
    drawn = tuple(int(self._mask_rng.random() < self.p_active)
                  for _ in range(4))
    self.rockfall_mask = (drawn if mask is None
                          else tuple(int(b) for b in mask))
    assert len(self.rockfall_mask) == 4
    sev = [SEVERITIES[int(self._sev_rng.choice(3, p=self.severity_probs))]
           for _ in range(4)]
    self._severity = (list(sev) if severities is None
                      else [str(s) for s in severities])
    self._impair_leg_draw = [
        self._sev_rng.choice(4, size=self.impair_legs, replace=False)
        for _ in range(4)]
    self._drop_jitter = self._sev_rng.uniform(
        -ROCK_JITTER, ROCK_JITTER, size=(4, len(ROCK_RADII), 2))
    self._triggered = [False] * 4
    self._dwell = [0] * 4
    self._dropped = [False] * 4
    self._hit = [False] * 4              # first rock-ant contact consumed
    self._dead = False
    self._impaired_acts = np.zeros(0, int)
    self._impaired_leg_ids = []
    self._drop_step = {}
    self._hit_step = {}
    self._t = 0

  def reset(self, mask=None, severities=None):
    """``mask``/``severities`` overrides are for probes/gates/GIFs only;
    normal use samples both."""
    self._begin_episode(mask=mask, severities=severities)
    return super().reset()               # _RockSim.reset_model parks rocks

  # ---- rockfall mechanics ---------------------------------------------------
  def _in_region(self, i, x, y):
    return (abs(x - self._site_x[i]) <= TRIG_HALF_X
            and TRIG_Y_BAND[0] <= self._site_sign[i] * y <= TRIG_Y_BAND[1])

  @staticmethod
  def _mud_coverage(x, y):
    """Drag coverage in [0, 1]: 1 on the centerline core, linear falloff
    to 0 at MUD_EDGE_Y. Position-only -> identical for every mask."""
    if not MUD_X[0] <= x <= MUD_X[1]:
      return 0.0
    ay = abs(y)
    if ay <= MUD_CORE_Y:
      return 1.0
    if ay >= MUD_EDGE_Y:
      return 0.0
    return (MUD_EDGE_Y - ay) / (MUD_EDGE_Y - MUD_CORE_Y)

  def _drop_site(self, i):
    d = self._env.data
    x, y = float(d.qpos[0]), float(d.qpos[1])
    lead = ROCK_DROP_LEAD * min(max(float(d.qvel[0]), 0.0), 2.0)
    for k, (dx, dy, z) in enumerate(ROCK_DROP_OFFSETS):
      jx, jy = self._drop_jitter[i][k]
      qa, va = self._rock_qadr[i][k], self._rock_vadr[i][k]
      d.qpos[qa] = np.clip(x + lead + dx + jx, 0.6, 7.4)
      d.qpos[qa + 1] = np.clip(y + self._site_sign[i] * dy + jy,
                               -1.75, 1.75)
      d.qpos[qa + 2] = z
      d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      d.qvel[va:va + 6] = 0.0
      d.qvel[va + 2] = -ROCK_DROP_VZ
    self._dropped[i] = True
    self._drop_step[self.site_names[i]] = self._t

  def _apply_impairment(self, i):
    legs = [int(l) for l in self._impair_leg_draw[i]]
    self._impaired_leg_ids = sorted(set(self._impaired_leg_ids) | set(legs))
    acts = sorted({a for leg in self._impaired_leg_ids
                   for a in self._leg_acts[leg]})
    self._impaired_acts = np.asarray(acts, int)
    if self.impair_damping_mult != 1.0:
      m = self._env.model
      for leg in legs:
        for dof in self._leg_dofs[leg]:
          m.dof_damping[dof] = (self._orig_damping[dof]
                                * self.impair_damping_mult)

  def _rock_contacts(self):
    """(sites whose DROPPED rocks touch the ant, any-debris-contact flag).

    Only rocks of already-dropped sites count: parked storage rocks rest on
    the floor far outside the maze with static, mask-independent contacts
    that are not rockfall events."""
    d = self._env.data
    dropped_gids = [g for i in range(4) if self._dropped[i]
                    for g in self._rock_gids[i]]
    if not dropped_gids:
      return set(), False
    dropped_gids = frozenset(dropped_gids)
    hit_sites, any_rock = set(), False
    for c in range(d.ncon):
      g1, g2 = d.contact[c].geom1, d.contact[c].geom2
      r = g1 if g1 in dropped_gids else (
          g2 if g2 in dropped_gids else None)
      if r is None:
        continue
      any_rock = True
      other = g2 if r == g1 else g1
      if other in self._ant_gids:
        for i in range(4):
          if self._dropped[i] and r in self._rock_gids[i]:
            hit_sites.add(i)
    return hit_sites, any_rock

  def _info(self, extra):
    out = {'rockfall_mask': self.rockfall_mask,
           'severity': tuple(self._severity),
           'triggered': tuple(self._triggered),
           'dropped': tuple(self._dropped),
           'hit': tuple(self._hit),
           'dead': self._dead,
           'impaired_legs': tuple(self._impaired_leg_ids),
           'drop_steps': dict(self._drop_step),
           'hit_steps': dict(self._hit_step)}
    out.update(extra)
    return out

  def step(self, action):
    self._t += 1
    if self._dead:                       # buried: frozen until episode end
      return (self._flatten(self._last_obs), 0.0, False,
              self._info({'rock_ant_contact': False,
                          'rock_any_contact': False,
                          'mud_coverage': 0.0}))
    u = self._env
    x, y = float(u.data.qpos[0]), float(u.data.qpos[1])
    marching = float(u.data.qvel[0]) > TRIG_MIN_VX
    for i in range(4):                   # trigger check BEFORE physics
      if self._triggered[i]:
        continue
      self._dwell[i] = (self._dwell[i] + 1
                        if marching and self._in_region(i, x, y) else 0)
      if self._dwell[i] >= TRIG_DWELL:
        self._triggered[i] = True        # inactive site: flag only,
        if self.rockfall_mask[i]:        # zero physical difference
          self._drop_site(i)
    a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
    if len(self._impaired_acts):
      a[self._impaired_acts] *= self.impair_gear_scale
    u.data.ctrl[:] = a * CTRL_SCALE
    ant_hit_now, any_rock_now = False, False
    mud_c = 0.0
    for _ in range(u.frame_skip):        # substep checks: frame_skip must
      c = self._mud_coverage(float(u.data.qpos[0]), float(u.data.qpos[1]))
      mud_c = max(mud_c, c)
      u.data.qfrc_applied[0] = -self.mud_drag * c * float(u.data.qvel[0])
      u.data.qfrc_applied[1] = -self.mud_drag * c * float(u.data.qvel[1])
      mujoco.mj_step(u.model, u.data)    # not swallow a brief impact
      hit_sites, any_rock = self._rock_contacts()
      any_rock_now = any_rock_now or any_rock
      for i in hit_sites:
        ant_hit_now = True
        if not self._dropped[i] or self._hit[i]:
          continue
        self._hit[i] = True
        self._hit_step[self.site_names[i]] = self._t
        if self._severity[i] == 'severe':
          self._dead = True              # buried under the rockfall
        elif self._severity[i] == 'impaired':
          self._apply_impairment(i)
    self._last_obs = u._obs_dict()
    if self._dead:
      reward = 0.0
    else:
      dist = float(np.linalg.norm(np.asarray(self._last_obs['achieved_goal'])
                                  - np.asarray(u.goal)))
      reward = float(dist <= SUCCESS_DIST)
    return (self._flatten(self._last_obs), reward, False,
            self._info({'rock_ant_contact': ant_hit_now,
                        'rock_any_contact': any_rock_now,
                        'mud_coverage': mud_c}))
