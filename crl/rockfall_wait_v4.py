"""Rockfall AntMaze V4: one route, a rockfall that PASSES, and an expert
that waits it out.

V3 (crl/tworoute_rockfall_v3.py) made the latent-controlled decision a
ROUTE choice: the sighted teacher took the shortcut when clear and the
detour when active. That put two opposite gaits at the same (s, g) in the
data, and the single tanh-Gaussian actor averaged them into a blend that
neither chooses nor commits (see the V3 detour probe). V4 keeps the maze,
the rocks and the latent, and changes what the decision IS:

  * the rockfall is an EVENT with a duration, not a trap. Crossing the mouth
    line (x >= MOUTH_X in the bottom row, 1.2 units before the band) fires
    the trigger once; if the latent is active, rock waves fall into the
    band every WAVE_PERIOD steps for ROCKFALL_STEPS steps, then the rocks
    are parked again and the corridor is clear -- the rockfall has passed.
    An inactive trigger is a flag with zero physical difference (V2/V3
    hiddenness convention).
  * the expert always takes the same route (the straight BR shortcut, goal
    (8,0)); what the latent controls is WHETHER IT STOPS at the mouth. When
    clear it walks straight through; when active it holds (zero torque,
    crouched, measured drift 0.1-0.4 units) until the rockfall has passed,
    then walks. The u-blind learner sees one route and, at the mouth, two
    behaviours for the same (s, g): walk (~70%) and stand (~30%).

Why this is the cleaner instrument. In V3 the average of 'east' and 'north'
is a heading nobody demonstrated, so mode averaging landed the learner on a
knife edge and the outcome became reset-noise roulette. The average of
'walk' and 'stand' is a slower walk -- it still enters the band -- so mode
averaging now pushes the learner INTO the hazard instead of sideways; and
the critic, which only ever saw the shortcut taken when it was safe (the
waiting episodes reach the goal ~85 steps later), scores 'walk' above
'stand' at the mouth. Both mechanisms point the same way: a u-blind agent
should walk in and die at ~P(active), while the expert dies at 0.

Rock waves are aimed the V3 way (forward lead on vx, the pattern's travel
axis on x) but the aim point is clipped INTO the band, [AIM_X0, AIM_X1]
plus the pattern's along-travel offsets, so a waiting ant at the mouth
(x <= ~2.0) is never under a rock (nearest landing x ~3.3, radius 0.17,
leg reach ~0.75: 0.35 clearance) and an ant anywhere inside the band is.
Waves reuse the same four rock bodies: each wave teleports them back above
the band, so rocks on the floor from the previous wave vanish and fall
again. Contact with a dropped rock kills while the rockfall is open; once
parked the rocks cannot touch anything.

Who knows the latent, and when. The latent is drawn at reset (RNG
bookkeeping; audits force it through reset(rockfall_active=...)), but nothing
in the world can observe it before the mouth: ``revealed_rockfall_active`` is
None until the trigger fires and the latent afterwards. The trigger fires on
the observation in which the ant is first past the mouth line (checked after
physics, so the flag and the observation that shows the crossing arrive
together), and the expert reads it there and only there -- before the mouth
the expert is as blind as the learner, and its trajectory is identical for
both latents. ``privileged_rockfall_active`` stays the audits' oracle label.

Observation contract unchanged: 58-dim, zero-padded XY goal, no latent, no
rocks. Route labels are kept from V3 for continuity ('shortcut' on band
entry, 'detour' high in the west column) although the dataset only ever
shows the shortcut. New step-info fields: trigger_step, band_entry_step,
rock_waves, rockfall_open, rockfall_passed.
"""
import numpy as np

from crl.d4rl_ant import OfflineD4rlAntUMazeEnv
from crl.tworoute_rockfall_ant import (ROCK_DROP_OFFSETS, ROCK_DROP_VZ,
                                       ROCK_DROP_LEAD, P_ACTIVE)
from crl.tworoute_rockfall_v3 import (TwoRouteRockfallV3Env, HAZARD_X,
                                      HAZARD_HALF_Y, DETOUR_Y, DETOUR_MAX_X)

#: trigger line: 1.2 units west of the band mouth (HAZARD_X[0] = 2.6). The
#: ant walks ~0.1 units/step, so a non-stopping ant reaches the band ~12
#: steps after the trigger, right as the first wave lands (~8 steps of fall).
MOUTH_X = 1.4
#: rockfall duration in env steps once triggered (active latent only).
ROCKFALL_STEPS = 72
#: a wave every WAVE_PERIOD steps: 6 waves at +0, +12, ..., +60; the last
#: lands by +68 < ROCKFALL_STEPS.
WAVE_PERIOD = 12
#: aim point b = clip(x + lead, AIM_X0, AIM_X1); rocks land at b + da + jitter
#: with da in [0.35, 1.05] (ROCK_DROP_OFFSETS), clipped to the band's east
#: edge. A waiting ant (x + lead < AIM_X0) gets the pattern at 3.35..4.05.
AIM_X = (3.0, 4.3)
ROCK_X_MAX = HAZARD_X[1]
GOAL_CORNER = 'br'


def mouth_line():
  """(x, y0, y1) of the trigger line, for probes/renderers."""
  return (MOUTH_X, -HAZARD_HALF_Y, HAZARD_HALF_Y)


class RockfallWaitV4Env(TwoRouteRockfallV3Env):
  """V3 machinery (BR goal), mouth trigger, timed rock waves that pass."""

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl', p_active=P_ACTIVE):
    super().__init__(goal_corner=GOAL_CORNER,
                     max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode, p_active=p_active)
    self._t = 0
    self._trigger_step = None
    self._band_entry_step = None
    self._waves = 0
    self._rockfall_passed = False

  # ---- geometry --------------------------------------------------------
  @staticmethod
  def _at_mouth(x, y):
    return x >= MOUTH_X and abs(y) < HAZARD_HALF_Y

  @property
  def revealed_rockfall_active(self):
    """What the world shows at the mouth: None before the trigger, the
    latent afterwards. The sighted expert decides from this, nothing else."""
    return bool(self._rockfall_active) if self._rock_triggered else None

  @property
  def rockfall_open(self):
    """Privileged: rocks are falling / lying in the band right now."""
    return bool(self._rock_triggered and self._rockfall_active
                and not self._rockfall_passed)

  # ---- lifecycle -------------------------------------------------------
  def reset(self, rockfall_active=None):
    self._t = 0
    self._trigger_step = None
    self._band_entry_step = None
    self._waves = 0
    self._rockfall_passed = False
    return super().reset(rockfall_active=rockfall_active)

  def _info(self, success):
    info = super()._info(success)
    info.update({'trigger_step': self._trigger_step,
                 'band_entry_step': self._band_entry_step,
                 'rock_waves': int(self._waves),
                 'rockfall_open': self.rockfall_open,
                 'rockfall_passed': bool(self._rockfall_passed)})
    return info

  # ---- rocks -----------------------------------------------------------
  def _drop_rocks(self):
    """One wave: teleport the four rocks above the band, aimed ahead of the
    ant but never west of AIM_X[0] (see module doc), downward launch."""
    d = self._env.data
    x, y = float(d.qpos[0]), float(d.qpos[1])
    vx = float(d.qvel[0])
    #: forward lead only: the band is entered from the west, and an ant
    #: backing away from the mouth must not pull the pattern onto itself.
    lead = ROCK_DROP_LEAD * float(np.clip(vx, 0.0, 2.0))
    b = float(np.clip(x + lead, AIM_X[0], AIM_X[1]))
    for k, (dc, da, z) in enumerate(ROCK_DROP_OFFSETS):
      jx, jy = self._drop_jitter[k]
      qa, va = self._rock_qadr[k], self._rock_vadr[k]
      d.qpos[qa] = np.clip(b + da + jx, AIM_X[0], ROCK_X_MAX)
      d.qpos[qa + 1] = np.clip(y + dc + jy, -1.75, 1.75)
      d.qpos[qa + 2] = z
      d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      d.qvel[va:va + 6] = 0.0
      d.qvel[va + 2] = -ROCK_DROP_VZ
    self._rock_dropped = True
    self._waves += 1

  def _park_rocks(self):
    """The rockfall has passed: rocks back to storage, contacts void."""
    d = self._env.data
    home = self._env._home_qpos
    for qa, va in zip(self._rock_qadr, self._rock_vadr):
      d.qpos[qa:qa + 7] = home[qa:qa + 7]
      d.qvel[va:va + 6] = 0.0
    self._rock_dropped = False

  def step(self, action):
    if self._failed:
      #: absorbing terminal failure, as in V2/V3.
      return self._flatten(self._last_obs), 0.0, True, self._info(False)
    #: wave / park schedule BEFORE physics, clocked from the trigger step
    #: (the trigger itself is set after physics, see below; the first wave
    #: drops in the step after the crossing is observed, as in V3).
    if self._rock_triggered and not self._rockfall_passed:
      since = self._t - self._trigger_step
      if since >= ROCKFALL_STEPS:
        self._rockfall_passed = True
        if self._rockfall_active:
          self._park_rocks()
      elif self._rockfall_active and since % WAVE_PERIOD == 0:
        self._drop_rocks()
    #: GRANDPARENT step: V3.step() carries the band trigger, V2.step() the
    #: west-column one; neither may run on top of ours.
    obs, reward, _, _ = OfflineD4rlAntUMazeEnv.step(self, action)
    self._t += 1
    if reward > 0:
      self._succeeded = True
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))
    if self._in_band(x, y):
      if not self._entered_hazard:
        self._band_entry_step = self._t
      self._entered_hazard = True
      self._route = 'shortcut'
    if self._route is None and y >= DETOUR_Y and x < DETOUR_MAX_X:
      self._route = 'detour'
    self._rock_contact = self._dropped_rock_contact()
    if self._rock_contact and not self._succeeded:
      self._failed = True
      return obs, 0.0, True, self._info(False)
    #: trigger AFTER physics and the contact check, once per episode, for
    #: either latent: the observation returned here is the first one past
    #: the mouth line, and revealed_rockfall_active turns from None into
    #: the latent alongside it. Only the active latent has consequences.
    if not self._rock_triggered and self._at_mouth(x, y):
      self._rock_triggered = True
      self._trigger_step = self._t
    return obs, float(reward), False, self._info(self._succeeded)
