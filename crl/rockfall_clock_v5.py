"""Rockfall AntMaze V5: the rockfall runs on ITS OWN CLOCK.

V4 (crl/rockfall_wait_v4.py) tied the rockfall to the ant: crossing the
mouth line fired the trigger, and an active latent then dropped six rock
waves over ROCKFALL_STEPS steps. That made "hold at the mouth, then walk" a
blind-safe policy, but physically the rockfall WAITED FOR THE ANT -- no
matter when the ant arrived, the burst began then. V5 keeps the maze, the
rocks, the latent, the mouth line and the 58-dim obs, and moves the burst
onto a schedule the ant cannot influence:

  * latent ``active ~ Bernoulli(P_ACTIVE)`` per episode, drawn from the
    UNCHANGED ``_active_rng`` stream (the latent sequence for a seed equals
    V4's and V3-br's);
  * a start step ``t0 ~ Uniform{T0_MIN..T0_MAX}`` drawn EVERY reset from a
    NEW dedicated rng (``_sched_rng``), regardless of the latent and of any
    override, so the ant's reset-noise stream and the jitter stream are
    untouched: a u=clear V5 episode is byte-identical to u=clear V4 / V3-br
    under the same actions;
  * if active, wave k (k = 0..5) drops at the TOP of the env step whose
    pre-step counter is ``t0 + WAVE_PERIOD * k``, and the rocks are parked at
    the top of the step with counter ``t0 + ROCKFALL_STEPS`` -- whether or
    not the ant is anywhere near the band. If inactive nothing physical
    happens at all (the V2/V3/V4 hiddenness convention).

V4 equivalence. V4 dropped wave 0 at the top of the step whose counter equals
its trigger step (set post-physics on the mouth crossing). So V5 with
``reset(rockfall_active=True, rockfall_start=<V4 trigger_step>)`` and the
same actions reproduces the V4 active episode step for step: same obs, same
rock positions, same death step (smoke gate T3b). The aim rule keeps V4's
convention inside the corridor (see ``_drop_rocks``). Two things are NOT
step-for-step and are deliberate: info['rockfall_open'] turns True with the
first drop (V4's flag was already True in the info of the trigger step,
before any rock existed), and neither 'trigger_step' nor V2's
'rock_triggered' is emitted (V5 has no trigger).

Contact detection (inherited from V2/V3/V4, kept for T3b parity). A rock
touching the ant counts only if the contact is still in the contact list
AFTER the env step's frame_skip substeps. A rock that hits the ant and
bounces off inside a step is therefore NOT a flagged contact and not a
failure, although the ant's state (and so the obs) already diverges from
the clear twin at that step; measured on sighted 'go' replays the first obs
divergence coincides with the first flagged contact in most episodes and
precedes it by a few steps in the rest. A hiddenness gate must therefore
compare the twins until the first step at which a dropped rock is WITHIN
REACH of the ant (``rock_within_reach`` below, an on-demand privileged
readout), not until the flagged contact, and report the divergence step,
the first within-reach step and the first flagged contact separately. A
per-substep detector would move the death step earlier than V4's and was
rejected for that reason.

Hiddenness is not bit-exact at the float64 level once the rocks leave
their parking spot: the constraint set changes (parked rock-floor contacts
vanish in flight and reappear on landing) and the Newton solver then
perturbs the ant's solution at ~1e-15 even with the ant motionless far
away. With zero actions this reaches the float32 obs after ~100 steps in
near-zero velocity dims (<= ~5e-13); with random or walking actions it was
exactly 0.0 up to the first rock within reach. Gate hiddenness with a 1e-6
tolerance (as the V4 smoke did) and report the exact-0 divergence step
alongside.

Who knows what, and when. The learner's 58-dim obs carries no latent, no
rocks and no clock (zero-padded XY goal as before). The sighted expert
knows the whole timetable from t = 0 through the privileged side channel
``schedule`` ("the ant always knows when the rocks come") but only ACTS on
it at the mouth line: if the burst would overlap its crossing it holds
(zero torque) until the rocks are parked, otherwise it walks straight
through. ``revealed_rockfall_active`` is kept for V4 tooling only.

Blind consequences. With t0 <= T0_MAX the burst [t0, t0 + 72] always
overlaps a straight crossing (V4 measured: mouth at step 12-21, band
23-31 -> 47-57), so do(go) | active still dies ~1.0; a blind "hold until
step BLIND_WAIT_UNTIL = T0_MAX + ROCKFALL_STEPS" is safe but slow; the V3-br
detour is safe and ~225 steps. The confounding is V4's: P(success | go
observed) ~ 1 vs P(success | do(go)) ~ 0.70.

The route coin (the expert occasionally taking the detour for coverage)
lives in the EXPERT (scripts/rockfall_clock_v5_teacher.py), not here: the
env has one hazard schedule and no notion of route intent. Route labels
('shortcut' on band entry, 'detour' high in the west column) are V3's.

New step-info fields: t, mouth_step (first observation past the mouth
line, the V4 trigger_step convention, for hesitation readouts; latched only
while the route is not 'detour', so detour rows keep None instead of a
latch near the goal), rockfall_start, rockfall_end. 'trigger_step' and
'rock_triggered' are NOT emitted.
"""
import mujoco
import numpy as np

from crl.d4rl_ant import OfflineD4rlAntUMazeEnv
from crl.tworoute_rockfall_ant import (TwoRouteRockfallAntEnv,
                                       ROCK_DROP_OFFSETS, ROCK_DROP_VZ,
                                       ROCK_DROP_LEAD, P_ACTIVE)
from crl.tworoute_rockfall_v3 import HAZARD_HALF_Y, DETOUR_Y, DETOUR_MAX_X
from crl.rockfall_wait_v4 import (RockfallWaitV4Env, MOUTH_X,  # noqa: F401
                                  ROCKFALL_STEPS, WAVE_PERIOD, AIM_X,
                                  ROCK_X_MAX, GOAL_CORNER, mouth_line)

#: the burst start step is uniform on {T0_MIN, ..., T0_MAX} (inclusive).
T0_MIN = 0
T0_MAX = 30
#: rng stream offset for the schedule draw (disjoint from _Sim's seed, the
#: goal rng's +777, the latent's +91_211 and the jitter's +55_803).
_SCHED_SEED_OFFSET = 77_419
#: the blind always-wait reference releases at this env step: the latest
#: possible burst has been parked by then whatever the latent.
BLIND_WAIT_UNTIL = T0_MAX + ROCKFALL_STEPS
#: "within reach" for the hiddenness gates: a dropped rock whose surface is
#: closer than this to any ant geom at the END of a step may touch the ant
#: inside the next step. Measured on 12 sighted-'go' replays (seeds 21-23,
#: t0 natural/0/11/30): the gap at the end of the step before the first
#: obs divergence was 0.02..0.45 (a landing rock closes ~0.5 per step),
#: and the twins were identical at every step before the first within-reach
#: step. Diagnostics only; the failure rule is V4's.
ROCK_REACH_DIST = 0.6


class RockfallClockV5Env(RockfallWaitV4Env):
  """V4 machinery (BR goal, mouth line, timed waves) with the burst clocked
  from a per-episode schedule instead of the ant's mouth crossing."""

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl', p_active=P_ACTIVE):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode, p_active=p_active)
    self._sched_rng = np.random.default_rng(seed + _SCHED_SEED_OFFSET)
    self._t0 = 0
    self._mouth_step = None

  # ---- privileged side channels (never in the learner obs) ---------------
  @property
  def privileged_rockfall_start(self):
    """Burst start step if the latent is active, else None."""
    return int(self._t0) if self._rockfall_active else None

  @property
  def privileged_rockfall_end(self):
    """Step at whose top the rocks are parked (active only), else None."""
    return int(self._t0) + ROCKFALL_STEPS if self._rockfall_active else None

  @property
  def schedule(self):
    """What the sighted expert reads: the env clock and the timetable
    (start/end None when inactive)."""
    return {'t': int(self._t), 'active': bool(self._rockfall_active),
            'start': self.privileged_rockfall_start,
            'end': self.privileged_rockfall_end}

  @property
  def rockfall_open(self):
    """Privileged: rocks are falling / lying in the band right now, i.e.
    True from the info of the step that dropped wave 0 (pre-step counter
    == t0) until the info before the parking step. NOT V4-equivalent at one
    index: V4's flag was already True in the info of its trigger step,
    before any rock had dropped."""
    return bool(self._rockfall_active and self._rock_dropped
                and not self._rockfall_passed)

  @property
  def rock_ant_distance(self):
    """Privileged, on demand (not in info): the smallest surface distance
    between any DROPPED rock and any ant geom in the current state, or
    None when no rock is dropped. Exact via mujoco.mj_geomDistance. The
    hiddenness gates (T3c, causal-audit pairs) compare the active/clear
    twins until the first step at which this drops below ROCK_REACH_DIST,
    because a rock that hits the ant and bounces off inside a step never
    shows up in ``_dropped_rock_contact`` (see the module doc)."""
    if not self._rock_dropped:
      return None
    m, d = self._env.model, self._env.data
    best = float('inf')
    for gr in self._rock_gids:
      for ga in self._ant_gids:
        dist = float(mujoco.mj_geomDistance(m, d, int(gr), int(ga),
                                            best, None))
        if dist < best:
          best = dist
    return best

  @property
  def rock_within_reach(self):
    """Privileged: a dropped rock is closer than ROCK_REACH_DIST to the
    ant (see ``rock_ant_distance``)."""
    dist = self.rock_ant_distance
    return bool(dist is not None and dist < ROCK_REACH_DIST)

  @property
  def revealed_rockfall_active(self):
    """Kept for V4 tooling only: the latent once the ant is past the mouth
    line, None before. The V5 expert reads ``schedule`` instead."""
    return (bool(self._rockfall_active) if self._mouth_step is not None
            else None)

  # ---- lifecycle -------------------------------------------------------
  def reset(self, rockfall_active=None, rockfall_start=None):
    """``rockfall_active`` / ``rockfall_start`` overrides are for probes and
    gates only. The schedule rng is consumed FIRST and in fixed order every
    reset, regardless of the latent and of either override, so paired
    probes keep byte-identical downstream draws. ``rockfall_start`` may be
    any int. Beyond the horizon: the burst never comes. NEGATIVE: the burst
    is already under way at reset -- the waves scheduled before step 0 are
    skipped, the remaining ones fall on the same WAVE_PERIOD lattice
    (``(t - start) % WAVE_PERIOD == 0``) and the rocks are parked at
    ``start + ROCKFALL_STEPS`` as usual (e.g. start=-5: 5 waves at steps
    7, 19, ..., 55, parked at 67). This is how the decision-rule probes
    place the burst END near the mouth arrival; a natural t0 is never
    negative."""
    self._t0 = int(self._sched_rng.integers(T0_MIN, T0_MAX + 1))
    if rockfall_start is not None:
      self._t0 = int(rockfall_start)
    self._t = 0
    self._mouth_step = None
    self._band_entry_step = None
    self._waves = 0
    self._rockfall_passed = False
    #: V4.reset resets the same counters plus _trigger_step; _rock_triggered
    #: stays False for the whole episode (V5 never uses V4's trigger).
    return super().reset(rockfall_active=rockfall_active)

  def _info(self, success):
    #: the V2 base fields, called directly so V4's trigger_step never
    #: appears. V2's 'rock_triggered' is dropped too: V5 has no trigger, so
    #: the flag would be constantly False and a V2-V4 reader would take
    #: it for "the hazard never fired" while six waves fell.
    info = TwoRouteRockfallAntEnv._info(self, success)
    info.pop('rock_triggered', None)
    info.update({'t': int(self._t),
                 'mouth_step': self._mouth_step,
                 'band_entry_step': self._band_entry_step,
                 'rock_waves': int(self._waves),
                 'rockfall_open': self.rockfall_open,
                 'rockfall_passed': bool(self._rockfall_passed),
                 'rockfall_start': self.privileged_rockfall_start,
                 'rockfall_end': self.privileged_rockfall_end})
    return info

  # ---- rocks -----------------------------------------------------------
  def _drop_rocks(self):
    """One wave. Timing is the schedule's; only the AIM depends on the ant,
    and only when the ant is in the corridor row at or past the mouth line
    (``abs(y) < HAZARD_HALF_Y and x >= MOUTH_X``, no upper x bound): then
    exactly V4's aim (forward lead on vx, clipped into AIM_X, y clipped to
    +-1.75), so an ant anywhere inside the band is under the pattern (V4
    convention) and a V4 active episode is reproduced step for step -- also
    for an ant that has already left the band eastwards or sits at the goal
    (8, 0): the aim then clips to AIM_X[1] and the rocks land at x <=
    ROCK_X_MAX, exactly where V4 would put them. Otherwise (ant west of the
    mouth line, or off the corridor row: the detour's north and east
    columns) the reference is ``(AIM_X[0], 0.0)`` with zero lead: the
    pattern lands on the band's default footprint (x ~3.35..4.05, y = dc +
    jitter) and never depends on where an absent ant is.

    Lethal extent (V4 convention, now relevant because the burst is not
    tied to the crossing): rocks land at x up to ROCK_X_MAX plus the rock
    radius (~5.6) and the ant's legs reach ~0.75 units from the torso, so
    an ant just EAST of the band (x <~ 6.3, |y| < 2) while the burst is open
    can still be hit -- the hazard does not end sharply at HAZARD_X[1]; the
    teacher's CROSS_MAX accounts for it. Symmetrically the first landing
    (x ~3.2) is reachable ~1 step BEFORE band entry (x ~2.4), which is what
    the teacher's MOUTH_TO_BAND_MIN accounts for."""
    d = self._env.data
    x, y = float(d.qpos[0]), float(d.qpos[1])
    if abs(y) < HAZARD_HALF_Y and x >= MOUTH_X:
      vx = float(d.qvel[0])
      #: forward lead only (V4): an ant backing away from the mouth must
      #: not pull the pattern onto itself.
      lead = ROCK_DROP_LEAD * float(np.clip(vx, 0.0, 2.0))
      b = float(np.clip(x + lead, AIM_X[0], AIM_X[1]))
      y_ref = y
    else:
      b = float(AIM_X[0])
      y_ref = 0.0
    for k, (dc, da, z) in enumerate(ROCK_DROP_OFFSETS):
      jx, jy = self._drop_jitter[k]
      qa, va = self._rock_qadr[k], self._rock_vadr[k]
      d.qpos[qa] = np.clip(b + da + jx, AIM_X[0], ROCK_X_MAX)
      d.qpos[qa + 1] = np.clip(y_ref + dc + jy, -1.75, 1.75)
      d.qpos[qa + 2] = z
      d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      d.qvel[va:va + 6] = 0.0
      d.qvel[va + 2] = -ROCK_DROP_VZ
    self._rock_dropped = True
    self._waves += 1

  def step(self, action):
    if self._failed:
      #: absorbing terminal failure, as in V2/V3/V4.
      return self._flatten(self._last_obs), 0.0, True, self._info(False)
    #: wave / park schedule BEFORE physics, clocked from t0 (the ant plays
    #: no part). An inactive latent does nothing physical. If t0 +
    #: ROCKFALL_STEPS lies beyond the episode the rocks simply stay and
    #: rockfall_passed stays False.
    if self._rockfall_active and not self._rockfall_passed:
      since = self._t - self._t0
      if since >= ROCKFALL_STEPS:
        self._rockfall_passed = True
        if self._rock_dropped:
          self._park_rocks()
      elif since >= 0 and since % WAVE_PERIOD == 0:
        self._drop_rocks()
    #: GREAT-GRANDPARENT step, exactly as V4: V4.step carries the mouth
    #: trigger, V3.step the band trigger, V2.step the west-column one; none
    #: may run on top of ours.
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
    #: mouth latch AFTER physics and the contact check (V4's trigger_step
    #: convention): the observation returned here is the first one past the
    #: mouth line. Diagnostics only (hesitation readouts); nothing physical
    #: hangs on it. Not latched once the route is 'detour': _at_mouth has
    #: no upper x bound, so a detour ant descending the east column into the
    #: bottom row near the goal would otherwise "reach the mouth" at step
    #: ~200-250 and poison hesitation = band_entry_step - mouth_step.
    if (self._mouth_step is None and self._route != 'detour'
        and self._at_mouth(x, y)):
      self._mouth_step = self._t
    return obs, float(reward), False, self._info(self._succeeded)
