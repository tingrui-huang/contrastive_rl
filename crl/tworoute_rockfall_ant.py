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
state -- same convention as rockfall_ant._begin_episode). If the ant enters
the shortcut hazard band while the latent is active, the episode ends in a
TERMINAL FAILURE: step() returns done=True with info['failure']=True and the
env becomes absorbing (physics frozen, actions ignored) in case a fixed-
horizon caller keeps stepping. There are no physical rocks in V0 -- the
hazard is an explicit region trigger; MuJoCo rock bodies can be added later
behind the same info contract.

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
else None; info['entered_hazard'] is the raw band-entry flag.
"""
import numpy as np

from crl.d4rl_ant import (_Sim, OfflineD4rlAntUMazeEnv, build_maze_xml,
                          R, G, SCALING)

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


def hazard_zone():
  """(x0, x1, y0, y1) of the hazard band, for probes/renderers."""
  return (-HAZARD_HALF_X, HAZARD_HALF_X, HAZARD_Y[0], HAZARD_Y[1])


class TwoRouteRockfallAntEnv(OfflineD4rlAntUMazeEnv):
  """Offline-contract two-route AntMaze with the latent shortcut hazard."""

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl', p_active=P_ACTIVE):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    xml, offset = build_maze_xml(TWO_ROUTE_MAZE)
    self._env = _Sim(xml, seed)          # replace sim with the two-route model
    self._torso_offset = offset          # R cell unchanged -> same world frame
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
    self._rockfall_active = False
    self._entered_hazard = False
    self._failed = False
    self._succeeded = False
    self._route = None

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
  def reset(self, rockfall_active=None):
    """``rockfall_active`` override is for probes/gates/sanity checks only;
    normal use samples it. The rng is consumed in a fixed order regardless of
    the override, so forced-latent probes keep downstream draws identical."""
    drawn = bool(self._active_rng.random() < self.p_active)
    self._rockfall_active = (drawn if rockfall_active is None
                             else bool(rockfall_active))
    self._entered_hazard = False
    self._failed = False
    self._succeeded = False
    self._route = None
    return super().reset()

  def _info(self, success):
    return {'rockfall_active': bool(self._rockfall_active),
            'entered_hazard': bool(self._entered_hazard),
            'failure': bool(self._failed),
            'dead': bool(self._failed),   # repo-convention alias
            'success': bool(success),
            'route': self._route}

  def step(self, action):
    if self._failed:
      #: absorbing terminal failure: physics frozen, action ignored, reward
      #: can never fire (mirrors the repo's dead-state convention) -- but
      #: unlike the fixed-length envs we ALSO return done=True.
      return self._flatten(self._last_obs), 0.0, True, self._info(False)
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
    if in_hazard and self._rockfall_active and not self._succeeded:
      self._failed = True
      return obs, 0.0, True, self._info(False)
    return obs, float(reward), False, self._info(self._succeeded)
