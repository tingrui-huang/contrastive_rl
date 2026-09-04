"""Two-route AntMaze rockfall, V3: the hazard sits on the EAST leg and the
canonical heading points straight at it.

Same ring maze, same physical-rock machinery, same 58-dim obs contract and
one-canonical-pose protocol as V2 (crl/tworoute_rockfall_ant.py). What V3
changes is WHERE the choice costs and pays:

  * The hazard band moves from the west column to the BOTTOM ROW -- the leg
    the native east-facing pose walks straight into. In V2 the shortcut
    required a 90-deg turn, so the lazy/averaged behaviour landed on the
    safe route and the benchmark under-measured the danger; in V3 doing
    nothing IS the dangerous route.
  * The goal corner is a parameter, and the two registered variants form a
    controlled pair -- identical dynamics, identical confounding, identical
    SPARSE reference numbers (shortcut 0.70 / detour 0.96 / oracle 0.988);
    the ONLY manipulated variable is the incentive the algorithm's own
    discounted-reachability objective (gamma = 0.99, effective horizon
    ~1/(1-gamma) = 100 steps) assigns to the shortcut:

      'tr' goal (8,8): shortcut E->N ~156 steps, detour N->E ~164 steps.
          Equal length (Manhattan L-paths to the opposite corner are equal
          by construction); the only cost difference is the detour's start
          turn (~8 steps). Discounted refs: shortcut 0.146, DETOUR 0.185
          (best blind), oracle 0.201. The objective is ~indifferent, so
          'tr' isolates the start-dynamics (mode averaging, the lazy east
          default) from any incentive.
      'br' goal (8,0): shortcut straight E ~77 steps, detour N->E->S ~225
          steps, ratio 2.9x. The third leg is forced by topology: goal and
          start share a row with the central wall between them, so the
          alternative must leave the row and come back. Discounted refs:
          SHORTCUT 0.323 (best blind), detour 0.100, oracle 0.353. The
          objective rationally prefers the dangerous route: an agent that
          walks into the rocks is optimizing correctly and dies only of
          the latent it cannot see.

  Feasibility was measured before this module was written (walker-only
  relay drivers, lane translated to 0, frame switch at 7.5): tr shortcut
  1.000 / detour 0.960; br shortcut 1.000 / detour 0.960; the band is
  crossed by 100% of shortcut runs and 0% of detour runs in both variants.

Route labelling: band entry is AUTHORITATIVE for 'shortcut' exactly as in
V2. The detour signature becomes "high in the WEST column" (y >= 6 with
x < 2): unlike V2 an x-guard is required, because the tr shortcut also
reaches y >= 6 -- in the EAST column -- on its second leg. (Belt and
braces: the shortcut crosses the band on its first leg, so by the time it
is high the label is already latched.)

Rock aim is the V2 pattern transposed: the ant crosses the band travelling
along x, so the lead runs along qvel[0] and the drop pattern's long axis is
x (clip [2.3, 5.7]) with the corridor width on y (clip [-1.75, 1.75]).
"""
import numpy as np

from crl.d4rl_ant import OfflineD4rlAntUMazeEnv
from crl.tworoute_rockfall_ant import (TwoRouteRockfallAntEnv,
                                       ROCK_DROP_OFFSETS, ROCK_DROP_LEAD,
                                       ROCK_DROP_VZ, P_ACTIVE)

#: hazard band = interior of the bottom-middle cell (world x in [2, 6]) with
#: the same 0.6-unit mouth margins as V2; full corridor width across.
HAZARD_X = (2.6, 5.4)
HAZARD_HALF_Y = 2.0
#: detour signature: high in the west column (see module doc for the x-guard).
DETOUR_Y = 6.0
DETOUR_MAX_X = 2.0
#: goal corner -> maze cell (row, col) in the unchanged TWO_ROUTE_MAZE grid.
GOAL_CELLS = {'tr': (3, 3), 'br': (1, 3)}


def hazard_zone():
  """(x0, x1, y0, y1) of the V3 hazard band, for probes/renderers."""
  return (HAZARD_X[0], HAZARD_X[1], -HAZARD_HALF_Y, HAZARD_HALF_Y)


class TwoRouteRockfallV3Env(TwoRouteRockfallAntEnv):
  """V2 machinery, east-leg hazard, parameterised goal corner."""

  def __init__(self, goal_corner, max_episode_steps=700, seed=0,
               render_mode=None, eval_goals=None, eval_goal_mode='d4rl',
               p_active=P_ACTIVE):
    if goal_corner not in GOAL_CELLS:
      raise ValueError(f'goal_corner must be one of {sorted(GOAL_CELLS)}, '
                       f'got {goal_corner!r}')
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode, p_active=p_active)
    self.goal_corner = str(goal_corner)
    self._eval_goal_cell_xy = self._cell_xy(GOAL_CELLS[self.goal_corner])

  # ---- transposed hazard geometry -----------------------------------------
  @staticmethod
  def _in_band(x, y):
    return abs(y) < HAZARD_HALF_Y and HAZARD_X[0] <= x <= HAZARD_X[1]

  def _drop_rocks(self):
    """V2's aimed drop with the axes swapped: lead along the x-velocity, the
    pattern's travel axis on x, corridor width on y."""
    d = self._env.data
    x, y = float(d.qpos[0]), float(d.qpos[1])
    vx = float(d.qvel[0])
    sgn = 1.0 if vx >= 0 else -1.0
    lead = ROCK_DROP_LEAD * min(abs(vx), 2.0) * sgn
    for k, (dc, da, z) in enumerate(ROCK_DROP_OFFSETS):
      #: dc = across-corridor offset, da = along-travel offset (V2 names them
      #: dx, dy for a +y crossing; V3 swaps the world axes, not the pattern).
      jx, jy = self._drop_jitter[k]
      qa, va = self._rock_qadr[k], self._rock_vadr[k]
      d.qpos[qa] = np.clip(x + lead + sgn * da + jx, 2.3, 5.7)
      d.qpos[qa + 1] = np.clip(y + dc + jy, -1.75, 1.75)
      d.qpos[qa + 2] = z
      d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      d.qvel[va:va + 6] = 0.0
      d.qvel[va + 2] = -ROCK_DROP_VZ
    self._rock_dropped = True

  def step(self, action):
    if self._failed:
      #: absorbing terminal failure, as in V2.
      return self._flatten(self._last_obs), 0.0, True, self._info(False)
    d = self._env.data
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    if not self._rock_triggered and self._in_band(x0, y0):
      self._rock_triggered = True
      if self._rockfall_active:
        self._drop_rocks()
    #: call the GRANDPARENT's step: V2's step() would re-run its own
    #: west-column trigger/label logic on top of ours.
    obs, reward, _, _ = OfflineD4rlAntUMazeEnv.step(self, action)
    if reward > 0:
      self._succeeded = True
    x, y = (float(self._last_obs['achieved_goal'][0]),
            float(self._last_obs['achieved_goal'][1]))
    if self._in_band(x, y):
      self._entered_hazard = True
      #: band entry stays AUTHORITATIVE (failure => route == 'shortcut').
      self._route = 'shortcut'
    if self._route is None and y >= DETOUR_Y and x < DETOUR_MAX_X:
      self._route = 'detour'
    self._rock_contact = self._dropped_rock_contact()
    if self._rock_contact and not self._succeeded:
      self._failed = True
      return obs, 0.0, True, self._info(False)
    return obs, float(reward), False, self._info(self._succeeded)
