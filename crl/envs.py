"""Modern env registry for the contrastive RL port.

Provides a tiny uniform env interface (no gym/gymnasium API churn to worry about
in the training loop):

    obs = env.reset()                       # -> np.float32 [full_obs_dim]
    obs, reward, done, info = env.step(a)   # reward in {0, 1}

plus metadata attributes: ``obs_dim`` (state width), ``goal_dim``,
``action_dim``, ``max_episode_steps``, ``start_index``, ``end_index``.

Observations are the flat ``concat([state, goal])`` layout the algorithm expects
(same as the original repo after ObservationFilterWrapper):
  * point_{Map}   : state=[x,y], goal=full state    -> start=0, end=-1
  * fetch_reach   : state=obs(10), goal=gripper xyz  -> start=0, end=3
  * fetch_push    : state=obs(25), goal=object xyz   -> start=3, end=6

PointEnv is a self-contained numpy port (runs on Windows/Colab CPU, no MuJoCo).
The Fetch envs wrap ``gymnasium-robotics`` (modern ``mujoco`` bindings) and are
intended to run on Colab.
"""
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 2D navigation (numpy port of point_env.PointEnv).
# ---------------------------------------------------------------------------
WALLS = {
    'Small': np.zeros((4, 4), dtype=int),
    'FourRooms': np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U': np.array([
        [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
}


def resize_walls(walls, factor):
  (height, width) = walls.shape
  row = np.array([i for i in range(height) for _ in range(factor)])
  col = np.array([i for i in range(width) for _ in range(factor)])
  walls = walls[row][:, col]
  assert walls.shape == (factor * height, factor * width)
  return walls


class PointEnv:
  """2D navigation environment (numpy port of point_env.PointEnv)."""

  start_index = 0
  end_index = -1

  def __init__(self, walls='Small', resize_factor=1, action_noise=0.01,
               max_episode_steps=50, seed=0):
    walls_arr = WALLS[walls]
    if resize_factor > 1:
      walls_arr = resize_walls(walls_arr, resize_factor)
    self._walls = walls_arr
    self._height, self._width = self._walls.shape
    self._action_noise = action_noise
    self._rng = np.random.default_rng(seed)
    self.max_episode_steps = max_episode_steps

    self.obs_dim = 2                       # state = [x, y]
    self.goal_dim = 2                      # goal  = full state (start=0,end=-1)
    self.action_dim = 2
    self._low = np.array([0.0, 0.0])
    self._high = np.array([self._height, self._width])
    self.reset()

  def _sample_empty_state(self):
    candidates = np.where(self._walls == 0)
    idx = self._rng.integers(0, len(candidates[0]))
    state = np.array([candidates[0][idx], candidates[1][idx]], dtype=float)
    state += self._rng.uniform(size=2)
    return state

  def _discretize_state(self, state):
    ij = np.floor(state).astype(int)
    ij = np.clip(ij, [0, 0], np.array(self._walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    if np.any(state < self._low) or np.any(state > self._high):
      return True
    (i, j) = self._discretize_state(state)
    return self._walls[i, j] == 1

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self.goal = self._sample_empty_state()
    self.state = self._sample_empty_state()
    return self._get_obs()

  def step(self, action):
    action = np.array(action, dtype=float).copy()
    if self._action_noise > 0:
      action += self._rng.normal(0, self._action_noise, (2,))
    action = np.clip(action, -1.0, 1.0)
    num_substeps = 10
    dt = 1.0 / num_substeps
    for _ in range(num_substeps):
      for axis in range(len(action)):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    reward = float(dist < 2.0)
    return obs, reward, False, {}


# ---------------------------------------------------------------------------
# Two-route gate maze (confounder-qualification env v0, numpy).
# SUPERSEDED by TwoRouteSwampEnv (point_two_route_swamp_v0) below -- kept as
# the qualified static-gate reference/ablation.
#
# Fixed start (left) and goal (right). Two routes:
#   * UPPER (short): the straight middle corridor through a single GATE cell.
#   * LOWER (long) : a detour down through the bottom corridor, always open.
# An episode-level hidden binary gate U ~ Bernoulli(gate_prob) decides whether
# the gate cell is passable (open) or a wall (closed). U is sampled at reset and
# is NOT part of the observation (learner sees XY only) -> U is a hidden
# confounder: a gate-aware teacher takes the shortcut when open and the safe
# route when closed, so the *behavior* couples U to both the action (at the
# fork) and the transition (at the gate front), while the learner cannot see U.
#
# Grid is indexed [x][y] (x horizontal, y vertical when plotted). 1 = wall.
# ---------------------------------------------------------------------------
# Base (gate-CLOSED) wall grid. Open grid = this with the gate cell freed.
_TWO_ROUTE_GATE_WALLS = np.array([
    [1, 1, 1, 0, 1],   # x=0  start cell at y=3
    [1, 0, 0, 0, 1],   # x=1  fork: middle (y=3) + drop to bottom (y=2->y=1)
    [1, 0, 1, 0, 1],   # x=2
    [1, 0, 1, 0, 1],   # x=3  gate-front cell at y=3
    [1, 0, 1, 1, 1],   # x=4  GATE cell at y=3 (1=closed here in the base grid)
    [1, 0, 1, 0, 1],   # x=5
    [1, 0, 1, 0, 1],   # x=6
    [1, 0, 0, 0, 1],   # x=7  join: bottom (y=1->y=2) rejoins middle (y=3)
    [1, 1, 1, 0, 1],   # x=8  goal cell at y=3
], dtype=int)


class TwoRouteGateEnv:
  """Fixed two-route maze with an episode-level hidden gate (confounder U)."""

  start_index = 0
  end_index = -1                       # goal = full 2D state (point_ convention)

  GATE_CELL = (4, 3)                   # U-controlled cell (short route)
  GATE_FRONT_CELL = (3, 3)             # cell immediately before the gate
  FORK_CELL = (1, 3)                   # upper/lower routes diverge here
  START_CELL = (0, 3)
  GOAL_CELL = (8, 3)
  START = np.array([0.5, 3.5])         # fixed start (center of START_CELL)
  GOAL = np.array([8.5, 3.5])          # fixed goal  (center of GOAL_CELL)

  def __init__(self, action_noise=0.01, max_episode_steps=50, seed=0,
               gate_prob=0.5, fixed_gate=None, jitter=0.0):
    self._walls_closed = _TWO_ROUTE_GATE_WALLS.copy()
    self._walls_open = self._walls_closed.copy()
    self._walls_open[self.GATE_CELL] = 0
    self._height, self._width = self._walls_closed.shape
    self._action_noise = action_noise
    self._rng = np.random.default_rng(seed)
    self.max_episode_steps = max_episode_steps
    self._gate_prob = gate_prob
    self._fixed_gate = fixed_gate      # None => sample; True/False => forced
    self._jitter = jitter

    self.obs_dim = 2                   # state = [x, y]  (U is NOT exposed)
    self.goal_dim = 2                  # goal  = full state
    self.action_dim = 2
    self._low = np.array([0.0, 0.0])
    self._high = np.array([self._height, self._width])
    self._gate_open = True
    self.reset()

  # -- gate / walls -------------------------------------------------------- #
  @property
  def _walls(self):
    """Current-episode wall grid (reflects the sampled gate)."""
    return self._walls_open if self._gate_open else self._walls_closed

  @property
  def gate_open(self):
    return self._gate_open

  def set_gate(self, is_open):
    """Force the gate (for clone rollouts / matched-state probes)."""
    self._gate_open = bool(is_open)

  # -- dynamics (mirrors PointEnv) ----------------------------------------- #
  def _discretize_state(self, state):
    ij = np.floor(state).astype(int)
    ij = np.clip(ij, [0, 0], np.array(self._walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    if np.any(state < self._low) or np.any(state > self._high):
      return True
    (i, j) = self._discretize_state(state)
    return self._walls[i, j] == 1

  def _get_obs(self):
    # No gate bit: the learner observation is XY (+ goal XY) only.
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def _jittered(self, center):
    c = np.asarray(center, float).copy()
    if self._jitter > 0:
      c += self._rng.uniform(-self._jitter, self._jitter, size=2)
    return c

  def reset(self):
    if self._fixed_gate is None:
      self._gate_open = bool(self._rng.random() < self._gate_prob)
    else:
      self._gate_open = bool(self._fixed_gate)
    self.goal = self._jittered(self.GOAL)
    self.state = self._jittered(self.START)
    return self._get_obs()

  def step(self, action):
    action = np.array(action, dtype=float).copy()
    if self._action_noise > 0:
      action += self._rng.normal(0, self._action_noise, (2,))
    action = np.clip(action, -1.0, 1.0)
    num_substeps = 10
    dt = 1.0 / num_substeps
    for _ in range(num_substeps):
      for axis in range(len(action)):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    reward = float(dist < 2.0)
    return obs, reward, False, {}


# ---------------------------------------------------------------------------
# Two-route SWAMP maze (confounder-qualification env v1, numpy).
#
# Same two-route geometry as the gate env, but the confounder is DYNAMIC and
# acts through the transition function instead of geometry: the short route
# crosses THREE consecutive SWAMP cells, each independently active with
# probability ``active_prob`` (default 0.2). Active cells are NEVER walls --
# they multiply movement by ``slow_factor`` while the point is inside them
# (a trap that is recoverable by crawling back out, but very costly).
#
# Hidden state U_t = the 3 swamp bits. Resampling protocol:
#   * sampled fresh at reset,
#   * resampled at the END of every step while the agent is OUTSIDE the swamp
#     corridor (start area, pre-swamp HOLDING cell, lower route),
#   * FROZEN while the agent is inside any swamp cell (the configuration met
#     at entry persists until it exits back out).
#
# The learner observes [x, y, goal_x, goal_y] ONLY -- no swamp bits, no wait
# counter. The gate-aware teacher (scripts/qualify_two_route_swamp.py) reads
# ``swamp_bits`` at the holding cell: all clear -> shortcut; any active ->
# wait one step, re-check after the resample, clear -> shortcut else take the
# always-safe lower route.
# ---------------------------------------------------------------------------
_TWO_ROUTE_SWAMP_WALLS = np.array([
    [1, 1, 1, 0, 1],   # x=0  start cell at y=3
    [1, 0, 0, 0, 1],   # x=1  fork: middle (y=3) + drop to bottom (y=2->y=1)
    [1, 0, 1, 0, 1],   # x=2  HOLDING cell at y=3 (pre-swamp decision point)
    [1, 0, 1, 0, 1],   # x=3  swamp cell 0
    [1, 0, 1, 0, 1],   # x=4  swamp cell 1
    [1, 0, 1, 0, 1],   # x=5  swamp cell 2
    [1, 0, 1, 0, 1],   # x=6
    [1, 0, 0, 0, 1],   # x=7  join: bottom (y=1->y=2) rejoins middle (y=3)
    [1, 1, 1, 0, 1],   # x=8  goal cell at y=3
], dtype=int)


class TwoRouteSwampEnv:
  """Two-route maze whose short route crosses 3 hidden dynamic swamp cells."""

  start_index = 0
  end_index = -1                       # goal = full 2D state (point_ convention)

  SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
  HOLDING_CELL = (2, 3)                # pre-swamp decision/holding cell
  FORK_CELL = (1, 3)                   # upper/lower routes diverge here
  START_CELL = (0, 3)
  GOAL_CELL = (8, 3)
  START = np.array([0.5, 3.5])
  GOAL = np.array([8.5, 3.5])

  def __init__(self, action_noise=0.01, max_episode_steps=50, seed=0,
               active_prob=0.2, slow_factor=0.02):
    self._walls = _TWO_ROUTE_SWAMP_WALLS.copy()
    self._height, self._width = self._walls.shape
    self._action_noise = action_noise
    self._rng = np.random.default_rng(seed)
    self.max_episode_steps = max_episode_steps
    self.active_prob = active_prob
    self.slow_factor = slow_factor
    self._auto_resample = True
    self._bits = np.zeros(3, dtype=bool)

    self.obs_dim = 2                   # state = [x, y]  (U is NOT exposed)
    self.goal_dim = 2                  # goal  = full state
    self.action_dim = 2
    self._low = np.array([0.0, 0.0])
    self._high = np.array([self._height, self._width])
    self.reset()

  # -- hidden swamp state (teacher/probe access ONLY; never in the obs) ----- #
  @property
  def swamp_bits(self):
    return self._bits.copy()

  def set_swamp(self, bits):
    """Force the 3 swamp bits (for clone rollouts / matched-state probes)."""
    self._bits = np.asarray(bits, dtype=bool).copy()

  def set_auto_resample(self, enabled):
    """Disable to hold a forced configuration fixed during probes."""
    self._auto_resample = bool(enabled)

  def _resample(self):
    self._bits = self._rng.random(3) < self.active_prob

  def _swamp_index(self, state):
    c = tuple(self._discretize_state(state))
    for k, sc in enumerate(self.SWAMP_CELLS):
      if c == sc:
        return k
    return None

  def _in_swamp_corridor(self, state):
    return self._swamp_index(state) is not None

  def _in_active_swamp(self, state):
    k = self._swamp_index(state)
    return k is not None and bool(self._bits[k])

  # -- dynamics (PointEnv substep scheme + swamp slowdown) ------------------ #
  def _discretize_state(self, state):
    ij = np.floor(state).astype(int)
    ij = np.clip(ij, [0, 0], np.array(self._walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    # Swamp cells are NEVER walls; only the static grid + boundary block.
    if np.any(state < self._low) or np.any(state > self._high):
      return True
    (i, j) = self._discretize_state(state)
    return self._walls[i, j] == 1

  def _get_obs(self):
    # XY + goal only: no swamp bits, no wait counter, no time index.
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self.goal = self.GOAL.copy()
    self.state = self.START.copy()
    if self._auto_resample:
      self._resample()
    return self._get_obs()

  def step(self, action):
    action = np.array(action, dtype=float).copy()
    if self._action_noise > 0:
      action += self._rng.normal(0, self._action_noise, (2,))
    action = np.clip(action, -1.0, 1.0)
    num_substeps = 10
    dt = 1.0 / num_substeps
    for _ in range(num_substeps):
      # Trap: while inside an ACTIVE swamp cell, motion is scaled down hard
      # (recoverable -- the point can crawl back out -- but very costly).
      factor = self.slow_factor if self._in_active_swamp(self.state) else 1.0
      for axis in range(len(action)):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis] * factor
        if not self._is_blocked(new_state):
          self.state = new_state
    # U resamples ONLY while the agent ends the step OUTSIDE the corridor;
    # the configuration met at entry stays frozen until it exits. The bits a
    # policy reads between steps are exactly the bits governing the next step.
    if self._auto_resample and not self._in_swamp_corridor(self.state):
      self._resample()
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    reward = float(dist < 2.0)
    return obs, reward, False, {}


class TwoRouteSwampMatchedEnv(TwoRouteSwampEnv):
  """MiniGrid-matched variant of the swamp env (point_two_route_swamp_matched_v0).

  IDENTICAL geometry, dynamics, and freeze/resample timing to the strong
  TwoRouteSwampEnv -- the ONLY change is the per-cell swamp activation
  probability p=0.10 (vs 0.20 in the strong stress-test setting), to line up
  with the MiniGrid WindyCorridor confounder strength. The MiniGrid-matched
  TEACHER (5% episode-level force-safe, wait-until-clear) is a behavior policy
  that lives in the collector/qualifier, NOT in the env. The strong
  point_two_route_swamp_v0 env is left completely unchanged."""

  def __init__(self, action_noise=0.01, max_episode_steps=50, seed=0,
               active_prob=0.10, slow_factor=0.02):
    super().__init__(action_noise=action_noise,
                     max_episode_steps=max_episode_steps, seed=seed,
                     active_prob=active_prob, slow_factor=slow_factor)


class TwoRouteSwampWindyEnv(TwoRouteSwampEnv):
  """Windy-LETHAL swamp (point_two_route_swamp_windy_v0) -- the wind+lava
  design: per-step confounder + terminal trap.

  Differences from TwoRouteSwampEnv (same geometry, obs, action, horizon):
    * bits resample at the END of EVERY step, inside or outside the corridor
      -- wind semantics, NO entry freeze.
    * ending a step inside an ACTIVE swamp cell is TERMINAL (lava semantics):
      the agent is dead for the rest of the episode -- frozen in place,
      actions ignored, reward can never fire. done stays False (fixed-length
      episode contract; the pipeline layout is unchanged).
  Timing contract: the bits a policy reads BEFORE acting are exactly the
  bits used for THIS step's death check; they redraw afterwards. So a
  bits-aware teacher can look one cell ahead ("is the cell I would land in
  active right now?") -- the per-step u->a reaction, WindyCorridor-style.
  slow_factor is unused (lethality replaces the slowdown trap).
  """

  def __init__(self, action_noise=0.01, max_episode_steps=50, seed=0,
               active_prob=0.10, slow_factor=0.02):
    self._dead = False
    super().__init__(action_noise=action_noise,
                     max_episode_steps=max_episode_steps, seed=seed,
                     active_prob=active_prob, slow_factor=slow_factor)

  @property
  def dead(self):
    return self._dead

  def reset(self):
    self._dead = False
    return super().reset()

  def step(self, action):
    if self._dead:                       # absorbing: frozen until episode end
      if self._auto_resample:
        self._resample()
      return self._get_obs(), 0.0, False, {}
    action = np.array(action, dtype=float).copy()
    if self._action_noise > 0:
      action += self._rng.normal(0, self._action_noise, (2,))
    action = np.clip(action, -1.0, 1.0)
    num_substeps = 10
    dt = 1.0 / num_substeps
    for _ in range(num_substeps):        # full speed -- no slowdown trap
      for axis in range(len(action)):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state
    # death check with the bits that governed THIS step, then redraw
    if self._in_active_swamp(self.state):
      self._dead = True
    if self._auto_resample:
      self._resample()
    obs = self._get_obs()
    if self._dead:
      return obs, 0.0, False, {}
    dist = np.linalg.norm(self.goal - self.state)
    return obs, float(dist < 2.0), False, {}


# ---------------------------------------------------------------------------
# Fetch (gymnasium-robotics wrapper). Colab-oriented; needs `mujoco` +
# `gymnasium-robotics`. Flattens Dict obs to concat([state, desired_goal]).
# ---------------------------------------------------------------------------
class FetchEnv:
  """Wraps a gymnasium-robotics Fetch env into the flat [state, goal] layout."""

  # (start_index, end_index, candidate gym ids) per task.
  _SPECS = {
      'fetch_reach': (0, 3, ['FetchReach-v4', 'FetchReach-v3', 'FetchReach-v2']),
      'fetch_push': (3, 6, ['FetchPush-v4', 'FetchPush-v3', 'FetchPush-v2']),
      'fetch_push_start_at_obj': (3, 6, ['FetchPush-v4', 'FetchPush-v3',
                                         'FetchPush-v2']),
      # Literal as-shipped original defaults: full-state goal (start=0,end=-1).
      'fetch_push_original_style': (0, -1, ['FetchPush-v4', 'FetchPush-v3',
                                            'FetchPush-v2']),
      # EasyPush Level 1: object-only goal, fixed object, short +x push.
      'fetch_push_easy_fixed': (3, 6, ['FetchPush-v4', 'FetchPush-v3',
                                       'FetchPush-v2']),
      # EasyPush Level 2B: fixed object, short push in a RANDOM direction.
      'fetch_push_easy_multidir': (3, 6, ['FetchPush-v4', 'FetchPush-v3',
                                          'FetchPush-v2']),
      # EasyPush Level 2C: random goal direction, but NEUTRAL (fixed -x) gripper
      # start -> policy must infer + acquire the push direction from the goal.
      'fetch_push_easy_neutral_dir': (3, 6, ['FetchPush-v4', 'FetchPush-v3',
                                             'FetchPush-v2']),
      # Calibration rung (state-based approx of original image-Push geometry):
      # fixed -x gripper + goal in a CONE around +x (the measured natural push).
      'fetch_push_easy_conedir': (3, 6, ['FetchPush-v4', 'FetchPush-v3',
                                         'FetchPush-v2']),
      # IMAGE version of conedir: obs = concat(frame, goal_frame), each a
      # flattened 64x64x3 uint8 render from the original's fixed 'camera2'.
      # Goal slice (0,-1): the relabel goal is the FULL future frame (identity
      # obs_to_goal, as in the original image envs).
      'fetch_push_image_conedir': (0, -1, ['FetchPush-v4', 'FetchPush-v3',
                                           'FetchPush-v2']),
  }

  def __init__(self, task='fetch_reach', max_episode_steps=50, seed=0,
               render_mode=None, start_at_obj=False, original_style=False,
               easy_fixed=False, multidir=False, neutral_dir=False,
               cone_dir=False, cone_half_width=np.pi / 3,
               image_obs=False, image_size=64):
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  (registers the envs)
    gym.register_envs(gymnasium_robotics)

    self._start_at_obj = start_at_obj
    self._original_style = original_style
    self._easy_fixed = easy_fixed
    self._multidir = multidir
    self._neutral_dir = neutral_dir
    self._cone_dir = cone_dir
    self._cone_hw = cone_half_width           # cone half-width around +x
    self._last_dir = np.array([1.0, 0.0, 0.0])   # goal direction (audit hook)
    self._rng = np.random.default_rng(seed)
    # EasyPush L1 geometry: object pinned +x of the gripper home (~1.363) so the
    # gripper starts BEHIND it; goal a short +x push away (6-9 cm). Both on table.
    self._obj0 = np.array([1.40, 0.75, 0.425])
    self._push_range = (0.06, 0.09)
    self.use_image_obs = bool(image_obs)
    self._img_size = int(image_size)
    self._goal_img = None
    self.start_index, self.end_index, ids = self._SPECS[task]
    make_kwargs = dict(max_episode_steps=max_episode_steps,
                       render_mode=render_mode)
    if image_obs:
      # Frames come from env.render(); the render buffer is sized 64x64 like
      # the original FetchPushImage (fetch_envs.py:264 rendered height=width=64).
      make_kwargs.update(render_mode='rgb_array',
                         width=self._img_size, height=self._img_size)
    last_err = None
    for env_id in ids:
      try:
        self._env = gym.make(env_id, **make_kwargs)
        self._env_id = env_id
        break
      except Exception as e:  # pylint: disable=broad-except
        last_err = e
    else:
      raise RuntimeError(
          f'Could not create any of {ids}. Last error: {last_err}')

    self._seed = seed
    self.max_episode_steps = max_episode_steps
    obs, _ = self._env.reset(seed=seed)
    self.obs_dim = int(obs['observation'].shape[0])          # 10 reach / 25 push
    self._dg_dim = int(obs['desired_goal'].shape[0])         # 3
    # original_style: goal half is a FULL-width state vector with desired_goal
    # embedded at [0:3] and [3:6] (ports fetch_envs.FetchPushEnv.observation),
    # so the relabel goal (start=0,end=-1) is the full future state.
    self.goal_dim = self.obs_dim if original_style else self._dg_dim
    self.action_dim = int(self._env.action_space.shape[0])
    self._success_threshold = 0.05
    if image_obs:
      # Visual-only setup; dynamics untouched. Must run BEFORE any render.
      self._hide_visual_markers()
      self._set_camera2()
      self.obs_dim = self._img_size * self._img_size * 3
      self.goal_dim = self.obs_dim

  # ---- image-obs helpers (port of fetch_envs.FetchPushImage rendering) ----
  def _hide_visual_markers(self):
    """The goal must NOT leak into the frame. Ports fetch_envs.py's two hacks:
    site_xpos[0]=1e6 each render (hide red goal marker, fetch_envs.py:263) and
    geom_rgba[1:5]=0 (hide mocap box + 'lasers', fetch_envs.py:187) -- done
    here once via alpha=0 (persistent; the modern _render_callback re-places
    the target site every render, so moving it would not stick)."""
    u = self._env.unwrapped
    mj = u._mujoco
    sid = mj.mj_name2id(u.model, mj.mjtObj.mjOBJ_SITE, 'target0')
    if sid >= 0:
      u.model.site_rgba[sid, 3] = 0.0
    bid = mj.mj_name2id(u.model, mj.mjtObj.mjOBJ_BODY, 'robot0:mocap')
    if bid >= 0:
      for gid in range(u.model.ngeom):
        if u.model.geom_bodyid[gid] == bid:
          u.model.geom_rgba[gid, 3] = 0.0

  def _set_camera2(self):
    """Fixed side camera = the original's 'camera2' (fetch_envs.py:274-278:
    distance 0.65, azimuth 90, elevation -40), with the lookat recentered on
    the conedir workspace (object pinned at x=1.40 vs ~1.15-1.25 originally;
    camera2's literal lookat=[1.25,0.8,0.4] pushes the action off-frame)."""
    cam = {'lookat': np.array([1.43, 0.75, 0.42]), 'distance': 0.65,
           'azimuth': 90.0, 'elevation': -40.0}
    r = self._env.unwrapped.mujoco_renderer
    r.default_cam_config = cam            # applied when the viewer is created
    for v in getattr(r, '_viewers', {}).values():  # or now, if it exists
      v.cam.lookat[:] = cam['lookat']
      v.cam.distance, v.cam.azimuth, v.cam.elevation = (
          cam['distance'], cam['azimuth'], cam['elevation'])

  def _frame(self):
    """One flattened uint8 RGB frame (fetch_envs.py observation())."""
    img = self._env.render()
    return np.asarray(img, dtype=np.uint8).reshape(-1)

  def _render_goal_image(self, theta, push):
    """Goal-image recipe of FetchPushImage.reset (fetch_envs.py:209-226) in
    conedir geometry: put the OBJECT at the goal, gripper next to it on the -x
    side (original used _move_hand_to_obj's -x offset), snapshot a frame. All
    scripted; the sim is fully reset afterwards, so dynamics are untouched."""
    u = self._env.unwrapped
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    g = (self._obj0 + push * d).astype(float)
    u.goal = g.copy()
    q = np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'),
                 dtype=float).copy()
    q[0:3] = g
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    u._utils.set_joint_qpos(u.model, u.data, 'object0:joint', q)
    u._mujoco.mj_forward(u.model, u.data)
    zc = self._obj0[2]
    beside = g + np.array([-0.04, 0.0, 0.0])
    grip0 = np.asarray(u._get_obs()['observation'][:3])
    self._move_gripper_to([grip0[0], grip0[1], zc + 0.12])    # lift
    self._move_gripper_to([beside[0], beside[1], zc + 0.12])  # move over
    self._move_gripper_to([beside[0], beside[1], zc + 0.005])  # descend
    # Re-pin the object AT the goal (undo any placement nudge) and snapshot.
    q = np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'),
                 dtype=float).copy()
    q[0:3] = g
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    u._utils.set_joint_qpos(u.model, u.data, 'object0:joint', q)
    u._mujoco.mj_forward(u.model, u.data)
    return self._frame()

  def _flatten(self, obs):
    if self._original_style:
      g = np.zeros(self.obs_dim, dtype=np.float32)
      g[:3] = obs['desired_goal']       # matches fetch_envs.py:96
      g[3:6] = obs['desired_goal']      # matches fetch_envs.py:97
      return np.concatenate([obs['observation'], g]).astype(np.float32)
    return np.concatenate(
        [obs['observation'], obs['desired_goal']]).astype(np.float32)

  def _move_hand_to_obj(self, obs):
    """Scripted gripper->object move at reset (ports FetchPushImage's
    _move_hand_to_obj): drive the gripper to within 0.06 of the object, just
    behind it (-0.02 in x, ready to push). Steps the UNWRAPPED env so these
    scripted moves do not count against the episode's TimeLimit."""
    u = self._env.unwrapped
    for _ in range(100):
      hand = np.asarray(obs['observation'][:3])
      target = np.asarray(obs['achieved_goal']) + np.array([-0.02, 0.0, 0.0])
      delta = target - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1.0, 1.0), [0.0]]).astype(np.float32)
      obs = u.step(a)[0]
    return obs

  def _reset_easy_fixed(self):
    """EasyPush L1: pin object at a fixed spot, set a short +x goal, start-at-obj."""
    u = self._env.unwrapped
    q = np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'),
                 dtype=float).copy()
    q[0:3] = self._obj0
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    u._utils.set_joint_qpos(u.model, u.data, 'object0:joint', q)
    u._mujoco.mj_forward(u.model, u.data)
    push = float(self._rng.uniform(*self._push_range))
    yj = float(self._rng.uniform(-0.01, 0.01))
    u.goal = np.array([self._obj0[0] + push, self._obj0[1] + yj, self._obj0[2]],
                      dtype=float)
    obs = u._get_obs()
    return self._move_hand_to_obj(obs)      # gripper starts behind the object

  def _pin_object(self):
    u = self._env.unwrapped
    q = np.array(u._utils.get_joint_qpos(u.model, u.data, 'object0:joint'),
                 dtype=float).copy()
    q[0:3] = self._obj0
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    u._utils.set_joint_qpos(u.model, u.data, 'object0:joint', q)
    u._mujoco.mj_forward(u.model, u.data)

  def _move_gripper_to(self, target, n=40):
    """Scripted move of the gripper to a world xyz target (unwrapped steps)."""
    u = self._env.unwrapped
    obs = u._get_obs()
    for _ in range(n):
      grip = np.asarray(obs['observation'][:3])
      delta = np.asarray(target) - grip
      if np.linalg.norm(delta) < 0.02:
        break
      a = np.concatenate([np.clip(delta * 10.0, -1, 1), [0.0]]).astype(np.float32)
      obs = u.step(a)[0]
    return obs

  def _reset_multidir(self):
    """EasyPush L2B: fixed object; goal a short push in a RANDOM direction.
    Places the gripper BEHIND the object relative to the goal direction via a
    lift -> move-over -> descend path (so it works for any direction without
    knocking the object)."""
    u = self._env.unwrapped
    self._pin_object()
    theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    self._last_dir = d
    push = float(self._rng.uniform(*self._push_range))
    u.goal = (self._obj0 + push * d).astype(float)
    zc = self._obj0[2]
    behind = self._obj0 - 0.04 * d                       # behind wrt goal dir
    grip0 = np.asarray(u._get_obs()['observation'][:3])
    self._move_gripper_to([grip0[0], grip0[1], zc + 0.12])   # 1. lift
    self._move_gripper_to([behind[0], behind[1], zc + 0.12])  # 2. move over
    self._move_gripper_to([behind[0], behind[1], zc + 0.005])  # 3. descend
    self._pin_object()   # re-pin: undo any nudge from placement -> clean geometry
    return u._get_obs()

  def _reset_neutral_dir(self):
    """EasyPush L2C: fixed object; goal in a RANDOM direction; but the gripper
    always starts on the object's -x side, INDEPENDENT of the goal direction, so
    the policy must infer + acquire the correct push direction from the goal."""
    u = self._env.unwrapped
    self._pin_object()
    theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    self._last_dir = d
    push = float(self._rng.uniform(*self._push_range))
    u.goal = (self._obj0 + push * d).astype(float)
    zc = self._obj0[2]
    start = self._obj0 + np.array([-0.04, 0.0, 0.0])     # FIXED -x side
    grip0 = np.asarray(u._get_obs()['observation'][:3])
    self._move_gripper_to([grip0[0], grip0[1], zc + 0.12])  # lift
    self._move_gripper_to([start[0], start[1], zc + 0.12])  # over -x side
    self._move_gripper_to([start[0], start[1], zc + 0.005])  # descend
    self._pin_object()
    return u._get_obs()

  def _reset_conedir(self, theta=None, push=None):
    """Calibration rung: fixed -x gripper (like L2C) but goal direction sampled
    from a CONE around +x (the empirically-measured natural push direction from
    a -x-side gripper). State-based approximation of the original image-Push
    geometry -- NOT an exact reproduction. ``theta``/``push`` can be pinned by
    the image path so the episode goal matches the pre-rendered goal image."""
    u = self._env.unwrapped
    self._pin_object()
    if theta is None:
      theta = float(self._rng.uniform(-self._cone_hw, self._cone_hw))  # around +x
    if push is None:
      push = float(self._rng.uniform(*self._push_range))
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    self._last_dir = d
    u.goal = (self._obj0 + push * d).astype(float)
    zc = self._obj0[2]
    start = self._obj0 + np.array([-0.04, 0.0, 0.0])     # FIXED -x side
    grip0 = np.asarray(u._get_obs()['observation'][:3])
    self._move_gripper_to([grip0[0], grip0[1], zc + 0.12])
    self._move_gripper_to([start[0], start[1], zc + 0.12])
    self._move_gripper_to([start[0], start[1], zc + 0.005])
    self._pin_object()
    return u._get_obs()

  def reset(self):
    obs, _ = self._env.reset()
    if self.use_image_obs:
      # Two-reset recipe of FetchPushImage.reset: build the goal image on a
      # scratch sim, then fully reset and set up the REAL episode with the SAME
      # sampled goal, so the goal image depicts this episode's target.
      theta = float(self._rng.uniform(-self._cone_hw, self._cone_hw))
      push = float(self._rng.uniform(*self._push_range))
      self._goal_img = self._render_goal_image(theta, push)
      obs, _ = self._env.reset()
      obs = self._reset_conedir(theta=theta, push=push)
      self._desired = obs['desired_goal']
      return np.concatenate([self._frame(), self._goal_img])
    if self._cone_dir:
      obs = self._reset_conedir()
    elif self._neutral_dir:
      obs = self._reset_neutral_dir()
    elif self._multidir:
      obs = self._reset_multidir()
    elif self._easy_fixed:
      obs = self._reset_easy_fixed()
    elif self._start_at_obj:
      obs = self._move_hand_to_obj(obs)
    self._desired = obs['desired_goal']
    return self._flatten(obs)

  def step(self, action):
    obs, _, _, _, _ = self._env.step(np.asarray(action, dtype=np.float32))
    dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    reward = float(dist < self._success_threshold)   # from sim state, not pixels
    if self.use_image_obs:
      return np.concatenate([self._frame(), self._goal_img]), reward, False, {}
    return self._flatten(obs), reward, False, {}

  def render(self):
    """Returns an RGB frame (needs render_mode='rgb_array' + MUJOCO_GL)."""
    return self._env.render()


# ---------------------------------------------------------------------------
# AntMaze / PointMaze (gymnasium-robotics). Prepends achieved_goal (xy) to the
# proprioceptive obs so the relabel goal slice is [0:2] (xy) -- matching the
# original ant_env's expose_all_qpos. Sparse reward = xy-distance < threshold.
# ---------------------------------------------------------------------------
_MAZE_IDS = {
    'antmaze_umaze':  ('AntMaze_UMaze-v5', 700),
    'antmaze_medium': ('AntMaze_Medium-v5', 1000),
    'antmaze_large':  ('AntMaze_Large-v5', 1000),
}


class MazeEnv:
  """Wraps a gymnasium-robotics Ant/Point maze into the flat [state, goal]
  layout: state = concat([achieved_goal(xy), proprio]), goal = desired_goal(xy),
  goal slice = [0:2]. Fixed-length episodes (no unhealthy early termination);
  start + goal randomized per reset (approximates the paper's non_zero_reset)."""

  start_index = 0
  end_index = 2

  def __init__(self, env_id, max_episode_steps, success_threshold=0.5,
               include_contact_forces=False, seed=0, render_mode=None):
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  (registers the envs)
    gym.register_envs(gymnasium_robotics)
    self._success_threshold = success_threshold
    kwargs = dict(max_episode_steps=max_episode_steps, continuing_task=False,
                  reset_target=True, render_mode=render_mode)
    if 'Ant' in env_id:                          # Ant-specific knobs
      kwargs['include_cfrc_ext_in_observation'] = include_contact_forces
      kwargs['terminate_when_unhealthy'] = False
    self._env = gym.make(env_id, **kwargs)
    self._env_id = env_id
    self.max_episode_steps = max_episode_steps
    obs, _ = self._env.reset(seed=seed)
    # Audit ACTUAL shapes at runtime (do NOT hard-code the obs dim).
    self._ag_dim = int(np.asarray(obs['achieved_goal']).shape[0])   # 2 (xy)
    proprio = int(np.asarray(obs['observation']).shape[0])
    self.obs_dim = self._ag_dim + proprio        # prepend xy -> full state width
    self.goal_dim = self._ag_dim                 # 2 (xy)
    self.action_dim = int(self._env.action_space.shape[0])
    self._last_obs = obs

  def _flatten(self, obs):
    state = np.concatenate([np.asarray(obs['achieved_goal']),
                            np.asarray(obs['observation'])])
    return np.concatenate(
        [state, np.asarray(obs['desired_goal'])]).astype(np.float32)

  def reset(self):
    obs, _ = self._env.reset()
    self._last_obs = obs
    return self._flatten(obs)

  def step(self, action):
    obs, _, _, _, _ = self._env.step(np.asarray(action, dtype=np.float32))
    self._last_obs = obs
    dist = float(np.linalg.norm(np.asarray(obs['achieved_goal'])
                                - np.asarray(obs['desired_goal'])))
    reward = float(dist < self._success_threshold)
    return self._flatten(obs), reward, False, {}

  def render(self):
    return self._env.render()


class NearGoalOpenMazeEnv(MazeEnv):
  """AntMaze_Open with near goals, for the locomotion qualification A/B.

  Identical physics/obs/reward to MazeEnv; ONLY the reset distribution differs:
  the start cell is uniform over free cells and the goal cell is the same or an
  orthogonally-adjacent free cell, rejection-sampled until the start-goal
  distance lies in [d_min, d_max] (<=30 tries, keep the last draw otherwise)."""

  def __init__(self, env_id='AntMaze_Open-v5', max_episode_steps=300,
               d_min=1.0, d_max=4.5, seed=0, render_mode=None):
    super().__init__(env_id, max_episode_steps, seed=seed,
                     render_mode=render_mode)
    self._d_min, self._d_max = d_min, d_max
    mz = self._env.unwrapped.maze
    self._free = [(r, c) for r in range(len(mz.maze_map))
                  for c in range(len(mz.maze_map[0])) if mz.maze_map[r][c] == 0]
    self._reset_rng = np.random.default_rng(seed + 991)

  def reset(self):
    rng = self._reset_rng
    obs = None
    for _ in range(30):
      rc = self._free[rng.integers(len(self._free))]
      nbrs = [(r, c) for (r, c) in self._free
              if abs(r - rc[0]) + abs(c - rc[1]) <= 1]  # same + orthogonal
      gc = nbrs[rng.integers(len(nbrs))]
      obs, _ = self._env.reset(options={'reset_cell': np.array(rc),
                                        'goal_cell': np.array(gc)})
      d = float(np.linalg.norm(np.asarray(obs['achieved_goal'])
                               - np.asarray(obs['desired_goal'])))
      if self._d_min <= d <= self._d_max:
        break
    self._last_obs = obs
    return self._flatten(obs)


# State layout of the flattened Ant state (obs_dim=29):
#   [0:2]  achieved xy          (= qpos[0:2])
#   [2]    torso height z       (= qpos[2])
#   [3:7]  torso quaternion     (= qpos[3:7])
#   [7:15] joint angles         (= qpos[7:15])
#   [15:18] torso linear vel    (= qvel[0:3])
#   [18:21] torso angular vel   (= qvel[3:6])
#   [21:29] joint velocities    (= qvel[6:14])
ANT_GOAL_BLOCKS = {
    'xy': tuple(range(0, 2)),
    'pose': tuple(range(2, 7)),            # z + quaternion
    'velocity': tuple(range(15, 21)),      # linear + angular
    'joints': tuple(range(7, 15)) + tuple(range(21, 29)),
}
ANT_COMPACT_GOAL_IDX = (ANT_GOAL_BLOCKS['xy'] + ANT_GOAL_BLOCKS['pose']
                        + ANT_GOAL_BLOCKS['velocity'])          # 13 dims
ANT_FULL_GOAL_IDX = tuple(range(29))                            # 29 dims


class RichGoalOpenMazeEnv(NearGoalOpenMazeEnv):
  """Near-goal open Ant with a richer commanded-goal representation.

  Same physics/reward/reset distribution as NearGoalOpenMazeEnv (reward and
  success stay XY-based). Only the GOAL half of the flat observation changes:
  at reset, the ant is teleported to the commanded goal cell (keeping its
  post-reset pose), settled for 50 zero-action env-steps (as the original
  ant_envs.AntMaze did), the full 29-dim settled state is snapshotted, and
  ``goal_indices`` of it become the commanded goal vector. The pre-settle
  start state is restored bit-exactly (qpos/qvel) afterwards.

  Relabeled goals (replay) use the same indices of FUTURE states, matching
  the original's obs_to_goal(full state) semantics when goal_indices=(0..28).
  """

  def __init__(self, goal_indices, **kw):
    super().__init__(**kw)
    assert tuple(goal_indices[:2]) == (0, 1), 'goal must start with XY'
    self.goal_indices = tuple(int(i) for i in goal_indices)
    self.goal_dim = len(self.goal_indices)
    self._goal_vec = np.zeros(self.goal_dim, np.float32)
    self._goal_state_full = np.zeros(29, np.float32)

  def _flatten(self, obs):
    state = np.concatenate([np.asarray(obs['achieved_goal']),
                            np.asarray(obs['observation'])])
    return np.concatenate([state, self._goal_vec]).astype(np.float32)

  def reset(self):
    # Parent does the near-goal rejection reset and sets self._last_obs, but
    # calls _flatten before we snapshot the goal state -- so re-flatten after.
    super().reset()
    obs = self._last_obs
    u = self._env.unwrapped
    import mujoco as _mj
    qpos0 = np.asarray(u.data.qpos).copy()
    qvel0 = np.asarray(u.data.qvel).copy()
    goal_xy = np.asarray(obs['desired_goal'], dtype=np.float64)
    # Teleport to the goal cell (keep pose), settle 50 zero-action env steps.
    u.data.qpos[:2] = goal_xy
    u.data.qvel[:] = 0.0
    _mj.mj_forward(u.model, u.data)
    u.data.ctrl[:] = 0.0
    frame_skip = getattr(u, 'frame_skip', 5)
    for _ in range(50 * frame_skip):
      _mj.mj_step(u.model, u.data)
    self._goal_state_full = np.concatenate(
        [np.asarray(u.data.qpos), np.asarray(u.data.qvel)]).astype(np.float32)
    self._goal_vec = self._goal_state_full[list(self.goal_indices)]
    # Restore the start state bit-exactly.
    u.data.qpos[:] = qpos0
    u.data.qvel[:] = qvel0
    _mj.mj_forward(u.model, u.data)
    return self._flatten(obs)


# ---------------------------------------------------------------------------
def make_env(env_name, config, seed=0, render_mode=None):
  """Builds an env and fills obs/goal/action dims + episode length into config.

  Returns the env; mutates ``config`` in place with obs_dim, goal_dim,
  action_dim, max_episode_steps, start_index, end_index.
  ``render_mode='rgb_array'`` enables Fetch frame rendering (for GIFs on Colab).
  """
  if env_name == 'point_two_route_swamp_matched_v0':
    env = TwoRouteSwampMatchedEnv(max_episode_steps=50, seed=seed)
  elif env_name == 'point_two_route_swamp_windy_v0':
    env = TwoRouteSwampWindyEnv(max_episode_steps=50, seed=seed)
  elif env_name == 'point_two_route_swamp_v0':
    env = TwoRouteSwampEnv(max_episode_steps=50, seed=seed)
  elif env_name == 'point_two_route_gate_v0':   # superseded v0 (kept as ablation)
    env = TwoRouteGateEnv(max_episode_steps=50, seed=seed)
  elif env_name.startswith('point_'):
    walls = env_name.split('_', 1)[1]
    # Only the 11x11 maps get the longer horizon (matches original env_utils.load).
    steps = 100 if '11x11' in walls else 50
    env = PointEnv(walls=walls, max_episode_steps=steps, seed=seed)
  elif env_name == 'antmaze_open_near':
    env = NearGoalOpenMazeEnv(seed=seed, render_mode=render_mode)
  elif env_name == 'antmaze_open_near_gcompact':
    env = RichGoalOpenMazeEnv(ANT_COMPACT_GOAL_IDX, seed=seed,
                              render_mode=render_mode)
  elif env_name == 'antmaze_open_near_gfull':
    env = RichGoalOpenMazeEnv(ANT_FULL_GOAL_IDX, seed=seed,
                              render_mode=render_mode)
  elif env_name == 'd4rl_ant_umaze_gfull':
    from crl.d4rl_ant import D4rlAntUMazeEnv
    env = D4rlAntUMazeEnv(seed=seed, render_mode=render_mode)
  elif env_name == 'offline_ant_umaze':
    # OFFLINE d4rl antmaze-umaze contract: zero-padded XY goal; eval goals
    # come from the offline dataset's empirical per-episode goals when the
    # config points at an offline .npz that carries an 'eval_goals' array.
    from crl.d4rl_ant import OfflineD4rlAntUMazeEnv
    eval_goals = None
    if getattr(config, 'offline_dataset', ''):
      with np.load(config.offline_dataset) as _d:
        if 'eval_goals' in _d:
          eval_goals = _d['eval_goals'].copy()
    env = OfflineD4rlAntUMazeEnv(seed=seed, render_mode=render_mode,
                                 eval_goals=eval_goals,
                                 eval_goal_mode=getattr(config,
                                                        'eval_goal_mode', 'd4rl'))
  elif env_name.startswith('antmaze_'):
    env_id, steps = _MAZE_IDS[env_name]
    env = MazeEnv(env_id, max_episode_steps=steps, seed=seed,
                  render_mode=render_mode)
  elif env_name in ('fetch_reach', 'fetch_push'):
    env = FetchEnv(task=env_name, max_episode_steps=50, seed=seed,
                   render_mode=render_mode)
  elif env_name == 'fetch_push_start_at_obj':
    env = FetchEnv(task='fetch_push_start_at_obj', start_at_obj=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_original_style':
    env = FetchEnv(task='fetch_push_original_style', original_style=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_easy_fixed':
    env = FetchEnv(task='fetch_push_easy_fixed', easy_fixed=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_easy_multidir':
    env = FetchEnv(task='fetch_push_easy_multidir', multidir=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_easy_neutral_dir':
    env = FetchEnv(task='fetch_push_easy_neutral_dir', neutral_dir=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_easy_conedir':
    env = FetchEnv(task='fetch_push_easy_conedir', cone_dir=True,
                   max_episode_steps=50, seed=seed, render_mode=render_mode)
  elif env_name == 'fetch_push_image_conedir':
    env = FetchEnv(task='fetch_push_image_conedir', cone_dir=True,
                   image_obs=True, max_episode_steps=50, seed=seed,
                   render_mode=render_mode)
  else:
    raise NotImplementedError(f'Unknown env: {env_name}')

  config.obs_dim = env.obs_dim
  config.goal_dim = env.goal_dim
  config.action_dim = env.action_dim
  config.max_episode_steps = env.max_episode_steps
  config.start_index = env.start_index
  config.end_index = env.end_index
  config.goal_indices = getattr(env, 'goal_indices', None)
  if getattr(env, 'use_image_obs', False):
    config.use_image_obs = True
  return env
