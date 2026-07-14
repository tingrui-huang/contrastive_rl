"""D4RL-faithful AntMaze (separate reproduction branch; no gymnasium-robotics).

Reimplements the ORIGINAL contrastive_rl ant_umaze task as closely as possible
without the dead d4rl dependency, from the exact upstream sources:

  * physics: the verbatim d4rl/locomotion/assets/ant.xml (vendored at
    crl/assets/d4rl_ant.xml): timestep 0.02, integrator RK4, actuator
    ctrlrange +-30 gear 1, armature 1, damping 1, init z 0.55. Actions in
    [-1, 1] are scaled to ctrlrange (acme CanonicalSpecWrapper semantics):
    ctrl = 30 * clip(a, -1, 1). frame_skip = 5 -> dt = 0.1 s per env step
    (2x the gymnasium branch).
  * maze: d4rl/locomotion/maze_env.py wall construction -- one box geom of
    half-extents (S/2, S/2, H*S/2) at z = H*S/2 per wall cell, S = 4.0,
    H = 0.5; world origin at the R cell.
  * task: ant_envs.py AntMaze wrapper -- U_MAZE map, non_zero_reset (50%
    free-cell reset / 50% goal-sampler reset), goal sampled from G cells
    with +-0.25*S noise, episode length 700, done always False, reward =
    (dist(xy, settled_goal_xy) <= 0.5).
  * goal: FULL 29-dim settled goal observation -- the ant is teleported to
    the goal xy (keeping its post-reset pose/velocities), settled for 50
    zero-action env steps, and the snapshot [qpos, qvel] is the commanded
    goal (original obs_to_goal with start=0/end=-1 => relabeled goals are
    full future states via goal_indices = range(29)).

Known, documented approximations:
  * d4rl weighted the reset-cell distribution by accessibility; we sample
    open cells uniformly.
  * single-process collection (1 actor) instead of 4 async actors.

The wrapper exposes the same interface as crl.envs.MazeEnv (obs_dim,
goal_dim, goal_indices, _flatten, _env.unwrapped with .model/.data/.goal/
.step) so every existing gate/probe/training script works unchanged.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                     'd4rl_ant.xml')

R, G = 'r', 'g'
U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]
SCALING = 4.0
MAZE_HEIGHT = 0.5
INIT_QPOS = np.array([0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
FRAME_SKIP = 5
CTRL_SCALE = 30.0
SUCCESS_DIST = 0.5


def build_maze_xml(maze_map=U_MAZE, scaling=SCALING, height=MAZE_HEIGHT):
  """Vendored ant.xml + d4rl maze_env.py wall geoms (origin at the R cell)."""
  tree = ET.parse(ASSET)
  worldbody = tree.find('.//worldbody')
  rr, cc = next((r, c) for r in range(len(maze_map))
                for c in range(len(maze_map[0])) if maze_map[r][c] == R)
  tx, ty = cc * scaling, rr * scaling
  for r in range(len(maze_map)):
    for c in range(len(maze_map[0])):
      if maze_map[r][c] == 1:
        ET.SubElement(
            worldbody, 'geom', name=f'block_{r}_{c}', type='box',
            pos=f'{c * scaling - tx} {r * scaling - ty} '
                f'{height / 2 * scaling}',
            size=f'{scaling / 2} {scaling / 2} {height / 2 * scaling}',
            material='', contype='1', conaffinity='1',
            rgba='0.7 0.5 0.3 1.0')
  return ET.tostring(tree.getroot(), encoding='unicode'), (tx, ty)


class _Sim:
  """Minimal mujoco core mimicking the probe-facing gym unwrapped surface."""

  def __init__(self, xml, seed):
    self.model = mujoco.MjModel.from_xml_string(xml)
    self.data = mujoco.MjData(self.model)
    self.frame_skip = FRAME_SKIP
    self.goal = np.zeros(2)
    self._rng = np.random.default_rng(seed)

  @property
  def unwrapped(self):
    return self

  def reset_model(self):
    qpos = INIT_QPOS + self._rng.uniform(-0.1, 0.1, self.model.nq)
    qvel = self._rng.standard_normal(self.model.nv) * 0.1
    self.data.qpos[:] = qpos
    self.data.qvel[:] = qvel
    mujoco.mj_forward(self.model, self.data)

  def _obs_dict(self):
    qpos = np.asarray(self.data.qpos)
    qvel = np.asarray(self.data.qvel)
    return {'achieved_goal': qpos[:2].copy(),
            'observation': np.concatenate([qpos[2:], qvel]),
            'desired_goal': np.asarray(self.goal, float).copy()}

  def do_simulation(self, ctrl, n):
    self.data.ctrl[:] = ctrl
    for _ in range(n):
      mujoco.mj_step(self.model, self.data)

  def step(self, action):
    a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
    self.do_simulation(a * CTRL_SCALE, self.frame_skip)
    return self._obs_dict(), 0.0, False, False, {}

  def reset(self, seed=None):
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    self.reset_model()
    return self._obs_dict(), {}


class D4rlAntUMazeEnv:
  """Original-task wrapper: flat obs = [state(29), settled_goal_state(29)]."""

  start_index = 0
  end_index = -1

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None):
    del render_mode
    xml, (tx, ty) = build_maze_xml()
    self._env = _Sim(xml, seed)
    self._torso_offset = (tx, ty)
    self.max_episode_steps = max_episode_steps
    self.obs_dim = 29
    self.goal_indices = tuple(range(29))
    self.goal_dim = 29
    self.action_dim = 8
    self._rng = np.random.default_rng(seed + 777)
    self._goal_vec = np.zeros(29, np.float32)
    self._goal_state_full = np.zeros(29, np.float32)
    self._open, self._goal_cells = [], []
    for r in range(len(U_MAZE)):
      for c in range(len(U_MAZE[0])):
        if U_MAZE[r][c] in (R, G, 0):
          self._open.append((r, c))
        if U_MAZE[r][c] == G:
          self._goal_cells.append((r, c))
    self._last_obs = None

  def _cell_xy(self, rc):
    tx, ty = self._torso_offset
    return np.array([rc[1] * SCALING - tx, rc[0] * SCALING - ty])

  def _sample_goal_xy(self):
    rc = self._goal_cells[self._rng.integers(len(self._goal_cells))]
    return self._cell_xy(rc) + self._rng.uniform(-0.25 * SCALING,
                                                 0.25 * SCALING, 2)

  def _sample_reset_xy(self):
    if self._rng.random() < 0.5:            # non_zero_reset: goal-sampler half
      return self._sample_goal_xy()
    rc = self._open[self._rng.integers(len(self._open))]
    return self._cell_xy(rc) + self._rng.uniform(-0.25 * SCALING,
                                                 0.25 * SCALING, 2)

  def _flatten(self, obs):
    state = np.concatenate([np.asarray(obs['achieved_goal']),
                            np.asarray(obs['observation'])])
    return np.concatenate([state, self._goal_vec]).astype(np.float32)

  def reset(self):
    u = self._env
    u.reset_model()
    u.data.qpos[:2] = self._sample_reset_xy()
    mujoco.mj_forward(u.model, u.data)
    qpos0 = np.asarray(u.data.qpos).copy()
    qvel0 = np.asarray(u.data.qvel).copy()
    # settle the ant at the goal xy for the full-state commanded goal
    u.data.qpos[:2] = self._sample_goal_xy()
    mujoco.mj_forward(u.model, u.data)
    for _ in range(50):
      u.do_simulation(np.zeros(8), u.frame_skip)
    self._goal_state_full = np.concatenate(
        [np.asarray(u.data.qpos), np.asarray(u.data.qvel)]).astype(np.float32)
    self._goal_vec = self._goal_state_full.copy()
    u.goal = self._goal_state_full[:2].astype(float).copy()  # settled xy
    u.data.qpos[:] = qpos0
    u.data.qvel[:] = qvel0
    mujoco.mj_forward(u.model, u.data)
    obs = u._obs_dict()
    self._last_obs = obs
    return self._flatten(obs)

  def step(self, action):
    obs = self._env.step(action)[0]
    self._last_obs = obs
    dist = float(np.linalg.norm(np.asarray(obs['achieved_goal'])
                                - np.asarray(self._env.goal)))
    reward = float(dist <= SUCCESS_DIST)
    return self._flatten(obs), reward, False, {}

  def render(self):
    raise NotImplementedError


#: d4rl U_MAZE goal cell for antmaze-umaze (the SINGLE 'G' in the upstream
#: collection/eval maze, d4rl/locomotion/maze_env.py U_MAZE). In our frame
#: (R at cell (1,1) -> origin) this is xy (0, 8): the interior of the U's far
#: corner where every good trajectory ends.
D4RL_EVAL_GOAL_CELL = (3, 1)


class OfflineD4rlAntUMazeEnv(D4rlAntUMazeEnv):
  """OFFLINE antmaze-umaze contract (upstream OfflineAntWrapper, ant_env.py).

  Differences from the online task above:
    * goal observation = ``zeros(29)`` with ``[:2] = goal xy`` -- the exact
      zero-padded goal the original offline experiments trained/evaluated
      with. NO settled full-state goal, no goal-settling simulation.
    * reset at the R cell only (the dataset's single start cell): plain
      d4rl reset noise around INIT_QPOS, no non_zero_reset teleport.

  Evaluation goal source (``eval_goal_mode``):
    * ``'d4rl'`` (DEFAULT, the benchmark protocol): the exact d4rl
      ``goal_sampler`` on the single U_MAZE goal cell (3,1) -- cell xy
      ``(0, 8)`` plus per-coordinate noise ``U(0, 0.25*S) + U(0, 0.5)*0.25*S``
      (S = scaling = 4 -> noise in [0, 1.5]), resampled every episode
      (matches d4rl v2_resets). This is what the paper / D4RL score report.
    * ``'dataset'``: replay the EMPIRICAL per-episode dataset ``infos/goal``
      (``eval_goals`` from the .npz). NOTE these were collected with ~2x the
      benchmark noise (mean (1.5, 9.5) vs the benchmark's (0.75, 8.75)), so
      this distribution is FARTHER into the maze and materially HARDER than
      the standard eval -- kept only for provenance/comparison.
    * ``'fixed'``: a single fixed goal = cell(3,1) + mean noise (0.75, 8.75),
      held constant across episodes.

  Everything else (physics, action scaling, 700-step horizon, reward =
  dist(xy, goal_xy) <= 0.5, goal_indices=range(29) relabeling contract)
  is inherited unchanged.
  """

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl'):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode)
    self._eval_goals = (None if eval_goals is None
                        else np.asarray(eval_goals, np.float32))
    if eval_goal_mode not in ('d4rl', 'dataset', 'fixed'):
      raise ValueError(f'unknown eval_goal_mode {eval_goal_mode!r}')
    if eval_goal_mode == 'dataset' and self._eval_goals is None:
      raise ValueError("eval_goal_mode='dataset' needs an eval_goals array")
    self.eval_goal_mode = eval_goal_mode
    self._eval_goal_cell_xy = self._cell_xy(D4RL_EVAL_GOAL_CELL)  # (0, 8)

  def _d4rl_goal_sampler(self):
    """Verbatim d4rl goal_sampler noise on the single U_MAZE goal cell."""
    base = self._eval_goal_cell_xy
    noise = (self._rng.uniform(0.0, 0.25 * SCALING, 2)
             + self._rng.uniform(0.0, 0.5, 2) * 0.25 * SCALING)
    return np.maximum(base + noise, 0.0).astype(np.float32)

  def _eval_goal_xy(self):
    if self.eval_goal_mode == 'd4rl':
      return self._d4rl_goal_sampler()
    if self.eval_goal_mode == 'fixed':
      # cell(3,1) + MEAN d4rl noise: E[U(0,0.25S)+U(0,0.5)*0.25S] = 0.1875*S
      # per coord = (0.75, 0.75) at S=4 -> the canonical (0.75, 8.75).
      return (self._eval_goal_cell_xy
              + np.array([0.1875 * SCALING, 0.1875 * SCALING])).astype(np.float32)
    return self._eval_goals[self._rng.integers(len(self._eval_goals))]

  def reset(self):
    u = self._env
    u.reset_model()                      # INIT_QPOS +-0.1 noise at the R cell
    mujoco.mj_forward(u.model, u.data)
    gxy = self._eval_goal_xy()
    self._goal_vec = np.zeros(29, np.float32)
    self._goal_vec[:2] = gxy             # zero-padded XY goal contract
    self._goal_state_full = self._goal_vec.copy()
    u.goal = np.asarray(gxy, float).copy()
    obs = u._obs_dict()
    self._last_obs = obs
    return self._flatten(obs)
