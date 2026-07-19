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


# ---------------------------------------------------------------------------
# Litter corridor: hidden one-sided obstacle ("blind ant in a cluttered
# corridor"). Confounded variant of the OFFLINE task above.
# ---------------------------------------------------------------------------

#: Obstacle zone in the bottom corridor (y ~ 0, interior |y| < 2): the
#: straight segment before the right turn at (8, 0). World-frame x-range.
LITTER_ZONE_X = (2.5, 5.5)
#: Main pile |y| band on the active side (skirt edge to wall).
LITTER_PILE_Y = (0.7, 2.0)
LITTER_PILE_HEIGHT = 1.0        # tall: not traversable (walls are 2.0)
#: Skirt: rubble spilled from the pile toward the centerline, on the SAME
#: side as the pile (it moves with U -- both mirrored copies are compiled in
#: and the inactive side is buried with its pile). Height tapers UP toward
#: the pile: the centerline edge is steppable at low speed, deeper skirt
#: trips a fast ant. The opposite half-corridor stays completely clean, so
#: the clean lane is ~2.0 wide (an ant's gait wander fits comfortably).
LITTER_SKIRT_Y = (0.05, 0.75)   # |y| band of skirt box centers
LITTER_SKIRT_N = 14
LITTER_SKIRT_HALF_XY = (0.10, 0.22)    # half-extent range per box
LITTER_SKIRT_H0 = (0.10, 0.16)  # height range at the centerline edge
LITTER_SKIRT_H1 = (0.16, 0.34)  # height range at the pile edge
#: Slick strip ("leachate" seeping from the pile): a flush, low-friction
#: plate under the skirt band, same side as the pile. Friction -- unlike box
#: height -- is intrinsically speed-sensitive: fast push-off exceeds the
#: friction cone and the ant sprawls; slow careful steps hold. priority=1
#: makes the contact pair use the slick's friction against the ant's feet.
LITTER_SLICK_Y = (0.0, 0.9)     # |y| band of the slick plate
LITTER_SLICK_H = 0.02           # flush: never trips by geometry
LITTER_SLICK_FRICTION = '0.08 0.005 0.0001'
#: Collapse semantics: the pile is UNSTABLE. A contact with any pile/skirt
#: geom whose normal force exceeds this threshold buries the ant -- it is
#: dead (absorbing) for the rest of the episode: physics frozen, actions
#: ignored, reward can never fire, done stays False (fixed-length episode
#: contract, exactly the windy-lethal freeze pattern). Slow, low-impact
#: contact (careful stepping/grinding) stays safe. Forces are checked every
#: SUBSTEP so frame_skip cannot swallow an impact spike. The slick plate is
#: excluded (liquid cannot collapse; weight-bearing on it is harmless).
#: None disables collapse (calibration mode). Value set by the Stage-1
#: calibration run (see artifacts/litter_env/collapse_calibration.json).
LITTER_COLLAPSE_FORCE = None    # placeholder until calibrated
#: Skirt layout is generated ONCE from this seed and mirrored exactly, so
#: the two U configurations are geometrically identical up to reflection --
#: frozen across episodes, env instances and env seeds (a second hidden
#: variable would pollute both the hiddenness gate and attribution).
LITTER_LAYOUT_SEED = 7
LITTER_HIDE_Z = -10.0           # buried z for the inactive side


def build_litter_xml(maze_map=U_MAZE, scaling=SCALING, height=MAZE_HEIGHT):
  """Maze xml + litter geoms.

  Adds, per corridor side, one main-pile box plus LITTER_SKIRT_N skirt boxes
  (exact mirror images between sides; per episode the inactive side is buried
  below the floor). Returns (xml_string, torso_offset, layout) where layout
  lists every litter geom with its side ('pos'/'neg') and geometry.
  """
  xml, offset = build_maze_xml(maze_map, scaling, height)
  root = ET.fromstring(xml)
  worldbody = root.find('.//worldbody')
  x0, x1 = LITTER_ZONE_X
  cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
  y0, y1 = LITTER_PILE_Y
  cy, hy = (y0 + y1) / 2, (y1 - y0) / 2
  hz = LITTER_PILE_HEIGHT / 2
  layout = []
  for side, sign in (('pos', 1.0), ('neg', -1.0)):
    ET.SubElement(worldbody, 'geom', name=f'litter_pile_{side}', type='box',
                  pos=f'{cx} {sign * cy} {hz}', size=f'{hx} {hy} {hz}',
                  material='', contype='1', conaffinity='1',
                  rgba='0.35 0.55 0.3 1.0')
    layout.append({'name': f'litter_pile_{side}', 'side': side, 'x': cx,
                   'y': sign * cy, 'half_x': hx, 'half_y': hy,
                   'height': LITTER_PILE_HEIGHT, 'yaw_deg': 0.0})
  k0, k1 = LITTER_SLICK_Y
  kc, kh = (k0 + k1) / 2, (k1 - k0) / 2
  for side, sign in (('pos', 1.0), ('neg', -1.0)):
    tag = f'litter_slick_{side}'
    ET.SubElement(worldbody, 'geom', name=tag, type='box',
                  pos=f'{cx} {sign * kc} {LITTER_SLICK_H / 2}',
                  size=f'{hx} {kh} {LITTER_SLICK_H / 2}',
                  friction=LITTER_SLICK_FRICTION, priority='1',
                  material='', contype='1', conaffinity='1',
                  rgba='0.25 0.25 0.3 0.9')
    layout.append({'name': tag, 'side': side, 'x': cx, 'y': sign * kc,
                   'half_x': hx, 'half_y': kh, 'height': LITTER_SLICK_H,
                   'yaw_deg': 0.0})
  rng = np.random.default_rng(LITTER_LAYOUT_SEED)
  s0, s1 = LITTER_SKIRT_Y
  for i in range(LITTER_SKIRT_N):
    sx, sy = rng.uniform(*LITTER_SKIRT_HALF_XY, size=2)
    rx = rng.uniform(x0 + sx, x1 - sx)
    ry = rng.uniform(s0, s1)
    frac = (ry - s0) / (s1 - s0)       # 0 at centerline edge, 1 at pile edge
    h0 = rng.uniform(*LITTER_SKIRT_H0)
    h1 = rng.uniform(*LITTER_SKIRT_H1)
    rh = h0 + frac * (h1 - h0)
    yaw = rng.uniform(0.0, 180.0)      # compiler angle="degree"
    for side, sign in (('pos', 1.0), ('neg', -1.0)):
      tag = f'litter_skirt_{i}_{side}'
      ET.SubElement(worldbody, 'geom', name=tag, type='box',
                    pos=f'{rx} {sign * ry} {rh / 2}',
                    size=f'{sx} {sy} {rh / 2}', euler=f'0 0 {sign * yaw}',
                    material='', contype='1', conaffinity='1',
                    rgba='0.45 0.4 0.35 1.0')
      layout.append({'name': tag, 'side': side, 'x': float(rx),
                     'y': float(sign * ry), 'half_x': float(sx),
                     'half_y': float(sy), 'height': float(rh),
                     'yaw_deg': float(sign * yaw)})
  return ET.tostring(root, encoding='unicode'), offset, layout


class LitterOfflineAntUMazeEnv(OfflineD4rlAntUMazeEnv):
  """Confounded 'cluttered corridor' variant of the offline umaze task.

  Hidden episode variable ``U`` in {0, 1}, P = 0.5 each, sampled ONCE per
  reset from an rng stream independent of the reset-noise / goal streams:
    * U = 1: litter (pile + spilled skirt) on the +y side of the corridor;
    * U = 0: exact mirror image on the -y side.
  The skirt tapers down toward the centerline (steppable at low speed near
  y=0, tripping height near the pile); the opposite half-corridor is clean.
  The 58-dim learner obs is UNCHANGED (pure proprioception + zero-padded
  goal): U leaves no trace in the observation until contact.

  Teacher code reads ``privileged_u``; step() reports diagnostics
  (u_side, pile/rubble contact counts at the post-step state, in_zone) in
  the info dict -- sidecar only, NEVER part of the observation.
  """

  def __init__(self, max_episode_steps=700, seed=0, render_mode=None,
               eval_goals=None, eval_goal_mode='d4rl'):
    super().__init__(max_episode_steps=max_episode_steps, seed=seed,
                     render_mode=render_mode, eval_goals=eval_goals,
                     eval_goal_mode=eval_goal_mode)
    xml, offset, layout = build_litter_xml()
    self._env = _Sim(xml, seed)          # replace sim with the litter model
    self._torso_offset = offset          # same maze -> same origin
    self.litter_layout = layout
    m = self._env.model
    def gid(n):
      return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    side_u = {'pos': 1, 'neg': 0}
    self._side_gids = {1: [], 0: []}     # every litter geom, grouped by side
    for item in layout:
      self._side_gids[side_u[item['side']]].append(gid(item['name']))
    self._pile_gid = {u: gid(f'litter_pile_{s}')
                      for s, u in side_u.items()}
    self._skirt_gids = {u: frozenset(g for g in gids
                                     if g != self._pile_gid[u])
                        for u, gids in self._side_gids.items()}
    self._home_pos = {g: m.geom_pos[g].copy()
                      for gids in self._side_gids.values() for g in gids}
    self._slick_gids = frozenset(
        gid(f'litter_slick_{s}') for s in ('pos', 'neg'))
    #: geoms whose hard impact collapses the pile (slick excluded)
    self._collapsible_gids = {
        u: frozenset(g for g in gids if g not in self._slick_gids)
        for u, gids in self._side_gids.items()}
    self._u_rng = np.random.default_rng(seed + 20260719)
    self.u_side = None
    self.collapse_force = LITTER_COLLAPSE_FORCE
    self._dead = False
    self._force_buf = np.zeros(6)
    self.episode_contacts = {'pile': 0, 'rubble': 0}
    self.episode_max_force = 0.0
    self.episode_max_hforce = 0.0
    self.episode_max_himpulse = 0.0

  @property
  def dead(self):
    """Buried by a collapsed pile: absorbing until episode end."""
    return self._dead

  @property
  def privileged_u(self):
    """Teacher-only obstacle-side bit. Learners must never read this."""
    return self.u_side

  def zone_info(self):
    """Geometry handles for waypoint controllers / probes."""
    clean_sign = -1.0 if self.u_side == 1 else 1.0
    return {'zone_x': LITTER_ZONE_X,
            'clean_lane_y': clean_sign * 1.05,
            'middle_lane_y': 0.0,
            'pile_y_band': LITTER_PILE_Y,
            'skirt_y_band': LITTER_SKIRT_Y}

  def _apply_u(self, u_side):
    m = self._env.model
    for u, gids in self._side_gids.items():
      for g in gids:
        m.geom_pos[g] = self._home_pos[g]
        if u != u_side:
          m.geom_pos[g][2] = LITTER_HIDE_Z
    self.u_side = int(u_side)

  def reset(self, u_side=None):
    """``u_side`` override is for probes/gates only; normal use samples U."""
    self._apply_u(self._u_rng.integers(2) if u_side is None else u_side)
    self.episode_contacts = {'pile': 0, 'rubble': 0}
    self.episode_max_force = 0.0
    self.episode_max_hforce = 0.0
    self.episode_max_himpulse = 0.0
    self._dead = False
    return super().reset()               # mj_forward runs after _apply_u

  def _count_litter_contacts(self):
    d = self._env.data
    active_pile = self._pile_gid[self.u_side]
    active_skirt = self._skirt_gids[self.u_side]
    pile = rubble = 0
    for i in range(d.ncon):
      for g in (d.contact[i].geom1, d.contact[i].geom2):
        if g == active_pile:
          pile += 1
        elif g in active_skirt:
          rubble += 1
    return pile, rubble

  def _litter_contact_forces(self):
    """Contact-force stats against collapsible litter geoms at the CURRENT
    physics state (call after each substep).

    Returns (max_normal, max_horizontal, normal_z_at_max_horizontal).
    ``horizontal`` projects the normal force onto the ground plane: a foot
    bearing weight ON TOP of low rubble has a near-vertical contact normal
    (harmless), while a body/leg ramming the SIDE of the litter has a normal
    with a large horizontal component -- only the latter can "collapse" the
    pile. contact.frame[:3] is the contact normal in world coordinates."""
    d = self._env.data
    active = self._collapsible_gids[self.u_side]
    fmax = hmax = 0.0
    nz_at_hmax = 1.0
    for i in range(d.ncon):
      c = d.contact[i]
      if c.geom1 in active or c.geom2 in active:
        mujoco.mj_contactForce(self._env.model, d, i, self._force_buf)
        f = abs(float(self._force_buf[0]))
        n = c.frame[:3]
        h = f * float(np.hypot(n[0], n[1]))
        if f > fmax:
          fmax = f
        if h > hmax:
          hmax = h
          nz_at_hmax = abs(float(n[2]))
    return fmax, hmax, nz_at_hmax

  def _info(self, pile, rubble, stats):
    xy = np.asarray(self._last_obs['achieved_goal'])
    out = {'u_side': self.u_side, 'pile_contacts': pile,
           'rubble_contacts': rubble, 'dead': self._dead,
           'litter_force': stats.get('max_litter_normal_force', 0.0),
           'in_zone': bool(LITTER_ZONE_X[0] <= xy[0] <= LITTER_ZONE_X[1]
                           and abs(xy[1]) < 2.0)}
    out.update(stats)
    return out

  _ZERO_STATS = {'max_litter_normal_force': 0.0,
                 'max_horizontal_normal_force': 0.0,
                 'max_horizontal_impulse': 0.0,
                 'contact_normal_z': 1.0,
                 'precontact_planar_speed': 0.0}

  def step(self, action):
    if self._dead:                       # buried: frozen until episode end
      return (self._flatten(self._last_obs), 0.0, False,
              self._info(0, 0, dict(self._ZERO_STATS)))
    u = self._env
    a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
    u.data.ctrl[:] = a * CTRL_SCALE
    pre_speed = float(np.hypot(u.data.qvel[0], u.data.qvel[1]))
    sub_dt = u.model.opt.timestep
    fmax = hmax = himp = 0.0
    nz = 1.0
    for _ in range(u.frame_skip):        # substep force checks: frame_skip
      mujoco.mj_step(u.model, u.data)    # must not swallow an impact spike
      f, h, nz_h = self._litter_contact_forces()
      himp += h * sub_dt                 # horizontal impulse over the step
      if f > fmax:
        fmax = f
      if h > hmax:
        hmax, nz = h, nz_h
    stats = {'max_litter_normal_force': fmax,
             'max_horizontal_normal_force': hmax,
             'max_horizontal_impulse': himp,
             'contact_normal_z': nz,
             'precontact_planar_speed': pre_speed}
    self._last_obs = u._obs_dict()
    self.episode_max_force = max(self.episode_max_force, fmax)
    self.episode_max_hforce = max(self.episode_max_hforce, hmax)
    self.episode_max_himpulse = max(self.episode_max_himpulse, himp)
    #: collapse trigger uses the HORIZONTAL normal force: weight bearing on
    #: top of rubble (vertical normal) can never bury the ant.
    if (self.collapse_force is not None
        and hmax > self.collapse_force):
      self._dead = True                  # pile collapsed onto the ant
    pile, rubble = self._count_litter_contacts()
    self.episode_contacts['pile'] += pile
    self.episode_contacts['rubble'] += rubble
    if self._dead:
      reward = 0.0
    else:
      dist = float(np.linalg.norm(np.asarray(self._last_obs['achieved_goal'])
                                  - np.asarray(u.goal)))
      reward = float(dist <= SUCCESS_DIST)
    return (self._flatten(self._last_obs), reward, False,
            self._info(pile, rubble, stats))
