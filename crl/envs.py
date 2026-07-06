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
  }

  def __init__(self, task='fetch_reach', max_episode_steps=50, seed=0,
               render_mode=None, start_at_obj=False, original_style=False,
               easy_fixed=False, multidir=False):
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  (registers the envs)
    gym.register_envs(gymnasium_robotics)

    self._start_at_obj = start_at_obj
    self._original_style = original_style
    self._easy_fixed = easy_fixed
    self._multidir = multidir
    self._last_dir = np.array([1.0, 0.0, 0.0])   # goal direction (audit hook)
    self._rng = np.random.default_rng(seed)
    # EasyPush L1 geometry: object pinned +x of the gripper home (~1.363) so the
    # gripper starts BEHIND it; goal a short +x push away (6-9 cm). Both on table.
    self._obj0 = np.array([1.40, 0.75, 0.425])
    self._push_range = (0.06, 0.09)
    self.start_index, self.end_index, ids = self._SPECS[task]
    last_err = None
    for env_id in ids:
      try:
        self._env = gym.make(env_id, max_episode_steps=max_episode_steps,
                             render_mode=render_mode)
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

  def reset(self):
    obs, _ = self._env.reset()
    if self._multidir:
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
    reward = float(dist < self._success_threshold)
    return self._flatten(obs), reward, False, {}

  def render(self):
    """Returns an RGB frame (needs render_mode='rgb_array' + MUJOCO_GL)."""
    return self._env.render()


# ---------------------------------------------------------------------------
def make_env(env_name, config, seed=0, render_mode=None):
  """Builds an env and fills obs/goal/action dims + episode length into config.

  Returns the env; mutates ``config`` in place with obs_dim, goal_dim,
  action_dim, max_episode_steps, start_index, end_index.
  ``render_mode='rgb_array'`` enables Fetch frame rendering (for GIFs on Colab).
  """
  if env_name.startswith('point_'):
    walls = env_name.split('_', 1)[1]
    # Larger maps get longer horizons (mirrors env_utils.load heuristics).
    steps = 100 if walls in ('FourRooms', 'Maze11x11', 'Spiral11x11') else 50
    env = PointEnv(walls=walls, max_episode_steps=steps, seed=seed)
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
  else:
    raise NotImplementedError(f'Unknown env: {env_name}')

  config.obs_dim = env.obs_dim
  config.goal_dim = env.goal_dim
  config.action_dim = env.action_dim
  config.max_episode_steps = env.max_episode_steps
  config.start_index = env.start_index
  config.end_index = env.end_index
  return env
