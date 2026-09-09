r"""Top-down GIFs of the V6 long two-rockfall benchmark.

Seven clips, all from the unmodified env; only the behaviour policy differs.

EXPERT (the privileged sighted teacher, which reads env.schedule and makes one
local go/wait decision per zone):

  expert_clear_go      U1=0, U2=0   -- both zones disarmed, walks straight
                                       through the shortcut
  expert_wait_zone1    U1=1, U2=0   -- the zone-1 burst overlaps its crossing,
                                       so it HOLDS at the zone-1 mouth until
                                       the rocks are parked, then crosses both
  expert_wait_zone2    U1=0, U2=1   -- zone 1 is clear and it walks through,
                                       then holds at the zone-2 mouth
  expert_detour        route coin   -- the long safe perimeter route: north
                                       column, top row, east column. It never
                                       consults either schedule and never
                                       enters a band; the rocks fall on their
                                       own clock while the ant is elsewhere.

AGENT (a trained checkpoint, deterministic tanh(mu), natural U/t0 draws --
the latents are NEVER forced for the agent clips, so what you see is what the
policy meets on its own):

  agent_success        no rock ever touches it
  agent_death_zone1    killed by the zone-1 rockfall
  agent_death_zone2    killed by the zone-2 rockfall

Agent clips are found by replaying the eval seed's episode sequence and
recording the first episode of each outcome, so they are real eval episodes
rather than staged ones. The episode index of each is printed and written into
the caption.

Overlay: goal ring, both mouth lines (yellow), both band footprints (red), and
a caption carrying t, both latents, both teacher decisions (normal vs executed
where they differ), the hold state, the route and both schedules.

Usage:
  python scripts/render_rockfall_clock_v6_gifs.py                    # expert only
  python scripts/render_rockfall_clock_v6_gifs.py --ckpt <final.pkl> # + agent
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                  # noqa: E402
from crl import rockfall_clock_v6 as V6           # noqa: E402
import rockfall_clock_v6_teacher as CT            # noqa: E402

OUT = os.path.join(CT.OUT, 'gifs')
EVERY, FPS = 4, 20
#: the V6 maze is long and shallow (x about -6..30, y about -6..14), so the
#: frame is wide and the camera sits high enough to hold the whole ring.
W, H = 880, 400
#: lookat is the maze centre (x -6..30, y -6..14) and the distance is the one
#: that makes the 20-unit y span fill the frame at fovy 45: half height =
#: d*tan(22.5) = 11.0 world units, i.e. 18.15 px per unit. Verified by drawing
#: crosshairs at the start, the goal and both mouth lines and checking they
#: land on the ant and the corridor (artifacts/rockfall_clock_v6/gifs/
#: _calib.png).
CAM = dict(lookat=(12.0, 4.0, 0.0), distance=26.6, elevation=-90.0,
           azimuth=90.0)
GOAL_R = 0.5
BAR = 34


class Recorder:
  """Top-down frames with the geometry and the privileged state drawn on."""

  def __init__(self, env, title):
    self.env, self.title = env, title
    #: the default offscreen framebuffer is 640x480; this maze is 36 world
    #: units wide, so it needs a wide frame. Raising the model's offscreen
    #: size before the renderer builds its GL context is the documented way.
    model = env._env.model
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), H)
    self.renderer = mujoco.Renderer(model, H, W)
    self.cam = mujoco.MjvCamera()
    self.cam.lookat[:] = CAM['lookat']
    self.cam.distance = CAM['distance']
    self.cam.elevation = CAM['elevation']
    self.cam.azimuth = CAM['azimuth']
    self.frames = []
    self._px = None

  def _calibrate(self):
    """world (x, y) -> pixel, from the camera geometry rather than by hand."""
    fovy = float(self.env._env.model.vis.global_.fovy)
    half_h = CAM['distance'] * np.tan(np.radians(fovy / 2.0))
    scale = (H / 2.0) / half_h                     # pixels per world unit
    cx, cy = CAM['lookat'][0], CAM['lookat'][1]

    def to_px(x, y):
      return (W / 2.0 + (x - cx) * scale, H / 2.0 - (y - cy) * scale)

    self._px = (to_px, scale)

  def grab(self, t, goal_xy, u1, u2, status='', sched=''):
    if self._px is None:
      self._calibrate()
    to_px, scale = self._px
    self.renderer.update_scene(self.env._env.data, camera=self.cam)
    im = Image.fromarray(self.renderer.render().copy())
    dr = ImageDraw.Draw(im)
    #: both hazard bands, coloured by whether that zone is armed this episode
    for zone, armed in ((1, u1), (2, u2)):
      x0, x1, y0, y1 = V6.hazard_zone(zone)
      a = to_px(x0, y1)
      b = to_px(x1, y0)
      dr.rectangle([a[0], a[1], b[0], b[1]],
                   outline=(230, 40, 40) if armed else (120, 90, 90), width=2)
      mx = to_px(V6.MOUTH_X[zone], 0)[0]
      dr.line([mx, to_px(0, y0)[1], mx, to_px(0, y1)[1]],
              fill=(240, 220, 60), width=2)
      dr.text((a[0] + 3, a[1] - 12), 'Z%d' % zone,
              fill=(230, 40, 40) if armed else (150, 120, 120))
    gx, gy = to_px(*goal_xy)
    r = GOAL_R * scale
    dr.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(60, 220, 60), width=3)
    dr.rectangle([0, 0, W, BAR], fill=(0, 0, 0))
    dr.text((6, 3), self.title, fill=(255, 255, 255))
    dr.text((6, 17),
            'U1=%s U2=%s  t=%3d  %s' % ('ON ' if u1 else 'off',
                                        'ON ' if u2 else 'off', t, status),
            fill=(255, 220, 120) if (u1 or u2) else (180, 255, 180))
    if sched:
      dr.text((W - 350, 17), sched, fill=(170, 190, 255))
    self.frames.append(np.asarray(im))

  def save(self, path):
    frames = self.frames + [self.frames[-1]] * FPS
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    print('  %-46s %4d frames' % (path, len(self.frames)), flush=True)
    self.renderer.close()


def outcome(info):
  if info.get('success'):
    return 'SUCCESS'
  if info.get('failure'):
    return 'DEAD (zone %s)' % info.get('failure_zone')
  return 'timeout'


def sched_text(env):
  bits = []
  for zone in (1, 2):
    start = env.privileged_rockfall_start(zone)
    if start is None:
      bits.append('Z%d -' % zone)
    else:
      bits.append('Z%d %d..%d' % (zone, start,
                                  env.privileged_rockfall_end(zone)))
  return '  '.join(bits)


def teacher_status(teacher):
  """What the teacher decided, and what it is executing, per zone."""
  normal, executed = teacher.normal_decisions, teacher.executed_decisions
  bits = []
  for zone in (1, 2):
    n, e = normal[zone], executed[zone]
    if n is None and e is None:
      continue
    bits.append('Z%d:%s' % (zone, e if n == e or n is None
                            else '%s->%s' % (n, e)))
  hold = ('HOLD@Z%d' % teacher.holding_zone) if teacher.holding_zone else ''
  return ' '.join(bits + [hold, 'route=%s' % teacher.route]).strip()


def expert_clips(out, seed, horizon):
  """The four expert clips. Latents are forced so each clip shows its case."""
  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = horizon
  cfg.max_episode_steps = horizon
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed)
  specs = (
      ('expert_clear_go', False, False, 'shortcut',
       'EXPERT  both zones clear  ->  straight through'),
      ('expert_wait_zone1', True, False, 'shortcut',
       'EXPERT  zone 1 armed  ->  holds at the Z1 mouth, then crosses'),
      ('expert_wait_zone2', False, True, 'shortcut',
       'EXPERT  zone 2 armed  ->  crosses Z1, holds at the Z2 mouth'),
      ('expert_detour', True, True, 'detour',
       'EXPERT  route coin = detour  ->  long safe perimeter, no band'),
  )
  for name, u1, u2, route, title in specs:
    rec = Recorder(env, title)
    o = env.reset(rockfall_active_1=u1, rockfall_active_2=u2)
    teacher.fresh(route=route)
    goal = o[29:31].copy()
    info = {}
    for t in range(horizon):
      action = teacher.act(o, env.schedule)
      status = teacher_status(teacher)
      if t % EVERY == 0:
        rec.grab(t, goal, u1, u2, status, sched_text(env))
      o, r, done, info = env.step(action)
      if done or r > 0:
        break
    rec.grab(int(info.get('t', 0)), goal, u1, u2,
             '%s  %s' % (teacher_status(teacher), outcome(info)),
             sched_text(env))
    rec.save(os.path.join(out, name + '.gif'))
    print('     %-20s %s in %d steps' % (name, outcome(info),
                                         info.get('t', -1)), flush=True)


def agent_clips(ckpt, out, seed, horizon, env_name, max_episodes):
  """Replay the eval sequence and record the first episode of each outcome.

  Latents are NEVER forced here: these are natural draws, so a clip shows what
  the policy actually runs into.

  Frames are captured DURING the episode as mujoco state snapshots and only
  rendered once the outcome is known. Re-running an episode is not an option:
  a second reset() draws fresh ant reset noise, so the replay would be a
  different episode wearing the same label.
  """
  import jax.numpy as jnp
  from types import SimpleNamespace
  import eval_rockfall_clock_v6_baseline as EV

  args = SimpleNamespace(
      env_name=env_name, npz=None, horizon=horizon,
      p_active_1=V6.P_ACTIVE_1, p_active_2=V6.P_ACTIVE_2,
      t0_min_1=V6.T0_MIN_1, t0_max_1=V6.T0_MAX_1,
      t0_min_2=V6.T0_MIN_2, t0_max_2=V6.T0_MAX_2, seed=seed)
  #: the policy and the env come from the eval script, so a clip is produced
  #: by the same objects the reported numbers were.
  act_mean, step, _ = EV.build_mean_policy(ckpt, args)
  _, env = EV._configure_env(args, seed=seed)
  print('  agent checkpoint @ step %d' % step, flush=True)

  wanted = {
      'agent_success': lambda i: bool(i.get('success')),
      'agent_death_zone1': lambda i: bool(i.get('failure'))
                                     and i.get('failure_zone') == 1,
      'agent_death_zone2': lambda i: bool(i.get('failure'))
                                     and i.get('failure_zone') == 2,
  }
  titles = {'agent_success': 'AGENT  reaches the goal untouched',
            'agent_death_zone1': 'AGENT  killed by the zone-1 rockfall',
            'agent_death_zone2': 'AGENT  killed by the zone-2 rockfall'}
  found = {}
  for episode in range(max_episodes):
    if len(found) == len(wanted):
      break
    o = env.reset()
    u1 = bool(env.privileged_rockfall_active_1)
    u2 = bool(env.privileged_rockfall_active_2)
    goal = o[29:31].copy()
    sched = sched_text(env)
    snaps, info = [], {}
    data = env._env.data
    for t in range(horizon):
      action = np.asarray(act_mean(jnp.asarray(o[None], jnp.float32))[0])
      if t % EVERY == 0:
        snaps.append((t, np.asarray(data.qpos).copy(),
                      np.asarray(data.qvel).copy()))
      o, r, done, info = env.step(action)
      if done or r > 0:
        break
    snaps.append((int(info.get('t', 0)), np.asarray(data.qpos).copy(),
                  np.asarray(data.qvel).copy()))
    for name, test in wanted.items():
      if name in found or not test(info):
        continue
      found[name] = episode
      rec = Recorder(env, '%s   (eval episode %d)'
                     % (titles[name], episode))
      keep_qpos = np.asarray(data.qpos).copy()
      keep_qvel = np.asarray(data.qvel).copy()
      for k, (t, qpos, qvel) in enumerate(snaps):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(env._env.model, data)
        rec.grab(t, goal, u1, u2,
                 outcome(info) if k == len(snaps) - 1 else '', sched)
      data.qpos[:] = keep_qpos
      data.qvel[:] = keep_qvel
      mujoco.mj_forward(env._env.model, data)
      rec.save(os.path.join(out, name + '.gif'))
      print('     %-20s episode %d, %s in %s steps'
            % (name, episode, outcome(info), info.get('t', -1)), flush=True)
  missing = sorted(set(wanted) - set(found))
  if missing:
    print('  NOT FOUND in %d episodes: %s' % (max_episodes, missing),
          flush=True)
  return found


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--ckpt', default=None,
                  help='trained checkpoint for the three agent clips')
  ap.add_argument('--env-name', default='offline_antmaze_rockfall_clock_v6_gxy')
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--expert-seed', type=int, default=101)
  ap.add_argument('--horizon', type=int, default=CT.HORIZON)
  ap.add_argument('--max-episodes', type=int, default=40)
  ap.add_argument('--skip-expert', action='store_true')
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  print('V6 GIFs -> %s' % args.out_dir)
  if not args.skip_expert:
    print('EXPERT CLIPS')
    expert_clips(args.out_dir, args.expert_seed, args.horizon)
  if args.ckpt:
    print('AGENT CLIPS')
    agent_clips(args.ckpt, args.out_dir, args.seed, args.horizon,
                args.env_name, args.max_episodes)


if __name__ == '__main__':
  main()
