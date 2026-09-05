"""Top-down GIFs of the V5 rockfall-clock benchmark.

Expert clips (latent set through reset(rockfall_active=...); the expert
reads the timetable from env.schedule and acts on it at the mouth line):
  expert_clear_go         u=clear  -> walks straight through
  expert_active_wait      u=active, natural t0 -> the burst overlaps the
                          crossing, so it holds at the mouth (zero torque)
                          until the rocks are parked, then walks through
  expert_active_forced_go u=active, told 'go' anyway: the do(go) experiment
  expert_detour           u=active, route coin = detour: north column, top
                          row, east column; the rocks fall on the band on
                          their own clock while the ant is far away

Agent clips (--ckpt, deterministic tanh(mu), eval seed 909, episodes
replayed from episode 0 so clip k is the k-th eval episode): the first
K_AGENT episodes of each latent. Needs scripts/eval_rockfall_clock_v5_
baseline.py (imported lazily, so the expert clips work without it).

Overlay: goal ring, the mouth line (yellow) and the band edges (red), a
caption with u / t / decision / HOLD / route on the left and the schedule
(t0..end, wave count, parked) on the right. Frames every 4 env steps at
20 fps.

Usage: python scripts/render_rockfall_clock_v5_gifs.py [--ckpt PKL]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from PIL import Image, ImageDraw           # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_clock_v5 as V5   # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y  # noqa: E402
import rockfall_clock_v5_teacher as CT     # noqa: E402
from render_tworoute_v3_gifs import (      # noqa: E402
    Recorder, outcome, to_px, EVERY, FPS)

OUT = os.path.join(CT.OUT, 'gifs')
K_AGENT = 3
#: expert clips: (name, latent, route, forced intent)
EXPERT_CLIPS = (('expert_clear_go', False, 'shortcut', None),
                ('expert_active_wait', True, 'shortcut', None),
                ('expert_active_forced_go', True, 'shortcut', 'go'),
                ('expert_detour', True, 'detour', None))


class V5Recorder(Recorder):
  """Recorder + mouth line / band edges + the schedule in the caption."""

  def grab(self, t, goal_xy, u, status='', state=''):
    super().grab(t, goal_xy, u, status)
    im = Image.fromarray(self.frames.pop())
    dr = ImageDraw.Draw(im)
    x0, y0 = to_px(V5.MOUTH_X, -HAZARD_HALF_Y)
    x1, y1 = to_px(V5.MOUTH_X, HAZARD_HALF_Y)
    dr.line([x0, y0, x1, y1], fill=(255, 230, 0), width=2)
    for bx in HAZARD_X:
      x0, y0 = to_px(bx, -HAZARD_HALF_Y)
      x1, y1 = to_px(bx, HAZARD_HALF_Y)
      dr.line([x0, y0, x1, y1], fill=(255, 60, 60), width=2)
    if state:
      dr.text((200, 16), state, fill=(255, 255, 255))
    self.frames.append(np.asarray(im))


def rock_state(env, info):
  """Right-hand caption: the timetable (privileged, from env.schedule) and
  the burst's progress. Empty for a clear episode: nothing physical
  happens and the learner-facing frame should not hint otherwise."""
  sched = env.schedule
  if not sched['active']:
    return ''
  s = f't0={sched["start"]}..{sched["end"]}'
  if info.get('rockfall_passed'):
    return s + ' parked'
  if info.get('rockfall_open'):
    return s + f' wave {info["rock_waves"]}'
  return s


def expert_status(teacher, intent):
  """Left-hand caption: what the expert is doing right now."""
  if teacher.route == 'detour':
    return 'route: detour'
  if teacher.holding:
    return f'HOLD until {teacher.release_step}'
  if intent is not None:
    return f'forced {intent}'
  return f'-> {teacher.decision}' if teacher.decision else ''


def expert_clips(out, goal_rep='full'):
  cfg, teacher = CT.make_teacher()
  #: the scripted teacher reads the state and o[29:31] (the goal xy), both of
  #: which exist under either goal contract, so its clips are the same either
  #: way; the flag is here so a _gxy run can be rendered end to end.
  env_name = CT.ENV_NAME + ('_gxy' if goal_rep == 'xy' else '')
  env = envs_mod.make_env(env_name, cfg, seed=101)
  for name, u, route, intent in EXPERT_CLIPS:
    if intent:
      title = f'EXPERT [forced: do({intent})]'
    elif route == 'detour':
      title = 'EXPERT (route coin: detour)'
    else:
      title = 'EXPERT (reads the schedule, acts at the mouth)'
    rec = V5Recorder(env, title)
    o = env.reset(rockfall_active=u)
    teacher.fresh(route=route)
    goal = o[29:31].copy()
    info = {}
    for t in range(CT.HORIZON):
      if t % EVERY == 0:
        rec.grab(t, goal, u, expert_status(teacher, intent),
                 rock_state(env, info))
      a = teacher.act(o, env.schedule, intent)
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    rec.grab(t + 1, goal, u, outcome(info), rock_state(env, info))
    rec.save(os.path.join(out, f'{name}.gif'))
    print(f'    {name}: u={u} route={info.get("route")} {outcome(info)} '
          f'steps={t + 1} t0={info.get("rockfall_start")} '
          f'mouth={info.get("mouth_step")} '
          f'band_entry={info.get("band_entry_step")} '
          f'decision={teacher.decision} hold={teacher.hold_steps_done} '
          f'waves={info.get("rock_waves")}', flush=True)


def agent_clips(ckpt, out, seed=909):
  import jax.numpy as jnp
  import eval_rockfall_clock_v5_baseline as EV
  from eval_rockfall_clock_v5_baseline import build_policy, make_env
  #: a checkpoint trained on the upstream XY-goal env needs that env here too;
  #: build_policy asserts the widths, so a mismatch aborts rather than
  #: rendering a meaningless clip.
  if EV.infer_goal_rep(ckpt) == 'xy':
    EV.ENV_NAME = EV.ENV_NAME + '_gxy'
    print(f'  agent clips use {EV.ENV_NAME}', flush=True)
  act, _, _ = build_policy(ckpt)
  _, env = make_env(seed)
  done_u = {False: 0, True: 0}
  k = -1
  while min(done_u.values()) < K_AGENT:
    k += 1
    o = env.reset()
    u = bool(env.privileged_rockfall_active)
    goal = o[29:31].copy()
    rec = (V5Recorder(env, f'NAIVE CRL (blind to u and the clock)  ep {k}')
           if done_u[u] < K_AGENT else None)
    info = {}
    for t in range(CT.HORIZON):
      if rec is not None and t % EVERY == 0:
        route = info.get('route')
        rec.grab(t, goal, u, f'route: {route}' if route else '',
                 rock_state(env, info))
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    if rec is not None:
      rec.grab(t + 1, goal, u, outcome(info), rock_state(env, info))
      tag = 'active' if u else 'clear'
      rec.save(os.path.join(out, f'agent_k{k}_{tag}_'
                            f'{outcome(info).split()[0].lower()}.gif'))
      done_u[u] += 1
      print(f'    ep {k}: u={u} {outcome(info)} steps={t + 1} '
            f'route={info.get("route")} '
            f't0={info.get("rockfall_start")} '
            f'mouth={info.get("mouth_step")} '
            f'band_entry={info.get("band_entry_step")}', flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=None, help='agent checkpoint (pkl)')
  ap.add_argument('--no-expert', action='store_true')
  ap.add_argument('--goal-rep', choices=['full', 'xy'], default='full',
                  help='goal contract for the EXPERT clips; the agent clips '
                       'follow their checkpoint')
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  if not args.no_expert:
    print('expert clips', flush=True)
    expert_clips(args.out, args.goal_rep)
  if args.ckpt:
    print('agent clips', flush=True)
    agent_clips(args.ckpt, args.out)


if __name__ == '__main__':
  main()
