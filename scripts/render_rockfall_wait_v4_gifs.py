"""Top-down GIFs of the V4 rockfall-wait benchmark.

Expert clips (latent set through reset(rockfall_active=...); the expert
learns it only at the mouth line):
  expert_clear_go         u=clear  -> walks straight through
  expert_active_wait      u=active -> holds at the mouth (zero torque) until
                          the rockfall has passed, then walks through
  expert_active_forced_go u=active, told 'go' anyway: the do(go) experiment

Agent clips (--agent-ckpt, deterministic tanh(mu), eval seed 909, episodes
replayed from episode 0 so clip k is the k-th eval episode): the first
K_AGENT episodes of each latent.

Overlay: goal ring, the mouth line (yellow) and the band edges (red), a
caption with u / t / rockfall state. Frames every 4 env steps at 20 fps.

Usage: python scripts/render_rockfall_wait_v4_gifs.py [--agent-ckpt PKL]
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
from crl import rockfall_wait_v4 as V4    # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y  # noqa: E402
import rockfall_wait_v4_teacher as WT      # noqa: E402
from render_tworoute_v3_gifs import (      # noqa: E402
    Recorder, outcome, to_px, EVERY, FPS)

OUT = os.path.join(WT.OUT, 'gifs')
K_AGENT = 3


class V4Recorder(Recorder):
  """Recorder + mouth line / band edges + rockfall state in the caption."""

  def grab(self, t, goal_xy, u, status='', state=''):
    super().grab(t, goal_xy, u, status)
    im = Image.fromarray(self.frames.pop())
    dr = ImageDraw.Draw(im)
    x0, y0 = to_px(V4.MOUTH_X, -HAZARD_HALF_Y)
    x1, y1 = to_px(V4.MOUTH_X, HAZARD_HALF_Y)
    dr.line([x0, y0, x1, y1], fill=(255, 230, 0), width=2)
    for bx in HAZARD_X:
      x0, y0 = to_px(bx, -HAZARD_HALF_Y)
      x1, y1 = to_px(bx, HAZARD_HALF_Y)
      dr.line([x0, y0, x1, y1], fill=(255, 60, 60), width=2)
    if state:
      dr.text((200, 16), state, fill=(255, 255, 255))
    self.frames.append(np.asarray(im))


def rock_state(info):
  if info.get('rockfall_open'):
    return f'ROCKFALL wave {info["rock_waves"]}'
  if info.get('rockfall_passed'):
    return 'passed'
  return ''


def expert_clips(out):
  cfg, teacher = WT.make_teacher()
  env = envs_mod.make_env(WT.ENV_NAME, cfg, seed=101)
  for name, u, intent in (('expert_clear_go', False, None),
                          ('expert_active_wait', True, None),
                          ('expert_active_forced_go', True, 'go')):
    rec = V4Recorder(env, 'EXPERT [forced: do(go)]' if intent
                     else 'EXPERT (learns u at the mouth)')
    o = env.reset(rockfall_active=u)
    teacher.fresh()
    goal = o[29:31].copy()
    info = {}
    for t in range(WT.HORIZON):
      if t % EVERY == 0:
        rec.grab(t, goal, u, 'HOLD' if teacher.holding
                 else (f'-> {teacher.decision}' if teacher.decision else ''),
                 rock_state(info))
      a = teacher.act(o, intent, revealed=env.revealed_rockfall_active)
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    rec.grab(t + 1, goal, u, outcome(info), rock_state(info))
    rec.save(os.path.join(out, f'{name}.gif'))


def agent_clips(ckpt, out, seed=909):
  import jax.numpy as jnp
  from eval_rockfall_wait_v4_baseline import build_policy, make_env
  act, _, _ = build_policy(ckpt)
  _, env = make_env(seed)
  done_u = {False: 0, True: 0}
  k = -1
  while min(done_u.values()) < K_AGENT:
    k += 1
    o = env.reset()
    u = bool(env.privileged_rockfall_active)
    goal = o[29:31].copy()
    rec = (V4Recorder(env, f'NAIVE CRL (blind to u)  ep {k}')
           if done_u[u] < K_AGENT else None)
    info = {}
    for t in range(WT.HORIZON):
      if rec is not None and t % EVERY == 0:
        rec.grab(t, goal, u, '', rock_state(info))
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      if done or r > 0:
        break
    if rec is not None:
      rec.grab(t + 1, goal, u, outcome(info), rock_state(info))
      tag = 'active' if u else 'clear'
      rec.save(os.path.join(out, f'agent_k{k}_{tag}_'
                            f'{outcome(info).split()[0].lower()}.gif'))
      done_u[u] += 1
      print(f'    ep {k}: u={u} {outcome(info)} steps={t + 1} '
            f'band_entry={info.get("band_entry_step")} '
            f'trigger={info.get("trigger_step")}', flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--agent-ckpt', default=None)
  ap.add_argument('--no-expert', action='store_true')
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)
  if not args.no_expert:
    print('expert clips', flush=True)
    expert_clips(OUT)
  if args.agent_ckpt:
    print('agent clips', flush=True)
    agent_clips(args.agent_ckpt, OUT)


if __name__ == '__main__':
  main()
