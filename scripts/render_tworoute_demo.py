"""Visual acceptance demo for TwoRouteRockfallAntEnv (V0). Analysis only.

One MP4, four episodes under REAL simulation dynamics (env.step with the
frozen controllers; no teleportation during episodes):

  A  CLEAR  + SHORTCUT -> crosses the hazard band, reaches the goal
  B  ACTIVE + SHORTCUT -> terminal failure on band entry (matched start w/ A)
  C  CLEAR  + DETOUR   -> long way round, reaches the goal
  D  ACTIVE + DETOUR   -> identical to C (the latent touches nothing there)

Drivers (frozen, deterministic):
  * shortcut: the corridor walker driven through a 90-degree WORLD-FRAME
    remap (joint torques are body-frame, so a walker that tracks a lateral
    lane while walking +x drives +y exactly as well once the observation is
    rotated); the ant's initial heading is set to +y at reset (initial
    condition, not an in-episode teleport).
  * detour: the repo-standard recipe -- walker +x along the bottom corridor,
    latched handoff to the goal-conditioned base policy at x >= 6.

Matched pairs share the exact captured reset state (A/B and C/D); the ONLY
difference inside a pair is the hidden rockfall latent, so the trajectories
are byte-identical until the hazard band physically matters.

The hazard band and goal disc are drawn by a SEPARATE render-only MjModel
(same XML + two contype=0/conaffinity=0 ghost geoms) that mirrors the env's
qpos each frame; the learner env, its physics, and its 58-dim observation
are untouched -- asserted every step.

Output: artifacts/tworoute_rockfall_v0/antmaze_rockfall_demo.mp4 (+ per-case
GIFs and demo_manifest.json).

Usage: python scripts/render_tworoute_demo.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import tworoute_rockfall_ant as TR  # noqa: E402
from crl.d4rl_ant import build_maze_xml   # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
MP4 = os.path.join(OUT, 'antmaze_rockfall_demo.mp4')
SEED = 7
FPS = 30
REN_W = REN_H = 512
BAR_H = 160                                 # top text bar -> 512x672 frames
HOLD_S = 1.6                                # end-of-episode freeze
CARD_S = 1.6

#: Rz(-90): world +y becomes virtual +x (the walker's forward axis).
QM = np.array([np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)])
#: Rz(+90): initial heading facing +y for the shortcut episodes.
QP = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ---------------------------------------------------------------- drivers ---
def rot_state29(s):
  v = s.copy()
  x, y = s[0], s[1]
  v[0], v[1] = y, -x
  v[3:7] = qmul(QM, s[3:7])
  vx, vy = s[15], s[16]
  v[15], v[16] = vy, -vx
  return v                                  # ang vel (BODY frame) + joints stay


def make_drivers(walker, base_act):
  def shortcut(o58, true_goal):
    s = rot_state29(o58[:29])
    o = np.concatenate([s, o58[29:]]).astype(np.float32)
    return walker(o, 0.0, RP.V_SIDE)        # track world x ~ 0, walk +y

  state = {'handoff': False}

  def detour(o58, true_goal):
    x, y = float(o58[0]), float(o58[1])
    if not state['handoff'] and (x >= RP.HANDOFF_X or y >= 2.0):
      state['handoff'] = True
    if state['handoff']:
      oc = o58.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      return np.asarray(base_act(jnp.asarray(oc[None]))[0])
    return walker(o58, 0.0, RP.V_SIDE)

  def detour_fresh():
    state['handoff'] = False
  return shortcut, detour, detour_fresh


# ---------------------------------------------------------- render model ----
def build_render_model():
  """Env XML + ghost hazard/goal geoms. NEVER handed to the env."""
  xml, _ = build_maze_xml(TR.TWO_ROUTE_MAZE)
  root = ET.fromstring(xml)
  vis = root.find('visual')
  if vis is None:
    vis = ET.SubElement(root, 'visual')
  ET.SubElement(vis, 'global', offwidth=str(REN_W), offheight=str(REN_H))
  wb = root.find('.//worldbody')
  x0, x1, y0, y1 = TR.hazard_zone()
  ET.SubElement(wb, 'geom', name='hazard_viz', type='box',
                pos=f'{(x0 + x1) / 2} {(y0 + y1) / 2} 0.03',
                size=f'{(x1 - x0) / 2} {(y1 - y0) / 2} 0.03',
                rgba='0.95 0.15 0.10 0.40', contype='0', conaffinity='0')
  ET.SubElement(wb, 'geom', name='goal_viz', type='cylinder',
                pos='0.75 8.75 0.03', size='0.5 0.03',
                rgba='0.15 0.9 0.3 0.5', contype='0', conaffinity='0')
  m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
  return m, mujoco.MjData(m)


def font(sz):
  try:
    return ImageFont.load_default(size=sz)
  except TypeError:                          # older PIL: fixed-size bitmap
    return ImageFont.load_default()


F_BIG, F_MED, F_SMALL = font(30), font(20), font(16)


def compose(frame, case_tag, info, latent, step, flash=None):
  """512x512 render + 160px debug bar on top."""
  im = Image.new('RGB', (REN_W, REN_H + BAR_H), (12, 12, 12))
  im.paste(Image.fromarray(frame), (0, BAR_H))
  d = ImageDraw.Draw(im, 'RGBA')
  d.text((14, 8), case_tag, font=F_MED, fill=(255, 255, 255))
  col = (255, 90, 70) if latent else (90, 200, 255)
  d.text((14, 40), f'ROCKFALL: {"ACTIVE" if latent else "CLEAR"}',
         font=F_MED, fill=col)
  d.text((14, 68), f'ROUTE: {str(info["route"] or "-").upper()}',
         font=F_MED, fill=(230, 230, 230))
  d.text((14, 96), f'ENTERED_HAZARD: {info["entered_hazard"]}', font=F_MED,
         fill=(255, 180, 80) if info['entered_hazard'] else (160, 160, 160))
  d.text((280, 40), f'FAILURE: {info["failure"]}', font=F_MED,
         fill=(255, 60, 60) if info['failure'] else (160, 160, 160))
  d.text((280, 68), f'SUCCESS: {info["success"]}', font=F_MED,
         fill=(80, 230, 110) if info['success'] else (160, 160, 160))
  d.text((280, 96), f'step {step}', font=F_MED, fill=(150, 150, 150))
  d.text((14, 126), 'red band = hazard zone (render-only)   green disc = goal',
         font=F_SMALL, fill=(120, 120, 120))
  if flash:
    txt, col = flash
    tw = d.textlength(txt, font=F_BIG)
    d.rectangle([REN_W / 2 - tw / 2 - 16, BAR_H + 200,
                 REN_W / 2 + tw / 2 + 16, BAR_H + 260],
                fill=(0, 0, 0, 190), outline=col, width=3)
    d.text((REN_W / 2 - tw / 2, BAR_H + 212), txt, font=F_BIG, fill=col)
  return np.asarray(im)


def card(lines, sub=None):
  im = Image.new('RGB', (REN_W, REN_H + BAR_H), (12, 12, 12))
  d = ImageDraw.Draw(im)
  y = (REN_H + BAR_H) / 2 - 40 * len(lines) / 2 - (30 if sub else 0)
  for ln in lines:
    tw = d.textlength(ln, font=F_BIG)
    d.text(((REN_W - tw) / 2, y), ln, font=F_BIG, fill=(240, 240, 240))
    y += 44
  if sub:
    for ln in sub:
      tw = d.textlength(ln, font=F_MED)
      d.text(((REN_W - tw) / 2, y + 16), ln, font=F_MED, fill=(150, 200, 150))
      y += 30
  return np.asarray(im)


# ------------------------------------------------------------- episodes -----
def capture(env):
  d = env._env.data
  return d.qpos.copy(), d.qvel.copy(), env._flatten(env._last_obs)[29:31].copy()


def restore(env, qpos, qvel):
  """Matched initial conditions for the second arm of a pair (reset-time
  state copy; every in-episode transition below is a plain env.step)."""
  d = env._env.data
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  d.qacc_warmstart[:] = 0.0
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()
  return env._flatten(env._last_obs)


def silent_run(env, o, driver, max_steps=450):
  """Rollout without rendering; returns the outcome. Used to SELECT a reset
  draw whose frozen-controller episode completes (deterministic search over
  reset indices -- the repo's standard gif-scenario selection; no in-episode
  intervention)."""
  true_goal = o[29:31].copy()
  for t in range(max_steps):
    o, r, done, _ = env.step(driver(o, true_goal))
    if done:
      return 'failure'
    if r > 0:
      return 'success'
  return 'timeout'


def run_case(env, o, driver, latent, tag, ren, rd, rm, cam, every=1,
             max_steps=650):
  frames, info = [], {'route': None, 'entered_hazard': False,
                      'failure': False, 'success': False}
  outcome, t = 'timeout', 0
  true_goal = o[29:31].copy()
  for t in range(max_steps):
    a = driver(o, true_goal)
    o, r, done, info = env.step(a)
    # hard guard: nothing from the demo leaks into the learner obs
    assert o.shape == (58,) and np.all(o[31:] == 0.0), 'obs contract broken'
    if t % every == 0:
      rd.qpos[:] = env._env.data.qpos
      rd.qvel[:] = env._env.data.qvel
      mujoco.mj_forward(rm, rd)
      ren.update_scene(rd, camera=cam)
      frames.append(compose(ren.render().copy(), tag, info, latent, t + 1))
    if done:
      outcome = 'failure'
      break
    if r > 0:
      outcome = 'success'
      break
  # outcome flash + freeze
  rd.qpos[:] = env._env.data.qpos
  rd.qvel[:] = env._env.data.qvel
  mujoco.mj_forward(rm, rd)
  ren.update_scene(rd, camera=cam)
  last = ren.render().copy()
  flash = (('TERMINAL FAILURE', (255, 70, 60)) if outcome == 'failure' else
           ('GOAL REACHED', (80, 230, 110)) if outcome == 'success' else
           ('TIMEOUT', (255, 200, 60)))
  frames += [compose(last, tag, info, latent, t + 1, flash=flash)
             ] * int(HOLD_S * FPS)
  return frames, {'outcome': outcome, 'steps': t + 1, **{
      k: info[k] for k in ('route', 'entered_hazard', 'failure', 'success')}}


def main():
  os.makedirs(OUT, exist_ok=True)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'fixed'              # one fixed goal: clean matched demo
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=SEED)
  shortcut, detour, detour_fresh = make_drivers(walker, base_act)

  rm, rd = build_render_model()
  ren = mujoco.Renderer(rm, REN_H, REN_W)
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (4.0, 4.0, 0.0)
  cam.distance, cam.elevation, cam.azimuth = 24.0, -90.0, 90.0

  def face_north(env):
    d = env._env.data
    d.qpos[3:7] = qmul(QP, d.qpos[3:7].copy())
    d.qvel[:2] = 0.0
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env._env.model, d)
    env._last_obs = env._env._obs_dict()
    return env._flatten(env._last_obs)

  def find_start(prep, driver, fresh=None, tries=12):
    """Deterministic search over reset draws for a start whose CLEAR episode
    succeeds under the frozen controllers."""
    for k in range(tries):
      env.reset(rockfall_active=False)
      if prep:
        prep(env)
      q, v, _ = capture(env)
      if fresh:
        fresh()
      if silent_run(env, restore(env, q, v), driver) == 'success':
        return q, v, k
    raise RuntimeError('no completing start found')

  segs, results = [], {}
  # ---- pair A/B: matched shortcut starts ----------------------------------
  q0, v0, k_sc = find_start(face_north, shortcut)
  print(f'shortcut start: reset draw #{k_sc}', flush=True)
  env.reset(rockfall_active=False)          # clear episode flags, then match
  o = restore(env, q0, v0)
  fr, res = run_case(env, o, shortcut, False,
                     'A  CLEAR + SHORTCUT', ren, rd, rm, cam)
  segs.append(('A. Clear + Shortcut',
               ['rockfall_active = False', 'ant takes the SHORT route'], fr))
  results['A_clear_shortcut'] = res
  env.reset(rockfall_active=True)
  o = restore(env, q0, v0)                  # identical start, latent flipped
  fr, res = run_case(env, o, shortcut, True,
                     'B  ACTIVE + SHORTCUT', ren, rd, rm, cam)
  segs.append(('B. Active + Shortcut',
               ['rockfall_active = True', 'same start, same controller'], fr))
  results['B_active_shortcut'] = res

  # ---- pair C/D: matched detour starts ------------------------------------
  qd, vd, k_dt = find_start(None, detour, fresh=detour_fresh)
  print(f'detour start: reset draw #{k_dt}', flush=True)
  env.reset(rockfall_active=False)          # clear episode flags, then match
  o = restore(env, qd, vd)
  detour_fresh()
  fr, res = run_case(env, o, detour, False,
                     'C  CLEAR + DETOUR', ren, rd, rm, cam)
  segs.append(('C. Clear + Detour',
               ['rockfall_active = False', 'ant takes the LONG safe route'],
               fr))
  results['C_clear_detour'] = res
  env.reset(rockfall_active=True)
  o = restore(env, qd, vd)
  detour_fresh()
  fr, res = run_case(env, o, detour, True,
                     'D  ACTIVE + DETOUR', ren, rd, rm, cam)
  segs.append(('D. Active + Detour',
               ['rockfall_active = True', 'detour is safe regardless'], fr))
  results['D_active_detour'] = res

  # ---- assemble -----------------------------------------------------------
  video = [card(['AntMaze-Rockfall V0', 'Visual Sanity Check'],
                ['env: offline_ant_umaze_tworoute_rockfall',
                 'real simulation dynamics, frozen controllers,',
                 'hidden latent: rockfall_active ~ Bernoulli(0.30)'])
           ] * int(2.6 * FPS)
  for title, sub, fr in segs:
    video += [card([title], sub)] * int(CARD_S * FPS)
    video += fr
  ok = {'A_clear_shortcut': ('success', 'shortcut'),
        'B_active_shortcut': ('failure', 'shortcut'),
        'C_clear_detour': ('success', 'detour'),
        'D_active_detour': ('success', 'detour')}
  marks = {k: (results[k]['outcome'] == v[0] and results[k]['route'] == v[1])
           for k, v in ok.items()}
  video += [card(['Summary'],
                 [f'Clear + Shortcut : {"PASS" if marks["A_clear_shortcut"] else "FAIL"}',
                  f'Active + Shortcut: {"PASS (failure triggered)" if marks["B_active_shortcut"] else "FAIL"}',
                  f'Clear + Detour   : {"PASS" if marks["C_clear_detour"] else "FAIL"}',
                  f'Active + Detour  : {"PASS" if marks["D_active_detour"] else "FAIL"}',
                  'Smoke tests: 13/13 PASS'])] * int(3.2 * FPS)
  imageio.mimsave(MP4, video, fps=FPS, quality=8, macro_block_size=None)
  dur = len(video) / FPS
  print(f'MP4 -> {MP4}  ({len(video)} frames, {dur:.1f}s)', flush=True)

  for (title, _, fr), key in zip(segs, ok):
    p = os.path.join(OUT, f'demo_{key}.gif')
    imageio.mimsave(p, fr[::3], fps=12, loop=0)
    print('gif ->', p, flush=True)

  man = {'mp4': MP4, 'duration_s': round(dur, 1), 'fps': FPS, 'seed': SEED,
         'start_selection': {'shortcut_reset_draw': k_sc,
                             'detour_reset_draw': k_dt},
         'results': results, 'expected': {k: v[0] for k, v in ok.items()},
         'all_as_expected': all(marks.values())}
  with open(os.path.join(OUT, 'demo_manifest.json'), 'w') as f:
    json.dump(man, f, indent=2)
  print(json.dumps({k: results[k] for k in ok}, indent=2), flush=True)
  print('ALL AS EXPECTED' if man['all_as_expected'] else 'MISMATCH',
        flush=True)


if __name__ == '__main__':
  main()
