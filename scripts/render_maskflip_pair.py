"""Side-by-side paired mask-flip GIFs for the clean-dataset ant. Analysis only.

The visual form of test A in scripts/leak_probe_clean_ant.py: one reset, one
policy, two worlds. LEFT panel has every rockfall site OFF, RIGHT panel has
them ON (severity 'severe'). The ant sees the same 58-dim observation contract
in both; the mask is not in it.

If the mask has NOT leaked, the two panels are frame-for-frame identical until
the rock is physically dropped -- the ant walks into the hazard exactly as
confidently as it walks through the empty corridor, and only then do the panels
split. If the ant were reading the mask, the right panel would peel away
EARLIER, while the corridor still looks empty.

The caption bar prints the live step, the step the rock dropped, and the step
the two observation streams first differed, so the claim is checkable frame by
frame rather than on trust.

Usage:
  python scripts/render_maskflip_pair.py                  # 3 pairs with events
  python scripts/render_maskflip_pair.py --n 1 --mp4
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                              # noqa: E402
import imageio                             # noqa: E402
try:
  from PIL import Image, ImageDraw         # noqa: E402
  _PIL = True
except Exception:                          # pylint: disable=broad-except
  _PIL = False
from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from leak_probe_clean_ant import build, set_state  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_walk_gifs/maskflip'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
SEED = 33_871
P_ACTIVE = 0.30
HORIZON = 800


def arm(env, act, o, ren, cam, every, horizon):
  """One world. Returns rendered frames (with the step index they came from),
  the observation stream, and the first drop / first contact / death steps."""
  frames, steps, obs_l = [], [], []
  fdrop = fcont = dead_at = None
  for t in range(horizon):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    obs_l.append(o.copy())
    o, r, _, info = env.step(a)
    if any(info['dropped']) and fdrop is None:
      fdrop = t
    if info['rock_any_contact'] and fcont is None:
      fcont = t
    if info['dead'] and dead_at is None:
      dead_at = t
    if t % every == 0:
      d = env._env.data
      cam.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.4)
      ren.update_scene(d, camera=cam)
      frames.append(ren.render().copy())
      steps.append(t)
    if float(r) > 0 or (dead_at is not None and t > dead_at + 12):
      break
  return {'frames': frames, 'steps': steps, 'obs': np.asarray(obs_l),
          'first_drop': fdrop, 'first_contact': fcont, 'dead_at': dead_at,
          'success': float(r) > 0}


def compose(a, b, div, fdrop, fcont):
  """Pad both arms to the same length and stack them side by side with a
  caption bar. The caption is the audit trail, not decoration."""
  n = max(len(a['frames']), len(b['frames']))
  out = []
  for i in range(n):
    fa = a['frames'][min(i, len(a['frames']) - 1)]
    fb = b['frames'][min(i, len(b['frames']) - 1)]
    step = (a['steps'] if i < len(a['steps']) else b['steps'])[
        min(i, len(a['steps'] if i < len(a['steps']) else b['steps']) - 1)]
    pair = np.concatenate([fa, fb], axis=1)
    if not _PIL:
      out.append(pair)
      continue
    im = Image.new('RGB', (pair.shape[1], pair.shape[0] + 30), (0, 0, 0))
    im.paste(Image.fromarray(pair), (0, 30))
    d = ImageDraw.Draw(im)
    half = pair.shape[1] // 2
    d.text((6, 3), 'mask OFF (no rocks)', fill=(120, 220, 255))
    d.text((half + 6, 3), 'mask ON (all 4 sites armed)', fill=(255, 150, 120))
    tail = f'step {step}'
    if fdrop is not None:
      tail += f'   rock dropped @ {fdrop}'
      tail += ('   paths still identical' if step <= fdrop else
               f'   paths differ since {div}' if div is not None else '')
    d.text((6, 17), tail, fill=(230, 230, 230))
    d.line([(half, 30), (half, im.height)], fill=(0, 0, 0), width=2)
    out.append(np.asarray(im))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--n', type=int, default=3, help='pairs WITH an event')
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--every', type=int, default=4)
  ap.add_argument('--fps', type=int, default=25)
  ap.add_argument('--height', type=int, default=240)
  ap.add_argument('--width', type=int, default=320)
  ap.add_argument('--severity', default='severe',
                  choices=['severe', 'mild', 'impaired'])
  ap.add_argument('--mp4', action='store_true')
  ap.add_argument('--max-tries', type=int, default=40)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  ext = 'mp4' if args.mp4 else 'gif'

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = float(args.p_active)
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.rockfall_reset_fix = True
  act, step = build(cfg, args.ckpt)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)

  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)
  ren = mujoco.Renderer(env._env.model, args.height, args.width)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 8.0, -55.0, -90.0
  sev = (args.severity,) * 4

  rows, k, tries = [], 0, 0
  while k < args.n and tries < args.max_tries:
    tries += 1
    env.reset()
    q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
    goal = env._flatten(env._last_obs)[29:31].copy()

    o = set_state(env, q0, v0, goal, (0, 0, 0, 0), sev)
    A = arm(env, act, o, ren, cam, args.every, args.horizon)
    o = set_state(env, q0, v0, goal, (1, 1, 1, 1), sev)
    B = arm(env, act, o, ren, cam, args.every, args.horizon)

    if B['first_drop'] is None:
      print(f'  [skip try {tries}] no rock dropped in the armed world',
            flush=True)
      continue
    n = min(len(A['obs']), len(B['obs']))
    dif = np.abs(A['obs'][:n] - B['obs'][:n]).max(axis=1)
    div = int(np.argmax(dif > 1e-9)) if (dif > 1e-9).any() else None
    lead = None if div is None else div - B['first_drop']
    ok = div is None or div > B['first_drop']

    nm = (f'maskflip{k}_drop{B["first_drop"]}_div{div}'
          f'_{"reactive" if ok else "LEAK"}.{ext}')
    frames = compose(A, B, div, B['first_drop'], B['first_contact'])
    p = os.path.join(args.out_dir, nm)
    if args.mp4:
      imageio.mimsave(p, frames, fps=args.fps, quality=8,
                      macro_block_size=None)
    else:
      imageio.mimsave(p, frames, fps=args.fps, loop=0)
    print(f'[pair {k}] drop@{B["first_drop"]} contact@{B["first_contact"]} '
          f'div@{div} (lead {lead}) -> {nm}', flush=True)
    rows.append({'pair': k, 'file': nm, 'first_drop': B['first_drop'],
                 'first_contact': B['first_contact'], 'div': div,
                 'steps_div_after_drop': lead, 'no_leak': bool(ok),
                 'off_world': {'success': A['success'], 'dead': A['dead_at']},
                 'on_world': {'success': B['success'], 'dead': B['dead_at']}})
    k += 1

  with open(os.path.join(args.out_dir, 'maskflip_manifest.json'), 'w') as f:
    json.dump({'ckpt': args.ckpt, 'step': step, 'seed': args.seed,
               'severity': args.severity, 'pairs': rows}, f, indent=2)
  print(f'\n{k} paired GIFs -> {args.out_dir}', flush=True)


if __name__ == '__main__':
  main()
