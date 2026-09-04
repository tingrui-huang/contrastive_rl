"""Smoke gate for the V4 rockfall-wait benchmark (crl/rockfall_wait_v4.py).

Run BEFORE the 300-episode teacher audit and before any dataset:

  T1 registration: 58-dim obs, goal in the (8, 0) cell at o[29:31] (d4rl
     in-cell goal noise), rocks parked on reset
  T2 mouth trigger: fires exactly once, at the first step with x >= MOUTH_X,
     for both latents; never in the west column
  T3 hiddenness: (a) u=clear V4 is step-for-step identical to u=clear V3-br
     under the same seed and actions (the inactive trigger is a flag);
     (b) u=active vs u=clear under the same 'go' actions diverge only after
     the trigger (rock teleports cannot touch the ant before they land)
  T4 hazard is real: do(go) | active kills > 0.9 (n=20)
  T5 the wait works: sighted policy | active: 0 deaths, success > 0.9,
     6 waves, rocks parked at the end, band entered after the window closed
  T6 the hold keeps the ant out of the band while the window is open (max x
     during the hold < HAZARD_X[0]) and drifts < 0.8 (84-step crouch)
  T7 the env's own latent stream (reset() without an argument) is the V3-br
     stream under the same seed (same rng seed offset)
  T8 window map: hold H steps at the mouth then go, H in {0,24,48,60,84}:
     deaths for H <= 48 (rocks still falling when the ant is inside),
     survival for H = 84 (the expert's hold)

Writes artifacts/rockfall_wait_v4/smoke.json. Exit 1 on any FAIL.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                       # noqa: E402
from crl import rockfall_wait_v4 as V4                 # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X          # noqa: E402
import rockfall_wait_v4_teacher as WT                  # noqa: E402
import tworoute_v3_teacher as TT                       # noqa: E402

OUT = os.path.join(WT.OUT, 'smoke.json')
RESULTS = []


def check(name, ok, detail):
  RESULTS.append({'test': name, 'pass': bool(ok), 'detail': detail})
  print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)


def rollout(env, teacher, u, intent, hold=None, horizon=WT.HORIZON):
  """Teacher rollout; hold=None uses the teacher's own intent logic, an int
  forces a zero-torque hold of that many steps at the mouth then 'go'.
  Returns (trace dict, info)."""
  o = env.reset(rockfall_active=u)
  teacher.fresh()
  obs, xs, holding_xy, info = [o.copy()], [], [], {}
  hold_left, latched = None, False
  for t in range(horizon):
    if hold is not None:
      if not latched and V4.RockfallWaitV4Env._at_mouth(float(o[0]), float(o[1])):
        latched, hold_left = True, hold
      if hold_left:
        hold_left -= 1
        holding_xy.append((float(o[0]), float(o[1])))
        a = np.zeros(8, np.float32)
      else:
        a = teacher.act(o, 'go')
    else:
      a = teacher.act(o, intent, revealed=env.revealed_rockfall_active)
      if teacher.holding:
        holding_xy.append((float(o[0]), float(o[1])))
    o, r, done, info = env.step(a)
    obs.append(o.copy())
    if done or r > 0:
      break
  return {'obs': np.asarray(obs), 'steps': t + 1, 'holding_xy': holding_xy,
          'hold_steps': teacher.hold_steps_done}, info


def main():
  cfg, teacher = WT.make_teacher()
  env = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)

  # T1
  o = env.reset(rockfall_active=False)
  info = env._info(False)
  d = env._env.data
  rock_x = [float(d.qpos[qa]) for qa in env._rock_qadr]
  check('T1 registration',
        o.shape == (58,) and abs(float(o[29]) - 8.0) < 1.5
        and abs(float(o[30])) < 1.5 and not info['rock_dropped']
        and all(x < -20 for x in rock_x) and info['trigger_step'] is None,
        f'obs {o.shape}, goal {o[29:31].tolist()}, rock x {rock_x}')

  # T2: trigger fires once at the first x >= MOUTH_X, both latents
  t2 = []
  for u in (False, True):
    o = env.reset(rockfall_active=u)
    teacher.fresh()
    first_cross, trig = None, None
    for t in range(WT.HORIZON):
      if first_cross is None and V4.RockfallWaitV4Env._at_mouth(
          float(o[0]), float(o[1])):
        first_cross = t
      o, r, done, info = env.step(teacher.act(o, 'go'))
      if trig is None and info['trigger_step'] is not None:
        trig = info['trigger_step']
      if done or r > 0:
        break
    t2.append((u, first_cross, trig, info['trigger_step']))
  check('T2 mouth trigger',
        all(fc is not None and tr == fc and tr_end == fc
            for _, fc, tr, tr_end in t2),
        f'(u, first x>=mouth, trigger_step first seen, final) = {t2}')

  # T3a: u=clear V4 == u=clear V3-br under the same seed and actions
  env3 = envs_mod.make_env(TT.env_name('br'), cfg, seed=7)
  env4 = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)
  _, t3 = TT.make_teacher('br')
  o3 = env3.reset(rockfall_active=False)
  o4 = env4.reset(rockfall_active=False)
  t3.fresh()
  maxdiff, steps3 = float(np.max(np.abs(o3 - o4))), 0
  for t in range(WT.HORIZON):
    a = t3.act(o3, 'shortcut')
    o3, r3, d3, i3 = env3.step(a)
    o4, r4, d4, i4 = env4.step(a)
    maxdiff = max(maxdiff, float(np.max(np.abs(o3 - o4))))
    steps3 = t + 1
    if d3 or r3 > 0 or d4 or r4 > 0:
      break
  check('T3a clear V4 == clear V3-br', maxdiff == 0.0 and i3['success']
        and i4['success'], f'max |obs diff| {maxdiff:.2e} over {steps3} steps')

  # T3b: same actions, u=active vs u=clear: identical up to the trigger
  envA = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)
  envC = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)
  oA = envA.reset(rockfall_active=True)
  oC = envC.reset(rockfall_active=False)
  teacher.fresh()
  div_step, trig_step, dead_step = None, None, None
  for t in range(WT.HORIZON):
    a = teacher.act(oC, 'go')
    oA, rA, dA, iA = envA.step(a)
    oC, rC, dC, iC = envC.step(a)
    if trig_step is None and iA['trigger_step'] is not None:
      trig_step = iA['trigger_step']
    if div_step is None and float(np.max(np.abs(oA - oC))) > 1e-6:
      div_step = t + 1
    if dA:
      dead_step = t + 1
      break
    if dC or rC > 0:
      break
  check('T3b active/clear identical until the trigger',
        trig_step is not None and (div_step is None or div_step > trig_step),
        f'trigger at {trig_step}, first |diff|>1e-6 at {div_step}, '
        f'death at {dead_step}')

  # T4: do(go) | active
  deaths, steps = 0, []
  for k in range(20):
    tr, info = rollout(env, teacher, True, 'go')
    deaths += int(bool(info.get('failure')))
    steps.append(tr['steps'])
  check('T4 do(go) | active kills', deaths >= 18,
        f'{deaths}/20 deaths, steps {min(steps)}-{max(steps)}')

  # T5 + T6: sighted wait | active
  rows = []
  for k in range(20):
    tr, info = rollout(env, teacher, True, 'wait')
    hx = [p[0] for p in tr['holding_xy']]
    drift = (float(np.hypot(tr['holding_xy'][-1][0] - tr['holding_xy'][0][0],
                            tr['holding_xy'][-1][1] - tr['holding_xy'][0][1]))
             if len(tr['holding_xy']) > 1 else 0.0)
    rows.append({'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'waves': info['rock_waves'],
                 'parked': not info['rock_dropped'],
                 'passed': info['rockfall_passed'],
                 'trigger_step': info['trigger_step'],
                 'band_entry_step': info['band_entry_step'],
                 'hold_steps': tr['hold_steps'],
                 'hold_max_x': max(hx) if hx else None,
                 'hold_drift': round(drift, 3), 'steps': tr['steps']})
  n_ok = sum(r['success'] for r in rows)
  n_dead = sum(r['failure'] for r in rows)
  after = [r['band_entry_step'] is not None and r['trigger_step'] is not None
           and r['band_entry_step'] >= r['trigger_step'] + V4.ROCKFALL_STEPS
           for r in rows]
  check('T5 sighted wait | active',
        n_dead == 0 and n_ok >= 18 and all(r['waves'] == 6 for r in rows)
        and all(r['parked'] and r['passed'] for r in rows) and all(after),
        f'{n_ok}/20 success, {n_dead} deaths, waves '
        f'{sorted(set(r["waves"] for r in rows))}, parked '
        f'{sum(r["parked"] for r in rows)}/20, band entry after window '
        f'{sum(after)}/20, steps {min(r["steps"] for r in rows)}-'
        f'{max(r["steps"] for r in rows)}')
  hmx = [r['hold_max_x'] for r in rows if r['hold_max_x'] is not None]
  drifts = [r['hold_drift'] for r in rows]
  check('T6 hold stays out of the band',
        hmx and max(hmx) < HAZARD_X[0] and max(drifts) < 0.8,
        f'max x while holding {max(hmx):.2f} (band at {HAZARD_X[0]}), '
        f'drift {min(drifts):.2f}-{max(drifts):.2f}, hold steps '
        f'{sorted(set(r["hold_steps"] for r in rows))}')

  # T7: env-drawn latent stream == V3-br stream under the same seed
  e4 = envs_mod.make_env(WT.ENV_NAME, cfg, seed=3)
  e3 = envs_mod.make_env(TT.env_name('br'), cfg, seed=3)
  u4 = [bool(e4.reset() is not None and e4.privileged_rockfall_active)
        for _ in range(40)]
  u3 = [bool(e3.reset() is not None and e3.privileged_rockfall_active)
        for _ in range(40)]
  check('T7 latent stream == V3-br', u4 == u3,
        f'first 40: {sum(u4)} active in V4, {sum(u3)} in V3-br, equal={u4 == u3}')

  # T8: window map
  t8 = {}
  for H in (0, 24, 48, 60, 84):
    dead = 0
    for k in range(10):
      _, info = rollout(env, teacher, True, 'go', hold=H)
      dead += int(bool(info.get('failure')))
    t8[H] = dead
  check('T8 window map (deaths/10 by hold)',
        t8[0] >= 9 and t8[24] >= 9 and t8[48] >= 8 and t8[84] == 0,
        f'{t8}')

  os.makedirs(WT.OUT, exist_ok=True)
  ok = all(r['pass'] for r in RESULTS)
  with open(OUT, 'w') as f:
    json.dump({'all_pass': ok, 'results': RESULTS,
               'wait_rows_active': rows, 'window_map': t8}, f, indent=2)
  print(('ALL PASS' if ok else 'SOME FAIL') + f' -> {OUT}', flush=True)
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
