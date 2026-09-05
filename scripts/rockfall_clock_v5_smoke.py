"""Smoke gate for the V5 rockfall-clock benchmark (crl/rockfall_clock_v5.py).

Run BEFORE the 300-episode teacher audit and before any dataset. V5 moves
the V4 rockfall onto its own clock (burst start t0 ~ U{0..30} per episode,
waves fall whether or not the ant is near the band), so on top of the V4
gates (registration, hiddenness, hazard real, the wait works, the hold
stays out of the band, latent stream) it must prove the SCHEDULE: that t0
is drawn every reset from its own rng, that the waves and the parking land
on the exact step indices the module doc promises, that a V4 active
episode is reproduced step for step when t0 is forced to V4's trigger
step, and that the sighted expert's overlap rule and the blind
always-wait reference behave as the spec says.

  T1  registration: 58-dim obs, goal in the (8, 0) cell at o[29:31], the
      'schedule' side channel, info carries mouth_step / rockfall_start and
      no 'trigger_step'; rocks parked on reset
  T2  schedule: 40 plain resets draw t0 in [T0_MIN, T0_MAX] for BOTH
      latents (info['rockfall_start'] None when inactive, env._t0 still
      set); forced t0 = 17 + active + the blind 'wait' intent: wave k+1
      first reported in the info of step index 17 + 12k (k = 0..5),
      rockfall_passed first True in the info of step index 89 (= t0 +
      ROCKFALL_STEPS) with the rocks parked from there on; forced inactive:
      no wave, no rock, ever
  T3a u=clear V5 == u=clear V4 == u=clear V3-br: the V5 expert's actions
      replayed open loop on V4 and V3-br under the same seed: |obs diff|
      == 0.0 over the whole episode, same success (the schedule rng is a
      separate stream, so the ant's reset noise is untouched)
  T3b V4 equivalence: a V4 active do(go) episode (trigger step T, actions,
      death step) replayed on V5 with reset(rockfall_active=True,
      rockfall_start=T): |obs diff| == 0.0 up to and including the death
      step, same death step, same rock_waves
  T3c hiddenness: the same 'go' actions under forced active (natural t0)
      vs clear are identical (|diff| <= 1e-6) until the first step at which
      a dropped rock is within reach of the ant (env.rock_within_reach; a
      rock can hit and bounce off inside a step without a flagged contact,
      see the module doc); the exact-0 divergence step, the first
      within-reach step, the first flagged contact, the band entry and the
      death step are reported
  T4  hazard is real: do(go) | active, natural t0: deaths >= 18/20
  T5  the sighted expert | active (shortcut): 20/20 success, 0 deaths,
      decision 'wait' in all, 6 waves, rocks parked, band entered at or
      after the release step, hold steps within [40, 95]
  T6  the hold keeps the ant out of the band: max x while holding <
      HAZARD_X[0] in all T5 episodes and no rock contact while holding
  T7  streams: the env-drawn latent over 200 plain resets equals V4's and
      V3-br's under the same seed; the t0 sequence over 60 resets is
      identical under forced clear and forced active
  T8  window map: forced hold H in {0, 30, 60, 90} at the mouth then go,
      active, natural t0, 10 episodes each (expected deaths ~10, 10,
      mixed, 0), plus the blind always-wait reference (hold until step
      BLIND_WAIT_UNTIL) under the active latent: 0 deaths, 20/20 success
  T9  detour: forced route 'detour', 10 active + 10 clear: success >= 18,
      band never entered, 0 deaths, rocks fall when active but never touch
  T10 the general rule: forced t0 = 200 (burst after the crossing), active:
      the expert decides 'go', 0 hold steps, success; forced t0 = 0: the
      expert decides 'wait', release step 72, success

Writes artifacts/rockfall_clock_v5/smoke.json. Exit 1 on any FAIL.

Run:  python scripts/rockfall_clock_v5_smoke.py [--gates T1,T2,...]
      (--gates runs a subset; all_pass in the json then covers only those)
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                       # noqa: E402
from crl import rockfall_clock_v5 as V5                # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X          # noqa: E402
import rockfall_clock_v5_teacher as CT                 # noqa: E402
import rockfall_wait_v4_teacher as WT                  # noqa: E402
import tworoute_v3_teacher as TT                       # noqa: E402

OUT = os.path.join(CT.OUT, 'smoke.json')
GATES = ('T1', 'T2', 'T3a', 'T3b', 'T3c', 'T4', 'T5', 'T6', 'T7', 'T8',
         'T9', 'T10')
#: hiddenness tolerance on the float32 obs (the V4 smoke's; see the module
#: doc of crl/rockfall_clock_v5.py for why exact 0 is not gated).
HIDDEN_TOL = 1e-6
RESULTS = []


def check(name, ok, detail):
  RESULTS.append({'test': name, 'pass': bool(ok), 'detail': detail})
  print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)


def _at_mouth(o):
  return V5.RockfallClockV5Env._at_mouth(float(o[0]), float(o[1]))


def rollout(env, teacher, u, intent, route='shortcut', hold=None,
            rockfall_start=None, horizon=CT.HORIZON):
  """Teacher rollout. intent None uses the expert's own decision, 'go' /
  'wait' / 'detour' force it; an int ``hold`` bypasses the teacher for a
  zero-torque hold of that many steps at the mouth and then drives 'go'.
  Returns (trace dict, final info). The trace keeps the obs, the actions,
  every step info, the xy while holding and the rock-contact flags of the
  holding steps (T2 reads the infos, T3a replays the actions, T6 reads the
  hold rows)."""
  o = env.reset(rockfall_active=None if u is None else bool(u),
                rockfall_start=rockfall_start)
  teacher.fresh(route=route)
  obs, acts, infos, holding_xy, hold_contact = [o.copy()], [], [], [], []
  info, hold_left, latched, t = {}, None, False, 0
  for t in range(horizon):
    was_holding = False
    if hold is not None:
      if not latched and _at_mouth(o):
        latched, hold_left = True, hold
      if hold_left:
        hold_left -= 1
        was_holding = True
        a = np.zeros(8, np.float32)
      else:
        a = teacher.act(o, env.schedule, 'go')
    else:
      a = teacher.act(o, env.schedule, intent)
      was_holding = teacher.holding
    if was_holding:
      holding_xy.append((float(o[0]), float(o[1])))
    o, r, done, info = env.step(a)
    obs.append(o.copy())
    acts.append(np.asarray(a, np.float32))
    infos.append(info)
    if was_holding:
      hold_contact.append(bool(info['rock_contact']))
    if done or r > 0:
      break
  return {'obs': np.asarray(obs), 'actions': acts, 'infos': infos,
          'steps': t + 1, 'holding_xy': holding_xy,
          'hold_contact': hold_contact,
          'hold_steps': teacher.hold_steps_done,
          'decision': teacher.decision,
          'release_step': teacher.release_step}, info


def _replay(env, actions, u, rockfall_start=None):
  """Open-loop replay of recorded actions on a fresh reset; returns (obs
  array, final info, steps). ``rockfall_start`` is V5-only: the V4 and
  V3-br envs replayed in T3a take ``rockfall_active`` alone."""
  kw = {} if rockfall_start is None else {'rockfall_start': rockfall_start}
  o = env.reset(rockfall_active=u, **kw)
  obs, info, n = [o.copy()], {}, 0
  for a in actions:
    o, r, done, info = env.step(a)
    obs.append(o.copy())
    n += 1
    if done or r > 0:
      break
  return np.asarray(obs), info, n


def _maxdiff(a, b):
  n = min(len(a), len(b))
  return float(np.max(np.abs(a[:n] - b[:n]))) if n else 0.0


def _t5_row(tr, info):
  hx = [p[0] for p in tr['holding_xy']]
  return {'success': bool(info.get('success')),
          'failure': bool(info.get('failure')),
          'decision': tr['decision'],
          'waves': info['rock_waves'],
          'parked': not info['rock_dropped'],
          'passed': info['rockfall_passed'],
          'rockfall_start': info['rockfall_start'],
          'rockfall_end': info['rockfall_end'],
          'mouth_step': info['mouth_step'],
          'band_entry_step': info['band_entry_step'],
          'release_step': tr['release_step'],
          'hold_steps': tr['hold_steps'],
          'hold_max_x': max(hx) if hx else None,
          'hold_contact': any(tr['hold_contact']),
          'steps': tr['steps']}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--gates', default=','.join(GATES),
                  help='comma-separated subset of ' + ','.join(GATES))
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  gates = tuple(g.strip() for g in args.gates.split(',') if g.strip())
  bad = [g for g in gates if g not in GATES]
  if bad:
    raise SystemExit(f'unknown gates {bad}; choose from {GATES}')

  cfg, teacher = CT.make_teacher()
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=7)
  extra = {}

  # T1
  if 'T1' in gates:
    o = env.reset(rockfall_active=False)
    info = env._info(False)
    d = env._env.data
    rock_x = [float(d.qpos[qa]) for qa in env._rock_qadr]
    sched = env.schedule
    check('T1 registration',
          o.shape == (58,) and abs(float(o[29]) - 8.0) < 1.5
          and abs(float(o[30])) < 1.5 and not info['rock_dropped']
          and all(x < -20 for x in rock_x)
          and isinstance(sched, dict)
          and set(sched) == {'t', 'active', 'start', 'end'}
          and 'mouth_step' in info and 'rockfall_start' in info
          and 'trigger_step' not in info and 'rock_triggered' not in info,
          f'obs {o.shape}, goal {o[29:31].tolist()}, rock x {rock_x}, '
          f'schedule {sched}, info keys {sorted(info)}')

  # T2: t0 drawn every reset for both latents; forced t0 = 17 timetable
  if 'T2' in gates:
    draws = []
    for _ in range(40):
      env.reset()
      info = env._info(False)
      draws.append((bool(env.privileged_rockfall_active), int(env._t0),
                    info['rockfall_start']))
    draws_ok = all(V5.T0_MIN <= t0 <= V5.T0_MAX
                   and (start == t0 if u else start is None)
                   for u, t0, start in draws)
    n_inactive = sum(not u for u, _, _ in draws)
    tr, info = rollout(env, teacher, True, 'wait', rockfall_start=17)
    infos = tr['infos']

    def first_idx(pred):
      hits = [i for i, inf in enumerate(infos) if pred(inf)]
      return hits[0] if hits else None

    wave_idx = [first_idx(lambda inf, k=k: inf['rock_waves'] == k + 1)
                for k in range(6)]
    passed_idx = first_idx(lambda inf: inf['rockfall_passed'])
    parked_after = (passed_idx is not None
                    and all(not inf['rock_dropped']
                            for inf in infos[passed_idx:]))
    waves_ok = wave_idx == [17 + V5.WAVE_PERIOD * k for k in range(6)]
    tr_in, info_in = rollout(env, teacher, False, 'go', rockfall_start=17)
    inactive_ok = (all(inf['rock_waves'] == 0 and not inf['rock_dropped']
                       and inf['rockfall_start'] is None
                       for inf in tr_in['infos'])
                   and info_in.get('success'))
    check('T2 schedule',
          draws_ok and waves_ok and passed_idx == 17 + V5.ROCKFALL_STEPS
          and parked_after and not info.get('failure') and inactive_ok,
          f'40 plain resets: t0 range '
          f'{min(t for _, t, _ in draws)}-{max(t for _, t, _ in draws)}, '
          f'{n_inactive} inactive (start None), ok={draws_ok}; forced '
          f't0=17 + blind wait: wave first-seen step idx {wave_idx}, '
          f'passed first at {passed_idx}, parked from there on '
          f'{parked_after}, alive={not info.get("failure")}, steps '
          f'{tr["steps"]}; forced inactive: no rock ever={inactive_ok}')
    extra['t2'] = {'draws': draws, 'wave_first_idx': wave_idx,
                   'passed_first_idx': passed_idx}

  # T3a: u=clear V5 == u=clear V4 == u=clear V3-br under the same actions
  if 'T3a' in gates:
    env5 = envs_mod.make_env(CT.ENV_NAME, cfg, seed=7)
    env4 = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)
    env3 = envs_mod.make_env(TT.env_name('br'), cfg, seed=7)
    tr5, i5 = rollout(env5, teacher, False, None)
    o4, i4, n4 = _replay(env4, tr5['actions'], False)
    o3, i3, n3 = _replay(env3, tr5['actions'], False)
    d54, d53 = _maxdiff(tr5['obs'], o4), _maxdiff(tr5['obs'], o3)
    check('T3a clear V5 == clear V4 == clear V3-br',
          d54 == 0.0 and d53 == 0.0 and n4 == tr5['steps']
          and n3 == tr5['steps'] and i5['success'] and i4['success']
          and i3['success'] and tr5['decision'] == 'go',
          f'max |obs diff| V5-V4 {d54:.2e}, V5-V3br {d53:.2e} over '
          f'{tr5["steps"]}/{n4}/{n3} steps, success '
          f'{i5["success"]}/{i4["success"]}/{i3["success"]}, decision '
          f'{tr5["decision"]}')

  # T3b: V4 active do(go) reproduced on V5 with t0 = V4's trigger step
  if 'T3b' in gates:
    env4 = envs_mod.make_env(WT.ENV_NAME, cfg, seed=7)
    env5 = envs_mod.make_env(CT.ENV_NAME, cfg, seed=7)
    _, t4 = WT.make_teacher()
    o = env4.reset(rockfall_active=True)
    t4.fresh()
    obs4, acts4, i4, n4 = [o.copy()], [], {}, 0
    for t in range(CT.HORIZON):
      a = t4.act(o, 'go')
      o, r, done, i4 = env4.step(a)
      obs4.append(o.copy())
      acts4.append(np.asarray(a, np.float32))
      n4 = t + 1
      if done or r > 0:
        break
    obs4 = np.asarray(obs4)
    T = i4['trigger_step']
    o5, i5, n5 = _replay(env5, acts4, True, rockfall_start=T)
    d45 = _maxdiff(obs4, o5)
    check('T3b V4 active == V5 with t0 = V4 trigger step',
          T is not None and d45 == 0.0 and n5 == n4
          and len(o5) == len(obs4)
          and bool(i5['failure']) == bool(i4['failure'])
          and i5['rock_waves'] == i4['rock_waves']
          and i5['rockfall_start'] == T,
          f'V4 trigger step {T}, death V4 {i4["failure"]} at {n4}, V5 '
          f'{i5["failure"]} at {n5}, max |obs diff| {d45:.2e}, waves '
          f'V4 {i4["rock_waves"]} / V5 {i5["rock_waves"]}')
    extra['t3b'] = {'trigger_step': T, 'v4_steps': n4, 'v5_steps': n5,
                    'v4_failure': bool(i4['failure']),
                    'v5_failure': bool(i5['failure'])}

  # T3c: same 'go' actions, forced active (natural t0) vs clear
  if 'T3c' in gates:
    envA = envs_mod.make_env(CT.ENV_NAME, cfg, seed=7)
    envC = envs_mod.make_env(CT.ENV_NAME, cfg, seed=7)
    oA = envA.reset(rockfall_active=True)
    oC = envC.reset(rockfall_active=False)
    teacher.fresh()
    t0 = envA.privileged_rockfall_start
    div_exact, div_tol, reach_step, contact_step, dead_step = (None,) * 5
    band_step, n = None, 0
    for t in range(CT.HORIZON):
      a = teacher.act(oC, envC.schedule, 'go')
      oA, rA, dA, iA = envA.step(a)
      oC, rC, dC, iC = envC.step(a)
      n = t + 1
      diff = float(np.max(np.abs(oA - oC)))
      if div_exact is None and diff > 0.0:
        div_exact = n
      if div_tol is None and diff > HIDDEN_TOL:
        div_tol = n
      if reach_step is None and envA.rock_within_reach:
        reach_step = n
      if contact_step is None and iA['rock_contact']:
        contact_step = n
      band_step = iA['band_entry_step']
      if dA:
        dead_step = n
        break
      if dC or rC > 0:
        break
    check('T3c active/clear identical until a rock is within reach',
          div_tol is None or (reach_step is not None
                              and div_tol >= reach_step),
          f't0 {t0}, first |diff|>0 at {div_exact}, first |diff|>1e-6 at '
          f'{div_tol}, first rock within reach at {reach_step}, first '
          f'flagged contact at {contact_step}, band entry {band_step}, '
          f'death at {dead_step}, {n} steps')
    extra['t3c'] = {'t0': t0, 'div_exact': div_exact, 'div_tol': div_tol,
                    'reach_step': reach_step, 'contact_step': contact_step,
                    'band_entry_step': band_step, 'dead_step': dead_step}

  # T4: do(go) | active, natural t0
  if 'T4' in gates:
    deaths, steps, t0s = 0, [], []
    for k in range(20):
      tr, info = rollout(env, teacher, True, 'go')
      deaths += int(bool(info.get('failure')))
      steps.append(tr['steps'])
      t0s.append(info['rockfall_start'])
    check('T4 do(go) | active kills', deaths >= 18,
          f'{deaths}/20 deaths, steps {min(steps)}-{max(steps)}, t0 '
          f'{min(t0s)}-{max(t0s)}')

  # T5 + T6: the sighted expert | active (shortcut)
  rows = []
  if 'T5' in gates or 'T6' in gates:
    for k in range(20):
      tr, info = rollout(env, teacher, True, None)
      rows.append(_t5_row(tr, info))
    extra['wait_rows_active'] = rows
  if 'T5' in gates:
    n_ok = sum(r['success'] for r in rows)
    n_dead = sum(r['failure'] for r in rows)
    after = [r['band_entry_step'] is not None and r['release_step'] is not None
             and r['band_entry_step'] >= r['release_step'] for r in rows]
    holds = [r['hold_steps'] for r in rows]
    check('T5 sighted expert | active',
          n_dead == 0 and n_ok == 20
          and all(r['decision'] == 'wait' for r in rows)
          and all(r['waves'] == 6 for r in rows)
          and all(r['parked'] and r['passed'] for r in rows) and all(after)
          and all(40 <= h <= 95 for h in holds),
          f'{n_ok}/20 success, {n_dead} deaths, decisions '
          f'{sorted(set(str(r["decision"]) for r in rows))}, waves '
          f'{sorted(set(r["waves"] for r in rows))}, parked '
          f'{sum(r["parked"] for r in rows)}/20, band entry >= release '
          f'{sum(after)}/20, hold steps min/mean/max {min(holds)}/'
          f'{np.mean(holds):.1f}/{max(holds)}, steps '
          f'{min(r["steps"] for r in rows)}-{max(r["steps"] for r in rows)}')
  if 'T6' in gates:
    hmx = [r['hold_max_x'] for r in rows if r['hold_max_x'] is not None]
    n_contact = sum(r['hold_contact'] for r in rows)
    check('T6 hold stays out of the band',
          bool(hmx) and len(hmx) == len(rows) and max(hmx) < HAZARD_X[0]
          and n_contact == 0,
          f'max x while holding {max(hmx) if hmx else None} (band at '
          f'{HAZARD_X[0]}), episodes with a hold {len(hmx)}/{len(rows)}, '
          f'rock contact while holding in {n_contact} episodes')

  # T7: latent stream == V4 == V3-br; t0 stream independent of the latent
  if 'T7' in gates:
    e5 = envs_mod.make_env(CT.ENV_NAME, cfg, seed=3)
    e4 = envs_mod.make_env(WT.ENV_NAME, cfg, seed=3)
    e3 = envs_mod.make_env(TT.env_name('br'), cfg, seed=3)
    u5 = [bool(e5.reset() is not None and e5.privileged_rockfall_active)
          for _ in range(200)]
    u4 = [bool(e4.reset() is not None and e4.privileged_rockfall_active)
          for _ in range(200)]
    u3 = [bool(e3.reset() is not None and e3.privileged_rockfall_active)
          for _ in range(200)]
    eC = envs_mod.make_env(CT.ENV_NAME, cfg, seed=5)
    eA = envs_mod.make_env(CT.ENV_NAME, cfg, seed=5)
    t0C = [int(eC.reset(rockfall_active=False) is not None and eC._t0)
           for _ in range(60)]
    t0A = [int(eA.reset(rockfall_active=True) is not None and eA._t0)
           for _ in range(60)]
    check('T7 streams', u5 == u4 and u5 == u3 and t0C == t0A
          and all(V5.T0_MIN <= t <= V5.T0_MAX for t in t0C),
          f'latent over 200 resets: {sum(u5)} active in V5, {sum(u4)} in '
          f'V4, {sum(u3)} in V3-br, equal={u5 == u4 and u5 == u3}; t0 over '
          f'60 resets clear vs active equal={t0C == t0A}, {len(set(t0C))} '
          f'distinct values, range {min(t0C)}-{max(t0C)}')
    extra['t7'] = {'t0_stream_seed5': t0C}

  # T8: window map + the blind always-wait reference
  if 'T8' in gates:
    t8 = {}
    for H in (0, 30, 60, 90):
      dead = 0
      for k in range(10):
        _, info = rollout(env, teacher, True, 'go', hold=H)
        dead += int(bool(info.get('failure')))
      t8[H] = dead
    blind = []
    for k in range(20):
      tr, info = rollout(env, teacher, True, 'wait')
      blind.append({'success': bool(info.get('success')),
                    'failure': bool(info.get('failure')),
                    'hold_steps': tr['hold_steps'],
                    'release_step': tr['release_step'],
                    'rockfall_end': info['rockfall_end'],
                    'band_entry_step': info['band_entry_step'],
                    'steps': tr['steps']})
    b_ok = sum(b['success'] for b in blind)
    b_dead = sum(b['failure'] for b in blind)
    check('T8 window map (deaths/10 by hold) + blind always-wait',
          t8[0] >= 9 and t8[30] >= 9 and t8[90] == 0 and b_dead == 0
          and b_ok == 20
          and all(b['release_step'] == V5.BLIND_WAIT_UNTIL for b in blind),
          f'{t8}; blind wait | active: {b_ok}/20 success, {b_dead} deaths, '
          f'release {sorted(set(b["release_step"] for b in blind))}, hold '
          f'steps {min(b["hold_steps"] for b in blind)}-'
          f'{max(b["hold_steps"] for b in blind)}, steps '
          f'{min(b["steps"] for b in blind)}-{max(b["steps"] for b in blind)}')
    extra['window_map'] = t8
    extra['blind_wait_active'] = blind

  # T9: the detour never crosses the band and never meets a rock
  if 'T9' in gates:
    drows = []
    for u in (True, False):
      for k in range(10):
        tr, info = rollout(env, teacher, u, None, route='detour')
        drows.append({'rockfall_active': u,
                      'success': bool(info.get('success')),
                      'failure': bool(info.get('failure')),
                      'entered_hazard': bool(info.get('entered_hazard')),
                      'route_realized': info.get('route'),
                      'waves': info['rock_waves'],
                      'parked': not info['rock_dropped'],
                      'any_contact': any(inf['rock_contact']
                                         for inf in tr['infos']),
                      'mouth_step': info['mouth_step'],
                      'steps': tr['steps']})
    d_ok = sum(r['success'] for r in drows)
    d_dead = sum(r['failure'] for r in drows)
    act_rows = [r for r in drows if r['rockfall_active']]
    check('T9 detour',
          d_ok >= 18 and d_dead == 0
          and not any(r['entered_hazard'] for r in drows)
          and not any(r['any_contact'] for r in drows)
          and all(r['waves'] == 6 and r['parked'] for r in act_rows)
          and all(r['waves'] == 0 for r in drows
                  if not r['rockfall_active']),
          f'{d_ok}/20 success, {d_dead} deaths, band entered '
          f'{sum(r["entered_hazard"] for r in drows)}/20, routes '
          f'{sorted(set(str(r["route_realized"]) for r in drows))}, waves '
          f'active {sorted(set(r["waves"] for r in act_rows))}, contact '
          f'{sum(r["any_contact"] for r in drows)}, steps '
          f'{min(r["steps"] for r in drows)}-{max(r["steps"] for r in drows)}')
    extra['detour_rows'] = drows

  # T10: the overlap rule at both ends
  if 'T10' in gates:
    tr_late, i_late = rollout(env, teacher, True, None, rockfall_start=200)
    tr_zero, i_zero = rollout(env, teacher, True, None, rockfall_start=0)
    check('T10 general rule (t0 = 200 -> go; t0 = 0 -> wait until 72)',
          tr_late['decision'] == 'go' and tr_late['hold_steps'] == 0
          and i_late['success'] and not i_late['failure']
          and tr_zero['decision'] == 'wait'
          and tr_zero['release_step'] == V5.ROCKFALL_STEPS + CT.RELEASE_MARGIN
          and i_zero['success'] and not i_zero['failure'],
          f't0=200: decision {tr_late["decision"]}, hold '
          f'{tr_late["hold_steps"]}, success {i_late["success"]}, waves '
          f'{i_late["rock_waves"]}, steps {tr_late["steps"]}; t0=0: '
          f'decision {tr_zero["decision"]}, release '
          f'{tr_zero["release_step"]}, hold {tr_zero["hold_steps"]}, '
          f'success {i_zero["success"]}, band entry '
          f'{i_zero["band_entry_step"]}, steps {tr_zero["steps"]}')

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  ok = all(r['pass'] for r in RESULTS)
  with open(args.out, 'w') as f:
    json.dump({'all_pass': ok, 'gates_run': list(gates),
               'all_gates_run': tuple(gates) == GATES,
               'results': RESULTS, **extra}, f, indent=2)
  print(('ALL PASS' if ok else 'SOME FAIL') + f' -> {args.out}', flush=True)
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
