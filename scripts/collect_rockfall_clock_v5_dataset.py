"""Offline dataset collector for the V5 rockfall-clock AntMaze benchmark.

Port of scripts/collect_rockfall_wait_v4_dataset.py onto
crl/rockfall_clock_v5.py: the rockfall runs on ITS OWN CLOCK (an active
latent drops six waves starting at t0 ~ U{0..30}, whether or not the ant is
near the band), and the sighted teacher (scripts/rockfall_clock_v5_teacher.py)
knows the timetable from t = 0 but only acts on it at the mouth line: it
walks straight through when the burst misses its crossing (every clear
episode) and HOLDS at the mouth (zero torque) until the rocks are parked
when the burst overlaps it (every active shortcut episode in the natural
t0 range).

Two datasets from the same collector, chosen by --p-far:

  p_far = 0.0   antmaze_rockfall_clock_v5        every episode takes the BR
                                                 shortcut
  p_far = 0.05  antmaze_rockfall_clock_v5_far05  ~5% of the episodes take the
                                                 V3-br DETOUR (north column,
                                                 top row, east column)

The route coin is drawn by THIS collector's recorded rng (seed+7000),
independently of the latent coin (seed+5000), exactly like the discrete
WindyCorridor expert's ``NEAR if rng.random() < p_near else FAR``. The
detour never crosses the band and never waits: it is pure coverage of the
safe alternative for the learner and carries no information about the
latent (checked by the second permutation test below).

  learner npz: obs [N, L+1, 58] float32, act [N, L+1, 8] float32 (last row
               dummy zeros), lengths [N] (valid obs rows), eval_goals [N, 2],
               meta (json string). NOTHING privileged: no latent, no clock,
               no rocks, no route label.
  sidecar npz: per-episode audit fields (rockfall_active, route, intent,
               outcome, t0 / mouth / band-entry steps, hold steps, ...) +
               step torso traces. NEVER a training input; consumed only by
               the audits.

Every episode starts from ONE canonical pose, so the t=0 observation
carries no intent information -- checked, not asserted, by the permutation
leak test (verbatim from V2/V3/V4), run by latent AND, when p_far > 0, by
intended route.

What the u-blind learner sees: ~95% (100%) of the kept episodes take the
corridor; at the mouth, ~70% of those keep walking and ~30% stand still for
~60-95 steps (until t0 + 72) and then walk; the rest take a ~225-step
detour. All of them reach the goal.

Pre-registered predictions (gamma=0.99, 0.99**steps on success else 0),
derived from the V4 driver numbers (go ~77 steps, 0.70 sparse) and the V5
blind always-wait release at step BLIND_WAIT_UNTIL = 102: always-go 0.70 /
~0.32; always-wait ~1.0 / ~0.99**(102 + 62) ~ 0.19; always-detour ~1.0 /
0.99**225 ~ 0.10; oracle ~1.0 / 0.70*0.99**77 + 0.30*0.99**(~105+62) ~ 0.38.
The teacher audit + causal audit measure the real ones.

Usage: python scripts/collect_rockfall_clock_v5_dataset.py --p-far 0.0
       python scripts/collect_rockfall_clock_v5_dataset.py --p-far 0.05
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod              # noqa: E402
from crl import rockfall_clock_v5 as V5       # noqa: E402
import rockfall_clock_v5_teacher as CT        # noqa: E402

OUT_ROOT = CT.OUT
NAME_BASE = 'antmaze_rockfall_clock_v5'
HORIZON = CT.HORIZON

#: analytic predictions (see module doc); the audits report measured values.
PREDICTED_REFS = {
    'sparse_success': {'always_go': 0.70, 'always_wait': 1.0,
                       'always_detour': 1.0, 'oracle': 1.0},
    'discounted_gamma_0.99': {'always_go': 0.32, 'always_wait': 0.19,
                              'always_detour': 0.10, 'oracle': 0.38,
                              'best_blind': 'always_go'},
    'discounted_definition': '0.99**steps on success, else 0.0; mean over '
                             'episodes',
    'derivation': 'V4 go driver: 0.70 sparse, ~77 steps; blind wait '
                  f'releases at step {V5.BLIND_WAIT_UNTIL} then ~62 steps '
                  'to the goal: 0.99**164 ~ 0.19; detour ~225 steps: '
                  '0.99**225 ~ 0.10; oracle holds until t0 + 72 (~105) '
                  'on active episodes only',
}


def dataset_name(p_far):
  """p_far == 0 -> the plain set; else the far{pp} suffix (far05)."""
  if p_far <= 0.0:
    return NAME_BASE
  return f'{NAME_BASE}_far{int(round(p_far * 100)):02d}'


def _dprime(g0, g1):
  pooled = np.sqrt((g0.var(0, ddof=1) + g1.var(0, ddof=1)) / 2.0) + 1e-9
  return np.abs(g0.mean(0) - g1.mean(0)) / pooled


def leak_test(o0, mask, name0, name1):
  """MEASURED leak check on the t=0 observation, permutation null
  (verbatim from the V2/V3/V4 collectors); ``mask`` splits the episodes
  into two groups (``name0`` where True, ``name1`` where False)."""
  m = np.asarray(mask, bool)
  if m.sum() <= 1 or (~m).sum() <= 1:
    return {'passes': None, 'reason': f'a group is empty ({name0} '
                                      f'{int(m.sum())}, {name1} '
                                      f'{int((~m).sum())})'}
  d = _dprime(o0[m], o0[~m])
  rng = np.random.default_rng(0)
  k, B = int(m.sum()), 2000
  null = np.empty(B)
  for b in range(B):
    p = rng.permutation(len(o0))
    null[b] = _dprime(o0[p][:k], o0[p][k:]).max()
  p95 = float(np.percentile(null, 95))
  return {'test': 'permutation null on per-dim d-prime at t=0, B=2000',
          f'n_{name0}': int(m.sum()), f'n_{name1}': int((~m).sum()),
          'max_dprime': round(float(d.max()), 4),
          'argmax_dim': int(d.argmax()),
          'null_p95': round(p95, 4),
          'n_dims_above_null_p95': int((d > p95).sum()),
          'p_value': round(float((null >= d.max()).mean()), 4),
          'passes': bool(d.max() <= p95)}


def _print_leak(label, leak):
  if leak.get('passes') is None:
    print(f'{label}: SKIPPED ({leak.get("reason")})', flush=True)
    return
  print(f"{label}: max d'={leak['max_dprime']} (dim {leak['argmax_dim']}) "
        f"vs chance p95 {leak['null_p95']}, p={leak['p_value']} -> "
        f"{'PASS' if leak['passes'] else 'FAIL'}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=400)
  ap.add_argument('--seed', type=int, default=606)
  ap.add_argument('--p-active', type=float, default=CT.P_ACTIVE)
  ap.add_argument('--p-far', type=float, default=0.0,
                  help='detour coin (0.0 -> plain set, 0.05 -> far05)')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=None)
  args = ap.parse_args()
  name = dataset_name(args.p_far)
  out_dir = args.out_dir or os.path.join(OUT_ROOT, 'dataset')
  os.makedirs(out_dir, exist_ok=True)

  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = int(args.horizon)
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=args.seed)
  #: two recorded coins of the collector, independent of each other and of
  #: the env's own streams: the latent (seed+5000) and the route (seed+7000).
  u_rng = np.random.default_rng(args.seed + 5000)
  route_rng = np.random.default_rng(args.seed + 7000)

  N, L = args.episodes, args.horizon + 1
  obs = np.zeros((N, L, 58), np.float32)
  act = np.zeros((N, L, 8), np.float32)
  lengths = np.zeros(N, np.int64)
  eval_goals = np.zeros((N, 2), np.float32)
  tx = np.full((N, L), np.nan, np.float32)
  ty = np.full((N, L), np.nan, np.float32)
  rows = []
  discarded = []

  e = 0
  while e < N:
    u = bool(u_rng.random() < args.p_active)
    route = 'detour' if route_rng.random() < args.p_far else 'shortcut'
    o = env.reset(rockfall_active=u)
    teacher.fresh(route=route)
    obs[e, 0] = o
    tx[e, 0], ty[e, 0] = o[0], o[1]
    eval_goals[e] = o[29:31]
    ret, info = 0.0, {}
    for t in range(args.horizon):
      #: the expert knows the timetable from t=0 (env.schedule, privileged)
      #: but only acts on it at the mouth; the detour never reads it
      a = teacher.act(o, env.schedule)
      o, r, done, info = env.step(a)
      act[e, t] = a
      obs[e, t + 1] = o
      tx[e, t + 1], ty[e, t + 1] = o[0], o[1]
      ret += float(r)
      if done or r > 0:
        break
    #: what the expert did, for the sidecar: 'detour' on the detour route,
    #: else the mouth decision, or 'go' if it never reached the mouth (a
    #: redraw below)
    if route == 'detour':
      intent = 'detour'
    else:
      intent = teacher.decision or 'go'
      if info.get('mouth_step') is not None:
        #: natural t0 <= T0_MAX always overlaps the crossing: the sighted
        #: rule is 'wait' iff active (see the teacher's module doc)
        assert intent == ('wait' if u else 'go'), (e, u, intent)
    #: REDRAW an uncommitted episode (V3/V4 rule, per route): a shortcut
    #: episode that never entered the band, or a detour episode that never
    #: reached the north column, is a demonstrator failure (the ant
    #: shuffled in the start cell for the whole horizon), not a
    #: demonstration. Committed timeouts are kept.
    committed = (bool(info.get('entered_hazard')) if route == 'shortcut'
                 else info.get('route') == 'detour')
    if not committed:
      discarded.append({'rockfall_active': u, 'route': route,
                        'intent': intent,
                        'final_xy': [round(float(tx[e, t + 1]), 3),
                                     round(float(ty[e, t + 1]), 3)],
                        'ep_length': int(t + 1)})
      obs[e, :] = 0.0
      act[e, :] = 0.0
      tx[e, :] = np.nan
      ty[e, :] = np.nan
      continue
    lengths[e] = t + 2                     # valid obs rows (0 .. t+1)
    rows.append({'episode_id': e, 'rockfall_active': u,
                 'route': route, 'route_realized': info.get('route'),
                 'intent': intent,
                 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'rock_dropped': bool(info.get('rock_dropped')),
                 'rockfall_passed': bool(info.get('rockfall_passed')),
                 'rock_waves': int(info.get('rock_waves', 0)),
                 'rockfall_start': info.get('rockfall_start'),
                 'mouth_step': info.get('mouth_step'),
                 'band_entry_step': info.get('band_entry_step'),
                 'hold_steps': int(teacher.hold_steps_done),
                 'release_step': teacher.release_step,
                 'return': ret, 'ep_length': int(t + 1)})
    e += 1
    if e % 50 == 0:
      print(f'  {e}/{N} episodes ({len(discarded)} discarded)', flush=True)
  print(f'discarded {len(discarded)} uncommitted episodes '
        f'({len(discarded) / (len(discarded) + N):.3f} of draws)', flush=True)
  #: the sighted expert never dies (teacher audit gate); a death here is a
  #: collector bug, not a data point.
  n_deaths = int(sum(r['failure'] for r in rows))
  assert n_deaths == 0, f'{n_deaths} expert deaths in the kept episodes'

  #: MEASURED leak checks on the t=0 observation: by latent (as V4) and,
  #: when the route coin is live, by intended route. Both must pass: one
  #: canonical pose, nothing at t=0 tells the learner what comes.
  o0 = obs[:, 0, :]
  latent = np.array([r['rockfall_active'] for r in rows], bool)
  routes = np.array([r['route'] for r in rows])
  leak_latent = leak_test(o0, ~latent, 'clear', 'active')
  _print_leak('latent-leak check (t=0 obs by latent)', leak_latent)
  if args.p_far > 0.0:
    leak_route = leak_test(o0, routes == 'shortcut', 'shortcut', 'detour')
    _print_leak('route-leak check (t=0 obs by route)', leak_route)
  else:
    leak_route = {'passes': None, 'reason': 'p_far == 0: no detour episodes'}
  leak_ok = (leak_latent.get('passes') is not False
             and leak_route.get('passes') is not False)
  print(f"leak checks overall: {'PASS' if leak_ok else 'FAIL'}", flush=True)

  intents = [r['intent'] for r in rows]
  n_wait = intents.count('wait')
  n_go = intents.count('go')
  n_detour = intents.count('detour')
  n_active = int(latent.sum())
  meta = {'name': name, 'env': CT.ENV_NAME,
          'obs_dim': 29, 'goal_dim': 29, 'action_dim': 8,
          'horizon': args.horizon, 'p_active': args.p_active,
          'p_far': args.p_far,
          'collection_seed': args.seed,
          'teacher': 'sighted rockfall_clock_v5_teacher (knows the '
                     'timetable from t=0 via env.schedule; at the mouth: '
                     'burst overlaps the crossing -> hold until the rocks '
                     'are parked, else go); detour with probability p_far '
                     'from the collector\'s route coin, independent of the '
                     'latent; one canonical pose (see scripts/'
                     'rockfall_clock_v5_teacher.py)',
          'mechanics': {'mouth_x': V5.MOUTH_X,
                        'rockfall_steps': V5.ROCKFALL_STEPS,
                        'wave_period': V5.WAVE_PERIOD,
                        'aim_x': list(V5.AIM_X),
                        't0_range': [V5.T0_MIN, V5.T0_MAX],
                        'blind_wait_until': V5.BLIND_WAIT_UNTIL},
          'decision_rule': {'mouth_to_band_min': CT.MOUTH_TO_BAND_MIN,
                            'cross_max': CT.CROSS_MAX,
                            'release_margin': CT.RELEASE_MARGIN,
                            'rule': 'wait iff active and start < t_mouth + '
                                    'cross_max and end > t_mouth + '
                                    'mouth_to_band_min; release at end + '
                                    'release_margin'},
          'learner_eval_protocol': 'reset() -- one canonical pose (native '
                                   'd4rl east); whether to stop at the '
                                   "mouth or take the detour is the policy's "
                                   'own action',
          'latent_visibility': 'rockfall_active / t0 NEVER in obs; sidecar '
                               'only',
          'latent_leak_check': leak_latent,
          'route_leak_check': leak_route,
          'reference_numbers_predicted': PREDICTED_REFS,
          'intent_counts': {'go': n_go, 'wait': n_wait, 'detour': n_detour},
          'route_counts': {'shortcut': int((routes == 'shortcut').sum()),
                           'detour': int((routes == 'detour').sum())},
          'latent_counts': {'clear': N - n_active, 'active': n_active},
          'n_deaths': n_deaths,
          'uncommitted_discards': {
              'n': len(discarded),
              'frac_of_draws': round(len(discarded) / (len(discarded) + N), 4),
              'rule': "shortcut: redrawn when info['entered_hazard'] is "
                      'False (the ant never reached the band); detour: '
                      "redrawn when info['route'] != 'detour' (never "
                      'reached the north column); committed timeouts are '
                      'KEPT',
              'episodes': discarded}}
  learner_path = os.path.join(out_dir, f'{name}.npz')
  np.savez_compressed(learner_path, obs=obs, act=act, lengths=lengths,
                      eval_goals=eval_goals, meta=json.dumps(meta))
  side_path = os.path.join(out_dir, f'{name}_sidecar.npz')

  def col(key, dtype=None, none=-1):
    v = [r[key] if r[key] is not None else none for r in rows]
    return np.array(v, dtype) if dtype else np.array(v)

  np.savez_compressed(
      side_path,
      episode_id=np.arange(N),
      rockfall_active=col('rockfall_active'),
      route=col('route'),
      route_realized=col('route_realized', none='none'),
      intent=col('intent'),
      success=col('success'), failure=col('failure'),
      entered_hazard=col('entered_hazard'),
      rock_dropped=col('rock_dropped'),
      rockfall_passed=col('rockfall_passed'),
      rock_waves=col('rock_waves', np.int64),
      rockfall_start=col('rockfall_start', np.int64),
      mouth_step=col('mouth_step', np.int64),
      band_entry_step=col('band_entry_step', np.int64),
      hold_steps=col('hold_steps', np.int64),
      release_step=col('release_step', np.int64),
      ep_return=col('return'), ep_length=col('ep_length', np.int64),
      step_torso_x=tx, step_torso_y=ty,
      collection_seed=np.int64(args.seed),
      p_far=np.float64(args.p_far))

  succ = float(np.mean([r['success'] for r in rows]))
  fail = float(np.mean([r['failure'] for r in rows]))
  holds = [r['hold_steps'] for r in rows if r['intent'] == 'wait']
  hold_txt = (f' | hold {np.mean(holds):.1f} [{min(holds)}, {max(holds)}]'
              if holds else '')
  print(f'\n{N} episodes | success {succ:.3f} | failure {fail:.3f} | '
        f'go {n_go} / wait {n_wait} / detour {n_detour}{hold_txt}')
  print(f'transitions {int(np.sum(lengths - 1))}')
  print('->', learner_path)
  print('->', side_path, flush=True)


if __name__ == '__main__':
  main()
