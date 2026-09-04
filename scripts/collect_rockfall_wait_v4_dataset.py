"""Offline dataset collector for the V4 rockfall-wait AntMaze benchmark.

Port of scripts/collect_tworoute_v3_dataset.py onto crl/rockfall_wait_v4.py:
one route (BR shortcut, goal (8,0)), a timed rockfall triggered at the band
mouth, and a sighted teacher (scripts/rockfall_wait_v4_teacher.py) that
walks straight through when the latent is clear and HOLDS at the mouth
(zero torque, ~84 steps) until the rockfall has passed when it is active.

  learner npz: obs [N, L+1, 58] float32, act [N, L+1, 8] float32 (last row
               dummy zeros), lengths [N] (valid obs rows), eval_goals [N, 2],
               meta (json string). NOTHING privileged.
  sidecar npz: per-episode audit fields (rockfall_active, intent, outcome,
               trigger / band-entry steps, hold steps, ...) + step torso
               traces. NEVER a training input; consumed only by the audit.

The latent is drawn by this collector's own recorded rng (Bernoulli
p_active) and passed to reset(). Every episode starts from ONE canonical
pose, so the t=0 observation carries no intent information -- checked, not
asserted, by the permutation leak test (verbatim from V2/V3).

What the u-blind learner sees: every kept episode takes the same corridor;
at the mouth, ~70% of them keep walking and ~30% stand still for ~84 steps
and then walk. The waiting episodes reach the goal too, ~85 steps later.

Pre-registered predictions (gamma=0.99, 0.99**steps on success else 0),
derived from the V3-br driver numbers (go 0.70 sparse / 0.323 discounted at
~77 steps) and the 84-step hold: always-go 0.70 / 0.323; always-wait ~0.99
/ 0.70*0.99**77 + 0.30*0.99**161 ~ 0.198 + ...; oracle ~0.99 sparse /
~0.382 discounted. The teacher audit + causal audit measure the real ones.

Usage: python scripts/collect_rockfall_wait_v4_dataset.py [--episodes 400]
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
from crl import rockfall_wait_v4 as V4        # noqa: E402
import rockfall_wait_v4_teacher as WT         # noqa: E402

OUT_ROOT = WT.OUT
NAME = 'antmaze_rockfall_wait_v4'
HORIZON = WT.HORIZON

#: analytic predictions (see module doc); the audits report measured values.
PREDICTED_REFS = {
    'sparse_success': {'always_go': 0.70, 'always_wait': 0.99,
                       'oracle': 0.99},
    'discounted_gamma_0.99': {'always_go': 0.323, 'always_wait': 0.198,
                              'oracle': 0.382, 'best_blind': 'always_go'},
    'discounted_definition': '0.99**steps on success, else 0.0; mean over '
                             'episodes',
    'derivation': 'V3-br go driver: 0.70 sparse, ~77 steps; wait adds '
                  f'{WT.HOLD_STEPS} hold steps: 0.99**161 ~ 0.198',
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--episodes', type=int, default=400)
  ap.add_argument('--seed', type=int, default=606)
  ap.add_argument('--p-active', type=float, default=WT.P_ACTIVE)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=None)
  args = ap.parse_args()
  out_dir = args.out_dir or os.path.join(OUT_ROOT, 'dataset')
  os.makedirs(out_dir, exist_ok=True)

  cfg, teacher = WT.make_teacher()
  cfg.rockfall_max_steps = int(args.horizon)
  env = envs_mod.make_env(WT.ENV_NAME, cfg, seed=args.seed)
  u_rng = np.random.default_rng(args.seed + 5000)

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
    o = env.reset(rockfall_active=u)
    teacher.fresh()
    obs[e, 0] = o
    tx[e, 0], ty[e, 0] = o[0], o[1]
    eval_goals[e] = o[29:31]
    ret, info = 0.0, {}
    for t in range(args.horizon):
      #: the expert is blind before the mouth; it reads the latent there
      a = teacher.act(o, revealed=env.revealed_rockfall_active)
      o, r, done, info = env.step(a)
      act[e, t] = a
      obs[e, t + 1] = o
      tx[e, t + 1], ty[e, t + 1] = o[0], o[1]
      ret += float(r)
      if done or r > 0:
        break
    #: what the expert did, for the sidecar: the mouth decision, or 'go'
    #: if it never reached the mouth (a redraw below)
    intent = teacher.decision or 'go'
    if info.get('trigger_step') is not None:
      assert intent == ('wait' if u else 'go'), (e, u, intent)
    #: REDRAW an episode that never entered the band (V3 rule, one route):
    #: the ant shuffled in the start cell for the whole horizon, which is a
    #: demonstrator failure, not a demonstration. Band-entered timeouts are
    #: kept.
    if not info.get('entered_hazard'):
      discarded.append({'rockfall_active': u, 'intent': intent,
                        'final_xy': [round(float(tx[e, t + 1]), 3),
                                     round(float(ty[e, t + 1]), 3)],
                        'ep_length': int(t + 1)})
      obs[e, :] = 0.0
      act[e, :] = 0.0
      tx[e, :] = np.nan
      ty[e, :] = np.nan
      continue
    lengths[e] = t + 2                     # valid obs rows (0 .. t+1)
    rows.append({'episode_id': e, 'rockfall_active': u, 'intent': intent,
                 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'rock_dropped': bool(info.get('rock_dropped')),
                 'rockfall_passed': bool(info.get('rockfall_passed')),
                 'rock_waves': int(info.get('rock_waves', 0)),
                 'trigger_step': info.get('trigger_step'),
                 'band_entry_step': info.get('band_entry_step'),
                 'hold_steps': int(teacher.hold_steps_done),
                 'return': ret, 'ep_length': int(t + 1)})
    e += 1
    if e % 50 == 0:
      print(f'  {e}/{N} episodes ({len(discarded)} discarded)', flush=True)
  print(f'discarded {len(discarded)} uncommitted episodes '
        f'({len(discarded) / (len(discarded) + N):.3f} of draws)', flush=True)

  #: MEASURED latent-leak check on the t=0 observation, permutation null
  #: (verbatim from the V2/V3 collectors; groups = intent).
  def _dprime(g0, g1):
    pooled = np.sqrt((g0.var(0, ddof=1) + g1.var(0, ddof=1)) / 2.0) + 1e-9
    return np.abs(g0.mean(0) - g1.mean(0)) / pooled

  intents = np.array([r['intent'] for r in rows])
  o0, m = obs[:, 0, :], intents == 'go'
  if m.sum() > 1 and (~m).sum() > 1:
    d = _dprime(o0[m], o0[~m])
    rng = np.random.default_rng(0)
    k, B = int(m.sum()), 2000
    null = np.empty(B)
    for b in range(B):
      p = rng.permutation(len(o0))
      null[b] = _dprime(o0[p][:k], o0[p][k:]).max()
    p95 = float(np.percentile(null, 95))
    leak = {'test': 'permutation null on per-dim d-prime at t=0, B=2000',
            'n_go': int(m.sum()), 'n_wait': int((~m).sum()),
            'max_dprime': round(float(d.max()), 4),
            'argmax_dim': int(d.argmax()),
            'null_p95': round(p95, 4),
            'n_dims_above_null_p95': int((d > p95).sum()),
            'p_value': round(float((null >= d.max()).mean()), 4),
            'passes': bool(d.max() <= p95)}
  else:
    leak = {'passes': None, 'reason': 'an intent group is empty'}
  print(f"latent-leak check: max d'={leak.get('max_dprime')} (dim "
        f"{leak.get('argmax_dim')}) vs chance p95 {leak.get('null_p95')}, "
        f"p={leak.get('p_value')} -> "
        f"{'PASS' if leak.get('passes') else 'FAIL'}", flush=True)

  n_wait = sum(r['intent'] == 'wait' for r in rows)
  meta = {'name': NAME, 'env': WT.ENV_NAME,
          'obs_dim': 29, 'goal_dim': 29, 'action_dim': 8,
          'horizon': args.horizon, 'p_active': args.p_active,
          'collection_seed': args.seed,
          'teacher': 'sighted rockfall_wait_v4_teacher (clear->go, '
                     'active->wait at the mouth then go); one route from '
                     'the single canonical pose -- the hold is the only '
                     'latent-dependent behaviour (see scripts/'
                     'rockfall_wait_v4_teacher.py)',
          'mechanics': {'mouth_x': V4.MOUTH_X,
                        'rockfall_steps': V4.ROCKFALL_STEPS,
                        'wave_period': V4.WAVE_PERIOD,
                        'aim_x': list(V4.AIM_X),
                        'teacher_hold_steps': WT.HOLD_STEPS},
          'learner_eval_protocol': 'reset() -- one canonical pose (native '
                                   'd4rl east); whether to stop at the '
                                   "mouth is the policy's own action",
          'latent_visibility': 'rockfall_active NEVER in obs; sidecar only',
          'latent_leak_check': leak,
          'reference_numbers_predicted': PREDICTED_REFS,
          'intent_counts': {'go': N - n_wait, 'wait': n_wait},
          'uncommitted_discards': {
              'n': len(discarded),
              'frac_of_draws': round(len(discarded) / (len(discarded) + N), 4),
              'rule': "redrawn when info['entered_hazard'] is False, i.e. "
                      'the ant never reached the band (a failed start); '
                      'band-entered timeouts are KEPT',
              'episodes': discarded}}
  learner_path = os.path.join(out_dir, f'{NAME}.npz')
  np.savez_compressed(learner_path, obs=obs, act=act, lengths=lengths,
                      eval_goals=eval_goals, meta=json.dumps(meta))
  side_path = os.path.join(out_dir, f'{NAME}_sidecar.npz')

  def col(key, dtype=None, none=-1):
    v = [r[key] if r[key] is not None else none for r in rows]
    return np.array(v, dtype) if dtype else np.array(v)

  np.savez_compressed(
      side_path,
      episode_id=np.arange(N),
      rockfall_active=col('rockfall_active'),
      intent=col('intent'),
      success=col('success'), failure=col('failure'),
      entered_hazard=col('entered_hazard'),
      rock_dropped=col('rock_dropped'),
      rockfall_passed=col('rockfall_passed'),
      rock_waves=col('rock_waves', np.int64),
      trigger_step=col('trigger_step', np.int64),
      band_entry_step=col('band_entry_step', np.int64),
      hold_steps=col('hold_steps', np.int64),
      ep_return=col('return'), ep_length=col('ep_length', np.int64),
      step_torso_x=tx, step_torso_y=ty,
      collection_seed=np.int64(args.seed))

  succ = float(np.mean([r['success'] for r in rows]))
  fail = float(np.mean([r['failure'] for r in rows]))
  print(f'\n{N} episodes | success {succ:.3f} | failure {fail:.3f} | '
        f'wait {n_wait}/{N}')
  print(f'transitions {int(np.sum(lengths - 1))}')
  print('->', learner_path)
  print('->', side_path, flush=True)


if __name__ == '__main__':
  main()
