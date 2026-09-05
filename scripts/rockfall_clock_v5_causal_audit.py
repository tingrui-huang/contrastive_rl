"""Causal audit of the V5 rockfall-clock datasets: observational vs
interventional value of 'go', 'wait' and 'detour', and the hiddenness of
the latent.

Observational (sidecar of a collected dataset; the sighted teacher chose
'go' / 'wait' from the timetable, the caller's coin chose the detour):
    P(wait observed)                 ~ P(active) * (1 - p_far) = 0.30 / 0.285
    P(detour observed)               ~ p_far (0 in the plain set, 0.05 far05)
    P(success | go observed)         ~ 1.0   (go was only ever taken when the
                                              burst missed the crossing, i.e.
                                              when clear in the natural range)
    P(success | wait observed)       ~ 1.0
    P(success | detour observed)     ~ 1.0
    discounted (0.99**steps on success) by observed intent

Interventional (fresh envs, NATURAL latent Bernoulli(0.30) drawn by the
env, NATURAL t0 ~ U{0..30} drawn by the env, the same env seed in every
arm so the arms are paired episode for episode):
    do(go):      P(success) ~ P(clear) = 0.70, P(death) ~ 0.30 -- the burst
                 [t0, t0 + 72] with t0 <= 30 always overlaps a straight
                 crossing
    do(wait):    the BLIND always-wait reference, hold at the mouth until
                 env step BLIND_WAIT_UNTIL = 102: P(success) ~ 1.0, slow
    do(detour):  the V3-br detour, never crosses the band: P(success) ~ 1.0,
                 ~225 steps
    sighted:     the oracle (always the shortcut; holds only when the
                 timetable overlaps its crossing), same seeds

The gap P(success | go observed) - P(success | do(go)) = P(active) is the
confounding the u-blind learner inherits: the critic sees 'go' followed by
success in every go-episode, so it scores walking through the mouth at the
observational value. ``best_blind`` is the argmax of the three blind
discounted values; under gamma=0.99 the blind wait costs ~0.99**102 and the
detour ~0.99**225 relative to go's ~0.99**78, so always-go is expected to
remain the best BLIND policy by discounted return -- the discounted
objective and the confounded critic point the same way.

Hiddenness: paired rollouts (same env seed, same 'go' actions computed on
the clear twin) under forced u=active (natural t0) vs u=clear. The rocks
drop at t0 <= 30 while the ant is still west of the mouth, so the twins must
stay identical until a dropped rock is WITHIN REACH of the ant
(``env.rock_within_reach``, see the V5 module doc: a rock that hits the ant
and bounces off inside a step never shows up in the flagged contact, so the
flagged contact / death step is too late a reference). Reports, per pair,
the first |obs diff| > 1e-6 step, the first exact-0 divergence step, the
first within-reach step, the first flagged contact / death step and the
band entry step; passes iff every divergence is at or after the first
within-reach step.

Writes artifacts/rockfall_clock_v5/causal_audit.json (reference_numbers
block consumed by eval_rockfall_clock_v5_baseline.py).

Usage: python scripts/rockfall_clock_v5_causal_audit.py --dataset both
           [--n 250] [--n-pairs 40] [--seed 303]
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

OUT = os.path.join(CT.OUT, 'causal_audit.json')
SIDECAR_DIR = os.path.join(CT.OUT, 'dataset')
#: dataset variants -> learner npz stem (the sidecar is '<stem>_sidecar.npz')
DATASETS = {'near': 'antmaze_rockfall_clock_v5',
            'far05': 'antmaze_rockfall_clock_v5_far05'}
INTENTS = ('go', 'wait', 'detour')
#: the blind arms competing for ``best_blind``
BLIND_ARMS = ('always_go', 'always_wait', 'always_detour')
GAMMA = CT.GAMMA
#: hiddenness tolerance on the 58-dim float32 obs (the V4 smoke convention;
#: the exact-0 step is reported alongside, see the V5 module doc).
DIV_TOL = 1e-6


def _rate(xs):
  return round(float(np.mean(xs)), 4) if len(xs) else None


def _mean(xs, nd=1):
  return round(float(np.mean(xs)), nd) if len(xs) else None


def _disc(rows):
  #: 0.99**steps on success, else 0.0, mean over ALL episodes (the V4
  #: causal-audit convention; the teacher audit averages over successes).
  return _rate([GAMMA ** r['steps'] if r['success'] else 0.0 for r in rows])


def sidecar_path(name, sidecar_dir=SIDECAR_DIR):
  return os.path.join(sidecar_dir, f'{DATASETS[name]}_sidecar.npz')


def observational(path):
  """Per-dataset block from the sidecar. ``-1`` encodes None in the step
  columns (V4 collector convention); hesitation = band_entry_step -
  mouth_step over the shortcut rows that have both."""
  s = np.load(path, allow_pickle=True)
  intent = np.asarray(s['intent']).astype(str)
  route = np.asarray(s['route']).astype(str)
  succ = np.asarray(s['success']).astype(bool)
  fail = np.asarray(s['failure']).astype(bool)
  u = np.asarray(s['rockfall_active']).astype(bool)
  L = np.asarray(s['ep_length']).astype(int)
  t0 = np.asarray(s['rockfall_start']).astype(int)
  mouth = np.asarray(s['mouth_step']).astype(int)
  entry = np.asarray(s['band_entry_step']).astype(int)
  hold = np.asarray(s['hold_steps']).astype(int)
  entered = np.asarray(s['entered_hazard']).astype(bool)
  disc = np.where(succ, GAMMA ** L, 0.0)
  has_hes = (mouth >= 0) & (entry >= 0)
  hes = entry - mouth
  sc = route == 'shortcut'
  out = {'n': int(len(intent)), 'P_active': _rate(u),
         'P_go_observed': _rate(intent == 'go'),
         'P_wait_observed': _rate(intent == 'wait'),
         'P_detour_observed': _rate(intent == 'detour'),
         'P_wait_given_active_shortcut': _rate(intent[sc & u] == 'wait'),
         'P_wait_given_clear_shortcut': _rate(intent[sc & ~u] == 'wait'),
         'intent_equals_latent_shortcut': _rate(
             (intent[sc] == 'wait') == u[sc]),
         't0_active_min_max': ([int(t0[u].min()), int(t0[u].max())]
                               if u.any() else None),
         'detour_entered_band': _rate(entered[route == 'detour'])}
  for it in INTENTS:
    m = intent == it
    mh = m & has_hes
    out[f'{it}_observed'] = {
        'n': int(m.sum()), 'success': _rate(succ[m]),
        'failure': _rate(fail[m]), 'discounted': _rate(disc[m]),
        'mean_steps': _mean(L[m]),
        'n_active': int((m & u).sum()),
        'success_by_latent': {'clear': _rate(succ[m & ~u]),
                              'active': _rate(succ[m & u])},
        'mean_hold_steps': _mean(hold[m]),
        'hesitation_mean': _mean(hes[mh]),
        'hesitation_min_max': ([int(hes[mh].min()), int(hes[mh].max())]
                               if mh.any() else None)}
  out['success_pooled'] = _rate(succ)
  out['failure_pooled'] = _rate(fail)
  out['discounted_pooled'] = _rate(disc)
  out['success_by_latent'] = {'clear': _rate(succ[~u]),
                              'active': _rate(succ[u])}
  return out


def _arm_block(rows):
  act = [r for r in rows if r['rockfall_active']]
  clr = [r for r in rows if not r['rockfall_active']]

  def by(rs):
    return {'n': len(rs), 'success': _rate([r['success'] for r in rs]),
            'failure': _rate([r['failure'] for r in rs]),
            'discounted': _disc(rs),
            'mean_steps_success': _mean([r['steps'] for r in rs
                                         if r['success']])}
  return {'n': len(rows), 'success': _rate([r['success'] for r in rows]),
          'failure': _rate([r['failure'] for r in rows]),
          'timeout': _rate([not r['success'] and not r['failure']
                            for r in rows]),
          'discounted': _disc(rows),
          'mean_steps_success': _mean([r['steps'] for r in rows
                                       if r['success']]),
          'entered_band': _rate([r['entered_hazard'] for r in rows]),
          'mean_hold_steps': _mean([r['hold_steps'] for r in rows]),
          'P_wait_decided_given_active': _rate(
              [r['intent'] == 'wait' for r in act]),
          'by_latent': {'clear': by(clr), 'active': by(act)}}


def interventional(n, seed):
  """do(go), blind do(wait), do(detour) and the sighted expert on fresh
  envs with the same seed: the env draws the latent AND t0, so the four
  arms see the same (latent, t0) sequence (checked and reported)."""
  cfg, teacher = CT.make_teacher()
  res, draws = {}, {}
  for name, intent in (('do_go', 'go'), ('do_wait', 'wait'),
                       ('do_detour', 'detour'), ('sighted', None)):
    env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed)
    rows = []
    for k in range(n):
      rows.append(CT.teacher_episode(env, teacher, None, intent=intent))
      if (k + 1) % 50 == 0:
        print(f'    {name} {k + 1}/{n}', flush=True)
    draws[name] = [(r['rockfall_active'], r['rockfall_start']) for r in rows]
    res[name] = _arm_block(rows)
    print(f'  {name}: success {res[name]["success"]} failure '
          f'{res[name]["failure"]} discounted {res[name]["discounted"]} '
          f'mean steps (success) {res[name]["mean_steps_success"]}',
          flush=True)
  paired = all(draws[k] == draws['do_go'] for k in draws)
  res['paired_latent_and_t0'] = bool(paired)
  res['n_active'] = int(sum(a for a, _ in draws['do_go']))
  return res


def hiddenness(n_pairs, seed):
  """Active (natural t0) vs clear twin, same 'go' actions; see module doc."""
  cfg, teacher = CT.make_teacher()
  rows = []
  for k in range(n_pairs):
    eA = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed + k)
    eC = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed + k)
    oA = eA.reset(rockfall_active=True)
    oC = eC.reset(rockfall_active=False)
    teacher.fresh()
    t0 = int(eA.privileged_rockfall_start)
    div, div0, reach, contact, dead, entry = None, None, None, None, None, None
    d0 = float(np.max(np.abs(oA - oC)))
    for t in range(CT.HORIZON):
      #: actions from the CLEAR twin (the blind 'go' policy); the schedule
      #: is not consulted under intent 'go'.
      a = teacher.act(oC, eC.schedule, 'go')
      oA, rA, dA, iA = eA.step(a)
      oC, rC, dC, iC = eC.step(a)
      diff = float(np.max(np.abs(oA - oC)))
      if div0 is None and diff > 0.0:
        div0 = t + 1
      if div is None and diff > DIV_TOL:
        div = t + 1
      if reach is None and eA.rock_within_reach:
        reach = t + 1
      if contact is None and iA.get('rock_contact'):
        contact = t + 1
      if entry is None and iA['band_entry_step'] is not None:
        entry = int(iA['band_entry_step'])
      if dA:
        dead = t + 1
        break
      if dC or rC > 0:
        break
    rows.append({'seed': seed + k, 'reset_diff': d0, 'rockfall_start': t0,
                 'rockfall_end': t0 + V5.ROCKFALL_STEPS,
                 'divergence_step': div, 'divergence_step_exact0': div0,
                 'within_reach_step': reach, 'contact_step': contact,
                 'death_step': dead, 'band_entry_step': entry,
                 'active_success': bool(iA.get('success')),
                 'clear_success': bool(iC.get('success'))})
    print(f'  pair {k}: t0 {t0} band entry {entry} within reach {reach} '
          f'|diff|>{DIV_TOL:g} at {div} (exact-0 at {div0}) contact {contact} '
          f'death {dead}', flush=True)

  def gap(r, key):
    return (None if r['divergence_step'] is None or r[key] is None
            else r['divergence_step'] - r[key])
  #: a pair passes iff the twins were identical (to DIV_TOL) at every step
  #: before the first within-reach step; a pair that never diverged passes.
  ok = all(r['reset_diff'] == 0.0
           and (r['divergence_step'] is None
                or (r['within_reach_step'] is not None
                    and r['divergence_step'] >= r['within_reach_step']))
           for r in rows)
  gaps_reach = [gap(r, 'within_reach_step') for r in rows
                if gap(r, 'within_reach_step') is not None]
  gaps_entry = [gap(r, 'band_entry_step') for r in rows
                if gap(r, 'band_entry_step') is not None]
  return {'passes': bool(ok), 'n_pairs': n_pairs, 'tolerance': DIV_TOL,
          'n_diverged': int(sum(r['divergence_step'] is not None
                                for r in rows)),
          'n_died_active': int(sum(r['death_step'] is not None
                                   for r in rows)),
          'min_gap_divergence_minus_within_reach': (
              int(min(gaps_reach)) if gaps_reach else None),
          'min_gap_divergence_minus_band_entry': (
              int(min(gaps_entry)) if gaps_entry else None),
          'mean_gap_divergence_minus_band_entry': _mean(gaps_entry),
          'pairs': rows}


def reference_numbers(itv, obs_by_dataset):
  disc = {'always_go': itv['do_go']['discounted'],
          'always_wait': itv['do_wait']['discounted'],
          'always_detour': itv['do_detour']['discounted'],
          'oracle': itv['sighted']['discounted']}
  best_blind = max(BLIND_ARMS, key=lambda k: disc[k])
  go_obs = {name: o['go_observed']['success']
            for name, o in obs_by_dataset.items()}
  go_obs_disc = {name: o['go_observed']['discounted']
                 for name, o in obs_by_dataset.items()}
  return {
      'sparse_success': {'always_go': itv['do_go']['success'],
                         'always_wait': itv['do_wait']['success'],
                         'always_detour': itv['do_detour']['success'],
                         'oracle': itv['sighted']['success']},
      'discounted_gamma_0.99': {**disc, 'best_blind': best_blind},
      'discounted_definition': '0.99**steps on success, else 0.0; mean over '
                               'episodes',
      'P_success_go_observed': go_obs,
      'confounding_gap': {
          'P_success_go_observed': go_obs,
          'P_success_do_go': itv['do_go']['success'],
          'gap': {name: (round(v - itv['do_go']['success'], 4)
                         if v is not None else None)
                  for name, v in go_obs.items()},
          'discounted_go_observed': go_obs_disc,
          'discounted_do_go': itv['do_go']['discounted']}}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', choices=('near', 'far05', 'both'),
                  default='both')
  ap.add_argument('--n', type=int, default=250)
  ap.add_argument('--n-pairs', type=int, default=40)
  ap.add_argument('--seed', type=int, default=303)
  ap.add_argument('--sidecar-dir', default=SIDECAR_DIR)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  names = ('near', 'far05') if args.dataset == 'both' else (args.dataset,)
  obs = {}
  for name in names:
    path = sidecar_path(name, args.sidecar_dir)
    if not os.path.exists(path):
      raise FileNotFoundError(f'{name} sidecar missing: {path} (run '
                              'collect_rockfall_clock_v5_dataset.py first)')
    print(f'observational ({name}: {path})', flush=True)
    obs[name] = observational(path)
    print(json.dumps(obs[name], indent=2), flush=True)
  print('interventional', flush=True)
  itv = interventional(args.n, args.seed)
  print(f'  paired latent+t0 across arms: {itv["paired_latent_and_t0"]}',
        flush=True)
  print('hiddenness', flush=True)
  hid = hiddenness(args.n_pairs, args.seed + 700)
  print(json.dumps({k: v for k, v in hid.items() if k != 'pairs'}, indent=2),
        flush=True)
  refs = reference_numbers(itv, obs)
  print(json.dumps(refs, indent=2), flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump({'env': CT.ENV_NAME, 'seed': args.seed, 'n': args.n,
               'datasets': {name: sidecar_path(name, args.sidecar_dir)
                            for name in names},
               'mechanics': {'mouth_x': V5.MOUTH_X,
                             'rockfall_steps': V5.ROCKFALL_STEPS,
                             'wave_period': V5.WAVE_PERIOD,
                             't0_range': [V5.T0_MIN, V5.T0_MAX],
                             'blind_wait_until': V5.BLIND_WAIT_UNTIL,
                             'teacher_mouth_to_band_min': CT.MOUTH_TO_BAND_MIN,
                             'teacher_cross_max': CT.CROSS_MAX,
                             'teacher_release_margin': CT.RELEASE_MARGIN},
               'observational': obs, 'interventional': itv,
               'hiddenness': hid, 'reference_numbers': refs}, f, indent=2)
  print('->', args.out, flush=True)
  sys.exit(0 if hid['passes'] else 1)


if __name__ == '__main__':
  main()
