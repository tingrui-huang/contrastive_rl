"""Causal audit of the V4 rockfall-wait dataset: observational vs
interventional value of 'go' and 'wait', and the hiddenness of the latent.

Observational (sidecar of the collected dataset; the sighted teacher chose
the intent from the latent):
    P(wait observed)                 ~ P(active) = 0.30
    P(success | go observed)         ~ 1.0   (go was only ever taken when clear)
    P(success | wait observed)       ~ 1.0
    discounted (0.99**steps on success) by observed intent

Interventional (fresh env, natural Bernoulli(0.30) latent, n episodes each):
    do(go):    P(success) ~ P(clear) = 0.70, P(death) ~ 0.30
    do(wait):  P(success) ~ 1.0, ~84 steps slower
    sighted:   the oracle, same seeds (paired)

The gap P(success | go observed) - P(success | do(go)) = P(active) is the
confounding the u-blind learner inherits: the critic sees 'go' followed by
success in every go-episode, so it scores walking through the mouth at the
observational value. Under gamma=0.99 the always-go blind policy is ALSO the
best blind policy by discounted return (0.99**78 vs 0.99**162), so the
discounted objective and the confounded critic point the same way.

Hiddenness: paired rollouts (same env seed, same 'go' actions) under forced
u=active vs u=clear must be identical until the trigger step (rock teleports
cannot touch the ant before the rocks land); reports the divergence step vs
the trigger step across pairs.

Writes artifacts/rockfall_wait_v4/causal_audit.json (reference_numbers
block consumed by eval_rockfall_wait_v4_baseline.py).

Usage: python scripts/rockfall_wait_v4_causal_audit.py [--n 250] [--seed 303]
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

OUT = os.path.join(WT.OUT, 'causal_audit.json')
SIDECAR = os.path.join(WT.OUT, 'dataset',
                       'antmaze_rockfall_wait_v4_sidecar.npz')
GAMMA = WT.GAMMA


def _rate(xs):
  return round(float(np.mean(xs)), 4) if len(xs) else None


def _disc(rows):
  return _rate([GAMMA ** r['steps'] if r['success'] else 0.0 for r in rows])


def observational(path):
  s = np.load(path, allow_pickle=True)
  intent = np.asarray(s['intent']).astype(str)
  succ = np.asarray(s['success']).astype(bool)
  fail = np.asarray(s['failure']).astype(bool)
  u = np.asarray(s['rockfall_active']).astype(bool)
  L = np.asarray(s['ep_length']).astype(int)
  trig = np.asarray(s['trigger_step']).astype(int)
  entry = np.asarray(s['band_entry_step']).astype(int)
  hes = entry - trig
  disc = np.where(succ, GAMMA ** L, 0.0)
  out = {'n': int(len(intent)), 'P_wait_observed': _rate(intent == 'wait'),
         'P_active': _rate(u),
         'intent_equals_latent': _rate((intent == 'wait') == u)}
  for it in ('go', 'wait'):
    m = intent == it
    out[f'{it}_observed'] = {
        'n': int(m.sum()), 'success': _rate(succ[m]),
        'failure': _rate(fail[m]), 'discounted': _rate(disc[m]),
        'mean_steps': round(float(L[m].mean()), 1),
        'hesitation_mean': round(float(hes[m].mean()), 1),
        'hesitation_min_max': [int(hes[m].min()), int(hes[m].max())]}
  out['success_pooled'] = _rate(succ)
  out['discounted_pooled'] = _rate(disc)
  return out


def interventional(n, seed):
  cfg, teacher = WT.make_teacher()
  res = {}
  for name, intent in (('do_go', 'go'), ('do_wait', 'wait'),
                       ('sighted', None)):
    env = envs_mod.make_env(WT.ENV_NAME, cfg, seed=seed)
    u_rng = np.random.default_rng(seed + 5000)
    rows = []
    for k in range(n):
      u = bool(u_rng.random() < WT.P_ACTIVE)
      rows.append(WT.teacher_episode(env, teacher, u, intent=intent))
    act = [r for r in rows if r['rockfall_active']]
    clr = [r for r in rows if not r['rockfall_active']]
    res[name] = {
        'n': n, 'success': _rate([r['success'] for r in rows]),
        'failure': _rate([r['failure'] for r in rows]),
        'discounted': _disc(rows),
        'mean_steps_success': _rate([r['steps'] for r in rows
                                     if r['success']]),
        'by_latent': {
            'clear': {'n': len(clr),
                      'success': _rate([r['success'] for r in clr]),
                      'failure': _rate([r['failure'] for r in clr]),
                      'discounted': _disc(clr)},
            'active': {'n': len(act),
                       'success': _rate([r['success'] for r in act]),
                       'failure': _rate([r['failure'] for r in act]),
                       'discounted': _disc(act)}}}
    print(f'  {name}: success {res[name]["success"]} failure '
          f'{res[name]["failure"]} discounted {res[name]["discounted"]}',
          flush=True)
  return res


def hiddenness(n_pairs, seed):
  cfg, teacher = WT.make_teacher()
  rows = []
  for k in range(n_pairs):
    eA = envs_mod.make_env(WT.ENV_NAME, cfg, seed=seed + k)
    eC = envs_mod.make_env(WT.ENV_NAME, cfg, seed=seed + k)
    oA = eA.reset(rockfall_active=True)
    oC = eC.reset(rockfall_active=False)
    teacher.fresh()
    div, trig, dead = None, None, None
    d0 = float(np.max(np.abs(oA - oC)))
    for t in range(WT.HORIZON):
      a = teacher.act(oC, 'go')
      oA, rA, dA, iA = eA.step(a)
      oC, rC, dC, iC = eC.step(a)
      if trig is None and iA['trigger_step'] is not None:
        trig = int(iA['trigger_step'])
      if div is None and float(np.max(np.abs(oA - oC))) > 1e-6:
        div = t + 1
      if dA:
        dead = t + 1
        break
      if dC or rC > 0:
        break
    rows.append({'seed': seed + k, 'reset_diff': d0, 'trigger_step': trig,
                 'divergence_step': div, 'death_step': dead})
  ok = all(r['reset_diff'] == 0.0 and r['trigger_step'] is not None
           and (r['divergence_step'] is None
                or r['divergence_step'] > r['trigger_step']) for r in rows)
  return {'passes': bool(ok), 'n_pairs': n_pairs, 'pairs': rows,
          'min_gap_divergence_minus_trigger': min(
              r['divergence_step'] - r['trigger_step'] for r in rows
              if r['divergence_step'] is not None)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=250)
  ap.add_argument('--n-pairs', type=int, default=10)
  ap.add_argument('--seed', type=int, default=303)
  ap.add_argument('--sidecar', default=SIDECAR)
  args = ap.parse_args()
  print('observational (sidecar)', flush=True)
  obs = observational(args.sidecar)
  print(json.dumps(obs, indent=2), flush=True)
  print('interventional', flush=True)
  itv = interventional(args.n, args.seed)
  print('hiddenness', flush=True)
  hid = hiddenness(args.n_pairs, args.seed + 700)
  print(json.dumps(hid, indent=2), flush=True)
  refs = {
      'sparse_success': {'always_go': itv['do_go']['success'],
                         'always_wait': itv['do_wait']['success'],
                         'oracle': itv['sighted']['success']},
      'discounted_gamma_0.99': {
          'always_go': itv['do_go']['discounted'],
          'always_wait': itv['do_wait']['discounted'],
          'oracle': itv['sighted']['discounted'],
          'best_blind': ('always_go' if itv['do_go']['discounted']
                         >= itv['do_wait']['discounted'] else 'always_wait')},
      'discounted_definition': '0.99**steps on success, else 0.0; mean over '
                               'episodes',
      'confounding_gap': {
          'P_success_go_observed': obs['go_observed']['success'],
          'P_success_do_go': itv['do_go']['success'],
          'gap': round(obs['go_observed']['success']
                       - itv['do_go']['success'], 4),
          'discounted_go_observed': obs['go_observed']['discounted'],
          'discounted_do_go': itv['do_go']['discounted']}}
  print(json.dumps(refs, indent=2), flush=True)
  os.makedirs(WT.OUT, exist_ok=True)
  with open(OUT, 'w') as f:
    json.dump({'env': WT.ENV_NAME, 'seed': args.seed, 'n': args.n,
               'mechanics': {'mouth_x': V4.MOUTH_X,
                             'rockfall_steps': V4.ROCKFALL_STEPS,
                             'wave_period': V4.WAVE_PERIOD,
                             'teacher_hold_steps': WT.HOLD_STEPS},
               'observational': obs, 'interventional': itv,
               'hiddenness': hid, 'reference_numbers': refs}, f, indent=2)
  print('->', OUT, flush=True)
  sys.exit(0 if hid['passes'] else 1)


if __name__ == '__main__':
  main()
