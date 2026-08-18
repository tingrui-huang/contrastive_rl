"""Phase 1 audit + STOP gate for the fixed bad-demonstrator dataset.

Reports the pre-registered audit quantities and computes

    r_fatal = N_fatal_transitions / N_bad_demo_transitions

where a FATAL transition is the single (s_c, a_c) -> s'_settled step of a
settled rock-death episode (the episode is truncated at collapse+2, so each
dead episode contributes exactly one).

STOP GATE: if fewer than 10 natural settled-fatal transitions occur in the
pre-registered dataset the script reports and exits non-zero rather than
silently collecting more data.

Tier-1 bad outcomes are audited SEMANTICALLY only -- they are never used as
labels, never filter the dataset, and never reach the Flow trainer:
  * settled rock death (sidecar dead=True);
  * fallen / non-recoverable pose  (torso z < 0.35 sustained to episode end);
  * physically stuck                (XY displacement < 0.25 m over the last
                                     100 steps of a non-dead, non-success
                                     episode).
Ordinary wall contact, mild contact, inefficiency, wrong route and recoverable
mistakes are explicitly NOT Tier-1.

Usage:
  python scripts/audit_bad_demo.py
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402

DIR = 'artifacts/bad_demo_fixed'
NAME = 'bad_demo_blind_p30_h800_settle80'
Z_FALLEN = 0.35
STUCK_WINDOW, STUCK_DX = 100, 0.25
MIN_FATAL_TRANSITIONS = 10


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default=DIR)
  ap.add_argument('--name', default=NAME)
  args = ap.parse_args()
  npz = os.path.join(args.dir, args.name + '.npz')
  side = os.path.join(args.dir, args.name + '_sidecar.npz')
  d = np.load(npz, allow_pickle=True)
  s = np.load(side, allow_pickle=True)
  man = json.load(open(os.path.join(args.dir, 'collection_manifest.json')))

  obs, act, L = d['obs'], d['act'], np.asarray(d['lengths'], np.int64)
  n_eps = obs.shape[0]
  n_trans = int((L - 1).sum())
  dead = np.asarray(s['dead'], bool)
  succ = np.asarray(s['success'], float)
  route = np.asarray(s['route'])
  tx, ty = s['step_torso_x'], s['step_torso_y']

  # hazard exposure: entered >=1 site trigger region
  exposed = np.zeros(n_eps, bool)
  for e in range(n_eps):
    x, y = tx[e, :L[e] - 1], ty[e, :L[e] - 1]
    for i, (_, sx, sgn) in enumerate(RA.ROCKFALL_SITES):
      if np.any((np.abs(x - sx) <= RA.TRIG_HALF_X)
                & (RA.TRIG_Y_BAND[0] <= sgn * y)
                & (sgn * y <= RA.TRIG_Y_BAND[1])):
        exposed[e] = True
        break

  # Tier-1 bad outcomes (semantic audit only)
  n_fatal_eps = int(dead.sum())
  n_fatal_trans = n_fatal_eps          # one settled-fatal step per dead ep
  fallen = np.zeros(n_eps, bool)
  stuck = np.zeros(n_eps, bool)
  for e in range(n_eps):
    if dead[e]:
      continue
    z = obs[e, :L[e], 2]
    fallen[e] = bool(z[-1] < Z_FALLEN and z[-min(20, len(z)):].max()
                     < Z_FALLEN)
    if succ[e] == 0 and L[e] > STUCK_WINDOW + 1:
      xy = obs[e, :L[e], :2]
      stuck[e] = bool(np.linalg.norm(xy[-1] - xy[-STUCK_WINDOW]) < STUCK_DX)

  r_fatal = n_fatal_trans / n_trans
  batch, steps = 1024, 20_000
  audit = {
      'dataset': {'npz': npz, 'npz_sha256': C.sha256_file(npz),
                  'sidecar_sha256': C.sha256_file(side),
                  'controller': man['controller'],
                  'protocol': man['protocol'],
                  'death_settle_substeps': man['death_settle_substeps'],
                  'env_seed': man['env_seed'],
                  'dataset_seed': man['dataset_seed'],
                  'git_commit': man['git_commit']},
      'episodes': int(n_eps), 'transitions': n_trans,
      'success_rate': float(succ.mean()),
      'hazard_exposure_rate': float(exposed.mean()),
      'settled_fatal_episodes': n_fatal_eps,
      'settled_fatal_transitions': n_fatal_trans,
      'r_fatal': r_fatal,
      'tier1_bad_semantic_audit': {
          'settled_rock_death_episodes': n_fatal_eps,
          'fallen_non_recoverable_episodes': int(fallen.sum()),
          'stuck_episodes': int(stuck.sum()),
          'note': ('semantic audit only -- never used as labels, never '
                   'filters the dataset, never reaches the Flow trainer. '
                   'Wall contact / mild contact / inefficiency / wrong route '
                   'are explicitly NOT Tier-1.')},
      'route_distribution': {k: int(v) for k, v in
                             Counter(route.tolist()).items()},
      'base_side_distribution': {k: int(v) for k, v in
                                 Counter(np.asarray(s['base_side']).tolist()
                                         ).items()},
      'episode_length': {'mean': float(L.mean()), 'min': int(L.min()),
                         'max': int(L.max())},
      'observation_ranges': {
          'state_min': float(np.min([obs[e, :L[e], :29].min()
                                     for e in range(n_eps)])),
          'state_max': float(np.max([obs[e, :L[e], :29].max()
                                     for e in range(n_eps)])),
          'torso_z_min': float(np.min([obs[e, :L[e], 2].min()
                                       for e in range(n_eps)])),
          'torso_z_max': float(np.max([obs[e, :L[e], 2].max()
                                       for e in range(n_eps)]))},
      'action_ranges': {'min': float(np.min([act[e, :L[e] - 1].min()
                                             for e in range(n_eps)])),
                        'max': float(np.max([act[e, :L[e] - 1].max()
                                             for e in range(n_eps)]))},
      'implied_fatal_exposure_in_v1_training': {
          'note': ('expected settled-fatal samples seen over the 20k-step '
                   'budget at batch 1024 under source-mixture beta'),
          **{'beta_%.2f' % b: {
              'expected_fatal_per_batch': b * r_fatal * batch,
              'expected_fatal_samples_total': b * r_fatal * batch * steps,
              'expected_repeats_per_distinct_fatal_transition':
                  (b * r_fatal * batch * steps / n_fatal_trans
                   if n_fatal_trans else None)}
             for b in (0.05, 0.10, 0.15, 0.20)}},
      'stop_gate': {
          'min_required_fatal_transitions': MIN_FATAL_TRANSITIONS,
          'observed': n_fatal_trans,
          'passed': n_fatal_trans >= MIN_FATAL_TRANSITIONS},
  }
  json.dump(audit, open(os.path.join(args.dir, 'audit.json'), 'w'), indent=2)

  print('episodes %d | transitions %d | success %.3f | hazard exposure %.3f'
        % (n_eps, n_trans, audit['success_rate'],
           audit['hazard_exposure_rate']))
  print('settled-fatal: %d episodes = %d transitions | r_fatal = %.6f '
        '(1 in %.0f)' % (n_fatal_eps, n_fatal_trans, r_fatal,
                         1 / r_fatal if r_fatal else float('inf')))
  print('tier-1 other: fallen %d | stuck %d'
        % (audit['tier1_bad_semantic_audit']['fallen_non_recoverable_episodes'],
           audit['tier1_bad_semantic_audit']['stuck_episodes']))
  print('routes %s | lengths mean %.0f'
        % (audit['route_distribution'], audit['episode_length']['mean']))
  print('obs range [%.2f, %.2f] | action range [%.2f, %.2f]'
        % (audit['observation_ranges']['state_min'],
           audit['observation_ranges']['state_max'],
           audit['action_ranges']['min'], audit['action_ranges']['max']))
  print('\nimplied fatal exposure during V1 training (batch 1024, 20k steps):')
  for b in (0.05, 0.10, 0.15, 0.20):
    q = audit['implied_fatal_exposure_in_v1_training']['beta_%.2f' % b]
    print('  beta %.2f: %.3f fatal/batch, %.0f fatal samples total, '
          '~%.0f repeats per distinct fatal transition'
          % (b, q['expected_fatal_per_batch'],
             q['expected_fatal_samples_total'],
             q['expected_repeats_per_distinct_fatal_transition']))
  ok = audit['stop_gate']['passed']
  print('\nSTOP GATE: %s (%d >= %d settled-fatal transitions)'
        % ('PASS' if ok else 'FAIL -- STOP AND REPORT', n_fatal_trans,
           MIN_FATAL_TRANSITIONS))
  print('audit -> %s' % os.path.join(args.dir, 'audit.json'))
  return 0 if ok else 3


if __name__ == '__main__':
  sys.exit(main())
