"""Paired episode-bootstrap comparison of the B-layer agreement between runs
(r1, R, COV) of the torque-coverage experiment: the same decided pairs, the
same fresh outcomes, each run's three critics scored with the primary
readout (region-integrated, radius 0.5, deployed min), agreement averaged
over the three seeds per run, and the DIFFERENCE between runs resampled
over held-out episodes (the pairs of one episode move together).  Also the
per-seed differences.  Implements the manifest's pre-registered line:
COV's pooled dense B agreement above r1 and above R by > 2 s.e. (paired
bootstrap) and above 0.5 by 2 s.e., with a positive pick gain.

usage: V6_... probe_v6_coverage_paired_boot.py --cont-ckpt <BC> --runs r1=<dir> R=<dir> COV=<dir> [--marginals r1=<replay> ...]
"""
import argparse
import json
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402
import diag_v6_r1_abc as ABC  # noqa: E402
from crl import checkpoint  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--cont-ckpt', required=True)
  ap.add_argument('--runs', nargs='+', required=True, help='name=critic dir (seed_0..2/final.pkl inside)')
  ap.add_argument('--marginals', nargs='*', default=[], help='name=replay (under V6_BRANCH_OUT) for the goal marginal; weighted if audit_weight')
  ap.add_argument('--layer', default='B')
  ap.add_argument('--out', default='diag_r1_abc/paired_boot_B.json')
  args = ap.parse_args()
  out = B.OUT / 'diag_r1_abc'
  man, keys, info = ABC.load_keys(out)
  bundle = R.policy_bundle(args.cont_ckpt); nets = bundle['nets']
  marg = dict(m.split('=') for m in args.marginals)
  ks = [k for k in keys if k[0] == args.layer]
  o = np.stack([keys[k]['obs0'] for k in ks]); a = np.stack([keys[k]['act0'] for k in ks])
  # decided pairs (scorer-independent)
  rng = np.random.default_rng(11)
  anchors = {}
  for k in ks:
    anchors.setdefault(k[1], []).append(k[2])
  pairs = []
  for aid, cands in anchors.items():
    for i in range(len(cands)):
      for j in range(i + 1, len(cands)):
        pi, pj = keys[(args.layer, aid, cands[i])]['p'], keys[(args.layer, aid, cands[j])]['p']
        dd = sorted(set(pi) & set(pj))
        cls, d, _ = ABC.classify_pair(pi, pj, dd, rng)
        if cls == 'decided':
          pairs.append((aid, cands[i], cands[j], np.sign(d), keys[(args.layer, aid, cands[i])]['episode']))
  print(f'{len(pairs)} decided pairs in layer {args.layer}', flush=True)
  correct = {}
  for spec in args.runs:
    name, d = spec.split('=')
    frames = R.marginal_goal_frames(B.OUT / marg[name], per_path=4, seed=0, weighted=True) if name in marg else None
    for s in (0, 1, 2):
      st = checkpoint.load_checkpoint(B.OUT / d / f'seed_{s}' / 'final.pkl')[1]
      sc = R.region_scorers(nets, st.q_params, frames, ABC.RADIUS, max_goals=512)(o, a)['min'] if frames is not None else R.exact_scorers(nets, st.q_params)(o, a)['min']
      idx = {k: i for i, k in enumerate(ks)}
      correct[(name, s)] = np.array([float(np.sign(sc[idx[(args.layer, aid, ci)]] - sc[idx[(args.layer, aid, cj)]]) == sg) for (aid, ci, cj, sg, ep) in pairs])
  eps = np.array([p[4] for p in pairs]); ueps = np.unique(eps)
  runs = [r.split('=')[0] for r in args.runs]
  run_mean = {r: np.mean([correct[(r, s)] for s in (0, 1, 2)], axis=0) for r in runs}
  brng = np.random.default_rng(5)
  boots = {r: [] for r in runs}; boots_seed = {(r, s): [] for r in runs for s in (0, 1, 2)}
  for _ in range(2000):
    pick = brng.choice(ueps, size=len(ueps), replace=True)
    m = np.concatenate([np.flatnonzero(eps == e) for e in pick])
    for r in runs:
      boots[r].append(run_mean[r][m].mean())
      for s in (0, 1, 2):
        boots_seed[(r, s)].append(correct[(r, s)][m].mean())
  res = {'layer': args.layer, 'n_decided_pairs': len(pairs), 'n_episodes': int(len(ueps)), 'runs': {}}
  for r in runs:
    b = np.array(boots[r])
    res['runs'][r] = {'agreement_seed_mean': float(run_mean[r].mean()), 'se': float(b.std()),
                      'per_seed': {s: {'agreement': float(correct[(r, s)].mean()), 'se': float(np.std(boots_seed[(r, s)]))} for s in (0, 1, 2)}}
  res['differences'] = {}
  for r1_, r2_ in [(x, y) for x in runs for y in runs if x != y]:
    d = np.array(boots[r1_]) - np.array(boots[r2_])
    res['differences'][f'{r1_} - {r2_}'] = {'mean': float(run_mean[r1_].mean() - run_mean[r2_].mean()), 'se': float(d.std()),
                                          'z': float((run_mean[r1_].mean() - run_mean[r2_].mean()) / max(d.std(), 1e-9)),
                                          'per_seed': {s: {'diff': float(correct[(r1_, s)].mean() - correct[(r2_, s)].mean()),
                                                           'se': float(np.std(np.array(boots_seed[(r1_, s)]) - np.array(boots_seed[(r2_, s)])))} for s in (0, 1, 2)}}
  (B.OUT / args.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
  print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
  main()
