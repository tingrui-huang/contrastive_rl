#!/usr/bin/env python
"""AntMaze V6 hybrid control diagnostic (ett_futures_hybrid/, 2026-09-20; run from the repo root on the node holding the tables).  Are the hybrid paths the simulator paths?  Per anchor, compare the obs / act rows of the hybrid and the sealed simulator branch up
to the shorter length; where do they first differ; how the outcome pairs relate to the cut."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP
from exp_v6_learned_ett import REGIONS, region_of
H = MP.OUT
anchors, _ = MP.AnchorSet.load(H / 'anchors.npz'); reg = region_of(anchors.state[:, :2])
def load(p):
  with np.load(p, allow_pickle=False) as d:
    return {k: d[k] for k in ('obs_rows', 'act_rows', 'offset', 'length', 'outcome', 'steps', 'u1', 'u2')}
A = load(H / 'branches_cf.npz'); B = load(H / 'ett_futures_hybrid' / 'branches_ett.npz')
n = len(A['length']); first_diff = np.full(n, -1); maxdiff = np.zeros(n); same_len = np.zeros(n, bool)
for k in range(n):
  la, lb = A['length'][k], B['length'][k]; m = min(la, lb)
  oa = A['obs_rows'][A['offset'][k]:A['offset'][k] + m]; ob = B['obs_rows'][B['offset'][k]:B['offset'][k] + m]
  d = np.abs(oa - ob).max(axis=1); bad = np.nonzero(d > 1e-4)[0]
  first_diff[k] = int(bad[0]) if len(bad) else -1; maxdiff[k] = float(d.max()); same_len[k] = la == lb
oa_, ob_ = A['outcome'].astype(str), B['outcome'].astype(str)
pairs = {f'{a}->{b}': int(((oa_ == a) & (ob_ == b)).sum()) for a in ('success', 'death', 'timeout') for b in ('success', 'death', 'timeout')}
ident = first_diff < 0
res = {'n': n, 'identical_up_to_shorter': float(ident.mean()), 'identical_and_same_length': float((ident & same_len).mean()), 'identical_and_same_outcome': float((ident & (oa_ == ob_)).mean()),
       'first_diff_row_if_any': {'p10': float(np.percentile(first_diff[~ident], 10)) if (~ident).sum() else None, 'med': float(np.median(first_diff[~ident])) if (~ident).sum() else None, 'p90': float(np.percentile(first_diff[~ident], 90)) if (~ident).sum() else None, 'n': int((~ident).sum())},
       'maxdiff_med_nonidentical': float(np.median(maxdiff[~ident])) if (~ident).sum() else None,
       'outcome_pairs sim->hybrid': pairs, 'outcome_agreement': float((oa_ == ob_).mean()),
       'death_both: hybrid earlier / same / later': [int(((oa_ == 'death') & (ob_ == 'death') & (B['steps'] < A['steps'])).sum()), int(((oa_ == 'death') & (ob_ == 'death') & (B['steps'] == A['steps'])).sum()), int(((oa_ == 'death') & (ob_ == 'death') & (B['steps'] > A['steps'])).sum())],
       'death_both: hybrid - sim steps med': float(np.median((B['steps'] - A['steps'])[(oa_ == 'death') & (ob_ == 'death')])),
       'identical share | sim context inactive (u1=u2=0)': float(ident[~A['u1'] & ~A['u2']].mean()), 'identical share | sim context active': float(ident[A['u1'] | A['u2']].mean()),
       'identical share | sim inactive, long paths (>= 100 rows both)': float(ident[~A['u1'] & ~A['u2'] & (A['length'] >= 100) & (B['length'] >= 100)].mean()), 'n_long_inactive': int((~A['u1'] & ~A['u2'] & (A['length'] >= 100) & (B['length'] >= 100)).sum()),
       'n_nonidentical | sim inactive': int((~ident & ~A['u1'] & ~A['u2']).sum()), 'n_sim_inactive': int((~A['u1'] & ~A['u2']).sum()),
       'identical share | sim u1 only / u2 only / both': [float(ident[A['u1'] & ~A['u2']].mean()), float(ident[~A['u1'] & A['u2']].mean()), float(ident[A['u1'] & A['u2']].mean())],
       'first_diff row | sim active & nonidentical & both non-death (p10/med/p90)': [float(np.percentile(first_diff[~ident & (oa_ != 'death') & (ob_ != 'death')], q)) for q in (10, 50, 90)] if (~ident & (oa_ != 'death') & (ob_ != 'death')).sum() else None,
       'n nonidentical & both non-death': int((~ident & (oa_ != 'death') & (ob_ != 'death')).sum()),
       'by_region identical share': {REGIONS[i]: float(ident[reg == i].mean()) for i in range(len(REGIONS)) if (reg == i).sum() >= 100}}
MP.write_json(H / 'ett_futures_hybrid' / 'paths_vs_sim.json', res)
print(json.dumps(res, indent=1))
