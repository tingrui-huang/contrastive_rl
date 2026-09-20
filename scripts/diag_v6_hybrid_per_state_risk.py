#!/usr/bin/env python
"""AntMaze V6 hybrid control diagnostic (ett_futures_hybrid/, 2026-09-20; run from the repo root on the node holding the tables).  Conditional risk at the state level: anchors grouped by the xy cell (1 x 1) and the time bin (40 steps) -- similar states -- give per-group branch death /
success rates in each table (fresh contexts per branch).  Calibration of the hybrid's (and v4's) group rate against the simulator's."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP
from exp_v6_learned_ett import REGIONS, region_of
H = MP.OUT
anchors, _ = MP.AnchorSet.load(H / 'anchors.npz')
reg = region_of(anchors.state[:, :2])
xy = np.floor(anchors.state[:, :2]).astype(np.int64); gid = (xy[:, 0] + 50) * 100000 + (xy[:, 1] + 50) * 1000 + anchors.t.astype(np.int64) // 40
ug, inv = np.unique(gid, return_inverse=True); cnt = np.bincount(inv)
def load(p):
  with np.load(p, allow_pickle=False) as d:
    return d['outcome'].astype(str)
T = {n: load(p) for n, p in (('sim', H / 'branches_cf.npz'), ('hybrid', H / 'ett_futures_hybrid' / 'branches_ett.npz'), ('v4', H / 'ett_futures_v4' / 'branches_ett.npz'))}
res = {}
ok = cnt >= 8
greg = np.zeros(len(ug), np.int64); greg[inv] = reg
for kind in ('death', 'success'):
  R = {n: np.bincount(inv, weights=(oc == kind).astype(float)) / cnt for n, oc in T.items()}
  row = {'n_groups': int(ok.sum())}
  for n in ('hybrid', 'v4'):
    a, b = R['sim'][ok], R[n][ok]
    bins = np.digitize(a, [0.2, 0.4, 0.6, 0.8])
    row[n] = {'corr': float(np.corrcoef(a, b)[0, 1]), 'mean_abs_diff': float(np.abs(a - b).mean()), 'rmse': float(np.sqrt(((a - b) ** 2).mean())),
              'calibration (sim-rate bin -> mean sim, mean model, n)': {f'bin{i}': (float(a[bins == i].mean()), float(b[bins == i].mean()), int((bins == i).sum())) for i in range(5) if (bins == i).sum() >= 20},
              'by_region': {REGIONS[i]: {'corr': float(np.corrcoef(a[greg[ok] == i], b[greg[ok] == i])[0, 1]), 'mean_sim': float(a[greg[ok] == i].mean()), 'mean_model': float(b[greg[ok] == i].mean()), 'n': int((greg[ok] == i).sum())} for i in range(len(REGIONS)) if (greg[ok] == i).sum() >= 50}}
  # the sampling floor: two independent simulator draws of the same group would also disagree; estimate by splitting each group's anchors in halves
  half = (np.arange(len(gid)) % 2 == 0)
  a1 = np.bincount(inv, weights=((T['sim'] == kind) & half).astype(float)) / np.maximum(np.bincount(inv, weights=half.astype(float)), 1)
  a2 = np.bincount(inv, weights=((T['sim'] == kind) & ~half).astype(float)) / np.maximum(np.bincount(inv, weights=(~half).astype(float)), 1)
  row['sim_split_half_corr'] = float(np.corrcoef(a1[ok], a2[ok])[0, 1]); row['sim_split_half_mean_abs_diff'] = float(np.abs(a1[ok] - a2[ok]).mean())
  h1 = np.bincount(inv, weights=((T['hybrid'] == kind) & half).astype(float)) / np.maximum(np.bincount(inv, weights=half.astype(float)), 1)
  row['sim_half_vs_hybrid_other_half_corr'] = float(np.corrcoef(a1[ok], np.bincount(inv, weights=((T['hybrid'] == kind) & ~half).astype(float))[ok] / np.maximum(np.bincount(inv, weights=(~half).astype(float)), 1)[ok])[0, 1])
  res[kind] = row
MP.write_json(H / 'ett_futures_hybrid' / 'per_state_risk.json', res)
print(json.dumps(res, indent=1))
