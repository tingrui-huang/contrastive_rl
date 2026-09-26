#!/usr/bin/env python
"""AntMaze V6 hybrid control diagnostic (ett_futures_hybrid/, 2026-09-20; run from the repo root on the node holding the tables).  The actual NCE positive (future-goal) distribution under each branch table: anchor by weight, m ~ P(m) prop. to 0.999^m over
1..L-1, goal row = root + m.  Tabulated by the goal row's region / terminal kind / last-row xy of death branches.  No training."""
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
reg_anchor = region_of(anchors.state[:, :2]); W = anchors.weight
rng = np.random.default_rng(0); N = 2_000_000; gamma = 0.999
cdf = np.cumsum(W / W.sum()); cdf[-1] = 1.0
K = np.searchsorted(cdf, rng.random(N))

def dist(p):
  with np.load(p, allow_pickle=False) as d:
    off, L, oc = d['offset'], d['length'], d['outcome'].astype(str); xy = d['obs_rows'][:, :2]
  n = L[K] - 1; u = rng.random(N)
  m = np.clip(np.ceil(np.log1p(-u * (1.0 - gamma ** n)) / np.log(gamma)).astype(np.int64), 1, n)
  g = off[K] + m; gxy = xy[g]; last = (m == n); kind = np.where(last, oc[K], 'mid')
  rg = region_of(gxy.astype(np.float64))
  out = {'goal_kind': {k: float((kind == k).mean()) for k in ('mid', 'success', 'death', 'timeout')},
         'goal_region': {REGIONS[i]: float((rg == i).mean()) for i in range(len(REGIONS))},
         'death_goal_region': {REGIONS[i]: float((rg[kind == 'death'] == i).mean()) for i in range(len(REGIONS))},
         'm_med': float(np.median(m)), 'm_mean': float(m.mean()),
         'goal_region_by_anchor_region': {REGIONS[i]: {REGIONS[j]: float((rg[reg_anchor[K] == i] == j).mean()) for j in range(len(REGIONS)) if (rg[reg_anchor[K] == i] == j).mean() >= 0.01} for i in range(len(REGIONS)) if (reg_anchor[K] == i).sum() >= 1000},
         'death_last_xy_by_region': {}}
  # the death frame's xy (the last row of a death branch) by the anchor's region
  for i, r in enumerate(REGIONS):
    sel = (oc == 'death') & (reg_anchor == i)
    if sel.sum() >= 100:
      lx = xy[off[sel] + L[sel] - 1]
      out['death_last_xy_by_region'][r] = {'n': int(sel.sum()), 'x_med': float(np.median(lx[:, 0])), 'y_med': float(np.median(lx[:, 1])), 'x_p10': float(np.percentile(lx[:, 0], 10)), 'x_p90': float(np.percentile(lx[:, 0], 90)),
                                          'in_band': float(np.mean([bool(v) for v in __import__('fit_v6_ett_one_step_v2').in_band(lx.astype(np.float64))]))}
  return out

res = {name: dist(p) for name, p in (('sim', H / 'branches_cf.npz'), ('hybrid', H / 'ett_futures_hybrid' / 'branches_ett.npz'), ('v4', H / 'ett_futures_v4' / 'branches_ett.npz'))}
MP.write_json(H / 'ett_futures_hybrid' / 'nce_future_dist.json', res)
for k in ('goal_kind', 'goal_region', 'death_goal_region'):
  print(k); [print(f"  {n:7s}", {kk: round(v, 3) for kk, v in r[k].items()}) for n, r in res.items()]
print('m', {n: (r['m_med'], round(r['m_mean'], 1)) for n, r in res.items()})
print('death last xy by anchor region'); [print(f"  {n:7s}", {kk: (v['n'], v['x_med'], v['y_med'], round(v['in_band'], 3)) for kk, v in r['death_last_xy_by_region'].items()}) for n, r in res.items()]
print('goal region by anchor region (sim / hybrid / v4)')
for ar in res['sim']['goal_region_by_anchor_region']:
  print(' ', ar, {g: (res['sim']['goal_region_by_anchor_region'][ar].get(g), res['hybrid']['goal_region_by_anchor_region'][ar].get(g), res['v4']['goal_region_by_anchor_region'][ar].get(g)) for g in set(res['sim']['goal_region_by_anchor_region'][ar]) | set(res['hybrid']['goal_region_by_anchor_region'][ar])})
