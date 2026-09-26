#!/usr/bin/env python
"""AntMaze V6 hybrid control diagnostic (ett_futures_hybrid/, 2026-09-20; run from the repo root on the node holding the tables).  Hybrid vs simulator vs v4 branch tables: outcome rates conditional on the sampled context and the death-time distributions (no obs rows loaded)."""
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
reg = region_of(anchors.state[:, :2]); W = anchors.weight

def load(p):
  with np.load(p, allow_pickle=False) as d:
    return {k: d[k] for k in ('outcome', 'steps', 'length', 'u1', 'u2')}

T = {'sim': load(H / 'branches_cf.npz'), 'hybrid': load(H / 'ett_futures_hybrid' / 'branches_ett.npz'), 'v4': load(H / 'ett_futures_v4' / 'branches_ett.npz')}

def ks(a, b):
  if len(a) < 20 or len(b) < 20:
    return None
  x = np.sort(np.concatenate([a, b])); fa = np.searchsorted(np.sort(a), x, 'right') / len(a); fb = np.searchsorted(np.sort(b), x, 'right') / len(b)
  return float(np.abs(fa - fb).max())

out = {}
for i, r in enumerate(REGIONS):
  m = reg == i
  if m.sum() < 100:
    continue
  row = {'n': int(m.sum())}
  for name, t in T.items():
    oc = t['outcome'].astype(str); st = t['steps']
    row[name] = {'death': float((oc[m] == 'death').mean()), 'success': float((oc[m] == 'success').mean()), 'timeout': float((oc[m] == 'timeout').mean()),
                 'death|u1': float((oc[m & t['u1']] == 'death').mean()) if (m & t['u1']).sum() else None, 'death|~u1': float((oc[m & ~t['u1']] == 'death').mean()),
                 'death|u2': float((oc[m & t['u2']] == 'death').mean()), 'death|~u2': float((oc[m & ~t['u2']] == 'death').mean()),
                 'death_steps_med': float(np.median(st[m & (oc == 'death')])) if (m & (oc == 'death')).sum() else None,
                 'success_steps_med': float(np.median(st[m & (oc == 'success')])) if (m & (oc == 'success')).sum() else None}
  for name in ('hybrid', 'v4'):
    a, b = T['sim'], T[name]
    row[f'ks_death_steps_{name}'] = ks(a['steps'][m & (a['outcome'].astype(str) == 'death')], b['steps'][m & (b['outcome'].astype(str) == 'death')])
    row[f'ks_success_steps_{name}'] = ks(a['steps'][m & (a['outcome'].astype(str) == 'success')], b['steps'][m & (b['outcome'].astype(str) == 'success')])
    row[f'ks_length_{name}'] = ks(a['length'][m], b['length'][m])
  out[r] = row
# pooled, anchor-weighted, conditional on u1/u2
pooled = {}
for name, t in T.items():
  oc = t['outcome'].astype(str)
  pooled[name] = {c: float((W[k] * (oc[k] == 'death')).sum() / W[k].sum()) for c, k in (('death|u1', t['u1']), ('death|~u1', ~t['u1']), ('death|u2', t['u2']), ('death|~u2', ~t['u2']))}
  pooled[name]['death_steps_med'] = float(np.median(t['steps'][oc == 'death'])); pooled[name]['ks_death_steps_vs_sim'] = ks(T['sim']['steps'][T['sim']['outcome'].astype(str) == 'death'], t['steps'][oc == 'death'])
  pooled[name]['ks_length_vs_sim'] = ks(T['sim']['length'], t['length'])
MP.write_json(H / 'ett_futures_hybrid' / 'table_compare.json', {'pooled': pooled, 'by_region': out})
print(json.dumps(pooled, indent=1))
for r, row in out.items():
  print(f"{r:13s} n={row['n']:6d} death sim/hyb/v4 {row['sim']['death']:.3f}/{row['hybrid']['death']:.3f}/{row['v4']['death']:.3f}  death|u1 {row['sim']['death|u1']}/{row['hybrid']['death|u1']}/{row['v4']['death|u1']}  death|u2 {row['sim']['death|u2']:.3f}/{row['hybrid']['death|u2']:.3f}/{row['v4']['death|u2']:.3f}  dsteps med {row['sim']['death_steps_med']}/{row['hybrid']['death_steps_med']}/{row['v4']['death_steps_med']}  KS death hyb {row['ks_death_steps_hybrid']} v4 {row['ks_death_steps_v4']}  KS len hyb {row['ks_length_hybrid']} v4 {row['ks_length_v4']}")
