#!/usr/bin/env python
"""AntMaze V6 hybrid control diagnostic (ett_futures_hybrid/, 2026-09-20; run from the repo root on the node holding the tables).  Where do the non-identical pairs differ?  For the simulator-death anchors whose hybrid path is at least as long: is the first
differing row the simulator's death frame, and how does that frame differ from the physics-only state (xy / z / velocities)?"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP
H = MP.OUT
def load(p):
  with np.load(p, allow_pickle=False) as d:
    return {k: d[k] for k in ('obs_rows', 'offset', 'length', 'outcome', 'steps', 'u1', 'u2')}
A = load(H / 'branches_cf.npz'); B = load(H / 'ett_futures_hybrid' / 'branches_ett.npz')
oa, ob = A['outcome'].astype(str), B['outcome'].astype(str)
sel = np.nonzero((oa == 'death') & (B['length'] >= A['length']))[0]
at_death = 0; before = 0; none = 0; dxy = []; dz = []; dv = []; dpre = []
for k in sel:
  la = A['length'][k]; ra = A['obs_rows'][A['offset'][k]:A['offset'][k] + la]; rb = B['obs_rows'][B['offset'][k]:B['offset'][k] + la]
  d = np.abs(ra - rb).max(axis=1); bad = np.nonzero(d > 1e-4)[0]
  if len(bad) == 0:
    none += 1; continue
  if bad[0] == la - 1:
    at_death += 1; dxy.append(np.linalg.norm(ra[-1, :2] - rb[-1, :2])); dz.append(ra[-1, 2] - rb[-1, 2]); dv.append(np.abs(ra[-1, 15:29] - rb[-1, 15:29]).max())
  else:
    before += 1; dpre.append(int(la - 1 - bad[0]))
res = {'n_sim_death_with_hybrid_at_least_as_long': int(len(sel)), 'identical_incl_death_frame': none, 'first_diff_at_death_frame': at_death, 'first_diff_before_death_frame': before,
       'rows_before_death_when_earlier (p10/med/p90)': [float(np.percentile(dpre, q)) for q in (10, 50, 90)] if dpre else None,
       'death_frame_vs_physics: xy shift med/p90': [float(np.median(dxy)), float(np.percentile(dxy, 90))], 'z diff med/p10/p90': [float(np.median(dz)), float(np.percentile(dz, 10)), float(np.percentile(dz, 90))], 'max |dvel| med/p90': [float(np.median(dv)), float(np.percentile(dv, 90))]}
MP.write_json(H / 'ett_futures_hybrid' / 'death_frame_check.json', res)
print(json.dumps(res, indent=1))
