"""Read-only profile of a branch file by the anchor's logged route and region.

For every branch (one per anchor) of `branches_cf.npz` (round 1, start-agent
continuation) or `round2/branches_cf_r2_s<k>.npz` (round 2, the lineage agent
continues) this groups the anchors by the route the LOGGED episode realised
(sidecar `route_realized`: shortcut / detour) and by the region of the anchor
state, and reports the anchor-weighted outcome shares, the share of branches
that reach the top corridor (y >= 5.5: the branch itself went around), the
share that reach the hazard corridor and the mean length.  Numpy only; no
model is loaded.

  python scripts/diag_v6_round2_branch_profile.py --branches <npz> [--branches <npz> ...] --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as M  # noqa: E402
from diag_v6_pilot_trajectories import region  # noqa: E402


def profile(branch_path, anchors, route):
  with np.load(branch_path, allow_pickle=False) as d:
    obs, offset, length = d['obs_rows'], d['offset'], d['length']
    outcome, steps, aid, t = d['outcome'].astype(str), d['steps'], d['anchor_id'], d['t']
    meta = json.loads(str(d['meta']))
  assert np.all(aid == np.arange(len(aid))), 'branches are not in anchor order'
  xy = obs[:, :2]
  reach_top = np.zeros(len(aid), bool); reach_hazard = np.zeros(len(aid), bool); max_y = np.zeros(len(aid), np.float32)
  for k in range(len(aid)):
    seg = xy[offset[k]:offset[k] + length[k]]
    max_y[k] = seg[:, 1].max() if len(seg) else np.nan
    reach_top[k] = bool((seg[:, 1] >= 5.5).any())
    reach_hazard[k] = bool(((seg[:, 0] >= 8.0) & (seg[:, 0] <= 16.0) & (np.abs(seg[:, 1]) < 2.5)).any())   # middle third of the shortcut corridor
  w = anchors.weight
  reg = np.array([region(float(p[0]), float(p[1])) for p in anchors.state[:, :2]])
  logged_route = route[anchors.episode]
  out = {'meta': {k: meta[k] for k in ('continuation_ckpt', 'continuation_ckpt_sha256', 'anchors', 'wall_seconds')}, 'groups': {}}

  def block(m):
    ww = w[m]; W = ww.sum()
    return {'n': int(m.sum()), 'weight': float(W),
            'success': float((ww * (outcome[m] == 'success')).sum() / W), 'death': float((ww * (outcome[m] == 'death')).sum() / W),
            'timeout': float((ww * (outcome[m] == 'timeout')).sum() / W),
            'reach_top_corridor': float((ww * reach_top[m]).sum() / W), 'reach_hazard_corridor': float((ww * reach_hazard[m]).sum() / W),
            'mean_steps': float((ww * steps[m]).sum() / W)}

  for r in ('shortcut', 'detour'):
    for g in ('start', 'shortcut_corridor', 'west_column', 'top_west_corner', 'top_corridor', 'east_column', 'goal_area'):
      m = (logged_route == r) & (reg == g)
      if m.sum() >= 5:
        out['groups'][f'logged {r} | anchor in {g}'] = block(m)
    m = (logged_route == r) & (t == 0)
    if m.any():
      out['groups'][f'logged {r} | reset row (t = 0)'] = block(m)
    m = (logged_route == r) & (t >= 1) & (t < 30)
    if m.any():
      out['groups'][f'logged {r} | t in [1, 30)'] = block(m)
    out['groups'][f'logged {r} | all'] = block(logged_route == r)
  out['groups']['all'] = block(np.ones(len(aid), bool))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--branches', action='append', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  anchors, _ = M.AnchorSet.load(M.OUT / 'anchors.npz')
  with np.load(M.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  res = {}
  for b in args.branches:
    res[Path(b).name] = profile(Path(b), anchors, route)
    print(f'== {b}', flush=True)
    for g, v in res[Path(b).name]['groups'].items():
      print(f'  {g:48s} n {v["n"]:6d} w {v["weight"]:.3f}  succ {v["success"]:.3f} death {v["death"]:.3f} timeout {v["timeout"]:.3f}  '
            f'top {v["reach_top_corridor"]:.3f} hazard {v["reach_hazard_corridor"]:.3f}  steps {v["mean_steps"]:.0f}', flush=True)
  M.write_json(Path(args.out), res)


if __name__ == '__main__':
  main()
