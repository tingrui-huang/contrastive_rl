"""Per-actor coverage/diversity report from a replay snapshot (gate tool).

Episode provenance: insertion is block-major/actor-minor, so episode k
belongs to actor k % num_actors. Reports per-actor and pooled state coverage,
pairwise overlap (Jaccard over 2 m cells), start/goal diversity, moving
fraction, and torso health.
"""
import argparse
import json

import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--replay', required=True)
  ap.add_argument('--num_actors', type=int, default=4)
  ap.add_argument('--out', required=True)
  ap.add_argument('--tag', default='replay')
  args = ap.parse_args()
  d = np.load(args.replay)
  obs, n_eps, N = d['obs'], int(d['num_eps']), args.num_actors

  cells = [set() for _ in range(N)]
  starts, goals = [[] for _ in range(N)], [[] for _ in range(N)]
  moving, z_lo = [[] for _ in range(N)], [[] for _ in range(N)]
  for k in range(n_eps):
    a = k % N
    xy = obs[k, :, :2]
    cells[a].update({(round(float(x) / 2), round(float(y) / 2))
                     for x, y in xy})
    starts[a].append(tuple(np.round(xy[0], 1)))
    goals[a].append(tuple(np.round(obs[k, 0, 29:31], 1)))
    disp = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    moving[a].append(float(np.mean(disp > 5e-3)))
    z_lo[a].append(float(np.mean(obs[k, :, 2] < 0.3)))

  pooled = set().union(*cells)
  jac = [[round(len(cells[i] & cells[j]) / max(len(cells[i] | cells[j]), 1), 3)
          for j in range(N)] for i in range(N)]
  rep = {
      'tag': args.tag, 'episodes': n_eps, 'num_actors': N,
      'per_actor': [{
          'episodes': len(starts[a]),
          'unique_cells_2m': len(cells[a]),
          'unique_starts': len(set(starts[a])),
          'unique_goals': len(set(goals[a])),
          'moving_frac_mean': float(np.mean(moving[a])) if moving[a] else None,
          'fall_step_frac_mean': float(np.mean(z_lo[a])) if z_lo[a] else None,
      } for a in range(N)],
      'pooled_unique_cells_2m': len(pooled),
      'pairwise_cell_jaccard': jac,
      'all_actors_contribute': bool(all(len(c) > 0 for c in cells)
                                    and n_eps >= N),
  }
  json.dump(rep, open(args.out, 'w'), indent=2)
  print(json.dumps(rep, indent=1))


if __name__ == '__main__':
  main()
