"""Steps 1+2 of the continuous-Manski port: propensity + neighborhood gates.

Fits the binned propensity P_hat(bin | cell) on the frozen swamp dataset
(episode-level holdout), sweeps the sector count, builds the BFS V_lb proxy
and the pessimistic worst-neighbor table, and reports four gates:

  G1_propensity_beats_uniform   held-out log-likelihood beats the uniform-
                                over-bins baseline by a clear margin
  G2_decision_cells_stochastic  the HOLDING cell's propensity entropy exceeds
                                EVERY downstream shortcut-corridor cell's
                                (behavior is stochastic at the u-driven
                                wait-vs-go decision, then commits). The
                                comparison set is the same corridor, NOT all
                                cells: the lower route's entropy is dominated
                                by the random-episode background, which is
                                expected and not a decision signal.
  G3_bfs_map_valid              goal at distance 0, every passable cell
                                reachable
  G4_holding_teleport_backward  the HOLDING cell's pessimistic teleport
                                target strictly increases BFS distance

Writes  <out>/propensity_table.npz   (counts, probs, config -- the table the
                                      Thm-2 sampler will load)
        <out>/propensity_report.json (gates, sweep, entropy/dist maps)

The dataset's audit-only fields (swamp_bits, route_label, ...) are NEVER
read here: the propensity must come from (obs, act) alone.

Run:
  python -m scripts.fit_propensity
"""
import argparse
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl import manski
from crl.config import Config


def _grid_str(walls, values, marks, fmt='{:5.2f}'):
  """ASCII grid (x = rows, y = cols); walls '  ###', unvisited '    -'."""
  lines = ['      ' + ''.join(f'  y={j} ' for j in range(walls.shape[1]))]
  for i in range(walls.shape[0]):
    row = []
    for j in range(walls.shape[1]):
      if walls[i, j] == 1:
        cell = '  ###'
      else:
        v = values[i, j]
        cell = '    -' if not np.isfinite(v) else fmt.format(v)
      row.append(cell + (marks.get((i, j), ' ')))
    lines.append(f'  x={i} ' + ''.join(row))
  return '\n'.join(lines)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='datasets/swamp_matched_teacher_s0.npz')
  ap.add_argument('--env_name', default='point_two_route_swamp_matched_v0')
  ap.add_argument('--sectors', default='4,8,16',
                  help='comma list swept for held-out log-likelihood')
  ap.add_argument('--pick', type=int, default=8,
                  help='sector count used for the saved table')
  ap.add_argument('--zero_thresh', type=float, default=0.15)
  ap.add_argument('--holdout', type=float, default=0.2)
  ap.add_argument('--alpha', type=float, default=1.0)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='artifacts/manski_port')
  args = ap.parse_args()

  cfg = Config(env_name=args.env_name)
  env = envs_mod.make_env(args.env_name, cfg, seed=args.seed)
  walls = env._walls
  goal_cell = manski.cells_of(walls, np.asarray(env.GOAL))
  holding, fork = tuple(env.HOLDING_CELL), tuple(env.FORK_CELL)

  data = np.load(args.dataset, allow_pickle=True)
  obs, act = data['obs'], data['act']
  ne, length = obs.shape[0], obs.shape[1]
  # act[:, -1] is a dummy zero row (collect layout) -> use t = 0..L-2
  xy = obs[:, :length - 1, :2].reshape(-1, 2)
  a = act[:, :length - 1].reshape(-1, 2)
  cells = manski.cells_of(walls, xy)

  rng = np.random.default_rng(args.seed)
  perm = rng.permutation(ne)
  n_hold = max(1, int(round(ne * args.holdout)))
  hold_eps = np.zeros(ne, bool)
  hold_eps[perm[:n_hold]] = True
  hold_mask = np.repeat(hold_eps, length - 1)

  # ---- Step 1: sector sweep on episode-level holdout ---------------------- #
  # `taken_phat` = mean P_hat(bin of the LOGGED action | cell) on holdout:
  # the per-step walk-survival probability of the Thm-2 sampler. Over a
  # Geom(1-gamma) walk pessimism compounds as (1-taken_phat) per step, so
  # this -- not the log-likelihood -- is the quantity that decides whether
  # the lower bound is informative or vacuous.
  sweep = []
  for k in (int(s) for s in args.sectors.split(',')):
    bins = manski.action_bins(a, k, args.zero_thresh)
    _, probs = manski.fit_propensity(
        walls, cells[~hold_mask], bins[~hold_mask], k, args.alpha)
    ll = manski.mean_loglik(probs, cells[hold_mask], bins[hold_mask])
    uniform = -float(np.log(manski.n_bins(k)))
    hp = probs[cells[hold_mask][:, 0], cells[hold_mask][:, 1],
               bins[hold_mask]]
    taken = float(np.mean(hp))
    sweep.append(dict(sectors=k, holdout_loglik=ll, uniform_baseline=uniform,
                      margin=ll - uniform, taken_phat=taken))
    print(f'sectors={k:3d}  held-out LL {ll:+.4f}  uniform {uniform:+.4f}  '
          f'margin {ll - uniform:+.4f}  taken-phat {taken:.3f}')

  pick = args.pick
  bins = manski.action_bins(a, pick, args.zero_thresh)
  counts, probs = manski.fit_propensity(walls, cells, bins, pick, args.alpha)
  ent = manski.entropy_map(counts, args.alpha)

  # ---- Step 2: BFS V_lb proxy + pessimistic teleport table ---------------- #
  dist = manski.bfs_dist_map(walls, goal_cell)
  worst = manski.worst_neighbor_map(walls, dist)

  # per-cell mean taken-phat map (full data, smoothed table)
  taken_all = probs[cells[:, 0], cells[:, 1], bins]
  tp_sum = np.zeros(walls.shape)
  tp_cnt = np.zeros(walls.shape)
  np.add.at(tp_sum, (cells[:, 0], cells[:, 1]), taken_all)
  np.add.at(tp_cnt, (cells[:, 0], cells[:, 1]), 1.0)
  tp_map = np.where(tp_cnt > 0, tp_sum / np.maximum(tp_cnt, 1), np.nan)

  marks = {holding: 'H', fork: 'F', tuple(goal_cell): 'G'}
  print('\npropensity entropy (nats), sectors=%d  [H=holding F=fork G=goal]'
        % pick)
  print(_grid_str(walls, ent, marks))
  print('\nmean taken-action P_hat per cell (walk survival per step)')
  print(_grid_str(walls, tp_map, marks))
  print('\nBFS distance to goal (V_lb proxy: larger = worse)')
  print(_grid_str(walls, np.where(dist < 0, np.nan, dist).astype(float),
                  marks, fmt='{:5.0f}'))
  print('\npessimistic teleports (worst N-member changes cell):')
  for c, w in sorted(worst.items()):
    if w != c:
      print(f'  {c} -> {w}   (dist {dist[c]} -> {dist[w]})')

  # ---- gates --------------------------------------------------------------- #
  picked = next(s for s in sweep if s['sectors'] == pick)
  g1 = picked['margin'] > 0.2
  # downstream shortcut corridor: same column as holding, holding.x+1 .. goal-1
  downstream = [(i, holding[1]) for i in range(holding[0] + 1, int(goal_cell[0]))
                if walls[i, holding[1]] == 0]
  g2 = bool(all(ent[holding] > ent[c] for c in downstream))
  g3 = bool(dist[tuple(goal_cell)] == 0
            and all(dist[c] >= 0 for c in map(tuple, np.argwhere(walls == 0))))
  g4 = bool(dist[worst[holding]] > dist[holding])
  # G1b: binning-artifact check. Where behavior COMMITS (the corridor's end,
  # visited almost purely by teacher flow) the taken-phat must be high; a
  # sector edge slicing the behavioral mode would cap it near 0.5. Low
  # survival in EARLY corridor cells is not an artifact -- it is the real
  # teacher+random mixture there -- so it is reported, not gated.
  down_tp = {str(c): float(tp_map[c]) for c in downstream}
  g1b = bool(max(tp_map[c] for c in downstream) > 0.75)
  gates = [
      dict(name='G1b_corridor_walk_survival', passed=g1b,
           metrics=dict(downstream_taken_phat=down_tp, threshold='> 0.6'),),
      dict(name='G1_propensity_beats_uniform', passed=bool(g1),
           metrics=picked, threshold='margin > 0.2 nats'),
      dict(name='G2_decision_cells_stochastic', passed=g2,
           metrics=dict(holding_entropy=float(ent[holding]),
                        downstream_corridor={str(c): float(ent[c])
                                             for c in downstream},
                        fork_entropy_info=float(ent[fork]))),
      dict(name='G3_bfs_map_valid', passed=g3,
           metrics=dict(goal_dist=int(dist[tuple(goal_cell)]),
                        max_dist=int(dist.max()))),
      dict(name='G4_holding_teleport_backward', passed=g4,
           metrics=dict(holding=list(holding), target=list(worst[holding]),
                        dist_from=int(dist[holding]),
                        dist_to=int(dist[worst[holding]]))),
  ]
  print()
  for g in gates:
    print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']}  {g['metrics']}")

  os.makedirs(args.out, exist_ok=True)
  np.savez(os.path.join(args.out, 'propensity_table.npz'),
           counts=counts, probs=probs,
           config=json.dumps(dict(
               dataset=args.dataset, env_name=args.env_name, sectors=pick,
               zero_thresh=args.zero_thresh, alpha=args.alpha,
               seed=args.seed, n_transitions=int(len(a)))))
  report = dict(gates=gates, sweep=sweep,
                all_passed=bool(all(g['passed'] for g in gates)),
                entropy_map=np.where(np.isfinite(ent), ent, -1).tolist(),
                bfs_dist_map=dist.tolist(),
                worst_neighbors={str(k): list(v) for k, v in worst.items()})
  with open(os.path.join(args.out, 'propensity_report.json'), 'w') as f:
    json.dump(report, f, indent=1)
  print(f"\n{'ALL GATES PASS' if report['all_passed'] else 'GATES FAILED'}"
        f' -> {args.out}')


if __name__ == '__main__':
  main()
