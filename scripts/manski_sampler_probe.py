"""Step 3 gates for the continuous-Manski port: the Thm-2 sampler probe.

Runs the ManskiSampler on the frozen swamp dataset (no training) and checks
that the pessimism machinery behaves as the theory predicts BEFORE any
gradient is spent:

  G5_degeneration_no_teleports   p_override=1 produces ZERO teleports and its
                                 endpoint-cell distribution is compared (INFO)
                                 against the existing replay.py truncated-
                                 geometric law -- the TV gap quantifies the
                                 continuation-vs-truncation difference and is
                                 why the baseline must be retrained with the
                                 walk sampler for single-variable comparisons.
  G6_teleports_at_decision       teleport rate per continue-decision at the
                                 HOLDING cell exceeds the mean rate over the
                                 downstream shortcut corridor (pessimism
                                 concentrates at the confounded decision).
  G7_endpoints_pessimistic       mean BFS distance of Manski endpoints >
                                 mean BFS distance of p_override=1 endpoints
                                 (the lower bound actually pulls mass toward
                                 worse futures).

Writes <out>/sampler_probe_report.json and <out>/teleport_maps.npz
(teleport/visit heatmaps; the teleport heatmap is a paper figure).

Run:
  python -m scripts.manski_sampler_probe
"""
import argparse
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl import manski
from crl.config import Config


def _cell_hist(walls, xy):
  cells = manski.cells_of(walls, xy)
  h = np.zeros(walls.shape)
  np.add.at(h, (cells[:, 0], cells[:, 1]), 1.0)
  return h / h.sum()


def _grid(walls, values, fmt='{:6.3f}'):
  lines = ['      ' + ''.join(f'   y={j}  ' for j in range(walls.shape[1]))]
  for i in range(walls.shape[0]):
    row = []
    for j in range(walls.shape[1]):
      row.append('   ### ' if walls[i, j] == 1 else fmt.format(values[i, j]))
    lines.append(f'  x={i} ' + ''.join(row))
  return '\n'.join(lines)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='datasets/swamp_matched_teacher_s0.npz')
  ap.add_argument('--env_name', default='point_two_route_swamp_matched_v0')
  ap.add_argument('--table', default='artifacts/manski_port/propensity_table.npz')
  ap.add_argument('--batch', type=int, default=20000)
  ap.add_argument('--gamma', type=float, default=None,
                  help='default: Config.discount')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='artifacts/manski_port')
  args = ap.parse_args()

  cfg = Config(env_name=args.env_name)
  gamma = cfg.discount if args.gamma is None else args.gamma
  env = envs_mod.make_env(args.env_name, cfg, seed=args.seed)
  walls = env._walls
  goal_cell = tuple(manski.cells_of(walls, np.asarray(env.GOAL)))
  holding = tuple(env.HOLDING_CELL)

  data = np.load(args.dataset, allow_pickle=True)
  obs, act = data['obs'], data['act']
  table = np.load(args.table, allow_pickle=True)
  tcfg = json.loads(str(table['config']))
  probs = table['probs']

  dist = manski.bfs_dist_map(walls, goal_cell)
  sampler = manski.ManskiSampler(
      obs, act, walls, probs, tcfg['sectors'], tcfg['zero_thresh'],
      dist, gamma, seed=args.seed)

  full = sampler.sample(args.batch)
  degen = sampler.sample(args.batch, p_override=1.0)
  ref_xy = manski.replay_law_endpoints(obs, gamma, args.batch,
                                       seed=args.seed + 1)

  # ---- G5: degeneration ---------------------------------------------------- #
  degen_teleports = int(degen['teleport_map'].sum())
  tv = 0.5 * np.abs(_cell_hist(walls, degen['endpoint_xy'])
                    - _cell_hist(walls, ref_xy)).sum()
  g5 = degen_teleports == 0

  # ---- G6: teleports concentrate at the decision cell ---------------------- #
  rate = np.where(full['visit_map'] > 0,
                  full['teleport_map'] / np.maximum(full['visit_map'], 1), 0.0)
  downstream = [(i, holding[1]) for i in range(holding[0] + 1, goal_cell[0])
                if walls[i, holding[1]] == 0]
  down_mean = float(np.mean([rate[c] for c in downstream]))
  g6 = bool(rate[holding] > down_mean)

  # ---- G7: endpoints are pessimistic --------------------------------------- #
  def mean_bfs(xy):
    c = manski.cells_of(walls, xy)
    return float(np.mean(dist[c[:, 0], c[:, 1]]))
  bfs_full, bfs_degen = mean_bfs(full['endpoint_xy']), mean_bfs(degen['endpoint_xy'])
  g7 = bool(bfs_full > bfs_degen)

  # ---- G8: pessimistic but NOT vacuous -------------------------------------- #
  # If Geom(1-gamma) compounding absorbs every walk at the worst corner, the
  # endpoint no longer depends on the anchor and the critic gets no ranking
  # signal. Anchor-vs-endpoint BFS correlation measures surviving signal.
  def bfs_of(xy):
    c = manski.cells_of(walls, xy)
    return dist[c[:, 0], c[:, 1]].astype(float)
  corr_full = float(np.corrcoef(bfs_of(full['anchor_xy']),
                                bfs_of(full['endpoint_xy']))[0, 1])
  corr_degen = float(np.corrcoef(bfs_of(degen['anchor_xy']),
                                 bfs_of(degen['endpoint_xy']))[0, 1])
  g8 = bool(corr_full > 0.15)

  teleport_frac = float(full['teleport_map'].sum()
                        / max(full['visit_map'].sum(), 1))
  print(f'gamma={gamma}  batch={args.batch}  '
        f'teleport fraction per continue-step: {teleport_frac:.3f}')
  print('\nteleport RATE per continue-decision (full Manski walk):')
  print(_grid(walls, rate))
  print('\nteleport COUNTS:')
  print(_grid(walls, full['teleport_map'].astype(float), fmt='{:6.0f}'))
  print(f'\nendpoint mean BFS dist: manski {bfs_full:.3f}  '
        f'no-pessimism {bfs_degen:.3f}  replay-law TV(info) {tv:.3f}')

  gates = [
      dict(name='G5_degeneration_no_teleports', passed=bool(g5),
           metrics=dict(degen_teleports=degen_teleports,
                        replay_law_tv_info=float(tv))),
      dict(name='G6_teleports_at_decision', passed=g6,
           metrics=dict(holding_rate=float(rate[holding]),
                        downstream_mean_rate=down_mean)),
      dict(name='G7_endpoints_pessimistic', passed=g7,
           metrics=dict(manski_mean_bfs=bfs_full,
                        no_pessimism_mean_bfs=bfs_degen)),
      dict(name='G8_not_vacuous', passed=g8,
           metrics=dict(anchor_endpoint_bfs_corr=corr_full,
                        no_pessimism_corr=corr_degen,
                        threshold='> 0.15')),
  ]
  print()
  for g in gates:
    print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']}  {g['metrics']}")

  os.makedirs(args.out, exist_ok=True)
  np.savez(os.path.join(args.out, 'teleport_maps.npz'),
           teleport_map=full['teleport_map'], visit_map=full['visit_map'],
           rate_map=rate)
  report = dict(gamma=gamma, batch=args.batch, teleport_frac=teleport_frac,
                gates=gates,
                all_passed=bool(all(g['passed'] for g in gates)))
  with open(os.path.join(args.out, 'sampler_probe_report.json'), 'w') as f:
    json.dump(report, f, indent=1)
  print(f"\n{'ALL GATES PASS' if report['all_passed'] else 'GATES FAILED'}"
        f' -> {args.out}')


if __name__ == '__main__':
  main()
