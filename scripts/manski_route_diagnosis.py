"""Diagnose WHY the causal arm still takes the shortcut.

Two layers, separating sampler-level truth from what the critic learned:

  1. SAMPLER: Monte-Carlo d_lb / d_pi at the decision cells. For dataset
     anchors at the FORK / HOLDING cell holding a given action bin, run the
     Thm-2 walk (p_override None = d_lb, 1.0 = d_pi) and report
     P(endpoint == goal cell) and endpoint mean BFS distance. If
     d_lb(shortcut) >= d_lb(safe), the pessimism itself never certified the
     safe route (coverage problem in the DATA), and no actor could recover it.
  2. CRITIC: f(s, a, g) for both trained arms at the same decision points --
     does either critic rank the safe action above the shortcut action?

Run:
  python -m scripts.manski_route_diagnosis
"""
import json

import numpy as np

from crl import envs as envs_mod
from crl import manski
from crl.config import Config

ENV = 'point_two_route_swamp_matched_v0'
GOAL_CELL = (8, 3)
POINTS = {
    'fork':    dict(cell=(1, 3), xy=(1.5, 3.5),
                    actions={'shortcut_+x': (1.0, 0.0),
                             'safe_down_-y': (0.0, -1.0)}),
    'holding': dict(cell=(2, 3), xy=(2.5, 3.5),
                    actions={'enter_swamp_+x': (1.0, 0.0),
                             'stay': (0.0, 0.0),
                             'back_-x': (-1.0, 0.0)}),
}


def sampler_part(dataset, table_path, gamma, walks=4000, seed=0,
                 hazard=False, reachable=False):
  data = np.load(dataset, allow_pickle=True)
  obs, act = data['obs'], data['act']
  table = np.load(table_path, allow_pickle=True)
  tcfg = json.loads(str(table['config']))
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=seed)
  walls = env._walls
  dist = manski.bfs_dist_map(walls, GOAL_CELL)
  hazard_cells = env.SWAMP_CELLS if (hazard or reachable) else ()
  if reachable:
    print(f'[reachable-set N(s,x): pessimism ONLY on swamp-involving '
          f'actions, worst = stuck (absorb); u support {hazard_cells}]')
  elif hazard:
    print(f'[hazard V_lb: absorbing swamp cells {hazard_cells}]')
  sampler = manski.ManskiSampler(obs, act, walls, table['probs'],
                                 tcfg['sectors'], tcfg['zero_thresh'],
                                 dist, gamma, seed=seed,
                                 hazard_cells=hazard_cells,
                                 reachable_n=reachable)
  bins_flat = manski.action_bins(act.reshape(-1, 2), tcfg['sectors'],
                                 tcfg['zero_thresh'])
  print(f'== SAMPLER d_lb vs d_pi   (gamma={gamma}, {walks} walks/entry) ==')
  print(f'{"point":9s} {"action":16s} {"n_anchors":>9s} '
        f'{"P(goal)dlb":>11s} {"P(goal)dpi":>11s} {"bfs_dlb":>8s} {"bfs_dpi":>8s}')
  out = {}
  for pname, p in POINTS.items():
    ci, cj = p['cell']
    at_cell = (sampler._anchorable & (sampler._cell_i == ci)
               & (sampler._cell_j == cj))
    for aname, a in p['actions'].items():
      abin = int(manski.action_bins(np.array([a]), tcfg['sectors'],
                                    tcfg['zero_thresh'])[0])
      pool = np.where(at_cell & (bins_flat == abin))[0]
      if not len(pool):
        print(f'{pname:9s} {aname:16s} {0:9d}   (no dataset anchors)')
        continue
      anchors = pool[sampler._rng.integers(len(pool), size=walks)]
      row = {}
      for tag, pov in (('dlb', None), ('dpi', 1.0)):
        end = sampler.walk_from(anchors, p_override=pov)
        ec = manski.cells_of(walls, sampler._xy[end])
        row['pgoal_' + tag] = float(np.mean(
            (ec[:, 0] == GOAL_CELL[0]) & (ec[:, 1] == GOAL_CELL[1])))
        row['bfs_' + tag] = float(np.mean(dist[ec[:, 0], ec[:, 1]]))
      out[f'{pname}/{aname}'] = dict(n_anchors=int(len(pool)), **row)
      print(f'{pname:9s} {aname:16s} {len(pool):9d} '
            f'{row["pgoal_dlb"]:11.4f} {row["pgoal_dpi"]:11.4f} '
            f'{row["bfs_dlb"]:8.3f} {row["bfs_dpi"]:8.3f}')
  return out


def critic_part(ckpts, seed=0):
  import jax.numpy as jnp
  from crl.report_maze import load_nets
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=seed)
  goal = np.asarray(env.GOAL, np.float32)
  print('\n== CRITIC f(s,a,g) at the decision points ==')
  for arm, ck in ckpts.items():
    nets, state, _, step = load_nets(ENV, ck, cfg)
    print(f'[{arm}]  ckpt={ck} (step {step})')
    for pname, p in POINTS.items():
      s = np.asarray(p['xy'], np.float32)
      names = list(p['actions'])
      acts = np.stack([np.asarray(p['actions'][n], np.float32) for n in names])
      obs_k = jnp.asarray(np.tile(np.concatenate([s, goal])[None],
                                  (len(acts), 1)))
      q = np.diag(np.asarray(nets.q_network.apply(
          state.q_params, obs_k, jnp.asarray(acts))))
      order = ' > '.join(names[k] for k in np.argsort(-q))
      qs = '  '.join(f'{n}={q[k]:+.3f}' for k, n in enumerate(names))
      print(f'  {pname:9s} {qs}   ranking: {order}')


def main():
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='datasets/swamp_matched_teacher_s0.npz')
  ap.add_argument('--table', default='artifacts/manski_port/propensity_table.npz')
  ap.add_argument('--gamma', type=float, default=0.95)
  ap.add_argument('--ckpt_causal', default='swamp_manski_s0/best.pkl')
  ap.add_argument('--ckpt_baseline', default='swamp_walkbase_s0/best.pkl')
  ap.add_argument('--no_critic', action='store_true',
                  help='sampler part only (pre-training MC check)')
  ap.add_argument('--hazard', action='store_true',
                  help='treat the static swamp-corridor cells as V_lb=0 '
                       'absorbing hazards (discrete lava analogue)')
  ap.add_argument('--reachable', action='store_true',
                  help='action-dependent reachable-set N(s,x) (the discrete '
                       'worst_case_kernel semantics): pessimism only on '
                       'swamp-involving actions, worst = stuck (absorb)')
  args = ap.parse_args()
  sampler_part(args.dataset, args.table, gamma=args.gamma,
               hazard=args.hazard, reachable=args.reachable)
  if not args.no_critic:
    critic_part({'causal': args.ckpt_causal, 'baseline': args.ckpt_baseline})


if __name__ == '__main__':
  main()
