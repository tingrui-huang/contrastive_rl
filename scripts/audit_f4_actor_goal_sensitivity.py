"""Does the actor's action depend on the goal?  Compare fixed-critic actor arms.

At the 16 held-out fork roots, the mode action tanh(loc) of each actor under:
  * the canonical task goal (8.5, 3.5);
  * a goal inside the lower corridor (1.5, 1.0);
  * a goal inside the shortcut corridor (4.5, 3.5);
  * the start cell (0.5, 3.5);
  * 64 relabeled future goals drawn from the D replay (F4 future states, the
    kind of goal the actor was trained with).
Reports the mean mode action per goal, the mean over roots of the spread of
the mode action across the 64 replay goals (goal sensitivity), and the
displacement between the canonical and the lower-corridor goal.

Usage::
  python scripts/audit_f4_actor_goal_sensitivity.py \\
      --arm "clip rg0.5=outputs/pointmaze_fixed_dcritic_actor_v1/critic_D1" \\
      --arm "acme rg0.5=outputs/pointmaze_fixed_dcritic_actor_acme_v1/critic_D1" \\
      --arm "acme rg0=outputs/pointmaze_fixed_dcritic_actor_acme_rg0_v1/critic_D1"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
OUT = ROOT / 'outputs' / 'pointmaze_actor_goal_sensitivity_v1'
NAMED_GOALS = {'canonical (8.5,3.5)': (8.5, 3.5), 'lower corridor (1.5,1.0)': (1.5, 1.0),
               'shortcut corridor (4.5,3.5)': (4.5, 3.5), 'start (0.5,3.5)': (0.5, 3.5)}


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--arm', action='append', required=True,
                  help='LABEL=dir containing actor_s{0,1,2}/final.pkl')
  ap.add_argument('--n-replay-goals', type=int, default=64)
  args = ap.parse_args(argv)
  import jax.numpy as jnp
  from crl import checkpoint
  nets = make_network()
  with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
               / 'root_selection.npz', allow_pickle=False) as d:
    roots = d['state'].astype(np.float32)
  n = len(roots)
  rng = np.random.default_rng(0)
  with np.load(FIX / 'replay_D.npz', allow_pickle=False) as d:
    obs = d['obs']
    lengths = d['lengths']
  # relabeled future goals: random rows of random episodes, the full F4 state
  ep = rng.integers(0, len(obs), size=args.n_replay_goals)
  t = np.array([rng.integers(1, int(lengths[e])) for e in ep])
  replay_goals = obs[ep, t, :8].astype(np.float32)
  goals = {k: np.tile(np.array(v, np.float32), 4) for k, v in NAMED_GOALS.items()}

  OUT.mkdir(parents=True, exist_ok=True)
  lines = ['# Goal sensitivity of the fixed-critic actors at the 16 fork roots', '',
           '| arm | seed | ' + ' | '.join(f'mode @ {k}' for k in NAMED_GOALS)
           + ' | spread over 64 replay goals (std x, y) | |mode(canonical) - mode(lower)| |',
           '|---|---|' + '---|' * (len(NAMED_GOALS) + 2)]
  results = {}
  for spec in args.arm:
    label, d = spec.split('=', 1)
    results[label] = []
    for a in range(3):
      path = Path(d) / f'actor_s{a}' / 'final.pkl'
      if not path.exists():
        continue
      _, st = checkpoint.load_checkpoint(path)
      def mode_for(goal_rows):
        o = np.concatenate([roots, goal_rows], 1).astype(np.float32)
        return np.tanh(np.asarray(nets.policy_network.apply(st.policy_params, jnp.asarray(o)).loc))
      named = {k: mode_for(np.broadcast_to(g, roots.shape)) for k, g in goals.items()}
      per_goal = np.stack([mode_for(np.broadcast_to(g, roots.shape)) for g in replay_goals])  # [G, n, 2]
      spread = per_goal.std(0).mean(0)
      disp = np.linalg.norm(named['canonical (8.5,3.5)'] - named['lower corridor (1.5,1.0)'], axis=1).mean()
      row = {'seed': a, 'named': {k: v.mean(0) for k, v in named.items()},
             'replay_goal_spread': spread, 'canonical_minus_lower': float(disp),
             'lower_goal_mode_y_mean': float(named['lower corridor (1.5,1.0)'][:, 1].mean())}
      results[label].append(row)
      lines.append(f'| {label} | {a} | ' + ' | '.join(
          f'({row["named"][k][0]:+.2f},{row["named"][k][1]:+.2f})' for k in NAMED_GOALS)
          + f' | ({spread[0]:.3f}, {spread[1]:.3f}) | {disp:.3f} |')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  json.dump({k: [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else
                       {k3: v3.tolist() for k3, v3 in vv.items()} if isinstance(vv, dict) else vv)
                  for kk, vv in r.items()} for r in v] for k, v in results.items()},
            open(OUT / 'results.json', 'w', encoding='utf-8'), indent=2)
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
