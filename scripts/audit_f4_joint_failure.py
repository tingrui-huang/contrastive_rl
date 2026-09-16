"""Why joint training with balanced BC rows fails while the two-stage recipe works.

Two read-only audits on the Step 9 checkpoints
(outputs/pointmaze_balanced_bc_joint_v1):

A. Actor state at the 16 held-out fork roots under the task goal: mode
   action, pre-tanh loc, tanh slope 1 - tanh(loc)^2 (how far the critic's
   gradient can still move the action), policy scale, and the critic's own
   score at the actor's mode versus at DOWN and RIGHT.  A saturated x with a
   critic that scores DOWN above the mode is the "committed early, cannot
   recover" signature.

B. Critic preference along the APPROACH to the fork, not only at the roots:
   Qbar(DOWN loc) - Qbar(RIGHT loc) at policy width 0.75 for frame-stacked
   states x in {0.5..1.45} x y in {3.2, 3.5, 3.8}, with a rightward velocity
   (as the learner arrives) and standing still.  The 16 roots all lie at
   x ~ 1.45-1.5; a critic can rank DOWN first on all of them and still prefer
   RIGHT everywhere the policy actually decides.

Outputs: outputs/pointmaze_balanced_bc_joint_v1/failure_audit/{REPORT.md,results.json}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
J = ROOT / 'outputs' / 'pointmaze_balanced_bc_joint_v1'
STEP8 = ROOT / 'outputs' / 'pointmaze_fixed_dcritic_actor_bcbal_v1' / 'critic_D1' / 'balanced'
OUT = J / 'failure_audit'
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
XS = (0.5, 0.8, 1.1, 1.3, 1.45)
YS = (3.2, 3.5, 3.8)
DOWN_LOC = np.array([0.3, -6.0], np.float32)      # the Step 8 actors' resting loc
RIGHT_LOC = np.array([5.0, -0.1], np.float32)     # the joint actors' resting loc
WIDTH = 0.75

ACTOR_SETS = {
    'joint 30k shared': [J / 'joint' / 'b30k' / f'seed_{s}' / 'shared' / 'final.pkl' for s in range(3)],
    'joint 30k balanced': [J / 'joint' / 'b30k' / f'seed_{s}' / 'balanced' / 'final.pkl' for s in range(3)],
    'joint 300k shared': [J / 'joint' / 'b300k' / f'seed_{s}' / 'shared' / 'final.pkl' for s in range(3)],
    'joint 300k balanced': [J / 'joint' / 'b300k' / f'seed_{s}' / 'balanced' / 'final.pkl' for s in range(3)],
    'frozen joint-300k critic, fresh balanced actors': [
        J / 'refreeze' / 'critic_b300k_s0' / f'actor_s{a}' / 'final.pkl' for a in range(3)],
    'frozen joint-30k critic, fresh balanced actors': [
        J / 'refreeze' / 'critic_b30k_s0' / f'actor_s{a}' / 'final.pkl' for a in range(3)],
    'frozen D1, fresh balanced actors (Step 8)': [STEP8 / f'actor_s{a}' / 'final.pkl' for a in range(3)],
}
CRITICS = {
    'D1 (diag-fix seed 1, 30k, RTX 4080)': FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl',
    'D0': FIX / 'seeds' / 'seed_0' / 'crl' / 'D' / 'final.pkl',
    'D2': FIX / 'seeds' / 'seed_2' / 'crl' / 'D' / 'final.pkl',
    'joint 30k balanced seed 0 (RTX 5060 Ti)': J / 'joint' / 'b30k' / 'seed_0' / 'balanced' / 'final.pkl',
    'joint 30k shared seed 2': J / 'joint' / 'b30k' / 'seed_2' / 'shared' / 'final.pkl',
    'joint 300k balanced seed 0': J / 'joint' / 'b300k' / 'seed_0' / 'balanced' / 'final.pkl',
    'joint 300k balanced seed 1': J / 'joint' / 'b300k' / 'seed_1' / 'balanced' / 'final.pkl',
    'joint 300k shared seed 2': J / 'joint' / 'b300k' / 'seed_2' / 'shared' / 'final.pkl',
    'C seed 2 (sealed, fails)': ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
                                / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl',
}


def main():
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  nets = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  obs = np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1).astype(np.float32)

  @jax.jit
  def q_of(qp, o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  def q_mean(qp, o, a):
    return float(np.mean(np.asarray(q_of(qp, jnp.asarray(o), jnp.asarray(a)))))

  eps = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (256, 2)), np.float32)

  def qbar(qp, o_row, loc):
    a = np.tanh(loc[None] + WIDTH * eps).astype(np.float32)
    return q_mean(qp, np.repeat(o_row[None], len(a), 0), a)

  results = {'actors': {}, 'critics': {}}
  lines = ['# Why joint training fails and the two-stage recipe works', '',
           '## A. Actors at the 16 fork roots (task goal), mean over roots', '',
           '| actor set | seed | mode (x, y) | pre-tanh loc (x, y) | tanh slope (x, y) | scale (x, y) '
           '| q(mode) | q(DOWN) | q(RIGHT) |', '|---|---|---|---|---|---|---:|---:|---:|']
  down = np.tile(np.array([[0.0, -1.0]], np.float32), (16, 1))
  right = np.tile(np.array([[1.0, 0.0]], np.float32), (16, 1))
  for label, paths in ACTOR_SETS.items():
    rows = []
    for a, p in enumerate(paths):
      if not p.exists():
        continue
      _, st = checkpoint.load_checkpoint(p)
      d = nets.policy_network.apply(st.policy_params, jnp.asarray(obs))
      loc, scale = np.asarray(d.loc), np.asarray(d.scale)
      mode = np.tanh(loc)
      row = {'seed': a, 'mode': mode.mean(0).tolist(), 'loc': loc.mean(0).tolist(),
             'tanh_slope': (1 - mode ** 2).mean(0).tolist(), 'scale': scale.mean(0).tolist(),
             'q_mode': q_mean(st.q_params, obs, mode.astype(np.float32)),
             'q_down': q_mean(st.q_params, obs, down), 'q_right': q_mean(st.q_params, obs, right)}
      rows.append(row)
      lines.append(f'| {label} | {a} | ({row["mode"][0]:+.2f}, {row["mode"][1]:+.2f}) '
                   f'| ({row["loc"][0]:+.2f}, {row["loc"][1]:+.2f}) '
                   f'| ({row["tanh_slope"][0]:.3f}, {row["tanh_slope"][1]:.3f}) '
                   f'| ({row["scale"][0]:.2f}, {row["scale"][1]:.2f}) '
                   f'| {row["q_mode"]:+.2f} | {row["q_down"]:+.2f} | {row["q_right"]:+.2f} |')
    results['actors'][label] = rows

  lines += ['', '## B. Critic preference along the approach: Qbar(DOWN loc) - Qbar(RIGHT loc), '
            f'policy width {WIDTH}', '',
            'Frame-stacked states arriving from the left at 0.3 per step (v>0) and standing still (v=0); '
            'positive = DOWN preferred.  The 16 evaluation roots sit at x = 1.44-1.50.', '']
  for label, p in CRITICS.items():
    _, st = checkpoint.load_checkpoint(p)
    qp = st.q_params
    table = {}
    lines += [f'### {label}', '', '| y | ' + ' | '.join(f'x={x} v>0' for x in XS) + ' | '
              + ' | '.join(f'x={x} v=0' for x in XS) + ' |', '|---|' + '---:|' * (2 * len(XS))]
    for y in YS:
      vals = {}
      for vx, tag in ((0.3, 'moving'), (0.0, 'still')):
        for x in XS:
          frames = [np.array([x - k * vx, y], np.float32) for k in range(4)]   # newest first
          o = np.concatenate([np.concatenate(frames), GOAL]).astype(np.float32)
          vals[f'{tag}_x{x}'] = qbar(qp, o, DOWN_LOC) - qbar(qp, o, RIGHT_LOC)
      table[str(y)] = vals
      lines.append(f'| {y} | ' + ' | '.join(f'{vals[f"moving_x{x}"]:+.2f}' for x in XS) + ' | '
                   + ' | '.join(f'{vals[f"still_x{x}"]:+.2f}' for x in XS) + ' |')
    n_pos = sum(v > 0 for row in table.values() for v in row.values())
    lines += ['', f'cells preferring DOWN: {n_pos}/{2 * len(XS) * len(YS)}', '']
    results['critics'][label] = {'table': table, 'cells_preferring_down': int(n_pos)}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  (OUT / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
