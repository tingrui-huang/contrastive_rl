"""The fork's action landscape: what the law says per action region, what each critic scores.

Step 11's first stratified critics have point margins q(DOWN) - q(RIGHT) of
-0.10 / +0.05 / +0.27 but width-0.75 margins of -0.36 / -0.38 / -0.25 on the
jitter rows, i.e. the neighbourhoods the actor averages over disagree with
the two probe points.  This audit puts the two side by side on a coarse
action grid:

A. The law: for fork anchors of replay_E (buffer law), the recorded action
   binned on a 6 x 6 grid over [-1, 1]^2; per bin the anchor mass and the
   discounted P(goal cell (8,3)) -- the quantity an exact NCE critic would
   exponentiate.  Also the mass-weighted P((8,3)) over the DOWN-loc and
   RIGHT-loc neighbourhoods (tanh(loc + 0.75 eps)), i.e. the law's own
   "width margin".
B. Each critic: mean over the Step 10b jitter rows (fork anchor, jittering
   (8,3) goal frame) of q(s, a, g) - q(s, RIGHT, g) at the bin centres, and
   the width-neighbourhood averages, so the sign and the place of every
   disagreement with A is visible.

Usage: python scripts/audit_f4_fork_action_landscape.py --critics "strat seed 2=outputs/.../final.pkl" ...
Outputs: outputs/pointmaze_fork_strata_v1/action_landscape/{REPORT.md,results.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_f4_fork_strata as strata_mod   # noqa: E402
from audit_f4_goal_frames import frame_type, DOWN_LOC, RIGHT_LOC   # noqa: E402

REPLAY_E = strata_mod.REPLAY_E
OUT = strata_mod.OUT / 'action_landscape'
NB = 6
EDGES = np.linspace(-1, 1, NB + 1)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--critics', nargs='+', required=True, help='label=path pairs')
  ap.add_argument('--rows', type=int, default=1500)
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  from crl.bc_balanced import GroupBalancedBCSampler
  net = strata_mod.make_network()
  rng = np.random.default_rng(777)
  with np.load(REPLAY_E, allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths'].astype(np.int64)
  smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=0.25, seed=777)
  tr, ii, jj = smp.tr, smp.ii.astype(np.int64), smp.jj.astype(np.int64)
  w = smp.w_original
  s_cell = np.floor(obs[tr, ii, :2]).astype(np.int64)
  g_cell = np.floor(obs[tr, jj, :2]).astype(np.int64)
  fork = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3)
  g83 = (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)
  a_rec = np.clip(act[tr, ii], -1, 1)
  bx = np.minimum(np.searchsorted(EDGES, a_rec[:, 0], side='right') - 1, NB - 1)
  by = np.minimum(np.searchsorted(EDGES, a_rec[:, 1], side='right') - 1, NB - 1)
  # ---- A. the law on the grid ----
  mass = np.zeros((NB, NB)); p83 = np.full((NB, NB), np.nan)
  for i in range(NB):
    for j in range(NB):
      m = fork & (bx == i) & (by == j)
      tot = w[m].sum()
      mass[i, j] = tot
      if tot > 0:
        p83[i, j] = w[m & g83].sum() / tot
  mass /= mass.sum()
  eps = rng.standard_normal((4096, 2)).astype(np.float32)

  def law_neighbourhood(loc):
    a = np.tanh(loc[None] + 0.75 * eps)
    bi = np.minimum(np.searchsorted(EDGES, a[:, 0], side='right') - 1, NB - 1)
    bj = np.minimum(np.searchsorted(EDGES, a[:, 1], side='right') - 1, NB - 1)
    vals = p83[bi, bj]
    return float(np.nanmean(vals)), float(np.mean(np.isnan(vals)))
  law_down, law_down_nan = law_neighbourhood(DOWN_LOC)
  law_right, law_right_nan = law_neighbourhood(RIGHT_LOC)
  res = {'grid_edges': EDGES.tolist(), 'law_mass': mass.tolist(), 'law_p83': p83.tolist(),
         'law_width': {'down_loc_p83': law_down, 'right_loc_p83': law_right,
                       'log_ratio': float(np.log(law_down / law_right)),
                       'unbinned_share_down': law_down_nan, 'unbinned_share_right': law_right_nan}}

  def grid_table(vals, fmt='{:+.2f}', title=''):
    lines = [f'**{title}**', '', '| a_y \\ a_x | ' + ' | '.join(f'{c:+.2f}' for c in CENTRES) + ' |',
             '|---|' + '---:|' * NB]
    for j in range(NB - 1, -1, -1):        # top row = a_y near +1
      lines.append(f'| {CENTRES[j]:+.2f} | ' + ' | '.join(
          ('--' if np.isnan(vals[i, j]) else fmt.format(vals[i, j])) for i in range(NB)) + ' |')
    return lines + ['']
  L = ['# Fork action landscape: the law versus the critics', '',
       f'Grid: {NB} x {NB} bins over the action square; rows a_y (top = up), columns a_x (right = +x). '
       'A: fork anchors of replay_E under the buffer law. B: mean over the Step 10b jitter rows of '
       'q(s, a_bin_centre, g) - q(s, RIGHT, g); the width rows average tanh(loc + 0.75 eps) over the bins.', '',
       '## A. The law', '']
  L += grid_table(mass, '{:.3f}', 'anchor mass per bin (share of fork anchors)')
  L += grid_table(p83, '{:.2f}', 'P(goal cell (8,3) | fork anchor, action bin), discounted law')
  L += [f'Law over the width neighbourhoods: DOWN loc {law_down:.3f} (unbinned share {law_down_nan:.2f}), '
        f'RIGHT loc {law_right:.3f} ({law_right_nan:.2f}); log ratio **{np.log(law_down / law_right):+.3f}** '
        f'(point target: bin (0,-1) vs bin (1,0)).', '']
  # ---- B. critics ----
  rows = strata_mod.jitter_rows(n_rows=args.rows)
  grid = np.stack(np.meshgrid(CENTRES, CENTRES, indexing='ij'), -1).reshape(-1, 2).astype(np.float32)
  eps_j = jnp.asarray(rng.standard_normal((256, 2)).astype(np.float32))

  @jax.jit
  def q_grid(qp, o):
    o_rep = jnp.repeat(o[:, None, :], grid.shape[0], 1).reshape(-1, o.shape[1])
    a = jnp.broadcast_to(jnp.asarray(grid)[None], (o.shape[0],) + grid.shape).reshape(-1, 2)
    phi, psi = net.representation_network.apply(qp, o_rep, a)
    return jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], NB, NB)

  @jax.jit
  def q_point(qp, o, a):
    phi, psi = net.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  @jax.jit
  def qbar(qp, o, loc):
    a = jnp.tanh(loc[None, None, :] + 0.75 * eps_j[None])
    a = jnp.broadcast_to(a, (o.shape[0],) + a.shape[1:])
    o_rep = jnp.repeat(o[:, None, :], eps_j.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = net.representation_network.apply(qp, o_rep, a.reshape(-1, 2))
    return jnp.mean(jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1), axis=1)

  L += ['## B. The critics: q(a) - q(RIGHT) on the jitter rows', '']
  res['critics'] = {}
  o_all = jnp.asarray(rows)
  r_ = jnp.tile(jnp.array([[1.0, 0.0]], jnp.float32), (256, 1))
  for spec in args.critics:
    label, path = spec.split('=', 1)
    qp = checkpoint.load_checkpoint(path)[1].q_params
    g_acc = np.zeros((NB, NB)); wd = []; wr = []; qr_all = []
    for k in range(0, len(rows), 256):
      o = o_all[k:k + 256]
      qr = np.asarray(q_point(qp, o, r_[:o.shape[0]]))
      g_acc += (np.asarray(q_grid(qp, o)) - qr[:, None, None]).sum(0)
      wd.append(np.asarray(qbar(qp, o, jnp.asarray(DOWN_LOC))) - qr)
      wr.append(np.asarray(qbar(qp, o, jnp.asarray(RIGHT_LOC))) - qr)
    g_mean = g_acc / len(rows)
    wd, wr = float(np.concatenate(wd).mean()), float(np.concatenate(wr).mean())
    res['critics'][label] = {'grid': g_mean.tolist(), 'width_down_minus_right': wd - wr,
                             'point_down_minus_right': float(g_mean[NB // 2 - 1, 0] + g_mean[NB // 2, 0]) / 2}
    L += grid_table(g_mean, '{:+.2f}', f'{label}: q(a) - q(RIGHT); width DOWN loc {wd:+.3f}, RIGHT loc {wr:+.3f}, '
                    f'margin **{wd - wr:+.3f}**')
    print(f'{label}: width margin {wd - wr:+.3f}', flush=True)
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (OUT / 'results.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  print('\n'.join(L))
  return 0


if __name__ == '__main__':
  sys.exit(main())
