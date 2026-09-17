"""The goal frames the actor trains on versus the goal frame the probe uses.

audit_f4_right_family.py found that the RIGHT-family critics prefer RIGHT on
the replay's fork -> goal-cell-(8,3) rows at the policy width although they
rank DOWN first on the 16 held-out roots under the canonical goal
tile((8.5, 3.5), 4).  The two probes differ in the goal frame: the canonical
goal is a stationary 4-frame stack, whereas the relabeled (8,3) goals of the
replay are frames of an agent moving around the goal cell (median
frame-to-frame displacement 0.22; stationary frames are 3% of the mass) or
arriving into it.  This audit measures the critics and the actors on those
training goal frames directly, stratum by stratum, and crosses the two
factors (anchor state, goal frame):

D1. fork -> (8,3) rows drawn by the buffer law, stratified by goal-frame
    type -- jitter (all four frames inside (8,3)), parked (jitter with
    displacement < 0.05), arrival along the middle route, arrival from below
    -- and by displacement bin: per critic the point margin q(DOWN) -
    q(RIGHT), the width-0.75 margin Qbar(DOWN loc) - Qbar(RIGHT loc), and the
    argmax of the point score over the action square classified as
    lower/shortcut by its noiseless physics; per actor the mode under those
    rows and the share of modes with y < -0.5.
D2. Cross probes: the 16 roots x {canonical goal, replay jitter frames,
    replay arrival frames}; replay fork anchors (t = 1 and all) x canonical
    goal.  Which factor carries the DOWN preference.
D3. The data's own answer: from fork anchors, by recorded action sector, the
    discounted probability that the relabeled goal lies in cell (8,3) --
    what an exact NCE critic would rank.

Outputs: outputs/pointmaze_balanced_bc_joint_v1/right_family_audit/goal_frames_{REPORT.md,results.json}.
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
from audit_f4_right_family import ACTORS, CRITICS, FIX, OUT, WIDTH, GOAL   # noqa: E402
from crl.envs import _TWO_ROUTE_SWAMP_WALLS as WALLS   # noqa: E402

DOWN_LOC = np.array([0.3, -6.0], np.float32)
RIGHT_LOC = np.array([5.0, -0.1], np.float32)
DISP_BINS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 10.0))


def frame_type(g):
  """g: [N, 8] goal stacks, newest first."""
  fr = g.reshape(-1, 4, 2)
  inside = np.all(np.floor(fr) == np.array([8, 3]), axis=(1, 2))
  disp = np.linalg.norm(g[:, :2] - g[:, 2:4], axis=1)
  below = (fr[:, :, 1] < 3.0).any(1)
  t = np.full(len(g), 'arrival-middle', dtype=object)
  t[inside] = 'jitter'
  t[inside & (disp < 0.05)] = 'parked'
  t[~inside & below] = 'arrival-below'
  return t, disp


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--rows', type=int, default=8000)
  ap.add_argument('--seed', type=int, default=777)
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  from crl.bc_balanced import GroupBalancedBCSampler
  nets = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None, log_prob_mode='acme')
  rng = np.random.default_rng(args.seed)
  with np.load(FIX / 'replay_D.npz', allow_pickle=False) as d:
    obs, act, lengths, src = d['obs'], d['act'], d['lengths'], d['audit_source']
  smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=0.25, seed=args.seed)
  w = smp.w_original
  s_cell = np.floor(obs[smp.tr, smp.ii, :2]).astype(np.int64)
  g_cell = np.floor(obs[smp.tr, smp.jj, :2]).astype(np.int64)
  fork = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3)
  f83 = fork & (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)

  def draw(mask, n):
    ww = np.where(mask, w, 0.0)
    cdf = np.cumsum(ww / ww.sum())
    pos = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'), len(cdf) - 1)
    return pos
  pos = draw(f83, args.rows)
  tr, ii, jj = smp.tr[pos], smp.ii[pos].astype(np.int64), smp.jj[pos].astype(np.int64)
  S = obs[tr, ii, :8].astype(np.float32)
  Gf = obs[tr, jj, :8].astype(np.float32)
  A_rec = act[tr, ii].astype(np.float32)
  gtype, disp = frame_type(Gf)
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)

  eps = jnp.asarray(rng.standard_normal((256, 2)).astype(np.float32))
  grid = np.stack(np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41), indexing='ij'), -1).reshape(-1, 2).astype(np.float32)

  @jax.jit
  def q_point(qp, o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  @jax.jit
  def qbar(qp, o, loc):
    a = jnp.tanh(loc[None, None, :] + WIDTH * eps[None])
    a = jnp.broadcast_to(a, (o.shape[0],) + a.shape[1:])
    o_rep = jnp.repeat(o[:, None, :], eps.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = nets.representation_network.apply(qp, o_rep, a.reshape(-1, 2))
    return jnp.mean(jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1), axis=1)

  @jax.jit
  def q_grid(qp, o):
    o_rep = jnp.repeat(o[:, None, :], grid.shape[0], 1).reshape(-1, o.shape[1])
    a = jnp.broadcast_to(jnp.asarray(grid)[None], (o.shape[0],) + grid.shape).reshape(-1, 2)
    phi, psi = nets.representation_network.apply(qp, o_rep, a)
    return jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1)

  @jax.jit
  def mode_of(pp, o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  def batched(fn, params, o, per_row=None, const=None, bs=1024):
    out = []
    for k in range(0, len(o), bs):
      extra = []
      if per_row is not None:
        extra.append(jnp.asarray(per_row[k:k + bs]))
      if const is not None:
        extra.append(const)
      out.append(np.asarray(fn(*params, jnp.asarray(o[k:k + bs]), *extra)))
    return np.concatenate(out)

  down = np.tile(np.array([[0.0, -1.0]], np.float32), (args.rows, 1))
  right = np.tile(np.array([[1.0, 0.0]], np.float32), (args.rows, 1))

  def physics_of_argmax(o, qg):
    """Noiseless one-step exit of the argmax action from the anchor's newest
    frame under the env's axis-by-axis substep scheme and the real wall grid:
    lower if the point ends below y = 3, shortcut if it ends at x >= 2."""
    a = grid[np.argmax(qg, axis=1)]
    x, y = o[:, 0].astype(np.float64).copy(), o[:, 1].astype(np.float64).copy()
    h, wdt = WALLS.shape

    def blocked(px, py):
      out = (px < 0) | (py < 0) | (px > h) | (py > wdt)
      ci = np.clip(np.floor(px).astype(int), 0, h - 1)
      cj = np.clip(np.floor(py).astype(int), 0, wdt - 1)
      return out | (WALLS[ci, cj] == 1)
    for _ in range(10):
      nx = x + 0.1 * a[:, 0]
      x = np.where(blocked(nx, y), x, nx)
      ny = y + 0.1 * a[:, 1]
      y = np.where(blocked(x, ny), y, ny)
    return np.where(y < 3.0, 'lower', np.where(x >= 2.0, 'shortcut', 'stay')), a

  crit = {n: checkpoint.load_checkpoint(p)[1].q_params for n, p in CRITICS.items()}
  actors = {n: checkpoint.load_checkpoint(p)[1].policy_params for n, p in ACTORS.items()}
  res = {'settings': vars(args), 'D1': {}, 'D2': {}, 'D3': {}}
  L = ['# Goal frames: what the actor trains on versus what the probe measures', '',
       f'{args.rows:,} fork -> (8,3) rows drawn by the buffer law (mass 0.0069 of the buffer, '
       '13% of the fork rows).  Goal-frame types: jitter = all four frames inside cell (8,3); '
       'parked = jitter with newest displacement < 0.05; arrival-middle / arrival-below = some '
       'frame outside (8,3), from y >= 3 / y < 3.  Margins: point = q(DOWN) - q(RIGHT); width '
       f'= Qbar(DOWN loc) - Qbar(RIGHT loc) at width {WIDTH}; positive = DOWN preferred.', '']
  O = np.concatenate([S, Gf], 1)
  strata = {t: gtype == t for t in ('jitter', 'parked', 'arrival-middle', 'arrival-below')}
  strata['jitter+parked'] = (gtype == 'jitter') | (gtype == 'parked')
  for lo, hi in DISP_BINS:
    strata[f'jitter disp [{lo}, {hi})'] = strata['jitter+parked'] & (disp >= lo) & (disp < hi)
  # ---- D1 -----------------------------------------------------------------
  L += ['## D1. Critics on the training goal frames', '',
        '| critic | stratum | rows (share) | point margin | width margin | argmax -> lower / shortcut |',
        '|---|---|---:|---:|---:|---:|']
  for cn, qp in crit.items():
    pm = batched(q_point, (qp,), O, per_row=down) - batched(q_point, (qp,), O, per_row=right)
    wm = batched(qbar, (qp,), O, const=jnp.asarray(DOWN_LOC)) - batched(qbar, (qp,), O, const=jnp.asarray(RIGHT_LOC))
    qg = batched(q_grid, (qp,), O)
    phys, _ = physics_of_argmax(O, qg)
    res['D1'][cn] = {}
    for sn, m in strata.items():
      if m.sum() < 30:
        continue
      row = {'rows': int(m.sum()), 'share': float(m.mean()), 'point_margin': float(pm[m].mean()),
             'width_margin': float(wm[m].mean()), 'argmax_lower': float((phys[m] == 'lower').mean()),
             'argmax_shortcut': float((phys[m] == 'shortcut').mean())}
      res['D1'][cn][sn] = row
      L.append(f'| {cn} | {sn} | {row["rows"]} ({row["share"]:.2f}) | {row["point_margin"]:+.3f} '
               f'| {row["width_margin"]:+.3f} | {row["argmax_lower"]:.2f} / {row["argmax_shortcut"]:.2f} |')
  L += ['', '### Actors on the same rows: mean mode (x, y) and share of modes with y < -0.5', '',
        '| actor | ' + ' | '.join(s for s in strata if s.startswith('jitter') or s.startswith('arrival') or s == 'parked') + ' |',
        '|---|' + '---:|' * len([s for s in strata if s.startswith('jitter') or s.startswith('arrival') or s == 'parked'])]
  res['D1']['actors'] = {}
  for an, pp in actors.items():
    md = batched(mode_of, (pp,), O)
    cells, rec = [], {}
    for sn, m in strata.items():
      if not (sn.startswith('jitter') or sn.startswith('arrival') or sn == 'parked'):
        continue
      if m.sum() < 30:
        cells.append('--')
        continue
      mx, my = md[m].mean(0)
      fd = float((md[m][:, 1] < -0.5).mean())
      rec[sn] = {'mode': [float(mx), float(my)], 'down_share': fd}
      cells.append(f'({mx:+.2f}, {my:+.2f}) {fd:.2f}')
    res['D1']['actors'][an] = rec
    L.append(f'| {an} | ' + ' | '.join(cells) + ' |')
  # ---- D2 -----------------------------------------------------------------
  L += ['', '## D2. Cross probes: anchor factor x goal-frame factor (point margin / width margin)', '',
        '| critic | 16 roots x canonical | 16 roots x jitter frames | 16 roots x arrival-middle | '
        '16 roots x arrival-below | replay t=1 anchors x canonical | all replay anchors x canonical |',
        '|---|---:|---:|---:|---:|---:|---:|']
  jit_frames = Gf[strata['jitter+parked']][:300]
  am_frames = Gf[strata['arrival-middle']][:300]
  ab_frames = Gf[strata['arrival-below']][:300]
  t1 = ii == 1
  probes = {
      '16 roots x canonical': np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1),
      '16 roots x jitter frames': np.concatenate([np.repeat(roots, len(jit_frames), 0),
                                                  np.tile(jit_frames, (16, 1))], 1),
      '16 roots x arrival-middle': np.concatenate([np.repeat(roots, len(am_frames), 0),
                                                   np.tile(am_frames, (16, 1))], 1),
      '16 roots x arrival-below': np.concatenate([np.repeat(roots, len(ab_frames), 0),
                                                  np.tile(ab_frames, (16, 1))], 1),
      'replay t=1 anchors x canonical': np.concatenate([S[t1][:2000], np.broadcast_to(GOAL, (min(2000, t1.sum()), 8))], 1),
      'all replay anchors x canonical': np.concatenate([S[:3000], np.broadcast_to(GOAL, (min(3000, len(S)), 8))], 1),
  }
  for cn, qp in crit.items():
    cells, rec = [], {}
    for pn, o in probes.items():
      o = o.astype(np.float32)
      d_ = np.tile(np.array([[0.0, -1.0]], np.float32), (len(o), 1))
      r_ = np.tile(np.array([[1.0, 0.0]], np.float32), (len(o), 1))
      pm = float((batched(q_point, (qp,), o, per_row=d_) - batched(q_point, (qp,), o, per_row=r_)).mean())
      wm = float((batched(qbar, (qp,), o, const=jnp.asarray(DOWN_LOC)) - batched(qbar, (qp,), o, const=jnp.asarray(RIGHT_LOC))).mean())
      rec[pn] = {'point': pm, 'width': wm}
      cells.append(f'{pm:+.2f} / {wm:+.2f}')
    res['D2'][cn] = rec
    L.append(f'| {cn} | ' + ' | '.join(cells) + ' |')
  L += ['', '### Actors on the cross probes: mean mode (x, y)', '',
        '| actor | ' + ' | '.join(probes) + ' |', '|---|' + '---:|' * len(probes)]
  res['D2']['actors'] = {}
  for an, pp in actors.items():
    rec, cells = {}, []
    for pn, o in probes.items():
      md = batched(mode_of, (pp,), o.astype(np.float32)).mean(0)
      rec[pn] = md.tolist()
      cells.append(f'({md[0]:+.2f}, {md[1]:+.2f})')
    res['D2']['actors'][an] = rec
    L.append(f'| {an} | ' + ' | '.join(cells) + ' |')
  # ---- D3 -----------------------------------------------------------------
  L += ['', '## D3. The data\'s own answer: P(goal cell = (8,3) | fork anchor, recorded action sector), '
        'discounted relabeling law, by source', '']
  ang = np.degrees(np.arctan2(act[smp.tr, smp.ii, 1], act[smp.tr, smp.ii, 0]))
  sectors = {'R': (ang > -22.5) & (ang < 22.5), 'DR': (ang <= -22.5) & (ang >= -67.5),
             'D': (ang < -67.5) & (ang > -112.5)}
  g83 = (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)
  L += ['| source | sector | fork-anchor mass | P((8,3) goal) | P(jitter (8,3) goal) |', '|---|---|---:|---:|---:|']
  src_row = src[smp.tr]
  gt_all = np.full(len(w), '', dtype=object)
  gt_all[f83], _ = frame_type(obs[smp.tr[f83], smp.jj[f83].astype(np.int64), :8])
  for s, sname in ((-1, 'all'), (0, 'original'), (1, 'C queries'), (2, 'diagonal queries')):
    ms = fork if s < 0 else fork & (src_row == s)
    for sec, mm in sectors.items():
      m = ms & mm
      tot = w[m].sum()
      p83 = w[m & g83].sum() / tot if tot > 0 else float('nan')
      pj = w[m & g83 & ((gt_all == 'jitter') | (gt_all == 'parked'))].sum() / tot if tot > 0 else float('nan')
      res['D3'][f'{sname}|{sec}'] = {'mass': float(tot), 'p83': float(p83), 'p_jitter83': float(pj)}
      L.append(f'| {sname} | {sec} | {tot:.5f} | {p83:.3f} | {pj:.3f} |')
  (OUT / 'goal_frames_REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (OUT / 'goal_frames_results.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  print('\n'.join(L))
  return 0


if __name__ == '__main__':
  sys.exit(main())
