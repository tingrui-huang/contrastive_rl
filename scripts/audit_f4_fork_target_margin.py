"""What margin does the replay's own law ask the critic for at the fork, and do the E critics reach it?

Step 10 found three of five E critics scoring RIGHT at or above the gentle
DOWN action at the 550 training contexts although the added paths from those
contexts reach the goal 0.76 (DOWN) vs 0.25 (RIGHT).  Before choosing between
"the critic under-fits the fork rows" (batch share, budget, capacity) and "the
law it fits does not ask for DOWN", this script computes the target directly.

At the NCE optimum f(s, a, g) = log p(g | s, a) / p(g) + const, so the
DOWN-vs-RIGHT margin the critic is asked for at a fork state, for a goal
frame in cell (8,3), is

    target = log P(g in (8,3) | s, a in DOWN) - log P(g in (8,3) | s, a in RIGHT)

under the buffer's discounted relabeling law over EVERY source that
contributes fork anchors (original teacher / random-walker episodes, C
queries, diagonal queries, E queries).  The per-path reach rate of one source
is not the target; the discounted, source-mixed conditional is.

A. The target on the fork anchors of replay_E, overall and by source, with
   the discounted (8,3) mass per anchor split into "jitter" goal frames
   (all four frames inside the cell) and arrival frames; also at gamma 0.99
   and undiscounted (reach anywhere in the future) to show what the discount
   does; and restricted to the 550 training contexts' states.
B. The E critics (5) and D critics (3) on the same rows: point margin
   q(DOWN) - q(RIGHT) and width-0.75 margin Qbar(DOWN loc) - Qbar(RIGHT loc)
   with (i) the canonical stationary goal and (ii) the replay's jitter goal
   frames, at the 550 contexts and on the buffer-law fork -> (8,3) rows.

Outputs: outputs/pointmaze_corner_coverage_fix_v1/target_margin/{REPORT.md,results.json}.
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
from audit_f4_goal_frames import frame_type, DOWN_LOC, RIGHT_LOC   # noqa: E402

FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
E = ROOT / 'outputs' / 'pointmaze_corner_coverage_fix_v1'
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
OUT = E / 'target_margin'
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
WIDTH = 0.75
SOURCE_NAMES = {0: 'original', 1: 'C queries', 2: 'diagonal queries', 3: 'E queries'}

CRITICS = {f'E seed {s}': E / 'seeds' / f'seed_{s}' / 'crl' / 'E' / 'final.pkl' for s in range(5)}
CRITICS.update({f'D seed {s}': FIX / 'seeds' / f'seed_{s}' / 'crl' / 'D' / 'final.pkl' for s in range(3)})
FAMILY = {'E seed 0': 'RIGHT', 'E seed 1': 'RIGHT', 'E seed 2': 'DOWN', 'E seed 3': 'RIGHT',
          'E seed 4': 'RIGHT', 'D seed 0': 'CORNER', 'D seed 1': 'DOWN', 'D seed 2': 'CORNER'}


def sectors(a):
  ang = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
  return {'DOWN': (ang < -67.5) & (ang > -112.5), 'DR': (ang <= -22.5) & (ang >= -67.5),
          'RIGHT': (ang > -22.5) & (ang < 22.5)}


def law_weights(lengths, tr, ii, jj, gamma):
  lt = lengths[tr]
  k = lt - 1 - ii.astype(np.int64)
  if gamma >= 1.0:
    return 1.0 / k / (lt - 1) / len(lengths)
  return (gamma ** (jj - ii).astype(np.float64) / (gamma * (1.0 - gamma ** k) / (1.0 - gamma))
          / (lt - 1) / len(lengths))


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--rows', type=int, default=4000)
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  from crl.bc_balanced import GroupBalancedBCSampler
  nets = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None, log_prob_mode='acme')
  rng = np.random.default_rng(777)
  with np.load(E / 'replay_E.npz', allow_pickle=False) as d:
    obs, act, lengths, src = d['obs'], d['act'], d['lengths'].astype(np.int64), d['audit_source']
  smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=0.25, seed=777)
  tr, ii, jj = smp.tr, smp.ii.astype(np.int64), smp.jj.astype(np.int64)
  s_cell = np.floor(obs[tr, ii, :2]).astype(np.int64)
  g_cell = np.floor(obs[tr, jj, :2]).astype(np.int64)
  fork = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3)
  g83 = (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)
  gfr = obs[tr[fork & g83], jj[fork & g83], :8]
  gt = np.full(len(tr), '', dtype=object)
  gt[fork & g83], _ = frame_type(gfr)
  jitter83 = g83 & ((gt == 'jitter') | (gt == 'parked'))
  sec = sectors(act[tr, ii])
  src_row = src[tr]
  # the 550 training contexts: their states (198 roots x replicates) -> match fork anchors by state
  roots = np.load(SEALED / 'construction_roots.npz')['state'].astype(np.float32)
  S_fork = obs[tr, ii, :8].astype(np.float32)
  dmin = np.full(len(tr), np.inf)
  idx = np.nonzero(fork)[0]
  for k in range(0, len(idx), 200000):
    sl = idx[k:k + 200000]
    dd = np.linalg.norm(S_fork[sl][:, None, :] - roots[None], axis=2).min(1)
    dmin[sl] = dd
  at_ctx = fork & (dmin < 1e-4)

  res = {'A': {}, 'B': {}}
  L = ['# The fork target margin under the replay-E law, and the critics against it', '',
       'target = log P(goal cell (8,3) | fork anchor, recorded action in sector) for DOWN minus RIGHT, '
       'under the discounted relabeling law (gamma 0.95 unless stated), mixed over every source; '
       'jitter = goal frame with all four frames inside (8,3).', '']
  # ---- A ----
  L += ['## A. The law\'s own answer at the fork', '',
        '| anchors | law | sector | mass | P((8,3)) | P(jitter (8,3)) | P(absorbed-looking goal: 4 equal frames outside (8,3)) |',
        '|---|---|---|---:|---:|---:|---:|']
  g_frames = obs[tr, jj, :8]
  still = (np.linalg.norm(g_frames[:, :2] - g_frames[:, 6:8], axis=1) < 1e-6) & ~g83
  for aname, amask in (('all fork anchors', fork), ('550 training contexts', at_ctx)):
    for lname, gamma in (('gamma 0.95', 0.95), ('gamma 0.99', 0.99), ('undiscounted', 1.0)):
      w = law_weights(lengths, tr, ii, jj, gamma)
      p = {}
      for sname, smask in sec.items():
        m = amask & smask
        tot = w[m].sum()
        p[sname] = {'mass': float(tot), 'p83': float(w[m & g83].sum() / tot),
                    'p_jit': float(w[m & jitter83].sum() / tot), 'p_still': float(w[m & still].sum() / tot)}
        L.append(f'| {aname} | {lname} | {sname} | {tot:.5f} | {p[sname]["p83"]:.3f} | {p[sname]["p_jit"]:.3f} | {p[sname]["p_still"]:.3f} |')
      tgt = np.log(p['DOWN']['p83'] / p['RIGHT']['p83'])
      tgt_j = np.log(p['DOWN']['p_jit'] / p['RIGHT']['p_jit'])
      res['A'][f'{aname}|{lname}'] = {'per_sector': p, 'target_p83': float(tgt), 'target_jitter': float(tgt_j)}
      L.append(f'| {aname} | {lname} | **target DOWN-RIGHT** | | **{tgt:+.3f}** | **{tgt_j:+.3f}** | |')
  L += ['', '### By source (all fork anchors, gamma 0.95)', '',
        '| source | sector | share of the sector\'s fork mass | P((8,3)) | P(jitter (8,3)) |', '|---|---|---:|---:|---:|']
  w = law_weights(lengths, tr, ii, jj, 0.95)
  res['A']['by_source'] = {}
  for s, sname in SOURCE_NAMES.items():
    for secn in ('DOWN', 'RIGHT'):
      m = fork & sec[secn] & (src_row == s)
      tot = w[m].sum()
      share = tot / w[fork & sec[secn]].sum()
      p83 = w[m & g83].sum() / tot if tot > 0 else float('nan')
      pj = w[m & jitter83].sum() / tot if tot > 0 else float('nan')
      res['A']['by_source'][f'{sname}|{secn}'] = {'share': float(share), 'p83': float(p83), 'p_jit': float(pj)}
      L.append(f'| {sname} | {secn} | {share:.3f} | {p83:.3f} | {pj:.3f} |')
  # counterfactual mixtures: drop the original rows' fork anchors / keep only synthetic
  L += ['', '### What the target would be under other mixtures (all fork anchors, gamma 0.95)', '',
        '| mixture | P((8,3)|DOWN) | P((8,3)|RIGHT) | target |', '|---|---:|---:|---:|']
  res['A']['mixtures'] = {}
  for mname, keep in (('as trained (all sources)', np.ones(len(tr), bool)),
                      ('synthetic sources only (C + diag + E)', src_row != 0),
                      ('original only', src_row == 0),
                      ('C + E only (cardinal queries)', (src_row == 1) | (src_row == 3)),
                      ('E only', src_row == 3)):
    pd_ = w[fork & sec['DOWN'] & keep & g83].sum() / w[fork & sec['DOWN'] & keep].sum()
    pr_ = w[fork & sec['RIGHT'] & keep & g83].sum() / w[fork & sec['RIGHT'] & keep].sum()
    res['A']['mixtures'][mname] = {'p_down': float(pd_), 'p_right': float(pr_), 'target': float(np.log(pd_ / pr_))}
    L.append(f'| {mname} | {pd_:.3f} | {pr_:.3f} | {np.log(pd_ / pr_):+.3f} |')
  # ---- B ----
  eps = jnp.asarray(rng.standard_normal((256, 2)).astype(np.float32))

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

  def margins(qp, o):
    o = jnp.asarray(o)
    d_ = jnp.tile(jnp.array([[0.0, -1.0]], jnp.float32), (o.shape[0], 1))
    r_ = jnp.tile(jnp.array([[1.0, 0.0]], jnp.float32), (o.shape[0], 1))
    pm, wm = [], []
    for k in range(0, o.shape[0], 1024):
      pm.append(np.asarray(q_point(qp, o[k:k + 1024], d_[k:k + 1024]) - q_point(qp, o[k:k + 1024], r_[k:k + 1024])))
      wm.append(np.asarray(qbar(qp, o[k:k + 1024], jnp.asarray(DOWN_LOC)) - qbar(qp, o[k:k + 1024], jnp.asarray(RIGHT_LOC))))
    return float(np.concatenate(pm).mean()), float(np.concatenate(wm).mean())

  # rows: (i) 550-context states x canonical goal; (ii) the same states x jitter frames of the law;
  # (iii) buffer-law fork -> jitter-(8,3) rows
  ctx_states = roots
  jit_pool = obs[tr[fork & jitter83], jj[fork & jitter83], :8].astype(np.float32)
  jit_w = w[fork & jitter83]
  pick = rng.choice(len(jit_pool), size=min(300, len(jit_pool)), p=jit_w / jit_w.sum())
  jit_frames = jit_pool[pick]
  probes = {
      '198 context states x canonical': np.concatenate([ctx_states, np.broadcast_to(GOAL, ctx_states.shape)], 1),
      '198 context states x jitter frames': np.concatenate([np.repeat(ctx_states, 40, 0), np.tile(jit_frames[:40], (len(ctx_states), 1))], 1),
  }
  ww = np.where(fork & jitter83, w, 0.0)
  cdf = np.cumsum(ww / ww.sum())
  pos = np.minimum(np.searchsorted(cdf, rng.random(args.rows), side='right'), len(cdf) - 1)
  probes['buffer-law fork -> jitter (8,3) rows'] = np.concatenate(
      [obs[tr[pos], ii[pos], :8], obs[tr[pos], jj[pos], :8]], 1).astype(np.float32)
  L += ['', '## B. The critics on those rows: point margin / width-0.75 margin (DOWN - RIGHT)', '',
        '| critic | actor family | ' + ' | '.join(probes) + ' |', '|---|---|' + '---:|' * len(probes)]
  for cn, path in CRITICS.items():
    qp = checkpoint.load_checkpoint(path)[1].q_params
    cells, rec = [], {}
    for pn, o in probes.items():
      pm, wm = margins(qp, o.astype(np.float32))
      rec[pn] = {'point': pm, 'width': wm}
      cells.append(f'{pm:+.3f} / **{wm:+.3f}**')
    res['B'][cn] = rec
    L.append(f'| {cn} | {FAMILY[cn]} | ' + ' | '.join(cells) + ' |')
    print(L[-1], flush=True)
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (OUT / 'results.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  print('\n'.join(L))
  return 0


if __name__ == '__main__':
  sys.exit(main())
