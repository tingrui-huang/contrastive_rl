"""Which future supervision produced the critic's preference for the stalled action (AntMaze V6; user's plan after ccc42f6).

A  futures    the two learned futures tables (draw 1 / draw 2 of the S ETT; the sealed simulator table as a reference) compared on the
              reset / start anchors, per anchor: outcome, length, no-route timeout (the path never leaves the start region), the
              gamma-law masses (reach 0.5 / near 2.0 / goal area / far / death frame, as exp_v6_repeated_draws.path_masses) plus the
              STALL mass (rows with xy speed < 0.03) and the START-region mass; paired draw 1 - draw 2 per anchor.
B  qsurface   on the same 64 reset states, the twin-min critic logit along the action path a(l) = (1 - l) a_forward + l a_stall
              (a_forward = the progressing policy's mode, also the start agent's; a_stall = the stalled policy's mode) under the two
              runs' critics and several goal sets: the task goal, goals near it (disc of radius 2.0), the actor stream's relabelled
              goals of the anchor's own logged episode, and each table's own critic-training goal marginal (anchors by weight, the
              gamma law) -- where the bump Q(a_stall) > Q(a_forward) appears tells which goal distribution pulled the critic.

  python scripts/diag_v6_reset_futures.py futures
  python scripts/diag_v6_reset_futures.py qsurface --stall ... --fwd ... --critics d1=... d2=...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_repeated_draws as RD  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

OUT = MP.OUT / 'reset_futures'
STATE_DIM, OBS_W, GOAL_DIM = MP.STATE_DIM, MP.OBS_W, MP.GOAL_DIM
GAMMA = MP.GAMMA
SLOW = 0.03
TABLES = {'draw1': MP.OUT / 'ett_futures_v4s20' / 'branches_ett.npz', 'draw2': MP.OUT / 'ett_futures_v4s20_draw2' / 'branches_ett.npz', 'sim': MP.OUT / 'branches_cf.npz'}
LAMBDAS = np.linspace(0.0, 1.0, 11)
N_NEAR, N_MARG, N_REL, SEED = 32, 256, 16, 215_000_000


def groups_of(anchors):
  t = anchors.t.astype(np.int64); reg = region_of(anchors.state[:, :2].astype(np.float64))
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); g = S['group'].astype(str)
  reset64 = np.zeros(anchors.n, bool); reset64[S['anchor_id'][g == 'reset']] = True
  return {'reset64 (the diagnostics\' states)': reset64, 'reset (all t = 0)': t == 0, 'start_early (start region, t 1-40)': (reg == REGIONS.index('start')) & (t >= 1) & (t <= 40)}


def masses(xy, goal_xy, outcome):
  """path_masses + the stall mass (speed < 0.03) and the start-region mass, all under the gamma law over rows 1..L-1."""
  m = RD.path_masses(xy, goal_xy, outcome)
  L = len(xy); mm = np.arange(1, L); g = GAMMA ** mm; g /= g.sum()
  sp = np.linalg.norm(xy[1:] - xy[:-1], axis=1); reg = region_of(xy[1:].astype(np.float64))
  m['stall'] = float((g * (sp < SLOW)).sum()); m['start_region'] = float((g * (reg == REGIONS.index('start'))).sum())
  m['slow_rows_share'] = float((sp < SLOW).mean()); m['left_start'] = bool((reg != REGIONS.index('start')).any())
  return m


def mode_futures(args):
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  G = groups_of(anchors); keys = sorted(set(np.flatnonzero(np.any(np.stack(list(G.values())), axis=0)).tolist()))
  T = {name: MP.BranchFutures(anchors, p) for name, p in TABLES.items()}
  rows = {name: {} for name in T}
  t0 = time.time()
  for name, F in T.items():
    for k in keys:
      o, n = F.path(k); xy = o[:, :2].astype(np.float64); oc = str(F.outcome[k])
      m = masses(xy, anchors.goal_xy[k].astype(np.float64), oc)
      rows[name][k] = {'outcome': oc, 'length': int(F.lengths[k]), 'no_route_timeout': bool(oc == 'timeout' and not m['left_start']), **m}
  print(f'{len(keys)} anchors x {len(T)} tables in {time.time() - t0:.0f} s', flush=True)
  Q = ('length', 'reach0.5', 'near2.0', 'goal_area', 'far', 'death_frame', 'stall', 'start_region', 'slow_rows_share')
  res = {'tables': {k: str(v) for k, v in TABLES.items()}, 'groups': {}, 'per_anchor_reset64': {}}
  for gname, mask in G.items():
    ks = [k for k in keys if mask[k]]
    blk = {'n': len(ks), 'per_table': {}, 'paired_draw1_minus_draw2': {}}
    for name in T:
      R = [rows[name][k] for k in ks]
      blk['per_table'][name] = {'success': float(np.mean([r['outcome'] == 'success' for r in R])), 'death': float(np.mean([r['outcome'] == 'death' for r in R])), 'timeout': float(np.mean([r['outcome'] == 'timeout' for r in R])),
                                'no_route_timeout': float(np.mean([r['no_route_timeout'] for r in R])), 'never_left_start': float(np.mean([not r['left_start'] for r in R])),
                                **{q: {'mean': float(np.mean([r[q] for r in R])), 'median': float(np.median([r[q] for r in R])), 'p90': float(np.percentile([r[q] for r in R], 90))} for q in Q}}
    for q in Q:
      d = np.array([rows['draw1'][k][q] - rows['draw2'][k][q] for k in ks])
      blk['paired_draw1_minus_draw2'][q] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None, 'share_draw1_higher_by_0.1': float((d > 0.1).mean()), 'share_draw2_higher_by_0.1': float((d < -0.1).mean()), 'abs_mean': float(np.abs(d).mean())}
    blk['outcome_agreement_draw1_draw2'] = float(np.mean([rows['draw1'][k]['outcome'] == rows['draw2'][k]['outcome'] for k in ks]))
    blk['same_outcome_as_sim'] = {name: float(np.mean([rows[name][k]['outcome'] == rows['sim'][k]['outcome'] for k in ks])) for name in ('draw1', 'draw2')}
    res['groups'][gname] = blk
  r64 = [k for k in keys if G['reset64 (the diagnostics\' states)'][k]]
  res['per_anchor_reset64'] = {str(k): {name: {q: rows[name][k][q] for q in ('outcome', 'length', 'no_route_timeout', 'stall', 'start_region', 'near2.0', 'goal_area', 'reach0.5', 'death_frame')} for name in T} for k in r64}
  MP.write_json(OUT / 'futures_report.json', res)
  L = ['# The futures of the reset / start anchors: learned table draw 1 vs draw 2 (the S ETT; the same dataset and models, another generation seed), the sealed simulator table as reference', '',
       'Masses = the gamma-law (0.999) mass over the path rows 1..L-1: within 0.5 / 2.0 of the task goal, in the goal area, in the far regions, on the death frame, on STALL rows (xy speed < 0.03), in the START region.  No-route timeout = a timeout whose path never leaves the start region.', '']
  for gname, blk in res['groups'].items():
    L += [f"## {gname} (n {blk['n']}; outcome agreement draw 1 / draw 2 {blk['outcome_agreement_draw1_draw2']:.2f}; same outcome as the simulator table: draw 1 {blk['same_outcome_as_sim']['draw1']:.2f}, draw 2 {blk['same_outcome_as_sim']['draw2']:.2f})", '',
          '| table | success / death / timeout | no-route timeout | never left start | length mean / median / p90 | stall mass mean / median / p90 | start-region mass mean | reach0.5 / near2.0 / goal_area mass (mean) | far / death-frame mass | slow rows share |', '|---|---|---:|---:|---|---|---:|---|---|---:|']
    for name, v in blk['per_table'].items():
      L.append(f"| {name} | {v['success']:.3f} / {v['death']:.3f} / {v['timeout']:.3f} | {v['no_route_timeout']:.3f} | {v['never_left_start']:.3f} | {v['length']['mean']:.0f} / {v['length']['median']:.0f} / {v['length']['p90']:.0f} | {v['stall']['mean']:.3f} / {v['stall']['median']:.3f} / {v['stall']['p90']:.3f} | {v['start_region']['mean']:.3f} | "
               f"{v['reach0.5']['mean']:.4f} / {v['near2.0']['mean']:.4f} / {v['goal_area']['mean']:.4f} | {v['far']['mean']:.4f} / {v['death_frame']['mean']:.3f} | {v['slow_rows_share']['mean']:.3f} |")
    L += ['', '| paired draw 1 - draw 2 | mean +- se | draw 1 higher by > 0.1 | draw 2 higher by > 0.1 | mean abs diff |', '|---|---|---:|---:|---:|']
    for q, v in blk['paired_draw1_minus_draw2'].items():
      L.append(f"| {q} | {v['mean']:+.4f} +- {v['se'] if v['se'] is None else format(v['se'], '.4f')} | {v['share_draw1_higher_by_0.1']:.2f} | {v['share_draw2_higher_by_0.1']:.2f} | {v['abs_mean']:.4f} |")
    L.append('')
  # the reset64 anchors with the largest draw-1 excess of stall / start-region mass
  d = sorted(r64, key=lambda k: -(rows['draw1'][k]['stall'] - rows['draw2'][k]['stall']))
  L += ['## The 64 diagnostic reset anchors: the 12 with the largest draw-1 excess of stall mass (and the 12 largest draw-2 excess)', '', '| anchor | draw 1: outcome / length / stall / start-region / near2.0 / goal_area | draw 2: same | sim: same |', '|---:|---|---|---|']
  for k in d[:12] + d[-12:]:
    cells = [f"{rows[n][k]['outcome']} / {rows[n][k]['length']} / {rows[n][k]['stall']:.3f} / {rows[n][k]['start_region']:.3f} / {rows[n][k]['near2.0']:.3f} / {rows[n][k]['goal_area']:.3f}" for n in ('draw1', 'draw2', 'sim')]
    L.append(f'| {k} | ' + ' | '.join(cells) + ' |')
  (OUT / 'FUTURES.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


# ---------------------------------------------------------------- qsurface
def marginal_goals(anchors, F, rng, n):
  cdf = np.cumsum(anchors.weight / anchors.weight.sum()); cdf[-1] = 1.0
  ks = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'), anchors.n - 1); G = np.zeros((n, 2), np.float32)
  for i, k in enumerate(ks):
    nf = int(F.lengths[k] - 1); u = rng.random(); mm = int(np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** nf)) / np.log(GAMMA)), 1, max(nf, 1))); G[i] = F.goal_at(int(k), mm)
  return G


def mode_qsurface(args):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  OUT.mkdir(parents=True, exist_ok=True)
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); g = S['group'].astype(str); m = g == 'reset'
  st, gxy, ep, tt, a_log = S['state'][m], S['goal_xy'][m], S['episode'][m].astype(np.int64), S['t'][m].astype(np.int64), S['logged'][m]
  n = len(tt)
  crit = dict(kv.split('=', 1) for kv in args.critics); qp = {c: checkpoint.load_checkpoint(p)[1].q_params for c, p in crit.items()}
  pols = {}
  for lab, p in (('stall', args.stall), ('fwd', args.fwd), ('start', str(MP.START_CKPT))):
    pp = checkpoint.load_checkpoint(p)[1].policy_params
    pols[lab] = jax.jit(lambda o, pp=pp: jnp.tanh(nets.policy_network.apply(pp, o).loc))

  @jax.jit
  def q_diag(qparams, o31, a):
    return jnp.diag(jnp.min(nets.q_network.apply(qparams, o31, a), axis=-1))
  o31_task = np.concatenate([st, gxy], axis=1).astype(np.float32)
  A = {lab: np.asarray(pols[lab](jnp.asarray(o31_task))) for lab in pols}; A['logged'] = a_log.astype(np.float32)
  rng = np.random.default_rng(SEED)
  T = {'draw1': MP.BranchFutures(anchors, TABLES['draw1']), 'draw2': MP.BranchFutures(anchors, TABLES['draw2'])}
  marg = {name: marginal_goals(anchors, F, rng, N_MARG) for name, F in T.items()}
  goal_sets = {}
  for si in range(n):
    near = gxy[si] + rng.uniform(-2.0, 2.0, size=(N_NEAR * 3, 2)); near = near[np.linalg.norm(near - gxy[si], axis=1) <= 2.0][:N_NEAR]
    e, t = int(ep[si]), int(tt[si]); L = int(lengths[e]); nf = L - 1 - t; u = rng.random(N_REL)
    mm = np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** max(nf, 1))) / np.log(GAMMA)).astype(np.int64), 1, max(nf, 1))
    goal_sets[si] = {'task': gxy[si][None].astype(np.float32), 'near2.0': near.astype(np.float32), 'relabelled (actor stream)': obs[e, t + mm, :2].astype(np.float32),
                     'marginal draw1 (critic training)': marg['draw1'], 'marginal draw2 (critic training)': marg['draw2']}
  paths = {'fwd(d2 final) -> stall(d1 final)': ('fwd', 'stall'), 'start agent -> stall': ('start', 'stall'), 'logged torque -> stall': ('logged', 'stall')}
  res = {'critics': crit, 'policies': {'stall': args.stall, 'fwd': args.fwd, 'start': str(MP.START_CKPT)}, 'lambdas': LAMBDAS.tolist(), 'n_states': int(n),
         'action_gap_mean': {k: float(np.abs(A[a] - A[b]).mean()) for k, (a, b) in paths.items()}, 'curves': {}}
  for pname, (a0, a1) in paths.items():
    for c in crit:
      for gname in goal_sets[0]:
        curve = np.zeros((n, len(LAMBDAS)))
        for si in range(n):
          Gs = goal_sets[si][gname]; o31 = np.concatenate([np.repeat(st[si][None], len(Gs), axis=0), Gs], axis=1).astype(np.float32)
          for li, lam in enumerate(LAMBDAS):
            a = ((1 - lam) * A[a0][si] + lam * A[a1][si]).astype(np.float32)
            curve[si, li] = float(np.asarray(q_diag(qp[c], jnp.asarray(o31), jnp.asarray(np.repeat(a[None], len(Gs), axis=0)))).mean())
        bump = curve[:, -1] - curve[:, 0]; amax = LAMBDAS[np.argmax(curve, axis=1)]
        res['curves'][f'{pname}|{c}|{gname}'] = {'mean_curve': curve.mean(axis=0).tolist(), 'bump_mean': float(bump.mean()), 'bump_se': float(bump.std(ddof=1) / np.sqrt(n)), 'share_stall_higher': float((bump > 0).mean()),
                                                 'argmax_lambda_mean': float(amax.mean()), 'share_argmax_at_stall_end': float((amax >= 0.9).mean()), 'share_argmax_at_fwd_end': float((amax <= 0.1).mean()), 'share_interior_max': float(((amax > 0.1) & (amax < 0.9)).mean())}
        print(f'{pname} | {c} | {gname}: bump {bump.mean():+.3f} (stall higher in {(bump > 0).mean():.2f})', flush=True)
  MP.write_json(OUT / 'qsurface_report.json', res)
  L = ['# The critics\' twin-min logit along the action path forward -> stall at the 64 reset states, by goal set', '',
       f"Path a(l) = (1 - l) a_forward + l a_stall, l = 0 .. 1 in 11 steps; a_stall = the stalled policy's mode ({args.stall}), a_forward = the progressing policy's mode ({args.fwd}), the start agent's mode, or the logged first torque.  Goal sets per state: the task goal; {N_NEAR} goals within 2.0 of it; {N_REL} relabelled future goals of the anchor's logged episode (the actor stream's law); {N_MARG} goals from each table's critic-training marginal (anchors by weight, the gamma law).  Bump = Q(l = 1) - Q(l = 0) per state, mean over states; 'stall higher' = the share of states with a positive bump.", '',
       '| action path | critic | goal set | Q at l = 0 / 0.5 / 1 (mean) | bump mean +- se | stall higher (share) | argmax l mean | argmax at the stall end / forward end / interior |', '|---|---|---|---|---|---:|---:|---|']
  for k, v in res['curves'].items():
    p_, c, gname = k.split('|'); mc = v['mean_curve']
    L.append(f"| {p_} | {c} | {gname} | {mc[0]:.2f} / {mc[5]:.2f} / {mc[-1]:.2f} | {v['bump_mean']:+.3f} +- {v['bump_se']:.3f} | {v['share_stall_higher']:.2f} | {v['argmax_lambda_mean']:.2f} | {v['share_argmax_at_stall_end']:.2f} / {v['share_argmax_at_fwd_end']:.2f} / {v['share_interior_max']:.2f} |")
  L += ['', 'Mean action gaps: ' + ', '.join(f'{k}: {v:.3f}' for k, v in res['action_gap_mean'].items())]
  (OUT / 'QSURFACE.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


# --------------------------------------------------------------- decompose
def mode_decompose(args):
  """f(s, a, g) = phi(s, a) . psi(g) per twin head: |phi|, |psi|, cos(phi, psi) at the 64 reset states for the stalled / progressing / start /
  logged actions under each critic, with the task goal and the critic-training marginal goals -- does the stalled action's higher score come
  from a longer phi or from a better angle?"""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  OUT.mkdir(parents=True, exist_ok=True)
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); g = S['group'].astype(str); m = g == 'reset'
  st, gxy, a_log = S['state'][m], S['goal_xy'][m], S['logged'][m].astype(np.float32); n = len(st)
  crit = dict(kv.split('=', 1) for kv in args.critics); qp = {c: checkpoint.load_checkpoint(p)[1].q_params for c, p in crit.items()}
  reprs = jax.jit(lambda p_, o, a: nets.representation_network.apply(p_, o, a))
  pols = {}
  for lab, p_ in (('stall', args.stall), ('fwd', args.fwd), ('start', str(MP.START_CKPT))):
    pp = checkpoint.load_checkpoint(p_)[1].policy_params
    pols[lab] = jax.jit(lambda o, pp=pp: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  o31 = np.concatenate([st, gxy], axis=1).astype(np.float32)
  A = {lab: np.asarray(pols[lab](jnp.asarray(o31))) for lab in pols}; A['logged'] = a_log
  rng = np.random.default_rng(SEED + 1)
  T = {'draw1': MP.BranchFutures(anchors, TABLES['draw1']), 'draw2': MP.BranchFutures(anchors, TABLES['draw2'])}
  marg = marginal_goals(anchors, T['draw1'], rng, N_MARG)
  goal_sets = {'task': gxy.astype(np.float32), 'marginal draw1': marg}
  res = {'critics': crit, 'actions': list(A), 'per': {}}
  dummy = jnp.zeros((N_MARG, 8), jnp.float32)
  for c in crit:
    phi = {lab: np.asarray(reprs(qp[c], jnp.asarray(o31), jnp.asarray(A[lab]))[0]) for lab in A}          # [n, d, 2]
    for gname, G in goal_sets.items():
      og = np.concatenate([np.repeat(st[:1], len(G), axis=0), G], axis=1).astype(np.float32)             # psi ignores the state columns
      psi = np.asarray(reprs(qp[c], jnp.asarray(og), dummy[:len(G)])[1])                                   # [m, d, 2]
      for lab in A:
        ph = phi[lab]; nph = np.linalg.norm(ph, axis=1)                                                    # [n, 2]
        if gname == 'task':
          ps = psi; nps = np.linalg.norm(ps, axis=1); dots = np.einsum('idh,idh->ih', ph, ps); cos = dots / (nph * nps + 1e-8)
          f = np.min(dots, axis=1)
        else:
          nps = np.linalg.norm(psi, axis=1); dots = np.einsum('idh,jdh->ijh', ph, psi); cos = dots / (nph[:, None] * nps[None] + 1e-8)
          f = np.min(dots.mean(axis=1), axis=1); cos = cos.mean(axis=1)
        res['per'][f'{c}|{gname}|{lab}'] = {'phi_norm_mean': float(nph.mean()), 'psi_norm_mean': float(nps.mean()), 'cos_mean': float(cos.mean()), 'twin_min_logit_mean': float(f.mean())}
  # the stall - fwd contrast split: log f = log|phi| + log|psi| + log cos is not defined for negative cos; report the multiplicative ratios instead
  for c in crit:
    for gname in goal_sets:
      a, b = res['per'][f'{c}|{gname}|stall'], res['per'][f'{c}|{gname}|fwd']
      res['per'][f'{c}|{gname}|stall_vs_fwd'] = {'phi_norm_ratio': a['phi_norm_mean'] / b['phi_norm_mean'], 'cos_diff': a['cos_mean'] - b['cos_mean'], 'logit_diff': a['twin_min_logit_mean'] - b['twin_min_logit_mean']}
  MP.write_json(OUT / 'decompose_report.json', res)
  L = ['# The critic score at the 64 reset states split into |phi(s, a)|, |psi(g)| and cos(phi, psi) (mean over the twin heads and states)', '',
       '| critic | goal set | action | mean abs phi | mean abs psi | mean cos | twin-min logit |', '|---|---|---|---:|---:|---:|---:|']
  for k, v in res['per'].items():
    c, gname, lab = k.split('|')
    if lab == 'stall_vs_fwd':
      L.append(f"| {c} | {gname} | **stall vs fwd** | ratio {v['phi_norm_ratio']:.3f} | | diff {v['cos_diff']:+.3f} | diff {v['logit_diff']:+.3f} |")
    else:
      L.append(f"| {c} | {gname} | {lab} | {v['phi_norm_mean']:.3f} | {v['psi_norm_mean']:.3f} | {v['cos_mean']:+.3f} | {v['twin_min_logit_mean']:.3f} |")
  (OUT / 'DECOMPOSE.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('futures', 'qsurface', 'decompose'))
  ap.add_argument('--stall', default=None); ap.add_argument('--fwd', default=None)
  ap.add_argument('--critics', nargs='*', default=[])
  args = ap.parse_args(argv)
  {'futures': mode_futures, 'qsurface': mode_qsurface, 'decompose': mode_decompose}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
