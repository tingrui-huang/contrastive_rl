"""Two checks on existing results only (AntMaze V6; user's plan after 555f22c): no new trajectories, no training.

ruler     The critic and the consequences on the SAME ruler.  At the 64 reset states and the four first actions (the logged torque, the
          stalled seed-4 actor's mode, the progressing seed-4 actor's mode, the start agent's mode) the gamma-law mass of a goal REGION
          (within 0.5 / 2.0 of the task goal, the goal area, the far regions, the start region) as (i) the simulator gives it (paired
          hidden draws, the start agent's continuation), (ii) the S ETT gives it (same protocol), and (iii) the critic READS it.  The
          recipe's critic is a binary NCE with one positive and B - 1 in-batch negatives per row, so its logit estimates
          f(s, a, g) = log p_gamma(g | s, a) / p(g) - log(B - 1) with p(g) the critic-training marginal (anchors by weight, the gamma law
          on the table).  The mass of a region G under the critic is therefore (B - 1) E_{g ~ p}[exp f(s, a, g) 1{g in G}], estimated on
          a large sample of the marginal; reported CALIBRATED (with the (B - 1) factor; the total over all g should be 1) and
          SELF-NORMALISED (the region's share of the critic's total).  Per state, per action, per twin head and with the twin-min logit
          the actor sees.  "The score at the exact goal point" (QSURFACE) is not a mass; this is.

normrank  The ranking after representation normalisation, computed in the actor's order: per head cos(phi_h(s, a), psi_h(g)), THEN the
          min over the heads per goal, THEN the mean over the goal set (DECOMPOSE.md took the mean over goals before the min).  Per state,
          the stalled action vs the progressing / start / logged actions under the task goal, near goals, the actor stream's relabelled
          goals and the critic-training marginal: does the collapsed critic's stall preference weaken, does the recovering critic's
          walking preference survive?  The scale is controlled: the temperature tau* is matched so that the normalised score has the same
          within-row spread over sampled actions as the unnormalised logit (the same BC : critic balance as trained), and the objective
          AS TRAINED (E over 16 sampled actions with common random numbers + 0.05 NLL) on the 2,048 logged start rows is recomputed with
          the normalised score at tau* (and at tau* / 2, 2 tau*).  A post-hoc normalisation of a critic trained without it is a
          necessary-condition check for config.repr_norm, not its effect.

  python scripts/diag_v6_critic_ruler.py ruler --stall D1/seed_4/final.pkl --fwd D2/seed_4/final.pkl --critics d1_final=... d1_20k=... d2_final=... d2_20k=...
  python scripts/diag_v6_critic_ruler.py normrank --stall ... --fwd ... --actor20k D1/seed_4/20000.pkl --critics ...
"""
from __future__ import annotations

import argparse
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
import diag_v6_reset_futures as RF  # noqa: E402
import diag_v6_stall_onset as SO  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

OUT = MP.OUT / 'critic_ruler'
STATE_DIM, OBS_W, GOAL_DIM, GAMMA = MP.STATE_DIM, MP.OBS_W, MP.GOAL_DIM, MP.GAMMA
B_TRAIN = MP.BATCH                      # the critic's NCE batch: exp f = p(g | s, a) / p(g) / (B - 1)
N_MARG, SEED = 8192, 217_000_000
N_NEAR, N_REL, N_MARG_RANK = 32, 16, 512
N_SAMPLES = 16
REGION_KEYS = ('reach0.5', 'near2.0', 'goal_area', 'far', 'start_region')
CANDS = ('logged', 'stall', 'prog', 'start')
HEADS = ('h0', 'h1', 'min')


def _setup(args):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  OUT.mkdir(parents=True, exist_ok=True)
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  S = np.load(RD.OUT / 'states.npz', allow_pickle=False); g = S['group'].astype(str); idx = np.flatnonzero(g == 'reset')
  D = {'idx': idx, 'st': S['state'][idx], 'gxy': S['goal_xy'][idx], 'ep': S['episode'][idx].astype(np.int64), 't': S['t'][idx].astype(np.int64), 'a_log': S['logged'][idx].astype(np.float32)}
  D['o31'] = np.concatenate([D['st'], D['gxy']], axis=1).astype(np.float32)
  crit = dict(kv.split('=', 1) for kv in args.critics); qp = {c: checkpoint.load_checkpoint(p)[1].q_params for c, p in crit.items()}
  pols = {}
  for lab, p in (('stall', args.stall), ('fwd', args.fwd), ('start', str(MP.START_CKPT))):
    pp = checkpoint.load_checkpoint(p)[1].policy_params
    pols[lab] = jax.jit(lambda o, pp=pp: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  A = {'logged': D['a_log'], 'stall': np.asarray(pols['stall'](jnp.asarray(D['o31']))), 'prog': np.asarray(pols['fwd'](jnp.asarray(D['o31']))), 'start': np.asarray(pols['start'](jnp.asarray(D['o31'])))}
  reprs = jax.jit(lambda p_, o, a: nets.representation_network.apply(p_, o, a))
  return cfg, nets, D, crit, qp, A, reprs


def _psi(reprs, qp, st0, G):
  """psi(g) for goals G [m, 2]: the goal encoder reads the goal columns only (the state columns are a dummy)."""
  import jax.numpy as jnp
  og = np.concatenate([np.repeat(st0[None], len(G), axis=0), G], axis=1).astype(np.float32)
  return np.asarray(reprs(qp, jnp.asarray(og), jnp.zeros((len(G), 8), jnp.float32))[1])                 # [m, d, 2]


def region_masks(G, goal_xy):
  d = np.linalg.norm(G - goal_xy[None], axis=1); reg = region_of(G.astype(np.float64))
  return {'reach0.5': d <= RD.REACH_R, 'near2.0': d <= 2.0, 'goal_area': reg == REGIONS.index('goal_area'), 'far': np.isin(reg, FAR), 'start_region': reg == REGIONS.index('start')}


def _agg(R, keys, cont_filter=None):
  """Per (candidate, state): the mean over the paired draws of every key in `keys` that the file holds; {cand: {state: {key: mean}}}, draws per candidate."""
  cand = R['cand'].astype(str); st = R['state'].astype(np.int64); m = np.ones(len(cand), bool)
  if cont_filter is not None and 'cont' in R:
    m = R['cont'].astype(str) == cont_filter
  keys = [k for k in keys if k in R]; out = {}; n_draw = {}
  for c in sorted(set(cand[m])):
    out[c] = {}
    for s in sorted(set(st[m & (cand == c)])):
      sel = m & (cand == c) & (st == s); out[c][int(s)] = {k: float(np.asarray(R[k], np.float64)[sel].mean()) for k in keys}; n_draw[c] = int(sel.sum())
  return out, n_draw


def _se(x):
  x = np.asarray(x, np.float64); return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')


# --------------------------------------------------------------------- ruler
def sim_start_region_mass(R):
  """The start-region gamma-law mass of every simulator rollout row from its stored xy path (the same law as path_masses)."""
  if 'xy_rows' not in R.files:
    return None
  xy_all, off, L = R['xy_rows'], R['offset'].astype(np.int64), R['length'].astype(np.int64); out = np.zeros(len(off)); chk = np.zeros(len(off))
  gxy = R['goal_xy'] if 'goal_xy' in R.files else None
  for r in range(len(off)):
    xy = xy_all[off[r]:off[r] + L[r]]; n = len(xy); m = np.arange(1, n); g = GAMMA ** m; g /= g.sum(); reg = region_of(xy[1:].astype(np.float64))
    out[r] = float((g * (reg == REGIONS.index('start'))).sum()); chk[r] = float((g * (reg == REGIONS.index('goal_area'))).sum())
  return out, chk


def mode_ruler(args):
  import jax.numpy as jnp
  cfg, nets, D, crit, qp, A, reprs = _setup(args); n = len(D['idx'])
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  rng = np.random.default_rng(SEED)
  T = {'draw1': MP.BranchFutures(anchors, RF.TABLES['draw1']), 'draw2': MP.BranchFutures(anchors, RF.TABLES['draw2'])}
  marg = {name: RF.marginal_goals(anchors, F, rng, N_MARG) for name, F in T.items()}
  masks = {name: [region_masks(G, D['gxy'][si]) for si in range(n)] for name, G in marg.items()}                  # per marginal, per state
  p_marg = {name: {k: float(np.mean([masks[name][si][k].mean() for si in range(n)])) for k in REGION_KEYS} for name in marg}
  # ---- the simulator and the model (existing rollouts; per (candidate, state) means over the paired draws)
  Rs = np.load(MP.OUT / args.sim, allow_pickle=False)
  sim_arrays = {k: Rs[k] for k in Rs.files if k not in ('xy_rows', 'meta', 'offset', 'length')}
  sr = sim_start_region_mass(Rs)
  if sr is not None:
    assert np.allclose(sr[1], Rs['goal_area'], atol=1e-6), 'xy_rows law mismatch'
    sim_arrays['start_region'] = sr[0]
  Ps, nds = _agg(sim_arrays, REGION_KEYS); nds = {str(k): v for k, v in nds.items()}
  Rm = np.load(MP.OUT / args.model, allow_pickle=False)
  Pm, ndm = _agg({k: Rm[k] for k in Rm.files if k not in ('meta', 'early', 'early_state', 'early_cand', 'early_cont', 'a0')}, REGION_KEYS, cont_filter='start'); ndm = {str(k): v for k, v in ndm.items()}
  # per_state keys the states by the absolute row of states.npz; align to the 64 reset rows
  real = {c: {k: np.array([Ps[c][int(D['idx'][si])][k] for si in range(n)]) for k in Ps[c][int(D['idx'][0])]} for c in CANDS}
  ett = {c: {k: np.array([Pm[c][int(D['idx'][si])][k] for si in range(n)]) for k in Pm[c][int(D['idx'][0])]} for c in CANDS}
  # ---- the critic's readout
  read = {}                                                                                   # read[c][marg][cand][head][key] -> [n]
  t0 = time.time()
  for c in crit:
    read[c] = {}
    phi = {lab: np.asarray(reprs(qp[c], jnp.asarray(D['o31']), jnp.asarray(A[lab]))[0]).astype(np.float64) for lab in CANDS}   # [n, d, 2]
    for mname, G in marg.items():
      psi = _psi(reprs, qp[c], D['st'][0], G).astype(np.float64)                                                                # [M, d, 2]
      Mk = {k: np.stack([masks[mname][si][k] for si in range(n)]) for k in REGION_KEYS}                                          # [n, M]
      read[c][mname] = {}
      for lab in CANDS:
        f = np.einsum('idh,jdh->ijh', phi[lab], psi)                                                                          # [n, M, 2]
        blk = {}
        for hname, fh in (('h0', f[..., 0]), ('h1', f[..., 1]), ('min', f.min(axis=-1))):
          w = np.exp(fh) * (B_TRAIN - 1); total = w.mean(axis=1)
          r = {'total': total, 'w_max_share': w.max(axis=1) / w.sum(axis=1)}
          for k in REGION_KEYS:
            r[k] = (w * Mk[k]).mean(axis=1); r[k + '_share'] = r[k] / total; r[k + '_ratio'] = r[k] / np.maximum(Mk[k].mean(axis=1), 1e-9)
          # a half-split of the marginal sample as the Monte-Carlo error of the region masses
          h1, h2 = slice(0, N_MARG // 2), slice(N_MARG // 2, N_MARG)
          r['goal_area_half_diff'] = np.abs((w[:, h1] * Mk['goal_area'][:, h1]).mean(axis=1) - (w[:, h2] * Mk['goal_area'][:, h2]).mean(axis=1))
          blk[hname] = r
        read[c][mname][lab] = blk
    print(f'{c}: readout done ({time.time() - t0:.0f} s)', flush=True)
  own = {c: ('draw2' if c.startswith('d2') else 'draw1') for c in crit}
  # ---- summaries
  res = {'critics': crit, 'own_marginal': own, 'sim': args.sim, 'model': args.model, 'draws_per_state': {'sim': nds, 'model': ndm}, 'n_states': int(n), 'n_marginal': N_MARG, 'B_train': B_TRAIN,
         'marginal_region_mass': p_marg, 'per_action': {}, 'paired': {}, 'agreement': {}, 'overestimation': {}}
  for k in REGION_KEYS:
    for c in CANDS:
      row = {'real': float(np.nanmean(real[c][k])) if k in real[c] else None, 'ett': float(np.nanmean(ett[c][k])) if k in ett[c] else None}
      for cn in crit:
        rr = read[cn][own[cn]][c]
        row[cn] = {h: {'mass': float(rr[h][k].mean()), 'share': float(rr[h][k + '_share'].mean()), 'ratio': float(rr[h][k + '_ratio'].mean()), 'total': float(rr[h]['total'].mean())} for h in HEADS}
        row[cn]['min_other_marginal'] = float(read[cn]['draw2' if own[cn] == 'draw1' else 'draw1'][c]['min'][k].mean())
      res['per_action'][f'{k}|{c}'] = row
  for ref in ('logged', 'prog'):
    for k in REGION_KEYS:
      blk = {}
      if k in real['stall']:
        d = real['stall'][k] - real[ref][k]; blk['real'] = {'mean': float(np.nanmean(d)), 'se': _se(d[np.isfinite(d)]), 'share_pos': float(np.nanmean(d > 0))}
      if k in ett['stall']:
        d = ett['stall'][k] - ett[ref][k]; blk['ett'] = {'mean': float(np.nanmean(d)), 'se': _se(d[np.isfinite(d)]), 'share_pos': float(np.nanmean(d > 0))}
      for cn in crit:
        rr = read[cn][own[cn]]
        blk[cn] = {h: {'mass': {'mean': float((rr['stall'][h][k] - rr[ref][h][k]).mean()), 'se': _se(rr['stall'][h][k] - rr[ref][h][k]), 'share_pos': float(((rr['stall'][h][k] - rr[ref][h][k]) > 0).mean())},
                       'share': {'mean': float((rr['stall'][h][k + '_share'] - rr[ref][h][k + '_share']).mean()), 'share_pos': float(((rr['stall'][h][k + '_share'] - rr[ref][h][k + '_share']) > 0).mean())}} for h in HEADS}
      res['paired'][f'stall_minus_{ref}|{k}'] = blk
  # agreement across the 256 (state, action) pairs and across the 64 per-state contrasts
  for k in REGION_KEYS:
    if k not in real['logged']:
      continue
    xr = np.concatenate([real[c][k] for c in CANDS]); ok = np.isfinite(xr)
    blk = {}
    if k in ett['logged']:
      xe = np.concatenate([ett[c][k] for c in CANDS]); o2 = ok & np.isfinite(xe)
      blk['ett_vs_real'] = {'pearson': float(np.corrcoef(xe[o2], xr[o2])[0, 1]), 'spearman': _spearman(xe[o2], xr[o2])}
    for cn in crit:
      for h in ('min', 'h0'):
        xc = np.concatenate([read[cn][own[cn]][c][h][k] for c in CANDS])
        dr = real['stall'][k] - real['logged'][k]; dc = read[cn][own[cn]]['stall'][h][k] - read[cn][own[cn]]['logged'][h][k]; o3 = np.isfinite(dr)
        blk[f'{cn}|{h}'] = {'pearson_pairs': float(np.corrcoef(xc[ok], xr[ok])[0, 1]), 'spearman_pairs': _spearman(xc[ok], xr[ok]), 'spearman_within_state': _within_state_spearman(xc, xr, n),
                            'contrast_stall_minus_logged_pearson': float(np.corrcoef(dc[o3], dr[o3])[0, 1]) if dr[o3].std() > 0 else None, 'contrast_sign_agreement': float(np.mean(np.sign(dc[o3]) == np.sign(dr[o3])))}
    res['agreement'][k] = blk
  for k in ('goal_area', 'near2.0', 'reach0.5'):
    for cn in crit:
      for c in CANDS:
        q = read[cn][own[cn]][c]['min'][k] / np.maximum(real[c][k], 1e-4); q = q[np.isfinite(q)]
        res['overestimation'][f'{k}|{cn}|{c}'] = {'median': float(np.median(q)), 'q25': float(np.percentile(q, 25)), 'q75': float(np.percentile(q, 75)), 'share_gt_2': float((q > 2).mean()), 'share_lt_0.5': float((q < 0.5).mean()),
                                                 'mc_half_diff_over_mass': float(np.median(read[cn][own[cn]][c]['min']['goal_area_half_diff'] / np.maximum(read[cn][own[cn]][c]['min']['goal_area'], 1e-9)))}
  MP.write_json(OUT / 'ruler_report.json', res)
  np.savez_compressed(OUT / 'ruler_per_state.npz', idx=D['idx'], **{f'real|{c}|{k}': real[c][k] for c in CANDS for k in real[c]}, **{f'ett|{c}|{k}': ett[c][k] for c in CANDS for k in ett[c]},
                      **{f'critic|{cn}|{c}|{h}|{k}': read[cn][own[cn]][c][h][k] for cn in crit for c in CANDS for h in HEADS for k in list(REGION_KEYS) + ['total']})
  # ---- report
  L = ['# The critic and the consequences on the same ruler: gamma-law region masses at the 64 reset states, per first action', '',
       f"Real = the simulator (paired hidden draws, the start agent's continuation; {nds} draws per state); ETT = the S ETT with the start continuation ({ndm} draws per state); critic = (B - 1) E_g[exp f(s, a, g) 1{{g in G}}] over {N_MARG} goals of the critic's own training marginal (B = {B_TRAIN}; 'total' = the same over all g, 1 for a calibrated critic; 'share' = mass / total).  Regions: within 0.5 / 2.0 of the task goal, the goal area, the far regions, the start region.  The marginal's own region masses: " + ', '.join(f"{k} {p_marg['draw1'][k]:.4f}" for k in REGION_KEYS) + ' (draw 1).', '']
  for k in REGION_KEYS:
    L += [f'## {k}', '', '| first action | real mass | ETT mass | ' + ' | '.join(f'{cn}: min-head mass / share / total' for cn in crit) + ' |', '|---|---:|---:|' + '|'.join('---' for _ in crit) + '|']
    for c in CANDS:
      row = res['per_action'][f'{k}|{c}']
      cells = [f"{row[cn]['min']['mass']:.4f} / {row[cn]['min']['share']:.4f} / {row[cn]['min']['total']:.2f}" for cn in crit]
      L.append(f"| {c} | {'-' if row['real'] is None else format(row['real'], '.4f')} | {'-' if row['ett'] is None else format(row['ett'], '.4f')} | " + ' | '.join(cells) + ' |')
    L += ['', '| paired contrast | real | ETT | ' + ' | '.join(f'{cn}: min-head mass (share pos) / share' for cn in crit) + ' |', '|---|---|---|' + '|'.join('---' for _ in crit) + '|']
    for ref in ('logged', 'prog'):
      blk = res['paired'][f'stall_minus_{ref}|{k}']
      cells = [f"{blk[cn]['min']['mass']['mean']:+.4f} +- {blk[cn]['min']['mass']['se']:.4f} ({blk[cn]['min']['mass']['share_pos']:.2f}) / {blk[cn]['min']['share']['mean']:+.4f}" for cn in crit]
      L.append(f"| stall - {ref} | " + ' | '.join(f"{blk[x]['mean']:+.4f} +- {blk[x]['se']:.4f} ({blk[x]['share_pos']:.2f})" if x in blk else '-' for x in ('real', 'ett')) + ' | ' + ' | '.join(cells) + ' |')
    L.append('')
  L += ['## Per-head totals (calibration; mean over states)', '', '| critic | action | h0 total | h1 total | min total | max-weight share (min) |', '|---|---|---:|---:|---:|---:|']
  for cn in crit:
    for c in CANDS:
      rr = read[cn][own[cn]][c]; L.append(f"| {cn} | {c} | {rr['h0']['total'].mean():.2f} | {rr['h1']['total'].mean():.2f} | {rr['min']['total'].mean():.2f} | {rr['min']['w_max_share'].mean():.3f} |")
  L += ['', '## Agreement with the real masses (256 (state, action) pairs; the per-state stall - logged contrast)', '', '| region | reader | Pearson (pairs) | Spearman (pairs) | Spearman within state (mean over states, 4 actions) | contrast Pearson | contrast sign agreement |', '|---|---|---:|---:|---:|---:|---:|']
  for k, blk in res['agreement'].items():
    for rd, v in blk.items():
      if rd == 'ett_vs_real':
        L.append(f"| {k} | ETT | {v['pearson']:.2f} | {v['spearman']:.2f} | | | |")
      else:
        L.append(f"| {k} | {rd} | {v['pearson_pairs']:.2f} | {v['spearman_pairs']:.2f} | {v['spearman_within_state']:.2f} | {'-' if v['contrast_stall_minus_logged_pearson'] is None else format(v['contrast_stall_minus_logged_pearson'], '.2f')} | {v['contrast_sign_agreement']:.2f} |")
  L += ['', '## Over-estimation factor: critic (min-head) mass / real mass per (state, action)', '', '| region | critic | action | median | IQR | share > 2x | share < 0.5x | MC half-split error / mass (median) |', '|---|---|---|---:|---|---:|---:|---:|']
  for key, v in res['overestimation'].items():
    k, cn, c = key.split('|'); L.append(f"| {k} | {cn} | {c} | {v['median']:.2f} | {v['q25']:.2f} - {v['q75']:.2f} | {v['share_gt_2']:.2f} | {v['share_lt_0.5']:.2f} | {v['mc_half_diff_over_mass']:.2f} |")
  (OUT / 'RULER.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


def _spearman(x, y):
  from scipy.stats import spearmanr
  return float(spearmanr(x, y).correlation)


def _within_state_spearman(xc, xr, n):
  from scipy.stats import spearmanr
  xc = xc.reshape(len(CANDS), n); xr = xr.reshape(len(CANDS), n); v = []
  for si in range(n):
    if np.isfinite(xr[:, si]).all() and xr[:, si].std() > 0 and xc[:, si].std() > 0:
      v.append(spearmanr(xc[:, si], xr[:, si]).correlation)
  return float(np.mean(v)) if v else float('nan')


# ------------------------------------------------------------------ normrank
def _scores_np(phi, psi):
  """phi [n, d, 2], psi [m, d, 2] -> unnormalised twin-min f [n, m] and twin-min cos [n, m] (min over heads per goal)."""
  dots = np.einsum('idh,jdh->ijh', phi, psi); nrm = np.linalg.norm(phi, axis=1)[:, None, :] * np.linalg.norm(psi, axis=1)[None, :, :]
  return dots.min(axis=-1), (dots / (nrm + 1e-8)).min(axis=-1)


def mode_normrank(args):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg, nets, D, crit, qp, A, reprs = _setup(args); n = len(D['idx'])
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  rng = np.random.default_rng(SEED + 7)
  T = {'draw1': MP.BranchFutures(anchors, RF.TABLES['draw1']), 'draw2': MP.BranchFutures(anchors, RF.TABLES['draw2'])}
  marg = {name: RF.marginal_goals(anchors, F, rng, N_MARG_RANK) for name, F in T.items()}
  goal_sets = {}
  for si in range(n):
    near = D['gxy'][si] + rng.uniform(-2.0, 2.0, size=(N_NEAR * 3, 2)); near = near[np.linalg.norm(near - D['gxy'][si], axis=1) <= 2.0][:N_NEAR]
    e, t = int(D['ep'][si]), int(D['t'][si]); Lr = int(lengths[e]); nf = Lr - 1 - t; u = rng.random(N_REL)
    mm = np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** max(nf, 1))) / np.log(GAMMA)).astype(np.int64), 1, max(nf, 1))
    goal_sets[si] = {'task': D['gxy'][si][None].astype(np.float32), 'near2.0': near.astype(np.float32), 'relabelled': obs[e, t + mm, :2].astype(np.float32), 'marginal draw1': marg['draw1'], 'marginal draw2': marg['draw2']}
  GS = list(goal_sets[0])
  # ---- A. the 64 reset states, the four mode actions: per goal set the mean over goals of the per-goal twin-min score
  fu = {c: {lab: {gs: np.zeros(n) for gs in GS} for lab in CANDS} for c in crit}; cn_ = {c: {lab: {gs: np.zeros(n) for gs in GS} for lab in CANDS} for c in crit}
  phinorm = {c: {} for c in crit}
  for c in crit:
    phi = {lab: np.asarray(reprs(qp[c], jnp.asarray(D['o31']), jnp.asarray(A[lab]))[0]).astype(np.float64) for lab in CANDS}
    phinorm[c] = {lab: np.linalg.norm(phi[lab], axis=1) for lab in CANDS}                                                        # [n, 2]
    psi_m = {gs: _psi(reprs, qp[c], D['st'][0], goal_sets[0][gs]).astype(np.float64) for gs in ('marginal draw1', 'marginal draw2')}
    for si in range(n):
      for gs in GS:
        psi = psi_m[gs] if gs in psi_m else _psi(reprs, qp[c], D['st'][0], goal_sets[si][gs]).astype(np.float64)
        for lab in CANDS:
          f, cs = _scores_np(phi[lab][si:si + 1], psi); fu[c][lab][gs][si] = f.mean(); cn_[c][lab][gs][si] = cs.mean()
  res = {'critics': crit, 'n_states': int(n), 'goal_sets': {gs: int(len(goal_sets[0][gs])) for gs in GS}, 'modes': {}, 'phi_norm': {}, 'objective': {}}
  for c in crit:
    res['phi_norm'][c] = {lab: {'h0': float(phinorm[c][lab][:, 0].mean()), 'h1': float(phinorm[c][lab][:, 1].mean())} for lab in CANDS}
    for gs in GS:
      for ref in ('prog', 'start', 'logged'):
        du = fu[c]['stall'][gs] - fu[c][ref][gs]; dn = cn_[c]['stall'][gs] - cn_[c][ref][gs]
        res['modes'][f'{c}|{gs}|stall_minus_{ref}'] = {'unnorm_nats': {'mean': float(du.mean()), 'se': _se(du), 'share_stall_pref': float((du > 0).mean())},
                                                       'norm_cos': {'mean': float(dn.mean()), 'se': _se(dn), 'share_stall_pref': float((dn > 0).mean())},
                                                       'share_flipped': float(np.mean(np.sign(du) != np.sign(dn))), 'levels_unnorm': [float(fu[c]['stall'][gs].mean()), float(fu[c][ref][gs].mean())], 'levels_norm': [float(cn_[c]['stall'][gs].mean()), float(cn_[c][ref][gs].mean())]}
  # ---- B. the scale-matched temperature and the objective as trained on the 2,048 logged start rows
  rows, n_reset = SO.logged_start_rows(obs, act, lengths)
  e, t = rows[:, 0], rows[:, 1]; s = obs[e, t, :STATE_DIM].astype(np.float32); a_log = act[e, t].astype(np.float32)
  rr = np.random.default_rng(SO.SEED_ROWS + 1)
  nf = np.maximum(lengths[e] - 1 - t, 1); u = rr.random(len(rows)); m = np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** nf)) / np.log(GAMMA)).astype(np.int64), 1, nf)
  g = obs[e, t + m, :2].astype(np.float32); o31 = jnp.asarray(np.concatenate([s, g], axis=1))
  keys = jax.random.split(jax.random.PRNGKey(SO.SEED_ROWS + 2), N_SAMPLES)
  pol = {'actor20k': checkpoint.load_checkpoint(args.actor20k)[1].policy_params, 'stall': checkpoint.load_checkpoint(args.stall)[1].policy_params, 'prog': checkpoint.load_checkpoint(args.fwd)[1].policy_params}

  @jax.jit
  def samples(pp, qparams, o, k):
    d = nets.policy_network.apply(pp, o)
    def one(kk):
      a = nets.sample(d, kk); phi, psi = nets.representation_network.apply(qparams, o, a)
      dots = jnp.einsum('bdh,bdh->bh', phi, psi); cos = dots / (jnp.linalg.norm(phi, axis=1) * jnp.linalg.norm(psi, axis=1) + 1e-8)
      return jnp.min(dots, axis=-1), jnp.min(cos, axis=-1)
    return jax.vmap(one)(k)                                                                    # ([K, B], [K, B])

  @jax.jit
  def nll(pp, o, a):
    return -nets.log_prob(nets.policy_network.apply(pp, o), a)
  bc = float(cfg.bc_coef); NLL = {p: np.asarray(nll(pol[p], o31, jnp.asarray(a_log))) for p in pol}
  is_reset = np.arange(len(rows)) < n_reset
  for c in crit:
    F = {}; C = {}
    for p in pol:
      f, cs = samples(pol[p], qp[c], o31, keys); F[p] = np.asarray(f, np.float64); C[p] = np.asarray(cs, np.float64)                  # [K, B]
    # within-row spread over the pooled 48 sampled actions of the three policies
    Fp = np.concatenate([F[p] for p in pol]); Cp = np.concatenate([C[p] for p in pol])
    var_f = Fp.var(axis=0, ddof=1).mean(); var_c = Cp.var(axis=0, ddof=1).mean(); tau = float(np.sqrt(var_c / var_f))
    blk = {'tau_star': tau, 'within_row_std_unnorm': float(np.sqrt(var_f)), 'within_row_std_cos': float(np.sqrt(var_c)), 'per_policy': {}, 'stall_minus_prog': {}}
    for p in pol:
      r = {'bc_nll': float(NLL[p].mean()), 'EQ_unnorm': float(F[p].mean()), 'loss_unnorm': float((1 - bc) * (-F[p].mean()) + bc * NLL[p].mean()), 'Ecos': float(C[p].mean())}
      for tname, tv in (('tau*', tau), ('tau*/2', tau / 2), ('2tau*', 2 * tau)):
        q = C[p].mean() / tv; r[f'EQ_norm@{tname}'] = float(q); r[f'loss_norm@{tname}'] = float((1 - bc) * (-q) + bc * NLL[p].mean())
      blk['per_policy'][p] = r
    du = F['stall'].mean(axis=0) - F['prog'].mean(axis=0); dn = (C['stall'].mean(axis=0) - C['prog'].mean(axis=0)) / tau
    for lab, msk in (('all', np.ones(len(rows), bool)), ('reset', is_reset)):
      blk['stall_minus_prog'][lab] = {'unnorm': {'mean': float(du[msk].mean()), 'share_stall_pref': float((du[msk] > 0).mean())}, 'norm@tau*': {'mean': float(dn[msk].mean()), 'share_stall_pref': float((dn[msk] > 0).mean())},
                                      'loss_diff_unnorm': float((1 - bc) * (-du[msk].mean()) + bc * (NLL['stall'][msk].mean() - NLL['prog'][msk].mean())),
                                      'loss_diff_norm@tau*': float((1 - bc) * (-dn[msk].mean()) + bc * (NLL['stall'][msk].mean() - NLL['prog'][msk].mean())),
                                      'share_rows_flipped': float(np.mean(np.sign(du[msk]) != np.sign(dn[msk])))}
    res['objective'][c] = blk
    print(f'{c}: tau* {tau:.4f}; stall - prog unnorm {du.mean():+.3f} ({(du > 0).mean():.2f}) -> norm@tau* {dn.mean():+.3f} ({(dn > 0).mean():.2f})', flush=True)
  MP.write_json(OUT / 'normrank_report.json', res)
  # ---- report
  L = ['# The ranking after representation normalisation (per head cos, THEN min over heads per goal, THEN mean over goals), the scale matched', '',
       f"A. The 64 reset states, the mode actions (stalled seed-4 actor / progressing seed-4 actor / start agent / logged torque), goal sets per state: the task goal, {N_NEAR} goals within 2.0, {N_REL} relabelled goals of the anchor's logged episode (the actor stream's law), {N_MARG_RANK} goals of each table's critic-training marginal.  Unnorm = the twin-min logit (nats); norm = the twin-min cosine (per head cos, min over heads).  'stall pref' = the share of states where the stalled action scores higher; 'flipped' = the share of states whose sign changes under normalisation.", '',
       '| critic | goal set | vs | unnorm: stall - other (nats) +- se (stall pref) | norm: stall - other (cos) +- se (stall pref) | flipped | levels unnorm stall / other | levels cos stall / other |', '|---|---|---|---|---|---:|---|---|']
  for key, v in res['modes'].items():
    c, gs, ref = key.split('|'); L.append(f"| {c} | {gs} | {ref.replace('stall_minus_', '')} | {v['unnorm_nats']['mean']:+.3f} +- {v['unnorm_nats']['se']:.3f} ({v['unnorm_nats']['share_stall_pref']:.2f}) | {v['norm_cos']['mean']:+.4f} +- {v['norm_cos']['se']:.4f} ({v['norm_cos']['share_stall_pref']:.2f}) | {v['share_flipped']:.2f} | {v['levels_unnorm'][0]:.2f} / {v['levels_unnorm'][1]:.2f} | {v['levels_norm'][0]:+.3f} / {v['levels_norm'][1]:+.3f} |")
  L += ['', '| critic | action | abs phi h0 | abs phi h1 |', '|---|---|---:|---:|']
  for c in crit:
    for lab in CANDS:
      L.append(f"| {c} | {lab} | {res['phi_norm'][c][lab]['h0']:.3f} | {res['phi_norm'][c][lab]['h1']:.3f} |")
  L += ['', f"B. The objective as trained on {len(rows)} valid logged start rows ({n_reset} reset rows; relabelled goals; {N_SAMPLES} sampled actions per row per policy, common random numbers): loss = {1 - bc:.2f} (-E Q) + {bc:.2f} NLL.  tau* per critic = sqrt(within-row variance of the twin-min cosine / within-row variance of the twin-min logit) over the 48 pooled sampled actions of the three policies -- at tau* the normalised critic term has the SAME action spread as the unnormalised one (the same BC : critic balance); tau* / 2 doubles the critic term, 2 tau* halves it.", '',
        '| critic | tau* | policy | BC NLL | E Q unnorm / loss | E cos | loss norm @ tau* | @ tau*/2 | @ 2tau* |', '|---|---:|---|---:|---|---:|---:|---:|---:|']
  for c, blk in res['objective'].items():
    for p, r in blk['per_policy'].items():
      L.append(f"| {c} | {blk['tau_star']:.4f} | {p} | {r['bc_nll']:.2f} | {r['EQ_unnorm']:.2f} / {r['loss_unnorm']:.2f} | {r['Ecos']:+.4f} | {r['loss_norm@tau*']:.2f} | {r['loss_norm@tau*/2']:.2f} | {r['loss_norm@2tau*']:.2f} |")
  L += ['', '| critic | rows | E Q(stall) - E Q(prog) unnorm (stall pref) | norm @ tau* (stall pref) | rows flipped | loss(stall) - loss(prog) unnorm | norm @ tau* |', '|---|---|---|---|---:|---:|---:|']
  for c, blk in res['objective'].items():
    for lab, v in blk['stall_minus_prog'].items():
      L.append(f"| {c} | {lab} | {v['unnorm']['mean']:+.3f} ({v['unnorm']['share_stall_pref']:.2f}) | {v['norm@tau*']['mean']:+.3f} ({v['norm@tau*']['share_stall_pref']:.2f}) | {v['share_rows_flipped']:.2f} | {v['loss_diff_unnorm']:+.3f} | {v['loss_diff_norm@tau*']:+.3f} |")
  (OUT / 'NORMRANK.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('ruler', 'normrank'))
  ap.add_argument('--stall', required=True); ap.add_argument('--fwd', required=True); ap.add_argument('--actor20k', default=None)
  ap.add_argument('--critics', nargs='+', required=True)
  ap.add_argument('--sim', default='repeated_draws/rollouts_start_reset_stall.npz'); ap.add_argument('--model', default='ett_crossover/v4s20_reset_stall/model_rollouts.npz')
  args = ap.parse_args(argv)
  {'ruler': mode_ruler, 'normrank': mode_normrank}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
