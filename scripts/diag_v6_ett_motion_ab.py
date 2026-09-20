#!/usr/bin/env python
"""AntMaze V6 one-step ETT -- does the motion / stationary model turn the (un-executed) advice a_b into motion?  (user's check, 2026-09-20)

The physical audit (ett_context) showed one-step motion is a function of (s, a_q) only: the hidden context enters through the
onset.  The v3 motion regression and the stationary gate nevertheless take features(s, a_b, a_q) = [s, a_b, a_q, a_q - a_b,
a_q * a_b], and at generation a_b is the generator's draw.  Two checks on the held-out rows / anchors of each fold, fixed models,
no training:

A  one-step sensitivity: fix (s, a_q), swap a_b (real teacher -> generator draw / zero / a_q / another row's a_b) and measure the
   shift of the predicted next state and the stationary gate; the physically meaningful counterpart is the shift when the
   EXECUTED torque a_q is swapped between rows with a_b fixed; scales: the real displacement and the model's own one-step error.
B  closed-loop divergence on the far-route legs and the pre-mouth stall anchors: the model rolled from the anchor with the start
   agent's mode on the model state, advice from the generator (true context) / a_b = a_q (diagonal) / the simulator branch's real
   advice sequence; against the simulator branch of the same anchor: reach / timeout, the FIRST step where the model leaves the
   simulator path (|xy| > 0.25) and what happened there (stationary gate fired while the simulator moved, or drift); plus the
   teacher-forced one-step error profile along the simulator paths (false-stationary rate at moving rows).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_ett_futures as EF  # noqa: E402
import fit_v6_ett_advice as FA  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from fit_v6_ett_one_step_v2 import STAT_TOL, in_band  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'ett_motion_ab'
ETT = 'v3'
STATE_DIM, OBS_W, ACTION_DIM, HORIZON = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM, MP.HORIZON
FAR = [REGIONS.index(n) for n in ('west_column', 'top_corridor', 'east_column')]
SEED = 207_000_000
MAX_ROWS_A = 40_000
DEV_TOL = 0.25
READ = {'A_ab_shift_vs_aq_shift': 0.30, 'A_stat_flip_rate': 0.05, 'B_outcome_gap_gen_vs_diag': 0.05}   # read-out thresholds for "the advice matters to the motion", stated in advance


def step_chunked(P, s, ab, aq, kh, kb, chunk=32768):
  outs = [[], [], []]
  for i in range(0, len(s), chunk):
    for o, v in zip(outs, P.step(s[i:i + chunk], ab[i:i + chunk], aq[i:i + chunk], kh[i:i + chunk], kb[i:i + chunk])):
      o.append(v)
  return tuple(np.concatenate(o) for o in outs)


def q(x):
  x = np.asarray(x, float)
  return {'mean': float(x.mean()), 'median': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'p99': float(np.percentile(x, 99))} if len(x) else {}


# ------------------------------------------------------------------ check A
def check_a(fold, R, P, gen, rng):
  ho = np.flatnonzero(R.valid & (R.fold_row == fold))
  s, ab, aq, s2 = R.bobs[:, :STATE_DIM], R.adv, R.bact, None
  reg = R.region; band = R.band
  d_real = np.abs(R.bobs[ho + 1, :STATE_DIM] - R.bobs[ho, :STATE_DIM]).max(axis=1)
  strata = {'all': np.ones(len(ho), bool), 'start': reg[ho] == REGIONS.index('start'), 'pre_zone1': reg[ho] == REGIONS.index('pre_zone1'),
            'band': band[ho], 'far_legs_moving': np.isin(reg[ho], FAR) & (d_real >= STAT_TOL), 'stationary_real': d_real < STAT_TOL, 'hold_rows': R.hold[ho]}
  res = {}
  for name, m in strata.items():
    rows = ho[m]
    if len(rows) < 200:
      continue
    if len(rows) > MAX_ROWS_A:
      rows = np.sort(rng.choice(rows, size=MAX_ROWS_A, replace=False))
    S, AB, AQ = s[rows], ab[rows], aq[rows]; S2 = R.bobs[rows + 1, :STATE_DIM]
    kh, kb = R.kh[rows], R.kb[rows]
    base, _, pst0 = P.step(S, AB, AQ, kh, kb)
    x = R.feats(rows, gen.norm); _, ab_gen, _ = gen.sample(x, rng, decision='map')
    perm = rng.permutation(len(rows))
    alts = {'generator_draw': ab_gen, 'zero_hold': np.zeros_like(AB), 'diagonal_aq': AQ.copy(), 'other_row_ab': AB[perm]}
    out = {'rows': int(len(rows)), 'real_displacement_xy': q(np.linalg.norm(S2[:, :2] - S[:, :2], axis=1)), 'model_error_xy': q(np.linalg.norm(base[:, :2] - S2[:, :2], axis=1)),
           'stat_gate_on_share_base': float((pst0 > 0.5).mean()), 'real_stationary_share': float((np.abs(S2 - S).max(axis=1) < STAT_TOL).mean()), 'alts': {}}
    sd = P.norm['s_std']
    for k, A in alts.items():
      pred, _, pst = P.step(S, A, AQ, kh, kb)
      out['alts'][k] = {'xy_shift': q(np.linalg.norm(pred[:, :2] - base[:, :2], axis=1)), 'state_shift_std_units': q(np.linalg.norm((pred - base) / sd, axis=1) / np.sqrt(STATE_DIM)),
                        'stat_flip_rate': float(((pst > 0.5) != (pst0 > 0.5)).mean()), 'mean_abs_dp_stat': float(np.abs(pst - pst0).mean()), 'mean_abs_ab_change': float(np.abs(A - AB).mean())}
    # the physical counterpart: swap the EXECUTED torque between rows (a_b fixed)
    pred, _, pst = P.step(S, AB, AQ[perm], kh, kb)
    out['aq_swap'] = {'xy_shift': q(np.linalg.norm(pred[:, :2] - base[:, :2], axis=1)), 'state_shift_std_units': q(np.linalg.norm((pred - base) / sd, axis=1) / np.sqrt(STATE_DIM)),
                      'stat_flip_rate': float(((pst > 0.5) != (pst0 > 0.5)).mean()), 'mean_abs_aq_change': float(np.abs(AQ[perm] - AQ).mean())}
    res[name] = out
  return res


# ------------------------------------------------------------------ check B
def rollout(P, gen, mode, anchors, ks, R, advice, rng, ctx_true=True, max_len=None):
  """Closed-loop model paths from anchors ks (deterministic motion; the onset integrated, not sampled).  advice in {'gen', 'diag', 'real_seq'}."""
  n = len(ks); e, t = anchors.episode[ks].astype(np.int64), anchors.t[ks].astype(np.int64)
  goal = anchors.goal_xy[ks].astype(np.float32); s = anchors.state[ks].astype(np.float32); a0 = anchors.action[ks].astype(np.float32)
  z = (R.u1[ks], R.u2[ks], R.t01[ks], R.t02[ks])
  la, lrun = R.log_prev
  st = FA.PathState(lrun[ks], la[ks], R.m_log[ks])
  max_steps = np.minimum(HORIZON - t, max_len if max_len else HORIZON)
  alive = np.ones(n, bool); reached = np.zeros(n, bool); L = np.ones(n, np.int64)
  xy_hist = np.full((n, int(max_steps.max()) + 1, 2), np.nan, np.float32); xy_hist[:, 0] = s[:, :2]
  stat_hist = np.zeros((n, int(max_steps.max())), bool); surv = np.ones(n); pdeath = np.zeros(n)
  for j in range(int(max_steps.max())):
    active = alive & (j < max_steps)
    if not active.any():
      break
    o31 = np.concatenate([s, goal], axis=1)
    aq = a0 if j == 0 else mode(o31)
    t_abs = t + j
    band_now = in_band(s[:, :2].astype(np.float64)); kb_now = st.band_counter(np.arange(n), j, band_now)
    m1, m2 = st.mouth_update(np.arange(n), t_abs, s[:, :2].astype(np.float64))
    if advice == 'gen':
      x = FA.features(o31, FA.context_features(t_abs, *z), FA.mouth_features(t_abs, m1, m2, z[2], z[3]), st.kh_prev, kb_now, st.prev_ab, gen.norm)
      hold, ab, _ = gen.sample(x, rng, decision='map')
    elif advice == 'diag':
      ab = aq.copy(); hold = np.zeros(n, bool)
    else:                                                    # the simulator branch's real advice at the same step (its own row j; the last known row beyond its end)
      rj = R.boff[ks] + np.minimum(j, R.blen[ks] - 2)
      ab = R.adv[rj].astype(np.float32); hold = R.hold[rj]
    kh_now = st.hold_counter(np.arange(n), j, hold)
    s_next, p_on, p_stat = P.step(s, ab, aq, kh_now, kb_now)
    stat_hist[active, j] = p_stat[active] > 0.5
    pdeath[active] += surv[active] * p_on[active]; surv[active] *= (1 - p_on[active])
    r_now = active & (np.linalg.norm(s_next[:, :2] - goal, axis=1) <= EF.REACH_R)
    xy_hist[active, j + 1] = s_next[active, :2]
    L[active] += 1; reached |= r_now; alive &= ~r_now
    st.advance(np.arange(n), hold, ab)
    s = np.where(active[:, None], s_next, s).astype(np.float32)
  return {'reached': reached, 'L': L, 'xy': xy_hist, 'stat': stat_hist, 'p_death': pdeath}


def first_deviation(xy_model, L_model, xy_sim, L_sim, stat_model):
  """Per path: the first step j where |xy_model[j] - xy_sim[j]| > DEV_TOL (both defined); classify the model's behaviour there."""
  n = len(L_model); first = np.full(n, -1); kind = np.full(n, 'none', dtype='<U16')
  for i in range(n):
    m = min(int(L_model[i]), int(L_sim[i]))
    d = np.linalg.norm(xy_model[i, :m] - xy_sim[i, :m], axis=1)
    hit = np.flatnonzero(d > DEV_TOL)
    if not len(hit):
      continue
    j = int(hit[0]); first[i] = j
    sim_moved = np.linalg.norm(xy_sim[i, j] - xy_sim[i, j - 1]) > 1e-3 if j >= 1 else True
    model_moved = np.linalg.norm(xy_model[i, j] - xy_model[i, j - 1]) > 1e-3 if j >= 1 else True
    gate = bool(stat_model[i, j - 1]) if j >= 1 else False
    kind[i] = 'stat_gate_stalled' if (gate and sim_moved) else ('model_stalled' if (sim_moved and not model_moved) else ('sim_stalled' if (not sim_moved and model_moved) else 'drift'))
  return first, kind


def check_b(fold, R, P, gen, mode, anchors, rng):
  reg = region_of(anchors.state[:, :2]); fa = R.fold_anchor
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    oc_cf = d['outcome'].astype(str)
  far_row = (R.bobs[:, 1] >= 6.0) & (R.bobs[:, 0] < 2.0)
  entered_far = np.zeros(anchors.n, bool)
  for k in range(anchors.n):
    a, b = R.boff[k], R.boff[k] + R.blen[k]
    entered_far[k] = far_row[a + 1:b].any()
  groups = {'far_legs': np.isin(reg, FAR) & (fa == fold), 'start_sim_far_success': (reg == REGIONS.index('start')) & entered_far & (oc_cf == 'success') & (fa == fold),
            'pre_zone1_sim_timeout': (reg == REGIONS.index('pre_zone1')) & (oc_cf == 'timeout') & (fa == fold), 'pre_zone1_sim_success': (reg == REGIONS.index('pre_zone1')) & (oc_cf == 'success') & (fa == fold)}
  res = {}
  for name, m in groups.items():
    ks = np.flatnonzero(m)
    if len(ks) < 30:
      res[name] = {'n': int(len(ks)), 'note': 'too few anchors'}; continue
    if len(ks) > 1500:
      ks = np.sort(rng.choice(ks, size=1500, replace=False))
    L_sim = R.blen[ks]; xy_sim = np.full((len(ks), int(L_sim.max()), 2), np.nan, np.float32)
    for i, k in enumerate(ks):
      xy_sim[i, :L_sim[i]] = R.bobs[R.boff[k]:R.boff[k] + L_sim[i], :2]
    out = {'n': int(len(ks)), 'sim': {'success': float((oc_cf[ks] == 'success').mean()), 'timeout': float((oc_cf[ks] == 'timeout').mean()), 'death': float((oc_cf[ks] == 'death').mean()), 'mean_rows': float(L_sim.mean())}, 'model': {}}
    for adv in ('gen', 'diag', 'real_seq'):
      r = rollout(P, gen, mode, anchors, ks, R, adv, np.random.default_rng(SEED + 7 * fold + 1))
      first, kind = first_deviation(r['xy'], r['L'], xy_sim, L_sim, r['stat'])
      dev = first >= 0
      out['model'][adv] = {'reach': float(r['reached'].mean()), 'timeout_like': float((~r['reached']).mean()), 'p_death_mean': float(r['p_death'].mean()), 'mean_rows': float(r['L'].mean()),
                           'stat_gate_share_of_steps': float(r['stat'].sum() / max(r['L'].sum() - len(ks), 1)),
                           'deviated_share': float(dev.mean()), 'first_deviation_step': q(first[dev]) if dev.any() else {}, 'deviation_kind': {k: float((kind[dev] == k).mean()) for k in np.unique(kind[dev])} if dev.any() else {}}
    # teacher-forced one-step profile along the simulator paths of these anchors
    rows = np.concatenate([np.arange(R.boff[k], R.boff[k] + R.blen[k] - 1) for k in ks])
    pred, _, pst = step_chunked(P, R.bobs[rows, :STATE_DIM], R.adv[rows], R.bact[rows], R.kh[rows], R.kb[rows])
    real_next = R.bobs[rows + 1, :STATE_DIM]; moving = np.abs(real_next - R.bobs[rows, :STATE_DIM]).max(axis=1) >= STAT_TOL
    out['teacher_forced_one_step'] = {'rows': int(len(rows)), 'xy_error': q(np.linalg.norm(pred[:, :2] - real_next[:, :2], axis=1)), 'false_stationary_rate_at_moving_rows': float((pst[moving] > 0.5).mean()) if moving.any() else float('nan'),
                                      'missed_stationary_rate': float((pst[~moving] <= 0.5).mean()) if (~moving).any() else float('nan'), 'moving_share': float(moving.mean())}
    res[name] = out
    print(f'  B {name}: n {len(ks)} sim success {out["sim"]["success"]:.3f}; model reach gen {out["model"]["gen"]["reach"]:.3f} diag {out["model"]["diag"]["reach"]:.3f} real_seq {out["model"]["real_seq"]["reach"]:.3f}', flush=True)
  return res


def run(args):
  OUT.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  R = FA.Rows()
  anchors = R.anchors
  mode = EF._policy_mode_batched()
  F2.OUT = F2.OUT_V4 if ETT == 'v4' else F2.OUT_V3
  res = {'read_out_thresholds': READ, 'dev_tol': DEV_TOL, 'folds': {}}
  for f in range(FA.N_FOLDS):
    if args.fold is not None and int(args.fold) != f:
      continue
    P = F2.Predictor2(F2.load_model(f)); gen = FA.load_generator(f)
    rng = np.random.default_rng(SEED + f)
    A = check_a(f, R, P, gen, rng)
    print(f'fold {f}: A done {time.time() - t0:.0f} s', flush=True)
    B = check_b(f, R, P, gen, mode, anchors, rng)
    res['folds'][f'fold{f}'] = {'A': A, 'B': B}
    print(f'fold {f}: B done {time.time() - t0:.0f} s', flush=True)
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res), encoding='utf-8')
  print('done', flush=True)


def f3(x):
  return f'{x:.3f}' if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else 'nan'


def write_md(res):
  L = ['# Does the motion / stationary model turn the advice a_b into motion?  (fixed v3 models; held-out rows / anchors per fold)', '',
       f"Read-out thresholds (stated in advance): {json.dumps(res['read_out_thresholds'])}; deviation tolerance {res['dev_tol']}.", '',
       '## A. One-step sensitivity to a_b with (s, a_q) fixed (xy shift of the predicted next state; median / p90), against the a_q swap and the model error', '',
       '| fold | stratum | rows | real displ. xy med | model error xy med / p90 | stat on (base) / real stationary | generator draw: xy shift med / p90, stat flip | zero hold | diagonal a_q | other row a_b | **a_q swap** (executed torque): xy shift med / p90, stat flip |',
       '|---|---|---:|---:|---|---|---|---|---|---|---|']
  for fk, F in res['folds'].items():
    for name, o in F['A'].items():
      al = o['alts']
      cell = lambda a: f"{a['xy_shift']['median']:.4f} / {a['xy_shift']['p90']:.4f}, {a['stat_flip_rate']:.3f}"  # noqa: E731
      L.append(f"| {fk} | {name} | {o['rows']} | {o['real_displacement_xy']['median']:.4f} | {o['model_error_xy']['median']:.4f} / {o['model_error_xy']['p90']:.4f} | {o['stat_gate_on_share_base']:.3f} / {o['real_stationary_share']:.3f} | "
               f"{cell(al['generator_draw'])} | {cell(al['zero_hold'])} | {cell(al['diagonal_aq'])} | {cell(al['other_row_ab'])} | **{cell(o['aq_swap'])}** |")
  L += ['', '## B. Closed-loop model paths from held-out anchors vs the simulator branch of the same anchor', '',
        '| fold | group | n | sim success / timeout / death (rows) | advice | model reach / p_death / rows | stat-gate share of steps | deviated share | first deviation step med / p90 | kind (share) |',
        '|---|---|---:|---|---|---|---:|---:|---|---|']
  for fk, F in res['folds'].items():
    for name, o in F['B'].items():
      if 'note' in o:
        L.append(f"| {fk} | {name} | {o['n']} | {o['note']} | | | | | | |"); continue
      for adv, m in o['model'].items():
        fd = m['first_deviation_step']
        L.append(f"| {fk} | {name} | {o['n']} | {o['sim']['success']:.3f} / {o['sim']['timeout']:.3f} / {o['sim']['death']:.3f} ({o['sim']['mean_rows']:.0f}) | {adv} | {m['reach']:.3f} / {m['p_death_mean']:.3f} / {m['mean_rows']:.0f} | {m['stat_gate_share_of_steps']:.3f} | {m['deviated_share']:.3f} | "
                 f"{(f3(fd.get('median')) + ' / ' + f3(fd.get('p90'))) if fd else '-'} | {json.dumps({k: round(v, 2) for k, v in m['deviation_kind'].items()})} |")
      tf = o['teacher_forced_one_step']
      L.append(f"| {fk} | {name} | | teacher-forced one-step along the sim paths ({tf['rows']} rows): xy error med {tf['xy_error']['median']:.4f} / p90 {tf['xy_error']['p90']:.4f}; false-stationary at moving rows {tf['false_stationary_rate_at_moving_rows']:.4f}; missed stationary {f3(tf['missed_stationary_rate'])}; moving share {tf['moving_share']:.3f} | | | | | | |")
  return '\n'.join(L) + '\n'


def main(argv=None):
  ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('run',)); ap.add_argument('--fold', type=int, default=None); ap.add_argument('--ett', choices=('v3', 'v4'), default='v3')
  args = ap.parse_args(argv)
  global ETT, OUT
  ETT = args.ett
  if ETT == 'v4':
    OUT = MP.OUT / 'ett_motion_ab_v4'
  run(args)


if __name__ == '__main__':
  main()
