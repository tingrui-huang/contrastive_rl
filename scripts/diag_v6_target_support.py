#!/usr/bin/env python
"""AntMaze V6 -- does the better candidate action get more TASK-GOAL target support?  (no rollouts, no training)

The user's question after a408cef: at the SAME state and task goal, does the
truly better candidate action (the one whose future reaches the task goal)
receive higher actual NCE target support -- the probability that the critic's
positive goal, drawn by the stream's own law, lands at the task goal -- and
does the current termination rule weaken or reverse that advantage compared
with an explicit ABSORBING alternative?

Data: the stage-1 paired query futures (exp_v6_query_coverage: q0 logged, q1
mode, q2..q5 samples; two hazard draws paired across the six queries) at the
4,276 start-region anchors, per lineage; the sealed CF branch table and the
recorded episodes give the goal MARGINALS (the law of the NCE negatives) under
both rules.  Nothing is generated.

Laws, per branch of L rows (root m = 0), anchor time t, remaining horizon
H = HORIZON - t (the branch's own step cap):
  current    P(m) ~ gamma^m, m = 1..L-1                         (CriticStream.draw)
  absorbing  P(m) ~ gamma^m, m = 1..H; rows m >= L-1 of a success / death
             branch are its ACTUAL terminal row (the reach position or the
             death position); a timeout has L-1 == H and is unchanged.
No relabelling: a death is not filled in as a goal, a reach position is not
replaced by the goal centre, success and death both get the tail.  The
absorbing law changes the learnt future distribution and would have to be
disclosed if ever trained with.

Read-outs (anchor-weighted; strata all start / reset t = 0 / t in [1, 30) / t >= 30):
  R1 per query: success, far completion, death, timeout; expected task-goal
     mass under both laws; the marginal for scale.
  R2 within (anchor, draw) pairs of queries with different outcomes: P(mass_good
     > mass_bad), mean difference, ratio of means -- by pair type.
  R3 per anchor: rank agreement across the six queries between the success
     rate (over the two draws) and the expected mass; top-1 agreement.
  R4 per anchor: far-going vs short-going queries (by the route the future
     took in both draws): delta success vs delta task-goal mass under both
     laws, paired-by-anchor bootstrap; sign agreement.  PRIMARY.
The judgement is written in the manifest before the numbers exist.
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
import exp_v6_clip_round as CR  # noqa: E402
import exp_v6_query_coverage as QC  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'target_support'
GAMMA, HORIZON = MP.GAMMA, MP.HORIZON
RADII = {'reach_0.5': 0.5, 'within_1': 1.0, 'within_2': 2.0}
GOAL_SETS = list(RADII) + ['goal_area']
PRIMARY_G, SECONDARY_G = 'reach_0.5', 'goal_area'
LAWS = ('cur', 'abs')
CLASSES = ('far_success', 'short_success', 'far_timeout', 'short_timeout', 'far_death', 'short_death')
PAIR_TYPES = [('far_success', 'short_death'), ('far_success', 'short_timeout'), ('far_success', 'far_timeout'), ('far_success', 'short_success'),
              ('short_success', 'short_death'), ('short_success', 'short_timeout'), ('success', 'non_success')]
STRATA = ('all_start', 'reset', 't_1_30', 't_ge_30')
BOOT, BOOT_SEED = 2000, 149_000_001
NQ, ND = QC.N_QUERIES, QC.DRAWS


# -------------------------------------------------------------------- laws
def masses(obs_rows, off, L, t, outcome, goal_xy_of_branch):
  """Per-branch task-goal masses under the current and the absorbing law.  Returns {goal_set: {'cur': [K], 'abs': [K]}}, H, tail."""
  K = len(L)
  row_of = np.repeat(np.arange(K), L)
  m = np.arange(len(row_of)) - np.repeat(off, L)
  after = m >= 1
  xy = np.ascontiguousarray(obs_rows[:, :2]).astype(np.float64)
  dist = np.linalg.norm(xy - goal_xy_of_branch[row_of], axis=1)
  cats = {name: dist <= r for name, r in RADII.items()}
  cats['goal_area'] = region_of(xy) == REGIONS.index('goal_area')
  g = np.where(after, GAMMA ** m, 0.0)
  z_cur = np.bincount(row_of, weights=g, minlength=K)
  timeout = outcome == 'timeout'
  H = np.where(timeout, L - 1, np.maximum(HORIZON - t, L - 1)).astype(np.int64)      # a branch timeout has L-1 == HORIZON-t by construction (asserted below)
  z_abs = GAMMA * (1.0 - GAMMA ** H) / (1.0 - GAMMA)                                  # sum_{m=1}^{H} gamma^m
  tail = (GAMMA ** L - GAMMA ** (H + 1)) / (1.0 - GAMMA)                              # sum_{m=L}^{H} gamma^m  (0 when L-1 == H)
  tail = np.where(timeout, 0.0, tail)
  term = off + L - 1
  out = {}
  for name, c in cats.items():
    s = np.bincount(row_of, weights=g * c, minlength=K)
    out[name] = {'cur': s / np.maximum(z_cur, 1e-12), 'abs': (s + c[term] * tail) / z_abs}
  return out, H, tail


def wm(x, w):
  w = np.asarray(w, float); x = np.asarray(x, float)
  return float((w * x).sum() / w.sum()) if w.sum() > 0 else float('nan')


def boot_ci(d, w, rng):
  d = np.asarray(d, float); w = np.asarray(w, float); n = len(d)
  if n == 0:
    return {'mean': float('nan'), 'ci95': [float('nan')] * 2, 'n': 0}
  vals = np.empty(BOOT)
  for b in range(BOOT):
    i = rng.integers(0, n, n); vals[b] = (w[i] * d[i]).sum() / max(w[i].sum(), 1e-12)
  return {'mean': wm(d, w), 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], 'se': float(vals.std()), 'n': int(n)}


# --------------------------------------------------------------- marginals
def marginals(anchors):
  """Anchor-weighted goal marginal (the NCE negatives' law) under both laws, for the CF branch table and the recorded futures."""
  res = {}
  t0 = time.time()
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    b_obs, off_b, Lb, oc_b, tb = d['obs_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64), d['outcome'].astype(str), d['t'].astype(np.int64)
  assert np.array_equal(tb, anchors.t)
  M, H, tail = masses(b_obs, off_b, Lb, tb, oc_b, anchors.goal_xy.astype(np.float64))
  to = oc_b == 'timeout'
  assert np.all((Lb - 1)[to] == (HORIZON - tb)[to]), 'a CF timeout branch does not end at the horizon'
  res['CF_sealed'] = {G: {law: wm(M[G][law], anchors.weight) for law in LAWS} for G in GOAL_SETS}
  res['CF_sealed']['tail_share_mean'] = wm(tail / (GAMMA * (1.0 - GAMMA ** H) / (1.0 - GAMMA)), anchors.weight)
  res['CF_sealed']['outcome'] = {k: wm(oc_b == k, anchors.weight) for k in ('success', 'death', 'timeout')}
  del b_obs
  print(f'CF marginal done {time.time() - t0:.0f} s', flush=True)
  obs, _, lengths, _ = MP.load_dataset()
  Lr = (anchors.ep_length - anchors.t).astype(np.int64)
  off_r = np.concatenate([[0], np.cumsum(Lr)[:-1]])
  rows_e = np.repeat(anchors.episode, Lr); rows_t = np.repeat(anchors.t, Lr) + (np.arange(Lr.sum()) - np.repeat(off_r, Lr))
  r_obs = obs[rows_e, rows_t, :MP.OBS_W]
  last_xy = obs[np.arange(len(lengths)), lengths - 1, :2]; goal_l = obs[np.arange(len(lengths)), 0, MP.STATE_DIM:MP.OBS_W]
  ep_out = np.where(np.linalg.norm(last_xy - goal_l, axis=1) <= RADII['reach_0.5'], 'success', np.where(lengths >= HORIZON, 'timeout', 'death'))
  oc_r = ep_out[anchors.episode]
  M, H, tail = masses(r_obs, off_r, Lr, anchors.t.astype(np.int64), oc_r, anchors.goal_xy.astype(np.float64))
  res['O_recorded'] = {G: {law: wm(M[G][law], anchors.weight) for law in LAWS} for G in GOAL_SETS}
  res['O_recorded']['tail_share_mean'] = wm(tail / (GAMMA * (1.0 - GAMMA ** H) / (1.0 - GAMMA)), anchors.weight)
  res['O_recorded']['outcome'] = {k: wm(oc_r == k, anchors.weight) for k in ('success', 'death', 'timeout')}
  res['O_recorded']['rows_with_L_minus_1_above_horizon'] = int(((Lr - 1) > (HORIZON - anchors.t)).sum())
  del r_obs
  print(f'O marginal done {time.time() - t0:.0f} s', flush=True)
  return res


# ------------------------------------------------------------ per lineage
def lineage_arrays(s, anchors):
  """[A, ND, NQ] arrays over the start anchors (sorted anchor ids): outcome class, success, entered_far, length, masses."""
  with np.load(QC.branch_path(s), allow_pickle=False) as d:
    obs, off, L = d['obs_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64)
    aid, qid, dr, t = (d[k].astype(np.int64) for k in ('anchor_id', 'query_id', 'draw', 't')); oc = d['outcome'].astype(str)
    meta = json.loads(str(d['meta']))
  assert np.array_equal(t, anchors.t[aid])
  K = len(L); row_of = np.repeat(np.arange(K), L); m = np.arange(len(row_of)) - np.repeat(off, L); after = m >= 1
  xy = obs[:, :2]
  far_row = (xy[:, 1] >= CR.DETOUR_Y) & (xy[:, 0] < CR.DETOUR_MAX_X)
  entered_far = np.bincount(row_of, weights=(far_row & after), minlength=K) > 0
  M, H, tail = masses(obs, off, L, t, oc, anchors.goal_xy[aid].astype(np.float64))
  to = oc == 'timeout'
  assert np.all((L - 1)[to] == (HORIZON - t)[to]), 'a query timeout branch does not end at the horizon'
  del obs
  ids = np.unique(aid); A = len(ids)
  pos = np.searchsorted(ids, aid)
  flat = pos * (ND * NQ) + dr * NQ + qid
  assert len(np.unique(flat)) == K == A * ND * NQ, 'the query table is not a complete (anchor, draw, query) grid'
  order = np.argsort(flat)

  def grid(x):
    return np.asarray(x)[order].reshape(A, ND, NQ)
  succ = grid(oc == 'success'); death = grid(oc == 'death'); tmo = grid(to); far = grid(entered_far)
  cls = np.where(succ, np.where(far, 0, 1), np.where(tmo, np.where(far, 2, 3), np.where(far, 4, 5)))
  arr = {'ids': ids, 'succ': succ, 'death': death, 'timeout': tmo, 'far': far, 'cls': cls, 'length': grid(L), 'tail_share': grid(tail / (GAMMA * (1.0 - GAMMA ** H) / (1.0 - GAMMA))),
         'mass': {G: {law: grid(M[G][law]) for law in LAWS} for G in GOAL_SETS}, 'meta': meta}
  return arr


def strata_masks(anchors, ids):
  tt = anchors.t[ids]
  return {'all_start': np.ones(len(ids), bool), 'reset': tt == 0, 't_1_30': (tt >= 1) & (tt < 30), 't_ge_30': tt >= 30}


def r1_per_query(arr, W, mask):
  out = {}
  w2 = np.broadcast_to((W * mask)[:, None], (len(W), ND))
  for q in range(NQ):
    row = {'success': wm(arr['succ'][:, :, q], w2), 'completed_far': wm(arr['succ'][:, :, q] & arr['far'][:, :, q], w2), 'entered_far': wm(arr['far'][:, :, q], w2),
           'death': wm(arr['death'][:, :, q], w2), 'timeout': wm(arr['timeout'][:, :, q], w2), 'length': wm(arr['length'][:, :, q], w2)}
    for G in GOAL_SETS:
      for law in LAWS:
        row[f'{G}_{law}'] = wm(arr['mass'][G][law][:, :, q], w2)
        s = arr['succ'][:, :, q]
        row[f'{G}_{law}_given_success'] = wm(arr['mass'][G][law][:, :, q][s], w2[s]) if s.any() else float('nan')
    out[QC.QUERY_NAMES[q]] = row
  return out


def r2_pairs(arr, W, mask):
  I, J = np.triu_indices(NQ, 1)
  ci, cj = arr['cls'][:, :, I], arr['cls'][:, :, J]
  w = np.broadcast_to((W * mask)[:, None, None], ci.shape)
  is_s = lambda c: c <= 1  # noqa: E731
  out = {}
  for cg, cb in PAIR_TYPES:
    if cg == 'success':
      gi, gj, bi, bj = is_s(ci), is_s(cj), ~is_s(ci), ~is_s(cj)
    else:
      gi, gj, bi, bj = ci == CLASSES.index(cg), cj == CLASSES.index(cg), ci == CLASSES.index(cb), cj == CLASSES.index(cb)
    fwd = gi & bj; bwd = bi & gj; sel = (fwd | bwd) & (w > 0)
    rec = {'n_pairs': int(sel.sum()), 'weight': float(w[sel].sum())}
    for G in (PRIMARY_G, SECONDARY_G):
      for law in LAWS:
        mi, mj = arr['mass'][G][law][:, :, I], arr['mass'][G][law][:, :, J]
        mg = np.where(fwd, mi, mj)[sel]; mb = np.where(fwd, mj, mi)[sel]; ww = w[sel]
        rec[f'{G}_{law}'] = {'p_good_gt_bad': wm(mg > mb, ww), 'p_tie': wm(mg == mb, ww), 'mean_diff': wm(mg - mb, ww),
                             'mean_good': wm(mg, ww), 'mean_bad': wm(mb, ww), 'ratio_of_means': (wm(mg, ww) / wm(mb, ww) if wm(mb, ww) > 0 else float('inf'))} if sel.any() else None
    out[f'{cg}_vs_{cb}'] = rec
  return out


def r3_rank(arr, W, mask):
  r = arr['succ'].mean(axis=1)                       # [A, NQ] success rate over the draws
  I, J = np.triu_indices(NQ, 1)
  dr = r[:, I] - r[:, J]
  valid = (dr != 0) & (mask[:, None]) & (W[:, None] > 0)
  out = {}
  for G in (PRIMARY_G, SECONDARY_G):
    for law in LAWS:
      Em = arr['mass'][G][law].mean(axis=1)
      dm = Em[:, I] - Em[:, J]
      conc = np.sign(dm) == np.sign(dr); tie = dm == 0
      w = np.broadcast_to(W[:, None], dr.shape)
      any_diff = (r.max(axis=1) > r.min(axis=1)) & mask & (W > 0)
      top = np.argmax(Em, axis=1)
      agree = r[np.arange(len(r)), top] == r.max(axis=1)
      out[f'{G}_{law}'] = {'p_concordant_excl_ties': wm(conc[valid & ~tie], w[valid & ~tie]) if (valid & ~tie).any() else float('nan'),
                           'tie_share': wm(tie[valid], w[valid]) if valid.any() else float('nan'), 'n_pairs': int(valid.sum()),
                           'top1_agreement': wm(agree[any_diff], W[any_diff]) if any_diff.any() else float('nan'), 'n_anchors_with_difference': int(any_diff.sum())}
  return out


def r4_route(arr, W, mask, rng):
  """Far-going (entered the far route in both draws) vs short-going (in neither) queries, per anchor; the PRIMARY read-out."""
  far_q = arr['far'].all(axis=1); short_q = ~arr['far'].any(axis=1)          # [A, NQ]
  r = arr['succ'].mean(axis=1)
  both = far_q.any(axis=1) & short_q.any(axis=1) & mask & (W > 0)
  mixed_share = wm((~far_q & ~short_q).mean(axis=1)[mask], W[mask]) if mask.any() else float('nan')
  out = {'n_anchors_both_classes': int(both.sum()), 'weight_both': float(W[both].sum()), 'weight_stratum': float(W[mask].sum()),
         'share_far_queries': wm(far_q.mean(axis=1)[mask], W[mask]) if mask.any() else float('nan'), 'mixed_route_query_share': mixed_share}
  if not both.any():
    return out
  wb = W[both]

  def cmean(x, sel):
    return (x * sel).sum(axis=1) / np.maximum(sel.sum(axis=1), 1)
  ds = cmean(r, far_q)[both] - cmean(r, short_q)[both]
  out['delta_success'] = boot_ci(ds, wb, rng)
  out['success_far'] = wm(cmean(r, far_q)[both], wb); out['success_short'] = wm(cmean(r, short_q)[both], wb)
  for G in GOAL_SETS:
    for law in LAWS:
      Em = arr['mass'][G][law].mean(axis=1)
      dm = cmean(Em, far_q)[both] - cmean(Em, short_q)[both]
      rec = boot_ci(dm, wb, rng)
      rec.update({'mass_far': wm(cmean(Em, far_q)[both], wb), 'mass_short': wm(cmean(Em, short_q)[both], wb),
                  'p_sign_agree_given_far_better': wm(dm > 0, wb * (ds > 0)) if (ds > 0).any() else float('nan'),
                  'p_sign_reversed_given_far_better': wm(dm < 0, wb * (ds > 0)) if (ds > 0).any() else float('nan'),
                  'p_sign_agree_given_short_better': wm(dm < 0, wb * (ds < 0)) if (ds < 0).any() else float('nan'),
                  'weight_far_better': float(wb[ds > 0].sum()), 'weight_short_better': float(wb[ds < 0].sum()), 'weight_tied': float(wb[ds == 0].sum())})
      out[f'{G}_{law}'] = rec
  return out


# ---------------------------------------------------------------- driver
def seal(anchors):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists():
    return MP.read_json(p)
  ids = QC.start_anchor_ids(anchors)
  man = {'diagnostic': 'AntMaze V6: task-goal target support per candidate action on the stage-1 query futures, current vs absorbing termination law (no rollouts, no training)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'),
         'inputs': {'queries': {f'seed_{s}': (MP.sha256(QC.branch_path(s)) if QC.branch_path(s).exists() else 'missing') for s in MP.SEEDS}, 'branches_cf': MP.sha256(MP.OUT / 'branches_cf.npz')},
         'states': {'n_start_anchors': int(len(ids)), 'weight': float(anchors.weight[ids].sum()), 'n_reset': int((anchors.t[ids] == 0).sum()), 'weight_reset': float(anchors.weight[ids][anchors.t[ids] == 0].sum())},
         'laws': {'current': 'P(m) ~ gamma^m over m = 1..L-1 (CriticStream.draw; each path normalised over its own rows)',
                  'absorbing': 'P(m) ~ gamma^m over m = 1..H, H = HORIZON - t; rows m >= L-1 of a success / death branch are the branch\'s actual terminal row; a timeout (L-1 == H) is unchanged.  No relabelling: no death filled in as a goal, no reach position replaced by the goal centre, success and death both get the tail',
                  'gamma': GAMMA, 'horizon': HORIZON},
         'goal_sets': {**{k: f'||xy - task goal|| <= {v}' for k, v in RADII.items()}, 'goal_area': 'the maze region (exp_v6_learned_ett.region_of)'}, 'primary_goal_set': PRIMARY_G, 'secondary_goal_set': SECONDARY_G,
         'marginal': 'anchor-weighted mean of the per-anchor law over ALL 53,747 anchors: the CF sealed branch table (what the lineage critics trained on) and the recorded futures (arm O); the law of the NCE negatives',
         'weights': 'anchor weight (the critic anchor law); unweighted (per-anchor equal) in report.json', 'strata': list(STRATA), 'primary_stratum': 'reset',
         'route_classes': 'a query is far-going if its future entered the far route in BOTH draws, short-going if in neither; mixed queries are excluded from R4 and their share reported',
         'read_outs': {'R1': 'per query: success, far completion, death, timeout, expected task-goal mass under both laws',
                       'R2': 'within (anchor, draw) pairs of queries with different outcome classes: P(mass_good > mass_bad), mean difference, ratio of means',
                       'R3': 'per anchor: concordance across the six queries between the success rate (two draws) and the expected mass; top-1 agreement',
                       'R4': 'per anchor with both route classes: delta success (far - short) vs delta expected task-goal mass under both laws; paired-by-anchor bootstrap; sign agreement (PRIMARY)'},
         'bootstrap': {'paired_by': 'anchor', 'resamples': BOOT, 'seed': BOOT_SEED},
         'judgement': {'J1_current_weakens_or_reverses': f'in the reset stratum, {PRIMARY_G}: far-going queries succeed more (delta_success mean > 0, CI lower bound > 0) while under the CURRENT law delta_mass is not above zero (CI covers 0 or lies below) OR p_sign_agree_given_far_better < 0.5',
                       'J2_absorbing_improves': f'under the ABSORBING law delta_mass mean > 0 with CI lower bound > 0 AND p_sign_agree_given_far_better above the current law\'s',
                       'decision': {'J1 and J2 in every lineage': "the user's option 1: a small O / CF mainline comparison with the same termination handling in both arms (NCE, BC 0.05, actor data, weights, budget unchanged; disclosed as a changed future law) -- the user's call",
                                    'otherwise': "the user's option 2: the termination rule is not a found cause; the problem stays in how the learner / policy update uses the target; stop this hypothesis"},
                       'secondary': 'R3 concordance and top-1 agreement (both laws); R2 far_success vs short_success (the pure speed penalty) -- reported, not part of the rule'}}
  MP.write_json(p, man)
  return man


def run(args):
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  man = seal(anchors)
  res = {'manifest_sealed_at': man['sealed_at'], 'marginals': marginals(anchors), 'lineages': {}}
  for s in MP.SEEDS:
    if not QC.branch_path(s).exists():
      print(f'lineage {s}: queries missing', flush=True); continue
    t0 = time.time()
    arr = lineage_arrays(s, anchors)
    W = anchors.weight[arr['ids']]; masks = strata_masks(anchors, arr['ids'])
    np.savez_compressed(OUT / f'per_branch_s{s}.npz', anchor_id=arr['ids'], succ=arr['succ'], death=arr['death'], timeout=arr['timeout'], entered_far=arr['far'], cls=arr['cls'], length=arr['length'],
                        tail_share=arr['tail_share'], weight=W, t=anchors.t[arr['ids']], **{f'{G}_{law}': arr['mass'][G][law] for G in GOAL_SETS for law in LAWS},
                        layout=np.asarray('[anchor (sorted id), draw, query]; cls = ' + ','.join(CLASSES)))
    rec = {'meta': {k: arr['meta'].get(k) for k in ('continuation_ckpt_sha256', 'n_branches', 'outcome_counts', 'policy_scale_median_at_anchors')}, 'n_anchors': int(len(arr['ids'])), 'strata': {}}
    for name, mask in masks.items():
      rng = np.random.default_rng(BOOT_SEED + 10 * s + STRATA.index(name))
      blk = {'n_anchors': int(mask.sum()), 'weight': float(W[mask].sum())}
      for wname, ww in (('weighted', W), ('unweighted', np.ones_like(W))):
        blk[wname] = {'R1': r1_per_query(arr, ww, mask), 'R2': r2_pairs(arr, ww, mask), 'R3': r3_rank(arr, ww, mask), 'R4': r4_route(arr, ww, mask, rng)}
      rec['strata'][name] = blk
    res['lineages'][f'seed_{s}'] = rec
    print(f'lineage {s} done {time.time() - t0:.0f} s', flush=True)
  res['judgement'] = judge(res)
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_report_md(res, man), encoding='utf-8')
  print(json.dumps(res['judgement'], indent=1), flush=True)


def judge(res):
  out = {'per_lineage': {}, 'rule': 'manifest.judgement'}
  ok_all = True
  for key, rec in res['lineages'].items():
    r4 = rec['strata']['reset']['weighted']['R4']
    if 'delta_success' not in r4:
      out['per_lineage'][key] = {'J1': None, 'J2': None, 'note': 'no reset anchor with both route classes'}; ok_all = False; continue
    ds = r4['delta_success']; dc = r4[f'{PRIMARY_G}_cur']; da = r4[f'{PRIMARY_G}_abs']
    far_better = ds['mean'] > 0 and ds['ci95'][0] > 0
    cur_not_above = (dc['ci95'][0] <= 0) or (dc['mean'] <= 0)
    j1 = bool(far_better and (cur_not_above or dc['p_sign_agree_given_far_better'] < 0.5))
    j2 = bool(da['mean'] > 0 and da['ci95'][0] > 0 and da['p_sign_agree_given_far_better'] > dc['p_sign_agree_given_far_better'])
    out['per_lineage'][key] = {'J1': j1, 'J2': j2, 'far_better': bool(far_better), 'delta_success': ds['mean'], 'delta_success_ci': ds['ci95'],
                               'delta_mass_cur': dc['mean'], 'delta_mass_cur_ci': dc['ci95'], 'p_agree_cur': dc['p_sign_agree_given_far_better'],
                               'delta_mass_abs': da['mean'], 'delta_mass_abs_ci': da['ci95'], 'p_agree_abs': da['p_sign_agree_given_far_better']}
    ok_all = ok_all and j1 and j2
  out['J1_and_J2_every_lineage'] = bool(ok_all and len(res['lineages']) == len(MP.SEEDS))
  return out


def f3(x):
  return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else (f'{x:.4f}' if abs(x) < 0.01 else f'{x:.3f}')


def ci(rec):
  return f"{f3(rec['mean'])} [{f3(rec['ci95'][0])}, {f3(rec['ci95'][1])}]"


def write_report_md(res, man):
  L = ['# Task-goal target support per candidate action: current vs absorbing termination law', '',
       f"Sealed {man['sealed_at']} (git {man['git_head'][:8]}); `manifest.json`, `report.json`, `per_branch_s*.npz` (not committed).  No rollouts, no training.", '',
       f"Laws: current = {man['laws']['current']}; absorbing = {man['laws']['absorbing']}.  gamma {GAMMA}, horizon {HORIZON}.", '',
       '## Marginals (the law of the NCE negatives; anchor-weighted over all anchors)', '',
       '| source | success / death / timeout | ' + ' | '.join(f'{G} cur / abs' for G in GOAL_SETS) + ' | mean tail share (abs) |', '|---|---|' + '---:|' * len(GOAL_SETS) + '---:|']
  for src, rec in res['marginals'].items():
    o = rec['outcome']
    L.append(f"| {src} | {o['success']:.3f} / {o['death']:.3f} / {o['timeout']:.3f} | " + ' | '.join(f"{rec[G]['cur']:.4f} / {rec[G]['abs']:.4f}" for G in GOAL_SETS) + f" | {rec['tail_share_mean']:.3f} |")
  for key, rec in res['lineages'].items():
    L += ['', f'## Lineage {key[-1]}', '']
    for st in STRATA:
      blk = rec['strata'][st]; w = blk['weighted']
      L += [f"### {st}: {blk['n_anchors']} anchors, weight {blk['weight']:.4f}", '', '**R1 per query (anchor-weighted).**', '',
            f'| query | success | completed far | entered far | death | timeout | length | {PRIMARY_G} cur | {PRIMARY_G} abs | {PRIMARY_G} cur given success | {PRIMARY_G} abs given success | {SECONDARY_G} cur | {SECONDARY_G} abs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
      for q, row in w['R1'].items():
        L.append(f"| {q} | {row['success']:.3f} | {row['completed_far']:.3f} | {row['entered_far']:.3f} | {row['death']:.3f} | {row['timeout']:.3f} | {row['length']:.0f} | {row[f'{PRIMARY_G}_cur']:.5f} | {row[f'{PRIMARY_G}_abs']:.4f} | "
                 f"{f3(row[f'{PRIMARY_G}_cur_given_success'])} | {f3(row[f'{PRIMARY_G}_abs_given_success'])} | {row[f'{SECONDARY_G}_cur']:.4f} | {row[f'{SECONDARY_G}_abs']:.4f} |")
      r4 = w['R4']
      L += ['', f"**R4 far-going vs short-going queries (PRIMARY): {r4['n_anchors_both_classes']} anchors with both classes (weight {r4['weight_both']:.4f} of {r4['weight_stratum']:.4f}); far-going query share {f3(r4['share_far_queries'])}, mixed-route share {f3(r4['mixed_route_query_share'])}.**", '']
      if 'delta_success' in r4:
        L += [f"delta success (far - short) = {ci(r4['delta_success'])}; success far {r4['success_far']:.3f} vs short {r4['success_short']:.3f}.", '',
              '| goal set | law | mass far | mass short | delta [95 % CI] | P(sign agrees given far better) | P(reversed given far better) | weight far better / short better / tied |', '|---|---|---:|---:|---|---:|---:|---|']
        for G in GOAL_SETS:
          for law in LAWS:
            m = r4[f'{G}_{law}']
            L.append(f"| {G} | {law} | {f3(m['mass_far'])} | {f3(m['mass_short'])} | {ci(m)} | {f3(m['p_sign_agree_given_far_better'])} | {f3(m['p_sign_reversed_given_far_better'])} | {m['weight_far_better']:.4f} / {m['weight_short_better']:.4f} / {m['weight_tied']:.4f} |")
      L += ['', '**R3 rank agreement across the six queries (success rate over the two draws vs expected mass).**', '', '| goal set / law | P(concordant, ties excluded) | tie share | pairs | top-1 agreement | anchors with a difference |', '|---|---:|---:|---:|---:|---:|']
      for k, m in w['R3'].items():
        L.append(f"| {k} | {f3(m['p_concordant_excl_ties'])} | {f3(m['tie_share'])} | {m['n_pairs']} | {f3(m['top1_agreement'])} | {m['n_anchors_with_difference']} |")
      L += ['', '**R2 within-(anchor, draw) pairs by outcome class.**', '', f'| pair (good vs bad) | pairs | P(good > bad) cur / abs | mean diff cur / abs | ratio of means cur / abs | {SECONDARY_G}: P(good > bad) cur / abs |', '|---|---:|---|---|---|---|']
      for k, m in w['R2'].items():
        if m.get(f'{PRIMARY_G}_cur') is None:
          L.append(f'| {k} | 0 | | | | |'); continue
        c, a = m[f'{PRIMARY_G}_cur'], m[f'{PRIMARY_G}_abs']; c2, a2 = m[f'{SECONDARY_G}_cur'], m[f'{SECONDARY_G}_abs']
        L.append(f"| {k} | {m['n_pairs']} | {f3(c['p_good_gt_bad'])} / {f3(a['p_good_gt_bad'])} | {f3(c['mean_diff'])} / {f3(a['mean_diff'])} | {f3(c['ratio_of_means'])} / {f3(a['ratio_of_means'])} | {f3(c2['p_good_gt_bad'])} / {f3(a2['p_good_gt_bad'])} |")
      L.append('')
  J = res['judgement']
  L += ['## Judgement (manifest rule; reset stratum, ' + PRIMARY_G + ')', '', '| lineage | far better | delta success | delta mass cur | P agree cur | delta mass abs | P agree abs | J1 | J2 |', '|---|---|---|---|---:|---|---:|---|---|']
  for key, j in J['per_lineage'].items():
    if j.get('J1') is None:
      L.append(f"| {key} | | | | | | | {j.get('note')} | |"); continue
    L.append(f"| {key} | {j['far_better']} | {f3(j['delta_success'])} [{f3(j['delta_success_ci'][0])}, {f3(j['delta_success_ci'][1])}] | {f3(j['delta_mass_cur'])} [{f3(j['delta_mass_cur_ci'][0])}, {f3(j['delta_mass_cur_ci'][1])}] | {f3(j['p_agree_cur'])} | "
             f"{f3(j['delta_mass_abs'])} [{f3(j['delta_mass_abs_ci'][0])}, {f3(j['delta_mass_abs_ci'][1])}] | {f3(j['p_agree_abs'])} | {j['J1']} | {j['J2']} |")
  L += ['', f"J1 and J2 in every lineage: **{J['J1_and_J2_every_lineage']}**.", '']
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='mode', required=True)
  sub.add_parser('run')
  args = ap.parse_args(argv)
  run(args)


if __name__ == '__main__':
  main()
