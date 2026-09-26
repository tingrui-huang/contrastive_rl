"""Step 1 of the simulator-paired one-step ETT (AntMaze V6): the supervision
tuples (s, a_b, a_q, s', onset) for every logged transition and every
branch transition, with the expert advice a_b walked along each path.

Built on Step 0 (`audit_v6_ett_context.py`): the teacher's advice at a
context is a deterministic function of the route intent, the position
history and the timetable, verified bit-exactly on the log.  Here:

  * logged rows (diagonal): a_b = a_q = the logged torque, s' = the logged
    next state, onset = 0 (the d05 log has no failures);
  * branch rows: the teacher state at the anchor is reconstructed from the
    logged history, the zone latches are re-derived at the recorded
    consultation steps under the branch's redrawn timetable (Step 0's
    rule), and the teacher is then WALKED along the branch's own rows
    (its relay state evolving with the branch path, the rule consulted
    when the path first stands at a mouth under that timetable): a_b_j =
    the advice at branch row j; a_q_j = the executed torque (row 0 the
    logged query, rows >= 1 the start agent's mode torques); s' = the next
    branch row; onset = 1 on the last transition of a branch that died.
    Row 0's advice must equal Step 0's a_b (checked).

Cross-fitting folds are assigned by source episode (3 folds, the seed of
`exp_v6_learned_ett`), so a model that generates an anchor's futures never
sees that anchor's episode.  Output: `ett_context/supervision.npz` --
for the branch rows, `advice_rows` / `onset_rows` / `valid_rows` aligned
with `branches_cf.npz` (the states and torques are read from there), the
per-branch redrawn timetable; for the log rows, nothing new (a_b = a_q),
only the fold table.  `supervision.json` reports the composition.

  JAX_PLATFORMS=cpu python scripts/build_v6_ett_supervision.py --workers 20
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
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import exp_v6_mainline_pilot as MP  # noqa: E402
import audit_v6_ett_context as AC  # noqa: E402

OUT = AC.OUT
TOL = 1e-5
N_FOLDS, FOLD_SEED = 3, 161_000_001


def folds_by_source_episode(anchors):
  src = anchors.source_episode.astype(np.int64)
  uniq = np.unique(src)
  perm = np.random.default_rng(FOLD_SEED).permutation(uniq)
  fold_of = {}
  for f in range(N_FOLDS):
    for e in perm[f::N_FOLDS]:
      fold_of[int(e)] = f
  return fold_of


def _worker(args):
  """Walk the teacher along every branch of a chunk of episodes.  Returns advice rows, hold flags and checks."""
  ep_list, seed = args
  H = dict(np.load(OUT / 'hidden.npz', allow_pickle=False))
  BA = dict(np.load(OUT / 'branch_advice.npz', allow_pickle=False))
  obs, act, lengths, _ = MP.load_dataset()
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    off, L = d['offset'].astype(np.int64), d['length'].astype(np.int64)
    ks_all = np.flatnonzero(np.isin(anchors.episode, ep_list))
    rows_needed = np.concatenate([np.arange(off[k], off[k] + L[k]) for k in ks_all])
    bobs = d['obs_rows'][rows_needed]
  row_pos = {int(r): i for i, r in enumerate(rows_needed)}
  env, teacher = AC.make_env_teacher(seed)
  env.reset()
  out = []
  for e in ep_list:
    Le = int(lengths[e]); u = {1: bool(H['u1'][e]), 2: bool(H['u2'][e])}; t0 = {1: int(H['t0_1'][e]), 2: int(H['t0_2'][e])}
    ks = np.flatnonzero(anchors.episode == e)
    _, _, dstep, snaps = AC.replay_episode(env, teacher, obs[e], Le, str(H['intent'][e]), u, t0, anchor_ts=[int(anchors.t[k]) for k in ks])
    for k in ks:
      k = int(k); t = int(anchors.t[k])
      un = {1: bool(BA['u1_new'][k]), 2: bool(BA['u2_new'][k])}; tn = {1: int(BA['t0_1_new'][k]), 2: int(BA['t0_2_new'][k])}
      sn = AC.rederive_latches(snaps[t], un, tn, dstep)
      AC.restore_teacher(teacher, sn)
      n = int(L[k]); adv = np.zeros((n, 8), np.float32); hold = np.zeros(n, bool)
      for j in range(n - 1):                                   # the last row has a dummy torque and no successor
        o = AC.obs58(env, bobs[row_pos[int(off[k] + j)]])
        adv[j] = teacher.act(o, AC.schedule_of(un, tn, t + j)); hold[j] = teacher._holding_zone is not None
      out.append((k, adv, hold, float(np.abs(adv[0] - BA['a_b'][k]).max())))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None, help='episodes (smoke)')
  args = ap.parse_args()
  from multiprocessing import get_context
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  obs, act, lengths, _ = MP.load_dataset()
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    off, L, oc, bact = d['offset'].astype(np.int64), d['length'].astype(np.int64), d['outcome'].astype(str), d['act_rows']
    n_rows = int(d['obs_rows'].shape[0])
  BA = dict(np.load(OUT / 'branch_advice.npz', allow_pickle=False))
  eps = sorted(set(anchors.episode.tolist()))
  if args.limit:
    eps = eps[:args.limit]
  n_parts = max(1, args.workers * 4)
  parts = [eps[i::n_parts] for i in range(n_parts)]; parts = [p for p in parts if p]
  t0 = time.time()
  if args.workers <= 1:
    res = [_worker((p, 195_000_000 + i)) for i, p in enumerate(parts)]
  else:
    with get_context('spawn').Pool(args.workers) as pool:
      res = pool.map(_worker, [(p, 195_000_000 + i) for i, p in enumerate(parts)])
  advice = np.zeros((n_rows, 8), np.float32); hold = np.zeros(n_rows, bool); valid = np.zeros(n_rows, bool); onset = np.zeros(n_rows, bool)
  anchor_of_row = np.repeat(np.arange(anchors.n), L); j_of_row = np.arange(n_rows) - np.repeat(off, L)
  root_check = []
  for part in res:
    for k, adv, hd, chk in part:
      advice[off[k]:off[k] + L[k]] = adv; hold[off[k]:off[k] + L[k]] = hd; valid[off[k]:off[k] + L[k] - 1] = True
      if oc[k] == 'death':
        onset[off[k] + L[k] - 1] = True                       # the transition INTO the last row killed the ant: label sits on row len-2 -> len-1
      root_check.append(chk)
  onset_rows = np.zeros(n_rows, bool)
  death = np.flatnonzero(oc == 'death')
  onset_rows[off[death] + L[death] - 2] = True                 # (s_{len-2}, a, s_{len-1}) is the fatal transition
  fold_of = folds_by_source_episode(anchors)
  fold_anchor = np.array([fold_of[int(e)] for e in anchors.source_episode]); fold_row = fold_anchor[anchor_of_row]
  # composition
  a_q = bact
  diff = np.abs(advice - a_q).max(axis=1)
  diag = (diff <= TOL) & valid
  q0 = (np.abs(a_q).max(axis=1) == 0) & valid; b0 = (np.abs(advice).max(axis=1) == 0) & valid
  cls = np.where(~valid, 'invalid', np.where(diag, 'diagonal', np.where(b0 & ~q0, 'off_hold', np.where(~b0 & q0, 'off_release', 'off_other'))))
  root = j_of_row == 0
  summ = {'branch_rows': n_rows, 'valid_transitions': int(valid.sum()), 'root_advice_matches_step0': int(sum(1 for c in root_check if c == 0.0)), 'root_advice_maxdiff': float(max(root_check)) if root_check else None,
          'onset_transitions': int(onset_rows.sum()), 'deaths_in_branch_file': int((oc == 'death').sum()),
          'classes_all_valid': {c: int((cls == c).sum()) for c in ('diagonal', 'off_hold', 'off_release', 'off_other')},
          'classes_root_rows': {c: int(((cls == c) & root).sum()) for c in ('diagonal', 'off_hold', 'off_release', 'off_other')},
          'classes_agent_rows (j >= 1)': {c: int(((cls == c) & ~root & valid).sum()) for c in ('diagonal', 'off_hold', 'off_release', 'off_other')},
          'off_other_median_maxdiff_agent_rows': float(np.median(diff[(cls == 'off_other') & ~root])) if ((cls == 'off_other') & ~root).any() else None,
          'onset_by_class': {c: int((onset_rows & (cls == c)).sum()) for c in ('diagonal', 'off_hold', 'off_release', 'off_other')},
          'hold_share_valid': float(hold[valid].mean()),
          'log_rows': int((lengths - 1).sum()), 'log_onsets': 0,
          'folds': {'n_folds': N_FOLDS, 'seed': FOLD_SEED, 'unit': 'source episode', 'anchors_per_fold': [int((fold_anchor == f).sum()) for f in range(N_FOLDS)], 'rows_per_fold': [int((fold_row == f).sum()) for f in range(N_FOLDS)]},
          'wall_seconds': time.time() - t0, 'limit': args.limit}
  np.savez_compressed(OUT / 'supervision.npz', advice_rows=advice, hold_rows=hold, valid_rows=valid, onset_rows=onset_rows, class_rows=cls.astype('<U11'), fold_rows=fold_row.astype(np.int8),
                      fold_anchor=fold_anchor.astype(np.int8), anchor_of_row=anchor_of_row.astype(np.int32), j_of_row=j_of_row.astype(np.int32),
                      u1_new=BA['u1_new'], u2_new=BA['u2_new'], t0_1_new=BA['t0_1_new'], t0_2_new=BA['t0_2_new'],
                      log_fold_episode=np.array([fold_of[int(anchors.source_episode[np.flatnonzero(anchors.episode == e)[0]])] if (anchors.episode == e).any() else -1 for e in range(len(lengths))], np.int8),
                      meta=np.asarray(json.dumps({'note': 'oracle-supervised engineering stage: advice from the privileged teacher walked along the simulator branches; onset = the fatal branch transition',
                                                  'branches_sha256': MP.sha256(MP.OUT / 'branches_cf.npz'), 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz')}, sort_keys=True)))
  MP.write_json(OUT / 'supervision.json', summ)
  print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
  main()
