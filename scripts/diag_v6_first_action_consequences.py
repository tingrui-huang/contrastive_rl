"""The real consequences of ONE first action at the reset states (user's step 2 after b4aa391): the logged torque, the stalled seed-4 actor's
mode, the progressing seed-4 (draw 2) actor's mode and the start agent's mode, each followed by the SAME frozen start agent, under paired
hidden draws -- in the simulator (exp_v6_repeated_draws.py generate --cands ... --extra-cands ...) and through the S ETT
(diag_v6_ett_crossover.py run --cands ... --conts start).  Reports per candidate: success / death / timeout / far entry and the
gamma-law goal masses the critic's sampling law actually provides (reach 0.5 / near 2.0 / goal area / death frame), paired against the
logged torque per state; simulator vs model side by side.

  python scripts/diag_v6_first_action_consequences.py report --sim repeated_draws/rollouts_start_reset_stall.npz --model ett_crossover/v4s20_reset_stall/model_rollouts.npz
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

import exp_v6_mainline_pilot as MP  # noqa: E402

OUT = MP.OUT / 'first_action_consequences'
KEYS = ('success', 'death', 'timeout', 'far_entry', 'reach0.5', 'near2.0', 'goal_area', 'death_frame')


def per_state(R, cont_filter=None):
  """Per (candidate, state): the mean over the paired draws of every key; returns {cand: {state: {key: mean}}} and the draw count."""
  oc = R['outcome'].astype(str); cand = R['cand'].astype(str); st = R['state'].astype(np.int64)
  m = np.ones(len(oc), bool)
  if cont_filter is not None and 'cont' in R:
    m = R['cont'].astype(str) == cont_filter
  val = {'success': (oc == 'success').astype(float), 'death': (oc == 'death').astype(float), 'timeout': (oc == 'timeout').astype(float), 'far_entry': R['far_entry'].astype(float),
         **{k: R[k].astype(float) for k in ('reach0.5', 'near2.0', 'goal_area', 'death_frame')}}
  out = {}; n_draw = {}
  for c in sorted(set(cand[m])):
    out[c] = {}
    for s in sorted(set(st[m & (cand == c)])):
      sel = m & (cand == c) & (st == s)
      out[c][int(s)] = {k: float(v[sel].mean()) for k, v in val.items()}; n_draw[c] = int(sel.sum())
  return out, n_draw


def summarise(P):
  cands = list(P); states = sorted(set.intersection(*[set(P[c]) for c in cands]))
  res = {'n_states': len(states), 'per_cand': {}, 'paired_vs_logged': {}, 'paired_vs_start': {}}
  for c in cands:
    res['per_cand'][c] = {k: float(np.mean([P[c][s][k] for s in states])) for k in KEYS}
  for ref, lab in (('logged', 'paired_vs_logged'), ('start', 'paired_vs_start')):
    if ref not in P:
      continue
    for c in cands:
      if c == ref:
        continue
      blk = {}
      for k in KEYS:
        d = np.array([P[c][s][k] - P[ref][s][k] for s in states])
        blk[k] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))), 'share_better': float((d > 0).mean()), 'share_worse': float((d < 0).mean())}
      res[lab][c] = blk
  return res


def mode_report(args):
  OUT.mkdir(parents=True, exist_ok=True)
  Rs = np.load(MP.OUT / args.sim, allow_pickle=False); Ps, nds = per_state({k: Rs[k] for k in Rs.files if k not in ('xy_rows', 'meta')})
  res = {'sim': {'file': args.sim, 'draws_per_state': nds, **summarise(Ps)}}
  if args.model:
    Rm = np.load(MP.OUT / args.model, allow_pickle=False); Pm, ndm = per_state({k: Rm[k] for k in Rm.files if k not in ('meta', 'early', 'early_state', 'early_cand', 'early_cont', 'a0')}, cont_filter='start')
    res['model'] = {'file': args.model, 'draws_per_state': ndm, **summarise(Pm)}
    # per-state agreement of the candidate-minus-logged contrast between the simulator and the model
    agree = {}
    for c in Pm:
      if c == 'logged' or c not in Ps:
        continue
      states = sorted(set(Ps[c]) & set(Pm[c]) & set(Ps['logged']) & set(Pm['logged']))
      for k in ('success', 'goal_area', 'near2.0'):
        ds = np.array([Ps[c][s][k] - Ps['logged'][s][k] for s in states]); dm = np.array([Pm[c][s][k] - Pm['logged'][s][k] for s in states])
        agree[f'{c}|{k}'] = {'corr': float(np.corrcoef(ds, dm)[0, 1]) if ds.std() > 0 and dm.std() > 0 else None, 'sign_agreement': float(np.mean(np.sign(ds) == np.sign(dm)))}
    res['sim_vs_model_per_state'] = agree
  MP.write_json(OUT / 'report.json', res)
  L = ['# The consequences of one first action at the reset states, then the frozen start agent (paired hidden draws): simulator vs the S ETT', '',
       'Per candidate first action: the mean over states of the per-state draw means; paired contrasts vs the logged torque and vs the start agent\'s action (mean +- s.e. over states; share of states where the candidate is higher).  Masses = the gamma-law (0.999) masses of the path rows: reach 0.5 / near 2.0 / goal area / death frame -- what the critic\'s sampling law provides.', '']
  for side in ('sim', 'model'):
    if side not in res:
      continue
    r = res[side]
    L += [f"## {side} ({r['file']}; {r['n_states']} states; draws per state {r['draws_per_state']})", '', '| first action | success | death | timeout | far entry | reach0.5 | near2.0 | goal_area | death_frame |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, v in r['per_cand'].items():
      L.append(f"| {c} | {v['success']:.3f} | {v['death']:.3f} | {v['timeout']:.3f} | {v['far_entry']:.3f} | {v['reach0.5']:.4f} | {v['near2.0']:.4f} | {v['goal_area']:.4f} | {v['death_frame']:.4f} |")
    for lab in ('paired_vs_logged', 'paired_vs_start'):
      if not r[lab]:
        continue
      L += ['', f"| {lab.replace('_', ' ')}: candidate | success | timeout | goal_area mass | near2.0 mass | death_frame mass |", '|---|---|---|---|---|---|']
      for c, blk in r[lab].items():
        L.append(f"| {c} | " + ' | '.join(f"{blk[k]['mean']:+.3f} +- {blk[k]['se']:.3f} ({blk[k]['share_better']:.2f})" for k in ('success', 'timeout', 'goal_area', 'near2.0', 'death_frame')) + ' |')
    L.append('')
  if 'sim_vs_model_per_state' in res:
    L += ['## Per-state agreement of the candidate-minus-logged contrast, simulator vs model', '', '| candidate | key | corr | sign agreement |', '|---|---|---:|---:|']
    for k, v in res['sim_vs_model_per_state'].items():
      c, key = k.split('|'); L.append(f"| {c} | {key} | {'-' if v['corr'] is None else format(v['corr'], '.2f')} | {v['sign_agreement']:.2f} |")
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('report',))
  ap.add_argument('--sim', default='repeated_draws/rollouts_start_reset_stall.npz'); ap.add_argument('--model', default='ett_crossover/v4s20_reset_stall/model_rollouts.npz')
  args = ap.parse_args(argv)
  mode_report(args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
