"""A/B/C comparison for the settled-failure-bank alpha=0.1 experiment.

  A  alpha=0 clean baseline      failneg_clean_p30_h800_resetfix_a0_s0_300k
  B  alpha=0.1, LEGACY bank      failneg_clean_p30_h800_resetfix_a01_s0_300k
  C  alpha=0.1, SETTLED bank     failneg_settledbank_p30_h800_resetfix_a01_s0_300k

Collects, per run and per checkpoint (best per the existing selection rule,
plus final):
  * behavior metrics from the authoritative diagnosis
    (eval_h800_resetfix/diag_{best,final}/diagnosis.json: success, hazard
    exposure, drop rate, center/route usage, success-by-mask-pattern);
  * training diagnostics from metrics.json (positive / ordinary-negative /
    failure-negative logits, critic loss terms, actor loss) -- last entry and
    the full failure-logit trajectory for the B-vs-C distribution question.

Writes artifacts/settled_failure_bank_alpha01/comparison.json and prints the
table. Runs with whatever subset of the three runs exists (C may be pending).
"""
import json
import os

import numpy as np

RUNS = {
    'A_alpha0': 'failneg_clean_p30_h800_resetfix_a0_s0_300k',
    'B_alpha01_legacy_bank': 'failneg_clean_p30_h800_resetfix_a01_s0_300k',
    'C_alpha01_settled_bank':
        'failneg_settledbank_p30_h800_resetfix_a01_s0_300k',
}
# the synced-back SSH run may live nested under its transfer folder
_ALT = ('failneg_settledbank_a01_s0_300k/'
        'failneg_settledbank_p30_h800_resetfix_a01_s0_300k')
if not os.path.isdir(RUNS['C_alpha01_settled_bank']) and os.path.isdir(_ALT):
  RUNS['C_alpha01_settled_bank'] = _ALT
BEHAVIOR_KEYS = ('naive_success', 'naive_hazard_exposure_rate',
                 'naive_drop_rate', 'naive_center_fraction',
                 'naive_left_fraction', 'naive_right_fraction',
                 'naive_impact_recovery_rate',
                 'naive_trigger_avoidance_rate')
TRAIN_KEYS = ('logits_pos', 'logits_neg', 'logits_fail_neg', 'logits_gap',
              'critic_loss', 'critic_pos_term', 'critic_neg_ord_term',
              'critic_neg_fail_term', 'actor_loss', 'binary_accuracy',
              'policy_entropy')


def load_run(d):
  out = {'dir': d, 'present': os.path.isdir(d)}
  if not out['present']:
    return out
  mpath = os.path.join(d, 'metrics.json')
  if os.path.exists(mpath):
    m = json.load(open(mpath))
    best = max((r for r in m if r.get('success') is not None),
               key=lambda r: r['success'], default=None)
    out['train_last'] = {k: m[-1].get(k) for k in TRAIN_KEYS
                         if m[-1].get(k) is not None}
    out['best_eval'] = ({'step': best['step'], 'success': best['success']}
                        if best else None)
    fl = [(r['step'], r['logits_fail_neg']) for r in m
          if 'logits_fail_neg' in r]
    if fl:
      out['logits_fail_neg_trajectory'] = {
          'first': fl[0], 'last': fl[-1],
          'min': min(v for _, v in fl), 'max': max(v for _, v in fl)}
  for tag in ('best', 'final'):
    p = os.path.join(d, 'eval_h800_resetfix', f'diag_{tag}',
                     'diagnosis.json')
    if os.path.exists(p):
      dg = json.load(open(p))
      out[f'diag_{tag}'] = {k: dg.get(k) for k in
                            ('checkpoint', 'step', 'n_eval', 'verdict',
                             'naive_success_by_mask_pattern')
                            } | {k: dg.get(k) for k in BEHAVIOR_KEYS}
  return out


def main():
  rows = {name: load_run(d) for name, d in RUNS.items()}
  comp = {'runs': rows,
          'note': ('B vs C isolates the failure-bank representation: same '
                   'clean dataset/recipe/seed/steps/eval; only the 16 bank '
                   'states changed (legacy frozen -> N=80 settled).')}
  out_dir = 'artifacts/settled_failure_bank_alpha01'
  os.makedirs(out_dir, exist_ok=True)
  json.dump(comp, open(os.path.join(out_dir, 'comparison.json'), 'w'),
            indent=2)

  print(f"{'metric':38s}" + ''.join(f'{n[:22]:>24s}' for n in rows))
  for tag in ('best', 'final'):
    for k in BEHAVIOR_KEYS:
      vals = []
      for r in rows.values():
        v = r.get(f'diag_{tag}', {}).get(k) if r['present'] else None
        vals.append('--' if v is None else f'{v:.3f}')
      print(f'{tag}.{k:36s}'[:38] + ''.join(f'{v:>24s}' for v in vals))
  for k in ('logits_pos', 'logits_neg', 'logits_fail_neg', 'logits_gap'):
    vals = []
    for r in rows.values():
      v = r.get('train_last', {}).get(k) if r['present'] else None
      vals.append('--' if v is None else f'{v:.2f}')
    print(f'train.{k:32s}'[:38] + ''.join(f'{v:>24s}' for v in vals))
  print(f"\nwrote {os.path.join(out_dir, 'comparison.json')}")


if __name__ == '__main__':
  main()
