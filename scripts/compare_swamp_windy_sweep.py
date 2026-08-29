"""Aggregate the windy-swamp alpha sweep across arms and seeds.

The sweep launcher prints a flat per-run table; this collapses it to the thing
the experiment is actually about: does restricting anchors (scheme C) and/or
adding failure-state negatives move the learner off the confounded shortcut?

Pre-registered read-out, in order of importance:

  worst_case   success under all_active (bits frozen [1,1,1], corridor entry is
               instant death). The naive reference scores 0.0 here while an
               always-safe policy scores 1.0 -- this is the number the whole
               setup exists to move.
  entry        fraction of episodes that enter the swamp corridor at all. The
               mechanism by which worst_case can improve is route switching, so
               a worst_case gain WITHOUT an entry drop means something else
               happened and should be treated as suspicious.
  natural      success under the env's real per-step resampling.
  died         death rate under natural.

Seeds are aggregated as mean +/- sample std. With 3 seeds these are wide; the
alpha trend across arms matters more than any single cell.

Usage:
  python scripts/compare_swamp_windy_sweep.py
  python scripts/compare_swamp_windy_sweep.py --root artifacts --json out.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ORDER = ['baseline', 'anchorcut', 'failneg_a005', 'failneg_a01', 'failneg_a02',
         'balanced', 'balancedfail_a005', 'balancedfail_a01',
         'balancedfail_a02']
LABEL = {
    'baseline': 'baseline (no cut, a=0)',
    'anchorcut': 'schemeC (a=0)',
    'failneg_a005': 'schemeC + bank a=0.05',
    'failneg_a01': 'schemeC + bank a=0.10',
    'failneg_a02': 'schemeC + bank a=0.20',
    'balanced': 'schemeC+balanced (a=0)',
    'balancedfail_a005': 'schemeC+bal + bank a=0.05',
    'balancedfail_a01': 'schemeC+bal + bank a=0.10',
    'balancedfail_a02': 'schemeC+bal + bank a=0.20',
}
FIELDS = [('all_active', 'success', 'worst_case'),
          ('natural', 'success', 'natural'),
          ('natural', 'entry', 'entry'),
          ('natural', 'died', 'died'),
          ('all_clear', 'success', 'clear')]


def collect(root):
  runs = {}
  pat = os.path.join(root, 'swamp_windy_*_s[0-9]', 'deployment_report.json')
  for p in sorted(glob.glob(pat)):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r'swamp_windy_(.+)_s(\d+)$', name)
    if not m:
      continue
    arm, seed = m.group(1), int(m.group(2))
    try:
      with open(p) as f:
        d = json.load(f)
    except (OSError, json.JSONDecodeError):
      continue
    L = d.get('learner', {})
    row = {out: L.get(cond, {}).get(key) for cond, key, out in FIELDS}
    row['verdict'] = d.get('verdict')
    row['safe_ref'] = (d.get('always_safe', {})
                       .get('all_active', {}).get('success'))
    runs.setdefault(arm, {})[seed] = row
  return runs


def agg(vals):
  v = [x for x in vals if isinstance(x, (int, float))]
  if not v:
    return None, None, 0
  return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def load_critic(path):
  """Optional critic-probe output, keyed by run dir name.

  Merged in because the decisive question here is not deployment success alone
  but whether the CRITIC and the POLICY agree: scheme C was measured to flip
  the critic's fork preference while leaving corridor entry at 1.00. Reading
  the two tables side by side is what makes that visible.
  """
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, json.JSONDecodeError):
    return {}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--root', default='artifacts')
  ap.add_argument('--json', default='')
  ap.add_argument('--critic', default='artifacts/swamp_windy_failure_bank/'
                                      'critic_fork_margin.json',
                  help='output of probe_swamp_windy_critic.py; merged if present')
  args = ap.parse_args()

  runs = collect(args.root)
  if not runs:
    raise SystemExit(f'no deployment_report.json found under {args.root}/'
                     'swamp_windy_*_s<seed>/')

  arms = [a for a in ORDER if a in runs] + \
         [a for a in sorted(runs) if a not in ORDER]

  cols = ['worst_case', 'entry', 'natural', 'died', 'clear']
  print('=' * 92)
  print('WINDY-SWAMP ALPHA SWEEP  (mean +/- std over seeds; '
        'worst_case = success under all_active)')
  print('=' * 92)
  W = 15                       # must fit "  m.mm+-s.ss" without truncation
  hdr = f'{"arm":<26}{"n":>3}' + ''.join(f'{c:>{W}}' for c in cols)
  print(hdr)
  print('-' * len(hdr))

  table = {}
  base_wc = None
  for a in arms:
    seeds = runs[a]
    cells, line = {}, f'{LABEL.get(a, a):<26}{len(seeds):>3}'
    for c in cols:
      m, s, n = agg([seeds[k].get(c) for k in seeds])
      cells[c] = {'mean': m, 'std': s, 'n': n}
      line += f'{"-":>{W}}' if m is None else f'{m:.2f}+-{s:.2f}'.rjust(W)
    table[a] = cells
    if a == 'baseline':
      base_wc = cells['worst_case']['mean']
    print(line)

  safe = [seeds[k].get('safe_ref') for a in arms for seeds in [runs[a]]
          for k in seeds]
  sm, _, sn = agg(safe)
  if sm is not None:
    print('-' * len(hdr))
    print(f'{"always-safe reference":<26}{sn:>3}'
          + f'{sm:.2f}'.rjust(W) + f'{0.00:.2f}'.rjust(W)
          + '-'.rjust(W) + f'{0.00:.2f}'.rjust(W) + '-'.rjust(W))

  # ---- critic layer, merged in so the dissociation is visible per arm ----
  crit = load_critic(args.critic)
  if crit:
    byarm = {}
    for name, r in crit.items():
      m = re.match(r'swamp_windy_(.+)_s\d+$', name)
      if m:
        byarm.setdefault(m.group(1), []).append(r)
    if byarm:
      order = [a for a in ORDER if a in byarm] + \
              [a for a in sorted(byarm) if a not in ORDER]
      print()
      print('CRITIC LAYER (actor-independent; from probe_swamp_windy_critic.py)')
      h = (f'{"arm":<26}{"n":>3}{"fork_margin":>16}{"f(bank as goal)":>18}'
           f'{"seeds preferring":>20}')
      print(h)
      print('-' * len(h))
      for a in order:
        rs = byarm[a]
        fm = np.mean([x['fork_margin'] for x in rs])
        fs = np.std([x['fork_margin'] for x in rs], ddof=1) if len(rs) > 1 else 0.0
        vb = np.mean([x.get('v_bank_as_goal', np.nan) for x in rs])
        # Report how many SEEDS prefer each side, not one seed's argmax
        # vector: mixing a per-seed argmax with a cross-seed mean margin reads
        # as a contradiction (a safe-ward argmax next to a negative mean).
        n_safe = sum(1 for x in rs if x['fork_margin'] > 0)
        pref = f'{n_safe}/{len(rs)} seeds safe'
        print(f'{LABEL.get(a, a):<26}{len(rs):>3}'
              + f'{fm:+.3f}+-{fs:.3f}'.rjust(16)
              + f'{vb:.3f}'.rjust(18) + pref.rjust(20))
      print('\n  READ THE +- COLUMN FIRST. Measured on seeds 0+1, the fork'
            ' margin is\n  SEED-dominated: every scheme-C arm was positive on'
            ' seed 0 and negative on\n  seed 1, with the between-seed gap'
            ' larger than the whole alpha range. A\n  margin whose std is'
            ' comparable to its mean says nothing about the arm.\n'
            '  f(bank as goal) is the trustworthy critic column -- it fell'
            ' monotonically\n  with alpha on BOTH seeds (-2.2 and -2.6 logits'
            ' at alpha=0.2), i.e. the bank\n  reliably moves the quantity it'
            ' TARGETS while `entry` stays at 1.00.')

  print()
  print('READ-OUT')
  if base_wc is not None:
    print(f'  baseline worst_case = {base_wc:.2f}'
          + ('   (confounded shortcut reproduced)' if base_wc < 0.2 else ''))
    for a in arms:
      if a == 'baseline':
        continue
      m = table[a]['worst_case']['mean']
      e0 = table['baseline']['entry']['mean']
      e = table[a]['entry']['mean']
      if m is None:
        continue
      d = m - base_wc
      note = ''
      if d > 0.1 and e is not None and e0 is not None:
        note = ('  <- entry dropped %.2f, consistent with a route switch'
                % (e0 - e)) if (e0 - e) > 0.1 else \
               '  <- WARNING: worst_case rose WITHOUT an entry drop; ' \
               'inspect before believing it'
      print(f'  {LABEL.get(a, a):<26} worst_case {m:.2f}  '
            f'(delta {d:+.2f}){note}')
  print()
  print('  Per-seed spread matters: with n=3 a 0.1 shift is inside the noise.')
  print('  A real effect should be monotone-ish in alpha AND show the entry drop.')

  if args.json:
    with open(args.json, 'w') as f:
      json.dump({'table': table, 'per_run': runs}, f, indent=2)
    print(f'\nwrote {args.json}')


if __name__ == '__main__':
  main()
