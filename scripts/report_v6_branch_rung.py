"""One table for a V6 branch-replay rung: every evaluated policy on the same
natural draws -- the vanilla actors, the gradient-trained actors on the
branch critics (joint 100k, frozen-critic BC sweep), and the decoding
policies (per-step rank, segment-rank with its controls) -- with the route
split, the benchmark's discounted score and the slow-shortcut audit.

  python scripts/report_v6_branch_rung.py outputs/antmaze_branch_replay_p050

Reads the driver's eval_{mean,sample}.json files and the rank/*.json files
written by the rank-policy evaluators; writes REPORT_rung.md next to them.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_v6_branch_replay import _headline  # noqa: E402


def row(label, ev):
  h = _headline(ev['summary'], ev.get('episodes'))
  rows = ev.get('episodes') or []
  det = [r for r in rows if r.get('route') == 'detour']
  sc = [r for r in rows if r.get('route') == 'shortcut']
  f = lambda v: f'{v:.3f}' if isinstance(v, (int, float)) else '--'
  det_txt = f'{np.mean([r["success"] for r in det]):.2f} / {np.mean([r["timeout"] for r in det]):.2f}' if det else '--'
  sc_txt = f'{np.mean([r["failure"] for r in sc]):.2f}' if sc else '--'
  return (f'| {label} | {ev.get("n", len(rows))} | {f(h["success_rate"])} | {f(h["failure_rate"])} | {f(h["timeout_rate"])} '
          f'| **{f(h["detour_rate"])}** | {f(h["shortcut_rate"])} | {f(h.get("discounted"))} '
          f'| {f(h.get("mouth1_median"))} / {f(h.get("mouth1_after_burst"))} | {det_txt} | {sc_txt} |')


def main(out_dir):
  out = Path(out_dir)
  L = [f'# {out.name}: every evaluated policy, natural draws at the rung density', '',
       '| policy | n | success | failure | timeout | detour | shortcut | discounted (g 0.99) | mouth 1 median step / after-burst share '
       '| detour episodes: success / timeout | shortcut episodes: failure |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|']
  groups = [
      ('vanilla g0.999 (recorded data)', 'vanilla_g0999/seed_*/eval_{pol}.json'),
      ('joint 100k on the 100k branch critic', 'joint/seed_*/eval_{pol}.json'),
      ('frozen 30k critic, actor bc 0.05', 'joint_frozen30k/seed_*/eval_{pol}.json'),
      ('frozen 30k critic, actor bc 0.5', 'joint_frozen30k_bc0.5/seed_*/eval_{pol}.json'),
      ('pure BC (bc 1.0)', 'joint_frozen30k_bc1.0/seed_*/eval_{pol}.json'),
  ]
  for label, pat in groups:
    for pol in ('mean', 'sample'):
      for p in sorted(glob.glob(str(out / pat.format(pol=pol)))):
        seed = p.split('seed_')[1].split(os.sep)[0].split('/')[0]
        L.append(row(f'{label}, seed {seed}, {pol}', json.load(open(p, encoding='utf-8'))))
  rank_files = sorted(glob.glob(str(out / 'rank' / '*.json')))
  for p in rank_files:
    ev = json.load(open(p, encoding='utf-8'))
    crit = Path(ev.get('critic_ckpt', '')).parent
    label = f'{ev.get("policy")} | critic {crit.parent.name}/{crit.name}'
    if ev.get('random_pick'):
      label = f'{ev.get("policy")} | no critic (uniform pick)'
    L.append(row(label, ev))
  L += ['', 'mode = tanh(loc), the repo and the paper\'s evaluation convention; sample = tanh(loc + scale eps), secondary.  '
        'rank policies: K proposals scored by the critic, argmax (per-step: K tanh-normal samples of the BC actor; '
        'segrank: K recorded 25-step torque segments at the start, run open-loop, BC mode elsewhere).  '
        'The rockfall clocks run from the reset (zone 1 closes by step 117): an after-burst share near one means '
        'the shortcut survives by lateness.']
  (out / 'REPORT_rung.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L))


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/antmaze_branch_replay_p050')
