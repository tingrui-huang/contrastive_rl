"""Vanilla offline CRL on one rung of the V6 teacher-detour ladder, N seeds.

Plain baseline only: ``run_v6_failneg.py --alpha 0.0`` (no failure bank is
loaded; the failure-negative branch is skipped entirely), the frozen V6
recipe (bc_coef 0.05, batch 1024, twin-Q, XY goals), a fixed update budget,
the final checkpoint, and the authoritative natural-draw evaluation at a
fixed reset seed under BOTH evaluation policies:

  mean     deterministic tanh(loc)            (the repository headline)
  sample   tanh-normal samples, fixed action seed

reporting success / failure / timeout and the shortcut / detour route rates.

Usage::

  python scripts/run_v6_ladder_baseline.py all --rung 0.05 --seeds 0 1 2
  python scripts/run_v6_ladder_baseline.py aggregate
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import train_rockfall_clock_v6_baseline as B          # noqa: E402

OUT_ROOT = os.path.join(ROOT, 'artifacts', 'v6_ladder_baseline')
STEPS = 100_000            # the budget of every V6 sweep in notes/
EVAL_N = 300               # natural-draw episodes, reset seed 909 (as in notes/)
EVAL_SEED = 909
ACTION_SEED = 9909
POLICIES = ('mean', 'sample')
RUN_ROOT = os.path.join(ROOT, 'artifacts', 'v6_failneg', 'runs')


def run(cmd, log):
  os.makedirs(os.path.dirname(log), exist_ok=True)
  print('  $', ' '.join(cmd), '->', os.path.relpath(log, ROOT), flush=True)
  with open(log, 'w', encoding='utf-8') as handle:
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle,
                   stderr=subprocess.STDOUT)


def run_dir(rung, seed, steps):
  """The launcher's own naming, so its manifests are where it expects."""
  return os.path.join(
      RUN_ROOT, 'v6fn_a0_s%d_%dk_gxy_p0.4-0.4_h800%s'
      % (seed, steps // 1000, B.run_detour_suffix(rung)))


def train(rung, seed, steps, log_dir):
  d = run_dir(rung, seed, steps)
  if os.path.isfile(os.path.join(d, 'final.pkl')):
    print(f'seed {seed}: final.pkl exists', flush=True)
    return d
  run([sys.executable, 'scripts/run_v6_failneg.py', '--alpha', '0.0',
       '--seed', str(seed), '--steps', str(steps),
       '--teacher-detour-prob', '%g' % rung],
      os.path.join(log_dir, f'train_s{seed}.log'))
  if not os.path.isfile(os.path.join(d, 'final.pkl')):
    raise RuntimeError(f'no final.pkl in {d}')
  return d


def evaluate(d, seed, n, log_dir):
  out = {}
  for policy in POLICIES:
    path = os.path.join(d, f'eval_rockfall_clock_v6_{policy}.json')
    if not os.path.isfile(path):
      run([sys.executable, 'scripts/eval_rockfall_clock_v6_baseline.py',
           '--ckpt', os.path.join(d, 'final.pkl'), '--n', str(n),
           '--seed', str(EVAL_SEED), '--policy', policy,
           '--action-seed', str(ACTION_SEED), '--out-dir', d],
          os.path.join(log_dir, f'eval_s{seed}_{policy}.log'))
    with open(path, encoding='utf-8') as f:
      out[policy] = json.load(f)
  return out


def row(record):
  o = record['summary']['overall']
  routes = o['routes']
  return {
      'success': o['success'], 'failure': o['failure'],
      'timeout': o['timeout'],
      'shortcut': routes['shortcut']['rate'],
      'detour': routes['detour']['rate'],
      'no_route': routes['no_route']['rate'],
      'mean_steps': o.get('mean_steps'),
      'ckpt_step': record['ckpt_step'], 'n': record['n_eval'],
  }


def summarize(rung, seeds, steps, n, per_seed, out_dir):
  keys = ('success', 'failure', 'timeout', 'shortcut', 'detour', 'no_route')
  mean = {p: {k: sum(per_seed[s][p][k] for s in seeds) / len(seeds)
              for k in keys} for p in POLICIES}
  result = {'teacher_detour_prob': rung, 'seeds': list(seeds),
            'steps': steps, 'bc_coef': 0.05, 'fail_neg_alpha': 0.0,
            'eval_n': n, 'eval_seed': EVAL_SEED, 'action_seed': ACTION_SEED,
            'per_seed': {str(s): per_seed[s] for s in seeds},
            'mean_over_seeds': mean}
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
  lines = [f'# V6 ladder baseline, teacher detour {rung:g}: vanilla CRL, '
           f'bc 0.05, alpha 0, {steps // 1000}k updates, final checkpoint, '
           f'n={n} natural draws at reset seed {EVAL_SEED}', '',
           '| seed | policy | success | failure | timeout | shortcut | detour |',
           '|---|---|---:|---:|---:|---:|---:|']
  for s in seeds:
    for p in POLICIES:
      x = per_seed[s][p]
      lines.append(f'| {s} | {p} | {x["success"]:.3f} | {x["failure"]:.3f} | '
                   f'{x["timeout"]:.3f} | {x["shortcut"]:.3f} | '
                   f'{x["detour"]:.3f} |')
  for p in POLICIES:
    x = mean[p]
    lines.append(f'| mean | {p} | {x["success"]:.3f} | {x["failure"]:.3f} | '
                 f'{x["timeout"]:.3f} | {x["shortcut"]:.3f} | '
                 f'{x["detour"]:.3f} |')
  with open(os.path.join(out_dir, 'REPORT.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
  print('\n'.join(lines), flush=True)


def aggregate():
  rungs = sorted(d for d in os.listdir(OUT_ROOT)
                 if d.startswith('far') and os.path.isfile(
                     os.path.join(OUT_ROOT, d, 'results.json')))
  if not rungs:
    raise SystemExit(f'no completed rungs under {OUT_ROOT}')
  lines = ['# V6 ladder baseline: vanilla CRL across teacher-detour rungs',
           '', 'Mean over seeds; n natural-draw episodes per policy at one '
           'fixed reset seed; mean = deterministic tanh(loc), sample = '
           'tanh-normal samples with a fixed action seed.', '',
           '| rung | seeds | policy | success | failure | timeout | shortcut '
           '| detour |', '|---|---|---|---:|---:|---:|---:|---:|']
  per = ['', 'Per seed:', '',
         '| rung | seed | policy | success | failure | timeout | shortcut '
         '| detour |', '|---|---|---|---:|---:|---:|---:|---:|']
  for d in rungs:
    with open(os.path.join(OUT_ROOT, d, 'results.json'), encoding='utf-8') as f:
      r = json.load(f)
    for p in POLICIES:
      x = r['mean_over_seeds'][p]
      lines.append(f'| {r["teacher_detour_prob"]:g} | {len(r["seeds"])} | {p} '
                   f'| {x["success"]:.3f} | {x["failure"]:.3f} | '
                   f'{x["timeout"]:.3f} | {x["shortcut"]:.3f} | '
                   f'{x["detour"]:.3f} |')
    for s, arms in r['per_seed'].items():
      for p in POLICIES:
        x = arms[p]
        per.append(f'| {r["teacher_detour_prob"]:g} | {s} | {p} | '
                   f'{x["success"]:.3f} | {x["failure"]:.3f} | '
                   f'{x["timeout"]:.3f} | {x["shortcut"]:.3f} | '
                   f'{x["detour"]:.3f} |')
  text = '\n'.join(lines + per) + '\n'
  with open(os.path.join(OUT_ROOT, 'REPORT.md'), 'w', encoding='utf-8') as f:
    f.write(text)
  print(text)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('all', 'aggregate'))
  ap.add_argument('--rung', type=float, default=None)
  ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--n', type=int, default=EVAL_N)
  args = ap.parse_args(argv)
  if args.command == 'aggregate':
    aggregate()
    return 0
  if args.rung is None:
    raise SystemExit('--rung is required')
  rung = float(args.rung)
  tag = B.detour_tag(rung)
  out_dir = os.path.join(OUT_ROOT, tag)
  log_dir = os.path.join(out_dir, 'logs')
  per_seed = {}
  for seed in args.seeds:
    started = time.time()
    d = train(rung, seed, args.steps, log_dir)
    records = evaluate(d, seed, args.n, log_dir)
    per_seed[seed] = {p: row(records[p]) for p in POLICIES}
    per_seed[seed]['run_dir'] = os.path.relpath(d, ROOT)
    print(f'rung {rung:g} seed {seed} done in {time.time() - started:.0f}s: '
          + ' | '.join(f'{p} success {per_seed[seed][p]["success"]:.3f} '
                       f'detour {per_seed[seed][p]["detour"]:.3f}'
                       for p in POLICIES), flush=True)
  summarize(rung, args.seeds, args.steps, args.n, per_seed, out_dir)
  return 0


if __name__ == '__main__':
  sys.exit(main())
