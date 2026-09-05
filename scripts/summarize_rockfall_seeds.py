"""Seed-pooled summary of the rockfall benchmarks (V3 two-route, V4 wait,
V5 clock).

Reads the per-run eval JSONs (episode rows included) and reports every
headline rate two ways: per seed, then mean +- sd across seeds, and pooled
over all episodes with a Wilson 95% interval. Reference numbers (sighted
expert, always-go / always-wait blind policies) come from the causal audit
stored in the eval file.

    python scripts/summarize_rockfall_seeds.py --bench v4      # mean + critic_select
    python scripts/summarize_rockfall_seeds.py --bench v3br
    python scripts/summarize_rockfall_seeds.py --bench v5      # grouped near / far05

Writes <bench root>/seed_summary.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOTS = {'v4': 'artifacts/rockfall_wait_v4',
         'v3br': 'artifacts/tworoute_rockfall_v3/br',
         'v5': 'artifacts/rockfall_clock_v5'}
#: V5 datasets: 'near' (P_FAR = 0, shortcut only) and 'far05' (P_FAR = 0.05,
#: one detour in twenty for coverage).  Runs are grouped by this label, read
#: from the run id ``v5clock_{variant}_{method}_s{seed}_{steps}k``.
V5_VARIANTS = ('near', 'far05')
HESITATION_HOLD = 60      # steps mouth -> band that count as "waited"
STOPPED_STEPS = 10        # near-zero-speed steps in the zone that count as "stopped"


def wilson(k, n, z=1.96):
  if n == 0:
    return [None, None]
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return [round(float(c - h), 4), round(float(c + h), 4)]


def _r(x, nd=4):
  return None if x is None else round(float(x), nd)


def rate(rows, key, cond=None):
  xs = [bool(r[key]) for r in rows if (cond is None or cond(r))
        and r.get(key) is not None]
  return (sum(xs), len(xs))


def mean_of(rows, key, cond=None):
  xs = [r[key] for r in rows if (cond is None or cond(r))
        and r.get(key) is not None]
  return _r(np.mean(xs)) if xs else None


def is_active(r):
  return bool(r['u'])


def is_clear(r):
  return not r['u']


# --------------------------------------------------------------------- V4
def v4_metrics(rows):
  aug = []
  for r in rows:
    h = r.get('hesitation')
    m = r.get('mouth')
    aug.append(dict(
        r, timeout=(not r['success'] and not r['failure']),
        waited=(h is not None and h >= HESITATION_HOLD),
        stopped=(r['stop_steps'] >= STOPPED_STEPS),
        q_mode_gt_q_zero=(None if m is None else m['q_mode'] > m['q_zero']),
        q_mode=(None if m is None else m['q_mode']),
        q_zero=(None if m is None else m['q_zero']),
        sigma=(None if m is None else m['sigma_mean'])))
  hes_act = [r['hesitation'] for r in rows if r['u']
             and r['hesitation'] is not None]
  hes_clr = [r['hesitation'] for r in rows if not r['u']
             and r['hesitation'] is not None]
  return {
      'rates': {
          'success': rate(aug, 'success'),
          'death': rate(aug, 'failure'),
          'timeout': rate(aug, 'timeout'),
          'death_given_active': rate(aug, 'failure', is_active),
          'success_given_active': rate(aug, 'success', is_active),
          'success_given_clear': rate(aug, 'success', is_clear),
          'death_given_clear': rate(aug, 'failure', is_clear),
          'entered_band': rate(aug, 'entered_hazard'),
          'entered_while_open_given_active': rate(aug, 'entered_while_open',
                                                  is_active),
          'waited_rate': rate(aug, 'waited'),
          'waited_rate_given_active': rate(aug, 'waited', is_active),
          'stopped_rate': rate(aug, 'stopped'),
          'stopped_rate_given_active': rate(aug, 'stopped', is_active),
          'mouth_q_mode_gt_q_zero': rate(aug, 'q_mode_gt_q_zero')},
      'means': {
          'hesitation_active': _r(np.mean(hes_act), 1) if hes_act else None,
          'hesitation_clear': _r(np.mean(hes_clr), 1) if hes_clr else None,
          'stop_steps': mean_of(aug, 'stop_steps'),
          'stop_steps_active': mean_of(aug, 'stop_steps', is_active),
          'zone_vmin': mean_of(aug, 'zone_vmin'),
          'discounted': mean_of(aug, 'discounted'),
          'discounted_active': mean_of(aug, 'discounted', is_active),
          'discounted_clear': mean_of(aug, 'discounted', is_clear),
          'steps': mean_of(aug, 'steps'),
          'mouth_q_mode': mean_of(aug, 'q_mode'),
          'mouth_q_zero': mean_of(aug, 'q_zero'),
          'mouth_sigma': mean_of(aug, 'sigma')}}


def v4_load(root):
  out = {}
  for mode in ('mean', 'critic_select'):
    runs = []
    for f in sorted(glob.glob(os.path.join(
        root, 'runs', '*', f'eval_rockfall_wait_v4_{mode}.json'))):
      d = json.load(open(f))
      runs.append({'run': os.path.basename(os.path.dirname(f)),
                   'ckpt_step': d['ckpt_step'], 'n': d['n_eval'],
                   'refs': d.get('reference_numbers'),
                   'episodes': d['episodes']})
    if runs:
      out[mode] = runs
  return out


# --------------------------------------------------------------------- V3
def v3_metrics(rows):
  aug = [dict(r, timeout=(not r['success'] and not r['failure']),
              shortcut=(r['route'] == 'shortcut'),
              detour=(r['route'] == 'detour'),
              no_route=(r['route'] is None)) for r in rows]

  def active_shortcut(r):
    return r['u'] and r['shortcut']

  return {
      'rates': {
          'success': rate(aug, 'success'),
          'death': rate(aug, 'failure'),
          'timeout': rate(aug, 'timeout'),
          'shortcut_rate': rate(aug, 'shortcut'),
          'detour_rate': rate(aug, 'detour'),
          'no_route_rate': rate(aug, 'no_route'),
          'death_given_active': rate(aug, 'failure', is_active),
          'death_given_active_and_shortcut': rate(aug, 'failure',
                                                  active_shortcut),
          'success_given_clear': rate(aug, 'success', is_clear),
          'detour_given_active': rate(aug, 'detour', is_active),
          'detour_given_clear': rate(aug, 'detour', is_clear)},
      'means': {
          'discounted': mean_of(aug, 'discounted'),
          'discounted_active': mean_of(aug, 'discounted', is_active),
          'discounted_clear': mean_of(aug, 'discounted', is_clear),
          'steps': mean_of(aug, 'steps')}}


def v3_load(root):
  out = {}
  runs = []
  for f in sorted(glob.glob(os.path.join(root, 'runs', '*',
                                         'eval_tworoute_v3.json'))):
    run = os.path.basename(os.path.dirname(f))
    if run.endswith('_g999'):
      continue
    d = json.load(open(f))
    runs.append({'run': run, 'ckpt_step': d['ckpt_step'], 'n': d['n_eval'],
                 'refs': d.get('reference_numbers'),
                 'episodes': d['episodes']})
  if runs:
    out['mean'] = runs
  runs = []
  for f in sorted(glob.glob(os.path.join(root, 'critic_select_*_n300.json'))):
    d = json.load(open(f))
    eps = d['episodes_critic_selected']
    for e in eps:                       # critic-select rows carry no discount
      e.setdefault('discounted', 0.99 ** e['steps'] if e['success'] else 0.0)
    runs.append({'run': os.path.basename(os.path.dirname(d['ckpt'])),
                 'ckpt_step': d['step'], 'n': d['n'], 'refs': None,
                 'episodes': eps})
  if runs:
    out['critic_select'] = runs
  return out


# --------------------------------------------------------------------- V5
#: V5 (rockfall on its own clock) evaluates like V4 -- the same wait/stop/
#: mouth-critic readouts -- but the far05 dataset gives the learner a second
#: route, so the V3 route rates come back: which route the agent takes, and
#: whether it dies on the shortcut specifically.  Runs are grouped by dataset
#: variant because the two variants answer different questions (near: does
#: the learner wait; far05: does it blend go / wait / detour), so their seeds
#: must never be pooled together.
def v5_variant(run, default=None):
  """Dataset variant of a V5 run id (``v5clock_{variant}_...``)."""
  parts = run.split('_')
  if len(parts) >= 2 and parts[0] == 'v5clock' and parts[1] in V5_VARIANTS:
    return parts[1]
  for v in V5_VARIANTS:
    if f'_{v}_' in f'_{run}_':
      return v
  return default


def v5_metrics(rows):
  m = v4_metrics(rows)
  aug = [dict(r, shortcut=(r.get('route') == 'shortcut'),
              detour=(r.get('route') == 'detour'),
              no_route=(r.get('route') is None)) for r in rows]

  def active_shortcut(r):
    return r['u'] and r['shortcut']

  m['rates'].update({
      'shortcut_rate': rate(aug, 'shortcut'),
      'detour_rate': rate(aug, 'detour'),
      'no_route_rate': rate(aug, 'no_route'),
      'shortcut_given_active': rate(aug, 'shortcut', is_active),
      'detour_given_active': rate(aug, 'detour', is_active),
      'detour_given_clear': rate(aug, 'detour', is_clear),
      'death_given_active_and_shortcut': rate(aug, 'failure',
                                              active_shortcut)})
  return m


def v5_load(root):
  """{variant: {mode: [runs]}}; a run's variant comes from its id, falling
  back to the eval file's own 'variant' field."""
  out = {}
  for mode in ('mean', 'critic_select'):
    for f in sorted(glob.glob(os.path.join(
        root, 'runs', '*', f'eval_rockfall_clock_v5_{mode}.json'))):
      run = os.path.basename(os.path.dirname(f))
      d = json.load(open(f))
      variant = v5_variant(run, d.get('variant'))
      if variant is None:
        print(f'  skip {f}: cannot infer variant (near / far05)')
        continue
      out.setdefault(variant, {}).setdefault(mode, []).append(
          {'run': run, 'ckpt_step': d['ckpt_step'], 'n': d['n_eval'],
           'refs': d.get('reference_numbers'),
           'episodes': d['episodes']})
  return out


# ------------------------------------------------------------------ pooling
def pool(runs, metrics_fn):
  per_seed = {r['run']: metrics_fn(r['episodes']) for r in runs}
  all_rows = [e for r in runs for e in r['episodes']]
  pooled = metrics_fn(all_rows)
  out = {'n_seeds': len(runs), 'runs': [r['run'] for r in runs],
         'n_episodes_pooled': len(all_rows), 'rates': {}, 'means': {}}
  for key in pooled['rates']:
    ps = [k / n for k, n in (per_seed[s]['rates'][key] for s in per_seed)
          if n]
    k, n = pooled['rates'][key]
    out['rates'][key] = {
        'per_seed': [_r(p) for p in ps],
        'seed_mean': _r(np.mean(ps)) if ps else None,
        'seed_sd': _r(np.std(ps, ddof=1)) if len(ps) > 1 else None,
        'pooled': _r(k / n) if n else None, 'pooled_n': n,
        'pooled_ci95': wilson(k, n)}
  for key in pooled['means']:
    vals = [per_seed[s]['means'][key] for s in per_seed]
    vs = [v for v in vals if v is not None]
    out['means'][key] = {
        'per_seed': vals,
        'seed_mean': _r(np.mean(vs)) if vs else None,
        'seed_sd': _r(np.std(vs, ddof=1)) if len(vs) > 1 else None,
        'pooled': pooled['means'][key]}
  return out


def fmt_rate(b):
  if b['seed_mean'] is None:
    return '   n/a'
  sd = f' +- {b["seed_sd"]:.3f}' if b['seed_sd'] is not None else ''
  lo, hi = b['pooled_ci95']
  return (f'{b["seed_mean"]:.3f}{sd:10s}  pooled {b["pooled"]:.3f} '
          f'[{lo:.3f}, {hi:.3f}]  per-seed '
          + ' '.join(f'{p:.2f}' for p in b['per_seed']))


def fmt_mean(b):
  if b['seed_mean'] is None:
    return '   n/a'
  sd = f' +- {b["seed_sd"]:.3f}' if b['seed_sd'] is not None else ''
  return (f'{b["seed_mean"]:.3f}{sd:10s}  per-seed '
          + ' '.join('n/a' if p is None else f'{p:.2f}'
                     for p in b['per_seed']))


def print_modes(bench, data, metrics, label=None):
  """Pool every mode of one bench (or one V5 variant); returns {mode: pooled}."""
  modes = {}
  head = bench if label is None else f'{bench} | {label}'
  for mode, runs in data.items():
    p = pool(runs, metrics)
    modes[mode] = p
    print(f'\n=== {head} | {mode} | seeds {p["runs"]} | '
          f'{p["n_episodes_pooled"]} episodes ===')
    print('  rates (seed mean +- sd | pooled [Wilson 95%] | per seed)')
    for key, b in p['rates'].items():
      print(f'    {key:36s} {fmt_rate(b)}')
    print('  means')
    for key, b in p['means'].items():
      print(f'    {key:36s} {fmt_mean(b)}')
  return modes


def main_v5(root):
  data = v5_load(root)
  if not data:
    sys.exit(f'no eval files under {root}')
  report = {'bench': 'v5', 'root': root, 'variants': {}}
  for variant in [v for v in V5_VARIANTS if v in data]:
    refs = next((r['refs'] for m in data[variant].values() for r in m
                 if r['refs']), None)
    report['variants'][variant] = {
        'reference_numbers': refs,
        'modes': print_modes('v5', data[variant], v5_metrics, variant)}
    if refs:
      print(f'\n  reference numbers ({variant}):', json.dumps(refs))
  out = os.path.join(root, 'seed_summary.json')
  with open(out, 'w') as f:
    json.dump(report, f, indent=2)
  print('->', out)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--bench', choices=sorted(ROOTS), required=True)
  ap.add_argument('--root', default=None)
  args = ap.parse_args()
  root = args.root or ROOTS[args.bench]
  if args.bench == 'v5':
    main_v5(root)
    return
  loader, metrics = ((v4_load, v4_metrics) if args.bench == 'v4'
                     else (v3_load, v3_metrics))
  data = loader(root)
  if not data:
    sys.exit(f'no eval files under {root}')
  refs = next((r['refs'] for m in data.values() for r in m if r['refs']),
              None)
  report = {'bench': args.bench, 'root': root, 'reference_numbers': refs,
            'modes': {}}
  for mode, runs in data.items():
    p = pool(runs, metrics)
    report['modes'][mode] = p
    print(f'\n=== {args.bench} | {mode} | seeds {p["runs"]} | '
          f'{p["n_episodes_pooled"]} episodes ===')
    print('  rates (seed mean +- sd | pooled [Wilson 95%] | per seed)')
    for key, b in p['rates'].items():
      print(f'    {key:36s} {fmt_rate(b)}')
    print('  means')
    for key, b in p['means'].items():
      print(f'    {key:36s} {fmt_mean(b)}')
  if refs:
    print('\n  reference numbers:', json.dumps(refs))
  out = os.path.join(root, 'seed_summary.json')
  with open(out, 'w') as f:
    json.dump(report, f, indent=2)
  print('->', out)


if __name__ == '__main__':
  main()
