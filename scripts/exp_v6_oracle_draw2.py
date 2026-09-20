#!/usr/bin/env python
"""AntMaze V6 -- a SECOND DRAW of the sealed simulator branch table (2026-09-20, after the hybrid control).

The hybrid control showed that the hybrid table equals the sealed simulator table up to the death cut (paths row-identical, cell-level
death rates within the simulator's own split-half noise, the NCE positive distribution identical to 0.003) and still trains policies
0.341 vs 0.449.  The critic sees a future row only as its xy goal (exp_v6_mainline_pilot.BranchFutures.goal_at), so the impact death
frame cannot be what it reads (the xy and the contact-perturbed continuations are not thereby excluded).  Before anything else the
oracle reference has to be replicated under a fresh future-table draw: the five oracle seeds share ONE hazard draw per anchor
(HAZARD_SEED0 + anchor_id), so "5 / 5 seeds" checked the training and evaluation randomness, not the generation's.  This driver
regenerates the simulator branches with
fresh hazard / clock / jitter draws (HAZARD_SEED0_2 + anchor_id; everything else identical: anchors, the logged torque once, the start
agent's mode, termination, no filtering) and trains the same five seeds with the same recipe.
  generate  -> oracle_draw2/branches_cf_draw2.npz + generation_draw2.json (vs the sealed table: shares, row identity on anchors inactive in both draws)
  seal / train --seed s / evaluate / report  (the ETT-pipeline machinery; arm label CF2; runs under oracle_draw2/CF/seed_s)
Reading (the user's rule, fixed before stage 2): this is a REPLICATION CHECK of the oracle reference under a fresh future-table draw --
two tables cannot estimate the table-draw variance, and the seed s.e. of CF - O (0.024) is NOT a line for "the same oracle".  Report
CF2 - CF per paired seed with its seed s.e., whether CF2 - O still carries the gain, and where the hybrid sits relative to BOTH oracle
tables.  Three readings kept open: (i) CF2 also near 0.45 and stably above the hybrid -> stronger evidence that the learned risk /
conditional futures are biased (not yet localised to the death cut); (ii) CF2 down near the hybrid -> table-draw sensitivity comes
first, and the 0.45 has to be restated as the result under one particular oracle table; (iii) in between or too uncertain -> not
separable, not forced into either.
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
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'oracle_draw2'
BRANCHES = OUT / 'branches_cf_draw2.npz'
HAZARD_SEED0_2 = 232_000_000       # the sealed table: 132_000_000 + anchor_id
GEN_WORKER_SEED0_2 = 233_000_000   # the sealed table: 133_000_000 + part
SEEDS, EVAL, OVERRIDES = EF.SEEDS, EF.EVAL, EF.OVERRIDES
HYBRID = MP.OUT / 'ett_futures_hybrid'


def run_dir2(s):
  return MP.run_dir('CF', s, OUT)


def _ks(a, b):
  if len(a) < 20 or len(b) < 20:
    return None
  x = np.sort(np.concatenate([a, b])); fa = np.searchsorted(np.sort(a), x, 'right') / len(a); fb = np.searchsorted(np.sort(b), x, 'right') / len(b)
  return float(np.abs(fa - fb).max())


def mode_generate(args):
  if BRANCHES.exists() and not args.force:
    print(f'{BRANCHES} exists', flush=True); return
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  MP.HAZARD_SEED0, MP.GEN_WORKER_SEED0 = HAZARD_SEED0_2, GEN_WORKER_SEED0_2       # generate_branches reads the module globals; the workers get the seeds as arguments
  meta, _ = MP.generate_branches(anchors, MP.START_CKPT, args.workers, BRANCHES, limit=args.limit)
  meta['draw'] = 2; meta['hazard_seed0'] = HAZARD_SEED0_2; meta['worker_seed0'] = GEN_WORKER_SEED0_2
  summ = MP.generation_summary(BRANCHES, anchors)
  cmp = compare_tables(anchors)
  MP.write_json(OUT / 'generation_draw2.json', {'meta': meta, 'summary': summ, 'vs_sealed_table': cmp})
  print(json.dumps({'pooled': cmp['pooled_weighted'], 'rows': cmp['mean_rows'], 'identity': cmp['row_identity'], 'wall': meta['wall_seconds']}, indent=1), flush=True)


def compare_tables(anchors):
  """Draw 2 vs the sealed draw: outcome shares (anchor-weighted, by region), death-time KS, and row identity on the anchors whose
  hazards were inactive in BOTH draws (the physics and the continuation are deterministic: those paths must coincide)."""
  def load(p):
    with np.load(p, allow_pickle=False) as d:
      return {k: d[k] for k in ('obs_rows', 'offset', 'length', 'outcome', 'steps', 'u1', 'u2')}
  A = load(MP.OUT / 'branches_cf.npz'); B = load(BRANCHES)
  n = len(B['length']); oa, ob = A['outcome'].astype(str)[:n], B['outcome'].astype(str)
  W = anchors.weight[:n]; reg = region_of(anchors.state[:n, :2])
  cmp = {'pooled_weighted': {k: {'draw2': float((W * (ob == k)).sum() / W.sum()), 'sealed': float((W * (oa == k)).sum() / W.sum())} for k in ('success', 'death', 'timeout')},
         'mean_rows': {'draw2': float(B['length'].mean()), 'sealed': float(A['length'][:n].mean())},
         'by_region': {REGIONS[i]: {'n': int((reg == i).sum()), **{k: {'draw2': float((ob[reg == i] == k).mean()), 'sealed': float((oa[reg == i] == k).mean())} for k in ('success', 'death', 'timeout')},
                                    'ks_death_steps': _ks(A['steps'][:n][(reg == i) & (oa == 'death')], B['steps'][(reg == i) & (ob == 'death')])} for i in range(len(REGIONS)) if (reg == i).sum() >= 100},
         'ks_death_steps_pooled': _ks(A['steps'][:n][oa == 'death'], B['steps'][ob == 'death']), 'ks_length_pooled': _ks(A['length'][:n], B['length']),
         'outcome_agreement_per_anchor': float((oa == ob).mean()), 'hazard_share': {'draw2': [float(B['u1'].mean()), float(B['u2'].mean())], 'sealed': [float(A['u1'][:n].mean()), float(A['u2'][:n].mean())]}}
  both_inactive = np.nonzero(~A['u1'][:n] & ~A['u2'][:n] & ~B['u1'] & ~B['u2'])[0]
  ident = 0; first = []
  for k in both_inactive:
    la, lb = A['length'][k], B['length'][k]
    ra = A['obs_rows'][A['offset'][k]:A['offset'][k] + la]; rb = B['obs_rows'][B['offset'][k]:B['offset'][k] + lb]
    if la == lb and np.abs(ra - rb).max() <= 1e-4:
      ident += 1
    else:
      m = min(la, lb); d = np.abs(ra[:m] - rb[:m]).max(axis=1); bad = np.nonzero(d > 1e-4)[0]; first.append(int(bad[0]) if len(bad) else int(m))
  cmp['row_identity'] = {'anchors_inactive_in_both_draws': int(len(both_inactive)), 'identical_paths': ident, 'share': float(ident / max(1, len(both_inactive))),
                         'first_diff_row_if_any (p10/med/p90)': ([float(np.percentile(first, q)) for q in (10, 50, 90)] if first else None)}
  return cmp


def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6 mainline: a second hazard draw of the simulator branch table (arm CF2) vs the sealed draw (CF), the recorded futures (O) and the hybrid control; five paired seeds',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(),
         'status': 'oracle (simulator futures) -- engineering stage; a replication check of the oracle reference under a fresh future-table draw (the generation randomness, which the five training seeds of one table did not cover)',
         'generation': {'construction': 'exp_v6_mainline_pilot.generate_branches unchanged: the logged torque once from the restored logged state, then the start agent mode closed-loop in the simulator; fresh hazard / clock / jitter draws per anchor; ends on reach / death / horizon 800 - t; no filtering',
                        'hazard_seed0': HAZARD_SEED0_2, 'worker_seed0': GEN_WORKER_SEED0_2, 'sealed_draw_hazard_seed0': 132_000_000, 'anchors': 'all 53,747 pilot anchors (anchors.npz)', 'continuation_ckpt_sha256': MP.sha256(MP.START_CKPT)},
         'arms': {'CF2': 'critic futures = branches_cf_draw2.npz', 'CF': 'critic futures = the sealed simulator branches (branches_cf.npz)', 'O': 'critic futures = the recorded continuation', 'hybrid': 'ett_futures_hybrid (simulator motion + learned risk), for reference'},
         'recipe': {'variant': EF.BASE_VARIANT, 'overrides': OVERRIDES, 'unchanged': ['start checkpoint', 'anchors', 'actor / BC rows (logged buffer)', 'bc 0.05', f'{MP.UPDATES} updates', 'stream seeds by seed']},
         'seeds': list(SEEDS), 'evaluation': {**EVAL, 'targets': 'CF2 seeds 0-4 on the same 300 episodes as CF / O / hybrid / start'},
         'judgement': {'nature': 'a replication check of the oracle reference under a fresh future-table draw; two tables do not estimate the table-draw variance; the seed s.e. of CF - O is not a line for "the same oracle"',
                       'oracle_draw2': 'CF2 - O under the mainline rule (mean > 2 x seed s.e. and 5 / 5): does the gain survive the re-draw',
                       'draw_check': 'CF2 - CF per paired seed with its seed s.e. (reported, no threshold)',
                       'hybrid_reference': 'hybrid - CF2 and hybrid - CF reported: where the hybrid sits relative to both oracle tables',
                       'readings_kept_open': ['CF2 near 0.45 and stably above the hybrid: stronger evidence of a biased learned risk / conditional futures, not yet localised to the death cut',
                                              'CF2 down near the hybrid: table-draw sensitivity first; the 0.45 restated as the result under one particular oracle table',
                                              'in between or too uncertain: not separable'],
                       'no_selection': 'every final evaluated once; no tuning after the result'}}
  MP.write_json(p, man); print(json.dumps(man, indent=1), flush=True)


def mode_train(args):
  s = int(args.seed)
  if (run_dir2(s) / 'final.pkl').exists() and not args.force:
    print(f'{run_dir2(s)} final exists', flush=True); return
  MP.train_arm('CF', s, base=OUT, branch_path=BRANCHES, overrides=OVERRIDES, inputs=MP.OUT)


def mode_evaluate(args):
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  for s in SEEDS:
    if args.only and f'CF2/seed_{s}' not in args.only:
      continue
    ck = run_dir2(s) / 'final.pkl'
    if not ck.exists():
      print(f'-- missing CF2/seed_{s}', flush=True); continue
    print(f'== evaluate CF2/seed_{s}', flush=True)
    print(MP.D.evaluate_ckpt(ck, run_dir2(s), EVAL['policy']), flush=True)


def mode_report(args):
  man = MP.read_json(OUT / 'manifest.json')
  dirs = {'CF2': run_dir2, 'CF': lambda s: EF.run_dir('CF', s), 'O': lambda s: EF.run_dir('O', s), 'hybrid': lambda s: MP.run_dir('CF', s, HYBRID)}
  E = {arm: {s: MP._episodes(EF.eval_file(f(s))) for s in SEEDS if EF.eval_file(f(s)).exists()} for arm, f in dirs.items()}
  start = MP._episodes(EF.eval_file(MP.OUT / 'start_agent')) if EF.eval_file(MP.OUT / 'start_agent').exists() else None
  res = {'sealed_at': man['sealed_at'], 'eval': EVAL, 'headline': {arm: {s: MP._headline(e) for s, e in by.items()} for arm, by in E.items()}, 'paired': {}}
  if start is not None:
    res['headline']['start'] = MP._headline(start)
  full = [arm for arm in dirs if len(E[arm]) == len(SEEDS)]
  for label, a, b in (('CF2 - O', 'CF2', 'O'), ('CF - O', 'CF', 'O'), ('CF2 - CF', 'CF2', 'CF'), ('hybrid - CF2', 'hybrid', 'CF2'), ('hybrid - CF', 'hybrid', 'CF')):
    if a in full and b in full:
      res['paired'][label] = {key: MP.paired_block(E[a], E[b], key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  if start is not None and 'CF2' in full:
    res['paired']['CF2 - start'] = {key: MP.paired_block(E['CF2'], start, key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  g = OUT / 'generation_draw2.json'
  res['generation'] = MP.read_json(g)['vs_sealed_table'] if g.exists() else None
  def blk(label):
    return res['paired'].get(label, {}).get('success', {}) or {}
  J = {'nature': 'replication check under a fresh future-table draw (two tables; no variance estimate; no equivalence threshold)',
       'oracle_draw2_CF2_minus_O_rule_met': blk('CF2 - O').get('improvement_rule_met'), 'CF2_minus_O_mean': blk('CF2 - O').get('mean'),
       'CF2_minus_CF_mean': blk('CF2 - CF').get('mean'), 'CF2_minus_CF_seed_se': blk('CF2 - CF').get('seed_se'), 'CF2_minus_CF_same_direction': blk('CF2 - CF').get('seeds_same_direction'),
       'hybrid_minus_CF2_mean': blk('hybrid - CF2').get('mean'), 'hybrid_minus_CF2_seed_se': blk('hybrid - CF2').get('seed_se'), 'hybrid_minus_CF2_same_direction': blk('hybrid - CF2').get('seeds_same_direction'),
       'hybrid_minus_CF_mean': blk('hybrid - CF').get('mean'), 'reading': 'left to the written rule (three readings kept open); not computed here'}
  res['judgement'] = J
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res, man), encoding='utf-8')
  print(json.dumps(J, indent=1), flush=True)


def write_md(res, man):
  L = ['# A second hazard draw of the simulator branch table (CF2) vs the sealed draw (CF), the recorded futures (O) and the hybrid control: five paired seeds', '',
       f"Sealed {man['sealed_at']}.  Evaluation seed {res['eval']['seed']}, {res['eval']['n']} episodes, policy mode; the same episodes for every checkpoint.", '',
       '## Headline (success / detour / death / timeout)', '', '| arm | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean |', '|---|' + '---|' * (len(SEEDS) + 1)]
  for arm, by in res['headline'].items():
    if arm == 'start':
      h = by; L.append(f"| start | {h['success']:.3f} / {h['detour']:.2f} / {h['death']:.2f} / {h['timeout']:.2f} | " + ' | ' * (len(SEEDS) - 1) + ' |'); continue
    cells = [(f"{by[s]['success']:.3f} / {by[s]['detour']:.2f} / {by[s]['death']:.2f} / {by[s]['timeout']:.2f}" if s in by else '') for s in SEEDS]
    mean = np.mean([by[s]['success'] for s in by]) if by else float('nan')
    L.append(f'| {arm} | ' + ' | '.join(cells) + f' | {mean:.3f} |')
  L += ['', '## Paired differences (success)', '', '| comparison | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean | seed s.e. | same direction | rule |', '|---|' + '---:|' * len(SEEDS) + '---:|---:|---:|---|']
  for label, blk in res['paired'].items():
    b = blk['success']
    L.append(f'| {label} | ' + ' | '.join((f"{b['per_seed'][s]['mean']:+.3f}" if s in b['per_seed'] else f"{b['per_seed'][str(s)]['mean']:+.3f}") for s in SEEDS) + f" | {b['mean']:+.3f} | {b['seed_se']:.3f} | {b['seeds_same_direction']} | {b['improvement_rule_met']} |")
  L += ['', '## Other keys (seed mean of the paired difference)', '', '| comparison | detour | death | timeout |', '|---|---:|---:|---:|']
  for label, blk in res['paired'].items():
    L.append(f"| {label} | {blk['detour']['mean']:+.3f} | {blk['failure']['mean']:+.3f} | {blk['timeout']['mean']:+.3f} |")
  c = res.get('generation')
  if c:
    L += ['', '## Draw 2 vs the sealed draw (anchor-weighted outcome shares)', '', '| | success draw2 / sealed | death draw2 / sealed | timeout draw2 / sealed | KS death time |', '|---|---|---|---|---|',
          f"| pooled | {c['pooled_weighted']['success']['draw2']:.3f} / {c['pooled_weighted']['success']['sealed']:.3f} | {c['pooled_weighted']['death']['draw2']:.3f} / {c['pooled_weighted']['death']['sealed']:.3f} | {c['pooled_weighted']['timeout']['draw2']:.3f} / {c['pooled_weighted']['timeout']['sealed']:.3f} | {c['ks_death_steps_pooled']} |"]
    for r, v in c['by_region'].items():
      L.append(f"| {r} (n {v['n']}) | {v['success']['draw2']:.3f} / {v['success']['sealed']:.3f} | {v['death']['draw2']:.3f} / {v['death']['sealed']:.3f} | {v['timeout']['draw2']:.3f} / {v['timeout']['sealed']:.3f} | {v['ks_death_steps']} |")
    L += ['', f"Row identity on the anchors inactive in both draws: {json.dumps(c['row_identity'])}; per-anchor outcome agreement {c['outcome_agreement_per_anchor']:.3f}."]
  L += ['', '## Judgement', '', '```', json.dumps(res['judgement'], indent=1), '```', '']
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('generate', 'seal', 'train', 'evaluate', 'report'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--workers', type=int, default=18)
  ap.add_argument('--only', nargs='*', default=None)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'generate': mode_generate, 'seal': mode_seal, 'train': mode_train, 'evaluate': mode_evaluate, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
