#!/usr/bin/env python
"""AntMaze V6 -- the equal-weight MULTI-FUTURE control (user's plan after oracle_draw2, go given 2026-09-20 night).

oracle_draw2 showed the oracle gain to be table-draw dependent: the same recipe on two hazard draws of the simulator table gives 0.449
and 0.351 (per-seed differences up to 0.24), each anchor carrying ONE random future.  This control gives every anchor its TWO complete
futures (the sealed draw and draw 2): the anchor weight is unchanged, one of the two tables is chosen uniformly first, then the goal row
follows the geometric law within that path (exp_v6_mainline_pilot.MultiFutures).  Every outcome kept, no splicing, the NCE and actor losses,
the recipe (critic clip 0.1, BC 0.05, 30k updates, the start agent) and the streams' RNG consumption unchanged.  Five seeds, draw 8909.
Arm label MF; runs under multi_futures/CF/seed_s.  Compared with both single-draw arms (CF, CF2), O, the hybrid and the start agent.
Readings written before the run (none forced): (a) MF at or above the better draw's mean with a narrower seed spread -> two futures per
anchor stabilise the supervision; (b) MF between the two draws -> averaging without stabilisation; (c) MF at or below the lower draw ->
more futures per anchor do not help under this recipe.  MF - O under the mainline rule (mean > 2 x seed s.e. and 5 / 5) reported.
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
import exp_v6_oracle_draw2 as D2  # noqa: E402

OUT = MP.OUT / 'multi_futures'
TABLES = (MP.OUT / 'branches_cf.npz', D2.BRANCHES)
SEEDS, EVAL, OVERRIDES = EF.SEEDS, EF.EVAL, EF.OVERRIDES


def run_dir_mf(s):
  return MP.run_dir('CF', s, OUT)


def mode_check(args):
  """The sampler on the real tables: table share, the geometric law within each table, goals equal to the chosen table's rows."""
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  bases = [MP.BranchFutures(anchors, p) for p in TABLES]
  mf = MP.MultiFutures(bases, MP.GAMMA)
  st = MP.CriticStream(anchors, mf, 1024, MP.GAMMA, 12345)
  J = np.zeros(2, np.int64); M = [[], []]; ok = 0; n = 0
  for _ in range(50):
    tr, (k, m_unused) = st.sample(); j, m = mf.last_jm
    for jj in (0, 1):
      J[jj] += int((j == jj).sum()); M[jj].extend(m[j == jj].tolist())
    g = tr.observation[:, MP.STATE_DIM:]
    ref = np.stack([bases[int(jj)].goal_at(int(kk), int(mm)) for jj, kk, mm in zip(j, k, m)])
    ok += int(np.array_equal(g, ref.astype(np.float32))); n += 1
    # the single-table stream draws the same anchors with the same seed
  st1 = MP.CriticStream(anchors, bases[0], 1024, MP.GAMMA, 12345); st2 = MP.CriticStream(anchors, mf, 1024, MP.GAMMA, 12345)
  same_anchors = all(np.array_equal(st1.sample()[1][0], st2.sample()[1][0]) for _ in range(5))
  res = {'table_share': (J / J.sum()).tolist(), 'm_median_by_table': [float(np.median(M[0])), float(np.median(M[1]))], 'm_mean_by_table': [float(np.mean(M[0])), float(np.mean(M[1]))],
         'goals_equal_chosen_table_rows': ok == n, 'anchor_sequence_identical_to_single_table_stream': bool(same_anchors), 'batches': n}
  OUT.mkdir(parents=True, exist_ok=True); MP.write_json(OUT / 'sampler_check.json', res)
  print(json.dumps(res, indent=1), flush=True)


def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6 mainline: equal-weight multi-future supervision (arm MF: the sealed simulator draw + draw 2, one chosen uniformly per positive) vs the single-draw arms CF / CF2, O, the hybrid and the start agent; five paired seeds',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'oracle (simulator futures) -- engineering stage',
         'futures': {'tables': [str(p) for p in TABLES], 'tables_sha256': [MP.sha256(p) for p in TABLES], 'law': 'anchor by weight (unchanged); table j uniform over the two; m ~ gamma^m over 1..L_j - 1 within table j; every outcome kept; no splicing',
                     'rng': 'the one future uniform is split (j = floor(2u), u\' = 2u - j): anchor sequence identical to the single-table arms', 'sampler_check': (MP.read_json(OUT / 'sampler_check.json') if (OUT / 'sampler_check.json').exists() else 'run `check` first')},
         'arms': {'MF': 'critic futures = both tables, equal weight', 'CF': 'the sealed draw', 'CF2': 'draw 2', 'O': 'the recorded continuation', 'hybrid': 'ett_futures_hybrid (reference)'},
         'recipe': {'variant': EF.BASE_VARIANT, 'overrides': OVERRIDES, 'unchanged': ['start checkpoint', 'anchors', 'actor / BC rows (logged buffer)', 'bc 0.05', f'{MP.UPDATES} updates', 'stream seeds by seed', 'NCE and actor losses']},
         'seeds': list(SEEDS), 'evaluation': {**EVAL, 'targets': 'MF seeds 0-4 on the same 300 episodes as CF / CF2 / O / hybrid / start'},
         'judgement': {'mainline_rule': 'MF - O: mean > 2 x seed s.e. and 5 / 5 (reported)',
                       'readings_kept_open': ['(a) MF at or above the better draw\'s mean (0.449) with a narrower seed spread than either draw: two futures per anchor stabilise the supervision',
                                              '(b) MF between the two draws\' means (0.351 .. 0.449): averaging without stabilisation',
                                              '(c) MF at or below the lower draw (0.351): more futures per anchor do not help under this recipe'],
                       'no_selection': 'every final evaluated once; no tuning after the result'}}
  MP.write_json(p, man); print(json.dumps(man, indent=1), flush=True)


def mode_train(args):
  s = int(args.seed)
  if (run_dir_mf(s) / 'final.pkl').exists() and not args.force:
    print(f'{run_dir_mf(s)} final exists', flush=True); return
  MP.train_arm('CF', s, base=OUT, branch_path=list(TABLES), overrides=OVERRIDES, inputs=MP.OUT)


def mode_evaluate(args):
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  for s in SEEDS:
    if args.only and f'MF/seed_{s}' not in args.only:
      continue
    ck = run_dir_mf(s) / 'final.pkl'
    if not ck.exists():
      print(f'-- missing MF/seed_{s}', flush=True); continue
    print(f'== evaluate MF/seed_{s}', flush=True)
    print(MP.D.evaluate_ckpt(ck, run_dir_mf(s), EVAL['policy']), flush=True)


def mode_report(args):
  man = MP.read_json(OUT / 'manifest.json')
  dirs = {'MF': run_dir_mf, 'CF': lambda s: EF.run_dir('CF', s), 'CF2': D2.run_dir2, 'O': lambda s: EF.run_dir('O', s), 'hybrid': lambda s: MP.run_dir('CF', s, D2.HYBRID)}
  E = {arm: {s: MP._episodes(EF.eval_file(f(s))) for s in SEEDS if EF.eval_file(f(s)).exists()} for arm, f in dirs.items()}
  start = MP._episodes(EF.eval_file(MP.OUT / 'start_agent')) if EF.eval_file(MP.OUT / 'start_agent').exists() else None
  res = {'sealed_at': man['sealed_at'], 'eval': EVAL, 'headline': {arm: {s: MP._headline(e) for s, e in by.items()} for arm, by in E.items()}, 'paired': {}}
  if start is not None:
    res['headline']['start'] = MP._headline(start)
  full = [arm for arm in dirs if len(E[arm]) == len(SEEDS)]
  for label, a, b in (('MF - O', 'MF', 'O'), ('MF - CF', 'MF', 'CF'), ('MF - CF2', 'MF', 'CF2'), ('MF - hybrid', 'MF', 'hybrid'), ('CF - O', 'CF', 'O'), ('CF2 - O', 'CF2', 'O'), ('CF2 - CF', 'CF2', 'CF')):
    if a in full and b in full:
      res['paired'][label] = {key: MP.paired_block(E[a], E[b], key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  if start is not None and 'MF' in full:
    res['paired']['MF - start'] = {key: MP.paired_block(E['MF'], start, key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  spread = {arm: {'min': float(min(h['success'] for h in by.values())), 'max': float(max(h['success'] for h in by.values())), 'mean': float(np.mean([h['success'] for h in by.values()]))} for arm, by in res['headline'].items() if arm != 'start' and by}
  res['success_spread'] = spread
  def blk(label):
    return res['paired'].get(label, {}).get('success', {}) or {}
  J = {'MF_minus_O_rule_met': blk('MF - O').get('improvement_rule_met'), 'MF_minus_O_mean': blk('MF - O').get('mean'), 'MF_minus_O_seed_se': blk('MF - O').get('seed_se'),
       'MF_minus_CF_mean': blk('MF - CF').get('mean'), 'MF_minus_CF_same_direction': blk('MF - CF').get('seeds_same_direction'),
       'MF_minus_CF2_mean': blk('MF - CF2').get('mean'), 'MF_minus_CF2_same_direction': blk('MF - CF2').get('seeds_same_direction'),
       'success_spread': spread, 'reading': 'left to the written readings (a) / (b) / (c); not computed here'}
  res['judgement'] = J
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res, man), encoding='utf-8')
  print(json.dumps(J, indent=1), flush=True)


def write_md(res, man):
  L = ['# Equal-weight multi-future supervision (MF: sealed draw + draw 2) vs the single-draw arms, O and the hybrid: five paired seeds', '',
       f"Sealed {man['sealed_at']}.  Evaluation seed {res['eval']['seed']}, {res['eval']['n']} episodes, policy mode; the same episodes for every checkpoint.", '',
       '## Headline (success / detour / death / timeout)', '', '| arm | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean | min .. max |', '|---|' + '---|' * (len(SEEDS) + 2)]
  for arm, by in res['headline'].items():
    if arm == 'start':
      h = by; L.append(f"| start | {h['success']:.3f} / {h['detour']:.2f} / {h['death']:.2f} / {h['timeout']:.2f} | " + ' | ' * (len(SEEDS)) + ' |'); continue
    cells = [(f"{by[s]['success']:.3f} / {by[s]['detour']:.2f} / {by[s]['death']:.2f} / {by[s]['timeout']:.2f}" if s in by else '') for s in SEEDS]
    sp = res['success_spread'][arm]
    L.append(f'| {arm} | ' + ' | '.join(cells) + f" | {sp['mean']:.3f} | {sp['min']:.3f} .. {sp['max']:.3f} |")
  L += ['', '## Paired differences (success)', '', '| comparison | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean | seed s.e. | same direction | rule |', '|---|' + '---:|' * len(SEEDS) + '---:|---:|---:|---|']
  for label, blk in res['paired'].items():
    b = blk['success']
    L.append(f'| {label} | ' + ' | '.join((f"{b['per_seed'][s]['mean']:+.3f}" if s in b['per_seed'] else f"{b['per_seed'][str(s)]['mean']:+.3f}") for s in SEEDS) + f" | {b['mean']:+.3f} | {b['seed_se']:.3f} | {b['seeds_same_direction']} | {b['improvement_rule_met']} |")
  L += ['', '## Other keys (seed mean of the paired difference)', '', '| comparison | detour | death | timeout |', '|---|---:|---:|---:|']
  for label, blk in res['paired'].items():
    L.append(f"| {label} | {blk['detour']['mean']:+.3f} | {blk['failure']['mean']:+.3f} | {blk['timeout']['mean']:+.3f} |")
  L += ['', '## Judgement', '', '```', json.dumps(res['judgement'], indent=1), '```', '']
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('check', 'seal', 'train', 'evaluate', 'report'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--only', nargs='*', default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'check': mode_check, 'seal': mode_seal, 'train': mode_train, 'evaluate': mode_evaluate, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
