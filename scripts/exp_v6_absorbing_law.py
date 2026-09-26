#!/usr/bin/env python
"""AntMaze V6 mainline -- O / CF under the ABSORBING future law (the user's sparse-supervision hypothesis, 2026-09-20).

Registered as a NEW hypothesis, not a re-reading of target_support (whose
pre-registered rule tested the ORDERING of the target law and found it
correct): the current law -- a future ends on its reach / death frame, each
path normalised over its own rows -- gives a reset row one task-goal
positive per ~250 critic batches (0.1 % of its draws; ~120 in a 30k run),
so the far-route advantage, although correctly ORDERED in the target law, is
rarely practised directly.  The absorbing law (target_support: reset-row
task-goal mass 0.001 -> 0.2, ordering kept) raises that count ~180x.  Does
the same learner, on the same budget, then turn the futures into a larger
CF - O policy gain and a better critic ranking of unseen start candidates?

What changes (disclosed): the critic's positive law only.
  truncated  P(m) ~ gamma^m, m = 1..L-1                          (the sealed recipe)
  absorbing  P(m) ~ gamma^m, m = 1..H, H = HORIZON - t; rows m >= L-1 of a path that
             ended before the horizon (reach or death) are its ACTUAL terminal row; a
             path that ran to the horizon is unchanged.  No relabelling.
Applied identically to both arms (exp_v6_mainline_pilot.AbsorbingFutures); the
NCE's in-batch negatives follow from the changed positives.  Everything else
is the current mainline (variant critic_clip0.1): the same sealed start
checkpoint (start agent actor, fresh paired critic), anchors, branch file,
actor / BC rows from the logged buffer, bc 0.05, critic clip 0.1, learning
rates, 30,000 updates, three paired seeds, the stream seeds (so the anchor
sequence is identical to the clip arms; only the goals differ).  This is a
changed learning target, not a more accurate computation of the old one.

Evaluation: a fresh paired draw (seed 7909, 300 episodes, policy mode) for the
six new finals, the six current clip finals and the start agent.  Critic
read-out: the within-anchor ranking of the stage-1 query candidates
(q0..q5, fixed (state, action) -> outcome labels) at the reset rows.
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
import exp_v6_query_coverage as QC  # noqa: E402
import exp_v6_query_train as QT  # noqa: E402

OUT = MP.OUT / 'absorbing_law'
BASE_VARIANT = 'critic_clip0.1'
OVERRIDES = {'critic_clip': 0.1, 'future_law': 'absorbing'}
EVAL = {'n': 300, 'seed': 7909, 'policy': 'mean'}    # fresh draw (909 / 2909 / 3909 / 4909 / 6909 used; 5909 pre-registered elsewhere; 616_000_005 / 616_500_000 reserved)
ARMS = MP.ARMS


def run_dir(arm, s):
  return MP.run_dir(arm, s, OUT)


def clip_dir(arm, s):
  return MP.run_dir(arm, s, MP.variant_base(BASE_VARIANT))


def eval_file(d):
  return d / f'eval_mean_s{EVAL["seed"]}.json'


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    Lb, tb = d['length'].astype(np.int64), d['t'].astype(np.int64)
  tail_cf = np.maximum(MP.HORIZON - tb, Lb - 1) - (Lb - 1)
  Lr = (anchors.ep_length - anchors.t).astype(np.int64)
  tail_o = np.maximum(MP.HORIZON - anchors.t, Lr - 1) - (Lr - 1)
  w = anchors.weight / anchors.weight.sum()
  man = {'experiment': 'AntMaze V6 mainline: O vs CF under the absorbing future law (sparse-supervision hypothesis); the current mainline (critic_clip0.1) otherwise unchanged',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'oracle (simulator) futures; engineering stage; a changed learning target, disclosed',
         'hypothesis': ('the target law already orders the good actions correctly (target_support), but under the truncated law a reset row yields a task-goal positive in 0.1 % of its draws; '
                        'with the absorbing law (~20 %) the same learner on the same budget turns the futures into a larger CF - O gain and a better critic ranking of unseen start candidates'),
         'not_a_rereading': 'target_support tested the ORDERING of the target law under its own pre-registered rule and found it correct; that result stands.  This is a new, separately registered hypothesis about supervision sparsity',
         'change': {'overrides': OVERRIDES, 'law': 'P(m) ~ gamma^m over m = 1..H, H = HORIZON - t; rows m >= L-1 of a path that ended before the horizon return its actual terminal row (reach position or death position); a path that ran to the horizon is unchanged; no relabelling',
                    'applied_to': 'both arms identically (AbsorbingFutures wraps RecordedFutures / BranchFutures); NCE in-batch negatives follow from the changed positives',
                    'tail_rows_anchor_weighted_mean': {'O': float((w * tail_o).sum()), 'CF': float((w * tail_cf).sum())},
                    'share_of_positive_draws_on_the_terminal_row_anchor_weighted': {'O': float((w * ((MP.GAMMA ** (Lr - 1) - MP.GAMMA ** (np.maximum(MP.HORIZON - anchors.t, Lr - 1) + 1)) / (MP.GAMMA * (1 - MP.GAMMA ** np.maximum(MP.HORIZON - anchors.t, Lr - 1))))).sum()),
                                                                                    'CF': float((w * ((MP.GAMMA ** (Lb - 1) - MP.GAMMA ** (np.maximum(MP.HORIZON - tb, Lb - 1) + 1)) / (MP.GAMMA * (1 - MP.GAMMA ** np.maximum(MP.HORIZON - tb, Lb - 1))))).sum())}},
         'unchanged': ['start checkpoint (start agent actor; fresh paired critic by seed)', 'anchors.npz and branches_cf.npz (no new queries, no regeneration)', 'actor / BC rows from the logged buffer (ActorStream)',
                       'bc 0.05', 'critic grad clip 0.1', 'NCE and actor losses', 'learning rates, Adam', f'{MP.UPDATES} updates, batch {MP.BATCH}, G {MP.G}', 'seeds 0 / 1 / 2 and the stream seeds (same anchor sequence as the clip arms)',
                       'no continued training: every arm starts from the sealed start checkpoint'],
         'base_reference': {'variant': BASE_VARIANT, 'finals': {f'{arm}/seed_{s}': (MP.sha256(clip_dir(arm, s) / 'final.pkl') if (clip_dir(arm, s) / 'final.pkl').exists() else 'missing on this machine') for arm in ARMS for s in MP.SEEDS}},
         'start_ckpt_sha256': MP.sha256(MP.START_CKPT) if MP.START_CKPT.exists() else 'missing on this machine', 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'), 'branches_sha256': MP.sha256(MP.OUT / 'branches_cf.npz'),
         'evaluation': {**EVAL, 'targets': 'the six new finals, the six clip finals, the start agent; the same 300 episodes'},
         'critic_readout': 'within-anchor AUROC (success label) of the critic score over the stage-1 query candidates q0..q5 at the reset rows, per lineage, new CF critic vs the clip CF critic (the candidates and labels are fixed (state, action) -> outcome pairs from the stage-1 generation)',
         'judgement': {'primary': 'CF_abs - O_abs success on the common episodes, per paired seed: mean > 2 x seed s.e. and 3 / 3 (the mainline rule)',
                       'vs_current': 'CF_abs - CF_clip success: mean > 2 x seed s.e. and 3 / 3 (the new law must beat the current arms, not only its own O)',
                       'critic': 'reset-row within-anchor AUROC of CF_abs critics above the CF_clip critics in 3 / 3 lineages (reported; supports the mechanism)',
                       'reading': {'primary and vs_current met': 'the sparse-supervision hypothesis is supported: a mainline candidate with a disclosed change of the future law (confirmation on a further draw would follow)',
                                   'otherwise': 'not supported: stop tuning around the sampling rule (user, 2026-09-20)'},
                       'no_selection': 'six finals evaluated once; no tuning after the result'}}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# ------------------------------------------------------------------- train
def mode_train(args):
  if not (OUT / 'manifest.json').exists():
    raise SystemExit('seal first')
  for s in MP.SEEDS:
    if args.seed is not None and int(args.seed) != s:
      continue
    MP.train_arm(args.arm, s, base=OUT, overrides=OVERRIDES, inputs=MP.OUT)
    m = MP.read_json(run_dir(args.arm, s) / 'train_manifest.json')
    assert m['future_law'] == 'absorbing' and m['critic_clip'] == 0.1, m['future_law']
    # the anchor sequence must equal the clip arm's (same stream seeds): compare the first-batch anchor hash when the clip manifest is here
    cm = clip_dir(args.arm, s) / 'train_manifest.json'
    if cm.exists():
      c = MP.read_json(cm)
      same = c['first_batches']['critic_anchor_ids'] == m['first_batches']['critic_anchor_ids']
      print(f'first-batch critic anchors equal to {BASE_VARIANT}: {same}', flush=True)
      assert same, 'the anchor sequence differs from the clip arm'


# ---------------------------------------------------------------- evaluate
def mode_evaluate(args):
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = EVAL['seed'], EVAL['n']
  todo = []
  for arm in ARMS:
    for s in MP.SEEDS:
      todo.append((f'abs {arm}/seed_{s}', run_dir(arm, s) / 'final.pkl', run_dir(arm, s)))
      todo.append((f'clip {arm}/seed_{s}', clip_dir(arm, s) / 'final.pkl', clip_dir(arm, s)))
  todo.append(('start', MP.START_CKPT, MP.OUT / 'start_agent'))
  for name, ck, od in todo:
    if args.only and name not in args.only:
      continue
    if not ck.exists():
      print(f'-- missing {name}: {ck}', flush=True); continue
    print(f'== evaluate {name}', flush=True)
    print(MP.D.evaluate_ckpt(ck, od, EVAL['policy']), flush=True)


# ----------------------------------------------------------------- readout
def mode_readout(args):
  """The critic ranking of the stage-1 query candidates at the reset rows: new CF critic vs the clip CF critic, per lineage."""
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  res = {}
  for s in MP.SEEDS:
    with np.load(QC.branch_path(s), allow_pickle=False) as d:
      q_anchor, q_query, q_out = d['anchor_id'].astype(np.int64), d['query_id'].astype(np.int64), d['outcome'].astype(str)
      q_action = d['query_action'].astype(np.float32)
    key = q_anchor * 100 + q_query
    uk, inv = np.unique(key, return_inverse=True)
    lab = np.bincount(inv, weights=(q_out == 'success'), minlength=len(uk)) / np.bincount(inv, minlength=len(uk))
    a_of = np.zeros((len(uk), MP.ACTION_DIM), np.float32); a_of[inv] = q_action
    k_of, q_of = uk // 100, uk % 100
    obs31 = anchors.obs31[k_of]
    blk = {}
    for name, ck in (('CF_abs', run_dir('CF', s) / 'final.pkl'), ('CF_clip', clip_dir('CF', s) / 'final.pkl'), ('O_abs', run_dir('O', s) / 'final.pkl'), ('O_clip', clip_dir('O', s) / 'final.pkl')):
      if not ck.exists():
        continue
      f = QT.critic_scores(ck, obs31, a_of)
      out = {}
      for sname, msk in (('reset rows', anchors.t[k_of] == 0), ('all start anchors', np.ones(len(uk), bool))):
        aucs, argmax_ok = [], []
        for k in np.unique(k_of[msk]):
          i = np.flatnonzero((k_of == k) & msk); l = lab[i] > 0
          if l.any() and (~l).any():
            aucs.append(QT.auroc(f[i], l))
          argmax_ok.append(bool(lab[i[np.argmax(f[i])]] > 0))
        out[sname] = {'anchors': int(len(np.unique(k_of[msk]))), 'anchors_with_both_outcomes': len(aucs), 'within_anchor_auroc_success': float(np.mean(aucs)) if aucs else None,
                      'share_argmax_query_succeeded': float(np.mean(argmax_ok)), 'f_mean_by_query': {QC.QUERY_NAMES[q]: float(np.mean(f[msk & (q_of == q)])) for q in range(QC.N_QUERIES)}}
      blk[name] = out
    res[f'seed_{s}'] = blk
    print(f'lineage {s}: ' + ', '.join(f"{n} reset AUROC {v['reset rows']['within_anchor_auroc_success']:.3f}" for n, v in blk.items()), flush=True)
  MP.write_json(OUT / 'readout.json', res)


# ------------------------------------------------------------------ report
def mode_report(args):
  man = MP.read_json(OUT / 'manifest.json')
  E = {}
  for arm in ARMS:
    for s in MP.SEEDS:
      for tag, d in (('abs', run_dir(arm, s)), ('clip', clip_dir(arm, s))):
        p = eval_file(d)
        if p.exists():
          E[f'{tag}_{arm}'] = E.get(f'{tag}_{arm}', {}); E[f'{tag}_{arm}'][s] = MP._episodes(p)
  start = MP._episodes(eval_file(MP.OUT / 'start_agent')) if eval_file(MP.OUT / 'start_agent').exists() else None
  res = {'sealed_at': man['sealed_at'], 'eval': EVAL, 'headline': {}, 'paired': {}}
  for name, by in E.items():
    res['headline'][name] = {s: MP._headline(e) for s, e in by.items()}
  if start is not None:
    res['headline']['start'] = MP._headline(start)
  pairs = [('CF_abs - O_abs', 'abs_CF', 'abs_O'), ('CF_clip - O_clip', 'clip_CF', 'clip_O'), ('CF_abs - CF_clip', 'abs_CF', 'clip_CF'), ('O_abs - O_clip', 'abs_O', 'clip_O')]
  for label, a, b in pairs:
    if a in E and b in E and len(E[a]) == len(MP.SEEDS) and len(E[b]) == len(MP.SEEDS):
      res['paired'][label] = {key: MP.paired_block(E[a], E[b], key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  if start is not None:
    for a in ('abs_CF', 'abs_O'):
      if a in E and len(E[a]) == len(MP.SEEDS):
        res['paired'][f'{a} - start'] = {key: MP.paired_block(E[a], start, key=key) for key in ('success', 'detour', 'failure', 'timeout')}
  # the difference of gains: (CF_abs - O_abs) - (CF_clip - O_clip), episode-paired per seed
  if all(k in E and len(E[k]) == len(MP.SEEDS) for k in ('abs_CF', 'abs_O', 'clip_CF', 'clip_O')):
    gain_abs = {s: {'success': E['abs_CF'][s]['success'].astype(float) - E['abs_O'][s]['success'].astype(float)} for s in MP.SEEDS}
    gain_clip = {s: {'success': E['clip_CF'][s]['success'].astype(float) - E['clip_O'][s]['success'].astype(float)} for s in MP.SEEDS}
    res['paired']['gain_abs - gain_clip'] = {'success': MP.paired_block(gain_abs, gain_clip, key='success')}
  rp = OUT / 'readout.json'
  res['critic_readout'] = MP.read_json(rp) if rp.exists() else None
  res['judgement'] = judge(res)
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res, man), encoding='utf-8')
  print(json.dumps(res['judgement'], indent=1), flush=True)


def _rule(blk):
  return bool(blk['improvement_rule_met']) if blk else None


def judge(res):
  P = res['paired']
  out = {'primary_CF_abs_minus_O_abs': _rule(P.get('CF_abs - O_abs', {}).get('success')), 'vs_current_CF_abs_minus_CF_clip': _rule(P.get('CF_abs - CF_clip', {}).get('success'))}
  cr = res.get('critic_readout')
  if cr:
    ups = [cr[k]['CF_abs']['reset rows']['within_anchor_auroc_success'] > cr[k]['CF_clip']['reset rows']['within_anchor_auroc_success'] for k in cr if 'CF_abs' in cr[k] and 'CF_clip' in cr[k]]
    out['critic_reset_auroc_up'] = f'{sum(ups)} / {len(ups)}'
  out['supported'] = bool(out['primary_CF_abs_minus_O_abs'] and out['vs_current_CF_abs_minus_CF_clip'])
  return out


def f3(x):
  return 'nan' if x is None else f'{x:.3f}'


def write_md(res, man):
  L = ['# O vs CF under the absorbing future law (sparse-supervision hypothesis)', '',
       f"Sealed {man['sealed_at']}.  Evaluation seed {EVAL['seed']}, {EVAL['n']} episodes, policy mode; the same episodes for every checkpoint.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3 / 3.", '',
       '## Headline (success / detour / death / timeout)', '', '| run | seed 0 | seed 1 | seed 2 |', '|---|---|---|---|']
  for name, by in res['headline'].items():
    if name == 'start':
      h = by; L.append(f"| start | {h['success']:.3f} / {h['detour']:.2f} / {h['death']:.2f} / {h['timeout']:.2f} | | |"); continue
    L.append(f'| {name} | ' + ' | '.join((f"{by[s]['success']:.3f} / {by[s]['detour']:.2f} / {by[s]['death']:.2f} / {by[s]['timeout']:.2f}" if s in by else '') for s in MP.SEEDS) + ' |')
  L += ['', '## Paired differences (success; per seed, seed mean, seed s.e., seeds in direction)', '', '| comparison | seed 0 | seed 1 | seed 2 | mean | seed s.e. | in direction | rule |', '|---|---:|---:|---:|---:|---:|---:|---|']
  for label, blk in res['paired'].items():
    b = blk['success']
    L.append(f'| {label} | ' + ' | '.join(f"{b['per_seed'][s]['mean']:+.3f}" if s in b['per_seed'] else f"{b['per_seed'][str(s)]['mean']:+.3f}" for s in MP.SEEDS) + f" | {b['mean']:+.3f} | {b['seed_se']:.3f} | {b['seeds_same_direction']} | {_rule(b)} |")
  L += ['', '## Other keys (seed mean of the paired difference)', '', '| comparison | detour | death | timeout |', '|---|---:|---:|---:|']
  for label, blk in res['paired'].items():
    if 'detour' in blk:
      L.append(f"| {label} | {blk['detour']['mean']:+.3f} | {blk['failure']['mean']:+.3f} | {blk['timeout']['mean']:+.3f} |")
  cr = res.get('critic_readout')
  if cr:
    L += ['', '## Critic read-out: within-anchor AUROC (success) over the stage-1 query candidates', '', '| lineage | CF_abs reset / all | CF_clip reset / all | O_abs reset / all | O_clip reset / all |', '|---|---|---|---|---|']
    for k, v in cr.items():
      cell = lambda n: (f"{v[n]['reset rows']['within_anchor_auroc_success']:.3f} / {v[n]['all start anchors']['within_anchor_auroc_success']:.3f}" if n in v else '')  # noqa: E731
      L.append(f"| {k} | {cell('CF_abs')} | {cell('CF_clip')} | {cell('O_abs')} | {cell('O_clip')} |")
  L += ['', '## Judgement', '', '```', json.dumps(res['judgement'], indent=1), '```', '']
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('seal', 'train', 'evaluate', 'readout', 'report'))
  ap.add_argument('--arm', choices=ARMS, default='CF')
  ap.add_argument('--seed', type=int, default=None)
  ap.add_argument('--only', nargs='*', default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': mode_seal, 'train': mode_train, 'evaluate': mode_evaluate, 'readout': mode_readout, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
