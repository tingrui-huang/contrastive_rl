"""AntMaze V6: does pretraining the NCE critic on the full recorded data, then
continuing on the counterfactual branch replay, give a critic (and an actor)
that generalises where the random-init branch critic does not?

The user's proposal after the A/B/C, coverage and supervised diagnostics:
a network trained from scratch on ~13k exact branch keys must learn the
Ant's pose / velocity / torque relations AND the counterfactual value at
once; the recorded trajectories (268k transitions) carry the former.  So:

  stage 1  vanilla CRL critic on the full recorded dataset, the V6 recipe
           (gamma 0.999, 100k updates) -- a representation, not a value
           to be trusted (expert confounding remains);
  stage 2  the SAME NCE continued from that checkpoint on the branch replay
           (R arm: every positive future is a counterfactual continuation,
           no recorded futures mixed back, no ordering constraint), 30k
           updates, the replay's anchor weights;
  stage 3  actors from the pure-BC walker, frozen critic, the original actor
           loss with bc 0.05, 30k updates, 300-episode deployment evaluation
           (mean policy) -- for the treatment AND both controls.

Arms (three seeds each, seeds 0 / 1 / 2 shared):
  T   pretrain -> branch (stages 1 + 2)            treatment
  C1  random init -> branch (critics_armR, exist)  the existing recipe
  C2  pretrain only (stage 1)                      attributes any gain to the pretraining alone

This IS an added training arrangement (the loss is unchanged, the schedule
is not) and is reported as such; the pretraining cost is reported.

  python scripts/exp_v6_pretrain_branch.py seal
  (node scripts run the stages with run_v6_branch_replay.py)
  python scripts/exp_v6_pretrain_branch.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402

EXP = 'exp_pretrain_branch'
SEEDS = (0, 1, 2)
ARMS = {'T': {'critic_dir': 'critics_armT', 'joint_dir': 'joint_T', 'critic': 'vanilla 100k on the recorded data -> 30k NCE on the R replay (V6_CRITIC_INIT)'},
        'C1': {'critic_dir': 'critics_armR', 'joint_dir': 'joint_C1', 'critic': 'random init -> 30k NCE on the R replay (existing)'},
        'C2': {'critic_dir': 'critics_pre100k_only', 'joint_dir': 'joint_C2', 'critic': 'vanilla 100k on the recorded data only'}}


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def git_info():
  try:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    st = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(ROOT), text=True).splitlines()
    return {'head': head, 'dirty': [l for l in st if not l.startswith('??')], 'untracked_count': sum(l.startswith('??') for l in st)}
  except Exception as e:  # noqa
    return {'error': str(e)}


def seal(args):
  out = B.OUT / EXP; out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed)')
  O = B.OUT
  prov = {'dataset': str(B.DATASET), 'dataset_sha256': sha256(B.DATASET), 'r_replay': str(O / 'exp_r1_coverage' / 'replay_armR.npz'),
          'r_replay_sha256': sha256(O / 'exp_r1_coverage' / 'replay_armR.npz'),
          'actor_init': str(O / 'joint_purebc' / 'seed_0' / 'final.pkl'), 'actor_init_sha256': sha256(O / 'joint_purebc' / 'seed_0' / 'final.pkl'),
          'c1_critics': {s: sha256(O / 'critics_armR' / f'seed_{s}' / 'final.pkl') for s in SEEDS},
          'abc_outcomes_sha256': sha256(O / 'diag_r1_abc' / 'outcomes.npz'), 'abc_manifest_sha256': sha256(O / 'diag_r1_abc' / 'manifest.json'),
          'existing_c1_metrics': str(O / 'diag_r1_abc' / 'metrics_armR.json'),
          'scripts': {f: sha256(ROOT / f) for f in ('scripts/exp_v6_pretrain_branch.py', 'scripts/run_v6_branch_replay.py', 'scripts/diag_v6_r1_abc.py', 'scripts/probe_v6_coverage_paired_boot.py')}}
  man = {'experiment': 'pretrain on the recorded data -> continue NCE on the branch replay; actor from BC; controls C1 (random -> branch) and C2 (pretrain only)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git': git_info(), 'provenance': prov,
         'stages': {'1_pretrain': {'data': 'recorded p050 dataset (1000 episodes, 268,112 transitions), recorded futures', 'recipe': 'run_v6_branch_replay.train_vanilla: V6 vanilla at gamma 0.999',
                                   'updates': 100_000, 'seeds': list(SEEDS), 'tag': '_pre100k', 'note': 'a representation; its values are not trusted (expert confounding)'},
                    '2_branch': {'data': 'R replay (35,756 paths, anchor weights 1 / 0.25, every future a counterfactual continuation)', 'recipe': 'the same NCE, critic warm-started from stage 1 (V6_CRITIC_INIT, step counter reset), 30,000 updates, V6_ANCHOR_WEIGHTS=audit',
                                 'no_recorded_futures': True, 'no_ordering_constraint': True, 'seeds': list(SEEDS), 'tag': '_armT'},
                    '3_actor': {'init': 'pure-BC walker joint_purebc/seed_0 (lambda 1.0, 100k) for every arm', 'critic': 'frozen (lr 0)', 'loss': 'original actor loss + bc 0.05 (balanced BC rows from the recorded data, as in round 1)',
                                'critic_batches': 'the R replay with its anchor weights for every arm (only the critic differs)', 'updates': 30_000, 'seeds': list(SEEDS),
                                'evaluation': '300 natural draws, seed 909, deployment (mean) policy: success, failure, timeout, detour, discounted, mouth timing'}},
         'arms': ARMS,
         'critic_evaluation': 'diag_v6_r1_abc analyze on the sealed A/B/C outcomes (development sets) for T and C2; C1 = metrics_armR.json; paired episode bootstrap (probe_v6_coverage_paired_boot) T vs C1 vs C2 on B and C',
         'decision_rules': {'critic_level': 'T counts as an improvement over C1 on B (resp. C) if the seed-mean agreement difference exceeds 2 paired s.e. with all three seeds in the same direction and a positive pick gain',
                            'policy_level': 'T counts as a pipeline improvement if its mean success over 3 seeds exceeds both C1 and C2 by more than 2 x the pooled seed s.e. AND its detour rate exceeds 0.05 on at least 2 of 3 seeds; '
                                            'the reference walker (pure BC) has success 0.253, detour 0.000',
                            'attribution': 'a gain present in C2 as well is attributed to pretraining, not to the branch data; a gain in T but not C2 is attributed to the schedule pretrain -> branch',
                            'nonsignificant': 'inconclusive, not equivalence'},
         'budget': {'critic_updates': {'T': 130_000, 'C1': 30_000, 'C2': 100_000}, 'actor_updates': 30_000, 'evaluation_episodes': 300, 'no_extension': True, 'no_seed_dropping': True, 'no_sweeps': True},
         'cost_accounting': 'wall seconds per stage from the runs\' branch_manifest.json, reported in REPORT.md'}
  (out / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
  print(json.dumps({k: man[k] for k in ('stages', 'decision_rules', 'budget')}, indent=1), flush=True)


def _headline(path):
  import run_v6_branch_replay as D
  ev = json.loads(Path(path).read_text(encoding='utf-8'))
  return D._headline(ev['summary'], ev.get('episodes'))


def report(args):
  out = B.OUT / EXP; O = B.OUT
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  f_ = lambda v, fmt='.3f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
  L = ['# Pretrain (recorded data) -> branch replay: critics and actors against the random-init recipe', '',
       f'Sealed {man["sealed_at"]}.  Arms: T = vanilla 100k -> 30k NCE on the R replay; C1 = random init -> 30k NCE on the R replay (existing critics_armR); '
       'C2 = vanilla 100k only.  Actors: pure-BC init, frozen critic, bc 0.05, 30k, 300-episode mean-policy evaluation.  This is an added training '
       'schedule (the loss is unchanged).', '']
  # --- costs
  L += ['## Cost (wall seconds from the run manifests)', '', '| arm | stage | seed 0 / 1 / 2 |', '|---|---|---|']
  for arm, spec in ARMS.items():
    for stage, d in (('critic', spec['critic_dir']), ('actor', spec['joint_dir'])):
      cells = []
      for s in SEEDS:
        m = O / d / f'seed_{s}' / 'branch_manifest.json'
        cells.append(f'{json.loads(m.read_text(encoding="utf-8"))["wall_seconds"]:.0f}' if m.exists() else '-')
      L.append(f'| {arm} | {stage} ({d}) | {" / ".join(cells)} |')
  for s in SEEDS:
    m = O / 'vanilla_g0999_pre100k' / f'seed_{s}' / 'branch_manifest.json'
    if m.exists():
      L.append(f'| T, C2 | pretraining (vanilla_g0999_pre100k) seed {s} | {json.loads(m.read_text(encoding="utf-8"))["wall_seconds"]:.0f} |')
  # --- critic A/B/C
  metr = {'T': O / 'diag_r1_abc' / 'metrics_armT.json', 'C1': O / 'diag_r1_abc' / 'metrics_armR.json', 'C2': O / 'diag_r1_abc' / 'metrics_pre100k.json'}
  L += ['', '## Critics on the A / B / C diagnostic (region min readout; agreement among decided pairs (s.e.); pick gain)', '',
        '| arm | seed | A | B | C | C matched |', '|---|---|---|---|---|---|']
  for arm, p in metr.items():
    if not p.exists():
      continue
    res = json.loads(p.read_text(encoding='utf-8'))
    for sc in [s for s in res['layers']['A'] if s.endswith('region min')]:
      cells = []
      for ly in ('A', 'B', 'C', 'C_matched'):
        m = res['layers'][ly][sc]['pooled']
        cells.append(f'{f_(m["agreement_decided"], ".2f")} ({f_(m["se_decided"])}) n {m["n_decided"]}; gain {f_(m["pick_gain"], "+.3f")}')
      L.append(f'| {arm} | {sc.split(" ")[0].split("/")[-1]} | ' + ' | '.join(cells) + ' |')
  pb = O / 'diag_r1_abc' / 'paired_boot_pretrain_B.json'
  for ly in ('B', 'C'):
    p = O / 'diag_r1_abc' / f'paired_boot_pretrain_{ly}.json'
    if p.exists():
      j = json.loads(p.read_text(encoding='utf-8'))
      L += ['', f'Paired episode bootstrap on {ly} ({j["n_decided_pairs"]} decided pairs, {j["n_episodes"]} episodes): '
            + '; '.join(f'{r} {v["agreement_seed_mean"]:.3f} +- {v["se"]:.3f}' for r, v in j['runs'].items()) + '.  Differences: '
            + '; '.join(f'{k} {v["mean"]:+.3f} +- {v["se"]:.3f} (z {v["z"]:.1f}; per seed ' + ', '.join(f'{x["diff"]:+.3f}' for x in v['per_seed'].values()) + ')'
                        for k, v in j['differences'].items() if k.startswith('T'))]
  # --- actors
  L += ['', '## Deployment evaluation of the actors (300 natural draws, mean policy)', '',
        '| arm | seed | success | failure | timeout | detour | shortcut | discounted | mouth 1 / 2 median | after-burst 1 / 2 |', '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
  rows = {}
  for arm, spec in ARMS.items():
    for s in SEEDS:
      p = O / spec['joint_dir'] / f'seed_{s}' / 'eval_mean.json'
      if not p.exists():
        continue
      h = _headline(p); rows.setdefault(arm, []).append(h)
      L.append(f'| {arm} | {s} | {f_(h["success_rate"])} | {f_(h["failure_rate"])} | {f_(h["timeout_rate"])} | {f_(h["detour_rate"])} | {f_(h["shortcut_rate"])} | '
               f'{f_(h.get("discounted"))} | {f_(h.get("mouth1_median"), ".0f")} / {f_(h.get("mouth2_median"), ".0f")} | {f_(h.get("mouth1_after_burst"), ".2f")} / {f_(h.get("mouth2_after_burst"), ".2f")} |')
  L += ['', 'Reference: pure-BC walker (the actor init) success 0.253, detour 0.000; round-1 vanilla actors success 0.26, detour 0.003-0.007.', '']
  if rows:
    L += ['| arm | mean success (seed s.e.) | mean detour (seed s.e.) | mean discounted |', '|---|---|---|---|']
    stats = {}
    for arm, hs in rows.items():
      su = np.array([h['success_rate'] for h in hs]); de = np.array([h['detour_rate'] for h in hs]); di = np.array([h.get('discounted') or 0 for h in hs])
      stats[arm] = (su, de)
      L.append(f'| {arm} | {su.mean():.3f} ({su.std(ddof=1) / np.sqrt(len(su)) if len(su) > 1 else 0:.3f}) | {de.mean():.3f} ({de.std(ddof=1) / np.sqrt(len(de)) if len(de) > 1 else 0:.3f}) | {di.mean():.3f} |')
    if all(a in stats for a in ('T', 'C1', 'C2')):
      su_T, de_T = stats['T']
      verdict = []
      for c in ('C1', 'C2'):
        su_c = stats[c][0]
        pooled_se = np.sqrt((su_T.var(ddof=1) + su_c.var(ddof=1)) / len(su_T)) if len(su_T) > 1 else float('inf')
        verdict.append(f'T - {c} success {su_T.mean() - su_c.mean():+.3f} (pooled seed s.e. {pooled_se:.3f}; {"> 2 s.e." if su_T.mean() - su_c.mean() > 2 * pooled_se else "not > 2 s.e."})')
      verdict.append(f'T detour > 0.05 on {int((de_T > 0.05).sum())} of {len(de_T)} seeds')
      L += ['', '## Pre-registered policy-level decision', ''] + [f'- {v}' for v in verdict]
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'report'))
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': seal, 'report': report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
