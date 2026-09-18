"""AntMaze V6: does a larger share of detour demonstrations in the offline
data let the whole pipeline (BC walker -> critic -> actor) work?  A data-
composition experiment, not a method change.

Two 1,000-episode datasets are drawn from the p050 pool (the original
dataset + the five Plan-A shards, minus the old held-out 100 episodes and
the 30 Cnew reserved episodes), by route label only, whole episodes,
successes and failures kept, NESTED so that composition is the only
difference:

  d05   50 detour + 950 shortcut episodes   (the original 5.4%)
  d20  200 detour + 800 shortcut episodes   (the 50 and the 800 are subsets)

Per dataset (same recipe everywhere; seeds 0 / 1 / 2 where seeded):
  1. pure-BC walker (lambda 1.0, balanced BC rows, 100k) -> 300-episode
     deployment evaluation (route choice from the start) and the placed-on-
     detour completion test on rows of the Cnew reserved episodes;
  2. vanilla CRL critic on the recorded data (30k) and the branch critic on
     an r1-protocol replay whose continuation is THIS dataset's BC walker
     (30k); actors from this dataset's BC, frozen critic, bc 0.05, 30k, BC
     rows from this dataset; 300-episode deployment evaluation;
  3. the effective training shares of detour-episode rows at every level
     (dataset episodes / rows, the vanilla anchor law, the replay's anchor
     mass by stratum and general share, the balanced BC sampler's draws).

Pre-registered reading: an effect requires a mean difference over three
seeds larger than twice the pooled seed s.e. with 3 / 3 seeds in the same
direction; within a dataset "branch - vanilla" attributes a gain to the
counterfactual futures; within a method "d20 - d05" attributes it to the
demonstrations.  Oracle-stage generation; the BC controls are single-seed.

  python scripts/exp_v6_detour_ratio.py datasets   (V6_DATASET_STEM = the merged pool)
  python scripts/exp_v6_detour_ratio.py seal
  python scripts/exp_v6_detour_ratio.py audit --ratio d05|d20
  python scripts/exp_v6_detour_ratio.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402

EXP = 'exp_detour_ratio'
RATIOS = {'d05': (50, 950), 'd20': (200, 800)}
DDIR = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset'
STEM = 'antmaze_rockfall_clock_v6_p050_{ratio}'


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


def exp_dir():
  return B.OUT / EXP


# ---------------------------------------------------------------- datasets
def datasets(args):
  """From the merged pool (V6_DATASET_STEM = ..._plus5k): nested route-labelled draws."""
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, act, lengths, goals, meta = d['obs'], d['act'], d['lengths'], d['eval_goals'], json.loads(str(d['meta']))
  with np.load(B.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  old_held = set(json.loads(str(np.load(B.OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  ec = json.loads((B.OUT / 'exp_episode_coverage' / 'manifest.json').read_text(encoding='utf-8'))
  cnew = set(ec['episodes']['new_held_out_detour (reserved, Cnew)'])
  excluded = old_held | cnew
  det = [e for e in range(len(route)) if route[e] == 'detour' and e not in excluded]
  sc_ = [e for e in range(len(route)) if route[e] == 'shortcut' and e not in excluded]
  rng = np.random.default_rng(args.seed)
  det_order = rng.permutation(det).tolist(); sc_order = rng.permutation(sc_).tolist()
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  info = {'pool': {'detour': len(det), 'shortcut': len(sc_), 'excluded_old_held_out': len(old_held), 'excluded_cnew': len(cnew)}, 'selection_seed': args.seed, 'nested': True, 'ratios': {}}
  for ratio, (n_d, n_s) in RATIOS.items():
    eps = sorted(det_order[:n_d] + sc_order[:n_s])
    stem = STEM.format(ratio=ratio)
    m = dict(meta); m.update({'n_episodes': len(eps), 'n_transitions': int((lengths[eps] - 1).sum()), 'composition': f'{n_d} detour + {n_s} shortcut episodes drawn by route label from the p050 pool (nested across ratios)',
                              'source_episode_ids_in_merged_pool': eps, 'ratio': ratio})
    np.savez_compressed(DDIR / f'{stem}_gxy.npz', obs=obs[eps], act=act[eps], lengths=lengths[eps], eval_goals=goals[eps], meta=np.asarray(json.dumps(m, sort_keys=True)))
    np.savez_compressed(DDIR / f'{stem}_sidecar.npz', route_realized=route[eps], source_episode=np.array(eps), meta=np.asarray(json.dumps({'ratio': ratio, 'nested': True})))
    det_rows = int((lengths[[e for e in eps if route[e] == 'detour']] - 1).sum()); all_rows = int((lengths[eps] - 1).sum())
    info['ratios'][ratio] = {'episodes': len(eps), 'detour_episodes': n_d, 'shortcut_episodes': n_s, 'episode_share_detour': n_d / len(eps), 'row_share_detour': det_rows / all_rows,
                             'rows': all_rows, 'detour_episode_ids': sorted(det_order[:n_d]), 'gxy_sha256': sha256(DDIR / f'{stem}_gxy.npz'), 'sidecar_sha256': sha256(DDIR / f'{stem}_sidecar.npz')}
    print(ratio, {k: v for k, v in info['ratios'][ratio].items() if k != 'detour_episode_ids'}, flush=True)
  (out / 'datasets.json').write_text(json.dumps(info, indent=1), encoding='utf-8')


# -------------------------------------------------------------------- seal
def seal(args):
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed)')
  info = json.loads((out / 'datasets.json').read_text(encoding='utf-8'))
  ec = json.loads((B.OUT / 'exp_episode_coverage' / 'manifest.json').read_text(encoding='utf-8'))
  man = {'experiment': 'detour-demonstration share 5% vs 20% (nested 1,000-episode datasets): BC walker, vanilla CRL and branch critics, actors', 'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'git': git_info(), 'datasets': info,
         'provenance': {'merged_pool': str(B.DATASET), 'merged_pool_sha256': sha256(B.DATASET), 'cnew_reserved_episodes': ec['episodes']['new_held_out_detour (reserved, Cnew)'],
                        'scripts': {f: sha256(ROOT / f) for f in ('scripts/exp_v6_detour_ratio.py', 'scripts/run_v6_branch_replay.py', 'scripts/build_v6_policy_replay.py', 'scripts/diag_v6_first_step_crossover.py')}},
         'stages': {'bc': 'run_v6_branch_replay joint: fresh actor, critic term off (V6_JOINT_FRESH=1, V6_BC_COEF=1.0, JOINT_MODE=frozen, critic stub = the dataset copy), 100k, seed 0; balanced BC rows (displacement, cell 4, cap 0.25)',
                    'bc_eval': '300 natural draws (seed 909, mean policy): success / failure / timeout / detour; placed-on-detour completion (diag walker mode, 100 rows per leg from the Cnew reserved episodes, paired hazards) against the generation walker',
                    'vanilla_critic': 'run_v6_branch_replay vanilla: V6 recipe at gamma 0.999 on the dataset, 30k, seeds 0-2',
                    'branch_replay': 'build_v6_policy_replay build (r1 protocol: 1,500 dense anchors x {recorded, sample0, sample1} x 2 draws + every-60th general anchors x {recorded, sample0} x 1 draw; no hold-out split), continuation = this dataset\'s BC walker mode',
                    'branch_critic': 'run_v6_branch_replay critics on that replay, 30k, seeds 0-2, uniform anchors (row 0)',
                    'actors': 'both critics: pure-BC init (this dataset\'s), frozen critic, original actor loss, bc 0.05, balanced BC rows from this dataset, 30k, seeds 0-2; vanilla actors\' critic batches from the dataset, branch actors\' from the replay',
                    'eval': '300 natural draws, mean policy; success, failure, timeout, detour, discounted, mouth timing',
                    'shares': 'audit mode: dataset episode / row shares; vanilla anchor law (episode-uniform); replay anchor mass by stratum and the general anchors\' detour share; balanced BC sampler draws (100 x 1024)'},
         'fixed': {'env': 'p_active 0.5 / 0.5, horizon 800', 'action': '8-d single-step torque', 'losses': 'NCE + original actor loss, bc 0.05', 'gamma': 0.999, 'budgets': {'bc': 100_000, 'critic': 30_000, 'actor': 30_000}, 'no_sweeps': True},
         'decision_rules': {'effect': 'mean over 3 seeds differs by > 2 x pooled seed s.e. with 3 / 3 seeds in the same direction',
                            'branch_minus_vanilla_within_ratio': 'attributes a gain to the counterfactual futures', 'd20_minus_d05_within_method': 'attributes a gain to the demonstrations',
                            'bc_control': 'informative only (single seed): route choice from the start and placed completion'},
         'nodes': {'d05': 'node 30021 (RTX 4090)', 'd20': 'node 30043 (RTX 4090 laptop)'}}
  (out / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
  print('sealed', flush=True)


# ------------------------------------------------------------------- audit
def audit(args):
  """Effective detour-episode shares at every level for one ratio (run with V6_DATASET_STEM = that ratio)."""
  ratio = args.ratio
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths']
  with np.load(B.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  det = route == 'detour'
  res = {'ratio': ratio, 'dataset': {'episode_share_detour': float(det.mean()), 'row_share_detour': float((lengths[det] - 1).sum() / (lengths - 1).sum())}}
  # vanilla critic anchor law: episode uniform, then row uniform -> episode share
  res['vanilla_critic_anchor_share_detour'] = float(det.mean())
  # branch replay: anchor mass by stratum and the detour share of the general anchors
  rp = B.OUT / f'replay_policy_{ratio}.npz'
  if rp.exists():
    with np.load(rp, allow_pickle=False) as d:
      kind = d['audit_kind'].astype(str); ep = d['audit_episode'].astype(int)
    w = np.ones(len(kind)); tot = w.sum()
    res['branch_replay'] = {'paths': int(len(kind)), 'mass_by_stratum': {s: float((kind == s).sum() / tot) for s in np.unique(kind)},
                            'general_share_detour_episode_paths': float(det[ep[kind == 'general']].mean()),
                            'total_share_paths_from_detour_episodes': float(det[ep].mean())}
  # balanced BC sampler draws
  import run_v6_branch_replay as D
  from crl.bc_balanced import GroupBalancedBCSampler
  cfg = D.joint_config(0, 1)
  from crl import envs as envs_mod
  envs_mod.make_env(cfg.env_name, cfg, seed=1)
  smp = GroupBalancedBCSampler(obs, act, lengths, cfg.discount, cfg.obs_dim, cell=cfg.bc_balance_cell, n_sectors=cfg.bc_balance_sectors, wait_eps=cfg.bc_balance_wait_eps,
                               cap=cfg.bc_balance_cap, seed=10_000, region_mode=cfg.bc_balance_region, goal_indices=getattr(cfg, 'goal_indices', None))
  rng = np.random.default_rng(0); shares = []
  for _ in range(100):
    pos = np.searchsorted(smp.cdf, rng.random(1024), side='right'); pos = np.minimum(pos, len(smp.cdf) - 1)
    shares.append(det[smp.tr[pos]].mean())
  res['balanced_bc_sampler_share_detour_rows'] = {'mean': float(np.mean(shares)), 'sd_over_batches': float(np.std(shares))}
  res['bc_sampler_stats'] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in smp.stats.items()}
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  (out / f'shares_{ratio}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
  print(json.dumps(res, indent=1, default=float), flush=True)


# ------------------------------------------------------------------ report
def _headline(path):
  import run_v6_branch_replay as D
  ev = json.loads(Path(path).read_text(encoding='utf-8'))
  return D._headline(ev['summary'], ev.get('episodes'))


def report(args):
  out = exp_dir(); O = B.OUT
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  f_ = lambda v, fmt='.3f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
  L = ['# Detour-demonstration share 5% vs 20%: BC walker, vanilla and branch critics, actors', '',
       f'Sealed {man["sealed_at"]}.  Nested 1,000-episode datasets from the p050 pool; same recipe per dataset.  '
       'Reading rules: effect = mean over 3 seeds > 2 x pooled seed s.e. with 3/3 seeds in the same direction.', '', '## Effective shares of detour-episode rows', '',
       '| ratio | dataset episodes | dataset rows | vanilla critic anchors | replay: paths from detour episodes | replay general anchors | balanced BC draws |', '|---|---:|---:|---:|---:|---:|---:|']
  for r in RATIOS:
    p = out / f'shares_{r}.json'
    if p.exists():
      s = json.loads(p.read_text(encoding='utf-8')); br = s.get('branch_replay', {})
      L.append(f'| {r} | {s["dataset"]["episode_share_detour"]:.3f} | {s["dataset"]["row_share_detour"]:.3f} | {s["vanilla_critic_anchor_share_detour"]:.3f} | '
               f'{f_(br.get("total_share_paths_from_detour_episodes"))} | {f_(br.get("general_share_detour_episode_paths"))} | {s["balanced_bc_sampler_share_detour_rows"]["mean"]:.3f} |')
  L += ['', '## BC walkers (single seed): route choice from the start (300 natural draws) and placed completion (Cnew rows)', '',
        '| ratio | success | failure | timeout | detour | shortcut | placed north: BC reach (generation) | placed east: BC reach (generation) |', '|---|---:|---:|---:|---:|---:|---|---|']
  for r in RATIOS:
    p = O / f'joint_purebc_{r}' / 'seed_0' / 'eval_mean.json'
    if not p.exists():
      continue
    h = _headline(p); wk = O / f'diag_walker_{r}' / 'summary.json'
    w = json.loads(wk.read_text(encoding='utf-8')) if wk.exists() else {}
    def cell(leg):
      b = w.get(leg, {}).get('bc') or w.get(f'{leg}|bc') or {}; g = w.get(leg, {}).get('generation') or w.get(f'{leg}|generation') or {}
      return f'{f_(b.get("reach"))} ({f_(g.get("reach"))})'
    L.append(f'| {r} | {f_(h["success_rate"])} | {f_(h["failure_rate"])} | {f_(h["timeout_rate"])} | {f_(h["detour_rate"])} | {f_(h["shortcut_rate"])} | {cell("north")} | {cell("east")} |')
  L += ['', '## Actors (300 natural draws, mean policy; pure-BC init, frozen critic, bc 0.05, 30k)', '',
        '| ratio | critic | seed | success | failure | timeout | detour | discounted |', '|---|---|---|---:|---:|---:|---:|---:|']
  stats = {}
  for r in RATIOS:
    for meth in ('van', 'br'):
      for s in (0, 1, 2):
        p = O / f'joint_{meth}_{r}' / f'seed_{s}' / 'eval_mean.json'
        if not p.exists():
          continue
        h = _headline(p); stats.setdefault((r, meth), []).append(h)
        L.append(f'| {r} | {"vanilla" if meth == "van" else "branch"} | {s} | {f_(h["success_rate"])} | {f_(h["failure_rate"])} | {f_(h["timeout_rate"])} | {f_(h["detour_rate"])} | {f_(h.get("discounted"))} |')
  if stats:
    L += ['', '| ratio | critic | mean success (seed s.e.) | mean detour (seed s.e.) |', '|---|---|---|---|']
    agg = {}
    for (r, meth), hs in stats.items():
      su = np.array([h['success_rate'] for h in hs]); de = np.array([h['detour_rate'] for h in hs]); agg[(r, meth)] = (su, de)
      se = lambda v: (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
      L.append(f'| {r} | {"vanilla" if meth == "van" else "branch"} | {su.mean():.3f} ({se(su):.3f}) | {de.mean():.3f} ({se(de):.3f}) |')
    L += ['', '## Pre-registered comparisons', '']
    def cmp(a, b, label):
      if a in agg and b in agg:
        for qi, qn in ((0, 'success'), (1, 'detour')):
          x, y = agg[a][qi], agg[b][qi]
          if len(x) > 1 and len(y) > 1:
            pse = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / len(x)); d = x.mean() - y.mean(); same = int(np.sum(np.sign(x - y) == np.sign(d))) if d != 0 else 0
            L.append(f'- {label}, {qn}: {d:+.3f} (pooled seed s.e. {pse:.3f}; {"> 2 s.e." if abs(d) > 2 * pse else "not > 2 s.e."}; same-direction seeds {same}/3)')
    cmp(('d05', 'br'), ('d05', 'van'), 'd05 branch - vanilla'); cmp(('d20', 'br'), ('d20', 'van'), 'd20 branch - vanilla')
    cmp(('d20', 'van'), ('d05', 'van'), 'vanilla d20 - d05'); cmp(('d20', 'br'), ('d05', 'br'), 'branch d20 - d05')
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('datasets', 'seal', 'audit', 'report'))
  ap.add_argument('--ratio', choices=tuple(RATIOS), default='d05')
  ap.add_argument('--seed', type=int, default=2028)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'datasets': datasets, 'seal': seal, 'audit': audit, 'report': report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
