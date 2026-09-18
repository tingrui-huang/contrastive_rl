"""AntMaze V6: actor updates under IDENTICAL batches, frozen critics only differ.

In the agent-update round each actor's critic-term batches came from its
critic's own training data (row-0 anchors of the branch replay for the
branch critics; row-0 anchors of the dataset -- the reset rows -- for the
recorded-data critics), so the actors differed in the states they were
updated at, not only in the critic.  Here every actor starts from the fixed
agent and is updated with the SAME (state, goal) batch stream, the SAME
balanced-BC batch stream and the SAME JAX key (seed-determined samplers:
TrajectoryBuffer seed = seed, GroupBalancedBCSampler seed = 10000 + seed,
state key = PRNGKey(30000 + seed)); only the frozen critic changes.

  batch source S1 = the d20 dataset with row-0 anchors (what the reference
                    actors used: critic term at the reset rows)
  batch source S2 = the control replay (what the control actors used)
  critics         = vanilla (critics_van_d20), control (critics_br_d20),
                    round1, ext_ag, ext_bc

  python scripts/exp_v6_same_batch.py audit_batches   (proves the streams are identical across critic tags)
  python scripts/exp_v6_same_batch.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402

EXP = 'exp_same_batch'
EVAL_SEED = 2909
CRITICS = (('vanilla', 'critics_van_d20'), ('control', 'critics_br_d20'), ('round1', 'critics_round1'), ('ext_ag', 'critics_ext_ag'), ('ext_bc', 'critics_ext_bc'))
SOURCES = {'S1_dataset_row0': 'critic_stub_d20.npz', 'S2_control_replay': 'replay_policy_d20.npz'}
# actor directories: S1 -> joint_same_<critic> (vanilla = the existing joint_vanref_round1); S2 -> joint_samer_<critic> (control = the existing joint_ctrl_round1)
ACTOR_DIRS = {('S1_dataset_row0', 'vanilla'): 'joint_vanref_round1', ('S2_control_replay', 'control'): 'joint_ctrl_round1'}


def actor_dir(source, critic):
  if (source, critic) in ACTOR_DIRS:
    return ACTOR_DIRS[(source, critic)]
  return f'joint_same_{critic}' if source == 'S1_dataset_row0' else f'joint_samer_{critic}'


def _h(x):
  return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def audit_batches(args):
  """Build the joint stage's buffer and BC sampler for two critic tags and compare the first draws."""
  import run_v6_branch_replay as D
  from crl import offline_audit
  from crl.bc_balanced import GroupBalancedBCSampler
  res = {}
  for src, path in SOURCES.items():
    os.environ['V6_BRANCH_REPLAY'] = str(B.OUT / path)
    import importlib; importlib.reload(D)
    draws = {}
    for tag in ('_van_d20', '_br_d20'):
      os.environ['V6_CRITIC_TAG'] = tag; os.environ['V6_JOINT_TAG'] = f'_audit{tag}'; os.environ['V6_JOINT_MODE'] = 'frozen'
      importlib.reload(D)
      cfg = D.joint_config(args.seed, 30000)
      from crl import envs as envs_mod
      envs_mod.make_env(cfg.env_name, cfg, seed=1)          # populates obs / goal / action dims (as prep_joint does)
      buffer, _ = offline_audit.build_offline_buffer(cfg.offline_dataset, cfg, prepare=D.row0_prepare)
      with np.load(cfg.bc_dataset, allow_pickle=False) as d:
        obs_b, act_b, len_b = d['obs'], d['act'], np.asarray(d['lengths']).astype(np.int64)
      bc = GroupBalancedBCSampler(obs_b, act_b, len_b, cfg.discount, cfg.obs_dim, cell=cfg.bc_balance_cell, n_sectors=cfg.bc_balance_sectors, wait_eps=cfg.bc_balance_wait_eps,
                                  cap=cfg.bc_balance_cap, seed=10_000 + int(cfg.seed), region_mode=cfg.bc_balance_region, goal_indices=getattr(cfg, 'goal_indices', None))
      hs = []
      for _ in range(args.n_batches):
        b = buffer.sample(cfg.batch_size); c = bc.sample(cfg.batch_size)
        hs.append({'critic_batch': _h(np.concatenate([np.asarray(b.observation).ravel(), np.asarray(b.action).ravel()])), 'bc_batch': _h(np.concatenate([np.asarray(c.observation).ravel(), np.asarray(c.action).ravel()]))})
      draws[tag] = hs
    same = all(draws['_van_d20'][i] == draws['_br_d20'][i] for i in range(args.n_batches))
    res[src] = {'replay': path, 'n_batches': args.n_batches, 'identical_across_critic_tags': bool(same), 'hashes': draws['_van_d20']}
    print(src, 'identical across critic tags:', same, flush=True)
  out = B.OUT / EXP; out.mkdir(parents=True, exist_ok=True)
  (out / 'batch_audit.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


def _eval(path):
  import run_v6_branch_replay as D
  ev = json.loads(Path(path).read_text(encoding='utf-8'))
  h = D._headline(ev['summary'], ev.get('episodes')); s = ev['summary']; bl = s['by_latent']
  haz = [k for k in bl if k != 'U00']
  h['no_hazard_success'] = float(bl['U00']['success']); h['hazard_success'] = float(sum(bl[k]['success_n'] for k in haz) / max(1, sum(bl[k]['n'] for k in haz)))
  h['zone_deaths'] = (int(s['overall']['zone1_deaths']), int(s['overall']['zone2_deaths']))
  return h


def report(args):
  out = B.OUT / EXP; out.mkdir(parents=True, exist_ok=True)
  f_ = lambda v, fmt='.3f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
  start = _eval(B.OUT / 'joint_van_d20' / 'seed_0' / f'eval_mean_s{EVAL_SEED}.json')
  L = ['# Identical batches, frozen critics only differ: actors from the fixed agent on the 2909 draw', '',
       f'Start (fixed agent): success {start["success_rate"]:.3f}, detour {start["detour_rate"]:.3f}, timeout {start["timeout_rate"]:.3f}, no-hazard {start["no_hazard_success"]:.3f}, hazard {start["hazard_success"]:.3f}.  '
       'Batch audit: `batch_audit.json`.  Rule: mean over 3 seeds, > 2 x pooled seed s.e. and 3 / 3 seeds.', '']
  stats = {}
  for src, path in SOURCES.items():
    L += [f'## Batch source {src} ({path})', '', '| critic | seed | success | failure | timeout | detour | no-hazard success | hazard success | zone-1 / zone-2 deaths |', '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for name, cdir in CRITICS:
      hs = []
      for s in (0, 1, 2):
        p = B.OUT / actor_dir(src, name) / f'seed_{s}' / f'eval_mean_s{EVAL_SEED}.json'
        if not p.exists():
          continue
        h = _eval(p); hs.append(h)
        L.append(f'| {name} | {s} | {h["success_rate"]:.3f} | {h["failure_rate"]:.3f} | {h["timeout_rate"]:.3f} | {h["detour_rate"]:.3f} | {h["no_hazard_success"]:.3f} | {h["hazard_success"]:.3f} | {h["zone_deaths"][0]} / {h["zone_deaths"][1]} |')
      if hs:
        stats[(src, name)] = (np.array([h['success_rate'] for h in hs]), np.array([h['detour_rate'] for h in hs]))
    L.append('')
    if any(k[0] == src for k in stats):
      se = lambda v: (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
      L += ['| critic | mean success (seed s.e.) | vs start | mean detour (seed s.e.) |', '|---|---|---|---|']
      for name, _ in CRITICS:
        if (src, name) in stats:
          su, de = stats[(src, name)]; d = su.mean() - start['success_rate']; same = int(np.sum(np.sign(su - start['success_rate']) == np.sign(d))) if d != 0 else 0
          L.append(f'| {name} | {su.mean():.3f} ({se(su):.3f}) | {d:+.3f} ({"> 2 s.e." if abs(d) > 2 * se(su) and se(su) > 0 else "not > 2 s.e."}; {same}/{len(su)}) | {de.mean():.3f} ({se(de):.3f}) |')
      L += ['', 'Pairwise (success):']
      names = [n for n, _ in CRITICS if (src, n) in stats]
      for i, a in enumerate(names):
        for b in names[i + 1:]:
          x, y = stats[(src, a)][0], stats[(src, b)][0]
          if len(x) > 1 and len(y) > 1:
            pse = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / len(x)); d = x.mean() - y.mean(); same = int(np.sum(np.sign(x - y) == np.sign(d))) if d != 0 else 0
            L.append(f'- {a} - {b}: {d:+.3f} (pooled seed s.e. {pse:.3f}; {"> 2 s.e." if abs(d) > 2 * pse else "not > 2 s.e."}; {same}/3)')
      L.append('')
  # the same critic across sources
  L += ['## Same critic, different batch source (S1 - S2, success)', '']
  for name, _ in CRITICS:
    if ('S1_dataset_row0', name) in stats and ('S2_control_replay', name) in stats:
      x, y = stats[('S1_dataset_row0', name)][0], stats[('S2_control_replay', name)][0]
      pse = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / len(x)); d = x.mean() - y.mean(); same = int(np.sum(np.sign(x - y) == np.sign(d))) if d != 0 else 0
      L.append(f'- {name}: {d:+.3f} (pooled seed s.e. {pse:.3f}; {"> 2 s.e." if abs(d) > 2 * pse else "not > 2 s.e."}; {same}/3)')
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('audit_batches', 'report'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-batches', type=int, default=3)
  args = ap.parse_args(argv)
  {'audit_batches': audit_batches, 'report': report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
