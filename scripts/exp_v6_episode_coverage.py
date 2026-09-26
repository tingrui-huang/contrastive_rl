"""AntMaze V6: does spreading the counterfactual supervision over more
INDEPENDENT original episodes improve critic generalisation to common
held-out episodes (turning / north-leg states first)?  NARROW vs BROAD at
equal anchor counts, equal branch budget, equal weights.

Plan A (user-approved after the feasibility audit): 5,000 additional p050
episodes collected with the ORIGINAL collector settings (p_active 0.5 /
0.5, teacher detour probability 0.05, t0 ranges, horizon 800; seeds
607-611, five shards) -- the source behaviour mode, route proportion and
environment are unchanged, so the episode count is the only intervention.

Arms (three seeds each, random init, 30k NCE, the R protocol):
  NARROW  the R replay as it is: turning / north-leg anchors from the 49
          training detour episodes of the original dataset (300 + 300
          anchors, median 5-6 per episode); retrained on the same GPU as
          BROAD (the old critics_armR are not used as the control).
  BROAD   the same replay with ONLY the turning / north-leg dense anchors
          replaced: 300 + 300 anchors drawn round-robin over the enlarged
          detour pool (the 49 original + the new shards' training detour
          episodes, ~2 anchors per episode), every other stratum, the
          general anchors, the candidates (recorded + 2 policy samples),
          the draws (2 per recorded torque, 8 per policy torque), the
          anchor weights (1 / 0.25), the continuation and the law unchanged.

Held-out evaluation ("Cnew"): 30 detour episodes reserved from the new
shards (never in any training pool), 32 turning + 32 north-leg anchors,
candidates recorded + mode + 3 policy samples, 16 fresh paired draws --
the diagnostic's protocol (`scripts/diag_v6_r1_abc.py`).  The existing C
set is a development check.  A / B membership is reclassified: the
turning / north-leg A anchors of the diagnostic are NARROW training keys
but not BROAD training keys.

Modes: merge (datasets) -> seal (manifest, before any rollout) -> build
(BROAD replay + Cnew outcomes) -> (critics via run_v6_branch_replay.py)
-> analyze -> report.  Oracle-stage diagnostic; no actor unless the
pre-registered critic criterion passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402
import diag_v6_r1_abc as ABC  # noqa: E402

EXP = 'exp_episode_coverage'
DIAG_NEW = 'diag_cnew'
MERGED_STEM = os.environ.get('V6_EC_MERGED_STEM', 'antmaze_rockfall_clock_v6_p050_plus5k')
SHARD_SEEDS = tuple(int(x) for x in os.environ.get('V6_EC_SHARDS', '607,608,609,610,611').split(','))
STRATA_INTERVENED = ('turn', 'north_leg')
N_PER_STRATUM = {'turn': 300, 'north_leg': 300}
N_HELD_DETOUR = 30
N_EVAL_PER_STRATUM = 32
CAND_SEED = 818_000_003        # BROAD policy samples: base + 7919 * anchor_id (new anchor ids >= 2_000_000)
DRAW_SEED = 818_500_000        # BROAD training draws: base + 1000 * (anchor_id - 2_000_000) + draw
EVAL_CAND_SEED = 919_000_005   # Cnew candidates
EVAL_DRAW_SEED = 919_500_000   # Cnew draws: base + 100 * index + draw
NEW_AID0 = 2_000_000
CNEW_AID0 = 3_000_000
NEW_WEIGHT = 0.25


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


# ------------------------------------------------------------------- merge
def merge(args):
  """Original p050 (episodes 0-999) + shards (1000 each, in seed order) -> one gxy + sidecar."""
  ddir = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset'
  parts = [('antmaze_rockfall_clock_v6_p050', 606)] + [(f'antmaze_rockfall_clock_v6_p050_s{s}', s) for s in SHARD_SEEDS]
  obs, act, lengths, goals, routes, src_seed, src_ep, audits = [], [], [], [], [], [], [], {}
  L = None
  for stem, seed in parts:
    with np.load(ddir / f'{stem}_gxy.npz', allow_pickle=False) as d:
      o, a, ln, g = d['obs'], d['act'], d['lengths'], d['eval_goals']
      if L is None:
        L = o.shape[1]; meta = json.loads(str(d['meta']))
      assert o.shape[1] == L, (stem, o.shape)
    with np.load(ddir / f'{stem}_sidecar.npz', allow_pickle=True) as sc:
      r = sc['route_realized'].astype(str)
      for k in ('p_active_1', 'p_active_2', 'teacher_detour_prob', 'horizon', 'environment_version', 'collection_seed'):
        v = sc[k].item() if sc[k].shape == () else sc[k].tolist()
        audits.setdefault(k, {})[stem] = v
    aud = ddir / f'{stem}_composition_audit.json'
    audits.setdefault('composition_audit', {})[stem] = json.loads(aud.read_text(encoding='utf-8')) if aud.exists() else None
    obs.append(o); act.append(a); lengths.append(ln); goals.append(g); routes.append(r)
    src_seed.append(np.full(len(ln), seed)); src_ep.append(np.arange(len(ln)))
    print(f'{stem}: {len(ln)} episodes, detour {(r == "detour").sum()}, seed {seed}', flush=True)
  # identical settings across parts
  for k in ('p_active_1', 'p_active_2', 'teacher_detour_prob', 'horizon', 'environment_version'):
    assert len(set(map(str, audits[k].values()))) == 1, (k, audits[k])
  obs = np.concatenate(obs); act = np.concatenate(act); lengths = np.concatenate(lengths); goals = np.concatenate(goals); routes = np.concatenate(routes)
  meta.update({'n_episodes': int(len(lengths)), 'n_transitions': int((lengths - 1).sum()), 'merged_from': [p[0] for p in parts],
               'merged_seeds': [p[1] for p in parts], 'note': 'episodes 0-999 = the original p050 dataset; shard k occupies 1000k..1000k+999'})
  np.savez_compressed(ddir / f'{MERGED_STEM}_gxy.npz', obs=obs, act=act, lengths=lengths, eval_goals=goals, meta=np.asarray(json.dumps(meta, sort_keys=True)))
  np.savez_compressed(ddir / f'{MERGED_STEM}_sidecar.npz', route_realized=routes, source_seed=np.concatenate(src_seed), source_episode=np.concatenate(src_ep),
                      meta=np.asarray(json.dumps({'merged_from': [p[0] for p in parts], 'settings': {k: v for k, v in audits.items() if k != 'composition_audit'}})))
  summary = {'episodes': int(len(lengths)), 'detour': int((routes == 'detour').sum()), 'per_part': {p[0]: {'episodes': int(len(rr)), 'detour': int((rr == 'detour').sum())} for p, rr in zip(parts, [r for r in np.split(routes, np.cumsum([1000] * (len(parts) - 1)))])},
             'audits': audits, 'merged_gxy_sha256': sha256(ddir / f'{MERGED_STEM}_gxy.npz'), 'merged_sidecar_sha256': sha256(ddir / f'{MERGED_STEM}_sidecar.npz')}
  exp_dir().mkdir(parents=True, exist_ok=True)
  (exp_dir() / 'merge_summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
  print(json.dumps({k: summary[k] for k in ('episodes', 'detour', 'per_part')}, indent=1), flush=True)


# -------------------------------------------------------------------- seal
def seal(args):
  """Requires V6_DATASET_STEM = the merged dataset."""
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed)')
  assert MERGED_STEM in str(B.DATASET), f'V6_DATASET_STEM must be the merged dataset, got {B.DATASET}'
  obs, act, lengths, meta, route = R.load_dataset()
  n_orig = 1000
  r_replay = B.OUT / 'exp_r1_coverage' / 'replay_armR.npz'
  old_held = set(json.loads(str(np.load(B.OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  rng = np.random.default_rng(args.seed)
  # --- NARROW anchors = the R replay's dense turn / north-leg anchors (from the R audit)
  with np.load(r_replay, allow_pickle=False) as d:
    kind = d['audit_kind'].astype(str); aid = d['audit_anchor_id'].astype(int); ep = d['audit_episode'].astype(int); at = d['audit_anchor_time'].astype(int)
    cand = d['audit_cand'].astype(str); a0 = d['act'][:, 0]
  narrow = {}
  for i in range(len(kind)):
    if kind[i] in STRATA_INTERVENED:
      e = narrow.setdefault(int(aid[i]), {'episode': int(ep[i]), 't': int(at[i]), 'stratum': kind[i], 'torques': {}})
      e['torques'][cand[i]] = a0[i].astype(float).tolist()
  narrow_eps = {s: sorted({v['episode'] for v in narrow.values() if v['stratum'] == s}) for s in STRATA_INTERVENED}
  # --- the enlarged detour pool: original training detour episodes + new shards' detour episodes, minus the reserved evaluation episodes
  new_det = [e for e in range(n_orig, len(route)) if route[e] == 'detour']
  held_new = sorted(rng.choice(new_det, size=N_HELD_DETOUR, replace=False).tolist())
  train_new = [e for e in new_det if e not in set(held_new)]
  orig_train_det = [e for e in range(n_orig) if route[e] == 'detour' and e not in old_held]
  pool = sorted(orig_train_det + train_new)
  # --- BROAD anchors: round-robin over the pool within each intervened stratum (outcomes never consulted)
  masks = R.dense_masks(obs, lengths, route, np.isin(np.arange(len(route)), pool))
  broad, aid_new = {}, NEW_AID0
  for s in STRATA_INTERVENED:
    rows = {}
    for e, t in zip(*np.nonzero(masks[s])):
      rows.setdefault(int(e), []).append(int(t))
    picked = R_select_round_robin(rows, N_PER_STRATUM[s], rng)
    for e, t in picked:
      broad[aid_new] = {'episode': e, 't': t, 'stratum': s}; aid_new += 1
  # BROAD candidates: recorded + 2 policy samples at the anchor (new stream)
  bundle = R.policy_bundle(args.cont_ckpt)
  ids = sorted(broad)
  o0 = np.stack([obs[broad[a]['episode'], broad[a]['t']] for a in ids])
  loc, scale = bundle['params_fn'](o0)
  for i, a in enumerate(ids):
    e, t = broad[a]['episode'], broad[a]['t']
    cs = R.candidates_for(loc[i], scale[i], act[e, t], a, ['recorded', 'sample0', 'sample1'], CAND_SEED)
    broad[a]['torques'] = {c: v.astype(float).tolist() for c, v in cs}
    broad[a]['obs'] = o0[i].astype(float).tolist()
  # --- Cnew evaluation anchors on the reserved episodes (diagnostic manifest format)
  mh = R.dense_masks(obs, lengths, route, np.isin(np.arange(len(route)), held_new))
  eval_anchors, idx = [], 0
  for s in STRATA_INTERVENED:
    rows = {}
    for e, t in zip(*np.nonzero(mh[s])):
      rows.setdefault(int(e), []).append(int(t))
    for e, t in R_select_round_robin(rows, N_EVAL_PER_STRATUM, rng):
      o = obs[e, t]
      l_, s_ = bundle['params_fn'](o[None]); l_, s_ = l_[0], s_[0]
      cs = R.candidates_for(l_, s_, act[e, t], CNEW_AID0 + idx, ['recorded', 'mode', 'sample0', 'sample1', 'sample2'], EVAL_CAND_SEED)
      eval_anchors.append({'index': idx, 'layer': 'C', 'anchor_id': CNEW_AID0 + idx, 'episode': int(e), 't': int(t), 'stratum': s,
                           'obs': o.astype(float).tolist(), 'goal_xy': o[B.STATE_DIM:B.STATE_DIM + 2].astype(float).tolist(),
                           'policy_loc': l_.astype(float).tolist(), 'policy_scale': s_.astype(float).tolist(), 'policy_mode': np.tanh(l_).astype(float).tolist(),
                           'candidates': {f'C|{c}': v.astype(float).tolist() for c, v in cs}, 'original_outcomes': {},
                           'previously_examined': 'never (reserved evaluation episodes of the new shards)',
                           'draw_seeds': [EVAL_DRAW_SEED + 100 * idx + r for r in range(ABC.N_DRAWS)]})
      idx += 1
  # candidate stats for the diagnostic's coverage table
  import jax.numpy as jnp
  from crl import networks
  for rec in eval_anchors:
    l_ = np.asarray(rec['policy_loc'], np.float32); s_ = np.asarray(rec['policy_scale'], np.float32); mode = np.asarray(rec['policy_mode'], np.float32)
    A_rec = np.asarray(rec['candidates']['C|recorded'], np.float32); rec['candidate_stats'] = {}
    for c, v in rec['candidates'].items():
      v = np.asarray(v, np.float32)
      lp = float(np.asarray(networks.tanh_normal_log_prob(networks.TanhNormalParams(jnp.asarray(l_[None]), jnp.asarray(s_[None])), jnp.asarray(v[None])))[0])
      rec['candidate_stats'][c] = {'log_prob_policy': lp, 'dist_to_mode': float(np.linalg.norm(v - mode)), 'dist_to_recorded': float(np.linalg.norm(v - A_rec)), 'frac_saturated': float(np.mean(np.abs(v) > 0.95))}
  # --- coverage bookkeeping
  broad_eps = {s: sorted({v['episode'] for v in broad.values() if v['stratum'] == s}) for s in STRATA_INTERVENED}
  coverage = {s: {'narrow_anchors': sum(v['stratum'] == s for v in narrow.values()), 'narrow_episodes': len(narrow_eps[s]),
                  'broad_anchors': sum(v['stratum'] == s for v in broad.values()), 'broad_episodes': len(broad_eps[s]),
                  'broad_anchors_per_episode_max': max(sum(1 for v in broad.values() if v['stratum'] == s and v['episode'] == e) for e in broad_eps[s]),
                  'pool_episodes': len([e for e in pool if masks[s][e].any()]), 'eval_episodes': len({a['episode'] for a in eval_anchors if a['stratum'] == s}),
                  'eval_anchors': sum(a['stratum'] == s for a in eval_anchors)} for s in STRATA_INTERVENED}
  # remaining distribution differences: anchor time and goal
  def tstats(d, s):
    ts = np.array([v['t'] for v in d.values() if v['stratum'] == s]); return [int(ts.min()), float(np.median(ts)), int(ts.max())]
  diffs = {s: {'narrow_t_min_med_max': tstats(narrow, s), 'broad_t_min_med_max': tstats(broad, s),
               'narrow_goal_mean': np.mean([obs[v['episode'], v['t'], B.STATE_DIM:B.STATE_DIM + 2] for v in narrow.values() if v['stratum'] == s], axis=0).round(3).tolist(),
               'broad_goal_mean': np.mean([obs[v['episode'], v['t'], B.STATE_DIM:B.STATE_DIM + 2] for v in broad.values() if v['stratum'] == s], axis=0).round(3).tolist()} for s in STRATA_INTERVENED}
  # weights (both arms): identical to R by construction (same anchor counts, same per-path weights)
  n_paths_new = len(broad) * (2 + 2 * 8)
  man = {'experiment': 'episode coverage NARROW vs BROAD (turning / north-leg anchors over 49 vs the enlarged detour pool), R protocol', 'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git': git_info(),
         'provenance': {'merged_dataset': str(B.DATASET), 'merged_dataset_sha256': sha256(B.DATASET), 'merged_sidecar_sha256': sha256(B.SIDECAR),
                        'merge_summary': json.loads((out / 'merge_summary.json').read_text(encoding='utf-8')) if (out / 'merge_summary.json').exists() else None,
                        'r_replay': str(r_replay), 'r_replay_sha256': sha256(r_replay), 'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': sha256(args.cont_ckpt),
                        'abc_manifest_sha256': sha256(B.OUT / 'diag_r1_abc' / 'manifest.json'), 'abc_outcomes_sha256': sha256(B.OUT / 'diag_r1_abc' / 'outcomes.npz'),
                        'scripts': {f: sha256(ROOT / f) for f in ('scripts/exp_v6_episode_coverage.py', 'scripts/run_v6_branch_replay.py', 'scripts/diag_v6_r1_abc.py', 'scripts/build_v6_policy_replay.py', 'scripts/probe_v6_coverage_paired_boot.py')}},
         'episodes': {'original_training_detour': orig_train_det, 'original_held_out': sorted(old_held), 'new_detour_total': len(new_det), 'new_held_out_detour (reserved, Cnew)': held_new,
                      'new_training_detour': train_new, 'broad_pool_size': len(pool), 'separation': 'Cnew episodes appear in no training pool of either arm; the old held-out episodes stay held out'},
         'arms': {'NARROW': {'replay': 'exp_r1_coverage/replay_armR.npz (verbatim)', 'critic_dir': 'critics_narrow', 'turn_north_anchors': {str(a): {k: v for k, v in narrow[a].items() if k != 'torques'} for a in sorted(narrow)}},
                  'BROAD': {'replay': 'exp_episode_coverage/replay_broad.npz', 'critic_dir': 'critics_broad',
                            'construction': 'R replay with its turn / north-leg dense paths removed and the BROAD anchors\' paths added (recorded 2 draws w 1.0, sample0 / sample1 8 draws w 0.25 each); every other path verbatim',
                            'turn_north_anchors': {str(a): {k: v for k, v in broad[a].items() if k not in ('obs',)} for a in sorted(broad)}}},
         'coverage': coverage, 'remaining_differences': diffs,
         'seeds': {'broad_candidate_seed_base': CAND_SEED, 'broad_draw_seed_rule': f'{DRAW_SEED} + 1000 * (anchor_id - {NEW_AID0}) + draw', 'cnew_candidate_seed_base': EVAL_CAND_SEED,
                   'cnew_draw_seed_rule': f'{EVAL_DRAW_SEED} + 100 * index + draw', 'critic_seeds': [0, 1, 2], 'selection_seed': args.seed,
                   'note': 'all bases distinct from every earlier stream (r1, hold-out, validate, ABC, arms R/COV, fresh-untouched); candidate stride 7919 checked'},
         'budget': {'broad_new_paths': n_paths_new, 'cnew_paths': len(eval_anchors) * 5 * ABC.N_DRAWS, 'critics': '2 arms x 3 seeds x 30k NCE, random init, V6_ANCHOR_WEIGHTS=audit, one GPU (node 30021)',
                    'expected_wall': 'rollouts ~15 min (15 workers), critics ~15 min, analysis ~10 min', 'storage': 'BROAD replay ~1.7 GB, Cnew outcomes < 1 MB', 'no_extension': True},
         'evaluation': {'primary': 'Cnew: 64 anchors on 30 reserved detour episodes, candidates recorded + mode + 3 samples, 16 paired draws; region-integrated readout (arm\'s own weighted marginal), '
                                   'agreement on identical decided pairs, pick gain as in the diagnostic, ties / weak reported, per seed, paired episode bootstrap NARROW vs BROAD',
                        'development': 'the existing C set (diag_r1_abc); A / B reclassified: turn / north-leg A anchors are NARROW keys, not BROAD keys',
                        'criterion': 'BROAD improves Cnew pick gain over NARROW by > 2 paired s.e., with positive absolute pick gain and consistent per-seed direction; agreement corroborates; '
                                     'three training seeds limit the training-randomness side of the uncertainty',
                        'actor_stage': 'only if the criterion passes: both arms, pure-BC init, frozen critics, bc 0.05, 30k, 300 matched episodes'},
         'cnew_diag_manifest': {'diagnostic': 'Cnew held-out anchors (reserved detour episodes of the new shards)', 'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'provenance': {'reference_commit': 'see git', 'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': sha256(args.cont_ckpt), 'critics': {}},
                                'continuation': {'mode': 'frozen deployment policy, tanh(loc) mode, closed-loop from step 2', 'horizon': B.HORIZON},
                                'law': {'discount': R.GAMMA, 'radius': ABC.RADIUS}, 'anchors': eval_anchors}}
  # seed-stream separation check (candidate bases mod stride)
  bases = {'r1': R.R1_SEED + 11, 'abc_B': ABC.ABC_SEED + 17, 'cov': 515_000_003, 'fresh': 616_000_005, 'broad': CAND_SEED, 'cnew': EVAL_CAND_SEED}
  for a in bases:
    for b in bases:
      if a < b:
        assert (bases[a] - bases[b]) % 7919 != 0, (a, b)
  (out / 'manifest.json').write_text(json.dumps(man, indent=1, default=float), encoding='utf-8')
  (B.OUT / DIAG_NEW).mkdir(parents=True, exist_ok=True)
  (B.OUT / DIAG_NEW / 'manifest.json').write_text(json.dumps(man['cnew_diag_manifest'], indent=1, default=float), encoding='utf-8')
  print(json.dumps({'coverage': coverage, 'remaining_differences': diffs, 'episodes': {k: (v if not isinstance(v, list) else len(v)) for k, v in man['episodes'].items()}}, indent=1), flush=True)


def R_select_round_robin(rows_by_episode, n, rng):
  eps = list(rows_by_episode); rng.shuffle(eps)
  for e in eps:
    rng.shuffle(rows_by_episode[e])
  out = []
  while len(out) < n and any(rows_by_episode.values()):
    for e in eps:
      if rows_by_episode[e] and len(out) < n:
        out.append((e, rows_by_episode[e].pop()))
  return out


# ------------------------------------------------------------------- build
def build(args):
  """BROAD replay (from the R replay) and the Cnew outcomes."""
  out = exp_dir(); man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  assert sha256(B.DATASET) == man['provenance']['merged_dataset_sha256']
  assert sha256(args.cont_ckpt) == man['provenance']['continuation_sha256']
  r_replay = Path(man['provenance']['r_replay'])
  broad = man['arms']['BROAD']['turn_north_anchors']
  jobs = []
  for a_s, rec in broad.items():
    a = int(a_s)
    for c, v in rec['torques'].items():
      nd = 2 if c == 'recorded' else 8
      for r in range(nd):
        jobs.append((a, rec['episode'], rec['t'], rec['stratum'], c, np.asarray(v, np.float32), r, DRAW_SEED + 1000 * (a - NEW_AID0) + r, True))
  jobs_c = []
  for rec in man['cnew_diag_manifest']['anchors']:
    for c, v in rec['candidates'].items():
      for r, seed in enumerate(rec['draw_seeds']):
        jobs_c.append((rec['anchor_id'], rec['episode'], rec['t'], rec['stratum'], c, np.asarray(v, np.float32), r, seed, False))
  if args.limit:
    jobs = jobs[:args.limit]; jobs_c = jobs_c[:args.limit]
  with np.load(r_replay, allow_pickle=False) as d:
    kind = d['audit_kind'].astype(str)
    keep = np.flatnonzero(~np.isin(kind, list(STRATA_INTERVENED)))
    L = d['obs'].shape[1]; meta = json.loads(str(d['meta']))
    n_tot = len(keep) + len(jobs)
    obs_p = np.zeros((n_tot, L, B.STATE_DIM + 2), np.float32); act_p = np.zeros((n_tot, L, B.ACTION_DIM), np.float32); lengths_p = np.zeros(n_tot, np.int64)
    obs_p[:len(keep)] = d['obs'][keep]; act_p[:len(keep)] = d['act'][keep]; lengths_p[:len(keep)] = d['lengths'][keep]
    aud = {k: list(d[f'audit_{k}'][keep]) for k in ('kind', 'cand', 'episode', 'anchor_time', 'anchor_id', 'draw', 'success', 'failure', 'u1', 'u2', 'p_goal', 'max_y')}
    weight = list(d['audit_weight'][keep])
  print(f'build BROAD: {len(keep)} R paths verbatim (turn / north-leg removed) + {len(jobs)} new paths + {len(jobs_c)} Cnew paths; {args.workers} workers', flush=True)
  t0 = time.time()
  parts = ABC.R.DG.chunks(jobs + jobs_c, args.workers * 6)
  arglist = [(p, DRAW_SEED + 7 + i, args.cont_ckpt) for i, p in enumerate(parts)]
  k = len(keep); c_rows = []
  with get_context('spawn').Pool(args.workers) as pool:
    for res in pool.imap_unordered(R._run_jobs, arglist):
      for r in res:
        if 'obs' in r:
          n = len(r['obs']); obs_p[k, :n] = r['obs']; act_p[k, :n] = r['act']; obs_p[k, n:] = r['obs'][-1]; lengths_p[k] = n
          for key, val in (('kind', r['set']), ('cand', r['cand']), ('episode', r['episode']), ('anchor_time', r['t']), ('anchor_id', r['anchor_id']), ('draw', r['draw']),
                           ('success', r['success']), ('failure', r['failure']), ('u1', r['u1']), ('u2', r['u2']), ('p_goal', r['p_goal']), ('max_y', r['max_y'])):
            aud[key].append(val)
          weight.append(1.0 if r['cand'] == 'recorded' else NEW_WEIGHT); k += 1
        else:
          c_rows.append(r)
  wall = time.time() - t0
  assert k == n_tot
  m = dict(meta); m.update({'arm': 'episode_coverage_BROAD', 'source_replay': str(r_replay), 'intervened_strata': list(STRATA_INTERVENED), 'draw_seed_base': DRAW_SEED, 'candidate_seed_base': CAND_SEED})
  w = np.array(weight)
  path = out / 'replay_broad.npz'
  np.savez_compressed(path, obs=obs_p, act=act_p, lengths=lengths_p, eval_goals=obs_p[:, 0, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float32), meta=np.asarray(json.dumps(m, sort_keys=True)),
                      audit_weight=w, audit_kind=np.array(aud['kind']), audit_cand=np.array(aud['cand']), audit_episode=np.array(aud['episode'], np.int32),
                      audit_anchor_time=np.array(aud['anchor_time'], np.int16), audit_anchor_id=np.array(aud['anchor_id'], np.int32), audit_draw=np.array(aud['draw'], np.int8),
                      audit_success=np.array(aud['success']), audit_failure=np.array(aud['failure']), audit_u1=np.array(aud['u1']), audit_u2=np.array(aud['u2']),
                      audit_p_goal=np.array(aud['p_goal'], np.float32), audit_max_y=np.array(aud['max_y'], np.float32))
  kind_a = np.array(aud['kind']); cand_a = np.array(aud['cand']); tot = w.sum()
  summary = {'paths': int(n_tot), 'new_paths': len(jobs), 'wall_seconds': wall, 'replay_sha256': sha256(path), 'bytes': int(os.path.getsize(path)),
             'effective_weights': {'state_mass_dense': float(w[kind_a != 'general'].sum() / tot), 'state_mass_general': float(w[kind_a == 'general'].sum() / tot),
                                   'mixture_recorded': float(w[cand_a == 'recorded'].sum() / tot), 'mixture_policy': float(w[cand_a != 'recorded'].sum() / tot),
                                   'within_dense_recorded': float(w[(kind_a != 'general') & (cand_a == 'recorded')].sum() / w[kind_a != 'general'].sum()),
                                   **{f'mass_{s}': float(w[kind_a == s].sum() / tot) for s in R.DENSE_SETS}},
             'new_paths_reach': float(np.mean(aud['success'][len(keep):])), 'new_paths_death': float(np.mean(aud['failure'][len(keep):])), 'new_paths_p_goal': float(np.mean(aud['p_goal'][len(keep):]))}
  if c_rows:
    R.save_table(B.OUT / DIAG_NEW / 'outcomes.npz', c_rows, {'wall_seconds': wall, 'paths': len(c_rows), 'continuation_ckpt': str(args.cont_ckpt)})
    summary['cnew_paths'] = len(c_rows)
  # the R replay's own stratum masses for the audit
  with np.load(r_replay, allow_pickle=False) as d:
    wr = d['audit_weight']; kr = d['audit_kind'].astype(str)
    summary['narrow_effective_weights'] = {f'mass_{s}': float(wr[kr == s].sum() / wr.sum()) for s in R.DENSE_SETS}
    summary['narrow_effective_weights'].update({'state_mass_dense': float(wr[kr != 'general'].sum() / wr.sum())})
  (out / 'build_broad.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  print(json.dumps(summary, indent=1), flush=True)


# ----------------------------------------------------------------- analyze
def analyze(args):
  """Cnew (primary) and the development A/B/C for both arms; paired bootstraps."""
  out = exp_dir()
  for arm, cdir, marg in (('narrow', 'critics_narrow', 'exp_r1_coverage/replay_armR.npz'), ('broad', 'critics_broad', f'{EXP}/replay_broad.npz')):
    critics = [str(B.OUT / cdir / f'seed_{s}' / 'final.pkl') for s in (0, 1, 2)]
    for diag, tag in ((DIAG_NEW, arm), ('diag_r1_abc', f'ep_{arm}')):
      cmd = [sys.executable, str(ROOT / 'scripts' / 'diag_v6_r1_abc.py'), 'analyze', '--cont-ckpt', str(args.cont_ckpt), '--critics', *critics, '--allow-unsealed-critics',
             '--marginal-replay', marg, '--tag', tag, '--out', diag]
      print(' '.join(cmd), flush=True); subprocess.run(cmd, check=True, cwd=str(ROOT))
  for L_ in ('C',):
    cmd = [sys.executable, str(ROOT / 'scripts' / 'probe_v6_coverage_paired_boot.py'), '--cont-ckpt', str(args.cont_ckpt), '--runs', 'NARROW=critics_narrow', 'BROAD=critics_broad',
           '--marginals', 'NARROW=exp_r1_coverage/replay_armR.npz', f'BROAD={EXP}/replay_broad.npz', '--layer', L_, '--diag-dir', DIAG_NEW, '--out', f'{DIAG_NEW}/paired_boot_{L_}.json']
    print(' '.join(cmd), flush=True); subprocess.run(cmd, check=True, cwd=str(ROOT), env={**os.environ, 'JAX_PLATFORMS': 'cpu'})


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('merge', 'seal', 'build', 'analyze'))
  ap.add_argument('--cont-ckpt', default='')
  ap.add_argument('--seed', type=int, default=2027)
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'merge': merge, 'seal': seal, 'build': build, 'analyze': analyze}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
