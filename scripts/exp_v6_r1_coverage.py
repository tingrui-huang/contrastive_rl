"""AntMaze V6, after the A/B/C diagnostic: is within-state TORQUE coverage the
bottleneck at B?  An equal-budget, controlled comparison of two replays
built on the round-1 anchors, each critic then read with the SAME A/B/C
diagnostic (`scripts/diag_v6_r1_abc.py`, its sealed outcomes).

Arms (per dense round-1 anchor; the general anchors and every recorded-torque
path are the round-1 paths verbatim):

  R    repeated outcomes: the same two policy torques the round-1 replay
       carries (sample0, sample1), eight fresh continuation outcomes each
       -> 16 policy branches per state, 2 distinct torques;
  COV  action coverage: eight distinct torques sampled independently from
       the same frozen policy at that state (a training-torque RNG stream
       disjoint from every validation stream), two fresh continuation
       outcomes each -> 16 policy branches per state, 8 distinct torques.

Every branch executes exactly ONE queried 8-d torque and then the frozen
round-1 continuation policy (its mode) closed-loop; hazards, clocks and
jitter are redrawn per draw and paired across the candidates of a state
(as in round 1); horizon, hold, law unchanged.

Mandatory controls (all exact, none approximated):
  * state weights: every path carries an anchor weight -- round-1 paths 1,
    new policy paths 4/16 = 0.25 -- so a dense anchor's total mass (6 units)
    and the general anchors' mass are exactly round 1's; the driver samples
    anchors (and hence the in-batch NCE negatives) with these weights
    (V6_ANCHOR_WEIGHTS=audit -> TrajectoryBuffer.set_anchor_strata weights);
  * recorded-vs-policy mixture: dense recorded 2 x 1 = 2 units against
    policy 16 x 0.25 = 4 units, i.e. 1/3 : 2/3 as in round 1;
  * validation torques stay unseen: training torques come from
    TRAIN_TORQUE_SEED, training draws from TRAIN_DRAW_SEED; the B torques
    (ABC_SEED + 17) and the fresh untouched torques (FRESH_TORQUE_SEED) are
    other streams (seed bases are checked to differ by non-multiples of
    7919, the per-anchor stride);
  * equal budget, fixed protocol: the same anchors, environment, p_active,
    continuation, law, horizon, NCE, architecture, gamma, optimiser, steps
    (30k), batch (1024), critic seeds (0, 1, 2) for both arms; no per-arm
    tuning, no extension.

Evaluation: the COV critics' "familiar torques" at the A anchors are its
own training torques (cov0, cov1), so those two are also run at the A
anchors with the diagnostic's draw seeds (common random numbers with the
existing A / B outcomes) and merged; R's familiar torques are the existing
A layer.  B and C are the diagnostic's sealed outcomes, unchanged.

No actor is trained.  Oracle-stage diagnostic.

  python scripts/exp_v6_r1_coverage.py seal    --cont-ckpt <pure-BC>
  python scripts/exp_v6_r1_coverage.py build   --cont-ckpt <pure-BC> --arm R --workers 12
  python scripts/exp_v6_r1_coverage.py build   --cont-ckpt <pure-BC> --arm COV --workers 12
  (critics: run_v6_branch_replay.py critics with V6_BRANCH_REPLAY=<arm replay> V6_CRITIC_TAG=_armR|_armCOV V6_ANCHOR_WEIGHTS=audit)
  python scripts/exp_v6_r1_coverage.py analyze --cont-ckpt <pure-BC> --arm R|COV
  python scripts/exp_v6_r1_coverage.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
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

TRAIN_TORQUE_SEED = 515_000_003
TRAIN_DRAW_SEED = 515_500_000
FRESH_TORQUE_SEED = 616_000_005
FRESH_DRAW_SEED = 616_500_000
STRIDE = 7919                        # candidates_for: rng = default_rng(base + 7919 * anchor_id)
N_BRANCHES = 16
ARMS = {'R': {'n_torques': 2, 'n_draws': 8, 'torques': 'round-1 sample0, sample1 (verbatim)'},
        'COV': {'n_torques': 8, 'n_draws': 2, 'torques': 'eight new samples from the frozen policy (TRAIN_TORQUE_SEED)'}}
NEW_WEIGHT = 4.0 / N_BRANCHES        # the four round-1 policy paths' mass spread over sixteen
EXP_DIR = 'exp_r1_coverage'
ABC_DIR = 'diag_r1_abc'


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def git_info():
  try:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(ROOT), text=True).splitlines()
    return {'head': head, 'dirty': [l for l in status if not l.startswith('??')], 'untracked': [l for l in status if l.startswith('??')]}
  except Exception as e:  # noqa
    return {'error': str(e)}


def exp_dir():
  return B.OUT / EXP_DIR


def load_r1(replay):
  """The round-1 replay's audit + row-0 keys, and the dense anchors' recorded / sample torques."""
  with np.load(replay, allow_pickle=False) as d:
    kind = d['audit_kind'].astype(str); cand = d['audit_cand'].astype(str); aid = d['audit_anchor_id'].astype(int)
    draw = d['audit_draw'].astype(int); ep = d['audit_episode'].astype(int); at = d['audit_anchor_time'].astype(int)
    a0 = d['act'][:, 0]; o0 = d['obs'][:, 0]; pg = d['audit_p_goal']
  dense = {}
  for i in range(len(aid)):
    if kind[i] == 'general':
      continue
    e = dense.setdefault(aid[i], {'episode': ep[i], 't': at[i], 'stratum': kind[i], 'obs0': o0[i], 'torques': {}, 'paths': []})
    e['torques'][cand[i]] = a0[i].astype(np.float32)
    e['paths'].append(i)
  return {'kind': kind, 'cand': cand, 'aid': aid, 'draw': draw, 'episode': ep, 't': at, 'p_goal': pg, 'n': len(aid)}, dense


# -------------------------------------------------------------------- seal
def seal(args):
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed); --force to reseal before any build')
  replay = B.OUT / args.replay
  abc = B.OUT / ABC_DIR
  abc_man = json.loads((abc / 'manifest.json').read_text(encoding='utf-8'))
  audit, dense = load_r1(replay)
  # seed-stream separation: bases must not differ by a multiple of the per-anchor stride
  bases = {'r1_training_torques': R.R1_SEED + 11, 'abc_B_validation_torques': ABC.ABC_SEED + 17,
           'train_torques_COV': TRAIN_TORQUE_SEED, 'fresh_untouched_torques': FRESH_TORQUE_SEED}
  sep = {}
  for a in bases:
    for b in bases:
      if a < b:
        sep[f'{a} vs {b}'] = int((bases[a] - bases[b]) % STRIDE)
  assert all(v != 0 for v in sep.values()), sep
  draw_ranges = {'r1_training_draws': [R.R1_SEED, R.R1_SEED + 1000 * 6000], 'r1_holdout_draws': [R.R1_SEED + 300_000 + 1000 * R.HOLDOUT_ID0, R.R1_SEED + 300_000 + 1000 * (R.HOLDOUT_ID0 + 400)],
                 'abc_draws': [ABC.ABC_SEED, ABC.ABC_SEED + 100 * 400], 'train_draws_arms': [TRAIN_DRAW_SEED, TRAIN_DRAW_SEED + 1000 * 1500],
                 'fresh_untouched_draws': [FRESH_DRAW_SEED, FRESH_DRAW_SEED + 100 * 400]}
  rs = sorted(draw_ranges.values())
  assert all(rs[i][1] <= rs[i + 1][0] for i in range(len(rs) - 1)), draw_ranges
  # the policy at every dense anchor (for the COV torques) -- sampled now, sealed, never re-drawn
  bundle = R.policy_bundle(args.cont_ckpt)
  aids = [int(a) for a in sorted(dense)]
  o0 = np.stack([dense[a]['obs0'] for a in aids])
  loc, scale = bundle['params_fn'](o0)
  cov_torques = {}
  for i, a in enumerate(aids):
    cov_torques[a] = {f'cov{k}': v.astype(float).tolist() for k, (_, v) in enumerate(
        R.candidates_for(loc[i], scale[i], dense[a]['torques']['recorded'], a, [f'sample{k}' for k in range(8)], TRAIN_TORQUE_SEED))}
  # check: no COV torque coincides with a B validation torque or an r1 torque at the same anchor
  abc_info = {x['anchor_id']: x for x in abc_man['anchors'] if x['layer'] == 'AB'}
  min_dist_B, min_dist_r1 = np.inf, np.inf
  for a in aids:
    T = np.array(list(cov_torques[a].values()))
    for c, v in dense[a]['torques'].items():
      min_dist_r1 = min(min_dist_r1, float(np.linalg.norm(T - np.asarray(v)[None], axis=1).min()))
    if a in abc_info:
      for c, v in abc_info[a]['candidates'].items():
        if c.startswith('B|'):
          min_dist_B = min(min_dist_B, float(np.linalg.norm(T - np.asarray(v)[None], axis=1).min()))
  # effective weights (identical for both arms by construction)
  w_r1 = np.ones(audit['n']); n_dense = len(aids)
  mass_general = float((audit['kind'] == 'general').sum()); mass_dense_rec = float(((audit['kind'] != 'general') & (audit['cand'] == 'recorded')).sum())
  mass_dense_pol = n_dense * N_BRANCHES * NEW_WEIGHT; tot = mass_general + mass_dense_rec + mass_dense_pol
  weights = {'round1_path_weight': 1.0, 'new_policy_path_weight': NEW_WEIGHT, 'paths_round1_verbatim': int(mass_general + mass_dense_rec),
             'new_policy_paths_per_arm': n_dense * N_BRANCHES, 'total_mass': tot,
             'state_mass': {'dense_anchors': (mass_dense_rec + mass_dense_pol) / tot, 'general_anchors': mass_general / tot, 'per_dense_anchor': 6.0 / tot, 'per_general_anchor': 2.0 / tot},
             'mixture': {'recorded': (mass_dense_rec + mass_general / 2) / tot, 'policy': (mass_dense_pol + mass_general / 2) / tot,
                         'within_dense_recorded': mass_dense_rec / (mass_dense_rec + mass_dense_pol), 'within_dense_policy': mass_dense_pol / (mass_dense_rec + mass_dense_pol)},
             'round1_reference': {'state_mass_dense': 9000 / 17756, 'mixture_recorded': (3000 + 4378) / 17756, 'within_dense_recorded': 1 / 3}}
  # fixed training protocol (from the driver's config, the round-1 recipe)
  import run_v6_branch_replay as D
  cfg = D.base_config(0, replay, 30_000, out / '_cfg')
  hp = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(cfg).items()
        if k in ('batch_size', 'learning_rate', 'actor_learning_rate', 'discount', 'repr_dim', 'repr_norm', 'repr_norm_temp', 'hidden_layer_sizes',
                 'twin_q', 'use_layer_norm', 'bc_coef', 'bc_sampling', 'max_number_of_steps', 'num_sgd_steps_per_step', 'env_name',
                 'rockfall_p_active_1', 'rockfall_p_active_2', 'max_episode_steps', 'eval_goal_mode')}
  hp.update({'critic_steps': 30_000, 'milestones': [10_000, 20_000], 'critic_seeds': [0, 1, 2], 'anchor_rule': 'row 0 of every path, weighted (V6_ANCHOR_WEIGHTS=audit)',
             'driver': 'scripts/run_v6_branch_replay.py critics (train_critic), unchanged apart from the weighted prepare'})
  man = {'experiment': 'within-state torque coverage vs repeated outcomes, equal budget, round-1 anchors', 'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'git': git_info(),
         'provenance': {'r1_replay': str(replay), 'r1_replay_sha256': sha256(replay), 'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': sha256(args.cont_ckpt),
                        'dataset': str(B.DATASET), 'dataset_sha256': sha256(B.DATASET), 'abc_manifest_sha256': sha256(abc / 'manifest.json'),
                        'abc_outcomes_sha256': sha256(abc / 'outcomes.npz'), 'abc_reference_commit': abc_man['provenance']['reference_commit'],
                        'r1_critics': abc_man['provenance']['critics']},
         'anchors': {'dense_anchor_ids': aids, 'n_dense': n_dense, 'strata': {s: int(sum(dense[a]['stratum'] == s for a in aids)) for s in R.DENSE_SETS},
                     'general_paths_verbatim': int(mass_general), 'dense_recorded_paths_verbatim': int(mass_dense_rec),
                     'abc_A_anchor_ids': sorted(abc_info), 'abc_C_anchor_ids': sorted(x['anchor_id'] for x in abc_man['anchors'] if x['layer'] == 'C')},
         'arms': {arm: {**spec, 'branches_per_state': N_BRANCHES, 'new_paths': n_dense * N_BRANCHES} for arm, spec in ARMS.items()},
         'seeds': {'train_torque_seed_base': TRAIN_TORQUE_SEED, 'train_draw_seed_rule': 'TRAIN_DRAW_SEED + 1000 * anchor_id + draw (shared across the candidates of an anchor; R uses draws 0-7, COV draws 0-1)',
                   'train_draw_seed_base': TRAIN_DRAW_SEED, 'abc_B_torque_seed_base': ABC.ABC_SEED + 17, 'abc_draw_seed_base': ABC.ABC_SEED,
                   'fresh_torque_seed_base': FRESH_TORQUE_SEED, 'fresh_draw_seed_base': FRESH_DRAW_SEED, 'stream_separation_mod_stride': sep, 'draw_seed_ranges': draw_ranges,
                   'cov_torque_min_dist_to_B_validation': min_dist_B, 'cov_torque_min_dist_to_r1_torques': min_dist_r1},
         'weights': weights, 'training': hp,
         'evaluation': {'diagnostic': 'scripts/diag_v6_r1_abc.py analyze on the sealed diag_r1_abc outcomes (B, C unchanged); layer A = recorded + the arm\'s two familiar torques '
                                      '(R: sample0/sample1 from the existing A outcomes; COV: cov0/cov1 run at the A anchors with the diagnostic\'s draw seeds)',
                        'primary_readout': 'region-integrated exp(f), radius 0.5, NCE goal marginal of the arm\'s own weighted replay, deployed min over heads',
                        'secondary': 'exact-goal logit, per head', 'uncertainty': 'episode bootstrap', 'metrics': 'agreement among decided pairs, decided/tie/weak counts, critic-selected P_goal gain, cross-fit selector, stratum tables, A x B cross pairs',
                        'decision_rules': {'case1': 'COV improves B over r1 and R with positive held-out gain -> action coverage is a bottleneck; then fresh untouched torques, no actor',
                                           'case2': 'COV improves B, C still chance -> action and state generalisation separate', 'case3': 'R improves, COV not -> outcome variance',
                                           'case4': 'neither improves B -> stop enlarging the replay; the procedure does not use within-state coverage'},
                        'B_improvement_line': 'pooled dense B agreement above r1 and R by > 2 episode-bootstrap s.e. and above 0.5 by 2 s.e., with positive pick gain (> 2 s.e.)'},
         'cov_torques': {str(a): v for a, v in cov_torques.items()}}
  (out / 'manifest.json').write_text(json.dumps(man, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
  print(json.dumps({k: man[k] for k in ('weights', 'seeds')}, indent=1, default=float), flush=True)
  print(f'sealed: {n_dense} dense anchors, strata {man["anchors"]["strata"]}', flush=True)


# ------------------------------------------------------------------- build
def build(args):
  out = exp_dir()
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  assert sha256(args.cont_ckpt) == man['provenance']['continuation_sha256']
  replay = B.OUT / args.replay
  assert sha256(replay) == man['provenance']['r1_replay_sha256']
  arm = args.arm; spec = ARMS[arm]
  path = out / f'replay_arm{arm}.npz'
  if path.exists() and not args.force:
    raise SystemExit(f'{path} exists')
  audit, dense = load_r1(replay)
  aids = man['anchors']['dense_anchor_ids']
  # jobs: (anchor_id, episode, t, stratum, cand, action, draw, pair_seed, keep_path)
  jobs = []
  for a in aids:
    e = dense[a]
    if arm == 'R':
      torques = {c: e['torques'][c] for c in ('sample0', 'sample1')}
    else:
      torques = {c: np.asarray(v, np.float32) for c, v in man['cov_torques'][str(a)].items()}
    for c, v in torques.items():
      for r in range(spec['n_draws']):
        jobs.append((a, int(e['episode']), int(e['t']), e['stratum'], c, np.asarray(v, np.float32), r, TRAIN_DRAW_SEED + 1000 * a + r, True))
  n_new = len(jobs)
  assert n_new == len(aids) * N_BRANCHES
  # COV's familiar torques at the diagnostic's A anchors, with the diagnostic's draw seeds (no paths kept)
  jobs_a = []
  if arm == 'COV':
    abc_man = json.loads((B.OUT / ABC_DIR / 'manifest.json').read_text(encoding='utf-8'))
    for x in abc_man['anchors']:
      if x['layer'] != 'AB':
        continue
      for c in ('cov0', 'cov1'):
        v = np.asarray(man['cov_torques'][str(x['anchor_id'])][c], np.float32)
        for r, seed in enumerate(x['draw_seeds']):
          jobs_a.append((x['anchor_id'], x['episode'], x['t'], x['stratum'], f'A|{c}', v, r, seed, False))
  if args.limit:
    jobs = jobs[:args.limit]; jobs_a = jobs_a[:args.limit]
  # the arm replay: round-1 general + dense recorded paths verbatim, then the new policy paths
  with np.load(replay, allow_pickle=False) as d:
    keep = np.flatnonzero((audit['kind'] == 'general') | (audit['cand'] == 'recorded'))
    L = d['obs'].shape[1]; meta = json.loads(str(d['meta']))
    n_tot = len(keep) + len(jobs)
    obs_p = np.zeros((n_tot, L, B.STATE_DIM + 2), np.float32); act_p = np.zeros((n_tot, L, B.ACTION_DIM), np.float32)
    lengths_p = np.zeros(n_tot, np.int64)
    obs_p[:len(keep)] = d['obs'][keep]; act_p[:len(keep)] = d['act'][keep]; lengths_p[:len(keep)] = d['lengths'][keep]
    aud = {k: list(d[f'audit_{k}'][keep]) for k in ('kind', 'cand', 'episode', 'anchor_time', 'anchor_id', 'draw', 'success', 'failure', 'u1', 'u2', 'p_goal', 'max_y')}
  weight = [1.0] * len(keep)
  print(f'build arm {arm}: {len(keep)} round-1 paths verbatim + {len(jobs)} new policy paths ({spec["n_torques"]} torques x {spec["n_draws"]} draws x {len(aids)} anchors)'
        + (f' + {len(jobs_a)} A-layer paths' if jobs_a else '') + f'; {args.workers} workers', flush=True)
  t0 = time.time()
  parts = ABC.R.DG.chunks(jobs + jobs_a, args.workers * 6)
  arglist = [(p, TRAIN_DRAW_SEED + 7 + i, args.cont_ckpt) for i, p in enumerate(parts)]
  k = len(keep); a_rows = []
  with get_context('spawn').Pool(args.workers) as pool:
    for res in pool.imap_unordered(R._run_jobs, arglist):
      for r in res:
        if 'obs' in r:
          n = len(r['obs'])
          obs_p[k, :n] = r['obs']; act_p[k, :n] = r['act']; obs_p[k, n:] = r['obs'][-1]; lengths_p[k] = n
          for key, val in (('kind', r['set']), ('cand', r['cand']), ('episode', r['episode']), ('anchor_time', r['t']), ('anchor_id', r['anchor_id']),
                           ('draw', r['draw']), ('success', r['success']), ('failure', r['failure']), ('u1', r['u1']), ('u2', r['u2']), ('p_goal', r['p_goal']), ('max_y', r['max_y'])):
            aud[key].append(val)
          weight.append(NEW_WEIGHT); k += 1
        else:
          a_rows.append(r)
  wall = time.time() - t0
  assert k == n_tot, (k, n_tot)
  print(f'rollouts done in {wall:.0f}s', flush=True)
  m = dict(meta); m.update({'arm': f'coverage_arm_{arm}', 'source_replay': str(replay), 'branches_per_state': N_BRANCHES, **spec,
                            'anchor_weights': 'audit_weight (round-1 paths 1.0, new policy paths 0.25)', 'train_draw_seed_base': TRAIN_DRAW_SEED,
                            'train_torque_seed_base': (TRAIN_TORQUE_SEED if arm == 'COV' else None)})
  w = np.array(weight)
  np.savez_compressed(path, obs=obs_p, act=act_p, lengths=lengths_p, eval_goals=obs_p[:, 0, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float32),
                      meta=np.asarray(json.dumps(m, sort_keys=True)), audit_weight=w,
                      audit_kind=np.array(aud['kind']), audit_cand=np.array(aud['cand']), audit_episode=np.array(aud['episode'], np.int32),
                      audit_anchor_time=np.array(aud['anchor_time'], np.int16), audit_anchor_id=np.array(aud['anchor_id'], np.int32),
                      audit_draw=np.array(aud['draw'], np.int8), audit_success=np.array(aud['success']), audit_failure=np.array(aud['failure']),
                      audit_u1=np.array(aud['u1']), audit_u2=np.array(aud['u2']), audit_p_goal=np.array(aud['p_goal'], np.float32), audit_max_y=np.array(aud['max_y'], np.float32))
  kind = np.array(aud['kind']); cand = np.array(aud['cand']); tot = w.sum()
  summary = {'arm': arm, 'paths': n_tot, 'new_paths': len(jobs), 'wall_seconds': wall, 'bytes': int(os.path.getsize(path)), 'replay_sha256': sha256(path),
             'effective_weights': {'state_mass_dense': float(w[kind != 'general'].sum() / tot), 'state_mass_general': float(w[kind == 'general'].sum() / tot),
                                   'mixture_recorded': float(w[cand == 'recorded'].sum() / tot), 'mixture_policy': float(w[cand != 'recorded'].sum() / tot),
                                   'within_dense_recorded': float(w[(kind != 'general') & (cand == 'recorded')].sum() / w[kind != 'general'].sum())},
             'new_paths_reach': float(np.mean(aud['success'][len(keep):])), 'new_paths_death': float(np.mean(aud['failure'][len(keep):])),
             'new_paths_p_goal': float(np.mean(aud['p_goal'][len(keep):]))}
  if a_rows:
    R.save_table(B.OUT / ABC_DIR / f'outcomes_arm{arm}_A.npz', a_rows, {'arm': arm, 'cands': ['A|cov0', 'A|cov1'], 'draw_seeds': 'the diag_r1_abc manifest seeds'})
    summary['A_layer_paths'] = len(a_rows)
  (out / f'build_arm{arm}.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  print(json.dumps(summary, indent=1), flush=True)


# ----------------------------------------------------------------- analyze
def analyze(args):
  out = exp_dir(); arm = args.arm
  critics = [str(B.OUT / f'critics_arm{arm}' / f'seed_{s}' / 'final.pkl') for s in (0, 1, 2)]
  cmd = [sys.executable, str(ROOT / 'scripts' / 'diag_v6_r1_abc.py'), 'analyze', '--cont-ckpt', str(args.cont_ckpt), '--critics', *critics,
         '--allow-unsealed-critics', '--marginal-replay', f'{EXP_DIR}/replay_arm{arm}.npz', '--tag', f'arm{arm}']
  if arm == 'COV':
    cmd += ['--a-cands', 'recorded,cov0,cov1', '--extra-outcomes', f'{ABC_DIR}/outcomes_armCOV_A.npz']
  print(' '.join(cmd), flush=True)
  subprocess.run(cmd, check=True, cwd=str(ROOT))


# ------------------------------------------------------------------ report
def _m(res, layer, scorer, stratum='pooled'):
  return res['layers'][layer][scorer][stratum]


def report(args):
  out = exp_dir(); abc = B.OUT / ABC_DIR
  runs = {'r1': json.loads((abc / 'metrics.json').read_text(encoding='utf-8'))}
  for arm in ARMS:
    p = abc / f'metrics_arm{arm}.json'
    if p.exists():
      runs[arm] = json.loads(p.read_text(encoding='utf-8'))
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  f_ = ABC._f
  L = ['# Torque coverage vs repeated outcomes: r1 vs R vs COV on the A / B / C diagnostic', '',
       f'Manifest `{out / "manifest.json"}` (sealed {man["sealed_at"]}, git {man["git"].get("head", "?")[:10]}, dirty {len(man["git"].get("dirty", []))} files).  '
       f'Both arms: 1,500 round-1 dense anchors, 16 policy branches per state, round-1 general and recorded paths verbatim, anchor weights 1 / 0.25, '
       f'30k NCE, seeds 0 / 1 / 2, one query torque then the frozen continuation.  Readout = region-integrated exp(f), radius 0.5, the arm\'s own weighted '
       'goal marginal, deployed min over heads; agreement among decided pairs (|log ratio| > 0.3 on 16 fresh draws), s.e. = episode bootstrap; pick gain = '
       'P_goal of the critic\'s argmax candidate minus the candidate mean.  A = recorded + the arm\'s two familiar torques (r1 / R: sample0, sample1; COV: cov0, cov1); '
       'B and C = the sealed diagnostic outcomes, unchanged.  No actor trained.', '',
       '## Effective anchor weights and mixture (from the built replays)', '',
       '| arm | paths | state mass dense / general | mixture recorded / policy | within dense recorded | round-1 reference |', '|---|---:|---|---|---:|---|']
  for arm in ARMS:
    p = out / f'build_arm{arm}.json'
    if p.exists():
      b = json.loads(p.read_text(encoding='utf-8')); w = b['effective_weights']
      L.append(f'| {arm} | {b["paths"]} | {w["state_mass_dense"]:.4f} / {w["state_mass_general"]:.4f} | {w["mixture_recorded"]:.4f} / {w["mixture_policy"]:.4f} | '
               f'{w["within_dense_recorded"]:.4f} | {9000 / 17756:.4f} / {8756 / 17756:.4f}; {(3000 + 4378) / 17756:.4f} / {(6000 + 4378) / 17756:.4f}; {1 / 3:.4f} |')
  L += ['', '## Pooled dense strata (agreement (s.e.) same / opposite; decided pairs; pick gain (s.e.))', '',
        '| run | seed | A | B | C | A x B cross | C matched |', '|---|---|---|---|---|---|---|']
  for run, res in runs.items():
    scorers = [s for s in res['layers']['A'] if s.endswith('region min')]
    for sc in scorers:
      cells = []
      for layer in ('A', 'B', 'C', 'AB_cross', 'C_matched'):
        m = _m(res, layer, sc)
        cells.append(f'{f_(m["agreement_decided"])} ({f_(m["se_decided"], ".3f")}) {m["same"]}/{m["opposite"]}; n {m["n_decided"]}; gain {f_(m["pick_gain"], "+.3f")} ({f_(m["pick_gain_se"], ".3f")})')
      L.append(f'| {run} | {sc.split(" ")[0].split("/")[-1]} | ' + ' | '.join(cells) + ' |')
  L += ['', '## Layer B by stratum (agreement (s.e.), decided pairs, pick gain)', '',
        '| run | seed | start_early | start_late | shortcut_early | turn | north_leg |', '|---|---|---|---|---|---|---|']
  for run, res in runs.items():
    for sc in [s for s in res['layers']['B'] if s.endswith('region min')]:
      cells = []
      for st in R.DENSE_SETS:
        m = _m(res, 'B', sc, st)
        cells.append(f'{f_(m["agreement_decided"])} ({f_(m["se_decided"], ".3f")}) n {m["n_decided"]} g {f_(m["pick_gain"], "+.3f")}')
      L.append(f'| {run} | {sc.split(" ")[0].split("/")[-1]} | ' + ' | '.join(cells) + ' |')
  L += ['', '## Secondary readouts, pooled B (agreement / pick gain)', '', '| run | seed | region min | region h0 | region h1 | exact min | exact h0 | exact h1 |', '|---|---|---|---|---|---|---|---|']
  for run, res in runs.items():
    seeds = sorted({s.split(' ')[0] for s in res['layers']['B'] if s != 'random'})
    for sd in seeds:
      cells = []
      for ro in ('region min', 'region h0', 'region h1', 'exact min', 'exact h0', 'exact h1'):
        m = _m(res, 'B', f'{sd} {ro}')
        cells.append(f'{f_(m["agreement_decided"])} / {f_(m["pick_gain"], "+.3f")}')
      L.append(f'| {run} | {sd.split("/")[-1]} | ' + ' | '.join(cells) + ' |')
  L += ['', '## Reference: cross-fitted empirical selector gain (outcomes only, scorer-independent)', '']
  for run, res in runs.items():
    L.append(f'- {run}: A {f_(_m(res, "A", "random")["cross_fit_selector_gain"], "+.3f")}, B {f_(_m(res, "B", "random")["cross_fit_selector_gain"], "+.3f")}, C {f_(_m(res, "C", "random")["cross_fit_selector_gain"], "+.3f")}')
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'build', 'analyze', 'report'))
  ap.add_argument('--cont-ckpt', default='')
  ap.add_argument('--replay', default='replay_policy_r1.npz')
  ap.add_argument('--arm', choices=tuple(ARMS), default='R')
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': seal, 'build': build, 'analyze': analyze, 'report': report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
