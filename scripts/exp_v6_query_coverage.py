"""AntMaze V6 mainline -- query coverage, stage 1: a GENERATION comparison of
logged-query vs extended-query futures at the training start / early-decision
anchors (user's plan, 2026-09-20, after the review of the real-update check).

Why.  The mainline generates ONE future per anchor whose first torque is the
LOGGED one (`action = self.a.action[k]` in the critic stream; the policy then
continues closed-loop).  Along that path every later step has an action, so
nothing is missing from the rollout -- but "several different actions at the
same state, each with its own future" and "those (s, a_query) pairs entering
the NCE" do not occur: the critic receives no supervision at the policy's
alternative actions.  PointMaze's successful branch replay additionally kept
~7,700 extra query branches; the AntMaze mainline deliberately dropped them
to isolate the future source.  That is a real coverage difference between the
two pipelines, NOT yet shown to be the cause of the performance gap -- this
stage measures what the extension would add BEFORE any training.

Design (user's specification, fixed before generation).
  * States: every anchor of the pilot anchor set (`anchors.npz`, the critic
    anchor law) whose position is in the start region -- the starts and the
    early decision states (t median 10; 99 % have t <= 30).  No state is
    chosen from evaluation results.
  * Current policy = the lineage agent of seed s (the clipped CF final,
    `variants/critic_clip0.1/CF/seed_s/final.pkl`) -- the policy whose stable
    shortcut choice is at issue.  Queries per anchor: q0 the logged torque,
    q1 the agent's mode tanh(loc), q2..q5 K = 4 samples tanh(loc + scale *
    eps) with eps pinned by (lineage, anchor, j).  Candidates are NOT chosen
    from evaluation results.
  * Every query is executed ONCE from the restored logged state at absolute
    time t and continued by the SAME agent (mode, closed-loop) to the first
    reach frame / death / the absolute horizon -- the mainline branch
    interface (`exp_v6_mainline_pilot._branch_one`).  Same goal (the logged
    task goal).  PAIRED hazard draws: two draws per anchor, each identical
    across the six queries (draw 0 uses the mainline's seed HAZARD_SEED0 +
    anchor_id, so q0 / draw 0 is the clip-round-A regeneration at these
    anchors; draw 1 uses a fresh offset).
  * Nothing is filtered: every outcome is kept.

Read-out (anchor-weighted; the extended set spreads the anchor's weight
equally over its six queries): far-route ENTRY and COMPLETION (entered y >= 6
at x < 2 after the query AND the branch reached the goal), success / death /
timeout, branch length, and the critic goal marginal's mass (truncated
geometric law, gamma 0.999) in the far legs / goal area / hazard zones --
logged-only (q0) vs the extended union (q0..q5) vs the added queries (q1..q5),
per lineage, paired-by-anchor bootstrap; draw-to-draw agreement as the
reliability check; strata (reset rows, t in [1, 30), logged shortcut /
detour episodes).

Gate for stage 2 (the training comparison; pre-registered here, before any
generation) -- "a reliable contrast" means, in EVERY lineage:
  G1  far-route COMPLETION share, extended union - logged >= +0.03 (absolute)
      with the paired bootstrap 95 % CI lower bound > 0;
  G2  the goal-area mass of the critic goal marginal under the extended union
      >= 0.8 x its value under the logged queries (the task-goal positives are
      not destroyed);
  G3  at least 100 distinct start anchors have >= 1 far-route-complete future
      among the ADDED queries in BOTH hazard draws (the gain is not carried by
      a handful of anchors or one draw).
Stage 2 (only if G1-G3 pass, and on the user's go): the new (s, a_query)
pairs become critic anchors -- per-state total weight, training budget,
actor / BC data, the CRL losses, critic clip 0.1 and BC 0.05 unchanged -- and
the comparison is logged-query vs extended-query futures under the SAME
continuation policy in both arms.  Oracle (simulator) futures: an engineering
stage, not offline identification.

Modes:
  python scripts/exp_v6_query_coverage.py seal
  python scripts/exp_v6_query_coverage.py generate --lineage s --workers 30 [--limit n]
  python scripts/exp_v6_query_coverage.py report
Outputs under outputs/antmaze_branch_replay_p050/exp_mainline_pilot/query_coverage/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_clip_round as CR  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'query_coverage'
K_SAMPLES = 4
N_QUERIES = 2 + K_SAMPLES                  # q0 logged, q1 mode, q2..q5 samples
QUERY_NAMES = ['logged', 'mode'] + [f'sample{j}' for j in range(K_SAMPLES)]
DRAWS = 2
HAZARD_SEED1 = 142_000_000                 # draw 1: HAZARD_SEED1 + anchor_id (draw 0 = MP.HAZARD_SEED0 + anchor_id, the mainline pairing)
QUERY_SEED0 = 143_000_000                  # eps for the samples: default_rng(QUERY_SEED0 + lineage)
WORKER_SEED0 = 144_000_000
BOOT = 2000
BOOT_SEED = 145_000_001
GATES = {'G1_completed_far_gain': {'quantity': 'far-route completion share, extended union (q0..q5, equal weights) minus logged (q0), anchor-weighted over the start-region anchors, both draws',
                                   'rule': '>= +0.03 absolute AND paired-by-anchor bootstrap 95 % CI lower bound > 0, in every lineage'},
         'G2_goal_mass_kept': {'quantity': 'goal-area mass of the critic goal marginal (truncated geometric, gamma 0.999), extended union vs logged', 'rule': 'extended >= 0.8 x logged, in every lineage'},
         'G3_spread': {'quantity': 'distinct start anchors with >= 1 far-route-complete future among the ADDED queries (q1..q5)', 'rule': '>= 100 in draw 0 AND >= 100 in draw 1, in every lineage'}}
SETS = {'logged': [0], 'mode': [1], 'samples': list(range(2, N_QUERIES)), 'added': list(range(1, N_QUERIES)), 'extended': list(range(N_QUERIES))}
QUANT = ('entered_far', 'completed_far', 'entered_zone', 'success', 'death', 'timeout', 'length', 'mass_far', 'mass_goal', 'mass_zone')


def start_anchor_ids(anchors):
  reg = region_of(anchors.state[:, :2])
  return np.flatnonzero(reg == REGIONS.index('start'))


def branch_path(s):
  return OUT / f'queries_s{s}.npz'


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  ids = start_anchor_ids(anchors)
  man = {'experiment': 'AntMaze V6 mainline, query coverage stage 1: logged-query vs extended-query futures at the start / early-decision anchors (generation only; no training)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'),
         'states': {'rule': 'every pilot anchor whose xy is in the start region (exp_v6_learned_ett.region_of)', 'n': int(len(ids)), 'weight': float(anchors.weight[ids].sum()),
                    't_median': float(np.median(anchors.t[ids])), 'share_t_le_30': float(np.mean(anchors.t[ids] <= 30)), 'selection_from_evaluation': 'none'},
         'current_policy': {f'seed_{s}': {'path': str(CR.lineage_ckpt(s)), 'sha256': (MP.sha256(CR.lineage_ckpt(s)) if CR.lineage_ckpt(s).exists() else 'missing on this machine')} for s in MP.SEEDS},
         'queries': {'q0': 'the logged torque', 'q1': 'the lineage agent mode tanh(loc) at the logged obs31', f'q2..q{N_QUERIES - 1}': f'K = {K_SAMPLES} samples tanh(loc + scale * eps), eps ~ N(0, 1) from default_rng({QUERY_SEED0} + lineage), shape (n_start_anchors, K, 8), pinned',
                     'executed': 'once, from the restored logged state at absolute time t (build_v6_branch_replay.restore)'},
         'continuation': 'the SAME lineage agent, mode tanh(loc), closed-loop to the first reach frame / death / absolute horizon 800; the logged task goal; no goal hold (exp_v6_mainline_pilot._branch_one)',
         'hazard_draws': {'per_anchor': DRAWS, 'paired': 'identical across the six queries of an anchor', 'draw_0': f'{MP.HAZARD_SEED0} + anchor_id (the mainline / clip-round-A pairing)', 'draw_1': f'{HAZARD_SEED1} + anchor_id'},
         'filtering': 'none: every outcome kept', 'weights': 'anchor weight (the critic anchor law); the extended union spreads it equally over the six queries',
         'quantities': list(QUANT), 'sets': SETS, 'query_names': QUERY_NAMES,
         'strata': ['all start anchors', 'reset rows (t = 0)', 't in [1, 30)', 'logged shortcut episodes', 'logged detour episodes'],
         'reliability': 'draw-to-draw agreement of far-route completion per (anchor, query); distinct anchors with an added far-complete future per draw',
         'bootstrap': {'paired_by': 'anchor', 'resamples': BOOT, 'seed': BOOT_SEED}, 'gates': GATES,
         'stage_2_if_gates_pass': 'training comparison logged-query vs extended-query futures under the SAME continuation policy: the new (s, a_query) pairs become critic anchors; per-state total weight, budget, actor / BC data, losses, clip 0.1, BC 0.05 unchanged; on the user\'s go',
         'status': 'oracle (simulator) futures; engineering stage, not offline identification'}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# ---------------------------------------------------------------- generate
def query_actions(ckpt, obs31, lineage):
  """q1 = mode, q2.. = K pinned samples of the lineage agent at obs31 [n, 31]; returns [n, 1 + K, 8], loc, scale."""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt); pp = st.policy_params
  f = jax.jit(lambda o: nets.policy_network.apply(pp, o))
  loc, scale = [], []
  for i in range(0, len(obs31), 2048):
    d = f(jnp.asarray(obs31[i:i + 2048], jnp.float32)); loc.append(np.asarray(d.loc, np.float64)); scale.append(np.asarray(d.scale, np.float64))
  loc, scale = np.concatenate(loc), np.concatenate(scale)
  eps = np.random.default_rng(QUERY_SEED0 + int(lineage)).standard_normal((len(obs31), K_SAMPLES, MP.ACTION_DIM))
  q = np.concatenate([np.tanh(loc)[:, None], np.tanh(loc[:, None] + scale[:, None] * eps)], axis=1)
  return q.astype(np.float32), loc, scale


def _qc_worker(args):
  jobs, worker_seed, ckpt = args
  import build_v6_branch_replay as B
  env, _ = B._worker_env(worker_seed)
  act_fn = MP.mode_policy(ckpt)
  obs, _, _, _ = MP.load_dataset()
  out = []
  for (jid, aid, e, t, q, d, hz, a_q) in jobs:
    r = MP._branch_one(env, act_fn, obs[e, t, :MP.OBS_W], np.asarray(a_q, np.float32), t, hz, MP.HORIZON - int(t))
    r.update({'job_id': int(jid), 'anchor_id': int(aid), 'episode': int(e), 't': int(t), 'query_id': int(q), 'draw': int(d), 'hazard_seed': int(hz)})
    out.append(r)
  return out


def mode_generate(args):
  from multiprocessing import get_context
  s = int(args.lineage); ck = CR.lineage_ckpt(s)
  if not ck.exists():
    raise SystemExit(f'lineage agent missing: {ck}')
  out_path = branch_path(s)
  if out_path.exists() and not args.force:
    print(f'{out_path} exists', flush=True); return
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  ids = start_anchor_ids(anchors)
  if args.limit:
    ids = ids[:int(args.limit)]
  qa, loc, scale = query_actions(ck, anchors.obs31[ids], s)
  jobs = []
  for i, k in enumerate(ids):
    e, t = int(anchors.episode[k]), int(anchors.t[k])
    for d in range(DRAWS):
      hz = (MP.HAZARD_SEED0 if d == 0 else HAZARD_SEED1) + int(k)
      queries = [anchors.action[k]] + [qa[i, j] for j in range(1 + K_SAMPLES)]
      for q, a_q in enumerate(queries):
        jobs.append((len(jobs), int(k), e, t, q, d, hz, np.asarray(a_q, np.float32).tolist()))
  n_parts = max(1, args.workers * 4)
  parts = [jobs[i::n_parts] for i in range(n_parts)]; parts = [p for p in parts if p]
  wargs = [(p, WORKER_SEED0 + 1000 * s + i, str(ck)) for i, p in enumerate(parts)]
  t0 = time.time()
  if args.workers <= 1:
    res = [_qc_worker(a) for a in wargs]
  else:
    with get_context('spawn').Pool(args.workers) as pool:
      res = pool.map(_qc_worker, wargs)
  res = sorted([r for part in res for r in part], key=lambda r: r['job_id'])
  wall = time.time() - t0
  lengths = np.array([len(r['obs']) for r in res], np.int64)
  offset = np.concatenate([[0], np.cumsum(lengths)[:-1]])
  outcomes = np.array([r['outcome'] for r in res])
  meta = {'lineage': s, 'continuation_ckpt': str(ck), 'continuation_ckpt_sha256': MP.sha256(ck), 'n_anchors': int(len(ids)), 'n_queries': N_QUERIES, 'draws': DRAWS, 'n_branches': int(len(res)),
          'query_names': QUERY_NAMES, 'eps_seed': QUERY_SEED0 + s, 'hazard_seed_draw0': MP.HAZARD_SEED0, 'hazard_seed_draw1': HAZARD_SEED1, 'wall_seconds': wall,
          'policy_scale_median_at_anchors': float(np.median(scale)), 'policy_scale_p10_p90': [float(np.percentile(scale, 10)), float(np.percentile(scale, 90))],
          'outcome_counts': {k: int((outcomes == k).sum()) for k in np.unique(outcomes)}, 'filtering': 'none'}
  np.savez_compressed(out_path, obs_rows=np.concatenate([r['obs'] for r in res]).astype(np.float32),
                      act_rows=np.concatenate([r['act'] for r in res]).astype(np.float32), offset=offset, length=lengths,
                      anchor_id=np.array([r['anchor_id'] for r in res], np.int64), query_id=np.array([r['query_id'] for r in res], np.int64),
                      draw=np.array([r['draw'] for r in res], np.int64), episode=np.array([r['episode'] for r in res], np.int64), t=np.array([r['t'] for r in res], np.int64),
                      outcome=outcomes, steps=np.array([r['steps'] for r in res], np.int64), u1=np.array([r['u1'] for r in res]), u2=np.array([r['u2'] for r in res]),
                      restore_maxdiff=np.array([r['restore_maxdiff'] for r in res], np.float32), hazard_seed=np.array([r['hazard_seed'] for r in res], np.int64),
                      query_action=np.stack([r['act'][0] for r in res]).astype(np.float32), policy_loc=loc.astype(np.float32), policy_scale=scale.astype(np.float32),
                      start_anchor_ids=ids.astype(np.int64), meta=np.asarray(json.dumps(meta, sort_keys=True)))
  MP.write_json(OUT / f'generation_s{s}.json', meta)
  print(json.dumps(meta, indent=1), flush=True)


# ------------------------------------------------------------------ report
def features(path):
  """Per-branch features (same definitions as exp_v6_clip_round.branch_features)."""
  with np.load(path, allow_pickle=False) as d:
    xy = np.ascontiguousarray(d['obs_rows'][:, :2]).astype(np.float64)
    off, L = d['offset'].astype(np.int64), d['length'].astype(np.int64)
    oc = d['outcome'].astype(str)
    keys = {k: d[k] for k in ('anchor_id', 'query_id', 'draw', 't', 'restore_maxdiff', 'query_action', 'policy_scale', 'start_anchor_ids')}
    meta = json.loads(str(d['meta']))
  K = len(L)
  row_of = np.repeat(np.arange(K), L)
  m = np.arange(len(xy)) - np.repeat(off, L)
  after = m >= 1
  far_row = (xy[:, 1] >= CR.DETOUR_Y) & (xy[:, 0] < CR.DETOUR_MAX_X)
  zone_row = (np.abs(xy[:, 1]) < 2.0) & (((xy[:, 0] >= CR.ZONE_X[1][0]) & (xy[:, 0] <= CR.ZONE_X[1][1])) | ((xy[:, 0] >= CR.ZONE_X[2][0]) & (xy[:, 0] <= CR.ZONE_X[2][1])))
  entered_far = np.bincount(row_of, weights=(far_row & after), minlength=K) > 0
  entered_zone = np.bincount(row_of, weights=(zone_row & after), minlength=K) > 0
  success = oc == 'success'
  w = np.where(after, CR.GAMMA ** m, 0.0)
  z = np.bincount(row_of, weights=w, minlength=K); w = w / np.maximum(z[row_of], 1e-12)
  reg = region_of(xy)
  mass = {name: np.bincount(row_of, weights=w * np.isin(reg, idx), minlength=K) for name, idx in (('far', CR.FAR), ('goal', [CR.GOAL]), ('zone', CR.ZONES))}
  F = {'entered_far': entered_far, 'completed_far': entered_far & success, 'entered_zone': entered_zone, 'success': success, 'death': oc == 'death', 'timeout': oc == 'timeout',
       'length': L.astype(np.float64), 'mass_far': mass['far'], 'mass_goal': mass['goal'], 'mass_zone': mass['zone']}
  return {k: v.astype(np.float64) for k, v in F.items()}, keys, meta


def per_anchor(F, keys, ids, qset, draws=(0, 1)):
  """Mean of each quantity over the queries in qset and the draws, per start anchor (aligned with ids)."""
  pos = {int(a): i for i, a in enumerate(ids)}
  sel = np.isin(keys['query_id'], qset) & np.isin(keys['draw'], draws)
  idx = np.array([pos[int(a)] for a in keys['anchor_id'][sel]])
  cnt = np.bincount(idx, minlength=len(ids)).astype(np.float64)
  return {q: np.bincount(idx, weights=F[q][sel], minlength=len(ids)) / np.maximum(cnt, 1) for q in QUANT}, cnt


def boot_delta(d, w, rng):
  n = len(d); est = CR.wmean(d, w); vals = np.empty(BOOT)
  for b in range(BOOT):
    i = rng.integers(0, n, n); vals[b] = (w[i] * d[i]).sum() / w[i].sum()
  return {'delta': est, 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], 'se': float(vals.std())}


def mode_report(args):
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  man = MP.read_json(OUT / 'manifest.json')
  res = {'manifest_sha256': MP.sha256(OUT / 'manifest.json'), 'lineages': {}}
  gate_rows = []
  for s in MP.SEEDS:
    p = branch_path(s)
    if not p.exists():
      print(f'lineage {s}: no queries file yet', flush=True); continue
    F, keys, meta = features(p)
    ids = keys['start_anchor_ids']; W = anchors.weight[ids]; t = anchors.t[ids]; logged = route[anchors.episode[ids]]
    strata = {'all start anchors': np.ones(len(ids), bool), 'reset rows (t = 0)': t == 0, 't in [1, 30)': (t >= 1) & (t < 30), 'logged shortcut episodes': logged == 'shortcut', 'logged detour episodes': logged == 'detour'}
    rng = np.random.default_rng(BOOT_SEED + s)
    L = {'meta': meta, 'n_anchors': int(len(ids)), 'restore_maxdiff_max': float(keys['restore_maxdiff'].max()), 'policy_scale_median': float(np.median(keys['policy_scale'])), 'sets': {}, 'strata': {}, 'per_query': {}, 'reliability': {}}
    PA = {name: per_anchor(F, keys, ids, qset)[0] for name, qset in SETS.items()}
    for name, A in PA.items():
      L['sets'][name] = {q: CR.wmean(A[q], W) for q in QUANT}
    for q_id, qn in enumerate(QUERY_NAMES):
      A, _ = per_anchor(F, keys, ids, [q_id]); L['per_query'][qn] = {q: CR.wmean(A[q], W) for q in QUANT}
    # paired deltas extended - logged and added - logged (all start anchors)
    L['deltas'] = {}
    for a_name in ('extended', 'added', 'mode', 'samples'):
      L['deltas'][f'{a_name}_minus_logged'] = {q: boot_delta(PA[a_name][q] - PA['logged'][q], W, rng) for q in ('entered_far', 'completed_far', 'success', 'death', 'timeout', 'mass_goal', 'mass_far')}
    for sname, msk in strata.items():
      blk = {'n': int(msk.sum()), 'weight': float(W[msk].sum())}
      if msk.sum() >= 20:
        for name in ('logged', 'mode', 'samples', 'added', 'extended'):
          blk[name] = {q: CR.wmean(PA[name][q][msk], W[msk]) for q in ('entered_far', 'completed_far', 'success', 'death', 'timeout', 'mass_goal')}
        blk['completed_far_extended_minus_logged'] = boot_delta((PA['extended']['completed_far'] - PA['logged']['completed_far'])[msk], W[msk], rng)
      L['strata'][sname] = blk
    # reliability: draw-to-draw agreement of far completion per (anchor, query); distinct anchors with an added far-complete future per draw
    cf = F['completed_far']; key = keys['anchor_id'] * 100 + keys['query_id']
    d0 = {int(k): v for k, v in zip(key[keys['draw'] == 0], cf[keys['draw'] == 0])}; d1 = {int(k): v for k, v in zip(key[keys['draw'] == 1], cf[keys['draw'] == 1])}
    common = sorted(set(d0) & set(d1)); a0 = np.array([d0[k] for k in common]); a1 = np.array([d1[k] for k in common])
    both, either = float(((a0 > 0) & (a1 > 0)).sum()), float(((a0 > 0) | (a1 > 0)).sum())
    added = np.isin(keys['query_id'], SETS['added'])
    distinct = {f'draw_{d}': int(len(np.unique(keys['anchor_id'][added & (keys['draw'] == d) & (cf > 0)]))) for d in range(DRAWS)}
    distinct_logged = {f'draw_{d}': int(len(np.unique(keys['anchor_id'][(keys['query_id'] == 0) & (keys['draw'] == d) & (cf > 0)]))) for d in range(DRAWS)}
    L['reliability'] = {'pairs': int(len(common)), 'far_complete_rate_draw0': float(a0.mean()), 'far_complete_rate_draw1': float(a1.mean()), 'agreement': float(np.mean((a0 > 0) == (a1 > 0))),
                        'jaccard_of_far_complete': (both / either if either > 0 else None), 'distinct_anchors_with_added_far_complete': distinct, 'distinct_anchors_with_logged_far_complete': distinct_logged}
    # user's read-out (2026-09-20): where the current mode does NOT complete the far route, does another query?  (per draw; reset rows and all start anchors)
    alt = {}
    for sname, amask in (('reset rows', anchors.t[ids] == 0), ('all start anchors', np.ones(len(ids), bool))):
      aset = set(int(a) for a in ids[amask]); blk = {}
      for d in range(DRAWS):
        sel = (keys['draw'] == d) & np.isin(keys['anchor_id'], list(aset))
        A, Q, C, S = keys['anchor_id'][sel], keys['query_id'][sel], cf[sel] > 0, F['success'][sel] > 0
        mode_fail_far = {int(a) for a, q, c in zip(A, Q, C) if q == 1 and not c}
        mode_fail_succ = {int(a) for a, q, c in zip(A, Q, S) if q == 1 and not c}
        other_far = {int(a) for a, q, c in zip(A, Q, C) if q != 1 and c}
        other_far_nonlog = {int(a) for a, q, c in zip(A, Q, C) if q >= 2 and c}
        other_succ = {int(a) for a, q, c in zip(A, Q, S) if q != 1 and c}
        blk[f'draw_{d}'] = {'mode_not_far_complete': len(mode_fail_far), 'of_which_another_query_far_complete': len(mode_fail_far & other_far), 'of_which_a_sample_far_complete': len(mode_fail_far & other_far_nonlog),
                            'mode_not_success': len(mode_fail_succ), 'of_which_another_query_success': len(mode_fail_succ & other_succ)}
      alt[sname] = blk
    L['alternative_query_when_mode_fails'] = alt
    g1 = L['deltas']['extended_minus_logged']['completed_far']; g2 = (L['sets']['extended']['mass_goal'], L['sets']['logged']['mass_goal'])
    L['gates'] = {'G1_completed_far_gain': {'delta': g1['delta'], 'ci95': g1['ci95'], 'pass': bool(g1['delta'] >= 0.03 and g1['ci95'][0] > 0)},
                  'G2_goal_mass_kept': {'extended': g2[0], 'logged': g2[1], 'pass': bool(g2[0] >= 0.8 * g2[1])},
                  'G3_spread': {**distinct, 'pass': bool(all(v >= 100 for v in distinct.values()))}}
    gate_rows.append(all(v['pass'] for v in L['gates'].values()))
    res['lineages'][f'seed_{s}'] = L
  res['gate'] = {'rule': 'G1, G2 and G3 in every lineage', 'lineages_reported': len(gate_rows), 'passed': bool(len(gate_rows) == len(MP.SEEDS) and all(gate_rows))}
  MP.write_json(OUT / 'report.json', res)
  write_report_md(res, man)
  print('GATE', 'PASSED' if res['gate']['passed'] else 'NOT PASSED', f'({len(gate_rows)} / {len(MP.SEEDS)} lineages reported)', flush=True)


def write_report_md(res, man):
  L = ['# Query coverage, stage 1: logged-query vs extended-query futures at the start / early-decision anchors (generation only)', '',
       f'`manifest.json` (sealed before generation; gates pre-registered), `report.json`.  States = every pilot anchor in the start region ({man["states"]["n"]} anchors, weight {man["states"]["weight"]:.3f}, t median {man["states"]["t_median"]:.0f}); '
       f'queries q0 logged torque / q1 lineage-agent mode / q2..q5 four pinned samples; the same lineage agent continues every query (mode, closed-loop); two paired hazard draws per anchor; nothing filtered.  '
       'Anchor-weighted shares; "extended" spreads each anchor\'s weight equally over its six queries; deltas are paired by anchor (bootstrap 95 % CI).  '
       'Entered far = y >= 6 at x < 2 after the query; completed far = entered and reached the goal; masses = the critic goal marginal (truncated geometric, gamma 0.999).', '']
  g = res['gate']
  L += [f'**Gate for stage 2** ({g["rule"]}): **{"PASSED" if g["passed"] else "NOT PASSED"}** ({g["lineages_reported"]} / 3 lineages reported).', '']
  for lin, Lg in res['lineages'].items():
    G = Lg['gates']
    L += [f'## {lin} (n = {Lg["n_anchors"]} anchors; policy scale median {Lg["policy_scale_median"]:.3f}; restore max diff {Lg["restore_maxdiff_max"]:.1e})', '',
          f'G1 completed-far gain (extended - logged) {G["G1_completed_far_gain"]["delta"]:+.4f} CI [{G["G1_completed_far_gain"]["ci95"][0]:+.4f}, {G["G1_completed_far_gain"]["ci95"][1]:+.4f}] -> {"pass" if G["G1_completed_far_gain"]["pass"] else "FAIL"}; '
          f'G2 goal mass extended / logged {G["G2_goal_mass_kept"]["extended"]:.3f} / {G["G2_goal_mass_kept"]["logged"]:.3f} -> {"pass" if G["G2_goal_mass_kept"]["pass"] else "FAIL"}; '
          f'G3 distinct anchors with an added far-complete future draw 0 / 1: {G["G3_spread"]["draw_0"]} / {G["G3_spread"]["draw_1"]} -> {"pass" if G["G3_spread"]["pass"] else "FAIL"}.', '',
          '| query set | entered far | completed far | entered zone | success | death | timeout | length | mass far | mass goal | mass zone |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for name in ('logged', 'mode', 'samples', 'added', 'extended'):
      b = Lg['sets'][name]
      L.append(f'| {name} | {b["entered_far"]:.3f} | {b["completed_far"]:.3f} | {b["entered_zone"]:.3f} | {b["success"]:.3f} | {b["death"]:.3f} | {b["timeout"]:.3f} | {b["length"]:.0f} | {b["mass_far"]:.3f} | {b["mass_goal"]:.3f} | {b["mass_zone"]:.3f} |')
    L += ['', '| single query | entered far | completed far | success | death | timeout | mass goal |', '|---|---|---|---|---|---|---|']
    for qn, b in Lg['per_query'].items():
      L.append(f'| {qn} | {b["entered_far"]:.3f} | {b["completed_far"]:.3f} | {b["success"]:.3f} | {b["death"]:.3f} | {b["timeout"]:.3f} | {b["mass_goal"]:.3f} |')
    L += ['', '| paired delta vs logged | entered far | completed far | success | death | timeout | mass goal | mass far |', '|---|---|---|---|---|---|---|---|']
    for dn, D in Lg['deltas'].items():
      f = lambda q: f'{D[q]["delta"]:+.4f} [{D[q]["ci95"][0]:+.4f}, {D[q]["ci95"][1]:+.4f}]'
      L.append(f'| {dn} | {f("entered_far")} | {f("completed_far")} | {f("success")} | {f("death")} | {f("timeout")} | {f("mass_goal")} | {f("mass_far")} |')
    L += ['', '| stratum | n | weight | completed far logged / mode / samples / added / extended | extended - logged (CI) | success logged / extended | mass goal logged / extended |', '|---|---|---|---|---|---|---|']
    for sname, b in Lg['strata'].items():
      if 'logged' in b:
        d = b['completed_far_extended_minus_logged']
        L.append(f'| {sname} | {b["n"]} | {b["weight"]:.4f} | {b["logged"]["completed_far"]:.3f} / {b["mode"]["completed_far"]:.3f} / {b["samples"]["completed_far"]:.3f} / {b["added"]["completed_far"]:.3f} / {b["extended"]["completed_far"]:.3f} | '
                 f'{d["delta"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] | {b["logged"]["success"]:.3f} / {b["extended"]["success"]:.3f} | {b["logged"]["mass_goal"]:.3f} / {b["extended"]["mass_goal"]:.3f} |')
      else:
        L.append(f'| {sname} | {b["n"]} | {b["weight"]:.4f} | (fewer than 20 anchors) | | | |')
    if 'alternative_query_when_mode_fails' in Lg:
      L += ['', '| stratum | draw | mode not far-complete | of which another query far-complete | of which a sample far-complete | mode not success | of which another query success |', '|---|---|---|---|---|---|---|']
      for sname, blk in Lg['alternative_query_when_mode_fails'].items():
        for dn, b in blk.items():
          L.append(f'| {sname} | {dn[-1]} | {b["mode_not_far_complete"]} | {b["of_which_another_query_far_complete"]} | {b["of_which_a_sample_far_complete"]} | {b["mode_not_success"]} | {b["of_which_another_query_success"]} |')
    R = Lg['reliability']
    L += ['', f'Reliability: far-complete rate per (anchor, query) draw 0 / draw 1 = {R["far_complete_rate_draw0"]:.4f} / {R["far_complete_rate_draw1"]:.4f}; agreement of the far-complete indicator across draws {R["agreement"]:.3f}; '
              f'Jaccard of the far-complete sets {R["jaccard_of_far_complete"] if R["jaccard_of_far_complete"] is None else round(R["jaccard_of_far_complete"], 3)}; '
              f'distinct anchors with a logged far-complete future draw 0 / 1: {R["distinct_anchors_with_logged_far_complete"]["draw_0"]} / {R["distinct_anchors_with_logged_far_complete"]["draw_1"]}.', '']
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('mode', choices=['seal', 'generate', 'report'])
  ap.add_argument('--lineage', type=int, default=0)
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': mode_seal, 'generate': mode_generate, 'report': mode_report}[args.mode](args)


if __name__ == '__main__':
  main()
