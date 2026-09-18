"""AntMaze V6 round-1 critic diagnostic: which generalisation boundary fails?

The three round-1 critics (`critics_r1/seed_{0,1,2}/final.pkl`, 30k NCE
updates on `replay_policy_r1.npz`) order the candidate torques of their
own training keys the way the two recorded draws went (~0.90) and are at
chance on the held-out keys (~0.50) under both the exact-goal and the
region-integrated readout (`gate_policy_r1_v2*.md`).  This diagnostic
separates three explanations with frozen checkpoints and FRESH oracle
outcomes:

  A. memorisation of the particular futures realised during training
     (familiar state, familiar torque, fresh consequences);
  B. failure to generalise to new torques at a familiar state
     (same anchors, three new samples from the frozen policy);
  C. failure to generalise to states of other episodes
     (the episode-held-out anchors, their existing candidates, fresh
     consequences).

Every layer: the anchor's exact recorded state, goal and absolute time are
restored, the candidate torque is executed once, and the SAME frozen
continuation policy the replay was built with acts closed-loop (its mode)
to the horizon -- no donor tails, no intent -- with 16 new independent
hazard / clock / jitter draws per anchor, paired across the candidates of
an anchor (and across layers A and B, which share anchors and seeds).
P_goal = the replay's relabeling law (gamma 0.999, radius 0.5, zero-torque
hold after reaching).  Nothing is trained or tuned; the simulator calls are
oracle diagnostics, not offline evidence.

Modes
  seal     provenance (hashes, configs) and the manifest: anchors selected
           per stratum without consulting outcomes or critic scores (spread
           across episodes), their observations, goals, candidate torques,
           the new seeds and the analysis rules; the original outcome
           records; which states earlier diagnostics already examined.
  run      the fresh outcomes (per-draw records).
  analyze  the frozen critics on every key (region-integrated readout over
           the training NCE goal marginal, per head and deployed min;
           exact-goal logits as the secondary readout), the layer
           comparisons, the cross-fitted empirical selector, coverage,
           episode-clustered uncertainty; metrics.json + REPORT.md.

  V6_DATASET_STEM=antmaze_rockfall_clock_v6_p050 V6_P_ACTIVE=0.5 V6_BRANCH_OUT=... \\
    python scripts/diag_v6_r1_abc.py seal    --cont-ckpt <pure-BC final.pkl> --critics <3 ckpts>
    python scripts/diag_v6_r1_abc.py run     --cont-ckpt <...> --workers 10
    python scripts/diag_v6_r1_abc.py analyze --cont-ckpt <...> --critics <3 ckpts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402

ABC_SEED = 909_707_000          # disjoint from every round-1 seed range (build, hold-out, validate)
STRATA = ('start_early', 'start_late', 'shortcut_early', 'turn', 'north_leg')
N_PER_STRATUM, N_DRAWS, N_NEW, THR, RADIUS = 32, 16, 3, 0.3, 0.5
A_CANDS = ('recorded', 'sample0', 'sample1')
C_CANDS = ('recorded', 'mode', 'sample0', 'sample1', 'sample2')
C_MATCHED = ('recorded', 'sample0', 'sample1')
EPS_P = R.EPS_P
BOOT_DECIDED = 0.9              # draw-bootstrap: share of resamples with the point estimate's sign


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def out_dir(args):
  return B.OUT / args.out


# -------------------------------------------------------------------- seal
def select_anchors(pool, n, rng):
  """Round-robin over episodes (random episode order, random anchor within an
  episode) until n anchors or the pool is exhausted.  Outcomes and scores
  are never consulted."""
  by_ep = {}
  for a in pool:
    by_ep.setdefault(a['episode'], []).append(a)
  eps = list(by_ep)
  rng.shuffle(eps)
  for e in eps:
    rng.shuffle(by_ep[e])
  chosen = []
  while len(chosen) < n and any(by_ep.values()):
    for e in eps:
      if by_ep[e] and len(chosen) < n:
        chosen.append(by_ep[e].pop())
  return chosen


def seal(args):
  out = out_dir(args)
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit(f'{out / "manifest.json"} exists; the diagnostic is sealed (use --force to reseal before any run)')
  replay = B.OUT / args.replay
  holdout = B.OUT / args.holdout
  prov = {'reference_commit': args.ref_commit, 'dataset': str(B.DATASET), 'dataset_sha256': sha256(B.DATASET),
          'replay': str(replay), 'replay_sha256': sha256(replay), 'holdout_table': str(holdout), 'holdout_sha256': sha256(holdout),
          'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': sha256(args.cont_ckpt),
          'critics': {str(c): sha256(c) for c in args.critics}}
  with np.load(replay, allow_pickle=False) as d:
    prov['replay_meta'] = json.loads(str(d['meta']))
  for c in args.critics:
    m = Path(c).parent / 'branch_manifest.json'
    if m.exists():
      prov.setdefault('critic_manifests', {})[str(c)] = json.loads(m.read_text(encoding='utf-8'))
  pm = Path(args.cont_ckpt).parent / 'prep.json'
  if pm.exists():
    prov['continuation_prep'] = json.loads(pm.read_text(encoding='utf-8'))
  obs, act, lengths, meta, route_rec = R.load_dataset()
  rng = np.random.default_rng(args.seed)
  # ---- layer A / B pool: the critics' own fit keys, dense strata, all three original candidates present
  fit = R.keys_from_replay(replay)
  by_anchor = {}
  for (aid, cand), e in fit.items():
    by_anchor.setdefault(aid, {})[cand] = e
  pool_a = {s: [] for s in STRATA}
  for aid, cands in by_anchor.items():
    e0 = next(iter(cands.values()))
    if e0['set'] in STRATA and all(c in cands for c in A_CANDS):
      pool_a[e0['set']].append({'anchor_id': aid, 'episode': e0['episode'], 't': e0['t'], 'stratum': e0['set'], 'cands': cands})
  # ---- layer C pool: the episode-held-out table's anchors
  T = R.load_table(holdout)
  hk = R.keyed(T)
  by_h = {}
  t_of = {}
  for i in range(len(T['anchor_id'])):
    t_of[int(T['anchor_id'][i])] = int(T['t'][i])
  for (aid, cand), e in hk.items():
    by_h.setdefault(aid, {})[cand] = e
  pool_c = {s: [] for s in STRATA}
  for aid, cands in by_h.items():
    e0 = next(iter(cands.values()))
    if e0['set'] in STRATA and all(c in cands for c in C_CANDS):
      pool_c[e0['set']].append({'anchor_id': aid, 'episode': e0['episode'], 't': t_of[aid], 'stratum': e0['set'], 'cands': cands})
  bundle = R.policy_bundle(args.cont_ckpt)
  anchors, idx = [], 0
  prov_check = {'max_abs_obs_diff_A': 0.0, 'max_abs_obs_diff_C': 0.0}
  for s in STRATA:
    for a in select_anchors(pool_a[s], args.n_per_stratum, rng):
      e, t = a['episode'], a['t']
      o0 = a['cands']['recorded']['obs0']
      prov_check['max_abs_obs_diff_A'] = max(prov_check['max_abs_obs_diff_A'], float(np.abs(o0 - obs[e, t]).max()))
      loc, scale = bundle['params_fn'](o0[None])
      loc, scale = loc[0], scale[0]
      cand_a = {c: a['cands'][c]['act0'].astype(np.float32) for c in A_CANDS}
      cand_b = {f'new{k}': v for k, (_, v) in enumerate(R.candidates_for(loc, scale, act[e, t], a['anchor_id'], [f'sample{k}' for k in range(N_NEW)], ABC_SEED + 17))}
      mode = np.tanh(loc).astype(np.float32)
      rec = {'index': idx, 'layer': 'AB', 'anchor_id': int(a['anchor_id']), 'episode': int(e), 't': int(t), 'stratum': s,
             'obs': o0.astype(float).tolist(), 'goal_xy': o0[B.STATE_DIM:B.STATE_DIM + 2].astype(float).tolist(),
             'policy_loc': loc.astype(float).tolist(), 'policy_scale': scale.astype(float).tolist(), 'policy_mode': mode.astype(float).tolist(),
             'candidates': {**{f'A|{c}': v.astype(float).tolist() for c, v in cand_a.items()},
                            **{f'B|{c}': v.astype(float).tolist() for c, v in cand_b.items()}},
             'original_outcomes': {f'A|{c}': {str(d): p for d, p in a['cands'][c]['p'].items()} for c in A_CANDS},
             'previously_examined': 'fit key of the round-1 critics; scored in gate_policy_r1_v2_fit (original two draws)',
             'draw_seeds': [ABC_SEED + 100 * idx + r for r in range(N_DRAWS)]}
      anchors.append(rec); idx += 1
  for s in STRATA:
    for a in select_anchors(pool_c[s], args.n_per_stratum, rng):
      e, t = a['episode'], a['t']
      o0 = a['cands']['recorded']['obs0']
      prov_check['max_abs_obs_diff_C'] = max(prov_check['max_abs_obs_diff_C'], float(np.abs(o0 - obs[e, t]).max()))
      loc, scale = bundle['params_fn'](o0[None])
      loc, scale = loc[0], scale[0]
      rec = {'index': idx, 'layer': 'C', 'anchor_id': int(a['anchor_id']), 'episode': int(e), 't': int(t), 'stratum': s,
             'obs': o0.astype(float).tolist(), 'goal_xy': o0[B.STATE_DIM:B.STATE_DIM + 2].astype(float).tolist(),
             'policy_loc': loc.astype(float).tolist(), 'policy_scale': scale.astype(float).tolist(), 'policy_mode': np.tanh(loc).astype(float).tolist(),
             'candidates': {f'C|{c}': a['cands'][c]['act0'].astype(float).tolist() for c in C_CANDS},
             'original_outcomes': {f'C|{c}': {str(d): p for d, p in a['cands'][c]['p'].items()} for c in C_CANDS},
             'previously_examined': 'episode-held-out anchor of holdout_policy_r1.npz; scored in gate_policy_r1 / _v2 (original four draws, seeds R1_SEED+300000+...)',
             'draw_seeds': [ABC_SEED + 100 * idx + r for r in range(N_DRAWS)]}
      anchors.append(rec); idx += 1
  # the policy's log-likelihood and the torque distances of every candidate (recorded once, never used to select)
  import jax.numpy as jnp
  from crl import networks
  for rec in anchors:
    loc = np.asarray(rec['policy_loc'], np.float32); scale = np.asarray(rec['policy_scale'], np.float32)
    mode = np.asarray(rec['policy_mode'], np.float32)
    A_rec = np.asarray(rec['candidates'].get('A|recorded', rec['candidates'].get('C|recorded')), np.float32)
    rec['candidate_stats'] = {}
    for c, v in rec['candidates'].items():
      v = np.asarray(v, np.float32)
      lp = float(np.asarray(networks.tanh_normal_log_prob(networks.TanhNormalParams(jnp.asarray(loc[None]), jnp.asarray(scale[None])), jnp.asarray(v[None])))[0])
      rec['candidate_stats'][c] = {'log_prob_policy': lp, 'dist_to_mode': float(np.linalg.norm(v - mode)),
                                   'dist_to_recorded': float(np.linalg.norm(v - A_rec)), 'frac_saturated': float(np.mean(np.abs(v) > 0.95))}
  counts = {}
  for rec in anchors:
    k = (rec['layer'], rec['stratum'])
    c = counts.setdefault(f'{k[0]}|{k[1]}', {'anchors': 0, 'episodes': set()})
    c['anchors'] += 1; c['episodes'].add(rec['episode'])
  counts = {k: {'anchors': v['anchors'], 'episodes': len(v['episodes'])} for k, v in counts.items()}
  manifest = {
      'diagnostic': 'AntMaze V6 round-1 critic: memorisation (A) vs new-action (B) vs new-episode-state (C)',
      'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'provenance': prov, 'provenance_check': prov_check,
      'continuation': {'mode': 'frozen deployment policy, tanh(loc) mode, closed-loop from step 2', 'donor_tails': False, 'intent': None,
                       'horizon': B.HORIZON, 'post_goal': 'zero-torque hold to the horizon (absorbing goal)', 'termination': 'env done (death) or horizon'},
      'law': {'discount': R.GAMMA, 'radius': RADIUS, 'goal_region': 'XY within radius 0.5 of the anchor episode\'s recorded goal',
              'p_goal': 'build_v6_branch_replay.goal_frame_probability from row 0'},
      'readouts': {'primary': 'region-integrated exp(f) over the training NCE goal marginal within radius 0.5 of the anchor goal '
                              '(marginal_goal_frames(replay, per_path=4, seed=0), identical frames across the candidates of an anchor, '
                              'weighted by the in-region mass), per twin head and deployed min over heads (min on logits before exp)',
                   'secondary': 'exact recorded-goal logit per head and min'},
      'design': {'strata': list(STRATA), 'n_per_stratum_target': args.n_per_stratum, 'n_draws': N_DRAWS, 'n_new_actions': N_NEW,
                 'layer_A': 'fit-key anchors, original candidates ' + str(A_CANDS) + ', fresh draws',
                 'layer_B': 'same anchors as A, three new policy samples (candidate seed ABC_SEED+17), same draw seeds as A (common random numbers)',
                 'layer_C': 'episode-held-out anchors, original candidates ' + str(C_CANDS) + ', fresh draws; matched subset ' + str(C_MATCHED),
                 'draw_seed_rule': 'ABC_SEED + 100 * anchor_index + draw; all candidates of an anchor (and layers A and B) share a draw seed',
                 'selection': 'round-robin over episodes with rng seed %d; outcomes, scores and validated-pair membership not consulted' % args.seed},
      'analysis_rules': {
          'pair_classes': 'per within-anchor pair, log ratio of 16-draw mean P (floor 1e-3): tie = exactly 0; weak = 0 < |ratio| <= 0.3; '
                          'decided = |ratio| > 0.3; bootstrap-decided = decided and >= 0.9 of 1000 draw-resamples keep the sign',
          'agreement': 'share of decided pairs whose critic ordering matches; ties never counted as wrong; s.e. by episode bootstrap',
          'pick_gain': 'P[argmax critic] - mean_c P_c on all 16 draws (the critic does not see outcomes), and - P_mode where the mode is a candidate',
          'cross_fit_selector': 'argmax of mean P on draws 0-7 evaluated on draws 8-15, and the swap; gain vs the eval-half candidate mean',
          'layer_A_memorisation': 'critic vs ORIGINAL two-draw labels, critic vs FRESH sixteen-draw labels at identical keys, original vs fresh labels',
          'not_done': 'no expansion of draws, no head/radius sweep, no model update'},
      'counts': counts, 'anchors': anchors}
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
  print(json.dumps({'counts': counts, 'provenance_check': prov_check, 'anchors': len(anchors)}, indent=1), flush=True)


# --------------------------------------------------------------------- run
def run(args):
  out = out_dir(args)
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  if (out / 'outcomes.npz').exists() and not args.force:
    raise SystemExit('outcomes.npz exists')
  assert sha256(args.cont_ckpt) == man['provenance']['continuation_sha256'], 'continuation checkpoint differs from the sealed one'
  jobs = []
  for rec in man['anchors']:
    for cand, torque in rec['candidates'].items():
      for r, seed in enumerate(rec['draw_seeds']):
        jobs.append((rec['anchor_id'], rec['episode'], rec['t'], rec['stratum'], cand, np.asarray(torque, np.float32), r, seed, False))
  if args.limit:
    jobs = jobs[:args.limit]
  print(f'run: {len(man["anchors"])} anchors, {len(jobs)} paths, {args.workers} workers', flush=True)
  t0 = time.time()
  res = R.run_parallel(jobs, args.workers, ABC_SEED + 5, args.cont_ckpt)
  wall = time.time() - t0
  R.save_table(out / ('outcomes.npz' if not args.limit else 'outcomes_smoke.npz'), res,
               {'wall_seconds': wall, 'paths': len(res), 'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': man['provenance']['continuation_sha256']})
  print(f'done in {wall:.0f}s, {len(res)} paths', flush=True)


# ----------------------------------------------------------------- analyze
def load_keys(out, smoke=False, extra=()):
  man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  T = R.load_table(out / ('outcomes_smoke.npz' if smoke else 'outcomes.npz'))
  for path in extra:
    # additional outcome tables (e.g. an arm's own training torques at the A anchors,
    # generated with the same draw seeds); same columns, concatenated
    T2 = R.load_table(path)
    T = {k: (np.concatenate([T[k], T2[k]]) if k != 'meta' else T[k]) for k in T}
  info = {a['anchor_id']: a for a in man['anchors']}
  keys = {}
  for i in range(len(T['p_goal'])):
    aid = int(T['anchor_id'][i]); cand = str(T['cand'][i]); layer = cand.split('|')[0]
    k = (layer, aid, cand.split('|')[1])
    e = keys.setdefault(k, {'obs0': T['obs0'][i], 'act0': T['act0'][i], 'set': str(T['set'][i]), 'episode': int(T['episode'][i]),
                            'p': {}, 'success': {}, 'failure': {}, 'around': {}})
    d = int(T['draw'][i])
    e['p'][d] = float(T['p_goal'][i]); e['success'][d] = bool(T['success'][i]); e['failure'][d] = bool(T['failure'][i])
    e['around'][d] = bool(T['max_y'][i] >= B.DETOUR_Y)
  return man, keys, info


def rankdata(x):
  x = np.asarray(x, float); order = np.argsort(x, kind='mergesort'); ranks = np.empty(len(x)); sx = x[order]
  i = 0
  while i < len(x):
    j = i
    while j + 1 < len(x) and sx[j + 1] == sx[i]:
      j += 1
    ranks[order[i:j + 1]] = (i + j) / 2 + 1
    i = j + 1
  return ranks


def log_ratio(pi, pj, draws):
  return float(np.log((np.mean([pi[d] for d in draws]) + EPS_P) / (np.mean([pj[d] for d in draws]) + EPS_P)))


def classify_pair(pi, pj, draws, rng):
  d = log_ratio(pi, pj, draws)
  if d == 0:
    return 'tie', d, 0.0
  if abs(d) <= THR:
    return 'weak', d, 0.0
  arr_i = np.array([pi[x] for x in draws]); arr_j = np.array([pj[x] for x in draws])
  n = len(draws); same = 0
  for _ in range(1000):
    pick = rng.integers(0, n, n)
    b = np.log((arr_i[pick].mean() + EPS_P) / (arr_j[pick].mean() + EPS_P))
    same += int(np.sign(b) == np.sign(d))
  return 'decided', d, same / 1000


def layer_metrics(keys, layer, cands_filter, scores, strata, rng, mixed_with=None):
  """Per scorer: within-anchor pair classes on the fresh 16 draws, agreement
  among decided / bootstrap-decided pairs (episode-bootstrap s.e.), pick gains,
  cross-fitted empirical selector, all per stratum and pooled.  ``cands_filter``
  = candidate labels to keep; ``mixed_with`` = a second layer whose candidates
  are paired against this layer's (they share anchors and draws)."""
  anchors = {}
  for (ly, aid, c), e in keys.items():
    if ly == layer and (cands_filter is None or c in cands_filter):
      anchors.setdefault(aid, {})[(ly, c)] = e
    if mixed_with and ly == mixed_with:
      anchors.setdefault(aid, {})[(ly, c)] = e
  draws = list(range(N_DRAWS))
  h1, h2 = draws[:N_DRAWS // 2], draws[N_DRAWS // 2:]
  # pair classes (independent of the scorer), cached
  pairs = {}
  for aid, cs in anchors.items():
    names = list(cs)
    if mixed_with:
      names = [(ly, c) for (ly, c) in names]
    for i in range(len(names)):
      for j in range(i + 1, len(names)):
        if mixed_with and names[i][0] == names[j][0]:
          continue                            # mixed layer: only cross pairs
        dd = sorted(set(cs[names[i]]['p']) & set(cs[names[j]]['p']))
        if not dd:
          continue
        cls, d, boot = classify_pair(cs[names[i]]['p'], cs[names[j]]['p'], dd, rng)
        pairs[(aid, names[i], names[j])] = (cls, d, boot)
  out = {}
  for sname, sc in scores.items():
    out[sname] = {}
    for stratum in list(strata) + ['pooled']:
      cls_count = {'tie': 0, 'weak': 0, 'decided': 0, 'boot_decided': 0}
      agree, agree_boot, per_ep = [], [], {}
      gains, gains_mode, gain_cf, n_anch = [], [], [], 0
      p_pick, p_mean, p_mode, p_best_insample = [], [], [], []
      for aid, cs in anchors.items():
        e0 = next(iter(cs.values()))
        if not (stratum == 'pooled' or e0['set'] == stratum):
          continue
        n_anch += 1
        ep = e0['episode']
        pe = per_ep.setdefault(ep, {'same': 0, 'opp': 0, 'same_b': 0, 'opp_b': 0})
        for (a_, ni, nj), (cls, d, boot) in pairs.items():
          if a_ != aid:
            continue
          cls_count[cls] += 1
          if cls != 'decided':
            continue
          fs = np.sign(sc[(ni[0], aid, ni[1])] - sc[(nj[0], aid, nj[1])])
          ok = float(fs == np.sign(d))
          agree.append(ok); pe['same' if ok else 'opp'] += 1
          if boot >= BOOT_DECIDED:
            cls_count['boot_decided'] += 1
            agree_boot.append(ok); pe['same_b' if ok else 'opp_b'] += 1
        # picks (the critic never sees outcomes: all 16 draws evaluate its pick)
        names = list(cs)
        if mixed_with:
          continue
        Pm = np.array([np.mean(list(cs[n]['p'].values())) for n in names])
        F = np.array([sc[(n[0], aid, n[1])] for n in names])
        k = int(np.argmax(F))
        gains.append(Pm[k] - Pm.mean()); p_pick.append(Pm[k]); p_mean.append(Pm.mean())
        if any(n[1] == 'mode' for n in names):
          im = [i for i, n in enumerate(names) if n[1] == 'mode'][0]
          gains_mode.append(Pm[k] - Pm[im]); p_mode.append(Pm[im])
        # cross-fitted empirical selector (scorer-independent; computed once per anchor, stored under every scorer for the table)
        dd = set.intersection(*[set(cs[n]['p']) for n in names])
        d1, d2 = [d for d in h1 if d in dd], [d for d in h2 if d in dd]
        if d1 and d2:
          P1 = np.array([np.mean([cs[n]['p'][d] for d in d1]) for n in names]); P2 = np.array([np.mean([cs[n]['p'][d] for d in d2]) for n in names])
          gain_cf.append(0.5 * ((P2[int(np.argmax(P1))] - P2.mean()) + (P1[int(np.argmax(P2))] - P1.mean())))
        p_best_insample.append(Pm.max())
      def boot_se(key_s, key_o):
        eps_ = [e for e in per_ep if per_ep[e][key_s] + per_ep[e][key_o] > 0]
        if len(eps_) < 2:
          return None
        brng = np.random.default_rng(1); bs = []
        for _ in range(1000):
          pick = brng.choice(len(eps_), size=len(eps_), replace=True)
          sm = sum(per_ep[eps_[k]][key_s] for k in pick); op = sum(per_ep[eps_[k]][key_o] for k in pick)
          bs.append(sm / (sm + op) if sm + op else np.nan)
        return float(np.nanstd(bs))
      def mean_se(v):
        v = np.array(v)
        return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else ((float(v.mean()), None) if len(v) else (None, None))
      g, gse = mean_se(gains); gm, gmse = mean_se(gains_mode); gc, gcse = mean_se(gain_cf)
      out[sname][stratum] = {
          'n_anchors': n_anch, 'n_episodes': len({next(iter(cs.values()))['episode'] for aid, cs in anchors.items()
                                                  if stratum == 'pooled' or next(iter(cs.values()))['set'] == stratum}),
          'pairs': dict(cls_count), 'n_pairs': sum(v for k, v in cls_count.items() if k != 'boot_decided'),
          'agreement_decided': float(np.mean(agree)) if agree else None, 'se_decided': boot_se('same', 'opp'),
          'n_decided': len(agree), 'same': int(sum(agree)), 'opposite': int(len(agree) - sum(agree)),
          'agreement_boot_decided': float(np.mean(agree_boot)) if agree_boot else None, 'se_boot_decided': boot_se('same_b', 'opp_b'),
          'n_boot_decided': len(agree_boot),
          'pick_gain': g, 'pick_gain_se': gse, 'pick_gain_vs_mode': gm, 'pick_gain_vs_mode_se': gmse,
          'p_pick': float(np.mean(p_pick)) if p_pick else None, 'p_mean': float(np.mean(p_mean)) if p_mean else None,
          'p_mode': float(np.mean(p_mode)) if p_mode else None,
          'p_best_in_sample_not_achievable': float(np.mean(p_best_insample)) if p_best_insample else None,
          'cross_fit_selector_gain': gc, 'cross_fit_selector_gain_se': gcse}
  return out


def original_vs_fresh(keys, info, layer, cands, strata, scores):
  """Layer A (and C for continuity): the critic against the ORIGINAL labels,
  against the FRESH labels, and the labels against each other."""
  anchors = {}
  for (ly, aid, c), e in keys.items():
    if ly == layer and c in cands:
      anchors.setdefault(aid, {})[c] = e
  draws = list(range(N_DRAWS))
  out = {}
  for stratum in list(strata) + ['pooled']:
    rows = {'orig_pairs_decided': 0, 'fresh_pairs_decided': 0, 'both_decided': 0, 'labels_same': 0, 'labels_opposite': 0,
            'orig_decided_fresh_tie_or_weak': 0, 'critic': {}}
    per_key_orig, per_key_fresh = [], []
    for aid, cs in anchors.items():
      if not (stratum == 'pooled' or next(iter(cs.values()))['set'] == stratum):
        continue
      if any(f'{layer}|{c}' not in info[aid]['original_outcomes'] for c in cs):
        continue                                  # anchors without original records (e.g. reserved evaluation episodes)
      orig = {c: {int(d): p for d, p in info[aid]['original_outcomes'][f'{layer}|{c}'].items()} for c in cs}
      for c in cs:
        per_key_orig.append(np.mean(list(orig[c].values()))); per_key_fresh.append(np.mean(list(cs[c]['p'].values())))
      names = list(cs)
      for i in range(len(names)):
        for j in range(i + 1, len(names)):
          ci, cj = names[i], names[j]
          d_o = log_ratio(orig[ci], orig[cj], list(orig[ci]))
          d_f = log_ratio(cs[ci]['p'], cs[cj]['p'], sorted(set(cs[ci]['p']) & set(cs[cj]['p'])))
          o_dec, f_dec = abs(d_o) > THR, abs(d_f) > THR
          rows['orig_pairs_decided'] += int(o_dec); rows['fresh_pairs_decided'] += int(f_dec)
          if o_dec and f_dec:
            rows['both_decided'] += 1
            rows['labels_same' if np.sign(d_o) == np.sign(d_f) else 'labels_opposite'] += 1
          if o_dec and not f_dec:
            rows['orig_decided_fresh_tie_or_weak'] += 1
          for sname, sc in scores.items():
            fs = np.sign(sc[(layer, aid, ci)] - sc[(layer, aid, cj)])
            r = rows['critic'].setdefault(sname, {'vs_orig_same': 0, 'vs_orig_opp': 0, 'vs_fresh_same': 0, 'vs_fresh_opp': 0,
                                                  'vs_fresh_same_where_orig_agreed': 0, 'vs_fresh_opp_where_orig_agreed': 0})
            if o_dec:
              r['vs_orig_same' if fs == np.sign(d_o) else 'vs_orig_opp'] += 1
            if f_dec:
              r['vs_fresh_same' if fs == np.sign(d_f) else 'vs_fresh_opp'] += 1
              if o_dec and fs == np.sign(d_o):
                r['vs_fresh_same_where_orig_agreed' if fs == np.sign(d_f) else 'vs_fresh_opp_where_orig_agreed'] += 1
    po, pf = np.array(per_key_orig), np.array(per_key_fresh)
    if len(po) > 2:
      rows['per_key_spearman_orig_fresh'] = float(np.corrcoef(rankdata(po), rankdata(pf))[0, 1])
    rows['per_key_mean_abs_diff'] = float(np.abs(po - pf).mean()) if len(po) else None
    rows['per_key_n'] = int(len(po))
    for sname, r in rows['critic'].items():
      r['agreement_vs_orig'] = r['vs_orig_same'] / max(1, r['vs_orig_same'] + r['vs_orig_opp']) if r['vs_orig_same'] + r['vs_orig_opp'] else None
      r['agreement_vs_fresh'] = r['vs_fresh_same'] / max(1, r['vs_fresh_same'] + r['vs_fresh_opp']) if r['vs_fresh_same'] + r['vs_fresh_opp'] else None
    rows['labels_agreement'] = rows['labels_same'] / rows['both_decided'] if rows['both_decided'] else None
    out[stratum] = rows
  return out


def coverage(keys, man, layer, cands):
  info = {a['anchor_id']: a for a in man['anchors']}
  out = {}
  for stratum in list(STRATA) + ['pooled']:
    aids = sorted({aid for (ly, aid, c) in keys if ly == layer and (stratum == 'pooled' or keys[(ly, aid, c)]['set'] == stratum)})
    if not aids:
      continue
    ts = np.array([info[a]['t'] for a in aids]); goals = np.array([info[a]['goal_xy'] for a in aids])
    O = np.array([info[a]['obs'] for a in aids])[:, :B.STATE_DIM]
    spread, dmode, lp, sat = [], [], [], []
    for a in aids:
      A = np.array([keys[(layer, a, c)]['act0'] for c in cands if (layer, a, c) in keys])
      if len(A) == 0:
        continue
      if len(A) > 1:
        spread.append(np.mean([np.linalg.norm(A[i] - A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]))
      for c in cands:
        st = info[a].get('candidate_stats', {}).get(f'{layer}|{c}')   # manifests without the candidate statistics (the agent-round Cdev) skip them
        if st:
          dmode.append(st['dist_to_mode']); lp.append(st['log_prob_policy']); sat.append(st['frac_saturated'])
    out[stratum] = {'n_anchors': len(aids), 'n_episodes': len({info[a]['episode'] for a in aids}),
                    't_min_median_max': [int(ts.min()), float(np.median(ts)), int(ts.max())],
                    'goal_xy_mean': goals.mean(0).round(3).tolist(), 'goal_xy_std': goals.std(0).round(3).tolist(),
                    'state_pairwise_dist_mean': float(np.mean([np.linalg.norm(O[i] - O[j]) for i in range(len(O)) for j in range(i + 1, len(O))])) if len(O) > 1 else None,
                    'candidate_torque_spread': float(np.mean(spread)) if spread else None,
                    'candidate_dist_to_mode': float(np.mean(dmode)) if dmode else None,
                    'candidate_log_prob_policy': float(np.mean(lp)) if lp else None,
                    'candidate_frac_saturated': float(np.mean(sat)) if sat else None}
  return out


def nearest_state_distance(man):
  """How far the C states sit from the A states, against A's own nearest-neighbour distance (standardised state)."""
  A = np.array([a['obs'][:B.STATE_DIM] for a in man['anchors'] if a['layer'] == 'AB'])
  C = np.array([a['obs'][:B.STATE_DIM] for a in man['anchors'] if a['layer'] == 'C'])
  if len(A) < 2 or len(C) < 1:
    return {'A_to_nearest_A_median': None, 'C_to_nearest_A_median': None}
  mu, sd = A.mean(0), A.std(0) + 1e-6
  A_, C_ = (A - mu) / sd, (C - mu) / sd
  dAA = np.sqrt(((A_[:, None] - A_[None]) ** 2).sum(-1)); np.fill_diagonal(dAA, np.inf)
  dCA = np.sqrt(((C_[:, None] - A_[None]) ** 2).sum(-1))
  return {'A_to_nearest_A_median': float(np.median(dAA.min(1))), 'C_to_nearest_A_median': float(np.median(dCA.min(1)))}


def analyze(args):
  out = out_dir(args)
  man, keys, info = load_keys(out, smoke=args.smoke, extra=[B.OUT / e for e in args.extra_outcomes])
  critic_hashes = {}
  for c in args.critics:
    critic_hashes[str(c)] = sha256(c)
    if not args.allow_unsealed_critics:
      assert critic_hashes[str(c)] == man['provenance']['critics'][str(c)], f'{c} differs from the sealed checkpoint'
  from crl import checkpoint
  bundle = R.policy_bundle(args.cont_ckpt)
  nets = bundle['nets']
  # the goal marginal of the replay the critics trained on (weighted when that replay carries anchor weights)
  frames = R.marginal_goal_frames(B.OUT / (args.marginal_replay or args.replay), per_path=4, seed=0, weighted=True)
  a_cands = tuple(args.a_cands.split(','))
  ks = list(keys)
  o_all = np.stack([keys[k]['obs0'] for k in ks]); a_all = np.stack([keys[k]['act0'] for k in ks])
  scores = {}
  rng0 = np.random.default_rng(3)
  scores['random'] = dict(zip(ks, rng0.standard_normal(len(ks))))
  for c in args.critics:
    p = Path(c); label = f'{p.parent.parent.name}/{p.parent.name}'
    st = checkpoint.load_checkpoint(p)[1]
    rg = R.region_scorers(nets, st.q_params, frames, RADIUS, max_goals=512)(o_all, a_all)
    ex = R.exact_scorers(nets, st.q_params)(o_all, a_all)
    for h in ('min', 'h0', 'h1'):
      scores[f'{label} region {h}'] = dict(zip(ks, rg[h]))
      scores[f'{label} exact {h}'] = dict(zip(ks, ex[h]))
  rng = np.random.default_rng(11)
  strata = list(STRATA)
  res = {'layers': {}, 'orig_vs_fresh': {}, 'coverage': {}, 'nearest_state': nearest_state_distance(man)}
  res['layers']['A'] = layer_metrics(keys, 'A', set(a_cands), scores, strata, rng)
  res['layers']['A_samples_only'] = layer_metrics(keys, 'A', set(a_cands) - {'recorded'}, scores, strata, rng)
  res['layers']['B'] = layer_metrics(keys, 'B', None, scores, strata, rng)
  res['layers']['AB_cross'] = layer_metrics(keys, 'A', set(a_cands), scores, strata, rng, mixed_with='B')
  res['layers']['C'] = layer_metrics(keys, 'C', set(C_CANDS), scores, strata, rng)
  res['layers']['C_matched'] = layer_metrics(keys, 'C', set(C_MATCHED), scores, strata, rng)
  res['orig_vs_fresh']['A'] = original_vs_fresh(keys, info, 'A', tuple(c for c in a_cands if f'A|{c}' in man['anchors'][0]['original_outcomes']), strata, scores)
  res['orig_vs_fresh']['C'] = original_vs_fresh(keys, info, 'C', C_CANDS, strata, scores)
  res['coverage'] = {'A': coverage(keys, man, 'A', a_cands), 'B': coverage(keys, man, 'B', ('new0', 'new1', 'new2')), 'C': coverage(keys, man, 'C', C_CANDS)}
  res['critic_hashes'] = critic_hashes; res['a_cands'] = list(a_cands); res['marginal_replay'] = str(args.marginal_replay or args.replay)
  # outcome summary per layer / candidate
  res['outcomes'] = {}
  for (ly, aid, c), e in keys.items():
    r = res['outcomes'].setdefault(f'{ly}|{c}', {'n': 0, 'reach': 0.0, 'death': 0.0, 'p': 0.0, 'around': 0.0})
    n = len(e['p']); r['n'] += n
    r['reach'] += sum(e['success'].values()); r['death'] += sum(e['failure'].values()); r['p'] += sum(e['p'].values()); r['around'] += sum(e['around'].values())
  for r in res['outcomes'].values():
    for k in ('reach', 'death', 'p', 'around'):
      r[k] = r[k] / max(1, r['n'])
  suffix = f'_{args.tag}' if args.tag else ''
  (out / f'metrics{suffix}.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
  (out / f'REPORT{suffix}.md').write_text(report(res, man, args), encoding='utf-8')
  print((out / f'REPORT{suffix}.md').read_text(encoding='utf-8'), flush=True)


def _f(v, fmt='.2f'):
  return format(v, fmt) if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and np.isnan(v)) else '-'


def report(res, man, args):
  critics = sorted({s.split(' ')[0] for s in next(iter(res['layers'].values())) if s != 'random'})
  L = ['# Round-1 critic diagnostic: memorisation (A) vs new torques (B) vs new-episode states (C)', '',
       f'Sealed manifest: {man["sealed_at"]}; reference commit {man["provenance"].get("reference_commit", "see git")}.  Frozen: continuation '
       f'{Path(man["provenance"]["continuation_ckpt"]).parent.name} ({man["provenance"]["continuation_sha256"][:12]}), critics '
       + ', '.join(f'{Path(k).parent.name} ({v[:12]})' for k, v in (man['provenance'].get('critics') or res.get('critic_hashes', {})).items())
       + f'.  {N_DRAWS} fresh paired draws per anchor (seeds ABC_SEED + 100 i + r), one query torque then the frozen policy mode '
       f'closed-loop; P_goal = gamma {R.GAMMA}, radius {RADIUS}.  Primary readout = region-integrated exp(f) over the training NCE goal '
       f'marginal (radius 0.5, per head and deployed min); secondary = exact recorded-goal logit.  Pair classes on the 16-draw log ratio: '
       f'tie (= 0), weak (<= {THR}), decided (> {THR}); bootstrap-decided = decided and >= {BOOT_DECIDED:.0%} of draw-resamples keep the sign.  '
       'Agreement = among decided pairs (ties are never counted as wrong); s.e. = episode bootstrap.  Oracle-stage diagnostic; nothing was trained.', '',
       '## Anchors', '', '| layer | stratum | anchors | episodes | t min / median / max | goal xy mean (std) | candidate torque spread | dist to mode | log pi(a) | frac saturated |',
       '|---|---|---:|---:|---|---|---:|---:|---:|---:|']
  for ly in ('A', 'B', 'C'):
    for s, c in res['coverage'][ly].items():
      if c['t_min_median_max'] is None:
        continue
      L.append(f'| {ly} | {s} | {c["n_anchors"]} | {c["n_episodes"]} | {c["t_min_median_max"][0]} / {c["t_min_median_max"][1]:.0f} / {c["t_min_median_max"][2]} | '
               f'{c["goal_xy_mean"]} ({c["goal_xy_std"]}) | {_f(c["candidate_torque_spread"])} | {_f(c["candidate_dist_to_mode"])} | {_f(c["candidate_log_prob_policy"], ".1f")} | {_f(c["candidate_frac_saturated"])} |')
  ns = res['nearest_state']
  L += ['', f'Standardised-state distance: A anchor to its nearest other A anchor, median {_f(ns["A_to_nearest_A_median"])}; C anchor to its nearest A anchor, median {_f(ns["C_to_nearest_A_median"])}.  '
        'C differs from A in episode, and may differ in pose, goal and time distributions (see the table); B differs from A only in the torque.', '',
        '## Fresh outcomes by layer and candidate', '', '| key | paths | reach | death | P_goal | went around |', '|---|---:|---:|---:|---:|---:|']
  for k, r in sorted(res['outcomes'].items()):
    L.append(f'| {k} | {r["n"]} | {r["reach"]:.2f} | {r["death"]:.2f} | {r["p"]:.3f} | {r["around"]:.2f} |')
  L += ['', '## Labels: original records against fresh outcomes at identical keys', '',
        '| layer | stratum | keys | Spearman(orig mean P, fresh mean P) | mean abs diff | pairs decided by orig | by fresh | both | labels same / opposite | orig-decided but fresh tie/weak |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
  for ly in ('A', 'C'):
    for s, r in res['orig_vs_fresh'][ly].items():
      L.append(f'| {ly} | {s} | {r["per_key_n"]} | {_f(r.get("per_key_spearman_orig_fresh"))} | {_f(r["per_key_mean_abs_diff"], ".3f")} | {r["orig_pairs_decided"]} | '
               f'{r["fresh_pairs_decided"]} | {r["both_decided"]} | {r["labels_same"]} / {r["labels_opposite"]} | {r["orig_decided_fresh_tie_or_weak"]} |')
  L += ['', '## Layer A: the critic against the original labels and against fresh labels at the same keys (region min)', '',
        '| critic | stratum | vs original: same / opposite (agreement) | vs fresh: same / opposite (agreement) | where the critic matched the original label, fresh same / opposite |',
        '|---|---|---|---|---|']
  for ck in critics:
    for s, r in res['orig_vs_fresh']['A'].items():
      c = r['critic'].get(f'{ck} region min')
      if c:
        L.append(f'| {ck} | {s} | {c["vs_orig_same"]} / {c["vs_orig_opp"]} ({_f(c["agreement_vs_orig"])}) | {c["vs_fresh_same"]} / {c["vs_fresh_opp"]} ({_f(c["agreement_vs_fresh"])}) | '
                 f'{c["vs_fresh_same_where_orig_agreed"]} / {c["vs_fresh_opp_where_orig_agreed"]} |')
  for layer_name, title in (('A', 'Layer A: familiar state, familiar torque, fresh outcomes'), ('A_samples_only', 'Layer A, the two policy samples only'),
                            ('B', 'Layer B: familiar state, three new torques, fresh outcomes'), ('AB_cross', 'A x B cross pairs (shared anchors and draws)'),
                            ('C', 'Layer C: held-out-episode states, existing five candidates, fresh outcomes'), ('C_matched', 'Layer C, matched candidate types (recorded, sample0, sample1)')):
    L += ['', f'## {title}', '',
          '| scorer | stratum | anchors (episodes) | pairs tie / weak / decided (boot) | agreement decided (s.e.) same / opp | agreement boot-decided (s.e.) | pick gain (s.e.) | vs mode (s.e.) | P pick / mean / mode | cross-fit empirical selector gain (s.e.) |',
          '|---|---|---:|---|---|---|---:|---:|---|---:|']
    for sname, per in res['layers'][layer_name].items():
      if sname != 'random' and (' region min' not in sname and not args.all_readouts):
        continue
      for s, m in per.items():
        if s != 'pooled' and not args.per_stratum_all and layer_name in ('AB_cross', 'A_samples_only'):
          continue
        L.append(f'| {sname} | {s} | {m["n_anchors"]} ({m["n_episodes"]}) | {m["pairs"]["tie"]} / {m["pairs"]["weak"]} / {m["pairs"]["decided"]} ({m["pairs"]["boot_decided"]}) | '
                 f'**{_f(m["agreement_decided"])}** ({_f(m["se_decided"], ".3f")}) {m["same"]} / {m["opposite"]} | {_f(m["agreement_boot_decided"])} ({_f(m["se_boot_decided"], ".3f")}) | '
                 f'{_f(m["pick_gain"], "+.3f")} ({_f(m["pick_gain_se"], ".3f")}) | {_f(m["pick_gain_vs_mode"], "+.3f")} ({_f(m["pick_gain_vs_mode_se"], ".3f")}) | '
                 f'{_f(m["p_pick"], ".3f")} / {_f(m["p_mean"], ".3f")} / {_f(m["p_mode"], ".3f")} | {_f(m["cross_fit_selector_gain"], "+.3f")} ({_f(m["cross_fit_selector_gain_se"], ".3f")}) |')
  # all readouts, pooled only
  L += ['', '## All readouts, pooled (agreement among decided pairs; pick gain)', '',
        '| scorer | A | A samples-only | B | A x B | C | C matched |', '|---|---|---|---|---|---|---|']
  for sname in res['layers']['A']:
    cells = []
    for layer_name in ('A', 'A_samples_only', 'B', 'AB_cross', 'C', 'C_matched'):
      m = res['layers'][layer_name][sname]['pooled']
      cells.append(f'{_f(m["agreement_decided"])} ({m["n_decided"]}) / {_f(m["pick_gain"], "+.3f")}')
    L.append(f'| {sname} | ' + ' | '.join(cells) + ' |')
  L += ['', '## Verdicts (rules from the task; per layer, pooled dense strata, region min readout)', '']
  for ck in critics:
    parts = []
    for layer_name in ('A', 'B', 'C'):
      m = res['layers'][layer_name][f'{ck} region min']['pooled']
      ag, se, n = m['agreement_decided'], m['se_decided'], m['n_decided']
      if ag is None or n < 20 or se is None:
        v = 'inconclusive (too few decided pairs)'
      elif ag - 2 * se > 0.5 and m['pick_gain'] is not None and m['pick_gain'] > 0:
        v = 'succeeds (above chance, positive pick gain)'
      elif ag + 2 * se < 0.6:
        v = 'fails (interval below 0.6)'
      else:
        v = 'inconclusive (interval spans chance and useful agreement)'
      parts.append(f'{layer_name}: {v} [{_f(ag)} +- {_f(se, ".3f")}, n {n}, gain {_f(m["pick_gain"], "+.3f")}]')
    L.append(f'- {ck}: ' + '; '.join(parts))
  cf = res['layers']['A']['random']['pooled']['cross_fit_selector_gain']
  L.append(f'- usable selection signal in the fresh outcomes themselves (cross-fitted empirical selector, layer A pooled): {_f(cf, "+.3f")}; '
           f'B: {_f(res["layers"]["B"]["random"]["pooled"]["cross_fit_selector_gain"], "+.3f")}; C: {_f(res["layers"]["C"]["random"]["pooled"]["cross_fit_selector_gain"], "+.3f")} '
           '(a selector that saw eight draws, evaluated on the other eight; the round-1 gate status is unchanged).')
  return '\n'.join(L) + '\n'


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'run', 'analyze'))
  ap.add_argument('--cont-ckpt', required=True)
  ap.add_argument('--critics', nargs='*', default=[])
  ap.add_argument('--replay', default='replay_policy_r1.npz')
  ap.add_argument('--holdout', default='holdout_policy_r1.npz')
  ap.add_argument('--out', default='diag_r1_abc')
  ap.add_argument('--ref-commit', default='69e6572')
  ap.add_argument('--seed', type=int, default=2026)
  ap.add_argument('--n-per-stratum', type=int, default=N_PER_STRATUM)
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
  ap.add_argument('--limit', type=int, default=0, help='run: smoke with the first N paths')
  ap.add_argument('--smoke', action='store_true', help='analyze: read outcomes_smoke.npz')
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--all-readouts', action='store_true', help='report: every readout in the per-layer tables')
  ap.add_argument('--per-stratum-all', action='store_true')
  ap.add_argument('--tag', default='', help='analyze: suffix for REPORT_/metrics_ (keeps the r1 files)')
  ap.add_argument('--allow-unsealed-critics', action='store_true', help='analyze: critics not in the sealed manifest (their hashes are recorded in the metrics)')
  ap.add_argument('--marginal-replay', default='', help='analyze: replay for the goal marginal (default --replay); weighted if it carries audit_weight')
  ap.add_argument('--a-cands', default='recorded,sample0,sample1', help='analyze: the layer-A candidate labels (training torques of the critics under test)')
  ap.add_argument('--extra-outcomes', nargs='*', default=[], help='analyze: extra outcome tables under V6_BRANCH_OUT to merge')
  args = ap.parse_args(argv)
  {'seal': seal, 'run': run, 'analyze': analyze}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
