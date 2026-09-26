"""Diagnostic control: can DIRECT supervised prediction of the rollout target
P_goal(s, a, g) generalise where the round-1 NCE critic does not?

Not a method change: the CRL implementation, its checkpoints and the actor
stage are untouched.  Two small scalar MLPs are fitted by squared error to
the R-arm replay's own per-path targets (audit_p_goal: the relabeling law's
goal-frame probability from row 0 -- gamma 0.999 over the path's rows 1..L-1,
normalised over the path's own length, XY within radius 0.5 of the anchor
episode's recorded goal, the path ending at death or the horizon 800 - t,
zero-torque hold after reaching) and read with the SAME A / B / C
diagnostic (`scripts/diag_v6_r1_abc.py`, its sealed fresh outcomes):

  S  h(s, a, g): the critic's inputs exactly (29-d state, 2-d goal, 8-d torque)
  N  h0(s, g):   the same without the torque -- every candidate at a state ties

Inputs contain no absolute time, hazard latents, clocks, episode / anchor
ids, route labels or candidate identities (neither does the NCE critic).
Absolute time affects the target (hazard clocks run from the reset; the
remaining horizon sets the normalisation) without being an input -- for
the NCE critic and for these predictors alike.

Training keys = the R replay's exact (anchor, candidate) keys with their
paths' weights summed and their targets weight-averaged: for squared error,
sum_i w_i (h(x) - y_i)^2 = (sum_i w_i) (h(x) - ybar_w)^2 + const, so the
aggregated objective is the path-level one; keys are drawn per batch with
probability proportional to the summed weight (the R replay's anchor law:
round-1 paths 1, new policy paths 0.25 -> state mass and recorded / policy
mixture exactly round 1's).  The batch index draws depend only on the seed,
so S and N see identical key draws.  Inputs standardised with training-key
statistics only.  Sigmoid output (the target lies in [0, 1]); Adam, lr 3e-4,
batch 1024, 30,000 updates, seeds 0 / 1 / 2, final checkpoint only.

  python scripts/diag_v6_supervised_pgoal.py seal --replay exp_r1_coverage/replay_armR.npz
  python scripts/diag_v6_supervised_pgoal.py train --model S --seed 0   (x 6)
  python scripts/diag_v6_supervised_pgoal.py analyze --cont-ckpt <pure-BC> --critics <R critics>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402
import diag_v6_r1_abc as ABC  # noqa: E402

OUT_DIR = 'diag_supervised_pgoal'
HIDDEN = (1024, 1024)
LR, BATCH, STEPS, SEEDS = 3e-4, 1024, 30_000, (0, 1, 2)
LOG_EVERY, FULL_EVERY = 100, 2500
A_CANDS = ('recorded', 'sample0', 'sample1')      # R's familiar torques at the A anchors (not COV's)


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def out_dir():
  return B.OUT / OUT_DIR


# -------------------------------------------------------------------- data
def training_keys(replay_path):
  """Exact (anchor, candidate) keys of the replay with summed weights and
  weight-averaged targets; the key's obs / act are identical across its paths."""
  with np.load(replay_path, allow_pickle=False) as d:
    o0 = d['obs'][:, 0, :B.STATE_DIM + 2].astype(np.float32); a0 = d['act'][:, 0].astype(np.float32)
    w = d['audit_weight'].astype(np.float64); p = d['audit_p_goal'].astype(np.float64)
    aid = d['audit_anchor_id'].astype(int); cand = d['audit_cand'].astype(str); kind = d['audit_kind'].astype(str)
    ep = d['audit_episode'].astype(int); n_paths = len(w)
  keys = {}
  for i in range(n_paths):
    k = (aid[i], cand[i])
    e = keys.setdefault(k, {'obs': o0[i], 'act': a0[i], 'w': 0.0, 'wp': 0.0, 'n': 0, 'kind': kind[i], 'episode': ep[i], 'p': []})
    assert np.array_equal(e['obs'], o0[i]) and np.array_equal(e['act'], a0[i]), 'key rows differ'
    e['w'] += w[i]; e['wp'] += w[i] * p[i]; e['n'] += 1; e['p'].append(p[i])
  ks = list(keys)
  X_obs = np.stack([keys[k]['obs'] for k in ks]); X_act = np.stack([keys[k]['act'] for k in ks])
  W = np.array([keys[k]['w'] for k in ks]); Y = np.array([keys[k]['wp'] / keys[k]['w'] for k in ks])
  meta = {'n_paths': int(n_paths), 'n_keys': len(ks), 'total_weight': float(W.sum()),
          'keys_by_kind_cand': {}, 'paths_per_key': {}}
  for k in ks:
    kk = f'{keys[k]["kind"]}|{k[1]}'
    meta['keys_by_kind_cand'][kk] = meta['keys_by_kind_cand'].get(kk, 0) + 1
    meta['paths_per_key'][str(keys[k]['n'])] = meta['paths_per_key'].get(str(keys[k]['n']), 0) + 1
  kinds = np.array([keys[k]['kind'] for k in ks]); cands = np.array([k[1] for k in ks]); eps = np.array([keys[k]['episode'] for k in ks])
  aids = np.array([k[0] for k in ks])
  return {'obs': X_obs, 'act': X_act, 'w': W, 'y': Y, 'kind': kinds, 'cand': cands, 'episode': eps, 'anchor_id': aids, 'meta': meta}


def git_info():
  try:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    st = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(ROOT), text=True).splitlines()
    return {'head': head, 'dirty': [l for l in st if not l.startswith('??')], 'untracked_count': sum(l.startswith('??') for l in st)}
  except Exception as e:  # noqa
    return {'error': str(e)}


# -------------------------------------------------------------------- seal
def seal(args):
  out = out_dir(); out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed)')
  replay = B.OUT / args.replay
  D = training_keys(replay)
  abc = B.OUT / ABC_DIR()
  abc_man = json.loads((abc / 'manifest.json').read_text(encoding='utf-8'))
  # leakage checks: no C episode among training episodes; no B torque among training torques at the A anchors
  c_eps = {x['episode'] for x in abc_man['anchors'] if x['layer'] == 'C'}
  train_eps = set(D['episode'].tolist())
  assert not (c_eps & train_eps), 'held-out C episodes appear in the training keys'
  b_min = np.inf
  for x in abc_man['anchors']:
    if x['layer'] != 'AB':
      continue
    m = D['anchor_id'] == x['anchor_id']
    if m.any():
      T = D['act'][m]
      for c, v in x['candidates'].items():
        if c.startswith('B|'):
          b_min = min(b_min, float(np.linalg.norm(T - np.asarray(v, np.float32)[None], axis=1).min()))
  kinds = D['kind']; cands = D['cand']; W = D['w']
  weights = {'state_mass_dense': float(W[kinds != 'general'].sum() / W.sum()), 'state_mass_general': float(W[kinds == 'general'].sum() / W.sum()),
             'mixture_recorded': float(W[cands == 'recorded'].sum() / W.sum()), 'mixture_policy': float(W[cands != 'recorded'].sum() / W.sum()),
             'within_dense_recorded': float(W[(kinds != 'general') & (cands == 'recorded')].sum() / W[kinds != 'general'].sum()),
             'round1_reference': {'state_mass_dense': 9000 / 17756, 'mixture_recorded': 7378 / 17756, 'within_dense_recorded': 1 / 3}}
  mu_o, sd_o = D['obs'].mean(0), D['obs'].std(0) + 1e-6; mu_a, sd_a = D['act'].mean(0), D['act'].std(0) + 1e-6
  man = {'diagnostic': 'direct supervised prediction of P_goal (S: s,a,g; N: s,g) on the R-arm replay, read with the A/B/C diagnostic',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git': git_info(),
         'provenance': {'replay': str(replay), 'replay_sha256': sha256(replay), 'abc_manifest_sha256': sha256(abc / 'manifest.json'),
                        'abc_outcomes_sha256': sha256(abc / 'outcomes.npz'), 'critics': {str(c): sha256(c) for c in args.critics},
                        'continuation_ckpt': str(args.cont_ckpt), 'continuation_sha256': sha256(args.cont_ckpt) if args.cont_ckpt else None,
                        'scripts': {f: sha256(ROOT / f) for f in ('scripts/diag_v6_supervised_pgoal.py', 'scripts/diag_v6_r1_abc.py', 'scripts/build_v6_policy_replay.py')}},
         'target': {'name': 'audit_p_goal', 'definition': 'build_v6_branch_replay.goal_frame_probability from row 0: sum_{j=1}^{L-1} gamma^j 1[|xy_j - goal| <= 0.5] / sum_{j=1}^{L-1} gamma^j',
                    'gamma': 0.999, 'radius': 0.5, 'path_end': 'env done (death) or horizon 800 - t; zero-torque hold after reaching (parked frames count as near)',
                    'goal': 'the anchor episode\'s recorded goal xy = the continuation\'s task goal = the critic\'s goal input',
                    'not_in_inputs': 'absolute time t (hazard clocks from reset, remaining horizon), hazard latents, jitter -- for NCE and for S/N alike'},
         'inputs': {'S': 'obs[:31] (29-d Ant state + 2-d goal xy) and the 8-d torque; standardised with training-key mean/std',
                    'N': 'obs[:31] only, same standardisation', 'standardisation': {'obs_mean': mu_o.tolist(), 'obs_std': sd_o.tolist(), 'act_mean': mu_a.tolist(), 'act_std': sd_a.tolist()},
                    'excluded': 'time, hazards, clocks, episode/anchor ids, route labels, candidate identities'},
         'training': {'keys': D['meta'], 'aggregation': 'exact (anchor, candidate) key; weight = sum of path weights; target = weight-averaged P_goal; '
                                                        'squared-error equivalence: sum_i w_i (h - y_i)^2 = (sum w_i)(h - ybar_w)^2 + const',
                      'effective_weights': weights, 'sampler': 'keys drawn per batch with probability proportional to weight; numpy default_rng(seed) -> identical draws for S and N',
                      'model': {'hidden': list(HIDDEN), 'activation': 'relu', 'init': 'VarianceScaling(1.0, fan_avg, uniform) as the critic encoders', 'output': 'Linear(1) -> sigmoid'},
                      'loss': 'mean squared error on P_goal (conditional mean; diagnostic only)', 'optimizer': 'adam lr 3e-4 eps 1e-7', 'batch': BATCH, 'updates': STEPS,
                      'seeds': list(SEEDS), 'checkpoint': 'final only (no selection on A/B/C)', 'device': 'one GPU, all six runs', 'sweeps': 'none'},
         'evaluation': {'sets': 'diag_r1_abc sealed fresh outcomes (development diagnostics, already inspected); reserved validation seeds unused',
                        'layer_A_candidates': list(A_CANDS), 'layer_B': 'new0-2', 'layer_C': 'recorded, mode, sample0-2 (+ matched subset)',
                        'comparators': 'R NCE critics (region-integrated readout, radius 0.5, weighted R marginal, deployed min); S; N; random',
                        'metrics': 'training-key weighted MSE; fresh-outcome MSE vs 16-draw sample mean with the sampling-noise floor; agreement on the same decided pairs '
                                   '(tie / same / opposite reported; ties never counted as wrong); pick gain as in the diagnostic with uniform random tie-breaking; '
                                   'per seed; paired episode-bootstrap differences S - NCE, S - N; per stratum',
                        'predeclared': {'positive_transfer': 'positive B and C pick gain AND improvement over R NCE supported by paired uncertainty (> 2 s.e.) with consistent per-seed direction',
                                        'B_alone': 'B improvement alone does not establish cross-episode transfer', 'nonsignificant': 'inconclusive, not equivalence',
                                        'fit_failure': 'if S does not fit its training targets, reported separately, not read as a generalisation failure'}},
         'leakage_checks': {'C_episodes_in_training': 0, 'min_dist_training_torque_to_B_torque_same_anchor': b_min}}
  (out / 'manifest.json').write_text(json.dumps(man, indent=1, default=float), encoding='utf-8')
  np.savez_compressed(out / 'training_keys.npz', obs=D['obs'], act=D['act'], w=D['w'], y=D['y'], kind=D['kind'], cand=D['cand'], episode=D['episode'], anchor_id=D['anchor_id'])
  print(json.dumps({'keys': D['meta'], 'weights': weights, 'leakage': man['leakage_checks']}, indent=1), flush=True)


def ABC_DIR():
  return 'diag_r1_abc'


# ------------------------------------------------------------------- model
def build_model(with_action):
  import haiku as hk
  import jax

  def fn(x):
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    h = hk.nets.MLP(list(HIDDEN), w_init=init, activation=jax.nn.relu, activate_final=True)(x)
    return jax.nn.sigmoid(hk.Linear(1, w_init=init)(h)[:, 0])
  return hk.without_apply_rng(hk.transform(fn))


def features(man, obs, act, with_action):
  st = man['inputs']['standardisation']
  xo = (obs[:, :B.STATE_DIM + 2] - np.asarray(st['obs_mean'], np.float32)) / np.asarray(st['obs_std'], np.float32)
  if not with_action:
    return xo.astype(np.float32)
  xa = (act - np.asarray(st['act_mean'], np.float32)) / np.asarray(st['act_std'], np.float32)
  return np.concatenate([xo, xa], 1).astype(np.float32)


def train(args):
  import jax
  import jax.numpy as jnp
  import optax
  out = out_dir(); man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  with_action = args.model == 'S'
  d = out / f'{args.model}_seed{args.seed}'
  if (d / 'final.pkl').exists() and not args.force:
    raise SystemExit(f'{d} exists')
  d.mkdir(parents=True, exist_ok=True)
  with np.load(out / 'training_keys.npz', allow_pickle=False) as k:
    X = features(man, k['obs'], k['act'], with_action); Y = k['y'].astype(np.float32); W = k['w'].astype(np.float64)
  n = len(Y); pw = W / W.sum()
  model = build_model(with_action)
  key = jax.random.PRNGKey(1000 + args.seed)
  params = model.init(key, jnp.zeros((1, X.shape[1]), jnp.float32))
  opt = optax.adam(LR, eps=1e-7); opt_state = opt.init(params)

  def loss_fn(p, x, y):
    return jnp.mean((model.apply(p, x) - y) ** 2)

  @jax.jit
  def step(p, s, x, y):
    l, g = jax.value_and_grad(loss_fn)(p, x, y)
    upd, s = opt.update(g, s, p)
    return optax.apply_updates(p, upd), s, l

  @jax.jit
  def predict(p, x):
    return model.apply(p, x)

  def full_mse(p):
    pr = np.concatenate([np.asarray(predict(p, jnp.asarray(X[i:i + 8192]))) for i in range(0, n, 8192)])
    return float(np.sum(W * (pr - Y) ** 2) / W.sum()), pr
  rng = np.random.default_rng(args.seed)           # the key draws: seed only -> identical for S and N
  curve = []; t0 = time.time()
  const_mse = float(np.sum(W * (Y - np.sum(pw * Y)) ** 2) / W.sum())
  for it in range(1, STEPS + 1):
    idx = rng.choice(n, size=BATCH, replace=True, p=pw)
    params, opt_state, l = step(params, opt_state, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
    if it % LOG_EVERY == 0 or it == 1:
      rec = {'step': it, 'batch_mse': float(l)}
      if it % FULL_EVERY == 0 or it == STEPS:
        rec['weighted_mse_all_keys'] = full_mse(params)[0]
      curve.append(rec)
  mse, pred = full_mse(params)
  with (d / 'final.pkl').open('wb') as f:
    pickle.dump({'params': jax.tree_util.tree_map(np.asarray, params), 'model': args.model, 'seed': args.seed, 'with_action': with_action}, f)
  np.save(d / 'train_predictions.npy', pred)
  summary = {'model': args.model, 'seed': args.seed, 'updates': STEPS, 'wall_seconds': time.time() - t0, 'weighted_mse_all_keys': mse,
             'constant_predictor_mse': const_mse, 'r2_weighted': (1 - mse / const_mse) if const_mse > 0 else None, 'curve': curve,
             'device': str(jax.devices()[0])}
  (d / 'train_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  print(json.dumps({k: v for k, v in summary.items() if k != 'curve'}, indent=1), flush=True)


# ----------------------------------------------------------------- analyze
def load_predictor(path):
  import jax.numpy as jnp
  import jax
  with Path(path).open('rb') as f:
    ck = pickle.load(f)
  model = build_model(ck['with_action']); params = ck['params']

  @jax.jit
  def predict(x):
    return model.apply(params, x)
  return (lambda X: np.concatenate([np.asarray(predict(jnp.asarray(X[i:i + 8192]))) for i in range(0, len(X), 8192)])), ck['with_action']


def pair_eval(keys, layer, cands, scores, strata, rng, tie_breaks=200):
  """Tie-aware version of the diagnostic's pair metrics: for each scorer, among
  the decided pairs (same rule as diag_v6_r1_abc.classify_pair) count same /
  opposite / tie; agreement = same / (same + opposite); pick gain = P_goal of
  the scorer's argmax candidate (uniform random tie-breaking, averaged over
  ``tie_breaks`` draws) minus the candidate mean; episode-bootstrap s.e.;
  per-pair correctness kept for paired comparisons."""
  anchors = {}
  for (ly, aid, c), e in keys.items():
    if ly == layer and c in cands:
      anchors.setdefault(aid, {})[c] = e
  pairs = []
  for aid, cs in anchors.items():
    names = sorted(cs, key=R._cand_order)
    for i in range(len(names)):
      for j in range(i + 1, len(names)):
        pi, pj = cs[names[i]]['p'], cs[names[j]]['p']
        dd = sorted(set(pi) & set(pj))
        cls, d, boot = ABC.classify_pair(pi, pj, dd, rng)
        if cls == 'decided':
          pairs.append((aid, names[i], names[j], np.sign(d), cs[names[i]]['episode'], cs[names[i]]['set']))
  out = {'n_pairs_decided': len(pairs), 'n_anchors': len(anchors), 'n_episodes': len({e['episode'] for cs in anchors.values() for e in cs.values()}),
         'scorers': {}, 'per_pair': {}}
  trng = np.random.default_rng(77)
  for sname, sc in scores.items():
    corr = np.zeros(len(pairs)); tie = np.zeros(len(pairs), bool)
    for q, (aid, ci, cj, sg, ep, st) in enumerate(pairs):
      fs = np.sign(sc[(layer, aid, ci)] - sc[(layer, aid, cj)])
      tie[q] = fs == 0; corr[q] = float(fs == sg)
    # picks with uniform random tie-breaking
    gains, gains_mode = {}, {}
    for aid, cs in anchors.items():
      names = sorted(cs, key=R._cand_order)
      Pm = np.array([np.mean(list(cs[c]['p'].values())) for c in names]); F = np.array([sc[(layer, aid, c)] for c in names])
      g = []
      for _ in range(tie_breaks):
        k = int(np.argmax(F + 1e-9 * trng.standard_normal(len(F))))
        g.append(Pm[k] - Pm.mean())
      gains[aid] = float(np.mean(g))
      if 'mode' in names:
        gains_mode[aid] = float(np.mean([Pm[int(np.argmax(F + 1e-9 * trng.standard_normal(len(F))))] - Pm[names.index('mode')] for _ in range(tie_breaks)]))
    res = {}
    for stratum in list(strata) + ['pooled']:
      m = np.array([stratum == 'pooled' or p[5] == stratum for p in pairs], dtype=bool); eps = np.array([p[4] for p in pairs], dtype=int)
      dec = m & ~tie; n_same = int(corr[dec].sum()); n_opp = int(dec.sum() - n_same); n_tie = int((m & tie).sum())
      ag = n_same / max(1, n_same + n_opp) if n_same + n_opp else None
      # episode bootstrap of the agreement (non-tied pairs) and of the pick gain
      ueps = np.unique(eps[m]); se_ag, se_g = None, None
      an_in = [a for a, cs in anchors.items() if stratum == 'pooled' or next(iter(cs.values()))['set'] == stratum]
      g_vals = np.array([gains[a] for a in an_in]); g_eps = np.array([next(iter(anchors[a].values()))['episode'] for a in an_in])
      if len(ueps) > 1:
        brng = np.random.default_rng(5); ba, bg = [], []
        for _ in range(1000):
          pick = brng.choice(ueps, size=len(ueps), replace=True)
          sel = np.concatenate([np.flatnonzero((eps == e) & dec) for e in pick]) if dec.any() else np.array([], int)
          ba.append(corr[sel].mean() if len(sel) else np.nan)
          selg = np.concatenate([np.flatnonzero(g_eps == e) for e in pick])
          bg.append(g_vals[selg].mean() if len(selg) else np.nan)
        se_ag, se_g = float(np.nanstd(ba)), float(np.nanstd(bg))
      res[stratum] = {'n_decided': int(m.sum()), 'same': n_same, 'opposite': n_opp, 'tie': n_tie, 'agreement': ag, 'se': se_ag,
                      'n_anchors': len(an_in), 'n_episodes': int(len(np.unique(g_eps))) if len(g_eps) else 0,
                      'pick_gain': float(g_vals.mean()) if len(g_vals) else None, 'pick_gain_se': se_g,
                      'pick_gain_vs_mode': float(np.mean([gains_mode[a] for a in an_in if a in gains_mode])) if any(a in gains_mode for a in an_in) else None}
    out['scorers'][sname] = res
    out['per_pair'][sname] = {'correct': corr, 'tie': tie}
  out['pair_episodes'] = np.array([p[4] for p in pairs], dtype=int); out['pair_strata'] = np.array([p[5] for p in pairs])
  return out


def paired_diff(ev, a_names, b_names):
  """Seed-averaged paired episode bootstrap of agreement(a) - agreement(b) over
  the pairs where neither scorer ties (ties excluded from both)."""
  eps = ev['pair_episodes']; ueps = np.unique(eps)
  ca = np.mean([ev['per_pair'][n]['correct'] for n in a_names], axis=0); cb = np.mean([ev['per_pair'][n]['correct'] for n in b_names], axis=0)
  ta = np.any([ev['per_pair'][n]['tie'] for n in a_names], axis=0); tb = np.any([ev['per_pair'][n]['tie'] for n in b_names], axis=0)
  ok = ~(ta | tb)
  if ok.sum() == 0:
    return {'n_pairs': 0}
  brng = np.random.default_rng(9); bs = []
  for _ in range(2000):
    pick = brng.choice(ueps, size=len(ueps), replace=True)
    sel = np.concatenate([np.flatnonzero((eps == e) & ok) for e in pick])
    bs.append((ca[sel] - cb[sel]).mean() if len(sel) else np.nan)
  d = float((ca[ok] - cb[ok]).mean()); se = float(np.nanstd(bs))
  return {'n_pairs': int(ok.sum()), 'diff': d, 'se': se, 'z': d / max(se, 1e-9)}


def analyze(args):
  out = out_dir(); man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  abc = B.OUT / ABC_DIR()
  keysd = ABC.load_keys(abc)
  abc_man, keys, info = keysd
  ks = list(keys)
  o_all = np.stack([keys[k]['obs0'] for k in ks]); a_all = np.stack([keys[k]['act0'] for k in ks])
  scores = {}
  rng0 = np.random.default_rng(3); scores['random'] = dict(zip(ks, rng0.standard_normal(len(ks))))
  # NCE critics (R), region readout with the R replay's weighted marginal
  from crl import checkpoint
  bundle = R.policy_bundle(args.cont_ckpt); nets = bundle['nets']
  frames = R.marginal_goal_frames(Path(man['provenance']['replay']), per_path=4, seed=0, weighted=True)
  for c in args.critics:
    assert sha256(c) == man['provenance']['critics'][str(c)], f'{c} differs from the sealed critic'
    st = checkpoint.load_checkpoint(c)[1]
    rg = R.region_scorers(nets, st.q_params, frames, ABC.RADIUS, max_goals=512)(o_all, a_all)
    scores[f'NCE_R/{Path(c).parent.name}'] = dict(zip(ks, rg['min']))
  # S and N predictors
  preds = {}
  for model in ('S', 'N'):
    for s in SEEDS:
      fn, wa = load_predictor(out / f'{model}_seed{s}' / 'final.pkl')
      pr = fn(features(man, o_all, a_all, wa))
      scores[f'{model}/seed_{s}'] = dict(zip(ks, pr)); preds[f'{model}/seed_{s}'] = pr
  # prediction error on fresh outcomes (S, N) with the sampling-noise floor of a 16-draw mean
  P16 = np.array([np.mean(list(keys[k]['p'].values())) for k in ks]); V16 = np.array([np.var(list(keys[k]['p'].values()), ddof=1) / len(keys[k]['p']) for k in ks])
  layer_of = np.array([k[0] for k in ks]); cand_of = np.array([k[2] for k in ks])
  mse = {}
  for name, pr in preds.items():
    mse[name] = {}
    for ly, cands in (('A', A_CANDS), ('B', ('new0', 'new1', 'new2')), ('C', ABC.C_CANDS)):
      m = (layer_of == ly) & np.isin(cand_of, cands)
      if m.sum() < 2:
        mse[name][ly] = {'n_keys': int(m.sum()), 'mse_vs_16draw_mean': None, 'noise_floor_var_of_mean': None, 'const_predictor_mse': None, 'corr_pred_vs_mean': None}
        continue
      mse[name][ly] = {'mse_vs_16draw_mean': float(np.mean((pr[m] - P16[m]) ** 2)), 'noise_floor_var_of_mean': float(np.mean(V16[m])),
                       'const_predictor_mse': float(np.mean((P16[m] - P16[m].mean()) ** 2)), 'n_keys': int(m.sum()),
                       'corr_pred_vs_mean': float(np.corrcoef(pr[m], P16[m])[0, 1])}
  # training fit
  fit = {f'{model}/seed_{s}': json.loads((out / f'{model}_seed{s}' / 'train_summary.json').read_text(encoding='utf-8')) for model in ('S', 'N') for s in SEEDS}
  fit = {k: {kk: v[kk] for kk in ('weighted_mse_all_keys', 'constant_predictor_mse', 'r2_weighted', 'wall_seconds', 'device')} for k, v in fit.items()}
  # per-key fresh-vs-training at A (the training keys' own fresh 16 draws)
  strata = list(R.DENSE_SETS)
  rng = np.random.default_rng(11)
  ev = {'A': pair_eval(keys, 'A', set(A_CANDS), scores, strata, rng), 'B': pair_eval(keys, 'B', {'new0', 'new1', 'new2'}, scores, strata, rng),
        'C': pair_eval(keys, 'C', set(ABC.C_CANDS), scores, strata, rng), 'C_matched': pair_eval(keys, 'C', set(ABC.C_MATCHED), scores, strata, rng)}
  nce = [f'NCE_R/seed_{s}' for s in SEEDS]; S_ = [f'S/seed_{s}' for s in SEEDS]; N_ = [f'N/seed_{s}' for s in SEEDS]
  diffs = {}
  for ly, e in ev.items():
    diffs[ly] = {'S - NCE_R': paired_diff(e, S_, nce), 'S - N': paired_diff(e, S_, N_), 'NCE_R - N': paired_diff(e, nce, N_),
                 'per_seed S - NCE_R': {s: paired_diff(e, [f'S/seed_{s}'], [f'NCE_R/seed_{s}']) for s in SEEDS}}
  res = {'fit': fit, 'fresh_mse': mse, 'layers': {ly: {'n_pairs_decided': e['n_pairs_decided'], 'n_anchors': e['n_anchors'], 'n_episodes': e['n_episodes'], 'scorers': e['scorers']} for ly, e in ev.items()},
         'paired': diffs, 'critic_hashes': {str(c): sha256(c) for c in args.critics}}
  (out / 'metrics.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
  np.savez_compressed(out / 'eval_predictions.npz', layer=layer_of, cand=cand_of, anchor_id=np.array([k[1] for k in ks]), p16=P16,
                      **{name.replace('/', '_'): pr for name, pr in preds.items()},
                      **{name.replace('/', '_'): np.array([scores[name][k] for k in ks]) for name in scores if name.startswith('NCE')})
  (out / 'REPORT.md').write_text(report(res, man), encoding='utf-8')
  print((out / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


def _f(v, fmt='.2f'):
  return format(v, fmt) if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and np.isnan(v)) else '-'


def report(res, man):
  L = ['# Direct supervised P_goal predictors (S: s,a,g; N: s,g) vs the R NCE critics on the A / B / C diagnostic', '',
       f'Sealed {man["sealed_at"]}; git {man["git"].get("head", "?")[:10]} ({len(man["git"].get("dirty", []))} dirty tracked files).  Training keys: {man["training"]["keys"]["n_keys"]} exact '
       f'(anchor, candidate) keys from {man["training"]["keys"]["n_paths"]} R-replay paths (state mass dense {man["training"]["effective_weights"]["state_mass_dense"]:.4f}, recorded mixture '
       f'{man["training"]["effective_weights"]["mixture_recorded"]:.4f}, within-dense recorded {man["training"]["effective_weights"]["within_dense_recorded"]:.4f}).  '
       f'{HIDDEN} relu MLP, sigmoid output, squared error, Adam {LR}, batch {BATCH}, {STEPS:,} updates, seeds {list(SEEDS)}, final checkpoint only; S and N share the key draws.  '
       'Evaluation: the sealed diag_r1_abc fresh outcomes (development sets); decided pairs as in the diagnostic; ties reported, never counted as wrong; pick gain with uniform '
       'random tie-breaking; s.e. = episode bootstrap.  No actor trained; the NCE checkpoints untouched.', '',
       '## 1. Training fit (weighted MSE over the training keys, all keys)', '', '| run | weighted MSE | constant-predictor MSE | weighted R^2 | wall s | device |', '|---|---:|---:|---:|---:|---|']
  for k, v in res['fit'].items():
    L.append(f'| {k} | {v["weighted_mse_all_keys"]:.4f} | {v["constant_predictor_mse"]:.4f} | {_f(v["r2_weighted"], ".3f")} | {v["wall_seconds"]:.0f} | {v["device"]} |')
  L += ['', '## 2. Fresh-outcome prediction error (MSE against the 16-draw sample mean; noise floor = mean sampling variance of that mean)', '',
        '| run | layer | keys | MSE | noise floor | const MSE | corr(pred, mean) |', '|---|---|---:|---:|---:|---:|---:|']
  for name, per in res['fresh_mse'].items():
    for ly, v in per.items():
      L.append(f'| {name} | {ly} | {v["n_keys"]} | {_f(v["mse_vs_16draw_mean"], ".4f")} | {_f(v["noise_floor_var_of_mean"], ".4f")} | {_f(v["const_predictor_mse"], ".4f")} | {_f(v["corr_pred_vs_mean"])} |')
  for ly, e in res['layers'].items():
    L += ['', f'## Layer {ly}: {e["n_pairs_decided"]} decided pairs, {e["n_anchors"]} anchors, {e["n_episodes"]} episodes', '',
          '| scorer | stratum | same / opposite / tie | agreement (s.e.) | pick gain (s.e.) | vs mode |', '|---|---|---|---|---|---|']
    for sname, per in e['scorers'].items():
      for st, m in per.items():
        if st != 'pooled' and sname == 'random':
          continue
        L.append(f'| {sname} | {st} | {m["same"]} / {m["opposite"]} / {m["tie"]} | **{_f(m["agreement"])}** ({_f(m["se"], ".3f")}) | {_f(m["pick_gain"], "+.4f")} ({_f(m["pick_gain_se"], ".4f")}) | {_f(m["pick_gain_vs_mode"], "+.4f")} |')
  L += ['', '## Paired differences (seed-averaged agreement, episode bootstrap, pairs where neither scorer ties)', '', '| layer | comparison | pairs | diff | s.e. | z |', '|---|---|---:|---:|---:|---:|']
  for ly, d in res['paired'].items():
    for name, v in d.items():
      if name.startswith('per_seed'):
        for s, vv in v.items():
          L.append(f'| {ly} | S - NCE_R seed {s} | {vv.get("n_pairs", 0)} | {_f(vv.get("diff"), "+.3f")} | {_f(vv.get("se"), ".3f")} | {_f(vv.get("z"), ".2f")} |')
      else:
        L.append(f'| {ly} | {name} | {v.get("n_pairs", 0)} | {_f(v.get("diff"), "+.3f")} | {_f(v.get("se"), ".3f")} | {_f(v.get("z"), ".2f")} |')
  return '\n'.join(L) + '\n'


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'train', 'analyze'))
  ap.add_argument('--replay', default='exp_r1_coverage/replay_armR.npz')
  ap.add_argument('--cont-ckpt', default='')
  ap.add_argument('--critics', nargs='*', default=[])
  ap.add_argument('--model', choices=('S', 'N'), default='S')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--steps', type=int, default=0, help='smoke only: override the update budget')
  args = ap.parse_args(argv)
  global STEPS
  if args.steps:
    STEPS = int(args.steps)
  {'seal': seal, 'train': train, 'analyze': analyze}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
