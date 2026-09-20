"""Step 3a of the simulator-paired one-step ETT (AntMaze V6): full-length
rollout diagnostic of the fitted one-step model against the held-out
simulator branches -- distributional, not trajectory-by-trajectory.

User's decision (2026-09-20): before any future is generated, roll the
model out over the WHOLE remaining horizon from held-out anchors under two
advice processes and compare the distributions of outcomes with the
reference branches:

  C  the simulator teacher supplies a_b along the MODEL path (the same
     construction as the supervision: state reconstructed at the anchor,
     latches re-derived under the branch's own redrawn timetable, relay
     walked along the path) -- isolates the motion + onset model;
  A  the protocol's memoryless nominal a_b ~ P(a_b | s) (a mixture density
     network fitted on the d05 log, visible state only, never the hazard
     clock) -- the same-condition control for the advice process.

The continuation policy is the start agent's mode (the reference branches'
continuation); the first query is the logged torque; the path ends at
reach (|xy - goal| <= 0.5), at the horizon (800 - t), or at onset.  Motion
and the mode policy are deterministic, so under C every anchor has ONE
model path; the onset head gives a hazard p_j per step and the outcome
distribution is integrated EXACTLY along the path (survival S_j =
prod_{i<j} (1 - p_i); P(death) = 1 - S_end; the death time / position
distribution = S_{j-1} p_j over the steps; P(reach) = S at the reach step).
Under A the advice is sampled, so K paths per anchor, each integrated the
same way.  No path continues after an onset (zero recovery by
construction of the branch semantics).

Compared per anchor stratum (region) and pooled, model vs the reference
branches (one realised outcome per anchor): death / reach / timeout /
entered-far rates (model = mean of the exact per-anchor probabilities),
the distributions of the death time (steps after the anchor), the death
x-position and the time to reach (weighted two-sample KS against the
branches' realised values), and the per-anchor discrimination of the
model's P(death) against the realised death (AUROC).  Thresholds are
pre-registered in `manifest.json`: rate gaps <= 0.05 (entered-far
<= 0.03), KS <= 0.15, each on the pooled sample and in every stratum with
>= 200 anchors.  Reading (user's): C fails -> fix the motion / onset model
first; C passes and A fails -> the advice process is the bottleneck (then
a latched nominal from the visible history may be tried); both pass ->
the protocol nominal suffices.

  python scripts/diag_v6_ett_rollout.py seal
  python scripts/diag_v6_ett_rollout.py nominal                      # GPU: the MDN on the log
  python scripts/diag_v6_ett_rollout.py roll --variant C --fold f --workers 20     # CPU
  python scripts/diag_v6_ett_rollout.py roll --variant A --fold f --workers 20
  python scripts/diag_v6_ett_rollout.py report
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402

EC = MP.OUT / 'ett_context'
OS = MP.OUT / 'ett_one_step'
OUT = MP.OUT / 'ett_rollout'
V2 = False                       # set by --v2: the v2 models (stationary atom, history features), output dir ett_rollout_v2/
OUT_V2 = MP.OUT / 'ett_rollout_v2'
V3 = False                       # --v3: the v3 models (v2 code path), outputs under ett_rollout_v3/
OUT_V3 = MP.OUT / 'ett_rollout_v3'
HIST_FIX = False                 # --hist-fix: the history counters passed to the model equal the training rows' run_length (consecutive
                                 # flagged rows BEFORE the current one: 0 at the first hold / band row); without it the rollout passed 1
                                 # at the first row (an offset of +1 on every hold / band row).  Outputs under <OUT>/hist_fix/.
                                 # NOT exact (user's review of a408cef): run_length counts 0 at a flagged PATH START but 1 at a flagged row
                                 # that follows an unflagged one; the uniform -1 fixes the first case and breaks the second.
HIST_EXACT = False               # --hist-exact: the counters replicate run_length row by row (last reset index = the path start or the last
                                 # unflagged row; counter = j - last when flagged, else 0).  Outputs under <OUT>/hist_exact/.
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W
HORIZON = MP.HORIZON
SUCCESS_DIST = 0.5
DETOUR_Y, DETOUR_MAX_X = 6.0, 2.0
N_PER_FOLD, K_DRAWS_A = 2000, 4
SAMPLE_SEED, ADVICE_SEED = 201_000_000, 202_000_000
THRESH = {'rate_gap': 0.05, 'far_gap': 0.03, 'ks': 0.15, 'min_stratum_anchors': 200}
MDN = {'k': 5, 'hidden': (256, 256), 'steps': 20_000, 'batch': 1024, 'lr': 3e-4, 'seed': 0, 'val_episodes': 50, 'min_log_std': -5.0}


def model_dir():
  return MP.OUT / 'ett_one_step_v3' if V3 else (MP.OUT / 'ett_one_step_v2' if V2 else OS)


# ------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6 one-step ETT, Step 3a: full-length rollout diagnostic vs the held-out simulator branches (distributional), advice C (simulator teacher) vs A (memoryless nominal)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'status': 'oracle-supervised engineering stage; nothing downstream generated',
         'models': {f'fold{f}': {'path': str(model_dir() / f'model_fold{f}.pkl'), 'json_sha256': MP.sha256(model_dir() / f'model_fold{f}.json')} for f in range(3)}, 'model_version': ('v3' if V3 else ('v2' if V2 else 'v1')),
         'history_counters': ('exact: replicate fit_v6_ett_one_step_v2.run_length row by row' if HIST_EXACT else ('fix: uniform -1 (0 at the first flagged row; not exact after an unflagged row)' if HIST_FIX else 'raw: +1 at every flagged row vs the training rows')),
         'one_step_check_sha256': MP.sha256(model_dir() / 'check.json'), 'supervision_sha256': MP.sha256(EC / 'supervision.json'),
         'anchors': f'{N_PER_FOLD} held-out anchors per fold, uniform over anchors, seed {SAMPLE_SEED}; strata = anchor region', 'continuation': 'start agent mode (the reference branches\' continuation); first query = the logged torque',
         'termination': 'reach (|xy - goal| <= 0.5), horizon 800 - t, onset (exact hazard integration along the path; no recovery)',
         'advice': {'C': 'simulator teacher along the model path under the branch\'s redrawn timetable (Step 0 / 1 construction)',
                    'B': f'the learnt advice generator v3 (fit_v6_ett_advice.py; state + own history incl. the first mouth arrival + a persistent hidden context drawn per path from the prior; MAP hold decision, sampled torque; cross-fitted by fold; gates3_map.json all folds pass), K = {K_DRAWS_A} advice paths per anchor, seed {ADVICE_SEED}; the evaluation episode\'s actual context is never read',
                    'A': f'memoryless nominal a_b ~ MDN(s) fitted on the d05 log (k {MDN["k"]}, {MDN["hidden"]}, {MDN["steps"]} steps, seed {MDN["seed"]}; fixed final iterate), K = {K_DRAWS_A} sampled advice paths per anchor, seed {ADVICE_SEED}'},
         'comparison': {'rates': 'death / reach / timeout / entered-far: model mean of exact per-anchor probabilities vs the branches\' realised rates',
                        'distributions': 'death time (steps after the anchor), death x, time to reach: weighted two-sample KS vs the branches\' realised values',
                        'discrimination': 'AUROC of the per-anchor P(death) against the realised branch death (reported, not gated)'},
         'thresholds': THRESH, 'gate_rule': 'every rate gap and KS within threshold on the pooled sample and in every stratum with >= min_stratum_anchors anchors; evaluated separately for C and A',
         'reading': {'C_fails': 'fix the motion / onset model first', 'C_passes_A_fails': 'the advice process is the bottleneck; a latched nominal from the visible history is the next candidate',
                     'both_pass': 'the protocol nominal suffices for the futures'}}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# ---------------------------------------------------------------- nominal
def build_mdn(k=MDN['k']):
  import haiku as hk
  import jax.numpy as jnp

  def net(x):
    h = hk.nets.MLP(list(MDN['hidden']), activate_final=True)(x)
    logits = hk.Linear(k)(h); mu = hk.Linear(k * 8)(h).reshape(-1, k, 8); ls = jnp.maximum(hk.Linear(k * 8)(h).reshape(-1, k, 8), MDN['min_log_std'])
    return logits, mu, ls
  return hk.without_apply_rng(hk.transform(net))


def mode_nominal(args):
  import jax
  import jax.numpy as jnp
  import optax
  OUT.mkdir(parents=True, exist_ok=True)
  out = OUT / 'nominal_mdn.pkl'
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  obs, act, lengths, _ = MP.load_dataset()
  ee = np.repeat(np.arange(len(lengths)), lengths - 1); tt = np.concatenate([np.arange(L - 1) for L in lengths])
  X = obs[ee, tt, :OBS_W].astype(np.float32); A = act[ee, tt].astype(np.float32)
  rng = np.random.default_rng(MDN['seed'])
  val_eps = set(rng.choice(len(lengths), size=MDN['val_episodes'], replace=False).tolist())
  vmask = np.isin(ee, list(val_eps)); tr = np.flatnonzero(~vmask); va = np.flatnonzero(vmask)
  mean, std = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-3)
  Xn = ((X - mean) / std).astype(np.float32)
  net = build_mdn(); params = net.init(jax.random.PRNGKey(MDN['seed']), jnp.zeros((1, OBS_W), jnp.float32))
  opt = optax.adam(MDN['lr']); ost = opt.init(params)

  def nll(p, x, a):
    logits, mu, ls = net.apply(p, x)
    lp = -0.5 * jnp.sum(((a[:, None, :] - mu) / jnp.exp(ls)) ** 2 + 2 * ls + jnp.log(2 * jnp.pi), axis=-1) + jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jax.scipy.special.logsumexp(lp, axis=-1))

  @jax.jit
  def step(p, o, x, a):
    l, g = jax.value_and_grad(nll)(p, x, a); up, o = opt.update(g, o, p); return optax.apply_updates(p, up), o, l
  vnll = jax.jit(nll)
  t0, hist = time.time(), []
  for it in range(1, MDN['steps'] + 1):
    i = rng.choice(tr, size=MDN['batch'])
    params, ost, l = step(params, ost, jnp.asarray(Xn[i]), jnp.asarray(A[i]))
    if it % 1000 == 0:
      v = float(np.mean([float(vnll(params, jnp.asarray(Xn[va[j:j + 8192]]), jnp.asarray(A[va[j:j + 8192]]))) for j in range(0, len(va), 8192)]))
      hist.append({'step': it, 'train_nll': float(l), 'val_nll': v}); print(f'[nominal step {it:>6}] train nll {float(l):.3f} val nll {v:.3f} {it / (time.time() - t0):.0f} it/s', flush=True)
  md = {'params': jax.tree_util.tree_map(np.asarray, params), 'mean': mean.astype(np.float32), 'std': std.astype(np.float32), 'mdn': MDN, 'history': hist, 'n_train_rows': int(len(tr)), 'val_episodes': sorted(val_eps),
        'note': 'fixed final iterate (no selection); visible state + goal xy only'}
  with out.open('wb') as f:
    pickle.dump(md, f)
  MP.write_json(out.with_suffix('.json'), {k: v for k, v in md.items() if k not in ('params', 'mean', 'std')})
  print(f'saved {out} ({time.time() - t0:.0f} s)', flush=True)


class Nominal:
  def __init__(self, md):
    import jax
    import jax.numpy as jnp
    net = build_mdn(); p = jax.tree_util.tree_map(jnp.asarray, md['params'])
    self.mean, self.std = md['mean'], md['std']
    self._f = jax.jit(lambda x: net.apply(p, x))

  def sample(self, obs31, rng):
    logits, mu, ls = (np.asarray(v) for v in self._f(((obs31[None] - self.mean) / self.std).astype(np.float32)))
    pr = np.exp(logits[0] - logits[0].max()); pr /= pr.sum()
    c = rng.choice(len(pr), p=pr)
    return np.clip(mu[0, c] + np.exp(ls[0, c]) * rng.standard_normal(8), -1.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------- roll
def sample_anchors(fold):
  S = np.load(EC / 'supervision.npz', allow_pickle=False)
  fa = S['fold_anchor'].astype(np.int64)
  ks = np.flatnonzero(fa == fold)
  rng = np.random.default_rng(SAMPLE_SEED + fold)
  return np.sort(rng.choice(ks, size=min(N_PER_FOLD, len(ks)), replace=False))


def _worker(args):
  ks, fold, variant, seed, v2, v3, hist_fix, hist_exact = args
  global V2, V3, OUT, HIST_FIX, HIST_EXACT
  V3 = bool(v3); V2 = bool(v2) or V3; HIST_FIX = bool(hist_fix); HIST_EXACT = bool(hist_exact)
  OUT = OUT_V3 if V3 else (OUT_V2 if V2 else MP.OUT / 'ett_rollout')      # spawned workers re-import the module: carry the flags explicitly
  if HIST_EXACT:
    OUT = OUT / 'hist_exact'
  elif HIST_FIX:
    OUT = OUT / 'hist_fix'
  import jax
  import jax.numpy as jnp
  import audit_v6_ett_context as AC
  from crl import checkpoint
  if V2:
    import fit_v6_ett_one_step_v2 as F2
    from fit_v6_ett_one_step_v2 import Predictor2, HOLD_TOL, in_band
    if V3:
      F2.OUT = F2.OUT_V3
    P = Predictor2(F2.load_model(fold))
  else:
    from fit_v6_ett_one_step import Predictor, load_model
    P = Predictor(load_model(fold))
  H = dict(np.load(EC / 'hidden.npz', allow_pickle=False)); BA = dict(np.load(EC / 'branch_advice.npz', allow_pickle=False))
  obs, act, lengths, _ = MP.load_dataset(); anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(MP.START_CKPT); pp = st.policy_params
  mode = jax.jit(lambda o: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  env, teacher = AC.make_env_teacher(seed); env.reset()
  nominal = Nominal(pickle.load(((MP.OUT / 'ett_rollout') / 'nominal_mdn.pkl').open('rb'))) if variant == 'A' else None     # the same nominal for v1 and v2
  if variant == 'B':
    import fit_v6_ett_advice as FA
    FA.DECISION = 'map'                                # the generator's MAP hold decision (the teacher's rule is deterministic); torque sampled
    gen = FA.load_generator(fold)
  out = []
  for k in ks:
    k = int(k); e, t = int(anchors.episode[k]), int(anchors.t[k])
    goal = obs[e, t, STATE_DIM:OBS_W].astype(np.float32); s0 = obs[e, t, :STATE_DIM].astype(np.float32); a_q0 = act[e, t].astype(np.float32)
    un = {1: bool(BA['u1_new'][k]), 2: bool(BA['u2_new'][k])}; tn = {1: int(BA['t0_1_new'][k]), 2: int(BA['t0_2_new'][k])}
    if variant == 'C':
      u = {1: bool(H['u1'][e]), 2: bool(H['u2'][e])}; t0 = {1: int(H['t0_1'][e]), 2: int(H['t0_2'][e])}
      _, _, dstep, snaps = AC.replay_episode(env, teacher, obs[e], int(lengths[e]), str(H['intent'][e]), u, t0, anchor_ts=[t])
      snap0 = AC.rederive_latches(snaps[t], un, tn, dstep)
    n_draw = 1 if variant == 'C' else K_DRAWS_A
    for d in range(n_draw):
      rng = np.random.default_rng(ADVICE_SEED + 1000 * k + d)
      if variant == 'C':
        AC.restore_teacher(teacher, snap0)
      s = s0.copy(); p_seq, x_seq, y_seq = [], [], []
      max_steps = HORIZON - t; reached = False; far = -1
      if variant == 'B':                                 # one hidden context per advice path, kept for the whole path; the generator's own history
        z = FA.sample_context(rng, 1)
        kh_prev = int(FA.zero_run_before(act, np.array([e]), np.array([t]))[0]); prev_ab = (act[e, t - 1] if t >= 1 else np.zeros(8, np.float32)).astype(np.float32)
        last_b_gen = 0; m_gen = FA.first_mouth_in_prefix(obs[e], t)         # the logged prefix's first mouth arrival per zone (-1 = not yet)
      kh = kb = 0                                        # v2 history features along THIS path: consecutive hold steps, steps since band entry
      last_h = last_b = 0                                # --hist-exact: index of the last reset row of run_length (path start, or the last unflagged row)
      for j in range(max_steps):
        o31 = np.concatenate([s, goal])
        a_q = a_q0 if j == 0 else np.asarray(mode(jnp.asarray(o31[None])), np.float32)[0]
        if variant == 'C':
          a_b = teacher.act(AC.obs58(env, o31), AC.schedule_of(un, tn, t + j))
        elif variant == 'A':
          a_b = nominal.sample(o31, rng)
        else:
          band_now = bool(in_band(s[None, :2])[0])
          if j == 0 or not band_now:
            last_b_gen = j
          kb_gen = (j - last_b_gen) if band_now else 0
          ctx = FA.context_features(np.array([t + j]), *[np.asarray(v) for v in z])
          for zi, zz in enumerate((1, 2)):
            if m_gen[zi] < 0 and bool(FA.at_mouth(s[None, :2].astype(np.float64), zz)[0]):
              m_gen[zi] = t + j
          mf = FA.mouth_features(np.array([t + j]), np.array([m_gen[0]]), np.array([m_gen[1]]), np.asarray(z[2]), np.asarray(z[3]))
          hold_g, ab_g, _ = gen.sample(FA.features(o31[None], ctx, mf, np.array([kh_prev]), np.array([kb_gen]), prev_ab[None], gen.norm), rng)
          a_b = ab_g[0]; kh_prev = kh_prev + 1 if bool(hold_g[0]) else 0; prev_ab = a_b
        if V2:
          hold = bool(np.abs(a_b).max() < HOLD_TOL); band = bool(in_band(s[None, :2])[0])
          kh = kh + 1 if hold else 0
          kb = kb + 1 if band else 0
          if HIST_EXACT:                                 # run_length: reset = (~flag) | start; last = the latest reset index; value = j - last if flag else 0
            if j == 0 or not hold:
              last_h = j
            if j == 0 or not band:
              last_b = j
            kh_in, kb_in = (j - last_h if hold else 0), (j - last_b if band else 0)
          else:
            kh_in, kb_in = (max(kh - 1, 0), max(kb - 1, 0)) if HIST_FIX else (kh, kb)
          s_next, p, _ = P.step(s[None], a_b[None], a_q[None], np.array([kh_in]), np.array([kb_in])); s_next, p = s_next[0], float(p[0])
        else:
          s_next, p = P.step(s[None], a_b[None], a_q[None]); s_next, p = s_next[0], float(p[0])
        p_seq.append(p); x_seq.append(float(s_next[0])); y_seq.append(float(s_next[1]))
        s = s_next
        if far < 0 and s[1] >= DETOUR_Y and s[0] < DETOUR_MAX_X:
          far = j + 1
        if np.linalg.norm(s[:2] - goal) <= SUCCESS_DIST:
          reached = True; break
      p_arr = np.asarray(p_seq, np.float64); S_before = np.concatenate([[1.0], np.cumprod(1 - p_arr)])     # S_before[j] = survival into step j
      death_w = S_before[:-1] * p_arr                                                                     # death at transition j -> j+1
      S_end = float(S_before[-1])
      out.append({'anchor': k, 'draw': d, 'steps': len(p_arr), 'reached': bool(reached), 'p_death': float(death_w.sum()), 'p_reach': (S_end if reached else 0.0), 'p_timeout': (0.0 if reached else S_end),
                  'p_far': (float(S_before[far]) if far >= 0 else 0.0), 'far_step': far, 'reach_step': (len(p_arr) if reached else -1),
                  'death_w': death_w.astype(np.float32), 'x_seq': np.asarray(x_seq, np.float16), 'final_xy': [x_seq[-1], y_seq[-1]], 'max_p': float(p_arr.max()) if len(p_arr) else 0.0})
  return out


def mode_roll(args):
  from multiprocessing import get_context
  OUT.mkdir(parents=True, exist_ok=True)
  f, v = int(args.fold), args.variant
  out = OUT / f'roll_{v}_fold{f}.pkl'
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  if v == 'A' and not ((MP.OUT / 'ett_rollout') / 'nominal_mdn.pkl').exists():      # the nominal lives with the v1 diagnostic; shared by v2
    raise SystemExit('fit the nominal first')
  if v == 'B':
    import fit_v6_ett_advice as FA
    if not FA.model_path(f).exists():
      raise SystemExit('fit the advice generator first')
  ks = sample_anchors(f)
  if args.limit:
    ks = ks[:args.limit]
  n = max(1, args.workers * 3); parts = [ks[i::n] for i in range(n)]; parts = [p for p in parts if len(p)]
  t0 = time.time()
  if args.workers <= 1:
    res = [_worker((p, f, v, 203_000_000 + i, V2, V3, HIST_FIX, HIST_EXACT)) for i, p in enumerate(parts)]
  else:
    with get_context('spawn').Pool(args.workers) as pool:
      res = pool.map(_worker, [(p, f, v, 203_000_000 + i, V2, V3, HIST_FIX, HIST_EXACT) for i, p in enumerate(parts)])
  rows = [r for part in res for r in part]
  with out.open('wb') as fh:
    pickle.dump({'fold': f, 'variant': v, 'anchors': ks, 'rows': rows, 'wall_seconds': time.time() - t0}, fh)
  print(f'{v} fold {f}: {len(rows)} paths from {len(ks)} anchors in {time.time() - t0:.0f} s; mean p_death {np.mean([r["p_death"] for r in rows]):.3f} p_reach {np.mean([r["p_reach"] for r in rows]):.3f} p_far {np.mean([r["p_far"] for r in rows]):.3f}', flush=True)


# ------------------------------------------------------------------ report
def wks(x, wx, y, wy):
  """Weighted two-sample Kolmogorov-Smirnov distance."""
  x, y = np.asarray(x, float), np.asarray(y, float); wx, wy = np.asarray(wx, float), np.asarray(wy, float)
  if wx.sum() <= 0 or wy.sum() <= 0:
    return float('nan')
  grid = np.sort(np.concatenate([x, y]))
  ox, oy = np.argsort(x), np.argsort(y)
  Fx = np.cumsum(wx[ox]) / wx.sum(); Fy = np.cumsum(wy[oy]) / wy.sum()
  fx = np.searchsorted(x[ox], grid, side='right'); fy = np.searchsorted(y[oy], grid, side='right')
  cx = np.where(fx > 0, Fx[np.maximum(fx - 1, 0)], 0.0); cy = np.where(fy > 0, Fy[np.maximum(fy - 1, 0)], 0.0)
  return float(np.abs(cx - cy).max())


def auroc(score, pos):
  from scipy.stats import rankdata
  pos = np.asarray(pos, bool); r = rankdata(score); n1, n0 = int(pos.sum()), int((~pos).sum())
  return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def mode_report(args):
  from exp_v6_learned_ett import REGIONS, region_of
  man = MP.read_json(OUT / 'manifest.json')
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    boff, blen, boc = d['offset'].astype(np.int64), d['length'].astype(np.int64), d['outcome'].astype(str)
    bxy = d['obs_rows'][:, :2]
  reg = region_of(anchors.state[:, :2])
  res = {'variants': {}, 'thresholds': THRESH}
  for v in ('C', 'A', 'B'):
    rows = []
    for f in range(3):
      p = OUT / f'roll_{v}_fold{f}.pkl'
      if not p.exists():
        continue
      with p.open('rb') as fh:
        rows += pickle.load(fh)['rows']
    if not rows:
      continue
    ks = np.array([r['anchor'] for r in rows]); n_draw = len(rows) / len(np.unique(ks))
    # reference per anchor
    ref_death = boc[ks] == 'death'; ref_reach = boc[ks] == 'success'; ref_timeout = boc[ks] == 'timeout'; ref_steps = blen[ks] - 1
    ref_far = np.array([bool(((bxy[boff[k] + 1:boff[k] + blen[k], 1] >= DETOUR_Y) & (bxy[boff[k] + 1:boff[k] + blen[k], 0] < DETOUR_MAX_X)).any()) for k in ks])
    ref_death_x = np.array([bxy[boff[k] + blen[k] - 1, 0] for k in ks])
    m_death = np.array([r['p_death'] for r in rows]); m_reach = np.array([r['p_reach'] for r in rows]); m_timeout = np.array([r['p_timeout'] for r in rows]); m_far = np.array([r['p_far'] for r in rows])
    strata = {'all': np.ones(len(rows), bool)}
    rr = reg[ks]
    for r_, name in enumerate(REGIONS):
      if (rr == r_).sum() >= THRESH['min_stratum_anchors'] * n_draw:
        strata[f'anchor in {name}'] = rr == r_
    V = {'n_paths': len(rows), 'n_anchors': int(len(np.unique(ks))), 'strata': {}}
    gates_ok = True
    for name, m in strata.items():
      # death time / x distributions: model weights = death_w per step; reference = realised deaths
      mt, mx, mw = [], [], []
      for r, keep in zip(rows, m):
        if keep and r['p_death'] > 1e-6:
          w = r['death_w']; idx = np.flatnonzero(w > 1e-7)
          mt.append(idx + 1); mx.append(r['x_seq'][idx].astype(float)); mw.append(w[idx])
      mt = np.concatenate(mt) if mt else np.zeros(0); mx = np.concatenate(mx) if mx else np.zeros(0); mw = np.concatenate(mw) if mw else np.zeros(0)
      rd = m & ref_death
      # reach time: model reach steps weighted by p_reach; reference successes
      mr_t = np.array([r['reach_step'] for r, keep in zip(rows, m) if keep and r['reached']]); mr_w = np.array([r['p_reach'] for r, keep in zip(rows, m) if keep and r['reached']])
      rs = m & ref_reach
      blk = {'n_anchors': int(m.sum() / n_draw), 'death_rate': {'sim': float(ref_death[m].mean()), 'model': float(m_death[m].mean())}, 'reach_rate': {'sim': float(ref_reach[m].mean()), 'model': float(m_reach[m].mean())},
             'timeout_rate': {'sim': float(ref_timeout[m].mean()), 'model': float(m_timeout[m].mean())}, 'far_rate': {'sim': float(ref_far[m].mean()), 'model': float(m_far[m].mean())},
             'ks_death_time': wks(mt, mw, ref_steps[rd], np.ones(int(rd.sum()))) if rd.sum() >= 20 and len(mt) else None,
             'ks_death_x': wks(mx, mw, ref_death_x[rd], np.ones(int(rd.sum()))) if rd.sum() >= 20 and len(mx) else None,
             'ks_reach_time': wks(mr_t, mr_w, ref_steps[rs], np.ones(int(rs.sum()))) if rs.sum() >= 20 and len(mr_t) else None,
             'death_time_median': {'sim': float(np.median(ref_steps[rd])) if rd.any() else None, 'model': (float(np.interp(0.5, np.cumsum(mw[np.argsort(mt)]) / mw.sum(), np.sort(mt))) if len(mt) else None)},
             'reach_time_median': {'sim': float(np.median(ref_steps[rs])) if rs.any() else None, 'model': (float(np.interp(0.5, np.cumsum(mr_w[np.argsort(mr_t)]) / mr_w.sum(), np.sort(mr_t))) if len(mr_t) else None)},
             'auroc_p_death_vs_realised': auroc(m_death[m], ref_death[m]) if ref_death[m].any() and (~ref_death[m]).any() else None}
      g = {'death': abs(blk['death_rate']['sim'] - blk['death_rate']['model']) <= THRESH['rate_gap'], 'reach': abs(blk['reach_rate']['sim'] - blk['reach_rate']['model']) <= THRESH['rate_gap'],
           'timeout': abs(blk['timeout_rate']['sim'] - blk['timeout_rate']['model']) <= THRESH['rate_gap'], 'far': abs(blk['far_rate']['sim'] - blk['far_rate']['model']) <= THRESH['far_gap'],
           'ks_death_time': (blk['ks_death_time'] is None) or blk['ks_death_time'] <= THRESH['ks'], 'ks_death_x': (blk['ks_death_x'] is None) or blk['ks_death_x'] <= THRESH['ks'],
           'ks_reach_time': (blk['ks_reach_time'] is None) or blk['ks_reach_time'] <= THRESH['ks']}
      blk['gates'] = g; blk['passed'] = bool(all(g.values())); gates_ok = gates_ok and blk['passed']
      V['strata'][name] = blk
    V['all_gates_passed'] = gates_ok
    res['variants'][v] = V
  MP.write_json(OUT / 'report.json', res)
  L = ['# Step 3a: full-length model rollouts vs the held-out simulator branches (distributional)', '', f'`manifest.json` (thresholds sealed), `report.json`.  Model = exact hazard integration along the deterministic path (C: one path per anchor; A: {K_DRAWS_A} sampled-advice paths).  '
       'Reference = the branch of the same anchor (one realised outcome).  A KS entry is blank when fewer than 20 realised events exist in the stratum.', '']
  for v, V in res['variants'].items():
    L += [f'## Advice {v}: **{"ALL GATES PASSED" if V["all_gates_passed"] else "GATE(S) FAILED"}** ({V["n_anchors"]} anchors, {V["n_paths"]} paths)', '',
          '| stratum | anchors | death sim / model | reach sim / model | timeout sim / model | far sim / model | KS death time | KS death x | KS reach time | death time median sim / model | reach time median sim / model | AUROC P(death) | pass |',
          '|---|---:|---|---|---|---|---:|---:|---:|---|---|---:|---|']
    for name, b in V['strata'].items():
      fmt = lambda q: f'{b[q]["sim"]:.3f} / {b[q]["model"]:.3f}'
      ks_ = lambda q: ('-' if b[q] is None else f'{b[q]:.3f}')
      med = lambda q: f'{b[q]["sim"] if b[q]["sim"] is None else round(b[q]["sim"])} / {b[q]["model"] if b[q]["model"] is None else round(b[q]["model"])}'
      L.append(f'| {name} | {b["n_anchors"]} | {fmt("death_rate")} | {fmt("reach_rate")} | {fmt("timeout_rate")} | {fmt("far_rate")} | {ks_("ks_death_time")} | {ks_("ks_death_x")} | {ks_("ks_reach_time")} | {med("death_time_median")} | {med("reach_time_median")} | '
               f'{"-" if b["auroc_p_death_vs_realised"] is None else format(b["auroc_p_death_vs_realised"], ".3f")} | {"yes" if b["passed"] else "NO: " + ", ".join(k for k, ok in b["gates"].items() if not ok)} |')
    L.append('')
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'nominal', 'roll', 'report'))
  ap.add_argument('--variant', choices=('C', 'A', 'B'), default='C')
  ap.add_argument('--fold', type=int, default=0)
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--v2', action='store_true', help='use the v2 one-step models; outputs under ett_rollout_v2/')
  ap.add_argument('--v3', action='store_true', help='use the v3 one-step models (v2 code path); outputs under ett_rollout_v3/')
  ap.add_argument('--hist-fix', action='store_true', help='history counters aligned with the training rows (0 at the first hold / band row); outputs under <OUT>/hist_fix/')
  ap.add_argument('--hist-exact', action='store_true', help='history counters replicate fit_v6_ett_one_step_v2.run_length row by row; outputs under <OUT>/hist_exact/')
  args = ap.parse_args(argv)
  global V2, V3, OUT, HIST_FIX, HIST_EXACT
  if args.v3:
    V3 = True; V2 = True; OUT = OUT_V3
  elif args.v2:
    V2 = True; OUT = OUT_V2
  if args.hist_exact:
    HIST_EXACT = True; OUT = OUT / 'hist_exact'
  elif args.hist_fix:
    HIST_FIX = True; OUT = OUT / 'hist_fix'
  OUT.mkdir(parents=True, exist_ok=True)
  {'seal': mode_seal, 'nominal': mode_nominal, 'roll': mode_roll, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
