"""AntMaze V6 mainline: a learned ETT model of the blind agent's continuation
(ORACLE-LABEL-TRAINED, cross-fitted) and the arm CF-learned (CFL).

Status under notes/MAINLINE_CONTRACT.md (sections 1a / 1b).  The pilot's arm
CF draws the critic's positive futures from simulator branches.  The intended
method replaces the simulator by a learned effect-of-treatment-on-the-treated
(ETT) model of the blind agent's continuation, fitted offline.  What this
script fits is the ENGINEERING INTERMEDIATE version the user allowed on
2026-09-19: the model is supervised on the pilot's oracle branches (the
simulator's labels).  It therefore tests the INTERFACE -- can a learned
generator of the critic's positive futures replace the simulator in arm CF
and keep the gain? -- and NOT the offline identification (whether such a
model can be fitted from the logged data alone, whose continuations are the
teacher's, not the blind agent's).  Every report of it says
"oracle-label-trained learned ETT".

The model.  Input: the anchor's 29-dim Ant state (standardised on the
training anchors) and the 8-dim logged torque.  Heads: (1) the discounted
future-goal marginal p_gamma(g | s, a) as a categorical over 0.5 x 0.5 maze
cells (x in [-2.5, 26.5], y in [-2.5, 10.5]: 58 x 26 = 1,508 cells) -- the
quantity the critic consumes; its training sampler is the pilot's own law
(future row m with P(m) proportional to gamma^m, gamma 0.999, truncated at
the branch end = reach / death / horizon), so route and death truncation are
inside the marginal; (2) the absorbing position m = 10 / 50 / 200 steps
after the query (path[min(m, len - 1)]) -- short- and long-segment checks;
(3) the branch outcome (success / death / timeout).  MLP 1024-1024 (ReLU),
cross-entropy on every head, Adam 3e-4, batch 4,096, 40,000 steps, early
stopping on held-out episodes (the marginal head's NLL).

Cross-fitting.  The d05 source episodes are permuted (seed 161000001) into 3
folds.  Model f trains on the anchors of the episodes NOT in fold f (a 10 %
slice of those episodes held out for early stopping) and writes the futures
of the anchors IN fold f: the model that generates an anchor's future never
saw that anchor's oracle branch nor any anchor of its episode.  A position-
only reference model (input: the anchor's xy) is fitted the same way, to
show how much of the held-out likelihood the pose / velocity / torque carry.

Arm CF-learned (CFL).  The sealed recipe of variants/critic_clip0.1 (critic
gradient clip 0.1 before Adam; NCE, actor loss, bc 0.05, gamma 0.999, the
start-agent initialisation, 30,000 updates, three paired seeds; identical
anchors, anchor weights and actor batches), with the critic's positive goal
of anchor k drawn from the learned marginal (`exp_v6_mainline_pilot.
LearnedFutures`: the stream's future uniform selects the cell by inverse
CDF, a uniform jitter inside the cell from the source's own RNG, so the
anchor sequence is the other arms').  Compared on the development draw (seed
3909, 300 episodes, mode) with the clipped O and CF (oracle) arms already
evaluated there.  Rules, sealed before fitting: primary CFL - O success
(mean over the 3 paired seeds > 2 x seed s.e. and 3/3); practical CFL -
start; CFL - CF(oracle) reported with its seed s.e. (how much of the oracle
gain the learned generator keeps).  No checkpoint or model selection after
the evaluation; the early stopping of the ETT model is on held-out
likelihood, before any policy is trained.

  python scripts/exp_v6_learned_ett.py seal
  python scripts/exp_v6_learned_ett.py fit --fold 0 [--inputs full|xy]
  python scripts/exp_v6_learned_ett.py generate        # -> futures_learned.npz, check.json, CHECK.md
  python scripts/exp_v6_learned_ett.py train --seeds 0 1 2
  python scripts/exp_v6_learned_ett.py evaluate --seeds 0 1 2
  python scripts/exp_v6_learned_ett.py report
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

import exp_v6_mainline_pilot as MP  # noqa: E402  (the pilot's pins: d05, gamma 0.999, eval seed 3909)

OUT = MP.OUT / 'learned_ett'
RECIPE = 'critic_clip0.1'                         # the sealed recipe the arm runs under (variants/critic_clip0.1/manifest.json)
GAMMA = MP.GAMMA
N_FOLDS, FOLD_SEED, ES_FRACTION = 3, 161_000_001, 0.10
X0, X1, Y0, Y1, CELL = -2.5, 26.5, -2.5, 10.5, 0.5
NX, NY = int(round((X1 - X0) / CELL)), int(round((Y1 - Y0) / CELL))
NCELL = NX * NY
HORIZONS = (10, 50, 200)
OUTCOMES = ('success', 'death', 'timeout')
FIT = {'hidden': (1024, 1024), 'batch': 4096, 'steps': 40_000, 'lr': 3e-4, 'val_every': 1000, 'val_pairs': 100_000, 'val_seed': 162_000_000, 'sample_seed': 163_000_000}
INPUT_SETS = ('full', 'xy')


# ------------------------------------------------------------------- grid
def cell_of(xy):
  ix = np.clip(np.floor((xy[:, 0] - X0) / CELL).astype(np.int64), 0, NX - 1)
  iy = np.clip(np.floor((xy[:, 1] - Y0) / CELL).astype(np.int64), 0, NY - 1)
  return iy * NX + ix


def cells_xy():
  c = np.arange(NCELL); ix = c % NX; iy = c // NX
  return np.stack([X0 + (ix + 0.5) * CELL, Y0 + (iy + 0.5) * CELL], axis=1).astype(np.float32)


REGIONS = ('start', 'pre_zone1', 'zone1', 'between', 'zone2', 'post_zone2', 'goal_area', 'west_column', 'top_corridor', 'east_column', 'other')


def region_of(xy):
  x, y = xy[:, 0], xy[:, 1]
  r = np.full(len(x), REGIONS.index('other'), np.int64)
  def put(m, name):
    r[m] = REGIONS.index(name)
  row = np.abs(y) < 2
  put((x < 2) & (y < 2), 'start')
  put((x >= 2) & (x < 6.6) & row, 'pre_zone1')
  put((x >= 6.6) & (x <= 9.4) & row, 'zone1')
  put((x > 9.4) & (x < 14.6) & row, 'between')
  put((x >= 14.6) & (x <= 17.4) & row, 'zone2')
  put((x > 17.4) & (x < 22) & row, 'post_zone2')
  put((x >= 22) & (y < 2), 'goal_area')
  put((x < 2) & (y >= 2) & (y < 6), 'west_column')
  put((y >= 6) & (x < 22), 'top_corridor')
  put((x >= 22) & (y >= 2), 'east_column')
  return r


CELL_REGION = region_of(cells_xy())
FAR = [REGIONS.index(n) for n in ('west_column', 'top_corridor', 'east_column')]
ZONES = [REGIONS.index(n) for n in ('zone1', 'zone2')]


# ------------------------------------------------------------------- data
def load_branches():
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    xy = np.ascontiguousarray(d['obs_rows'][:, :2]).astype(np.float32)
    off, L = d['offset'].astype(np.int64), d['length'].astype(np.int64)
    oc, ep, t = d['outcome'].astype(str), d['episode'].astype(np.int64), d['t'].astype(np.int64)
  return xy, off, L, oc, ep, t


def folds(anchors):
  """Per anchor: fold (0..2) of its source episode and the early-stop flag
  (the last 10 % of every fold's episodes in the permutation order)."""
  src = anchors.source_episode.astype(np.int64)
  uniq = np.unique(src)
  perm = np.random.default_rng(FOLD_SEED).permutation(uniq)
  fold_of, es_of = {}, {}
  for f in range(N_FOLDS):
    eps = perm[f::N_FOLDS]
    n_es = int(np.ceil(ES_FRACTION * len(eps)))
    for i, e in enumerate(eps):
      fold_of[int(e)] = f; es_of[int(e)] = bool(i >= len(eps) - n_es)
  fold = np.array([fold_of[int(e)] for e in src]); es = np.array([es_of[int(e)] for e in src])
  return fold, es


def features(anchors, inputs):
  if inputs == 'full':
    return np.concatenate([anchors.state, anchors.action], axis=1).astype(np.float32)
  if inputs == 'xy':
    return anchors.state[:, :2].astype(np.float32)
  raise ValueError(inputs)


def trunc_geom(rng, n):
  """m ~ P(m) proportional to gamma^m on 1..n (the pilot's CriticStream law)."""
  u = rng.random(len(n))
  m = np.ceil(np.log1p(-u * (1.0 - GAMMA ** n)) / np.log(GAMMA)).astype(np.int64)
  return np.clip(m, 1, n)


def geom_weights(L):
  """Row weights of the truncated geometric law for every branch row (root 0)."""
  n_rows = int(L.sum())
  anchor_of_row = np.repeat(np.arange(len(L)), L)
  m = np.arange(n_rows) - np.repeat(np.cumsum(L) - L, L)          # row index within the path
  w = np.where(m >= 1, GAMMA ** m, 0.0)
  z = np.bincount(anchor_of_row, weights=w, minlength=len(L))
  return anchor_of_row, m, w / z[anchor_of_row]


# ------------------------------------------------------------------ model
def build(n_in):
  import haiku as hk
  import jax.numpy as jnp

  def fwd(x):
    h = hk.nets.MLP(list(FIT['hidden']), activate_final=True)(x)
    out = {'marginal': hk.Linear(NCELL)(h), 'outcome': hk.Linear(len(OUTCOMES))(h)}
    for m in HORIZONS:
      out[f'h{m}'] = hk.Linear(NCELL)(h)
    return out
  return hk.without_apply_rng(hk.transform(fwd))


def model_path(inputs, fold, tag=''):
  return OUT / f'model_{inputs}_fold{fold}{tag}.pkl'


def load_model(path):
  with Path(path).open('rb') as f:
    return pickle.load(f)


def apply_model(md, X, chunk=8192):
  """Log-probabilities of every head for the rows of X (numpy), chunked."""
  import jax
  import jax.numpy as jnp
  net = build(X.shape[1])
  params = jax.tree_util.tree_map(jnp.asarray, md['params'])
  fn = jax.jit(lambda x: {k: jax.nn.log_softmax(v, axis=-1) for k, v in net.apply(params, x).items()})
  Xn = (X - md['norm']['mean']) / md['norm']['std']
  outs = {}
  for i in range(0, len(Xn), chunk):
    o = fn(jnp.asarray(Xn[i:i + chunk], jnp.float32))
    for k, v in o.items():
      outs.setdefault(k, []).append(np.asarray(v, np.float32))
  return {k: np.concatenate(v) for k, v in outs.items()}


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  anchors, ameta = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  fold, es = folds(anchors)
  vb = MP.variant_base(RECIPE)
  man = {'experiment': 'AntMaze V6 mainline: oracle-label-trained learned ETT (cross-fitted) and the arm CF-learned under the sealed critic_clip0.1 recipe',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(),
         'status': ('ENGINEERING INTERMEDIATE EXPERIMENT (user, 2026-09-19): the ETT model is supervised on the pilot\'s simulator branches, so it tests whether a '
                    'learned generator of the critic\'s positive futures can replace the simulator in arm CF and keep the gain -- NOT whether such a model can be '
                    'identified from the logged data alone (the logged continuations are the teacher\'s).  Reported as "oracle-label-trained learned ETT".'),
         'inputs': {'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'), 'branches_sha256': MP.sha256(MP.OUT / 'branches_cf.npz'), 'start_ckpt_sha256': MP.sha256(MP.START_CKPT),
                    'recipe_manifest_sha256': MP.sha256(vb / 'manifest.json'), 'recipe': RECIPE, 'recipe_overrides': MP.VARIANTS[RECIPE], 'base_pilot_manifest_sha256': MP.sha256(MP.OUT / 'manifest.json')},
         'model': {'input_full': '29-dim Ant state (standardised on the training anchors) + 8-dim logged torque', 'input_xy_reference': 'the anchor\'s xy only (position-only reference, not used by the arm)',
                   'heads': {'marginal': f'p_gamma(g | s, a): categorical over {NX} x {NY} = {NCELL} cells of {CELL} (x [{X0}, {X1}], y [{Y0}, {Y1}]); training sampler = the pilot\'s future law (P(m) ~ gamma^m, gamma {GAMMA}, truncated at the branch end)',
                             'horizons': {f'h{m}': f'absorbing position {m} steps after the query, path[min({m}, len - 1)]' for m in HORIZONS}, 'outcome': list(OUTCOMES)},
                   'fit': FIT, 'anchor_sampling': 'by anchor weight (multiplicity), as the critic stream', 'early_stopping': 'best held-out marginal NLL over the validation checks (held-out episodes of the training folds); no other selection'},
         'cross_fitting': {'n_folds': N_FOLDS, 'fold_seed': FOLD_SEED, 'unit': 'source episode (merged-pool id)', 'es_fraction_of_training_episodes': ES_FRACTION,
                           'rule': 'model f: train on folds != f minus the early-stop slice, early-stop on that slice, GENERATE for fold f',
                           'anchors_per_fold': [int((fold == f).sum()) for f in range(N_FOLDS)], 'weight_per_fold': [float(anchors.weight[fold == f].sum()) for f in range(N_FOLDS)],
                           'es_anchors_per_fold': [int(((fold == f) & es).sum()) for f in range(N_FOLDS)]},
         'checks_before_the_arm': ['held-out marginal NLL (exact expectation over the oracle path under the truncated geometric law), full vs xy-only reference',
                                   'region-mass agreement per anchor (L1 over the maze regions) vs the oracle path, by anchor stratum (reset / own region / detour vs shortcut episode)',
                                   'death: outcome-head P(death) vs the oracle branch outcome (AUC); hazard-zone and far-route masses vs the oracle',
                                   'horizon heads: NLL and expected-position error at m = 10 / 50 / 200'],
         'arm': {'name': 'CFL', 'futures': 'LearnedFutures over futures_learned.npz (one cell draw by the stream\'s future uniform, uniform jitter inside the cell from the source\'s own RNG)',
                 'everything_else': 'the sealed critic_clip0.1 recipe: same anchors and weights, actor stream and seeds, start-agent initialisation, fresh paired critics, 30,000 updates, evaluation seed 3909 mode',
                 'next_rows': 'the recorded next rows (unused by the Monte-Carlo NCE loss; Transition layout only)'},
         'comparisons': {'primary': 'CFL - O(critic_clip0.1) success, paired per episode; rule: mean over the 3 paired seeds > 2 x seed s.e. and 3/3',
                         'practical': 'CFL - start success (required separately)', 'kept': 'CFL - CF(critic_clip0.1, oracle) success with its seed s.e.: how much of the oracle gain the learned generator keeps',
                         'reported': 'success, detour, death, timeout, hazard / no-hazard success, the route ledger'},
         'no_selection': 'one model family, one fold seed, one arm; the final checkpoints evaluated once on the development draw; nothing chosen after the result',
         'reading': {'kept': 'CFL - O meets the rule -> the interface holds: a learned generator of positive futures (here oracle-label-trained) carries the gain',
                     'not_kept': 'CFL - O does not meet the rule -> the loss lies between the oracle branches and the learned marginal (the checks say where)'}}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# --------------------------------------------------------------------- fit
def mode_fit(args):
  import jax
  import jax.numpy as jnp
  import optax
  OUT.mkdir(parents=True, exist_ok=True)
  f, inputs = int(args.fold), args.inputs
  out = model_path(inputs, f, args.tag)
  if out.exists() and not args.force:
    print(f'{out} exists', flush=True); return
  steps = int(args.steps or FIT['steps'])
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  xy, off, L, oc, ep, t = load_branches()
  assert np.array_equal(ep, anchors.episode) and np.array_equal(t, anchors.t)
  fold, es = folds(anchors)
  tr = np.where((fold != f) & ~es)[0]; va = np.where((fold != f) & es)[0]
  X = features(anchors, inputs)
  mean = X[tr].mean(axis=0); std = np.maximum(X[tr].std(axis=0), 1e-3)
  Xn = ((X - mean) / std).astype(np.float32)
  oc_idx = np.array([OUTCOMES.index(o) for o in oc], np.int64)
  n_fut = L - 1
  assert np.all(n_fut >= 1)

  def draw(rng, pool, cdf, B):
    k = pool[np.minimum(np.searchsorted(cdf, rng.random(B), side='right'), len(pool) - 1)]
    n = n_fut[k]
    m = trunc_geom(rng, n)
    cm = cell_of(xy[off[k] + m])
    ch = [cell_of(xy[off[k] + np.minimum(h, n)]) for h in HORIZONS]
    return Xn[k], cm, ch, oc_idx[k]

  def cdf_of(pool):
    w = anchors.weight[pool]; c = np.cumsum(w / w.sum()); c[-1] = 1.0; return c
  cdf_tr, cdf_va = cdf_of(tr), cdf_of(va)
  rng_val = np.random.default_rng(FIT['val_seed'] + 10 * f + INPUT_SETS.index(inputs))
  VX, Vcm, Vch, Vco = draw(rng_val, va, cdf_va, FIT['val_pairs'])
  rng = np.random.default_rng(FIT['sample_seed'] + 10 * f + INPUT_SETS.index(inputs))
  net = build(X.shape[1])
  params = net.init(jax.random.PRNGKey(100 * f + INPUT_SETS.index(inputs)), jnp.zeros((1, X.shape[1]), jnp.float32))
  opt = optax.adam(FIT['lr']); opt_state = opt.init(params)

  def loss_fn(p, x, cm, ch, co):
    o = net.apply(p, x)
    ce = lambda logits, y: -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), y[:, None], axis=1)[:, 0])
    parts = {'marginal': ce(o['marginal'], cm), 'outcome': ce(o['outcome'], co)}
    for i, h in enumerate(HORIZONS):
      parts[f'h{h}'] = ce(o[f'h{h}'], ch[i])
    return sum(parts.values()), parts

  @jax.jit
  def step(p, os_, x, cm, ch, co):
    (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, cm, ch, co)
    up, os_ = opt.update(g, os_, p)
    return optax.apply_updates(p, up), os_, l, parts

  @jax.jit
  def val_parts(p, x, cm, ch, co):
    return loss_fn(p, x, cm, ch, co)[1]

  def validate(p):
    acc = {}
    n = 0
    for i in range(0, len(VX), 8192):
      sl = slice(i, i + 8192)
      parts = val_parts(p, jnp.asarray(VX[sl]), jnp.asarray(Vcm[sl]), [jnp.asarray(c[sl]) for c in Vch], jnp.asarray(Vco[sl]))
      w = len(VX[sl]); n += w
      for k, v in parts.items():
        acc[k] = acc.get(k, 0.0) + float(v) * w
    return {k: v / n for k, v in acc.items()}
  hist, best, best_params, best_step = [], float('inf'), None, 0
  t0 = time.time()
  for it in range(1, steps + 1):
    x, cm, ch, co = draw(rng, tr, cdf_tr, FIT['batch'])
    params, opt_state, l, parts = step(params, opt_state, jnp.asarray(x), jnp.asarray(cm), [jnp.asarray(c) for c in ch], jnp.asarray(co))
    if it % FIT['val_every'] == 0 or it == steps:
      v = validate(params)
      hist.append({'step': it, 'train_loss': float(l), 'train': {k: float(x_) for k, x_ in parts.items()}, 'val': v})
      if v['marginal'] < best:
        best, best_step = v['marginal'], it
        best_params = jax.tree_util.tree_map(np.asarray, params)
      print(f'[fit {inputs} fold {f} step {it:>6}] train {float(l):.3f} val marginal {v["marginal"]:.4f} outcome {v["outcome"]:.4f} '
            + ' '.join(f'h{h} {v[f"h{h}"]:.3f}' for h in HORIZONS) + f' best {best:.4f}@{best_step} {it / (time.time() - t0):.1f} it/s', flush=True)
  md = {'params': best_params, 'norm': {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}, 'inputs': inputs, 'fold': f, 'best_step': best_step,
        'best_val_marginal_nll': best, 'steps': steps, 'fit': FIT, 'grid': {'x0': X0, 'x1': X1, 'y0': Y0, 'y1': Y1, 'cell': CELL, 'nx': NX, 'ny': NY},
        'n_train_anchors': int(len(tr)), 'n_es_anchors': int(len(va)), 'history': hist, 'wall_seconds': time.time() - t0, 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz')}
  with out.open('wb') as fh:
    pickle.dump(md, fh)
  MP.write_json(out.with_suffix('.json'), {k: v for k, v in md.items() if k not in ('params', 'norm')})
  print(f'saved {out}: best val marginal NLL {best:.4f} at step {best_step} ({time.time() - t0:.0f} s)', flush=True)


# --------------------------------------------------------------- generate
def auc_rank(score, pos):
  """AUC by ranks (ties averaged)."""
  from scipy.stats import rankdata
  r = rankdata(score)
  n1, n0 = int(pos.sum()), int((~pos).sum())
  if n1 == 0 or n0 == 0:
    return float('nan')
  return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mode_generate(args):
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  xy, off, L, oc, ep, t = load_branches()
  assert np.array_equal(ep, anchors.episode) and np.array_equal(t, anchors.t)
  fold, es = folds(anchors)
  K = anchors.n
  # oracle side: exact truncated-geometric weights over the branch rows
  anchor_of_row, m_of_row, w_row = geom_weights(L)
  cell_row = cell_of(xy)
  reg_row = CELL_REGION[cell_row]
  oracle_region = np.zeros((K, len(REGIONS)))
  for r in range(len(REGIONS)):
    oracle_region[:, r] = np.bincount(anchor_of_row, weights=w_row * (reg_row == r), minlength=K)
  oracle_h = {h: cell_row[off + np.minimum(h, L - 1)] for h in HORIZONS}
  oracle_death = (oc == 'death')
  oc_idx = np.array([OUTCOMES.index(o) for o in oc])
  cxy = cells_xy()
  region_ind = np.zeros((NCELL, len(REGIONS)), np.float32)
  region_ind[np.arange(NCELL), CELL_REGION] = 1.0
  man = MP.read_json(OUT / 'manifest.json')
  detour_ids = set(int(e) for e in MP.read_json(MP.OUT / 'manifest.json')['dataset']['selection']['detour_episode_ids'])
  strata = {'all': np.ones(K, bool), 'reset (t = 0)': anchors.t == 0, 'detour-episode anchors': np.array([int(e) in detour_ids for e in anchors.source_episode]),
            'shortcut-episode anchors': np.array([int(e) not in detour_ids for e in anchors.source_episode])}
  own = region_of(anchors.state[:, :2])
  for r, name in enumerate(REGIONS):
    if (own == r).sum() >= 50:
      strata[f'anchors in {name}'] = own == r
  W = anchors.weight / anchors.weight.sum()
  results, probs_full, outcome_full, model_meta = {}, None, None, {}
  for inputs in INPUT_SETS:
    X = features(anchors, inputs)
    logp = {}
    for f in range(N_FOLDS):
      md = load_model(model_path(inputs, f))
      assert md['inputs'] == inputs and md['fold'] == f
      sel = np.where(fold == f)[0]
      o = apply_model(md, X[sel])
      for k, v in o.items():
        logp.setdefault(k, np.zeros((K, v.shape[1]), np.float32))[sel] = v
      model_meta[f'{inputs}_fold{f}'] = {'path': str(model_path(inputs, f)), 'sha256': MP.sha256(model_path(inputs, f)), 'best_step': md['best_step'], 'best_val_marginal_nll': md['best_val_marginal_nll']}
    probs = np.exp(logp['marginal'])
    # 1. exact held-out NLL of the oracle path under the marginal
    nll_row = -logp['marginal'][anchor_of_row, cell_row]
    nll = np.bincount(anchor_of_row, weights=w_row * nll_row, minlength=K)
    # 2. region masses
    model_region = probs @ region_ind
    l1 = np.abs(model_region - oracle_region).sum(axis=1)
    # 3. death, zones, far route
    p_death = np.exp(logp['outcome'][:, OUTCOMES.index('death')])
    zone_m, zone_o = model_region[:, ZONES].sum(axis=1), oracle_region[:, ZONES].sum(axis=1)
    far_m, far_o = model_region[:, FAR].sum(axis=1), oracle_region[:, FAR].sum(axis=1)
    goal_m, goal_o = model_region[:, REGIONS.index('goal_area')], oracle_region[:, REGIONS.index('goal_area')]
    # 4. horizon heads
    hz = {}
    for h in HORIZONS:
      lp = logp[f'h{h}']
      nll_h = -lp[np.arange(K), oracle_h[h]]
      ex = np.exp(lp) @ cxy
      dist = np.linalg.norm(ex - cxy[oracle_h[h]], axis=1)
      amax = np.linalg.norm(cxy[lp.argmax(axis=1)] - cxy[oracle_h[h]], axis=1)
      hz[h] = {'nll': nll_h, 'dist_expected': dist, 'dist_argmax': amax}
    res = {}
    for sname, sm in strata.items():
      w = W[sm] / W[sm].sum()
      r = {'n': int(sm.sum()), 'weight_share': float(W[sm].sum()), 'marginal_nll': float((w * nll[sm]).sum()), 'region_l1': float((w * l1[sm]).sum()),
           'death_auc': auc_rank(p_death[sm], oracle_death[sm]), 'p_death_model_mean': float((w * p_death[sm]).sum()), 'death_rate_oracle': float((w * oracle_death[sm]).sum()),
           'outcome_accuracy': float((w * (logp['outcome'][sm].argmax(axis=1) == oc_idx[sm])).sum()),
           'zone_mass_model': float((w * zone_m[sm]).sum()), 'zone_mass_oracle': float((w * zone_o[sm]).sum()), 'zone_mass_l1': float((w * np.abs(zone_m - zone_o)[sm]).sum()),
           'far_mass_model': float((w * far_m[sm]).sum()), 'far_mass_oracle': float((w * far_o[sm]).sum()), 'far_mass_l1': float((w * np.abs(far_m - far_o)[sm]).sum()),
           'goal_mass_model': float((w * goal_m[sm]).sum()), 'goal_mass_oracle': float((w * goal_o[sm]).sum()),
           'horizons': {h: {k: float((w * v[sm]).sum()) for k, v in hz[h].items()} for h in HORIZONS}}
      res[sname] = r
    # per-fold marginal NLL (each fold = one model)
    res['per_fold_marginal_nll'] = [float((W[fold == f] / W[fold == f].sum() * nll[fold == f]).sum()) for f in range(N_FOLDS)]
    results[inputs] = res
    if inputs == 'full':
      probs_full, outcome_full = probs, np.exp(logp['outcome'])
  # the futures file (the full model only)
  meta = {'note': 'ORACLE-LABEL-TRAINED learned ETT: the discounted-future-goal marginal of the blind continuation, fitted on the pilot\'s simulator branches, cross-fitted by source episode',
          'grid': {'x0': X0, 'x1': X1, 'y0': Y0, 'y1': Y1, 'cell': CELL, 'nx': NX, 'ny': NY}, 'gamma': GAMMA, 'models': {k: v for k, v in model_meta.items() if k.startswith('full')},
          'fold_seed': FOLD_SEED, 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'), 'branches_sha256': MP.sha256(MP.OUT / 'branches_cf.npz')}
  np.savez_compressed(OUT / 'futures_learned.npz', episode=anchors.episode, t=anchors.t, fold=fold, probs=probs_full.astype(np.float16), outcome_probs=outcome_full.astype(np.float32),
                      cells_xy=cxy, cell=np.float32(CELL), meta=np.asarray(json.dumps(meta, sort_keys=True)))
  check = {'models': model_meta, 'results': results, 'futures_sha256': MP.sha256(OUT / 'futures_learned.npz'), 'regions': list(REGIONS), 'strata_n': {k: int(v.sum()) for k, v in strata.items()}}
  MP.write_json(OUT / 'check.json', check)
  write_check_md(check)
  print(json.dumps({k: {s: {kk: results[k][s][kk] for kk in ('marginal_nll', 'region_l1', 'death_auc')} for s in ('all', 'reset (t = 0)')} for k in results}, indent=1), flush=True)


def write_check_md(check):
  R = check['results']
  L = ['# Oracle-label-trained learned ETT: checks against the oracle branches (cross-fitted, every anchor scored by a model that never saw its episode)', '',
       f'`check.json`.  Marginal NLL = exact expectation over the oracle path under the truncated geometric law of -log p(cell); region L1 = sum over the {len(REGIONS)} maze regions of |model mass - oracle mass|; '
       'death AUC = the outcome head\'s P(death) ranking the branches that died; masses are anchor-weighted means.  "full" = state + torque input (the model the arm uses); "xy" = position-only reference.', '',
       '## Held-out likelihood and region agreement by stratum', '',
       '| stratum | n | model | marginal NLL | region L1 | death AUC | P(death) model / oracle | zone mass model / oracle | far-route mass model / oracle | goal-area mass model / oracle |',
       '|---|---:|---|---:|---:|---:|---|---|---|---|']
  for s in R['full']:
    if s == 'per_fold_marginal_nll':
      continue
    for inputs in INPUT_SETS:
      r = R[inputs][s]
      L.append(f'| {s} | {r["n"]} | {inputs} | {r["marginal_nll"]:.3f} | {r["region_l1"]:.3f} | {r["death_auc"]:.3f} | {r["p_death_model_mean"]:.3f} / {r["death_rate_oracle"]:.3f} | '
               f'{r["zone_mass_model"]:.3f} / {r["zone_mass_oracle"]:.3f} | {r["far_mass_model"]:.3f} / {r["far_mass_oracle"]:.3f} | {r["goal_mass_model"]:.3f} / {r["goal_mass_oracle"]:.3f} |')
  L += ['', 'Per-fold marginal NLL (full): ' + ' / '.join(f'{v:.3f}' for v in R['full']['per_fold_marginal_nll']) + '; (xy): ' + ' / '.join(f'{v:.3f}' for v in R['xy']['per_fold_marginal_nll']), '',
        '## Horizon heads (absorbing position m steps after the query), all anchors', '', '| model | m | NLL | expected-position error | argmax-cell error |', '|---|---:|---:|---:|---:|']
  for inputs in INPUT_SETS:
    for h in HORIZONS:
      q = R[inputs]['all']['horizons'][str(h)] if str(h) in R[inputs]['all']['horizons'] else R[inputs]['all']['horizons'][h]
      L.append(f'| {inputs} | {h} | {q["nll"]:.3f} | {q["dist_expected"]:.2f} | {q["dist_argmax"]:.2f} |')
  L += ['', '## Models', '', '| model | best step | best held-out marginal NLL (early-stop slice) |', '|---|---:|---:|']
  for k, v in check['models'].items():
    L.append(f'| {k} | {v["best_step"]} | {v["best_val_marginal_nll"]:.4f} |')
  (OUT / 'CHECK.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ------------------------------------------------------------------- arm
def mode_train(args):
  if not (OUT / 'futures_learned.npz').exists():
    raise SystemExit('generate first')
  for s in args.seeds:
    MP.train_arm('CFL', s, base=OUT, overrides=dict(MP.VARIANTS[RECIPE]), inputs=MP.OUT, branch_path=OUT / 'futures_learned.npz')


def mode_evaluate(args):
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = MP.EVAL['seed'], MP.EVAL['n']
  for s in args.seeds:
    ck = OUT / 'CFL' / f'seed_{s}' / 'final.pkl'
    if not ck.exists():
      print(f'skip seed {s}: no final checkpoint', flush=True); continue
    print(f'== evaluate CFL/seed_{s}', flush=True)
    MP.D.evaluate_ckpt(ck, ck.parent, MP.EVAL['policy'])


def mode_report(args):
  vb = MP.variant_base(RECIPE); es = MP.EVAL['seed']
  paths = {'start': MP.OUT / 'start_agent' / f'eval_mean_s{es}.json'}
  for arm in MP.ARMS:
    for s in MP.SEEDS:
      paths[f'{arm}/seed_{s}'] = MP.run_dir(arm, s, vb) / f'eval_mean_s{es}.json'
  for s in MP.SEEDS:
    paths[f'CFL/seed_{s}'] = OUT / 'CFL' / f'seed_{s}' / f'eval_mean_s{es}.json'
  missing = [k for k, p in paths.items() if not p.exists()]
  if missing:
    raise SystemExit(f'missing evaluations: {missing}')
  E = {k: MP._episodes(p) for k, p in paths.items()}
  raw = {k: MP.read_json(p)['episodes'] for k, p in paths.items()}
  check = MP.read_json(OUT / 'check.json') if (OUT / 'check.json').exists() else None
  L = ['# Arm CF-learned (oracle-label-trained learned ETT) vs the clipped O and CF (oracle) arms', '',
       f'`manifest.json` (sealed before fitting).  The critic\'s positive futures of arm CFL come from the cross-fitted learned marginal (`futures_learned.npz`, `CHECK.md`); everything else is the sealed '
       f'{RECIPE} recipe.  Same {MP.EVAL["n"]} evaluation episodes (seed {es}, mode) as the clipped O / CF arms.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3/3.  '
       'Status: an engineering intermediate experiment on the interface (the ETT model saw simulator labels); not the offline identification.', '',
       '## Per policy', '', '| policy | success | detour | death | timeout | success no hazard | success hazard | mean steps |', '|---|---:|---:|---:|---:|---:|---:|---:|']
  for k, e in E.items():
    h = MP._headline(e)
    L.append(f'| {k} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_no_hazard"]:.3f} | {h["success_hazard"]:.3f} | {h["mean_steps"]:.0f} |')
  by = {arm: {s: E[f'{arm}/seed_{s}'] for s in MP.SEEDS} for arm in ('O', 'CF', 'CFL')}
  comps = [('CFL - O [primary]', by['CFL'], by['O']), ('CFL - start [practical]', by['CFL'], E['start']), ('CFL - CF(oracle) [kept?]', by['CFL'], by['CF']), ('CF(oracle) - O [reference]', by['CF'], by['O'])]
  res = {}
  L += ['', '## Paired differences on the common episodes (per seed; seed mean, seed s.e., episode-bootstrap s.e.)', '']
  for key in ('success', 'detour', 'failure', 'timeout'):
    L += [f'### {key}', '', '| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |', '|---|---|---:|---:|---:|---|---|']
    for cname, a, b in comps:
      r = MP.paired_block(a, b, key=key); res[f'{cname}:{key}'] = r
      per = ' / '.join(f'{v["mean"]:+.3f}' for v in r['per_seed'].values())
      rule = ('met' if r['improvement_rule_met'] else 'not met') if (key == 'success' and ('primary' in cname or 'practical' in cname)) else '-'
      L.append(f'| {cname} | {per} | {r["mean"]:+.3f} | {r["seed_se"]:.3f} | {r["episode_bootstrap_se_of_mean"]:.3f} | {r["seeds_same_direction"]} | {rule} |')
    L.append('')
  L += ['## Route ledger', '', *MP.ledger_table(raw), '']
  p, q, kept = res['CFL - O [primary]:success'], res['CFL - start [practical]:success'], res['CFL - CF(oracle) [kept?]:success']
  ref = res['CF(oracle) - O [reference]:success']
  L += ['## Reading', '',
        f'Primary CFL - O success: {p["mean"]:+.3f} (seed s.e. {p["seed_se"]:.3f}, boot {p["episode_bootstrap_se_of_mean"]:.3f}, {p["seeds_same_direction"]}) -> {"MET" if p["improvement_rule_met"] else "NOT MET"}; '
        f'practical CFL - start {q["mean"]:+.3f} ({q["seed_se"]:.3f}, {q["seeds_same_direction"]}) -> {"MET" if q["improvement_rule_met"] else "NOT MET"}.  '
        f'Against the oracle arm: CFL - CF {kept["mean"]:+.3f} (seed s.e. {kept["seed_se"]:.3f}, {kept["seeds_same_direction"]}); the oracle gain on these episodes is CF - O {ref["mean"]:+.3f} ({ref["seed_se"]:.3f}).', '']
  if check:
    a = check['results']['full']['all']; b = check['results']['xy']['all']
    L.append(f'ETT model checks (`CHECK.md`): held-out marginal NLL {a["marginal_nll"]:.3f} (position-only reference {b["marginal_nll"]:.3f}), region L1 {a["region_l1"]:.3f}, death AUC {a["death_auc"]:.3f}, '
             f'far-route mass {a["far_mass_model"]:.3f} vs oracle {a["far_mass_oracle"]:.3f}, zone mass {a["zone_mass_model"]:.3f} vs {a["zone_mass_oracle"]:.3f}.')
  L += ['', 'Oracle-label-trained learned ETT under the disclosed optimizer-stabilisation recipe; not the original learner, not the offline identification.']
  MP.write_json(OUT / 'results.json', {'headlines': {k: MP._headline(e) for k, e in E.items()}, 'paired': res, 'ledger': {k: MP.route_ledger(r) for k, r in raw.items()}})
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'fit', 'generate', 'train', 'evaluate', 'report'))
  ap.add_argument('--fold', type=int, default=0)
  ap.add_argument('--inputs', choices=INPUT_SETS, default='full')
  ap.add_argument('--steps', type=int, default=None, help='fit: override the step count (smoke only)')
  ap.add_argument('--tag', default='', help='fit: model file suffix (smoke only)')
  ap.add_argument('--seeds', type=int, nargs='+', default=list(MP.SEEDS))
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  {'seal': mode_seal, 'fit': mode_fit, 'generate': mode_generate, 'train': mode_train, 'evaluate': mode_evaluate, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
