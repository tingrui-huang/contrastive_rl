"""Diagonal OBSERVATIONAL transition diagnostic for the z_v1 PointMaze.

Observational and diagonal only. No off-diagonal action intervention, no
Lipschitz, no Flow, no worst-case candidate generation, no policy change.

THE QUESTION. For a still-alive learner-visible state and the LOGGED action,
what one-step outcome follows? The answer is deliberately factorised into two
conceptually different objects rather than one 3-D regressor:

    XY    Delta_xy = (x' - x, y' - y)      a smooth, near-deterministic map
    SINK  Y = 1[z_t = 0 and z_{t+1} < 0]   a Bernoulli event

Fitting a single 3-D MSE head over (dx, dy, dz) would average the two outcomes
of the sink event into a nonexistent "half-sunk" state, which is exactly the
failure mode this split avoids. Whether the split is warranted is what section
15 decides, from evidence rather than assertion.

WHAT p_sink IS NOT. Action selection in this dataset is confounded by the
hidden swamp bit: the teacher dodges when it can see danger. So

    p_sink(s, a')  =  P_obs(sink next step | s, a')

is an OBSERVATIONAL conditional, not P(sink | do(a)). Nothing here licenses a
causal reading, and none is claimed.

D_alive. Only transitions whose SOURCE is on the ground, z_t = 0. That keeps
every ordinary transition, every clear-swamp entry and all first-contact fatal
transitions (0 -> -0.12), and drops the post-contact sinking chain
(-0.12 -> -0.24 -> ... -> -0.5) and the settled tails, whose outcome is
mechanical and would swamp the statistics.

NATURAL CLASS DISTRIBUTION IS PRESERVED. No class weighting, no oversampling,
no balanced batches, no focal loss, no synthetic positives -- otherwise
p_sink would stop being an observational probability. If the imbalance causes
collapse, that IS the result and it is reported as such; a balanced variant
would be a separate, separately-labelled ablation.

LEARNER-VISIBLE INPUTS ONLY. swamp_bits, the active/clear label, _dead, the
teacher's privileged view and the env's internal action noise are never model
inputs. swamp_bits is loaded solely for the section-11 audit, and the observed
next XY is used only to DEFINE the at-risk subset, never as a feature.

Metrics (AUROC, average precision, Brier, log loss, ECE) are implemented in
numpy: sklearn is not in the validated node environment and adding it for
four formulas is not worth the provenance churn.

Usage:
  python scripts/diag_transition_z_v1.py
  python scripts/diag_transition_z_v1.py --epochs 60 --seed 0
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.20')

import haiku as hk                                # noqa: E402
import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import optax                                      # noqa: E402

from diag_transition_mlp import split_episodes    # noqa: E402
from diag_action_lipschitz import wall_margin     # noqa: E402

DATASET = 'datasets/swamp_windy_z_v1_merged_s0.npz'
OUT_DIR = 'artifacts/swamp_windy_z_v1_transition_diag'
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
HIDDEN = (256, 256)
Z_ENTRY = -0.12


# --------------------------------------------------------------------------- #
# metrics (numpy; sklearn is not in the node environment)
# --------------------------------------------------------------------------- #
def auroc(y, p):
  y = np.asarray(y).astype(bool)
  if y.all() or not y.any():
    return float('nan')
  r = np.empty(len(p), float)
  order = np.argsort(p, kind='mergesort')
  sp = p[order]
  # average ranks over ties, so exact duplicates cannot inflate the score
  i = 0
  ranks = np.empty(len(p), float)
  while i < len(sp):
    j = i
    while j + 1 < len(sp) and sp[j + 1] == sp[i]:
      j += 1
    ranks[i:j + 1] = 0.5 * (i + j) + 1.0
    i = j + 1
  r[order] = ranks
  n1 = int(y.sum())
  n0 = len(y) - n1
  return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, p):
  """AUPRC as average precision (the step-wise sum, not trapezoid)."""
  y = np.asarray(y).astype(bool)
  if not y.any():
    return float('nan')
  o = np.argsort(-p, kind='mergesort')
  ys = y[o]
  tp = np.cumsum(ys)
  prec = tp / np.arange(1, len(ys) + 1)
  return float((prec * ys).sum() / ys.sum())


def log_loss(y, p, eps=1e-12):
  p = np.clip(p, eps, 1 - eps)
  return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
  return float(np.mean((p - y) ** 2))


def calibration(y, p, n_bins=10, by_quantile=True):
  """Bin table + ECE. Quantile bins by default: with prevalence <1% a fixed
  0-1 grid puts essentially every sample in one bin and says nothing."""
  if by_quantile:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
      edges = np.linspace(0, 1, n_bins + 1)
  else:
    edges = np.linspace(0, 1, n_bins + 1)
  rows, ece = [], 0.0
  for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    m = (p >= lo) & (p <= hi if i == len(edges) - 2 else p < hi)
    if not m.any():
      continue
    rows.append({'lo': float(lo), 'hi': float(hi), 'n': int(m.sum()),
                 'mean_pred': float(p[m].mean()),
                 'empirical': float(y[m].mean())})
    ece += m.mean() * abs(p[m].mean() - y[m].mean())
  return rows, float(ece)


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90)),
          'p95': float(np.percentile(v, 95)),
          'p99': float(np.percentile(v, 99)),
          'min': float(v.min()), 'max': float(v.max())}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_alive(path):
  with np.load(path, allow_pickle=False) as d:
    obs, act, bits = d['obs'], d['act'], d['swamp_bits']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
    mode = np.asarray(d['teacher_mode'])
  n, L, W = obs.shape
  assert W == 6, 'expected obs width 6 ([x,y,z,gx,gy,gz]), got %d' % W
  xyz = obs[:, :, :3]
  s = xyz[:, :-1, :].reshape(-1, 3)
  sn = xyz[:, 1:, :].reshape(-1, 3)
  a = act[:, :-1, :].reshape(-1, 2)
  ep = np.repeat(np.arange(n)[:, None], L - 1, axis=1).reshape(-1)
  # bits governing each transition (AUDIT ONLY -- never a model input)
  bt = bits[:, :-1, :].reshape(-1, 3)

  alive = s[:, 2] == 0.0                      # source on the ground
  sink = alive & (sn[:, 2] < 0.0)             # first contact
  cell = np.clip(np.floor(sn[:, :2]).astype(int), 0, [8, 4])
  lands = -np.ones(len(sn), int)
  for k, (cx, cy) in enumerate(SWAMP_CELLS):
    lands[(cell[:, 0] == cx) & (cell[:, 1] == cy)] = k
  return dict(s=s, s_next=sn, a=a, ep=ep, bits=bt, alive=alive, sink=sink,
              lands=lands, ep_mode=mode, ep_died=died, n_eps=n, L=L,
              delta_xy=(sn[:, :2] - s[:, :2]))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def make_net(out_dim):
  def f(x):
    return hk.nets.MLP([HIDDEN[0], HIDDEN[1], out_dim],
                       activation=jax.nn.relu)(x)
  return hk.without_apply_rng(hk.transform(f))


def train_net(x_tr, y_tr, x_va, y_va, out_dim, loss_kind, seed, epochs, batch,
              lr, label):
  """Shared trainer. loss_kind 'mse' (XY) or 'bce' (sink, UNWEIGHTED)."""
  net = make_net(out_dim)
  mu, sd = x_tr.mean(0), x_tr.std(0)
  sd = np.where(sd < 1e-6, 1.0, sd)          # z_t is constant in D_alive

  def nx(x):
    return ((x - mu) / sd).astype(np.float32)

  xt, xv = nx(x_tr), nx(x_va)
  params = net.init(jax.random.PRNGKey(seed), jnp.asarray(xt[:2]))
  opt = optax.adam(lr)
  ostate = opt.init(params)

  def loss_fn(p, x, y):
    o = net.apply(p, x)
    if loss_kind == 'mse':
      return jnp.mean(jnp.sum((o - y) ** 2, axis=-1))
    return jnp.mean(optax.sigmoid_binary_cross_entropy(o[:, 0], y))

  @jax.jit
  def step(p, o, x, y):
    l, g = jax.value_and_grad(loss_fn)(p, x, y)
    u, o = opt.update(g, o)
    return optax.apply_updates(p, u), o, l

  @jax.jit
  def vloss(p, x, y):
    return loss_fn(p, x, y)

  rng = np.random.default_rng(seed)
  n = len(xt)
  jxv, jyv = jnp.asarray(xv), jnp.asarray(y_va)
  best, bp, bep, hist = np.inf, params, -1, []
  for e in range(epochs):
    perm = rng.permutation(n)
    for k in range(0, n - batch + 1, batch):
      sl = perm[k:k + batch]
      params, ostate, _ = step(params, ostate, jnp.asarray(xt[sl]),
                               jnp.asarray(y_tr[sl]))
    v = float(vloss(params, jxv, jyv))
    hist.append(v)
    if v < best:
      best, bp, bep = v, params, e
    if e % 10 == 0 or e == epochs - 1:
      print('    [%s] epoch %3d  val %.6f%s' % (label, e, v,
                                                '  *' if bep == e else ''))
  print('    [%s] best val %.6f @ %d' % (label, best, bep))

  def predict(x):
    o = np.asarray(net.apply(bp, jnp.asarray(nx(x))))
    return o if loss_kind == 'mse' else 1.0 / (1.0 + np.exp(-o[:, 0]))
  bundle = {'params': jax.tree_util.tree_map(np.asarray, bp), 'mu': mu,
            'sd': sd, 'out_dim': out_dim, 'loss': loss_kind,
            'best_val': best, 'best_epoch': bep, 'history': hist}
  return predict, bundle


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--dataset', default=DATASET)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--epochs', type=int, default=40)
  ap.add_argument('--sink-epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(['git'] + list(a),
                                     cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''
  import hashlib

  def csha(p):
    h = hashlib.sha256()
    with np.load(p, allow_pickle=False) as d:
      for k in sorted(d.files):
        x = d[k]
        h.update(k.encode()); h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode()); h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

  out = {'analysis_script': 'scripts/diag_transition_z_v1.py',
         'dataset': args.dataset, 'dataset_content_sha256': csha(args.dataset),
         'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
         'head_at_runtime': git('rev-parse', 'HEAD'),
         'dirty': bool(git('status', '--porcelain', '--', 'crl', 'scripts')),
         'config': {k: getattr(args, k) for k in
                    ('seed', 'epochs', 'sink_epochs', 'batch', 'lr')}}
  print('=' * 96)
  print('Z-v1 DIAGONAL OBSERVATIONAL TRANSITION DIAGNOSTIC')
  print('=' * 96)
  print('  dataset %s' % args.dataset)
  print('  content sha %s' % out['dataset_content_sha256'])
  print('  code commit %s   dirty %s' % (out['code_commit'], out['dirty']))

  # ------------------------------------------------------------------ 2
  D = build_alive(args.dataset)
  al = D['alive']
  print('\n2. D_alive')
  print('  all transitions            %s' % format(len(al), ','))
  print('  z_t = 0  (D_alive)         %s' % format(int(al.sum()), ','))
  print('  excluded (z_t < 0)         %s' % format(int((~al).sum()), ','))
  print('  first-contact sinks in it  %s' % format(int(D['sink'].sum()), ','))
  print('  swamp-landing in it        %s'
        % format(int((al & (D['lands'] >= 0)).sum()), ','))
  assert D['sink'].sum() == int(D['ep_died'].sum()), (
      'first-contact count != failed-episode count')
  assert np.allclose(D['s_next'][D['sink'], 2], Z_ENTRY, atol=1e-3), (
      'first-contact z is not -0.12 -- wrong dataset version?')
  out['2_alive'] = {'all_transitions': int(len(al)),
                    'alive_transitions': int(al.sum()),
                    'excluded_z_negative_source': int((~al).sum()),
                    'first_contact_sinks': int(D['sink'].sum()),
                    'swamp_landing_alive': int((al & (D['lands'] >= 0)).sum())}

  # ------------------------------------------------------------------ 3
  which = split_episodes(D['ep_mode'], D['ep_died'], args.seed)
  spl = which[D['ep']]
  print('\n3. EPISODE-LEVEL SPLIT (stratified on teacher_mode x died)')
  print('  %-8s%9s%12s%14s%16s' % ('split', 'eps', 'trans', 'first-contact',
                                   'prevalence'))
  sel = {}
  for sp in ('train', 'val', 'test'):
    m = (spl == sp) & al
    sel[sp] = np.where(m)[0]
    ns = int(D['sink'][m].sum())
    print('  %-8s%9d%12s%14d%16.6f'
          % (sp, int((which == sp).sum()), format(int(m.sum()), ','), ns,
             ns / max(int(m.sum()), 1)))
  out['3_split'] = {sp: {'episodes': int((which == sp).sum()),
                         'transitions': int(len(sel[sp])),
                         'first_contact': int(D['sink'][sel[sp]].sum())}
                    for sp in sel}

  # ------------------------------------------------------------------ 4/6
  x_sa = np.concatenate([D['s'][:, :2], D['a']], axis=1).astype(np.float32)
  x_s = D['s'][:, :2].astype(np.float32)
  y_xy = D['delta_xy'].astype(np.float32)
  print('\n4A. XY MODEL  (x,y,ax,ay) -> delta_xy   [2x256 ReLU, MSE]')
  p_sa, b_sa = train_net(x_sa[sel['train']], y_xy[sel['train']],
                         x_sa[sel['val']], y_xy[sel['val']], 2, 'mse',
                         args.seed, args.epochs, args.batch, args.lr, 'xy_sa')
  p_s, b_s = train_net(x_s[sel['train']], y_xy[sel['train']],
                       x_s[sel['val']], y_xy[sel['val']], 2, 'mse',
                       args.seed, args.epochs, args.batch, args.lr, 'xy_s')

  te = sel['test']
  err = {'zero_delta': np.linalg.norm(y_xy[te], axis=1),
         'state_only': np.linalg.norm(p_s(x_s[te]) - y_xy[te], axis=1),
         'state_action': np.linalg.norm(p_sa(x_sa[te]) - y_xy[te], axis=1)}
  print('\n6. XY TEST ERROR  ||pred - true||')
  print('  %-16s%9s%9s%9s%9s%9s%9s' % ('', 'mean', 'median', 'p90', 'p95',
                                       'p99', 'max'))
  for k, v in err.items():
    d = dist(v)
    print('  %-16s%9.4f%9.4f%9.4f%9.4f%9.4f%9.4f'
          % (k, d['mean'], d['median'], d['p90'], d['p95'], d['p99'],
             d['max']))
  out['6_xy'] = {k: dist(v) for k, v in err.items()}

  lands_te = D['lands'][te]
  sink_te = D['sink'][te]
  marg = wall_margin(D['s'][te][:, :2])
  strata = {'ordinary_non_swamp': (lands_te < 0),
            'clear_swamp_entry': (lands_te >= 0) & ~sink_te,
            'first_contact_fatal': sink_te,
            'near_wall (margin<0.25)': marg < 0.25}
  print('\n  stratified (state_action)')
  print('  %-26s%9s%9s%9s%9s' % ('', 'n', 'mean', 'median', 'p99'))
  out['6_xy_strata'] = {}
  for nm, m in strata.items():
    d = dist(err['state_action'][m])
    out['6_xy_strata'][nm] = d
    if d['n']:
      print('  %-26s%9s%9.4f%9.4f%9.4f'
            % (nm, format(d['n'], ','), d['mean'], d['median'], d['p99']))
  ce = out['6_xy_strata']['clear_swamp_entry']
  fe = out['6_xy_strata']['first_contact_fatal']
  print('  clear vs fatal entry XY error: %.4f vs %.4f  (ratio %.2f) -- their'
        % (ce['mean'], fe['mean'], fe['mean'] / max(ce['mean'], 1e-12)))
  print('  XY physics is identical BY DESIGN, so these should be comparable.')

  # ------------------------------------------------------------------ 4B/7
  y_sk = D['sink'].astype(np.float32)
  print('\n4B. SINK MODEL  (x,y,ax,ay) -> P(sink)  [2x256 ReLU, plain BCE,')
  print('    NATURAL class distribution: no weighting/oversampling/focal]')
  print('    train prevalence %.6f (%d positives of %s)'
        % (y_sk[sel['train']].mean(), int(y_sk[sel['train']].sum()),
           format(len(sel['train']), ',')))
  p_sink, b_sink = train_net(x_sa[sel['train']], y_sk[sel['train']],
                             x_sa[sel['val']], y_sk[sel['val']], 1, 'bce',
                             args.seed, args.sink_epochs, args.batch, args.lr,
                             'sink')
  ps_te = p_sink(x_sa[te])
  yv = y_sk[te]
  prev = float(yv.mean())
  g = {'n': int(len(yv)), 'positives': int(yv.sum()),
       'negatives': int((1 - yv).sum()), 'prevalence': prev,
       'auroc': auroc(yv, ps_te), 'auprc': average_precision(yv, ps_te),
       'auprc_baseline_prevalence': prev,
       'auprc_lift': average_precision(yv, ps_te) / max(prev, 1e-12),
       'log_loss': log_loss(yv, ps_te), 'brier': brier(yv, ps_te),
       'mean_pred': float(ps_te.mean())}
  print('\n7. GLOBAL SINK METRICS (whole alive test set)')
  for k in ('n', 'positives', 'negatives', 'prevalence', 'auroc', 'auprc',
            'auprc_baseline_prevalence', 'auprc_lift', 'log_loss', 'brier',
            'mean_pred'):
    print('  %-28s %s' % (k, ('%.6f' % g[k]) if isinstance(g[k], float)
                          else format(g[k], ',')))
  print('  NB global AUROC is inflated: most transitions are nowhere near a')
  print('  swamp. Section 8 is the meaningful number.')
  out['7_sink_global'] = g

  # ------------------------------------------------------------------ 8
  atrisk = lands_te >= 0            # next XY lands in a swamp cell
  ya, pa = yv[atrisk], ps_te[atrisk]
  pv = float(ya.mean())
  r = {'n': int(len(ya)), 'n_clear': int((1 - ya).sum()),
       'n_fatal': int(ya.sum()), 'fatal_prevalence': pv,
       'auroc': auroc(ya, pa), 'auprc': average_precision(ya, pa),
       'auprc_baseline_prevalence': pv,
       'auprc_lift': average_precision(ya, pa) / max(pv, 1e-12),
       'brier': brier(ya, pa), 'log_loss': log_loss(ya, pa),
       'p_clear_mean': float(pa[ya == 0].mean()),
       'p_clear_median': float(np.median(pa[ya == 0])),
       'p_fatal_mean': float(pa[ya == 1].mean()),
       'p_fatal_median': float(np.median(pa[ya == 1]))}
  print('\n8. PRIMARY -- AT-RISK SUBSET (next XY enters a swamp cell)')
  for k, v in r.items():
    print('  %-28s %s' % (k, ('%.6f' % v) if isinstance(v, float)
                          else format(v, ',')))
  cells = D['lands'][te][atrisk]
  r['by_cell'] = {}
  print('  by swamp cell:')
  print('  %-8s%9s%9s%12s%10s%12s%12s' % ('cell', 'n', 'fatal', 'prevalence',
                                          'auroc', 'p(clear)', 'p(fatal)'))
  for k in range(3):
    m = cells == k
    if not m.any():
      continue
    yk, pk = ya[m], pa[m]
    e = {'n': int(m.sum()), 'fatal': int(yk.sum()),
         'prevalence': float(yk.mean()), 'auroc': auroc(yk, pk),
         'p_clear': float(pk[yk == 0].mean()) if (yk == 0).any() else None,
         'p_fatal': float(pk[yk == 1].mean()) if (yk == 1).any() else None}
    r['by_cell']['S%d' % k] = e
    print('  %-8s%9d%9d%12.5f%10.4f%12.5f%12.5f'
          % ('S%d' % k, e['n'], e['fatal'], e['prevalence'], e['auroc'],
             e['p_clear'] or float('nan'), e['p_fatal'] or float('nan')))
  out['8_sink_at_risk'] = r

  # ------------------------------------------------------------------ 9
  print('\n9. CALIBRATION (quantile bins; no post-hoc calibration applied)')
  out['9_calibration'] = {}
  for nm, (yy, pp) in (('full_alive_test', (yv, ps_te)),
                       ('at_risk_test', (ya, pa))):
    rows, ece = calibration(yy, pp)
    out['9_calibration'][nm] = {'bins': rows, 'ece': ece}
    print('  -- %s   ECE %.5f' % (nm, ece))
    print('     %-24s%10s%14s%14s' % ('bin', 'n', 'mean_pred', 'empirical'))
    for b in rows:
      print('     [%8.5f,%8.5f]%10s%14.5f%14.5f'
            % (b['lo'], b['hi'], format(b['n'], ','), b['mean_pred'],
               b['empirical']))

  # ------------------------------------------------------------------ 10
  print('\n10. OBSERVATIONAL AMBIGUITY in learner-visible v = (x, y, ax, ay)')
  from scipy.spatial import cKDTree
  v_all = np.concatenate([D['s'][:, :2], D['a']], axis=1)
  m_ar = al & (D['lands'] >= 0)
  V = v_all[m_ar]
  Y = D['sink'][m_ar]
  print('  swamp-entry transitions: %s  (%d fatal, %d clear)'
        % (format(len(V), ','), int(Y.sum()), int((~Y).sum())))
  print('  per-dimension spread (x, y, ax, ay): %s'
        % np.round(V.std(0), 4).tolist())
  amb = {}
  tf, tc = cKDTree(V[Y]), cKDTree(V[~Y])
  d_f2c, _ = tc.query(V[Y])        # fatal -> nearest clear
  d_c2f, _ = tf.query(V[~Y])       # clear -> nearest fatal
  for nm, dd in (('fatal_to_nearest_clear', d_f2c),
                 ('clear_to_nearest_fatal', d_c2f)):
    amb[nm] = dist(dd)
    amb[nm]['frac_within'] = {str(e): float((dd < e).mean())
                              for e in (0.01, 0.025, 0.05, 0.1)}
    d = amb[nm]
    print('  %-24s mean %.4f  median %.4f  p10 %.4f  p90 %.4f  min %.5f'
          % (nm, d['mean'], d['median'], d['p10'], d['p90'], d['min']))
    print('     within eps: ' + '  '.join(
        '%s:%.3f' % (e, d['frac_within'][e]) for e in d['frac_within']))
  # local neighbourhoods containing BOTH labels
  tree = cKDTree(V)
  mixed = {}
  for eps in (0.01, 0.025, 0.05, 0.1):
    idx = tree.query_ball_point(V, eps)
    both = np.array([Y[i].any() and (~Y[i]).any() for i in idx])
    mixed[str(eps)] = {'frac_neighbourhoods_mixed': float(both.mean()),
                       'mean_neighbourhood_size':
                           float(np.mean([len(i) for i in idx]))}
    print('  eps=%-6s mixed-label neighbourhoods %.4f   mean |N| %.1f'
          % (eps, both.mean(), np.mean([len(i) for i in idx])))
  amb['mixed_neighbourhoods'] = mixed
  out['10_ambiguity'] = amb

  # ------------------------------------------------------------------ 11
  print('\n11. HIDDEN-U AUDIT (swamp_bits used ONLY here, never as input)')
  bt = D['bits']
  lan = D['lands']
  bit_of = np.where(lan >= 0, bt[np.arange(len(bt)), np.clip(lan, 0, 2)], 0)
  fatal_bit_ok = bool(bit_of[D['sink']].all())
  clear_m = al & (lan >= 0) & ~D['sink']
  clear_bit_clear = float((bit_of[clear_m] == 0).mean())
  print('  every first-contact sink had its relevant bit ACTIVE : %s'
        % fatal_bit_ok)
  print('  clear entries whose relevant bit was CLEAR           : %.6f'
        % clear_bit_clear)
  # matched pairs: near-identical learner-visible v, opposite hidden U
  n_match = int((d_f2c < 0.05).sum())
  print('  fatal entries with a clear entry within eps=0.05 in (x,y,ax,ay): '
        '%d of %d (%.3f)' % (n_match, int(Y.sum()), n_match / max(int(Y.sum()), 1)))
  print('  => at matched learner-visible (s,a), the outcome is decided by the')
  print('     hidden bit, which is exactly the confounding this env encodes.')
  out['11_hidden_u'] = {'all_fatal_had_active_bit': fatal_bit_ok,
                        'frac_clear_entries_with_clear_bit': clear_bit_clear,
                        'fatal_with_clear_neighbour_within_0.05': n_match,
                        'n_fatal': int(Y.sum())}
  assert fatal_bit_ok, 'a first-contact sink had no active bit'

  # ------------------------------------------------------------------ 12
  print('\n12. ACTION SENSITIVITY (observational model; NOT causal)')
  probes = [('holding (2.5,3.5)', (2.5, 3.5)), ('pre-S1 (3.5,3.5)', (3.5, 3.5)),
            ('pre-S2 (4.5,3.5)', (4.5, 3.5)), ('fork (1.5,3.5)', (1.5, 3.5))]
  gg = np.linspace(-1, 1, 9)
  gx, gy = np.meshgrid(gg, gg)
  A = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
  out['12_action_sensitivity'] = {'grid': gg.tolist(), 'probes': {}}
  print('  p_sink over a 9x9 action grid; rows = a_y, cols = a_x')
  for nm, (px, py) in probes:
    S = np.repeat(np.array([[px, py]], np.float32), len(A), 0)
    pr = p_sink(np.concatenate([S, A], 1)).reshape(9, 9)
    out['12_action_sensitivity']['probes'][nm] = {
        'p_sink_grid': pr.tolist(), 'min': float(pr.min()),
        'max': float(pr.max()), 'range': float(pr.max() - pr.min())}
    print('  %-20s min %.5f  max %.5f  range %.5f  (max at a=[%+.2f,%+.2f])'
          % (nm, pr.min(), pr.max(), pr.max() - pr.min(),
             A[pr.argmax()][0], A[pr.argmax()][1]))

  # ------------------------------------------------------------------ 14
  os.makedirs(args.out_dir, exist_ok=True)
  mp = os.path.join(args.out_dir, 'models.pkl')
  with open(mp, 'wb') as f:
    pickle.dump({'xy_state_action': b_sa, 'xy_state_only': b_s,
                 'sink': b_sink, 'dataset': args.dataset,
                 'dataset_content_sha256': out['dataset_content_sha256'],
                 'config': out['config']}, f)
  sp_path = os.path.join(args.out_dir, 'split_indices.npz')
  np.savez_compressed(sp_path, episode_split=which,
                      train=sel['train'], val=sel['val'], test=sel['test'])
  import hashlib as _h
  out['14_saved'] = {}
  for nm, pth in (('models', mp), ('split_indices', sp_path)):
    with open(pth, 'rb') as f:
      out['14_saved'][nm] = {'path': pth,
                             'sha256': _h.sha256(f.read()).hexdigest()}
  out['training_history'] = {'xy_state_action': b_sa['history'],
                             'xy_state_only': b_s['history'],
                             'sink': b_sink['history']}
  jp = os.path.join(args.out_dir, 'transition_diag.json')
  with open(jp, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nsaved %s' % mp)
  print('saved %s' % sp_path)
  print('wrote %s' % jp)


if __name__ == '__main__':
  main()
