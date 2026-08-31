"""Deterministic one-step transition diagnostic on the windy-swamp PointMaze.

QUESTION THIS ANSWERS. Before building anything that reasons about worst-case
transitions, how much of the one-step dynamics is learnable at all from the
LEARNER-VISIBLE inputs, and how much of that needs the action? Three models on
the same target:

    1. state-only     [x, y]          -> delta_xy
    2. state+action   [x, y, ax, ay]  -> delta_xy
    3. zero-delta     predict 0       (the "nothing moves" floor)

Same 2x256 ReLU MLP and MSE for both neural models. Deliberately plain: no
Lipschitz constraint, no spectral norm, no GroupSort, no flow model, no
counterfactual training. Those come later, and these are the numbers they have
to beat.

WHAT TO EXPECT, and why the stratification is the point. The env integrates the
action over 10 substeps at dt=0.1, one axis at a time, rejecting any substep
that would enter a wall, so in free space delta_xy is very nearly the action
itself and a state+action model should be close to exact. The interesting
residual sits in two places:

  * walls, where per-axis rejection makes delta a function of geometry;
  * the swamp corridor, where ending a step in an ACTIVE cell is terminal and
    the point freezes for the rest of the episode. Whether that happens is a
    function of the hidden bit U, which is NOT in the observation
    (crl/envs.py: "state = [x, y]  (U is NOT exposed)"). Two transitions with
    identical (s, a) therefore have different deltas depending on U, and NO
    deterministic model on [x, y, ax, ay] can separate them -- it can only
    average them. The clear-vs-active split is what makes that visible.

POST-DEATH FROZEN TRANSITIONS. Once dead the point is frozen and every later
delta is exactly zero -- 21,914 of this dataset's 22,433 zero-delta transitions
come from that tail. Those rows are trivially predictable and would flatter
every model, so the PRIMARY train/val/test set drops them. The transition that
ENTERS death (s_{r-1} -> s_r, the last real move) is KEPT: it is a genuine
transition and it is precisely the one a worst-case model has to get right. It
is also reported as its own stratum. Metrics on the FULL set, frozen rows
included, are reported separately.

SPLIT is by EPISODE, never by transition -- consecutive transitions inside an
episode are strongly dependent, so a transition-level split leaks. 80/10/10,
stratified on (teacher_mode, died) so every behaviour mode and the death rate
keep their proportions across all three parts.

swamp_bits is read for STRATIFICATION AND REPORTING ONLY and never enters a
model input. Nothing here touches the CRL/policy training pipeline.

Usage:
  python scripts/diag_transition_mlp.py
  python scripts/diag_transition_mlp.py --dataset datasets/swamp_windy_merged_s0.npz
  python scripts/diag_transition_mlp.py --epochs 60 --seed 1
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

# Tiny MLP on a few hundred thousand rows: keep it out of the GPU preallocator.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import haiku as hk                                # noqa: E402
import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
import optax                                      # noqa: E402

DATASET = 'datasets/swamp_windy_merged_s0.npz'
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
GRID_HI = [8, 4]                                  # walls array is 9 x 5
HIDDEN = (256, 256)                               # "2 x 256 ReLU"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def death_rows(obs, bits, died):
  """Row index of the death STATE per episode, -1 if the episode survived.

  Same timing contract as crl.envs.TwoRouteSwampWindyEnv.step and
  scripts/make_swamp_failure_bank.py: bits[t] govern step t, the death check
  runs on the state REACHED by that step, then the bits redraw. So the death
  state sits at row t+1 where cell(row t+1) is swamp k and bits[t, k] is set.
  """
  n_eps, L, _ = obs.shape
  cell = np.clip(np.floor(obs[:, :, :2]).astype(int), 0, GRID_HI)
  sw = -np.ones((n_eps, L), int)
  for k, (cx, cy) in enumerate(SWAMP_CELLS):
    sw[(cell[:, :, 0] == cx) & (cell[:, :, 1] == cy)] = k
  out = np.full(n_eps, -1, np.int64)
  for e in np.where(died)[0]:
    for t in range(L - 1):
      k = sw[e, t + 1]
      if k >= 0 and bits[e, t, k]:
        out[e] = t + 1
        break
  return out, sw


def build_transitions(path):
  """Flatten to per-transition arrays plus the audit labels used to stratify."""
  with np.load(path, allow_pickle=False) as d:
    obs, act, bits = d['obs'], d['act'], d['swamp_bits']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
    mode = np.asarray(d['teacher_mode'])
  n_eps, L, W = obs.shape
  assert W == 4, 'expected obs width 4 ([x,y,gx,gy]), got %d' % W
  T = L - 1                                       # act[:, -1] is a dummy

  xy = obs[:, :, :2]
  s = xy[:, :-1, :]                               # [N, T, 2]
  s_next = xy[:, 1:, :]
  a = act[:, :-1, :]
  delta = s_next - s

  drow, sw = death_rows(obs, bits, died)
  assert np.all(drow[died] > 0), 'a dead episode has no reconstructable death row'

  # Which swamp cell does this transition LAND in, and was that cell active
  # during the step? bits[:, :-1] are the bits governing transitions 0..T-1.
  land = sw[:, 1:]                                # [N, T] swamp idx of s_{t+1}
  landed_swamp = land >= 0
  landed_active = np.zeros_like(landed_swamp)
  for k in range(3):
    landed_active |= (land == k) & bits[:, :-1, k].astype(bool)
  landed_clear = landed_swamp & ~landed_active

  # PRIMARY mask: drop transitions at or after the death row (the frozen tail).
  # t = drow - 1 is the move that ENTERS death and is KEPT.
  t_idx = np.arange(T)[None, :]
  frozen = died[:, None] & (t_idx >= drow[:, None])
  entering = died[:, None] & (t_idx == (drow - 1)[:, None])
  primary = ~frozen

  ep_id = np.repeat(np.arange(n_eps)[:, None], T, axis=1)

  def flat(A):
    return A.reshape(-1, A.shape[-1]) if A.ndim == 3 else A.reshape(-1)

  return dict(
      s=flat(s).astype(np.float32), a=flat(a).astype(np.float32),
      delta=flat(delta).astype(np.float32),
      ep=flat(ep_id), primary=flat(primary), entering=flat(entering),
      landed_swamp=flat(landed_swamp), landed_active=flat(landed_active),
      landed_clear=flat(landed_clear),
      ep_mode=mode, ep_died=died, ep_drow=drow,
      n_eps=n_eps, T=T)


def split_episodes(mode, died, seed, fracs=(0.8, 0.1, 0.1)):
  """80/10/10 over EPISODES, stratified on (teacher_mode, died).

  Stratifying keeps each behaviour mode and the death rate at their dataset
  proportions in all three parts; without it a 600-episode stratum like the bad
  demonstrator can land almost entirely in one split.
  """
  rng = np.random.default_rng(seed)
  which = np.full(len(mode), '', dtype='<U5')
  for m in np.unique(mode):
    for dd in (False, True):
      idx = np.where((mode == m) & (died == dd))[0]
      if idx.size == 0:
        continue
      idx = rng.permutation(idx)
      n = idx.size
      n_tr = int(round(fracs[0] * n))
      n_va = int(round(fracs[1] * n))
      # Keep val/test non-empty wherever the stratum can afford it, so a small
      # stratum still shows up in the held-out metrics instead of silently
      # rounding away.
      if n >= 3:
        n_tr = min(n_tr, n - 2)
        n_va = max(1, min(n_va, n - n_tr - 1))
      which[idx[:n_tr]] = 'train'
      which[idx[n_tr:n_tr + n_va]] = 'val'
      which[idx[n_tr + n_va:]] = 'test'
  assert not np.any(which == ''), 'an episode was left unassigned by the split'
  return which


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_mlp():
  """2 x 256 ReLU -> 2. Identical for both neural arms; only the input differs."""
  def f(x):
    return hk.nets.MLP([HIDDEN[0], HIDDEN[1], 2], activation=jax.nn.relu)(x)
  return hk.without_apply_rng(hk.transform(f))


def train_mlp(x_tr, y_tr, x_va, y_va, seed, epochs, batch, lr, label):
  net = make_mlp()
  # Standardise INPUTS on train statistics only: x is O(9), y is O(4) and the
  # actions are O(1), and an unscaled input makes the first layer fight that
  # spread. The TARGET is left raw, so every error reported below is in maze
  # units and is directly comparable across models.
  mu, sd = x_tr.mean(0), x_tr.std(0)
  sd = np.where(sd < 1e-6, 1.0, sd)

  def nx(x):
    return ((x - mu) / sd).astype(np.float32)

  xt, xv = nx(x_tr), nx(x_va)
  params = net.init(jax.random.PRNGKey(seed), jnp.asarray(xt[:2]))
  opt = optax.adam(lr)
  opt_state = opt.init(params)

  def loss_fn(p, x, y):
    return jnp.mean(jnp.sum((net.apply(p, x) - y) ** 2, axis=-1))

  @jax.jit
  def step(p, o, x, y):
    l, g = jax.value_and_grad(loss_fn)(p, x, y)
    upd, o = opt.update(g, o)
    return optax.apply_updates(p, upd), o, l

  @jax.jit
  def val_loss(p, x, y):
    return loss_fn(p, x, y)

  rng = np.random.default_rng(seed)
  n = xt.shape[0]
  best, best_params, best_ep = np.inf, params, -1
  jxv, jyv = jnp.asarray(xv), jnp.asarray(y_va)
  for ep in range(epochs):
    perm = rng.permutation(n)
    for k in range(0, n - batch + 1, batch):
      sl = perm[k:k + batch]
      params, opt_state, _ = step(params, opt_state,
                                  jnp.asarray(xt[sl]), jnp.asarray(y_tr[sl]))
    v = float(val_loss(params, jxv, jyv))
    if v < best:
      best, best_params, best_ep = v, params, ep
    if ep % 10 == 0 or ep == epochs - 1:
      star = '  *' if best_ep == ep else ''
      print('    [%s] epoch %3d  val MSE %.6f%s' % (label, ep, v, star))
  print('    [%s] best val MSE %.6f @ epoch %d' % (label, best, best_ep))

  def predict(x):
    return np.asarray(net.apply(best_params, jnp.asarray(nx(x))))

  # The input standardisation is part of the fitted model, so mu/sd travel with
  # the params. Anything reloading this (scripts/diag_action_lipschitz.py) must
  # differentiate through the SAME normalisation or its Jacobian is off by a
  # factor of 1/sd per input.
  bundle = {'params': jax.tree_util.tree_map(np.asarray, best_params),
            'mu': np.asarray(mu, np.float64), 'sd': np.asarray(sd, np.float64),
            'hidden': list(HIDDEN), 'best_val_mse': float(best),
            'best_epoch': int(best_ep), 'label': label}
  return predict, bundle


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def err_stats(pred, true):
  e = np.linalg.norm(pred - true, axis=1)
  if e.size == 0:
    return {'n': 0, 'mean': None, 'median': None, 'p90': None}
  return {'n': int(e.size), 'mean': float(e.mean()),
          'median': float(np.median(e)), 'p90': float(np.percentile(e, 90))}


HDR = '  %-34s%9s%10s%10s%10s' % ('', 'n', 'mean', 'median', 'p90')


def row(name, st):
  if st['n'] == 0:
    return '  %-34s%9s%10s%10s%10s' % (name, '-', '-', '-', '-')
  return '  %-34s%9s%10.4f%10.4f%10.4f' % (
      name, format(st['n'], ','), st['mean'], st['median'], st['p90'])


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--dataset', default=DATASET)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--epochs', type=int, default=40)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--json', default='artifacts/transition_diag/'
                                    'transition_mlp_diag.json')
  ap.add_argument('--save-params', default='artifacts/transition_diag/'
                                           'transition_mlp_params.pkl',
                  help='where to pickle the fitted params + input '
                       'standardisation; pass "" to skip')
  args = ap.parse_args()

  print('=' * 78)
  print('ONE-STEP TRANSITION DIAGNOSTIC (deterministic, diagonal)')
  print('=' * 78)
  print('  dataset : %s' % args.dataset)
  d = build_transitions(args.dataset)
  total = d['n_eps'] * d['T']
  print('  episodes: %d x %d transitions = %s total'
        % (d['n_eps'], d['T'], format(total, ',')))
  print('  primary (post-death frozen removed): %s   excluded: %s'
        % (format(int(d['primary'].sum()), ','),
           format(int((~d['primary']).sum()), ',')))
  ent = d['entering']
  print('  entering-death transitions KEPT: %s  (mean |delta| %.4f)'
        % (format(int(ent.sum()), ','),
           float(np.linalg.norm(d['delta'][ent], axis=1).mean())))

  which = split_episodes(d['ep_mode'], d['ep_died'], args.seed)
  modes = np.unique(d['ep_mode'])
  print('\n  EPISODE SPLIT (stratified on teacher_mode x died)')
  head = '  %-8s%7s%9s' % ('split', 'eps', 'died%')
  for m in modes:
    head += '%8s' % ('m%d' % m)
  print(head)
  for sp in ('train', 'val', 'test', 'ALL'):
    msk = np.ones(len(which), bool) if sp == 'ALL' else (which == sp)
    line = '  %-8s%7d%8.2f%%' % (sp, int(msk.sum()),
                                 d['ep_died'][msk].mean() * 100)
    for m in modes:
      line += '%8d' % int(((d['ep_mode'] == m) & msk).sum())
    print(line)

  tr_split = which[d['ep']]
  x_sa = np.concatenate([d['s'], d['a']], axis=1)
  x_s = d['s']
  y = d['delta']

  def sel(sp, prim):
    m = tr_split == sp
    return (m & d['primary']) if prim else m

  i_tr, i_va = sel('train', True), sel('val', True)
  print('\n  transitions (primary):  train %s   val %s   test %s'
        % (format(int(i_tr.sum()), ','), format(int(i_va.sum()), ','),
           format(int(sel('test', True).sum()), ',')))

  print('\n  training state-only  [x,y] -> delta_xy')
  p_s, b_s = train_mlp(x_s[i_tr], y[i_tr], x_s[i_va], y[i_va],
                       args.seed, args.epochs, args.batch, args.lr, 'state')
  print('\n  training state+action  [x,y,ax,ay] -> delta_xy')
  p_sa, b_sa = train_mlp(x_sa[i_tr], y[i_tr], x_sa[i_va], y[i_va],
                         args.seed, args.epochs, args.batch, args.lr,
                         'state+act')

  # Persist the fitted models so downstream analyses (e.g. the action-Jacobian
  # / local-Lipschitz probe) can load THIS model instead of retraining one.
  if args.save_params:
    os.makedirs(os.path.dirname(args.save_params) or '.', exist_ok=True)
    with open(args.save_params, 'wb') as f:
      pickle.dump({'state_only': b_s, 'state_action': b_sa,
                   'dataset': args.dataset, 'seed': args.seed,
                   'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr},
                  f)
    print('\n  saved fitted params -> %s' % args.save_params)

  models = {
      'zero_delta': lambda i: np.zeros_like(y[i]),
      'state_only': lambda i: p_s(x_s[i]),
      'state_action': lambda i: p_sa(x_sa[i]),
  }

  out = {'dataset': args.dataset, 'seed': args.seed, 'epochs': args.epochs,
         'n_episodes': int(d['n_eps']), 'n_transitions_total': int(total),
         'n_primary': int(d['primary'].sum()),
         'n_entering_death': int(ent.sum()),
         'split_episodes': {sp: int((which == sp).sum())
                            for sp in ('train', 'val', 'test')},
         'results': {}}

  for setname, prim, key in (
      ('PRIMARY (post-death frozen EXCLUDED)', True, 'primary'),
      ('FULL (post-death frozen INCLUDED)', False, 'full')):
    idx = sel('test', prim)
    print('\n' + '=' * 78)
    print('HELD-OUT TEST -- %s' % setname)
    print('=' * 78)
    print('  Euclidean delta error ||pred - true||, maze units')
    print(HDR)
    out['results'][key] = {}
    for name, fn in models.items():
      st = err_stats(fn(idx), y[idx])
      print(row(name, st))
      out['results'][key][name] = {'overall': st}

    print('\n  stratified by the cell the step LANDS IN'
          '  (swamp_bits: audit only, never a model input)')
    strata = [('non-swamp', idx & ~d['landed_swamp']),
              ('swamp cell, gate CLEAR', idx & d['landed_clear']),
              ('swamp cell, gate ACTIVE', idx & d['landed_active']),
              ('  of which: ENTERING death', idx & ent)]
    for name, fn in models.items():
      print('    -- %s' % name)
      print(HDR)
      for sname, smask in strata:
        st = err_stats(fn(smask), y[smask])
        print(row(sname, st))
        out['results'][key][name][sname.strip()] = st

  # ------------------------------------------------------------ sensitivity
  print('\n' + '=' * 78)
  print('ACTION SENSITIVITY PROBE (fixed states, 21x21 action grid in [-1,1]^2)')
  print('=' * 78)
  print('  A state-only model CANNOT respond to the action, so its spread is 0')
  print('  by construction and is printed as the control. For state+action,')
  print('  resp = std of the predicted delta over the action grid, and')
  print('  ||p-a|| = gap to the free-space ideal delta = a.')
  probes = [('start (0.5,3.5)', (0.5, 3.5)),
            ('fork (1.5,3.5)', (1.5, 3.5)),
            ('holding (2.5,3.5)', (2.5, 3.5)),
            ('swamp0 (3.5,3.5)', (3.5, 3.5)),
            ('swamp2 (5.5,3.5)', (5.5, 3.5)),
            ('lower route (1.5,1.5)', (1.5, 1.5)),
            ('post-join (7.5,3.5)', (7.5, 3.5))]
  g = np.linspace(-1, 1, 21)
  gx, gy = np.meshgrid(g, g)
  acts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
  print('\n  %-24s%-14s%11s%11s%13s'
        % ('probe state', 'model', 'resp(std)', 'mean||p||', 'mean||p-a||'))
  out['action_sensitivity'] = {}
  for pname, (px, py) in probes:
    st = np.repeat(np.array([[px, py]], np.float32), len(acts), axis=0)
    preds = (('state_only', p_s(st)),
             ('state_action', p_sa(np.concatenate([st, acts], axis=1))))
    for mname, pr in preds:
      resp = float(np.linalg.norm(pr.std(axis=0)))
      mn = float(np.linalg.norm(pr, axis=1).mean())
      gap = float(np.linalg.norm(pr - acts, axis=1).mean())
      print('  %-24s%-14s%11.4f%11.4f%13.4f' % (pname, mname, resp, mn, gap))
      out['action_sensitivity'].setdefault(pname, {})[mname] = {
          'response_std': resp, 'mean_pred_norm': mn,
          'mean_gap_to_action': gap}
    print()

  if args.json:
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
      json.dump(out, f, indent=2)
    print('wrote %s' % args.json)


if __name__ == '__main__':
  main()
