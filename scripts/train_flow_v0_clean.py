"""V0 conditional Flow Matching on CLEAN factual adjacent transitions.

ENGINEERING / PLUMBING EXPERIMENT ONLY. Learns the local next-state
increment distribution q_theta(delta_s | s) from the authoritative CLEAN
offline dataset and nothing else:

  * training pairs are mechanically built as (s_t, s_{t+1}) from
    learner-visible observations obs[e, t, :29] of D_clean;
  * NO failure bank, NO settled failure states, NO held-out same-anchor
    pairs, NO forced masks, NO severity / dead / rock metadata, NO critic
    scores, NO success labels. The builder reads only obs + lengths + meta
    from the clean npz (which by construction contains only dead=False
    episodes) -- it never opens a sidecar or any failure artifact;
  * NO action conditioning, NO goal conditioning, NO privileged inputs.

Model: v_theta(x_t, t, s_norm) -> R^29, a small MLP. Objective: straight
(rectified) flow matching,
    x_1 = normalized delta, x_0 ~ N(0, I), t ~ U(0,1),
    x_t = (1-t) x_0 + t x_1,   L = || v(x_t, t, s) - (x_1 - x_0) ||^2.

Splits are at the TRAJECTORY level (adjacent transitions inside one episode
are near-duplicates, so a per-transition split would leak). Normalization
statistics come from the TRAIN split only.

Usage:
  python scripts/train_flow_v0_clean.py [--steps 20000] [--out artifacts/flow_v0_clean]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402  (sha256/git provenance)

CLEAN_NPZ = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
             'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
OUT_DIR = 'artifacts/flow_v0_clean'
OBS_DIM = 29
STD_FLOOR = 1e-3        # epsilon floor for near-constant dimensions


def build_pairs(npz_path, val_frac, seed):
  """(s_t, s_{t+1}) pairs from clean factual observations, split by episode."""
  with np.load(npz_path, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)          # [E, L, 58]
    lengths = np.asarray(d['lengths'], np.int64)
    meta = json.loads(str(d['meta']))
  assert meta['obs_dim'] == OBS_DIM
  n_eps = obs.shape[0]
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n_eps)
  n_val = max(1, int(round(val_frac * n_eps)))
  val_eps = np.sort(perm[:n_val])
  train_eps = np.sort(perm[n_val:])
  assert not (set(val_eps.tolist()) & set(train_eps.tolist()))

  def stack(eps):
    s = np.concatenate([obs[e, :lengths[e] - 1, :OBS_DIM] for e in eps])
    s2 = np.concatenate([obs[e, 1:lengths[e], :OBS_DIM] for e in eps])
    return s, s2 - s

  s_tr, d_tr = stack(train_eps)
  s_va, d_va = stack(val_eps)
  split = {'seed': seed, 'val_frac': val_frac, 'n_episodes': int(n_eps),
           'train_episode_ids': train_eps.tolist(),
           'val_episode_ids': val_eps.tolist(),
           'n_train_episodes': int(len(train_eps)),
           'n_val_episodes': int(len(val_eps)),
           'n_train_transitions': int(len(s_tr)),
           'n_val_transitions': int(len(s_va)),
           'split_level': 'trajectory (episode) -- NOT per-transition'}
  return (s_tr, d_tr), (s_va, d_va), split, meta


def norm_stats(s, dlt):
  """Train-split normalization statistics with an epsilon std floor."""
  return {
      'state_mean': s.mean(0), 'state_std': np.maximum(s.std(0), STD_FLOOR),
      'delta_mean': dlt.mean(0),
      'delta_std': np.maximum(dlt.std(0), STD_FLOOR),
      'std_floor': STD_FLOOR,
      'n_state_dims_at_floor': int((s.std(0) < STD_FLOOR).sum()),
      'n_delta_dims_at_floor': int((dlt.std(0) < STD_FLOOR).sum())}


def make_net(hidden, dim):
  def _v(x, t, s):
    h = jnp.concatenate([x, t, s], axis=-1)
    return hk.nets.MLP(list(hidden) + [dim], activation=jax.nn.relu,
                       name='vfield')(h)
  return hk.without_apply_rng(hk.transform(_v))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--npz', default=CLEAN_NPZ)
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--steps', type=int, default=20_000)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--hidden', default='512,512')
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--val-frac', type=float, default=0.15)
  ap.add_argument('--val-every', type=int, default=500)
  ap.add_argument('--val-batches', type=int, default=20)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  hidden = tuple(int(h) for h in args.hidden.split(','))

  # guard: V0 must never be pointed at a failure/oracle artifact
  low = args.npz.lower()
  for bad in ('failure_bank', 'rockfail', 'deaths_extended', 'same_anchor',
              'settled', 'sidecar'):
    assert bad not in low, ('V0 dataset must be clean factual data, got '
                            + args.npz)

  (s_tr, d_tr), (s_va, d_va), split, meta = build_pairs(
      args.npz, args.val_frac, args.seed)
  st = norm_stats(s_tr, d_tr)
  print('train {} eps / {} pairs | val {} eps / {} pairs'.format(
      split['n_train_episodes'], split['n_train_transitions'],
      split['n_val_episodes'], split['n_val_transitions']), flush=True)
  print('std floor hits: state {} delta {}'.format(
      st['n_state_dims_at_floor'], st['n_delta_dims_at_floor']), flush=True)

  def nz(x, m, s):
    return (x - m) / s
  S_tr = jnp.asarray(nz(s_tr, st['state_mean'], st['state_std']))
  X_tr = jnp.asarray(nz(d_tr, st['delta_mean'], st['delta_std']))
  S_va = jnp.asarray(nz(s_va, st['state_mean'], st['state_std']))
  X_va = jnp.asarray(nz(d_va, st['delta_mean'], st['delta_std']))

  net = make_net(hidden, OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, k0 = jax.random.split(key)
  params = net.init(k0, X_tr[:2], jnp.zeros((2, 1)), S_tr[:2])
  n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  opt = optax.adam(args.lr)
  opt_state = opt.init(params)
  print('vector field MLP {} -> {} | {} params'.format(hidden, OBS_DIM, n_par),
        flush=True)

  def fm_loss(p, s, x1, x0, t):
    xt = (1.0 - t) * x0 + t * x1
    v = net.apply(p, xt, t, s)
    return jnp.mean(jnp.sum((v - (x1 - x0)) ** 2, axis=-1))

  @jax.jit
  def update(p, os_, s, x1, x0, t):
    l, g = jax.value_and_grad(fm_loss)(p, s, x1, x0, t)
    upd, os_ = opt.update(g, os_)
    return optax.apply_updates(p, upd), os_, l

  @jax.jit
  def eval_loss(p, s, x1, x0, t):
    return fm_loss(p, s, x1, x0, t)

  # FROZEN validation batches (fixed indices, noise and t) so the val curve
  # is comparable across steps.
  vkey = jax.random.PRNGKey(args.seed + 777)
  vb = []
  for _ in range(args.val_batches):
    vkey, ki, kn, kt = jax.random.split(vkey, 4)
    idx = jax.random.randint(ki, (args.batch,), 0, X_va.shape[0])
    vb.append((S_va[idx], X_va[idx],
               jax.random.normal(kn, (args.batch, OBS_DIM)),
               jax.random.uniform(kt, (args.batch, 1))))

  curves = []
  t0 = time.time()
  run_l = None
  for step in range(1, args.steps + 1):
    key, ki, kn, kt = jax.random.split(key, 4)
    idx = jax.random.randint(ki, (args.batch,), 0, X_tr.shape[0])
    x0 = jax.random.normal(kn, (args.batch, OBS_DIM))
    t = jax.random.uniform(kt, (args.batch, 1))
    params, opt_state, l = update(params, opt_state, S_tr[idx], X_tr[idx],
                                  x0, t)
    l = float(l)
    assert np.isfinite(l), 'ABORT: non-finite train loss at step {}'.format(step)
    run_l = l if run_l is None else 0.98 * run_l + 0.02 * l
    if step % args.val_every == 0 or step == 1:
      vl = float(np.mean([float(eval_loss(params, *b)) for b in vb]))
      assert np.isfinite(vl), 'ABORT: non-finite val loss at {}'.format(step)
      sps = step / (time.time() - t0)
      curves.append({'step': step, 'train_loss_ema': run_l,
                     'train_loss': l, 'val_loss': vl, 'steps_per_sec': sps})
      print('[{:6d}] train {:.4f} val {:.4f} ({:.1f} it/s)'.format(
          step, run_l, vl, sps), flush=True)

  ckpt = {'params': jax.device_get(params),
          'hidden': hidden, 'obs_dim': OBS_DIM,
          'norm': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in st.items()},
          'config': {'steps': args.steps, 'batch': args.batch,
                     'lr': args.lr, 'seed': args.seed,
                     'val_frac': args.val_frac, 'hidden': list(hidden),
                     'objective': 'straight/rectified flow matching, '
                                  'x_t=(1-t)x0+t x1, target x1-x0',
                     'target': 'normalized delta_s = s_{t+1} - s_t',
                     'conditioning': 'normalized s_t only (no action, no '
                                     'goal, no privileged variables)'},
          'n_params': n_par}
  with open(os.path.join(args.out, 'flow_v0.pkl'), 'wb') as f:
    pickle.dump(ckpt, f)
  np.savez_compressed(os.path.join(args.out, 'norm_stats.npz'),
                      **{k: np.asarray(v) for k, v in st.items()})
  split['npz'] = args.npz
  split['npz_sha256'] = C.sha256_file(args.npz)
  split['dataset_note'] = meta['note']
  json.dump(split, open(os.path.join(args.out, 'split_manifest.json'), 'w'),
            indent=2)
  json.dump({'curves': curves, 'n_params': n_par, 'hidden': list(hidden),
             'config': ckpt['config'], 'git_commit': C.git_commit(),
             'wall_time_sec': time.time() - t0},
            open(os.path.join(args.out, 'train_log.json'), 'w'), indent=2)
  print('\nfinal: train {:.4f} val {:.4f} -> {}'.format(
      curves[-1]['train_loss_ema'], curves[-1]['val_loss'], args.out),
      flush=True)


if __name__ == '__main__':
  main()
