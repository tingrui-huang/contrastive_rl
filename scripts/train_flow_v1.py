"""Phase 2: V1 Flow training on D_good + D_bad-demo source mixture.

The ONLY change vs the V0/V0.5 baselines is the TRAINING SUPPORT:

    D_batch ~ (1 - beta) * D_good  +  beta * D_bad-demo

with factual transitions drawn UNIFORMLY inside each source pool. beta is a
source-mixture probability, NOT a death fraction. Nothing else moves:

  * family 'S'  -> the exact V0 architecture   q(ds | s)
  * family 'SA' -> the exact V0.5 architecture q(ds | s, a), raw [-1,1] actions
  * same straight flow-matching objective, target ds = s' - s;
  * same optimizer / lr / batch / 20k budget / seed convention;
  * FROZEN clean-data normalization (V0 stats, never recomputed);
  * D_good pool = the FROZEN V0 TRAIN episodes only, so the V0 validation
    episodes stay clean and every comparison remains apples-to-apples.

Deliberately NOT done in this first V1 experiment: no death oversampling, no
failure-local windows, no use of _dead to build minibatches, no failure
labels, no critic guidance. Every bad-demo transition is just a factual
transition; the Flow sees no failure information of any kind.

Usage:
  python scripts/train_flow_v1.py --family SA --beta 0.10
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
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import make_net, OBS_DIM   # noqa: E402
from train_flow_v05_clean_action import make_net_a  # noqa: E402

V0_DIR = 'artifacts/flow_v0_clean'
BAD_NPZ = ('artifacts/bad_demo_fixed/'
           'bad_demo_blind_p30_h800_settle80.npz')
OUT_ROOT = 'artifacts/flow_v1_sweep'


def stack_pairs(npz_path, eps=None):
  """(s, a, ds) factual adjacent tuples; eps=None uses every episode."""
  with np.load(npz_path, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    ln = np.asarray(d['lengths'], np.int64)
  idx = range(obs.shape[0]) if eps is None else eps
  s = np.concatenate([obs[e, :ln[e] - 1, :OBS_DIM] for e in idx])
  a = np.concatenate([act[e, :ln[e] - 1] for e in idx])
  s2 = np.concatenate([obs[e, 1:ln[e], :OBS_DIM] for e in idx])
  return s, a, s2 - s


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--family', choices=('S', 'SA'), required=True)
  ap.add_argument('--beta', type=float, required=True)
  ap.add_argument('--v0-dir', default=V0_DIR)
  ap.add_argument('--bad-npz', default=BAD_NPZ)
  ap.add_argument('--out-root', default=OUT_ROOT)
  ap.add_argument('--steps', type=int, default=20_000)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--hidden', default='512,512')
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--val-every', type=int, default=500)
  ap.add_argument('--val-batches', type=int, default=20)
  args = ap.parse_args()
  hidden = tuple(int(h) for h in args.hidden.split(','))
  run_id = 'V1-%s-b%03d' % (args.family, round(args.beta * 100))
  out = os.path.join(args.out_root, run_id)
  os.makedirs(out, exist_ok=True)

  # ---- provenance ---------------------------------------------------------
  split = json.load(open(os.path.join(args.v0_dir, 'split_manifest.json')))
  good_npz = split['npz']
  good_sha = C.sha256_file(good_npz)
  bad_sha = C.sha256_file(args.bad_npz)
  assert good_sha == split['npz_sha256'], 'clean dataset drifted'
  bad_man = json.load(open(os.path.join(os.path.dirname(args.bad_npz),
                                        'collection_manifest.json')))
  assert bad_man['npz_sha256'] == bad_sha, 'bad-demo dataset drifted'
  assert bad_man['death_settle_substeps'] == 80, 'bad-demo not settle-80'
  nz_ = np.load(os.path.join(args.v0_dir, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  dev = jax.devices()
  print('=' * 66)
  print('RUN            :', run_id)
  print('git commit     :', C.git_commit())
  print('family / beta  : %s / %.2f' % (args.family, args.beta))
  print('clean npz sha  :', good_sha)
  print('bad-demo sha   :', bad_sha)
  print('normalization  :', os.path.join(args.v0_dir, 'norm_stats.npz'))
  print('seed / steps   : %d / %d' % (args.seed, args.steps))
  print('jax devices    : %s (%s)' % (dev, jax.default_backend()))
  print('=' * 66, flush=True)

  # ---- pools ---------------------------------------------------------------
  tr_eps = np.asarray(split['train_episode_ids'], np.int64)
  va_eps = np.asarray(split['val_episode_ids'], np.int64)
  sg, ag, dg = stack_pairs(good_npz, tr_eps)
  sv, av, dv = stack_pairs(good_npz, va_eps)          # clean val (unchanged)
  sb, ab, db = stack_pairs(args.bad_npz)              # every bad-demo pair
  assert len(sg) == split['n_train_transitions'], 'good split mismatch'
  print('pools: good %d | bad %d | clean-val %d'
        % (len(sg), len(sb), len(sv)), flush=True)

  def nz(x, m, s):
    return (x - m) / s
  Sg = jnp.asarray(nz(sg, nrm['state_mean'], nrm['state_std']))
  Xg = jnp.asarray(nz(dg, nrm['delta_mean'], nrm['delta_std']))
  Ag = jnp.asarray(ag)
  Sb = jnp.asarray(nz(sb, nrm['state_mean'], nrm['state_std']))
  Xb = jnp.asarray(nz(db, nrm['delta_mean'], nrm['delta_std']))
  Ab = jnp.asarray(ab)
  Sv = jnp.asarray(nz(sv, nrm['state_mean'], nrm['state_std']))
  Xv = jnp.asarray(nz(dv, nrm['delta_mean'], nrm['delta_std']))
  Av = jnp.asarray(av)

  use_a = args.family == 'SA'
  net = make_net_a(hidden, OBS_DIM) if use_a else make_net(hidden, OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, k0 = jax.random.split(key)
  init_args = ((Xg[:2], jnp.zeros((2, 1)), Sg[:2], Ag[:2]) if use_a
               else (Xg[:2], jnp.zeros((2, 1)), Sg[:2]))
  params = net.init(k0, *init_args)
  n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  opt = optax.adam(args.lr)
  opt_state = opt.init(params)
  print('%s vector field %s | %d params' % (args.family, hidden, n_par),
        flush=True)

  def fm_loss(p, s, a, x1, x0, t):
    xt = (1.0 - t) * x0 + t * x1
    v = net.apply(p, xt, t, s, a) if use_a else net.apply(p, xt, t, s)
    return jnp.mean(jnp.sum((v - (x1 - x0)) ** 2, axis=-1))

  @jax.jit
  def update(p, os_, s, a, x1, x0, t):
    l, g = jax.value_and_grad(fm_loss)(p, s, a, x1, x0, t)
    upd, os_ = opt.update(g, os_)
    return optax.apply_updates(p, upd), os_, l

  @jax.jit
  def eval_loss(p, s, a, x1, x0, t):
    return fm_loss(p, s, a, x1, x0, t)

  @jax.jit
  def draw(k, ng, nb):
    """Source-mixture batch: Bernoulli(beta) picks the bad pool per slot."""
    k1, k2, k3 = jax.random.split(k, 3)
    ig = jax.random.randint(k1, (args.batch,), 0, ng)
    ib = jax.random.randint(k2, (args.batch,), 0, nb)
    from_bad = jax.random.uniform(k3, (args.batch,)) < args.beta
    s = jnp.where(from_bad[:, None], Sb[ib], Sg[ig])
    a = jnp.where(from_bad[:, None], Ab[ib], Ag[ig])
    x1 = jnp.where(from_bad[:, None], Xb[ib], Xg[ig])
    return s, a, x1, from_bad.mean()

  vkey = jax.random.PRNGKey(args.seed + 777)
  vb = []
  for _ in range(args.val_batches):
    vkey, ki, kn, kt = jax.random.split(vkey, 4)
    idx = jax.random.randint(ki, (args.batch,), 0, Xv.shape[0])
    vb.append((Sv[idx], Av[idx], Xv[idx],
               jax.random.normal(kn, (args.batch, OBS_DIM)),
               jax.random.uniform(kt, (args.batch, 1))))

  curves, t0, run_l, bad_frac = [], time.time(), None, []
  for step in range(1, args.steps + 1):
    key, kd, kn, kt = jax.random.split(key, 4)
    s, a, x1, bf = draw(kd, len(sg), len(sb))
    bad_frac.append(float(bf))
    x0 = jax.random.normal(kn, (args.batch, OBS_DIM))
    t = jax.random.uniform(kt, (args.batch, 1))
    params, opt_state, l = update(params, opt_state, s, a, x1, x0, t)
    l = float(l)
    assert np.isfinite(l), 'ABORT: non-finite train loss at %d' % step
    run_l = l if run_l is None else 0.98 * run_l + 0.02 * l
    if step % args.val_every == 0 or step == 1:
      vl = float(np.mean([float(eval_loss(params, *b)) for b in vb]))
      assert np.isfinite(vl), 'ABORT: non-finite val loss at %d' % step
      sps = step / (time.time() - t0)
      curves.append({'step': step, 'train_loss_ema': run_l,
                     'clean_val_loss': vl, 'steps_per_sec': sps})
      print('[%6d] train %.4f clean-val %.4f (%.1f it/s)'
            % (step, run_l, vl, sps), flush=True)

  ckpt = {'params': jax.device_get(params), 'hidden': hidden,
          'obs_dim': OBS_DIM, 'family': args.family, 'beta': args.beta,
          'norm': {k: v.tolist() for k, v in nrm.items()},
          'config': {'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
                     'seed': args.seed, 'hidden': list(hidden),
                     'objective': 'straight flow matching, target x1-x0',
                     'target': 'normalized delta_s',
                     'conditioning': ('s and a (raw)' if use_a else 's only'),
                     'mixture': 'source-level beta over (D_good, D_bad)'},
          'n_params': n_par}
  with open(os.path.join(out, 'flow_v1.pkl'), 'wb') as f:
    pickle.dump(ckpt, f)
  json.dump({'run_id': run_id, 'family': args.family, 'beta': args.beta,
             'curves': curves, 'n_params': n_par,
             'config': ckpt['config'],
             'realized_bad_fraction_mean': float(np.mean(bad_frac)),
             'pools': {'good_transitions': int(len(sg)),
                       'bad_transitions': int(len(sb)),
                       'clean_val_transitions': int(len(sv))},
             'provenance': {'git_commit': C.git_commit(),
                            'clean_npz': good_npz, 'clean_sha256': good_sha,
                            'bad_npz': args.bad_npz, 'bad_sha256': bad_sha,
                            'norm_source': os.path.join(args.v0_dir,
                                                        'norm_stats.npz'),
                            'jax_backend': jax.default_backend(),
                            'jax_devices': [str(x) for x in dev]},
             'wall_time_sec': time.time() - t0},
            open(os.path.join(out, 'train_log.json'), 'w'), indent=2)
  print('\nfinal: train %.4f clean-val %.4f | realized bad frac %.4f -> %s'
        % (curves[-1]['train_loss_ema'], curves[-1]['clean_val_loss'],
           float(np.mean(bad_frac)), out), flush=True)


if __name__ == '__main__':
  main()
