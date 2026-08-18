"""V0.5 ABLATION: action-conditioned Flow Matching on the SAME clean data.

The single scientific change vs scripts/train_flow_v0_clean.py is the
conditioning set:

    V0    v_theta(x_t, t, s)          ->  q(delta_s | s)
    V0.5  v_theta(x_t, t, s, a)       ->  q(delta_s | s, a)

Everything else is held fixed and REUSED from V0, not recomputed:
  * the same authoritative clean factual npz (sha checked against the V0
    split manifest);
  * the FROZEN V0 trajectory split (identical train/val episode ids);
  * the FROZEN V0 normalization statistics (state and delta mean/std);
  * same target (normalized delta_s), same straight-flow objective, same
    hidden widths / depth / activation, optimizer, lr, batch, step budget,
    seed, and the same fixed-step Euler sampler at eval time.

Actions enter RAW: they are already bounded in [-1, 1] with per-dim std
~0.68-0.88 on the clean training split, i.e. already on the same scale as
the normalized state/delta inputs, so no extra transform is introduced
(the measured statistics are recorded in action_stats.json).

Clean-only, exactly as V0: the builder reads obs / act / lengths from
D_clean and nothing else. No failure bank, settled states, pilot death
transitions, held-out pairs, fresh death stream, _dead, mask, severity,
privileged teacher signal, critic score, or counterfactual replay.

Usage:
  python scripts/train_flow_v05_clean_action.py [--steps 20000]
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

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402  (reuse V0 constants)

V0_DIR = 'artifacts/flow_v0_clean'
OUT_DIR = 'artifacts/flow_v05_clean_action'
ACT_DIM = 8


def make_net_a(hidden, dim):
  """V0's vector field with the action appended to the condition."""
  def _v(x, t, s, a):
    h = jnp.concatenate([x, t, s, a], axis=-1)
    return hk.nets.MLP(list(hidden) + [dim], activation=jax.nn.relu,
                       name='vfield')(h)
  return hk.without_apply_rng(hk.transform(_v))


def build_tuples(npz_path, train_eps, val_eps):
  """(s_t, a_t, s_{t+1}) on the FROZEN V0 episode split."""
  with np.load(npz_path, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)

  def stack(eps):
    s = np.concatenate([obs[e, :lengths[e] - 1, :OBS_DIM] for e in eps])
    a = np.concatenate([act[e, :lengths[e] - 1] for e in eps])
    s2 = np.concatenate([obs[e, 1:lengths[e], :OBS_DIM] for e in eps])
    return s, a, s2 - s

  return stack(train_eps), stack(val_eps)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--v0-dir', default=V0_DIR)
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--steps', type=int, default=20_000)
  ap.add_argument('--batch', type=int, default=1024)
  ap.add_argument('--hidden', default='512,512')
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--val-every', type=int, default=500)
  ap.add_argument('--val-batches', type=int, default=20)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  hidden = tuple(int(h) for h in args.hidden.split(','))

  # ---- reuse the FROZEN V0 split + normalization --------------------------
  split = json.load(open(os.path.join(args.v0_dir, 'split_manifest.json')))
  npz = split['npz']
  assert C.sha256_file(npz) == split['npz_sha256'], 'clean dataset drifted'
  low = npz.lower()
  for bad in ('failure_bank', 'rockfail', 'deaths_extended', 'same_anchor',
              'settled', 'sidecar'):
    assert bad not in low, 'V0.5 must train on clean factual data only'
  train_eps = np.asarray(split['train_episode_ids'], np.int64)
  val_eps = np.asarray(split['val_episode_ids'], np.int64)
  nz_ = np.load(os.path.join(args.v0_dir, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}

  (s_tr, a_tr, d_tr), (s_va, a_va, d_va) = build_tuples(
      npz, train_eps, val_eps)
  assert len(s_tr) == split['n_train_transitions'], 'split mismatch vs V0'
  assert len(s_va) == split['n_val_transitions'], 'split mismatch vs V0'
  print('FROZEN V0 split reused: %d/%d eps, %d/%d transitions'
        % (len(train_eps), len(val_eps), len(s_tr), len(s_va)), flush=True)

  # ---- action statistics (documented; actions kept RAW) -------------------
  astats = {'source': 'clean TRAIN split only',
            'transform': 'none (raw bounded actions)',
            'rationale': ('already bounded in [-1, 1] with per-dim std '
                          '0.68-0.88, i.e. the same scale as the normalized '
                          'state/delta inputs, so no extra transform is '
                          'introduced'),
            'min': float(a_tr.min()), 'max': float(a_tr.max()),
            'per_dim_mean': a_tr.mean(0).tolist(),
            'per_dim_std': a_tr.std(0).tolist(),
            'frac_abs_gt_0.99': float((np.abs(a_tr) > 0.99).mean())}
  json.dump(astats, open(os.path.join(args.out, 'action_stats.json'), 'w'),
            indent=2)

  def nz(x, m, s):
    return (x - m) / s
  S_tr = jnp.asarray(nz(s_tr, nrm['state_mean'], nrm['state_std']))
  X_tr = jnp.asarray(nz(d_tr, nrm['delta_mean'], nrm['delta_std']))
  A_tr = jnp.asarray(a_tr)
  S_va = jnp.asarray(nz(s_va, nrm['state_mean'], nrm['state_std']))
  X_va = jnp.asarray(nz(d_va, nrm['delta_mean'], nrm['delta_std']))
  A_va = jnp.asarray(a_va)

  net = make_net_a(hidden, OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, k0 = jax.random.split(key)
  params = net.init(k0, X_tr[:2], jnp.zeros((2, 1)), S_tr[:2], A_tr[:2])
  n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  opt = optax.adam(args.lr)
  opt_state = opt.init(params)
  print('vector field MLP %s -> %d | %d params (V0 had 308253; the delta is '
        'the %d extra first-layer weights for the action)'
        % (hidden, OBS_DIM, n_par, ACT_DIM * hidden[0]), flush=True)

  def fm_loss(p, s, a, x1, x0, t):
    xt = (1.0 - t) * x0 + t * x1
    v = net.apply(p, xt, t, s, a)
    return jnp.mean(jnp.sum((v - (x1 - x0)) ** 2, axis=-1))

  @jax.jit
  def update(p, os_, s, a, x1, x0, t):
    l, g = jax.value_and_grad(fm_loss)(p, s, a, x1, x0, t)
    upd, os_ = opt.update(g, os_)
    return optax.apply_updates(p, upd), os_, l

  @jax.jit
  def eval_loss(p, s, a, x1, x0, t):
    return fm_loss(p, s, a, x1, x0, t)

  # frozen validation batches (same construction/seed offset as V0)
  vkey = jax.random.PRNGKey(args.seed + 777)
  vb = []
  for _ in range(args.val_batches):
    vkey, ki, kn, kt = jax.random.split(vkey, 4)
    idx = jax.random.randint(ki, (args.batch,), 0, X_va.shape[0])
    vb.append((S_va[idx], A_va[idx], X_va[idx],
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
    params, opt_state, l = update(params, opt_state, S_tr[idx], A_tr[idx],
                                  X_tr[idx], x0, t)
    l = float(l)
    assert np.isfinite(l), 'ABORT: non-finite train loss at %d' % step
    run_l = l if run_l is None else 0.98 * run_l + 0.02 * l
    if step % args.val_every == 0 or step == 1:
      vl = float(np.mean([float(eval_loss(params, *b)) for b in vb]))
      assert np.isfinite(vl), 'ABORT: non-finite val loss at %d' % step
      sps = step / (time.time() - t0)
      curves.append({'step': step, 'train_loss_ema': run_l,
                     'train_loss': l, 'val_loss': vl, 'steps_per_sec': sps})
      print('[%6d] train %.4f val %.4f (%.1f it/s)'
            % (step, run_l, vl, sps), flush=True)

  ckpt = {'params': jax.device_get(params), 'hidden': hidden,
          'obs_dim': OBS_DIM, 'act_dim': ACT_DIM,
          'norm': {k: v.tolist() for k, v in nrm.items()},
          'norm_source': os.path.join(args.v0_dir, 'norm_stats.npz'),
          'action_transform': 'none (raw)',
          'config': {'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
                     'seed': args.seed, 'hidden': list(hidden),
                     'objective': 'straight/rectified flow matching, '
                                  'x_t=(1-t)x0+t x1, target x1-x0',
                     'target': 'normalized delta_s = s_{t+1} - s_t',
                     'conditioning': 'normalized s_t AND raw action a_t'},
          'n_params': n_par}
  with open(os.path.join(args.out, 'flow_v05.pkl'), 'wb') as f:
    pickle.dump(ckpt, f)
  json.dump({'curves': curves, 'n_params': n_par, 'hidden': list(hidden),
             'config': ckpt['config'], 'git_commit': C.git_commit(),
             'wall_time_sec': time.time() - t0,
             'split_source': os.path.join(args.v0_dir,
                                          'split_manifest.json'),
             'norm_source': ckpt['norm_source'],
             'npz': npz, 'npz_sha256': split['npz_sha256']},
            open(os.path.join(args.out, 'train_log.json'), 'w'), indent=2)
  print('\nfinal: train %.4f val %.4f -> %s'
        % (curves[-1]['train_loss_ema'], curves[-1]['val_loss'], args.out),
        flush=True)


if __name__ == '__main__':
  main()
