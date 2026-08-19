"""Phase 6: train V3 on the DIVERSE factual failure pool.

The single scientific change from V2-SA-l001 is the failure pool:

    196 blind-lane failure transitions  ->  603 more diverse factual ones
    (D_fail^diverse = D_fail^196 U D_fail^407, built and gated in
     scripts/audit_failure_diversity.py)

Everything else is frozen and reused verbatim from V2-SA-l001: the
action-conditioned architecture, flow-matching target, optimizer, lr, batch,
20k budget, seed convention, frozen clean state/delta normalization, raw
bounded actions, and lambda = 0.01.

    D_batch ~ 0.99 * D_good  +  0.01 * D_fail^diverse

Ordinary bad-demo transitions are NOT used, no extra weighting term is
added, no hyperparameter is swept, and the model still receives only (s, a).

Usage:
  python scripts/train_flow_v3.py
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
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from train_flow_v05_clean_action import make_net_a  # noqa: E402
from train_flow_v1 import stack_pairs      # noqa: E402
from train_flow_v2 import build_fail_pool  # noqa: E402

V0_DIR = 'artifacts/flow_v0_clean'
V2_CKPT = 'artifacts/flow_v2_failure_local/V2-SA-l001/flow_v2.pkl'
ROOT = 'artifacts/flow_v3_diverse_failure'
POOL = os.path.join(ROOT, 'failure_pool_diversity_audit',
                    'failure_pool_diverse.npz')
OLD_BAD_DIR = 'artifacts/bad_demo_fixed'
OLD_BAD_NAME = 'bad_demo_blind_p30_h800_settle80'
OUT = os.path.join(ROOT, 'flow_v3')
RUN_ID = 'V3-SA-diverse-l001'
LAMBDA = 0.01
N_POOL_EXPECTED = 603


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
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

  # ---- provenance gates ----------------------------------------------------
  split = json.load(open(os.path.join(V0_DIR, 'split_manifest.json')))
  good_npz = split['npz']
  good_sha = C.sha256_file(good_npz)
  assert good_sha == split['npz_sha256'], 'clean dataset drifted'
  with open(V2_CKPT, 'rb') as f:
    v2 = pickle.load(f)
  assert v2['family'] == 'SA' and abs(v2['lam'] - LAMBDA) < 1e-12
  assert tuple(v2['hidden']) == hidden, 'architecture differs from V2'
  for k in ('steps', 'batch', 'lr', 'seed'):
    assert v2['config'][k] == getattr(args, k), \
        'V3 %s=%s differs from the frozen V2 value %s' % (
            k, getattr(args, k), v2['config'][k])
  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  for k in nrm:
    assert np.array_equal(nrm[k], np.asarray(v2['norm'][k], np.float32)), \
        'normalization drifted from the frozen V0/V2 stats'

  # combined diverse pool
  z = np.load(POOL, allow_pickle=True)
  sf = np.asarray(z['state'], np.float32)
  af = np.asarray(z['action'], np.float32)
  df = np.asarray(z['delta'], np.float32)
  src = np.asarray(z['source'])
  assert len(sf) == N_POOL_EXPECTED, 'pool is %d, expected %d' % (
      len(sf), N_POOL_EXPECTED)
  n_old = int((src == 'old196').sum())
  assert n_old == 196, 'old subgroup is %d, expected 196' % n_old
  # the old 196 must still be present bitwise
  So, Ao, Do, _ = build_fail_pool(OLD_BAD_DIR, OLD_BAD_NAME)
  assert np.array_equal(np.sort(sf[src == 'old196'], axis=0),
                        np.sort(So, axis=0)), 'old 196 anchors altered'
  audit = json.load(open(os.path.join(
      ROOT, 'failure_pool_diversity_audit', 'diversity_audit.json')))
  assert audit['phase5D_gate']['PASS'], 'diversity gate did not pass'

  prov = {
      'run_id': RUN_ID, 'lambda': LAMBDA,
      'scientific_change': '196 -> 603 diverse factual failure transitions',
      'v2_reference_ckpt': V2_CKPT,
      'v2_reference_sha256': C.sha256_file(V2_CKPT),
      'v2_config': v2['config'],
      'clean_npz': good_npz, 'clean_sha256': good_sha,
      'old_failure_pool': OLD_BAD_NAME, 'n_old': n_old,
      'diverse_pool_npz': POOL, 'diverse_pool_sha256': C.sha256_file(POOL),
      'n_combined': int(len(sf)),
      'per_source_counts': {s: int((src == s).sum())
                            for s in sorted(set(src.tolist()))},
      'normalization_source': os.path.join(V0_DIR, 'norm_stats.npz'),
      'architecture': {'hidden': list(hidden), 'family': 'SA'},
      'training': {'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
                   'seed': args.seed},
      'diversity_gate_passed': True,
      'git_commit': C.git_commit(),
      'jax_backend': jax.default_backend()}
  json.dump(prov, open(os.path.join(args.out, 'provenance.json'), 'w'),
            indent=2)
  print('=' * 70)
  print('RUN            :', RUN_ID)
  print('git commit     :', prov['git_commit'])
  print('lambda         : %.4f (unchanged)' % LAMBDA)
  print('failure pool   : %d (%s)' % (len(sf), prov['per_source_counts']))
  print('clean npz sha  :', good_sha[:24])
  print('pool sha       :', prov['diverse_pool_sha256'][:24])
  print('arch/opt       : frozen == V2-SA-l001')
  print('=' * 70, flush=True)

  # ---- data ----------------------------------------------------------------
  tr_eps = np.asarray(split['train_episode_ids'], np.int64)
  va_eps = np.asarray(split['val_episode_ids'], np.int64)
  sg, ag, dg = stack_pairs(good_npz, tr_eps)
  sv, av, dv = stack_pairs(good_npz, va_eps)
  assert len(sg) == split['n_train_transitions']

  def nz(x, m, s):
    return (x - m) / s

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  Sg = jnp.asarray(nz(sg, nrm['state_mean'], nrm['state_std']))
  Xg, Ag = jnp.asarray(nzd(dg)), jnp.asarray(ag)
  Sf = jnp.asarray(nz(sf, nrm['state_mean'], nrm['state_std']))
  Xf, Af = jnp.asarray(nzd(df)), jnp.asarray(af)
  Sv = jnp.asarray(nz(sv, nrm['state_mean'], nrm['state_std']))
  Xv, Av = jnp.asarray(nzd(dv)), jnp.asarray(av)

  acct = {'lambda': LAMBDA,
          'expected_fail_per_batch': LAMBDA * args.batch,
          'total_fail_exposures': LAMBDA * args.batch * args.steps,
          'avg_repeats_per_fatal_transition':
              LAMBDA * args.batch * args.steps / len(sf),
          'v2_comparison': ('V2 had 196 transitions -> ~1045 repeats each; '
                            'V3 spreads the same exposure over 603')}
  print('accounting: %.1f fail/batch | %.0f total | ~%.0f repeats each'
        % (acct['expected_fail_per_batch'], acct['total_fail_exposures'],
           acct['avg_repeats_per_fatal_transition']), flush=True)

  net = make_net_a(hidden, OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, k0 = jax.random.split(key)
  params = net.init(k0, Xg[:2], jnp.zeros((2, 1)), Sg[:2], Ag[:2])
  n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  assert n_par == v2['n_params'], 'parameter count differs from V2'
  opt = optax.adam(args.lr)
  opt_state = opt.init(params)

  def fm_loss(p, s, a, x1, x0, t):
    xt = (1.0 - t) * x0 + t * x1
    return jnp.mean(jnp.sum((net.apply(p, xt, t, s, a) - (x1 - x0)) ** 2,
                            axis=-1))

  @jax.jit
  def update(p, os_, s, a, x1, x0, t):
    l, g = jax.value_and_grad(fm_loss)(p, s, a, x1, x0, t)
    upd, os_ = opt.update(g, os_)
    return optax.apply_updates(p, upd), os_, l

  @jax.jit
  def eval_loss(p, s, a, x1, x0, t):
    return fm_loss(p, s, a, x1, x0, t)

  @jax.jit
  def draw(k, ng, nf):
    k1, k2, k3 = jax.random.split(k, 3)
    ig = jax.random.randint(k1, (args.batch,), 0, ng)
    i_f = jax.random.randint(k2, (args.batch,), 0, nf)
    from_fail = jax.random.uniform(k3, (args.batch,)) < LAMBDA
    return (jnp.where(from_fail[:, None], Sf[i_f], Sg[ig]),
            jnp.where(from_fail[:, None], Af[i_f], Ag[ig]),
            jnp.where(from_fail[:, None], Xf[i_f], Xg[ig]),
            from_fail.mean())

  vkey = jax.random.PRNGKey(args.seed + 777)
  vb = []
  for _ in range(args.val_batches):
    vkey, ki, kn, kt = jax.random.split(vkey, 4)
    idx = jax.random.randint(ki, (args.batch,), 0, Xv.shape[0])
    vb.append((Sv[idx], Av[idx], Xv[idx],
               jax.random.normal(kn, (args.batch, OBS_DIM)),
               jax.random.uniform(kt, (args.batch, 1))))

  curves, t0, run_l, fr = [], time.time(), None, []
  for step in range(1, args.steps + 1):
    key, kd, kn, kt = jax.random.split(key, 4)
    s, a, x1, f = draw(kd, len(sg), len(sf))
    fr.append(float(f))
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
          'obs_dim': OBS_DIM, 'family': 'SA', 'lam': LAMBDA,
          'run_id': RUN_ID,
          'norm': {k: v.tolist() for k, v in nrm.items()},
          'config': dict(v2['config'], mixture='0.99 D_good + 0.01 '
                                               'D_fail^diverse (603)'),
          'n_params': n_par}
  with open(os.path.join(args.out, 'flow_v3.pkl'), 'wb') as f:
    pickle.dump(ckpt, f)
  json.dump({'run_id': RUN_ID, 'lambda': LAMBDA, 'curves': curves,
             'n_params': n_par, 'accounting': acct,
             'realized_fail_fraction_mean': float(np.mean(fr)),
             'provenance': prov, 'wall_time_sec': time.time() - t0},
            open(os.path.join(args.out, 'train_log.json'), 'w'), indent=2)
  print('\nfinal: train %.4f clean-val %.4f | realized fail frac %.4f -> %s'
        % (curves[-1]['train_loss_ema'], curves[-1]['clean_val_loss'],
           float(np.mean(fr)), args.out), flush=True)


if __name__ == '__main__':
  main()
