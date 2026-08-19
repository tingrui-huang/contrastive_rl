"""V2: failure-LOCAL balanced Flow, action-conditioned family only.

V1 showed that source-level mixing over the whole bad-demo pool never gives
the fatal mode meaningful gradient exposure (r_fatal = 7.8e-4, ~16 repeats
per distinct fatal transition even at beta=0.20). V2 tests the direct
hypothesis: reweight the FATAL TRANSITION ITSELF.

    D_batch ~ (1 - lambda) * D_good  +  lambda * D_fail

  D_good : the same authoritative CLEAN expert factual transitions used by
           V0/V0.5 (frozen V0 train episodes only).
  D_fail : EXACTLY the 196 audited settled-fatal transitions, one per
           naturally occurring death,
               (s_predeath, a_predeath, s'_settled)
           = (obs[e, c], act[e, c], obs[e, lengths[e]-1]) of each dead
           bad-demo episode. No pre-fatal windows, no previous 2/3/5 steps,
           no mild contacts, no wall collisions, no fallen/stuck transitions,
           no counterfactual or forced-mask transitions, and NO ordinary
           bad-demo transitions (that source is deliberately excluded here so
           the V1 confound is removed).

lambda is the fraction of minibatch slots drawn from D_fail. It is NOT a
calibrated fatal probability: V2 is a SUPPORT-ORIENTED conditional generator,
trained so that a rare-but-factual mode survives, not a calibrated estimator
of P(s'|s,a). No calibration claim is made.

The model still receives ONLY (s, a). _dead / terminal metadata is used
offline to build D_fail and never enters the network. Architecture,
objective, optimizer, lr, batch, 20k budget, frozen normalization, raw
actions and the clean split are all exactly V0.5.

Usage:
  python scripts/train_flow_v2.py --lam 0.025
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

V0_DIR = 'artifacts/flow_v0_clean'
BAD_DIR = 'artifacts/bad_demo_fixed'
BAD_NAME = 'bad_demo_blind_p30_h800_settle80'
OUT_ROOT = 'artifacts/flow_v2_failure_local'
N_FAIL_EXPECTED = 196


def build_fail_pool(bad_dir, name):
  """The 196 settled-fatal transitions, one per dead episode."""
  d = np.load(os.path.join(bad_dir, name + '.npz'), allow_pickle=True)
  s = np.load(os.path.join(bad_dir, name + '_sidecar.npz'), allow_pickle=True)
  obs, act = np.asarray(d['obs'], np.float32), np.asarray(d['act'], np.float32)
  ln = np.asarray(d['lengths'], np.int64)
  dead = np.asarray(s['dead'], bool)
  col = np.asarray(s['collapse_step'], np.int64)
  ids = np.where(dead)[0]
  S, A, S2 = [], [], []
  for e in ids:
    c, last = int(col[e]), int(ln[e]) - 1
    assert last == c + 1, 'dead episode not truncated at collapse+1'
    S.append(obs[e, c, :OBS_DIM])
    A.append(act[e, c])
    S2.append(obs[e, last, :OBS_DIM])
  S, A, S2 = np.stack(S), np.stack(A), np.stack(S2)
  return S, A, S2 - S, ids


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--lam', type=float, required=True)
  ap.add_argument('--v0-dir', default=V0_DIR)
  ap.add_argument('--bad-dir', default=BAD_DIR)
  ap.add_argument('--bad-name', default=BAD_NAME)
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
  # pre-registered run ids (exact names from the V2 spec)
  RUN_IDS = {0.01: 'V2-SA-l001', 0.025: 'V2-SA-l0025',
             0.05: 'V2-SA-l005', 0.10: 'V2-SA-l010'}
  run_id = RUN_IDS.get(round(args.lam, 6),
                       'V2-SA-lam%g' % args.lam)
  out = os.path.join(args.out_root, run_id)
  os.makedirs(out, exist_ok=True)

  # ---- provenance ---------------------------------------------------------
  split = json.load(open(os.path.join(args.v0_dir, 'split_manifest.json')))
  good_npz = split['npz']
  good_sha = C.sha256_file(good_npz)
  assert good_sha == split['npz_sha256'], 'clean dataset drifted'
  bad_man = json.load(open(os.path.join(args.bad_dir,
                                        'collection_manifest.json')))
  bad_npz = os.path.join(args.bad_dir, args.bad_name + '.npz')
  assert C.sha256_file(bad_npz) == bad_man['npz_sha256'], 'bad-demo drifted'
  assert bad_man['death_settle_substeps'] == 80
  nz_ = np.load(os.path.join(args.v0_dir, 'norm_stats.npz'))
  nrm = {k: np.asarray(nz_[k], np.float32) for k in
         ('state_mean', 'state_std', 'delta_mean', 'delta_std')}

  tr_eps = np.asarray(split['train_episode_ids'], np.int64)
  va_eps = np.asarray(split['val_episode_ids'], np.int64)
  sg, ag, dg = stack_pairs(good_npz, tr_eps)
  sv, av, dv = stack_pairs(good_npz, va_eps)
  sf, af, df, fail_eps = build_fail_pool(args.bad_dir, args.bad_name)
  assert len(sf) == N_FAIL_EXPECTED, 'expected %d fatal transitions, got %d' \
      % (N_FAIL_EXPECTED, len(sf))

  dev = jax.devices()
  print('=' * 68)
  print('RUN            :', run_id)
  print('git commit     :', C.git_commit())
  print('lambda         : %.4f  (fraction of minibatch from D_fail)' % args.lam)
  print('clean npz sha  :', good_sha)
  print('bad-demo sha   :', bad_man['npz_sha256'])
  print('D_good / D_fail: %d / %d transitions' % (len(sg), len(sf)))
  print('normalization  :', os.path.join(args.v0_dir, 'norm_stats.npz'))
  print('seed / steps   : %d / %d' % (args.seed, args.steps))
  print('jax devices    : %s (%s)' % (dev, jax.default_backend()))
  print('=' * 68, flush=True)

  # ---- how extreme are the fatal deltas under the FROZEN clean stats? -----
  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  nfd, ngd = nzd(df), nzd(dg)
  extremity = {
      'fatal_delta_l2_normalized': {
          'median': float(np.median(np.linalg.norm(nfd, axis=1))),
          'p10': float(np.percentile(np.linalg.norm(nfd, axis=1), 10)),
          'p90': float(np.percentile(np.linalg.norm(nfd, axis=1), 90)),
          'max': float(np.linalg.norm(nfd, axis=1).max())},
      'clean_delta_l2_normalized': {
          'median': float(np.median(np.linalg.norm(ngd, axis=1))),
          'p90': float(np.percentile(np.linalg.norm(ngd, axis=1), 90)),
          'max': float(np.linalg.norm(ngd, axis=1).max())},
      'fatal_max_abs_per_dim_sigma': float(np.abs(nfd).max()),
      'clean_max_abs_per_dim_sigma': float(np.abs(ngd).max()),
      'frac_fatal_beyond_clean_p999': float(
          (np.linalg.norm(nfd, axis=1)
           > np.percentile(np.linalg.norm(ngd, axis=1), 99.9)).mean()),
      'note': ('fatal deltas expressed in the FROZEN clean normalization; '
               'normalization was NOT recomputed after adding them')}
  print('fatal delta L2 (norm): median %.2f p90 %.2f max %.2f  |  clean: '
        'median %.2f p90 %.2f max %.2f'
        % (extremity['fatal_delta_l2_normalized']['median'],
           extremity['fatal_delta_l2_normalized']['p90'],
           extremity['fatal_delta_l2_normalized']['max'],
           extremity['clean_delta_l2_normalized']['median'],
           extremity['clean_delta_l2_normalized']['p90'],
           extremity['clean_delta_l2_normalized']['max']), flush=True)

  # ---- training accounting -------------------------------------------------
  acct = {'lambda': args.lam,
          'expected_fail_samples_per_batch': args.lam * args.batch,
          'total_fail_gradient_exposures': args.lam * args.batch * args.steps,
          'avg_repeats_per_fatal_transition':
              args.lam * args.batch * args.steps / len(sf),
          'v1_best_comparison': ('V1 beta=0.20 gave 0.16 fail/batch, 3196 '
                                 'total, ~16 repeats each')}
  print('accounting: %.1f fail/batch | %.0f total exposures | ~%.0f repeats '
        'per fatal transition'
        % (acct['expected_fail_samples_per_batch'],
           acct['total_fail_gradient_exposures'],
           acct['avg_repeats_per_fatal_transition']), flush=True)

  def nz(x, m, s):
    return (x - m) / s
  Sg = jnp.asarray(nz(sg, nrm['state_mean'], nrm['state_std']))
  Xg = jnp.asarray(nzd(dg))
  Ag = jnp.asarray(ag)
  Sf = jnp.asarray(nz(sf, nrm['state_mean'], nrm['state_std']))
  Xf = jnp.asarray(nzd(df))
  Af = jnp.asarray(af)
  Sv = jnp.asarray(nz(sv, nrm['state_mean'], nrm['state_std']))
  Xv = jnp.asarray(nzd(dv))
  Av = jnp.asarray(av)

  net = make_net_a(hidden, OBS_DIM)
  key = jax.random.PRNGKey(args.seed)
  key, k0 = jax.random.split(key)
  params = net.init(k0, Xg[:2], jnp.zeros((2, 1)), Sg[:2], Ag[:2])
  n_par = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  opt = optax.adam(args.lr)
  opt_state = opt.init(params)
  print('SA vector field %s | %d params' % (hidden, n_par), flush=True)

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
    from_fail = jax.random.uniform(k3, (args.batch,)) < args.lam
    s = jnp.where(from_fail[:, None], Sf[i_f], Sg[ig])
    a = jnp.where(from_fail[:, None], Af[i_f], Ag[ig])
    x1 = jnp.where(from_fail[:, None], Xf[i_f], Xg[ig])
    return s, a, x1, from_fail.mean()

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
          'obs_dim': OBS_DIM, 'family': 'SA', 'lam': args.lam,
          'norm': {k: v.tolist() for k, v in nrm.items()},
          'config': {'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
                     'seed': args.seed, 'hidden': list(hidden),
                     'objective': 'straight flow matching, target x1-x0',
                     'target': 'normalized delta_s',
                     'conditioning': 's and a (raw); no failure label',
                     'mixture': 'failure-local lambda over (D_good, D_fail)',
                     'purpose': 'support-oriented generator, NOT calibrated'},
          'n_params': n_par}
  with open(os.path.join(out, 'flow_v2.pkl'), 'wb') as f:
    pickle.dump(ckpt, f)
  json.dump({'run_id': run_id, 'lambda': args.lam, 'family': 'SA',
             'curves': curves, 'n_params': n_par, 'config': ckpt['config'],
             'realized_fail_fraction_mean': float(np.mean(fr)),
             'accounting': acct,
             'fatal_delta_extremity': extremity,
             'pools': {'good_transitions': int(len(sg)),
                       'fail_transitions': int(len(sf)),
                       'fail_source_episode_ids': fail_eps.tolist(),
                       'clean_val_transitions': int(len(sv))},
             'provenance': {'git_commit': C.git_commit(),
                            'clean_npz': good_npz, 'clean_sha256': good_sha,
                            'bad_npz': bad_npz,
                            'bad_sha256': bad_man['npz_sha256'],
                            'norm_source': os.path.join(args.v0_dir,
                                                        'norm_stats.npz'),
                            'jax_backend': jax.default_backend()},
             'wall_time_sec': time.time() - t0},
            open(os.path.join(out, 'train_log.json'), 'w'), indent=2)
  print('\nfinal: train %.4f clean-val %.4f | realized fail frac %.4f -> %s'
        % (curves[-1]['train_loss_ema'], curves[-1]['clean_val_loss'],
           float(np.mean(fr)), out), flush=True)


if __name__ == '__main__':
  main()
