"""Longitudinal diagnostics across training checkpoints (50k / 100k / 150k).

Tests ONE hypothesis: is the behavior flow's excess boundary mass at |a| = 1
simply the result of under-training? The Euler-step sweep already ruled out
coarse integration as the cause (finer steps made it worse), so the remaining
cheap explanation is that the velocity field has not converged.

Every checkpoint is evaluated against ONE cached diagnostic bank -- the same
validation rows, the same true actions, the same Gaussian noise, the same
shuffled-context derangement, the same fixed-context set. All randomness is
generated once, before any checkpoint is loaded, and no PRNG is consumed inside
the per-checkpoint loop. The only variable is the parameters.

The validation flow-MSE reuses ``train_flow._fixed_val_batches`` verbatim, so
the number reported here is the same quantity the training loop logged (same
batches, same frozen (x0, t)), not a re-derived approximation.

Raw (unclipped) integration is reported at N=10 and N=50. N=50 is a FINER EULER
APPROXIMATION to the learned flow -- not the exact ODE solution, and not ground
truth. It is used only to ask whether the learned velocity field is aiming
outside the action box independently of discretization coarseness.

Trains nothing. No discriminator, no target policy, no classifier, no causal
weighting. Does not modify any checkpoint.

Run:

  python -m propensity.eval_training_progress \
      --config artifacts/propensity_flow/rockfall_v2_p30_h800_s0/config.json \
      --ckpt 50000=artifacts/propensity_flow/rockfall_v2_p30_h800_s0/latest.pkl \
      --ckpt 100000=artifacts/propensity_flow/rockfall_v2_p30_h800_s0_cont150k/checkpoint_100000.pkl \
      --ckpt 150000=artifacts/propensity_flow/rockfall_v2_p30_h800_s0_cont150k/checkpoint_150000.pkl
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from propensity import checkpoint as ckpt_mod
from propensity import flow as flow_mod
from propensity.dataset import BehaviorDataset
from propensity.train_flow import _fixed_val_batches   # identical val protocol

BOUNDARY_TOL = 1e-6
MAIN_N = 10          # the CFQL-aligned production sampler; unchanged here
FINE_N = 50          # finer Euler approximation, diagnostic only


def _at_boundary(a, tol=BOUNDARY_TOL):
  return np.abs(np.abs(a) - 1.0) <= tol


def _fmt(v, prec=4, width=9):
  return '[' + ' '.join(f'{float(x):{width}.{prec}f}' for x in v) + ']'


def build_parser():
  p = argparse.ArgumentParser(
      description='Longitudinal behavior-flow diagnostics across checkpoints.')
  p.add_argument('--config', required=True,
                 help='config.json describing dataset/split/architecture')
  p.add_argument('--ckpt', action='append', required=True,
                 help='label=path (repeatable), e.g. 100000=.../checkpoint_100000.pkl')
  p.add_argument('--seed', type=int, default=0,
                 help='diagnostic seed; fixes the shared bank for ALL ckpts')
  p.add_argument('--n-global', type=int, default=8192)
  p.add_argument('--n-fixed-contexts', type=int, default=8)
  p.add_argument('--n-fixed-samples', type=int, default=128)
  p.add_argument('--n-collapse-contexts', type=int, default=256)
  p.add_argument('--n-cond', type=int, default=1024)
  p.add_argument('--k-cond', type=int, default=32)
  p.add_argument('--n-train-mse', type=int, default=20,
                 help='fixed TRAIN batches for a comparable train flow-MSE')
  p.add_argument('--out', default=None)
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  with open(args.config) as f:
    meta = json.load(f)
  ckpts = []
  for spec in args.ckpt:
    label, _, path = spec.partition('=')
    ckpts.append((label, path))

  ds = BehaviorDataset(meta['dataset_path'], val_frac=meta['val_frac'],
                       seed=meta['split_seed'],
                       state_mode=('obs' if meta['context_mode'] == 'context'
                                   else 'state'),
                       split_level=meta['split_level'])
  if ds.fingerprint['sha256'] != meta['dataset_sha256']:
    print('ABORT: dataset sha256 mismatch vs the training record.')
    return 1
  A = ds.action_dim
  fc = meta['flow_config']
  net = flow_mod.make_flow_network(flow_mod.FlowConfig(
      context_dim=fc['context_dim'], action_dim=fc['action_dim'],
      hidden_sizes=tuple(fc['hidden_sizes']),
      time_features=fc['time_features'], time_max_freq=fc['time_max_freq'],
      use_layer_norm=fc['use_layer_norm']))

  # ======================================================================== #
  # Cached diagnostic bank -- generated ONCE, before any checkpoint loads.
  # ======================================================================== #
  rng = np.random.default_rng(args.seed)
  val_idx = ds._indices('val')                                # noqa: SLF001
  tr_idx = ds._indices('train')                               # noqa: SLF001

  n_a = min(args.n_global, val_idx.size)
  rows_a = rng.choice(val_idx, size=n_a, replace=False)
  ctx_a = jnp.asarray(ds._state[rows_a])                      # noqa: SLF001
  real_a = np.asarray(ds._action[rows_a])                     # noqa: SLF001
  z_a = jnp.asarray(rng.standard_normal((n_a, A), dtype=np.float32))

  n_b, k_b = args.n_fixed_contexts, args.n_fixed_samples
  rows_b = rng.choice(val_idx, size=n_b, replace=False)
  ctx_b = jnp.asarray(np.repeat(ds._state[rows_b], k_b, axis=0))  # noqa: SLF001
  z_b = jnp.asarray(rng.standard_normal((n_b * k_b, A), dtype=np.float32))

  n_cc = min(args.n_collapse_contexts, val_idx.size)
  rows_cc = rng.choice(val_idx, size=n_cc, replace=False)
  ctx_cc = jnp.asarray(np.repeat(ds._state[rows_cc], k_b, axis=0))  # noqa: SLF001
  z_cc = jnp.asarray(rng.standard_normal((n_cc * k_b, A), dtype=np.float32))

  n_c, k_c = min(args.n_cond, val_idx.size), args.k_cond
  rows_c = rng.choice(val_idx, size=n_c, replace=False)
  base_c = np.asarray(ds._state[rows_c])                      # noqa: SLF001
  act_c = np.asarray(ds._action[rows_c])                      # noqa: SLF001
  perm = (np.arange(n_c) + 1 + rng.integers(0, n_c - 1)) % n_c
  assert np.all(perm != np.arange(n_c))
  ctx_c = jnp.asarray(np.repeat(base_c, k_c, axis=0))
  ctx_c_shuf = jnp.asarray(np.repeat(base_c[perm], k_c, axis=0))
  z_c = jnp.asarray(rng.standard_normal((n_c * k_c, A), dtype=np.float32))

  # Fixed val batches -- IDENTICAL protocol/seed to the training loop.
  val_batches = _fixed_val_batches(ds, meta['batch_size'], meta['val_batches'],
                                   meta['val_noise_seed'], A)
  # Fixed TRAIN batches, same construction, so train MSE is comparable too.
  trng = np.random.default_rng(meta['val_noise_seed'] + 1)
  tkey = jax.random.PRNGKey(meta['val_noise_seed'] + 1)
  train_batches = []
  for _ in range(args.n_train_mse):
    r = tr_idx[trng.integers(0, tr_idx.size, size=meta['batch_size'])]
    tkey, k0, k1 = jax.random.split(tkey, 3)
    train_batches.append((
        jnp.asarray(ds._state[r]), jnp.asarray(ds._action[r]),   # noqa: SLF001
        jax.random.normal(k0, (meta['batch_size'], A)),
        jax.random.uniform(k1, (meta['batch_size'], 1))))

  bm_real = _at_boundary(real_a).mean(axis=0)
  real_std, real_mean = real_a.std(axis=0), real_a.mean(axis=0)

  print('=' * 80)
  print('propensity.eval_training_progress -- longitudinal behavior-flow check')
  print('=' * 80)
  print(f'dataset          {os.path.basename(meta["dataset_path"])}')
  print(f'context mode     {meta["context_mode"]}  '
        f'(s={meta["state_dim"]}, g_cmd={meta["goal_dim"]}, '
        f'c={fc["context_dim"]})  action_dim={A}')
  print(f'diagnostic seed  {args.seed}  (ONE cached bank shared by all '
        f'checkpoints; no PRNG consumed in the loop)')
  print(f'val protocol     train_flow._fixed_val_batches, seed '
        f'{meta["val_noise_seed"]}, {meta["val_batches"]} batches')
  print(f'samplers         main N={MAIN_N} (per-step clipped) | raw N='
        f'{MAIN_N},{FINE_N} (unclipped; N={FINE_N} is a FINER EULER '
        f'APPROXIMATION, not the exact ODE solution)')
  print(f'real reference   coord boundary '
        f'{float(_at_boundary(real_a).mean()):.4f} | row boundary '
        f'{float(_at_boundary(real_a).any(axis=1).mean()):.4f} | mean std '
        f'{real_std.mean():.4f}')
  print(f'                 dim1 real P(|a_1|=1) = {bm_real[1]:.4f}')
  print()

  rows = []
  for label, path in ckpts:
    step, params, _ = ckpt_mod.load_checkpoint(path)

    clipped = jax.jit(lambda c, z, p=params: flow_mod.sample_actions_from_noise(
        net.apply, p, c, z, num_steps=MAIN_N, clip=True))
    raw10 = jax.jit(lambda c, z, p=params: flow_mod.sample_actions_from_noise(
        net.apply, p, c, z, num_steps=MAIN_N, clip=False))
    raw50 = jax.jit(lambda c, z, p=params: flow_mod.sample_actions_from_noise(
        net.apply, p, c, z, num_steps=FINE_N, clip=False))
    lossf = jax.jit(lambda p, c, a, x0, t: flow_mod.flow_matching_loss_fixed(
        net.apply, p, c, a, x0, t))

    # ---- A. flow training quality -------------------------------------- #
    val_mse = float(np.mean([float(lossf(params, *b)) for b in val_batches]))
    train_mse = float(np.mean([float(lossf(params, *b)) for b in train_batches]))

    # ---- B. main clipped sampler, N=10 ---------------------------------- #
    gen = np.asarray(clipped(ctx_a, z_a))
    atb = _at_boundary(gen)
    coord_b, row_b = float(atb.mean()), float(atb.any(axis=1).mean())
    bm_gen = atb.mean(axis=0)

    # ---- C. raw unclipped --------------------------------------------- #
    g10 = np.asarray(raw10(ctx_a, z_a))
    g50 = np.asarray(raw50(ctx_a, z_a))
    oob10 = float(np.mean((g10 < -1.0) | (g10 > 1.0)))
    oob10_row = float(np.mean(np.any((g10 < -1.0) | (g10 > 1.0), axis=1)))
    oob50 = float(np.mean((g50 < -1.0) | (g50 > 1.0)))
    oob50_row = float(np.mean(np.any((g50 < -1.0) | (g50 > 1.0), axis=1)))

    # ---- D. distribution ----------------------------------------------- #
    gen_std, gen_mean = gen.std(axis=0), gen.mean(axis=0)
    std_ratio = float(gen_std.mean() / real_std.mean())

    # ---- E. conditionality --------------------------------------------- #
    def nearest(cr):
      s = np.asarray(clipped(cr, z_c)).reshape(n_c, k_c, A)
      d = np.linalg.norm(s - act_c[:, None, :], axis=-1)
      return d.min(axis=1)

    d_ok, d_bad = nearest(ctx_c), nearest(ctx_c_shuf)
    win = float(np.mean(d_ok < d_bad))

    # ---- F. stochasticity ---------------------------------------------- #
    s_b = np.asarray(clipped(ctx_b, z_b)).reshape(n_b, k_b, A)
    ctx_std = s_b.std(axis=1).mean(axis=1)
    s_cc = np.asarray(clipped(ctx_cc, z_cc)).reshape(n_cc, k_b, A)
    sd_cc = s_cc.std(axis=1)
    dead = sd_cc < 1e-6

    rows.append({
        'label': label, 'path': path, 'step': int(step),
        'train_flow_mse_fixed': train_mse, 'val_flow_mse_fixed': val_mse,
        'clipped_coord_boundary': coord_b, 'clipped_row_boundary': row_b,
        'boundary_per_dim': bm_gen.tolist(),
        'boundary_ratio_per_dim': (bm_gen / np.maximum(bm_real, 1e-9)).tolist(),
        'dim1_boundary': float(bm_gen[1]),
        'dim1_ratio': float(bm_gen[1] / max(bm_real[1], 1e-9)),
        'raw_oob_n10': oob10, 'raw_oob_row_n10': oob10_row,
        'raw_oob_n50': oob50, 'raw_oob_row_n50': oob50_row,
        'gen_mean_per_dim': gen_mean.tolist(),
        'gen_std_per_dim': gen_std.tolist(),
        'gen_mean_std': float(gen_std.mean()),
        'gen_over_real_std': std_ratio,
        'correct_mean_min_l2': float(d_ok.mean()),
        'shuffled_mean_min_l2': float(d_bad.mean()),
        'conditional_win_rate': win,
        'median_ctx_mean_std': float(np.median(ctx_std)),
        'collapsed_contexts': int((ctx_std < 1e-6).sum()),
        'collapsed_contexts_of': n_b,
        'per_dim_collapse_fraction': float(dead.mean()),
        'contexts_any_dim_collapsed': int(dead.any(axis=1).sum()),
        'nan_inf': int((~np.isfinite(gen)).sum() + (~np.isfinite(s_cc)).sum()),
    })
    r = rows[-1]
    print(f'  [{label:>7}] val {val_mse:.4f} | boundary {coord_b:.4f} '
          f'| dim1 {bm_gen[1]:.4f} | rawOOB10 {oob10:.4f} rawOOB50 {oob50:.4f} '
          f'| std {std_ratio:.4f} | corr {d_ok.mean():.4f} win {win:.4f}',
          flush=True)

  # ---- summary table ---------------------------------------------------- #
  print()
  print('LONGITUDINAL SUMMARY')
  print(f'  real reference: coord boundary '
        f'{float(_at_boundary(real_a).mean()) * 100:.2f}%  '
        f'dim1 {bm_real[1] * 100:.2f}%  mean std {real_std.mean():.4f}')
  hdr = (f'{"ckpt":>8} {"trainMSE":>9} {"valMSE":>8} {"coordB%":>8} '
         f'{"dim1B%":>8} {"rawOOB10":>9} {"rawOOB50":>9} {"gen/real":>9} '
         f'{"corrL2":>8} {"shufL2":>8} {"win":>7} {"collDim%":>9}')
  print(hdr)
  print('-' * len(hdr))
  for r in rows:
    print(f'{r["label"]:>8} {r["train_flow_mse_fixed"]:>9.4f} '
          f'{r["val_flow_mse_fixed"]:>8.4f} '
          f'{100 * r["clipped_coord_boundary"]:>8.2f} '
          f'{100 * r["dim1_boundary"]:>8.2f} {r["raw_oob_n10"]:>9.4f} '
          f'{r["raw_oob_n50"]:>9.4f} {r["gen_over_real_std"]:>9.4f} '
          f'{r["correct_mean_min_l2"]:>8.4f} '
          f'{r["shuffled_mean_min_l2"]:>8.4f} '
          f'{r["conditional_win_rate"]:>7.4f} '
          f'{100 * r["per_dim_collapse_fraction"]:>9.2f}')

  print()
  print('PER-DIMENSION BOUNDARY MASS  P(|a_i| = 1)   [clipped sampler, N=10]')
  print(f'{"dim":>4} {"real":>8} ' +
        ' '.join(f'{r["label"]:>18}' for r in rows))
  for i in range(A):
    cells = ' '.join(f'{r["boundary_per_dim"][i]:>10.4f}'
                     f'({r["boundary_ratio_per_dim"][i]:>5.2f}x)' for r in rows)
    print(f'{i:>4} {bm_real[i]:>8.4f} {cells}')

  print()
  print('STOCHASTICITY / STABILITY')
  for r in rows:
    print(f'  {r["label"]:>7}  median ctx std {r["median_ctx_mean_std"]:.4f}  '
          f'| collapsed ctx {r["collapsed_contexts"]}/'
          f'{r["collapsed_contexts_of"]}  '
          f'| (ctx,dim) collapse {100 * r["per_dim_collapse_fraction"]:.2f}%  '
          f'| ctx with >=1 dead dim {r["contexts_any_dim_collapsed"]}/256  '
          f'| NaN/Inf {r["nan_inf"]}')

  payload = {
      'diagnostic_seed': args.seed,
      'config': args.config,
      'dataset': meta['dataset_path'],
      'dataset_sha256': meta['dataset_sha256'],
      'main_flow_steps': MAIN_N, 'fine_flow_steps': FINE_N,
      'fine_n_caveat': 'N=50 is a finer Euler approximation to the learned '
                       'flow, NOT the exact ODE solution or ground truth',
      'shared_bank': {
          'shared_across_all_checkpoints': [
              'validation rows', 'true actions', 'contexts',
              'Gaussian noise banks (explicit z)',
              'shuffled-context derangement', 'fixed-context set',
              'fixed val/train batches and their (x0, t)'],
          'prng_consumed_in_loop': False,
          'cache_signature': {'rows_a_sum': int(rows_a.sum()),
                              'rows_c_sum': int(rows_c.sum()),
                              'perm_sum': int(perm.sum()),
                              'z_a_sum': float(np.asarray(z_a).sum())},
      },
      'real': {'coord_boundary': float(_at_boundary(real_a).mean()),
               'row_boundary': float(_at_boundary(real_a).any(axis=1).mean()),
               'boundary_per_dim': bm_real.tolist(),
               'mean_per_dim': real_mean.tolist(),
               'std_per_dim': real_std.tolist(),
               'mean_std': float(real_std.mean())},
      'checkpoints': rows,
  }
  out = args.out or 'artifacts/propensity_flow/training_progress.json'
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with open(out, 'w') as f:
    json.dump(payload, f, indent=2)
  print(f'\nresults -> {out}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
