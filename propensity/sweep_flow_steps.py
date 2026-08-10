"""Controlled sweep over the number of Euler integration steps N.

Question: is the excess generated boundary mass (|a| = 1) caused by COARSE
NUMERICAL INTEGRATION, or by the learned velocity field / the Gaussian-base +
hard-clip design?

Method: hold everything fixed except ``num_steps``. The SAME trained checkpoint
is reused (never retrained), and every source of randomness is pre-generated
ONCE before the sweep and replayed identically for each N:

  * validation row indices for each diagnostic block;
  * the true validation actions those rows carry;
  * the base Gaussian noise z for every sampled action -- passed EXPLICITLY to
    ``flow.sample_actions_from_noise``, so "same z" is enforced by construction
    rather than inferred from PRNG-shape invariance;
  * the derangement permutation used by the shuffled-context diagnostic;
  * the fixed-context diagnostic contexts and their noise bank.

The only variable across arms is N. No PRNG is consumed inside the per-N loop.

This script trains nothing, loads no target policy, builds no discriminator,
constructs no environment, and touches neither the dataset nor crl/.

Run:

  python -m propensity.sweep_flow_steps \
      --ckpt-dir artifacts/propensity_flow/rockfall_v2_p30_h800_s0 \
      --checkpoint best --flow-steps 5,10,20,50
"""
import argparse
import csv
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from propensity import checkpoint as ckpt_mod
from propensity import flow as flow_mod
from propensity.dataset import BehaviorDataset

#: Clipping writes EXACTLY lo/hi, so an exact float comparison is the correct
#: boundary test. A tiny tolerance is kept so a value that legitimately lands on
#: the boundary through arithmetic is not missed.
BOUNDARY_TOL = 1e-6


def _fmt(v, prec=4, width=9):
  return '[' + ' '.join(f'{float(x):{width}.{prec}f}' for x in v) + ']'


def _at_boundary(a, tol=BOUNDARY_TOL):
  return np.abs(np.abs(a) - 1.0) <= tol


def build_parser():
  p = argparse.ArgumentParser(
      description='Euler-step sweep for the behavior flow sampler. Sampling '
                  'only: no retraining, no Stage-3 code.')
  p.add_argument('--ckpt-dir', required=True)
  p.add_argument('--checkpoint', default='best', choices=('best', 'latest'))
  p.add_argument('--dataset', default=None)
  p.add_argument('--flow-steps', default='5,10,20,50',
                 help='comma-separated N values to sweep')
  p.add_argument('--seed', type=int, default=0,
                 help='DIAGNOSTIC seed: fixes every cached random array, '
                      'shared by all N (recorded in the output).')
  p.add_argument('--n-global', type=int, default=8192)
  p.add_argument('--n-fixed-contexts', type=int, default=8)
  p.add_argument('--n-fixed-samples', type=int, default=128)
  p.add_argument('--n-collapse-contexts', type=int, default=256)
  p.add_argument('--n-cond', type=int, default=1024)
  p.add_argument('--k-cond', type=int, default=32)
  p.add_argument('--timing-repeats', type=int, default=3)
  p.add_argument('--out', default=None,
                 help='JSON path (default <ckpt-dir>/flow_step_sweep.json)')
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  steps_list = [int(x) for x in args.flow_steps.split(',') if x]

  meta = ckpt_mod.load_metadata(args.ckpt_dir)
  dataset = args.dataset or meta['dataset_path']
  ds = BehaviorDataset(dataset, val_frac=meta['val_frac'],
                       seed=meta['split_seed'],
                       state_mode=('obs' if meta['context_mode'] == 'context'
                                   else 'state'),
                       split_level=meta['split_level'])
  if ds.fingerprint['sha256'] != meta['dataset_sha256']:
    print('ABORT: dataset sha256 does not match the training record.')
    return 1
  A = ds.action_dim

  fc = meta['flow_config']
  net = flow_mod.make_flow_network(flow_mod.FlowConfig(
      context_dim=fc['context_dim'], action_dim=fc['action_dim'],
      hidden_sizes=tuple(fc['hidden_sizes']),
      time_features=fc['time_features'], time_max_freq=fc['time_max_freq'],
      use_layer_norm=fc['use_layer_norm']))
  ckpt_path = os.path.join(args.ckpt_dir, f'{args.checkpoint}.pkl')
  step, params, _ = ckpt_mod.load_checkpoint(ckpt_path)

  # ======================================================================== #
  # Pre-generate EVERY random array once. Nothing below draws randomness.
  # ======================================================================== #
  rng = np.random.default_rng(args.seed)
  val_idx = ds._indices('val')                                # noqa: SLF001

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
  ctx_c_base = np.asarray(ds._state[rows_c])                  # noqa: SLF001
  act_c = np.asarray(ds._action[rows_c])                      # noqa: SLF001
  perm = (np.arange(n_c) + 1 + rng.integers(0, n_c - 1)) % n_c   # derangement
  assert np.all(perm != np.arange(n_c))
  ctx_c = jnp.asarray(np.repeat(ctx_c_base, k_c, axis=0))
  ctx_c_shuf = jnp.asarray(np.repeat(ctx_c_base[perm], k_c, axis=0))
  # ONE noise bank, reused by both the correct- and shuffled-context arms, so
  # the only difference between them is the conditioning vector.
  z_c = jnp.asarray(rng.standard_normal((n_c * k_c, A), dtype=np.float32))

  cache_sig = {
      'rows_a_sha': int(rows_a.sum()), 'rows_c_sha': int(rows_c.sum()),
      'perm_sha': int(perm.sum()),
      'z_a_sum': float(np.asarray(z_a).sum()),
      'z_c_sum': float(np.asarray(z_c).sum()),
  }

  print('=' * 78)
  print('propensity.sweep_flow_steps -- Euler integration-step sweep')
  print('=' * 78)
  print(f'checkpoint       {ckpt_path}  (step {step})  [REUSED, not retrained]')
  print(f'dataset          {os.path.basename(dataset)}')
  print(f'context mode     {meta["context_mode"]}  '
        f'(s={meta["state_dim"]}, g_cmd={meta["goal_dim"]}, '
        f'c={fc["context_dim"]})  action_dim={A}')
  print(f'diagnostic seed  {args.seed}   (all random arrays cached ONCE; '
        f'identical for every N)')
  print(f'sweep            N = {steps_list}')
  print(f'blocks           A n={n_a} | B {n_b}x{k_b} | collapse {n_cc}x{k_b} '
        f'| C {n_c}xK={k_c}')
  print(f'boundary test    | |a| - 1 | <= {BOUNDARY_TOL}')

  # jitted per (N, batch-shape) -- N is a Python constant captured at trace time
  def make_sampler(n_steps, clip=True):
    return jax.jit(lambda c, z: flow_mod.sample_actions_from_noise(
        net.apply, params, c, z, num_steps=n_steps, clip=clip))

  def make_overshoot(n_steps):
    return jax.jit(lambda c, z: flow_mod.integrate_with_overshoot(
        net.apply, params, c, z, num_steps=n_steps))

  bm_real = _at_boundary(real_a).mean(axis=0)
  st_real = {'mean': real_a.mean(axis=0), 'std': real_a.std(axis=0),
             'min': real_a.min(axis=0), 'max': real_a.max(axis=0)}

  results = []
  for N in steps_list:
    t_block = time.time()
    smp = make_sampler(N, clip=True)
    smp_raw = make_sampler(N, clip=False)
    ovr = make_overshoot(N)

    # ---- A. boundary + distribution ------------------------------------- #
    gen_a = np.asarray(smp(ctx_a, z_a))
    gen_raw = np.asarray(smp_raw(ctx_a, z_a))
    _, over = ovr(ctx_a, z_a)                       # [N, n_a, A]
    over = np.asarray(over)
    fired = over > 0.0
    clip_rate = float(fired.mean())
    ov_vals = over[fired]
    ov_med = float(np.median(ov_vals)) if ov_vals.size else 0.0
    ov_p90 = float(np.percentile(ov_vals, 90)) if ov_vals.size else 0.0
    ov_max = float(ov_vals.max()) if ov_vals.size else 0.0

    atb = _at_boundary(gen_a)
    coord_b = float(atb.mean())
    row_b = float(atb.any(axis=1).mean())
    bm_gen = atb.mean(axis=0)
    oob = float(np.mean((gen_a < -1.0) | (gen_a > 1.0)))
    raw_oob = float(np.mean((gen_raw < -1.0) | (gen_raw > 1.0)))

    gen_mean, gen_std = gen_a.mean(axis=0), gen_a.std(axis=0)
    std_ratio = float(gen_std.mean() / st_real['std'].mean())

    # ---- B. fixed-context stochasticity --------------------------------- #
    s_b = np.asarray(smp(ctx_b, z_b)).reshape(n_b, k_b, A)
    sd_b = s_b.std(axis=1)                                    # [n_b, A]
    ctx_mean_std = sd_b.mean(axis=1)
    pair = np.array([
        float(np.mean(np.linalg.norm(s_b[i, ::2] - s_b[i, 1::2], axis=-1)))
        for i in range(n_b)])
    collapsed_ctx = int((ctx_mean_std < 1e-6).sum())

    s_cc = np.asarray(smp(ctx_cc, z_cc)).reshape(n_cc, k_b, A)
    sd_cc = s_cc.std(axis=1)
    dead = sd_cc < 1e-6
    pinned = _at_boundary(s_cc.mean(axis=1))
    frac_pinned = float(pinned[dead].mean()) if dead.any() else 0.0

    # ---- C. conditionality ---------------------------------------------- #
    def nearest(ctx_rep):
      s = np.asarray(smp(ctx_rep, z_c)).reshape(n_c, k_c, A)
      d = np.linalg.norm(s - act_c[:, None, :], axis=-1)
      return d.min(axis=1), d.mean(axis=1)

    d_ok, dm_ok = nearest(ctx_c)
    d_bad, dm_bad = nearest(ctx_c_shuf)
    win = float(np.mean(d_ok < d_bad))

    # ---- E. stability + timing ------------------------------------------ #
    nan_inf = int((~np.isfinite(gen_a)).sum() + (~np.isfinite(s_b)).sum()
                  + (~np.isfinite(s_cc)).sum())
    jax.block_until_ready(smp(ctx_a, z_a))            # warm (already compiled)
    t0 = time.time()
    for _ in range(args.timing_repeats):
      jax.block_until_ready(smp(ctx_a, z_a))
    sample_s = (time.time() - t0) / args.timing_repeats

    results.append({
        'flow_steps': N,
        'coord_boundary_fraction': coord_b,
        'row_boundary_fraction': row_b,
        'clip_update_fraction': clip_rate,
        'overshoot_median': ov_med, 'overshoot_p90': ov_p90,
        'overshoot_max': ov_max,
        'out_of_box_fraction': oob,
        'raw_unclipped_out_of_box_fraction': raw_oob,
        'gen_mean_per_dim': gen_mean.tolist(),
        'gen_std_per_dim': gen_std.tolist(),
        'gen_mean_std': float(gen_std.mean()),
        'gen_over_real_std': std_ratio,
        'boundary_mass_gen_per_dim': bm_gen.tolist(),
        'boundary_ratio_per_dim': (bm_gen / np.maximum(bm_real, 1e-9)).tolist(),
        'correct_mean_min_l2': float(d_ok.mean()),
        'correct_median_min_l2': float(np.median(d_ok)),
        'shuffled_mean_min_l2': float(d_bad.mean()),
        'shuffled_median_min_l2': float(np.median(d_bad)),
        'correct_mean_of_mean_l2': float(dm_ok.mean()),
        'shuffled_mean_of_mean_l2': float(dm_bad.mean()),
        'conditional_win_rate': win,
        'median_ctx_mean_std': float(np.median(ctx_mean_std)),
        'min_ctx_mean_std': float(ctx_mean_std.min()),
        'max_ctx_mean_std': float(ctx_mean_std.max()),
        'mean_pairwise_l2': float(pair.mean()),
        'collapsed_contexts': collapsed_ctx,
        'collapsed_contexts_of': n_b,
        'per_dim_collapse_fraction': float(dead.mean()),
        'per_dim_collapse_pinned_fraction': frac_pinned,
        'contexts_any_dim_collapsed': int(dead.any(axis=1).sum()),
        'contexts_all_dims_collapsed': int(dead.all(axis=1).sum()),
        'median_ctx_mean_std_wide': float(np.median(sd_cc.mean(axis=1))),
        'nan_inf_count': nan_inf,
        'sample_seconds_block_A': sample_s,
        'block_wall_clock_s': round(time.time() - t_block, 2),
    })
    r = results[-1]
    print(f'  [N={N:>3}] boundary {coord_b:.4f} | clip-rate {clip_rate:.4f} '
          f'| std ratio {std_ratio:.4f} | correct {d_ok.mean():.4f} '
          f'| shuffled {d_bad.mean():.4f} | win {win:.4f} '
          f'| {sample_s * 1000:.0f} ms', flush=True)

  # ======================================================================== #
  # Summary table
  # ======================================================================== #
  print()
  print('SUMMARY  (real coord boundary fraction = '
        f'{float(_at_boundary(real_a).mean()):.4f}, '
        f'real mean std = {st_real["std"].mean():.4f})')
  hdr = (f'{"N":>4} {"coordB%":>8} {"rowB%":>8} {"clipUpd%":>9} '
         f'{"gen/real std":>12} {"corrL2":>8} {"shufL2":>8} {"win":>7} '
         f'{"collCtx":>8} {"perDimColl":>11} {"time_ms":>8}')
  print(hdr)
  print('-' * len(hdr))
  for r in results:
    print(f'{r["flow_steps"]:>4} {100 * r["coord_boundary_fraction"]:>8.2f} '
          f'{100 * r["row_boundary_fraction"]:>8.2f} '
          f'{100 * r["clip_update_fraction"]:>9.2f} '
          f'{r["gen_over_real_std"]:>12.4f} {r["correct_mean_min_l2"]:>8.4f} '
          f'{r["shuffled_mean_min_l2"]:>8.4f} '
          f'{r["conditional_win_rate"]:>7.4f} '
          f'{r["collapsed_contexts"]:>4}/{r["collapsed_contexts_of"]:<3} '
          f'{100 * r["per_dim_collapse_fraction"]:>10.2f}% '
          f'{1000 * r["sample_seconds_block_A"]:>8.1f}')

  print()
  print('PER-DIMENSION BOUNDARY MASS  P(|a_i| = 1)')
  print(f'{"dim":>4} {"real":>8} ' +
        ' '.join(f'{"N=" + str(N):>17}' for N in steps_list))
  for i in range(A):
    cells = ' '.join(
        f'{r["boundary_mass_gen_per_dim"][i]:>9.4f}'
        f'({r["boundary_ratio_per_dim"][i]:>6.2f}x)' for r in results)
    print(f'{i:>4} {bm_real[i]:>8.4f} {cells}')

  print()
  print('OVERSHOOT BEFORE CLIP (per Euler update, where the clip fired)')
  print(f'{"N":>4} {"median":>10} {"p90":>10} {"max":>10}')
  for r in results:
    print(f'{r["flow_steps"]:>4} {r["overshoot_median"]:>10.4f} '
          f'{r["overshoot_p90"]:>10.4f} {r["overshoot_max"]:>10.4f}')

  print()
  print('STABILITY')
  for r in results:
    print(f'  N={r["flow_steps"]:>3}  NaN/Inf={r["nan_inf_count"]}  '
          f'raw-unclipped OOB={r["raw_unclipped_out_of_box_fraction"]:.4f}  '
          f'clipped OOB={r["out_of_box_fraction"]:.4f}')

  payload = {
      'checkpoint': ckpt_path, 'step': int(step), 'dataset': dataset,
      'retrained': False,
      'diagnostic_seed': args.seed,
      'fairness': {
          'shared_across_all_N': [
              'validation row indices', 'true validation actions', 'contexts',
              'base Gaussian noise z (passed explicitly)',
              'shuffled-context derangement permutation',
              'fixed-context contexts and noise bank',
              'collapse-check contexts and noise bank'],
          'only_variable': 'num_steps',
          'prng_consumed_inside_sweep_loop': False,
          'cache_signature': cache_sig,
      },
      'boundary_tolerance': BOUNDARY_TOL,
      'real': {'boundary_mass_per_dim': bm_real.tolist(),
               'coord_boundary_fraction': float(_at_boundary(real_a).mean()),
               'row_boundary_fraction':
                   float(_at_boundary(real_a).any(axis=1).mean()),
               'mean_per_dim': st_real['mean'].tolist(),
               'std_per_dim': st_real['std'].tolist(),
               'mean_std': float(st_real['std'].mean())},
      'blocks': {'n_global': n_a, 'n_fixed_contexts': n_b,
                 'n_fixed_samples': k_b, 'n_collapse_contexts': n_cc,
                 'n_cond': n_c, 'k_cond': k_c},
      'sweep': results,
  }
  out = args.out or os.path.join(args.ckpt_dir, 'flow_step_sweep.json')
  with open(out, 'w') as f:
    json.dump(payload, f, indent=2)
  print(f'\nresults -> {out}')

  csv_path = os.path.splitext(out)[0] + '.csv'
  cols = ['flow_steps', 'coord_boundary_fraction', 'row_boundary_fraction',
          'clip_update_fraction', 'gen_mean_std', 'gen_over_real_std',
          'correct_mean_min_l2', 'shuffled_mean_min_l2',
          'conditional_win_rate', 'collapsed_contexts',
          'per_dim_collapse_fraction', 'median_ctx_mean_std',
          'mean_pairwise_l2', 'overshoot_median', 'overshoot_p90',
          'nan_inf_count', 'sample_seconds_block_A']
  with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in results:
      w.writerow(r)
  print(f'csv     -> {csv_path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
