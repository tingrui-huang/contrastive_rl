"""Stage 2 diagnostics for the behavior flow model.

A decreasing flow-matching loss is not evidence that ``mu_omega(a | c)``
resembles the behavior policy, so this script asks four separate questions on
HELD-OUT validation data only (the Stage-1 episode-level split; no environment
is constructed and no rollout is performed).

  A. BOUNDS +
     GLOBAL      -- per-dim min/max/mean/std of real validation actions vs
                    generated actions, the fraction of generated coordinates
                    outside the [-1, 1] action box, and how often the per-step
                    clip actually fires. The old UNCLIPPED path is sampled from
                    the SAME z and reported alongside, so the effect of the
                    correction stays measurable rather than hidden.
  B. FIXED-CONTEXT STOCHASTICITY
                 -- for each of several fixed validation contexts, draw K
                    samples with different noise z and report the spread, so a
                    collapsed (deterministic) sampler is visible.
  C. CONDITIONAL -- does the model USE the context? For a held-out (c_i, a_i),
                    compare the nearest-sample L2 distance from a_i to K
                    samples drawn under the CORRECT context against samples
                    drawn under a SHUFFLED context c_j (j != i). Two shuffles
                    are run: the full context, and the goal columns only (the
                    latter isolates g_cmd, which the audit flagged as read by
                    the teacher only after handoff).
                    This is a DIAGNOSTIC, never a training objective.
  D. REPRODUCIBILITY
                 -- same seed reproduces the same numbers; no NaN/Inf; the
                    train/val split is still disjoint; no eval data was used
                    for training.

Run:

  python -m propensity.eval_flow \
      --ckpt-dir artifacts/propensity_flow/rockfall_v2_p30_h800_smoke \
      --checkpoint best
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


def _fmt(v, prec=4, width=9):
  return '[' + ' '.join(f'{float(x):{width}.{prec}f}' for x in v) + ']'


def _stats(a):
  a = np.asarray(a)
  return {'min': a.min(axis=0).tolist(), 'max': a.max(axis=0).tolist(),
          'mean': a.mean(axis=0).tolist(), 'std': a.std(axis=0).tolist()}


def _print_stats(name, st):
  print(f'  {name} min  {_fmt(st["min"])}')
  print(f'  {name} max  {_fmt(st["max"])}')
  print(f'  {name} mean {_fmt(st["mean"])}')
  print(f'  {name} std  {_fmt(st["std"])}')


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 2 behavior-flow diagnostics (held-out data only).')
  p.add_argument('--ckpt-dir', required=True)
  p.add_argument('--checkpoint', default='best', choices=('best', 'latest'))
  p.add_argument('--dataset', default=None,
                 help='override the dataset path recorded in config.json')
  p.add_argument('--flow-steps', type=int, default=None,
                 help='Euler steps (default: the value recorded at training).')
  p.add_argument('--seed', type=int, default=0, help='diagnostic sampling seed')
  p.add_argument('--n-global', type=int, default=8192,
                 help='validation rows for diagnostic A')
  p.add_argument('--n-fixed-contexts', type=int, default=8,
                 help='contexts for diagnostic B')
  p.add_argument('--n-fixed-samples', type=int, default=128,
                 help='samples per fixed context for diagnostic B (>= 100)')
  p.add_argument('--n-collapse-contexts', type=int, default=256,
                 help='contexts for the per-dimension collapse check in B')
  p.add_argument('--n-cond', type=int, default=1024,
                 help='held-out tuples for diagnostic C')
  p.add_argument('--k-cond', type=int, default=32,
                 help='samples per context for diagnostic C')
  p.add_argument('--json', default=None, help='write the report here')
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  meta = ckpt_mod.load_metadata(args.ckpt_dir)
  dataset = args.dataset or meta['dataset_path']
  flow_steps = (args.flow_steps if args.flow_steps is not None
                else meta['flow_steps_default'])

  # ---- rebuild the SAME split and model ----------------------------------- #
  ds = BehaviorDataset(dataset, val_frac=meta['val_frac'],
                       seed=meta['split_seed'],
                       state_mode=('obs' if meta['context_mode'] == 'context'
                                   else 'state'),
                       split_level=meta['split_level'])
  if ds.fingerprint['sha256'] != meta['dataset_sha256']:
    print('ABORT: dataset sha256 does not match the training record.')
    return 1
  action_dim, context_dim = ds.action_dim, ds.context_dim

  fc = meta['flow_config']
  fcfg = flow_mod.FlowConfig(context_dim=fc['context_dim'],
                             action_dim=fc['action_dim'],
                             hidden_sizes=tuple(fc['hidden_sizes']),
                             time_features=fc['time_features'],
                             time_max_freq=fc['time_max_freq'],
                             use_layer_norm=fc['use_layer_norm'])
  net = flow_mod.make_flow_network(fcfg)
  ckpt_path = os.path.join(args.ckpt_dir, f'{args.checkpoint}.pkl')
  step, params, _ = ckpt_mod.load_checkpoint(ckpt_path)

  # The BehaviorFlow sampler: per-step clipped Euler integration (CFQL-aligned).
  @jax.jit
  def sample(context, key):
    return flow_mod.sample_actions(net.apply, params, context, key, action_dim,
                                   num_steps=flow_steps, clip=True)

  # DIAGNOSTIC ONLY: the old unclipped path, kept so the correction stays
  # measurable in the before/after comparison. Never used downstream.
  @jax.jit
  def sample_raw(context, key):
    return flow_mod.sample_actions_raw(net.apply, params, context, key,
                                       action_dim, num_steps=flow_steps)

  @jax.jit
  def sample_stats(context, key):
    return flow_mod.sample_actions_with_clip_stats(
        net.apply, params, context, key, action_dim, num_steps=flow_steps)

  def sample_multi(context, key, k):
    """K samples per row (clipped sampler), jitted per distinct K."""
    fn = jax.jit(lambda c, kk: flow_mod.sample_actions_multi(
        net.apply, params, c, kk, action_dim, k, num_steps=flow_steps,
        clip=True))
    return np.asarray(fn(jnp.asarray(context), key))

  # Sampler provenance is recorded HERE rather than rewritten into the training
  # run's config.json -- that file records what the run actually recorded at the
  # time, and is left untouched. See sampler.json written at the end.
  sampler_record = {
      'sampler': 'euler_per_step_clipped',
      'flow_steps': int(flow_steps),
      'flow_steps_provenance': flow_mod.FLOW_STEPS_PROVENANCE,
      'per_step_action_clip': list(flow_mod.ACTION_BOX),
      'clip_provenance': flow_mod.CLIP_PROVENANCE,
      'clip_applies_to': 'action sampling / numerical integration only',
      'training_objective_changed': False,
      'network_output_squashed': False,
      'offline_actions_modified': False,
      'training_config_provenance_at_train_time':
          meta.get('flow_steps_provenance'),
  }

  print('=' * 74)
  print('propensity.eval_flow -- Stage 2 behavior flow diagnostics')
  print('=' * 74)
  print(f'checkpoint     {ckpt_path}  (step {step})  [REUSED, not retrained]')
  print(f'dataset        {dataset}')
  print(f'context mode   {meta["context_mode"]}  '
        f'(s={meta["state_dim"]}, g_cmd={meta["goal_dim"]}, c={context_dim})')
  print(f'sampler        euler, per-step clip to '
        f'{list(flow_mod.ACTION_BOX)}  ({flow_mod.CLIP_PROVENANCE})')
  print(f'flow steps     {flow_steps}  ({flow_mod.FLOW_STEPS_PROVENANCE})')
  print(f'val rows       {ds.n_val}  (train {ds.n_train})')
  print(f'val flow-MSE   best={meta.get("best_val_flow_mse")}  '
        f'final={meta.get("final_val_flow_mse")}')
  report = {'checkpoint': ckpt_path, 'step': int(step), 'dataset': dataset,
            'flow_steps': flow_steps, 'context_mode': meta['context_mode'],
            'sampler': sampler_record,
            'val_flow_mse_best': meta.get('best_val_flow_mse'),
            'val_flow_mse_final': meta.get('final_val_flow_mse')}

  rng = np.random.default_rng(args.seed)
  key = jax.random.PRNGKey(args.seed)
  val_idx = ds._indices('val')                                # noqa: SLF001

  # ======================================================================== #
  # A. GLOBAL held-out action statistics
  # ======================================================================== #
  n_a = min(args.n_global, val_idx.size)
  rows = rng.choice(val_idx, size=n_a, replace=False)
  ctx_a = jnp.asarray(ds._state[rows])                        # noqa: SLF001
  real_a = np.asarray(ds._action[rows])                       # noqa: SLF001
  key, sub = jax.random.split(key)
  # Same PRNG key for both samplers => the SAME z, so the two paths differ only
  # by the per-step clip.
  gen_a = np.asarray(sample(ctx_a, sub))                        # corrected
  gen_raw = np.asarray(sample_raw(ctx_a, sub))                  # old behavior
  _, n_clip, n_upd = sample_stats(ctx_a, sub)
  n_clip, n_upd = int(n_clip), int(n_upd)

  def _oob(a):
    return (flow_mod.out_of_box_fraction(a),
            float(np.mean(np.any((a < -1.0) | (a > 1.0), axis=1))))

  st_real, st_gen, st_raw = _stats(real_a), _stats(gen_a), _stats(gen_raw)
  oob, oob_rows = _oob(gen_a)
  oob_raw, oob_raw_rows = _oob(gen_raw)
  oob_real, _ = _oob(real_a)
  at_bound = float(np.mean(np.abs(np.abs(gen_a) - 1.0) < 1e-6))
  at_bound_rows = float(np.mean(
      np.any(np.abs(np.abs(gen_a) - 1.0) < 1e-6, axis=1)))
  at_bound_real = float(np.mean(np.abs(np.abs(real_a) - 1.0) < 1e-6))

  print()
  print(f'A. ACTION BOUNDS + GLOBAL HELD-OUT STATISTICS  (n={n_a})')
  _print_stats('real', st_real)
  print()
  _print_stats('gen ', st_gen)
  print()
  print('  -- bounds (corrected per-step-clipped sampler) --')
  print(f'  generated coords outside [-1, 1]      : {oob:.6f}  '
        f'({oob * gen_a.size:.0f} / {gen_a.size})')
  print(f'  generated ROWS with any coord outside : {oob_rows:.6f}')
  print(f'  real validation coords outside box    : {oob_real:.6f}')
  print(f'  generated finite                      : '
        f'{bool(np.isfinite(gen_a).all())}')
  print('  -- how active is the clip? --')
  print(f'  Euler coord-updates that triggered clip: {n_clip / n_upd:.6f}  '
        f'({n_clip} / {n_upd})')
  print(f'  final coords exactly at +/-1          : {at_bound:.6f}')
  print(f'  final ROWS with any coord at +/-1     : {at_bound_rows:.6f}')
  print(f'  (real validation coords at +/-1       : {at_bound_real:.6f})')
  print('  -- old UNCLIPPED sampler, same z, for comparison --')
  print(f'  raw coords outside [-1, 1]            : {oob_raw:.6f}')
  print(f'  raw ROWS with any coord outside       : {oob_raw_rows:.6f}')
  print(f'  mean |raw - corrected|                : '
        f'{float(np.abs(gen_raw - gen_a).mean()):.6f}')

  # Per-dim boundary mass. Clipping cannot create out-of-box samples, but it
  # PILES the former out-of-box mass onto the boundary. If the generated
  # boundary mass far exceeds the real one, |a|==1 becomes a signature a later
  # discriminator could exploit -- the same class of shortcut the clip removed.
  bm_real = (np.abs(np.abs(real_a) - 1.0) < 1e-6).mean(axis=0)
  bm_gen = (np.abs(np.abs(gen_a) - 1.0) < 1e-6).mean(axis=0)
  bm_ratio = bm_gen / np.maximum(bm_real, 1e-9)
  # How far past the box did the raw path go where the clip fired? Small
  # excursions => near-boundary density the model gets slightly wrong; large
  # ones => the flow genuinely aims outside the support.
  at_b = np.abs(np.abs(gen_a) - 1.0) < 1e-6
  exc = (np.abs(gen_raw[at_b]) - 1.0) if at_b.any() else np.zeros(1)
  print('  -- boundary mass (per dim) --')
  print(f'  real |a|==1  {_fmt(bm_real)}')
  print(f'  gen  |a|==1  {_fmt(bm_gen)}')
  print(f'  ratio g/r    {_fmt(bm_ratio, prec=2)}')
  print(f'  raw excursion where clipped: median {np.median(exc):.4f}  '
        f'p90 {np.percentile(exc, 90):.4f}  max {exc.max():.4f}')
  print(f'  clipped coords with excursion < 0.05  : '
        f'{float((exc < 0.05).mean()):.4f}')
  boundary_extra = {
      'boundary_mass_real_per_dim': bm_real.tolist(),
      'boundary_mass_gen_per_dim': bm_gen.tolist(),
      'boundary_mass_ratio_per_dim': bm_ratio.tolist(),
      'boundary_mass_ratio_overall': float(at_bound / max(at_bound_real, 1e-9)),
      'raw_excursion_median': float(np.median(exc)),
      'raw_excursion_p90': float(np.percentile(exc, 90)),
      'raw_excursion_max': float(exc.max()),
      'clipped_coords_excursion_lt_0p05': float((exc < 0.05).mean()),
  }
  report['A_global'] = {
      'n': int(n_a), 'real': st_real,
      'generated_clipped': st_gen, 'generated_raw_unclipped': st_raw,
      'gen_out_of_box_entry_fraction': oob,
      'gen_out_of_box_row_fraction': oob_rows,
      'real_out_of_box_entry_fraction': oob_real,
      'raw_out_of_box_entry_fraction': oob_raw,
      'raw_out_of_box_row_fraction': oob_raw_rows,
      'clip_trigger_fraction_of_euler_updates': n_clip / n_upd,
      'clip_trigger_count': n_clip, 'euler_coord_updates': n_upd,
      'final_coords_at_boundary_fraction': at_bound,
      'final_rows_with_boundary_coord_fraction': at_bound_rows,
      'real_coords_at_boundary_fraction': at_bound_real,
      'mean_abs_raw_minus_corrected': float(np.abs(gen_raw - gen_a).mean()),
      'gen_finite': bool(np.isfinite(gen_a).all()),
      'sampler': 'euler_per_step_clipped',
      **boundary_extra,
  }

  # ======================================================================== #
  # B. FIXED-CONTEXT STOCHASTICITY
  # ======================================================================== #
  n_ctx, n_s = args.n_fixed_contexts, args.n_fixed_samples
  rows_b = rng.choice(val_idx, size=n_ctx, replace=False)
  ctx_b = np.asarray(ds._state[rows_b])                       # noqa: SLF001
  key, sub = jax.random.split(key)
  samples_b = sample_multi(ctx_b, sub, n_s)                   # [n_ctx, n_s, A]
  per_ctx_std = samples_b.std(axis=1)                         # [n_ctx, A]
  # Pairwise spread within a context, as a collapse check independent of std.
  spread = np.array([
      float(np.mean(np.linalg.norm(
          samples_b[i][rng.integers(0, n_s, 512)]
          - samples_b[i][rng.integers(0, n_s, 512)], axis=-1)))
      for i in range(n_ctx)])
  collapsed = per_ctx_std.mean(axis=1) < 1e-6
  print()
  print(f'B. FIXED-CONTEXT STOCHASTICITY  ({n_ctx} contexts x {n_s} samples,'
        f' per-step-clipped sampler)')
  print(f'  {"ctx":>4}  {"mean per-dim std":>16}  {"min std":>9}  '
        f'{"max std":>9}  {"mean pair L2":>12}  finite')
  for i in range(n_ctx):
    print(f'  {i:>4}  {per_ctx_std[i].mean():>16.6f}  '
          f'{per_ctx_std[i].min():>9.6f}  {per_ctx_std[i].max():>9.6f}  '
          f'{spread[i]:>12.6f}  {bool(np.isfinite(samples_b[i]).all())}')
  print(f'  collapsed contexts (mean std < 1e-6): {int(collapsed.sum())}'
        f' / {n_ctx}')
  print(f'  all samples finite: {bool(np.isfinite(samples_b).all())}')

  # Per-DIMENSION collapse over a much wider context sample. A whole context
  # rarely collapses, but per-step clipping can pin an individual coordinate to
  # the boundary for a given context -- that is what this catches.
  n_cc = min(args.n_collapse_contexts, val_idx.size)
  rows_cc = rng.choice(val_idx, size=n_cc, replace=False)
  s_cc = sample_multi(np.asarray(ds._state[rows_cc]),               # noqa: SLF001
                      jax.random.PRNGKey(args.seed + 11), n_s)
  sd_cc = s_cc.std(axis=1)                                          # [n_cc, A]
  dead = sd_cc < 1e-6
  pinned = np.abs(np.abs(s_cc.mean(axis=1)) - 1.0) < 1e-6
  frac_pinned = float(pinned[dead].mean()) if dead.any() else 0.0
  print(f'  -- per-dimension collapse over {n_cc} contexts x {n_s} samples --')
  print(f'  (context, dim) pairs with std < 1e-6  : '
        f'{float(dead.mean()):.4f}  ({int(dead.sum())} / {dead.size})')
  print(f'    of those, pinned exactly at +/-1    : {frac_pinned:.4f}')
  print(f'  contexts with ALL dims collapsed      : '
        f'{int(dead.all(axis=1).sum())} / {n_cc}')
  print(f'  contexts with >=1 dim collapsed       : '
        f'{int(dead.any(axis=1).sum())} / {n_cc}')
  print(f'  median per-context mean std           : '
        f'{float(np.median(sd_cc.mean(axis=1))):.4f}')
  report['B_fixed_context'] = {
      'n_contexts': int(n_ctx), 'n_samples': int(n_s),
      'per_context_mean_std': per_ctx_std.mean(axis=1).tolist(),
      'per_context_min_std': per_ctx_std.min(axis=1).tolist(),
      'per_context_max_std': per_ctx_std.max(axis=1).tolist(),
      'per_context_mean_pairwise_l2': spread.tolist(),
      'n_collapsed': int(collapsed.sum()),
      'all_finite': bool(np.isfinite(samples_b).all()),
      'collapse_n_contexts': int(n_cc),
      'per_dim_collapse_fraction': float(dead.mean()),
      'per_dim_collapse_count': int(dead.sum()),
      'per_dim_collapse_pinned_at_boundary_fraction': frac_pinned,
      'contexts_all_dims_collapsed': int(dead.all(axis=1).sum()),
      'contexts_any_dim_collapsed': int(dead.any(axis=1).sum()),
      'median_per_context_mean_std': float(np.median(sd_cc.mean(axis=1))),
  }

  # ======================================================================== #
  # C. CONDITIONAL BEHAVIOR SANITY CHECK
  # ======================================================================== #
  # Metric: nearest-sample L2, d(a_i, C) = min_k ||a_i - A_k||_2.
  n_c, k_c = min(args.n_cond, val_idx.size), args.k_cond
  rows_c = rng.choice(val_idx, size=n_c, replace=False)
  ctx_c = np.asarray(ds._state[rows_c])                       # noqa: SLF001
  act_c = np.asarray(ds._action[rows_c])                      # noqa: SLF001

  # derangement over the held-out rows (j != i for every i)
  perm = (np.arange(n_c) + 1 + rng.integers(0, n_c - 1)) % n_c
  assert np.all(perm != np.arange(n_c))
  ctx_shuffled = ctx_c[perm]

  # goal-only shuffle: keep s_i, swap in g_cmd from row j. Isolates g_cmd.
  ctx_goal_shuffled = None
  if meta['context_mode'] == 'context':
    s_dim = meta['state_dim']
    ctx_goal_shuffled = ctx_c.copy()
    ctx_goal_shuffled[:, s_dim:] = ctx_c[perm][:, s_dim:]

  def nearest_dist(context):
    key_local = jax.random.PRNGKey(args.seed + 7)
    s = sample_multi(context, key_local, k_c)                 # [n, k, A]
    d = np.linalg.norm(s - act_c[:, None, :], axis=-1)        # [n, k]
    return d.min(axis=1), d.mean(axis=1)

  d_ok, dm_ok = nearest_dist(ctx_c)
  d_bad, dm_bad = nearest_dist(ctx_shuffled)
  print()
  print(f'C. CONDITIONAL SANITY  (n={n_c} held-out tuples, K={k_c} samples,'
        f' metric = nearest-sample L2 to a_i,\n'
        f'   per-step-clipped sampler)')
  print(f'  {"variant":<26} {"mean min-L2":>12}  {"median":>9}  '
        f'{"mean of mean-L2":>15}')
  print(f'  {"correct context c_i":<26} {d_ok.mean():>12.4f}  '
        f'{np.median(d_ok):>9.4f}  {dm_ok.mean():>15.4f}')
  print(f'  {"shuffled context c_j":<26} {d_bad.mean():>12.4f}  '
        f'{np.median(d_bad):>9.4f}  {dm_bad.mean():>15.4f}')
  win = float(np.mean(d_ok < d_bad))
  print(f'  correct < shuffled (per-example win rate): {win:.4f}')
  print(f'  gap (shuffled - correct) mean min-L2     : '
        f'{d_bad.mean() - d_ok.mean():+.4f}')
  cond = {'n': int(n_c), 'k': int(k_c), 'metric': 'nearest_sample_l2',
          'correct_mean_min_l2': float(d_ok.mean()),
          'correct_median_min_l2': float(np.median(d_ok)),
          'shuffled_mean_min_l2': float(d_bad.mean()),
          'shuffled_median_min_l2': float(np.median(d_bad)),
          'correct_mean_of_mean_l2': float(dm_ok.mean()),
          'shuffled_mean_of_mean_l2': float(dm_bad.mean()),
          'win_rate_correct_lt_shuffled': win,
          'gap_mean_min_l2': float(d_bad.mean() - d_ok.mean()),
          'expected': 'correct < shuffled'}

  if ctx_goal_shuffled is not None:
    d_g, dm_g = nearest_dist(ctx_goal_shuffled)
    win_g = float(np.mean(d_ok < d_g))
    print(f'  {"goal-only shuffled":<26} {d_g.mean():>12.4f}  '
          f'{np.median(d_g):>9.4f}  {dm_g.mean():>15.4f}')
    print(f'  correct < goal-shuffled (win rate)       : {win_g:.4f}')
    print('  [note] the audit found the scripted pre-handoff walker reads only '
          'obs[:29],\n         so a small goal-only effect is EXPECTED and is '
          'not a model defect.')
    cond['goal_shuffled_mean_min_l2'] = float(d_g.mean())
    cond['goal_shuffled_median_min_l2'] = float(np.median(d_g))
    cond['win_rate_correct_lt_goal_shuffled'] = win_g
    cond['gap_goal_only'] = float(d_g.mean() - d_ok.mean())

  verdict_c = 'PASS' if d_ok.mean() < d_bad.mean() else 'FAIL'
  cond['verdict'] = verdict_c
  print(f'  VERDICT (full-context shuffle): {verdict_c}')
  report['C_conditional'] = cond

  # ======================================================================== #
  # D. REPRODUCIBILITY / OFFLINE INTEGRITY
  # ======================================================================== #
  key_r = jax.random.PRNGKey(args.seed)
  _, sub_r = jax.random.split(key_r)
  gen_repeat = np.asarray(sample(ctx_a, sub_r))
  same = bool(np.array_equal(gen_a, gen_repeat))
  ok_split, gates, details = ds.check()
  finite = bool(np.isfinite(gen_a).all() and np.isfinite(samples_b).all()
                and np.isfinite(d_ok).all() and np.isfinite(d_bad).all())
  print()
  print('D. REPRODUCIBILITY / OFFLINE INTEGRITY')
  print(f'  same seed -> identical samples          : {same}')
  print(f'  all diagnostics finite (no NaN/Inf)     : {finite}')
  print(f'  Stage-1 gates still pass                : {ok_split}')
  print(f'  train/val index overlap                 : '
        f'{details["index_overlap"]}')
  print(f'  train/val episode overlap               : '
        f'{details["episode_overlap"]}')
  print(f'  diagnostics used validation rows only   : True '
        f'(rows drawn from ds._indices("val"))')
  print(f'  environment constructed / rollouts run  : False')
  print(f'  dataset sha256 matches training record  : True')
  report['D_reproducibility'] = {
      'same_seed_identical_samples': same, 'all_finite': finite,
      'stage1_gates_pass': ok_split,
      'index_overlap': details['index_overlap'],
      'episode_overlap': details['episode_overlap'],
      'diagnostics_on_validation_only': True,
      'env_constructed': False,
      'dataset_sha256_matches': True,
  }

  overall = (verdict_c == 'PASS' and finite and ok_split and same
             and int(collapsed.sum()) == 0 and oob == 0.0 and oob_rows == 0.0)
  report['verdict'] = 'PASS' if overall else 'REVIEW'
  print()
  print(f'OVERALL: {report["verdict"]}')

  out = args.json or os.path.join(args.ckpt_dir, 'diagnostics.json')
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with open(out, 'w') as f:
    json.dump(report, f, indent=2)
  print(f'report -> {out}')

  # Sampler provenance sidecar. Written SEPARATELY from the training run's
  # config.json so the historical record of what that run actually used is not
  # overwritten.
  sampler_path = os.path.join(args.ckpt_dir, 'sampler.json')
  with open(sampler_path, 'w') as f:
    json.dump(sampler_record, f, indent=2)
  print(f'sampler record -> {sampler_path}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
