"""Stage 3B: finalize + characterize the canonical agreement score D(s, g_cmd, a).

Runs everything needed to declare the propensity-side approximation
integration-ready:

  1. multi-seed held-out metrics for the ACTIVE formulation only
     (state_cmdgoal_action); the state_action / action / context models are
     reported strictly as diagnostics and controls;
  2. pairwise seed stability of the score (Pearson/Spearman, tail overlap) --
     the point is to CHARACTERIZE estimator variance, not to demand agreement;
  3. the retained Stage-3A sanity checks: context-only leakage control,
     action-only baseline, correct- vs shuffled-context;
  4. a DIAGNOSTIC-ONLY multi-seed averaged score, to see whether averaging
     buys stability. This is not proposed as the canonical algorithm;
  5. an end-to-end exercise of the public API in propensity/agreement.py,
     including jit and vmap, so the downstream contract is verified here.

The perturbation probe is reported as descriptive information only. It is NOT
an acceptance criterion: D is a behavior-vs-target agreement classifier, not an
explicitly trained distance-to-manifold estimator.

Offline throughout: frozen dataset, frozen CRL actor, no environment, no
rollouts, no BehaviorFlow.

Run:

  python -m propensity.eval_agreement \
      --seed-run 0=artifacts/support_discriminator/D_state_cmdgoal_action \
      --seed-run 1=artifacts/support_discriminator/D_state_cmdgoal_action_seed1 \
      --seed-run 2=artifacts/support_discriminator/D_state_cmdgoal_action_seed2 \
      --baseline A=artifacts/support_discriminator/D_state_action \
      --baseline action=artifacts/support_discriminator/D_action \
      --baseline context=artifacts/support_discriminator/D_context
"""
import argparse
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from propensity import agreement as agr
from propensity import crl_policy_adapter as crl_adapter
from propensity import discriminator as disc_mod
from propensity.dataset import BehaviorDataset
from propensity.eval_discriminator import (_avg_precision, _bce, _quantiles,
                                           _spearman, load_model)
from propensity.train_discriminator import make_splits

PERTURB_DELTAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50)


def _roc_auc(scores, labels):
  scores, labels = np.asarray(scores, float), np.asarray(labels, int)
  order = np.argsort(scores, kind='mergesort')
  ranks = np.empty(len(scores), float)
  s = scores[order]
  i = 0
  while i < len(s):
    j = i
    while j + 1 < len(s) and s[j + 1] == s[i]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
    i = j + 1
  n_pos = int(labels.sum())
  n_neg = len(labels) - n_pos
  if n_pos == 0 or n_neg == 0:
    return float('nan')
  return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
               / (n_pos * n_neg))


def _tail_overlap(x, y, q, low):
  tx = (x <= np.percentile(x, q)) if low else (x >= np.percentile(x, 100 - q))
  ty = (y <= np.percentile(y, q)) if low else (y >= np.percentile(y, 100 - q))
  return float((tx & ty).sum() / max((tx | ty).sum(), 1))


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 3B: finalize the agreement score D(s, g_cmd, a).')
  p.add_argument('--seed-run', action='append', required=True,
                 help='seed=run_dir for the ACTIVE formulation (repeatable)')
  p.add_argument('--baseline', action='append', default=[],
                 help='label=run_dir diagnostics/controls (repeatable)')
  p.add_argument('--n-test', type=int, default=8192)
  p.add_argument('--seed', type=int, default=0, help='diagnostic bank seed')
  p.add_argument('--out',
                 default='artifacts/support_discriminator/stage3b_agreement.json')
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  seed_runs = [(lbl, path) for lbl, _, path in
               (s.partition('=') for s in args.seed_run)]
  baselines = [(lbl, path) for lbl, _, path in
               (s.partition('=') for s in args.baseline)]

  # ---- canonical models via the PUBLIC API -------------------------------- #
  models = {lbl: agr.load_agreement_model(path) for lbl, path in seed_runs}
  ref_model = models[seed_runs[0][0]]
  spec, meta = ref_model.spec, ref_model.meta
  logit_fn, score_fn = agr.make_agreement_fns(spec)     # jitted

  ds = BehaviorDataset(meta['dataset_path'], val_frac=meta['val_frac'],
                       seed=meta['stage1_split_seed'], state_mode='obs',
                       split_level='episode')
  if ds.fingerprint['sha256'] != meta['dataset_sha256']:
    print('ABORT: dataset sha256 mismatch.')
    return 1
  S, A = spec.state_dim, spec.action_dim
  rows, eps = make_splits(ds, len(meta['episodes']['dev']),
                          meta['dev_split_seed'])
  lengths = ds._lengths                                        # noqa: SLF001
  ep_of_row = ds._episode_of_row                               # noqa: SLF001
  starts = np.concatenate([[0], np.cumsum(lengths - 1)])
  t_of_row = np.arange(ds.n_transitions) - starts[ep_of_row]

  # ---- one cached held-out bank, shared by every model -------------------- #
  rng = np.random.default_rng(args.seed)
  test_rows = rows['test']
  n = min(args.n_test, len(test_rows))
  sel = test_rows[rng.choice(len(test_rows), size=n, replace=False)]
  raw = ds._state                                              # noqa: SLF001
  s = np.asarray(raw[sel][:, :S])
  g_cmd_full = np.asarray(raw[sel][:, S:])          # FULL stored g_cmd [n, 29]
  a_real = np.asarray(ds._action[sel])                         # noqa: SLF001
  groups = ep_of_row[sel]

  # g_query: hindsight future goal. Used ONLY to ask the frozen CRL actor for a
  # proposed action. It is never an input to D.
  goal_src = crl_adapter.load_goal_source(meta['dataset_path'], S)
  j_future = crl_adapter.sample_future_goal_index(
      lengths, ep_of_row[sel], t_of_row[sel],
      np.random.default_rng(args.seed + 11))
  g_query = np.asarray(goal_src[ep_of_row[sel], j_future])
  _, crl_apply, crl_info = crl_adapter.load_frozen_crl_actor(
      meta['crl']['checkpoint'], S, ds._goal_dim, A)           # noqa: SLF001
  a_crl, a_crl_mode = crl_adapter.crl_actions(
      crl_apply, s, g_query,
      np.random.default_rng(args.seed + 12).standard_normal((n, A)))

  perm = (np.arange(n) + 1 + rng.integers(0, n - 1)) % n
  assert np.all(perm != np.arange(n))

  print('=' * 82)
  print('propensity.eval_agreement -- Stage 3B: canonical agreement score')
  print('=' * 82)
  print(f'ACTIVE formulation   D(s, g_cmd, a)   [input_spec '
        f'{spec.input_spec}]')
  print(f'  state_dim {spec.state_dim} | g_cmd full {spec.g_cmd_dim_full} '
        f'-> live indices {list(spec.g_cmd_indices)} ({spec.g_cmd_dim}) | '
        f'action_dim {spec.action_dim} | input_dim {spec.input_dim}')
  print(f'dataset  {os.path.basename(meta["dataset_path"])} '
        f'sha {meta["dataset_sha256"][:16]}...')
  print(f'CRL      step {crl_info["step"]} FROZEN | g_query = '
        f'{meta["query_goal_semantics"]["probability_rule"]}')
  print(f'bank     {n} contexts / {len(np.unique(groups))} held-out episodes '
        f'(test = Stage-1 val)')
  print('D IS A SOFT AGREEMENT SURROGATE -- NOT A CONTINUOUS PROPENSITY')

  report = {'active_formulation': 'D(s, g_cmd, a)',
            'input_spec': spec.input_spec, 'spec': spec.asdict(),
            'n_test_contexts': int(n),
            'test_episodes': [int(x) for x in np.unique(groups)],
            'dataset_sha256': meta['dataset_sha256'],
            'crl_checkpoint': meta['crl']['checkpoint'],
            'crl_step': crl_info['step'],
            'crl_dataset_sha256': meta.get('crl_dataset_sha256'),
            'query_goal_semantics': meta['query_goal_semantics'],
            'g_cmd_is_input_to_D': True,
            'g_query_is_input_to_D': False,
            'behaviorflow_used': False,
            'uniform_reference_used': False,
            'neighborhood_bandwidth_used': False,
            'environment_rollouts_used': False,
            'is_calibrated_propensity': False,
            'seeds': {}, 'baselines': {}}

  # ======================================================================== #
  # 1. Multi-seed held-out metrics (ACTIVE formulation)
  # ======================================================================== #
  print()
  print('1. MULTI-SEED HELD-OUT METRICS -- ACTIVE D(s, g_cmd, a)')
  hdr = (f'{"seed":>5} {"ROC-AUC":>8} {"PR-AUC":>8} {"BCE":>8} '
         f'{"real mean":>10} {"real sd":>8} {"CRL mean":>9} {"CRL sd":>8}')
  print(hdr)
  print('-' * len(hdr))
  scores_real, scores_crl = {}, {}
  for lbl, _ in seed_runs:
    m = models[lbl]
    sr = np.asarray(score_fn(m.params, s, g_cmd_full, a_real))
    sc = np.asarray(score_fn(m.params, s, g_cmd_full, a_crl))
    scores_real[lbl], scores_crl[lbl] = sr, sc
    lg = np.concatenate([np.asarray(logit_fn(m.params, s, g_cmd_full, a_real)),
                         np.asarray(logit_fn(m.params, s, g_cmd_full, a_crl))])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    rec = {'roc_auc': _roc_auc(lg, y), 'pr_auc': _avg_precision(lg, y),
           'bce': _bce(lg, y),
           'score_real_mean': float(sr.mean()), 'score_real_std': float(sr.std()),
           'score_crl_mean': float(sc.mean()), 'score_crl_std': float(sc.std()),
           'score_real_quantiles': _quantiles(sr),
           'score_crl_quantiles': _quantiles(sc),
           'training_seed': m.meta['seed'], 'dev_auc': m.meta['best_dev_auc']}
    report['seeds'][lbl] = rec
    print(f'{lbl:>5} {rec["roc_auc"]:>8.4f} {rec["pr_auc"]:>8.4f} '
          f'{rec["bce"]:>8.4f} {rec["score_real_mean"]:>10.4f} '
          f'{rec["score_real_std"]:>8.4f} {rec["score_crl_mean"]:>9.4f} '
          f'{rec["score_crl_std"]:>8.4f}')

  # ======================================================================== #
  # 2. Pairwise seed stability (on TARGET actions -- the downstream use case)
  # ======================================================================== #
  print()
  print('2. PAIRWISE SEED STABILITY of the score on CRL target actions')
  print(f'{"pair":>10} {"pearson":>9} {"spearman":>9} {"mean|d|":>9} '
        f'{"low10% ov":>10} {"high10% ov":>11}')
  pairs = {}
  for a_lbl, b_lbl in itertools.combinations(scores_crl, 2):
    x, y_ = scores_crl[a_lbl], scores_crl[b_lbl]
    rec = {'pearson': float(np.corrcoef(x, y_)[0, 1]),
           'spearman': _spearman(x, y_),
           'mean_abs_difference': float(np.abs(x - y_).mean()),
           'lowest_decile_overlap': _tail_overlap(x, y_, 10, True),
           'highest_decile_overlap': _tail_overlap(x, y_, 10, False)}
    pairs[f'{a_lbl}-{b_lbl}'] = rec
    print(f'{a_lbl + "-" + b_lbl:>10} {rec["pearson"]:>9.4f} '
          f'{rec["spearman"]:>9.4f} {rec["mean_abs_difference"]:>9.4f} '
          f'{rec["lowest_decile_overlap"]:>10.4f} '
          f'{rec["highest_decile_overlap"]:>11.4f}')
  report['pairwise_seed_stability'] = pairs

  # ======================================================================== #
  # 3. Controls and sanity checks
  # ======================================================================== #
  print()
  print('3. CONTROLS AND SANITY CHECKS')
  base = {}
  for lbl, path in baselines:
    bmeta, bfn = load_model(path)
    gsl = np.asarray(g_cmd_full[:, bmeta['cmdgoal_indices']])
    lp = np.asarray(bfn(disc_mod.assemble_inputs(bmeta['input_spec'], s, gsl,
                                                 a_real)))
    ln = np.asarray(bfn(disc_mod.assemble_inputs(bmeta['input_spec'], s, gsl,
                                                 a_crl)))
    lg = np.concatenate([lp, ln])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    base[lbl] = {'input_spec': bmeta['input_spec'], 'roc_auc': _roc_auc(lg, y),
                 'bce': _bce(lg, y), 'role': 'DIAGNOSTIC ONLY -- not used by '
                                             'the causal learner'}
    print(f'   {lbl:<8} ({bmeta["input_spec"]:<20}) ROC-AUC '
          f'{base[lbl]["roc_auc"]:.4f}  BCE {base[lbl]["bce"]:.4f}')
  report['baselines'] = base

  # correct vs shuffled context, per seed, on REAL behavior actions
  print()
  print('   correct-context vs shuffled-context (REAL behavior actions)')
  ctx = {}
  for lbl, _ in seed_runs:
    m = models[lbl]
    ok = scores_real[lbl]
    bad = np.asarray(score_fn(m.params, s[perm], g_cmd_full[perm], a_real))
    ctx[lbl] = {'correct_mean': float(ok.mean()),
                'shuffled_mean': float(bad.mean()),
                'paired_difference': float((ok - bad).mean()),
                'fraction_correct_gt_shuffled': float((ok > bad).mean())}
    print(f'   seed {lbl}: correct {ok.mean():.4f} vs shuffled '
          f'{bad.mean():.4f}  (diff {(ok - bad).mean():+.4f}, win '
          f'{(ok > bad).mean():.4f})')
  report['correct_vs_shuffled_context'] = ctx

  # descriptive perturbation probe (NOT an acceptance criterion)
  pr = np.random.default_rng(args.seed + 21)
  direction = pr.standard_normal((n, A))
  direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
  m0 = models[seed_runs[0][0]]
  curve = [float(np.asarray(score_fn(
      m0.params, s, g_cmd_full,
      np.clip(a_real + d * direction, -1.0, 1.0))).mean())
      for d in PERTURB_DELTAS]
  report['perturbation_probe_descriptive'] = {
      'deltas': list(PERTURB_DELTAS), 'mean_score': curve,
      'note': 'DESCRIPTIVE ONLY -- monotonicity is not an acceptance criterion'}
  print('   perturbation (descriptive): ' +
        '  '.join(f'{d:.2f}:{c:.4f}' for d, c in zip(PERTURB_DELTAS, curve)))

  # ======================================================================== #
  # 4. Diagnostic-only multi-seed averaged score
  # ======================================================================== #
  print()
  print('4. DIAGNOSTIC-ONLY averaged score  D_avg = mean_k sigmoid(logit_k)')
  stacked_crl = np.stack([scores_crl[l] for l, _ in seed_runs])
  stacked_real = np.stack([scores_real[l] for l, _ in seed_runs])
  avg_crl, avg_real = stacked_crl.mean(0), stacked_real.mean(0)
  per_seed_sd = float(stacked_crl.std(axis=0).mean())
  # leave-one-out: how much does dropping a seed move the average?
  loo = []
  k = len(seed_runs)
  for i in range(k):
    other = np.delete(stacked_crl, i, axis=0).mean(0)
    loo.append(float(np.abs(other - avg_crl).mean()))
  single_pair_mad = float(np.mean([v['mean_abs_difference']
                                   for v in pairs.values()]))
  lg = np.concatenate([np.log(avg_real / (1 - avg_real + 1e-12) + 1e-12),
                       np.log(avg_crl / (1 - avg_crl + 1e-12) + 1e-12)])
  y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
  report['averaged_score_diagnostic'] = {
      'roc_auc': _roc_auc(lg, y),
      'cross_seed_sd_of_score_mean': per_seed_sd,
      'single_model_pairwise_mad': single_pair_mad,
      'leave_one_out_mad_of_average': loo,
      'status': 'DIAGNOSTIC ONLY -- multi-seed averaging is NOT the canonical '
                'algorithm; the canonical model is a single seed of '
                'D(s, g_cmd, a)'}
  print(f'   averaged-score ROC-AUC              : '
        f'{report["averaged_score_diagnostic"]["roc_auc"]:.4f}')
  print(f'   cross-seed sd of per-example score  : {per_seed_sd:.4f}')
  print(f'   single-model pairwise mean|diff|    : {single_pair_mad:.4f}')
  print(f'   leave-one-out shift of the average  : '
        f'{["%.4f" % v for v in loo]}')

  # ======================================================================== #
  # 5. Public API contract check
  # ======================================================================== #
  print()
  print('5. PUBLIC API CONTRACT (propensity/agreement.py)')
  scorer = agr.AgreementScorer(m0)
  w = np.asarray(scorer.score(s[:64], g_cmd_full[:64], a_crl[:64]))
  in01 = bool((w >= 0).all() and (w <= 1).all())
  # arbitrary leading dims + vmap, no reshaping required
  s3 = jnp.asarray(s[:32].reshape(4, 8, S))
  g3 = jnp.asarray(g_cmd_full[:32].reshape(4, 8, spec.g_cmd_dim_full))
  a3 = jnp.asarray(a_crl[:32].reshape(4, 8, A))
  w3 = np.asarray(score_fn(m0.params, s3, g3, a3))
  vm = jax.vmap(lambda ss, gg, aa: score_fn(m0.params, ss, gg, aa))
  wv = np.asarray(vm(s3, g3, a3))
  presliced = np.asarray(score_fn(
      m0.params, s[:64], g_cmd_full[:64][:, list(spec.g_cmd_indices)],
      a_crl[:64]))
  api = {'score_in_unit_interval': in01,
         'batched_shape_ok': list(w.shape) == [64],
         'leading_dims_ok': list(w3.shape) == [4, 8],
         'vmap_matches_batched': bool(np.allclose(w3, wv, atol=1e-6)),
         'full_and_presliced_g_cmd_agree':
             bool(np.allclose(w, presliced, atol=1e-6)),
         'jit_ok': True}
  report['api_contract'] = api
  for k_, v in api.items():
    print(f'   {k_:<34} {v}')

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
  print()
  print('D IS: a learned soft behavior-vs-target agreement surrogate.')
  print('D IS NOT: P(A=a|s,g_cmd), a calibrated propensity, a density, a '
        'density ratio,\n          or a causal branch probability.')
  print(f'report -> {os.path.abspath(args.out)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
